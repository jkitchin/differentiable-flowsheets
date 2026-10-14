"""Tests for the REE separation plugin: smoke, API contracts and validation."""

import jax
import numpy as np
import pytest

pytest.importorskip("difflow_ree")

from difflow.streams import get_flows, make_stream


def _ext(name="TestExtractant", **kw):
    from difflow_ree import create_custom_extractant

    args = dict(name=name, full_name="Test", formula="C8H17O4P", molecular_weight=220.0,
                ph_coefficients={"La": {"a": -8.0, "b": 2.2, "c": 0.01},
                                 "Nd": {"a": -7.5, "b": 2.4, "c": 0.01}},
                temperature_coefficients={"La": -1500.0, "Nd": -1700.0})
    args.update(kw)
    return create_custom_extractant(**args)


def test_database_smoke():
    from difflow_ree import get_element, get_extractant, list_extractants, list_ree_elements

    assert "Nd" in list_ree_elements()
    assert get_element("Nd").atomic_number == 60
    assert "D2EHPA" in list_extractants()
    assert "Nd" in get_extractant("D2EHPA").ph_coefficients


def test_distribution_smoke_and_grad():
    from difflow_ree import REEDistribution

    dist = REEDistribution(extractant="D2EHPA", elements=("La", "Nd", "Dy"))
    D = dist.get_D_all(pH=1.0, T=298.15)
    assert all(np.isfinite(float(v)) for v in D.values())
    assert np.isfinite(float(dist.get_separation_factor("Nd", "La", pH=1.0)))
    assert np.isfinite(float(jax.grad(lambda pH: dist.get_D("Nd", pH))(1.0)))


def test_extractor_conserves_mass():
    from difflow_ree import REEExtractor, REEExtractorParams

    extractor = REEExtractor(REEExtractorParams(n_stages=1, extractant="D2EHPA",
                                                elements=("Dy", "Nd"), diluent="n-Heptane", pH=1.6))
    feed = make_stream({"H2O": 10.0, "Nd": 0.2, "Dy": 0.14, "Fe": 0.553}, T=298.15, P=101325.0)
    solvent = make_stream({"D2EHPA": 10.0, "n-Heptane": 20.0, "Nd": 0.0, "Dy": 0.0, "Fe": 0.0},
                          T=298.15, P=101325.0)
    raff, ext, _ = extractor(feed, solvent)
    fi, si, ro, eo = (get_flows(s) for s in (feed, solvent, raff, ext))
    for sp in set(fi) | set(si):
        total_in = float(fi.get(sp, 0.0)) + float(si.get(sp, 0.0))
        assert abs(total_in - float(ro.get(sp, 0.0)) - float(eo.get(sp, 0.0))) < 1e-10, sp


def test_oxalate_precipitator_smoke():
    from difflow_ree import OxalatePrecipitator, PrecipitatorParams

    precip = OxalatePrecipitator(PrecipitatorParams(elements=("Nd", "Dy"), precipitant_excess=1.5))
    feed = make_stream({"H2O": 10.0, "Nd": 0.1, "Dy": 0.05}, T=298.15, P=101325.0)
    oxalate = make_stream({"H2O": 5.0, "C2O4": 0.3}, T=298.15, P=101325.0)
    _, _, info = precip(feed, oxalate)
    assert np.isfinite(float(info["total_precipitated"]))


def test_cerium_oxidizer_registers_its_solid_outlet():
    """Regression (#371): the CeO2 solid is an outlet a flowsheet can place."""
    from difflow.catalog import catalog
    from difflow.flowsheet import Unit, _parse_outlets
    from difflow_ree import CeriumOxidizer, CeriumOxidizerParams

    assert catalog()["CeriumOxidizer"].ports.n_outlets == 2
    oxidizer = CeriumOxidizer(CeriumOxidizerParams(elements=("La", "Ce", "Nd")))
    feed = make_stream({"H2O": 10.0, "La": 0.1, "Ce": 0.2, "Nd": 0.05}, T=298.15, P=101325.0)
    outlets = _parse_outlets(oxidizer(feed), Unit("ce_ox", oxidizer, ["feed"], ["filtrate", "ceo2"]))
    assert set(outlets) == {"filtrate", "ceo2"}


def test_separation_train_balance_reads_stream_products():
    """Regression (#371): stream products used to count as zero in the balance."""
    from difflow_ree.flowsheets.full_train import FullSeparationTrain, SeparationTrainParams

    elements = ("La", "Ce", "Pr", "Nd")
    train = FullSeparationTrain(SeparationTrainParams(elements=elements))
    result = train(make_stream({"H2O": 10.0, "La": 0.3, "Ce": 0.5, "Pr": 0.05, "Nd": 0.15},
                               T=298.15, P=101325.0))
    total = sum(float(get_flows(p).get(e, 0.0)) for p in result["products"].values() for e in elements)
    assert result["mass_balance"]["total_out"] == pytest.approx(total)
    assert total > 0.0


def test_extract_strip_circuit_smoke():
    from difflow_ree import ExtractStripCircuit, ExtractStripParams

    circuit = ExtractStripCircuit(ExtractStripParams(extractant="D2EHPA", elements=("La", "Nd", "Dy"),
                                                     n_extraction_stages=3, n_stripping_stages=2))
    results = circuit(make_stream({"H2O": 10.0, "La": 0.01, "Nd": 0.02, "Dy": 0.01},
                                  T=298.15, P=101325.0))
    assert "product" in results and np.isfinite(float(results["recovery"]))


def test_split_shell_cascade_smoke():
    from difflow_ree import SplitShellCascade, SplitShellParams
    from difflow_ree.flowsheets.split_shell import optimize_split_points

    cascade = SplitShellCascade(SplitShellParams(extractant="D2EHPA", elements=("La", "Nd", "Dy"),
                                                 n_stages=12, split_points=(4, 8)))
    feed = make_stream({"H2O": 10.0, "La": 0.03, "Nd": 0.03, "Dy": 0.03}, T=298.15, P=101325.0)
    solvent = make_stream({"kerosene": 10.0, "D2EHPA": 5.0, "La": 0.0, "Nd": 0.0, "Dy": 0.0},
                          T=298.15, P=101325.0)
    results = cascade(feed, solvent)
    assert {"product_1", "product_2", "raffinate"} <= set(results["products"])
    splits = optimize_split_points(elements=("La", "Nd", "Dy"), extractant="D2EHPA",
                                   n_stages=30, n_products=3, pH=0.5)
    assert len(splits) == 2


def test_economics_smoke():
    from difflow_ree import REEPricing, estimate_capex, minimum_selling_price

    assert REEPricing().get_price("Nd", purity="99%", form="oxide") > 0
    assert estimate_capex(annual_ree_tonnes=1000, n_stages_extraction=10,
                          n_stages_scrubbing=5, n_stages_stripping=5)["total"] > 0
    assert np.isfinite(minimum_selling_price(opex=1e6, capex=5e6, annual_production_kg=1e5,
                                             target_roi=0.15))


def test_custom_extractant_register_update_remove():
    from difflow_ree import REEDistribution, get_extractant_database

    db = get_extractant_database()
    db.add_extractant("MyExtractant", _ext("MyExtractant"))
    try:
        with pytest.raises(ValueError, match="already exists"):
            db.add_extractant("MyExtractant", _ext("MyExtractant"))
        db.update_extractant("MyExtractant", _ext("MyExtractant", molecular_weight=250.0))
        assert db.get("MyExtractant").molecular_weight == 250.0
        D = REEDistribution(extractant="MyExtractant", elements=("La", "Nd")).get_D_all(pH=3.0, T=298.15)
        assert np.isfinite(float(D["Nd"]))
    finally:
        db.remove_extractant("MyExtractant")
    assert "MyExtractant" not in db.list_extractants()


def test_custom_extractant_validation():
    with pytest.raises(ValueError, match="ph_coefficients is required"):
        _ext(ph_coefficients={}, temperature_coefficients={"La": -1500.0})
    with pytest.raises(ValueError, match="Element mismatch"):
        _ext(temperature_coefficients={"La": -1500.0})
    with pytest.raises(ValueError, match="missing required keys"):
        _ext(ph_coefficients={"La": {"a": -8.0, "b": 2.2}}, temperature_coefficients={"La": -1500.0})
