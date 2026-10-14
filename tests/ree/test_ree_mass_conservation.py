"""Smoke tests for the REE unit operations: they run, and species balance."""

import warnings

import pytest

from difflow.streams import get_flows, make_stream
from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams
from difflow_ree.units.extraction import (
    MixerSettlerParams,
    REEExtractor,
    REEExtractorParams,
    REEMixerSettler,
)
from difflow_ree.units.precipitation import (
    CarbonatePrecipitator,
    HydroxidePrecipitator,
    OxalatePrecipitator,
    PrecipitatorParams,
)
from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _assert_balanced(inlets, outlets, atol=1e-8):
    def totals(streams):
        out = {}
        for s in streams:
            for k, v in get_flows(s).items():
                out[k] = out.get(k, 0.0) + float(v)
        return out
    t_in, t_out = totals(inlets), totals(outlets)
    for k in set(t_in) | set(t_out):
        assert t_in.get(k, 0.0) == pytest.approx(t_out.get(k, 0.0), abs=atol), k


def test_extractor_balances():
    feed = make_stream({"H2O": 10.0, "Nd": 0.2, "Dy": 0.14, "Fe": 0.553},
                       T=298.15, P=101325.0)
    solvent = make_stream({"D2EHPA": 10.0, "n-Heptane": 20.0},
                          T=298.15, P=101325.0)
    raff, ext, info = REEExtractor(REEExtractorParams(
        n_stages=3, extractant="D2EHPA", diluent="n-Heptane",
        elements=("Dy", "Nd"), pH=1.0))(feed, solvent)
    _assert_balanced([feed, solvent], [raff, ext])


def test_mixer_settler_balances():
    aq = make_stream({"H2O": 10.0, "Nd": 0.1, "Dy": 0.05, "Fe": 0.3},
                     T=298.15, P=101325.0)
    org = make_stream({"D2EHPA": 5.0, "kerosene": 15.0}, T=298.15, P=101325.0)
    aq_out, org_out, info = REEMixerSettler(MixerSettlerParams(
        extractant="D2EHPA", elements=("Nd", "Dy"), pH=1.0))(aq, org)
    _assert_balanced([aq, org], [aq_out, org_out])


@pytest.mark.parametrize("unit", [
    REEScrubber(ScrubberParams(n_stages=3, extractant="D2EHPA",
                               elements=("Nd",), pH=0.5)),
    REEStripper(StripperParams(n_stages=3, extractant="D2EHPA",
                               elements=("Nd",), pH=0.0)),
], ids=["scrubber", "stripper"])
def test_scrubber_and_stripper_balance(unit):
    org = make_stream({"D2EHPA": 5.0, "kerosene": 10.0, "Na_org": 0.7,
                       "Nd": 0.05}, 298.15, 101325.0)
    aq = make_stream({"H2O": 5.0, "HCl": 0.4, "Fe": 0.01}, 298.15, 101325.0)
    a, o, info = unit(org, aq)
    assert info["dropped_species"] == ()
    _assert_balanced([org, aq], [a, o])


@pytest.mark.parametrize("cls,reagent", [(OxalatePrecipitator, "C2O4"),
                                         (CarbonatePrecipitator, "CO3"),
                                         (HydroxidePrecipitator, "NaOH")])
def test_precipitator_smoke(cls, reagent):
    feed = make_stream({"H2O": 55.5, "Nd": 0.10, "Dy": 0.02}, 298.15, 101325.0)
    rg = make_stream({"H2O": 10.0, reagent: 1.0}, 298.15, 101325.0)
    filt, solid, info = cls(PrecipitatorParams(elements=("Nd", "Dy")))(feed, rg)
    for e in ("Nd", "Dy"):
        total = float(get_flows(filt).get(e, 0.0)) + float(get_flows(solid)[e])
        assert total == pytest.approx(float(get_flows(feed)[e]))
    assert float(get_flows(solid)["Nd"]) > 0.0


def test_missing_solubility_data_is_named():
    with pytest.raises(ValueError, match=r"No \w+ solubility product .*'Ho'"):
        OxalatePrecipitator(PrecipitatorParams(elements=("Nd", "Ho")))


def test_cerium_oxidizer_smoke():
    feed = make_stream({"H2O": 100.0, "La": 1.0, "Ce": 2.0, "Fe": 0.3},
                       298.15, 101325.0)
    filt, solid, _ = CeriumOxidizer(CeriumOxidizerParams(
        elements=("La", "Ce")))(feed)
    assert float(get_flows(filt)["Fe"]) == pytest.approx(0.3)
    assert float(get_flows(filt)["La"]) == pytest.approx(1.0)
