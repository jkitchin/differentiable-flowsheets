"""Regression tests for LLE and distillation bug fixes (issues #81, #82, #83)."""

import jax

from difflow import (
    ContactorParams,
    CascadeParams,
    DifferentialContactor,
    DistributionCoeffs,
    IdealThermo,
    LLEEquilibrium,
    MultistageCascade,
    SpeciesData,
    get_flows,
    make_stream,
)
from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams

jax.config.update("jax_enable_x64", True)


def _eq(species, K0):
    return LLEEquilibrium(
        solutes=list(species), aqueous_carrier="H2O", organic_carrier="Solvent",
        K_coeffs=DistributionCoeffs(species=tuple(species), K0=tuple(K0)),
    )


def test_issue81_K_convention_not_inverted():
    cascade = MultistageCascade(CascadeParams(
        n_stages=5, equilibrium=_eq(["solute"], [10.0]), flow_config="counter_current"))
    feed = make_stream({"H2O": 10.0, "solute": 1.0}, T=298.15, P=101325.0)
    solvent = make_stream({"Solvent": 10.0, "solute": 0.0}, T=298.15, P=101325.0)
    _, extract, _ = cascade(feed, solvent, T=298.15)
    assert float(get_flows(extract)["solute"]) > 0.9


def test_issue82_counter_current_rate_based_extracts():
    feed = make_stream({"H2O": 100.0, "A": 5.0, "B": 3.0, "Solvent": 0.0},
                       T=298.15, P=101325.0)
    solvent = make_stream({"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
                          T=298.15, P=101325.0)
    params = ContactorParams(
        length=2.0, area=0.5, equilibrium=_eq(["A", "B"], [5.0, 2.0]), n_segments=20,
        flow_config="counter_current", mass_transfer_model="rate_based", Kla=0.1,
    )
    _, _, info = DifferentialContactor(params)(feed, solvent, T=298.15)
    c = info["profiles"]["c_aq"]["A"]
    assert float(c[-1]) < float(c[0])


def test_issue83_shortcut_duty_signs():
    thermo = IdealThermo({
        "benzene": SpeciesData(name="benzene", MW=78.11, Cp_coeffs=(136.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(33900.0, 0.38, 562.0),
                               antoine_coeffs=(13.82, 2788.0, -52.36)),
        "toluene": SpeciesData(name="toluene", MW=92.14, Cp_coeffs=(157.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(38000.0, 0.38, 591.8),
                               antoine_coeffs=(13.93, 3096.0, -53.67)),
    })
    column = ShortcutColumn(ShortcutColumnParams(
        species_order=["benzene", "toluene"], light_key="benzene", heavy_key="toluene",
        x_D_LK=0.95, x_B_HK=0.95), thermo)
    feed = make_stream({"benzene": 0.5, "toluene": 0.5}, T=370.0, P=101325.0)
    _, _, info = column(feed, R=2.0)
    assert float(info["Q_condenser"]) < 0 < float(info["Q_reboiler"])
