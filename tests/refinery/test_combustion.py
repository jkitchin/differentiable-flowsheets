"""Tests for the shared coke-combustion module (#470): difflow_refinery.combustion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow_refinery import combustion as cb
from difflow_refinery import thermochemistry as tc
from difflow_refinery.fcc import regenerator as rg
from difflow_refinery.fcc import species as sp

jax.config.update("jax_enable_x64", True)

EL = {"C": jnp.asarray(0.9), "H": jnp.asarray(0.07), "S": jnp.asarray(0.02),
      "N": jnp.asarray(0.01)}

# float.hex of fcc.regenerator.combustion BEFORE the move into
# difflow_refinery.combustion (#470), on the inputs of test_fcc_regenerator_bit_identical.
PIN = {
    ("arthur", "flue_o2"): {
        "nitrogen": "0x1.091a946412132p+8", "oxygen": "0x1.ea4b369e15b38p+2",
        "carbon_dioxide": "0x1.c3598da5cd774p+3", "carbon_monoxide": "0x1.e69cf0cc8410dp+5",
        "water": "0x1.15c71c71c71c7p+5", "sulfur_dioxide": "0x1.3f66affd71dddp-1",
        "o2_air": "0x1.18a7b5ce28b93p+6"},
    ("arthur", "air_rate"): {
        "nitrogen": "0x1.121b053ac8a54p+10", "oxygen": "0x1.c7f50c965b642p+7",
        "carbon_dioxide": "0x1.c3598da5cd774p+3", "carbon_monoxide": "0x1.e69cf0cc8410dp+5",
        "water": "0x1.15c71c71c71c7p+5", "sulfur_dioxide": "0x1.3f66affd71dddp-1",
        "o2_air": "0x1.227b46e43f899p+8"},
    ("fixed", "flue_o2"): {
        "nitrogen": "0x1.62172df64f726p+8", "oxygen": "0x1.2f42bbae8bbf2p+3",
        "carbon_dioxide": "0x1.cd1da33d347c7p+5", "carbon_monoxide": "0x1.14ab61f185e46p+4",
        "water": "0x1.15c71c71c71c7p+5", "sulfur_dioxide": "0x1.3f66affd71dddp-1",
        "o2_air": "0x1.76fd29cf091d8p+6"},
    ("fixed", "air_rate"): {
        "nitrogen": "0x1.121b053ac8a54p+10", "oxygen": "0x1.9c6c249be3405p+7",
        "carbon_dioxide": "0x1.cd1da33d347c7p+5", "carbon_monoxide": "0x1.14ab61f185e46p+4",
        "water": "0x1.15c71c71c71c7p+5", "sulfur_dioxide": "0x1.3f66affd71dddp-1",
        "o2_air": "0x1.227b46e43f899p+8"},
}


@pytest.mark.parametrize("modes", sorted(PIN))
def test_fcc_regenerator_bit_identical(modes):
    """The FCC regenerator gives the same bits after the move (#470 acceptance)."""
    p = {"arthur_factor": 1.0, "co_co2": 0.3, "flue_o2": 0.02, "air_rate": 40.0}
    o = rg.combustion(p, EL, jnp.asarray(980.0), *modes)
    got = {**{k: float(v).hex() for k, v in o["flue"].items()},
           "o2_air": float(o["air"]["oxygen"]).hex()}
    assert got == PIN[modes]


def test_flue_enthalpy_bit_identical():
    o = cb.burn(EL, 0.3, flue_o2=0.02)
    assert float(sp.flue_enthalpy(o["flue"], 980.0)).hex() == "-0x1.57009e10b6608p+24"


def test_fcc_uses_the_shared_module():
    assert rg.arthur_co_co2 is cb.arthur_co_co2
    assert sp.CP_IG is cb.CP_IG and sp.HF_298 is cb.HF_298
    assert sp.ATOMIC_WEIGHT is cb.ATOMIC_WEIGHT
    assert sp.h_ideal_gas is cb.h_ideal_gas
    assert (sp.AIR_O2, sp.T_REF) == (cb.AIR_O2, cb.T_REF)


# --- cited constants --------------------------------------------------------

def test_cited_constants():
    assert cb.ATOMIC_WEIGHT == {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}
    assert cb.AIR_O2 == 0.2095                       # US Std Atmosphere 1976: 20.9476 vol%
    assert cb.ARTHUR_A == pytest.approx(2511.886, abs=1e-3)
    assert cb.ARTHUR_E_CAL * 4.184 == pytest.approx(51882, abs=1)   # 51.88 kJ/mol
    assert cb.R_CAL == pytest.approx(8.314462618 / 4.184, rel=1e-6)
    for n in cb.GASES:                               # the one table, not a copy
        assert cb.CP_IG[n] == tc.species(n).cp and cb.HF_298[n] == tc.Hf(n)


def test_arthur_ratio_secondary_value():
    """CO/CO2 ~ 7.5 at 1073 K, as quoted with the constants (char-combustion reviews)."""
    assert float(cb.arthur_co_co2(1073.0)) == pytest.approx(7.49, abs=0.01)
    T = jnp.linspace(*cb.ARTHUR_RANGE, 20)
    assert np.all(np.diff(np.asarray(cb.arthur_co_co2(T))) > 0)


# --- closed forms: heats of combustion (CODATA) ----------------------------

@pytest.mark.parametrize("element,Q_per_mol,mw", [
    ("C", 393.51e3, 12.011),        # CO2(g) Hf, CODATA
    ("H", 241.826e3 / 2, 1.008),    # 1/2 H2O(g) Hf, CODATA
    ("S", 296.81e3, 32.06),         # SO2(g) Hf, CODATA
])
def test_heat_of_combustion_codata(element, Q_per_mol, mw):
    el = {k: 0.0 for k in "CHSN"}
    el[element] = mw * 1e-3        # one mole/s
    assert float(cb.heat_of_combustion(el, 0.0)) == pytest.approx(Q_per_mol, rel=1e-9)


def test_partial_burn_heat_is_co_hf():
    """All carbon to CO (ratio -> inf) releases -Hf(CO) = 110.53 kJ/mol (CODATA)."""
    el = {"C": 12.011e-3, "H": 0.0, "S": 0.0, "N": 0.0}
    q = float(cb.heat_of_combustion(el, 1e12))
    assert q == pytest.approx(110.53e3, rel=1e-6)


# --- balances ---------------------------------------------------------------

def _element_out(o, e):
    return float(sp.element_flow(o["flue"], e))


def _element_in(o, e):
    return float(sp.element_flow(o["air"], e))


@pytest.mark.parametrize("r", [0.0, 0.3, 4.0])
@pytest.mark.parametrize("air", [dict(flue_o2=0.02), dict(air_rate=40.0)])
def test_element_balances_close(r, air):
    o = cb.burn(EL, r, **air)
    for e in "CHSN":
        assert _element_out(o, e) == pytest.approx(float(EL[e]) + _element_in(o, e), rel=1e-12)
    assert _element_out(o, "O") == pytest.approx(_element_in(o, "O"), rel=1e-12)
    assert float(o["flue"]["carbon_monoxide"] / o["flue"]["carbon_dioxide"]) == pytest.approx(
        r, abs=1e-12)


def test_flue_o2_spec_and_air_round_trip():
    o = cb.burn(EL, 0.3, flue_o2=0.02)
    y = o["flue"]["oxygen"] / sum(o["flue"].values())
    assert float(y) == pytest.approx(0.02, rel=1e-12)
    air_kg = float(sp.mass_of(o["air"]))
    assert air_kg == pytest.approx(float(o["air"]["oxygen"]) / cb.AIR_O2 * cb.MW_AIR / 1e3,
                                   rel=1e-12)
    o2 = cb.burn(EL, 0.3, air_rate=air_kg)
    for k in o["flue"]:
        assert float(o2["flue"][k]) == pytest.approx(float(o["flue"][k]), rel=1e-10)


def test_one_air_spec_required():
    with pytest.raises(ValueError):
        cb.burn(EL, 0.0)
    with pytest.raises(ValueError):
        cb.burn(EL, 0.0, flue_o2=0.02, air_rate=1.0)


def test_energy_balance_closes():
    """Adiabatic flame: H(air at 298) + 0 (coke) = H(flue at T) solves to a T
    at which heat_of_combustion equals the flue's sensible heat."""
    o = cb.burn(EL, 0.0, flue_o2=0.02)
    q = float(cb.heat_of_combustion(EL, 0.0))
    sens = float(cb.gas_enthalpy(o["flue"], 1200.0) - cb.gas_enthalpy(o["flue"], 298.15))
    h_air = float(cb.gas_enthalpy(o["air"], 298.15))
    h_flue = float(cb.gas_enthalpy(o["flue"], 1200.0))
    assert h_air - h_flue == pytest.approx(q - sens, rel=1e-10)


# --- gradients --------------------------------------------------------------

def test_check_grads():
    def f(x):
        el = {"C": x[0], "H": x[1], "S": x[2], "N": x[3]}
        o = cb.burn(el, cb.arthur_co_co2(x[4]), flue_o2=x[5])
        return jnp.stack([cb.gas_enthalpy(o["flue"], x[4]), o["air"]["oxygen"],
                          o["flue"]["carbon_monoxide"]])
    x = jnp.array([0.9, 0.07, 0.02, 0.01, 980.0, 0.02])
    check_grads(f, (x,), order=1, modes=["fwd", "rev"])
