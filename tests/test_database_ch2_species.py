"""Ideal-thermo records for the species that had critical constants only
(issue #442, Ch. 2).

Antoine sets are tested for their normal boiling points in
tests/test_antoine_database.py. Here: each ideal-gas Cp at 298.15 K against
a separately tabulated value (CRC Handbook, JANAF, or the Cp(298 K) column
of Poling-Prausnitz-O'Connell App. A -- not the polynomial the cubic was
fitted to), each latent heat at Tb against the reference equation of state
(evaluated with CoolProp 8.0), and the records' internal consistency.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import IdealThermo
from difflow.database import (
    SOURCE_CITATIONS,
    _CH2_IDEAL,
    get_critical_props,
    get_species_data,
    list_species,
)

jax.config.update("jax_enable_x64", True)

NEW = sorted(_CH2_IDEAL)

#: Ideal-gas Cp at 298.15 K (J/mol/K) from an independent tabulation.
CP298 = {
    "carbon_monoxide": 29.1,       # CRC
    "chloromethane": 40.8,         # CRC
    "dichloromethane": 51.0,       # CRC
    "heavy_water": 34.255,         # JANAF
    "m_xylene": 125.71,            # PPO App. A Cp(298) column
    "o_xylene": 132.31,            # PPO
    "p_xylene": 126.02,            # PPO
    "methyl_acetate": 86.0,        # CRC
    "methyl_ethyl_ketone": 101.7,  # CRC (PPO column: 103.26)
    "n_nonane": 210.41,            # PPO
    "n_decane": 233.05,            # PPO
    "neopentane": 120.8,           # PPO
    "sulfur_dioxide": 39.9,        # CRC
    "tetrahydrofuran": 76.3,       # CRC
    "formaldehyde": 35.4,          # CRC
    "acetaldehyde": 55.3,          # CRC
}

#: Reference-EOS normal boiling point (K) and latent heat there (J/mol).
HVAP_REF = {
    "carbon_monoxide": (81.6382, 6013),
    "chloromethane": (249.1727, 21613),
    "heavy_water": (374.5488, 41496),   # also the stored value's source
    "m_xylene": (412.2139, 36112),
    "o_xylene": (417.5210, 36384),
    "p_xylene": (411.4704, 35709),
    "n_nonane": (423.9130, 37012),
    "n_decane": (447.2702, 39326),
    "neopentane": (282.6550, 22776),
    "sulfur_dioxide": (263.1370, 24956),
}

#: Species that are not liquid at 298.15 K and 1 atm: no liquid Cp stored.
NO_LIQUID = {"carbon_monoxide", "chloromethane", "sulfur_dioxide",
             "neopentane", "formaldehyde"}


def test_the_gap_is_closed_except_helium_and_acetylene():
    missing = set(list_species("critical")) - set(list_species("ideal"))
    assert missing == {"helium", "acetylene"}


def test_every_new_species_is_covered_here():
    assert set(CP298) == set(NEW)


@pytest.mark.parametrize("name", NEW)
def test_record(name):
    d = get_species_data(name)
    crit = get_critical_props(name)
    assert d.MW == pytest.approx(crit.MW)
    assert d.Hvap_coeffs[2] == pytest.approx(crit.Tc)
    assert "chemicals package" in SOURCE_CITATIONS[name]
    if name in NO_LIQUID:
        assert d.Cp_coeffs == d.Cp_vapor_coeffs
    else:
        assert d.Cp_coeffs != d.Cp_vapor_coeffs
        assert d.Cp_coeffs[0] > d.Cp_vapor_coeffs[0]  # liquid Cp > gas Cp


@pytest.mark.parametrize("name", NEW)
def test_ideal_gas_cp_at_298(name):
    th = IdealThermo({name: get_species_data(name)})
    assert float(th.Cp_ig(name, 298.15)) == pytest.approx(CP298[name], rel=0.02)


@pytest.mark.parametrize("name", NEW)
def test_ideal_gas_cp_rises_with_T(name):
    th = IdealThermo({name: get_species_data(name)})
    T = jnp.linspace(300.0, 800.0, 26)
    cp = jax.vmap(lambda t: th.Cp_ig(name, t))(T)
    assert jnp.all(jnp.diff(cp) > 0)


@pytest.mark.parametrize("name", sorted(HVAP_REF))
def test_latent_heat_at_tb_against_reference_eos(name):
    Tb, H = HVAP_REF[name]
    th = IdealThermo({name: get_species_data(name)})
    assert float(th.Hvap(name, Tb)) == pytest.approx(H, rel=0.015)


@pytest.mark.parametrize("name", NEW)
def test_watson_passes_through_its_anchor(name):
    from difflow.database import _IDEAL_THERMO_DATA
    raw = _IDEAL_THERMO_DATA[name]
    th = IdealThermo({name: get_species_data(name)})
    assert float(th.Hvap(name, raw["Hvap_T"])) == pytest.approx(raw["Hvap"][0], rel=1e-12)


@pytest.mark.parametrize("name", NEW)
def test_enthalpy_path_and_gradients(name):
    th = IdealThermo({name: get_species_data(name)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Hv = lambda T: th.H_pure(name, T, "vapor")
        check_grads(Hv, (400.0,), order=1, modes=["rev"])
        # vapor above liquid by the latent heat at the anchor
        Tv = th.vaporization_T(name)
        assert float(th.latent_heat(name, Tv)) == pytest.approx(
            float(th.Hvap(name, Tv)), rel=1e-12)
        assert float(jax.grad(Hv)(400.0)) == pytest.approx(
            float(th.Cp_ig(name, 400.0)), rel=1e-10)
