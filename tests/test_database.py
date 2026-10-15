"""Tests for the thermodynamic property database API."""

import jax.numpy as jnp
import pytest

from difflow import IdealThermo, PengRobinson
from difflow.database import (
    get_alkanes,
    get_btex,
    get_common_solvents,
    get_critical_props,
    get_species_data,
    get_species_info,
    list_species,
    resolve_alias,
    watson_coeffs,
)


def test_lookup_is_case_insensitive():
    assert get_critical_props("Methane").Tc == get_critical_props("METHANE").Tc


@pytest.mark.parametrize("fn", [get_critical_props, get_species_data, get_species_info])
def test_unknown_species_raises_keyerror(fn):
    with pytest.raises(KeyError):
        fn("unobtainium")


def test_aliases():
    assert resolve_alias("CO2") == "carbon_dioxide"
    assert get_critical_props("butane").name == "n_butane"
    assert get_species_data("MeOH").name == "methanol"


def test_list_species_is_sorted_and_has_common_species():
    species = list_species()
    assert species == sorted(species)
    assert {"water", "methane", "benzene", "ethanol"} <= set(species)


def test_species_info():
    info = get_species_info("methane")
    assert "critical" in info and "ideal_thermo" in info
    info = get_species_info("helium")
    assert "critical" in info and "ideal_thermo" not in info


def test_convenience_groups_build_models():
    assert len(get_alkanes(4)) == 4
    assert "benzene" in get_btex()
    eos = PengRobinson(get_alkanes(4))
    Z = eos.solve_Z(jnp.array(250.0), jnp.array(1e6), jnp.ones(4) / 4, phase="vapor")
    assert jnp.isfinite(Z)
    thermo = IdealThermo(get_common_solvents())
    assert jnp.isfinite(thermo.Cp("water", 300.0))


def test_watson_coeffs_refuses_a_reference_past_critical():
    with pytest.raises(ValueError):
        watson_coeffs(10000.0, 0.38, 300.0, 316.0)


def test_hydrotreating_species_are_in_both_tables():
    for name in ("hydrogen", "hydrogen_sulfide", "thiophene"):
        assert get_species_data(name).name == name
        assert get_critical_props(name).name == name
