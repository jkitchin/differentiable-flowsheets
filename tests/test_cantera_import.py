"""Tests for the Cantera data import module."""

from pathlib import Path

import jax.numpy as jnp
import pytest

from difflow import IdealThermo, PengRobinson
from difflow.cantera_import import (
    import_critical_props,
    import_reactions,
    import_species_data,
    list_available_reactions,
    list_available_species,
    load_mechanism,
    nasa7_to_cp_coeffs,
    nasa7_to_enthalpy_coeffs,
)

TEST_MECHANISM = Path(__file__).parent / 'data' / 'test_mechanism.yaml'


def test_nasa7_conversions_smoke():
    hi = [4.63659493, 0.00274131991, -9.95828531e-07, 1.60373011e-10, -9.16103468e-15, -49024.9341, -1.9348955]
    lo = [2.35677352, 0.00898459677, -7.12356269e-06, 2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222]
    assert len(nasa7_to_cp_coeffs(lo, hi)) == 4
    assert jnp.isfinite(nasa7_to_enthalpy_coeffs(lo, hi, use_high_T=False))


def test_species_import_with_ideal_thermo():
    species = import_species_data(TEST_MECHANISM, species_names=['CH4', 'O2'])
    assert set(species) == {'CH4', 'O2'}
    Cp = IdealThermo(species).Cp_mix({'CH4': jnp.array(1.0), 'O2': jnp.array(2.0)}, jnp.array(400.0))
    assert jnp.isfinite(Cp)


def test_missing_species_warning():
    with pytest.warns(UserWarning, match="not found"):
        import_species_data(TEST_MECHANISM, species_names=['CH4', 'NONEXISTENT'])


def test_critical_props_with_eos():
    props = import_critical_props(TEST_MECHANISM, ['CH4', 'CO2'])
    z = PengRobinson(props).solve_Z(T=jnp.array(300.0), P=jnp.array(1e6),
                                    y=jnp.array([0.5, 0.5]), phase='vapor')
    assert jnp.isfinite(z)


def test_reaction_structure():
    rxn = import_reactions(TEST_MECHANISM)[0]
    assert {'equation', 'reactants', 'products', 'rate_params'} <= set(rxn)
    assert {'A', 'Ea', 'n'} <= set(rxn['rate_params'])


def test_load_mechanism_and_listing():
    species, reactions = load_mechanism(TEST_MECHANISM, species_names=['CH4', 'O2', 'CO2'])
    assert len(species) == 3 and len(reactions) >= 1
    assert 'CH4' in list_available_species(TEST_MECHANISM)
    assert list_available_reactions(TEST_MECHANISM)
