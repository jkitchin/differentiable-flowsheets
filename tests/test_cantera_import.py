"""Tests for Cantera data import module."""

import pytest
import jax.numpy as jnp
from pathlib import Path

from difflow.cantera_import import (
    import_species_data,
    import_critical_props,
    import_reactions,
    load_mechanism,
    list_available_species,
    list_available_reactions,
    nasa7_to_cp_coeffs,
    nasa7_to_enthalpy_coeffs,
)
from difflow import IdealThermo, PengRobinson


# Path to test data
TEST_DATA_DIR = Path(__file__).parent / 'data'
TEST_MECHANISM = TEST_DATA_DIR / 'test_mechanism.yaml'


class TestNASA7Conversion:
    """Tests for NASA7 polynomial conversion."""

    def test_nasa7_to_cp_coeffs(self):
        """Test conversion of NASA7 to Cp coefficients."""
        # CO2 high-T coefficients from test file
        coeffs_high = [4.63659493, 0.00274131991, -9.95828531e-07,
                       1.60373011e-10, -9.16103468e-15, -49024.9341, -1.9348955]
        coeffs_low = [2.35677352, 0.00898459677, -7.12356269e-06,
                      2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222]

        cp_coeffs = nasa7_to_cp_coeffs(coeffs_low, coeffs_high)

        # Should return 4 coefficients
        assert len(cp_coeffs) == 4

        # Coefficients should be reasonable (positive first term for most gases)
        a, b, c, d = cp_coeffs
        assert a > 0  # Constant term should be positive

    def test_nasa7_enthalpy(self):
        """Test enthalpy of formation extraction."""
        # CO2 coefficients
        coeffs_high = [4.63659493, 0.00274131991, -9.95828531e-07,
                       1.60373011e-10, -9.16103468e-15, -49024.9341, -1.9348955]
        coeffs_low = [2.35677352, 0.00898459677, -7.12356269e-06,
                      2.45919022e-09, -1.43699548e-13, -48371.9697, 9.90105222]

        Hf = nasa7_to_enthalpy_coeffs(coeffs_low, coeffs_high, use_high_T=False)

        # CO2 formation enthalpy should be negative (exothermic formation)
        # Roughly -393 kJ/mol
        assert Hf < 0
        assert -500000 < Hf < -300000  # Reasonable range in J/mol


class TestSpeciesImport:
    """Tests for species data import."""

    def test_import_all_species(self):
        """Test importing all species from mechanism."""
        species_data = import_species_data(TEST_MECHANISM)

        assert len(species_data) >= 5
        assert 'CH4' in species_data
        assert 'O2' in species_data
        assert 'CO2' in species_data

    def test_import_specific_species(self):
        """Test importing specific species."""
        species_data = import_species_data(
            TEST_MECHANISM,
            species_names=['CH4', 'O2', 'CO2']
        )

        assert len(species_data) == 3
        assert 'CH4' in species_data
        assert 'N2' not in species_data

    def test_species_data_structure(self):
        """Test that imported SpeciesData has correct structure."""
        species_data = import_species_data(TEST_MECHANISM, ['CH4'])

        ch4 = species_data['CH4']
        assert ch4.name == 'CH4'
        assert ch4.MW == pytest.approx(16.04, rel=0.01)  # C + 4*H
        assert len(ch4.Cp_coeffs) == 4
        assert len(ch4.Hvap_coeffs) == 3
        assert len(ch4.antoine_coeffs) == 3

    def test_use_with_ideal_thermo(self):
        """Test that imported data works with IdealThermo."""
        species_data = import_species_data(
            TEST_MECHANISM,
            species_names=['CH4', 'O2', 'CO2', 'H2O']
        )

        thermo = IdealThermo(species_data)

        # Should be able to compute Cp
        Cp = thermo.Cp_mix(
            {'CH4': jnp.array(1.0), 'O2': jnp.array(2.0),
             'CO2': jnp.array(0.0), 'H2O': jnp.array(0.0)},
            jnp.array(400.0)
        )

        assert jnp.isfinite(Cp)
        assert float(Cp) > 0

    def test_missing_species_warning(self):
        """Test that missing species generates warning."""
        with pytest.warns(UserWarning, match="not found"):
            import_species_data(
                TEST_MECHANISM,
                species_names=['CH4', 'NONEXISTENT']
            )


class TestCriticalPropsImport:
    """Tests for critical properties import."""

    def test_import_critical_props(self):
        """Test importing critical properties."""
        props = import_critical_props(TEST_MECHANISM)

        assert len(props) >= 4
        assert 'CH4' in props
        assert 'CO2' in props

    def test_critical_props_values(self):
        """Test that critical property values are correct."""
        props = import_critical_props(TEST_MECHANISM, ['CH4', 'CO2'])

        ch4 = props['CH4']
        assert ch4.Tc == pytest.approx(190.6, rel=0.01)
        assert ch4.Pc == pytest.approx(4.6e6, rel=0.01)
        assert ch4.omega == pytest.approx(0.011, rel=0.1)

        co2 = props['CO2']
        assert co2.Tc == pytest.approx(304.2, rel=0.01)
        assert co2.Pc == pytest.approx(7.38e6, rel=0.01)
        assert co2.omega == pytest.approx(0.228, rel=0.1)

    def test_use_with_eos(self):
        """Test that imported props work with EOS."""
        props = import_critical_props(TEST_MECHANISM, ['CH4', 'CO2'])

        eos = PengRobinson(props)

        # Should be able to compute compressibility using solve_Z
        z = eos.solve_Z(
            T=jnp.array(300.0),
            P=jnp.array(1e6),
            y=jnp.array([0.5, 0.5]),
            phase='vapor'
        )

        assert jnp.isfinite(z)
        assert 0 < float(z) < 2  # Reasonable range


class TestReactionsImport:
    """Tests for reaction import."""

    def test_import_reactions(self):
        """Test importing reactions."""
        reactions = import_reactions(TEST_MECHANISM)

        assert len(reactions) >= 3

    def test_reaction_structure(self):
        """Test reaction data structure."""
        reactions = import_reactions(TEST_MECHANISM)

        rxn = reactions[0]  # CH4 + 2 O2 <=> CO2 + 2 H2O
        assert 'equation' in rxn
        assert 'reactants' in rxn
        assert 'products' in rxn
        assert 'rate_params' in rxn

        # Check stoichiometry parsing
        assert rxn['reactants'].get('CH4', 0) == pytest.approx(1.0)
        assert rxn['reactants'].get('O2', 0) == pytest.approx(2.0)
        assert rxn['products'].get('CO2', 0) == pytest.approx(1.0)
        assert rxn['products'].get('H2O', 0) == pytest.approx(2.0)

    def test_rate_params(self):
        """Test Arrhenius rate parameter extraction."""
        reactions = import_reactions(TEST_MECHANISM)

        rxn = reactions[0]
        rate = rxn['rate_params']

        assert rate['A'] == pytest.approx(1e13, rel=0.01)
        assert rate['Ea'] == pytest.approx(200000.0, rel=0.01)
        assert rate['n'] == pytest.approx(0.0, abs=0.01)


class TestLoadMechanism:
    """Tests for complete mechanism loading."""

    def test_load_mechanism(self):
        """Test loading complete mechanism."""
        species, reactions = load_mechanism(TEST_MECHANISM)

        assert len(species) >= 5
        assert len(reactions) >= 3

    def test_load_specific_species(self):
        """Test loading mechanism with specific species."""
        species, reactions = load_mechanism(
            TEST_MECHANISM,
            species_names=['CH4', 'O2', 'CO2']
        )

        assert len(species) == 3
        # Reactions should still be all loaded
        assert len(reactions) >= 3


class TestListFunctions:
    """Tests for listing functions."""

    def test_list_species(self):
        """Test listing available species."""
        names = list_available_species(TEST_MECHANISM)

        assert 'CH4' in names
        assert 'O2' in names
        assert 'CO2' in names
        assert len(names) >= 5

    def test_list_reactions(self):
        """Test listing available reactions."""
        equations = list_available_reactions(TEST_MECHANISM)

        assert len(equations) >= 3
        # First reaction should be methane combustion
        assert any('CH4' in eq for eq in equations)


class TestIntegration:
    """Integration tests combining imports with difflow operations."""

    def test_full_workflow(self):
        """Test complete workflow from import to calculation."""
        # Import data
        species_data = import_species_data(
            TEST_MECHANISM,
            ['CH4', 'O2', 'CO2', 'H2O', 'N2']
        )
        critical_props = import_critical_props(TEST_MECHANISM, ['CH4', 'O2'])

        # Create thermo object
        thermo = IdealThermo(species_data)

        # Create EOS
        eos = PengRobinson(critical_props)

        # Calculate mixture Cp
        composition = {
            'CH4': jnp.array(1.0),
            'O2': jnp.array(2.0),
            'CO2': jnp.array(0.0),
            'H2O': jnp.array(0.0),
            'N2': jnp.array(7.52),
        }
        Cp = thermo.Cp_mix(composition, jnp.array(500.0))
        assert jnp.isfinite(Cp)

        # Calculate EOS properties
        z = eos.solve_Z(
            T=jnp.array(300.0),
            P=jnp.array(5e6),
            y=jnp.array([0.9, 0.1]),
            phase='vapor'
        )
        assert jnp.isfinite(z)

    def test_reaction_rate_calculation(self):
        """Test using imported kinetics for rate calculation."""
        reactions = import_reactions(TEST_MECHANISM)

        # Get CH4 combustion reaction
        ch4_rxn = reactions[0]
        A = ch4_rxn['rate_params']['A']
        Ea = ch4_rxn['rate_params']['Ea']
        n = ch4_rxn['rate_params']['n']

        # Calculate rate constant at 1000 K
        R = 8.314
        T = 1000.0
        k = A * (T ** n) * jnp.exp(-Ea / (R * T))

        assert jnp.isfinite(k)
        assert float(k) > 0
