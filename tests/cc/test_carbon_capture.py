"""Tests for carbon capture plugin (difflow_cc).

These tests verify the basic functionality of:
- Database loading (solvents, adsorbents, membranes)
- Equilibrium models (isotherms, VLE)
- Unit operations (absorber, stripper, membrane, adsorption)
- JAX compatibility (gradients, JIT)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import grad, jit

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Database Tests
# =============================================================================

class TestSolventDatabase:
    """Tests for amine solvent database."""

    def test_list_solvents(self):
        """Test that solvents can be listed."""
        from difflow_cc import list_solvents

        solvents = list_solvents()
        assert len(solvents) >= 7
        assert "MEA" in solvents
        assert "PZ" in solvents
        assert "MDEA" in solvents

    def test_get_solvent_mea(self):
        """Test getting MEA properties."""
        from difflow_cc import get_solvent

        mea = get_solvent("MEA")
        assert mea.name == "MEA"
        assert mea.MW == pytest.approx(61.08, rel=0.01)
        assert mea.heat_of_absorption == pytest.approx(82.0, rel=0.1)
        assert mea.loading_capacity == pytest.approx(0.5, rel=0.1)

    def test_get_solvent_kinetics(self):
        """Test solvent kinetics data."""
        from difflow_cc import get_solvent

        mea = get_solvent("MEA")
        assert mea.kinetics["mechanism"] == "zwitterion"
        assert mea.kinetics["k2_25C"] == pytest.approx(5900, rel=0.1)

        pz = get_solvent("PZ")
        assert pz.kinetics["k2_25C"] > mea.kinetics["k2_25C"]  # PZ is faster

    def test_unknown_solvent_raises(self):
        """Test that unknown solvent raises KeyError."""
        from difflow_cc import get_solvent

        with pytest.raises(KeyError):
            get_solvent("UNKNOWN")


class TestAdsorbentDatabase:
    """Tests for adsorbent database."""

    def test_list_adsorbents(self):
        """Test that adsorbents can be listed."""
        from difflow_cc import list_adsorbents

        adsorbents = list_adsorbents()
        assert len(adsorbents) >= 8
        assert "Zeolite_13X" in adsorbents
        assert "Mg_MOF_74" in adsorbents

    def test_get_adsorbent_zeolite(self):
        """Test getting Zeolite 13X properties."""
        from difflow_cc import get_adsorbent

        zeolite = get_adsorbent("Zeolite_13X")
        assert zeolite.name == "Zeolite_13X"
        assert zeolite.material_type == "zeolite"
        assert zeolite.CO2_capacity == pytest.approx(5.5, rel=0.2)
        assert zeolite.heat_of_adsorption == pytest.approx(34.65, rel=0.1)

    def test_adsorbent_isotherms(self):
        """Test that isotherms are loaded."""
        from difflow_cc import get_adsorbent

        zeolite = get_adsorbent("Zeolite_13X")
        assert "CO2" in zeolite.isotherms
        assert zeolite.isotherms["CO2"].model == "toth"


class TestMembraneDatabase:
    """Tests for membrane database."""

    def test_list_membranes(self):
        """Test that membranes can be listed."""
        from difflow_cc import list_membranes

        membranes = list_membranes()
        assert len(membranes) >= 9
        assert "Matrimid" in membranes
        assert "ZIF8_Matrimid" in membranes

    def test_get_membrane_matrimid(self):
        """Test getting Matrimid properties."""
        from difflow_cc import get_membrane

        mem = get_membrane("Matrimid")
        assert mem.name == "Matrimid"
        assert mem.membrane_type == "polymeric"
        assert mem.permeability["CO2"] == pytest.approx(10.0, rel=0.1)
        assert mem.selectivity["CO2_N2"] == pytest.approx(31.0, rel=0.1)


# =============================================================================
# Equilibrium Model Tests
# =============================================================================

class TestIsotherms:
    """Tests for adsorption isotherm models."""

    def test_langmuir_basic(self):
        """Test basic Langmuir isotherm."""
        from difflow_cc import langmuir

        # At P=0, q should be 0
        q = langmuir(P=0.0, q_sat=5.0, b=1e-5)
        assert float(q) == pytest.approx(0.0, abs=1e-10)

        # At high P, q should approach q_sat
        q = langmuir(P=1e10, q_sat=5.0, b=1e-5)
        assert float(q) == pytest.approx(5.0, rel=0.01)

    def test_langmuir_temperature_dependence(self):
        """Test temperature-dependent Langmuir."""
        from difflow_cc import langmuir_T

        P = 10000.0  # Pa
        q_sat = 5.0
        b0 = 1e-9
        Q = 30000.0  # J/mol

        # Higher T should give lower loading
        q_low_T = langmuir_T(P, T=298.0, q_sat=q_sat, b0=b0, Q=Q)
        q_high_T = langmuir_T(P, T=373.0, q_sat=q_sat, b0=b0, Q=Q)
        assert float(q_low_T) > float(q_high_T)

    def test_isotherm_class(self):
        """Test Isotherm class from database."""
        from difflow_cc import get_isotherm

        iso = get_isotherm("Zeolite_13X", "CO2")
        P = 15000.0  # Pa (~15% CO2 at 1 bar)
        T = 298.15

        q = iso(P, T)
        assert float(q) > 0
        assert float(q) < 10  # Reasonable capacity

    def test_working_capacity_psa(self):
        """Test PSA working capacity calculation."""
        from difflow_cc import get_isotherm, working_capacity_PSA

        iso = get_isotherm("Zeolite_13X", "CO2")
        wc = working_capacity_PSA(iso, P_ads=101325, P_des=10000, T=298.15)

        assert float(wc) > 0
        assert float(wc) < iso(101325, 298.15)  # Less than full capacity


class TestVLE:
    """Tests for amine VLE models."""

    def test_henry_constant(self):
        """Test Henry's constant calculation."""
        from difflow_cc import henry_constant

        H = henry_constant(T=298.15, solvent="MEA")
        assert float(H) > 0
        assert float(H) < 1e10  # Reasonable magnitude

    def test_co2_equilibrium_pressure(self):
        """Test CO2 equilibrium pressure."""
        from difflow_cc import co2_equilibrium_pressure

        P = co2_equilibrium_pressure(loading=0.3, T=313.15, solvent="MEA")
        assert float(P) > 0

        # Higher loading should give higher P
        P_low = co2_equilibrium_pressure(loading=0.1, T=313.15, solvent="MEA")
        P_high = co2_equilibrium_pressure(loading=0.4, T=313.15, solvent="MEA")
        assert float(P_high) > float(P_low)

    def test_amine_vle_class(self):
        """Test AmineVLE class."""
        from difflow_cc import AmineVLE

        vle = AmineVLE("MEA", C_amine=5000)
        assert vle.loading_capacity == pytest.approx(0.5, rel=0.1)
        assert vle.heat_of_absorption == pytest.approx(82.0, rel=0.1)


# =============================================================================
# Unit Operation Tests
# =============================================================================

class TestAmineAbsorber:
    """Tests for amine absorber."""

    def test_absorber_creation(self):
        """Test absorber instantiation."""
        from difflow_cc import AbsorberParams, AmineAbsorber

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            solvent_conc=30.0,
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(params)
        assert absorber.params.solvent == "MEA"

    def test_absorber_call(self):
        """Test absorber operation."""
        from difflow_cc import AbsorberParams, AmineAbsorber
        from difflow.streams import make_stream

        params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(params)

        # Create flue gas feed
        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},  # 10% CO2
            T=313.15,
            P=101325.0,
        )

        gas_out, solvent_out, info = absorber(feed)

        assert "capture_efficiency" in info
        assert 0 < float(info["capture_efficiency"]) < 1
        assert float(gas_out["F_CO2"]) < float(feed["F_CO2"])


class TestAmineStripper:
    """Tests for amine stripper."""

    def test_stripper_creation(self):
        """Test stripper instantiation."""
        from difflow_cc import StripperParams, AmineStripper

        params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
        )
        stripper = AmineStripper(params)
        assert stripper.params.solvent == "MEA"


class TestMembraneSeparator:
    """Tests for membrane separator."""

    def test_membrane_creation(self):
        """Test membrane instantiation."""
        from difflow_cc import MembraneParams, MembraneSeparator

        params = MembraneParams(
            membrane_type="Matrimid",
            area=1000.0,
            pressure_ratio=10.0,
        )
        membrane = MembraneSeparator(params)
        assert membrane.params.membrane_type == "Matrimid"

    def test_membrane_call(self):
        """Test membrane operation."""
        from difflow_cc import MembraneParams, MembraneSeparator
        from difflow.streams import make_stream

        params = MembraneParams(
            membrane_type="Matrimid",
            area=1000.0,
            pressure_ratio=10.0,
            feed_pressure=1000000.0,
        )
        membrane = MembraneSeparator(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 9.0},
            T=298.15,
            P=1000000.0,
        )

        retentate, permeate, info = membrane(feed)

        assert "stage_cut" in info
        assert 0 < float(info["stage_cut"]) < 1
        assert "CO2_purity" in info


class TestAdsorptionUnits:
    """Tests for adsorption units."""

    def test_psa_creation(self):
        """Test PSA unit instantiation."""
        from difflow_cc import AdsorptionParams, PSAUnit

        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
        )
        psa = PSAUnit(params)
        assert psa.params.adsorbent == "Zeolite_13X"

    def test_psa_call(self):
        """Test PSA operation."""
        from difflow_cc import AdsorptionParams, PSAUnit
        from difflow.streams import make_stream

        params = AdsorptionParams(
            adsorbent="Zeolite_13X",
            cycle_type="PSA",
            P_adsorption=500000.0,
            P_desorption=100000.0,
            bed_mass=100.0,
        )
        psa = PSAUnit(params)

        feed = make_stream(
            flows={"CO2": 1.0, "N2": 4.0},  # 20% CO2
            T=298.15,
            P=500000.0,
        )

        product, offgas, info = psa(feed)

        assert "working_capacity" in info
        assert float(info["working_capacity"]) > 0
        assert "recovery" in info

    def test_tsa_creation(self):
        """Test TSA unit instantiation."""
        from difflow_cc import AdsorptionParams, TSAUnit

        params = AdsorptionParams(
            adsorbent="PEI_Silica",
            cycle_type="TSA",
            T_adsorption=298.15,
            T_desorption=373.15,
        )
        tsa = TSAUnit(params)
        assert tsa.params.cycle_type == "TSA"


# =============================================================================
# JAX Compatibility Tests
# =============================================================================

class TestJAXCompatibility:
    """Tests for JAX compatibility (gradients, JIT)."""

    def test_isotherm_gradient(self):
        """Test that isotherms are differentiable."""
        from difflow_cc import langmuir_T

        def loading_fn(P):
            return langmuir_T(P, T=298.15, q_sat=5.0, b0=1e-9, Q=30000.0)

        dq_dP = grad(loading_fn)(10000.0)
        assert jnp.isfinite(dq_dP)
        assert float(dq_dP) > 0  # Loading increases with pressure

    def test_isotherm_jit(self):
        """Test that isotherms can be JIT compiled."""
        from difflow_cc import langmuir_T

        @jit
        def loading_fn(P, T):
            return langmuir_T(P, T, q_sat=5.0, b0=1e-9, Q=30000.0)

        q = loading_fn(10000.0, 298.15)
        assert jnp.isfinite(q)

    def test_vle_gradient(self):
        """Test that VLE is differentiable."""
        from difflow_cc import co2_equilibrium_pressure

        def pressure_fn(loading):
            return co2_equilibrium_pressure(loading, T=313.15, solvent="MEA")

        dP_dalpha = grad(pressure_fn)(0.3)
        assert jnp.isfinite(dP_dalpha)
        assert float(dP_dalpha) > 0  # Pressure increases with loading

    def test_working_capacity_gradient(self):
        """Test gradient of working capacity w.r.t. temperature."""
        from difflow_cc import get_isotherm

        iso = get_isotherm("Zeolite_13X", "CO2")

        def wc_fn(T):
            q_ads = iso(15000.0, T)
            q_des = iso(1500.0, T)
            return q_ads - q_des

        dwc_dT = grad(wc_fn)(298.15)
        assert jnp.isfinite(dwc_dT)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_amine_capture_loop(self):
        """Test basic amine capture loop."""
        from difflow_cc import (
            AbsorberParams, AmineAbsorber,
            StripperParams, AmineStripper,
        )
        from difflow.streams import make_stream

        # Create absorber
        abs_params = AbsorberParams(
            solvent="MEA",
            n_stages=10,
            L_G_ratio=3.0,
        )
        absorber = AmineAbsorber(abs_params)

        # Create stripper
        strip_params = StripperParams(
            solvent="MEA",
            T_reboiler=393.15,
            target_lean_loading=0.2,
        )
        stripper = AmineStripper(strip_params)

        # Flue gas (realistic scale)
        flue_gas = make_stream(
            flows={"CO2": 15.0, "N2": 85.0},  # 15% CO2
            T=313.15,
            P=101325.0,
        )

        # Run absorber
        treated_gas, rich_solvent, abs_info = absorber(flue_gas)

        # Run stripper
        lean_solvent, co2_product, strip_info = stripper(rich_solvent)

        # Verify outputs
        assert float(abs_info["capture_efficiency"]) > 0
        assert float(strip_info["specific_energy"]) > 0
        # Note: Specific energy values from simplified model may differ
        # from rigorous models. Just verify it's positive and finite.
        assert jnp.isfinite(strip_info["specific_energy"])
