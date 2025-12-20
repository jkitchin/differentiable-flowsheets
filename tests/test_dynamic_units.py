"""Tests for DynamicUnit interface on refactored unit operations.

Tests verify that CSTR, PFR, and FedBatchReactor implement the DynamicUnit
protocol correctly and can be used with the unified dynamic framework.
"""

import pytest
import jax
import jax.numpy as jnp

from difflow.dynamic import (
    DynamicUnit,
    integrate,
    integrate_unit,
    StateSpec,
)
from difflow.streams import make_stream, get_flows
from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.pfr import PFR, PFRParams, GasPFR, GasPFRParams
from difflow.units.fed_batch import FedBatchReactor, FedBatchParams


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_rate_fn():
    """Simple first-order rate function: A -> B."""
    def rate_fn(C, T, params):
        k = params.get("k", 0.1)
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def arrhenius_rate_fn():
    """Arrhenius rate function."""
    def rate_fn(C, T, params):
        k0 = params.get("k0", 1e6)
        Ea = params.get("Ea", 50000.0)
        R = 8.314
        k = k0 * jnp.exp(-Ea / (R * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def stoich_A_to_B():
    """Stoichiometry for A -> B."""
    return jnp.array([[-1.0], [1.0]])


@pytest.fixture
def inlet_stream():
    """Standard inlet stream."""
    return make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)


# =============================================================================
# CSTR DynamicUnit Interface Tests
# =============================================================================

class TestCSTRDynamicInterface:
    """Tests for CSTR's DynamicUnit interface."""

    def test_cstr_is_dynamic_unit(self, simple_rate_fn, stoich_A_to_B):
        """Test that CSTR satisfies DynamicUnit protocol."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))
        assert isinstance(cstr, DynamicUnit)

    def test_cstr_state_spec_isothermal(self, simple_rate_fn, stoich_A_to_B):
        """Test state_spec for isothermal CSTR."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ), mode="isothermal")

        spec = cstr.state_spec()
        assert spec.n_states == 2  # n_A, n_B (no T for isothermal)
        assert "n_A" in spec.names
        assert "n_B" in spec.names
        assert "T" not in spec.names

    def test_cstr_state_spec_adiabatic(self, simple_rate_fn, stoich_A_to_B):
        """Test state_spec for adiabatic CSTR includes temperature."""
        from difflow.thermo import IdealThermo, SpeciesData

        # Create thermo for adiabatic mode with full SpeciesData
        thermo = IdealThermo({
            "A": SpeciesData(
                name="A",
                MW=50.0,
                Cp_coeffs=(75.0, 0, 0, 0),
                Hvap_coeffs=(30000.0, 0.38, 0, 0),
                antoine_coeffs=(10.0, 1500.0, -40.0),
            ),
            "B": SpeciesData(
                name="B",
                MW=50.0,
                Cp_coeffs=(75.0, 0, 0, 0),
                Hvap_coeffs=(30000.0, 0.38, 0, 0),
                antoine_coeffs=(10.0, 1500.0, -40.0),
            ),
        })

        cstr = CSTR(
            CSTRParams(
                V=1.0,
                rate_fn=simple_rate_fn,
                stoich=stoich_A_to_B,
                rate_params={"k": 0.1},
                species_order=["A", "B"],
                dH_rxn=jnp.array([-50000.0]),
            ),
            thermo=thermo,
            mode="adiabatic",
        )

        spec = cstr.state_spec()
        assert spec.n_states == 3  # n_A, n_B, T
        assert "T" in spec.names

    def test_cstr_initial_state(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test initial state computation."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y0 = cstr.initial_state({"inlet": inlet_stream})
        assert y0.shape == (2,)
        assert y0[0] > 0  # Some initial moles

    def test_cstr_derivatives(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test derivative computation."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y = jnp.array([60.0, 0.0])  # 60 mol A, 0 mol B
        dy = cstr.derivatives(jnp.array(0.0), y, {"inlet": inlet_stream})

        assert dy.shape == (2,)
        # With reaction consuming A, both inlet/outlet flow and reaction affect dy

    def test_cstr_outputs(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test output stream computation."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y = jnp.array([30.0, 30.0])  # 30 mol A, 30 mol B
        outputs = cstr.outputs(jnp.array(0.0), y, {"inlet": inlet_stream})

        assert "outlet" in outputs
        outlet = outputs["outlet"]
        assert "F_A" in outlet
        assert "F_B" in outlet
        # Check composition reflects state (50% A, 50% B)
        flows = get_flows(outlet)
        total = flows["A"] + flows["B"]
        assert jnp.isclose(flows["A"] / total, 0.5, atol=0.01)

    def test_cstr_integrate_unit(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test integration using integrate_unit."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        result = integrate_unit(
            cstr,
            inputs={"inlet": inlet_stream},
            t_span=(0.0, 600.0),  # 10 minutes
            method="RK4",
            n_steps=100,
        )

        assert result.info.success
        assert result.trajectory.y.shape[0] == 101  # n_steps + 1

    def test_cstr_gradient_through_integration(
        self, simple_rate_fn, stoich_A_to_B, inlet_stream
    ):
        """Test gradient computation through integration."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        def objective(y0):
            f = lambda t, y: cstr.derivatives(t, y, {"inlet": inlet_stream})
            result = integrate(f, y0, (0.0, 100.0), method="RK4", n_steps=50)
            return result.y_final[1]  # Final moles of B

        y0 = jnp.array([60.0, 0.0])
        grad = jax.grad(objective)(y0)

        # Gradient should exist and be reasonable
        assert not jnp.any(jnp.isnan(grad))


# =============================================================================
# PFR DynamicUnit Interface Tests
# =============================================================================

class TestPFRDynamicInterface:
    """Tests for PFR's DynamicUnit interface (pseudo-steady-state)."""

    def test_pfr_is_dynamic_unit(self, simple_rate_fn, stoich_A_to_B):
        """Test that PFR satisfies DynamicUnit protocol."""
        pfr = PFR(PFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))
        assert isinstance(pfr, DynamicUnit)

    def test_pfr_state_spec(self, simple_rate_fn, stoich_A_to_B):
        """Test state_spec returns outlet flows."""
        pfr = PFR(PFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        spec = pfr.state_spec()
        assert "F_out_A" in spec.names
        assert "F_out_B" in spec.names

    def test_pfr_derivatives_are_zero(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test that PFR derivatives are zero (pseudo-steady-state)."""
        pfr = PFR(PFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y0 = pfr.initial_state({"inlet": inlet_stream})
        dy = pfr.derivatives(jnp.array(0.0), y0, {"inlet": inlet_stream})

        # All derivatives should be zero for pseudo-steady-state
        assert jnp.allclose(dy, 0.0)

    def test_pfr_outputs(self, simple_rate_fn, stoich_A_to_B, inlet_stream):
        """Test output stream computation runs spatial integration."""
        pfr = PFR(PFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
            n_save_points=50,
        ))

        y0 = pfr.initial_state({"inlet": inlet_stream})
        outputs = pfr.outputs(jnp.array(0.0), y0, {"inlet": inlet_stream})

        assert "outlet" in outputs
        outlet = outputs["outlet"]
        flows = get_flows(outlet)

        # Some A should be converted to B
        assert flows["A"] < 1.0  # Less than inlet
        assert flows["B"] > 0.0  # Product formed


class TestGasPFRDynamicInterface:
    """Tests for GasPFR's DynamicUnit interface."""

    def test_gas_pfr_is_dynamic_unit(self, simple_rate_fn, stoich_A_to_B):
        """Test that GasPFR satisfies DynamicUnit protocol."""
        pfr = GasPFR(GasPFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))
        assert isinstance(pfr, DynamicUnit)

    def test_gas_pfr_state_spec_includes_T_P(self, simple_rate_fn, stoich_A_to_B):
        """Test state_spec includes T and P for gas PFR."""
        pfr = GasPFR(GasPFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        spec = pfr.state_spec()
        assert "T_out" in spec.names
        assert "P_out" in spec.names


# =============================================================================
# FedBatchReactor DynamicUnit Interface Tests
# =============================================================================

class TestFedBatchDynamicInterface:
    """Tests for FedBatchReactor's DynamicUnit interface."""

    def test_fed_batch_is_dynamic_unit(self, simple_rate_fn, stoich_A_to_B):
        """Test that FedBatchReactor satisfies DynamicUnit protocol."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))
        assert isinstance(reactor, DynamicUnit)

    def test_fed_batch_state_spec_includes_volume(self, simple_rate_fn, stoich_A_to_B):
        """Test state_spec includes volume for fed-batch."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        spec = reactor.state_spec()
        assert "V" in spec.names
        assert "n_A" in spec.names
        assert "n_B" in spec.names

    def test_fed_batch_initial_state(self, simple_rate_fn, stoich_A_to_B):
        """Test initial state computation."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y0 = reactor.initial_state(
            inputs={},
            params={"C0": {"A": 100.0, "B": 0.0}, "T0": 350.0}
        )

        # State should be [V, n_A, n_B]
        assert y0.shape == (3,)
        assert y0[0] == 1.0  # V = V0
        assert y0[1] == 100.0  # n_A = V0 * C0["A"]
        assert y0[2] == 0.0  # n_B = 0

    def test_fed_batch_derivatives_batch_mode(self, simple_rate_fn, stoich_A_to_B):
        """Test derivatives in batch mode (no feed)."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y = jnp.array([1.0, 100.0, 0.0])  # V=1, n_A=100, n_B=0
        dy = reactor.derivatives(
            jnp.array(0.0), y, {},
            params={"T_spec": 350.0}
        )

        # dV/dt should be 0 (no feed)
        assert dy[0] == 0.0
        # dn_A/dt should be negative (A consumed)
        assert dy[1] < 0
        # dn_B/dt should be positive (B produced)
        assert dy[2] > 0

    def test_fed_batch_integrate_with_dynamic_framework(
        self, simple_rate_fn, stoich_A_to_B
    ):
        """Test integration using the unified dynamic framework."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        # Create derivative function with params
        params = {"T_spec": 350.0}

        def f(t, y):
            return reactor.derivatives(t, y, {}, params)

        y0 = jnp.array([1.0, 100.0, 0.0])
        result = integrate(f, y0, (0.0, 100.0), method="RK4", n_steps=100)

        assert result.info.success
        # A should decrease, B should increase
        assert result.y_final[1] < 100.0
        assert result.y_final[2] > 0.0
        # Volume should stay constant (batch mode)
        assert jnp.isclose(result.y_final[0], 1.0)

    def test_fed_batch_outputs(self, simple_rate_fn, stoich_A_to_B):
        """Test outputs computation."""
        reactor = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        y = jnp.array([1.0, 50.0, 50.0])
        outputs = reactor.outputs(jnp.array(0.0), y, {}, params={"T_spec": 350.0})

        assert "contents" in outputs
        assert "V" in outputs
        assert "n" in outputs
        assert "C" in outputs

        # Check concentrations
        assert jnp.isclose(outputs["C"]["A"], 50.0)
        assert jnp.isclose(outputs["C"]["B"], 50.0)


# =============================================================================
# Integration Tests - Mixed Unit Types
# =============================================================================

class TestMixedDynamicUnits:
    """Tests for using different unit types together."""

    def test_all_units_share_interface(
        self, simple_rate_fn, stoich_A_to_B, inlet_stream
    ):
        """Test that all units provide the same interface methods."""
        cstr = CSTR(CSTRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        pfr = PFR(PFRParams(
            V=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        fb = FedBatchReactor(FedBatchParams(
            V0=1.0,
            rate_fn=simple_rate_fn,
            stoich=stoich_A_to_B,
            rate_params={"k": 0.1},
            species_order=["A", "B"],
        ))

        units = [cstr, pfr, fb]

        for unit in units:
            # All should have these methods
            assert hasattr(unit, 'state_spec')
            assert hasattr(unit, 'initial_state')
            assert hasattr(unit, 'derivatives')
            assert hasattr(unit, 'outputs')

            # All should return StateSpec
            spec = unit.state_spec()
            assert isinstance(spec, StateSpec)
            assert spec.n_states > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
