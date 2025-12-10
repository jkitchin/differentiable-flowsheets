"""Tests for CSTR unit operation."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_thermo():
    """Simple two-component thermodynamics."""
    species_data = {
        "A": SpeciesData(
            "A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            "B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def simple_rate_fn():
    """First-order reaction rate function."""
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


class TestCSTR:
    def test_cstr_creation(self, simple_thermo, simple_rate_fn):
        """Test CSTR can be created."""
        stoich = jnp.array([[-1.0], [+1.0]])  # A → B
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
        assert cstr is not None

    def test_cstr_isothermal(self, simple_thermo, simple_rate_fn):
        """Test isothermal CSTR operation."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = CSTRParams(
            V=jnp.array(1.0),
            rate_fn=simple_rate_fn,
            stoich=stoich,
            rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

        inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
        outlet, info = cstr(inlet, T_spec=350.0)

        # Check mass balance (A + B should be conserved)
        flows_in = get_flows(inlet)
        flows_out = get_flows(outlet)
        total_in = float(flows_in["A"]) + float(flows_in["B"])
        total_out = float(flows_out["A"]) + float(flows_out["B"])

        assert total_out == pytest.approx(total_in, rel=1e-6)

        # Check some B was produced
        assert float(flows_out["B"]) > 0

        # Check conversion is reported
        assert "conversion" in info
        assert info["conversion"]["A"] > 0

    def test_cstr_differentiability(self, simple_thermo, simple_rate_fn):
        """Test that CSTR is differentiable w.r.t. volume."""
        stoich = jnp.array([[-1.0], [+1.0]])

        def outlet_B(V):
            params = CSTRParams(
                V=V,
                rate_fn=simple_rate_fn,
                stoich=stoich,
                rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
                species_order=["A", "B"],
            )
            cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")
            inlet = make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0)
            outlet, _ = cstr(inlet, T_spec=350.0)
            return outlet["F_B"]

        # Compute gradient
        grad_V = jax.grad(outlet_B)(jnp.array(1.0))

        # Gradient should be positive (more volume = more conversion = more B)
        assert float(grad_V) > 0
