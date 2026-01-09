"""Tests for Plug Flow Reactor (PFR) unit operations.

Tests cover:
- PFRParams validation
- PFR isothermal operation
- PFR adiabatic operation
- GasPFR with pressure drop
- GasPFR adiabatic operation
- Gradient compatibility
- DynamicUnit interface methods
- Initialization methods
"""

import pytest
import jax
import jax.numpy as jnp
from jax import Array
import numpy.testing as npt

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

from difflow.units.pfr import (
    PFR,
    PFRParams,
    GasPFR,
    GasPFRParams,
    pfr_conversion_analytical,
)
from difflow.streams import make_stream, get_flows, total_flow
from difflow.thermo import IdealThermo, SpeciesData


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_thermo():
    """Simple two-component thermodynamics for A -> B reaction."""
    species_data = {
        "A": SpeciesData(
            name="A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
            Hf=0.0,
        ),
        "B": SpeciesData(
            name="B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
            Hf=-10000.0,
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def first_order_rate_fn():
    """First-order reaction rate function A -> B.

    r = k * C_A where k = A * exp(-Ea/RT)
    """
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def rate_params():
    """Kinetic parameters for A -> B reaction."""
    return {
        "A": jnp.array(1e8),  # Pre-exponential factor (1/s)
        "Ea": jnp.array(50000.0),  # Activation energy (J/mol)
        "k": 0.1,  # Effective rate constant for initialization
    }


@pytest.fixture
def simple_stoich():
    """Stoichiometry matrix for A -> B reaction."""
    return jnp.array([[-1.0], [+1.0]])


@pytest.fixture
def feed_stream():
    """Standard feed stream for PFR tests."""
    return make_stream(
        {"A": 10.0, "B": 0.0},
        T=350.0,
        P=101325.0,
    )


# =============================================================================
# PFRParams Tests
# =============================================================================


class TestPFRParams:
    """Tests for PFRParams validation."""

    def test_valid_params(self, first_order_rate_fn, rate_params, simple_stoich):
        """Test creating valid PFRParams."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        assert params.V == 1.0
        assert params.species_order == ["A", "B"]

    def test_stoich_species_mismatch(self, first_order_rate_fn, rate_params):
        """Test that mismatched stoichiometry raises error."""
        stoich = jnp.array([[-1.0], [+1.0], [+0.0]])  # 3 species in stoich
        with pytest.raises(ValueError, match="Stoichiometry matrix has 3 rows"):
            PFRParams(
                V=1.0,
                rate_fn=first_order_rate_fn,
                stoich=stoich,
                rate_params=rate_params,
                species_order=["A", "B"],  # Only 2 species
            )

    def test_dH_rxn_mismatch(self, first_order_rate_fn, rate_params, simple_stoich):
        """Test that mismatched dH_rxn raises error."""
        dH_rxn = jnp.array([-10000.0, -5000.0])  # 2 reactions
        with pytest.raises(ValueError, match="dH_rxn has 2 values"):
            PFRParams(
                V=1.0,
                rate_fn=first_order_rate_fn,
                stoich=simple_stoich,  # 1 reaction
                rate_params=rate_params,
                species_order=["A", "B"],
                dH_rxn=dH_rxn,
            )

    def test_params_dict_access(self, first_order_rate_fn, rate_params, simple_stoich):
        """Test ParamsMixin dict-like access."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        assert params["V"] == 1.0
        assert "V" in params
        assert list(params.keys())[:2] == ["V", "rate_fn"]


# =============================================================================
# PFR Isothermal Tests
# =============================================================================


class TestPFRIsothermal:
    """Tests for isothermal PFR operation."""

    def test_pfr_creation(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test PFR can be created."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")
        assert pfr is not None

    def test_pfr_isothermal_operation(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test isothermal PFR produces expected output."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)

        # Check mass balance (A + B conserved)
        inlet_flows = get_flows(feed_stream)
        outlet_flows = get_flows(outlet)
        total_in = inlet_flows["A"] + inlet_flows["B"]
        total_out = outlet_flows["A"] + outlet_flows["B"]
        npt.assert_allclose(total_out, total_in, rtol=1e-6)

        # Check some B was produced
        assert float(outlet_flows["B"]) > 0

        # Check A was consumed
        assert float(outlet_flows["A"]) < float(inlet_flows["A"])

        # Check conversion is reported
        assert "conversion" in info
        assert float(info["conversion"]["A"]) > 0

    def test_pfr_conversion_increases_with_volume(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that conversion increases with reactor volume."""
        conversions = []

        for V in [0.5, 1.0, 2.0]:
            params = PFRParams(
                V=V,
                rate_fn=first_order_rate_fn,
                stoich=simple_stoich,
                rate_params=rate_params,
                species_order=["A", "B"],
            )
            pfr = PFR(params, thermo=simple_thermo, mode="isothermal")
            _, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)
            conversions.append(float(info["conversion"]["A"]))

        # Conversion should increase with volume
        assert conversions[1] > conversions[0]
        assert conversions[2] > conversions[1]

    def test_pfr_conversion_increases_with_temperature(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that conversion increases with temperature."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        _, info_low_T = pfr(feed_stream, volumetric_flow=0.1, T_spec=330.0)
        _, info_high_T = pfr(feed_stream, volumetric_flow=0.1, T_spec=370.0)

        assert float(info_high_T["conversion"]["A"]) > float(info_low_T["conversion"]["A"])

    def test_pfr_profiles_output(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that profiles are returned in info."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            n_save_points=51,
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        _, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)

        # Check profiles exist
        assert "profiles" in info
        profiles = info["profiles"]
        assert "V" in profiles
        assert "F" in profiles
        assert "T" in profiles

        # Check profile dimensions
        assert profiles["V"].shape == (51,)
        assert profiles["F"].shape == (51, 2)
        assert profiles["T"].shape == (51,)

        # Check volume ranges from 0 to V
        npt.assert_allclose(profiles["V"][0], 0.0)
        npt.assert_allclose(profiles["V"][-1], 1.0)


# =============================================================================
# PFR Adiabatic Tests
# =============================================================================


class TestPFRAdiabatic:
    """Tests for adiabatic PFR operation."""

    def test_adiabatic_requires_thermo(
        self, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test that adiabatic mode requires thermo."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-10000.0]),
        )
        with pytest.raises(ValueError, match="Thermo object required"):
            PFR(params, thermo=None, mode="adiabatic")

    def test_adiabatic_requires_dH_rxn(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test that adiabatic mode requires dH_rxn."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            # No dH_rxn
        )
        with pytest.raises(ValueError, match="dH_rxn required"):
            PFR(params, thermo=simple_thermo, mode="adiabatic")

    def test_pfr_adiabatic_exothermic(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test adiabatic PFR with exothermic reaction (T increases)."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),  # Exothermic
        )
        pfr = PFR(params, thermo=simple_thermo, mode="adiabatic")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1)

        # Temperature should increase for exothermic reaction
        assert float(outlet["T"]) > float(feed_stream["T"])

        # Check mass balance
        inlet_flows = get_flows(feed_stream)
        outlet_flows = get_flows(outlet)
        total_in = inlet_flows["A"] + inlet_flows["B"]
        total_out = outlet_flows["A"] + outlet_flows["B"]
        npt.assert_allclose(total_out, total_in, rtol=1e-5)

    def test_pfr_adiabatic_endothermic(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test adiabatic PFR with endothermic reaction (T decreases)."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([+50000.0]),  # Endothermic
        )
        pfr = PFR(params, thermo=simple_thermo, mode="adiabatic")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1)

        # Temperature should decrease for endothermic reaction
        assert float(outlet["T"]) < float(feed_stream["T"])


# =============================================================================
# GasPFR Tests
# =============================================================================


class TestGasPFRParams:
    """Tests for GasPFRParams validation."""

    def test_valid_params(self, first_order_rate_fn, rate_params, simple_stoich):
        """Test creating valid GasPFRParams."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            alpha=1000.0,  # Pressure drop parameter
        )
        assert params.V == 1.0
        assert params.alpha == 1000.0


class TestGasPFRIsothermal:
    """Tests for isothermal gas-phase PFR."""

    def test_gas_pfr_no_pressure_drop(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test gas PFR with no pressure drop."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            alpha=None,  # No pressure drop
        )
        pfr = GasPFR(params, thermo=simple_thermo, mode="isothermal")

        feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = pfr(feed, T_spec=350.0)

        # Pressure should be nearly unchanged
        npt.assert_allclose(outlet["P"], feed["P"], rtol=1e-6)

        # Check mass balance
        inlet_flows = get_flows(feed)
        outlet_flows = get_flows(outlet)
        total_in = inlet_flows["A"] + inlet_flows["B"]
        total_out = outlet_flows["A"] + outlet_flows["B"]
        npt.assert_allclose(total_out, total_in, rtol=1e-5)

    def test_gas_pfr_with_pressure_drop(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test gas PFR with pressure drop."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            alpha=10000.0,  # Moderate pressure drop
        )
        pfr = GasPFR(params, thermo=simple_thermo, mode="isothermal")

        feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = pfr(feed, T_spec=350.0)

        # Pressure should decrease
        assert float(outlet["P"]) < float(feed["P"])

        # Check pressure drop is reported
        assert "pressure_drop" in info
        assert float(info["pressure_drop"]) > 0

    def test_gas_pfr_profiles_include_pressure(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test that gas PFR profiles include pressure."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            alpha=5000.0,
            n_save_points=51,
        )
        pfr = GasPFR(params, thermo=simple_thermo, mode="isothermal")

        feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        _, info = pfr(feed, T_spec=350.0)

        profiles = info["profiles"]
        assert "P" in profiles
        assert "Q" in profiles
        assert profiles["P"].shape == (51,)
        assert profiles["Q"].shape == (51,)

        # Pressure should decrease along reactor
        assert profiles["P"][-1] < profiles["P"][0]


class TestGasPFRAdiabatic:
    """Tests for adiabatic gas-phase PFR."""

    def test_gas_pfr_adiabatic_creation(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test adiabatic gas PFR can be created."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),
            alpha=5000.0,
        )
        pfr = GasPFR(params, thermo=simple_thermo, mode="adiabatic")
        assert pfr is not None

    def test_gas_pfr_adiabatic_operation(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test adiabatic gas PFR with pressure drop."""
        params = GasPFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),  # Exothermic
            alpha=5000.0,
        )
        pfr = GasPFR(params, thermo=simple_thermo, mode="adiabatic")

        feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
        outlet, info = pfr(feed)

        # Temperature should increase (exothermic)
        assert float(outlet["T"]) > float(feed["T"])

        # Pressure should decrease
        assert float(outlet["P"]) < float(feed["P"])

        # Mass balance
        inlet_flows = get_flows(feed)
        outlet_flows = get_flows(outlet)
        total_in = inlet_flows["A"] + inlet_flows["B"]
        total_out = outlet_flows["A"] + outlet_flows["B"]
        npt.assert_allclose(total_out, total_in, rtol=1e-4)


# =============================================================================
# Gradient Compatibility Tests
# =============================================================================


class TestPFRGradient:
    """Tests for PFR gradient compatibility."""

    def test_pfr_gradient_wrt_volume(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test PFR gradient with respect to volume."""
        stoich = simple_stoich

        def get_conversion(V):
            params = PFRParams(
                V=V,
                rate_fn=first_order_rate_fn,
                stoich=stoich,
                rate_params=rate_params,
                species_order=["A", "B"],
            )
            pfr = PFR(params, thermo=simple_thermo, mode="isothermal")
            feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, info = pfr(feed, volumetric_flow=0.1, T_spec=350.0)
            return info["conversion"]["A"]

        # Compute gradient
        grad_fn = jax.grad(get_conversion)
        grad_V = grad_fn(1.0)

        # Gradient should be positive (more volume = more conversion)
        assert jnp.isfinite(grad_V)
        assert float(grad_V) > 0

    def test_pfr_gradient_wrt_temperature(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test PFR gradient with respect to temperature."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        def get_conversion(T):
            outlet, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=T)
            return info["conversion"]["A"]

        grad_fn = jax.grad(get_conversion)
        grad_T = grad_fn(350.0)

        # Gradient should be positive (higher T = higher rate = more conversion)
        assert jnp.isfinite(grad_T)
        assert float(grad_T) > 0

    def test_gas_pfr_gradient_wrt_alpha(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test GasPFR gradient with respect to pressure drop parameter."""
        stoich = simple_stoich

        def get_outlet_pressure(alpha):
            params = GasPFRParams(
                V=1.0,
                rate_fn=first_order_rate_fn,
                stoich=stoich,
                rate_params=rate_params,
                species_order=["A", "B"],
                alpha=alpha,
            )
            pfr = GasPFR(params, thermo=simple_thermo, mode="isothermal")
            feed = make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)
            outlet, info = pfr(feed, T_spec=350.0)
            return outlet["P"]

        grad_fn = jax.grad(get_outlet_pressure)
        grad_alpha = grad_fn(5000.0)

        # Gradient should be negative (more alpha = more pressure drop = lower outlet P)
        assert jnp.isfinite(grad_alpha)
        assert float(grad_alpha) < 0


# =============================================================================
# Analytical Conversion Tests
# =============================================================================


class TestPFRAnalytical:
    """Tests for analytical PFR conversion formula."""

    def test_pfr_conversion_first_order(self):
        """Test first-order analytical conversion."""
        k = jnp.array(0.1)  # 1/s
        tau = jnp.array(10.0)  # s

        X = pfr_conversion_analytical(k, tau, order=1)

        # X = 1 - exp(-k*tau) = 1 - exp(-1) ≈ 0.632
        expected = 1.0 - jnp.exp(-1.0)
        npt.assert_allclose(X, expected, rtol=1e-6)

    def test_pfr_conversion_limits(self):
        """Test analytical conversion at limits."""
        # Very short residence time -> low conversion
        X_low = pfr_conversion_analytical(0.1, 0.1)
        assert float(X_low) < 0.1

        # Very long residence time -> high conversion
        X_high = pfr_conversion_analytical(0.1, 100.0)
        assert float(X_high) > 0.99

    def test_pfr_conversion_second_order_not_implemented(self):
        """Test that second-order raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            pfr_conversion_analytical(0.1, 10.0, order=2)

    def test_pfr_conversion_invalid_order(self):
        """Test that invalid order raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported reaction order"):
            pfr_conversion_analytical(0.1, 10.0, order=3)


# =============================================================================
# DynamicUnit Interface Tests
# =============================================================================


class TestPFRDynamicInterface:
    """Tests for PFR DynamicUnit interface methods."""

    def test_state_spec_isothermal(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test state_spec for isothermal PFR."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        spec = pfr.state_spec()

        # Should have 2 state variables (F_out_A, F_out_B)
        assert len(spec.variables) == 2
        assert spec.variables[0].name == "F_out_A"
        assert spec.variables[1].name == "F_out_B"

    def test_state_spec_adiabatic(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich
    ):
        """Test state_spec for adiabatic PFR."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),
        )
        pfr = PFR(params, thermo=simple_thermo, mode="adiabatic")

        spec = pfr.state_spec()

        # Should have 3 state variables (F_out_A, F_out_B, T_out)
        assert len(spec.variables) == 3
        assert spec.variables[2].name == "T_out"

    def test_derivatives_pseudo_steady(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that derivatives return zeros (pseudo-steady-state)."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        state = jnp.array([5.0, 5.0])  # Arbitrary state
        inputs = {"inlet": feed_stream}
        derivs = pfr.derivatives(0.0, state, inputs)

        # All derivatives should be zero
        npt.assert_array_equal(derivs, jnp.zeros(2))


# =============================================================================
# Initialization Tests
# =============================================================================


class TestPFRInitialize:
    """Tests for PFR initialization method."""

    def test_initialize_returns_estimates(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that initialize returns proper estimates."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        result = pfr.initialize(feed_stream, volumetric_flow=0.1)

        # Check result structure
        assert "outlet" in result
        assert "states" in result
        assert "info" in result

        # Check states
        assert "conversion" in result["states"]
        assert "residence_time" in result["states"]

        # Check info
        assert result["info"]["method"] == "analytical_pfr_estimate"

    def test_initialize_adiabatic_includes_temperature(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test that adiabatic initialize includes temperature estimate."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),
        )
        pfr = PFR(params, thermo=simple_thermo, mode="adiabatic")

        result = pfr.initialize(feed_stream, volumetric_flow=0.1)

        # Should have T_out in states
        assert "T_out" in result["states"]


# =============================================================================
# Edge Cases
# =============================================================================


class TestPFREdgeCases:
    """Tests for PFR edge cases."""

    def test_very_small_volume(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test PFR with very small volume (low conversion)."""
        params = PFRParams(
            V=0.001,  # Very small
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)

        # Conversion should be small (note: depends on kinetics, so use 0.1 as threshold)
        assert float(info["conversion"]["A"]) < 0.1

    def test_very_large_volume(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test PFR with very large volume (high conversion)."""
        params = PFRParams(
            V=100.0,  # Very large
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)

        # Conversion should be very high
        assert float(info["conversion"]["A"]) > 0.99

    def test_no_reaction_at_low_temperature(
        self, simple_thermo, first_order_rate_fn, rate_params, simple_stoich, feed_stream
    ):
        """Test PFR at very low temperature (negligible reaction)."""
        params = PFRParams(
            V=1.0,
            rate_fn=first_order_rate_fn,
            stoich=simple_stoich,
            rate_params=rate_params,
            species_order=["A", "B"],
        )
        pfr = PFR(params, thermo=simple_thermo, mode="isothermal")

        outlet, info = pfr(feed_stream, volumetric_flow=0.1, T_spec=250.0)  # Low T

        # Conversion should be small compared to higher temperatures
        # At 250K vs 350K, conversion should be significantly lower
        _, info_high_T = pfr(feed_stream, volumetric_flow=0.1, T_spec=350.0)
        assert float(info["conversion"]["A"]) < float(info_high_T["conversion"]["A"])
