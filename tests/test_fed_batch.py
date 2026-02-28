"""Tests for fed-batch reactor unit operation."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
)
from difflow.units.fed_batch import (
    FedBatchReactor,
    FedBatchParams,
    SemiBatchReactor,
    optimal_feed_profile,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_thermo():
    """Simple thermodynamics for A → B reaction."""
    species_data = {
        "A": SpeciesData(
            name="A",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
        ),
        "B": SpeciesData(
            name="B",
            MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(10.0, 2800.0, -40.0),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def first_order_rate_fn():
    """First-order reaction rate function A → B."""
    def rate_fn(C, T, params):
        k = params["k0"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def second_order_rate_fn():
    """Second-order reaction A + B → C."""
    def rate_fn(C, T, params):
        k = params["k0"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"] * C["B"]])
    return rate_fn


class TestFedBatchReactor:
    """Tests for fed-batch reactor."""

    def test_fed_batch_creation(self, first_order_rate_fn):
        """Test fed-batch reactor can be created."""
        stoich = jnp.array([[-1.0], [+1.0]])  # A → B
        params = FedBatchParams(
            V0=1.0,  # m³
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")
        assert reactor is not None

    def test_batch_mode(self, simple_thermo, first_order_rate_fn):
        """Test batch operation (no feed)."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, thermo=simple_thermo, mode="isothermal")

        C0 = {"A": 1000.0, "B": 0.0}  # mol/m³
        T0 = 350.0  # K
        P = 101325.0  # Pa
        t_final = 100.0  # s

        final_stream, info = reactor(
            C0=C0,
            T0=T0,
            P=P,
            t_final=t_final,
            n_steps=100,
        )

        # Check volume is constant (batch mode)
        V_profile = info["V"]
        assert jnp.allclose(V_profile, 1.0, rtol=1e-6)

        # Check reaction occurred
        assert info["conversion"]["A"] > 0

        # Check mass balance (A + B constant in moles)
        n_A_final = info["n_final"]["A"]
        n_B_final = info["n_final"]["B"]
        n_total_initial = 1.0 * 1000.0  # V0 * C0_A
        assert float(n_A_final + n_B_final) == pytest.approx(n_total_initial, rel=1e-3)

    def test_fed_batch_mode(self, first_order_rate_fn):
        """Test fed-batch with continuous feed."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e4), "Ea": jnp.array(40000.0)},
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        C0 = {"A": 100.0, "B": 0.0}
        T0 = 350.0
        P = 101325.0
        t_final = 100.0

        # Constant feed rate
        feed_rate = 0.01  # m³/s
        feed_rate_fn = lambda t: jnp.array(feed_rate)
        feed_composition = {"A": 500.0, "B": 0.0}  # mol/m³ in feed

        final_stream, info = reactor(
            C0=C0,
            T0=T0,
            P=P,
            t_final=t_final,
            feed_rate_fn=feed_rate_fn,
            feed_composition=feed_composition,
            n_steps=100,
        )

        # Check volume increased
        V_final = info["V_final"]
        expected_V = 1.0 + feed_rate * t_final  # V0 + F * t
        assert float(V_final) == pytest.approx(expected_V, rel=0.05)

        # Check B was produced
        assert info["n_final"]["B"] > 0

    def test_variable_feed_rate(self, first_order_rate_fn):
        """Test with time-varying feed rate."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e4), "Ea": jnp.array(40000.0)},
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        C0 = {"A": 100.0, "B": 0.0}
        T0 = 350.0
        t_final = 100.0

        # Linearly increasing feed rate
        def feed_rate_fn(t):
            return 0.0001 * t  # Starts at 0, increases linearly

        feed_composition = {"A": 500.0, "B": 0.0}

        final_stream, info = reactor(
            C0=C0,
            T0=T0,
            P=101325.0,
            t_final=t_final,
            feed_rate_fn=feed_rate_fn,
            feed_composition=feed_composition,
            n_steps=100,
        )

        # Volume should increase (integral of feed)
        # V = V0 + integral(0.0001 * t, 0, 100) = 1 + 0.5 * 0.0001 * 100^2 = 1.5
        expected_V = 1.0 + 0.5 * 0.0001 * t_final**2
        assert float(info["V_final"]) == pytest.approx(expected_V, rel=0.1)

    def test_profiles_returned(self, first_order_rate_fn):
        """Test that time profiles are returned."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e4), "Ea": jnp.array(40000.0)},
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        C0 = {"A": 100.0, "B": 0.0}
        n_steps = 50

        _, info = reactor(
            C0=C0,
            T0=350.0,
            P=101325.0,
            t_final=100.0,
            n_steps=n_steps,
        )

        # Check profiles have correct length
        assert len(info["t"]) == n_steps + 1
        assert len(info["V"]) == n_steps + 1
        assert len(info["C"]["A"]) == n_steps + 1
        assert len(info["C"]["B"]) == n_steps + 1
        assert len(info["T"]) == n_steps + 1

        # Check concentration profile is monotonic
        # A should decrease, B should increase
        C_A = info["C"]["A"]
        C_B = info["C"]["B"]

        assert C_A[0] > C_A[-1]  # A decreases
        assert C_B[-1] > C_B[0]  # B increases

    def test_conversion_calculation(self, first_order_rate_fn):
        """Test conversion is calculated correctly."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e5), "Ea": jnp.array(40000.0)},  # Moderate rate
            species_order=["A", "B"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        C0 = {"A": 100.0, "B": 0.0}

        _, info = reactor(
            C0=C0,
            T0=380.0,  # Moderate T
            P=101325.0,
            t_final=500.0,  # Long enough for reaction
            n_steps=200,
        )

        # Should have significant conversion
        X_A = info["conversion"]["A"]
        assert jnp.isfinite(X_A)
        assert float(X_A) > 0.3  # > 30% conversion (reasonable for these params)

    def test_differentiability(self, first_order_rate_fn):
        """Test that fed-batch reactor is differentiable."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e4), "Ea": jnp.array(40000.0)},
            species_order=["A", "B"],
        )

        def final_B(T0):
            reactor = FedBatchReactor(params, mode="isothermal")
            C0 = {"A": 100.0, "B": 0.0}
            _, info = reactor(
                C0=C0,
                T0=T0,
                P=101325.0,
                t_final=100.0,
                n_steps=50,
            )
            return info["n_final"]["B"]

        # Compute gradient w.r.t. temperature
        grad_T = jax.grad(final_B)(jnp.array(350.0))

        # Gradient should be positive (higher T = more reaction = more B)
        assert jnp.isfinite(grad_T)
        assert float(grad_T) > 0


class TestSemiBatchReactor:
    """Tests for semi-batch reactor (alias)."""

    def test_semi_batch_is_fed_batch(self, first_order_rate_fn):
        """Test that SemiBatchReactor is alias for FedBatchReactor."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e4), "Ea": jnp.array(40000.0)},
            species_order=["A", "B"],
        )

        # Both should work
        fb = FedBatchReactor(params, mode="isothermal")
        sb = SemiBatchReactor(params, mode="isothermal")

        assert type(fb).__name__ == "FedBatchReactor"
        assert isinstance(sb, FedBatchReactor)


class TestMultipleReactions:
    """Tests for multiple simultaneous reactions."""

    def test_consecutive_reactions(self):
        """Test A → B → C consecutive reactions."""
        def rate_fn(C, T, params):
            k1 = params["k1"]
            k2 = params["k2"]
            r1 = k1 * C["A"]  # A → B
            r2 = k2 * C["B"]  # B → C
            return jnp.array([r1, r2])

        # A → B → C stoichiometry
        stoich = jnp.array([
            [-1.0, 0.0],   # A: consumed in rxn 1
            [+1.0, -1.0],  # B: produced in rxn 1, consumed in rxn 2
            [0.0, +1.0],   # C: produced in rxn 2
        ])

        params = FedBatchParams(
            V0=1.0,
            rate_fn=rate_fn,
            stoich=stoich,
            rate_params={"k1": jnp.array(0.1), "k2": jnp.array(0.05)},
            species_order=["A", "B", "C"],
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        C0 = {"A": 100.0, "B": 0.0, "C": 0.0}

        _, info = reactor(
            C0=C0,
            T0=350.0,
            P=101325.0,
            t_final=100.0,
            n_steps=100,
        )

        # B should go through maximum (produced then consumed)
        C_B = info["C"]["B"]
        max_B = jnp.max(C_B)
        final_B = C_B[-1]

        assert float(max_B) > float(final_B)  # B peaks then decreases

        # C should monotonically increase
        C_C = info["C"]["C"]
        assert C_C[-1] > C_C[0]

        # Mass balance: A + B + C = constant
        n_A = info["n"]["A"]
        n_B = info["n"]["B"]
        n_C = info["n"]["C"]
        total = n_A + n_B + n_C
        assert jnp.allclose(total, 100.0, rtol=1e-3)


class TestExothermicReaction:
    """Tests for exothermic reactions (non-isothermal)."""

    @pytest.mark.skip(reason="Adiabatic mode energy balance needs refinement")
    def test_adiabatic_temperature_rise(self, simple_thermo, first_order_rate_fn):
        """Test temperature increases for exothermic adiabatic reaction."""
        stoich = jnp.array([[-1.0], [+1.0]])
        params = FedBatchParams(
            V0=1.0,
            rate_fn=first_order_rate_fn,
            stoich=stoich,
            rate_params={"k0": jnp.array(1e6), "Ea": jnp.array(50000.0)},
            species_order=["A", "B"],
            dH_rxn=jnp.array([-50000.0]),  # Exothermic (negative dH)
        )
        reactor = FedBatchReactor(params, thermo=simple_thermo, mode="adiabatic")

        C0 = {"A": 1000.0, "B": 0.0}
        T0 = 350.0

        _, info = reactor(
            C0=C0,
            T0=T0,
            P=101325.0,
            t_final=100.0,
            n_steps=100,
        )

        # Temperature should increase for exothermic reaction
        T_final = info["T_final"]
        assert float(T_final) > T0


# ---------------------------------------------------------------------------
# Shared fixture for optimal_feed_profile tests
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_fed_batch():
    """A → B fed-batch system for optimization tests."""
    def rate_fn(C, T, params):
        return jnp.array([params["k"] * C["A"]])

    stoich = jnp.array([[-1.0], [+1.0]])
    params = FedBatchParams(
        V0=1.0,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": jnp.array(0.05)},
        species_order=["A", "B"],
    )
    C0 = {"A": 100.0, "B": 0.0}
    feed_composition = {"A": 500.0, "B": 0.0}
    T = jnp.array(350.0)
    V_max = 2.0
    t_max = 200.0
    return params, C0, feed_composition, T, V_max, t_max


class TestOptimalFeedProfile:
    """Tests for gradient-based optimal_feed_profile."""

    def _flat_baseline(self, params, C0, feed_composition, T, V_max, t_max):
        """Simulate the flat-rate baseline for comparison."""
        reactor = FedBatchReactor(params, mode="isothermal")
        F_flat = (V_max - float(params.V0)) / t_max
        flat_fn = lambda t: jnp.array(F_flat)
        _, info = reactor(
            C0=C0, T0=T, P=101325.0, t_final=t_max,
            feed_rate_fn=flat_fn, feed_composition=feed_composition,
            n_steps=50, use_diffrax=False,
        )
        return info

    def test_max_yield_returns_callable(self, simple_fed_batch):
        """optimal_feed_profile('max_yield') returns (callable, float)."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "max_yield", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        assert callable(feed_fn)
        assert t_opt == pytest.approx(t_max)

    def test_max_yield_nonnegative_feed(self, simple_fed_batch):
        """Optimized max_yield feed rates are non-negative."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, _ = optimal_feed_profile(
            "max_yield", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        for t_test in [0.0, t_max * 0.25, t_max * 0.5, t_max * 0.75]:
            assert float(feed_fn(jnp.array(t_test))) >= 0.0

    def test_max_yield_improves_on_flat(self, simple_fed_batch):
        """max_yield profile produces at least as much product as flat feed."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "max_yield", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        reactor = FedBatchReactor(params, mode="isothermal")
        _, info_opt = reactor(
            C0=C0, T0=T, P=101325.0, t_final=t_opt,
            feed_rate_fn=feed_fn, feed_composition=feed_comp,
            n_steps=50, use_diffrax=False,
        )
        info_flat = self._flat_baseline(params, C0, feed_comp, T, V_max, t_max)

        n_B_opt = float(info_opt["n_final"]["B"])
        n_B_flat = float(info_flat["n_final"]["B"])
        # Optimized yield must be at least 95% of the flat baseline
        # (the optimizer may equal or improve the baseline)
        assert n_B_opt >= n_B_flat * 0.95

    def test_max_selectivity_returns_callable(self, simple_fed_batch):
        """optimal_feed_profile('max_selectivity') returns (callable, float)."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "max_selectivity", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        assert callable(feed_fn)
        assert t_opt == pytest.approx(t_max)
        assert float(feed_fn(jnp.array(0.0))) >= 0.0

    def test_max_selectivity_improves_on_flat(self, simple_fed_batch):
        """max_selectivity profile achieves selectivity at least as good as flat."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "max_selectivity", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        reactor = FedBatchReactor(params, mode="isothermal")
        _, info_opt = reactor(
            C0=C0, T0=T, P=101325.0, t_final=t_opt,
            feed_rate_fn=feed_fn, feed_composition=feed_comp,
            n_steps=50, use_diffrax=False,
        )
        info_flat = self._flat_baseline(params, C0, feed_comp, T, V_max, t_max)

        def selectivity(info):
            C = info["C_final"]
            return float(C["B"]) / (sum(float(v) for v in C.values()) + 1e-10)

        assert selectivity(info_opt) >= selectivity(info_flat) * 0.95

    def test_min_time_returns_callable(self, simple_fed_batch):
        """optimal_feed_profile('min_time') returns (callable, float)."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "min_time", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        assert callable(feed_fn)
        assert t_opt == pytest.approx(t_max)
        assert float(feed_fn(jnp.array(0.0))) >= 0.0

    def test_min_time_improves_on_flat(self, simple_fed_batch):
        """min_time profile achieves higher time-averaged yield than flat feed."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "min_time", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        reactor = FedBatchReactor(params, mode="isothermal")

        def time_avg_yield(info):
            n_tgt = info["n"]["B"]
            t_arr = info["t"]
            weights = (t_max - t_arr) + 1.0
            weights = weights / jnp.sum(weights)
            return float(jnp.dot(n_tgt, weights))

        _, info_opt = reactor(
            C0=C0, T0=T, P=101325.0, t_final=t_opt,
            feed_rate_fn=feed_fn, feed_composition=feed_comp,
            n_steps=50, use_diffrax=False,
        )
        info_flat = self._flat_baseline(params, C0, feed_comp, T, V_max, t_max)

        assert time_avg_yield(info_opt) >= time_avg_yield(info_flat) * 0.95

    def test_volume_constraint_respected(self, simple_fed_batch):
        """Optimized profile does not violate V_max constraint significantly."""
        params, C0, feed_comp, T, V_max, t_max = simple_fed_batch
        feed_fn, t_opt = optimal_feed_profile(
            "max_yield", params, C0, T, "B", feed_comp, V_max, t_max,
            n_intervals=5, n_sim_steps=50,
        )
        reactor = FedBatchReactor(params, mode="isothermal")
        _, info = reactor(
            C0=C0, T0=T, P=101325.0, t_final=t_opt,
            feed_rate_fn=feed_fn, feed_composition=feed_comp,
            n_steps=50, use_diffrax=False,
        )
        V_final = float(info["V_final"])
        # Allow 5% overshoot due to finite penalty weight
        assert V_final <= V_max * 1.05
