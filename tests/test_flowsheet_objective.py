"""Tests for Flowsheet.make_objective_fn and create_objective (issue #62).

Verifies that the objective function actually varies when unit parameters are
changed, and that JAX gradients flow through the parameter updates correctly.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    Unit,
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
    create_objective,
)

# Enable 64-bit precision for numerical stability in tests
jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_thermo():
    """Two-component thermodynamics (A and B)."""
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
    """First-order Arrhenius rate function: A -> B."""
    def rate_fn(C, T, params):
        k = params["A"] * jnp.exp(-params["Ea"] / (8.314 * T))
        return jnp.array([k * C["A"]])
    return rate_fn


@pytest.fixture
def cstr_flowsheet(simple_thermo, simple_rate_fn):
    """A minimal single-CSTR flowsheet (no recycles)."""
    stoich = jnp.array([[-1.0], [+1.0]])  # A -> B
    params = CSTRParams(
        V=jnp.array(1.0),
        rate_fn=simple_rate_fn,
        stoich=stoich,
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        species_order=["A", "B"],
    )
    cstr = CSTR(params, thermo=simple_thermo, mode="isothermal")

    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit(
        name="reactor",
        operation=cstr,
        inlet_names=["feed"],
        outlet_names=["product"],
        params={"T_spec": 350.0},
    ))
    return fs


# ---------------------------------------------------------------------------
# Tests: objective varies with parameters
# ---------------------------------------------------------------------------


class TestMakeObjectiveFnVariesWithParams:
    """The objective must not be constant w.r.t. the optimisation variables."""

    def test_objective_changes_with_volume(self, cstr_flowsheet):
        """Larger reactor volume -> higher B yield -> different objective."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        val_small = float(objective({"reactor.V": jnp.array(0.5)}))
        val_large = float(objective({"reactor.V": jnp.array(5.0)}))

        assert val_small != val_large, (
            "Objective is constant: parameter update had no effect on the solve"
        )
        # Larger volume -> more conversion -> more B
        assert val_large > val_small

    def test_objective_does_not_modify_original_flowsheet(self, cstr_flowsheet):
        """Calling make_objective_fn must not mutate the source flowsheet."""
        original_V = float(cstr_flowsheet.units[0].operation.params.V)

        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)
        _ = objective({"reactor.V": jnp.array(99.0)})

        assert float(cstr_flowsheet.units[0].operation.params.V) == pytest.approx(
            original_V
        ), "make_objective_fn mutated the original flowsheet"

    def test_create_objective_function_varies(self, cstr_flowsheet):
        """create_objective (module-level helper) must also vary with params."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = create_objective(cstr_flowsheet, obj_fn)

        val1 = float(objective({"reactor.V": jnp.array(1.0)}))
        val2 = float(objective({"reactor.V": jnp.array(3.0)}))

        assert val1 != val2


# ---------------------------------------------------------------------------
# Tests: gradient flows through parameter updates
# ---------------------------------------------------------------------------


class TestMakeObjectiveFnGradients:
    """JAX gradients should flow through the parameter update."""

    def test_grad_wrt_volume_is_positive(self, cstr_flowsheet):
        """d(F_B)/d(V) should be positive at V=1."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        grad_fn = jax.grad(lambda V: objective({"reactor.V": V}))
        g = grad_fn(jnp.array(1.0))

        assert jnp.isfinite(g), "Gradient is not finite"
        assert float(g) > 0.0, "Expected positive gradient d(F_B)/d(V)"

    def test_grad_is_finite_and_nonzero(self, cstr_flowsheet):
        """Gradient must be finite and non-zero (not a constant function)."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        grad_fn = jax.grad(lambda V: objective({"reactor.V": V}))
        g = grad_fn(jnp.array(2.0))

        assert jnp.isfinite(g), "Gradient is not finite"
        assert float(g) != 0.0, "Gradient is zero: optimising a constant function"


# ---------------------------------------------------------------------------
# Tests: error handling in _apply_params
# ---------------------------------------------------------------------------


class TestApplyParamsErrors:
    def test_missing_unit_raises_key_error(self, cstr_flowsheet):
        """A dotted key with an unknown unit name must raise KeyError."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        with pytest.raises(KeyError, match="nonexistent"):
            objective({"nonexistent.V": jnp.array(1.0)})

    def test_missing_dot_raises_value_error(self, cstr_flowsheet):
        """A key without a dot must raise ValueError."""
        def obj_fn(streams):
            return streams["product"]["F_B"]

        objective = cstr_flowsheet.make_objective_fn(obj_fn)

        with pytest.raises(ValueError, match="dot notation"):
            objective({"reactor_V": jnp.array(1.0)})


# ---------------------------------------------------------------------------
# Tests: feed streams as levers
# ---------------------------------------------------------------------------


class TestFeedStreamParams:
    """``feed:<name>.<field>`` keys make feed rate and composition levers."""

    def test_total_flow_scales_the_feed(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({"feed:feed.total_flow": 20.0})

        assert float(fs.feeds["feed"]["F_A"]) == pytest.approx(20.0)
        # Composition held: B was zero and stays zero.
        assert float(fs.feeds["feed"]["F_B"]) == pytest.approx(0.0)
        # Original untouched.
        assert float(cstr_flowsheet.feeds["feed"]["F_A"]) == pytest.approx(10.0)

    def test_mole_fraction_holds_the_total(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({"feed:feed.x_B": 0.25})
        feed = fs.feeds["feed"]

        assert float(feed["F_A"] + feed["F_B"]) == pytest.approx(10.0)
        assert float(feed["F_B"]) == pytest.approx(2.5)
        assert float(feed["F_A"]) == pytest.approx(7.5)

    def test_conditions_and_species_flows(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({
            "feed:feed.T": 320.0,
            "feed:feed.P": 2e5,
            "feed:feed.F_B": 1.0,
        })
        feed = fs.feeds["feed"]

        assert float(feed["T"]) == pytest.approx(320.0)
        assert float(feed["P"]) == pytest.approx(2e5)
        assert float(feed["F_B"]) == pytest.approx(1.0)

    def test_feed_and_unit_keys_together(self, cstr_flowsheet):
        fs = cstr_flowsheet._apply_params({
            "reactor.V": jnp.array(3.0),
            "feed:feed.total_flow": 5.0,
        })

        assert float(fs.units[0].operation.params.V) == pytest.approx(3.0)
        assert float(fs.feeds["feed"]["F_A"]) == pytest.approx(5.0)

    def test_unknown_feed_name_raises(self, cstr_flowsheet):
        with pytest.raises(KeyError, match="No feed stream named"):
            cstr_flowsheet._apply_params({"feed:nope.total_flow": 1.0})

    def test_unknown_feed_field_raises(self, cstr_flowsheet):
        with pytest.raises(KeyError, match="Unknown feed field"):
            cstr_flowsheet._apply_params({"feed:feed.enthalpy": 1.0})

    def test_missing_dot_raises(self, cstr_flowsheet):
        with pytest.raises(ValueError, match="feed:<stream_name>"):
            cstr_flowsheet._apply_params({"feed:feed": 1.0})

    def test_objective_varies_with_feed_rate(self, cstr_flowsheet):
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        small = float(objective({"feed:feed.total_flow": jnp.array(5.0)}))
        large = float(objective({"feed:feed.total_flow": jnp.array(20.0)}))

        assert large > small

    def test_gradient_through_a_feed_lever(self, cstr_flowsheet):
        """The whole point: d(product)/d(feed rate) must be finite and right."""
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        def scalar(F):
            return objective({"feed:feed.total_flow": F})

        F0 = jnp.array(10.0)
        g = float(jax.grad(scalar)(F0))
        h = 1e-4
        fd = (float(scalar(F0 + h)) - float(scalar(F0 - h))) / (2 * h)

        assert jnp.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-5)

    def test_gradient_through_a_composition_lever(self, cstr_flowsheet):
        objective = cstr_flowsheet.make_objective_fn(
            lambda streams: streams["product"]["F_B"]
        )

        def scalar(x):
            return objective({"feed:feed.x_B": x})

        x0 = jnp.array(0.2)
        g = float(jax.grad(scalar)(x0))
        h = 1e-5
        fd = (float(scalar(x0 + h)) - float(scalar(x0 - h))) / (2 * h)

        assert jnp.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-4)
