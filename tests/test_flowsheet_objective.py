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
