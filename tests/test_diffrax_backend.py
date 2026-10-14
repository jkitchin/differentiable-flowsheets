"""Smoke tests for the diffrax integration backend."""

import jax
import jax.numpy as jnp
import pytest

from difflow.dynamic import DynamicCSTR, DynamicFlowsheet, DynamicTank, integrate
from difflow.streams import make_stream

try:
    from difflow.dynamic import (
        HAS_DIFFRAX,
        integrate_diffrax,
        integrate_stiff,
        list_diffrax_solvers,
    )
except ImportError:
    HAS_DIFFRAX = False

pytestmark = pytest.mark.skipif(not HAS_DIFFRAX, reason="diffrax not installed")


def _decay(t, y):
    return -y


def test_integrate_with_diffrax():
    result = integrate(_decay, jnp.array([1.0]), (0.0, 1.0), method="diffrax")
    assert jnp.all(jnp.isfinite(result.y_final))
    assert result.info.success


def test_integrate_stiff_runs():
    result = integrate_stiff(lambda t, y: -1000 * y, jnp.array([1.0]), (0.0, 0.1))
    assert jnp.all(jnp.isfinite(result.y_final))


def test_jit_and_grad_through_integrate_diffrax():
    """#361: solver statistics used int() and so broke jit."""
    def y_final(k):
        return integrate_diffrax(lambda t, y: -k * y, jnp.array([1.0]), (0.0, 1.0)).y_final[0]

    assert jnp.isfinite(jax.jit(y_final)(2.0))
    assert jnp.isfinite(jax.grad(y_final)(1.0))
    result = integrate_diffrax(_decay, jnp.array([1.0]), (0.0, 1.0))
    assert int(result.info.n_steps) > 0


def test_flowsheet_with_diffrax():
    cstr = DynamicCSTR(
        volume=1.0, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
        stoich=jnp.array([[-1.0], [1.0]]), species_order=["A", "B"],
        rate_params={"k": 0.1}, name="reactor",
    )
    tank = DynamicTank(max_volume=10.0, species_order=["A", "B"], name="storage")
    fs = DynamicFlowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0))
    fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["reactor_out"])
    fs.add_unit(tank, inlet_names=["reactor_out"], outlet_names=["product"])
    result = integrate(fs.derivatives, fs.initial_state(), (0.0, 10.0), method="diffrax:tsit5")
    assert jnp.all(jnp.isfinite(result.y_final))


def test_solver_listing_and_invalid_solver():
    assert "tsit5" in list_diffrax_solvers()
    with pytest.raises(ValueError, match="Unknown diffrax solver"):
        integrate(_decay, jnp.array([1.0]), (0, 1), method="diffrax:invalid_solver")
