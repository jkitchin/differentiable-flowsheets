"""Smoke tests for the fed-batch reactor."""

import jax
import jax.numpy as jnp
import pytest

from difflow.catalog import describe_class
from difflow.flowsheet import Flowsheet, Unit
from difflow.units.fed_batch import (
    FedBatchParams, FedBatchReactor, SemiBatchReactor, optimal_feed_profile,
)


def rate_fn(C, T, p):
    return jnp.array([p["k"] * C["A"]])


def _params(k=0.05):
    return FedBatchParams(V0=1.0, rate_fn=rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
                          rate_params={"k": jnp.array(k)}, species_order=["A", "B"])


def test_fed_batch_reactor():
    n = 20
    _, info = FedBatchReactor(_params(), mode="isothermal")(
        C0={"A": 100.0, "B": 0.0}, T0=350.0, P=101325.0, t_final=100.0,
        feed_rate_fn=lambda t: jnp.array(0.01), feed_composition={"A": 500.0, "B": 0.0},
        n_steps=n)
    assert len(info["t"]) == len(info["V"]) == len(info["C"]["A"]) == n + 1
    assert bool(jnp.isfinite(info["V_final"])) and float(info["n_final"]["B"]) > 0
    assert isinstance(SemiBatchReactor(_params()), FedBatchReactor)


def test_grad_is_finite():
    def nB(k):
        p = _params().update(rate_params={"k": k})
        return FedBatchReactor(p, mode="isothermal")(
            C0={"A": 100.0, "B": 0.0}, T0=350.0, P=101325.0, t_final=10.0,
            n_steps=10)[1]["n_final"]["B"]
    assert bool(jnp.isfinite(jax.grad(nB)(jnp.array(0.05))))


def test_optimal_feed_profile():
    feed_fn, t_opt = optimal_feed_profile(
        "max_yield", _params(), {"A": 100.0, "B": 0.0}, jnp.array(350.0), "B",
        {"A": 500.0, "B": 0.0}, 2.0, 200.0, n_intervals=3, n_sim_steps=20)
    assert callable(feed_fn) and float(feed_fn(jnp.array(10.0))) >= 0.0
    assert t_opt == pytest.approx(200.0)


def test_in_a_flowsheet_with_no_inlets():
    """C0 is a call parameter, not an inlet."""
    for cls in (FedBatchReactor, SemiBatchReactor):
        assert "C0" not in describe_class(cls).ports.inlets
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_unit(Unit("fb", FedBatchReactor(_params(0.1)), [], ["out"], {
        "C0": {"A": 1000.0, "B": 0.0}, "T0": 350.0, "P": 101325.0, "t_final": 10.0}))
    out = fs.solve()["out"]
    assert bool(jnp.isfinite(out["F_A"])) and float(out["F_B"]) > 0
