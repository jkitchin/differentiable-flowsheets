"""Smoke tests for batch distillation (#426)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.staged import constant_alpha, tabulated
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.batch_distillation import (
    BatchDistillation, BatchDistillationParams, batch_cuts, rayleigh,
    rayleigh_multicomponent,
)

jax.config.update("jax_enable_x64", True)

ALPHA = 2.5


def charge(a=50.0, b=50.0):
    return {"F_a": a, "F_b": b, "T": 350.0, "P": 1e5}


def make(mode="constant_reflux", **kw):
    base = dict(species_order=["a", "b"], n_stages=2, R=3.0, boilup=2.0, mode=mode)
    base.update(kw)
    return BatchDistillation(BatchDistillationParams(**base), curve=constant_alpha(ALPHA))


def test_rayleigh_smoke_and_grad():
    r = rayleigh(constant_alpha(2.5), 0.5, x_W=0.2)
    assert bool(r["ok"]) and np.isfinite(float(r["W_over_F"]))
    g = jax.grad(lambda a: rayleigh(constant_alpha(a), 0.5, W_over_F=0.4)["x_W"])(2.5)
    assert np.isfinite(float(g))


def test_rayleigh_exactly_one_spec():
    with pytest.raises(ValueError):
        rayleigh(constant_alpha(2.0), 0.5)
    with pytest.raises(ValueError):
        rayleigh(constant_alpha(2.0), 0.5, W_over_F=0.5, x_W=0.3)


def test_rayleigh_flag_not_nan_past_an_azeotrope():
    x = np.linspace(0, 1, 101)
    y = x + 0.4 * x * (1 - x) * (0.5 - x) * 4
    r = rayleigh(tabulated(x, np.maximum.accumulate(y)), 0.8, x_W=0.2)
    assert not bool(r["ok"])
    assert np.isfinite(float(r["x_D_avg"]))


@pytest.mark.parametrize("mode,kw", [("constant_reflux", {}),
                                     ("constant_composition", {"x_D_spec": 0.9})])
def test_batch_distillation_balance(mode, kw):
    still, dist, info = make(mode, **kw)(charge(40.0, 60.0), stop={"x_W": 0.25}, n_traj=3)
    assert float(still["F_a"] + dist["F_a"]) == pytest.approx(40.0, abs=1e-8)
    assert float(still["F_b"] + dist["F_b"]) == pytest.approx(60.0, abs=1e-8)
    assert np.isfinite(float(info["t_batch"]))


def test_batch_distillation_jit():
    u = make(R=2.0)
    f = jax.jit(lambda ch: u(ch, stop={"x_W": 0.2}, n_traj=3)[2]["t_batch"])
    assert np.isfinite(float(f(charge())))


def test_batch_distillation_validation():
    with pytest.raises(NotImplementedError):
        BatchDistillationParams(species_order=["a", "b"], holdup=1.0)
    with pytest.raises(ValueError):
        BatchDistillationParams(species_order=["a", "b", "c"])
    with pytest.raises(ValueError):
        make()(charge())
    with pytest.raises(ValueError):
        make()(charge(), stop={"bogus": 1.0})
    with pytest.raises(ValueError):
        BatchDistillation(BatchDistillationParams(species_order=["a", "b"]))


def test_batch_cuts_smoke():
    res = batch_cuts(make(), charge(), [{"distillate": 20.0}, {"x_W": 0.2}])
    assert len(res["cuts"]) == 2
    for v in res["balance"].values():
        assert abs(float(v)) < 1e-8


def test_curve_is_not_shared_between_units():
    """The jitted body takes the curve as an argument (it is not in the value key)."""
    p = BatchDistillationParams(["a", "b"], n_stages=0, R=0.0)
    out = [BatchDistillation(p, curve=constant_alpha(a))(charge(), stop={"distillate": 40.0})[2]
           for a in (2.0, 4.0)]
    assert float(out[0]["x_D_avg"]) != float(out[1]["x_D_avg"])


def test_rayleigh_multicomponent_smoke():
    def sd(name, b):
        return SpeciesData(name=name, MW=50.0, Cp_coeffs=(80.0, 0, 0, 0),
                           Hvap_coeffs=(3.0e4, 0.38, 520.0), antoine_coeffs=(9.0, b, -50.0))
    thermo = IdealThermo({"A": sd("A", 1100.0), "B": sd("B", 1250.0)})
    r = rayleigh_multicomponent(thermo, ("A", "B"), jnp.array([0.4, 0.6]), 0.5, 1e5, n_steps=10)
    assert float(jnp.sum(r["x_W"])) == pytest.approx(1.0, abs=1e-9)
