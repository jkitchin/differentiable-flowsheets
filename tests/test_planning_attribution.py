"""Smoke, API and validation tests for difflow.planning.attribution."""

import jax.numpy as jnp
import numpy as np
import pytest

from difflow.planning import Block, attribute_deltas

MOVE = [1.0, 1.0, 1.0]


def model_fn(u):
    return {"yield": 2.0 * u[0] + 1.0 * u[1] + 0.5 * u[2],
            "purity": 0.9 * jnp.exp(-0.05 * u[0])}


def make_block():
    return Block(name="b", fn=model_fn, u_names=["x0", "x1", "x2"],
                 y_names=["yield", "purity"],
                 lb=[0.0, 0.0, 0.0], ub=[10.0, 10.0, 10.0])


def history(seed, n=50):
    rng = np.random.default_rng(seed)
    u = np.column_stack([rng.uniform(2, 8, n), rng.uniform(2, 8, n), np.full(n, 5.0)])
    return u, np.arange(n, dtype=float)


def test_attribute_deltas_smoke():
    blk = make_block()
    u, t = history(0)
    rng = np.random.default_rng(1)
    y = 2.0 * u[:, 0] + u[:, 1] + 0.5 * u[:, 2] + 0.8 + rng.normal(0, 0.05, len(t))
    res = attribute_deltas(blk, u, {"yield": y}, sigma_y=0.05, t=t, move=MOVE)
    assert res.unobserved == ["purity"]
    # x2 never moved, so it is reported as not estimable rather than guessed.
    x2 = res.outputs["yield"].coefficient("x2")
    assert x2.status == "not estimable" and np.isnan(x2.estimate)
    assert isinstance(res.table(), str)
    mods = res.to_modifiers()
    assert np.isfinite(float(mods.eps[blk.y_index("yield")]))


def test_log_outputs_smoke():
    blk = make_block()
    u, t = history(2)
    purity = 0.9 * np.exp(-0.05 * u[:, 0]) * 1.1
    res = attribute_deltas(blk, u, {"purity": purity}, sigma_y=0.005, t=t,
                           move=MOVE, log_outputs={"purity": 0.0})
    out = res.to_modifiers().correct(blk, jnp.asarray(res.u_ref))
    assert np.all(np.isfinite(np.asarray(out)))


def test_input_validation():
    blk = make_block()
    u, t = history(16, n=10)
    with pytest.raises(KeyError):
        attribute_deltas(blk, u, {"nope": np.zeros(10)}, sigma_y=1.0)
    with pytest.raises(ValueError):
        attribute_deltas(blk, u[:, :2], {"yield": np.zeros(10)}, sigma_y=1.0)
    unbounded = Block(name="c", fn=lambda x: {"y": x[0]}, u_names=["x"], y_names=["y"])
    with pytest.raises(ValueError, match="move"):
        attribute_deltas(unbounded, u[:, :1], {"y": np.zeros(10)}, sigma_y=1.0)
