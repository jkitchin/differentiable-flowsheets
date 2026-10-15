"""Smoke tests for chaining units into one differentiable function (#334)."""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.plant import AD_MODES, Chain, Stage, ad_mode_table

jax.config.update("jax_enable_x64", True)


@jax.custom_vjp
def _rev_only(x):
    """``y = [x0 * x1, sin(x0)]``; reverse mode only."""
    return jnp.stack([x[0] * x[1], jnp.sin(x[0])])


def _rev_fwd(x):
    return _rev_only(x), x


def _rev_bwd(x, g):
    return (jnp.stack([g[0] * x[1] + g[1] * jnp.cos(x[0]), g[0] * x[0]]),)


_rev_only.defvjp(_rev_fwd, _rev_bwd)


def _fwd_only(y):
    """``z = sqrt(y0 + y1^2)`` by a Newton ``while_loop`` (forward mode only)."""
    a = y[0] + y[1] ** 2

    def body(c):
        z, k = c
        return 0.5 * (z + a / z), k + 1

    z, _ = jax.lax.while_loop(lambda c: c[1] < 40, body, (jnp.asarray(1.0) + 0.0 * a, 0))
    return {"z": z, "y0": y[0]}


X = jnp.asarray([0.7, 1.3])


@pytest.fixture
def mixed():
    return Chain(Stage("hdt-like", _rev_only, modes="rev"), Stage("reformer-like", _fwd_only, modes="fwd"))


def test_mixed_chain_jacobian_by_the_chain_rule(mixed):
    res = mixed.jacobian(X)
    assert res.method == "chain"
    assert np.all(np.isfinite(np.asarray(res.jacobian)))
    with pytest.raises(ValueError, match="hdt-like"):
        mixed.jacobian(X, method="fwd")


def test_bad_stages_and_methods_say_what_is_wrong():
    with pytest.raises(ValueError, match="subset"):
        Stage("x", lambda v: v, modes=("sideways",))
    with pytest.raises(ValueError, match="unique"):
        Chain(Stage("x", lambda v: v), Stage("x", lambda v: v))
    with pytest.raises(ValueError, match="unknown method"):
        Chain(Stage("x", lambda v: v)).jacobian(X, method="adjoint")


def _resolve(path: str):
    obj = importlib.import_module("difflow_refinery")
    for part in path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            obj = importlib.import_module(f"{obj.__name__}.{part}")
    return obj


def test_the_ad_mode_table_names_real_objects():
    for m in AD_MODES.values():
        for part in m.obj.split(","):
            for path in part.split("/"):
                _resolve(path.strip())
    assert "| catalytic reformer |" in ad_mode_table()
