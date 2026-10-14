"""Smoke, decision-logic and validation tests for difflow.planning.curvature."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block
from difflow.planning.curvature import (
    block_curvature, block_hvp, check_model_order, classify_definiteness,
    hessian_of, hvp,
)


def quadratic_block() -> Block:
    """y = x^2 + 0.5 z^2 (PSD), and x*z (a saddle: indefinite)."""
    return Block(
        name="q",
        fn=lambda u: jnp.array([u[0] ** 2 + 0.5 * u[1] ** 2, u[0] * u[1]]),
        u_names=["x", "z"], y_names=["bowl", "saddle"],
        lb=[0.0, 0.0], ub=[2.0, 2.0], u0=[1.0, 1.0])


def test_hvp_and_hessian_smoke():
    assert hvp(lambda x: jnp.sum(x ** 3), jnp.array([1.0, 2.0]), jnp.array([1.0, 0.0])).shape == (2,)
    blk = quadratic_block()
    assert block_hvp(blk, "bowl", jnp.array([1.0, 0.0])).shape == (2,)
    assert hessian_of(blk, {"bowl": 1.0, "saddle": 2.0}).shape == (2, 2)


def test_model_order_recommendation():
    blk = quadratic_block()
    assert check_model_order(blk, "bowl").recommended == "quadratic"
    # An indefinite Hessian is refused even though the quadratic fits exactly.
    rep = check_model_order(blk, "saddle")
    assert rep.recommended == "linear" and "nonconvex" in rep.caveat
    assert "recommended: linear" in rep.summary()
    # A PSD model is refused when maximising.
    assert check_model_order(blk, "bowl", sense="max").recommended == "linear"


def test_classify_definiteness():
    assert block_curvature(quadratic_block(), "bowl").definiteness == "psd"
    assert classify_definiteness(np.array([1e-14, 2e-14])) == "psd"
    assert classify_definiteness(np.array([-1.0, 1.0])) == "indefinite"
    assert classify_definiteness(np.array([0.0, 0.0])) == "zero"
    assert classify_definiteness(np.array([-2.0, -1.0])) == "nsd"


def test_validation():
    with pytest.raises(ValueError, match="sense"):
        check_model_order(quadratic_block(), "bowl", sense="maximise")
    with pytest.raises((KeyError, ValueError)):
        block_curvature(quadratic_block(), "not_an_output")
    with pytest.raises(ValueError, match="direction"):
        check_model_order(quadratic_block(), "bowl", direction=np.array([1.0, 2.0, 3.0]))
