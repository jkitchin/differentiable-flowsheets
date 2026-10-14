"""Tests for difflow.planning.curvature — second-order models by HVP.

The claims under test are the ones the module is built to support:

1. ``test_hvp_matches_exact_hessian`` / ``test_hessian_matches_central_
   differences`` — the curvature is right at all.
2. ``test_quadratic_is_exact_on_a_quadratic`` — the second-order model
   reproduces a quadratic response exactly, where the first-order one does not.
3. ``test_indefinite_curvature_is_not_recommended`` — a well-fitting quadratic
   whose Hessian is indefinite is refused, because a nonconvex QP subproblem
   forfeits the global optimality every downstream guarantee rests on. This is
   the regression that keeps the module honest.
4. ``test_hessian_costs_order_n_not_n_squared`` — the cost claim.
5. ``test_definiteness_is_a_property_of_the_point`` — the reason the check
   cannot be done once and cached.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block
from difflow.planning.curvature import (
    Curvature, block_curvature, block_hvp, check_model_order,
    classify_definiteness, hessian_of, hvp, scalar_output,
)


# -- fixtures --------------------------------------------------------------

def quadratic_block() -> Block:
    """y = x^2 + 0.5 z^2 (PSD), and x*z (a saddle: indefinite)."""
    return Block(
        name="q",
        fn=lambda u: jnp.array([u[0] ** 2 + 0.5 * u[1] ** 2, u[0] * u[1]]),
        u_names=["x", "z"], y_names=["bowl", "saddle"],
        lb=[0.0, 0.0], ub=[2.0, 2.0], u0=[1.0, 1.0])


def cubic_block() -> Block:
    """A response the quadratic model improves but does not nail."""
    return Block(name="c", fn=lambda u: jnp.array([u[0] ** 3]),
                 u_names=["x"], y_names=["y"], lb=[0.5], ub=[2.5], u0=[1.5])


# -- 1. the curvature is correct ------------------------------------------

def test_hvp_matches_exact_hessian():
    blk = quadratic_block()
    H = np.asarray(hessian_of(blk, "bowl"))
    for v in (np.array([1.0, 0.0]), np.array([0.0, 1.0]),
              np.array([0.7, -0.3])):
        got = np.asarray(block_hvp(blk, "bowl", jnp.asarray(v)))
        npt.assert_allclose(got, H @ v, rtol=1e-10, atol=1e-12)


def test_hvp_on_a_plain_callable():
    f = lambda x: jnp.sum(x ** 3)
    u = jnp.array([1.0, 2.0])
    npt.assert_allclose(np.asarray(hvp(f, u, jnp.array([1.0, 0.0]))),
                        [6.0, 0.0], rtol=1e-10)


def test_hessian_matches_central_differences():
    """The AD Hessian against the 2n^2 method it replaces."""
    blk = cubic_block()
    u0 = np.array([1.5])
    h = 1e-5
    f = scalar_output(blk, "y")
    fd = (float(f(jnp.asarray(u0 + h))) - 2 * float(f(jnp.asarray(u0)))
          + float(f(jnp.asarray(u0 - h)))) / h ** 2
    npt.assert_allclose(float(np.asarray(hessian_of(blk, "y"))[0, 0]), fd,
                        rtol=1e-5)


def test_hessian_is_symmetric():
    H = np.asarray(block_curvature(quadratic_block(), "saddle").H)
    npt.assert_allclose(H, H.T, rtol=0, atol=0)


# -- 2. the second-order model is better ----------------------------------

def test_quadratic_is_exact_on_a_quadratic():
    rep = check_model_order(quadratic_block(), "bowl")
    assert rep.quadratic_is_exact
    assert rep.max_linear_error > 1e-3      # the linear model is not
    assert rep.improvement == float("inf")
    assert rep.recommended == "quadratic"


def test_quadratic_helps_but_is_not_exact_on_a_cubic():
    rep = check_model_order(cubic_block(), "y", radius=0.5)
    assert not rep.quadratic_is_exact
    assert rep.max_quadratic_error < rep.max_linear_error
    assert rep.improvement > 1.0


def test_model_error_grows_along_the_step():
    """Both models are local; the report must show that, not hide it."""
    rep = check_model_order(cubic_block(), "y", radius=0.5, n_points=5)
    err = np.abs(rep.linear_error)
    assert np.all(np.diff(err) > 0), "linear error should grow with the step"


# -- 3. the refusal that keeps the guarantees ------------------------------

def test_indefinite_curvature_is_not_recommended():
    """A perfect fit is still refused when the Hessian is indefinite.

    The saddle output is reproduced exactly by its quadratic model, so a
    recommendation made on accuracy alone would say "quadratic". It must not:
    an indefinite Hessian makes the subproblem a nonconvex QP, which is
    solved locally rather than globally and whose duals stop reading as
    prices -- and every guarantee in difflow.planning chains off exactly
    those two properties.
    """
    rep = check_model_order(quadratic_block(), "saddle")
    assert rep.curvature.definiteness == "indefinite"
    assert rep.quadratic_is_exact          # it fits perfectly, and still
    assert rep.recommended == "linear"     # is not recommended
    assert rep.caveat is not None
    assert "nonconvex" in rep.caveat


def test_sense_decides_which_definiteness_is_convex():
    psd = block_curvature(quadratic_block(), "bowl")
    assert psd.definiteness == "psd"
    assert psd.convex_for("min")
    assert not psd.convex_for("max")


def test_psd_model_is_refused_when_maximising():
    rep = check_model_order(quadratic_block(), "bowl", sense="max")
    assert rep.recommended == "linear"
    assert rep.caveat is not None


def test_a_linear_block_has_zero_curvature_and_is_convex_either_way():
    blk = Block(name="lin", fn=lambda u: jnp.array([2.0 * u[0] + 3.0]),
                u_names=["x"], y_names=["y"], lb=[0.0], ub=[1.0])
    curv = block_curvature(blk, "y")
    assert curv.definiteness == "zero"
    assert curv.convex_for("min") and curv.convex_for("max")
    rep = check_model_order(blk, "y")
    assert rep.recommended == "linear"     # nothing to gain
    assert rep.caveat is None


def test_bad_sense_rejected():
    with pytest.raises(ValueError, match="sense"):
        check_model_order(quadratic_block(), "bowl", sense="maximise")


# -- 4. the cost claim -----------------------------------------------------

def test_hessian_costs_order_n_not_n_squared():
    """The exact Hessian is O(n) evaluations, so it scales like a Jacobian.

    Counted rather than timed: a wall-clock test is flaky in CI, and the
    claim is about how many times the model is traced, not how fast it runs.
    """
    calls = {"n": 0}

    def counted(u):
        calls["n"] += 1
        return jnp.array([jnp.sum(u ** 2) + 0.1 * jnp.sum(u) ** 2])

    for n in (4, 16):
        calls["n"] = 0
        blk = Block(name="b", fn=counted, u_names=[f"u{i}" for i in range(n)],
                    y_names=["y"], lb=[0.0] * n, ub=[1.0] * n)
        hessian_of(blk, "y")
        # Tracing happens a small fixed number of times regardless of n --
        # JAX builds the whole Hessian from one traced graph.
        assert calls["n"] <= 4, f"n={n} traced {calls['n']} times"


def test_hessian_of_a_weighted_objective():
    """Curvature of the priced objective, not of a single output."""
    blk = quadratic_block()
    H = np.asarray(hessian_of(blk, {"bowl": 1.0, "saddle": 2.0}))
    H_expected = (np.asarray(hessian_of(blk, "bowl"))
                  + 2.0 * np.asarray(hessian_of(blk, "saddle")))
    npt.assert_allclose(H, H_expected, rtol=1e-10, atol=1e-12)


# -- 5. definiteness depends on where you are ------------------------------

def test_definiteness_is_a_property_of_the_point():
    """Why the check cannot be done once and cached.

    y = x^3 curves upward for x > 0 and downward for x < 0, so the same
    block is convex at one operating point and concave at another. A planner
    that decided the model order once at commissioning would be wrong half
    the time.
    """
    blk = Block(name="s", fn=lambda u: jnp.array([u[0] ** 3]),
                u_names=["x"], y_names=["y"], lb=[-2.0], ub=[2.0])
    assert block_curvature(blk, "y", u0=jnp.array([1.0])).definiteness == "psd"
    assert block_curvature(blk, "y", u0=jnp.array([-1.0])).definiteness == "nsd"


def test_classify_definiteness_tolerance_is_relative():
    """A weakly curved model has genuine zero eigenvalues.

    An absolute cut-off would call a well-conditioned but small Hessian
    indefinite purely from rounding, which is the same trap the IPM inertia
    test in difflow_power avoids by equilibrating first.
    """
    assert classify_definiteness(np.array([1e-14, 2e-14])) == "psd"
    assert classify_definiteness(np.array([-1e-12, 1.0])) == "psd"
    assert classify_definiteness(np.array([-1.0, 1.0])) == "indefinite"
    assert classify_definiteness(np.array([0.0, 0.0])) == "zero"
    assert classify_definiteness(np.array([-2.0, -1.0])) == "nsd"


# -- reporting -------------------------------------------------------------

def test_summary_mentions_the_recommendation_and_the_caveat():
    text = check_model_order(quadratic_block(), "saddle").summary()
    assert "recommended: linear" in text
    assert "caveat" in text
    assert "indefinite" in text


def test_unknown_output_rejected():
    with pytest.raises((KeyError, ValueError)):
        block_curvature(quadratic_block(), "not_an_output")


def test_direction_length_validated():
    with pytest.raises(ValueError, match="direction"):
        check_model_order(quadratic_block(), "bowl",
                          direction=np.array([1.0, 2.0, 3.0]))
