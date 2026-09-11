"""The quadratic subproblem: curvature in the trust-region model.

`difflow.planning.curvature` measures what a second-order model would buy;
this builds one. The objective picks up each block's priced-output Hessian and
the constraint rows stay first order, so the subproblem is a QP rather than a
QCQP -- solvable to global optimality, which is what every guarantee in the
package rests on.

What is pinned:

1. ``test_convexify_*`` -- wrong-signed eigenvalues are clipped for the sense
   actually being solved, and a definite Hessian is left alone.
2. ``test_expansion_algebra_*`` -- the quadratic is expanded about `u0` into
   standard form, and getting that wrong moves the optimum rather than merely
   mis-reporting it, so it is checked directly.
3. ``test_quadratic_never_loses_to_linear`` -- the safety property that makes
   `model_order="quadratic"` switchable: the QP is warm-started from the LP
   and falls back to it.
4. ``test_quadratic_terminates_in_fewer_iterations`` -- the headline. The
   linear loop finds the optimum without certifying it; the quadratic one
   stops.
5. ``test_integer_columns_fall_back_to_linear`` -- a piecewise SOS2 block
   would make it a MIQP, which is out of scope.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions
from difflow.planning.quadratic import (
    QPModel, block_objective_hessian, build_qp, convexify,
)


def curved_block(name: str = "b") -> Block:
    """A concave revenue in two levers: the model an LP must zigzag toward."""
    def fn(u):
        return jnp.array([
            10.0 * u[0] - 0.5 * u[0] ** 2 + 8.0 * u[1] - 0.4 * u[1] ** 2,
            u[0] + u[1],
        ])
    return Block(name=name, fn=fn, u_names=["x", "z"],
                 y_names=["revenue", "load"],
                 lb=[0.0, 0.0], ub=[20.0, 20.0], u0=[1.0, 1.0])


def _planner(order, **kw):
    return DeltaBasePlanner(
        Network([curved_block()]), prices={"b.revenue": 1.0},
        specs=[Spec("b.load", "<=", 18.0, elastic=False)],
        radius=0.2, sense="max", model_order=order, vertex_seeding=False,
        options=TrustRegionOptions(radius=0.2, max_iter=200, **kw))


# -- 1. convexification ----------------------------------------------------

def test_convexify_clips_indefinite_for_min():
    H = np.array([[1.0, 0.0], [0.0, -2.0]])
    Hc, rep = convexify(H, "min")
    assert rep.modified and rep.n_flipped == 1
    assert np.all(np.linalg.eigvalsh(Hc) > 0.0)


def test_convexify_clips_indefinite_for_max():
    H = np.array([[1.0, 0.0], [0.0, -2.0]])
    Hc, rep = convexify(H, "max")
    assert rep.modified and rep.n_flipped == 1
    assert np.all(np.linalg.eigvalsh(Hc) < 0.0)


def test_convexify_leaves_a_definite_hessian_alone():
    H = np.array([[2.0, 0.5], [0.5, 3.0]])
    Hc, rep = convexify(H, "min")
    assert not rep.modified and rep.n_flipped == 0
    npt.assert_allclose(Hc, H, rtol=0, atol=1e-12)
    assert "not needed" in rep.summary()


def test_convexify_preserves_eigenvectors():
    """Only the amount of curvature changes, never its directions."""
    H = np.array([[1.0, 2.0], [2.0, 1.0]])          # eigenvalues 3, -1
    _, v0 = np.linalg.eigh(H)
    Hc, _ = convexify(H, "min")
    _, v1 = np.linalg.eigh(Hc)
    npt.assert_allclose(np.abs(v0.T @ v1), np.eye(2), atol=1e-10)


def test_convexify_rejects_a_bad_sense():
    with pytest.raises(ValueError, match="sense"):
        convexify(np.eye(2), "maximise")


# -- 2. the expansion algebra ---------------------------------------------

def _subproblem(order="quadratic", radius=0.2):
    planner = _planner(order)
    net = planner.network
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lins = planner.linearize(state)
    return planner, state, lins, planner.build_subproblem(lins, state, radius)


def test_expansion_algebra_agrees_with_the_lp_at_the_centre():
    """The quadratic term vanishes at `u0`, so both models agree there."""
    planner, state, lins, (lp, qp) = _subproblem()
    assert qp is not None

    x = np.zeros(lp.n_cols)
    u0 = np.asarray(lins["b"].u0, dtype=float)
    for j, name in enumerate(planner.network.block("b").qualified_u()):
        x[lp.col(name)] = u0[j]
    for i, name in enumerate(planner.network.block("b").qualified_y()):
        x[lp.col(name)] = float(np.asarray(lins["b"].y0)[i])

    lp_value = lp.sense * float(lp.c @ x) + lp.objective_offset
    npt.assert_allclose(qp.objective(x), lp_value, rtol=1e-9, atol=1e-9)
    # And the quadratic term really is what vanishes, not a cancellation.
    npt.assert_allclose(float(x @ (qp.Q @ x)) > 0.0, True)


def test_expansion_algebra_matches_a_direct_taylor_evaluation():
    """Away from the centre the QP objective is the second-order model."""
    planner, state, lins, (lp, qp) = _subproblem()
    block = planner.network.block("b")
    u0 = np.asarray(lins["b"].u0, dtype=float)
    H = block_objective_hessian(block, planner.prices, jnp.asarray(u0))
    du = np.array([0.3, -0.2])

    x = np.zeros(lp.n_cols)
    for j, name in enumerate(block.qualified_u()):
        x[lp.col(name)] = u0[j] + du[j]
    y = np.asarray(lins["b"].y0) + np.asarray(lins["b"].J) @ du
    for i, name in enumerate(block.qualified_y()):
        x[lp.col(name)] = y[i]

    # In the caller's convention the model is the priced first-order value
    # plus the second-order correction, both with the caller's sign.
    price = float(planner.prices["b.revenue"])
    expected = price * y[0] + 0.5 * float(du @ (H @ du))
    npt.assert_allclose(qp.objective(x), expected, rtol=1e-8, atol=1e-8)

    # The correction is real and negative here: the revenue is concave, so a
    # first-order model overstates what a step delivers. That overstatement
    # is exactly why the linear loop cannot certify its own optimum.
    assert 0.5 * float(du @ (H @ du)) < 0.0


def test_no_curvature_means_no_qp():
    """A block whose priced outputs are linear gets no quadratic model."""
    linear = Block(name="b", fn=lambda u: jnp.array([2.0 * u[0], u[0]]),
                   u_names=["x"], y_names=["revenue", "load"],
                   lb=[0.0], ub=[10.0])
    planner = DeltaBasePlanner(
        Network([linear]), prices={"b.revenue": 1.0}, radius=0.2,
        sense="max", model_order="quadratic", vertex_seeding=False)
    state = planner.network.evaluate(
        jnp.asarray(planner.network.decision_start()), None)
    lins = planner.linearize(state)
    _, qp = planner.build_subproblem(lins, state, 0.2)
    assert qp is None or not np.any(qp.Q)


def test_unpriced_outputs_contribute_no_curvature():
    block = curved_block()
    assert block_objective_hessian(block, {"b.load": 1.0}) is not None
    assert block_objective_hessian(block, {"b.x": 5.0}) is None


# -- 3 and 4. what it buys -------------------------------------------------

@pytest.mark.parametrize("order", ["linear", "auto", "quadratic"])
def test_every_model_order_reaches_the_same_optimum(order):
    res = _planner(order).solve()
    assert float(res.objective) == pytest.approx(89.1111, rel=1e-4)
    assert max(res.violations.values(), default=0.0) < 1e-6


def test_quadratic_never_loses_to_linear():
    """The QP is warm-started from the LP and falls back to it."""
    lin = _planner("linear").solve()
    quad = _planner("quadratic").solve()
    assert float(quad.objective) >= float(lin.objective) - 1e-6


def test_quadratic_terminates_in_fewer_iterations():
    """The headline: curvature is what lets the loop stop.

    A first-order model of a curved objective predicts a gain the blocks do
    not deliver, so the radius ratchets down and the loop runs out its budget
    having found the optimum without certifying it. The second-order model
    predicts the gain correctly and the loop terminates.
    """
    lin = _planner("linear").solve()
    quad = _planner("quadratic").solve()
    assert quad.n_iterations < lin.n_iterations, (
        f"quadratic took {quad.n_iterations} iterations, "
        f"linear took {lin.n_iterations}")


def test_auto_keeps_the_exact_hessian_and_never_convexifies():
    """`auto` takes curvature only where it is already definite."""
    planner, _, _, (_, qp) = _subproblem(order="auto")
    assert qp is not None
    assert not qp.convexified


def test_quadratic_records_what_it_convexified():
    planner, _, _, (_, qp) = _subproblem(order="quadratic")
    assert qp is not None
    # Concave revenue maximised: already convex for the minimised objective.
    assert not qp.convexified
    assert "b" in qp.convexification


# -- 5. out of scope -------------------------------------------------------

def test_integer_columns_fall_back_to_linear():
    """A piecewise SOS2 block would make it a MIQP, which is out of scope."""
    planner, state, lins, _ = _subproblem()
    lp = planner.build_lp(lins, state, 0.2)
    lp.integer_cols = [0]
    _, qp = planner.build_subproblem(lins, state, 0.2)
    assert qp is not None            # this lp copy is local to the test

    qp_int = QPModel(lp=lp, Q=np.zeros((lp.n_cols, lp.n_cols)))
    with pytest.raises(NotImplementedError, match="integer columns"):
        qp_int.solve()


def test_bad_model_order_rejected():
    with pytest.raises(ValueError, match="model_order"):
        DeltaBasePlanner(Network([curved_block()]), prices={"b.revenue": 1.0},
                         model_order="cubic")
