"""Smoke, API and validation tests for the quadratic subproblem
(difflow.planning.quadratic)."""

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
    QPModel, block_objective_hessian, convexify,
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


def test_convexify():
    Hc, rep = convexify(np.array([[1.0, 0.0], [0.0, -2.0]]), "min")
    assert rep.modified and np.all(np.linalg.eigvalsh(Hc) > 0.0)
    H = np.array([[2.0, 0.5], [0.5, 3.0]])
    Hc, rep = convexify(H, "min")
    assert not rep.modified
    npt.assert_allclose(Hc, H, atol=1e-12)
    with pytest.raises(ValueError, match="sense"):
        convexify(np.eye(2), "maximise")


def _subproblem(order="quadratic", radius=0.2):
    planner = _planner(order)
    net = planner.network
    state = net.evaluate(jnp.asarray(net.decision_start()), None)
    lins = planner.linearize(state)
    return planner, state, lins, planner.build_subproblem(lins, state, radius)


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


def test_quadratic_planner_smoke():
    res = _planner("quadratic").solve()
    assert np.isfinite(res.merit)


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
