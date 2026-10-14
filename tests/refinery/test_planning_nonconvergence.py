"""A non-converged crude unit is NaN, and a plan never lands there (#297).

The rejection test runs the planner into the region where the column fails
and checks it backs off; the same column with the mask off shows what the
planner would otherwise have taken -- a column that did not converge, scored
as a plan.

These were ``TestNonConvergence`` in test_planning.py, on the 30-stage
column and four planner runs (about ten minutes). The edge is the same one on
the 8-stage column of test_planning.py: a pumparound that takes more heat
than the feed brings leaves no reflux, and the column does not converge.
"""

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from difflow.planning import Block, DeltaBasePlanner, Network, state_is_finite
from difflow.planning.planner import TrustRegionOptions
from difflow_refinery.planning import cdu_block

from .test_planning import BPD, P_IN, T_IN, build_unit

jax.config.update("jax_enable_x64", True)

#: pa1.duty (MW): the base converges, FAIL_MW does not (the column is dry).
BASE_MW = 0.1
FAIL_MW = 0.3
BOUNDS = {"pa1.duty": (0.05, 0.5)}


@pytest.fixture(scope="module")
def unit():
    return build_unit()


@pytest.fixture(scope="module")
def masked(unit):
    return cdu_block(unit, ["pa1.duty"], ["naphtha.tbp95", "furnace.fired"], rate=BPD, T=T_IN, P=P_IN,
                     bounds=BOUNDS)


class TestNonConvergence:
    def test_a_failed_solve_is_nan(self, masked):
        u = np.array([FAIL_MW])
        assert not bool(masked.fn.solve(u).converged)
        assert np.all(np.isnan(np.asarray(masked.fn(u))))
        assert np.all(np.isfinite(np.asarray(masked.fn(np.array([BASE_MW])))))

    def test_a_non_converging_proposal_is_rejected(self):
        """The planner side of it, on a block that is NaN past an edge (the masked
        column's behaviour above) and cheap to evaluate: the planner proposes past
        the edge, rejects it, and lands on a point inside it. Running it on the
        column itself costs a failing Newton solve per rejected proposal, about
        fifteen minutes."""
        EDGE = 0.24

        def fn(u):
            y = jnp.stack([jnp.sqrt(u[0]) * 10.0, u[0] ** 2])
            return jnp.where(u[0] > EDGE, jnp.nan, y)

        blk = Block(name="cdu", fn=fn, u_names=["pa1.duty"], y_names=["a", "b"],
                    lb=[0.05], ub=[0.5], u0=[0.1], jit=False)
        planner = DeltaBasePlanner(Network([blk]), prices={"cdu.pa1.duty": 1000.0},
                                   options=TrustRegionOptions(radius=0.3, radius_min=1e-2, tol=1e-6,
                                                              max_iter=30),
                                   vertex_seeding=False)
        res = planner.solve()
        failed = [h for h in res.history if "not evaluable" in h.lp_status]
        assert failed, "the plan never proposed past the edge; the test proves nothing"
        assert not any(h.accepted for h in failed)
        for h in res.history:
            if h.accepted:
                assert float(h.decisions[0]) <= EDGE
        duty = res.plan["cdu.pa1.duty"]
        assert 0.1 < duty <= EDGE
        assert state_is_finite(res.state)
