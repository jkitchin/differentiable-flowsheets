"""A block that cannot be evaluated is never a plan.

A rigorous submodel -- a crude column, a flash, any block with an inner
solve -- has operating points where the solve has no answer. The convention
is that such a block returns NaN, and these tests pin what the planner does
with it: the proposal is rejected, never scored, never linearised.

The arithmetic alone does not get this right, which is why it is written
down rather than left to IEEE:

* a NaN *merit* already fails ``rho >= eta_accept``, but
* a NaN *spec output* scores ZERO violation -- ``max(0.0, nan)`` is ``0.0``
  in Python -- so restoration would take a failed solve for a feasible point,
  and
* a NaN output that is neither priced nor in a spec leaves the merit finite,
  so the main loop would accept the point and then try to linearise at it.

``test_a_nan_spec_output_would_have_read_as_satisfied`` and
``test_restoration_would_have_called_a_failed_solve_feasible`` are the
regressions for the second; ``test_an_unpriced_nan_is_still_rejected`` for
the third.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network, state_is_finite
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions

#: The block converges for x <= EDGE and "fails" beyond it.
EDGE = 1.0


def cliff_block(nan_outputs=("out", "q")):
    """One lever; the named outputs are NaN past ``EDGE``, as a failed solve's are."""

    names = ["out", "q"]

    def fn(u):
        x = u[0]
        fail = x > EDGE
        vals = {"out": x, "q": x}
        return jnp.stack([jnp.where(fail & (n in nan_outputs), jnp.nan, vals[n])
                          for n in names])

    return Block(name="plant", fn=fn, u_names=["x"], y_names=names,
                 lb=[0.0], ub=[2.0], u0=[0.5])


def _planner(block, prices, specs=(), **kw):
    opts = TrustRegionOptions(radius=0.3, radius_min=1e-4, max_iter=60)
    return DeltaBasePlanner(Network([block]), prices=prices, specs=specs,
                            options=opts, vertex_seeding=False, **kw)


class TestScoring:
    def test_state_is_finite(self):
        net = Network([cliff_block()])
        assert state_is_finite(net.evaluate([0.5]))
        assert not state_is_finite(net.evaluate([1.5]))

    def test_a_nan_spec_output_would_have_read_as_satisfied(self):
        """The hole the explicit rule closes: Python's max drops the NaN."""
        spec = Spec("plant.q", ">=", 1.5)
        assert spec.violation({"plant.q": float("nan")}) == 0.0

        planner = _planner(cliff_block(), {"plant.out": 1.0}, [spec])
        scored = planner.score([1.6])
        assert scored["evaluable"] is False
        assert scored["merit"] == -np.inf
        assert scored["total_violation"] == np.inf

    def test_an_evaluable_point_scores_as_before(self):
        planner = _planner(cliff_block(), {"plant.out": 2.0})
        scored = planner.score([0.75])
        assert scored["evaluable"] is True
        assert scored["merit"] == pytest.approx(1.5)


class TestTheLoop:
    def test_a_nan_proposal_is_rejected(self):
        """The price pushes x past the edge; the plan stops at it, from below."""
        res = _planner(cliff_block(), {"plant.out": 1.0}).solve()
        failed = [h for h in res.history if "not evaluable" in h.lp_status]
        assert failed, "the loop never proposed past the edge; the test proves nothing"
        assert not any(h.accepted for h in failed)
        for h in res.history:
            if h.accepted:
                assert h.decisions[0] <= EDGE
        assert res.plan["plant.x"] <= EDGE
        assert res.plan["plant.x"] == pytest.approx(EDGE, abs=1e-3)
        assert np.isfinite(res.merit)

    def test_an_unpriced_nan_is_still_rejected(self):
        """Merit is finite here -- only ``q`` fails -- and the point is still refused."""
        res = _planner(cliff_block(nan_outputs=("q",)), {"plant.out": 1.0}).solve()
        assert any("not evaluable" in h.lp_status for h in res.history)
        assert res.plan["plant.x"] <= EDGE
        assert state_is_finite(res.state)

    def test_rejected_without_the_accept_test_too(self):
        """``accept_test=False`` takes worse points; it cannot take no point."""
        res = _planner(cliff_block(), {"plant.out": 1.0}, accept_test=False).solve()
        assert any("not evaluable" in h.lp_status for h in res.history)
        assert res.plan["plant.x"] <= EDGE
        assert state_is_finite(res.state)

    def test_restoration_would_have_called_a_failed_solve_feasible(self):
        """An inelastic spec only the failed region "meets" is not met.

        ``q >= 1.5`` cannot be reached where the block converges, and past
        the edge its NaN reads as zero violation. Restoration must keep the
        least-violating *evaluable* point, not jump into the failure.
        """
        spec = Spec("plant.q", ">=", 1.5, elastic=False)
        res = _planner(cliff_block(), {"plant.out": 1.0}, [spec]).solve()
        restoration = [h for h in res.history if h.restoration]
        assert restoration
        for h in restoration:
            if h.accepted:
                assert h.decisions[0] <= EDGE
        assert state_is_finite(res.state)
        assert res.total_violation == pytest.approx(1.5 - res.plan["plant.x"])


class TestStarts:
    def test_an_unevaluable_start_raises(self):
        with pytest.raises(ValueError, match="no starting point could be evaluated"):
            _planner(cliff_block(), {"plant.out": 1.0}).solve(u0=[1.5])

    def test_an_unevaluable_seed_loses_to_an_evaluable_one(self):
        res = _planner(cliff_block(), {"plant.out": 1.0}).solve(seeds=[[1.8]])
        reasons = sorted(a.reason for a in res.attempts)
        assert "start_not_evaluable" in reasons
        bad = next(a for a in res.attempts if a.reason == "start_not_evaluable")
        assert bad.merit == -np.inf and bad.lp_model is None
        assert res.reason != "start_not_evaluable"
        assert res.plan["plant.x"] <= EDGE
