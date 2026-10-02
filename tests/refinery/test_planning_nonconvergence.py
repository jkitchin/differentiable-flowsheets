"""A non-converged crude unit is NaN, and a plan never lands there (#297).

The rejection test runs the planner into the region where the column fails
and checks it backs off; the same run with the mask off shows what the
planner would otherwise have taken -- a column that did not converge, scored
as a plan.

These were ``TestNonConvergence`` in test_planning.py. They are the bulk of
that file's cost -- about ten minutes of it on a CI runner -- and ``pytest
--dist loadfile`` keeps a file on one worker, so together they made a
single-worker tail on whichever shard drew them (#315). The tests share no
compiled code with the rest of that file (each builds its own block), so
splitting them off costs nothing but the ``unit`` fixture, which is cheap.
"""

import jax
import numpy as np
import pytest

from difflow.planning import DeltaBasePlanner, Network, state_is_finite
from difflow.planning.planner import TrustRegionOptions
from difflow_refinery.planning import cdu_block

from .test_planning import BPD, P_IN, T_IN, build_unit

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def unit():
    return build_unit()


class TestNonConvergence:
    #: 5% overflash and 25 MW out of PA1 dries the column (test_furnace.py
    #: shows the same edge at 150 kg/s); 15 MW converges.
    FAIL_MW = 30.0

    def test_a_failed_solve_is_nan(self, unit):
        blk = cdu_block(unit, ["pa1.duty"], ["naphtha.tbp95", "furnace.fired"], rate=BPD, T=T_IN, P=P_IN,
                        bounds={"pa1.duty": (10.0, 30.0)})
        u = np.array([self.FAIL_MW])
        assert not bool(blk.fn.solve(u).converged)
        assert np.all(np.isnan(np.asarray(blk.fn(u))))
        assert np.all(np.isfinite(np.asarray(blk.fn(np.array([15.0])))))

    def test_unmasked_it_would_have_looked_like_an_answer(self, unit):
        blk = cdu_block(unit, ["pa1.duty"], ["naphtha.tbp95"], rate=BPD, T=T_IN, P=P_IN,
                        bounds={"pa1.duty": (10.0, 30.0)}, mask_nonconverged=False)
        assert np.all(np.isfinite(np.asarray(blk.fn(np.array([self.FAIL_MW])))))

    @staticmethod
    def _plan(unit, mask):
        """Heat recovered from PA1 is credited; the credit pushes toward the edge."""
        blk = cdu_block(unit, ["pa1.duty"], ["naphtha.tbp95", "kero.tbp95", "furnace.fired"],
                        rate=BPD, T=T_IN, P=P_IN, bounds={"pa1.duty": (10.0, 30.0)},
                        mask_nonconverged=mask)
        planner = DeltaBasePlanner(Network([blk]), prices={"cdu.pa1.duty": 1000.0},
                                   options=TrustRegionOptions(radius=0.3, radius_min=1e-2, tol=1e-6,
                                                              max_iter=30),
                                   vertex_seeding=False)
        return blk, planner.solve()

    @pytest.mark.slow
    def test_a_non_converging_proposal_is_rejected(self, unit):
        blk, res = self._plan(unit, mask=True)
        failed = [h for h in res.history if "not evaluable" in h.lp_status]
        assert failed, "the plan never proposed past the edge; the test proves nothing"
        assert not any(h.accepted for h in failed)
        assert any(h.decisions[0] > 24.5 for h in failed)
        # every accepted point, and the plan, is a converged column
        for h in res.history:
            if h.accepted:
                assert bool(blk.fn.solve(h.decisions).converged)
        duty = res.plan["cdu.pa1.duty"]
        assert 23.0 < duty < 25.0
        assert bool(blk.fn.solve(np.array([duty])).converged)
        assert state_is_finite(res.state)

    @pytest.mark.slow
    def test_without_the_mask_the_plan_is_a_failed_column(self, unit):
        """The control: the planner takes the bound, where the column did not converge."""
        blk, res = self._plan(unit, mask=False)
        duty = res.plan["cdu.pa1.duty"]
        assert duty == pytest.approx(30.0)
        assert not bool(blk.fn.solve(np.array([duty])).converged)
