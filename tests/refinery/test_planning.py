"""The crude unit as a delta-base planning block (#297).

What is pinned here, and why:

* The block's delta vectors are the column's own derivatives:
  ``check_delta_vectors`` against central differences on the 30-stage test
  column, and ``check_delta_health`` clean on the default outputs.
* Levers and outputs are in planner units (bbl/d, MW, C, K, kg/h), and the
  block says so in its metadata, which is what an export writes out.
* A spec the block does not lever is held as the planner means it: a held
  product *yield* follows the crude rate, not a held absolute rate that
  hands the whole rate change to the residue.
* A non-converged column is NaN, and a plan never lands there. The rejection
  test runs the planner into the region where the column fails and checks
  it backs off; the same run with the mask off shows what the planner would
  otherwise have taken -- a column that did not converge, scored as a plan.
* CDU -> product value, planned, then re-scored in a fresh nonlinear CDU
  solve that converges and agrees with the planner's state.
"""

import json

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow.planning import (DeltaBasePlanner, Network, check_delta_health, check_delta_vectors,
                              state_is_finite)
from difflow.planning.export import DeltaVectorSet, write_json
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions
from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.planning import (available_levers, cdu_block, link_cdu,
                                       product_value_block)
from difflow_refinery.thermo import ColumnThermo

from .test_column import LIGHT, PCT, T_C, _atmospheric_params

jax.config.update("jax_enable_x64", True)

ASSAY = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT)
BPD = 95_000.0
T_IN, P_IN = 273.15 + 240.0, 6e5
PRODUCTS = ["naphtha", "kero", "diesel", "ago", "residue"]


@pytest.fixture(scope="module")
def unit():
    """The 30-stage atmospheric column of test_unit.py, closed by a 5% overflash."""
    crude = characterize(ASSAY)
    th = ColumnThermo.from_characterization(crude)
    kg_s = BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    base = _atmospheric_params(feed, th, pa1_duty=15e6)
    params = _atmospheric_params(feed, th, pa1_duty=15e6, specs=base.specs + (cc.overflash(0.05),))
    return dr.CrudeUnit(ASSAY, params)


@pytest.fixture(scope="module")
def solved(unit):
    return unit.solve(BPD, T=T_IN, P=P_IN)


@pytest.fixture(scope="module")
def block(unit):
    """Four levers, the default outputs."""
    return cdu_block(unit, ["crude.rate", "naphtha.yield", "kero.yield", "pa1.duty"],
                     rate=BPD, T=T_IN, P=P_IN)


class TestLevers:
    def test_what_the_test_column_offers(self, unit):
        lv = available_levers(unit)
        for name, units in [("crude.rate", "bbl/d"), ("preheat.T", "C"), ("naphtha.yield", "-"),
                            ("naphtha.bpd", "bbl/d"), ("overflash", "-"), ("pa1.duty", "MW"),
                            ("pa2.dT", "K"), ("bottom.steam", "kg/h"), ("kero.steam", "kg/h")]:
            assert lv[name].units == units
        # the column is closed by an overflash, so it has no coil-outlet lever
        assert "furnace.cot" not in lv

    @pytest.mark.parametrize("name", ["naphtha.yield", "naphtha.bpd", "pa1.duty", "overflash",
                                      "pa1.dT", "bottom.steam", "preheat.T"])
    def test_planner_units_round_trip(self, unit, name):
        lv = available_levers(unit)[name]
        feed_volume = BPD * cc.BARREL / 86400.0
        assert lv.from_si(lv.to_si(1.2345, feed_volume), feed_volume) == pytest.approx(1.2345)

    def test_planner_units_mean_what_they_say(self, unit):
        lv = available_levers(unit)
        feed_volume = BPD * cc.BARREL / 86400.0
        assert lv["pa1.duty"].to_si(15.0, feed_volume) == pytest.approx(15e6)
        assert lv["preheat.T"].to_si(240.0, feed_volume) == pytest.approx(513.15)
        assert lv["naphtha.bpd"].to_si(19_000.0, feed_volume) == pytest.approx(0.2 * feed_volume)
        assert lv["naphtha.yield"].to_si(0.2, feed_volume) == pytest.approx(0.2 * feed_volume)
        # 1 mol/s of steam is 64.854 kg/h
        assert lv["bottom.steam"].from_si(1.0, feed_volume) == pytest.approx(64.854)

    def test_bad_requests_say_what_is_wrong(self, unit):
        kw = dict(rate=BPD, T=T_IN, P=P_IN, jit=False)
        with pytest.raises(ValueError, match="no lever"):
            cdu_block(unit, ["furnace.cot"], **kw)
        with pytest.raises(ValueError, match="same spec"):
            cdu_block(unit, ["naphtha.yield", "naphtha.bpd"], **kw)
        with pytest.raises(ValueError, match="are levers"):
            cdu_block(unit, ["naphtha.yield"], ["naphtha.yield"], **kw)
        with pytest.raises(ValueError, match="no output"):
            cdu_block(unit, ["naphtha.yield"], ["naphtha.octane"], **kw)
        with pytest.raises(ValueError, match="outside its bounds"):
            cdu_block(unit, ["pa1.duty"], ["kero.tbp95"], bounds={"pa1.duty": (20.0, 30.0)}, **kw)


class TestBlock:
    def test_units_metadata(self, block):
        assert block.metadata["u_units"] == ["bbl/d", "-", "-", "MW"]
        units = dict(zip(block.y_names, block.metadata["y_units"]))
        assert units["kero.bpd"] == "bbl/d"
        assert units["kero.tbp95"] == "C"
        assert units["gap.kero_diesel"] == "K"
        assert units["cut.kero_diesel"] == "C"
        assert units["furnace.fired"] == "MW"
        assert block.fn.output_units["steam.total"] == "kg/h"
        assert len(block.metadata["y_units"]) == len(block.y_names)

    def test_base_point_is_the_unit_solve(self, block, solved):
        np.testing.assert_allclose(block.u0, [BPD, 0.2, 0.11, 15.0], rtol=1e-12)
        y = dict(zip(block.y_names, np.asarray(block.fn(block.u0))))
        props = solved.properties
        for p in PRODUCTS:
            assert y[f"{p}.bpd"] == pytest.approx(float(props[p].bpd), rel=1e-9)
            assert y[f"{p}.tbp95"] == pytest.approx(float(props[p].tbp_at(95)) - 273.15, abs=1e-6)
        assert y["gap.kero_diesel"] == pytest.approx(float(solved.gaps()[("kero", "diesel")]), abs=1e-6)
        assert y["furnace.fired"] == pytest.approx(float(solved.column.furnace_fired_duty) / 1e6, rel=1e-9)

    def test_default_outputs_are_healthy(self, block):
        """No held yield (a constant row) and no light-ends TBP5 (a kink) by default."""
        assert "diesel.yield" not in block.y_names
        assert "naphtha.tbp5" not in block.y_names
        assert "naphtha.yield" not in block.y_names        # a lever
        assert "steam.total" not in block.y_names          # no steam lever: a constant
        report = check_delta_health(block)
        assert report.ok, [str(f) for f in report.findings]

    def test_delta_vectors_against_finite_differences(self, block):
        out = check_delta_vectors(block)
        assert out["passed"], (out["max_rel_error"], out["max_abs_error"])

    def test_a_held_yield_follows_the_crude(self, unit):
        blk = cdu_block(unit, ["crude.rate"], ["diesel.bpd", "diesel.yield", "residue.yield"],
                        rate=BPD, T=T_IN, P=P_IN)
        y = np.asarray(blk.fn(np.array([BPD * 1.05])))
        assert y[0] == pytest.approx(0.17 * BPD * 1.05, rel=1e-9)
        assert y[1] == pytest.approx(0.17, rel=1e-9)
        assert y[2] == pytest.approx(0.47, rel=1e-9)

    def test_kink_in_the_light_ends_tbp5(self, unit):
        """Why naphtha TBP5 is not a default output: it sits on n-butane's node.

        A TBP point is piecewise linear in the product's cumulative volume,
        one node per component; at the base point the naphtha's 5% point is
        exactly the n-butane node, and its slope is one-sided.
        """
        blk = cdu_block(unit, ["naphtha.yield"], ["naphtha.tbp5"], rate=BPD, T=T_IN, P=P_IN)
        f = lambda x: float(blk.evaluate(np.array([x]))[0])
        h = 1e-3
        left = (f(0.2) - f(0.2 - h)) / h
        right = (f(0.2 + h) - f(0.2)) / h
        assert left > 1.5 * right > 0

    def test_export_carries_the_units(self, block, tmp_path):
        path = write_json(DeltaVectorSet.from_block(block), tmp_path / "cdu.json")
        text = json.dumps(json.load(open(path)))
        assert "bbl/d" in text and "MW" in text


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


@pytest.mark.slow
class TestNetwork:
    """CDU -> product value, planned, then re-scored in the nonlinear column."""

    PRICES = {"naphtha": 70.0, "kero": 95.0, "diesel": 90.0, "ago": 80.0, "residue": 55.0}
    KERO_EP = 235.0

    @pytest.fixture(scope="class")
    def plan(self, unit, solved):
        cdu = cdu_block(unit, ["crude.rate", "naphtha.yield", "kero.yield", "overflash"],
                        [f"{p}.bpd" for p in PRODUCTS]
                        + ["naphtha.tbp95", "kero.tbp95", "gap.kero_diesel", "furnace.fired"],
                        rate=BPD, T=T_IN, P=P_IN,
                        bounds={"crude.rate": (85_000.0, 100_000.0), "naphtha.yield": (0.17, 0.23),
                                "kero.yield": (0.08, 0.15), "overflash": (0.03, 0.07)})
        value = product_value_block(self.PRICES,
                                    rates={p: float(solved.properties[p].bpd) for p in PRODUCTS})
        net = Network([cdu, value], link_cdu(cdu, value))
        planner = DeltaBasePlanner(
            net, prices={"value.revenue": 1.0, "cdu.crude.rate": -65.0,      # $/bbl crude
                         "cdu.furnace.fired": -700.0},                       # $/MW/d fuel
            specs=[Spec("cdu.kero.tbp95", "<=", self.KERO_EP), Spec("cdu.naphtha.tbp95", "<=", 165.0)],
            options=TrustRegionOptions(radius=0.3, radius_min=1e-4, tol=1e-6, max_iter=40),
            vertex_seeding=False)
        return cdu, net, planner, planner.solve()

    def test_network_is_healthy(self, plan):
        _, net, _, _ = plan
        report = check_delta_health(net)
        assert report.ok, [str(f) for f in report.findings]

    def test_the_plan_terminates_on_its_own(self, plan):
        res = plan[-1]
        assert res.converged, res.reason
        assert res.total_violation <= 1e-6

    def test_the_plan_improves_on_the_base(self, plan):
        res = plan[-1]
        start = res.history[0].merit          # the merit at the base point
        assert res.merit > start + 1e5

    def test_the_kero_end_point_binds(self, plan):
        """Kero is the best-paid product: the plan takes it until its end point stops it."""
        res = plan[-1]
        assert res.state.values["cdu.kero.tbp95"] == pytest.approx(self.KERO_EP, abs=0.05)

    def test_rescored_in_the_nonlinear_column(self, plan, unit):
        """A fresh unit solve at the plan converges and reproduces the planner's state."""
        cdu, _, _, res = plan
        u = np.array([res.plan[f"cdu.{n}"] for n in cdu.u_names])
        fresh = cdu.fn.solve(u)
        assert bool(fresh.converged)
        y = cdu.fn.outputs_of(fresh)
        for n in cdu.y_names:
            assert float(y[n]) == pytest.approx(float(res.state.values[f"cdu.{n}"]), rel=1e-8, abs=1e-6)
        assert float(fresh.properties["kero"].tbp_at(95)) - 273.15 <= self.KERO_EP + 0.05
