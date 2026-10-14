"""The crude unit as a delta-base planning block (#297).

What is pinned here, and why:

* The block's delta vectors are the column's own derivatives:
  ``check_delta_vectors`` against central differences, and
  ``check_delta_health`` clean.
* Levers and outputs are in planner units (bbl/d, MW, C, K, kg/h), and the
  block says so in its metadata, which is what an export writes out; the
  default outputs leave out the rows that would be flagged (the light-ends
  TBP5, a constant steam total).
* A non-converged column is NaN, and a plan never lands there: see
  test_planning_nonconvergence.py.
* CDU -> product value, planned, then re-scored in a fresh nonlinear CDU
  solve that converges and agrees with the planner's state.

Not run, for cost (each is another compile of the column): the held-yield
semantics (a spec the block does not lever holds the *yield*, so it follows
the crude rate), the TBP5 kink, the JSON export of a block, and the 30-stage
atmospheric column with side products.

The unit is an 8-stage column with one pumparound and a 5 % overflash (the
30-stage atmospheric column with side products costs minutes per compile and
adds nothing to what the planning layer does with it).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow.planning import DeltaBasePlanner, Network, check_delta_health, check_delta_vectors
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions
from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.planning import (available_levers, cdu_block, link_cdu,
                                       product_value_block)
from difflow_refinery.thermo import ColumnThermo

jax.config.update("jax_enable_x64", True)

ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)
BPD = 12_000.0
T_IN, P_IN = 430.0, 4e5
PRODUCTS = ["naphtha", "residue"]
PA_DUTY = 0.1e6   # W


def build_unit(pa_duty=PA_DUTY):
    """The 8-stage column closed by a 5% overflash, with one pumparound."""
    crude = characterize(ASSAY)
    th = ColumnThermo.from_characterization(crude)
    kg_s = BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    Vf = float(th.std_volume(jnp.stack([feed[f"F_{n}"] for n in th.names])))
    params = cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=20.0, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 0.15 * Vf), cc.pumparound_duty("pa1", pa_duty),
               cc.pumparound_delta_t("pa1", 40.0), cc.overflash(0.05)),
        pumparounds=(cc.Pumparound("pa1", 5, 3),), max_iter=40)
    return dr.CrudeUnit(ASSAY, params)


@pytest.fixture(scope="module")
def unit():
    return build_unit()


@pytest.fixture(scope="module")
def solved(unit):
    return unit.solve(BPD, T=T_IN, P=P_IN)


@pytest.fixture(scope="module")
def block(unit):
    """Two levers, four outputs: what the planner below uses, so one compile serves every test."""
    return cdu_block(unit, ["crude.rate", "naphtha.yield"],
                     [f"{p}.bpd" for p in PRODUCTS] + ["naphtha.tbp95", "furnace.fired"],
                     rate=BPD, T=T_IN, P=P_IN,
                     bounds={"crude.rate": (10_000.0, 13_000.0), "naphtha.yield": (0.12, 0.18)})


class TestLevers:
    def test_what_the_test_column_offers(self, unit):
        lv = available_levers(unit)
        for name, units in [("crude.rate", "bbl/d"), ("preheat.T", "C"), ("naphtha.yield", "-"),
                            ("naphtha.bpd", "bbl/d"), ("overflash", "-"), ("pa1.duty", "MW"),
                            ("pa1.dT", "K"), ("bottom.steam", "kg/h")]:
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
        assert lv["naphtha.bpd"].to_si(0.15 * BPD, feed_volume) == pytest.approx(0.15 * feed_volume)
        assert lv["naphtha.yield"].to_si(0.15, feed_volume) == pytest.approx(0.15 * feed_volume)
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


class TestBlock:
    def test_units_metadata(self, block):
        assert block.metadata["u_units"] == ["bbl/d", "-"]
        units = dict(zip(block.y_names, block.metadata["y_units"]))
        assert units["naphtha.bpd"] == "bbl/d"
        assert units["naphtha.tbp95"] == "C"
        assert units["furnace.fired"] == "MW"
        assert len(block.metadata["y_units"]) == len(block.y_names)

    def test_default_outputs_leave_out_the_unhealthy_rows(self, block):
        """No light-ends TBP5 (a kink), and no steam total with no steam lever, by default."""
        default = block.fn.default_outputs()
        assert "naphtha.tbp5" not in default
        assert "steam.total" not in default              # no steam lever: a constant
        assert "furnace.fired" in default and "gap.naphtha_residue" in default

    def test_base_point_is_the_unit_solve(self, block, solved):
        np.testing.assert_allclose(block.u0, [BPD, 0.15], rtol=1e-12)
        y = dict(zip(block.y_names, np.asarray(block.fn(block.u0))))
        props = solved.properties
        for p in PRODUCTS:
            assert y[f"{p}.bpd"] == pytest.approx(float(props[p].bpd), rel=1e-9)
        assert y["naphtha.tbp95"] == pytest.approx(float(props["naphtha"].tbp_at(95)) - 273.15, abs=1e-6)
        assert y["furnace.fired"] == pytest.approx(float(solved.column.furnace_fired_duty) / 1e6, rel=1e-9)

    def test_delta_vectors_are_healthy_and_match_finite_differences(self, block):
        out = check_delta_vectors(block)
        assert out["passed"], (out["max_rel_error"], out["max_abs_error"])
        report = check_delta_health(block)
        assert report.ok, [str(f) for f in report.findings]


class TestNetwork:
    """CDU -> product value, planned, then re-scored in the nonlinear column."""

    PRICES = {"naphtha": 70.0, "residue": 55.0}
    NAPHTHA_EP = 165.0

    @pytest.fixture(scope="class")
    def plan(self, block, solved):
        cdu = block
        value = product_value_block(self.PRICES,
                                    rates={p: float(solved.properties[p].bpd) for p in PRODUCTS})
        net = Network([cdu, value], link_cdu(cdu, value))
        planner = DeltaBasePlanner(
            net, prices={"value.revenue": 1.0, "cdu.crude.rate": -65.0,      # $/bbl crude
                         "cdu.furnace.fired": -700.0},                       # $/MW/d fuel
            specs=[Spec("cdu.naphtha.tbp95", "<=", self.NAPHTHA_EP)],
            options=TrustRegionOptions(radius=0.3, radius_min=1e-4, tol=1e-6, max_iter=20),
            vertex_seeding=False)
        return cdu, net, planner, planner.solve()

    def test_the_plan_terminates_on_its_own(self, plan):
        res = plan[-1]
        assert res.converged, res.reason
        assert res.total_violation <= 1e-6

    def test_the_plan_improves_on_the_base(self, plan):
        res = plan[-1]
        start = res.history[0].merit          # the merit at the base point
        assert res.merit > start

    def test_the_end_point_binds(self, plan):
        res = plan[-1]
        assert res.state.values["cdu.naphtha.tbp95"] == pytest.approx(self.NAPHTHA_EP, abs=0.05)

    def test_rescored_in_the_nonlinear_column(self, plan, unit):
        """A fresh unit solve at the plan converges and reproduces the planner's state."""
        cdu, _, _, res = plan
        u = np.array([res.plan[f"cdu.{n}"] for n in cdu.u_names])
        fresh = cdu.fn.solve(u)
        assert bool(fresh.converged)
        y = cdu.fn.outputs_of(fresh)
        for n in cdu.y_names:
            assert float(y[n]) == pytest.approx(float(res.state.values[f"cdu.{n}"]), rel=1e-8, abs=1e-6)
        assert float(fresh.properties["naphtha"].tbp_at(95)) - 273.15 <= self.NAPHTHA_EP + 0.05
