"""CCR reformer (#474): activity from coke, moving-bed coke balance, regenerator."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.integrate import quad

from difflow_refinery import combustion as cb
from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
from difflow_refinery.reforming.ccr import (
    CCR_ENVELOPE, PT_MASS, CCRParams, CCRReformer, Regenerator, RegeneratorParams, activity,
    ccr_reformer_params, mean_activity)

jax.config.update("jax_enable_x64", True)


class TestActivity:
    def test_fresh_catalyst_is_fully_active(self):
        assert float(activity(0.0, 10.0)) == 1.0

    @pytest.mark.parametrize("c0,c1", [(0.0, 0.05), (0.01, 0.02), (0.03, 0.0300001), (0.02, 0.02)])
    def test_mean_activity_is_the_path_average(self, c0, c1):
        alpha = 10.0
        if c1 == c0:
            want = np.exp(-alpha * c0)
        else:
            want = quad(lambda c: np.exp(-alpha * c), c0, c1)[0] / (c1 - c0)
        assert float(mean_activity(c0, c1, alpha)) == pytest.approx(want, rel=1e-12)

    def test_series_branch_joins_smoothly(self):
        # Either side of |x| = 1e-4 the two branches agree to the series' error.
        a = mean_activity(0.01, 0.01 + 0.99e-5, 10.0)
        b = mean_activity(0.01, 0.01 + 1.01e-5, 10.0)
        assert float(a - b) == pytest.approx(10.0 * np.exp(-0.1) * 0.02e-5 / 2, rel=1e-3)

    def test_gradients(self):
        check_grads(lambda c0, c1, al: mean_activity(c0, c1, al), (0.01, 0.04, 10.0),
                    order=2, modes=["fwd", "rev"])
        check_grads(lambda c1: mean_activity(0.02, c1, 10.0), (0.02,), order=1,
                    modes=["fwd", "rev"])


class TestBedCokes:
    def test_satisfies_the_moving_bed_balance(self):
        u = CCRReformer(ccr_reformer_params())
        s = np.array([1e-4, 2e-4, 3e-4, 5e-4])
        c = u._bed_cokes(s, 0.02, 10.0, 0.001)
        c_in = np.concatenate([[0.001], c[:-1]])
        a = np.array([float(mean_activity(ci, co, 10.0)) for ci, co in zip(c_in, c)])
        np.testing.assert_allclose(c - c_in, s * a / 0.02, rtol=1e-10)

    def test_no_deactivation_is_a_cumulative_sum(self):
        u = CCRReformer(ccr_reformer_params())
        s = np.array([1e-4, 2e-4, 3e-4, 5e-4])
        c = u._bed_cokes(s, 0.02, 0.0, 0.001)
        np.testing.assert_allclose(c, 0.001 + np.cumsum(s) / 0.02, rtol=1e-12)


class TestRegenerator:
    G, C = 0.02, 0.05

    def test_coke_burned_is_the_coke_lifted(self):
        rg = Regenerator()(self.G, self.C)
        B = self.G * (self.C - RegeneratorParams().coke_regen)
        assert float(rg["coke_burn"]) == pytest.approx(B, rel=1e-14)
        assert float(rg["elements"]["C"] + rg["elements"]["H"]) == pytest.approx(B, rel=1e-14)

    def test_coke_composition_from_h_to_c(self):
        el = Regenerator(RegeneratorParams(coke_HC=0.5)).elements(1.0)
        nC = el["C"] / cb.ATOMIC_WEIGHT["C"]
        nH = el["H"] / cb.ATOMIC_WEIGHT["H"]
        assert float(nH / nC) == pytest.approx(0.5, rel=1e-14)

    def test_burn_zone_is_o2_limited_at_the_inlet_o2(self):
        p = RegeneratorParams()
        rg = Regenerator(p)(self.G, self.C)
        gas = rg["burn_gas"]
        assert float(gas["oxygen"] / sum(gas.values())) == pytest.approx(p.y_O2, rel=1e-12)
        assert "oxygen" not in rg["burn_outlet"] or float(rg["burn_outlet"]["oxygen"]) == 0.0
        # The air brings exactly the O2 the coke takes: C + H/4 per mol.
        el = rg["elements"]
        o2 = el["C"] * 1000 / cb.ATOMIC_WEIGHT["C"] + el["H"] * 1000 / cb.ATOMIC_WEIGHT["H"] / 4
        assert float(rg["air"]["oxygen"]) == pytest.approx(float(o2), rel=1e-12)

    def test_element_balance_of_the_burn_zone(self):
        rg = Regenerator()(self.G, self.C)
        i, o, el = rg["burn_gas"], rg["burn_outlet"], rg["elements"]
        C = lambda g: g.get("carbon_dioxide", 0.0) + g.get("carbon_monoxide", 0.0)
        H = lambda g: 2 * g.get("water", 0.0)
        O = lambda g: (2 * g.get("oxygen", 0.0) + 2 * g.get("carbon_dioxide", 0.0)
                       + g.get("carbon_monoxide", 0.0) + g.get("water", 0.0))
        assert float(C(o) - C(i)) == pytest.approx(float(el["C"]) * 1000 / cb.ATOMIC_WEIGHT["C"], rel=1e-12)
        assert float(H(o) - H(i)) == pytest.approx(float(el["H"]) * 1000 / cb.ATOMIC_WEIGHT["H"], rel=1e-12)
        assert float(O(o)) == pytest.approx(float(O(i)), rel=1e-12)
        assert float(o["nitrogen"]) == pytest.approx(float(i["nitrogen"]), rel=1e-14)

    def test_peak_temperature_closes_the_energy_balance(self):
        rg = Regenerator()(self.G, self.C)
        T_in = RegeneratorParams().T_burn_in
        gained = cb.gas_enthalpy(rg["burn_outlet"], rg["T_peak"]) - cb.gas_enthalpy(rg["burn_outlet"], T_in)
        assert float(gained) == pytest.approx(float(rg["heat"]), rel=1e-10)

    @pytest.mark.parametrize("y", [0.005, 0.008, 0.01])
    def test_temperature_rise_closed_form(self, y):
        # dT = y_O2 (heat per mol O2) / cp of the burn gas: the burn-front rise.
        rg = Regenerator(RegeneratorParams(y_O2=y))(self.G, self.C)
        T_in = RegeneratorParams().T_burn_in
        q = float(rg["heat"] / rg["air"]["oxygen"])
        n = sum(rg["burn_outlet"].values())
        cp = float((cb.gas_enthalpy(rg["burn_outlet"], rg["T_peak"])
                    - cb.gas_enthalpy(rg["burn_outlet"], T_in)) / (rg["T_peak"] - T_in) / n)
        n_in = sum(rg["burn_gas"].values())
        assert float(rg["T_peak"] - T_in) == pytest.approx(y * q / cp * float(n_in / n), rel=1e-9)
        # Hand check: CH0.6 releases ~400 kJ per mol O2; over an N2-rich gas Cp of
        # ~31-33 J/mol/K that is ~12-13 K per 0.1 % O2 (no published rule found).
        assert 10.0 < float(rg["T_peak"] - T_in) / (y / 0.001) < 15.0

    def test_peak_does_not_depend_on_the_coke_rate(self):
        # The recycle scales with the O2: the front rise is set by y_O2, not by the burn rate.
        a = Regenerator()(self.G, self.C)["T_peak"]
        b = Regenerator()(2 * self.G, self.C)["T_peak"]
        assert float(a) == pytest.approx(float(b), rel=1e-12)

    def test_design_burn_capacity(self):
        # [AFPM13]: 52 lb coke/h per 1000 lb/h circulation; at the 5.25 wt% design
        # coke (less the 0.2 wt% left on) the burn uses (5.25-0.2)/5.2 of it.
        from difflow_refinery.reforming.ccr import DESIGN_BURN_RATIO, DESIGN_SPENT_COKE
        assert DESIGN_BURN_RATIO == pytest.approx(52 / 1000)
        rg = Regenerator()(self.G, DESIGN_SPENT_COKE)
        assert float(rg["burn_capacity_used"]) == pytest.approx((0.0525 - 0.002) / 0.052, rel=1e-12)

    def test_defaults_inside_published_ranges(self):
        p = RegeneratorParams()
        assert 0.005 <= p.y_O2 <= 0.015                     # [US5498756]
        assert p.coke_regen <= 0.002                        # [US7638101]
        assert 0.5 <= p.coke_HC <= 0.7                      # [US5376607]
        assert 0.8 <= p.chloride_wt <= 1.2                  # [US5498756] preferred
        assert p.T_burn_max == pytest.approx(595 + 273.15)  # [AFPM16]
        assert 370 + 273.15 <= p.T_burn_in <= 550 + 273.15  # [US7638101]
        r = ccr_reformer_params()
        lo, hi = CCR_ENVELOPE["P_reactor"]
        assert lo <= float(r.P_reactor) <= hi
        lo, hi = CCR_ENVELOPE["H2_HC"]
        assert lo <= r.H2_HC <= hi

    def test_reduction_hydrogen(self):
        p = RegeneratorParams()
        rg = Regenerator(p)(self.G, self.C)
        want = 2.0 * self.G * p.pt_wt / 100 * 1000 / PT_MASS
        assert float(rg["H2_reduction"]) == pytest.approx(want, rel=1e-14)

    def test_gradients(self):
        def f(G, C, y):
            rg = Regenerator(RegeneratorParams(y_O2=y))(G, C)
            return jnp.stack([rg["T_peak"], rg["coke_burn"], rg["burn_capacity_used"],
                              rg["H2_reduction"], rg["recycle"]])
        check_grads(f, (self.G, self.C, 0.008), order=1, modes=["fwd", "rev"], eps=1e-6, rtol=1e-6)


def test_reformer_activity_length_is_checked():
    with pytest.raises(ValueError, match="activity"):
        ReformerParams(activity=(1.0, 1.0))
    with pytest.raises(ValueError, match="leave it None"):
        CCRReformer().solve(lean_naphtha(), ccr_reformer_params(activity=(1.0,) * 4))


# ----- flowsheet ------------------------------------------------------------------


@pytest.fixture(scope="module")
def base():
    return CCRReformer().solve(lean_naphtha())


class TestCCRFlowsheet:
    def test_converges(self, base):
        assert base.converged
        assert base.reformer.converged

    def test_coke_made_is_coke_burned(self, base):
        b = base.balances()
        assert abs(float(b["coke"])) < 1e-8
        assert abs(float(b["catalyst_coke"])) < 1e-8
        for k in ("mass", "carbon", "hydrogen"):
            assert abs(float(b[k])) < 1e-9, k

    def test_activities_fall_along_the_train(self, base):
        a = np.asarray(base.activity)
        assert np.all(np.diff(a) < 0) and a[0] < 1.0
        c = np.asarray(base.coke)
        assert np.all(np.diff(c) > 0)

    def test_envelope(self, base):
        o = base.outputs()
        lo, hi = CCR_ENVELOPE["RON"]
        assert lo <= float(o["reformate.RON"]) <= hi
        assert float(o["ccr.burn_T_margin_K"]) > 0.0
        assert float(o["ccr.burn_capacity_used"]) < 1.0

    def test_default_circulation_is_the_design_circulation(self, base):
        # The default circulation puts 5.25 wt% coke on the spent catalyst of
        # lean_naphtha(): the UOP design basis [AFPM16].
        assert float(base.outputs()["ccr.spent_coke_wt"]) == pytest.approx(5.25, abs=2e-3)

    def test_another_feed_stays_in_the_coke_envelope(self, base):
        from difflow_refinery.reforming import rich_naphtha
        r = CCRReformer(base.params, base.ccr).solve(rich_naphtha())
        lo, hi = CCR_ENVELOPE["spent_coke_wt"]
        assert lo <= float(r.outputs()["ccr.spent_coke_wt"]) <= hi

    def test_infinite_circulation_is_semi_regen_start_of_run(self):
        # Coke-free catalyst everywhere: the CCR is the fresh-catalyst reformer.
        p = ccr_reformer_params()
        ccr = CCRParams(circulation=1e9, regenerator=RegeneratorParams(coke_regen=0.0))
        a = CCRReformer(p, ccr).solve(lean_naphtha())
        b = CatalyticReformer(p).solve(lean_naphtha())
        np.testing.assert_allclose(np.asarray(a.activity), 1.0, atol=1e-10)
        oa, ob = a.outputs(), b.outputs()
        for k in ("reformate.RON", "reformate.yield_vol", "h2.net_mol_s", "coke.kg_h"):
            assert float(oa[k]) == pytest.approx(float(ob[k]), rel=1e-8), k

    def test_more_circulation_less_coke_more_octane(self, base):
        p, ccr = base.params, base.ccr
        r = CCRReformer(p, replace(ccr, circulation=2 * ccr.circulation)).solve(
            lean_naphtha(), tear_initial=base.tear)
        assert float(r.coke[-1]) < float(base.coke[-1])
        assert np.all(np.asarray(r.activity) > np.asarray(base.activity))
        assert float(r.outputs()["reformate.RON"]) > float(base.outputs()["reformate.RON"])


def test_ccr_block(base):
    from difflow_refinery.reforming.planning import CCR_OUTPUTS, ccr_block

    blk = ccr_block(CCRReformer(base.params, base.ccr), lean_naphtha(),
                    levers=["wait", "circulation"])
    assert blk.u0[1] == pytest.approx(float(base.ccr.circulation) * 3600.0)
    y = np.asarray(blk.fn(jnp.asarray(blk.u0)))
    o = base.outputs()
    np.testing.assert_allclose(y, [float(o[k]) for k in CCR_OUTPUTS], rtol=1e-6)
