"""Tests for difflow.units.membranes: gas permeation, reverse osmosis, UF.

The plug-flow gas models are checked against independent hand integrations
in the feed-side mole fraction (the textbook form, Geankoplis 5e §13.6-13.7),
not against themselves.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.integrate import quad, solve_ivp

from difflow import make_stream
from difflow.units import membranes as M

jax.config.update("jax_enable_x64", True)

ALPHA, PR, XF = 10.0, 0.25, 0.5
F2 = jnp.array([0.5, 0.5])
Q2 = jnp.array([1e-2, 1e-3])  # alpha = 10 with P_h = 1, P_l = 0.25


def _yl(x):
    return float(M.binary_permeate_composition(x, ALPHA, PR))


class TestBinaryClosedForms:
    def test_quadratic_satisfies_the_flux_ratio(self):
        for x in (0.05, 0.25, 0.5, 0.9):
            y = _yl(x)
            lhs = y / (1 - y)
            rhs = ALPHA * (x - PR * y) / ((1 - x) - PR * (1 - y))
            assert lhs == pytest.approx(rhs, rel=1e-12)

    def test_alpha_one_gives_no_separation(self):
        assert float(M.binary_permeate_composition(0.3, 1.0, 0.2)) == pytest.approx(0.3)

    def test_reject_composition_inverts_permeate(self):
        for x in (0.1, 0.25, 0.4):
            assert float(M.binary_reject_composition(_yl(x), ALPHA, PR)) == pytest.approx(x, rel=1e-12)

    def test_complete_mixing_given_theta_or_xo_agree(self):
        a = M.binary_complete_mixing(XF, ALPHA, PR, x_o=0.25)
        b = M.binary_complete_mixing(XF, ALPHA, PR, theta=a["theta"])
        assert float(b["x_o"]) == pytest.approx(0.25, abs=1e-10)
        # material balance
        assert float(a["theta"] * a["y_p"] + (1 - a["theta"]) * 0.25) == pytest.approx(XF)

    def test_multicomponent_complete_mixing_reduces_to_binary(self):
        a = M.binary_complete_mixing(XF, ALPHA, PR, x_o=0.25)
        A = M.complete_mixing_area(1.0, a["theta"], a["y_p"], 0.25, Q2[0], 1.0, PR)
        ret, perm = M.complete_mixing(F2, Q2, A, 1.0, PR)
        assert float(jnp.sum(perm)) == pytest.approx(float(a["theta"]), rel=1e-9)
        assert float(perm[0] / jnp.sum(perm)) == pytest.approx(float(a["y_p"]), rel=1e-9)
        assert float(M.area_for_stage_cut(F2, Q2, 1.0, PR, a["theta"])) == pytest.approx(float(A), rel=1e-8)

    def test_units(self):
        assert M.BARRER == pytest.approx(3.3464e-16, rel=1e-4)
        assert float(M.permeance_from_permeability(1.0, 1.0)) == pytest.approx(M.GPU, rel=1e-12)


class TestPlugFlowAgainstHand:
    def test_cross_flow_matches_x_integration(self):
        x_o = 0.25
        lnr = quad(lambda x: 1 / (_yl(x) - x), XF, x_o)[0]
        theta = 1 - np.exp(lnr)

        def dA(x):
            L = np.exp(quad(lambda s: 1 / (_yl(s) - s), XF, x)[0])
            return -L / (_yl(x) - x) * _yl(x) / (1e-2 * (x - PR * _yl(x)))

        area = quad(dA, XF, x_o, limit=100)[0]
        ret, perm, _ = M.cross_flow(F2, Q2, area, 1.0, PR, 200)
        assert float(jnp.sum(perm)) == pytest.approx(theta, rel=1e-7)
        assert float(ret[0] / jnp.sum(ret)) == pytest.approx(x_o, abs=1e-7)

    def test_countercurrent_matches_x_integration(self):
        x_o = 0.25

        def rhs(x, s):
            L, _ = s
            y = _yl(x) if L - 1 < 1e-10 else (L * x - x_o) / (L - 1)
            JA, JB = 1e-2 * (x - PR * y), 1e-3 * ((1 - x) - PR * (1 - y))
            dL = L / (JA / (JA + JB) - x)
            return [dL, dL / (JA + JB)]

        sol = solve_ivp(rhs, [x_o, XF], [1.0, 0.0], rtol=1e-10, atol=1e-12)
        Lf, A = sol.y[:, -1]
        ret, perm, prof = M.countercurrent(F2, Q2, A / Lf, 1.0, PR, 200)
        assert float(jnp.sum(perm)) == pytest.approx(1 - 1 / Lf, rel=1e-6)
        assert float(ret[0] / jnp.sum(ret)) == pytest.approx(x_o, abs=1e-6)
        assert float(prof["residual"]) < 1e-8

    def test_flow_pattern_ranking(self):
        # Same area: countercurrent > cross > cocurrent > complete mixing in purity.
        purity = {}
        for pat in M.FLOW_PATTERNS:
            _, perm, _ = M.permeation_stage(F2, Q2, 200.0, 1.0, PR, pat, 30)
            purity[pat] = float(perm[0] / jnp.sum(perm))
        assert (purity["countercurrent"] > purity["cross_flow"] > purity["cocurrent"]
                > purity["complete_mixing"])

    def test_three_species_conserve(self):
        F = jnp.array([0.2, 0.3, 0.5])
        Q = jnp.array([5e-3, 1e-3, 2e-4])
        for pat in ("cross_flow", "cocurrent"):
            ret, perm, _ = M.permeation_stage(F, Q, 100.0, 1.0, 0.2, pat, 50)
            assert np.allclose(ret + perm, F)
            assert bool(jnp.all(perm >= 0))


class TestGradients:
    def test_cross_flow_grad(self):
        f = lambda A, pl: M.cross_flow(F2, Q2, A, 1.0, pl, 40)[1][0]
        check_grads(f, (150.0, 0.25), order=1, modes=["rev"], rtol=1e-4)

    def test_countercurrent_grad(self):
        f = lambda A: jnp.sum(M.countercurrent(F2, Q2, A, 1.0, PR, 40)[1])
        check_grads(f, (150.0,), order=1, modes=["rev"], rtol=1e-4)

    def test_ro_flux_grad(self):
        f = lambda dP, c: M.ro_local_flux(dP, c, 2.8e-12, 1e-8, k=3e-5, phi=0.93)["J_v"]
        check_grads(f, (6e6, 600.0), order=1, modes=["rev"], rtol=1e-4)


class TestCountercurrentSolveCost:
    """Countercurrent shooting stops at convergence instead of running every iteration.

    Reference values are from the earlier fixed-count solver (40 Newton steps per stage,
    40 bisection steps for the area); the faster solver must reproduce them.
    """

    GPU = 3.348e-10
    F = 1000 / 22.414 * 1000 / 3600  # 1000 Nm^3/h in mol/s
    Fi = jnp.array([0.4 * F, 0.6 * F])
    Q = jnp.array([100.0, 5.0]) * GPU

    def test_stage_matches_fixed_count_solver(self):
        perm = M.countercurrent(self.Fi, self.Q, 100.0, 5e5, 1.05e5, 40)[1]
        np.testing.assert_allclose(perm, [0.3487682804065479, 0.048684585979673164], rtol=1e-10)


    def test_area_for_stage_cut_matches_fixed_count_solver(self):
        A = jax.jit(lambda P: M.area_for_stage_cut(self.Fi, self.Q, P, 1.05e5, 0.3,
                                                   "countercurrent", 40))
        for P, ref in [(3e5, 3320.8473734128875), (13e5, 324.49176521775985)]:
            assert float(A(P)) == pytest.approx(ref, rel=1e-10)

    def test_illinois_needs_few_evaluations(self):
        # A dozen false-position steps reach round-off where bisection would need about 40.
        f = lambda x: x**3 + x - 1.0
        root = M._illinois_increasing(f, 0.0, 1.0, n_iter=12)
        assert abs(float(f(root))) < 1e-13

    def test_area_for_stage_cut_grad(self):
        f = lambda P, theta: M.area_for_stage_cut(self.Fi, self.Q, P, 1.05e5, theta,
                                                  "countercurrent", 40)
        check_grads(f, (5e5, 0.3), order=1, modes=["rev"], eps=1e-4, rtol=1e-4)


class TestRunDry:
    """More area than the feed needs: the feed side empties and everything permeates.

    The flux per unit area depends on mole fractions, not on how much gas is left, so a
    plug-flow stage runs dry at a finite area. Past it the stages used to return NaN.
    """

    GPU = 3.348e-10
    Fi = 1000 / 22.414 / 3.6 * jnp.array([0.15, 0.85])
    Q = jnp.array([100.0, 5.0]) * GPU
    P_h, P_l = 13e5, 1.05e5
    # Near the dry end only the slow gas is left and its flux Q (P_h - P_l) is constant,
    # so the cut rises linearly in area at this slope until it reaches 1.
    slope = float(5.0 * GPU * (13e5 - 1.05e5) / jnp.sum(Fi))

    def test_empty_feed_side_has_zero_flux(self):
        y, J = M.local_permeate_composition(jnp.zeros(2), self.Q, self.P_h, self.P_l)
        assert float(J) == 0.0 and np.all(np.isfinite(y))
        g = jax.grad(lambda x: M.local_permeate_composition(x, self.Q, self.P_h, self.P_l)[1])
        assert np.all(np.isfinite(g(jnp.zeros(2))))

    @pytest.mark.parametrize("pattern", ["cross_flow", "cocurrent", "countercurrent"])
    def test_large_area_permeates_everything(self, pattern):
        R, V, prof = M.permeation_stage(self.Fi, self.Q, 1e4, self.P_h, self.P_l, pattern)
        np.testing.assert_allclose(V, self.Fi, rtol=1e-12)
        assert np.all(np.asarray(R) == 0.0)
        assert np.all(np.isfinite(np.asarray(prof["L"])))
        if pattern == "countercurrent":
            assert bool(prof["exhausted"])
            np.testing.assert_allclose(prof["L"][0], self.Fi)    # feed end
            assert float(jnp.sum(prof["L"][-1])) == 0.0            # dry before the retentate end

    def test_countercurrent_is_continuous_through_the_dry_point(self):
        # Just short of running dry the shooting solve converges and the cut sits on the
        # constant-flux line; just past it the stage is dry. Both sides give finite gradients.
        cut = jax.jit(lambda A: jnp.sum(M.countercurrent(self.Fi, self.Q, A, self.P_h, self.P_l)[1])
                      / jnp.sum(self.Fi))
        dcut = jax.jit(jax.grad(cut))
        c0 = float(cut(5300.0))
        for A in (5305.0, 5310.0, 5312.0):
            assert float(cut(A)) == pytest.approx(c0 + self.slope * (A - 5300.0), abs=1e-9)
            assert float(dcut(A)) == pytest.approx(self.slope, rel=1e-6)
        A_dry = 5300.0 + (1.0 - c0) / self.slope
        for A in (A_dry + 0.2, A_dry + 5.0):
            assert float(cut(A)) == 1.0 and float(dcut(A)) == 0.0

    def test_countercurrent_unchanged_short_of_dry(self):
        R, _, prof = M.countercurrent(self.Fi, self.Q, 3000.0, self.P_h, self.P_l)
        assert not bool(prof["exhausted"]) and float(prof["residual"]) < 1e-13
        np.testing.assert_allclose(R, [5.825562e-06, 4.625748], rtol=1e-6)


class TestGasPermeator:
    def test_conserves_and_reports(self):
        feed = make_stream({"CO2": 0.13, "N2": 0.87}, T=313.0, P=2e5)
        unit = M.GasPermeator(M.GasPermeationParams(
            permeances={"CO2": 1000 * M.GPU, "N2": 1000 / 30 * M.GPU}, area=50.0,
            flow_pattern="cross_flow", pressure_ratio=10.0, n_steps=40))
        ret, perm, info = unit(feed)
        assert float(ret["F_CO2"] + perm["F_CO2"]) == pytest.approx(0.13)
        assert float(perm["P"]) == pytest.approx(2e4)
        assert float(info["selectivity"]["CO2"]) == pytest.approx(30.0)

    def test_stage_cut_design_mode(self):
        feed = make_stream({"O2": 0.21, "N2": 0.79}, T=298.15, P=1e6)
        unit = M.GasPermeator(M.GasPermeationParams(
            permeances={"O2": 10 * M.GPU, "N2": 2 * M.GPU}, stage_cut=0.3,
            flow_pattern="cross_flow", pressure_ratio=5.0, n_steps=40))
        _, _, info = unit(feed)
        assert float(info["stage_cut"]) == pytest.approx(0.3, abs=1e-8)

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            M.GasPermeationParams(permeances={}, area=1.0, flow_pattern="spiral")
        with pytest.raises(ValueError):
            M.GasPermeationParams(permeances={}, area=1.0, pressure_ratio=0.5)
        with pytest.raises(ValueError):
            M.GasPermeationParams(permeances={})

    def test_cc_membrane_cross_flow_matches_core(self):
        from difflow_cc.units.membrane import MembraneParams, MembraneSeparator
        feed = make_stream({"CO2": 0.13, "N2": 0.87}, T=298.15, P=2e5)
        cc = MembraneSeparator(MembraneParams(membrane_type="PIM_1", area=20.0, pressure_ratio=8.0,
                                              flow_pattern="cross_flow", n_steps=40))
        _, perm_cc, info = cc(feed)
        Q = {s: cc._permeance(s, 298.15) for s in ("CO2", "N2")}
        core = M.GasPermeator(M.GasPermeationParams(permeances=Q, area=20.0, pressure_ratio=8.0,
                                                    flow_pattern="cross_flow", n_steps=40))
        _, perm, _ = core(feed)
        assert float(perm_cc["F_CO2"]) == pytest.approx(float(perm["F_CO2"]), rel=1e-10)
        assert float(info["CO2_recovery"]) > 0


class TestReverseOsmosis:
    def test_perfect_membrane_no_polarization(self):
        f = M.ro_local_flux(6e6, 600.0, 2.8e-12, 0.0)
        pi = float(M.osmotic_pressure(600.0, 298.15))
        assert float(f["J_v"]) == pytest.approx(2.8e-12 * (6e6 - pi), rel=1e-9)
        assert float(f["c_p"]) == pytest.approx(0.0, abs=1e-12)

    def test_polarization_lowers_flux_and_rejection(self):
        a = M.ro_local_flux(6e6, 600.0, 2.8e-12, 1e-8)
        b = M.ro_local_flux(6e6, 600.0, 2.8e-12, 1e-8, k=2e-5)
        assert float(b["J_v"]) < float(a["J_v"])
        assert float(b["rejection"]) < float(a["rejection"])
        # film theory holds at the solution
        E = np.exp(float(b["J_v"]) / 2e-5)
        assert (float(b["c_w"]) - float(b["c_p"])) == pytest.approx(E * (600.0 - float(b["c_p"])), rel=1e-8)

    def test_osmotic_pressure_vant_hoff(self):
        # 0.1 mol/L NaCl, ideal: 2 x 100 x R x 298.15 Pa
        assert float(M.osmotic_pressure(100.0, 298.15)) == pytest.approx(2 * 100 * 8.314462618 * 298.15)

    def test_element_recovery_below_limit(self):
        Qf = 100 / 3600
        c = 35 / 58.44 * 1000
        feed = make_stream({"H2O": Qf * 997 / 0.018015, "NaCl": Qf * c}, T=298.15, P=101325 + 60e5)
        ro = M.ReverseOsmosis(M.ReverseOsmosisParams(A_w=2.78e-12, B=0.0, area=3e4, n_steps=300))
        conc, perm, info = ro(feed)
        assert float(info["recovery"]) < float(info["recovery_limit"])
        assert float(info["recovery"]) == pytest.approx(float(info["recovery_limit"]), rel=0.02)
        assert float(conc["F_NaCl"]) == pytest.approx(Qf * c, rel=1e-9)


class TestUltrafiltration:
    def test_flux_caps_at_gel_limit(self):
        J_lo = M.uf_flux(0.2e5, 1e12, 1e-3, 1e-5, 10.0, 250.0)
        J_hi = M.uf_flux(5e5, 1e12, 1e-3, 1e-5, 10.0, 250.0)
        assert float(J_lo) == pytest.approx(0.2e5 / (1e-3 * 1e12))
        assert float(J_hi) == pytest.approx(1e-5 * np.log(25.0))

    def test_polarization_modulus(self):
        assert float(M.polarization_modulus(1e-5, 1e-5)) == pytest.approx(np.e)
        assert float(M.polarization_modulus(1e-5, 1e-5, R=0.0)) == pytest.approx(1.0)

    def test_bio_wall_concentration_uses_the_core_modulus(self):
        # A fully retained solute polarizes; the bio helper used to return
        # the bulk concentration for R = 1 (formula with R and 1 - R swapped).
        from difflow_bio.units.filtration import concentration_polarization, gel_layer_flux
        assert float(concentration_polarization(10.0, 1e-5, 1e-5, 1.0)) == pytest.approx(10 * np.e)
        assert float(concentration_polarization(10.0, 1e-5, 1e-5, 0.0)) == pytest.approx(10.0)
        assert float(gel_layer_flux(10.0, 250.0, 1e-5)) == pytest.approx(1e-5 * np.log(25.0))

    def test_batch_time_matches_quad(self):
        t = float(M.batch_concentration_time(1.0, 0.1, 10.0, 1e-5, 5.0, 250.0))
        ref = quad(lambda V: 1 / (10.0 * 1e-5 * np.log(250.0 * V / 5.0)), 0.1, 1.0)[0]
        assert t == pytest.approx(ref, rel=1e-8)
