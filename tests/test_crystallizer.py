"""Tests for the crystallization library and the Crystallizer unit (issue #433).

Oracles (nothing here is a published textbook number; see the end of this docstring):

* the yield balance against an independent 3x3 linear solve of the solute and water
  balances (NumPy), and a hand-worked round-number example;
* the MSMPR moments against ``scipy.integrate.quad`` of the distribution, the
  dominant size as the argmax of ``L^3 n``, the mass median as the root of the
  quadrature cumulative mass fraction, and ``fit_msmpr`` on synthetic data;
* the kinetic steady state against an independent ``scipy.optimize.brentq`` on
  ``dC`` and its ``k_g -> infinity`` limit against the equilibrium yield;
* ``Evaporator -> Crystallizer`` in a ``Flowsheet`` (with a mother-liquor recycle);
* ``check_grads`` of the yield and the size functions.

The solubility curves and kinetic constants below are SYNTHETIC (illustrative, with
Na2SO4-like molar masses so the numbers have a plausible size). Not reproduced: the
Geankoplis Sec. 12.11, McCabe Ch. 27 and Wankat Ch. 17 worked examples, the sucrose,
Na2SO4 and Na2CO3 solubility data and the 32.4 C decahydrate transition. The issue
calls those numbers "unverified, from recollection" and none could be recalled with
confidence, so no test claims agreement with a published example.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy import integrate, optimize

from difflow import (
    Crystallizer,
    CrystallizerParams,
    Solubility,
    crystal_balance,
    crystal_yield,
    fit_msmpr,
    growth_rate,
    make_stream,
    msmpr_distribution,
    msmpr_dominant_size,
    msmpr_from_magma,
    msmpr_mass_cv,
    msmpr_mass_median_size,
    msmpr_moments,
    msmpr_steady_state,
    nucleation_rate,
    supersaturation,
)

K0 = 273.15
M_S, M_W = 142.0, 18.015


def hydrate_sol(nu=10.0):
    """Synthetic: c_sat = 0.35 + 0.0075 (T - 273.15) kg/kg."""
    return Solubility.polynomial([0.35, 0.0075], hydrate_waters=nu, solute_MW=M_S, solvent_MW=M_W)


def numpy_balance(S0, W, c, nu):
    """Independent solve of the three balances for (C, W_L, S_L)."""
    fw = nu * M_W / (M_S + nu * M_W)
    fs = 1.0 - fw
    A = np.array([[fs, 0.0, 1.0],          # S0 = fs C + S_L
                  [fw, 1.0, 0.0],          # W  = fw C + W_L
                  [0.0, c, -1.0]])         # S_L = c W_L
    return np.linalg.solve(A, np.array([S0, W, 0.0]))


# =============================================================================
# Solubility and supersaturation
# =============================================================================


class TestSolubility:
    def test_polynomial_and_per_100g(self):
        s = Solubility.polynomial([10.0, 0.5], per_100g=True)
        assert float(s.c_sat(K0 + 20.0)) == pytest.approx(0.20)
        assert float(s.per_100g(K0 + 20.0)) == pytest.approx(20.0)

    def test_van_hoff(self):
        A, B = 2.0, -1000.0
        s = Solubility.vant_hoff(A, B)
        assert float(s.c_sat(300.0)) == pytest.approx(math.exp(A + B / 300.0))
        s100 = Solubility.vant_hoff(A, B, per_100g=True)
        assert float(s100.c_sat(300.0)) == pytest.approx(0.01 * math.exp(A + B / 300.0))

    def test_table_interpolates_and_is_monotone(self):
        T = np.array([280.0, 290.0, 300.0, 320.0])
        c = np.array([0.10, 0.15, 0.25, 0.60])
        s = Solubility.table(T, c)
        assert np.allclose(np.asarray(s.c_sat(T)), c)
        grid = np.asarray(s.c_sat(np.linspace(280, 320, 200)))
        assert np.all(np.diff(grid) > 0)

    def test_c_sat_floor_keeps_it_positive(self):
        s = Solubility.polynomial([0.1, -0.01])
        assert float(s.c_sat(K0 + 50.0)) > 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            Solubility("bogus", coeffs=[1.0])
        with pytest.raises(ValueError):
            Solubility("polynomial")
        with pytest.raises(ValueError):
            Solubility.polynomial([0.1], upper=Solubility.polynomial([0.2]))

    def test_supersaturation_kinds(self):
        s = Solubility.polynomial([0.2])
        assert float(supersaturation(0.3, 300.0, s)) == pytest.approx(1.5)
        assert float(supersaturation(0.3, 300.0, s, "absolute")) == pytest.approx(0.1)
        assert float(supersaturation(0.3, 300.0, s, "relative")) == pytest.approx(0.5)
        with pytest.raises(ValueError):
            supersaturation(0.3, 300.0, s, "bogus")

    def test_is_a_pytree_and_jits(self):
        s = hydrate_sol()
        leaves, tree = jax.tree_util.tree_flatten(s)
        s2 = jax.tree_util.tree_unflatten(tree, leaves)
        assert float(s2.c_sat(300.0)) == pytest.approx(float(s.c_sat(300.0)))
        f = jax.jit(lambda sol, T: sol.c_sat(T))
        assert float(f(s, 300.0)) == pytest.approx(float(s.c_sat(300.0)))

    def test_crystal_fractions(self):
        fs, fw = hydrate_sol(10.0).crystal_fractions(300.0)
        assert float(fs) == pytest.approx(M_S / (M_S + 10 * M_W))
        assert float(fs + fw) == pytest.approx(1.0)
        fs0, fw0 = hydrate_sol(0.0).crystal_fractions(300.0)
        assert (float(fs0), float(fw0)) == (1.0, 0.0)


class TestTransition:
    """Two solid forms with a smooth logistic switch in T (never a Python ``if`` on T)."""

    def sol(self):
        upper = Solubility.polynomial([0.45, 0.002], hydrate_waters=0.0, solute_MW=M_S, solvent_MW=M_W)
        return Solubility.polynomial([0.05, 0.015], hydrate_waters=10.0, solute_MW=M_S, solvent_MW=M_W,
                                     upper=upper, T_transition=K0 + 32.0, width=0.5)

    def test_limits_are_the_two_forms(self):
        s = self.sol()
        lo, hi = K0 + 20.0, K0 + 45.0
        assert float(s.c_sat(lo)) == pytest.approx(0.05 + 0.015 * 20.0, abs=1e-9)
        assert float(s.c_sat(hi)) == pytest.approx(0.45 + 0.002 * 45.0, abs=1e-9)
        assert float(s.hydration(lo)) == pytest.approx(10.0, abs=1e-9)
        assert float(s.hydration(hi)) == pytest.approx(0.0, abs=1e-9)

    def test_midpoint_is_the_average(self):
        s = self.sol()
        T = K0 + 32.0
        lo = 0.05 + 0.015 * 32.0
        hi = 0.45 + 0.002 * 32.0
        assert float(s.c_sat(T)) == pytest.approx(0.5 * (lo + hi))
        assert float(s.hydration(T)) == pytest.approx(5.0)

    def test_traced_T_and_finite_gradient_through_the_switch(self):
        s = self.sol()
        f = jax.jit(lambda T: crystal_yield(0.3, 1.0, T, s))
        g = jax.jit(jax.grad(lambda T: crystal_yield(0.3, 1.0, T, s)))
        for T in (K0 + 25.0, K0 + 31.9, K0 + 32.0, K0 + 32.3, K0 + 40.0):
            assert np.isfinite(float(f(T))) and np.isfinite(float(g(T)))

    def test_balances_close_across_the_transition(self):
        s = self.sol()
        for T in (K0 + 25.0, K0 + 32.0, K0 + 40.0):
            b = crystal_balance(0.3, 1.0, T, s)
            assert float(b["crystal_solute"] + b["liquor_solute"]) == pytest.approx(0.3, abs=1e-14)
            assert float(b["crystal_water"] + b["liquor_water"]) == pytest.approx(1.0, abs=1e-14)


# =============================================================================
# Yield
# =============================================================================


class TestYield:
    def test_anhydrous_round_number(self):
        s = Solubility.polynomial([0.2], solute_MW=100.0)
        assert float(crystal_yield(0.3, 1.0, 300.0, s)) == pytest.approx(0.1, abs=1e-9)

    def test_hydrate_hand_worked(self):
        """M_s = 100, nu M_w = 90 (nu = 5, M_w = 18): C = (S0 - cW)/(f_s - c f_w)."""
        s = Solubility.polynomial([0.2], hydrate_waters=5.0, solute_MW=100.0, solvent_MW=18.0)
        fs, fw = 100.0 / 190.0, 90.0 / 190.0
        expect = (0.3 - 0.2 * 1.0) / (fs - 0.2 * fw)
        assert expect == pytest.approx(0.2317073170731707)
        assert float(crystal_yield(0.3, 1.0, 300.0, s)) == pytest.approx(expect, abs=1e-9)

    @pytest.mark.parametrize("nu", [0.0, 5.0, 7.0, 10.0])
    @pytest.mark.parametrize("E", [0.0, 0.2])
    def test_closes_both_balances_to_1e_12_and_matches_numpy(self, nu, E):
        S0, W0, T = 0.50, 1.0, K0 + 5.0
        s = hydrate_sol(nu)
        b = crystal_balance(S0, W0, T, s, E)
        assert abs(float(b["crystal_solute"] + b["liquor_solute"]) - S0) < 1e-12
        assert abs(float(b["crystal_water"] + b["liquor_water"]) - (W0 - E)) < 1e-12
        assert float(b["liquor_solute"] / b["liquor_water"]) == pytest.approx(float(s.c_sat(T)), rel=1e-12)
        C, WL, SL = numpy_balance(S0, W0 - E, float(s.c_sat(T)), nu)
        assert float(b["crystals"]) == pytest.approx(C, rel=1e-5, abs=1e-9)
        assert float(b["liquor_water"]) == pytest.approx(WL, rel=1e-5)
        assert float(b["yield_fraction"]) == pytest.approx((S0 - SL) / S0, rel=1e-5)

    def test_cooling_and_evaporation_increase_the_yield(self):
        s = hydrate_sol()
        base = float(crystal_yield(0.6, 1.0, K0 + 5.0, s))
        assert base > 0.0
        assert float(crystal_yield(0.6, 1.0, K0 + 0.0, s)) > base
        assert float(crystal_yield(0.6, 1.0, K0 + 5.0, s, 0.1)) > base

    def test_undersaturated_gives_zero_yield_and_a_finite_gradient(self):
        s = hydrate_sol()
        C = crystal_yield(0.1, 1.0, K0 + 25.0, s)
        assert float(C) == pytest.approx(0.0, abs=1e-12)
        g = jax.grad(crystal_yield, argnums=(0, 1, 2, 4))(0.1, 1.0, K0 + 25.0, s, 0.0)
        assert all(np.isfinite(float(x)) for x in g)

    def test_soft_clip_is_smooth_through_saturation(self):
        s = Solubility.polynomial([0.2], solute_MW=100.0)
        xs = np.linspace(0.19, 0.21, 21)
        ys = np.array([float(crystal_yield(x, 1.0, 300.0, s)) for x in xs])
        assert np.all(np.diff(ys) >= 0)
        assert ys[0] < 1e-9 and ys[-1] == pytest.approx(0.01, abs=1e-9)

    @pytest.mark.parametrize("nu", [0.0, 10.0])
    def test_check_grads(self, nu):
        s = hydrate_sol(nu)
        f = lambda S0, W0, T, E: crystal_yield(S0, W0, T, s, E)
        check_grads(f, (0.50, 1.0, K0 + 5.0, 0.1), order=1, modes=["rev"])
        check_grads(lambda T: f(0.5, 1.0, T, 0.0), (K0 + 5.0,), order=2, modes=["rev"])


# =============================================================================
# MSMPR
# =============================================================================

N0, G, TAU = 3.0e9, 2.0e-8, 1800.0


class TestMSMPR:
    def test_moments_match_quadrature(self):
        Gt = G * TAU
        mom = np.asarray(msmpr_moments(N0, G, TAU, (0, 1, 2, 3, 4)))
        for k in range(5):
            q, _ = integrate.quad(lambda L: L ** k * float(msmpr_distribution(L, N0, G, TAU)),
                                  0.0, 60 * Gt, epsabs=0, epsrel=1e-12, limit=400)
            assert mom[k] == pytest.approx(q, rel=1e-8)
            assert mom[k] == pytest.approx(math.factorial(k) * N0 * Gt ** (k + 1), rel=1e-13)

    def test_dominant_size_is_the_argmax_of_the_mass_density(self):
        Gt = G * TAU
        res = optimize.minimize_scalar(lambda L: -L ** 3 * float(msmpr_distribution(L, N0, G, TAU)),
                                       bracket=(0.5 * Gt, 3 * Gt, 8 * Gt), tol=1e-12)
        assert float(msmpr_dominant_size(G, TAU)) == pytest.approx(res.x, rel=1e-6)
        assert float(msmpr_dominant_size(G, TAU)) == pytest.approx(3 * G * TAU)

    def test_mass_median_is_the_half_point_of_the_cumulative_mass(self):
        Gt = G * TAU
        m3 = 6.0 * N0 * Gt ** 4
        L50 = float(msmpr_mass_median_size(G, TAU))
        q, _ = integrate.quad(lambda L: L ** 3 * float(msmpr_distribution(L, N0, G, TAU)),
                              0.0, L50, epsabs=0, epsrel=1e-12)
        assert q / m3 == pytest.approx(0.5, abs=1e-10)
        assert L50 / Gt == pytest.approx(3.672, abs=1e-3)

    def test_cv_of_the_mass_distribution_is_one_half(self):
        Gt = G * TAU
        w = lambda L: L ** 3 * math.exp(-L / Gt)
        m0, _ = integrate.quad(w, 0, 80 * Gt, epsabs=0, epsrel=1e-12)
        m1, _ = integrate.quad(lambda L: L * w(L), 0, 80 * Gt, epsabs=0, epsrel=1e-12)
        m2, _ = integrate.quad(lambda L: L * L * w(L), 0, 80 * Gt, epsabs=0, epsrel=1e-12)
        mean = m1 / m0
        cv = math.sqrt(m2 / m0 - mean ** 2) / mean
        assert cv == pytest.approx(msmpr_mass_cv(), rel=1e-8)

    def test_from_magma_roundtrip(self):
        rho, kv = 1600.0, math.pi / 6
        M_T = 25.0
        n0 = msmpr_from_magma(M_T, rho, kv, G, TAU)
        m3 = float(msmpr_moments(n0, G, TAU, (3,))[0])
        assert rho * kv * m3 == pytest.approx(M_T, rel=1e-12)

    def test_fit_recovers_G_tau_and_n0(self):
        L = np.linspace(1e-5, 8 * G * TAU, 40)
        n = np.asarray(msmpr_distribution(L, N0, G, TAU))
        Gt, n0 = fit_msmpr(L, n)
        assert float(Gt) == pytest.approx(G * TAU, rel=1e-10)
        assert float(n0) == pytest.approx(N0, rel=1e-10)

    def test_fit_is_robust_to_noise(self):
        rng = np.random.default_rng(3)
        L = np.linspace(1e-5, 6 * G * TAU, 60)
        n = np.asarray(msmpr_distribution(L, N0, G, TAU)) * np.exp(rng.normal(0, 0.03, L.size))
        Gt, n0 = fit_msmpr(L, n)
        assert float(Gt) == pytest.approx(G * TAU, rel=0.05)
        assert float(n0) == pytest.approx(N0, rel=0.15)

    def test_growth_and_nucleation_laws(self):
        assert float(growth_rate(0.02, 1e-4, 1.5)) == pytest.approx(1e-4 * 0.02 ** 1.5)
        assert float(nucleation_rate(0.02, 30.0, 1e6, 2.0, 0.5)) == pytest.approx(1e6 * 30 ** 0.5 * 0.02 ** 2)
        assert float(growth_rate(-0.1, 1e-4, 1.5)) == 0.0
        assert np.isfinite(float(jax.grad(growth_rate)(0.0, 1e-4, 0.5)))

    def test_check_grads_of_sizes(self):
        f = lambda x: msmpr_mass_median_size(jnp.exp(x[0]), jnp.exp(x[1]))
        check_grads(f, (jnp.array([-18.0, 7.0]),), order=2, modes=["rev"])


# =============================================================================
# Kinetic steady state
# =============================================================================

SOL0 = Solubility.polynomial([0.35, 0.0075], hydrate_waters=10.0, solute_MW=M_S, solvent_MW=M_W)
KIN = (2.0e-6, 1.5, 1.0e9, 2.0, 0.6)     # synthetic k_g, g, k_b, b, j
RHO_C, KV, RHO_L = 1460.0, math.pi / 6, 1250.0


def steady(tau=900.0, kin=KIN, T=K0 + 5.0, S0=0.60, W0=1.0, E=0.0, sol=SOL0):
    return msmpr_steady_state(S0, W0, T, sol, kin, tau, crystal_density=RHO_C,
                              shape_factor=KV, liquor_density=RHO_L, water_evaporated=E)


def brentq_delta_C(tau, kin, T, S0, W0, E, sol):
    """The same closure solved independently: root of the third-moment mismatch in dC."""
    k_g, g, k_b, b, j = kin
    W = W0 - E
    cs = float(sol.c_sat(T))
    fs, fw = (float(x) for x in sol.crystal_fractions(T))

    def mismatch(dC):
        c = cs + dC
        C = (S0 - c * W) / (fs - c * fw)
        WL = W - fw * C
        M_T = C / ((WL + c * WL) / RHO_L)
        return M_T - RHO_C * KV * 6.0 * (k_b * M_T ** j * dC ** b / (k_g * dC ** g)) * (k_g * dC ** g * tau) ** 4

    hi = S0 / W - cs
    return optimize.brentq(mismatch, hi * 1e-14, hi * (1 - 1e-12), xtol=1e-300, rtol=1e-14, maxiter=500)


class TestKinetic:
    def test_solute_balance_closes_and_dC_positive(self):
        r = steady()
        assert float(r["delta_C"]) > 0.0
        assert abs(float(r["crystal_solute"] + r["liquor_solute"]) - 0.60) < 1e-12
        assert abs(float(r["crystal_water"] + r["liquor_water"]) - 1.0) < 1e-12
        assert float(r["c_liquor"]) == pytest.approx(float(r["c_sat"] + r["delta_C"]), rel=1e-14)
        assert float(r["c_liquor"]) < 0.60

    def test_third_moment_closes_the_magma_density(self):
        r = steady()
        assert abs(float(r["closure"])) < 1e-10
        m3 = float(r["moments"][3])
        assert RHO_C * KV * m3 == pytest.approx(float(r["M_T"]), rel=1e-10)
        assert float(r["n0"]) == pytest.approx(float(r["B0"] / r["G"]), rel=1e-14)
        assert float(r["B0"]) == pytest.approx(
            float(nucleation_rate(r["delta_C"], r["M_T"], KIN[2], KIN[3], KIN[4])), rel=1e-12)

    @pytest.mark.parametrize("tau", [300.0, 900.0, 3600.0])
    def test_matches_an_independent_root_find(self, tau):
        r = steady(tau)
        dC = brentq_delta_C(tau, KIN, K0 + 5.0, 0.60, 1.0, 0.0, SOL0)
        assert float(r["delta_C"]) == pytest.approx(dC, rel=1e-8)

    def test_sizes(self):
        r = steady()
        assert float(r["L_D"]) == pytest.approx(3 * float(r["G"]) * 900.0, rel=1e-14)
        assert float(r["L_50"]) / float(r["L_D"]) == pytest.approx(3.672060748850896 / 3.0, rel=1e-9)
        assert float(r["cv"]) == 0.5

    def test_residence_time_is_not_3_G_tau_at_fixed_G(self):
        """G falls as tau grows (dC falls), so L_D(tau) < 3 G(tau0) tau."""
        a, b = steady(600.0), steady(2400.0)
        assert float(b["G"]) < float(a["G"])
        assert float(b["L_D"]) < 3 * float(a["G"]) * 2400.0

    def test_equilibrium_limit_as_k_g_grows(self):
        eq = float(crystal_yield(0.60, 1.0, K0 + 5.0, SOL0))
        errs = []
        for kg in (1e-6, 1e-4, 1e-2, 1.0, 1e3, 1e6, 1e12):
            kin = (kg,) + KIN[1:]
            errs.append(abs(float(steady(kin=kin)["crystals"]) - eq) / eq)
        assert all(b < a for a, b in zip(errs, errs[1:]))
        assert errs[-1] < 1e-8
        r = steady(kin=(1e12,) + KIN[1:])
        assert float(r["crystals"]) == pytest.approx(eq, rel=1e-8)
        assert float(r["delta_C"]) < 1e-6

    def test_undersaturated_feed_is_inactive(self):
        r = steady(S0=0.2)
        assert not bool(r["active"])
        assert float(r["crystals"]) == pytest.approx(0.0, abs=1e-12)
        assert float(r["G"]) == 0.0
        g = jax.grad(lambda t: steady(tau=t, S0=0.2)["crystals"])(900.0)
        assert np.isfinite(float(g))

    def test_j_above_one_is_rejected(self):
        with pytest.raises(ValueError):
            steady(kin=KIN[:4] + (1.5,))

    @pytest.mark.parametrize("key", ["L_D", "L_50"])
    def test_check_grads_of_size_wrt_tau_kg_kb(self, key):
        def f(x):
            kin = (jnp.exp(x[1]), KIN[1], jnp.exp(x[2]), KIN[3], KIN[4])
            return jnp.log(steady(tau=jnp.exp(x[0]), kin=kin)[key])

        x0 = jnp.array([math.log(900.0), math.log(KIN[0]), math.log(KIN[2])])
        check_grads(f, (x0,), order=1, modes=["rev"], atol=1e-6, rtol=1e-6)

    def test_size_versus_tau_gradient_matches_finite_difference(self):
        f = lambda t: steady(tau=t)["L_50"]
        g = float(jax.grad(f)(900.0))
        h = 1e-3 * 900.0
        fd = (float(f(900.0 + h)) - float(f(900.0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-6)
        # a longer residence time always gives a larger median here
        assert g > 0.0


# =============================================================================
# The unit
# =============================================================================


def feed_stream(S0=0.60, W0=1.0, T=K0 + 60.0, solvent="H2O", solute="Na2SO4"):
    return make_stream({solute: S0 * 1000.0 / M_S, solvent: W0 * 1000.0 / M_W}, T=T, P=1.0e5)


def params(**kw):
    base = dict(solute="Na2SO4", solute_MW=M_S, mode="cooling", T=K0 + 5.0,
                solubility_coeffs=[0.35, 0.0075], hydrate_waters=10.0)
    base.update(kw)
    return CrystallizerParams(**base)


class TestCrystallizerUnit:
    def test_equilibrium_streams_close_species_by_species(self):
        feed = feed_stream()
        cry, liq, info = Crystallizer(params())(feed)
        for k in ("F_Na2SO4", "F_H2O"):
            assert float(cry[k] + liq[k]) == pytest.approx(float(feed[k]), rel=1e-12)
        assert float(cry["F_H2O"] / cry["F_Na2SO4"]) == pytest.approx(10.0, rel=1e-12)
        assert cry["T"] == pytest.approx(K0 + 5.0) and liq["T"] == pytest.approx(K0 + 5.0)
        # the liquor is saturated
        c = float(liq["F_Na2SO4"] * M_S / (liq["F_H2O"] * M_W))
        assert c == pytest.approx(float(SOL0.c_sat(K0 + 5.0)), rel=1e-12)
        assert float(info["yield_kg_s"]) == pytest.approx(float(cry["F_Na2SO4"]) * M_S / 1000.0)
        assert abs(float(info["solute_balance_error"])) < 1e-12
        assert abs(float(info["water_balance_error"])) < 1e-12
        assert float(info["supersaturation"]) == pytest.approx(1.0, rel=1e-12)

    def test_matches_the_library(self):
        cry, liq, info = Crystallizer(params())(feed_stream())
        ref = crystal_balance(0.60, 1.0, K0 + 5.0, SOL0)
        assert float(info["crystal_mass"]) == pytest.approx(float(ref["crystals"]), rel=1e-12)

    def test_evaporative_mode_defaults_T_to_the_feed(self):
        p = params(mode="evaporative", T=None, water_evaporated=0.3)
        cry, liq, info = Crystallizer(p)(feed_stream(S0=0.3, T=K0 + 25.0))
        assert float(cry["T"]) == pytest.approx(K0 + 25.0)
        ref = crystal_balance(0.3, 1.0, K0 + 25.0, SOL0, 0.3)
        assert float(info["crystal_mass"]) == pytest.approx(float(ref["crystals"]), rel=1e-12)
        assert float(info["water_evaporated"]) == pytest.approx(0.3)
        # evaporated water leaves the unit: mole balance in = out + vapor
        out_w = float(cry["F_H2O"] + liq["F_H2O"]) + 0.3 * 1000.0 / M_W
        assert out_w == pytest.approx(float(feed_stream(S0=0.3)["F_H2O"]), rel=1e-12)

    def test_evaporation_fraction_equals_water_evaporated(self):
        a = Crystallizer(params(mode="cooling_evaporative", evaporation_fraction=0.25))(feed_stream())[2]
        b = Crystallizer(params(mode="cooling_evaporative", water_evaporated=0.25))(feed_stream())[2]
        assert float(a["crystal_mass"]) == pytest.approx(float(b["crystal_mass"]), rel=1e-14)

    def test_duty_terms(self):
        p = params(mode="cooling_evaporative", water_evaporated=0.1, cp_solution=3500.0,
                   heat_of_crystallization=1.0e5)
        _, _, info = Crystallizer(p)(feed_stream())
        m_F = 0.60 + 1.0
        assert float(info["Q_sensible"]) == pytest.approx(m_F * 3500.0 * (-55.0))
        assert float(info["Q_crystallization"]) == pytest.approx(-1.0e5 * float(info["crystal_mass"]))
        assert float(info["Q_evaporation"]) > 0.0
        assert float(info["Q"]) == pytest.approx(float(info["Q_sensible"] + info["Q_evaporation"]
                                                       + info["Q_crystallization"]))

    def test_undersaturated_feed_has_no_crystals(self):
        cry, liq, info = Crystallizer(params(T=K0 + 60.0))(feed_stream(S0=0.1))
        assert float(info["yield_kg_s"]) == pytest.approx(0.0, abs=1e-12)
        assert float(liq["F_Na2SO4"]) == pytest.approx(float(feed_stream(S0=0.1)["F_Na2SO4"]))

    def test_other_species_stay_in_the_liquor(self):
        feed = dict(feed_stream())
        feed["F_NaCl"] = jnp.asarray(5.0)
        cry, liq, _ = Crystallizer(params())(feed)
        assert float(cry["F_NaCl"]) == 0.0 and float(liq["F_NaCl"]) == 5.0

    def test_kinetic_mode_in_the_unit(self):
        p = params(kinetics=list(KIN), residence_time=900.0, crystal_density=RHO_C,
                   liquor_density=RHO_L)
        cry, liq, info = Crystallizer(p)(feed_stream())
        ref = steady(900.0)
        assert float(info["delta_C"]) == pytest.approx(float(ref["delta_C"]), rel=1e-10)
        assert float(info["L_D"]) == pytest.approx(float(ref["L_D"]), rel=1e-10)
        assert float(info["delta_C"]) > 0.0
        assert float(info["supersaturation"]) > 1.0
        assert float(cry["F_Na2SO4"] + liq["F_Na2SO4"]) == pytest.approx(float(feed_stream()["F_Na2SO4"]),
                                                                           rel=1e-12)

    def test_volume_sets_the_residence_time(self):
        feed = feed_stream()
        Q = (0.60 + 1.0) / RHO_L
        p = params(kinetics=list(KIN), volume=900.0 * Q, crystal_density=RHO_C, liquor_density=RHO_L)
        _, _, info = Crystallizer(p)(feed)
        assert float(info["tau"]) == pytest.approx(900.0, rel=1e-12)

    def test_volume_uses_the_product_flow_in_evaporative_mode(self):
        """MSMPR tau = V / Q_product: the evaporated solvent does not pass through the magma."""
        feed = feed_stream()
        E = 0.4
        Q_product = (0.60 + 1.0 - E) / RHO_L
        p = params(kinetics=list(KIN), volume=900.0 * Q_product, crystal_density=RHO_C, liquor_density=RHO_L,
                   mode="cooling_evaporative", water_evaporated=E)
        _, _, info = Crystallizer(p)(feed)
        assert float(info["tau"]) == pytest.approx(900.0, rel=1e-12)

    def test_unit_is_differentiable_in_T_and_residence_time(self):
        def yield_T(T):
            return Crystallizer(params(T=T))(feed_stream())[2]["yield_kg_s"]

        check_grads(yield_T, (K0 + 5.0,), order=1, modes=["rev"])
        assert float(jax.grad(yield_T)(K0 + 5.0)) < 0.0

        def L_D(t):
            p = params(kinetics=list(KIN), residence_time=t, crystal_density=RHO_C, liquor_density=RHO_L)
            return Crystallizer(p)(feed_stream())[2]["L_50"]

        assert np.isfinite(float(jax.grad(L_D)(900.0)))

    def test_kinetic_mode_under_jit(self):
        """Regression: float(j) on jnp.asarray(constant) raised ConcretizationTypeError under jit."""
        def L50(t):
            p = params(kinetics=list(KIN), residence_time=t, crystal_density=RHO_C, liquor_density=RHO_L)
            return Crystallizer(p)(feed_stream())[2]["L_50"]

        assert float(jax.jit(L50)(900.0)) == pytest.approx(float(L50(900.0)), rel=1e-12)
        assert float(jax.jit(jax.grad(L50))(900.0)) == pytest.approx(float(jax.grad(L50)(900.0)), rel=1e-10)

    def test_gradient_with_respect_to_kinetics(self):
        """Regression: CrystallizerParams validated j with float(), failing for traced kinetics."""
        def lnL50(logk):   # log of (k_g, g, k_b, b); j = 1 held exactly
            p = params(kinetics=jnp.append(jnp.exp(logk), 1.0), residence_time=900.0, crystal_density=RHO_C,
                       liquor_density=RHO_L)
            return jnp.log(Crystallizer(p)(feed_stream())[2]["L_50"])

        k_g, gg, k_b, b, _ = KIN
        logk = jnp.log(jnp.array([k_g, gg, k_b, b]))   # j = 1: M_T cancels from the closure
        g = jax.jit(jax.grad(lnL50))(logk)
        # Delta C^(b+3g) ~ 1/(k_b k_g^3 tau^4) and L50 = 3.672 k_g Delta C^g tau
        assert float(g[0]) == pytest.approx(b / (b + 3 * gg), rel=1e-6)
        assert float(g[2]) == pytest.approx(-gg / (b + 3 * gg), rel=1e-6)
        check_grads(lnL50, (logk,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)

    def test_jit(self):
        f = jax.jit(lambda feed: Crystallizer(params())(feed)[2]["yield_kg_s"])
        assert float(f(feed_stream())) == pytest.approx(
            float(Crystallizer(params())(feed_stream())[2]["yield_kg_s"]))

    @pytest.mark.parametrize("kw", [
        dict(mode="bogus"),
        dict(mode="cooling", water_evaporated=0.1),
        dict(mode="evaporative", T=None),
        dict(mode="cooling", T=None),
        dict(water_evaporated=0.1, evaporation_fraction=0.1, mode="evaporative"),
        dict(kinetics=[1, 1, 1, 1, 1]),
        dict(mode="adiabatic", water_evaporated=0.1),
        dict(mode="adiabatic", T=None),
        dict(kinetics=[1, 1, 1, 1, 2.0], residence_time=1.0),
        dict(transition_T=None, upper_solubility_coeffs=[0.4]),
    ])
    def test_parameter_validation(self, kw):
        with pytest.raises(ValueError):
            params(**kw)

    def test_solubility_from_params_includes_the_transition(self):
        p = params(upper_solubility_coeffs=[0.45, 0.002], upper_hydrate_waters=0.0,
                   transition_T=K0 + 32.4, solubility_coeffs=[0.05, 0.015])
        s = Crystallizer(p).solubility
        assert float(s.hydration(K0 + 10.0)) == pytest.approx(10.0, abs=1e-6)
        assert float(s.hydration(K0 + 50.0)) == pytest.approx(0.0, abs=1e-6)

    def test_g_per_100g_basis(self):
        p = params(solubility_coeffs=[35.0, 0.75], solubility_basis="g_per_100g")
        assert float(Crystallizer(p).solubility.c_sat(K0 + 15.0)) == pytest.approx(float(SOL0.c_sat(K0 + 15.0)))

    def test_table_model(self):
        p = params(solubility_model="table", solubility_T=[280.0, 300.0, 320.0],
                   solubility_c=[0.2, 0.4, 0.9], T=290.0)
        _, _, info = Crystallizer(p)(feed_stream(S0=0.6))
        assert float(info["c_sat"]) == pytest.approx(float(Solubility.table([280.0, 300.0, 320.0],
                                                                            [0.2, 0.4, 0.9]).c_sat(290.0)))


# =============================================================================
# Evaporator -> Crystallizer (wex20-2 structure, synthetic data)
# =============================================================================


class TestEvaporatorToCrystallizer:
    def evaporator(self, x_product):
        from difflow import Evaporator, EvaporatorParams

        return Evaporator(EvaporatorParams(
            solute_species=["Na2SO4"], solute_MW=[M_S], solvent="water", x_product=x_product,
            bpr_model="ideal", vant_hoff_i=3.0, P_vapor_space=60e3, steam_P=300e3, U=1500.0))

    def crystallizer(self):
        return Crystallizer(params(solvent="water"))

    def fresh(self):
        return make_stream({"Na2SO4": 0.30 * 1000 / M_S, "water": 1.7 * 1000 / M_W}, T=K0 + 40.0, P=1e5)

    def test_in_series(self):
        from difflow.flowsheet import Flowsheet, Unit

        fs = Flowsheet(species_order=["water", "Na2SO4"])
        fs.add_feed("feed", self.fresh())
        fs.add_unit(Unit("evap", self.evaporator(0.30), ["feed"], ["conc", "vap"]))
        fs.add_unit(Unit("xtal", self.crystallizer(), ["conc"], ["crystals", "liquor"]))
        out = fs.solve()
        feed = self.fresh()
        assert float(out["crystals"]["F_Na2SO4"] + out["liquor"]["F_Na2SO4"]) == pytest.approx(
            float(feed["F_Na2SO4"]), rel=1e-12)
        assert float(out["crystals"]["F_water"]) == pytest.approx(10.0 * float(out["crystals"]["F_Na2SO4"]),
                                                                  rel=1e-12)
        assert float(out["crystals"]["F_Na2SO4"]) > 0.0
        # evaporator water + crystallizer water (crystals + liquor) = feed water
        w = float(out["vap"]["F_water"] + out["crystals"]["F_water"] + out["liquor"]["F_water"])
        assert w == pytest.approx(float(feed["F_water"]), rel=1e-10)

    def test_with_a_mother_liquor_recycle(self):
        from difflow import Mixer
        from difflow.flowsheet import Flowsheet, Unit

        fs = Flowsheet(species_order=["water", "Na2SO4"])
        fs.add_feed("feed", self.fresh())
        fs.add_unit(Unit("mix", Mixer(["water", "Na2SO4"]), ["feed", "recycle"], ["evap_in"]))
        fs.add_unit(Unit("evap", self.evaporator(0.30), ["evap_in"], ["conc", "vap"]))
        fs.add_unit(Unit("xtal", self.crystallizer(), ["conc"], ["crystals", "liquor"]))
        fs.add_recycle("liquor", "recycle")
        guess = {"recycle": make_stream({"Na2SO4": 0.5, "water": 40.0}, T=K0 + 15.0, P=1e5)}
        out = fs.solve(tear_initial=guess, tol=1e-12, max_iter=400, on_nonconvergence="raise")
        fresh = self.fresh()
        # at steady state the crystals carry exactly the fresh solute
        assert float(out["crystals"]["F_Na2SO4"]) == pytest.approx(float(fresh["F_Na2SO4"]), rel=1e-7)
        # and the recycle loop is closed
        assert float(out["recycle"]["F_Na2SO4"]) == pytest.approx(float(out["liquor"]["F_Na2SO4"]), rel=1e-7)
        # the water leaves with the evaporator vapor, apart from the water of crystallization
        w_out = float(out["vap"]["F_water"] + out["crystals"]["F_water"])
        assert w_out == pytest.approx(float(fresh["F_water"]), rel=1e-6)


# =============================================================================
# Adiabatic (vacuum) crystallizer
# =============================================================================


class TestAdiabatic:
    def unit(self, **kw):
        return Crystallizer(params(mode="adiabatic", T=K0 + 30.0, cp_solution=3200.0, **kw))

    def test_heat_balance_closes_at_the_duty(self):
        for dH, duty in ((0.0, 0.0), (2.0e5, 0.0), (2.0e5, -5.0e4)):
            _, _, info = self.unit(heat_of_crystallization=dH, duty=duty)(feed_stream())
            assert float(info["water_evaporated"]) > 0.0
            assert float(info["Q"]) == pytest.approx(duty, abs=1e-6)

    def test_hand_flash(self):
        """With dH_c = 0: E = m_F cp (T_F - T) / h_fg(T)."""
        from difflow.water import water_latent_heat

        _, _, info = self.unit()(feed_stream())
        E = 1.6 * 3200.0 * 30.0 / float(water_latent_heat(K0 + 30.0))
        assert float(info["water_evaporated"]) == pytest.approx(E, rel=1e-12)

    def test_flashing_beats_cooling_alone(self):
        cool = Crystallizer(params(T=K0 + 30.0))(feed_stream())[2]["crystal_mass"]
        flash = self.unit()(feed_stream())[2]["crystal_mass"]
        assert float(flash) > float(cool)

    def test_balances_and_gradient(self):
        feed = feed_stream()
        cry, liq, info = self.unit(heat_of_crystallization=2.0e5)(feed)
        w_vap = float(info["water_evaporated"]) * 1000.0 / M_W
        assert float(cry["F_H2O"] + liq["F_H2O"]) + w_vap == pytest.approx(float(feed["F_H2O"]), rel=1e-12)

        def crystals(T):
            return Crystallizer(params(mode="adiabatic", T=T, cp_solution=3200.0,
                                       heat_of_crystallization=2.0e5))(feed)[2]["crystal_mass"]

        check_grads(crystals, (K0 + 30.0,), order=1, modes=["rev"])


# =============================================================================
# Screen analysis -> population density
# =============================================================================


class TestSieve:
    def test_formula(self):
        from difflow import sieve_population_density

        L, n = sieve_population_density([0.2], [1.0e-3], [0.5e-3], 120.0, 1600.0, 1.0)
        assert float(L[0]) == pytest.approx(0.75e-3)
        assert float(n[0]) == pytest.approx(120.0 * 0.2 / (1600.0 * 0.75e-3 ** 3 * 0.5e-3), rel=1e-14)

    def test_msmpr_screen_analysis_recovers_G_tau(self):
        """Mass on each screen from the exact MSMPR mass law; the fit recovers G tau to a few %."""
        from difflow import sieve_population_density

        Gt, n0, rho, kv = 2.0e-4, 1.0e12, 1600.0, 0.6
        edges = np.array([0.1, 0.15, 0.2, 0.3, 0.42, 0.6, 0.85, 1.18, 1.7]) * 1e-3

        def cum_mass(L):            # int_0^L rho kv L^3 n dL
            x = L / Gt
            return 6 * rho * kv * n0 * Gt ** 4 * (1 - np.exp(-x) * (1 + x + x ** 2 / 2 + x ** 3 / 6))

        M_T = 6 * rho * kv * n0 * Gt ** 4
        w = (cum_mass(edges[1:]) - cum_mass(edges[:-1])) / M_T
        L, n = sieve_population_density(w, edges[1:], edges[:-1], M_T, rho, kv)
        Gt_fit, n0_fit = fit_msmpr(L, n)
        assert float(Gt_fit) == pytest.approx(Gt, rel=0.05)
        assert float(n0_fit) == pytest.approx(n0, rel=0.2)


# =============================================================================
# Published worked examples (values recalled; see the docstring of the class)
# =============================================================================


class TestPublishedExamples:
    """Geankoplis, Transport Processes and Separation Process Principles, 4e/5e, Example 12.11-1.

    10 000 kg of a 30 wt% Na2CO3 solution is cooled to 293 K; the salt crystallizes as
    Na2CO3.10H2O and the solubility is 21.5 kg Na2CO3 per 100 kg water. Printed answers
    (recalled, not checked against the book): (a) no evaporation, 6370 kg of crystals;
    (b) 3 % of the solution (300 kg water) evaporated, 6630 kg. Geankoplis rounds the
    molar masses to 106.0 and 286.2, which accounts for the last digit.
    """

    sol = Solubility.polynomial([0.215], hydrate_waters=10.0, solute_MW=105.99, solvent_MW=18.015)

    @pytest.mark.parametrize("E, printed", [(0.0, 6370.0), (300.0, 6630.0)])
    def test_geankoplis_12_11_1(self, E, printed):
        C = float(crystal_yield(3000.0, 7000.0, 293.15, self.sol, water_evaporated=E))
        assert C == pytest.approx(printed, rel=3e-3)
