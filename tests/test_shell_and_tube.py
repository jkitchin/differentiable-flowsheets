"""Tests for difflow.shell_and_tube: Kern shell side, geometry, U iteration.

What "validated" means here -- please read before trusting a number.

**The Kern kerosene-crude example (Kern 1950 Ex. 7.x / Coulson & Richardson
Vol 6 Ch. 12 worked examples) is NOT reproduced.** Its inputs and printed
answers could not be recalled with confidence, and a fabricated textbook
number is worse than none; no test here asserts one. What is checked instead:

* every Kern equation (``D_e``, ``A_s``, ``G_s``, ``Re_s``, ``h_o`` and the
  shell ``dP``) and the tube-side equations (velocity, Reynolds number,
  Gnielinski ``h_i`` with a Colebrook factor, friction plus return ``dP``) are
  compared with a **plain-Python/SciPy hand calculation** of the stated formula
  (1e-9), using inputs of the same kind as the textbook example (hydrocarbon
  shell side, 19.05 mm tubes on 23.81 mm triangular pitch) but *illustrative*
  properties;
* the whole design loop (area from ``Q/(U F LMTD)``, geometry, film
  coefficients, ``overall_U``, repeat) is re-implemented independently in
  plain Python with a ``scipy`` fixed-point and compared with the
  ``optx.fixed_point`` design (1e-7);
* ``F`` is compared with an independent closed form, and the exact 1-2N
  effectiveness with ``Q = U A F LMTD`` solved by ``brentq``;
* the design (area/duty) and the rating (``rate`` and the library
  ``ShellAndTubeHX``) agree for the same geometry and U;
* gradients: ``check_grads`` of the area with respect to baffle spacing and
  tube velocity (through the ``optx.fixed_point``), and against finite
  differences;
* the Kern constants (0.36, 0.55, 0.576, 0.19), the ``Db`` constants
  (``K1``, ``n1``) and the tube-count constants were **recalled from memory**
  and are only sanity-banded here (against a brute-force tube-lattice count);
  they are NOT verified against the books.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow import ShellAndTubeHX, ShellAndTubeHXParams, make_stream
from difflow import fluids, heat_transfer as ht
from difflow import shell_and_tube as st
from difflow.units.heat_exchanger import (
    effectiveness_shell_and_tube,
    lmtd_correction_factor,
)

# illustrative hydrocarbon properties (inputs only)
DO, DI = 0.01905, 0.01483          # 3/4 in tube, 14.83 mm ID (illustrative)
PT = 1.25 * DO                      # 23.81 mm triangular pitch
SHELL = dict(rho=780.0, mu=4.0e-4, k=0.13, Cp=2200.0)     # hot, on the shell side
TUBE = dict(rho=830.0, mu=2.9e-3, k=0.13, Cp=2050.0)      # cold, in the tubes
HOT = {"T": 473.0, "m_dot": 5.0}
COLD = {"T": 303.0, "m_dot": 18.0}


def make_params(**kw):
    base = dict(
        tube_OD=DO, tube_ID=DI,
        rho_hot=SHELL["rho"], mu_hot=SHELL["mu"], k_hot=SHELL["k"], Cp_hot=SHELL["Cp"],
        rho_cold=TUBE["rho"], mu_cold=TUBE["mu"], k_cold=TUBE["k"], Cp_cold=TUBE["Cp"],
        tube_side="cold", tube_length=4.88, R_fi=1e-4, R_fo=1e-4, k_wall=45.0,
    )
    base.update(kw)
    return st.ShellAndTubeDesignParams(**base)


# ---------------------------------------------------------------------------
# independent (hand) implementations
# ---------------------------------------------------------------------------

def colebrook(Re, rel):
    if Re < 2100:
        return 64.0 / Re
    g = lambda f: 1 / math.sqrt(f) + 2 * math.log10(rel / 3.7 + 2.51 / (Re * math.sqrt(f)))
    return brentq(g, 1e-4, 0.5, xtol=1e-14, rtol=1e-14)


def hand_gnielinski_h(Re, Pr, k, D, rel):
    f = colebrook(Re, rel)
    Nu = (f / 8) * (Re - 1000) * Pr / (1 + 12.7 * math.sqrt(f / 8) * (Pr ** (2 / 3) - 1))
    return Nu * k / D


def hand_kern(m, rho, mu, k, Cp, Ds, do, Pt, lB, tri=True):
    de = 1.10 / do * (Pt**2 - 0.917 * do**2) if tri else 1.27 / do * (Pt**2 - 0.785 * do**2)
    As = (Pt - do) / Pt * Ds * lB
    Gs = m / As
    Re = Gs * de / mu
    Pr = Cp * mu / k
    Nu = 0.36 * Re**0.55 * Pr ** (1 / 3)
    return dict(de=de, As=As, Gs=Gs, us=Gs / rho, Re=Re, h=Nu * k / de)


def hand_shell_dP(m, rho, mu, Ds, do, Pt, lB, L, tri=True):
    r = hand_kern(m, rho, mu, 1.0, 1.0, Ds, do, Pt, lB, tri)
    f = math.exp(0.576 - 0.19 * math.log(r["Re"]))
    return f * (Ds / r["de"]) * (L / lB) * 0.5 * rho * r["us"] ** 2


def hand_tube_side(m, rho, mu, Nt, Np, di, L, eps=4.6e-5, K=4.0):
    v = m / (rho * math.pi / 4 * di**2 * Nt / Np)
    Re = rho * v * di / mu
    f = colebrook(Re, eps / di)
    return dict(v=v, Re=Re, f=f, dP=Np * (f * L / di + K) * 0.5 * rho * v**2)


def hand_bundle(Nt, do, Np, layout="triangular"):
    table = st.BUNDLE_CONSTANTS[layout]
    K1, n1 = table[Np]
    return do * (Nt / K1) ** (1 / n1)


def hand_F(R, P):
    S = math.sqrt(R * R + 1)
    return S / (R - 1) * math.log((1 - P) / (1 - R * P)) / math.log((2 - P * (R + 1 - S)) / (2 - P * (R + 1 + S)))


def hand_design(params, hot, cold, T_hot_out, U0=500.0, tube_side="cold", velocity=None, lB_fixed=None):
    """Plain-Python design loop: substitution on U, no JAX."""
    Th, Tc = hot["T"], cold["T"]
    Ch = hot["m_dot"] * params.Cp_hot
    Cc = cold["m_dot"] * params.Cp_cold
    Q = Ch * (Th - T_hot_out)
    Tco = Tc + Q / Cc
    dT1, dT2 = Th - Tco, T_hot_out - Tc
    lm = (dT1 - dT2) / math.log(dT1 / dT2)
    R, P = (Th - T_hot_out) / (Tco - Tc), (Tco - Tc) / (Th - Tc)
    F = hand_F(R, P)
    do, di, Np = params.tube_OD, params.tube_ID, params.n_tube_passes
    Pt = params.pitch_ratio * do
    tube = TUBE if tube_side == "cold" else SHELL
    shell = SHELL if tube_side == "cold" else TUBE
    m_t = cold["m_dot"] if tube_side == "cold" else hot["m_dot"]
    m_s = hot["m_dot"] if tube_side == "cold" else cold["m_dot"]
    U = U0
    for _ in range(500):
        A = Q / (U * F * lm)
        if velocity is None:
            L = params.tube_length
            Nt = A / (math.pi * do * L)
        else:
            Nt = Np * m_t / (tube["rho"] * velocity * math.pi / 4 * di**2)
            L = A / (math.pi * do * Nt)
        Ds = hand_bundle(Nt, do, int(Np)) + params.shell_clearance
        lB = params.baffle_spacing_ratio * Ds if lB_fixed is None else lB_fixed
        ts = hand_tube_side(m_t, tube["rho"], tube["mu"], Nt, Np, di, L)
        Pr_t = tube["Cp"] * tube["mu"] / tube["k"]
        h_i = hand_gnielinski_h(ts["Re"], Pr_t, tube["k"], di, params.roughness / di) if ts["Re"] > 4000 else None
        assert h_i is not None, "hand calculation implemented for turbulent tube side only"
        ks = hand_kern(m_s, shell["rho"], shell["mu"], shell["k"], shell["Cp"], Ds, do, Pt, lB)
        inv = (1 / ks["h"] + params.R_fo + do * math.log(do / di) / (2 * params.k_wall)
               + params.R_fi * do / di + do / (di * h_i))
        Un = 1 / inv
        if abs(Un - U) < 1e-12 * U:
            U = Un
            break
        U = 0.5 * U + 0.5 * Un
    A = Q / (U * F * lm)
    return dict(U=U, A=A, Nt=Nt, L=L, Ds=Ds, lB=lB, h_i=h_i, h_o=ks["h"], F=F, lmtd=lm, Q=Q,
                dP_t=ts["dP"], dP_s=hand_shell_dP(m_s, shell["rho"], shell["mu"], Ds, do, Pt, lB, L), v=ts["v"])


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_equivalent_diameter_hand_and_exact(self):
        tri = float(st.equivalent_diameter(PT, DO, "triangular"))
        sq = float(st.equivalent_diameter(PT, DO, "square"))
        assert tri == pytest.approx(1.10 / DO * (PT**2 - 0.917 * DO**2), rel=1e-13)
        assert sq == pytest.approx(1.27 / DO * (PT**2 - 0.785 * DO**2), rel=1e-13)
        # the constants are rounded versions of the exact pitch-cell hydraulic diameters
        tri_exact = 4 * (PT**2 * math.sqrt(3) / 4 - math.pi * DO**2 / 8) / (math.pi * DO / 2)
        sq_exact = 4 * (PT**2 - math.pi * DO**2 / 4) / (math.pi * DO)
        assert tri == pytest.approx(tri_exact, rel=0.03)   # 0.917 vs 0.9069 and 1.10 vs 1.103
        assert sq == pytest.approx(sq_exact, rel=0.005)

    @pytest.mark.parametrize("layout", ["triangular", "square"])
    @pytest.mark.parametrize("Db", [0.4, 0.8, 1.2])
    def test_tube_counts_sanity_band_vs_lattice(self, layout, Db):
        """Sanity band vs a brute-force lattice count (one tube-centred lattice; the count
        depends on the alignment by a few percent, and ignores pass lanes)."""
        R = Db / 2 - DO / 2
        n = int(R / PT) + 3
        cnt = 0
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                if layout == "square":
                    x, y = PT * i, PT * j
                else:
                    x, y = PT * (i + 0.5 * j), PT * math.sqrt(3) / 2 * j
                cnt += math.hypot(x, y) <= R
        kak = float(st.tube_count(Db, DO, PT, layout, 2.0))
        assert 0.8 * cnt < kak < 1.15 * cnt
        # bundle constants (pitch 1.25 d_o): the Nt of a bundle of diameter Db
        K1, n1 = st.BUNDLE_CONSTANTS[layout][2]
        Nt_b = K1 * (Db / DO) ** n1
        assert 0.65 * cnt < Nt_b < 1.15 * cnt

    def test_bundle_diameter_hand_and_roundtrip(self):
        for layout in st.LAYOUTS:
            for Np in (1, 2, 4, 6, 8):
                for Nt in (50.0, 400.0):
                    assert float(st.bundle_diameter(Nt, DO, Np, layout)) == pytest.approx(
                        hand_bundle(Nt, DO, Np, layout), rel=1e-12)

    def test_bundle_diameter_continuous_in_passes_and_differentiable(self):
        f = lambda n: st.bundle_diameter(300.0, DO, n, "triangular")
        # interpolation hits the table rows and lies between them
        assert float(f(2.0)) == pytest.approx(hand_bundle(300.0, DO, 2), rel=1e-12)
        mid = float(f(3.0))
        assert min(float(f(2.0)), float(f(4.0))) - 1e-9 <= mid <= max(float(f(2.0)), float(f(4.0))) + 1e-9
        g = float(jax.grad(f)(3.0))
        fd = (float(f(3.0 + 1e-6)) - float(f(3.0 - 1e-6))) / 2e-6
        assert g == pytest.approx(fd, rel=1e-6)

    def test_tube_count_integer_and_ctp(self):
        assert float(st.tube_count(0.5, DO, PT, "triangular", 2.0, integer=True)) == math.floor(
            float(st.tube_count(0.5, DO, PT, "triangular", 2.0)))
        hand = 0.90 * math.pi / 4 * 0.5**2 / (0.866 * PT**2)
        assert float(st.tube_count(0.5, DO, PT, "triangular", 2.0)) == pytest.approx(hand, rel=1e-13)

    def test_bwg_inside_diameter(self):
        # 3/4 in BWG 16: wall 0.065 in -> ID 0.620 in (standard tube table, from memory)
        assert st.tube_inside_diameter(0.75 * 0.0254, 16) == pytest.approx(0.620 * 0.0254, rel=1e-12)
        with pytest.raises(KeyError, match="BWG"):
            st.tube_inside_diameter(0.0254, 99)

    def test_bad_layout(self):
        with pytest.raises(ValueError):
            st.equivalent_diameter(PT, DO, "hexagonal")


# ---------------------------------------------------------------------------
# Kern shell side and tube side vs hand calculation
# ---------------------------------------------------------------------------

class TestKern:
    @pytest.mark.parametrize("layout", ["triangular", "square"])
    @pytest.mark.parametrize("lB", [0.1, 0.25])
    def test_shell_side_hand_calc(self, layout, lB):
        Ds, m = 0.45, 5.0
        r = st.kern_shell_side(m, **SHELL, shell_ID=Ds, tube_OD=DO, pitch=PT, baffle_spacing=lB, layout=layout)
        h = hand_kern(m, SHELL["rho"], SHELL["mu"], SHELL["k"], SHELL["Cp"], Ds, DO, PT, lB, layout == "triangular")
        for key, hk in [("De", "de"), ("As", "As"), ("Gs", "Gs"), ("us", "us"), ("Re", "Re"), ("h_o", "h")]:
            assert float(r[key]) == pytest.approx(h[hk], rel=1e-12), key

    def test_wall_viscosity_correction(self):
        a = st.kern_shell_side(5.0, **SHELL, shell_ID=0.45, tube_OD=DO, pitch=PT, baffle_spacing=0.2)
        b = st.kern_shell_side(5.0, **SHELL, shell_ID=0.45, tube_OD=DO, pitch=PT, baffle_spacing=0.2,
                               mu_wall=2 * SHELL["mu"])
        assert float(b["h_o"] / a["h_o"]) == pytest.approx(0.5**0.14, rel=1e-13)

    def test_shell_pressure_drop_hand_calc(self):
        Ds, lB, L = 0.45, 0.15, 4.88
        r = st.kern_shell_pressure_drop(5.0, SHELL["rho"], SHELL["mu"], Ds, DO, PT, lB, L)
        assert float(r["dP"]) == pytest.approx(hand_shell_dP(5.0, SHELL["rho"], SHELL["mu"], Ds, DO, PT, lB, L), rel=1e-12)
        assert float(r["n_crossings"]) == pytest.approx(L / lB)
        # the 8 j_f form of Coulson & Richardson with 8 jf = f
        assert float(r["f"]) == pytest.approx(math.exp(0.576 - 0.19 * math.log(float(r["Re"]))), rel=1e-13)

    def test_shell_trends(self):
        """h_o rises and dP rises sharply as the baffles close up."""
        h = [float(st.kern_shell_side(5.0, **SHELL, shell_ID=0.45, tube_OD=DO, pitch=PT, baffle_spacing=lB)["h_o"])
             for lB in (0.4, 0.2, 0.1)]
        dp = [float(st.kern_shell_pressure_drop(5.0, SHELL["rho"], SHELL["mu"], 0.45, DO, PT, lB, 4.88)["dP"])
              for lB in (0.4, 0.2, 0.1)]
        assert h[0] < h[1] < h[2]
        assert dp[0] < dp[1] < dp[2]
        assert dp[2] / dp[1] > 4.0     # ~ lB^-2.81

    def test_tube_side_hand_calc(self):
        Nt, Np, L, m = 120.0, 2.0, 4.88, 18.0
        r = st.tube_side_pressure_drop(m, TUBE["rho"], TUBE["mu"], Nt, Np, DI, L)
        h = hand_tube_side(m, TUBE["rho"], TUBE["mu"], Nt, Np, DI, L)
        assert float(r["v"]) == pytest.approx(h["v"], rel=1e-12)
        assert float(r["Re"]) == pytest.approx(h["Re"], rel=1e-12)
        assert float(r["f"]) == pytest.approx(h["f"], rel=1e-6)    # Newton Colebrook vs brentq
        assert float(r["dP"]) == pytest.approx(h["dP"], rel=1e-6)
        assert float(r["dP_return"]) == pytest.approx(Np * 4.0 * 0.5 * TUBE["rho"] * h["v"] ** 2, rel=1e-12)
        # uses the library friction factor
        assert float(r["f"]) == pytest.approx(float(fluids.friction_factor(r["Re"], 4.6e-5 / DI)), rel=1e-14)

    def test_tube_side_h_hand_calc(self):
        Nt, Np, L, m = 120.0, 2.0, 4.88, 18.0
        h = hand_tube_side(m, TUBE["rho"], TUBE["mu"], Nt, Np, DI, L)
        Pr = TUBE["Cp"] * TUBE["mu"] / TUBE["k"]
        assert h["Re"] > 4000
        h_hand = hand_gnielinski_h(h["Re"], Pr, TUBE["k"], DI, 4.6e-5 / DI)
        h_lib = float(ht.internal_h(h["Re"], Pr, TUBE["k"], DI, L=L * Np, rel_roughness=4.6e-5 / DI))
        assert h_lib == pytest.approx(h_hand, rel=1e-6)

    def test_grads(self):
        f = lambda lB: st.kern_shell_side(5.0, **SHELL, shell_ID=0.45, tube_OD=DO, pitch=PT, baffle_spacing=lB)["h_o"]
        g = lambda lB: st.kern_shell_pressure_drop(5.0, SHELL["rho"], SHELL["mu"], 0.45, DO, PT, lB, 4.88)["dP"]
        check_grads(f, (0.2,), order=2, modes=["rev"])
        check_grads(g, (0.2,), order=2, modes=["rev"])
        t = lambda Nt: st.tube_side_pressure_drop(18.0, TUBE["rho"], TUBE["mu"], Nt, 2.0, DI, 4.88)["dP"]
        check_grads(t, (120.0,), order=1, modes=["rev"], rtol=1e-4)


# ---------------------------------------------------------------------------
# F factor and exact 1-2N effectiveness (library fixes made for #403)
# ---------------------------------------------------------------------------

class TestFAndEffectiveness:
    @pytest.mark.parametrize("R,P", [(0.5, 0.3), (2.0, 0.3), (3.35, 0.175), (0.298, 0.58), (1.6, 0.25)])
    def test_F_matches_closed_form_away_from_R1(self, R, P):
        """Regression: lmtd_correction_factor returned the R = 1 value for every |R-1| > 0.075."""
        assert float(lmtd_correction_factor(R, P, 1)) == pytest.approx(hand_F(R, P), rel=1e-12)

    def test_F_continuous_through_R1(self):
        vals = [float(lmtd_correction_factor(R, 0.4, 1)) for R in np.linspace(0.9, 1.1, 41)]
        assert np.max(np.abs(np.diff(vals))) < 0.01   # no jump where the blend ends
        assert float(lmtd_correction_factor(1.2, 0.4, 1)) == pytest.approx(hand_F(1.2, 0.4), rel=1e-12)

    @pytest.mark.parametrize("Cr", [0.0, 0.3, 0.8, 1.0])
    @pytest.mark.parametrize("NTU", [0.3, 1.0, 3.0])
    def test_effectiveness_vs_UA_F_LMTD(self, Cr, NTU):
        """eps(NTU, Cr) reproduces Q = UA F LMTD solved independently by brentq."""
        if Cr == 0.0:
            assert float(effectiveness_shell_and_tube(NTU, 1e-12, 1)) == pytest.approx(1 - math.exp(-NTU), rel=1e-9)
            return
        Cmin, Cmax = 1000.0, 1000.0 / Cr
        Th, Tc = 400.0, 300.0
        UA = NTU * Cmin
        Cr_ = Cr

        def resid(Q):
            Tho, Tco = Th - Q / Cmin, Tc + Q / Cmax          # hot side is the Cmin side
            R, P = (Th - Tho) / (Tco - Tc), (Tco - Tc) / (Th - Tc)
            try:
                if abs(R - 1) < 1e-9:   # R = 1 limit
                    s2 = math.sqrt(2)
                    F = P * s2 / ((1 - P) * math.log((2 - P * (2 - s2)) / (2 - P * (2 + s2))))
                else:
                    F = hand_F(R, P)
                d1, d2 = Th - Tco, Tho - Tc
                lm = d1 if abs(d1 - d2) < 1e-9 else (d1 - d2) / math.log(d1 / d2)
            except (ValueError, ZeroDivisionError):   # beyond the feasible region: Q too large
                return -Q
            return UA * F * lm - Q

        Q = brentq(resid, 1e-3, Cmin * (Th - Tc) * 0.999999, xtol=1e-12)
        eps = float(effectiveness_shell_and_tube(NTU, Cr_, 1))
        assert eps * Cmin * (Th - Tc) == pytest.approx(Q, rel=1e-8)

    def test_two_shells_between_one_shell_and_counter_current(self):
        from difflow.units.heat_exchanger import effectiveness_counter_current
        NTU, Cr = 2.0, 0.7
        e1 = float(effectiveness_shell_and_tube(NTU, Cr, 1))
        e2 = float(effectiveness_shell_and_tube(NTU, Cr, 2))
        ecc = float(effectiveness_counter_current(NTU, Cr))
        assert e1 < e2 < ecc
        # Cr = 1 branch is continuous with the general one
        assert float(effectiveness_shell_and_tube(NTU, 1.0, 2)) == pytest.approx(
            float(effectiveness_shell_and_tube(NTU, 1.0 - 1e-6, 2)), rel=1e-5)

    def test_ShellAndTubeHX_now_satisfies_UA_F_LMTD(self):
        """ShellAndTubeHX used Q = F * Q_cc(UA), a few percent off Q = UA F LMTD."""
        MW = 0.018
        UA = 11000.0
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=UA, Cp_hot=2200 * MW, Cp_cold=2050 * MW))
        ho, co, info = hx(make_stream({"A": 5.0 / MW}, 473.0, 101325.0), make_stream({"A": 18.0 / MW}, 303.0, 101325.0))
        d1, d2 = 473.0 - float(co["T"]), float(ho["T"]) - 303.0
        lm = (d1 - d2) / math.log(d1 / d2)
        assert UA * float(info["F_correction"]) * lm == pytest.approx(float(info["Q"]), rel=1e-9)
        R = (473.0 - float(ho["T"])) / (float(co["T"]) - 303.0)
        P = (float(co["T"]) - 303.0) / 170.0
        assert float(info["F_correction"]) == pytest.approx(hand_F(R, P), rel=1e-9)


# ---------------------------------------------------------------------------
# the design loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def design_length_mode():
    d = st.ShellAndTubeDesign(make_params())
    return d, d(HOT, COLD, T_hot_out=373.0)


class TestDesign:
    def test_converges_from_reasonable_guesses(self):
        """Same answer from typical-U guesses (150 to 5000); residual at convergence."""
        Us = []
        for U0 in (150.0, 500.0, 2000.0, 5000.0):
            d = st.ShellAndTubeDesign(make_params(U_guess=U0))
            r = d(HOT, COLD, T_hot_out=373.0)
            assert bool(r["converged"]) and float(r["U_residual"]) < 1e-9
            Us.append(float(r["U"]))
        assert max(Us) / min(Us) - 1 < 1e-8

    def test_low_guess_finds_the_second_fixed_point_and_says_so(self):
        """Fixed tube length: positive feedback (more area, more tubes, slower, lower h_i) gives
        a second, large-area laminar fixed point. It is a genuine root of the same equations
        (residual ~ 0), flagged, with a warning; fixing the tube velocity removes it."""
        practical = st.ShellAndTubeDesign(make_params())(HOT, COLD, T_hot_out=373.0)
        with pytest.warns(UserWarning, match="large-area"):
            odd = st.ShellAndTubeDesign(make_params(U_guess=50.0))(HOT, COLD, T_hot_out=373.0)
        assert bool(odd["converged"]) and bool(odd["tube_laminar"]) and not bool(practical["tube_laminar"])
        assert float(odd["area"]) > 5 * float(practical["area"])
        v = st.ShellAndTubeDesign(make_params(U_guess=50.0, tube_velocity=1.3, tube_length=None))(HOT, COLD, T_hot_out=373.0)
        v2 = st.ShellAndTubeDesign(make_params(U_guess=3000.0, tube_velocity=1.3, tube_length=None))(HOT, COLD, T_hot_out=373.0)
        assert float(v["area"]) == pytest.approx(float(v2["area"]), rel=1e-8)

    def test_matches_independent_python_loop_length_mode(self, design_length_mode):
        d, r = design_length_mode
        h = hand_design(d.params, HOT, COLD, 373.0)
        for key, hk in [("U", "U"), ("area", "A"), ("n_tubes", "Nt"), ("shell_ID", "Ds"), ("h_i", "h_i"),
                        ("h_o", "h_o"), ("F", "F"), ("LMTD", "lmtd"), ("Q", "Q"), ("dP_tube", "dP_t"),
                        ("dP_shell", "dP_s")]:
            assert float(r[key]) == pytest.approx(h[hk], rel=2e-6), key

    def test_matches_independent_python_loop_velocity_mode(self):
        d = st.ShellAndTubeDesign(make_params(tube_velocity=1.3, tube_length=None))
        r = d(HOT, COLD, T_hot_out=373.0)
        h = hand_design(d.params, HOT, COLD, 373.0, velocity=1.3)
        for key, hk in [("U", "U"), ("area", "A"), ("n_tubes", "Nt"), ("tube_length", "L"),
                        ("shell_ID", "Ds"), ("dP_tube", "dP_t"), ("dP_shell", "dP_s")]:
            assert float(r[key]) == pytest.approx(h[hk], rel=2e-6), key
        assert float(r["tube_velocity"]) == pytest.approx(1.3, rel=1e-12)

    def test_internal_consistency(self, design_length_mode):
        d, r = design_length_mode
        p = d.params
        # Q = U A F LMTD
        assert float(r["U"] * r["area"] * r["F"] * r["LMTD"]) == pytest.approx(float(r["Q"]), rel=1e-9)
        # U is overall_U of the reported films
        U = ht.overall_U(r["h_i"], r["h_o"], p.tube_ID, p.tube_OD, p.k_wall, p.R_fi, p.R_fo)
        assert float(U) == pytest.approx(float(r["U"]), rel=1e-8)
        # area = pi d_o N_t L
        assert float(r["area"]) == pytest.approx(math.pi * p.tube_OD * float(r["n_tubes"]) * float(r["tube_length"]), rel=1e-10)
        # energy balance and Ds = Db + clearance, l_B = ratio * Ds
        assert float(r["Q"]) == pytest.approx(5.0 * SHELL["Cp"] * (473.0 - 373.0), rel=1e-12)
        assert float(r["Q"]) == pytest.approx(18.0 * TUBE["Cp"] * (float(r["T_cold_out"]) - 303.0), rel=1e-12)
        assert float(r["shell_ID"]) == pytest.approx(float(r["bundle_ID"]) + p.shell_clearance, rel=1e-13)
        assert float(r["baffle_spacing"]) == pytest.approx(p.baffle_spacing_ratio * float(r["shell_ID"]), rel=1e-13)
        assert float(r["n_baffles"]) == pytest.approx(float(r["tube_length"] / r["baffle_spacing"]) - 1, rel=1e-13)
        assert 100.0 < float(r["U"]) < 2000.0  # plausible band for hydrocarbon-hydrocarbon, loose

    @pytest.mark.parametrize("spec", ["Q", "T_hot_out", "T_cold_out"])
    def test_three_ways_to_specify_the_duty(self, design_length_mode, spec):
        d, ref = design_length_mode
        kw = {"Q": float(ref["Q"]), "T_hot_out": 373.0, "T_cold_out": float(ref["T_cold_out"])}[spec]
        r = d(HOT, COLD, **{spec: kw})
        assert float(r["area"]) == pytest.approx(float(ref["area"]), rel=1e-9)

    def test_spec_validation(self, design_length_mode):
        d, _ = design_length_mode
        with pytest.raises(ValueError, match="exactly one"):
            d(HOT, COLD)
        with pytest.raises(ValueError, match="exactly one"):
            d(HOT, COLD, Q=1e6, T_hot_out=373.0)
        with pytest.raises(ValueError, match="unknown override"):
            d(HOT, COLD, T_hot_out=373.0, bogus=1.0)
        with pytest.raises(ValueError, match="tube_side"):
            st.ShellAndTubeDesign(make_params(tube_side="middle"))
        with pytest.raises(ValueError, match="tube_velocity or tube_length"):
            st.ShellAndTubeDesign(make_params(tube_length=None))(HOT, COLD, T_hot_out=373.0)

    def test_hot_side_in_tubes_swaps_roles(self):
        d = st.ShellAndTubeDesign(make_params(tube_side="hot", tube_length=4.88))
        r = d(HOT, COLD, T_hot_out=373.0)
        assert bool(r["converged"])
        # tube side now carries the 5 kg/s hot oil
        t = st.tube_side_pressure_drop(5.0, SHELL["rho"], SHELL["mu"], r["n_tubes"], 2.0, DI, r["tube_length"])
        assert float(r["dP_tube"]) == pytest.approx(float(t["dP"]), rel=1e-12)
        assert float(r["Re_shell"]) == pytest.approx(float(
            st.kern_shell_side(18.0, **TUBE, shell_ID=r["shell_ID"], tube_OD=DO, pitch=PT,
                               baffle_spacing=r["baffle_spacing"])["Re"]), rel=1e-12)

    def test_overrides_equal_params(self):
        r1 = st.ShellAndTubeDesign(make_params(baffle_spacing=0.12))(HOT, COLD, T_hot_out=373.0)
        r2 = st.ShellAndTubeDesign(make_params())(HOT, COLD, T_hot_out=373.0, baffle_spacing=0.12)
        assert float(r1["area"]) == pytest.approx(float(r2["area"]), rel=1e-12)
        assert float(r1["baffle_spacing"]) == 0.12

    def test_jit(self, design_length_mode):
        d, r = design_length_mode
        f = jax.jit(lambda lB, v: d(HOT, COLD, T_hot_out=373.0, baffle_spacing=lB, tube_velocity=v, warn=False)["area"])
        d2 = st.ShellAndTubeDesign(make_params(tube_velocity=1.2, tube_length=None, baffle_spacing=0.2))
        assert float(f(0.2, 1.2)) == pytest.approx(float(d2(HOT, COLD, T_hot_out=373.0)["area"]), rel=1e-12)


# ---------------------------------------------------------------------------
# gradients
# ---------------------------------------------------------------------------

class TestGradients:
    @staticmethod
    def area(lB, v):
        d = st.ShellAndTubeDesign(make_params(tube_length=None, tube_velocity=1.2, baffle_spacing=0.25, tol=1e-13))
        return d(HOT, COLD, T_hot_out=373.0, baffle_spacing=lB, tube_velocity=v, warn=False)["area"]

    def test_check_grads_area_wrt_baffle_spacing_and_tube_velocity(self):
        # rev-mode (implicit adjoint through optx.fixed_point); fwd is checked below by finite differences
        check_grads(self.area, (0.25, 1.2), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    def test_grad_vs_central_finite_differences(self):
        g = jax.grad(self.area, argnums=(0, 1))(0.25, 1.2)
        h = 1e-5
        fd_l = (float(self.area(0.25 + h, 1.2)) - float(self.area(0.25 - h, 1.2))) / (2 * h)
        fd_v = (float(self.area(0.25, 1.2 + h)) - float(self.area(0.25, 1.2 - h))) / (2 * h)
        assert float(g[0]) == pytest.approx(fd_l, rel=1e-5)
        assert float(g[1]) == pytest.approx(fd_v, rel=1e-5)

    def test_gradient_in_length_mode_and_fouling(self):
        d = st.ShellAndTubeDesign(make_params(tol=1e-13))
        f = lambda lB, Rfo: d(HOT, COLD, T_hot_out=373.0, baffle_spacing=lB, R_fo=Rfo, warn=False)["area"]
        g = jax.grad(f, argnums=(0, 1))(0.2, 1e-4)
        h = 1e-6
        fd_l = (float(f(0.2 + h, 1e-4)) - float(f(0.2 - h, 1e-4))) / (2 * h)
        fd_r = (float(f(0.2, 1e-4 + 1e-8)) - float(f(0.2, 1e-4 - 1e-8))) / 2e-8
        assert float(g[0]) == pytest.approx(fd_l, rel=1e-4)
        assert float(g[1]) == pytest.approx(fd_r, rel=1e-4)
        assert float(g[1]) > 0  # more fouling, more area

    def test_gradient_wrt_passes_and_flow(self):
        d = st.ShellAndTubeDesign(make_params(tube_length=None, tube_velocity=1.2, tol=1e-13))
        f = lambda n, m: d(HOT, {"T": 303.0, "m_dot": m}, T_hot_out=373.0, n_tube_passes=n, warn=False)["dP_tube"]
        g = jax.grad(f, argnums=(0, 1))(3.0, 18.0)
        assert np.all(np.isfinite([float(x) for x in g]))
        h = 1e-5
        fd = (float(f(3.0 + h, 18.0)) - float(f(3.0 - h, 18.0))) / (2 * h)
        assert float(g[0]) == pytest.approx(fd, rel=1e-4)


# ---------------------------------------------------------------------------
# design vs rating, rounding
# ---------------------------------------------------------------------------

class TestRating:
    def test_rate_reproduces_design(self, design_length_mode):
        d, r = design_length_mode
        rt = d.rate(HOT, COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"])
        assert float(rt["Q"]) == pytest.approx(float(r["Q"]), rel=1e-8)
        assert float(rt["T_hot_out"]) == pytest.approx(373.0, abs=1e-5)
        assert float(rt["T_cold_out"]) == pytest.approx(float(r["T_cold_out"]), abs=1e-5)
        assert float(rt["U"]) == pytest.approx(float(r["U"]), rel=1e-8)
        assert float(rt["dP_shell"]) == pytest.approx(float(r["dP_shell"]), rel=1e-8)
        assert float(rt["dP_tube"]) == pytest.approx(float(r["dP_tube"]), rel=1e-8)

    def test_ShellAndTubeHX_agrees_with_design_for_same_geometry_and_U(self, design_length_mode):
        """The rating unit, fed the design's U*A, returns the design's duty and outlet temperatures."""
        d, r = design_length_mode
        MW = 0.018   # arbitrary: C_hot = m Cp is what matters
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=float(r["U"] * r["area"]),
                                                 Cp_hot=SHELL["Cp"] * MW, Cp_cold=TUBE["Cp"] * MW))
        ho, co, info = hx(make_stream({"A": 5.0 / MW}, 473.0, 101325.0), make_stream({"A": 18.0 / MW}, 303.0, 101325.0))
        assert float(info["Q"]) == pytest.approx(float(r["Q"]), rel=1e-8)
        assert float(ho["T"]) == pytest.approx(373.0, abs=1e-5)
        assert float(co["T"]) == pytest.approx(float(r["T_cold_out"]), abs=1e-5)
        assert float(info["F_correction"]) == pytest.approx(float(r["F"]), rel=1e-7)
        # same through rate() with the same U
        rt = d.rate(HOT, COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"], U=r["U"])
        assert float(rt["Q"]) == pytest.approx(float(info["Q"]), rel=1e-9)

    def test_rate_with_given_U_overrides_correlation(self, design_length_mode):
        d, r = design_length_mode
        rt = d.rate(HOT, COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"], U=0.5 * r["U"])
        assert float(rt["U"]) == pytest.approx(0.5 * float(r["U"]))
        assert float(rt["Q"]) < float(r["Q"])
        assert float(rt["U_correlation"]) == pytest.approx(float(r["U"]), rel=1e-8)

    def test_round_design_and_rerate(self, design_length_mode):
        d, r = design_length_mode
        g = st.round_design(r)
        assert g["n_tubes"] == math.ceil(float(r["n_tubes"])) and g["n_tubes"] >= float(r["n_tubes"])
        assert g["n_tube_passes"] in st.STANDARD_TUBE_PASSES
        assert g["baffle_spacing"] <= float(r["baffle_spacing"]) + 1e-12
        assert g["shell_ID"] >= float(r["shell_ID"]) - 1e-12
        rt = d.rate(HOT, COLD, g["n_tubes"], g["tube_length"], g["shell_ID"], g["baffle_spacing"], g["n_tube_passes"])
        # rounding changes velocities and h; the re-rated duty should still be near the target
        assert float(rt["Q"]) == pytest.approx(float(r["Q"]), rel=0.08)

    def test_round_design_passes_to_nearest_standard(self):
        base = {"n_tubes": 10.2, "tube_length": 3.0, "shell_ID": 0.3, "baffle_spacing": 0.1}
        assert st.round_design({**base, "n_tube_passes": 2.9})["n_tube_passes"] == 2.0   # |2.9-2| < |2.9-4|
        assert st.round_design({**base, "n_tube_passes": 3.4})["n_tube_passes"] == 4.0
        assert st.round_design({**base, "n_tube_passes": 5.0})["n_tube_passes"] == 4.0   # tie 4/6: first minimum
        assert st.round_design({**base, "n_tube_passes": 7.5})["n_tube_passes"] == 8.0


# ---------------------------------------------------------------------------
# phase change, limits and warnings
# ---------------------------------------------------------------------------

class TestPhaseChangeAndWarnings:
    def test_condensing_shell_constant_h(self):
        """Isothermal hot side: F = 1, shell h fixed, shell dP not modelled."""
        p = make_params(hot_isothermal=True, shell_h=6000.0, tube_length=3.0)
        d = st.ShellAndTubeDesign(p)
        Q = 8.0e5
        r = d({"T": 400.0, "m_dot": 1.0}, COLD, Q=Q)
        assert float(r["F"]) == 1.0
        assert float(r["T_hot_out"]) == 400.0
        dT1, dT2 = 400.0 - float(r["T_cold_out"]), 400.0 - 303.0
        lm = (dT1 - dT2) / math.log(dT1 / dT2)
        assert float(r["U"] * r["area"] * lm) == pytest.approx(Q, rel=1e-9)
        assert float(r["h_o"]) == 6000.0 and float(r["dP_shell"]) == 0.0
        # rating agrees: isothermal side, Cr -> 0
        rt = d.rate({"T": 400.0, "m_dot": 1.0}, COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"])
        assert float(rt["Q"]) == pytest.approx(Q, rel=1e-8)

    def test_condensing_shell_nusselt_callable(self):
        """Shell h from heat_transfer.nusselt_film_condensation as a function of geometry."""
        def shell_h(g):
            n_col = jnp.sqrt(g["n_tubes"])    # tubes per vertical row, crude
            return ht.nusselt_film_condensation(k_l=0.6, rho_l=960.0, rho_v=0.6, mu_l=2.8e-4, h_fg=2.26e6,
                                                Cp_l=4216.0, dT=8.0, L=g["tube_OD"], n_tubes=n_col)
        d = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=shell_h, tube_length=3.0, tol=1e-13))
        r = d({"T": 373.15, "m_dot": 1.0}, COLD, Q=6.0e5)
        assert bool(r["converged"]) and 1000.0 < float(r["h_o"]) < 20000.0
        g = jax.grad(lambda Q: d({"T": 373.15, "m_dot": 1.0}, COLD, Q=Q, warn=False)["area"])(6.0e5)
        assert float(g) > 0

    def test_boiling_cold_side_isothermal(self):
        p = make_params(cold_isothermal=True, tube_side="hot", shell_h=3000.0, tube_length=3.0)
        r = st.ShellAndTubeDesign(p)({"T": 480.0, "m_dot": 4.0}, {"T": 400.0, "m_dot": 1.0}, Q=4.0e5)
        assert float(r["F"]) == 1.0 and float(r["T_cold_out"]) == 400.0
        assert float(r["T_hot_out"]) == pytest.approx(480.0 - 4.0e5 / (4.0 * SHELL["Cp"]), rel=1e-12)

    @pytest.mark.filterwarnings("ignore:tube-side Re")
    def test_low_F_warns(self):
        # strong temperature cross: cold outlet well above hot outlet -> F small
        d = st.ShellAndTubeDesign(make_params())
        with pytest.warns(UserWarning, match="correction factor"):
            r = d({"T": 420.0, "m_dot": 5.0}, {"T": 300.0, "m_dot": 3.0}, T_cold_out=405.0)
        assert bool(r["F_too_low"]) and float(r["F"]) < 0.75

    def test_no_warning_for_a_good_design(self, design_length_mode):
        d, _ = design_length_mode
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            d(HOT, COLD, T_hot_out=373.0)

    def test_pressure_drop_limits_flag_and_warn(self):
        d = st.ShellAndTubeDesign(make_params(dP_tube_max=1e3, dP_shell_max=1e3))
        with pytest.warns(UserWarning) as rec:
            r = d(HOT, COLD, T_hot_out=373.0)
        msgs = " ".join(str(w.message) for w in rec)
        assert "tube-side pressure drop" in msgs and "shell-side pressure drop" in msgs
        assert bool(r["dP_tube_exceeded"]) and bool(r["dP_shell_exceeded"])
        r2 = d(HOT, COLD, T_hot_out=373.0, warn=False)
        assert bool(r2["dP_tube_exceeded"])
        ok = st.ShellAndTubeDesign(make_params(dP_tube_max=1e7, dP_shell_max=1e7))(HOT, COLD, T_hot_out=373.0)
        assert not bool(ok["dP_tube_exceeded"]) and not bool(ok["dP_shell_exceeded"])

    def test_no_warning_machinery_under_grad(self):
        d = st.ShellAndTubeDesign(make_params(dP_tube_max=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.grad(lambda lB: d(HOT, COLD, T_hot_out=373.0, baffle_spacing=lB)["area"])(0.2)


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_class_attributes(self):
        c = st.ShellAndTubeDesign
        assert c.equations and c.assumptions and c.references
        assert all(isinstance(x, str) for x in c.equations + c.assumptions + c.references)

    def test_params_dict_like(self):
        p = make_params()
        assert p["tube_OD"] == DO and "tube_ID" in p
        assert p.update(tube_length=6.0).tube_length == 6.0
