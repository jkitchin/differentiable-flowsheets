"""Tests for :mod:`difflow.staged` (McCabe-Thiele stepper, pinches, Kremser).

What is and is not verified here (no textbook example numbers are claimed):

* Hand calculation of the benzene-toluene illustration in the issue
  (constant alpha 2.5, z 0.4, x_D 0.95, x_B 0.05, saturated liquid): R_min,
  Fenske N_min and the integer staircase are re-derived with a plain Python
  loop that shares no code with the library.
* Analytic oracles: Fenske at total reflux, the closed-form pinch, Kremser,
  the Lewis relation for a Murphree efficiency on straight lines.
* The tangent pinch is checked against a SciPy PCHIP + dense-grid maximum on
  the same table.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow import staged as S
from difflow.phase_diagrams import xy_curve
from difflow.units.distillation import minimum_reflux_ratio

jax.config.update("jax_enable_x64", True)

AL, Z, XD, XB = 2.5, 0.4, 0.95, 0.05


def _y_alpha(x, a=AL):
    return a * x / (1.0 + (a - 1.0) * x)


def _hand_integer_stages(R, q=1.0, a=AL, z=Z, xd=XD, xb=XB):
    """Plain-Python staircase, hard feed switch; returns the integer count."""
    yq_den = None
    # feed line / rectifying intersection
    # q x - (q-1) y = z ; y = (R x + xd)/(R+1)
    x_f = (z + (q - 1.0) * xd / (R + 1.0)) / (q - (q - 1.0) * R / (R + 1.0))
    y_f = (R * x_f + xd) / (R + 1.0)
    m_s = (y_f - xb) / (x_f - xb)
    x = xd
    n = 0
    while x > xb and n < 200:
        y = xd if n == 0 else (
            (R * x + xd) / (R + 1.0) if x >= x_f else xb + m_s * (x - xb))
        x = y / (a - (a - 1.0) * y)
        n += 1
    return n


# ---------------------------------------------------------------------------
# wex12-1 illustration
# ---------------------------------------------------------------------------


class TestIllustration:
    @pytest.mark.slow
    def test_rmin_and_nmin(self):
        c = S.constant_alpha(AL)
        y_f = _y_alpha(Z)
        assert float(S.minimum_reflux(c, Z, 1.0, XD)) == pytest.approx(
            (XD - y_f) / (y_f - Z), rel=1e-9)
        assert float(S.minimum_reflux(c, Z, 1.0, XD)) == pytest.approx(1.4444, abs=5e-5)
        assert float(S.total_reflux_stages(c, XD, XB)) == pytest.approx(6.4269, abs=5e-5)

    @pytest.mark.slow
    def test_total_reflux_equals_fenske(self):
        fen = np.log(XD / (1 - XD) * (1 - XB) / XB) / np.log(AL)
        c = S.constant_alpha(AL)
        assert float(S.total_reflux_stages(c, XD, XB)) == pytest.approx(fen, rel=1e-9)
        assert float(S.minimum_stages(AL, XD, XB)) == pytest.approx(fen, rel=1e-12)
        assert float(S.minimum_stages(c, XD, XB)) == pytest.approx(fen, rel=1e-9)

    @pytest.mark.slow
    def test_twelve_stages_and_hand_loop(self):
        c = S.constant_alpha(AL)
        Rmin = float(S.minimum_reflux(c, Z, 1.0, XD))
        R = 1.5 * Rmin
        assert R == pytest.approx(2.1667, abs=1e-4)
        assert 100 * (Z - XB) / (XD - XB) == pytest.approx(38.9, abs=0.05)
        res = S.mccabe_thiele(c, S.rectifying_line(R, XD),
                              S.stripping_line(R, XD, XB, Z, 1.0), XD, XB,
                              q_line=S.q_line(1.0, Z))
        assert _hand_integer_stages(R) == 12
        assert int(np.ceil(float(res["N"]) - 1e-9)) == 12
        assert 11.0 < float(res["N"]) < 12.0
        # sharp feed switch reproduces the hand loop's feed stage (stage 6)
        sharp = S.mccabe_thiele(c, S.rectifying_line(R, XD),
                                S.stripping_line(R, XD, XB, Z, 1.0), XD, XB,
                                q_line=S.q_line(1.0, Z), feed_width=1e-7)
        assert float(sharp["feed_stage"]) == pytest.approx(6.0, abs=1e-6)
        assert float(sharp["N_rect"]) + float(sharp["N_strip"]) == pytest.approx(
            float(sharp["N"]))

    @pytest.mark.slow
    def test_real_benzene_toluene_curve(self, bt_thermo):
        d = xy_curve(bt_thermo, ["benzene", "toluene"], 101325.0, n=41)
        c = S.tabulated(d["x"], d["y"])
        Rmin = float(S.minimum_reflux(c, Z, 1.0, XD))
        # between the constant-alpha 2.2 and 2.8 answers (alpha(x) ~ 2.4-2.6)
        lo = float(S.minimum_reflux(S.constant_alpha(2.8), Z, 1.0, XD))
        hi = float(S.minimum_reflux(S.constant_alpha(2.2), Z, 1.0, XD))
        assert lo < Rmin < hi
        R = 1.5 * Rmin
        res = S.mccabe_thiele(c, S.rectifying_line(R, XD),
                              S.stripping_line(R, XD, XB, Z, 1.0), XD, XB,
                              q_line=S.q_line(1.0, Z))
        assert 8.0 < float(res["N"]) < 16.0
        assert float(res["pinched"]) < 1e-6


@pytest.fixture(scope="module")
def bt_thermo():
    from difflow.database import get_species_data
    from difflow.thermo import IdealThermo
    return IdealThermo({s: get_species_data(s) for s in ("benzene", "toluene")})


# ---------------------------------------------------------------------------
# Lines and q
# ---------------------------------------------------------------------------


class TestLines:
    def test_q_line_special_cases(self):
        zz = 0.4
        for q in (1.0, 0.0, 1.5, -0.5, 0.5):
            ql = np.asarray(S.q_line(q, zz))
            # always through (z, z)
            assert ql[0] * zz + ql[1] * zz == pytest.approx(zz)
        v = np.asarray(S.q_line(1.0, zz))
        assert v[1] == 0.0 and v[0] * zz == pytest.approx(zz)          # vertical x = z
        h = np.asarray(S.q_line(0.0, zz))
        assert h[0] == 0.0 and h[2] / h[1] == pytest.approx(zz)        # horizontal y = z
        assert float(S.line_slope_intercept(S.q_line(1.5, zz))[0]) == pytest.approx(3.0)
        assert float(S.line_slope_intercept(S.q_line(-0.5, zz))[0]) == pytest.approx(1 / 3)

    def test_rectifying_and_boilup(self):
        s, b = S.line_slope_intercept(S.rectifying_line(2.0, 0.9))
        assert float(s) == pytest.approx(2 / 3) and float(b) == pytest.approx(0.3)
        s, b = S.line_slope_intercept(S.stripping_line_from_boilup(2.0, 0.1))
        assert float(s) == pytest.approx(1.5) and float(b) == pytest.approx(-0.05)
        # through (x_B, x_B)
        assert float(S.line_y(S.stripping_line_from_boilup(2.0, 0.1), 0.1)) == pytest.approx(0.1)

    def test_operating_line(self):
        ln = S.operating_line(2.0, 4.0, 0.1, 0.0)
        assert float(S.line_y(ln, 0.1)) == pytest.approx(0.0)
        assert float(S.line_slope_intercept(ln)[0]) == pytest.approx(0.5)

    def test_open_steam_through_origin_of_bottoms(self):
        ln = S.stripping_line(2.0, XD, XB, Z, 1.0, open_steam=True)
        assert float(S.line_y(ln, XB)) == pytest.approx(0.0, abs=1e-14)
        xf, yf = S.feed_intersection(1.0, Z, 2.0, XD)
        assert float(S.line_y(ln, xf)) == pytest.approx(float(yf))

    def test_ratio_round_trip(self):
        x = jnp.array([0.0, 0.3, 0.9])
        assert np.allclose(S.from_ratio(S.to_ratio(x)), x)

    @pytest.mark.parametrize("q", [1.0, 0.0, 1.4, -0.3, 0.5])
    def test_rmin_matches_hand_pinch_any_q(self, q):
        # constant alpha: pinch at the q-line / curve intersection
        from scipy.optimize import brentq
        a, b, c = q, 1.0 - q, Z
        f = lambda x: a * x + b * _y_alpha(x) - c
        xq = brentq(f, Z if q > 1 else 0.0, 1.0 if q > 1 else Z) if q != 1.0 else Z
        yq = _y_alpha(xq)
        expect = (XD - yq) / (yq - xq)
        got = float(S.minimum_reflux(S.constant_alpha(AL), Z, q, XD))
        assert got == pytest.approx(expect, rel=1e-8)

    def test_rmin_moves_with_q(self):
        c = S.constant_alpha(AL)
        r = [float(S.minimum_reflux(c, Z, q, XD)) for q in (1.5, 1.0, 0.5, 0.0, -0.5)]
        assert all(b > a for a, b in zip(r, r[1:]))  # vapor feed needs more reflux

    def test_legacy_minimum_reflux_ratio_honours_q(self):
        c = S.constant_alpha(AL)
        for q in (1.0, 0.0, 1.5):
            old = float(minimum_reflux_ratio(Z, 1 - Z, XD, AL, q))
            assert old == pytest.approx(float(S.minimum_reflux(c, Z, q, XD)), rel=1e-9)
        # q = 1 equals the Underwood closed form with no floor
        closed = (XD / Z - AL * (1 - XD) / (1 - Z)) / (AL - 1)
        assert float(minimum_reflux_ratio(Z, 1 - Z, XD, AL)) == pytest.approx(closed, rel=1e-9)
        assert float(minimum_reflux_ratio(Z, 1 - Z, XD, AL, 0.0)) > float(
            minimum_reflux_ratio(Z, 1 - Z, XD, AL, 1.0))
        # not floored at 0.1
        assert float(minimum_reflux_ratio(0.5, 0.5, 0.51, 10.0)) < 0.1


# ---------------------------------------------------------------------------
# Tangent pinch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ew_curve():
    import jax.numpy as jnp
    from difflow.thermo import IdealThermo, SpeciesData
    from difflow.units.lle import NRTLParams

    def sd(name, ant, Tc):
        return SpeciesData(name=name, MW=1.0, Cp_coeffs=(0, 0, 0, 0),
                           Hvap_coeffs=(1.0, 0.38, Tc), antoine_coeffs=ant)

    th = IdealThermo({"ethanol": sd("ethanol", (10.24677, 1598.673, -46.424), 514.0),
                      "water": sd("water", (9.6543, 1435.264, -64.848), 647.0)})
    nr = NRTLParams(species=("ethanol", "water"),
                    a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
                    b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
                    alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]))
    d = xy_curve(th, ["ethanol", "water"], 101325.0, 101, nr)
    return d, S.tabulated(d["x"], d["y"])


@pytest.mark.slow
class TestTangentPinch:
    @pytest.mark.slow
    def test_ethanol_water_tangent_exceeds_feed_pinch(self, ew_curve):
        d, c = ew_curve
        z, xd = 0.2, 0.86
        got = float(S.minimum_reflux(c, z, 1.0, xd))
        yz = float(c(z))
        feed_pinch = (xd - yz) / (yz - z)
        assert got > feed_pinch * 1.05
        # independent re-implementation: SciPy PCHIP + a dense grid
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(np.asarray(d["x"]), np.asarray(d["y"]))
        xs = np.linspace(z, xd - 1e-6, 400001)
        s = np.max((xd - f(xs)) / (xd - xs))
        assert got == pytest.approx(s / (1 - s), rel=1e-6)
        # the maximiser is interior: tangent, not the feed point
        x_star = xs[np.argmax((xd - f(xs)) / (xd - xs))]
        assert z + 0.05 < x_star < xd - 0.05

    def test_no_tangent_for_constant_alpha(self):
        # concave curve: pinch is the feed point, no interior maximum
        c = S.constant_alpha(AL)
        xs = np.linspace(Z, XD - 1e-6, 2001)
        phi = (XD - _y_alpha(xs)) / (XD - xs)
        assert np.argmax(phi) == 0

    @pytest.mark.slow
    def test_rmin_gradient_through_tangent(self, ew_curve):
        _, c = ew_curve
        g = jax.grad(lambda xd: S.minimum_reflux(c, 0.2, 1.0, xd))(0.86)
        h = 1e-5
        fd = (S.minimum_reflux(c, 0.2, 1.0, 0.86 + h) - S.minimum_reflux(c, 0.2, 1.0, 0.86 - h)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-4)

    def test_minimum_solvent_ratio_matches_straight_line_closed_form(self):
        m = 2.0
        c = S.linear_equilibrium(m)
        x_in, x_out, y_in = 0.1, 0.001, 0.0
        expect = (x_in - x_out) / (m * x_in - y_in)
        assert float(S.minimum_solvent_ratio(c, x_in, x_out, y_in)) == pytest.approx(expect, rel=1e-9)

    def test_minimum_solvent_ratio_tangent(self):
        # a curve with an interior (tangent) pinch: sigmoid-ish concave-up then down
        xs = np.linspace(0, 0.1, 41)
        ys = 4 * xs / (1 + 30 * xs) * 1.0 + 6 * xs**2  # convex bit at the top
        c = S.tabulated(xs, ys)
        got = float(S.minimum_solvent_ratio(c, 0.1, 0.002, 0.0))
        from scipy.interpolate import PchipInterpolator
        f = PchipInterpolator(xs, ys)
        g = np.linspace(0.002 + 1e-6, 0.1, 200001)
        slope = np.min((f(g) - 0.0) / (g - 0.002))
        assert got == pytest.approx(1 / slope, rel=1e-6)


# ---------------------------------------------------------------------------
# Kremser and Lewis
# ---------------------------------------------------------------------------


class TestKremser:
    @pytest.mark.parametrize("E,expect", [(1.5, 8.70), (2.0, 5.66), (5.0, 2.72)])
    def test_stages_values(self, E, expect):
        assert float(S.kremser_stages(E, 0.99)) == pytest.approx(expect, abs=0.005)

    def test_round_trip(self):
        for A in (0.8, 1.0, 1.3, 4.0):
            f = 0.7 if A < 1 else 0.95
            N = S.kremser_stages(A, f)
            assert float(S.kremser_fraction(A, N)) == pytest.approx(f, rel=1e-10)

    def test_a_below_one_and_unreachable(self):
        # recovery tends to A (< 1) as N grows
        assert float(S.kremser_fraction(0.8, 200.0)) == pytest.approx(0.8, rel=1e-6)
        assert np.isinf(float(S.kremser_stages(0.8, 0.9)))

    def test_continuity_and_gradient_across_a_equal_one(self):
        f = 0.9
        for g in (lambda A: S.kremser_stages(A, f), lambda A: S.kremser_fraction(A, 3.0)):
            vals = [float(g(1.0 + e)) for e in (-1e-6, -1e-10, 0.0, 1e-10, 1e-6)]
            assert max(vals) - min(vals) < 1e-4 * abs(vals[2]) + 1e-5
            gr = [float(jax.grad(g)(1.0 + e)) for e in (-1e-3, -1e-9, 0.0, 1e-9, 1e-3)]
            assert max(gr) - min(gr) < 5e-2 * abs(gr[2])
        assert float(S.kremser_stages(1.0, 0.9)) == pytest.approx(9.0, rel=1e-12)
        assert float(S.kremser_fraction(1.0, 3.0)) == pytest.approx(0.75, rel=1e-12)

    def test_check_grads(self):
        check_grads(lambda A: S.kremser_stages(A, 0.95), (1.7,), order=1, modes=["rev"])
        check_grads(lambda A, N: S.kremser_fraction(A, N), (0.7, 4.0), order=1, modes=["rev"])

    def test_ree_stages_kremser_fixed(self):
        from difflow_ree.equilibrium import stages_kremser
        for E, expect in [(1.5, 8.70), (2.0, 5.66), (5.0, 2.72)]:
            assert float(stages_kremser(E, 1.0, 0.99)) == pytest.approx(expect, abs=0.005)
        # still floored at one stage (a cascade has at least one); core is not
        assert float(stages_kremser(10.0, 1.0, 0.5)) == 1.0
        assert float(S.kremser_stages(10.0, 0.5)) < 1.0
        # equals the core function
        from difflow.units.lle import stages_for_recovery
        assert float(stages_kremser(3.0, 1.0, 0.99)) == pytest.approx(
            float(stages_for_recovery(3.0, 1.0, 0.99)), rel=1e-12)

    def test_reexports(self):
        import difflow_ree
        from difflow.units import lle
        assert lle.kremser_stages is S.kremser_stages
        assert difflow_ree.kremser_fraction is S.kremser_fraction


def _absorber(A, rec, m=1.0, y_in=0.01, V=1.0):
    """Pure-solvent absorber as stepper inputs: lines through (x_top, y_out), (x_bot, y_in)."""
    L = A * m * V
    y_out = y_in * (1.0 - rec)
    x_bot = (V / L) * (y_in - y_out)
    return L, V, y_out, x_bot


class TestStepperOracles:
    @pytest.mark.parametrize("A", [0.0 + 1.5, 2.0, 0.9])
    def test_log_fraction_equals_kremser(self, A):
        rec = 0.7 if A < 1 else 0.98
        L, V, y_out, x_bot = _absorber(A, rec)
        c = S.linear_equilibrium(1.0)
        top = S.operating_line(L, V, 0.0, y_out)
        res = S.mccabe_thiele(c, top, None, 0.0, x_bot, fraction="log", n_max=200)
        assert float(res["N"]) == pytest.approx(float(S.kremser_stages(A, rec)), rel=1e-8)

    def test_linear_fraction_agrees_at_whole_stages(self):
        A, rec = 2.0, None
        N = 4.0
        rec = float(S.kremser_fraction(A, N))
        L, V, y_out, x_bot = _absorber(A, rec)
        res = S.mccabe_thiele(S.linear_equilibrium(1.0), S.operating_line(L, V, 0.0, y_out),
                              None, 0.0, x_bot, fraction="linear", n_max=50)
        assert float(res["N"]) == pytest.approx(N, rel=1e-8)

    def test_murphree_lewis_relation(self):
        A, rec, E = 1.5, 0.98, 0.7
        lam = 1.0 / A
        L, V, y_out, x_bot = _absorber(A, rec)
        top = S.operating_line(L, V, 0.0, y_out)
        eq = S.linear_equilibrium(1.0)
        n_eq = S.mccabe_thiele(eq, top, None, 0.0, x_bot, fraction="log", n_max=200)["N"]
        n_act = S.mccabe_thiele(eq.with_efficiency(E), top, None, 0.0, x_bot,
                                fraction="log", n_max=200)["N"]
        e_og = np.log(1 + E * (lam - 1)) / np.log(lam)
        # Lewis: E_OG = N_equilibrium / N_actual, exact for straight lines
        assert float(n_eq / n_act) == pytest.approx(e_og, rel=1e-7)
        assert float(n_act) > float(n_eq)

    def test_efficiency_one_is_ideal(self):
        c = S.constant_alpha(AL)
        R = 2.2
        args = (S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0), XD, XB)
        a = S.mccabe_thiele(c, *args, q_line=S.q_line(1.0, Z))["N"]
        b = S.mccabe_thiele(c.with_efficiency(1.0), *args, q_line=S.q_line(1.0, Z))["N"]
        assert float(b) == pytest.approx(float(a), rel=1e-9)

    def test_murphree_on_the_distillation_diagram(self):
        c = S.constant_alpha(AL)
        R = 2.5
        args = (S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0), XD, XB)
        n1 = float(S.mccabe_thiele(c, *args, q_line=S.q_line(1.0, Z))["N"])
        n7 = float(S.mccabe_thiele(c.with_efficiency(0.7), *args, q_line=S.q_line(1.0, Z))["N"])
        assert 1.3 * n1 < n7 < n1 / 0.7 * 1.05  # more stages, ~ N/E


# ---------------------------------------------------------------------------
# Options: partial condenser, open steam, subcooled feed, specified feed stage
# ---------------------------------------------------------------------------


class TestOptions:
    def _run(self, R=2.5, q=1.0, **kw):
        c = S.constant_alpha(AL)
        open_steam = kw.pop("open_steam", False)
        return S.mccabe_thiele(c, S.rectifying_line(R, XD),
                               S.stripping_line(R, XD, XB, Z, q, open_steam=open_steam),
                               XD, XB, q_line=S.q_line(q, Z), **kw)

    def test_partial_condenser_is_one_of_the_stages(self):
        total = self._run()
        part = self._run(partial_condenser=True)
        assert float(part["N"]) == pytest.approx(float(total["N"]))   # same staircase
        assert float(part["N_column"]) == pytest.approx(float(total["N"]) - 1.0)

    def test_open_steam_needs_fewer_stages_than_reboiler_diagram(self):
        # with the stripping line aimed at (x_B, 0) the last steps are larger
        base = float(self._run()["N"])
        steam = float(self._run(open_steam=True)["N"])
        assert steam < base
        # hand check of the open-steam staircase
        R = 2.5
        xf, yf = XD * 0 + Z, (R * Z + XD) / (R + 1)
        m = yf / (xf - XB)
        x, n = XD, 0
        while x > XB:
            y = XD if n == 0 else ((R * x + XD) / (R + 1) if x >= xf else m * (x - XB))
            x = y / (AL - (AL - 1) * y)
            n += 1
        assert n - 1 < steam <= n

    def test_subcooled_feed_changes_the_count(self):
        n1 = float(self._run(q=1.0)["N"])
        n13 = float(self._run(q=1.3)["N"])
        assert n13 < n1   # subcooled feed lifts the stripping line off the curve

    def test_hand_loop_other_q(self):
        for q in (0.5, 1.3):
            N = float(self._run(q=q, feed_width=1e-7)["N"])
            n = _hand_integer_stages(2.5, q=q)
            assert n - 1 < N <= n + 1e-9

    def test_specified_feed_stage(self):
        c = S.constant_alpha(AL)
        R = 2.5
        top, bot = S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0)
        opt = S.mccabe_thiele(c, top, bot, XD, XB, q_line=S.q_line(1.0, Z))
        ns = {f: float(S.mccabe_thiele(c, top, bot, XD, XB, mode="specified",
                                       feed_stage=f)["N"]) for f in (6, 7, 8, 10)}
        # the optimal feed stage is the (near) minimum of N over the feed stage
        assert min(ns.values()) >= float(opt["N"]) - 1e-3
        assert ns[6] <= ns[7] <= ns[8] <= ns[10]   # later than optimal only costs stages
        r = S.mccabe_thiele(c, top, bot, XD, XB, mode="specified", feed_stage=6.0)
        assert float(r["feed_stage"]) == pytest.approx(6.0)

    def test_pinched_flag_and_cap(self):
        c = S.constant_alpha(AL)
        Rmin = float(S.minimum_reflux(c, Z, 1.0, XD))
        R = 0.97 * Rmin    # below the minimum: stalls at the pinch
        res = S.mccabe_thiele(c, S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0),
                              XD, XB, q_line=S.q_line(1.0, Z), n_max=60)
        assert float(res["N"]) == pytest.approx(60.0)
        assert float(res["pinched"]) > 0.5
        assert not np.any(np.isnan(np.asarray(res["x"])))

    def test_bad_arguments(self):
        c = S.constant_alpha(AL)
        with pytest.raises(ValueError):
            S.mccabe_thiele(c, S.rectifying_line(2.0, XD), S.stripping_line(2.0, XD, XB, Z, 1.0),
                            XD, XB)
        with pytest.raises(ValueError):
            S.mccabe_thiele(c, S.rectifying_line(2.0, XD), None, XD, XB, fraction="nope")


# ---------------------------------------------------------------------------
# Gradients, smoothness, jit / vmap
# ---------------------------------------------------------------------------


def _N_of(R, xd, q, alpha):
    c = S.constant_alpha(alpha)
    return S.mccabe_thiele(c, S.rectifying_line(R, xd), S.stripping_line(R, xd, XB, Z, q),
                           xd, XB, q_line=S.q_line(q, Z))["N"]


class TestGradients:
    @pytest.mark.slow
    def test_check_grads_fractional_N(self):
        for i, x0 in enumerate((2.2, 0.95, 1.0, 2.5)):
            f = lambda v, i=i: _N_of(*[v if j == i else a for j, a in enumerate((2.2, 0.95, 1.0, 2.5))])
            check_grads(f, (x0,), order=1, modes=["rev"], eps=1e-6, atol=1e-4, rtol=1e-4)

    @pytest.mark.slow
    def test_check_grads_rmin(self):
        c = S.constant_alpha(AL)
        check_grads(lambda z: S.minimum_reflux(c, z, 1.0, XD), (0.4,), order=1, modes=["rev"])
        check_grads(lambda q: S.minimum_reflux(c, Z, q, XD), (0.8,), order=1, modes=["rev"])
        check_grads(lambda z, q: S.minimum_reflux(c, z, q, XD), (0.4, 1.2), order=1, modes=["rev"])

    def test_rmin_gradient_wrt_alpha(self):
        g = jax.grad(lambda a: S.minimum_reflux(S.constant_alpha(a), Z, 1.0, XD))(AL)
        h = 1e-6
        fd = (S.minimum_reflux(S.constant_alpha(AL + h), Z, 1.0, XD)
              - S.minimum_reflux(S.constant_alpha(AL - h), Z, 1.0, XD)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-6)

    @pytest.mark.slow
    def test_N_of_R_monotone_and_continuous(self):
        c = S.constant_alpha(AL)
        Rmin = S.minimum_reflux(c, Z, 1.0, XD)
        ratios = jnp.linspace(1.05, 3.0, 400)
        Ns = np.asarray(jax.vmap(lambda r: _N_of(r * Rmin, XD, 1.0, AL))(ratios))
        d = np.diff(Ns)
        assert np.all(d <= 1e-9)                 # monotone decreasing
        assert np.max(-d) < 0.6                  # no stair steps (a stair would be ~1)
        # derivative w.r.t. R is finite and negative throughout
        g = np.asarray(jax.vmap(jax.grad(lambda r: _N_of(r * Rmin, XD, 1.0, AL)))(ratios))
        assert np.all(np.isfinite(g)) and np.all(g < 0)

    def test_jit_and_vmap_over_R(self):
        c = S.constant_alpha(AL)
        f = jax.jit(lambda R: S.mccabe_thiele(
            c, S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0), XD, XB,
            q_line=S.q_line(1.0, Z))["N"])
        Rs = jnp.array([1.8, 2.2, 3.0])
        v = jax.vmap(f)(Rs)
        assert v.shape == (3,)
        assert np.allclose(v, [float(f(r)) for r in Rs])
        assert v[0] > v[1] > v[2]

    @pytest.mark.slow
    def test_grad_through_tabulated_curve(self, ew_curve):
        d, _ = ew_curve

        def N(scale):
            c = S.tabulated(d["x"], jnp.clip(d["y"] * scale, 0, 1))
            R = 4.0
            return S.mccabe_thiele(c, S.rectifying_line(R, 0.85),
                                   S.stripping_line(R, 0.85, 0.02, 0.2, 1.0), 0.85, 0.02,
                                   q_line=S.q_line(1.0, 0.2))["N"]
        g = jax.grad(N)(1.0)
        h = 1e-6
        fd = (N(1.0 + h) - N(1.0 - h)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-4)

    @pytest.mark.slow
    def test_pchip_shape_preserving_and_inverse(self, ew_curve):
        d, c = ew_curve
        xs = jnp.linspace(0.0, 1.0, 501)
        ys = c(xs)
        assert np.all(np.diff(np.asarray(ys)) > -1e-12)     # monotone
        assert np.allclose(c.inverse(ys), xs, atol=2e-4)
        assert np.allclose(c(d["x"]), d["y"], atol=1e-12)    # interpolates the data


class TestPlot:
    def test_plot_mccabe_thiele_runs(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        from difflow.visualization import plot_mccabe_thiele
        c = S.constant_alpha(AL)
        R = 2.2
        top, bot, ql = S.rectifying_line(R, XD), S.stripping_line(R, XD, XB, Z, 1.0), S.q_line(1.0, Z)
        res = S.mccabe_thiele(c, top, bot, XD, XB, q_line=ql)
        ax = plot_mccabe_thiele(res, c, top_line=top, bottom_line=bot, q_line=ql)
        assert len(ax.lines) >= 5
        labels = [t.get_text() for t in ax.texts]
        assert "1" in labels and str(int(np.ceil(float(res["N"])))) in labels
