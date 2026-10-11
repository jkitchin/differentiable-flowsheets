"""Tests for the generic Absorber / Stripper (#428).

What is and is not verified
---------------------------
Verified against independent oracles:

* the Kremser fraction against a plain-Python closed form, continuity through
  ``A = 1``, and the ``A < 1`` limit (removal tends to ``A``);
* ``model="stages"`` against a separate SciPy ``fsolve`` of the nonlinear
  tray-to-tray equations (mole-fraction form, total flows as unknowns), and
  ``"kremser"`` against ``"stages"`` for dilute linear equilibrium, with and
  without a Murphree efficiency (the efficiency formula is exact there), and
  their divergence growing with concentration;
* the packed model against Colburn's closed form (written out here), the
  numerical ``N_OG`` integral, ``Z = H_OG N_OG`` and rating/design inversion;
* ``minimum_solvent_rate`` / ``minimum_stripping_gas_rate`` against a dense
  NumPy chord-slope scan (rich-end and tangent pinches, a tabulated curve),
  the closed form in the straight case, and the smooth blow-up of the stage
  count as ``L -> L_min``;
* the stripper against the stripping-factor Kremser relation
  ``x_out/x_in = (S-1)/(S^(N+1)-1)`` and a TCE-in-water case worked by hand;
* per-species mass balances in every mode, ``check_grads`` on the outlet solute
  flow, and an absorber-stripper flowsheet with solvent recycle through
  ``Flowsheet.solve`` differentiated with respect to the solvent rate.

NOT verified: no published textbook worked example (Seader-Henley-Roper Ch. 6,
McCabe-Smith-Harriott Ch. 18, Geankoplis Ch. 10, Wankat Ch. 12/16, Crittenden
et al. Ch. 14) is reproduced. The issue quotes them only "from recollection"
and the inputs and printed answers could not be retrieved, so quoting a number
would not be a validation. The TCE case below is a hand calculation with the
(from-memory) Henry's constant of ``HENRY_WATER``, not the cited example. The
Henry's constants and van 't Hoff coefficients are themselves unverified
order-of-magnitude values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import fsolve

from difflow import Flowsheet, make_stream
from difflow.flowsheet import Unit
from difflow.mass_transfer import PACKINGS, fuller_diffusivity, wilke_chang_diffusivity
from difflow.staged import kremser_fraction, kremser_stages, tabulated
from difflow.units.absorption import (
    HENRY_WATER,
    Absorber,
    AbsorptionParams,
    Stripper,
    AbsorptionParams,
    henry_constant,
    henry_params,
    kremser_absorption,
    minimum_solvent_rate,
    minimum_stripping_gas_rate,
    solute_free_operating_line,
)

jax.config.update("jax_enable_x64", True)

T0, P0 = 298.15, 101325.0


def _np_fraction(A, N):
    return (A ** (N + 1) - A) / (A ** (N + 1) - 1.0)


def _acetone():
    return henry_params(["acetone"])


def _streams(y=0.02, V=1.0, L=2.0, solute="acetone"):
    gas = make_stream({"air": V * (1 - y), solute: V * y}, T0, P0)
    liq = make_stream({"H2O": L}, T0, P0)
    return gas, liq


def _absorber(**kw):
    henry, C = _acetone()
    kw.setdefault("henry", henry)
    kw.setdefault("henry_C", C)
    return Absorber(AbsorptionParams(**kw))


def _np_stages(y_in, V_in_inert, L_inert, K, N, E=1.0):
    """Independent tray-to-tray solve (mole fractions, one solute) with fsolve."""
    # unknowns: x_1..x_N (liquid leaving stage n), y_1..y_N; stage 1 is the top, x_0 = 0, y_{N+1} = y_in
    def total_flows(x, y):
        # inert-based totals: V_n = Gc/(1-y_n), L_n = Lc/(1-x_n)
        return V_in_inert / (1.0 - y), L_inert / (1.0 - x)

    def resid(u):
        x, y = u[:N], u[N:]
        V, L = total_flows(x, y)
        y_next = np.append(y[1:], y_in)
        x_prev = np.insert(x[:-1], 0, 0.0)
        V_next = np.append(V[1:], V_in_inert / (1.0 - y_in))
        L_prev = np.insert(L[:-1], 0, L_inert)
        r1 = y - (y_next + E * (K * x - y_next))
        r2 = L_prev * x_prev + V_next * y_next - L * x - V * y
        return np.concatenate([r1, r2])

    u0 = np.concatenate([np.full(N, y_in / K / 2.0), np.full(N, y_in / 2.0)])
    u = fsolve(resid, u0, xtol=1e-14)
    assert np.max(np.abs(resid(u))) < 1e-12
    x, y = u[:N], u[N:]
    return y[0], x[-1]


# ---------------------------------------------------------------------------
# Kremser
# ---------------------------------------------------------------------------


class TestKremser:
    @pytest.mark.parametrize("A", [0.3, 0.9, 1.7, 5.0])
    @pytest.mark.parametrize("N", [1.0, 3.0, 7.5])
    def test_closed_form(self, A, N):
        assert float(kremser_absorption(A, N)) == pytest.approx(_np_fraction(A, N), rel=1e-12)

    def test_continuous_through_A_equal_one(self):
        N = 6.0
        vals = [float(kremser_absorption(1.0 + d, N)) for d in (-1e-6, -1e-9, 0.0, 1e-9, 1e-6)]
        assert vals[2] == pytest.approx(N / (N + 1.0), rel=1e-12)
        assert np.all(np.abs(np.array(vals) - N / (N + 1.0)) < 1e-5)
        g = jax.grad(lambda a: kremser_absorption(a, N))(1.0)
        assert np.isfinite(g)

    def test_A_below_one_is_bounded_by_A(self):
        for A in (0.2, 0.7, 0.95):
            f = float(kremser_absorption(A, 400.0))
            assert f == pytest.approx(A, rel=1e-9)
            assert float(kremser_absorption(A, 5.0)) < A

    def test_unit_matches_closed_form_fraction(self):
        """Pure solvent, dilute: recovery equals the Kremser fraction at the unit's own A."""
        ab = _absorber(n_stages=6.0, variable_flow=False)
        gas, liq = _streams(y=1e-9, L=2.0)
        _, _, info = ab(gas, liq)
        K = HENRY_WATER["acetone"].H298 / P0
        A = 2.0 / (K * 1.0)
        assert float(info["A"]["acetone"]) == pytest.approx(A, rel=1e-12)
        assert float(info["recovery"]["acetone"]) == pytest.approx(_np_fraction(A, 6.0), rel=1e-12)

    def test_solute_in_entering_solvent(self):
        """v_out = v_in (1-f) + (l_in/A) f with a non-clean solvent."""
        ab = _absorber(n_stages=4.0, variable_flow=False)
        gas, _ = _streams(y=1e-9)
        liq = make_stream({"H2O": 2.0, "acetone": 1e-9}, T0, P0)
        g, l, info = ab(gas, liq)
        K = HENRY_WATER["acetone"].H298 / P0
        A = 2.0 / K
        f = _np_fraction(A, 4.0)
        v_in, l_in = 1e-9 * (1 - 1e-9) * 0 + float(gas["F_acetone"]), 1e-9
        expect = v_in * (1 - f) + l_in / A * f
        assert float(g["F_acetone"]) == pytest.approx(expect, rel=1e-9)

    def test_stages_for_recovery_matches_closed_form(self):
        ab = _absorber(recovery=0.95, variable_flow=False)
        gas, liq = _streams(y=1e-9, L=3.0)
        _, _, info = ab(gas, liq)
        K = HENRY_WATER["acetone"].H298 / P0
        A = 3.0 / K
        N = np.log((A - 0.95) / (1 - 0.95)) / np.log(A) - 1.0
        assert float(info["n_stages"]) == pytest.approx(N, rel=1e-9)
        assert float(info["recovery"]["acetone"]) == pytest.approx(0.95, abs=1e-12)


# ---------------------------------------------------------------------------
# Kremser vs stage-by-stage
# ---------------------------------------------------------------------------


class TestStagesMode:
    @pytest.mark.parametrize("N", [1, 3, 8])
    def test_against_independent_numpy_solve(self, N):
        y = 0.03
        ab = _absorber(model="stages", n_stages=N)
        gas, liq = _streams(y=y, V=1.0, L=2.0)
        g, l, info = ab(gas, liq)
        assert bool(info["converged"])
        K = HENRY_WATER["acetone"].H298 / P0
        y_top, x_bot = _np_stages(y, 1.0 - y, 2.0, K, N)
        assert float(g["F_acetone"] / (g["F_acetone"] + g["F_air"])) == pytest.approx(y_top, rel=1e-8)
        assert float(l["F_acetone"] / (l["F_acetone"] + l["F_H2O"])) == pytest.approx(x_bot, rel=1e-8)

    def test_murphree_against_numpy(self):
        y, N, E = 0.02, 5, 0.6
        ab = _absorber(model="stages", n_stages=N, stage_efficiency=E)
        gas, liq = _streams(y=y)
        g, l, info = ab(gas, liq)
        K = HENRY_WATER["acetone"].H298 / P0
        y_top, _ = _np_stages(y, 1.0 - y, 2.0, K, N, E=E)
        assert float(g["F_acetone"] / (g["F_acetone"] + g["F_air"])) == pytest.approx(y_top, rel=1e-8)

    @pytest.mark.parametrize("E", [1.0, 0.7])
    def test_dilute_linear_kremser_agrees_to_1e4(self, E):
        gas, liq = _streams(y=1e-6, L=2.0)
        out = {}
        for model in ("kremser", "stages"):
            ab = _absorber(model=model, n_stages=6, stage_efficiency=E)
            _, _, info = ab(gas, liq)
            out[model] = float(info["recovery"]["acetone"])
        assert abs(out["kremser"] - out["stages"]) < 1e-4

    def test_concentrated_case_diverges_in_the_expected_direction(self):
        """The gap grows with concentration (Kremser's mean A is only an approximation)."""
        gaps = []
        for y in (1e-6, 0.02, 0.08, 0.15):
            gas, liq = _streams(y=y, L=2.0)
            r = {}
            for model in ("kremser", "stages"):
                ab = _absorber(model=model, n_stages=6)
                r[model] = float(ab(gas, liq)[2]["recovery"]["acetone"])
            gaps.append(abs(r["kremser"] - r["stages"]))
        assert gaps[0] < 1e-4
        assert gaps == sorted(gaps)
        assert gaps[-1] > 1e-3

    def test_fractional_or_traced_stage_count_is_refused(self):
        gas, liq = _streams()
        with pytest.raises(ValueError, match="integer"):
            _absorber(model="stages", n_stages=3.5)(gas, liq)
        with pytest.raises(ValueError, match="design"):
            _absorber(model="stages", recovery=0.9)(gas, liq)


# ---------------------------------------------------------------------------
# Minimum solvent rate
# ---------------------------------------------------------------------------


def _brute_min_slope(Ystar, Y_out, X_in, X_end, maximize=True, n=400001):
    X = np.linspace(X_in, X_end, n)[1:]
    s = (Ystar(X) - Y_out) / (X - X_in)
    return s.max() if maximize else s.min()


class TestMinimumRate:
    def test_straight_closed_form(self):
        """Henry's law at K > 1: the pinch is at the rich end."""
        from difflow.staged import linear_equilibrium, to_ratio

        K = 2.0
        curve = linear_equilibrium(K)
        y_in, y_out, G = 0.05, 0.005, 3.0
        Y_in, Y_out = y_in / (1 - y_in), y_out / (1 - y_out)
        Ls = float(minimum_solvent_rate(curve, Y_in, Y_out, 0.0, G))
        x_end = y_in / K
        X_end = x_end / (1 - x_end)
        assert Ls == pytest.approx(G * (Y_in - Y_out) / X_end, rel=1e-9)

    @pytest.mark.parametrize("K", [0.5, 2.0, 30.0])
    def test_against_dense_scan(self, K):
        from difflow.staged import linear_equilibrium

        curve = linear_equilibrium(K)
        Y_in, Y_out, X_in, G = 0.08, 0.004, 0.00005, 5.0
        X_end = (Y_in / (1 + Y_in) / K)
        X_end = X_end / (1 - X_end)

        def Ystar(X):
            x = X / (1 + X)
            y = K * x
            return y / (1 - y)

        ref = _brute_min_slope(Ystar, Y_out, X_in, X_end) * G
        assert float(minimum_solvent_rate(curve, Y_in, Y_out, X_in, G)) == pytest.approx(ref, rel=1e-6)

    def test_tangent_pinch_on_a_tabulated_curve(self):
        # an S-shaped (inflected) equilibrium curve in mole fractions: the pinch is interior
        x = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10])
        y = np.array([0.0, 0.002, 0.004, 0.009, 0.022, 0.050, 0.080, 0.105, 0.120])
        curve = tabulated(x, y)
        Y_in, Y_out, X_in, G = 0.12 / 0.88, 0.0002, 0.0, 1.0
        X_end = float(curve.inverse(Y_in / (1 + Y_in)))
        X_end = X_end / (1 - X_end)

        def Ystar(X):
            xx = X / (1 + X)
            yy = np.asarray(curve(xx))
            return yy / (1 - yy)

        ref = _brute_min_slope(Ystar, Y_out, X_in, X_end) * G
        got = float(minimum_solvent_rate(curve, Y_in, Y_out, X_in, G))
        assert got == pytest.approx(ref, rel=1e-5)
        # and the pinch is not at the rich end
        end_slope = (Ystar(np.array([X_end]))[0] - Y_out) / (X_end - X_in) * G
        assert got > end_slope * (1 + 1e-3)

    def test_infeasible_spec_is_inf_not_nan(self):
        from difflow.staged import linear_equilibrium

        curve = linear_equilibrium(2.0)
        # Y_out below equilibrium with the entering liquid
        v = float(minimum_solvent_rate(curve, 0.05, 0.001, 0.01, 1.0))
        assert np.isinf(v)

    def test_stripper_mirror_closed_form(self):
        from difflow.staged import linear_equilibrium

        K = 3.0
        curve = linear_equilibrium(K)
        X_in, X_out, L_s = 0.002, 0.0001, 4.0

        def Ystar(X):
            x = X / (1 + X)
            y = K * x
            return y / (1 - y)

        ref = L_s / _brute_min_slope(Ystar, 0.0, X_out, X_in, maximize=False)
        got = float(minimum_stripping_gas_rate(curve, X_in, X_out, 0.0, L_s))
        assert got == pytest.approx(ref, rel=1e-6)

    def test_operating_line_is_straight_in_ratio_coordinates(self):
        slope, icpt = solute_free_operating_line(2.0, 5.0, 0.001, 0.01)
        assert float(slope) == 2.5
        # G_s (Y_in - Y_out) = L_s (X_out - X_in) holds along the line
        Y_in, X_out = 0.1, (0.1 - float(icpt)) / 2.5
        assert 2.0 * (Y_in - 0.01) == pytest.approx(5.0 * (X_out - 0.001), rel=1e-12)

    def test_stage_count_blows_up_smoothly_at_L_min(self):
        """Dilute design at recovery r: A_min = r; N grows without bound as L -> L_min+."""
        K = HENRY_WATER["acetone"].H298 / P0
        r = 0.9
        gas, _ = _streams(y=1e-9, V=1.0)
        Lmin = r * K * 1.0
        N = []
        for d in (1.5, 1.1, 1.01, 1.001, 1.0001, 1.00001):
            ab = _absorber(recovery=r, variable_flow=False)
            _, _, info = ab(gas, make_stream({"H2O": Lmin * d}, T0, P0))
            n = float(info["n_stages"])
            assert np.isfinite(n)
            N.append(n)
        assert N == sorted(N) and N[-1] > 8 * N[1]
        # at and below L_min the specification is unreachable: inf, never NaN, and flagged
        ab = _absorber(recovery=r, variable_flow=False)
        _, _, info = ab(gas, make_stream({"H2O": Lmin * 0.9}, T0, P0))
        assert np.isinf(float(info["n_stages"])) and not bool(info["feasible"])
        assert not np.isnan(float(info["recovery"]["acetone"]))

    def test_one_and_a_half_times_minimum_by_hand(self):
        """L = 1.5 L_min, dilute straight lines: N from the closed form (no textbook number claimed)."""
        K = HENRY_WATER["acetone"].H298 / P0
        r = 0.95
        gas, _ = _streams(y=1e-9)
        ab = _absorber(recovery=r, variable_flow=False)
        _, _, info = ab(gas, make_stream({"H2O": 1.5 * r * K}, T0, P0))
        A = 1.5 * r
        N = np.log((A - r) / (1 - r)) / np.log(A) - 1.0
        assert float(info["n_stages"]) == pytest.approx(N, rel=1e-9)
        assert float(info["L_over_Lmin"]) == pytest.approx(1.5, rel=1e-4)

    def test_info_L_min_matches_library(self):
        ab = _absorber(n_stages=6.0)
        gas, liq = _streams(y=0.02, L=2.0)
        _, _, info = ab(gas, liq)
        assert float(info["L_min"]) < 2.0 and float(info["L_over_Lmin"]) > 1.0
        # a stage-limited absorber cannot reach beyond its L_min
        assert float(info["L_min"]) * float(info["L_over_Lmin"]) == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Packed
# ---------------------------------------------------------------------------


def _packed(**kw):
    henry, C = _acetone()
    base = dict(henry=henry, henry_C=C, model="packed", variable_flow=False, H_OG=0.6)
    base.update(kw)
    return Absorber(AbsorptionParams(**base))


class TestPacked:
    def test_rating_matches_colburn_closed_form(self):
        gas, liq = _streams(y=1e-9, L=2.0)
        _, _, info = _packed(height=3.0)(gas, liq)
        K = HENRY_WATER["acetone"].H298 / P0
        lam = K / 2.0
        N = 3.0 / 0.6
        phi = (1 - lam) / (np.exp((1 - lam) * N) - lam)
        assert float(info["recovery"]["acetone"]) == pytest.approx(1 - phi, rel=1e-9)
        assert float(info["N_OG"]["acetone"]) == pytest.approx(N, rel=1e-12)

    def test_design_height_is_H_OG_times_N_OG_and_rating_inverts(self):
        gas, liq = _streams(y=1e-9, L=2.0)
        _, _, d = _packed(recovery=0.9)(gas, liq)
        Z = float(d["height"])
        assert Z == pytest.approx(0.6 * float(d["N_OG"]["acetone"]), rel=1e-12)
        _, _, r = _packed(height=Z)(gas, liq)
        assert float(r["recovery"]["acetone"]) == pytest.approx(0.9, abs=1e-10)

    def test_colburn_equals_numerical_integral_for_straight_lines(self):
        gas, liq = _streams(y=1e-9, L=2.0)
        _, _, c = _packed(recovery=0.9)(gas, liq)
        _, _, i = _packed(recovery=0.9, ntu_method="integral")(gas, liq)
        assert float(i["height"]) == pytest.approx(float(c["height"]), rel=1e-6)
        assert float(i["N_OG_integral"]) == pytest.approx(float(c["N_OG"]["acetone"]), rel=1e-6)

    def test_concentrated_integral_differs_from_dilute(self):
        gas, liq = _streams(y=0.1, L=2.0)
        _, _, a = _packed(recovery=0.9, ntu_method="integral")(gas, liq)
        _, _, b = _packed(recovery=0.9, ntu_method="integral", concentrated=True)(gas, liq)
        assert float(a["N_OG_integral"]) != pytest.approx(float(b["N_OG_integral"]), rel=1e-3)
        assert np.isfinite(float(b["N_OG_integral"]))

    def test_onda_films_assemble_into_H_OG(self):
        """H_OG = H_G + lambda H_L with the Onda coefficients (built here from the library functions)."""
        from difflow.mass_transfer import (
            onda_gas_coefficient,
            onda_liquid_coefficient,
            onda_wetted_area,
        )

        pk = PACKINGS["pall_metal_50"]
        D_G, D_L = 1.1e-5, 1.2e-9
        D = 0.5
        gas, liq = _streams(y=1e-9, V=2.0, L=60.0)
        ab = _packed(H_OG=None, packing="pall_metal_50", diameter=D, height=2.0,
                     D_G={"acetone": D_G}, D_L={"acetone": D_L})
        _, _, info = ab(gas, liq)
        area = np.pi * D**2 / 4
        V, L = 2.0, 60.0
        T, P = T0, P0
        R = 8.314462618
        u_G = V * R * T / (P * area)
        u_L = L * 0.018015 / (997.0 * area)
        a_w = onda_wetted_area(u_L, 997.0, 8.9e-4, 0.072, pk)
        k_c = onda_gas_coefficient(u_G, P * 0.02897 / (R * T), 1.85e-5, D_G, pk)
        k_L = onda_liquid_coefficient(u_L, 997.0, 8.9e-4, D_L, 0.072, pk, a_w=a_w)
        H_G = (V / area) / (k_c * P / (R * T) * a_w)
        H_L = (L / area) / (k_L * 997.0 / 0.018015 * a_w)
        K = HENRY_WATER["acetone"].H298 / P
        lam = K * V / L
        assert float(info["H_OG"]["acetone"]) == pytest.approx(float(H_G + lam * H_L), rel=1e-12)
        assert 0.05 < float(info["H_OG"]["acetone"]) < 20.0
        assert float(info["height"]) == 2.0

    def test_packed_requires_height_or_recovery(self):
        gas, liq = _streams()
        with pytest.raises(ValueError, match="height"):
            _packed()(gas, liq)
        with pytest.raises(ValueError, match="not both"):
            _packed(height=2.0, recovery=0.9)(gas, liq)

    def test_Onda_path_needs_diffusivities(self):
        gas, liq = _streams()
        with pytest.raises(ValueError, match="D_G"):
            _packed(H_OG=None, packing="pall_metal_50", diameter=0.5, height=2.0)(gas, liq)


# ---------------------------------------------------------------------------
# Stripper
# ---------------------------------------------------------------------------


def _stripper(**kw):
    henry, C = henry_params(["TCE"])
    kw.setdefault("henry", henry)
    kw.setdefault("henry_C", C)
    return Stripper(AbsorptionParams(**kw))


class TestStripper:
    def _tce_streams(self, S, L=1000.0):
        K = HENRY_WATER["TCE"].H298 / P0
        V = S * L / K
        gas = make_stream({"air": V}, T0, P0)
        liq = make_stream({"H2O": L, "TCE": 1e-9}, T0, P0)
        return gas, liq, K

    def test_matches_stripping_factor_kremser_by_hand(self):
        """x_out/x_in = (S-1)/(S^(N+1)-1) with S = K V/L: a hand-worked TCE-in-water case."""
        S, N = 3.0, 6
        gas, liq, K = self._tce_streams(S)
        _, l_out, info = _stripper(n_stages=float(N), variable_flow=False)(gas, liq)
        remaining = (S - 1.0) / (S ** (N + 1) - 1.0)
        assert float(l_out["F_TCE"] / liq["F_TCE"]) == pytest.approx(remaining, rel=1e-9)
        assert float(info["recovery"]["TCE"]) == pytest.approx(1 - remaining, rel=1e-9)
        assert float(info["S"]["TCE"]) == pytest.approx(S, rel=1e-12)

    def test_design_stages_and_G_min_for_99_percent_removal(self):
        S = 4.0
        gas, liq, K = self._tce_streams(S)
        r = 0.99
        _, _, info = _stripper(recovery=r, variable_flow=False)(gas, liq)
        phi = 1 - r
        N = np.log(1.0 + (S - 1.0) / phi) / np.log(S) - 1.0
        assert float(info["n_stages"]) == pytest.approx(N, rel=1e-9)
        # dilute minimum gas rate: S_min = r (rich-end pinch with a clean gas)
        assert float(info["G_over_Gmin"]) == pytest.approx(S / r, rel=1e-4)

    def test_stages_mode_agrees_with_kremser_for_a_stripper(self):
        gas, liq, _ = self._tce_streams(3.0)
        out = {}
        for m in ("kremser", "stages"):
            _, _, i = _stripper(model=m, n_stages=5, variable_flow=m == "kremser")(gas, liq)
            out[m] = float(i["recovery"]["TCE"])
        assert abs(out["kremser"] - out["stages"]) < 1e-6

    def test_packed_stripper_design(self):
        gas, liq, K = self._tce_streams(3.0)
        _, l, info = _stripper(model="packed", H_OG=0.8, recovery=0.99, variable_flow=False)(gas, liq)
        assert float(info["recovery"]["TCE"]) == pytest.approx(0.99, abs=1e-10)
        assert float(info["height"]) == pytest.approx(0.8 * float(info["N_OG"]["TCE"]), rel=1e-12)
        _, _, i2 = _stripper(model="packed", H_OG=0.8, height=float(info["height"]),
                             variable_flow=False)(gas, liq)
        assert float(i2["recovery"]["TCE"]) == pytest.approx(0.99, abs=1e-10)

    def test_stripping_cannot_beat_its_factor(self):
        """S < 1: infinite stages remove only the fraction S."""
        gas, liq, K = self._tce_streams(0.5)
        _, _, i = _stripper(n_stages=500.0, variable_flow=False)(gas, liq)
        assert float(i["recovery"]["TCE"]) == pytest.approx(0.5, rel=1e-9)

    def test_heat_of_absorption_sign(self):
        ab = _absorber(n_stages=6.0)
        _, _, i = ab(*_streams())
        assert float(i["Q_absorption"]) > 0 and float(i["delta_T_adiabatic"]) > 0
        gas, liq, _ = self._tce_streams(3.0)
        liq = make_stream({"H2O": 1000.0, "TCE": 1e-3}, T0, P0)
        _, _, j = _stripper(n_stages=5.0)(gas, liq)
        assert float(j["Q_absorption"]) < 0  # desorption is endothermic

    def test_van_t_hoff_temperature_dependence(self):
        H = float(henry_constant(2.0e5, 318.15, 5000.0))
        assert H == pytest.approx(2.0e5 * np.exp(-5000.0 * (1 / 318.15 - 1 / 298.15)), rel=1e-12)
        assert H > 2.0e5  # less soluble when warm


# ---------------------------------------------------------------------------
# Mass balances
# ---------------------------------------------------------------------------


MODES = [
    dict(model="kremser", n_stages=5.3),
    dict(model="kremser", recovery=0.9),
    dict(model="kremser", n_stages=4.0, stage_efficiency=0.6, variable_flow=False),
    dict(model="stages", n_stages=6),
    dict(model="packed", height=4.0, H_OG=0.7),
    dict(model="packed", recovery=0.8, H_OG=0.7),
]


@pytest.mark.parametrize("mode", MODES, ids=lambda m: "-".join(f"{k}={v}" for k, v in m.items()))
@pytest.mark.parametrize("cls", ["absorber", "stripper"])
def test_per_species_mass_balance(mode, cls):
    henry, C = henry_params(["acetone", "NH3", "benzene"])
    Params = AbsorptionParams if cls == "absorber" else AbsorptionParams
    Unit_ = Absorber if cls == "absorber" else Stripper
    mode = dict(mode)
    if cls == "stripper":
        if "recovery" in mode:
            mode["recovery"] = 0.5
        # clean stripping gas, rich liquid carrying all three solutes
        gas = make_stream({"air": 4.0}, T0, P0)
        liq = make_stream({"H2O": 3.0, "acetone": 0.02, "NH3": 0.004, "benzene": 0.0005}, T0, P0)
    else:
        gas = make_stream({"air": 1.0, "acetone": 0.02, "NH3": 0.004, "benzene": 0.0}, T0, P0)
        liq = make_stream({"H2O": 3.0, "acetone": 0.001, "benzene": 0.0005}, T0, P0)
    u = Unit_(Params(henry=henry, henry_C=C, key_solute="acetone", **mode))
    g, l, info = u(gas, liq)
    for s in ("air", "acetone", "NH3", "benzene", "H2O"):
        tin = float(gas.get(f"F_{s}", 0.0)) + float(liq.get(f"F_{s}", 0.0))
        tout = float(g.get(f"F_{s}", 0.0)) + float(l.get(f"F_{s}", 0.0))
        assert abs(tin - tout) < 1e-10, (s, tin, tout)
    assert float(g["F_air"]) == float(gas["F_air"])
    assert float(l["F_H2O"]) == 3.0
    # non-negative up to round-off (a fully transferred species can land at -1e-18)
    assert all(float(v) >= -1e-14 for v in list(g.values()) + list(l.values()))
    assert bool(info["converged"])


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _out_solute(model="kremser", **kw):
    """gas-out acetone flow as a function of (L, V, n_stages, E, H, height)."""
    def f(L, V, n, E, H, Z):
        params = dict(henry={"acetone": H}, henry_C={"acetone": 5000.0}, model=model,
                      stage_efficiency=E)
        if model == "packed":
            params.update(height=Z, H_OG=0.7, variable_flow=True)
        else:
            params.update(n_stages=n)
        u = Absorber(AbsorptionParams(**params))
        gas = make_stream({"air": V, "acetone": 0.02}, T0, P0)
        liq = make_stream({"H2O": L}, T0, P0)
        return u(gas, liq)[0]["F_acetone"]
    return f


ARGS = (2.0, 1.0, 5.4, 0.8, 2.0e5, 3.0)


@pytest.mark.parametrize("model", ["kremser", "packed"])
@pytest.mark.parametrize("i", range(6))
def test_check_grads_outlet_solute(model, i):
    f = _out_solute(model)
    if (model == "packed" and i in (2, 3)) or (model == "kremser" and i == 5):
        pytest.skip("parameter not used by this model")

    def g(x):
        a = list(ARGS)
        a[i] = x
        return f(*a)

    check_grads(g, (ARGS[i],), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)


def test_check_grads_design_and_stripper():
    henry, C = henry_params(["TCE"])

    def f(L):
        u = Absorber(AbsorptionParams(henry={"acetone": 2e5}, recovery=0.9))
        return u(make_stream({"air": 1.0, "acetone": 0.02}, T0, P0), make_stream({"H2O": L}, T0, P0))[2]["n_stages"]

    check_grads(f, (3.0,), order=1, modes=["rev"], atol=1e-5, rtol=1e-5)

    def g(V):
        u = Stripper(AbsorptionParams(henry=henry, henry_C=C, n_stages=5.0))
        return u(make_stream({"air": V}, T0, P0), make_stream({"H2O": 1000.0, "TCE": 0.01}, T0, P0))[1]["F_TCE"]

    check_grads(g, (4.0,), order=1, modes=["rev"], atol=1e-6, rtol=1e-5)


def test_gradient_of_stage_count_wrt_L_is_negative_and_matches_finite_difference():
    def n(L):
        u = Absorber(AbsorptionParams(henry={"acetone": 2e5}, recovery=0.9))
        return u(make_stream({"air": 1.0, "acetone": 0.02}, T0, P0), make_stream({"H2O": L}, T0, P0))[2]["n_stages"]

    g = float(jax.grad(n)(3.0))
    fd = (float(n(3.0 + 1e-5)) - float(n(3.0 - 1e-5))) / 2e-5
    assert g < 0 and g == pytest.approx(fd, rel=1e-5)


# ---------------------------------------------------------------------------
# Absorber-stripper flowsheet with solvent recycle
# ---------------------------------------------------------------------------


class SolventMakeup:
    """Ideal flow-controlled makeup/purge: rescales the recycled liquid so its water rate is ``L``."""

    def __init__(self, L):
        self.L = L

    def __call__(self, inlet):
        k = self.L / inlet["F_H2O"]
        return make_stream({"H2O": self.L, "acetone": inlet["F_acetone"] * k}, inlet["T"], inlet["P"])


def _loop(L):
    henry, C = _acetone()
    fs = Flowsheet(species_order=["acetone", "H2O"])
    fs.add_feed("gas", make_stream({"air": 1.0, "acetone": 0.02}, T0, P0))
    fs.add_feed("strip_air", make_stream({"air2": 0.6}, T0, P0))
    ab = Absorber(AbsorptionParams(henry=henry, henry_C=C, n_stages=6.0, T=T0))
    st = Stripper(AbsorptionParams(henry=henry, henry_C=C, n_stages=5.0, T=340.0))
    fs.add_unit(Unit("absorber", ab, ["gas", "lean"], ["treated", "rich"]))
    fs.add_unit(Unit("stripper", st, ["strip_air", "rich"], ["offgas", "lean_hot"]))
    fs.add_unit(Unit("makeup", SolventMakeup(L), ["lean_hot"], ["lean_set"]))
    fs.add_recycle("lean_set", "lean")
    guess = {"lean": make_stream({"acetone": 0.0, "H2O": L}, T0, P0)}
    return fs, guess


def _treated_acetone(L):
    fs, guess = _loop(L)
    streams = fs.solve(tear_initial=guess, tol=1e-12, max_iter=200)
    return streams["treated"]["F_acetone"]


class TestRecycleFlowsheet:
    def test_converges_and_balances(self):
        fs, guess = _loop(2.0)
        s = fs.solve(tear_initial=guess, tol=1e-12, max_iter=200)
        # acetone leaves only in the treated gas and the stripper off-gas
        a_in = float(s["gas"]["F_acetone"])
        a_out = float(s["treated"]["F_acetone"]) + float(s["offgas"]["F_acetone"])
        assert a_out == pytest.approx(a_in, rel=1e-9)
        # the recycle is closed: the absorber saw the stripper's lean liquid
        assert float(s["lean"]["F_H2O"]) == pytest.approx(2.0, rel=1e-12)
        assert float(s["lean"]["F_acetone"]) == pytest.approx(float(s["lean_hot"]["F_acetone"]), rel=1e-9)
        assert float(s["lean"]["F_acetone"]) > 0.0
        assert float(s["treated"]["F_acetone"]) < 0.2 * a_in

    def test_gradient_wrt_solvent_rate(self):
        L0 = 2.0
        g = float(jax.grad(_treated_acetone)(L0))
        h = 1e-4
        fd = (float(_treated_acetone(L0 + h)) - float(_treated_acetone(L0 - h))) / (2 * h)
        assert g < 0  # more solvent, cleaner gas
        assert g == pytest.approx(fd, rel=1e-4)

    def test_L_over_G_tradeoff_has_an_interior_optimum(self):
        """Treated-gas loss falls with L while a stripper-gas cost rises: the total has a minimum."""
        Ls = np.linspace(0.8, 6.0, 14)
        loss = np.array([float(_treated_acetone(L)) for L in Ls])
        cost = loss * 40.0 + 0.15 * Ls   # arbitrary weights: solute lost vs solvent circulation
        k = int(np.argmin(cost))
        assert 0 < k < len(Ls) - 1
