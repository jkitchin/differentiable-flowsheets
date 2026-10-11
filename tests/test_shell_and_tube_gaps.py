"""Issue #435: wall-state ``shell_h``, optional tube viscosity correction, N shells in series,
shell-side Re range flag, F = 0 warning.

Property values are illustrative inputs. The viscosity exponents (0.11 / 0.25) are from memory and
unverified; these tests check the *implementation* of that stated form (arithmetic, limits,
gradients), not the exponents against a source.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import heat_transfer as ht
from difflow import shell_and_tube as st
from difflow.units.heat_exchanger import (
    LowCorrectionFactorWarning,
    ShellAndTubeHX,
    ShellAndTubeHXParams,
    lmtd_correction_factor,
)

from difflow import make_stream

# illustrative inputs (the same kind as tests/test_shell_and_tube.py)
DO, DI = 0.01905, 0.01483
SHELL = dict(rho=780.0, mu=4.0e-4, k=0.13, Cp=2200.0)
TUBE = dict(rho=830.0, mu=2.9e-3, k=0.13, Cp=2050.0)
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

STEAM = {"T": 373.15, "m_dot": 1.0}
CONDENSATE = dict(k_l=0.6, rho_l=960.0, rho_v=0.6, mu_l=2.8e-4, h_fg=2.26e6, Cp_l=4216.0)   # illustrative


def condensing(dT_fn):
    def shell_h(g):
        return ht.nusselt_film_condensation(**CONDENSATE, dT=dT_fn(g), L=g["tube_OD"], n_tubes=jnp.sqrt(g["n_tubes"]))
    return shell_h


# ---------------------------------------------------------------------------
# 1. wall state to shell_h
# ---------------------------------------------------------------------------

class TestWallStateShellH:
    def design(self, dT_fn, **kw):
        return st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=condensing(dT_fn),
                                                 tube_length=3.0, tol=1e-13, **kw))

    def test_loop_closes_and_differs_from_fixed_guess(self):
        d = self.design(lambda g: g["dT_film"])
        r = d(STEAM, COLD, Q=6.0e5)
        assert bool(r["converged"]) and float(r["U_residual"]) < 1e-9
        # the film dT the callable saw is the flux over the film coefficient it returned
        dT = float(r["q_flux"] / r["h_o"])
        assert 3.0 < dT < 30.0
        # a fixed guess that is the converged value reproduces the design
        fixed = self.design(lambda g: dT)
        rf = fixed(STEAM, COLD, Q=6.0e5)
        assert float(rf["area"]) == pytest.approx(float(r["area"]), rel=1e-7)
        # a different fixed guess does not
        for guess in (5.0, 8.0):
            ra = self.design(lambda g, x=guess: x)(STEAM, COLD, Q=6.0e5)
            assert abs(float(ra["area"]) / float(r["area"]) - 1.0) > 0.01

    def test_wall_temperature_consistent(self):
        seen = {}

        def shell_h(g):
            seen.update(T_wall=g["T_wall"], dT=g["dT_film"], q=g["q_flux"], Ts=g["T_shell"], Tt=g["T_tube"])
            return condensing(lambda gg: gg["dT_film"])(g)
        d = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=shell_h, tube_length=3.0, tol=1e-13))
        r = d(STEAM, COLD, Q=6.0e5, warn=False)
        # re-evaluate at the solution to read the final wall state
        assert float(seen["Ts"]) == pytest.approx(373.15)
        assert float(seen["T_wall"]) < 373.15 and float(seen["T_wall"]) > float(seen["Tt"])
        assert float(seen["Ts"] - seen["T_wall"]) == pytest.approx(float(r["q_flux"] / r["h_o"]), rel=1e-6)
        assert float(seen["dT"]) == pytest.approx(float(seen["Ts"] - seen["T_wall"]), rel=1e-12)

    def test_rating_agrees_with_design(self):
        d = self.design(lambda g: g["dT_film"])
        r = d(STEAM, COLD, Q=6.0e5)
        rt = d.rate(STEAM, COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"])
        assert float(rt["Q"]) == pytest.approx(6.0e5, rel=1e-8)
        assert float(rt["h_o"]) == pytest.approx(float(r["h_o"]), rel=1e-8)

    def test_boiling_shell_side_cold(self):
        """Boiling shell side (cold isothermal, hot liquid in the tubes): wall hotter than T_sat."""
        def shell_h(g):    # Mostinski nucleate boiling: h depends on the flux the loop supplies
            return ht.mostinski(g["q_flux"], 4600.0, 1000.0)     # illustrative P_c, P in kPa
        p = make_params(cold_isothermal=True, tube_side="hot", shell_h=shell_h, tube_length=3.0, tol=1e-13)
        r = st.ShellAndTubeDesign(p)({"T": 480.0, "m_dot": 4.0}, {"T": 400.0, "m_dot": 1.0}, Q=4.0e5, warn=False)
        assert bool(r["converged"])
        assert float(r["h_o"]) == pytest.approx(float(ht.mostinski(r["q_flux"], 4600.0, 1000.0)), rel=1e-9)
        assert float(r["h_o"]) > 0.0

    def test_gradients(self):
        d = self.design(lambda g: g["dT_film"])
        f = lambda Q: d(STEAM, COLD, Q=Q, warn=False)["area"]
        check_grads(f, (6.0e5,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)
        g = jax.grad(f)(6.0e5)
        h = 1e3
        assert float(g) == pytest.approx(float((f(6.0e5 + h) - f(6.0e5 - h)) / (2 * h)), rel=1e-5)
        # and in a fouling resistance
        fR = lambda R: d(STEAM, COLD, Q=6.0e5, R_fo=R, warn=False)["area"]
        assert float(jax.grad(fR)(1e-4)) > 0.0

    def test_geometry_only_callable_unchanged(self):
        """A callable that reads only the old keys gives the old answer."""
        old = lambda g: 6000.0 + 0.0 * g["area"]
        d = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=old, tube_length=3.0))
        ref = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=6000.0, tube_length=3.0))
        assert float(d(STEAM, COLD, Q=6.0e5)["area"]) == pytest.approx(float(ref(STEAM, COLD, Q=6.0e5)["area"]), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. optional viscosity correction
# ---------------------------------------------------------------------------

class TestViscosityCorrection:
    Re, Pr = 2.0e4, 6.0

    def test_off_by_default_and_ratio_one_is_neutral(self):
        base = ht.internal_nusselt(self.Re, self.Pr)
        assert float(ht.internal_nusselt(self.Re, self.Pr, mu_ratio=0.5)) == float(base)      # ignored, as before
        assert float(ht.internal_nusselt(self.Re, self.Pr, mu_ratio=1.0, viscosity_correction=True)) == pytest.approx(float(base), rel=1e-14)

    @pytest.mark.parametrize("method", ["gnielinski", "dittus_boelter"])
    @pytest.mark.parametrize("heating", [True, False])
    def test_factor_is_ratio_to_the_n(self, method, heating):
        n = ht.VISCOSITY_EXPONENT_HEATING if heating else ht.VISCOSITY_EXPONENT_COOLING
        base = ht.internal_nusselt(self.Re, self.Pr, method=method, heating=heating)
        corr = ht.internal_nusselt(self.Re, self.Pr, mu_ratio=1.7, method=method, heating=heating, viscosity_correction=True)
        assert float(corr / base) == pytest.approx(1.7**n, rel=1e-12)

    def test_sieder_tate_gets_no_second_factor(self):
        a = ht.internal_nusselt(self.Re, self.Pr, mu_ratio=1.7, method="sieder_tate")
        b = ht.internal_nusselt(self.Re, self.Pr, mu_ratio=1.7, method="sieder_tate", viscosity_correction=True)
        assert float(a) == float(b)

    def test_laminar_unaffected_and_smooth_gradient(self):
        lam = ht.internal_nusselt(500.0, self.Pr, mu_ratio=2.0, viscosity_correction=True)
        assert float(lam) == pytest.approx(float(ht.internal_nusselt(500.0, self.Pr, mu_ratio=2.0)), rel=1e-14)
        f = lambda Re, mr: ht.internal_nusselt(Re, self.Pr, mu_ratio=mr, viscosity_correction=True)
        check_grads(f, (3000.0, 1.3), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)   # in the blend region
        check_grads(f, (3.0e4, 1.3), order=1, modes=["rev"], rtol=1e-5, atol=1e-8)

    def test_design_uses_mu_wall_only_when_enabled(self):
        base = dict(tube_velocity=1.5, tube_length=None, mu_wall_cold=1.5e-3)
        off = st.ShellAndTubeDesign(make_params(**base))(HOT, COLD, T_hot_out=373.0, warn=False)
        on = st.ShellAndTubeDesign(make_params(tube_viscosity_correction=True, **base))(HOT, COLD, T_hot_out=373.0, warn=False)
        none = st.ShellAndTubeDesign(make_params(tube_velocity=1.5, tube_length=None))(HOT, COLD, T_hot_out=373.0, warn=False)
        assert float(off["h_i"]) == pytest.approx(float(none["h_i"]), rel=1e-14)   # unchanged default
        # mu_b/mu_w = 2.9e-3/1.5e-3 > 1, cold side heated: h_i rises by the factor (the same Re, Pr)
        ratio = (2.9e-3 / 1.5e-3) ** ht.VISCOSITY_EXPONENT_HEATING
        assert float(on["h_i"] / off["h_i"]) == pytest.approx(ratio, rel=2e-2)  # geometry shifts a little at the fixed point
        assert float(on["h_i"]) > float(off["h_i"])


# ---------------------------------------------------------------------------
# 6. shells in series
# ---------------------------------------------------------------------------

# 340 -> 316 K hot, 300 -> 324 K cold: R = 1, P = 0.6 > P_max(1 shell) = 0.586, a cross one shell cannot make
CROSS_HOT = {"T": 340.0, "m_dot": 5.0}
CROSS_COLD = {"T": 300.0, "m_dot": 5.0}
CROSS_Q = 5.0 * 2050.0 * 24.0


def cross_params(N, **kw):
    return make_params(rho_hot=780.0, Cp_hot=2050.0, n_shell_passes=N, tube_velocity=1.2, tube_length=None, **kw)


class TestShellsInSeries:
    def test_one_shell_unchanged(self):
        a = st.ShellAndTubeDesign(make_params())(HOT, COLD, T_hot_out=373.0, warn=False)
        b = st.ShellAndTubeDesign(make_params(n_shell_passes=1))(HOT, COLD, T_hot_out=373.0, warn=False)
        for k in ("U", "area", "F", "n_tubes", "shell_ID", "dP_tube", "dP_shell"):
            assert float(a[k]) == float(b[k])
        assert float(b["area_per_shell"]) == pytest.approx(float(b["area"]))

    def test_one_shell_cannot_two_can(self):
        R, P = 1.0, 0.6
        assert float(lmtd_correction_factor(R, P, 1)) == 0.0
        assert float(lmtd_correction_factor(R, P, 2)) > 0.5
        with pytest.warns(LowCorrectionFactorWarning, match="F = 0"):
            r1 = st.ShellAndTubeDesign(cross_params(1))(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
        assert bool(r1["F_zero"]) and float(r1["F"]) == 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r2 = st.ShellAndTubeDesign(cross_params(2))(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
        assert not bool(r2["F_zero"]) and bool(r2["converged"])
        assert math.isfinite(float(r2["area"])) and float(r2["area"]) > 0.0
        assert int(r2["n_shell_passes"]) == 2
        assert float(r2["area"]) == pytest.approx(2.0 * float(r2["area_per_shell"]), rel=1e-12)
        # Q = U A F LMTD over the total area
        assert float(r2["U"] * r2["area"] * r2["F"] * r2["LMTD"]) == pytest.approx(CROSS_Q, rel=1e-9)

    def test_more_shells_raise_F_and_cut_area(self):
        # a feasible duty for one shell: F rises and the total area falls with N
        kw = dict(T_hot_out=373.0)
        out = {}
        for N in (1, 2, 3):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out[N] = st.ShellAndTubeDesign(make_params(n_shell_passes=N, tube_velocity=1.5, tube_length=None))(
                    {"T": 473.0, "m_dot": 5.0}, {"T": 303.0, "m_dot": 18.0}, **kw)
        assert float(out[1]["F"]) < float(out[2]["F"]) <= float(out[3]["F"]) <= 1.0
        assert float(out[2]["area"]) < float(out[1]["area"])

    def test_pressure_drops_are_totals(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d = st.ShellAndTubeDesign(cross_params(2))
            r = d(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
        tube = st.tube_side_pressure_drop(CROSS_COLD["m_dot"], 830.0, 2.9e-3, r["n_tubes"], 2.0, 0.01483, r["tube_length"])
        assert float(r["dP_tube"]) == pytest.approx(2.0 * float(tube["dP"]), rel=1e-9)

    def test_design_vs_rating_agree_N2(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d = st.ShellAndTubeDesign(cross_params(2))
            r = d(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
            rt = d.rate(CROSS_HOT, CROSS_COLD, r["n_tubes"], r["tube_length"], r["shell_ID"], r["baffle_spacing"])
        assert float(rt["Q"]) == pytest.approx(CROSS_Q, rel=1e-8)
        assert float(rt["area"]) == pytest.approx(float(r["area"]), rel=1e-10)
        assert float(rt["dP_tube"]) == pytest.approx(float(r["dP_tube"]), rel=1e-9)
        assert float(rt["T_hot_out"]) == pytest.approx(float(r["T_hot_out"]), abs=1e-6)
        # the library rating exchanger agrees for the same UA and shells
        MW = 0.018   # arbitrary: C = m Cp is what matters
        hx = ShellAndTubeHX(ShellAndTubeHXParams(UA=float(r["U"] * r["area"]), n_shell_passes=2,
                                                 Cp_hot=2050.0 * MW, Cp_cold=TUBE["Cp"] * MW))
        ho, co, info = hx(make_stream({"A": 5.0 / MW}, 340.0, 101325.0), make_stream({"A": 5.0 / MW}, 300.0, 101325.0))
        assert float(info["Q"]) == pytest.approx(CROSS_Q, rel=1e-7)
        assert float(ho["T"]) == pytest.approx(float(r["T_hot_out"]), abs=1e-5)
        assert float(co["T"]) == pytest.approx(float(r["T_cold_out"]), abs=1e-5)

    def test_round_design_then_rate_consistent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            d = st.ShellAndTubeDesign(cross_params(2))
            r = d(CROSS_HOT, CROSS_COLD, Q=CROSS_Q)
            geom = st.round_design(r)
            rt = d.rate(CROSS_HOT, CROSS_COLD, **geom)
        assert float(rt["area"]) >= 2.0 * math.pi * 0.01905 * geom["n_tubes"] * geom["tube_length"] * (1 - 1e-12)
        # rounding the tube count up and the baffles down leaves the rated duty within a few percent
        assert float(rt["Q"]) == pytest.approx(CROSS_Q, rel=0.1)

    def test_gradients_N2(self):
        d = st.ShellAndTubeDesign(cross_params(2))
        f = lambda Q: d(CROSS_HOT, CROSS_COLD, Q=Q, warn=False)["area"]
        check_grads(f, (CROSS_Q,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)
        fv = lambda v: d(CROSS_HOT, CROSS_COLD, Q=CROSS_Q, tube_velocity=v, warn=False)["area"]
        check_grads(fv, (1.2,), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    def test_F_zero_silent_under_jit(self):
        d = st.ShellAndTubeDesign(cross_params(1))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.jit(lambda Q: d(CROSS_HOT, CROSS_COLD, Q=Q)["F"])(CROSS_Q)


# ---------------------------------------------------------------------------
# 7. Kern range flag
# ---------------------------------------------------------------------------

class TestKernRange:
    def test_laminar_shell_flag_and_warning(self):
        # a viscous shell fluid at low flow: Re_s of a few hundred
        p = make_params(mu_hot=0.05, rho_hot=900.0, k_hot=0.14, Cp_hot=2000.0, tube_velocity=1.5, tube_length=None)
        hot = {"T": 400.0, "m_dot": 0.8}
        with pytest.warns(UserWarning, match="shell-side Re"):
            r = st.ShellAndTubeDesign(p)(hot, COLD, T_hot_out=350.0)
        assert bool(r["shell_laminar"]) and bool(r["shell_Re_out_of_range"])
        assert float(r["Re_shell"]) < st.KERN_RE_MIN

    def test_in_range_is_quiet_and_flags_false(self):
        d = st.ShellAndTubeDesign(make_params(tube_velocity=1.5, tube_length=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            r = d(HOT, COLD, T_hot_out=373.0)
        assert st.KERN_RE_MIN < float(r["Re_shell"]) < st.KERN_RE_MAX
        assert not bool(r["shell_laminar"]) and not bool(r["shell_Re_out_of_range"])

    def test_too_turbulent_flag(self):
        p = make_params(mu_hot=2e-5, tube_velocity=1.5, tube_length=None)   # gas-like viscosity, huge Re_s
        r = st.ShellAndTubeDesign(p)({"T": 473.0, "m_dot": 50.0}, COLD, T_hot_out=373.0, warn=False)
        assert float(r["Re_shell"]) > st.KERN_RE_MAX
        assert bool(r["shell_Re_out_of_range"]) and not bool(r["shell_laminar"])

    def test_phase_change_shell_has_no_flag(self):
        d = st.ShellAndTubeDesign(make_params(hot_isothermal=True, shell_h=6000.0, tube_length=3.0))
        r = d(STEAM, COLD, Q=6.0e5)
        assert not bool(r["shell_Re_out_of_range"])

    def test_silent_under_grad(self):
        d = st.ShellAndTubeDesign(make_params(mu_hot=0.05, tube_velocity=1.5, tube_length=None))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            jax.grad(lambda m: d({"T": 400.0, "m_dot": m}, COLD, T_hot_out=350.0)["area"])(0.8)
