"""Tests for difflow.fluids: friction factors, fittings and flow meters.

What "validated" means here (no textbook number is quoted from memory):

* Colebrook is checked against an *independent* solution of the same implicit
  equation (scipy ``brentq``), to 1e-10.
* The two Colebrook limits are checked against their closed forms: the
  Prandtl smooth-pipe law and the von Karman fully rough law.
* Moody-chart spot values are the standard published ones to the 3 significant
  figures the chart can be read to (smooth pipe: Re = 1e4, 1e5, 1e6;
  fully rough: e/D = 0.05, 0.01, 0.001).
* Churchill / Haaland / Swamee-Jain are compared with Colebrook.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow import fluids
from difflow.fluids import (
    FITTINGS,
    darcy_pressure_drop,
    equivalent_length,
    fitting_K,
    friction_factor,
    fully_rough_friction_factor,
    hydraulic_diameter,
    minor_loss,
    orifice_dp,
    orifice_flow,
    reynolds_number,
    venturi_dp,
    venturi_flow,
)


def _colebrook_reference(Re, e):
    """Independent solution of Colebrook-White by bracketing."""
    x = brentq(lambda x: x + 2 * np.log10(e / 3.7 + 2.51 * x / Re), 1.0, 200.0, xtol=1e-15, rtol=1e-15)
    return x**-2


RES = np.logspace(np.log10(4000.0), 8, 25)
RELS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05]


class TestBasics:
    def test_reynolds(self):
        assert float(reynolds_number(1000.0, 2.0, 0.05, 1e-3)) == pytest.approx(1e5)

    def test_hydraulic_diameter(self):
        D = 0.1
        assert float(hydraulic_diameter(np.pi * D**2 / 4, np.pi * D)) == pytest.approx(D)
        Do, Di = 0.1, 0.06  # annulus: D_h = Do - Di
        area = np.pi / 4 * (Do**2 - Di**2)
        assert float(hydraulic_diameter(area, np.pi * (Do + Di))) == pytest.approx(Do - Di)

    def test_darcy_and_minor(self):
        assert float(darcy_pressure_drop(0.02, 100.0, 0.05, 1000.0, 2.0)) == pytest.approx(0.02 * 2000 * 0.5 * 1000 * 4)
        assert float(minor_loss(2.0, 1000.0, 3.0)) == pytest.approx(9000.0)

    def test_unknown_method(self):
        with pytest.raises(ValueError):
            friction_factor(1e5, 0.0, "blasius")


class TestFrictionFactor:
    @pytest.mark.parametrize("e", RELS)
    def test_colebrook_matches_independent_solution(self, e):
        f = np.array([float(friction_factor(Re, e)) for Re in RES])
        ref = np.array([_colebrook_reference(Re, e) for Re in RES])
        np.testing.assert_allclose(f, ref, rtol=1e-10)

    def test_smooth_limit_is_prandtl(self):
        for Re in [1e4, 1e5, 1e6, 1e7, 1e8]:
            f = float(friction_factor(Re, 0.0))
            # Prandtl's smooth-pipe law with the Colebrook constant 2.51:
            # 1/sqrt(f) = 2 log10(Re sqrt(f) / 2.51)  (= 2 log10(Re sqrt f) - 0.7996)
            assert 1 / np.sqrt(f) == pytest.approx(2 * np.log10(Re * np.sqrt(f) / 2.51), abs=1e-9)

    def test_rough_limit_is_von_karman(self):
        for e in [1e-3, 1e-2, 0.05]:
            f = float(friction_factor(1e12, e))  # far into the rough regime
            assert f == pytest.approx(float(fully_rough_friction_factor(e)), rel=5e-4)

    @pytest.mark.parametrize(
        "Re,e,published",
        [
            (1e4, 0.0, 0.0309),   # smooth-pipe curve of the Moody chart
            (1e5, 0.0, 0.0180),
            (1e6, 0.0, 0.0116),
            (1e8, 0.0, 0.0059),
            (1e9, 0.05, 0.0716),  # fully rough plateaus
            (1e9, 0.01, 0.0380),
            (1e9, 0.001, 0.0196),
        ],
    )
    def test_moody_chart_values(self, Re, e, published):
        # Moody-chart values are quoted to ~3 significant figures.
        assert float(friction_factor(Re, e)) == pytest.approx(published, rel=1e-2)

    def test_laminar_is_exactly_64_over_Re(self):
        for Re in [1.0, 10.0, 500.0, 1000.0, 2099.0]:
            for e in [0.0, 1e-3, 0.05]:
                for m in ["colebrook", "haaland", "swamee_jain"]:
                    assert float(friction_factor(Re, e, m)) == 64.0 / Re

    def test_churchill_laminar_is_64_over_Re(self):
        for Re in [10.0, 500.0, 1000.0]:
            assert float(friction_factor(Re, 1e-3, "churchill")) == pytest.approx(64.0 / Re, rel=1e-6)

    def test_turbulent_methods_agree_with_colebrook(self):
        # Observed worst cases over Re 4e3..1e8, e/D 0..0.05: Churchill 3.1%
        # (all at Re < 1e4), Haaland 1.4%, Swamee-Jain 3.1%. Above Re = 1e4
        # Churchill is within 2.2% and from 1e5 within 1%.
        for Re in RES:
            for e in RELS:
                ref = _colebrook_reference(Re, e)
                ch = float(friction_factor(Re, e, "churchill"))
                tol = 0.032 if Re < 1e4 else (0.022 if Re < 1e5 else 0.0151)
                assert ch == pytest.approx(ref, rel=tol), (Re, e)
                assert float(friction_factor(Re, e, "haaland")) == pytest.approx(ref, rel=0.015)
                assert float(friction_factor(Re, e, "swamee_jain")) == pytest.approx(ref, rel=0.035)

    def test_churchill_within_2pct_of_colebrook_from_1e4(self):
        worst = 0.0
        for Re in RES[RES >= 1e4]:
            for e in RELS:
                worst = max(worst, abs(float(friction_factor(Re, e, "churchill")) / _colebrook_reference(Re, e) - 1))
        assert worst < 0.022

    def test_transition_is_monotone_and_bounded(self):
        Re = np.linspace(1500.0, 6000.0, 400)
        f = np.asarray(jax.vmap(lambda r: friction_factor(r, 1e-4))(jnp.asarray(Re)))
        assert np.all(np.isfinite(f))
        lam, turb = 64.0 / Re, np.array([_colebrook_reference(r, 1e-4) for r in Re])
        assert np.all(f >= np.minimum(lam, turb) - 1e-12) and np.all(f <= np.maximum(lam, turb) + 1e-12)
        assert np.all(np.abs(np.diff(f)) < 2e-3)  # no jump

    def test_jit_vmap_and_broadcast(self):
        Re = jnp.array([1e3, 1e5, 1e7])
        f = jax.jit(friction_factor)(Re, 1e-4)
        assert f.shape == (3,)
        assert jnp.all(jnp.isfinite(f))

    def test_laminar_blend_weight_is_c2(self):
        w = fluids._laminar_weight
        assert float(w(fluids.RE_LAMINAR)) == 0.0 and float(w(fluids.RE_TURBULENT)) == 1.0
        for Re in [fluids.RE_LAMINAR, fluids.RE_TURBULENT]:
            assert float(jax.grad(w)(Re)) == pytest.approx(0.0, abs=1e-12)


RE_POINTS = [1e3, 2300.0, 4000.0, 1e5]
E_REL = 1e-3


@pytest.mark.parametrize("method", ["colebrook", "churchill", "haaland", "swamee_jain"])
class TestFrictionGradients:
    @pytest.mark.parametrize("Re0", RE_POINTS)
    def test_grad_wrt_Re(self, method, Re0):
        check_grads(lambda s: friction_factor(Re0 * s, E_REL, method), (jnp.asarray(1.0),), order=1, modes=["rev", "fwd"], rtol=1e-4, atol=1e-8)

    @pytest.mark.parametrize("Re0", RE_POINTS)
    def test_grad_wrt_rel_roughness(self, method, Re0):
        check_grads(lambda s: friction_factor(Re0, E_REL * s, method), (jnp.asarray(1.0),), order=1, modes=["rev", "fwd"], rtol=1e-4, atol=1e-8)

    @pytest.mark.parametrize("Re0", RE_POINTS)
    def test_grad_wrt_D(self, method, Re0):
        """Re = 4 m / (pi D mu) and e/D both depend on D at fixed flow."""
        mu, m, eps, D0 = 1e-3, 1.0, 4.6e-5, 0.05
        # choose the mass flow so that Re = Re0 at D0
        m = Re0 * jnp.pi * D0 * mu / 4.0

        def f_of_D(s):  # s = D / D0, so the finite-difference step is relative
            D = D0 * s
            return friction_factor(4.0 * m / (jnp.pi * D * mu), eps / D, method)

        check_grads(f_of_D, (jnp.asarray(1.0),), order=1, modes=["rev", "fwd"], rtol=1e-4, atol=1e-8)

    @pytest.mark.parametrize("Re0", RE_POINTS)
    def test_gradients_finite(self, method, Re0):
        g = jax.grad(lambda r, e: friction_factor(r, e, method), argnums=(0, 1))(jnp.asarray(Re0), jnp.asarray(E_REL))
        assert all(np.isfinite(float(x)) for x in g)

    def test_gradients_finite_for_smooth_pipe(self, method):
        g = jax.grad(lambda r, e: friction_factor(r, e, method), argnums=(0, 1))(jnp.asarray(1e5), jnp.asarray(0.0))
        assert all(np.isfinite(float(x)) for x in g)


def test_colebrook_gradient_is_the_implicit_derivative():
    """d f/d Re from the IFT, computed by hand from the Colebrook equation."""
    Re, e = 1e5, 1e-3
    f = float(friction_factor(Re, e))
    x = f**-0.5
    a = e / 3.7 + 2.51 * x / Re
    gx = 1 + (2 / np.log(10)) * (2.51 / Re) / a
    gRe = (2 / np.log(10)) * (-2.51 * x / Re**2) / a
    dx_dRe = -gRe / gx
    expected = -2 * x**-3 * dx_dRe
    assert float(jax.grad(friction_factor)(Re, e)) == pytest.approx(expected, rel=1e-8)


class TestFittings:
    def test_every_fitting_has_a_source(self):
        assert FITTINGS
        required = {
            "elbow_90_standard", "elbow_90_long_radius", "tee_through", "tee_branch",
            "gate_valve", "globe_valve", "check_valve_swing", "entrance_sharp", "exit",
        }
        assert required <= set(FITTINGS)
        for name, fit in FITTINGS.items():
            assert fit.kind in ("K", "LD"), name
            assert "Crane" in fit.source, name

    def test_constant_K(self):
        assert float(fitting_K("exit")) == 1.0
        assert float(fitting_K("entrance_sharp")) == 0.5

    def test_crane_f_T_for_2_inch_pipe(self):
        """Crane TP-410 tabulates f_T = 0.019 for 2-in nominal pipe.

        2-in Sch 40 ID = 0.0525 m (2.067 in) with epsilon = 4.6e-5 m gives
        f_T = 0.0190 from the von Karman fully rough equation; that
        reproduces the published value and ties the roughness convention to
        Crane's.
        """
        D = 2.067 * 0.0254
        f_T = float(fully_rough_friction_factor(4.6e-5 / D))
        assert f_T == pytest.approx(0.019, abs=5e-4)
        assert float(fitting_K("elbow_90_standard", rel_roughness=4.6e-5 / D)) == pytest.approx(30 * 0.019, rel=0.03)
        assert float(fitting_K("globe_valve", f_T=0.019)) == pytest.approx(6.46)

    def test_LD_needs_roughness(self):
        with pytest.raises(ValueError):
            fitting_K("gate_valve")
        with pytest.raises(KeyError):
            fitting_K("nonexistent")

    def test_equivalent_length(self):
        D, f = 0.05, 0.02
        K = fitting_K("elbow_90_standard", f_T=f)
        assert float(equivalent_length(K, D, f)) == pytest.approx(30 * D)
        # same drop as a straight run
        rho, v = 1000.0, 2.0
        assert float(minor_loss(K, rho, v)) == pytest.approx(float(darcy_pressure_drop(f, equivalent_length(K, D, f), D, rho, v)))

    def test_K_gradient_wrt_D(self):
        check_grads(lambda s: fitting_K("tee_branch", rel_roughness=4.6e-5 / (0.05 * s)), (jnp.asarray(1.0),), order=1, modes=["rev"])


class TestFlowMeters:
    D, d, rho = 0.1, 0.05, 998.0

    def test_orifice_hand_calculation(self):
        beta = 0.5
        dP = 20e3
        Q = 0.61 * np.pi / 4 * 0.05**2 * np.sqrt(2 * dP / (998.0 * (1 - beta**4)))
        assert float(orifice_flow(dP, self.D, self.d, self.rho)) == pytest.approx(Q, rel=1e-12)

    @pytest.mark.parametrize("fl,dp,Cd", [(orifice_flow, orifice_dp, 0.61), (venturi_flow, venturi_dp, 0.98)])
    def test_inverse_round_trip(self, fl, dp, Cd):
        for dP in [1e2, 1e4, 3e5]:
            Q = fl(dP, self.D, self.d, self.rho)
            assert float(dp(Q, self.D, self.d, self.rho)) == pytest.approx(dP, rel=1e-10)
        Q = 3e-3
        assert float(fl(dp(Q, self.D, self.d, self.rho), self.D, self.d, self.rho)) == pytest.approx(Q, rel=1e-10)

    def test_defaults_and_cd(self):
        assert float(venturi_flow(1e4, self.D, self.d, self.rho)) > float(orifice_flow(1e4, self.D, self.d, self.rho))
        a = float(orifice_flow(1e4, self.D, self.d, self.rho, Cd=0.6))
        assert float(orifice_flow(1e4, self.D, self.d, self.rho, Cd=0.3)) == pytest.approx(a / 2)

    def test_gradients(self):
        check_grads(lambda s: orifice_flow(1e4 * s, self.D, self.d, self.rho), (jnp.asarray(1.0),), order=1, modes=["rev"])
        check_grads(lambda s: orifice_dp(3e-3, self.D, self.d * s, self.rho), (jnp.asarray(1.0),), order=1, modes=["rev"])

    def test_zero_dp_gives_zero_flow(self):
        assert float(orifice_flow(0.0, self.D, self.d, self.rho)) == pytest.approx(0.0, abs=1e-9)
