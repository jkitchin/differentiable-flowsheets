"""Tests for difflow.compressible: nozzles, isothermal and Fanno gas flow.

What "validated" means here (no textbook number is quoted from memory):

* The isothermal pipe equation and the Fanno relations are checked against an
  *independent* numerical integration of the differential momentum balance
  (scipy ``solve_ivp``), not against their own closed forms.
* The nozzle flow is checked against its limits: the choked flux, continuity
  at the critical ratio, and the incompressible Bernoulli flow at small
  pressure drop.
* Gradients by ``check_grads`` and against finite differences, including the
  implicit-function gradients of the two root-finding solvers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.integrate import solve_ivp

from difflow.compressible import (
    R_GAS,
    adiabatic_outlet_mach,
    choked_mass_flux,
    critical_pressure_ratio,
    fanno_parameter,
    fanno_ratios,
    isothermal_outlet_pressure,
    isothermal_pipe_mass_flux,
    mach_number,
    nozzle_mass_flow,
    speed_of_sound,
    stagnation_ratios,
)

AIR = 0.02897
CH4 = 0.01604


class TestSoundAndStagnation:
    def test_speed_of_sound_air(self):
        # sqrt(1.4 * 8.314462618 * 293.15 / 0.02897) = 343.2 m/s
        assert float(speed_of_sound(293.15, AIR)) == pytest.approx(343.2, abs=0.1)

    def test_mach(self):
        c = float(speed_of_sound(300.0, AIR))
        assert float(mach_number(0.5 * c, 300.0, AIR)) == pytest.approx(0.5)

    def test_stagnation_at_sonic(self):
        r = stagnation_ratios(1.0, 1.4)
        assert float(r.T0_T) == pytest.approx(1.2)
        assert float(r.P0_P) == pytest.approx(1.2**3.5)
        assert float(r.rho0_rho) == pytest.approx(1.2**2.5)

    @pytest.mark.parametrize("g, expected", [(1.4, 0.528282), (1.3, 0.545727), (1.667, 0.487)])
    def test_critical_ratio(self, g, expected):
        assert float(critical_pressure_ratio(g)) == pytest.approx(expected, abs=1e-3)

    def test_critical_ratio_is_sonic(self):
        # P0/P at M = 1 is the inverse of the critical ratio
        assert float(critical_pressure_ratio(1.33) * stagnation_ratios(1.0, 1.33).P0_P) == pytest.approx(1.0)


class TestNozzle:
    P0, T0, A = 8e5, 320.0, 2e-4

    def test_choked_equals_sonic_flux(self):
        m = nozzle_mass_flow(self.P0, self.T0, 1e5, self.A, AIR, 1.4, Cd=0.9)
        assert float(m) == pytest.approx(0.9 * self.A * float(choked_mass_flux(self.P0, self.T0, AIR, 1.4)))

    def test_sonic_flux_from_throat_state(self):
        # G* = rho* c* with the throat at T* = T0/1.2 and P* = 0.5283 P0
        Tt = self.T0 / 1.2
        Pt = self.P0 * float(critical_pressure_ratio(1.4))
        rho = Pt * AIR / (R_GAS * Tt)
        assert float(choked_mass_flux(self.P0, self.T0, AIR)) == pytest.approx(rho * float(speed_of_sound(Tt, AIR)))

    def test_continuous_at_critical_ratio(self):
        rc = float(critical_pressure_ratio(1.4))
        m = [float(nozzle_mass_flow(self.P0, self.T0, self.P0 * r, self.A, AIR)) for r in (rc - 1e-6, rc, rc + 1e-6)]
        assert m[0] == pytest.approx(m[1], rel=1e-9)
        assert m[2] == pytest.approx(m[1], rel=1e-9)

    def test_back_pressure_gradient(self):
        g = jax.grad(lambda Pb: nozzle_mass_flow(self.P0, self.T0, Pb, self.A, AIR))
        assert float(g(1e5)) == 0.0                     # choked: no influence
        assert float(g(6e5)) < 0.0                      # subsonic: lower Pb, more flow
        assert abs(float(g(self.P0 * 0.5283 + 1.0))) < 1e-8 * self.A * 1e3  # smooth at the switch

    def test_incompressible_limit(self):
        dP = 100.0
        m = float(nozzle_mass_flow(self.P0, self.T0, self.P0 - dP, self.A, AIR))
        rho = self.P0 * AIR / (R_GAS * self.T0)
        assert m == pytest.approx(self.A * np.sqrt(2 * rho * dP), rel=1e-4)

    def test_no_flow_at_equal_pressure(self):
        assert float(nozzle_mass_flow(self.P0, self.T0, self.P0, self.A, AIR)) == 0.0

    def test_grads(self):
        check_grads(lambda P0, T0, Pb: nozzle_mass_flow(P0, T0, Pb, self.A, AIR), (8e5, 320.0, 6e5), order=1, modes=["fwd", "rev"])


def _isothermal_ode_P2(G, P1, T, L, D, f, MW):
    """Integrate dP/dx (1 - G^2 RT/(MW P^2)) = -f G^2 RT/(2 D MW P)."""
    a = R_GAS * T / MW

    def rhs(x, P):
        return [-f * G**2 * a / (2 * D * P[0]) / (1 - G**2 * a / P[0] ** 2)]

    sol = solve_ivp(rhs, (0.0, L), [P1], rtol=1e-11, atol=1e-6)
    return sol.y[0, -1]


class TestIsothermalPipe:
    # natural-gas line: 50 km of 0.5 m pipe, 60 -> 40 bar, 288 K
    args = dict(T=288.0, L=5e4, D=0.5, f=0.012, MW=0.0175)

    def test_against_ode_integration(self):
        G = float(isothermal_pipe_mass_flux(60e5, 40e5, **self.args))
        assert _isothermal_ode_P2(G, 60e5, *self.args.values()) == pytest.approx(40e5, rel=1e-7)

    def test_short_high_velocity_line_against_ode(self):
        # acceleration term matters: short, small line near the limit
        a = dict(T=300.0, L=20.0, D=0.02, f=0.02, MW=AIR)
        G = float(isothermal_pipe_mass_flux(5e5, 2.5e5, **a))
        r = isothermal_outlet_pressure(G, 5e5, **a)
        assert not bool(r.choked)
        assert float(r.P2) == pytest.approx(_isothermal_ode_P2(G, 5e5, *a.values()), rel=1e-6)
        assert 0.3 < float(r.mach2) < 1 / np.sqrt(1.4)

    def test_roundtrip(self):
        for P2 in (59e5, 50e5, 30e5, 10e5):
            G = isothermal_pipe_mass_flux(60e5, P2, **self.args)
            r = isothermal_outlet_pressure(G, 60e5, **self.args)
            assert float(r.P2) == pytest.approx(P2, rel=1e-9)

    def test_long_line_limit(self):
        # without the log term: P1^2 - P2^2 = G^2 RT fL/(MW D)
        G = float(isothermal_pipe_mass_flux(60e5, 40e5, **self.args))
        a = self.args
        G_long = np.sqrt((60e5**2 - 40e5**2) * a["MW"] * a["D"] / (R_GAS * a["T"] * a["f"] * a["L"]))
        assert G == pytest.approx(G_long, rel=1e-3)
        assert G < G_long

    def test_choked(self):
        a = dict(T=300.0, L=20.0, D=0.02, f=0.02, MW=AIR)
        r = isothermal_outlet_pressure(2000.0, 5e5, **a)
        assert bool(r.choked)
        assert float(r.P2) == pytest.approx(float(r.P_choke))
        assert float(r.mach2) == pytest.approx(1 / np.sqrt(1.4))

    def test_choke_pressure_is_isothermal_limit(self):
        r = isothermal_outlet_pressure(100.0, 60e5, **self.args)
        v = 100.0 * R_GAS * 288.0 / (0.0175 * float(r.P_choke))
        assert v == pytest.approx(np.sqrt(R_GAS * 288.0 / 0.0175))

    def test_grads(self):
        G = float(isothermal_pipe_mass_flux(60e5, 40e5, **self.args))

        def P2(G, L, D, f):
            return isothermal_outlet_pressure(G, 60e5, 288.0, L, D, f, 0.0175).P2

        # check in relative coordinates: the arguments span 0.01 to 5e4
        x0 = jnp.array([G, 5e4, 0.5, 0.012])
        check_grads(lambda s: P2(*(x0 * s)) / 1e6, (jnp.ones(4),), order=1, modes=["fwd", "rev"], eps=1e-5)
        check_grads(lambda s: isothermal_pipe_mass_flux(60e5 * s[0], 40e5 * s[1], **self.args), (jnp.ones(2),), order=1)

    def test_implicit_gradient_matches_fd(self):
        G = float(isothermal_pipe_mass_flux(60e5, 40e5, **self.args))
        dP2_dD = float(jax.grad(lambda D: isothermal_outlet_pressure(G, 60e5, 288.0, 5e4, D, 0.012, 0.0175).P2)(0.5))
        h = 1e-6
        fd = (
            _isothermal_ode_P2(G, 60e5, 288.0, 5e4, 0.5 + h, 0.012, 0.0175)
            - _isothermal_ode_P2(G, 60e5, 288.0, 5e4, 0.5 - h, 0.012, 0.0175)
        ) / (2 * h)
        assert dP2_dD == pytest.approx(fd, rel=1e-4)


def _fanno_ode_M2(M1, fL_D, g):
    """Integrate d(M^2)/d(fx/D) = gamma M^4 (1 + (gamma-1)/2 M^2) / (1 - M^2)."""

    def rhs(s, y):
        m2 = y[0]
        return [g * m2**2 * (1 + 0.5 * (g - 1) * m2) / (1 - m2)]

    sol = solve_ivp(rhs, (0.0, fL_D), [M1**2], rtol=1e-11, atol=1e-13)
    return np.sqrt(sol.y[0, -1])


class TestFanno:
    def test_zero_at_sonic(self):
        assert float(fanno_parameter(1.0, 1.4)) == pytest.approx(0.0, abs=1e-14)
        r = fanno_ratios(1.0, 1.4)
        assert float(r.P_Pstar) == pytest.approx(1.0)
        assert float(r.T_Tstar) == pytest.approx(1.0)
        assert float(r.P0_P0star) == pytest.approx(1.0)

    @pytest.mark.parametrize("M1", [0.1, 0.3, 0.6])
    def test_parameter_against_ode(self, M1):
        # integrating fL*/D from M1 must just reach M = 1
        L_star = float(fanno_parameter(M1, 1.4))
        M_near = _fanno_ode_M2(M1, 0.999 * L_star, 1.4)
        assert float(fanno_parameter(M_near, 1.4)) == pytest.approx(0.001 * L_star, rel=1e-5)
        out = adiabatic_outlet_mach(M1, 0.5 * L_star, 1.4)
        assert float(out.M2) == pytest.approx(_fanno_ode_M2(M1, 0.5 * L_star, 1.4), rel=1e-7)

    def test_pressure_ratio_against_continuity(self):
        # adiabatic, constant area: P2/P1 = (M1/M2) sqrt(T2/T1)
        out = adiabatic_outlet_mach(0.3, 2.0, 1.3)
        assert float(out.P2_P1) == pytest.approx(0.3 / float(out.M2) * np.sqrt(float(out.T2_T1)))
        assert float(out.T2_T1) < 1.0 and float(out.P2_P1) < 1.0

    def test_choked(self):
        out = adiabatic_outlet_mach(0.5, 2.0, 1.4)   # fL*/D(0.5) = 1.069 < 2
        assert bool(out.choked)
        assert float(out.M2) == 1.0

    def test_grads(self):
        check_grads(lambda M1, fl: adiabatic_outlet_mach(M1, fl, 1.4).M2, (0.3, 2.0), order=1, modes=["fwd", "rev"], eps=1e-5)

    def test_choked_gradient_is_finite(self):
        g = jax.grad(lambda fl: adiabatic_outlet_mach(0.5, fl, 1.4).P2_P1)(2.0)
        assert np.isfinite(float(g))


class TestJit:
    def test_jit_vmap(self):
        f = jax.jit(jax.vmap(lambda L: isothermal_outlet_pressure(50.0, 60e5, 288.0, L, 0.5, 0.012, 0.0175).P2))
        P2 = f(jnp.array([1e4, 5e4, 1e5]))
        assert np.all(np.diff(np.asarray(P2)) < 0)


# =============================================================================
# Additions for the compressor chapter (G05-8): nozzle flag, Z, line flow with
# friction, pressure profile, polytropic efficiency, staging
# =============================================================================

from difflow.compressible import (  # noqa: E402
    isentropic_efficiency_from_polytropic,
    isothermal_distance,
    isothermal_limit_mach,
    isothermal_pipe_flow,
    isothermal_pressure_profile,
    nozzle_flow,
    optimal_stage_ratios,
    polytropic_efficiency_from_isentropic,
    polytropic_exponent,
)
from difflow import fluids  # noqa: E402


class TestNozzleFlow:
    A10 = np.pi * 0.01**2 / 4

    @pytest.mark.parametrize("Pb, choked", [(1e5, True), (5e5, True), (8e5, False), (9.5e5, False)])
    def test_matches_nozzle_mass_flow_and_flags_choking(self, Pb, choked):
        r = nozzle_flow(10e5, 293.15, Pb, self.A10, AIR, 1.4, 0.85)
        m = nozzle_mass_flow(10e5, 293.15, Pb, self.A10, AIR, 1.4, 0.85)
        assert float(r.m_dot) == pytest.approx(float(m), rel=1e-14)
        assert bool(r.choked) is choked
        if choked:
            assert float(r.mach_throat) == pytest.approx(1.0)
            assert float(r.P_throat) == pytest.approx(10e5 * float(critical_pressure_ratio(1.4)))
        else:
            assert float(r.P_throat) == pytest.approx(Pb)
            # Throat Mach reproduces the stagnation pressure ratio.
            assert float(stagnation_ratios(r.mach_throat, 1.4).P0_P) == pytest.approx(10e5 / Pb)

    def test_book_values(self):
        """Air, 10 bar, 293.15 K, 10 mm, Cd 0.85 (chapter 5 oracle)."""
        flows = [float(nozzle_flow(10e5, 293.15, Pb, self.A10, AIR, 1.4, 0.85).m_dot)
                 for Pb in (1e5, 8e5, 9.5e5)]
        assert flows == pytest.approx([0.15757, 0.12902, 0.07079], rel=3e-4)

    def test_back_pressure_gradient(self):
        def m(Pb):
            return nozzle_flow(10e5, 293.15, Pb, self.A10, AIR, 1.4, 0.85).m_dot

        assert float(jax.grad(m)(5e5)) == 0.0
        assert float(jax.grad(m)(8e5)) == pytest.approx(-2.35e-7, rel=2e-3)


class TestCompressibilityFactor:
    def test_Z_scales_RT_over_M(self):
        assert float(speed_of_sound(300.0, CH4, 1.3, Z=0.85)) == pytest.approx(
            float(speed_of_sound(300.0, CH4 / 0.85, 1.3)), rel=1e-14)
        G1 = isothermal_pipe_mass_flux(60e5, 40e5, 288.15, 5e4, 0.6, 0.0115, 0.018, Z=0.851)
        G2 = isothermal_pipe_mass_flux(60e5, 40e5, 288.15, 5e4, 0.6, 0.0115, 0.018 / 0.851)
        assert float(G1) == pytest.approx(float(G2), rel=1e-14)

    def test_isothermal_limit_mach(self):
        assert float(isothermal_limit_mach(1.4)) == pytest.approx(0.845154, abs=1e-6)


class TestIsothermalPipeFlow:
    """wex05-1: 60 -> 40 bar, 50 km, 0.6 m, 45 um, 288.15 K, Z 0.851, mu 1.2e-5."""

    ARGS = (60e5, 40e5, 288.15, 50e3, 0.6, 45e-6, 1.2e-5, 0.0180042)

    def test_book_values_and_consistency(self):
        r = isothermal_pipe_flow(*self.ARGS, Z=0.851)
        assert float(r.G) == pytest.approx(429.84, abs=0.02)
        assert float(r.m_dot) == pytest.approx(121.54, abs=0.02)
        assert float(r.f) == pytest.approx(0.011461, abs=1e-6)
        # f is the Colebrook factor at the Reynolds number of the flow it gives.
        assert float(r.f) == pytest.approx(
            float(fluids.friction_factor(r.Re, 45e-6 / 0.6)), rel=1e-12)
        assert float(r.G) == pytest.approx(float(isothermal_pipe_mass_flux(
            60e5, 40e5, 288.15, 50e3, 0.6, r.f, 0.0180042, Z=0.851)), rel=1e-14)

    def test_gradients(self):
        def m(P1_bar, D):
            return isothermal_pipe_flow(P1_bar * 1e5, 40e5, 288.15, 50e3, D, 45e-6,
                                        1.2e-5, 0.0180042, Z=0.851).m_dot

        check_grads(m, (60.0, 0.6), order=1, modes=["rev"], eps=1e-5)

    def test_profile_and_distance_are_inverse(self):
        r = isothermal_pipe_flow(*self.ARGS, Z=0.851)
        x = jnp.linspace(0.0, 50e3, 6)
        prof = isothermal_pressure_profile(x, r.G, 60e5, 288.15, 0.6, r.f, 0.0180042, 1.3, 0.851)
        assert prof.P2.shape == (6,)
        assert float(prof.P2[-1]) == pytest.approx(40e5, rel=1e-9)
        assert float(prof.P2[0]) == pytest.approx(60e5, rel=1e-12)
        back = isothermal_distance(prof.P2[1:], r.G, 60e5, 288.15, 0.6, r.f, 0.0180042, 0.851)
        np.testing.assert_allclose(np.asarray(back), np.asarray(x[1:]), rtol=1e-8)
        assert not bool(jnp.any(prof.choked))
        assert float(prof.mach2[-1]) == pytest.approx(0.032, abs=5e-4)


class TestPolytropic:
    def test_exponent(self):
        n = polytropic_exponent(0.8, 1.4)
        assert float((n - 1) / n) == pytest.approx(0.4 / (1.4 * 0.8), rel=1e-14)
        assert float(polytropic_exponent(1.0, 1.3)) == pytest.approx(1.3)
        ne = polytropic_exponent(0.8, 1.4, compression=False)
        assert float((ne - 1) / ne) == pytest.approx(0.8 * 0.4 / 1.4, rel=1e-14)

    @pytest.mark.parametrize("r, eta_s", [(2.0, 0.780), (4.0, 0.759), (10.0, 0.729)])
    def test_book_values(self, r, eta_s):
        """gamma 1.4, eta_p 0.80 (chapter oracle)."""
        assert float(isentropic_efficiency_from_polytropic(r, 0.8, 1.4)) == pytest.approx(eta_s, abs=6e-4)

    def test_temperature_rise_is_consistent(self):
        """eta_s from eta_p reproduces the polytropic outlet temperature."""
        r, g, ep = 5.0, 1.35, 0.82
        n = polytropic_exponent(ep, g)
        T2_poly = r ** ((n - 1) / n)
        T2_s = 1.0 + (r ** ((g - 1) / g) - 1.0) / isentropic_efficiency_from_polytropic(r, ep, g)
        assert float(T2_s) == pytest.approx(float(T2_poly), rel=1e-12)

    @pytest.mark.parametrize("compression", [True, False])
    def test_inverse_and_ordering(self, compression):
        es = isentropic_efficiency_from_polytropic(6.0, 0.85, 1.4, compression)
        assert float(polytropic_efficiency_from_isentropic(6.0, es, 1.4, compression)) == pytest.approx(0.85, rel=1e-12)
        assert (float(es) < 0.85) is compression  # compressor: eta_s < eta_p; turbine: >

    @pytest.mark.parametrize("compression", [True, False])
    def test_small_ratio_limit(self, compression):
        """As r -> 1 the stage is infinitesimal and eta_s -> eta_p."""
        es = isentropic_efficiency_from_polytropic(1.0 + 1e-6, 0.8, 1.4, compression)
        assert float(es) == pytest.approx(0.8, abs=1e-6)

    def test_gradients(self):
        check_grads(lambda r, e: isentropic_efficiency_from_polytropic(r, e, 1.4),
                    (4.0, 0.8), order=1, modes=["rev"])
        check_grads(lambda r, e: polytropic_efficiency_from_isentropic(r, e, 1.4, False),
                    (4.0, 0.8), order=1, modes=["rev"])


class TestStaging:
    def test_equal_ratios(self):
        r = optimal_stage_ratios(1e5, 110e5, 4)
        assert r.shape == (4,)
        assert float(r[0]) == pytest.approx(3.2385, abs=1e-4)
        assert float(1e5 * jnp.prod(r)) == pytest.approx(110e5, rel=1e-12)

    def test_equal_ratios_minimize_ideal_gas_work(self):
        """Intercooled ideal gas: total work sum(r_i^k - 1) is least at equal r_i."""
        k = 0.4 / 1.4

        def work(log_r1):  # two stages, r1 * r2 = 30
            r1 = jnp.exp(log_r1)
            return r1**k + (30.0 / r1) ** k

        r_opt = optimal_stage_ratios(1.0, 30.0, 2)[0]
        assert abs(float(jax.grad(work)(jnp.log(r_opt)))) < 1e-12

    def test_rejects_zero_stages(self):
        with pytest.raises(ValueError):
            optimal_stage_ratios(1e5, 1e6, 0)
