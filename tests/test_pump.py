"""Tests for the CentrifugalPump unit, curves, system curve and operating point."""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

import difflow
from difflow import (
    CentrifugalPump,
    CentrifugalPumpParams,
    IdealThermo,
    Pipe,
    PipeParams,
    SpeciesData,
    fit_pump_curve,
    make_stream,
    npsh_available,
    operating_point,
    pumps_in_parallel,
    pumps_in_series,
    system_curve,
)
from difflow.catalog import describe_operation
from difflow.streams import get_flows
from difflow.units.pump import PumpCurve

RHO, MU, MW_W = 998.2, 1.002e-3, 18.015  # water, 20 C (CRC / Perry's)
G = 9.80665

# Illustrative "vendor" data for a mid-size end-suction pump (NOT a published
# datasheet): flow (L/s) and head (m) at the reference speed 1750 rpm.
Q_DATA = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0]) * 1e-3
H_DATA = np.array([40.0, 39.6, 37.9, 35.6, 31.8, 27.7])
H_COEFFS = np.asarray(fit_pump_curve(Q_DATA, H_DATA, 2))
NPSHR_COEFFS = [1.5, 0.0, 2.5e3]  # 1.5 m + 2.5e3 Q^2: 2.5 m at 20 L/s
PUMP = dict(head_coeffs=H_COEFFS, rho=RHO, eta_max=0.72, Q_bep=20e-3,
            npshr_coeffs=NPSHR_COEFFS, N_ref=1750.0, D_ref=0.25, MW=MW_W)

D4 = 4.026 * 0.0254  # 4-in Sch 40 inside diameter (m)
LIFT, L_PIPE, K_FIT, EPS = 12.0, 400.0, 5.3, 4.6e-5


def pipe(D=D4, L=L_PIPE, **kw):
    base = dict(L=L, D=D, rho=RHO, mu=MU, MW=MW_W, fittings=K_FIT, roughness=EPS)
    base.update(kw)
    return Pipe(PipeParams(**base))


def water(Q, P=1.0e5, T=293.15):
    return make_stream({"W": Q * RHO / (MW_W * 1e-3)}, T, P)


def colebrook_np(Re, e):
    if Re < 2100:
        return 64.0 / Re
    x = brentq(lambda x: x + 2 * np.log10(e / 3.7 + 2.51 * x / Re), 1.0, 200.0, xtol=1e-15)
    return x**-2


class TestTransferPumpOperatingPoint:
    """Tank-to-tank transfer: static lift, 400 m of 4-in Sch 40 steel, fittings K = 5.3.

    No published worked example is quoted (the vendor curve is illustrative;
    textbook examples such as McCabe Ex. 8.x use graph-read curves we cannot
    reproduce exactly). Verification is an independent calculation that uses
    numpy.polyfit for the curve and scipy brentq for both the Colebrook-White
    friction factor and the pump/system intersection, sharing no difflow code.
    """

    def independent(self):
        c = np.polyfit(Q_DATA, H_DATA, 2)[::-1]
        A = np.pi / 4 * D4**2

        def gap(Q):
            v = Q / A
            Re = RHO * v * D4 / MU
            f = colebrook_np(Re, EPS / D4)
            sys_ = LIFT + (f * L_PIPE / D4 + K_FIT) * v**2 / (2 * G)
            return c[0] + c[1] * Q + c[2] * Q**2 - sys_

        return brentq(gap, 1e-4, 0.03, xtol=1e-14)

    def test_matches_independent_calculation(self):
        Q_ref = self.independent()
        sysc = system_curve(LIFT, pipe())
        Q = float(operating_point(CentrifugalPump(CentrifugalPumpParams(**PUMP)), sysc))
        assert Q == pytest.approx(Q_ref, rel=1e-6)  # issue asks for 2%; agreement is far tighter
        assert 0.005 < Q < 0.025  # inside the fitted vendor range

    def test_info_at_operating_point(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**PUMP))
        Q, info = operating_point(pump, system_curve(LIFT, pipe()), return_info=True)
        assert float(info["head"]) == pytest.approx(float(system_curve(LIFT, pipe()).head(Q)), rel=1e-9)
        # the unit at that flow delivers the same head and power rho g Q H / eta
        out, i2 = pump(water(float(Q)))
        assert float(i2["head"]) == pytest.approx(float(info["head"]), rel=1e-9)
        assert float(info["shaft_power"]) == pytest.approx(float(i2["shaft_power"]), rel=1e-9)
        assert float(out["P"]) == pytest.approx(1e5 + RHO * G * float(i2["head"]), rel=1e-12)

    def test_fit_pump_curve_matches_polyfit_and_other_orders(self):
        np.testing.assert_allclose(H_COEFFS, np.polyfit(Q_DATA, H_DATA, 2)[::-1], rtol=1e-8)
        c3 = fit_pump_curve(Q_DATA, H_DATA, 3)
        assert c3.shape == (4,)
        np.testing.assert_allclose(c3, np.polyfit(Q_DATA, H_DATA, 3)[::-1], rtol=1e-6, atol=1e-6)

    def test_cubic_head_curve_operating_point(self):
        c3 = fit_pump_curve(Q_DATA, H_DATA, 3)
        curve = PumpCurve(c3)
        sysc = system_curve(LIFT, pipe())
        Q = operating_point(curve, sysc)
        assert float(curve.head(Q) - sysc.head(Q)) == pytest.approx(0.0, abs=1e-8)


class TestAffinityLaws:
    def test_pure_friction_system_scales_linearly_with_speed(self):
        # H_sys = k Q^2 exactly: Q* doubles when N doubles
        class Quad:
            def head(self, Q):
                return 3.0e4 * jnp.asarray(Q) ** 2

        c = lambda N: PumpCurve(H_COEFFS, speed_ratio=N / 1750.0)
        Q1 = operating_point(c(1750.0), Quad())
        Q2 = operating_point(c(3500.0), Quad())
        assert float(Q2 / Q1) == pytest.approx(2.0, rel=1e-9)

    def test_fully_rough_pipe_no_static_head_doubles_flow(self):
        rough = pipe(roughness=0.05 * D4)  # relative roughness 0.05: f nearly Re-independent
        sysc = system_curve(0.0, rough)
        Q1 = operating_point(PumpCurve(H_COEFFS), sysc)
        Q2 = operating_point(PumpCurve(H_COEFFS, speed_ratio=1.2), sysc)
        assert float(Q2 / Q1) == pytest.approx(1.2, rel=5e-3)

    def test_smooth_pipe_no_static_head_nearly_linear(self):
        sysc = system_curve(0.0, pipe())
        Q1 = operating_point(PumpCurve(H_COEFFS), sysc)
        Q2 = operating_point(PumpCurve(H_COEFFS, speed_ratio=1.2), sysc)
        # Re-dependence of f makes it not exact, but within a few percent of linear
        assert float(Q2 / Q1) == pytest.approx(1.2, rel=0.03)

    def test_static_head_breaks_proportionality(self):
        sysc = system_curve(LIFT, pipe())
        Q1 = operating_point(PumpCurve(H_COEFFS), sysc)
        Q2 = operating_point(PumpCurve(H_COEFFS, speed_ratio=1.2), sysc)
        ratio = float(Q2 / Q1)
        assert ratio > 1.0  # more speed, more flow
        # but NOT proportional: with a static head the flow grows faster than N here
        # (the smooth-pipe case above is within 3 % of linear)
        assert abs(ratio - 1.2) / 1.2 > 0.05

    def test_curve_scaling_laws_Q_H_P(self):
        base, fast = PumpCurve(H_COEFFS, eta_coeffs=[0, 90.0, -2e3]), PumpCurve(
            H_COEFFS, eta_coeffs=[0, 90.0, -2e3], speed_ratio=1.5, diameter_ratio=0.9)
        s = 1.5 * 0.9
        Q = 0.012
        assert float(fast.head(s * Q)) == pytest.approx(s**2 * float(base.head(Q)), rel=1e-12)
        assert float(fast.efficiency(s * Q)) == pytest.approx(float(base.efficiency(Q)), rel=1e-12)
        assert float(fast.shaft_power(s * Q, RHO)) == pytest.approx(s**3 * float(base.shaft_power(Q, RHO)), rel=1e-12)

    def test_unit_N_and_D_scale_the_curve(self):
        a = CentrifugalPump(CentrifugalPumpParams(**PUMP))
        b = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "N": 1750.0 * 1.1, "D": 0.25 * 0.95}))
        s = 1.1 * 0.95
        _, ia = a(water(0.01))
        _, ib = b(water(0.01 * s))
        assert float(ib["head"]) == pytest.approx(s**2 * float(ia["head"]), rel=1e-12)
        assert float(ib["shaft_power"]) == pytest.approx(s**3 * float(ia["shaft_power"]), rel=1e-12)


class TestSeriesParallel:
    def test_series_heads_add_at_equal_flow(self):
        a = PumpCurve(H_COEFFS)
        b = PumpCurve(H_COEFFS, speed_ratio=0.8)
        ser = pumps_in_series(a, b)
        for Q in (0.0, 0.005, 0.012, 0.02):
            assert float(ser.head(Q)) == pytest.approx(float(a.head(Q) + b.head(Q)), rel=1e-12)

    def test_parallel_flows_add_at_equal_head(self):
        a = PumpCurve(H_COEFFS)
        b = PumpCurve(H_COEFFS, speed_ratio=0.9)
        par = pumps_in_parallel(a, b)
        Q = 0.030
        q = par.split(Q)
        assert float(jnp.sum(q)) == pytest.approx(Q, rel=1e-10)
        assert float(a.head(q[0])) == pytest.approx(float(b.head(q[1])), rel=1e-9)
        assert float(par.head(Q)) == pytest.approx(float(a.head(q[0])), rel=1e-12)

    def test_identical_parallel_pumps_split_equally(self):
        a = PumpCurve(H_COEFFS)
        par = pumps_in_parallel(a, a, a)
        np.testing.assert_allclose(np.asarray(par.split(0.045)), 0.015, rtol=1e-10)
        assert float(par.head(0.045)) == pytest.approx(float(a.head(0.015)), rel=1e-10)

    def test_operating_points_of_combinations(self):
        a = PumpCurve(H_COEFFS)
        sysc = system_curve(LIFT, pipe())
        Q1 = float(operating_point(a, sysc))
        Qs = float(operating_point(pumps_in_series(a, a), sysc))
        Qp = float(operating_point(pumps_in_parallel(a, a), sysc))
        assert Qs > Q1 and Qp > Q1
        # neither doubles the flow into a system with static head and friction
        assert Qs < 2 * Q1 and Qp < 2 * Q1
        # consistency: each solves its own head balance
        assert float(pumps_in_series(a, a).head(Qs) - sysc.head(Qs)) == pytest.approx(0.0, abs=1e-7)
        assert float(pumps_in_parallel(a, a).head(Qp) - sysc.head(Qp)) == pytest.approx(0.0, abs=1e-7)

    def test_series_efficiency_and_power(self):
        e = [0, 90.0, -2e3]
        a, b = PumpCurve(H_COEFFS, e), PumpCurve(H_COEFFS, e)
        ser = pumps_in_series(a, b)
        Q = 0.012
        assert float(ser.shaft_power(Q, RHO)) == pytest.approx(2 * float(a.shaft_power(Q, RHO)), rel=1e-12)
        assert float(ser.efficiency(Q, RHO)) == pytest.approx(float(a.efficiency(Q)), rel=1e-12)

    def test_parallel_power_and_npshr(self):
        e = [0, 90.0, -2e3]
        a = PumpCurve(H_COEFFS, e, NPSHR_COEFFS)
        par = pumps_in_parallel(a, a)
        Q = 0.024
        assert float(par.shaft_power(Q, RHO)) == pytest.approx(2 * float(a.shaft_power(Q / 2, RHO)), rel=1e-9)
        assert float(par.npshr(Q)) == pytest.approx(float(a.npshr(Q / 2)), rel=1e-9)
        assert float(pumps_in_series(a, a).npshr(Q)) == pytest.approx(float(a.npshr(Q)), rel=1e-12)

    def test_accepts_pump_units(self):
        u = CentrifugalPump(CentrifugalPumpParams(**PUMP))
        assert float(pumps_in_series(u, u).head(0.01)) == pytest.approx(2 * float(u.curve.head(0.01)), rel=1e-12)


class TestNPSH:
    # water at 95 C: Psat = 84.55 kPa, rho = 961.9 kg/m^3 (steam tables)
    def test_npsh_available_formula(self):
        v = 1.5
        expect = (101325.0 - 84550.0) / (961.9 * G) + v**2 / (2 * G) + 2.0 - 0.7
        assert float(npsh_available(101325.0, 84550.0, 961.9, v, 2.0, 0.7)) == pytest.approx(expect, rel=1e-12)

    def test_hot_liquid_near_bubble_point_negative_margin(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "rho": 961.9, "Psat": 84550.0}))
        inlet = make_stream({"W": 0.02 * 961.9 / (MW_W * 1e-3)}, 368.15, 101325.0)
        _, info = pump(inlet)
        assert float(info["NPSHa"]) < float(info["NPSHr"])
        assert float(info["npsh_margin"]) < 0.0
        assert float(info["NPSHa"]) == pytest.approx(1.78, abs=0.01)

    def test_cold_water_positive_margin(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0}))
        _, info = pump(water(0.02, P=101325.0))
        assert float(info["npsh_margin"]) > 5.0

    def test_margin_drops_with_flow_and_speed(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0}))
        m = [float(pump(water(q, P=101325.0))[1]["npsh_margin"]) for q in (0.005, 0.015, 0.025)]
        assert m[0] > m[1] > m[2]
        fast = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0, "N": 2100.0}))
        assert float(fast(water(0.015, P=101325.0))[1]["NPSHr"]) > float(pump(water(0.015, P=101325.0))[1]["NPSHr"])

    def test_psat_from_thermo_antoine(self):
        sp = {"W": SpeciesData("W", MW_W, (75.0, 0, 0, 0), (4e4, 0.38, 647.0), (10.1, 1700.0, -40.0))}
        th = IdealThermo(sp)
        params = CentrifugalPumpParams(**{**PUMP, "MW": None})
        inlet = water(0.02, P=101325.0)
        _, info = CentrifugalPump(params, thermo=th)(inlet)
        _, ref = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": float(th.Psat("W", 293.15))}))(inlet)
        assert float(info["NPSHa"]) == pytest.approx(float(ref["NPSHa"]), rel=1e-12)

    def test_suction_velocity_head(self):
        a = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0}))
        b = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0, "D_suction": 0.1}))
        v = 0.02 / (np.pi / 4 * 0.1**2)
        d = float(b(water(0.02))[1]["NPSHa"] - a(water(0.02))[1]["NPSHa"])
        assert d == pytest.approx(v**2 / (2 * G), rel=1e-9)


class TestUnit:
    def test_outlet_pressure_and_conservation(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "motor_efficiency": 0.9}))
        inlet = water(0.015)
        out, info = pump(inlet)
        H = float(np.polyval(H_COEFFS[::-1], 0.015))
        assert float(info["head"]) == pytest.approx(H, rel=1e-12)
        assert float(out["P"]) == pytest.approx(1e5 + RHO * G * H, rel=1e-12)
        assert float(out["T"]) == float(inlet["T"])
        assert float(get_flows(out)["W"]) == float(get_flows(inlet)["W"])
        eta = 0.72 * (1 - ((0.015 - 0.02) / 0.02) ** 2)
        assert float(info["efficiency"]) == pytest.approx(eta, rel=1e-12)
        assert float(info["shaft_power"]) == pytest.approx(RHO * G * 0.015 * H / eta, rel=1e-12)
        assert float(info["electric_power"]) == pytest.approx(float(info["shaft_power"]) / 0.9, rel=1e-12)

    def test_temperature_rise_with_cp(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "cp": 4182.0}))
        out, info = pump(water(0.015))
        eta, H = float(info["efficiency"]), float(info["head"])
        assert float(out["T"]) - 293.15 == pytest.approx(G * H * (1 - eta) / (eta * 4182.0), rel=1e-9)

    def test_missing_molar_mass_and_efficiency(self):
        with pytest.raises(ValueError, match="molar mass"):
            CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "MW": None}))(water(0.01))
        bare = CentrifugalPump(CentrifugalPumpParams(head_coeffs=H_COEFFS, rho=RHO, MW=MW_W))
        _, info = bare(water(0.01))
        assert "shaft_power" not in info and "NPSHa" not in info
        with pytest.raises(ValueError, match="efficiency"):
            bare.curve.efficiency(0.01)

    def test_jit_and_vmap(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**PUMP))
        f = jax.jit(lambda q: pump(water(q))[1]["head"])
        assert float(f(0.01)) == pytest.approx(float(pump(water(0.01))[1]["head"]), rel=1e-12)
        qs = jnp.array([0.005, 0.01, 0.02])
        assert jax.vmap(f)(qs).shape == (3,)

    def test_in_series_with_pipe_unit(self):
        # sequential-modular: pump then pipe returns to the suction pressure at the operating flow
        pump = CentrifugalPump(CentrifugalPumpParams(**PUMP))
        pp = pipe(dz=LIFT)
        sysc = system_curve(0.0, pp)
        Q = float(operating_point(pump, sysc))
        out, _ = pump(water(Q, P=2e5))
        end, _ = pp(out)
        assert float(end["P"]) == pytest.approx(2e5, abs=1e-3)

    def test_eo_residuals(self):
        pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "cp": 4182.0}))
        inlet = water(0.015)
        outlet, _ = pump(inlet)
        r = pump.eo_residuals([inlet], [outlet])
        assert r.shape == (3,)
        np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-8)
        bad = dict(outlet, P=outlet["P"] + 500.0)
        assert float(pump.eo_residuals([inlet], [bad])[-1]) == pytest.approx(500.0)
        J = jax.jacobian(lambda i: pump.eo_residuals([i], [outlet]))(inlet)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in J.values())


class TestGradients:
    def _Q(self, x):
        """Q* as a function of relative scale factors (D/D_ref, N/N_ref, pipe D / D4)."""
        sD, sN, sP = x
        curve = PumpCurve(H_COEFFS, speed_ratio=sN, diameter_ratio=sD)
        return operating_point(curve, system_curve(LIFT, pipe(D=D4 * sP)))

    def test_check_grads_Q_star_wrt_impeller_speed_and_pipe_diameter(self):
        x0 = jnp.array([1.0, 1.0, 1.0])
        check_grads(self._Q, (x0,), order=1, modes=["rev", "fwd"], rtol=1e-4, atol=1e-8)
        g = np.asarray(jax.grad(self._Q)(x0))
        assert np.all(g > 0)  # bigger impeller, faster, bigger pipe all raise the flow

    @pytest.mark.parametrize("i", [0, 1, 2])
    def test_gradient_matches_finite_difference(self, i):
        x0 = jnp.array([1.0, 1.0, 1.0])
        h = 1e-5
        e = jnp.zeros(3).at[i].set(h)
        fd = (self._Q(x0 + e) - self._Q(x0 - e)) / (2 * h)
        assert float(jax.grad(self._Q)(x0)[i]) == pytest.approx(float(fd), rel=1e-5)

    def test_grad_wrt_pump_and_system_parameters(self):
        def Q(p):
            h0, lift, L = p
            c = jnp.asarray(H_COEFFS).at[0].set(h0)
            return operating_point(PumpCurve(c), system_curve(lift, pipe(L=L)))

        check_grads(Q, (jnp.array([40.0, LIFT, L_PIPE]),), order=1, modes=["rev"], rtol=1e-4, atol=1e-8)

    def test_grad_through_series_parallel(self):
        def Q(s, par):
            c = PumpCurve(H_COEFFS, speed_ratio=s)
            comb = pumps_in_parallel(c, c) if par else pumps_in_series(c, PumpCurve(H_COEFFS))
            return operating_point(comb, system_curve(LIFT, pipe()))

        for par in (False, True):
            check_grads(lambda s: Q(s[0], par), (jnp.array([1.0]),), order=1, modes=["rev"], rtol=1e-4, atol=1e-8)

    def test_economic_pipe_diameter_with_vfd_speed(self):
        """Fixed duty 15 L/s: for each pipe diameter the VFD speed is set so the operating
        point hits the duty; the annualised cost then has an interior optimum in D."""
        from difflow.economics import pump_cost, pump_electricity_cost

        Q_DUTY = 0.015
        eta_c = [0, 90.0, -2e3]

        def speed_for_duty(D):
            sysc = system_curve(LIFT, pipe(D=D))
            f = lambda s, _: PumpCurve(H_COEFFS, speed_ratio=s).head(Q_DUTY) - sysc.head(Q_DUTY)
            return optx.root_find(f, optx.Newton(rtol=1e-12, atol=1e-10), jnp.asarray(1.0)).value

        def cost(logD):
            D = jnp.exp(logD)
            curve = PumpCurve(H_COEFFS, eta_c, speed_ratio=speed_for_duty(D))
            H, eta = curve.head(Q_DUTY), curve.efficiency(Q_DUTY)
            P_kW = curve.shaft_power(Q_DUTY, RHO) / 1e3
            piping = 12.0 * (D / 0.0254) ** 1.1 * L_PIPE  # $/yr, illustrative placeholder
            return piping + 0.2 * pump_cost(P_kW) + 8000 * 3600 * pump_electricity_cost(Q_DUTY, H, eta)

        g = jax.jit(jax.grad(cost))
        h = jax.jit(jax.grad(jax.grad(cost)))
        x = jnp.log(0.12)
        for _ in range(12):  # exact-derivative Newton on log D
            x = x - g(x) / h(x)
        sol = type("Sol", (), {"value": x})
        D_opt = float(jnp.exp(sol.value))
        assert 0.05 < D_opt < 0.5
        assert abs(float(jax.grad(cost)(sol.value))) < 1e-4 * float(cost(sol.value))
        for factor in (0.7, 1.4):
            assert cost(sol.value) < cost(sol.value + jnp.log(factor))
        # the chosen speed delivers the duty at the operating point
        curve = PumpCurve(H_COEFFS, speed_ratio=speed_for_duty(D_opt))
        assert float(operating_point(curve, system_curve(LIFT, pipe(D=D_opt)))) == pytest.approx(Q_DUTY, rel=1e-8)


class TestRegistration:
    def test_catalog_describes_pump(self):
        spec = describe_operation("CentrifugalPump")
        assert spec.ports.n_inlets == 1 and spec.ports.n_outlets == 1
        names = {p.name for p in spec.parameters}
        assert {"head_coeffs", "rho", "N", "D", "N_ref", "D_ref", "motor_efficiency", "npshr_coeffs"} <= names
        assert {p.name for p in spec.parameters if p.required} == {"head_coeffs", "rho"}
        assert spec.equations and spec.assumptions and spec.references

    def test_top_level_exports(self):
        for n in ("CentrifugalPump", "CentrifugalPumpParams", "system_curve",
                  "operating_point", "pumps_in_series", "pumps_in_parallel", "fit_pump_curve", "npsh_available"):
            assert hasattr(difflow, n)
