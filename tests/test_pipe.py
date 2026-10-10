"""Tests for the Pipe unit operation (liquid Darcy-Weisbach flow)."""

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow import Pipe, PipeParams, SpeciesData, IdealThermo, fluids, make_stream
from difflow.catalog import describe_operation

RHO, MU, MW_W = 998.2, 1.002e-3, 18.015  # water, 20 C (CRC / Perry's)
G = 9.80665


def water(Q, P=3e5, T=293.15):
    """Water stream carrying volumetric flow Q (m^3/s)."""
    return make_stream({"W": Q * RHO / (MW_W * 1e-3)}, T, P)


def params(**kw):
    base = dict(L=100.0, D=2.067 * 0.0254, rho=RHO, mu=MU, MW=MW_W)
    base.update(kw)
    return PipeParams(**base)


def colebrook_independent(Re, e):
    x = brentq(lambda x: x + 2 * np.log10(e / 3.7 + 2.51 * x / Re), 1.0, 200.0, xtol=1e-15)
    return x**-2


class TestValidation:
    def test_water_3_Ls_in_2inch_sch40_steel(self):
        """3 L/s of 20 C water, 100 m of 2-in Sch 40 commercial steel.

        No published worked example of exactly this case is quoted: the
        reference here is an independent hand calculation (inputs: 2-in
        Sch 40 ID = 2.067 in; epsilon = 4.6e-5 m; water 998.2 kg/m^3,
        1.002 mPa s) using scipy's root finder on the Colebrook-White equation
        and none of difflow's code. Result: v = 1.386 m/s, Re = 7.25e4,
        f = 0.02254, dP = 41.1 kPa (4.20 m of water).
        """
        D, Q, L, eps = 2.067 * 0.0254, 3e-3, 100.0, 4.6e-5
        v = Q / (np.pi / 4 * D**2)
        Re = RHO * v * D / MU
        f = colebrook_independent(Re, eps / D)
        dP = f * L / D * 0.5 * RHO * v**2

        out, info = Pipe(params())(water(Q))
        assert float(info["v"]) == pytest.approx(v, rel=1e-12)
        assert float(info["Re"]) == pytest.approx(Re, rel=1e-12)
        assert float(info["f"]) == pytest.approx(f, rel=1e-9)
        assert float(info["dP_friction"]) == pytest.approx(dP, rel=1e-9)
        assert float(out["P"]) == pytest.approx(3e5 - dP, rel=1e-9)
        assert (round(v, 3), round(f, 5), round(dP / 1e3, 1)) == (1.386, 0.02254, 41.1)

    def test_churchill_close_to_colebrook(self):
        _, a = Pipe(params())(water(3e-3))
        _, b = Pipe(params(friction_method="churchill"))(water(3e-3))
        assert float(b["dP"]) == pytest.approx(float(a["dP"]), rel=0.02)

    def test_laminar_hagen_poiseuille(self):
        """Laminar dP = 128 mu L Q / (pi D^4), exactly (f = 64/Re)."""
        D, L, Q, mu = 0.02, 10.0, 1e-5, 0.05
        out, info = Pipe(params(L=L, D=D, mu=mu))(water(Q))
        assert float(info["Re"]) < 2100
        assert float(info["dP"]) == pytest.approx(128 * mu * L * Q / (np.pi * D**4), rel=1e-12)

    def test_static_head_and_temperature_flows_unchanged(self):
        out, info = Pipe(params(L=1e-6, dz=10.0))(water(1e-3))
        assert float(info["dP_static"]) == pytest.approx(RHO * G * 10.0)
        assert float(out["P"]) == pytest.approx(3e5 - RHO * G * 10.0, rel=1e-4)
        inlet = water(1e-3)
        assert float(out["T"]) == float(inlet["T"])
        assert float(out["F_W"]) == float(inlet["F_W"])
        # downhill gains pressure
        _, down = Pipe(params(L=1e-6, dz=-10.0))(water(1e-3))
        assert float(down["dP_static"]) < 0

    def test_fittings_dict_and_total_K(self):
        D = 2.067 * 0.0254
        p = params(fittings={"elbow_90_standard": 4, "gate_valve": 1, "entrance_sharp": 1, "exit": 1})
        _, info = Pipe(p)(water(3e-3))
        fT = 1 / (2 * np.log10(3.7 * D / 4.6e-5)) ** 2
        K = 4 * 30 * fT + 8 * fT + 0.5 + 1.0
        assert float(info["K_total"]) == pytest.approx(K, rel=1e-12)
        v = float(info["v"])
        assert float(info["dP_minor"]) == pytest.approx(K * 0.5 * RHO * v**2, rel=1e-12)
        _, info2 = Pipe(params(fittings=K))(water(3e-3))
        assert float(info2["dP_minor"]) == pytest.approx(float(info["dP_minor"]), rel=1e-12)
        assert float(info["head_loss"]) == pytest.approx((float(info["dP_friction"]) + float(info["dP_minor"])) / (RHO * G))

    def test_no_fittings_means_no_minor_loss(self):
        assert float(Pipe(params())(water(3e-3))[1]["dP_minor"]) == 0.0

    def test_callable_properties(self):
        # density and viscosity as callables of the stream (here of T)
        rho = lambda s: RHO - 0.5 * (s["T"] - 293.15)
        mu = lambda s: MU * jnp.exp(-0.02 * (s["T"] - 293.15))
        hot = water(3e-3, T=333.15)
        _, a = Pipe(params(rho=rho, mu=mu))(hot)
        _, b = Pipe(params(rho=float(rho(hot)), mu=float(mu(hot))))(hot)
        assert float(a["dP"]) == pytest.approx(float(b["dP"]), rel=1e-12)
        # hotter water is less viscous: lower Delta P than the cold case
        _, cold = Pipe(params(rho=rho, mu=mu))(water(3e-3, T=293.15))
        assert float(a["Re"]) > float(cold["Re"])

    def test_mixture_MW_dict_and_thermo(self):
        sp = {
            "A": SpeciesData("A", 18.0, (75.0, 0, 0, 0), (4e4, 0.38, 647.0), (10.0, 1700.0, -40.0)),
            "B": SpeciesData("B", 46.0, (110.0, 0, 0, 0), (4e4, 0.38, 513.0), (10.0, 1600.0, -40.0)),
        }
        inlet = make_stream({"A": 40.0, "B": 10.0}, 300.0, 2e5)
        expected_mass = (40 * 18.0 + 10 * 46.0) * 1e-3
        for unit in (
            Pipe(params(MW={"A": 18.0, "B": 46.0})),
            Pipe(params(MW=None), thermo=IdealThermo(sp)),
        ):
            _, info = unit(inlet)
            assert float(info["Q"]) == pytest.approx(expected_mass / RHO, rel=1e-12)

    def test_missing_molar_mass_raises(self):
        with pytest.raises(ValueError, match="molar mass"):
            Pipe(params(MW=None))(water(1e-3))

    def test_jit(self):
        unit = Pipe(params())
        f = jax.jit(lambda q: unit(water(q))[0]["P"])
        assert float(f(3e-3)) == pytest.approx(float(unit(water(3e-3))[0]["P"]))

    def test_zero_flow_is_finite(self):
        out, info = Pipe(params())(water(0.0))
        assert np.isfinite(float(out["P"])) and float(info["dP"]) == pytest.approx(0.0, abs=1e-9)
        g = jax.grad(lambda q: Pipe(params())(water(q))[1]["dP"])(0.0)
        assert np.isfinite(float(g))


class TestGradients:
    @pytest.mark.parametrize("method", ["colebrook", "churchill", "haaland", "swamee_jain"])
    @pytest.mark.parametrize("Q", [1e-4, 3e-3, 2e-2])
    def test_grad_dP_wrt_D(self, method, Q):
        """Gradient of dP w.r.t. the diameter (relative step), incl. fittings."""
        D0 = 2.067 * 0.0254

        def dP(s):
            p = params(D=D0 * s, friction_method=method, fittings={"elbow_90_standard": 3, "exit": 1})
            return Pipe(p)(water(Q))[1]["dP"]

        check_grads(dP, (jnp.asarray(1.0),), order=1, modes=["rev", "fwd"], rtol=1e-4, atol=1e-6)
        g = float(jax.grad(dP)(1.0))
        assert np.isfinite(g) and g < 0  # larger pipe, lower friction loss

    def test_grad_wrt_flow_L_roughness_and_fluid(self):
        def dP(x):
            Q, L, eps, rho, mu = x
            p = params(L=L, roughness=eps, rho=rho, mu=mu)
            inlet = make_stream({"W": Q * rho / (MW_W * 1e-3)}, 293.15, 3e5)
            return Pipe(p)(inlet)[1]["dP"]

        x0 = jnp.array([3e-3, 100.0, 4.6e-5, RHO, MU])
        check_grads(lambda s: dP(x0 * s), (jnp.ones(5),), order=1, modes=["rev"], rtol=1e-4, atol=1e-6)

    def test_economic_pipe_diameter_optimization(self):
        """Annualized pipe + pumping cost has an interior minimum found by grad."""
        Q, L = 5e-3, 200.0

        def cost(logD):
            D = jnp.exp(logD)
            _, info = Pipe(params(L=L, D=D))(water(Q))
            pump_kW = info["dP"] * Q / 0.7 / 1e3
            return 2500.0 * D * L + 3000.0 * pump_kW  # $/yr, illustrative

        solver = optx.BFGS(rtol=1e-10, atol=1e-10)
        sol = optx.minimise(lambda x, _: cost(x), solver, jnp.log(0.1), max_steps=200, throw=False)
        D_opt = float(jnp.exp(sol.value))
        assert 0.02 < D_opt < 0.5
        assert abs(float(jax.grad(cost)(sol.value))) < 1e-3 * float(cost(sol.value))
        assert cost(jnp.log(D_opt)) < cost(jnp.log(0.5 * D_opt)) and cost(jnp.log(D_opt)) < cost(jnp.log(2 * D_opt))


class TestEO:
    def test_eo_residuals_vanish_at_sequential_solution(self):
        unit = Pipe(params(fittings={"gate_valve": 2}, dz=3.0))
        inlet = water(3e-3)
        outlet, _ = unit(inlet)
        r = unit.eo_residuals([inlet], [outlet])
        assert r.shape == (3,)  # n_species + 2
        np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-9)

    def test_eo_residuals_nonzero_when_off(self):
        unit = Pipe(params())
        inlet = water(3e-3)
        outlet, info = unit(inlet)
        bad = dict(outlet, P=outlet["P"] + 1000.0)
        r = unit.eo_residuals([inlet], [bad])
        assert float(r[-1]) == pytest.approx(1000.0)

    def test_eo_residual_jacobian_finite(self):
        unit = Pipe(params())
        inlet = water(3e-3)
        outlet, _ = unit(inlet)
        J = jax.jacobian(lambda i: unit.eo_residuals([i], [outlet]))(inlet)
        assert all(np.all(np.isfinite(np.asarray(v))) for v in J.values())


class TestParallelBranches:
    """Two unequal pipes between the same nodes: solved with root finding."""

    def test_parallel_branches_equal_dP_and_flows_sum(self):
        Q_total = 6e-3
        a = Pipe(params(L=80.0, D=0.0525))
        b = Pipe(params(L=150.0, D=0.0409, fittings={"globe_valve": 1}))

        def dP(unit, Q):
            return unit(water(Q))[1]["dP"]

        def residual(x, _):  # x = fraction of the flow through branch a
            return dP(a, x * Q_total) - dP(b, (1 - x) * Q_total)

        sol = optx.root_find(residual, optx.Newton(rtol=1e-12, atol=1e-10), jnp.asarray(0.5), throw=True)
        x = float(sol.value)
        Qa, Qb = x * Q_total, (1 - x) * Q_total
        assert 0 < x < 1
        assert Qa + Qb == pytest.approx(Q_total, rel=1e-14)
        assert float(dP(a, Qa)) == pytest.approx(float(dP(b, Qb)), rel=1e-9)
        assert Qa > Qb  # shorter, wider branch carries more

        # the outlets of the two branches meet at one pressure
        Pa = float(a(water(Qa))[0]["P"])
        Pb = float(b(water(Qb))[0]["P"])
        assert Pa == pytest.approx(Pb, rel=1e-9)

        # and the split is differentiable (implicit function theorem)
        def split(L_b):
            bb = Pipe(params(L=L_b, D=0.0409, fittings={"globe_valve": 1}))
            r = lambda x, _: dP(a, x * Q_total) - dP(bb, (1 - x) * Q_total)
            return optx.root_find(r, optx.Newton(rtol=1e-12, atol=1e-10), jnp.asarray(0.5), throw=True).value

        assert float(jax.grad(split)(150.0)) > 0  # longer branch b -> more flow in a


class TestRegistration:
    def test_catalog_describes_pipe(self):
        spec = describe_operation("Pipe")
        assert spec.ports.n_inlets == 1 and spec.ports.n_outlets == 1
        names = {p.name for p in spec.parameters}
        assert {"L", "D", "rho", "mu", "roughness", "dz", "fittings", "friction_method"} <= names
        required = {p.name for p in spec.parameters if p.required}
        assert {"L", "D", "rho", "mu"} <= required
        assert spec.equations and spec.assumptions and spec.references

    def test_params_default_roughness_is_commercial_steel(self):
        assert PipeParams(L=1.0, D=0.1, rho=1000.0, mu=1e-3).roughness == 4.6e-5
