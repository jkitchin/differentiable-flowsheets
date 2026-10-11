"""Pipe/fluids additions from the unit-operations book gaps (G03-*, Ch4 #2/#3/#6/#7)."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from scipy.optimize import brentq

from difflow import Pipe, PipeParams, fluids, make_stream
from difflow.units.pipe import STEEL_ROUGHNESS, PipeCavitationWarning, stream_vapour_pressure
from difflow.units.pump import system_curve

RHO, MU, D2 = 998.96, 1.0214e-3, 0.0525  # the book's water at 20 C, 2-in Sch 40
G = 9.80665
FIT_MP = {"entrance_sharp": 1, "elbow_90_standard": 4, "gate_valve": 1, "exit": 1}


# -- G03-2 / Ch4 #2: NPS schedule table ---------------------------------------

class TestNPS:
    @pytest.mark.parametrize("nps, mm", [(1, 26.64), (1.5, 40.89), (2, 52.50), (3, 77.93),
                                         (4, 102.26), (6, 154.05), (10, 254.51)])
    def test_book_sch40_ids(self, nps, mm):
        # ASME B36.10M IDs printed in the book (Ch 3 table, Ch 4 examples)
        assert fluids.pipe_inside_diameter(nps, "40") * 1e3 == pytest.approx(mm, abs=0.01)

    def test_published_points(self):
        # B36.10M rows, OD and wall in inches
        assert fluids.pipe_outside_diameter("1/2") == pytest.approx(0.840 * 0.0254)
        assert fluids.pipe_wall_thickness(8, "80") == pytest.approx(0.500 * 0.0254)
        assert fluids.pipe_wall_thickness(24, "160") == pytest.approx(2.344 * 0.0254)
        assert fluids.pipe_id("12", "STD") == pytest.approx((12.75 - 0.75) * 0.0254)
        assert fluids.pipe_id(12, "40") == pytest.approx((12.75 - 0.812) * 0.0254)

    def test_parsing_and_errors(self):
        a = fluids.pipe_id("1-1/2", "Sch 80")
        assert a == fluids.pipe_id(1.5, 80) == fluids.pipe_id('1 1/2"', "80")
        assert fluids.pipe_id(2, "std") == fluids.pipe_id(2, "40")
        with pytest.raises(KeyError):
            fluids.pipe_id(22, "40")        # B36.10M has no 22-in Sch 40
        with pytest.raises(KeyError):
            fluids.pipe_id(7, "40")
        with pytest.raises(KeyError):
            fluids.pipe_id(2, "120")

    def test_table_consistency(self):
        for nps, (od, walls) in fluids.NPS_SCHEDULES.items():
            for sch in ("10", "40", "80", "160"):
                if sch in walls and "10" in walls and sch != "10":
                    assert walls[sch] > walls["10"]
            ws = [walls[s] for s in ("10", "40", "80", "160") if s in walls]
            assert ws == sorted(ws)
            assert 0 < fluids.pipe_id(nps, "XS") < od * 0.0254

    def test_standard_sizes_and_snap(self):
        sizes = fluids.standard_sizes("40")
        ids = [d for _, d in sizes]
        assert ids == sorted(ids) and 22.0 not in [n for n, _ in sizes]
        assert fluids.snap_to_standard_size(0.095)[0] == 4.0
        assert fluids.snap_to_standard_size(0.095, direction="down")[0] == 3.5
        assert fluids.snap_to_standard_size(0.0905, direction="nearest")[0] == 3.5
        with pytest.raises(ValueError):
            fluids.snap_to_standard_size(2.0)


# -- G03-7 / Ch4 #7: fittings ---------------------------------------------------

class TestFittings:
    def test_mixed_named_and_numeric_K(self):
        rel = STEEL_ROUGHNESS / 0.04089
        p = Pipe(PipeParams(L=20.0, D=0.04089, rho=RHO, mu=MU, fittings={"elbow_90_standard": 4, "K": 8.0}))
        K = float(p.hydraulics_at_flow(2e-3)["K_total"])
        assert K == pytest.approx(4 * float(fluids.fitting_K("elbow_90_standard", rel_roughness=rel)) + 8.0)
        # the Ch 4 strainer case: entrance + K = 2 equals a separate zero-length pipe
        a = Pipe(PipeParams(L=15, D=0.1023, rho=538, mu=1.2e-4, fittings={"entrance_sharp": 1, "K": 2.0}))
        b1 = Pipe(PipeParams(L=15, D=0.1023, rho=538, mu=1.2e-4, fittings={"entrance_sharp": 1}))
        b2 = Pipe(PipeParams(L=0.0, D=0.1023, rho=538, mu=1.2e-4, fittings=2.0))
        q = 0.02
        assert float(a.hydraulics_at_flow(q)["dP"]) == pytest.approx(
            float(b1.hydraulics_at_flow(q)["dP"] + b2.hydraulics_at_flow(q)["dP"]), rel=1e-12)

    def test_new_fittings(self):
        f_T = 0.019  # 2-in
        assert float(fluids.fitting_K("butterfly_valve", f_T=f_T)) == pytest.approx(0.855)
        assert float(fluids.fitting_K("plug_valve", f_T=f_T)) == pytest.approx(0.342)
        # reprinted 2-in K values (fluidnetworkstudio): 7.5 and 1.4
        assert float(fluids.fitting_K("foot_valve", f_T=0.018)) == pytest.approx(7.5, rel=0.02)
        assert float(fluids.fitting_K("foot_valve_hinged", f_T=f_T)) == pytest.approx(1.4, rel=0.03)
        assert float(fluids.fitting_K("strainer_y")) == 2.0
        assert float(fluids.fitting_K("strainer_basket")) == 1.5
        for name in ("butterfly_valve_10_14", "butterfly_valve_16_24"):
            assert "Crane" in fluids.FITTINGS[name].source
        assert "NOT Crane" in fluids.FITTINGS["strainer_y"].source

    def test_fittings_K_grad(self):
        g = jax.grad(lambda e: fluids.fittings_K({"gate_valve": 2, "K": 3.0}, rel_roughness=e))(1e-3)
        assert np.isfinite(float(g)) and float(g) > 0


# -- G03-5: flow from available head ---------------------------------------------

class TestFlowAtHead:
    def pipe(self, eps=4.6e-5, **kw):
        base = dict(L=120.0, D=D2, rho=RHO, mu=MU, roughness=eps, fittings=FIT_MP)
        base.update(kw)
        return Pipe(PipeParams(**base))

    def test_book_motivating_problem(self):
        Q = float(self.pipe().flow_at_head(15.0))
        assert Q * 1e3 == pytest.approx(5.113, abs=5e-4)
        # independent: brentq on the forward hydraulics
        ref = brentq(lambda q: float(self.pipe().hydraulics_at_flow(q)["head_loss"]) - 15.0, 1e-4, 0.05, xtol=1e-14)
        assert Q == pytest.approx(ref, rel=1e-9)

    def test_implicit_gradient(self):
        g = float(jax.grad(lambda e: self.pipe(e).flow_at_head(15.0))(4.6e-5))
        assert g == pytest.approx(-8.78, abs=0.01)
        # IFT by hand: -(dh/deps)/(dh/dQ)
        Q0 = self.pipe().flow_at_head(15.0)
        hl = lambda q, e: self.pipe(e).hydraulics_at_flow(q)["head_loss"]
        ift = -jax.grad(hl, 1)(Q0, 4.6e-5) / jax.grad(hl, 0)(Q0, 4.6e-5)
        assert g == pytest.approx(float(ift), rel=1e-8)
        check_grads(lambda H: self.pipe().flow_at_head(H), (jnp.asarray(15.0),), order=1, modes=["fwd", "rev"])

    def test_flow_at_dp_with_elevation_and_laminar(self):
        p = self.pipe(dz=5.0)
        Q = p.flow_at_dp(2e5)
        assert float(p.hydraulics_at_flow(Q)["dP"]) == pytest.approx(2e5, rel=1e-10)
        assert float(p.flow_at_dp(0.9 * RHO * G * 5.0)) == 0.0   # cannot lift the column
        lam = Pipe(PipeParams(L=5000.0, D=0.15405, rho=870.0, mu=0.05))
        Ql = float(lam.flow_at_dp(217e3))
        assert Ql == pytest.approx(217e3 * math.pi * 0.15405**4 / (128 * 0.05 * 5000.0), rel=1e-9)
        assert float(jax.jit(lambda d: lam.flow_at_dp(d))(217e3)) == pytest.approx(Ql)


# -- G03-6: non-circular cross-section -----------------------------------------------

def test_flow_area_annulus():
    Do, Di = 0.07793, 0.04826
    A = math.pi / 4 * (Do**2 - Di**2)
    p = Pipe(PipeParams(L=24.0, D=Do - Di, rho=RHO, mu=MU, flow_area=A))
    h = p.hydraulics_at_flow(4e-3)
    assert float(h["v"]) == pytest.approx(1.36, abs=0.005)
    assert float(h["dP"]) / 1e3 == pytest.approx(19.6, abs=0.05)
    # same as fluids by hand with D_h
    v = 4e-3 / A
    Re = RHO * v * (Do - Di) / MU
    f = fluids.friction_factor(Re, 4.6e-5 / (Do - Di))
    assert float(h["dP"]) == pytest.approx(float(fluids.darcy_pressure_drop(f, 24.0, Do - Di, RHO, v)), rel=1e-12)
    # scale the area so check_grads' finite-difference step (1e-4) is relative
    check_grads(lambda s: Pipe(PipeParams(L=24.0, D=Do - Di, rho=RHO, mu=MU, flow_area=s * A)).hydraulics_at_flow(4e-3)["dP"],
                (jnp.asarray(1.0),), order=1, modes=["fwd", "rev"])


# -- G03-1: below vapour pressure ----------------------------------------------------

class TestCavitation:
    def feed(self, P=4.0e5):
        return make_stream({"water": 3e-3 * RHO / 18.015e-3}, T=293.15, P=P)

    def uphill(self, **kw):
        return Pipe(PipeParams(L=100.0, D=D2, rho=RHO, mu=MU, MW=18.015, dz=40.0, **kw))

    def test_book_repro_warns_and_flags(self):
        with pytest.warns(PipeCavitationWarning, match="vapour pressure"):
            out, info = self.uphill()(self.feed())
        assert float(out["P"]) == pytest.approx(-33.1e3, abs=100)
        assert bool(info["P_out_below_vapour"])
        # IAPWS saturation pressure of water at 20 C: 2339 Pa
        assert float(info["P_vap"]) == pytest.approx(2339.2, abs=1.0)

    def test_no_warning_when_fine(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", PipeCavitationWarning)
            _, info = self.uphill()(self.feed(9e5))
        assert not bool(info["P_out_below_vapour"])

    def test_jit_flag_no_warning(self):
        f = jax.jit(lambda P: self.uphill()(self.feed(P))[1]["P_out_below_vapour"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", PipeCavitationWarning)
            assert bool(f(4e5)) and not bool(f(9e5))
        g = jax.grad(lambda P: self.uphill()(self.feed(P))[0]["P"])(9e5)
        assert float(g) == pytest.approx(1.0)

    def test_user_P_vap_and_unknown_species(self):
        _, info = self.uphill(P_vap=5e5)(self.feed(9e5))
        assert bool(info["P_out_below_vapour"])
        s = make_stream({"W": 166.0}, T=293.15, P=1e5)
        p = Pipe(PipeParams(L=1.0, D=D2, rho=RHO, mu=MU, MW=18.0))
        _, info = p(s)
        assert np.isnan(float(info["P_vap"])) and not bool(info["P_out_below_vapour"])

    def test_raoult_mixture(self):
        s = make_stream({"water": 1.0, "n_heptane": 1.0}, T=350.0, P=1e5)
        from difflow.water import water_saturation_pressure
        hep = 10 ** (9.02832 - 1268.636 / (350.0 - 56.199))
        assert float(stream_vapour_pressure(s)) == pytest.approx(
            0.5 * float(water_saturation_pressure(350.0)) + 0.5 * hep, rel=1e-12)


# -- G03-3: properties default to liquid_properties -------------------------------------

def test_rho_mu_default_to_liquid_properties():
    from difflow.liquid_properties import liquid_density, liquid_viscosity
    s = make_stream({"water": 166.0}, T=300.0, P=3e5)
    _, a = Pipe(PipeParams(L=50.0, D=D2))(s)
    rho, mu = float(liquid_density("water", 300.0)), float(liquid_viscosity("water", 300.0))
    _, b = Pipe(PipeParams(L=50.0, D=D2, rho=rho, mu=mu, MW=18.01528))(s)
    for k in ("Q", "Re", "dP"):
        assert float(a[k]) == pytest.approx(float(b[k]), rel=1e-4)
    with pytest.raises(ValueError, match="inlet"):
        Pipe(PipeParams(L=50.0, D=D2)).hydraulics_at_flow(1e-3)
    sc = system_curve(10.0, Pipe(PipeParams(L=50.0, D=D2)), delta_P=1e5, inlet=s)
    assert np.isfinite(float(sc.head(2e-3)))
    check_grads(lambda T: Pipe(PipeParams(L=50.0, D=D2))(make_stream({"water": 166.0}, T=T, P=3e5))[1]["dP"],
                (jnp.asarray(300.0),), order=1, modes=["rev"], rtol=1e-4)


# -- G03-9: transition non-monotonicity (documented) ---------------------------------------

def test_transition_blend_dp_non_monotonic_in_mu():
    dp = lambda mu: float(Pipe(PipeParams(L=5000.0, D=0.15405, rho=870.0, mu=mu)).hydraulics_at_flow(0.012)["dP"])
    assert dp(0.050) / 1e3 == pytest.approx(217, abs=0.5)
    assert dp(0.025) / 1e3 == pytest.approx(234, abs=0.5)
    assert "234 kPa at 25 cP" in fluids.friction_factor.__doc__


# -- Ch4 #3 / #6: meters --------------------------------------------------------------------

class TestMeters:
    def test_orifice_permanent_loss(self):
        assert float(fluids.orifice_permanent_loss(20e3, 0.5, 0.61)) / 1e3 == pytest.approx(14.6, abs=0.05)
        b, C = 0.5, 0.61
        root = math.sqrt(1 - b**4 * (1 - C**2))
        assert float(fluids.orifice_permanent_loss(1.0, b, C)) == pytest.approx((root - C * b**2) / (root + C * b**2))
        # fluids' dP_orifice example: 9069.47 Pa
        assert float(fluids.orifice_permanent_loss(17e3, 0.05 / 0.07366, 0.61512)) == pytest.approx(9069.4747, rel=1e-6)
        check_grads(lambda b_: fluids.orifice_permanent_loss(2e4, b_, 0.6), (jnp.asarray(0.5),), order=2,
                    modes=["fwd", "rev"])

    def test_venturi_permanent_loss(self):
        # fluids' dP_venturi_tube example (D = 73.66 mm, d = 50 mm, 17 kPa): 1788.57 Pa
        assert float(fluids.venturi_permanent_loss(17e3, 0.05 / 0.07366, 0.07366)) == pytest.approx(1788.57, rel=5e-3)
        assert float(fluids.venturi_permanent_loss(1.0, 0.5, 0.065)) == pytest.approx(0.1429)
        assert float(fluids.venturi_permanent_loss(1.0, 0.5, 1.0)) == pytest.approx(0.0751)

    def test_orifice_cd_rhg(self):
        # fluids' C_Reader_Harris_Gallagher example (air, flange taps): 0.5990326277
        Re = 4 * 0.12 / (math.pi * 0.07391 * 1.85e-5)
        assert float(fluids.orifice_cd(0.0222 / 0.07391, Re, 0.07391, "flange")) == pytest.approx(
            0.5990326277163659, rel=1e-9)
        # infinite-Re corner-tap limit: 0.5961 + 0.0261 b^2 - 0.216 b^8 + 0.043 b^4/(1-b^4) - 0.123+0.080 terms
        b = 0.5
        Cinf = 0.5961 + 0.0261 * b**2 - 0.216 * b**8 + (0.043 + 0.080 - 0.123) * b**4 / (1 - b**4)
        assert float(fluids.orifice_cd(b, 1e14, 0.1, "corner")) == pytest.approx(Cinf, abs=2e-5)
        # small pipe correction is continuous at 71.12 mm
        c1 = float(fluids.orifice_cd(0.5, 1e5, 0.07112 - 1e-9, "corner"))
        c2 = float(fluids.orifice_cd(0.5, 1e5, 0.07112 + 1e-9, "corner"))
        assert c1 == pytest.approx(c2, abs=1e-9)
        # and adds 0.011 (0.75 - b)(2.8 - D/25.4 mm) below it
        dc = float(fluids.orifice_cd(0.5, 1e5, 0.0254, "corner")) - float(fluids.orifice_cd(0.5, 1e5, 0.1, "corner"))
        assert dc == pytest.approx(0.011 * 0.25 * 1.8, rel=1e-12)
        assert 0.59 < float(fluids.orifice_cd(0.5, 1e5, 0.1, "D-D/2")) < 0.62
        with pytest.raises(ValueError):
            fluids.orifice_cd(0.5, 1e5, 0.1, "pipe")

    def test_orifice_flow_iso5167(self):
        D, d, rho, mu, dP = 0.1023, 0.05, 998.0, 1e-3, 20e3
        Q = fluids.orifice_flow(dP, D, d, rho, Cd="iso5167", mu=mu)
        Re = 4 * rho * float(Q) / (math.pi * D * mu)
        C = float(fluids.orifice_cd(d / D, Re, D))
        assert float(Q) == pytest.approx(float(fluids.orifice_flow(dP, D, d, rho, Cd=C)), rel=1e-10)
        assert float(fluids.orifice_dp(Q, D, d, rho, Cd="iso5167", mu=mu)) == pytest.approx(dP, rel=1e-10)
        # relative scale factors so check_grads' FD step (1e-4) is relative
        check_grads(lambda s: fluids.orifice_flow(s * dP, D, d, rho, Cd="iso5167", mu=mu), (jnp.asarray(1.0),),
                    order=1, modes=["fwd", "rev"])
        check_grads(lambda s: fluids.orifice_flow(dP, D, d, rho, Cd="iso5167", mu=s * mu), (jnp.asarray(1.0),),
                    order=1, modes=["rev"])
        with pytest.raises(ValueError, match="mu"):
            fluids.orifice_flow(dP, D, d, rho, Cd="iso5167")
