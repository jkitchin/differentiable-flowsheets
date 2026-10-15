"""Smoke and validation tests for the pipe/fluids helpers (NPS tables, fittings,
inverse hydraulics, cavitation flag, meters)."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import Pipe, PipeParams, fluids, make_stream
from difflow.units.pipe import PipeCavitationWarning, stream_vapour_pressure

RHO, MU, D2 = 998.96, 1.0214e-3, 0.0525


def test_nps_lookup_and_errors():
    assert fluids.pipe_id("1-1/2", "Sch 80") == fluids.pipe_id(1.5, 80)
    assert fluids.pipe_id(2, "std") == fluids.pipe_id(2, "40")
    for bad in ((22, "40"), (7, "40"), (2, "120")):
        with pytest.raises(KeyError):
            fluids.pipe_id(*bad)
    assert len(fluids.standard_sizes("40")) > 0
    assert fluids.snap_to_standard_size(0.095)[0] > 0
    with pytest.raises(ValueError):
        fluids.snap_to_standard_size(2.0)


def test_fittings_smoke_and_grad():
    p = Pipe(PipeParams(L=20.0, D=0.04089, rho=RHO, mu=MU,
                        fittings={"elbow_90_standard": 4, "K": 8.0}))
    assert np.isfinite(float(p.hydraulics_at_flow(2e-3)["K_total"]))
    g = jax.grad(lambda e: fluids.fittings_K({"gate_valve": 2, "K": 3.0}, rel_roughness=e))(1e-3)
    assert np.isfinite(float(g))


def test_flow_at_head_and_dp():
    p = Pipe(PipeParams(L=120.0, D=D2, rho=RHO, mu=MU, dz=5.0))
    assert np.isfinite(float(p.flow_at_head(15.0)))
    Q = p.flow_at_dp(2e5)
    assert float(p.hydraulics_at_flow(Q)["dP"]) == pytest.approx(2e5, rel=1e-8)
    assert float(p.flow_at_dp(0.9 * RHO * 9.80665 * 5.0)) == 0.0


def test_flow_area_annulus_smoke():
    A = math.pi / 4 * (0.07793**2 - 0.04826**2)
    p = Pipe(PipeParams(L=24.0, D=0.07793 - 0.04826, rho=RHO, mu=MU, flow_area=A))
    assert np.isfinite(float(p.hydraulics_at_flow(4e-3)["dP"]))


class TestCavitation:
    def feed(self, P=4.0e5):
        return make_stream({"water": 3e-3 * RHO / 18.015e-3}, T=293.15, P=P)

    def uphill(self, **kw):
        return Pipe(PipeParams(L=100.0, D=D2, rho=RHO, mu=MU, MW=18.015, dz=40.0, **kw))

    def test_warns_and_flags(self):
        with pytest.warns(PipeCavitationWarning, match="vapour pressure"):
            _, info = self.uphill()(self.feed())
        assert bool(info["P_out_below_vapour"])

    def test_no_warning_under_jit(self):
        f = jax.jit(lambda P: self.uphill()(self.feed(P))[1]["P_out_below_vapour"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", PipeCavitationWarning)
            f(4e5)

    def test_unknown_species_gives_nan_P_vap(self):
        s = make_stream({"W": 166.0}, T=293.15, P=1e5)
        _, info = Pipe(PipeParams(L=1.0, D=D2, rho=RHO, mu=MU, MW=18.0))(s)
        assert np.isnan(float(info["P_vap"])) and not bool(info["P_out_below_vapour"])

    def test_mixture_vapour_pressure_smoke(self):
        s = make_stream({"water": 1.0, "n_heptane": 1.0}, T=350.0, P=1e5)
        assert np.isfinite(float(stream_vapour_pressure(s)))


def test_rho_mu_default_to_liquid_properties():
    s = make_stream({"water": 166.0}, T=300.0, P=3e5)
    _, info = Pipe(PipeParams(L=50.0, D=D2))(s)
    assert np.isfinite(float(info["dP"]))
    with pytest.raises(ValueError, match="inlet"):
        Pipe(PipeParams(L=50.0, D=D2)).hydraulics_at_flow(1e-3)


def test_meters_smoke():
    assert np.isfinite(float(fluids.orifice_permanent_loss(20e3, 0.5, 0.61)))
    assert np.isfinite(float(fluids.venturi_permanent_loss(1.0, 0.5, 0.065)))
    assert np.isfinite(float(fluids.orifice_cd(0.5, 1e5, 0.1, "corner")))
    with pytest.raises(ValueError):
        fluids.orifice_cd(0.5, 1e5, 0.1, "pipe")
    g = jax.grad(lambda b: fluids.orifice_permanent_loss(2e4, b, 0.6))(jnp.asarray(0.5))
    assert np.isfinite(float(g))
    with pytest.raises(ValueError, match="mu"):
        fluids.orifice_flow(20e3, 0.1023, 0.05, 998.0, Cd="iso5167")
