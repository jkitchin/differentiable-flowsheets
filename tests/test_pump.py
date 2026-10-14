"""Tests for the CentrifugalPump unit, curves, system curve and operating point."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow
from difflow import (
    CentrifugalPump,
    CentrifugalPumpParams,
    Pipe,
    PipeParams,
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

RHO, MU, MW_W = 998.2, 1.002e-3, 18.015
Q_DATA = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0]) * 1e-3
H_DATA = np.array([40.0, 39.6, 37.9, 35.6, 31.8, 27.7])
H_COEFFS = np.asarray(fit_pump_curve(Q_DATA, H_DATA, 2))
PUMP = dict(head_coeffs=H_COEFFS, rho=RHO, eta_max=0.72, Q_bep=20e-3,
            npshr_coeffs=[1.5, 0.0, 2.5e3], N_ref=1750.0, D_ref=0.25, MW=MW_W)
LIFT = 12.0


def pipe(L=400.0):
    return Pipe(PipeParams(L=L, D=0.1023, rho=RHO, mu=MU, MW=MW_W, fittings=5.3, roughness=4.6e-5))


def water(Q, P=1.0e5, T=293.15):
    return make_stream({"W": Q * RHO / (MW_W * 1e-3)}, T, P)


def test_operating_point_smoke():
    pump = CentrifugalPump(CentrifugalPumpParams(**PUMP))
    Q, info = operating_point(pump, system_curve(LIFT, pipe()), return_info=True)
    assert 0.0 < float(Q) < 0.03
    assert np.isfinite(float(info["head"])) and np.isfinite(float(info["shaft_power"]))


def test_series_and_parallel_smoke():
    a = PumpCurve(H_COEFFS)
    assert np.isfinite(float(pumps_in_series(a, a).head(0.01)))
    q = pumps_in_parallel(a, a).split(0.03)
    assert float(jnp.sum(q)) == pytest.approx(0.03, rel=1e-10)


def test_npsh_available_smoke():
    assert np.isfinite(float(npsh_available(101325.0, 84550.0, 961.9, 1.5, 2.0, 0.7)))


def test_suction_pressure_validation():
    with pytest.raises(ValueError, match="D_suction"):
        CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0, "suction_pressure": "static"}))(water(0.01))
    with pytest.raises(ValueError, match="suction_pressure"):
        CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "suction_pressure": "gauge"}))(water(0.01))


def test_unit_smoke_conserves_flow():
    pump = CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "Psat": 2339.0}))
    inlet = water(0.015)
    out, info = pump(inlet)
    assert float(get_flows(out)["W"]) == float(get_flows(inlet)["W"])
    for k in ("head", "shaft_power", "efficiency", "NPSHa", "npsh_margin"):
        assert np.isfinite(float(info[k]))
    assert float(out["P"]) > float(inlet["P"])


def test_missing_molar_mass_and_efficiency():
    with pytest.raises(ValueError, match="molar mass"):
        CentrifugalPump(CentrifugalPumpParams(**{**PUMP, "MW": None}))(water(0.01))
    bare = CentrifugalPump(CentrifugalPumpParams(head_coeffs=H_COEFFS, rho=RHO, MW=MW_W))
    with pytest.raises(ValueError, match="efficiency"):
        bare.curve.efficiency(0.01)


def test_grad_operating_point_finite():
    g = jax.grad(lambda L: operating_point(PumpCurve(H_COEFFS), system_curve(LIFT, pipe(L=L))))(400.0)
    assert np.isfinite(float(g))


def test_catalog_and_exports():
    spec = describe_operation("CentrifugalPump")
    assert spec.ports.n_inlets == 1 and spec.ports.n_outlets == 1
    for n in ("CentrifugalPump", "system_curve", "operating_point", "npsh_available"):
        assert hasattr(difflow, n)
