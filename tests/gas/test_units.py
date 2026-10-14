"""Smoke tests for difflow_gas unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow.streams import make_stream
from difflow_gas.streams import FLOW_KEY, NotAGasStream, gas_stream
from difflow_gas.units import (
    AffineFlow,
    BackPipe,
    Compressor,
    CompressorBoost,
    ControlValveDrop,
    FlowMinus,
    FlowSplit,
    GasPipe,
    Junction,
    OpenValve,
    PipePressure,
    PressureDrivenPipe,
    PressureEqual,
    SourceHead,
    TearSplit,
    adiabatic_power_w,
)

BETA = 1.0e8   # Pa^2/(kg/s)^2
T = 283.15
P0 = 50.0e5


@pytest.mark.parametrize("make,arity", [
    (lambda: Compressor(1.2), 1),
    (lambda: CompressorBoost(1.2, 1), 2),
    (lambda: GasPipe(BETA), 1),
    (lambda: BackPipe(BETA), 2),
    (lambda: PipePressure(BETA, 1), 2),
    (lambda: PressureDrivenPipe(BETA), 2),
    (lambda: OpenValve(), 1),
    (lambda: PressureEqual(), 2),
    (lambda: ControlValveDrop(1.0e5, 1), 2),
    (lambda: SourceHead(P0), 1),
    (lambda: Junction(), 2),
    (lambda: FlowSplit(1.0), 1),
    (lambda: TearSplit(), 2),
    (lambda: AffineFlow(0.0, (1.0, -1.0), T, P0), 2),
    (lambda: FlowMinus(), 2),
])
def test_unit_smoke(make, arity):
    out = make()(*[gas_stream(10.0, T, P0)] * arity)
    outs = out if isinstance(out, tuple) else (out,)
    for o in outs:
        assert jnp.isfinite(o[FLOW_KEY]) and jnp.isfinite(o["P"])


def test_flow_split_conserves_mass():
    a, b = FlowSplit(4.0)(gas_stream(10.0, T, P0))
    assert float(a[FLOW_KEY] + b[FLOW_KEY]) == pytest.approx(10.0, abs=1e-8)


def test_adiabatic_power_is_finite():
    inlet = gas_stream(10.0, T, P0)
    assert jnp.isfinite(adiabatic_power_w(inlet, Compressor(1.3)(inlet)))


def test_gas_pipe_gradient_is_finite():
    g = jax.grad(lambda beta: GasPipe(beta)(gas_stream(10.0, T, P0))["P"])(BETA)
    assert jnp.isfinite(g)


def test_pipe_pressure_rejects_bad_direction():
    with pytest.raises(ValueError):
        PipePressure(BETA, 0)


def test_affine_flow_checks_arity():
    unit = AffineFlow(const=0.0, signs=(1.0, -1.0), T_k=T, P_pa=P0)
    with pytest.raises(ValueError):
        unit(gas_stream(1.0, T, P0))


def test_non_gas_stream_raises_not_a_gas_stream():
    chemistry = make_stream({"water": 1.0, "ethanol": 2.0}, T, P0)
    with pytest.raises(NotAGasStream, match="species_order"):
        Compressor(1.2)(chemistry)
    assert issubclass(NotAGasStream, KeyError)
