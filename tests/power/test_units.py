"""Smoke and contract tests for difflow_power.units and difflow_power.streams."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.streams import (
    apparent_power,
    complex_power,
    complex_voltage,
    current,
    power_factor,
    power_stream,
)
from difflow_power.units import (
    BranchDrop,
    BranchFlow,
    BranchParams,
    BusNode,
    GeneratorInject,
    GeneratorParams,
    LadderClose,
    LadderCloseParams,
    LoadDraw,
    LoadParams,
    PowerSplit,
    SeriesBranch,
    ShuntDraw,
    ShuntParams,
    SlackSource,
    SlackSourceParams,
    SplitParams,
    Transformer,
)

REGISTERED = {
    "SeriesBranch", "BranchDrop", "BranchFlow", "Transformer",
    "SlackSource", "LoadDraw", "ShuntDraw", "GeneratorInject",
    "BusNode", "PowerSplit", "LadderClose",
}


def _finite(stream):
    return all(bool(jnp.all(jnp.isfinite(v))) for v in stream.values())


def test_stream_helpers_smoke():
    s = power_stream(0.8, 0.6, 1.0, 0.0)
    assert set(s) >= {"F_P", "F_Q", "P", "T"}
    for f in (apparent_power, power_factor, complex_power,
              complex_voltage, current):
        assert bool(jnp.isfinite(f(s)))


def test_units_smoke():
    s = power_stream(1.0, 0.2, 1.0, 0.0)
    params = BranchParams(r=0.01, x=0.10, b=0.05)
    outlet, _ = SeriesBranch(params)(s)
    assert _finite(outlet)
    (s_from, s_to), _ = BranchFlow(params)(s, outlet)
    assert _finite(s_from) and _finite(s_to)
    assert _finite(BranchDrop(params)(s, current(s))[0])
    assert _finite(Transformer(BranchParams(x=0.1, tap=0.95))(s)[0])
    assert _finite(LoadDraw(LoadParams(0.5, 0.2))(s)[0])
    assert _finite(ShuntDraw(ShuntParams(b_pu=0.2))(s)[0])
    assert _finite(GeneratorInject(
        GeneratorParams(p_pu=1.0, q_pu=0.3, cost=(0.11, 5.0, 150.0)))(s)[0])
    assert _finite(SlackSource(SlackSourceParams(1.04, 0.0))(s)[0])
    assert _finite(BusNode()(s, s)[0])
    a, b = PowerSplit(SplitParams(0.3))(s)[0]
    assert float(a["F_P"] + b["F_P"]) == pytest.approx(float(s["F_P"]))
    assert _finite(LadderClose(LadderCloseParams(1.02, 0.0))(s, s)[0])


def test_unit_grad_is_finite():
    grad = jax.grad(
        lambda x: SeriesBranch(BranchParams(r=0.01, x=x, b=0.05))(
            power_stream(1.0, 0.2, 1.0, 0.0)
        )[0]["P"]
    )(0.1)
    assert bool(jnp.isfinite(grad))


def test_unit_validation():
    with pytest.raises(ValueError, match="at least one inlet"):
        BusNode()()
    with pytest.raises(ValueError, match="use SeriesBranch"):
        Transformer(BranchParams(r=0.01, x=0.1))
    with pytest.raises(ValueError, match="line charging"):
        Transformer(BranchParams(x=0.1, b=0.02, tap=0.95))


def test_params_support_the_difflow_mixin_api():
    params = BranchParams(r=0.01, x=0.1)
    assert params["x"] == 0.1
    assert "tap" in params
    assert params.update(x=0.2)["x"] == 0.2
    assert params["x"] == 0.1


def test_registered_units_have_report_metadata():
    from difflow.plugins import OperationRegistry
    from difflow.report.metadata import get_metadata

    registry = OperationRegistry()
    dp.register(registry)
    for name in REGISTERED:
        meta = get_metadata(registry.get(name))
        assert meta.symbol and meta.description and meta.equations, name
