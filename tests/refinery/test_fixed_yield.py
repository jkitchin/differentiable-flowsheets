"""Fixed-yield blocks (#537): smoke, gradient, validation, stream wiring.
The benchmark reproduction and the sensitivity ranking are by hand in
tests/reference/du2025_fixed_yield_check.py."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.fixed_yield import FixedYieldBlock, PropertyStream, balance

jax.config.update("jax_enable_x64", True)

SDA = FixedYieldBlock("sda", {"vr": 1.0}, {"dao": 0.26, "pitch": 0.73},
                      transfer={"pitch": {"S_wt": ("vr", 1.15), "CCR_wt": ("vr", 1.04)}},
                      fixed={"dao": {"SG": 0.95, "S_wt": 0.6, "CCR_wt": 0.2}})
VR = PropertyStream(jnp.asarray(10.0), {"SG": 1.03, "S_wt": 3.0, "CCR_wt": 20.0})


def test_smoke_and_loss():
    out = SDA({"vr": VR})
    assert float(out["dao"].mass) == pytest.approx(2.6)
    assert float(out["pitch"].props["S_wt"]) == pytest.approx(3.45)
    assert float(out["loss"].mass) == pytest.approx(0.1)          # the 1 % the yields leave
    assert abs(float(balance(SDA, {"vr": VR}, out))) < 1e-14


def test_grad_finite():
    g = jax.grad(lambda m: SDA({"vr": PropertyStream(m, VR.props)})["pitch"].mass)(10.0)
    assert float(g) == pytest.approx(0.73)


def test_bad_block_raises():
    with pytest.raises(ValueError, match="unknown inlet"):
        FixedYieldBlock("x", {"a": 1.0}, {"b": 1.0}, transfer={"b": {"SG": ("z", 1.0)}})
    with pytest.raises(ValueError, match="positive"):
        FixedYieldBlock("x", {"a": 0.0}, {"b": 1.0})


def test_missing_feed_raises():
    with pytest.raises(KeyError, match="no feed"):
        SDA({})


def test_missing_property_raises():
    with pytest.raises(KeyError, match="lacks"):
        SDA({"vr": PropertyStream(jnp.asarray(1.0), {"SG": 1.0})})


def test_stream_wiring():
    out = SDA({"vr": VR})
    feed = PropertyStream(out["pitch"].mass, {**out["pitch"].props, "SG": 1.09}).to_coker_feed()
    assert float(feed.ccr) == pytest.approx(0.208)
    comp = PropertyStream(jnp.asarray(1.0), {"SG": 0.74, "RON": 95.0, "S_wt": 0.001}).to_blend_component("x")
    assert float(comp.properties["S_ppm"]) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="needs"):
        PropertyStream(jnp.asarray(1.0), {}).to_coker_feed()


def test_du2025_adapter_mini():
    from difflow_refinery.benchmarks.du2025 import load_case
    from difflow_refinery.benchmarks.du2025.units import fixed_yield_block
    data = load_case(Path(__file__).parent / "du2025_mini" / "case1")
    blk = fixed_yield_block(data, "Upf0", named=False)
    assert blk.outlets == {"s4": 0.7, "s5": 0.3} and blk.transfer["s4"]["Q0"] == ("s1", 0.5)
    assert blk.fixed["s5"] == {"Q0": 0.2, "Q1": 0.75}
    out = blk({"s1": PropertyStream(jnp.asarray(2.0), {"Q0": 0.4})})
    assert float(out["s4"].props["Q0"]) == pytest.approx(0.2)
    assert "loss" not in out
