"""Tests for difflow_power.units and difflow_power.streams.

The units are checked against each other and against the
equation-oriented branch flows, because a unit operation's only
obligation is to agree with the physics the rest of the plugin already
states. :class:`SeriesBranch` inverts the branch relation and
:class:`BranchFlow` evaluates it directly, so they must agree exactly
--- and if they do not, one of them is wrong in a way no single-unit
test would show.
"""

import jax
import jax.numpy as jnp
import numpy as np
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

PARAMS = [
    BranchParams(r=0.01, x=0.10, b=0.05),          # a line
    BranchParams(r=0.0, x=0.06, tap=0.95),         # a transformer
    BranchParams(r=0.01, x=0.10, shift=0.15),      # a phase shifter
]


def test_stream_slots_carry_what_the_docstring_says():
    s = power_stream(1.0, 0.3, 1.02, 0.05)
    assert float(s["F_P"]) == 1.0
    assert float(s["F_Q"]) == 0.3
    assert float(s["P"]) == 1.02          # voltage magnitude
    assert float(s["T"]) == 0.05          # voltage angle
    assert complex(complex_power(s)) == pytest.approx(1.0 + 0.3j)
    assert complex(complex_voltage(s)) == pytest.approx(
        1.02 * np.exp(0.05j)
    )


def test_stream_derived_quantities():
    s = power_stream(0.8, 0.6, 1.0, 0.0)
    assert float(apparent_power(s)) == pytest.approx(1.0)
    assert float(power_factor(s)) == pytest.approx(0.8)
    assert complex(current(s)) == pytest.approx(0.8 - 0.6j)


@pytest.mark.parametrize("params", PARAMS)
def test_series_branch_and_branch_flow_agree(params):
    """The explicit inversion must reproduce the direct evaluation."""
    inlet = power_stream(1.0, 0.2, 1.0, 0.0)
    outlet, info = SeriesBranch(params)(inlet)
    (s_from, s_to), _ = BranchFlow(params)(inlet, outlet)

    assert float(s_from["F_P"]) == pytest.approx(float(inlet["F_P"]))
    assert float(s_from["F_Q"]) == pytest.approx(float(inlet["F_Q"]))
    # The unit passes power ONWARD, so it is minus the power into the
    # branch at the receiving end.
    assert float(s_to["F_P"]) == pytest.approx(-float(outlet["F_P"]))
    assert float(s_to["F_Q"]) == pytest.approx(-float(outlet["F_Q"]))


@pytest.mark.parametrize("params", PARAMS)
def test_a_branch_never_generates_real_power(params):
    _, info = SeriesBranch(params)(power_stream(1.0, 0.2, 1.0, 0.0))
    assert float(info["loss_p"]) >= -1e-15


def test_line_loss_is_i_squared_r():
    params = BranchParams(r=0.02, x=0.2, b=0.0)
    inlet = power_stream(1.0, 0.0, 1.0, 0.0)
    _, info = SeriesBranch(params)(inlet)
    assert float(info["loss_p"]) == pytest.approx(
        float(info["current"] ** 2 * params.r)
    )


def test_branch_drop_agrees_with_series_branch():
    """Same branch, same result, whether the flow or the current is given."""
    params = BranchParams(r=0.01, x=0.1, b=0.05)
    inlet = power_stream(1.0, 0.2, 1.0, 0.0)
    by_power, _ = SeriesBranch(params)(inlet)
    by_current, _ = BranchDrop(params)(inlet, current(inlet))
    assert float(by_current["P"]) == pytest.approx(float(by_power["P"]))
    assert float(by_current["T"]) == pytest.approx(float(by_power["T"]))


def test_load_draw_subtracts_constant_power_regardless_of_voltage():
    unit = LoadDraw(LoadParams(0.5, 0.2))
    for vm in (0.9, 1.0, 1.1):
        outlet, info = unit(power_stream(1.0, 0.4, vm, 0.0))
        assert float(outlet["F_P"]) == pytest.approx(0.5)
        assert float(outlet["F_Q"]) == pytest.approx(0.2)
        assert float(outlet["P"]) == pytest.approx(vm)
    # The CURRENT drawn does depend on voltage --- that is the
    # nonlinearity a power flow iterates around.
    low = unit(power_stream(1.0, 0.4, 0.9, 0.0))[1]["current"]
    high = unit(power_stream(1.0, 0.4, 1.1, 0.0))[1]["current"]
    assert float(low) > float(high)


def test_shunt_output_falls_with_the_square_of_voltage():
    """The known weakness of capacitor banks, straight from the model."""
    unit = ShuntDraw(ShuntParams(b_pu=0.2))
    at_one = unit(power_stream(0.0, 0.0, 1.0, 0.0))[1]["drawn_q"]
    at_low = unit(power_stream(0.0, 0.0, 0.9, 0.0))[1]["drawn_q"]
    assert float(at_low) == pytest.approx(float(at_one) * 0.81)
    # A capacitor INJECTS vars, so the draw is negative.
    assert float(at_one) < 0.0


def test_generator_inject_adds_power_and_prices_it():
    unit = GeneratorInject(
        GeneratorParams(p_pu=1.0, q_pu=0.3, cost=(0.11, 5.0, 150.0))
    )
    outlet, info = unit(power_stream(0.0, 0.0, 1.0, 0.0))
    assert float(outlet["F_P"]) == pytest.approx(1.0)
    assert float(info["cost"]) == pytest.approx(
        0.11 * 100 ** 2 + 5 * 100 + 150
    )


def test_slack_source_pins_the_voltage_and_passes_power():
    unit = SlackSource(SlackSourceParams(1.04, 0.0))
    outlet, info = unit(power_stream(2.0, 0.5, 0.3, 1.0))
    assert float(outlet["P"]) == pytest.approx(1.04)
    assert float(outlet["T"]) == pytest.approx(0.0)
    assert float(outlet["F_P"]) == pytest.approx(2.0)


def test_bus_node_sums_power_and_shares_the_first_voltage():
    outlet, info = BusNode()(
        power_stream(1.0, 0.2, 1.01, 0.03),
        power_stream(-0.4, 0.1, 0.5, 0.0),     # a wrong voltage, ignored
    )
    assert float(outlet["F_P"]) == pytest.approx(0.6)
    assert float(outlet["F_Q"]) == pytest.approx(0.3)
    assert float(outlet["P"]) == pytest.approx(1.01)
    assert info["n_inlets"] == 2


def test_bus_node_needs_an_inlet():
    with pytest.raises(ValueError, match="at least one inlet"):
        BusNode()()


def test_power_split_conserves_and_keeps_the_bus_voltage():
    (a, b), _ = PowerSplit(SplitParams(0.3))(
        power_stream(1.0, 0.4, 1.02, 0.01)
    )
    assert float(a["F_P"]) == pytest.approx(0.3)
    assert float(b["F_P"]) == pytest.approx(0.7)
    assert float(a["P"]) == float(b["P"]) == pytest.approx(1.02)


def test_ladder_close_has_its_fixed_point_at_zero_leftover():
    unit = LadderClose(LadderCloseParams(1.02, 0.0))
    infeed = power_stream(1.0, 0.3, 1.02, 0.0)
    # A leftover of zero must leave the infeed alone.
    out, _ = unit(power_stream(0.0, 0.0, 0.9, 0.0), infeed)
    assert float(out["F_P"]) == pytest.approx(1.0)
    assert float(out["F_Q"]) == pytest.approx(0.3)
    # A positive leftover means too much was pushed in.
    out, _ = unit(power_stream(0.1, 0.0, 0.9, 0.0), infeed)
    assert float(out["F_P"]) == pytest.approx(0.9)


def test_transformer_refuses_to_be_a_line():
    with pytest.raises(ValueError, match="use SeriesBranch"):
        Transformer(BranchParams(r=0.01, x=0.1))
    with pytest.raises(ValueError, match="line charging"):
        Transformer(BranchParams(x=0.1, b=0.02, tap=0.95))
    Transformer(BranchParams(x=0.1, tap=0.95))       # fine


def test_units_are_differentiable_end_to_end():
    """A two-unit chain, differentiated with respect to the reactance."""

    def delivered(x):
        outlet, _ = SeriesBranch(BranchParams(r=0.01, x=x, b=0.05))(
            power_stream(1.0, 0.2, 1.0, 0.0)
        )
        return jnp.asarray(outlet["P"])

    grad = jax.grad(delivered)(0.1)
    step = 1e-7
    want = (delivered(0.1 + step) - delivered(0.1 - step)) / (2 * step)
    assert float(grad) == pytest.approx(float(want), rel=1e-5)
    assert float(grad) < 0.0        # more reactance, more voltage drop


def test_params_support_the_difflow_mixin_api():
    params = BranchParams(r=0.01, x=0.1)
    assert params["x"] == 0.1
    assert "tap" in params
    assert params.update(x=0.2)["x"] == 0.2
    assert params["x"] == 0.1       # functional, not in place


def test_registry_registration_covers_the_units():
    from difflow.plugins import OperationRegistry

    registry = OperationRegistry()
    dp.register(registry)
    names = {
        "SeriesBranch", "BranchDrop", "BranchFlow", "Transformer",
        "SlackSource", "LoadDraw", "ShuntDraw", "GeneratorInject",
        "BusNode", "PowerSplit", "LadderClose",
    }
    for name in names:
        assert registry.get(name) is not None
