"""Tests for difflow_gas unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_gas.physics import compressor_power
from difflow.streams import make_stream
from difflow_gas.streams import (
    FLOW_KEY,
    NotAGasStream,
    gas_flow,
    gas_stream,
)
from difflow_gas.units import (
    MIN_P_SQUARED,
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


# ---------------------------------------------------------------------------
# pipes
# ---------------------------------------------------------------------------


def test_gas_pipe_drops_pressure_forward_flow():
    out = GasPipe(BETA)(gas_stream(10.0, T, P0))
    assert float(out["P"]) == pytest.approx((P0**2 - BETA * 100.0) ** 0.5)
    assert float(out[FLOW_KEY]) == 10.0


def test_gas_pipe_raises_pressure_reverse_flow():
    out = GasPipe(BETA)(gas_stream(-10.0, T, P0))
    assert float(out["P"]) == pytest.approx((P0**2 + BETA * 100.0) ** 0.5)


def test_gas_pipe_floor_active_for_unphysical_state():
    # drop far exceeding the inlet head: pressure floors at 0.5 bar
    out = GasPipe(BETA)(gas_stream(1000.0, T, P0))
    assert float(out["P"]) == pytest.approx(MIN_P_SQUARED**0.5)


def test_back_pipe_inverts_gas_pipe():
    q = 12.0
    node = GasPipe(BETA)(gas_stream(q, T, P0))
    src = BackPipe(BETA)(node, gas_stream(q, T, 1.0))
    assert float(src["P"]) == pytest.approx(P0, rel=1e-12)
    assert float(src[FLOW_KEY]) == q


@pytest.mark.parametrize("q", [15.0, -15.0])
def test_pipe_pressure_forward_direction_matches_gas_pipe(q):
    parent = gas_stream(0.0, T, P0)
    flow = gas_stream(q, T, 1.0)
    out = PipePressure(BETA, +1)(parent, flow)
    ref = GasPipe(BETA)(gas_stream(q, T, P0))
    assert float(out["P"]) == pytest.approx(float(ref["P"]), rel=1e-12)
    assert float(out[FLOW_KEY]) == q


@pytest.mark.parametrize("q", [15.0, -15.0])
def test_pipe_pressure_directions_are_inverse(q):
    """Traversing an arc forward then backward recovers the pressure."""
    parent = gas_stream(0.0, T, P0)
    flow = gas_stream(q, T, 1.0)
    down = PipePressure(BETA, +1)(parent, flow)
    back = PipePressure(BETA, -1)(down, flow)
    assert float(back["P"]) == pytest.approx(P0, rel=1e-12)


def test_pipe_pressure_rejects_bad_direction():
    with pytest.raises(ValueError):
        PipePressure(BETA, 0)


@pytest.mark.parametrize("q", [8.0, -8.0])
def test_pressure_driven_pipe_inverts_gas_pipe(q):
    upstream = gas_stream(0.0, T, P0)
    downstream = GasPipe(BETA)(gas_stream(q, T, P0))
    out = PressureDrivenPipe(BETA)(upstream, downstream)
    assert float(out[FLOW_KEY]) == pytest.approx(q, rel=1e-12)
    # output carries the downstream pressure
    assert float(out["P"]) == pytest.approx(float(downstream["P"]))


def test_pressure_driven_pipe_zero_flow_at_equal_pressures():
    s = gas_stream(0.0, T, P0)
    assert float(PressureDrivenPipe(BETA)(s, s)[FLOW_KEY]) == 0.0


# ---------------------------------------------------------------------------
# compressors
# ---------------------------------------------------------------------------


def test_compressor_boosts_pressure():
    out = Compressor(1.25)(gas_stream(10.0, T, P0))
    assert float(out["P"]) == pytest.approx(1.25 * P0)
    assert float(out[FLOW_KEY]) == 10.0


def test_compressor_boost_directions_are_inverse():
    parent = gas_stream(0.0, T, P0)
    flow = gas_stream(10.0, T, 1.0)
    up = CompressorBoost(1.25, +1)(parent, flow)
    assert float(up["P"]) == pytest.approx(1.25 * P0)
    back = CompressorBoost(1.25, -1)(up, flow)
    assert float(back["P"]) == pytest.approx(P0, rel=1e-12)


def test_adiabatic_power_matches_physics_formula():
    inlet = gas_stream(10.0, T, P0)
    outlet = Compressor(1.3)(inlet)
    w_stream = float(adiabatic_power_w(inlet, outlet))
    w_direct = compressor_power(10.0, 1.3, t_in_k=T)
    assert w_stream == pytest.approx(w_direct, rel=1e-12)


# ---------------------------------------------------------------------------
# valves, control valves, short pipes
# ---------------------------------------------------------------------------


def test_open_valve_and_pressure_equal_preserve_pressure():
    inlet = gas_stream(7.0, T, P0)
    assert float(OpenValve()(inlet)["P"]) == P0
    out = PressureEqual()(gas_stream(0.0, T, P0), gas_stream(7.0, T, 1.0))
    assert float(out["P"]) == P0
    assert float(out[FLOW_KEY]) == 7.0


def test_control_valve_drop_directions():
    parent = gas_stream(0.0, T, P0)
    flow = gas_stream(5.0, T, 1.0)
    dp = 2.0e5
    down = ControlValveDrop(dp, +1)(parent, flow)
    assert float(down["P"]) == pytest.approx(P0 - dp)
    up = ControlValveDrop(dp, -1)(down, flow)
    assert float(up["P"]) == pytest.approx(P0, rel=1e-12)


def test_control_valve_floors_pressure():
    parent = gas_stream(0.0, T, 1.0e5)
    out = ControlValveDrop(5.0e5, +1)(parent, gas_stream(1.0, T, 1.0))
    assert float(out["P"]) == pytest.approx(0.5e5)


# ---------------------------------------------------------------------------
# topology units
# ---------------------------------------------------------------------------


def test_source_head_pins_pressure():
    out = SourceHead(P0)(gas_stream(30.0, T, 1.0))
    assert float(out["P"]) == P0
    assert float(out[FLOW_KEY]) == 30.0


def test_affine_flow():
    unit = AffineFlow(const=-5.0, signs=(1.0, -1.0), T_k=T, P_pa=P0)
    out = unit(gas_stream(20.0, T, P0), gas_stream(3.0, T, P0))
    assert float(out[FLOW_KEY]) == pytest.approx(-5.0 + 20.0 - 3.0)


def test_affine_flow_checks_arity():
    unit = AffineFlow(const=0.0, signs=(1.0, -1.0), T_k=T, P_pa=P0)
    with pytest.raises(ValueError):
        unit(gas_stream(1.0, T, P0))


def test_flow_split_conserves_mass():
    a, b = FlowSplit(4.0)(gas_stream(10.0, T, P0))
    assert float(a[FLOW_KEY]) == 4.0
    assert float(a[FLOW_KEY] + b[FLOW_KEY]) == pytest.approx(10.0)
    assert float(a["P"]) == float(b["P"]) == P0


def test_tear_split_takes_spec_flow():
    a, b = TearSplit()(gas_stream(10.0, T, P0), gas_stream(-2.5, T, 1.0))
    assert float(a[FLOW_KEY]) == -2.5
    assert float(b[FLOW_KEY]) == pytest.approx(12.5)


def test_junction_sums_flows_pressure_from_first_inlet():
    out = Junction()(gas_stream(4.0, 280.0, P0), gas_stream(6.0, 300.0, 2 * P0))
    assert float(out[FLOW_KEY]) == pytest.approx(10.0)
    assert float(out["P"]) == P0
    # flow-weighted temperature
    assert float(out["T"]) == pytest.approx((4 * 280 + 6 * 300) / 10)


def test_flow_minus():
    out = FlowMinus()(gas_stream(10.0, T, P0), gas_stream(4.0, T, 1.0))
    assert float(out[FLOW_KEY]) == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# differentiability of parametric units
# ---------------------------------------------------------------------------


@pytest.mark.release
def test_units_are_differentiable_in_their_params():
    def head(P_set):
        return SourceHead(P_set)(gas_stream(1.0, T, 1.0))["P"]

    assert float(jax.grad(head)(P0)) == pytest.approx(1.0)

    def boosted(ratio):
        return Compressor(ratio)(gas_stream(1.0, T, P0))["P"]

    assert float(jax.grad(boosted)(1.2)) == pytest.approx(P0)

    def dropped(beta):
        return GasPipe(beta)(gas_stream(10.0, T, P0))["P"]

    g = float(jax.grad(dropped)(BETA))
    expected = -100.0 / (2.0 * (P0**2 - BETA * 100.0) ** 0.5)
    assert g == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# the stream convention
# ---------------------------------------------------------------------------


class TestAStreamThatIsNotGas:
    """Every unit here reads ``F_gas``, and says so when it cannot.

    The plugin models one pseudo-species, so a gas unit on a flowsheet
    whose species are anything else cannot work. Read bare, that came
    out as ``KeyError: 'F_gas'`` --- a stream key the user never typed,
    raised from inside a solve, with nothing to say that the species
    order was the thing to change.
    """

    def test_the_message_names_what_the_stream_actually_carries(self):
        chemistry = make_stream({"water": 1.0, "ethanol": 2.0}, T, P0)
        with pytest.raises(NotAGasStream) as raised:
            Compressor(1.2)(chemistry)
        message = str(raised.value)
        assert "ethanol, water" in message, "what it carries"
        assert "species_order" in message, "and what to change"
        assert FLOW_KEY in message

    def test_it_is_still_a_key_error(self):
        """Code that already catches one around a solve keeps working."""
        assert issubclass(NotAGasStream, KeyError)
        with pytest.raises(KeyError):
            GasPipe(BETA)(make_stream({"water": 1.0}, T, P0))

    def test_the_message_is_not_repr_quoted(self):
        """`KeyError.__str__` is `repr(args[0])`, wrong for a sentence."""
        message = str(NotAGasStream("plain words"))
        assert message == "plain words"

    def test_an_empty_stream_says_so_rather_than_listing_nothing(self):
        with pytest.raises(NotAGasStream, match="no species at all"):
            gas_flow({"T": T, "P": P0})

    @pytest.mark.parametrize("make,arity", [
        (lambda: Compressor(1.2), 1),
        (lambda: CompressorBoost(1.2, 1), 2),
        (lambda: GasPipe(BETA), 1),
        (lambda: BackPipe(BETA), 2),
        (lambda: PipePressure(BETA, 1), 2),
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
    def test_every_unit_that_reads_a_flow_reports_it_the_same_way(
            self, make, arity):
        """One helper behind them all, so none is left raising the bare one.

        `PressureDrivenPipe` is the one unit absent here, and rightly:
        it never reads a flow, it *computes* one from the two pressures.
        """
        wrong = make_stream({"water": 1.0}, T, P0)
        with pytest.raises(NotAGasStream):
            make()(*[wrong] * arity)

    def test_the_pressure_driven_pipe_needs_no_flow_to_read(self):
        """The exception that shows the rule is about reading, not units."""
        out = PressureDrivenPipe(BETA)(gas_stream(0.0, T, P0),
                                       gas_stream(0.0, T, P0 * 0.9))
        assert FLOW_KEY in out, "it produces a flow rather than reading one"

    def test_a_real_gas_stream_is_untouched(self):
        """The guard reads the flow through, it does not reshape it."""
        good = gas_stream(-7.5, T, P0)
        assert float(gas_flow(good)) == pytest.approx(-7.5), "signed, as ever"
        assert gas_flow(good) is good[FLOW_KEY]
