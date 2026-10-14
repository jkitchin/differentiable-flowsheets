"""Tests for DynamicFlowsheet: wiring, state, simulation, events, args and loops."""

import jax
import jax.numpy as jnp
import pytest

from difflow.dynamic import (
    DynamicCSTR,
    DynamicFlowsheet,
    DynamicFlowsheetResult,
    DynamicTank,
    EventSpec,
    InstantaneousUnit,
    StateSpec,
    StateVar,
    integrate,
)
from difflow.streams import make_stream

SPECIES = ["A", "B"]


def _feed(F=1.0, T=350.0):
    return make_stream({"A": F, "B": 0.0}, T=T, P=101325.0)


def _cstr(name="reactor", k=0.1):
    return DynamicCSTR(volume=1.0, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
                       stoich=jnp.array([[-1.0], [1.0]]), species_order=SPECIES,
                       rate_params={"k": k}, name=name)


def _two_unit_fs():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", _feed())
    fs.add_unit(_cstr(), inlet_names=["feed"], outlet_names=["reactor_out"])
    fs.add_unit(DynamicTank(max_volume=10.0, species_order=SPECIES, name="storage"),
                inlet_names=["reactor_out"], outlet_names=["product"])
    return fs


def test_build_and_evaluate():
    fs = _two_unit_fs()
    assert fs.unit_names == ["reactor", "storage"] and fs.n_states == 5
    assert "storage.V" in fs.combined_state_spec().names
    assert "n_units=2" in repr(fs)
    y0 = fs.initial_state()
    assert y0.shape == (5,)
    assert jnp.all(jnp.isfinite(fs.derivatives(0.0, y0)))
    assert {"feed", "reactor_out", "product"} <= set(fs.outputs(0.0, y0))


def test_simulate_and_result_access():
    fs = _two_unit_fs()
    r = fs.simulate(t_span=(0.0, 10.0), method="RK4", n_steps=10)
    assert isinstance(r, DynamicFlowsheetResult)
    assert r.trajectory.y.shape == (11, 5) and r.events == []
    assert r.unit_trajectory("storage").y.shape == (11, 3)
    assert r.unit_state_at("reactor", -1).shape == (2,)
    assert "n_A" in r.get_state_dict("reactor", -1)
    assert jnp.all(jnp.isfinite(fs.steady_state(tol=1e-2, max_iter=10)))


def test_threshold_event_detected():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", _feed())
    fs.add_unit(_cstr(), inlet_names=["feed"], outlet_names=["out"])
    y0 = fs.initial_state()
    event = EventSpec(name="B_up", condition_fn=lambda t, y: y[1] - (y0[1] + 1e-3), direction=1)
    r = fs.simulate(t_span=(0.0, 50.0), method="RK4", n_steps=50, events=[event])
    assert r.events and r.events[0].name == "B_up"


def test_time_varying_feed_and_explicit_connection():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", lambda t: _feed(jnp.where(t < 5.0, 1.0, 2.0)))
    fs.add_unit(_cstr(), inlet_names=["feed"], outlet_names=["out"])
    fs.add_unit(DynamicTank(max_volume=10.0, species_order=SPECIES, name="tank"),
                inlet_names=["tank_inlet"], outlet_names=["product"])
    fs.connect("out", "tank_inlet")
    r = fs.simulate(t_span=(0.0, 10.0), method="RK4", n_steps=10)
    assert jnp.all(jnp.isfinite(r.y_final))


def test_steady_state_unit_in_dynamic_flowsheet():
    from difflow.units.cstr import CSTR, CSTRParams

    cstr = CSTR(CSTRParams(V=1.0, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
                           stoich=jnp.array([[-1.0], [1.0]]), rate_params={"k": 0.1},
                           species_order=SPECIES))
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", _feed())
    fs.add_unit(cstr, inlet_names=["feed"], outlet_names=["out"])
    assert jnp.all(jnp.isfinite(fs.simulate((0.0, 10.0), method="RK4", n_steps=10).y_final))


def test_gradient_through_simulation():
    def b_final(k):
        fs = DynamicFlowsheet(species_order=SPECIES)
        fs.add_feed("feed", _feed())
        fs.add_unit(_cstr(k=k), inlet_names=["feed"], outlet_names=["out"])
        return fs.simulate(t_span=(0.0, 10.0), method="RK4", n_steps=10).y_final[1]

    assert jnp.isfinite(jax.grad(b_final)(0.1))


def test_wiring_errors():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", _feed())
    fs.add_unit(_cstr(), inlet_names=["feed"], outlet_names=["out1"])
    with pytest.raises(ValueError, match="already exists"):
        fs.add_unit(_cstr(), inlet_names=["out1"], outlet_names=["out2"])

    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_unit(_cstr(), inlet_names=["missing_feed"], outlet_names=["out"])
    with pytest.raises(ValueError, match="not found"):
        fs.initial_state()

    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("fresh", _feed())
    fs.add_unit(_cstr(), inlets={"inlet": "fresh"}, outlets={"nope": "product"})
    with pytest.raises(ValueError, match="no output 'nope'"):
        fs.initial_state()


def test_port_maps():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("fresh", _feed())
    fs.add_unit(_cstr("r1"), inlets={"inlet": "fresh"}, outlets={"outlet": "mid"})
    fs.add_unit(_cstr("r2"), {"inlet": "mid"}, {"outlet": "out"})
    assert fs.units[1].inlet_names == ["mid"]
    assert {"fresh", "mid", "out"} <= set(fs.outputs(0.0, fs.initial_state()))


# -- stateless units and algebraic loops (#390) --------------------------------

class _LagHX:
    """Toy exchanger whose outlets each read only their own side's inlet."""

    output_ports = ("hot_out", "cold_out")
    output_dependencies = {"hot_out": ("hot",), "cold_out": ("cold",)}
    name = "hx"

    def state_spec(self):
        return StateSpec([StateVar("Q")])

    def initial_state(self, inputs, params=None):
        return jnp.zeros(1)

    def partial_outputs(self, t, state, inputs, params=None):
        out = {}
        if "hot" in inputs:
            out["hot_out"] = {**inputs["hot"], "T": inputs["hot"]["T"] - state[0] / 10.0}
        if "cold" in inputs:
            out["cold_out"] = {**inputs["cold"], "T": inputs["cold"]["T"] + state[0] / 10.0}
        return out

    def outputs(self, t, state, inputs, params=None):
        return self.partial_outputs(t, state, inputs, params)

    def derivatives(self, t, state, inputs, params=None):
        return jnp.atleast_1d((inputs["hot"]["T"] - inputs["cold"]["T"] - state[0]) / 5.0)


def _loop(hx):
    heater = InstantaneousUnit(
        fn=lambda inputs, params: {"outlet": {**inputs["inlet"], "T": inputs["inlet"]["T"] + 5.0}},
        name="heater")
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("feed", _feed(T=300.0))
    fs.add_unit(hx, inlets={"hot": "hot_feed", "cold": "feed"},
                outlets={"hot_out": "product", "cold_out": "preheated"})
    fs.add_unit(heater, inlets={"inlet": "preheated"}, outlets={"outlet": "hot_feed"})
    return fs


def test_loop_through_a_stateless_unit():
    fs = _loop(_LagHX())
    assert fs.n_states == 1
    plan = [(e.name, None if p is None else set(p)) for e, p in fs._evaluation_plan()]
    assert plan == [("hx", {"cold_out"}), ("heater", {"outlet"}), ("hx", {"hot_out"})]
    assert jnp.all(jnp.isfinite(fs.simulate((0.0, 5.0), method="RK4", n_steps=5).y_final))

    class Opaque(_LagHX):
        output_dependencies = None

    with pytest.raises(ValueError, match="Algebraic loop"):
        _loop(Opaque()).initial_state()


# -- integrator args and InstantaneousUnit(t, args) (#390, #396) ---------------

def test_integrate_passes_args():
    r = integrate(lambda t, y, a: -a["k"] * y, jnp.array([1.0]), (0.0, 1.0), "RK4",
                  args={"k": jnp.array(2.0)}, n_steps=10)
    assert jnp.isfinite(r.y_final[0])


def test_args_feed_does_not_retrace():
    fs = DynamicFlowsheet(species_order=SPECIES)
    fs.add_feed("fresh", lambda t, args: _feed(args["F"]))
    fs.add_unit(_cstr(), inlets={"inlet": "fresh"}, outlets={"outlet": "product"})
    y0 = fs.initial_state(args={"F": jnp.array(1.0)})
    traces = []
    original = fs.derivatives

    def counting(*a, **k):
        traces.append(1)
        return original(*a, **k)

    fs.derivatives = counting
    fs.simulate((0.0, 5.0), y0=y0, method="diffrax:tsit5", args={"F": jnp.array(1.0)})
    n = len(traces)
    fs.simulate((0.0, 5.0), y0=y0, method="diffrax:tsit5", args={"F": jnp.array(2.0)})
    assert len(traces) == n


def test_instantaneous_unit_signatures():
    def run(unit, t, args=None):
        fs = DynamicFlowsheet(species_order=SPECIES)
        fs.add_feed("feed", _feed(T=300.0))
        fs.add_unit(unit, inlets={"inlet": "feed"}, outlets={"outlet": "out"})
        return float(fs.outputs(jnp.asarray(t), fs.initial_state(None, args), None, args)["out"]["T"])

    legacy = InstantaneousUnit(fn=lambda inputs, params: {"outlet": {**inputs["inlet"], "T": 333.0}})
    assert run(legacy, 1.0) == pytest.approx(333.0)
    with_args = InstantaneousUnit(
        fn=lambda t, inputs, params, args: {"outlet": {**inputs["inlet"], "T": args["T0"] + t}})
    assert run(with_args, 2.0, {"T0": jnp.array(400.0)}) == pytest.approx(402.0)

    class Op:
        def __call__(self, inlet, T_out):
            return {**inlet, "T": T_out}

    assert run(InstantaneousUnit(Op(), call_params={"T_out": lambda t, p: 300.0 + t}), 7.0) \
        == pytest.approx(307.0)
