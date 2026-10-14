"""Flowsheet.make_objective_fn and create_objective (issue #62)."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    Splitter,
    Unit,
    IdealThermo,
    SpeciesData,
    make_stream,
    create_objective,
)

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def cstr_flowsheet():
    species_data = {
        s: SpeciesData(s, MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                       Hvap_coeffs=(35000.0, 0.38, 500.0),
                       antoine_coeffs=(10.0, 3000.0, -50.0))
        for s in ("A", "B")
    }

    def rate_fn(C, T, params):
        return jnp.array([params["A"] * jnp.exp(-params["Ea"] / (8.314 * T)) * C["A"]])

    params = CSTRParams(
        V=jnp.array(1.0), rate_fn=rate_fn, stoich=jnp.array([[-1.0], [+1.0]]),
        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(50000.0)},
        species_order=["A", "B"],
    )
    cstr = CSTR(params, thermo=IdealThermo(species_data), mode="isothermal")
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit(name="reactor", operation=cstr, inlet_names=["feed"],
                     outlet_names=["product"], params={"T_spec": 350.0}))
    return fs


def _F_B(streams):
    return streams["product"]["F_B"]


def test_objective_uses_params_and_does_not_mutate(cstr_flowsheet):
    objective = cstr_flowsheet.make_objective_fn(_F_B)
    assert float(objective({"reactor.V": jnp.array(0.5)})) != float(
        objective({"reactor.V": jnp.array(5.0)}))
    assert float(cstr_flowsheet.units[0].operation.params.V) == pytest.approx(1.0)
    assert jnp.isfinite(create_objective(cstr_flowsheet, _F_B)({"reactor.V": 2.0}))


def test_objective_grad_finite(cstr_flowsheet):
    objective = cstr_flowsheet.make_objective_fn(_F_B)
    assert jnp.isfinite(jax.grad(lambda V: objective({"reactor.V": V}))(jnp.array(1.0)))


def test_apply_params_levers(cstr_flowsheet):
    fs = cstr_flowsheet._apply_params({
        "reactor.V": jnp.array(3.0),
        "reactor.T_spec": 360.0,
        "feed:feed.total_flow": 20.0,
    })
    assert float(fs.units[0].operation.params.V) == pytest.approx(3.0)
    assert fs.units[0].params["T_spec"] == 360.0
    assert float(fs.feeds["feed"]["F_A"]) == pytest.approx(20.0)
    assert float(cstr_flowsheet.feeds["feed"]["F_A"]) == pytest.approx(10.0)
    feed = cstr_flowsheet._apply_params({"feed:feed.x_B": 0.25}).feeds["feed"]
    assert float(feed["F_B"]) == pytest.approx(2.5)


@pytest.mark.parametrize("key, exc, match", [
    ("nonexistent.V", KeyError, "nonexistent"),
    ("reactor_V", ValueError, "dot notation"),
    ("feed:nope.total_flow", KeyError, "No feed stream named"),
    ("feed:feed.enthalpy", KeyError, "Unknown feed field"),
    ("feed:feed", ValueError, "feed:<stream_name>"),
])
def test_apply_params_errors(cstr_flowsheet, key, exc, match):
    with pytest.raises(exc, match=match):
        cstr_flowsheet._apply_params({key: 1.0})


def test_unknown_unit_params_key_raises():
    fs = Flowsheet(species_order=["A"])
    fs.add_feed("feed", make_stream({"A": 1.0}, T=300.0, P=1e5))
    fs.add_unit(Unit("split", Splitter(["A"]), ["feed"], ["a", "b"], {"split_frac": 0.5}))
    assert fs._apply_params({"split.split_frac": 0.2}).units[0].params["split_frac"] == 0.2
    with pytest.raises(AttributeError, match="not keys of its Unit.params"):
        fs._apply_params({"split.nonsense": 1.0})


def test_solve_keywords_are_forwarded(cstr_flowsheet, monkeypatch):
    seen = {}
    orig = Flowsheet.solve

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(Flowsheet, "solve", spy)
    create_objective(cstr_flowsheet, _F_B, max_iter=777)({"reactor.V": 1.0})
    assert seen["max_iter"] == 777


def _react(inlet, X=0.5):
    out = dict(inlet)
    out["F_A"], out["F_B"] = inlet["F_A"] * (1 - X), inlet["F_B"] + X * inlet["F_A"]
    return out


def _mix(feed, rec):
    out = {k: feed[k] + rec[k] for k in ("F_A", "F_B")}
    out["T"], out["P"] = feed["T"], feed["P"]
    return out


def test_grad_after_a_jitted_grad():
    """Regression: after jit(grad(f)), grad(jit(f)) raised in equinox's cache."""
    sp = ["A", "B"]

    def f(X):
        fs = Flowsheet(species_order=sp, default_P=None)
        fs.add_feed("feed", make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=1e5))
        fs.add_unit(Unit("mix", _mix, ["feed", "rec"], ["rx_in"]))
        fs.add_unit(Unit("rx", _react, ["rx_in"], ["rx_out"], {"X": X}))
        fs.add_unit(Unit("split", Splitter(sp), ["rx_out"], ["rec", "prod"],
                         {"split_frac": 0.5}))
        fs.add_recycle("rec", "rec")
        return fs.solve(max_iter=200)["prod"]["F_B"]

    jax.jit(jax.grad(f))(0.4)
    assert jnp.isfinite(jax.grad(jax.jit(f))(0.4))
