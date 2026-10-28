"""Tests for the unified dynamic modeling framework: state specs, integrators, units."""

import jax.numpy as jnp

from difflow.dynamic import (
    DynamicCSTR,
    DynamicTank,
    DynamicUnit,
    StateSpec,
    StateVar,
    StateVector,
    integrate,
    integrate_rk4,
    integrate_rk45,
    integrate_unit,
    molar_states,
    reactor_states,
    rk4_step,
    sensitivity_analysis,
    thermal_state,
)
from difflow.streams import make_stream


def _decay(t, y):
    return -y


def test_state_spec_api():
    spec = StateSpec([StateVar("n", bounds=(0.0, None)), StateVar("T", scale=300.0)])
    assert spec.names == ["n", "T"] and spec.get_index("T") == 1
    assert spec.get_indices(["T", "n"]) == [1, 0]
    assert (spec + StateSpec([StateVar("P")])).n_states == 3
    assert spec.add(StateVar("P")).n_states == 3 and spec.n_states == 2
    assert jnp.allclose(spec.get_scales(), jnp.array([1.0, 300.0]))
    lower, _ = spec.get_bounds()
    assert lower[0] == 0.0
    state = StateVector.from_dict({"n": 1.0, "T": 2.0}, spec)
    assert state["T"] == 2.0 and state.to_dict()["n"] == 1.0
    assert jnp.allclose(state.get(["T", "n"]), jnp.array([2.0, 1.0]))


def test_state_factories():
    assert molar_states(["A", "B"]).names == ["n_A", "n_B"]
    assert thermal_state().names == ["T"]
    assert "T" in reactor_states(["A"], include_T=True).names


def test_integrators_run():
    y0 = jnp.array([1.0, 2.0])
    result = integrate_rk4(_decay, y0, (0.0, 1.0), n_steps=10)
    assert result.trajectory.y.shape == (11, 2) and result.info.success
    for method in ("RK4", "Euler"):
        assert jnp.all(jnp.isfinite(integrate(_decay, y0, (0.0, 1.0), method=method,
                                              n_steps=10).y_final))
    assert jnp.all(jnp.isfinite(integrate_rk45(_decay, y0, (0.0, 1.0)).y_final))
    assert jnp.all(jnp.isfinite(rk4_step(_decay, 0.0, y0, 0.1)))


def test_sensitivity_analysis_runs():
    _, jac = sensitivity_analysis(lambda t, y, p: -p[0] * y, jnp.array([1.0]),
                                  jnp.array([0.5]), (0.0, 1.0), method="RK4", n_steps=10)
    assert jnp.all(jnp.isfinite(jac))


def test_dynamic_cstr_and_tank():
    cstr = DynamicCSTR(volume=1.0, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
                       stoich=jnp.array([[-1.0], [1.0]]), species_order=["A", "B"],
                       rate_params={"k": 0.1}, mode="isothermal")
    tank = DynamicTank(max_volume=10.0, species_order=["A", "B"], isothermal=True)
    inputs = {"inlet": make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=101325.0)}
    for unit in (cstr, tank):
        assert isinstance(unit, DynamicUnit)
        y0 = unit.initial_state(inputs)
        assert jnp.all(jnp.isfinite(unit.derivatives(0.0, y0, inputs)))
        assert "outlet" in unit.outputs(0.0, y0, inputs)
    result = integrate_unit(cstr, inputs=inputs, t_span=(0.0, 10.0), method="RK4", n_steps=10)
    assert result.info.success


# --- DynamicCSTR heat input: specified duty and jacket (#518) ---------------

def _no_rxn_cstr(**kw):
    return DynamicCSTR(volume=0.01, rate_fn=lambda C, T, p: jnp.array([0.0]),
                       stoich=jnp.array([[-1.0], [1.0]]), species_order=["A", "B"], **kw)


_FEED = {"inlet": make_stream({"A": 1.0, "B": 0.0}, T=300.0, P=1e5)}


def _T_final(unit, inputs=_FEED, t_end=3000.0):
    return integrate_unit(unit, inputs, (0.0, t_end), method="RK4", n_steps=300).y_final[-1]


def test_dynamic_cstr_specified_duty_heats():
    # Steady energy balance with no reaction: F Cp (T - T_in) = Q, Cp = 75 J/(mol K).
    T = _T_final(_no_rxn_cstr(mode="specified_duty", duty=750.0))
    assert abs(float(T) - (300.0 + 750.0 / (1.0 * 75.0))) < 1e-6
    T_t = _T_final(_no_rxn_cstr(mode="specified_duty", duty=lambda t: 750.0))
    assert abs(float(T_t) - float(T)) < 1e-9


def test_dynamic_cstr_jacket_runs_and_grad_UA_finite():
    import jax

    T = _T_final(_no_rxn_cstr(mode="jacket", UA=75.0, Tc=320.0))
    assert jnp.isfinite(T) and float(T) > 300.0
    g = jax.grad(lambda ua: _T_final(_no_rxn_cstr(mode="jacket", UA=ua, Tc=320.0), t_end=600.0))(75.0)
    assert jnp.isfinite(g)


def test_dynamic_cstr_jacket_inputs_through_flowsheet():
    import jax

    from difflow.dynamic import DynamicFlowsheet

    fs = DynamicFlowsheet(["A", "B"])
    fs.add_feed("feed", _FEED["inlet"])
    fs.add_feed("cw", make_stream({"A": 0.0, "B": 0.0}, T=320.0, P=1e5))
    fs.add_unit(_no_rxn_cstr(mode="jacket", UA=75.0, Tc="coolant"),
                inlets={"inlet": "feed", "coolant": "cw"}, outlets={"outlet": "out"}, name="r")

    def T_end(ua):
        return fs.simulate((0.0, 600.0), method="RK4", n_steps=60, params={"UA": ua}).y_final[-1]

    assert jnp.isfinite(jax.grad(T_end)(75.0))
    # A params override is per call: no tracer is left in the unit after grad.
    assert jnp.isfinite(fs.simulate((0.0, 600.0), method="RK4", n_steps=60).y_final[-1])

    fs2 = DynamicFlowsheet(["A", "B"])
    fs2.add_feed("feed", _FEED["inlet"])
    fs2.add_unit(_no_rxn_cstr(mode="jacket", UA=75.0, Tc=lambda t, args: args["Tc"]),
                 inlets={"inlet": "feed"}, outlets={"outlet": "out"}, name="r")
    res = fs2.simulate((0.0, 600.0), method="RK4", n_steps=60, args={"Tc": jnp.array(340.0)})
    assert jnp.isfinite(res.y_final[-1])


def test_dynamic_cstr_heat_input_validation():
    import pytest

    with pytest.raises(ValueError, match="needs duty"):
        _no_rxn_cstr(mode="specified_duty")
    with pytest.raises(ValueError, match="needs UA"):
        _no_rxn_cstr(mode="jacket", Tc=300.0)
    with pytest.raises(ValueError, match="needs Tc"):
        _no_rxn_cstr(mode="jacket", UA=1.0)
    with pytest.raises(ValueError, match="only by mode='jacket'"):
        _no_rxn_cstr(mode="adiabatic", UA=1.0)
    with pytest.raises(ValueError, match="only by mode='specified_duty'"):
        _no_rxn_cstr(mode="jacket", UA=1.0, Tc=300.0, duty=5.0)
    with pytest.raises(ValueError, match="mode must be one of"):
        _no_rxn_cstr(mode="cooled")
