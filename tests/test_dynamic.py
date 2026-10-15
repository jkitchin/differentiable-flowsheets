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
