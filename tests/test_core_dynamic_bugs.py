"""Regression tests for bug fixes #120, #121, #122, #124 (simplest form)."""

import jax
import jax.numpy as jnp

from difflow.dynamic import (
    DynamicCSTR,
    DynamicFlashDrum,
    DynamicTank,
    integrate_dae,
    integrate_unit,
)
from difflow.initialization import wegstein_acceleration
from difflow.streams import make_stream

jax.config.update("jax_enable_x64", True)


def test_bug120_non_equimolar_adiabatic_cstr_is_finite():
    cstr = DynamicCSTR(
        volume=1.0, species_order=["A", "B"],
        rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
        stoich=jnp.array([[-1.0], [2.0]]), rate_params={"k": 0.5},
        mode="adiabatic", dH_rxn=jnp.array([-10000.0]),
    )
    feed = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
    result = integrate_unit(cstr, inputs={"inlet": feed}, t_span=(0.0, 1.0), n_steps=50)
    assert jnp.isfinite(result.trajectory.y[-1, cstr.state_spec().get_index("T")])


def test_bug121_tank_volume_does_not_blow_up():
    tank = DynamicTank(max_volume=0.01, species_order=["A", "B"])
    feed = make_stream({"A": 0.5, "B": 0.5}, T=300.0, P=101325.0)
    result = integrate_unit(tank, inputs={"inlet": feed}, t_span=(0.0, 1.0), n_steps=100)
    V_final = float(result.trajectory.y[-1, tank.state_spec().get_index("V")])
    assert 0 < V_final and abs(V_final - 0.005) < 0.001


def test_bug122_wegstein_is_finite():
    result = wegstein_acceleration(jnp.array([0.0]), jnp.array([0.25]),
                                   jnp.array([0.25]), jnp.array([0.375]))
    assert jnp.isfinite(result).all()


def test_bug124_dae_result_contains_residual_info():
    flash = DynamicFlashDrum(volume=1.0, species_order=["A", "B"])
    feed = make_stream({"A": 0.5, "B": 0.5}, T=350.0, P=101325.0)
    result = integrate_dae(flash, inputs={"inlet": feed}, t_span=(0.0, 1.0),
                           method="RK4", n_steps=100)
    assert jnp.isfinite(result.info["max_algebraic_residual"])
    assert "algebraic_converged" in result.info
