"""Smoke tests for :class:`difflow.ActivityThermo`."""

import jax
import jax.numpy as jnp
import numpy as np

from difflow import ActivityThermo, IdealThermo, NRTLParams, SpeciesData

jax.config.update("jax_enable_x64", True)


def _thermo():
    def sd(name, antoine, Tc, MW):
        return SpeciesData(name=name, MW=MW, Cp_coeffs=(80.0, 0, 0, 0),
                           Hvap_coeffs=(4.0e4, 0.38, Tc), antoine_coeffs=antoine)
    ideal = IdealThermo({
        "ethanol": sd("ethanol", (10.24677, 1598.673, -46.424), 514.0, 46.07),
        "water": sd("water", (9.6543, 1435.264, -64.848), 647.0, 18.015),
    })
    nrtl = NRTLParams(("ethanol", "water"),
                      a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
                      b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
                      alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]))
    return ActivityThermo(ideal, nrtl), ideal


def test_k_values_smoke_and_api():
    thermo, ideal = _thermo()
    Kd = thermo.K_values(355.0, 101325.0, {"ethanol": 0.3, "water": 0.7})
    Ka = thermo.K_values_array(355.0, 101325.0, jnp.array([0.3, 0.7]))
    assert Ka.shape == (2,) and np.all(np.isfinite(np.asarray(Ka)))
    assert float(Kd["water"]) == float(Ka[1])
    assert thermo.species_order == ideal.species_order
    assert thermo.K_depends_on_composition


def test_k_values_grad_finite():
    thermo, _ = _thermo()

    def f(x1, T):
        return jnp.sum(thermo.K_values_array(T, 101325.0, jnp.array([x1, 1 - x1])) ** 2)

    g = jax.grad(f, argnums=(0, 1))(0.3, 355.0)
    assert all(np.isfinite(float(v)) for v in g)
