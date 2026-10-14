"""Smoke tests for the Peng-Robinson and SRK equations of state."""

import jax
import jax.numpy as jnp
import pytest

from difflow.database import get_critical_props
from difflow.eos import (
    SRK, CriticalProperties, PengRobinson, flash_TP_eos,
    phase_identification_parameter,
)
from difflow.thermo import CubicThermo, IdealThermo, SpeciesData

C3C4 = {
    "propane": CriticalProperties(name="propane", Tc=369.8, Pc=4.25e6, omega=0.152, MW=44.10),
    "butane": CriticalProperties(name="butane", Tc=425.1, Pc=3.80e6, omega=0.200, MW=58.12),
}
Y = jnp.array([0.5, 0.5])


def _finite(*xs):
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(x)))) for x in xs)


@pytest.mark.parametrize("EOS", [PengRobinson, SRK])
def test_cubic_eos_properties(EOS):
    eos = EOS(C3C4)
    assert eos.n_species == 2 and eos.species_order == ["propane", "butane"]
    T, P = jnp.array(320.0), jnp.array(3e5)
    vals = [eos.alpha(T), eos.solve_Z(T, P, Y, phase="vapor"),
            eos.molar_volume(T, P, Y, phase="vapor"),
            eos.fugacity_coefficient(T, P, Y, phase="vapor"), eos.K_values_wilson(T, P),
            eos.enthalpy_departure(T, P, Y, "vapor"), eos.entropy_departure(T, P, Y, "vapor"),
            phase_identification_parameter(eos, T, P, Y, "vapor")]
    assert _finite(*vals)


def test_flash_two_phase():
    V, x, y = flash_TP_eos(PengRobinson(C3C4), Y, 320.0, 8.5e5)
    assert 0.0 < float(V) < 1.0
    assert float(jnp.sum(x)) == pytest.approx(1.0, rel=1e-3)
    assert float(jnp.sum(y)) == pytest.approx(1.0, rel=1e-3)


def test_grad_is_finite():
    eos = PengRobinson(C3C4)
    g = jax.grad(lambda T: jnp.sum(eos.fugacity_coefficient(T, 3e5, Y, "vapor"))
                 + eos.enthalpy_departure(T, 3e5, Y, "vapor"))(jnp.array(320.0))
    assert _finite(g)


def test_cubic_thermo():
    ideal = IdealThermo({
        "propane": SpeciesData(name="propane", MW=44.10, Cp_coeffs=(73.0, 0.0, 0.0, 0.0),
                               Hvap_coeffs=(18000.0, 0.38, 369.8),
                               antoine_coeffs=(13.72, 1872.5, -25.16)),
        "butane": SpeciesData(name="butane", MW=58.12, Cp_coeffs=(98.0, 0.0, 0.0, 0.0),
                              Hvap_coeffs=(22000.0, 0.38, 425.1),
                              antoine_coeffs=(13.98, 2292.4, -27.86)),
    })
    cubic = CubicThermo(ideal, PengRobinson(C3C4))
    flows = {"propane": 1.0, "butane": 0.5}
    T, P = jnp.array(320.0), jnp.array(3e5)
    assert _finite(cubic.stream_enthalpy(flows, T, phase="vapor", P=P),
                   cubic.stream_entropy(flows, T, "vapor", P))


def test_liquid_root_is_not_floored():
    """#417: a Z = 0.01 floor replaced water's true liquid root (~0.001)."""
    eos = PengRobinson({"water": get_critical_props("water")})
    assert float(eos.solve_Z(300.0, 1e5, jnp.array([1.0]), "liquid")) < 0.005
