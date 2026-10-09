"""Core-database heat capacities: liquid Cp and ideal-gas Cp are different fields (#393)."""

import jax
import jax.numpy as jnp
import pytest

from difflow import IdealThermo, PengRobinson
from difflow.database import (
    _IDEAL_THERMO_DATA,
    get_critical_props,
    get_species_data,
)
from difflow.thermo import CubicThermo

ALL = sorted(_IDEAL_THERMO_DATA)


def _poly(c, T):
    return c[0] + c[1] * T + c[2] * T**2 + c[3] * T**3


def _cubic(species):
    ideal = IdealThermo({s: get_species_data(s) for s in species})
    eos = PengRobinson({s: get_critical_props(s) for s in species})
    return ideal, CubicThermo(ideal, eos)


@pytest.mark.parametrize("name", ALL)
def test_every_species_has_an_ideal_gas_cubic(name):
    """No constant stand-ins: Cp_ig is a real cubic in T for every record."""
    d = get_species_data(name)
    assert d.Cp_vapor_coeffs is not None
    assert any(abs(x) > 0 for x in d.Cp_vapor_coeffs[1:]), f"{name} Cp_ig is constant"
    assert 25.0 < _poly(d.Cp_vapor_coeffs, 298.15) < 300.0


def test_cubic_thermo_water_ideal_gas_cp_is_not_the_liquid_value():
    """The reported bug: CubicThermo used water's liquid 75.3 as its ideal-gas Cp."""
    _, thermo = _cubic(["water"])
    cp = float(thermo.Cp_mix({"water": 1.0}, 300.0))
    assert cp == pytest.approx(33.6, rel=0.01)


def test_ideal_thermo_liquid_cp_stays_liquid():
    ideal, _ = _cubic(["water", "methanol", "n_heptane"])
    assert float(ideal.Cp("water", 298.15)) == pytest.approx(75.3, rel=0.01)
    assert float(ideal.Cp("methanol", 298.15)) == pytest.approx(81.0, rel=0.01)
    assert float(ideal.Cp("n_heptane", 298.15)) == pytest.approx(225.0, rel=0.01)
    # ... and the ideal-gas Cp is the other field
    assert float(ideal.Cp_ig("water", 298.15)) == pytest.approx(33.6, rel=0.01)
    assert float(ideal.Cp_ig("methanol", 298.15)) == pytest.approx(44.1, rel=0.01)
    assert float(ideal.Cp_ig("n_heptane", 298.15)) == pytest.approx(165.2, rel=0.01)


def test_n_heptane_ideal_gas_cp_rises_with_temperature():
    """A constant 166 was ~40% low at 600 K, where a hydrotreater runs."""
    ideal, _ = _cubic(["n_heptane"])
    assert float(ideal.Cp_ig("n_heptane", 600.0)) == pytest.approx(290.0, rel=0.03)


def test_cubic_enthalpy_uses_ig_cp_and_ignores_liquid_cp():
    ideal, thermo = _cubic(["water"])
    d = ideal.species["water"]
    T = 500.0
    # integral of the ideal-gas cubic from Tref
    a, b, c, e = d.Cp_vapor_coeffs
    T0 = d.Tref
    expect = (a * (T - T0) + b / 2 * (T**2 - T0**2) + c / 3 * (T**3 - T0**3)
              + e / 4 * (T**4 - T0**4))
    got = float(thermo.stream_enthalpy({"water": 1.0}, T))   # P=None: no departure
    assert got == pytest.approx(expect, rel=1e-9)
    assert got < 0.5 * 75.3 * (T - T0)    # not the liquid-Cp integral


def test_ig_entropy_integral_matches_ig_cp():
    ideal, _ = _cubic(["water"])
    d = ideal.species["water"]
    f = lambda T: ideal.S_ig_T("water", T)
    T = 400.0
    dS = float(jax.grad(f)(T))
    assert dS == pytest.approx(_poly(d.Cp_vapor_coeffs, T) / T, rel=1e-9)


def test_falls_back_to_cp_coeffs_when_no_ig_field():
    """User/importer data that stores the ideal-gas Cp in Cp_coeffs keeps working."""
    from difflow.thermo import SpeciesData

    sd = SpeciesData("X", 10.0, (30.0, 0.01, 0.0, 0.0), (1e4, 0.38, 500.0), (9.0, 1000.0, -50.0))
    ideal = IdealThermo({"X": sd})
    assert float(ideal.Cp_ig("X", 300.0)) == pytest.approx(33.0)


def test_species_without_liquid_fall_back_to_the_ig_cubic():
    for name in ("methane", "nitrogen", "oxygen", "ethylene", "hydrogen"):
        d = get_species_data(name)
        assert d.Cp_coeffs == d.Cp_vapor_coeffs
