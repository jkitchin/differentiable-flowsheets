"""Pure-component saturation temperature/pressure helpers (issue #442, Ch. 2).

References: reference-EOS saturation temperatures (the equations of state
the NIST WebBook fluid tables use; IAPWS-95 for water), via CoolProp 8.0.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

from difflow import (IdealThermo, PengRobinson, saturation_pressure,
                     saturation_temperature)
from difflow.database import get_critical_props, get_species_data
from difflow.thermo import AntoineRangeWarning

jax.config.update("jax_enable_x64", True)

#: Reference-EOS T_sat (K) and dT_sat/dP (K/Pa) at 1 MPa.
REF_1MPA = {"methane": (149.1388, 2.1859e-5), "propane": (300.0923, 3.9809e-5),
            "water": (453.0280, 4.3453e-5)}


# ------------------------------------------------------------------ ideal


def test_ideal_normal_boiling_point_of_water():
    T = float(saturation_temperature("water", 101325.0))
    assert T == pytest.approx(373.124, abs=0.1)


def test_ideal_inverts_psat():
    th = IdealThermo({s: get_species_data(s) for s in ("benzene", "toluene")})
    for s, P in (("benzene", 5e4), ("toluene", 1.5e5)):
        T = saturation_temperature(s, P, thermo=th)
        assert float(th.Psat(s, T)) == pytest.approx(P, rel=1e-10)
        assert float(saturation_pressure(s, T, thermo=th)) == pytest.approx(P, rel=1e-10)


def test_ideal_wagner_thermo_is_used():
    th = IdealThermo({"propane": get_species_data("propane", vapor_pressure="wagner")})
    T = saturation_temperature("propane", 1e6, thermo=th)
    assert float(th.Psat("propane", T)) == pytest.approx(1e6, rel=1e-9)
    assert float(T) == pytest.approx(REF_1MPA["propane"][0], abs=0.15)


@pytest.mark.parametrize("vp", ["antoine", "wagner"])
def test_ideal_gradient_is_clausius_clapeyron(vp):
    th = IdealThermo({"methane": get_species_data("methane", vapor_pressure=vp)})
    f = lambda P: saturation_temperature("methane", P, thermo=th)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AntoineRangeWarning)
        check_grads(f, (1e6,), order=1, modes=["rev", "fwd"], eps=10.0)
        g = float(jax.grad(f)(1e6))
    # dT/dP > 0 and equals 1 / (dPsat/dT) at the saturation point
    T = f(1e6)
    dPdT = float(jax.grad(lambda t: th.Psat("methane", t))(T))
    assert g > 0.0
    assert g == pytest.approx(1.0 / dPdT, rel=1e-8)
    tol = 0.03 if vp == "antoine" else 0.01
    assert g == pytest.approx(REF_1MPA["methane"][1], rel=tol)


def test_ideal_jit_and_vmap():
    f = jax.jit(jax.vmap(lambda P: saturation_temperature("water", P)))
    T = f(jnp.array([5e4, 101325.0, 2e5]))
    assert jnp.all(jnp.diff(T) > 0)


def test_ideal_range_warning():
    with pytest.warns(AntoineRangeWarning):
        saturation_temperature("water", 1e7)   # 584 K, far above the fit


def test_vaporization_T_is_not_the_saturation_temperature():
    """vaporization_T is the enthalpy-path temperature (<= 298.15 K)."""
    th = IdealThermo({"toluene": get_species_data("toluene")})
    assert th.vaporization_T("toluene") == pytest.approx(298.15)
    assert float(saturation_temperature("toluene", 101325.0, thermo=th)) == pytest.approx(
        383.75, abs=0.1)


# -------------------------------------------------------------------- EOS


@pytest.mark.slow  # ~75 s: three EOS saturation solves plus a reverse pass
def test_eos_saturation_temperature_and_gradient():
    """Peng-Robinson T_sat of pure methane at 1 MPa: near the reference
    (PR's own saturation curve, within 0.3 K), with dT/dP from implicit
    differentiation within 1% of the reference slope."""
    eos = PengRobinson({"methane": get_critical_props("methane")})
    f = lambda P: saturation_temperature("methane", P, model="eos", eos=eos)
    T_ref, dTdP_ref = REF_1MPA["methane"]
    assert float(f(1e6)) == pytest.approx(T_ref, abs=0.3)
    g = float(jax.grad(f)(1e6))
    assert g > 0.0
    assert g == pytest.approx(dTdP_ref, rel=0.01)
    h = 2e3
    fd = (float(f(1e6 + h)) - float(f(1e6 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=1e-4)


def test_eos_species_in_a_mixture_eos_and_pressure_roundtrip():
    eos = PengRobinson({s: get_critical_props(s) for s in ("methane", "propane")})
    T = saturation_temperature("propane", 1e6, model="eos", eos=eos)
    assert float(T) == pytest.approx(REF_1MPA["propane"][0], abs=0.5)
    P = saturation_pressure("propane", T, model="eos", eos=eos)
    assert float(P) == pytest.approx(1e6, rel=1e-6)


def test_bad_model_and_species():
    with pytest.raises(ValueError):
        saturation_temperature("water", 1e5, model="raoult")
    th = IdealThermo({"water": get_species_data("water")})
    with pytest.raises(KeyError):
        saturation_temperature("benzene", 1e5, thermo=th)
    eos = PengRobinson({"water": get_critical_props("water")})
    with pytest.raises(KeyError):
        saturation_pressure("benzene", 350.0, model="eos", eos=eos)
