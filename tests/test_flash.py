"""Smoke tests for Flash, EOSFlash, PHFlash, Splitter and Mixer."""

import pytest
import jax
import jax.numpy as jnp
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from difflow.units.flash import (
    Flash, FlashParams, Splitter, Mixer,
    EOSFlash, EOSFlashParams, PHFlash,
)
from difflow.streams import make_stream, get_flows, total_flow
from difflow.thermo import IdealThermo, SpeciesData
from difflow.eos import PengRobinson, CriticalProperties


@pytest.fixture
def binary_thermo():
    species_data = {
        "Light": SpeciesData(
            name="Light", MW=72.0, Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(26000.0, 0.38, 470.0),
            antoine_coeffs=(10.422, 1687.537, -38.44), Hf=0.0,
        ),
        "Heavy": SpeciesData(
            name="Heavy", MW=114.0, Cp_coeffs=(190.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 570.0),
            antoine_coeffs=(10.186, 2004.68, -60.53), Hf=0.0,
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def flash_params():
    return FlashParams(species_order=["Light", "Heavy"])


@pytest.fixture
def binary_feed():
    return make_stream({"Light": 50.0, "Heavy": 50.0}, T=350.0, P=30000.0)


def test_flash_params_validation():
    with pytest.raises(ValueError, match="cannot be empty"):
        FlashParams(species_order=[])
    with pytest.raises(ValueError, match="duplicate"):
        FlashParams(species_order=["A", "B", "A"])


def test_flash_smoke(binary_thermo, flash_params, binary_feed):
    flash = Flash(flash_params, binary_thermo)
    liquid, vapor, info = flash(binary_feed)
    assert jnp.isfinite(info["V_frac"])
    npt.assert_allclose(total_flow(liquid) + total_flow(vapor),
                        total_flow(binary_feed), rtol=1e-8)


def test_flash_bubble_dew_and_initialize(binary_thermo, flash_params, binary_feed):
    flash = Flash(flash_params, binary_thermo)
    for v in (flash.bubble_point_pressure(binary_feed),
              flash.dew_point_pressure(binary_feed),
              flash.bubble_point_temperature(binary_feed),
              flash.dew_point_temperature(binary_feed)):
        assert jnp.isfinite(v)
    result = flash.initialize(binary_feed)
    assert {"liquid", "vapor", "states", "info"} <= set(result)


def test_flash_grad_finite(binary_thermo, flash_params):
    flash = Flash(flash_params, binary_thermo)

    def vfrac(T):
        return flash(make_stream({"Light": 50.0, "Heavy": 50.0}, T=T, P=30000.0))[2]["V_frac"]

    assert jnp.isfinite(jax.grad(vfrac)(350.0))


def test_splitter():
    feed = make_stream({"A": 100.0, "B": 50.0}, T=300.0, P=101325.0)
    splitter = Splitter(species_order=["A", "B"])
    out1, out2, _ = splitter(feed, split_frac=0.3)
    npt.assert_allclose(get_flows(out1)["A"] + get_flows(out2)["A"], 100.0, rtol=1e-10)
    *outs, info = splitter(feed, split_frac=[0.5, 0.3, 0.2])
    assert len(outs) == 3 and info["n_outlets"] == 3

    g = jax.grad(lambda s: total_flow(splitter(feed, split_frac=s)[0]))(0.5)
    assert jnp.isfinite(g)


def test_mixer(binary_thermo):
    s1 = make_stream({"Light": 30.0, "Heavy": 20.0}, T=300.0, P=101325.0)
    s2 = make_stream({"Light": 20.0, "Heavy": 30.0}, T=350.0, P=101325.0)
    outlet, info = Mixer(species_order=["Light", "Heavy"], thermo=binary_thermo)(s1, s2)
    npt.assert_allclose(total_flow(outlet), 100.0, rtol=1e-10)
    assert jnp.isfinite(outlet["T"]) and info["n_inlets"] == 2
    with pytest.raises(ValueError, match="At least one inlet"):
        Mixer(species_order=["A"])()


def test_eos_flash_smoke():
    eos = PengRobinson({
        "methane": CriticalProperties("methane", 190.6, 4.6e6, 0.011, 16.04),
        "propane": CriticalProperties("propane", 369.8, 4.2e6, 0.152, 44.10),
    })
    feed = make_stream({"methane": 50.0, "propane": 50.0}, T=250.0, P=2.0e6)
    flash = EOSFlash(EOSFlashParams(species_order=["methane", "propane"]), eos)
    liquid, vapor, info = flash(feed)
    assert jnp.isfinite(info["V_frac"])
    npt.assert_allclose(total_flow(liquid) + total_flow(vapor), total_flow(feed), rtol=1e-6)


def test_eos_flash_params_validation():
    with pytest.raises(ValueError, match="eos_type must be"):
        EOSFlashParams(species_order=["A"], eos_type="INVALID")


def test_ph_flash_smoke(binary_thermo, flash_params):
    ph = PHFlash(flash_params, binary_thermo)
    feed = make_stream({"Light": 50.0, "Heavy": 50.0}, T=380.0, P=101325.0)
    liquid, vapor, info = ph(feed, P=30000.0)
    assert {"T_flash", "H_inlet", "T_inlet", "converged"} <= set(info)
    assert jnp.isfinite(info["T_flash"])
    npt.assert_allclose(total_flow(liquid) + total_flow(vapor), total_flow(feed), rtol=1e-6)
    with pytest.raises(ValueError, match="inlet_phase"):
        ph(feed, inlet_phase="gas")


def test_ph_flash_pure_letdown_not_clipped():
    """Regression: saturated water let down used to return V_frac = 0 at the clip."""
    from difflow.database import get_species_data
    th = IdealThermo({"water": get_species_data("water")})
    ph = PHFlash(FlashParams(species_order=["water"]), th)
    _, _, info = ph(make_stream({"water": 1.0}, T=453.15, P=12e5), P=1.5e5)
    assert 0.0 < float(info["V_frac"]) < 1.0
