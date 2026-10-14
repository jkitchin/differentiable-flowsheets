"""Tests for the PFR and GasPFR unit operations: smoke, API and validation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

jax.config.update("jax_enable_x64", True)

from difflow.units.pfr import PFR, PFRParams, GasPFR, GasPFRParams, pfr_conversion_analytical
from difflow.streams import make_stream, get_flows
from difflow.thermo import IdealThermo, SpeciesData

SPECIES = ["A", "B"]
STOICH = jnp.array([[-1.0], [1.0]])
RATE_PARAMS = {"A": jnp.array(1e8), "Ea": jnp.array(50000.0), "k": 0.1}


def rate_fn(C, T, p):
    return jnp.array([p["A"] * jnp.exp(-p["Ea"] / (8.314 * T)) * C["A"]])


@pytest.fixture
def thermo():
    return IdealThermo({
        "A": SpeciesData(name="A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(35000.0, 0.38, 500.0),
                         antoine_coeffs=(10.0, 3000.0, -50.0), Hf=0.0),
        "B": SpeciesData(name="B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(30000.0, 0.38, 450.0),
                         antoine_coeffs=(10.0, 2800.0, -40.0), Hf=-10000.0),
    })


def feed():
    return make_stream({"A": 10.0, "B": 0.0}, T=350.0, P=101325.0)


def pfr_params(**kw):
    base = dict(V=1.0, rate_fn=rate_fn, stoich=STOICH, rate_params=RATE_PARAMS,
                species_order=SPECIES)
    base.update(kw)
    return PFRParams(**base)


def gas_params(**kw):
    base = dict(V=1.0, rate_fn=rate_fn, stoich=STOICH, rate_params=RATE_PARAMS,
                species_order=SPECIES)
    base.update(kw)
    return GasPFRParams(**base)


def _total(s):
    f = get_flows(s)
    return float(f["A"] + f["B"])


def test_params_validation(thermo):
    with pytest.raises(ValueError, match="Stoichiometry matrix has 3 rows"):
        pfr_params(stoich=jnp.array([[-1.0], [1.0], [0.0]]))
    with pytest.raises(ValueError, match="dH_rxn has 2 values"):
        pfr_params(dH_rxn=jnp.array([-1e4, -5e3]))
    with pytest.raises(ValueError, match="Thermo object required"):
        PFR(pfr_params(dH_rxn=jnp.array([-1e4])), thermo=None, mode="adiabatic")
    with pytest.raises(ValueError, match="dH_rxn required"):
        PFR(pfr_params(), thermo=thermo, mode="adiabatic")


def test_pfr_isothermal_smoke(thermo):
    pfr = PFR(pfr_params(n_save_points=11), thermo=thermo, mode="isothermal")
    outlet, info = pfr(feed(), volumetric_flow=0.1, T_spec=350.0)
    npt.assert_allclose(_total(outlet), _total(feed()), rtol=1e-6)
    assert np.isfinite(float(info["conversion"]["A"]))
    assert info["profiles"]["F"].shape == (11, 2)


def test_pfr_adiabatic_smoke(thermo):
    pfr = PFR(pfr_params(V=0.001, dH_rxn=jnp.array([-50000.0])), thermo=thermo, mode="adiabatic")
    outlet, _ = pfr(feed(), volumetric_flow=0.1)
    assert np.isfinite(float(outlet["T"]))
    npt.assert_allclose(_total(outlet), _total(feed()), rtol=1e-5)


def test_gas_pfr_smoke(thermo):
    pfr = GasPFR(gas_params(V=0.001, dH_rxn=jnp.array([-50000.0]), alpha=5000.0),
                 thermo=thermo, mode="adiabatic")
    outlet, info = pfr(feed())
    assert np.isfinite(float(outlet["T"])) and np.isfinite(float(info["pressure_drop"]))
    npt.assert_allclose(_total(outlet), _total(feed()), rtol=1e-4)


def test_grad_finite(thermo):
    def f(V):
        pfr = PFR(pfr_params(V=V), thermo=thermo, mode="isothermal")
        return pfr(feed(), volumetric_flow=0.1, T_spec=350.0)[0]["F_B"]

    assert np.isfinite(float(jax.grad(f)(1.0)))


def test_analytical_conversion():
    assert np.isfinite(float(pfr_conversion_analytical(0.1, 10.0, order=1)))
    g = jax.grad(lambda k: pfr_conversion_analytical(k, 10.0, order=2, C_A0=2.0))(0.1)
    assert np.isfinite(float(g))
    with pytest.raises(ValueError, match="Unsupported reaction order"):
        pfr_conversion_analytical(0.1, 10.0, order=3)


def test_dynamic_interface_and_initialize(thermo):
    iso = PFR(pfr_params(), thermo=thermo, mode="isothermal")
    assert [v.name for v in iso.state_spec().variables] == ["F_out_A", "F_out_B"]
    npt.assert_array_equal(iso.derivatives(0.0, jnp.array([5.0, 5.0]), {"inlet": feed()}),
                           jnp.zeros(2))
    result = iso.initialize(feed(), volumetric_flow=0.1)
    assert {"outlet", "states", "info"} <= set(result)
    adi = PFR(pfr_params(dH_rxn=jnp.array([-50000.0])), thermo=thermo, mode="adiabatic")
    assert adi.state_spec().variables[2].name == "T_out"
    assert "T_out" in adi.initialize(feed(), volumetric_flow=0.1)["states"]
