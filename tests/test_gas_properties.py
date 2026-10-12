"""Tests for difflow.gas_properties against NIST and textbook values."""

import csv
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from difflow.gas_properties import (
    SUTHERLAND,
    GasRangeWarning,
    gas_density,
    gas_heat_capacity,
    gas_kinematic_viscosity,
    gas_prandtl,
    gas_property_range,
    gas_thermal_conductivity,
    gas_transport_properties,
    gas_viscosity,
    list_gas_species,
    sutherland_conductivity,
    sutherland_viscosity,
)

REF = Path(__file__).parent / "reference"
NIST_GASES = ["nitrogen", "oxygen", "argon", "carbon_dioxide", "carbon_monoxide",
              "methane", "hydrogen", "helium", "water"]
# Worst fit deviations from scripts/fit_gas_transport.py, with a small margin.
TOL = {"mu": 2e-3, "k": 5.5e-3, "cp": 1.5e-3}


def nist(gas):
    return np.loadtxt(REF / "nist_gases_1atm" / f"{gas}.csv", delimiter=",", comments="#", skiprows=2)


class TestAgainstNIST:
    @pytest.mark.parametrize("gas", NIST_GASES)
    def test_every_point(self, gas):
        d = nist(gas)
        T = jnp.asarray(d[:, 0])
        np.testing.assert_allclose(np.asarray(gas_viscosity(gas, T)), d[:, 2], rtol=TOL["mu"])
        np.testing.assert_allclose(np.asarray(gas_thermal_conductivity(gas, T)), d[:, 3], rtol=TOL["k"])
        np.testing.assert_allclose(np.asarray(gas_heat_capacity(gas, T)), d[:, 1] * 1e3, rtol=TOL["cp"])

    def test_oracles(self):
        # NIST WebBook at 300 K, 1 bar (the book's checks): N2 17.9, CH4 11.1 uPa s
        assert float(gas_viscosity("N2", 300.0)) == pytest.approx(17.9e-6, rel=3e-3)
        assert float(gas_viscosity("CH4", 300.0)) == pytest.approx(11.1e-6, rel=5e-3)
        # NIST 1 atm, 300 K: CO2 15.02 uPa s, H2 8.95 uPa s, O2 20.65 uPa s
        assert float(gas_viscosity("CO2", 300.0)) == pytest.approx(15.02e-6, rel=3e-3)
        assert float(gas_viscosity("H2", 300.0)) == pytest.approx(8.95e-6, rel=3e-3)
        assert float(gas_viscosity("O2", 300.0)) == pytest.approx(20.65e-6, rel=3e-3)


class TestAir:
    def test_incropera_table_a4_300K(self):
        # Incropera, Fundamentals of Heat and Mass Transfer, Table A.4, air at 300 K
        assert float(gas_viscosity("air", 300.0)) == pytest.approx(184.6e-7, rel=5e-3)
        assert float(gas_thermal_conductivity("air", 300.0)) == pytest.approx(26.3e-3, rel=5e-3)
        assert float(gas_heat_capacity("air", 300.0)) == pytest.approx(1007.0, rel=2e-3)
        assert float(gas_prandtl("air", 300.0)) == pytest.approx(0.707, rel=6e-3)

    def test_engineering_toolbox_table(self):
        with open(REF / "air_engineering_toolbox.csv") as f:
            rows = list(csv.reader(line for line in f if not line.startswith("#")))[1:]
        for prop, T, v in rows:
            fn = gas_viscosity if prop == "mu" else gas_thermal_conductivity
            assert float(fn("air", float(T))) == pytest.approx(float(v), rel=1e-3), (prop, T)

    def test_sutherland_white(self):
        # White's air constants: mu(273) = 1.716e-5 exactly; within 1 % of the table 250-500 K
        assert float(sutherland_viscosity(273.0, "air")) == pytest.approx(1.716e-5, rel=1e-12)
        for T in (250.0, 300.0, 400.0, 500.0):
            assert float(sutherland_viscosity(T, "air")) == pytest.approx(float(gas_viscosity("air", T)), rel=1e-2)
        assert float(sutherland_conductivity(300.0, "nitrogen")) == pytest.approx(
            float(gas_thermal_conductivity("N2", 300.0)), rel=3e-2)
        assert float(sutherland_viscosity(300.0, mu0=1.716e-5, S=111.0)) == pytest.approx(
            float(sutherland_viscosity(300.0, "air")))
        assert set(SUTHERLAND) >= {"air", "nitrogen", "oxygen", "argon"}


class TestAPI:
    def test_aliases_and_errors(self):
        assert float(gas_viscosity("steam", 500.0)) == float(gas_viscosity("water", 500.0))
        assert float(gas_viscosity("H2O", 500.0)) == float(gas_viscosity("water", 500.0))
        assert "air" in list_gas_species()
        with pytest.raises(KeyError, match="sutherland"):
            gas_viscosity("unobtainium", 300.0)

    def test_range_warning_concrete_only(self):
        lo, hi = gas_property_range("methane", "mu")
        assert (lo, hi) == (200.0, 620.0)
        with pytest.warns(GasRangeWarning):
            gas_viscosity("methane", 700.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gas_viscosity("methane", 400.0)
            jax.grad(lambda T: gas_viscosity("methane", T))(700.0)

    def test_density_and_bundle(self):
        rho = float(gas_density("air", 300.0))
        assert rho == pytest.approx(101325 * 28.9585e-3 / (8.314462618 * 300.0), rel=1e-12)
        p = gas_transport_properties("air", 300.0)
        assert float(p["nu"]) == pytest.approx(float(gas_kinematic_viscosity("air", 300.0)))
        assert float(p["Pr"]) == pytest.approx(float(gas_prandtl("air", 300.0)))
        assert float(p["alpha"]) == pytest.approx(float(p["k"] / (p["rho"] * p["cp"])))

    @pytest.mark.slow
    def test_vectorized_jit_and_gradients(self):
        T = jnp.linspace(300.0, 600.0, 4)
        assert jax.jit(lambda t: gas_viscosity("air", t))(T).shape == (4,)
        for gas in ("air", "methane", "carbon_dioxide", "water"):
            T0 = 500.0
            for fn in (gas_viscosity, gas_thermal_conductivity, gas_heat_capacity, gas_prandtl):
                check_grads(lambda t: fn(gas, t), (T0,), order=2, modes=["fwd", "rev"], rtol=1e-5)
        # gas viscosity rises with T (unlike a liquid's)
        assert float(jax.grad(lambda t: gas_viscosity("air", t))(300.0)) > 0
