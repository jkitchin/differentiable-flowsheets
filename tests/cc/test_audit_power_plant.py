"""Regression tests for the carbon-capture audit (d): power-plant bases."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow_cc.integration.power_plant import (
    PowerPlantIntegration, PowerPlantParams, flue_gas_flow_rate,
)


def test_zero_capture_load_zero_penalty():
    """(d): zero steam and compression still reported 0.372 vs base 0.400."""
    p = PowerPlantParams()
    r = PowerPlantIntegration(p).analyze(steam_duty=0.0, compression_power=0.0)
    assert float(r["total_penalty_MW"]) == 0.0
    assert float(r["efficiency_with_capture"]) == pytest.approx(p.net_efficiency, rel=1e-12)
    assert float(r["efficiency_points_lost"]) == pytest.approx(0.0, abs=1e-15)


def test_flue_gas_on_net_basis():
    """Flue gas follows fuel = gross (1 - aux) / net efficiency."""
    p = PowerPlantParams()
    fuel_W = p.gross_power * 1e6 * (1 - p.auxiliary_fraction) / p.net_efficiency
    co2 = fuel_W / (p.fuel_heating_value * 1e6) * p.fuel_carbon_content / 0.012
    assert float(flue_gas_flow_rate(p)) == pytest.approx(co2 / p.flue_gas_CO2_fraction, rel=1e-12)


def test_penalty_reduces_efficiency_linearly():
    p = PowerPlantParams()
    r = PowerPlantIntegration(p).analyze(steam_duty=0.0, compression_power=30e6)
    fuel = p.gross_power * (1 - p.auxiliary_fraction) / p.net_efficiency
    assert float(r["efficiency_points_lost"]) == pytest.approx(30.0 / fuel, rel=1e-12)
