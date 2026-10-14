"""Smoke tests for difflow.units.column_design (#429)."""

import warnings

import jax
import numpy as np
import pytest

from difflow.mass_transfer import PACKINGS
from difflow.units import column_design as cd
from difflow.units.column_design import (
    column_hardware_cost, fair_capacity_factor, flash_drum_size, flooding_velocity,
    fractional_entrainment, gpdc_flood, gpdc_pressure_drop, hetp, kister_gill_flood_dp,
    oconnell_efficiency, packed_column_diameter, size_column, tray_column_diameter,
    tray_pressure_drop, weep_point, weir_crest,
)

jax.config.update("jax_enable_x64", True)

TRAY = dict(V_molar=100.0, MW_V=80.0, rho_V=2.9, L_molar=80.0, MW_L=85.0, rho_L=790.0, sigma=0.02)
HYD = dict(V_vol=2.76, L_vol=0.0086, rho_V=2.9, rho_L=790.0, diameter=1.67)
PACKED = dict(V_molar=100.0, MW_V=29.0, rho_V=1.2, L_molar=500.0, MW_L=18.0, rho_L=1000.0, mu_L=1e-3)


def _finite(*vals):
    return all(np.isfinite(float(v)) for v in vals)


def test_tray_correlations_smoke():
    D, info = tray_column_diameter(**TRAY, tray_spacing=0.6, fraction_flood=0.8)
    dP, i = tray_pressure_drop(**HYD)
    margin, _ = weep_point(V_vol=2.76, L_vol=0.0086, rho_V=2.9, diameter=1.67)
    dc, _ = cd.downcomer_backup(**HYD, tray_spacing=0.6)
    assert _finite(D, info["F_LV"], dP, margin, dc, fair_capacity_factor(0.1, 0.6, 0.02),
                   flooding_velocity(0.09, 790.0, 2.9), weir_crest(0.01, 1.0),
                   oconnell_efficiency(1.0, 1.0))


def test_weep_K2_callable():
    m, i = weep_point(2.0, 0.0086, 2.9, 1.67, K2=lambda h: 29.0 + 0.02 * h)
    assert _finite(m, i["K2"])


def test_entrainment_placeholder_warns():
    with pytest.warns(cd.EntrainmentPlaceholderWarning):
        fractional_entrainment(0.1, 80.0)


def test_packed_correlations_smoke():
    D, i = packed_column_diameter(**PACKED, packing=PACKINGS["pall_metal_25"])
    Dr, _ = packed_column_diameter(**PACKED, packing=PACKINGS["pall_metal_25"], flood="robbins")
    assert _finite(D, Dr, i["dP_per_m"], gpdc_flood(0.1, 100.0, 1000.0, 1.2, 1e-3),
                   kister_gill_flood_dp(183.0), gpdc_pressure_drop(1.5, 8.0, 998.0, 1.2, 1e-3, 150.0),
                   cd.robbins_flood(5.0, 1.2, 998.0, 1e-3, 183.0, 174.0),
                   hetp("random", d_p=0.025), hetp("structured", a_p=250.0))


def test_packed_validation():
    with pytest.raises(ValueError):
        gpdc_pressure_drop(1.0, 1.0, 1000.0, 1.2, 1e-3, 183.0, method="nope")
    with pytest.raises(ValueError):
        packed_column_diameter(**PACKED, packing=PACKINGS["pall_metal_25"], flood="nope")
    with pytest.raises(ValueError):
        packed_column_diameter(**PACKED)
    with pytest.raises(ValueError):
        hetp("nope")


def test_structured_packing_warns_no_capacity_model():
    with pytest.warns(cd.PackingCapacityWarning):
        packed_column_diameter(**PACKED, packing=PACKINGS["mellapak_250y"], flood="robbins")
    with warnings.catch_warnings():
        warnings.simplefilter("error", cd.PackingCapacityWarning)
        packed_column_diameter(**PACKED, packing=PACKINGS["mellapak_250y"], F_pd=66.0)


def test_size_column_and_cost_smoke():
    s = size_column(20, 0.5, 1.5, tray_spacing=0.6, extra_height=3.0)
    p = size_column(None, None, 1.5, packed_height=6.0, extra_height=2.0)
    assert _finite(column_hardware_cost(s)["total"],
                   column_hardware_cost(p, "packing_random", material="stainless")["total"])


def test_size_column_and_cost_validation():
    s = size_column(20, 0.5, 1.5, tray_spacing=0.6)
    for bad in (dict(), dict(tray_spacing=0.6, hetp=0.5), dict(hetp=0.5, packed_height=6.0)):
        with pytest.raises(ValueError):
            size_column(10, 1.0, 1.5, **bad)
    with pytest.raises(ValueError):
        column_hardware_cost(s, material="unobtainium")
    with pytest.raises(ValueError):
        column_hardware_cost(s, "nope")


def test_cost_grad_finite():
    def cost(ff):
        D, _ = tray_column_diameter(**TRAY, fraction_flood=ff)
        return column_hardware_cost(size_column(20.0, 0.6, D, tray_spacing=0.6))["total"]

    assert np.isfinite(float(jax.grad(cost)(0.8)))


def test_flash_drum_size_smoke_and_grad():
    D, info = flash_drum_size(1.0, 0.001, 550.0, 15.0)
    assert _finite(D, info["height"], info["volume"])
    g = jax.grad(lambda q: flash_drum_size(q, 0.001, 550.0, 15.0)[0])(1.0)
    assert np.isfinite(float(g))
