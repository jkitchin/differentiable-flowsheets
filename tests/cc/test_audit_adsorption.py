"""Regression tests for the carbon-capture audit: adsorption units (C6, (b))."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import get_flows, make_stream, total_flow
from difflow_cc import AdsorptionParams, PSAUnit, TSAUnit, TVSAUnit, VSAUnit

UNITS = [PSAUnit, VSAUnit, TSAUnit, TVSAUnit]
WET = {"CO2": 1.5, "N2": 7.5, "O2": 0.5, "H2O": 0.5}


@pytest.mark.parametrize("unit", UNITS)
def test_every_species_conserved(unit):
    """(b): non-CO2/N2 species lost 1 %; TVSA lost 1 % of its N2."""
    feed = make_stream(WET, T=298.15, P=101325.0)
    prod, off, _ = unit(AdsorptionParams(adsorbent="Zeolite_13X", bed_mass=100.0))(feed)
    fp, fo = get_flows(prod), get_flows(off)
    for sp, f in WET.items():
        assert float(fp.get(sp, 0.0) + fo.get(sp, 0.0)) == pytest.approx(f, abs=1e-12), sp
        assert float(fo.get(sp, 0.0)) >= 0.0


@pytest.mark.parametrize("unit", UNITS)
def test_reported_purity_is_product_purity(unit):
    """C6: PSA reported 0.9928 for a 0.9841 product."""
    feed = make_stream({"CO2": 1.0, "N2": 4.0}, T=298.15, P=5e5)
    prod, _, info = unit(AdsorptionParams(adsorbent="Zeolite_13X",
                                          P_adsorption=5e5, P_desorption=1e5))(feed)
    assert float(info["purity"]) == pytest.approx(
        float(get_flows(prod)["CO2"] / total_flow(prod)), rel=1e-12)


@pytest.mark.parametrize("unit", [VSAUnit, TSAUnit])
def test_no_negative_n2_with_n2_lean_feed(unit):
    feed = make_stream({"CO2": 5.0, "N2": 1e-4, "H2O": 0.1}, T=298.15, P=101325.0)
    _, off, _ = unit(AdsorptionParams(adsorbent="Zeolite_13X"))(feed)
    assert all(float(v) >= 0.0 for v in get_flows(off).values())


@pytest.mark.parametrize("unit", UNITS)
def test_operating_variables_move_recovery(unit):
    """C6: recovery sat on the 0.95 cap; bed mass/cycle time/n_beds inert."""
    feed = make_stream({"CO2": 1.5, "N2": 8.5}, T=298.15, P=101325.0)

    def rec(**kw):
        return float(unit(AdsorptionParams(adsorbent="Zeolite_13X", **kw))(feed)[2]["recovery"])

    base = rec()
    assert base < 0.95
    assert rec(bed_mass=2000.0) > base
    assert rec(n_beds=4) > base
    assert rec(t_adsorption=600.0) < base

    g = jax.grad(lambda m: unit(AdsorptionParams(adsorbent="Zeolite_13X", bed_mass=m))(feed)[2]["recovery"])(1000.0)
    assert float(g) > 0.0


def test_tsa_cold_desorption_flagged_not_nan():
    """C6: T_des < T_ads gave NaN product purity while info said 0.998."""
    feed = make_stream({"CO2": 0.5, "N2": 9.5}, T=298.15, P=101325.0)
    prod, _, info = TSAUnit(AdsorptionParams(adsorbent="PEI_Silica", T_desorption=280.0))(feed)
    assert not bool(info["feasible"])
    assert float(info["purity"]) == 0.0
    assert float(info["CO2_captured"]) == 0.0
    assert jnp.isfinite(info["heating_power"]) and float(info["heating_power"]) >= 0.0


def test_vsa_dilute_feed_depends_on_vacuum():
    """C6: P_CO2_des = 0.3 P_des made a 400 ppm feed uncapturable at any vacuum."""
    feed = make_stream({"CO2": 0.0004, "N2": 0.9996}, T=298.15, P=101325.0)

    def run(P_des):
        return VSAUnit(AdsorptionParams(adsorbent="Zeolite_13X", P_desorption=P_des))(feed)[2]

    shallow, deep = run(1e4), run(1e3)
    assert not bool(shallow["feasible"])
    assert bool(deep["feasible"])
    assert float(deep["CO2_captured"]) > 0.0
