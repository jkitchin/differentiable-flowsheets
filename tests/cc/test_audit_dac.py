"""Regression tests for the carbon-capture audit: DAC units (C1, C8)."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream
from difflow_cc.units.dac import (
    CO2_AMBIENT, DACParams, LiquidDACParams, LiquidSolventDAC, SolidSorbentDAC,
)


class TestSolidSorbentBoundedByFeed:
    """C1: capture cannot exceed the CO2 in the air processed."""

    def test_default_capture_not_above_air_co2(self):
        p = DACParams()
        _, info = SolidSorbentDAC(p)()
        n_air = p.air_velocity * p.cross_section * 101325.0 / (8.314 * 298.15)
        duty = p.cycle_time_ads / (p.cycle_time_ads + p.cycle_time_des)
        avail = p.n_units * n_air * duty * CO2_AMBIENT
        # Before the fix: 106.2 mol/s captured from 6.87 mol/s.
        assert float(info["CO2_captured_mol_s"]) <= avail
        assert float(info["CO2_captured_mol_s"]) == pytest.approx(
            float(info["capture_efficiency"]) * avail, rel=1e-12)

    def test_ambient_air_stream_is_used(self):
        dac = SolidSorbentDAC(DACParams())
        no_co2 = make_stream({"CO2": 0.0, "N2": 1.0}, T=298.15, P=101325.0)
        _, info = dac(no_co2)
        assert float(info["CO2_captured_mol_s"]) == 0.0
        air = make_stream({"CO2": 0.42, "N2": 999.58}, T=298.15, P=101325.0)
        _, info = dac(air)
        assert 0.0 < float(info["CO2_captured_mol_s"]) <= 0.42
        assert float(info["CO2_slip_mol_s"]) == pytest.approx(
            0.42 - float(info["CO2_captured_mol_s"]))

    def test_small_bed_limits_capture(self):
        """When the sorbent is the bottleneck, bed size matters."""
        small = SolidSorbentDAC(DACParams(bed_length=0.001))()[1]
        smaller = SolidSorbentDAC(DACParams(bed_length=0.0005))()[1]
        assert float(small["sorbent_utilization"]) == pytest.approx(1.0)
        assert float(smaller["CO2_captured_mol_s"]) == pytest.approx(
            0.5 * float(small["CO2_captured_mol_s"]), rel=1e-9)

    def test_desorption_colder_than_adsorption_rejected(self):
        with pytest.raises(ValueError, match="T_desorption"):
            DACParams(T_desorption=290.0)


class TestLiquidDACOperatingVariables:
    """C8: capture depends on L/G, depth and air velocity."""

    def _eff(self, **kw):
        return float(LiquidSolventDAC(LiquidDACParams(**kw))()[1]["capture_efficiency"])

    def test_no_liquid_no_capture(self):
        # Before the fix: 0.75 at L/G = 0.
        assert self._eff(L_G_ratio=0.0) == 0.0

    def test_default_design_point(self):
        assert self._eff() == pytest.approx(0.75, rel=1e-12)

    def test_trends(self):
        assert self._eff(contactor_height=12.0) > self._eff()
        assert self._eff(air_velocity=3.0) < self._eff()
        assert self._eff(L_G_ratio=0.2) < self._eff()

    def test_gradient(self):
        def cap(h):
            return LiquidSolventDAC(LiquidDACParams(contactor_height=h))()[1]["CO2_captured_mol_s"]
        g = jax.grad(cap)(8.0)
        fd = (cap(8.0 + 1e-4) - cap(8.0 - 1e-4)) / 2e-4
        assert float(g) > 0.0
        assert float(g) == pytest.approx(float(fd), rel=1e-6)
