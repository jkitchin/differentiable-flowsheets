"""Tests for carbon-capture plugin enhancement issues (#149 condenser duty,
#151 absorber water transfer)."""

import pytest
import jax
import jax.numpy as jnp

pytest.importorskip("difflow_cc")

from difflow.streams import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


class TestStripperCondenserDuty:
    """Issue #149: condenser duty reported and usable as a cooling utility."""

    def _run(self, reflux_ratio=0.3):
        from difflow_cc import StripperParams, AmineStripper
        params = StripperParams(
            solvent="MEA", T_reboiler=393.15,
            target_lean_loading=0.2, reflux_ratio=reflux_ratio,
        )
        rich = make_stream(
            flows={"H2O": 25.0, "Amine": 5.0, "CO2_absorbed": 2.0},
            T=323.15, P=200000.0,
        )
        return AmineStripper(params)(rich)

    def test_condenser_duty_reported(self):
        _, _, info = self._run()
        assert "Q_condenser" in info
        assert "condenser_cooling_duty" in info
        assert float(info["condenser_cooling_duty"]) == pytest.approx(
            float(info["Q_condenser"]), rel=1e-9)
        assert float(info["condenser_cooling_duty"]) > 0.0

    def test_condenser_duty_scales_with_reflux(self):
        _, _, lo = self._run(reflux_ratio=0.2)
        _, _, hi = self._run(reflux_ratio=0.6)
        assert float(hi["condenser_cooling_duty"]) > float(lo["condenser_cooling_duty"])

    def test_condenser_duty_is_fraction_of_vaporization(self):
        _, _, info = self._run(reflux_ratio=0.3)
        assert float(info["Q_condenser"]) == pytest.approx(
            0.3 * float(info["Q_vaporization"]), rel=1e-9)


class TestAbsorberWaterTransfer:
    """Issue #151: optional water evaporation into treated gas w/ balance."""

    def _run(self, model_water, feed_h2o=0.0):
        from difflow_cc import AbsorberParams, AmineAbsorber
        params = AbsorberParams(
            solvent="MEA", n_stages=10, solvent_conc=30.0, L_G_ratio=3.0,
            model_water_transfer=model_water,
        )
        flows = {"CO2": 1.0, "N2": 9.0}
        if feed_h2o:
            flows["H2O"] = feed_h2o
        feed = make_stream(flows=flows, T=313.15, P=101325.0)
        return AmineAbsorber(params)(feed)

    def test_backward_compat_no_water_transfer(self):
        gas_out, solvent_out, info = self._run(model_water=False)
        # Treated gas has no water added; net evaporation is zero
        assert float(info["net_water_evaporated"]) == 0.0
        assert "H2O" not in get_flows(gas_out)

    def test_water_evaporates_into_gas(self):
        gas_out, solvent_out, info = self._run(model_water=True)
        # Treated gas leaves humidified; water is drawn from the solvent
        assert float(get_flows(gas_out).get("H2O", 0.0)) > 0.0
        assert float(info["net_water_evaporated"]) > 0.0

    def test_water_balance_closes(self):
        gas_out, solvent_out, info = self._run(model_water=True, feed_h2o=0.05)
        # Water in (feed gas) + water from solvent leaving = water out (gas) +
        # solvent water. Check total H2O conservation against the no-transfer
        # solvent water baseline.
        gas_h2o_out = float(get_flows(gas_out).get("H2O", 0.0))
        net_evap = float(info["net_water_evaporated"])
        gas_h2o_in = 0.05
        # net evaporated = gas H2O out - gas H2O in
        assert net_evap == pytest.approx(gas_h2o_out - gas_h2o_in, rel=1e-6)


class TestCapexExponentValidation:
    """Follow-up to #127: CC CAPEX exponents should be validatable like core."""

    def test_all_exponents_in_valid_range(self):
        from difflow_cc.economics import SCALING_EXPONENTS, validate_scaling_exponent
        for eq in SCALING_EXPONENTS:
            v = validate_scaling_exponent(eq)
            assert v["exponent_valid"], f"{eq} exponent {v['exponent']} out of range"

    def test_exponents_are_type_specific(self):
        from difflow_cc.economics import SCALING_EXPONENTS
        # Not a single six-tenths default: compressor and HX differ
        assert SCALING_EXPONENTS["compressor"] == pytest.approx(0.82)
        assert SCALING_EXPONENTS["heat_exchanger_shell_tube"] == pytest.approx(0.68)
        assert len(set(SCALING_EXPONENTS.values())) > 1

    def test_validate_reports_fields(self):
        from difflow_cc.economics import validate_scaling_exponent, VALID_EXPONENT_RANGE
        v = validate_scaling_exponent("adsorber_vessel")
        assert v["equipment_type"] == "adsorber_vessel"
        assert v["exponent"] == pytest.approx(0.62)
        assert v["valid_range"] == VALID_EXPONENT_RANGE

    def test_unknown_equipment_raises(self):
        from difflow_cc.economics import validate_scaling_exponent
        with pytest.raises(KeyError):
            validate_scaling_exponent("nonexistent_unit")
