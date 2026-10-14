"""Regression tests for the carbon-capture audit: C12 and (c)."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream, total_flow
from difflow_cc import (
    CompressionTrain, CompressionTrainParams, LeanRichExchanger, LeanRichExchangerParams,
)
from difflow_cc.units.heat_integration import HeatRecoverySystem, HeatRecoverySystemParams


def _duties(lean_in, rich_in, lean_out, rich_out, Cp=75.0):
    q_lean = float(total_flow(lean_in)) * Cp * (float(lean_in["T"]) - float(lean_out["T"]))
    q_rich = float(total_flow(rich_in)) * Cp * (float(rich_out["T"]) - float(rich_in["T"]))
    return q_lean, q_rich


@pytest.mark.parametrize("eff, F_lean, F_rich", [(0.95, 50.0, 100.0), (0.85, 200.0, 100.0),
                                                 (0.95, 100.0, 50.0), (0.5, 100.0, 100.0)])
def test_lean_rich_energy_balance(eff, F_lean, F_rich):
    """C12/(c): 262.5 vs 285 kW at eff 0.95; 45 kW lost at 200/100 mol/s."""
    lean = make_stream({"H2O": F_lean}, T=393.15, P=2e5)
    rich = make_stream({"H2O": F_rich}, T=313.15, P=2e5)
    lo, ro, info = LeanRichExchanger(LeanRichExchangerParams(effectiveness=eff))(lean, rich)
    q_lean, q_rich = _duties(lean, rich, lo, ro)
    assert q_lean == pytest.approx(q_rich, rel=1e-12)
    assert float(info["Q"]) == pytest.approx(q_lean, rel=1e-12)
    # Reported effectiveness is the achieved one.
    C_min = 75.0 * min(F_lean, F_rich)
    assert float(info["effectiveness"]) == pytest.approx(q_lean / (C_min * 80.0), rel=1e-12)
    # Minimum approach honoured at both ends.
    assert float(ro["T"]) <= 393.15 - 10.0 + 1e-9
    assert float(lo["T"]) >= 313.15 + 10.0 - 1e-9


def test_heat_recovery_system_balances():
    lean = make_stream({"H2O": 200.0}, T=393.15, P=2e5)
    rich = make_stream({"H2O": 100.0}, T=313.15, P=2e5)
    sys = HeatRecoverySystem(HeatRecoverySystemParams(lrhx_effectiveness=0.95))
    _, rich_hot, info = sys(lean, rich)
    q_lean = 200.0 * 75.0 * (393.15 - float(info["T_lean_after_lrhx"]))
    q_rich = 100.0 * 75.0 * (float(rich_hot["T"]) - 313.15)
    assert q_lean == pytest.approx(q_rich, rel=1e-12)
    assert float(info["Q_lrhx"]) == pytest.approx(q_rich, rel=1e-12)
    assert float(info["lrhx_effectiveness"]) <= 0.95


@pytest.mark.parametrize("P_in", [101325.0, 2e5, 5e5])
def test_compression_uses_stream_pressure(P_in):
    """C12: a 1 atm feed left at 74 bar instead of ~150 (P_inlet param used)."""
    out, info = CompressionTrain(CompressionTrainParams())(
        make_stream({"CO2": 1.0}, T=313.15, P=P_in))
    assert float(info["P_inlet"]) == P_in
    # Within the intercooler pressure drops of the 150 bar target.
    assert float(out["P"]) == pytest.approx(150e5, rel=0.03)


def test_compression_inconsistent_P_inlet_rejected():
    with pytest.raises(ValueError, match="does not match"):
        CompressionTrain(CompressionTrainParams(P_inlet=2e5))(
            make_stream({"CO2": 1.0}, T=313.15, P=101325.0))
