"""Smoke tests for pipe insulation heat loss, cost and economic thickness."""

import jax
import numpy as np
import pytest

from difflow.economics import (
    economic_insulation_thickness,
    insulation_annual_cost,
    insulation_installed_cost,
    insulation_volume_per_length,
)
from difflow.heat_transfer import insulated_pipe_heat_loss

D, K_INS, H_OUT, T_P, T_A = 0.1143, 0.05, 10.0, 450.0, 293.15
CASE = dict(cost_per_m3=3000.0, heat_cost=10.0)


def test_insulation_smoke():
    assert np.isfinite(float(insulated_pipe_heat_loss(D, 0.05, K_INS, H_OUT, T_P, T_A)))
    assert np.isfinite(float(insulation_volume_per_length(D, 0.04)))
    assert np.isfinite(float(insulation_installed_cost(D, 0.05, 2500.0, base_year=2010, target_year=2022)))
    annual = insulation_annual_cost(0.05, D, K_INS, H_OUT, T_P, T_A, **CASE)
    assert {"capital", "heat", "total"} <= set(annual)
    out = economic_insulation_thickness(D, K_INS, H_OUT, T_P, T_A, **CASE)
    assert {"thickness", "total", "interior", "heat_loss", "bare_heat_loss"} <= set(out)
    assert np.isfinite(float(out["thickness"]))


def test_cepci_needs_both_years():
    with pytest.raises(ValueError, match="both"):
        insulation_installed_cost(D, 0.05, 2500.0, base_year=2010)


def test_optimum_grad_is_finite():
    g = jax.grad(lambda c: economic_insulation_thickness(
        D, K_INS, H_OUT, T_P, T_A, cost_per_m3=3000.0, heat_cost=c)["thickness"])(10.0)
    assert np.isfinite(float(g))
