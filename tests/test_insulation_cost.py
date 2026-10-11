"""Pipe insulation: heat loss, installed cost and the economic thickness.

No published cost coefficient is used: the unit cost is a required input
and every number here is illustrative. What is verified: the heat loss
against the series-resistance arithmetic, the cost model against hand
arithmetic and CEPCI escalation, the optimum against a dense NumPy grid
minimum and the stationarity condition, and its gradients against finite
differences (implicit differentiation through the bisection root find).
"""

import math

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from difflow.economics import (
    capital_recovery_factor,
    economic_insulation_thickness,
    get_cepci,
    insulation_annual_cost,
    insulation_installed_cost,
    insulation_volume_per_length,
)
from difflow.heat_transfer import critical_insulation_radius, insulated_pipe_heat_loss

D, K_INS, H_OUT, T_P, T_A = 0.1143, 0.05, 10.0, 450.0, 293.15   # 4-in steam line
CASE = dict(cost_per_m3=3000.0, heat_cost=10.0)                 # ILLUSTRATIVE prices


def test_heat_loss_series_resistances():
    t = 0.05
    r1, r2 = D / 2, D / 2 + t
    R = math.log(r2 / r1) / (2 * math.pi * K_INS) + 1.0 / (H_OUT * 2 * math.pi * r2)
    assert float(insulated_pipe_heat_loss(D, t, K_INS, H_OUT, T_P, T_A)) == pytest.approx((T_P - T_A) / R, rel=1e-12)
    bare = (T_P - T_A) * H_OUT * math.pi * D
    assert float(insulated_pipe_heat_loss(D, 0.0, K_INS, H_OUT, T_P, T_A)) == pytest.approx(bare, rel=1e-12)


def test_heat_loss_peaks_at_critical_radius():
    # Thin wire: r < r_cr = k/h, so the loss rises then falls; maximum at r_cr.
    k, h, d = 0.5, 10.0, 0.01
    r_cr = float(critical_insulation_radius(k, h))
    dq = jax.grad(lambda t: insulated_pipe_heat_loss(d, t, k, h, 350.0, 300.0))
    assert float(dq(r_cr - d / 2)) == pytest.approx(0.0, abs=1e-9)


def test_volume_and_cost_hand():
    t = 0.04
    V = math.pi / 4 * ((D + 2 * t) ** 2 - D**2)
    assert float(insulation_volume_per_length(D, t)) == pytest.approx(V, rel=1e-12)
    c = insulation_installed_cost(D, t, 2500.0, cost_per_m2=40.0, cost_per_m=15.0)
    assert float(c) == pytest.approx(2500.0 * V + 40.0 * math.pi * (D + 2 * t) + 15.0, rel=1e-12)


def test_cepci_escalation():
    base = insulation_installed_cost(D, 0.05, 2500.0)
    esc = insulation_installed_cost(D, 0.05, 2500.0, base_year=2010, target_year=2022)
    assert float(esc) == pytest.approx(float(base) * get_cepci(2022) / get_cepci(2010), rel=1e-12)
    with pytest.raises(ValueError, match="both"):
        insulation_installed_cost(D, 0.05, 2500.0, base_year=2010)


def test_annual_cost_parts():
    out = insulation_annual_cost(0.05, D, K_INS, H_OUT, T_P, T_A, hours_per_year=8000.0,
                                 rate=0.1, years=10.0, **CASE)
    crf = float(capital_recovery_factor(0.1, 10.0))
    q = float(insulated_pipe_heat_loss(D, 0.05, K_INS, H_OUT, T_P, T_A))
    assert float(out["capital"]) == pytest.approx(crf * float(insulation_installed_cost(D, 0.05, 3000.0)), rel=1e-12)
    assert float(out["heat"]) == pytest.approx(q * 8000 * 3600 * 10.0 * 1e-9, rel=1e-12)
    assert float(out["total"]) == pytest.approx(float(out["capital"]) + float(out["heat"]), rel=1e-12)


def test_optimum_matches_grid_minimum():
    out = economic_insulation_thickness(D, K_INS, H_OUT, T_P, T_A, **CASE)
    assert bool(out["interior"])
    ts = np.linspace(0.001, 0.3, 30001)
    tac = np.asarray(jax.vmap(lambda t: insulation_annual_cost(t, D, K_INS, H_OUT, T_P, T_A, **CASE)["total"])(ts))
    t_grid = ts[np.argmin(tac)]
    assert float(out["thickness"]) == pytest.approx(t_grid, abs=2e-5)
    assert float(out["total"]) <= tac.min() + 1e-9
    g = jax.grad(lambda t: insulation_annual_cost(t, D, K_INS, H_OUT, T_P, T_A, **CASE)["total"])(out["thickness"])
    assert abs(float(g)) < 1e-6
    assert float(out["heat_loss"]) < float(out["bare_heat_loss"])


def test_thicker_when_heat_is_dearer_or_insulation_cheaper():
    t = lambda **kw: float(economic_insulation_thickness(D, K_INS, H_OUT, T_P, T_A, **{**CASE, **kw})["thickness"])
    assert t(heat_cost=20.0) > t() > t(heat_cost=5.0)
    assert t(cost_per_m3=1500.0) > t()


def test_no_interior_optimum_flagged():
    out = economic_insulation_thickness(D, K_INS, H_OUT, T_P, T_A, cost_per_m3=3000.0, heat_cost=1e-4)
    assert not bool(out["interior"])
    assert float(out["thickness"]) == pytest.approx(1e-4, abs=1e-8)


def test_gradient_of_optimum_matches_fd():
    def t_opt(heat_cost, k_ins, c_v):
        return economic_insulation_thickness(D, k_ins, H_OUT, T_P, T_A, cost_per_m3=c_v,
                                             heat_cost=heat_cost)["thickness"]

    x = (10.0, K_INS, 3000.0)
    grads = jax.grad(t_opt, argnums=(0, 1, 2))(*x)
    for i, g in enumerate(grads):
        step = [1e-4, 1e-7, 1e-2][i]
        xp = list(x); xm = list(x)
        xp[i] += step; xm[i] -= step
        fd = (float(t_opt(*xp)) - float(t_opt(*xm))) / (2 * step)
        assert float(g) == pytest.approx(fd, rel=1e-4)
    assert float(grads[0]) > 0 and float(grads[2]) < 0


def test_cost_grads():
    # O(1) multipliers so the finite-difference step suits every argument.
    check_grads(lambda st, sc: insulation_installed_cost(D, st * 0.05, sc * 2500.0, 30.0, 5.0), (1.0, 1.0),
                order=1, modes=["rev", "fwd"])
    check_grads(lambda st, sk: insulated_pipe_heat_loss(D, st * 0.05, sk * K_INS, H_OUT, T_P, T_A), (1.0, 1.0),
                order=1, modes=["rev", "fwd"])
