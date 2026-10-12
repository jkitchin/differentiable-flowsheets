"""Tests for installed piping cost vs diameter / NPS and schedule (difflow.economics)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import difflow.economics as econ
from difflow import Pipe, PipeParams
from difflow.fluids import pipe_inside_diameter, pipe_weight_per_length

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nps,sch,kg_m", [
    # ASME B36.10M plain-end weights (kg/m); the standard computes them from
    # dimensions rounded to 0.01 mm, hence the 0.01 kg/m tolerance
    (1, "40", 2.50), (2, "40", 5.44), (4, "40", 16.07), (4, "80", 22.32), (6, "40", 28.26),
])
def test_pipe_weight_matches_b36_10m(nps, sch, kg_m):
    assert pipe_weight_per_length(nps, sch) == pytest.approx(kg_m, abs=0.01)


def test_diameter_exponents_and_continuity():
    c = lambda D: float(econ.pipe_cost_per_length(D, 100.0, 0.0254, 2019, 2019))
    assert c(0.0254) == pytest.approx(100.0)
    assert c(0.1016) / c(0.0508) == pytest.approx(2.0**1.5, rel=1e-12)   # above 1 in
    assert c(0.0127) / c(0.00635) == pytest.approx(2.0, rel=1e-12)       # below 1 in
    assert c(0.0254 * (1 - 1e-9)) == pytest.approx(c(0.0254 * (1 + 1e-9)), rel=1e-8)


def test_cepci_escalation():
    base = econ.pipe_cost_per_length(0.1, 150.0, 0.0525, 2010, 2010)
    esc = econ.pipe_cost_per_length(0.1, 150.0, 0.0525, 2010, 2022)
    np.testing.assert_allclose(float(esc / base), econ.get_cepci(2022) / econ.get_cepci(2010), rtol=1e-12)


def test_gradient_is_local_exponent():
    lnC = lambda lnD: jnp.log(econ.pipe_cost_per_length(jnp.exp(lnD), 150.0, 0.0525, 2019, 2019))
    np.testing.assert_allclose(float(jax.grad(lnC)(math.log(0.1))), 1.5, rtol=1e-12)
    np.testing.assert_allclose(float(jax.grad(lnC)(math.log(0.015))), 1.0, rtol=1e-12)


def test_nps_schedule_scaling():
    kw = dict(C_ref=200.0, nps_ref=2, base_year=2019, target_year=2019)
    c2 = float(econ.pipe_cost_nps(2, "40", **kw))
    c4 = float(econ.pipe_cost_nps(4, "40", **kw))
    c4_80 = float(econ.pipe_cost_nps(4, "80", **kw))
    assert c2 == pytest.approx(200.0)
    ratio_D = (pipe_inside_diameter(4) / pipe_inside_diameter(2)) ** 1.5
    assert c4 / c2 == pytest.approx(ratio_D, rel=1e-12)
    assert c4_80 / c4 == pytest.approx(pipe_weight_per_length(4, "80") / pipe_weight_per_length(4, "40"),
                                       rel=1e-12)
    assert float(econ.pipe_cost_nps(4, "40", length=50.0, **kw)) == pytest.approx(50 * c4)
    with pytest.raises(KeyError):
        econ.pipe_cost_nps(22, "40", **kw)


def _annual_cost(D, Q, C_ref=150.0, D_ref=0.0525):
    """Annualized pipe capital (20 %/yr) + pumping energy at $0.10/kWh, eta 0.7, 8000 h."""
    L = 1000.0
    capital = 0.20 * L * econ.pipe_cost_per_length(D, C_ref, D_ref, 2019, 2019)
    v = Q / (math.pi / 4 * D**2)
    pipe = Pipe(PipeParams(L=L, D=D, rho=998.2, mu=1.002e-3))
    dP = pipe.hydraulics_at_flow(Q)["dP"]
    energy = dP * Q / 0.7 / 1000.0 * 8000.0 * 0.10
    return capital + energy


def test_economic_diameter_scaling():
    """The optimum D scales close to Q^0.45 (Peters & Timmerhaus' turbulent result for n = 1.5)."""
    def opt(Q):
        r = minimize_scalar(lambda lnD: float(_annual_cost(math.exp(lnD), Q)),
                            bounds=(math.log(0.03), math.log(0.6)), method="bounded",
                            options={"xatol": 1e-10})
        return math.exp(r.x)
    D1, D2 = opt(0.01), opt(0.04)
    exponent = math.log(D2 / D1) / math.log(4.0)
    assert 0.40 < exponent < 0.48
    # jax.grad of the annual cost vanishes at the optimum
    g = float(jax.grad(_annual_cost)(D1, 0.01))
    assert abs(g) * D1 < 1e-4 * float(_annual_cost(D1, 0.01))
