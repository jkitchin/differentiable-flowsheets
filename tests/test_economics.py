"""Smoke tests for difflow.economics."""

import warnings

import jax
import jax.numpy as jnp
import pytest

from difflow import economics as econ


def _finite(*xs):
    return all(bool(jnp.all(jnp.isfinite(jnp.asarray(x)))) for x in xs)


def test_indices():
    assert econ.get_cepci(2020) > 0
    assert _finite(econ.escalate_cost(1e6, base_year=2019, target_year=2024),
                   econ.inflation_factor(2020, 2025, annual_rate=0.02))
    with pytest.raises(ValueError):
        econ.get_cepci(1950)


def test_capital_costs():
    costs = [econ.reactor_cost(jnp.array(5.0), t) for t in econ.REACTOR_COSTS]
    costs += [econ.heat_exchanger_cost(jnp.array(100.0), t) for t in econ.HEAT_EXCHANGER_COSTS]
    costs += [econ.pump_cost(jnp.array(10.0), t) for t in econ.PUMP_COSTS]
    costs += [econ.compressor_cost(1000.0, "centrifugal"),
              econ.evaporator_cost(100.0, "falling_film"),
              econ.installed_cost(jnp.array(1e6), lang_factor=4.74),
              econ.total_capital_investment(jnp.array(1e6))]
    assert _finite(*costs) and all(float(c) > 0 for c in costs)
    with pytest.raises(ValueError, match="Unknown evaporator type"):
        econ.evaporator_cost(10.0, "calandria")


def test_utility_and_operating_costs():
    duty = jnp.array(1e6)
    vals = [econ.steam_cost_from_duty(duty, "medium_pressure"), econ.cooling_water_cost(duty),
            econ.electricity_cost(jnp.array(100.0)),
            econ.refrigeration_cost(duty, temperature_level=-20.0),
            econ.raw_material_cost(jnp.array(1.0), jnp.array(50.0)),
            econ.operating_labor_cost(n_operators_per_shift=4, n_shifts=3),
            econ.maintenance_cost(jnp.array(1e6)),
            econ.simple_opex(raw_materials=jnp.array(1e6), utilities=jnp.array(0.5e6),
                             fixed_capital=jnp.array(10e6)),
            econ.total_utility_cost(econ.UtilityConsumption(
                heating_duty=1e6, cooling_duty=2e6, electricity=100.0))]
    assert _finite(*vals)


def test_profitability():
    cf = jnp.full(10, 50.0)
    vals = [econ.present_value(100.0, jnp.array(0.1), jnp.array(5.0)),
            econ.npv(cf, jnp.array(0.1), jnp.array(100.0)),
            econ.irr(cf, jnp.array(100.0)),
            econ.simple_payback(jnp.array(50.0), jnp.array(100.0)),
            econ.discounted_payback(cf, jnp.array(100.0), jnp.array(0.1)),
            econ.minimum_selling_price(jnp.array(1000.0), jnp.array(100.0)),
            econ.annualized_cost(jnp.array(1e6), jnp.array(1e5), jnp.array(0.1), jnp.array(20.0)),
            econ.generate_cash_flows(capital_investment=jnp.array(1e6),
                                     annual_revenue=jnp.array(5e5),
                                     annual_opex=jnp.array(2e5))]
    assert _finite(*vals)
    res = econ.full_cash_flow_analysis(capital_investment=1e6, annual_revenue=4e5,
                                       annual_opex=1.5e5)
    assert _finite(res.npv, res.irr, res.payback, res.cash_flow)


def test_zero_rate_annuity_factors_are_finite():
    crf = econ.capital_recovery_factor(jnp.array(0.0), jnp.array(10.0))
    pvf = econ.present_value_factor(jnp.array(0.0), jnp.array(10.0))
    assert float(crf) == pytest.approx(0.1) and float(pvf) == pytest.approx(10.0)


def test_grad_is_finite():
    def f(V, rate):
        tci = econ.total_capital_investment(econ.reactor_cost(V, "cstr_jacketed"))
        return econ.npv(jnp.full(20, 1e6), rate, tci)
    g = jax.grad(f, argnums=(0, 1))(jnp.array(5.0), jnp.array(0.0))
    assert _finite(*g)


def test_cost_range_warning():
    with pytest.warns(econ.CostRangeWarning, match="outside the correlation"):
        econ.compressor_cost(30.0, "centrifugal")
    with warnings.catch_warnings():
        warnings.simplefilter("error", econ.CostRangeWarning)
        econ.compressor_cost(1000.0, "centrifugal")
        jax.grad(lambda s: econ.compressor_cost(s, "centrifugal"))(30.0)


_NEW_COSTS = [
    (lambda s, t: econ.dryer_cost(s, t), econ.DRYER_COSTS, "Unknown dryer type"),
    (lambda s, t: econ.crystallizer_cost(s, t), econ.CRYSTALLIZER_COSTS,
     "Unknown crystallizer type"),
    (lambda s, t: econ.cooling_tower_cost(s, t), econ.COOLING_TOWER_COSTS,
     "Unknown cooling tower type"),
    (lambda s, t: econ.separator_cost(s, t), econ.SEPARATOR_COSTS, "Unknown separator type"),
]


def _mid(p):
    return (p.S_min * p.S_max) ** 0.5


@pytest.mark.parametrize("fn,table,_", _NEW_COSTS)
def test_new_equipment_costs_smoke_and_grad(fn, table, _):
    with warnings.catch_warnings():
        warnings.simplefilter("error", econ.CostRangeWarning)
        for kind, p in table.items():
            c = fn(jnp.array(_mid(p)), kind)
            g = jax.grad(lambda s: fn(s, kind))(jnp.array(_mid(p)))
            assert _finite(c, g) and float(c) > 0, kind
    assert "filter_press" in econ.SEPARATOR_COSTS


@pytest.mark.parametrize("fn,table,msg", _NEW_COSTS)
def test_new_equipment_costs_validation(fn, table, msg):
    with pytest.raises(ValueError, match=msg):
        fn(1.0, "no_such_type")
    kind, p = next(iter(table.items()))
    with pytest.warns(econ.CostRangeWarning, match="outside the correlation"):
        fn(p.S_max * 10.0, kind)
