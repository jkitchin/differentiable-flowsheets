"""REE economics: capex/opex/revenue smoke tests and input validation."""

import math

import pytest

from difflow_ree.economics.costs import (
    calculate_profit,
    calculate_revenue,
    capex_basis,
    estimate_capex,
    estimate_opex,
    minimum_selling_price,
)


def test_capex_rows_sum_to_total():
    capex = estimate_capex(500, 8, 4, 4)
    assert all(isinstance(v, float) and math.isfinite(v) for v in capex.values())
    rows = sum(v for k, v in capex.items() if k != "total")
    assert rows == pytest.approx(capex["total"], rel=1e-12)


def test_unknown_scope_or_year_raises():
    with pytest.raises(ValueError, match="unknown scope"):
        capex_basis("whatever")
    with pytest.raises(ValueError, match="no CEPCI value"):
        estimate_capex(500, 8, 4, 4, year=1975)


def test_opex_revenue_profit_smoke():
    opex = estimate_opex(annual_ree_tonnes=100, capex=1e7, extractant="D2EHPA")
    assert math.isfinite(opex["total"]) and opex["total"] > 0
    revenue = calculate_revenue({"Nd": 1000.0, "Dy": 100.0})
    assert revenue["total"] > 0
    profit = calculate_profit(revenue["total"], opex["total"], 1e7)
    assert "net_income" in profit
    assert math.isfinite(minimum_selling_price(opex["total"], 1e7, 1e5))
