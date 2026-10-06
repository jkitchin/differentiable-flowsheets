"""Cost bases declared by units, and the tea tool (MCP phase 5)."""

import pytest

from difflow.agent import Workbench
from difflow.catalog import catalog
from difflow.economics import capital
from difflow.economics.indices import DEFAULT_CURRENT_YEAR


@pytest.fixture()
def wb():
    bench = Workbench()
    bench.open_example("03_reactor_recycle")
    assert bench.solve()["converged"]
    return bench


def test_declared_bases_name_real_correlations():
    declared = {n: s.cost_basis for n, s in catalog().items() if s.cost_basis}
    assert {"CSTR", "PFR", "GasPFR", "FedBatchReactor", "SemiBatchReactor"} <= set(declared)
    for name, basis in declared.items():
        table = getattr(capital, basis["table"])
        assert basis["type"] in table, name


def test_the_price_is_the_economics_module_price(wb):
    answer = wb.tea()
    reactor = answer["units"][0]
    expected = float(capital.reactor_cost(0.5, "cstr_jacketed", DEFAULT_CURRENT_YEAR))
    assert reactor["purchased_cost"] == pytest.approx(expected, rel=1e-12)
    assert answer["total_capital_investment"] == pytest.approx(expected * 4.74 * 1.15)
    assert {u["unit"] for u in answer["uncosted"]} == {"mixer", "flash", "splitter"}


def test_capex_is_a_differentiable_quantity(wb):
    wb.tea()
    values = wb.evaluate(["capex"])["values"]
    assert values["capex"] == pytest.approx(wb.tea()["total_capital_investment"])
    d = wb.sensitivity("capex", wrt=["reactor.V"])["levers"][0]["derivative"]
    h = 1e-4
    pts = wb.sweep("reactor.V", ["capex"], values=[0.5 - h, 0.5 + h])["points"]
    assert d == pytest.approx((pts[1]["capex"] - pts[0]["capex"]) / (2 * h), rel=1e-5)


def test_a_size_outside_the_correlation_is_flagged(wb):
    assert wb.update_unit("reactor", params={"V": 250.0})["ok"]
    answer = wb.tea(define=False)
    assert answer["units"][0]["in_range"] is False
    assert "outside" in answer["warnings"][0]
    assert "capex" not in wb.list_quantities()["quantities"]
