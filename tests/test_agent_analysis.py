"""Expressions and the analysis tools built on them (MCP phase 3)."""

import pytest

from difflow.agent import Workbench
from difflow.agent.expressions import ExpressionError, parse, references

PURITY = "vapor.F_ethyl_acetate / vapor.total_flow"


@pytest.fixture(scope="module")
def wb():
    bench = Workbench()
    assert bench.open_example("01_flash_drum")["ok"]
    assert bench.solve()["converged"]
    return bench


def test_anything_outside_the_whitelist_is_refused():
    with pytest.raises(ExpressionError):
        parse("__import__('os')")


def test_references():
    assert references("a.b * c + exp(d.e)") == {"a.b", "c", "d.e"}


def test_an_unknown_name_says_what_is_known(wb):
    answer = wb.evaluate(["vapour.T"])
    assert not answer["ok"] and "vapor" in answer["error"]


def test_a_cycle_is_refused(wb):
    assert wb.define_quantity("a1", "b1 + 1")["ok"]
    assert not wb.define_quantity("b1", "b1 * 2")["ok"]
    assert wb.define_quantity("b1", "a1 * 2")["ok"]
    assert "cycle" in wb.evaluate(["a1"])["error"]
    wb.remove_quantity("a1")
    wb.remove_quantity("b1")


def test_an_unknown_lever_is_named(wb):
    answer = wb.sensitivity(PURITY, wrt=["reactor.Volume"])
    assert not answer["ok"] and "reactor.Volume" in answer["error"]


def test_report(wb):
    answer = wb.report()
    assert answer["ok"] and "Flowsheet Report" in answer["report"]
