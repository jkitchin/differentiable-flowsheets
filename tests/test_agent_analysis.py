"""Expressions and the analysis tools built on them (MCP phase 3).

The point of the expression language is that an objective written as a
string differentiates exactly like one written as a Python function, so
the checks here compare against JAX on the hand-written function and
against finite differences, and confirm that what cannot be expressed is
refused rather than evaluated.
"""

import jax
import pytest

from difflow.agent import Workbench
from difflow.agent.expressions import ExpressionError, parse, references

PURITY = "vapor.F_ethyl_acetate / vapor.total_flow"


@pytest.fixture(scope="module")
def wb():
    bench = Workbench()
    assert bench.open_example("03_reactor_recycle")["ok"]
    assert bench.solve()["converged"]
    return bench


class TestLanguage:
    @pytest.mark.parametrize("text", [
        "__import__('os')", "os.system('ls')", "a.b.c", "[1, 2]", "x if y else z",
        "lambda: 1", "vapor['T']", "True", "econ.npv.__globals__",
    ])
    def test_anything_outside_the_whitelist_is_refused(self, text):
        with pytest.raises(ExpressionError):
            parse(text)

    def test_references(self):
        assert references("a.b * c + exp(d.e)") == {"a.b", "c", "d.e"}

    def test_values_match_the_streams(self, wb):
        streams = wb.get_streams(["vapor"])["streams"]["vapor"]
        total = sum(v for k, v in streams.items() if k.startswith("F_"))
        value = wb.evaluate([PURITY])["values"][PURITY]
        assert value == pytest.approx(streams["F_ethyl_acetate"] / total)

    def test_unit_parameters_resolve(self, wb):
        assert wb.evaluate(["reactor.V"])["values"]["reactor.V"] == 0.5

    def test_an_unknown_name_says_what_is_known(self, wb):
        answer = wb.evaluate(["vapour.T"])
        assert not answer["ok"] and "vapor" in answer["error"]


class TestQuantities:
    def test_defined_quantities_compose_and_are_saved(self, wb):
        assert wb.define_quantity("purity", PURITY)["ok"]
        assert wb.define_quantity("score", "2 * purity")["ok"]
        values = wb.evaluate(["purity", "score"])["values"]
        assert values["score"] == pytest.approx(2 * values["purity"])
        doc = wb.get_flowsheet("json")["flowsheet"]
        assert doc["view"]["quantities"]["score"] == "2 * purity"

    def test_a_cycle_is_refused(self, wb):
        assert wb.define_quantity("a1", "b1 + 1")["ok"]
        assert not wb.define_quantity("b1", "b1 * 2")["ok"]
        assert wb.define_quantity("b1", "a1 * 2")["ok"]
        assert "cycle" in wb.evaluate(["a1"])["error"]
        wb.remove_quantity("a1")
        wb.remove_quantity("b1")


class TestDerivatives:
    def test_the_expression_gradient_is_the_function_gradient(self, wb):
        """A string objective and a Python one give the same derivative."""
        fs = wb.sessions["main"].flowsheet

        def by_hand(params):
            s = fs._apply_params(params).solve(on_nonconvergence="ignore")
            v = s["vapor"]
            total = sum(v[k] for k in v if k.startswith("F_"))
            return v["F_ethyl_acetate"] / total

        expected = jax.grad(by_hand)({"reactor.V": 0.5})["reactor.V"]
        answer = wb.sensitivity(PURITY, wrt=["reactor.V"])
        assert answer["converged"] is True
        assert answer["levers"][0]["derivative"] == pytest.approx(float(expected), rel=1e-6)

    def test_the_derivative_agrees_with_a_sweep(self, wb):
        h = 1e-4
        points = wb.sweep("reactor.V", [PURITY], values=[0.5 - h, 0.5 + h])["points"]
        fd = (points[1][PURITY] - points[0][PURITY]) / (2 * h)
        d = wb.sensitivity(PURITY, wrt=["reactor.V"])["levers"][0]["derivative"]
        assert d == pytest.approx(fd, rel=1e-4)

    def test_an_unknown_lever_is_named(self, wb):
        answer = wb.sensitivity(PURITY, wrt=["reactor.Volume"])
        assert not answer["ok"] and "reactor.Volume" in answer["error"]


class TestOptimizeAndUncertainty:
    # An SLSQP run is dozens of solves with gradients; under a loaded xdist
    # run it can pass the Workbench's 300 s default and come back as a
    # timeout ({"ok": False, "timed_out": True}), which is not what these
    # tests are about, so they give it room.
    def test_a_bounded_optimum_meets_its_constraint(self):
        bench = Workbench()
        bench.open_example("03_reactor_recycle")
        profit = "20.0 * vapor.F_ethyl_acetate - 0.5 * reactor.V"
        answer = bench.optimize(profit, {"reactor.V": [0.05, 5.0]}, maximize=True,
                                constraints=[{"expression": PURITY, "ub": 0.33}],
                                timeout=3600)
        assert answer["success"], answer["message"]
        assert answer["constraints"][0]["value"] <= 0.33 + 1e-6
        assert answer["value"] > 20.0 * 0.2554 - 0.25     # better than the start

    def test_apply_writes_the_optimum(self):
        bench = Workbench()
        bench.open_example("03_reactor_recycle")
        answer = bench.optimize("-vapor.F_ethyl_acetate + 0.1 * reactor.V",
                                {"reactor.V": [0.05, 5.0]}, apply=True,
                                timeout=3600)
        assert answer["applied"] == ["reactor.V"]
        assert bench.evaluate(["reactor.V"])["values"]["reactor.V"] == \
            pytest.approx(answer["levers"]["reactor.V"])
        assert bench.undo()["ok"]

    def test_linear_uncertainty_shares_sum_to_one(self, wb):
        answer = wb.uncertainty(PURITY, {"reactor.V": 0.05, "feed:feed.T": 2.0})
        assert answer["std_linear"] > 0
        assert sum(answer["contributions"].values()) == pytest.approx(1.0)


def test_report(wb):
    answer = wb.report()
    assert answer["ok"] and "Flowsheet Report" in answer["report"]
