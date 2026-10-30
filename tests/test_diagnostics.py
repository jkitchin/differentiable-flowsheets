"""Diagnosis and convergence: difflow.diagnostics and the agent's doctor tools.

Contracts only: the regime classifier, the remedy/symptom tables, and the
agent's diagnose / tear-analysis tools on cheap flowsheets.
"""

import pytest

from difflow.agent import Workbench
from difflow.convergence import get_case
from difflow.diagnostics import (
    REMEDIES,
    SYMPTOMS,
    classify_history,
    matching_symptoms,
    solve_findings,
)


def bench(case_name: str) -> Workbench:
    wb = Workbench()
    wb.sessions["main"] = wb._Session(flowsheet=get_case(case_name).build())
    return wb


class TestInnerSolves:

    def test_a_balance_that_does_not_close_is_an_audit_failure(self):
        wb = Workbench()
        wb.open_example("02_reactor_flash")
        fs = wb.sessions["main"].flowsheet
        wb.solve()
        fs.last_solve_unit_info["reactor"]["converged"] = False
        fs.last_solve_unit_info["reactor"]["balance_residual"] = 0.2
        audit = wb.sessions["main"]._audit(wb.sessions["main"].streams)
        assert any("reactor's own solve" in w for w in audit["warnings"])
        kinds = [f.kind for f in solve_findings(fs)]
        assert "inner-solve" in kinds


# =============================================================================
# Reading the iteration, and the remedy table
# =============================================================================


class TestClassify:
    @pytest.mark.parametrize("history, regime", [
        ([1.0, 0.9, 0.81, 0.73, 0.66], "creeping"),
        ([1.0, 1.5, 2.2, 3.4, 5.0], "diverging"),
        ([1.0, 0.5, 0.9, 0.4, 0.95, 0.45, 0.9], "oscillating"),
        ([1.0, 1.0, 1.0, 1.0], "stalled"),
        ([1.0, float("nan")], "nan"),
        (None, "no-history"),
    ])
    def test_regimes(self, history, regime):
        assert classify_history(history, False) == regime

    def test_a_converged_solve_is_converged(self):
        assert classify_history([1.0, 2.0], True) == "converged"

    def test_every_remedy_is_a_valid_solve_setting(self):
        from difflow.workspace.session import _solver_option

        for remedy in REMEDIES.values():
            assert remedy.kind in ("numerics", "problem")
            for key, value in remedy.options.items():
                _solver_option(key, value)

    def test_symptoms_name_real_remedies(self):
        for symptom in SYMPTOMS:
            assert set(symptom.remedies) <= set(REMEDIES), symptom.title
        assert matching_symptoms("TracerBoolConversionError: x")[0].title \
            == "TracerArrayConversionError"

    def test_the_editor_reads_the_same_cards(self):
        from difflow.workspace.context import TROUBLESHOOTING

        assert [t[0] for t in TROUBLESHOOTING] == [s.title for s in SYMPTOMS]


# =============================================================================
# The tools
# =============================================================================


class TestDiagnose:

    def test_an_unfinished_flowsheet_is_stopped_before_the_solve(self):
        wb = Workbench()
        wb.open_example("01_flash_drum")
        wb.remove_feed("feed")
        answer = wb.diagnose()
        assert answer["solved"] is False
        assert answer["findings"][0]["kind"] == "unfed-inlet"

    def test_tear_analysis_and_trace(self):
        wb = bench("two_loop_recycle")
        tears = wb.tear_analysis()
        assert tears["ok"] and len(tears["declared"]) == 2
        trace = wb.trace_solve()
        assert trace["converged"] and trace["regime"] == "converged"
        assert trace["history"]
