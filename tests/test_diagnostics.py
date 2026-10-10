"""Diagnosis and convergence: difflow.diagnostics and the agent's doctor tools.

The bar for phase 2: starting from the default settings, ``converge``
finds a passing setting for every case in the convergence corpus, each
judged "converged AND correct", and ``diagnose`` names what is wrong with
the ones that fail.
"""

import jax
import pytest

from difflow.agent import Workbench
from difflow.convergence import CORPUS, get_case
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


# =============================================================================
# The record a solve leaves
# =============================================================================


class TestSolveRecord:
    def test_the_history_is_the_residual_per_iteration(self):
        fs = get_case("flash_recycle").build()
        fs.solve(acceleration="wegstein", max_iter=7, on_nonconvergence="ignore")
        assert len(fs.last_solve_history) == 7
        assert fs.last_solve_history[-1] == fs.last_solve_residual

    def test_the_fixed_point_path_keeps_no_history(self):
        fs = get_case("flash_recycle").build()
        fs.solve(acceleration="none", on_nonconvergence="ignore")
        assert fs.last_solve_history is None

    def test_unit_info_is_kept(self):
        wb = Workbench()
        wb.open_example("03_reactor_recycle")
        fs = wb.sessions["main"].flowsheet
        fs.solve()
        assert {"reactor", "flash"} <= set(fs.last_solve_unit_info)
        assert bool(fs.last_solve_unit_info["reactor"]["converged"])

    def test_nothing_is_kept_while_tracing(self):
        """Info holding tracers must not outlive jax.grad."""
        wb = Workbench()
        wb.open_example("03_reactor_recycle")
        fs = wb.sessions["main"].flowsheet
        f = fs.make_objective_fn(lambda s: s["purge"]["F_ethyl_acetate"])
        g = jax.grad(f)({"reactor.V": 0.5})["reactor.V"]
        assert float(g) > 0
        assert fs.last_solve_unit_info == {}


class TestInnerSolves:
    @pytest.mark.parametrize("example", ["01_flash_drum", "02_reactor_flash"])
    def test_a_good_solve_closes_its_balance(self, example):
        wb = Workbench()
        wb.open_example(example)
        wb.solve()
        for name, info in wb.get_unit_info()["units"].items():
            assert info["converged"] is True, name
            assert info["balance_residual"] < 1e-10, name

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
        from difflow.gui.session import _solver_option

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
        from difflow.gui.context import TROUBLESHOOTING

        assert [t[0] for t in TROUBLESHOOTING] == [s.title for s in SYMPTOMS]


# =============================================================================
# The tools
# =============================================================================


class TestDiagnose:
    def test_a_failing_loop_is_named_with_its_remedies(self):
        answer = bench("phase_coupled_flash").diagnose()
        kinds = [f["kind"] for f in answer["findings"]]
        assert answer["converged"] is False and kinds[0] == "not-converged"
        assert "wegstein" in answer["remedies"]

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


# Only Wegstein converges phase_coupled_flash, so an unbudgeted converge
# runs the whole ladder (~50 s on a laptop, ~160 s on a CI runner) and four
# tests doing it were most of a CI shard. The default acceleration is
# already Anderson, so the ladder skips it and Wegstein is the second trial
# (current, wegstein): budget=2 reaches the same verdict. One
# applied run serves every per-commit assertion about it, and the full-ladder
# run is the release tier's.
PCF_BUDGET = 2


@pytest.fixture(scope="module")
def pcf_applied():
    wb = bench("phase_coupled_flash")
    return wb, wb.converge(apply=True, budget=PCF_BUDGET, timeout=3600)


class TestConverge:
    @pytest.mark.parametrize("case", [
        pytest.param(c.name, marks=pytest.mark.release)
        if c.name == "phase_coupled_flash" else c.name for c in CORPUS])
    def test_every_corpus_case_gets_a_passing_setting(self, case):
        answer = bench(case).converge(timeout=3600)
        assert answer["passed"], [(t["remedy"], t["problems"]) for t in answer["trials"]]

    def test_the_known_remedy_is_found(self, pcf_applied):
        """The corpus records that only Wegstein solves this one."""
        _, answer = pcf_applied
        assert answer["passed"]
        assert [(t["remedy"], t["passed"]) for t in answer["trials"]] == [
            ("current", False), ("wegstein", True)]
        assert answer["best"]["remedy"] == "wegstein"

    def test_apply_keeps_a_numerics_remedy_and_it_is_undoable(self, pcf_applied):
        wb, answer = pcf_applied
        assert answer["applied"] == "wegstein"
        assert wb.sessions["main"].flowsheet.view["solver"] == {"acceleration": "wegstein"}
        assert wb.solve()["converged"] is True
        # The corpus builds this flowsheet in Python, which the session
        # cannot serialize, so there is no undo record and it says so.
        assert answer["undoable"] is False

    def test_an_applied_remedy_on_a_saved_flowsheet_is_undoable(self):
        wb = Workbench()
        wb.open_example("03_reactor_recycle")
        assert wb.set_solver_options({"acceleration": "none", "max_iter": 3})["ok"]
        answer = wb.converge(apply=True, timeout=3600)
        assert answer["applied"] and answer["undoable"] is True
        assert wb.undo()["ok"]
        assert wb.sessions["main"].solver_options()["max_iter"] == 3

    def test_without_apply_the_settings_are_untouched(self):
        wb = Workbench()
        wb.open_example("03_reactor_recycle")
        assert wb.set_solver_options({"acceleration": "none", "max_iter": 3})["ok"]
        answer = wb.converge(timeout=3600)
        assert answer["passed"] and answer["best"]["remedy"] != "current"
        assert answer["applied"] is None
        assert wb.sessions["main"].solver_options()["max_iter"] == 3
        assert wb.sessions["main"].solver_options()["acceleration"] == "none"

    def test_a_signed_answer_passes_with_a_caveat(self):
        """Negative flows qualify an answer; they do not refute it."""
        answer = bench("signed_tear").converge(timeout=3600)
        best = next(t for t in answer["trials"] if t["passed"])
        assert any(c.startswith("negative flows") for c in best["caveats"])
