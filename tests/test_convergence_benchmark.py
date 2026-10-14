"""Tests for :mod:`difflow.convergence`, the pass-rate benchmark.

The benchmark's own correctness is what these pin, not the pass rate it
reports: that number is a measurement and is meant to move.  What must not
move is the machinery that produces it --- that a raised solve is counted
rather than propagated, that "the solver said it converged" and "the answer
audits" stay separate, and that no case is rigged so a strategy starts on
the answer.

Everything here runs against cheap analytic cases or synthetic ones built
in the test.
"""

import warnings

import jax.numpy as jnp
import pytest

from difflow.convergence import (
    CORPUS,
    Case,
    Outcome,
    Report,
    _feed_guess,
    _product_streams,
    cases,
    get_case,
    run_case,
    total_mole_balance,
)
from difflow.flowsheet import ConvergenceWarning, Flowsheet, Unit
from difflow.streams import get_flows, make_stream


# =============================================================================
# The corpus itself
# =============================================================================

class TestCorpus:

    def test_names_are_unique(self):
        names = [c.name for c in CORPUS]
        assert len(names) == len(set(names))

    def test_every_case_says_why_it_is_hard(self):
        """An unexplained hard case measures something nobody can act on."""
        for case in CORPUS:
            assert case.difficulty.strip(), f"{case.name} has no difficulty note"
            assert len(case.difficulty) > 40, (
                f"{case.name}: 'difficulty' should be a sentence, not a label")

    def test_build_returns_a_fresh_flowsheet_each_time(self):
        """Solve records its verdict on the object; a shared one leaks it."""
        for case in CORPUS:
            first, second = case.build(), case.build()
            assert first is not second
            assert first.last_solve_converged is None
            assert second.last_solve_converged is None

    def test_every_case_has_a_recycle(self):
        """A flowsheet with no recycle does not exercise the tear solver."""
        for case in CORPUS:
            assert case.build().recycles, f"{case.name} has no recycle"

    def test_get_case_rejects_an_unknown_name(self):
        with pytest.raises(KeyError, match="unknown case"):
            get_case("no_such_case")

    def test_get_case_finds_every_corpus_member(self):
        """The loop used to raise on the first non-match instead of the last."""
        for case in CORPUS:
            assert get_case(case.name) is case

    def test_cases_filters_by_tag(self):
        flash = cases(tags=["flash"])
        assert flash
        assert all("flash" in c.tags for c in flash)
        assert len(flash) < len(CORPUS)

    def test_the_corpus_has_a_control_that_is_meant_to_pass(self):
        """All-failing corpora cannot tell a hard case from a broken solver."""
        assert any("control" in c.tags for c in CORPUS)


# =============================================================================
# Auditing
# =============================================================================

def _straight_through() -> Flowsheet:
    """feed -> unit -> product, no recycle.  Balances exactly."""
    fs = Flowsheet(["A"])
    fs.add_feed("feed", make_stream({"A": 2.0}, 300.0, 101325.0))
    fs.add_unit(Unit(
        "u", lambda s: make_stream(get_flows(s), s["T"], s["P"]),
        ["feed"], ["product"]))
    return fs


class TestBalanceAudit:

    def test_a_recycle_source_is_not_a_product(self):
        """It is consumed at the far end of the tear, so it does not leave."""
        fs = get_case("high_gain_recycle").build()
        products = _product_streams(fs)
        assert "rec_src" not in products
        assert "product" in products

    def test_a_closed_balance_scores_zero(self):
        fs = _straight_through()
        streams = {"product": make_stream({"A": 2.0}, 300.0, 101325.0)}
        assert total_mole_balance(fs, streams) == pytest.approx(0.0)

    def test_a_broken_balance_is_caught(self):
        fs = _straight_through()
        streams = {"product": make_stream({"A": 1.0}, 300.0, 101325.0)}
        assert total_mole_balance(fs, streams) == pytest.approx(0.5)

    def test_a_diverged_answer_reports_inf_rather_than_raising(self):
        """A failing case still has to report a number."""
        fs = _straight_through()
        streams = {"product": make_stream({"A": jnp.inf}, 300.0, 101325.0)}
        assert total_mole_balance(fs, streams) == float("inf")

    def test_a_missing_product_does_not_raise(self):
        fs = _straight_through()
        assert total_mole_balance(fs, {}) == pytest.approx(1.0)


# =============================================================================
# Outcomes: the solver's verdict against the audit
# =============================================================================

class TestOutcomeVerdict:
    """The distinction the whole module exists to make."""

    def test_converged_and_correct_passes(self):
        o = Outcome("c", "anderson", "unit", converged=True, correct=True)
        assert o.passed
        assert o.verdict == "pass"

    def test_converged_but_wrong_does_not_pass(self):
        """The failure mode that a bare convergence flag cannot see."""
        o = Outcome("c", "anderson", "unit", converged=True, correct=False)
        assert not o.passed
        assert o.verdict == "WRONG"

    def test_correct_but_not_converged_does_not_pass(self):
        o = Outcome("c", "none", "unit", converged=False, correct=True)
        assert not o.passed
        assert o.verdict == "no-conv"

    def test_a_raise_is_its_own_verdict(self):
        o = Outcome("c", "none", "unit", error="ValueError: boom")
        assert not o.passed
        assert o.verdict == "raised"


# =============================================================================
# Running
# =============================================================================

class TestRunCase:


    def test_a_raising_case_is_counted_not_propagated(self):
        """A benchmark that dies on its hardest case reports a short corpus."""
        def build():
            fs = Flowsheet(["A"])
            fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 101325.0))

            def explode(feed, tear):
                raise RuntimeError("unit blew up")

            fs.add_unit(Unit("boom", explode, ["feed", "tear"], ["out"]))
            fs.add_recycle("out", "tear")
            return fs

        case = Case(name="explodes", build=build,
                    difficulty="A unit that raises, to check the harness "
                               "records a raise rather than propagating it.")
        outcome = run_case(case)
        assert outcome.error is not None
        assert "unit blew up" in outcome.error
        assert outcome.traceback is not None
        assert not outcome.passed
        assert outcome.verdict == "raised"


    def test_non_convergence_does_not_warn_through(self):
        """Counting non-convergence is the job; announcing each one is not."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            run_case(get_case("overshoot_loop"), acceleration="none")

    def test_an_unknown_initialization_is_rejected(self):
        with pytest.raises(ValueError, match="unknown initialization"):
            run_case(get_case("two_phase_flash"), initialization="telepathy")


class TestFeedGuess:

    def test_it_sums_the_feeds_and_reaches_every_tear(self):
        fs = get_case("high_gain_recycle").build()
        guess = _feed_guess(fs)
        assert set(guess) == set(fs.recycles.values())
        assert float(guess["rec"]["F_A"]) == pytest.approx(1.0)

    def test_it_covers_every_species_in_order(self):
        """A guess missing a species fails as a KeyError frames away."""
        fs = get_case("high_gain_recycle").build()
        for stream in _feed_guess(fs).values():
            for species in fs.species_order:
                assert f"F_{species}" in stream


# =============================================================================
# Reporting
# =============================================================================

def _report(*verdicts) -> Report:
    """A report from (acceleration, converged, correct) triples."""
    report = Report()
    for i, (accel, converged, correct) in enumerate(verdicts):
        report.outcomes.append(Outcome(
            case=f"case{i}", acceleration=accel, initialization="unit",
            converged=converged, correct=correct, iterations=10,
            balance_error=0.0 if correct else 1.0))
    return report


class TestReport:

    def test_pass_rate_counts_only_audited_answers(self):
        report = _report(("anderson", True, True), ("anderson", True, False))
        assert report.pass_rate == pytest.approx(0.5)

    def test_convergence_rate_is_the_solvers_own_verdict(self):
        """The gap between the two rates is the point of keeping both."""
        report = _report(("anderson", True, True), ("anderson", True, False))
        assert report.convergence_rate == pytest.approx(1.0)
        assert report.convergence_rate > report.pass_rate

    def test_empty_report_does_not_divide_by_zero(self):
        empty = Report()
        assert empty.pass_rate != empty.pass_rate  # NaN
        assert empty.convergence_rate != empty.convergence_rate

    def test_by_groups_pass_counts(self):
        report = _report(("anderson", True, True), ("none", False, False),
                         ("none", True, True))
        assert report.by("acceleration") == {"anderson": (1, 1), "none": (1, 2)}

    def test_mean_iterations_ignores_failures_by_default(self):
        """A failed solve used max_iter by definition; it measures the cap."""
        report = Report()
        report.outcomes.append(Outcome("a", "none", "unit", converged=True,
                                       correct=True, iterations=5))
        report.outcomes.append(Outcome("b", "none", "unit", converged=False,
                                       correct=False, iterations=100))
        assert report.mean_iterations() == {"none": 5.0}
        assert report.mean_iterations(only_passed=False) == {"none": 52.5}

    def test_failures_lists_everything_that_did_not_pass(self):
        report = _report(("anderson", True, True), ("none", True, False))
        assert [o.acceleration for o in report.failures()] == ["none"]

    def test_as_text_reports_both_rates_and_every_solve(self):
        report = _report(("anderson", True, True), ("none", True, False))
        text = report.as_text()
        assert "pass rate" in text
        assert "convergence rate" in text
        assert "WRONG" in text
        for outcome in report.outcomes:
            assert outcome.case in text

    def test_as_text_survives_an_infinite_balance_error(self):
        report = Report()
        report.outcomes.append(Outcome("diverged", "none", "unit",
                                       converged=False, correct=False,
                                       balance_error=float("inf")))
        assert "inf" in report.as_text()

    def test_to_csv_writes_one_row_per_solve(self, tmp_path):
        report = _report(("anderson", True, True), ("none", True, False))
        path = tmp_path / "bench.csv"
        report.to_csv(str(path))
        rows = path.read_text().strip().splitlines()
        assert len(rows) == 3  # header + 2
        assert "passed" in rows[0]


# =============================================================================
# The measurements the corpus was built to make
# =============================================================================

