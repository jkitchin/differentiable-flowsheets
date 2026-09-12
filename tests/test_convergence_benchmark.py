"""Tests for :mod:`difflow.convergence`, the pass-rate benchmark.

The benchmark's own correctness is what these pin, not the pass rate it
reports: that number is a measurement and is meant to move.  What must not
move is the machinery that produces it --- that a raised solve is counted
rather than propagated, that "the solver said it converged" and "the answer
audits" stay separate, and that no case is rigged so a strategy starts on
the answer.

The full grid is minutes of JAX compilation, so it is marked ``slow`` and
everything else here runs against the two cheap analytic cases or against
synthetic ones built in the test.
"""

import warnings

import jax.numpy as jnp
import pytest

from difflow.convergence import (
    ACCELERATIONS,
    CORPUS,
    INITIALIZATIONS,
    Case,
    Outcome,
    Report,
    _feed_guess,
    _product_streams,
    cases,
    get_case,
    run_benchmark,
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


class TestNoCaseIsRigged:
    """A strategy must not start on the answer it is being scored on.

    ``overshoot_loop`` originally used ``g(x) = 3F - 2x``, whose fixed point
    is ``x = F`` --- exactly the ``"feed"`` guess.  That strategy scored two
    free passes on a case it had not solved.
    """

    @pytest.mark.parametrize("case", CORPUS, ids=lambda c: c.name)
    def test_the_feed_guess_is_not_already_the_answer(self, case):
        fs = case.build()
        guess = _feed_guess(fs)
        # No single acceleration converges every case -- that is the
        # corpus's whole point -- so take the first that does.
        for acceleration in ("anderson", "wegstein", "none"):
            fs = case.build()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solved = fs.solve(acceleration=acceleration, tol=1e-10,
                                  max_iter=500, on_nonconvergence="ignore")
            if fs.last_solve_converged:
                break
        else:
            pytest.fail(f"{case.name} converges under no acceleration, so "
                        "its 'feed' guess cannot be checked against an answer")

        for dest, start in guess.items():
            source = next(s for s, d in fs.recycles.items() if d == dest)
            answer = solved[source]
            gap = max(
                abs(float(start[f"F_{s}"]) - float(answer[f"F_{s}"]))
                for s in fs.species_order)
            scale = max(
                max(abs(float(answer[f"F_{s}"])) for s in fs.species_order),
                1e-12)
            assert gap / scale > 1e-6, (
                f"{case.name}: the 'feed' guess for {dest!r} is already the "
                "converged answer, so that strategy is not being tested")


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

    def test_the_control_case_passes(self):
        outcome = run_case(get_case("two_phase_flash"), acceleration="anderson")
        assert outcome.passed, outcome
        assert outcome.method == "anderson"
        assert outcome.iterations is not None

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

    def test_a_check_that_raises_makes_the_answer_wrong(self):
        def bad_check(fs, streams):
            raise ValueError("cannot audit")

        case = Case(name="unauditable", build=get_case("two_phase_flash").build,
                    difficulty="A check that raises, to confirm an "
                               "unauditable answer counts as wrong.",
                    check=bad_check)
        outcome = run_case(case)
        assert not outcome.correct
        assert outcome.balance_error == float("inf")
        assert "check raised" in outcome.error

    def test_non_convergence_does_not_warn_through(self):
        """Counting non-convergence is the job; announcing each one is not."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            run_case(get_case("overshoot_loop"), acceleration="none")

    def test_an_unknown_initialization_is_rejected(self):
        with pytest.raises(ValueError, match="unknown initialization"):
            run_case(get_case("two_phase_flash"), initialization="telepathy")

    @pytest.mark.parametrize("initialization", INITIALIZATIONS)
    def test_every_initialization_strategy_runs(self, initialization):
        outcome = run_case(get_case("two_phase_flash"),
                           initialization=initialization)
        assert outcome.passed, outcome


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


class TestRunBenchmark:

    def test_it_covers_the_whole_grid(self):
        report = run_benchmark(cases=["two_phase_flash"],
                               accelerations=("anderson", "wegstein"),
                               initializations=("unit", "default"))
        assert len(report.outcomes) == 4
        assert {o.acceleration for o in report.outcomes} == {"anderson", "wegstein"}
        assert {o.initialization for o in report.outcomes} == {"unit", "default"}

    def test_it_accepts_case_objects_as_well_as_names(self):
        report = run_benchmark(cases=[get_case("two_phase_flash")],
                               accelerations=("anderson",),
                               initializations=("unit",))
        assert len(report.outcomes) == 1

    def test_the_report_carries_the_settings_it_ran_under(self):
        report = run_benchmark(cases=["two_phase_flash"],
                               accelerations=("anderson",),
                               initializations=("unit",),
                               tol=1e-6, max_iter=42)
        assert report.tol == 1e-6
        assert report.max_iter == 42


# =============================================================================
# The measurements the corpus was built to make
# =============================================================================

class TestTheFindings:
    """These pin *why* each hard case is in the corpus.

    They assert the qualitative finding, not the exact iteration count: the
    point of a benchmark is that its numbers move.
    """

    def test_anderson_rescues_a_divergent_map_that_substitution_cannot(self):
        divergent = run_case(get_case("overshoot_loop"), acceleration="none")
        accelerated = run_case(get_case("overshoot_loop"), acceleration="anderson")
        assert not divergent.passed
        assert accelerated.passed

    def test_anderson_is_not_a_free_win(self):
        """phase_coupled_flash is in the corpus precisely because it isn't."""
        anderson = run_case(get_case("phase_coupled_flash"),
                            acceleration="anderson", initialization="unit")
        wegstein = run_case(get_case("phase_coupled_flash"),
                            acceleration="wegstein", initialization="unit")
        assert wegstein.passed
        assert not anderson.passed

    def test_anderson_solves_a_disjunction_without_binaries(self):
        """The negative result #251 most needs.

        ``regime_switch`` is a unit picking between two linear branches with
        the answer exactly on the boundary -- the disjunction that issue
        proposes handing to a MILP's binaries. Anderson lands on it in two
        iterations.
        """
        assert run_case(get_case("regime_switch"),
                        acceleration="anderson").passed

    def test_a_signed_tear_beats_both_accelerated_methods(self):
        """The one case that inverts the corpus ranking.

        ``clip_negative_flows`` defaults to True and is applied by the
        Wegstein and Anderson paths but not by the unaccelerated one. On a
        tear whose answer is genuinely negative, that projection is the
        difference between solving it and not.
        """
        plain = run_case(get_case("signed_tear"), acceleration="none")
        wegstein = run_case(get_case("signed_tear"), acceleration="wegstein")
        anderson = run_case(get_case("signed_tear"), acceleration="anderson")
        assert plain.passed, "plain substitution should solve this one"
        assert not wegstein.passed
        assert not anderson.passed

    def test_turning_the_clip_off_rescues_the_signed_tear(self):
        """Pins the remedy the case's `difficulty` claims."""
        clipped = run_case(get_case("signed_tear"), acceleration="anderson")
        unclipped = run_case(get_case("signed_tear"), acceleration="anderson",
                             clip_negative_flows=False)
        assert not clipped.passed
        assert unclipped.passed
        assert unclipped.iterations < 30

    def test_a_rigorous_column_in_a_recycle_is_not_the_hard_part(self):
        """#251 names near-pinch columns; every setting solves this one."""
        for acceleration in ACCELERATIONS:
            outcome = run_case(get_case("column_recycle"),
                               acceleration=acceleration)
            assert outcome.passed, outcome

    def test_the_multi_loop_case_really_has_two_tears(self):
        """Otherwise it is not testing what it says it tests."""
        fs = get_case("two_loop_recycle").build()
        assert len(fs.recycles) == 2

    @pytest.mark.slow
    def test_an_absolute_tear_tolerance_can_accept_a_wrong_answer(self):
        """The trap ``trace_recycle`` exists to expose.

        ``tol`` is an absolute residual on the tear.  On a loop of gain
        ``g``, a step of ``d`` means a remaining error of ``d / (1 - g)``,
        so at ``g = 0.97`` the residual understates the error by about 33x.
        With a tear whose converged value is ~3e-5, Wegstein stops on a
        residual inside 1e-8 while still about 1% out --- reporting
        convergence and meaning it, on the solver's own terms.
        """
        outcome = run_case(get_case("trace_recycle"), acceleration="wegstein",
                           initialization="unit")
        assert outcome.converged is True, "the solver should claim success"
        assert not outcome.correct, "and the audit should disagree"
        assert outcome.verdict == "WRONG"

    @pytest.mark.slow
    def test_the_solver_could_have_known_it_was_wrong(self):
        """#264: the same solve, with the error measured rather than assumed.

        The audit is the case's own analytic answer and is available only
        because someone wrote it down. The error estimate is measured from
        the flowsheet itself, and it agrees --- which is what makes it
        usable on a flowsheet nobody has an analytic answer for.
        """
        outcome = run_case(get_case("trace_recycle"), acceleration="wegstein",
                           initialization="unit")
        assert outcome.verdict == "WRONG"
        assert outcome.gain == pytest.approx(0.97, abs=1e-6)
        # The tear settles near 3.33e-5, so a relative audit error of ~9e-3
        # is an absolute ~3e-7 -- what the estimate reads, from a residual
        # of 9.3e-9.
        assert outcome.error_estimate == pytest.approx(3.0e-7, rel=0.05)
        assert outcome.error_estimate > 30 * outcome.residual

    @pytest.mark.slow
    def test_the_signed_tear_amplification_was_the_reference_not_the_solver(self):
        """#264: ``signed_tear`` was measuring a rounded constant.

        Its ``tol`` was loosened to 1e-4 for a 1.2e-6 error attributed to
        ``M`` being non-normal. The reference was written to six decimals
        and is itself 4.4e-6 from the answer; plain substitution lands
        7.6e-9 away, *inside* its own tear residual rather than 450 times
        it. With the reference at full precision the case audits at 1e-6
        like every other analytic one.
        """
        outcome = run_case(get_case("signed_tear"), acceleration="none",
                           initialization="unit")
        assert outcome.passed
        assert outcome.balance_error < 1e-8
        assert get_case("signed_tear").tol == 1e-6
        # Measured amplification below one: the error is smaller than the
        # step, not 450 times it.
        assert outcome.error_estimate < outcome.residual


@pytest.fixture(scope="module")
def full_grid() -> Report:
    """The whole corpus, once.

    Several minutes of JAX compilation, so it is shared rather than rebuilt
    per test.  Safe to share: a :class:`Report` is read-only once returned,
    and each solve inside it built its own flowsheet.
    """
    return run_benchmark()


@pytest.mark.slow
class TestFullGrid:

    def test_it_covers_every_combination(self, full_grid):
        assert len(full_grid.outcomes) == (
            len(CORPUS) * len(ACCELERATIONS) * len(INITIALIZATIONS))

    def test_the_corpus_measures_something(self, full_grid):
        """All-pass or all-fail would mean the corpus is mistuned."""
        assert 0.0 < full_grid.pass_rate < 1.0

    def test_no_case_raises_at_any_setting(self, full_grid):
        """A raise is a harness-visible outcome, but not one we expect here."""
        raised = [o for o in full_grid.outcomes if o.error]
        assert not raised, [o.error for o in raised]

    def test_every_case_passes_somewhere(self, full_grid):
        """A case no setting solves cannot discriminate between settings."""
        for case, (passed, _) in full_grid.by("case").items():
            assert passed > 0, f"{case} fails under every setting"

    def test_the_report_renders(self, full_grid):
        text = full_grid.as_text()
        for case in CORPUS:
            assert case.name in text
