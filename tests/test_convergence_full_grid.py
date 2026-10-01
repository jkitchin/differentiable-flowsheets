"""The whole convergence corpus, solved once: every case x acceleration x init.

Split out of ``test_convergence_benchmark.py`` because the fixture below is
the longest single piece of work in the per-commit suite, and CI runs with
``--dist loadfile``: a file goes to one worker whole.  In one file with the
rest of the benchmark's tests it ran alone at the end of its shard for ten
minutes after every other worker had finished.  As its own module it runs
alongside them.
"""

import pytest

from difflow.convergence import (
    ACCELERATIONS,
    CORPUS,
    INITIALIZATIONS,
    Report,
    run_benchmark,
)


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
