"""Tests for the assistant's brief: difflow.gui.context and docs_index.

The assistant panel's value is not the model, it is what the model is
handed. So these tests are about the brief itself --- that it contains
the parameter values the flowsheet actually holds, that a failed solve
brings its own diagnostics and the troubleshooting card for the symptom,
that retrieval finds the right chapter, and that a brief too long for a
3B model's window is trimmed from the bottom rather than truncated at
the end, where the question is.

No language model is involved anywhere here, which is the point: the
half of the feature difflow is responsible for is testable on its own.
"""

import json

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Flowsheet,
    IdealThermo,
    Mixer,
    Unit,
    get_species_data,
    make_stream,
    mass_action_kinetics,
)
from difflow.gui import context, docs_index
from difflow.gui.session import FlowsheetSession

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def session():
    """A recycle flowsheet, solved, in a session."""
    thermo = IdealThermo({n: get_species_data(n) for n in SPECIES})
    kin = mass_action_kinetics([{
        "equation": "water -> ethanol",
        "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
        "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
    }], SPECIES)
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                    T=350.0, P=101325.0))
    fs.add_unit(Unit("mix", Mixer(SPECIES, thermo), ["feed", "recycle"],
                     ["mixed"]))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=1.5, molar_density=1000.0, **kin.params_kwargs())),
        ["mixed"], ["rx"]))
    fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                     ["rx"], ["liq", "vap"]))
    fs.add_recycle("vap", "recycle")
    session = FlowsheetSession(flowsheet=fs)
    session.solve()
    return session


# -- the docs index ---------------------------------------------------

class TestDocsIndex:
    """The build-time retrieval index over ``docs/``."""

    def test_headings_inside_code_fences_are_not_sections(self):
        # Every docs file is a tutorial full of Python, and a `# Build
        # the flowsheet` comment inside a fence is a comment. Counting
        # naively turns one chapter into a hundred one-line "sections".
        text = ("# Real heading\nprose\n\n```python\n"
                "# Not a heading\nx = 1\n```\n\nmore prose\n")
        sections = docs_index.split(text, "x.md")
        assert [s["heading"] for s in sections] == ["Real heading"]
        assert "# Not a heading" in sections[0]["text"]

    def test_a_section_carries_its_breadcrumb(self):
        text = "# Chapter\nintro\n\n## Part\nbody\n\n### Gotchas\ncareful\n"
        by_heading = {s["heading"]: s for s in docs_index.split(text, "x.md")}
        assert by_heading["Gotchas"]["path"] == "Chapter > Part > Gotchas"
        assert by_heading["Gotchas"]["anchor"] == "gotchas"

    def test_a_long_section_is_cut_at_a_paragraph(self):
        body = ("para\n\n" * 900)
        section = docs_index.split(f"# H\n{body}", "x.md")[0]
        assert len(section["text"]) <= docs_index.MAX_SECTION
        assert section["text"].endswith("para")

    def test_the_committed_index_is_what_the_docs_say(self):
        # The index is built output, committed like the JS bundle, and
        # this is the check that it has not drifted from the prose.
        committed = json.loads(docs_index.INDEX.read_text())
        fresh = docs_index.build()
        assert committed["sections"] == fresh["sections"], (
            "docs/ has changed since the index was built; "
            "run `make gui-build` (or python src/difflow/gui/docs_index.py)"
        )

    def test_search_finds_the_chapter_that_is_about_the_question(self):
        index = docs_index.load()
        assert index is not None
        hits = docs_index.search(index, "delta vectors trust region planning")
        assert hits[0]["source"] == "planning.md"
        assert hits[0]["score"] > 0

    def test_a_question_of_pure_stopwords_retrieves_nothing(self):
        # Rather than retrieving whatever happens to be longest.
        assert docs_index.search(docs_index.load(), "how do the and") == []

    def test_a_missing_index_is_not_an_error(self, tmp_path):
        assert docs_index.load(tmp_path / "nope.json") is None


# -- the briefs -------------------------------------------------------

class TestBlockPack:

    def test_it_carries_the_values_this_unit_actually_holds(self, session):
        pack = context.block(session, name="reactor", question="volume")
        text = pack.prompt()
        assert "V [m^3] = 1.5" in text          # the value, and its units
        assert "Operation: CSTR" in text

    def test_a_parameter_it_cannot_print_is_named_not_dropped(self, session):
        # Same rule as the published page: a parameter that vanishes
        # reads as a parameter the unit does not have.
        assert "rate_fn = <function>" in context.block(
            session, name="reactor").prompt()

    def test_it_carries_the_equations_and_the_docstring(self, session):
        text = context.block(session, name="reactor").prompt()
        assert "Equations (LaTeX)" in text
        assert "Continuous Stirred Tank Reactor" in text

    def test_an_operation_with_no_node_is_answered_from_the_catalog(self, session):
        pack = context.block(session, operation="Heater", question="duty")
        assert "Operation: Heater" in pack.prompt()
        assert "Parameters of Heater" in pack.prompt()

    def test_a_name_nothing_is_called_says_so(self, session):
        pack = context.block(session, name="nonesuch")
        assert "No unit called 'nonesuch'" in pack.prompt()


class TestFlowsheetPack:

    def test_it_names_the_units_the_wiring_and_the_recycle(self, session):
        text = context.flowsheet(session, "what is this?").prompt()
        assert "reactor (CSTR): mixed -> rx" in text
        assert "vap is recycled to recycle" in text
        assert "water, ethanol" in text

    def test_it_carries_the_generated_python(self, session):
        assert "difflow.codegen" in context.flowsheet(session).prompt()


class TestSolvePack:

    def test_it_reports_the_diagnostics_nothing_else_reports(self, session):
        text = context.solve(session, "how did that go?").prompt()
        assert "method:" in text and "residual:" in text
        assert "tear streams: recycle" in text

    def test_an_unsolved_flowsheet_says_so_rather_than_inventing(self, session):
        fresh = FlowsheetSession(flowsheet=Flowsheet(species_order=SPECIES))
        assert "has not been solved" in context.solve(fresh).prompt()

    def test_the_card_matches_the_symptom(self, session):
        text = context.solve(session, "I get a NaN in the gradient").prompt()
        assert "Troubleshooting: NaN in the result or the gradient" in text
        assert "TracerArrayConversionError" not in text

    def test_a_solve_that_worked_does_not_get_the_convergence_card(self, session):
        text = context.solve(session, "what units are in this?").prompt()
        assert "recycle did not converge" not in text

    def test_a_raised_solve_puts_the_traceback_first(self, session):
        session.solve_error = "ValueError: boom"
        try:
            pack = context.solve(session, "why?")
            assert pack.sections[0].title == "It raised"
            assert "boom" in pack.prompt()
        finally:
            session.solve_error = None


class TestPlanningPack:

    @pytest.fixture(scope="class")
    def dvs(self):
        from difflow.planning import chain
        from difflow.planning.export import DeltaVectorSet

        return DeltaVectorSet.from_result(
            chain.two_plant_chain().planner(radius=0.3).solve())

    def test_the_jacobian_is_readable_as_a_table(self, dvs):
        pack = context.planning(dvs, "which lever moves the margin?")
        table = next(s for s in pack.sections
                     if s.title.startswith("Delta vectors for block 'ngl'"))
        # Names are stripped of the block prefix that is already in the
        # heading, so the columns fit rather than truncating.
        assert "ethane_recovery" in table.body
        assert "ngl.ethane_recovery" not in table.body
        assert "Base case:" in table.body

    def test_it_says_where_the_linear_model_stops_being_the_flowsheet(self, dvs):
        assert "trust radius" in context.planning(dvs).prompt()

    def test_only_binding_marginals_are_quoted(self, dvs):
        text = context.planning(dvs).prompt()
        if "Shadow prices" in text:
            rows = [line for line in text.splitlines()
                    if line.startswith("- ") and ": " in line]
            assert not any(line.endswith(": 0") for line in rows)

    def test_a_constraint_reads_as_the_lp_holds_it(self, dvs):
        if dvs.specs:
            assert "ngl.T_colfeed <= 236.0" in context.planning(dvs).prompt()


# -- the budget -------------------------------------------------------

class TestBudget:

    def test_the_cheapest_sections_go_first(self):
        sections = [context.Section("keep", "x" * 400, priority=9),
                    context.Section("maybe", "y" * 400, priority=5),
                    context.Section("drop", "z" * 400, priority=1)]
        kept, notes = context.fit(sections, budget=250)
        assert [s.title for s in kept] == ["keep", "maybe"]
        assert "dropped 'drop'" in notes[0]

    def test_the_subject_is_never_dropped(self):
        sections = [context.Section("subject", "x" * 8000, priority=9)]
        kept, _ = context.fit(sections, budget=10)
        assert [s.title for s in kept] == ["subject"]

    def test_order_survives_trimming(self):
        sections = [context.Section(str(i), "x" * 300, priority=i)
                    for i in range(6)]
        kept, _ = context.fit(sections, budget=250)
        assert [s.title for s in kept] == sorted(
            (s.title for s in kept), key=int)

    def test_a_real_pack_fits_the_window(self, session):
        for kind in ("flowsheet", "block", "solve"):
            pack = session.context(kind=kind, name="reactor",
                                   question="what is going on here?")
            assert pack["tokens"] <= context.BUDGET + 200, kind


# -- the dispatch -----------------------------------------------------

class TestPack:

    def test_every_kind_answers(self, session):
        for kind in ("flowsheet", "block", "solve"):
            answer = session.context(kind=kind, name="reactor", question="?")
            assert answer["ok"] and answer["prompt"]
            assert answer["sections"] and answer["system"]

    def test_planning_says_what_is_missing_rather_than_failing(self, session):
        answer = session.context(kind="planning")
        assert answer["ok"] is False
        assert "Planning panel" in answer["error"]

    def test_an_unknown_kind_is_an_answer_not_an_exception(self, session):
        answer = session.context(kind="astrology")
        assert answer["ok"] is False
        assert "astrology" in answer["error"]

    def test_the_prompt_is_the_sections_plus_the_question(self, session):
        answer = session.context(kind="flowsheet", question="how many units?")
        assert answer["prompt"].endswith("how many units?")
        for section in answer["sections"]:
            assert section["title"] in answer["prompt"]
