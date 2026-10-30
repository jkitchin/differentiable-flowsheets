"""FlowsheetSession (difflow.workspace) without a server.

The session is the engine the agent tools drive, and the book's browser
app serves. These are the tests that need no socket: files, how a failure
is reported, the results a solve carries, the console's figures, links
from an operation to the book, and the debugger's record and replay. The
routes, the page and the HTTP plumbing are tested with the app.
"""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import (
    CSTR,
    CSTRParams,
    Flash,
    FlashParams,
    Flowsheet,
    Heater,
    HeaterParams,
    IdealThermo,
    Mixer,
    Unit,
    get_species_data,
    make_stream,
    mass_action_kinetics,
    serialize,
)
from difflow.workspace import FlowsheetSession, console, doclinks

SPECIES = ["water", "ethanol"]


@pytest.fixture(scope="module")
def thermo():
    return IdealThermo({n: get_species_data(n) for n in SPECIES})


def build_flowsheet(thermo, V=1.0):
    """A reactor with a data-built rate law, then a flash."""
    kin = mass_action_kinetics([{
        "equation": "water -> ethanol",
        "reactants": {"water": 1.0}, "products": {"ethanol": 1.0},
        "rate_params": {"A": 1.0e3, "Ea": 40_000.0, "n": 0.0},
    }], SPECIES)
    fs = Flowsheet(species_order=SPECIES)
    fs.add_feed("feed", make_stream(
        {"water": 1.0, "ethanol": 0.1}, T=350.0, P=101325.0
    ))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=V, molar_density=1000.0, **kin.params_kwargs()
    )), ["feed"], ["rx"]))
    fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                     ["rx"], ["liq", "vap"]))
    return fs


class TestFiles:
    def test_a_session_loads_a_flowsheet_from_a_path(self, thermo, tmp_path):
        path = tmp_path / "plant.json"
        serialize.save(build_flowsheet(thermo), path)

        session = FlowsheetSession(path=path)
        assert [u.name for u in session.flowsheet.units] == ["reactor", "flash"]

    def test_save_writes_a_readable_file(self, thermo, tmp_path):
        path = tmp_path / "plant.json"
        session = FlowsheetSession(build_flowsheet(thermo), path)
        assert session.save()["ok"]
        assert [u.name for u in serialize.load(path).units] == ["reactor", "flash"]

    def test_save_without_a_path_is_refused_not_raised(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        result = session.save()
        assert not result["ok"] and "Save As" in result["error"]

class TestFailureReporting:
    def test_a_failing_solve_is_reported_not_raised(self, thermo):
        """The browser needs the message; a traceback at the socket is useless."""
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0}, T=350.0, P=101325.0))
        # a Heater takes one inlet; two is a modelling error, not a crash
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)),
                         ["feed", "recycle"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("liq", "recycle")

        result = FlowsheetSession(fs).solve()
        assert not result["ok"]
        assert result["error"], "a failure must carry a message"

    def test_an_unregistered_unit_makes_the_code_panel_report(self, thermo):
        class HomeMadeUnit:
            params = None

        fs = build_flowsheet(thermo)
        fs.add_unit(Unit("mystery", HomeMadeUnit(), ["rx"], ["out"]))
        assert "registry" in FlowsheetSession(fs).code()["error"]


class TestResults:
    """The solve reports how it solved, not only what it found."""

    def test_a_solve_carries_its_diagnostics(self, thermo):
        answer = FlowsheetSession(build_flowsheet(thermo)).solve()
        assert answer["ok"]
        # No recycle here, so this is the sequential path -- and saying so
        # is the point: the panel must be able to tell the two apart.
        assert answer["method"] == "direct"
        assert answer["converged"] is True
        assert answer["tear_streams"] == []
        assert answer["residual"] == 0.0
        assert answer["tol"] is not None
        assert answer["species"] == SPECIES

    def test_an_edit_makes_the_last_solve_stale(self, thermo):
        session = FlowsheetSession(build_flowsheet(thermo))
        assert session.solve()["ok"]
        assert session.levers()["solved"] is True
        assert session.patch_unit("reactor", {"params": {"V": 2.0}})["ok"]
        # The streams on screen describe the flowsheet as it was.
        assert session.levers()["solved"] is False
        assert session.levers()["outputs"] == []

class TestConsoleFigures:
    """The seam plots will arrive through, exercised without matplotlib."""

    def test_a_broken_hook_does_not_eat_the_result(self):
        """A renderer that fails must not lose a number computed correctly."""
        def broken(ns):
            raise RuntimeError("no display")

        answer = console.Console(display_hooks=[broken]).run("6 * 7")
        assert answer["error"] is None
        assert answer["outputs"][0]["text"] == "42"
        assert "no display" in answer["outputs"][1]["text"]
        assert answer["outputs"][1]["stream"] == "stderr"

class TestWhereTheBookTalksAboutAUnit:
    """Every palette entry offers the documentation for that unit.

    Resolved against ``static/docs-index.json`` rather than written down:
    a hand-kept table of 87 operations against prose that gets
    reorganised is wrong within a release, and wrong silently, because a
    link to a renamed heading still returns 200 and lands at the top of
    the page. :mod:`tests.test_doclinks` is where the resolution rules
    themselves are pinned; what is here is the editor's side of it.
    """

    def test_a_unit_with_its_own_section_lands_on_it(self):
        url = doclinks.url_for("CSTR")
        assert url.startswith(doclinks.BASE_URL)
        assert "unit-operations-chemical.html#" in url

    def test_a_name_the_book_never_uses_gets_no_link(self):
        """Reported as an absence rather than papered over with a home page."""
        assert doclinks.url_for("NotAnOperationAtAll") is None
        assert doclinks.url_for("") is None

    def test_a_missing_index_is_no_links_rather_than_no_editor(self, monkeypatch):
        """A source checkout that has never run the front-end build.

        Patched at :func:`docs_index.load` rather than at its ``INDEX``
        constant: the path is a default argument, bound once at
        definition, so rebinding the module attribute would leave the
        real index still being read and this test passing for no reason.
        """
        from difflow.workspace import doclinks as dl

        monkeypatch.setattr(dl.docs_index, "load", lambda *a, **k: None)
        dl.resolve.cache_clear()
        try:
            assert dl.url_for("CSTR") is None
        finally:
            dl.resolve.cache_clear()




class TestTheDebuggerOnTheSession:
    def test_a_what_if_replays_the_recording(self, thermo):
        fs = Flowsheet(species_order=SPECIES)
        fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                        T=350.0, P=101325.0))
        fs.add_unit(Unit("mix", Mixer(SPECIES, thermo), ["feed", "recycle"], ["mixed"]))
        fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0)), ["mixed"], ["hot"]))
        fs.add_unit(Unit("flash", Flash(FlashParams(species_order=SPECIES), thermo),
                         ["hot"], ["liq", "vap"]))
        fs.add_recycle("vap", "recycle")
        session = FlowsheetSession(fs)
        assert "record a solve first" in session.debug_whatif(0, "recycle", "T", 1)["error"]

        trace = session.debug_record()["trace"]
        k = next(i for i, f in enumerate(trace["frames"]) if f["checkpoint"] == "iteration")
        answer = session.debug_whatif(k, "recycle", "T", 361.0)
        assert answer["ok"], answer
        assert answer["trace"]["result"]["diverged_at"] is None
        assert not session.debug_whatif(0, "recycle", "T", 1)["ok"]       # not an iteration
        assert not session.debug_whatif(k, "recycle", "T", "warm")["ok"]

        assert session.patch_unit("heat", {"params": {"T_out": 365.0}})["ok"]
        stale = session.debug_whatif(k, "recycle", "T", 361.0)
        assert not stale["ok"] and "changed since" in stale["error"]

