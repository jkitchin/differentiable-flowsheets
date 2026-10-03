"""The editor's Examples menu: every example opens, and solves."""

import warnings

import pytest

from difflow.gui import TOKEN_HEADER, FlowsheetSession, examples
from tests.test_gui import Client

KEYS = [e["key"] for e in examples.listing()]


def test_there_are_examples():
    assert KEYS


@pytest.mark.parametrize("key", KEYS)
def test_an_example_opens_and_solves_cleanly(key):
    """A warning counts as a failure: an example is what a new user sees
    first, and one that opens with a DefaultCpWarning teaches them to
    ignore warnings."""
    session = FlowsheetSession()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert session.open_example(key) == {"ok": True}
        assert session.context_error is None
        answer = session.solve()
    assert answer["ok"], answer.get("error")
    assert session.pending == {}


@pytest.mark.parametrize("entry", examples.listing(), ids=KEYS)
def test_an_example_says_what_it_is(entry):
    assert entry["title"] != entry["key"]
    assert entry["description"]


def test_opening_an_example_forgets_the_file(tmp_path):
    """Save must not write the example over the file the editor was opened on."""
    session = FlowsheetSession(path=tmp_path / "plant.json")
    session.open_example(KEYS[0])
    assert session.path is None
    assert not session.save()["ok"]
    assert not (tmp_path / "plant.json").exists()


def test_an_unknown_example_is_refused_and_changes_nothing():
    session = FlowsheetSession()
    before = session.flowsheet
    answer = session.open_example("../../pyproject")
    assert answer["ok"] is False
    assert session.flowsheet is before


def test_an_example_that_fails_to_load_keeps_the_open_file(tmp_path, monkeypatch):
    """A refused example must not forget the file, its undo or its edits."""
    session = FlowsheetSession(path=tmp_path / "plant.json")
    session.open_example(KEYS[0])
    session.path = tmp_path / "plant.json"
    assert session.set_solver_options({"tol": 1e-6})["ok"]
    dirty, path = session.dirty, session.path
    monkeypatch.setattr(examples, "document", lambda key: {"units": "garbage"})
    answer = session.open_example(KEYS[0])
    assert answer["ok"] is False
    assert session.path == path
    assert session.dirty == dirty
    assert session.undo()["ok"]


def test_the_routes():
    live = Client(FlowsheetSession())
    try:
        status, listing = live.get_json("/api/examples")
        assert status == 200
        assert [e["key"] for e in listing["examples"]] == KEYS

        status, answer = live.post("/api/examples/open", {"key": KEYS[-1]})
        assert (status, answer) == (200, {"ok": True})
        status, doc = live.get_json("/api/flowsheet")
        assert doc["flowsheet"]["view"]["title"] == listing["examples"][-1]["title"]
        assert doc["path"] == ""
    finally:
        live.close()


def test_opening_needs_the_token():
    live = Client(FlowsheetSession())
    try:
        status, _ = live.post("/api/examples/open", {"key": KEYS[0]},
                              headers={TOKEN_HEADER: "wrong"})
        assert status == 403
    finally:
        live.close()
