"""The editor's Examples menu: every example opens (opened, not solved)."""

import warnings

import pytest

from difflow.workspace import FlowsheetSession, examples

KEYS = [e["key"] for e in examples.listing()]


@pytest.mark.parametrize("key", KEYS)
def test_an_example_opens_cleanly(key):
    session = FlowsheetSession()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert session.open_example(key) == {"ok": True}
    assert session.context_error is None
    assert session.pending == {}
    assert session.flowsheet.units


def test_opening_an_example_forgets_the_file(tmp_path):
    session = FlowsheetSession(path=tmp_path / "plant.json")
    session.open_example(KEYS[0])
    assert session.path is None
    assert not session.save()["ok"]


def test_an_unknown_example_is_refused_and_changes_nothing():
    session = FlowsheetSession()
    before = session.flowsheet
    assert session.open_example("../../pyproject")["ok"] is False
    assert session.flowsheet is before


