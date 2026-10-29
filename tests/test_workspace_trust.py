"""The editor asks before a file runs code.

A flowsheet can carry code --- its code context, a ``.py`` script, and
functions or classes of yours named by import path. Opening one used to
run that code with no question asked; now it opens nothing until trusted,
and reports what it would have run so the editor can show it.
"""

import json
import textwrap

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from difflow import Flowsheet, Heater, HeaterParams, Unit, serialize
from difflow.workspace import FlowsheetSession

CONTEXT = "MARKER = 'ran'\n"


def plain(tmp_path, name="plain.json", context=None):
    """A flowsheet file, with a code context when given one."""
    fs = Flowsheet(species_order=["water"])
    fs.add_unit(Unit("h", Heater(HeaterParams(T_out=360.0)), ["a"], ["b"]))
    if context:
        fs.view = {"code_context": context}
    return serialize.save(fs, tmp_path / name)


@pytest.fixture
def user_module(tmp_path, monkeypatch):
    (tmp_path / "trustlab.py").write_text(textwrap.dedent('''
        from dataclasses import dataclass
        from difflow.params_mixin import ParamsMixin

        @dataclass
        class GainParams(ParamsMixin):
            gain: float = 1.0

        class Gain:
            def __init__(self, params: GainParams):
                self.params = params

            def __call__(self, inlet):
                return inlet
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    import trustlab
    return importlib.reload(trustlab)


class TestOpenFile:
    def test_a_file_with_code_is_not_opened_untrusted(self, tmp_path):
        path = plain(tmp_path, context=CONTEXT)
        session = FlowsheetSession()
        answer = session.open_file(str(path))
        assert answer["ok"] is False
        assert answer["needs_trust"]["code"] == CONTEXT
        assert "MARKER" not in session.bindings, "the code did not run"

    def test_trust_opens_it_and_runs_the_code(self, tmp_path):
        path = plain(tmp_path, context=CONTEXT)
        session = FlowsheetSession()
        assert session.open_file(str(path), trust=True)["ok"]
        assert session.bindings["MARKER"] == "ran"

    def test_a_file_without_code_opens_without_asking(self, tmp_path):
        assert FlowsheetSession().open_file(str(plain(tmp_path)))["ok"]

    def test_a_script_is_asked_about(self, tmp_path):
        script = tmp_path / "plant.py"
        script.write_text("raise SystemExit('this must not run')\n")
        answer = FlowsheetSession().open_file(str(script))
        assert answer["needs_trust"]["kind"] == "script"

    def test_a_module_the_file_names_is_listed(self, tmp_path, user_module):
        fs = Flowsheet(species_order=["water"])
        fs.add_unit(Unit("g", user_module.Gain(user_module.GainParams(2.0)),
                         ["a"], ["b"]))
        path = serialize.save(fs, tmp_path / "gain.json")
        answer = FlowsheetSession().open_file(str(path))
        assert answer["needs_trust"]["modules"] == ["trustlab"]

    def test_undo_after_a_trusted_open_still_imports(self, tmp_path, user_module):
        """Undo rebuilds from JSON; it must not need asking again."""
        fs = Flowsheet(species_order=["water"])
        fs.add_unit(Unit("g", user_module.Gain(user_module.GainParams(2.0)),
                         ["a"], ["b"]))
        session = FlowsheetSession()
        assert session.open_file(str(serialize.save(fs, tmp_path / "g.json")),
                                 trust=True)["ok"]
        assert session.patch_unit("g", {"params": {"gain": 3.0}})["ok"]
        assert session.undo()["ok"]


class TestStartup:
    def test_an_untrusted_file_is_held_not_loaded(self, tmp_path):
        path = plain(tmp_path, context=CONTEXT)
        session = FlowsheetSession(path=path)
        assert session.blocked["path"] == str(path.resolve())
        assert session.path is None, "Save must not write an empty canvas over it"
        assert session.document()["blocked"]["code"] == CONTEXT
        assert "MARKER" not in session.bindings

    def test_trusting_the_held_file_opens_it(self, tmp_path):
        path = plain(tmp_path, context=CONTEXT)
        session = FlowsheetSession(path=path)
        assert session.open_file(session.blocked["path"], trust=True)["ok"]
        assert session.blocked is None and session.bindings["MARKER"] == "ran"
        assert "blocked" not in session.document()

    def test_a_trusting_session_opens_it_directly(self, tmp_path):
        session = FlowsheetSession(path=plain(tmp_path, context=CONTEXT),
                                   trust=True)
        assert session.blocked is None and session.bindings["MARKER"] == "ran"


class TestWhatNeedsNoAsking:
    def test_a_file_saved_in_this_session(self, tmp_path):
        session = FlowsheetSession()
        assert session.set_code_context(CONTEXT)["ok"]
        assert session.save(str(tmp_path / "mine.json"))["ok"]
        assert session.open_file(str(tmp_path / "mine.json"))["ok"]

    def test_a_bundled_example(self):
        from difflow.workspace import examples

        key = examples.listing()[0]["key"]
        assert FlowsheetSession().open_example(key)["ok"]

    def test_a_document_with_code_sent_to_replace_is_asked_about(self, tmp_path):
        document = json.loads(plain(tmp_path, context=CONTEXT).read_text())
        session = FlowsheetSession()
        assert "needs_trust" in session.replace(document)
        assert session.replace(document, trust=True)["ok"]


class TestAgent:
    def test_without_exec_a_file_with_code_is_refused(self, tmp_path):
        """--no-exec blocked .py files, yet a .json's code context ran."""
        from difflow.agent.workbench import Workbench

        path = str(plain(tmp_path, context=CONTEXT))
        assert "needs_trust" in Workbench(allow_exec=False).open_file(path)
        assert Workbench(allow_exec=True).open_file(path)["ok"]
