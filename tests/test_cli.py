"""The ``difflow`` command's dispatch.

Only the routing is tested here --- each subcommand has its own tests
(``test_report.py``, ``test_planning_export.py``, the book app's own), and
repeating them through the front door would only assert that argparse
still works.
"""

import pytest

from difflow import cli


class FakeEntry:
    """A stand-in for the book app's ``difflow.apps`` entry point."""

    module = "difflow_app.server"

    def __init__(self, calls):
        self.calls = calls

    def load(self):
        return lambda argv: self.calls.append(argv) or 0


@pytest.fixture
def installed(monkeypatch):
    """The editor installed: calls to it are recorded rather than served."""
    calls = []
    monkeypatch.setattr("difflow.apps.find",
                        lambda name="gui": FakeEntry(calls) if name == "gui" else None)
    return calls


@pytest.fixture
def missing(monkeypatch):
    monkeypatch.setattr("difflow.apps.find", lambda name="gui": None)


class TestTheEditorIsAnApp:
    def test_no_arguments_open_it_when_installed(self, installed):
        assert cli.main([]) == 0
        assert installed == [[]]

    def test_gui_passes_its_arguments_through(self, installed):
        assert cli.main(["gui", "plant.json", "--port", "9000"]) == 0
        assert installed == [["plant.json", "--port", "9000"]]

    def test_no_arguments_print_usage_without_it(self, missing, capsys):
        assert cli.main([]) == 0
        assert "usage: difflow" in capsys.readouterr().out

    def test_gui_says_where_the_editor_comes_from(self, missing, capsys):
        assert cli.main(["gui"]) == 2
        assert "difflow book" in capsys.readouterr().err

    def test_the_agent_tool_says_so_too(self, missing, tmp_path):
        from difflow.agent import Workbench

        answer = Workbench().open_in_editor(path=str(tmp_path / "x.json"))
        assert not answer["ok"] and "difflow book" in answer["error"]


class TestDispatch:
    @pytest.mark.parametrize("name, module", [
        ("report", "difflow.report.cli"),
        ("plan-export", "difflow.planning.cli"),
    ])
    def test_every_command_names_something_importable(self, name, module):
        import importlib

        target, attr = cli.COMMANDS[name]
        assert target == module
        assert callable(getattr(importlib.import_module(target), attr))

    def test_a_bare_script_still_means_report(self, tmp_path, capsys):
        # `difflow model.py` predates the subcommands and is what the
        # older docs say; it has to keep working.
        script = tmp_path / "model.py"
        script.write_text(
            "from difflow import Flowsheet, Unit, Splitter\n"
            "from difflow.streams import make_stream\n"
            "fs = Flowsheet(['methane', 'ethane'])\n"
            "fs.add_feed('feed', make_stream({'methane': 1.0, 'ethane': 0.5},"
            " 300, 101325))\n"
            "fs.add_unit(Unit('split', Splitter(['methane', 'ethane']),\n"
            "                 ['feed'], ['top', 'bot'],"
            " params={'split_frac': 0.6}))\n"
        )
        assert cli.main([str(script), "--no-git"]) == 0
        assert "# Flowsheet Report" in capsys.readouterr().out

    def test_a_typo_is_reported_as_one(self, capsys):
        # The fall-through to `report` is narrow on purpose: handing
        # `frobnicate` to the report parser would fail on a missing
        # script three frames later and never mention the real commands.
        assert cli.main(["frobnicate"]) == 2
        err = capsys.readouterr().err
        assert "no such command or file" in err
        assert "gui, mcp, plan-export, report" in err


class TestHelp:
    def test_help_lists_every_command(self, capsys):
        assert cli.main(["--help"]) == 0
        out = capsys.readouterr().out
        for name in cli.COMMANDS:
            assert name in out

    def test_version_is_the_installed_one(self, capsys):
        import difflow

        assert cli.main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == difflow.__version__

    def test_the_subcommand_help_calls_itself_by_the_right_name(self, capsys):
        # A subcommand's help printing "usage: difflow.debug.cli" sends the
        # reader to an invocation they did not type.
        with pytest.raises(SystemExit):
            cli.main(["debug", "--help"])
        assert capsys.readouterr().out.startswith("usage: difflow debug")
