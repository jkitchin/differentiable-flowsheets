"""The ``difflow`` command's dispatch.

Only the routing is tested here --- each subcommand has its own tests
(``test_gui.py``, ``test_report.py``, ``test_planning_export.py``), and
repeating them through the front door would only assert that argparse
still works.
"""

import pytest

from difflow import cli


@pytest.fixture
def recorded_gui(monkeypatch):
    """Capture what ``difflow gui`` would serve, without serving it."""
    calls = []
    monkeypatch.setattr("difflow.gui.server.serve",
                        lambda **kw: calls.append(kw))
    return calls


class TestTheEditorIsTheDefault:
    def test_no_arguments_opens_the_editor(self, recorded_gui):
        # A bare `difflow` used to be a usage error from the report
        # parser, which is a poor thing to greet someone with. The
        # editor is what a new user wants and has nothing to print.
        from difflow.gui.server import DEFAULT_PORT

        assert cli.main([]) == 0
        assert recorded_gui == [{"path": None, "port": DEFAULT_PORT,
                                 "open_browser": True, "token": None,
                                 "stay": False}]

    def test_gui_takes_its_own_arguments(self, recorded_gui):
        assert cli.main(["gui", "plant.json", "--port", "9000",
                         "--no-browser"]) == 0
        assert recorded_gui == [{"path": "plant.json", "port": 9000,
                                 "open_browser": False, "token": None,
                                 "stay": False}]

    def test_stay_is_passed_through(self, recorded_gui):
        # Without it the editor stops when its last page closes, which
        # is what someone at a terminal wants and what someone running
        # it as a service very much does not.
        assert cli.main(["gui", "--stay"]) == 0
        assert recorded_gui[0]["stay"] is True


class TestDispatch:
    @pytest.mark.parametrize("name, module", [
        ("gui", "difflow.gui.server"),
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
        assert "gui, plan-export, report" in err


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
        # `difflow gui --help` printing "usage: difflow.gui" sends the
        # reader to an invocation they did not type.
        with pytest.raises(SystemExit):
            cli.main(["gui", "--help"])
        assert capsys.readouterr().out.startswith("usage: difflow gui")
