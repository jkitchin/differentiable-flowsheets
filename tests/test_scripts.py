"""Opening a flowsheet that is written as a Python script."""

import pytest

from difflow import scripts

PLANT = '''\
from difflow import Flowsheet, Unit, make_stream, Heater, HeaterParams

SPECIES = ["water", "ethanol"]

fs = Flowsheet(species_order=SPECIES)
fs.add_feed("feed", make_stream({"water": 1.0, "ethanol": 0.1},
                                T=300.0, P=101325.0))
fs.add_unit(Unit("heat", Heater(HeaterParams(T_out=360.0, Cp=75.0)),
                 ["feed"], ["hot"]))
'''


@pytest.fixture
def plant(tmp_path):
    path = tmp_path / "plant.py"
    path.write_text(PLANT)
    return path


class TestRecognizing:
    def test_a_script_is_a_py_file(self):
        assert scripts.is_script("plant.py")
        assert scripts.is_script("/a/b/PLANT.PY")

    def test_a_flowsheet_file_is_not_one(self):
        assert not scripts.is_script("plant.json")
        assert not scripts.is_script(None)
        assert not scripts.is_script("")

    def test_the_save_target_sits_beside_the_script(self):
        assert scripts.save_target("/a/plant.py").name == "plant.json"


class TestLoading:
    def test_it_returns_the_flowsheet_the_script_built(self, plant):
        fs = scripts.load_flowsheet(plant)
        assert [u.name for u in fs.units] == ["heat"]
        assert list(fs.feeds) == ["feed"]

    def test_the_flowsheet_solves(self, plant):
        streams = scripts.load_flowsheet(plant).solve()
        assert float(streams["hot"]["T"]) == pytest.approx(360.0)

    def test_the_first_flowsheet_defined_is_the_one(self, tmp_path):
        path = tmp_path / "two.py"
        path.write_text(
            PLANT + '\nlater = Flowsheet(species_order=["argon"])\n'
        )
        assert scripts.load_flowsheet(path).species_order == [
            "water", "ethanol"
        ]

    def test_the_main_block_does_not_run(self, tmp_path):
        # The part of a script that solves, sweeps or plots. Opening a
        # flowsheet in an editor should not cost an hour of optimization.
        path = tmp_path / "guarded.py"
        path.write_text(PLANT + '''
if __name__ == "__main__":
    raise AssertionError("the main block ran")
''')
        assert [u.name for u in scripts.load_flowsheet(path).units] == ["heat"]


class TestSayingWhatWentWrong:
    def test_a_script_that_raises_names_itself_and_the_cause(self, tmp_path):
        path = tmp_path / "bad.py"
        path.write_text('raise ValueError("no feed data")\n')
        with pytest.raises(scripts.ScriptError) as exc:
            scripts.load_flowsheet(path)
        assert "bad.py" in str(exc.value)
        assert "no feed data" in str(exc.value)

    def test_the_original_exception_is_kept(self, tmp_path):
        path = tmp_path / "bad.py"
        path.write_text('raise ValueError("no feed data")\n')
        with pytest.raises(scripts.ScriptError) as exc:
            scripts.load_flowsheet(path)
        assert isinstance(exc.value.__cause__, ValueError)

    def test_a_script_with_no_flowsheet_says_where_to_put_one(self, tmp_path):
        path = tmp_path / "empty.py"
        path.write_text("x = 1\n")
        with pytest.raises(scripts.ScriptError, match="module level"):
            scripts.load_flowsheet(path)

    def test_a_flowsheet_built_inside_a_function_is_not_found(self, tmp_path):
        path = tmp_path / "fn.py"
        path.write_text('''
from difflow import Flowsheet

def build():
    return Flowsheet(species_order=["A"])
''')
        with pytest.raises(scripts.ScriptError, match="no Flowsheet"):
            scripts.load_flowsheet(path)


class TestInTheEditor:
    def test_the_session_opens_a_script(self, plant):
        from difflow.gui.session import FlowsheetSession

        session = FlowsheetSession(path=plant)
        document = session.document()
        assert [u["name"] for u in document["flowsheet"]["units"]] == ["heat"]

    def test_it_saves_beside_the_script_and_not_over_it(self, plant):
        from difflow.gui.session import FlowsheetSession

        session = FlowsheetSession(path=plant)
        before = plant.read_text()
        answer = session.save()
        assert answer["ok"]
        assert answer["path"].endswith("plant.json")
        assert plant.read_text() == before

    def test_the_document_reports_both_names(self, plant):
        from difflow.gui.session import FlowsheetSession

        document = FlowsheetSession(path=plant).document()
        assert document["source"] == str(plant)
        assert document["path"].endswith("plant.json")

    def test_a_json_file_reports_no_source(self, plant):
        from difflow.gui.session import FlowsheetSession

        FlowsheetSession(path=plant).save()
        opened = FlowsheetSession(path=plant.with_suffix(".json"))
        assert opened.document()["source"] == ""
        assert opened.source is None

    def test_a_script_wins_over_a_json_of_the_same_name(self, plant):
        # The script is what was named, so it is what gets run, even
        # once an editor session has left a `plant.json` beside it.
        from difflow.gui.session import FlowsheetSession

        FlowsheetSession(path=plant).save()
        plant.write_text(PLANT.replace('"heat"', '"renamed"'))
        session = FlowsheetSession(path=plant)
        assert [u.name for u in session.flowsheet.units] == ["renamed"]


class TestTheCommandLine:
    """`difflow gui plant.py` -- the door this is all here to open."""

    def test_a_script_that_raises_is_reported_not_traced(self, tmp_path, capsys):
        from difflow.gui.server import main

        path = tmp_path / "bad.py"
        path.write_text('raise ValueError("no feed data")\n')
        assert main([str(path), "--no-browser"]) == 1
        assert "no feed data" in capsys.readouterr().err

    def test_a_script_with_no_flowsheet_is_reported_too(self, tmp_path, capsys):
        from difflow.gui.server import main

        path = tmp_path / "empty.py"
        path.write_text("x = 1\n")
        assert main([str(path), "--no-browser"]) == 1
        assert "no Flowsheet" in capsys.readouterr().err

    def test_the_usage_line_mentions_scripts(self, capsys):
        from difflow.gui.server import main

        with pytest.raises(SystemExit):
            main(["--help"])
        assert "Python script" in capsys.readouterr().out
