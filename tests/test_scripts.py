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


def test_recognizing_scripts():
    assert scripts.is_script("/a/b/PLANT.PY")
    assert not scripts.is_script("plant.json")
    assert not scripts.is_script(None)
    assert scripts.save_target("/a/plant.py").name == "plant.json"


def test_load_flowsheet_skips_main_block(tmp_path):
    path = tmp_path / "guarded.py"
    path.write_text(PLANT + '''
if __name__ == "__main__":
    raise AssertionError("the main block ran")
''')
    fs = scripts.load_flowsheet(path)
    assert [u.name for u in fs.units] == ["heat"]
    assert float(fs.solve()["hot"]["T"]) == pytest.approx(360.0)


def test_a_script_that_raises(tmp_path):
    path = tmp_path / "bad.py"
    path.write_text('raise ValueError("no feed data")\n')
    with pytest.raises(scripts.ScriptError, match="no feed data") as exc:
        scripts.load_flowsheet(path)
    assert isinstance(exc.value.__cause__, ValueError)


def test_a_script_with_no_flowsheet(tmp_path):
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n")
    with pytest.raises(scripts.ScriptError, match="module level"):
        scripts.load_flowsheet(path)


def test_gui_session_saves_beside_the_script(plant):
    from difflow.gui.session import FlowsheetSession

    session = FlowsheetSession(path=plant)
    assert [u["name"] for u in session.document()["flowsheet"]["units"]] == ["heat"]
    before = plant.read_text()
    answer = session.save()
    assert answer["ok"] and answer["path"].endswith("plant.json")
    assert plant.read_text() == before


def test_cli_reports_a_bad_script(tmp_path, capsys):
    from difflow.gui.server import main

    path = tmp_path / "bad.py"
    path.write_text('raise ValueError("no feed data")\n')
    assert main([str(path), "--no-browser"]) == 1
    assert "no feed data" in capsys.readouterr().err
