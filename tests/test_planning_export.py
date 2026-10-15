"""Tests for exporting delta vectors to LP-style planning systems."""

import csv
import json

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    CSTR,
    CSTRParams,
    Flowsheet,
    IdealThermo,
    SpeciesData,
    Unit,
    make_stream,
)
from difflow.planning import (
    Block,
    DeltaBasePlanner,
    DeltaVectorSet,
    Network,
    sanitize_name,
    write_csv,
    write_iterations_csv,
    write_json,
    write_lp,
    write_mps,
)

jax.config.update("jax_enable_x64", True)


def _bowl():
    return Block(name="q",
                 fn=lambda u: jnp.array([-(u[0] - 0.3) ** 2 - (u[1] - 0.7) ** 2]),
                 u_names=["x", "y"], y_names=["f"], lb=[0.0, 0.0], ub=[1.0, 1.0])


@pytest.fixture(scope="module")
def plan_result():
    return DeltaBasePlanner(Network([_bowl()]), prices={"q.f": 1.0},
                            radius=0.3, vertex_seeding=False).solve()


@pytest.fixture(scope="module")
def dvs(plan_result):
    return DeltaVectorSet.from_result(plan_result)


def _flowsheet():
    thermo = IdealThermo({
        s: SpeciesData(s, MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                       Hvap_coeffs=(35000.0, 0.38, 500.0),
                       antoine_coeffs=(10.0, 3000.0, -50.0))
        for s in ("A", "B")})

    def rate_fn(C, T, params):
        return jnp.array([params["k"] * C["A"]])

    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("reactor", CSTR(CSTRParams(
        V=jnp.array(1.0), rate_fn=rate_fn, stoich=jnp.array([[-1.0], [1.0]]),
        rate_params={"k": jnp.array(1.0)}, species_order=["A", "B"]),
        thermo=thermo, mode="isothermal"),
        ["feed"], ["product"], params={"T_spec": 350.0}))
    return fs


# ---------------------------------------------------------------------------
# The IR
# ---------------------------------------------------------------------------


def test_delta_vector_set_smoke(dvs):
    v = dvs.vector("q")
    assert v.u_names == ["q.x", "q.y"] and v.y_names == ["q.f"]
    assert len(v.J) == 1 and len(v.J[0]) == 2
    assert v.scaled_J is not None
    assert v.predict(v.u0) == pytest.approx(v.y0)
    assert dvs.meta["format"] == "delta-vectors/1"
    with pytest.raises(KeyError, match="no block"):
        dvs.vector("nope")


def test_from_block_needs_no_planner():
    blk = Block(name="free", fn=lambda u: jnp.array([u[0] * 2.0]),
                u_names=["a"], y_names=["f"], u0=[1.0])
    one = DeltaVectorSet.from_block(blk, radius=0.2)
    assert one.vectors[0].J == [[2.0]]
    assert one.vectors[0].radius == pytest.approx(0.2)
    assert one.links == [] and one.specs == []


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_write_json_round_trips(dvs, tmp_path):
    back = json.load(open(write_json(dvs, tmp_path / "p.json"), encoding="utf-8"))
    got = back["vectors"][0]
    v = dvs.vector("q")
    assert got["J"] == v.J and got["u0"] == v.u0 and got["y0"] == v.y0


def test_write_csv_smoke(dvs, tmp_path):
    written = write_csv(dvs, tmp_path / "t", scaled=True)
    assert all(p.endswith(".csv") for p in written)
    mapping = {r["name"]: r["lp_name"] for r in
               csv.DictReader(open(tmp_path / "t" / "names.csv", encoding="utf-8"))}
    assert mapping["q.x"] == "q_x"


def test_write_iterations_csv(plan_result, tmp_path):
    rows = list(csv.DictReader(open(
        write_iterations_csv(plan_result, tmp_path / "i.csv"), encoding="utf-8")))
    assert len(rows) == len(plan_result.history)


def test_sanitize_name():
    assert sanitize_name("feed:F1.total_flow") == "feed_F1_total_flow"
    assert sanitize_name("2nd_stage") == "v2nd_stage"


def test_write_lp_and_mps(plan_result, tmp_path):
    pytest.importorskip("pyomo")
    lp = open(write_lp(plan_result.lp_model, tmp_path / "p.lp"), encoding="utf-8").read()
    assert "q_x" in lp
    mps = open(write_mps(plan_result.lp_model, tmp_path / "p.mps"), encoding="utf-8").read()
    assert "COLUMNS" in mps


# ---------------------------------------------------------------------------
# Block.from_flowsheet
# ---------------------------------------------------------------------------


def test_block_from_flowsheet_smoke():
    blk = Block.from_flowsheet(_flowsheet(), u=["reactor.V", "feed:feed.total_flow"],
                               y=["product.F_B"], name="plant")
    assert blk.u0 == pytest.approx([1.0, 10.0])
    assert blk.u_names == ["reactor_V", "feed_feed_total_flow"]
    assert blk.metadata["source"] == "flowsheet"


def test_block_from_flowsheet_validation():
    fs = _flowsheet()
    with pytest.raises(ValueError, match="at least one lever"):
        Block.from_flowsheet(fs, u=[], y=["product.F_B"])
    with pytest.raises(ValueError, match="<stream>.<quantity>"):
        Block.from_flowsheet(fs, u=["reactor.V"], y=["product"])
    with pytest.raises(KeyError):
        Block.from_flowsheet(fs, u=["nope.V"], y=["product.F_B"])


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------


_BLOCK_SCRIPT = '''
import jax.numpy as jnp
from difflow.planning import Block
blk = Block(name="free", fn=lambda u: jnp.array([u[0] * 2.0]),
            u_names=["a"], y_names=["f"], u0=[1.0])
'''


@pytest.fixture
def block_script(tmp_path):
    path = tmp_path / "block_source.py"
    path.write_text(_BLOCK_SCRIPT)
    return str(path)


def test_cli_json_to_stdout(block_script, capsys):
    from difflow.planning.cli import main

    assert main([block_script]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["vectors"][0]["block"] == "free"


def test_cli_dispatched_from_the_difflow_entry_point(block_script, tmp_path):
    from difflow.report.cli import main

    out = tmp_path / "plan.json"
    assert main(["plan-export", block_script, "-o", str(out)]) == 0
    assert json.loads(out.read_text())["vectors"]


def test_cli_errors(block_script, tmp_path, capsys):
    from difflow.planning.cli import main

    assert main([block_script, "--format", "csv"]) == 2
    assert "needs -o/--output" in capsys.readouterr().err
    assert main([block_script, "--format", "lp", "-o", str(tmp_path / "x.lp")]) == 2
    assert "must run the planner" in capsys.readouterr().err

    empty = tmp_path / "empty.py"
    empty.write_text("x = 1\n")
    with pytest.raises(SystemExit, match="no PlanResult"):
        main([str(empty)])


def test_cli_bare_flowsheet_needs_levers(tmp_path):
    from difflow.planning.cli import main

    path = tmp_path / "fs.py"
    path.write_text("from difflow import Flowsheet\nfs = Flowsheet(species_order=['A'])\n")
    with pytest.raises(SystemExit, match="needs levers and outputs"):
        main([str(path)])
