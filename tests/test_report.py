"""Tests for the self-documenting :mod:`difflow.report` subsystem."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from difflow import (
    Flowsheet,
    Splitter,
    Unit,
    build_optimization_report,
    build_report,
    diff_reports,
    track_database_access,
)
from difflow.database import get_critical_props
from difflow.plugins import load_plugins, registry
from difflow.report import flowsheet_diagram, flowsheet_svg, get_metadata
from difflow.streams import make_stream


@pytest.fixture(scope="module", autouse=True)
def _ensure_plugins_loaded():
    load_plugins()


def _make_simple_flowsheet() -> Flowsheet:
    fs = Flowsheet(["methane", "ethane"])
    fs.add_feed(
        "feed",
        make_stream({"methane": 1.0, "ethane": 0.5}, 300.0, 101325.0),
    )
    fs.add_unit(
        Unit(
            "split",
            Splitter(["methane", "ethane"]),
            ["feed"],
            ["top", "bot"],
            params={"split_frac": 0.6},
        )
    )
    return fs


def test_build_report_and_renderers():
    fs = _make_simple_flowsheet()
    rep = build_report(fs, streams=fs.solve(), include_git=False)
    assert rep.topology.units == ["split"]
    assert {r.name for r in rep.results} >= {"top", "bot"}
    assert all(abs(b.residual) < 1e-6 for b in rep.balance_checks)
    assert rep.convergence.method == "direct"
    assert "# Flowsheet Report" in rep.to_markdown()
    assert json.loads(rep.to_json())["topology"]["units"] == ["split"]
    assert "<svg" in rep.to_html()
    assert r"\section*{Flowsheet Report}" in rep.to_latex()


def test_metadata_contract_for_registered_units():
    missing_eq: list[str] = []
    missing_refs: list[str] = []
    for name, info in registry.list_operations().items():
        m = get_metadata(info.cls)
        assert m.symbol and isinstance(m.symbol, str), name
        if not m.equations:
            missing_eq.append(name)
        if not m.references:
            missing_refs.append(name)
    assert not missing_eq, f"Units missing equations: {missing_eq}"
    assert not missing_refs, f"Units missing references: {missing_refs}"


def test_optimization_report_section():
    opt = build_optimization_report(
        lambda d: (d["V"] - 2.0) ** 2 + 2.0 * (d["reflux"] - 1.5) ** 2,
        design_point={"V": 3.0, "reflux": 1.5},
        bounds={"V": (1.0, 3.0)},
        objective_name="Test cost",
    )
    assert {v.name for v in opt.variables} == {"V", "reflux"}
    assert all(jnp.isfinite(v.gradient) for v in opt.variables)
    rep = _make_simple_flowsheet().report(include_git=False, optimization=opt)
    assert "Optimization and Sensitivity" in rep.to_markdown()


class _AffineRecycle:
    def __call__(self, inlet):
        return make_stream({"A": 0.5 + 0.5 * inlet["F_A"]}, inlet["T"], inlet["P"])


def test_convergence_recycle_solve():
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", _AffineRecycle(), ["tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    rep = fs.report(streams=fs.solve(acceleration="anderson", tol=1e-10), include_git=False)
    assert rep.convergence.method == "anderson"
    assert rep.convergence.tear_streams == ["tear"]
    assert rep.convergence.converged is True


def test_report_annotates_accessed_species():
    fs = _make_simple_flowsheet()
    with track_database_access() as tracker:
        streams = fs.solve()
        get_critical_props("methane")
    assert tracker.was_accessed("methane")
    rep = fs.report(streams=streams, include_git=False, db_access=tracker)
    by_name = {s.name: s for s in rep.species}
    assert by_name["methane"].accessed is True
    assert by_name["ethane"].accessed is False


def test_flowsheet_diagram():
    fs = _make_simple_flowsheet()
    default = flowsheet_diagram(fs)
    assert default == flowsheet_svg(fs.report(include_git=False))
    assert flowsheet_diagram(fs, {"split": {"x": 500, "y": 300}}) != default
    assert flowsheet_diagram(fs, {"deleted": {"x": 9, "y": 9}}) == default


def test_report_diff():
    fs = _make_simple_flowsheet()
    rep = fs.report(streams=fs.solve(), include_git=False)
    assert diff_reports(rep, rep).is_empty
    fs.units[0].params["split_frac"] = 0.3
    rep2 = fs.report(streams=fs.solve(), include_git=False)
    assert not rep.diff(rep2).is_empty


def test_cli_markdown_roundtrip(tmp_path, capsys):
    script = tmp_path / "build_fs.py"
    script.write_text(
        "from difflow import Flowsheet, Unit, Splitter\n"
        "from difflow.streams import make_stream\n"
        "fs = Flowsheet(['methane','ethane'])\n"
        "fs.add_feed('feed', make_stream({'methane':1.0,'ethane':0.5},300,101325))\n"
        "fs.add_unit(Unit('split', Splitter(['methane','ethane']),\n"
        "                 ['feed'], ['top','bot'], params={'split_frac': 0.6}))\n"
    )
    from difflow.report.cli import main

    assert main([str(script), "--no-git"]) == 0
    assert "# Flowsheet Report" in capsys.readouterr().out
