"""Tests for the self-documenting :mod:`difflow.report` subsystem."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

import difflow
from difflow import (
    Flowsheet,
    Splitter,
    Unit,
    __version__,
    build_optimization_report,
    build_report,
    diff_reports,
    track_database_access,
)
from difflow.database import get_critical_props
from difflow.plugins import load_plugins, registry
from difflow.report import flowsheet_svg, get_metadata
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


def test_version_exported():
    assert isinstance(__version__, str)
    assert __version__


def test_build_report_without_streams_runs():
    fs = _make_simple_flowsheet()
    rep = build_report(fs, include_git=False)

    assert rep.provenance.difflow_version
    assert rep.topology.units == ["split"]
    assert {u.name for u in rep.units} == {"split"}
    assert rep.feeds and rep.feeds[0].name == "feed"
    assert rep.results is None
    assert rep.balance_checks is None


def test_build_report_with_streams_populates_results_and_balance():
    fs = _make_simple_flowsheet()
    streams = fs.solve()
    rep = build_report(fs, streams=streams, include_git=False)

    assert rep.results is not None
    names = {r.name for r in rep.results}
    assert "top" in names and "bot" in names
    assert rep.balance_checks is not None
    for b in rep.balance_checks:
        assert abs(b.residual) < 1e-6


def test_renderers_produce_non_empty_output():
    fs = _make_simple_flowsheet()
    rep = fs.report(include_git=False)

    md = rep.to_markdown()
    assert "# Flowsheet Report" in md
    assert "Unit Operations" in md

    js = rep.to_json()
    data = json.loads(js)
    assert data["provenance"]["difflow_version"] == __version__
    assert data["topology"]["units"] == ["split"]

    html = rep.to_html()
    assert "<html" in html and "Flowsheet Report" in html

    tex = rep.to_latex()
    assert r"\section*{Flowsheet Report}" in tex


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


def test_docstring_fallback_on_unstructured_class():
    class LegacyUnit:
        """A legacy unit with no structured metadata.

        Key equations:
            F_out = F_in
            T_out = T_in

        Assumptions:
            Steady state
            Ideal

        References:
            Example reference 2024
        """

    m = get_metadata(LegacyUnit)
    assert m.equations == ["F_out = F_in", "T_out = T_in"]
    assert "Steady state" in m.assumptions
    assert m.references == ["Example reference 2024"]


def test_report_round_trip_through_json():
    fs = _make_simple_flowsheet()
    streams = fs.solve()
    rep = fs.report(streams=streams, include_git=False)
    data = json.loads(rep.to_json())
    assert data["topology"]["units"] == ["split"]
    assert {u["name"] for u in data["units"]} == {"split"}
    # Species table should include both species with populated source strings.
    species_names = {s["name"] for s in data["species"]}
    assert {"methane", "ethane"} <= species_names
    for s in data["species"]:
        if s["name"] in {"methane", "ethane"}:
            assert s["source"]
            assert s["Tc"] is not None


def _quadratic_objective(d):
    # Minimum at V=2.0, reflux=1.5; J* = 0. dJ/dV = 2(V-2), dJ/dreflux = 4(r-1.5).
    return (d["V"] - 2.0) ** 2 + 2.0 * (d["reflux"] - 1.5) ** 2


def test_build_optimization_report_gradient_and_tornado():
    opt = build_optimization_report(
        _quadratic_objective,
        design_point={"V": 2.0, "reflux": 1.5},
        bounds={"V": (1.0, 3.0), "reflux": (0.5, 2.5)},
        objective_name="Test cost",
        objective_units="USD",
        sense="minimize",
    )

    assert opt.objective_name == "Test cost"
    assert opt.objective_units == "USD"
    assert opt.objective_value == pytest.approx(0.0, abs=1e-9)
    # Source auto-derived from the callable.
    assert "_quadratic_objective" in opt.objective_source

    by_name = {v.name: v for v in opt.variables}
    assert set(by_name) == {"V", "reflux"}
    # At the optimum the gradient is ~0 for both variables.
    assert by_name["V"].gradient == pytest.approx(0.0, abs=1e-6)
    assert by_name["reflux"].gradient == pytest.approx(0.0, abs=1e-6)
    assert by_name["V"].lower == 1.0 and by_name["V"].upper == 3.0

    # Tornado present, sorted by decreasing swing.
    assert opt.tornado is not None
    swings = [t.swing for t in opt.tornado]
    assert swings == sorted(swings, reverse=True)
    # reflux (coeff 2, half-width 1.0 -> swing 0) vs V (coeff 1, half-width 1.0
    # -> swing 0): both symmetric about optimum so J(low)==J(high) here.
    for t in opt.tornado:
        assert t.swing == pytest.approx(0.0, abs=1e-9)


def test_build_optimization_report_gradient_off_optimum():
    opt = build_optimization_report(
        _quadratic_objective,
        design_point={"V": 3.0, "reflux": 1.5},
        objective_name="Test cost",
    )
    by_name = {v.name: v for v in opt.variables}
    # dJ/dV = 2(V-2) = 2 at V=3.
    assert by_name["V"].gradient == pytest.approx(2.0, rel=1e-5)
    # Elasticity = (dJ/dx)(x/J) = 2 * 3 / 1 = 6.
    assert by_name["V"].elasticity == pytest.approx(6.0, rel=1e-5)
    # No bounds -> no tornado.
    assert opt.tornado is None


def test_report_includes_optimization_section_all_renderers():
    fs = _make_simple_flowsheet()
    opt = build_optimization_report(
        _quadratic_objective,
        design_point={"V": 3.0, "reflux": 1.5},
        bounds={"V": (1.0, 3.0)},
        objective_name="Test cost",
    )
    rep = fs.report(include_git=False, optimization=opt)

    assert rep.optimization is opt

    md = rep.to_markdown()
    assert "Optimization and Sensitivity" in md
    assert "Test cost" in md

    data = json.loads(rep.to_json())
    assert data["optimization"]["objective_name"] == "Test cost"
    assert {v["name"] for v in data["optimization"]["variables"]} == {"V", "reflux"}

    html = rep.to_html()
    assert "Optimization and Sensitivity" in html

    tex = rep.to_latex()
    assert "Optimization and Sensitivity" in tex


def test_report_without_optimization_omits_section():
    fs = _make_simple_flowsheet()
    rep = fs.report(include_git=False)
    assert rep.optimization is None
    assert "Optimization and Sensitivity" not in rep.to_markdown()
    data = json.loads(rep.to_json())
    assert data["optimization"] is None


class _AffineRecycle:
    """out = offset + slope * in (single species 'A'); contractive for |slope|<1."""

    def __init__(self, offset: float, slope: float):
        self.offset = offset
        self.slope = slope

    def __call__(self, inlet):
        return make_stream(
            {"A": self.offset + self.slope * inlet["F_A"]}, inlet["T"], inlet["P"]
        )


def _make_recycle_flowsheet() -> Flowsheet:
    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("loop", _AffineRecycle(0.5, 0.5), ["tear"], ["loop_out"]))
    fs.add_recycle("loop_out", "tear")
    return fs


# --- Section F: recycle convergence -----------------------------------------

def test_convergence_direct_solve():
    fs = _make_simple_flowsheet()
    streams = fs.solve()
    rep = fs.report(streams=streams, include_git=False)

    assert rep.convergence is not None
    assert rep.convergence.method == "direct"
    assert rep.convergence.iterations == 0
    assert rep.convergence.converged is True
    assert rep.convergence.tear_streams == []
    assert "Recycle Convergence" in rep.to_markdown()


def test_convergence_recycle_solve():
    fs = _make_recycle_flowsheet()
    streams = fs.solve(acceleration="anderson", tol=1e-10)
    rep = fs.report(streams=streams, include_git=False)

    c = rep.convergence
    assert c is not None
    assert c.method == "anderson"
    assert c.tear_streams == ["tear"]
    assert c.converged is True
    assert c.iterations is not None and c.iterations >= 0
    assert c.residual is not None and c.residual < 1e-9
    data = json.loads(rep.to_json())
    assert data["convergence"]["method"] == "anderson"


def test_convergence_absent_without_streams():
    fs = _make_simple_flowsheet()
    rep = fs.report(include_git=False)
    assert rep.convergence is None
    assert "Recycle Convergence" not in rep.to_markdown()


# --- Section D upgrade: instrumented database access -------------------------

def test_track_database_access_records_lookups():
    with track_database_access() as tracker:
        get_critical_props("methane")
        get_critical_props("co2")  # alias -> carbon_dioxide
    assert tracker.was_accessed("methane")
    assert tracker.was_accessed("carbon_dioxide")
    assert tracker.was_accessed("co2")  # alias resolves
    assert not tracker.was_accessed("ethane")
    assert "critical" in tracker.kinds("methane")


def test_report_annotates_accessed_species():
    fs = _make_simple_flowsheet()
    with track_database_access() as tracker:
        streams = fs.solve()
        # Simulate a unit that consults the database for methane only.
        get_critical_props("methane")
    rep = fs.report(streams=streams, include_git=False, db_access=tracker)

    by_name = {s.name: s for s in rep.species}
    assert by_name["methane"].accessed is True
    assert by_name["ethane"].accessed is False
    md = rep.to_markdown()
    assert "accessed" in md  # column shown only when tracked


def test_report_species_accessed_none_when_untracked():
    fs = _make_simple_flowsheet()
    rep = fs.report(include_git=False)
    assert all(s.accessed is None for s in rep.species)
    # No 'accessed' column header when tracking was not used.
    assert "accessed" not in rep.to_markdown()


# --- v2: embedded diagram ----------------------------------------------------

def test_flowsheet_svg_contains_units():
    fs = _make_simple_flowsheet()
    rep = fs.report(include_git=False)
    svg = flowsheet_svg(rep)
    assert svg.startswith("<svg")
    assert "split" in svg
    assert "feed" in svg
    # Embedded in HTML by default.
    assert "<svg" in rep.to_html()
    # Suppressible.
    assert "<svg" not in rep.to_html(embed_diagram=False)


def test_flowsheet_svg_empty_for_no_units():
    fs = Flowsheet(["methane"])
    rep = fs.report(include_git=False)
    assert flowsheet_svg(rep) == ""


# --- v2: report diff ---------------------------------------------------------

def _make_gain_flowsheet(gain: float) -> Flowsheet:
    from dataclasses import dataclass

    from difflow.params_mixin import ParamsMixin

    @dataclass
    class _GainParams(ParamsMixin):
        gain: float = 1.0

    class _GainUnit:
        symbol = "Gain"
        equations = [r"F^\mathrm{out} = g\, F^\mathrm{in}"]
        references = ["test"]

        def __init__(self, g: float):
            self.params = _GainParams(gain=g)

        def __call__(self, inlet):
            return make_stream(
                {"A": self.params.gain * inlet["F_A"]}, inlet["T"], inlet["P"]
            )

    fs = Flowsheet(species_order=["A"], default_flow=1.0)
    fs.add_feed("feed", make_stream({"A": 1.0}, 300.0, 1e5))
    fs.add_unit(Unit("gain", _GainUnit(gain), ["feed"], ["out"]))
    return fs


def test_report_diff_detects_param_and_result_changes():
    fs1 = _make_gain_flowsheet(1.0)
    rep1 = fs1.report(streams=fs1.solve(), include_git=False)

    fs2 = _make_gain_flowsheet(2.0)
    rep2 = fs2.report(streams=fs2.solve(), include_git=False)

    d = rep1.diff(rep2)
    assert not d.is_empty
    # gain param change captured on the "gain" unit.
    changes = {uc.unit: uc for uc in d.unit_param_changes}
    assert "gain" in changes
    names = {c.name for c in changes["gain"].changes}
    assert "gain" in names
    # Result stream "out" changed (1.0 -> 2.0).
    assert {sc.stream for sc in d.result_changes} == {"out"}
    md = d.to_markdown()
    assert "# Report Diff" in md and "gain" in md


def test_report_diff_empty_for_identical():
    fs = _make_simple_flowsheet()
    s = fs.solve()
    rep = fs.report(streams=s, include_git=False)
    d = diff_reports(rep, rep)
    assert d.is_empty
    assert "No differences" in d.to_markdown()


# --- v2: markdown snapshot regression ---------------------------------------

def _redact_provenance(md: str) -> str:
    """Drop the volatile Provenance section (timestamp, versions, platform)."""
    out, skip = [], False
    for line in md.splitlines():
        if line.startswith("## Provenance"):
            skip = True
            continue
        if skip and line.startswith("## "):
            skip = False
        if not skip:
            out.append(line)
    return "\n".join(out).rstrip() + "\n"


def test_markdown_snapshot_regression():
    """The canonical flowsheet's Markdown must not drift unexpectedly.

    Regenerate the snapshot with::

        DIFFLOW_UPDATE_SNAPSHOTS=1 pytest tests/test_report.py -k snapshot
    """
    import os
    from pathlib import Path

    fs = _make_simple_flowsheet()
    streams = fs.solve()
    current = _redact_provenance(
        fs.report(streams=streams, include_git=False).to_markdown()
    )

    snap = Path(__file__).parent / "snapshots" / "report_canonical.md"
    if os.environ.get("DIFFLOW_UPDATE_SNAPSHOTS"):
        snap.write_text(current)
    expected = snap.read_text()
    assert current == expected, (
        "Canonical report Markdown drifted. If intentional, regenerate with "
        "DIFFLOW_UPDATE_SNAPSHOTS=1 pytest -k snapshot."
    )


def test_cli_markdown_roundtrip(tmp_path, capsys):
    """Exercise the ``difflow report`` CLI end-to-end on a tiny script."""
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

    rc = main([str(script), "--no-git"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# Flowsheet Report" in out
    assert "split" in out
