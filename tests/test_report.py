"""Tests for the self-documenting :mod:`difflow.report` subsystem."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

import difflow
from difflow import Flowsheet, Splitter, Unit, __version__, build_report
from difflow.plugins import load_plugins, registry
from difflow.report import get_metadata
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
