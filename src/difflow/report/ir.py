"""Intermediate representation for difflow reports.

The IR is a tree of plain dataclasses that captures everything a renderer
needs.  It is deliberately decoupled from any rendering logic so that
Markdown / JSON / HTML / LaTeX renderers can all consume the same object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Provenance:
    """Environment and version information for a run."""

    difflow_version: str
    plugin_versions: dict[str, str]
    jax_version: str
    jax_backend: str
    jax_x64: bool
    python_version: str
    platform: str
    timestamp: str
    git_commit: str | None = None
    git_dirty: bool | None = None


@dataclass
class ParamRow:
    """A single row in the per-unit parameter table."""

    name: str
    symbol: str
    units: str
    value_repr: str


@dataclass
class UnitReport:
    """Per-unit entry in a Report."""

    name: str
    type: str
    plugin: str
    symbol: str
    description: str
    equations: list[str]
    assumptions: list[str]
    references: list[str]
    parameters: list[ParamRow]
    inlet_names: list[str]
    outlet_names: list[str]
    numerical_method: str | None = None


@dataclass
class Edge:
    """Directed connection between units via a named stream."""

    stream: str
    source: str | None
    target: str | None


@dataclass
class RecycleInfo:
    """A recycle loop in a flowsheet."""

    source_stream: str
    dest_stream: str


@dataclass
class Topology:
    """Flowsheet connectivity."""

    units: list[str]
    edges: list[Edge]
    recycles: list[RecycleInfo]
    species_order: list[str]


@dataclass
class SpeciesRow:
    """Per-species thermophysical data row."""

    name: str
    MW: float | None = None
    Tc: float | None = None
    Pc: float | None = None
    omega: float | None = None
    Cp_coeffs: tuple | None = None
    Hvap_coeffs: tuple | None = None
    antoine_coeffs: tuple | None = None
    Hf: float | None = None
    source: str = ""


@dataclass
class FeedSummary:
    """A feed stream summary."""

    name: str
    T: float
    P: float
    flows: dict[str, float]


@dataclass
class ResultSummary:
    """A solved (outlet or intermediate) stream summary."""

    name: str
    T: float
    P: float
    flows: dict[str, float]


@dataclass
class BalanceCheck:
    """Mass balance closure check for a flowsheet."""

    species: str
    feed_total: float
    outlet_total: float
    residual: float


@dataclass
class Report:
    """Full self-documenting report for a flowsheet."""

    provenance: Provenance
    topology: Topology
    units: list[UnitReport]
    species: list[SpeciesRow]
    feeds: list[FeedSummary]
    results: list[ResultSummary] | None = None
    balance_checks: list[BalanceCheck] | None = None
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        from difflow.report.renderers.markdown import to_markdown

        return to_markdown(self)

    def to_json(self, indent: int = 2) -> str:
        from difflow.report.renderers.json_renderer import to_json

        return to_json(self, indent=indent)

    def to_html(self, embed_diagram: bool = True) -> str:
        from difflow.report.renderers.html import to_html

        return to_html(self, embed_diagram=embed_diagram)

    def to_latex(self) -> str:
        from difflow.report.renderers.latex import to_latex

        return to_latex(self)
