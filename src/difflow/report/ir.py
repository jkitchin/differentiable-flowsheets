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
class DecisionVariable:
    """A decision variable of an optimization study (report section G).

    Attributes:
        name: variable name (matches a key of the design point dict).
        value: value at the reported design point (typically the optimum).
        lower: lower bound, if one was supplied.
        upper: upper bound, if one was supplied.
        gradient: objective gradient ``dJ/dx`` at the design point, if it
            could be computed by automatic differentiation.
        elasticity: normalized sensitivity ``(dJ/dx)(x/J)`` at the design
            point, if both the gradient and a non-zero objective are known.
    """

    name: str
    value: float
    lower: float | None = None
    upper: float | None = None
    gradient: float | None = None
    elasticity: float | None = None


@dataclass
class TornadoRow:
    """One-at-a-time swing of the objective over a variable's bounds.

    Attributes:
        variable: decision-variable name.
        low_value: variable value at the low end (its lower bound).
        high_value: variable value at the high end (its upper bound).
        low_output: objective with the variable at ``low_value`` and all
            others held at the design point.
        high_output: objective with the variable at ``high_value``.
        swing: ``abs(high_output - low_output)`` — the tornado bar length.
    """

    variable: str
    low_value: float
    high_value: float
    low_output: float
    high_output: float
    swing: float


@dataclass
class OptimizationReport:
    """Optimization / sensitivity summary for a flowsheet (report section G).

    Attributes:
        objective_name: human-readable name of the objective (e.g.
            "Levelized cost of capture").
        objective_value: objective value at the design point.
        objective_units: units of the objective, if any.
        objective_source: where the objective is defined (function name,
            module path, or a free-text description).
        sense: "minimize" or "maximize".
        variables: per-decision-variable rows (value, bounds, gradient).
        tornado: one-at-a-time swings sorted by decreasing magnitude, or
            ``None`` when no bounds were supplied.
        notes: free-form notes attached to the study.
    """

    objective_name: str
    objective_value: float
    objective_units: str = ""
    objective_source: str = ""
    sense: str = "minimize"
    variables: list[DecisionVariable] = field(default_factory=list)
    tornado: list[TornadoRow] | None = None
    notes: list[str] = field(default_factory=list)


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
    optimization: OptimizationReport | None = None
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
