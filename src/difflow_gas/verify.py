"""Verify a sequential gas network solution against the full equation set.

A sequential-modular solve satisfies most equations by construction
(each unit enforces its own physics), so the meaningful check is to
evaluate ALL equation-oriented residuals on the solved state, exactly
as a simultaneous NLP would formulate them:

* nodal mass balances (kg/s),
* squared-pressure laws of every pipe and resistor (bar^2),
* pressure equalities of valves and short pipes (bar),
* control-valve drop relations (bar), against the drops the flowsheet
  was built with.

The only iterated equations of a converged sequential solve are the
chord laws, whose residual is the tear convergence error; everything
else should sit at floating-point noise. Reporting is in bar / bar^2
because those are the natural magnitudes to eyeball.

The equations themselves live in :func:`difflow_gas.residuals.
network_residuals`, which is the single definition of the network's
equation set; this module is the reporting layer over it, turning the
flat residual vector into the labelled dicts that are easier to read.
Note that the report deliberately omits the compressor block that
``network_residuals`` also returns: ``p_to = ratio * p_from`` holds by
construction for whatever ratio a sequential solve was built with, so
there is nothing to check. An equation-oriented or reconciliation
formulation does have to carry it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from difflow_gas.flowsheets import arc_flow_stream, node_pressure_stream
from difflow_gas.network import Decomposition, GasNetwork
from difflow_gas.residuals import (
    gas_state_layout,
    network_residuals,
    residual_names,
)
from difflow_gas.streams import FLOW_KEY


def arc_flows_kg_s(streams, dec: Decomposition) -> dict[str, float]:
    """Signed arc flows (kg/s, from->to positive) from solved streams."""
    return {
        aid: float(streams[arc_flow_stream(dec, aid)][FLOW_KEY])
        for aid in dec.arcs
    }


def node_pressures_bar(streams, dec: Decomposition) -> dict[str, float]:
    """Node pressures (bar) from solved streams."""
    return {
        node: float(streams[node_pressure_stream(node)]["P"]) / 1e5
        for node in dec.order
    }


@dataclass
class ResidualReport:
    """Equation-oriented residuals of a network state."""

    max_node_imbalance_kg_s: float
    #: |p_f^2 - p_t^2 - beta q|q|| over pipes and resistors
    max_resistance_residual_bar2: float
    #: |p_f - p_t| over valves and short pipes (0.0 if none)
    max_equality_dp_bar: float
    #: |p_f - p_t - dp_cv| over control valves (0.0 if none)
    max_control_valve_residual_bar: float
    node_imbalance: dict[str, float] = field(default_factory=dict)
    resistance_residual_bar2: dict[str, float] = field(default_factory=dict)
    equality_dp_bar: dict[str, float] = field(default_factory=dict)
    control_valve_residual_bar: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """All residuals at or below solver-converged magnitudes."""
        return (
            self.max_node_imbalance_kg_s < 1e-6
            and self.max_resistance_residual_bar2 < 1e-6
            and self.max_equality_dp_bar < 1e-9
            and self.max_control_valve_residual_bar < 1e-9
        )


def residuals_from_values(
    p_bar: dict[str, float],
    q_kg_s: dict[str, float],
    network: GasNetwork,
    cv_drops_bar: dict[str, float] | None = None,
) -> ResidualReport:
    """Evaluate the full residual set on raw pressures (bar) / flows (kg/s).

    Works for ANY method's solution of the same network: a difflow
    sequential solve (via :func:`residual_report`), a Pyomo
    SequentialDecomposition state, or an equation-oriented solver's
    variable values (add the slack node back if it was fixed and
    presolved out).

    Args:
        p_bar: node id -> pressure (bar).
        q_kg_s: arc id -> signed flow (kg/s, from->to positive).
        network: the network the state belongs to.
        cv_drops_bar: control valve drops (bar) the state was solved
            with; defaults to 0 for every control valve.
    """
    layout = gas_state_layout(network)
    # eps_flow=0 selects the exact q|q| rather than its smoothed form:
    # a verification should report the equations as written, not the
    # C-infinity surrogate a solver differentiates.
    values = network_residuals(
        layout.pack(p_bar, q_kg_s, network.supply_kg_s),
        network,
        layout,
        cv_drops_bar=cv_drops_bar,
        eps_flow=0.0,
    )

    imbalance: dict[str, float] = {}
    resistance: dict[str, float] = {}
    equality: dict[str, float] = {}
    control_valve: dict[str, float] = {}
    blocks = {
        "balance_": imbalance,
        "resistance_": resistance,
        "equality_": equality,
        "cv_": control_valve,
        # "compressor_" is intentionally not reported; see the module
        # docstring.
    }
    for name, value in zip(residual_names(network, layout), values):
        for prefix, target in blocks.items():
            if name.startswith(prefix):
                target[name[len(prefix):]] = float(value)
                break

    def _absmax(d):
        return max((abs(v) for v in d.values()), default=0.0)

    return ResidualReport(
        max_node_imbalance_kg_s=_absmax(imbalance),
        max_resistance_residual_bar2=_absmax(resistance),
        max_equality_dp_bar=_absmax(equality),
        max_control_valve_residual_bar=_absmax(control_valve),
        node_imbalance=imbalance,
        resistance_residual_bar2=resistance,
        equality_dp_bar=equality,
        control_valve_residual_bar=control_valve,
    )


def residual_report(
    streams,
    network: GasNetwork,
    dec: Decomposition,
    cv_drops_bar: dict[str, float] | None = None,
) -> ResidualReport:
    """Evaluate the full residual set on a solved difflow flowsheet."""
    return residuals_from_values(
        node_pressures_bar(streams, dec),
        arc_flows_kg_s(streams, dec),
        network,
        cv_drops_bar=cv_drops_bar,
    )


def bounds_report(
    streams, network: GasNetwork, dec: Decomposition
) -> dict[str, tuple[float, float, float, bool]]:
    """Node -> (pmin, p, pmax, within) in bar.

    Requires ``network.pressure_bounds_bar``; nodes without an entry
    get (0, p, inf, True).
    """
    bounds = network.pressure_bounds_bar or {}
    p = node_pressures_bar(streams, dec)
    out = {}
    for node, val in p.items():
        lo, hi = bounds.get(node, (0.0, float("inf")))
        out[node] = (lo, val, hi, lo - 1e-9 <= val <= hi + 1e-9)
    return out


def compressor_report(
    p_bar: dict[str, float],
    q_kg_s: dict[str, float],
    ratios: dict[str, float],
    network: GasNetwork,
) -> dict[str, dict]:
    """Per-station flow, ratio and limit margins (bar) at a solution.

    Margins use ``network.compressor_limits`` where available (missing
    stations get unbounded limits, so their margins are trivially
    large).
    """
    limits = network.compressor_limits or {}
    out = {}
    for cs in network.compressor_ids():
        a = network.arcs[cs]
        lim = limits.get(cs)
        p_in_min = lim.pressure_in_min_bar if lim else 0.0
        p_out_max = lim.pressure_out_max_bar if lim else float("inf")
        out[cs] = {
            "q_kg_s": q_kg_s[cs],
            "ratio": ratios[cs],
            "inlet_margin_bar": p_bar[a.from_node] - p_in_min,
            "outlet_margin_bar": p_out_max - p_bar[a.to_node],
        }
    return out
