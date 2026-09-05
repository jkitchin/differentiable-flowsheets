"""Check a solved state against the full equation set and every limit.

A power flow satisfies its own equations by construction, so verifying
one is not about the solver: it is about everything a solver was never
asked to enforce. A converged power flow will happily hand back a
generator 200 MVAr past its capability curve, a bus at 0.85 pu, and a
line at 140% of its rating, because none of those is an equation.
:func:`operating_report` says so, in engineering units, for each.

The equations themselves live in
:func:`difflow_power.residuals.power_flow_residuals`, which is the
single definition of the network's equation set; this module is the
reporting layer over it, turning the flat residual vector into labelled
dicts that are easier to read. Nothing here restates any physics, and
nothing here should: a verifier that reimplements the model is checking
the model against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.network import PowerNetwork
from difflow_power.residuals import (
    PowerStateLayout,
    branch_flows,
    power_flow_residuals,
    power_state_layout,
)


@dataclass
class OperatingReport(ParamsMixin):
    """Equation residuals and limit violations of a network state.

    Attributes:
        max_p_mismatch_mw: largest real power imbalance at any bus.
        max_q_mismatch_mvar: largest reactive imbalance at any bus.
        angle_reference_error_rad: how far the slack angle sits from its
            reference. Nonzero means the state was not solved with this
            network's reference.
        p_mismatch_mw: per-bus real imbalance.
        q_mismatch_mvar: per-bus reactive imbalance.
        voltage_violations: bus -> pu outside its limits (signed;
            negative is under, positive is over).
        generator_violations: ``"<gen>.<p|q>"`` -> MW / MVAr outside
            its box.
        thermal_violations: branch -> loading as a fraction of rating,
            for branches above 1.0.
        worst_loading: highest branch loading, or 0.0 if nothing is
            rated.
    """

    max_p_mismatch_mw: float
    max_q_mismatch_mvar: float
    angle_reference_error_rad: float
    p_mismatch_mw: dict[str, float] = field(default_factory=dict)
    q_mismatch_mvar: dict[str, float] = field(default_factory=dict)
    voltage_violations: dict[str, float] = field(default_factory=dict)
    generator_violations: dict[str, float] = field(default_factory=dict)
    thermal_violations: dict[str, float] = field(default_factory=dict)
    worst_loading: float = 0.0

    @property
    def solved(self) -> bool:
        """Whether the EQUATIONS hold, saying nothing about the limits.

        Separate from :attr:`feasible` on purpose: a state can solve the
        physics exactly and still be an operating point no one would
        run.
        """
        return (
            self.max_p_mismatch_mw < 1e-6
            and self.max_q_mismatch_mvar < 1e-6
            and abs(self.angle_reference_error_rad) < 1e-9
        )

    @property
    def feasible(self) -> bool:
        """Whether every limit holds as well."""
        return (
            self.solved
            and not self.voltage_violations
            and not self.generator_violations
            and not self.thermal_violations
        )

    def summary(self) -> str:
        parts = [
            f"mismatch {self.max_p_mismatch_mw:.2e} MW / "
            f"{self.max_q_mismatch_mvar:.2e} MVAr"
        ]
        if self.voltage_violations:
            parts.append(f"{len(self.voltage_violations)} voltage")
        if self.generator_violations:
            parts.append(f"{len(self.generator_violations)} generator")
        if self.thermal_violations:
            parts.append(
                f"{len(self.thermal_violations)} thermal "
                f"(worst {100 * self.worst_loading:.0f}%)"
            )
        if len(parts) == 1:
            parts.append("no limit violations")
        return "; ".join(parts)

    def __repr__(self) -> str:
        return f"OperatingReport({self.summary()})"


def operating_report(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout | None = None,
    *,
    tol: float = 1e-6,
) -> OperatingReport:
    """Full residual and limit report for a packed state.

    Args:
        x: the packed state to check.
        network: the network it belongs to.
        layout: the packing; the plain one by default.
        tol: how far past a limit counts as a violation, in the limit's
            own units (pu for voltages, MW/MVAr for generators, a
            fraction for loadings).

    Returns:
        An :class:`OperatingReport`.

    Example:
        >>> res = solve_power_flow(net)                # doctest: +SKIP
        >>> operating_report(res.x, net).solved        # doctest: +SKIP
        True
    """
    layout = layout or power_state_layout(network)
    base = network.base_mva
    residuals = power_flow_residuals(x, network, layout)
    n = layout.n_bus
    p_mis = {
        b: float(residuals[i]) * base for i, b in enumerate(layout.buses)
    }
    q_mis = {
        b: float(residuals[n + i]) * base for i, b in enumerate(layout.buses)
    }

    vm = x[layout.slice_vm]
    voltage: dict[str, float] = {}
    for i, bid in enumerate(layout.buses):
        bus = network.buses[bid]
        value = float(vm[i])
        if value < bus.vm_min - tol:
            voltage[bid] = value - bus.vm_min
        elif value > bus.vm_max + tol:
            voltage[bid] = value - bus.vm_max

    pg = x[layout.slice_pg] * base
    qg = x[layout.slice_qg] * base
    generator: dict[str, float] = {}
    for i, gid in enumerate(layout.generators):
        gen = network.generators[gid]
        for tag, value, lo, hi in (
            ("p", float(pg[i]), gen.p_min_mw, gen.p_max_mw),
            ("q", float(qg[i]), gen.q_min_mvar, gen.q_max_mvar),
        ):
            if value < lo - tol:
                generator[f"{gid}.{tag}"] = value - lo
            elif value > hi + tol:
                generator[f"{gid}.{tag}"] = value - hi

    s_from, s_to = branch_flows(x, network, layout)
    thermal: dict[str, float] = {}
    worst = 0.0
    for i, aid in enumerate(network.branch_ids):
        rate = network.branches[aid].rate_mva
        if rate is None:
            continue
        mva = float(jnp.maximum(jnp.abs(s_from[i]), jnp.abs(s_to[i]))) * base
        loading = mva / rate
        worst = max(worst, loading)
        if loading > 1.0 + tol:
            thermal[aid] = loading

    return OperatingReport(
        max_p_mismatch_mw=max(abs(v) for v in p_mis.values()),
        max_q_mismatch_mvar=max(abs(v) for v in q_mis.values()),
        angle_reference_error_rad=float(residuals[-1]),
        p_mismatch_mw=p_mis,
        q_mismatch_mvar=q_mis,
        voltage_violations=voltage,
        generator_violations=generator,
        thermal_violations=thermal,
        worst_loading=worst,
    )


def branch_loss_report(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout | None = None,
) -> dict[str, float]:
    """Real power lost in each branch (MW).

    Every entry of a physically sensible solution is NON-NEGATIVE: a
    passive branch cannot generate real power. A lossless one --- a
    transformer modelled with ``r = 0``, as every standard case file
    does --- lands on zero from whichever side roundoff puts it, so
    judge negatives against a tolerance rather than against zero. A
    genuinely negative loss means the admittance block, the tap
    convention or the sign of a flow is wrong, which makes this the
    cheapest useful check on a new case file.
    """
    layout = layout or power_state_layout(network)
    s_from, s_to = branch_flows(x, network, layout)
    losses = jnp.real(s_from + s_to) * network.base_mva
    return {
        aid: float(losses[i]) for i, aid in enumerate(network.branch_ids)
    }
