"""AC power flow: closing the equation set with setpoints and solving it.

:mod:`difflow_power.residuals` states the physics --- ``2 n_bus``
balance equations plus one angle reference. That is fewer equations
than the ``2 n_bus + 2 n_gen`` unknowns, and deliberately so: the
shortfall is exactly the operator's freedom, and a *power flow* spends
it by declaring setpoints, while an *optimal* power flow spends it by
minimising cost. This module does the first.

The specification
-----------------

The classical bus-type specification, stated here as ``2 n_gen - 1``
extra equations rather than by eliminating variables:

=========================  =============================================
bus / unit                 what is specified
=========================  =============================================
slack bus                  voltage magnitude (and the angle, by the
                           reference row already in the residuals);
                           its generators' real power is whatever
                           balances the system
PV bus                     voltage magnitude, and each generator's
                           real power
PQ bus                     nothing --- but any generator sitting on one
                           has both its real and reactive output fixed
several units on one bus   reactive output shared in proportion to
                           reactive capability; real output likewise at
                           the slack bus
=========================  =============================================

Counting: ``n_ctrl`` voltage rows, ``n_gen_ctrl - n_ctrl`` var-sharing
rows, ``n_gen_pv`` scheduled-MW rows, ``k_slack - 1`` MW-sharing rows
and ``2 n_gen_pq`` rows come to exactly ``2 n_gen - 1``, so the system
is square for any arrangement of units.

Writing the specification as equations rather than as an eliminated
sub-vector costs a slightly larger Newton system and buys two things:
the Jacobian is the *same* matrix the OPF and the state estimator see
(so scaling and conditioning carry over), and switching a bus from PV
to PQ --- what a var-limit check does --- is a relabelling rather than
a re-derivation.

Solving
-------

Newton-Raphson, as it has been since Tinney and Hart (1967), here as an
``optimistix`` root find. The gradient of the solution with respect to
anything the residual depends on --- a load, a line reactance, a
setpoint --- comes from the implicit function theorem applied at the
converged point, which optimistix does by default. It never
differentiates the iteration, so the cost of a gradient is one linear
solve regardless of how many Newton steps the forward pass took, and
the answer does not depend on the initial guess.

The flat start (all magnitudes at their setpoint or 1.0 pu, all angles
at zero) converges in 3-5 iterations on well-conditioned transmission
cases. It is not reliable near the loadability limit, where the
Jacobian is singular at the nose of the P-V curve; there the honest
answer is that the solution has ceased to exist, and
:attr:`PowerFlowResult.converged` says so rather than returning a
plausible-looking non-solution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.network import PowerNetwork
from difflow_power.residuals import (
    PowerStateLayout,
    branch_flows,
    power_flow_residuals,
    power_state_layout,
)


# =============================================================================
# The setpoint closure
# =============================================================================


def _share_weights(values: list[float]) -> list[float]:
    """Normalised sharing weights, falling back to an equal split.

    Units on a common bus share reactive output in proportion to
    reactive capability --- the convention MATPOWER's ``pfsoln`` uses,
    and roughly what a joint var controller does. Units with no range
    at all (a fixed-var machine) would give 0/0, so those buses split
    equally instead.
    """
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in values]


@dataclass
class Specification(ParamsMixin):
    """The setpoints that close a power flow.

    Built by :func:`specification_from_network`, which reads the bus
    kinds and generator schedules. Override an entry to study a
    different operating point without rebuilding the network.

    Attributes:
        vm_setpoint: bus id -> regulated voltage magnitude (pu), for
            slack and PV buses.
        pg_setpoint: generator id -> scheduled real output (pu), for
            every generator except those at the slack bus.
        pq_generators: generator ids sitting on PQ buses, whose reactive
            output is fixed too.
        qg_setpoint: generator id -> fixed reactive output (pu), for
            :attr:`pq_generators`.
    """

    vm_setpoint: dict[str, float] = field(default_factory=dict)
    pg_setpoint: dict[str, float] = field(default_factory=dict)
    pq_generators: tuple[str, ...] = ()
    qg_setpoint: dict[str, float] = field(default_factory=dict)


def specification_from_network(network: PowerNetwork) -> Specification:
    """Read a network's bus kinds and schedules into a :class:`Specification`.

    A generator's own ``vm_setpoint`` wins over its bus's when both are
    given, matching how an AVR is actually configured. Several units on
    one bus must agree; disagreeing setpoints are a modelling error and
    are rejected here rather than silently resolved.
    """
    slack = network.slack_bus
    vm_setpoint: dict[str, float] = {}
    for bid, bus in network.buses.items():
        if bus.kind not in ("slack", "pv"):
            continue
        wanted = [
            network.generators[g].vm_setpoint
            for g in network.generators_at(bid)
            if network.generators[g].vm_setpoint is not None
        ]
        if len(set(wanted)) > 1:
            raise ValueError(
                f"units on bus {bid!r} regulate to different voltages "
                f"{sorted(set(wanted))}; they are electrically at the same "
                "point and cannot"
            )
        vm_setpoint[bid] = wanted[0] if wanted else bus.vm_setpoint

    pg_setpoint = {}
    pq_generators = []
    qg_setpoint = {}
    for gid, gen in network.generators.items():
        if network.buses[gen.bus].kind == "pq":
            pq_generators.append(gid)
            pg_setpoint[gid] = gen.p_mw / network.base_mva
            qg_setpoint[gid] = gen.q_mvar / network.base_mva
        elif gen.bus != slack:
            pg_setpoint[gid] = gen.p_mw / network.base_mva
    return Specification(
        vm_setpoint=vm_setpoint,
        pg_setpoint=pg_setpoint,
        pq_generators=tuple(pq_generators),
        qg_setpoint=qg_setpoint,
    )


def specification_names(
    network: PowerNetwork, spec: Specification
) -> list[str]:
    """Names of the setpoint residual rows, in the order returned."""
    names = [f"vm_set_{b}" for b in spec.vm_setpoint]
    for bid in network.buses:
        if network.buses[bid].kind == "pq":
            continue
        gens = network.generators_at(bid)
        names += [f"q_share_{g}" for g in gens[1:]]
        if bid == network.slack_bus:
            names += [f"p_share_{g}" for g in gens[1:]]
    names += [f"pg_set_{g}" for g in spec.pg_setpoint]
    names += [f"qg_set_{g}" for g in spec.pq_generators]
    return names


def setpoint_residuals(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout,
    spec: Specification,
) -> Array:
    """The ``2 n_gen - 1`` equations that close a power flow.

    Traceable in ``x`` and in the setpoint values, so a solved state can
    be differentiated with respect to an AVR setpoint or a unit's
    schedule --- the two knobs an operator actually turns.

    Returns:
        Residuals in the order given by :func:`specification_names`, in
        pu (powers) and pu (voltages).
    """
    state = layout.unpack_arrays(x, network)
    bus_i = network.bus_index
    gen_i = {g: i for i, g in enumerate(layout.generators)}
    rows: list[Array] = []

    for bid, vm_set in spec.vm_setpoint.items():
        rows.append(state.vm[bus_i[bid]] - jnp.asarray(vm_set))

    for bid in network.buses:
        if network.buses[bid].kind == "pq":
            continue
        gens = network.generators_at(bid)
        if len(gens) < 2:
            continue
        q_w = _share_weights(
            [
                network.generators[g].q_max_mvar - network.generators[g].q_min_mvar
                for g in gens
            ]
        )
        q_total = sum(state.qg[gen_i[g]] for g in gens)
        for g, w in list(zip(gens, q_w))[1:]:
            rows.append(state.qg[gen_i[g]] - w * q_total)
        if bid == network.slack_bus:
            p_w = _share_weights(
                [
                    network.generators[g].p_max_mw - network.generators[g].p_min_mw
                    for g in gens
                ]
            )
            p_total = sum(state.pg[gen_i[g]] for g in gens)
            for g, w in list(zip(gens, p_w))[1:]:
                rows.append(state.pg[gen_i[g]] - w * p_total)

    for gid, pg_set in spec.pg_setpoint.items():
        rows.append(state.pg[gen_i[gid]] - jnp.asarray(pg_set))
    for gid in spec.pq_generators:
        rows.append(state.qg[gen_i[gid]] - jnp.asarray(spec.qg_setpoint[gid]))

    if not rows:
        return jnp.zeros((0,), dtype=jnp.float64)
    return jnp.stack([jnp.asarray(r, dtype=jnp.float64) for r in rows])


def power_flow_system(
    network: PowerNetwork,
    layout: PowerStateLayout,
    spec: Specification,
    *,
    branch_params: dict[str, Array] | None = None,
    demand: tuple[Array, Array] | None = None,
) -> Callable[[Array], Array]:
    """Close the physics with the specification into a square ``F(x)``.

    Returns:
        A callable of the packed state returning ``2 n_bus + 2 n_gen``
        residuals: the physics from
        :func:`~difflow_power.residuals.power_flow_residuals` stacked on
        the setpoints from :func:`setpoint_residuals`.
    """

    def residual(x: Array) -> Array:
        return jnp.concatenate(
            [
                power_flow_residuals(
                    x, network, layout,
                    branch_params=branch_params, demand=demand,
                ),
                setpoint_residuals(x, network, layout, spec),
            ]
        )

    return residual


def flat_start(
    network: PowerNetwork,
    layout: PowerStateLayout,
    spec: Specification | None = None,
) -> Array:
    """The standard initial guess: setpoint magnitudes, zero angles.

    Every voltage magnitude at its regulated value (1.0 pu where there
    is none), every angle at the reference, generation at schedule and
    no vars. This is the "flat start" that converges in a handful of
    Newton steps on any well-conditioned transmission case, and the one
    every power-flow program has used for fifty years.
    """
    spec = spec or specification_from_network(network)
    vm = jnp.asarray(
        [
            spec.vm_setpoint.get(b, network.buses[b].vm_setpoint)
            for b in layout.buses
        ],
        dtype=jnp.float64,
    )
    va = jnp.full(
        layout.n_bus, network.buses[network.slack_bus].va_reference,
        dtype=jnp.float64,
    )
    pg = jnp.asarray(
        [
            network.generators[g].p_mw / network.base_mva
            for g in layout.generators
        ],
        dtype=jnp.float64,
    )
    qg = jnp.asarray(
        [
            network.generators[g].q_mvar / network.base_mva
            for g in layout.generators
        ],
        dtype=jnp.float64,
    )
    pd, qd = network.load_arrays_pu()
    return layout.pack(
        vm, va, pg, qg,
        pd=[pd[network.bus_index[b]] for b in layout.demand_buses],
        qd=[qd[network.bus_index[b]] for b in layout.demand_buses],
    )


# =============================================================================
# Solving
# =============================================================================


def solve_state(
    network: PowerNetwork,
    layout: PowerStateLayout,
    spec: Specification,
    x0: Array,
    *,
    branch_params: dict[str, Array] | None = None,
    demand: tuple[Array, Array] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 50,
    throw: bool = False,
) -> tuple[Array, dict[str, Any]]:
    """Newton-solve the closed system; the jit- and grad-safe entry point.

    Everything that might be differentiated --- ``x0``,
    ``branch_params``, ``demand``, the setpoint values inside ``spec``
    --- goes through as JAX values, so this composes under ``jit``,
    ``grad`` and ``vmap``. The gradient comes from the implicit function
    theorem at the converged point, so it costs one linear solve however
    many Newton steps the forward pass took.

    Returns:
        ``(x, stats)`` with ``stats["num_steps"]`` and
        ``stats["converged"]``. ``throw=False`` (the default) returns
        the last iterate on non-convergence rather than raising, which
        is what lets a loadability sweep walk off the nose of the P-V
        curve and report where it happened.

    Example:
        >>> d_dispatch_d_load = jax.jacobian(
        ...     lambda pd: solve_state(net, lay, spec, x0,
        ...                            demand=(pd, qd))[0]
        ... )(pd)                                      # doctest: +SKIP
    """
    residual = power_flow_system(
        network, layout, spec, branch_params=branch_params, demand=demand
    )
    solver = optx.Newton(rtol=rtol, atol=atol)
    sol = optx.root_find(
        lambda y, args: residual(y),
        solver,
        x0,
        max_steps=max_steps,
        throw=throw,
    )
    stats = dict(sol.stats)
    stats["converged"] = sol.result == optx.RESULTS.successful
    return sol.value, stats


@dataclass
class PowerFlowResult(ParamsMixin):
    """A solved power flow, in engineering units.

    Attributes:
        network: the network solved.
        layout: the state packing.
        x: the converged packed state (pu).
        converged: whether Newton reached the tolerance.
        num_steps: Newton iterations taken.
        max_mismatch_mw: largest bus power mismatch (MW or MVAr), the
            number to look at before believing anything else here.
    """

    network: PowerNetwork
    layout: PowerStateLayout
    x: Array
    converged: bool
    num_steps: int
    max_mismatch_mw: float

    # -- bus quantities ---------------------------------------------------

    @property
    def vm(self) -> dict[str, float]:
        """Bus voltage magnitudes (pu)."""
        x = self.x[self.layout.slice_vm]
        return {b: float(x[i]) for i, b in enumerate(self.layout.buses)}

    @property
    def va_degrees(self) -> dict[str, float]:
        """Bus voltage angles (degrees) --- how they are always reported."""
        x = self.x[self.layout.slice_va]
        return {
            b: float(jnp.degrees(x[i]))
            for i, b in enumerate(self.layout.buses)
        }

    @property
    def vm_kv(self) -> dict[str, float]:
        """Bus voltage magnitudes in kV, on each bus's own base."""
        return {
            b: v * self.network.buses[b].base_kv for b, v in self.vm.items()
        }

    # -- generation -------------------------------------------------------

    @property
    def pg_mw(self) -> dict[str, float]:
        """Generator real output (MW)."""
        x = self.x[self.layout.slice_pg] * self.network.base_mva
        return {g: float(x[i]) for i, g in enumerate(self.layout.generators)}

    @property
    def qg_mvar(self) -> dict[str, float]:
        """Generator reactive output (MVAr)."""
        x = self.x[self.layout.slice_qg] * self.network.base_mva
        return {g: float(x[i]) for i, g in enumerate(self.layout.generators)}

    @property
    def total_generation_mw(self) -> float:
        return sum(self.pg_mw.values())

    # -- branch flows -----------------------------------------------------

    def flows(self) -> tuple[Array, Array]:
        """Complex branch flows ``(s_from, s_to)`` in per unit."""
        return branch_flows(self.x, self.network, self.layout)

    @property
    def branch_mw(self) -> dict[str, tuple[float, float]]:
        """Real power into each branch at ``(from, to)`` end, in MW."""
        s_f, s_t = self.flows()
        base = self.network.base_mva
        return {
            a: (float(jnp.real(s_f[i])) * base, float(jnp.real(s_t[i])) * base)
            for i, a in enumerate(self.network.branch_ids)
        }

    @property
    def branch_mva(self) -> dict[str, float]:
        """Apparent power on each branch (MVA), the larger of its two ends."""
        s_f, s_t = self.flows()
        base = self.network.base_mva
        return {
            a: float(jnp.maximum(jnp.abs(s_f[i]), jnp.abs(s_t[i]))) * base
            for i, a in enumerate(self.network.branch_ids)
        }

    @property
    def branch_loading(self) -> dict[str, float]:
        """Loading of each rated branch as a fraction of its rating."""
        mva = self.branch_mva
        return {
            a: mva[a] / br.rate_mva
            for a, br in self.network.branches.items()
            if br.rate_mva is not None
        }

    @property
    def losses_mw(self) -> float:
        """Total real transmission loss (MW)."""
        s_f, s_t = self.flows()
        return float(jnp.sum(jnp.real(s_f + s_t))) * self.network.base_mva

    # -- limit checking ---------------------------------------------------

    def violations(self, tol: float = 1e-6) -> dict[str, str]:
        """Limits the solved point breaks, as ``{what: description}``.

        A power flow does not enforce limits --- that is the difference
        between it and an OPF --- so a converged solution routinely sits
        outside a generator's var range or over a line rating. This says
        where, so the answer is not mistaken for a feasible dispatch.
        """
        out: dict[str, str] = {}
        vm = self.vm
        for bid, bus in self.network.buses.items():
            if vm[bid] < bus.vm_min - tol:
                out[f"vm_{bid}"] = (
                    f"{vm[bid]:.4f} pu below vm_min {bus.vm_min:.4f}"
                )
            elif vm[bid] > bus.vm_max + tol:
                out[f"vm_{bid}"] = (
                    f"{vm[bid]:.4f} pu above vm_max {bus.vm_max:.4f}"
                )
        pg, qg = self.pg_mw, self.qg_mvar
        for gid, gen in self.network.generators.items():
            if pg[gid] < gen.p_min_mw - tol:
                out[f"pg_{gid}"] = (
                    f"{pg[gid]:.2f} MW below p_min {gen.p_min_mw:.2f}"
                )
            elif pg[gid] > gen.p_max_mw + tol:
                out[f"pg_{gid}"] = (
                    f"{pg[gid]:.2f} MW above p_max {gen.p_max_mw:.2f}"
                )
            if qg[gid] < gen.q_min_mvar - tol:
                out[f"qg_{gid}"] = (
                    f"{qg[gid]:.2f} MVAr below q_min {gen.q_min_mvar:.2f}"
                )
            elif qg[gid] > gen.q_max_mvar + tol:
                out[f"qg_{gid}"] = (
                    f"{qg[gid]:.2f} MVAr above q_max {gen.q_max_mvar:.2f}"
                )
        for aid, loading in self.branch_loading.items():
            if loading > 1.0 + tol:
                out[f"rate_{aid}"] = f"{100 * loading:.1f}% of rating"
        return out

    def summary(self) -> str:
        """A one-paragraph description of the solved point."""
        state = "converged" if self.converged else "DID NOT CONVERGE"
        n_bad = len(self.violations())
        vm = self.vm
        lo = min(vm, key=vm.get)
        hi = max(vm, key=vm.get)
        return (
            f"{self.network.name or 'power flow'}: {state} in "
            f"{self.num_steps} Newton steps, max mismatch "
            f"{self.max_mismatch_mw:.2e} MW; "
            f"{self.total_generation_mw:.1f} MW generated for "
            f"{self.network.total_load_mw:.1f} MW load "
            f"({self.losses_mw:.2f} MW loss); voltages "
            f"{vm[lo]:.3f} ({lo}) to {vm[hi]:.3f} ({hi}); "
            f"{n_bad} limit violations"
        )

    def __repr__(self) -> str:
        return f"PowerFlowResult({self.summary()})"


def solve_power_flow(
    network: PowerNetwork,
    *,
    layout: PowerStateLayout | None = None,
    spec: Specification | None = None,
    x0: Array | None = None,
    branch_params: dict[str, Array] | None = None,
    demand: tuple[Array, Array] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    max_steps: int = 50,
) -> PowerFlowResult:
    """Solve a network's power flow from a flat start.

    Args:
        network: the network to solve.
        layout: state packing; defaults to the plain one.
        spec: the setpoints closing the system; defaults to reading them
            off the network's bus kinds and generator schedules.
        x0: initial guess; defaults to :func:`flat_start`.
        branch_params: overrides for ``r``/``x``/``b``/``g``/``tap``/
            ``shift``, as arrays in branch order.
        demand: ``(pd, qd)`` per-bus arrays (pu) replacing the network's
            loads.
        rtol, atol: Newton tolerances on the residual.
        max_steps: iteration cap. Non-convergence is reported in the
            result, not raised.

    Returns:
        A :class:`PowerFlowResult`.

    Example:
        >>> import difflow_power as dp
        >>> res = dp.solve_power_flow(dp.cases.case9())
        >>> res.converged
        True
    """
    layout = layout or power_state_layout(network)
    spec = spec or specification_from_network(network)
    if x0 is None:
        x0 = flat_start(network, layout, spec)

    x, stats = solve_state(
        network, layout, spec, x0,
        branch_params=branch_params, demand=demand,
        rtol=rtol, atol=atol, max_steps=max_steps,
    )
    physics = power_flow_residuals(
        x, network, layout, branch_params=branch_params, demand=demand
    )
    max_mismatch = float(
        jnp.max(jnp.abs(physics[: 2 * layout.n_bus]))
    ) * network.base_mva
    return PowerFlowResult(
        network=network,
        layout=layout,
        x=x,
        converged=bool(stats["converged"]),
        num_steps=int(stats.get("num_steps", 0)),
        max_mismatch_mw=max_mismatch,
    )
