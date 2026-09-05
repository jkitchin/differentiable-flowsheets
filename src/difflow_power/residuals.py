"""JAX-traceable equation-oriented residuals of an AC network.

This module is the single definition of a network's equation set: one
flat state vector in, one residual array out, fully traceable, so that
``jax.jacobian`` yields the constraint Jacobian that the power flow,
the OPF, state estimation and observability analysis all need. Every
other module in the plugin is a *consumer* of what is defined here ---
:mod:`difflow_power.powerflow` closes the system with setpoints,
:mod:`difflow_power.opf` optimises subject to it,
:mod:`difflow_power.estimation` reconciles measurements against it,
:mod:`difflow_power.verify` reports it. None of them restates the
physics.

The equations are checked against an independent restatement in
``tests.power.test_residuals.reference_residuals`` --- written in
polar form, straight from a textbook --- rather than against
``verify``, which would be checking this code against itself.

State vector
------------

:class:`PowerStateLayout` packs, in this order:

===================  ==========================================
block                entries
===================  ==========================================
``vm_<bus>``         voltage magnitudes (pu), bus order
``va_<bus>``         voltage angles (rad), bus order
``pg_<gen>``         real generation (pu), generator order
``qg_<gen>``         reactive generation (pu), generator order
``pd_<bus>``         optional real demand (pu)
``qd_<bus>``         optional reactive demand (pu)
``tap_<branch>``     optional transformer tap ratios
``shift_<branch>``   optional phase shift angles (rad)
``bsh_<bus>``        optional switched shunt susceptance (pu)
===================  ==========================================

The first four blocks are always present; the rest are opt-in, and
exist because the same equation set has to serve four different
questions. A power flow treats demand as known. State estimation does
not --- meter readings of load are noisy and are precisely what a
reconciliation corrects --- so ``pd``/``qd`` move into the state there.
Controllable transformers and switched shunts are decision variables to
an OPF and parameters to a power flow, so they move likewise.

Residual blocks
---------------

Returned in this order, labelled by :func:`residual_names`:

1. real power balance, one per bus (pu),
2. reactive power balance, one per bus (pu),
3. the angle reference, one row, ``va_slack - va_ref`` (rad).

The reference row is not bookkeeping. The AC equations are invariant
under adding a constant to every bus angle: rotate the whole solution
and the injections are unchanged. Without a row pinning one angle the
Jacobian is rank ``2n - 1``, not ``2n``, for purely structural reasons,
and every downstream method that inverts it --- Newton, the KKT solve
in the OPF, the reconciliation normal equations --- fails on a network
that is perfectly well posed physically. Pinning the slack angle
restores full rank. (This is the same argument that makes
:mod:`difflow_gas` carry boundary flows as state: a structural rank
deficiency is fixed in the formulation, not in the linear solver.)

What is NOT here
----------------

Voltage limits, generator boxes, thermal ratings and angle-difference
limits are *inequalities*, not equations. They live in
:mod:`difflow_power.opf`, which is the only consumer that can act on
them. Setpoints (a PV bus holding its voltage, a scheduled unit holding
its MW) are also not here: they are how a *power flow* chooses to close
an underdetermined system, and :mod:`difflow_power.powerflow` adds them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.network import PowerNetwork
from difflow_power.physics import (
    branch_admittances,
    branch_power_flows,
    build_ybus,
    voltage_rectangular,
)


class PowerState(NamedTuple):
    """A network state as arrays, ready for the equations.

    Produced by :meth:`PowerStateLayout.unpack_arrays`, which fills any
    block the layout does not carry from the network's own data, so the
    equations never have to ask whether a quantity is a variable or a
    parameter.

    Attributes:
        vm: voltage magnitudes (pu), shape ``(n_bus,)``.
        va: voltage angles (rad), shape ``(n_bus,)``.
        pg: real generation (pu), shape ``(n_gen,)``.
        qg: reactive generation (pu), shape ``(n_gen,)``.
        pd: real demand (pu), shape ``(n_bus,)``.
        qd: reactive demand (pu), shape ``(n_bus,)``.
        tap: transformer tap ratios, shape ``(n_branch,)``.
        shift: phase shift angles (rad), shape ``(n_branch,)``.
        b_shunt: bus shunt susceptance (pu), shape ``(n_bus,)``.
    """

    vm: Array
    va: Array
    pg: Array
    qg: Array
    pd: Array
    qd: Array
    tap: Array
    shift: Array
    b_shunt: Array

    @property
    def v_complex(self) -> Array:
        """Complex bus voltages ``vm exp(j va)``."""
        return voltage_rectangular(self.vm, self.va)


@dataclass
class PowerStateLayout(ParamsMixin):
    """Packing of an AC network state into a flat vector.

    Build one with :func:`power_state_layout` rather than by hand; it
    snapshots the network's bus, branch and generator orders, which are
    insertion orders on the network's dicts and would otherwise be
    re-derived on every access.

    Attributes:
        buses: bus ids, in bus order (the ``vm``/``va`` block order).
        generators: generator ids (the ``pg``/``qg`` block order).
        branches: branch ids, for the optional tap/shift blocks.
        demand_buses: buses whose demand is a state variable.
        tap_branches: branches whose tap ratio is a state variable.
        shift_branches: branches whose phase shift is a state variable.
        shunt_buses: buses whose shunt susceptance is a state variable.
    """

    buses: list[str]
    generators: list[str]
    branches: list[str] = field(default_factory=list)
    demand_buses: list[str] = field(default_factory=list)
    tap_branches: list[str] = field(default_factory=list)
    shift_branches: list[str] = field(default_factory=list)
    shunt_buses: list[str] = field(default_factory=list)

    # -- sizes ------------------------------------------------------------

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_gen(self) -> int:
        return len(self.generators)

    @property
    def n_demand(self) -> int:
        return len(self.demand_buses)

    @property
    def n_extra(self) -> int:
        """Size of the optional tap / shift / shunt blocks."""
        return (
            len(self.tap_branches)
            + len(self.shift_branches)
            + len(self.shunt_buses)
        )

    @property
    def size(self) -> int:
        """Length of the packed state vector."""
        return (
            2 * self.n_bus + 2 * self.n_gen + 2 * self.n_demand + self.n_extra
        )

    @property
    def n_residual(self) -> int:
        """Length of the residual vector: two balances per bus, plus one."""
        return 2 * self.n_bus + 1

    # -- names and slices -------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Variable names, in packed order."""
        return (
            [f"vm_{b}" for b in self.buses]
            + [f"va_{b}" for b in self.buses]
            + [f"pg_{g}" for g in self.generators]
            + [f"qg_{g}" for g in self.generators]
            + [f"pd_{b}" for b in self.demand_buses]
            + [f"qd_{b}" for b in self.demand_buses]
            + [f"tap_{a}" for a in self.tap_branches]
            + [f"shift_{a}" for a in self.shift_branches]
            + [f"bsh_{b}" for b in self.shunt_buses]
        )

    @property
    def slice_vm(self) -> slice:
        return slice(0, self.n_bus)

    @property
    def slice_va(self) -> slice:
        return slice(self.n_bus, 2 * self.n_bus)

    @property
    def slice_pg(self) -> slice:
        return slice(2 * self.n_bus, 2 * self.n_bus + self.n_gen)

    @property
    def slice_qg(self) -> slice:
        start = 2 * self.n_bus + self.n_gen
        return slice(start, start + self.n_gen)

    @property
    def slice_pd(self) -> slice:
        start = 2 * self.n_bus + 2 * self.n_gen
        return slice(start, start + self.n_demand)

    @property
    def slice_qd(self) -> slice:
        start = 2 * self.n_bus + 2 * self.n_gen + self.n_demand
        return slice(start, start + self.n_demand)

    @property
    def slice_extra(self) -> slice:
        start = 2 * self.n_bus + 2 * self.n_gen + 2 * self.n_demand
        return slice(start, self.size)

    @property
    def default_scale(self) -> Array:
        """Typical magnitude of each entry, for scaling unmeasured ones.

        Per unit does most of the work: magnitudes sit at 1.0 and
        injections within an order of it. Angles are the outlier ---
        a stressed transmission corridor runs 0.3 rad --- and taps are
        near 1.
        """
        return jnp.asarray(
            [1.0] * self.n_bus
            + [0.3] * self.n_bus
            + [1.0] * (2 * self.n_gen)
            + [1.0] * (2 * self.n_demand)
            + [1.0] * len(self.tap_branches)
            + [0.1] * len(self.shift_branches)
            + [0.1] * len(self.shunt_buses),
            dtype=jnp.float64,
        )

    def index(self, name: str) -> int:
        """Position of a named variable in the packed vector."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a state variable of this layout; "
                f"it carries {self.size} variables starting "
                f"{self.names[:4]}"
            ) from None

    def indices(self, names: Sequence[str]) -> list[int]:
        """Positions of several named variables."""
        return [self.index(n) for n in names]

    # -- packing ----------------------------------------------------------

    def pack(
        self,
        vm,
        va,
        pg,
        qg,
        pd=None,
        qd=None,
        extra: dict[str, float] | None = None,
    ) -> Array:
        """Flatten the state blocks into one vector.

        Every block may be given as a dict keyed by bus / generator /
        branch id, or as an array already in this layout's order.

        Args:
            vm: voltage magnitudes (pu).
            va: voltage angles (rad).
            pg: real generation (pu).
            qg: reactive generation (pu).
            pd: real demand (pu) for :attr:`demand_buses`; required if
                the layout carries them.
            qd: reactive demand (pu), likewise.
            extra: ``{"tap_<branch>": v, "shift_<branch>": v,
                "bsh_<bus>": v}``; missing taps default to 1.0, missing
                shifts and shunts to 0.0.

        Returns:
            The packed state, shape ``(size,)``.
        """
        extra = extra or {}
        parts = [
            _as_ordered(vm, self.buses, "vm"),
            _as_ordered(va, self.buses, "va"),
            _as_ordered(pg, self.generators, "pg"),
            _as_ordered(qg, self.generators, "qg"),
        ]
        if self.n_demand:
            if pd is None or qd is None:
                raise ValueError(
                    "this layout carries demand as state; pass pd and qd"
                )
            parts.append(_as_ordered(pd, self.demand_buses, "pd"))
            parts.append(_as_ordered(qd, self.demand_buses, "qd"))
        parts.append(
            jnp.asarray(
                [extra.get(f"tap_{a}", 1.0) for a in self.tap_branches]
                + [extra.get(f"shift_{a}", 0.0) for a in self.shift_branches]
                + [extra.get(f"bsh_{b}", 0.0) for b in self.shunt_buses],
                dtype=jnp.float64,
            )
        )
        return jnp.concatenate(
            [jnp.atleast_1d(jnp.asarray(p, dtype=jnp.float64)) for p in parts]
        )

    def unpack(self, x: Array) -> dict[str, Array]:
        """Split a packed state into ``{variable name: scalar}``.

        For reporting. The equations use
        :meth:`unpack_arrays` instead, which stays in array form.
        """
        x = jnp.asarray(x)
        return {name: x[i] for i, name in enumerate(self.names)}

    def unpack_arrays(
        self,
        x: Array,
        network: PowerNetwork,
        branch_params: dict[str, Array] | None = None,
    ) -> PowerState:
        """Split a packed state into a :class:`PowerState` of arrays.

        Blocks the layout does not carry are filled from ``network``:
        demand from its loads, taps and shifts from its branches, shunt
        susceptance from its buses. That is what lets one equation set
        serve a power flow (where those are parameters) and an OPF or a
        reconciliation (where some of them are variables) without any
        branching in the equations themselves.

        Args:
            x: the packed state.
            network: the network supplying the defaults.
            branch_params: overrides for those defaults, applied BEFORE
                the state's own entries. The precedence matters and is
                deliberate: network default, then caller override, then
                state variable. A tap the layout carries is a decision
                variable and must win over a fixed value passed in; a
                tap the layout does not carry is a parameter, and
                passing it here is the only way to differentiate with
                respect to it.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape != (self.size,):
            raise ValueError(
                f"state has shape {x.shape}, expected {(self.size,)}"
            )
        vm = x[self.slice_vm]
        va = x[self.slice_va]
        pg = x[self.slice_pg]
        qg = x[self.slice_qg]

        pd, qd = network.load_arrays_pu()
        if self.n_demand:
            where = [network.bus_index[b] for b in self.demand_buses]
            pd = pd.at[jnp.asarray(where)].set(x[self.slice_pd])
            qd = qd.at[jnp.asarray(where)].set(x[self.slice_qd])

        params = network.branch_param_arrays()
        if branch_params:
            params = {**params, **branch_params}
        tap, shift = params["tap"], params["shift"]
        b_shunt = jnp.imag(network.shunt_array_pu())

        off = self.slice_extra.start
        if self.tap_branches:
            where = [network.branch_ids.index(a) for a in self.tap_branches]
            tap = tap.at[jnp.asarray(where)].set(
                x[off:off + len(self.tap_branches)]
            )
            off += len(self.tap_branches)
        if self.shift_branches:
            where = [network.branch_ids.index(a) for a in self.shift_branches]
            shift = shift.at[jnp.asarray(where)].set(
                x[off:off + len(self.shift_branches)]
            )
            off += len(self.shift_branches)
        if self.shunt_buses:
            where = [network.bus_index[b] for b in self.shunt_buses]
            b_shunt = b_shunt.at[jnp.asarray(where)].set(
                x[off:off + len(self.shunt_buses)]
            )

        return PowerState(vm, va, pg, qg, pd, qd, tap, shift, b_shunt)

    def embed(
        self, x: Array, source: "PowerStateLayout", fill: float = float("nan")
    ) -> Array:
        """Re-pack a vector from another layout into this one.

        Adding demand or a controllable tap to the state changes the
        pack order, so a measurement vector built for the plain layout
        does not fit the extended one. This maps by variable NAME,
        filling entries the source does not carry with ``fill``. ``nan``
        is right for a measurement vector, since the added entries are
        unmeasured and :func:`difflow.reconciliation.reconcile` masks
        them out.
        """
        x = jnp.asarray(x, dtype=jnp.float64)
        if x.shape != (source.size,):
            raise ValueError(
                f"x has shape {x.shape}, expected {(source.size,)} for the "
                "source layout"
            )
        where = {nm: i for i, nm in enumerate(source.names)}
        return jnp.asarray(
            [
                x[where[nm]] if nm in where else jnp.asarray(fill)
                for nm in self.names
            ],
            dtype=jnp.float64,
        )


def _as_ordered(values, keys: Sequence[str], block: str) -> Array:
    """Coerce a dict-or-array block into an array in ``keys`` order."""
    if isinstance(values, dict):
        missing = [k for k in keys if k not in values]
        if missing:
            raise KeyError(f"{block} block is missing {missing}")
        return jnp.asarray([values[k] for k in keys], dtype=jnp.float64)
    arr = jnp.asarray(values, dtype=jnp.float64)
    if arr.shape != (len(keys),):
        raise ValueError(
            f"{block} block has shape {arr.shape}, expected {(len(keys),)}"
        )
    return arr


def power_state_layout(
    network: PowerNetwork,
    *,
    demand_buses: Sequence[str] | None = None,
    tap_branches: Sequence[str] = (),
    shift_branches: Sequence[str] = (),
    shunt_buses: Sequence[str] = (),
) -> PowerStateLayout:
    """Build the state layout of a network.

    Args:
        network: the network whose state is to be packed.
        demand_buses: buses whose demand is an unknown rather than a
            parameter. ``None`` (the default) carries none; pass
            ``"all"`` semantics by giving every load bus explicitly.
            State estimation is what wants these.
        tap_branches: branches whose tap ratio is a decision variable.
            Must actually be transformers in the network sense --- a
            branch with line charging is not one.
        shift_branches: branches whose phase shift is a decision
            variable.
        shunt_buses: buses whose shunt susceptance is switchable.

    Returns:
        A :class:`PowerStateLayout`.

    Raises:
        ValueError: if any named bus or branch is not in the network.
    """
    for aid in list(tap_branches) + list(shift_branches):
        if aid not in network.branches:
            raise ValueError(f"branch {aid!r} is not in the network")
    for bid in list(demand_buses or ()) + list(shunt_buses):
        if bid not in network.buses:
            raise ValueError(f"bus {bid!r} is not in the network")
    for aid in tap_branches:
        if network.branches[aid].b != 0.0:
            raise ValueError(
                f"branch {aid!r} has line charging (b={network.branches[aid].b}), "
                "so it is a line, not a transformer; its tap is not a "
                "control"
            )
    return PowerStateLayout(
        buses=list(network.bus_ids),
        generators=list(network.generator_ids),
        branches=list(network.branch_ids),
        demand_buses=list(demand_buses or ()),
        tap_branches=list(tap_branches),
        shift_branches=list(shift_branches),
        shunt_buses=list(shunt_buses),
    )


def residual_names(network: PowerNetwork, layout: PowerStateLayout) -> list[str]:
    """Names of the residual entries, in the order they are returned."""
    return (
        [f"p_balance_{b}" for b in layout.buses]
        + [f"q_balance_{b}" for b in layout.buses]
        + [f"va_ref_{network.slack_bus}"]
    )


def state_ybus(
    network: PowerNetwork,
    state: PowerState,
    branch_params: dict[str, Array] | None = None,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    """Admittance matrix and branch blocks for a state.

    Kept separate from :func:`power_flow_residuals` because the branch
    blocks are wanted on their own for thermal limits and flow reports,
    and rebuilding them is the expensive part of an evaluation.

    Args:
        network: the network.
        state: unpacked state; supplies the taps, shifts and shunt
            susceptance actually in force.
        branch_params: optional overrides for ``r``, ``x``, ``b``,
            ``g``, ``tap`` and ``shift``. Traced values are fine ---
            this is how a solved state is differentiated with respect to
            a line reactance. A tap or shift the LAYOUT carries as a
            variable takes precedence, since it is a decision, not a
            parameter.

    Returns:
        ``(ybus, (yff, yft, ytf, ytt))``.
    """
    params = network.branch_param_arrays()
    if branch_params:
        unknown = set(branch_params) - set(params)
        if unknown:
            raise KeyError(
                f"unknown branch parameters {sorted(unknown)}; expected "
                f"some of {sorted(params)}"
            )
        params = {**params, **branch_params}
    # The tap and shift come from the STATE, which
    # :meth:`PowerStateLayout.unpack_arrays` has already resolved from
    # the network's defaults, any override in ``branch_params``, and the
    # layout's own variables, in that order of precedence. Reading them
    # from ``params`` here instead would drop a controllable tap the
    # layout carries.
    blocks = branch_admittances(
        params["r"], params["x"], params["b"],
        state.tap, state.shift, params["g"],
    )
    f_idx, t_idx = network.branch_index_arrays()
    y_shunt = jnp.real(network.shunt_array_pu()) + 1j * state.b_shunt
    ybus = build_ybus(network.n_bus, f_idx, t_idx, *blocks, y_shunt)
    return ybus, blocks


def bus_injection_arrays(
    network: PowerNetwork,
    state: PowerState,
    demand: tuple[Array, Array] | None = None,
) -> tuple[Array, Array]:
    """Net scheduled injection ``(p, q)`` per bus (pu), generation minus load.

    Several generators on one bus are summed, which is what a bus
    balance sees; the OPF still holds them apart as separate decisions.

    Args:
        network: the network.
        state: unpacked state, supplying the generation and the demand.
        demand: optional ``(pd, qd)`` per-bus arrays (pu) replacing the
            state's. Traced values are fine: this is how a solved state
            is differentiated with respect to load, which is the
            sensitivity every operator actually wants.
    """
    gen_idx = network.generator_bus_indices()
    p = jnp.zeros(network.n_bus, dtype=jnp.float64).at[gen_idx].add(state.pg)
    q = jnp.zeros(network.n_bus, dtype=jnp.float64).at[gen_idx].add(state.qg)
    pd, qd = (state.pd, state.qd) if demand is None else demand
    return p - pd, q - qd


def power_flow_residuals(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout,
    *,
    branch_params: dict[str, Array] | None = None,
    demand: tuple[Array, Array] | None = None,
) -> Array:
    """Full equation-oriented residual vector of a network state.

    Traceable, jittable and differentiable: ``x`` and the values in
    ``branch_params`` may all be JAX values, while everything taken from
    ``network`` and ``layout`` is static Python. Passing a coefficient
    here rather than editing the network is what makes it
    differentiable --- :class:`~difflow_power.network.Branch` validates
    its arguments with Python comparisons and so cannot be built under a
    trace.

    The equations are

    .. math::

        S_i^{sched} - V_i \\overline{(Y_{bus} V)_i} = 0,
        \\qquad \\theta_{slack} - \\theta_{ref} = 0

    with :math:`S^{sched} = (P_g - P_d) + j(Q_g - Q_d)`, split into real
    and imaginary parts. A positive residual means the bus is
    scheduled to inject more than the network can carry away at this
    voltage.

    Args:
        x: packed state, shape ``(layout.size,)``, from
            :meth:`PowerStateLayout.pack`.
        network: the network the state belongs to.
        layout: the packing used for ``x``.
        branch_params: optional overrides for the branch parameter
            arrays; see :func:`state_ybus`.
        demand: optional ``(pd, qd)`` per-bus arrays (pu) replacing the
            network's loads; see :func:`bus_injection_arrays`.

    Returns:
        Residuals in the order given by :func:`residual_names`: ``n_bus``
        real balances (pu), ``n_bus`` reactive balances (pu), then one
        angle reference row (rad).

    Example:
        >>> layout = power_state_layout(net)
        >>> x = layout.pack(vm, va, pg, qg)
        >>> A = jax.jacobian(power_flow_residuals)(x, net, layout)
    """
    state = layout.unpack_arrays(x, network, branch_params)
    ybus, _ = state_ybus(network, state, branch_params)

    v = state.v_complex
    s_network = v * jnp.conj(ybus @ v)
    p_sched, q_sched = bus_injection_arrays(network, state, demand)

    slack = network.bus_index[network.slack_bus]
    va_ref = network.buses[network.slack_bus].va_reference

    return jnp.concatenate(
        [
            p_sched - jnp.real(s_network),
            q_sched - jnp.imag(s_network),
            jnp.atleast_1d(state.va[slack] - va_ref),
        ]
    )


def branch_flows(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout,
    *,
    branch_params: dict[str, Array] | None = None,
) -> tuple[Array, Array]:
    """Complex power entering each branch at its two ends (pu).

    Both measured INTO the branch, so ``s_from + s_to`` is the branch's
    own loss. Thermal limits apply to the larger of the two magnitudes,
    because a lossy branch carries more at its sending end.

    Returns:
        ``(s_from, s_to)``, complex, shape ``(n_branch,)``.
    """
    state = layout.unpack_arrays(x, network, branch_params)
    _, blocks = state_ybus(network, state, branch_params)
    f_idx, t_idx = network.branch_index_arrays()
    return branch_power_flows(state.v_complex, f_idx, t_idx, *blocks)


def total_losses(
    x: Array,
    network: PowerNetwork,
    layout: PowerStateLayout,
    *,
    branch_params: dict[str, Array] | None = None,
) -> Array:
    """Total real power lost in the branches (pu).

    The sum over branches of ``Re(s_from + s_to)``. Bus shunts are NOT
    included: their loss is a load, not a transmission loss.
    """
    s_from, s_to = branch_flows(x, network, layout, branch_params=branch_params)
    return jnp.sum(jnp.real(s_from + s_to))
