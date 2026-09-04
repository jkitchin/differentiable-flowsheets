"""Electrical network data model: buses, branches, generators, loads.

The central object is :class:`PowerNetwork`, a plain description of a
balanced positive-sequence AC network that carries no solver state and
no solution. Everything downstream --- the equation set
(:mod:`difflow_power.residuals`), the power flow
(:mod:`difflow_power.powerflow`), the optimal power flow
(:mod:`difflow_power.opf`), state estimation
(:mod:`difflow_power.estimation`) --- is a function of one of these.

The model deliberately mirrors the MATPOWER case format, because that
is the lingua franca of the field: a network built here has the same
five component types, the same per-unit conventions and the same
branch model as ``case9.m``, so results can be compared row by row
against MATPOWER, PYPOWER, PowerModels or pandapower.

Component types
---------------

======================  =================================================
:class:`Bus`            a node: carries the complex voltage, a fixed
                        shunt, and voltage limits
:class:`Branch`         a line, transformer or phase shifter (one model,
                        see :func:`difflow_power.physics.branch_admittances`)
:class:`Generator`      a real/reactive injection with box limits and a
                        cost curve
:class:`Load`           a constant-power withdrawal
======================  =================================================

There is no separate shunt component: a shunt is a property of the bus
it hangs off, which is how every case format stores it and how the
admittance matrix uses it.

Bus kinds
---------

``"slack"`` (exactly one, the angle reference and the balancing
injection), ``"pv"`` (voltage-controlled, must host a generator) and
``"pq"`` (a load or transit bus). The kind matters to a *power flow*,
which uses it to decide what is specified and what is unknown; an OPF
ignores everything but the slack, because it chooses voltages and
injections itself subject to limits. :meth:`PowerNetwork.with_kinds`
re-labels a network without rebuilding it.

Units
-----

Impedances, susceptances and voltages are per unit on ``base_mva``;
powers are in MW / MVAr at the interface (that is how case files and
engineers quote them) and converted to per unit exactly once, by the
``*_pu`` accessors here. Angles are stored in radians; case files use
degrees, which :func:`difflow_power.cases` converts on the way in.

Ordering
--------

Buses, branches, generators and loads keep their INSERTION order, not
a sorted order. Bus labels are usually numeric strings, where sorting
would interleave ``"10"`` between ``"1"`` and ``"2"`` and make every
printed vector unreadable; insertion order instead reproduces the case
file row for row. The order is snapshotted into
:class:`~difflow_power.residuals.PowerStateLayout`, so nothing
downstream depends on re-deriving it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_power.physics import (
    DEFAULT_BASE_MVA,
    branch_admittances,
    build_ybus,
)

#: the three bus kinds of a power flow
BUS_KINDS = ("slack", "pv", "pq")


@dataclass
class Bus(ParamsMixin):
    """A network node.

    Attributes:
        kind: ``"slack"``, ``"pv"`` or ``"pq"``.
        base_kv: nominal line-to-line voltage (kV). Reporting only ---
            the per-unit equations never see it --- but needed to turn
            a pu voltage back into volts.
        vm_min: lower voltage magnitude limit (pu).
        vm_max: upper voltage magnitude limit (pu).
        vm_setpoint: voltage the bus is held at when it is a ``slack``
            or ``pv`` bus in a power flow. Ignored by the OPF, which
            chooses voltages itself.
        g_shunt_mw: shunt conductance, quoted as the MW it would draw at
            1.0 pu voltage (MATPOWER's ``Gs``).
        b_shunt_mvar: shunt susceptance, quoted as the MVAr it would
            INJECT at 1.0 pu voltage (MATPOWER's ``Bs``; positive is a
            capacitor, negative a reactor).
        va_reference: angle (rad) the slack bus is pinned to. Ignored
            for other kinds.
        name: optional human label for plots and reports.
    """

    kind: str = "pq"
    base_kv: float = 1.0
    vm_min: float = 0.9
    vm_max: float = 1.1
    vm_setpoint: float = 1.0
    g_shunt_mw: float = 0.0
    b_shunt_mvar: float = 0.0
    va_reference: float = 0.0
    name: str | None = None

    def __post_init__(self):
        if self.kind not in BUS_KINDS:
            raise ValueError(
                f"bus kind {self.kind!r} is not one of {BUS_KINDS}"
            )
        if self.vm_min > self.vm_max:
            raise ValueError(
                f"bus voltage limits cross: vm_min={self.vm_min} > "
                f"vm_max={self.vm_max}"
            )
        if self.base_kv <= 0.0:
            raise ValueError(
                f"bus base_kv must be positive, got {self.base_kv}"
            )


@dataclass
class Branch(ParamsMixin):
    """A transmission line, transformer or phase shifter.

    One model serves all three; see
    :func:`difflow_power.physics.branch_admittances`. A line has
    ``tap=1, shift=0``; a tap-changing transformer has ``b=0`` and
    ``tap != 1``; a phase shifter has ``shift != 0``.

    Attributes:
        from_bus: bus id at the tap side.
        to_bus: bus id at the other side.
        r: series resistance (pu).
        x: series reactance (pu).
        b: TOTAL line charging susceptance (pu), half applied at each
            end.
        g: total line charging conductance (pu); zero in every standard
            case.
        tap: off-nominal turns ratio magnitude at the from end. A case
            file's 0 means "no transformer" and is normalised to 1.0
            here, so nothing downstream has to know that convention.
        shift: phase shift (RADIANS), positive meaning the from side
            leads. Case files store degrees.
        rate_mva: long-term thermal rating (MVA), or ``None`` for an
            unlimited branch. A case file's 0 also means unlimited.
        angle_min: minimum ``va_from - va_to`` (rad), or ``None``.
        angle_max: maximum ``va_from - va_to`` (rad), or ``None``.
        name: optional human label.
    """

    from_bus: str
    to_bus: str
    r: float = 0.0
    x: float = 0.0
    b: float = 0.0
    g: float = 0.0
    tap: float = 1.0
    shift: float = 0.0
    rate_mva: float | None = None
    angle_min: float | None = None
    angle_max: float | None = None
    name: str | None = None

    def __post_init__(self):
        if self.from_bus == self.to_bus:
            raise ValueError(
                f"branch {self.name or ''} is a self-loop at "
                f"{self.from_bus!r}"
            )
        if self.tap == 0.0:
            self.tap = 1.0          # case-file convention: 0 means 1
        if self.tap < 0.0:
            raise ValueError(f"branch tap must be positive, got {self.tap}")
        if self.r == 0.0 and self.x == 0.0:
            raise ValueError(
                "branch has zero impedance (r = x = 0); model a closed "
                "switch as a small reactance, not as a short circuit"
            )
        if self.rate_mva is not None:
            if self.rate_mva == 0.0:
                self.rate_mva = None      # case-file convention
            elif self.rate_mva < 0.0:
                raise ValueError(
                    f"branch rating must be positive, got {self.rate_mva}"
                )

    @property
    def is_transformer(self) -> bool:
        """True if the branch has an off-nominal tap or a phase shift."""
        return self.tap != 1.0 or self.shift != 0.0


@dataclass
class Generator(ParamsMixin):
    """A dispatchable real and reactive power injection.

    Attributes:
        bus: bus id it injects at.
        p_min_mw: minimum real output (MW). Negative is legitimate ---
            that is how a pumped-storage unit or a curtailable import
            is modelled.
        p_max_mw: maximum real output (MW).
        q_min_mvar: minimum reactive output (MVAr).
        q_max_mvar: maximum reactive output (MVAr).
        cost: polynomial cost coefficients, HIGHEST order first, with
            real power in MW and cost in $/h --- MATPOWER's ``gencost``
            order. The default is a flat $0 curve.
        p_mw: initial / scheduled real output (MW), used as the OPF's
            starting point and as a power flow's PV setpoint.
        q_mvar: initial reactive output (MVAr).
        vm_setpoint: voltage the unit regulates its bus to (pu), or
            ``None`` to take the bus's own setpoint.
        name: optional human label.
    """

    bus: str
    p_min_mw: float = 0.0
    p_max_mw: float = 0.0
    q_min_mvar: float = 0.0
    q_max_mvar: float = 0.0
    cost: tuple[float, ...] = (0.0,)
    p_mw: float = 0.0
    q_mvar: float = 0.0
    vm_setpoint: float | None = None
    name: str | None = None

    def __post_init__(self):
        self.cost = tuple(float(c) for c in self.cost)
        if self.p_min_mw > self.p_max_mw:
            raise ValueError(
                f"generator at {self.bus!r}: p_min_mw={self.p_min_mw} > "
                f"p_max_mw={self.p_max_mw}"
            )
        if self.q_min_mvar > self.q_max_mvar:
            raise ValueError(
                f"generator at {self.bus!r}: q_min_mvar={self.q_min_mvar} > "
                f"q_max_mvar={self.q_max_mvar}"
            )


@dataclass
class Load(ParamsMixin):
    """A constant-power withdrawal.

    Constant power is the standard steady-state model and the hardest
    one for a solver (a constant-impedance load would be linear in
    ``V^2``); voltage-dependent ZIP loads are a generalisation this
    plugin does not carry.

    Attributes:
        bus: bus id it draws from.
        p_mw: real demand (MW), positive drawing power.
        q_mvar: reactive demand (MVAr), positive drawing lagging vars.
        name: optional human label.
    """

    bus: str
    p_mw: float = 0.0
    q_mvar: float = 0.0
    name: str | None = None


@dataclass
class PowerNetwork:
    """A balanced positive-sequence AC network.

    Attributes:
        buses: bus id -> :class:`Bus`, in insertion order.
        branches: branch id -> :class:`Branch`, in insertion order.
        generators: generator id -> :class:`Generator`.
        loads: load id -> :class:`Load`. Several loads may share a bus;
            they are summed by :meth:`load_arrays_pu`.
        base_mva: system power base. Every per-unit quantity is on it.
        name: optional label for the case.

    Raises:
        ValueError: on a structurally invalid network --- no slack bus
            or more than one, a component attached to an unknown bus, a
            PV bus with no generator, or a network split into islands.
    """

    buses: dict[str, Bus]
    branches: dict[str, Branch]
    generators: dict[str, Generator] = field(default_factory=dict)
    loads: dict[str, Load] = field(default_factory=dict)
    base_mva: float = DEFAULT_BASE_MVA
    name: str | None = None

    def __post_init__(self):
        if not self.buses:
            raise ValueError("network has no buses")
        if self.base_mva <= 0.0:
            raise ValueError(f"base_mva must be positive, got {self.base_mva}")

        for bid, br in self.branches.items():
            for end in (br.from_bus, br.to_bus):
                if end not in self.buses:
                    raise ValueError(
                        f"branch {bid!r} attaches to unknown bus {end!r}"
                    )
        for gid, gen in self.generators.items():
            if gen.bus not in self.buses:
                raise ValueError(
                    f"generator {gid!r} attaches to unknown bus {gen.bus!r}"
                )
        for lid, load in self.loads.items():
            if load.bus not in self.buses:
                raise ValueError(
                    f"load {lid!r} attaches to unknown bus {load.bus!r}"
                )

        slacks = [b for b, bus in self.buses.items() if bus.kind == "slack"]
        if len(slacks) != 1:
            raise ValueError(
                "a network needs exactly one slack bus (the angle "
                f"reference and balancing injection); found {len(slacks)}: "
                f"{slacks}"
            )
        gen_buses = {g.bus for g in self.generators.values()}
        for bid, bus in self.buses.items():
            if bus.kind in ("pv", "slack") and bid not in gen_buses:
                raise ValueError(
                    f"bus {bid!r} is a {bus.kind} bus but hosts no "
                    "generator; a bus can only hold its voltage if "
                    "something injects vars there"
                )

        islands = self.islands()
        if len(islands) > 1:
            raise ValueError(
                f"network splits into {len(islands)} islands "
                f"{[sorted(i)[:4] for i in islands]}; solve each "
                "separately, each with its own slack bus"
            )

    # -- topology ---------------------------------------------------------

    @property
    def bus_ids(self) -> list[str]:
        """Bus ids in insertion order (the state vector's bus order)."""
        return list(self.buses)

    @property
    def branch_ids(self) -> list[str]:
        """Branch ids in insertion order."""
        return list(self.branches)

    @property
    def generator_ids(self) -> list[str]:
        """Generator ids in insertion order."""
        return list(self.generators)

    @property
    def bus_index(self) -> dict[str, int]:
        """Bus id -> position in the bus order."""
        return {b: i for i, b in enumerate(self.buses)}

    @property
    def slack_bus(self) -> str:
        """The single slack bus id."""
        return next(b for b, bus in self.buses.items() if bus.kind == "slack")

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_branch(self) -> int:
        return len(self.branches)

    @property
    def n_gen(self) -> int:
        return len(self.generators)

    def buses_of_kind(self, kind: str) -> list[str]:
        """Bus ids of one kind, in bus order."""
        if kind not in BUS_KINDS:
            raise ValueError(f"unknown bus kind {kind!r}")
        return [b for b, bus in self.buses.items() if bus.kind == kind]

    def generators_at(self, bus_id: str) -> list[str]:
        """Ids of the generators attached to a bus, in generator order."""
        return [g for g, gen in self.generators.items() if gen.bus == bus_id]

    def adjacency(self) -> dict[str, set[str]]:
        """Bus id -> set of buses one branch away (undirected)."""
        adj: dict[str, set[str]] = {b: set() for b in self.buses}
        for br in self.branches.values():
            adj[br.from_bus].add(br.to_bus)
            adj[br.to_bus].add(br.from_bus)
        return adj

    def islands(self) -> list[set[str]]:
        """Connected components of the network, largest first.

        More than one means the power flow has no unique solution: each
        island needs its own angle reference and its own balancing
        injection.
        """
        adj = self.adjacency()
        seen: set[str] = set()
        found: list[set[str]] = []
        for start in self.buses:
            if start in seen:
                continue
            comp = {start}
            queue = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                for nbr in adj[node]:
                    if nbr not in seen:
                        seen.add(nbr)
                        comp.add(nbr)
                        queue.append(nbr)
            found.append(comp)
        return sorted(found, key=len, reverse=True)

    @property
    def cycle_rank(self) -> int:
        """Number of independent loops; 0 means a radial network.

        Radial networks (distribution feeders) admit the sequential
        forward/backward sweep of
        :mod:`difflow_power.flowsheet`; meshed ones do not.
        """
        return self.n_branch - self.n_bus + len(self.islands())

    @property
    def is_radial(self) -> bool:
        """True if the network has no loops."""
        return self.cycle_rank == 0

    # -- numerical arrays -------------------------------------------------

    def branch_index_arrays(self) -> tuple[Array, Array]:
        """From- and to-bus INDEX arrays, shape ``(n_branch,)`` each.

        Static integer arrays: the topology is never differentiated.
        """
        idx = self.bus_index
        f = jnp.asarray(
            [idx[br.from_bus] for br in self.branches.values()], dtype=int
        )
        t = jnp.asarray(
            [idx[br.to_bus] for br in self.branches.values()], dtype=int
        )
        return f, t

    def branch_param_arrays(self) -> dict[str, Array]:
        """Branch parameters as ``{name: (n_branch,) array}``.

        Keys ``r``, ``x``, ``b``, ``g``, ``tap``, ``shift``. Pass these
        (or a modified copy) to
        :func:`difflow_power.physics.branch_admittances` to differentiate
        with respect to a line parameter.
        """
        brs = list(self.branches.values())
        return {
            key: jnp.asarray(
                [getattr(br, key) for br in brs], dtype=jnp.float64
            )
            for key in ("r", "x", "b", "g", "tap", "shift")
        }

    def shunt_array_pu(self) -> Array:
        """Per-bus shunt admittance (pu), shape ``(n_bus,)``, complex.

        ``Ysh = (Gs + j Bs) / base_mva`` with ``Gs``, ``Bs`` in MW and
        MVAr at 1.0 pu, matching the case-file convention.
        """
        return jnp.asarray(
            [
                (bus.g_shunt_mw + 1j * bus.b_shunt_mvar) / self.base_mva
                for bus in self.buses.values()
            ],
            dtype=jnp.complex128,
        )

    def load_arrays_pu(self) -> tuple[Array, Array]:
        """Per-bus aggregated demand ``(pd, qd)`` in per unit.

        Several loads on one bus are summed; a bus with no load gets 0.
        """
        idx = self.bus_index
        pd = [0.0] * self.n_bus
        qd = [0.0] * self.n_bus
        for load in self.loads.values():
            i = idx[load.bus]
            pd[i] += load.p_mw / self.base_mva
            qd[i] += load.q_mvar / self.base_mva
        return (
            jnp.asarray(pd, dtype=jnp.float64),
            jnp.asarray(qd, dtype=jnp.float64),
        )

    def generator_bus_indices(self) -> Array:
        """Bus index of each generator, shape ``(n_gen,)``.

        The incidence used to map a generator dispatch vector onto bus
        injections; several generators may share an index.
        """
        idx = self.bus_index
        return jnp.asarray(
            [idx[g.bus] for g in self.generators.values()], dtype=int
        )

    def ybus(self, params: dict[str, Array] | None = None) -> Array:
        """Bus admittance matrix (pu), shape ``(n_bus, n_bus)``, complex.

        Args:
            params: optional overrides for the branch parameter arrays
                from :meth:`branch_param_arrays`. Traced values are
                fine, which is how a solved state is differentiated with
                respect to a line reactance.

        Returns:
            The dense complex admittance matrix.
        """
        p = self.branch_param_arrays()
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise KeyError(
                    f"unknown branch parameters {sorted(unknown)}; "
                    f"expected some of {sorted(p)}"
                )
            p = {**p, **params}
        f_idx, t_idx = self.branch_index_arrays()
        yff, yft, ytf, ytt = branch_admittances(
            p["r"], p["x"], p["b"], p["tap"], p["shift"], p["g"]
        )
        return build_ybus(
            self.n_bus, f_idx, t_idx, yff, yft, ytf, ytt, self.shunt_array_pu()
        )

    def branch_rate_array_pu(self, default: float = jnp.inf) -> Array:
        """Thermal ratings (pu), shape ``(n_branch,)``.

        Unrated branches take ``default``, which is ``inf`` --- the
        honest encoding of "no limit", and one the OPF drops rather than
        carrying an inactive constraint.
        """
        return jnp.asarray(
            [
                default if br.rate_mva is None else br.rate_mva / self.base_mva
                for br in self.branches.values()
            ],
            dtype=jnp.float64,
        )

    # -- summaries and edits ----------------------------------------------

    @property
    def total_load_mw(self) -> float:
        return sum(load.p_mw for load in self.loads.values())

    @property
    def total_load_mvar(self) -> float:
        return sum(load.q_mvar for load in self.loads.values())

    @property
    def total_capacity_mw(self) -> float:
        return sum(g.p_max_mw for g in self.generators.values())

    def with_kinds(self, kinds: dict[str, str]) -> "PowerNetwork":
        """A copy with some bus kinds relabelled.

        Useful for switching a case between power-flow conventions ---
        making a PV bus the slack, or turning a generator bus into a PQ
        bus to model a unit on manual excitation.
        """
        buses = {
            bid: (replace(bus, kind=kinds[bid]) if bid in kinds else bus)
            for bid, bus in self.buses.items()
        }
        unknown = set(kinds) - set(self.buses)
        if unknown:
            raise KeyError(f"unknown buses {sorted(unknown)}")
        return replace(self, buses=buses)

    def scaled_load(self, factor: float) -> "PowerNetwork":
        """A copy with every load scaled by ``factor``.

        The standard way to sweep a case towards its loadability limit.
        """
        loads = {
            lid: replace(
                load, p_mw=load.p_mw * factor, q_mvar=load.q_mvar * factor
            )
            for lid, load in self.loads.items()
        }
        return replace(self, loads=loads)

    def summary(self) -> str:
        """A one-paragraph description of the case."""
        kinds = {k: len(self.buses_of_kind(k)) for k in BUS_KINDS}
        n_xfmr = sum(br.is_transformer for br in self.branches.values())
        n_rated = sum(
            br.rate_mva is not None for br in self.branches.values()
        )
        return (
            f"{self.name or 'network'}: {self.n_bus} buses "
            f"({kinds['slack']} slack, {kinds['pv']} PV, {kinds['pq']} PQ), "
            f"{self.n_branch} branches ({n_xfmr} transformers, "
            f"{n_rated} rated), {self.n_gen} generators "
            f"({self.total_capacity_mw:.0f} MW capacity), "
            f"{len(self.loads)} loads ({self.total_load_mw:.1f} MW + "
            f"{self.total_load_mvar:.1f} MVAr), "
            f"{self.cycle_rank} loops, base {self.base_mva:.0f} MVA"
        )

    def __repr__(self) -> str:
        return f"PowerNetwork({self.summary()})"
