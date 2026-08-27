"""JAX-traceable equation-oriented residuals of a gas network.

This module is the single definition of a network's equation set: one
flat state vector in, one residual array out, fully traceable, so that
``jax.jacobian`` yields the constraint Jacobian that data
reconciliation, observability analysis and equation-oriented
optimization all need.

:mod:`difflow_gas.verify` is the reporting layer over it, unflattening
the same vector into labelled dicts of floats. It reports every block
below except the compressor relation, which a sequential solve
satisfies by construction.

The equations are checked against an independent restatement in
:func:`tests.gas.test_residuals.reference_residuals` rather than
against ``verify``, which would be checking this code against itself.

State vector
------------

:class:`GasStateLayout` packs, in this order:

===================  ==========================================
block                entries
===================  ==========================================
``p_<node>``         node pressures (bar), sorted node order
``q_<arc>``          signed arc flows (kg/s), sorted arc order
``s_<node>``         node boundary flows (kg/s, + into network)
``eta_<arc>``        optional pipe efficiency multipliers
``ratio_<arc>``      optional compressor pressure ratios
``dp_<arc>``         optional control valve drops (bar)
===================  ==========================================

The boundary flows are *state*, not parameters. That is deliberate:
with the supplies fixed, the node-balance block of the Jacobian is the
network's incidence matrix, whose rank is only ``n_nodes - 1``, so the
Jacobian loses full row rank for purely structural reasons. Carrying
the supplies as variables gives every balance row a ``+1`` in its own
column and restores full rank. It also keeps unbalanced *measured*
nominations out of :class:`~difflow_gas.network.GasNetwork`, which
rejects supplies that do not sum to zero.

Residual blocks
---------------

Returned in this order, labelled by :func:`residual_names`:

1. nodal mass balance, one per node (kg/s),
2. resistance law of every pipe and resistor (bar^2),
3. pressure equality of every valve and short pipe (bar),
4. control valve drop relation (bar),
5. **compressor relation** ``p_to - ratio * p_from`` (bar).

``verify`` reports blocks 1-4 and drops block 5, because a sequential
solve satisfies it by construction and there is nothing to check. A
reconciliation or equation-oriented formulation cannot drop it: the
relation is what ties the two sides of a compressor station together,
and without it the downstream pressures are unconstrained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

from difflow_gas.network import GasNetwork
from difflow_gas.physics import EPS_FLOW

#: Pa^2 -> bar^2, the conversion applied to ``network.beta``
BETA_PA2_TO_BAR2 = 1.0e10


def _kind_arcs(network: GasNetwork, kinds: Sequence[str]) -> list[str]:
    """Sorted ids of the arcs whose kind is one of ``kinds``."""
    return sorted(
        aid for aid, a in network.arcs.items() if a.kind in kinds
    )


@dataclass
class GasStateLayout(ParamsMixin):
    """Packing of a gas network state into a flat vector.

    Build one with :func:`gas_state_layout` rather than by hand; it
    snapshots the derived node list, which
    :attr:`~difflow_gas.network.GasNetwork.nodes` recomputes on every
    access.

    Attributes:
        nodes: node ids, sorted (pressure block order)
        arcs: arc ids, sorted (flow block order)
        supply_nodes: nodes carrying a boundary flow variable
        efficiency_arcs: pipes/resistors whose efficiency multiplier is
            a state variable rather than 1.0
        ratio_arcs: compressors whose pressure ratio is a state
            variable rather than a fixed parameter
        cv_arcs: control valves whose drop is a state variable
    """

    nodes: list[str]
    arcs: list[str]
    supply_nodes: list[str]
    efficiency_arcs: list[str] = field(default_factory=list)
    ratio_arcs: list[str] = field(default_factory=list)
    cv_arcs: list[str] = field(default_factory=list)

    @property
    def n_p(self) -> int:
        return len(self.nodes)

    @property
    def n_q(self) -> int:
        return len(self.arcs)

    @property
    def n_s(self) -> int:
        return len(self.supply_nodes)

    @property
    def n_extra(self) -> int:
        return (
            len(self.efficiency_arcs)
            + len(self.ratio_arcs)
            + len(self.cv_arcs)
        )

    @property
    def size(self) -> int:
        """Length of the packed state vector."""
        return self.n_p + self.n_q + self.n_s + self.n_extra

    @property
    def names(self) -> list[str]:
        """Variable names, in packed order."""
        return (
            [f"p_{n}" for n in self.nodes]
            + [f"q_{a}" for a in self.arcs]
            + [f"s_{n}" for n in self.supply_nodes]
            + [f"eta_{a}" for a in self.efficiency_arcs]
            + [f"ratio_{a}" for a in self.ratio_arcs]
            + [f"dp_{a}" for a in self.cv_arcs]
        )

    @property
    def default_scale(self) -> Array:
        """Typical magnitude of each entry, for scaling unmeasured ones.

        Pressures in bar and flows in kg/s both sit around 50 in a
        transmission network; the dimensionless extras are O(1).
        """
        return jnp.array(
            [50.0] * self.n_p
            + [50.0] * self.n_q
            + [50.0] * self.n_s
            + [1.0] * self.n_extra,
            dtype=jnp.float64,
        )

    def index(self, name: str) -> int:
        """Position of a named variable in the packed vector."""
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a state variable of this layout; "
                f"expected one of {self.names}"
            ) from None

    def indices(self, names: Sequence[str]) -> list[int]:
        """Positions of several named variables."""
        return [self.index(n) for n in names]

    @property
    def slice_p(self) -> slice:
        return slice(0, self.n_p)

    @property
    def slice_q(self) -> slice:
        return slice(self.n_p, self.n_p + self.n_q)

    @property
    def slice_s(self) -> slice:
        return slice(self.n_p + self.n_q, self.n_p + self.n_q + self.n_s)

    @property
    def slice_extra(self) -> slice:
        return slice(self.n_p + self.n_q + self.n_s, self.size)

    def pack(
        self,
        p_bar: dict[str, float],
        q_kg_s: dict[str, float],
        s_kg_s: dict[str, float] | None = None,
        extra: dict[str, float] | None = None,
    ) -> Array:
        """Flatten pressures, flows, supplies and extras into a vector.

        Args:
            p_bar: node id -> pressure (bar)
            q_kg_s: arc id -> signed flow (kg/s)
            s_kg_s: node id -> boundary flow (kg/s); missing entries
                default to 0.0
            extra: name -> value for the ``eta_``/``ratio_``/``dp_``
                entries; missing efficiencies and ratios default to
                1.0, missing valve drops to 0.0

        Returns:
            The packed state, shape ``(layout.size,)``.
        """
        s_kg_s = s_kg_s or {}
        extra = extra or {}
        values = (
            [p_bar[n] for n in self.nodes]
            + [q_kg_s[a] for a in self.arcs]
            + [s_kg_s.get(n, 0.0) for n in self.supply_nodes]
            + [extra.get(f"eta_{a}", 1.0) for a in self.efficiency_arcs]
            + [extra.get(f"ratio_{a}", 1.0) for a in self.ratio_arcs]
            + [extra.get(f"dp_{a}", 0.0) for a in self.cv_arcs]
        )
        return jnp.asarray(values, dtype=jnp.float64)

    def unpack(
        self, x: Array
    ) -> tuple[dict[str, Array], dict[str, Array],
               dict[str, Array], dict[str, Array]]:
        """Split a packed state into ``(p_bar, q_kg_s, s_kg_s, extra)``.

        The values stay as JAX scalars, so this is safe under
        ``jit``/``grad``.
        """
        x = jnp.asarray(x)
        p = {n: x[i] for i, n in enumerate(self.nodes)}
        off = self.n_p
        q = {a: x[off + i] for i, a in enumerate(self.arcs)}
        off += self.n_q
        s = {n: x[off + i] for i, n in enumerate(self.supply_nodes)}
        off += self.n_s
        extra: dict[str, Array] = {}
        for i, a in enumerate(self.efficiency_arcs):
            extra[f"eta_{a}"] = x[off + i]
        off += len(self.efficiency_arcs)
        for i, a in enumerate(self.ratio_arcs):
            extra[f"ratio_{a}"] = x[off + i]
        off += len(self.ratio_arcs)
        for i, a in enumerate(self.cv_arcs):
            extra[f"dp_{a}"] = x[off + i]
        return p, q, s, extra


def gas_state_layout(
    network: GasNetwork,
    *,
    efficiency_arcs: Sequence[str] = (),
    ratio_arcs: Sequence[str] = (),
    cv_arcs: Sequence[str] = (),
    supply_nodes: Sequence[str] | None = None,
) -> GasStateLayout:
    """Build the state layout of a network.

    Args:
        network: the network whose state is to be packed.
        efficiency_arcs: pipes or resistors whose efficiency multiplier
            is an unknown (fouling, roughness drift) instead of 1.0.
        ratio_arcs: compressors whose pressure ratio is a state
            variable instead of a fixed parameter.
        cv_arcs: control valves whose drop is a state variable.
        supply_nodes: nodes carrying a boundary flow variable; defaults
            to every node with an entry in ``network.supply_kg_s``.

    Returns:
        A :class:`GasStateLayout`.

    Raises:
        ValueError: if any named arc is missing or of the wrong kind.
    """
    for aid in efficiency_arcs:
        if aid not in network.arcs:
            raise ValueError(f"efficiency arc {aid!r} is not in the network")
        if network.arcs[aid].kind not in ("pipe", "resistor"):
            raise ValueError(
                f"efficiency arc {aid!r} is a {network.arcs[aid].kind}, "
                "but only pipes and resistors have a resistance law"
            )
    for aid in ratio_arcs:
        if network.arcs.get(aid) is None or network.arcs[aid].kind != "compressor":
            raise ValueError(f"ratio arc {aid!r} is not a compressor")
    for aid in cv_arcs:
        if (
            network.arcs.get(aid) is None
            or network.arcs[aid].kind != "control_valve"
        ):
            raise ValueError(f"cv arc {aid!r} is not a control valve")

    nodes = list(network.nodes)
    if supply_nodes is None:
        supply_nodes = sorted(network.supply_kg_s)
    unknown = set(supply_nodes) - set(nodes)
    if unknown:
        raise ValueError(f"supply nodes not in the network: {sorted(unknown)}")

    return GasStateLayout(
        nodes=nodes,
        arcs=sorted(network.arcs),
        supply_nodes=list(supply_nodes),
        efficiency_arcs=sorted(efficiency_arcs),
        ratio_arcs=sorted(ratio_arcs),
        cv_arcs=sorted(cv_arcs),
    )


def residual_names(network: GasNetwork, layout: GasStateLayout) -> list[str]:
    """Names of the residual entries, in the order they are returned."""
    return (
        [f"balance_{n}" for n in layout.nodes]
        + [f"resistance_{a}" for a in _kind_arcs(network, ("pipe", "resistor"))]
        + [f"equality_{a}" for a in _kind_arcs(network, ("valve", "short_pipe"))]
        + [f"cv_{a}" for a in _kind_arcs(network, ("control_valve",))]
        + [f"compressor_{a}" for a in _kind_arcs(network, ("compressor",))]
    )


def _signed_square(q: Array, eps_flow: float) -> Array:
    """``q |q|``, optionally smoothed as ``q sqrt(q^2 + eps^2)``.

    The smoothed form is C-infinity everywhere, which matters because
    the exact ``q |q|`` has a discontinuous second derivative at zero.
    At the default ``eps_flow`` the relative bias is ``eps^2 / (2 q^2)``
    --- around 5e-9 at 1 kg/s, far below any meter's precision --- but
    pass ``eps_flow=0.0`` to recover ``verify``'s exact form.
    """
    if eps_flow > 0.0:
        return q * jnp.sqrt(q * q + eps_flow * eps_flow)
    return q * jnp.abs(q)


def network_residuals(
    x: Array,
    network: GasNetwork,
    layout: GasStateLayout,
    *,
    ratios: dict[str, float] | None = None,
    cv_drops_bar: dict[str, float] | None = None,
    efficiencies: dict[str, float] | None = None,
    eps_flow: float = EPS_FLOW,
) -> Array:
    """Full equation-oriented residual vector of a network state.

    Traceable, jittable and differentiable: ``x`` and the values in
    ``ratios``, ``cv_drops_bar`` and ``efficiencies`` may all be JAX
    values, while everything taken from ``network`` and ``layout`` is
    static Python. Passing a coefficient here rather than baking it
    into ``network.beta`` is what makes it differentiable ---
    :class:`~difflow_gas.network.GasNetwork` validates its arguments
    with Python comparisons and so cannot be built under a trace.

    Args:
        x: packed state, shape ``(layout.size,)``, from
            :meth:`GasStateLayout.pack`.
        network: the network the state belongs to.
        layout: the packing used for ``x``.
        ratios: compressor arc id -> pressure ratio, for stations whose
            ratio is *not* in the state (defaults to 1.0).
        cv_drops_bar: control valve arc id -> drop (bar), for valves
            whose drop is not in the state (defaults to 0.0).
        efficiencies: pipe/resistor arc id -> multiplier on ``beta``,
            for arcs whose efficiency is not in the state (defaults to
            1.0). Values above 1.0 mean *more* resistance, i.e.
            fouling. Use this to differentiate the reconciled state
            with respect to a fixed pipe coefficient; put the entry in
            the layout instead to *estimate* it.
        eps_flow: smoothing of ``|q|`` in the resistance law (kg/s);
            0.0 gives the exact non-smooth form.

    Returns:
        Residuals in the order given by :func:`residual_names`, mixing
        kg/s (balances), bar^2 (resistance) and bar (the rest).

    Example:
        >>> layout = gas_state_layout(net)
        >>> x = layout.pack(p_bar, q_kg_s, net.supply_kg_s)
        >>> A = jax.jacobian(network_residuals)(x, net, layout)
    """
    ratios = ratios or {}
    cv_drops_bar = cv_drops_bar or {}
    efficiencies = efficiencies or {}
    p, q, s, extra = layout.unpack(x)

    zero = jnp.asarray(0.0, dtype=jnp.float64)

    balance = []
    for node in layout.nodes:
        acc = s.get(node, zero)
        for aid, a in network.arcs.items():
            if a.from_node == node:
                acc = acc - q[aid]
            if a.to_node == node:
                acc = acc + q[aid]
        balance.append(acc)

    resistance = []
    for aid in _kind_arcs(network, ("pipe", "resistor")):
        a = network.arcs[aid]
        beta_bar2 = network.beta[aid] / BETA_PA2_TO_BAR2
        eta = extra.get(f"eta_{aid}", efficiencies.get(aid, 1.0))
        resistance.append(
            p[a.from_node] ** 2
            - p[a.to_node] ** 2
            - eta * beta_bar2 * _signed_square(q[aid], eps_flow)
        )

    equality = []
    for aid in _kind_arcs(network, ("valve", "short_pipe")):
        a = network.arcs[aid]
        equality.append(p[a.from_node] - p[a.to_node])

    control_valve = []
    for aid in _kind_arcs(network, ("control_valve",)):
        a = network.arcs[aid]
        dp = extra.get(f"dp_{aid}", cv_drops_bar.get(aid, 0.0))
        control_valve.append(p[a.from_node] - p[a.to_node] - dp)

    compressor = []
    for aid in _kind_arcs(network, ("compressor",)):
        a = network.arcs[aid]
        ratio = extra.get(f"ratio_{aid}", ratios.get(aid, 1.0))
        compressor.append(p[a.to_node] - ratio * p[a.from_node])

    blocks = balance + resistance + equality + control_valve + compressor
    return jnp.stack([jnp.asarray(b, dtype=jnp.float64) for b in blocks])
