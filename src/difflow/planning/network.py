"""Networks of planning blocks, linked output-to-input.

A :class:`Network` is a directed acyclic graph of :class:`~difflow.planning
.block.Block` objects.  A link ``("ngl.residue_F", "power.fuel_F")`` says the
downstream block's *input* is not a free decision — it is whatever the
upstream block produces.

Blocks are linearised individually and the links become equality rows in the
LP.  That is the structure a delta-base planning model actually has, and at
first order it is equivalent to differentiating the composed chain: LP
elimination of the link rows reproduces the chain rule.  Keeping the blocks
separate preserves the delta vectors as an inspectable artefact, which is the
thing planners audit.

Recycles *inside* a block are expected and are exactly where difflow earns
its keep — a flowsheet's tear solve is differentiated implicitly, so the
reduced Jacobian comes back for free.  Recycles *between* blocks are rejected:
merge the loop into a single block (i.e. a single flowsheet) instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import jax.numpy as jnp
from jax import Array

from difflow.planning.block import Block


@dataclass(frozen=True)
class Link:
    """A directed connection from a block output to a block input.

    Attributes:
        source: Qualified output name, ``"<block>.<y>"``.
        target: Qualified input name, ``"<block>.<u>"``.
    """

    source: str
    target: str

    @property
    def source_block(self) -> str:
        return self.source.split(".", 1)[0]

    @property
    def target_block(self) -> str:
        return self.target.split(".", 1)[0]


def _qualify(name: str, what: str) -> str:
    if "." not in name:
        raise ValueError(
            f"{what} {name!r} must be qualified as '<block>.<variable>'")
    return name


@dataclass
class Network:
    """A DAG of planning blocks.

    Attributes:
        blocks: The blocks, in any order.
        links: Connections as ``(source_output, target_input)`` pairs or
            :class:`Link` objects.

    Example:
        >>> net = Network([ngl, power],
        ...               links=[("ngl.residue_F", "power.fuel_F")])
        >>> net.decision_names
        ['ngl.ethane_recovery', 'ngl.T_coldbox', 'ngl.split', 'power.alloc']
    """

    blocks: list[Block]
    links: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if not self.blocks:
            raise ValueError("Network needs at least one block")
        names = [b.name for b in self.blocks]
        if len(set(names)) != len(names):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate block names: {dup}")
        self._by_name = {b.name: b for b in self.blocks}

        links: list[Link] = []
        for item in self.links:
            if isinstance(item, Link):
                link = item
            else:
                src, tgt = item
                link = Link(_qualify(src, "link source"),
                            _qualify(tgt, "link target"))
            self._validate_link(link)
            links.append(link)
        self.links = links

        targets = [l.target for l in links]
        if len(set(targets)) != len(targets):
            dup = sorted({t for t in targets if targets.count(t) > 1})
            raise ValueError(
                f"input(s) {dup} are the target of more than one link")
        self._linked_inputs = {l.target: l for l in links}
        self._order = self._topological_order()

    def _validate_link(self, link: Link) -> None:
        if link.source_block not in self._by_name:
            raise KeyError(f"link source refers to unknown block "
                           f"{link.source_block!r}")
        if link.target_block not in self._by_name:
            raise KeyError(f"link target refers to unknown block "
                           f"{link.target_block!r}")
        self._by_name[link.source_block].y_index(link.source)
        self._by_name[link.target_block].u_index(link.target)

    def _topological_order(self) -> list[str]:
        """Kahn's algorithm; raises on an inter-block recycle."""
        deps: dict[str, set[str]] = {b.name: set() for b in self.blocks}
        for link in self.links:
            if link.source_block != link.target_block:
                deps[link.target_block].add(link.source_block)
            else:
                raise ValueError(
                    f"link {link.source} -> {link.target} is a self-recycle on "
                    f"block {link.source_block!r}. Recycles belong *inside* a "
                    "block, where difflow differentiates the tear solve "
                    "implicitly.")
        order: list[str] = []
        ready = sorted(n for n, d in deps.items() if not d)
        remaining = dict(deps)
        while ready:
            n = ready.pop(0)
            order.append(n)
            del remaining[n]
            newly = []
            for m, d in remaining.items():
                if n in d:
                    d.discard(n)
                    if not d:
                        newly.append(m)
            ready.extend(sorted(newly))
            ready.sort()
        if remaining:
            cycle = sorted(remaining)
            raise ValueError(
                f"the planning network has a recycle among blocks {cycle}. "
                "difflow.planning linearises a DAG; merge the recycling blocks "
                "into a single block (a single flowsheet with a tear solve) so "
                "that the reduced Jacobian is obtained by implicit "
                "differentiation.")
        return order

    # -- structure -------------------------------------------------------

    @property
    def order(self) -> list[str]:
        """Block names in evaluation (topological) order."""
        return list(self._order)

    def block(self, name: str) -> Block:
        """Look up a block by name."""
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"unknown block {name!r}")

    def is_linked(self, qualified_input: str) -> bool:
        """True when an input is driven by a link rather than a decision."""
        return qualified_input in self._linked_inputs

    @property
    def decision_names(self) -> list[str]:
        """Qualified names of the free decisions, in block-declaration order."""
        out = []
        for b in self.blocks:
            for name in b.qualified_u():
                if not self.is_linked(name):
                    out.append(name)
        return out

    @property
    def output_names(self) -> list[str]:
        """Qualified names of all block outputs."""
        return [n for b in self.blocks for n in b.qualified_y()]

    @property
    def input_names(self) -> list[str]:
        """Qualified names of all block inputs, free and linked."""
        return [n for b in self.blocks for n in b.qualified_u()]

    @property
    def n_decisions(self) -> int:
        """Number of free decisions."""
        return len(self.decision_names)

    def decision_bounds(self) -> tuple[Array, Array]:
        """Lower and upper bounds on the free decisions."""
        lb, ub = [], []
        for b in self.blocks:
            for j, name in enumerate(b.qualified_u()):
                if not self.is_linked(name):
                    lb.append(b.lb[j])
                    ub.append(b.ub[j])
        return jnp.asarray(lb), jnp.asarray(ub)

    def decision_start(self) -> Array:
        """Nominal values of the free decisions (the cold start)."""
        vals = []
        for b in self.blocks:
            for j, name in enumerate(b.qualified_u()):
                if not self.is_linked(name):
                    vals.append(b.u0[j])
        return jnp.asarray(vals)

    # -- evaluation ------------------------------------------------------

    def evaluate(self, decisions: Array | Mapping[str, Any],
                 theta: Mapping[str, Mapping[str, Any]] | None = None,
                 ) -> "NetworkState":
        """Evaluate every block in topological order.

        Args:
            decisions: Free-decision values, either an array ordered like
                :attr:`decision_names` or a dict keyed by qualified name.
            theta: Optional ``{block_name: theta_dict}`` parameter override.

        Returns:
            A :class:`NetworkState` holding each block's inputs and outputs.
        """
        d = self.decision_array(decisions)
        pos = {name: i for i, name in enumerate(self.decision_names)}

        u_by_block: dict[str, list[Any]] = {}
        y_by_block: dict[str, Array] = {}
        values: dict[str, Any] = {}

        for bname in self._order:
            b = self._by_name[bname]
            u = []
            for j, qname in enumerate(b.qualified_u()):
                link = self._linked_inputs.get(qname)
                if link is None:
                    u.append(d[pos[qname]])
                else:
                    u.append(values[link.source])
            u_vec = jnp.stack([jnp.asarray(x, dtype=float) for x in u])
            th = None if theta is None else theta.get(bname)
            y_vec = b.evaluate(u_vec, th)
            u_by_block[bname] = u_vec
            y_by_block[bname] = y_vec
            for j, qname in enumerate(b.qualified_u()):
                values[qname] = u_vec[j]
            for i, qname in enumerate(b.qualified_y()):
                values[qname] = y_vec[i]

        return NetworkState(decisions=d, u=u_by_block, y=y_by_block,
                            values=values)

    def decision_array(self, decisions) -> Array:
        """Coerce dict-or-array decisions to an array in :attr:`decision_names` order.

        Args:
            decisions: An array of the right length, or a dict keyed by
                qualified decision name.

        Returns:
            A 1-D array ordered like :attr:`decision_names`.
        """
        if isinstance(decisions, Mapping):
            names = self.decision_names
            missing = [n for n in names if n not in decisions]
            if missing:
                raise KeyError(f"missing decision values for {missing}")
            return jnp.stack(
                [jnp.asarray(decisions[n], dtype=float) for n in names])
        arr = jnp.atleast_1d(jnp.asarray(decisions, dtype=float))
        if arr.shape != (self.n_decisions,):
            raise ValueError(
                f"decisions have shape {tuple(arr.shape)}, expected "
                f"({self.n_decisions},) to match decision_names")
        return arr

    def block_inputs(self, decisions) -> dict[str, Array]:
        """Full input vector for each block, resolving links."""
        return self.evaluate(decisions).u

    def clip(self, decisions) -> Array:
        """Clip free decisions to their bounds."""
        d = self.decision_array(decisions)
        lb, ub = self.decision_bounds()
        return jnp.clip(d, lb, ub)

    def __repr__(self) -> str:
        return (f"Network(blocks={[b.name for b in self.blocks]}, "
                f"links={len(self.links)}, n_decisions={self.n_decisions})")


@dataclass
class NetworkState:
    """Result of evaluating a network.

    Attributes:
        decisions: The free-decision array.
        u: ``{block: input array}`` with links resolved.
        y: ``{block: output array}``.
        values: Flat ``{qualified name: scalar}`` for every input and output.
    """

    decisions: Array
    u: dict[str, Array]
    y: dict[str, Array]
    values: dict[str, Any]

    def __getitem__(self, key: str):
        try:
            return self.values[key]
        except KeyError:
            raise KeyError(f"{key!r} is not a variable in this network")

    def get(self, key: str, default=None):
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, float]:
        """All variable values as plain floats."""
        return {k: float(v) for k, v in self.values.items()}
