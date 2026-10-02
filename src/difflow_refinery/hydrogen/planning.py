r"""The hydrogen network as a :class:`difflow.planning.Block` (``h2_block``).

Levers (planner units), built from a template :class:`HydrogenNetwork`:

=========================  ========  ==============================================
lever                      units     what it sets
=========================  ========  ==============================================
``<producer>.h2``          mol/s     the producer's H2 (its impurity mix is held)
``<producer>.purity``      mol%      the producer's purity
``<consumer>.makeup``      Nm3/h     the consumer's makeup H2 demand
``<header>.min_purge``     mol/s     H2 that must leave as purge
``<header>.psa.recovery``  \-        PSA H2 recovery
=========================  ========  ==============================================

The units match the other blocks' outputs, so the links are one-to-one:
``("reformer.h2.net_mol_s", "h2.reformer.h2")``,
``("reformer.h2.purity", "h2.reformer.purity")`` (both from
:func:`~difflow_refinery.reforming.planning.reformer_block`, purity in mol%)
and ``("nht.h2.makeup", "h2.nht.makeup")`` (from
:func:`~difflow_refinery.hydrotreating.planning.hdt_block`, Nm3/h).
:func:`link_reformer` and :func:`link_hdt` write them.

Outputs are any keys of :attr:`H2NetworkResult.outputs` (default:
``h2.surplus``, each header's purity and surplus, each consumer's purity and
purity margin, fuel gas and swing H2). Purities come out in mole fractions,
as the network reports them.

The block is a few dozen flops: ``ad_mode="rev"`` by default, either works.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.planning import Block
from difflow_refinery.hydrogen.network import H2, NM3_H_PER_MOL_S, HydrogenNetwork, output_units

_LEVER_UNITS = {"h2": "mol/s", "purity": "mol%", "makeup": "Nm3/h", "min_purge": "mol/s", "recovery": "-"}


class H2PlanningModel:
    """``u -> y`` for a :class:`HydrogenNetwork` (the ``fn`` of :func:`h2_block`)."""

    def __init__(self, network: HydrogenNetwork, levers: Sequence[str], outputs: Sequence[str]):
        self.network = network
        self.levers = list(levers)
        self.outputs = list(outputs)
        self._prod = {p.name: p for p in network.producers}
        self._cons = {c.name: c for c in network.consumers}
        self._head = {h.name: h for h in network.headers}
        base = network.solve()
        self.base = base
        self.u0 = []
        for k in self.levers:
            self.u0.append(float(self._base_value(k)))
        missing = sorted(set(self.outputs) - set(base.outputs))
        if missing:
            raise ValueError(f"unknown outputs {missing}; available: {sorted(base.outputs)}")

    def _split(self, lever: str):
        for name in list(self._prod) + list(self._cons) + list(self._head):
            if lever.startswith(name + "."):
                return name, lever[len(name) + 1:]
        raise ValueError(f"unknown lever {lever!r}")

    def _base_value(self, lever: str):
        name, what = self._split(lever)
        if name in self._prod and what == "h2":
            return self._prod[name].h2
        if name in self._prod and what == "purity":
            return 100.0 * self._prod[name].purity
        if name in self._cons and what == "makeup":
            return jnp.asarray(self._cons[name].h2_demand) * NM3_H_PER_MOL_S
        if name in self._head and what == "min_purge":
            return self._head[name].min_purge
        if name in self._head and what == "psa.recovery":
            if self._head[name].psa is None:
                raise ValueError(f"header {name} has no PSA")
            return self._head[name].psa.recovery
        raise ValueError(f"unknown lever {lever!r} (levers: <producer>.h2, <producer>.purity, "
                         "<consumer>.makeup, <header>.min_purge, <header>.psa.recovery)")

    def network_at(self, u) -> HydrogenNetwork:
        """The template network with the levers set to ``u``."""
        vals = {k: u[i] for i, k in enumerate(self.levers)}
        prods = []
        for p in self.network.producers:
            F = p.flows
            h2 = vals.get(f"{p.name}.h2", F[H2])
            y = vals[f"{p.name}.purity"] / 100.0 if f"{p.name}.purity" in vals else p.purity
            imp = F.at[H2].set(0.0)
            imp = imp / jnp.sum(imp)
            prods.append(dataclasses.replace(p, flows=(imp * h2 * (1.0 - y) / y).at[H2].set(h2)))
        cons = [dataclasses.replace(c, h2_demand=vals[f"{c.name}.makeup"] / NM3_H_PER_MOL_S)
                if f"{c.name}.makeup" in vals else c for c in self.network.consumers]
        heads = []
        for h in self.network.headers:
            ch = {}
            if f"{h.name}.min_purge" in vals:
                ch["min_purge"] = vals[f"{h.name}.min_purge"]
            if f"{h.name}.psa.recovery" in vals:
                ch["psa"] = dataclasses.replace(h.psa, recovery=vals[f"{h.name}.psa.recovery"])
            heads.append(dataclasses.replace(h, **ch) if ch else h)
        return self.network.replace(producers=prods, consumers=cons, headers=heads)

    def __call__(self, u) -> Array:
        o = self.network_at(jnp.asarray(u, dtype=float)).solve().outputs
        return jnp.stack([jnp.asarray(o[k], dtype=float) for k in self.outputs])


def default_levers(network: HydrogenNetwork) -> list[str]:
    """Every producer's ``h2`` and ``purity`` and every consumer's ``makeup``."""
    return ([f"{p.name}.{w}" for p in network.producers for w in ("h2", "purity")]
            + [f"{c.name}.makeup" for c in network.consumers])


def default_outputs(network: HydrogenNetwork) -> list[str]:
    """Surplus, header purities and surpluses, consumer purities and margins, fuel gas, swing H2."""
    base = network.solve().outputs
    out = ["h2.surplus"]
    for h in network.headers:
        out += [f"{h.name}.purity", f"{h.name}.h2_surplus"]
    for c in network.consumers:
        out.append(f"{c.name}.purity")
        if f"{c.name}.purity_margin" in base:
            out.append(f"{c.name}.purity_margin")
    out += ["fuel_gas.mol_s", "fuel_gas.h2_mol_s", "swing.h2"]
    return out


def h2_block(network: HydrogenNetwork, levers: Sequence[str] | None = None,
             outputs: Sequence[str] | None = None, *, name: str = "h2",
             bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.2,
             jit: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a hydrogen network.

    Args:
        network: The template :class:`HydrogenNetwork` (its values are the base point).
        levers: Lever names (module docstring); default :func:`default_levers`.
        outputs: Output names; default :func:`default_outputs`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}``; otherwise ``u0 (1 -/+ span)``, purities capped at 100.
        span: Default relative half-width of the bounds.
        jit: Compile the block.
        **kwargs: Passed to :class:`~difflow.planning.Block` (``ad_mode`` defaults to ``"rev"``).
    """
    levers = list(levers or default_levers(network))
    outputs = list(outputs or default_outputs(network))
    model = H2PlanningModel(network, levers, outputs)
    bounds = dict(bounds or {})
    lb, ub = [], []
    for k, x in zip(levers, model.u0):
        lo, hi = bounds.get(k, (x - abs(x) * span, x + abs(x) * span))
        if k.endswith(".purity") and k not in bounds:
            hi = min(hi, 100.0)
        if not lo <= x <= hi:
            raise ValueError(f"{k}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    kwargs.setdefault("ad_mode", "rev")
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.hydrogen.h2_block")
    metadata.setdefault("u_units", [_LEVER_UNITS[k.rsplit(".", 1)[-1]] for k in levers])
    metadata.setdefault("y_units", [output_units(k) for k in outputs])
    return Block(name=name, fn=model, u_names=levers, y_names=outputs, lb=lb, ub=ub, u0=model.u0,
                 jit=jit, metadata=metadata, **kwargs)


def link_reformer(reformer_block_name: str = "reformer", producer: str = "reformer",
                  h2_block_name: str = "h2") -> list[tuple[str, str]]:
    """Links from a ``reformer_block``'s ``h2.net_mol_s``/``h2.purity`` to a producer's levers."""
    return [(f"{reformer_block_name}.h2.net_mol_s", f"{h2_block_name}.{producer}.h2"),
            (f"{reformer_block_name}.h2.purity", f"{h2_block_name}.{producer}.purity")]


def link_hdt(hdt_block_name: str, consumer: str | None = None, h2_block_name: str = "h2") -> list[tuple[str, str]]:
    """The link from an ``hdt_block``'s ``h2.makeup`` (Nm3/h) to a consumer's ``makeup`` lever."""
    return [(f"{hdt_block_name}.h2.makeup", f"{h2_block_name}.{consumer or hdt_block_name}.makeup")]


__all__ = ["H2PlanningModel", "h2_block", "default_levers", "default_outputs", "link_reformer", "link_hdt"]
