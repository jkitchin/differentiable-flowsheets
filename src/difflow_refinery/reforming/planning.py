"""The reformer as a :class:`difflow.planning.Block` (``reformer_block``).

Levers (planner units):

==================  =======  ==================================================
lever               units    what it sets
==================  =======  ==================================================
``wait``            C        weighted average inlet temperature (all inlets move together)
``P_separator``     bar      separator pressure (the reactors run ``loop_dP`` above)
``H2_HC``           mol/mol  recycle hydrogen per mol naphtha hydrocarbon
``LHSV``            1/h      space velocity (catalyst inventory fixed by the base design)
``feed.rate``       kg/s     naphtha rate (composition held)
``feed.naphthenes`` vol%     naphtha naphthene content (other groups rescaled)
``c4_recovery``     \-       butane share to LPG (the RVP stand-in)
==================  =======  ==================================================

A note on ``LHSV`` and ``feed.rate``: the catalyst inventory follows from
the feed rate and LHSV (``ReformerParams.LHSV``), so with ``feed.rate`` a
lever and ``LHSV`` not, more feed means more catalyst, not a higher space
velocity. Make ``LHSV`` the lever for a fixed-feed severity study.

Outputs are the keys of :meth:`ReformerResult.outputs` (units in
:data:`~difflow_refinery.reforming.unit.OUTPUT_UNITS`): reformate yield,
RON/MON, aromatics, benzene, RVP, net hydrogen and purity, LPG and fuel gas,
reactor temperature drops, heater duties, compressor power, coke make and
cycle length. They link to a :class:`~difflow_refinery.BlendPool` component
(``reformate.RON`` ...) and to a hydrogen balance block (``h2.net_mol_s``).

The block runs the reformer's traced path -- difflow's implicit-differentiated
recycle fixed point, warm-started at the base solution -- so it is
differentiated in forward mode (``ad_mode="fwd"``, the default here): the
reactor integrations are ``diffrax.ForwardMode``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.planning import Block
from difflow_refinery.reforming.feed import NaphthaFeed
from difflow_refinery.reforming.unit import CatalyticReformer, ReformerParams, output_units

#: Levers and their units.
LEVERS: dict[str, str] = {
    "wait": "C", "P_separator": "bar", "H2_HC": "mol/mol", "LHSV": "1/h",
    "feed.rate": "kg/s", "feed.naphthenes": "vol%", "c4_recovery": "-",
}

#: Default outputs of :func:`reformer_block`.
DEFAULT_OUTPUTS: tuple[str, ...] = (
    "reformate.yield_vol", "reformate.RON", "reformate.MON", "reformate.aromatics_vol",
    "reformate.benzene_vol", "reformate.RVP_psi", "reformate.kg_s", "h2.net_mol_s",
    "h2.purity", "lpg.kg_s", "fuel_gas.kg_s", "heaters.fired_MW", "compressor.power_MW",
    "coke.kg_h", "rx1.dT")


class ReformerPlanningModel:
    """``u -> y`` for a reformer (the ``fn`` of :func:`reformer_block`)."""

    def __init__(self, reformer: CatalyticReformer, feed: NaphthaFeed, levers: Sequence[str],
                 outputs: Sequence[str], max_iter: int = 400, tol: float = 1e-10):
        unknown = sorted(set(levers) - set(LEVERS))
        if unknown:
            raise ValueError(f"unknown levers {unknown}; available: {sorted(LEVERS)}")
        self.reformer = reformer
        self.feed = feed
        self.levers = list(levers)
        self.outputs = list(outputs)
        self.max_iter = max_iter
        self.tol = tol
        base = reformer.solve(feed, tol=tol)
        if not base.converged:
            raise RuntimeError("the reformer did not converge at the base point")
        self.base = base
        self.tear = base.tear
        missing = sorted(set(self.outputs) - set(base.outputs()))
        if missing:
            raise ValueError(f"unknown outputs {missing}")
        p = reformer.params
        g = feed.group_fractions("volume")
        self._base_values = {
            "wait": float(p.wait) - 273.15, "P_separator": float(p.P_separator) / 1e5,
            "H2_HC": float(p.H2_HC), "LHSV": float(p.LHSV), "feed.rate": float(feed.mass_flow),
            "feed.naphthenes": 100.0 * float(g["naphthenes"]), "c4_recovery": float(p.c4_recovery),
        }
        self.u0 = [self._base_values[k] for k in self.levers]

    def _params_and_feed(self, u):
        p = self.reformer.params
        feed = self.feed
        vals = dict(zip(self.levers, [u[i] for i in range(len(self.levers))]))
        if "wait" in vals:
            p = p.with_wait(vals["wait"] + 273.15)
        upd = {}
        if "P_separator" in vals:
            upd["P_separator"] = vals["P_separator"] * 1e5
        for k in ("H2_HC", "LHSV", "c4_recovery"):
            if k in vals:
                upd[k] = vals[k]
        if upd:
            p = p.update(**upd)
        if "feed.naphthenes" in vals:
            feed = feed.with_group_fraction("naphthenes", vals["feed.naphthenes"] / 100.0)
        if "feed.rate" in vals:
            feed = feed.scaled(vals["feed.rate"] / feed.mass_flow)
        return p, feed

    def solve(self, u):
        p, feed = self._params_and_feed(jnp.asarray(u, dtype=float))
        return self.reformer.solve(feed, p, tear_initial=self.tear, tol=self.tol,
                                   max_iter=self.max_iter, on_nonconvergence="ignore")

    def __call__(self, u) -> Array:
        o = self.solve(u).outputs()
        return jnp.stack([jnp.asarray(o[k]) for k in self.outputs])


def reformer_block(reformer: CatalyticReformer, feed: NaphthaFeed, levers: Sequence[str],
                   outputs: Sequence[str] | None = None, *, name: str = "reformer",
                   bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.05,
                   jit: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a catalytic reformer.

    Args:
        reformer: The :class:`CatalyticReformer` (its params are the base point).
        feed: The base :class:`NaphthaFeed`.
        levers: Lever names (see :data:`LEVERS`).
        outputs: Output names; default :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}`` in planner units; otherwise
            ``u0 (1 -/+ span)``.
        span: Default relative half-width of the bounds.
        jit: Compile the block.
        **kwargs: Passed to :class:`~difflow.planning.Block`; ``ad_mode``
            defaults to ``"fwd"`` (required: see the module docstring).
    """
    outputs = list(outputs or DEFAULT_OUTPUTS)
    model = ReformerPlanningModel(reformer, feed, levers, outputs)
    bounds = dict(bounds or {})
    lb, ub = [], []
    for k, x in zip(model.levers, model.u0):
        lo, hi = bounds.get(k, (x - abs(x) * span, x + abs(x) * span))
        if not lo <= x <= hi:
            raise ValueError(f"{k}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    kwargs.setdefault("ad_mode", "fwd")
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.reforming.reformer_block")
    metadata.setdefault("u_units", [LEVERS[k] for k in model.levers])
    metadata.setdefault("y_units", [output_units(k) for k in outputs])
    return Block(name=name, fn=model, u_names=list(levers), y_names=outputs, lb=lb, ub=ub,
                 u0=model.u0, jit=jit, metadata=metadata, **kwargs)


__all__ = ["DEFAULT_OUTPUTS", "LEVERS", "ReformerPlanningModel", "reformer_block"]
