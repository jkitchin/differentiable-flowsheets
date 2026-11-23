"""The delayed coker as a delta-base planning block (:func:`coker_block`).

Like :func:`difflow_refinery.fcc.fcc_block`: the block's ``fn(u)``
evaluates the coker at the levers ``u`` and returns the chosen outputs, so
its AD Jacobian is the coker's delta vectors. Levers are any numeric
:class:`~difflow_refinery.coking.CokerParams` field (``drum_P``,
``coil_outlet_T``, ``recycle``, ...), plus ``feed.<field>`` for the pooled
feed (``feed.SG``, ``feed.sulfur``, ``feed.ccr``, ...). Outputs are keys of
:meth:`~difflow_refinery.coking.DelayedCoker.solve`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.planning.block import Block

from difflow_refinery.coking.unit import CokerFeed, CokerParams, DelayedCoker

DEFAULT_OUTPUTS = ("yield.dry_gas", "yield.lpg", "yield.h2s", "yield.naphtha", "yield.diesel",
                   "yield.gas_oil", "yield.coke", "naphtha.SG", "gas_oil.SG", "coke.S_wt")


class CokerPlanningModel:
    """``u -> y`` for a planning block."""

    def __init__(self, unit: DelayedCoker, feed: CokerFeed, levers: Sequence[str],
                 outputs: Sequence[str]):
        self.unit, self.feed = unit, feed
        self.levers, self.outputs = list(levers), list(outputs)
        numeric = {f.name for f in dataclasses.fields(CokerParams)
                   if isinstance(getattr(unit.params, f.name), (int, float))}
        feed_fields = {f.name for f in dataclasses.fields(CokerFeed)}
        u0 = []
        for lv in self.levers:
            if lv.startswith("feed.") and lv[5:] in feed_fields:
                u0.append(float(getattr(feed, lv[5:])))
            elif lv in numeric:
                u0.append(float(getattr(unit.params, lv)))
            else:
                raise ValueError(f"unknown coker lever {lv!r}")
        known = set(unit.solve(feed))
        bad = [o for o in self.outputs if o not in known]
        if bad:
            raise ValueError(f"unknown coker output(s) {bad}")
        self.u0 = u0

    def __call__(self, u) -> Array:
        u = jnp.asarray(u, dtype=float)
        feed_changes, overrides = {}, {}
        for i, lv in enumerate(self.levers):
            (feed_changes if lv.startswith("feed.") else overrides)[lv.removeprefix("feed.")] = u[i]
        feed = dataclasses.replace(self.feed, **feed_changes) if feed_changes else self.feed
        res = self.unit.solve(feed, **overrides)
        return jnp.stack([res[n] for n in self.outputs])


def coker_block(unit: DelayedCoker, feed: CokerFeed, levers: Sequence[str],
                outputs: Sequence[str] | None = None, *, name: str = "coker",
                bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.05,
                jit: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for the delayed coker at ``feed``.

    Args:
        unit: The :class:`~difflow_refinery.coking.DelayedCoker`.
        feed: The base (pooled) :class:`~difflow_refinery.coking.CokerFeed`.
        levers: Lever names (see module docstring).
        outputs: Output names; default :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}``; default ``u0 (1 -/+ span)``.
        span: Default relative half-width of the bounds.
        jit: Compile the block.
        **kwargs: Passed to :class:`~difflow.planning.Block`.

    Raises:
        ValueError: On an unknown lever or output, or a base value outside
            its bounds.
    """
    outputs = list(outputs or DEFAULT_OUTPUTS)
    model = CokerPlanningModel(unit, feed, levers, outputs)
    bounds = dict(bounds or {})
    lb, ub = [], []
    for lv, x in zip(model.levers, model.u0):
        lo, hi = bounds.get(lv, (x - abs(x) * span, x + abs(x) * span) if x else (-span, span))
        if not lo <= x <= hi:
            raise ValueError(f"{lv}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.coking.coker_block")
    return Block(name=name, fn=model, u_names=list(levers), y_names=outputs,
                 lb=lb, ub=ub, u0=model.u0, jit=jit, metadata=metadata, **kwargs)
