"""The FCC as a delta-base planning block (:func:`fcc_block`).

Like :func:`difflow_refinery.planning.cdu_block`: the block's ``fn(u)``
solves the heat-balanced unit at the levers ``u`` and returns the chosen
outputs, so its AD Jacobian is the FCC's delta vectors -- including the
heat-balance coupling (a ROT move changes catalyst circulation, which
changes coke and regenerator temperature), which a fixed yield vector in a
planning LP does not have.

Levers are any numeric :class:`~difflow_refinery.fcc.FCCParams` field
(``riser_outlet_T``, ``feed_T``, ``activity``, ``flue_o2``, ...) plus
``feed.<field>`` for the bulk feed (``feed.mass``, ``feed.ccr``,
``feed.Kw`` ...). Outputs are names in the unit's ``outputs`` dict
(``conversion``, ``yield.gasoline``, ``yield.c4``, ``regenerator_T``,
``gasoline.RON``, ``lco.cetane_index`` ...). Units are the unit's SI units
(K, kg/s, mass fractions) and are recorded in ``Block.metadata``; a
non-converged solve returns NaN, which the planner rejects.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow.planning.block import Block

from difflow_refinery.fcc.feed import FCCFeed
from difflow_refinery.fcc.unit import FCCUnit

DEFAULT_OUTPUTS = ("conversion", "yield.dry_gas", "yield.c3", "yield.c4", "yield.gasoline",
                   "yield.lco", "yield.slurry", "yield.coke", "regenerator_T", "cat_oil",
                   "air_rate", "gasoline.RON", "gasoline.sulfur", "lco.cetane_index")


def _units(name: str) -> str:
    if name.endswith("_T") or name in ("regenerator_T", "mix_T") or ".tbp" in name:
        return "K"
    if name in ("air_rate", "catalyst_circulation", "coke_burn") or name == "feed.mass":
        return "kg/s"
    if name.endswith("RON") or name.endswith("MON") or name.endswith("cetane_index"):
        return "-"
    return "-"


class FCCPlanningModel:
    """``u -> y`` for a planning block. ``solve(u)`` gives the full result."""

    def __init__(self, unit: FCCUnit, feed: FCCFeed, levers: Sequence[str],
                 outputs: Sequence[str], mask_nonconverged: bool = True):
        self.unit, self.feed = unit, feed
        self.levers, self.outputs = list(levers), list(outputs)
        self.mask = mask_nonconverged
        base = unit.theta(feed)["params"]
        u0 = []
        for lv in self.levers:
            if lv.startswith("feed."):
                u0.append(float(getattr(feed, lv[5:])))
            elif lv in base and base[lv].ndim == 0:
                u0.append(float(base[lv]))
            else:
                raise ValueError(f"unknown or non-scalar FCC lever {lv!r}")
        self.u0 = u0

    def solve(self, u) -> dict:
        u = jnp.asarray(u, dtype=float)
        feed_changes, overrides = {}, {}
        for i, lv in enumerate(self.levers):
            if lv.startswith("feed."):
                feed_changes[lv[5:]] = u[i]
            else:
                overrides[lv] = u[i]
        feed = dataclasses.replace(self.feed, **feed_changes) if feed_changes else self.feed
        return self.unit.solve(feed, **overrides)

    def __call__(self, u) -> Array:
        res = self.solve(u)
        y = jnp.stack([res["outputs"][n] for n in self.outputs])
        if self.mask:
            y = jnp.where(res["converged"], y, jnp.nan)
        return y


def fcc_block(unit: FCCUnit, feed: FCCFeed, levers: Sequence[str],
              outputs: Sequence[str] | None = None, *, name: str = "fcc",
              bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.05,
              jit: bool = True, mask_nonconverged: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for the FCC at ``feed``.

    Args:
        unit: The :class:`~difflow_refinery.fcc.FCCUnit`.
        feed: The base :class:`~difflow_refinery.fcc.FCCFeed`.
        levers: Lever names (see module docstring).
        outputs: Output names; default :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        bounds: ``{lever: (lo, hi)}``; default ``u0 (1 -/+ span)``.
        span: Default relative half-width of the bounds.
        jit: Compile the block.
        mask_nonconverged: NaN outputs where the heat balance did not converge.
        **kwargs: Passed to :class:`~difflow.planning.Block`.
    """
    outputs = list(outputs or DEFAULT_OUTPUTS)
    model = FCCPlanningModel(unit, feed, levers, outputs, mask_nonconverged)
    bounds = dict(bounds or {})
    lb, ub = [], []
    for lv, x in zip(model.levers, model.u0):
        lo, hi = bounds.get(lv, (x - abs(x) * span, x + abs(x) * span) if x else (-span, span))
        if not lo <= x <= hi:
            raise ValueError(f"{lv}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.fcc.fcc_block")
    metadata.setdefault("u_units", [_units(l) for l in levers])
    metadata.setdefault("y_units", [_units(o) for o in outputs])
    return Block(name=name, fn=model, u_names=list(levers), y_names=outputs,
                 lb=lb, ub=ub, u0=model.u0, jit=jit, metadata=metadata, **kwargs)
