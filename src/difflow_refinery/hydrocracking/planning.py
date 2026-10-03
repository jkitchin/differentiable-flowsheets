"""The hydrocracker as a delta-base planning block (``hcu_block``), modelled on ``hdt_block``.

Levers, in planner units:

====================  =========  ==================================================
``feed.bpd``          bbl/d      fresh feed rate (composition held at the base); with
                                 the catalyst volumes fixed, both LHSVs move with it
``crack.T_in``        C          cracking first-bed inlet temperature (the
                                 conversion handle; WABT is an output)
``pretreat.T_in``     C          pretreat first-bed inlet temperature
``h2_oil``            Nm3/m3     treat-gas H2/oil ratio
``pressure``          bar        reactor pressure
``uco_recycle``       -          fraction of the UCO recycled (recycle units only)
``uco.cut_point``     C          diesel/UCO TBP cut point of the fractionator
====================  =========  ==================================================

Outputs (:data:`HCU_OUTPUTS`): product rates in bbl/d (LPG, light and heavy
naphtha, kerosene, diesel, UCO bleed), conversion per pass and overall,
naphtha/middle-distillate ratio, chemical H2 and makeup (Nm3/h), cracking
WABT (C), diesel cetane index and sulfur, kerosene and diesel gravity.

A link from the vacuum unit is ``("vdu.vgo.bpd", "hcu.vgo.bpd")``: pass
``feed_product="vgo"`` to rename the feed lever. Reverse mode only
(``ad_mode="rev"`` is forced): the reactor's diffrax adjoint is a ``custom_vjp``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from difflow.planning.block import Block

from difflow_refinery.hydrocracking.unit import Hydrocracker
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3
from difflow_refinery.hydrotreating.unit import BARREL

_DAY = 86400.0
_C = 273.15

LEVER_UNITS = {"feed.bpd": "bbl/d", "crack.T_in": "C", "pretreat.T_in": "C", "h2_oil": "Nm3/m3",
               "pressure": "bar", "uco_recycle": "-", "uco.cut_point": "C"}

#: Outputs and their planner units.
HCU_OUTPUTS: dict[str, str] = {
    "lpg.bpd": "bbl/d", "light_naphtha.bpd": "bbl/d", "heavy_naphtha.bpd": "bbl/d", "kerosene.bpd": "bbl/d",
    "diesel.bpd": "bbl/d", "uco_bleed.bpd": "bbl/d", "conversion.per_pass": "-", "conversion.overall": "-",
    "naphtha_to_middle_distillate": "-", "h2.chemical": "Nm3/h", "h2.makeup": "Nm3/h", "wabt.crack": "C",
    "diesel.cetane_index": "-", "diesel.S_wppm": "wppm", "kerosene.sg": "-", "diesel.sg": "-",
    "h2.chemical_scf_bbl": "scf/bbl",
}
DEFAULT_OUTPUTS = ("heavy_naphtha.bpd", "kerosene.bpd", "diesel.bpd", "conversion.per_pass", "h2.chemical")


class HCUPlanningModel:
    """``u -> y`` for a :class:`Hydrocracker` at fixed feed composition."""

    def __init__(self, unit: Hydrocracker, feed: Mapping, levers: Sequence[str], outputs: Sequence[str],
                 lever_alias: Mapping[str, str] | None = None, mask_nonconverged: bool = True):
        self.unit = unit
        self.feed = dict(feed)
        self.alias = dict(lever_alias or {})
        self.levers = list(levers)
        self.canon = [self.alias.get(l, l) for l in self.levers]
        bad = [l for l in self.canon if l not in LEVER_UNITS]
        if bad:
            raise ValueError(f"unknown levers {bad}; available: {sorted(LEVER_UNITS)}")
        if "uco_recycle" in self.canon and not unit.recycle:
            raise ValueError("uco_recycle is a lever only on a unit built with the UCO recycle")
        bad = [o for o in outputs if o not in HCU_OUTPUTS]
        if bad:
            raise ValueError(f"unknown outputs {bad}; available: {sorted(HCU_OUTPUTS)}")
        self.outputs = list(outputs)
        self.mask_nonconverged = mask_nonconverged
        th = unit.theta(feed)
        self.Q0 = float(unit._feed_volume(th))
        p = unit.params
        self.base = {"feed.bpd": self.Q0 * _DAY / BARREL, "crack.T_in": float(p.T_crack) - _C,
                     "pretreat.T_in": float(p.T_pretreat) - _C, "h2_oil": float(p.h2_oil),
                     "pressure": float(p.P) / 1e5, "uco_recycle": float(p.uco_recycle),
                     "uco.cut_point": float(p.cut_points[-1]) - _C}
        self.u0 = [self.base[c] for c in self.canon]

    def solve(self, u):
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        v = {c: u[i] for i, c in enumerate(self.canon)}
        p = self.unit.params
        scale = v.get("feed.bpd", self.base["feed.bpd"]) / self.base["feed.bpd"]
        feed = {k: (val * scale if k.startswith("F_") else val) for k, val in self.feed.items()}
        cps = tuple(p.cut_points[:-1]) + (v.get("uco.cut_point", self.base["uco.cut_point"]) + _C,)
        params = dataclasses.replace(
            p, T_crack=v.get("crack.T_in", self.base["crack.T_in"]) + _C,
            T_pretreat=v.get("pretreat.T_in", self.base["pretreat.T_in"]) + _C,
            lhsv_pretreat=p.lhsv_pretreat * scale, lhsv_crack=p.lhsv_crack * scale,
            h2_oil=v.get("h2_oil", p.h2_oil), P=v.get("pressure", p.P / 1e5) * 1e5,
            uco_recycle=v.get("uco_recycle", p.uco_recycle), cut_points=cps)
        return self.unit.solve(feed, params=params, warn=False)

    def outputs_of(self, res) -> dict:
        o = res.outputs
        bpd = lambda k: o[f"{k}.volume"] * _DAY / BARREL
        out = {f"{k}.bpd": bpd(k) for k in ("lpg", "light_naphtha", "heavy_naphtha", "kerosene", "diesel")}
        out["uco_bleed.bpd"] = bpd("uco") * (1.0 - jnp.asarray(self.unit.params.uco_recycle)) \
            if "uco_recycle" not in self.canon else o["uco_bleed.rate"] / o["uco.rate"] * bpd("uco")
        for k in ("conversion.per_pass", "conversion.overall", "naphtha_to_middle_distillate",
                  "diesel.cetane_index", "diesel.S_wppm", "kerosene.sg", "diesel.sg", "h2.chemical_scf_bbl"):
            out[k] = o[k]
        out["h2.chemical"] = o["h2.chemical"] / MOL_PER_NM3 * 3600.0
        out["h2.makeup"] = o["h2.makeup"] / MOL_PER_NM3 * 3600.0
        out["wabt.crack"] = o["wabt.crack"] - _C
        return out

    def __call__(self, u):
        res = self.solve(u)
        allv = self.outputs_of(res)
        y = jnp.stack([jnp.asarray(allv[n], dtype=float) for n in self.outputs])
        if not self.mask_nonconverged:
            return y
        return jnp.where(res.converged, y, jnp.nan)


def hcu_block(unit: Hydrocracker, feed: Mapping, levers: Sequence[str], outputs: Sequence[str] | None = None, *,
              name: str = "hcu", feed_product: str | None = None,
              bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.05,
              jit: bool = False, mask_nonconverged: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a hydrocracker (arguments as for ``hdt_block``)."""
    alias = {}
    lv = list(levers)
    if feed_product is not None:
        lv = [f"{feed_product}.bpd" if l == "feed.bpd" else l for l in lv]
        alias[f"{feed_product}.bpd"] = "feed.bpd"
    outputs = list(outputs or DEFAULT_OUTPUTS)
    model = HCUPlanningModel(unit, feed, lv, outputs, alias, mask_nonconverged)
    lb, ub = [], []
    bounds = dict(bounds or {})
    for nme, x in zip(lv, model.u0):
        if nme in bounds:
            lo, hi = bounds[nme]
        else:
            w = abs(x) * span if x != 0 else span
            lo, hi = x - w, x + w
        lb.append(lo)
        ub.append(hi)
    kwargs.setdefault("ad_mode", "rev")
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.hydrocracking.hcu_block")
    metadata.setdefault("u_units", [LEVER_UNITS[c] for c in model.canon])
    metadata.setdefault("y_units", [HCU_OUTPUTS[o] for o in outputs])
    return Block(name=name, fn=model, u_names=lv, y_names=outputs, lb=lb, ub=ub, u0=model.u0, jit=jit,
                 metadata=metadata, **kwargs)


__all__ = ["HCUPlanningModel", "hcu_block", "HCU_OUTPUTS", "DEFAULT_OUTPUTS", "LEVER_UNITS"]
