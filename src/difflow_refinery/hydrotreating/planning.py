"""The hydrotreater as a delta-base planning block (``hdt_block``), modelled on ``cdu_block``.

Levers, in planner units:

==================  =========  ====================================================
``feed.bpd``        bbl/d      feed rate (the feed's composition is held at the
                               base); with the catalyst volume fixed, LHSV moves
                               with it
``reactor.T_in``    C          first-bed inlet temperature; every bed inlet moves
                               by the same amount (the WABT handle -- WABT itself
                               is an output, write a row on it)
``h2_oil``          Nm3/m3     treat-gas H2/oil ratio
``pressure``        bar        reactor pressure
``purge``           -          purge fraction
==================  =========  ====================================================

Outputs (any of :data:`HDT_OUTPUTS`): product sulfur and nitrogen (wppm),
gravity, cetane index, aromatics, product and wild-naphtha rates (bbl/d),
chemical H2 consumption and makeup (Nm3/h), WABT and total temperature rise
(C, K), recycle compressor power (kW), H2 partial pressure at the inlet (bar).

A link from the crude unit is ``("cdu.<product>.bpd", "hdt.feed.bpd")``;
:func:`link_cdu` builds those for every input the HDT block shares by name
with the CDU's outputs (e.g. name the lever's source by passing
``feed_product="diesel"``, which renames the lever ``diesel.bpd``).

Differentiation is reverse mode only (``ad_mode="rev"`` is forced): the
reactor's diffrax adjoint is a ``custom_vjp``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from difflow.planning.block import Block

from difflow_refinery.hydrotreating.unit import BARREL, Hydrotreater
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3

_DAY = 86400.0
_C = 273.15

LEVER_UNITS = {"feed.bpd": "bbl/d", "reactor.T_in": "C", "h2_oil": "Nm3/m3", "pressure": "bar", "purge": "-"}

#: Outputs and their planner units.
HDT_OUTPUTS: dict[str, str] = {
    "product.S_wppm": "wppm", "product.N_wppm": "wppm", "product.sg": "-", "product.api": "API",
    "product.cetane_index": "-", "product.aromatics_vol": "vol%", "product.bpd": "bbl/d",
    "naphtha.bpd": "bbl/d", "product.yield": "-", "h2.chemical": "Nm3/h", "h2.makeup": "Nm3/h",
    "h2.chemical_scf_bbl": "scf/bbl", "wabt": "C", "reactor.dT_total": "K", "compressor.power": "kW",
    "reactor.pH2_in": "bar", "hds.conversion": "-",
}
DEFAULT_OUTPUTS = ("product.S_wppm", "product.bpd", "h2.chemical", "h2.makeup", "wabt", "reactor.dT_total",
                   "product.cetane_index")


class HDTPlanningModel:
    """``u -> y`` for a :class:`Hydrotreater` at fixed feed composition (see module docstring)."""

    def __init__(self, unit: Hydrotreater, feed: Mapping, levers: Sequence[str], outputs: Sequence[str],
                 lever_alias: Mapping[str, str] | None = None, mask_nonconverged: bool = True):
        self.unit = unit
        self.feed = dict(feed)
        self.alias = dict(lever_alias or {})
        self.levers = list(levers)
        canon = [self.alias.get(l, l) for l in self.levers]
        bad = [l for l in canon if l not in LEVER_UNITS]
        if bad:
            raise ValueError(f"unknown levers {bad}; available: {sorted(LEVER_UNITS)}")
        self.canon = canon
        bad = [o for o in outputs if o not in HDT_OUTPUTS]
        if bad:
            raise ValueError(f"unknown outputs {bad}; available: {sorted(HDT_OUTPUTS)}")
        self.outputs = list(outputs)
        self.mask_nonconverged = mask_nonconverged
        th = unit.theta(feed)
        self.Q0 = float(unit._feed_volume(th))
        p = unit.params
        self.base = {"feed.bpd": self.Q0 * _DAY / BARREL, "reactor.T_in": float(p.T_in[0]) - _C,
                     "h2_oil": float(p.h2_oil), "pressure": float(p.P) / 1e5, "purge": float(p.purge)}
        self.u0 = [self.base[c] for c in canon]

    def solve(self, u):
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        v = {c: u[i] for i, c in enumerate(self.canon)}
        p = self.unit.params
        scale = v.get("feed.bpd", self.base["feed.bpd"]) / self.base["feed.bpd"]
        feed = {k: (val * scale if k.startswith("F_") else val) for k, val in self.feed.items()}
        dT = v.get("reactor.T_in", self.base["reactor.T_in"]) - self.base["reactor.T_in"]
        params = dataclasses.replace(
            p, T_in=tuple(jnp.asarray(t) + dT for t in p.T_in), lhsv=p.lhsv * scale,
            h2_oil=v.get("h2_oil", p.h2_oil), P=v.get("pressure", p.P / 1e5) * 1e5,
            purge=v.get("purge", p.purge))
        return self.unit.solve(feed, params=params, warn=False)

    def outputs_of(self, res) -> dict:
        o = res.outputs
        hour = 3600.0
        return {
            "product.S_wppm": o["product.S_wppm"], "product.N_wppm": o["product.N_wppm"],
            "product.sg": o["product.sg"], "product.api": o["product.api"],
            "product.cetane_index": o["product.cetane_index"], "product.aromatics_vol": o["product.aromatics_vol"],
            "product.bpd": o["product.volume"] * _DAY / BARREL, "naphtha.bpd": o["naphtha.volume"] * _DAY / BARREL,
            "product.yield": o["product.yield"],
            "h2.chemical": o["h2.chemical"] / MOL_PER_NM3 * hour, "h2.makeup": o["h2.makeup"] / MOL_PER_NM3 * hour,
            "h2.chemical_scf_bbl": o["h2.chemical_scf_bbl"], "wabt": o["wabt"] - _C,
            "reactor.dT_total": o["reactor.dT_total"], "compressor.power": o["compressor.power"] / 1e3,
            "reactor.pH2_in": o["reactor.pH2_in"] / 1e5, "hds.conversion": o["hds.conversion"],
        }

    def __call__(self, u):
        res = self.solve(u)
        allv = self.outputs_of(res)
        y = jnp.stack([jnp.asarray(allv[n], dtype=float) for n in self.outputs])
        if not self.mask_nonconverged:
            return y
        return jnp.where(res.converged, y, jnp.nan)


def hdt_block(unit: Hydrotreater, feed: Mapping, levers: Sequence[str], outputs: Sequence[str] | None = None, *,
              name: str = "hdt", feed_product: str | None = None,
              bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.1,
              jit: bool = False, mask_nonconverged: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a hydrotreater.

    Args:
        unit: The :class:`Hydrotreater` (its params are the base point).
        feed: The base feed stream (``F_<char.names>``).
        levers: From :data:`LEVER_UNITS` (``"feed.bpd"`` is renamed
            ``"<feed_product>.bpd"`` when ``feed_product`` is given).
        outputs: From :data:`HDT_OUTPUTS`; default :data:`DEFAULT_OUTPUTS`.
        name: Block name.
        feed_product: Name of the CDU product feeding the unit, so the feed
            lever is ``"<product>.bpd"`` and :func:`link_cdu` finds it.
        bounds, span: As for ``cdu_block``.
        jit: ``Hydrotreater.solve`` is already compiled; leave False.
        mask_nonconverged: Return NaN where the unit does not converge.
    """
    alias = {}
    lv = list(levers)
    if feed_product is not None:
        lv = [f"{feed_product}.bpd" if l == "feed.bpd" else l for l in lv]
        alias[f"{feed_product}.bpd"] = "feed.bpd"
    outputs = list(outputs or DEFAULT_OUTPUTS)
    model = HDTPlanningModel(unit, feed, lv, outputs, alias, mask_nonconverged)
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
    metadata.setdefault("source", "difflow_refinery.hydrotreating.hdt_block")
    metadata.setdefault("u_units", [LEVER_UNITS[c] for c in model.canon])
    metadata.setdefault("y_units", [HDT_OUTPUTS[o] for o in outputs])
    return Block(name=name, fn=model, u_names=lv, y_names=outputs, lb=lb, ub=ub, u0=model.u0, jit=jit,
                 metadata=metadata, **kwargs)


__all__ = ["HDTPlanningModel", "hdt_block", "HDT_OUTPUTS", "DEFAULT_OUTPUTS", "LEVER_UNITS"]
