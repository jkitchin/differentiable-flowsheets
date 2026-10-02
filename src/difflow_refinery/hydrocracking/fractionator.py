"""The hydrocracker's product fractionator: a documented SIMPLIFIED split, not a column.

The separator liquid (cracked product, unconverted oil, dissolved gases) is
split into off-gas, LPG, light naphtha, heavy naphtha, kerosene, diesel and
unconverted oil (UCO) by TBP cut points. This is **not** a stage column:
there is no MESH, no reflux, no duty. It is the idealized fractionation a
yield model uses, made differentiable:

* every **gas** goes whole to one product (:data:`GAS_PRODUCT`): H2, H2S,
  NH3, C1, C2 to off-gas; C3 and C4 to LPG; C5 and C6 light ends to light
  naphtha; water to sour water;
* every **cut** ``i`` is shared between neighbouring liquid products by a
  smooth step in its boiling point about each cut point ``T_c``::

      above_i(T_c) = S((Tb_i - T_c) / w),   S(x) = 0 (x <= -1), 1 (x >= 1),
                                              6 t^5 - 15 t^4 + 10 t^3 with t = (1 + x)/2 in between

  the quintic "smootherstep" on ``[-1, 1]`` (C2, exactly 0 and 1 outside):
  product ``k`` gets ``above(T_{k-1}) - above(T_k)``. ``w`` (default 15 K)
  stands for the overlap a real fractionator leaves between neighbouring
  cuts; it is also what makes a product yield differentiable in its cut
  point on a grid of 20-25 K cuts. Because the step is exactly 1 above
  ``T_c + w``, the UCO is exactly empty below ``T_uco - w`` -- which lets the
  unit tear the UCO recycle on a fixed set of cuts.

Attributes follow their cut (``Flows.split``), so every element balance
closes exactly. The products' TBP curves, gravity and the planner's
properties are computed in :mod:`.unit` from the split streams.

What it does NOT do (stated in the docs): fractionator duties, side-stripper
steam, product flash points, and the D86 overlap a real column gives (the
step width is a stand-in, not a column calculation). The issue allows a
``StageColumn`` fractionator or "a documented simplified split"; this is the
second. A column on :class:`~difflow_refinery.vacuum.column.StageColumn` with
side draws would replace :func:`fractionate` without changing its callers.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.hydroprocessing.layout import Flows, Layout

#: Liquid products, lightest first, separated by the ``n - 1`` cut points.
LIQUID_PRODUCTS: tuple[str, ...] = ("light_naphtha", "heavy_naphtha", "kerosene", "diesel", "uco")
#: Every product the fractionator makes.
PRODUCTS: tuple[str, ...] = ("off_gas", "lpg") + LIQUID_PRODUCTS + ("sour_water",)
#: Where each gas species goes.
GAS_PRODUCT: dict[str, str] = {
    "hydrogen": "off_gas", "hydrogen_sulfide": "off_gas", "ammonia": "off_gas", "methane": "off_gas",
    "ethane": "off_gas", "propane": "lpg", "isobutane": "lpg", "n_butane": "lpg",
    "isopentane": "light_naphtha", "n_pentane": "light_naphtha", "n_hexane": "light_naphtha",
    "water": "sour_water",
}
#: Default TBP cut points (K): LN/HN 85 C, HN/kerosene 165 C, kerosene/diesel 260 C, diesel/UCO 370 C.
DEFAULT_CUT_POINTS: tuple[float, ...] = (358.15, 438.15, 533.15, 643.15)


def smootherstep(x: Array) -> Array:
    """``S(x)`` on ``[-1, 1]``: 0 below, 1 above, ``6t^5 - 15t^4 + 10t^3`` with ``t = (1 + x)/2`` between."""
    t = jnp.clip((jnp.asarray(x, dtype=float) + 1.0) / 2.0, 0.0, 1.0)
    return t**3 * (t * (6.0 * t - 15.0) + 10.0)


def cut_shares(Tb: Array, cut_points: Array, width) -> Array:
    """``(n_cut, 5)`` share of each cut going to each liquid product."""
    above = smootherstep((Tb[:, None] - cut_points[None, :]) / width)          # (n, 4)
    ones = jnp.ones((Tb.shape[0], 1))
    zeros = jnp.zeros((Tb.shape[0], 1))
    hi = jnp.concatenate([ones, above], axis=1)
    lo = jnp.concatenate([above, zeros], axis=1)
    return hi - lo


def fractionate(liquid: Flows, layout: Layout, Tb: Array, cut_points: Sequence, width) -> dict[str, Flows]:
    """Split ``liquid`` into :data:`PRODUCTS` (see the module docstring)."""
    cp = jnp.stack([jnp.asarray(c, dtype=float) for c in cut_points])
    if cp.shape[0] != len(LIQUID_PRODUCTS) - 1:
        raise ValueError(f"need {len(LIQUID_PRODUCTS) - 1} cut points")
    shares = cut_shares(jnp.asarray(Tb, dtype=float), cp, width)
    out = {}
    for name in PRODUCTS:
        gf = jnp.asarray([1.0 if GAS_PRODUCT.get(g, "off_gas") == name else 0.0 for g in layout.gases])
        if name in LIQUID_PRODUCTS:
            cf = shares[:, LIQUID_PRODUCTS.index(name)]
        else:
            cf = jnp.zeros(layout.n_cut)
        out[name] = liquid.split(gf, cf)
    return out


def uco_cuts(Tb, uco_cut_point: float, width: float, margin: float = 30.0) -> np.ndarray:
    """Indices of the cuts that can carry UCO when its cut point stays within ``margin`` K of ``uco_cut_point``."""
    Tb = np.asarray(Tb, dtype=float)
    return np.nonzero(Tb > uco_cut_point - width - margin)[0]


__all__ = ["LIQUID_PRODUCTS", "PRODUCTS", "GAS_PRODUCT", "DEFAULT_CUT_POINTS", "smootherstep", "cut_shares",
           "fractionate", "uco_cuts"]
