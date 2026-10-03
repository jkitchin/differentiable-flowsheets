"""Optional product fractionator after a hydrotreater (#328): a TBP sigmoid split, NOT a column.

A distillate hydrotreater fed kerosene, diesel and AGO together makes one
treated liquid (the stripper bottoms). A refinery fractionates it into jet
and diesel. :func:`fractionate` does that the way the FCC and hydrocracker
fractionators of this plugin do: an idealized, differentiable split of the
product grid's components by their TBP boiling point, not a stage column.

The split. Products ``p_1 .. p_n`` (lightest first) are separated by
``n - 1`` cut points ``T_1 < ... < T_{n-1}`` (TBP, K). Component ``i`` of the
product grid (:attr:`~.unit.HydrotreaterResult.product_char`: liquid light
ends, then the treated cuts) with boiling point ``Tb_i`` goes to product
``k`` with the share::

    a_k(Tb) = sigmoid((Tb - T_k) / w),   a_0 = 1, a_n = 0
    share_k = a_{k-1} - a_k

-- the logistic step of the FCC main fractionator
(:mod:`difflow_refinery.fcc.fractionator`), generalised to ``n`` products.
The shares telescope to exactly one, so every component's moles (and the
treated attributes riding with them) are conserved to round-off, and a
product's yield and properties are smooth in every cut point -- also when
a cut point sits midway between two grid cuts, where a step that is flat
beyond ``+/- w`` (the hydrocracker's quintic) would give an exactly zero
derivative. ``w`` stands for the overlap of a real column's neighbouring
products; the default 8 K is ILLUSTRATIVE (a column calculation would set
it), not fitted to any column.

Dissolved real gases. The wild naphtha (and anything else fed that is not
the stripper bottoms) carries H2, H2S, NH3, C1 and C2 dissolved at the
drum's conditions; they are not blend components. They go whole to an
``off_gas`` stream (the fractionator's overhead gas), so the liquid
products are blendable and the mass balance still closes.

What it does NOT do: duties, reflux, side-stripper steam, product flash
points (a jet or diesel pool still takes a measured ``flash_C``), and the
D86 overlap a real column produces (``w`` is a stand-in). A StageColumn
fractionator could replace :func:`fractionate` without changing its
callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.hydroprocessing.layout import gas_mw

C_TO_K = 273.15

#: Default products of a distillate hydrotreater's fractionator, lightest first.
DEFAULT_PRODUCTS: tuple[str, ...] = ("jet", "diesel")
#: Default TBP cut point (K): jet / diesel at 240 C. ILLUSTRATIVE; a Jet A
#: end point is set by its freeze point and D86 FBP (300 C max), which the
#: cut point is the lever on.
DEFAULT_CUT_POINTS: tuple[float, ...] = (240.0 + C_TO_K,)
#: Default sigmoid width (K). ILLUSTRATIVE (see the module docstring).
DEFAULT_WIDTH: float = 8.0


def split_shares(Tb: Array, cut_points: Array, width) -> Array:
    """``(n_comp, n_products)`` share of each component going to each product (rows sum to 1)."""
    Tb = jnp.asarray(Tb, dtype=float)
    cp = jnp.asarray(cut_points, dtype=float).reshape(-1)
    above = jax.nn.sigmoid((Tb[:, None] - cp[None, :]) / width)            # (n, n_cut_points)
    ones = jnp.ones((Tb.shape[0], 1))
    zeros = jnp.zeros((Tb.shape[0], 1))
    return jnp.concatenate([ones, above], axis=1) - jnp.concatenate([above, zeros], axis=1)


@dataclass(frozen=True)
class FractionationResult:
    """The products of :func:`fractionate`.

    Attributes:
        products: ``{name: stream}``, each an ``F_<product_char.names>`` stream
            (mol/s) with ``T`` and ``P`` -- straight into
            ``BlendComponent.from_stream(name, stream, char)``.
        off_gas: The dissolved real gases (``F_<gas>``, mol/s), not on the blend grid.
        char: The product grid (``HydrotreaterResult.product_char``).
        rates: ``{name: kg/s}`` of every product and ``"off_gas"``.
        feed_rate: kg/s fed (the hydrotreater streams named in ``feeds``).
        balance: ``|sum of rates - feed_rate| / feed_rate``.
        cut_points: The cut points used (K).
    """

    products: dict
    off_gas: dict
    char: object
    rates: dict
    feed_rate: Array
    balance: Array
    cut_points: Array

    def table(self) -> str:
        """Product rates as text."""
        rows = [f"{k}: {float(v):.4f} kg/s ({100 * float(v) / float(self.feed_rate):.2f} wt%)"
                for k, v in self.rates.items()]
        return "\n".join(rows + [f"mass closure {float(self.balance):.1e}"])


def fractionate(res, cut_points: Sequence = DEFAULT_CUT_POINTS, products: Sequence[str] = DEFAULT_PRODUCTS,
                width=DEFAULT_WIDTH, feeds: Sequence[str] = ("product",), T=298.15,
                P=101325.0) -> FractionationResult:
    """Split a hydrotreater's liquid product(s) into named products at TBP cut points.

    Args:
        res: A :class:`~.unit.HydrotreaterResult`.
        cut_points: ``len(products) - 1`` TBP cut points (K), increasing.
            Traceable: a product property is differentiable in them.
        products: Product names, lightest first.
        width: Sigmoid width (K), traceable.
        feeds: The hydrotreater streams fed: ``("product",)`` (the stripper
            bottoms; a distillate unit's jet/diesel split) or ``("product",
            "wild_naphtha")`` (a naphtha unit's light/heavy naphtha split).
        T, P: Temperature and pressure given to the product streams.

    Returns a :class:`FractionationResult`.
    """
    products = tuple(products)
    cp = jnp.stack([jnp.asarray(c, dtype=float) for c in cut_points]) if len(cut_points) else jnp.zeros(0)
    if cp.shape[0] != len(products) - 1:
        raise ValueError(f"{len(products)} products need {len(products) - 1} cut points, got {cp.shape[0]}")
    try:
        cpv = np.asarray(cp)
        if np.any(np.diff(cpv) <= 0.0):
            raise ValueError(f"cut points must increase, got {cpv}")
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        pass
    if "off_gas" in products:
        raise ValueError("'off_gas' is the dissolved-gas stream's name; pick another product name")
    char = res.product_char
    names = list(char.names)
    moles = jnp.zeros(len(names))
    gases: dict[str, Array] = {}
    feed_rate = jnp.asarray(0.0)
    for f in feeds:
        s = res.product_stream(f, T=T, P=P, gases=True)
        moles = moles + jnp.stack([jnp.asarray(s.get(f"F_{n}", 0.0), dtype=float) for n in names])
        for k, v in s.items():
            if k.startswith("F_") and k[2:] not in names:
                gases[k] = gases.get(k, 0.0) + v
        feed_rate = feed_rate + res.stream_mass(s)
    shares = split_shares(char.Tb, cp, width)
    mw = jnp.asarray(char.mw)
    out, rates = {}, {}
    for j, p in enumerate(products):
        m = moles * shares[:, j]
        st = {f"F_{n}": m[i] for i, n in enumerate(names)}
        st["T"] = jnp.asarray(T, dtype=float)
        st["P"] = jnp.asarray(P, dtype=float)
        out[p] = st
        rates[p] = jnp.sum(m * mw) / 1000.0
    off = dict(gases)
    off["T"] = jnp.asarray(T, dtype=float)
    off["P"] = jnp.asarray(P, dtype=float)
    rates["off_gas"] = sum((jnp.asarray(v) * gas_mw(k[2:]) / 1000.0 for k, v in gases.items()),
                           jnp.asarray(0.0))
    total = sum(rates.values(), jnp.asarray(0.0))
    balance = jnp.abs(total - feed_rate) / jnp.maximum(feed_rate, 1e-300)
    return FractionationResult(products=out, off_gas=off, char=char, rates=rates, feed_rate=feed_rate,
                               balance=balance, cut_points=cp)


__all__ = ["DEFAULT_PRODUCTS", "DEFAULT_CUT_POINTS", "DEFAULT_WIDTH", "split_shares", "FractionationResult",
           "fractionate"]
