"""What a crude column's products are: rates, yields, gravity, boiling range.

A refinery reads a crude unit's output as yields (volume percent of crude),
gravities (API) and boiling ranges -- above all the *gap* between neighbouring
cuts: the 5 % point of the heavier product less the 95 % point of the
lighter. A positive gap means the column separates the pair cleanly, an
overlap (negative gap) that it does not. Fractionation is judged on it more
than on any tray temperature.

Boiling ranges here are TBP, built from the pseudo-components a product is
made of: each component's volume is centred on its boiling point and the
cumulative curve is interpolated linearly between those centres. That is the
approximation simulators make, and it is exactly as fine as the cuts: a
5 %/95 % point can be no better resolved than a cut's width (20 C below
400 C with the default cut points), and a gap of a few degrees is within it.
ASTM D86 conversion is not done; TBP is what the gap is defined on.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from difflow_refinery import correlations as corr
from difflow_refinery.column import BARREL
from difflow_refinery.thermo import RHO_WATER_60F, ColumnThermo

_DAY = 86400.0

#: The TBP points reported for every product (volume percent distilled).
TBP_POINTS = (5.0, 10.0, 50.0, 90.0, 95.0)


@dataclass(frozen=True)
class ProductProperties:
    """One hydrocarbon product.

    Attributes:
        name: The product.
        mole, mass, volume: Rate in mol/s, kg/s and standard m^3/s (60 F).
        bpd: Standard barrels per day.
        yield_volume, yield_mass: Fraction of the crude feed, by volume and
            by mass (``None`` without a feed to compare with).
        sg, api: Specific gravity (60/60 F) and API gravity.
        mw: Number-average molecular weight (g/mol).
        tbp: TBP temperatures (K) at :data:`TBP_POINTS`, volume basis.
    """

    name: str
    mole: Array
    mass: Array
    volume: Array
    bpd: Array
    yield_volume: Array | None
    yield_mass: Array | None
    sg: Array
    api: Array
    mw: Array
    tbp: Array

    def tbp_at(self, percent: float) -> Array:
        """The TBP temperature (K) at one of :data:`TBP_POINTS`."""
        return self.tbp[TBP_POINTS.index(float(percent))]


jax.tree_util.register_dataclass(
    ProductProperties,
    data_fields=["mole", "mass", "volume", "bpd", "yield_volume", "yield_mass", "sg", "api", "mw", "tbp"],
    meta_fields=["name"],
)


def _component_flows(stream: dict, thermo: ColumnThermo) -> Array:
    return jnp.stack([jnp.asarray(stream[f"F_{n}"], dtype=float) for n in thermo.names])


def tbp_curve(flows: Array, thermo: ColumnThermo, percents=TBP_POINTS) -> Array:
    """TBP temperatures (K) of a mixture of the thermo's components.

    Each component's standard volume is centred on its normal boiling point;
    the cumulative volume curve through those centres is interpolated
    linearly, and held flat beyond the first and last component present.
    Differentiable in the flows and in the boiling points.
    """
    vol = flows * thermo.MW / (1000.0 * thermo.SG * RHO_WATER_60F)
    order = jnp.argsort(thermo.Tb)
    v, Tb = vol[order], thermo.Tb[order]
    total = jnp.sum(v)
    centre = (jnp.cumsum(v) - 0.5 * v) / total
    # Components a product holds none of sit at centre 0 or 1 with their
    # own boiling points, and would drag the ends of the curve to them. The
    # curve is held flat outside the centres of the components it contains.
    present = v > 1e-10 * total
    lo = jnp.min(jnp.where(present, centre, 1.0))
    hi = jnp.max(jnp.where(present, centre, 0.0))
    frac = jnp.clip(jnp.asarray(percents, dtype=float) / 100.0, lo, hi)
    return jnp.interp(frac, centre, Tb)


def product_properties(products: dict, thermo: ColumnThermo, feed: dict | None = None
                       ) -> dict[str, ProductProperties]:
    """Rates, yields, gravities and TBP ranges of a column's products.

    Args:
        products: Product streams, as in :attr:`CrudeColumnResult.products`.
            The decanted ``"water"`` is skipped, and so is the water vapour
            that leaves in an offgas: these are hydrocarbon properties.
        thermo: The column's thermo.
        feed: The crude feed stream, for yields. Optional.

    Returns:
        ``{name: ProductProperties}`` in the products' order.
    """
    if feed is not None:
        f = _component_flows(feed, thermo)
        feed_volume, feed_mass = thermo.std_volume(f), thermo.mass(f)
    out = {}
    for name, stream in products.items():
        if name == "water":
            continue
        flows = _component_flows(stream, thermo)
        mole = jnp.sum(flows)
        mass = thermo.mass(flows)
        volume = thermo.std_volume(flows)
        sg = mass / (volume * RHO_WATER_60F)
        out[name] = ProductProperties(
            name=name, mole=mole, mass=mass, volume=volume, bpd=volume * _DAY / BARREL,
            yield_volume=None if feed is None else volume / feed_volume,
            yield_mass=None if feed is None else mass / feed_mass,
            sg=sg, api=corr.api_from_sg(sg), mw=1000.0 * mass / mole,
            tbp=tbp_curve(flows, thermo),
        )
    return out


def gaps(properties: dict[str, ProductProperties], order) -> dict[tuple[str, str], Array]:
    """The 5-95 gap (K) between each neighbouring pair of products.

    ``TBP5(heavier) - TBP95(lighter)``: positive is a gap (clean
    separation), negative an overlap.

    Args:
        properties: From :func:`product_properties`.
        order: Product names, lightest first.
    """
    return {(a, b): properties[b].tbp_at(5) - properties[a].tbp_at(95)
            for a, b in zip(order[:-1], order[1:])}


def table(properties: dict[str, ProductProperties]) -> str:
    """A plain-text yield table, lightest product first: rate, yield, API, TBP range."""
    rows = [f"{'product':<10} {'bbl/d':>9} {'vol %':>6} {'wt %':>6} {'API':>6} "
            f"{'TBP5 C':>7} {'TBP50 C':>8} {'TBP95 C':>8}"]
    def c(T):
        return round(float(T) - 273.15) + 0  # + 0: no "-0"

    for name, p in sorted(properties.items(), key=lambda kv: float(kv[1].tbp_at(50))):
        yv = "" if p.yield_volume is None else f"{100 * float(p.yield_volume):6.1f}"
        ym = "" if p.yield_mass is None else f"{100 * float(p.yield_mass):6.1f}"
        rows.append(
            f"{name:<10} {float(p.bpd):9.0f} {yv:>6} {ym:>6} {float(p.api):6.1f} "
            f"{c(p.tbp_at(5)):7d} {c(p.tbp_at(50)):8d} {c(p.tbp_at(95)):8d}"
        )
    return "\n".join(rows)


__all__ = ["ProductProperties", "TBP_POINTS", "gaps", "product_properties", "table", "tbp_curve"]
