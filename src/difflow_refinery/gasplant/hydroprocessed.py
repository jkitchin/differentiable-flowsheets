"""Hydroprocessing products onto a gas-plant component table (#327).

A hydrotreater (or hydrocracker) leaves its liquids on its own *product
grid* -- :attr:`HydrotreaterResult.product_char`, a
:class:`~difflow_refinery.BlendCharacterization` of the liquid light ends
(C3-C5, as real species) and the TREATED cuts -- not on a characterization
:func:`~difflow_refinery.gasplant.gas_components` can take. This module puts
such a product on a :class:`~difflow_refinery.gasplant.GasComponents`
table, so a gas-plant column (a naphtha :func:`~difflow_refinery.gasplant.splitter`,
a stabilizer) can follow the unit:

.. code-block:: python

    from difflow_refinery.gasplant import GasPlantColumn, splitter
    from difflow_refinery.gasplant.hydroprocessed import hydroprocessed_feed

    feed = hydroprocessed_feed(nht_res, ("product", "wild_naphtha"), T=C + 120, P=4e5)
    col = GasPlantColumn(splitter(feed.components, light=feed.below(C + 85.0), ...))
    light, heavy, info = col(feed.stream)

The table, component by component:

* **Real gases** dissolved in the liquid but not on the product grid (H2,
  H2S, C1, C2 of a wild naphtha; the stripper bottoms carries none): real
  components with :func:`~difflow_refinery.gasplant.light_component_data`'s
  constants. Ammonia has none there (no ideal-gas Cp or LHV in the gas
  plant's tables); it is refused, or dropped with ``unsupported="drop"`` and
  its flow reported in :attr:`HydroprocessedFeed.dropped`.
* **Light ends on the grid** (propane ... n-pentane): real components, same
  constants -- except the molar mass, which is the product grid's (the
  atomic weights every hydroprocessing balance closes on; it differs from
  the database's in the fifth figure), so the feed's mass is the unit
  product's to round-off.
* **Treated cuts**: pseudocomponents with the grid's ``MW`` (treated: the
  hydrotreater moves it with the hydrogen it adds and the heteroatoms it
  removes), ``Tc``, ``Pc`` and ``omega``, and a Watson-Nelson ideal-gas
  Cp (:func:`~difflow_refinery.correlations.cp_ideal_gas_coeffs`) from the
  grid's ``Tb``, ``SG`` and ``MW`` -- the correlation
  :class:`~difflow_refinery.Characterization` uses for its own cuts. The
  hydrotreater carries ``Tb``, ``Tc``, ``Pc`` and ``omega`` over from the
  feed cut unchanged (saturating a ring moves them by less than the
  characterization's own uncertainty); only ``MW`` and ``SG`` are treated.

Relation to ``gas_components(light, pseudo=char)`` (and to the crude-unit
helper of #326, which builds the table from a crude-unit characterization):
this is the same table, built from a product grid instead. The two are not
mixed in one column -- a CDU naphtha and an HDT naphtha carry different
cuts under the same names -- which is why this is a separate constructor
rather than an option of either.

:func:`resolve_product` is the shared front end: it accepts a
``HydrotreaterResult`` (one outlet name or several, summed), a
``FractionationResult`` (one product name), anything with
``product_stream``/``product_char`` (a hydrocracker), or an explicit
``(stream, char)`` pair; :meth:`difflow_refinery.reforming.NaphthaFeed.from_hydrotreater`
reads its input the same way.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Literal, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.correlations import cp_ideal_gas_coeffs
from difflow_refinery.gasplant.components import (
    PSEUDO_LHV_J_PER_KG, GasComponents, default_kij, light_component_data)
from difflow_refinery.hydroprocessing.layout import gas_mw
from difflow_refinery.thermo import LIGHT_ENDS

#: Non-``F_`` keys of a stream that are carried over unchanged.
_STATE_KEYS = ("T", "P")


def _is_light(name: str) -> bool:
    """A product-grid component that is a real species (a light end), not a cut."""
    return name in LIGHT_ENDS


def resolve_product(source, product: str | Sequence[str] | None = None, char=None) -> tuple[dict, object]:
    """``(stream, grid)`` of a hydroprocessing product.

    Args:
        source: A ``HydrotreaterResult`` (or anything with ``product_stream``
            and ``product_char``, e.g. a hydrocracker result), a
            ``FractionationResult``, or an ``F_<name>`` stream (then
            ``char`` is required).
        product: For a unit result, the outlet name or names (summed;
            default ``"product"``, the stripper bottoms). For a
            ``FractionationResult``, the product name (required unless it
            has one product). Ignored for a stream.
        char: The grid (:class:`~difflow_refinery.BlendCharacterization`)
            of an explicit stream.

    Returns the stream (``F_`` flows in mol/s, with ``T`` and ``P`` where the
    source has them) and its :class:`~difflow_refinery.BlendCharacterization`.
    A unit outlet is taken WITH its dissolved real gases (``gases=True``),
    so nothing the unit made is silently lost.
    """
    if isinstance(source, Mapping):
        if char is None:
            raise ValueError("an explicit stream needs char= (its BlendCharacterization)")
        return dict(source), char
    if hasattr(source, "products") and hasattr(source, "off_gas") and hasattr(source, "char"):
        prods = source.products
        if product is None:
            if len(prods) != 1:
                raise ValueError(f"name the fractionator product: one of {sorted(prods)}")
            product = next(iter(prods))
        if not isinstance(product, str) or product not in prods:
            raise ValueError(f"{product!r} is not a product of the fractionation; have {sorted(prods)}")
        return dict(prods[product]), source.char
    if hasattr(source, "product_stream") and hasattr(source, "product_char"):
        names = ("product",) if product is None else ((product,) if isinstance(product, str) else tuple(product))
        takes_gases = "gases" in inspect.signature(source.product_stream).parameters
        out: dict = {}
        for n in names:
            s = source.product_stream(n, gases=True) if takes_gases else source.product_stream(n)
            for k, v in s.items():
                if k.startswith("F_"):
                    out[k] = out.get(k, 0.0) + v
                elif k in _STATE_KEYS and k not in out:
                    out[k] = v
        return out, source.product_char
    raise TypeError(f"cannot read a hydroprocessing product from {type(source).__name__}")


def stream_mass(stream: Mapping, char) -> Array:
    """kg/s of an ``F_`` stream: grid components at the grid's molar masses, real gases at theirs.

    The same rule as :meth:`HydrotreaterResult.stream_mass`.
    """
    mw = {n: char.MW[i] for i, n in enumerate(char.names)}
    m = jnp.asarray(0.0)
    for k, v in stream.items():
        if k.startswith("F_"):
            n = k[2:]
            m = m + jnp.asarray(v) * (mw[n] if n in mw else gas_mw(n))
    return m / 1000.0


def product_components(char, gases: Sequence[str] = (), kij: Optional[Mapping] = None,
                       cuts: Optional[Sequence[str]] = None) -> GasComponents:
    """A gas-plant component table over a hydroprocessing product grid.

    Args:
        char: The product grid (``HydrotreaterResult.product_char``,
            ``FractionationResult.char``), a
            :class:`~difflow_refinery.BlendCharacterization` whose names are
            light ends (real species) and cuts.
        gases: Real gases to put in front of the grid (``"hydrogen"``,
            ``"hydrogen_sulfide"``, ``"methane"``, ...), for the ones dissolved
            in the stream; any already on the grid are skipped.
        kij: PR ``kij`` overrides, ``{(a, b): value}``; others take
            :data:`~difflow_refinery.gasplant.PR_KIJ` or zero.
        cuts: Only these of the grid's cuts (default: all).

    Order: ``gases``, then the grid's light ends, then its cuts. Traceable in
    the grid's arrays (names are static).
    """
    names = list(char.names)
    light = [n for n in names if _is_light(n)]
    all_cuts = [n for n in names if not _is_light(n)]
    if cuts is not None:
        unknown = sorted(set(cuts) - set(all_cuts))
        if unknown:
            raise ValueError(f"{unknown} are not cuts of the product grid")
        all_cuts = [c for c in all_cuts if c in set(cuts)]
    real = [g for g in gases if g not in names] + light
    if len(set(real)) != len(real):
        raise ValueError(f"duplicate components in {real}")
    rows = [light_component_data(n) for n in real]
    for n, r in zip(real, rows):
        if r["name"] != n:
            raise ValueError(f"{n!r} is an alias of {r['name']!r}; use the database name")
    gi = {n: i for i, n in enumerate(names)}
    # molar masses: the grid's for its light ends, the atomic weights for added gases
    MW_r = jnp.stack([jnp.asarray(char.MW[gi[n]], dtype=float) if n in gi else jnp.asarray(gas_mw(n))
                      for n in real]) if real else jnp.zeros(0)
    col = lambda key: jnp.asarray([r[key] for r in rows], dtype=float).reshape(-1)  # noqa: E731
    ci = jnp.asarray([gi[c] for c in all_cuts], dtype=int)
    Tb, SG, MWc = (jnp.asarray(a, dtype=float)[ci] for a in (char.Tb, char.SG, char.MW))
    cp_c = cp_ideal_gas_coeffs(Tb, SG, MWc)
    all_names = tuple(real + all_cuts)
    K = default_kij(all_names)
    for (a, b), v in dict(kij or {}).items():
        i, j = all_names.index(a), all_names.index(b)
        K[i, j] = K[j, i] = float(v)
    return GasComponents(
        MW=jnp.concatenate([MW_r, MWc]),
        Tc=jnp.concatenate([col("Tc"), jnp.asarray(char.Tc, dtype=float)[ci]]),
        Pc=jnp.concatenate([col("Pc"), jnp.asarray(char.Pc, dtype=float)[ci]]),
        omega=jnp.concatenate([col("omega"), jnp.asarray(char.omega, dtype=float)[ci]]),
        cp=jnp.concatenate([jnp.asarray([r["cp"] for r in rows], dtype=float).reshape(-1, 4), cp_c]),
        kij=jnp.asarray(K),
        lhv=jnp.concatenate([col("lhv"), PSEUDO_LHV_J_PER_KG * MWc / 1000.0]),
        pseudo=jnp.concatenate([jnp.zeros(len(real)), jnp.ones(len(all_cuts))]),
        names=all_names)


@dataclass(frozen=True)
class HydroprocessedFeed:
    """A hydroprocessing product on a gas-plant component table.

    Attributes:
        components: The :class:`~difflow_refinery.gasplant.GasComponents`.
        stream: ``F_<components.names>`` (mol/s) with ``T`` and ``P`` -- a
            feed for a :class:`~difflow_refinery.gasplant.GasPlantColumn`.
        mass_flow: kg/s of ``stream`` on the table's molar masses: the
            product's mass (:func:`stream_mass`) less what was dropped.
        dropped: ``{gas: mol/s}`` left out (``unsupported="drop"``).
        char: The product grid it came from.
    """

    components: GasComponents
    stream: dict
    mass_flow: Array
    dropped: dict
    char: object

    def below(self, T_cut: float) -> tuple[str, ...]:
        """Names of the components boiling below ``T_cut`` (K): real ones by
        their normal boiling point, cuts by the grid's ``Tb``. A ``light=``
        for :func:`~difflow_refinery.gasplant.splitter` (concrete)."""
        Tb = dict(zip(self.char.names, np.asarray(self.char.Tb)))
        out = []
        for n in self.components.names:
            t = Tb.get(n, LIGHT_ENDS[n][1] if n in LIGHT_ENDS else -np.inf)
            if t < T_cut:
                out.append(n)
        return tuple(out)


def hydroprocessed_feed(source, product: str | Sequence[str] | None = None, *, char=None,
                        T=None, P=None, kij: Optional[Mapping] = None,
                        unsupported: Literal["raise", "drop"] = "raise",
                        min_flow: float = 0.0) -> HydroprocessedFeed:
    """A hydroprocessing product as a gas-plant feed: its table and its stream.

    Args:
        source, product, char: The product (see :func:`resolve_product`):
            ``hydroprocessed_feed(res)`` is the stripper bottoms,
            ``hydroprocessed_feed(res, ("product", "wild_naphtha"))`` the
            whole treated naphtha, ``hydroprocessed_feed(fr, "heavy_naphtha")``
            a fractionator product.
        T, P: Feed temperature (K) and pressure (Pa); default the stream's.
        kij: PR ``kij`` overrides.
        unsupported: What to do with a dissolved gas the gas plant has no
            constants for (ammonia): ``"raise"`` (default) or ``"drop"``
            (its flow goes to :attr:`HydroprocessedFeed.dropped`).
        min_flow: Leave out grid cuts whose flow is at most this (mol/s);
            every cut is kept by default. A fractionator product carries a
            sigmoid tail of every cut; this trims the table, not the mass
            (a trimmed cut's flow goes to ``dropped`` too).

    Returns a :class:`HydroprocessedFeed`. The stream's mass is the product's
    to round-off when nothing is dropped. Traceable in the flows (with a
    concrete ``min_flow`` selection).
    """
    stream, grid = resolve_product(source, product, char)
    names = list(grid.names)
    extra = [k[2:] for k in stream if k.startswith("F_") and k[2:] not in names]
    dropped: dict = {}
    gases = []
    for g in extra:
        try:
            light_component_data(g)
        except KeyError:
            if unsupported == "drop":
                dropped[g] = stream[f"F_{g}"]
                continue
            raise ValueError(
                f"the gas plant has no constants for {g!r} (in the stream at "
                f"{float(np.asarray(stream[f'F_{g}'])):.3g} mol/s); pass unsupported='drop'") from None
        gases.append(g)
    cuts = None
    if min_flow > 0.0:
        cuts = []
        for n in names:
            if _is_light(n):
                continue
            f = stream.get(f"F_{n}", 0.0)
            if float(np.asarray(f)) > min_flow:
                cuts.append(n)
            else:
                dropped[n] = f
    comps = product_components(grid, gases=gases, kij=kij, cuts=cuts)
    z = jnp.asarray(0.0)
    out = {f"F_{n}": jnp.asarray(stream.get(f"F_{n}", z), dtype=float) for n in comps.names}
    out["T"] = jnp.asarray(stream.get("T", 298.15) if T is None else T, dtype=float)
    out["P"] = jnp.asarray(stream.get("P", 101325.0) if P is None else P, dtype=float)
    flows = jnp.stack([out[f"F_{n}"] for n in comps.names])
    return HydroprocessedFeed(components=comps, stream=out, mass_flow=flows @ comps.MW / 1000.0,
                              dropped=dropped, char=grid)


__all__ = ["resolve_product", "stream_mass", "product_components", "HydroprocessedFeed",
           "hydroprocessed_feed"]
