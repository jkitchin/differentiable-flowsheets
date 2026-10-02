"""Crude-unit product streams onto a gas-plant component table (#326).

A crude unit's offgas and unstabilised naphtha are streams on the crude's
whole characterization (``F_<light end>``, ``F_pc01`` ... ``F_pcNN``,
``F_water``). The gas plant works on a :class:`GasComponents` table that
holds only the light ends and the cuts the naphtha actually carries.
:func:`gas_plant_feed` is the bridge, and it does the three things every
chain from the CDU used to do by hand (examples 38 and 40):

1. **Select the cuts.** Keep a cut when its molar flow is more than
   ``min_fraction`` of the *basis* stream's total molar flow as the crude
   unit reports it (water included), in the basis stream (the naphtha by
   default). The heavy cuts would add a column to every EOS call to carry
   about 1e-4 of the feed. The selection is a **static** choice: it is
   made on the concrete flows (``jax.lax.stop_gradient``), or passed as
   ``cuts=`` -- which is what to do under ``jax.jit``.
2. **Fold the rest.** The molar flow of every cut not kept goes into the
   heaviest cut kept (the last kept one in the characterization's order),
   in every stream. Moles are conserved exactly; mass is not, since the
   folded cuts are heavier: the change is reported, per stream, as
   ``fold_mass_change`` (it is ``sum_folded F_i (MW_dest - MW_i)``);
   ``mass_change`` adds the (small) difference between the database molar
   masses of the light ends and the characterization's.
3. **Drop the water.** The overhead accumulator decants it; the crude
   unit's ``F_water`` (and anything else in ``drop=``) is removed and its
   mass reported.

The result is differentiable with respect to the stream flows (and the
characterization's arrays, through :func:`gas_components`): the folding
is a sum, the selection is fixed.

H2S
    The crude unit makes no H2S: the assay's sulfur stays on the cuts.
    Two ways to put some in the offgas, both explicit:

    * ``h2s={"offgas": 0.02}`` -- an **assumption**, in mol of H2S added
      per mol of the stream's water-free flow (examples 38 and 40 use
      0.02, "2 mol %"; the H2S is then 0.02/1.02 of the stream);
    * ``h2s_flow={"offgas": evolved_h2s(products, char, fraction)}`` --
      a flow in mol/s, here from a **sulfur balance**: a ``fraction`` of
      the sulfur the crude unit's products carry, leaving as H2S.

    :func:`evolved_h2s` is a basis, not a model. How much of a crude's
    sulfur evolves as H2S in the furnace and column (dissolved H2S, thermal
    decomposition of thiols and sulfides) is crude- and severity-specific;
    no value of ``fraction`` is sourced here, so it has no default and any
    number passed is the caller's assumption (illustrative). The cuts'
    sulfur is not reduced by it: the H2S sulfur is counted twice, once as
    gas and once on the cuts, and ``evolved_h2s`` returns it so that a
    sulfur balance can subtract it.

Example::

    feed = gas_plant_feed(cdu.products, char,
                          ["hydrogen_sulfide", "ethane", "propane", ...],
                          h2s={"offgas": 0.02}, T=313.15, P=1.3e5)
    comps, offgas, naphtha = feed.components, feed["offgas"], feed["naphtha"]
    print(feed.summary())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np

from difflow_refinery.gasplant.components import GasComponents, gas_components

#: Molar mass of sulfur (g/mol), as the hydroprocessing layout uses it.
MW_S = 32.065
#: Molar mass of H2S (g/mol), H2S = 2 x 1.00794 + 32.065.
MW_H2S = 34.08088
#: Molar mass of water (g/mol), the crude unit's ``WATER_MW``.
MW_WATER = 18.015

_Num = Union[float, jax.Array]


@dataclass(frozen=True)
class GasPlantFeed:
    """Crude-unit streams on a gas-plant component table.

    Attributes:
        components: The :class:`GasComponents` (``light`` then ``cuts``).
        streams: ``{name: stream}``, each ``{"F_<component>": ..., "T", "P"}``
            on ``components.names``, in the order the streams were given.
        cuts: The cuts kept, in the characterization's order.
        folded_cuts: The cuts folded away.
        fold_into: The cut that received them (``cuts[-1]``).
        basis: The stream the cuts were selected on.
        min_fraction: The selection threshold.
        folded_mol: ``{stream: mol/s}`` of folded cuts.
        folded_kg: ``{stream: kg/s}`` of folded cuts, at their own molar mass.
        mass_change: ``{stream: kg/s}`` by which the stream's mass on the
            gas-plant table differs from its water-free mass on the
            characterization (H2S added left out): the fold (the folded
            moles at the receiving cut's molar mass, less ``folded_kg``;
            negative, since the folded cuts are the heavier ones) plus any
            difference between a light end's database molar mass and the
            characterization's.
        fold_mass_change: ``{stream: kg/s}``, the fold's part of
            ``mass_change`` alone.
        dropped_mol, dropped_kg: ``{stream: {component: value}}`` removed
            (water, and anything else in ``drop=``).
        h2s_mol: ``{stream: mol/s}`` of H2S added.
        basis_total: Total molar flow of the basis stream as given (water
            included), the denominator of the selection.
    """

    components: GasComponents
    streams: dict
    cuts: tuple
    folded_cuts: tuple
    fold_into: str
    basis: str
    min_fraction: float
    folded_mol: dict
    folded_kg: dict
    mass_change: dict
    fold_mass_change: dict
    dropped_mol: dict
    dropped_kg: dict
    h2s_mol: dict
    basis_total: _Num

    def __getitem__(self, name: str) -> dict:
        return self.streams[name]

    @property
    def folded_fraction(self) -> _Num:
        """Folded moles of every stream, relative to the basis stream's total.

        The number examples 38 and 40 printed ("folded into pcNN: 1e-4 of the
        naphtha"). Per stream, :meth:`folded_fraction_of` is the fraction of
        that stream.
        """
        return sum(self.folded_mol.values()) / self.basis_total

    def folded_fraction_of(self, name: str) -> _Num:
        """Folded moles of stream ``name`` over its own water-free total."""
        return self.folded_mol[name] / jnp.sum(self.flows(self.streams[name]))

    def flows(self, stream: Mapping) -> jax.Array:
        """Molar flows of ``stream`` in ``components.names`` order (mol/s)."""
        return jnp.stack([jnp.asarray(stream[f"F_{n}"]) for n in self.components.names])

    def mass(self, stream: Mapping) -> jax.Array:
        """Mass flow of ``stream`` (kg/s)."""
        return jnp.sum(self.flows(stream) * self.components.MW) / 1000.0

    def summary(self) -> str:
        """Kept and folded cuts, and the mass folded, dropped and added."""
        lines = [f"components: {', '.join(self.components.names)}",
                 f"cuts kept ({self.basis} > {self.min_fraction:g}): "
                 f"{self.cuts[0]}..{self.cuts[-1]}; folded into {self.fold_into}: "
                 f"{', '.join(self.folded_cuts) or 'none'}",
                 f"folded: {float(self.folded_fraction):.1e} of the {self.basis} (molar)"]
        for n in self.streams:
            drop = sum(float(v) for v in self.dropped_kg[n].values())
            lines.append(f"  {n}: folded {float(self.folded_mol[n]):.3e} mol/s "
                         f"({float(self.folded_kg[n]):.3e} kg/s); mass change "
                         f"{float(self.mass_change[n]):+.2e} kg/s; dropped {drop:.4g} kg/s"
                         + (f"; H2S added {float(self.h2s_mol[n]):.4g} mol/s"
                            if n in self.h2s_mol else ""))
        return "\n".join(lines)


def _concrete(x) -> float:
    try:
        return float(np.asarray(jax.lax.stop_gradient(x)))
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        raise TypeError("gas_plant_feed selects its cuts on concrete flows; under "
                        "jax.jit pass cuts= (the selection is a static choice)") from None


def _component_mw(char) -> dict:
    return dict(zip(char.names, np.asarray(char.component_MW, dtype=float)))


def evolved_h2s(products: Mapping[str, Mapping], char, fraction: _Num,
                streams: Optional[Sequence[str]] = None) -> jax.Array:
    """H2S (mol/s) from a ``fraction`` of the sulfur the crude unit's products carry.

    ``sum_streams sum_i F_i MW_i S_i * fraction / MW_S`` with ``S_i`` the
    characterization's sulfur mass fraction (``characterize`` of an assay
    with ``sulfur_wt``). A sulfur balance, not a model of H2S evolution;
    ``fraction`` has no default because no value is sourced here (see the
    module docstring) -- whatever is passed is an assumption. Differentiable
    in the flows, the sulfur and ``fraction``.

    Args:
        products: The crude unit's product streams.
        char: The characterization they are on; must carry ``sulfur``.
        fraction: Fraction of that sulfur leaving as H2S.
        streams: Which products to count (default: all of them).
    """
    sulfur = getattr(char, "sulfur", None)
    try:
        no_sulfur = sulfur is None or not np.any(np.asarray(sulfur))
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        no_sulfur = False                                  # traced: trust the caller
    if no_sulfur:
        raise ValueError("the characterization carries no sulfur: characterize an Assay "
                         "given sulfur_wt=..., or pass h2s= as an explicit assumption")
    mw = jnp.asarray(char.component_MW, dtype=float)
    S = jnp.asarray(char.sulfur, dtype=float)
    total = 0.0
    for name in (products if streams is None else streams):
        s = products[name]
        F = jnp.stack([jnp.asarray(s.get(f"F_{n}", 0.0), dtype=float) for n in char.names])
        total = total + jnp.sum(F * mw * S)               # g/s of sulfur
    return fraction * total / MW_S


def gas_plant_feed(products: Mapping[str, Mapping], char, light: Sequence[str], *,
                   streams: Sequence[str] = ("offgas", "naphtha"),
                   basis: str = "naphtha",
                   min_fraction: float = 1e-3,
                   cuts: Optional[Sequence[str]] = None,
                   drop: Sequence[str] = ("water",),
                   h2s: Optional[Mapping[str, _Num]] = None,
                   h2s_flow: Optional[Mapping[str, _Num]] = None,
                   T: Union[None, _Num, Mapping[str, _Num]] = None,
                   P: Union[None, _Num, Mapping[str, _Num]] = None,
                   kij: Optional[Mapping] = None) -> GasPlantFeed:
    """Put crude-unit product streams on a gas-plant component table.

    See the module docstring for the rules (select, fold, drop) and the H2S
    options.

    Args:
        products: ``{name: stream}`` from the crude unit
            (``CrudeUnit(...).solve(...).products``); values may be traced.
        char: The :class:`~difflow_refinery.Characterization` the streams
            are on (the crude unit's own, or one on the same cut points).
        light: Database names of the real components, in table order. Must
            cover every light end the streams carry; ``"hydrogen_sulfide"``
            is put first if H2S is added and it is not listed.
        streams: Which products to convert.
        basis: The stream the cuts are selected on.
        min_fraction: Keep a cut whose molar flow in ``basis`` exceeds this
            fraction of ``basis``'s total (water included).
        cuts: The cuts to keep, overriding the selection (needed under
            ``jax.jit``, where the flows are not concrete).
        drop: Components removed from every stream (the decanted water).
        h2s: ``{stream: mol H2S per mol of its water-free flow}``, an
            explicit assumption.
        h2s_flow: ``{stream: mol/s}`` of H2S, e.g. from :func:`evolved_h2s`.
            Added to any ``h2s``.
        T, P: Stream temperature (K) and pressure (Pa): a value for all, a
            ``{stream: value}`` mapping, or ``None`` to keep the stream's own.
        kij: Passed to :func:`gas_components`.

    Returns:
        A :class:`GasPlantFeed`.
    """
    streams = tuple(streams)
    for n in streams + (basis,):
        if n not in products:
            raise KeyError(f"no product {n!r}; have {sorted(products)}")
    pseudo = tuple(char.pseudo_names)
    h2s, h2s_flow = dict(h2s or {}), dict(h2s_flow or {})
    light = list(light)
    if (h2s or h2s_flow) and "hydrogen_sulfide" not in light:
        light = ["hydrogen_sulfide"] + light

    base = products[basis]
    total = sum(v for k, v in base.items() if k.startswith("F_"))
    if cuts is None:
        tot = _concrete(total)
        cuts = [n for n in pseudo if _concrete(base.get(f"F_{n}", 0.0)) / tot > min_fraction]
    else:
        unknown = sorted(set(cuts) - set(pseudo))
        if unknown:
            raise ValueError(f"{unknown} are not cuts of the characterization")
        cuts = [n for n in pseudo if n in set(cuts)]
    if not cuts:
        raise ValueError(f"no cut of {basis!r} is above min_fraction={min_fraction:g}")
    folded = tuple(n for n in pseudo if n not in cuts)
    dest = cuts[-1]
    comps = gas_components(light, pseudo=char, cuts=cuts, kij=kij)
    names = set(comps.names)
    mw = _component_mw(char)
    mw_dest = mw[dest]
    drop = tuple(drop)

    def pick(v, n):
        return v.get(n) if isinstance(v, Mapping) else v

    table_mw = dict(zip(comps.names, list(comps.MW)))
    out, f_mol, f_kg, dm, fdm, d_mol, d_kg, added = {}, {}, {}, {}, {}, {}, {}, {}
    for n in streams:
        src = products[n]
        flows = {k[2:]: v for k, v in src.items() if k.startswith("F_")}
        # water-free mass as given, on the characterization's molar masses
        m_src = sum(v * (mw[k] if k in mw else table_mw.get(k, np.nan)) / 1000.0
                    for k, v in flows.items() if k not in drop)
        if n in h2s or n in h2s_flow:
            extra = 0.0
            if n in h2s:
                extra = extra + h2s[n] * sum(v for k, v in flows.items() if k not in drop)
            if n in h2s_flow:
                extra = extra + h2s_flow[n]
            flows["hydrogen_sulfide"] = flows.get("hydrogen_sulfide", 0.0) + extra
            added[n] = extra
        s = {f"F_{c}": 0.0 for c in comps.names}
        fm, fk = 0.0, 0.0
        d_mol[n], d_kg[n] = {}, {}
        for k, v in flows.items():
            if k in drop:
                d_mol[n][k] = v
                d_kg[n][k] = v * (MW_WATER if k == "water" else mw.get(k, np.nan)) / 1000.0
                continue
            if k in folded:
                s[f"F_{dest}"] += v
                fm = fm + v
                fk = fk + v * mw[k] / 1000.0
                continue
            if k not in names:
                raise ValueError(f"stream {n!r} carries {k!r}, which is neither in light= "
                                 f"nor a cut; add it to light or to drop=")
            s[f"F_{k}"] += v
        Tn, Pn = pick(T, n), pick(P, n)
        s["T"] = src["T"] if Tn is None else Tn
        s["P"] = src["P"] if Pn is None else Pn
        out[n] = s
        f_mol[n], f_kg[n] = fm, fk
        fdm[n] = fm * mw_dest / 1000.0 - fk
        m_added = added[n] * table_mw["hydrogen_sulfide"] / 1000.0 if n in added else 0.0
        dm[n] = jnp.sum(jnp.stack([jnp.asarray(s[f"F_{c}"]) for c in comps.names]) * comps.MW) \
            / 1000.0 - m_src - m_added
    return GasPlantFeed(components=comps, streams=out, cuts=tuple(cuts),
                        folded_cuts=folded, fold_into=dest, basis=basis,
                        min_fraction=float(min_fraction), folded_mol=f_mol,
                        folded_kg=f_kg, mass_change=dm, fold_mass_change=fdm, dropped_mol=d_mol,
                        dropped_kg=d_kg, h2s_mol=added, basis_total=total)
