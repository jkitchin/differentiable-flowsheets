"""From a characterized product stream to a hydrotreater feed on the attribute layout.

A crude-unit product is a difflow stream of ``F_<name>`` flows over
``char.names``. :func:`hdt_feed` turns it into :class:`~difflow_refinery.hydroprocessing.layout.Flows`
on an :data:`~difflow_refinery.hydrotreating.kinetics.HDT_ATTRIBUTES` layout,
reading every cut's composition from ``char.composition`` (#305): its
hydrogen, sulfur and nitrogen mass fractions and their class splits become
atom flows, its hydrocarbon types molecule counts.

* Carbon is ``1 - H - S - N`` by mass (oxygen and metals neglected), exactly
  the composition module's definition, so the cut's mass from its atoms
  equals ``F * MW``.
* **Types as molecules.** A cut's volume fractions of paraffins, naphthenes,
  aromatics and olefins are taken as its *mole* fractions. Within one narrow
  cut the molar volumes of the types differ by 10-20 %, so this is an
  approximation; product aromatics are reported back through the same rule,
  so an untreated cut reports exactly the volume fractions it came in with.
* **Aromatic rings.** The composition module carries total aromatics; the
  kinetics needs mono, di and poly. :data:`DEFAULT_AROMATIC_SPLIT` is an
  ILLUSTRATIVE split by boiling point (all mono in naphtha; about 60/30/10 in
  diesel; more polyaromatic in the vacuum range), from no particular crude
  and no source -- the shape the hydrotreating literature describes, numbers
  chosen here. Pass ``aromatic_split=`` with measured ones (e.g. from IP 391
  or EN 12916).

:func:`straight_run_cut` makes an idealized straight-run feed (every cut whose
boiling point lies in a TBP range, in its crude proportion), for running a
hydrotreater without a crude unit -- the analogue of
:func:`difflow_refinery.vacuum.atmospheric_residue`.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.composition import HC_TYPES, NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import ATOMIC_MASS, DEFAULT_GASES, Flows, Layout
from difflow_refinery.hydrotreating.kinetics import (
    AROMATIC_CLASSES, CRACK_GAS_SPLIT, HDT_ATTRIBUTE_ELEMENTS, HDT_ATTRIBUTES)

#: Illustrative aromatic-ring split by boiling point: ``(T_K, shares)``, each
#: row the fraction of a cut's aromatic molecules that are mono-, di- and
#: poly-aromatic; linear in between, flat beyond. ILLUSTRATIVE (see module
#: docstring).
DEFAULT_AROMATIC_SPLIT: tuple = (
    (423.15, 473.15, 523.15, 573.15, 623.15, 723.15),
    ((1.00, 0.00, 0.00), (0.90, 0.10, 0.00), (0.75, 0.22, 0.03),
     (0.62, 0.28, 0.10), (0.52, 0.30, 0.18), (0.40, 0.30, 0.30)),
)


def aromatic_split(Tb: Array, table=DEFAULT_AROMATIC_SPLIT) -> Array:
    """``(n, 3)`` mono/di/poly shares of each cut's aromatics at boiling point ``Tb``."""
    Ts, rows = table
    Ts = jnp.asarray(Ts, dtype=float)
    rows = jnp.asarray(rows, dtype=float)
    out = jnp.stack([jnp.interp(jnp.asarray(Tb, dtype=float), Ts, rows[:, k]) for k in range(3)], axis=-1)
    return out / jnp.sum(out, axis=-1, keepdims=True)


def hdt_layout(char, cuts: Sequence[str], extra_gases: Sequence[str] = ()) -> Layout:
    """The hydrotreater layout: default gases, the char's light ends, water, and ``cuts``."""
    gases = list(DEFAULT_GASES)
    for g in tuple(char.light_names) + tuple(extra_gases) + ("water",):
        if g not in gases:
            gases.append(g)
    for g in CRACK_GAS_SPLIT:
        if g not in gases:
            gases.append(g)
    return Layout(gases=tuple(gases), cuts=tuple(cuts), attributes=HDT_ATTRIBUTES,
                  attribute_elements=HDT_ATTRIBUTE_ELEMENTS)


def cut_indices(char, cuts: Sequence[str]) -> np.ndarray:
    """Indices of ``cuts`` among ``char.names``."""
    names = list(char.names)
    return np.asarray([names.index(c) for c in cuts])


def hdt_feed(char, stream: Mapping, layout: Layout, aromatic_table=DEFAULT_AROMATIC_SPLIT) -> Flows:
    """The feed ``stream`` (``F_<char.names>``, mol/s) as attribute flows on ``layout``.

    Cuts of the characterization the layout does not carry must have no
    flow in ``stream`` (they would be lost); gases not in the layout
    likewise. Traceable in the stream and in ``char``.
    """
    comp = char.composition
    if comp is None:
        raise ValueError("the characterization has no composition: characterize(assay, composition=True)")
    idx = cut_indices(char, layout.cuts)
    F = jnp.stack([jnp.asarray(stream.get(f"F_{c}", 0.0), dtype=float) for c in layout.cuts])
    MW = jnp.asarray(char.component_MW)[idx]
    m = F * MW                                          # g/s
    H = comp.hydrogen[idx]
    S = comp.sulfur[idx]
    N = comp.nitrogen[idx]
    C = 1.0 - H - S - N
    cols = {"C": m * C / ATOMIC_MASS["C"], "H": m * H / ATOMIC_MASS["H"]}
    for k, c in enumerate(SULFUR_CLASSES):
        cols[f"S_{c}"] = m * S * comp.sulfur_split[idx, k] / ATOMIC_MASS["S"]
    for k, c in enumerate(NITROGEN_CLASSES):
        cols[f"N_{c}"] = m * N * comp.nitrogen_split[idx, k] / ATOMIC_MASS["N"]
    types = comp.hc_type[idx]                           # (nc, 4) P N A O
    arom = F * types[:, HC_TYPES.index("aromatics")]
    Tb = jnp.concatenate([jnp.zeros(len(char.light_names)), char.Tb])[idx]
    sp = aromatic_split(Tb, aromatic_table)
    for k, c in enumerate(AROMATIC_CLASSES):
        cols[f"A_{c}"] = arom * sp[:, k]
    cols["olefins"] = F * types[:, HC_TYPES.index("olefins")]
    cols["naphthenes"] = F * types[:, HC_TYPES.index("naphthenes")]
    attr = jnp.stack([cols[a] for a in layout.attributes], axis=-1)
    gas = jnp.stack([jnp.asarray(stream.get(f"F_{g}", 0.0), dtype=float) for g in layout.gases])
    return Flows(gas=gas, cut=F, attr=attr)


def straight_run_cut(char, T_lo: float | None = None, T_hi: float | None = None, rate: Array | float = 1.0,
                     basis: str = "mass", cuts: Sequence[str] | None = None) -> dict:
    """An idealized straight-run cut: every cut with ``T_lo <= Tb < T_hi`` in its crude proportion.

    Args:
        char: The characterization.
        T_lo, T_hi: TBP range (K) the cut's pseudo-components' boiling points lie in
            (concrete: it selects the cuts).
        rate: Total flow: kg/s (``basis="mass"``) or mol/s (``"mole"``).
        cuts: The pseudo-component names instead of a TBP range -- needed when
            ``char`` is traced (the range selection reads concrete boiling
            points). :func:`select_cuts` gives them from a concrete one.

    Returns a difflow stream of ``F_<name>`` (only the selected cuts), 298 K, 1 atm.
    """
    if cuts is None:
        cuts = select_cuts(char, T_lo, T_hi)
    sel = [list(char.pseudo_names).index(c) for c in cuts]
    k = len(char.light_names)
    w = jnp.asarray(char.mass_fraction)[jnp.asarray([k + i for i in sel])]
    MW = jnp.asarray(char.MW)[jnp.asarray(sel)]
    if basis == "mass":
        mol = jnp.asarray(rate, dtype=float) * 1000.0 * (w / jnp.sum(w)) / MW
    elif basis == "mole":
        x = w / MW
        mol = jnp.asarray(rate, dtype=float) * x / jnp.sum(x)
    else:
        raise ValueError("basis must be 'mass' or 'mole'")
    out = {f"F_{char.pseudo_names[i]}": mol[j] for j, i in enumerate(sel)}
    out["T"] = jnp.asarray(298.15)
    out["P"] = jnp.asarray(101325.0)
    return out


def select_cuts(char, T_lo: float, T_hi: float) -> tuple[str, ...]:
    """Names of the pseudo-components with ``T_lo <= Tb < T_hi`` (concrete ``char``)."""
    Tb = np.asarray(char.Tb)
    sel = [char.pseudo_names[i] for i, t in enumerate(Tb) if T_lo <= t < T_hi]
    if not sel:
        raise ValueError(f"no pseudo-component boils in [{T_lo}, {T_hi}) K")
    return tuple(sel)


__all__ = ["select_cuts", "DEFAULT_AROMATIC_SPLIT", "aromatic_split", "hdt_layout", "cut_indices", "hdt_feed",
           "straight_run_cut"]
