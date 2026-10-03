"""From a characterized residue stream to a residue-desulfurizer feed on the attribute layout.

:func:`rds_feed` is :func:`difflow_refinery.hydrotreating.feed.hdt_feed`
(carbon, hydrogen, the five sulfur classes, the nitrogen classes, the ring
classes, olefins and naphthenes, all from ``char.composition``) plus the
three residue attributes of :mod:`.kinetics`:

* ``S_residue`` -- a share :func:`residue_s_share` of each cut's sulfur,
  taken from its five classes in proportion (so the total sulfur and the
  class split of the rest are unchanged). The share rises with boiling
  point: zero through the vacuum gas oil, a third of the sulfur of the
  vacuum residue, half of it in the heaviest lump --
  :data:`DEFAULT_RESIDUE_S_SHARE`, ILLUSTRATIVE (the shape the
  residue-desulfurization literature describes: the asphaltene and resin
  sulfur concentrates in the heaviest fraction; numbers chosen here). Pass
  ``residue_s_share=`` with measured ones (e.g. the sulfur of the C7
  asphaltenes from an ASTM D6560 / IP 143 separation).
* ``NiV`` -- the characterization's ``nickel_vanadium`` mass fraction per
  cut (#301), as moles of a nominal metal of :data:`.kinetics.METAL_MW`.
* ``CCR`` -- its ``ccr`` mass fraction, as moles of carbon.

Metals and CCR carbon are not part of a cut's mass in the layout (see
:mod:`.kinetics`), so the feed's mass from its atoms is still ``F * MW``.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax.numpy as jnp
from jax import Array

from difflow_refinery.composition import SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, cut_indices, hdt_feed, hdt_layout
from difflow_refinery.residue.kinetics import CCR_MW, METAL_MW, RDS_ATTRIBUTE_ELEMENTS, RDS_ATTRIBUTES

#: Share of a cut's sulfur that is refractory residue sulfur, by normal
#: boiling point ``(T_K, share)``; linear in between, flat beyond.
#: ILLUSTRATIVE (see module docstring).
DEFAULT_RESIDUE_S_SHARE: tuple = (
    (723.15, 811.15, 873.15, 973.15, 1073.15),
    (0.0, 0.10, 0.25, 0.35, 0.50),
)


def residue_s_share(Tb: Array, table=DEFAULT_RESIDUE_S_SHARE) -> Array:
    """``(n,)`` refractory share of each cut's sulfur at boiling point ``Tb`` (K)."""
    Ts, v = table
    return jnp.interp(jnp.asarray(Tb, dtype=float), jnp.asarray(Ts, dtype=float), jnp.asarray(v, dtype=float))


def rds_layout(char, cuts: Sequence[str]) -> Layout:
    """The residue layout: the hydrotreater's gases (incl. light ends and water) and :data:`RDS_ATTRIBUTES`."""
    base = hdt_layout(char, cuts)
    return Layout(gases=base.gases, cuts=base.cuts, attributes=RDS_ATTRIBUTES,
                  attribute_elements=RDS_ATTRIBUTE_ELEMENTS)


def _hdt_view(layout: Layout) -> Layout:
    n = len(RDS_ATTRIBUTES) - 3
    return Layout(gases=layout.gases, cuts=layout.cuts, attributes=layout.attributes[:n],
                  attribute_elements=layout.attribute_elements[:n])


def rds_feed(char, stream: Mapping, layout: Layout, residue_s_share_table=DEFAULT_RESIDUE_S_SHARE,
             aromatic_table=DEFAULT_AROMATIC_SPLIT) -> Flows:
    """The residue ``stream`` (``F_<char.names>``, mol/s) as attribute flows on ``layout``.

    Cuts of the characterization the layout does not carry must have no flow
    in ``stream``. Traceable in the stream and in ``char``.
    """
    base = hdt_feed(char, stream, _hdt_view(layout), aromatic_table)
    idx = cut_indices(char, layout.cuts)
    k = len(char.light_names)
    Tb = jnp.concatenate([jnp.zeros(k), char.Tb])[idx]
    share = residue_s_share(Tb, residue_s_share_table)
    ai = {a: i for i, a in enumerate(layout.attributes)}
    s_cols = jnp.asarray([ai[f"S_{c}"] for c in SULFUR_CLASSES])
    attr = jnp.concatenate([base.attr, jnp.zeros((layout.n_cut, 3))], axis=1)
    S_cls = attr[:, s_cols]
    attr = attr.at[:, s_cols].set(S_cls * (1.0 - share)[:, None])
    attr = attr.at[:, ai["S_residue"]].set(jnp.sum(S_cls, axis=1) * share)
    m = base.cut * jnp.asarray(char.component_MW)[idx]                     # g/s
    for key, field, mw in (("NiV", "nickel_vanadium", METAL_MW), ("CCR", "ccr", CCR_MW)):
        frac = getattr(char, field, None)
        if frac is not None:
            attr = attr.at[:, ai[key]].set(m * jnp.asarray(frac)[idx] / mw)
    return Flows(gas=base.gas, cut=base.cut, attr=attr)


__all__ = ["DEFAULT_RESIDUE_S_SHARE", "residue_s_share", "rds_layout", "rds_feed"]
