"""The hydrocracker's stream layout, its feed, and the pretreat bed's VGO parameter set.

The layout is the hydrotreater's (:func:`~difflow_refinery.hydrotreating.feed.hdt_layout`)
with two changes: the per-cut attributes gain ``"cracked"`` (see
:mod:`.kinetics`), and the gases always include isopentane and n-pentane,
which cracking makes. A feed is read exactly as the hydrotreater reads one
(:func:`~difflow_refinery.hydrotreating.feed.hdt_feed`: atoms, heteroatom
classes and type counts from ``char.composition``), with zero cracked
molecules.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import jax.numpy as jnp

from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydrocracking.kinetics import HCU_ATTRIBUTE_ELEMENTS, HCU_ATTRIBUTES, HCU_GAS_SPLIT
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, hdt_feed, hdt_layout
from difflow_refinery.hydrotreating.kinetics import HDTKineticParams

#: Pretreat-bed (hydrotreating) constants for a VGO on a NiMo pretreat
#: catalyst. ILLUSTRATIVE, like the hydrotreater's defaults they modify:
#: HDN is made several times faster (a NiMo catalyst at 150 bar is chosen for
#: HDN, which is what protects the cracking catalyst), so that a VGO of about
#: 1000-2000 wppm N leaves the pretreat bed at 385 C, LHSV ~1.5 1/h with a few
#: tens of wppm organic N; the cracking leak is kept small.
VGO_PRETREAT_PARAMS = HDTKineticParams(
    T_ref=653.15,
    hds_k=(4.0e-5, 2.0e-5, 1.2e-5, 1.0e-5, 6.0e-6),
    hdn_k=(3.0e-6, 4.0e-6),
    hdn_E=(110e3, 110e3),
    hda_k=(4.0e-6, 1.5e-6, 3.0e-7),
    crack_k=1.0e-10,
)


def hcu_layout(char, cuts: Sequence[str]) -> Layout:
    """The hydrocracker layout on ``cuts``: the hydrotreater's gases plus C5s, :data:`HCU_ATTRIBUTES`."""
    extra = tuple(g for g in HCU_GAS_SPLIT if g not in ("methane", "ethane", "propane", "isobutane", "n_butane"))
    base = hdt_layout(char, cuts, extra_gases=extra)
    return Layout(gases=base.gases, cuts=base.cuts, attributes=HCU_ATTRIBUTES,
                  attribute_elements=HCU_ATTRIBUTE_ELEMENTS)


def _hdt_view(layout: Layout) -> Layout:
    from difflow_refinery.hydrotreating.kinetics import HDT_ATTRIBUTE_ELEMENTS, HDT_ATTRIBUTES
    return Layout(gases=layout.gases, cuts=layout.cuts, attributes=HDT_ATTRIBUTES,
                  attribute_elements=HDT_ATTRIBUTE_ELEMENTS)


def hcu_feed(char, stream: Mapping, layout: Layout, aromatic_table=DEFAULT_AROMATIC_SPLIT) -> Flows:
    """The feed ``stream`` (``F_<char.names>``) as attribute flows on a :func:`hcu_layout` (no cracked molecules)."""
    f = hdt_feed(char, stream, _hdt_view(layout), aromatic_table)
    attr = jnp.concatenate([f.attr, jnp.zeros((f.attr.shape[0], 1))], axis=1)
    return Flows(gas=f.gas, cut=f.cut, attr=attr)


__all__ = ["VGO_PRETREAT_PARAMS", "hcu_layout", "hcu_feed"]
