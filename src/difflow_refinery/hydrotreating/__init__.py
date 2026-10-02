"""Hydrotreating (#306): HDS/HDN/HDA kinetics and the hydrotreater unit.

* :mod:`.kinetics` -- :class:`HDTKinetics`, :class:`HDTKineticParams` and the
  model-compound thermochemistry behind the stoichiometry and heats.
* :mod:`.feed` -- characterized stream -> attribute flows; :func:`straight_run_cut`.
* :mod:`.unit` -- :class:`Hydrotreater` (reactor + HPS + recycle + stripper).
* :mod:`.planning` -- :func:`hdt_block` for delta-base planning.

See ``docs/unit-operations-refinery.md`` ("The hydrotreater").
"""

from difflow_refinery.hydrotreating import feed, kinetics, unit
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, hdt_feed, hdt_layout, straight_run_cut
from difflow_refinery.hydrotreating.kinetics import HDT_ATTRIBUTES, HDTKineticParams, HDTKinetics
from difflow_refinery.hydrotreating.unit import (
    OUTPUT_UNITS, Hydrotreater, HydrotreaterConvergenceWarning, HydrotreaterParams, HydrotreaterResult,
    TargetSpec)

__all__ = [
    "feed", "kinetics", "unit",
    "DEFAULT_AROMATIC_SPLIT", "hdt_feed", "hdt_layout", "straight_run_cut",
    "HDT_ATTRIBUTES", "HDTKineticParams", "HDTKinetics",
    "OUTPUT_UNITS", "Hydrotreater", "HydrotreaterConvergenceWarning", "HydrotreaterParams",
    "HydrotreaterResult", "TargetSpec",
]
