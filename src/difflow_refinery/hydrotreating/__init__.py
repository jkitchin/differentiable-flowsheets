"""Hydrotreating (#306): HDS/HDN/HDA kinetics and the hydrotreater unit.

* :mod:`.kinetics` -- :class:`HDTKinetics`, :class:`HDTKineticParams` and the
  model-compound thermochemistry behind the stoichiometry and heats.
* :mod:`.feed` -- characterized stream -> attribute flows; :func:`straight_run_cut`.
* :mod:`.unit` -- :class:`Hydrotreater` (reactor + HPS + recycle + stripper).
* :mod:`.fractionator` -- :func:`fractionate`, the optional product fractionator (#328).
* :mod:`.planning` -- :func:`hdt_block` for delta-base planning.

See ``docs/unit-operations-refinery.md`` ("The hydrotreater").
"""

from difflow_refinery.hydrotreating import feed, fractionator, kinetics, unit
from difflow_refinery.hydrotreating.feed import DEFAULT_AROMATIC_SPLIT, hdt_feed, hdt_layout, straight_run_cut
from difflow_refinery.hydrotreating.fractionator import FractionationResult, fractionate
from difflow_refinery.hydrotreating.kinetics import (
    HDT_ATTRIBUTES, NAPHTHA_HDT_PARAMS, HDTKineticParams, HDTKinetics)
from difflow_refinery.hydrotreating.unit import (
    OUTPUT_UNITS, Hydrotreater, HydrotreaterConvergenceWarning, HydrotreaterParams, HydrotreaterResult,
    TargetSpec)

__all__ = [
    "feed", "fractionator", "kinetics", "unit",
    "DEFAULT_AROMATIC_SPLIT", "hdt_feed", "hdt_layout", "straight_run_cut",
    "FractionationResult", "fractionate",
    "HDT_ATTRIBUTES", "NAPHTHA_HDT_PARAMS", "HDTKineticParams", "HDTKinetics",
    "OUTPUT_UNITS", "Hydrotreater", "HydrotreaterConvergenceWarning", "HydrotreaterParams",
    "HydrotreaterResult", "TargetSpec",
]
