"""Shared hydroprocessing machinery: trickle-bed reactor, HP separator, recycle-gas loop, stripper.

Kinetics-agnostic: the hydrotreater (:mod:`difflow_refinery.hydrotreating`)
plugs its kinetic model into these pieces, and a hydrocracker can plug in
another. See ``docs/unit-operations-refinery.md`` ("Hydroprocessing
building blocks") for the API contract.

* :mod:`.layout` -- :class:`Layout` (gases, cuts, per-cut attributes) and
  :class:`Flows`, the stream state with exact element bookkeeping.
* :mod:`.thermo` -- vectorised Peng-Robinson, enthalpies, Rackett volumes.
* :mod:`.separator` -- :func:`pr_flash` (negative flash, implicit
  gradients) and :class:`HPSeparator`.
* :mod:`.reactor` -- :class:`TrickleBedReactor`, the :class:`KineticModel`
  protocol, :class:`ReactionContext` and :class:`Rates`.
* :mod:`.recycle` -- knock-out, amine, purge, compressor, makeup and the
  Newton tear :func:`solve_tear`.
* :mod:`.stripper` -- steam stripper on :class:`~difflow_refinery.vacuum.column.StageColumn`
  and the overhead drum.
"""

from difflow_refinery.hydroprocessing import layout, reactor, recycle, separator, stripper, thermo
from difflow_refinery.hydroprocessing.layout import DEFAULT_GASES, Flows, Layout
from difflow_refinery.hydroprocessing.reactor import (
    KineticModel, Rates, ReactionContext, ReactorOptions, ReactorResult, TrickleBedReactor,
    check_element_conservation, integrate_bed)
from difflow_refinery.hydroprocessing.recycle import (
    amine_scrub, compress, knockout, makeup_for_ratio, purge_split, solve_tear)
from difflow_refinery.hydroprocessing.separator import FlashResult, HPSeparator, pr_flash
from difflow_refinery.hydroprocessing.stripper import StripperSpec, overhead_drum, strip
from difflow_refinery.hydroprocessing.thermo import Components

__all__ = [
    "layout", "reactor", "recycle", "separator", "stripper", "thermo",
    "DEFAULT_GASES", "Flows", "Layout", "Components",
    "KineticModel", "Rates", "ReactionContext", "ReactorOptions", "ReactorResult", "TrickleBedReactor",
    "check_element_conservation", "integrate_bed",
    "amine_scrub", "compress", "knockout", "makeup_for_ratio", "purge_split", "solve_tear",
    "FlashResult", "HPSeparator", "pr_flash", "StripperSpec", "overhead_drum", "strip",
]
