"""Dynamic (holdup-based) versions of the refinery units.

The steady units of :mod:`difflow_refinery` solve for a steady state. The
classes here give them holdups -- material in reactor cells, separators and
gas loops; heat in catalyst and furnaces -- and implement difflow's
:class:`~difflow.dynamic.DynamicUnit` protocol, so they run in a
:class:`~difflow.dynamic.DynamicFlowsheet` and every trajectory is
differentiable. They reuse the steady units' rate laws, phase models and
enthalpies, and their ``t -> infinity`` limit is the steady unit's answer
(tested; see ``docs/unit-operations-refinery.md``, "Dynamic operation").

* :mod:`.hydroprocessing` -- trickle beds of cells, charge heater, HP
  separator, recycle-gas loop.
* :mod:`.hydrotreater` -- :class:`DynamicHydrotreater`, built from a solved
  :class:`~difflow_refinery.hydrotreating.Hydrotreater`.
* :mod:`.reforming` -- :class:`DynamicReformerTrain`: the reforming beds and
  interstage heaters with catalyst heat capacity and coke on catalyst.
* :mod:`.holdup` -- :class:`QuasiSteadyHoldup`, a steady unit behind outlet
  holdups (for units whose internals are fast).
* :mod:`.signals` -- setpoints and disturbances as integrator ``args``.
"""

from difflow_refinery.dynamic.signals import evaluate, signal_args, smoothstep
from difflow_refinery.dynamic.hydroprocessing import (
    BedGeometry, DynamicHPSeparator, DynamicTrickleBedReactor, FurnaceLag, RecycleGasLoop, holdup_outflow)
from difflow_refinery.dynamic.hydrotreater import DynamicHydrotreater, HoldupParams

__all__ = ["evaluate", "signal_args", "smoothstep", "BedGeometry", "DynamicHPSeparator",
           "DynamicTrickleBedReactor", "FurnaceLag", "RecycleGasLoop", "holdup_outflow",
           "DynamicHydrotreater", "HoldupParams"]

from difflow_refinery.dynamic.reforming import DynamicReformerTrain, DynamicReformingReactor  # noqa: E402

__all__ += ["DynamicReformerTrain", "DynamicReformingReactor"]

from difflow_refinery.dynamic.holdup import QuasiSteadyHoldup  # noqa: E402

__all__ += ["QuasiSteadyHoldup"]
