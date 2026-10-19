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
* :mod:`.beds` -- :class:`DynamicEquilibriumBed`, an adiabatic equilibrium
  bed (shift, Claus, hydrogenation) in thermal cells.
* :mod:`.hydrogen` -- :class:`DynamicHydrogenPlant`: SMR tubes with tube and
  catalyst heat capacity, furnace lag, shift bed, PSA lag.
* :mod:`.treating` -- :class:`DynamicAminePlant`: the amine loop with its
  lean and rich solvent inventories.
* :mod:`.sulfur` -- :class:`DynamicSRU`: Claus and hydrogenation beds,
  reheaters, TGTU solvent inventory, air-demand control.
* :mod:`.crude` -- :class:`DynamicCrudeUnit`: furnace lag and product
  holdups around the steady crude column.
* :mod:`.signals` -- setpoints and disturbances as integrator ``args``.
"""

from difflow_refinery.dynamic.signals import evaluate, signal_args, smoothstep
from difflow_refinery.dynamic.hydroprocessing import (
    BedGeometry, DynamicHPSeparator, DynamicTrickleBedReactor, FurnaceLag, RecycleGasLoop, holdup_outflow)
from difflow_refinery.dynamic.hydrotreater import DynamicHydrotreater, HoldupParams

__all__ = ["evaluate", "signal_args", "smoothstep", "BedGeometry", "DynamicHPSeparator",
           "DynamicTrickleBedReactor", "FurnaceLag", "RecycleGasLoop", "holdup_outflow",
           "DynamicHydrotreater", "HoldupParams"]

from difflow_refinery.dynamic.reforming import DynamicReformerLoop, DynamicReformerTrain, DynamicReformingReactor  # noqa: E402

__all__ += ["DynamicReformerLoop", "DynamicReformerTrain", "DynamicReformingReactor"]

from difflow_refinery.dynamic.holdup import QuasiSteadyHoldup  # noqa: E402

__all__ += ["QuasiSteadyHoldup"]

from difflow_refinery.dynamic.beds import DynamicEquilibriumBed, catalyst_heat_capacity  # noqa: E402
from difflow_refinery.dynamic.hydrogen import (  # noqa: E402
    DynamicHydrogenPlant, DynamicReformerTube, FiringLag, SMRHoldupParams)
from difflow_refinery.dynamic.treating import AmineHoldupParams, DynamicAminePlant, DynamicAmineUnit  # noqa: E402
from difflow_refinery.dynamic.sulfur import AirDemandControl, DynamicSRU, SRUHoldupParams, TGTUSolvent  # noqa: E402
from difflow_refinery.dynamic.crude import CDUHoldupParams, DynamicCrudeUnit  # noqa: E402

__all__ += ["DynamicEquilibriumBed", "catalyst_heat_capacity", "DynamicHydrogenPlant", "DynamicReformerTube",
            "FiringLag", "SMRHoldupParams", "AmineHoldupParams", "DynamicAminePlant", "DynamicAmineUnit",
            "AirDemandControl", "DynamicSRU", "SRUHoldupParams", "TGTUSolvent", "CDUHoldupParams",
            "DynamicCrudeUnit"]

from difflow_refinery.dynamic.hydrocracker import (  # noqa: E402
    DynamicHydrocracker, DynamicRunawayWarning, HydrocrackerHoldupParams, QuenchControlledReactor)

__all__ += ["DynamicHydrocracker", "DynamicRunawayWarning", "HydrocrackerHoldupParams", "QuenchControlledReactor"]
