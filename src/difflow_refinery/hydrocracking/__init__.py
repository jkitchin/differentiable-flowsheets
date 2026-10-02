"""VGO hydrocracking (#307): continuous-lumping cracking kinetics and the hydrocracker unit.

* :mod:`.kinetics` -- :class:`HCKinetics` (continuous lumping after
  Laxminarasimhan, Verma & Ramachandran 1996, or discrete lumps; organic-N
  inhibition; H2 from the hydrogen balance; heat per H2), :class:`HCKineticParams`.
* :mod:`.feed` -- the hydrocracker layout and feed; :data:`VGO_PRETREAT_PARAMS`.
* :mod:`.fractionator` -- the simplified product split by TBP cut points.
* :mod:`.fixed_point` -- substitution with an adjoint gradient (the UCO recycle).
* :mod:`.unit` -- :class:`Hydrocracker` (pretreat + cracking reactors, HPS,
  recycle gas, fractionator, UCO recycle).
* :mod:`.planning` -- :func:`hcu_block` for delta-base planning.

See ``docs/unit-operations-refinery.md`` ("The hydrocracker").
"""

from difflow_refinery.hydrocracking import feed, fixed_point, fractionator, kinetics, unit
from difflow_refinery.hydrocracking.feed import VGO_PRETREAT_PARAMS, hcu_feed, hcu_layout
from difflow_refinery.hydrocracking.fractionator import DEFAULT_CUT_POINTS, LIQUID_PRODUCTS, PRODUCTS
from difflow_refinery.hydrocracking.kinetics import (
    CRACK_BED_HDT_PARAMS, HCU_ATTRIBUTES, HCU_GAS_SPLIT, HCKineticParams, HCKinetics)
from difflow_refinery.hydrocracking.unit import (
    OUTPUT_UNITS, Hydrocracker, HydrocrackerConvergenceWarning, HydrocrackerParams, HydrocrackerResult, bmci)

__all__ = [
    "feed", "fixed_point", "fractionator", "kinetics", "unit",
    "VGO_PRETREAT_PARAMS", "hcu_feed", "hcu_layout",
    "DEFAULT_CUT_POINTS", "LIQUID_PRODUCTS", "PRODUCTS",
    "CRACK_BED_HDT_PARAMS", "HCU_ATTRIBUTES", "HCU_GAS_SPLIT", "HCKineticParams", "HCKinetics",
    "OUTPUT_UNITS", "Hydrocracker", "HydrocrackerConvergenceWarning", "HydrocrackerParams",
    "HydrocrackerResult", "bmci",
]
