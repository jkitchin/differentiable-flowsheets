"""difflow_refinery: petroleum refining for difflow.

Phase 1 is crude characterisation: a TBP assay cut into pseudo-components
whose properties come from the standard petroleum correlations, differentiable
end to end with respect to the assay data.

>>> import difflow_refinery as dr
>>> assay = dr.Assay(tbp_percent=[0, 50, 100], tbp_T=[300.0, 600.0, 900.0], sg=0.85)
>>> crude = dr.characterize(assay)
"""

from difflow_refinery import correlations
from difflow_refinery.assay import (
    DEFAULT_CUT_WIDTHS,
    LIGHT_END_SG,
    Assay,
    Characterization,
    characterize,
    default_cut_points,
    fit_antoine,
)
from difflow_refinery.correlations import (
    CRITICAL_METHODS,
    acentric_factor,
    api_from_sg,
    critical_properties,
    sg_from_api,
    vapor_pressure,
    watson_k,
)

__all__ = [
    "Assay",
    "Characterization",
    "characterize",
    "default_cut_points",
    "fit_antoine",
    "DEFAULT_CUT_WIDTHS",
    "LIGHT_END_SG",
    "correlations",
    "CRITICAL_METHODS",
    "critical_properties",
    "acentric_factor",
    "vapor_pressure",
    "watson_k",
    "sg_from_api",
    "api_from_sg",
]
