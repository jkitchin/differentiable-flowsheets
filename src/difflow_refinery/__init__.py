"""difflow_refinery: petroleum refining for difflow.

Crude characterisation -- a TBP assay cut into pseudo-components whose
properties come from the standard petroleum correlations -- and the crude
unit that separates them: a furnace and an atmospheric column, an
equation-oriented MESH model with side strippers, pumparounds and stripping
steam, reporting its products as yields, gravities and TBP ranges.
Differentiable end to end: a yield or a duty has a gradient with respect to
the column's specs, its feed, and the assay data behind its thermodynamics.

>>> import difflow_refinery as dr
>>> assay = dr.Assay(tbp_percent=[0, 50, 100], tbp_T=[300.0, 600.0, 900.0], sg=0.85)
>>> crude = dr.characterize(assay)
"""

from difflow_refinery import column, correlations, products
from difflow_refinery.assay import (
    DEFAULT_CUT_WIDTHS,
    LIGHT_END_SG,
    Assay,
    Characterization,
    characterize,
    default_cut_points,
    fit_antoine,
)
from difflow_refinery.column import (
    CrudeColumn,
    CrudeColumnParams,
    CrudeColumnResult,
    Furnace,
    Pumparound,
    SideProduct,
    Spec,
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
from difflow_refinery.products import ProductProperties, product_properties
from difflow_refinery.thermo import ColumnThermo, water_vapor_pressure
from difflow_refinery.unit import (
    CrudeDistillationUnit,
    CrudeDistillationUnitParams,
    CrudeUnit,
    CrudeUnitResult,
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
    "column",
    "ColumnThermo",
    "water_vapor_pressure",
    "CrudeColumn",
    "CrudeColumnParams",
    "CrudeColumnResult",
    "CrudeDistillationUnit",
    "CrudeDistillationUnitParams",
    "CrudeUnit",
    "CrudeUnitResult",
    "Furnace",
    "ProductProperties",
    "product_properties",
    "products",
    "Pumparound",
    "SideProduct",
    "Spec",
    "CRITICAL_METHODS",
    "critical_properties",
    "acentric_factor",
    "vapor_pressure",
    "watson_k",
    "sg_from_api",
    "api_from_sg",
    "register",
]


def register(registry):
    """Register the refinery unit operations with difflow.

    Called by ``difflow.plugins.load_plugins()`` when the plugin is
    discovered through its entry point.

    Args:
        registry: difflow OperationRegistry instance
    """
    registry.register(
        name="CrudeDistillationUnit",
        cls=CrudeDistillationUnit,
        category="refinery",
        description="Crude unit: furnace and atmospheric column with side "
                    "strippers and pumparounds, built on a TBP assay",
        plugin="difflow_refinery",
    )
