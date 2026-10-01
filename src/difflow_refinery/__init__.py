"""difflow_refinery: petroleum refining for difflow.

Crude characterisation -- a TBP assay cut into pseudo-components whose
properties come from the standard petroleum correlations -- and the crude
unit that separates them: a furnace and an atmospheric column, an
equation-oriented MESH model with side strippers, pumparounds and stripping
steam, reporting its products as yields, gravities and TBP ranges -- and
the product blending pool (:mod:`difflow_refinery.blending`), which blends
finished components into gasoline, jet, ULSD or fuel oil with the
nonlinear rules refiners use and reports signed spec margins.

The vacuum unit (:mod:`difflow_refinery.vacuum`) takes the atmospheric
residue to LVGO, HVGO, slop and vacuum residue. It is built on its own
characterization (cut down to 800 C, with sulfur, nitrogen, CCR and metals
per cut) and its own stage-network column, which live in that subpackage.
The gas plant (:mod:`difflow_refinery.gasplant`) recovers the light ends:
absorber-deethanizer, debutanizer and splitters on a cubic EOS that carries
real light components and naphtha pseudocomponents together, with the
wet-gas compressor, amine treating as a removal fraction, and LPG, fuel gas
and naphtha qualities.
Differentiable end to end: a yield or a duty has a gradient with respect to
the column's specs, its feed, and the assay data behind its thermodynamics.

>>> import difflow_refinery as dr
>>> assay = dr.Assay(tbp_percent=[0, 50, 100], tbp_T=[300.0, 600.0, 900.0], sg=0.85)
>>> crude = dr.characterize(assay)
"""

from difflow_refinery import column, correlations, gasplant, products, vacuum
from difflow_refinery.assay import (
    DEFAULT_CUT_WIDTHS,
    LIGHT_END_SG,
    Assay,
    Characterization,
    characterize,
    default_cut_points,
    fit_antoine,
)
from difflow_refinery.blending import (
    PRODUCT_DERIVED,
    PRODUCT_SPECS,
    PROPERTY_RULES,
    TBP_D86,
    TEMPERATURE_INDEX_EXPONENTS,
    T_RVP,
    BlendComponent,
    BlendPool,
    BlendResult,
    BlendSpec,
    EthylRT70,
    cetane_index_d4737,
    cetane_index_d976,
    ethyl_rt70,
    flash_point_blend,
    mass_blend,
    raoult_rvp,
    refutas_blend,
    refutas_vbn,
    refutas_viscosity,
    rvp_index_blend,
    smooth_violation,
    tbp_evaporated,
    tbp_temperature,
    tbp_to_d86,
    temperature_index_blend,
    volume_blend,
)
from difflow_refinery.characterization import (
    PSI,
    RHO_WATER_15C,
    BlendCharacterization,
    edmister_omega,
    lee_kesler_psat,
    riazi_daubert_mw,
    riazi_daubert_pc,
    riazi_daubert_tc,
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
from difflow_refinery.gasplant import (
    AmineTreater,
    AmineTreaterParams,
    GasCompressor,
    GasCompressorParams,
    GasPlantColumn,
    GasPlantColumnParams,
)
from difflow_refinery.vacuum import (
    CrackingWarning,
    PseudoComponents,
    StageSpec,
    VacuumColumn,
    VacuumColumnParams,
    VacuumConvergenceWarning,
    default_vacuum_specs,
)
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
    "BlendComponent", "BlendPool", "BlendResult", "BlendSpec", "EthylRT70",
    "BlendCharacterization",
    "PRODUCT_DERIVED", "PRODUCT_SPECS", "PROPERTY_RULES", "TBP_D86",
    "TEMPERATURE_INDEX_EXPONENTS", "T_RVP", "PSI", "RHO_WATER_15C",
    "cetane_index_d4737", "cetane_index_d976", "ethyl_rt70",
    "flash_point_blend", "mass_blend", "raoult_rvp", "refutas_blend",
    "refutas_vbn", "refutas_viscosity", "rvp_index_blend",
    "smooth_violation", "tbp_evaporated", "tbp_temperature", "tbp_to_d86",
    "temperature_index_blend", "volume_blend",
    "edmister_omega", "lee_kesler_psat", "riazi_daubert_mw",
    "riazi_daubert_pc", "riazi_daubert_tc",
    "vacuum",
    "gasplant",
    "AmineTreater",
    "AmineTreaterParams",
    "GasCompressor",
    "GasCompressorParams",
    "GasPlantColumn",
    "GasPlantColumnParams",
    "CrackingWarning",
    "PseudoComponents",
    "StageSpec",
    "VacuumColumn",
    "VacuumColumnParams",
    "VacuumConvergenceWarning",
    "default_vacuum_specs",
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
    registry.register(
        name="VacuumColumn",
        cls=VacuumColumn,
        category="refinery",
        description="Vacuum distillation: atmospheric residue to LVGO, HVGO, "
                    "slop and vacuum residue",
        plugin="difflow_refinery",
    )
    registry.register(
        name="GasPlantColumn",
        cls=GasPlantColumn,
        category="refinery",
        description="Light-ends column on a cubic EOS: absorber-deethanizer, "
                    "debutanizer, C3/C4 splitter, deisobutanizer",
        plugin="difflow_refinery",
    )
    registry.register(
        name="GasCompressor",
        cls=GasCompressor,
        category="refinery",
        description="Wet-gas compressor: isentropic stages with intercoolers "
                    "and knockout drums",
        plugin="difflow_refinery",
    )
    registry.register(
        name="AmineTreater",
        cls=AmineTreater,
        category="refinery",
        description="Amine treating as a fixed H2S removal fraction",
        plugin="difflow_refinery",
    )
