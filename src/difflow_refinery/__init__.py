"""difflow_refinery: refinery models for difflow.

Currently the product blending pool (:mod:`difflow_refinery.blending`) and
the small pseudocomponent characterization it computes properties from
(:mod:`difflow_refinery.characterization`).

Like :mod:`difflow.planning`, this is a library of models rather than a set
of stream-in/stream-out palette operations, so it registers no
``difflow.plugins`` entry point: a blend pool is called with a recipe and
returns properties and spec margins, which is not the shape a GUI unit has.

Example::

    from difflow_refinery import BlendComponent, BlendPool

    reformate = BlendComponent.from_properties(
        "reformate", SG=0.80, RON=98.0, MON=88.0, RVP_psi=3.5, S_ppm=1.0,
        olefins_vol=1.0, aromatics_vol=65.0)
    ...
    res = BlendPool("gasoline")([reformate, fcc, alkylate, butane],
                                recipe=[0.3, 0.4, 0.25, 0.05])
    res.properties["RON"], res.margins
"""

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
    Characterization,
    edmister_omega,
    lee_kesler_psat,
    riazi_daubert_mw,
    riazi_daubert_pc,
    riazi_daubert_tc,
)

__all__ = [
    "BlendComponent", "BlendPool", "BlendResult", "BlendSpec", "EthylRT70",
    "Characterization",
    "PRODUCT_DERIVED", "PRODUCT_SPECS", "PROPERTY_RULES", "TBP_D86",
    "TEMPERATURE_INDEX_EXPONENTS", "T_RVP", "PSI", "RHO_WATER_15C",
    "cetane_index_d4737", "cetane_index_d976", "ethyl_rt70",
    "flash_point_blend", "mass_blend", "raoult_rvp", "refutas_blend",
    "refutas_vbn", "refutas_viscosity", "rvp_index_blend",
    "smooth_violation", "tbp_evaporated", "tbp_temperature", "tbp_to_d86",
    "temperature_index_blend", "volume_blend",
    "edmister_omega", "lee_kesler_psat", "riazi_daubert_mw",
    "riazi_daubert_pc", "riazi_daubert_tc",
]
