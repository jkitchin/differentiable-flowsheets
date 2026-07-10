"""Degradation module for biopharmaceutical products.

Provides models for product stability and degradation during:
- Manufacturing (hold times, processing stress)
- Storage (shelf life prediction)
- Formulation development

Degradation pathways modeled:
- Aggregation: Physical/chemical aggregation
- Deamidation: Asn/Gln modification
- Oxidation: Met/Trp/Cys oxidation
- Fragmentation: Peptide bond cleavage
- Glycation: Sugar modification

All functions are JAX-compatible for automatic differentiation.

References:
    Manning MC et al. (2010). Pharm Res 27:544.
        (Protein degradation pathways)
    Wang W et al. (2007). J Pharm Sci 96:1.
        (mAb stability considerations)
    Vlasak J, Ionescu R (2011). Curr Pharm Biotechnol 12:1526.
        (Heterogeneity in therapeutic antibodies)
"""

from difflow_bio.degradation.stability import (
    # Aggregation
    aggregation_rate,
    aggregation_arrhenius,
    aggregate_fraction,
    stretched_exponential_fraction,
    lumry_eyring_fraction,
    # Deamidation
    deamidation_rate,
    deamidation_ph_dependent,
    deamidation_fraction,
    # Oxidation
    oxidation_rate,
    oxidation_peroxide,
    oxidation_fraction,
    # Fragmentation
    fragmentation_rate,
    fragmentation_fraction,
    # Combined models
    total_degradation,
    shelf_life,
    # Classes
    DegradationModel,
    DegradationParams,
    get_degradation_model,
)

__all__ = [
    # Aggregation
    "aggregation_rate",
    "aggregation_arrhenius",
    "aggregate_fraction",
    # Deamidation
    "deamidation_rate",
    "deamidation_ph_dependent",
    "deamidation_fraction",
    # Oxidation
    "oxidation_rate",
    "oxidation_peroxide",
    "oxidation_fraction",
    # Fragmentation
    "fragmentation_rate",
    "fragmentation_fraction",
    # Combined
    "total_degradation",
    "shelf_life",
    # Classes
    "DegradationModel",
    "DegradationParams",
    "get_degradation_model",
]
