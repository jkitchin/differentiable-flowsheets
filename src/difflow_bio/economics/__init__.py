"""Economics module for biopharmaceutical manufacturing.

The cost-of-goods model is :func:`cogs_breakdown`: each chromatography step
has its own resin and its cycles follow the product load, labor follows the
batches and steps, and buffers, QC, single-use items and failed batches are
counted. Prices and the process come from data (:func:`load_cost_model` on a
YAML or JSON file; ``data/mab_reference.yaml`` is the annotated template).
The older ``estimate_*`` functions remain for coarse estimates.

Provides cost estimation for:
- Capital expenditures (CAPEX): equipment, facilities
- Operating expenditures (OPEX): consumables, labor, utilities
- Profitability analysis: NPV, IRR, cost per gram

References:
    Farid SS et al. (2007). Biotechnol Prog 23:3.
        Economic modeling of bioprocesses.
    Pollock J et al. (2013). Biotechnol Bioeng 110:206.
        DSP cost benchmarking.
"""

from difflow_bio.economics.cogs import (
    CATEGORIES,
    REFERENCE,
    ChromatographyStep,
    CostBasis,
    FiltrationStep,
    ProcessSpec,
    YieldStep,
    cogs_breakdown,
    cycles_per_batch,
    load_cost_model,
)
from difflow_bio.economics.costs import (
    # Cost dataclasses
    ConsumableCosts,
    EquipmentCosts,
    OperatingCosts,
    # CAPEX functions
    estimate_bioreactor_capex,
    estimate_chromatography_capex,
    estimate_filtration_capex,
    estimate_facility_capex,
    estimate_total_capex,
    # OPEX functions
    estimate_resin_cost,
    estimate_membrane_cost,
    estimate_media_cost,
    estimate_labor_cost,
    estimate_utilities_cost,
    estimate_total_opex,
    # Analysis functions
    calculate_cogs,
    calculate_profit,
    cost_per_gram,
)

__all__ = [
    # Step-by-step cost of goods (#358)
    "CATEGORIES",
    "REFERENCE",
    "ChromatographyStep",
    "CostBasis",
    "FiltrationStep",
    "ProcessSpec",
    "YieldStep",
    "cogs_breakdown",
    "cycles_per_batch",
    "load_cost_model",
    # Dataclasses
    "ConsumableCosts",
    "EquipmentCosts",
    "OperatingCosts",
    # CAPEX
    "estimate_bioreactor_capex",
    "estimate_chromatography_capex",
    "estimate_filtration_capex",
    "estimate_facility_capex",
    "estimate_total_capex",
    # OPEX
    "estimate_resin_cost",
    "estimate_membrane_cost",
    "estimate_media_cost",
    "estimate_labor_cost",
    "estimate_utilities_cost",
    "estimate_total_opex",
    # Analysis
    "calculate_cogs",
    "calculate_profit",
    "cost_per_gram",
]
