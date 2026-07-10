"""Difflow Bio: Bio manufacturing unit operations plugin for difflow.

This plugin provides unit operations for biopharmaceutical manufacturing:

Upstream (Cell Culture):
- ContinuousBioreactor (chemostat)
- FedBatchBioreactor
- Growth kinetics (Monod, substrate inhibition, etc.)

Downstream (Purification):
- Centrifuge, DiscStackCentrifuge
- Ultrafiltration, Diafiltration, TFF
- ProteinAChromatography, IonExchangeChromatography, SizeExclusionChromatography

Flowsheets:
- mAbDSPTrain: Standard 3-column mAb purification
- PlatformDSP: Configurable multi-step DSP
- ViralClearanceTrain: Viral safety focused purification

Kinetics:
- Growth models (Monod, Contois, logistic, Andrews, etc.)
- Production models (Luedeking-Piret, growth/non-growth associated)
- Metabolism models (substrate uptake, OUR, CER)

Degradation:
- Aggregation kinetics
- Deamidation and oxidation
- Shelf life prediction

Database:
- Cell line properties (CHO, HEK293, NS0)
- Chromatography resin properties
- Membrane properties

Equilibrium:
- Binding isotherm models (Langmuir, SMA, etc.)

Economics:
- CAPEX/OPEX estimation
- Cost per gram analysis

Usage:
    # Automatic loading via entry point
    from difflow.plugins import load_plugins
    load_plugins()

    # Or import directly
    from difflow_bio import ContinuousBioreactor, ProteinAChromatography
"""

# Bioreactor operations
from difflow_bio.units.bioreactors import (
    ContinuousBioreactor,
    FedBatchBioreactor,
    BioreactorParams,
    FedBatchParams,
    monod_kinetics,
    substrate_inhibition_kinetics,
    product_inhibition_kinetics,
    contois_kinetics,
    dilution_rate,
    residence_time,
    optimal_dilution_rate,
)

# Centrifuge operations
from difflow_bio.units.centrifuge import (
    Centrifuge,
    CentrifugeParams,
    DiscStackCentrifuge,
    DiscStackParams,
    stokes_velocity,
    critical_particle_diameter,
    disc_stack_sigma,
    tubular_bowl_sigma,
    centrifuge_scale_up,
    g_force,
)

# Filtration operations
from difflow_bio.units.filtration import (
    Ultrafiltration,
    UltrafiltrationParams,
    Diafiltration,
    DiafiltrationParams,
    TFF,
    concentration_polarization,
    gel_layer_flux,
    diavolumes_required,
    rejection_from_mw,
)

# Chromatography operations
from difflow_bio.units.chromatography import (
    ProteinAChromatography,
    ProteinAParams,
    IonExchangeChromatography,
    IEXParams,
    SizeExclusionChromatography,
    SECParams,
    langmuir_isotherm,
    linear_isotherm,
    langmuir_freundlich_isotherm,
    dynamic_binding_capacity,
    column_productivity,
    resolution,
    plate_count,
    hetp,
)

# Database
from difflow_bio.database import (
    CellLine,
    CellLineDatabase,
    get_cell_line,
    list_cell_lines,
    Resin,
    ResinDatabase,
    get_resin,
    list_resins,
    BioMembrane,
    BioMembraneDatabase,
    get_bio_membrane,
    list_bio_membranes,
    get_kinetic_params_array,
    get_resin_capacity_array,
)

# Equilibrium models
from difflow_bio.equilibrium import (
    langmuir_binding,
    langmuir_competitive,
    steric_mass_action,
    linear_partition,
    langmuir_ph_dependent,
    breakthrough_curve,
    column_efficiency,
    van_deemter,
    BindingIsotherm,
    get_binding_isotherm,
)

# Economics
from difflow_bio.economics import (
    ConsumableCosts,
    EquipmentCosts,
    OperatingCosts,
    estimate_bioreactor_capex,
    estimate_chromatography_capex,
    estimate_filtration_capex,
    estimate_facility_capex,
    estimate_total_capex,
    estimate_resin_cost,
    estimate_membrane_cost,
    estimate_media_cost,
    estimate_labor_cost,
    estimate_utilities_cost,
    estimate_total_opex,
    calculate_cogs,
    calculate_profit,
    cost_per_gram,
)

# Flowsheets
from difflow_bio.flowsheets import (
    mAbDSPTrain,
    mAbDSPParams,
    PlatformDSP,
    PlatformDSPParams,
    ViralClearanceTrain,
    ViralClearanceParams,
)

# Kinetics
from difflow_bio.kinetics import (
    # Growth
    monod,
    monod_inhibition,
    contois,
    logistic,
    tessier,
    moser,
    andrews,
    death_rate,
    net_growth_rate,
    GrowthModel,
    GrowthModelParams,
    get_growth_model,
    # Production
    luedeking_piret,
    growth_associated,
    non_growth_associated,
    overflow_production,
    product_inhibited_production,
    substrate_limited_production,
    ProductionModel,
    ProductionModelParams,
    get_production_model,
    # Metabolism
    substrate_uptake_rate,
    oxygen_uptake_rate,
    co2_evolution_rate,
    maintenance_energy,
    specific_substrate_uptake,
    yield_coefficient,
    metabolic_quotient,
    MetabolismModel,
    MetabolismParams,
    get_metabolism_model,
)

# Degradation
from difflow_bio.degradation import (
    aggregation_rate,
    aggregation_arrhenius,
    aggregate_fraction,
    stretched_exponential_fraction,
    lumry_eyring_fraction,
    deamidation_rate,
    deamidation_ph_dependent,
    deamidation_fraction,
    oxidation_rate,
    oxidation_peroxide,
    oxidation_fraction,
    fragmentation_rate,
    fragmentation_fraction,
    total_degradation,
    shelf_life,
    DegradationModel,
    DegradationParams,
    get_degradation_model,
)

__all__ = [
    # Bioreactors
    "ContinuousBioreactor",
    "FedBatchBioreactor",
    "BioreactorParams",
    "FedBatchParams",
    "monod_kinetics",
    "substrate_inhibition_kinetics",
    "product_inhibition_kinetics",
    "contois_kinetics",
    "dilution_rate",
    "residence_time",
    "optimal_dilution_rate",
    # Centrifuge
    "Centrifuge",
    "CentrifugeParams",
    "DiscStackCentrifuge",
    "DiscStackParams",
    "stokes_velocity",
    "critical_particle_diameter",
    "disc_stack_sigma",
    "tubular_bowl_sigma",
    "centrifuge_scale_up",
    "g_force",
    # Filtration
    "Ultrafiltration",
    "UltrafiltrationParams",
    "Diafiltration",
    "DiafiltrationParams",
    "TFF",
    "concentration_polarization",
    "gel_layer_flux",
    "diavolumes_required",
    "rejection_from_mw",
    # Chromatography
    "ProteinAChromatography",
    "ProteinAParams",
    "IonExchangeChromatography",
    "IEXParams",
    "SizeExclusionChromatography",
    "SECParams",
    "langmuir_isotherm",
    "linear_isotherm",
    "langmuir_freundlich_isotherm",
    "dynamic_binding_capacity",
    "column_productivity",
    "resolution",
    "plate_count",
    "hetp",
    # Database
    "CellLine",
    "CellLineDatabase",
    "get_cell_line",
    "list_cell_lines",
    "Resin",
    "ResinDatabase",
    "get_resin",
    "list_resins",
    "BioMembrane",
    "BioMembraneDatabase",
    "get_bio_membrane",
    "list_bio_membranes",
    "get_kinetic_params_array",
    "get_resin_capacity_array",
    # Equilibrium
    "langmuir_binding",
    "langmuir_competitive",
    "steric_mass_action",
    "linear_partition",
    "langmuir_ph_dependent",
    "breakthrough_curve",
    "column_efficiency",
    "van_deemter",
    "BindingIsotherm",
    "get_binding_isotherm",
    # Economics
    "ConsumableCosts",
    "EquipmentCosts",
    "OperatingCosts",
    "estimate_bioreactor_capex",
    "estimate_chromatography_capex",
    "estimate_filtration_capex",
    "estimate_facility_capex",
    "estimate_total_capex",
    "estimate_resin_cost",
    "estimate_membrane_cost",
    "estimate_media_cost",
    "estimate_labor_cost",
    "estimate_utilities_cost",
    "estimate_total_opex",
    "calculate_cogs",
    "calculate_profit",
    "cost_per_gram",
    # Flowsheets
    "mAbDSPTrain",
    "mAbDSPParams",
    "PlatformDSP",
    "PlatformDSPParams",
    "ViralClearanceTrain",
    "ViralClearanceParams",
    # Kinetics - Growth
    "monod",
    "monod_inhibition",
    "contois",
    "logistic",
    "tessier",
    "moser",
    "andrews",
    "death_rate",
    "net_growth_rate",
    "GrowthModel",
    "GrowthModelParams",
    "get_growth_model",
    # Kinetics - Production
    "luedeking_piret",
    "growth_associated",
    "non_growth_associated",
    "overflow_production",
    "product_inhibited_production",
    "substrate_limited_production",
    "ProductionModel",
    "ProductionModelParams",
    "get_production_model",
    # Kinetics - Metabolism
    "substrate_uptake_rate",
    "oxygen_uptake_rate",
    "co2_evolution_rate",
    "maintenance_energy",
    "specific_substrate_uptake",
    "yield_coefficient",
    "metabolic_quotient",
    "MetabolismModel",
    "MetabolismParams",
    "get_metabolism_model",
    # Degradation
    "aggregation_rate",
    "aggregation_arrhenius",
    "aggregate_fraction",
    "stretched_exponential_fraction",
    "lumry_eyring_fraction",
    "deamidation_rate",
    "deamidation_ph_dependent",
    "deamidation_fraction",
    "oxidation_rate",
    "oxidation_peroxide",
    "oxidation_fraction",
    "fragmentation_rate",
    "fragmentation_fraction",
    "total_degradation",
    "shelf_life",
    "DegradationModel",
    "DegradationParams",
    "get_degradation_model",
    # Registration
    "register",
]


def register(registry):
    """Register bio manufacturing operations with difflow plugin system.

    This function is called automatically when difflow.plugins.load_plugins()
    is invoked, via the entry point defined in pyproject.toml.

    Args:
        registry: The difflow OperationRegistry instance
    """
    # Bioreactors
    registry.register(
        "ContinuousBioreactor",
        ContinuousBioreactor,
        category="bioreactors",
        description="Continuous stirred-tank bioreactor (chemostat)",
        plugin="difflow_bio",
    )
    registry.register(
        "FedBatchBioreactor",
        FedBatchBioreactor,
        category="bioreactors",
        description="Fed-batch bioreactor with substrate feeding",
        plugin="difflow_bio",
    )

    # Centrifuge
    registry.register(
        "Centrifuge",
        Centrifuge,
        category="separations",
        description="Centrifuge using Sigma factor theory",
        plugin="difflow_bio",
    )
    registry.register(
        "DiscStackCentrifuge",
        DiscStackCentrifuge,
        category="separations",
        description="Disc-stack centrifuge for continuous separation",
        plugin="difflow_bio",
    )

    # Filtration
    registry.register(
        "Ultrafiltration",
        Ultrafiltration,
        category="filtration",
        description="Ultrafiltration for protein concentration",
        plugin="difflow_bio",
    )
    registry.register(
        "Diafiltration",
        Diafiltration,
        category="filtration",
        description="Diafiltration for buffer exchange",
        plugin="difflow_bio",
    )
    registry.register(
        "TFF",
        TFF,
        category="filtration",
        description="Tangential flow filtration system",
        plugin="difflow_bio",
    )

    # Chromatography
    registry.register(
        "ProteinAChromatography",
        ProteinAChromatography,
        category="chromatography",
        description="Protein A affinity chromatography for mAb capture",
        plugin="difflow_bio",
    )
    registry.register(
        "IonExchangeChromatography",
        IonExchangeChromatography,
        category="chromatography",
        description="Ion exchange chromatography (CEX/AEX)",
        plugin="difflow_bio",
    )
    registry.register(
        "SizeExclusionChromatography",
        SizeExclusionChromatography,
        category="chromatography",
        description="Size exclusion chromatography for polishing",
        plugin="difflow_bio",
    )
