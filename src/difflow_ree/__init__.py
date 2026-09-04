"""Rare Earth Element Separation Plugin for difflow.

This plugin provides comprehensive tools for modeling and optimizing
rare earth element (REE) solvent extraction processes.

Features:
- Database of 10 commercial REE properties
- 4 extractant systems (D2EHPA, PC88A, Cyanex272, TBP)
- pH-dependent distribution coefficient models
- Loading and speciation corrections
- Unit operations: extraction, scrubbing, stripping, precipitation
- Pre-built flowsheet templates
- Kinetics: mass transfer and extraction kinetics
- Degradation: extractant degradation models
- Economic analysis tools

Quick Start:
    >>> from difflow_ree import (
    ...     get_element, get_extractant,
    ...     REEExtractor, ExtractStripCircuit,
    ... )
    >>>
    >>> # Get Nd properties
    >>> nd = get_element("Nd")
    >>> print(f"Nd price: ${nd.price_usd_kg}/kg")
    >>>
    >>> # Create extraction circuit
    >>> from difflow_ree.flowsheets import ExtractStripParams, ExtractStripCircuit
    >>> params = ExtractStripParams(
    ...     extractant="D2EHPA",
    ...     elements=("La", "Nd", "Dy"),
    ... )
    >>> circuit = ExtractStripCircuit(params)
"""

__version__ = "0.1.0"

# =============================================================================
# Database Access
# =============================================================================

from difflow_ree.database import (
    # Data classes
    REEElement,
    PHCoefficients,
    Extractant,
    SeparationFactorData,
    # Database classes
    REEDatabase,
    ExtractantDatabase,
    SeparationFactorDatabase,
    # Database singletons
    get_ree_database,
    get_extractant_database,
    get_sf_database,
    # Convenience functions
    get_element,
    get_extractant,
    list_ree_elements,
    list_extractants,
    get_separation_factor,
    # Custom creation helpers
    create_custom_element,
    create_custom_extractant,
    # Extraction mechanisms (#195)
    EXTRACTION_MECHANISMS,
    normalize_mechanism,
    # JAX-compatible accessors
    get_atomic_weight_array,
    get_price_array,
    get_ionic_radius_array,
)

# =============================================================================
# Equilibrium Models
# =============================================================================

from difflow_ree.equilibrium import (
    # Distribution coefficients
    REEDistribution,
    get_distribution_coefficient,
    get_distribution_coefficients,
    # Loading
    LoadingIsotherm,
    langmuir_loading,
    loading_correction,
    # Speciation
    REESpeciation,
    sulfate_speciation,
    chloride_speciation,
    # Activity models and their declared validity ranges (#194)
    ACTIVITY_MODELS,
    AQUEOUS_MEDIA,
    NITRATE_BEARING_MEDIA,
    DAVIES_MAX_IONIC_STRENGTH,
    DAVIES_SIGN_CHANGE_IONIC_STRENGTH,
    activity_coefficient,
)

# =============================================================================
# Unit Operations
# =============================================================================

from difflow_ree.units import (
    # Extraction
    REEExtractor,
    REEExtractorParams,
    REEMixerSettler,
    MixerSettlerParams,
    # Scrubbing
    REEScrubber,
    ScrubberParams,
    # Stripping
    REEStripper,
    StripperParams,
    # Precipitation
    OxalatePrecipitator,
    CarbonatePrecipitator,
    HydroxidePrecipitator,
    PrecipitatorParams,
    # Cerium
    CeriumOxidizer,
    CeriumOxidizerParams,
)

# =============================================================================
# Flowsheet Templates
# =============================================================================

from difflow_ree.flowsheets import (
    ExtractStripCircuit,
    ExtractStripParams,
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
    SplitShellCascade,
    SplitShellParams,
    FullSeparationTrain,
    SeparationTrainParams,
    GroupSeparator,
)

# =============================================================================
# Economics
# =============================================================================

from difflow_ree.economics import (
    REEPricing,
    ReagentCosts,
    OperatingCosts,
    estimate_capex,
    estimate_opex,
    calculate_revenue,
    calculate_profit,
    minimum_selling_price,
)

# =============================================================================
# Kinetics
# =============================================================================

from difflow_ree.kinetics import (
    # Mass transfer
    film_mass_transfer,
    overall_mass_transfer,
    diffusion_coefficient,
    sherwood_correlation,
    mass_transfer_rate,
    enhancement_factor,
    MassTransferModel,
    MassTransferParams,
    get_mass_transfer_model,
    # Extraction kinetics
    forward_extraction_rate,
    reverse_extraction_rate,
    net_extraction_rate,
    approach_to_equilibrium,
    stage_residence_time,
    ExtractionKineticsModel,
    ExtractionKineticsParams,
    get_extraction_kinetics_model,
)

# =============================================================================
# Degradation
# =============================================================================

from difflow_ree.degradation import (
    oxidative_degradation_rate,
    hydrolytic_degradation_rate,
    solubility_loss_rate,
    total_degradation_rate,
    extractant_lifetime,
    makeup_rate,
    ExtractantDegradationModel,
    ExtractantDegradationParams,
    get_degradation_model,
)


# =============================================================================
# Plugin Registration
# =============================================================================

def register(registry):
    """Register REE plugin operations with difflow.

    This function is called by difflow.plugins.load_plugins()
    when the plugin is discovered via entry points.

    Args:
        registry: difflow OperationRegistry instance
    """
    # Register extraction operations
    registry.register(
        name="REEExtractor",
        cls=REEExtractor,
        category="ree_extraction",
        description="Multi-stage REE solvent extraction cascade",
        plugin="difflow_ree",
    )

    registry.register(
        name="REEMixerSettler",
        cls=REEMixerSettler,
        category="ree_extraction",
        description="Single mixer-settler stage for REE extraction",
        plugin="difflow_ree",
    )

    # Register scrubbing operations
    registry.register(
        name="REEScrubber",
        cls=REEScrubber,
        category="ree_extraction",
        description="Multi-stage scrubbing section for impurity removal",
        plugin="difflow_ree",
    )

    # Register stripping operations
    registry.register(
        name="REEStripper",
        cls=REEStripper,
        category="ree_extraction",
        description="Multi-stage stripping section for product recovery",
        plugin="difflow_ree",
    )

    # Register precipitation operations
    registry.register(
        name="OxalatePrecipitator",
        cls=OxalatePrecipitator,
        category="ree_precipitation",
        description="Oxalate precipitation for REE recovery",
        plugin="difflow_ree",
    )

    registry.register(
        name="CarbonatePrecipitator",
        cls=CarbonatePrecipitator,
        category="ree_precipitation",
        description="Carbonate precipitation for REE recovery",
        plugin="difflow_ree",
    )

    registry.register(
        name="HydroxidePrecipitator",
        cls=HydroxidePrecipitator,
        category="ree_precipitation",
        description="Hydroxide precipitation for selective REE separation",
        plugin="difflow_ree",
    )

    # Register cerium operations
    registry.register(
        name="CeriumOxidizer",
        cls=CeriumOxidizer,
        category="ree_extraction",
        description="Cerium oxidation and CeO2 precipitation",
        plugin="difflow_ree",
    )

    # Register flowsheet templates
    registry.register(
        name="ExtractStripCircuit",
        cls=ExtractStripCircuit,
        category="ree_flowsheets",
        description="Basic 2-section extract-strip circuit",
        plugin="difflow_ree",
    )

    registry.register(
        name="ExtractScrubStripCircuit",
        cls=ExtractScrubStripCircuit,
        category="ree_flowsheets",
        description="Industrial 3-section extract-scrub-strip circuit",
        plugin="difflow_ree",
    )

    registry.register(
        name="SplitShellCascade",
        cls=SplitShellCascade,
        category="ree_flowsheets",
        description="Multi-product split-shell cascade design",
        plugin="difflow_ree",
    )

    registry.register(
        name="FullSeparationTrain",
        cls=FullSeparationTrain,
        category="ree_flowsheets",
        description="Complete REE separation train",
        plugin="difflow_ree",
    )

    registry.register(
        name="GroupSeparator",
        cls=GroupSeparator,
        category="ree_flowsheets",
        description="Light/Middle/Heavy REE group separator",
        plugin="difflow_ree",
    )

    return {
        "name": "difflow_ree",
        "version": __version__,
        "operations": [
            "REEExtractor",
            "REEMixerSettler",
            "REEScrubber",
            "REEStripper",
            "OxalatePrecipitator",
            "CarbonatePrecipitator",
            "HydroxidePrecipitator",
            "CeriumOxidizer",
            "ExtractStripCircuit",
            "ExtractScrubStripCircuit",
            "SplitShellCascade",
            "FullSeparationTrain",
            "GroupSeparator",
        ],
    }


# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Data classes
    "REEElement",
    "PHCoefficients",
    "Extractant",
    "SeparationFactorData",
    # Database classes
    "REEDatabase",
    "ExtractantDatabase",
    "SeparationFactorDatabase",
    # Database singletons
    "get_ree_database",
    "get_extractant_database",
    "get_sf_database",
    # Convenience functions
    "get_element",
    "get_extractant",
    "list_ree_elements",
    "list_extractants",
    "get_separation_factor",
    # Custom creation helpers
    "create_custom_element",
    "create_custom_extractant",
    # JAX-compatible accessors
    "get_atomic_weight_array",
    "get_price_array",
    "get_ionic_radius_array",
    # Equilibrium
    "REEDistribution",
    "get_distribution_coefficient",
    "get_distribution_coefficients",
    "LoadingIsotherm",
    "langmuir_loading",
    "loading_correction",
    "REESpeciation",
    "sulfate_speciation",
    "chloride_speciation",
    "ACTIVITY_MODELS",
    "AQUEOUS_MEDIA",
    "NITRATE_BEARING_MEDIA",
    "DAVIES_MAX_IONIC_STRENGTH",
    "DAVIES_SIGN_CHANGE_IONIC_STRENGTH",
    "activity_coefficient",
    "EXTRACTION_MECHANISMS",
    "normalize_mechanism",
    # Units
    "REEExtractor",
    "REEExtractorParams",
    "REEMixerSettler",
    "MixerSettlerParams",
    "REEScrubber",
    "ScrubberParams",
    "REEStripper",
    "StripperParams",
    "OxalatePrecipitator",
    "CarbonatePrecipitator",
    "HydroxidePrecipitator",
    "PrecipitatorParams",
    "CeriumOxidizer",
    "CeriumOxidizerParams",
    # Flowsheets
    "ExtractStripCircuit",
    "ExtractStripParams",
    "ExtractScrubStripCircuit",
    "ExtractScrubStripParams",
    "SplitShellCascade",
    "SplitShellParams",
    "FullSeparationTrain",
    "SeparationTrainParams",
    "GroupSeparator",
    # Economics
    "REEPricing",
    "ReagentCosts",
    "OperatingCosts",
    "estimate_capex",
    "estimate_opex",
    "calculate_revenue",
    "calculate_profit",
    "minimum_selling_price",
    # Kinetics - Mass transfer
    "film_mass_transfer",
    "overall_mass_transfer",
    "diffusion_coefficient",
    "sherwood_correlation",
    "mass_transfer_rate",
    "enhancement_factor",
    "MassTransferModel",
    "MassTransferParams",
    "get_mass_transfer_model",
    # Kinetics - Extraction
    "forward_extraction_rate",
    "reverse_extraction_rate",
    "net_extraction_rate",
    "approach_to_equilibrium",
    "stage_residence_time",
    "ExtractionKineticsModel",
    "ExtractionKineticsParams",
    "get_extraction_kinetics_model",
    # Degradation
    "oxidative_degradation_rate",
    "hydrolytic_degradation_rate",
    "solubility_loss_rate",
    "total_degradation_rate",
    "extractant_lifetime",
    "makeup_rate",
    "ExtractantDegradationModel",
    "ExtractantDegradationParams",
    "get_degradation_model",
    # Registration
    "register",
]
