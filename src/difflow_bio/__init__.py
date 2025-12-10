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
