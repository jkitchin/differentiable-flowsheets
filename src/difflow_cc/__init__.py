"""Carbon Capture Plugin for difflow.

This plugin provides comprehensive tools for modeling and optimizing
carbon capture processes using:

- **Amine-based absorption**: MEA, DEA, MDEA, piperazine, amino acids
- **Membrane separation**: Polymeric, mixed-matrix, facilitated transport
- **Adsorption systems**: PSA, TSA, VSA, TVSA with various adsorbents

All models are fully differentiable using JAX, enabling gradient-based
optimization, sensitivity analysis, and integration with machine learning.

Quick Start:
    >>> from difflow_cc import (
    ...     get_solvent, get_adsorbent, get_membrane,
    ...     AmineAbsorber, AbsorberParams,
    ...     PSAUnit, AdsorptionParams,
    ... )
    >>>
    >>> # Get MEA properties
    >>> mea = get_solvent('MEA')
    >>> print(f"MEA heat of absorption: {mea.heat_of_absorption} kJ/mol")
    >>>
    >>> # Create amine absorber
    >>> params = AbsorberParams(solvent='MEA', n_stages=10)
    >>> absorber = AmineAbsorber(params)

Features:
    - Database of 7 amine solvents with kinetic and thermodynamic data
    - Database of 8 adsorbent materials with isotherm parameters
    - Database of 9 membrane materials with permeability data
    - Simplified equilibrium-stage models (fast, differentiable)
    - Extensibility hooks for rigorous rate-based models
    - JAX-compatible throughout for automatic differentiation

References:
    Rochelle GT (2009). Amine scrubbing for CO2 capture. Science 325:1652.
    Ruthven DM (1984). Principles of Adsorption and Adsorption Processes.
    Baker RW (2012). Membrane Technology and Applications, 3rd ed.
"""

__version__ = "0.1.0"

# =============================================================================
# Database Access
# =============================================================================

from difflow_cc.database import (
    # Database classes
    SolventDatabase,
    AdsorbentDatabase,
    MembraneDatabase,
    # Data classes
    AmineSolvent,
    SolventBlend,
    Adsorbent,
    IsothermParams,
    Membrane,
    # Convenience functions
    get_solvent,
    get_adsorbent,
    get_membrane,
    list_solvents,
    list_adsorbents,
    list_membranes,
    # JAX accessors
    get_solvent_kinetic_params,
    get_isotherm_params_array,
)

# =============================================================================
# Equilibrium Models
# =============================================================================

from difflow_cc.equilibrium import (
    # Isotherms
    langmuir,
    sips,
    toth,
    dual_site_langmuir,
    langmuir_T,
    sips_T,
    toth_T,
    dual_site_langmuir_T,
    Isotherm,
    get_isotherm,
    working_capacity_PSA,
    working_capacity_TSA,
    # VLE
    henry_constant,
    co2_equilibrium_pressure,
    co2_loading,
    AmineVLE,
    # Solubility
    co2_physical_solubility,
    diffusivity_co2_amine,
)

# =============================================================================
# Kinetics Models
# =============================================================================

from difflow_cc.kinetics import (
    # Reaction kinetics
    reaction_rate_constant,
    enhancement_factor,
    hatta_number,
    pseudo_first_order_rate,
    # Mass transfer
    gas_film_coefficient,
    liquid_film_coefficient,
    overall_mass_transfer,
    interfacial_area,
)

# =============================================================================
# Unit Operations
# =============================================================================

from difflow_cc.units import (
    # Amine absorption
    AbsorberParams,
    AmineAbsorber,
    # Amine stripping
    StripperParams,
    AmineStripper,
    # Membrane separation
    MembraneParams,
    MembraneSeparator,
    MultistageMembrane,
    # Adsorption
    AdsorptionParams,
    PSAUnit,
    TSAUnit,
    VSAUnit,
    TVSAUnit,
)


# =============================================================================
# Plugin Registration
# =============================================================================

def register(registry):
    """Register carbon capture plugin operations with difflow.

    This function is called by difflow.plugins.load_plugins()
    when the plugin is discovered via entry points.

    Args:
        registry: difflow OperationRegistry instance
    """
    # Register amine absorption units
    registry.register(
        name="AmineAbsorber",
        cls=AmineAbsorber,
        category="carbon_capture_amine",
        description="Amine absorption column for CO2 capture",
        plugin="difflow_cc",
    )

    registry.register(
        name="AmineStripper",
        cls=AmineStripper,
        category="carbon_capture_amine",
        description="Amine stripper for solvent regeneration",
        plugin="difflow_cc",
    )

    # Register membrane units
    registry.register(
        name="MembraneSeparator",
        cls=MembraneSeparator,
        category="carbon_capture_membrane",
        description="Single-stage membrane gas separator",
        plugin="difflow_cc",
    )

    registry.register(
        name="MultistageMembrane",
        cls=MultistageMembrane,
        category="carbon_capture_membrane",
        description="Multi-stage membrane cascade",
        plugin="difflow_cc",
    )

    # Register adsorption units
    registry.register(
        name="PSAUnit",
        cls=PSAUnit,
        category="carbon_capture_adsorption",
        description="Pressure swing adsorption unit",
        plugin="difflow_cc",
    )

    registry.register(
        name="TSAUnit",
        cls=TSAUnit,
        category="carbon_capture_adsorption",
        description="Temperature swing adsorption unit",
        plugin="difflow_cc",
    )

    registry.register(
        name="VSAUnit",
        cls=VSAUnit,
        category="carbon_capture_adsorption",
        description="Vacuum swing adsorption unit",
        plugin="difflow_cc",
    )

    registry.register(
        name="TVSAUnit",
        cls=TVSAUnit,
        category="carbon_capture_adsorption",
        description="Temperature-vacuum swing adsorption unit",
        plugin="difflow_cc",
    )

    return {
        "name": "difflow_cc",
        "version": __version__,
        "operations": [
            "AmineAbsorber",
            "AmineStripper",
            "MembraneSeparator",
            "MultistageMembrane",
            "PSAUnit",
            "TSAUnit",
            "VSAUnit",
            "TVSAUnit",
        ],
    }


# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Database classes
    "SolventDatabase",
    "AdsorbentDatabase",
    "MembraneDatabase",
    "AmineSolvent",
    "SolventBlend",
    "Adsorbent",
    "IsothermParams",
    "Membrane",
    # Convenience functions
    "get_solvent",
    "get_adsorbent",
    "get_membrane",
    "list_solvents",
    "list_adsorbents",
    "list_membranes",
    "get_solvent_kinetic_params",
    "get_isotherm_params_array",
    # Isotherms
    "langmuir",
    "sips",
    "toth",
    "dual_site_langmuir",
    "langmuir_T",
    "sips_T",
    "toth_T",
    "dual_site_langmuir_T",
    "Isotherm",
    "get_isotherm",
    "working_capacity_PSA",
    "working_capacity_TSA",
    # VLE
    "henry_constant",
    "co2_equilibrium_pressure",
    "co2_loading",
    "AmineVLE",
    # Solubility
    "co2_physical_solubility",
    "diffusivity_co2_amine",
    # Kinetics
    "reaction_rate_constant",
    "enhancement_factor",
    "hatta_number",
    "pseudo_first_order_rate",
    "gas_film_coefficient",
    "liquid_film_coefficient",
    "overall_mass_transfer",
    "interfacial_area",
    # Units - Amine
    "AbsorberParams",
    "AmineAbsorber",
    "StripperParams",
    "AmineStripper",
    # Units - Membrane
    "MembraneParams",
    "MembraneSeparator",
    "MultistageMembrane",
    # Units - Adsorption
    "AdsorptionParams",
    "PSAUnit",
    "TSAUnit",
    "VSAUnit",
    "TVSAUnit",
    # Registration
    "register",
]
