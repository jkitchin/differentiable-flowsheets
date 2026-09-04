"""Equilibrium models for REE solvent extraction.

Two modelling levels sit here, behind one interface (#196):

**L1, the correlation.** ``REEDistribution`` evaluates ``log10(D)`` at a
*specified* pH from tabulated coefficients, with loading and speciation as
multiplicative corrections.

**L2, the closed mass-action model.** ``MassActionSection`` solves a whole
counter-current section's component balances against a reaction network
carried as data (``network``, ``data/reaction_networks.yaml``), in log
concentration, through ``difflow.eo_solver.solve_residual_system``. pH is an
*output* of the proton balance, competitive loading is an outcome of one
shared free-extractant balance, and every component is conserved exactly.

The stream vocabulary is declared once as a superset in ``schema`` so cascade
code can be level-agnostic; the two levels' *degrees of freedom* genuinely
differ (pH is an input at L1 and an output at L2), and
``mass_action.base_addition_for_pH`` is the explicit inverse problem that maps
between them.

This module provides:
- pH-dependent distribution coefficient models
- Extractant loading/saturation effects
- Aqueous phase speciation
- Reaction networks as data, and the mass-action closure over them (#196)
"""

from difflow_ree.equilibrium.distribution import (
    REEDistribution,
    get_distribution_coefficient,
    get_distribution_coefficients,
    get_separation_factor,
)
from difflow_ree.equilibrium.loading import (
    LoadingIsotherm,
    langmuir_loading,
    loading_correction,
)
from difflow_ree.equilibrium.network import (
    # Reaction network carried as data (#196)
    NETWORK_MECHANISMS,
    COMPONENT_ROLES,
    Component,
    Reaction,
    NetworkTemplate,
    ReactionNetwork,
    list_networks,
    get_network_template,
    build_network,
    log_K_from_correlation,
    correlation_ph_slope_defect,
    network_for_extractant,
    cation_exchange_network,
)
from difflow_ree.equilibrium.schema import (
    # The stream schema superset shared by both levels (#196)
    ANION_CHARGES,
    COUNTER_ION_CHARGES,
    REEStreamSchema,
)
from difflow_ree.equilibrium.mass_action import (
    # Mass-action closure (#196)
    ANION_CLOSURES,
    MassActionParams,
    MassActionSection,
    MassActionSolution,
    make_section_residual,
    section_scales,
    aqueous_component_totals,
    organic_component_totals,
    correlation_initial_guess,
    solve_section,
    solve_stage,
    charge_imbalance,
    base_addition_bounds,
    base_addition_for_pH,
)
from difflow_ree.equilibrium.speciation import (
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
    activity_coefficient_davies,
)

__all__ = [
    "REEDistribution",
    "get_distribution_coefficient",
    "get_distribution_coefficients",
    "get_separation_factor",
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
    "activity_coefficient_davies",
    # Reaction networks as data (#196)
    "NETWORK_MECHANISMS",
    "COMPONENT_ROLES",
    "Component",
    "Reaction",
    "NetworkTemplate",
    "ReactionNetwork",
    "list_networks",
    "get_network_template",
    "build_network",
    "log_K_from_correlation",
    "correlation_ph_slope_defect",
    "network_for_extractant",
    "cation_exchange_network",
    # Stream schema superset (#196)
    "ANION_CHARGES",
    "COUNTER_ION_CHARGES",
    "REEStreamSchema",
    # Mass-action closure (#196)
    "ANION_CLOSURES",
    "MassActionParams",
    "MassActionSection",
    "MassActionSolution",
    "make_section_residual",
    "section_scales",
    "aqueous_component_totals",
    "organic_component_totals",
    "correlation_initial_guess",
    "solve_section",
    "solve_stage",
    "charge_imbalance",
    "base_addition_bounds",
    "base_addition_for_pH",
]
