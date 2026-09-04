"""Equilibrium models for REE solvent extraction.

This module provides:
- pH-dependent distribution coefficient models
- Extractant loading/saturation effects
- Aqueous phase speciation
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
]
