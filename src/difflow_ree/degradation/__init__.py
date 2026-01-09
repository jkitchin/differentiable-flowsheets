"""Degradation module for REE extraction solvents.

Provides models for extractant degradation including:
- Oxidative degradation
- Hydrolytic degradation
- Solubility losses
- Third phase formation

All functions are JAX-compatible for automatic differentiation.

References:
    Ritcey GM, Ashbrook AW (1984). Solvent Extraction.
    Paiva AP (1999). Hydrometallurgy 53:131.
        (Degradation of organophosphorus extractants)
"""

from difflow_ree.degradation.extractant_degradation import (
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

__all__ = [
    "oxidative_degradation_rate",
    "hydrolytic_degradation_rate",
    "solubility_loss_rate",
    "total_degradation_rate",
    "extractant_lifetime",
    "makeup_rate",
    "ExtractantDegradationModel",
    "ExtractantDegradationParams",
    "get_degradation_model",
]
