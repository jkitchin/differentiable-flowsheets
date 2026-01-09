"""Kinetics module for REE extraction processes.

Provides models for extraction/stripping kinetics including:
- Mass transfer kinetics
- Interfacial reaction kinetics
- Phase transfer rates

All functions are JAX-compatible for automatic differentiation.

References:
    Danesi PR (1984). Solvent Extr Ion Exch 2:29.
        (Mass transfer in SX)
    Musikas C, Hubert H (1987). Ion Exch Solvent Extr 10:1.
        (Kinetics of lanthanide extraction)
"""

from difflow_ree.kinetics.mass_transfer import (
    film_mass_transfer,
    overall_mass_transfer,
    diffusion_coefficient,
    sherwood_correlation,
    mass_transfer_rate,
    enhancement_factor,
    MassTransferModel,
    MassTransferParams,
    get_mass_transfer_model,
)

from difflow_ree.kinetics.extraction_kinetics import (
    forward_extraction_rate,
    reverse_extraction_rate,
    net_extraction_rate,
    approach_to_equilibrium,
    stage_residence_time,
    ExtractionKineticsModel,
    ExtractionKineticsParams,
    get_extraction_kinetics_model,
)

__all__ = [
    # Mass transfer
    "film_mass_transfer",
    "overall_mass_transfer",
    "diffusion_coefficient",
    "sherwood_correlation",
    "mass_transfer_rate",
    "enhancement_factor",
    "MassTransferModel",
    "MassTransferParams",
    "get_mass_transfer_model",
    # Extraction kinetics
    "forward_extraction_rate",
    "reverse_extraction_rate",
    "net_extraction_rate",
    "approach_to_equilibrium",
    "stage_residence_time",
    "ExtractionKineticsModel",
    "ExtractionKineticsParams",
    "get_extraction_kinetics_model",
]
