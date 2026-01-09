"""Kinetics models for carbon capture.

This module provides reaction kinetics and mass transfer
correlations for amine-based CO2 absorption.

Models include:
- Zwitterion mechanism for primary/secondary amines
- Base-catalyzed hydration for tertiary amines
- Enhancement factor calculations
- Mass transfer correlations for packed columns
"""

from difflow_cc.kinetics.amine_kinetics import (
    reaction_rate_constant,
    enhancement_factor,
    hatta_number,
    pseudo_first_order_rate,
)

from difflow_cc.kinetics.mass_transfer import (
    gas_film_coefficient,
    liquid_film_coefficient,
    overall_mass_transfer,
    interfacial_area,
)

__all__ = [
    # Reaction kinetics
    "reaction_rate_constant",
    "enhancement_factor",
    "hatta_number",
    "pseudo_first_order_rate",
    # Mass transfer
    "gas_film_coefficient",
    "liquid_film_coefficient",
    "overall_mass_transfer",
    "interfacial_area",
]
