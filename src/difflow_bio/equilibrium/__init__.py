"""Equilibrium models for biopharmaceutical processes.

This module provides JAX-compatible implementations of
equilibrium and binding models used in chromatography
and other bioseparation processes.

Models implemented:
- Langmuir: Single-site binding
- Langmuir competitive: Multi-component binding
- Steric mass action: Ion exchange with steric effects
- Linear: Linear partitioning

References:
    Carta G, Jungbauer A (2010). Protein Chromatography.
        Wiley-VCH.
    Brooks CA, Cramer SM (1992). AIChE J 38:1969.
        (Steric mass action model)
"""

from difflow_bio.equilibrium.isotherms import (
    langmuir_binding,
    langmuir_competitive,
    steric_mass_action,
    linear_partition,
    langmuir_ph_dependent,
    dynamic_binding_capacity,
    breakthrough_curve,
    column_efficiency,
    van_deemter,
    BindingIsotherm,
    get_binding_isotherm,
)

__all__ = [
    "langmuir_binding",
    "langmuir_competitive",
    "steric_mass_action",
    "linear_partition",
    "langmuir_ph_dependent",
    "dynamic_binding_capacity",
    "breakthrough_curve",
    "column_efficiency",
    "van_deemter",
    "BindingIsotherm",
    "get_binding_isotherm",
]
