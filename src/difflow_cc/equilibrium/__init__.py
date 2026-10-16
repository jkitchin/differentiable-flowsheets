"""Equilibrium models for carbon capture.

This module provides thermodynamic equilibrium models for:
- Adsorption isotherms (Langmuir, Sips, Toth, dual-site Langmuir)
- Vapor-liquid equilibrium for amine-CO2-H2O systems
- H2S-CO2-amine-water speciation (Kent-Eisenberg, MDEA and DEA)
- CO2 solubility correlations

All models are fully JAX-compatible for automatic differentiation.
"""

from difflow_cc.equilibrium.isotherms import (
    # Isotherm functions
    langmuir,
    sips,
    toth,
    dual_site_langmuir,
    # Temperature-dependent versions
    langmuir_T,
    sips_T,
    toth_T,
    dual_site_langmuir_T,
    # Isotherm class
    Isotherm,
    get_isotherm,
    # Working capacity
    working_capacity_PSA,
    working_capacity_TSA,
)

from difflow_cc.equilibrium.vle import (
    # VLE functions
    henry_constant,
    co2_equilibrium_pressure,
    co2_loading,
    # VLE model class
    AmineVLE,
    AmineVLEParams,
)

from difflow_cc.equilibrium.acid_gas import (
    KentEisenbergAmine,
    KE_AMINES,
    amine_molality,
    ke_speciate,
    ke_partial_pressures,
    ke_loading,
)

from difflow_cc.equilibrium.solubility import (
    co2_physical_solubility,
    diffusivity_co2_amine,
)

__all__ = [
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
    "AmineVLEParams",
    # H2S-CO2-amine (Kent-Eisenberg)
    "KentEisenbergAmine",
    "KE_AMINES",
    "amine_molality",
    "ke_speciate",
    "ke_partial_pressures",
    "ke_loading",
    # Solubility
    "co2_physical_solubility",
    "diffusivity_co2_amine",
]
