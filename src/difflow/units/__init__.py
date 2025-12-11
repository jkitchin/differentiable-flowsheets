"""Unit operations for difflow.

Core unit operations:
- CSTR: Continuous stirred tank reactor
- PFR: Plug flow reactor
- Flash: Vapor-liquid equilibrium flash
- Mixer/Splitter: Stream combining and splitting
- MultistageCascade: Liquid-liquid extraction

For bio manufacturing operations (bioreactors, centrifuge, filtration,
chromatography), use the difflow_bio plugin:
    from difflow_bio import ContinuousBioreactor, ProteinAChromatography
"""

from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.pfr import PFR, PFRParams, GasPFR, GasPFRParams
from difflow.units.flash import Flash, FlashParams, Mixer, Splitter
from difflow.units.lle import (
    MultistageCascade,
    CascadeParams,
    DifferentialContactor,
    ContactorParams,
    LLEEquilibrium,
    DistributionCoeffs,
    NRTLParams,
    UNIQUACParams,
    nrtl_activity_coefficients,
    uniquac_activity_coefficients,
    get_K_values,
    separation_factor,
    minimum_solvent_ratio,
    stages_for_recovery,
)

__all__ = [
    # CSTR
    "CSTR",
    "CSTRParams",
    # PFR
    "PFR",
    "PFRParams",
    "GasPFR",
    "GasPFRParams",
    # Flash
    "Flash",
    "FlashParams",
    "Mixer",
    "Splitter",
    # LLE
    "MultistageCascade",
    "CascadeParams",
    "DifferentialContactor",
    "ContactorParams",
    "LLEEquilibrium",
    "DistributionCoeffs",
    "NRTLParams",
    "UNIQUACParams",
    "nrtl_activity_coefficients",
    "uniquac_activity_coefficients",
    "get_K_values",
    "separation_factor",
    "minimum_solvent_ratio",
    "stages_for_recovery",
]
