"""Difflow: Differentiable flowsheet framework for chemical processes.

Core package providing:
- Stream representation and utilities
- Thermodynamic property calculations
- Unit operations (CSTR, PFR, Flash, LLE)
- Flowsheet management with recycle solving
- Differentiable solvers
- Technoeconomic analysis (TEA)

For bio manufacturing operations, install the difflow_bio plugin:
    pip install difflow[bio]
    # or
    from difflow_bio import ContinuousBioreactor, ProteinAChromatography

All models support automatic differentiation for gradient-based optimization.
"""

from difflow.streams import (
    Stream,
    make_stream,
    combine_streams,
    get_flows,
    get_species,
    total_flow,
    mole_fractions,
    scale_stream,
)
from difflow.thermo import IdealThermo, SpeciesData
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
from difflow.flowsheet import Flowsheet, Unit
from difflow.solvers import fixed_point_solve, newton_solve, rachford_rice

# Plugin infrastructure
from difflow.plugins import (
    registry,
    load_plugins,
    discover_plugins,
    register_operation,
    UnitOperation,
    OperationRegistry,
)

# Economics submodule
from difflow import economics

__all__ = [
    # Streams
    "Stream",
    "make_stream",
    "combine_streams",
    "get_flows",
    "get_species",
    "total_flow",
    "mole_fractions",
    "scale_stream",
    # Thermodynamics
    "IdealThermo",
    "SpeciesData",
    # Unit operations - CSTR
    "CSTR",
    "CSTRParams",
    # Unit operations - PFR
    "PFR",
    "PFRParams",
    "GasPFR",
    "GasPFRParams",
    # Unit operations - Flash
    "Flash",
    "FlashParams",
    "Mixer",
    "Splitter",
    # Unit operations - LLE
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
    # Flowsheet
    "Flowsheet",
    "Unit",
    # Solvers
    "fixed_point_solve",
    "newton_solve",
    "rachford_rice",
    # Plugin infrastructure
    "registry",
    "load_plugins",
    "discover_plugins",
    "register_operation",
    "UnitOperation",
    "OperationRegistry",
    # Economics
    "economics",
]
