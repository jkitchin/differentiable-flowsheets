"""Difflow: Differentiable flowsheet framework for chemical processes.

Core package providing:
- Stream representation and utilities
- Thermodynamic property calculations (ideal + cubic EOS)
- Unit operations (CSTR, PFR, Fed-batch, Flash, LLE, Distillation)
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
from difflow.eos import (
    PengRobinson,
    SRK,
    CriticalProperties,
    EOSParams,
    flash_TP_eos,
)
from difflow.database import (
    get_species,
    get_critical_props,
    get_species_info,
    list_species,
    get_alkanes,
    get_btex,
    get_common_solvents,
)
from difflow.units.cstr import CSTR, CSTRParams
from difflow.units.pfr import PFR, PFRParams, GasPFR, GasPFRParams
from difflow.units.fed_batch import (
    FedBatchReactor,
    FedBatchParams,
    SemiBatchReactor,
    batch_time_for_conversion,
    optimal_feed_profile,
)
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
from difflow.units.distillation import (
    ShortcutColumn,
    ShortcutColumnParams,
    DistillationColumn,
    DistillationColumnParams,
    fenske_stages,
    minimum_reflux_ratio,
    gilliland_stages,
    column_diameter,
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
    # Thermodynamics - Ideal
    "IdealThermo",
    "SpeciesData",
    # Thermodynamics - Equations of State
    "PengRobinson",
    "SRK",
    "CriticalProperties",
    "EOSParams",
    "flash_TP_eos",
    # Property Database
    "get_species",
    "get_critical_props",
    "get_species_info",
    "list_species",
    "get_alkanes",
    "get_btex",
    "get_common_solvents",
    # Unit operations - CSTR
    "CSTR",
    "CSTRParams",
    # Unit operations - PFR
    "PFR",
    "PFRParams",
    "GasPFR",
    "GasPFRParams",
    # Unit operations - Fed-batch
    "FedBatchReactor",
    "FedBatchParams",
    "SemiBatchReactor",
    "batch_time_for_conversion",
    "optimal_feed_profile",
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
    # Unit operations - Distillation
    "ShortcutColumn",
    "ShortcutColumnParams",
    "DistillationColumn",
    "DistillationColumnParams",
    "fenske_stages",
    "minimum_reflux_ratio",
    "gilliland_stages",
    "column_diameter",
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
