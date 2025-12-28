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
    get_species_data,
    get_critical_props,
    get_species_info,
    list_species,
    get_alkanes,
    get_btex,
    get_common_solvents,
)
from difflow.uncertainty import (
    linear_propagation,
    monte_carlo_propagation,
    sensitivity_analysis,
    sobol_indices,
    propagate_covariance,
)
from difflow.cantera_import import (
    import_species_data,
    import_critical_props,
    import_reactions,
    load_mechanism,
    list_available_species,
    list_available_reactions,
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
from difflow.units.heat_exchanger import (
    Heater,
    HeaterParams,
    Cooler,
    CoolerParams,
    CounterCurrentHX,
    CoCurrentHX,
    HeatExchangerParams,
    log_mean_temperature_difference,
    design_heat_exchanger,
    size_heat_exchanger,
)
from difflow.flowsheet import Flowsheet, Unit
from difflow.initialization import (
    AndersonAccelerator,
    wegstein_acceleration,
    anderson_acceleration_step,
    Initializable,
    InitializationResult,
    estimate_cstr_conversion,
    estimate_outlet_temperature,
    estimate_flash_split,
    FlowsheetGraph,
    find_cycles,
    select_tear_streams,
)

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

# Visualization submodule (optional - requires ipycytoscape)
try:
    from difflow import visualization
    from difflow.visualization import visualize_flowsheet, FlowsheetVisualizer
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False
    visualization = None
    visualize_flowsheet = None
    FlowsheetVisualizer = None

# Dynamic modeling submodule
from difflow import dynamic
from difflow.dynamic import (
    # State management
    StateVar,
    StateSpec,
    StateVector,
    molar_states,
    thermal_state,
    reactor_states,
    # Base classes
    DynamicUnit,
    DynamicUnitBase,
    DynamicCSTR,
    DynamicTank,
    # Integrators
    integrate,
    integrate_unit,
    integrate_rk4,
    integrate_rk45,
    IntegrationResult,
    Trajectory,
    # Flowsheet
    DynamicFlowsheet,
    DynamicFlowsheetResult,
    # DAE
    DAEUnit,
    DAEUnitBase,
    AlgebraicVar,
    AlgebraicSpec,
    integrate_dae,
    DAEResult,
    DynamicFlashDrum,
)

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
    "get_species_data",
    "get_critical_props",
    "get_species_info",
    "list_species",
    "get_alkanes",
    "get_btex",
    "get_common_solvents",
    # Uncertainty Propagation
    "linear_propagation",
    "monte_carlo_propagation",
    "sensitivity_analysis",
    "sobol_indices",
    "propagate_covariance",
    # Cantera Import
    "import_species_data",
    "import_critical_props",
    "import_reactions",
    "load_mechanism",
    "list_available_species",
    "list_available_reactions",
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
    # Unit operations - Heat Exchangers
    "Heater",
    "HeaterParams",
    "Cooler",
    "CoolerParams",
    "CounterCurrentHX",
    "CoCurrentHX",
    "HeatExchangerParams",
    "log_mean_temperature_difference",
    "design_heat_exchanger",
    "size_heat_exchanger",
    # Flowsheet
    "Flowsheet",
    "Unit",
    # Initialization & Acceleration
    "AndersonAccelerator",
    "wegstein_acceleration",
    "anderson_acceleration_step",
    "Initializable",
    "InitializationResult",
    "estimate_cstr_conversion",
    "estimate_outlet_temperature",
    "estimate_flash_split",
    "FlowsheetGraph",
    "find_cycles",
    "select_tear_streams",
    # Plugin infrastructure
    "registry",
    "load_plugins",
    "discover_plugins",
    "register_operation",
    "UnitOperation",
    "OperationRegistry",
    # Economics
    "economics",
    # Visualization (optional)
    "visualization",
    "visualize_flowsheet",
    "FlowsheetVisualizer",
    # Dynamic modeling
    "dynamic",
    "StateVar",
    "StateSpec",
    "StateVector",
    "molar_states",
    "thermal_state",
    "reactor_states",
    "DynamicUnit",
    "DynamicUnitBase",
    "DynamicCSTR",
    "DynamicTank",
    "integrate",
    "integrate_unit",
    "integrate_rk4",
    "integrate_rk45",
    "IntegrationResult",
    "Trajectory",
    "DynamicFlowsheet",
    "DynamicFlowsheetResult",
    # DAE
    "DAEUnit",
    "DAEUnitBase",
    "AlgebraicVar",
    "AlgebraicSpec",
    "integrate_dae",
    "DAEResult",
    "DynamicFlashDrum",
]
