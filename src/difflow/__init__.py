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

import jax
jax.config.update("jax_enable_x64", True)

from importlib import metadata as _md

try:
    __version__ = _md.version("difflow")
except _md.PackageNotFoundError:  # pragma: no cover - not installed
    __version__ = "0.0.0+unknown"

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
from difflow.thermo import IdealThermo, CubicThermo, SpeciesData
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
    track_database_access,
    DatabaseAccessTracker,
)
from difflow.liquid_properties import (
    get_liquid_properties,
    liquid_density,
    liquid_molar_density,
    liquid_viscosity,
    liquid_thermal_conductivity,
    stream_liquid_properties,
)
from difflow.gas_properties import (
    gas_viscosity,
    gas_thermal_conductivity,
    gas_heat_capacity,
    gas_prandtl,
)
from difflow.base_database import BaseDatabase
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
# NASA Glenn (pyglenn) and DWSIM imports are exposed as namespaces, not
# flattened, because their import_species_data / import_critical_props mirror
# the Cantera names. Usage: difflow.pyglenn_import.import_species_data([...]),
# difflow.dwsim_import.import_critical_props([...]). Both underlying tools are
# optional dependencies, imported lazily inside the adapters' functions.
from difflow import fluids
from difflow import heat_transfer
from difflow import gas_properties
from difflow import pyglenn_import
from difflow import dwsim_import
from difflow.params_mixin import ParamsMixin
from difflow.numerics import (
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_power,
    safe_exp,
    smooth_max,
    smooth_min,
    smooth_clamp,
    safe_arccos,
)
from difflow.units.base import (
    UnitBase,
    ReactorBase,
    estimate_volumetric_flow,
    estimate_outlet_composition,
    estimate_adiabatic_temperature,
    estimate_residence_time,
    estimate_cstr_conversion,
    estimate_pfr_conversion,
)
from difflow.units.cstr import CSTR, CSTRParams, CSTRDensityWarning
from difflow.units.pfr import GasPFRBedWarning
from difflow.units.pfr import PFR, PFRParams, GasPFR, GasPFRParams
from difflow.units.fed_batch import (
    FedBatchReactor,
    FedBatchParams,
    SemiBatchReactor,
    batch_time_for_conversion,
    optimal_feed_profile,
)
from difflow.units.flash import (
    Flash,
    FlashParams,
    EOSFlash,
    EOSFlashParams,
    PHFlash,
    Mixer,
    Splitter,
    ComponentSplitter,
    ComponentSplitterParams,
)
from difflow.units.stoichiometric import (
    StoichiometricReactor,
    StoichiometricReactorParams,
)
from difflow.activity import (
    ActivityModel,
    activity_gamma,
    WilsonParams,
    MargulesParams,
    VanLaarParams,
)
from difflow import staged  # noqa: F401  (difflow.staged: McCabe-Thiele, Kremser)
from difflow.units import column_design  # noqa: F401  (difflow.units.column_design: tray/packing hardware)
from difflow.phase_diagrams import (
    bubble_T,
    txy,
    pxy,
    xy_curve,
    find_azeotrope,
    ternary_lle,
    lever_rule,
    tie_line_through,
    difference_point,
    hunter_nash,
)
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
from difflow.units.leaching import (
    Leacher,
    LeacherParams,
    tabulate_underflow,
    washing_recovery,
    washing_stages,
)
from difflow.units.absorption import (
    Absorber,
    AbsorptionParams,
    Stripper,
    minimum_solvent_rate,
    minimum_stripping_gas_rate,
    solute_free_operating_line,
)
from difflow.units.crystallizer import (
    Crystallizer,
    CrystallizerParams,
    Solubility,
    supersaturation,
    crystal_yield,
    crystal_balance,
    msmpr_distribution,
    msmpr_moments,
    msmpr_dominant_size,
    msmpr_mass_median_size,
    msmpr_mass_cv,
    msmpr_from_magma,
    fit_msmpr,
    growth_rate,
    nucleation_rate,
    msmpr_steady_state,
)
from difflow.units.mechanical import (
    Settler,
    SettlerParams,
    CakeFilter,
    CakeFilterParams,
    Cyclone,
    CycloneParams,
    lse_max,
    solids_flux,
    batch_flux_curve,
    hindered_velocity_model,
    thickener_area_coe_clevenger,
    thickener_area_talmage_fitch,
    thickener_area_kynch,
    kynch_construction,
    overflow_rate_area,
    overflow_rate_capture,
    ruth_constants,
    ruth_constant_pressure,
    filtrate_volume,
    filtrate_rate,
    ruth_constant_rate,
    time_to_pressure,
    fit_ruth,
    compressible_cake,
    specific_cake_resistance,
    cake_solids_per_filtrate,
    wash_time,
    cycle_throughput,
    optimal_cycle,
    filter_area_for_cycle,
    cyclone_geometry,
    lapple_turns,
    lapple_cut_size,
    lapple_efficiency,
    overall_efficiency,
    cyclone_pressure_drop,
    dust_laden_gas,
)
from difflow.units.dryer import (
    Dryer,
    DryerParams,
    constant_drying_rate,
    drying_periods,
    drying_rate_curve,
    drying_time,
    dry_basis,
    equilibrium_moisture,
    free_moisture,
    heat_transfer_coefficient_tray,
    solids_moisture,
    water_activity,
    wet_basis,
)
from difflow.units.cooling_tower import (
    CoolingTower,
    CoolingTowerParams,
    maximum_L_over_G,
    merkel_number,
)
from difflow.units.batch_distillation import (
    BatchDistillation,
    BatchDistillationParams,
    batch_cuts,
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
from difflow.units.evaporator import (
    Evaporator,
    EvaporatorParams,
    MultiEffectEvaporator,
    MultiEffectEvaporatorParams,
    MechanicalVaporRecompression,
    MVRParams,
    UnverifiedDataWarning,
    boiling_point_rise,
)
from difflow.units.heat_exchanger import (
    DefaultCpWarning,
    LowCorrectionFactorWarning,
    PhaseChangeWarning,
    Heater,
    HeaterParams,
    Cooler,
    CoolerParams,
    CounterCurrentHX,
    EnthalpyCounterCurrentHX,
    EnthalpyHXParams,
    CoCurrentHX,
    CrossFlowHX,
    HeatExchangerParams,
    ShellAndTubeHX,
    ShellAndTubeHXParams,
    lmtd_correction_factor,
    effectiveness_shell_and_tube,
    log_mean_temperature_difference,
    design_heat_exchanger,
    size_heat_exchanger,
)
from difflow.units.pipe import Pipe, PipeCavitationWarning, PipeParams
from difflow.units.pump import (
    CentrifugalPump,
    CentrifugalPumpParams,
    fit_pump_curve,
    npsh_available,
    operating_point,
    OperatingPointWarning,
    pumps_in_parallel,
    pumps_in_series,
    system_curve,
)
from difflow import shell_and_tube  # library (not a palette unit): Kern design, see docs
from difflow import fouling  # library: Ebert-Panchal threshold fouling model
from difflow.units.eos_units import (
    Turboexpander,
    TurboexpanderParams,
    Compressor,
    CompressorParams,
    JTValve,
    JTValveParams,
    ComponentSeparator,
    ComponentSeparatorParams,
    flash_PH_eos,
    flash_PS_eos,
    EOSUnitConvergenceWarning,
)
from difflow.units.gas_turbine import (
    Combustor,
    CombustorParams,
    GasCompressor,
    GasCompressorParams,
    GasTurbine,
    GasTurbineParams,
    BraytonCycleParams,
    brayton_cycle,
    make_cycle_thermo,
)
from difflow.combustion import IdealGasThermo
from difflow.flowsheet import (
    Flowsheet,
    Unit,
    DegreesOfFreedom,
    create_objective,
    ConvergenceWarning,
    ConvergenceError,
    TearToleranceWarning,
)
from difflow.report import (
    ConvergenceInfo,
    OptimizationReport,
    Report,
    ReportDiff,
    build_optimization_report,
    build_report,
    diff_reports,
    report,
    to_html,
    to_json,
    to_latex,
    to_markdown,
)
from difflow.eo_solver import (
    EOSolver,
    EOSolveResult,
    EOStateLayout,
    solve_residual_system,
)
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
    StreamEdge,
    find_cycles,
    select_tear_streams,
    calculation_order,
    analyze_tears,
    TearAnalysis,
    CycleEnumerationWarning,
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

# Parameter estimation submodule
from difflow import estimation

# Data reconciliation submodule
from difflow import reconciliation

# Delta-base planning submodule
from difflow import planning

# Moving-horizon estimation submodule (#201)
from difflow import mhe

# Steady-state operability screening submodule (#199)
from difflow import operability

# Flexibility analysis submodule (#198)
from difflow import flexibility

# Two-stage stochastic programming over a scenario sample
from difflow import stochastic

# External solver bridges: pounce / discopt (#203)
from difflow import solvers

# Declarative kinetics
from difflow.kinetics import (
    KineticsSpecError,
    ReactionSet,
    mass_action_kinetics,
)

# Flowsheet serialization, code generation and publishing
from difflow import codegen, gui, publish as publish_module, serialize
from difflow.publish import SweepAxis, SweepResult, publish, sweep

# Operation catalog
from difflow.catalog import (
    OperationSchema,
    ParameterSpec,
    PortSpec,
    catalog,
    describe_operation,
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
    # Params Mixin
    "ParamsMixin",
    # Numerical utilities
    "safe_divide",
    "safe_log",
    "safe_sqrt",
    "safe_power",
    "safe_exp",
    "smooth_max",
    "smooth_min",
    "smooth_clamp",
    "safe_arccos",
    # Unit Base Classes and Helpers
    "UnitBase",
    "ReactorBase",
    "estimate_volumetric_flow",
    "estimate_outlet_composition",
    "estimate_adiabatic_temperature",
    "estimate_residence_time",
    "estimate_cstr_conversion",
    "estimate_pfr_conversion",
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
    "CubicThermo",
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
    "track_database_access",
    "DatabaseAccessTracker",
    "BaseDatabase",
    # Liquid properties
    "get_liquid_properties",
    "liquid_density",
    "liquid_molar_density",
    "liquid_viscosity",
    "liquid_thermal_conductivity",
    "stream_liquid_properties",
    "gas_viscosity",
    "gas_thermal_conductivity",
    "gas_heat_capacity",
    "gas_prandtl",
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
    # NASA Glenn (pyglenn) Import (namespace: difflow.pyglenn_import)
    "pyglenn_import",
    # DWSIM Import (namespace: difflow.dwsim_import)
    "dwsim_import",
    # Unit operations - CSTR
    "CSTR",
    "CSTRParams",
    "CSTRDensityWarning",
    "GasPFRBedWarning",
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
    "EOSFlash",
    "EOSFlashParams",
    "PHFlash",
    "Mixer",
    "Splitter",
    "ComponentSplitter",
    "ComponentSplitterParams",
    "StoichiometricReactor",
    "StoichiometricReactorParams",
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
    "ActivityModel",
    "activity_gamma",
    "WilsonParams",
    "MargulesParams",
    "VanLaarParams",
    "bubble_T",
    "txy",
    "pxy",
    "xy_curve",
    "find_azeotrope",
    "ternary_lle",
    "lever_rule",
    "tie_line_through",
    "difference_point",
    "hunter_nash",
    "uniquac_activity_coefficients",
    "get_K_values",
    "separation_factor",
    "minimum_solvent_ratio",
    "stages_for_recovery",
    # Leaching and washing
    "Leacher",
    "LeacherParams",
    "tabulate_underflow",
    "washing_recovery",
    "washing_stages",
    # Unit operations - Absorption and stripping
    "Absorber",
    "AbsorptionParams",
    "Stripper",
    "minimum_solvent_rate",
    "minimum_stripping_gas_rate",
    "solute_free_operating_line",
    # Unit operations - Humidification and cooling towers
    "CoolingTower",
    "CoolingTowerParams",
    "maximum_L_over_G",
    "merkel_number",
    # Mechanical separations
    "Settler",
    "SettlerParams",
    "CakeFilter",
    "CakeFilterParams",
    "Cyclone",
    "CycloneParams",
    "lse_max",
    "solids_flux",
    "batch_flux_curve",
    "hindered_velocity_model",
    "thickener_area_coe_clevenger",
    "thickener_area_talmage_fitch",
    "thickener_area_kynch",
    "kynch_construction",
    "overflow_rate_area",
    "overflow_rate_capture",
    "ruth_constants",
    "ruth_constant_pressure",
    "filtrate_volume",
    "filtrate_rate",
    "ruth_constant_rate",
    "time_to_pressure",
    "fit_ruth",
    "compressible_cake",
    "specific_cake_resistance",
    "cake_solids_per_filtrate",
    "wash_time",
    "cycle_throughput",
    "optimal_cycle",
    "filter_area_for_cycle",
    "cyclone_geometry",
    "lapple_turns",
    "lapple_cut_size",
    "lapple_efficiency",
    "overall_efficiency",
    "cyclone_pressure_drop",
    "dust_laden_gas",
    # Unit operations - Drying
    "Crystallizer",
    "CrystallizerParams",
    "Solubility",
    "supersaturation",
    "crystal_yield",
    "crystal_balance",
    "msmpr_distribution",
    "msmpr_moments",
    "msmpr_dominant_size",
    "msmpr_mass_median_size",
    "msmpr_mass_cv",
    "msmpr_from_magma",
    "fit_msmpr",
    "growth_rate",
    "nucleation_rate",
    "msmpr_steady_state",
    "Dryer",
    "DryerParams",
    "constant_drying_rate",
    "dry_basis",
    "drying_periods",
    "drying_rate_curve",
    "drying_time",
    "equilibrium_moisture",
    "free_moisture",
    "heat_transfer_coefficient_tray",
    "solids_moisture",
    "water_activity",
    "wet_basis",
    # Unit operations - Distillation
    "ShortcutColumn",
    "ShortcutColumnParams",
    "BatchDistillation",
    "BatchDistillationParams",
    "batch_cuts",
    "DistillationColumn",
    "DistillationColumnParams",
    "fenske_stages",
    "minimum_reflux_ratio",
    "gilliland_stages",
    "column_diameter",
    # Unit operations - Evaporators
    "Evaporator",
    "EvaporatorParams",
    "MultiEffectEvaporator",
    "MultiEffectEvaporatorParams",
    "MechanicalVaporRecompression",
    "MVRParams",
    "UnverifiedDataWarning",
    "boiling_point_rise",
    # Unit operations - Heat Exchangers
    "DefaultCpWarning",
    "LowCorrectionFactorWarning",
    "PhaseChangeWarning",
    "Heater",
    "HeaterParams",
    "Cooler",
    "CoolerParams",
    "CounterCurrentHX",
    "EnthalpyCounterCurrentHX",
    "EnthalpyHXParams",
    "CoCurrentHX",
    "CrossFlowHX",
    "HeatExchangerParams",
    "ShellAndTubeHX",
    "ShellAndTubeHXParams",
    "lmtd_correction_factor",
    "effectiveness_shell_and_tube",
    "log_mean_temperature_difference",
    "design_heat_exchanger",
    "size_heat_exchanger",
    # EOS-consistent process units
    "Turboexpander",
    "TurboexpanderParams",
    "Compressor",
    "CompressorParams",
    "JTValve",
    "Pipe",
    "PipeCavitationWarning",
    "PipeParams",
    "CentrifugalPump",
    "CentrifugalPumpParams",
    "fit_pump_curve",
    "npsh_available",
    "operating_point",
    "OperatingPointWarning",
    "pumps_in_parallel",
    "pumps_in_series",
    "system_curve",
    "JTValveParams",
    "ComponentSeparator",
    "ComponentSeparatorParams",
    "flash_PH_eos",
    "flash_PS_eos",
    "EOSUnitConvergenceWarning",
    # Combustion / gas-turbine units
    "Combustor",
    "CombustorParams",
    "GasCompressor",
    "GasCompressorParams",
    "GasTurbine",
    "GasTurbineParams",
    "BraytonCycleParams",
    "brayton_cycle",
    "make_cycle_thermo",
    "IdealGasThermo",
    # Flowsheet
    "Flowsheet",
    "Unit",
    "create_objective",
    "ConvergenceWarning",
    "ConvergenceError",
    "TearToleranceWarning",
    # Reporting
    "Report",
    "OptimizationReport",
    "ConvergenceInfo",
    "ReportDiff",
    "build_report",
    "build_optimization_report",
    "diff_reports",
    "report",
    "to_markdown",
    "to_json",
    "to_html",
    "to_latex",
    "__version__",
    # EO Solver
    "EOSolver",
    "EOSolveResult",
    "EOStateLayout",
    "solve_residual_system",
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
    "StreamEdge",
    "find_cycles",
    "select_tear_streams",
    "calculation_order",
    "analyze_tears",
    "TearAnalysis",
    "DegreesOfFreedom",
    "CycleEnumerationWarning",
    # Plugin infrastructure
    "registry",
    "load_plugins",
    "discover_plugins",
    "register_operation",
    "UnitOperation",
    "OperationRegistry",
    # Parameter estimation
    "estimation",
    # Data reconciliation
    "reconciliation",
    # Delta-base planning
    "planning",
    # Moving-horizon estimation
    "mhe",
    # Operability screening
    "operability",
    # Flexibility analysis
    "flexibility",
    # Stochastic programming
    "stochastic",
    # External solver bridges
    "solvers",
    # Declarative kinetics
    "mass_action_kinetics",
    "ReactionSet",
    "KineticsSpecError",
    # Flowsheet serialization, code generation and publishing
    "serialize",
    "codegen",
    "publish",
    "sweep",
    "SweepAxis",
    "SweepResult",
    "gui",
    # Operation catalog
    "catalog",
    "describe_operation",
    "OperationSchema",
    "ParameterSpec",
    "PortSpec",
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


# ---------------------------------------------------------------------
# Populate the operation registry with the core units.
#
# The registry was previously filled only by plugins, so it held the
# bio, carbon-capture, gas and REE units but none of the reactors,
# separators, columns or exchangers. This runs last because it reads
# __all__ to discover them.
# ---------------------------------------------------------------------

from difflow.catalog import register_core_operations as _register_core_operations

_register_core_operations()
