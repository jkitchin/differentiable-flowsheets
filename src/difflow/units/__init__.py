"""Unit operations for difflow.

Core unit operations:
- CSTR: Continuous stirred tank reactor
- PFR: Plug flow reactor
- FedBatchReactor: Fed-batch/semi-batch reactor
- Flash: Vapor-liquid equilibrium flash
- Mixer/Splitter: Stream combining and splitting
- MultistageCascade: Liquid-liquid extraction
- ShortcutColumn/DistillationColumn: Distillation
- Heater/Cooler: Single-stream heat exchange with utility
- Evaporator/MultiEffectEvaporator/MechanicalVaporRecompression: concentration of
  a non-volatile solute with boiling-point rise
- CounterCurrentHX/CoCurrentHX: Two-stream heat exchangers
- Pipe: Incompressible Darcy-Weisbach pipe with fittings (see difflow.fluids)
- CentrifugalPump: pump curves, affinity laws, NPSH, system curve, operating point

For bio manufacturing operations (bioreactors, centrifuge, filtration,
chromatography), use the difflow_bio plugin:
    from difflow_bio import ContinuousBioreactor, ProteinAChromatography
"""

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
from difflow.units.cstr import CSTR, CSTRParams
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
    rayleigh,
    rayleigh_multicomponent,
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
    HeatExchangerParams,
    ShellAndTubeHX,
    ShellAndTubeHXParams,
    lmtd_correction_factor,
    log_mean_temperature_difference,
    effectiveness_counter_current,
    effectiveness_co_current,
    effectiveness_shell_and_tube,
    design_heat_exchanger,
    size_heat_exchanger,
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
    water_saturation_pressure,
    water_saturation_temperature,
    water_latent_heat,
    water_liquid_enthalpy,
    water_vapor_enthalpy,
)
from difflow.units.eos_units import (
    Turboexpander,
    TurboexpanderParams,
    Compressor,
    CompressorParams,
    JTValve,
    JTValveParams,
    ComponentSeparator,
    ComponentSeparatorParams,
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
from difflow.units.pipe import Pipe, PipeParams
from difflow.units.pump import (
    CentrifugalPump,
    CentrifugalPumpParams,
    fit_pump_curve,
    npsh_available,
    operating_point,
    pumps_in_parallel,
    pumps_in_series,
    system_curve,
)

__all__ = [
    # Base classes and helpers
    "UnitBase",
    "ReactorBase",
    "estimate_volumetric_flow",
    "estimate_outlet_composition",
    "estimate_adiabatic_temperature",
    "estimate_residence_time",
    "estimate_cstr_conversion",
    "estimate_pfr_conversion",
    # CSTR
    "CSTR",
    "CSTRParams",
    # PFR
    "PFR",
    "PFRParams",
    "GasPFR",
    "GasPFRParams",
    # Fed-batch
    "FedBatchReactor",
    "FedBatchParams",
    "SemiBatchReactor",
    "batch_time_for_conversion",
    "optimal_feed_profile",
    # Flash
    "Flash",
    "FlashParams",
    "EOSFlash",
    "EOSFlashParams",
    "PHFlash",
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
    # Leaching and washing
    "Leacher",
    "LeacherParams",
    "tabulate_underflow",
    "washing_recovery",
    "washing_stages",
    # Absorption and stripping
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
    # Distillation
    "ShortcutColumn",
    "ShortcutColumnParams",
    "BatchDistillation",
    "BatchDistillationParams",
    "batch_cuts",
    "rayleigh",
    "rayleigh_multicomponent",
    "DistillationColumn",
    "DistillationColumnParams",
    "fenske_stages",
    "minimum_reflux_ratio",
    "gilliland_stages",
    "column_diameter",
    # Evaporators
    "Evaporator",
    "EvaporatorParams",
    "MultiEffectEvaporator",
    "MultiEffectEvaporatorParams",
    "MechanicalVaporRecompression",
    "MVRParams",
    "UnverifiedDataWarning",
    "boiling_point_rise",
    "water_saturation_pressure",
    "water_saturation_temperature",
    "water_latent_heat",
    "water_liquid_enthalpy",
    "water_vapor_enthalpy",
    # Heat Exchangers
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
    "HeatExchangerParams",
    "ShellAndTubeHX",
    "ShellAndTubeHXParams",
    "lmtd_correction_factor",
    "effectiveness_shell_and_tube",
    "log_mean_temperature_difference",
    "effectiveness_counter_current",
    "effectiveness_co_current",
    "design_heat_exchanger",
    "size_heat_exchanger",
    # Liquid pipe flow
    "Pipe",
    "PipeParams",
    "CentrifugalPump",
    "CentrifugalPumpParams",
    "fit_pump_curve",
    "npsh_available",
    "operating_point",
    "pumps_in_parallel",
    "pumps_in_series",
    "system_curve",
    # EOS-consistent process units
    "Turboexpander",
    "TurboexpanderParams",
    "Compressor",
    "CompressorParams",
    "JTValve",
    "JTValveParams",
    "ComponentSeparator",
    "ComponentSeparatorParams",
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
]
