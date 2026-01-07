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
- CounterCurrentHX/CoCurrentHX: Two-stream heat exchangers

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
    effectiveness_counter_current,
    effectiveness_co_current,
    design_heat_exchanger,
    size_heat_exchanger,
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
    # Distillation
    "ShortcutColumn",
    "ShortcutColumnParams",
    "DistillationColumn",
    "DistillationColumnParams",
    "fenske_stages",
    "minimum_reflux_ratio",
    "gilliland_stages",
    "column_diameter",
    # Heat Exchangers
    "Heater",
    "HeaterParams",
    "Cooler",
    "CoolerParams",
    "CounterCurrentHX",
    "CoCurrentHX",
    "HeatExchangerParams",
    "log_mean_temperature_difference",
    "effectiveness_counter_current",
    "effectiveness_co_current",
    "design_heat_exchanger",
    "size_heat_exchanger",
]
