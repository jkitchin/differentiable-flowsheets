"""Difflow: Differentiable flowsheet framework for chemical processes."""

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

# Bio manufacturing operations
from difflow.units.bioreactors import (
    ContinuousBioreactor,
    FedBatchBioreactor,
    BioreactorParams,
    FedBatchParams,
    monod_kinetics,
    substrate_inhibition_kinetics,
    product_inhibition_kinetics,
    contois_kinetics,
    dilution_rate,
    residence_time,
    optimal_dilution_rate,
)
from difflow.units.centrifuge import (
    Centrifuge,
    CentrifugeParams,
    DiscStackCentrifuge,
    DiscStackParams,
    stokes_velocity,
    critical_particle_diameter,
    disc_stack_sigma,
    tubular_bowl_sigma,
    centrifuge_scale_up,
    g_force,
)
from difflow.units.filtration import (
    Ultrafiltration,
    UltrafiltrationParams,
    Diafiltration,
    DiafiltrationParams,
    TFF,
    concentration_polarization,
    gel_layer_flux,
    diavolumes_required,
    rejection_from_mw,
)
from difflow.units.chromatography import (
    ProteinAChromatography,
    ProteinAParams,
    IonExchangeChromatography,
    IEXParams,
    SizeExclusionChromatography,
    SECParams,
    langmuir_isotherm,
    linear_isotherm,
    langmuir_freundlich_isotherm,
    dynamic_binding_capacity,
    column_productivity,
    resolution,
    plate_count,
    hetp,
)

from difflow.flowsheet import Flowsheet, Unit
from difflow.solvers import fixed_point_solve, newton_solve, rachford_rice

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
    # Unit operations - Bioreactors
    "ContinuousBioreactor",
    "FedBatchBioreactor",
    "BioreactorParams",
    "FedBatchParams",
    "monod_kinetics",
    "substrate_inhibition_kinetics",
    "product_inhibition_kinetics",
    "contois_kinetics",
    "dilution_rate",
    "residence_time",
    "optimal_dilution_rate",
    # Unit operations - Centrifuge
    "Centrifuge",
    "CentrifugeParams",
    "DiscStackCentrifuge",
    "DiscStackParams",
    "stokes_velocity",
    "critical_particle_diameter",
    "disc_stack_sigma",
    "tubular_bowl_sigma",
    "centrifuge_scale_up",
    "g_force",
    # Unit operations - Filtration
    "Ultrafiltration",
    "UltrafiltrationParams",
    "Diafiltration",
    "DiafiltrationParams",
    "TFF",
    "concentration_polarization",
    "gel_layer_flux",
    "diavolumes_required",
    "rejection_from_mw",
    # Unit operations - Chromatography
    "ProteinAChromatography",
    "ProteinAParams",
    "IonExchangeChromatography",
    "IEXParams",
    "SizeExclusionChromatography",
    "SECParams",
    "langmuir_isotherm",
    "linear_isotherm",
    "langmuir_freundlich_isotherm",
    "dynamic_binding_capacity",
    "column_productivity",
    "resolution",
    "plate_count",
    "hetp",
    # Flowsheet
    "Flowsheet",
    "Unit",
    # Solvers
    "fixed_point_solve",
    "newton_solve",
    "rachford_rice",
]
