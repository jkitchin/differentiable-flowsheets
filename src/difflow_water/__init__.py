"""difflow_water: water treatment, desalination and networks for difflow.

Phase 1: seawater properties, a seawater reverse-osmosis stage, a pressure
exchanger, water pumps, WaterTAP-compatible levelized cost of water, and the
seawater RO + pressure exchanger flowsheet. Phase 1b: a Pitzer activity
model, seawater/brine speciation and mineral saturation indices (scaling),
RO trains and pressure-exchanger leakage. Phase 2: water distribution
networks (:mod:`difflow_water.network`), solved equation-oriented and
differentiably, cross-checked against EPANET 2.2. Phase 3a: thermal
desalination (:mod:`difflow_water.thermal`): mechanical vapor compression
(cross-checked against WaterTAP) and multi-effect distillation. Phase 3b:
conventional treatment (:mod:`difflow_water.treatment`): GAC (WaterTAP's
CPHSDM), disinfection, sedimentation and rapid filtration.
The plan for the rest (distribution networks, conventional treatment,
thermal desalination, wastewater) is ``src/difflow_water/DESIGN.md``.

Every equation and coefficient is tied to a reference whose verification
status is recorded in ``docs/unit-operations-water.md``. WaterTAP is the
benchmark: ``benchmarks/watertap/`` runs the same cases in WaterTAP.

Example:
    >>> import difflow_water as dw
    >>> feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035)
    >>> ro = dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=50.0)
    >>> r = dw.swro_performance(feed, ro, ro_pressure=65e5)
    >>> 0 < float(r["recovery"]) < 1
    True
"""

__version__ = "0.7.0"

from difflow_water import chemistry, network, thermal, treatment
from difflow_water.chemistry import pitzer, seawater, speciation
from difflow_water.chemistry.speciation import (
    reference_seawater,
    seawater_feed,
    speciate,
)
from difflow_water.economics import (
    CostingParams,
    capital_recovery_factor,
    cost_hp_pump,
    cost_lp_pump,
    cost_pressure_exchanger,
    cost_ro,
    lcow,
)
from difflow_water.flowsheets import build_swro_flowsheet, swro_performance
from difflow_water.membranes import (
    PressureExchanger,
    PressureExchangerParams,
    ROStage,
    ROStageParams,
    ROTrain,
    ROTrainParams,
    local_flux,
)
from difflow_water.pumps import WaterPump, WaterPumpParams
from difflow_water.streams import (
    SALT,
    WATER,
    combine_seawater,
    concentration_mg_per_L,
    from_watertap_state,
    make_seawater_stream,
    mass_flow,
    mass_flows,
    salinity,
    stream_density,
    to_watertap_state,
    volumetric_flow,
)
from difflow_water.thermal import (
    MEDDesalinator,
    MEDParams,
    MVCCostingParams,
    MVCDesalinator,
    MVCParams,
    med_performance,
    mvc_performance,
)
from difflow_water.treatment import (
    Disinfection,
    DisinfectionParams,
    Filtration,
    FiltrationParams,
    GACAdsorber,
    GACParams,
    Sedimentation,
    SedimentationParams,
)

__all__ = [
    "__version__",
    # chemistry
    "chemistry", "network", "seawater", "pitzer", "speciation", "speciate", "reference_seawater",
    "seawater_feed",
    # streams
    "WATER", "SALT", "make_seawater_stream", "mass_flow", "mass_flows", "salinity",
    "stream_density", "volumetric_flow", "concentration_mg_per_L", "combine_seawater",
    "to_watertap_state", "from_watertap_state",
    # units
    "ROStage", "ROStageParams", "local_flux", "ROTrain", "ROTrainParams",
    "PressureExchanger", "PressureExchangerParams",
    "WaterPump", "WaterPumpParams",
    "MVCDesalinator", "MVCParams", "MVCCostingParams", "mvc_performance",
    "MEDDesalinator", "MEDParams", "med_performance", "thermal",
    "GACAdsorber", "GACParams", "Disinfection", "DisinfectionParams",
    "Sedimentation", "SedimentationParams", "Filtration", "FiltrationParams", "treatment",
    # economics
    "CostingParams", "capital_recovery_factor", "cost_ro", "cost_hp_pump", "cost_lp_pump",
    "cost_pressure_exchanger", "lcow",
    # flowsheets
    "build_swro_flowsheet", "swro_performance",
]


def register(registry):
    """Register difflow_water operations with difflow.

    Called by ``difflow.plugins.load_plugins()`` through the
    ``difflow.plugins`` entry point.

    Args:
        registry: difflow ``OperationRegistry``.
    """
    for name, cls, description in [
        ("ROStage", ROStage,
         "Seawater reverse-osmosis stage (solution-diffusion, film theory, plug flow)"),
        ("ROTrain", ROTrain,
         "Concentrate-staged train of seawater RO stages with interstage boosters"),
        ("PressureExchanger", PressureExchanger,
         "Isobaric pressure exchanger with feed bypass outlet"),
        ("WaterPump", WaterPump,
         "Isothermal water pump with mechanical efficiency"),
        ("MVCDesalinator", MVCDesalinator,
         "Single-stage mechanical vapor compression plant (WaterTAP's MVC layout)"),
        ("MEDDesalinator", MEDDesalinator,
         "Parallel-feed multi-effect distillation plant"),
        ("GACAdsorber", GACAdsorber,
         "Granular activated carbon contactor (constant-pattern HSDM, WaterTAP's GAC)"),
        ("Disinfection", Disinfection,
         "Disinfection contact tank: Chick-Watson kinetics in CSTRs in series"),
        ("Sedimentation", Sedimentation,
         "Ideal settling basin: overflow-rate removal of particle classes"),
        ("Filtration", Filtration,
         "Rapid granular filter: Yao colloid filtration, Carman-Kozeny head loss"),
    ]:
        registry.register(name=name, cls=cls, category="water",
                          description=description, plugin="difflow_water")
