"""Difflow: Differentiable flowsheet framework for chemical processes.

Core package providing:
- Stream representation and utilities
- Thermodynamic property calculations
- Unit operations (CSTR, Flash, LLE)
- Flowsheet management with recycle solving
- Differentiable solvers

For bio manufacturing operations, install the difflow_bio plugin:
    pip install difflow[bio]
    # or
    from difflow_bio import ContinuousBioreactor, ProteinAChromatography
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

# Visualization (optional - requires plotly)
try:
    from difflow.visualization import (
        FlowsheetGraph,
        Node,
        Edge,
        render_flowsheet,
        show_flowsheet,
        UNIT_STYLES,
        get_unit_style,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

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
    # Visualization (when available)
    "FlowsheetGraph",
    "Node",
    "Edge",
    "render_flowsheet",
    "show_flowsheet",
    "UNIT_STYLES",
    "get_unit_style",
]
