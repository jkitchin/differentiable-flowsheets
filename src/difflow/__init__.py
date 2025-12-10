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
    # Unit operations
    "CSTR",
    "CSTRParams",
    "Flash",
    "FlashParams",
    "Mixer",
    "Splitter",
    # Flowsheet
    "Flowsheet",
    "Unit",
    # Solvers
    "fixed_point_solve",
    "newton_solve",
    "rachford_rice",
]
