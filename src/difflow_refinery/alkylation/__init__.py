"""Alkylation: C3-C5 olefins + isobutane over H2SO4 or HF, with the isobutane recycle.

* :mod:`~difflow_refinery.alkylation.correlations` -- the yield, octane and
  acid correlations of Sauer, Colville & Burwick (1964) as restated by
  Bracken & McCormick (1968) and the GAMS model ``process.gms``, and that
  optimisation problem itself (:func:`solve_process_gms`).
* :mod:`~difflow_refinery.alkylation.species` -- the real species and the
  60 F volumes (COSTALD) and liquid heats of formation the model needs.
* :mod:`~difflow_refinery.alkylation.reactor` -- the reactor: per-olefin
  stoichiometry to TMP/DMH/DMP alkylate and C12 heavy ends, the split set
  by the correlation's yield.
* :mod:`~difflow_refinery.alkylation.unit` -- the unit: makeup, reactor,
  depropanizer, deisobutanizer and debutanizer as a difflow ``Flowsheet``
  with the DIB overhead as the recycle tear.
* :mod:`~difflow_refinery.alkylation.planning` -- :func:`alky_block`.

The correlations are regressions on one 1960s plant: illustrative, to be
refitted to a unit's data before they are used to plan. See
``docs/unit-operations-refinery.md``, "Alkylation".
"""

from difflow_refinery.alkylation import correlations, species
from difflow_refinery.alkylation.correlations import (
    PROCESS_OPTIMUM_PROFIT,
    SAUER_1964,
    ProcessSolution,
    SauerCorrelation,
    acid_per_alkylate,
    alkylate_yield,
    motor_octane,
    solve_process_gms,
)
from difflow_refinery.alkylation.feeds import (
    C3C4_COMPOSITION,
    C4_COMPOSITION,
    c3c4_olefin_feed,
    c4_olefin_feed,
)
from difflow_refinery.alkylation.planning import (
    ALKY_LEVERS,
    alky_block,
)
from difflow_refinery.alkylation.reactor import (
    DEFAULT_SELECTIVITY,
    AlkylationRangeWarning,
    AlkylationReactor,
    AlkylationReactorParams,
    alkylation_thermo,
    feed_stream,
    flows_array,
)
from difflow_refinery.alkylation.species import (
    ALKYLATE,
    ALKYLATION_SPECIES,
    HEAVY_END,
    INERTS,
    OLEFINS,
)
from difflow_refinery.alkylation.unit import (
    OUTPUT_UNITS,
    AlkylationResult,
    AlkylationUnit,
    AlkylationUnitParams,
    ColumnSpec,
    IsobutaneMakeup,
    IsobutaneMakeupParams,
    alkylate_properties,
    atom_balance,
)

__all__ = [
    "correlations", "species",
    "PROCESS_OPTIMUM_PROFIT", "SAUER_1964", "ProcessSolution", "SauerCorrelation",
    "acid_per_alkylate", "alkylate_yield", "motor_octane", "solve_process_gms",
    "C3C4_COMPOSITION", "C4_COMPOSITION", "c3c4_olefin_feed", "c4_olefin_feed",
    "ALKY_LEVERS", "alky_block",
    "DEFAULT_SELECTIVITY", "AlkylationRangeWarning", "AlkylationReactor",
    "AlkylationReactorParams", "alkylation_thermo", "feed_stream", "flows_array",
    "ALKYLATE", "ALKYLATION_SPECIES", "HEAVY_END", "INERTS", "OLEFINS",
    "OUTPUT_UNITS", "AlkylationResult", "AlkylationUnit", "AlkylationUnitParams",
    "ColumnSpec", "IsobutaneMakeup", "IsobutaneMakeupParams", "alkylate_properties",
    "atom_balance",
]
