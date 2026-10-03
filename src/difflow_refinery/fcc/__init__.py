"""Fluid catalytic cracking (FCC): riser, regenerator heat balance, main fractionator.

Issue #308. A riser reactor with lumped cracking kinetics (Weekman-Nace
3-lump, Lee 4-lump, Ancheyta 5-lump) and catalyst deactivation, integrated
in height with diffrax; a regenerator burning the coke to CO/CO2/H2O/SO2;
the two solved TOGETHER as the unit's heat balance (catalyst-to-oil ratio
and regenerator temperature are unknowns, the riser outlet temperature a
spec) with implicit gradients; the lumps mapped onto real light species and
product pseudocomponents; and a simplified main fractionator.

Kinetic constants and most product-property parameters are ILLUSTRATIVE
defaults: see ``docs/unit-operations-refinery.md``, "The fluid catalytic
cracker", for what is sourced and what is not.

>>> from difflow_refinery.fcc import FCCUnit, FCCParams, FCCFeed  # doctest: +SKIP
"""

from difflow_refinery.fcc import fractionator, kinetics, regenerator, riser, species
from difflow_refinery.fcc.feed import FCCFeed
from difflow_refinery.fcc.kinetics import (
    GROUPS,
    ILLUSTRATIVE_5LUMP,
    LumpScheme,
    Reaction,
    ancheyta_5,
    deactivation,
    get_scheme,
    lee_4,
    voorhies_coke,
    weekman_nace_3,
)
from difflow_refinery.fcc.regenerator import arthur_co_co2
from difflow_refinery.fcc.unit import (
    FCCConvergenceWarning,
    FCCParams,
    FCCUnit,
    RegeneratorTemperatureWarning,
)
from difflow_refinery.fcc.planning import FCCPlanningModel, fcc_block

__all__ = [
    "FCCFeed", "FCCParams", "FCCUnit", "FCCConvergenceWarning", "RegeneratorTemperatureWarning",
    "GROUPS", "ILLUSTRATIVE_5LUMP", "LumpScheme", "Reaction",
    "ancheyta_5", "lee_4", "weekman_nace_3", "get_scheme",
    "deactivation", "voorhies_coke", "arthur_co_co2",
    "FCCPlanningModel", "fcc_block",
    "fractionator", "kinetics", "regenerator", "riser", "species",
]
