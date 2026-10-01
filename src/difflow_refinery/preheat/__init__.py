"""The crude preheat train, desalter and preflash drum in front of the crude unit.

:class:`PreheatTrain` is the cold train, desalter, hot train and preflash
drum on their own, given hot streams; :class:`PreheatedCrudeUnit` couples
it to the atmospheric column, whose pumparounds and products are the hot
streams, so that the furnace inlet temperature -- and with it the fired
duty -- is an answer rather than an input. Exchanger fouling enters as a
resistance on each exchanger's UA; :mod:`~difflow_refinery.preheat.fouling`
has the Ebert-Panchal model, with illustrative constants.
"""

from difflow_refinery.preheat.flash import ColumnThermoEnthalpy, crude_enthalpy, crude_split
from difflow_refinery.preheat.fouling import (
    EbertPanchal,
    exchanger_film_temperature,
    film_temperature,
    fouling_rate,
    fouling_rates,
)
from difflow_refinery.preheat.ops import (
    CrudeUnitWithPreheat,
    CrudeUnitWithPreheatParams,
    Desalter,
    DesalterUnitParams,
    PreflashDrum,
    PreflashDrumUnitParams,
)
from difflow_refinery.preheat.train import (
    DesalterParams,
    HotStream,
    PreflashDrumParams,
    PreheatExchanger,
    PreheatTrain,
    PreheatTrainParams,
    PreheatTrainResult,
)
from difflow_refinery.preheat.unit import PreheatedCrudeUnit, PreheatedUnitResult

__all__ = [
    "ColumnThermoEnthalpy", "crude_enthalpy", "crude_split",
    "EbertPanchal", "exchanger_film_temperature", "film_temperature", "fouling_rate", "fouling_rates",
    "CrudeUnitWithPreheat", "CrudeUnitWithPreheatParams", "Desalter", "DesalterUnitParams",
    "PreflashDrum", "PreflashDrumUnitParams",
    "DesalterParams", "HotStream", "PreflashDrumParams", "PreheatExchanger", "PreheatTrain",
    "PreheatTrainParams", "PreheatTrainResult", "PreheatedCrudeUnit", "PreheatedUnitResult",
]
