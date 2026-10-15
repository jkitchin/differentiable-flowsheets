"""Sulfur recovery: modified-Claus train and SCOT-type tail-gas treating (#472).

A library, NOT a palette operation (see ``CLAUDE.md`` here):

* :mod:`.vapor` -- sulfur allotropes S2...S8, saturation over the liquid
  (measured total, Ferreira & Lobo 2011), dew point.
* :mod:`.claus` -- furnace (adiabatic Gibbs), WHB/condensers, converters,
  air-demand control (:func:`solve_air`).
* :mod:`.tgtu` -- hydrogenation/hydrolysis, quench, MDEA absorber, the
  closed SRU with the acid-gas recycle (:func:`solve_sru`).
"""

from difflow_refinery.sulfur import claus, tgtu, vapor
from difflow_refinery.sulfur.claus import (AIR_O2, SPECIES, ClausParams, ConverterParams,
                                           claus_train, converter, cool, furnace,
                                           oxygen_demand, solve_air, steam_make)
from difflow_refinery.sulfur.tgtu import (TGTUParams, hydrogenate, incinerate, quench,
                                          solve_sru, tgtu as treat_tail_gas)
from difflow_refinery.sulfur.vapor import (condense, dew_point, saturation_pressures,
                                           total_vapor_pressure)

__all__ = ["AIR_O2", "SPECIES", "ClausParams", "ConverterParams", "TGTUParams", "claus",
           "claus_train", "condense", "converter", "cool", "dew_point", "furnace",
           "hydrogenate", "incinerate", "oxygen_demand", "quench", "saturation_pressures",
           "solve_air", "solve_sru", "steam_make", "tgtu", "total_vapor_pressure",
           "treat_tail_gas", "vapor"]
