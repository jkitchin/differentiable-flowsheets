"""Refinery amine treating (#471): absorbers, regenerator, the closed loop.

A library, not a palette operation (the gas plant's ``AmineTreater`` stays
the fixed-fraction shortcut). The equilibrium is the Kent-Eisenberg model
of :mod:`difflow_cc.equilibrium.acid_gas`; rates, transport and thermal
constants are in :mod:`.properties`, each with its source.

>>> from difflow_refinery import treating
>>> p = treating.AmineUnitParams("MDEA", 0.45, (treating.Contactor(
...     "fuel", 120.0, 8, treating.TrayParams(area=200.0, k_G=1e-5, k_L=1e-4)),))
"""

from difflow_refinery.treating import properties
from difflow_refinery.treating.absorber import AbsorberResult, TrayParams, absorb, efficiencies
from difflow_refinery.treating.properties import AmineProperties, amine_properties
from difflow_refinery.treating.regenerator import RegeneratorResult, regenerate
from difflow_refinery.treating.unit import (AmineUnitParams, AmineUnitResult, Contactor,
                                            lean_water, rich_flash, solve_amine_unit)

__all__ = [
    "AbsorberResult", "AmineProperties", "AmineUnitParams", "AmineUnitResult", "Contactor",
    "RegeneratorResult", "TrayParams", "absorb", "amine_properties", "efficiencies",
    "lean_water", "properties", "regenerate", "rich_flash", "solve_amine_unit",
]
