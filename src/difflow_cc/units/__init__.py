"""Unit operations for carbon capture.

This module provides differentiable unit operations for:
- Amine absorption columns
- Amine strippers/regenerators
- Membrane separators
- Adsorption systems (PSA, TSA, VSA, TVSA)

All units follow the difflow convention:
    outlet, info = unit(inlet, **kwargs)

where outlet is a Stream dict and info contains operation details.
"""

from difflow_cc.units.absorber import (
    AbsorberParams,
    AmineAbsorber,
)

from difflow_cc.units.stripper import (
    StripperParams,
    AmineStripper,
)

from difflow_cc.units.membrane import (
    MembraneParams,
    MembraneSeparator,
    MultistageMembrane,
)

from difflow_cc.units.adsorption import (
    AdsorptionParams,
    PSAUnit,
    TSAUnit,
    VSAUnit,
    TVSAUnit,
)

__all__ = [
    # Amine absorption
    "AbsorberParams",
    "AmineAbsorber",
    # Amine stripping
    "StripperParams",
    "AmineStripper",
    # Membrane
    "MembraneParams",
    "MembraneSeparator",
    "MultistageMembrane",
    # Adsorption
    "AdsorptionParams",
    "PSAUnit",
    "TSAUnit",
    "VSAUnit",
    "TVSAUnit",
]
