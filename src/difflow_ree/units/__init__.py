"""Unit operations for REE solvent extraction.

This module provides REE-specific unit operations:
- REEExtractor: Multi-stage extraction cascade
- REEScrubber: Scrubbing section for impurity removal
- REEStripper: Stripping section for product recovery
- Precipitators: Oxalate, carbonate, hydroxide precipitation
- CeriumOxidizer: Selective Ce removal via oxidation
"""

from difflow_ree.units.extraction import (
    REEExtractor,
    REEExtractorParams,
    REEMixerSettler,
    MixerSettlerParams,
)
from difflow_ree.units.scrubbing import (
    REEScrubber,
    ScrubberParams,
)
from difflow_ree.units.stripping import (
    REEStripper,
    StripperParams,
)
from difflow_ree.units.precipitation import (
    OxalatePrecipitator,
    CarbonatePrecipitator,
    HydroxidePrecipitator,
    PrecipitatorParams,
)
from difflow_ree.units.cerium import (
    CeriumOxidizer,
    CeriumOxidizerParams,
)

__all__ = [
    # Extraction
    "REEExtractor",
    "REEExtractorParams",
    "REEMixerSettler",
    "MixerSettlerParams",
    # Scrubbing
    "REEScrubber",
    "ScrubberParams",
    # Stripping
    "REEStripper",
    "StripperParams",
    # Precipitation
    "OxalatePrecipitator",
    "CarbonatePrecipitator",
    "HydroxidePrecipitator",
    "PrecipitatorParams",
    # Cerium
    "CeriumOxidizer",
    "CeriumOxidizerParams",
]
