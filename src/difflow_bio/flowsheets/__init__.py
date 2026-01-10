"""Pre-built flowsheet templates for biopharmaceutical manufacturing.

This module provides ready-to-use flowsheet configurations:
- mAbDSPTrain: Standard monoclonal antibody downstream process
- PlatformDSP: Generic platform DSP train
- ViralClearanceTrain: Viral safety focused purification

All flowsheets are fully differentiable using JAX.
"""

from difflow_bio.flowsheets.mab_dsp import (
    mAbDSPTrain,
    mAbDSPParams,
)
from difflow_bio.flowsheets.platform import (
    PlatformDSP,
    PlatformDSPParams,
)
from difflow_bio.flowsheets.viral_clearance import (
    ViralClearanceTrain,
    ViralClearanceParams,
)

__all__ = [
    "mAbDSPTrain",
    "mAbDSPParams",
    "PlatformDSP",
    "PlatformDSPParams",
    "ViralClearanceTrain",
    "ViralClearanceParams",
]
