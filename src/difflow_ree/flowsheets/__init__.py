"""Pre-built flowsheet templates for REE separation.

This module provides ready-to-use flowsheet configurations:
- ExtractStripCircuit: Basic 2-section circuit
- ExtractScrubStripCircuit: Industrial 3-section standard
- SplitShellCascade: Multi-product branched design
- FullSeparationTrain: Complete REE separation plant
"""

from difflow_ree.flowsheets.extract_strip import (
    ExtractStripCircuit,
    ExtractStripParams,
)
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.split_shell import (
    SplitShellCascade,
    SplitShellParams,
)
from difflow_ree.flowsheets.full_train import (
    FullSeparationTrain,
    SeparationTrainParams,
    GroupSeparator,
)

__all__ = [
    "ExtractStripCircuit",
    "ExtractStripParams",
    "ExtractScrubStripCircuit",
    "ExtractScrubStripParams",
    "SplitShellCascade",
    "SplitShellParams",
    "FullSeparationTrain",
    "SeparationTrainParams",
    "GroupSeparator",
]
