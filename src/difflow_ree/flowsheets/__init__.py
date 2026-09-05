"""Pre-built flowsheet templates for REE separation.

This module provides ready-to-use flowsheet configurations:
- ExtractStripCircuit: Basic 2-section circuit
- ExtractScrubStripCircuit: Industrial 3-section standard
- SplitShellCascade: Multi-product branched design
- FullSeparationTrain: Complete REE separation plant (a fixed sequence)

and, from #202, the superstructure layer that lifts the two limits of
that fixed sequence:

- ``ports``: typed aqueous / organic / solid ports, so an outlet may only
  be connected to a compatible inlet.
- ``modules``: the module library --- extract-scrub-strip, split-shell,
  cerium oxidation, precipitation, and saponification with solvent
  regeneration (#197) --- each declaring those ports.
- ``train.SeparationTrain``: module instances plus a connectivity map,
  solved by :class:`difflow.flowsheet.Flowsheet`, so the organic loop is
  closed by adding an edge rather than by writing a solver.
- ``constraints``: third-phase onset, loading and hydraulic capacity as
  inequality constraints an optimizer can be given, not ``info`` flags it
  will walk straight past.
- ``screening``: the Fenske lower bound as a cheap admissibility filter,
  so infeasible topologies are rejected before anything is costed.
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

# -- superstructure layer (#202) --------------------------------------
from difflow_ree.flowsheets.ports import (
    DIRECTIONS,
    PHASES,
    Port,
    PortMismatchError,
    PortSet,
    check_connection,
    species_gap,
)
from difflow_ree.flowsheets.constraints import (
    CONSTRAINT_KINDS,
    ConstraintSet,
    OperatingConstraint,
    OperatingLimits,
    hydraulic_constraint,
    loading_constraint,
    phase_ratio_constraints,
    third_phase_constraint,
)
from difflow_ree.flowsheets.modules import (
    MODULE_LIBRARY,
    CeriumOxidationModule,
    ExtractScrubStripModule,
    PrecipitationModule,
    REEModule,
    SaponificationModule,
    SolventRegenerationParams,
    SplitShellModule,
    build_module,
    get_module_class,
    list_modules,
    module_from_dict,
    pad_stream,
    with_stages,
)
from difflow_ree.flowsheets.train import (
    Connection,
    Feed,
    SeparationTrain,
    TopologyError,
    TrainResult,
)
from difflow_ree.flowsheets.screening import (
    ScreeningReport,
    ScreeningVerdict,
    minimum_stages,
    screen_separation,
    screen_topologies,
    screen_train,
    separation_factor,
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
    # ports (#202)
    "PHASES",
    "DIRECTIONS",
    "Port",
    "PortSet",
    "PortMismatchError",
    "check_connection",
    "species_gap",
    # constraints (#202)
    "CONSTRAINT_KINDS",
    "OperatingConstraint",
    "ConstraintSet",
    "OperatingLimits",
    "third_phase_constraint",
    "loading_constraint",
    "hydraulic_constraint",
    "phase_ratio_constraints",
    # modules (#202)
    "REEModule",
    "ExtractScrubStripModule",
    "SplitShellModule",
    "CeriumOxidationModule",
    "PrecipitationModule",
    "SaponificationModule",
    "SolventRegenerationParams",
    "MODULE_LIBRARY",
    "list_modules",
    "get_module_class",
    "build_module",
    "module_from_dict",
    "pad_stream",
    "with_stages",
    # train (#202)
    "SeparationTrain",
    "Connection",
    "Feed",
    "TrainResult",
    "TopologyError",
    # screening (#202)
    "ScreeningVerdict",
    "ScreeningReport",
    "minimum_stages",
    "separation_factor",
    "screen_separation",
    "screen_train",
    "screen_topologies",
]
