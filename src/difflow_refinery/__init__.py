"""Refinery Plugin for difflow: crude characterization and a vacuum column.

- Assays and characterization (:mod:`difflow_refinery.vacuum.assay`): a TBP curve
  plus bulk properties cut into pseudocomponents, extended past 565 C into
  the residue on a probability scale, with a residue lump whose properties
  are set directly, and sulfur / nitrogen / CCR / metals per cut.
- Correlations (:mod:`difflow_refinery.vacuum.correlations`): Twu criticals and
  molecular weight, Kesler-Lee acentric factor and liquid Cp,
  Maxwell-Bonnell vapor pressure (the D1160 vacuum conversion), and the
  Riazi-Daubert and Lee-Kesler forms for comparison.
- A stage-network column (:mod:`difflow_refinery.vacuum.column`): Naphtali-Sandholm
  MESH equations with routed liquids (side draws, pumparounds,
  entrainment), any output specifiable in place of any knob, damped Newton
  and implicit-function gradients.
- :class:`VacuumColumn` (:mod:`difflow_refinery.vacuum.unit`): atmospheric
  residue to LVGO, HVGO, slop and vacuum residue.

Every number is a JAX function of the assay, the feed and the operating
specs, so the gradient of an HVGO end point or a residue yield with respect
to the furnace outlet temperature -- or to one point of the TBP curve -- is
exact and costs one linear solve.

Quick Start:
    >>> import difflow_refinery as dr
    >>> char = dr.vacuum.characterize(dr.vacuum.heavy_crude())
    >>> feed = dr.vacuum.atmospheric_residue(char, crude_rate_kg_s=100.0)
    >>> vdu = dr.VacuumColumn(dr.VacuumColumnParams(components=char.components))
    >>> overhead, lvgo, hvgo, slop, residue, info = vdu(feed)
    >>> info["properties"]["hvgo"]["T95"]   # K
"""

__version__ = "0.1.0"

from difflow_refinery import vacuum
from difflow_refinery.vacuum import (
    CrackingWarning,
    PseudoComponents,
    StageSpec,
    VacuumColumn,
    VacuumColumnParams,
    VacuumConvergenceWarning,
    default_vacuum_specs,
)

__all__ = [
    "vacuum",
    "CrackingWarning",
    "PseudoComponents",
    "StageSpec",
    "VacuumColumn",
    "VacuumColumnParams",
    "VacuumConvergenceWarning",
    "default_vacuum_specs",
    "register",
]


def register(registry):
    """Register refinery unit operations with difflow.

    Called by ``difflow.plugins.load_plugins()`` when the plugin is
    discovered via entry points.

    Args:
        registry: difflow OperationRegistry instance
    """
    registry.register(
        name="VacuumColumn",
        cls=VacuumColumn,
        category="refinery",
        description="Vacuum distillation: atmospheric residue to LVGO, HVGO, "
                    "slop and vacuum residue",
        plugin="difflow_refinery",
    )
