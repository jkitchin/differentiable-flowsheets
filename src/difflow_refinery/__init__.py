"""Refinery Plugin for difflow: crude characterization and a vacuum column.

- Assays and characterization (:mod:`difflow_refinery.assay`): a TBP curve
  plus bulk properties cut into pseudocomponents, extended past 565 C into
  the residue on a probability scale, with a residue lump whose properties
  are set directly, and sulfur / nitrogen / CCR / metals per cut.
- Correlations (:mod:`difflow_refinery.correlations`): Twu criticals and
  molecular weight, Kesler-Lee acentric factor and liquid Cp,
  Maxwell-Bonnell vapor pressure (the D1160 vacuum conversion), and the
  Riazi-Daubert and Lee-Kesler forms for comparison.
- A stage-network column (:mod:`difflow_refinery.column`): Naphtali-Sandholm
  MESH equations with routed liquids (side draws, pumparounds,
  entrainment), any output specifiable in place of any knob, damped Newton
  and implicit-function gradients.
- :class:`VacuumColumn` (:mod:`difflow_refinery.vacuum`): atmospheric
  residue to LVGO, HVGO, slop and vacuum residue.

Every number is a JAX function of the assay, the feed and the operating
specs, so the gradient of an HVGO end point or a residue yield with respect
to the furnace outlet temperature -- or to one point of the TBP curve -- is
exact and costs one linear solve.

Quick Start:
    >>> import difflow_refinery as dr
    >>> char = dr.characterize(dr.heavy_crude())
    >>> feed = dr.atmospheric_residue(char, crude_rate_kg_s=100.0)
    >>> vdu = dr.VacuumColumn(dr.VacuumColumnParams(components=char.components))
    >>> overhead, lvgo, hvgo, slop, residue, info = vdu(feed)
    >>> info["properties"]["hvgo"]["T95"]   # K
"""

import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0"

from difflow_refinery.assay import (  # noqa: E402
    Assay,
    Characterization,
    PseudoComponents,
    atmospheric_residue,
    characterize,
    heavy_crude,
    light_crude,
    product_properties,
    tbp_fraction,
    tbp_point,
    tbp_temperature,
    vacuum_cuts,
)
from difflow_refinery.column import ColumnLayout, Route, Spec, StageColumn  # noqa: E402
from difflow_refinery.thermo import ColumnThermo, steam_enthalpy  # noqa: E402
from difflow_refinery.vacuum import (  # noqa: E402
    CrackingWarning,
    VacuumColumn,
    VacuumColumnParams,
    VacuumConvergenceWarning,
    default_vacuum_specs,
)
from difflow_refinery import correlations  # noqa: E402

__all__ = [
    "Assay",
    "Characterization",
    "PseudoComponents",
    "atmospheric_residue",
    "characterize",
    "heavy_crude",
    "light_crude",
    "product_properties",
    "tbp_fraction",
    "tbp_point",
    "tbp_temperature",
    "vacuum_cuts",
    "ColumnLayout",
    "Route",
    "Spec",
    "StageColumn",
    "ColumnThermo",
    "steam_enthalpy",
    "CrackingWarning",
    "VacuumColumn",
    "VacuumColumnParams",
    "VacuumConvergenceWarning",
    "default_vacuum_specs",
    "correlations",
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
