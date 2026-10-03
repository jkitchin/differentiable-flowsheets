"""The vacuum distillation unit, and a vacuum view of the shared characterization.

The crude unit, this column and the blend pool characterize a crude ONCE,
with :func:`difflow_refinery.characterize` on an :class:`difflow_refinery.Assay`
that has a :class:`~difflow_refinery.HeavyEnd` (#301). The vacuum column's
property table is that characterization's ``pseudo_components()``, so the
crude unit's residue feeds it as it stands -- same grid, no re-cut -- and
nothing between the two units interpolates. What is here:

- :mod:`~difflow_refinery.vacuum.assay`: a compatibility view. Its
  ``Assay`` (Celsius, wt%) converts with ``to_assay()``, and its
  ``characterize`` runs the shared characterization on a vacuum cut grid and
  returns the old ``(components, yields, light_ends, Kw)`` shape.
  ``atmospheric_residue`` is an idealized TBP cut for running the column
  without a crude unit in front of it.
- :mod:`~difflow_refinery.vacuum.correlations`: the vacuum code's names for
  correlations that live once, in :mod:`difflow_refinery.correlations`.
- :mod:`~difflow_refinery.vacuum.column`: a stage-network column
  (Naphtali-Sandholm MESH with routed liquids, Murphree efficiencies and
  entrainment; any output specifiable in place of any knob).
- :class:`VacuumColumn`: atmospheric residue to LVGO, HVGO, slop and vacuum
  residue. Feed species it does not model (light ends, the crude unit's
  water) leave with the overhead, so a flowsheet still balances.

>>> from difflow_refinery import vacuum
>>> char = vacuum.characterize(vacuum.heavy_crude())
>>> feed = vacuum.atmospheric_residue(char, crude_rate_kg_s=100.0)
>>> vdu = vacuum.VacuumColumn(vacuum.VacuumColumnParams(components=char.components))
"""

from difflow_refinery.vacuum import correlations
from difflow_refinery.vacuum.assay import (
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
from difflow_refinery.vacuum.column import ColumnLayout, Route, StageColumn, StageSpec
from difflow_refinery.vacuum.thermo import ColumnThermo, steam_enthalpy
from difflow_refinery.vacuum.unit import (
    CrackingWarning,
    VacuumColumn,
    VacuumColumnParams,
    VacuumConvergenceWarning,
    default_vacuum_specs,
)

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
    "StageSpec",
    "StageColumn",
    "ColumnThermo",
    "steam_enthalpy",
    "CrackingWarning",
    "VacuumColumn",
    "VacuumColumnParams",
    "VacuumConvergenceWarning",
    "default_vacuum_specs",
    "correlations",
]
