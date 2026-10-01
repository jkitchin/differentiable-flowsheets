"""The vacuum distillation unit and the characterization it is built on.

A self-contained stack, kept in its own namespace because the crude unit
(:mod:`difflow_refinery.assay`, :mod:`difflow_refinery.column`) has a
characterization and a column of its own:

- :mod:`~difflow_refinery.vacuum.assay`: a TBP curve plus bulk properties cut
  into pseudocomponents, extended past 565 C into the residue on a
  probability scale, with a residue lump whose properties are set directly,
  and sulfur / nitrogen / CCR / metals per cut.
- :mod:`~difflow_refinery.vacuum.correlations`: Twu criticals and molecular
  weight, Kesler-Lee acentric factor and liquid Cp, Maxwell-Bonnell vapor
  pressure (the D1160 vacuum conversion).
- :mod:`~difflow_refinery.vacuum.column`: a stage-network column
  (Naphtali-Sandholm MESH with routed liquids, Murphree efficiencies and
  entrainment; any output specifiable in place of any knob).
- :class:`VacuumColumn`: atmospheric residue to LVGO, HVGO, slop and vacuum
  residue.

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
