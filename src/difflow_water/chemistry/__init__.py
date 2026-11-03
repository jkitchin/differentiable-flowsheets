"""Water chemistry for difflow_water (a library, not palette operations).

Phase 1 provides seawater properties (:mod:`difflow_water.chemistry.seawater`).
Carbonate equilibria, activity models and scaling indices are planned
(see ``src/difflow_water/DESIGN.md``) and are not here yet.
"""

from difflow_water.chemistry import seawater

__all__ = ["seawater"]
