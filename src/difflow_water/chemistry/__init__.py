"""Water chemistry for difflow_water (a library, not palette operations).

- :mod:`~difflow_water.chemistry.seawater`: seawater physical properties
  (Sharqawy et al. 2010).
- :mod:`~difflow_water.chemistry.pitzer`: Pitzer activity model (HMW form
  as in PHRQPITZ; parameters from PHREEQC pitzer.dat).
- :mod:`~difflow_water.chemistry.speciation`: seawater/brine speciation,
  alkalinity, pH and mineral saturation indices.
"""

from difflow_water.chemistry import pitzer, seawater, speciation

__all__ = ["seawater", "pitzer", "speciation"]
