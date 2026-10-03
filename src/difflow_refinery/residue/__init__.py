"""Residue desulfurization and the low-sulfur fuel-oil route (#331).

* :mod:`.kinetics` -- :class:`RDSKinetics`: the hydrotreating kinetics with
  residue constants, plus refractory residue sulfur, HDM (Ni+V onto the
  catalyst), CCR reduction and a small residue conversion. ILLUSTRATIVE
  constants, literature forms.
* :mod:`.feed` -- :func:`rds_layout`, :func:`rds_feed` (a characterized
  residue stream on the attribute layout).
* :mod:`.unit` -- :class:`ResidueDesulfurizer` (trickle beds on
  :mod:`difflow_refinery.hydroprocessing`, once-through treat gas, ideal
  product separation) and :class:`RDSResult`.
* :mod:`.fueloil` -- the fuel-oil pool: :func:`fuel_oil_blend` (the
  desulfurized residue and cutters in :class:`~difflow_refinery.BlendPool`)
  and :func:`cutter_fraction_for_sulfur`, the lever-arm arithmetic that
  says why cutter blending alone cannot carry a high-sulfur residue to
  0.5 wt%.

A library, not a palette operation. See ``docs/unit-operations-refinery.md``
("Residue desulfurization and fuel oil").
"""

from difflow_refinery.residue import feed, fueloil, kinetics, unit
from difflow_refinery.residue.feed import DEFAULT_RESIDUE_S_SHARE, rds_feed, rds_layout, residue_s_share
from difflow_refinery.residue.fueloil import (
    VLSFO_SPECS, atmospheric_residue_cut, cutter_fraction_for_sulfur, fuel_oil_blend)
from difflow_refinery.residue.kinetics import (
    RDS_ATTRIBUTES, RDSKineticParams, RDSKinetics, conversion_targets, residue_hdt_params)
from difflow_refinery.residue.unit import OUTPUT_UNITS, RDSParams, RDSResult, ResidueDesulfurizer

__all__ = [
    "feed", "fueloil", "kinetics", "unit",
    "DEFAULT_RESIDUE_S_SHARE", "rds_feed", "rds_layout", "residue_s_share",
    "VLSFO_SPECS", "atmospheric_residue_cut", "cutter_fraction_for_sulfur", "fuel_oil_blend",
    "RDS_ATTRIBUTES", "RDSKineticParams", "RDSKinetics", "conversion_targets", "residue_hdt_params",
    "OUTPUT_UNITS", "RDSParams", "RDSResult", "ResidueDesulfurizer",
]
