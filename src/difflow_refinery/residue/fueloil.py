"""The fuel-oil pool of the residue route: desulfurized residue plus cutters, and the cutter arithmetic.

Sulfur blends linearly by mass (:class:`~difflow_refinery.BlendPool`'s rule),
so a residue at ``S_r`` cut with a cutter at ``S_c`` reaches a spec ``S_s``
only with a cutter mass fraction of at least::

    x_c = (S_r - S_s) / (S_r - S_c)                      (cutter_fraction_for_sulfur)

For an atmospheric residue at 3 wt% and a 15 ppm diesel cutter that is 83 %
cutter by mass to make 0.5 wt%: the "fuel oil" would be a diesel with some
residue in it, worth less than the diesel it consumed, and the refinery
makes about a third as much diesel as residue. The vacuum residue is worse
(the VDU puts the sulfur in its bottoms). Cutter blending is what sets the
*viscosity* of a residual fuel; for the sulfur of a high-sulfur crude's
residue the conversion unit -- :class:`~.unit.ResidueDesulfurizer` -- has to
do the work, and the cutter then trims the last few tenths of a percent.

Specs: :data:`VLSFO_SPECS` is the pool's default fuel-oil spec set --
sulfur 0.50 wt% (the MARPOL Annex VI global cap from 2020), viscosity
380 cSt at 50 C, density 991 kg/m^3 at 15 C (SG 0.991) and CCR 18 wt%
(the ISO 8217 RMG 380 limits, as recalled -- unverified against the
standard's table).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from jax import Array

from difflow_refinery.blending import PRODUCT_SPECS, BlendComponent, BlendPool, BlendResult, BlendSpec
from difflow_refinery.hydrotreating.feed import straight_run_cut

#: Very-low-sulfur fuel oil specs (see module docstring).
VLSFO_SPECS: tuple[BlendSpec, ...] = tuple(PRODUCT_SPECS["fuel_oil"])

#: Properties a fuel-oil pool reads from a component.
FUEL_OIL_PROPERTIES: tuple[str, ...] = ("SG", "S_ppm", "N_ppm", "CCR_wt", "viscosity_cSt")


def cutter_fraction_for_sulfur(S_base, S_cutter, S_spec) -> Array:
    """Cutter mass fraction that brings a base stock at ``S_base`` to ``S_spec``.

    Any consistent unit (wt%, wppm). Sulfur blends by mass, so this is the
    lever rule ``(S_base - S_spec) / (S_base - S_cutter)``; a value outside
    ``[0, 1]`` means the spec is already met (``< 0``) or cannot be reached
    with that cutter (``> 1``).
    """
    return (S_base - S_spec) / (S_base - S_cutter)


def atmospheric_residue_cut(char, rate=1.0, T_lo: float = 623.15, basis: str = "mass",
                            cuts: Sequence[str] | None = None) -> dict:
    """An idealized atmospheric residue: every cut boiling above ``T_lo`` (K), in crude proportion.

    For running the residue route without a crude unit (the CDU's
    ``"residue"`` product is the real one). ``rate`` in kg/s (``basis="mass"``)
    or mol/s. ``cuts`` instead of ``T_lo`` when ``char`` is traced.
    """
    return straight_run_cut(char, T_lo, np.inf, rate=rate, basis=basis, cuts=cuts)


def _property_mode(c: BlendComponent) -> BlendComponent:
    if not c.has_composition:
        return c
    return BlendComponent.from_properties(c.name, **{k: c.properties[k] for k in FUEL_OIL_PROPERTIES
                                                     if k in c.properties})


def fuel_oil_blend(components: Sequence[BlendComponent], volumes, specs=VLSFO_SPECS,
                   rules=None) -> BlendResult:
    """Blend fuel oil from ``components`` at standard volume flows ``volumes`` (m^3/s at 15 C).

    Property mode (stream-mode components are converted), so the
    desulfurized residue (:meth:`~.unit.RDSResult.blend_component`) and
    cutters from any other unit's characterization blend together: sulfur,
    nitrogen and CCR by mass, SG by volume, viscosity by Refutas (by mass).
    Every component's viscosity must be at one temperature (50 C by default).
    """
    comps = [_property_mode(c) for c in components]
    return BlendPool("fuel_oil", specs=list(specs), rules=rules)(comps, volumes, basis="volume_flow")


__all__ = ["VLSFO_SPECS", "FUEL_OIL_PROPERTIES", "cutter_fraction_for_sulfur", "atmospheric_residue_cut",
           "fuel_oil_blend"]
