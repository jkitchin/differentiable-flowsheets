"""The FCC feed: a gas oil reduced to the bulk properties the lumped model reads.

A lumped riser model has ONE feed lump, so a feed enters it only through
bulk properties: rate, molar mass, gravity and Watson K (crackability),
hydrogen (the hydrogen balance), sulfur and nitrogen (product qualities,
basic-N poisoning), Conradson carbon and Ni+V (additive and contaminant
coke). :class:`FCCFeed` holds those, as ``jax`` arrays, computed from the
per-pseudocomponent properties of the shared characterization (#301) --
so a gradient of an FCC yield with respect to a crude-assay TBP point, or
to the VDU's cut, is one ``jax.grad``.

Three constructors:

* :meth:`FCCFeed.from_components` -- a property table (the VDU's
  :class:`~difflow_refinery.vacuum.PseudoComponents`, or a
  :class:`~difflow_refinery.assay.Characterization`) and the mass flow of
  each component.
* :meth:`FCCFeed.from_stream` -- a difflow stream (``F_<name>`` mol/s), e.g.
  the VDU's ``hvgo`` outlet, on such a table.
* :meth:`FCCFeed.from_characterization` -- an ideal TBP cut of a crude
  characterization between two temperatures (whole pseudocomponents whose
  Tb is inside), for running without a VDU.

Hydrogen. Per component it is taken from the characterization's
:class:`~difflow_refinery.composition.Composition` when there is one (#305),
otherwise estimated by Goossens (1997) from the refractive index the
composition module estimates (Riazi-Daubert/Huang) and d20 -- the same
functions :mod:`difflow_refinery.composition` uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery import correlations as corr
from difflow_refinery.composition import density_20c, estimate_refractive_index, hydrogen_goossens

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class FCCFeed:
    """Bulk properties of an FCC feed (all ``jax`` arrays).

    Attributes:
        mass: Feed rate (kg/s).
        MW: Number-average molar mass (g/mol).
        SG: Specific gravity 60/60 F (ideal mixing by volume).
        Tb: Mass-average (weight-average) boiling point (K).
        Kw: Watson K from ``Tb`` and ``SG``.
        hydrogen, sulfur, nitrogen, ccr, nickel_vanadium: Mass fractions in
            the feed (Ni+V as a mass fraction too: 30 wppm is 3e-5).
    """

    mass: Array
    MW: Array
    SG: Array
    Tb: Array
    Kw: Array
    hydrogen: Array
    sulfur: Array
    nitrogen: Array
    ccr: Array
    nickel_vanadium: Array

    @property
    def api(self) -> Array:
        return corr.api_from_sg(self.SG)

    def with_rate(self, mass) -> "FCCFeed":
        """A copy at another feed rate (kg/s)."""
        import dataclasses

        return dataclasses.replace(self, mass=jnp.asarray(mass, dtype=float))

    def as_dict(self) -> dict[str, Array]:
        return {k: getattr(self, k) for k in (
            "mass", "MW", "SG", "Tb", "Kw", "hydrogen", "sulfur", "nitrogen",
            "ccr", "nickel_vanadium")}

    @classmethod
    def from_components(cls, components, mass_flows: Array,
                        hydrogen: Array | None = None) -> "FCCFeed":
        """Bulk feed from a property table and per-component mass flows (kg/s).

        Args:
            components: Anything with per-component ``Tb``, ``SG``, ``MW`` and
                (optionally) ``sulfur``, ``nitrogen``, ``ccr``,
                ``nickel_vanadium`` arrays: a
                :class:`~difflow_refinery.vacuum.PseudoComponents` or the
                pseudo-component part of a characterization (see
                :meth:`from_characterization`).
            mass_flows: Mass flow of each component (kg/s).
            hydrogen: Hydrogen mass fraction of each component; default the
                Goossens (1997) estimate.
        """
        m = jnp.asarray(mass_flows, dtype=float)
        Tb, SG, MW = (jnp.asarray(getattr(components, k), dtype=float) for k in ("Tb", "SG", "MW"))
        if hydrogen is None:
            hydrogen = hydrogen_goossens(estimate_refractive_index(Tb, SG), density_20c(SG), MW)
        total = jnp.sum(m)
        w = m / total

        def avg(key):
            v = getattr(components, key, None)
            return jnp.asarray(0.0) if v is None else jnp.sum(w * jnp.asarray(v, dtype=float))

        sg = 1.0 / jnp.sum(w / SG)
        tb = jnp.sum(w * Tb)
        return cls(
            mass=total,
            MW=1.0 / jnp.sum(w / MW),
            SG=sg,
            Tb=tb,
            Kw=corr.watson_k(tb, sg),
            hydrogen=jnp.sum(w * jnp.asarray(hydrogen, dtype=float)),
            sulfur=avg("sulfur"),
            nitrogen=avg("nitrogen"),
            ccr=avg("ccr"),
            nickel_vanadium=avg("nickel_vanadium"),
        )

    @classmethod
    def from_stream(cls, stream: Mapping[str, Array], components,
                    hydrogen: Array | None = None) -> "FCCFeed":
        """Bulk feed from a difflow stream (``F_<name>``, mol/s) on ``components``.

        Species of the stream not in ``components.names`` are ignored here;
        :class:`~difflow_refinery.fcc.FCCUnit` passes them through to its
        dry-gas outlet so the flowsheet balance still closes.
        """
        mol = jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float)
                         for n in components.names])
        return cls.from_components(components, mol * jnp.asarray(components.MW) / 1000.0, hydrogen)

    @classmethod
    def from_characterization(cls, char, rate: float | Array, T_lo: float, T_hi: float
                              ) -> "FCCFeed":
        """An ideal TBP cut ``T_lo < Tb < T_hi`` (K) of a crude characterization.

        Whole pseudocomponents are taken (the selection needs concrete
        boiling points -- it fixes a shape), then re-scaled to ``rate``
        (kg/s). Differentiable in every assay number through the cuts'
        properties. Hydrogen comes from ``char.composition`` if it is set.
        """
        k = len(char.light_names)
        Tb = np.asarray(jax.lax.stop_gradient(char.Tb))
        idx = np.nonzero((Tb > T_lo) & (Tb < T_hi))[0]
        if idx.size == 0:
            raise ValueError(f"no pseudocomponent has {T_lo} K < Tb < {T_hi} K")
        mass = char.mass_fraction[k + idx]
        comp = _Table(
            Tb=char.Tb[idx], SG=char.SG[idx], MW=char.MW[idx],
            sulfur=char.sulfur[k + idx], nitrogen=char.nitrogen[k + idx],
            ccr=char.ccr[k + idx], nickel_vanadium=char.nickel_vanadium[k + idx])
        h = None
        if getattr(char, "composition", None) is not None:
            h = char.composition.hydrogen[k + idx]
        flows = jnp.asarray(rate, dtype=float) * mass / jnp.sum(mass)
        return cls.from_components(comp, flows, h)


@dataclass(frozen=True)
class _Table:
    Tb: Array
    SG: Array
    MW: Array
    sulfur: Array
    nitrogen: Array
    ccr: Array
    nickel_vanadium: Array


jax.tree_util.register_dataclass(
    FCCFeed,
    data_fields=["mass", "MW", "SG", "Tb", "Kw", "hydrogen", "sulfur", "nitrogen",
                 "ccr", "nickel_vanadium"],
    meta_fields=[],
)
