"""The REE stream schema, defined once as a superset (#196).

The correlation path (:class:`~difflow_ree.equilibrium.distribution.REEDistribution`
driving :class:`~difflow_ree.units.extraction.REEExtractor`) and the closed
mass-action path (:mod:`difflow_ree.equilibrium.mass_action`) sit behind one
interface so that cascade code is level-agnostic. Two things genuinely differ
between the levels and must not be hidden; this module handles the first of
them.

**State width.** The closed model needs species the correlation never looks
at: an acid balance, a counter-ion, an anion, water, the extractant inventory,
the loaded organic and the co-extracted acid. Rather than let each level grow
its own stream vocabulary, the schema is declared here once as the superset,
and the correlation path simply ignores the keys it does not use (it copies
unrecognised flows through untouched, which it already did).

**Degrees of freedom.** pH is an *input* to the correlation and an *output* of
the closed model, whose corresponding input is base addition or saponification
degree. That cannot be papered over by a schema; see
:func:`difflow_ree.equilibrium.mass_action.base_addition_for_pH`, the explicit
inverse problem that maps a design specified at one level onto the other.

Key conventions, which are the existing plugin conventions made explicit:

- A rare earth is keyed by its bare element symbol in *both* phases. Which
  phase it is in is a property of the stream, not of the key: the symbol in an
  aqueous stream is dissolved ``RE3+``, the same symbol in an organic stream is
  the loaded complex. This is what ``REEExtractor`` has always done.
- ``extractant`` is the **total** extractant molar flow on a **monomer**
  basis, free plus bound, exactly as ``REEExtractor``'s capacity calculation
  reads it. The reaction network converts to its own basis using
  ``monomers_per_component`` (2 for the dimeric acidic organophosphorus
  extractants).
- ``H`` is free acid in the aqueous phase, ``H_org`` co-extracted acid in the
  organic phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream


#: Anion keys the schema recognises, with their formal charge. An anion is one
#: conserved component in the reaction network, so exactly one of these is the
#: active anion for a given section.
ANION_CHARGES = {
    "Cl": -1,
    "NO3": -1,
    "SO4": -2,
}

#: Counter-ion keys the schema recognises, with their formal charge. The
#: counter-ion is a conserved component in every shipped network even before
#: saponification gives it a species to form (#196, #197).
COUNTER_ION_CHARGES = {
    "Na": 1,
    "NH4": 1,
    "K": 1,
}


@dataclass(frozen=True)
class REEStreamSchema:
    """Names of every quantity a REE stream may carry, at either level.

    Attributes:
        elements: REE symbols tracked by the section, in order.
        extractant: Species key for the extractant, e.g. ``"D2EHPA"``. Carries
            the **total** extractant molar flow on a **monomer** basis.
        diluent: Species key for the organic diluent, e.g. ``"kerosene"``.
        counter_ion: Species key for the aqueous counter-ion, a key of
            :data:`COUNTER_ION_CHARGES`, or None for a section with no
            counter-ion.
        anion: Species key for the aqueous anion, a key of
            :data:`ANION_CHARGES`.
        acid: Species key for free aqueous acid (protons). Default ``"H"``.
        organic_acid: Species key for co-extracted acid in the organic phase.
            Default ``"H_org"``.
        water: Species key for aqueous water. Default ``"H2O"``.
        organic_water: Species key for water dissolved in the organic phase.
            Default ``"H2O_org"``.

    Example:
        >>> s = REEStreamSchema(elements=("Nd", "Dy"), extractant="D2EHPA")
        >>> s.aqueous_keys()
        ('Nd', 'Dy', 'H', 'Na', 'Cl', 'H2O')
        >>> s.organic_keys()
        ('Nd', 'Dy', 'D2EHPA', 'kerosene', 'H_org', 'H2O_org')
    """

    elements: tuple[str, ...]
    extractant: str
    diluent: str = "kerosene"
    counter_ion: str | None = "Na"
    anion: str = "Cl"
    acid: str = "H"
    organic_acid: str = "H_org"
    water: str = "H2O"
    organic_water: str = "H2O_org"

    def __post_init__(self) -> None:
        """Validate the anion and counter-ion against the known charges.

        Raises:
            ValueError: If the anion or counter-ion is not recognised, or the
                element list is empty or contains duplicates.
        """
        if not self.elements:
            raise ValueError("REEStreamSchema requires at least one element.")
        if len(set(self.elements)) != len(self.elements):
            raise ValueError(
                f"REEStreamSchema elements contain duplicates: "
                f"{list(self.elements)}."
            )
        if self.anion not in ANION_CHARGES:
            raise ValueError(
                f"Unknown anion {self.anion!r}. Recognised: "
                f"{sorted(ANION_CHARGES)} (#196)."
            )
        if self.counter_ion is not None and self.counter_ion not in COUNTER_ION_CHARGES:
            raise ValueError(
                f"Unknown counter_ion {self.counter_ion!r}. Recognised: "
                f"{sorted(COUNTER_ION_CHARGES)}, or None (#196)."
            )

    # -- charges ---------------------------------------------------------

    @property
    def anion_charge(self) -> int:
        """Formal charge of the active anion (negative)."""
        return ANION_CHARGES[self.anion]

    @property
    def counter_ion_charge(self) -> int:
        """Formal charge of the counter-ion, or 0 when there is none."""
        if self.counter_ion is None:
            return 0
        return COUNTER_ION_CHARGES[self.counter_ion]

    # -- key sets --------------------------------------------------------

    def aqueous_keys(self) -> tuple[str, ...]:
        """Keys that may appear in an aqueous stream, in canonical order.

        Returns:
            Element symbols, free acid, counter-ion, anion, water.
        """
        keys = list(self.elements) + [self.acid]
        if self.counter_ion is not None:
            keys.append(self.counter_ion)
        keys += [self.anion, self.water]
        return tuple(keys)

    def organic_keys(self) -> tuple[str, ...]:
        """Keys that may appear in an organic stream, in canonical order.

        Returns:
            Element symbols (loaded complex), extractant total, diluent,
            co-extracted acid, water in organic.
        """
        return tuple(
            list(self.elements)
            + [self.extractant, self.diluent, self.organic_acid,
               self.organic_water]
        )

    def all_keys(self) -> tuple[str, ...]:
        """Every key in the superset, de-duplicated, aqueous keys first."""
        seen: list[str] = []
        for k in self.aqueous_keys() + self.organic_keys():
            if k not in seen:
                seen.append(k)
        return tuple(seen)

    def phase_of(self, key: str) -> str:
        """Phase a key belongs to when the stream's phase is ambiguous.

        Args:
            key: Species key.

        Returns:
            ``"organic"`` for the extractant, diluent, co-extracted acid and
            organic water; ``"aqueous"`` for everything else, including the
            element symbols, which are phase-ambiguous by design.
        """
        organic_only = {
            self.extractant, self.diluent, self.organic_acid,
            self.organic_water,
        }
        return "organic" if key in organic_only else "aqueous"

    # -- constructors ----------------------------------------------------

    def make_aqueous(
        self,
        element_flows: Mapping[str, float | Array],
        acid: float | Array = 0.0,
        counter_ion: float | Array = 0.0,
        anion: float | Array | None = None,
        water: float | Array = 0.0,
        T: float | Array = 298.15,
        P: float | Array = 101325.0,
        extra: Mapping[str, float | Array] | None = None,
    ) -> Stream:
        """Build an aqueous stream on this schema.

        Args:
            element_flows: Molar flow per REE symbol (mol/s). Missing elements
                default to zero.
            acid: Free acid molar flow (mol/s).
            counter_ion: Counter-ion molar flow (mol/s).
            anion: Anion molar flow (mol/s). None (default) closes the anion
                by electroneutrality, which is the physically required choice
                and the one the closed model assumes; see
                :meth:`electroneutral_anion_flow`.
            water: Water molar flow (mol/s).
            T: Temperature (K).
            P: Pressure (Pa).
            extra: Additional species carried through untouched.

        Returns:
            A difflow :class:`~difflow.streams.Stream`.

        Example:
            >>> s = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
            >>> feed = s.make_aqueous({"Nd": 0.1}, acid=0.01, water=50.0)
            >>> round(float(feed["F_Cl"]), 6)
            0.31
        """
        flows: dict[str, float | Array] = {
            el: element_flows.get(el, 0.0) for el in self.elements
        }
        flows[self.acid] = acid
        if self.counter_ion is not None:
            flows[self.counter_ion] = counter_ion
        if anion is None:
            anion = self.electroneutral_anion_flow(
                element_flows, acid, counter_ion
            )
        flows[self.anion] = anion
        flows[self.water] = water
        if extra:
            flows.update(extra)
        return make_stream(flows, T, P)

    def make_organic(
        self,
        extractant_flow: float | Array,
        diluent_flow: float | Array = 0.0,
        element_flows: Mapping[str, float | Array] | None = None,
        organic_acid: float | Array = 0.0,
        organic_water: float | Array = 0.0,
        T: float | Array = 298.15,
        P: float | Array = 101325.0,
        extra: Mapping[str, float | Array] | None = None,
    ) -> Stream:
        """Build an organic stream on this schema.

        Args:
            extractant_flow: **Total** extractant molar flow (mol/s), monomer
                basis, free plus bound.
            diluent_flow: Diluent molar flow (mol/s).
            element_flows: Loaded organic REE molar flows (mol/s).
            organic_acid: Co-extracted acid molar flow (mol/s).
            organic_water: Water in the organic phase (mol/s).
            T: Temperature (K).
            P: Pressure (Pa).
            extra: Additional species carried through untouched.

        Returns:
            A difflow :class:`~difflow.streams.Stream`.
        """
        element_flows = element_flows or {}
        flows: dict[str, float | Array] = {
            el: element_flows.get(el, 0.0) for el in self.elements
        }
        flows[self.extractant] = extractant_flow
        flows[self.diluent] = diluent_flow
        flows[self.organic_acid] = organic_acid
        flows[self.organic_water] = organic_water
        if extra:
            flows.update(extra)
        return make_stream(flows, T, P)

    def electroneutral_anion_flow(
        self,
        element_flows: Mapping[str, float | Array],
        acid: float | Array = 0.0,
        counter_ion: float | Array = 0.0,
    ) -> Array:
        """Anion flow that makes an aqueous stream electroneutral.

        An aqueous feed that is not electroneutral has no physical
        realisation, and handing one to the closed model produces a free
        proton concentration that silently absorbs the imbalance. Closing the
        anion here is the honest default.

        Args:
            element_flows: Molar flow per REE symbol (mol/s); every REE is
                taken as trivalent.
            acid: Free acid molar flow (mol/s).
            counter_ion: Counter-ion molar flow (mol/s).

        Returns:
            Anion molar flow (mol/s), ``= (3 sum RE + H + z_M M) / |z_X|``.
        """
        positive = jnp.asarray(acid, dtype=jnp.float64)
        positive = positive + self.counter_ion_charge * jnp.asarray(
            counter_ion, dtype=jnp.float64
        )
        for el in self.elements:
            positive = positive + 3.0 * jnp.asarray(
                element_flows.get(el, 0.0), dtype=jnp.float64
            )
        return positive / abs(self.anion_charge)
