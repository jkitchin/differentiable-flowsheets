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
- ``Na`` is the aqueous counter-ion and ``Na_org`` (#197) is the counter-ion
  bound to the organic phase as the salt of the extractant -- a *saponified*
  solvent. Both carry moles of counter-ion, so the pair is a partition and the
  counter-ion total is conserved across it.
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

#: Divalent counter-ions (#197). Magnesia saponification is a real industrial
#: route and it is kept in its own table for the same reason ``SO4`` is kept
#: out of the shipped reaction networks: every network in
#: ``reaction_networks.yaml`` declares a counter-ion component of charge +1 and
#: a species that holds one equivalent of base, and a divalent counter-ion is a
#: different tableau -- half as many formula units, twice the equivalents each.
#: :func:`difflow_ree.equilibrium.saponification.divalent_counter_ion_template`
#: derives that tableau from the shipped monovalent one, so the stoichiometry
#: still has a single source. Using one of these keys with a monovalent network
#: raises rather than quietly running with the wrong charge.
DIVALENT_COUNTER_ION_CHARGES = {
    "Mg": 2,
    "Ca": 2,
}

#: Every counter-ion key the schema accepts, monovalent and divalent.
ALL_COUNTER_ION_CHARGES = {
    **COUNTER_ION_CHARGES,
    **DIVALENT_COUNTER_ION_CHARGES,
}


def counter_ion_charge_of(name: str) -> int:
    """Formal charge of a counter-ion key.

    Args:
        name: Counter-ion key, e.g. ``"Na"`` or ``"Mg"``.

    Returns:
        The formal (positive) charge.

    Raises:
        KeyError: If the key is not recognised.

    Example:
        >>> counter_ion_charge_of("Na"), counter_ion_charge_of("Mg")
        (1, 2)
    """
    if name not in ALL_COUNTER_ION_CHARGES:
        raise KeyError(
            f"Unknown counter_ion {name!r}. Recognised: "
            f"{sorted(ALL_COUNTER_ION_CHARGES)} (#196, #197)."
        )
    return ALL_COUNTER_ION_CHARGES[name]


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
        ('Nd', 'Dy', 'D2EHPA', 'kerosene', 'H_org', 'H2O_org', 'Na_org')
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
        if (
            self.counter_ion is not None
            and self.counter_ion not in ALL_COUNTER_ION_CHARGES
        ):
            raise ValueError(
                f"Unknown counter_ion {self.counter_ion!r}. Recognised: "
                f"{sorted(ALL_COUNTER_ION_CHARGES)}, or None (#196, #197)."
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
        return ALL_COUNTER_ION_CHARGES[self.counter_ion]

    @property
    def organic_counter_ion(self) -> str | None:
        """Key carrying counter-ion bound to the organic phase (#197).

        A saponified solvent holds its counter-ion as the salt of the
        extractant, e.g. ``NaA`` in ``Na_org``. The key carries **moles of
        counter-ion**, not moles of the salt species, so it means the same
        thing for a divalent counter-ion, which occupies two extractant
        equivalents per ion.

        Returns:
            ``"<counter_ion>_org"``, or None when the schema has no
            counter-ion.
        """
        if self.counter_ion is None:
            return None
        return f"{self.counter_ion}_org"

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
            co-extracted acid, water in organic, and the saponified
            counter-ion (#197) when the schema has one.
        """
        keys = list(self.elements) + [
            self.extractant, self.diluent, self.organic_acid,
            self.organic_water,
        ]
        if self.organic_counter_ion is not None:
            keys.append(self.organic_counter_ion)
        return tuple(keys)

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
            self.organic_water, self.organic_counter_ion,
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
        counter_ion: float | Array | None = None,
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
            counter_ion: Counter-ion bound to the organic phase (mol/s), i.e.
                a saponified solvent (#197). None (the default) omits the key
                entirely, so an unsaponified stream is exactly what it always
                was; pass 0.0 to declare an explicitly unsaponified solvent.
                See :meth:`saponified_organic` for the usual way to build one.
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
        if counter_ion is not None:
            if self.organic_counter_ion is None:
                raise ValueError(
                    "make_organic was given a counter_ion loading but the "
                    "schema declares counter_ion=None, so there is no key to "
                    "carry it and no component to conserve it (#197)."
                )
            flows[self.organic_counter_ion] = counter_ion
        if extra:
            flows.update(extra)
        return make_stream(flows, T, P)

    def saponified_organic(
        self,
        extractant_flow: float | Array,
        saponification_degree: float | Array,
        monomers_per_component: float = 2.0,
        counter_ion_charge: int | None = None,
        **kwargs,
    ) -> Stream:
        """Build a pre-neutralized organic stream from a degree (#197).

        The saponification degree is the fraction of the extractant's
        exchangeable protons that have been replaced by the counter-ion, so
        the counter-ion the stream carries is

        .. math::

            n_M = \\frac{S\\,F_{\\mathrm{ext}}}
                        {m\\,z_M}

        with ``m`` the monomers per extractant component (2 for a dimeric
        extractant) and ``z_M`` the counter-ion charge: one equivalent per
        extractant component, ``z_M`` equivalents per counter-ion.

        Args:
            extractant_flow: **Total** extractant molar flow (mol/s), monomer
                basis.
            saponification_degree: Fraction neutralized, 0 to 1. May be a
                tracer -- this is the primary manipulated variable (#197).
            monomers_per_component: Extractant monomers per component; 2 for
                the dimeric acidic organophosphorus extractants. Read it off
                ``network.monomers_per_component`` rather than guessing.
            counter_ion_charge: Counter-ion charge; None uses the schema's.
            **kwargs: Forwarded to :meth:`make_organic`.

        Returns:
            A difflow :class:`~difflow.streams.Stream` carrying the bound
            counter-ion.

        Example:
            >>> s = REEStreamSchema(elements=("Nd",), extractant="D2EHPA")
            >>> org = s.saponified_organic(0.5, 0.4)
            >>> round(float(org["F_Na_org"]), 4)
            0.1
        """
        if counter_ion_charge is None:
            counter_ion_charge = self.counter_ion_charge
        if not counter_ion_charge:
            raise ValueError(
                "saponified_organic needs a counter-ion: the schema declares "
                "counter_ion=None, so the base has no cation to leave behind "
                "in the organic phase (#197)."
            )
        n_M = (
            jnp.asarray(saponification_degree, dtype=jnp.float64)
            * jnp.asarray(extractant_flow, dtype=jnp.float64)
            / (float(monomers_per_component) * float(counter_ion_charge))
        )
        return self.make_organic(extractant_flow, counter_ion=n_M, **kwargs)

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
