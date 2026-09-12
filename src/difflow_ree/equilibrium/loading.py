"""Extractant loading and saturation models.

When the organic phase approaches saturation with extracted REE,
the effective distribution coefficient decreases.

Models:
- Langmuir isotherm for extractant capacity
- Loading correction factors for D values
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Loading Isotherm Models
# =============================================================================

@dataclass
class LoadingIsotherm:
    """Extractant loading isotherm model.

    Models the relationship between aqueous REE concentration
    and organic phase loading.

    The stoichiometry ``m`` is the single source of truth for capacity: the
    extractant balance in monomer equivalents is

        [HA]_total = [HA]_free + m * [RE-complex]

    so the maximum loading is ``1 / m`` mol REE per mol extractant and the
    free-extractant exponent in :meth:`apparent_D` is the same ``m``. Storing
    ``m`` rather than a separate ``max_loading`` literal makes it impossible
    for the two to disagree (#191). Use :func:`get_loading_isotherm` to build
    one with ``m`` taken from the extractant database.

    .. warning:: **Breaking API change in #191.** ``max_loading`` used to be a
       constructor field with the default 0.33, so ``LoadingIsotherm(
       max_loading=0.33)`` was valid. It is now a read-only property derived
       from ``m``, and that call raises ``TypeError``. Construct with
       ``LoadingIsotherm(m=...)`` instead: the old ``max_loading=0.33``
       becomes ``m=3.0``, and the D2EHPA/PC88A/Cyanex272 records that the
       YAML declares as three *dimers* become ``m=6.0``, halving the capacity
       that the 0.33 literal claimed. ``max_ree_conc`` therefore also halves
       for those extractants relative to any pre-#191 result.

    Attributes:
        m: Extractant monomer equivalents bound per mol REE. The database
            value is 6.0 for the acidic organophosphorus extractants
            (3 dimers) and 3.0 for TBP; the default here is the monomer
            stoichiometry 3.0.
        K_L: Langmuir constant (L/mol)
        extractant_conc: Extractant concentration (M)
    """
    m: float = 3.0  # monomer equivalents per REE; max_loading = 1/m (#191)
    K_L: float = 10.0  # Langmuir constant
    extractant_conc: float = 0.5  # M

    @property
    def max_loading(self) -> float:
        """Maximum REE loading capacity (mol REE / mol extractant).

        Derived as ``1 / m`` so capacity and stoichiometry cannot
        disagree (#191).
        """
        return 1.0 / self.m

    @property
    def max_ree_conc(self) -> float:
        """Maximum REE concentration in organic (M)."""
        return self.max_loading * self.extractant_conc

    def loading(self, c_aq: Array | float) -> Array:
        """Calculate organic phase loading (Langmuir isotherm).

        q = q_max * K_L * c / (1 + K_L * c)

        Args:
            c_aq: Aqueous phase REE concentration (M)

        Returns:
            Organic phase REE concentration (M)
        """
        c_aq = jnp.asarray(c_aq)
        q_max = self.max_ree_conc
        return q_max * self.K_L * c_aq / (1 + self.K_L * c_aq)

    def loading_fraction(self, c_org: Array | float) -> Array:
        """Calculate fraction of extractant capacity used.

        Args:
            c_org: Organic phase REE concentration (M)

        Returns:
            Loading fraction (0 to 1)
        """
        c_org = jnp.asarray(c_org)
        return c_org / self.max_ree_conc

    def apparent_D(
        self,
        D_infinite: Array | float,
        theta: Array | float,
    ) -> Array:
        """Calculate apparent D accounting for loading.

        At high loading, effective D decreases due to
        reduced free extractant concentration:

            D_app = D_inf * (1 - theta)^m

        where ``1 - theta`` is the fraction of extractant still free and
        ``m`` is the number of extractant monomers bound per REE.

        Caller's obligation (#189): ``theta`` is a **dimensionless** loading
        fraction, mol REE bound divided by mol REE capacity, equivalently

            theta = m * n_REE(organic) / n_extractant

        computed from quantities in the same units (both concentrations or
        both molar flows). Passing a concentration, a molar flow, or a flow
        ratio here is a units error: the result is raised to the m-th power,
        so the error is amplified. :meth:`loading_fraction` converts an
        organic REE *concentration* to this fraction.

        Note (#190): ``D_inf`` must not already carry a free-extractant
        depletion factor. ``REEDistribution.get_D`` does carry one (its
        ``n * log10([HA]/C_ref)`` term), which is why the stage path in
        ``difflow_ree.units.extraction`` no longer calls this method.

        Args:
            D_infinite: D at infinite dilution (zero loading)
            theta: Dimensionless loading fraction (0 = clean solvent,
                1 = extractant fully saturated)

        Returns:
            Apparent distribution coefficient
        """
        D_infinite = jnp.asarray(D_infinite)
        theta = jnp.asarray(theta)
        # Stoichiometry effect: each REE binds m extractant monomers (#191)
        return D_infinite * jnp.power(jnp.maximum(1 - theta, 0.01), self.m)


def langmuir_loading(
    c_aq: Array | float,
    q_max: float,
    K_L: float,
) -> Array:
    """Langmuir isotherm for extractant loading.

    q = q_max * K_L * c / (1 + K_L * c)

    Args:
        c_aq: Aqueous phase concentration (M)
        q_max: Maximum loading capacity (M in organic)
        K_L: Langmuir constant (L/mol)

    Returns:
        Organic phase concentration at equilibrium
    """
    c_aq = jnp.asarray(c_aq)
    return q_max * K_L * c_aq / (1 + K_L * c_aq)


def freundlich_loading(
    c_aq: Array | float,
    K_F: float,
    n: float,
) -> Array:
    """Freundlich isotherm (empirical).

    q = K_F * c^(1/n)

    Args:
        c_aq: Aqueous phase concentration (M)
        K_F: Freundlich constant
        n: Freundlich exponent (n > 1 for favorable isotherm)

    Returns:
        Organic phase concentration
    """
    c_aq = jnp.asarray(c_aq)
    return K_F * jnp.power(c_aq, 1/n)


def langmuir_freundlich_loading(
    c_aq: Array | float,
    q_max: float,
    K_LF: float,
    n: float,
) -> Array:
    """Langmuir-Freundlich (Sips) isotherm.

    q = q_max * (K_LF * c)^n / (1 + (K_LF * c)^n)

    Combines features of both models.

    Args:
        c_aq: Aqueous phase concentration (M)
        q_max: Maximum loading capacity (M)
        K_LF: Langmuir-Freundlich constant
        n: Heterogeneity parameter

    Returns:
        Organic phase concentration
    """
    c_aq = jnp.asarray(c_aq)
    Kc_n = jnp.power(K_LF * c_aq, n)
    return q_max * Kc_n / (1 + Kc_n)


# =============================================================================
# Loading Correction for Multi-Component Systems
# =============================================================================

def loading_correction(
    D_values: dict[str, Array],
    c_org: dict[str, Array | float],
    isotherm: LoadingIsotherm,
) -> dict[str, Array]:
    """Apply loading correction to D values for all elements.

    In multi-component systems, total loading affects all D values:

        D_app,i = D_inf,i * (1 - theta_total)^m

    with ``theta_total`` the summed organic REE concentration divided by
    ``isotherm.max_ree_conc`` and ``m = isotherm.m``. It is the multi-component
    form of :meth:`LoadingIsotherm.apparent_D` and carries the same caller
    obligations: ``D_values`` must not already carry a free-extractant
    depletion factor (#190), and the free fraction is floored at 0.01 so the
    correction saturates rather than reaching zero.

    .. warning:: **Breaking change in #191.** Two things moved, and the output
       changes by orders of magnitude for the acidic extractants:

       * the exponent was the literal ``3``; it is now ``isotherm.m``, which
         is 6.0 for D2EHPA, PC88A and Cyanex272 (three dimers) and 3.0 for TBP;
       * ``theta_total`` is computed against ``isotherm.max_ree_conc``, which
         itself halved for those extractants because ``max_loading`` went from
         the 0.33 literal to ``1/m = 1/6``.

       At an organic REE concentration of 0.0413 M in 0.5 M D2EHPA the
       correction was ``(1 - 0.25030)^3 = 0.42136`` and is now
       ``(1 - 0.49560)^6 = 0.016468``, a factor of 25.59 smaller. Callers that
       calibrated against the old form must refit.

    Args:
        D_values: Dictionary of infinite-dilution D values
        c_org: Current organic concentrations for each element (M), keyed by
            element; keys need not match ``D_values``
        isotherm: Loading isotherm model

    Returns:
        Corrected D values accounting for total loading, with the same keys as
        ``D_values``

    Example:
        >>> from difflow_ree.equilibrium.loading import (
        ...     get_loading_isotherm, loading_correction)
        >>> iso = get_loading_isotherm("D2EHPA", 0.5)   # m = 6, cap 1/12 M
        >>> out = loading_correction({"Nd": 10.0}, {"Nd": 0.0413}, iso)
        >>> float(out["Nd"])                            # doctest: +ELLIPSIS
        0.1646...
    """
    # Calculate total loading fraction
    total_c_org = sum(jnp.asarray(c) for c in c_org.values())
    theta_total = isotherm.loading_fraction(total_c_org)

    # Apply correction to all elements
    # D_app = D_inf * (1 - theta)^m, with m the monomer stoichiometry taken
    # from the isotherm rather than hard-coded (#191)
    correction = jnp.power(jnp.maximum(1 - theta_total, 0.01), isotherm.m)

    return {elem: D * correction for elem, D in D_values.items()}


def competitive_langmuir(
    c_aq: dict[str, Array | float],
    K_L: dict[str, float],
    q_max: float,
) -> dict[str, Array]:
    """Competitive Langmuir isotherm for multi-component system.

    q_i = q_max * K_i * c_i / (1 + sum_j(K_j * c_j))

    Args:
        c_aq: Aqueous concentrations for each species
        K_L: Langmuir constants for each species
        q_max: Total maximum capacity (shared)

    Returns:
        Organic phase concentrations for each species
    """
    # Calculate denominator: 1 + sum(K_j * c_j)
    denominator = 1.0
    for species, c in c_aq.items():
        denominator = denominator + K_L[species] * jnp.asarray(c)

    # Calculate loading for each species
    q_org = {}
    for species, c in c_aq.items():
        q_org[species] = q_max * K_L[species] * jnp.asarray(c) / denominator

    return q_org


# =============================================================================
# Extractant Capacity Data
# =============================================================================

# Langmuir constants, DERIVED (#268). Capacity and stoichiometry are not
# duplicated here either: they come from the extractant record in
# ``difflow_ree.database`` (``Extractant.monomers_per_ree`` /
# ``Extractant.max_loading``), which reads the declared extraction mechanism
# from ``data/extractants.yaml``. There is one source of truth for m (#191).
#
# BREAKING CHANGE (#191): the per-extractant ``"stoichiometry"`` and
# ``"max_loading"`` keys this dict used to carry were deleted, not merely
# re-valued. ``EXTRACTANT_CAPACITIES["D2EHPA"]["max_loading"]`` raises
# ``KeyError``; read the record instead.
#
# CHANGE (#268): ``typical_K_L`` is no longer a literal either. It used to be
# a second extractant table that had to be hand-synced with the YAML, and it
# had drifted: asked what single pH would reconcile each stored table with the
# coefficients it was derived from, D2EHPA, PC88A and Cyanex272 came back with
# rms log10 residuals of 0.62, 0.85 and 0.92 at their best-fitting pH. No pH
# reconciled them -- they were not evaluated at a different condition, they
# described an extractant the database no longer contained. Nothing failed
# when that happened, because a *missing* extractant was tested for and a
# *stale* one was not.
#
# So they are computed, at the record's own declared reference conditions:
#
#     q = q_max K_L c  and  q = D c  in the trace limit, so
#     K_L = D(reference) / q_max,   q_max = [HA]_ref / monomers_per_ree
#
# The mapping below still reads like the dict it replaces --- indexing,
# iteration and ``.items()`` all work --- so call sites did not have to
# change. What changed is that there is nothing left to hand-sync.


def typical_K_L(
    extractant: str,
    concentration: float | None = None,
) -> dict[str, float]:
    """Langmuir constants for an extractant, from its own correlation (#268).

    In the trace limit the Langmuir isotherm ``q = q_max K_L c / (1 + K_L c)``
    is ``q = q_max K_L c``, and the distribution ratio gives ``q = D c``.  So

        K_L = D(reference conditions) / q_max,
        q_max = [HA]_ref / monomers_per_ree

    with the reference conditions the ones the record declares:
    ``reference_concentration`` together with ``reference_pH`` for a cation
    exchanger or ``reference_nitrate`` for a solvating one.  A record whose
    driving variable has no declared reference cannot be derived, and says so
    rather than falling back to a plausible number.

    Args:
        extractant: Extractant name.
        concentration: Extractant charge (M) to derive at. ``None`` (default)
            uses the record's own ``reference_concentration``, which is the
            declared basis; pass a value to ask what the constants would be at
            a different charge.

    Returns:
        ``{element: K_L}`` for every element the record's coefficient block
        covers.

    Raises:
        KeyError: If the extractant is not in the database.
        ValueError: If the record declares no reference for its driving
            variable.
    """
    from difflow_ree.database import get_extractant
    from difflow_ree.equilibrium.distribution import REEDistribution

    record = get_extractant(extractant)
    conc = (record.reference_concentration if concentration is None
            else float(concentration))

    if record.mechanism == "solvating":
        driving = {"nitrate_conc": record.reference_nitrate, "pH": None}
        missing = record.reference_nitrate is None
        which = "reference_nitrate"
        block = record.nitrate_coefficients
    else:
        driving = {"pH": record.reference_pH}
        missing = record.reference_pH is None
        which = "reference_pH"
        block = record.ph_coefficients

    if missing:
        raise ValueError(
            f"Extractant {extractant!r} declares no {which}, so there is no "
            "condition at which to evaluate its correlation and the Langmuir "
            "constants cannot be derived (#268). Add one to its record in "
            "data/extractants.yaml; it is a declared basis, and every derived "
            "quantity moves with it."
        )
    if not block:
        raise ValueError(
            f"Extractant {extractant!r} carries no coefficient block for "
            f"mechanism {record.mechanism!r}, so nothing can be derived "
            "from it."
        )

    elements = tuple(block)
    dist = REEDistribution(
        extractant=extractant, elements=elements, concentration=conc,
        **{k: v for k, v in driving.items() if k != "pH"},
    )
    q_max = conc / record.monomers_per_ree
    return {
        element: float(dist.get_D(element, **driving)) / q_max
        for element in elements
    }


class _DerivedCapacities(Mapping):
    """``EXTRACTANT_CAPACITIES``, computed rather than stored (#268).

    Reads like the dict it replaces. Values are memoised per extractant,
    since deriving them runs the correlation once per element and
    ``get_loading_isotherm`` is called per stage construction.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def _names(self) -> list[str]:
        from difflow_ree.database import get_extractant_database
        return get_extractant_database().list_extractants()

    def __getitem__(self, extractant: str) -> dict:
        if extractant not in self._cache:
            if extractant not in self._names():
                raise KeyError(extractant)
            self._cache[extractant] = {"typical_K_L": typical_K_L(extractant)}
        return self._cache[extractant]

    def __iter__(self):
        return iter(self._names())

    def __len__(self) -> int:
        return len(self._names())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<derived EXTRACTANT_CAPACITIES for {self._names()}>"


#: Langmuir constants per extractant, derived from the record (#268).
EXTRACTANT_CAPACITIES = _DerivedCapacities()


def get_loading_isotherm(
    extractant: str,
    concentration: float = 0.5,
) -> LoadingIsotherm:
    """Create loading isotherm for specified extractant.

    The stoichiometry ``m`` (and hence the capacity ``1/m``) is read from the
    extractant database, which derives it from the extraction mechanism
    declared in ``data/extractants.yaml``, rather than from a hard-coded
    literal here (#191). ``EXTRACTANT_CAPACITIES`` supplies only the Langmuir
    constants.

    Args:
        extractant: Extractant name
        concentration: Extractant concentration (M)

    Returns:
        LoadingIsotherm instance
    """
    if extractant not in EXTRACTANT_CAPACITIES:
        raise ValueError(f"Unknown extractant: {extractant}")

    from difflow_ree.database import get_extractant

    data = EXTRACTANT_CAPACITIES[extractant]
    # Use average K_L across elements
    avg_K_L = sum(data["typical_K_L"].values()) / len(data["typical_K_L"])

    return LoadingIsotherm(
        m=get_extractant(extractant).monomers_per_ree,
        K_L=avg_K_L,
        extractant_conc=concentration,
    )


def get_competitive_K_L(extractant: str) -> dict[str, float]:
    """Get Langmuir constants for competitive adsorption.

    Args:
        extractant: Extractant name

    Returns:
        Dictionary of K_L values for each REE
    """
    if extractant not in EXTRACTANT_CAPACITIES:
        raise ValueError(f"Unknown extractant: {extractant}")
    return EXTRACTANT_CAPACITIES[extractant]["typical_K_L"].copy()
