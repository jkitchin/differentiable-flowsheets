"""Extractant loading and saturation models.

When the organic phase approaches saturation with extracted REE,
the effective distribution coefficient decreases.

Models:
- Langmuir isotherm for extractant capacity
- Loading correction factors for D values
"""

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

# Langmuir constants only. Capacity and stoichiometry are NOT duplicated here:
# they are derived from the extractant record in ``difflow_ree.database``
# (``Extractant.monomers_per_ree`` / ``Extractant.max_loading``), which reads
# the declared extraction mechanism from ``data/extractants.yaml``. There is one
# source of truth for m (#191).
#
# BREAKING CHANGE (#191): this module-level dict is public, and the per-
# extractant ``"stoichiometry"`` and ``"max_loading"`` keys it used to carry
# were deleted, not merely re-valued. ``EXTRACTANT_CAPACITIES["D2EHPA"]
# ["max_loading"]`` now raises ``KeyError``; read
# ``difflow_ree.database.get_extractant("D2EHPA").max_loading`` (or
# ``.monomers_per_ree``) instead. The values also changed: the deleted
# ``max_loading`` literal was 0.33 for the acidic extractants where the
# database now derives 1/6, because the YAML declares three *dimers*.
EXTRACTANT_CAPACITIES = {
    "D2EHPA": {
        "typical_K_L": {
            "La": 5.0,
            "Ce": 8.0,
            "Pr": 12.0,
            "Nd": 15.0,
            "Sm": 30.0,
            "Eu": 40.0,
            "Gd": 50.0,
            "Tb": 80.0,
            "Dy": 100.0,
            "Y": 70.0,
        },
    },
    "PC88A": {
        "typical_K_L": {
            "La": 3.0,
            "Ce": 5.0,
            "Pr": 8.0,
            "Nd": 12.0,
            "Sm": 25.0,
            "Eu": 35.0,
            "Gd": 45.0,
            "Tb": 70.0,
            "Dy": 90.0,
            "Y": 60.0,
        },
    },
    "Cyanex272": {
        "typical_K_L": {
            "La": 2.0,
            "Ce": 3.0,
            "Pr": 5.0,
            "Nd": 8.0,
            "Sm": 15.0,
            "Eu": 20.0,
            "Gd": 25.0,
            "Tb": 40.0,
            "Dy": 50.0,
            "Y": 35.0,
        },
    },
    "TBP": {
        "typical_K_L": {
            "La": 2.0,
            "Ce": 2.5,
            "Pr": 3.0,
            "Nd": 3.5,
            "Sm": 5.0,
            "Eu": 6.0,
            "Gd": 7.0,
            "Tb": 9.0,
            "Dy": 10.0,
            "Y": 8.0,
        },
    },
}


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
