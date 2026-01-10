"""Extraction kinetics for REE solvent extraction.

Provides models for extraction/stripping reaction kinetics
including approach to equilibrium and residence time effects.

All functions use JAX numpy for automatic differentiation.

References:
    Danesi PR (1984). Solvent Extr Ion Exch 2:29.
    Sekine T, Hasegawa Y (1977). Solvent Extraction Chemistry.
"""

__all__ = [
    "forward_extraction_rate",
    "reverse_extraction_rate",
    "net_extraction_rate",
    "approach_to_equilibrium",
    "stage_residence_time",
    "ExtractionKineticsModel",
    "ExtractionKineticsParams",
    "get_extraction_kinetics_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Extraction Rate Functions
# =============================================================================

def forward_extraction_rate(
    C_aq: Array | float,
    C_extractant: Array | float,
    k_f: Array | float,
    n: Array | float = 3.0,
) -> Array:
    """Calculate forward extraction rate.

    r_f = k_f * C_aq * C_extractant^n

    For REE extraction with acidic extractants:
    REE³⁺ + 3HA → REE(A)₃ + 3H⁺

    Args:
        C_aq: Aqueous REE concentration (M)
        C_extractant: Extractant concentration (M)
        k_f: Forward rate constant
        n: Reaction order w.r.t. extractant

    Returns:
        Forward extraction rate (M/s)
    """
    C_aq = jnp.asarray(C_aq)
    C_extractant = jnp.asarray(C_extractant)
    k_f = jnp.asarray(k_f)
    n = jnp.asarray(n)

    return k_f * C_aq * jnp.power(C_extractant, n)


def reverse_extraction_rate(
    C_org: Array | float,
    C_H: Array | float,
    k_r: Array | float,
    n: Array | float = 3.0,
) -> Array:
    """Calculate reverse (stripping) rate.

    r_r = k_r * C_org * C_H^n

    Args:
        C_org: Organic REE concentration (M)
        C_H: Hydrogen ion concentration (M)
        k_r: Reverse rate constant
        n: Reaction order w.r.t. H+

    Returns:
        Reverse extraction rate (M/s)
    """
    C_org = jnp.asarray(C_org)
    C_H = jnp.asarray(C_H)
    k_r = jnp.asarray(k_r)
    n = jnp.asarray(n)

    return k_r * C_org * jnp.power(C_H, n)


def net_extraction_rate(
    C_aq: Array | float,
    C_org: Array | float,
    C_extractant: Array | float,
    C_H: Array | float,
    k_f: Array | float,
    k_r: Array | float,
    n: Array | float = 3.0,
) -> Array:
    """Calculate net extraction rate.

    r_net = r_f - r_r

    Args:
        C_aq: Aqueous REE concentration (M)
        C_org: Organic REE concentration (M)
        C_extractant: Extractant concentration (M)
        C_H: H+ concentration (M)
        k_f: Forward rate constant
        k_r: Reverse rate constant
        n: Reaction order

    Returns:
        Net extraction rate (M/s), positive = extraction
    """
    r_f = forward_extraction_rate(C_aq, C_extractant, k_f, n)
    r_r = reverse_extraction_rate(C_org, C_H, k_r, n)
    return r_f - r_r


def approach_to_equilibrium(
    t: Array | float,
    k_overall: Array | float,
) -> Array:
    """Calculate approach to equilibrium fraction.

    f = 1 - exp(-k * t)

    Args:
        t: Contact time (s)
        k_overall: Overall rate constant (1/s)

    Returns:
        Fraction of equilibrium achieved (0-1)
    """
    t = jnp.asarray(t)
    k_overall = jnp.asarray(k_overall)

    return 1 - jnp.exp(-k_overall * t)


def stage_residence_time(
    V: Array | float,
    Q: Array | float,
) -> Array:
    """Calculate stage residence time.

    tau = V / Q

    Args:
        V: Stage volume (m³)
        Q: Volumetric flow rate (m³/s)

    Returns:
        Residence time (s)
    """
    V = jnp.asarray(V)
    Q = jnp.asarray(Q)

    return V / Q


# =============================================================================
# Extraction Kinetics Model
# =============================================================================

@dataclass(repr=False)
class ExtractionKineticsParams(ParamsMixin):
    """Parameters for extraction kinetics.

    Attributes:
        k_f: Forward rate constant (1/s or M^-n/s)
        k_r: Reverse rate constant (1/s or M^-n/s)
        n: Reaction order w.r.t. extractant/H+
        C_extractant: Extractant concentration (M)
        E_a: Activation energy (J/mol)
        T_ref: Reference temperature (K)
    """
    k_f: float | Array = 0.1  # Fast extraction typical
    k_r: float | Array = 0.01
    n: float | Array = 3.0  # REE³⁺ stoichiometry
    C_extractant: float | Array = 0.5
    E_a: float | Array = 50000.0  # J/mol typical
    T_ref: float | Array = 298.15


class ExtractionKineticsModel:
    """Unified extraction kinetics model.

    Example:
        >>> model = ExtractionKineticsModel(ExtractionKineticsParams())
        >>> rate = model.net_rate(C_aq=0.01, C_org=0.0, pH=3.0)
    """

    def __init__(self, params: ExtractionKineticsParams):
        self.params = params

    def rate_at_T(
        self,
        T: Array | float,
        k_ref: Array | float,
    ) -> Array:
        """Arrhenius temperature correction."""
        p = self.params
        T = jnp.asarray(T)
        k_ref = jnp.asarray(k_ref)

        R = 8.314
        return k_ref * jnp.exp(-p.E_a / R * (1/T - 1/p.T_ref))

    def net_rate(
        self,
        C_aq: Array | float,
        C_org: Array | float,
        pH: Array | float,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate net extraction rate."""
        p = self.params
        C_H = jnp.power(10.0, -pH)

        k_f_T = self.rate_at_T(T, p.k_f)
        k_r_T = self.rate_at_T(T, p.k_r)

        return net_extraction_rate(
            C_aq, C_org, p.C_extractant, C_H, k_f_T, k_r_T, p.n
        )

    def equilibrium_D(
        self,
        pH: Array | float,
    ) -> Array:
        """Calculate equilibrium D from kinetics (k_f/k_r)."""
        p = self.params
        C_H = jnp.power(10.0, -pH)

        # At equilibrium: k_f * C_aq * C_ex^n = k_r * C_org * C_H^n
        # D = C_org/C_aq = k_f/k_r * (C_ex/C_H)^n
        return p.k_f / p.k_r * jnp.power(p.C_extractant / C_H, p.n)

    def time_to_equilibrium(
        self,
        target_approach: float = 0.95,
    ) -> Array:
        """Estimate time to reach target approach to equilibrium."""
        # Simplified: use k_f as overall rate
        k = self.params.k_f
        return -jnp.log(1 - target_approach) / k


def get_extraction_kinetics_model(**kwargs) -> ExtractionKineticsModel:
    """Create extraction kinetics model."""
    params = ExtractionKineticsParams(**kwargs)
    return ExtractionKineticsModel(params)
