"""Cell growth kinetic models.

Provides modular implementations of common growth rate expressions
used in bioreactor modeling.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Monod: Single substrate limitation
- Monod with inhibition: Substrate/product inhibition
- Contois: Cell density dependent
- Logistic: Carrying capacity limited
- Tessier: Exponential saturation
- Moser: Power-law saturation
- Andrews: Substrate inhibition (Haldane)

References:
    Monod J (1949). The Growth of Bacterial Cultures.
        Annu Rev Microbiol 3:371-394.
    Contois DE (1959). J Gen Microbiol 21:40-50.
    Andrews JF (1968). Biotechnol Bioeng 10:707-723.
"""

__all__ = [
    "monod",
    "monod_inhibition",
    "contois",
    "logistic",
    "tessier",
    "moser",
    "andrews",
    "death_rate",
    "net_growth_rate",
    "GrowthModel",
    "get_growth_model",
]

from dataclasses import dataclass
from typing import Callable, Literal

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Basic Growth Models
# =============================================================================

def monod(
    S: Array | float,
    mu_max: Array | float,
    K_s: Array | float,
) -> Array:
    """Monod growth kinetics.

    μ = μ_max * S / (K_s + S)

    Classic single-substrate limitation model.

    Args:
        S: Substrate concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_s: Half-saturation constant (g/L)

    Returns:
        Specific growth rate μ (1/h)

    Example:
        >>> mu = monod(S=2.0, mu_max=0.4, K_s=0.5)
        >>> # At S >> K_s, mu approaches mu_max
    """
    S = jnp.asarray(S)
    mu_max = jnp.asarray(mu_max)
    K_s = jnp.asarray(K_s)

    return mu_max * S / (K_s + S)


def monod_inhibition(
    S: Array | float,
    mu_max: Array | float,
    K_s: Array | float,
    K_i: Array | float = None,
    P: Array | float = None,
    K_p: Array | float = None,
) -> Array:
    """Monod kinetics with substrate and/or product inhibition.

    μ = μ_max * S / (K_s + S + S²/K_i) * (1 - P/K_p)

    Args:
        S: Substrate concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_s: Half-saturation constant (g/L)
        K_i: Substrate inhibition constant (g/L), optional
        P: Product concentration (g/L), optional
        K_p: Product inhibition constant (g/L), optional

    Returns:
        Specific growth rate μ (1/h)
    """
    S = jnp.asarray(S)
    mu_max = jnp.asarray(mu_max)
    K_s = jnp.asarray(K_s)

    # Base Monod term
    denom = K_s + S

    # Substrate inhibition (Andrews/Haldane)
    if K_i is not None:
        K_i = jnp.asarray(K_i)
        denom = denom + S * S / K_i

    mu = mu_max * S / denom

    # Product inhibition
    if P is not None and K_p is not None:
        P = jnp.asarray(P)
        K_p = jnp.asarray(K_p)
        inhibition = jnp.maximum(0.0, 1.0 - P / K_p)
        mu = mu * inhibition

    return mu


def contois(
    S: Array | float,
    X: Array | float,
    mu_max: Array | float,
    K_sx: Array | float,
) -> Array:
    """Contois growth kinetics.

    μ = μ_max * S / (K_sx * X + S)

    Cell density dependent model - K_s varies with biomass.

    Args:
        S: Substrate concentration (g/L)
        X: Cell concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_sx: Contois saturation constant (g substrate/g cells)

    Returns:
        Specific growth rate μ (1/h)
    """
    S = jnp.asarray(S)
    X = jnp.asarray(X)
    mu_max = jnp.asarray(mu_max)
    K_sx = jnp.asarray(K_sx)

    return mu_max * S / (K_sx * X + S)


def logistic(
    X: Array | float,
    mu_max: Array | float,
    X_max: Array | float,
) -> Array:
    """Logistic growth kinetics.

    μ = μ_max * (1 - X/X_max)

    Carrying capacity limited growth, substrate-independent.

    Args:
        X: Cell concentration (g/L or cells/mL)
        mu_max: Maximum specific growth rate (1/h)
        X_max: Maximum cell density (carrying capacity)

    Returns:
        Specific growth rate μ (1/h)
    """
    X = jnp.asarray(X)
    mu_max = jnp.asarray(mu_max)
    X_max = jnp.asarray(X_max)

    return mu_max * jnp.maximum(0.0, 1.0 - X / X_max)


def tessier(
    S: Array | float,
    mu_max: Array | float,
    K_s: Array | float,
) -> Array:
    """Tessier growth kinetics.

    μ = μ_max * (1 - exp(-S/K_s))

    Exponential saturation model.

    Args:
        S: Substrate concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_s: Saturation constant (g/L)

    Returns:
        Specific growth rate μ (1/h)
    """
    S = jnp.asarray(S)
    mu_max = jnp.asarray(mu_max)
    K_s = jnp.asarray(K_s)

    return mu_max * (1.0 - jnp.exp(-S / K_s))


def moser(
    S: Array | float,
    mu_max: Array | float,
    K_s: Array | float,
    n: Array | float = 2.0,
) -> Array:
    """Moser growth kinetics.

    μ = μ_max * S^n / (K_s^n + S^n)

    Power-law generalization of Monod.

    Args:
        S: Substrate concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_s: Half-saturation constant (g/L)
        n: Cooperativity exponent (default 2)

    Returns:
        Specific growth rate μ (1/h)
    """
    S = jnp.asarray(S)
    mu_max = jnp.asarray(mu_max)
    K_s = jnp.asarray(K_s)
    n = jnp.asarray(n)

    S_n = jnp.power(S, n)
    K_n = jnp.power(K_s, n)

    return mu_max * S_n / (K_n + S_n)


def andrews(
    S: Array | float,
    mu_max: Array | float,
    K_s: Array | float,
    K_i: Array | float,
) -> Array:
    """Andrews (Haldane) substrate inhibition kinetics.

    μ = μ_max * S / (K_s + S + S²/K_i)

    Classic substrate inhibition model with optimal growth rate
    at intermediate substrate concentrations.

    Args:
        S: Substrate concentration (g/L)
        mu_max: Maximum specific growth rate (1/h)
        K_s: Half-saturation constant (g/L)
        K_i: Inhibition constant (g/L)

    Returns:
        Specific growth rate μ (1/h)

    Notes:
        Optimal substrate concentration: S_opt = sqrt(K_s * K_i)
        Maximum achievable rate: mu_opt = mu_max / (1 + 2*sqrt(K_s/K_i))
    """
    S = jnp.asarray(S)
    mu_max = jnp.asarray(mu_max)
    K_s = jnp.asarray(K_s)
    K_i = jnp.asarray(K_i)

    return mu_max * S / (K_s + S + S * S / K_i)


# =============================================================================
# Death Rate and Net Growth
# =============================================================================

def death_rate(
    X: Array | float = None,
    S: Array | float = None,
    k_d: Array | float = 0.01,
    S_crit: Array | float = None,
) -> Array:
    """Calculate cell death rate.

    k_d_eff = k_d * (1 + optional substrate limitation term)

    Args:
        X: Cell concentration (not used in basic model)
        S: Substrate concentration (g/L), optional
        k_d: Basal death rate constant (1/h)
        S_crit: Critical substrate below which death accelerates (g/L)

    Returns:
        Effective death rate (1/h)
    """
    k_d = jnp.asarray(k_d)

    if S is not None and S_crit is not None:
        S = jnp.asarray(S)
        S_crit = jnp.asarray(S_crit)
        # Death rate increases at low substrate
        starvation_factor = 1.0 + jnp.exp(-S / S_crit)
        return k_d * starvation_factor

    return k_d


def net_growth_rate(
    mu: Array | float,
    k_d: Array | float,
) -> Array:
    """Calculate net specific growth rate.

    μ_net = μ - k_d

    Args:
        mu: Specific growth rate (1/h)
        k_d: Death rate (1/h)

    Returns:
        Net specific growth rate (1/h)
    """
    mu = jnp.asarray(mu)
    k_d = jnp.asarray(k_d)
    return mu - k_d


# =============================================================================
# Growth Model Class
# =============================================================================

@dataclass(repr=False)
class GrowthModelParams(ParamsMixin):
    """Parameters for growth model.

    Attributes:
        model: Model type
        mu_max: Maximum specific growth rate (1/h)
        K_s: Half-saturation constant (g/L)
        K_i: Substrate inhibition constant (g/L)
        K_p: Product inhibition constant (g/L)
        X_max: Maximum cell density (g/L)
        k_d: Death rate constant (1/h)
    """
    model: str = "monod"
    mu_max: float | Array = 0.4
    K_s: float | Array = 0.5
    K_i: float | Array = None
    K_p: float | Array = None
    X_max: float | Array = None
    k_d: float | Array = 0.01


class GrowthModel:
    """Unified interface for growth kinetic models.

    Example:
        >>> model = GrowthModel(GrowthModelParams(
        ...     model="monod",
        ...     mu_max=0.4,
        ...     K_s=0.5,
        ... ))
        >>> mu = model(S=2.0)
    """

    def __init__(self, params: GrowthModelParams):
        """Initialize growth model.

        Args:
            params: Growth model parameters
        """
        self.params = params

    def __call__(
        self,
        S: Array | float,
        X: Array | float = None,
        P: Array | float = None,
    ) -> Array:
        """Calculate specific growth rate.

        Args:
            S: Substrate concentration (g/L)
            X: Cell concentration (g/L), needed for Contois/logistic
            P: Product concentration (g/L), needed for product inhibition

        Returns:
            Specific growth rate μ (1/h)
        """
        p = self.params

        if p.model == "monod":
            mu = monod(S, p.mu_max, p.K_s)
        elif p.model == "monod_inhibition":
            mu = monod_inhibition(S, p.mu_max, p.K_s, p.K_i, P, p.K_p)
        elif p.model == "contois":
            if X is None:
                raise ValueError("Contois model requires X")
            mu = contois(S, X, p.mu_max, p.K_s)
        elif p.model == "logistic":
            if X is None:
                raise ValueError("Logistic model requires X")
            mu = logistic(X, p.mu_max, p.X_max)
        elif p.model == "andrews":
            mu = andrews(S, p.mu_max, p.K_s, p.K_i)
        elif p.model == "tessier":
            mu = tessier(S, p.mu_max, p.K_s)
        else:
            raise ValueError(f"Unknown model: {p.model}")

        return mu

    def net_rate(
        self,
        S: Array | float,
        X: Array | float = None,
        P: Array | float = None,
    ) -> Array:
        """Calculate net growth rate (mu - k_d)."""
        mu = self(S, X, P)
        return net_growth_rate(mu, self.params.k_d)


def get_growth_model(
    model_type: str = "monod",
    **kwargs,
) -> GrowthModel:
    """Create growth model with specified type.

    Args:
        model_type: Model name
        **kwargs: Model parameters

    Returns:
        GrowthModel instance
    """
    params = GrowthModelParams(model=model_type, **kwargs)
    return GrowthModel(params)
