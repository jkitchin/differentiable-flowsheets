"""Product formation kinetic models.

Provides modular implementations of production rate expressions
used in bioreactor modeling.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Luedeking-Piret: Growth and non-growth associated production
- Growth-associated: Production proportional to growth
- Non-growth associated: Constitutive production
- Overflow metabolism: Production at high substrate

References:
    Luedeking R, Piret EL (1959). J Biochem Microbiol Tech Eng 1:393.
    Shuler ML, Kargi F (2002). Bioprocess Engineering. Prentice Hall.
"""

__all__ = [
    "luedeking_piret",
    "growth_associated",
    "non_growth_associated",
    "overflow_production",
    "product_inhibited_production",
    "substrate_limited_production",
    "ProductionModel",
    "get_production_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Basic Production Models
# =============================================================================

def luedeking_piret(
    mu: Array | float,
    X: Array | float,
    alpha: Array | float,
    beta: Array | float,
) -> Array:
    """Luedeking-Piret production kinetics.

    dP/dt = alpha * mu * X + beta * X

    Classic two-parameter model separating growth-associated (alpha)
    and non-growth-associated (beta) production.

    Args:
        mu: Specific growth rate (1/h)
        X: Cell concentration (g/L)
        alpha: Growth-associated coefficient (g product/g cells)
        beta: Non-growth-associated coefficient (g product/g cells/h)

    Returns:
        Volumetric production rate (g/L/h)

    Example:
        >>> # Antibody production: mixed growth/non-growth associated
        >>> r_P = luedeking_piret(mu=0.02, X=10.0, alpha=0.1, beta=0.01)
    """
    mu = jnp.asarray(mu)
    X = jnp.asarray(X)
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)

    return alpha * mu * X + beta * X


def growth_associated(
    mu: Array | float,
    X: Array | float,
    Y_px: Array | float,
) -> Array:
    """Growth-associated production.

    dP/dt = Y_px * mu * X

    Product formation proportional to growth rate.
    Common for primary metabolites and growth-coupled products.

    Args:
        mu: Specific growth rate (1/h)
        X: Cell concentration (g/L)
        Y_px: Product yield on biomass (g product/g cells)

    Returns:
        Volumetric production rate (g/L/h)

    Notes:
        Equivalent to Luedeking-Piret with beta=0.
    """
    mu = jnp.asarray(mu)
    X = jnp.asarray(X)
    Y_px = jnp.asarray(Y_px)

    return Y_px * mu * X


def non_growth_associated(
    X: Array | float,
    q_p: Array | float,
) -> Array:
    """Non-growth-associated (constitutive) production.

    dP/dt = q_p * X

    Product formation independent of growth rate.
    Common for secondary metabolites and recombinant proteins.

    Args:
        X: Cell concentration (g/L)
        q_p: Specific production rate (g product/g cells/h)

    Returns:
        Volumetric production rate (g/L/h)

    Notes:
        Equivalent to Luedeking-Piret with alpha=0.
    """
    X = jnp.asarray(X)
    q_p = jnp.asarray(q_p)

    return q_p * X


def overflow_production(
    S: Array | float,
    X: Array | float,
    q_p_max: Array | float,
    S_crit: Array | float,
) -> Array:
    """Overflow metabolism production.

    dP/dt = q_p_max * X * max(0, (S - S_crit) / S)

    Models byproduct formation at high substrate concentrations.
    Common for lactate/acetate in mammalian/microbial cultures.

    Args:
        S: Substrate concentration (g/L)
        X: Cell concentration (g/L)
        q_p_max: Maximum specific production rate (g/g/h)
        S_crit: Critical substrate for overflow (g/L)

    Returns:
        Volumetric production rate (g/L/h)

    Example:
        >>> # Lactate production above critical glucose
        >>> r_lac = overflow_production(S=5.0, X=10.0, q_p_max=0.5, S_crit=2.0)
    """
    S = jnp.asarray(S)
    X = jnp.asarray(X)
    q_p_max = jnp.asarray(q_p_max)
    S_crit = jnp.asarray(S_crit)

    overflow_fraction = jnp.maximum(0.0, (S - S_crit) / (S + 1e-10))
    return q_p_max * X * overflow_fraction


def product_inhibited_production(
    mu: Array | float,
    X: Array | float,
    P: Array | float,
    alpha: Array | float,
    beta: Array | float,
    P_max: Array | float,
) -> Array:
    """Product-inhibited production kinetics.

    dP/dt = (alpha * mu * X + beta * X) * (1 - P/P_max)

    Luedeking-Piret with product inhibition term.

    Args:
        mu: Specific growth rate (1/h)
        X: Cell concentration (g/L)
        P: Product concentration (g/L)
        alpha: Growth-associated coefficient
        beta: Non-growth-associated coefficient
        P_max: Maximum product concentration (inhibitory)

    Returns:
        Volumetric production rate (g/L/h)

    Notes:
        Production stops when P reaches P_max.
    """
    mu = jnp.asarray(mu)
    X = jnp.asarray(X)
    P = jnp.asarray(P)
    alpha = jnp.asarray(alpha)
    beta = jnp.asarray(beta)
    P_max = jnp.asarray(P_max)

    base_rate = alpha * mu * X + beta * X
    inhibition = jnp.maximum(0.0, 1.0 - P / P_max)

    return base_rate * inhibition


def substrate_limited_production(
    S: Array | float,
    X: Array | float,
    q_p_max: Array | float,
    K_p: Array | float,
) -> Array:
    """Substrate-limited production kinetics.

    dP/dt = q_p_max * X * S / (K_p + S)

    Monod-like saturation for production rate.

    Args:
        S: Substrate concentration (g/L)
        X: Cell concentration (g/L)
        q_p_max: Maximum specific production rate (g/g/h)
        K_p: Half-saturation constant for production (g/L)

    Returns:
        Volumetric production rate (g/L/h)
    """
    S = jnp.asarray(S)
    X = jnp.asarray(X)
    q_p_max = jnp.asarray(q_p_max)
    K_p = jnp.asarray(K_p)

    return q_p_max * X * S / (K_p + S)


# =============================================================================
# Production Model Class
# =============================================================================

@dataclass(repr=False)
class ProductionModelParams(ParamsMixin):
    """Parameters for production model.

    Attributes:
        model: Model type ('luedeking_piret', 'growth_associated', etc.)
        alpha: Growth-associated coefficient (g/g)
        beta: Non-growth-associated coefficient (g/g/h)
        Y_px: Product yield on biomass (g/g)
        q_p: Specific production rate (g/g/h)
        q_p_max: Maximum specific production rate (g/g/h)
        K_p: Half-saturation constant (g/L)
        P_max: Maximum product concentration (g/L)
        S_crit: Critical substrate for overflow (g/L)
    """
    model: str = "luedeking_piret"
    alpha: float | Array = 0.1
    beta: float | Array = 0.01
    Y_px: float | Array = 0.1
    q_p: float | Array = 0.01
    q_p_max: float | Array = 0.1
    K_p: float | Array = 0.5
    P_max: float | Array = 50.0
    S_crit: float | Array = 2.0


class ProductionModel:
    """Unified interface for production kinetic models.

    Example:
        >>> model = ProductionModel(ProductionModelParams(
        ...     model="luedeking_piret",
        ...     alpha=0.1,
        ...     beta=0.01,
        ... ))
        >>> r_P = model(mu=0.02, X=10.0)
    """

    def __init__(self, params: ProductionModelParams):
        """Initialize production model.

        Args:
            params: Production model parameters
        """
        self.params = params

    def __call__(
        self,
        mu: Array | float = None,
        X: Array | float = 1.0,
        S: Array | float = None,
        P: Array | float = None,
    ) -> Array:
        """Calculate volumetric production rate.

        Args:
            mu: Specific growth rate (1/h)
            X: Cell concentration (g/L)
            S: Substrate concentration (g/L)
            P: Product concentration (g/L)

        Returns:
            Volumetric production rate (g/L/h)
        """
        p = self.params

        if p.model == "luedeking_piret":
            if mu is None:
                raise ValueError("Luedeking-Piret requires mu")
            return luedeking_piret(mu, X, p.alpha, p.beta)

        elif p.model == "growth_associated":
            if mu is None:
                raise ValueError("Growth-associated requires mu")
            return growth_associated(mu, X, p.Y_px)

        elif p.model == "non_growth_associated":
            return non_growth_associated(X, p.q_p)

        elif p.model == "overflow":
            if S is None:
                raise ValueError("Overflow model requires S")
            return overflow_production(S, X, p.q_p_max, p.S_crit)

        elif p.model == "product_inhibited":
            if mu is None or P is None:
                raise ValueError("Product-inhibited requires mu and P")
            return product_inhibited_production(mu, X, P, p.alpha, p.beta, p.P_max)

        elif p.model == "substrate_limited":
            if S is None:
                raise ValueError("Substrate-limited requires S")
            return substrate_limited_production(S, X, p.q_p_max, p.K_p)

        else:
            raise ValueError(f"Unknown model: {p.model}")

    def specific_rate(
        self,
        mu: Array | float = None,
        S: Array | float = None,
        P: Array | float = None,
    ) -> Array:
        """Calculate specific production rate (q_p = r_P / X).

        Returns:
            Specific production rate (g product/g cells/h)
        """
        # Calculate for X=1 to get specific rate
        return self(mu=mu, X=1.0, S=S, P=P)


def get_production_model(
    model_type: str = "luedeking_piret",
    **kwargs,
) -> ProductionModel:
    """Create production model with specified type.

    Args:
        model_type: Model name
        **kwargs: Model parameters

    Returns:
        ProductionModel instance
    """
    params = ProductionModelParams(model=model_type, **kwargs)
    return ProductionModel(params)
