"""Cellular metabolism models.

Provides modular implementations of substrate uptake, oxygen consumption,
and metabolic flux models for bioreactor simulation.

All functions use JAX numpy for automatic differentiation compatibility.

Models implemented:
- Substrate uptake: Growth + maintenance requirements
- Oxygen uptake rate (OUR): Aerobic respiration
- CO2 evolution rate (CER): Respiratory quotient
- Maintenance energy: Baseline metabolism

References:
    Pirt SJ (1965). Proc R Soc Lond B Biol Sci 163:224.
        (Maintenance energy concept)
    Shuler ML, Kargi F (2002). Bioprocess Engineering. Prentice Hall.
"""

__all__ = [
    "substrate_uptake_rate",
    "oxygen_uptake_rate",
    "co2_evolution_rate",
    "maintenance_energy",
    "specific_substrate_uptake",
    "yield_coefficient",
    "metabolic_quotient",
    "MetabolismModel",
    "get_metabolism_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide


# =============================================================================
# Substrate Uptake
# =============================================================================

def substrate_uptake_rate(
    mu: Array | float,
    X: Array | float,
    Y_xs: Array | float,
    m_s: Array | float = 0.0,
) -> Array:
    """Calculate substrate uptake rate.

    dS/dt = -(mu/Y_xs + m_s) * X

    Substrate consumed for growth and maintenance.

    Args:
        mu: Specific growth rate (1/h)
        X: Cell concentration (g/L)
        Y_xs: Biomass yield on substrate (g cells/g substrate)
        m_s: Maintenance coefficient (g substrate/g cells/h)

    Returns:
        Volumetric substrate uptake rate (g/L/h), negative value

    Notes:
        Returns negative value since substrate is consumed.
    """
    mu = jnp.asarray(mu)
    X = jnp.asarray(X)
    Y_xs = jnp.asarray(Y_xs)
    m_s = jnp.asarray(m_s)

    return -(mu / Y_xs + m_s) * X


def specific_substrate_uptake(
    mu: Array | float,
    Y_xs: Array | float,
    m_s: Array | float = 0.0,
) -> Array:
    """Calculate specific substrate uptake rate (q_s).

    q_s = mu/Y_xs + m_s

    Args:
        mu: Specific growth rate (1/h)
        Y_xs: Biomass yield on substrate (g/g)
        m_s: Maintenance coefficient (g/g/h)

    Returns:
        Specific substrate uptake rate (g substrate/g cells/h)
    """
    mu = jnp.asarray(mu)
    Y_xs = jnp.asarray(Y_xs)
    m_s = jnp.asarray(m_s)

    return mu / Y_xs + m_s


def maintenance_energy(
    X: Array | float,
    m_s: Array | float,
) -> Array:
    """Calculate maintenance energy requirement.

    dS_m/dt = m_s * X

    Substrate consumed for cell maintenance (not growth).

    Args:
        X: Cell concentration (g/L)
        m_s: Maintenance coefficient (g substrate/g cells/h)

    Returns:
        Maintenance substrate consumption (g/L/h)

    Notes:
        Typical values:
        - E. coli: 0.05-0.1 g glucose/g cells/h
        - CHO: 0.01-0.03 g glucose/g cells/h
    """
    X = jnp.asarray(X)
    m_s = jnp.asarray(m_s)

    return m_s * X


# =============================================================================
# Oxygen and CO2
# =============================================================================

def oxygen_uptake_rate(
    mu: Array | float,
    X: Array | float,
    Y_xo: Array | float,
    m_o: Array | float = 0.0,
) -> Array:
    """Calculate oxygen uptake rate (OUR).

    OUR = (mu/Y_xo + m_o) * X

    Oxygen consumed for growth and maintenance respiration.

    Args:
        mu: Specific growth rate (1/h)
        X: Cell concentration (g/L)
        Y_xo: Biomass yield on oxygen (g cells/g O2)
        m_o: Maintenance oxygen coefficient (g O2/g cells/h)

    Returns:
        Volumetric OUR (g O2/L/h)

    Notes:
        Typical values for aerobic cultures:
        - Y_xo: 0.5-1.5 g cells/g O2
        - m_o: 0.01-0.05 g O2/g cells/h
    """
    mu = jnp.asarray(mu)
    X = jnp.asarray(X)
    Y_xo = jnp.asarray(Y_xo)
    m_o = jnp.asarray(m_o)

    return (mu / Y_xo + m_o) * X


def co2_evolution_rate(
    OUR: Array | float,
    RQ: Array | float = 1.0,
) -> Array:
    """Calculate CO2 evolution rate (CER).

    CER = OUR * RQ

    Args:
        OUR: Oxygen uptake rate (g O2/L/h or mol O2/L/h)
        RQ: Respiratory quotient (mol CO2/mol O2)

    Returns:
        CER in same units as OUR

    Notes:
        RQ depends on substrate:
        - Glucose: RQ ≈ 1.0
        - Fatty acids: RQ ≈ 0.7
        - Organic acids: RQ > 1.0
    """
    OUR = jnp.asarray(OUR)
    RQ = jnp.asarray(RQ)

    return OUR * RQ


def metabolic_quotient(
    rate: Array | float,
    X: Array | float,
) -> Array:
    """Calculate metabolic quotient (specific rate).

    q = rate / X

    General function to convert volumetric to specific rate.

    Args:
        rate: Volumetric rate (g/L/h)
        X: Cell concentration (g/L)

    Returns:
        Specific rate (g/g cells/h)
    """
    rate = jnp.asarray(rate)
    X = jnp.asarray(X)

    return safe_divide(rate, X)


# =============================================================================
# Yield Coefficients
# =============================================================================

def yield_coefficient(
    delta_product: Array | float,
    delta_substrate: Array | float,
) -> Array:
    """Calculate yield coefficient.

    Y = -delta_product / delta_substrate

    Negative sign because substrate is consumed.

    Args:
        delta_product: Change in product (g/L)
        delta_substrate: Change in substrate (g/L), negative

    Returns:
        Yield coefficient (g product/g substrate)
    """
    delta_product = jnp.asarray(delta_product)
    delta_substrate = jnp.asarray(delta_substrate)

    return safe_divide(-delta_product, delta_substrate)


# =============================================================================
# Metabolism Model Class
# =============================================================================

@dataclass(repr=False)
class MetabolismParams(ParamsMixin):
    """Parameters for metabolism model.

    Attributes:
        Y_xs: Biomass yield on substrate (g/g)
        Y_xo: Biomass yield on oxygen (g/g)
        Y_ps: Product yield on substrate (g/g)
        m_s: Maintenance substrate coefficient (g/g/h)
        m_o: Maintenance oxygen coefficient (g/g/h)
        RQ: Respiratory quotient (mol CO2/mol O2)
    """
    Y_xs: float | Array = 0.5  # g cells/g glucose
    Y_xo: float | Array = 1.0  # g cells/g O2
    Y_ps: float | Array = 0.4  # g product/g glucose
    m_s: float | Array = 0.02  # g glucose/g cells/h
    m_o: float | Array = 0.01  # g O2/g cells/h
    RQ: float | Array = 1.0  # mol CO2/mol O2


class MetabolismModel:
    """Unified interface for metabolic calculations.

    Calculates substrate uptake, oxygen consumption, and CO2 evolution
    with consistent parameters.

    Example:
        >>> model = MetabolismModel(MetabolismParams(Y_xs=0.5, m_s=0.02))
        >>> rates = model.calculate_rates(mu=0.1, X=10.0)
    """

    def __init__(self, params: MetabolismParams):
        """Initialize metabolism model.

        Args:
            params: Metabolism parameters
        """
        self.params = params

    def substrate_rate(
        self,
        mu: Array | float,
        X: Array | float,
    ) -> Array:
        """Calculate substrate uptake rate.

        Args:
            mu: Specific growth rate (1/h)
            X: Cell concentration (g/L)

        Returns:
            Substrate uptake rate (g/L/h), negative
        """
        return substrate_uptake_rate(mu, X, self.params.Y_xs, self.params.m_s)

    def oxygen_rate(
        self,
        mu: Array | float,
        X: Array | float,
    ) -> Array:
        """Calculate oxygen uptake rate.

        Args:
            mu: Specific growth rate (1/h)
            X: Cell concentration (g/L)

        Returns:
            OUR (g O2/L/h)
        """
        return oxygen_uptake_rate(mu, X, self.params.Y_xo, self.params.m_o)

    def co2_rate(
        self,
        mu: Array | float,
        X: Array | float,
    ) -> Array:
        """Calculate CO2 evolution rate.

        Args:
            mu: Specific growth rate (1/h)
            X: Cell concentration (g/L)

        Returns:
            CER (g CO2/L/h equivalent)
        """
        our = self.oxygen_rate(mu, X)
        return co2_evolution_rate(our, self.params.RQ)

    def maintenance_rate(
        self,
        X: Array | float,
    ) -> Array:
        """Calculate maintenance substrate consumption.

        Args:
            X: Cell concentration (g/L)

        Returns:
            Maintenance consumption (g/L/h)
        """
        return maintenance_energy(X, self.params.m_s)

    def calculate_rates(
        self,
        mu: Array | float,
        X: Array | float,
    ) -> dict:
        """Calculate all metabolic rates.

        Args:
            mu: Specific growth rate (1/h)
            X: Cell concentration (g/L)

        Returns:
            Dict with substrate, oxygen, co2, maintenance rates
        """
        return {
            "substrate_rate": self.substrate_rate(mu, X),
            "oxygen_rate": self.oxygen_rate(mu, X),
            "co2_rate": self.co2_rate(mu, X),
            "maintenance_rate": self.maintenance_rate(X),
            "specific_substrate": specific_substrate_uptake(
                mu, self.params.Y_xs, self.params.m_s
            ),
        }


def get_metabolism_model(
    **kwargs,
) -> MetabolismModel:
    """Create metabolism model.

    Args:
        **kwargs: Model parameters

    Returns:
        MetabolismModel instance
    """
    params = MetabolismParams(**kwargs)
    return MetabolismModel(params)
