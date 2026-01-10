"""Mass transfer models for solvent extraction.

Provides film and overall mass transfer coefficient correlations
for REE extraction in mixer-settlers and columns.

All functions use JAX numpy for automatic differentiation.
"""

__all__ = [
    "film_mass_transfer",
    "overall_mass_transfer",
    "diffusion_coefficient",
    "sherwood_correlation",
    "mass_transfer_rate",
    "enhancement_factor",
    "MassTransferModel",
    "MassTransferParams",
    "get_mass_transfer_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Diffusion Coefficients
# =============================================================================

def diffusion_coefficient(
    T: Array | float,
    mu: Array | float,
    r_solute: Array | float = 3e-10,
) -> Array:
    """Estimate diffusion coefficient using Stokes-Einstein.

    D = kT / (6 * pi * mu * r)

    Args:
        T: Temperature (K)
        mu: Dynamic viscosity (Pa.s)
        r_solute: Effective solute radius (m)

    Returns:
        Diffusion coefficient (m²/s)

    Notes:
        Typical REE complex radius: 3-5 Å
        D ~ 10^-9 to 10^-10 m²/s
    """
    T = jnp.asarray(T)
    mu = jnp.asarray(mu)
    r_solute = jnp.asarray(r_solute)

    k_B = 1.38e-23  # Boltzmann constant
    return k_B * T / (6 * jnp.pi * mu * r_solute)


# =============================================================================
# Film Mass Transfer Coefficients
# =============================================================================

def film_mass_transfer(
    D: Array | float,
    delta: Array | float,
) -> Array:
    """Calculate film mass transfer coefficient.

    k = D / delta

    Args:
        D: Diffusion coefficient (m²/s)
        delta: Film thickness (m)

    Returns:
        Film mass transfer coefficient (m/s)
    """
    D = jnp.asarray(D)
    delta = jnp.asarray(delta)

    return D / delta


def sherwood_correlation(
    Re: Array | float,
    Sc: Array | float,
    correlation: str = "turbulent",
) -> Array:
    """Calculate Sherwood number from correlation.

    Args:
        Re: Reynolds number
        Sc: Schmidt number (mu / rho / D)
        correlation: Correlation type

    Returns:
        Sherwood number

    Notes:
        Sh = k * d / D
        Common correlations:
        - Turbulent droplet: Sh = 2 + 0.6 * Re^0.5 * Sc^0.33
        - Stirred tank: Sh = 0.13 * Re^0.67 * Sc^0.33
    """
    Re = jnp.asarray(Re)
    Sc = jnp.asarray(Sc)

    if correlation == "turbulent":
        # Ranz-Marshall for drops
        Sh = 2 + 0.6 * jnp.power(Re, 0.5) * jnp.power(Sc, 0.33)
    elif correlation == "stirred":
        # Calderbank correlation
        Sh = 0.13 * jnp.power(Re, 0.67) * jnp.power(Sc, 0.33)
    else:
        # Stagnant film (minimum)
        Sh = 2.0

    return Sh


def overall_mass_transfer(
    k_aq: Array | float,
    k_org: Array | float,
    D_eq: Array | float,
) -> Array:
    """Calculate overall mass transfer coefficient.

    1/K = 1/k_aq + D_eq/k_org

    Based on aqueous phase driving force.

    Args:
        k_aq: Aqueous film coefficient (m/s)
        k_org: Organic film coefficient (m/s)
        D_eq: Equilibrium distribution coefficient

    Returns:
        Overall mass transfer coefficient (m/s)
    """
    k_aq = jnp.asarray(k_aq)
    k_org = jnp.asarray(k_org)
    D_eq = jnp.asarray(D_eq)

    resistance = 1/k_aq + D_eq/k_org
    return 1 / resistance


# =============================================================================
# Mass Transfer Rates
# =============================================================================

def mass_transfer_rate(
    K: Array | float,
    a: Array | float,
    C_aq: Array | float,
    C_org: Array | float,
    D_eq: Array | float,
) -> Array:
    """Calculate volumetric mass transfer rate.

    r = K * a * (C_aq - C_org/D_eq)

    Args:
        K: Overall mass transfer coefficient (m/s)
        a: Interfacial area per volume (m²/m³)
        C_aq: Aqueous concentration (M)
        C_org: Organic concentration (M)
        D_eq: Equilibrium distribution coefficient

    Returns:
        Mass transfer rate (mol/m³/s)
    """
    K = jnp.asarray(K)
    a = jnp.asarray(a)
    C_aq = jnp.asarray(C_aq)
    C_org = jnp.asarray(C_org)
    D_eq = jnp.asarray(D_eq)

    driving_force = C_aq - C_org / D_eq
    return K * a * driving_force


def enhancement_factor(
    Ha: Array | float,
    E_inf: Array | float = 10.0,
) -> Array:
    """Calculate enhancement factor for fast reaction.

    E = sqrt(1 + Ha²) for pseudo-first-order
    E limited by E_inf for instantaneous reaction

    Args:
        Ha: Hatta number (reaction rate / diffusion rate)
        E_inf: Maximum enhancement (infinite reaction rate)

    Returns:
        Enhancement factor
    """
    Ha = jnp.asarray(Ha)
    E_inf = jnp.asarray(E_inf)

    E_pfo = jnp.sqrt(1 + Ha**2)
    return jnp.minimum(E_pfo, E_inf)


# =============================================================================
# Mass Transfer Model Class
# =============================================================================

@dataclass(repr=False)
class MassTransferParams(ParamsMixin):
    """Parameters for mass transfer model.

    Attributes:
        D_aq: Aqueous diffusivity (m²/s)
        D_org: Organic diffusivity (m²/s)
        delta_aq: Aqueous film thickness (m)
        delta_org: Organic film thickness (m)
        a: Interfacial area per volume (m²/m³)
    """
    D_aq: float | Array = 1e-9
    D_org: float | Array = 5e-10
    delta_aq: float | Array = 50e-6  # 50 micron
    delta_org: float | Array = 30e-6  # 30 micron
    a: float | Array = 500.0  # m²/m³ typical for mixer


class MassTransferModel:
    """Unified mass transfer model for extraction.

    Example:
        >>> model = MassTransferModel(MassTransferParams())
        >>> rate = model.transfer_rate(C_aq=0.01, C_org=0.0, D_eq=10.0)
    """

    def __init__(self, params: MassTransferParams):
        self.params = params

    def film_coefficients(self) -> tuple[Array, Array]:
        """Get aqueous and organic film coefficients."""
        p = self.params
        k_aq = film_mass_transfer(p.D_aq, p.delta_aq)
        k_org = film_mass_transfer(p.D_org, p.delta_org)
        return k_aq, k_org

    def overall_coefficient(self, D_eq: Array | float) -> Array:
        """Get overall mass transfer coefficient."""
        k_aq, k_org = self.film_coefficients()
        return overall_mass_transfer(k_aq, k_org, D_eq)

    def transfer_rate(
        self,
        C_aq: Array | float,
        C_org: Array | float,
        D_eq: Array | float,
    ) -> Array:
        """Calculate mass transfer rate."""
        K = self.overall_coefficient(D_eq)
        return mass_transfer_rate(K, self.params.a, C_aq, C_org, D_eq)


def get_mass_transfer_model(**kwargs) -> MassTransferModel:
    """Create mass transfer model."""
    params = MassTransferParams(**kwargs)
    return MassTransferModel(params)
