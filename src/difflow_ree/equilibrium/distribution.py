"""pH-dependent distribution coefficient models for REE extraction.

Distribution coefficient D = [REE]_org / [REE]_aq

Model: log10(D) = a + b*pH + c*pH^2 + d*(1/T - 1/Tref)

All functions are JAX-compatible for automatic differentiation.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow_ree.database import (
    get_extractant_database,
    get_extractant,
    PHCoefficients,
)


# =============================================================================
# Distribution Coefficient Model
# =============================================================================

@dataclass
class REEDistribution:
    """REE distribution coefficient calculator.

    Computes D values based on pH, temperature, and extractant concentration.

    Attributes:
        extractant: Name of extractant (D2EHPA, PC88A, Cyanex272, TBP)
        elements: List of REE symbols to include
        concentration: Extractant concentration (M)
    """
    extractant: str
    elements: tuple[str, ...]
    concentration: float = 0.5  # M

    def __post_init__(self):
        """Load extractant data."""
        self._ext_data = get_extractant(self.extractant)

    def get_D(
        self,
        element: str,
        pH: Array | float,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate distribution coefficient for single element.

        Args:
            element: REE symbol (e.g., "Nd")
            pH: Solution pH
            T: Temperature (K)

        Returns:
            Distribution coefficient D
        """
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        # Get pH coefficients
        coeffs = self._ext_data.ph_coefficients[element]

        # Base log(D) from pH correlation
        log_D = coeffs.a + coeffs.b * pH + coeffs.c * pH**2

        # Temperature correction
        T_ref = 298.15
        d_T = self._ext_data.temperature_coefficients[element]
        log_D = log_D + d_T * (1/T - 1/T_ref)

        # Concentration correction: D ∝ [HA]^n
        n = self._ext_data.concentration_exponent
        C_ref = self._ext_data.reference_concentration
        log_D = log_D + n * jnp.log10(self.concentration / C_ref)

        return jnp.power(10.0, log_D)

    def get_D_all(
        self,
        pH: Array | float,
        T: Array | float = 298.15,
    ) -> dict[str, Array]:
        """Calculate distribution coefficients for all elements.

        Args:
            pH: Solution pH
            T: Temperature (K)

        Returns:
            Dictionary mapping element symbols to D values
        """
        return {elem: self.get_D(elem, pH, T) for elem in self.elements}

    def get_D_array(
        self,
        pH: Array | float,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate D values as JAX array (in element order).

        Args:
            pH: Solution pH
            T: Temperature (K)

        Returns:
            JAX array of D values
        """
        D_list = [self.get_D(elem, pH, T) for elem in self.elements]
        return jnp.stack(D_list)

    def get_separation_factor(
        self,
        element1: str,
        element2: str,
        pH: Array | float,
        T: Array | float = 298.15,
    ) -> Array:
        """Calculate separation factor between two elements.

        SF = D1 / D2

        Args:
            element1: First element symbol
            element2: Second element symbol
            pH: Solution pH
            T: Temperature (K)

        Returns:
            Separation factor
        """
        D1 = self.get_D(element1, pH, T)
        D2 = self.get_D(element2, pH, T)
        return D1 / D2

    def optimal_pH_for_separation(
        self,
        element1: str,
        element2: str,
        pH_range: tuple[float, float] = (1.0, 5.0),
        n_points: int = 100,
        T: float = 298.15,
    ) -> tuple[float, float]:
        """Find pH that maximizes separation factor.

        Args:
            element1: Target element (to extract)
            element2: Impurity element (to reject)
            pH_range: pH range to search
            n_points: Number of evaluation points
            T: Temperature (K)

        Returns:
            Tuple of (optimal_pH, max_SF)
        """
        pH_values = jnp.linspace(pH_range[0], pH_range[1], n_points)
        SF_values = jnp.array([
            float(self.get_separation_factor(element1, element2, pH, T))
            for pH in pH_values
        ])
        max_idx = jnp.argmax(SF_values)
        return float(pH_values[max_idx]), float(SF_values[max_idx])


# =============================================================================
# Convenience Functions
# =============================================================================

def get_distribution_coefficient(
    element: str,
    extractant: str,
    pH: Array | float,
    T: Array | float = 298.15,
    concentration: float = 0.5,
) -> Array:
    """Calculate distribution coefficient for a single element.

    Args:
        element: REE symbol (e.g., "Nd")
        extractant: Extractant name (e.g., "D2EHPA")
        pH: Solution pH
        T: Temperature (K)
        concentration: Extractant concentration (M)

    Returns:
        Distribution coefficient D
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=(element,),
        concentration=concentration,
    )
    return dist.get_D(element, pH, T)


def get_distribution_coefficients(
    elements: list[str],
    extractant: str,
    pH: Array | float,
    T: Array | float = 298.15,
    concentration: float = 0.5,
) -> dict[str, Array]:
    """Calculate distribution coefficients for multiple elements.

    Args:
        elements: List of REE symbols
        extractant: Extractant name
        pH: Solution pH
        T: Temperature (K)
        concentration: Extractant concentration (M)

    Returns:
        Dictionary mapping element symbols to D values
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=tuple(elements),
        concentration=concentration,
    )
    return dist.get_D_all(pH, T)


def get_separation_factor(
    element1: str,
    element2: str,
    extractant: str,
    pH: Array | float,
    T: Array | float = 298.15,
    concentration: float = 0.5,
) -> Array:
    """Calculate separation factor between two elements.

    Args:
        element1: First element symbol
        element2: Second element symbol
        extractant: Extractant name
        pH: Solution pH
        T: Temperature (K)
        concentration: Extractant concentration (M)

    Returns:
        Separation factor D1/D2
    """
    D1 = get_distribution_coefficient(element1, extractant, pH, T, concentration)
    D2 = get_distribution_coefficient(element2, extractant, pH, T, concentration)
    return D1 / D2


# =============================================================================
# McCabe-Thiele Analysis
# =============================================================================

def equilibrium_line(
    D: Array | float,
    x: Array,
) -> Array:
    """Calculate equilibrium line y = D*x for McCabe-Thiele.

    Args:
        D: Distribution coefficient
        x: Aqueous phase concentration

    Returns:
        Organic phase concentration at equilibrium
    """
    return D * x


def operating_line_extraction(
    x: Array,
    x_in: float,
    y_in: float,
    S_F: float,
) -> Array:
    """Calculate extraction operating line for McCabe-Thiele.

    Material balance: F*x_in + S*y_in = F*x + S*y
    Rearranged: y = (F/S)*(x_in - x) + y_in

    Args:
        x: Aqueous phase concentration
        x_in: Inlet aqueous concentration
        y_in: Inlet organic concentration (usually 0 for fresh solvent)
        S_F: Solvent-to-feed ratio (S/F)

    Returns:
        Organic phase concentration
    """
    return (1/S_F) * (x_in - x) + y_in


def operating_line_stripping(
    y: Array,
    y_in: float,
    x_in: float,
    A_S: float,
) -> Array:
    """Calculate stripping operating line for McCabe-Thiele.

    Material balance: S*y_in + A*x_in = S*y + A*x
    Rearranged: x = (S/A)*(y_in - y) + x_in

    Args:
        y: Organic phase concentration
        y_in: Inlet organic concentration (loaded)
        x_in: Inlet aqueous concentration (strip solution)
        A_S: Aqueous-to-solvent ratio (A/S)

    Returns:
        Aqueous phase concentration
    """
    return (1/A_S) * (y_in - y) + x_in


def minimum_solvent_ratio(
    D: Array | float,
    x_in: float,
    x_out: float,
    y_in: float = 0.0,
) -> Array:
    """Calculate minimum solvent-to-feed ratio.

    At minimum S/F, operating line touches equilibrium line.

    Args:
        D: Distribution coefficient
        x_in: Inlet aqueous concentration
        x_out: Desired outlet aqueous concentration
        y_in: Inlet organic concentration

    Returns:
        Minimum S/F ratio
    """
    D = jnp.asarray(D)
    # At equilibrium: y* = D * x_in (maximum loading)
    y_max = D * x_in
    # Material balance: F*(x_in - x_out) = S*(y_max - y_in)
    # S/F = (x_in - x_out) / (y_max - y_in)
    return safe_divide(x_in - x_out, y_max - y_in)


def stages_kremser(
    D: Array | float,
    S_F: Array | float,
    recovery: float = 0.99,
) -> Array:
    """Calculate number of stages using Kremser equation.

    For counter-current extraction.

    Args:
        D: Distribution coefficient
        S_F: Solvent-to-feed ratio
        recovery: Desired recovery fraction

    Returns:
        Number of theoretical stages
    """
    D = jnp.asarray(D)
    S_F = jnp.asarray(S_F)

    # Extraction factor E = D * S/F
    E = D * S_F

    # Kremser equation
    # Recovery = (E^(N+1) - E) / (E^(N+1) - 1)
    # Solving for N:
    # N = log((recovery*(E-1) + 1) / E) / log(E) - 1

    # Handle E ≈ 1 case
    N = jnp.where(
        jnp.abs(E - 1.0) < 1e-6,
        recovery / (1 - recovery),  # Limit as E → 1
        jnp.log((recovery * (E - 1) + 1) / E) / jnp.log(E)
    )

    return jnp.maximum(N, 1.0)
