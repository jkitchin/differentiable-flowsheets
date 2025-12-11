"""Aqueous phase speciation models for REE.

REE in aqueous solution can exist as:
- Free ions: REE³⁺
- Sulfate complexes: REE(SO₄)⁺, REE(SO₄)₂⁻
- Chloride complexes: REECl²⁺, REECl₂⁺
- Nitrate complexes: REE(NO₃)²⁺, REE(NO₃)₂⁺
- Hydroxide complexes: REE(OH)²⁺ (at high pH)

Speciation affects extraction because typically only free REE³⁺
is extractable. Complexed species have lower effective D values.

All functions are JAX-compatible for automatic differentiation.
"""

from dataclasses import dataclass, field
from typing import Literal

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Stability Constants (log K values at 25°C, I=0)
# =============================================================================

# Sulfate complexation: REE³⁺ + SO₄²⁻ ⇌ REE(SO₄)⁺
LOG_K_SULFATE_1 = {
    "La": 3.64, "Ce": 3.68, "Pr": 3.71, "Nd": 3.74,
    "Sm": 3.82, "Eu": 3.85, "Gd": 3.88, "Tb": 3.91,
    "Dy": 3.94, "Y": 3.90,
}

# REE(SO₄)⁺ + SO₄²⁻ ⇌ REE(SO₄)₂⁻
LOG_K_SULFATE_2 = {
    "La": 2.10, "Ce": 2.15, "Pr": 2.18, "Nd": 2.20,
    "Sm": 2.28, "Eu": 2.30, "Gd": 2.32, "Tb": 2.35,
    "Dy": 2.38, "Y": 2.34,
}

# Chloride complexation: REE³⁺ + Cl⁻ ⇌ REECl²⁺
LOG_K_CHLORIDE_1 = {
    "La": 0.48, "Ce": 0.52, "Pr": 0.55, "Nd": 0.58,
    "Sm": 0.64, "Eu": 0.66, "Gd": 0.68, "Tb": 0.70,
    "Dy": 0.72, "Y": 0.68,
}

# REECl²⁺ + Cl⁻ ⇌ REECl₂⁺
LOG_K_CHLORIDE_2 = {
    "La": -0.30, "Ce": -0.25, "Pr": -0.22, "Nd": -0.20,
    "Sm": -0.15, "Eu": -0.12, "Gd": -0.10, "Tb": -0.08,
    "Dy": -0.06, "Y": -0.10,
}

# Nitrate complexation: REE³⁺ + NO₃⁻ ⇌ REE(NO₃)²⁺
LOG_K_NITRATE_1 = {
    "La": 1.15, "Ce": 1.18, "Pr": 1.20, "Nd": 1.22,
    "Sm": 1.28, "Eu": 1.30, "Gd": 1.32, "Tb": 1.34,
    "Dy": 1.36, "Y": 1.32,
}


# =============================================================================
# Speciation Calculator
# =============================================================================

@dataclass
class REESpeciation:
    """Calculate REE speciation in aqueous solution.

    Computes the fraction of REE as free ion vs. complexes.

    Attributes:
        elements: List of REE symbols
        medium: Aqueous medium type (sulfate, chloride, nitrate, mixed)
    """
    elements: tuple[str, ...]
    medium: Literal["sulfate", "chloride", "nitrate", "mixed"] = "sulfate"

    def free_fraction(
        self,
        element: str,
        ligand_conc: Array | float,
        pH: Array | float = 3.0,
    ) -> Array:
        """Calculate fraction of REE as free ion.

        alpha_free = [REE³⁺] / [REE]_total

        Args:
            element: REE symbol
            ligand_conc: Total ligand concentration (M)
            pH: Solution pH (affects some equilibria)

        Returns:
            Fraction of REE as free ion (0 to 1)
        """
        ligand_conc = jnp.asarray(ligand_conc)

        if self.medium == "sulfate":
            return self._sulfate_free_fraction(element, ligand_conc)
        elif self.medium == "chloride":
            return self._chloride_free_fraction(element, ligand_conc)
        elif self.medium == "nitrate":
            return self._nitrate_free_fraction(element, ligand_conc)
        else:
            # Mixed medium - average of contributions
            return self._mixed_free_fraction(element, ligand_conc)

    def _sulfate_free_fraction(
        self,
        element: str,
        SO4_conc: Array,
    ) -> Array:
        """Calculate free REE fraction in sulfate medium."""
        K1 = jnp.power(10.0, LOG_K_SULFATE_1[element])
        K2 = jnp.power(10.0, LOG_K_SULFATE_2[element])

        # alpha_free = 1 / (1 + K1*[SO4] + K1*K2*[SO4]^2)
        denom = 1 + K1 * SO4_conc + K1 * K2 * SO4_conc**2
        return 1.0 / denom

    def _chloride_free_fraction(
        self,
        element: str,
        Cl_conc: Array,
    ) -> Array:
        """Calculate free REE fraction in chloride medium."""
        K1 = jnp.power(10.0, LOG_K_CHLORIDE_1[element])
        K2 = jnp.power(10.0, LOG_K_CHLORIDE_2[element])

        denom = 1 + K1 * Cl_conc + K1 * K2 * Cl_conc**2
        return 1.0 / denom

    def _nitrate_free_fraction(
        self,
        element: str,
        NO3_conc: Array,
    ) -> Array:
        """Calculate free REE fraction in nitrate medium."""
        K1 = jnp.power(10.0, LOG_K_NITRATE_1[element])

        # Only considering first complex for nitrate
        denom = 1 + K1 * NO3_conc
        return 1.0 / denom

    def _mixed_free_fraction(
        self,
        element: str,
        total_ligand: Array,
    ) -> Array:
        """Approximate free fraction for mixed medium."""
        # Assume sulfate-dominated (most common in REE processing)
        return self._sulfate_free_fraction(element, total_ligand * 0.5)

    def effective_D(
        self,
        element: str,
        D_free: Array | float,
        ligand_conc: Array | float,
        pH: Array | float = 3.0,
    ) -> Array:
        """Calculate effective D accounting for speciation.

        D_eff = D_free * alpha_free

        Only free REE³⁺ is extracted, so complexation reduces
        the apparent distribution coefficient.

        Args:
            element: REE symbol
            D_free: Distribution coefficient for free REE
            ligand_conc: Ligand concentration (M)
            pH: Solution pH

        Returns:
            Effective distribution coefficient
        """
        alpha = self.free_fraction(element, ligand_conc, pH)
        return jnp.asarray(D_free) * alpha

    def speciation_distribution(
        self,
        element: str,
        ligand_conc: Array | float,
    ) -> dict[str, Array]:
        """Calculate full speciation distribution.

        Args:
            element: REE symbol
            ligand_conc: Ligand concentration (M)

        Returns:
            Dictionary with fraction of each species
        """
        ligand_conc = jnp.asarray(ligand_conc)

        if self.medium == "sulfate":
            K1 = jnp.power(10.0, LOG_K_SULFATE_1[element])
            K2 = jnp.power(10.0, LOG_K_SULFATE_2[element])

            denom = 1 + K1 * ligand_conc + K1 * K2 * ligand_conc**2

            return {
                "REE3+": 1.0 / denom,
                "REE(SO4)+": K1 * ligand_conc / denom,
                "REE(SO4)2-": K1 * K2 * ligand_conc**2 / denom,
            }

        elif self.medium == "chloride":
            K1 = jnp.power(10.0, LOG_K_CHLORIDE_1[element])
            K2 = jnp.power(10.0, LOG_K_CHLORIDE_2[element])

            denom = 1 + K1 * ligand_conc + K1 * K2 * ligand_conc**2

            return {
                "REE3+": 1.0 / denom,
                "REECl2+": K1 * ligand_conc / denom,
                "REECl2+": K1 * K2 * ligand_conc**2 / denom,
            }

        elif self.medium == "nitrate":
            K1 = jnp.power(10.0, LOG_K_NITRATE_1[element])

            denom = 1 + K1 * ligand_conc

            return {
                "REE3+": 1.0 / denom,
                "REE(NO3)2+": K1 * ligand_conc / denom,
            }

        else:
            return {"REE3+": self.free_fraction(element, ligand_conc)}


# =============================================================================
# Convenience Functions
# =============================================================================

def sulfate_speciation(
    element: str,
    SO4_conc: Array | float,
) -> dict[str, Array]:
    """Calculate REE speciation in sulfate solution.

    Args:
        element: REE symbol
        SO4_conc: Total sulfate concentration (M)

    Returns:
        Dictionary with fraction of each species
    """
    spec = REESpeciation(elements=(element,), medium="sulfate")
    return spec.speciation_distribution(element, SO4_conc)


def chloride_speciation(
    element: str,
    Cl_conc: Array | float,
) -> dict[str, Array]:
    """Calculate REE speciation in chloride solution.

    Args:
        element: REE symbol
        Cl_conc: Total chloride concentration (M)

    Returns:
        Dictionary with fraction of each species
    """
    spec = REESpeciation(elements=(element,), medium="chloride")
    return spec.speciation_distribution(element, Cl_conc)


def speciation_correction(
    D_values: dict[str, Array],
    elements: list[str],
    ligand_conc: float,
    medium: str = "sulfate",
) -> dict[str, Array]:
    """Apply speciation correction to D values.

    Args:
        D_values: Distribution coefficients (for free ion)
        elements: List of REE symbols
        ligand_conc: Ligand concentration (M)
        medium: Solution medium type

    Returns:
        Corrected D values accounting for speciation
    """
    spec = REESpeciation(elements=tuple(elements), medium=medium)

    corrected = {}
    for elem in elements:
        alpha = spec.free_fraction(elem, ligand_conc)
        corrected[elem] = D_values[elem] * alpha

    return corrected


# =============================================================================
# Ionic Strength Effects
# =============================================================================

def activity_coefficient_davies(
    z: int,
    ionic_strength: Array | float,
) -> Array:
    """Calculate activity coefficient using Davies equation.

    log(gamma) = -A * z² * (sqrt(I)/(1 + sqrt(I)) - 0.3*I)

    Valid for I < 0.5 M.

    Args:
        z: Ion charge
        ionic_strength: Ionic strength (M)

    Returns:
        Activity coefficient
    """
    I = jnp.asarray(ionic_strength)
    A = 0.509  # at 25°C for water

    sqrt_I = jnp.sqrt(I)
    log_gamma = -A * z**2 * (sqrt_I / (1 + sqrt_I) - 0.3 * I)

    return jnp.power(10.0, log_gamma)


def ionic_strength_from_composition(
    concentrations: dict[str, float],
    charges: dict[str, int],
) -> float:
    """Calculate ionic strength from solution composition.

    I = 0.5 * sum(c_i * z_i²)

    Args:
        concentrations: Molar concentrations of species
        charges: Charges of species

    Returns:
        Ionic strength (M)
    """
    I = 0.0
    for species, conc in concentrations.items():
        z = charges.get(species, 0)
        I += conc * z**2
    return 0.5 * I


def correct_K_for_ionic_strength(
    log_K_0: float,
    delta_z2: float,
    ionic_strength: float,
) -> float:
    """Correct stability constant for ionic strength.

    Uses simplified Debye-Hückel correction.

    log K(I) = log K(0) + A * delta_z² * (sqrt(I)/(1 + sqrt(I)))

    where delta_z² = sum(z_products²) - sum(z_reactants²)

    Args:
        log_K_0: Stability constant at I=0
        delta_z2: Change in sum of squared charges
        ionic_strength: Ionic strength (M)

    Returns:
        Corrected log K
    """
    A = 0.509
    sqrt_I = ionic_strength**0.5
    correction = A * delta_z2 * (sqrt_I / (1 + sqrt_I))
    return log_K_0 + correction
