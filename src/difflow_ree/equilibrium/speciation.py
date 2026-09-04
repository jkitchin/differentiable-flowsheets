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
                "REECl2": K1 * K2 * ligand_conc**2 / denom,
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

# Ionic strength (M) at which the Davies bracket
#
#     f(I) = sqrt(I)/(1 + sqrt(I)) - 0.3 I
#
# changes sign. Solving f(I) = 0 with x = sqrt(I) gives 0.3 x^2 + 0.3 x - 1 = 0,
# i.e. x = (-1 + sqrt(1 + 40/3))/2 and I = x^2 (#194). Above this ionic strength
# Davies predicts gamma > 1 for every ion, and any ratio of Davies coefficients
# built from it INVERTS: gamma_RE/gamma_H**3 = 10**(-6 A f) crosses 1 here and
# grows without bound. This is why difflow_ree never extrapolates Davies
# silently -- see REEDistribution and DAVIES_MAX_IONIC_STRENGTH.
DAVIES_SIGN_CHANGE_IONIC_STRENGTH = 1.940363884733242  # M

# Documented validity limit of the Davies equation (M).
DAVIES_MAX_IONIC_STRENGTH = 0.5

# Aqueous media difflow_ree recognizes. The same vocabulary as
# :class:`REESpeciation.medium`, reused by
# :class:`difflow_ree.equilibrium.distribution.REEDistribution` so that a
# nitrate-requiring (solvating) extractant declared to be operating in a
# chloride or sulfate liquor is detected rather than merely described in an
# error message (#195). "mixed" is treated as containing nitrate.
AQUEOUS_MEDIA = ("sulfate", "chloride", "nitrate", "mixed")

# Media that supply the nitrate (salting) anion a solvating extractant needs.
NITRATE_BEARING_MEDIA = ("nitrate", "mixed")


def activity_coefficient_davies(
    z: int,
    ionic_strength: Array | float,
) -> Array:
    """Calculate activity coefficient using Davies equation.

    log10(gamma) = -A * z^2 * f(I),  f(I) = sqrt(I)/(1 + sqrt(I)) - 0.3*I

    VALIDITY (#194). The equation is documented for I < 0.5 M
    (:data:`DAVIES_MAX_IONIC_STRENGTH`). Beyond that it is not merely an
    inaccurate extrapolation: ``f`` changes sign at
    I = 1.940363884733242 M (:data:`DAVIES_SIGN_CHANGE_IONIC_STRENGTH`, the
    root of 0.3 I + 0.3 sqrt(I) - 1 = 0), so above ~1.94 M this function
    returns gamma > 1 and any correction built as a ratio of Davies
    coefficients reverses direction. For example
    ``gamma_RE3+ / gamma_H+**3 = 10**(-6 A f)`` equals 0.228 at I = 0.1 M,
    0.245 at I = 1.0 M, 1.0 at I = 1.9404 M and 6.49 at I = 3.0 M -- the
    correction that *reduces* D in dilute solution *multiplies* it by 6.5 in a
    3 M liquor. This function does not guard against that; callers must
    (:class:`difflow_ree.equilibrium.distribution.REEDistribution` clamps the
    ionic strength it feeds here unless extrapolation is explicitly requested).

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


# Aqueous activity models available to the distribution correlations (#194).
# Each model declares its own documented validity range so that using it
# outside that range is a reported extrapolation rather than a silent one.
# Bromley and SIT are deliberately absent: difflow_ree does not carry their
# ion-interaction parameters and will not invent them.
ACTIVITY_MODELS: dict[str, dict] = {
    "davies": {
        "max_ionic_strength": DAVIES_MAX_IONIC_STRENGTH,  # M
        # Ionic strength at which the model's own bracket changes sign, so that
        # ratios of its coefficients invert. None for models that cannot invert.
        "sign_change_ionic_strength": DAVIES_SIGN_CHANGE_IONIC_STRENGTH,
        "description": (
            "Davies equation, log10(gamma) = -A z^2 "
            "(sqrt(I)/(1+sqrt(I)) - 0.3 I), A = 0.509 at 25 C"
        ),
        "reference": "Davies, C.W. Ion Association, Butterworths, 1962.",
    },
    "none": {
        # No correction at all: the correlation is used as the conditional
        # constant it is, at the ionic strength it was fitted at. This is the
        # defensible option for a 2-4 M chloride liquor.
        "max_ionic_strength": float("inf"),
        "sign_change_ionic_strength": None,
        "description": (
            "No activity correction; the correlation is treated as a "
            "conditional constant valid at its fitting ionic strength"
        ),
        "reference": "n/a",
    },
}


def activity_coefficient(
    z: int,
    ionic_strength: Array | float,
    model: str = "davies",
) -> Array:
    """Aqueous activity coefficient from a named activity model (#194).

    Args:
        z: Ion charge.
        ionic_strength: Ionic strength (M).
        model: Model name, a key of :data:`ACTIVITY_MODELS`. ``"davies"`` uses
            the Davies equation (valid for I < 0.5 M); ``"none"`` returns 1.0,
            i.e. the correlation is used as a conditional constant.

    Returns:
        Activity coefficient (dimensionless).

    Raises:
        ValueError: If ``model`` is not an implemented activity model. Models
            whose ion-interaction parameters difflow_ree does not carry
            (Bromley, SIT) raise here rather than being approximated.
    """
    if model not in ACTIVITY_MODELS:
        raise ValueError(
            f"Unknown activity model {model!r}. Implemented models: "
            f"{sorted(ACTIVITY_MODELS)}. Models such as Bromley or SIT are "
            "not implemented because difflow_ree does not carry their "
            "ion-interaction parameters."
        )
    if model == "none":
        return jnp.ones_like(jnp.asarray(ionic_strength, dtype=float))
    return activity_coefficient_davies(z, ionic_strength)


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
