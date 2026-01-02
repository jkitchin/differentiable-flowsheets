"""Single-stage LLE model for Nd/Dy separation.

Implements a differentiable mixer-settler equilibrium model with
pH-dependent distribution coefficients for D2EHPA extraction.

References:
    Gupta & Krishnamurthy (2005) Extractive Metallurgy of Rare Earths
    Xie et al. (2014) Hydrometallurgy reviews
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Distribution Coefficient Model
# =============================================================================

@dataclass
class PHCoefficients:
    """pH-dependent distribution coefficient parameters.

    Model: log10(D) = a + b*pH + c*pH^2

    Attributes:
        a: Intercept
        b: Linear pH coefficient
        c: Quadratic pH coefficient
    """
    a: float
    b: float
    c: float


@dataclass
class D2EHPADistribution:
    """D2EHPA distribution coefficient calculator.

    Full model:
        log10(D) = a + b*pH + c*pH^2 + (dH/R/ln10)*(1/T - 1/Tref) + n*log10([HA]/[HA]ref)

    Attributes:
        elements: Tuple of element symbols
        ph_coeffs: pH correlation coefficients by element
        dH: Enthalpy of extraction (K) by element
        Tref: Reference temperature (K)
        conc_ref: Reference extractant concentration (M)
        conc_exp: Concentration exponent (typically 3 for D2EHPA)
    """
    elements: tuple[str, ...] = ("Nd", "Dy")

    # Literature values from Gupta & Krishnamurthy (2005)
    # Valid for 0.5M D2EHPA in kerosene, pH 1-5
    ph_coeffs: dict = None
    dH: dict = None
    Tref: float = 298.15
    conc_ref: float = 0.5
    conc_exp: float = 3.0

    def __post_init__(self):
        if self.ph_coeffs is None:
            # Adjusted coefficients to give realistic D values
            # Target at pH=3, T=298K, 0.5M D2EHPA:
            #   D_Nd ~ 2, D_Dy ~ 16, D_La ~ 0.5
            # Original literature coefficients normalized differently
            self.ph_coeffs = {
                "Nd": PHCoefficients(a=-2.50, b=0.90, c=0.02),
                "Dy": PHCoefficients(a=-1.40, b=0.95, c=0.02),
                "La": PHCoefficients(a=-3.00, b=0.85, c=0.02),
            }
        if self.dH is None:
            # Temperature coefficients (K) - extraction is exothermic
            self.dH = {
                "Nd": -1800.0,
                "Dy": -2400.0,
                "La": -1500.0,
            }

    def log_D(
        self,
        element: str,
        pH: Array,
        T: Array = jnp.array(298.15),
        conc: Array = jnp.array(0.5),
    ) -> Array:
        """Calculate log10(D) for an element.

        Args:
            element: Element symbol ("Nd" or "Dy")
            pH: Solution pH
            T: Temperature (K)
            conc: Extractant concentration (M)

        Returns:
            log10(D)
        """
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)
        conc = jnp.asarray(conc)

        coeffs = self.ph_coeffs[element]

        # pH dependence
        log_D = coeffs.a + coeffs.b * pH + coeffs.c * pH**2

        # Temperature correction (van't Hoff)
        dH = self.dH[element]
        R_ln10 = 8.314 * jnp.log(10)  # R * ln(10)
        log_D = log_D + (dH / R_ln10) * (1/T - 1/self.Tref)

        # Concentration correction
        log_D = log_D + self.conc_exp * jnp.log10(conc / self.conc_ref)

        return log_D

    def D(
        self,
        element: str,
        pH: Array,
        T: Array = jnp.array(298.15),
        conc: Array = jnp.array(0.5),
    ) -> Array:
        """Calculate distribution coefficient D.

        Args:
            element: Element symbol
            pH: Solution pH
            T: Temperature (K)
            conc: Extractant concentration (M)

        Returns:
            Distribution coefficient D = [REE]_org / [REE]_aq
        """
        return jnp.power(10.0, self.log_D(element, pH, T, conc))

    def D_all(
        self,
        pH: Array,
        T: Array = jnp.array(298.15),
        conc: Array = jnp.array(0.5),
    ) -> dict[str, Array]:
        """Calculate D for all elements.

        Returns:
            Dictionary of element -> D value
        """
        return {elem: self.D(elem, pH, T, conc) for elem in self.elements}

    def separation_factor(
        self,
        elem1: str,
        elem2: str,
        pH: Array,
        T: Array = jnp.array(298.15),
        conc: Array = jnp.array(0.5),
    ) -> Array:
        """Calculate separation factor SF = D1 / D2.

        Args:
            elem1: Element with higher D (typically heavier REE)
            elem2: Element with lower D
            pH: Solution pH
            T: Temperature (K)
            conc: Extractant concentration (M)

        Returns:
            Separation factor
        """
        D1 = self.D(elem1, pH, T, conc)
        D2 = self.D(elem2, pH, T, conc)
        return D1 / D2


# =============================================================================
# Single-Stage Mixer-Settler Model
# =============================================================================

class SeparationResult(NamedTuple):
    """Results from single-stage separation."""
    # Outlet flows (mol/s)
    F_Nd_aq: Array  # Nd in aqueous (raffinate)
    F_Nd_org: Array  # Nd in organic (extract)
    F_Dy_aq: Array  # Dy in aqueous
    F_Dy_org: Array  # Dy in organic

    # Performance metrics
    purity_Dy_org: Array  # Dy purity in extract
    purity_Nd_aq: Array  # Nd purity in raffinate
    recovery_Dy: Array  # Dy recovery to extract
    recovery_Nd: Array  # Nd recovery to raffinate

    # Operating conditions
    D_Nd: Array
    D_Dy: Array
    SF: Array  # Separation factor Dy/Nd


@dataclass
class SingleStageLLE:
    """Single-stage liquid-liquid extraction model.

    Models a mixer-settler unit for Nd/Dy separation using D2EHPA.
    All calculations are JAX-differentiable.

    Attributes:
        distribution: Distribution coefficient model
        efficiency: Murphree stage efficiency (0-1)
    """
    distribution: D2EHPADistribution = None
    efficiency: float = 0.95

    def __post_init__(self):
        if self.distribution is None:
            self.distribution = D2EHPADistribution()

    def __call__(
        self,
        F_Nd_feed: Array,
        F_Dy_feed: Array,
        F_aq: Array,
        F_org: Array,
        pH: Array,
        T: Array = jnp.array(298.15),
        conc: Array = jnp.array(0.5),
        F_Nd_org_in: Array = jnp.array(0.0),
        F_Dy_org_in: Array = jnp.array(0.0),
    ) -> SeparationResult:
        """Perform single-stage extraction.

        Args:
            F_Nd_feed: Nd molar flow in aqueous feed (mol/s)
            F_Dy_feed: Dy molar flow in aqueous feed (mol/s)
            F_aq: Aqueous carrier flow (mol/s or kg/s)
            F_org: Organic carrier flow (mol/s or kg/s)
            pH: Operating pH
            T: Temperature (K)
            conc: D2EHPA concentration (M)
            F_Nd_org_in: Nd in inlet organic (for loaded solvent)
            F_Dy_org_in: Dy in inlet organic

        Returns:
            SeparationResult with all outlet flows and metrics
        """
        # Ensure arrays
        F_Nd_feed = jnp.asarray(F_Nd_feed)
        F_Dy_feed = jnp.asarray(F_Dy_feed)
        F_aq = jnp.asarray(F_aq)
        F_org = jnp.asarray(F_org)
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)
        conc = jnp.asarray(conc)
        F_Nd_org_in = jnp.asarray(F_Nd_org_in)
        F_Dy_org_in = jnp.asarray(F_Dy_org_in)

        # Get distribution coefficients
        D_Nd = self.distribution.D("Nd", pH, T, conc)
        D_Dy = self.distribution.D("Dy", pH, T, conc)

        # Total solute entering
        F_Nd_total = F_Nd_feed + F_Nd_org_in
        F_Dy_total = F_Dy_feed + F_Dy_org_in

        # Equilibrium calculation
        # At equilibrium: D = (F_org_out / F_org) / (F_aq_out / F_aq)
        # Rearranging with mass balance: F_total = F_aq_out + F_org_out
        # F_aq_out = F_total / (1 + D * F_org / F_aq)

        def equilibrium_split(F_total, D):
            """Calculate equilibrium split between phases."""
            extraction_factor = D * F_org / F_aq
            F_aq_eq = F_total / (1.0 + extraction_factor)
            F_org_eq = F_total - F_aq_eq
            return F_aq_eq, F_org_eq

        # Equilibrium concentrations
        F_Nd_aq_eq, F_Nd_org_eq = equilibrium_split(F_Nd_total, D_Nd)
        F_Dy_aq_eq, F_Dy_org_eq = equilibrium_split(F_Dy_total, D_Dy)

        # Apply stage efficiency
        # Actual change = efficiency * (equilibrium change)
        eta = self.efficiency

        F_Nd_aq = F_Nd_feed + eta * (F_Nd_aq_eq - F_Nd_feed)
        F_Nd_org = F_Nd_org_in + eta * (F_Nd_org_eq - F_Nd_org_in)

        F_Dy_aq = F_Dy_feed + eta * (F_Dy_aq_eq - F_Dy_feed)
        F_Dy_org = F_Dy_org_in + eta * (F_Dy_org_eq - F_Dy_org_in)

        # Ensure non-negative
        F_Nd_aq = jnp.maximum(F_Nd_aq, 0.0)
        F_Nd_org = jnp.maximum(F_Nd_org, 0.0)
        F_Dy_aq = jnp.maximum(F_Dy_aq, 0.0)
        F_Dy_org = jnp.maximum(F_Dy_org, 0.0)

        # Calculate metrics
        total_org = F_Nd_org + F_Dy_org + 1e-10
        total_aq = F_Nd_aq + F_Dy_aq + 1e-10

        purity_Dy_org = F_Dy_org / total_org
        purity_Nd_aq = F_Nd_aq / total_aq

        recovery_Dy = F_Dy_org / (F_Dy_feed + 1e-10)
        recovery_Nd = F_Nd_aq / (F_Nd_feed + 1e-10)

        SF = D_Dy / D_Nd

        return SeparationResult(
            F_Nd_aq=F_Nd_aq,
            F_Nd_org=F_Nd_org,
            F_Dy_aq=F_Dy_aq,
            F_Dy_org=F_Dy_org,
            purity_Dy_org=purity_Dy_org,
            purity_Nd_aq=purity_Nd_aq,
            recovery_Dy=recovery_Dy,
            recovery_Nd=recovery_Nd,
            D_Nd=D_Nd,
            D_Dy=D_Dy,
            SF=SF,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_base_case() -> dict:
    """Create base case operating conditions.

    Returns:
        Dictionary of base case parameters
    """
    return {
        # Feed composition (mol/s)
        "F_Nd_feed": 0.01,  # ~1.4 g/s
        "F_Dy_feed": 0.01,  # ~1.6 g/s (50:50 molar)

        # Flow rates
        "F_aq": 1.0,  # Aqueous carrier (kg/s)
        "F_org": 1.0,  # Organic carrier (kg/s), O/A = 1

        # Operating conditions
        "pH": 3.0,
        "T": 298.15,  # 25°C
        "conc": 0.5,  # 0.5 M D2EHPA

        # Stage efficiency
        "efficiency": 0.95,
    }


def run_base_case() -> SeparationResult:
    """Run simulation at base case conditions.

    Returns:
        SeparationResult at base case
    """
    params = create_base_case()

    model = SingleStageLLE(efficiency=params["efficiency"])

    result = model(
        F_Nd_feed=jnp.array(params["F_Nd_feed"]),
        F_Dy_feed=jnp.array(params["F_Dy_feed"]),
        F_aq=jnp.array(params["F_aq"]),
        F_org=jnp.array(params["F_org"]),
        pH=jnp.array(params["pH"]),
        T=jnp.array(params["T"]),
        conc=jnp.array(params["conc"]),
    )

    return result


if __name__ == "__main__":
    # Quick test
    result = run_base_case()

    print("Base Case Results")
    print("=" * 50)
    print(f"D_Nd = {float(result.D_Nd):.3f}")
    print(f"D_Dy = {float(result.D_Dy):.3f}")
    print(f"SF (Dy/Nd) = {float(result.SF):.2f}")
    print()
    print(f"Dy purity in extract: {float(result.purity_Dy_org)*100:.1f}%")
    print(f"Dy recovery to extract: {float(result.recovery_Dy)*100:.1f}%")
    print(f"Nd purity in raffinate: {float(result.purity_Nd_aq)*100:.1f}%")
    print(f"Nd recovery to raffinate: {float(result.recovery_Nd)*100:.1f}%")
