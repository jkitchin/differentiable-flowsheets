"""Extractant degradation models for solvent extraction.

Provides kinetic models for degradation of organophosphorus
and other extractants used in REE separation.

All functions use JAX numpy for automatic differentiation.

References:
    Ritcey GM, Ashbrook AW (1984). Solvent Extraction: Principles
        and Applications to Process Metallurgy. Elsevier.
    Paiva AP (1999). Hydrometallurgy 53:131.
"""

__all__ = [
    "oxidative_degradation_rate",
    "hydrolytic_degradation_rate",
    "solubility_loss_rate",
    "total_degradation_rate",
    "extractant_lifetime",
    "makeup_rate",
    "ExtractantDegradationModel",
    "ExtractantDegradationParams",
    "get_degradation_model",
]

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Physical Constants
# =============================================================================

R = 8.314  # J/mol/K


# =============================================================================
# Degradation Rate Functions
# =============================================================================

def oxidative_degradation_rate(
    C_extractant: Array | float,
    T: Array | float,
    k_ox: Array | float,
    E_a: Array | float = 60000.0,
    T_ref: Array | float = 298.15,
) -> Array:
    """Calculate oxidative degradation rate.

    dC/dt = -k_ox * C * exp(-E_a/R * (1/T - 1/T_ref))

    Args:
        C_extractant: Extractant concentration (M)
        T: Temperature (K)
        k_ox: Oxidation rate constant at T_ref (1/h)
        E_a: Activation energy (J/mol)
        T_ref: Reference temperature (K)

    Returns:
        Degradation rate (M/h)

    Notes:
        Organophosphorus extractants (D2EHPA, PC88A) are
        susceptible to oxidation at elevated temperatures.
        Typical k_ox: 1e-5 to 1e-4 /h
    """
    C_extractant = jnp.asarray(C_extractant)
    T = jnp.asarray(T)
    k_ox = jnp.asarray(k_ox)
    E_a = jnp.asarray(E_a)
    T_ref = jnp.asarray(T_ref)

    k_T = k_ox * jnp.exp(-E_a / R * (1/T - 1/T_ref))
    return k_T * C_extractant


def hydrolytic_degradation_rate(
    C_extractant: Array | float,
    C_H2O: Array | float,
    T: Array | float,
    k_hyd: Array | float,
    pH: Array | float = 3.0,
) -> Array:
    """Calculate hydrolytic degradation rate.

    Acid-catalyzed hydrolysis of extractant.

    Args:
        C_extractant: Extractant concentration (M)
        C_H2O: Water content in organic (M)
        T: Temperature (K)
        k_hyd: Hydrolysis rate constant (1/M/h)
        pH: Aqueous pH (affects H+ carryover)

    Returns:
        Degradation rate (M/h)

    Notes:
        Hydrolysis is promoted by:
        - High water content in organic
        - Low pH (acid catalysis)
        - Elevated temperature
    """
    C_extractant = jnp.asarray(C_extractant)
    C_H2O = jnp.asarray(C_H2O)
    T = jnp.asarray(T)
    k_hyd = jnp.asarray(k_hyd)
    pH = jnp.asarray(pH)

    # Acid catalysis factor
    C_H = jnp.power(10.0, -pH)
    acid_factor = jnp.sqrt(C_H / 1e-3)  # Normalized to pH 3

    # Temperature factor
    T_factor = jnp.exp(5000 * (1/298.15 - 1/T))

    return k_hyd * C_extractant * C_H2O * acid_factor * T_factor


def solubility_loss_rate(
    Q_org: Array | float,
    Q_aq: Array | float,
    C_extractant: Array | float,
    S_extractant: Array | float = 1e-5,
) -> Array:
    """Calculate extractant loss to aqueous phase.

    Entrainment and solubility losses.

    Args:
        Q_org: Organic flow rate (L/h)
        Q_aq: Aqueous flow rate (L/h)
        C_extractant: Extractant concentration (M)
        S_extractant: Solubility in aqueous (M)

    Returns:
        Loss rate (mol/h)

    Notes:
        D2EHPA solubility: ~10^-5 M
        PC88A solubility: ~10^-6 M
        Losses increase with A/O ratio
    """
    Q_org = jnp.asarray(Q_org)
    Q_aq = jnp.asarray(Q_aq)
    C_extractant = jnp.asarray(C_extractant)
    S_extractant = jnp.asarray(S_extractant)

    # Solubility loss
    solubility_loss = Q_aq * S_extractant

    # Entrainment (assume 0.1% of organic entrained)
    entrainment_fraction = 0.001
    entrainment_loss = Q_org * C_extractant * entrainment_fraction * (Q_aq / Q_org)

    return solubility_loss + entrainment_loss


def total_degradation_rate(
    C_extractant: Array | float,
    T: Array | float,
    k_ox: Array | float,
    k_hyd: Array | float,
    C_H2O: Array | float = 0.1,
    pH: Array | float = 3.0,
) -> Array:
    """Calculate total degradation rate from all pathways.

    Args:
        C_extractant: Extractant concentration (M)
        T: Temperature (K)
        k_ox: Oxidation rate constant (1/h)
        k_hyd: Hydrolysis rate constant (1/M/h)
        C_H2O: Water in organic (M)
        pH: Operating pH

    Returns:
        Total degradation rate (M/h)
    """
    r_ox = oxidative_degradation_rate(C_extractant, T, k_ox)
    r_hyd = hydrolytic_degradation_rate(C_extractant, C_H2O, T, k_hyd, pH)

    return r_ox + r_hyd


def extractant_lifetime(
    k_total: Array | float,
    target_depletion: Array | float = 0.1,
) -> Array:
    """Calculate extractant lifetime.

    Time for extractant to deplete by target fraction.

    Args:
        k_total: Total first-order degradation rate (1/h)
        target_depletion: Fraction depleted (default 10%)

    Returns:
        Lifetime (h)
    """
    k_total = jnp.asarray(k_total)
    target_depletion = jnp.asarray(target_depletion)

    return -jnp.log(1 - target_depletion) / k_total


def makeup_rate(
    degradation_rate: Array | float,
    loss_rate: Array | float,
    V_inventory: Array | float,
) -> Array:
    """Calculate required makeup rate.

    Args:
        degradation_rate: Chemical degradation (M/h)
        loss_rate: Physical losses (mol/h)
        V_inventory: Total solvent inventory (L)

    Returns:
        Makeup rate (mol/h)
    """
    degradation_rate = jnp.asarray(degradation_rate)
    loss_rate = jnp.asarray(loss_rate)
    V_inventory = jnp.asarray(V_inventory)

    return degradation_rate * V_inventory + loss_rate


# =============================================================================
# Degradation Model Class
# =============================================================================

@dataclass(repr=False)
class ExtractantDegradationParams(ParamsMixin):
    """Parameters for extractant degradation model.

    Attributes:
        extractant: Extractant name
        k_ox: Oxidation rate constant (1/h)
        k_hyd: Hydrolysis rate constant (1/M/h)
        S_aq: Aqueous solubility (M)
        E_a_ox: Activation energy for oxidation (J/mol)
        C_H2O: Water content in organic (M)
    """
    extractant: str = "D2EHPA"
    k_ox: float | Array = 1e-5  # 1/h
    k_hyd: float | Array = 1e-6  # 1/M/h
    S_aq: float | Array = 1e-5  # M
    E_a_ox: float | Array = 60000.0  # J/mol
    C_H2O: float | Array = 0.1  # M


# Typical values by extractant
EXTRACTANT_DEGRADATION_DATA = {
    "D2EHPA": {
        "k_ox": 1e-5,
        "k_hyd": 1e-6,
        "S_aq": 1e-5,
    },
    "PC88A": {
        "k_ox": 5e-6,
        "k_hyd": 5e-7,
        "S_aq": 1e-6,
    },
    "EHEHPA": {
        "k_ox": 2e-5,
        "k_hyd": 2e-6,
        "S_aq": 5e-6,
    },
    "Cyanex272": {
        "k_ox": 1e-5,
        "k_hyd": 1e-6,
        "S_aq": 2e-6,
    },
}


class ExtractantDegradationModel:
    """Unified extractant degradation model.

    Example:
        >>> model = ExtractantDegradationModel(ExtractantDegradationParams())
        >>> rate = model.total_rate(C=0.5, T=323.15, pH=3.0)
    """

    def __init__(self, params: ExtractantDegradationParams):
        self.params = params

        # Load extractant-specific data if available
        if params.extractant in EXTRACTANT_DEGRADATION_DATA:
            data = EXTRACTANT_DEGRADATION_DATA[params.extractant]
            # Use data as defaults if params not explicitly set
            self._k_ox = data["k_ox"] if params.k_ox is None else params.k_ox
            self._k_hyd = data["k_hyd"] if params.k_hyd is None else params.k_hyd
            self._S_aq = data["S_aq"] if params.S_aq is None else params.S_aq
        else:
            self._k_ox = params.k_ox
            self._k_hyd = params.k_hyd
            self._S_aq = params.S_aq

    def oxidation_rate(
        self,
        C: Array | float,
        T: Array | float,
    ) -> Array:
        """Calculate oxidative degradation rate."""
        return oxidative_degradation_rate(
            C, T, self._k_ox, self.params.E_a_ox
        )

    def hydrolysis_rate(
        self,
        C: Array | float,
        T: Array | float,
        pH: Array | float,
    ) -> Array:
        """Calculate hydrolytic degradation rate."""
        return hydrolytic_degradation_rate(
            C, self.params.C_H2O, T, self._k_hyd, pH
        )

    def total_rate(
        self,
        C: Array | float,
        T: Array | float,
        pH: Array | float = 3.0,
    ) -> Array:
        """Calculate total degradation rate."""
        return self.oxidation_rate(C, T) + self.hydrolysis_rate(C, T, pH)

    def annual_loss(
        self,
        C: Array | float,
        T: Array | float,
        V_inventory: Array | float,
        Q_org: Array | float,
        Q_aq: Array | float,
        pH: Array | float = 3.0,
    ) -> dict:
        """Calculate annual extractant losses and costs.

        Returns:
            Dict with degradation, solubility, entrainment losses
        """
        hours_per_year = 8760

        # Chemical degradation
        r_deg = float(self.total_rate(C, T, pH))
        deg_loss_kg = r_deg * V_inventory * hours_per_year * 0.3  # MW ~300

        # Physical losses
        r_loss = float(solubility_loss_rate(Q_org, Q_aq, C, self._S_aq))
        phys_loss_kg = r_loss * hours_per_year * 0.3

        return {
            "degradation_kg_yr": deg_loss_kg,
            "physical_loss_kg_yr": phys_loss_kg,
            "total_loss_kg_yr": deg_loss_kg + phys_loss_kg,
        }


def get_degradation_model(
    extractant: str = "D2EHPA",
    **kwargs,
) -> ExtractantDegradationModel:
    """Create degradation model for specified extractant."""
    params = ExtractantDegradationParams(extractant=extractant, **kwargs)
    return ExtractantDegradationModel(params)
