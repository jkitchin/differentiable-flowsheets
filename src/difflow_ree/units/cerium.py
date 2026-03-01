"""Cerium oxidation and separation unit operations.

Cerium is unique among lanthanides because it can be oxidized
from Ce³⁺ to Ce⁴⁺, which has very different chemistry:
- Ce⁴⁺ is much less soluble as CeO₂
- Ce⁴⁺ doesn't extract well with acidic extractants
- Enables selective Ce removal from mixed REE

This is industrially important because:
- Ce is often 40-50% of REE ores
- Removing Ce simplifies downstream separation
- CeO₂ has direct commercial applications

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows


# =============================================================================
# Cerium Oxidation Parameters
# =============================================================================

@dataclass(repr=False)
class CeriumOxidizerParams(ParamsMixin):
    """Parameters for cerium oxidation unit.

    Attributes:
        elements: All REE elements to track
        oxidant: Oxidizing agent (air, H2O2, NaOCl, electrolytic)
        oxidant_excess: Molar excess of oxidant
        pH: Operating pH (higher pH favors oxidation)
        temperature: Operating temperature (K)
        ce_conversion: Target Ce³⁺ → Ce⁴⁺ conversion
    """
    elements: tuple[str, ...]
    oxidant: Literal["air", "H2O2", "NaOCl", "electrolytic"] = "air"
    oxidant_excess: float = 2.0
    pH: float = 8.0  # Alkaline conditions favor Ce oxidation
    temperature: float = 353.15  # 80°C typical
    ce_conversion: float = 0.95


class CeriumOxidizer:
    """Cerium oxidation and precipitation unit.

    Oxidizes Ce³⁺ to Ce⁴⁺ and precipitates as CeO₂.

    Reaction sequence:
    1. 2Ce³⁺ + ½O₂ + 2OH⁻ → 2Ce⁴⁺ + H₂O + O²⁻
    2. Ce⁴⁺ + 2OH⁻ → CeO₂↓ + H₂O

    Or combined: 2Ce³⁺ + ½O₂ + 2OH⁻ → 2CeO₂↓ + H₂O

    Example:
        >>> params = CeriumOxidizerParams(
        ...     elements=("La", "Ce", "Pr", "Nd"),
        ...     oxidant="air",
        ...     pH=8.0,
        ... )
        >>> oxidizer = CeriumOxidizer(params)
        >>> filtrate, ceo2_solid, info = oxidizer(feed)
    """

    def __init__(self, params: CeriumOxidizerParams):
        """Initialize oxidizer.

        Args:
            params: Oxidizer parameters
        """
        self.params = params

    def __call__(
        self,
        feed: Stream,
        T: Array | float | None = None,
        pH: Array | float | None = None,
    ) -> tuple[Stream, dict, dict]:
        """Perform cerium oxidation and precipitation.

        Args:
            feed: Aqueous REE solution containing Ce
            T: Temperature (K)
            pH: Operating pH

        Returns:
            filtrate: Ce-depleted REE solution
            solid: CeO₂ precipitate (mol Ce/s)
            info: Process diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        pH = pH if pH is not None else p.pH
        T = jnp.asarray(T)
        pH = jnp.asarray(pH)

        feed_flows = get_flows(feed)

        # Check for Ce in feed
        if "Ce" not in p.elements:
            raise ValueError("Ce must be in elements list for CeriumOxidizer")

        F_Ce_in = jnp.asarray(feed_flows.get("Ce", 0.0))

        # Oxidation kinetics depend on:
        # - pH (higher = faster, more complete)
        # - Temperature (higher = faster)
        # - Oxidant type and excess

        # pH effect: oxidation potential increases with pH
        # E = E° - 0.059 × pH
        # At pH 8, Ce oxidation is thermodynamically favorable
        pH_factor = jnp.clip((pH - 6) / 4, 0.0, 1.0)  # 0 at pH 6, 1 at pH 10

        # Temperature effect (Arrhenius-like)
        T_ref = 353.15  # 80°C
        T_factor = jnp.exp(-3000 * (1/T - 1/T_ref))  # Ea ~ 25 kJ/mol

        # Oxidant efficiency
        oxidant_factors = {
            "air": 0.85,  # Slow but cheap
            "H2O2": 0.95,  # Fast and effective
            "NaOCl": 0.98,  # Very effective
            "electrolytic": 0.99,  # Most controlled
        }
        oxidant_factor = oxidant_factors.get(p.oxidant, 0.9)

        # Overall conversion
        base_conversion = p.ce_conversion
        actual_conversion = base_conversion * pH_factor * T_factor * oxidant_factor
        actual_conversion = jnp.clip(actual_conversion, 0.0, 0.999)

        # Ce removed as CeO2
        F_Ce_oxidized = F_Ce_in * actual_conversion
        F_Ce_remaining = F_Ce_in * (1 - actual_conversion)

        # Other REE pass through unchanged
        filtrate_flows = {"H2O": feed_flows.get("H2O", 1.0)}
        for elem in p.elements:
            if elem == "Ce":
                filtrate_flows[elem] = jnp.maximum(F_Ce_remaining, 0.0)
            else:
                filtrate_flows[elem] = jnp.asarray(feed_flows.get(elem, 0.0))

        # CeO2 solid product
        solid_flows = {"Ce": jnp.maximum(F_Ce_oxidized, 0.0)}

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        # Calculate Ce removal efficiency
        total_ree_in = sum(float(feed_flows.get(e, 0.0)) for e in p.elements)
        total_ree_out = sum(float(filtrate_flows.get(e, 0.0)) for e in p.elements)
        ce_fraction_in = safe_divide(float(F_Ce_in), total_ree_in)
        ce_fraction_out = safe_divide(float(F_Ce_remaining), total_ree_out)

        # Calculate oxidant consumption based on stoichiometry
        # 4Ce3+ + O2 + 4OH- -> 4CeO2 + 2H2O (for air/O2)
        # 2Ce3+ + H2O2 + 2OH- -> 2CeO2 + 2H2O (for H2O2)
        # 2Ce3+ + NaOCl + 2OH- -> 2CeO2 + NaCl + H2O (for NaOCl)
        # Ce3+ -> Ce4+ + e- (electrolytic: 1 Faraday per mol Ce)
        oxidant_stoich = {
            "air": 0.25,        # 0.25 mol O2 per mol Ce (O2 is 4-electron oxidant)
            "H2O2": 0.5,        # 0.5 mol H2O2 per mol Ce (2-electron oxidant)
            "NaOCl": 0.5,       # 0.5 mol NaOCl per mol Ce
            "electrolytic": 1.0, # 1 Faraday per mol Ce
        }
        stoich_ratio = oxidant_stoich.get(p.oxidant, 0.5)
        oxidant_consumed = F_Ce_oxidized * stoich_ratio * p.oxidant_excess

        # Electrons transferred (Ce3+ -> Ce4+ is 1-electron oxidation)
        electrons_transferred = F_Ce_oxidized  # mol e-/s

        info = {
            "oxidant": p.oxidant,
            "pH": float(pH),
            "T": float(T),
            "ce_conversion": float(actual_conversion),
            "ce_removed_mol_s": float(F_Ce_oxidized),
            "ce_fraction_in": ce_fraction_in,
            "ce_fraction_out": ce_fraction_out,
            "ceo2_mass_kg_s": float(F_Ce_oxidized) * 172.12 / 1000,  # CeO2 MW
            "other_ree_recovery": 1.0,  # Other REE unaffected
            "oxidant_consumed_mol_s": float(oxidant_consumed),
            "oxidant_stoich_ratio": stoich_ratio,
            "electrons_transferred_mol_s": float(electrons_transferred),
            "oxidant_excess": p.oxidant_excess,
        }

        return filtrate, solid_flows, info


# =============================================================================
# Alternative Ce Separation Methods
# =============================================================================

class CeriumSolventExtraction:
    """Selective Ce extraction using Ce⁴⁺ chemistry.

    Ce⁴⁺ has different extraction behavior than Ce³⁺:
    - Does NOT extract with D2EHPA, PC88A (acidic extractants)
    - CAN extract with TBP from nitric acid

    This enables selective rejection of Ce during extraction.
    """

    def __init__(
        self,
        elements: tuple[str, ...],
        oxidize_first: bool = True,
    ):
        """Initialize Ce solvent extraction separator.

        Args:
            elements: REE elements to track
            oxidize_first: Whether to oxidize Ce before extraction
        """
        self.elements = elements
        self.oxidize_first = oxidize_first

    def ce_rejection_factor(
        self,
        pH: float,
        oxidation_conversion: float = 0.9,
    ) -> float:
        """Calculate Ce rejection factor during extraction.

        Ce⁴⁺ has D ≈ 0 with acidic extractants.

        Args:
            pH: Extraction pH
            oxidation_conversion: Fraction of Ce oxidized to Ce⁴⁺

        Returns:
            Rejection factor (fraction of Ce NOT extracted)
        """
        # Ce³⁺ extracts normally, Ce⁴⁺ does not
        # Rejection = fraction as Ce⁴⁺
        return oxidation_conversion


class CeriumIonExchange:
    """Ion exchange separation of Ce⁴⁺.

    Ce⁴⁺ behaves differently on ion exchange resins:
    - Stronger binding as Ce⁴⁺ due to higher charge
    - Can be selectively eluted

    Used in some specialty separation processes.
    """

    def __init__(
        self,
        elements: tuple[str, ...],
        resin_type: str = "strong_acid_cation",
    ):
        """Initialize ion exchange separator.

        Args:
            elements: REE elements
            resin_type: Type of ion exchange resin
        """
        self.elements = elements
        self.resin_type = resin_type


# =============================================================================
# Utility Functions
# =============================================================================

def ce_oxidation_potential(pH: float, T: float = 298.15) -> float:
    """Calculate Ce³⁺/Ce⁴⁺ oxidation potential.

    Ce⁴⁺ + e⁻ → Ce³⁺   E° = +1.72 V (in acid)

    At higher pH, potential decreases:
    E = E° - 0.059 × pH

    Args:
        pH: Solution pH
        T: Temperature (K)

    Returns:
        Oxidation potential (V vs SHE)
    """
    E0 = 1.72  # Standard potential
    # Nernst equation correction for pH
    E = E0 - 0.059 * pH
    return E


def oxygen_requirement(
    ce_mol: float,
    oxidant: str = "air",
) -> dict[str, float]:
    """Calculate oxidant requirement for Ce oxidation.

    Reaction: 4Ce³⁺ + O₂ + 4OH⁻ → 4CeO₂ + 2H₂O

    Args:
        ce_mol: Moles of Ce to oxidize
        oxidant: Type of oxidant

    Returns:
        Dictionary with oxidant requirements
    """
    if oxidant == "air":
        # O2 is 21% of air
        o2_mol = ce_mol / 4
        air_mol = o2_mol / 0.21
        return {
            "O2_mol": o2_mol,
            "air_mol": air_mol,
            "air_L_STP": air_mol * 22.4,
        }
    elif oxidant == "H2O2":
        # 2Ce³⁺ + H₂O₂ + 2OH⁻ → 2CeO₂ + 2H₂O
        h2o2_mol = ce_mol / 2
        return {
            "H2O2_mol": h2o2_mol,
            "H2O2_kg": h2o2_mol * 34 / 1000,
        }
    elif oxidant == "NaOCl":
        # Ce³⁺ + ½NaOCl + OH⁻ → CeO₂ + ½NaCl + ½H₂O
        naocl_mol = ce_mol / 2
        return {
            "NaOCl_mol": naocl_mol,
            "NaOCl_kg": naocl_mol * 74.5 / 1000,
        }
    else:
        return {"note": "Electrolytic oxidation - calculate from current"}


def ceo2_product_value(
    ce_mol: float,
    purity: float = 0.995,
    grade: str = "standard",
) -> float:
    """Calculate value of CeO₂ product.

    Args:
        ce_mol: Moles of Ce in product
        purity: Product purity (0-1)
        grade: Product grade (standard, polishing, electronic)

    Returns:
        Product value (USD)
    """
    # CeO₂ prices (USD/kg) by grade
    prices = {
        "standard": 3.0,  # General industrial
        "polishing": 15.0,  # Glass/lens polishing
        "electronic": 50.0,  # High purity electronic grade
    }

    ceo2_kg = ce_mol * 172.12 / 1000  # CeO2 MW = 172.12
    base_price = prices.get(grade, 3.0)

    # Purity premium
    base_price = jnp.asarray(base_price, dtype=jnp.float64)
    price = jnp.where(purity > 0.999, base_price * 1.5,
            jnp.where(purity > 0.99, base_price * 1.2, base_price))

    return ceo2_kg * price
