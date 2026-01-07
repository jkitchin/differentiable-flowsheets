"""REE precipitation unit operations.

Precipitation is used to:
- Produce solid REE products (oxalate → oxide)
- Perform group separations
- Purify REE solutions

Common precipitants:
- Oxalic acid: REE₂(C₂O₄)₃ (calcined to oxide)
- Carbonate: REE₂(CO₃)₃
- Hydroxide: REE(OH)₃

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass, replace, fields, asdict as dc_asdict
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.database import get_ree_database


# =============================================================================
# Solubility Products (pKsp at 25°C)
# =============================================================================

# REE₂(C₂O₄)₃ solubility products
PKsp_OXALATE = {
    "La": 25.0, "Ce": 25.5, "Pr": 26.0, "Nd": 26.2,
    "Sm": 26.8, "Eu": 27.0, "Gd": 27.2, "Tb": 27.5,
    "Dy": 27.8, "Y": 27.0,
}

# REE₂(CO₃)₃ solubility products
PKsp_CARBONATE = {
    "La": 30.0, "Ce": 30.5, "Pr": 31.0, "Nd": 31.2,
    "Sm": 31.8, "Eu": 32.0, "Gd": 32.2, "Tb": 32.5,
    "Dy": 32.8, "Y": 32.0,
}

# REE(OH)₃ solubility products
PKsp_HYDROXIDE = {
    "La": 19.0, "Ce": 19.5, "Pr": 20.0, "Nd": 20.2,
    "Sm": 21.0, "Eu": 21.5, "Gd": 22.0, "Tb": 22.5,
    "Dy": 23.0, "Y": 22.0,
}


# =============================================================================
# Precipitator Parameters
# =============================================================================

@dataclass
class PrecipitatorParams:
    """Parameters for REE precipitation.

    Attributes:
        elements: REE elements to track
        precipitant_excess: Molar excess of precipitant (1.0 = stoichiometric)
        temperature: Operating temperature (K)
        residence_time: Reactor residence time (s)
        target_conversion: Target precipitation conversion (0-1)
    """
    elements: tuple[str, ...]
    precipitant_excess: float = 1.5  # 50% excess typical
    temperature: float = 298.15
    residence_time: float = 3600.0  # 1 hour typical
    target_conversion: float = 0.995

    def update(self, **kwargs) -> "PrecipitatorParams":
        """Return a new PrecipitatorParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., precipitant_excess=2.0)

        Returns:
            New PrecipitatorParams with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the params."""
        return key in {f.name for f in fields(self)}

    def keys(self):
        """Return field names for dict-like iteration."""
        return (f.name for f in fields(self))

    def values(self):
        """Return field values for dict-like iteration.

        Returns:
            Iterator over field values
        """
        return (getattr(self, f.name) for f in fields(self))

    def items(self):
        """Return (name, value) pairs for dict-like iteration.

        Returns:
            Iterator over (field_name, value) tuples
        """
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def __iter__(self):
        """Iterate over field names (like dict)."""
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        """Return number of fields."""
        return len(fields(self))

    def asdict(self) -> dict:
        """Convert params to a dictionary."""
        return dc_asdict(self)

    def __repr__(self) -> str:
        """Concise string representation."""
        def fmt(v):
            if v is None:
                return "None"
            if callable(v) and hasattr(v, '__name__'):
                return v.__name__
            if hasattr(v, 'shape'):
                if v.ndim == 0:
                    return f"{float(v):.4g}"
                return f"Array{list(v.shape)}"
            if isinstance(v, dict):
                items = ", ".join(f"{k}: {fmt(val)}" for k, val in v.items())
                return "{" + items + "}"
            if isinstance(v, (list, tuple)) and len(v) > 5:
                return f"{type(v).__name__}[{len(v)}]"
            return repr(v)
        items = ", ".join(f"{f.name}={fmt(getattr(self, f.name))}" for f in fields(self))
        return f"{self.__class__.__name__}({items})"


# =============================================================================
# Oxalate Precipitation
# =============================================================================

class OxalatePrecipitator:
    """Oxalate precipitation for REE recovery.

    Reaction: 2REE³⁺ + 3C₂O₄²⁻ → REE₂(C₂O₄)₃↓

    Oxalate precipitation produces high-purity REE product
    that can be calcined to oxide:
    REE₂(C₂O₄)₃ → REE₂O₃ + 3CO + 3CO₂

    Example:
        >>> params = PrecipitatorParams(
        ...     elements=("Nd", "Dy"),
        ...     precipitant_excess=1.5,
        ... )
        >>> precip = OxalatePrecipitator(params)
        >>> filtrate, solid, info = precip(feed, oxalic_acid)
    """

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, dict, dict]:
        """Perform oxalate precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Oxalic acid solution
            T: Temperature (K)

        Returns:
            filtrate: Aqueous filtrate (depleted in REE)
            solid: Dictionary of precipitated REE (mol/s)
            info: Precipitation diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        precip_flows = get_flows(precipitant)

        # Oxalic acid flow (C2O4 = oxalate)
        F_oxalate = precip_flows.get("C2O4", precip_flows.get("oxalic_acid", 0.0))

        filtrate_flows = {"H2O": feed_flows.get("H2O", 1.0)}
        solid_flows = {}
        precipitation_data = {}

        # Total REE for stoichiometry check
        total_ree = sum(feed_flows.get(e, 0.0) for e in p.elements)

        # Stoichiometry: 2 REE + 3 C2O4 → REE2(C2O4)3
        # Required oxalate = 1.5 × REE (mol basis)
        required_oxalate = 1.5 * total_ree
        actual_excess = F_oxalate / (required_oxalate + 1e-10)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))

            # Precipitation efficiency based on Ksp and excess
            # Higher pKsp = lower solubility = better precipitation
            pKsp = PKsp_OXALATE[elem]

            # Simple model: conversion increases with excess and pKsp
            # At stoichiometric (excess=1), conversion ~ 0.99 for typical pKsp
            base_conversion = 1 - jnp.power(10.0, -pKsp/10)
            conversion = jnp.minimum(
                base_conversion * jnp.sqrt(actual_excess),
                p.target_conversion
            )
            conversion = jnp.clip(conversion, 0.0, 0.9999)

            F_precipitated = F_in * conversion
            F_filtrate = F_in * (1 - conversion)

            filtrate_flows[elem] = jnp.maximum(F_filtrate, 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "conversion": conversion,
                "precipitated_mol_s": F_precipitated,
            }

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        # Calculate solid composition
        total_solid = sum(float(solid_flows[e]) for e in p.elements)
        solid_composition = {
            e: float(solid_flows[e]) / (total_solid + 1e-10)
            for e in p.elements
        }

        info = {
            "precipitant": "oxalate",
            "excess_ratio": float(actual_excess),
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "solid_composition": solid_composition,
            "product_formula": "REE2(C2O4)3",
        }

        return filtrate, solid_flows, info


# =============================================================================
# Carbonate Precipitation
# =============================================================================

class CarbonatePrecipitator:
    """Carbonate precipitation for REE recovery.

    Reaction: 2REE³⁺ + 3CO₃²⁻ → REE₂(CO₃)₃↓

    Carbonate precipitation is often used for:
    - Group precipitation from leach solutions
    - pH adjustment during processing

    Example:
        >>> params = PrecipitatorParams(
        ...     elements=("La", "Ce", "Nd"),
        ...     precipitant_excess=1.2,
        ... )
        >>> precip = CarbonatePrecipitator(params)
        >>> filtrate, solid, info = precip(feed, na2co3_solution)
    """

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, dict, dict]:
        """Perform carbonate precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Carbonate solution (Na2CO3 or (NH4)2CO3)
            T: Temperature (K)

        Returns:
            filtrate: Aqueous filtrate
            solid: Precipitated REE (mol/s)
            info: Precipitation diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        precip_flows = get_flows(precipitant)

        F_carbonate = precip_flows.get("CO3", precip_flows.get("carbonate", 0.0))

        filtrate_flows = {"H2O": feed_flows.get("H2O", 1.0)}
        solid_flows = {}
        precipitation_data = {}

        total_ree = sum(feed_flows.get(e, 0.0) for e in p.elements)
        required_carbonate = 1.5 * total_ree
        actual_excess = F_carbonate / (required_carbonate + 1e-10)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))

            pKsp = PKsp_CARBONATE[elem]
            base_conversion = 1 - jnp.power(10.0, -pKsp/12)
            conversion = jnp.minimum(
                base_conversion * jnp.sqrt(actual_excess),
                p.target_conversion
            )
            conversion = jnp.clip(conversion, 0.0, 0.9999)

            F_precipitated = F_in * conversion
            F_filtrate = F_in * (1 - conversion)

            filtrate_flows[elem] = jnp.maximum(F_filtrate, 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "conversion": conversion,
            }

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        total_solid = sum(float(solid_flows[e]) for e in p.elements)

        info = {
            "precipitant": "carbonate",
            "excess_ratio": float(actual_excess),
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "product_formula": "REE2(CO3)3",
        }

        return filtrate, solid_flows, info


# =============================================================================
# Hydroxide Precipitation
# =============================================================================

class HydroxidePrecipitator:
    """Hydroxide precipitation for REE recovery.

    Reaction: REE³⁺ + 3OH⁻ → REE(OH)₃↓

    Hydroxide precipitation can be selective based on pH:
    - Heavy REE precipitate at lower pH than light REE
    - Can achieve group separations by pH control

    Example:
        >>> params = PrecipitatorParams(
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ... )
        >>> precip = HydroxidePrecipitator(params)
        >>> filtrate, solid, info = precip(feed, naoh_solution, pH=8.5)
    """

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        pH: Array | float = 9.0,
        T: Array | float | None = None,
    ) -> tuple[Stream, dict, dict]:
        """Perform hydroxide precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Base solution (NaOH or NH4OH)
            pH: Target pH for precipitation
            T: Temperature (K)

        Returns:
            filtrate: Aqueous filtrate
            solid: Precipitated REE (mol/s)
            info: Precipitation diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)
        pH = jnp.asarray(pH)

        feed_flows = get_flows(feed)

        filtrate_flows = {"H2O": feed_flows.get("H2O", 1.0)}
        solid_flows = {}
        precipitation_data = {}

        # [OH-] from pH
        pOH = 14 - pH
        OH_conc = jnp.power(10.0, -pOH)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))

            pKsp = PKsp_HYDROXIDE[elem]
            Ksp = jnp.power(10.0, -pKsp)

            # Solubility: [REE³⁺] = Ksp / [OH⁻]³
            # If [REE³⁺] in solution < equilibrium, no precipitation
            # Higher pH (more OH-) = lower solubility = more precipitation

            # Saturation concentration
            c_sat = Ksp / jnp.power(OH_conc, 3)

            # Assume feed concentration (rough estimate)
            c_feed = F_in / (feed_flows.get("H2O", 1.0) + 1e-10)

            # Supersaturation ratio
            S = c_feed / (c_sat + 1e-20)

            # Conversion based on supersaturation
            # If S > 1, precipitation occurs
            conversion = jnp.where(
                S > 1,
                jnp.minimum(1 - 1/S, p.target_conversion),
                0.0
            )
            conversion = jnp.clip(conversion, 0.0, 0.9999)

            F_precipitated = F_in * conversion
            F_filtrate = F_in * (1 - conversion)

            filtrate_flows[elem] = jnp.maximum(F_filtrate, 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "supersaturation": float(S),
                "conversion": float(conversion),
                "precipitation_pH": float(14 + jnp.log10(jnp.power(Ksp/c_feed, 1/3) + 1e-20)),
            }

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        total_solid = sum(float(solid_flows[e]) for e in p.elements)

        info = {
            "precipitant": "hydroxide",
            "pH": float(pH),
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "product_formula": "REE(OH)3",
        }

        return filtrate, solid_flows, info

    def selective_precipitation_pH(
        self,
        target_element: str,
        reject_element: str,
        feed_conc: float = 0.01,  # M
    ) -> tuple[float, float]:
        """Find pH range for selective precipitation.

        Args:
            target_element: Element to precipitate
            reject_element: Element to keep in solution
            feed_conc: Feed concentration (M)

        Returns:
            Tuple of (min_pH, max_pH) for selectivity
        """
        pKsp_target = PKsp_HYDROXIDE[target_element]
        pKsp_reject = PKsp_HYDROXIDE[reject_element]

        # pH where target starts precipitating
        # [REE] = Ksp / [OH-]³
        # [OH-] = (Ksp / [REE])^(1/3)
        # pOH = -log10([OH-])
        # pH = 14 - pOH

        Ksp_target = 10**(-pKsp_target)
        Ksp_reject = 10**(-pKsp_reject)

        OH_target = (Ksp_target / feed_conc) ** (1/3)
        OH_reject = (Ksp_reject / feed_conc) ** (1/3)

        pH_target = 14 + jnp.log10(OH_target)
        pH_reject = 14 + jnp.log10(OH_reject)

        return float(pH_target), float(pH_reject)


# =============================================================================
# Convenience Functions
# =============================================================================

def oxalate_to_oxide_mass(
    oxalate_mol: float,
    element: str,
) -> float:
    """Calculate oxide mass from oxalate precipitation.

    REE₂(C₂O₄)₃ → REE₂O₃ (calcination)

    Args:
        oxalate_mol: Moles of REE in oxalate
        element: REE symbol

    Returns:
        Mass of oxide produced (g)
    """
    db = get_ree_database()
    elem_data = db.get(element)
    oxide_mw = elem_data.oxide_mw

    # 2 REE per formula unit of oxalate and oxide
    oxide_mol = oxalate_mol / 2
    return oxide_mol * oxide_mw


def precipitation_reagent_cost(
    ree_mol: float,
    precipitant: str,
    excess: float = 1.5,
) -> float:
    """Calculate precipitant reagent cost.

    Args:
        ree_mol: Moles of REE to precipitate
        precipitant: Type (oxalate, carbonate, hydroxide)
        excess: Molar excess ratio

    Returns:
        Reagent cost (USD)
    """
    # Approximate costs (USD/kg)
    costs = {
        "oxalate": 2.0,  # Oxalic acid
        "carbonate": 0.3,  # Na2CO3
        "hydroxide": 0.5,  # NaOH
    }

    # Molecular weights
    mw = {
        "oxalate": 90.03,  # H2C2O4
        "carbonate": 105.99,  # Na2CO3
        "hydroxide": 40.0,  # NaOH
    }

    # Stoichiometry (mol reagent per mol REE)
    stoich = {
        "oxalate": 1.5,
        "carbonate": 1.5,
        "hydroxide": 3.0,
    }

    reagent_mol = ree_mol * stoich[precipitant] * excess
    reagent_kg = reagent_mol * mw[precipitant] / 1000

    return reagent_kg * costs[precipitant]
