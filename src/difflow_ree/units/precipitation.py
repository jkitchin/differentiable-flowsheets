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

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide, safe_log
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.database import get_ree_database
#: Units for every :class:`PrecipitatorParams` field, shared by the oxalate,
#: carbonate and hydroxide routes -- they differ in chemistry, not in what
#: their parameters mean.
_PRECIPITATOR_UNITS = {
    "precipitant_excess": "-",
    "temperature": "K",
    "residence_time": "s",
    "target_conversion": "-",
    "coprecipitation_factor": "-",
}


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
# Shared accounting (2026 conservation and operating-point audits)
# =============================================================================
#
# Three defects shared the precipitators. (1) Conversion scaled with
# sqrt(excess) and nothing capped it by the reagent fed, so 0.5x oxalate
# precipitated 70.5 % of the REE, which needs 0.127 mol oxalate against 0.090
# supplied. (2) The filtrate was rebuilt as {H2O, elements}: every other feed
# species (HCl, HNO3, Fe, ...), the excess reagent and the precipitant's water
# left through no outlet. (3) The hydroxide route never read its precipitant
# at all. Each is fixed here once, for all three routes.

#: Base species a hydroxide precipitant may carry: name -> (cation left in
#: solution when its OH- is consumed, or None).
HYDROXIDE_BASES = {"NaOH": "Na", "KOH": "K", "NH4OH": "NH4", "OH": None}

#: Strong acids neutralised by base before any hydroxide precipitates:
#: name -> (anion left in solution, protons per formula unit).
STRONG_ACIDS = {"HCl": ("Cl", 1), "HNO3": ("NO3", 1), "H2SO4": ("SO4", 2)}

#: Litres per mole of water, to put a pH on a concentration scale.
_L_PER_MOL_WATER = 0.018015


def _require_pKsp(table: dict, elements, route: str, table_name: str) -> None:
    """Raise if ``table`` lacks a solubility product for any element.

    The tables cover the ten elements the extractant records do; Ho, Er, Tm,
    Yb and Lu used to fail later with a bare ``KeyError``. No values are
    invented for them: measured constants have to be added to the table.
    """
    missing = [e for e in elements if e not in table]
    if missing:
        raise ValueError(
            f"No {route} solubility product (pKsp) for {missing}: "
            f"{table_name} covers {sorted(table)}. Add measured values to "
            f"difflow_ree.units.precipitation.{table_name} (with a source) "
            f"before precipitating these elements."
        )


def _non_ree(flows: dict, elements, exclude=()) -> dict:
    """Every species in ``flows`` that is not an element or in ``exclude``."""
    return {k: jnp.asarray(v) for k, v in flows.items()
            if k not in elements and k not in exclude}


def _add(target: dict, extra: dict) -> dict:
    """Sum ``extra`` into ``target`` species by species."""
    for k, v in extra.items():
        target[k] = target[k] + v if k in target else v
    return target


def _reagent_scale(demand, supply):
    """Fraction of the unconstrained precipitation the reagent allows.

    1 when ``supply`` covers ``demand``, ``supply / demand`` when it does not,
    so no route precipitates more REE than its reagent can bind.
    """
    return jnp.minimum(1.0, safe_divide(jnp.maximum(supply, 0.0),
                                        jnp.maximum(demand, 1e-300)))


# =============================================================================
# Precipitator Parameters
# =============================================================================

@dataclass(repr=False)
class PrecipitatorParams(ParamsMixin):
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
    # Co-precipitation / common-ion coupling (#109). When > 0, an element's
    # conversion is boosted toward 1 in proportion to the bulk precipitation
    # extent of the other REE, capturing common-ion depression of solubility
    # and solid-solution co-precipitation (trace REE recovered above their
    # individual-Ksp prediction). 0 disables it (independent precipitation,
    # backward compatible). See Byrne & Kim, Geochim. Cosmochim. Acta 54, 2645
    # (1990) for REE co-precipitation.
    coprecipitation_factor: float = 0.0


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

    symbol = "Oxalate Precip."
    equations = [
        r"2\,\mathrm{RE}^{3+} + 3\,\mathrm{C}_2\mathrm{O}_4^{2-} \rightarrow \mathrm{RE}_2(\mathrm{C}_2\mathrm{O}_4)_3\,\downarrow",
        r"\mathrm{RE}_2(\mathrm{C}_2\mathrm{O}_4)_3 \xrightarrow{\Delta} \mathrm{RE}_2\mathrm{O}_3 + 3\,\mathrm{CO} + 3\,\mathrm{CO}_2",
        r"K_\mathrm{sp} = [\mathrm{RE}^{3+}]^2\,[\mathrm{C}_2\mathrm{O}_4^{2-}]^3",
    ]
    assumptions = [
        "Stoichiometric precipitation with user-controlled excess precipitant.",
        "High precipitation yield typical of oxalate (>99%) captured via efficiency parameter.",
        "Solid is filtered cleanly; no co-precipitation tracking.",
    ]
    references = [
        "Moldoveanu, G.A., Papangelakis, V.G. Hydrometallurgy, 117-118, 71 (2012).",
        "Habashi, F. Handbook of Extractive Metallurgy, Vol. 3, Wiley-VCH, 1997.",
    ]
    parameter_symbols = {"precipitant_excess": r"\phi_\mathrm{exc}", "temperature": "T"}
    parameter_units = _PRECIPITATOR_UNITS

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        _require_pKsp(PKsp_OXALATE, params.elements, "oxalate", "PKsp_OXALATE")
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        T: Array | float | None = None,
        feed_pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform oxalate precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Oxalic acid solution
            T: Temperature (K)
            feed_pH: Feed pH. When provided, the filtrate pH after
                precipitation is reported (#116): oxalic-acid precipitation
                releases 3 H+ per mole REE (6 H+ per RE2(C2O4)3), acidifying
                the filtrate.

        Returns:
            filtrate: Aqueous filtrate (depleted in REE). Carries every
                non-REE species of both inlets (the precipitant's water and
                the unreacted oxalate included).
            solid: Solid product stream: the precipitated REE and the
                oxalate bound with them (1.5 per REE, under the reagent's
                own species name).
            info: Precipitation diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        precip_flows = get_flows(precipitant)

        # Oxalic acid flow (C2O4 = oxalate)
        reagent = "C2O4" if "C2O4" in precip_flows else (
            "oxalic_acid" if "oxalic_acid" in precip_flows else "C2O4")
        F_oxalate = jnp.asarray(precip_flows.get(reagent, 0.0))

        filtrate_flows = {}
        solid_flows = {}
        precipitation_data = {}

        # Total REE for stoichiometry check
        total_ree = sum(feed_flows.get(e, 0.0) for e in p.elements)

        # Stoichiometry: 2 REE + 3 C2O4 → REE2(C2O4)3
        # Required oxalate = 1.5 × REE (mol basis)
        required_oxalate = 1.5 * total_ree
        actual_excess = safe_divide(F_oxalate, required_oxalate)

        # First pass: independent (individual-Ksp) conversions per element.
        base_conversions = {}
        for elem in p.elements:
            pKsp = PKsp_OXALATE[elem]
            base_conversion = 1 - jnp.power(10.0, -pKsp/10)
            conversion = jnp.minimum(
                base_conversion * jnp.sqrt(actual_excess),
                p.target_conversion
            )
            base_conversions[elem] = jnp.clip(conversion, 0.0, 0.9999)

        # Bulk precipitation extent (flow-weighted mean independent conversion),
        # used to drive co-precipitation of the remaining REE (#109).
        bulk_extent = safe_divide(
            sum(jnp.asarray(feed_flows.get(e, 0.0)) * base_conversions[e]
                for e in p.elements),
            jnp.maximum(total_ree, 1e-30),
        )

        conversions = {}
        for elem in p.elements:
            conversion = base_conversions[elem]
            # Co-precipitation / common-ion boost toward complete capture.
            conversion = conversion + p.coprecipitation_factor * (1.0 - conversion) * bulk_extent
            conversions[elem] = jnp.clip(conversion, 0.0, 0.9999)

        # No more REE than the oxalate fed can bind (1.5 C2O4 per REE).
        demand = 1.5 * sum(jnp.asarray(feed_flows.get(e, 0.0)) * conversions[e]
                           for e in p.elements)
        scale = _reagent_scale(demand, F_oxalate)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))
            pKsp = PKsp_OXALATE[elem]
            conversion = conversions[elem] * scale

            F_precipitated = F_in * conversion
            F_filtrate = F_in * (1 - conversion)

            filtrate_flows[elem] = jnp.maximum(F_filtrate, 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "conversion": conversion,
                "precipitated_mol_s": F_precipitated,
            }

        # Everything else passes through; the reagent splits into what the
        # solid binds and what is left in solution.
        total_solid = sum(solid_flows[e] for e in p.elements)
        bound = 1.5 * total_solid
        _add(filtrate_flows, _non_ree(feed_flows, p.elements))
        _add(filtrate_flows, _non_ree(precip_flows, p.elements, exclude=(reagent,)))
        _add(filtrate_flows, {reagent: jnp.maximum(F_oxalate - bound, 0.0)})
        solid_flows[reagent] = jnp.minimum(bound, F_oxalate)

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        # Create solid stream (precipitate at same T, P as filtrate)
        solid = make_stream(solid_flows, T, P)

        # Calculate solid composition
        solid_composition = {
            e: safe_divide(solid_flows[e], total_solid)
            for e in p.elements
        }

        info = {
            "precipitant": "oxalate",
            "excess_ratio": actual_excess,
            # Fraction of the unconstrained precipitation the oxalate fed
            # allows; below 1 the route is reagent-limited.
            "reagent_scale": scale,
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "solid_composition": solid_composition,
            "product_formula": "REE2(C2O4)3",
            "coprecipitation_factor": jnp.asarray(p.coprecipitation_factor),
        }

        # Filtrate pH after precipitation (#116). Oxalic-acid precipitation
        # releases 3 H+ per mole REE precipitated, acidifying the filtrate.
        if feed_pH is not None:
            V = jnp.asarray(feed_flows.get("H2O", 1.0))
            H_initial = jnp.power(10.0, -jnp.asarray(feed_pH)) * V
            H_released = 3.0 * total_solid
            H_final = H_initial + H_released
            info["h_plus_released"] = H_released
            info["pH_final"] = -jnp.log10(
                jnp.maximum(safe_divide(H_final, V), 1e-30)
            )

        return filtrate, solid, info


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

    symbol = "Carbonate Precip."
    equations = [
        r"2\,\mathrm{RE}^{3+} + 3\,\mathrm{CO}_3^{2-} \rightarrow \mathrm{RE}_2(\mathrm{CO}_3)_3\,\downarrow",
        r"\mathrm{RE}_2(\mathrm{CO}_3)_3 \xrightarrow{\Delta} \mathrm{RE}_2\mathrm{O}_3 + 3\,\mathrm{CO}_2",
    ]
    assumptions = [
        "Stoichiometric reaction with a sodium/ammonium carbonate solution.",
        "Bulk group-precipitation; composition-selective separation is handled upstream.",
    ]
    references = ["Habashi, F. Handbook of Extractive Metallurgy, Vol. 3, Wiley-VCH, 1997."]
    parameter_symbols = {"precipitant_excess": r"\phi_\mathrm{exc}"}
    parameter_units = _PRECIPITATOR_UNITS

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        _require_pKsp(PKsp_CARBONATE, params.elements, "carbonate", "PKsp_CARBONATE")
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform carbonate precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Carbonate solution (Na2CO3 or (NH4)2CO3)
            T: Temperature (K)

        Returns:
            filtrate: Aqueous filtrate. Carries every non-REE species of both
                inlets (the precipitant's water and the unreacted carbonate
                included).
            solid: Solid product stream: the precipitated REE and the
                carbonate bound with them (1.5 per REE, under the reagent's
                own species name).
            info: Precipitation diagnostics
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        precip_flows = get_flows(precipitant)

        reagent = "CO3" if "CO3" in precip_flows else (
            "carbonate" if "carbonate" in precip_flows else "CO3")
        F_carbonate = jnp.asarray(precip_flows.get(reagent, 0.0))

        filtrate_flows = {}
        solid_flows = {}
        precipitation_data = {}

        total_ree = sum(feed_flows.get(e, 0.0) for e in p.elements)
        required_carbonate = 1.5 * total_ree
        actual_excess = safe_divide(F_carbonate, required_carbonate)

        conversions = {}
        for elem in p.elements:
            pKsp = PKsp_CARBONATE[elem]
            base_conversion = 1 - jnp.power(10.0, -pKsp/12)
            conversion = jnp.minimum(
                base_conversion * jnp.sqrt(actual_excess),
                p.target_conversion
            )
            conversions[elem] = jnp.clip(conversion, 0.0, 0.9999)
        # No more REE than the carbonate fed can bind (1.5 CO3 per REE).
        demand = 1.5 * sum(jnp.asarray(feed_flows.get(e, 0.0)) * conversions[e]
                           for e in p.elements)
        scale = _reagent_scale(demand, F_carbonate)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))
            pKsp = PKsp_CARBONATE[elem]
            conversion = conversions[elem] * scale

            F_precipitated = F_in * conversion
            F_filtrate = F_in * (1 - conversion)

            filtrate_flows[elem] = jnp.maximum(F_filtrate, 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "conversion": conversion,
            }

        total_solid = sum(solid_flows[e] for e in p.elements)
        bound = 1.5 * total_solid
        _add(filtrate_flows, _non_ree(feed_flows, p.elements))
        _add(filtrate_flows, _non_ree(precip_flows, p.elements, exclude=(reagent,)))
        _add(filtrate_flows, {reagent: jnp.maximum(F_carbonate - bound, 0.0)})
        solid_flows[reagent] = jnp.minimum(bound, F_carbonate)

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        # Create solid stream (precipitate at same T, P as filtrate)
        solid = make_stream(solid_flows, T, P)

        info = {
            "precipitant": "carbonate",
            "excess_ratio": actual_excess,
            "reagent_scale": scale,
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "product_formula": "REE2(CO3)3",
        }

        return filtrate, solid, info


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

    symbol = "Hydroxide Precip."
    equations = [
        r"\mathrm{RE}^{3+} + 3\,\mathrm{OH}^- \rightarrow \mathrm{RE}(\mathrm{OH})_3\,\downarrow",
        r"K_\mathrm{sp} = [\mathrm{RE}^{3+}]\,[\mathrm{OH}^-]^3",
        r"\mathrm{pH}_\mathrm{ppt} = 14 - \tfrac{1}{3}\log_{10}\!\left(K_\mathrm{sp}/[\mathrm{RE}^{3+}]\right)",
    ]
    assumptions = [
        "Selectivity governed by per-element solubility products at the chosen pH.",
        "Base (NaOH) addition as the precipitant with user-controlled excess.",
    ]
    references = [
        "Baes, C.F., Mesmer, R.E. The Hydrolysis of Cations, Krieger, 1986.",
        "Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014).",
    ]
    parameter_symbols = {"precipitant_excess": r"\phi_\mathrm{exc}"}
    parameter_units = _PRECIPITATOR_UNITS

    def __init__(self, params: PrecipitatorParams):
        """Initialize precipitator.

        Args:
            params: Precipitator parameters
        """
        _require_pKsp(PKsp_HYDROXIDE, params.elements, "hydroxide", "PKsp_HYDROXIDE")
        self.params = params
        self._db = get_ree_database()

    def __call__(
        self,
        feed: Stream,
        precipitant: Stream,
        pH: Array | float = 9.0,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform hydroxide precipitation.

        Args:
            feed: Aqueous REE solution
            precipitant: Base solution. Its hydroxide is the species in
                :data:`HYDROXIDE_BASES` (NaOH, KOH, NH4OH, or bare OH).
            pH: Setpoint pH the base is dosed to
            T: Temperature (K)

        The setpoint pH fixes how much of each REE is above its hydroxide
        solubility. The base supplied then has to pay for it: it first
        neutralises the strong acid in either inlet (:data:`STRONG_ACIDS`),
        then binds 3 OH- per REE precipitated, and when it runs short the
        precipitation is scaled down to what it can pay for. The base used
        to be ignored, so 0 and 0.36 mol NaOH both precipitated 99.5 %
        (2026 operating-point audit, R10).

        Returns:
            filtrate: Aqueous filtrate. Carries every non-REE species of both
                inlets: unconsumed base and acid, the cation of consumed base
                (Na, K, NH4), the anion of neutralised acid (Cl, NO3, SO4),
                and the water, including what neutralisation makes.
            solid: Solid product stream: the precipitated REE and the OH
                bound with them (3 per REE).
            info: Precipitation diagnostics. ``pH_final`` is the setpoint
                when the base reaches it, and otherwise the pH of the net
                acid/base balance on a molar (mol/L) basis.
        """
        p = self.params
        T = T if T is not None else p.temperature
        T = jnp.asarray(T)
        pH = jnp.asarray(pH)

        feed_flows = get_flows(feed)
        precip_flows = get_flows(precipitant)
        zero = jnp.asarray(0.0)

        filtrate_flows = {}
        solid_flows = {}
        precipitation_data = {}

        # [OH-] from pH with temperature-dependent pKw
        # pKw ~ 14.0 at 25C, varies with T (Harned & Hamer correlation)
        T_C = T - 273.15
        pKw = 14.0 - 0.03 * (T_C - 25.0) / 25.0  # pKw decreases with T (more ionization)
        pKw = jnp.clip(pKw, 12.0, 15.0)  # Guard for extreme temperatures
        pOH = pKw - pH
        OH_conc = jnp.power(10.0, -pOH)

        conversions = {}
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
            c_feed = safe_divide(F_in, feed_flows.get("H2O", 1.0))

            # Supersaturation ratio
            S = safe_divide(c_feed, c_sat)

            # Conversion based on supersaturation
            # If S > 1, precipitation occurs
            conversion = jnp.where(
                S > 1,
                jnp.minimum(1 - 1/S, p.target_conversion),
                0.0
            )
            conversions[elem] = jnp.clip(conversion, 0.0, 0.9999)

            precipitation_data[elem] = {
                "pKsp": pKsp,
                "supersaturation": S,
                "precipitation_pH": 14 + safe_log(jnp.power(safe_divide(Ksp, c_feed), 1/3)) / jnp.log(10.0),
            }

        # Base supplied and acid it must neutralise first, from both inlets.
        both = _add(dict(_non_ree(feed_flows, p.elements)),
                    _non_ree(precip_flows, p.elements))
        base = {k: both[k] for k in HYDROXIDE_BASES if k in both}
        acid = {k: both[k] for k in STRONG_ACIDS if k in both}
        OH_supply = sum(base.values(), zero)
        H_acid = sum((STRONG_ACIDS[k][1] * v for k, v in acid.items()), zero)
        H_neutralised = jnp.minimum(H_acid, OH_supply)

        # No more REE than the base left after neutralisation can pay for.
        demand = 3.0 * sum(jnp.asarray(feed_flows.get(e, 0.0)) * conversions[e]
                           for e in p.elements)
        scale = _reagent_scale(demand, OH_supply - H_neutralised)

        for elem in p.elements:
            F_in = jnp.asarray(feed_flows.get(elem, 0.0))
            conversion = conversions[elem] * scale
            F_precipitated = F_in * conversion
            filtrate_flows[elem] = jnp.maximum(F_in * (1 - conversion), 0.0)
            solid_flows[elem] = jnp.maximum(F_precipitated, 0.0)
            precipitation_data[elem]["conversion"] = conversion

        total_solid = sum(solid_flows[e] for e in p.elements)
        OH_consumed = 3.0 * total_solid

        # Species accounting. Consumed base is drawn from each base species in
        # proportion; its cation stays in solution. Neutralised acid is drawn
        # the same way; its anion stays and its protons become water.
        others = {k: v for k, v in both.items() if k not in base and k not in acid}
        _add(filtrate_flows, others)
        used = OH_consumed + H_neutralised
        for k, v in base.items():
            take = v * safe_divide(used, jnp.maximum(OH_supply, 1e-300))
            _add(filtrate_flows, {k: v - take})
            if HYDROXIDE_BASES[k] is not None:
                _add(filtrate_flows, {HYDROXIDE_BASES[k]: take})
        for k, v in acid.items():
            take = v * safe_divide(H_neutralised, jnp.maximum(H_acid, 1e-300))
            anion, n_H = STRONG_ACIDS[k]
            _add(filtrate_flows, {k: v - take, anion: take})
        _add(filtrate_flows, {"H2O": H_neutralised})
        solid_flows["OH"] = OH_consumed

        P = feed["P"]
        filtrate = make_stream(filtrate_flows, T, P)

        # Create solid stream (precipitate at same T, P as filtrate)
        solid = make_stream(solid_flows, T, P)

        # Filtrate pH after precipitation (#116). Each RE(OH)3 formed consumes
        # 3 OH-. The old balance started from the free OH- at the setpoint
        # alone (1e-5 M at pH 9), so any real precipitation drove it to zero
        # and reported pH -16 (audit R10). It now balances the base supplied
        # against the acid and the OH- bound: when the excess reaches the
        # setpoint the setpoint holds; otherwise the net excess (base or
        # acid) sets the pH, through [OH-] - [H+] = n with [OH-][H+] = Kw.
        V_L = jnp.maximum(
            _L_PER_MOL_WATER * (jnp.asarray(feed_flows.get("H2O", 0.0))
                                + jnp.asarray(precip_flows.get("H2O", 0.0))),
            1e-300)
        n = (OH_supply - H_acid - OH_consumed) / V_L
        Kw = jnp.power(10.0, -pKw)
        # Root of [OH-]^2 - n [OH-] - Kw = 0, in the form that does not
        # cancel: for an acid excess (n < 0) n + sqrt(n^2 + 4 Kw) loses ~4
        # digits, so use the conjugate 2 Kw / (sqrt(n^2 + 4 Kw) - n).
        root = jnp.sqrt(n * n + 4.0 * Kw)
        OH_eq = jnp.where(n >= 0, 0.5 * (n + root),
                          2.0 * Kw / jnp.maximum(root - n, 1e-300))
        pH_balance = pKw + jnp.log10(jnp.maximum(OH_eq, 1e-300))
        pH_final = jnp.minimum(pH, pH_balance)

        info = {
            "precipitant": "hydroxide",
            "pH": pH,
            "precipitation_data": precipitation_data,
            "total_precipitated": total_solid,
            "product_formula": "REE(OH)3",
            # Post-precipitation filtrate chemistry (#116)
            "OH_consumed": OH_consumed,
            "OH_supplied": OH_supply,
            "acid_neutralised": H_neutralised,
            "reagent_scale": scale,
            "pH_final": pH_final,
        }

        return filtrate, solid, info

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
