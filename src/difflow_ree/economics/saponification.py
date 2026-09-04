"""Reagent and effluent accounting for saponified circuits (#197).

WHY THIS IS ECONOMICS *AND* ENVIRONMENT
---------------------------------------
The counter-ion balance that makes a saponified cascade model correct is the
same equation that predicts the raffinate load, so once
:mod:`difflow_ree.equilibrium.saponification` exists, kilograms of base per
kilogram of rare-earth oxide costs no extra machinery. Three equivalents of
base per mole of rare earth moved -- read off the tableau by
:attr:`~difflow_ree.equilibrium.network.ReactionNetwork.base_equivalents_per_mole_ree`,
not asserted here -- sets both the reagent bill and the effluent.

Which base is chosen decides *what kind* of effluent a plant has, and that is
the industry's defining environmental choice rather than a detail:

- **Ammonia** saponification produces the ammonium-nitrogen effluent that is
  the signature pollution problem of Chinese rare-earth separation. It is the
  cheapest base per equivalent and the most expensive to treat.
- **Sodium** trades that for a saline raffinate: no nitrogen, but sodium
  chloride that has to go somewhere.
- **Magnesium** gives two equivalents per mole and a less soluble salt, at a
  higher reagent cost.

:func:`compare_counter_ions` puts the three side by side on one basis, which
is the comparison the whole feature exists to make computable.

WHAT IS AND IS NOT A NUMBER FROM THIS FILE
------------------------------------------
Molar masses and stoichiometry are arithmetic. Prices are *illustrative order
of magnitude*, taken from the existing
:class:`~difflow_ree.economics.costs.ReagentCosts` defaults, which carry no
source either; pass your own. Nothing here estimates a treatment cost for the
effluent -- the loads are reported and the valuation is left to the caller,
because an ammonium-nitrogen discharge limit is a regulatory fact and not a
correlation.

References:
    Liao, C. et al. Clean separation technologies of rare earth resources in
    China. *J. Rare Earths* 31 (2013) 331-336.
    doi:10.1016/S1002-0721(12)60281-6 -- consulted for the effluent
    consequences of the ammonium saponification route; not the source of any
    number here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jax import Array

from difflow_ree.database import get_ree_database


# =============================================================================
# The reagents
# =============================================================================

@dataclass(frozen=True)
class BaseReagent:
    """A base used to saponify an acidic extractant (#197).

    Attributes:
        name: Reagent key, e.g. ``"NaOH"``.
        formula: Chemical formula as dosed.
        molar_mass: g/mol of the anhydrous reagent. Ammonia is accounted as
            ``NH3``, not as ammonia solution: a plant buys solution but the
            stoichiometry, the nitrogen load and the comparison between bases
            all run on the anhydrous basis, and quoting a 25% solution here
            would silently multiply the answer by four.
        equivalents_per_mole: Protons neutralized per mole of reagent -- 1 for
            NaOH, 2 for Mg(OH)2 and Na2CO3.
        counter_ion: Cation left behind in the circuit.
        counter_ion_per_mole: Moles of that cation per mole of reagent.
        nitrogen_per_mole: Moles of nitrogen per mole of reagent; non-zero
            only for the ammonium route, and it is the ammonium-nitrogen
            effluent load.
    """

    name: str
    formula: str
    molar_mass: float
    equivalents_per_mole: float
    counter_ion: str
    counter_ion_per_mole: float
    nitrogen_per_mole: float = 0.0

    @property
    def mass_per_equivalent(self) -> float:
        """Grams of reagent per equivalent of base."""
        return self.molar_mass / self.equivalents_per_mole


#: Bases the saponification accounting knows. Molar masses are IUPAC 2021
#: standard atomic weights summed over the formula -- arithmetic, not data.
BASE_REAGENTS: dict[str, BaseReagent] = {
    "NaOH": BaseReagent("NaOH", "NaOH", 39.997, 1.0, "Na", 1.0),
    "KOH": BaseReagent("KOH", "KOH", 56.105, 1.0, "K", 1.0),
    "NH3": BaseReagent("NH3", "NH3", 17.031, 1.0, "NH4", 1.0, 1.0),
    "NH4OH": BaseReagent("NH4OH", "NH4OH", 35.046, 1.0, "NH4", 1.0, 1.0),
    "Mg(OH)2": BaseReagent("Mg(OH)2", "Mg(OH)2", 58.320, 2.0, "Mg", 1.0),
    "MgO": BaseReagent("MgO", "MgO", 40.304, 2.0, "Mg", 1.0),
    "Na2CO3": BaseReagent("Na2CO3", "Na2CO3", 105.988, 2.0, "Na", 2.0),
}

#: Default reagent for each counter-ion. ``"H"`` means un-neutralized proton
#: exchange, which consumes no base at all.
DEFAULT_BASE_FOR_COUNTER_ION: dict[str, str | None] = {
    "H": None,
    "Na": "NaOH",
    "NH4": "NH3",
    "Mg": "Mg(OH)2",
    "K": "KOH",
}

#: Cation molar masses (g/mol), for the dissolved-salt load.
_CATION_MASS = {"Na": 22.990, "NH4": 18.039, "K": 39.098, "Mg": 24.305,
                "Ca": 40.078}

#: Anion molar masses (g/mol) and charges, for the dissolved-salt load.
_ANION_MASS = {"Cl": (35.453, 1), "NO3": (62.004, 1), "SO4": (96.06, 2)}


def base_for_counter_ion(counter_ion: str) -> BaseReagent:
    """Default base reagent for a counter-ion.

    Args:
        counter_ion: One of ``"Na"``, ``"NH4"``, ``"Mg"``, ``"K"``.

    Returns:
        The :class:`BaseReagent`.

    Raises:
        KeyError: If the counter-ion is unknown or is ``"H"``, which is
            un-neutralized proton exchange and consumes no base.

    Example:
        >>> base_for_counter_ion("NH4").formula
        'NH3'
    """
    name = DEFAULT_BASE_FOR_COUNTER_ION.get(counter_ion, "missing")
    if name is None:
        raise KeyError(
            "counter_ion='H' means the extractant is used un-neutralized, so "
            "there is no base and no reagent duty (#197)."
        )
    if name == "missing":
        raise KeyError(
            f"No default base for counter_ion {counter_ion!r}. Known: "
            f"{sorted(k for k, v in DEFAULT_BASE_FOR_COUNTER_ION.items() if v)}"
            f" (#197)."
        )
    return BASE_REAGENTS[name]


def get_base(base: str | BaseReagent) -> BaseReagent:
    """Resolve a base by name.

    Args:
        base: Reagent key or a :class:`BaseReagent`.

    Returns:
        The :class:`BaseReagent`.

    Raises:
        KeyError: If the name is unknown.
    """
    if isinstance(base, BaseReagent):
        return base
    if base not in BASE_REAGENTS:
        raise KeyError(
            f"Unknown base {base!r}. Known: {sorted(BASE_REAGENTS)} (#197)."
        )
    return BASE_REAGENTS[base]


# =============================================================================
# The rare-earth oxide basis
# =============================================================================

def ree_oxide_mass_flow(
    element_flows: Mapping[str, float | Array],
) -> Array:
    """Rare-earth **oxide** mass flow (kg/s) from element molar flows.

    Rare-earth production is quoted on an oxide (REO) basis, so a reagent
    intensity has to be too, and the conversion is not one oxide formula: the
    database carries ``Ce`` as ``CeO2``, ``Pr`` as ``Pr6O11`` and ``Tb`` as
    ``Tb4O7`` because that is how they are sold and reported. The oxide mass
    per mole of element is therefore ``oxide_mw`` divided by the number of
    metal atoms in the oxide formula, and that count is derived from the
    formula rather than assumed to be two.

    Args:
        element_flows: Molar flow (mol/s) per REE symbol.

    Returns:
        Oxide mass flow in kg/s.

    Example:
        >>> f"{float(ree_oxide_mass_flow({'Nd': 1.0}) * 1000):.3f}"
        '168.240'
    """
    db = get_ree_database()
    total = jnp.asarray(0.0, dtype=jnp.float64)
    for symbol, flow in element_flows.items():
        el = db.get(symbol)
        total = total + jnp.asarray(flow, dtype=jnp.float64) * (
            el.oxide_mw / _metal_atoms(el.oxide_formula, symbol) / 1000.0
        )
    return total


def _metal_atoms(oxide_formula: str, symbol: str) -> float:
    """Metal atoms per oxide formula unit, e.g. 6 for ``Pr6O11``.

    Args:
        oxide_formula: Formula from the element database.
        symbol: Element symbol.

    Returns:
        Number of metal atoms in one formula unit.

    Raises:
        ValueError: If the formula does not start with the element symbol.
    """
    if not oxide_formula.startswith(symbol):
        raise ValueError(
            f"Oxide formula {oxide_formula!r} does not start with the element "
            f"symbol {symbol!r}, so the metal-atom count cannot be read from "
            f"it (#197)."
        )
    rest = oxide_formula[len(symbol):]
    digits = ""
    for ch in rest:
        if ch.isdigit():
            digits += ch
        else:
            break
    return float(digits) if digits else 1.0


# =============================================================================
# The metric
# =============================================================================

def base_per_ree_oxide(
    base_flow: float | Array,
    element_flows: Mapping[str, float | Array],
    base: str | BaseReagent = "NaOH",
) -> Array:
    """Kilograms of base per kilogram of rare-earth oxide (#197).

    The metric the issue asks for, computed from what the flowsheet actually
    consumed rather than from a rule of thumb. Compare it against
    :func:`stoichiometric_base_per_ree_oxide`, which is the ideal three
    equivalents per mole: the gap is the circuit's reagent inefficiency.

    Args:
        base_flow: Base consumption (mol/s of the reagent, not equivalents).
        element_flows: Rare-earth **product** molar flows (mol/s), i.e. what
            was actually recovered.
        base: Reagent name or :class:`BaseReagent`.

    Returns:
        kg of base per kg of REO.

    Example:
        >>> # three equivalents of NaOH per mole of Nd
        >>> f"{float(base_per_ree_oxide(3.0, {'Nd': 1.0}, 'NaOH')):.4f}"
        '0.7132'
    """
    reagent = get_base(base)
    mass = jnp.asarray(base_flow, dtype=jnp.float64) * reagent.molar_mass / 1000.0
    return mass / ree_oxide_mass_flow(element_flows)


def stoichiometric_base_per_ree_oxide(
    element_flows: Mapping[str, float | Array],
    base: str | BaseReagent = "NaOH",
    equivalents_per_mole_ree: float = 3.0,
) -> Array:
    """Ideal kg base per kg REO from stoichiometry alone.

    ``RE3+ + 3 MA_org <-> RE(A)3_org + 3 M+`` moves three equivalents of base
    into the raffinate per mole of trivalent rare earth, so the floor on
    reagent consumption is

    .. math::

        \\frac{m_{\\mathrm{base}}}{m_{\\mathrm{REO}}}
          = \\frac{3\\,M_{\\mathrm{base}} / z_{\\mathrm{base}}}
                  {M_{\\mathrm{REO}}\\,/\\,n_{\\mathrm{metal}}}

    No circuit does better; every real one does worse, because the organic is
    never fully utilized and stripping returns some counter-ion to the
    aqueous phase.

    Args:
        element_flows: Rare-earth molar flows (mol/s); only the *ratios*
            matter.
        base: Reagent name or :class:`BaseReagent`.
        equivalents_per_mole_ree: Base equivalents per mole of rare earth.
            Read it off the reaction network
            (``network.base_equivalents_per_mole_ree``) rather than passing
            the default, so the number stays tied to the tableau.

    Returns:
        kg of base per kg of REO.

    Example:
        >>> f"{float(stoichiometric_base_per_ree_oxide({'Nd': 1.0}, 'NH3')):.4f}"
        '0.3037'
    """
    reagent = get_base(base)
    moles = jnp.asarray(0.0, dtype=jnp.float64)
    for flow in element_flows.values():
        moles = moles + jnp.asarray(flow, dtype=jnp.float64)
    base_moles = moles * equivalents_per_mole_ree / reagent.equivalents_per_mole
    return base_per_ree_oxide(base_moles, element_flows, reagent)


def dissolved_salt_per_ree_oxide(
    base_flow: float | Array,
    element_flows: Mapping[str, float | Array],
    base: str | BaseReagent = "NaOH",
    anion: str = "Cl",
) -> Array:
    """Kilograms of dissolved salt per kilogram of REO.

    Every counter-ion the base brings in leaves in the raffinate paired with
    the medium's anion. For sodium that is the saline raffinate that
    saponification trades the ammonium-nitrogen problem for.

    Args:
        base_flow: Base consumption (mol/s of reagent).
        element_flows: Rare-earth product molar flows (mol/s).
        base: Reagent name or :class:`BaseReagent`.
        anion: Medium anion, one of ``"Cl"``, ``"NO3"``, ``"SO4"``.

    Returns:
        kg of salt per kg of REO.

    Raises:
        KeyError: If the anion or the cation is not tabulated.

    Example:
        >>> f"{float(dissolved_salt_per_ree_oxide(3.0, {'Nd': 1.0})):.4f}"
        '1.0421'
    """
    reagent = get_base(base)
    if anion not in _ANION_MASS:
        raise KeyError(
            f"Unknown anion {anion!r}. Known: {sorted(_ANION_MASS)} (#197)."
        )
    if reagent.counter_ion not in _CATION_MASS:
        raise KeyError(
            f"No molar mass for cation {reagent.counter_ion!r} (#197)."
        )
    anion_mass, anion_charge = _ANION_MASS[anion]
    cation_mass = _CATION_MASS[reagent.counter_ion]
    # One formula unit of salt per cation charge equivalent.
    cation_charge = reagent.equivalents_per_mole / reagent.counter_ion_per_mole
    salt_mass = cation_mass + anion_mass * cation_charge / anion_charge
    cations = (
        jnp.asarray(base_flow, dtype=jnp.float64) * reagent.counter_ion_per_mole
    )
    return cations * salt_mass / 1000.0 / ree_oxide_mass_flow(element_flows)


def nitrogen_per_ree_oxide(
    base_flow: float | Array,
    element_flows: Mapping[str, float | Array],
    base: str | BaseReagent = "NaOH",
) -> Array:
    """Kilograms of effluent nitrogen per kilogram of REO.

    Zero for every base but the ammonium ones. This single number is why the
    counter-ion choice is an environmental decision and not a reagent-price
    decision: ammonia is the cheapest base per equivalent and the origin of
    the ammonium-nitrogen effluent that is the industry's signature pollution
    problem.

    Args:
        base_flow: Base consumption (mol/s of reagent).
        element_flows: Rare-earth product molar flows (mol/s).
        base: Reagent name or :class:`BaseReagent`.

    Returns:
        kg of N per kg of REO.

    Example:
        >>> f"{float(nitrogen_per_ree_oxide(3.0, {'Nd': 1.0}, 'NH3')):.4f}"
        '0.2498'
    """
    reagent = get_base(base)
    n_mass = (
        jnp.asarray(base_flow, dtype=jnp.float64)
        * reagent.nitrogen_per_mole * 14.007 / 1000.0
    )
    return n_mass / ree_oxide_mass_flow(element_flows)


# =============================================================================
# The bundle
# =============================================================================

@dataclass(frozen=True)
class SaponificationDuty:
    """Reagent and effluent duty of a saponified circuit (#197).

    Attributes:
        base: The reagent.
        base_flow: Reagent consumption (mol/s).
        base_mass_flow: Reagent consumption (kg/s).
        equivalents: Base equivalents consumed (eq/s).
        reo_mass_flow: Rare-earth oxide production (kg/s).
        kg_base_per_kg_reo: The headline metric.
        kg_base_per_kg_reo_stoichiometric: The three-equivalent floor.
        reagent_efficiency: Floor divided by actual, 1.0 at the floor.
        kg_nitrogen_per_kg_reo: Ammonium-nitrogen effluent load.
        kg_salt_per_kg_reo: Dissolved-salt load in the raffinate.
        usd_per_kg_reo: Reagent cost per kg of product.
    """

    base: BaseReagent
    base_flow: Array
    base_mass_flow: Array
    equivalents: Array
    reo_mass_flow: Array
    kg_base_per_kg_reo: Array
    kg_base_per_kg_reo_stoichiometric: Array
    reagent_efficiency: Array
    kg_nitrogen_per_kg_reo: Array
    kg_salt_per_kg_reo: Array
    usd_per_kg_reo: Array

    def report(self) -> str:
        """Human-readable statement of the duty.

        Returns:
            Multi-line string.
        """
        return "\n".join([
            f"Saponification duty ({self.base.formula}, "
            f"{self.base.counter_ion} counter-ion)",
            f"  base            : {float(self.base_flow):.6g} mol/s "
            f"({float(self.base_mass_flow):.6g} kg/s, "
            f"{float(self.equivalents):.6g} eq/s)",
            f"  REO production  : {float(self.reo_mass_flow):.6g} kg/s",
            f"  kg base / kg REO: {float(self.kg_base_per_kg_reo):.4f} "
            f"(stoichiometric floor "
            f"{float(self.kg_base_per_kg_reo_stoichiometric):.4f}, "
            f"efficiency {float(self.reagent_efficiency):.3f})",
            f"  kg N   / kg REO : {float(self.kg_nitrogen_per_kg_reo):.4f}",
            f"  kg salt/ kg REO : {float(self.kg_salt_per_kg_reo):.4f}",
            f"  USD    / kg REO : {float(self.usd_per_kg_reo):.4f}",
        ])


#: Illustrative reagent prices, USD per kg. These extend the equally
#: source-free defaults in :class:`~difflow_ree.economics.costs.ReagentCosts`;
#: pass your own.
BASE_PRICES_USD_KG: dict[str, float] = {
    "NaOH": 0.50,
    "KOH": 1.20,
    "NH3": 0.60,
    "NH4OH": 0.35,
    "Mg(OH)2": 0.40,
    "MgO": 0.35,
    "Na2CO3": 0.30,
}


def saponification_duty(
    base_flow: float | Array,
    element_flows: Mapping[str, float | Array],
    base: str | BaseReagent = "NaOH",
    anion: str = "Cl",
    equivalents_per_mole_ree: float = 3.0,
    price_usd_kg: float | None = None,
) -> SaponificationDuty:
    """Bundle the reagent and effluent metrics for one circuit (#197).

    Args:
        base_flow: Base consumption (mol/s of the reagent). Take it from a
            :class:`~difflow_ree.units.saponification.Saponifier`'s
            ``info["base_flow"]`` rather than assuming it.
        element_flows: Rare-earth **product** molar flows (mol/s).
        base: Reagent name or :class:`BaseReagent`.
        anion: Medium anion for the dissolved-salt load.
        equivalents_per_mole_ree: Base equivalents per mole of rare earth;
            read it off the reaction network.
        price_usd_kg: Reagent price; None uses :data:`BASE_PRICES_USD_KG`.

    Returns:
        A :class:`SaponificationDuty`.

    Example:
        >>> d = saponification_duty(3.0, {"Nd": 1.0}, "NaOH")
        >>> f"{float(d.kg_base_per_kg_reo):.4f}"
        '0.7132'
        >>> f"{float(d.reagent_efficiency):.4f}"
        '1.0000'
    """
    reagent = get_base(base)
    reo = ree_oxide_mass_flow(element_flows)
    flow = jnp.asarray(base_flow, dtype=jnp.float64)
    mass = flow * reagent.molar_mass / 1000.0
    actual = mass / reo
    floor = stoichiometric_base_per_ree_oxide(
        element_flows, reagent, equivalents_per_mole_ree
    )
    price = (
        BASE_PRICES_USD_KG.get(reagent.name, 0.5)
        if price_usd_kg is None else price_usd_kg
    )
    return SaponificationDuty(
        base=reagent,
        base_flow=flow,
        base_mass_flow=mass,
        equivalents=flow * reagent.equivalents_per_mole,
        reo_mass_flow=reo,
        kg_base_per_kg_reo=actual,
        kg_base_per_kg_reo_stoichiometric=floor,
        reagent_efficiency=floor / actual,
        kg_nitrogen_per_kg_reo=nitrogen_per_ree_oxide(flow, element_flows, reagent),
        kg_salt_per_kg_reo=dissolved_salt_per_ree_oxide(
            flow, element_flows, reagent, anion
        ),
        usd_per_kg_reo=actual * price,
    )


def compare_counter_ions(
    element_flows: Mapping[str, float | Array],
    counter_ions: tuple[str, ...] = ("Na", "NH4", "Mg"),
    anion: str = "Cl",
    equivalents_per_mole_ree: float = 3.0,
    reagent_efficiency: float = 1.0,
) -> dict[str, SaponificationDuty]:
    """Put the saponification routes side by side on one basis (#197).

    Ammonia trades a saline raffinate for an ammonium-nitrogen effluent;
    magnesium trades reagent cost for two equivalents per mole and a less
    soluble salt. Before this feature none of that was computable, because
    there was no counter-ion in the extraction path at all.

    Args:
        element_flows: Rare-earth product molar flows (mol/s).
        counter_ions: Counter-ions to compare.
        anion: Medium anion.
        equivalents_per_mole_ree: Base equivalents per mole of rare earth.
        reagent_efficiency: Fraction of the stoichiometric floor a real
            circuit achieves, 1.0 being the ideal. Applied to every route
            equally, so the comparison is not distorted by it.

    Returns:
        Mapping of counter-ion to :class:`SaponificationDuty`.

    Example:
        >>> d = compare_counter_ions({"Nd": 1.0})
        >>> f"{float(d['NH4'].kg_nitrogen_per_kg_reo):.4f}"
        '0.2498'
        >>> f"{float(d['Na'].kg_nitrogen_per_kg_reo):.4f}"
        '0.0000'
    """
    if not 0.0 < reagent_efficiency <= 1.0:
        raise ValueError(
            f"reagent_efficiency is a fraction of the stoichiometric floor "
            f"and must lie in (0, 1], got {reagent_efficiency} (#197)."
        )
    moles = jnp.asarray(0.0, dtype=jnp.float64)
    for flow in element_flows.values():
        moles = moles + jnp.asarray(flow, dtype=jnp.float64)
    out: dict[str, SaponificationDuty] = {}
    for ion in counter_ions:
        reagent = base_for_counter_ion(ion)
        base_flow = (
            moles * equivalents_per_mole_ree
            / reagent.equivalents_per_mole / reagent_efficiency
        )
        out[ion] = saponification_duty(
            base_flow, element_flows, reagent, anion, equivalents_per_mole_ree
        )
    return out
