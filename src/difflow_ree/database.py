"""Database loader for REE plugin.

Loads element properties, extractant data, and separation factors
from YAML files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import yaml

import jax.numpy as jnp
from jax import Array


# =============================================================================
# Data Directory
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Element Data Structures
# =============================================================================

@dataclass(frozen=True)
class REEElement:
    """Properties of a rare earth element."""
    symbol: str
    name: str
    atomic_number: int
    atomic_weight: float  # g/mol
    ionic_radius_pm: float  # picometers, CN=6, 3+
    density: float  # g/cm³
    melting_point: float  # K
    oxidation_states: tuple[int, ...]
    group: str  # light, middle, heavy
    oxide_formula: str
    oxide_mw: float  # g/mol
    price_usd_kg: float


class REEDatabase:
    """Database of rare earth element properties."""

    def __init__(self, yaml_path: Path | None = None):
        """Load element data from YAML file.

        Args:
            yaml_path: Path to elements.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "elements.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._elements: dict[str, REEElement] = {}
        self._groups: dict[str, list[str]] = {}

        for symbol, props in data["elements"].items():
            self._elements[symbol] = REEElement(
                symbol=symbol,
                name=props["name"],
                atomic_number=props["atomic_number"],
                atomic_weight=props["atomic_weight"],
                ionic_radius_pm=props["ionic_radius_pm"],
                density=props["density"],
                melting_point=props["melting_point"],
                oxidation_states=tuple(props["oxidation_states"]),
                group=props["group"],
                oxide_formula=props["oxide_formula"],
                oxide_mw=props["oxide_mw"],
                price_usd_kg=props["price_usd_kg"],
            )

        for group_name, group_data in data["groups"].items():
            self._groups[group_name] = group_data["elements"]

    def get(self, symbol: str) -> REEElement:
        """Get element by symbol."""
        if symbol not in self._elements:
            raise KeyError(f"Unknown REE: {symbol}. Available: {list(self._elements.keys())}")
        return self._elements[symbol]

    def __getitem__(self, symbol: str) -> REEElement:
        return self.get(symbol)

    def list_elements(self) -> list[str]:
        """List all available element symbols."""
        return list(self._elements.keys())

    def list_by_group(self, group: str) -> list[str]:
        """List elements in a group (light, middle, heavy)."""
        if group not in self._groups:
            raise KeyError(f"Unknown group: {group}. Available: {list(self._groups.keys())}")
        return self._groups[group]

    def get_atomic_weights(self, symbols: list[str]) -> dict[str, float]:
        """Get atomic weights for multiple elements."""
        return {s: self._elements[s].atomic_weight for s in symbols}

    def get_prices(self, symbols: list[str]) -> dict[str, float]:
        """Get prices (USD/kg) for multiple elements."""
        return {s: self._elements[s].price_usd_kg for s in symbols}

    def get_ionic_radii(self, symbols: list[str]) -> dict[str, float]:
        """Get ionic radii (pm) for multiple elements."""
        return {s: self._elements[s].ionic_radius_pm for s in symbols}


# =============================================================================
# Extractant Data Structures
# =============================================================================

@dataclass(frozen=True)
class PHCoefficients:
    """pH-dependent distribution coefficient parameters.

    log10(D) = a + b*pH + c*pH^2
    """
    a: float
    b: float
    c: float
    d: float = 0.0  # Temperature coefficient (optional)


@dataclass
class Extractant:
    """Properties of an extractant."""
    name: str
    full_name: str
    formula: str
    molecular_weight: float
    density: float
    pKa: float | None
    extractant_type: str
    typical_concentration: float
    stoichiometry_protons: int
    stoichiometry_extractant: int
    ph_coefficients: dict[str, PHCoefficients]
    temperature_coefficients: dict[str, float]
    valid_ph_range: tuple[float, float]
    valid_temp_range: tuple[float, float]
    reference_concentration: float
    concentration_exponent: float
    cost_usd_kg: float


class ExtractantDatabase:
    """Database of extractant properties and equilibrium data."""

    def __init__(self, yaml_path: Path | None = None):
        """Load extractant data from YAML file.

        Args:
            yaml_path: Path to extractants.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "extractants.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._extractants: dict[str, Extractant] = {}
        self._diluents: dict[str, dict] = data.get("diluents", {})
        self._modifiers: dict[str, dict] = data.get("modifiers", {})

        for name, props in data["extractants"].items():
            ph_coeffs = {}
            for element, coeffs in props["ph_coefficients"].items():
                ph_coeffs[element] = PHCoefficients(
                    a=coeffs["a"],
                    b=coeffs["b"],
                    c=coeffs["c"],
                    d=coeffs.get("d", 0.0),
                )

            self._extractants[name] = Extractant(
                name=name,
                full_name=props["full_name"],
                formula=props["formula"],
                molecular_weight=props["molecular_weight"],
                density=props["density"],
                pKa=props.get("pKa"),
                extractant_type=props["type"],
                typical_concentration=props["typical_concentration"],
                stoichiometry_protons=props["stoichiometry"]["protons_released"],
                stoichiometry_extractant=props["stoichiometry"]["extractant_molecules"],
                ph_coefficients=ph_coeffs,
                temperature_coefficients=props["temperature_coefficients"],
                valid_ph_range=tuple(props["valid_ph_range"]),
                valid_temp_range=tuple(props["valid_temp_range"]),
                reference_concentration=props["reference_concentration"],
                concentration_exponent=props["concentration_exponent"],
                cost_usd_kg=props["cost_usd_kg"],
            )

    def get(self, name: str) -> Extractant:
        """Get extractant by name."""
        if name not in self._extractants:
            raise KeyError(f"Unknown extractant: {name}. Available: {list(self._extractants.keys())}")
        return self._extractants[name]

    def __getitem__(self, name: str) -> Extractant:
        return self.get(name)

    def list_extractants(self) -> list[str]:
        """List all available extractant names."""
        return list(self._extractants.keys())

    def get_diluent(self, name: str) -> dict:
        """Get diluent properties."""
        if name not in self._diluents:
            raise KeyError(f"Unknown diluent: {name}")
        return self._diluents[name]

    def list_diluents(self) -> list[str]:
        """List available diluents."""
        return list(self._diluents.keys())


# =============================================================================
# Separation Factor Data
# =============================================================================

@dataclass
class SeparationFactorData:
    """Separation factor data for an extractant."""
    extractant: str
    conditions: dict
    adjacent_pairs: dict[str, float]
    group_pairs: dict[str, float]
    stages_for_99_purity: dict[str, int] | None = None


class SeparationFactorDatabase:
    """Database of separation factors."""

    def __init__(self, yaml_path: Path | None = None):
        """Load separation factor data from YAML file.

        Args:
            yaml_path: Path to separation_factors.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "separation_factors.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._data: dict[str, SeparationFactorData] = {}
        self._stages_data: dict[str, dict[str, int]] = data.get("stages_for_99_purity", {})

        for extractant, sf_data in data["separation_factors"].items():
            stages = self._stages_data.get(extractant)
            self._data[extractant] = SeparationFactorData(
                extractant=extractant,
                conditions=sf_data["conditions"],
                adjacent_pairs=sf_data["adjacent_pairs"],
                group_pairs=sf_data["group_pairs"],
                stages_for_99_purity=stages,
            )

    def get(self, extractant: str) -> SeparationFactorData:
        """Get separation factor data for an extractant."""
        if extractant not in self._data:
            raise KeyError(f"No SF data for extractant: {extractant}")
        return self._data[extractant]

    def get_sf(self, extractant: str, pair: str) -> float:
        """Get separation factor for a specific pair.

        Args:
            extractant: Extractant name (e.g., "D2EHPA")
            pair: Element pair (e.g., "Nd_Pr" or "Ce_La")

        Returns:
            Separation factor (heavier/lighter)
        """
        data = self.get(extractant)
        if pair in data.adjacent_pairs:
            return data.adjacent_pairs[pair]
        if pair in data.group_pairs:
            return data.group_pairs[pair]
        raise KeyError(f"No SF data for pair {pair} with {extractant}")

    def get_stages_needed(self, extractant: str, pair: str) -> int | None:
        """Get estimated stages for 99% purity separation."""
        data = self.get(extractant)
        if data.stages_for_99_purity is None:
            return None
        return data.stages_for_99_purity.get(pair)

    def list_extractants(self) -> list[str]:
        """List extractants with SF data."""
        return list(self._data.keys())


# =============================================================================
# Global Database Instances (lazy loaded)
# =============================================================================

_ree_db: REEDatabase | None = None
_extractant_db: ExtractantDatabase | None = None
_sf_db: SeparationFactorDatabase | None = None


def get_ree_database() -> REEDatabase:
    """Get the REE element database (singleton)."""
    global _ree_db
    if _ree_db is None:
        _ree_db = REEDatabase()
    return _ree_db


def get_extractant_database() -> ExtractantDatabase:
    """Get the extractant database (singleton)."""
    global _extractant_db
    if _extractant_db is None:
        _extractant_db = ExtractantDatabase()
    return _extractant_db


def get_sf_database() -> SeparationFactorDatabase:
    """Get the separation factor database (singleton)."""
    global _sf_db
    if _sf_db is None:
        _sf_db = SeparationFactorDatabase()
    return _sf_db


# =============================================================================
# Convenience Functions
# =============================================================================

def get_element(symbol: str) -> REEElement:
    """Get REE element properties."""
    return get_ree_database().get(symbol)


def get_extractant(name: str) -> Extractant:
    """Get extractant properties."""
    return get_extractant_database().get(name)


def list_ree_elements() -> list[str]:
    """List all available REE symbols."""
    return get_ree_database().list_elements()


def list_extractants() -> list[str]:
    """List all available extractants."""
    return get_extractant_database().list_extractants()


def get_separation_factor(extractant: str, pair: str) -> float:
    """Get separation factor for element pair with given extractant."""
    return get_sf_database().get_sf(extractant, pair)


# =============================================================================
# JAX-Compatible Data Accessors
# =============================================================================

def get_atomic_weight_array(symbols: list[str]) -> Array:
    """Get atomic weights as JAX array.

    Args:
        symbols: List of element symbols in desired order

    Returns:
        JAX array of atomic weights (g/mol)
    """
    db = get_ree_database()
    weights = [db.get(s).atomic_weight for s in symbols]
    return jnp.array(weights)


def get_price_array(symbols: list[str]) -> Array:
    """Get element prices as JAX array.

    Args:
        symbols: List of element symbols in desired order

    Returns:
        JAX array of prices (USD/kg)
    """
    db = get_ree_database()
    prices = [db.get(s).price_usd_kg for s in symbols]
    return jnp.array(prices)


def get_ionic_radius_array(symbols: list[str]) -> Array:
    """Get ionic radii as JAX array.

    Args:
        symbols: List of element symbols in desired order

    Returns:
        JAX array of ionic radii (pm)
    """
    db = get_ree_database()
    radii = [db.get(s).ionic_radius_pm for s in symbols]
    return jnp.array(radii)
