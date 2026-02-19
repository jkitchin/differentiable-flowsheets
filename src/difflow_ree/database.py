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

    def add_extractant(self, name: str, extractant: Extractant) -> None:
        """Add a custom extractant to the database at runtime.

        Args:
            name: Extractant name/identifier (e.g., "MyExtractant")
            extractant: Extractant object with all required properties

        Raises:
            ValueError: If extractant with this name already exists
            TypeError: If extractant is not an Extractant instance
        """
        if not isinstance(extractant, Extractant):
            raise TypeError(f"extractant must be an Extractant instance, got {type(extractant)}")

        if name in self._extractants:
            raise ValueError(
                f"Extractant '{name}' already exists. Use a different name or "
                "remove the existing one first."
            )

        self._extractants[name] = extractant

    def remove_extractant(self, name: str) -> None:
        """Remove an extractant from the database.

        Args:
            name: Extractant name to remove

        Raises:
            KeyError: If extractant doesn't exist
        """
        if name not in self._extractants:
            raise KeyError(f"Extractant '{name}' not found in database")
        del self._extractants[name]

    def update_extractant(self, name: str, extractant: Extractant) -> None:
        """Update an existing extractant in the database.

        Args:
            name: Extractant name to update
            extractant: New Extractant object

        Raises:
            KeyError: If extractant doesn't exist
            TypeError: If extractant is not an Extractant instance
        """
        if not isinstance(extractant, Extractant):
            raise TypeError(f"extractant must be an Extractant instance, got {type(extractant)}")

        if name not in self._extractants:
            raise KeyError(
                f"Extractant '{name}' not found. Use add_extractant() to create new extractants."
            )

        self._extractants[name] = extractant


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
# Custom Extractant Creation
# =============================================================================

def create_custom_extractant(
    name: str,
    full_name: str,
    formula: str,
    molecular_weight: float,
    ph_coefficients: dict[str, dict[str, float]],
    temperature_coefficients: dict[str, float],
    density: float = 1.0,
    pKa: float | None = None,
    extractant_type: str = "custom",
    typical_concentration: float = 0.5,
    stoichiometry_protons: int = 3,
    stoichiometry_extractant: int = 3,
    valid_ph_range: tuple[float, float] = (1.0, 5.0),
    valid_temp_range: tuple[float, float] = (283.0, 333.0),
    reference_concentration: float = 0.5,
    concentration_exponent: float = 3.0,
    cost_usd_kg: float = 10.0,
) -> Extractant:
    """Create a custom extractant with user-defined properties.

    This function validates inputs and creates an Extractant object that can be
    registered with the extractant database for use in simulations.

    Args:
        name: Short name/identifier (e.g., "MyExtractant")
        full_name: Full chemical name
        formula: Chemical formula
        molecular_weight: Molecular weight (g/mol)
        ph_coefficients: pH-dependent distribution coefficients for each REE element.
            Format: {"La": {"a": -8.0, "b": 2.2, "c": 0.01, "d": 0.0}, ...}
            Model: log10(D) = a + b*pH + c*pH^2 + d/T
        temperature_coefficients: Temperature correction for each element (K).
            Format: {"La": -1500, "Nd": -1700, ...}
        density: Density (g/mL), default 1.0
        pKa: Acid dissociation constant (None for neutral extractants)
        extractant_type: Type classification (e.g., "acidic_phosphoric", "custom")
        typical_concentration: Typical operating concentration (M), default 0.5
        stoichiometry_protons: Number of protons released per extraction, default 3
        stoichiometry_extractant: Number of extractant molecules, default 3
        valid_ph_range: Valid pH range as (min, max), default (1.0, 5.0)
        valid_temp_range: Valid temperature range in K as (min, max), default (283, 333)
        reference_concentration: Reference concentration for correlations (M), default 0.5
        concentration_exponent: Exponent n in D ∝ [HA]^n, default 3.0
        cost_usd_kg: Cost in USD per kg, default 10.0

    Returns:
        Extractant object ready for registration

    Raises:
        ValueError: If required parameters are missing or invalid
        TypeError: If parameter types are incorrect

    Example:
        >>> # Create custom extractant with properties for La and Nd
        >>> my_ext = create_custom_extractant(
        ...     name="MyExtractant",
        ...     full_name="My Novel Phosphoric Acid",
        ...     formula="C10H20O4P",
        ...     molecular_weight=250.0,
        ...     ph_coefficients={
        ...         "La": {"a": -8.0, "b": 2.2, "c": 0.01, "d": 0.0},
        ...         "Nd": {"a": -7.5, "b": 2.4, "c": 0.01, "d": 0.0},
        ...     },
        ...     temperature_coefficients={"La": -1500, "Nd": -1700},
        ...     pKa=3.5,
        ... )
        >>>
        >>> # Register it
        >>> from difflow_ree import get_extractant_database
        >>> db = get_extractant_database()
        >>> db.add_extractant("MyExtractant", my_ext)
        >>>
        >>> # Use it
        >>> from difflow_ree import REEDistribution
        >>> dist = REEDistribution(
        ...     extractant="MyExtractant",
        ...     elements=("La", "Nd"),
        ... )
    """
    # Validate required parameters
    if not name:
        raise ValueError("name cannot be empty")
    if not ph_coefficients:
        raise ValueError("ph_coefficients is required and cannot be empty")
    if not temperature_coefficients:
        raise ValueError("temperature_coefficients is required and cannot be empty")

    # Validate pH coefficients structure
    required_coeff_keys = {"a", "b", "c"}
    for element, coeffs in ph_coefficients.items():
        if not isinstance(coeffs, dict):
            raise TypeError(
                f"pH coefficients for {element} must be a dict, got {type(coeffs)}"
            )
        missing = required_coeff_keys - set(coeffs.keys())
        if missing:
            raise ValueError(
                f"pH coefficients for {element} missing required keys: {missing}. "
                "Required: a, b, c (d is optional)"
            )

    # Validate that elements match between pH and temperature coefficients
    ph_elements = set(ph_coefficients.keys())
    temp_elements = set(temperature_coefficients.keys())
    if ph_elements != temp_elements:
        raise ValueError(
            f"Element mismatch between pH coefficients and temperature coefficients. "
            f"pH elements: {ph_elements}, Temperature elements: {temp_elements}"
        )

    # Convert pH coefficient dicts to PHCoefficients objects
    ph_coeffs_objects = {}
    for element, coeffs in ph_coefficients.items():
        ph_coeffs_objects[element] = PHCoefficients(
            a=float(coeffs["a"]),
            b=float(coeffs["b"]),
            c=float(coeffs["c"]),
            d=float(coeffs.get("d", 0.0)),
        )

    # Create and return Extractant object
    return Extractant(
        name=name,
        full_name=full_name,
        formula=formula,
        molecular_weight=float(molecular_weight),
        density=float(density),
        pKa=float(pKa) if pKa is not None else None,
        extractant_type=extractant_type,
        typical_concentration=float(typical_concentration),
        stoichiometry_protons=int(stoichiometry_protons),
        stoichiometry_extractant=int(stoichiometry_extractant),
        ph_coefficients=ph_coeffs_objects,
        temperature_coefficients=temperature_coefficients,
        valid_ph_range=tuple(valid_ph_range),
        valid_temp_range=tuple(valid_temp_range),
        reference_concentration=float(reference_concentration),
        concentration_exponent=float(concentration_exponent),
        cost_usd_kg=float(cost_usd_kg),
    )


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
