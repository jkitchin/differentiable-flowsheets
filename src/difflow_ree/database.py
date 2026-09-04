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
    """Properties of a rare earth element.

    The heat-of-reaction fields (#119) are optional and default to ``None``;
    they are not fabricated. Supply them (with a literature citation) via
    ``elements.yaml`` or the add/update-element API when energy-balance or
    long-term economics modeling requires them.
    """
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
    # Optional thermodynamic properties (kJ/mol), user-supplied with citation.
    heat_of_extraction: float | None = None
    heat_of_scrubbing: float | None = None
    heat_of_stripping: float | None = None


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
                heat_of_extraction=props.get("heat_of_extraction"),
                heat_of_scrubbing=props.get("heat_of_scrubbing"),
                heat_of_stripping=props.get("heat_of_stripping"),
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

    def add_element(self, symbol: str, element: REEElement) -> None:
        """Add a custom element to the database at runtime.

        Args:
            symbol: Element symbol (e.g., "Ho")
            element: REEElement object with all required properties

        Raises:
            ValueError: If element with this symbol already exists
            TypeError: If element is not an REEElement instance
        """
        if not isinstance(element, REEElement):
            raise TypeError(f"element must be an REEElement instance, got {type(element)}")
        if symbol in self._elements:
            raise ValueError(
                f"Element '{symbol}' already exists. Use update_element() to modify "
                "or remove_element() first."
            )
        self._elements[symbol] = element

        # Add to appropriate group if it exists
        group = element.group
        if group in self._groups:
            if symbol not in self._groups[group]:
                self._groups[group].append(symbol)
        else:
            self._groups[group] = [symbol]

    def remove_element(self, symbol: str) -> None:
        """Remove an element from the database.

        Args:
            symbol: Element symbol to remove

        Raises:
            KeyError: If element doesn't exist
        """
        if symbol not in self._elements:
            raise KeyError(f"Element '{symbol}' not found in database")
        group = self._elements[symbol].group
        del self._elements[symbol]
        if group in self._groups and symbol in self._groups[group]:
            self._groups[group].remove(symbol)

    def update_element(self, symbol: str, element: REEElement) -> None:
        """Update an existing element in the database.

        Args:
            symbol: Element symbol to update
            element: New REEElement object

        Raises:
            KeyError: If element doesn't exist
            TypeError: If element is not an REEElement instance
        """
        if not isinstance(element, REEElement):
            raise TypeError(f"element must be an REEElement instance, got {type(element)}")
        if symbol not in self._elements:
            raise KeyError(
                f"Element '{symbol}' not found. Use add_element() to create new elements."
            )
        old_group = self._elements[symbol].group
        new_group = element.group
        self._elements[symbol] = element

        # Update group membership if group changed
        if old_group != new_group:
            if old_group in self._groups and symbol in self._groups[old_group]:
                self._groups[old_group].remove(symbol)
            if new_group in self._groups:
                if symbol not in self._groups[new_group]:
                    self._groups[new_group].append(symbol)
            else:
                self._groups[new_group] = [symbol]


# =============================================================================
# Extractant Data Structures
# =============================================================================

@dataclass(frozen=True)
class PHCoefficients:
    """Quadratic distribution-coefficient parameters in a driving variable.

    For a cation-exchange extractant the driving variable is pH on the
    *concentration* scale (``pH = -log10([H+])``, see the header of
    ``data/extractants.yaml``)::

        log10(D) = a + b*pH + c*pH^2

    The same container is reused for the solvating extractants (#195), where
    the driving variable is the nitrate log-ratio ``s = log10([NO3-]/[NO3-]_ref)``
    referenced to :attr:`Extractant.reference_nitrate`::

        log10(D) = a + b*s + c*s^2

    so that ``a`` is ``log10(D)`` *at the reference nitrate concentration* and
    ``b`` is the nitrate slope ``d log10(D) / d log10([NO3-])``.
    """
    a: float
    b: float
    c: float
    d: float = 0.0  # Temperature coefficient (optional)


# Normalized extraction mechanisms (#195). The mechanism decides which
# coefficient block drives D and whether a proton term appears in the
# activity correction (#194); it is data, not an assumption in the code.
EXTRACTION_MECHANISMS = ("cation_exchange", "solvating")

# Mapping from the historical free-form ``type`` field to the normalized
# mechanism. Stated explicitly in data/extractants.yaml as well (#195).
_TYPE_TO_MECHANISM = {
    "acidic_phosphoric": "cation_exchange",
    "acidic_phosphonic": "cation_exchange",
    "acidic_phosphinic": "cation_exchange",
    "acidic_carboxylic": "cation_exchange",
    "solvating_neutral": "solvating",
}


def normalize_mechanism(extractant_type: str) -> str:
    """Normalize an extractant ``type`` string to an extraction mechanism.

    Args:
        extractant_type: Free-form type from the extractant record, e.g.
            ``"acidic_phosphoric"`` or ``"solvating_neutral"``.

    Returns:
        One of :data:`EXTRACTION_MECHANISMS`. Types that are not recognized
        (including the ``"custom"`` default of
        :func:`create_custom_extractant`) fall back to ``"cation_exchange"``,
        which is the mechanism the ``ph_coefficients`` block describes.
    """
    if extractant_type in _TYPE_TO_MECHANISM:
        return _TYPE_TO_MECHANISM[extractant_type]
    if extractant_type.startswith("acidic"):
        return "cation_exchange"
    if extractant_type.startswith("solvating"):
        return "solvating"
    return "cation_exchange"


def _load_coefficient_block(
    block: dict | None,
) -> dict[str, PHCoefficients] | None:
    """Convert a YAML ``{element: {a, b, c, d}}`` block to PHCoefficients.

    Args:
        block: Raw YAML mapping, or None when the extractant does not carry
            that coefficient block.

    Returns:
        Mapping of element symbol to :class:`PHCoefficients`, or None if
        ``block`` was None.
    """
    if block is None:
        return None
    return {
        element: PHCoefficients(
            a=coeffs["a"],
            b=coeffs["b"],
            c=coeffs["c"],
            d=coeffs.get("d", 0.0),
        )
        for element, coeffs in block.items()
    }


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
    # None when the record carries no pH-driven correlation at all. That is not
    # an oversight: TBP's ``ph_coefficients`` block was DELETED because it
    # modelled a neutral solvating extractant (pKa null, protons_released 0) as
    # a weak cation exchanger, with no proton to exchange and no source. Any
    # consumer of this field must therefore handle None; there is no empty-dict
    # stand-in and no fallback to another block.
    ph_coefficients: dict[str, PHCoefficients] | None
    temperature_coefficients: dict[str, float]
    valid_ph_range: tuple[float, float]
    valid_temp_range: tuple[float, float]
    reference_concentration: float
    concentration_exponent: float
    cost_usd_kg: float
    # Optional degradation / thermo properties (#119), user-supplied with
    # citation; default None (no fabricated values). Solvent degradation rate
    # in 1/h; heats of extraction/scrubbing/stripping in kJ/mol.
    degradation_rate: float | None = None
    heat_of_extraction: float | None = None
    heat_of_scrubbing: float | None = None
    heat_of_stripping: float | None = None
    # Basis on which ``stoichiometry_extractant`` counts extractant species:
    # "dimer" for the acidic organophosphorus extractants, which are dimeric in
    # aliphatic diluents, "monomer" for neutral/solvating extractants (#191).
    stoichiometry_basis: str = "monomer"
    # Extraction mechanism (#195). None normalizes from ``extractant_type`` in
    # __post_init__; the YAML states it explicitly so the mechanism is data
    # rather than an inference.
    mechanism: str | None = None
    # Solvating-extraction data (#195). ``nitrate_coefficients`` is shaped like
    # ``ph_coefficients`` but is driven by log10([NO3-]/reference_nitrate).
    nitrate_coefficients: dict[str, PHCoefficients] | None = None
    reference_nitrate: float | None = None
    # True when the extractant only works from a nitrate (salting) medium, so
    # using it without a nitrate concentration is an error rather than a
    # silent fall-back to the pH correlation (#195).
    requires_nitrate: bool = False

    def __post_init__(self):
        """Normalize the extraction mechanism (#195)."""
        if self.mechanism is None:
            self.mechanism = normalize_mechanism(self.extractant_type)
        if self.mechanism not in EXTRACTION_MECHANISMS:
            raise ValueError(
                f"Extractant '{self.name}': unknown mechanism "
                f"{self.mechanism!r}. Supported mechanisms: "
                f"{list(EXTRACTION_MECHANISMS)}."
            )

    @property
    def monomers_per_ree(self) -> float:
        """Extractant monomer equivalents bound per mol REE.

        This is the ``m`` in the extractant balance

            [HA]_total = [HA]_free + m * [RE-complex]

        so the maximum organic loading is ``1 / m`` mol REE per mol extractant.
        Derived from the declared stoichiometry and its basis rather than
        hard-coded, so that the capacity and the free-extractant exponent can
        never disagree (#191).
        """
        per_species = 2.0 if self.stoichiometry_basis == "dimer" else 1.0
        return self.stoichiometry_extractant * per_species

    @property
    def max_loading(self) -> float:
        """Maximum REE loading capacity (mol REE per mol extractant)."""
        return 1.0 / self.monomers_per_ree


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
            ph_coeffs = _load_coefficient_block(props.get("ph_coefficients"))
            # Solvating-extraction coefficients, driven by the nitrate
            # log-ratio rather than by pH (#195). Absent for the acidic
            # cation exchangers.
            nitrate_coeffs = _load_coefficient_block(
                props.get("nitrate_coefficients")
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
                stoichiometry_basis=props["stoichiometry"].get("basis", "monomer"),
                ph_coefficients=ph_coeffs,
                temperature_coefficients=props["temperature_coefficients"],
                valid_ph_range=tuple(props["valid_ph_range"]),
                valid_temp_range=tuple(props["valid_temp_range"]),
                reference_concentration=props["reference_concentration"],
                concentration_exponent=props["concentration_exponent"],
                cost_usd_kg=props["cost_usd_kg"],
                degradation_rate=props.get("degradation_rate"),
                heat_of_extraction=props.get("heat_of_extraction"),
                heat_of_scrubbing=props.get("heat_of_scrubbing"),
                heat_of_stripping=props.get("heat_of_stripping"),
                # Mechanism and solvating-extraction data (#195)
                mechanism=props.get("mechanism"),
                nitrate_coefficients=nitrate_coeffs,
                reference_nitrate=props.get("reference_nitrate"),
                requires_nitrate=bool(
                    props["stoichiometry"].get("requires_nitrate", False)
                ),
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

    def add_element_to_extractant(
        self,
        extractant_name: str,
        element: str,
        ph_coefficients: dict[str, float],
        temperature_coefficient: float,
    ) -> None:
        """Add pH and temperature coefficients for a new element to an existing extractant.

        This allows using a new element (e.g., Ho) with an extractant without
        needing data for every element in the database.

        Args:
            extractant_name: Name of an existing extractant (e.g., "PC88A")
            element: Element symbol (e.g., "Ho")
            ph_coefficients: Dict with keys "a", "b", "c" and optional "d".
                Model: log10(D) = a + b*pH + c*pH^2 + d/T
            temperature_coefficient: Temperature correction coefficient (K).
                Used as: d*(1/T - 1/T_ref)

        Raises:
            KeyError: If extractant doesn't exist
            ValueError: If the extractant carries no ``ph_coefficients`` block
                at all, if the element already has data for this extractant, or
                if required coefficient keys are missing
        """
        if extractant_name not in self._extractants:
            raise KeyError(
                f"Extractant '{extractant_name}' not found. "
                f"Available: {list(self._extractants.keys())}"
            )

        extractant = self._extractants[extractant_name]

        if extractant.ph_coefficients is None:
            raise ValueError(
                f"Extractant '{extractant_name}' carries no 'ph_coefficients' "
                f"block, so a pH-driven coefficient for '{element}' cannot be "
                f"added to it. Its mechanism is {extractant.mechanism!r}; for "
                "a solvating extractant the pH-driven correlation does not "
                "exist and creating one here would reintroduce exactly the "
                "unsupported model that was deleted from the record. Add to "
                "'nitrate_coefficients' instead, or register a separate "
                "extractant with fitted pH coefficients."
            )

        if element in extractant.ph_coefficients:
            raise ValueError(
                f"Element '{element}' already has coefficients for '{extractant_name}'. "
                "Remove and re-add to update."
            )

        required_keys = {"a", "b", "c"}
        missing = required_keys - set(ph_coefficients.keys())
        if missing:
            raise ValueError(
                f"pH coefficients missing required keys: {missing}. "
                "Required: a, b, c (d is optional)"
            )

        extractant.ph_coefficients[element] = PHCoefficients(
            a=float(ph_coefficients["a"]),
            b=float(ph_coefficients["b"]),
            c=float(ph_coefficients["c"]),
            d=float(ph_coefficients.get("d", 0.0)),
        )
        extractant.temperature_coefficients[element] = float(temperature_coefficient)

    def remove_element_from_extractant(
        self,
        extractant_name: str,
        element: str,
    ) -> None:
        """Remove an element's coefficients from an extractant.

        Args:
            extractant_name: Name of the extractant
            element: Element symbol to remove

        Raises:
            KeyError: If extractant or element not found
        """
        if extractant_name not in self._extractants:
            raise KeyError(f"Extractant '{extractant_name}' not found")

        extractant = self._extractants[extractant_name]
        if (
            extractant.ph_coefficients is None
            or element not in extractant.ph_coefficients
        ):
            raise KeyError(
                f"Element '{element}' not found in extractant '{extractant_name}'"
            )

        del extractant.ph_coefficients[element]
        extractant.temperature_coefficients.pop(element, None)

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

    def add_pair(
        self,
        extractant: str,
        pair: str,
        sf: float,
        adjacent: bool = True,
        stages_99: int | None = None,
    ) -> None:
        """Add a separation factor for an element pair.

        Args:
            extractant: Extractant name (e.g., "PC88A"). Must already exist
                in the database (either from YAML or via add_separation_factors).
            pair: Element pair string (e.g., "Ho_Dy"). Convention: heavier_lighter.
            sf: Separation factor value (D_heavier / D_lighter, typically > 1)
            adjacent: If True, store as adjacent pair; if False, as group pair
            stages_99: Estimated stages for 99% purity separation (optional)

        Raises:
            KeyError: If extractant not found. Use add_separation_factors() to
                create a new extractant entry first.
            ValueError: If pair already exists for this extractant
        """
        if extractant not in self._data:
            raise KeyError(
                f"No SF data for extractant '{extractant}'. "
                "Use add_separation_factors() to create a new entry first."
            )

        data = self._data[extractant]
        if pair in data.adjacent_pairs or pair in data.group_pairs:
            raise ValueError(
                f"Pair '{pair}' already exists for '{extractant}'. "
                "Use remove_pair() first to update."
            )

        if adjacent:
            data.adjacent_pairs[pair] = float(sf)
        else:
            data.group_pairs[pair] = float(sf)

        if stages_99 is not None:
            if data.stages_for_99_purity is None:
                data.stages_for_99_purity = {}
            data.stages_for_99_purity[pair] = int(stages_99)

    def remove_pair(self, extractant: str, pair: str) -> None:
        """Remove a separation factor pair.

        Args:
            extractant: Extractant name
            pair: Element pair string

        Raises:
            KeyError: If extractant or pair not found
        """
        if extractant not in self._data:
            raise KeyError(f"No SF data for extractant '{extractant}'")

        data = self._data[extractant]
        if pair in data.adjacent_pairs:
            del data.adjacent_pairs[pair]
        elif pair in data.group_pairs:
            del data.group_pairs[pair]
        else:
            raise KeyError(f"Pair '{pair}' not found for '{extractant}'")

        if data.stages_for_99_purity and pair in data.stages_for_99_purity:
            del data.stages_for_99_purity[pair]

    def add_separation_factors(
        self,
        extractant: str,
        conditions: dict,
        adjacent_pairs: dict[str, float] | None = None,
        group_pairs: dict[str, float] | None = None,
        stages_for_99_purity: dict[str, int] | None = None,
    ) -> None:
        """Add separation factor data for a new extractant.

        Use this to create SF data for extractants not in the YAML database,
        or for custom extractants added via ExtractantDatabase.add_extractant().

        Args:
            extractant: Extractant name (e.g., "MyExtractant")
            conditions: Operating conditions dict (e.g., {"pH": 3.0, "temperature_K": 298})
            adjacent_pairs: Adjacent pair separation factors (e.g., {"Ho_Dy": 1.4})
            group_pairs: Non-adjacent group pair separation factors
            stages_for_99_purity: Estimated stages for 99% purity

        Raises:
            ValueError: If extractant already has SF data
        """
        if extractant in self._data:
            raise ValueError(
                f"SF data for '{extractant}' already exists. "
                "Use add_pair() to add individual pairs, or remove and re-add."
            )

        self._data[extractant] = SeparationFactorData(
            extractant=extractant,
            conditions=conditions,
            adjacent_pairs=dict(adjacent_pairs) if adjacent_pairs else {},
            group_pairs=dict(group_pairs) if group_pairs else {},
            stages_for_99_purity=dict(stages_for_99_purity) if stages_for_99_purity else None,
        )

    def remove_separation_factors(self, extractant: str) -> None:
        """Remove all separation factor data for an extractant.

        Args:
            extractant: Extractant name to remove

        Raises:
            KeyError: If extractant not found
        """
        if extractant not in self._data:
            raise KeyError(f"No SF data for extractant '{extractant}'")
        del self._data[extractant]


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
    stoichiometry_basis: str = "monomer",
    valid_ph_range: tuple[float, float] = (1.0, 5.0),
    valid_temp_range: tuple[float, float] = (283.0, 333.0),
    reference_concentration: float = 0.5,
    concentration_exponent: float = 3.0,
    cost_usd_kg: float = 10.0,
    mechanism: str | None = None,
    nitrate_coefficients: dict[str, dict[str, float]] | None = None,
    reference_nitrate: float | None = None,
    requires_nitrate: bool = False,
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
            Still REQUIRED here, even for ``mechanism="solvating"``. Note the
            asymmetry with the YAML path, where a record may omit the block
            entirely (TBP does, and :attr:`Extractant.ph_coefficients` is
            therefore ``dict | None``): a custom solvating extractant must
            still supply one. Loosening this is a separate change.
        temperature_coefficients: Temperature correction for each element (K).
            Format: {"La": -1500, "Nd": -1700, ...}
        density: Density (g/mL), default 1.0
        pKa: Acid dissociation constant (None for neutral extractants)
        extractant_type: Type classification (e.g., "acidic_phosphoric", "custom")
        typical_concentration: Typical operating concentration (M), default 0.5
        stoichiometry_protons: Number of protons released per extraction, default 3
        stoichiometry_extractant: Number of extractant molecules, default 3
        stoichiometry_basis: What ``stoichiometry_extractant`` counts,
            ``"monomer"`` (default, matching :class:`Extractant`) or
            ``"dimer"`` (#191). The acidic organophosphorus extractants
            (D2EHPA, PC88A, Cyanex 272) are DIMERIC in aliphatic diluents:
            their ``stoichiometry_extractant = 3`` counts three dimers, i.e.
            six monomer equivalents per REE, so
            :attr:`Extractant.monomers_per_ree` is 6 and
            :attr:`Extractant.max_loading` is 1/6. Leaving this at
            ``"monomer"`` for such an extractant reports 3 and 1/3 instead --
            a factor-of-two capacity error, which is exactly the bug #191 was
            filed about. Pass ``stoichiometry_basis="dimer"`` whenever the
            custom extractant is a dimerizing acid.
        valid_ph_range: Valid pH range as (min, max), default (1.0, 5.0)
        valid_temp_range: Valid temperature range in K as (min, max), default (283, 333)
        reference_concentration: Reference concentration for correlations (M), default 0.5
        concentration_exponent: Exponent n in D ∝ [HA]^n, default 3.0
        cost_usd_kg: Cost in USD per kg, default 10.0
        mechanism: Extraction mechanism, one of :data:`EXTRACTION_MECHANISMS`
            (#195). None normalizes from ``extractant_type``; an unrecognized
            type (including the ``"custom"`` default) normalizes to
            ``"cation_exchange"``, which is what ``ph_coefficients`` describes.
        nitrate_coefficients: Solvating-extraction coefficients keyed by element,
            same ``{a, b, c, d}`` shape as ``ph_coefficients`` but driven by
            ``log10([NO3-]/reference_nitrate)`` (#195). Required for
            ``mechanism="solvating"``.
        reference_nitrate: Nitrate concentration (M) the ``nitrate_coefficients``
            are referenced to, i.e. the concentration at which ``a`` equals
            ``log10(D)``.
        requires_nitrate: True when the extractant only works from a nitrate
            medium, so using it without a nitrate concentration is an error
            rather than a silent fall-back to the pH correlation (#195).

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
    if stoichiometry_basis not in ("monomer", "dimer"):
        raise ValueError(
            f"stoichiometry_basis must be 'monomer' or 'dimer', got "
            f"{stoichiometry_basis!r}. The acidic organophosphorus extractants "
            "are dimeric in aliphatic diluents (#191)."
        )
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
        # (#191) Without this the custom path silently reports monomers_per_ree
        # = 3 for a dimeric extractant that every built-in acidic record
        # reports as 6.
        stoichiometry_basis=stoichiometry_basis,
        ph_coefficients=ph_coeffs_objects,
        temperature_coefficients=temperature_coefficients,
        valid_ph_range=tuple(valid_ph_range),
        valid_temp_range=tuple(valid_temp_range),
        reference_concentration=float(reference_concentration),
        concentration_exponent=float(concentration_exponent),
        cost_usd_kg=float(cost_usd_kg),
        # Mechanism / solvating-extraction data (#195)
        mechanism=mechanism,
        nitrate_coefficients=_load_coefficient_block(nitrate_coefficients),
        reference_nitrate=(
            float(reference_nitrate) if reference_nitrate is not None else None
        ),
        requires_nitrate=bool(requires_nitrate),
    )


def create_custom_element(
    symbol: str,
    name: str,
    atomic_number: int,
    atomic_weight: float,
    ionic_radius_pm: float,
    density: float,
    melting_point: float,
    group: str,
    oxide_formula: str,
    oxide_mw: float,
    oxidation_states: tuple[int, ...] = (3,),
    price_usd_kg: float = 0.0,
) -> REEElement:
    """Create a custom REE element for registration with the database.

    This function validates inputs and creates an REEElement that can be
    registered with the element database via REEDatabase.add_element().

    Args:
        symbol: Element symbol (e.g., "Ho")
        name: Full element name (e.g., "Holmium")
        atomic_number: Atomic number
        atomic_weight: Atomic weight (g/mol)
        ionic_radius_pm: Ionic radius in picometers (CN=6, 3+ oxidation)
        density: Density (g/cm3)
        melting_point: Melting point (K)
        group: Group classification ("light", "middle", or "heavy")
        oxide_formula: Chemical formula of the oxide (e.g., "Ho2O3")
        oxide_mw: Molecular weight of the oxide (g/mol)
        oxidation_states: Tuple of possible oxidation states, default (3,)
        price_usd_kg: Market price in USD/kg, default 0.0

    Returns:
        REEElement object ready for registration

    Raises:
        ValueError: If required parameters are missing or invalid

    Example:
        >>> ho = create_custom_element(
        ...     symbol="Ho",
        ...     name="Holmium",
        ...     atomic_number=67,
        ...     atomic_weight=164.930,
        ...     ionic_radius_pm=90.1,
        ...     density=8.795,
        ...     melting_point=1734,
        ...     group="heavy",
        ...     oxide_formula="Ho2O3",
        ...     oxide_mw=377.86,
        ...     price_usd_kg=60.0,
        ... )
        >>>
        >>> from difflow_ree import get_ree_database
        >>> db = get_ree_database()
        >>> db.add_element("Ho", ho)
    """
    if not symbol:
        raise ValueError("symbol cannot be empty")
    if not name:
        raise ValueError("name cannot be empty")
    if group not in ("light", "middle", "heavy"):
        raise ValueError(f"group must be 'light', 'middle', or 'heavy', got '{group}'")
    if atomic_weight <= 0:
        raise ValueError(f"atomic_weight must be positive, got {atomic_weight}")
    if ionic_radius_pm <= 0:
        raise ValueError(f"ionic_radius_pm must be positive, got {ionic_radius_pm}")

    return REEElement(
        symbol=symbol,
        name=name,
        atomic_number=int(atomic_number),
        atomic_weight=float(atomic_weight),
        ionic_radius_pm=float(ionic_radius_pm),
        density=float(density),
        melting_point=float(melting_point),
        oxidation_states=tuple(oxidation_states),
        group=group,
        oxide_formula=oxide_formula,
        oxide_mw=float(oxide_mw),
        price_usd_kg=float(price_usd_kg),
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
