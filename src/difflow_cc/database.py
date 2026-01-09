"""Database module for Carbon Capture plugin.

Provides access to properties of amine solvents, adsorbent materials,
and membrane materials for CO2 capture simulations.

All property data is loaded from YAML files with literature-validated
parameters and appropriate references.

References:
    Solvents:
        - Versteeg GF, Van Swaaij WPM (1988). J Chem Eng Data 33:29
        - Kim I, Svendsen HF (2007). Int J Greenhouse Gas Control 1:390
        - Bishnoi S, Rochelle GT (2000). Chem Eng Sci 55:5531
    Adsorbents:
        - Cavenati S et al. (2004). J Chem Eng Data 49:1095
        - Caskey SR et al. (2008). J Am Chem Soc 130:10870
    Membranes:
        - Robeson LM (2008). J Membr Sci 320:390
        - Baker RW (2012). Membrane Technology and Applications
"""

__all__ = [
    "SolventDatabase",
    "AdsorbentDatabase",
    "MembraneDatabase",
    "AmineSolvent",
    "SolventBlend",
    "Adsorbent",
    "IsothermParams",
    "Membrane",
    "get_solvent",
    "get_adsorbent",
    "get_membrane",
    "list_solvents",
    "list_adsorbents",
    "list_membranes",
    "get_solvent_kinetic_params",
    "get_isotherm_params_array",
]

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import jax.numpy as jnp
from jax import Array


# =============================================================================
# Data Directory
# =============================================================================

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Solvent Data Structures
# =============================================================================

@dataclass(frozen=True)
class AmineSolvent:
    """Properties of an amine solvent for CO2 capture.

    Attributes:
        name: Short name (e.g., 'MEA')
        full_name: Full chemical name
        formula: Molecular formula
        solvent_type: primary, secondary, tertiary, cyclic_diamine, etc.
        MW: Molecular weight (g/mol)
        density: Liquid density at 25°C (kg/m³)
        viscosity_A: Arrhenius pre-exponential for viscosity
        viscosity_B: Arrhenius activation parameter (K)
        pKa_25C: Acid dissociation constant at 25°C
        pKa_a: Temperature dependence coefficient a
        pKa_b: Temperature dependence coefficient b (K)
        heat_of_absorption: Heat of CO2 absorption (kJ/mol)
        loading_capacity: Max CO2 loading (mol CO2/mol amine)
        kinetics: Dict with mechanism, A, Ea, k2_25C
        typical_concentration: Typical use concentration (wt%)
        regen_temperature: Regeneration temperature (K)
        regen_energy: Specific regeneration energy (GJ/tonne CO2)
        cost_usd_kg: Cost (USD/kg)
    """
    name: str
    full_name: str
    formula: str
    solvent_type: str
    MW: float
    density: float
    viscosity_A: float
    viscosity_B: float
    pKa_25C: float
    pKa_a: float
    pKa_b: float
    heat_of_absorption: float
    loading_capacity: float
    henry_a: float
    henry_b: float
    henry_c: float
    kinetics: dict[str, Any]
    diffusivity_D0: float
    diffusivity_Ed: float
    typical_concentration: float
    regen_temperature: float
    regen_energy: float
    cost_usd_kg: float


@dataclass(frozen=True)
class SolventBlend:
    """Properties of a solvent blend.

    Attributes:
        name: Blend name (e.g., 'CESAR1')
        description: Description of blend
        components: Dict mapping component name to mass fraction
        heat_of_absorption: Average heat of absorption (kJ/mol)
        regen_energy: Regeneration energy (GJ/tonne CO2)
    """
    name: str
    description: str
    components: dict[str, float]
    heat_of_absorption: float
    regen_energy: float


class SolventDatabase:
    """Database of amine solvent properties.

    Example:
        >>> db = SolventDatabase()
        >>> mea = db.get('MEA')
        >>> print(f"MEA k2 at 25°C: {mea.kinetics['k2_25C']} L/(mol*s)")
    """

    def __init__(self, yaml_path: Path | None = None):
        """Load solvent data from YAML file.

        Args:
            yaml_path: Path to solvents.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "solvents.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._solvents: dict[str, AmineSolvent] = {}
        self._blends: dict[str, SolventBlend] = {}

        for name, props in data["solvents"].items():
            self._solvents[name] = AmineSolvent(
                name=name,
                full_name=props["full_name"],
                formula=props["formula"],
                solvent_type=props["type"],
                MW=props["molecular_weight"],
                density=props["density_25C"],
                viscosity_A=props["viscosity_A"],
                viscosity_B=props["viscosity_B"],
                pKa_25C=props["pKa_25C"],
                pKa_a=props["pKa_a"],
                pKa_b=props["pKa_b"],
                heat_of_absorption=props["heat_of_absorption"],
                loading_capacity=props["loading_capacity"],
                henry_a=props["henry_a"],
                henry_b=props["henry_b"],
                henry_c=props["henry_c"],
                kinetics=props["kinetics"],
                diffusivity_D0=props["diffusivity_D0"],
                diffusivity_Ed=props["diffusivity_Ed"],
                typical_concentration=props["typical_concentration"],
                regen_temperature=props["regen_temperature"],
                regen_energy=props["regen_energy"],
                cost_usd_kg=props["cost_usd_kg"],
            )

        for name, blend_data in data.get("blends", {}).items():
            self._blends[name] = SolventBlend(
                name=name,
                description=blend_data["description"],
                components=blend_data["components"],
                heat_of_absorption=blend_data["heat_of_absorption"],
                regen_energy=blend_data["regen_energy"],
            )

    def get(self, name: str) -> AmineSolvent:
        """Get solvent by name."""
        if name not in self._solvents:
            raise KeyError(
                f"Unknown solvent: {name}. "
                f"Available: {list(self._solvents.keys())}"
            )
        return self._solvents[name]

    def __getitem__(self, name: str) -> AmineSolvent:
        return self.get(name)

    def list_solvents(self) -> list[str]:
        """List all available solvent names."""
        return list(self._solvents.keys())

    def list_by_type(self, solvent_type: str) -> list[str]:
        """List solvents by type (primary, secondary, tertiary, etc.)."""
        return [
            name for name, s in self._solvents.items()
            if s.solvent_type == solvent_type
        ]

    def get_blend(self, name: str) -> SolventBlend:
        """Get solvent blend by name."""
        if name not in self._blends:
            raise KeyError(
                f"Unknown blend: {name}. Available: {list(self._blends.keys())}"
            )
        return self._blends[name]

    def list_blends(self) -> list[str]:
        """List all available blend names."""
        return list(self._blends.keys())


# =============================================================================
# Adsorbent Data Structures
# =============================================================================

@dataclass(frozen=True)
class IsothermParams:
    """Parameters for an adsorption isotherm model.

    Attributes:
        model: Model type ('langmuir', 'sips', 'toth', 'dual_site_langmuir')
        params: Model-specific parameters
    """
    model: str
    params: dict[str, float]


@dataclass(frozen=True)
class Adsorbent:
    """Properties of an adsorbent material.

    Attributes:
        name: Short name
        full_name: Full name
        material_type: zeolite, MOF, activated_carbon, amine_functionalized
        surface_area: BET surface area (m²/g)
        pore_volume: Total pore volume (cm³/g)
        pore_diameter: Median pore diameter (Å)
        density_bulk: Bulk density (kg/m³)
        density_particle: Particle density (kg/m³)
        heat_capacity: Specific heat (J/(kg·K))
        thermal_conductivity: Thermal conductivity (W/(m·K))
        isotherms: Dict mapping species to IsothermParams
        CO2_capacity: CO2 capacity at 1 bar, 25°C (mol/kg)
        CO2_selectivity: CO2/N2 selectivity
        heat_of_adsorption: Isosteric heat of CO2 adsorption (kJ/mol)
        regen_temperature: Typical regeneration temperature (K)
        cost_usd_kg: Cost (USD/kg)
    """
    name: str
    full_name: str
    material_type: str
    surface_area: float
    pore_volume: float
    pore_diameter: float
    density_bulk: float
    density_particle: float
    heat_capacity: float
    thermal_conductivity: float
    isotherms: dict[str, IsothermParams]
    CO2_capacity: float
    CO2_selectivity: float
    heat_of_adsorption: float
    regen_temperature: float
    desorption_pressure: float
    cost_usd_kg: float


class AdsorbentDatabase:
    """Database of adsorbent material properties.

    Example:
        >>> db = AdsorbentDatabase()
        >>> zeolite = db.get('Zeolite_13X')
        >>> print(f"CO2 capacity: {zeolite.CO2_capacity} mol/kg")
    """

    def __init__(self, yaml_path: Path | None = None):
        """Load adsorbent data from YAML file.

        Args:
            yaml_path: Path to adsorbents.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "adsorbents.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._adsorbents: dict[str, Adsorbent] = {}

        for name, props in data["adsorbents"].items():
            # Parse isotherms
            isotherms = {}
            for species, iso_data in props.get("isotherms", {}).items():
                model = iso_data["model"]
                params = {k: v for k, v in iso_data.items() if k != "model"}
                isotherms[species] = IsothermParams(model=model, params=params)

            # Get selectivity (try CO2/N2, then CO2/CH4)
            selectivity = props.get("CO2_N2_selectivity",
                                     props.get("CO2_CH4_selectivity", 1.0))

            self._adsorbents[name] = Adsorbent(
                name=name,
                full_name=props["full_name"],
                material_type=props["type"],
                surface_area=props["surface_area"],
                pore_volume=props["pore_volume"],
                pore_diameter=props["pore_diameter"],
                density_bulk=props["density_bulk"],
                density_particle=props["density_particle"],
                heat_capacity=props["heat_capacity"],
                thermal_conductivity=props["thermal_conductivity"],
                isotherms=isotherms,
                CO2_capacity=props["CO2_capacity_1bar_25C"],
                CO2_selectivity=selectivity,
                heat_of_adsorption=props["heat_of_adsorption_CO2"],
                regen_temperature=props["regen_temperature"],
                desorption_pressure=props["desorption_pressure"],
                cost_usd_kg=props["cost_usd_kg"],
            )

    def get(self, name: str) -> Adsorbent:
        """Get adsorbent by name."""
        if name not in self._adsorbents:
            raise KeyError(
                f"Unknown adsorbent: {name}. "
                f"Available: {list(self._adsorbents.keys())}"
            )
        return self._adsorbents[name]

    def __getitem__(self, name: str) -> Adsorbent:
        return self.get(name)

    def list_adsorbents(self) -> list[str]:
        """List all available adsorbent names."""
        return list(self._adsorbents.keys())

    def list_by_type(self, material_type: str) -> list[str]:
        """List adsorbents by type (zeolite, MOF, etc.)."""
        return [
            name for name, a in self._adsorbents.items()
            if a.material_type == material_type
        ]


# =============================================================================
# Membrane Data Structures
# =============================================================================

@dataclass(frozen=True)
class Membrane:
    """Properties of a membrane material.

    Attributes:
        name: Short name
        full_name: Full name
        membrane_type: polymeric, mixed_matrix, facilitated_transport, ceramic
        permeability: Dict mapping species to permeability (Barrer)
        selectivity: Dict mapping pairs to selectivity
        activation_energy: Dict mapping species to Ep (J/mol)
        typical_thickness: Typical thickness (μm)
        max_temperature: Maximum operating temperature (K)
        max_pressure: Maximum pressure (bar)
        cost_usd_m2: Cost (USD/m²)
    """
    name: str
    full_name: str
    membrane_type: str
    permeability: dict[str, float]
    selectivity: dict[str, float]
    activation_energy: dict[str, float]
    typical_thickness: float
    max_temperature: float
    max_pressure: float
    cost_usd_m2: float


class MembraneDatabase:
    """Database of membrane material properties.

    Example:
        >>> db = MembraneDatabase()
        >>> mat = db.get('Matrimid')
        >>> print(f"CO2/N2 selectivity: {mat.selectivity['CO2_N2']}")
    """

    def __init__(self, yaml_path: Path | None = None):
        """Load membrane data from YAML file.

        Args:
            yaml_path: Path to membranes.yaml. If None, uses default.
        """
        if yaml_path is None:
            yaml_path = DATA_DIR / "membranes.yaml"

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        self._membranes: dict[str, Membrane] = {}
        self._robeson: dict[str, dict] = data.get("robeson_2008", {})

        for name, props in data["membranes"].items():
            # Handle permeability (may be nested for facilitated transport)
            if "permeability" in props:
                permeability = props["permeability"]
            elif "permeability_dry" in props:
                permeability = props["permeability_dry"]
            else:
                permeability = {}

            # Handle selectivity
            if "selectivity" in props:
                selectivity = props["selectivity"]
            elif "selectivity_dry" in props:
                selectivity = props["selectivity_dry"]
            else:
                selectivity = {}

            self._membranes[name] = Membrane(
                name=name,
                full_name=props["full_name"],
                membrane_type=props["type"],
                permeability=permeability,
                selectivity=selectivity,
                activation_energy=props.get("activation_energy", {}),
                typical_thickness=props["typical_thickness"],
                max_temperature=props["max_temperature"],
                max_pressure=props["max_pressure"],
                cost_usd_m2=props["cost_usd_m2"],
            )

    def get(self, name: str) -> Membrane:
        """Get membrane by name."""
        if name not in self._membranes:
            raise KeyError(
                f"Unknown membrane: {name}. "
                f"Available: {list(self._membranes.keys())}"
            )
        return self._membranes[name]

    def __getitem__(self, name: str) -> Membrane:
        return self.get(name)

    def list_membranes(self) -> list[str]:
        """List all available membrane names."""
        return list(self._membranes.keys())

    def list_by_type(self, membrane_type: str) -> list[str]:
        """List membranes by type."""
        return [
            name for name, m in self._membranes.items()
            if m.membrane_type == membrane_type
        ]

    def robeson_upper_bound(
        self,
        pair: str,
        selectivity: float | Array
    ) -> Array:
        """Calculate Robeson 2008 upper bound permeability.

        Args:
            pair: Gas pair (e.g., 'CO2_N2', 'CO2_CH4')
            selectivity: Selectivity value(s)

        Returns:
            Maximum permeability (Barrer) at upper bound

        References:
            Robeson LM (2008). J Membr Sci 320:390
        """
        if pair not in self._robeson:
            raise KeyError(f"No upper bound data for pair: {pair}")
        k = self._robeson[pair]["k"]
        n = self._robeson[pair]["n"]
        selectivity = jnp.asarray(selectivity)
        log_P = k - n * jnp.log10(selectivity)
        return jnp.power(10.0, log_P)


# =============================================================================
# Global Database Instances (lazy loaded)
# =============================================================================

_solvent_db: SolventDatabase | None = None
_adsorbent_db: AdsorbentDatabase | None = None
_membrane_db: MembraneDatabase | None = None


def get_solvent_database() -> SolventDatabase:
    """Get the solvent database (singleton)."""
    global _solvent_db
    if _solvent_db is None:
        _solvent_db = SolventDatabase()
    return _solvent_db


def get_adsorbent_database() -> AdsorbentDatabase:
    """Get the adsorbent database (singleton)."""
    global _adsorbent_db
    if _adsorbent_db is None:
        _adsorbent_db = AdsorbentDatabase()
    return _adsorbent_db


def get_membrane_database() -> MembraneDatabase:
    """Get the membrane database (singleton)."""
    global _membrane_db
    if _membrane_db is None:
        _membrane_db = MembraneDatabase()
    return _membrane_db


# =============================================================================
# Convenience Functions
# =============================================================================

def get_solvent(name: str) -> AmineSolvent:
    """Get amine solvent properties.

    Args:
        name: Solvent name (e.g., 'MEA', 'PZ', 'MDEA')

    Returns:
        AmineSolvent dataclass with all properties

    Example:
        >>> mea = get_solvent('MEA')
        >>> print(f"Heat of absorption: {mea.heat_of_absorption} kJ/mol")
    """
    return get_solvent_database().get(name)


def get_adsorbent(name: str) -> Adsorbent:
    """Get adsorbent material properties.

    Args:
        name: Adsorbent name (e.g., 'Zeolite_13X', 'Mg_MOF_74')

    Returns:
        Adsorbent dataclass with all properties

    Example:
        >>> zeolite = get_adsorbent('Zeolite_13X')
        >>> print(f"CO2 capacity: {zeolite.CO2_capacity} mol/kg")
    """
    return get_adsorbent_database().get(name)


def get_membrane(name: str) -> Membrane:
    """Get membrane material properties.

    Args:
        name: Membrane name (e.g., 'Matrimid', 'ZIF8_Matrimid')

    Returns:
        Membrane dataclass with all properties

    Example:
        >>> mem = get_membrane('Matrimid')
        >>> print(f"CO2 permeability: {mem.permeability['CO2']} Barrer")
    """
    return get_membrane_database().get(name)


def list_solvents() -> list[str]:
    """List all available amine solvents."""
    return get_solvent_database().list_solvents()


def list_adsorbents() -> list[str]:
    """List all available adsorbent materials."""
    return get_adsorbent_database().list_adsorbents()


def list_membranes() -> list[str]:
    """List all available membrane materials."""
    return get_membrane_database().list_membranes()


# =============================================================================
# JAX-Compatible Data Accessors
# =============================================================================

def get_solvent_kinetic_params(name: str) -> tuple[Array, Array]:
    """Get kinetic parameters as JAX arrays.

    Args:
        name: Solvent name

    Returns:
        (A, Ea) tuple where A is pre-exponential (L/(mol*s))
        and Ea is activation energy (J/mol)
    """
    solvent = get_solvent(name)
    A = jnp.array(solvent.kinetics["A"])
    Ea = jnp.array(solvent.kinetics["Ea"])
    return A, Ea


def get_isotherm_params_array(
    adsorbent_name: str,
    species: str = "CO2"
) -> dict[str, Array]:
    """Get isotherm parameters as JAX arrays.

    Args:
        adsorbent_name: Adsorbent name
        species: Species name (default 'CO2')

    Returns:
        Dict with parameter names as keys and JAX arrays as values
    """
    adsorbent = get_adsorbent(adsorbent_name)
    if species not in adsorbent.isotherms:
        raise KeyError(f"No isotherm data for {species} on {adsorbent_name}")

    iso = adsorbent.isotherms[species]
    return {k: jnp.array(v) for k, v in iso.params.items()}
