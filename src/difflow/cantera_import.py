"""Cantera data import utilities for difflow.

This module provides functions to import thermodynamic and kinetic data
from Cantera YAML files into difflow-compatible formats.

Features:
- Import species thermodynamic data (NASA polynomials → SpeciesData)
- Import critical properties for equations of state
- Import reaction mechanisms with Arrhenius kinetics
- Works with or without Cantera installed (YAML parsing fallback)

Usage:
    from difflow.cantera_import import (
        import_species_data,
        import_critical_props,
        import_reactions,
        load_mechanism,
    )

    # Load a mechanism file
    species, reactions = load_mechanism('gri30.yaml')

    # Or import specific species
    thermo_data = import_species_data('gri30.yaml', ['CH4', 'O2', 'CO2', 'H2O'])
    thermo = IdealThermo(thermo_data)

Limitations:
    - NASA9 polynomials are approximated to 4th order for difflow compatibility
    - Temperature ranges are simplified to single range (uses high-T coefficients)
    - Some Cantera thermo models (Shomate, etc.) not yet supported
    - Binary interaction parameters must be added manually for EOS
"""

from typing import Any
from pathlib import Path
import warnings

import jax.numpy as jnp

from difflow.thermo import SpeciesData
from difflow.eos import CriticalProperties


# Gas constant
R = 8.314462  # J/(mol·K)


# =============================================================================
# YAML Loading (works without Cantera)
# =============================================================================


def _load_yaml(filepath: str | Path) -> dict:
    """Load a YAML file.

    Uses PyYAML if available, otherwise tries ruamel.yaml.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed YAML content as dictionary
    """
    try:
        import yaml
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        try:
            from ruamel.yaml import YAML
            yaml = YAML()
            with open(filepath, 'r') as f:
                return dict(yaml.load(f))
        except ImportError:
            raise ImportError(
                "Either PyYAML or ruamel.yaml is required. "
                "Install with: pip install pyyaml"
            )


def _find_cantera_file(filename: str) -> Path:
    """Find a Cantera data file.

    Searches in order:
    1. Current directory
    2. Cantera data directory (if Cantera installed)
    3. Common Cantera data paths

    Args:
        filename: Name of file to find

    Returns:
        Path to file

    Raises:
        FileNotFoundError: If file not found
    """
    # Check current directory first
    if Path(filename).exists():
        return Path(filename)

    # Try Cantera's data directory
    try:
        import cantera as ct
        cantera_data = Path(ct.__file__).parent / 'data'
        if (cantera_data / filename).exists():
            return cantera_data / filename
    except ImportError:
        pass

    # Common locations
    common_paths = [
        Path.home() / '.cantera' / 'data',
        Path('/usr/local/share/cantera/data'),
        Path('/usr/share/cantera/data'),
    ]

    for path in common_paths:
        if (path / filename).exists():
            return path / filename

    raise FileNotFoundError(
        f"Could not find '{filename}'. "
        f"Provide full path or install Cantera: pip install cantera"
    )


# =============================================================================
# NASA Polynomial Conversion
# =============================================================================


def nasa7_to_cp_coeffs(
    coeffs_low: list[float],
    coeffs_high: list[float],
    T_mid: float = 1000.0,
    use_high_T: bool = True,
) -> tuple[float, float, float, float]:
    """Convert NASA7 polynomial coefficients to difflow Cp coefficients.

    NASA7 format: Cp/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4

    difflow format: Cp = a + b*T + c*T^2 + d*T^3 (J/mol·K)

    Args:
        coeffs_low: Low temperature NASA7 coefficients [a1-a7]
        coeffs_high: High temperature NASA7 coefficients [a1-a7]
        T_mid: Midpoint temperature for range selection
        use_high_T: If True, use high-T coefficients (better for typical process temps)

    Returns:
        Tuple of (a, b, c, d) for Cp = a + bT + cT² + dT³
    """
    coeffs = coeffs_high if use_high_T else coeffs_low

    # Convert from Cp/R to Cp (J/mol·K)
    a = coeffs[0] * R
    b = coeffs[1] * R
    c = coeffs[2] * R
    d = coeffs[3] * R
    # Note: We drop the T^4 term (coeffs[4]) for 4th-order approximation

    return (a, b, c, d)


def nasa7_to_enthalpy_coeffs(
    coeffs_low: list[float],
    coeffs_high: list[float],
    use_high_T: bool = True,
) -> float:
    """Extract heat of formation from NASA7 coefficients.

    NASA7: H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T

    At T=298.15 K, H = Hf (formation enthalpy)

    Args:
        coeffs_low: Low temperature NASA7 coefficients
        coeffs_high: High temperature NASA7 coefficients
        use_high_T: Which range to use

    Returns:
        Heat of formation at 298 K (J/mol)
    """
    coeffs = coeffs_high if use_high_T else coeffs_low
    T = 298.15

    # H/RT = a1 + a2*T/2 + a3*T^2/3 + a4*T^3/4 + a5*T^4/5 + a6/T
    H_RT = (
        coeffs[0] +
        coeffs[1] * T / 2 +
        coeffs[2] * T**2 / 3 +
        coeffs[3] * T**3 / 4 +
        coeffs[4] * T**4 / 5 +
        coeffs[5] / T
    )

    return H_RT * R * T


def nasa9_to_cp_coeffs(
    data: list[list[float]],
    T_target: float = 500.0,
) -> tuple[float, float, float, float]:
    """Convert NASA9 polynomial to difflow Cp coefficients.

    NASA9 format: Cp/R = a1/T^2 + a2/T + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4

    For typical process temperatures (300-600 K), we fit to standard polynomial.

    Args:
        data: List of [T_low, T_high, a1, a2, ..., a9] for each range
        T_target: Target temperature for coefficient selection

    Returns:
        Tuple of (a, b, c, d) for Cp = a + bT + cT² + dT³
    """
    # Find appropriate temperature range
    for region in data:
        T_low, T_high = region[0], region[1]
        if T_low <= T_target <= T_high:
            coeffs = region[2:]
            break
    else:
        # Use last region if T_target out of range
        coeffs = data[-1][2:]

    # NASA9: Cp/R = a1/T^2 + a2/T + a3 + a4*T + a5*T^2 + a6*T^3 + a7*T^4
    # Approximate as: Cp = a + bT + cT² + dT³
    # Using coefficients at standard conditions
    a = coeffs[2] * R  # constant term
    b = coeffs[3] * R  # T term
    c = coeffs[4] * R  # T^2 term
    d = coeffs[5] * R  # T^3 term

    return (a, b, c, d)


# =============================================================================
# Species Data Import
# =============================================================================


def _parse_species_thermo(species_entry: dict) -> dict:
    """Parse thermodynamic data from a species entry.

    Args:
        species_entry: Species dictionary from YAML

    Returns:
        Dictionary with thermo data
    """
    thermo = species_entry.get('thermo', {})
    model = thermo.get('model', 'NASA7')

    result = {
        'name': species_entry.get('name', 'unknown'),
        'model': model,
    }

    # Molecular weight from composition
    composition = species_entry.get('composition', {})
    MW = 0.0
    atomic_weights = {
        'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007,
        'S': 32.065, 'Ar': 39.948, 'He': 4.003, 'Ne': 20.180,
        'Cl': 35.453, 'F': 18.998, 'Br': 79.904, 'I': 126.904,
    }
    for element, count in composition.items():
        MW += atomic_weights.get(element, 0.0) * count
    result['MW'] = MW

    if model.upper() in ('NASA7', 'NASA'):
        temp_ranges = thermo.get('temperature-ranges', [300, 1000, 5000])
        data = thermo.get('data', [])

        if len(data) >= 2:
            coeffs_low = data[0]
            coeffs_high = data[1]
        elif len(data) == 1:
            coeffs_low = data[0]
            coeffs_high = data[0]
        else:
            coeffs_low = [3.5, 0, 0, 0, 0, 0, 0]
            coeffs_high = coeffs_low

        result['Cp_coeffs'] = nasa7_to_cp_coeffs(coeffs_low, coeffs_high)
        result['Hf'] = nasa7_to_enthalpy_coeffs(coeffs_low, coeffs_high, use_high_T=False)
        result['T_ranges'] = temp_ranges

    elif model.upper() == 'NASA9':
        data = thermo.get('data', [])
        result['Cp_coeffs'] = nasa9_to_cp_coeffs(data)
        result['Hf'] = 0.0  # Would need more complex extraction

    else:
        warnings.warn(f"Unsupported thermo model '{model}' for {result['name']}")
        result['Cp_coeffs'] = (30.0, 0.0, 0.0, 0.0)  # Default ideal gas
        result['Hf'] = 0.0

    return result


def _estimate_antoine_coeffs(
    name: str,
    Tb: float | None = None,
    Tc: float | None = None,
) -> tuple[float, float, float]:
    """Estimate Antoine coefficients from boiling point or critical properties.

    Uses Clausius-Clapeyron approximation if no data available.

    Args:
        name: Species name (for lookup)
        Tb: Normal boiling point (K)
        Tc: Critical temperature (K)

    Returns:
        Antoine coefficients (A, B, C) for log10(P/Pa) = A - B/(T+C)
    """
    # Default estimates based on molecular characteristics
    if Tb is not None:
        # Rough Antoine estimation from boiling point
        # Assumes Hvap ~ 88*Tb (Trouton's rule) in J/mol
        Hvap = 88.0 * Tb
        B = Hvap / (2.303 * R) * 0.9  # Approximate
        A = 10.0 + jnp.log10(101325.0) + B / (Tb - 43)
        C = -43.0
        return (float(A), float(B), float(C))

    # Very rough defaults
    return (10.0, 3000.0, -50.0)


def _estimate_hvap_coeffs(
    Tb: float | None = None,
    Tc: float | None = None,
) -> tuple[float, float, float]:
    """Estimate Watson correlation coefficients for Hvap.

    Hvap = A * (1 - T/Tc)^n

    Args:
        Tb: Normal boiling point (K)
        Tc: Critical temperature (K)

    Returns:
        Coefficients (A, n, Tc) for Hvap correlation
    """
    if Tc is None:
        Tc = 500.0  # Default

    if Tb is not None:
        # Estimate Hvap at Tb using Trouton's rule
        Hvap_b = 88.0 * Tb
        # Watson correlation: Hvap(T) = Hvap_ref * ((Tc-T)/(Tc-Tref))^n
        n = 0.38  # Typical value
        Tr_b = Tb / Tc
        A = Hvap_b / ((1 - Tr_b) ** n)
        return (float(A), n, float(Tc))

    return (35000.0, 0.38, float(Tc))


def import_species_data(
    filepath: str | Path,
    species_names: list[str] | None = None,
    section: str = 'species',
) -> dict[str, SpeciesData]:
    """Import species thermodynamic data from a Cantera YAML file.

    Converts NASA polynomial data to difflow's SpeciesData format.

    Args:
        filepath: Path to Cantera YAML file
        species_names: List of species to import. If None, imports all.
        section: Section name in YAML file containing species

    Returns:
        Dictionary mapping species names to SpeciesData objects

    Example:
        >>> data = import_species_data('gri30.yaml', ['CH4', 'O2', 'CO2'])
        >>> thermo = IdealThermo(data)
    """
    try:
        filepath = _find_cantera_file(filepath)
    except FileNotFoundError:
        pass  # Will try direct path

    yaml_data = _load_yaml(filepath)

    # Find species section
    if section in yaml_data:
        species_list = yaml_data[section]
    elif 'phases' in yaml_data:
        # Some files have species within phases
        species_list = []
        for phase in yaml_data.get('phases', []):
            if 'species' in phase:
                species_list.extend(phase['species'])
    else:
        species_list = yaml_data.get('species', [])

    result = {}

    for sp in species_list:
        if isinstance(sp, str):
            # Reference to species defined elsewhere
            continue

        name = sp.get('name', '')

        if species_names is not None and name not in species_names:
            continue

        try:
            thermo_data = _parse_species_thermo(sp)

            # Get critical properties if available for Hvap estimation
            crit = sp.get('critical-parameters', {})
            Tc = crit.get('critical-temperature')
            if isinstance(Tc, str):
                Tc = float(Tc.split()[0])

            # Estimate missing parameters
            Tb = sp.get('boiling-point')
            if isinstance(Tb, str):
                Tb = float(Tb.split()[0])

            antoine = _estimate_antoine_coeffs(name, Tb, Tc)
            hvap = _estimate_hvap_coeffs(Tb, Tc)

            result[name] = SpeciesData(
                name=name,
                MW=thermo_data['MW'],
                Cp_coeffs=thermo_data['Cp_coeffs'],
                Hvap_coeffs=hvap,
                antoine_coeffs=antoine,
                Hf=thermo_data['Hf'],
            )

        except Exception as e:
            warnings.warn(f"Could not parse species '{name}': {e}")

    if species_names is not None:
        missing = set(species_names) - set(result.keys())
        if missing:
            warnings.warn(f"Species not found in file: {missing}")

    return result


# =============================================================================
# Critical Properties Import
# =============================================================================


def import_critical_props(
    filepath: str | Path,
    species_names: list[str] | None = None,
) -> dict[str, CriticalProperties]:
    """Import critical properties from a Cantera YAML file.

    Args:
        filepath: Path to Cantera YAML file
        species_names: List of species to import. If None, imports all with crit props.

    Returns:
        Dictionary mapping species names to CriticalProperties objects

    Example:
        >>> props = import_critical_props('nist_fluids.yaml', ['methane', 'ethane'])
        >>> eos = PengRobinson(props)
    """
    try:
        filepath = _find_cantera_file(filepath)
    except FileNotFoundError:
        pass

    yaml_data = _load_yaml(filepath)
    species_list = yaml_data.get('species', [])

    result = {}

    for sp in species_list:
        if isinstance(sp, str):
            continue

        name = sp.get('name', '')

        if species_names is not None and name not in species_names:
            continue

        crit = sp.get('critical-parameters', {})
        if not crit:
            continue

        try:
            # Parse critical temperature
            Tc = crit.get('critical-temperature')
            if isinstance(Tc, str):
                Tc = float(Tc.split()[0])

            # Parse critical pressure
            Pc = crit.get('critical-pressure')
            if isinstance(Pc, str):
                value, unit = Pc.split()
                Pc = float(value)
                if unit.lower() in ('mpa', 'mega-pa'):
                    Pc *= 1e6
                elif unit.lower() in ('bar', 'bars'):
                    Pc *= 1e5
                elif unit.lower() in ('atm', 'atmospheres'):
                    Pc *= 101325
            elif Pc is None:
                continue

            # Parse acentric factor
            omega = crit.get('acentric-factor', 0.0)
            if isinstance(omega, str):
                omega = float(omega)

            # Get molecular weight from composition
            composition = sp.get('composition', {})
            MW = 0.0
            atomic_weights = {
                'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007,
                'S': 32.065, 'Ar': 39.948, 'He': 4.003,
            }
            for element, count in composition.items():
                MW += atomic_weights.get(element, 0.0) * count

            result[name] = CriticalProperties(
                name=name,
                Tc=float(Tc),
                Pc=float(Pc),
                omega=float(omega),
                MW=float(MW) if MW > 0 else 28.0,  # Default to N2 MW
            )

        except Exception as e:
            warnings.warn(f"Could not parse critical props for '{name}': {e}")

    return result


# =============================================================================
# Reaction Kinetics Import
# =============================================================================


def _parse_arrhenius(rate_const: dict) -> dict:
    """Parse Arrhenius rate parameters.

    Args:
        rate_const: Rate constant dictionary from YAML

    Returns:
        Dictionary with A (pre-exponential), Ea (activation energy), n (temp exponent)
    """
    # Standard Arrhenius: k = A * T^n * exp(-Ea/RT)
    A = rate_const.get('A', 1.0)
    if isinstance(A, str):
        A = float(A.split()[0])

    n = rate_const.get('b', 0.0)  # Temperature exponent
    if isinstance(n, str):
        n = float(n)

    Ea = rate_const.get('Ea', 0.0)
    if isinstance(Ea, str):
        value, unit = Ea.split()[0], Ea.split()[1] if len(Ea.split()) > 1 else 'J/mol'
        Ea = float(value)
        if 'cal' in unit.lower():
            Ea *= 4.184  # cal to J
        if 'kcal' in unit.lower():
            Ea *= 4184  # kcal to J
        if 'kj' in unit.lower():
            Ea *= 1000  # kJ to J

    return {'A': float(A), 'Ea': float(Ea), 'n': float(n)}


def import_reactions(
    filepath: str | Path,
    reaction_indices: list[int] | None = None,
) -> list[dict]:
    """Import reaction data from a Cantera YAML file.

    Args:
        filepath: Path to Cantera YAML file
        reaction_indices: Specific reaction indices to import. If None, imports all.

    Returns:
        List of reaction dictionaries with:
        - equation: Reaction equation string
        - reactants: Dict of {species: stoich_coeff}
        - products: Dict of {species: stoich_coeff}
        - rate_params: Dict with A, Ea, n for Arrhenius kinetics
        - reversible: Whether reaction is reversible

    Example:
        >>> rxns = import_reactions('gri30.yaml')
        >>> for rxn in rxns[:5]:
        ...     print(f"{rxn['equation']}: A={rxn['rate_params']['A']:.2e}")
    """
    try:
        filepath = _find_cantera_file(filepath)
    except FileNotFoundError:
        pass

    yaml_data = _load_yaml(filepath)
    reactions_list = yaml_data.get('reactions', [])

    result = []

    for i, rxn in enumerate(reactions_list):
        if reaction_indices is not None and i not in reaction_indices:
            continue

        try:
            equation = rxn.get('equation', '')

            # Parse stoichiometry from equation
            # Format: "2 H2 + O2 <=> 2 H2O" or "CH4 + 2 O2 => CO2 + 2 H2O"
            reversible = '<=>' in equation or '=' in equation.replace('=>', '')

            if '<=>' in equation:
                lhs, rhs = equation.split('<=>')
            elif '=>' in equation:
                lhs, rhs = equation.split('=>')
            elif '=' in equation:
                lhs, rhs = equation.split('=')
            else:
                continue

            def parse_side(side_str):
                """Parse one side of reaction equation."""
                species_dict = {}
                parts = side_str.strip().split('+')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    # Check for stoichiometric coefficient
                    tokens = part.split()
                    if len(tokens) >= 2 and tokens[0].replace('.', '').isdigit():
                        coeff = float(tokens[0])
                        species = tokens[1]
                    else:
                        coeff = 1.0
                        species = tokens[0] if tokens else part
                    species_dict[species] = coeff
                return species_dict

            reactants = parse_side(lhs)
            products = parse_side(rhs)

            # Parse rate parameters
            rate_const = rxn.get('rate-constant', {})
            if rate_const:
                rate_params = _parse_arrhenius(rate_const)
            else:
                rate_params = {'A': 1.0, 'Ea': 0.0, 'n': 0.0}

            result.append({
                'index': i,
                'equation': equation.strip(),
                'reactants': reactants,
                'products': products,
                'rate_params': rate_params,
                'reversible': reversible,
                'type': rxn.get('type', 'elementary'),
            })

        except Exception as e:
            warnings.warn(f"Could not parse reaction {i}: {e}")

    return result


# =============================================================================
# High-Level Loading Functions
# =============================================================================


def load_mechanism(
    filepath: str | Path,
    species_names: list[str] | None = None,
) -> tuple[dict[str, SpeciesData], list[dict]]:
    """Load a complete mechanism from a Cantera YAML file.

    Args:
        filepath: Path to Cantera YAML file
        species_names: Specific species to load. If None, loads all.

    Returns:
        Tuple of (species_data, reactions)

    Example:
        >>> species, rxns = load_mechanism('gri30.yaml')
        >>> thermo = IdealThermo(species)
        >>> print(f"Loaded {len(species)} species and {len(rxns)} reactions")
    """
    species = import_species_data(filepath, species_names)
    reactions = import_reactions(filepath)

    return species, reactions


def list_available_species(filepath: str | Path) -> list[str]:
    """List all species names in a Cantera YAML file.

    Args:
        filepath: Path to Cantera YAML file

    Returns:
        List of species names
    """
    try:
        filepath = _find_cantera_file(filepath)
    except FileNotFoundError:
        pass

    yaml_data = _load_yaml(filepath)
    species_list = yaml_data.get('species', [])

    names = []
    for sp in species_list:
        if isinstance(sp, dict):
            names.append(sp.get('name', ''))
        elif isinstance(sp, str):
            names.append(sp)

    return names


def list_available_reactions(filepath: str | Path) -> list[str]:
    """List all reaction equations in a Cantera YAML file.

    Args:
        filepath: Path to Cantera YAML file

    Returns:
        List of reaction equation strings
    """
    try:
        filepath = _find_cantera_file(filepath)
    except FileNotFoundError:
        pass

    yaml_data = _load_yaml(filepath)
    reactions_list = yaml_data.get('reactions', [])

    equations = []
    for rxn in reactions_list:
        if isinstance(rxn, dict):
            equations.append(rxn.get('equation', ''))

    return equations


# =============================================================================
# Cantera Direct Interface (if installed)
# =============================================================================


def import_from_cantera_solution(
    phase_name: str = 'gri30.yaml',
    species_names: list[str] | None = None,
) -> dict[str, SpeciesData]:
    """Import species data directly from a Cantera Solution object.

    This provides more accurate data than YAML parsing when Cantera is installed.

    Args:
        phase_name: Cantera phase/mechanism name
        species_names: Species to import. If None, imports all.

    Returns:
        Dictionary of SpeciesData objects

    Raises:
        ImportError: If Cantera is not installed
    """
    try:
        import cantera as ct
    except ImportError:
        raise ImportError(
            "Cantera is required for this function. "
            "Install with: pip install cantera"
        )

    gas = ct.Solution(phase_name)

    if species_names is None:
        species_names = gas.species_names

    result = {}

    for name in species_names:
        if name not in gas.species_names:
            warnings.warn(f"Species '{name}' not found in {phase_name}")
            continue

        idx = gas.species_index(name)
        sp = gas.species(idx)

        # Get molecular weight
        MW = sp.molecular_weight

        # Get Cp at multiple temperatures and fit polynomial
        T_points = [300, 400, 500, 600, 800, 1000]
        Cp_points = []
        for T in T_points:
            gas.TPX = T, 101325, f'{name}:1.0'
            Cp_points.append(gas.cp_mole)

        # Fit polynomial: Cp = a + bT + cT^2 + dT^3
        import numpy as np
        T_arr = np.array(T_points)
        Cp_arr = np.array(Cp_points)
        coeffs = np.polyfit(T_arr, Cp_arr, 3)
        Cp_coeffs = (float(coeffs[3]), float(coeffs[2]), float(coeffs[1]), float(coeffs[0]))

        # Get enthalpy of formation at 298 K
        gas.TPX = 298.15, 101325, f'{name}:1.0'
        Hf = gas.enthalpy_mole

        # Estimate Antoine and Hvap (not directly available from Cantera gas phase)
        antoine = _estimate_antoine_coeffs(name)
        hvap = _estimate_hvap_coeffs()

        result[name] = SpeciesData(
            name=name,
            MW=float(MW),
            Cp_coeffs=Cp_coeffs,
            Hvap_coeffs=hvap,
            antoine_coeffs=antoine,
            Hf=float(Hf),
        )

    return result
