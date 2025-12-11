"""Thermodynamic property database for common species.

This module provides pre-defined thermodynamic properties for common
gases and liquids, eliminating the need to manually specify properties.

Usage:
    from difflow.database import get_species, get_critical_props, SPECIES_DB

    # Get SpeciesData for ideal thermo
    methanol = get_species("methanol")
    thermo = IdealThermo({"methanol": methanol, "water": get_species("water")})

    # Get CriticalProperties for EOS
    methane = get_critical_props("methane")
    eos = PengRobinson({"methane": methane, "ethane": get_critical_props("ethane")})

Data sources:
    - NIST Chemistry WebBook
    - Perry's Chemical Engineers' Handbook (9th ed.)
    - Yaws' Critical Property Data for Chemical Engineers and Chemists
    - DIPPR 801 Database
"""

from difflow.thermo import SpeciesData
from difflow.eos import CriticalProperties


# =============================================================================
# Critical Properties Database
# =============================================================================

# Format: name -> (Tc [K], Pc [Pa], omega, MW [g/mol])
_CRITICAL_DATA = {
    # Light gases
    "hydrogen": (33.19, 1.313e6, -0.216, 2.016),
    "helium": (5.19, 0.227e6, -0.390, 4.003),
    "nitrogen": (126.2, 3.394e6, 0.039, 28.01),
    "oxygen": (154.6, 5.043e6, 0.022, 32.00),
    "carbon_monoxide": (132.9, 3.499e6, 0.066, 28.01),
    "carbon_dioxide": (304.2, 7.383e6, 0.224, 44.01),
    "hydrogen_sulfide": (373.5, 8.963e6, 0.090, 34.08),
    "ammonia": (405.4, 11.35e6, 0.253, 17.03),
    "sulfur_dioxide": (430.8, 7.884e6, 0.245, 64.06),

    # Alkanes (C1-C10)
    "methane": (190.6, 4.599e6, 0.011, 16.04),
    "ethane": (305.3, 4.872e6, 0.099, 30.07),
    "propane": (369.8, 4.248e6, 0.152, 44.10),
    "n_butane": (425.1, 3.796e6, 0.200, 58.12),
    "isobutane": (407.8, 3.640e6, 0.181, 58.12),
    "n_pentane": (469.7, 3.370e6, 0.252, 72.15),
    "isopentane": (460.4, 3.380e6, 0.229, 72.15),
    "neopentane": (433.8, 3.196e6, 0.197, 72.15),
    "n_hexane": (507.6, 3.025e6, 0.301, 86.18),
    "n_heptane": (540.2, 2.740e6, 0.350, 100.20),
    "n_octane": (568.7, 2.490e6, 0.399, 114.23),
    "n_nonane": (594.6, 2.290e6, 0.443, 128.26),
    "n_decane": (617.7, 2.110e6, 0.492, 142.28),

    # Alkenes
    "ethylene": (282.3, 5.041e6, 0.087, 28.05),
    "propylene": (365.6, 4.600e6, 0.140, 42.08),
    "1_butene": (419.5, 4.023e6, 0.191, 56.11),

    # Alkynes
    "acetylene": (308.3, 6.114e6, 0.190, 26.04),

    # Aromatics
    "benzene": (562.0, 4.895e6, 0.210, 78.11),
    "toluene": (591.8, 4.109e6, 0.264, 92.14),
    "ethylbenzene": (617.2, 3.609e6, 0.303, 106.17),
    "o_xylene": (630.3, 3.732e6, 0.310, 106.17),
    "m_xylene": (617.0, 3.541e6, 0.326, 106.17),
    "p_xylene": (616.2, 3.511e6, 0.322, 106.17),
    "styrene": (636.0, 3.840e6, 0.297, 104.15),

    # Alcohols
    "methanol": (512.6, 8.097e6, 0.565, 32.04),
    "ethanol": (513.9, 6.148e6, 0.645, 46.07),
    "1_propanol": (536.8, 5.175e6, 0.629, 60.10),
    "2_propanol": (508.3, 4.762e6, 0.668, 60.10),
    "1_butanol": (563.1, 4.423e6, 0.590, 74.12),

    # Ketones
    "acetone": (508.2, 4.701e6, 0.307, 58.08),
    "methyl_ethyl_ketone": (535.5, 4.150e6, 0.323, 72.11),

    # Aldehydes
    "formaldehyde": (408.0, 6.590e6, 0.253, 30.03),
    "acetaldehyde": (466.0, 5.570e6, 0.291, 44.05),

    # Ethers
    "dimethyl_ether": (400.1, 5.370e6, 0.200, 46.07),
    "diethyl_ether": (466.7, 3.640e6, 0.281, 74.12),
    "tetrahydrofuran": (540.2, 5.190e6, 0.225, 72.11),

    # Esters
    "methyl_acetate": (506.6, 4.750e6, 0.331, 74.08),
    "ethyl_acetate": (523.3, 3.880e6, 0.366, 88.11),

    # Acids
    "formic_acid": (588.0, 5.810e6, 0.473, 46.03),
    "acetic_acid": (592.0, 5.786e6, 0.467, 60.05),

    # Halogenated
    "chloromethane": (416.3, 6.679e6, 0.153, 50.49),
    "dichloromethane": (510.0, 6.080e6, 0.199, 84.93),
    "chloroform": (536.4, 5.472e6, 0.222, 119.38),
    "carbon_tetrachloride": (556.4, 4.560e6, 0.193, 153.82),

    # Water and common solvents
    "water": (647.1, 22.064e6, 0.344, 18.015),
    "heavy_water": (643.9, 21.671e6, 0.364, 20.03),
}


def get_critical_props(name: str) -> CriticalProperties:
    """Get critical properties for a species by name.

    Args:
        name: Species name (case-insensitive, underscores for spaces)

    Returns:
        CriticalProperties namedtuple

    Raises:
        KeyError: If species not found in database

    Example:
        >>> props = get_critical_props("methane")
        >>> props.Tc
        190.6
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _CRITICAL_DATA:
        available = ", ".join(sorted(_CRITICAL_DATA.keys()))
        raise KeyError(
            f"Species '{name}' not in database. "
            f"Available: {available}"
        )

    Tc, Pc, omega, MW = _CRITICAL_DATA[key]
    return CriticalProperties(
        name=key,
        Tc=Tc,
        Pc=Pc,
        omega=omega,
        MW=MW,
    )


# =============================================================================
# Ideal Thermodynamic Properties Database
# =============================================================================

# Format: name -> {
#     "MW": molecular weight (g/mol),
#     "Cp": (a, b, c, d) for Cp = a + bT + cT² + dT³ (J/mol/K),
#     "Hvap": (A, n, Tc) for Hvap = A*(1-T/Tc)^n (J/mol),
#     "antoine": (A, B, C) for log10(P/Pa) = A - B/(T+C),
#     "Hf": standard heat of formation (J/mol) at 298.15 K,
# }

_IDEAL_THERMO_DATA = {
    # Light gases
    "nitrogen": {
        "MW": 28.01,
        "Cp": (29.0, 0.0, 0.0, 0.0),  # Nearly constant
        "Hvap": (5577.0, 0.38, 126.2),
        "antoine": (8.61, 255.68, -6.6),
        "Hf": 0.0,
    },
    "oxygen": {
        "MW": 32.00,
        "Cp": (29.4, 0.0, 0.0, 0.0),
        "Hvap": (6820.0, 0.38, 154.6),
        "antoine": (8.68, 319.01, -6.45),
        "Hf": 0.0,
    },
    "carbon_dioxide": {
        "MW": 44.01,
        "Cp": (27.0, 0.042, -1.5e-5, 0.0),
        "Hvap": (16700.0, 0.38, 304.2),  # Sublimes at 1 atm
        "antoine": (9.81, 1347.79, -35.52),
        "Hf": -393510.0,
    },
    "ammonia": {
        "MW": 17.03,
        "Cp": (35.0, 0.0, 0.0, 0.0),
        "Hvap": (23350.0, 0.38, 405.4),
        "antoine": (10.20, 1596.49, -28.16),
        "Hf": -45940.0,
    },

    # Alkanes
    "methane": {
        "MW": 16.04,
        "Cp": (35.3, 0.0, 0.0, 0.0),
        "Hvap": (8180.0, 0.38, 190.6),
        "antoine": (8.68, 405.42, -26.09),
        "Hf": -74870.0,
    },
    "ethane": {
        "MW": 30.07,
        "Cp": (52.5, 0.0, 0.0, 0.0),
        "Hvap": (14690.0, 0.38, 305.3),
        "antoine": (9.04, 663.70, -16.47),
        "Hf": -84000.0,
    },
    "propane": {
        "MW": 44.10,
        "Cp": (73.5, 0.0, 0.0, 0.0),
        "Hvap": (19040.0, 0.38, 369.8),
        "antoine": (9.10, 803.81, -26.11),
        "Hf": -104700.0,
    },
    "n_butane": {
        "MW": 58.12,
        "Cp": (97.5, 0.0, 0.0, 0.0),
        "Hvap": (22390.0, 0.38, 425.1),
        "antoine": (9.05, 935.86, -34.42),
        "Hf": -125600.0,
    },
    "n_pentane": {
        "MW": 72.15,
        "Cp": (120.0, 0.0, 0.0, 0.0),
        "Hvap": (25790.0, 0.38, 469.7),
        "antoine": (9.02, 1075.78, -40.45),
        "Hf": -146800.0,
    },
    "n_hexane": {
        "MW": 86.18,
        "Cp": (143.0, 0.0, 0.0, 0.0),
        "Hvap": (28850.0, 0.38, 507.6),
        "antoine": (9.00, 1171.17, -48.78),
        "Hf": -167200.0,
    },
    "n_heptane": {
        "MW": 100.20,
        "Cp": (166.0, 0.0, 0.0, 0.0),
        "Hvap": (31770.0, 0.38, 540.2),
        "antoine": (9.02, 1264.90, -56.25),
        "Hf": -187800.0,
    },
    "n_octane": {
        "MW": 114.23,
        "Cp": (189.0, 0.0, 0.0, 0.0),
        "Hvap": (34410.0, 0.38, 568.7),
        "antoine": (9.02, 1351.99, -63.63),
        "Hf": -208600.0,
    },

    # Alkenes
    "ethylene": {
        "MW": 28.05,
        "Cp": (43.0, 0.0, 0.0, 0.0),
        "Hvap": (13540.0, 0.38, 282.3),
        "antoine": (9.08, 595.42, -15.09),
        "Hf": 52470.0,
    },
    "propylene": {
        "MW": 42.08,
        "Cp": (64.0, 0.0, 0.0, 0.0),
        "Hvap": (18420.0, 0.38, 365.6),
        "antoine": (9.10, 786.00, -25.52),
        "Hf": 20410.0,
    },

    # Aromatics
    "benzene": {
        "MW": 78.11,
        "Cp": (136.0, 0.0, 0.0, 0.0),
        "Hvap": (30720.0, 0.38, 562.0),
        "antoine": (9.11, 1211.03, -52.36),
        "Hf": 82880.0,
    },
    "toluene": {
        "MW": 92.14,
        "Cp": (157.0, 0.0, 0.0, 0.0),
        "Hvap": (33180.0, 0.38, 591.8),
        "antoine": (9.08, 1342.31, -53.67),
        "Hf": 50170.0,
    },
    "ethylbenzene": {
        "MW": 106.17,
        "Cp": (183.0, 0.0, 0.0, 0.0),
        "Hvap": (35570.0, 0.38, 617.2),
        "antoine": (9.08, 1421.91, -59.95),
        "Hf": 29790.0,
    },
    "styrene": {
        "MW": 104.15,
        "Cp": (182.0, 0.0, 0.0, 0.0),
        "Hvap": (36820.0, 0.38, 636.0),
        "antoine": (9.10, 1420.00, -60.00),
        "Hf": 147360.0,
    },

    # Alcohols
    "methanol": {
        "MW": 32.04,
        "Cp": (81.0, 0.0, 0.0, 0.0),
        "Hvap": (35210.0, 0.38, 512.6),
        "antoine": (10.20, 1582.27, -33.45),
        "Hf": -201200.0,
    },
    "ethanol": {
        "MW": 46.07,
        "Cp": (112.0, 0.0, 0.0, 0.0),
        "Hvap": (38560.0, 0.38, 513.9),
        "antoine": (10.32, 1718.10, -39.73),
        "Hf": -234800.0,
    },
    "1_propanol": {
        "MW": 60.10,
        "Cp": (144.0, 0.0, 0.0, 0.0),
        "Hvap": (41440.0, 0.38, 536.8),
        "antoine": (10.24, 1796.27, -48.25),
        "Hf": -255200.0,
    },
    "2_propanol": {
        "MW": 60.10,
        "Cp": (155.0, 0.0, 0.0, 0.0),
        "Hvap": (39850.0, 0.38, 508.3),
        "antoine": (10.16, 1664.17, -50.88),
        "Hf": -272700.0,
    },
    "1_butanol": {
        "MW": 74.12,
        "Cp": (177.0, 0.0, 0.0, 0.0),
        "Hvap": (43290.0, 0.38, 563.1),
        "antoine": (9.97, 1778.02, -59.08),
        "Hf": -274600.0,
    },

    # Ketones
    "acetone": {
        "MW": 58.08,
        "Cp": (125.0, 0.0, 0.0, 0.0),
        "Hvap": (29100.0, 0.38, 508.2),
        "antoine": (9.39, 1312.25, -32.52),
        "Hf": -217100.0,
    },

    # Ethers
    "dimethyl_ether": {
        "MW": 46.07,
        "Cp": (65.0, 0.0, 0.0, 0.0),
        "Hvap": (21510.0, 0.38, 400.1),
        "antoine": (9.21, 987.31, -25.18),
        "Hf": -184100.0,
    },
    "diethyl_ether": {
        "MW": 74.12,
        "Cp": (172.0, 0.0, 0.0, 0.0),
        "Hvap": (26520.0, 0.38, 466.7),
        "antoine": (9.12, 1098.20, -38.00),
        "Hf": -252100.0,
    },

    # Acids
    "formic_acid": {
        "MW": 46.03,
        "Cp": (99.0, 0.0, 0.0, 0.0),
        "Hvap": (22690.0, 0.38, 588.0),
        "antoine": (9.37, 1563.28, -42.15),
        "Hf": -378600.0,
    },
    "acetic_acid": {
        "MW": 60.05,
        "Cp": (124.0, 0.0, 0.0, 0.0),
        "Hvap": (23700.0, 0.38, 592.0),
        "antoine": (9.68, 1642.54, -39.76),
        "Hf": -432800.0,
    },

    # Water
    "water": {
        "MW": 18.015,
        "Cp": (75.3, 0.0, 0.0, 0.0),  # Liquid at 25°C
        "Hvap": (40660.0, 0.38, 647.1),
        "antoine": (10.20, 1730.63, -39.72),
        "Hf": -241826.0,
    },

    # Halogenated
    "chloroform": {
        "MW": 119.38,
        "Cp": (114.0, 0.0, 0.0, 0.0),
        "Hvap": (29240.0, 0.38, 536.4),
        "antoine": (9.08, 1163.03, -46.38),
        "Hf": -134100.0,
    },
    "carbon_tetrachloride": {
        "MW": 153.82,
        "Cp": (131.0, 0.0, 0.0, 0.0),
        "Hvap": (29820.0, 0.38, 556.4),
        "antoine": (9.02, 1212.02, -45.19),
        "Hf": -128200.0,
    },
}


def get_species(name: str) -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name.

    Args:
        name: Species name (case-insensitive, underscores for spaces)

    Returns:
        SpeciesData namedtuple

    Raises:
        KeyError: If species not found in database

    Example:
        >>> data = get_species("methanol")
        >>> data.MW
        32.04
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    if key not in _IDEAL_THERMO_DATA:
        available = ", ".join(sorted(_IDEAL_THERMO_DATA.keys()))
        raise KeyError(
            f"Species '{name}' not in database. "
            f"Available: {available}"
        )

    d = _IDEAL_THERMO_DATA[key]
    return SpeciesData(
        name=key,
        MW=d["MW"],
        Cp_coeffs=d["Cp"],
        Hvap_coeffs=d["Hvap"],
        antoine_coeffs=d["antoine"],
        Hf=d.get("Hf", 0.0),
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def list_species() -> list[str]:
    """List all available species in the database.

    Returns:
        Sorted list of species names
    """
    all_species = set(_CRITICAL_DATA.keys()) | set(_IDEAL_THERMO_DATA.keys())
    return sorted(all_species)


def get_species_info(name: str) -> dict:
    """Get all available properties for a species.

    Args:
        name: Species name

    Returns:
        Dictionary with all available properties
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    info = {"name": key}

    if key in _CRITICAL_DATA:
        Tc, Pc, omega, MW = _CRITICAL_DATA[key]
        info["critical"] = {
            "Tc": Tc,
            "Pc": Pc,
            "omega": omega,
            "MW": MW,
        }

    if key in _IDEAL_THERMO_DATA:
        info["ideal_thermo"] = _IDEAL_THERMO_DATA[key]

    if not info.get("critical") and not info.get("ideal_thermo"):
        raise KeyError(f"Species '{name}' not found in database")

    return info


# =============================================================================
# Group Retrieval (for easy multi-component setup)
# =============================================================================


def get_alkanes(n_carbon_max: int = 8) -> dict[str, CriticalProperties]:
    """Get critical properties for n-alkanes up to specified carbon number.

    Args:
        n_carbon_max: Maximum carbon number (default 8 = octane)

    Returns:
        Dictionary of species name -> CriticalProperties
    """
    alkanes = [
        "methane", "ethane", "propane", "n_butane", "n_pentane",
        "n_hexane", "n_heptane", "n_octane", "n_nonane", "n_decane",
    ]
    return {
        name: get_critical_props(name)
        for name in alkanes[:n_carbon_max]
        if name in _CRITICAL_DATA
    }


def get_btex() -> dict[str, CriticalProperties]:
    """Get critical properties for BTEX aromatics.

    Returns:
        Dictionary with benzene, toluene, ethylbenzene, xylenes
    """
    names = ["benzene", "toluene", "ethylbenzene", "o_xylene", "m_xylene", "p_xylene"]
    return {name: get_critical_props(name) for name in names}


def get_common_solvents() -> dict[str, SpeciesData]:
    """Get SpeciesData for common laboratory solvents.

    Returns:
        Dictionary with water, methanol, ethanol, acetone, etc.
    """
    names = [
        "water", "methanol", "ethanol", "acetone",
        "diethyl_ether", "chloroform", "benzene", "toluene",
    ]
    return {name: get_species(name) for name in names if name in _IDEAL_THERMO_DATA}


# =============================================================================
# Aliases for common naming variations
# =============================================================================

_ALIASES = {
    "butane": "n_butane",
    "pentane": "n_pentane",
    "hexane": "n_hexane",
    "heptane": "n_heptane",
    "octane": "n_octane",
    "nonane": "n_nonane",
    "decane": "n_decane",
    "isopropanol": "2_propanol",
    "ipa": "2_propanol",
    "mek": "methyl_ethyl_ketone",
    "thf": "tetrahydrofuran",
    "dcm": "dichloromethane",
    "co2": "carbon_dioxide",
    "co": "carbon_monoxide",
    "h2s": "hydrogen_sulfide",
    "nh3": "ammonia",
    "so2": "sulfur_dioxide",
    "h2o": "water",
    "meoh": "methanol",
    "etoh": "ethanol",
    "etbe": "diethyl_ether",
    "xylene": "m_xylene",  # Default to m-xylene
}


def resolve_alias(name: str) -> str:
    """Resolve common aliases to canonical names.

    Args:
        name: Species name or alias

    Returns:
        Canonical species name
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


# Make alias resolution automatic in get_* functions
_original_get_critical_props = get_critical_props
_original_get_species = get_species


def get_critical_props(name: str) -> CriticalProperties:
    """Get critical properties for a species by name (with alias support)."""
    return _original_get_critical_props(resolve_alias(name))


def get_species(name: str) -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name (with alias support)."""
    return _original_get_species(resolve_alias(name))
