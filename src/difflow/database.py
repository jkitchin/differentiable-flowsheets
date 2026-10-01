"""Thermodynamic property database for common species.

This module provides pre-defined thermodynamic properties for common
gases and liquids, eliminating the need to manually specify properties.

Usage:
    from difflow.database import get_species_data, get_critical_props

    # Get SpeciesData for ideal thermo
    methanol = get_species_data("methanol")
    thermo = IdealThermo({"methanol": methanol, "water": get_species_data("water")})

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
    "cis_2_butene": (435.5, 4.210e6, 0.202, 56.11),
    "trans_2_butene": (428.6, 4.100e6, 0.214, 56.11),
    "isobutylene": (417.9, 4.000e6, 0.194, 56.11),
    "1_pentene": (464.8, 3.560e6, 0.237, 70.13),
    "2_methyl_2_butene": (470.0, 3.420e6, 0.285, 70.13),

    # Naphthenes and branched paraffins (refinery light naphtha / gasoline
    # isomers). Tc, Pc: IUPAC critical-property reviews, as also tabulated
    # in Poling-Prausnitz-O'Connell 5e App. A; omega: PPO / DIPPR
    # consensus. See SOURCE_CITATIONS.
    "cyclohexane": (553.5, 4.073e6, 0.211, 84.16),
    "methylcyclopentane": (532.7, 3.790e6, 0.230, 84.16),
    "2_methylpentane": (497.7, 3.040e6, 0.279, 86.18),
    "3_methylpentane": (504.6, 3.120e6, 0.273, 86.18),
    "2_2_dimethylbutane": (489.0, 3.100e6, 0.232, 86.18),
    "2_3_dimethylbutane": (500.0, 3.150e6, 0.247, 86.18),
    "2_2_4_trimethylpentane": (543.8, 2.570e6, 0.303, 114.23),
    "2_3_4_trimethylpentane": (566.4, 2.730e6, 0.316, 114.23),
    "2_5_dimethylhexane": (550.0, 2.490e6, 0.355, 114.23),

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


# Per-species provenance for the report layer.  Any species without an
# explicit override inherits the default source string below — all built-in
# data is drawn from the sources listed in the module docstring.
_DEFAULT_SOURCE = "NIST Chemistry WebBook; Perry's 9e; Yaws; DIPPR 801"
SOURCE_CITATIONS: dict[str, str] = {
    name: _DEFAULT_SOURCE for name in _CRITICAL_DATA
}

# Refinery C4-C8 olefins, naphthenes and branched paraffins (issues #308-#310).
#
# How these values were obtained and checked: no primary source could be
# opened directly when they were entered (NIST WebBook unreachable). Each
# value was read from the data tables shipped with the `chemicals` Python
# package (C. Bell et al., v1.5.2), which transcribe the sources named below,
# and was accepted only where those independent tables agree (Tc to ~0.5 K,
# Pc to ~1%, omega to ~0.01, Hf to ~1 kJ/mol). Bibliographic details (page,
# table and edition numbers) are therefore marked "(unverified)": the numbers
# are cross-checked, the citations' fine print is not. The tests in
# tests/test_database_refinery_species.py pin every species to these
# references and tie Tc/Pc/omega to the Antoine set through Lee-Kesler.
_REFINERY_ISOMER_SOURCE = (
    "Tc, Pc: IUPAC critical-property review series 'Vapor-Liquid Critical "
    "Properties of Elements and Compounds' (Ambrose, Tsonopoulos et al., "
    "J. Chem. Eng. Data, 1995 onward; part and page numbers unverified), as "
    "also tabulated in Poling, Prausnitz & O'Connell, The Properties of Gases "
    "and Liquids, 5th ed., McGraw-Hill, 2001, App. A; cross-checked against "
    "CRC Handbook and NIST WebBook values. "
    "omega: Poling-Prausnitz-O'Connell 5e App. A / DIPPR 801 consensus, "
    "cross-checked against the PSRK (Horstmann et al. 2005) and Yaws tables. "
    "Ideal-gas Cp: cubic least-squares fit (250-1000 K) by this project to "
    "the TRC Thermodynamic Tables (Thermodynamics Research Center) ideal-gas "
    "correlation; agrees with the PPO 5e App. A tabulated Cp(298.15 K) to "
    "0.2%. Antoine A, B, C and range: PPO 5e App. A (section number "
    "unverified), converted to log10(P/Pa), T in K. "
    "Hvap: CRC Handbook of Chemistry and Physics, 'Enthalpy of Vaporization' "
    "table, dHvap at Tb (edition unverified); Watson A back-solved by this "
    "project. Hf: CRC Handbook, ideal-gas standard enthalpy of formation at "
    "298.15 K (edition unverified)."
)
# Per-species notes, appended to the shared citation above. "(unverified)"
# marks a value the cross-check could not settle.
_REFINERY_ISOMER_NOTES = {
    "cis_2_butene": "",
    "trans_2_butene": (
        " omega = 0.214 (unverified): sources span 0.205-0.218 "
        "(PSRK 0.214, Yaws 0.218, HEOS 0.210)."
    ),
    "isobutylene": (
        " Hvap: no CRC entry; Perry's Chemical Engineers' Handbook Table "
        "2-150 (C1 = 32614 J/mol, C2 = 0.38073, Tc = 417.9 K) evaluated at "
        "Tb = 266.25 K (edition and Tb source unverified). omega = 0.194: "
        "sources span 0.189-0.197."
    ),
    "1_pentene": "",
    "2_methyl_2_butene": (
        " Tc = 470.0 K (unverified): IUPAC, CRC and Yaws give 470.0 K but "
        "the NIST WebBook gives 473.9 K."
    ),
    "cyclohexane": "",
    "methylcyclopentane": (
        " omega = 0.230 (unverified): Yaws 0.230, PSRK 0.239. The chemicals "
        "package's 'IUPAC' rows for cyclohexane and methylcyclopentane are "
        "swapped; Tc and Pc here follow CRC, NIST WebBook, PSRK and Yaws, "
        "which agree."
    ),
    "2_methylpentane": "",
    "3_methylpentane": (
        " Cp: the PPO 5e App. A polynomial for this species does not "
        "reproduce its own tabulated Cp(298 K) (a transcription error in "
        "the source or its digitisation); the TRC fit used here does."
    ),
    "2_2_dimethylbutane": "",
    "2_3_dimethylbutane": "",
    "2_2_4_trimethylpentane": "",
    "2_3_4_trimethylpentane": (
        " Cp: PPO 5e App. A has no polynomial for this species, only "
        "Cp(298 K) = 191.59 J/mol/K, which the TRC fit matches."
    ),
    "2_5_dimethylhexane": (
        " omega = 0.355 (unverified): sources span 0.352-0.358."
    ),
}
for _name, _note in _REFINERY_ISOMER_NOTES.items():
    SOURCE_CITATIONS[_name] = _REFINERY_ISOMER_SOURCE + _note
del _name, _note


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
#         Note: A = A_NIST + 5 because NIST uses bar and we use Pa (1 bar = 1e5 Pa).
#     "Hf": standard heat of formation (J/mol) at 298.15 K,
# }
# Antoine sources: NIST Chemistry WebBook (https://webbook.nist.gov)
#   Hydrocarbons: Williamham, Taylor, et al., 1945
#   Alcohols: Ambrose and Sprake, 1970
#   Acetone: Ambrose, Sprake, et al., 1974
#   Water: Bridgeman and Aldrich, 1964 (304-333 K range)

_IDEAL_THERMO_DATA = {
    # Light gases
    "nitrogen": {
        "MW": 28.01,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A);
        # reproduces the former room-T constant (29.0) to ~1% at 298 K but is
        # accurate over the 50-1000 K span needed for cryogenic duties.
        "Cp": (31.15, -1.357e-2, 2.680e-5, -1.168e-8),
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
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (19.80, 7.344e-2, -5.602e-5, 1.715e-8),
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
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (19.25, 5.213e-2, 1.197e-5, -1.132e-8),
        "Hvap": (8180.0, 0.38, 190.6),
        "antoine": (8.68, 405.42, -26.09),
        "Hf": -74870.0,
    },
    "ethane": {
        "MW": 30.07,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (5.409, 1.781e-1, -6.938e-5, 8.713e-9),
        "Hvap": (14690.0, 0.38, 305.3),
        "antoine": (9.04, 663.70, -16.47),
        "Hf": -84000.0,
    },
    "propane": {
        "MW": 44.10,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (-4.224, 3.063e-1, -1.586e-4, 3.215e-8),
        "Hvap": (19040.0, 0.38, 369.8),
        "antoine": (9.10, 803.81, -26.11),
        "Hf": -104700.0,
    },
    "n_butane": {
        "MW": 58.12,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (9.487, 3.313e-1, -1.108e-4, -2.822e-9),
        "Hvap": (22390.0, 0.38, 425.1),
        "antoine": (9.05, 935.86, -34.42),
        "Hf": -125600.0,
    },
    "n_pentane": {
        "MW": 72.15,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (-3.626, 4.873e-1, -2.580e-4, 5.305e-8),
        "Hvap": (25790.0, 0.38, 469.7),
        "antoine": (9.02, 1075.78, -40.45),
        "Hf": -146800.0,
    },
    "n_hexane": {
        "MW": 86.18,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (-4.413, 5.820e-1, -3.119e-4, 6.494e-8),
        "Hvap": (28850.0, 0.38, 507.6),
        "antoine": (9.00266, 1171.530, -48.784),
        "Hf": -167200.0,
    },
    "n_heptane": {
        "MW": 100.20,
        "Cp": (166.0, 0.0, 0.0, 0.0),
        "Hvap": (31770.0, 0.38, 540.2),
        "antoine": (9.02832, 1268.636, -56.199),
        "Hf": -187800.0,
    },
    "n_octane": {
        "MW": 114.23,
        "Cp": (189.0, 0.0, 0.0, 0.0),
        "Hvap": (34410.0, 0.38, 568.7),
        "antoine": (9.04867, 1355.126, -63.633),
        "Hf": -208600.0,
    },

    # Branched alkanes (NGL isomers). These have critical properties above but
    # previously had no ideal-thermo record, so get_species_data raised
    # KeyError, blocking a full NGL mixture on the ideal-thermo/CubicThermo path.
    "isobutane": {
        "MW": 58.12,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (-1.390, 3.847e-1, -1.846e-4, 2.895e-8),
        # Watson Hvap interpolated between propane and n-butane (approximate).
        "Hvap": (21300.0, 0.38, 407.8),
        # Antoine: NIST WebBook, Das, Reed et al. 1973 (261-408 K); the NIST
        # log10(P/bar) A is shifted +5 for difflow's Pa convention.
        "antoine": (9.3281, 1132.108, 0.918),
        "Hf": -134200.0,
    },
    "isopentane": {
        "MW": 72.15,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp": (-9.525, 5.066e-1, -2.729e-4, 5.723e-8),
        # Watson Hvap interpolated between n-butane and n-pentane (approximate).
        "Hvap": (24700.0, 0.38, 460.4),
        # Antoine: NIST WebBook, Stull 1947 (190-301 K); NIST log10(P/bar) A
        # shifted +5 for difflow's Pa convention.
        "antoine": (8.90935, 1018.516, -40.081),
        "Hf": -153700.0,
    },

    # Refinery C4-C8 olefins, naphthenes and branched paraffins.
    # Unlike several older entries above (whose Watson "A" is simply dHvap at
    # Tb), the Watson A here is back-solved, A = dHvap(Tb) / (1 - Tb/Tc)**0.38,
    # so the correlation reproduces the CRC dHvap at the normal boiling point
    # (and lands within ~2% of the CRC value at 298 K). Cp is a cubic fit to
    # the TRC ideal-gas correlation over 250-1000 K (max deviation < 3%,
    # < 1% at 298 K). Antoine (PPO 5e App. A) is already in the Pa / K form
    # used here; its fitted range is recorded in T_antoine_min/max.
    "cis_2_butene": {
        "MW": 56.11,
        "Cp": (1.7909, 2.9268e-01, -9.6484e-05, -5.1338e-09),
        "Hvap": (34258.0, 0.38, 435.5),
        "antoine": (9.00958, 967.32, -35.277),
        "Hf": -7100.0,
        "T_antoine_min": 204.73,
        "T_antoine_max": 296.11,
    },
    "trans_2_butene": {
        "MW": 56.11,
        "Cp": (17.143, 2.5781e-01, -6.5496e-05, -1.4603e-08),
        "Hvap": (33475.0, 0.38, 428.6),
        "antoine": (9.00827, 967.5, -32.31),
        "Hf": -11400.0,
        "T_antoine_min": 201.83,
        "T_antoine_max": 293.29,
    },
    "isobutylene": {
        "MW": 56.11,
        "Cp": (10.678, 2.9565e-01, -1.2341e-04, 1.2269e-08),
        "Hvap": (32590.0, 0.38, 417.9),
        "antoine": (8.80956, 866.25, -38.51),
        "Hf": -16900.0,
        "T_antoine_min": 195.77,
        "T_antoine_max": 285.41,
    },
    "1_pentene": {
        "MW": 70.13,
        "Cp": (-4.7482e-01, 4.2488e-01, -2.0783e-04, 2.9140e-08),
        "Hvap": (37641.0, 0.38, 464.8),
        "antoine": (8.96914, 1044.01, -39.7),
        "Hf": -21100.0,
        "T_antoine_min": 223.89,
        "T_antoine_max": 324.32,
    },
    "2_methyl_2_butene": {
        "MW": 70.13,
        "Cp": (-5.2007, 4.3102e-01, -2.2737e-04, 4.6403e-08),
        "Hvap": (39786.0, 0.38, 470.0),
        "antoine": (9.09149, 1124.33, -36.52),
        "Hf": -41700.0,
        "T_antoine_min": 230.69,
        "T_antoine_max": 333.14,
    },
    "cyclohexane": {
        "MW": 84.16,
        "Cp": (-45.988, 5.6596e-01, -1.7435e-04, -3.1534e-08),
        "Hvap": (44156.0, 0.38, 553.5),
        "antoine": (8.93002, 1182.774, -52.532),
        "Hf": -123400.0,
        "T_antoine_min": 282.11,
        "T_antoine_max": 378.46,
    },
    "methylcyclopentane": {
        "MW": 84.16,
        "Cp": (-36.699, 5.5399e-01, -2.1029e-04, -4.3959e-09),
        "Hvap": (43221.0, 0.38, 532.7),
        "antoine": (9.18199, 1295.543, -34.76),
        "Hf": -106200.0,
        "T_antoine_min": 255.06,
        "T_antoine_max": 368.58,
    },
    "2_methylpentane": {
        "MW": 86.18,
        "Cp": (-13.136, 6.1350e-01, -3.2823e-04, 6.4361e-08),
        "Hvap": (42345.0, 0.38, 497.7),
        "antoine": (8.98332, 1145.8, -45.335),
        "Hf": -174600.0,
        "T_antoine_min": 246.9,
        "T_antoine_max": 356.5,
    },
    "3_methylpentane": {
        "MW": 86.18,
        "Cp": (-12.441, 5.9692e-01, -3.0058e-04, 5.1476e-08),
        "Hvap": (42600.0, 0.38, 504.6),
        "antoine": (8.99283, 1162.37, -44.864),
        "Hf": -171900.0,
        "T_antoine_min": 249.0,
        "T_antoine_max": 359.72,
    },
    "2_2_dimethylbutane": {
        "MW": 86.18,
        "Cp": (-8.973, 5.7983e-01, -2.6270e-04, 3.9235e-08),
        "Hvap": (39655.0, 0.38, 489.0),
        "antoine": (8.8959, 1090.16, -42.633),
        "Hf": -185900.0,
        "T_antoine_min": 237.4,
        "T_antoine_max": 345.89,
    },
    "2_3_dimethylbutane": {
        "MW": 86.18,
        "Cp": (-20.854, 6.3770e-01, -3.6163e-04, 8.5476e-08),
        "Hvap": (41355.0, 0.38, 500.0),
        "antoine": (8.93486, 1127.4, -44.184),
        "Hf": -178100.0,
        "T_antoine_min": 244.2,
        "T_antoine_max": 354.43,
    },
    "2_2_4_trimethylpentane": {
        "MW": 114.23,
        "Cp": (-23.936, 8.4297e-01, -4.6936e-04, 1.0446e-07),
        "Hvap": (47745.0, 0.38, 543.8),
        "antoine": (8.93646, 1257.85, -52.383),
        "Hf": -224000.0,
        "T_antoine_min": 275.5,
        "T_antoine_max": 398.38,
    },
    "2_3_4_trimethylpentane": {
        "MW": 114.23,
        "Cp": (-30.448, 9.0998e-01, -6.0879e-04, 1.7711e-07),
        "Hvap": (50052.0, 0.38, 566.4),
        "antoine": (8.977, 1314.31, -55.669),
        "Hf": -217300.0,
        "T_antoine_min": 300.19,
        "T_antoine_max": 413.19,
    },
    "2_5_dimethylhexane": {
        "MW": 114.23,
        "Cp": (-30.405, 8.6874e-01, -5.2198e-04, 1.2478e-07),
        "Hvap": (51098.0, 0.38, 550.0),
        "antoine": (8.98112, 1285.47, -58.902),
        "Hf": -222500.0,
        "T_antoine_min": 285.2,
        "T_antoine_max": 408.2,
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
        "antoine": (9.01814, 1203.835, -53.226),
        "Hf": 82880.0,
        "T_antoine_min": 287.7,
        "T_antoine_max": 354.07,
    },
    "toluene": {
        "MW": 92.14,
        "Cp": (157.0, 0.0, 0.0, 0.0),
        "Hvap": (33180.0, 0.38, 591.8),
        "antoine": (9.07827, 1343.943, -53.773),
        "Hf": 50170.0,
    },
    "ethylbenzene": {
        "MW": 106.17,
        "Cp": (183.0, 0.0, 0.0, 0.0),
        "Hvap": (35570.0, 0.38, 617.2),
        "antoine": (9.07488, 1419.315, -60.539),
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
        "antoine": (10.20409, 1581.341, -33.500),
        "Hf": -201200.0,
        "T_antoine_min": 288.1,
        "T_antoine_max": 356.83,
    },
    "ethanol": {
        "MW": 46.07,
        "Cp": (112.0, 0.0, 0.0, 0.0),
        "Hvap": (38560.0, 0.38, 513.9),
        "antoine": (10.24677, 1598.673, -46.424),
        "Hf": -234800.0,
        "T_antoine_min": 292.77,
        "T_antoine_max": 366.63,
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
        "antoine": (9.42448, 1312.253, -32.445),
        "Hf": -217100.0,
        "T_antoine_min": 259.16,
        "T_antoine_max": 507.60,
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
        "Hvap": (52100.0, 0.38, 592.0),  # Watson A calibrated to ~52 kJ/mol at 25C
        "antoine": (9.68, 1642.54, -39.76),
        "Hf": -432800.0,
    },

    # Water
    "water": {
        "MW": 18.015,
        "Cp": (75.3, 0.0, 0.0, 0.0),  # Liquid at 25°C
        "Hvap": (40660.0, 0.38, 647.1),
        "antoine": (10.20389, 1733.926, -39.485),
        "Hf": -285830.0,  # Liquid-phase formation enthalpy (consistent with liquid Cp)
        "T_antoine_min": 273.0,
        "T_antoine_max": 473.0,
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


# Alkylation species (issue #310): the C7 propylene alkylate (2,3- and
# 2,4-dimethylpentane), the C9 amylene alkylate (2,2,5-trimethylhexane), the
# heavy-end property surrogate (n-dodecane) and ideal-gas data for 1-butene,
# whose critical constants were already here. Obtained and checked exactly
# as the refinery isomers above: read from the data tables of the `chemicals`
# package (C. Bell et al., v1.5.2) and accepted where independent tables
# agree; values only one table carries, or on which the tables disagree, are
# marked "(unverified)" in the per-species note. Ideal-gas Cp is the same
# kind of cubic fit (250-1000 K) to the TRC correlation as above (max
# deviation from it 0.04-2.3%; n-dodecane the worst). Antoine: PPO 5e
# App. A, log10(P/Pa), T in K. Hvap: Watson A back-solved from the CRC
# dHvap at the CRC Tb. Hf: CRC ideal-gas value where CRC has one, else the
# API Technical Data Book (Albahri) value. Pinned in
# tests/refinery/test_alkylation.py::TestDatabaseSpecies.
_ALKYLATION_CRITICAL = {
    "2_3_dimethylpentane": (537.3, 2.910e6, 0.297, 100.20),
    "2_4_dimethylpentane": (519.8, 2.740e6, 0.304, 100.20),
    "2_2_5_trimethylhexane": (569.8, 2.330e6, 0.357, 128.26),
    "n_dodecane": (658.0, 1.820e6, 0.576, 170.34),
}
_ALKYLATION_IDEAL = {
    "1_butene": {
        "MW": 56.11,
        "Cp": (2.6419, 3.2085e-01, -1.4671e-04, 1.9026e-08),
        "Hvap": (32410.0, 0.38, 419.5),
        "antoine": (8.9178, 908.8, -34.61),
        "Hf": 100.0,
        "T_antoine_min": 196.41,
        "T_antoine_max": 285.88,
    },
    "2_3_dimethylpentane": {
        "MW": 100.20,
        "Cp": (-35.121, 7.9476e-01, -5.0099e-04, 1.3257e-07),
        "Hvap": (46715.0, 0.38, 537.3),
        "antoine": (8.98066, 1238.986, -51.208),
        "Hf": -198700.0,
        "T_antoine_min": 281.56,
        "T_antoine_max": 387.89,
    },
    "2_4_dimethylpentane": {
        "MW": 100.20,
        "Cp": (-23.801, 7.9644e-01, -5.2659e-04, 1.4802e-07),
        "Hvap": (45580.0, 0.38, 519.8),
        "antoine": (8.95442, 1193.612, -51.343),
        "Hf": -201600.0,
        "T_antoine_min": 262.4,
        "T_antoine_max": 378.01,
    },
    "2_2_5_trimethylhexane": {
        "MW": 128.26,
        "Cp": (-35.917, 9.7760e-01, -5.7667e-04, 1.3805e-07),
        "Hvap": (52981.0, 0.38, 569.8),
        "antoine": (8.97372, 1332.86, -61.34),
        "Hf": -253300.0,
        "T_antoine_min": 296.3,
        "T_antoine_max": 424.25,
    },
    "n_dodecane": {
        "MW": 170.34,
        "Cp": (-15.412, 1.1537, -5.8026e-04, 7.2305e-08),
        "Hvap": (73982.0, 0.38, 658.0),
        "antoine": (9.12285, 1639.27, -91.31),
        "Hf": -289400.0,
        "T_antoine_min": 372.89,
        "T_antoine_max": 520.24,
    },
}
_ALKYLATION_NOTES = {
    "1_butene": (
        " Ideal-gas data added for #310 (critical constants predate it). "
        "Hf = +0.1 kJ/mol (CRC); the tables span -0.5 to +0.1 kJ/mol "
        "(ATcT -0.03, Yaws -0.5)."
    ),
    "2_3_dimethylpentane": (
        " Hf = -198.7 kJ/mol (unverified): CRC -198.7, API TDB and Yaws "
        "-194.1. omega: Yaws 0.296, PSRK 0.299."
    ),
    "2_4_dimethylpentane": " omega: Yaws 0.302, PSRK 0.306.",
    "2_2_5_trimethylhexane": (
        " Pc = 2.330 MPa (unverified): no IUPAC value; Yaws and PSRK agree. "
        "omega = 0.357 (unverified): PSRK 0.357, Yaws 0.345. Hf: API TDB "
        "and Yaws (no CRC value)."
    ),
    "n_dodecane": (
        " omega = 0.576 (unverified): Yaws 0.576, PSRK 0.562. Hf: CRC "
        "-289.4, API TDB -290.8 kJ/mol."
    ),
}
_CRITICAL_DATA.update(_ALKYLATION_CRITICAL)
_IDEAL_THERMO_DATA.update(_ALKYLATION_IDEAL)
for _name, _note in _ALKYLATION_NOTES.items():
    SOURCE_CITATIONS[_name] = _REFINERY_ISOMER_SOURCE + _note
del _name, _note


def get_species_data(name: str) -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name.

    Args:
        name: Species name (case-insensitive, underscores for spaces)

    Returns:
        SpeciesData namedtuple

    Raises:
        KeyError: If species not found in database

    Example:
        >>> data = get_species_data("methanol")
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
        T_antoine_min=d.get("T_antoine_min", 0.0),
        T_antoine_max=d.get("T_antoine_max", 1e6),
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
    _record_access(name, "info")
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
    return {name: get_species_data(name) for name in names if name in _IDEAL_THERMO_DATA}


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
    "isobutene": "isobutylene",
    "2_methylpropene": "isobutylene",
    "isooctane": "2_2_4_trimethylpentane",
    "isohexane": "2_methylpentane",
    "neohexane": "2_2_dimethylbutane",
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


# =============================================================================
# Instrumented database-access tracking (report provenance)
# =============================================================================

import contextlib as _contextlib

#: stack of active access recorders; each is a list of (species, kind) tuples.
#: Using a stack rather than a single log makes ``track_database_access``
#: re-entrant (nested contexts each get their own record).
_active_recorders: list[list[tuple[str, str]]] = []


def _record_access(name: str, kind: str) -> None:
    """Note a database lookup for any active :func:`track_database_access`."""
    if not _active_recorders:
        return
    canonical = resolve_alias(name)
    for recorder in _active_recorders:
        recorder.append((canonical, kind))


class DatabaseAccessTracker:
    """Records which species/properties the database served during a solve.

    Obtain one from :func:`track_database_access`.  After the ``with`` block
    it exposes the set of species whose entries were read, which lets a
    :class:`~difflow.report.ir.Report` distinguish species the calculation
    actually depended on from species merely present in a stream.
    """

    def __init__(self, records: list[tuple[str, str]]):
        self._records = records

    @property
    def records(self) -> list[tuple[str, str]]:
        """All ``(species, kind)`` lookups in call order (with duplicates)."""
        return list(self._records)

    @property
    def species(self) -> list[str]:
        """Canonical species names accessed, in first-seen order."""
        seen: dict[str, None] = {}
        for name, _ in self._records:
            seen.setdefault(name, None)
        return list(seen)

    def was_accessed(self, name: str) -> bool:
        """True if ``name`` (any alias) was looked up during tracking."""
        return resolve_alias(name) in set(n for n, _ in self._records)

    def kinds(self, name: str) -> set[str]:
        """The kinds of lookup ("critical", "ideal", "info") for ``name``."""
        canonical = resolve_alias(name)
        return {k for n, k in self._records if n == canonical}


@_contextlib.contextmanager
def track_database_access():
    """Context manager that records database lookups made inside it.

    Wrap a solve to capture precise data provenance for a report::

        from difflow.database import track_database_access

        with track_database_access() as tracker:
            streams = fs.solve()

        rep = fs.report(streams, db_access=tracker)

    Yields:
        A :class:`DatabaseAccessTracker`; the records are complete once the
        ``with`` block exits (and remain readable afterward).
    """
    records: list[tuple[str, str]] = []
    _active_recorders.append(records)
    try:
        yield DatabaseAccessTracker(records)
    finally:
        _active_recorders.remove(records)


# Make alias resolution automatic in get_* functions
_original_get_critical_props = get_critical_props
_original_get_species_data = get_species_data


def get_critical_props(name: str) -> CriticalProperties:
    """Get critical properties for a species by name (with alias support)."""
    _record_access(name, "critical")
    return _original_get_critical_props(resolve_alias(name))


def get_species_data(name: str) -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name (with alias support)."""
    _record_access(name, "ideal")
    return _original_get_species_data(resolve_alias(name))
