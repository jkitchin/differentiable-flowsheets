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
    # The other three C4 olefins (FCC LPG). Tc, Pc and omega are the
    # constants of Lemmon & Ihmels' reference equations of state, "Thermodynamic
    # properties of the butenes Part II. Short fundamental equations of
    # state", Fluid Phase Equilib. 228-229, 173-187 (2005), as CoolProp
    # tabulates them. NIST WebBook (Tsonopoulos & Ambrose 1996) agrees on Tc
    # to 0.3 K; its Pc differs by up to 2% (cis 42.1, trans 41.0, iso 40.0 bar).
    "cis_2_butene": (435.75, 4.236e6, 0.2024, 56.106),
    "trans_2_butene": (428.61, 4.019e6, 0.2101, 56.106),
    "isobutylene": (418.09, 4.016e6, 0.1926, 56.106),

    # C6 isomers and naphthenes of light-naphtha isomerization (issue #311).
    # Tc, Pc and omega are the PSRK Revision 4 appendix values (Horstmann,
    # Jaborowski, Fischer & Gmehling, Fluid Phase Equilib. 227, 157 (2005));
    # Passut & Danner (1973) agrees on Tc to 0.3 K and Pc to 1%.
    "2_methylpentane": (497.7, 3.03975e6, 0.279, 86.177),
    "3_methylpentane": (504.6, 3.119797e6, 0.275, 86.177),
    "2_2_dimethylbutane": (489.0, 3.099532e6, 0.231, 86.177),
    "2_3_dimethylbutane": (500.0, 3.149181e6, 0.247, 86.177),
    "methylcyclopentane": (532.7, 3.789555e6, 0.239, 84.161),
    "cyclohexane": (553.8, 4.080358e6, 0.213, 84.161),

    # C5 olefins (amylenes; issues #308/#310). See SOURCE_CITATIONS.
    "1_pentene": (464.8, 3.560e6, 0.237, 70.13),
    "2_methyl_2_butene": (470.0, 3.420e6, 0.285, 70.13),

    # C8 alkylate paraffins (issue #310). Tc, Pc: IUPAC critical-property
    # reviews, as also tabulated in Poling-Prausnitz-O'Connell 5e App. A;
    # omega: PPO / DIPPR consensus. See SOURCE_CITATIONS.
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

    # Acids
    "formic_acid": (588.0, 5.810e6, 0.473, 46.03),
    "acetic_acid": (592.0, 5.786e6, 0.467, 60.05),

    # Esters. Tc from Majer & Svoboda's Watson fit and Pc from Ambrose,
    # Ellender et al. (1981), both via the NIST WebBook; omega computed
    # from the NIST Antoine equation at Tr = 0.7 (366.2 K, a short
    # extrapolation past its 349 K upper limit).
    "ethyl_acetate": (523.2, 3.882e6, 0.366, 88.106),

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
SOURCE_CITATIONS["ethyl_acetate"] = (
    "NIST Chemistry WebBook (C141786): Antoine, Polak & Mertl 1965; Pc, "
    "Ambrose, Ellender et al. 1981; Tc and Hvap, Majer & Svoboda 1985; "
    "Cp(l), Pintos, Bravo et al. 1988; Hf(g), Wiberg & Waldron 1991"
)


_BUTENES_SOURCE = (
    "Tc, Pc, omega: Lemmon & Ihmels, Fluid Phase Equilib. 228-229, 173 (2005); "
    "ideal-gas Cp: cubic fitted to TRC (1997) tables via NIST WebBook, 200-800 K; "
    "Hvap, Antoine, Hf: NIST WebBook (sources in database.py)")
SOURCE_CITATIONS.update({name: _BUTENES_SOURCE for name in
                         ("cis_2_butene", "trans_2_butene", "isobutylene")})
_C6_ISOMERS_SOURCE = (
    "Tc, Pc, omega: PSRK Rev. 4 appendix, Horstmann et al., Fluid Phase "
    "Equilib. 227, 157 (2005); ideal-gas Cp: cubic fitted to the NIST WebBook "
    "tables (Scott 1974; TRC 1997; Dorofeeva 1986), 273-800 K; Hvap, Antoine "
    "(Willingham, Taylor et al. 1945), Hf (Prosen & Rossini 1945; Prosen, "
    "Johnson et al. 1946): NIST WebBook (details in database.py)")
C6_ISOMERS = ("2_methylpentane", "3_methylpentane", "2_2_dimethylbutane",
              "2_3_dimethylbutane", "methylcyclopentane", "cyclohexane")
SOURCE_CITATIONS.update({name: _C6_ISOMERS_SOURCE for name in C6_ISOMERS})

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
    "1_pentene": "",
    "2_methyl_2_butene": (
        " Tc = 470.0 K (unverified): IUPAC, CRC and Yaws give 470.0 K but "
        "the NIST WebBook gives 473.9 K."
    ),
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
    key = name.lower().replace(" ", "_").replace("-", "_").replace(",", "_")
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
#     "Cp_ig": (a, b, c, d) for the IDEAL-GAS Cp = a + bT + cT² + dT³
#         (J/mol/K); required, and what CubicThermo reads (SpeciesData
#         .Cp_vapor_coeffs).
#     "Cp": the LIQUID heat capacity near 298 K, same form; optional. What
#         IdealThermo integrates for liquid enthalpy (SpeciesData.Cp_coeffs).
#         Left out where there is no liquid at ambient conditions (methane,
#         N2, O2, ethylene, ...): Cp_coeffs then falls back to "Cp_ig".
#         The two differ by a factor of 2-3 for water and the alcohols, so
#         they are never one field (issue #393).
#     "Hvap": (H_ref, n, Tc) and "Hvap_T": T_ref -- the MEASURED heat of
#         vaporization H_ref (J/mol) at T_ref (K), normally the normal boiling
#         point. `get_species_data` turns it into Watson's prefactor,
#         A = H_ref / (1 - T_ref/Tc)^n, so Hvap(T) = A (1 - T/Tc)^n passes
#         through H_ref at T_ref. The table used to hold H_ref in A's place,
#         which put every species' Hvap 30-40% low (water: 29.3 kJ/mol at
#         its boiling point instead of 40.66). Keeping the measured number
#         and its temperature is what lets a test check each entry against
#         the value it came from. T_ref is from the NIST WebBook.
#         An entry with no "Hvap_T" holds (A, n, Tc) itself, A already
#         back-solved from a measured value (the refinery species below).
#     "antoine": (A, B, C) for log10(P/Pa) = A - B/(T+C),
#         Note: A = A_NIST + 5 because NIST uses bar and we use Pa (1 bar = 1e5 Pa).
#     "Hf": standard heat of formation (J/mol) at 298.15 K,
# }
# Antoine sources: per species in ANTOINE_SOURCES (after these tables), with
# the range recorded in "T_antoine_min"/"T_antoine_max". Every entry
# reproduces its normal boiling point (tests/test_antoine_database.py).

_IDEAL_THERMO_DATA = {
    # Light gases
    "nitrogen": {
        "MW": 28.01,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A);
        # reproduces the former room-T constant (29.0) to ~1% at 298 K but is
        # accurate over the 50-1000 K span needed for cryogenic duties.
        "Cp_ig": (31.15, -1.357e-2, 2.680e-5, -1.168e-8),
        "Hvap": (5577.0, 0.38, 126.2),
        "Hvap_T": 77.34,
        "antoine": (8.63792, 257.877, -6.344),
        "T_antoine_min": 63.14,
        "T_antoine_max": 95.73,
        "Hf": 0.0,
    },
    "oxygen": {
        "MW": 32.00,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.1%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (29.3183, -0.00735043, 3.10953e-05, -1.86327e-08),
        "Hvap": (6820.0, 0.38, 154.6),
        "Hvap_T": 90.2,
        "antoine": (8.85845, 325.675, -5.667),
        "T_antoine_min": 54.36,
        "T_antoine_max": 131.01,
        "Hf": 0.0,
    },
    "carbon_dioxide": {
        "MW": 44.01,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (19.80, 7.344e-2, -5.602e-5, 1.715e-8),
        # CO2 sublimes at 1 atm, so no normal boiling point: anchored at the
        # triple point instead, 15.42 kJ/mol at 216.59 K (NIST WebBook fluid
        # tables, Span-Wagner EOS). The former 16.7 kJ/mol was a
        # Clausius-Clapeyron value "at 288 K", where the true latent heat is 7.8.
        "Hvap": (15420.0, 0.38, 304.2),
        "Hvap_T": 216.59,
        "antoine": (9.46212, 748.28, -16.9),
        "T_antoine_min": 216.0,
        "T_antoine_max": 289.38,
        "Hf": -393510.0,
    },
    "ammonia": {
        "MW": 17.03,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.2%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (31.567, -0.0028806, 6.93064e-05, -4.3892e-08),
        "Hvap": (23350.0, 0.38, 405.4),
        "Hvap_T": 239.82,
        "antoine": (9.86886, 1113.928, -10.409),
        "T_antoine_min": 239.6,
        "T_antoine_max": 401.79,
        "Hf": -45940.0,
    },

    # Alkanes
    "methane": {
        "MW": 16.04,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (19.25, 5.213e-2, 1.197e-5, -1.132e-8),
        "Hvap": (8180.0, 0.38, 190.6),
        "Hvap_T": 111.67,
        "antoine": (8.9895, 443.028, -0.49),
        "T_antoine_min": 90.99,
        "T_antoine_max": 189.99,
        "Hf": -74870.0,
    },
    "ethane": {
        "MW": 30.07,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (5.409, 1.781e-1, -6.938e-5, 8.713e-9),
        "Hvap": (14690.0, 0.38, 305.3),
        "Hvap_T": 184.6,
        "antoine": (8.93835, 659.739, -16.719),
        "T_antoine_min": 135.74,
        "T_antoine_max": 249.99,
        "Hf": -84000.0,
    },
    "propane": {
        "MW": 44.10,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (-4.224, 3.063e-1, -1.586e-4, 3.215e-8),
        "Hvap": (19040.0, 0.38, 369.8),
        "Hvap_T": 231.1,
        "antoine": (9.01158, 834.26, -22.763),
        "T_antoine_min": 147.9,
        "T_antoine_max": 339.78,
        "Hf": -104700.0,
    },
    "n_butane": {
        "MW": 58.12,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (9.487, 3.313e-1, -1.108e-4, -2.822e-9),
        "Hvap": (22390.0, 0.38, 425.1),
        "Hvap_T": 273.0,
        "antoine": (8.93266, 935.773, -34.361),
        "T_antoine_min": 200.5,
        "T_antoine_max": 331.23,
        "Hf": -125600.0,
    },
    "n_pentane": {
        "MW": 72.15,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (-3.626, 4.873e-1, -2.580e-4, 5.305e-8),
        "Hvap": (25790.0, 0.38, 469.7),
        "Hvap_T": 309.2,
        "antoine": (8.9892, 1070.617, -40.454),
        "T_antoine_min": 220.72,
        "T_antoine_max": 394.17,
        "Hf": -146800.0,
    },
    "n_hexane": {
        "MW": 86.18,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (-4.413, 5.820e-1, -3.119e-4, 6.494e-8),
        "Hvap": (28850.0, 0.38, 507.6),
        "Hvap_T": 341.9,
        "antoine": (9.00266, 1171.53, -48.784),
        "T_antoine_min": 243.54,
        "T_antoine_max": 425.99,
        "Hf": -167200.0,
    },
    "n_heptane": {
        "MW": 100.20,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.5%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (42.6329, 0.362966, 0.000263218, -3.11709e-07),
        # Liquid Cp at 298 K, Poling 5th ed. Table A: 225.0 J/mol/K.
        "Cp": (225.0, 0.0, 0.0, 0.0),
        "Hvap": (31770.0, 0.38, 540.2),
        "Hvap_T": 371.5,
        "antoine": (9.02832, 1268.636, -56.199),
        "T_antoine_min": 270.03,
        "T_antoine_max": 464.18,
        "Hf": -187800.0,
    },
    "n_octane": {
        "MW": 114.23,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.5%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (46.6923, 0.420985, 0.000291251, -3.55776e-07),
        # Liquid Cp at 298 K, Poling 5th ed. Table A: 254.2 J/mol/K.
        "Cp": (254.2, 0.0, 0.0, 0.0),
        "Hvap": (34410.0, 0.38, 568.7),
        "Hvap_T": 398.7,
        "antoine": (9.04867, 1355.126, -63.633),
        "T_antoine_min": 289.96,
        "T_antoine_max": 494.37,
        "Hf": -208600.0,
    },

    # Branched alkanes (NGL isomers). These have critical properties above but
    # previously had no ideal-thermo record, so get_species_data raised
    # KeyError, blocking a full NGL mixture on the ideal-thermo/CubicThermo path.
    "isobutane": {
        "MW": 58.12,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (-1.390, 3.847e-1, -1.846e-4, 2.895e-8),
        # Watson Hvap interpolated between propane and n-butane (approximate).
        "Hvap": (21300.0, 0.38, 407.8),
        "Hvap_T": 262.0,
        # Antoine: NIST WebBook, Aston, Kennedy et al. 1940 (see ANTOINE_SOURCES).
        "antoine": (8.94417, 912.141, -29.808),
        "T_antoine_min": 163.1,
        "T_antoine_max": 346.25,
        "Hf": -134200.0,
    },
    "isopentane": {
        "MW": 72.15,
        # Ideal-gas Cp cubic (Reid, Prausnitz & Poling 4th ed., App. A).
        "Cp_ig": (-9.525, 5.066e-1, -2.729e-4, 5.723e-8),
        # Watson Hvap interpolated between n-butane and n-pentane (approximate).
        "Hvap": (24700.0, 0.38, 460.4),
        "Hvap_T": 301.1,
        # Antoine: NIST WebBook, Willingham, Taylor et al. 1945 (see
        # ANTOINE_SOURCES).
        "antoine": (8.91457, 1020.012, -40.053),
        "T_antoine_min": 220.96,
        "T_antoine_max": 358.81,
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
    "1_pentene": {
        "MW": 70.13,
        "Cp_ig": (-4.7482e-01, 4.2488e-01, -2.0783e-04, 2.9140e-08),
        "Hvap": (37641.0, 0.38, 464.8),
        "antoine": (8.96914, 1044.01, -39.7),
        "T_antoine_min": 223.89,
        "T_antoine_max": 324.32,
        "Hf": -21100.0,
    },
    "2_methyl_2_butene": {
        "MW": 70.13,
        "Cp_ig": (-5.2007, 4.3102e-01, -2.2737e-04, 4.6403e-08),
        "Hvap": (39786.0, 0.38, 470.0),
        "antoine": (9.09149, 1124.33, -36.52),
        "T_antoine_min": 230.69,
        "T_antoine_max": 333.14,
        "Hf": -41700.0,
    },
    "2_2_4_trimethylpentane": {
        "MW": 114.23,
        "Cp_ig": (-23.936, 8.4297e-01, -4.6936e-04, 1.0446e-07),
        "Hvap": (47745.0, 0.38, 543.8),
        "antoine": (8.93646, 1257.85, -52.383),
        "T_antoine_min": 275.5,
        "T_antoine_max": 398.38,
        "Hf": -224000.0,
    },
    "2_3_4_trimethylpentane": {
        "MW": 114.23,
        "Cp_ig": (-30.448, 9.0998e-01, -6.0879e-04, 1.7711e-07),
        "Hvap": (50052.0, 0.38, 566.4),
        "antoine": (8.977, 1314.31, -55.669),
        "T_antoine_min": 300.19,
        "T_antoine_max": 413.19,
        "Hf": -217300.0,
    },
    "2_5_dimethylhexane": {
        "MW": 114.23,
        "Cp_ig": (-30.405, 8.6874e-01, -5.2198e-04, 1.2478e-07),
        "Hvap": (51098.0, 0.38, 550.0),
        "antoine": (8.98112, 1285.47, -58.902),
        "T_antoine_min": 285.2,
        "T_antoine_max": 408.2,
        "Hf": -222500.0,
    },

    # Alkenes
    "ethylene": {
        "MW": 28.05,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.6%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (22.9868, 0.0331137, 0.00015057, -1.2105e-07),
        "Hvap": (13540.0, 0.38, 282.3),
        "Hvap_T": 169.0,
        "antoine": (8.87261, 584.146, -18.307),
        "T_antoine_min": 126.97,
        "T_antoine_max": 219.87,
        "Hf": 52470.0,
    },
    "propylene": {
        "MW": 42.08,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (20.8685, 0.128863, 8.86068e-05, -1.01328e-07),
        "Hvap": (18420.0, 0.38, 365.6),
        "Hvap_T": 225.6,
        "antoine": (8.97488, 795.819, -24.884),
        "T_antoine_min": 146.2,
        "T_antoine_max": 317.66,
        "Hf": 20410.0,
    },
    # cis-2-butene, trans-2-butene, isobutylene. Cp: a cubic fitted here, by
    # least squares, to the TRC (1997) ideal-gas tables in the NIST WebBook at
    # 200-800 K (worst point 1.1%, cis at 200 K; tests/test_database.py
    # checks it against the tabulated values). Hvap: the Watson A that
    # reproduces one NIST WebBook value -- Majer & Svoboda (1985) at the normal
    # boiling point for cis (23.34 kJ/mol, 276.9 K) and trans (22.72, 274 K);
    # Lamb & Roper (1940) for isobutylene (22.8, 258 K). Antoine: NIST WebBook
    # log10(P/bar) fits with A shifted +5 for Pa -- Scott et al. 1944
    # (203-296 K), Guttman & Pitzer 1945 (202-274 K), Lamb & Roper 1940
    # (216-273 K); the range is recorded. Hf (gas, 298 K): Prosen, Maron &
    # Rossini, J. Res. NBS 46, 106 (1951).
    "cis_2_butene": {
        "MW": 56.106,
        "Cp_ig": (29.245, 0.11883, 2.4322e-4, -2.1312e-7),
        "Hvap": (34248.0, 0.38, 435.75),
        "antoine": (8.98744, 957.06, -36.504),
        "T_antoine_min": 203.06,
        "T_antoine_max": 295.91,
        "Hf": -7700.0,
    },
    "trans_2_butene": {
        "MW": 56.106,
        "Cp_ig": (35.445, 0.14124, 1.6382e-4, -1.5586e-7),
        "Hvap": (33472.0, 0.38, 428.61),
        "antoine": (9.0436, 982.166, -30.775),
        "T_antoine_min": 201.7,
        "T_antoine_max": 274.13,
        "Hf": -10800.0,
    },
    "isobutylene": {
        "MW": 56.106,
        "Cp_ig": (23.064, 0.21696, 3.1111e-5, -8.2884e-8),
        "Hvap": (32837.0, 0.38, 418.09),
        "antoine": (8.64709, 799.055, -46.615),
        "T_antoine_min": 216.4,
        "T_antoine_max": 273.0,
        "Hf": -17900.0,
    },

    # The C6 isomers and naphthenes of light-naphtha isomerization (#311).
    # Cp: a cubic fitted here, by least squares, to the NIST WebBook ideal-gas
    # tables at 273.15-800 K -- Scott (1974) for the four branched hexanes,
    # TRC (1997) for methylcyclopentane, Dorofeeva et al. (1986) for
    # cyclohexane; worst point 0.5% (tests/test_database.py checks them).
    # Hvap: the Watson A that reproduces one NIST WebBook value -- Majer &
    # Svoboda (1985) at the normal boiling point for 2-methylpentane (27.79
    # kJ/mol, 333.4 K); the 298 K standard value otherwise (3-MP 30.3,
    # 2,2-DMB 27.68, 2,3-DMB 29.12, MCP 31.7, cyclohexane 33.1 kJ/mol).
    # Antoine: NIST WebBook log10(P/bar) fits of Willingham, Taylor et al.
    # (1945) with A shifted +5 for Pa; the range is recorded. Hf (gas, 298 K):
    # Prosen & Rossini (1945) for the hexanes, Prosen, Johnson & Rossini
    # (1946) for the naphthenes, as the NIST WebBook gives them.
    "2_methylpentane": {
        "MW": 86.177,
        "Cp_ig": (-6.882, 0.57438, -2.5124e-4, 1.6512e-8),
        "Hvap": (42344.0, 0.38, 497.7),
        "antoine": (8.964, 1135.41, -46.578),
        "T_antoine_min": 238.84,
        "T_antoine_max": 402.79,
        "Hf": -174300.0,
    },
    "3_methylpentane": {
        "MW": 86.177,
        "Cp_ig": (-4.5919, 0.54702, -2.0104e-4, -1.1154e-8),
        "Hvap": (42553.0, 0.38, 504.6),
        "antoine": (8.97377, 1152.368, -46.021),
        "T_antoine_min": 288.44,
        "T_antoine_max": 337.23,
        "Hf": -171600.0,
    },
    "2_2_dimethylbutane": {
        "MW": 86.177,
        "Cp_ig": (-3.1491, 0.54281, -1.8902e-4, -6.8825e-9),
        "Hvap": (39577.0, 0.38, 489.0),
        "antoine": (8.87973, 1081.176, -43.807),
        "T_antoine_min": 288.53,
        "T_antoine_max": 323.68,
        "Hf": -185600.0,
    },
    "2_3_dimethylbutane": {
        "MW": 86.177,
        "Cp_ig": (-20.012, 0.6334, -3.5449e-4, 8.1562e-8),
        "Hvap": (41104.0, 0.38, 500.0),
        "antoine": (8.93473, 1127.187, -44.2),
        "T_antoine_min": 287.41,
        "T_antoine_max": 331.94,
        "Hf": -177800.0,
    },
    "methylcyclopentane": {
        "MW": 84.161,
        "Cp_ig": (-35.499, 0.53993, -1.5819e-4, -5.2279e-8),
        "Hvap": (43295.0, 0.38, 532.7),
        "antoine": (8.98773, 1186.059, -47.108),
        "T_antoine_min": 288.18,
        "T_antoine_max": 345.78,
        "Hf": -106700.0,
    },
    "cyclohexane": {
        "MW": 84.161,
        "Cp_ig": (-30.506, 0.46222, 3.0128e-5, -1.5954e-7),
        "Hvap": (44401.0, 0.38, 553.8),
        "antoine": (8.96988, 1203.526, -50.287),
        "T_antoine_min": 279.86,
        "T_antoine_max": 462.35,
        "Hf": -123100.0,
    },

    # Aromatics
    "benzene": {
        "MW": 78.11,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 1.2%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (-10.1815, 0.296606, 0.000108136, -2.09159e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (136.0, 0.0, 0.0, 0.0),
        "Hvap": (30720.0, 0.38, 562.0),
        "Hvap_T": 353.3,
        "antoine": (9.01814, 1203.835, -53.226),
        "T_antoine_min": 278.67,
        "T_antoine_max": 458.26,
        "Hf": 82880.0,
    },
    "toluene": {
        "MW": 92.14,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.9%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (-4.93911, 0.354613, 9.60085e-05, -2.08693e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (157.0, 0.0, 0.0, 0.0),
        "Hvap": (33180.0, 0.38, 591.8),
        "Hvap_T": 383.8,
        "antoine": (9.07827, 1343.943, -53.773),
        "T_antoine_min": 278.07,
        "T_antoine_max": 496.74,
        "Hf": 50170.0,
    },
    "ethylbenzene": {
        "MW": 106.17,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.7%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (-0.242239, 0.421222, 9.42296e-05, -2.25946e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (183.0, 0.0, 0.0, 0.0),
        "Hvap": (35570.0, 0.38, 617.2),
        "Hvap_T": 409.3,
        "antoine": (9.07488, 1419.315, -60.539),
        "T_antoine_min": 329.74,
        "T_antoine_max": 410.27,
        "Hf": 29790.0,
    },
    "styrene": {
        "MW": 104.15,
        # Ideal-gas Cp: Joback & Reid (1987) group contribution (Poling et al. 5th ed. ch. 3), cubic as published (118.9 J/mol/K at 298 K, 3% below NIST's 122).
        "Cp_ig": (-41.28, 0.6649, -0.0004655, 1.269e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (182.0, 0.0, 0.0, 0.0),
        "Hvap": (36820.0, 0.38, 636.0),
        "Hvap_T": 419.0,
        "antoine": (9.21948, 1525.059, -56.379),
        "T_antoine_min": 303.07,
        "T_antoine_max": 417.92,
        "Hf": 147360.0,
    },

    # Alcohols
    "methanol": {
        "MW": 32.04,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (31.7923, 0.00679435, 0.000147624, -1.01395e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (81.0, 0.0, 0.0, 0.0),
        "Hvap": (35210.0, 0.38, 512.6),
        "Hvap_T": 337.8,
        "antoine": (10.20409, 1581.341, -33.5),
        "T_antoine_min": 266.42,
        "T_antoine_max": 506.96,
        "Hf": -201200.0,
    },
    "ethanol": {
        "MW": 46.07,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (23.6027, 0.118707, 0.000106913, -1.15197e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (112.0, 0.0, 0.0, 0.0),
        "Hvap": (38560.0, 0.38, 513.9),
        "Hvap_T": 351.5,
        "antoine": (10.24677, 1598.673, -46.424),
        "T_antoine_min": 292.77,
        "T_antoine_max": 366.63,
        "Hf": -234800.0,
    },
    "1_propanol": {
        "MW": 60.10,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (23.6695, 0.190514, 0.000100385, -1.31984e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (144.0, 0.0, 0.0, 0.0),
        "Hvap": (41440.0, 0.38, 536.8),
        "Hvap_T": 370.3,
        "antoine": (10.31384, 1690.864, -51.804),
        "T_antoine_min": 292.4,
        "T_antoine_max": 370.5,
        "Hf": -255200.0,
    },
    "2_propanol": {
        "MW": 60.10,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.3%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (15.4575, 0.264236, -3.24956e-05, -6.44121e-08),
        # Liquid Cp near 298 K (constant).
        "Cp": (155.0, 0.0, 0.0, 0.0),
        "Hvap": (39850.0, 0.38, 508.3),
        "Hvap_T": 355.5,
        "antoine": (10.24268, 1580.92, -53.54),
        "T_antoine_min": 281.28,
        "T_antoine_max": 373.46,
        "Hf": -272700.0,
    },
    "1_butanol": {
        "MW": 74.12,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (18.5077, 0.299634, 4.63292e-05, -1.31003e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (177.0, 0.0, 0.0, 0.0),
        "Hvap": (43290.0, 0.38, 563.1),
        "Hvap_T": 390.6,
        "antoine": (9.54607, 1351.555, -93.34),
        "T_antoine_min": 295.8,
        "T_antoine_max": 391.0,
        "Hf": -274600.0,
    },

    # Ketones
    "acetone": {
        "MW": 58.08,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.4%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (29.4649, 0.127866, 0.000116622, -1.2041e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (125.0, 0.0, 0.0, 0.0),
        "Hvap": (29100.0, 0.38, 508.2),
        "Hvap_T": 329.3,
        "antoine": (9.42448, 1312.253, -32.445),
        "T_antoine_min": 259.16,
        "T_antoine_max": 507.6,
        "Hf": -217100.0,
    },

    # Ethers
    "dimethyl_ether": {
        "MW": 46.07,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.2%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (30.0773, 0.104655, 7.19138e-05, -7.38989e-08),
        "Hvap": (21510.0, 0.38, 400.1),
        "Hvap_T": 248.2,
        "antoine": (9.11475, 894.669, -30.604),
        "T_antoine_min": 194.93,
        "T_antoine_max": 248.24,
        "Hf": -184100.0,
    },
    "diethyl_ether": {
        "MW": 74.12,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.1%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (41.7122, 0.282224, -6.33997e-05, -1.24551e-08),
        # Liquid Cp near 298 K (constant).
        "Cp": (172.0, 0.0, 0.0, 0.0),
        "Hvap": (26520.0, 0.38, 466.7),
        "Hvap_T": 307.7,
        "antoine": (9.10962, 1090.64, -41.95),
        "T_antoine_min": 229.71,
        "T_antoine_max": 328.31,
        "Hf": -252100.0,
    },

    # Acids
    "formic_acid": {
        "MW": 46.03,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.3%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (23.607, 0.0837068, 7.76682e-05, -7.47304e-08),
        # Liquid Cp near 298 K (constant).
        "Cp": (99.0, 0.0, 0.0, 0.0),
        "Hvap": (22690.0, 0.38, 588.0),
        "Hvap_T": 373.9,
        "antoine": (9.48436, 1551.38, -27.44),
        "T_antoine_min": 278.0,
        "T_antoine_max": 398.0,
        "Hf": -378600.0,
    },
    "acetic_acid": {
        "MW": 60.05,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.6%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (19.6041, 0.127073, 0.000102988, -1.21408e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (124.0, 0.0, 0.0, 0.0),
        # Anchored at 25 C, not the boiling point: the vapor is mostly dimer,
        # so the calorimetric value at 391 K (23.7 kJ/mol) is to a dimerised
        # gas, while this thermo's vapor is ideal monomer. 52 kJ/mol at 25 C
        # is vaporization to monomer (Hf liquid -484.3 vs gas -432.8 kJ/mol).
        "Hvap": (52100.0, 0.38, 592.0),
        "Hvap_T": 298.15,
        "antoine": (9.68206, 1642.54, -39.764),
        "T_antoine_min": 290.26,
        "T_antoine_max": 391.01,
        "Hf": -432800.0,
    },

    # Esters
    "ethyl_acetate": {
        "MW": 88.106,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.6%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (56.1119, 0.129273, 0.000292231, -2.60916e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (168.94, 0.0, 0.0, 0.0),  # liquid, 298.15 K (Pintos, Bravo et al. 1988)
        # 31.94 kJ/mol at 350.3 K (Majer & Svoboda 1985); with n = 0.38 it
        # gives 35.3 kJ/mol at 298 K (NIST: 35 +/- 2).
        "Hvap": (31940.0, 0.38, 523.2),
        "Hvap_T": 350.3,
        # log10(P/Pa): Polak & Mertl (1965) as fitted by NIST, +5 for bar -> Pa.
        "antoine": (9.22809, 1245.702, -55.189),
        # Fit range 288.73-348.98 K (below Tb); recorded to 493.15 K, the
        # contiguous interval over which it is within 1% of the McGarry Wagner
        # equation (see ANTOINE_SOURCES).
        "T_antoine_min": 288.73,
        "T_antoine_max": 493.15,
        # Gas phase, like the other organics here (Wiberg & Waldron 1991).
        "Hf": -444800.0,
    },

    # Water
    "water": {
        "MW": 18.015,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.2%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (33.4944, -0.00809187, 3.34442e-05, -1.96886e-08),
        # Liquid Cp near 298 K (constant).
        "Cp": (75.3, 0.0, 0.0, 0.0),  # Liquid at 25°C
        "Hvap": (40660.0, 0.38, 647.1),
        "Hvap_T": 373.12,
        "antoine": (10.20389, 1733.926, -39.485),
        "T_antoine_min": 273.16,
        "T_antoine_max": 402.42,
        "Hf": -285830.0,  # Liquid-phase formation enthalpy (consistent with liquid Cp)
    },

    # Halogenated
    "chloroform": {
        "MW": 119.38,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.1%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (21.903, 0.20011, -0.000205687, 8.05422e-08),
        # Liquid Cp near 298 K (constant).
        "Cp": (114.0, 0.0, 0.0, 0.0),
        "Hvap": (29240.0, 0.38, 536.4),
        "Hvap_T": 334.3,
        "antoine": (9.20772, 1233.129, -40.953),
        "T_antoine_min": 215.0,
        "T_antoine_max": 334.4,
        "Hf": -134100.0,
    },
    "carbon_tetrachloride": {
        "MW": 153.82,
        # Ideal-gas Cp: cubic least-squares fit (250-800 K, max error 0.2%) to the Poling, Prausnitz & O'Connell 5th ed. (2001) Appendix A Cp/R quartic.
        "Cp_ig": (29.4036, 0.274006, -0.000363663, 1.70563e-07),
        # Liquid Cp near 298 K (constant).
        "Cp": (131.0, 0.0, 0.0, 0.0),
        "Hvap": (29820.0, 0.38, 556.4),
        "Hvap_T": 349.8,
        "antoine": (9.02291, 1221.781, -45.739),
        "T_antoine_min": 293.03,
        "T_antoine_max": 350.86,
        "Hf": -128200.0,
    },
}


def watson_coeffs(H_ref: float, n: float, Tc: float, T_ref: float) -> tuple:
    """Watson's ``(A, n, Tc)`` from one measured heat of vaporization.

    ``Hvap(T) = A (1 - T/Tc)^n`` passes through ``H_ref`` at ``T_ref`` when
    ``A = H_ref / (1 - T_ref/Tc)^n``. Handing ``H_ref`` over as ``A``
    instead is the error the table used to make: ``A`` is the value the
    correlation would take at 0 K, and at any real temperature
    ``(1 - T/Tc)^n`` is well below one.

    Example:
        >>> A, n, Tc = watson_coeffs(40660.0, 0.38, 647.1, 373.12)
        >>> round(A * (1 - 373.12 / Tc) ** n)
        40660
    """
    if not 0.0 < T_ref < Tc:
        raise ValueError(f"T_ref = {T_ref} K must lie between 0 and Tc = {Tc} K")
    return (H_ref / (1.0 - T_ref / Tc) ** n, n, Tc)


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
# API Technical Data Book (Albahri) value.
_ALKYLATION_CRITICAL = {
    "2_3_dimethylpentane": (537.3, 2.910e6, 0.297, 100.20),
    "2_4_dimethylpentane": (519.8, 2.740e6, 0.304, 100.20),
    "2_2_5_trimethylhexane": (569.8, 2.330e6, 0.357, 128.26),
    "n_dodecane": (658.0, 1.820e6, 0.576, 170.34),
}
_ALKYLATION_IDEAL = {
    "1_butene": {
        "MW": 56.11,
        "Cp_ig": (2.6419, 3.2085e-01, -1.4671e-04, 1.9026e-08),
        "Hvap": (32410.0, 0.38, 419.5),
        "antoine": (8.9178, 908.8, -34.61),
        "T_antoine_min": 196.41,
        "T_antoine_max": 285.88,
        "Hf": 100.0,
    },
    "2_3_dimethylpentane": {
        "MW": 100.20,
        "Cp_ig": (-35.121, 7.9476e-01, -5.0099e-04, 1.3257e-07),
        "Hvap": (46715.0, 0.38, 537.3),
        "antoine": (8.98066, 1238.986, -51.208),
        "T_antoine_min": 281.56,
        "T_antoine_max": 387.89,
        "Hf": -198700.0,
    },
    "2_4_dimethylpentane": {
        "MW": 100.20,
        "Cp_ig": (-23.801, 7.9644e-01, -5.2659e-04, 1.4802e-07),
        "Hvap": (45580.0, 0.38, 519.8),
        "antoine": (8.95442, 1193.612, -51.343),
        "T_antoine_min": 262.4,
        "T_antoine_max": 378.01,
        "Hf": -201600.0,
    },
    "2_2_5_trimethylhexane": {
        "MW": 128.26,
        "Cp_ig": (-35.917, 9.7760e-01, -5.7667e-04, 1.3805e-07),
        "Hvap": (52981.0, 0.38, 569.8),
        "antoine": (8.97372, 1332.86, -61.34),
        "T_antoine_min": 296.3,
        "T_antoine_max": 424.25,
        "Hf": -253300.0,
    },
    "n_dodecane": {
        "MW": 170.34,
        "Cp_ig": (-15.412, 1.1537, -5.8026e-04, 7.2305e-08),
        "Hvap": (73982.0, 0.38, 658.0),
        "antoine": (9.12285, 1639.27, -91.31),
        "T_antoine_min": 355.61,
        "T_antoine_max": 585.63,
        "Hf": -289400.0,
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


# Hydrotreating species (issue #391): H2, H2S and thiophene, the three every
# HDS model needs. Hf and the ideal-gas Cp cubic are those of
# difflow_refinery.thermochemistry (the one formation table, #339), copied
# rather than imported because the core must not import a plugin;
# tests/refinery/test_thermochemistry.py::test_core_database_agrees fails if the
# two drift apart. Thiophene's critical constants, the Hvap at Tb and the
# Antoine sets were read from the data tables of the `chemicals` package
# (C. Bell et al., v1.5.2), as for the refinery isomers above.
_HDS_CRITICAL = {
    # Tc = 579.4 K: CRC (IUPAC 580.0, NIST WebBook 579.56, Yaws 579.35).
    # Pc = 5.70 MPa: IUPAC, CRC and NIST WebBook (Yaws 5.69).
    # omega = 0.200: PSRK (Horstmann et al. 2005); Yaws 0.197.
    "thiophene": (579.4, 5.70e6, 0.200, 84.136),
}
_HDS_IDEAL = {
    "hydrogen": {
        "MW": 2.016,
        "Cp_ig": (27.44, 0.00855236, -1.38696e-05, 8.18211e-09),
        # H2 is supercritical (Tc = 33.19 K) at every process temperature, so
        # these two fields carry no meaning there: they are the real
        # cryogenic data, kept so the entry is not a placeholder. Hvap: CRC,
        # 0.900 kJ/mol at Tb = 20.39 K; Watson clips Tr at 0.999, so above Tc
        # it is a constant ~94 J/mol that cancels in every enthalpy
        # difference. Antoine: Poling-Prausnitz-O'Connell 5e, valid
        # 10.25-22.82 K only; above that it is an extrapolation whose one use
        # is to make Raoult's K for H2 large, as a non-condensable's should be.
        "Hvap": (900.0, 0.38, 33.19),
        "Hvap_T": 20.39,
        "antoine": (7.93954, 66.7954, 2.5),
        "T_antoine_min": 10.25,
        "T_antoine_max": 22.82,
        "Hf": 0.0,
    },
    "hydrogen_sulfide": {
        "MW": 34.08,
        "Cp_ig": (32.04997, 0.0007138444, 2.527715e-05, -1.226284e-08),
        # Hvap: CRC, 18.67 kJ/mol at Tb = 213.6 K. Antoine: PPO 5e,
        # 185.51-227.2 K. Supercritical above Tc = 373.5 K.
        "Hvap": (18670.0, 0.38, 373.5),
        "Hvap_T": 213.6,
        "antoine": (9.22882, 806.933, -21.76),
        "T_antoine_min": 185.51,
        "T_antoine_max": 254.79,
        "Hf": -20600.0,
    },
    "thiophene": {
        "MW": 84.136,
        "Cp_ig": (-31.77668, 0.4575949, -0.0003987942, 1.347712e-07),
        # Hvap: CRC, 31.48 kJ/mol at Tb = 357.15 K. Antoine: PPO 5e,
        # 267.2-381.16 K.
        "Hvap": (31480.0, 0.38, 579.4),
        "Hvap_T": 357.15,
        "antoine": (9.08416, 1246.02, -51.8),
        "T_antoine_min": 267.2,
        "T_antoine_max": 381.16,
        "Hf": 114900.0,
    },
}
_HDS_SOURCE = (
    "Hf and ideal-gas Cp: difflow_refinery.thermochemistry (H2: element, "
    "Cp TRC; H2S: CODATA, Cp JANAF; thiophene: CRC 114.9 kJ/mol, Cp TRC). "
    "Tc, Pc, omega, Hvap, Antoine: CRC Handbook, IUPAC, PSRK (Horstmann et "
    "al. 2005) and Poling-Prausnitz-O'Connell 5e App. A, as tabulated in the "
    "chemicals package v1.5.2 (details in database.py). H2 Hvap and Antoine "
    "are cryogenic data with no meaning at process temperatures."
)
_CRITICAL_DATA.update(_HDS_CRITICAL)
_IDEAL_THERMO_DATA.update(_HDS_IDEAL)
for _name in _HDS_IDEAL:
    SOURCE_CITATIONS[_name] = _HDS_SOURCE
del _name


# Ideal-thermo records for species that had critical constants only (issue
# #442, Ch. 2). Obtained as the refinery species above were: every number
# was read from the data tables shipped with the `chemicals` package
# (C. Bell et al., v1.5.2), which transcribe the sources named per entry,
# and accepted only after an independent check (NIST WebBook could not be
# reached): each Antoine set's implied normal boiling point against the
# reference-EOS value (CoolProp 8.0, the same reference equations of state
# the NIST WebBook fluid tables use) or the CRC Handbook Tb; each ideal-gas
# Cp at 298.15 K against a separately tabulated value (CRC, JANAF or PPO's
# own Cp(298 K) column); each Hvap against the reference EOS where there is
# one. tests/test_database_ch2_species.py pins all of them.
#
# Conventions as for the table above: "Cp_ig" is a cubic least-squares fit
# by this project to the cited ideal-gas correlation over the stated range;
# "Cp" (liquid, constant) is given only for species that are liquid at
# 298 K and 1 atm; Antoine is log10(P/Pa), T in K; "Hvap" is the measured
# value at "Hvap_T" with Watson n = 0.38 and Tc from _CRITICAL_DATA; Hf is
# the ideal-gas standard enthalpy of formation at 298.15 K.
#
# Not added: helium (Antoine 1.85-5.34 K only; the PPO set misses Tb by
# 0.08 K, 2%, and there is nothing to vaporize at process conditions) and
# acetylene (no normal boiling point: it sublimes at 1 atm, and no
# triple-point latent heat could be cross-checked).
_PPO = "Poling, Prausnitz & O'Connell 5e (2001) App. A"
_CH2_IDEAL = {
    "carbon_monoxide": {
        "MW": 28.01,
        # Ideal-gas Cp: cubic fit (250-800 K, max error 0.19%) to the PPO 5e
        # App. A Cp/R quartic; 29.05 at 298 K (CRC 29.1).
        "Cp_ig": (30.0245, -0.0106935, 3.02675e-05, -1.81671e-08),
        "Hvap": (6040.0, 0.38, 132.9),  # CRC, at Tb 81.65 K (ref. EOS 6013)
        "Hvap_T": 81.65,
        "antoine": (8.81912, 291.743, -5.154),
        "T_antoine_min": 69.73,
        "T_antoine_max": 88.08,
        "Hf": -110500.0,  # CRC
    },
    "chloromethane": {
        "MW": 50.49,
        # Cp_ig: cubic fit (250-800 K, max error 0.39%) to PPO 5e App. A.
        "Cp_ig": (22.9095, 0.0451626, 6.94007e-05, -6.29571e-08),
        "Hvap": (21400.0, 0.38, 416.3),  # CRC, at Tb (ref. EOS 21 613)
        "Hvap_T": 249.06,
        "antoine": (9.16533, 920.86, -27.57),
        "T_antoine_min": 184.6,
        "T_antoine_max": 265.87,
        "Hf": -81900.0,  # CRC
    },
    "dichloromethane": {
        "MW": 84.93,
        # Cp_ig: cubic fit (250-800 K, max error 0.15%) to PPO 5e App. A.
        "Cp_ig": (19.3164, 0.124199, -6.0483e-05, 1.67952e-09),
        "Cp": (100.0, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (28060.0, 0.38, 510.0),  # CRC, at Tb
        "Hvap_T": 313.15,
        "antoine": (9.07622, 1070.07, -49.91),
        "T_antoine_min": 235.2,
        "T_antoine_max": 333.36,
        "Hf": -95400.0,  # CRC
    },
    "heavy_water": {
        "MW": 20.03,
        # Cp_ig: cubic fit (250-800 K, max error 0.18%) to PPO 5e App. A;
        # 34.23 at 298 K (JANAF 34.26).
        "Cp_ig": (32.7768, -0.00472089, 3.93985e-05, -2.40454e-08),
        "Cp": (84.35, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        # At Tb, from the IAPWS reference EOS for D2O (Herrig, Thol, Harvey &
        # Lemmon, J. Phys. Chem. Ref. Data 47, 043102 (2018)) via CoolProp.
        "Hvap": (41500.0, 0.38, 643.9),
        "Hvap_T": 374.55,
        "antoine": (10.04327, 1616.76, -53.61),
        "T_antoine_min": 335.17,
        "T_antoine_max": 394.54,
        "Hf": -249200.0,  # gas, JANAF (Chase 1998)
    },
    "m_xylene": {
        "MW": 106.17,
        # Cp_ig: cubic fit (250-800 K, max error 0.59%) to PPO 5e App. A.
        "Cp_ig": (4.08942, 0.400608, 8.71022e-05, -2.01227e-07),
        "Cp": (188.44, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (35660.0, 0.38, 617.0),  # CRC, at Tb (ref. EOS 36 112)
        "Hvap_T": 412.22,
        "antoine": (9.14051, 1468.703, -57.03),
        "T_antoine_min": 308.54,
        "T_antoine_max": 439.56,
        "Hf": 17300.0,  # CRC
    },
    "o_xylene": {
        "MW": 106.17,
        # Cp_ig: cubic fit (250-800 K, max error 0.31%) to PPO 5e App. A.
        "Cp_ig": (11.1313, 0.425453, -2.59983e-05, -1.10183e-07),
        "Cp": (188.07, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (36240.0, 0.38, 630.3),  # CRC, at Tb (ref. EOS 36 384)
        "Hvap_T": 417.65,
        "antoine": (9.09789, 1458.706, -61.109),
        "T_antoine_min": 312.75,
        "T_antoine_max": 445.3,
        "Hf": 19100.0,  # CRC
    },
    "p_xylene": {
        "MW": 106.17,
        # Cp_ig: cubic fit (250-800 K, max error 0.66%) to PPO 5e App. A.
        "Cp_ig": (1.47592, 0.409632, 9.2401e-05, -2.1438e-07),
        "Cp": (181.66, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (35670.0, 0.38, 616.2),  # CRC, at Tb (ref. EOS 35 709)
        "Hvap_T": 411.38,
        "antoine": (9.10494, 1446.832, -58.523),
        "T_antoine_min": 307.81,
        "T_antoine_max": 438.88,
        "Hf": 18000.0,  # CRC
    },
    "methyl_acetate": {
        "MW": 74.08,
        # Cp_ig: cubic fit (298-800 K, the quartic's own lower limit; max
        # error 0.16%) to PPO 5e App. A.
        "Cp_ig": (23.2684, 0.21628, -4.05057e-06, -5.73302e-08),
        "Cp": (143.9, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (30320.0, 0.38, 506.6),  # CRC, at Tb
        "Hvap_T": 330.02,
        "antoine": (9.18621, 1156.43, -53.46),
        "T_antoine_min": 249.9,
        "T_antoine_max": 351.11,
        "Hf": -413300.0,  # CRC
    },
    "methyl_ethyl_ketone": {
        "MW": 72.11,
        # Cp_ig: cubic fit (250-800 K, max error 0.28%) to PPO 5e App. A.
        "Cp_ig": (40.7949, 0.196684, 7.72855e-05, -1.08013e-07),
        "Cp": (158.9, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (31300.0, 0.38, 535.5),  # CRC, at Tb
        "Hvap_T": 352.74,
        "antoine": (9.1386, 1232.63, -54.46),
        "T_antoine_min": 265.62,
        "T_antoine_max": 375.66,
        "Hf": -238500.0,  # CRC
    },
    "n_nonane": {
        "MW": 128.26,
        # Cp_ig: cubic fit (250-800 K, max error 0.60%) to PPO 5e App. A.
        "Cp_ig": (50.2015, 0.481856, 0.000315503, -3.99136e-07),
        "Cp": (284.45, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (37180.0, 0.38, 594.6),  # CRC, at Tb (ref. EOS 37 012)
        "Hvap_T": 423.97,
        "antoine": (9.07356, 1438.03, -70.456),
        "T_antoine_min": 319.57,
        "T_antoine_max": 451.64,
        "Hf": -228200.0,  # CRC
    },
    "n_decane": {
        "MW": 142.28,
        # Cp_ig: cubic fit (250-800 K, max error 0.62%) to PPO 5e App. A.
        "Cp_ig": (53.8242, 0.542058, 0.000342162, -4.43992e-07),
        "Cp": (314.54, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (39580.0, 0.38, 617.7),  # CRC, at Tb (ref. EOS 39 326)
        "Hvap_T": 447.3,
        "antoine": (9.06853, 1495.17, -79.292),
        "T_antoine_min": 338.53,
        "T_antoine_max": 476.15,
        "Hf": -249500.0,  # CRC
    },
    "neopentane": {
        "MW": 72.15,
        # Cp_ig: cubic fit (250-800 K, max error 1.8%) to PPO 5e App. A.
        "Cp_ig": (-8.52149, 0.542216, -0.000424207, 2.27351e-07),
        "Hvap": (22740.0, 0.38, 433.8),  # CRC, at Tb (ref. EOS 22 776)
        "Hvap_T": 282.63,
        "antoine": (8.83916, 938.234, -37.901),
        "T_antoine_min": 259.33,
        "T_antoine_max": 303.08,
        "Hf": -168000.0,  # CRC
    },
    "sulfur_dioxide": {
        "MW": 64.06,
        # Cp_ig: cubic fit (298-800 K, max error 0.08%) to the NIST WebBook
        # Shomate equation (Chase 1998, JANAF), 39.87 at 298 K. NOT the PPO
        # quartic, which as transcribed gives 42.2 J/mol/K at 298 K, 6% above
        # its own tabulated Cp(298) of 40.05.
        "Cp_ig": (25.1646, 0.0596707, -3.6835e-05, 6.08612e-09),
        "Hvap": (24940.0, 0.38, 430.8),  # CRC, at Tb (ref. EOS 24 956)
        "Hvap_T": 263.1,
        "antoine": (9.4072, 999.9, -35.96),
        "T_antoine_min": 199.71,
        "T_antoine_max": 279.47,
        "Hf": -296800.0,  # CRC
    },
    "tetrahydrofuran": {
        "MW": 72.11,
        # Cp_ig: cubic fit (250-800 K, max error 1.3%) to PPO 5e App. A.
        "Cp_ig": (4.13249, 0.177445, 0.000312103, -3.00984e-07),
        "Cp": (124.1, 0.0, 0.0, 0.0),  # liquid, 298 K, PPO 5e App. A
        "Hvap": (29810.0, 0.38, 540.2),  # CRC, at the CRC Tb
        "Hvap_T": 338.15,
        # Implied Tb 339.12 K; Perry's 8e Table 2-8 (DIPPR) agrees (339.12),
        # CRC lists 338.15 K.
        "antoine": (9.12142, 1203.11, -46.795),
        "T_antoine_min": 253.5,
        "T_antoine_max": 361.71,
        "Hf": -184100.0,  # CRC
    },
    "formaldehyde": {
        "MW": 30.03,
        # Cp_ig: cubic fit (250-800 K, max error 0.57%) to the TRC ideal-gas
        # correlation (TRC Thermodynamics of Organic Compounds in the Gas
        # State, Frenkel et al. 1994); 35.4 at 298 K (CRC 35.4).
        "Cp_ig": (30.8655, -0.0095431, 0.000102662, -6.46618e-08),
        # Perry's 8e Table 2-150 (DIPPR 106) at Tb; CRC has no entry.
        "Hvap": (23060.0, 0.38, 408.0),
        "Hvap_T": 254.05,
        "antoine": (9.281, 957.24, -30.15),
        "T_antoine_min": 190.0,
        "T_antoine_max": 271.0,
        "Hf": -108600.0,  # CRC
    },
    "acetaldehyde": {
        "MW": 44.05,
        # Cp_ig: cubic fit (250-800 K, max error 0.63%) to the TRC ideal-gas
        # correlation; 55.4 at 298 K (CRC 55.3).
        "Cp_ig": (25.6535, 0.0870445, 6.30589e-05, -6.76787e-08),
        "Cp": (89.0, 0.0, 0.0, 0.0),  # liquid, 298 K, CRC
        "Hvap": (25760.0, 0.38, 466.0),  # CRC, at Tb
        "Hvap_T": 293.25,
        "antoine": (9.1814, 1070.6, -37.15),
        "T_antoine_min": 219.0,
        "T_antoine_max": 313.0,
        "Hf": -166200.0,  # CRC
    },
}
_CH2_SOURCE = (
    "Antoine (log10 P/Pa): " + _PPO + " (Landolt-Bornstein IV/20 for "
    "formaldehyde and acetaldehyde); ideal-gas Cp: cubic fitted by this "
    "project to the " + _PPO + " Cp/R quartic (TRC correlation for the "
    "aldehydes, NIST/JANAF Shomate for SO2); liquid Cp(298 K): " + _PPO + " "
    "(CRC for acetaldehyde); Hvap at Tb: CRC Handbook (IAPWS D2O reference "
    "EOS for heavy water; Perry's 8e Table 2-150 for formaldehyde); Hf(g): "
    "CRC Handbook (JANAF for D2O). All as tabulated in the chemicals package "
    "v1.5.2; cross-checks and per-value notes in database.py."
)
_IDEAL_THERMO_DATA.update(_CH2_IDEAL)
for _name in _CH2_IDEAL:
    SOURCE_CITATIONS[_name] = _CH2_SOURCE
del _name


# Wagner vapor-pressure sets for the light gases (issue #442, Ch. 2). A
# three-constant Antoine set cannot follow a light gas from its triple point
# to Tc: methane's Prydz-Goodwin set is 2.4% high mid-range. The Wagner
# equation, ln(P/Pc) = (a tau + b tau^1.5 + c tau^e3 + d tau^e4)/Tr with
# tau = 1 - Tr, can. Two forms are used, whichever reproduces the reference
# EOS better over its range:
#
# - "2.5-5": Poling, Prausnitz & O'Connell 5e (2001) App. A, Table A
#   (Wagner 2.5-5 constants), as tabulated in chemicals v1.5.2.
# - "3-6": the original Wagner form as fitted by McGarry, Ind. Eng. Chem.
#   Process Des. Dev. 22, 313 (1983), as tabulated in chemicals v1.5.2.
#
# Format: name -> (form, (a, b, c, d, Tc [K], Pc [Pa]), T_min [K]). Tc and
# Pc are the set's own, not _CRITICAL_DATA's. Every set was checked against
# the reference-EOS saturation curve (CoolProp 8.0; the equations the NIST
# WebBook fluid tables use) from the triple point (or T_min) to Tc,
# wherever P > 1 kPa; the worst deviation is in the comment. tests/
# test_wagner.py pins the normal boiling points and spot pressures.
#
# These are used only when asked for -- get_species_data(name,
# vapor_pressure="wagner") -- so that existing Antoine-based results do not
# move; see get_species_data.
WAGNER_COEFFS: dict[str, tuple[str, tuple, float]] = {
    # PPO 2.5-5; max 0.12% (Antoine/Prydz-Goodwin: 2.35%)
    "methane": ("2.5-5", (-6.02242, 1.26652, -0.5707, -1.366,
                          190.551, 4.5992e6), 90.694),
    # PPO 2.5-5; max 0.63%
    "ethane": ("2.5-5", (-6.475, 1.41071, -1.144, -1.859,
                         305.33, 4.871e6), 90.352),
    # McGarry 3-6; max 0.30% (the PPO set: 1.6%)
    "propane": ("3-6", (-6.67833, 1.15437, -1.64984, -2.70017,
                        369.82, 4.25576e6), 145.0),
    # PPO 2.5-5; max 0.31%
    "n_butane": ("2.5-5", (-7.01763, 1.6777, -1.9739, -2.172,
                           425.25, 3.792e6), 134.86),
    # McGarry 3-6; max 0.97%
    "isobutane": ("3-6", (-6.95579, 1.5009, -2.52717, -1.49776,
                          408.14, 3.65801e6), 165.0),
    # PPO 2.5-5; max 0.11%
    "nitrogen": ("2.5-5", (-6.11102, 1.2189, -0.69366, -1.89893,
                           126.2, 3.4e6), 63.149),
    # McGarry 3-6; max 0.72%
    "oxygen": ("3-6", (-6.28275, 1.73619, -1.81349, -0.0253645,
                       154.7, 5.08987e6), 54.0),
    # McGarry 3-6; max 0.11% (triple point 216.6 K to Tc)
    "carbon_dioxide": ("3-6", (-6.95626, 1.19695, -3.12614, 2.99448,
                               304.15, 7.37499e6), 217.0),
    # PPO 2.5-5; max 0.13%
    "ammonia": ("2.5-5", (-7.28322, 1.5716, -1.85672, -2.39312,
                          405.5, 11.353e6), 195.41),
    # McGarry 3-6; max 0.42%
    "ethylene": ("3-6", (-6.32055, 1.16819, -1.55935, -1.83552,
                         282.55, 5.05088e6), 105.0),
    # McGarry 3-6; max 0.66%
    "propylene": ("3-6", (-6.64231, 1.21857, -1.81005, -2.48212,
                          364.85, 4.60523e6), 140.0),
    # McGarry 3-6; max 0.30%
    "carbon_monoxide": ("3-6", (-6.20798, 1.27885, -1.34533, -2.56842,
                                132.91, 3.50115e6), 71.0),
}
WAGNER_SOURCES: dict[str, str] = {
    name: ("Poling, Prausnitz & O'Connell, The Properties of Gases and "
           "Liquids, 5th ed. (2001), App. A, Wagner 2.5-5 constants"
           if form == "2.5-5" else
           "McGarry, Ind. Eng. Chem. Process Des. Dev. 22, 313 (1983), "
           "original Wagner (3-6) form")
    + " (as tabulated in chemicals v1.5.2)"
    for name, (form, _c, _t) in WAGNER_COEFFS.items()
}


# Antoine provenance, one entry per species in _IDEAL_THERMO_DATA. How the
# sets were chosen (2026-10): every entry's
# implied normal boiling point, B / (A - log10 101325) - C, was compared with
# the NIST WebBook value -- the reference-EOS saturation temperature at 1 atm
# where NIST has a fluid table, else its averaged Tboil. Eighteen entries
# were off by 2.5-100 K (O2, NH3, CO2, methane, ethane, propane, n-butane,
# ethylene, propylene, styrene, the propanols, 1-butanol, formic acid, the
# two ethers, CHCl3, CCl4) and had no range; five more (N2, n-pentane,
# isobutane, isopentane, acetic acid) were rounded or off-range sets, off by
# 0.2-1 K. Each was replaced by a NIST WebBook Antoine fit (log10(P/bar)
# with A shifted +5 for Pa) bracketing Tb; where no NIST fit reproduces Tb to
# 0.5 K or covers both ~298 K and Tb, by the Poling-Prausnitz-O'Connell 5e
# App. A set (already log10(P/Pa)), and for formic acid and CO2 (neither has
# a usable NIST fit) by the Landolt-Bornstein IV/20 set. PPO and Landolt
# values were read from the tables shipped with the `chemicals` package
# (C. Bell et al., v1.5.2), as for the refinery species above.
# The recorded range is the source's fit range; where NIST has a
# reference-EOS saturation table, it is extended to the contiguous interval
# over which the set is within 1% of that table (stated per entry).
# Names in the sources are as NIST prints them ('Williamham' is NIST's
# spelling of Willingham, Taylor et al. 1945).
ANTOINE_SOURCES: dict[str, str] = {
    'nitrogen': (
        "NIST WebBook (CAS 7727-37-9), Moussa, Muijlwijk, et al., 1966; fit "
        "range 63.14-78.00 K; recorded range 63.14-95.73 K, where it is within "
        "1% of the NIST WebBook reference-EOS saturation pressure"),
    'oxygen': (
        "NIST WebBook (CAS 7782-44-7), Brower and Thodos, 1968; fit range "
        "54.36-100.16 K; recorded range 54.36-131.01 K, where it is within 1% "
        "of the NIST WebBook reference-EOS saturation pressure"),
    'carbon_dioxide': (
        "Landolt-Bornstein New Series IV/20 (Dykyj, Svoboda, Wilhoit, Frenkel "
        "& Hall, Vapor Pressure of Chemicals), ln(P/Pa) form divided by ln 10 "
        "here; as tabulated in chemicals v1.5.2; range 216-273 K; recorded "
        "range 216-289.38 K, where it is within 1% of the NIST WebBook "
        "reference-EOS saturation pressure. CO2 has no normal boiling point "
        "(it sublimes at 1 atm): the set covers the liquid from the triple "
        "point (216.6 K) up"),
    'ammonia': (
        "NIST WebBook (CAS 7664-41-7), Stull, 1947; fit range 239.6-371.5 K; "
        "recorded range 239.6-401.79 K, where it is within 1% of the NIST "
        "WebBook reference-EOS saturation pressure"),
    'methane': (
        "NIST WebBook (CAS 74-82-8), Prydz and Goodwin, 1972; fit range "
        "90.99-189.99 K"),
    'ethane': (
        "NIST WebBook (CAS 74-84-0), Loomis and Walters, 1926; fit range "
        "135.74-199.91 K; recorded range 135.74-249.99 K, where it is within "
        "1% of the NIST WebBook reference-EOS saturation pressure"),
    'propane': (
        "NIST WebBook (CAS 74-98-6), Kemp and Egan, 1938; fit range "
        "166.02-231.41 K; recorded range 147.9-339.78 K, where it is within 1% "
        "of the NIST WebBook reference-EOS saturation pressure"),
    'n_butane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 200.5-292.03 K; recorded range 200.5-331.23 K, where "
        "it is within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'n_pentane': (
        "NIST WebBook (CAS 109-66-0), Osborn and Douslin, 1974; fit range "
        "268.8-341.37 K; recorded range 220.72-394.17 K, where it is within 1% "
        "of the NIST WebBook reference-EOS saturation pressure"),
    'n_hexane': (
        "NIST WebBook (CAS 110-54-3), Williamham, Taylor, et al., 1945; fit "
        "range 286.18-342.69 K; recorded range 243.54-425.99 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'n_heptane': (
        "NIST WebBook (CAS 142-82-5), Williamham, Taylor, et al., 1945; fit "
        "range 299.07-372.43 K; recorded range 270.03-464.18 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'n_octane': (
        "NIST WebBook (CAS 111-65-9), Williamham, Taylor, et al., 1945; fit "
        "range 326.08-399.72 K; recorded range 289.96-494.37 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'isobutane': (
        "NIST WebBook (CAS 75-28-5), Aston, Kennedy, et al., 1940, 2; fit "
        "range 188.06-261.54 K; recorded range 163.1-346.25 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'isopentane': (
        "NIST WebBook (CAS 78-78-4), Williamham, Taylor, et al., 1945; fit "
        "range 289.44-301.74 K; recorded range 220.96-358.81 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    '1_pentene': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 223.89-324.32 K"),
    '2_methyl_2_butene': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 230.69-333.14 K"),
    '2_2_4_trimethylpentane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 275.5-398.38 K"),
    '2_3_4_trimethylpentane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 300.19-413.19 K"),
    '2_5_dimethylhexane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 285.2-408.2 K"),
    'ethylene': (
        "NIST WebBook (CAS 74-85-1), Michels and Wassenaar, 1950; fit range "
        "149.37-188.57 K; recorded range 126.97-219.87 K, where it is within "
        "1% of the NIST WebBook reference-EOS saturation pressure"),
    'propylene': (
        "NIST WebBook (CAS 115-07-1), Powell and Giauque, 1939; fit range "
        "165.81-225.98 K; recorded range 146.2-317.66 K, where it is within 1% "
        "of the NIST WebBook reference-EOS saturation pressure"),
    'cis_2_butene': (
        "NIST WebBook (CAS 590-18-1), Scott, Ferguson, et al., 1944; fit range "
        "203.06-295.91 K"),
    'trans_2_butene': (
        "NIST WebBook (CAS 624-64-6), Guttman and Pitzer, 1945, 2; fit range "
        "201.70-274.13 K"),
    'isobutylene': (
        "NIST WebBook (CAS 115-11-7), Lamb and Roper, 1940; fit range "
        "216.40-273. K"),
    '2_methylpentane': (
        "NIST WebBook (CAS 107-83-5), Williamham, Taylor, et al., 1945; fit "
        "range 285.91-334.22 K; recorded range 238.84-402.79 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    '3_methylpentane': (
        "NIST WebBook (CAS 96-14-0), Williamham, Taylor, et al., 1945; fit "
        "range 288.44-337.23 K"),
    '2_2_dimethylbutane': (
        "NIST WebBook (CAS 75-83-2), Williamham, Taylor, et al., 1945; fit "
        "range 288.53-323.68 K"),
    '2_3_dimethylbutane': (
        "NIST WebBook (CAS 79-29-8), Williamham, Taylor, et al., 1945; fit "
        "range 287.41-331.94 K"),
    'methylcyclopentane': (
        "NIST WebBook (CAS 96-37-7), Williamham, Taylor, et al., 1945; fit "
        "range 288.18-345.78 K"),
    'cyclohexane': (
        "NIST WebBook (CAS 110-82-7), Williamham, Taylor, et al., 1945; fit "
        "range 293.06-354.73 K; recorded range 279.86-462.35 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'benzene': (
        "NIST WebBook (CAS 71-43-2), Williamham, Taylor, et al., 1945; fit "
        "range 287.70-354.07 K; recorded range 278.67-458.26 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'toluene': (
        "NIST WebBook (CAS 108-88-3), Williamham, Taylor, et al., 1945; fit "
        "range 308.52-384.66 K; recorded range 278.07-496.74 K, where it is "
        "within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'ethylbenzene': (
        "NIST WebBook (CAS 100-41-4), Williamham, Taylor, et al., 1945; fit "
        "range 329.74-410.27 K"),
    'styrene': (
        "NIST WebBook (CAS 100-42-5), Dreyer, Martin, et al., 1955; fit range "
        "303.07-417.92 K"),
    'methanol': (
        "NIST WebBook (CAS 67-56-1), Ambrose and Sprake, 1970; fit range "
        "288.1-356.83 K; recorded range 266.42-506.96 K, where it is within 1% "
        "of the NIST WebBook reference-EOS saturation pressure"),
    'ethanol': (
        "NIST WebBook (CAS 64-17-5), Ambrose and Sprake, 1970; fit range "
        "292.77-366.63 K"),
    '1_propanol': (
        "NIST WebBook (CAS 71-23-8), Kemme and Kreps, 1969; fit range "
        "292.4-370.5 K"),
    '2_propanol': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 281.28-373.46 K"),
    '1_butanol': (
        "NIST WebBook (CAS 71-36-3), Kemme and Kreps, 1969; fit range "
        "295.8-391.0 K"),
    'acetone': (
        "NIST WebBook (CAS 67-64-1), Ambrose, Sprake, et al., 1974; fit range "
        "259.16-507.60 K"),
    'dimethyl_ether': (
        "NIST WebBook (CAS 115-10-6), Kennedy, Sagenkahn, et al., 1941, 2; fit "
        "range 194.93-248.24 K"),
    'diethyl_ether': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 229.71-328.31 K"),
    'formic_acid': (
        "Landolt-Bornstein New Series IV/20 (Dykyj, Svoboda, Wilhoit, Frenkel "
        "& Hall, Vapor Pressure of Chemicals), ln(P/Pa) form divided by ln 10 "
        "here; as tabulated in chemicals v1.5.2; range 278-398 K. The vapor is "
        "partly dimerized; this is the total pressure"),
    'acetic_acid': (
        "NIST WebBook (CAS 64-19-7), McDonald, Shrader, et al., 1959; fit "
        "range 290.26-391.01 K"),
    'ethyl_acetate': (
        "NIST WebBook (CAS 141-78-6), Polák and Mertl, 1965; fit range "
        "288.73-348.98 K; recorded range 288.73-493.15 K, where it is within "
        "1% of the McGarry (1983) Wagner equation (A=-7.68521, B=1.36511, "
        "C=-4.0898, D=-1.75342, Tc=523.2 K, Pc=3.8301 MPa, as tabulated in "
        "`chemicals` v1.5.2; NIST has no reference-EOS table for it)"),
    'water': (
        "NIST WebBook (CAS 7732-18-5), Bridgeman and Aldrich, 1964; fit range "
        "304.-333. K; recorded range 273.16-402.42 K, where it is within 1% of "
        "the NIST WebBook reference-EOS saturation pressure"),
    'chloroform': (
        "NIST WebBook (CAS 67-66-3), Stull, 1947; fit range 215.-334.4 K"),
    'carbon_tetrachloride': (
        "NIST WebBook (CAS 56-23-5), Hildenbrand and McDonald, 1959, 2; fit "
        "range 293.03-350.86 K"),
    '1_butene': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 196.41-285.88 K"),
    '2_3_dimethylpentane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 281.56-387.89 K"),
    '2_4_dimethylpentane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 262.4-378.01 K"),
    '2_2_5_trimethylhexane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 296.3-424.25 K"),
    'n_dodecane': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 372.89-520.24 K; recorded range 355.61-585.63 K, where "
        "it is within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'hydrogen': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 10.25-22.82 K"),
    'hydrogen_sulfide': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 185.51-227.2 K; recorded range 185.51-254.79 K, where "
        "it is within 1% of the NIST WebBook reference-EOS saturation pressure"),
    'thiophene': (
        "Poling, Prausnitz & O'Connell 5e App. A (as tabulated in chemicals "
        "v1.5.2); range 267.2-381.16 K"),
}
# The issue #442 additions: PPO 5e App. A (log10(P/Pa), as tabulated in
# chemicals v1.5.2) over the source's range, except the two aldehydes.
ANTOINE_SOURCES.update({
    _name: (f"Poling, Prausnitz & O'Connell 5e App. A (as tabulated in "
            f"chemicals v1.5.2); range {_d['T_antoine_min']:g}-"
            f"{_d['T_antoine_max']:g} K")
    for _name, _d in _CH2_IDEAL.items()
    if _name not in ("formaldehyde", "acetaldehyde")
})
for _name in ("formaldehyde", "acetaldehyde"):
    _d = _CH2_IDEAL[_name]
    ANTOINE_SOURCES[_name] = (
        "Landolt-Bornstein New Series IV/20 (Dykyj, Svoboda, Wilhoit, Frenkel "
        "& Hall, Vapor Pressure of Chemicals), ln(P/Pa) form divided by ln 10 "
        f"here; as tabulated in chemicals v1.5.2; range "
        f"{_d['T_antoine_min']:g}-{_d['T_antoine_max']:g} K")
del _name, _d


def get_species_data(name: str, vapor_pressure: str = "antoine") -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name.

    Args:
        name: Species name (case-insensitive, underscores for spaces)
        vapor_pressure: ``"antoine"`` (default): the record as it has always
            been, Psat from the Antoine equation. ``"wagner"``: also attach
            the species' Wagner set from :data:`WAGNER_COEFFS`, which
            :meth:`~difflow.thermo.IdealThermo.Psat` then uses (the light
            gases; methane, for one, goes from 2.4% to 0.12% worst-case
            error against the reference EOS); a species without one raises
            ``KeyError``. ``"best"``: Wagner where there is a set, else
            Antoine. The default stays Antoine so that existing results do
            not move.

    Returns:
        SpeciesData namedtuple

    Raises:
        KeyError: If species not found in database (or, with
            ``vapor_pressure="wagner"``, has no Wagner set).
        ValueError: For an unknown ``vapor_pressure``.

    Example:
        >>> data = get_species_data("methanol")
        >>> data.MW
        32.04
        >>> get_species_data("methane", vapor_pressure="wagner").wagner_form
        '2.5-5'
    """
    if vapor_pressure not in ("antoine", "wagner", "best"):
        raise ValueError("vapor_pressure must be 'antoine', 'wagner' or "
                         f"'best', not {vapor_pressure!r}")
    key = name.lower().replace(" ", "_").replace("-", "_").replace(",", "_")
    if key not in _IDEAL_THERMO_DATA:
        available = ", ".join(sorted(_IDEAL_THERMO_DATA.keys()))
        raise KeyError(
            f"Species '{name}' not in database. "
            f"Available: {available}"
        )

    wagner = {}
    if vapor_pressure != "antoine":
        if key in WAGNER_COEFFS:
            form, coeffs, T_min = WAGNER_COEFFS[key]
            wagner = {"wagner_coeffs": tuple(coeffs), "wagner_form": form,
                      "T_wagner_min": T_min}
        elif vapor_pressure == "wagner":
            raise KeyError(
                f"Species '{name}' has no Wagner vapor-pressure set. "
                f"Available: {', '.join(sorted(WAGNER_COEFFS))}")

    d = _IDEAL_THERMO_DATA[key]
    return SpeciesData(
        **wagner,
        name=key,
        MW=d["MW"],
        # "Cp" is the liquid heat capacity where one is known; a species with
        # no liquid at ambient conditions (methane, N2, ...) has only "Cp_ig",
        # which then also serves IdealThermo's liquid path.
        Cp_coeffs=d.get("Cp", d["Cp_ig"]),
        Cp_vapor_coeffs=d["Cp_ig"],
        # Without "Hvap_T" the entry already holds Watson's prefactor A (the
        # refinery species, which back-solved A from a measured dHvap at Tb).
        Hvap_coeffs=(watson_coeffs(*d["Hvap"], d["Hvap_T"]) if "Hvap_T" in d
                     else tuple(d["Hvap"])),
        antoine_coeffs=d["antoine"],
        Hf=d.get("Hf", 0.0),
        T_antoine_min=d.get("T_antoine_min", 0.0),
        T_antoine_max=d.get("T_antoine_max", 1e6),
    )


# =============================================================================
# Convenience Functions
# =============================================================================


def list_species(kind: str = "ideal") -> list[str]:
    """List the species in the database.

    The database holds two tables: ideal-thermo records (Cp, Hvap,
    Antoine, Hf; :func:`get_species_data`) and critical constants (Tc, Pc,
    omega; :func:`get_critical_props`). Some species are only in the second
    (helium, the xylenes, n-decane, ...), so ``kind`` says which table the
    names must be in.

    Args:
        kind: ``"ideal"`` (default): names :func:`get_species_data`
            accepts. ``"critical"``: names :func:`get_critical_props`
            accepts. ``"any"``: names in either table, which
            :func:`get_species_info` accepts.

    Returns:
        Sorted list of species names

    Raises:
        ValueError: For any other ``kind``.

    Example:
        >>> "water" in list_species()
        True
        >>> "helium" in list_species(), "helium" in list_species("critical")
        (False, True)
    """
    if kind == "ideal":
        names = set(_IDEAL_THERMO_DATA)
    elif kind == "critical":
        names = set(_CRITICAL_DATA)
    elif kind == "any":
        names = set(_CRITICAL_DATA) | set(_IDEAL_THERMO_DATA)
    else:
        raise ValueError(
            f"kind must be 'ideal', 'critical' or 'any', not {kind!r}")
    return sorted(names)


def get_species_info(name: str) -> dict:
    """Get all available properties for a species.

    Args:
        name: Species name

    Returns:
        Dictionary with all available properties
    """
    _record_access(name, "info")
    key = name.lower().replace(" ", "_").replace("-", "_").replace(",", "_")
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
    "etoac": "ethyl_acetate",
    "etbe": "diethyl_ether",
    "xylene": "m_xylene",  # Default to m-xylene
    "isobutene": "isobutylene",
    "2_methylpropene": "isobutylene",
    "cis_butene": "cis_2_butene",
    "trans_butene": "trans_2_butene",
    "2mp": "2_methylpentane",
    "3mp": "3_methylpentane",
    "22dmb": "2_2_dimethylbutane",
    "23dmb": "2_3_dimethylbutane",
    "neohexane": "2_2_dimethylbutane",
    "diisopropyl": "2_3_dimethylbutane",
    "mcp": "methylcyclopentane",
    "isooctane": "2_2_4_trimethylpentane",
    "isohexane": "2_methylpentane",
}


def resolve_alias(name: str) -> str:
    """Resolve common aliases to canonical names.

    Args:
        name: Species name or alias

    Returns:
        Canonical species name
    """
    key = name.lower().replace(" ", "_").replace("-", "_").replace(",", "_")
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


def get_species_data(name: str, vapor_pressure: str = "antoine") -> SpeciesData:
    """Get SpeciesData for ideal thermodynamics by name (with alias support).

    See the alias-free original (``_original_get_species_data``) for
    ``vapor_pressure``: ``"antoine"`` (default), ``"wagner"`` or ``"best"``.

    Example:
        >>> get_species_data("CO2", vapor_pressure="best").wagner_form
        '3-6'
    """
    _record_access(name, "ideal")
    return _original_get_species_data(resolve_alias(name), vapor_pressure)
