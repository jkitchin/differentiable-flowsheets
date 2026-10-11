"""Antoine constants in difflow.database reproduce independent data.

Every species' Antoine set must give its normal boiling point (the T where
Psat = 1 atm) to within max(0.5 K, the reference's stated uncertainty), and
spot vapor pressures must match NIST. A regression test for the Ch. 2 gap
in which 18 entries were off by 2.5-100 K (propane's implied Tb was 254 K,
CO2's 170 K).

References are NOT the constants' own sources:

- ``TB_REFERENCE_EOS``: normal boiling points from the NIST Chemistry WebBook
  fluid tables (the reference equations of state; Lemmon, McLinden & Friend),
  saturation pressure 1.01325 bar.
- ``TB_WEBBOOK``: the first "Tboil" entry of the NIST WebBook phase-change
  data (TRC-evaluated averages), with its stated uncertainty in K.
- ``PSAT_NIST``: NIST WebBook fluid-table saturation pressures (bar).
"""

import math
import warnings

import pytest

from difflow import IdealThermo
from difflow.database import (
    ANTOINE_SOURCES, _IDEAL_THERMO_DATA, get_species_data, list_species,
    get_critical_props,
)
from difflow.thermo import AntoineRangeWarning

TB_REFERENCE_EOS = {
    "nitrogen": 77.355,
    "oxygen": 90.18781,
    "ammonia": 239.8343,
    "methane": 111.6672,
    "ethane": 184.5686,
    "propane": 231.0362,
    "n_butane": 272.6599,
    "n_pentane": 309.2093,
    "n_hexane": 341.8656,
    "n_heptane": 371.5504,
    "n_octane": 398.7938,
    "isobutane": 261.401,
    "isopentane": 300.9764,
    "ethylene": 169.3786,
    "propylene": 225.5309,
    "2_methylpentane": 333.3621,
    "cyclohexane": 353.8649,
    "benzene": 353.2188,
    "toluene": 383.7458,
    "methanol": 337.6324,
    "water": 373.1243,
    "n_dodecane": 489.4416,
    "hydrogen": 20.36891,
    "hydrogen_sulfide": 212.8549,
}

TB_WEBBOOK = {  # (Tb, stated uncertainty), K
    "1_pentene": (304.0, 8.0),
    "2_methyl_2_butene": (311.0, 1.0),
    "2_2_4_trimethylpentane": (372.4, 0.2),
    "2_3_4_trimethylpentane": (386.8, 0.3),
    "2_5_dimethylhexane": (382.1, 0.9),
    "cis_2_butene": (276.84, 0.08),
    "trans_2_butene": (274.2, 0.5),
    "isobutylene": (266.7, 0.7),
    "3_methylpentane": (336.4, 0.4),
    "2_2_dimethylbutane": (322.9, 0.1),
    "2_3_dimethylbutane": (331.2, 0.2),
    "methylcyclopentane": (345.0, 0.2),
    "ethylbenzene": (409.3, 0.4),
    "styrene": (419.0, 2.0),
    "ethanol": (351.5, 0.2),
    "1_propanol": (370.3, 0.5),
    "2_propanol": (355.5, 0.4),
    "1_butanol": (390.6, 0.8),
    "acetone": (329.3, 0.3),
    "dimethyl_ether": (248.2, 0.0),
    "diethyl_ether": (307.7, 0.4),
    "formic_acid": (373.9, 0.5),
    "acetic_acid": (391.2, 0.6),
    "ethyl_acetate": (350.2, 0.2),
    "chloroform": (334.3, 0.2),
    "carbon_tetrachloride": (349.8, 0.3),
    "1_butene": (266.8, 0.5),
    "2_3_dimethylpentane": (362.9, 0.3),
    "2_4_dimethylpentane": (353.7, 0.3),
    "2_2_5_trimethylhexane": (397.0, 2.0),
    "thiophene": (357.3, 0.6),
}

#: CO2 has no normal boiling point (it sublimes at 1 atm); its liquid Antoine
#: set is tested against PSAT_NIST instead.
NO_TB = {"carbon_dioxide"}

PSAT_NIST = [  # (species, T K, Psat bar, rel tol)
    ("carbon_dioxide", 216.6, 5.18145, 0.01),
    ("carbon_dioxide", 250.0, 17.8504, 0.01),
    ("carbon_dioxide", 280.0, 41.6074, 0.01),
    ("propane", 300.0, 9.97681, 0.01),
    ("ammonia", 300.0, 10.6112, 0.01),
    ("n_butane", 300.0, 2.57596, 0.01),
    ("propylene", 300.0, 12.0858, 0.01),
    ("water", 298.15, 0.0316993, 0.01),
    # Prydz-Goodwin's one set spans 91-190 K (to near Tc): it is 2.4% high at
    # its worst, mid-range; a three-constant fit cannot do better there.
    ("methane", 150.0, 10.3996, 0.025),
]


def _implied_tb(species):
    A, B, C = get_species_data(species).antoine_coeffs
    return B / (A - math.log10(101325.0)) - C


def test_every_species_has_a_reference_or_is_excused():
    names = set(_IDEAL_THERMO_DATA)
    covered = set(TB_REFERENCE_EOS) | set(TB_WEBBOOK) | NO_TB
    assert names <= covered, sorted(names - covered)


@pytest.mark.parametrize("species", sorted(set(TB_REFERENCE_EOS) | set(TB_WEBBOOK)))
def test_implied_normal_boiling_point(species):
    if species in TB_REFERENCE_EOS:
        ref, unc = TB_REFERENCE_EOS[species], 0.0
    else:
        ref, unc = TB_WEBBOOK[species]
    tol = max(0.5, unc)
    assert _implied_tb(species) == pytest.approx(ref, abs=tol)


@pytest.mark.parametrize("species,T,P_bar,rtol",
                         PSAT_NIST)
def test_vapor_pressure_against_nist(species, T, P_bar, rtol):
    th = IdealThermo({species: get_species_data(species)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", AntoineRangeWarning)  # all in range
        P = float(th.Psat(species, T))
    assert P / 1e5 == pytest.approx(P_bar, rel=rtol)


@pytest.mark.parametrize("species", sorted(_IDEAL_THERMO_DATA))
def test_every_species_has_a_finite_range_and_a_source(species):
    d = get_species_data(species)
    assert 0.0 < d.T_antoine_min < d.T_antoine_max < 2000.0
    assert ANTOINE_SOURCES[species].strip()


def test_sources_cover_exactly_the_ideal_species():
    assert set(ANTOINE_SOURCES) == set(_IDEAL_THERMO_DATA)


class TestListSpecies:
    def test_default_lists_only_names_get_species_data_accepts(self):
        for s in list_species():
            get_species_data(s)  # must not raise

    def test_critical_names_all_have_critical_props(self):
        for s in list_species("critical"):
            get_critical_props(s)

    def test_any_is_the_union(self):
        assert set(list_species("any")) == (set(list_species())
                                            | set(list_species("critical")))

    def test_bad_kind(self):
        with pytest.raises(ValueError):
            list_species("bogus")


def test_ethyl_acetate_recorded_range_against_wagner():
    """Ethyl acetate's range is extended past the Polak-Mertl fit (to 493.15 K)
    on a 1% check against the McGarry (1983) Wagner equation, as tabulated in
    the `chemicals` package (v1.5.2); NIST has no reference-EOS table for it."""
    import numpy as np

    A, B, C, D, Tc, Pc = -7.68521, 1.36511, -4.0898, -1.75342, 523.2, 3830100.0

    def wagner(T):
        t = 1.0 - T / Tc
        return Pc * np.exp((A * t + B * t**1.5 + C * t**3 + D * t**6) / (T / Tc))

    data = get_species_data("ethyl_acetate")
    th = IdealThermo({"ethyl_acetate": data})
    Ts = np.linspace(data.T_antoine_min, data.T_antoine_max, 60)
    P = np.array([float(th.Psat("ethyl_acetate", T)) for T in Ts])
    assert np.max(np.abs(P / wagner(Ts) - 1.0)) < 0.01
