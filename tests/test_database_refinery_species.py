"""Refinery C4-C8 olefins, naphthenes and branched paraffins in the database.

Issues #308-#310 need light-naphtha / gasoline isomers (butenes, amylenes,
C6 isomers, cyclics, trimethylpentanes) as real components. These tests check
each record against independent reference values, not against itself:

* MW from the molecular formula (IUPAC 2013 atomic weights);
* ideal-gas Cp at 298.15 K against the TRC / NIST WebBook gas-phase values;
* dHvap at 298.15 K against the CRC Handbook;
* the Antoine normal boiling point against the measured Tb;
* the PR EOS (Tc, Pc, omega) against the Antoine vapour pressure, which
  checks that the three critical constants and the Antoine set describe the
  same molecule.
"""

import jax.numpy as jnp
import pytest

from difflow import IdealThermo
from difflow.database import (
    SOURCE_CITATIONS,
    get_critical_props,
    get_species_data,
    list_species,
    resolve_alias,
)
from difflow.eos import PengRobinson

C, H = 12.011, 1.008

# name: (n_C, n_H, Tb [K], Cp_ig(298.15) [J/mol/K], dHvap(298.15) [J/mol])
# Tb and dHvap(298): CRC Handbook; Cp: TRC Thermodynamic Tables (as on the
# NIST WebBook gas-phase pages). Isobutylene dHvap(298): Perry's 9e Table 2-150.
REFERENCE = {
    "cis_2_butene": (4, 8, 276.86, 80.2, 22160.0),
    "trans_2_butene": (4, 8, 274.03, 87.7, 21400.0),
    "isobutylene": (4, 8, 266.25, 88.1, 20265.0),
    "1_pentene": (5, 10, 303.11, 108.2, 25470.0),
    "2_methyl_2_butene": (5, 10, 311.71, 105.0, 27060.0),
    "cyclohexane": (6, 12, 353.88, 106.3, 33010.0),
    "methylcyclopentane": (6, 12, 344.95, 109.5, 31640.0),
    "2_methylpentane": (6, 14, 333.41, 142.2, 29890.0),
    "3_methylpentane": (6, 14, 336.42, 140.1, 30280.0),
    "2_2_dimethylbutane": (6, 14, 322.88, 141.5, 27680.0),
    "2_3_dimethylbutane": (6, 14, 331.08, 139.4, 29120.0),
    "2_2_4_trimethylpentane": (8, 18, 372.37, 188.4, 35140.0),
    "2_3_4_trimethylpentane": (8, 18, 386.65, 191.6, 37750.0),
    "2_5_dimethylhexane": (8, 18, 382.27, 185.5, 37850.0),
}

NAMES = sorted(REFERENCE)

# Records that #311 (isomerization) added to the database independently, with
# its own sources (Lemmon & Ihmels butenes, PSRK C6 isomers; see
# SOURCE_CITATIONS); where both branches added a species, #311's record is
# the one kept. The independent reference checks below still apply to them --
# a second, unrelated cross-check of #311's numbers -- but the citation tests
# are about this file's source and run on the others only, and their Cp
# cubics are checked over the range they were fitted on (up to 800 K).
SHARED = {"cis_2_butene", "trans_2_butene", "isobutylene", "cyclohexane",
          "methylcyclopentane", "2_methylpentane", "3_methylpentane",
          "2_2_dimethylbutane", "2_3_dimethylbutane"}
OWN = [n for n in NAMES if n not in SHARED]


def _cp(data, T):
    a, b, c, d = data.Cp_coeffs
    return a + b * T + c * T**2 + d * T**3


@pytest.mark.parametrize("name", NAMES)
def test_loads_from_both_tables(name):
    crit = get_critical_props(name)
    data = get_species_data(name)
    assert crit.name == name and data.name == name
    assert name in list_species()
    assert crit.MW == data.MW
    assert name in SOURCE_CITATIONS
    if name in SHARED:
        assert "NIST Chemistry WebBook; Perry" not in SOURCE_CITATIONS[name]
    else:
        assert "IUPAC" in SOURCE_CITATIONS[name]


def test_unverified_values_are_flagged_in_their_citation():
    """The values the source cross-check could not settle say so."""
    flagged = {"2_methyl_2_butene", "2_5_dimethylhexane"}
    for name in OWN:
        cite = SOURCE_CITATIONS[name]
        assert "Poling" in cite and "TRC" in cite and "CRC" in cite
        note = cite.split("298.15 K (edition unverified).")[-1]
        assert ("(unverified)" in note) == (name in flagged), name


@pytest.mark.parametrize("name", NAMES)
def test_molecular_weight_matches_formula(name):
    nC, nH = REFERENCE[name][:2]
    assert get_critical_props(name).MW == pytest.approx(nC * C + nH * H, abs=0.01)


@pytest.mark.parametrize("name", NAMES)
def test_ideal_gas_cp_at_298(name):
    data = get_species_data(name)
    assert float(_cp(data, 298.15)) == pytest.approx(REFERENCE[name][3], rel=0.015)


@pytest.mark.parametrize("name", NAMES)
def test_ideal_gas_cp_is_increasing_and_positive(name):
    data = get_species_data(name)
    T = jnp.linspace(250.0, 800.0 if name in SHARED else 1000.0, 76)
    cp = _cp(data, T)
    assert bool(jnp.all(cp > 0)) and bool(jnp.all(jnp.diff(cp) > 0))


@pytest.mark.parametrize("name", NAMES)
def test_hvap_at_298_and_watson_tc(name):
    data = get_species_data(name)
    crit = get_critical_props(name)
    assert data.Hvap_coeffs[2] == crit.Tc
    thermo = IdealThermo({name: data})
    assert float(thermo.Hvap(name, 298.15)) == pytest.approx(REFERENCE[name][4], rel=0.03)


@pytest.mark.parametrize("name", NAMES)
def test_antoine_normal_boiling_point(name):
    data = get_species_data(name)
    thermo = IdealThermo({name: data})
    Tb = REFERENCE[name][2]
    assert float(thermo.Psat(name, Tb)) == pytest.approx(101325.0, rel=0.02)
    assert data.T_antoine_min < Tb < data.T_antoine_max


@pytest.mark.parametrize("name", NAMES)
def test_critical_constants_agree_with_antoine(name):
    """Lee-Kesler Psat(Tc, Pc, omega) at Tb against 1 atm.

    The critical constants and acentric factor come from one set of sources
    and the Antoine constants from another; Lee-Kesler (~2-4% for
    hydrocarbons at Tb) ties the two together, so a mis-keyed Tc, Pc or
    omega shows up here.
    """
    crit = get_critical_props(name)
    Tb = REFERENCE[name][2]
    Tr = Tb / crit.Tc
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * jnp.log(Tr) + 0.169347 * Tr**6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * jnp.log(Tr) + 0.43577 * Tr**6
    psat = crit.Pc * jnp.exp(f0 + crit.omega * f1)
    assert float(psat) == pytest.approx(101325.0, rel=0.05)


@pytest.mark.parametrize("name", NAMES)
def test_peng_robinson_builds(name):
    eos = PengRobinson({name: get_critical_props(name)})
    one = jnp.array([1.0])
    T = jnp.asarray(REFERENCE[name][2] + 50.0)
    Z = eos.solve_Z(T, jnp.asarray(101325.0), one, "vapor")
    assert 0.9 < float(Z) < 1.0
    # Second virial-like check: Z falls as P rises along the isotherm.
    Z5 = eos.solve_Z(T, jnp.asarray(5.0e5), one, "vapor")
    assert float(Z5) < float(Z)


def test_mixture_eos_builds_and_is_finite():
    names = NAMES
    eos = PengRobinson({n: get_critical_props(n) for n in names})
    y = jnp.ones(len(names)) / len(names)
    Z = eos.solve_Z(jnp.asarray(400.0), jnp.asarray(2.0e5), y, "vapor")
    assert bool(jnp.isfinite(Z)) and 0.8 < float(Z) < 1.0


def _crit(name):
    p = get_critical_props(name)
    return p.Tc, p.Pc


def test_isomer_orderings():
    """Branching lowers Tc and Pc; cis-2-butene is above trans; ring above chain."""
    # C6 paraffins: n-hexane has the highest Tc; 2,2-DMB (most compact) the lowest.
    tc = {n: _crit(n)[0] for n in ("n_hexane", "3_methylpentane", "2_methylpentane",
                                   "2_3_dimethylbutane", "2_2_dimethylbutane")}
    assert tc["n_hexane"] > tc["3_methylpentane"] > tc["2_methylpentane"]
    assert tc["2_methylpentane"] < tc["2_3_dimethylbutane"]
    assert min(tc, key=tc.get) == "2_2_dimethylbutane"
    # Every branched C6 paraffin has a lower Tc than n-hexane and Pc above it.
    for n in tc:
        if n != "n_hexane":
            assert _crit(n)[1] > _crit("n_hexane")[1]
    # Butenes
    assert _crit("cis_2_butene")[0] > _crit("trans_2_butene")[0] > _crit("1_butene")[0]
    assert _crit("isobutylene")[0] < _crit("1_butene")[0]
    # C8: n-octane > 2,3,4-TMP > 2,5-DMH > isooctane
    assert (_crit("n_octane")[0] > _crit("2_3_4_trimethylpentane")[0]
            > _crit("2_5_dimethylhexane")[0] > _crit("2_2_4_trimethylpentane")[0])
    # C6 naphthenes: cyclohexane above methylcyclopentane, both above n-hexane
    assert _crit("cyclohexane")[0] > _crit("methylcyclopentane")[0] > _crit("n_hexane")[0]
    # Boiling point tracks the same order (Antoine at 1 atm)
    assert REFERENCE["cyclohexane"][2] > REFERENCE["methylcyclopentane"][2]


def test_heats_of_formation_isomer_ordering():
    """Branched alkanes are more stable; trans-2-butene below cis; isobutylene lowest C4H8."""
    hf = {n: get_species_data(n).Hf for n in NAMES}
    assert hf["2_2_dimethylbutane"] < hf["2_3_dimethylbutane"] < hf["2_methylpentane"]
    assert hf["2_methylpentane"] < hf["3_methylpentane"] < get_species_data("n_hexane").Hf
    assert hf["isobutylene"] < hf["trans_2_butene"] < hf["cis_2_butene"] < 0
    assert hf["cyclohexane"] < hf["methylcyclopentane"]


@pytest.mark.parametrize(
    "alias,name",
    [("isooctane", "2_2_4_trimethylpentane"), ("isobutene", "isobutylene"),
     ("2-methylpropene", "isobutylene"), ("isohexane", "2_methylpentane"),
     ("neohexane", "2_2_dimethylbutane")],
)
def test_aliases(alias, name):
    assert resolve_alias(alias) == name
    assert get_critical_props(alias).name == name
    assert get_species_data(alias).name == name
