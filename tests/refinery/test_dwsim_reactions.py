"""difflow_refinery's reaction thermochemistry against DWSIM 9.0.5.

DWSIM ran in the generator (``reference/dwsim_reactions_generate.py``) and
wrote ``reference/dwsim_reactions_reference.json``; this file reads it.
DWSIM, .NET and pythonnet are not needed here.

DWSIM has no hydrotreater, FCC, reformer or alkylation kinetic model. What is
compared is the physics those units rest on -- formation data, heats of
reaction, equilibria and energy balances -- with DWSIM's equilibrium, Gibbs
and conversion reactors, two ways (``reference/dwsim_reactions_case.py``):

* ``hypo`` -- DWSIM's reactors on hypothetical compounds carrying difflow's
  own H_f, S (as G_f), Cp and critical constants: the IMPLEMENTATION. Every
  difference is reproduced by :mod:`._dwsim_rx_emulation`, which writes
  DWSIM's conventions down (Cp integrals by the midpoint rule, R = 8.314,
  pressures over 1 atm rather than difflow's 1 bar).
* ``dwsim`` -- DWSIM's database compounds: the DATA (ChemSep and friends
  against difflow's API TDB / Yaws / NIST / CODATA / CRC tables). These
  differences are pinned at their measured size, not tolerated away: a test
  fails if they MOVE, which means a data or model change on one side.

DWSIM's own reactors are checked before they are believed: an equilibrium
answer counts only if it IS the equilibrium of DWSIM's own numbers
(:func:`._dwsim_rx_compare.acceptance`); the ones that are not are listed in
:data:`DWSIM_REACTOR_MISSES`.

``release`` throughout but for :class:`TestReferenceFile` and
:class:`TestReferenceIsCurrent` (per commit: the file is intact and difflow's
constants are still the ones it was built on).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from . import _dwsim_rx_compare as cmp
from . import _dwsim_rx_emulation as em
from .reference import dwsim_reactions_case as rc

REF = cmp.ref()
REGENERATE = ("difflow's thermochemistry (or the cases) no longer match what the DWSIM reaction "
              "reference was built on; regenerate it: PYTHONPATH=src:tests python -m "
              "refinery.reference.dwsim_reactions_generate (needs DWSIM 9.0.5; see "
              "scripts/install_dwsim.sh)")


def _runs(node):
    """Every reactor run (a dict with ``errors`` and ``names``) under ``node``."""
    if isinstance(node, dict):
        if "errors" in node and "names" in node:
            yield node
        for v in node.values():
            yield from _runs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _runs(v)


def _max_dx(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------
# Per commit
# ---------------------------------------------------------------------------


class TestReferenceFile:
    def test_provenance(self):
        p = REF["provenance"]
        for key in ("generated", "script", "command", "dwsim", "dotnet", "pythonnet",
                    "reference_simulator", "difflow_commit"):
            assert p[key], key
        assert "9.0.5" in p["dwsim"]

    def test_sections(self):
        for key in ("species", "hypo_species", "isomerization", "reformer", "hydroprocessing",
                    "alkylation", "fcc", "inputs", "cases"):
            assert key in REF, key

    def test_errors_are_only_dwsims_equilibrium_reactor(self):
        """DWSIM raised only from its equilibrium reactor ("negative mole
        fractions" on the multi-reaction isomerization cases and one benzene
        case; a flash failure in adiabatic mode). Recorded, not hidden."""
        for sec in ("reformer", "alkylation", "fcc"):
            assert [r["errors"] for r in _runs(REF[sec]) if r["errors"]] == [], sec
        for sec in ("isomerization", "hydroprocessing"):
            for r in _runs(REF[sec]):
                if r["errors"]:
                    assert r.get("kind") == "equilibrium", r["errors"]

    def test_ideal_gas_runs_stay_vapour(self):
        """The Psat-shifted Raoult package never forms a liquid."""
        tree = {"isom": {k: v for k, v in REF["isomerization"].items()},
                "reformer": REF["reformer"]["equilibria"], "hdt": REF["hydroprocessing"],
                "fcc": REF["fcc"]}
        tree["isom"]["adiabatic"] = {k: v for k, v in tree["isom"]["adiabatic"].items()
                                     if k != "dwsim_pr"}
        for r in _runs(tree):
            if not r["errors"]:
                assert sum(r["F_liquid"]) < 1e-12

    def test_missing_compounds_are_the_known_ones(self):
        missing = {k for k, v in rc.HDT_DW.items() if v is None}
        assert missing == {"dibenzothiophene", "cyclohexylbenzene", "quinoline", "carbazole",
                           "tetrahydrophenanthrene"}
        for name, v in REF["hydroprocessing"]["heats"].items():
            if "missing_in_dwsim" in v:
                assert set(v["missing_in_dwsim"]) <= missing, name


class TestReferenceIsCurrent:
    """difflow's constants are an INPUT to both sides; if they move,
    regenerate -- never loosen."""

    @pytest.mark.parametrize("fn", ["isom_constants", "reformer_constants", "hdt_constants",
                                    "alky_constants", "fcc_constants"])
    def test_constants(self, fn):
        key = {"isom_constants": "isom", "reformer_constants": "reformer", "hdt_constants": "hdt",
               "alky_constants": "alky", "fcc_constants": "fcc"}[fn]
        now = json.loads(json.dumps(getattr(rc, fn)(), default=float))
        assert now == REF["inputs"][key], REGENERATE

    def test_cases(self):
        from .reference import isom_generate as ig

        for k, v in REF["cases"].items():
            assert json.loads(json.dumps(getattr(rc, k))) == v, (k, REGENERATE)
        charges = {k: ig.adiabatic_charge(k) for k in ig.ADIABATIC["feeds"]}
        assert json.loads(json.dumps(charges)) == REF["inputs"]["isom_cases"]["charges"], REGENERATE


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------

#: (case, which, reactor) whose answer is NOT DWSIM's own equilibrium
#: (:func:`._dwsim_rx_compare.acceptance`). DWSIM's equilibrium reactor:
#: "negative mole fractions" on the isomerization ring and benzene at
#: 573 K/100 bar; a silent non-convergence on the C6 paraffins at 400 K
#: (6.3e-2 off); naphthalene/tetralin on DWSIM's data, where tetralin's G_f
#: is missing (zero). DWSIM's Gibbs reactor (DirectMinimization): a minor
#: species left at zero or at its trace start (MCH at 773 K,
#: 2-methylhexane, benzene, naphthalene), 5e-6 to 1e-2 off, with no error.
DWSIM_REACTOR_MISSES = {
    ("aromatics benzene 573.15 K 100 bar", "dwsim", "equilibrium"),
    ("aromatics benzene 573.15 K 30 bar", "dwsim", "gibbs"),
    ("aromatics benzene 623.15 K 100 bar", "hypo", "gibbs"),
    ("aromatics benzene 623.15 K 30 bar", "hypo", "gibbs"),
    ("aromatics benzene 693.15 K 100 bar", "dwsim", "gibbs"),
    ("aromatics naphthalene 573.15 K 100 bar", "dwsim", "equilibrium"),
    ("aromatics naphthalene 573.15 K 100 bar", "hypo", "gibbs"),
    ("aromatics naphthalene 573.15 K 30 bar", "dwsim", "equilibrium"),
    ("aromatics naphthalene 573.15 K 30 bar", "hypo", "gibbs"),
    ("aromatics naphthalene 623.15 K 100 bar", "dwsim", "equilibrium"),
    ("aromatics naphthalene 623.15 K 30 bar", "dwsim", "equilibrium"),
    ("aromatics naphthalene 693.15 K 100 bar", "dwsim", "equilibrium"),
    ("aromatics naphthalene 693.15 K 100 bar", "hypo", "gibbs"),
    ("aromatics naphthalene 693.15 K 30 bar", "dwsim", "equilibrium"),
    ("isom C6P 400 K", "hypo", "equilibrium"),
    ("isom c6_ring 420 K", "dwsim", "equilibrium"),
    ("isom c6_ring 420 K", "hypo", "equilibrium"),
    ("isom c6_ring 480 K", "dwsim", "equilibrium"),
    ("isom c6_ring 480 K", "hypo", "equilibrium"),
    ("reformer c7_dehydrocyclization 773.15 K 10 bar", "dwsim", "gibbs"),
    ("reformer c7_dehydrocyclization 773.15 K 25 bar", "dwsim", "gibbs"),
    ("reformer mch_toluene 773.15 K 10 bar", "dwsim", "gibbs"),
    ("reformer mch_toluene 773.15 K 10 bar", "hypo", "gibbs"),
    ("reformer mch_toluene 773.15 K 25 bar", "dwsim", "gibbs"),
}

LABELS = sorted({(lab, w) for lab, w, *_ in cmp._cases()})


@pytest.mark.release
class TestHowDWSIMComputes:
    """DWSIM's conventions, reproduced: what makes the hypo runs agree."""

    def test_formation_functions_of_the_hypos(self):
        """DWSIM's ``int Cp dT``, ``int Cp/T dT`` and ``G_f(T)`` of difflow's
        Cp cubics are the midpoint rule of :mod:`._dwsim_rx_emulation`:
        measured 8.7e-11 J/mol, 1.7e-13 J/mol/K, 2.3e-10 J/mol."""
        for key in ("isom", "reformer"):
            cons = REF["inputs"][key]
            for k, tab in REF["hypo_species"][key].items():
                for Ts, v in tab.items():
                    T, c = float(Ts), cons[k]
                    assert v["h_sens"] == pytest.approx(em.int_cp(c["cp"], T, "midpoint"), abs=1e-8)
                    assert v["s_sens"] == pytest.approx(em.int_cp_over_T(c["cp"], T, "midpoint"),
                                                        abs=1e-10)
                    g = c["Hf"] + v["h_sens"] - T * ((c["Hf"] - v["Gf"]) / em.T0 + v["s_sens"])
                    assert v["g_f"] == pytest.approx(g, abs=1e-7)

    def test_the_emulation_is_difflow(self):
        """Under difflow's conventions the emulation IS difflow: the isomer
        families (closed form, 1e-12) and the IDAES adiabatic reference,
        which difflow's reactor matches to 1e-6 K
        (``test_isomerization_validation.py``)."""
        from pathlib import Path

        from difflow_refinery.isomerization import thermochem as tc

        K = REF["inputs"]["isom"]
        for fam, names in tc.FAMILIES.items():
            for T in (400.0, 550.0):
                A = [[K[x]["C"] for x in names], [K[x]["H"] for x in names]]
                n = em.species_equilibrium(K, names, A, [1.0] * len(names), T, 30e5, em.DIFFLOW)
                ref = tc.family_equilibrium(fam, T)
                assert _max_dx(n / n.sum(), [float(ref[x]) for x in names]) < 1e-12
        idaes = json.loads((Path(__file__).parent / "reference" / "isom_reference.json").read_text())
        for kind, case in idaes["adiabatic"].items():
            run = REF["isomerization"]["adiabatic"]["hypo"][kind]
            feed, rn = run["feed"], run["reacting"]
            inert = {k: v for k, v in feed.items() if k not in rn}
            T, _ = em.adiabatic(K, rn, run["element_rows"], [feed[k] for k in rn], inert, 413.15,
                                30e5, em.DIFFLOW)
            assert T == pytest.approx(case["T"], abs=1e-7)

    def test_which_answers_are_dwsims_own_equilibrium(self):
        """Every isothermal case has a DWSIM answer that is the equilibrium of
        DWSIM's own numbers, to 3e-6 (measured 2.1e-6 worst: the C6 ring at
        420 K on DWSIM's data, Gibbs reactor; the rest 1.8e-6 or better). The
        misses are exactly :data:`DWSIM_REACTOR_MISSES`."""
        acc = cmp.acceptance()
        assert {k for k, v in acc.items() if not v[0]} == DWSIM_REACTOR_MISSES
        for lab, w in LABELS:
            x, _ = cmp.accepted(lab, w)
            assert x is not None, (lab, w)
        assert max(v[1] for v in acc.values() if v[0]) < 2.2e-6


@pytest.mark.release
class TestImplementation:
    """(a): DWSIM's reactors on difflow's own constants."""

    @pytest.mark.parametrize("label", [lab for lab, w in LABELS if w == "hypo"])
    def test_equilibria(self, label):
        """The accepted DWSIM answer against difflow's. Every difference is
        DWSIM's conventions (midpoint Cp integrals, R, 1 atm): the same
        answer is the emulation under those conventions to 3e-6. Measured
        against difflow: isomer families 1.6e-5 (quadrature only; no moles
        change), the C6 ring 7e-6; the reformer and aromatics, where moles
        change, up to 1.1e-3 (mostly the 1 atm standard state, (1.01325)^dn
        on K)."""
        x, _ = cmp.accepted(label, "hypo")
        xd, _ = cmp.difflow_equilibrium(label)
        lim = 2.5e-5 if label.startswith("isom") else 1.5e-3
        assert _max_dx(x, xd) < lim

    def test_isomerization_adiabatic(self):
        """DWSIM's adiabatic Gibbs reactor on difflow's constants: 477.032 and
        522.304 K against difflow's 477.023 and 522.225 K. DWSIM's answer is
        9 mK and 0.10 K off ITS OWN model (the emulation under DWSIM's
        conventions: 477.023 and 522.200 K): its minimisation with inert
        species stops 2e-5 and 1e-4 (mole fraction) short of equilibrium,
        isothermally too (``isothermal_at_T_ad``)."""
        for kind, T_df, tol in (("paraffinic", 477.023118602291, 0.02),
                                ("benzene_rich", 522.22456099641, 0.12)):
            run = REF["isomerization"]["adiabatic"]["hypo"][kind]["gibbs"]
            assert run["T"] == pytest.approx(T_df, abs=tol)

    def test_reformer_bed_energy_balance(self):
        """DWSIM's conversion reactor, adiabatic, Peng-Robinson on difflow's
        constants, taken to difflow's bed outlet: 710.6506 K against
        difflow's 710.6503 K (dT -62.4994 against -62.4997 K). The 0.3 mK is
        DWSIM's R = 8.314 in the PR departure and its midpoint Cp integral."""
        b = cmp.reformer_bed()
        assert b["hypo"]["flow_error"] < 1e-12
        assert b["hypo"]["liquid"] == 0.0
        assert b["hypo"]["T_out"] == pytest.approx(b["T_out_difflow"], abs=1e-3)

    def test_fcc_regenerator_closure(self):
        """The coke burn: difflow's heat minus DWSIM's is exactly the formation
        and sensible-heat differences of the flue species (DWSIM's ChemSep
        against difflow's CODATA/RPP data), to 1 W in 21-39 MW: the reactor
        bookkeeping agrees, only the data differ."""
        f = REF["inputs"]["fcc"]
        D = rc.FCC_DW
        for r_co, rows in cmp.fcc_compare().items():
            flue = f["burn"][r_co]["flue"]
            for label, row in rows.items():
                T = float(label)
                pred = sum(n * (cmp.fcc_h(k, T) - cmp.dw_species(D[k], T)["Hf"]
                                - cmp.dw_species(D[k], T)["h_sens"])
                           for k, n in flue.items() if n)
                assert row["Q_difflow"] - row["Q_dwsim"] == pytest.approx(pred, abs=1.0)
                for k, n in ((k, n) for k, n in flue.items() if n):
                    assert row["flue_dwsim"][D[k]] == pytest.approx(n, rel=2e-6, abs=1e-9)


@pytest.mark.release
class TestData:
    """(b): DWSIM's database against difflow's tables, pinned at the measured
    size; a test fails if a difference MOVES."""

    def test_isomerization_families(self):
        """DWSIM's ChemSep data put isopentane at 0.762 of the C5s at 450 K
        against difflow's 0.820; the C6 paraffins and naphthenes differ by up
        to 0.067 and 0.052. Cause: dH of nC5 = iC5 is -6.94 kJ/mol on
        ChemSep, -8.10 in difflow (Prosen & Rossini); dG(450 K) of 2MP =
        23DMB 4.57 against 3.22 kJ/mol."""
        x, _ = cmp.accepted("isom C5 450 K", "dwsim")
        xd, _ = cmp.difflow_equilibrium("isom C5 450 K")
        assert xd[1] == pytest.approx(0.82001, abs=1e-5)
        assert x[1] == pytest.approx(0.76239, abs=1e-5)
        for fam, worst in (("C5", 0.0584), ("C6P", 0.0670), ("C6N", 0.0524)):
            m = max(_max_dx(cmp.accepted(lab, "dwsim")[0], cmp.difflow_equilibrium(lab)[0])
                    for lab, w in LABELS if w == "dwsim" and lab.startswith(f"isom {fam} "))
            assert m == pytest.approx(worst, abs=2e-4), fam

    def test_isomerization_adiabatic(self):
        """On DWSIM's data the adiabatic outlet is 473.19 and 519.58 K, 3.83
        and 2.65 K cooler than difflow's: less heat of isomerization (the
        ChemSep dH above). DWSIM's own convergence (<= 0.1 K) included."""
        for kind, d in (("paraffinic", -3.83), ("benzene_rich", -2.65)):
            T = REF["isomerization"]["adiabatic"]["dwsim"][kind]["gibbs"]["T"]
            T_df = REF["isomerization"]["isothermal_at_T_ad"]["dwsim"][kind]["T_set"]
            assert T - T_df == pytest.approx(d, abs=0.02)

    def test_reformer_reaction_thermochemistry(self):
        """Per reaction, difflow's reformer thermochemistry (API TDB H_f, Yaws
        S, TRC Cp fits) against DWSIM's ChemSep: dH(298 K) within 0.7 kJ/mol
        (ring expansion MCP = CH the largest), dH(773 K) within 1.4 kJ/mol,
        ln K at 773 K within 0.21 on one standard state (nP8 = A8 + 4 H2
        0.20, the C9 ring's dehydrogenation 0.13 the next)."""
        for name, row in cmp.reformer_reactions().items():
            a, b = row["298.15"], row["773.15"]
            assert abs(a["dH_difflow"] - a["dH_dwsim"]) < 700.0, name
            assert abs(b["dH_difflow"] - b["dH_dwsim"]) < 1400.0, name
            dlnK = b["lnK_difflow"] - (b["lnK_dwsim"] - b["dn"] * np.log(em.P0_DWSIM / em.P0_DIFFLOW))
            assert abs(dlnK) < 0.21, (name, dlnK)

    @pytest.mark.parametrize("label", [lab for lab, w in LABELS
                                       if w == "dwsim" and lab.startswith("reformer")])
    def test_reformer_equilibria(self, label):
        """DWSIM's data against difflow's at 700-773 K, 10-25 bar: within
        3.1e-3 in mole fraction (C6 rings at 700 K: the MCP/CH split)."""
        x, _ = cmp.accepted(label, "dwsim")
        assert _max_dx(x, cmp.difflow_equilibrium(label)[0]) < 3.2e-3

    def test_reformer_bed(self):
        """The same bed outlet on DWSIM's data: 711.163 K, 0.51 K warmer than
        difflow's (ChemSep's H_f and Cp make the bed 0.8 % less
        endothermic)."""
        b = cmp.reformer_bed()
        assert b["dwsim"]["flow_error"] < 1e-7
        assert b["dwsim"]["T_out"] - b["T_out_difflow"] == pytest.approx(0.512, abs=0.01)

    def test_hydroprocessing_heats(self):
        """Heats per mol H2 at 298 K within 1.9 kJ/mol of DWSIM for every
        model reaction DWSIM can form, but benzothiophene: ChemSep's H_f is
        137.0 kJ/mol against difflow's 166.3, so its HDS releases 42.6 kJ/mol
        H2 in DWSIM and 52.3 in difflow."""
        for name, row in cmp.hdt_heats().items():
            if "missing_in_dwsim" in row:
                continue
            d = row["per_H2_difflow"] - row["per_H2_dwsim"]
            if "benzothiophene" in name:
                assert d == pytest.approx(-9.76e3, abs=50.0)
            else:
                assert abs(d) < 1.9e3, (name, d)

    def test_hydroprocessing_heats_at_reactor_temperature(self):
        """difflow's per-class heats are 298 K values; DWSIM's conversion
        reactor at 350 C gives every model reaction 3 % to 15 % MORE heat
        (the reaction's dCp): HDS of thiophene -277 against -262 kJ/mol."""
        for name, row in cmp.hdt_heats().items():
            if "missing_in_dwsim" in row:
                continue
            ratio = row["dHT_dwsim"] / row["dH298_dwsim"]
            assert 1.03 < ratio < 1.16, (name, ratio)

    def test_aromatics_saturation_constant(self):
        """difflow's hydrotreating K for benzene + 3 H2 = cyclohexane takes dH
        and dS at 298 K as constant. With heat capacities -- DWSIM's ChemSep
        data, and difflow's OWN reformer thermochemistry -- ln K is lower by
        1.05 at 300 C, 1.29 at 350 C, 1.60 at 420 C (K 2.9x, 3.6x, 4.9x too
        large); those two agree with each other to 0.02 on one standard
        state."""
        import jax.numpy as jnp

        from difflow_refinery.reforming import species as sp
        from difflow_refinery.reforming import thermo as th

        v = np.zeros(sp.N_SPECIES)
        v[sp.INDEX["A6"]], v[sp.INDEX["H2"]], v[sp.INDEX["N6"]] = -1, -3, 1
        for T, gap in ((573.15, 1.05), (623.15, 1.29), (693.15, 1.60)):
            hdt = cmp.difflow_aromatic_lnK(2, T)
            dw = cmp.dw_reaction({"Benzene": -1, "Hydrogen": -3, "Cyclohexane": 1}, T)["lnK"]
            dw_bar = dw - 3 * np.log(em.P0_DWSIM / em.P0_DIFFLOW)
            ref_ = float(th.ln_K(jnp.asarray(v), T))
            assert abs(dw_bar - ref_) < 0.02
            assert hdt - ref_ == pytest.approx(gap, abs=0.03)

    def test_tetralin_has_no_gibbs_energy_in_dwsim(self):
        """DWSIM's 1,2,3,4-tetrahydronaphthalene (ChEDL Thermo) has H_f but
        G_f = 0, so DWSIM's naphthalene-saturation equilibrium is meaningless
        (ln K about 60). On difflow's constants DWSIM reproduces difflow's."""
        assert REF["species"]["1,2,3,4-Tetrahydronaphthalene"]["T"]["298.15"]["Gf"] == 0.0
        dw = cmp.dw_reaction({"Naphthalene": -1, "Hydrogen": -2,
                              "1,2,3,4-Tetrahydronaphthalene": 1}, 623.15)
        assert dw["lnK"] > 50.0

    def test_alkylation_heats(self):
        """Liquid heats of alkylation at 25 C: DWSIM (Peng-Robinson liquid,
        ChemSep H_f) is 1.4 to 6.2 kJ/mol LESS exothermic than difflow
        (H_f(g) - CRC Hvap). Of that, the gas-phase H_f differ by up to 3.7
        kJ/mol (propylene route); the rest is the heats of vaporisation (PR
        liquid departure against CRC). difflow's reactor heats (route A) are
        its species table exactly."""
        for name, row in cmp.alky_heats().items():
            d_liq = row["dH_liquid_difflow"] - row["dH_liquid_dwsim_298.15"]
            assert -6.3e3 < d_liq < -1.3e3, (name, d_liq)
            assert abs(row["dH_gas_difflow"] - row["dH_gas_dwsim"]) < 3.8e3, name
            assert row["vapor_out_298.15"] == 0.0
            if "difflow_reactor_dH_A" in row:
                assert row["difflow_reactor_dH_A"] == pytest.approx(row["dH_liquid_difflow"], rel=1e-12)

    def test_fcc_regenerator(self):
        """Coke (7 wt% H) burnt to 2 % O2: the heat of combustion at 25 C
        agrees to 1.1e-5 (water's H_f, -241.826 against -241.814 kJ/mol);
        to a 700-730 C flue DWSIM releases 0.05-0.07 % more (RPP Cp fits
        against ChemSep's)."""
        for r_co, rows in cmp.fcc_compare().items():
            a = rows["298.15"]
            assert abs(a["Q_difflow"] - a["Q_dwsim"]) / abs(a["Q_difflow"]) < 2e-5
            for label in ("973.15", "1003.15"):
                b = rows[label]
                rel = (b["Q_difflow"] - b["Q_dwsim"]) / abs(b["Q_difflow"])
                assert 4e-4 < rel < 8e-4, (r_co, label, rel)


@pytest.mark.release
@pytest.mark.slow
def test_reformer_bed_is_still_difflows():
    """The bed outlet DWSIM was given is still what difflow's bed makes."""
    now = rc.reformer_bed()
    ref = REF["inputs"]["reformer_bed"]
    np.testing.assert_allclose(now["F_out"], ref["F_out"], rtol=1e-8, atol=1e-10, err_msg=REGENERATE)
    assert now["T_out"] == pytest.approx(ref["T_out"], abs=1e-6), REGENERATE
