"""The refinery's one table of ideal-gas formation data (#339).

Three things are checked here:

* the GUARD: no difflow_refinery module carries its own copy of formation
  data (Hf, S0, an ideal-gas Cp table) for a species in
  :mod:`difflow_refinery.thermochemistry`, and the modules that expose such
  data expose the table's numbers;
* the PINS: chosen values against the compilations they cite, as read from
  the ``chemicals`` 1.5.2 data files when the table was built (written out
  here so a changed table value fails against its source, not against
  itself);
* the API: closed-form integrals, reactions, equilibrium constants, JAX.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import thermochemistry as tc

jax.config.update("jax_enable_x64", True)

PKG = Path(tc.__file__).resolve().parent
#: The commit #339 started from (every private copy still in place).
PRE_339 = "d50001f"

# =============================================================================
# The guard
# =============================================================================

#: Modules that may still hold their own formation data, and why. Shrinking
#: this is the point; growing it needs a reason a reviewer accepts.
ALLOWED = {
    # Separation thermo, not reaction thermochemistry: the ideal-gas Cp of
    # the Peng-Robinson enthalpy of the gas plant and of the hydroprocessing
    # separators, pinned by the IDAES and DWSIM gas-plant / HP-separator
    # references (same constants on both sides). No Hf, no S0.
    "gasplant/components.py": {"CP_IG"},
    "hydroprocessing/thermo.py": {"_CP_IG_GASES"},
}

#: Parameter / field names that carry formation data in a constructor.
_FORMATION_PARAMS = {"Hf", "Hf_gas", "dHf", "S0", "S", "cp"}


def _is_formation_name(name: str) -> bool:
    u = name.upper()
    parts = [p for p in u.split("_") if p]
    return (any(p in ("HF", "DHF", "HFG", "S0", "S298") for p in parts)
            or "MODEL_COMPOUND" in u or "FORMATION" in u)


def _is_cp_table_name(name: str) -> bool:
    u = name.upper()
    return u in ("_CP", "CP") or "CP_IG" in u or "CP_GAS" in u


def _nonzero_numbers(node) -> int:
    """Nonzero numeric literals in an expression, or 0 if it is computed (a
    call, attribute or comprehension: a value read from somewhere, e.g. the
    table, is not a copy). Bare names do not make it computed."""
    n = 0
    for sub in ast.walk(node):
        if isinstance(sub, (ast.Call, ast.Attribute, ast.DictComp, ast.ListComp,
                            ast.GeneratorExp, ast.SetComp)):
            return 0
        if isinstance(sub, ast.Constant) and isinstance(sub.value, (int, float)) \
                and not isinstance(sub.value, bool) and sub.value != 0:
            n += 1
    return n


def _species_keys(node) -> set:
    keys = set()
    if isinstance(node, ast.Dict):
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                try:
                    keys.add(tc.resolve(k.value))
                except KeyError:
                    pass
    return keys


def _signatures(tree) -> dict:
    """Positional parameter names of the module's functions and dataclasses
    that take formation data (a parameter called Hf, Hf_gas or dHf)."""
    sig = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names = [a.arg for a in node.args.args]
        elif isinstance(node, ast.ClassDef):
            names = [s.target.id for s in node.body
                     if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
        else:
            continue
        if {"Hf", "Hf_gas", "dHf"} & set(names):
            sig[node.name] = names
    return sig


def _copies(path: Path) -> list:
    """(line, kind, name) of every formation-data copy in one module."""
    tree = ast.parse(path.read_text())
    sig = _signatures(tree)
    imports_db_data = any(isinstance(n, ast.ImportFrom) and any(a.name == "get_species_data"
                                                                for a in n.names)
                          for n in ast.walk(tree))
    found = []
    for node in ast.walk(tree):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        for t in targets:
            if not isinstance(t, ast.Name):
                continue
            literal = _nonzero_numbers(value) > 0
            if _is_formation_name(t.id) and literal:
                found.append((node.lineno, "Hf/S0 literal", t.id))
            elif _is_cp_table_name(t.id) and literal and _species_keys(value):
                found.append((node.lineno, "Cp table", t.id))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in _FORMATION_PARAMS - {"S", "cp"} and _nonzero_numbers(kw.value) > 0:
                    found.append((node.lineno, f"{kw.arg}= literal", kw.arg))
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute)
                                                          else "")
            # a constructor of this module that takes formation data, given literals
            for p, a in zip(sig.get(fname, ()), node.args):
                if p in _FORMATION_PARAMS and _nonzero_numbers(a) > 0:
                    found.append((node.lineno, f"{p} literal in {fname}()", fname))
            # reading difflow.database's formation data instead of the table
            if isinstance(f, ast.Attribute) and f.attr == "get" and node.args \
                    and isinstance(node.args[0], ast.Constant) and node.args[0].value == "Hf":
                found.append((node.lineno, "reads a database Hf", "get('Hf')"))
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and node.slice.value == "Hf":
            found.append((node.lineno, "reads a database Hf", "['Hf']"))
        if isinstance(node, ast.Attribute) and node.attr == "Hf" and imports_db_data:
            found.append((node.lineno, "reads a database Hf", "get_species_data(...).Hf"))
    return found


def _modules():
    for p in sorted(PKG.rglob("*.py")):
        rel = p.relative_to(PKG).as_posix()
        if rel == "thermochemistry.py":
            continue
        yield rel, p


class TestNoPrivateCopies:
    @pytest.mark.parametrize("rel,path", list(_modules()), ids=lambda x: x if isinstance(x, str) else "")
    def test_no_module_carries_formation_data(self, rel, path):
        found = [f for f in _copies(path) if f[2] not in ALLOWED.get(rel, set())]
        assert not found, (f"{rel} carries its own formation data {found}: read it from "
                           "difflow_refinery.thermochemistry (#339)")

    def test_the_allowlist_is_not_stale(self):
        """Every allowed copy still exists (remove the entry when it goes)."""
        for rel, names in ALLOWED.items():
            have = {f[2] for f in _copies(PKG / rel)}
            assert names <= have, (rel, names - have)

    def test_the_guard_catches_a_copy(self, tmp_path):
        p = tmp_path / "m.py"
        p.write_text("from difflow.database import get_species_data\n"
                     "HF_GAS = {'benzene': 82930.0, 'water': _HF_H2O}\n"
                     "x = Species(Hf=-1.0)\n"
                     "CP_IG = {'water': (32.2, 1e-3, 0.0, 0.0)}\n"
                     "data = get_species_data('benzene')\n"
                     "h = data.Hf\n"
                     "S0 = 269.18\n"
                     "def _row(key, Hf, S0, cp):\n"
                     "    return key\n"
                     "R = [_row('benzene', 82930.0, 269.18, (1.0, 2.0, 3.0, 4.0))]\n")
        kinds = [k for _, k, _ in _copies(p)]
        assert kinds.count("Hf/S0 literal") == 2
        assert "Hf= literal" in kinds and "Cp table" in kinds and "reads a database Hf" in kinds
        assert {"Hf literal in _row()", "S0 literal in _row()", "cp literal in _row()"} <= set(kinds)

    def test_the_guard_catches_the_pre_339_copies(self):
        """Every private copy #339 removed, as it was at its parent commit."""
        import subprocess

        root = PKG.parent.parent
        for rel, want in (("isomerization/thermochem.py", "Hf literal in IsomSpecies()"),
                          ("reforming/species.py", "Hf literal in _S()"),
                          ("fcc/species.py", "Hf/S0 literal"),
                          ("gasplant/components.py", "Hf/S0 literal"),
                          ("alkylation/species.py", "reads a database Hf")):
            r = subprocess.run(["git", "-C", str(root), "show",
                                f"{PRE_339}:src/difflow_refinery/{rel}"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                pytest.skip("git history not available")
            p = Path(self._tmp) / rel.replace("/", "_")
            p.write_text(r.stdout)
            assert want in [k for _, k, _ in _copies(p)], rel

    @pytest.fixture(autouse=True)
    def _tmpdir(self, tmp_path):
        self._tmp = tmp_path

    def test_the_guard_passes_a_view(self, tmp_path):
        p = tmp_path / "m.py"
        p.write_text("HF_298 = {n: tc.Hf(n) for n in GASES}\n"
                     "CP_IG = {n: tc.species(n).cp for n in GASES}\n"
                     "HF = array('Hf')\n")
        assert _copies(p) == []

    def test_modules_expose_the_tables_numbers(self):
        from difflow_refinery.alkylation import species as al
        from difflow_refinery.fcc import species as fcc
        from difflow_refinery.isomerization import thermochem as iso
        from difflow_refinery.reforming import species as ref

        for s in ref.SPECIES.values():
            r = tc.species(s.table_key)
            assert (s.Hf, s.S0, tuple(s.cp)) == (r.Hf, r.S0, r.cp), s.key
        for s in iso.SPECIES:
            r = tc.species(s.name)
            assert (s.Hf, s.S, tuple(s.cp)) == (r.Hf, r.S0, r.cp), s.name
        for n in al.ALKYLATION_SPECIES:
            assert al.species(n).Hf_gas == tc.Hf(n), n
        for n in fcc.REGENERATOR_GASES:
            assert fcc.HF_298[n] == tc.Hf(n) and fcc.CP_IG[n] == tc.species(n).cp, n

    def test_gas_plant_heating_values_use_the_table(self):
        from difflow_refinery.gasplant.components import light_component_data

        d = light_component_data("propane")
        lhv = (tc.Hf("propane") - 3 * tc.Hf("carbon_dioxide") - 4 * tc.Hf("water"))
        assert d["lhv"] == pytest.approx(lhv, rel=1e-14)
        d = light_component_data("hydrogen_sulfide")
        assert d["lhv"] == pytest.approx(tc.Hf("hydrogen_sulfide") - tc.Hf("water")
                                         - tc.Hf("sulfur_dioxide"), rel=1e-14)

    def test_hydrotreating_model_compounds_are_the_table(self):
        """#338: the hydrotreater's model compounds are read from the table,
        not copied (its MODEL_COMPOUNDS is a view)."""
        from difflow_refinery.hydrotreating.kinetics import CRACK_GAS_SPLIT, MODEL_COMPOUNDS

        for name in list(MODEL_COMPOUNDS) + list(CRACK_GAS_SPLIT):
            assert tc.resolve(name) in tc.TABLE, name
        for name, (Hf, S0) in MODEL_COMPOUNDS.items():
            assert Hf == tc.species(name).Hf and S0 == tc.species(name).S0, name
        for name in ("4_6_dimethyldibenzothiophene", "3_3_dimethylbiphenyl",
                     "methylcyclohexyltoluene", "pyridine", "indole", "aniline", "n_pentane",
                     "trans_decalin"):
            assert name in tc.TABLE


# =============================================================================
# Pins against the cited compilations
# =============================================================================

#: CODATA Key Values (Cox, Wagman & Medvedev 1989): Hf (kJ/mol), S0 (J/mol/K).
CODATA = {
    "water": (-241.826, 188.835), "carbon_monoxide": (-110.53, 197.660),
    "carbon_dioxide": (-393.51, 213.785), "sulfur_dioxide": (-296.81, 248.223),
    "hydrogen_sulfide": (-20.6, 205.81), "ammonia": (-45.94, 192.77),
    "hydrogen": (0.0, 130.680), "nitrogen": (0.0, 191.609), "oxygen": (0.0, 205.152),
}

#: API TDB ideal-gas Hf (kJ/mol) as in chemicals 1.5.2 "API TDB Albahri Hf (g).tsv".
API_TDB = {
    "methane": -74.52, "propane": -104.69, "isobutane": -134.99, "isopentane": -153.70,
    "n_pentane": -146.71, "n_hexane": -166.95, "2_methylpentane": -174.69,
    "3_methylpentane": -172.06, "2_2_dimethylbutane": -184.68, "2_3_dimethylbutane": -176.80,
    "cyclohexane": -123.13, "methylcyclopentane": -106.69, "benzene": 82.93, "toluene": 50.17,
    "propylene": 19.71, "isobutylene": -16.90, "naphthalene": 150.58, "tetralin": 26.61,
    "phenanthrene": 207.10, "biphenyl": 182.09, "diethyl_sulfide": -83.47, "pyridine": 140.16,
    "2_2_4_trimethylpentane": -224.01, "n_dodecane": -290.79, "2_methylnonane": -256.52,
}

#: CRC Handbook (95th ed.) ideal-gas Hf (kJ/mol), chemicals 1.5.2, where CRC is chosen.
CRC = {"2_3_dimethylpentane": -198.7, "2_methylhexane": -194.5, "1_butene": 0.1,
       "trans_decalin": -182.1, "cyclohexylbenzene": -16.7, "quinoline": 200.5,
       "carbazole": 200.7, "thiophene": 114.9, "benzothiophene": 166.3,
       "dibenzothiophene": 205.1}

#: Yaws (2014) ideal-gas S0 (J/mol/K), chemicals 1.5.2 "Yaws Hf S0 (g).tsv".
YAWS_S0 = {"benzene": 269.18, "cyclohexane": 297.31, "naphthalene": 333.6, "tetralin": 366.22,
           "isopentane": 343.89, "n_pentane": 349.25, "2_2_dimethylbutane": 358.22,
           "phenanthrene": 396.01, "thiophene": 278.81, "methane": 186.6}


class TestPinnedToSources:
    @pytest.mark.parametrize("key", sorted(CODATA))
    def test_codata(self, key):
        s = tc.species(key)
        assert s.Hf == pytest.approx(CODATA[key][0] * 1e3, abs=1e-9)
        assert s.S0 == pytest.approx(CODATA[key][1], abs=1e-9)
        assert s.Hf_source in ("CODATA", "ELEMENT") and s.S0_source == "CODATA"

    @pytest.mark.parametrize("key", sorted(API_TDB))
    def test_api_tdb(self, key):
        assert tc.species(key).Hf == pytest.approx(API_TDB[key] * 1e3, abs=1e-6)
        assert tc.species(key).Hf_source == "API_TDB"

    @pytest.mark.parametrize("key", sorted(CRC))
    def test_crc(self, key):
        assert tc.species(key).Hf == pytest.approx(CRC[key] * 1e3, abs=1e-6)
        assert tc.species(key).Hf_source == "CRC"
        assert tc.species(key).note, "every departure from API TDB says why"

    @pytest.mark.parametrize("key", sorted(YAWS_S0))
    def test_yaws_entropy(self, key):
        assert tc.S0(key) == pytest.approx(YAWS_S0[key], abs=1e-9)

    def test_benzothiophene(self):
        """#339 item 3: the calorimetric values, not ChemSep's 137.0."""
        h = tc.Hf("benzothiophene") / 1e3
        assert abs(h - 166.28) < 0.5      # Sabbah (1979), +-0.48
        assert abs(h - 166.6) < 0.5       # Good (1972)
        assert abs(h - 137.0) > 25.0      # ChemSep (DWSIM)
        # HDS heat per H2, benzothiophene + 3 H2 -> ethylbenzene + H2S
        q = tc.reaction_enthalpy({"benzothiophene": -1, "hydrogen": -3, "ethylbenzene": 1,
                                  "hydrogen_sulfide": 1}) / 3.0
        assert q == pytest.approx(-52.37e3, abs=1.0)

    def test_tetrahydrophenanthrene(self):
        assert tc.Hf("tetrahydrophenanthrene") == 92.3e3       # NIST, +-1.3
        # S0 by the ring-saturation increment: the poly step's dS is the di step's
        ds_poly = tc.S0("tetrahydrophenanthrene") - tc.S0("phenanthrene")
        ds_di = tc.S0("tetralin") - tc.S0("naphthalene")
        assert ds_poly == pytest.approx(ds_di, abs=1e-9)

    def test_estimates_are_their_construction(self):
        ar = tc.Hf("toluene") - tc.Hf("benzene")
        cy = tc.Hf("methylcyclohexane") - tc.Hf("cyclohexane")
        assert tc.Hf("4_6_dimethyldibenzothiophene") == pytest.approx(
            tc.Hf("dibenzothiophene") + 2 * ar, abs=10.0)
        assert tc.Hf("3_3_dimethylbiphenyl") == pytest.approx(tc.Hf("biphenyl") + 2 * ar, abs=10.0)
        assert tc.Hf("methylcyclohexyltoluene") == pytest.approx(
            tc.Hf("cyclohexylbenzene") + ar + cy, abs=10.0)

    def test_rejected_entropies_are_absent(self):
        for k in ("benzothiophene", "carbazole", "dibenzothiophene"):
            assert tc.species(k).S0 is None
            with pytest.raises(ValueError):
                tc.S0(k)

    def test_crosscheck_spread(self):
        """Where the chosen source is in the cross-check, it IS that value, and
        the cross-checked rows agree with an independent compilation to 1.5
        kJ/mol."""
        for k, alts in tc.HF_CROSSCHECK.items():
            s = tc.species(k)
            if s.Hf_source in alts:
                assert alts[s.Hf_source] * 1e3 == pytest.approx(s.Hf, abs=1e-6), k
            if s.status == "cross-checked":
                assert min(abs(v * 1e3 - s.Hf) for n, v in alts.items()
                           if n in ("ATcT", "CRC", "API_TDB") and n != s.Hf_source) <= 1.5e3

    def test_every_row_is_sourced(self):
        for s in tc.TABLE.values():
            for src in (s.Hf_source, s.S0_source, s.cp_source):
                assert src in tc.SOURCES, (s.key, src)
            if s.status in ("unverified", "estimate"):
                assert s.note, s.key
            assert s.cp_range[0] == 298.15 and s.cp_range[1] >= 1000.0, s.key
            assert (s.S0 is None) == (s.S0_source == "none"), s.key

    def test_homologous_series(self):
        """n-alkane CH2 increments of the chosen set: -20.0 to -21.2 kJ/mol
        (Benson's -20.6 with the compilation's +-0.6 scatter)."""
        chain = ["n_butane", "n_pentane", "n_hexane", "n_heptane", "n_octane", "n_nonane",
                 "n_decane"]
        inc = np.diff([tc.Hf(k) for k in chain])
        assert np.all((inc > -21.25e3) & (inc < -19.95e3)), inc

    def test_difflow_database_agrees_where_it_overlaps(self):
        """difflow.database is difflow core's table (its own users, its own
        constants) and is NOT read by the refinery; where it holds the same
        ideal-gas species it agrees to 1.4 kJ/mol (n-dodecane, CRC there).
        Water is excluded: the database's is the liquid's."""
        from difflow.database import _IDEAL_THERMO_DATA as db

        worst = 0.0
        for k, s in tc.TABLE.items():
            if k in db and k != "water" and "Hf" in db[k]:
                worst = max(worst, abs(db[k]["Hf"] - s.Hf))
        assert worst < 1.4e3


#: The Cp source evaluated at a few temperatures (J/mol/K): the TRC correlation,
#: JANAF tabulated points or the WebBook Shomate equations, from chemicals 1.5.2.
CP_REFERENCE = {
    "hydrogen": {298.15: 28.832, 600.0: 29.3084, 1000.0: 30.234},
    "nitrogen": {298.15: 29.1238, 600.0: 30.104, 1000.0: 32.6917, 1500.0: 34.8472},
    "oxygen": {298.15: 29.3826, 600.0: 32.1072, 1000.0: 34.8639, 1500.0: 36.5495},
    "water": {298.15: 33.59, 600.0: 36.325, 1000.0: 41.268, 1500.0: 47.09},
    "carbon_monoxide": {298.15: 29.142, 600.0: 30.443, 1000.0: 33.183, 1500.0: 35.217},
    "carbon_dioxide": {298.15: 37.129, 600.0: 47.321, 1000.0: 54.308, 1500.0: 58.379},
    "sulfur_dioxide": {298.15: 39.878, 600.0: 49.049, 1000.0: 54.484, 1500.0: 57.036},
    "hydrogen_sulfide": {298.15: 34.192, 600.0: 38.936, 1000.0: 45.786},
    "ammonia": {298.15: 35.652, 600.0: 45.293, 1000.0: 56.491},
    "methane": {298.15: 35.6609, 600.0: 52.4413, 1000.0: 72.7914},
    "n_pentane": {298.15: 120.0957, 600.0: 208.8446, 1000.0: 281.3086},
    "isopentane": {298.15: 118.8947, 600.0: 209.9475, 1000.0: 286.0932},
    "n_hexane": {298.15: 142.5959, 600.0: 248.1391, 1000.0: 330.9873},
    "2_methylpentane": {298.15: 142.2057, 600.0: 250.9949, 1000.0: 336.9468},
    "3_methylpentane": {298.15: 140.1083, 600.0: 248.91, 1000.0: 335.8458},
    "2_2_dimethylbutane": {298.15: 141.4718, 600.0: 253.1033, 1000.0: 347.8358},
    "2_3_dimethylbutane": {298.15: 139.3947, 600.0: 250.0798, 1000.0: 340.719},
    "n_heptane": {298.15: 165.1907, 600.0: 287.5254, 1000.0: 381.3256},
    "ethylene": {298.15: 42.8815, 600.0: 70.6885, 1000.0: 93.8798},
    "propylene": {298.15: 64.3606, 600.0: 107.9606, 1000.0: 144.434},
    "1_butene": {298.15: 85.6452, 600.0: 146.6401, 1000.0: 196.1093},
    "methylcyclopentane": {298.15: 109.5137, 600.0: 219.6425, 1000.0: 303.6544},
    "cyclohexane": {298.15: 106.3322, 600.0: 224.7037, 1000.0: 315.348},
    "trans_decalin": {298.15: 168.5704, 600.0: 349.2685, 1000.0: 487.457},
    "benzene": {298.15: 82.5369, 600.0: 160.045, 1000.0: 211.3606},
    "toluene": {298.15: 103.7989, 600.0: 196.3214, 1000.0: 261.0054},
    "tetralin": {298.15: 152.4797, 600.0: 295.2354, 1000.0: 394.2407},
    "naphthalene": {298.15: 131.9391, 600.0: 251.7098, 1000.0: 328.8552},
    "biphenyl": {298.15: 165.2523, 600.0: 308.0758, 1000.0: 401.6399},
    "phenanthrene": {298.15: 185.7026, 600.0: 345.8568, 1000.0: 448.229},
    "tetrahydrophenanthrene": {298.15: 200.7384, 600.0: 384.7306, 1000.0: 511.1433},
    "thiophene": {298.15: 72.7854, 600.0: 128.2631, 1000.0: 161.5194},
    "pyridine": {298.15: 77.6081, 600.0: 148.1818, 1000.0: 194.9004},
    "n_dodecane": {298.15: 278.2881, 600.0: 484.7986, 1000.0: 632.2785},
}


class TestHeatCapacity:
    @pytest.mark.parametrize("key", sorted(CP_REFERENCE))
    def test_cubic_against_its_source(self, key):
        s = tc.species(key)
        for T, ref in CP_REFERENCE[key].items():
            assert float(tc.cp(key, T)) == pytest.approx(ref, rel=s.cp_max_dev + 1e-4), (key, T)

    def test_fits_are_close(self):
        for s in tc.TABLE.values():
            if s.cp_max_dev is not None:
                assert s.cp_max_dev < 0.02, s.key

    def test_cp_is_physical_over_its_range(self):
        for s in tc.TABLE.values():
            T = np.linspace(*s.cp_range, 60)
            c = np.asarray(tc.cp(s.key, T))
            assert np.all(c > 2.5 * tc.R), s.key                  # above monatomic
            if s.key not in ("hydrogen",):
                assert np.all(np.diff(c) > -1e-9), s.key            # rises with T

    def test_joback_against_analogues(self):
        """No correlation covers several heteroaromatics; their Cp is Joback's.
        Quinoline's sits 1.5-5.9 % below the TRC Cp of naphthalene, its
        isoelectronic all-carbon analogue, over 298-1000 K (Joback itself is
        1.1 % off TRC on naphthalene)."""
        T = np.linspace(298.15, 1000.0, 30)
        q, n = np.asarray(tc.cp("quinoline", T)), np.asarray(tc.cp("naphthalene", T))
        assert np.max(np.abs(q / n - 1.0)) < 0.06


# =============================================================================
# API
# =============================================================================


class TestAPI:
    def test_aliases(self):
        assert tc.resolve("H2") == "hydrogen"
        assert tc.resolve("propylbenzene") == "n_propylbenzene"
        assert tc.resolve("2-methylpentane") == "2_methylpentane"
        with pytest.raises(KeyError):
            tc.resolve("unobtainium")

    def test_closed_form_integrals(self):
        T = 750.0
        for k in ("benzene", "water", "n_hexane"):
            g = np.linspace(298.15, T, 20001)
            c = np.asarray(tc.cp(k, g))
            assert float(tc.sensible_enthalpy(k, T)) == pytest.approx(np.trapezoid(c, g), rel=1e-7)
            assert float(tc.sensible_entropy(k, T)) == pytest.approx(np.trapezoid(c / g, g), rel=1e-7)

    def test_derivatives(self):
        k = "toluene"
        T = 640.0
        assert float(jax.grad(lambda t: tc.enthalpy(k, t))(T)) == pytest.approx(float(tc.cp(k, T)))
        assert float(jax.grad(lambda t: tc.entropy(k, t))(T)) == pytest.approx(
            float(tc.cp(k, T)) / T)
        assert float(jax.grad(lambda t: tc.gibbs(k, t))(T)) == pytest.approx(
            -float(tc.entropy(k, T)))
        assert float(tc.entropy(k, T, P=10e5)) == pytest.approx(
            float(tc.entropy(k, T)) - tc.R * math.log(10.0))

    def test_reactions(self):
        nu = {"benzene": -1, "hydrogen": -3, "cyclohexane": 1}
        assert float(tc.reaction_enthalpy(nu)) == pytest.approx(-206.06e3, abs=1e-6)
        T = 623.15
        lnK = float(tc.ln_K(nu, T))
        assert lnK == pytest.approx(-float(tc.reaction_gibbs(nu, T)) / (tc.R * T))
        dH, dS = float(tc.reaction_enthalpy(nu, T)), float(tc.reaction_entropy(nu, T))
        assert lnK == pytest.approx(-(dH - T * dS) / (tc.R * T), rel=1e-12)
        # van 't Hoff, with the Cp-integrated dH
        assert float(jax.grad(lambda t: tc.ln_K(nu, t))(T)) == pytest.approx(dH / (tc.R * T**2))
        with pytest.raises(ValueError, match="balance"):
            tc.reaction_enthalpy({"benzene": -1, "hydrogen": -2, "cyclohexane": 1})

    def test_isomerization_heat(self):
        """nC5 -> iC5: -6.99 kJ/mol (was -8.10 with Prosen & Rossini)."""
        assert float(tc.reaction_enthalpy({"n_pentane": -1, "isopentane": 1})) == pytest.approx(
            -6.99e3, abs=1e-6)

    def test_ideal_gas_set(self):
        names = ("H2", "benzene", "cyclohexane")
        gas = tc.IdealGasSet(names)
        nu = np.array([-3.0, -1.0, 1.0])
        T = 600.0
        assert float(gas.ln_K(nu, T)) == pytest.approx(
            float(tc.ln_K({"benzene": -1, "H2": -3, "cyclohexane": 1}, T)), rel=1e-13)
        H = gas.enthalpy(jnp.array([400.0, 600.0]))
        assert H.shape == (2, 3)
        assert float(H[1, 1]) == pytest.approx(float(tc.enthalpy("benzene", 600.0)))
        np.testing.assert_allclose(gas.ln_K(np.stack([nu, -nu]), T),
                                   [float(gas.ln_K(nu, T)), -float(gas.ln_K(nu, T))])
        assert jax.jit(gas.gibbs)(T).shape == (3,)
        with pytest.raises(ValueError, match="S0"):
            tc.IdealGasSet(["benzothiophene"]).gibbs(T)
        assert gas.ELEMENTS["C"].tolist() == [0.0, 6.0, 6.0]


class TestCoreDatabaseAgrees:
    """The core database's HDS species carry this table's Hf and Cp (#391).

    ``difflow.database`` copies them because the core may not import a
    plugin; this is what keeps the copy from drifting.
    """

    @pytest.mark.parametrize("key", ["hydrogen", "hydrogen_sulfide", "thiophene"])
    def test_hf_cp_and_mw_match(self, key):
        from difflow.database import get_species_data

        d = get_species_data(key)
        row = tc.species(key)
        assert d.Hf == row.Hf
        assert tuple(d.Cp_coeffs) == tuple(row.cp)
        assert d.MW == pytest.approx(row.MW, abs=0.01)
