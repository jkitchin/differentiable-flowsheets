"""The refinery's one table of ideal-gas formation data (#339).

The GUARD: no difflow_refinery module carries its own copy of formation data
(Hf, S0, an ideal-gas Cp table) for a species in
:mod:`difflow_refinery.thermochemistry`, and the core database's copy agrees.
Then a smoke of the API.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import thermochemistry as tc

jax.config.update("jax_enable_x64", True)

PKG = Path(tc.__file__).resolve().parent

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


def test_no_module_carries_formation_data():
    for rel, path in _modules():
        found = [f for f in _copies(path) if f[2] not in ALLOWED.get(rel, set())]
        assert not found, (f"{rel} carries its own formation data {found}: read it from "
                           "difflow_refinery.thermochemistry (#339)")


def test_the_allowlist_is_not_stale():
    for rel, names in ALLOWED.items():
        have = {f[2] for f in _copies(PKG / rel)}
        assert names <= have, (rel, names - have)


def test_core_database_agrees():
    """difflow.database copies the HDS species (the core may not import a plugin, #391)."""
    from difflow.database import get_species_data

    for key in ("hydrogen", "hydrogen_sulfide", "thiophene"):
        d, row = get_species_data(key), tc.species(key)
        assert d.Hf == row.Hf and tuple(d.Cp_coeffs) == tuple(row.cp), key


def test_api_smoke_and_validation():
    assert tc.resolve("H2") == "hydrogen"
    with pytest.raises(KeyError):
        tc.resolve("unobtainium")
    nu = {"benzene": -1, "hydrogen": -3, "cyclohexane": 1}
    assert np.isfinite(float(tc.ln_K(nu, 623.15)))
    assert np.isfinite(float(jax.grad(lambda t: tc.ln_K(nu, t))(623.15)))
    with pytest.raises(ValueError, match="balance"):
        tc.reaction_enthalpy({"benzene": -1, "hydrogen": -2, "cyclohexane": 1})


def test_ideal_gas_set():
    gas = tc.IdealGasSet(("H2", "benzene", "cyclohexane"))
    assert gas.enthalpy(jnp.array([400.0, 600.0])).shape == (2, 3)
    assert jax.jit(gas.gibbs)(600.0).shape == (3,)
    with pytest.raises(ValueError, match="S0"):
        tc.IdealGasSet(["benzothiophene"]).gibbs(600.0)
