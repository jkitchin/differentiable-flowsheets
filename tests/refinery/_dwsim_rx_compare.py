"""The comparisons of ``test_dwsim_reactions.py``, as functions returning
numbers (the tests assert on them; the docs tables are printed from them by
``python -m refinery._dwsim_rx_compare``).

Every difflow number here is computed by difflow (or, for an equilibrium, by
:mod:`._dwsim_rx_emulation` under difflow's conventions, which reproduces
difflow's reactor and the IDAES reference to 1e-11); every DWSIM number is
read from ``reference/dwsim_reactions_reference.json``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import _dwsim_rx_emulation as em

HERE = Path(__file__).parent / "reference"


@lru_cache(maxsize=None)
def ref() -> dict:
    return json.loads((HERE / "dwsim_reactions_reference.json").read_text())


def _x(F):
    F = np.asarray(F, float)
    return F / F.sum()


# ---------------------------------------------------------------------------
# DWSIM's tabulated formation functions
# ---------------------------------------------------------------------------

def dw_species(name: str, T: float) -> dict:
    return ref()["species"][name]["T"][repr(float(T))]


def dw_reaction(nu_dw: dict, T: float) -> dict:
    """DWSIM's ``dH`` and ``dG`` (J/mol) of a reaction at a tabulated ``T``,
    from its database functions, and ``ln K`` as its reactors use it (R =
    8.314, 1 atm)."""
    dH = sum(v * (dw_species(n, T)["Hf"] + dw_species(n, T)["h_sens"]) for n, v in nu_dw.items())
    dG = sum(v * dw_species(n, T)["g_f"] for n, v in nu_dw.items())
    return {"dH": dH, "dG": dG, "lnK": -dG / (em.R_DWSIM * T)}


# ---------------------------------------------------------------------------
# Isomerization
# ---------------------------------------------------------------------------

def isom_consts():
    return ref()["inputs"]["isom"]


def _isom_A(names, labels=("C", "H")):
    K = isom_consts()
    return [[K[n][e] for n in names] for e in labels]


def isom_families(which: str, conv: dict, use_dwsim_g: bool = False, kind: str = "equilibrium"):
    """Max |x_DWSIM - x_emulated| over every family and T; the emulation
    under ``conv`` on difflow's constants (or on DWSIM's own tabulated
    ``g_f`` when ``use_dwsim_g``)."""
    from difflow_refinery.isomerization import thermochem as tc

    r = ref()
    worst, rows = 0.0, []
    for fam, names in tc.FAMILIES.items():
        for T, case in zip(r["inputs"]["isom_cases"]["family_T"], r["isomerization"]["families"][which][fam]):
            run = case[kind]
            assert not run["errors"], run["errors"]
            xd = _x(run["F"])
            if use_dwsim_g:
                gfun = lambda n, T=T: dw_species(r["cases"]["ISOM_DW"][n], T)["g_f"]  # noqa: E731
                n = em.species_equilibrium(isom_consts(), names, _isom_A(names), [1.0] * len(names), T,
                                           r["inputs"]["isom_cases"]["P"], conv, gibbs_fn=gfun)
            else:
                n = em.species_equilibrium(isom_consts(), names, _isom_A(names), [1.0] * len(names), T,
                                           r["inputs"]["isom_cases"]["P"], conv)
            xe = _x(n)
            d = np.abs(xd - xe)
            worst = max(worst, float(d.max()))
            rows.append({"family": fam, "T": T, "names": names, "dwsim": xd.tolist(),
                         "emulated": xe.tolist()})
    return worst, rows


def isom_ring(which: str, conv: dict, kind: str = "equilibrium"):
    r = ref()
    out = []
    for case in r["isomerization"]["c6_ring"][which]:
        run = {**case[kind], "feed": case["feed"]}
        names = list(run["feed"])
        T = case["T_set"]
        n = em.species_equilibrium(isom_consts(), names, _isom_A(names), list(run["feed"].values()), T,
                                   r["inputs"]["isom_cases"]["c6_ring"]["P"], conv)
        out.append({"T": T, "names": names, "dwsim": _x(run["F"]).tolist(), "emulated": _x(n).tolist(),
                    "max_dx": float(np.max(np.abs(_x(run["F"]) - _x(n))))})
    return out


def isom_adiabatic(which: str, conv: dict, kind_: str = "equilibrium"):
    r = ref()
    K = isom_consts()
    out = {}
    for kind, case in r["isomerization"]["adiabatic"][which].items():
        run = case[kind_]
        feed = case["feed"]
        rn = case["reacting"]
        A = case["element_rows"]
        inert = {k: v for k, v in feed.items() if k not in rn}
        T, n = em.adiabatic(K, rn, A, [feed[k] for k in rn], inert, r["inputs"]["isom_cases"]["adiabatic"]["T_in"],
                            r["inputs"]["isom_cases"]["adiabatic"]["P"], conv)
        names = list(feed)
        Fd = dict(zip(names, run["F"]))
        full = {**inert, **dict(zip(rn, n))}
        x_e = _x([full[k] for k in names])
        x_d = _x([Fd[k] for k in names])
        out[kind] = {"T_dwsim": run["T"], "T_emulated": T, "max_dx": float(np.max(np.abs(x_e - x_d))),
                     "names": names, "x_dwsim": x_d.tolist(), "x_emulated": x_e.tolist()}
    return out


# ---------------------------------------------------------------------------
# Reformer
# ---------------------------------------------------------------------------

def reformer_reactions():
    """Per reaction: difflow (reformer thermo) and DWSIM (database) dH at
    298.15 and 773.15 K, dG and ln K at 773.15 K."""
    import jax.numpy as jnp

    from difflow_refinery.reforming import species as sp
    from difflow_refinery.reforming import thermo as th

    r = ref()
    m = r["cases"]["REFORMER_DW"]
    out = {}
    for name, nu in r["cases"]["REFORMER_REACTIONS"].items():
        v = np.zeros(sp.N_SPECIES)
        for k, c in nu.items():
            v[sp.INDEX[k]] = c
        row = {}
        for T in (298.15, 773.15):
            dH = float(th.reaction_enthalpy(jnp.asarray(v), T))
            lnK = float(th.ln_K(jnp.asarray(v), T))
            dw = dw_reaction({m[k]: c for k, c in nu.items()}, T)
            row[repr(T)] = {"dH_difflow": dH, "dH_dwsim": dw["dH"],
                            "dG_difflow": -lnK * th.R * T, "dG_dwsim": dw["dG"],
                            "lnK_difflow": lnK, "lnK_dwsim": dw["lnK"],
                            "dn": float(sum(nu.values()))}
        out[name] = row
    return out


def reformer_equilibria(which: str, conv: dict, kind: str = "equilibrium"):
    r = ref()
    K = r["inputs"]["reformer"]
    out = {}
    for name, rows in r["reformer"]["equilibria"][which].items():
        res = []
        for case in rows:
            run = {**case[kind], **{k: case[k] for k in ("feed", "T_set", "P_set")}}
            keys = list(run["feed"])
            A = [[K[k]["C"] for k in keys], [K[k]["H"] for k in keys]]
            n = em.species_equilibrium(K, keys, A, list(run["feed"].values()), run["T_set"], run["P_set"], conv)
            res.append({"T": run["T_set"], "P": run["P_set"], "names": keys,
                        "dwsim": _x(run["F"]).tolist(), "emulated": _x(n).tolist(),
                        "max_dx": float(np.max(np.abs(_x(run["F"]) - _x(n))))})
        out[name] = res
    return out


def reformer_bed():
    r = ref()
    bed = r["inputs"]["reformer_bed"]
    out = {"T_in": bed["T_in"], "T_out_difflow": bed["T_out"], "dT_difflow": bed["T_out"] - bed["T_in"]}
    Fo = np.asarray(bed["F_out"])
    for which in ("hypo", "dwsim"):
        run = r["reformer"]["bed"][which]
        F = np.asarray(run["F"])
        out[which] = {"T_out": run["T"], "dT": run["T"] - bed["T_in"],
                      "flow_error": float(np.max(np.abs(F - Fo)) / Fo.sum()),
                      "errors": run["errors"], "liquid": float(np.sum(run["F_liquid"]))}
    return out


# ---------------------------------------------------------------------------
# Hydroprocessing
# ---------------------------------------------------------------------------

def hdt_heats():
    r = ref()
    mc = r["inputs"]["hdt"]["model_compounds"]
    out = {}
    for name, (nu, behind) in r["cases"]["HDT_REACTIONS"].items():
        h2 = -nu["hydrogen"]
        dH_df = sum(v * mc[k]["Hf"] for k, v in nu.items())
        run = r["hydroprocessing"]["heats"][name]
        row = {"behind": behind, "h2": h2, "dH_difflow": dH_df, "per_H2_difflow": dH_df / h2}
        if "missing_in_dwsim" in run:
            row["missing_in_dwsim"] = run["missing_in_dwsim"]
        else:
            xi = run["feed"][run["base"]] * run["conversion"] / -nu[run["base"]]
            row.update(dH298_dwsim=run["reaction_heat_298"]["r"], dHT_dwsim=run["Q"] / xi,
                       T=run["T_set"], per_H2_dwsim=run["reaction_heat_298"]["r"] / h2,
                       errors=run["errors"])
        out[name] = row
    return out


def aromatic_equilibria(which: str, conv: dict, kind: str = "equilibrium"):
    """Emulated equilibria on difflow's hydrotreating constants: Hf and S of
    MODEL_COMPOUNDS and their Cp (difflow's K is Cp-integrated since #338)."""
    r = ref()
    mc = r["inputs"]["hdt"]["model_compounds"]
    from .reference.dwsim_reactions_case import HDT_FORMULA

    consts = {k: {"Hf": v["Hf"], "S": v["S"], "cp": v["cp"]} for k, v in mc.items() if v["S"] is not None}
    out = {}
    for name, d in r["hydroprocessing"]["equilibria"].items():
        res = []
        for case in d[which]:
            run = {**case[kind], **{k: case[k] for k in ("feed", "T_set", "P_set")}}
            keys = list(run["feed"])
            A = [[HDT_FORMULA[k].get(e, 0) for k in keys] for e in ("C", "H")]
            n = em.species_equilibrium(consts, keys, A, list(run["feed"].values()), run["T_set"], run["P_set"], conv)
            arom = keys[1]
            sat = keys[2]
            xd = dict(zip(keys, run["F"]))
            ne = dict(zip(keys, n))
            res.append({"T": run["T_set"], "P": run["P_set"],
                        "aromatic_left_dwsim": xd[arom] / (xd[arom] + xd[sat]),
                        "aromatic_left_emulated": ne[arom] / (ne[arom] + ne[sat]),
                        "errors": run["errors"]})
        out[name] = res
    return out


def difflow_aromatic_lnK(step: int, T: float) -> float:
    """difflow's hydrotreating ln K (1 bar) of saturation step ``step`` at
    ``T``: what the rate law uses (Cp-integrated since #338)."""
    from difflow_refinery.hydrotreating.kinetics import aromatic_ln_K

    return float(aromatic_ln_K(T)[step])


def difflow_aromatic_lnK_constant(step: int, T: float) -> float:
    """The pre-#338 form, on the same (frozen) constants: 298 K dH and dS
    held constant, Cp neglected. For the record of the gap it closed."""
    from difflow_refinery.hydrotreating.kinetics import AROMATIC_REACTIONS

    mc = ref()["inputs"]["hdt"]["model_compounds"]
    nu = AROMATIC_REACTIONS[step]
    dH = sum(v * mc[k]["Hf"] for k, v in nu.items())
    dS = sum(v * mc[k]["S"] for k, v in nu.items())
    return -(dH - T * dS) / (em.R_DIFFLOW * T)


# ---------------------------------------------------------------------------
# Alkylation, FCC
# ---------------------------------------------------------------------------

def alky_heats():
    r = ref()
    sp = r["inputs"]["alky"]["species"]
    m = r["cases"]["ALKY_DW"]
    out = {}

    def row(nu, runs):
        base = runs[0]["base"]
        dl = sum(v * sp[k]["Hf_liquid"] for k, v in nu.items())
        dg = sum(v * sp[k]["Hf_gas"] for k, v in nu.items())
        dg_dw = sum(v * dw_species(m[k], 298.15)["Hf"] for k, v in nu.items())
        res = {"dH_liquid_difflow": dl, "dH_gas_difflow": dg, "dH_gas_dwsim": dg_dw}
        for run in runs:
            xi = run["feed"][base] / -nu[base]
            res[f"dH_liquid_dwsim_{run['T_set']}"] = run["Q"] / xi
            res[f"errors_{run['T_set']}"] = run["errors"]
            res[f"vapor_out_{run['T_set']}"] = float(sum(run["F_vapor"]))
        return res

    for name, nu in r["cases"]["ALKY_REACTIONS"].items():
        out[name] = row(nu, r["alkylation"]["single"][name])
    for olefin, routes in r["inputs"]["alky"]["routes"].items():
        out[f"route A, {olefin}"] = row(routes["A"], r["alkylation"]["route_A"][olefin])
        out[f"route A, {olefin}"]["difflow_reactor_dH_A"] = r["inputs"]["alky"]["route_heats"][olefin][0]
    return out


def fcc_h(name, T):
    f = ref()["inputs"]["fcc"]
    a, b, c, d = f["CP_IG"][name]
    F = lambda t: a * t + b * t ** 2 / 2 + c * t ** 3 / 3 + d * t ** 4 / 4  # noqa: E731
    return f["HF_298"][name] + F(T) - F(298.15)


def fcc_compare():
    r = ref()
    f = r["inputs"]["fcc"]
    out = {}
    for r_co, runs in r["fcc"].items():
        flue = f["burn"][r_co]["flue"]
        Hflue = lambda T: sum(v * fcc_h(k, T) for k, v in flue.items())  # noqa: E731
        res = {}
        for label, run in runs.items():
            if True:
                T = float(label)
                res[label] = {"Q_difflow": Hflue(T), "Q_dwsim": run["Q"], "errors": run["errors"],
                              "flue_dwsim": dict(zip(run["names"], run["F"]))}
        out[r_co] = res
    return out


# ---------------------------------------------------------------------------
# Is a DWSIM reactor answer DWSIM's own equilibrium?
# ---------------------------------------------------------------------------

#: A DWSIM equilibrium answer is ACCEPTED when its mole fractions are within
#: this of the ideal-gas equilibrium of DWSIM's own numbers (its tabulated
#: G_f(T) for database compounds; difflow's constants under DWSIM's
#: conventions for hypos). Accepted answers sit at 1e-11 to 1.4e-6; the
#: rejected ones (a minor species left at zero or at its start, a silent
#: non-convergence) much further off -- see the docs.
ACCEPT = 3.0e-6


def _cases():
    """Every isothermal equilibrium case: ``(label, which, case, keys,
    constants, element rows, n0, P, dwsim-name map)``."""
    r = ref()
    from difflow_refinery.isomerization import thermochem as tc

    from .reference.dwsim_reactions_case import HDT_FORMULA

    K = r["inputs"]["isom"]
    KR = r["inputs"]["reformer"]
    mc = r["inputs"]["hdt"]["model_compounds"]
    KH = {k: {"Hf": v["Hf"], "S": v["S"], "cp": v["cp"]} for k, v in mc.items() if v["S"] is not None}
    for which in ("dwsim", "hypo"):
        m = r["cases"]["ISOM_DW"]
        for fam, names in tc.FAMILIES.items():
            for case in r["isomerization"]["families"][which][fam]:
                yield (f"isom {fam} {case['T_set']:g} K", which, case, list(names), K,
                       [[K[n]["C"] for n in names], [K[n]["H"] for n in names]], [1.0] * len(names),
                       r["inputs"]["isom_cases"]["P"], m)
        for case in r["isomerization"]["c6_ring"][which]:
            names = list(case["feed"])
            yield (f"isom c6_ring {case['T_set']:g} K", which, case, names, K,
                   [[K[n]["C"] for n in names], [K[n]["H"] for n in names]],
                   list(case["feed"].values()), r["inputs"]["isom_cases"]["c6_ring"]["P"], m)
        for name, rows in r["reformer"]["equilibria"][which].items():
            for case in rows:
                keys = list(case["feed"])
                yield (f"reformer {name} {case['T_set']:g} K {case['P_set'] / 1e5:g} bar", which, case,
                       keys, KR, [[KR[k]["C"] for k in keys], [KR[k]["H"] for k in keys]],
                       list(case["feed"].values()), case["P_set"], r["cases"]["REFORMER_DW"])
        for name, d in r["hydroprocessing"]["equilibria"].items():
            for case in d[which]:
                keys = list(case["feed"])
                yield (f"aromatics {name} {case['T_set']:g} K {case['P_set'] / 1e5:g} bar", which, case,
                       keys, KH, [[HDT_FORMULA[k].get(e, 0) for k in keys] for e in ("C", "H")],
                       list(case["feed"].values()), case["P_set"], r["cases"]["HDT_DW"])


def own_equilibrium(which, case, keys, consts, A, n0, P, m):
    """The ideal-gas equilibrium of DWSIM's own numbers for this case."""
    T = case["T_set"]
    if which == "hypo":
        return _x(em.species_equilibrium(consts, keys, A, n0, T, P, em.DWSIM))
    g = lambda k: dw_species(m[k], T)["g_f"]  # noqa: E731
    return _x(em.species_equilibrium(consts, keys, A, n0, T, P, em.DWSIM, gibbs_fn=g))


@lru_cache(maxsize=None)
def acceptance():
    """``{(label, which, reactor): (accepted, max |dx| from DWSIM's own
    equilibrium or None, error or None)}``, every isothermal case, both
    DWSIM reactors."""
    out = {}
    for label, which, case, keys, consts, A, n0, P, m in _cases():
        x_own = own_equilibrium(which, case, keys, consts, A, n0, P, m)
        for kind in ("equilibrium", "gibbs"):
            run = case[kind]
            F = np.asarray(run.get("F") or [np.nan], float)
            if run["errors"] or not np.all(np.isfinite(F)) or F.sum() <= 0:
                out[(label, which, kind)] = (False, None, (run["errors"] or ["no flows"])[0][:60])
                continue
            d = float(np.max(np.abs(_x(F) - x_own)))
            out[(label, which, kind)] = (d < ACCEPT, d, None)
    return out


def accepted(label: str, which: str):
    """``(mole fractions, reactor)`` of the accepted DWSIM answer of a case,
    equilibrium reactor first; ``(None, None)`` when neither is accepted."""
    acc = acceptance()
    for lab, w, case, *_ in _cases():
        if lab == label and w == which:
            for kind in ("equilibrium", "gibbs"):
                if acc[(lab, w, kind)][0]:
                    return _x(case[kind]["F"]), kind
            return None, None
    raise KeyError(label)


def difflow_equilibrium(label: str):
    """difflow's answer for a case: the ideal-gas equilibrium of difflow's
    constants under difflow's conventions (which reproduces difflow's
    reactors -- see the tests)."""
    for lab, w, case, keys, consts, A, n0, P, m in _cases():
        if lab == label and w == "hypo":
            return _x(em.species_equilibrium(consts, keys, A, n0, case["T_set"], P, em.DIFFLOW)), keys
    raise KeyError(label)


if __name__ == "__main__":  # pragma: no cover - prints the numbers the docs quote
    acc = acceptance()
    print(f"{sum(1 for v in acc.values() if not v[0])} of {len(acc)} DWSIM reactor runs miss "
          f"their own equilibrium; accepted worst {max(v[1] for v in acc.values() if v[0]):.2g}")
    for k, v in sorted(acc.items()):
        if not v[0]:
            print("  miss", k, v[1] if v[1] is not None else v[2])
    print(json.dumps({"reformer_bed": reformer_bed(), "hdt": hdt_heats(), "alky": alky_heats(),
                      "reformer_reactions": reformer_reactions(), "fcc": fcc_compare()},
                     indent=1, default=float))
