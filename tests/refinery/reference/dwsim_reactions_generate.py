"""Generate ``dwsim_reactions_reference.json``: difflow_refinery's reaction
thermochemistry against DWSIM 9.0.5's Gibbs, equilibrium and conversion
reactors.

Run from the repository root (needs DWSIM 9.0.5, .NET 8 and pythonnet; see
``scripts/install_dwsim.sh``)::

    PYTHONPATH=src:tests python -m refinery.reference.dwsim_reactions_generate

DWSIM has no hydrotreater, FCC, reformer or alkylation kinetic model, so what
is compared is the transferable physics: formation data, heats of reaction,
equilibria, and energy balances (:mod:`.dwsim_reactions_case` lists the
cases). Most runs are made twice (see that module): on DWSIM's own database
compounds (``"dwsim"``: its data, its model) and on hypothetical compounds
carrying difflow's constants (``"hypo"``: DWSIM's implementation of the same
numbers).

difflow is called here only for INPUTS -- its constants, the constructed
isomerization feeds, and one reformer bed whose outlet composition DWSIM's
conversion reactor is given -- all frozen in the file; every difflow answer
compared is computed by the test.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "dwsim_reactions_reference.json"
PREFIX = "DF_"


# ---------------------------------------------------------------------------
# Flowsheets
# ---------------------------------------------------------------------------

def _psat_shift_overrides(session, names, shift):
    """Raoult never condenses: each vapour-pressure equation's ``A`` raised
    by ``shift`` (equation 101 only; anything else is refused)."""
    out = {}
    for n in names:
        c = session._probe.AvailableCompounds[n]
        eq = str(c.VaporPressureEquation)
        A = float(c.Vapor_Pressure_Constant_A)
        if eq == "101":          # ln P = A + B/T + ...
            out[n] = {"Vapor_Pressure_Constant_A": A + shift}
        elif eq == "1":          # P = A (DWSIM's "User" graphite)
            out[n] = {"Vapor_Pressure_Constant_A": max(A, 1.0e5) * math.exp(shift)}
        else:
            raise RuntimeError(f"{n}: vapour-pressure equation {eq}; cannot shift it")
    return out


def ideal_gas_fs(session, names):
    """DWSIM database compounds on Raoult's law with Psat shifted: an ideal gas."""
    from . import dwsim_reactions_case as rc
    from .dwsim_session import IDEAL_RAOULT

    return session.flowsheet("RAOULT", compounds=list(names), options=IDEAL_RAOULT,
                             overrides=_psat_shift_overrides(session, names, rc.PSAT_SHIFT))


def hypos(consts: dict, keys, S_H2: float, shift: float | None = None, cp: bool = True):
    """:class:`ThermoHypo` per key of ``consts`` (``Hf``, ``S``, ``cp``,
    ``C``, ``H``, ``MW``, ``Tc``, ``Pc``, ``omega``)."""
    from .dwsim_reactors import ThermoHypo, gibbs_of_formation
    from .dwsim_session import lee_kesler_eq101

    out = []
    for k in keys:
        c = consts[k]
        f = c.get("formula") or {e: n for e, n in (("C", c.get("C", 0)), ("H", c.get("H", 0))) if n}
        psat = None
        if shift is not None:
            A, B, C, D, E = lee_kesler_eq101(c["Tc"], c["Pc"], c["omega"])
            psat = (A + shift, B, C, D, E)
        out.append(ThermoHypo(name=PREFIX + k, MW=c["MW"], Tc=c["Tc"], Pc=c["Pc"], omega=c["omega"],
                              cp_ig=list(c["cp"]) if cp else None, Hf=c["Hf"], psat_eq101=psat,
                              formula=f, Gf=gibbs_of_formation(c["Hf"], c["S"], f, S_H2)))
    return out


def hypo_ideal_fs(session, consts, keys, S_H2, cp=True):
    from . import dwsim_reactions_case as rc
    from .dwsim_session import IDEAL_RAOULT

    return session.flowsheet("RAOULT", hypos=hypos(consts, keys, S_H2, rc.PSAT_SHIFT, cp=cp),
                             options=IDEAL_RAOULT)


def _run(fn, *a, **k):
    try:
        r = fn(*a, **k)
    except Exception as e:  # noqa: BLE001 - recorded, never hidden
        return {"errors": [f"{type(e).__name__}: {e}"]}
    return r


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def species_table(session, names):
    """DWSIM's ideal-gas functions of every database compound used, at
    :data:`T_TABLE`."""
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import pure_functions

    fs = ideal_gas_fs(session, names)
    return {n: {"constants": session.compound(n),
                "T": {repr(T): pure_functions(fs, n, T) for T in rc.T_TABLE}} for n in names}


def hypo_table(session, consts, keys, S_H2):
    """The same functions for the hypos on difflow's constants: how DWSIM
    integrates difflow's Cp (midpoint rule) and builds G_f(T)."""
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import pure_functions

    fs = hypo_ideal_fs(session, consts, keys, S_H2)
    return {k: {repr(T): pure_functions(fs, PREFIX + k, T) for T in rc.T_TABLE} for k in keys}


def _both(fs_factory, names_dw, reactions, F, T_in, P, T_out, gibbs_kw=None):
    """One equilibrium case on DWSIM's equilibrium reactor (``K`` from its
    Gibbs energies; the reactions given) and on its Gibbs reactor (free
    minimisation under element balances): ``{"equilibrium": ..., "gibbs": ...}``."""
    from .dwsim_reactors import run_equilibrium, run_gibbs

    rx = [{"name": n, "nu": {names_dw[k]: v for k, v in nu.items()},
           "base": names_dw[next(k for k, v in nu.items() if v < 0 and k not in ("hydrogen", "H2"))]}
          for n, nu in reactions.items()]
    out = {"equilibrium": _run(run_equilibrium, fs_factory(), rx, F, T_in, P, T_out=T_out),
           "gibbs": _run(run_gibbs, fs_factory(), F, T_in, P, T_out=T_out, phase="Vapor",
                         **(gibbs_kw or {}))}
    return out


def isomerization(session, inputs):
    from . import dwsim_reactions_case as rc
    from . import isom_generate as ig
    from difflow_refinery.isomerization import thermochem as tc

    consts = inputs["isom"]
    S_H2 = consts["hydrogen"]["S"]
    cd = ig.component_data()

    def factory(which, names):
        if which == "dwsim":
            return lambda: ideal_gas_fs(session, [rc.ISOM_DW[n] for n in names])
        return lambda: hypo_ideal_fs(session, consts, names, S_H2)

    def dwmap(which):
        return rc.ISOM_DW if which == "dwsim" else {n: PREFIX + n for n in consts}

    def rxns(names):
        return {k: nu for k, nu in rc.ISOM_REACTIONS.items() if set(nu) <= set(names)}

    out = {"families": {}, "c6_ring": {}, "adiabatic": {}, "isothermal_at_T_ad": {}}
    for which in ("dwsim", "hypo"):
        m = dwmap(which)
        fams = {}
        for fam, names in tc.FAMILIES.items():
            rows = []
            for T in ig.FAMILY_T:
                r = _both(factory(which, names), m, rxns(names), [1.0 / len(names)] * len(names),
                          T, ig.P_REACTOR, T)
                r.update(T_set=T, names=list(names))
                rows.append(r)
            fams[fam] = rows
        out["families"][which] = fams
        ring = []
        feed = dict(ig.C6_RING["feed"])
        for n in ("2_methylpentane", "3_methylpentane", "2_3_dimethylbutane", "2_2_dimethylbutane"):
            feed[n] = 1e-3
        for T in ig.C6_RING["T"]:
            r = _both(factory(which, list(feed)), m, rxns(list(feed)), list(feed.values()), T,
                      ig.C6_RING["P"], T)
            r.update(feed=feed, T_set=T)
            ring.append(r)
        out["c6_ring"][which] = ring
        adia, iso = {}, {}
        for kind in ig.ADIABATIC["feeds"]:
            feed = ig.adiabatic_charge(kind)
            names = list(feed)
            rn = [n for n in names if n == "hydrogen" or ig.SKELETON[n] in ("C5", "C6")]
            els = ig.independent_elements(cd, rn)
            M = [[ig.element_counts(cd, n).get(e, 0) for n in rn] for e in els]
            gkw = {"components": [m[n] for n in rn], "element_matrix": (els, M)}
            F = [feed[n] for n in names]
            r = _both(factory(which, names), m, rxns(names), F, ig.ADIABATIC["T_in"], ig.ADIABATIC["P"],
                      None, gkw)
            r.update(feed=feed, element_labels=els, element_rows=M, reacting=rn)
            adia[kind] = r
            # the same charge held at difflow's adiabatic temperature
            T_ad = difflow_adiabatic_T(consts, feed, rn, M)
            r = _both(factory(which, names), m, rxns(names), F, T_ad, ig.ADIABATIC["P"], T_ad, gkw)
            r.update(feed=feed, T_set=T_ad, reacting=rn)
            iso[kind] = r
        out["adiabatic"][which] = adia
        out["isothermal_at_T_ad"][which] = iso
    # DWSIM's own real-fluid model: Peng-Robinson with DWSIM's kij, VLE allowed
    real = {}
    for kind in ig.ADIABATIC["feeds"]:
        feed = ig.adiabatic_charge(kind)
        names = list(feed)
        rx = [{"name": n, "nu": {rc.ISOM_DW[k]: v for k, v in nu.items()},
               "base": rc.ISOM_DW[next(k for k, v in nu.items() if v < 0 and k != "hydrogen")]}
              for n, nu in rxns(names).items()]
        from .dwsim_reactors import run_equilibrium
        fs = session.flowsheet("PR", compounds=[rc.ISOM_DW[n] for n in names], kij="dwsim")
        r = _run(run_equilibrium, fs, rx, [feed[n] for n in names], ig.ADIABATIC["T_in"],
                 ig.ADIABATIC["P"], T_out=None)
        r.update(feed=feed, package="Peng-Robinson, DWSIM kij")
        real[kind] = r
    out["adiabatic"]["dwsim_pr"] = real
    return out


def difflow_adiabatic_T(consts, feed, reacting, element_rows):
    """difflow's adiabatic equilibrium temperature of a charge (K), at which
    it is also run isothermally: the emulation under difflow's conventions,
    which is difflow's reactor (and the IDAES reference, when current) to
    1e-6 K. Computed, not copied, since #339 moved the constants."""
    from .. import _dwsim_rx_emulation as em
    from . import isom_generate as ig

    inert = {k: v for k, v in feed.items() if k not in reacting}
    T, _ = em.adiabatic(consts, reacting, element_rows, [feed[k] for k in reacting], inert,
                        ig.ADIABATIC["T_in"], ig.ADIABATIC["P"], em.DIFFLOW)
    return T


def reformer(session, inputs):
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import run_conversion, sequential_conversions

    consts = inputs["reformer"]
    S_H2 = consts["H2"]["S"]
    out = {"equilibria": {}, "bed": {}}
    for which in ("dwsim", "hypo"):
        m = rc.REFORMER_DW if which == "dwsim" else {k: PREFIX + k for k in consts}
        eqs = {}
        for name, case in rc.REFORMER_EQUILIBRIA.items():
            keys = list(case["feed"])
            rows = []
            for T in case["T"]:
                for P in case["P"]:
                    if which == "dwsim":
                        fac = lambda keys=keys: ideal_gas_fs(session, [rc.REFORMER_DW[k] for k in keys])  # noqa: E731
                    else:
                        fac = lambda keys=keys: hypo_ideal_fs(session, consts, keys, S_H2)  # noqa: E731
                    F = [case["feed"][k] if case["feed"][k] > 0 else rc.TRACE for k in keys]
                    r = _both(fac, m, case["reactions"], F, T, P, T)
                    r.update(feed=dict(zip(keys, F)), T_set=T, P_set=P)
                    rows.append(r)
            eqs[name] = rows
        out["equilibria"][which] = eqs
    bed = inputs["reformer_bed"]
    names = bed["names"]
    formula = {k: {"C": consts[k]["C"], "H": consts[k]["H"]} for k in names}
    rxns = sequential_conversions(names, bed["F_in"], bed["F_out"], formula, pivot="C1", hydrogen="H2")
    for which in ("dwsim", "hypo"):
        if which == "dwsim":
            fs = session.flowsheet("PR", compounds=[rc.REFORMER_DW[k] for k in names], kij="zero")
            m = rc.REFORMER_DW
        else:
            fs = session.flowsheet("PR", hypos=hypos(consts, names, S_H2), kij="zero")
            m = {k: PREFIX + k for k in names}
        dr = [{**r, "nu": {m[k]: v for k, v in r["nu"].items()}, "base": m[r["base"]]} for r in rxns]
        res = _run(run_conversion, fs, dr, bed["F_in"], bed["T_in"], bed["P"], T_out=None)
        res["reactions"] = rxns
        out["bed"][which] = res
    return out


def hydroprocessing(session, inputs):
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import run_conversion

    out = {"heats": {}, "equilibria": {}}
    for name, (nu, _) in rc.HDT_REACTIONS.items():
        missing = [k for k in nu if rc.HDT_DW[k] is None]
        if missing:
            out["heats"][name] = {"missing_in_dwsim": missing}
            continue
        keys = list(nu)
        fs = ideal_gas_fs(session, [rc.HDT_DW[k] for k in keys])
        base = next(k for k, v in nu.items() if v < 0 and k != "hydrogen")
        F = [1.0 if k == base else (-2.0 * v + 1.0 if k == "hydrogen" else 0.0) for k, v in nu.items()]
        rx = [{"name": "r", "nu": {rc.HDT_DW[k]: v for k, v in nu.items()},
               "base": rc.HDT_DW[base], "conversion": 50.0}]
        r = _run(run_conversion, fs, rx, F, rc.HDT_T, 1.0e5, T_out=rc.HDT_T)
        r.update(feed=dict(zip(keys, F)), base=base, conversion=0.5, T_set=rc.HDT_T)
        out["heats"][name] = r
    mc = inputs["hdt"]["model_compounds"]
    S_H2 = mc["hydrogen"]["S"]
    for name, case in rc.AROMATIC_EQUILIBRIA.items():
        keys = list(case["feed"])
        consts = {}
        for k in keys:
            c = session.compound(rc.HDT_DW[k])
            consts[k] = {"Hf": mc[k]["Hf"], "S": mc[k]["S"], "cp": None,
                         "formula": rc.HDT_FORMULA[k], "MW": c["MW"],
                         "Tc": c["Tc"], "Pc": c["Pc"], "omega": c["omega"]}
        for which in ("dwsim", "hypo"):
            m = rc.HDT_DW if which == "dwsim" else {k: PREFIX + k for k in keys}
            if which == "dwsim":
                fac = lambda keys=keys: ideal_gas_fs(session, [rc.HDT_DW[k] for k in keys])  # noqa: E731
            else:
                # difflow's hydrotreating K: constant dH and dS (Cp zero)
                fac = lambda keys=keys, consts=consts: hypo_ideal_fs(session, consts, keys, S_H2, cp=False)  # noqa: E731
            rows = []
            for T in case["T"]:
                for P in case["P"]:
                    F = [case["feed"][k] if case["feed"][k] > 0 else rc.TRACE for k in keys]
                    r = _both(fac, m, case["reactions"], F, T, P, T)
                    r.update(feed=dict(zip(keys, F)), T_set=T, P_set=P)
                    rows.append(r)
            out["equilibria"].setdefault(name, {})[which] = rows
    return out


def alkylation(session, inputs):
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import run_conversion

    out = {"single": {}, "route_A": {}}

    def one(nu, base):
        keys = list(dict.fromkeys(["isobutane", *nu]))
        n_base = -nu[base]
        F = [10.0 if k == "isobutane" else (n_base if k == base else 0.0) for k in keys]
        rows = []
        for T in rc.ALKY_T:
            fs = session.flowsheet("PR", compounds=[rc.ALKY_DW[k] for k in keys], kij="dwsim")
            rx = [{"name": "alk", "nu": {rc.ALKY_DW[k]: v for k, v in nu.items()},
                   "base": rc.ALKY_DW[base], "conversion": 100.0, "phase": "Liquid"}]
            r = _run(run_conversion, fs, rx, F, T, rc.ALKY_P, T_out=T)
            r.update(feed=dict(zip(keys, F)), T_set=T, base=base)
            rows.append(r)
        return rows

    for name, nu in rc.ALKY_REACTIONS.items():
        base = next(k for k, v in nu.items() if v < 0 and k != "isobutane")
        out["single"][name] = one(nu, base)
    for olefin, routes in inputs["alky"]["routes"].items():
        out["route_A"][olefin] = one(routes["A"], olefin)
    return out


def fcc(session, inputs):
    from . import dwsim_reactions_case as rc
    from .dwsim_reactors import ThermoHypo, run_conversion
    from .dwsim_session import IDEAL_RAOULT

    f = inputs["fcc"]
    aw = f["ATOMIC_WEIGHT"]
    nC = f["elements_kg_s"]["C"] * 1000.0 / aw["C"]
    nH = f["elements_kg_s"]["H"] * 1000.0 / aw["H"]
    keys = list(rc.FCC_DW)
    D = rc.FCC_DW
    db = [D[k] for k in keys if k != "carbon"]
    coke_c = ThermoHypo(name="COKE_C", MW=aw["C"], Tc=33.0, Pc=13.0e5, omega=0.0,
                        cp_ig=[8.5, 0.0, 0.0, 0.0], Hf=0.0, formula={"C": 1}, Gf=0.0,
                        psat_eq101=(rc.PSAT_SHIFT + 30.0, 0.0, 0.0, 0.0, 0.0))
    out = {}
    for r_co, burn in f["burn"].items():
        r = float(r_co)
        air = burn["air"]
        F = {"carbon": nC, "hydrogen": nH / 2.0, "oxygen": air["oxygen"], "nitrogen": air["nitrogen"],
             "carbon_dioxide": 0.0, "carbon_monoxide": 0.0, "water": 0.0}
        rx = [{"name": "co2", "nu": {D["carbon"]: -1, D["oxygen"]: -1, D["carbon_dioxide"]: 1},
               "base": D["carbon"], "conversion": 100.0 / (1.0 + r), "phase": "Mixture"},
              {"name": "co", "nu": {D["carbon"]: -1, D["oxygen"]: -0.5, D["carbon_monoxide"]: 1},
               "base": D["carbon"], "conversion": 100.0, "phase": "Mixture"},
              {"name": "h2o", "nu": {D["hydrogen"]: -1, D["oxygen"]: -0.5, D["water"]: 1},
               "base": D["hydrogen"], "conversion": 100.0, "phase": "Mixture"}]
        runs = {}
        for T_out in (298.15, *rc.FCC["T_rg"]):
            fs = session.flowsheet("RAOULT", compounds=db, hypos=[coke_c], options=IDEAL_RAOULT,
                                   overrides=_psat_shift_overrides(session, db, rc.PSAT_SHIFT))
            order = {n: i for i, n in enumerate(fs.names)}
            vec = [0.0] * len(fs.names)
            for k in keys:
                vec[order[D[k]]] = F[k]
            res = _run(run_conversion, fs, rx, vec, 298.15, rc.FCC["P"], T_out=T_out)
            res["feed"] = F
            runs[repr(T_out)] = res
        out[r_co] = runs
    return out


# ---------------------------------------------------------------------------

def provenance(session) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=HERE).stdout.strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    p = session.provenance()
    p.update({
        "generated": datetime.date.today().isoformat(),
        "script": "tests/refinery/reference/dwsim_reactions_generate.py",
        "command": "PYTHONPATH=src:tests python -m refinery.reference.dwsim_reactions_generate",
        "difflow_commit": commit,
        "reference_simulator": (
            f"{p['dwsim']}: Reactor_Equilibrium (K from its Gibbs energies), Reactor_Gibbs (DirectMinimization, IPOPT off), Reactor_Conversion; "
            "ideal-gas runs on Raoult's Law (no Poynting/Henry) with every vapour pressure "
            "multiplied by exp(PSAT_SHIFT) so nothing condenses; energy runs on Peng-Robinson. "
            "'dwsim' runs use DWSIM's database compounds (ChemSep / ChEDL Thermo / User), "
            "'hypo' runs hypothetical compounds on difflow's Hf, S (as Gf), Cp and critical "
            "constants. DWSIM's reactors take P0 = 101325 Pa and R = 8.314."),
    })
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from . import dwsim_reactions_case as rc

    # difflow inputs first (JAX), then DWSIM (CoreCLR) in the same process
    inputs = {"isom": rc.isom_constants(), "reformer": rc.reformer_constants(),
              "hdt": rc.hdt_constants(), "alky": rc.alky_constants(), "fcc": rc.fcc_constants(),
              "reformer_bed": rc.reformer_bed()}
    from . import isom_generate as ig
    inputs["isom_cases"] = {"family_T": list(ig.FAMILY_T), "P": ig.P_REACTOR,
                            "c6_ring": ig.C6_RING, "adiabatic": ig.ADIABATIC,
                            "charges": {k: ig.adiabatic_charge(k) for k in ig.ADIABATIC["feeds"]}}

    from .dwsim_session import DWSIMSession

    session = DWSIMSession()
    db_names = sorted({*rc.ISOM_DW.values(), *rc.REFORMER_DW.values(),
                       *(v for v in rc.HDT_DW.values() if v), *rc.ALKY_DW.values(),
                       *(v for v in rc.FCC_DW.values() if v != "COKE_C")})
    cases = {k: getattr(rc, k) for k in ("PSAT_SHIFT", "T_TABLE", "REFORMER_REACTIONS",
                                         "REFORMER_EQUILIBRIA", "REFORMER_BED", "HDT_REACTIONS",
                                         "HDT_T", "AROMATIC_EQUILIBRIA", "ALKY_T", "ALKY_P",
                                         "ALKY_REACTIONS", "FCC", "TRACE", "ISOM_REACTIONS", "ISOM_DW", "REFORMER_DW", "HDT_DW",
                                         "ALKY_DW", "FCC_DW")}
    data = {"provenance": provenance(session), "cases": json.loads(json.dumps(cases)),
            "inputs": inputs}
    print("species", flush=True)
    data["species"] = species_table(session, db_names)
    data["hypo_species"] = {
        "isom": hypo_table(session, inputs["isom"], list(inputs["isom"]), inputs["isom"]["hydrogen"]["S"]),
        "reformer": hypo_table(session, inputs["reformer"], list(inputs["reformer"]),
                               inputs["reformer"]["H2"]["S"]),
    }
    for name, fn in (("isomerization", isomerization), ("reformer", reformer),
                     ("hydroprocessing", hydroprocessing), ("alkylation", alkylation), ("fcc", fcc)):
        print(name, flush=True)
        data[name] = fn(session, inputs)
    Path(a.out).write_text(json.dumps(data, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
