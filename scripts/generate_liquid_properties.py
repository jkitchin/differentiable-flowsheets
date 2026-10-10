"""Regenerate difflow's liquid-property data from `chemicals` and CoolProp.

Writes two files, both committed:

- ``src/difflow/_liquid_property_data.py``: one density, viscosity and
  thermal-conductivity correlation per built-in species, as plain Python.
  difflow evaluates them in JAX (``difflow.liquid_properties``) and never
  imports `chemicals` or CoolProp at runtime.
- ``tests/reference/liquid_properties_reference.json``: values the tests
  compare against. ``chemicals`` values check that the JAX equations
  reproduce the source correlation exactly; CoolProp values (saturated
  liquid) check that the correlation is right. Regenerate, never loosen.

Usage (needs ``pip install chemicals thermo CoolProp scipy`` and difflow)::

    python scripts/generate_liquid_properties.py

Source priority, chosen by agreement with CoolProp over the species both
tables and CoolProp cover (issue #406): density VDI-PPDS first (closer for
30 of 37 species, median error 0.1% vs 0.4%); viscosity and thermal
conductivity Perry's 8e first (median 4.0% vs 5.5% and 2.7% vs 4.1%, and
Perry's states each correlation's range, VDI-PPDS does not). Then the
other table, then an estimate from data difflow already stores. Heavy water
is in neither table and is fitted to CoolProp's heavy-water formulation.
See issue #406.
"""

from __future__ import annotations

import json
import math
import pprint
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

import chemicals  # noqa: E402
import CoolProp.CoolProp as CP  # noqa: E402
from chemicals import thermal_conductivity as K_tab  # noqa: E402
from chemicals import viscosity as MU_tab  # noqa: E402
from chemicals import volume as RHO_tab  # noqa: E402
from chemicals.dippr import EQ100, EQ101, EQ105  # noqa: E402
from scipy.optimize import curve_fit, least_squares  # noqa: E402
from thermo.viscosity import determine_PPDS9_limits  # noqa: E402

from difflow import database  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "src" / "difflow" / "_liquid_property_data.py"
REF_OUT = ROOT / "tests" / "reference" / "liquid_properties_reference.json"
R = 8.314462618

# Names `chemicals.CAS_from_any` does not resolve from the difflow key.
CAS_OVERRIDES = {
    "heavy_water": "7789-20-0",
    "2_propanol": "67-63-0",
    "m_xylene": "108-38-3",
    "o_xylene": "95-47-6",
    "p_xylene": "106-42-3",
    "cis_2_butene": "590-18-1",
    "trans_2_butene": "624-64-6",
}

# VDI-PPDS rows that `thermo` refuses to use (known bad fit).
VDI_MU_EXCLUDE = {"78-83-1"}

# Orrick-Erbar group contributions (Poling, Prausnitz & O'Connell, The
# Properties of Gases and Liquids, 5th ed., ch. 9):
#     ln(mu / (rho_20 M)) = A + B/T,  mu in cP, rho_20 in g/cm^3 at 20 C.
# Carbon atoms: A = -(6.95 + 0.21 N), B = 275 + 99 N; each tertiary carbon
# (R3CH) adds (-0.15, 35); each quaternary carbon (R4C) adds (-1.20, 400).
# Only the paraffin isomers the tables miss are listed: (N, tertiary, quaternary).
ORRICK_ERBAR_GROUPS = {
    "2_2_5_trimethylhexane": (9, 1, 1),
    "2_3_4_trimethylpentane": (8, 3, 0),
    "2_4_dimethylpentane": (7, 2, 0),
    "2_5_dimethylhexane": (8, 2, 0),
    # Covered by the tables; used only to measure the estimate's error.
    "2_2_4_trimethylpentane": (8, 1, 1),
    "2_3_dimethylpentane": (7, 2, 0),
    "2_2_dimethylbutane": (6, 0, 1),
    "2_3_dimethylbutane": (6, 2, 0),
}

SOURCES = {
    "vdi_ppds": (
        "VDI Heat Atlas, 2nd ed. (VDI-GVC, Springer, 2010), section D3, PPDS "
        "correlations for saturated liquids, as tabulated in the `chemicals` "
        "package (C. Bell et al., v1.5.2, MIT)"),
    "perry_8e": (
        "Perry's Chemical Engineers' Handbook, 8th ed. (Green & Perry, "
        "McGraw-Hill, 2008): liquid density table (DIPPR 105; table number "
        "unverified), Tables 2-313 (viscosity, DIPPR 101) and 2-315 (thermal "
        "conductivity, DIPPR 100), "
        "as tabulated in the `chemicals` package (v1.5.2, MIT)"),
    "rackett": (
        "Estimate: Rackett equation with the Yamada-Gunn Z_RA = 0.29056 - "
        "0.08775 omega, from difflow's Tc, Pc and omega (Poling, Prausnitz & "
        "O'Connell 5e, ch. 4)"),
    "orrick_erbar": (
        "Estimate: Orrick-Erbar group contribution (Poling, Prausnitz & "
        "O'Connell 5e, ch. 9), with rho_20 from the Rackett estimate"),
    "sato_riedel": (
        "Estimate: Sato-Riedel (Poling, Prausnitz & O'Connell 5e, ch. 10; "
        "constant 1.1053 as in `chemicals`), Tb from difflow's Antoine"),
    "coolprop_fit": (
        "Fitted by difflow to CoolProp 8.0.0 saturated-liquid values for "
        "HeavyWater (Herrig et al., J. Phys. Chem. Ref. Data 47, 043102 "
        "(2018) equation of state; transport from CoolProp's HeavyWater "
        "fluid file)"),
}


def cas_for(name: str) -> str:
    if name in CAS_OVERRIDES:
        return CAS_OVERRIDES[name]
    parts = name.split("_")
    i = next(k for k, s in enumerate(parts) if not s.isdigit())
    queries = [",".join(parts[:i]) + ("-" if i else "") + " ".join(parts[i:]),
               name.replace("_", "-"), name.replace("_", " ")]
    for q in queries:
        try:
            cas = chemicals.CAS_from_any(q)
        except ValueError:
            continue
        # A name can resolve to the wrong compound; the MW must agree.
        mw = chemicals.MW(chemicals.search_chemical(cas).formula)
        if abs(mw - database.get_critical_props(name).MW) < 0.1:
            return cas
    raise KeyError(f"no CAS for {name}; add it to CAS_OVERRIDES")


def row(df, values, cas):
    return values[df.index.get_loc(cas)].tolist()


def normal_boiling_point(name: str) -> float:
    A, B, C = database.get_species_data(name).antoine_coeffs
    return B / (A - math.log10(101325.0)) - C


def corr(form, coeffs, Tmin, Tmax, stated, source):
    return {"form": form, "coeffs": [float(c) for c in coeffs],
            "Tmin": float(Tmin), "Tmax": float(Tmax),
            "range_stated": bool(stated), "source": source}


def rackett(name):
    c = database.get_critical_props(name)
    zra = 0.29056 - 0.08775 * c.omega
    # rho_molar = (Pc / (R Tc)) / Z_RA^(1 + (1 - T/Tc)^(2/7)): DIPPR 105.
    return [c.Pc / (R * c.Tc), zra, c.Tc, 2.0 / 7.0]


def density(name, cas, Tm):
    if cas in RHO_tab.rho_data_VDI_PPDS_2.index:
        MW, Tc, rhoc, a, b, c, d = row(RHO_tab.rho_data_VDI_PPDS_2,
                                       RHO_tab.rho_values_VDI_PPDS_2, cas)
        return corr("PPDS10", [Tc, rhoc, a, b, c, d, MW],
                    Tm or 0.3 * Tc, Tc, False, "vdi_ppds")
    if cas in RHO_tab.rho_data_Perry_8E_105_l.index:
        C1, C2, C3, C4, Tmin, Tmax = row(RHO_tab.rho_data_Perry_8E_105_l,
                                         RHO_tab.rho_values_Perry_8E_105_l, cas)
        return corr("DIPPR105", [C1, C2, C3, C4], Tmin, Tmax, True, "perry_8e")
    Tc = database.get_critical_props(name).Tc
    return corr("DIPPR105", rackett(name), Tm, Tc, False, "rackett")


def orrick_erbar(name):
    N, tert, quat = ORRICK_ERBAR_GROUPS[name]
    A = -(6.95 + 0.21 * N) - 0.15 * tert - 1.20 * quat
    B = 275.0 + 99.0 * N + 35.0 * tert + 400.0 * quat
    MW = database.get_species_data(name).MW
    rho20 = EQ105(293.15, *rackett(name)) * MW / 1e6      # g/cm^3
    # mu [Pa s] = 1e-3 rho20 M exp(A + B/T): DIPPR 101 with C = D = E = 0.
    return [math.log(1e-3 * rho20 * MW) + A, B, 0.0, 0.0, 0.0]


def viscosity(name, cas, Tm, Tc):
    if cas in MU_tab.mu_data_Perrys_8E_2_313.index:
        *C, Tmin, Tmax = row(MU_tab.mu_data_Perrys_8E_2_313,
                             MU_tab.mu_values_Perrys_8E_2_313, cas)
        return corr("DIPPR101", C, Tmin, Tmax, True, "perry_8e")
    if cas in MU_tab.mu_data_VDI_PPDS_7.index and cas not in VDI_MU_EXCLUDE:
        coeffs = row(MU_tab.mu_data_VDI_PPDS_7, MU_tab.mu_values_PPDS_7, cas)
        Tmin, Tmax = determine_PPDS9_limits(coeffs, Tm, Tc)
        return corr("PPDS9", coeffs, Tmin, Tmax, False, "vdi_ppds")
    if name in ORRICK_ERBAR_GROUPS:
        return corr("DIPPR101", orrick_erbar(name), Tm,
                    normal_boiling_point(name), False, "orrick_erbar")
    return None


def conductivity(name, cas, Tm, Tc):
    if cas in K_tab.k_data_Perrys_8E_2_315.index:
        *C, Tmin, Tmax = row(K_tab.k_data_Perrys_8E_2_315,
                             K_tab.k_values_Perrys_8E_2_315, cas)
        return corr("DIPPR100", C, Tmin, Tmax, True, "perry_8e")
    if cas in K_tab.k_data_VDI_PPDS_9.index:
        coeffs = row(K_tab.k_data_VDI_PPDS_9, K_tab.k_values_VDI_PPDS_9, cas)
        return corr("DIPPR100", coeffs, Tm or 0.3 * Tc, Tc, False, "vdi_ppds")
    c = database.get_critical_props(name)
    return corr("SATO_RIEDEL", [c.MW, c.Tc, normal_boiling_point(name)],
                Tm, c.Tc, False, "sato_riedel")


def coolprop_name(cas):
    for f in CP.get_global_param_string("FluidsList").split(","):
        if cas in CP.get_fluid_param_string(f, "CAS").split(","):
            return f
    return None


def coolprop_sat(fluid, T):
    out = {}
    for key, prop in (("rho", "Dmass"), ("mu", "V"), ("k", "L")):
        try:
            out[key] = CP.PropsSI(prop, "T", T, "Q", 0, fluid)
        except Exception:
            out[key] = None
    return out


def fit_heavy_water(Tm):
    """DIPPR forms fitted to CoolProp's saturated-liquid heavy water."""
    fluid = "HeavyWater"
    Tc = CP.PropsSI("Tcrit", fluid)
    T = np.linspace(Tm + 1.0, 0.95 * Tc, 120)
    sat = [coolprop_sat(fluid, t) for t in T]
    MW = CP.PropsSI("molar_mass", fluid) * 1e3
    rho_m = np.array([s["rho"] for s in sat]) / MW * 1e3      # mol/m^3
    mu = np.array([s["mu"] for s in sat])
    k = np.array([s["k"] for s in sat])

    # Fit ln(rho) with ln(A) as the parameter: A is ~1e3 times B and D.
    r = rackett("heavy_water")
    fit = least_squares(
        lambda p: p[0] - (1.0 + (1.0 - T / Tc)**p[2]) * np.log(p[1])
        - np.log(rho_m),
        [math.log(r[0]), r[1], r[3]],
        bounds=([-np.inf, 0.01, 0.05], [np.inf, 0.99, 1.0]))
    p_rho = [math.exp(fit.x[0]), fit.x[1], fit.x[2]]
    p_mu, _ = curve_fit(lambda t, a, b, c, d: np.exp(a + b / t + c * np.log(t)
                                                    + d * t),
                        T, mu, p0=[-10.0, 1000.0, 0.0, 0.0], maxfev=20000)
    p_k = np.polyfit(T, k, 3)[::-1]
    Tmax = 0.95 * Tc
    return (corr("DIPPR105", [p_rho[0], p_rho[1], Tc, p_rho[2]],
                 Tm, Tc, False, "coolprop_fit"),
            corr("DIPPR101", [*p_mu[:3], p_mu[3], 1.0], Tm, Tmax, False,
                 "coolprop_fit"),
            corr("DIPPR100", [*p_k, 0.0], Tm, Tmax, False, "coolprop_fit"))


def chemicals_value(c, T):
    """The source library's own evaluation, to pin difflow's JAX equations."""
    f, p = c["form"], c["coeffs"]
    if f == "PPDS10":
        Tc, rhoc, a, b, cc, d, MW = p
        return RHO_tab.volume_VDI_PPDS(T, Tc, rhoc, a, b, cc, d) / MW * 1e3
    if f == "DIPPR105":
        return EQ105(T, *p)
    if f == "PPDS9":
        return MU_tab.PPDS9(T, *p)
    if f == "DIPPR101":
        return EQ101(T, *p)
    if f == "DIPPR100":
        return EQ100(T, *p)
    if f == "SATO_RIEDEL":
        MW, Tc, Tb = p
        return K_tab.Sato_Riedel(T, MW, Tb, Tc)
    raise ValueError(f)


def main():
    data, ref = {}, {"chemicals": {}, "coolprop": {}, "estimate_check": {}}
    for name in database.list_species():
        cas = cas_for(name)
        crit = database.get_critical_props(name)
        Tm = chemicals.Tm(cas) or 0.3 * crit.Tc
        if name == "heavy_water":
            rho, mu, k = fit_heavy_water(Tm)
        else:
            rho = density(name, cas, Tm)
            mu = viscosity(name, cas, Tm, crit.Tc)
            k = conductivity(name, cas, Tm, crit.Tc)
        if mu is None:
            raise RuntimeError(f"no viscosity for {name}; add groups or data")
        data[name] = {"CAS": cas, "MW": crit.MW, "Tm": float(Tm),
                      "rho": rho, "mu": mu, "k": k}

        pts = {}
        for key, c in (("rho", rho), ("mu", mu), ("k", k)):
            Ts = np.linspace(c["Tmin"], min(c["Tmax"], 0.95 * crit.Tc), 5)[1:-1]
            pts[key] = [[float(t), float(chemicals_value(c, t))] for t in Ts]
        ref["chemicals"][name] = pts

        fluid = coolprop_name(cas)
        if fluid is not None:
            Tmin_cp = max(CP.PropsSI("Tmin", fluid), Tm)
            Tc_cp = CP.PropsSI("Tcrit", fluid)
            cp_pts = []
            for t in np.linspace(Tmin_cp + 0.05 * (Tc_cp - Tmin_cp),
                                 0.8 * Tc_cp, 4):
                s = coolprop_sat(fluid, float(t))
                cp_pts.append({"T": float(t), **s})
            ref["coolprop"][name] = {"fluid": fluid, "points": cp_pts}

    # How well do the estimates do where the tables have an answer?
    for name in ("2_2_4_trimethylpentane", "2_3_dimethylpentane",
                 "2_2_dimethylbutane", "2_3_dimethylbutane", "n_heptane",
                 "n_octane", "toluene"):
        cas, Tm = data[name]["CAS"], data[name]["Tm"]
        crit = database.get_critical_props(name)
        T = 298.15
        est = {"rho": EQ105(T, *rackett(name)),
               "k": K_tab.Sato_Riedel(T, crit.MW, normal_boiling_point(name),
                                      crit.Tc)}
        if name in ORRICK_ERBAR_GROUPS:
            est["mu"] = EQ101(T, *orrick_erbar(name))
        tab = {key: chemicals_value(data[name][key], T) for key in est}
        ref["estimate_check"][name] = {
            key: {"T": T, "estimate": float(est[key]), "table": float(tab[key])}
            for key in est}

    header = (
        '"""Liquid density, viscosity and thermal-conductivity correlations.\n\n'
        "GENERATED by scripts/generate_liquid_properties.py -- do not edit.\n"
        "Evaluated by difflow.liquid_properties; see that module for the\n"
        "correlation forms and units, and issue #406 for how the sources were\n"
        'chosen.\n"""\n\n')
    body = ("SOURCES = " + pprint.pformat(SOURCES, width=79, sort_dicts=True)
            + "\n\n" + "LIQUID_DATA = "
            + pprint.pformat(data, width=79, sort_dicts=True) + "\n")
    DATA_OUT.write_text(header + body)
    REF_OUT.parent.mkdir(parents=True, exist_ok=True)
    REF_OUT.write_text(json.dumps(ref, indent=1, sort_keys=True) + "\n")
    by_source = {}
    for d in data.values():
        for key in ("rho", "mu", "k"):
            by_source.setdefault((key, d[key]["source"]), 0)
            by_source[(key, d[key]["source"])] += 1
    print(f"{len(data)} species -> {DATA_OUT.relative_to(ROOT)}")
    for (key, src), n in sorted(by_source.items()):
        print(f"  {key:4s} {src:14s} {n}")
    print(f"CoolProp reference for {len(ref['coolprop'])} species "
          f"-> {REF_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
