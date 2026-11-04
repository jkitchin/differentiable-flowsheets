"""Seawater speciation and mineral saturation indices (Pitzer, differentiable).

Given element totals (mol/kg water) for Na, K, Mg, Ca, Cl, SO4, total
carbonate ``C_T`` and alkalinity, at temperature T, this solves the mass
balances, the alkalinity balance and the mass-action laws of the aqueous
model of :mod:`difflow_water.chemistry.pitzer` (species H+, OH-, Na+, K+,
Mg2+, Ca2+, MgOH+, Cl-, SO4^2-, HSO4-, HCO3-, CO3^2-, CO2(aq), MgCO3(aq)),
then reports saturation indices ``SI = log10(IAP / K)`` for calcite,
aragonite, gypsum, anhydrite, brucite, magnesite, dolomite and halite.

- Equilibrium constants and their temperature dependence: PHREEQC's
  ``pitzer.dat`` (``data/pitzer_seawater.py``); analytic expressions where
  given, otherwise van't Hoff from ``log_k`` and ``delta_h`` (PHREEQC v3
  manual [C4]). Second-hand; cross-checked against PHREEQC.
- Alkalinity is PHREEQC's: HCO3- 1, CO3^2- 2, OH- 1, MgOH+ 1, MgCO3 2,
  H+ -1, HSO4- -1 (equivalents per mole).
- Reference seawater (:func:`reference_seawater`): the Reference
  Composition of Millero et al. 2008 [S3] as tabulated in the TEOS-10
  manual [S4] Table D.3 (content verified). Br- is counted as Cl-; Sr2+,
  F-, B(OH)3 and B(OH)4- are not in the aqueous model and are dropped
  (their net charge is 1e-5 of the total), and the alkalinity is the
  carbonate + hydroxide part (borate dropped).
- All saturation indices are at 1 atm; the pressure dependence of K is not
  included.

The Newton solve runs a fixed number of steps on stop-gradient values and
finishes with a value-neutral Newton step, so first derivatives with
respect to the totals and T are the implicit-function ones.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_water.chemistry import pitzer as pz
from difflow_water.chemistry.data import pitzer_seawater as db
from difflow_water.chemistry.seawater import MW_SEA_SALT

LN10 = np.log(10.0)
R_J = 8.314462618

#: element totals the solver takes, in this order (mol/kg water)
TOTALS = ("Na", "K", "Mg", "Ca", "Cl", "SO4", "C", "Alk")

# TEOS-10 manual Table D.3 (Millero et al. 2008 Reference Composition):
# mole fractions X_j x 1e7 of the solute particles.
REFERENCE_COMPOSITION_X = {
    "Na+": 4188071, "Mg+2": 471678, "Ca+2": 91823, "K+": 91159, "Sr+2": 810,
    "Cl-": 4874839, "SO4-2": 252152, "HCO3-": 15340, "Br-": 7520, "CO3-2": 2134,
    "B(OH)4-": 900, "F-": 610, "OH-": 71, "B(OH)3": 2807, "CO2": 86,
}

#: passes of (species -> activity coefficients) inside the residual; only the
#: minor species (OH-, HCO3-, HSO4-, MgOH+, CO2, MgCO3) depend on gamma
N_GAMMA_PASSES = 3

MINERALS = ("Calcite", "Aragonite", "Gypsum", "Anhydrite", "Brucite", "Magnesite",
            "Dolomite", "Halite")


def _ci(s):
    return pz.CATIONS.index(s)


def _ai(s):
    return pz.ANIONS.index(s)


def _ni(s):
    return pz.NEUTRALS.index(s)


def log10_k(entry: dict, T) -> Array:
    """log10 K at T (K) from a ``pitzer_seawater`` reaction entry."""
    T = jnp.asarray(T, dtype=jnp.float64)
    if "analytic" in entry:
        a = entry["analytic"]
        return a[0] + a[1] * T + a[2] / T + a[3] * jnp.log10(T) + a[4] / T**2 + a[5] * T**2
    dh, unit = entry.get("delta_h", (0.0, "kJ"))
    dh_j = dh * (4184.0 if unit.lower().startswith("kcal") else 1000.0)
    return entry["log_k"] - dh_j / (R_J * LN10) * (1.0 / T - 1.0 / 298.15)


def seawater_feed(S, pH=8.1, T=298.15) -> dict[str, Array]:
    """Reference-Composition seawater totals with the alkalinity set by a pH.

    The pH (MacInnes scale, PHREEQC's) at temperature T fixes the carbonate
    split; the alkalinity that results is returned in ``"Alk"`` and is then
    conserved when the water is concentrated. The default 8.1 is a typical
    surface-seawater value (ILLUSTRATIVE; use the measured pH of the feed).

    Args:
        S: Salinity (kg/kg).
        pH: Feed pH (MacInnes scale).
        T: Temperature at which the pH applies (K).

    Returns:
        dict over :data:`TOTALS` (mol/kg water).
    """
    tot = reference_seawater(S)
    tot["Alk"] = speciate(tot, T, pH=pH)["alkalinity"]
    return tot


def reference_seawater(S, alkalinity=None) -> dict[str, Array]:
    """Element totals (mol/kg water) of Reference-Composition seawater.

    Args:
        S: Salinity, mass fraction of sea salt (kg/kg).
        alkalinity: Alkalinity (eq/kg water); None for the Reference
            Composition's carbonate + hydroxide alkalinity at this salinity.

    Returns:
        dict over :data:`TOTALS`.
    """
    S = jnp.asarray(S, dtype=jnp.float64)
    x = {k: v * 1e-7 for k, v in REFERENCE_COMPOSITION_X.items()}
    m = S / ((1.0 - S) * MW_SEA_SALT)  # moles of solute particles per kg water
    tot = {
        "Na": x["Na+"] * m, "K": x["K+"] * m, "Mg": x["Mg+2"] * m, "Ca": x["Ca+2"] * m,
        "Cl": (x["Cl-"] + x["Br-"]) * m, "SO4": x["SO4-2"] * m,
        "C": (x["HCO3-"] + x["CO3-2"] + x["CO2"]) * m,
        "Alk": (x["HCO3-"] + 2.0 * x["CO3-2"] + x["OH-"]) * m,
    }
    if alkalinity is not None:
        tot["Alk"] = jnp.asarray(alkalinity, dtype=jnp.float64)
    return tot


def concentrate(totals: dict, factor) -> dict:
    """Totals scaled by a concentration factor (molality basis); all species alike."""
    return {k: v * factor for k, v in totals.items()}


def _constants(T):
    sp = db.SPECIES
    return {k: log10_k(sp[k], T) * LN10 for k in sp}  # ln K


def _model(x, tot, T, lnK, params, gam):
    """Species molalities given unknowns x and log activity coefficients gam."""
    lgc, lga, lgn, lnaw = gam
    mH, mMg, mCa, mSO4, mCO3 = jnp.exp(x)
    g = lambda arr, i: arr[i]  # noqa: E731
    ln_aH = x[0] + g(lgc, _ci("H+"))
    ln_aMg = x[1] + g(lgc, _ci("Mg+2"))
    ln_aSO4 = x[3] + g(lga, _ai("SO4-2"))
    ln_aCO3 = x[4] + g(lga, _ai("CO3-2"))
    mOH = jnp.exp(lnK["OH-"] + lnaw - ln_aH - g(lga, _ai("OH-")))
    mHCO3 = jnp.exp(lnK["HCO3-"] + ln_aCO3 + ln_aH - g(lga, _ai("HCO3-")))
    mCO2 = jnp.exp(lnK["CO2"] + ln_aCO3 + 2.0 * ln_aH - lnaw - g(lgn, _ni("CO2")))
    mHSO4 = jnp.exp(lnK["HSO4-"] + ln_aSO4 + ln_aH - g(lga, _ai("HSO4-")))
    mMgOH = jnp.exp(lnK["MgOH+"] + ln_aMg + lnaw - ln_aH - g(lgc, _ci("MgOH+")))
    mMgCO3 = jnp.exp(lnK["MgCO3"] + ln_aMg + ln_aCO3 - g(lgn, _ni("MgCO3")))
    mc = jnp.zeros(pz.NC)
    mc = mc.at[_ci("H+")].set(mH).at[_ci("Na+")].set(tot["Na"]).at[_ci("K+")].set(tot["K"])
    mc = mc.at[_ci("Mg+2")].set(mMg).at[_ci("Ca+2")].set(mCa).at[_ci("MgOH+")].set(mMgOH)
    ma = jnp.zeros(pz.NA)
    ma = ma.at[_ai("OH-")].set(mOH).at[_ai("Cl-")].set(tot["Cl"]).at[_ai("SO4-2")].set(mSO4)
    ma = ma.at[_ai("HSO4-")].set(mHSO4).at[_ai("HCO3-")].set(mHCO3).at[_ai("CO3-2")].set(mCO3)
    mn = jnp.zeros(pz.NN).at[_ni("CO2")].set(mCO2).at[_ni("MgCO3")].set(mMgCO3)
    return mc, ma, mn


def _gammas(mc, ma, mn, T, params):
    r = pz.activity(mc, ma, mn, T, params)
    return (r["ln_gamma_c"], r["ln_gamma_a"], r["ln_gamma_n"], r["ln_aw"]), r


def _passes(x, tot, T, lnK, params):
    """Species from x, with gamma iterated N_GAMMA_PASSES times (one traced body)."""
    gam0 = (jnp.zeros(pz.NC), jnp.zeros(pz.NA), jnp.zeros(pz.NN), jnp.asarray(0.0))
    m0 = _model(x, tot, T, lnK, params, gam0)

    def body(_, m):
        gam, _ = _gammas(*m, T, params)
        return _model(x, tot, T, lnK, params, gam)

    return jax.lax.fori_loop(0, N_GAMMA_PASSES, body, m0)


def _residual(x, tot, T, lnK, params, pH=None):
    # activity coefficients from the current composition (self-consistent at the root)
    mc, ma, mn = _passes(x, tot, T, lnK, params)
    gam, act = _gammas(mc, ma, mn, T, params)
    r = _balances(mc, ma, mn, tot)
    if pH is None:
        return r
    # fixed pH (MacInnes scale) replaces the alkalinity balance
    ph_mac = (-(x[0] + gam[0][_ci("H+")]) - _macinnes_shift(gam[1][_ai("Cl-")], act["I"], T, params)) / LN10
    return r.at[4].set(ph_mac - pH)


def _macinnes_shift(ln_g_cl, I, T, params):
    """``ln g_Cl - ln g_pm,KCl(I)``: the MacInnes scale offset of ln g_i per unit charge."""
    kcl = pz.activity(jnp.zeros(pz.NC).at[_ci("K+")].set(I), jnp.zeros(pz.NA).at[_ai("Cl-")].set(I),
                      jnp.zeros(pz.NN), T, params)
    return ln_g_cl - 0.5 * (kcl["ln_gamma_c"][_ci("K+")] + kcl["ln_gamma_a"][_ai("Cl-")])


def _balances(mc, ma, mn, tot):
    c, a, n = mc, ma, mn
    mg = c[_ci("Mg+2")] + c[_ci("MgOH+")] + n[_ni("MgCO3")]
    ca = c[_ci("Ca+2")]
    so4 = a[_ai("SO4-2")] + a[_ai("HSO4-")]
    ct = a[_ai("CO3-2")] + a[_ai("HCO3-")] + n[_ni("CO2")] + n[_ni("MgCO3")]
    alk = (a[_ai("HCO3-")] + 2.0 * a[_ai("CO3-2")] + 2.0 * n[_ni("MgCO3")] + a[_ai("OH-")]
           + c[_ci("MgOH+")] - c[_ci("H+")] - a[_ai("HSO4-")])
    return jnp.stack([mg / tot["Mg"] - 1.0, ca / tot["Ca"] - 1.0, so4 / tot["SO4"] - 1.0,
                      ct / tot["C"] - 1.0, (alk - tot["Alk"]) / tot["C"]])


def speciate(totals: dict, T=298.15, n_iter: int = 40, pH=None) -> dict:
    """Speciate a seawater (or brine) at temperature T.

    Args:
        totals: Element totals over :data:`TOTALS` (mol/kg water; ``Alk`` in
            eq/kg water), e.g. from :func:`reference_seawater`.
        T: Temperature (K).
        n_iter: Newton iterations (fixed; damped to one log unit per step).
        pH: When given, the pH (MacInnes scale, as PHREEQC reads and prints
            it) is fixed and ``totals["Alk"]`` is ignored; the result's
            ``alkalinity`` is then the computed one.

    Returns:
        dict with ``molality`` and ``ln_gamma`` (per species name), ``pH``
        (``-log10 a_H``, unscaled), ``pH_macinnes`` (MacInnes scale, what
        PHREEQC prints), ``ionic_strength``, ``osmotic_coefficient``,
        ``water_activity``, ``SI`` (per mineral), ``residual`` (max abs
        balance error, relative).
    """
    need = [k for k in TOTALS if not (k == "Alk" and pH is not None)]
    missing = [k for k in need if k not in totals]
    if missing:
        raise ValueError(f"speciate needs totals for {missing} (mol/kg water)")
    tot = {k: jnp.asarray(totals.get(k, 0.0), dtype=jnp.float64) for k in TOTALS}
    ph = jnp.asarray(0.0 if pH is None else pH, dtype=jnp.float64)
    return _speciate(tot, jnp.asarray(T, dtype=jnp.float64), ph, n_iter, pH is not None)


@partial(jax.jit, static_argnums=(3, 4))
def _speciate(tot, T, ph_value, n_iter, fixed_ph):
    pH = ph_value if fixed_ph else None
    params = pz.parameters(T)
    lnK = _constants(T)
    f = lambda x: _residual(x, tot, T, lnK, params, pH)  # noqa: E731
    x0 = jnp.log(jnp.stack([1e-8, tot["Mg"], tot["Ca"], tot["SO4"], 0.1 * tot["C"]]))

    def step(_, x):
        r = f(x)
        J = jax.jacfwd(f)(x)
        dx = jnp.linalg.solve(J, -r)
        dx = dx / jnp.maximum(1.0, jnp.max(jnp.abs(dx)))
        return x + dx

    xs = jax.lax.stop_gradient(jax.lax.fori_loop(0, n_iter, step, jax.lax.stop_gradient(x0)))
    # value-neutral Newton step: exact first derivatives by the implicit function theorem
    Js = jax.lax.stop_gradient(jax.jacfwd(f)(xs))
    r = f(xs)
    x = xs - jnp.linalg.solve(Js, r - jax.lax.stop_gradient(r))

    mc, ma, mn = _passes(x, tot, T, lnK, params)
    gam, act = _gammas(mc, ma, mn, T, params)
    lgc, lga, lgn, lnaw = gam
    lnm = {s: jnp.log(jnp.maximum(v, 1e-300)) for s, v in
           list(zip(pz.CATIONS, mc)) + list(zip(pz.ANIONS, ma)) + list(zip(pz.NEUTRALS, mn))}
    lng = dict(zip(pz.CATIONS, lgc)) | dict(zip(pz.ANIONS, lga)) | dict(zip(pz.NEUTRALS, lgn))
    lna = {s: lnm[s] + lng[s] for s in lnm}
    lna["H2O"] = lnaw
    si = saturation_indices(lna, T)
    # MacInnes scale: ln g_i' = ln g_i + z_i (ln g_Cl - ln g_pm,KCl(I))
    I = act["I"]
    shift = _macinnes_shift(lng["Cl-"], I, T, params)
    pH_unscaled = -lna["H+"] / LN10
    m = {s: jnp.exp(v) for s, v in lnm.items()}
    alk = (m["HCO3-"] + 2.0 * m["CO3-2"] + 2.0 * m["MgCO3"] + m["OH-"] + m["MgOH+"]
           - m["H+"] - m["HSO4-"])
    return {
        "molality": m,
        "ln_gamma": lng,
        "pH": pH_unscaled,
        "pH_macinnes": pH_unscaled - shift / LN10,
        "alkalinity": alk,
        "ionic_strength": I,
        "osmotic_coefficient": act["phi"],
        "water_activity": jnp.exp(lnaw),
        "SI": si,
        "residual": jnp.max(jnp.abs(_balances(mc, ma, mn, tot)[:4 if pH is not None else 5])),
    }


def saturation_indices(lna: dict, T) -> dict[str, Array]:
    """SI = log10(IAP/K) for :data:`MINERALS` from log activities (natural log)."""
    a = lna
    iap = {
        "Calcite": a["Ca+2"] + a["CO3-2"],
        "Aragonite": a["Ca+2"] + a["CO3-2"],
        "Gypsum": a["Ca+2"] + a["SO4-2"] + 2.0 * a["H2O"],
        "Anhydrite": a["Ca+2"] + a["SO4-2"],
        "Brucite": a["Mg+2"] + 2.0 * a["OH-"],
        "Magnesite": a["Mg+2"] + a["CO3-2"],
        "Dolomite": a["Ca+2"] + a["Mg+2"] + 2.0 * a["CO3-2"],
        "Halite": a["Na+"] + a["Cl-"],
    }
    return {k: iap[k] / LN10 - log10_k(db.PHASES[k], T) for k in MINERALS}
