"""H2S-CO2-alkanolamine-water equilibrium: the Kent-Eisenberg model.

Kent and Eisenberg (1976) treat the loaded amine solution as an ideal
electrolyte solution (every activity coefficient one) in which the acid
gases dissolve physically (Henry's law) and react:

====  =====================================  ==========================
 a    2 H2O = H3O+ + OH-                      K_w = [H+][OH-]
 b    CO2 + 2 H2O = H3O+ + HCO3-              K_b = [H+][HCO3-]/[CO2]
 c    HCO3- + H2O = H3O+ + CO3--              K_c = [H+][CO3--]/[HCO3-]
 d    H2S + H2O = H3O+ + HS-                  K_d = [H+][HS-]/[H2S]
 e    HS- + H2O = H3O+ + S--                  K_e = [H+][S--]/[HS-]
 f    RR'NH2+ + H2O = H3O+ + RR'NH            K_f = [H+][Am]/[AmH+]
 h    RR'NCOO- + H2O = RR'NH + HCO3-          K_h = [Am][HCO3-]/[Carb]
====  =====================================  ==========================

(``h``, the carbamate reversion, only for primary and secondary amines.)
Concentrations are molalities (mol per kg of water), ``p_i = H_i m_i``.
The reactions a-e keep their thermodynamic constants; the amine reactions
f and h take APPARENT constants regressed to loading data, which is where
the model puts all the non-ideality (Kent & Eisenberg 1976, as described by
Austgen 1989, p. 78, and Posey 1996, pp. 35-36).

Here ``ln K_f`` is the thermodynamic protonation constant (Posey 1996,
Table 5.8, itself fitted to measured pKa data) plus a fitted correction
``d0 + d1 ln(alpha_CO2 + alpha_H2S + alpha0)``, so the temperature dependence is
the measured heat of protonation rather than a fit to a narrow band of VLE
data. The loading term is the one Jou, Mather & Otto (1982) found the
apparent constant needs (the "extended" Kent-Eisenberg form; Patil, Malik
& Jobson 2006, Eq. 19, whose fitted CO2-MDEA coefficient is -0.40; ours is
-0.49). Without it the model is 80 % off at low loading, the known failing
of the original Kent-Eisenberg model. ``alpha0 = 0.001``, below the
lowest loading fitted (0.003), keeps the correction finite in an unloaded
solvent, where a bare ``ln alpha`` would make the amine 10^5 times weaker
a base. For DEA the
carbamate constant is fitted as ``a + b/T``. The fits are in
``tests/reference/amine_ke_fit.py`` and are tested against data that
were held out of them (``tests/cc/test_acid_gas_vle.py``).

The MDEA constants were fitted at 10-40 C. Against Jou, Mather & Otto
(1982), held out (25-120 C, 12-49 wt%, p <= 1 MPa; AAPD in p, printed by
``tests/reference/amine_ke_fit.py``), the error FALLS with temperature: CO2
is 11-13 % at 100-120 C and 14-19 % at 70 C, against 53-67 % at 25-40 C;
H2S at 100-120 C is 20 % (23 wt%) to 100 % (49 wt%), mostly from loadings
below 0.03, where the model is up to three times too high. The
temperature dependence is the measured heat of protonation, and it carries:
a fitted ``d3 (1/T - 1/313 K)`` term came out near zero. What does not
carry is the absolute level at 25-70 C, where the sources disagree with
each other: at 25 C and about 50 wt% Jou's H2S pressures are a third to
two fifths of Huttenhuis et al.'s (2007) at the same loading. Refits that added Jou's
data, a ``d2 ln m`` term, a temperature term or a per-gas Henry's-law
factor ``exp(h w)`` traded one source against the other and did not lower
the held-out errors overall, so the constants were left as fitted.

Sources of the constants (``ln K = C1 + C2/T + C3 ln T + C4 T``, T in K):

* a-d: Edwards, Maurer, Newman & Prausnitz (1978), AIChE J. 24:966, on the
  molality scale as tabulated by Posey (1996), Table 5.8, p. 107. (Austgen
  1989, Table 4.1, gives the same constants on the mole-fraction scale;
  they differ by ln(1000/18.015) per solute, as they should.)
* e: Giggenbach (1971); Meyer et al. (1983), via Austgen (1989), Table 4.1
  (mole fraction, 14-70 C), shifted by ln(1000/18.015) to molality.
  ``pK_e`` is about 17 at 25 C, so S-- is a trace in every amine solution.
* f: Posey (1996), Table 5.8: MDEA fitted to Schwabe et al. (1959), Kim et
  al. (1987), Oscarson et al. (1989); DEA to Bower et al. (1962), Oscarson
  et al. (1989). They give pKa 8.53 (MDEA) and 8.87 (DEA) at 25 C.
* Henry's constants (mole fraction, Pa): H2S, Edwards et al. (1978); CO2,
  Chen et al. (1979); both as tabulated by Austgen (1989), Table 4.2, p. 94.
  Converted to molality by the molar mass of water (dilute limit).

References:
    Kent RL, Eisenberg B (1976). Better data for amine treating.
        Hydrocarbon Processing 55(2):87-90.
    Austgen DM Jr (1989). A model of vapor-liquid equilibria for acid
        gas-alkanolamine-water systems. PhD dissertation, The University
        of Texas at Austin (OSTI 6346766).
    Posey ML (1996). PhD dissertation, The University of Texas at Austin
        (title as usually cited, "Thermodynamic model for acid gas loaded
        aqueous alkanolamine solutions"; not confirmed against a catalog).
    Patil P, Malik Z, Jobson M (2006). Prediction of CO2 and H2S
        solubility in aqueous MDEA solutions using an extended Kent and
        Eisenberg model. Distillation & Absorption 2006, IChemE Symp.
        Ser. 152:498-510.
    Edwards TJ, Maurer G, Newman J, Prausnitz JM (1978). Vapor-liquid
        equilibria in multicomponent aqueous solutions of volatile weak
        electrolytes. AIChE J 24:966-976.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, lax

__all__ = [
    "KentEisenbergAmine",
    "KE_AMINES",
    "ke_amine",
    "amine_molality",
    "ke_speciate",
    "ke_partial_pressures",
    "ke_loading",
]

#: molar mass of water (kg/mol)
M_WATER = 0.018015

# ln K = C1 + C2/T + C3 ln T + C4 T, molality scale (see the module docstring)
LNK_WATER = (140.932, -13445.9, -22.4773, 0.0)        # Posey 1996 Table 5.8 (Edwards 1978)
LNK_CO2 = (235.482, -12092.1, -36.7816, 0.0)          # Posey 1996 Table 5.8 (Edwards 1978)
LNK_HCO3 = (220.067, -12431.7, -35.4819, 0.0)         # Posey 1996 Table 5.8 (Edwards 1978)
LNK_H2S = (218.599, -12995.4, -33.5471, 0.0)          # Posey 1996 Table 5.8 (Edwards 1978)
LNK_HS = (-32.0 + 4.01655, -3338.0, 0.0, 0.0)          # Austgen 1989 Table 4.1 row e, to molality
# Henry's constants, mole fraction scale: ln(H / Pa)
LNH_H2S = (358.138, -13236.8, -55.0551, 0.059565)     # Edwards 1978, via Austgen 1989 Table 4.2
LNH_CO2 = (170.7126, -8477.711, -21.9574, 0.005781)   # Chen 1979, via Austgen 1989 Table 4.2


#: loading offset in the apparent protonation constant's loading term
ALPHA0 = 1e-3


def _lnk(c, T):
    return c[0] + c[1] / T + c[2] * jnp.log(T) + c[3] * T


def henry_molal(T, gas: str) -> Array:
    """Henry's constant on the molality scale (Pa kg/mol): ``p = H m``."""
    c = {"CO2": LNH_CO2, "H2S": LNH_H2S}[gas]
    return jnp.exp(_lnk(c, jnp.asarray(T, dtype=float))) * M_WATER


@dataclass(frozen=True)
class KentEisenbergAmine:
    """One amine's Kent-Eisenberg constants.

    Attributes:
        name: Amine name.
        MW: Molar mass (g/mol).
        protonation: ``(C1, C2, C3, C4)`` of the thermodynamic ``ln K_f``
            (molality), reaction f.
        apparent: ``(d0, d1, d2)`` of the fitted apparent correction,
            ``ln K_f,app - ln K_f = d0 + d1 ln(alpha_CO2 + alpha_H2S + ALPHA0)
            + d2 ln m_amine`` (the loading and concentration terms of the
            extended Kent-Eisenberg form, Patil, Malik & Jobson 2006, Eq. 19).
        carbamate: ``(a, b)`` of the apparent ``ln K_h = a + b/T``, or
            ``None`` for a tertiary amine (no carbamate).
        source: Where the constants come from.
    """

    name: str
    MW: float
    protonation: tuple
    apparent: tuple
    carbamate: tuple | None
    source: str

    def ln_Kf(self, T, alpha, ln_m):
        d0, d1, d2 = self.apparent
        return (_lnk(self.protonation, T) + d0 + d1 * jnp.log(alpha + ALPHA0)
                + d2 * ln_m)

    def ln_Kh(self, T):
        a, b = self.carbamate
        return a + b / T


# The fitted numbers (apparent, carbamate) are written by
# tests/reference/amine_ke_fit.py; the test suite refits and compares.
KE_AMINES = {
    "MDEA": KentEisenbergAmine(
        name="MDEA", MW=119.163,
        protonation=(-59.55, -1709.0, 8.01, 0.0),
        apparent=(-1.8393901031008881, -0.49244575785329014, 0.0),
        carbamate=None,
        source="ln K_f: Posey 1996 Table 5.8; the apparent correction fitted to Austgen 1989 Table 5.3 (CO2) "
               "and Huttenhuis et al. 2007 Tables 3-4 (H2S)"),
    "DEA": KentEisenbergAmine(
        name="DEA", MW=105.136,
        protonation=(-26.5, -4035.0, 3.44, 0.0),
        apparent=(-1.8603943315013263, -0.8409584058284636, 0.0),
        carbamate=(4.9516621374942265, -1973.3912668298922),
        source="ln K_f: Posey 1996 Table 5.8; the apparent correction and ln K_h fitted to Barreau et al. "
               "2006 Tables 1-4 (CO2 and H2S)"),
}


def ke_amine(name: str | KentEisenbergAmine) -> KentEisenbergAmine:
    """Look up an amine (``"MDEA"``, ``"DEA"``) or pass one through."""
    if isinstance(name, KentEisenbergAmine):
        return name
    try:
        return KE_AMINES[name]
    except KeyError:
        raise ValueError(f"no Kent-Eisenberg constants for {name!r}; have {sorted(KE_AMINES)}")


def amine_molality(w_amine, amine: str | KentEisenbergAmine) -> Array:
    """Amine molality (mol/kg water) of an unloaded solution of mass fraction ``w_amine``."""
    am = ke_amine(amine)
    w = jnp.asarray(w_amine, dtype=float)
    return w / (am.MW * 1e-3 * (1.0 - w))


def _constants(am, T, alpha, ln_m):
    lnK = dict(w=_lnk(LNK_WATER, T), b=_lnk(LNK_CO2, T), c=_lnk(LNK_HCO3, T),
               d=_lnk(LNK_H2S, T), e=_lnk(LNK_HS, T), f=am.ln_Kf(T, alpha, ln_m))
    if am.carbamate is not None:
        lnK["h"] = am.ln_Kh(T)
    return lnK


def _species(am, lnK, lh, la, lc, ls):
    """Every molality from ln[H+], ln[Am], ln[CO2], ln[H2S]."""
    lhco3 = lnK["b"] + lc - lh
    lhs = lnK["d"] + ls - lh
    sp = {
        "H+": jnp.exp(lh), "OH-": jnp.exp(lnK["w"] - lh),
        "Am": jnp.exp(la), "AmH+": jnp.exp(la + lh - lnK["f"]),
        "CO2": jnp.exp(lc), "HCO3-": jnp.exp(lhco3),
        "CO3--": jnp.exp(lnK["c"] + lhco3 - lh),
        "H2S": jnp.exp(ls), "HS-": jnp.exp(lhs), "S--": jnp.exp(lnK["e"] + lhs - lh),
    }
    sp["Carb-"] = (jnp.exp(la + lhco3 - lnK["h"]) if am.carbamate is not None
                   else jnp.zeros_like(lh))
    return sp


def _sums(sp):
    amine = sp["Am"] + sp["AmH+"] + sp["Carb-"]
    carbon = sp["CO2"] + sp["HCO3-"] + sp["CO3--"] + sp["Carb-"]
    sulfur = sp["H2S"] + sp["HS-"] + sp["S--"]
    pos = sp["H+"] + sp["AmH+"]
    neg = sp["OH-"] + sp["HCO3-"] + 2.0 * sp["CO3--"] + sp["HS-"] + 2.0 * sp["S--"] + sp["Carb-"]
    return amine, carbon, sulfur, pos, neg


def _newton(res, z0, cap, max_iter=100, tol=1e-12):
    """Damped Newton with a step cap and backtracking; returns (z, iterations, max|r|)."""
    alphas = 0.5 ** jnp.arange(16.0)

    def norm(z):
        r = res(z)
        return jnp.where(jnp.all(jnp.isfinite(r)), jnp.max(jnp.abs(r)), jnp.inf)

    def body(state):
        z, k, _ = state
        dz = -jnp.linalg.solve(jax.jacfwd(res)(z), res(z))
        dz = jnp.where(jnp.all(jnp.isfinite(dz)), dz, 0.0)
        dz = dz / jnp.maximum(1.0, jnp.max(jnp.abs(dz)) / cap)
        trials = jax.vmap(lambda a: norm(z + a * dz))(alphas)
        ok = trials < norm(z)
        a = jnp.where(jnp.any(ok), alphas[jnp.argmax(ok)], alphas[-1])
        z = z + a * dz
        return z, k + 1, norm(z)

    def cond(state):
        _, k, err = state
        return (k < max_iter) & (err > tol)

    return lax.while_loop(cond, body, (z0, jnp.asarray(0), norm(z0)))


def _implicit(res, z_star, theta):
    """One Newton step from a stop-gradient root: value-neutral, exact first derivatives."""
    theta_sg = jax.tree_util.tree_map(lax.stop_gradient, theta)
    z_star = lax.stop_gradient(z_star)
    J = lax.stop_gradient(jax.jacfwd(lambda z: res(z, theta_sg))(z_star))
    return z_star - jnp.linalg.solve(J, res(z_star, theta))


#: a total below this fraction of the amine is held at it, so its log stays finite
TRACE = 1e-30  # below any loading a stripped bottoms reaches (~1e-17)


def ke_speciate(amine, T, m_amine, alpha_CO2, alpha_H2S) -> dict:
    """Speciation of a loaded amine solution, and the acid gases' partial pressures.

    Args:
        amine: ``"MDEA"``, ``"DEA"`` or a :class:`KentEisenbergAmine`.
        T: Temperature (K).
        m_amine: Total amine molality (mol/kg water).
        alpha_CO2: CO2 loading (mol CO2 / mol amine).
        alpha_H2S: H2S loading (mol H2S / mol amine). A loading below
            ``TRACE`` is held at ``TRACE`` (a log needs it).

    Returns:
        dict with ``p_CO2``, ``p_H2S`` (Pa), ``species`` (molalities),
        ``pH`` (on the molality scale), ``converged``, ``residual``.

    Example:
        >>> m = amine_molality(0.5, "MDEA")
        >>> out = ke_speciate("MDEA", 313.15, m, 0.1, 0.05)
        >>> float(out["p_H2S"]) > 0
        True
    """
    am = ke_amine(amine)
    theta = dict(T=jnp.asarray(T, dtype=float), m=jnp.asarray(m_amine, dtype=float),
                 aC=jnp.maximum(jnp.asarray(alpha_CO2, dtype=float), TRACE),
                 aS=jnp.maximum(jnp.asarray(alpha_H2S, dtype=float), TRACE))

    def consts(th):
        return _constants(am, th["T"], th["aC"] + th["aS"], jnp.log(th["m"]))

    def res(z, th):
        lnK = consts(th)
        sp = _species(am, lnK, *z)
        a, c, s, pos, neg = _sums(sp)
        return jnp.stack([jnp.log(a / th["m"]), jnp.log(c / (th["aC"] * th["m"])),
                          jnp.log(s / (th["aS"] * th["m"])), jnp.log(pos / neg)])

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    lnK0 = consts(th0)
    # start: every acid molecule has protonated one amine (capped at 95 %)
    acid = jnp.minimum(th0["aC"] + th0["aS"], 0.95)
    m0 = th0["m"]
    la0 = jnp.log(m0 * (1.0 - acid))
    lh0 = lnK0["f"] + jnp.log(acid / (1.0 - acid))
    lc0 = jnp.log(th0["aC"] * m0) + lh0 - lnK0["b"]
    ls0 = jnp.log(th0["aS"] * m0) + lh0 - lnK0["d"]
    z0 = jnp.stack([lh0, la0, lc0, ls0])
    z_star, k, err = _newton(lambda z: res(z, th0), z0, cap=3.0)
    z = _implicit(res, z_star, theta)
    sp = _species(am, consts(theta), *z)
    return {
        "p_CO2": henry_molal(theta["T"], "CO2") * sp["CO2"],
        "p_H2S": henry_molal(theta["T"], "H2S") * sp["H2S"],
        "species": sp, "pH": -z[0] / jnp.log(10.0),
        "converged": err < 1e-9, "residual": err, "iterations": k,
    }


def ke_partial_pressures(amine, T, w_amine, alpha_CO2, alpha_H2S) -> tuple[Array, Array]:
    """``(p_CO2, p_H2S)`` (Pa) over a solution of amine mass fraction ``w_amine``
    (acid-gas free) at loadings ``alpha_CO2``, ``alpha_H2S``."""
    out = ke_speciate(amine, T, amine_molality(w_amine, amine), alpha_CO2, alpha_H2S)
    return out["p_CO2"], out["p_H2S"]


def ke_loading(amine, T, w_amine, p_CO2, p_H2S) -> dict:
    """Equilibrium loadings for given acid-gas partial pressures (the inverse).

    With the partial pressures fixed, the molecular CO2 and H2S follow from
    Henry's law, and the unknowns are ``ln[H+]`` and ``ln[Am]`` (amine and
    charge balances), plus the log of the total loading, on which the
    apparent protonation constant depends.

    Args:
        amine: Amine name or constants.
        T: Temperature (K).
        w_amine: Amine mass fraction (acid-gas free).
        p_CO2, p_H2S: Partial pressures (Pa); one may be zero.

    Returns:
        dict with ``alpha_CO2``, ``alpha_H2S``, ``species``, ``converged``.
    """
    am = ke_amine(amine)
    m = amine_molality(w_amine, am)
    T = jnp.asarray(T, dtype=float)
    floor = 1e-20
    theta = dict(T=T, m=m,
                 lc=jnp.log(jnp.maximum(jnp.asarray(p_CO2, dtype=float), floor)
                            / henry_molal(T, "CO2")),
                 ls=jnp.log(jnp.maximum(jnp.asarray(p_H2S, dtype=float), floor)
                            / henry_molal(T, "H2S")))

    def species(z, th):
        lnK = _constants(am, th["T"], jnp.exp(z[2]), jnp.log(th["m"]))
        return _species(am, lnK, z[0], z[1], th["lc"], th["ls"])

    def res(z, th):
        a, c, s, pos, neg = _sums(species(z, th))
        return jnp.stack([jnp.log(a / th["m"]), jnp.log(pos / neg),
                          jnp.log((c + s) / th["m"]) - z[2]])

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    # start from half the amine protonated, at a loading of one half
    lnK0 = _constants(am, th0["T"], 0.5, jnp.log(th0["m"]))
    z0 = jnp.stack([lnK0["f"], jnp.log(0.5 * th0["m"]), jnp.log(0.5)])
    z_star, k, err = _newton(lambda z: res(z, th0), z0, cap=3.0)
    z = _implicit(res, z_star, theta)
    sp = species(z, theta)
    _, c, s, _, _ = _sums(sp)
    return {"alpha_CO2": c / m, "alpha_H2S": s / m, "species": sp,
            "converged": err < 1e-9, "residual": err, "iterations": k}
