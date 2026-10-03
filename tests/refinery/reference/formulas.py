"""The thermodynamic model, written from the published equations.

Every function takes a math namespace ``m`` with ``exp``, ``log`` and
``sqrt``: :data:`NUMPY` evaluates on floats and arrays, :func:`pyomo_math`
builds Pyomo expressions, so the property model inside the IPOPT column is
the same text that layer 2 checks on numbers. Nothing here imports difflow:
these are transcriptions from the sources cited, and a transcription error
in either code base shows up as a disagreement.

The model is the one ``ColumnThermo`` states it is -- Raoult K-values over
Lee-Kesler vapour pressures, ideal-gas enthalpy from a cubic Cp, liquid
enthalpy one Watson latent heat below it -- because layer 3 is meant to test
the *column*, and the column only makes sense against the same property
model. Layer 2 is where the property model itself is tested, against IDAES
and against Peng-Robinson.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

R = 8.314462618  # J/mol/K (CODATA 2018)
T_REF = 298.15  # enthalpy datum, ideal gas
P_ATM = 101325.0

NUMPY = SimpleNamespace(exp=np.exp, log=np.log, sqrt=np.sqrt)


def pyomo_math():
    import pyomo.environ as pyo

    return SimpleNamespace(exp=pyo.exp, log=pyo.log, sqrt=pyo.sqrt)


# -----------------------------------------------------------------------------
# Vapour pressure
# -----------------------------------------------------------------------------


def lee_kesler_ln_pr(m, Tr, omega):
    """``ln(Psat/Pc)``, Lee & Kesler, AIChE J. 21, 510 (1975), Eqs. as given in
    Poling, Prausnitz & O'Connell, *Properties of Gases and Liquids*, 5th ed.
    (2001), Eq. 7-4.1 to 7-4.3."""
    f0 = 5.92714 - 6.09648 / Tr - 1.28862 * m.log(Tr) + 0.169347 * Tr**6
    f1 = 15.2518 - 15.6875 / Tr - 13.4721 * m.log(Tr) + 0.43577 * Tr**6
    return f0 + omega * f1


def lee_kesler_psat(m, T, Tc, Pc, omega):
    return Pc * m.exp(lee_kesler_ln_pr(m, T / Tc, omega))


def lee_kesler_omega(Tb, Tc, Pc):
    """The acentric factor that makes Lee-Kesler boil at ``Tb`` at 1 atm --
    Lee & Kesler's own definition of omega for a petroleum fraction
    (PPO 5th ed. Eq. 2-3.3)."""
    Tbr = Tb / Tc
    m = NUMPY
    lnp = math.log(P_ATM / Pc)
    f0 = 5.92714 - 6.09648 / Tbr - 1.28862 * m.log(Tbr) + 0.169347 * Tbr**6
    f1 = 15.2518 - 15.6875 / Tbr - 13.4721 * m.log(Tbr) + 0.43577 * Tbr**6
    return (lnp - f0) / f1


# Water, IAPWS-IF97 region 4 (the saturation equation), IAPWS R7-97(2012)
# Eq. 30 and Table 34. Deliberately not the Wagner-Pruss equation difflow uses:
# two different formulations of the same steam tables.
_IF97_N = (
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
)


def water_psat(m, T):
    """Saturation pressure of water (Pa), IAPWS-IF97 Eq. 30, 273.15-647.096 K."""
    n = _IF97_N
    th = T + n[8] / (T - n[9])
    A = th**2 + n[0] * th + n[1]
    B = n[2] * th**2 + n[3] * th + n[4]
    C = n[5] * th**2 + n[6] * th + n[7]
    return 1e6 * (2 * C / (-B + m.sqrt(B**2 - 4 * A * C))) ** 4


# -----------------------------------------------------------------------------
# Enthalpy
# -----------------------------------------------------------------------------


def h_ideal_gas(T, cp):
    """``integral(T_REF -> T) (a + bT + cT^2 + dT^3) dT`` (J/mol)."""
    a, b, c, d = cp

    def F(t):
        return a * t + b * t**2 / 2 + c * t**3 / 3 + d * t**4 / 4

    return F(T) - F(T_REF)


def watson_base(m, T, Tc, eps):
    """``1 - T/Tc`` floored at zero; ``eps > 0`` smooths the floor as
    ``(x + sqrt(x^2 + eps^2))/2`` (the model difflow states), ``eps = 0`` is
    the exact ``max(x, 0)`` of Watson (1943)."""
    x = 1.0 - T / Tc
    if eps == 0:
        return np.maximum(x, 0.0)
    return 0.5 * (x + m.sqrt(x * x + eps * eps))


def latent_heat(m, T, Tb, Tc, hvap_nb, eps, n=0.38):
    """Watson (1943): ``dHvap(T) = dHvap(Tb) ((1-Tr)/(1-Tbr))^n``, n = 0.38."""
    return hvap_nb * (watson_base(m, T, Tc, eps) / watson_base(m, Tb, Tc, eps)) ** n


# Water constants (the property model's, stated in difflow_refinery.thermo and
# repeated here as data): ideal-gas Cp from Reid, Prausnitz & Poling 4th ed.
# Appendix A; latent heat 40.66 kJ/mol at 373.124 K (IAPWS-95 gives 40.657);
# critical point 647.096 K.
WATER = {
    "MW": 18.015,
    "cp_ig": (32.24, 1.924e-3, 1.055e-5, -3.596e-9),
    "Tb": 373.124,
    "Tc": 647.096,
    "hvap_nb": 40660.0,
}


def water_h_vapor(T):
    return h_ideal_gas(T, WATER["cp_ig"])


def water_h_liquid(m, T, eps):
    return water_h_vapor(T) - latent_heat(m, T, WATER["Tb"], WATER["Tc"], WATER["hvap_nb"], eps)


# -----------------------------------------------------------------------------
# Plain-number property model over a component table (layer 2 and init)
# -----------------------------------------------------------------------------


class Props:
    """The ideal property model evaluated on numbers, for a component table
    as :func:`case.component_data` writes it."""

    def __init__(self, comp: dict, eps: float = 0.01):
        self.c = comp
        self.eps = eps
        self.n = len(comp["names"])
        self.Tc = np.asarray(comp["Tc"])
        self.Pc = np.asarray(comp["Pc"])
        self.Tb = np.asarray(comp["Tb"])
        self.w = np.asarray(comp["omega_vp"])
        self.hv = np.asarray(comp["hvap_nb"])
        self.cp = np.asarray(comp["cp_ig"]).T
        self.MW = np.asarray(comp["MW"])
        self.SG = np.asarray(comp["SG"])

    def psat(self, T):
        return lee_kesler_psat(NUMPY, T, self.Tc, self.Pc, self.w)

    def K(self, T, P):
        return self.psat(T) / P

    def hV(self, T):
        return h_ideal_gas(T, self.cp)

    def hL(self, T):
        return self.hV(T) - latent_heat(NUMPY, T, self.Tb, self.Tc, self.hv, self.eps)

    def std_volume(self, F):
        """Standard liquid volume, m^3/s (water at 60 F = 999.016 kg/m^3,
        API MPMS 11.1)."""
        return float(np.sum(F * self.MW / (1000.0 * self.SG * 999.016)))


def rachford_rice(z, K):
    """Vapour fraction of an ideal flash; 0 or 1 outside the two-phase range."""
    from scipy.optimize import brentq

    z, K = np.asarray(z), np.asarray(K)
    if np.sum(z * K) <= 1.0:
        return 0.0
    if np.sum(z / K) <= 1.0:
        return 1.0

    def g(b):
        return np.sum(z * (K - 1) / (1 + b * (K - 1)))

    lo = max(0.0, np.max(1 / (1 - K[K > 1])) if np.any(K > 1) else 0.0) + 1e-14
    hi = min(1.0, np.min(1 / (1 - K[K < 1])) if np.any(K < 1) else 1.0) - 1e-14
    lo, hi = max(lo, 0.0), min(hi, 1.0)
    return brentq(g, lo, hi, xtol=1e-15)


def flash_enthalpy(props: Props, F, T, P):
    """Ideal flash of molar flows ``F`` at ``T, P``: ``(beta, H)``."""
    F = np.asarray(F)
    z = F / F.sum()
    K = props.K(T, P)
    b = rachford_rice(z, K)
    x = z / (1 + b * (K - 1))
    y = K * x
    Ft = F.sum()
    H = Ft * ((1 - b) * np.sum(x * props.hL(T)) + b * np.sum(y * props.hV(T)))
    return b, H


# -----------------------------------------------------------------------------
# Product inspections
# -----------------------------------------------------------------------------


def product_inspection(props: Props, F, percents=(5, 10, 50, 90, 95)) -> dict:
    """Standard volume, mass, SG, API and the TBP curve of molar flows ``F``.

    SG is mass over standard volume relative to water at 60 F; API from SG
    (API MPMS 11.1). The TBP curve puts each component's standard volume at
    its normal boiling point, centred: the cumulative fraction at component
    ``j`` (in boiling order) is ``(sum_{k<j} v_k + v_j/2)/sum v``, linear
    between components and flat outside the first and last one present --
    the convention difflow documents for a pseudo-component product, written
    out again here.
    """
    F = np.asarray(F, dtype=float)
    v = F * props.MW / (1000.0 * props.SG * 999.016)
    mass = float(np.sum(F * props.MW) / 1000.0)
    vol = float(np.sum(v))
    sg = mass / (vol * 999.016)
    order = np.argsort(props.Tb)
    vs, Tb = v[order], props.Tb[order]
    centre = (np.cumsum(vs) - 0.5 * vs) / vol
    present = vs > 1e-10 * vol
    lo, hi = centre[present].min(), centre[present].max()
    frac = np.clip(np.asarray(percents) / 100.0, lo, hi)
    tbp = np.interp(frac, centre, Tb)
    return {"volume": vol, "mass": mass, "moles": float(F.sum()), "SG": sg,
            "API": 141.5 / sg - 131.5, "TBP": {str(p): float(t) for p, t in zip(percents, tbp)}}
