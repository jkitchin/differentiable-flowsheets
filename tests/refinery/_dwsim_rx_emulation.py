"""Ideal-gas reaction thermochemistry, written twice: difflow's conventions
and DWSIM 9.0.5's, for ``test_dwsim_reactions.py``.

difflow (isomerization, reformer): ``G_i(T) = H_f + int Cp dT - T (S +
int Cp/T dT)`` with exact integrals of the Cp cubic, ``R = 8.314462618``,
equilibrium constants on a 1 bar standard state.

DWSIM (``PropertyPackage.AUX_DELGF_T`` and its reactors, read from the IL):
the same Gibbs-Helmholtz route but the integrals by the midpoint rule
(``AUX_INT_CPDTi`` / ``AUX_INT_CPDT_Ti``: ``round(|T - T0|/10)`` intervals
clipped to [10, 100]), ``R = 8.314`` and pressures over ``P0 = 101325 Pa``.

:func:`equilibrium` is an element-potential Gibbs minimiser for an ideal gas;
given the same ``G_i/RT`` as either program it reproduces that program's
equilibrium, which is how the tests take a DWSIM-vs-difflow difference apart
into its causes (data, quadrature, R, standard state).
"""

from __future__ import annotations

import math

import numpy as np

T0 = 298.15
R_DIFFLOW = 8.314462618
R_DWSIM = 8.314
P0_DIFFLOW = 1.0e5
P0_DWSIM = 101325.0

DIFFLOW = {"quad": "exact", "R": R_DIFFLOW, "P0": P0_DIFFLOW}
DWSIM = {"quad": "midpoint", "R": R_DWSIM, "P0": P0_DWSIM}


def _n_mid(T: float) -> int:
    span = abs(T - T0)
    if span < 1:
        return 2
    if span < 3:
        return 4
    if span < 5:
        return 6
    return int(min(max(round(span / 10), 10), 100))


def _cp(cp, t):
    a, b, c, d = cp
    return a + b * t + c * t * t + d * t ** 3


def int_cp(cp, T: float, quad: str = "exact") -> float:
    """``int_T0^T Cp dT`` (J/mol) of a Cp cubic."""
    if cp is None:
        return 0.0
    a, b, c, d = cp
    if quad == "exact":
        F = lambda t: a * t + b * t ** 2 / 2 + c * t ** 3 / 3 + d * t ** 4 / 4  # noqa: E731
        return F(T) - F(T0)
    n = _n_mid(T)
    h = (T - T0) / n
    t = T0 + h / 2 + h * np.arange(n)
    return float(np.sum(_cp(cp, t)) * h)


def int_cp_over_T(cp, T: float, quad: str = "exact") -> float:
    """``int_T0^T Cp/T dT`` (J/mol/K)."""
    if cp is None:
        return 0.0
    a, b, c, d = cp
    if quad == "exact":
        F = lambda t: a * math.log(t) + b * t + c * t ** 2 / 2 + d * t ** 3 / 3  # noqa: E731
        return F(T) - F(T0)
    n = _n_mid(T)
    h = (T - T0) / n
    t = T0 + h / 2 + h * np.arange(n)
    return float(np.sum(_cp(cp, t) / t) * h)


def enthalpy(c: dict, T: float, quad: str = "exact") -> float:
    """``H_f + int Cp dT`` (J/mol) of a species ``{"Hf", "cp"}``."""
    return c["Hf"] + int_cp(c.get("cp"), T, quad)


def gibbs(c: dict, T: float, quad: str = "exact") -> float:
    """``H - T S`` at the standard pressure (J/mol), absolute-entropy basis
    (element terms cancel in a balanced reaction and do not move an
    element-constrained minimum)."""
    return enthalpy(c, T, quad) - T * (c["S"] + int_cp_over_T(c.get("cp"), T, quad))


def independent_rows(A: np.ndarray) -> np.ndarray:
    rows = []
    for r in A:
        if np.linalg.matrix_rank(np.array(rows + [r], float)) > len(rows):
            rows.append(r)
    return np.array(rows, float)


def equilibrium(g_RT, A, n0, P: float, P0: float, n_inert: float = 0.0, tol: float = 1e-12):
    """Ideal-gas Gibbs minimum of the species with ``G_i/RT = g_RT[i]``
    (standard state ``P0``) under element balances ``A n = A n0``; ``n_inert``
    moles ride along. Element potentials and ``ln N`` by Newton."""
    g = np.asarray(g_RT, float)
    A = independent_rows(np.asarray(A, float))
    n0 = np.asarray(n0, float)
    b = A @ n0
    lnP = math.log(P / P0)
    N = n0.sum() + n_inert
    x0 = np.maximum(n0 / N, 1e-3)
    lam, *_ = np.linalg.lstsq(A.T, g + lnP + np.log(x0), rcond=None)
    y = np.concatenate([lam, [math.log(N)]])
    m = len(lam)
    for _ in range(500):
        lam, lnN = y[:m], y[m]
        n = np.exp(np.clip(A.T @ lam - g - lnP + lnN, -700, 700))
        N = math.exp(lnN)
        sb = np.maximum(np.abs(b), 1e-300)
        F = np.concatenate([(A @ n - b) / sb, [(n.sum() + n_inert) / N - 1.0]])
        if np.max(np.abs(F)) < tol:
            break
        J = np.zeros((m + 1, m + 1))
        J[:m, :m] = (A * n) @ A.T / sb[:, None]
        J[:m, m] = (A @ n) / sb
        J[m, :m] = (A @ n) / N
        J[m, m] = -n_inert / N
        d = np.linalg.solve(J, -F)
        s = min(1.0, 2.0 / max(np.max(np.abs(d)), 1e-300))
        y = y + s * d
        if np.max(np.abs(s * d)) < 1e-15:
            break
    else:  # pragma: no cover
        raise RuntimeError(f"equilibrium did not converge: {F}")
    return n


def species_equilibrium(consts: dict, names, A, n0, T, P, conv: dict, n_inert=0.0, gibbs_fn=None):
    """:func:`equilibrium` of ``names`` (keys of ``consts``) at ``T``, ``P``
    under convention ``conv`` (:data:`DIFFLOW` or :data:`DWSIM`)."""
    gfun = gibbs_fn or (lambda n: gibbs(consts[n], T, conv["quad"]))
    g = np.array([gfun(n) for n in names]) / (conv["R"] * T)
    return equilibrium(g, A, n0, P, conv["P0"], n_inert=n_inert)


def adiabatic(consts: dict, names, A, n0, inert: dict, T_in, P, conv: dict, bracket=(300.0, 900.0)):
    """Adiabatic ideal-gas equilibrium: reacting ``names`` (initial ``n0``)
    plus ``inert`` ``{name: moles}``. Returns ``(T, n)``."""
    from scipy.optimize import brentq

    q = conv["quad"]
    H_in = (sum(n * enthalpy(consts[k], T_in, q) for k, n in zip(names, n0))
            + sum(v * enthalpy(consts[k], T_in, q) for k, v in inert.items()))
    ni = sum(inert.values())

    def resid(T):
        n = species_equilibrium(consts, names, A, n0, T, P, conv, n_inert=ni)
        H = (sum(x * enthalpy(consts[k], T, q) for k, x in zip(names, n))
             + sum(v * enthalpy(consts[k], T, q) for k, v in inert.items()))
        return H - H_in

    T = brentq(resid, *bracket, xtol=1e-12, rtol=1e-14)
    return T, species_equilibrium(consts, names, A, n0, T, P, conv, n_inert=ni)
