"""Generic physical-solvent absorber and stripper (Kremser, stage-by-stage, packed).

A gas of carrier plus dilute solutes rises through a column against a liquid
solvent. ``gas_in`` enters at the bottom and ``liquid_in`` at the top; the
solutes (the keys of ``params.henry``) transfer between the phases, every other
species is carried through on the side it enters (the carrier gas stays in the
gas, the solvent stays in the liquid; give them different names). The same
column is an **absorber** (solute from gas to liquid, ``A = L/(K V) > 1``) or a
**stripper** (solute from liquid to gas, stripping factor ``S = K V/L > 1``); the
two classes share one implementation and differ in defaults and in what
``recovery`` means.

Equilibrium
-----------
Henry's law in mole fractions, ``y_i = K_i x_i`` with ``K_i = H_i(T)/P`` and a
van 't Hoff temperature dependence ``H(T) = H_ref exp(-C (1/T - 1/T_ref))``
(:func:`henry_constant`; ``C = -dH_abs/R``, positive for an exothermic
absorption). By default the column is isothermal at ``params.T`` (default: the
gas inlet temperature) and the heat of absorption is reported, not used.

Energy balance
--------------
``energy_balance=True`` (``model="stages"``) adds an adiabatic stage energy
balance: the heat of absorption ``R C_i`` per mol transferred heats the stage,
both phases leave a stage at its temperature (constant ``cp_liquid``,
``cp_gas``), and each stage's ``K_i`` is evaluated at that temperature, so the
liquid warms on its way down and the solute gets less soluble.
``intercooler_stages`` cool the liquid leaving those stages back to
``intercooler_T``; ``info`` carries the temperature profile and the duties.
For a :class:`Stripper` with live steam, ``steam_species`` names the steam in
the stripping gas: the steam that condenses to heat the feed to the column
temperature and to supply the heat of desorption joins the liquid, and the
condensed rate and duty are reported.

Models
------
``"kremser"``
    Closed form per solute. With ``A_i`` the effective absorption factor (the
    geometric mean of the top and bottom values, ``A = sqrt(L_in L_out /
    (V_in V_out))/K``, iterated to consistency with the outlet flows; or the
    inlet value if ``variable_flow=False``. The geometric mean is the usual
    textbook simplification, e.g. Geankoplis 5e Sec. 10.6, Wankat 4e Ch. 12;
    it is *not* Edmister's effective factor ``A_e = sqrt(A_N (A_1 + 1) +
    0.25) - 0.5``, a separate method) and ``N_eq`` the
    equivalent equilibrium stages (``N_eq = N ln(1+E(S-1))/ln S``, ``S = 1/A``, for a Murphree
    vapor efficiency ``E``)::

        v_out = v_in (1 - f) + (l_in / A) f ,   f = (A^(N+1) - A) / (A^(N+1) - 1)

    which includes solute in the entering solvent (``l_in/A = K x_in V`` is the
    gas in equilibrium with it), holds for every ``A > 0`` (for ``A < 1`` the
    removal tends to ``A``, not 1) and goes smoothly through ``A = 1``. The stage
    count ``n_stages`` is a real number. Giving ``recovery`` makes ``n_stages`` an
    output (design).
``"stages"``
    Tray-to-tray: all component balances plus the Murphree stage relation
    solved as one linear system per solute, iterated on the total flows of every
    stage (fixed iteration count, so it is differentiable and jit-able). It
    carries the stage-to-stage flow change that Kremser's mean ``A`` only
    approximates, so it is the oracle the Kremser mode is checked against.
    ``n_stages`` must be a concrete integer.
``"packed"``
    Rate-based HTU/NTU. ``N_OG = Z/H_OG`` with ``H_OG = H_G + lambda H_L``
    (``lambda = 1/A``) from the Onda film coefficients of
    :mod:`difflow.mass_transfer` (needs the column ``diameter``, a ``packing``
    and the fluid properties in the params), or a user ``H_OG``. For straight
    operating and equilibrium lines (dilute) the outlet follows Colburn's
    closed form, the same ``v_out = v_in (1-f) + (l_in/A) f`` with
    ``1 - f = 1/(1 + (e^{eps N_OG} - 1)/eps)``, ``eps = 1 - lambda``. Rating mode
    takes ``height``; design mode takes ``recovery`` and returns the height
    (Colburn, or the numerical integral ``ntu_method="integral"``, which with
    ``concentrated=True`` includes the ``(1-y)_lm`` correction for a single
    key solute).

Limits
------
Dilute (or moderately concentrated, via the flow-corrected ``A``) solutes,
constant ``K`` per solute (isothermal unless ``energy_balance``, no
solute-dependent activity), no solvent volatility (the carrier leaves dry, the
solvent does not evaporate, so the evaporative cooling of a real water
scrubber is missing), a temperature profile only in the stages model, no reaction (use ``difflow_cc`` for amines). Flooding and
pressure drop (``fraction_flood`` sizing) are not part of this unit; give the
diameter. The Henry-constant table :data:`HENRY_WATER` is from memory (except CH4).

References
----------
- Kremser, Natl. Petroleum News 22 (21), 43 (1930); Edmister, AIChE J. 3 (1957) 165.
- Colburn, Trans. AIChE 35 (1939) 211.
- Onda, Takeuchi & Okumoto, J. Chem. Eng. Japan 1 (1968) 56.
- Seader, Henley & Roper, Separation Process Principles, Ch. 6 and 7;
  Treybal, Mass-Transfer Operations, Ch. 8 and 10 (equations only; no
  textbook numerical example is reproduced, see ``tests/test_absorption.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.mass_transfer import (
    PACKINGS,
    PackingData,
    ntu_og_colburn,
    ntu_og_integral,
    onda_gas_coefficient,
    onda_liquid_coefficient,
    onda_wetted_area,
)
from difflow.params_mixin import ParamsMixin
from difflow.staged import (
    EquilibriumCurve,
    _extremum_on_interval,
    _taylor_log_fraction,
    from_ratio,
    kremser_fraction,
    kremser_stages,
    linear_equilibrium,
    to_ratio,
)
from difflow.streams import Stream, get_flows, make_stream

__all__ = [
    "Absorber",
    "AbsorptionParams",
    "Stripper",
    "HENRY_WATER",
    "HenryData",
    "absorption_factor",
    "henry_constant",
    "henry_params",
    "kremser_absorption",
    "kremser_stages",
    "minimum_solvent_rate",
    "minimum_stripping_gas_rate",
    "solute_free_operating_line",
    "stripping_factor",
]

R_GAS = 8.314462618  # J/mol/K
_FLOOR = 1e-12
_N_PICARD = 14
_N_PICARD_STAGES = 40
_N_SWEEPS_ENERGY = 80
_N_STEAM_PASSES = 50


# ---------------------------------------------------------------------------
# Henry's law
# ---------------------------------------------------------------------------


def henry_constant(H_ref: Array | float, T: Array | float, C: Array | float = 0.0,
                   T_ref: Array | float = 298.15) -> Array:
    """Henry's constant (Pa, mole-fraction basis) at ``T``: van 't Hoff form.

    ``H(T) = H_ref exp(-C (1/T - 1/T_ref))`` with ``y P = H x`` and
    ``C = -dH_abs/R`` (K), positive for an exothermic absorption (``H`` rises
    with ``T``, so the solute is less soluble when hot).
    """
    return jnp.asarray(H_ref, float) * jnp.exp(
        -jnp.asarray(C, float) * (1.0 / jnp.asarray(T, float) - 1.0 / jnp.asarray(T_ref, float)))


@dataclass(frozen=True)
class HenryData:
    """A Henry's-constant record: ``H298`` (Pa, ``y P = H x``), ``C`` (K), source note."""

    H298: float
    C: float
    source: str = "from memory, approximate (+/-20-30 %), unverified"


#: Solutes in water near 25 C, converted to the mole-fraction basis
#: (``H [Pa] = 55343 [mol/m3] / H^cp [mol/(m3 Pa)]``, ``C = d ln H^cp / d(1/T)``).
#: **Every number except CH4 is from memory** of the Sander (2015, Atmos. Chem.
#: Phys. 15, 4399) compilation and common textbook values; they are
#: order-of-magnitude values with an uncertainty of 20-30 % (more for the
#: temperature coefficients), meant for examples and tests. CH4 is the cited
#: Sander (2015) value (``H^cp = 1.4e-5 mol/(m3 Pa)``, 1600 K), whose own spread
#: between sources is several percent. Use measured data for design.
HENRY_WATER: dict[str, HenryData] = {
    "acetone": HenryData(2.0e5, 5000.0),
    "NH3": HenryData(9.4e4, 4100.0),
    "CO2": HenryData(1.66e8, 2400.0),
    "H2S": HenryData(5.5e7, 2100.0),
    "SO2": HenryData(4.6e6, 3100.0),
    "TCE": HenryData(5.5e7, 4400.0),
    "benzene": HenryData(3.1e7, 4000.0),
    # 55343 / 1.4e-5 = 3.95e9 Pa
    "CH4": HenryData(3.95e9, 1600.0,
                     "Sander (2015), Atmos. Chem. Phys. 15, 4399: H^cp = 1.4e-5 mol/(m3 Pa) at "
                     "298.15 K, d ln H^cp/d(1/T) = 1600 K (+/- a few %, more for C)"),
}


def henry_params(species: list[str], table: dict[str, HenryData] | None = None):
    """``(henry, henry_C)`` dicts for :class:`AbsorptionParams` from a table."""
    table = HENRY_WATER if table is None else table
    return ({s: table[s].H298 for s in species}, {s: table[s].C for s in species})


# ---------------------------------------------------------------------------
# Library helpers (not palette units)
# ---------------------------------------------------------------------------


def absorption_factor(L: Array | float, V: Array | float, K: Array | float) -> Array:
    """``A = L/(K V)``."""
    return jnp.asarray(L, float) / (jnp.asarray(K, float) * jnp.asarray(V, float))


def stripping_factor(L: Array | float, V: Array | float, K: Array | float) -> Array:
    """``S = K V/L = 1/A``."""
    return jnp.asarray(K, float) * jnp.asarray(V, float) / jnp.asarray(L, float)


def kremser_absorption(A: Array | float, N: Array | float) -> Array:
    """Fraction of solute absorbed from the gas by pure solvent in ``N`` stages.

    ``(A^(N+1) - A)/(A^(N+1) - 1)`` for any ``A > 0``; the same function as
    :func:`difflow.staged.kremser_fraction`, re-exported here by name.
    """
    return kremser_fraction(A, N)


def solute_free_operating_line(G_s: Array | float, L_s: Array | float,
                               X_ref: Array | float, Y_ref: Array | float):
    """Absorber operating line in solute-free ratios, ``Y = slope X + intercept``.

    The solute balance ``G_s (Y_in - Y) = L_s (X_out - X)`` makes a straight line
    of slope ``L_s/G_s`` in ``X = x/(1-x)``, ``Y = y/(1-y)`` coordinates through
    any one known point ``(X_ref, Y_ref)`` of the column (the top
    ``(X_in, Y_out)`` of an absorber; the bottom ``(X_out, Y_in)`` of a stripper
    is the same line). Returns ``(slope, intercept)``.

    Args:
        G_s: Solute-free gas rate (mol/s).
        L_s: Solute-free liquid rate (mol/s).
        X_ref, Y_ref: A point on the line (liquid and gas solute ratios).
    """
    slope = jnp.asarray(L_s, float) / jnp.asarray(G_s, float)
    return slope, jnp.asarray(Y_ref, float) - slope * jnp.asarray(X_ref, float)


def _ratio_curve(curve: EquilibriumCurve):
    """``Y*(X)`` of a mole-fraction curve ``y*(x)``."""
    curve = curve.ideal()
    return lambda X: to_ratio(curve(from_ratio(X)))


def minimum_solvent_rate(curve: EquilibriumCurve, Y_in: Array | float, Y_out: Array | float,
                         X_in: Array | float, G_s: Array | float) -> Array:
    """Minimum solute-free solvent rate ``L_s,min`` of an absorber (mol/s).

    The operating line ``Y = Y_out + (L_s/G_s)(X - X_in)`` must stay on or above
    the equilibrium curve ``Y*(X)`` (ratio coordinates; ``curve`` is the
    mole-fraction ``y*(x)``). Its smallest admissible slope is the largest chord
    slope ``(Y*(X) - Y_out)/(X - X_in)`` over ``X`` from ``X_in`` to the liquid in
    equilibrium with the inlet gas, ``X*(Y_in)``: a pinch at the rich end for a
    curve that bends up (Henry's law with ``K > 1``), a tangent pinch (interior
    maximum) otherwise. ``inf`` if ``Y_out`` is below equilibrium with ``X_in``
    (the spec cannot be met at any rate). The textbook design rule is
    ``L = 1.2-2 L_min``.

    Args:
        curve: Equilibrium curve ``y*(x)`` in mole fractions.
        Y_in, Y_out: Gas solute ratios at the bottom and top.
        X_in: Liquid solute ratio at the top (entering solvent).
        G_s: Solute-free gas rate (mol/s).
    """
    Ystar = _ratio_curve(curve)
    Y_in = jnp.asarray(Y_in, float)
    Y_out = jnp.asarray(Y_out, float)
    X_in = jnp.asarray(X_in, float)
    X_end = to_ratio(curve.ideal().inverse(from_ratio(Y_in)))
    lo = X_in + 1e-9 * jnp.maximum(X_end - X_in, 1e-12)

    def chord(X):
        return (Ystar(X) - Y_out) / (X - X_in)

    slope = _extremum_on_interval(chord, lo, X_end, True)
    reachable = (Y_out >= Ystar(X_in)) & (slope > 0.0)
    return jnp.where(reachable, slope, jnp.inf) * jnp.asarray(G_s, float)


def minimum_stripping_gas_rate(curve: EquilibriumCurve, X_in: Array | float, X_out: Array | float,
                               Y_in: Array | float, L_s: Array | float) -> Array:
    """Minimum solute-free stripping-gas rate ``G_s,min`` (mol/s), the mirror image.

    The stripper line ``Y = Y_in + (L_s/G_s)(X - X_out)`` through
    ``(X_out, Y_in)`` (clean end) must stay on or below ``Y*(X)``; its largest
    slope is the smallest chord slope ``(Y*(X) - Y_in)/(X - X_out)`` up to the
    rich-liquid end ``X_in`` (a rich-end or tangent pinch), and
    ``G_s,min = L_s/slope_max``. ``inf`` if ``X_out`` is below equilibrium with
    ``Y_in`` (a lean end that no amount of gas reaches).

    Args:
        curve: Equilibrium curve ``y*(x)`` in mole fractions.
        X_in, X_out: Liquid solute ratios, rich (top) and lean (bottom).
        Y_in: Stripping-gas solute ratio at the bottom.
        L_s: Solute-free liquid rate (mol/s).
    """
    Ystar = _ratio_curve(curve)
    X_in = jnp.asarray(X_in, float)
    X_out = jnp.asarray(X_out, float)
    Y_in = jnp.asarray(Y_in, float)
    lo = X_out + 1e-9 * jnp.maximum(X_in - X_out, 1e-12)

    def chord(X):
        return (Ystar(X) - Y_in) / (X - X_out)

    slope = _extremum_on_interval(chord, lo, X_in, False)
    reachable = (Y_in <= Ystar(X_out)) & (slope > 0.0)
    return jnp.asarray(L_s, float) / jnp.where(reachable, slope, 0.0)


# ---------------------------------------------------------------------------
# Numerical kernels
# ---------------------------------------------------------------------------


def _stage_ratio(eta: Array | float, A: Array) -> Array:
    """Effective equilibrium stages per stage for a Murphree *vapor* efficiency.

    The Lewis relation ``E_o = ln(1 + E_MV (S - 1))/ln S`` with ``S = 1/A = K V/L``
    (exact for straight lines and constant flows; ``difflow_cc`` applies the
    ``A`` form, which is the liquid-side Murphree efficiency). Smooth through
    ``S = 1`` (limit ``E_MV``); no floor on ``ln S``.
    """
    return _taylor_log_fraction(jnp.asarray(eta, float), 1.0 / A - 1.0)


def _expm1_over(z: Array) -> Array:
    """``expm1(z)/z``, smooth at 0."""
    near = jnp.abs(z) < 1e-6
    zs = jnp.where(near, 1.0, z)
    return jnp.where(near, 1.0 + z / 2.0 + z * z / 6.0, jnp.expm1(zs) / zs)


def _colburn_unabsorbed(N_OG: Array, lam: Array) -> Array:
    """Fraction of the driving force left, ``phi = eps/(e^{eps N} - lam)``, ``eps = 1 - lam``."""
    eps = 1.0 - lam
    return 1.0 / (1.0 + N_OG * _expm1_over(eps * N_OG))


def _colburn_ntu(phi: Array, lam: Array) -> Array:
    """Inverse of :func:`_colburn_unabsorbed` via :func:`ntu_og_colburn` (``inf`` if unreachable).

    ``N_OG`` depends only on ``phi = (y_out - y*)/(y_in - y*)`` and ``lam``:
    ``ntu_og_colburn(1, phi, 0, 0, lam)`` is ``ln[(1-lam)/phi + lam]/(1-lam)``.
    """
    ok = phi > 0.0
    n = ntu_og_colburn(1.0, jnp.where(ok, phi, 0.5), 0.0, 0.0, lam)
    reach = ok & ((1.0 - lam) / jnp.where(ok, phi, 0.5) + lam > 0.0)
    return jnp.where(reach, n, jnp.inf)


def _stages_solve(v_in, l_in, K, Gc, Lc, E, N, n_iter=_N_PICARD_STAGES):
    """Counter-current Murphree tray column, all solutes; returns outlets and profiles.

    Stage ``n = 1`` is the top. Unknowns per solute: the component flows leaving
    each stage in the liquid (``l_n``) and the gas (``v_n``); balances
    ``l_{n-1} + v_{n+1} = l_n + v_n`` and the Murphree vapor relation
    ``v_n = V_n [(1-E) v_{n+1}/V_{n+1} + E K l_n/L_n]`` are linear for given total
    flows ``V_n``, ``L_n``, which a fixed number of Picard sweeps updates.
    """
    V_in = Gc + jnp.sum(v_in)
    L_in = Lc + jnp.sum(l_in)
    eye = jnp.eye(N)
    Sup = jnp.eye(N, k=1)
    Sub = jnp.eye(N, k=-1)

    def one(vi, li, Ki, V, L):
        Vnext = jnp.concatenate([V[1:], V_in[None]])
        c = (1.0 - E) * V / Vnext
        e = E * Ki * V / L
        Ablk = jnp.concatenate([-jnp.diag(e), eye - c[:, None] * Sup], axis=1)
        Bblk = jnp.concatenate([eye - Sub, eye - Sup], axis=1)
        M = jnp.concatenate([Ablk, Bblk], axis=0)
        rA = jnp.zeros(N).at[N - 1].set(c[N - 1] * vi)
        rB = jnp.zeros(N).at[0].add(li).at[N - 1].add(vi)
        z = jnp.linalg.solve(M, jnp.concatenate([rA, rB]))
        return z[:N], z[N:]

    solve_all = jax.vmap(one, in_axes=(0, 0, 0, None, None))

    def sweep(carry, _):
        V, L = carry
        l, v = solve_all(v_in, l_in, K, V, L)
        return (Gc + jnp.sum(v, axis=0), Lc + jnp.sum(l, axis=0)), None

    (V, L), _ = lax.scan(sweep, (jnp.full(N, V_in), jnp.full(N, L_in)), None, length=n_iter)
    l, v = solve_all(v_in, l_in, K, V, L)
    V2 = Gc + jnp.sum(v, axis=0)
    L2 = Lc + jnp.sum(l, axis=0)
    resid = jnp.max(jnp.abs(V2 - V)) + jnp.max(jnp.abs(L2 - L))
    return v[:, 0], l[:, N - 1], {"l": l, "v": v, "V": V2, "L": L2, "residual": resid}


def _stages_solve_energy(v_in, l_in, H_ref, C, T_ref, P, Gc, Lc, E, N, T_G_in, T_L_in,
                         cp_L, cp_G, ic, T_ic, n_iter=_N_SWEEPS_ENERGY):
    """Tray-to-tray column with a stage energy balance (adiabatic plus intercoolers).

    The component part is that of :func:`_stages_solve` with
    ``K_{i,n} = H_i(T_n)/P`` at the temperature of stage ``n``. Both phases leave
    a stage at its temperature ``T_n``; with sensible enthalpies
    ``cp (T - T_ref)`` and the heat of absorption ``R C_i`` released per mol of
    solute moving into the liquid on the stage, the stage balance is

    ``cp_L L_{n-1} T^L_{n-1} + cp_G V_{n+1} T_{n+1} + q_n = (cp_L L_n + cp_G V_n) T_n``

    (``T^L_{n-1}`` is ``T_ic`` when an intercooler cools the liquid leaving
    stage ``n-1``). A fixed number of damped Picard sweeps (component solve,
    flows, tridiagonal energy solve) gives the start of a Newton solve of all
    the equations at once (optimistix, implicit differentiation), because the
    plain substitution oscillates when the heat effect is large (a
    temperature bulge at a small L/G).
    """
    import optimistix as optx

    ns = v_in.shape[0]
    V_in = Gc + jnp.sum(v_in)
    L_in = Lc + jnp.sum(l_in)
    eye = jnp.eye(N)
    Sup = jnp.eye(N, k=1)
    Sub = jnp.eye(N, k=-1)

    def K_of(T, a):
        H_ref_, C_, P_ = a["H_ref"], a["C"], a["P"]
        return (H_ref_[:, None] * jnp.exp(-C_[:, None] * (1.0 / T[None, :] - 1.0 / T_ref))) / P_

    def one(vi, li, Ki, V, L):
        Vnext = jnp.concatenate([V[1:], V_in[None]])
        c = (1.0 - E) * V / Vnext
        e = E * Ki * V / L
        Ablk = jnp.concatenate([-jnp.diag(e), eye - c[:, None] * Sup], axis=1)
        Bblk = jnp.concatenate([eye - Sub, eye - Sup], axis=1)
        M = jnp.concatenate([Ablk, Bblk], axis=0)
        rA = jnp.zeros(N).at[N - 1].set(c[N - 1] * vi)
        rB = jnp.zeros(N).at[0].add(li).at[N - 1].add(vi)
        z = jnp.linalg.solve(M, jnp.concatenate([rA, rB]))
        return z[:N], z[N:]

    solve_all = jax.vmap(one, in_axes=(0, 0, 0, None, None))

    def heat(v, a):
        v_below = jnp.concatenate([v[:, 1:], a["v_in"][:, None]], axis=1)
        return R_GAS * jnp.sum(a["C"][:, None] * (v_below - v), axis=0)     # W released per stage

    def energy_matrix(V, L, q, a):
        cpL, cpG = a["cp_L"], a["cp_G"]
        diag = cpL * L + cpG * V
        sub = -cpL * L[:-1] * (1.0 - ic[:-1])                               # theta_{n-1} in row n
        sup = -cpG * V[1:]                                                  # theta_{n+1} in row n
        M = jnp.diag(diag) + jnp.diag(sub, -1) + jnp.diag(sup, 1)
        rhs = q + jnp.concatenate([jnp.zeros(1), cpL * L[:-1] * ic[:-1] * (a["T_ic"] - T_ref)])
        rhs = rhs.at[0].add(cpL * a["L_in"] * (a["T_L_in"] - T_ref))
        rhs = rhs.at[N - 1].add(cpG * a["V_in"] * (a["T_G_in"] - T_ref))
        return M, rhs

    args = {"v_in": v_in, "l_in": l_in, "H_ref": H_ref, "C": C, "P": P, "Gc": Gc, "Lc": Lc,
            "V_in": V_in, "L_in": L_in, "T_G_in": T_G_in, "T_L_in": T_L_in, "cp_L": cp_L,
            "cp_G": cp_G, "T_ic": T_ic, "E": E}

    def sweep(carry, _):
        V, L, T = carry
        l, v = solve_all(v_in, l_in, K_of(T, args), V, L)
        Vn = Gc + jnp.sum(v, axis=0)
        Ln = Lc + jnp.sum(l, axis=0)
        M, rhs = energy_matrix(Vn, Ln, heat(v, args), args)
        Tn = T_ref + jnp.linalg.solve(M, rhs)
        return (Vn, Ln, 0.5 * (T + Tn)), None

    T0 = jnp.full(N, T_L_in)
    (V, L, T), _ = lax.scan(sweep, (jnp.full(N, V_in), jnp.full(N, L_in), T0), None, length=n_iter)
    l0, v0 = solve_all(v_in, l_in, K_of(T, args), V, L)
    # Newton on z = (l, v, T - T_ref), all stages and solutes
    z0 = lax.stop_gradient(jnp.concatenate([l0.ravel(), v0.ravel(), T - T_ref]))

    def unpack(z):
        return z[:ns * N].reshape(ns, N), z[ns * N:2 * ns * N].reshape(ns, N), T_ref + z[2 * ns * N:]

    def resid(z, a):
        l, v, T = unpack(z)
        Vs = a["Gc"] + jnp.sum(v, axis=0)
        Ls = a["Lc"] + jnp.sum(l, axis=0)
        Vnext = jnp.concatenate([Vs[1:], a["V_in"][None]])
        v_below = jnp.concatenate([v[:, 1:], a["v_in"][:, None]], axis=1)
        l_above = jnp.concatenate([a["l_in"][:, None], l[:, :-1]], axis=1)
        K = K_of(T, a)
        Em = a["E"]
        r_eq = v - Vs * ((1.0 - Em) * v_below / Vnext + Em * K * l / Ls)
        r_bal = l_above + v_below - l - v
        M, rhs = energy_matrix(Vs, Ls, heat(v, a), a)
        r_T = (M @ (T - T_ref) - rhs) / (a["cp_L"] * a["L_in"])
        scale = a["V_in"] + a["L_in"]
        return jnp.concatenate([r_eq.ravel() / scale, r_bal.ravel() / scale, r_T])

    sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-13), z0, args=args,
                         max_steps=60, throw=False)
    z = jnp.where(jnp.all(jnp.isfinite(sol.value)), sol.value, z0)
    l, v, T = unpack(z)
    V2 = Gc + jnp.sum(v, axis=0)
    L2 = Lc + jnp.sum(l, axis=0)
    r = resid(z, args)
    Q_ic = cp_L * L2 * (T - T_ic) * ic
    return v[:, 0], l[:, N - 1], {"l": l, "v": v, "V": V2, "L": L2, "T": T, "K": K_of(T, args),
                                  "q": heat(v, args), "Q_ic": Q_ic,
                                  "residual": jnp.max(jnp.abs(r)) * (V_in + L_in)}


def _picard(step, n_iter: int = _N_PICARD):
    """Fixed-point iteration on the net transfer ``delta`` (mol/s, gas to liquid)."""
    def body(delta, _):
        _, new = step(delta)
        return new, None

    delta, _ = lax.scan(body, jnp.asarray(0.0), None, length=n_iter)
    aux, new = step(delta)
    return aux, delta, jnp.abs(new - delta)


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class AbsorptionParams(ParamsMixin):
    """Parameters of a gas-liquid column, shared by :class:`Absorber` and :class:`Stripper`.

    (Named ``AbsorptionParams``, not ``AbsorberParams``, so it cannot shadow
    ``difflow_cc.AbsorberParams`` or ``difflow_ree.StripperParams`` in
    ``difflow.serialize``'s lookup of Params classes by name.) For a
    :class:`Stripper`, ``recovery`` is the fraction of the key solute removed from
    the liquid.

    Attributes:
        henry: Henry's constants ``H_i`` (Pa, ``y_i P = H_i x_i``) at ``T_ref`` for
            every transferring solute; the keys are the solute species names.
            ``difflow.units.absorption.HENRY_WATER`` has a small table (from memory).
        henry_C: van 't Hoff coefficients ``C_i`` (K) per solute,
            ``H(T) = H_ref exp(-C (1/T - 1/T_ref))``; missing means 0 (no
            temperature dependence).
        T_ref: Reference temperature of ``henry`` (K).
        model: ``"kremser"``, ``"stages"`` (tray-to-tray, integer ``n_stages``) or
            ``"packed"`` (HTU/NTU).
        n_stages: Number of real stages (kremser and stages models). Continuous
            for ``"kremser"``; ignored when ``recovery`` is given (it is then an output).
        stage_efficiency: Murphree vapor efficiency of a stage (1 = equilibrium).
        P: Column pressure (Pa); ``None`` takes the gas inlet pressure.
        T: Column temperature (K, isothermal); ``None`` takes the gas inlet temperature.
        variable_flow: Use the geometric-mean absorption factor from the
            top and bottom flows (True), or the inlet flows (False).
        key_solute: The solute that sets ``recovery`` (design), ``L_min`` and the
            operating line; defaults to the first of ``henry``.
        recovery: Design specification for the key solute: fraction removed from
            the gas (absorber) or from the liquid (stripper). ``None`` = rating.
        packing: Packing name in ``difflow.mass_transfer.PACKINGS`` or a
            ``PackingData`` (packed model with Onda films).
        diameter: Column diameter (m) (packed model).
        height: Packed height (m) for rating; ignored with ``recovery`` (output).
        H_OG: Overall transfer-unit height (m); ``None`` computes it from Onda films.
        ntu_method: ``"colburn"`` (analytic, straight lines) or ``"integral"``
            (numerical ``N_OG`` for the key solute; design mode only).
        concentrated: With ``"integral"``, include the ``(1-y)_lm`` correction.
        D_G: Gas-phase diffusivity per solute (m^2/s) for the Onda gas film.
        D_L: Liquid-phase diffusivity per solute (m^2/s) for the Onda liquid film.
        mu_G: Gas viscosity (Pa s); default air at 25 C.
        mu_L: Liquid viscosity (Pa s); default water at 25 C.
        rho_L: Liquid density (kg/m^3); default water.
        sigma: Liquid surface tension (N/m); default water.
        MW_gas: Mean molar mass of the gas (kg/mol); default air.
        MW_liquid: Molar mass of the solvent (kg/mol); default water.
        cp_liquid: Liquid molar heat capacity (J/mol/K) for the adiabatic
            temperature-rise estimate, the stage energy balance and the
            live-steam preheat; default water.
        energy_balance: Solve a stage energy balance (``model="stages"`` only):
            the heat of absorption ``R C_i`` per mol heats the stage, both phases
            leave a stage at its temperature, and every stage's ``K_i`` is taken at
            that temperature. The inlets enter at their stream temperatures
            (``T`` must be None). False (default) keeps the isothermal column.
        cp_gas: Gas molar heat capacity (J/mol/K) for the energy balance and the
            live-steam balance; default air (use about 34 for steam).
        intercooler_stages: Stages (1 = top, ``n_stages`` = bottom) whose liquid
            outflow is cooled to ``intercooler_T`` before the next stage down (the
            bottom one cools the liquid product). Needs ``energy_balance``.
        intercooler_T: Temperature (K) the intercoolers bring the liquid to;
            ``None`` takes the liquid inlet temperature. A negative duty means
            the "intercooler" heats.
        steam_species: Live-steam stripping (:class:`Stripper` only): the name
            of the steam species in ``gas_in``. Steam condenses into the liquid
            to heat the feed to the column temperature and supply the heat of
            desorption; ``None`` (default) leaves the stripping gas inert.
        steam_latent_heat: Latent heat of the condensing steam at the column
            temperature (J/mol); default water at 100 C.
    """

    henry: dict[str, float]
    henry_C: dict[str, float] = field(default_factory=dict)
    T_ref: float = 298.15
    model: Literal["kremser", "stages", "packed"] = "kremser"
    n_stages: float = 5.0
    stage_efficiency: float = 1.0
    P: float | None = None
    T: float | None = None
    variable_flow: bool = True
    key_solute: str | None = None
    recovery: float | None = None
    packing: str | PackingData | None = None
    diameter: float | None = None
    height: float | None = None
    H_OG: float | None = None
    ntu_method: Literal["colburn", "integral"] = "colburn"
    concentrated: bool = False
    D_G: dict[str, float] = field(default_factory=dict)
    D_L: dict[str, float] = field(default_factory=dict)
    mu_G: float = 1.85e-5
    mu_L: float = 8.9e-4
    rho_L: float = 997.0
    sigma: float = 0.072
    MW_gas: float = 0.02897
    MW_liquid: float = 0.018015
    cp_liquid: float = 75.3
    energy_balance: bool = False
    cp_gas: float = 29.1
    intercooler_stages: tuple[int, ...] = ()
    intercooler_T: float | None = None
    steam_species: str | None = None
    steam_latent_heat: float = 40660.0

    def __post_init__(self):
        if not self.henry:
            raise ValueError("henry needs at least one solute")
        if self.model not in ("kremser", "stages", "packed"):
            raise ValueError("model must be 'kremser', 'stages' or 'packed'")
        if self.ntu_method not in ("colburn", "integral"):
            raise ValueError("ntu_method must be 'colburn' or 'integral'")
        if self.key_solute is not None and self.key_solute not in self.henry:
            raise ValueError(f"key_solute {self.key_solute!r} not in henry")
        self.intercooler_stages = tuple(int(n) for n in self.intercooler_stages)
        if self.energy_balance:
            if self.model != "stages":
                raise ValueError("energy_balance needs model='stages' (a stage temperature profile)")
            if self.T is not None:
                raise ValueError("energy_balance takes the inlet temperatures from the streams; leave T=None")
            if self.steam_species is not None:
                raise ValueError("energy_balance and steam_species cannot be combined")
        if self.intercooler_stages and not self.energy_balance:
            raise ValueError("intercooler_stages needs energy_balance=True")
        if self.steam_species is not None and self.steam_species in self.henry:
            raise ValueError("steam_species must be an inert of the gas, not a solute in henry")


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


class _GasLiquidColumn:
    """Shared implementation of :class:`Absorber` and :class:`Stripper`."""

    _role = "absorber"

    def __init__(self, params: AbsorptionParams):
        self.params = params

    # -- helpers -----------------------------------------------------------
    def _packing(self) -> PackingData:
        pk = self.params.packing
        if isinstance(pk, str):
            if pk not in PACKINGS:
                raise ValueError(f"unknown packing {pk!r}; one of {sorted(PACKINGS)}")
            return PACKINGS[pk]
        return pk

    def _film_heights(self, solutes, V_in, L_in, T, P):
        """Onda ``H_G``, ``H_L`` per solute (m) and diagnostics, at the inlet flows."""
        p = self.params
        pk = self._packing()
        if pk is None or p.diameter is None:
            raise ValueError("the packed model needs `packing` and `diameter` (or a user H_OG)")
        missing = [s for s in solutes if s not in p.D_G or s not in p.D_L]
        if missing:
            raise ValueError(f"the Onda films need D_G and D_L for {missing} "
                             "(e.g. difflow.mass_transfer.fuller_diffusivity, wilke_chang_diffusivity)")
        area = jnp.pi * jnp.asarray(p.diameter, float) ** 2 / 4.0
        rho_G = P * p.MW_gas / (R_GAS * T)
        u_G = V_in * R_GAS * T / (P * area)
        u_L = L_in * p.MW_liquid / (p.rho_L * area)
        D_G = jnp.stack([jnp.asarray(p.D_G[s], float) for s in solutes])
        D_L = jnp.stack([jnp.asarray(p.D_L[s], float) for s in solutes])
        a_w = onda_wetted_area(u_L, p.rho_L, p.mu_L, p.sigma, pk)
        k_c = onda_gas_coefficient(u_G, rho_G, p.mu_G, D_G, pk)
        k_L = onda_liquid_coefficient(u_L, p.rho_L, p.mu_L, D_L, p.sigma, pk, a_w=a_w)
        k_y = k_c * P / (R_GAS * T)
        k_x = k_L * p.rho_L / p.MW_liquid
        H_G = (V_in / area) / (k_y * a_w)
        H_L = (L_in / area) / (k_x * a_w)
        return H_G, H_L, {"a_w": a_w, "area": area, "k_y": k_y, "k_x": k_x, "u_G": u_G, "u_L": u_L}

    # -- call --------------------------------------------------------------
    def __call__(self, gas_in: Stream, liquid_in: Stream) -> tuple[Stream, Stream, dict]:
        """Contact the gas (bottom) with the liquid (top).

        Args:
            gas_in: Gas entering at the bottom (carrier plus solutes), mol/s.
            liquid_in: Liquid entering at the top (solvent plus any solute), mol/s.

        Returns:
            gas_out: Gas leaving at the top.
            liquid_out: Liquid leaving at the bottom.
            info: ``recovery`` and ``transferred`` per solute (mol/s gas to liquid),
                ``A`` and ``S`` (effective absorption and stripping factors), ``K``,
                ``n_stages`` / ``N_eq`` (kremser, stages; ``N_eq`` per solute) or ``N_OG``,
                ``H_OG`` (per solute), ``N_OL = N_OG / A`` and ``H_OL = A H_OG`` (per
                solute, the liquid-phase basis a stripper is usually designed on) and
                ``height`` (packed), ``L_min`` (absorber) or ``G_min`` (stripper)
                for the key solute and the ratio to it, the solute-free operating
                line and equilibrium curve (``X``, ``Y``, ``Y_eq``), ``Q_absorption``
                (W) and ``delta_T_adiabatic`` (K, an estimate; applied only with
                ``energy_balance``), ``converged``, ``feasible``. With
                ``energy_balance``: ``stage_T`` (K, top to bottom), ``stage_K``
                (per solute), ``stage_Q_absorption`` (W per stage),
                ``intercooler_duty`` (W per stage, heat removed), ``Q_intercooler``,
                ``Q_sensible`` (W, sensible heat gained by both phases),
                ``energy_balance_error`` (W), ``T_gas_out``, ``T_liquid_out``; ``T``,
                ``K``, ``A``, ``S`` and the minimum rate are then quoted at the mean
                stage temperature. With ``steam_species`` (live steam):
                ``steam_condensed`` (mol/s), ``steam_duty`` (W, latent heat of the
                condensed steam), ``Q_preheat``, ``Q_desorption``, ``Q_gas_sensible``
                (W, released by the gas cooling to the column temperature;
                ``steam_duty = Q_preheat + Q_desorption - Q_gas_sensible``),
                ``steam_fraction_condensed``.
        """
        if self.params.steam_species is not None:
            return self._live_steam(gas_in, liquid_in)
        return self._column(gas_in, liquid_in)

    def _live_steam(self, gas_in: Stream, liquid_in: Stream) -> tuple[Stream, Stream, dict]:
        """Live-steam stripper: an overall energy balance sets the steam that condenses.

        The column runs at ``T_s`` (``params.T``, default the steam inlet
        temperature). The steam that condenses joins the liquid at the top (it
        heats the feed by direct contact), so the column sees ``L_in + n`` and
        ``V_in - n``. With sensible heats at constant ``cp`` and the latent heat
        ``lambda`` at ``T_s``:

        ``n lambda = cp_L L_in (T_s - T_L,in) + cp_G V_in (T_s - T_G,in) + Q_des``

        where ``Q_des = -Q_absorption`` is the heat of desorption. ``Q_des``
        depends on what is stripped, so ``n`` is the fixed point of balance and
        column, found by substitution (the coupling is weak) and differentiated
        implicitly, so a traced call holds one column, not one per pass.
        """
        import optimistix as optx

        p = self.params
        if self._role != "stripper":
            raise ValueError("steam_species (live steam) is for a Stripper")
        s = p.steam_species
        gf = get_flows(gas_in)
        lf = get_flows(liquid_in)
        if s not in gf:
            raise ValueError(f"steam_species {s!r} is not in gas_in")
        steam_in = jnp.asarray(gf[s], float)
        T_s = jnp.asarray(p.T if p.T is not None else gas_in["T"], float)
        T_G = jnp.asarray(gas_in["T"], float)
        T_L = jnp.asarray(liquid_in["T"], float)
        V_in = sum(jnp.asarray(v, float) for v in gf.values())
        L_in = sum(jnp.asarray(v, float) for v in lf.values())
        lam = jnp.asarray(p.steam_latent_heat, float)
        Q_pre = p.cp_liquid * L_in * (T_s - T_L)
        Q_gas = p.cp_gas * V_in * (T_s - T_G)            # <= 0 for superheated steam

        n_max = 0.999 * steam_in

        def column_at(n):
            g2 = {**gf, s: steam_in - n}
            l2 = {**lf, s: jnp.asarray(lf.get(s, 0.0), float) + n}
            return self._column(make_stream(g2, gas_in["T"], gas_in["P"]),
                                make_stream(l2, liquid_in["T"], liquid_in["P"]))

        def condensed(n, _):
            Q_des = -column_at(n)[2]["Q_absorption"]
            return jnp.clip((Q_pre + Q_gas + Q_des) / lam, 0.0, n_max)

        # One column in a while loop, differentiated implicitly: unrolled passes
        # put a whole column per pass into a traced graph (#496).
        n0 = jnp.clip((Q_pre + Q_gas) / lam, 0.0, n_max)
        sol = optx.fixed_point(condensed, optx.FixedPointIteration(rtol=1e-10, atol=1e-14),
                               n0, max_steps=_N_STEAM_PASSES, throw=False)
        n = sol.value
        gas_out, liquid_out, info = column_at(n)
        Q_des = -info["Q_absorption"]
        n_raw = (Q_pre + Q_gas + Q_des) / lam
        info = dict(info)
        info.update(
            steam_condensed=n,
            steam_duty=n * lam,
            Q_preheat=Q_pre,
            Q_desorption=Q_des,
            Q_gas_sensible=-Q_gas,
            steam_fraction_condensed=n / steam_in,
            converged=info["converged"] & (jnp.abs(jnp.clip(n_raw, 0.0, 0.999 * steam_in) - n)
                                           <= 1e-6 * n + 1e-12),
            feasible=info["feasible"] & (n_raw < 0.999 * steam_in),
        )
        return gas_out, liquid_out, info

    def _column(self, gas_in: Stream, liquid_in: Stream) -> tuple[Stream, Stream, dict]:
        p = self.params
        solutes = list(p.henry)
        key = p.key_solute or solutes[0]
        k = solutes.index(key)
        design = p.recovery is not None
        gf = get_flows(gas_in)
        lf = get_flows(liquid_in)
        T = jnp.asarray(p.T if p.T is not None else gas_in["T"], float)
        T_G_in = T
        P = jnp.asarray(p.P if p.P is not None else gas_in["P"], float)
        H = jnp.stack([henry_constant(p.henry[s], T, p.henry_C.get(s, 0.0), p.T_ref) for s in solutes])
        K = H / P
        v_in = jnp.stack([jnp.asarray(gf.get(s, 0.0), float) for s in solutes])
        l_in = jnp.stack([jnp.asarray(lf.get(s, 0.0), float) for s in solutes])
        V_in = sum(jnp.asarray(v, float) for v in gf.values())
        L_in = sum(jnp.asarray(v, float) for v in lf.values())
        eta = jnp.asarray(p.stage_efficiency, float)
        Gc = V_in - jnp.sum(v_in)
        Lc = L_in - jnp.sum(l_in)
        feasible = jnp.asarray(True)
        extra: dict = {}
        T_gas_out = T_liq_out = T

        # design target for the key solute: v_out_key from the recovery
        r = None if not design else jnp.asarray(p.recovery, float)
        if design:
            if p.model == "stages":
                raise ValueError("recovery (design) is not available for model='stages': "
                                 "n_stages is a concrete integer there; use rating or the kremser model")
            v_out_target = v_in[k] * (1.0 - r) if self._role == "absorber" else v_in[k] + l_in[k] * r

        if p.model == "stages":
            if _is_tracer(p.n_stages):
                raise ValueError("model='stages' needs a concrete integer n_stages (not a traced value)")
            n_f = float(p.n_stages)
            if n_f != int(n_f) or n_f < 1:
                raise ValueError(f"model='stages' needs an integer n_stages >= 1, got {p.n_stages}")
            N = int(n_f)
            if p.energy_balance:
                bad = [n for n in p.intercooler_stages if not 1 <= n <= N]
                if bad:
                    raise ValueError(f"intercooler_stages {bad} outside 1..{N}")
                ic = jnp.asarray([1.0 if n + 1 in p.intercooler_stages else 0.0 for n in range(N)])
                T_L_in = jnp.asarray(liquid_in["T"], float)
                T_ic = jnp.asarray(p.intercooler_T if p.intercooler_T is not None else T_L_in, float)
                H_ref = jnp.stack([jnp.asarray(p.henry[s], float) for s in solutes])
                C_vh = jnp.stack([jnp.asarray(p.henry_C.get(s, 0.0), float) for s in solutes])
                v_out, l_out, prof = _stages_solve_energy(
                    v_in, l_in, H_ref, C_vh, jnp.asarray(p.T_ref, float), P, Gc, Lc, eta, N,
                    T, T_L_in, jnp.asarray(p.cp_liquid, float), jnp.asarray(p.cp_gas, float), ic, T_ic)
                T_gas_out = prof["T"][0]
                T_liq_out = ic[N - 1] * T_ic + (1.0 - ic[N - 1]) * prof["T"][N - 1]
                # the factors and the minimum rate below are quoted at the mean stage temperature
                T = jnp.mean(prof["T"])
                K = jnp.stack([henry_constant(p.henry[s], T, p.henry_C.get(s, 0.0), p.T_ref)
                               for s in solutes]) / P
                Q_rel = jnp.sum(prof["q"])
                Q_ic = jnp.sum(prof["Q_ic"])
                L_bot_, V_top_ = Lc + jnp.sum(l_out), Gc + jnp.sum(v_out)
                sens = (p.cp_liquid * (L_bot_ * (T_liq_out - p.T_ref) - L_in * (T_L_in - p.T_ref))
                        + p.cp_gas * (V_top_ * (T_gas_out - p.T_ref) - V_in * (T_G_in - p.T_ref)))
                extra.update(stage_T=prof["T"], stage_K={s: prof["K"][i] for i, s in enumerate(solutes)},
                             stage_Q_absorption=prof["q"], intercooler_duty=prof["Q_ic"],
                             Q_intercooler=Q_ic, Q_sensible=sens,
                             energy_balance_error=Q_rel - sens - Q_ic,
                             T_gas_out=T_gas_out, T_liquid_out=T_liq_out)
            else:
                v_out, l_out, prof = _stages_solve(v_in, l_in, K, Gc, Lc, eta, N)
            V_top = Gc + jnp.sum(v_out)
            L_bot = Lc + jnp.sum(l_out)
            A = jnp.sqrt(L_in * L_bot / (V_in * V_top)) / K if p.variable_flow else L_in / (V_in * K)
            n_act = jnp.asarray(float(N))
            N_eq = n_act * _stage_ratio(eta, A)
            converged = prof["residual"] < 1e-9 * (V_in + L_in)
            extra.update(stage_x=prof["l"] / prof["L"], stage_y=prof["v"] / prof["V"],
                         stage_V=prof["V"], stage_L=prof["L"])
        else:
            packed = p.model == "packed"
            if packed:
                if design and (p.height is not None):
                    raise ValueError("give height (rating) or recovery (design), not both")
                if not design and p.height is None:
                    raise ValueError("packed model: give height (rating) or recovery (design)")
                if p.H_OG is None:
                    H_G, H_L, film = self._film_heights(solutes, V_in, L_in, T, P)
                    extra.update(film)
                else:
                    H_G = H_L = None
                if design and p.ntu_method == "integral":
                    G_s = V_in - v_in[k]
                    L_s = L_in - l_in[k]
                    Y_in = v_in[k] / G_s
                    Y_out = v_out_target / G_s
                    X_in = l_in[k] / L_s
                    y_in, y_out = from_ratio(Y_in), from_ratio(Y_out)

                    def y_star(y):
                        Y = to_ratio(y)
                        X = X_in + (G_s / L_s) * (Y - Y_out)
                        return K[k] * from_ratio(X)

                    N_og_int = ntu_og_integral(y_in, y_out, y_star, concentrated=p.concentrated)
                    extra["N_OG_integral"] = N_og_int
            else:
                N_in = jnp.asarray(p.n_stages, float)

            def step(delta):
                V_top = jnp.maximum(V_in - delta, _FLOOR * V_in)
                L_bot = jnp.maximum(L_in + delta, _FLOOR * L_in)
                if p.variable_flow:
                    A = jnp.sqrt(L_in * L_bot / (V_in * V_top)) / K
                else:
                    A = L_in / (V_in * K)
                v_star = l_in / A
                if packed:
                    lam = 1.0 / A
                    H_OG = (jnp.full_like(A, p.H_OG) if H_G is None else H_G + lam * H_L)
                    if design:
                        phi_k = (v_out_target - v_star[k]) / (v_in[k] - v_star[k])
                        if p.ntu_method == "integral":
                            Z = H_OG[k] * N_og_int
                        else:
                            Z = H_OG[k] * _colburn_ntu(phi_k, lam[k])
                    else:
                        Z = jnp.asarray(p.height, float)
                    N_OG = Z / H_OG
                    f = 1.0 - _colburn_unabsorbed(N_OG, lam)
                    aux = {"A": A, "H_OG": H_OG, "N_OG": N_OG, "height": Z, "f": f, "v_star": v_star,
                           "N_eq": None, "n_act": None}
                else:
                    if design:
                        N_key = kremser_stages(A[k], 1.0 - (v_out_target - v_star[k]) / (v_in[k] - v_star[k]))
                        n_act = N_key / _stage_ratio(eta, A[k])
                    else:
                        n_act = N_in
                    N_eq = n_act * _stage_ratio(eta, A)
                    f = kremser_fraction(A, N_eq)
                    aux = {"A": A, "f": f, "v_star": v_star, "N_eq": N_eq, "n_act": n_act}
                v_out = v_in * (1.0 - f) + v_star * f
                aux["v_out"] = v_out
                return aux, jnp.sum(v_in - v_out)

            aux, delta, resid = _picard(step)
            A, v_out = aux["A"], aux["v_out"]
            l_out = l_in + v_in - v_out
            converged = resid < 1e-9 * (V_in + L_in)
            if packed:
                extra.update(N_OG=dict(zip(solutes, aux["N_OG"])), H_OG=dict(zip(solutes, aux["H_OG"])),
                             N_OL=dict(zip(solutes, aux["N_OG"] / A)),
                             H_OL=dict(zip(solutes, aux["H_OG"] * A)),
                             height=aux["height"])
                n_act = jnp.asarray(jnp.nan)
                N_eq = aux["N_OG"]
                feasible = jnp.isfinite(aux["height"]) & (aux["height"] >= 0.0)
            else:
                n_act, N_eq = aux["n_act"], aux["N_eq"]
                feasible = jnp.isfinite(n_act) & (n_act >= 0.0)

        transferred = v_in - v_out
        removed_gas = transferred / jnp.where(v_in > 0.0, v_in, 1.0)
        removed_liq = -transferred / jnp.where(l_in > 0.0, l_in, 1.0)
        recov = removed_gas if self._role == "absorber" else removed_liq

        # outlets
        gas_flows = {s: jnp.asarray(f, float) for s, f in gf.items() if s not in p.henry}
        liq_flows = {s: jnp.asarray(f, float) for s, f in lf.items() if s not in p.henry}
        for i, s in enumerate(solutes):
            gas_flows[s] = v_out[i]
            liq_flows[s] = l_out[i]
        gas_out = make_stream(gas_flows, T_gas_out, P)
        liquid_out = make_stream(liq_flows, T_liq_out, P)

        # key-solute operating line, equilibrium curve and minimum rate (solute-free ratios)
        G_s = V_in - v_in[k]
        L_s = L_in - l_in[k]
        Y_in = v_in[k] / G_s
        Y_out = v_out[k] / G_s
        X_in = l_in[k] / L_s
        X_out = l_out[k] / L_s
        curve = linear_equilibrium(K[k])
        Xs = jnp.linspace(0.0, 1.0, 41) * (jnp.maximum(X_in, X_out) - jnp.minimum(X_in, X_out)) + \
            jnp.minimum(X_in, X_out)
        slope, icpt = solute_free_operating_line(G_s, L_s, X_in, Y_out)
        # The minimum rate is for the specification. In design mode an infeasible
        # solvent rate leaves the outlets at the A-limited removal, not at the
        # target, so take the target end from `recovery` rather than the outlets.
        Y_out_min, X_out_min = Y_out, X_out
        if p.recovery is not None:
            if self._role == "absorber":
                Y_out_min = v_in[k] * (1.0 - p.recovery) / G_s
            else:
                X_out_min = l_in[k] * (1.0 - p.recovery) / L_s
        if self._role == "absorber":
            L_min = minimum_solvent_rate(curve, Y_in, Y_out_min, X_in, G_s)
            ratio = L_s / L_min
            minimum = {"L_min": L_min, "L_over_Lmin": ratio}
        else:
            G_min = minimum_stripping_gas_rate(curve, X_in, X_out_min, Y_in, L_s)
            minimum = {"G_min": G_min, "G_over_Gmin": G_s / G_min}
        # heat of absorption: C_i R per mol, released when solute goes into the liquid
        C = jnp.stack([jnp.asarray(p.henry_C.get(s, 0.0), float) for s in solutes])
        Q = jnp.sum(transferred * C) * R_GAS
        dT = Q / (jnp.sum(jnp.stack([jnp.asarray(v, float) for v in liq_flows.values()])) * p.cp_liquid)

        info = {
            "recovery": dict(zip(solutes, recov)),
            "transferred": dict(zip(solutes, transferred)),
            "A": dict(zip(solutes, A)),
            "S": dict(zip(solutes, 1.0 / A)),
            "K": dict(zip(solutes, K)),
            "key_solute": key,
            "n_stages": n_act,
            "N_eq": dict(zip(solutes, jnp.broadcast_to(N_eq, (len(solutes),)))),
            **minimum,
            "X": Xs,
            "Y": icpt + slope * Xs,
            "Y_eq": to_ratio(curve(from_ratio(Xs))),
            "Q_absorption": Q,
            "delta_T_adiabatic": dT,
            "T": T,
            "P": P,
            "converged": converged,
            "feasible": feasible,
            **extra,
        }
        return gas_out, liquid_out, info


_COMMON_ASSUMPTIONS = [
    "Isothermal at T by default (the heat of absorption is reported, not applied); with energy_balance "
    "(stages model) an adiabatic stage energy balance with constant cp, optional intercoolers, no solvent "
    "evaporation.",
    "Live steam (steam_species): the steam that heats the feed to T and supplies the heat of desorption "
    "condenses into the liquid at the top; constant latent heat and cp.",
    "Henry's law per solute, y = K x with K = H(T)/P; solutes do not interact.",
    "No solvent evaporation and no carrier dissolution: non-solute species stay on the side they enter.",
    "Constant total flows per stage, corrected through the geometric mean of the end absorption factors.",
    "No reaction (use the difflow_cc amine units for reactive absorption).",
]
_COMMON_EQUATIONS = [
    r"v_{out} = v_{in}(1-f) + \frac{l_{in}}{A} f,\quad f=\frac{A^{N_{eq}+1}-A}{A^{N_{eq}+1}-1},\quad A=\frac{L}{K V}",
    r"N_{eq} = N\,\frac{\ln\!\big(1+E(S-1)\big)}{\ln S},\ S=\frac1A\qquad\text{(Murphree vapor efficiency)}",
    r"Z = H_{OG} N_{OG},\quad H_{OG} = H_G + \lambda H_L,\quad \lambda = \frac{1}{A} = \frac{K V}{L}",
    r"N_{OG} = \frac{\ln\!\big[(1-\lambda)\,\tfrac{y_{in}-Kx_{in}}{y_{out}-Kx_{in}} + \lambda\big]}{1-\lambda}\qquad\text{(Colburn)}",
    r"\text{solute-free line: } G_s\,(Y_{in}-Y) = L_s\,(X_{out}-X),\ X=\tfrac{x}{1-x}",
]
_COMMON_REFERENCES = [
    "Kremser, A. Natl. Petroleum News 22(21), 43 (1930); Edmister, W.C. AIChE J. 3, 165 (1957).",
    "Colburn, A.P. Trans. AIChE 35, 211 (1939).",
    "Onda, K., Takeuchi, H., Okumoto, Y. J. Chem. Eng. Japan 1, 56 (1968).",
    "Seader, Henley & Roper, Separation Process Principles, Ch. 6-7.",
]
_COMMON_UNITS = {
    "henry": "Pa", "henry_C": "K", "T_ref": "K", "n_stages": "-", "stage_efficiency": "-",
    "P": "Pa", "T": "K", "recovery": "-", "diameter": "m", "height": "m", "H_OG": "m",
    "D_G": "m^2/s", "D_L": "m^2/s", "mu_G": "Pa s", "mu_L": "Pa s", "rho_L": "kg/m^3",
    "sigma": "N/m", "MW_gas": "kg/mol", "MW_liquid": "kg/mol", "cp_liquid": "J/(mol K)",
    "cp_gas": "J/(mol K)", "intercooler_T": "K", "intercooler_stages": "-", "steam_latent_heat": "J/mol",
}
_COMMON_SYMBOLS = {"n_stages": "N", "stage_efficiency": r"E_{MV}", "henry": "H", "recovery": r"\phi",
                   "height": "Z", "H_OG": "H_{OG}", "diameter": "D"}


class Absorber(_GasLiquidColumn):
    """Physical-solvent absorber: Kremser, stage-by-stage or packed (HTU/NTU).

    Two inlets, ``gas_in`` (bottom) and ``liquid_in`` (top), and two outlets;
    see the module docstring for the three models. ``params.recovery`` makes the
    stage count (kremser) or the packed height an output. Library helpers for
    the solute-free operating line and the minimum solvent rate are
    :func:`solute_free_operating_line` and :func:`minimum_solvent_rate`.
    """

    symbol = "Absorber"
    _role = "absorber"
    equations = _COMMON_EQUATIONS
    assumptions = _COMMON_ASSUMPTIONS
    references = _COMMON_REFERENCES
    parameter_units = _COMMON_UNITS
    parameter_symbols = _COMMON_SYMBOLS
    numerical_method = ("Closed-form Kremser with a fixed-iteration flow correction; tray-to-tray linear solve "
                        "per solute with Picard sweeps on the stage flows (with energy_balance: damped sweeps, then "
                        "an optimistix Newton solve of all stage equations); Colburn or Gauss-Legendre NTU.")
    outlet_roles = ["gas_out", "liquid_out"]

    def __init__(self, params: AbsorptionParams):
        super().__init__(params)


class Stripper(_GasLiquidColumn):
    """Physical stripper: solute from the liquid into a stripping gas (air, steam).

    The same column as :class:`Absorber` run the other way: ``gas_in`` is the
    stripping gas (bottom), ``liquid_in`` the rich liquid (top);
    ``params.recovery`` is the fraction of the key solute removed from the
    liquid, and ``info`` reports the minimum stripping-gas rate ``G_min``.
    The stripping factor ``S = K V/L`` is in ``info["S"]``.
    """

    symbol = "Stripper"
    _role = "stripper"
    equations = _COMMON_EQUATIONS
    assumptions = _COMMON_ASSUMPTIONS
    references = _COMMON_REFERENCES
    parameter_units = _COMMON_UNITS
    parameter_symbols = _COMMON_SYMBOLS
    numerical_method = Absorber.numerical_method
    outlet_roles = ["gas_out", "liquid_out"]

    def __init__(self, params: AbsorptionParams):
        super().__init__(params)
