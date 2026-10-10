"""Phase-diagram data: Txy, Pxy, xy, azeotropes and ternary LLE.

Pure JAX functions that return the curves used to read and teach phase
behaviour. Binary VLE uses the modified Raoult's law

.. math:: y_i P = x_i\\,\\gamma_i(x, T)\\,P_i^{sat}(T)

with ``gamma = 1`` when no ``activity_model`` is given (Raoult's law). Bubble
temperatures are root finds (optimistix Newton, implicit differentiation), so
every output is differentiable with respect to ``P``, ``T`` and the activity
parameters. Plot helpers live in :mod:`difflow.visualization.phase_diagrams`.

Limits
------
Vapor is ideal (low pressure) and the pure-component vapor pressures come from
the thermo object's Antoine equations. The binary helpers need exactly two
species.

References
----------
- Smith, Van Ness, Abbott. Introduction to Chemical Engineering
  Thermodynamics, 7e, Ch. 10-12.
- Seader, Henley, Roper. Separation Process Principles, 3e, Ch. 4 and 8.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.activity import activity_gamma



def _gamma(activity_model, x: Array, T: Array) -> Array:
    if activity_model is None:
        return jnp.ones_like(x)
    return activity_gamma(activity_model, x, T)


def _check_binary(species) -> tuple[str, str]:
    species = tuple(species)
    if len(species) != 2:
        raise ValueError(f"binary diagram helpers need exactly 2 species, got {len(species)}")
    return species


def _psat(thermo, species, T) -> Array:
    return jnp.stack([thermo.Psat(s, T) for s in species])


def _ideal_bubble_T(thermo, species, x: Array, P: Array) -> Array:
    """Initial guess: mole-fraction weighted Antoine normal-boiling temperatures."""
    Ts = []
    for s in species:
        A, B, C = thermo.species[s].antoine_coeffs
        Ts.append(B / (A - jnp.log10(P)) - C)
    return jnp.dot(x, jnp.stack(Ts))


def bubble_T(thermo, species, x: Array, P: Array | float,
             activity_model=None) -> Array:
    """Bubble-point temperature of liquid ``x`` at pressure ``P``.

    Solves ``ln(sum_i x_i gamma_i(x,T) Psat_i(T) / P) = 0`` by Newton's method
    with implicit differentiation.

    Args:
        thermo: Object with ``Psat(species, T)`` and ``species`` (e.g. ``IdealThermo``).
        species: Species names in the order of ``x``.
        x: Liquid mole fractions.
        P: Pressure (Pa).
        activity_model: Optional activity model (``gamma(x, T)`` protocol).

    Returns:
        Bubble temperature (K).
    """
    species = tuple(species)
    x = jnp.asarray(x, dtype=float)
    P = jnp.asarray(P, dtype=float)

    def resid(T, args):
        x_, P_ = args
        return jnp.log(jnp.sum(x_ * _gamma(activity_model, x_, T) * _psat(thermo, species, T)) / P_)

    T0 = _ideal_bubble_T(thermo, species, x, P)
    sol = optx.root_find(resid, optx.Newton(rtol=1e-10, atol=1e-10), T0,
                         args=(x, P), max_steps=60, throw=False)
    return sol.value


def txy(thermo, species, P: Array | float, n: int = 51,
        activity_model=None) -> dict[str, Array]:
    """Isobaric Txy diagram of a binary mixture.

    Args:
        thermo: Thermo object with ``Psat`` (``IdealThermo``).
        species: The two species names; ``x`` and ``y`` refer to the first.
        P: Pressure (Pa).
        n: Number of liquid-composition grid points (``x`` from 0 to 1).
        activity_model: Optional activity model; ``None`` means Raoult's law.

    Returns:
        dict with ``x`` (liquid mole fraction of ``species[0]``), ``y`` (vapor
        mole fraction in equilibrium), ``T`` (bubble temperature, K), ``P``
        and ``species``. The dew curve is ``(y, T)``.
    """
    species = _check_binary(species)
    P = jnp.asarray(P, dtype=float)
    x1 = jnp.linspace(0.0, 1.0, n)
    X = jnp.stack([x1, 1.0 - x1], axis=1)

    def point(xv):
        T = bubble_T(thermo, species, xv, P, activity_model)
        y = xv * _gamma(activity_model, xv, T) * _psat(thermo, species, T) / P
        return T, y[0]

    T, y1 = jax.vmap(point)(X)
    return {"x": x1, "y": y1, "T": T, "P": P, "species": species}


def pxy(thermo, species, T: Array | float, n: int = 51,
        activity_model=None) -> dict[str, Array]:
    """Isothermal Pxy diagram of a binary mixture.

    Args:
        thermo: Thermo object with ``Psat``.
        species: The two species names; ``x`` and ``y`` refer to the first.
        T: Temperature (K).
        n: Number of grid points.
        activity_model: Optional activity model; ``None`` means Raoult's law.

    Returns:
        dict with ``x``, ``y``, ``P`` (bubble pressure, Pa), ``T`` and
        ``species``. The dew curve is ``(y, P)``.
    """
    species = _check_binary(species)
    T = jnp.asarray(T, dtype=float)
    x1 = jnp.linspace(0.0, 1.0, n)
    X = jnp.stack([x1, 1.0 - x1], axis=1)
    ps = _psat(thermo, species, T)

    def point(xv):
        pi = xv * _gamma(activity_model, xv, T) * ps
        P = jnp.sum(pi)
        return P, pi[0] / P

    P, y1 = jax.vmap(point)(X)
    return {"x": x1, "y": y1, "P": P, "T": T, "species": species}


def _ln_alpha(thermo, species, activity_model, x1, P, T):
    """ln of the relative volatility K1/K2 at liquid composition x1."""
    xv = jnp.stack([x1, 1.0 - x1])
    if T is None:
        Tb = bubble_T(thermo, species, xv, P, activity_model)
    else:
        Tb = T
    g = _gamma(activity_model, xv, Tb)
    ps = _psat(thermo, species, Tb)
    return jnp.log(g[0] * ps[0] / (g[1] * ps[1])), Tb


def xy_curve(thermo, species, P: Array | float, n: int = 51,
             activity_model=None) -> dict[str, Array]:
    """Constant-pressure y-x curve with the relative volatility alpha(x).

    Args:
        thermo: Thermo object with ``Psat``.
        species: The two species names.
        P: Pressure (Pa).
        n: Number of grid points.
        activity_model: Optional activity model; ``None`` means Raoult's law.

    Returns:
        dict with ``x``, ``y``, ``T`` (bubble temperature), ``alpha`` (the
        relative volatility ``K1/K2 = (y1/x1)/(y2/x2)``, finite at the
        endpoints) and ``species``.
    """
    species = _check_binary(species)
    d = txy(thermo, species, P, n, activity_model)
    P = jnp.asarray(P, dtype=float)
    la = jax.vmap(lambda x1: _ln_alpha(thermo, species, activity_model, x1, P, None)[0])(d["x"])
    return {"x": d["x"], "y": d["y"], "T": d["T"], "alpha": jnp.exp(la),
            "P": P, "species": species}


def find_azeotrope(thermo, species, P: Array | float | None = None,
                   T: Array | float | None = None, activity_model=None,
                   n_scan: int = 101) -> dict[str, Array]:
    """Locate a binary azeotrope (``y = x``, ``alpha = 1``).

    Specify exactly one of ``P`` (azeotrope temperature is returned) or ``T``
    (azeotrope pressure is returned). The interior grid is scanned for a sign
    change of ``ln alpha`` (the first one is taken) and the root is refined
    by bisection with implicit differentiation. There is no Python branching
    on traced values: when there is no azeotrope the result is masked.

    Args:
        thermo: Thermo object with ``Psat``.
        species: The two species names.
        P: Pressure (Pa) for an isobaric azeotrope.
        T: Temperature (K) for an isothermal azeotrope.
        activity_model: Optional activity model (without one there is no
            azeotrope for constant-volatility Raoult mixtures).
        n_scan: Scan-grid points.

    Returns:
        dict with ``found`` (bool array), ``x`` (azeotrope mole fraction of
        ``species[0]``; NaN when not found), ``T`` and ``P`` (NaN when not
        found; the specified one is echoed).
    """
    species = _check_binary(species)
    if (P is None) == (T is None):
        raise ValueError("specify exactly one of P or T")
    Pj = None if P is None else jnp.asarray(P, dtype=float)
    Tj = None if T is None else jnp.asarray(T, dtype=float)

    def f(x1, _=None):
        return _ln_alpha(thermo, species, activity_model, x1, Pj, Tj)[0]

    grid = jnp.linspace(0.0, 1.0, n_scan)
    vals = jax.vmap(f)(grid)
    # sign change between consecutive grid points (endpoints included).
    changed = (vals[:-1] * vals[1:]) < 0.0
    found = jnp.any(changed)
    i = jnp.argmax(changed)
    lo = jnp.where(found, grid[i], 0.25)
    hi = jnp.where(found, grid[i + 1], 0.75)

    # Bisection locates the root; the bracket is only a start (no gradient),
    # then a Newton solve from there carries the implicit derivative. When
    # nothing was found the residual is replaced by x - 0.5, so the masked
    # branch stays finite under differentiation.
    def bis(_, st):
        l, h = st
        m = 0.5 * (l + h)
        left = f(l) * f(m) <= 0.0
        return jnp.where(left, l, m), jnp.where(left, m, h)

    lo, hi = jax.lax.stop_gradient(jax.lax.fori_loop(0, 40, bis, (lo, hi)))
    x_b = 0.5 * (lo + hi)

    def g(x1, _):
        return jnp.where(found, f(x1), x1 - 0.5)

    sol = optx.root_find(g, optx.Newton(rtol=1e-12, atol=1e-12), x_b,
                         max_steps=20, throw=False)
    x_az = sol.value
    _, Tb = _ln_alpha(thermo, species, activity_model, x_az, Pj, Tj)
    nan = jnp.nan
    if Pj is not None:
        out_T = jnp.where(found, Tb, nan)
        out_P = jnp.where(found, Pj, nan)
    else:
        xv = jnp.stack([x_az, 1.0 - x_az])
        Pz = jnp.sum(xv * _gamma(activity_model, xv, Tj) * _psat(thermo, species, Tj))
        out_T = jnp.where(found, Tj, nan)
        out_P = jnp.where(found, Pz, nan)
    return {"found": found, "x": jnp.where(found, x_az, nan),
            "T": out_T, "P": out_P, "species": species}


# ---------------------------------------------------------------------------
# Ternary liquid-liquid equilibrium
# ---------------------------------------------------------------------------

def _lle_model(lle_model):
    """Extract an activity model and species from an LLEEquilibrium or model."""
    from difflow.units.lle import LLEEquilibrium
    if isinstance(lle_model, LLEEquilibrium):
        if lle_model.activity_model == "NRTL" and lle_model.nrtl_params is not None:
            return lle_model.nrtl_params
        if lle_model.activity_model == "UNIQUAC" and lle_model.uniquac_params is not None:
            return lle_model.uniquac_params
        raise ValueError("ternary_lle needs an LLEEquilibrium with activity_model "
                         "'NRTL' or 'UNIQUAC' and the matching parameters")
    return lle_model


def ternary_lle(lle_model, T: Array | float, n_tie: int = 15,
                c_max: float = 0.6, guess: tuple | None = None) -> dict:
    """Binodal curve and tie lines of a type-I ternary LLE system.

    Species 0 and 1 are the partially miscible pair and species 2 the solute.
    Starting on the 0-1 edge (binary LLE), the solute mole fraction ``c`` in
    the species-0-rich phase I is swept upward; at each ``c`` the isoactivity
    conditions ``x_i^I gamma_i^I = x_i^II gamma_i^II`` (three equations in the
    three remaining unknowns) are solved by Newton's method, using the
    previous tie line as the initial guess (continuation with step halving).
    The sweep ends at the plait point, where the two phases merge (or at
    ``c_max``); only converged, non-trivial tie lines are returned.

    This routine solves the equations at fixed numbers; it is not
    differentiated.

    Args:
        lle_model: An :class:`~difflow.units.lle.LLEEquilibrium` using NRTL or
            UNIQUAC, or any activity model with three species.
        T: Temperature (K).
        n_tie: Number of tie lines.
        c_max: Largest solute fraction in phase I to attempt.
        guess: Optional ``(xA_I, xA_II)`` initial binary-edge mole fractions
            of species 0 in phases I and II (default ``(0.95, 0.05)``).

    Returns:
        dict with ``tie_lines`` (array ``(m, 2, 3)``: phase I and II
        compositions), ``valid`` (``(n_tie + 1,)`` bool: the binary-edge line plus ``n_tie`` targets), ``binodal`` (``(2m, 3)``,
        phase-I branch followed by the reversed phase-II branch, so it is a
        continuous closed-at-the-plait-point curve), ``plait`` (approximate
        plait point, the last tie line's midpoint) and ``T``.
    """
    import numpy as np
    model = _lle_model(lle_model)
    T = jnp.asarray(T, dtype=float)
    a0, b0 = guess if guess is not None else (0.95, 0.05)

    def resid(u, args):
        c, = args
        a1 = u[0]
        xI = jnp.stack([a1, 1.0 - c - a1, c])
        xII = jnp.stack([u[1], u[2], 1.0 - u[1] - u[2]])
        return (jnp.log(xI * activity_gamma(model, xI, T))
                - jnp.log(xII * activity_gamma(model, xII, T)))

    solver = optx.Newton(rtol=1e-12, atol=1e-12)

    def attempt(u0, c):
        sol = optx.root_find(resid, solver, u0, args=(c,), max_steps=60, throw=False)
        uu = sol.value
        r = float(jnp.max(jnp.abs(resid(uu, (c,)))))
        xI = np.array([float(uu[0]), 1.0 - c - float(uu[0]), c])
        xII = np.array([float(uu[1]), float(uu[2]), 1.0 - float(uu[1]) - float(uu[2])])
        ok = (r < 1e-8 and np.all(xI > 0) and np.all(xII > 0)
              and np.max(np.abs(xI - xII)) > 1e-3)
        return ok, uu, np.stack([xI, xII])

    # Start on the binary edge, then continue in c with step halving: a failed
    # step is retried shorter, and the sweep ends when the step underflows
    # (the plait point, where the two phases merge).
    targets = np.linspace(0.0, c_max, n_tie + 1)[1:]
    c, h = 1e-6, c_max / n_tie
    u = jnp.array([a0, b0, 1.0 - b0 - 1e-6])
    ok, u, tl = attempt(u, c)
    lines, valid = ([tl], [True]) if ok else ([], [])
    done = not ok
    for tgt in ([] if done else targets):
        while c < tgt - 1e-12 and not done:
            c_try = min(tgt, c + h)
            ok, u_try, tl = attempt(u, c_try)
            if ok:
                c, u, h = c_try, u_try, min(h * 1.5, c_max / n_tie)
            else:
                h *= 0.5
                if h < 1e-7:
                    done = True
        if done:
            break
        lines.append(tl)
        valid.append(True)
    valid_arr = np.zeros(n_tie + 1, dtype=bool)
    valid_arr[:len(valid)] = valid
    good = np.array(lines).reshape(-1, 2, 3)
    if len(good):
        binodal = np.concatenate([good[:, 0, :], good[::-1, 1, :]], axis=0)
        plait = good[-1].mean(axis=0)
    else:
        binodal = np.zeros((0, 3))
        plait = np.full(3, np.nan)
    return {"tie_lines": jnp.asarray(good), "valid": jnp.asarray(valid_arr),
            "binodal": jnp.asarray(binodal), "plait": jnp.asarray(plait), "T": T}
