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


# ---------------------------------------------------------------------------
# Ternary diagram geometry: mixing point, lever rule, tie lines, Hunter-Nash
# ---------------------------------------------------------------------------
#
# All of this is linear geometry in composition space, so it holds in any basis
# in which the "amounts" are measured: mass fractions with mass amounts, mole
# fractions with moles. The right-triangle coordinates ``(c1, c2)`` are an affine
# image of the composition, so lines and intersections computed there are exact.


def mix(points: Array, amounts: Array) -> Array:
    """Mixing point of compositions ``points`` blended in ``amounts``.

    The composition of the mixture of ``amounts[i]`` of ``points[i]``: the
    amount-weighted average, ``sum(a_i x_i)/sum(a_i)``. On a ternary diagram it
    lies on the line (or inside the polygon) joining the points, which is the
    graphical mixing rule. Pure jnp, so it is differentiable in both arguments;
    it conserves every component (``sum(a) * M = sum(a_i x_i)``).

    Args:
        points: Compositions, shape ``(k, n)`` (``n = 3`` for a ternary).
        amounts: Amounts of each, shape ``(k,)``, in the basis of the compositions.

    Returns:
        The mixture composition, shape ``(n,)``.

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.phase_diagrams import mix
        >>> mix(jnp.array([[1.0, 0, 0], [0, 1.0, 0]]), jnp.array([3.0, 1.0]))
        Array([0.75, 0.25, 0.  ], dtype=float64)
    """
    pts = jnp.asarray(points, dtype=float)
    a = jnp.asarray(amounts, dtype=float)
    return (a @ pts) / jnp.sum(a)


def lever_rule(M: Array, A: Array, B: Array) -> tuple[Array, Array]:
    """Lever rule: the fractions of ``A`` and ``B`` that make up the point ``M``.

    For ``M = f_A A + f_B B`` with ``f_A + f_B = 1`` on the line ``AB``,
    ``f_B = |AM|/|AB|`` and ``f_A = |MB|/|AB|``: the amount of a phase is
    proportional to the length of the *opposite* segment. Exact when ``M`` is
    on the line ``AB`` (as it is for a feed splitting along a tie line or a
    mixture of two streams); an ``M`` off the line is projected onto it. Pure
    jnp and differentiable.

    Args:
        M: The mixing or overall composition.
        A: First end point (for example the raffinate phase).
        B: Second end point (for example the extract phase).

    Returns:
        ``(f_A, f_B)``, the fractions of the total amount at ``M`` that are
        of composition ``A`` and ``B``. Outside the segment one is negative
        (the difference-point arithmetic of Hunter-Nash construction).

    Example:
        >>> import jax.numpy as jnp
        >>> from difflow.phase_diagrams import lever_rule
        >>> A, B = jnp.array([0.9, 0.0, 0.1]), jnp.array([0.1, 0.8, 0.1])
        >>> fA, fB = lever_rule(0.25 * A + 0.75 * B, A, B)
        >>> round(float(fA), 6), round(float(fB), 6)
        (0.25, 0.75)
    """
    M = jnp.asarray(M, dtype=float)
    A = jnp.asarray(A, dtype=float)
    B = jnp.asarray(B, dtype=float)
    d = B - A
    t = jnp.dot(M - A, d) / jnp.dot(d, d)
    return 1.0 - t, t


def _check_tie_data(data) -> np.ndarray:
    import numpy as np
    tl = np.asarray(data["tie_lines"], dtype=float)
    if tl.ndim != 3 or tl.shape[0] < 3:
        raise ValueError("need at least 3 tie lines (run ternary_lle with a larger n_tie)")
    return tl


class _TieCurves:
    """Conjugate curves ``x_I(s)``, ``x_II(s)`` through the tabulated tie lines.

    ``s`` is the tie-line index (0 at the binary edge, ``m - 1`` the last tie
    line before the plait point), interpolated by monotone cubic (PCHIP) in each
    of the right-triangle coordinates ``(c1, c2)`` so a point of one binodal
    branch has a conjugate point of the other at the same ``s``.
    """

    def __init__(self, data, n_grid: int = 4001):
        import numpy as np
        from difflow.staged import _pchip_slopes
        tl = _check_tie_data(data)
        self.m = tl.shape[0]
        s = jnp.arange(self.m, dtype=float)
        self.s = np.asarray(s)
        self.coef = {}
        for ph in (0, 1):
            y = jnp.asarray(tl[:, ph, 1:])
            self.coef[ph] = (np.asarray(y), np.stack([np.asarray(_pchip_slopes(s, y[:, j])) for j in (0, 1)], axis=1))
        self.grid = np.linspace(0.0, self.m - 1.0, n_grid)
        self.dense = {ph: self.point(ph, self.grid) for ph in (0, 1)}

    def point(self, phase: int, sq):
        """``(c1, c2)`` of the phase-``phase`` end of the tie line at index ``sq`` (scalar or array)."""
        import numpy as np
        y, d = self.coef[phase]
        sq = np.asarray(sq, dtype=float)
        i = np.clip(np.floor(sq).astype(int), 0, self.m - 2)
        t = sq - i
        t2, t3 = t * t, t * t * t
        t_, t2_, t3_ = t[..., None], t2[..., None], t3[..., None]
        return ((2 * t3_ - 3 * t2_ + 1) * y[i] + (t3_ - 2 * t2_ + t_) * d[i]
                + (-2 * t3_ + 3 * t2_) * y[i + 1] + (t3_ - t2_) * d[i + 1])

    def roots(self, g_at) -> list:
        """Parameters ``s`` (ascending) where ``g_at(s)`` changes sign on the dense grid, refined by bisection."""
        import numpy as np
        g = g_at(self.grid)
        out = []
        for i in np.nonzero(g[:-1] * g[1:] <= 0.0)[0]:
            lo, hi, glo = self.grid[i], self.grid[i + 1], g[i]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                gm = float(g_at(np.array(mid)))
                if glo * gm <= 0.0:
                    hi = mid
                else:
                    lo, glo = mid, gm
            out.append(0.5 * (lo + hi))
        return out


def _c3(c12) -> "np.ndarray":
    import numpy as np
    c12 = np.asarray(c12, dtype=float)
    return np.concatenate([1.0 - c12.sum(axis=-1, keepdims=True), c12], axis=-1)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def tie_line_through(lle_data: dict, point, n_grid: int = 4001) -> dict:
    """The tie line through a point of the two-phase region, by interpolation.

    The tabulated tie lines of :func:`ternary_lle` are joined into a conjugate
    curve (the end points of one tie line share a parameter ``s``, interpolated by
    monotone cubics), and ``s`` is found where the segment between the two
    interpolated ends passes through ``point``. A numeric, non-differentiated
    companion of :func:`ternary_lle`; its accuracy is that of the interpolation, so
    use enough tie lines (``n_tie``) for the accuracy you need.

    Args:
        lle_data: Result of :func:`ternary_lle`.
        point: Overall composition ``(3,)`` inside the two-phase region.
        n_grid: Grid points of the dense search along the conjugate curve.

    Returns:
        dict with ``x_I`` and ``x_II`` (the two phase compositions, phase I being
        species-0 rich as in :func:`ternary_lle`), ``frac_I`` and ``frac_II``
        (lever-rule fractions of the total amount in each phase), and ``s``
        (the interpolation index).

    Raises:
        ValueError: If ``point`` is not on a tie line of the tabulated range (a
            single-phase point, or closer to the plait point than the last tie line).
    """
    import numpy as np
    curves = _TieCurves(lle_data, n_grid)
    p = np.asarray(point, dtype=float)[1:]
    # collinearity of (x_II(s) - x_I(s)) with (point - x_I(s)) along the conjugate curve
    def g_at(sq):
        a, b = curves.point(0, sq), curves.point(1, sq)
        return _cross(b - a, p - a)

    hits = []
    for s in curves.roots(g_at):
        a, b = curves.point(0, s), curves.point(1, s)
        t = float(np.dot(p - a, b - a) / np.dot(b - a, b - a))
        if -1e-9 <= t <= 1.0 + 1e-9:
            hits.append((s, a, b, t))
    if not hits:
        raise ValueError("point is not on a tabulated tie line: it is outside the two-phase region "
                         "or closer to the plait point than the last tie line")
    s, a, b, t = hits[0]
    return {"x_I": jnp.asarray(_c3(a)), "x_II": jnp.asarray(_c3(b)),
            "frac_I": jnp.asarray(1.0 - t), "frac_II": jnp.asarray(t), "s": s}


def difference_point(F, E1, S, R_N, F_amount: float = 1.0, E1_amount: float | None = None,
                     S_amount: float = 1.0, R_N_amount: float | None = None) -> dict:
    """The Hunter-Nash difference point ``Delta = F - E_1 = R_N - S``.

    The net flow toward the raffinate end of a countercurrent cascade, in the
    extensive (amount-weighted) sense ``Delta = F - E_1 = R_N - S``. Every
    operating line ``E_{n+1} - R_n = -Delta`` passes through it. Geometrically it
    is the intersection of the lines ``F E_1`` and ``S R_N``; it lies *outside*
    the triangle, and at infinity (``amount = 0``) when the two lines are
    parallel (equal net flow), in which case ``composition`` is NaN and the
    operating lines are parallel.

    Pure jnp and differentiable. Give the amounts to get the extensive vector;
    with only the compositions ``E1_amount`` and ``R_N_amount`` follow from the
    mass balance ``F + S = E_1 + R_N`` and the lever rule on ``R_N M E_1``.

    Args:
        F: Feed composition ``(3,)``.
        E1: First-stage extract composition (leaving stage 1).
        S: Solvent composition.
        R_N: Final raffinate composition.
        F_amount: Feed amount.
        E1_amount: Extract amount (default: from the lever rule).
        S_amount: Solvent amount.
        R_N_amount: Final raffinate amount (default: total minus ``E1_amount``).

    Returns:
        dict with ``composition`` (``Delta`` as a composition, NaN at infinity),
        ``amount`` (``F - E_1``, signed), ``vector`` (the extensive 3-vector
        ``F_amount F - E1_amount E1``), ``M``, ``E1_amount`` and ``R_N_amount``.
        The operating line through a raffinate point ``R_n`` has direction
        ``-(vector - amount * R_n)`` in composition space (valid also when
        ``Delta`` is at infinity).
    """
    F, E1, S, R_N = (jnp.asarray(v, dtype=float) for v in (F, E1, S, R_N))
    M = mix(jnp.stack([F, S]), jnp.asarray([F_amount, S_amount]))
    tot = F_amount + S_amount
    if E1_amount is None:
        _, fE = lever_rule(M, R_N, E1)
        E1_amount = fE * tot
    if R_N_amount is None:
        R_N_amount = tot - E1_amount
    vec = F_amount * F - E1_amount * E1
    amt = F_amount - E1_amount
    near0 = jnp.abs(amt) < 1e-12
    comp = jnp.where(near0, jnp.nan, vec / jnp.where(near0, 1.0, amt))
    return {"composition": comp, "amount": amt, "vector": vec, "M": M,
            "E1_amount": E1_amount, "R_N_amount": R_N_amount}


def hunter_nash(lle_data: dict, F, S, x_RN, S_over_F: float = 1.0, carrier: int = 0,
                max_stages: int = 100, n_grid: int = 4001) -> dict:
    """Countercurrent extraction stages stepped off on the triangle (Hunter-Nash).

    The graphical construction, done numerically on the conjugate curves of the
    tabulated tie lines (see :func:`tie_line_through`): mix ``F`` and ``S`` to
    ``M``; the line ``R_N M`` meets the extract branch at ``E_1``; the difference
    point is ``Delta = F - E_1 = R_N - S``; then alternate a tie line (``E_n`` to
    ``R_n``) with an operating line from ``Delta`` through ``R_n`` to ``E_{n+1}``
    until the raffinate reaches the specified ``R_N``. The operating lines are
    carried by the extensive balance ``E_{n+1} = R_n + E_1 - F``, so they remain
    valid when ``Delta`` is at infinity. Not differentiated (like
    :func:`ternary_lle`); the stage count is real, the last stage counted by the
    fraction of the solute step needed to reach ``R_N``.

    Species 0 and 1 are the partially miscible pair and species 2 the solute,
    as in :func:`ternary_lle`; ``carrier`` is the feed diluent (its phase is
    the raffinate), so ``carrier=0`` has the species-1-rich phase II as the
    extract and ``carrier=1`` the reverse.

    Args:
        lle_data: Result of :func:`ternary_lle` (at least 3 tie lines).
        F: Feed composition ``(3,)``.
        S: Solvent composition ``(3,)``.
        x_RN: Final raffinate: a composition ``(3,)``, or a scalar solute
            fraction (the point of that solute fraction on the raffinate branch).
        S_over_F: Solvent amount per unit feed amount (same basis as the compositions).
        carrier: ``0`` or ``1``: the feed-diluent species that makes the raffinate phase.
        max_stages: Stepping limit.
        n_grid: Dense search grid along the conjugate curves.

    Returns:
        dict with ``M``, ``E1``, ``R_N``, ``delta`` (the :func:`difference_point`
        dict), ``E`` and ``R`` (``(n, 3)`` extract and raffinate leaving stages
        ``1..n``), ``n_stages`` (real; ``inf`` at a pinch, NaN if stepping
        fails), ``n_ideal`` (the integer ``ceil``), ``feasible``, ``F``, ``S``,
        ``carrier`` and ``lle_data`` (for :func:`difflow.visualization.plot_hunter_nash`).

    Raises:
        ValueError: If ``R_N`` is not on the raffinate branch within the tabulated tie lines,
            or the mixing point is not in the two-phase region (the line ``R_N M`` meets no
            extract-branch point beyond ``M``).
    """
    import numpy as np
    if carrier not in (0, 1):
        raise ValueError("carrier must be 0 or 1")
    raff, ext = (0, 1) if carrier == 0 else (1, 0)
    curves = _TieCurves(lle_data, n_grid)
    F = np.asarray(F, dtype=float)
    S = np.asarray(S, dtype=float)
    M = np.asarray(mix(np.stack([F, S]), np.array([1.0, S_over_F])))

    # final raffinate
    if np.ndim(x_RN) == 0:
        c = float(x_RN)
        r = curves.roots(lambda sq: curves.point(raff, sq)[..., 1] - c)
        if not r:
            raise ValueError("x_RN is not within the solute range of the raffinate branch "
                             "of the tabulated tie lines")
        R_N = _c3(curves.point(raff, r[0]))
    else:
        R_N = np.asarray(x_RN, dtype=float)
    cRN = float(R_N[2])

    # E_1: the line R_N -> M continued beyond M to the extract branch
    u = (M - R_N)[1:]
    t_M = 1.0
    def hit(p0, u, t_min):
        def g_at(sq):
            return _cross(curves.point(ext, sq) - p0, u)
        sols = []
        for s in curves.roots(g_at):
            P = curves.point(ext, s)
            t = float(np.dot(P - p0, u) / np.dot(u, u))
            if t > t_min:
                sols.append((s, P))
        return sols

    sols = hit(R_N[1:], u, t_M + 1e-9)
    if not sols:
        raise ValueError("the line R_N-M meets no extract-branch point beyond M: the mixing "
                         "point is outside the two-phase region (or beyond the tabulated tie lines)")
    s1, E1c = sols[0]
    E1 = _c3(E1c)
    dp = difference_point(F, E1, S, R_N, 1.0, None, S_over_F)
    w = -np.asarray(dp["vector"])           # E_1 - F in extensive terms: E_{n+1} = R_n + w
    t_w = -float(dp["amount"])

    Es, Rs, c_prev = [], [], float(F[2])
    n_stages, feasible = np.nan, True
    s = s1
    for n in range(1, max_stages + 1):
        E_n, R_n = _c3(curves.point(ext, s)), _c3(curves.point(raff, s))
        Es.append(E_n)
        Rs.append(R_n)
        c_n = float(R_n[2])
        if c_n <= cRN:
            n_stages = (n - 1) + (c_prev - cRN) / (c_prev - c_n)
            break
        if c_n >= c_prev - 1e-12:
            n_stages, feasible = np.inf, False      # pinch: no progress
            break
        c_prev = c_n
        un = (w - t_w * R_n)[1:]
        nxt = hit(R_n[1:], un, 1e-12)
        if not nxt:
            feasible = False                         # operating line leaves the tabulated range
            break
        s = nxt[0][0]
    else:
        feasible = False
    return {"M": jnp.asarray(M), "E1": jnp.asarray(E1), "R_N": jnp.asarray(R_N), "delta": dp,
            "E": jnp.asarray(np.array(Es)), "R": jnp.asarray(np.array(Rs)),
            "n_stages": float(n_stages),
            "n_ideal": int(np.ceil(n_stages - 1e-9)) if np.isfinite(n_stages) else None,
            "feasible": bool(feasible), "F": jnp.asarray(F), "S": jnp.asarray(S), "carrier": carrier,
            "lle_data": lle_data}
