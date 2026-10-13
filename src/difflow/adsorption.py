"""Adsorption, ion exchange and chromatography: the general (any-solute) methods.

Pure-JAX library functions (differentiable, float64) for the methods an
introductory separations course teaches. The plugins hold the
domain-specific units built on top of them: ``difflow_cc`` (CO2 PSA/TSA
cycles over a database of adsorbents) and ``difflow_bio`` (Protein A, ion
exchange and size-exclusion columns for proteins).

Isotherms
---------
Loading ``q`` (amount adsorbed per mass, or per volume, of adsorbent) in
equilibrium with fluid concentration ``c`` (any consistent units)::

    linear       q = K c
    Langmuir     q = q_max K c / (1 + K c)          (K is the affinity, 1/conc)
    Freundlich   q = K c**n                         (n < 1 is favourable)

(Geankoplis 5e §12.1; MSH 7e §25.1; Seader 3e §15.3.) :func:`fit_isotherm`
fits any of them to data by least squares, started from the classical
linearizations (``c/q`` against ``c`` for Langmuir, ``ln q`` against ``ln c``
for Freundlich).

Batch (stirred-tank) adsorption
-------------------------------
A volume ``V`` of solution at ``c0`` contacts a mass ``m`` of adsorbent
(initial loading ``q0``) until equilibrium. The solute balance and the
isotherm give the operating line and its intersection with the isotherm::

    V (c0 - c) = m (q(c) - q0)                                     (batch)

:func:`batch_adsorption` solves it by bisection (exact derivative by the
implicit function theorem).

Fixed beds
----------
*Scale-up from a breakthrough curve* (the length of unused bed, LUB;
Collins 1967; Geankoplis 5e §12.3; MSH 7e §25.2). From the effluent curve
``c/c0`` against ``t``, the stoichiometric (ideal-front) time and the usable
time up to the break point ``t_b`` are::

    t_s = int_0^inf (1 - c/c0) dt,      t_u = int_0^t_b (1 - c/c0) dt

The bed used up to the break point is ``L_B = L t_u / t_s`` and the unused
length ``LUB = L - L_B`` is the width of the mass-transfer zone, which
travels unchanged (constant pattern) for a favourable isotherm, so a longer
bed at the same velocity breaks through at ``t_b' = t_s (L' - LUB)/L``
(:func:`lub_analysis`, :func:`lub_break_time`).

*A numerical breakthrough model* (:func:`fixed_bed_breakthrough`).
Isothermal, dilute, constant superficial velocity ``v_s``, a linear driving
force (LDF) for the uptake, and optional axial dispersion::

    eps dc/dt + v_s dc/dz = eps D_ax d2c/dz2 - rho_b dq/dt
    dq/dt = k_LDF (q*(c) - q)

discretized by first-order upwind finite volumes (method of lines) and
integrated by diffrax. Upwinding adds a numerical dispersion
``v dz/2`` (a cell Peclet number of 2 per cell), so a front is spread over a
few cells; the default 80 cells is enough when the LDF spreading dominates
(``k_LDF L / v`` of order 10 or less), and more cells converge it. A
bed's overall material balance holds exactly: the area above the curve,
``int (1 - c/c0) dt``, tends to the stoichiometric time
``t_s = (L / v_s) (eps + rho_b q*(c0) / c0)``.

Without dispersion, a clean bed is solved in the corrected time
``theta = t - eps z / v_s`` (the time since the feed front passed ``z``),
which turns the balance EXACTLY into ``v_s dc/dz = -rho_b dq/dtheta``: the
fluid hold-up drops out, the step is limited by ``k_LDF`` alone and not by
the residence time of a cell, and a liquid-phase bed that lasts months
(residence time minutes) takes a few hundred steps. Along ``z`` each cell
is closed by a backward (implicit, unconditionally stable) step solved by
bisection, ``v_s (c_i - c_{i-1})/dz = -rho_b k_LDF (q*(c_i) - q_i)``.

For a linear isotherm the LDF model has the classical Klinkenberg (1948)
approximation (:func:`klinkenberg_breakthrough`; Ruthven 1984 §8.3), with
``xi = k K rho_b L / v_s`` and ``tau = k (t - eps L / v_s)``::

    c/c0 = (1/2) [1 + erf( sqrt(tau) - sqrt(xi) + 1/(8 sqrt(tau)) + 1/(8 sqrt(xi)) )]

The tests use it as the oracle for the numerical model.

Ion exchange
------------
Exchange of counter-ion A (valence ``z_A``) for B (valence ``z_B``) on a
resin of capacity ``Q`` (equivalents per volume of resin) from a solution of
total normality ``C``. With equivalent fractions ``x`` (solution) and ``y``
(resin) of A, the mass-action law is (Helfferich 1962; Wankat 4e §17.7;
Seader 3e §15.3)::

    (y/x)**z_B ((1-x)/(1-y))**z_A = K_AB (Q/C)**(z_A - z_B)

For equal valences ``y = K x / (1 + (K-1) x)`` (a constant separation
factor). :func:`ion_exchange_equilibrium` solves the general case.

Chromatography
--------------
Retention factor ``k = (t_R - t_0)/t_0``; plate count ``N = 16 (t_R/w)^2``
from the base width (``5.545 (t_R/w_1/2)^2`` from the half-height width);
resolution ``R_s = 2 (t_2 - t_1)/(w_1 + w_2)``; the Purnell equation
``R_s = (sqrt(N)/4) ((alpha-1)/alpha) (k_2/(1+k_2))``; and the van Deemter
plate height ``H = A + B/u + C u``, smallest at ``u = sqrt(B/C)`` where
``H = A + 2 sqrt(B C)`` (Wankat 4e §18.2; Seader 3e §15.6).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.special import erf

__all__ = [
    "linear_isotherm",
    "langmuir_isotherm",
    "freundlich_isotherm",
    "ISOTHERMS",
    "fit_isotherm",
    "batch_adsorption",
    "break_time",
    "lub_analysis",
    "lub_break_time",
    "fixed_bed_breakthrough",
    "klinkenberg_breakthrough",
    "stoichiometric_time",
    "ion_exchange_equilibrium",
    "retention_factor",
    "plate_count",
    "resolution",
    "purnell_resolution",
    "plates_for_resolution",
    "van_deemter",
    "van_deemter_optimum",
    "gaussian_peak",
]

_TINY = 1e-300


# =============================================================================
# Isotherms
# =============================================================================

def linear_isotherm(c: Array | float, K: Array | float) -> Array:
    """Linear (Henry's law) isotherm ``q = K c``.

    Args:
        c: Fluid-phase concentration (any units).
        K: Henry constant (units of q per unit of c).

    Returns:
        Equilibrium loading ``q``.

    Example:
        >>> float(linear_isotherm(2.0, 3.0))
        6.0
    """
    return jnp.asarray(K) * jnp.asarray(c)


def langmuir_isotherm(c: Array | float, q_max: Array | float,
                      K: Array | float | None = None, *,
                      K_d: Array | float | None = None) -> Array:
    """Langmuir isotherm ``q = q_max K c / (1 + K c)``.

    ``K`` is the affinity (adsorption equilibrium) constant in 1/(units of c).
    Texts that write ``q = q_max c / (K_d + c)`` use the dissociation
    constant ``K_d = 1/K`` (Geankoplis 5e eq. 12.1-3).

    Args:
        c: Fluid-phase concentration.
        q_max: Monolayer (saturation) capacity, units of q.
        K: Affinity constant, 1/(units of c).
        K_d: Dissociation constant in units of ``c`` (``K = 1/K_d``), the
            convention of ``difflow_bio.langmuir_isotherm``. Give exactly one
            of ``K`` and ``K_d``.

    Returns:
        Equilibrium loading ``q``.

    Example:
        >>> float(langmuir_isotherm(1.0, 2.0, 1.0))
        1.0
    """
    if (K is None) == (K_d is None):
        raise ValueError("give exactly one of K (affinity) or K_d (dissociation)")
    if K is None:
        c_, K_d_ = jnp.asarray(c), jnp.asarray(K_d)
        return jnp.asarray(q_max) * c_ / (K_d_ + c_)
    Kc = jnp.asarray(K) * jnp.asarray(c)
    return jnp.asarray(q_max) * Kc / (1.0 + Kc)


def freundlich_isotherm(c: Array | float, K: Array | float, n: Array | float) -> Array:
    """Freundlich isotherm ``q = K c**n`` (Geankoplis 5e eq. 12.1-2).

    ``n < 1`` is favourable. Some texts write the exponent as ``1/n``; pass
    its reciprocal. ``c <= 0`` returns 0 with a finite derivative.

    Args:
        c: Fluid-phase concentration.
        K: Freundlich constant (units of q per (units of c)**n).
        n: Exponent (-).

    Returns:
        Equilibrium loading ``q``.

    Example:
        >>> round(float(freundlich_isotherm(1.0, 0.2, 0.25)), 6)
        0.2
    """
    c = jnp.asarray(c)
    pos = c > 0
    c_safe = jnp.where(pos, c, 1.0)
    return jnp.where(pos, jnp.asarray(K) * c_safe ** jnp.asarray(n), 0.0)


#: Isotherm functions by name, ``f(c, **params)``; used by the fitting and
#: fixed-bed functions when an isotherm is named by a string.
ISOTHERMS: dict[str, Callable] = {
    "linear": linear_isotherm,
    "langmuir": langmuir_isotherm,
    "freundlich": freundlich_isotherm,
}
_PARAM_NAMES = {"linear": ("K",), "langmuir": ("q_max", "K"), "freundlich": ("K", "n")}


def _isotherm_fn(isotherm) -> Callable:
    if callable(isotherm):
        return isotherm
    try:
        return ISOTHERMS[isotherm]
    except KeyError:
        raise ValueError(f"Unknown isotherm {isotherm!r}; use one of {sorted(ISOTHERMS)} or a callable") from None


def _linearized_guess(model: str, c: Array, q: Array) -> dict:
    """The classical straight-line estimates (the hand method)."""
    if model == "linear":
        return {"K": jnp.sum(c * q) / jnp.sum(c * c)}
    if model == "langmuir":
        # c/q = 1/(q_max K) + c/q_max: slope 1/q_max, intercept 1/(q_max K)
        slope, icept = jnp.polyfit(c, c / q, 1)
        q_max = 1.0 / slope
        K = slope / icept
        ok = (q_max > 0) & (K > 0)
        return {"q_max": jnp.where(ok, q_max, 2.0 * jnp.max(q)),
                "K": jnp.where(ok, K, 1.0 / jnp.median(c))}
    if model == "freundlich":
        n, lnK = jnp.polyfit(jnp.log(c), jnp.log(q), 1)
        return {"K": jnp.exp(lnK), "n": n}
    raise ValueError(f"Unknown isotherm {model!r}; use one of {sorted(_PARAM_NAMES)}")


def fit_isotherm(
    c: Array,
    q: Array,
    model: str = "langmuir",
    p0: dict | None = None,
    relative: bool = False,
    n_iter: int = 60,
) -> dict:
    """Fit an isotherm to equilibrium data by nonlinear least squares.

    Minimizes ``sum (q_model - q)^2`` (or the relative residuals with
    ``relative=True``, which weights the low-concentration points as much as
    the high ones) by Levenberg-Marquardt on the logarithms of the positive
    parameters (the Freundlich exponent is fitted directly). The start is
    the classical linearization, also returned as ``"linearized"``.

    Args:
        c: Concentrations (1-D, all > 0).
        q: Loadings, same length.
        model: ``"linear"``, ``"langmuir"`` or ``"freundlich"``.
        p0: Optional starting parameters (else the linearized fit).
        relative: Fit relative instead of absolute residuals.
        n_iter: Levenberg-Marquardt iterations.

    Returns:
        Dict with ``"params"`` (dict of fitted parameters), ``"linearized"``
        (the straight-line estimates), ``"r2"`` (coefficient of
        determination of the nonlinear fit on q), ``"rmse"`` and
        ``"q_fit"``.

    Example:
        >>> import jax.numpy as jnp
        >>> c = jnp.array([0.5, 1.0, 2.0, 4.0])
        >>> fit = fit_isotherm(c, langmuir_isotherm(c, 3.0, 0.8))
        >>> round(float(fit["params"]["q_max"]), 4), round(float(fit["params"]["K"]), 4)
        (3.0, 0.8)
    """
    c = jnp.asarray(c, dtype=float)
    q = jnp.asarray(q, dtype=float)
    names = _PARAM_NAMES.get(model)
    if names is None:
        raise ValueError(f"Unknown isotherm {model!r}; use one of {sorted(_PARAM_NAMES)}")
    f = ISOTHERMS[model]
    lin = _linearized_guess(model, c, q)
    start = dict(lin if p0 is None else p0)
    logp = {k: (k != "n") for k in names}

    def to_theta(p):
        return jnp.array([jnp.log(p[k]) if logp[k] else p[k] for k in names], dtype=float)

    def to_params(theta):
        return {k: (jnp.exp(theta[i]) if logp[k] else theta[i]) for i, k in enumerate(names)}

    def resid(theta):
        r = f(c, **to_params(theta)) - q
        return r / q if relative else r

    jac = jax.jacfwd(resid)

    def step(carry, _):
        theta, lam = carry
        r = resid(theta)
        J = jac(theta)
        A = J.T @ J
        g = J.T @ r
        dtheta = -jnp.linalg.solve(A + lam * jnp.diag(jnp.diag(A) + 1e-12), g)
        new = theta + dtheta
        better = jnp.sum(resid(new) ** 2) < jnp.sum(r ** 2)
        theta = jnp.where(better, new, theta)
        lam = jnp.where(better, lam / 3.0, lam * 4.0)
        return (theta, lam), None

    (theta, _), _ = jax.lax.scan(step, (to_theta(start), jnp.asarray(1e-3)), None, length=n_iter)
    params = to_params(theta)
    q_fit = f(c, **params)
    ss_res = jnp.sum((q_fit - q) ** 2)
    ss_tot = jnp.sum((q - jnp.mean(q)) ** 2)
    return {
        "params": params,
        "linearized": lin,
        "r2": 1.0 - ss_res / ss_tot,
        "rmse": jnp.sqrt(ss_res / q.size),
        "q_fit": q_fit,
    }


# =============================================================================
# Root finding with an implicit derivative
# =============================================================================

def _bisect(g: Callable, lo: Array, hi: Array, n_iter: int = 200) -> Array:
    """Root of increasing ``g`` on ``[lo, hi]``; exact first derivative.

    Bisection gives the value (robust for any monotone ``g``, including
    Freundlich's infinite slope at zero). One Newton correction with a
    stopped-gradient slope carries the implicit-function derivative
    ``dx/dtheta = -g_theta / g_x`` through ``g``'s parameters.
    """
    lo0 = jax.lax.stop_gradient(lo)
    hi0 = jax.lax.stop_gradient(hi)

    def body(_, b):
        a, z = b
        m = 0.5 * (a + z)
        up = jax.lax.stop_gradient(g(m)) > 0
        return jnp.where(up, a, m), jnp.where(up, m, z)

    a, z = jax.lax.fori_loop(0, n_iter, body, (lo0, hi0))
    x0 = jax.lax.stop_gradient(0.5 * (a + z))
    slope = jax.lax.stop_gradient(jax.grad(lambda x: jnp.sum(g(x)))(x0))  # g is elementwise
    return x0 - g(x0) / jnp.where(jnp.abs(slope) > 0, slope, 1.0)


# =============================================================================
# Batch adsorption
# =============================================================================

def batch_adsorption(
    c0: Array | float,
    V: Array | float,
    m: Array | float,
    isotherm: str | Callable = "langmuir",
    q0: Array | float = 0.0,
    **iso_params,
) -> dict:
    """Equilibrium of a batch (stirred-tank) adsorber.

    Solves ``V (c0 - c) = m (q(c) - q0)`` for the final concentration ``c``.
    The left side falls and the right side rises with ``c``, so there is one
    root in ``[0, c0 + m q0 / V]``.

    Args:
        c0: Initial solution concentration (e.g. kg/m^3).
        V: Solution volume (e.g. m^3).
        m: Adsorbent mass (e.g. kg); ``q`` is per unit of ``m``.
        isotherm: ``"linear"``, ``"langmuir"``, ``"freundlich"`` or a
            callable ``f(c, **iso_params)``.
        q0: Initial loading of the adsorbent (0 for fresh adsorbent).
        **iso_params: Isotherm parameters (e.g. ``q_max=..., K=...``).

    Returns:
        Dict with ``"c"`` (final concentration), ``"q"`` (final loading),
        ``"removed"`` (fraction of the initial solute that left the
        solution, ``1 - c/c0``) and ``"adsorbed"`` (``V (c0 - c)``).

    Example:
        >>> r = batch_adsorption(1.0, 1.0, 1.0, "linear", K=1.0)
        >>> round(float(r["c"]), 6)
        0.5
    """
    f = _isotherm_fn(isotherm)
    c0, V, m, q0 = (jnp.asarray(x, dtype=float) for x in (c0, V, m, q0))
    c_hi = c0 + m * q0 / V

    def g(c):
        return m * (f(c, **iso_params) - q0) - V * (c0 - c)

    c = _bisect(g, jnp.zeros_like(c_hi), c_hi)
    q = f(c, **iso_params)
    return {"c": c, "q": q, "removed": 1.0 - c / c0, "adsorbed": V * (c0 - c)}


# =============================================================================
# Fixed beds: LUB scale-up
# =============================================================================

def _cumtrapz(y: Array, x: Array) -> Array:
    seg = 0.5 * (y[1:] + y[:-1]) * jnp.diff(x)
    return jnp.concatenate([jnp.zeros(1, dtype=seg.dtype), jnp.cumsum(seg)])


def break_time(t: Array, c_ratio: Array, c_break: Array | float = 0.05) -> Array:
    """Time at which a sampled breakthrough curve first reaches ``c_break``.

    Linear interpolation puts a kink in ``t_b`` wherever the crossing moves
    from one grid cell to the next, so ``d t_b/dL`` of a bed model wobbles by
    a few per cent with the time grid (worse in the curved tail). This uses a
    monotone cubic Hermite interpolant on the bracketing cell (slopes from
    finite differences, limited to ``[0, 3 * secant]``), solved by bisection
    with one implicit Newton step, so the value is accurate between samples
    and the derivative follows the curve rather than the grid.

    Args:
        t: Times (increasing).
        c_ratio: Effluent ``c/c0`` at those times (running maximum is used).
        c_break: Break-point ``c/c0``.

    Returns:
        The break time (the last time if the curve never reaches ``c_break``,
        the first if it starts above it).

    Example:
        >>> import jax.numpy as jnp
        >>> t = jnp.linspace(0.0, 4.0, 5)
        >>> round(float(break_time(t, t / 4.0, 0.3)), 6)
        1.2
    """
    t = jnp.asarray(t, dtype=float)
    r = jax.lax.cummax(jnp.asarray(c_ratio, dtype=float))
    cb = jnp.asarray(c_break, dtype=float)
    n = r.size
    hi = jnp.clip(jnp.searchsorted(r + jnp.arange(n) * 1e-12, cb), 1, n - 1)
    lo = hi - 1
    dt = t[hi] - t[lo]
    delta = (r[hi] - r[lo]) / dt
    m = jnp.gradient(r, t)
    m_lo = jnp.clip(m[lo], 0.0, 3.0 * delta)
    m_hi = jnp.clip(m[hi], 0.0, 3.0 * delta)
    r_lo, r_hi = r[lo], r[hi]

    def h(s):
        s2, s3 = s * s, s * s * s
        return ((2 * s3 - 3 * s2 + 1) * r_lo + (s3 - 2 * s2 + s) * dt * m_lo
                + (-2 * s3 + 3 * s2) * r_hi + (s3 - s2) * dt * m_hi)

    def body(_, b):
        a, c = b
        mid = 0.5 * (a + c)
        below = h(mid) < cb
        return jnp.where(below, mid, a), jnp.where(below, c, mid)

    a, c = jax.lax.fori_loop(0, 60, body, (jnp.zeros(()), jnp.ones(())))
    s0 = jax.lax.stop_gradient(0.5 * (a + c))
    slope = jax.lax.stop_gradient(jax.grad(h)(s0))
    s = s0 - (h(s0) - cb) / jnp.where(jnp.abs(slope) > 1e-30, slope, 1.0)
    s = jnp.clip(s, 0.0, 1.0)
    return t[lo] + s * dt


def lub_analysis(
    t: Array,
    c_ratio: Array,
    L: Array | float,
    c_break: Array | float = 0.05,
    refine: bool = False,
) -> dict:
    """Length of unused bed (LUB) from a measured breakthrough curve.

    Integrates the curve by the trapezoidal rule. The curve must rise to
    (nearly) 1 by its last point, or ``t_s`` is underestimated. The break
    time is the first crossing of ``c_break`` (linear interpolation on the
    running maximum of ``c_ratio``, so noise below the break point does not
    trigger it); ``refine=True`` uses :func:`break_time` instead, so the
    crossing and its gradient do not follow the time grid.

    Args:
        t: Times (increasing), any time unit.
        c_ratio: Effluent ``c/c0`` at those times.
        L: Bed length of the test (any length unit).
        c_break: Break-point ``c/c0`` (e.g. 0.01 or 0.05).
        refine: Locate the break with :func:`break_time` (monotone cubic
            Hermite) instead of linear interpolation. Use it when
            differentiating a modelled curve; the default reproduces a
            graphical reading of measured data.

    Returns:
        Dict with ``"t_break"``, ``"t_stoich"`` (``t_s``), ``"t_usable"``
        (``t_u``), ``"L_used"`` (``L t_u/t_s``), ``"LUB"``,
        ``"fraction_used"`` (``t_u/t_s``) and ``"c_end"`` (the last
        ``c/c0``; check it is close to 1).

    Example:
        >>> import jax.numpy as jnp
        >>> t = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        >>> r = lub_analysis(t, jnp.array([0.0, 0.0, 0.5, 1.0, 1.0]), L=1.0, c_break=0.5)
        >>> float(r["t_stoich"]), float(r["t_break"])
        (2.0, 2.0)
    """
    t = jnp.asarray(t, dtype=float)
    r = jnp.asarray(c_ratio, dtype=float)
    L = jnp.asarray(L, dtype=float)
    deficit = 1.0 - r
    cum = _cumtrapz(deficit, t)
    t_s = cum[-1]
    if refine:
        t_b = break_time(t, r, c_break)
    else:
        # strictly increasing abscissa for interp: add a vanishing ramp
        r_inc = jax.lax.cummax(r) + jnp.arange(r.size) * 1e-12
        t_b = jnp.interp(c_break, r_inc, t)
    t_u = jnp.interp(t_b, t, cum)
    frac = t_u / t_s
    return {
        "t_break": t_b,
        "t_stoich": t_s,
        "t_usable": t_u,
        "L_used": L * frac,
        "LUB": L * (1.0 - frac),
        "fraction_used": frac,
        "c_end": r[-1],
    }


def lub_break_time(
    L_new: Array | float,
    t_stoich: Array | float,
    L: Array | float,
    LUB: Array | float,
) -> Array:
    """Break time of a bed of length ``L_new`` by the LUB method.

    Same adsorbent, particle size, superficial velocity and feed as the test
    bed (length ``L``, stoichiometric time ``t_stoich``, unused length
    ``LUB``): the front moves at ``L / t_stoich`` and the mass-transfer zone
    keeps its width, so ``t_b = t_stoich (L_new - LUB) / L``.

    Args:
        L_new: New bed length.
        t_stoich: Stoichiometric time of the test bed.
        L: Test bed length.
        LUB: Length of unused bed from the test.

    Returns:
        Break time of the new bed (time units of ``t_stoich``).

    Example:
        >>> float(lub_break_time(2.0, 5.0, 1.0, 0.25))
        8.75
    """
    return jnp.asarray(t_stoich) * (jnp.asarray(L_new) - jnp.asarray(LUB)) / jnp.asarray(L)


# =============================================================================
# Fixed beds: numerical breakthrough model
# =============================================================================

def stoichiometric_time(
    L: Array | float,
    v_s: Array | float,
    epsilon: Array | float,
    rho_b: Array | float,
    c_feed: Array | float,
    q_feed: Array | float,
) -> Array:
    """Time for an ideal (shock) front to cross a clean bed.

    ``t_s = (L / v_s) (eps + rho_b q_feed / c_feed)``: the feed needed to
    load the bed to ``q_feed = q*(c_feed)`` and fill its voids.

    Args:
        L: Bed length (m).
        v_s: Superficial velocity (m/s).
        epsilon: Bed void fraction (-).
        rho_b: Bed density (mass of adsorbent per bed volume).
        c_feed: Feed concentration.
        q_feed: Loading in equilibrium with the feed.

    Returns:
        Stoichiometric time (s).

    Example:
        >>> float(stoichiometric_time(1.0, 0.1, 0.4, 500.0, 1.0, 0.01))
        54.0
    """
    return jnp.asarray(L) / jnp.asarray(v_s) * (
        jnp.asarray(epsilon) + jnp.asarray(rho_b) * jnp.asarray(q_feed) / jnp.asarray(c_feed))


def _bed_rhs(t, y, args):
    """Method-of-lines right side in scaled variables x = c/c0, th = q/q0."""
    iso, iso_params, c_feed, q_feed, v_int, D, dz, beta, k, x_in = args
    n = y.shape[0] // 2
    x, th = y[:n], y[n:]
    # inlet value: x_in(t) for a step (1.0) or a pulse handled by caller
    x_up = jnp.concatenate([jnp.atleast_1d(x_in), x[:-1]])
    adv = -v_int * (x - x_up) / dz
    x_dn = jnp.concatenate([x[1:], x[-1:]])
    disp = D * (x_up - 2.0 * x + x_dn) / dz**2
    # smooth positive part: ahead of the front x hovers at 0, and the kink of
    # max(x, 0) there made reverse-mode gradients through the adaptive solve
    # wrong by orders of magnitude (the forward answer was fine)
    x_pos = 0.5 * (x + jnp.sqrt(x * x + 1e-12))
    q_star = iso(x_pos * c_feed, **iso_params) / q_feed
    dth = k * (q_star - th)
    dx = adv + disp - beta * dth
    return jnp.concatenate([dx, dth])


def _march_ct(th, args):
    """Corrected-time form: the outlet of each cell from the loadings.

    Solves ``(x_i - x_{i-1}) + a (phi(x_i) - th_i) = 0`` cell by cell with
    ``phi(x) = q*(x c_feed)/q_feed``; increasing in ``x_i``, bracketed by
    ``[0, 1]`` while ``th <= 1`` and the feed is the largest concentration.
    """
    iso, iso_params, c_feed, q_feed, a = args

    def phi(x):
        return iso(x * c_feed, **iso_params) / q_feed

    def step(x_prev, th_i):
        x_i = _bisect(lambda x: x - x_prev + a * (phi(x) - th_i), jnp.asarray(0.0), jnp.asarray(1.0), n_iter=56)
        return x_i, x_i

    _, x = jax.lax.scan(step, jnp.asarray(1.0), th)
    return x


def _bed_rhs_ct(theta, th, args):
    """d(q/q_feed)/d(theta) in the corrected-time form."""
    iso, iso_params, c_feed, q_feed, a, k = args
    x = _march_ct(th, (iso, iso_params, c_feed, q_feed, a))
    return k * (iso(x * c_feed, **iso_params) / q_feed - th)


def fixed_bed_breakthrough(
    t: Array,
    L: Array | float,
    v_s: Array | float,
    epsilon: Array | float,
    rho_b: Array | float,
    c_feed: Array | float,
    k_ldf: Array | float,
    isotherm: str | Callable = "langmuir",
    D_ax: Array | float = 0.0,
    n_cells: int = 80,
    q_initial: Array | float = 0.0,
    c_initial: Array | float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_steps: int = 400_000,
    solver: str = "tsit5",
    method: str = "auto",
    **iso_params,
) -> dict:
    """Breakthrough curve of an isothermal fixed bed (LDF, method of lines).

    Feed at ``c_feed`` enters at ``t = 0`` a bed initially at
    ``c_initial`` (in the voids) and ``q_initial`` (on the solid), both as
    fractions of the feed values. The initial state is uniform along the
    bed (a clean bed, or one in equilibrium with some concentration).

    The model and its discretization are in the module docstring. Any
    consistent units: ``rho_b q`` must have the units of ``c``. For a
    chromatography column with ``q`` per volume of bed, pass ``rho_b = 1``.

    Args:
        t: Output times (s), increasing, starting at or after 0.
        L: Bed length (m).
        v_s: Superficial velocity (m/s).
        epsilon: Bed (interparticle) void fraction (-).
        rho_b: Bed density (adsorbent mass per bed volume).
        c_feed: Feed concentration.
        k_ldf: LDF mass-transfer coefficient (1/s).
        isotherm: ``"linear"``, ``"langmuir"``, ``"freundlich"`` or a
            module-level callable ``f(c, **iso_params)`` (a new lambda per
            call recompiles).
        D_ax: Axial dispersion coefficient (m^2/s), on top of the upwind
            numerical dispersion ``v dz/2``.
        n_cells: Number of finite volumes (static).
        q_initial: Initial loading as a fraction of ``q*(c_feed)``.
        c_initial: Initial void concentration as a fraction of ``c_feed``.
        rtol, atol: diffrax step-size tolerances.
        max_steps: diffrax step limit.
        solver: ``"tsit5"`` (explicit; fast while the step is set by the
            advection, the usual case) or ``"kvaerno5"`` (implicit; for a
            very fast uptake, ``k_ldf`` far above ``v/dz``; forward only:
            reverse mode through it fails inside lineax with the pinned
            versions). The tsit5 step is capped inside its stability region
            so reverse-mode gradients are correct.
        method: ``"corrected_time"`` (no dispersion, clean bed: exact, and
            fast however long the bed lasts compared with its residence
            time; the usual choice for liquid-phase beds), ``"mol"`` (method
            of lines in real time; needed for ``D_ax > 0`` or a preloaded
            bed) or ``"auto"`` (corrected time when ``D_ax``, ``q_initial``
            and ``c_initial`` are all the Python number 0).
        **iso_params: Isotherm parameters.

    Returns:
        Dict with ``"t"``, ``"c_out"`` (outlet ``c/c_feed`` at ``t``),
        ``"z"`` (cell centres), ``"c_profile"`` and ``"q_profile"``
        (``(len(t), n_cells)``, as fractions of ``c_feed`` and
        ``q*(c_feed)``), ``"q_feed"`` and ``"t_stoich"``
        (:func:`stoichiometric_time`). With the corrected-time method the
        profiles are at ``theta = t - eps L / v_s`` in every cell (they lag
        the real-time profile by at most the bed's void residence time).

    Example:
        >>> import jax.numpy as jnp
        >>> r = fixed_bed_breakthrough(jnp.linspace(0, 400, 5), L=1.0, v_s=0.01, epsilon=0.4,
        ...                            rho_b=1.0, c_feed=1.0, k_ldf=0.5, isotherm="linear", K=1.6)
        >>> round(float(r["t_stoich"]), 1)
        200.0
    """
    import diffrax

    iso = _isotherm_fn(isotherm)
    # arrays, not Python floats: diffrax's jit treats a float in args as static
    # (a recompile per value)
    iso_params = {k: jnp.asarray(v, dtype=float) for k, v in iso_params.items()}
    t = jnp.asarray(t, dtype=float)
    L, v_s, eps, rho_b, c_feed, k, D = (jnp.asarray(x, dtype=float)
                                         for x in (L, v_s, epsilon, rho_b, c_feed, k_ldf, D_ax))
    q_feed = iso(c_feed, **iso_params)
    v_int = v_s / eps
    dz = L / n_cells
    beta = rho_b * q_feed / (eps * c_feed)
    if method == "auto":
        clean = all(isinstance(v, (int, float)) and v == 0 for v in (D_ax, q_initial, c_initial))
        method = "corrected_time" if clean else "mol"
    if method == "corrected_time":
        return _breakthrough_ct(t, L, v_s, eps, rho_b, c_feed, q_feed, k, iso, iso_params, dz, n_cells,
                                rtol, atol, max_steps)
    if method != "mol":
        raise ValueError(f"method must be 'auto', 'corrected_time' or 'mol', not {method!r}")
    y0 = jnp.concatenate([jnp.full(n_cells, c_initial, dtype=float),
                          jnp.full(n_cells, q_initial, dtype=float)])
    args = (iso, iso_params, c_feed, q_feed, v_int, D, dz, beta, k, jnp.asarray(1.0))
    # Cap the step inside the explicit solver's stability region. Without the
    # cap the controller rides the stability boundary: the forward solution is
    # fine, but reverse-mode derivatives of that step sequence are garbage
    # (wrong sign, orders of magnitude off). Largest eigenvalue estimate:
    # advection + dispersion + LDF uptake with the isotherm's secant slope near
    # c = 0 (its steepest part for favourable isotherms).
    slope0 = iso(0.01 * c_feed, **iso_params) / (0.01 * q_feed)
    lam = 2.0 * v_int / dz + 4.0 * D / dz**2 + k * (1.0 + beta * slope0)
    dtmax = jax.lax.stop_gradient(2.5 / lam)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_bed_rhs),
        {"tsit5": diffrax.Tsit5, "kvaerno5": diffrax.Kvaerno5}[solver](),
        t0=0.0,
        t1=t[-1],
        dt0=None,
        y0=y0,
        args=args,
        saveat=diffrax.SaveAt(ts=t),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
        max_steps=max_steps,
    )
    ys = sol.ys
    cp, qp = ys[:, :n_cells], ys[:, n_cells:]
    z = (jnp.arange(n_cells) + 0.5) * dz
    return {
        "t": t,
        "c_out": cp[:, -1],
        "z": z,
        "c_profile": cp,
        "q_profile": qp,
        "q_feed": q_feed,
        "t_stoich": stoichiometric_time(L, v_s, eps, rho_b, c_feed, q_feed),
    }


def _breakthrough_ct(t, L, v_s, eps, rho_b, c_feed, q_feed, k, iso, iso_params, dz, n_cells, rtol, atol,
                     max_steps):
    """Corrected-time solve for :func:`fixed_bed_breakthrough`."""
    import diffrax

    a = rho_b * q_feed * k * dz / (v_s * c_feed)
    lag = eps * L / v_s
    theta = jnp.maximum(t - lag, 0.0)
    # eigenvalues of d(th)/d(theta) lie in [-k, 0]; cap the explicit step
    # inside the stability region (see the "mol" branch for why)
    dtmax = jax.lax.stop_gradient(2.5 / k)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_bed_rhs_ct),
        diffrax.Tsit5(),
        t0=0.0,
        t1=theta[-1],
        dt0=None,
        y0=jnp.zeros(n_cells),
        args=(iso, iso_params, c_feed, q_feed, a, k),
        saveat=diffrax.SaveAt(ts=theta),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol, dtmax=dtmax),
        max_steps=max_steps,
    )
    qp = sol.ys
    cp = jax.vmap(lambda th: _march_ct(th, (iso, iso_params, c_feed, q_feed, a)))(qp)
    # before the feed front leaves the bed, the outlet carries the clean
    # initial fluid (and the shift by the lag carries its derivative)
    c_out = jnp.where(t > lag, cp[:, -1], 0.0)
    return {
        "t": t,
        "c_out": c_out,
        "z": (jnp.arange(n_cells) + 0.5) * dz,
        "c_profile": cp,
        "q_profile": qp,
        "q_feed": q_feed,
        "t_stoich": stoichiometric_time(L, v_s, eps, rho_b, c_feed, q_feed),
    }


def klinkenberg_breakthrough(
    t: Array,
    L: Array | float,
    v_s: Array | float,
    epsilon: Array | float,
    rho_b: Array | float,
    K: Array | float,
    k_ldf: Array | float,
) -> Array:
    """Klinkenberg's approximate breakthrough curve: linear isotherm, LDF.

    ``c/c0 = (1/2)[1 + erf(sqrt(tau) - sqrt(xi) + 1/(8 sqrt(tau)) + 1/(8 sqrt(xi)))]``
    with ``xi = k K rho_b L / v_s`` and ``tau = k (t - eps L / v_s)``
    (Klinkenberg 1948; Ruthven 1984 eq. 8.22). Accurate to about 0.6 % for
    ``xi > 2``; ``c/c0 = 0`` before the fluid itself arrives.

    Args:
        t: Times (s).
        L: Bed length (m).
        v_s: Superficial velocity (m/s).
        epsilon: Void fraction (-).
        rho_b: Bed density.
        K: Henry constant (``q = K c``).
        k_ldf: LDF coefficient (1/s).

    Returns:
        Outlet ``c/c0`` at ``t``.

    Example:
        >>> round(float(klinkenberg_breakthrough(2000.0, 1.0, 0.01, 0.4, 1.0, 16.0, 0.1)), 2)
        0.5
    """
    t = jnp.asarray(t, dtype=float)
    xi = jnp.asarray(k_ldf) * jnp.asarray(K) * jnp.asarray(rho_b) * jnp.asarray(L) / jnp.asarray(v_s)
    tau = jnp.asarray(k_ldf) * (t - jnp.asarray(epsilon) * jnp.asarray(L) / jnp.asarray(v_s))
    pos = tau > 0
    st = jnp.sqrt(jnp.where(pos, tau, 1.0))
    sx = jnp.sqrt(xi)
    val = 0.5 * (1.0 + erf(st - sx + 1.0 / (8.0 * st) + 1.0 / (8.0 * sx)))
    return jnp.where(pos, val, 0.0)


# =============================================================================
# Ion exchange
# =============================================================================

def ion_exchange_equilibrium(
    x: Array | float,
    K: Array | float,
    z_A: float = 1,
    z_B: float = 1,
    C_total: Array | float = 1.0,
    Q: Array | float = 1.0,
) -> Array:
    """Resin equivalent fraction ``y`` of ion A in equilibrium with solution ``x``.

    Mass action for ``z_B A + z_A B(resin) = z_B A(resin) + z_A B``, in
    equivalent fractions::

        (y/x)**z_B ((1-x)/(1-y))**z_A = K (Q/C)**(z_A - z_B)

    For ``z_A = z_B`` the right side is ``K`` and ``y = K x/(1 + (K-1) x)``.
    For a divalent ion replacing a monovalent one (Ca2+ for Na+) dilution
    (small ``C``) favours the divalent ion: the selectivity grows as ``Q/C``.

    Args:
        x: Equivalent fraction of A in solution, in [0, 1].
        K: Selectivity coefficient of A over B (-).
        z_A, z_B: Valences (magnitudes) of A and B.
        C_total: Total solution normality (eq/L); only its ratio to ``Q``
            matters, and only when ``z_A != z_B``.
        Q: Resin capacity (eq/L of resin).

    Returns:
        Equivalent fraction ``y`` of A on the resin.

    Example:
        >>> round(float(ion_exchange_equilibrium(0.5, 3.0)), 6)
        0.75
    """
    x = jnp.asarray(x, dtype=float)
    lnrhs = jnp.log(jnp.asarray(K, dtype=float)) + (z_A - z_B) * jnp.log(
        jnp.asarray(Q, dtype=float) / jnp.asarray(C_total, dtype=float))
    xs = jnp.clip(x, 1e-15, 1.0 - 1e-15)

    def g(y):
        y = jnp.clip(y, 1e-300, 1.0 - 1e-16)
        return (z_B * (jnp.log(y) - jnp.log(xs)) + z_A * (jnp.log1p(-xs) - jnp.log1p(-y))) - lnrhs

    y = _bisect(g, jnp.zeros_like(xs), jnp.ones_like(xs))
    return jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, y))


# =============================================================================
# Chromatography
# =============================================================================

def retention_factor(t_R: Array | float, t_0: Array | float) -> Array:
    """Retention (capacity) factor ``k = (t_R - t_0)/t_0``.

    Args:
        t_R: Retention time of the solute.
        t_0: Retention time of an unretained tracer (dead time).

    Returns:
        ``k`` (-): the ratio of time spent on the stationary phase to time in the mobile phase.

    Example:
        >>> float(retention_factor(6.0, 2.0))
        2.0
    """
    return (jnp.asarray(t_R) - jnp.asarray(t_0)) / jnp.asarray(t_0)


def plate_count(t_R: Array | float, w: Array | float, width: str = "base") -> Array:
    """Theoretical plates from one Gaussian peak.

    ``N = 16 (t_R/w)^2`` with the base width (tangent intercepts, ``4 sigma``)
    or ``N = 8 ln 2 (t_R/w)^2 = 5.545 (t_R/w)^2`` with the width at half height.

    Args:
        t_R: Retention time (or volume).
        w: Peak width, same units.
        width: ``"base"`` or ``"half"``.

    Returns:
        Plate count ``N``.

    Example:
        >>> float(plate_count(10.0, 1.0))
        1600.0
    """
    fac = {"base": 16.0, "half": 8.0 * jnp.log(2.0)}[width]
    return fac * (jnp.asarray(t_R) / jnp.asarray(w)) ** 2


def resolution(t_R1, t_R2, w1, w2) -> Array:
    """Resolution of two peaks ``R_s = 2 (t_R2 - t_R1)/(w1 + w2)`` (base widths).

    ``R_s = 1.5`` is baseline separation of equal Gaussian peaks (0.13 % overlap).

    Args:
        t_R1, t_R2: Retention times.
        w1, w2: Base widths.

    Returns:
        ``R_s`` (-).

    Example:
        >>> float(resolution(10.0, 12.0, 1.0, 1.0))
        2.0
    """
    return 2.0 * (jnp.asarray(t_R2) - jnp.asarray(t_R1)) / (jnp.asarray(w1) + jnp.asarray(w2))


def purnell_resolution(N: Array | float, alpha: Array | float, k2: Array | float) -> Array:
    """Purnell equation ``R_s = (sqrt(N)/4) ((alpha-1)/alpha) (k2/(1+k2))``.

    Efficiency, selectivity and retention, each in its own factor.

    Args:
        N: Plate count of the column.
        alpha: Selectivity ``k2/k1`` (> 1).
        k2: Retention factor of the later peak.

    Returns:
        ``R_s`` (-).

    Example:
        >>> round(float(purnell_resolution(1600.0, 1.25, 4.0)), 3)
        1.6
    """
    alpha = jnp.asarray(alpha)
    k2 = jnp.asarray(k2)
    return jnp.sqrt(jnp.asarray(N)) / 4.0 * (alpha - 1.0) / alpha * k2 / (1.0 + k2)


def plates_for_resolution(R_s: Array | float, alpha: Array | float, k2: Array | float) -> Array:
    """Plates needed for a resolution (the Purnell equation solved for ``N``).

    Args:
        R_s: Target resolution.
        alpha: Selectivity (> 1).
        k2: Retention factor of the later peak.

    Returns:
        ``N = 16 R_s^2 (alpha/(alpha-1))^2 ((1+k2)/k2)^2``.

    Example:
        >>> round(float(plates_for_resolution(1.6, 1.25, 4.0)), 3)
        1600.0
    """
    alpha = jnp.asarray(alpha)
    k2 = jnp.asarray(k2)
    return 16.0 * jnp.asarray(R_s) ** 2 * (alpha / (alpha - 1.0)) ** 2 * ((1.0 + k2) / k2) ** 2


def van_deemter(u: Array | float, A: Array | float, B: Array | float, C: Array | float) -> Array:
    """Van Deemter plate height ``H = A + B/u + C u``.

    Args:
        u: Interstitial (linear) velocity.
        A: Eddy-dispersion term (length).
        B: Axial-diffusion term (length x velocity).
        C: Mass-transfer term (time).

    Returns:
        HETP ``H`` (length).

    Example:
        >>> float(van_deemter(2.0, 1.0, 4.0, 1.0))
        5.0
    """
    u = jnp.asarray(u)
    return jnp.asarray(A) + jnp.asarray(B) / u + jnp.asarray(C) * u


def van_deemter_optimum(A: Array | float, B: Array | float, C: Array | float) -> tuple[Array, Array]:
    """Velocity of the smallest plate height and that height.

    Args:
        A, B, C: Van Deemter coefficients.

    Returns:
        ``(u_opt, H_min) = (sqrt(B/C), A + 2 sqrt(B C))``.

    Example:
        >>> [float(v) for v in van_deemter_optimum(1.0, 4.0, 1.0)]
        [2.0, 5.0]
    """
    B, C = jnp.asarray(B), jnp.asarray(C)
    return jnp.sqrt(B / C), jnp.asarray(A) + 2.0 * jnp.sqrt(B * C)


def gaussian_peak(t: Array, t_R: Array | float, N: Array | float, area: Array | float = 1.0) -> Array:
    """A Gaussian elution peak of ``N`` plates (``sigma = t_R / sqrt(N)``).

    Args:
        t: Times.
        t_R: Retention time.
        N: Plate count.
        area: Peak area (amount injected over the volumetric flow, for a
            concentration-time peak).

    Returns:
        Signal at ``t``, integrating to ``area``.

    Example:
        >>> round(float(gaussian_peak(10.0, 10.0, 100.0)), 4)
        0.3989
    """
    t = jnp.asarray(t)
    s = jnp.asarray(t_R) / jnp.sqrt(jnp.asarray(N))
    return jnp.asarray(area) / (s * jnp.sqrt(2.0 * jnp.pi)) * jnp.exp(-0.5 * ((t - t_R) / s) ** 2)
