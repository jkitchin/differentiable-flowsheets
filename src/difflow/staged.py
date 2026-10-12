"""Graphical equilibrium-stage design: McCabe-Thiele stepping, pinches, Kremser.

Pure JAX functions (float64) for binary or pseudo-binary cascades. One stage
stepper serves distillation (Ch 12), absorption/stripping (Ch 15) and
immiscible extraction (Ch 17): it works on any equilibrium curve ``y*(x)`` and
any pair of straight operating lines.

Conventions
-----------
* A line is an array ``(a, b, c)`` meaning ``a x + b y = c``. Never slope and
  intercept, so a vertical q-line (q = 1) is representable.
* ``x`` is the liquid (or raffinate) composition, ``y`` the vapor (extract).
* Stepping starts at the *top* of the cascade and goes down. Stage ``k`` has
  the vapor ``y_k`` leaving it; ``x_k = x*(y_k)`` leaves it as liquid; the
  operating line then gives ``y_{k+1} = line(x_k)``. For distillation the
  top point is ``(x_D, x_D)``, so ``y_1 = x_D``. The staircase therefore
  counts the *reboiler* as the last stage and, with a partial condenser,
  the first stage is the condenser.

Fractional stage count
----------------------
The stepper returns ``N = sum_k c_k``, where ``c_k`` is the fraction of stage
``k`` needed. Full stages count 1, stages past the end count 0, and the stage
that crosses the target composition counts the fraction of its step needed:

* ``fraction="linear"`` (default): linear interpolation in ``x`` between the
  last two stage compositions, the textbook "fraction of a stage".
* ``fraction="log"``: assume the steps are geometric (``x_k - x_inf``
  proportional to ``r**k``), which is exact for straight equilibrium and
  operating lines (the Kremser case).
* ``fraction="log_ratio"``: the same in ``x/(1-x)``, exact at total reflux
  and constant alpha (the Fenske case).

``N`` is continuous in every input but only piecewise smooth: its derivative
jumps where one more whole stage starts to be needed. The convention changes
``dN/dR`` (not ``N`` at whole stages), so quote it with the result.
The feed switch is a sigmoid of width ``feed_width`` in ``x``.

Pinches
-------
``minimum_reflux`` takes the largest chord slope from ``(x_D, x_D)`` to the
curve over ``[x_q, x_D)`` (``x_q`` the q-line/curve intersection), so the
feed-line pinch and any tangent pinch are one computation and constant alpha
needs no special case. The maximiser is located with a stop-gradient search
and the value is re-evaluated there, so derivatives follow the envelope
theorem (and the endpoint dependence on ``z`` and ``q``). The result is not
floored.

Limits
------
Constant molar overflow (straight operating lines) and a single binary
equilibrium curve. A Murphree *vapor* efficiency (``curve.with_efficiency``)
is applied stage-wise to the operating-line/pseudo-equilibrium construction.
``minimum_reflux`` and friends use the equilibrium (E = 1) curve.

References
----------
- Wankat. Separation Process Engineering, Ch. 4 and 12.
- McCabe, Smith, Harriott. Unit Operations of Chemical Engineering, Ch. 21.
- Kremser (1930); Fenske (1932); Lewis (1922) for the Murphree-to-overall
  efficiency relation. Algebra re-derived here and checked in
  ``tests/test_staged.py``; no textbook example numbers are claimed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array, lax

__all__ = [
    "EquilibriumCurve",
    "constant_alpha",
    "linear_equilibrium",
    "tabulated",
    "operating_line",
    "rectifying_line",
    "stripping_line",
    "stripping_line_from_boilup",
    "q_line",
    "line_through",
    "line_y",
    "line_slope_intercept",
    "intersect_lines",
    "feed_intersection",
    "to_ratio",
    "from_ratio",
    "mccabe_thiele",
    "minimum_reflux",
    "total_reflux_stages",
    "minimum_stages",
    "minimum_solvent_ratio",
    "kremser_fraction",
    "kremser_stages",
]

_TINY = 1e-12


# ---------------------------------------------------------------------------
# Lines
# ---------------------------------------------------------------------------


def line_y(line: Array, x: Array) -> Array:
    """Evaluate ``y = (c - a x)/b`` on a line ``(a, b, c)``."""
    line = jnp.asarray(line)
    return (line[2] - line[0] * x) / line[1]


def line_slope_intercept(line: Array) -> tuple[Array, Array]:
    """Slope ``-a/b`` and intercept ``c/b`` of a non-vertical line."""
    line = jnp.asarray(line)
    return -line[0] / line[1], line[2] / line[1]


def line_through(p1: tuple, p2: tuple) -> Array:
    """The line ``(a, b, c)`` through two points ``(x1, y1)`` and ``(x2, y2)``."""
    x1, y1 = p1
    x2, y2 = p2
    a = y2 - y1
    b = -(x2 - x1)
    return jnp.stack([jnp.asarray(a, float), jnp.asarray(b, float),
                      jnp.asarray(a * x1 + b * y1, float)])


def intersect_lines(l1: Array, l2: Array) -> tuple[Array, Array]:
    """Intersection ``(x, y)`` of two lines in ``(a, b, c)`` form."""
    l1 = jnp.asarray(l1)
    l2 = jnp.asarray(l2)
    det = l1[0] * l2[1] - l2[0] * l1[1]
    x = (l1[2] * l2[1] - l2[2] * l1[1]) / det
    y = (l1[0] * l2[2] - l2[0] * l1[2]) / det
    return x, y


def operating_line(L: Array | float, V: Array | float, x_out: Array | float,
                   y_in: Array | float) -> Array:
    """General operating line from a material balance, ``(a, b, c)``.

    ``L (x - x_out) = V (y - y_in)``: the liquid ``L`` leaves at composition
    ``x_out`` where the vapor ``V`` enters at ``y_in`` (the same line for
    absorption, stripping and extraction; use solute-free ratios for a
    straight line with dilute-basis flows).

    Returns:
        ``[L, -V, L x_out - V y_in]``; slope ``L/V``.
    """
    L = jnp.asarray(L, float)
    V = jnp.asarray(V, float)
    return jnp.stack([L, -V, L * x_out - V * y_in])


def rectifying_line(R: Array | float, x_D: Array | float) -> Array:
    """Rectifying line ``y = R/(R+1) x + x_D/(R+1)`` as ``(a, b, c)``."""
    R = jnp.asarray(R, float)
    x_D = jnp.asarray(x_D, float)
    return jnp.stack([R, -(R + 1.0), -x_D])


def q_line(q: Array | float, z: Array | float) -> Array:
    """Feed line ``q x - (q - 1) y = z`` as ``(a, b, c)``.

    Slope ``q/(q-1)``; vertical for ``q = 1`` (saturated liquid), horizontal
    for ``q = 0`` (saturated vapor). Subcooled liquid has ``q > 1``,
    superheated vapor ``q < 0``.
    """
    q = jnp.asarray(q, float)
    z = jnp.asarray(z, float)
    return jnp.stack([q, 1.0 - q, z])


def feed_intersection(q: Array | float, z: Array | float, R: Array | float,
                      x_D: Array | float) -> tuple[Array, Array]:
    """Where the q-line meets the rectifying line, ``(x, y)``."""
    return intersect_lines(q_line(q, z), rectifying_line(R, x_D))


def stripping_line(R: Array | float, x_D: Array | float, x_B: Array | float,
                   z: Array | float, q: Array | float,
                   open_steam: bool = False) -> Array:
    """Stripping line through the feed intersection and the bottoms point.

    With a reboiler the line passes through ``(x_B, x_B)``; with open steam
    (no reboiler) through ``(x_B, 0)``.
    """
    xf, yf = feed_intersection(q, z, R, x_D)
    x_B = jnp.asarray(x_B, float)
    y_B = jnp.zeros_like(x_B) if open_steam else x_B
    return line_through((x_B, y_B), (xf, yf))


def stripping_line_from_boilup(boilup: Array | float, x_B: Array | float) -> Array:
    """Stripping line from the boilup ratio ``V_B/B`` (reboiler vapor / bottoms).

    ``L_bar = V_B + B`` and ``V_bar = V_B``, so the line is
    ``(r + 1) x - r y = x_B`` with ``r = V_B/B``.
    """
    r = jnp.asarray(boilup, float)
    return jnp.stack([r + 1.0, -r, jnp.asarray(x_B, float)])


def to_ratio(x: Array | float) -> Array:
    """Solute-free ratio ``x/(1-x)`` (straight absorption operating lines)."""
    x = jnp.asarray(x, float)
    return x / (1.0 - x)


def from_ratio(X: Array | float) -> Array:
    """Inverse of :func:`to_ratio`, ``X/(1+X)``."""
    X = jnp.asarray(X, float)
    return X / (1.0 + X)


# ---------------------------------------------------------------------------
# Monotone interpolation (PCHIP) and implicit inversion
# ---------------------------------------------------------------------------


def _pchip_slopes(x: Array, y: Array) -> Array:
    """Fritsch-Carlson / SciPy PCHIP derivatives for increasing ``x``."""
    h = jnp.diff(x)
    delta = jnp.diff(y) / h
    n = x.shape[0]
    if n == 2:
        return jnp.stack([delta[0], delta[0]])
    h0, h1 = h[:-1], h[1:]
    d0, d1 = delta[:-1], delta[1:]
    w1 = 2.0 * h1 + h0
    w2 = h1 + 2.0 * h0
    same = d0 * d1 > 0.0
    den = jnp.where(same, w1 / jnp.where(same, d0, 1.0) + w2 / jnp.where(same, d1, 1.0), 1.0)
    interior = jnp.where(same, (w1 + w2) / den, 0.0)

    def end(hh0, hh1, dd0, dd1):
        d = ((2.0 * hh0 + hh1) * dd0 - hh0 * dd1) / (hh0 + hh1)
        d = jnp.where(jnp.sign(d) != jnp.sign(dd0), 0.0, d)
        d = jnp.where((jnp.sign(dd0) != jnp.sign(dd1)) & (jnp.abs(d) > 3.0 * jnp.abs(dd0)),
                      3.0 * dd0, d)
        return d

    first = end(h[0], h[1], delta[0], delta[1])
    last = end(h[-1], h[-2], delta[-1], delta[-2])
    return jnp.concatenate([first[None], interior, last[None]])


def _pchip_eval(x: Array, y: Array, d: Array, xq: Array) -> Array:
    """Hermite evaluation, with linear extrapolation at the end slopes."""
    n = x.shape[0]
    i = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, n - 2)
    x0, x1 = x[i], x[i + 1]
    h = x1 - x0
    t = (xq - x0) / h
    t2, t3 = t * t, t * t * t
    val = ((2 * t3 - 3 * t2 + 1) * y[i] + (t3 - 2 * t2 + t) * h * d[i]
           + (-2 * t3 + 3 * t2) * y[i + 1] + (t3 - t2) * h * d[i + 1])
    lo = y[0] + d[0] * (xq - x[0])
    hi = y[-1] + d[-1] * (xq - x[-1])
    return jnp.where(xq < x[0], lo, jnp.where(xq > x[-1], hi, val))


def _solve_monotone(f, lo, hi, n_iter: int = 60):
    """Root of a monotone scalar ``f`` on ``[lo, hi]`` with implicit gradients.

    Bisection on stop-gradient values, then one Newton step with a
    stop-gradient slope, so ``d root/d params = -(df/dp)/(df/dx)`` for any
    parameters ``f`` closes over.

    ``f`` may be NaN over one end of the bracket (outside its physical
    domain, e.g. humid air below its dew point): the direction of
    monotonicity is then read from the finite end alone, and a NaN at a
    midpoint counts as lying on the NaN end's side of the root. Without
    this, a NaN at ``lo`` flipped the bisection, which then walked to the
    wrong end and returned a Newton step from there: a nearly right value
    with a wrong gradient.
    """
    def g(x):
        return lax.stop_gradient(f(x))

    g_lo, g_hi = g(jnp.asarray(lo, float)), g(jnp.asarray(hi, float))
    fin_lo, fin_hi = jnp.isfinite(g_lo), jnp.isfinite(g_hi)
    # With one end NaN, assume the root is in the bracket: the finite end's
    # sign then says which way f runs.
    increasing = jnp.where(fin_lo & fin_hi, g_hi >= g_lo,
                           jnp.where(fin_hi, g_hi > 0.0, g_lo < 0.0))
    nan_at_hi = fin_lo & ~fin_hi

    def body(_, carry):
        a, b = carry
        m = 0.5 * (a + b)
        gm = g(m)
        pos = gm > 0.0
        go_left = jnp.where(increasing, pos, ~pos)
        go_left = jnp.where(jnp.isfinite(gm), go_left, nan_at_hi)
        return jnp.where(go_left, a, m), jnp.where(go_left, m, b)

    a, b = lax.fori_loop(0, n_iter, body, (jnp.asarray(lo, float), jnp.asarray(hi, float)))
    x0 = lax.stop_gradient(0.5 * (a + b))
    slope = lax.stop_gradient(jax.grad(f)(x0))
    slope = jnp.where(jnp.abs(slope) > _TINY, slope, 1.0)
    return x0 - f(x0) / slope


# ---------------------------------------------------------------------------
# Equilibrium curve
# ---------------------------------------------------------------------------


@jax.tree_util.register_pytree_node_class
class EquilibriumCurve:
    """An equilibrium curve ``y*(x)`` with its inverse, as a JAX pytree.

    Build one with :func:`constant_alpha`, :func:`linear_equilibrium` or
    :func:`tabulated`; ``curve(x)`` is ``y*(x)``, ``curve.inverse(y)`` is
    ``x*(y)``. ``curve.with_efficiency(E)`` returns a copy carrying a
    Murphree vapor efficiency, applied by :func:`mccabe_thiele` as the
    pseudo-equilibrium ``y = y_op + E (y* - y_op)``.
    """

    def __init__(self, kind: str, data: dict, efficiency=None):
        self.kind = kind
        self.data = data
        self.efficiency = efficiency

    def tree_flatten(self):
        return (self.data, self.efficiency), self.kind

    @classmethod
    def tree_unflatten(cls, kind, children):
        return cls(kind, children[0], children[1])

    def __call__(self, x: Array | float) -> Array:
        """Equilibrium vapor composition ``y*(x)``."""
        x = jnp.asarray(x, float)
        if self.kind == "alpha":
            a = self.data["alpha"]
            return a * x / (1.0 + (a - 1.0) * x)
        if self.kind == "linear":
            return self.data["m"] * x
        d = self.data
        return _pchip_eval(d["x"], d["y"], d["dydx"], x)

    def inverse(self, y: Array | float) -> Array:
        """Equilibrium liquid composition ``x*(y)``."""
        y = jnp.asarray(y, float)
        if self.kind == "alpha":
            a = self.data["alpha"]
            return y / (a - (a - 1.0) * y)
        if self.kind == "linear":
            return y / self.data["m"]
        d = self.data
        return _pchip_eval(d["y"], d["x"], d["dxdy"], y)

    def with_efficiency(self, E: Array | float) -> "EquilibriumCurve":
        """Copy with a Murphree vapor efficiency ``E`` (``E = 1`` is ideal)."""
        return EquilibriumCurve(self.kind, self.data, jnp.asarray(E, float))

    def ideal(self) -> "EquilibriumCurve":
        """Copy without the efficiency."""
        return EquilibriumCurve(self.kind, self.data, None)


def constant_alpha(alpha: Array | float) -> EquilibriumCurve:
    """``y = alpha x / (1 + (alpha - 1) x)`` (constant relative volatility)."""
    return EquilibriumCurve("alpha", {"alpha": jnp.asarray(alpha, float)})


def linear_equilibrium(m: Array | float) -> EquilibriumCurve:
    """``y = m x`` (Henry's law or a constant distribution coefficient)."""
    return EquilibriumCurve("linear", {"m": jnp.asarray(m, float)})


def tabulated(x: Array, y: Array) -> EquilibriumCurve:
    """Curve from data ``(x, y)`` with monotone (PCHIP) interpolation.

    ``x`` and ``y`` must both be strictly increasing (true for a binary VLE
    curve from :func:`difflow.phase_diagrams.xy_curve`); both the curve and
    its inverse are C1 and monotone, so stepping has no gradient kinks from
    the interpolation. Outside the table the end slopes extrapolate linearly.
    """
    x = jnp.asarray(x, float)
    y = jnp.asarray(y, float)
    return EquilibriumCurve("table", {"x": x, "y": y, "dydx": _pchip_slopes(x, y),
                                      "dxdy": _pchip_slopes(y, x)})


# ---------------------------------------------------------------------------
# Kremser
# ---------------------------------------------------------------------------


def kremser_fraction(A: Array | float, N: Array | float) -> Array:
    """Fraction of solute recovered by ``N`` counter-current stages.

    ``(A^(N+1) - A)/(A^(N+1) - 1)`` for a pure inlet solvent, with ``A`` the
    absorption (or extraction) factor ``L/(m V)``; valid for any ``A > 0``
    (for ``A < 1`` the recovery tends to ``A``, not 1). The ``A -> 1`` limit
    ``N/(N+1)`` is a Taylor blend inside ``|ln A| < 1e-8``, where both
    branches are differentiable.
    """
    A = jnp.asarray(A, float)
    N = jnp.asarray(N, float)
    ell = jnp.log(A)
    near = jnp.abs(ell) < 1e-8
    m = jnp.where(near, 1.0, jnp.abs(ell))
    g = jnp.expm1(-N * m) / jnp.expm1(-(N + 1.0) * m)
    far = g * jnp.where(ell < 0.0, A, 1.0)
    series = N / (N + 1.0) * (1.0 + 0.5 * ell)
    return jnp.where(near, series, far)


def kremser_stages(A: Array | float, fraction: Array | float) -> Array:
    """Stages for a recovery ``fraction`` at absorption factor ``A``.

    ``N = ln((A - f)/(1 - f))/ln A - 1``, for any ``A > 0`` (no floor on
    ``ln A``: ``A < 1`` works as long as ``f < A``; otherwise ``inf``). The
    ``A -> 1`` limit ``f/(1-f)`` is a second-order Taylor blend inside
    ``|A - 1| < 1e-9``. Not floored at one stage.
    """
    A = jnp.asarray(A, float)
    f = jnp.asarray(fraction, float)
    a = 1.0 / (1.0 - f)
    eps = A - 1.0
    near = jnp.abs(eps) < 1e-9
    e = jnp.where(near, 0.5, eps)
    reachable = (1.0 + a * e) > 0.0
    ratio = jnp.log1p(a * jnp.where(reachable, e, 0.5)) / jnp.log1p(e)
    series = a * (1.0 + eps * (1.0 - a) / 2.0
                  + eps**2 * (a**2 / 3.0 - a / 4.0 - 1.0 / 12.0))
    n = jnp.where(near, series, ratio) - 1.0
    return jnp.where(near | reachable, n, jnp.inf)


# ---------------------------------------------------------------------------
# Stepper
# ---------------------------------------------------------------------------


def _taylor_log_fraction(u: Array, eps: Array) -> Array:
    """``ln(1 + u eps)/ln(1 + eps)`` for ``u`` in [0, 1], smooth at eps = 0."""
    near = jnp.abs(eps) < 1e-9
    e = jnp.where(near, 0.5, eps)
    exact = jnp.log1p(u * e) / jnp.log1p(e)
    series = u * (1.0 + eps * (1.0 - u) / 2.0
                  + eps**2 * (u**2 / 3.0 - u / 4.0 - 1.0 / 12.0))
    return jnp.where(near, series, exact)


def _credits(xs: Array, x_end: Array, sgn: Array, fraction: str) -> Array:
    """Stage credits ``c_k`` (k = 1..n) from the compositions ``x_0..x_n``."""
    if fraction == "log_ratio":
        def T(x):
            return x / (1.0 - jnp.clip(x, -1e6, 1.0 - 1e-12))
    else:
        def T(x):
            return x
    u = sgn * T(xs)
    u_end = sgn * T(x_end)
    up = u[:-1]
    uk = u[1:]
    upp = jnp.concatenate([2.0 * u[:1] - u[1:2], u[:-2]])
    d = u_end - up  # distance still to go before stage k
    step = uk - up
    ok = step > _TINY
    step_s = jnp.where(ok, step, 1.0)
    if fraction == "linear":
        frac = jnp.clip(d / step_s, 0.0, 1.0)
    else:
        prev = up - upp
        r = jnp.where(prev > _TINY, step_s / jnp.where(prev > _TINY, prev, 1.0), 1.0)
        r = jnp.clip(r, 1e-8, 1e8)
        dd = jnp.clip(d, 0.0, step_s)
        frac = _taylor_log_fraction(dd / step_s, r - 1.0)
    return jnp.where(d > 0.0, jnp.where(ok, frac, 1.0), 0.0)


def mccabe_thiele(
    curve: EquilibriumCurve,
    top_line: Array,
    bottom_line: Array | None,
    x_top: Array | float,
    x_bottom: Array | float,
    q_line: Array | None = None,
    n_max: int = 100,
    mode: str = "optimal_feed",
    feed_stage: Array | float | None = None,
    feed_width: float = 2e-4,
    fraction: str = "linear",
    partial_condenser: bool = False,
    reboiler_equilibrium: bool = False,
    lines: list | None = None,
    switch_x: list | None = None,
    switch_stages: list | None = None,
) -> dict:
    """Smooth McCabe-Thiele stepping with a fractional, differentiable ``N``.

    Starts at ``x_0 = x_top`` and repeats ``y_k = line(x_{k-1})``,
    ``x_k = x*(y_k)`` for ``n_max`` steps with ``lax.scan``; steps past
    ``x_bottom`` are masked (see the module docstring for the fractional
    convention). Works for descending ``x`` (distillation, ``x_top >
    x_bottom``) and ascending ``x`` (absorption with ``x_top`` the lean
    liquid). For distillation use the rectifying line as ``top_line`` and the
    stripping line as ``bottom_line``.

    Args:
        curve: Equilibrium curve; a Murphree efficiency on it is honoured.
        top_line: Operating line ``(a, b, c)`` used from the top.
        bottom_line: Second operating line, or ``None`` for a single-section
            cascade (absorber, stripper, extractor).
        x_top: Liquid composition at the top (``x_D`` for distillation).
        x_bottom: Target liquid composition at the bottom (``x_B``).
        q_line: Feed line; with ``mode="optimal_feed"`` the cascade switches
            to ``bottom_line`` once ``x`` passes the line intersection.
        n_max: Number of scan steps (also the cap on ``N``).
        mode: ``"optimal_feed"`` (needs ``q_line``) or ``"specified"``
            (needs ``feed_stage``: stages ``k >= feed_stage`` use the bottom
            line, a fractional value blends linearly).
        feed_stage: Feed stage number for ``mode="specified"``.
        feed_width: Sigmoid width (in ``x``) of the optimal-feed switch.
        fraction: ``"linear"``, ``"log"`` or ``"log_ratio"``.
        partial_condenser: Report the condenser as one of the ``N`` stages
            (the staircase is unchanged; ``N_column = N - 1``).
        reboiler_equilibrium: With a Murphree efficiency on ``curve``, treat
            the last stage (the reboiler) as an equilibrium stage, the usual
            textbook convention and the one :class:`DistillationColumn`
            uses: the trays step to the pseudo-equilibrium curve and the
            first step whose *equilibrium* drop reaches ``x_bottom`` is drawn
            to the true curve and ends the cascade. Ignored without an
            efficiency. ``N`` can jump by a fraction of a stage where the
            reboiler stage changes index (the trays and the reboiler take
            different-sized steps), so prefer the default for gradients
            across that point.
        lines: Any number of operating lines ``[(a, b, c), ...]`` from the
            top down, for columns with side draws, side feeds or several
            feeds (one line per section). Pass ``top_line=None`` and
            ``bottom_line=None`` with it. The switch from line ``j`` to line
            ``j + 1`` is given by ``switch_x[j]`` (the composition where it
            happens, usually the intersection of the two lines or of a
            q-line; a sigmoid of width ``feed_width``, as the
            ``optimal_feed`` mode) or by ``switch_stages[j]`` (a stage
            number, as the ``specified`` mode). ``mode``, ``q_line``,
            ``feed_stage`` and ``reboiler_equilibrium`` are then ignored.
        switch_x: Switch compositions, one per pair of adjacent lines.
        switch_stages: Switch stage numbers, one per pair of adjacent lines.

    Returns:
        dict with ``N`` (fractional equilibrium stages stepped, including
        the reboiler and, with a partial condenser, the condenser),
        ``N_column`` (``N`` less the condenser if partial), ``N_rect`` (stages
        above the feed stage), ``N_strip`` (the rest), ``feed_stage``
        (fractional; ``nan`` without a bottom line), ``x`` (``x_0..x_n``),
        ``y`` (``y_1..y_n``), ``credit`` (``c_1..c_n``), ``x_vertices`` /
        ``y_vertices`` (staircase for plotting; trim with ``credit > 0``),
        ``x_feed`` (the switch composition) and ``pinched`` (soft flag in
        [0, 1]: the cascade stalled short of ``x_bottom``). With ``lines``,
        also ``switch_stage`` (fractional stage number of each switch, the
        stage on which line ``j + 1`` first applies) and ``N_sections``
        (stages stepped on each line); ``feed_stage`` is the first switch.
    """
    if lines is not None:
        return _mccabe_thiele_sections(
            curve, lines, x_top, x_bottom, switch_x, switch_stages, n_max,
            feed_width, fraction, partial_condenser)
    if fraction not in ("linear", "log", "log_ratio"):
        raise ValueError("fraction must be 'linear', 'log' or 'log_ratio'")
    if mode not in ("optimal_feed", "specified"):
        raise ValueError("mode must be 'optimal_feed' or 'specified'")
    top_line = jnp.asarray(top_line, float)
    x_top = jnp.asarray(x_top, float)
    x_bottom = jnp.asarray(x_bottom, float)
    sgn = jnp.where(x_bottom >= x_top, 1.0, -1.0)
    two = bottom_line is not None
    if two:
        bottom_line = jnp.asarray(bottom_line, float)
        if mode == "optimal_feed":
            if q_line is None:
                raise ValueError("mode='optimal_feed' needs q_line")
            x_f, _ = intersect_lines(jnp.asarray(q_line, float), top_line)
        else:
            if feed_stage is None:
                raise ValueError("mode='specified' needs feed_stage")
            x_f = jnp.asarray(jnp.nan)
        fs = None if feed_stage is None else jnp.asarray(feed_stage, float)
    else:
        x_f = jnp.asarray(jnp.nan)
        fs = None

    def weight(x, k):
        """Weight of the bottom line for the drop from stage ``k`` (x = x_k)."""
        if not two:
            return jnp.zeros_like(x)
        if mode == "optimal_feed":
            return jax.nn.sigmoid(sgn * (x - x_f) / feed_width)
        return jnp.clip(k - fs + 1.0, 0.0, 1.0)

    def op_y(x, k):
        w = weight(x, k)
        yt = line_y(top_line, x)
        if not two:
            return yt
        return (1.0 - w) * yt + w * line_y(bottom_line, x)

    E = curve.efficiency
    span = jnp.maximum(jnp.abs(x_top), jnp.abs(x_bottom))
    lo = jnp.zeros(())
    hi = jnp.maximum(1.0, 4.0 * span)

    def next_x(y_k, k):
        if E is None:
            return curve.inverse(y_k)
        ideal = curve.ideal()

        def f(x):
            yo = op_y(x, k)
            return yo + E * (ideal(x) - yo) - y_k

        return _solve_monotone(f, lo, hi)

    def step(x_prev, k):
        y_k = op_y(x_prev, k - 1.0)
        x_k = next_x(y_k, k)
        return x_k, (x_k, y_k)

    ks = jnp.arange(1.0, n_max + 1.0)
    _, (xk, yk) = lax.scan(step, x_top, ks)
    xs = jnp.concatenate([x_top[None], xk])
    if reboiler_equilibrium and E is not None:
        # equilibrium drop from each y_k; the first one reaching x_bottom is
        # the reboiler, and the cascade holds there afterwards
        x_eq = curve.ideal().inverse(yk)
        reach = lax.stop_gradient(sgn * (x_eq - x_bottom) >= 0.0)
        found = jnp.any(reach)
        K = jnp.argmax(reach)  # 0-based index into yk / xk
        idx = jnp.arange(n_max)
        x_reb = x_eq[K]
        xk_mod = jnp.where(idx < K, xk, x_reb)
        xs = jnp.where(found, jnp.concatenate([x_top[None], xk_mod]), xs)
    c = _credits(xs, x_bottom, sgn, fraction)
    N = jnp.sum(c)
    if two:
        w_k = jax.vmap(weight)(xs[1:], ks)
        feed = 1.0 + jnp.sum(c * (1.0 - w_k))
    else:
        feed = jnp.asarray(jnp.nan)
    N_rect = feed - 1.0
    # staircase: (x_{k-1}, y_k) -> (x_k, y_k) -> (x_k, y_{k+1})
    y_next = jax.vmap(op_y)(xs[1:], ks)
    xv = jnp.stack([xs[:-1], xs[1:], xs[1:]], axis=1).reshape(-1)
    yv = jnp.stack([yk, yk, y_next], axis=1).reshape(-1)
    rem = jnp.maximum(0.0, sgn * (x_bottom - xs[-1])) / jnp.maximum(jnp.abs(x_bottom - x_top), _TINY)
    return {
        "N": N,
        "N_column": N - 1.0 if partial_condenser else N,
        "N_rect": N_rect,
        "N_strip": N - N_rect,
        "feed_stage": feed,
        "x": xs,
        "y": yk,
        "credit": c,
        "x_vertices": xv,
        "y_vertices": yv,
        "x_feed": x_f,
        "pinched": jnp.tanh(rem / 1e-3),
    }


def _mccabe_thiele_sections(curve, lines, x_top, x_bottom, switch_x,
                            switch_stages, n_max, feed_width, fraction,
                            partial_condenser):
    """:func:`mccabe_thiele` with a list of operating lines (side streams)."""
    lines = [jnp.asarray(L, float) for L in lines]
    m = len(lines) - 1
    if m < 1:
        raise ValueError("lines needs at least two operating lines")
    if (switch_x is None) == (switch_stages is None):
        raise ValueError("give exactly one of switch_x or switch_stages")
    sw = switch_x if switch_x is not None else switch_stages
    if len(sw) != m:
        raise ValueError(f"{len(lines)} lines need {m} switches, got {len(sw)}")
    sw = [jnp.asarray(v, float) for v in sw]
    x_top = jnp.asarray(x_top, float)
    x_bottom = jnp.asarray(x_bottom, float)
    sgn = jnp.where(x_bottom >= x_top, 1.0, -1.0)

    def weights(x, k):
        """Weight of line j+1 over line j for the drop from stage ``k``."""
        if switch_x is not None:
            return jnp.stack([jax.nn.sigmoid(sgn * (x - s_) / feed_width) for s_ in sw])
        return jnp.stack([jnp.clip(k - s_ + 1.0, 0.0, 1.0) for s_ in sw])

    def op_y(x, k):
        y = line_y(lines[0], x)
        for w, L in zip(weights(x, k), lines[1:]):
            y = (1.0 - w) * y + w * line_y(L, x)
        return y

    E = curve.efficiency
    span = jnp.maximum(jnp.abs(x_top), jnp.abs(x_bottom))
    lo = jnp.zeros(())
    hi = jnp.maximum(1.0, 4.0 * span)

    def next_x(y_k, k):
        if E is None:
            return curve.inverse(y_k)
        ideal = curve.ideal()

        def f(x):
            yo = op_y(x, k)
            return yo + E * (ideal(x) - yo) - y_k

        return _solve_monotone(f, lo, hi)

    def step(x_prev, k):
        y_k = op_y(x_prev, k - 1.0)
        x_k = next_x(y_k, k)
        return x_k, (x_k, y_k)

    ks = jnp.arange(1.0, n_max + 1.0)
    _, (xk, yk) = lax.scan(step, x_top, ks)
    xs = jnp.concatenate([x_top[None], xk])
    c = _credits(xs, x_bottom, sgn, fraction)
    N = jnp.sum(c)
    W = jax.vmap(weights)(xs[1:], ks)                      # (n_max, m)
    switch_stage = 1.0 + jnp.sum(c[:, None] * (1.0 - W), axis=0)
    bounds = jnp.concatenate([jnp.ones(1), switch_stage, (N + 1.0)[None]])
    y_next = jax.vmap(op_y)(xs[1:], ks)
    xv = jnp.stack([xs[:-1], xs[1:], xs[1:]], axis=1).reshape(-1)
    yv = jnp.stack([yk, yk, y_next], axis=1).reshape(-1)
    rem = jnp.maximum(0.0, sgn * (x_bottom - xs[-1])) / jnp.maximum(jnp.abs(x_bottom - x_top), _TINY)
    return {
        "N": N,
        "N_column": N - 1.0 if partial_condenser else N,
        "N_rect": switch_stage[0] - 1.0,
        "N_strip": N - switch_stage[0] + 1.0,
        "feed_stage": switch_stage[0],
        "switch_stage": switch_stage,
        "N_sections": jnp.diff(bounds),
        "x": xs,
        "y": yk,
        "credit": c,
        "x_vertices": xv,
        "y_vertices": yv,
        "x_feed": sw[0] if switch_x is not None else jnp.asarray(jnp.nan),
        "pinched": jnp.tanh(rem / 1e-3),
    }


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def _extremum_on_interval(f, lo, hi, maximize: bool, n_grid: int = 257, n_iter: int = 60):
    """Value of ``f`` at its extremum over ``[lo, hi]``, envelope-differentiable.

    A grid finds the best cell, a ternary search on stop-gradient values
    refines it, and the endpoints are compared. ``f`` is then evaluated at
    ``x = lo + t* (hi - lo)`` with ``t*`` fixed, so derivatives follow the
    interior (envelope) and endpoint (through ``lo``/``hi``) cases.
    """
    sign = 1.0 if maximize else -1.0

    def F(t):
        return sign * f(lax.stop_gradient(lo) + t * lax.stop_gradient(hi - lo))

    ts = jnp.linspace(0.0, 1.0, n_grid)
    vals = lax.stop_gradient(jax.vmap(F)(ts))
    k = jnp.argmax(vals)
    a = ts[jnp.maximum(k - 1, 0)]
    b = ts[jnp.minimum(k + 1, n_grid - 1)]

    def body(_, ab):
        a, b = ab
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        left = lax.stop_gradient(F(m1)) < lax.stop_gradient(F(m2))
        return jnp.where(left, m1, a), jnp.where(left, b, m2)

    a, b = lax.fori_loop(0, n_iter, body, (a, b))
    t_mid = 0.5 * (a + b)
    cands = jnp.stack([jnp.zeros(()), t_mid, jnp.ones(())])
    cv = lax.stop_gradient(jax.vmap(F)(cands))
    t_star = lax.stop_gradient(cands[jnp.argmax(cv)])
    return f(lo + t_star * (hi - lo))


def _q_curve_intersection(curve: EquilibriumCurve, qline: Array, z: Array) -> Array:
    """``x`` where the q-line meets the curve (bisection plus implicit step)."""
    a, b, c = qline[0], qline[1], qline[2]
    q = a
    lo = jnp.where(q > 1.0, z, 0.0)
    hi = jnp.where(q > 1.0, 1.0, z)

    def f(x):
        return a * x + b * curve(x) - c

    return _solve_monotone(f, lo, hi)


def minimum_reflux(curve: EquilibriumCurve, z: Array | float, q: Array | float,
                   x_D: Array | float) -> Array:
    """Minimum reflux ratio from the pinch, feed-line or tangent.

    The rectifying line through ``(x_D, x_D)`` must stay on or below the
    curve for ``x`` between the q-line/curve intersection ``x_q`` and
    ``x_D``, so its smallest slope is the largest chord slope
    ``(x_D - y*(x))/(x_D - x)`` over that interval; ``R_min = s/(1 - s)``.
    A tangent pinch (ethanol-water) shows up as an interior maximiser, a
    feed-line pinch as the endpoint ``x_q``. Not floored: a value below zero
    means ``x_D`` is unreachable on this curve (no pinch constrains it).

    Args:
        curve: Equilibrium curve (an attached efficiency is ignored).
        z: Feed composition.
        q: Feed quality (1 saturated liquid, 0 saturated vapor).
        x_D: Distillate composition.

    Returns:
        ``R_min`` as a JAX scalar; differentiable in ``z``, ``q``, ``x_D``
        and the curve parameters.
    """
    curve = curve.ideal()
    z = jnp.asarray(z, float)
    q = jnp.asarray(q, float)
    x_D = jnp.asarray(x_D, float)
    x_q = _q_curve_intersection(curve, q_line(q, z), z)
    hi = x_D - 1e-6 * jnp.maximum(x_D - x_q, 1e-3)

    def phi(x):
        return (x_D - curve(x)) / (x_D - x)

    s = _extremum_on_interval(phi, x_q, hi, True)
    return s / (1.0 - s)


def total_reflux_stages(curve: EquilibriumCurve, x_D: Array | float,
                        x_B: Array | float, n_max: int = 200,
                        fraction: str = "auto") -> Array:
    """Stages at total reflux: stepping between the curve and the diagonal.

    Args:
        curve: Equilibrium curve.
        x_D, x_B: Distillate and bottoms compositions.
        n_max: Scan length.
        fraction: Last-stage convention (see :func:`mccabe_thiele`);
            ``"auto"`` is ``"log_ratio"`` for a constant-alpha curve, where
            the result equals Fenske's expression exactly, else ``"linear"``.

    Returns:
        Fractional minimum stage count (including the reboiler).
    """
    if fraction == "auto":
        fraction = "log_ratio" if curve.kind == "alpha" else "linear"
    diag = jnp.array([1.0, -1.0, 0.0])
    res = mccabe_thiele(curve.ideal(), diag, None, x_D, x_B, n_max=n_max,
                        fraction=fraction)
    return res["N"]


def minimum_stages(alpha_or_curve, x_D: Array | float, x_B: Array | float) -> Array:
    """Fenske minimum stages: closed form for a constant ``alpha``.

    Given a number, ``ln[(x_D/(1-x_D)) ((1-x_B)/x_B)]/ln(alpha)``; given an
    :class:`EquilibriumCurve`, :func:`total_reflux_stages`.
    """
    if isinstance(alpha_or_curve, EquilibriumCurve):
        return total_reflux_stages(alpha_or_curve, x_D, x_B)
    x_D = jnp.asarray(x_D, float)
    x_B = jnp.asarray(x_B, float)
    return jnp.log(x_D / (1.0 - x_D) * (1.0 - x_B) / x_B) / jnp.log(
        jnp.asarray(alpha_or_curve, float))


def minimum_solvent_ratio(curve: EquilibriumCurve, x_in: Array | float,
                          x_out: Array | float, y_in: Array | float) -> Array:
    """Minimum ``S/F`` (flow of the ``y`` phase per flow of the ``x`` phase).

    Cascade: phase ``x`` enters at ``x_in`` and leaves at ``x_out``; the
    ``y`` phase enters at ``y_in`` (at the ``x_out`` end) and leaves rich.
    The operating line through ``(x_out, y_in)`` has slope ``F/S`` and must
    stay on or below the curve, so its largest slope is the smallest chord
    slope ``(y*(x) - y_in)/(x - x_out)`` over ``(x_out, x_in]``; ``S/F_min`` is
    its inverse. A pinch at the rich end gives
    ``(x_in - x_out)/(y*(x_in) - y_in)``; an interior minimum is a tangent
    pinch. Generalises the straight-equilibrium
    ``difflow_ree.equilibrium.minimum_solvent_ratio``.
    """
    curve = curve.ideal()
    x_in = jnp.asarray(x_in, float)
    x_out = jnp.asarray(x_out, float)
    y_in = jnp.asarray(y_in, float)
    lo = x_out + 1e-6 * jnp.maximum(x_in - x_out, 1e-3)

    def psi(x):
        return (curve(x) - y_in) / (x - x_out)

    slope = _extremum_on_interval(psi, lo, x_in, False)
    return 1.0 / slope
