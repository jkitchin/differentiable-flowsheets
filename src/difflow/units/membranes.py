"""Membrane separations: gas permeation, reverse osmosis and ultrafiltration.

Two palette units (``GasPermeator``, ``ReverseOsmosis``) and the pure-JAX
library functions behind them (SI units, differentiable). The carbon-capture
plugin's :class:`~difflow_cc.units.membrane.MembraneSeparator` (membrane
materials from a database, CO2-specific reporting) and the bio plugin's
ultrafiltration units build on the same functions.

Gas permeation (solution-diffusion)
-----------------------------------
The flux of species ``i`` through a dense membrane is

    J_i = Q_i (x_i P_h - y_i P_l)                    (mol/(m^2 s))

with ``Q_i = P_i/delta`` the permeance (permeability over thickness),
``x_i`` the feed-side (retentate) and ``y_i`` the permeate-side mole
fraction, ``P_h`` and ``P_l`` the feed and permeate pressures. The ideal
selectivity is ``alpha = Q_A/Q_B``, the **stage cut** ``theta`` is the
fraction of the feed that permeates, and the textbook pressure ratio is
``p_r = P_l/P_h`` (Geankoplis 5e §13.4; Wankat 4e Ch. 17; Seader 3e §14.3).
The unit takes ``pressure_ratio = P_h/P_l`` (> 1), as the carbon-capture
plugin does.

Four flow patterns are solved for any number of species:

* ``"complete_mixing"``: both sides well mixed, so the retentate and the
  permeate compositions apply over the whole membrane. Solved exactly by a
  bracketed scalar root (:func:`complete_mixing`).
* ``"cross_flow"``: feed in plug flow, permeate leaving each element
  without mixing along the membrane (the local permeate composition
  ``y_i = J_i / sum J`` is set by the local ``x``). Integrated along the area
  with RK4 (:func:`cross_flow`).
* ``"cocurrent"`` / ``"countercurrent"``: both sides in plug flow, the
  driving force using the bulk permeate composition. Cocurrent is an
  initial-value problem; countercurrent is solved by shooting from the
  retentate end, where the permeate flow is zero and its composition is the
  local one, with Newton's method on the retentate flows
  (:func:`countercurrent`).

For a binary, the complete-mixing permeate composition has the closed form
of Geankoplis 5e §13.4 (the quadratic in
:func:`binary_permeate_composition`); the same expression evaluated at the
local ``x`` is the cross-flow local permeate.

Reverse osmosis
---------------
Water flux ``J_v = A_w (Delta P - Delta pi)`` (m/s), solute flux
``J_s = B (c_w - c_p)``, film-theory polarization
``(c_w - c_p)/(c_b - c_p) = exp(J_v/k)``, and the permeate concentration
``c_p = J_s/J_v``. :func:`ro_local_flux` solves the three together;
:class:`ReverseOsmosis` integrates them along an element. Osmotic pressure is
van't Hoff's ``pi = phi i c R T`` (:func:`osmotic_pressure`).

Ultrafiltration
---------------
Gel-polarization flux ``J = k ln(c_g/c_b)`` (:func:`gel_polarization_flux`),
the flux against transmembrane pressure capped by it (:func:`uf_flux`), the
polarization modulus (:func:`polarization_modulus`) and the time of a batch
concentration (:func:`batch_concentration_time`).
"""

from __future__ import annotations

__all__ = [
    "BARRER",
    "GPU",
    "permeance_from_permeability",
    "binary_permeate_composition",
    "binary_reject_composition",
    "binary_complete_mixing",
    "complete_mixing_area",
    "local_permeate_composition",
    "complete_mixing",
    "cross_flow",
    "cocurrent",
    "countercurrent",
    "permeation_stage",
    "area_for_stage_cut",
    "GasPermeationParams",
    "GasPermeator",
    "osmotic_pressure",
    "ro_local_flux",
    "ReverseOsmosisParams",
    "ReverseOsmosis",
    "polarization_modulus",
    "gel_polarization_flux",
    "uf_flux",
    "batch_concentration_time",
]

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, get_flows, make_stream

R_GAS = 8.314462618  # J/(mol K)

#: 1 Barrer = 1e-10 cm^3(STP) cm/(cm^2 s cmHg) in mol m/(m^2 s Pa)
#: (1 cm^3(STP) = 1/22414 mol, 1 cmHg = 1333.22 Pa).
BARRER = 1e-10 / 22414.0 * 1e-2 / (1e-4 * 1333.224)
#: 1 GPU = 1e-6 cm^3(STP)/(cm^2 s cmHg) in mol/(m^2 s Pa).
GPU = 1e-6 / 22414.0 / (1e-4 * 1333.224)

FLOW_PATTERNS = ("complete_mixing", "cross_flow", "cocurrent", "countercurrent")


def permeance_from_permeability(permeability_barrer, thickness_um):
    """Permeance ``Q = P/delta`` in mol/(m^2 s Pa).

    Args:
        permeability_barrer: Permeability (Barrer).
        thickness_um: Selective-layer thickness (micrometres).

    Returns:
        Permeance, mol/(m^2 s Pa).

    Example:
        >>> round(float(permeance_from_permeability(100.0, 1.0)) / GPU, 6)
        100.0
    """
    return jnp.asarray(permeability_barrer) * BARRER / (jnp.asarray(thickness_um) * 1e-6)


# =============================================================================
# Scalar root helpers (bracketed bisection, implicit-function gradient)
# =============================================================================

def _bisect_increasing(fn, lo, hi, n_iter=100):
    """Root of ``fn`` (negative left, positive right of it) by bisection.

    Returns a stop-gradient value; pair with :func:`_implicit_step`.
    """
    def body(_, b):
        a, c = b
        m = 0.5 * (a + c)
        neg = fn(m) < 0.0
        return jnp.where(neg, m, a), jnp.where(neg, c, m)

    a, c = jax.lax.fori_loop(0, n_iter, body,
                             (jnp.asarray(lo, float), jnp.asarray(hi, float)))
    return jax.lax.stop_gradient(0.5 * (a + c))


def _illinois_increasing(fn, lo, hi, n_iter=40, tol=1e-13):
    """Root of an increasing ``fn`` bracketed by ``[lo, hi]`` by the Illinois method.

    For an ``fn`` too expensive for plain bisection's ~40 evaluations (each a countercurrent
    shooting solve): false position, with the retained end's value halved whenever the same end
    is kept twice, converges superlinearly and never leaves the bracket. It needs no derivative,
    so it compiles no larger than bisection. Stops when ``|fn| < tol`` or the bracket is narrower
    than ``tol``. Returns a stop-gradient value; pair with an implicit-function step.
    """
    # The first two passes evaluate the bracket ends (as if each replaced itself), so fn is traced
    # once, inside the loop.
    def body(state):
        k, a, fa, c, fc, side, _, xb, fb = state
        x = (a * fc - c * fa) / (fc - fa)
        x = jnp.where(jnp.isfinite(x) & (x > a) & (x < c), x, 0.5 * (a + c))
        x = jnp.where(k == 0, a, jnp.where(k == 1, c, x))
        fx = fn(x)
        left = jnp.where(k == 0, True, jnp.where(k == 1, False, fx < 0.0))  # x replaces a
        # Illinois: halve the value at the end kept for the second time running
        fc = jnp.where(left & (side == 1), 0.5 * fc, fc)
        fa = jnp.where(~left & (side == -1), 0.5 * fa, fa)
        a, fa = jnp.where(left, x, a), jnp.where(left, fx, fa)
        c, fc = jnp.where(left, c, x), jnp.where(left, fc, fx)
        side = jnp.where(k < 2, 0, jnp.where(left, 1, -1))
        # Near the root XLA may evaluate fn's round-off-level sign differently in the copies that
        # feed different selects, so the bracket can be updated inconsistently there: return the
        # best point seen rather than reading it back from the bracket.
        best = jnp.abs(fx) < jnp.abs(fb)
        xb, fb = jnp.where(best, x, xb), jnp.where(best, fx, fb)
        return k + 1, a, fa, c, fc, side, jnp.where(k < 2, jnp.inf, fx), xb, fb

    def unconverged(state):
        k, a, _, c, _, _, fx, _, _ = state
        return (k < n_iter + 2) & (jnp.abs(fx) > tol) & (c - a > tol)

    lo, hi = jnp.asarray(lo, float), jnp.asarray(hi, float)
    inf = jnp.asarray(jnp.inf)
    state = (0, lo, jnp.asarray(-1.0), hi, jnp.asarray(1.0), 0, inf, lo, inf)
    xb = jax.lax.while_loop(unconverged, body, state)[7]
    return jax.lax.stop_gradient(xb)


def _implicit_step(fn, u0):
    """Value-neutral Newton step from a stop-gradient root ``u0``.

    Gives the exact implicit-function first derivative of the root with
    respect to whatever ``fn`` closes over.
    """
    d = jax.lax.stop_gradient(jax.grad(fn)(u0))
    d = jnp.where(jnp.abs(d) > 1e-300, d, 1.0)
    r = fn(u0)
    return u0 - (r - jax.lax.stop_gradient(r)) / d


# =============================================================================
# Binary gas permeation, closed forms
# =============================================================================

def binary_permeate_composition(x, alpha, pr):
    """Permeate mole fraction in equilibrium with a feed-side ``x`` (binary).

    Solves ``y/(1-y) = alpha (x - pr y) / ((1-x) - pr (1-y))`` for ``y``:
    the complete-mixing permeate when ``x`` is the reject composition, and
    the local cross-flow permeate at any point. Written in the form that is
    stable at ``alpha = 1`` and ``pr = 0``.

    Args:
        x: Feed-side mole fraction of the faster species A.
        alpha: Ideal selectivity ``Q_A/Q_B``.
        pr: Pressure ratio ``P_l/P_h`` (< 1).

    Returns:
        Permeate mole fraction of A.

    Example:
        >>> round(float(binary_permeate_composition(0.25, 10.0, 0.25)), 4)
        0.6036
    """
    x, alpha, pr = jnp.asarray(x), jnp.asarray(alpha), jnp.asarray(pr)
    b = 1.0 + (alpha - 1.0) * (x + pr)
    disc = b * b - 4.0 * pr * (alpha - 1.0) * alpha * x
    return 2.0 * alpha * x / (b + jnp.sqrt(jnp.maximum(disc, 0.0)))


def binary_reject_composition(y, alpha, pr):
    """Feed-side mole fraction whose local permeate is ``y`` (binary).

    The inverse of :func:`binary_permeate_composition`, linear in ``x``.
    At ``y = x_f`` it is the **minimum reject composition** of a
    complete-mixing stage (the limit ``theta -> 1``).

    Args:
        y: Permeate mole fraction of A.
        alpha: Ideal selectivity.
        pr: Pressure ratio ``P_l/P_h``.

    Returns:
        Feed-side (reject) mole fraction of A.
    """
    y, alpha, pr = jnp.asarray(y), jnp.asarray(alpha), jnp.asarray(pr)
    r = y / (alpha * (1.0 - y))
    return (pr * y + r * (1.0 - pr * (1.0 - y))) / (1.0 + r)


def binary_complete_mixing(x_f, alpha, pr, x_o=None, theta=None):
    """Complete-mixing stage for a binary (Geankoplis 5e §13.4; Wankat 4e Ch. 17).

    Give either the reject composition ``x_o`` or the stage cut ``theta``.
    The balance ``x_f = theta y_p + (1 - theta) x_o`` closes the problem.

    Args:
        x_f: Feed mole fraction of the faster species.
        alpha: Ideal selectivity ``Q_A/Q_B``.
        pr: Pressure ratio ``P_l/P_h``.
        x_o: Reject (retentate) mole fraction.
        theta: Stage cut (permeate/feed).

    Returns:
        dict with ``y_p``, ``x_o``, ``theta``.
    """
    x_f = jnp.asarray(x_f, float)
    if (x_o is None) == (theta is None):
        raise ValueError("give exactly one of x_o and theta")
    if x_o is not None:
        x_o = jnp.asarray(x_o, float)
        y_p = binary_permeate_composition(x_o, alpha, pr)
        theta = (x_f - x_o) / (y_p - x_o)
    else:
        theta = jnp.asarray(theta, float)

        def g(x):
            return theta * binary_permeate_composition(x, alpha, pr) + (1.0 - theta) * x - x_f

        x0 = _bisect_increasing(g, 0.0, x_f)
        x_o = _implicit_step(g, x0)
        y_p = binary_permeate_composition(x_o, alpha, pr)
    return {"y_p": y_p, "x_o": x_o, "theta": theta}


def complete_mixing_area(feed_flow, theta, y_p, x_o, Q_A, P_h, P_l):
    """Membrane area of a complete-mixing stage from the flux of A.

    ``A = theta F y_p / (Q_A (P_h x_o - P_l y_p))``.

    Args:
        feed_flow: Feed molar flow (mol/s, or any flow unit consistent with
            ``Q_A``).
        theta: Stage cut.
        y_p: Permeate mole fraction of A.
        x_o: Reject mole fraction of A.
        Q_A: Permeance of A (flow/(area pressure)).
        P_h: Feed-side pressure.
        P_l: Permeate-side pressure.

    Returns:
        Area in the units implied by ``Q_A``.
    """
    return theta * feed_flow * y_p / (Q_A * (P_h * x_o - P_l * y_p))


# =============================================================================
# Multicomponent gas permeation
# =============================================================================

def _cm_permeate(t, F_i, Q, area, P_h, P_l):
    """Complete-mixing permeate fractions at retentate fraction ``t = 1 - theta``.

    The species balance ``(1-t) F y_i = A Q_i (x_i P_h - y_i P_l)`` with
    ``x_i = (F_i - (1-t) F y_i)/(t F)`` gives
    ``y_i = A Q_i z_i P_h / (F t (1-t) + A Q_i ((1-t) P_h + t P_l))``.
    """
    F = jnp.sum(F_i)
    z = F_i / F
    aq = area * Q
    denom = F * t * (1.0 - t) + aq * ((1.0 - t) * P_h + t * P_l)
    return aq * z * P_h / jnp.maximum(denom, 1e-300)


def _cm_retentate_fraction(F_i, Q, area, P_h, P_l):
    """Retentate fraction ``t`` of a complete-mixing stage of given area.

    ``g(t) = sum_i y_i(t) - 1`` is convex with ``g(0) = 0`` and at most one
    root in (0, 1), negative left of it and positive right.
    """
    def g(t):
        return jnp.sum(_cm_permeate(t, F_i, Q, area, P_h, P_l)) - 1.0

    t0 = _bisect_increasing(g, 0.0, 1.0)
    return jnp.clip(_implicit_step(g, t0), 0.0, 1.0)


def _cm_area_for_cut(F_i, Q, P_h, P_l, t):
    """Area at which a complete-mixing stage has retentate fraction ``t``."""
    F = jnp.sum(F_i)
    A_ref = F / (jnp.max(Q) * P_h)

    def h(s):
        A = A_ref * jnp.exp(s)
        return jnp.sum(_cm_permeate(t, F_i, Q, A, P_h, P_l)) - 1.0

    s0 = _bisect_increasing(h, -60.0, 60.0)
    return A_ref * jnp.exp(_implicit_step(h, s0))


def complete_mixing(F_i, Q, area, P_h, P_l):
    """Complete-mixing (both sides well mixed) permeator, any number of species.

    Args:
        F_i: Feed molar flows (n,), mol/s.
        Q: Permeances (n,), mol/(m^2 s Pa).
        area: Membrane area (m^2).
        P_h: Feed-side pressure (Pa).
        P_l: Permeate pressure (Pa).

    Returns:
        ``(retentate_flows, permeate_flows)``, each (n,).

    Example:
        >>> ret, perm = complete_mixing(jnp.array([0.5, 0.5]), jnp.array([1e-8, 1e-9]),
        ...                             50.0, 1e6, 1e5)
        >>> bool(jnp.all(ret + perm > 0.999))
        True
    """
    F_i = jnp.asarray(F_i, float)
    Q = jnp.asarray(Q, float)
    t = _cm_retentate_fraction(F_i, Q, jnp.asarray(area, float), P_h, P_l)
    y = _cm_permeate(t, F_i, Q, area, P_h, P_l)
    perm = jnp.minimum((1.0 - t) * jnp.sum(F_i) * y, F_i)
    return F_i - perm, perm


def local_permeate_composition(x, Q, P_h, P_l, n_iter=80):
    """Local permeate composition leaving a membrane element (cross flow).

    With no mixing on the permeate side, ``y_i = J_i / J`` where
    ``J_i = Q_i (x_i P_h - y_i P_l)``, so ``y_i = Q_i x_i P_h/(J + Q_i P_l)``
    and the total flux ``J`` is the root of ``sum_i y_i = 1`` (decreasing in
    ``J``, bracketed by ``[0, P_h sum Q_i x_i]``).

    Args:
        x: Feed-side mole fractions (n,).
        Q: Permeances (n,).
        P_h, P_l: Feed and permeate pressures.
        n_iter: Bisection halvings.

    Returns:
        ``(y, J)``: permeate mole fractions (n,) and total flux.
    """
    x = jnp.asarray(x, float)
    Q = jnp.asarray(Q, float)
    J_max = P_h * jnp.sum(Q * x)

    # Work in s = J/J_max in [0, 1]; f increasing in s.
    def f(s):
        J = s * J_max
        return 1.0 - jnp.sum(Q * x * P_h / (J + Q * P_l))

    s0 = _bisect_increasing(f, 0.0, 1.0, n_iter)
    s = _implicit_step(f, s0)
    J = s * J_max
    y = Q * x * P_h / (J + Q * P_l)
    return y / jnp.sum(y), J


def _rk4(rhs, y0, h, n_steps):
    """Fixed-step RK4 from ``y0`` with step ``h``; returns all states."""
    def step(y, _):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y1 = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return y1, y1

    _, ys = jax.lax.scan(step, y0, None, length=n_steps)
    return jnp.concatenate([y0[None], ys], axis=0)


def _safe_frac(v):
    tot = jnp.sum(v)
    return v / jnp.where(tot > 1e-300, tot, 1.0)


def cross_flow(F_i, Q, area, P_h, P_l, n_steps=100):
    """Cross-flow permeator: feed in plug flow, unmixed permeate.

    Integrates ``dL_i/dA = -Q_i (x_i P_h - y_i P_l)`` with the local permeate
    composition of :func:`local_permeate_composition`.

    Args:
        F_i: Feed molar flows (n,), mol/s.
        Q: Permeances (n,), mol/(m^2 s Pa).
        area: Membrane area (m^2).
        P_h, P_l: Feed and permeate pressures (Pa).
        n_steps: RK4 steps along the area.

    Returns:
        ``(retentate_flows, permeate_flows, profile)`` where ``profile`` has
        ``area`` (n_steps+1,), ``L`` (n_steps+1, n) feed-side flows and
        ``y_local`` (n_steps+1, n).
    """
    F_i = jnp.asarray(F_i, float)
    Q = jnp.asarray(Q, float)
    area = jnp.asarray(area, float)

    def rhs(L):
        L = jnp.maximum(L, 0.0)
        y, J = local_permeate_composition(_safe_frac(L), Q, P_h, P_l)
        return -J * y

    h = area / n_steps
    Ls = _rk4(rhs, F_i, h, n_steps)
    Ls = jnp.maximum(Ls, 0.0)
    ret = Ls[-1]
    y_loc = jax.vmap(lambda L: local_permeate_composition(_safe_frac(L), Q, P_h, P_l)[0])(Ls)
    profile = {"area": jnp.linspace(0.0, 1.0, n_steps + 1) * area, "L": Ls, "y_local": y_loc}
    return ret, F_i - ret, profile


def cocurrent(F_i, Q, area, P_h, P_l, n_steps=100):
    """Cocurrent plug-flow permeator (permeate flows with the feed).

    The driving force uses the bulk permeate composition ``V_i/V``; at the
    inlet, where ``V = 0``, it is the local composition.

    Args:
        F_i, Q, area, P_h, P_l, n_steps: As :func:`cross_flow`.

    Returns:
        ``(retentate_flows, permeate_flows, profile)``; ``profile`` has
        ``area``, ``L`` and ``y`` (bulk permeate composition).
    """
    F_i = jnp.asarray(F_i, float)
    Q = jnp.asarray(Q, float)
    area = jnp.asarray(area, float)
    delta = 1e-12 * jnp.sum(F_i)

    def rhs(L):
        L = jnp.maximum(L, 0.0)
        x = _safe_frac(L)
        V = F_i - L
        y_loc, _ = local_permeate_composition(x, Q, P_h, P_l)
        y = (V + delta * y_loc) / (jnp.sum(V) + delta)
        return -Q * jnp.maximum(x * P_h - y * P_l, 0.0)

    h = area / n_steps
    Ls = jnp.maximum(_rk4(rhs, F_i, h, n_steps), 0.0)
    Vs = F_i[None] - Ls
    ys = jax.vmap(_safe_frac)(Vs)
    profile = {"area": jnp.linspace(0.0, 1.0, n_steps + 1) * area, "L": Ls, "y": ys}
    return Ls[-1], F_i - Ls[-1], profile


def _counter_integrate(R, F_i, Q, area, P_h, P_l, n_steps):
    """Integrate a countercurrent permeator from the retentate end.

    The permeate at area coordinate ``a`` (feed end at 0) carries everything
    permeated between ``a`` and the closed retentate end, ``V_i = L_i - R_i``.
    Returns the feed-side flows from the retentate end (index 0) to the feed
    end (index n_steps).
    """
    delta = 1e-12 * jnp.sum(F_i)

    def rhs(L):
        # d L_i / d(-a): marching from the retentate end toward the feed.
        L = jnp.maximum(L, 0.0)
        x = _safe_frac(L)
        V = jnp.maximum(L - R, 0.0)
        y_loc, _ = local_permeate_composition(x, Q, P_h, P_l)
        y = (V + delta * y_loc) / (jnp.sum(V) + delta)
        return Q * jnp.maximum(x * P_h - y * P_l, 0.0)

    return _rk4(rhs, R, area / n_steps, n_steps)


_SHOOT_TOL = 1e-14     # countercurrent shooting: relative feed-end mismatch at which Newton stops


def countercurrent(F_i, Q, area, P_h, P_l, n_steps=100, newton_iter=40):
    """Countercurrent plug-flow permeator (permeate leaves at the feed end).

    Shooting: guess the retentate flows ``R_i``, integrate from the
    retentate end (where the permeate flow is zero) to the feed end, and
    solve ``L_i(feed end) = F_i`` by Newton's method on
    ``u_i = logit(R_i/F_i)``, started from the cross-flow solution. The
    converged root carries exact first derivatives (implicit function
    theorem).

    Args:
        F_i, Q, area, P_h, P_l, n_steps: As :func:`cross_flow`.
        newton_iter: Most Newton iterations (it stops earlier, once the
            relative feed-end mismatch is below 1e-14).

    Returns:
        ``(retentate_flows, permeate_flows, profile)``; ``profile`` has
        ``area`` (from the feed end), ``L``, ``y`` (bulk permeate) and
        ``residual`` (relative mismatch at the feed end).
    """
    F_i = jnp.asarray(F_i, float)
    Q = jnp.asarray(Q, float)
    area = jnp.asarray(area, float)
    R0, _, _ = cross_flow(F_i, Q, area, P_h, P_l, n_steps=max(n_steps // 2, 20))
    R0 = jax.lax.stop_gradient(R0)

    def resid(u, F_i, Q, area, P_h, P_l):
        R = F_i * jax.nn.sigmoid(u)
        Ls = _counter_integrate(R, F_i, Q, area, P_h, P_l, n_steps)
        return Ls[-1] / F_i - 1.0

    frac0 = jnp.clip(R0 / F_i, 1e-9, 1.0 - 1e-9)
    u0 = jnp.log(frac0) - jnp.log1p(-frac0)
    args = jax.lax.stop_gradient((F_i, Q, area, jnp.asarray(P_h, float), jnp.asarray(P_l, float)))

    # Newton converges quadratically (3-5 iterations from the cross-flow start), so it stops once
    # the feed-end mismatch reaches round-off rather than always running ``newton_iter`` steps.
    # Every input is stop-gradient: derivatives come from the implicit step below.
    def newton(state):
        k, u, r = state
        Jm = jax.jacfwd(resid)(u, *args)
        du = jnp.linalg.solve(Jm, -r)
        du = du * jnp.minimum(1.0, 2.0 / jnp.maximum(jnp.max(jnp.abs(du)), 1e-300))
        u = u + du
        return k + 1, u, resid(u, *args)

    def unconverged(state):
        k, _, r = state
        return (k < newton_iter) & (jnp.max(jnp.abs(r)) > _SHOOT_TOL)

    _, u_star, _ = jax.lax.while_loop(unconverged, newton, (0, u0, resid(u0, *args)))
    u_star = jax.lax.stop_gradient(u_star)
    # Implicit-function step: value-neutral, exact first derivatives.
    Jm = jax.lax.stop_gradient(jax.jacfwd(resid)(u_star, *args))
    r = resid(u_star, F_i, Q, area, P_h, P_l)
    u = u_star - jnp.linalg.solve(Jm, r - jax.lax.stop_gradient(r))
    R = F_i * jax.nn.sigmoid(u)
    Ls = _counter_integrate(R, F_i, Q, area, P_h, P_l, n_steps)[::-1]
    Vs = jnp.maximum(Ls - R[None], 0.0)
    ys = jax.vmap(_safe_frac)(Vs)
    profile = {"area": jnp.linspace(0.0, 1.0, n_steps + 1) * area, "L": Ls, "y": ys,
               "residual": jnp.max(jnp.abs(jax.lax.stop_gradient(r)))}
    return R, F_i - R, profile


def permeation_stage(F_i, Q, area, P_h, P_l, flow_pattern="complete_mixing", n_steps=100):
    """One gas-permeation stage in any of the four flow patterns.

    Args:
        F_i, Q, area, P_h, P_l: As :func:`cross_flow`.
        flow_pattern: ``"complete_mixing"``, ``"cross_flow"``,
            ``"cocurrent"`` or ``"countercurrent"``.
        n_steps: Integration steps (plug-flow patterns).

    Returns:
        ``(retentate_flows, permeate_flows, profile)``; ``profile`` is empty
        for complete mixing.
    """
    if flow_pattern == "complete_mixing":
        ret, perm = complete_mixing(F_i, Q, area, P_h, P_l)
        return ret, perm, {}
    if flow_pattern == "cross_flow":
        return cross_flow(F_i, Q, area, P_h, P_l, n_steps)
    if flow_pattern == "cocurrent":
        return cocurrent(F_i, Q, area, P_h, P_l, n_steps)
    if flow_pattern == "countercurrent":
        return countercurrent(F_i, Q, area, P_h, P_l, n_steps)
    raise ValueError(f"flow_pattern must be one of {FLOW_PATTERNS}, got {flow_pattern!r}")


def area_for_stage_cut(F_i, Q, P_h, P_l, theta, flow_pattern="complete_mixing", n_steps=100,
                       n_iter=60):
    """Membrane area that gives stage cut ``theta``.

    The stage cut rises monotonically with area, so the area is found by
    bisection in ``ln A`` (exact closed-form root for complete mixing) with
    an implicit-function step for the gradient.

    Args:
        F_i, Q, P_h, P_l: As :func:`cross_flow`.
        theta: Target stage cut in (0, 1).
        flow_pattern: See :func:`permeation_stage`.
        n_steps: Integration steps.
        n_iter: Bisection halvings.

    Returns:
        Area (m^2).
    """
    F_i = jnp.asarray(F_i, float)
    Q = jnp.asarray(Q, float)
    theta = jnp.asarray(theta, float)
    if flow_pattern == "complete_mixing":
        return _cm_area_for_cut(F_i, Q, P_h, P_l, 1.0 - theta)
    F = jnp.sum(F_i)
    A_cm = jax.lax.stop_gradient(_cm_area_for_cut(F_i, Q, P_h, P_l, 1.0 - theta))
    pattern = "cross_flow" if flow_pattern == "countercurrent" else flow_pattern

    def h(s, F_i, Q, P_h, P_l, theta, pattern=pattern):
        _, perm, _ = permeation_stage(F_i, Q, A_cm * jnp.exp(s), P_h, P_l, pattern, n_steps)
        return jnp.sum(perm) / jnp.sum(F_i) - theta

    args = (F_i, Q, P_h, P_l, theta)
    sargs = jax.lax.stop_gradient(args)
    s0 = _bisect_increasing(lambda s: h(s, *sargs), -6.0, 3.0, n_iter)
    if flow_pattern == "countercurrent":
        # Countercurrent needs a little less area than cross flow: refine from the
        # cross-flow area by the Illinois method (each evaluation is a shooting solve).
        def hc(s, F_i, Q, P_h, P_l, theta):
            _, perm, _ = countercurrent(F_i, Q, A_cm * jnp.exp(s), P_h, P_l, n_steps)
            return jnp.sum(perm) / jnp.sum(F_i) - theta

        s0 = _illinois_increasing(lambda s: hc(s, *sargs), s0 - 1.0, s0 + 0.2, 40)
        h = hc
    d = jax.lax.stop_gradient(jax.grad(h)(s0, *sargs))
    r = h(s0, *args)
    s = s0 - (r - jax.lax.stop_gradient(r)) / d
    del F
    return A_cm * jnp.exp(s)


# =============================================================================
# Gas permeation palette unit
# =============================================================================

@dataclass(repr=False)
class GasPermeationParams(ParamsMixin):
    """Parameters of a gas-permeation stage with given permeances.

    Attributes:
        permeances: Dict of species name -> permeance, mol/(m^2 s Pa)
            (``Q = P/delta``; multiply GPU by :data:`GPU`). A species not
            listed does not permeate.
        area: Membrane area (m^2). Ignored when ``stage_cut`` is given.
        stage_cut: Target stage cut (permeate/feed) in (0, 1); when given,
            the area is solved for and reported (design mode).
        flow_pattern: ``"complete_mixing"``, ``"cross_flow"``,
            ``"cocurrent"`` or ``"countercurrent"``.
        feed_pressure: Feed-side pressure (Pa); None uses the feed stream's.
        permeate_pressure: Permeate pressure (Pa); None uses
            ``feed_pressure / pressure_ratio``.
        pressure_ratio: Feed/permeate pressure ratio ``P_h/P_l`` (> 1),
            used when ``permeate_pressure`` is None.
        n_steps: Integration steps along the membrane (plug-flow patterns).
    """
    permeances: dict
    area: float | Array | None = None
    stage_cut: float | Array | None = None
    flow_pattern: str = "complete_mixing"
    feed_pressure: float | Array | None = None
    permeate_pressure: float | Array | None = None
    pressure_ratio: float | Array = 10.0
    n_steps: int = 100

    def __post_init__(self):
        """Reject impossible settings (concrete values only)."""
        if self.flow_pattern not in FLOW_PATTERNS:
            raise ValueError(f"flow_pattern must be one of {FLOW_PATTERNS}, got {self.flow_pattern!r}")
        if self.area is None and self.stage_cut is None:
            raise ValueError("give the membrane area or a target stage_cut")
        try:
            ratio = float(self.pressure_ratio)
        except Exception:
            ratio = None
        if ratio is not None and ratio <= 1.0:
            raise ValueError(f"pressure_ratio must exceed 1 (feed/permeate), got {ratio}")
        if self.stage_cut is not None:
            try:
                cut = float(self.stage_cut)
            except Exception:
                cut = None
            if cut is not None and not 0.0 < cut < 1.0:
                raise ValueError(f"stage_cut must be in (0, 1), got {cut}")


class GasPermeator:
    """Gas-permeation membrane stage (solution-diffusion), any flow pattern.

    Splits a gas feed into a retentate (feed side, at the feed pressure) and
    a permeate (at the permeate pressure) for any number of species, with
    the permeances given directly. The four flow patterns bracket real
    modules: complete mixing is the worst separation, countercurrent the best
    (hollow-fibre modules approach it; spiral-wound elements are near cross
    flow).

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({'O2': 0.21, 'N2': 0.79}, T=298.15, P=1e6)
        >>> unit = GasPermeator(GasPermeationParams(
        ...     permeances={'O2': 10 * GPU, 'N2': 2 * GPU}, area=1e5,
        ...     flow_pattern='cross_flow', pressure_ratio=5.0))
        >>> retentate, permeate, info = unit(feed)
        >>> 0.21 < float(info['permeate_fraction']['O2']) < 1
        True
    """

    symbol = "Gas permeator"
    equations = [
        r"J_i = Q_i\,(x_i P_h - y_i P_l)\qquad \text{(solution-diffusion)}",
        r"\alpha_{AB} = Q_A/Q_B,\quad \theta = \dot n_P/\dot n_F,\quad p_r = P_l/P_h",
        r"\frac{y}{1-y} = \frac{\alpha\,(x - p_r y)}{(1-x) - p_r(1-y)}\qquad \text{(binary, complete mixing / local)}",
    ]
    assumptions = [
        "Solution-diffusion transport with constant permeances.",
        "Isothermal; no pressure drop on either side.",
        "Plug-flow patterns: no axial mixing; the permeate-side driving force uses the bulk permeate composition.",
    ]
    references = [
        "Geankoplis, C.J. Transport Processes and Separation Process Principles, 5e, §13.3-13.7.",
        "Wankat, P.C. Separation Process Engineering, 4e, Ch. 17.",
        "Seader, J.D., Henley, E.J., Roper, D.K. Separation Process Principles, 3e, §14.3.",
        "Walawender, W.P., Stern, S.A. Sep. Sci. 7, 553 (1972).",
    ]
    parameter_symbols = {"area": "A", "stage_cut": r"\theta", "pressure_ratio": r"P_h/P_l",
                         "feed_pressure": "P_h", "permeate_pressure": "P_l"}
    parameter_units = {"permeances": "mol/(m^2 s Pa)", "area": "m^2", "stage_cut": "-",
                       "feed_pressure": "Pa", "permeate_pressure": "Pa", "pressure_ratio": "-",
                       "n_steps": "-", "flow_pattern": "-"}
    numerical_method = ("Complete mixing: bracketed root; cross flow and cocurrent: RK4 along the area; "
                        "countercurrent: shooting with Newton; implicit-function gradients.")

    def __init__(self, params: GasPermeationParams):
        """Initialize.

        Args:
            params: :class:`GasPermeationParams`.
        """
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Separate a gas feed.

        Args:
            feed: Feed stream (mol/s, K, Pa).

        Returns:
            ``(retentate, permeate, info)``; ``info`` has ``stage_cut``,
            ``area``, ``permeate_fraction`` and ``retentate_fraction``
            (mole fractions by species), ``recovery`` (fraction of each
            species' feed in the permeate), ``selectivity`` (relative to the
            slowest listed species), ``P_feed``, ``P_permeate``,
            ``pressure_ratio`` and ``profile`` (plug-flow patterns).
        """
        p = self.params
        flows = get_flows(feed)
        species = list(flows)
        F_i = jnp.stack([jnp.asarray(flows[s], float) for s in species])
        Q = jnp.stack([jnp.asarray(p.permeances.get(s, 0.0), float) for s in species])
        P_h = jnp.asarray(feed["P"] if p.feed_pressure is None else p.feed_pressure, float)
        P_l = jnp.asarray(P_h / p.pressure_ratio if p.permeate_pressure is None else p.permeate_pressure,
                          float)
        try:
            ph, pl = float(P_h), float(P_l)
        except Exception:
            ph = pl = None
        if ph is not None and pl >= ph:
            raise ValueError(f"permeate pressure ({pl} Pa) must be below the feed pressure ({ph} Pa)")
        if p.stage_cut is not None:
            area = area_for_stage_cut(F_i, Q, P_h, P_l, p.stage_cut, p.flow_pattern, p.n_steps)
        else:
            area = jnp.asarray(p.area, float)
        ret, perm, profile = permeation_stage(F_i, Q, area, P_h, P_l, p.flow_pattern, p.n_steps)
        T = feed["T"]
        retentate = make_stream({s: ret[k] for k, s in enumerate(species)}, T, P_h)
        permeate = make_stream({s: perm[k] for k, s in enumerate(species)}, T, P_l)
        y = _safe_frac(perm)
        x = _safe_frac(ret)
        listed = [k for k, s in enumerate(species) if s in p.permeances]
        Qmin = jnp.min(Q[jnp.array(listed)]) if listed else jnp.asarray(1.0)
        info = {
            "stage_cut": jnp.sum(perm) / jnp.sum(F_i),
            "area": area,
            "permeate_fraction": {s: y[k] for k, s in enumerate(species)},
            "retentate_fraction": {s: x[k] for k, s in enumerate(species)},
            "recovery": {s: perm[k] / jnp.maximum(F_i[k], 1e-300) for k, s in enumerate(species)},
            "selectivity": {s: Q[k] / jnp.maximum(Qmin, 1e-300) for k, s in enumerate(species)},
            "P_feed": P_h,
            "P_permeate": P_l,
            "pressure_ratio": P_h / P_l,
            "flow_pattern": p.flow_pattern,
            "profile": profile,
        }
        return retentate, permeate, info


# =============================================================================
# Reverse osmosis
# =============================================================================

def osmotic_pressure(c, T, i=2.0, phi=1.0):
    """Van't Hoff osmotic pressure ``pi = phi i c R T`` (Pa).

    Args:
        c: Solute concentration (mol/m^3 of solution, ~ mol/m^3 of water
            for dilute solutions).
        T: Temperature (K).
        i: Ions per formula unit (2 for NaCl).
        phi: Osmotic coefficient (1 ideal; about 0.93 for seawater NaCl).

    Returns:
        Osmotic pressure (Pa).

    Example:
        >>> round(float(osmotic_pressure(100.0, 298.15)) / 101325, 2)
        4.89
    """
    return jnp.asarray(phi) * jnp.asarray(i) * jnp.asarray(c) * R_GAS * jnp.asarray(T)


def ro_local_flux(dP, c_b, A_w, B, k=None, T=298.15, i=2.0, phi=1.0, n_iter=100):
    """Local water flux of a reverse-osmosis membrane with polarization.

    Solves together (solution-diffusion with film theory)::

        J_v = A_w (dP - (pi(c_w) - pi(c_p)))
        c_p = J_s/J_v = B (c_w - c_p)/J_v
        (c_w - c_p)/(c_b - c_p) = exp(J_v/k)

    which reduce to ``c_p = c_b E/(J_v/B + E)`` and
    ``c_w = c_p (J_v + B)/B`` with ``E = exp(J_v/k)``, leaving one
    increasing scalar equation in ``J_v`` on ``[0, A_w dP]``.

    Args:
        dP: Transmembrane hydraulic pressure difference (Pa).
        c_b: Bulk feed-side solute concentration (mol/m^3).
        A_w: Water permeance (m/(s Pa)); a mass-flux coefficient in
            kg/(s m^2 Pa) divided by the water density.
        B: Solute permeance (m/s); 0 for a perfect membrane.
        k: Feed-side mass-transfer coefficient (m/s); None for no
            polarization.
        T: Temperature (K).
        i, phi: Van't Hoff factor and osmotic coefficient
            (:func:`osmotic_pressure`).
        n_iter: Bisection halvings.

    Returns:
        dict with ``J_v`` (m/s), ``J_s`` (mol/(m^2 s)), ``c_w``, ``c_p``
        (mol/m^3), ``rejection`` ``1 - c_p/c_b`` (observed),
        ``polarization`` ``c_w/c_b`` and ``delta_pi`` (Pa).
    """
    dP = jnp.asarray(dP, float)
    c_b = jnp.asarray(c_b, float)
    A_w = jnp.asarray(A_w, float)
    B = jnp.asarray(B, float)
    slope = osmotic_pressure(1.0, T, i, phi)
    Jmax = A_w * dP

    def parts(Jv):
        E = jnp.exp(Jv / k) if k is not None else jnp.ones_like(Jv)
        c_p = c_b * E * B / (Jv + B * E + 1e-300)
        c_w = jnp.where(B > 0, c_p * (Jv + B) / jnp.where(B > 0, B, 1.0), c_b * E)
        return c_w, c_p

    def g(s):
        Jv = s * Jmax
        c_w, c_p = parts(Jv)
        return Jv - A_w * (dP - slope * (c_w - c_p))

    def g_scaled(s):
        return g(s) / Jmax

    s0 = _bisect_increasing(g_scaled, 0.0, 1.0, n_iter)
    s = _implicit_step(g_scaled, s0)
    Jv = s * Jmax
    c_w, c_p = parts(Jv)
    return {
        "J_v": Jv,
        "J_s": Jv * c_p,
        "c_w": c_w,
        "c_p": c_p,
        "rejection": 1.0 - c_p / c_b,
        "polarization": c_w / c_b,
        "delta_pi": slope * (c_w - c_p),
    }


@dataclass(repr=False)
class ReverseOsmosisParams(ParamsMixin):
    """Parameters of a reverse-osmosis element or train.

    Attributes:
        A_w: Water permeance (m/(s Pa)).
        B: Solute permeance (m/s).
        area: Membrane area (m^2).
        solute: Name of the solute species in the feed stream.
        k: Feed-channel mass-transfer coefficient (m/s); None for no
            concentration polarization.
        permeate_pressure: Permeate pressure (Pa).
        vant_hoff_i: Ions per formula unit of the solute.
        osmotic_coefficient: Osmotic coefficient ``phi``.
        water_density: Density used to turn water moles into volume (kg/m^3).
        n_steps: Integration steps along the element.
    """
    A_w: float | Array
    B: float | Array
    area: float | Array
    solute: str = "NaCl"
    k: float | Array | None = None
    permeate_pressure: float | Array = 101325.0
    vant_hoff_i: float | Array = 2.0
    osmotic_coefficient: float | Array = 1.0
    water_density: float | Array = 997.0
    n_steps: int = 60


class ReverseOsmosis:
    """Reverse-osmosis element: solution-diffusion with polarization, plug flow.

    Integrates ``dQ/dA = -J_v`` and ``d(Q c)/dA = -J_v c_p`` along the feed
    channel, with the local fluxes from :func:`ro_local_flux`. The feed
    stream carries water (``H2O``) and the solute in mol/s; the water volume
    is taken as moles x 0.018015 kg/mol / ``water_density``. The flux
    vanishes smoothly where the concentrate osmotic pressure reaches the
    applied pressure, which is the thermodynamic recovery limit
    ``1 - pi_f/dP`` for a perfect membrane. Near that limit the flux falls
    steeply, so use enough ``n_steps`` that one step does not carry the
    concentrate past it (fixed-step RK4 overshoots there).

    Example:
        >>> from difflow import make_stream
        >>> feed = make_stream({'H2O': 55.4, 'NaCl': 0.6}, T=298.15, P=6e6)
        >>> ro = ReverseOsmosis(ReverseOsmosisParams(A_w=1e-11, B=1e-8, area=40.0))
        >>> conc, perm, info = ro(feed)
        >>> 0 < float(info['recovery']) < 1
        True
    """

    symbol = "RO"
    equations = [
        r"J_v = A_w\,(\Delta P - \Delta\pi),\qquad J_s = B\,(c_w - c_p)",
        r"\frac{c_w - c_p}{c_b - c_p} = e^{J_v/k}\qquad \text{(film theory)}",
        r"\pi = \phi\, i\, c\, R T\qquad \text{(van't Hoff)}",
    ]
    assumptions = [
        "Solution-diffusion; constant permeances A_w and B.",
        "Isothermal; no feed-channel pressure drop; plug flow along the element.",
        "Ideal (van't Hoff) osmotic pressure scaled by an osmotic coefficient.",
    ]
    references = [
        "Geankoplis, C.J. Transport Processes and Separation Process Principles, 5e, §13.9-13.10.",
        "Wankat, P.C. Separation Process Engineering, 4e, Ch. 17.",
        "Baker, R.W. Membrane Technology and Applications, 3e, Ch. 5.",
    ]
    parameter_symbols = {"A_w": "A_w", "B": "B", "area": "A", "k": "k",
                         "permeate_pressure": "P_l"}
    parameter_units = {"A_w": "m/(s Pa)", "B": "m/s", "area": "m^2", "k": "m/s",
                       "permeate_pressure": "Pa", "vant_hoff_i": "-", "osmotic_coefficient": "-",
                       "water_density": "kg/m^3", "n_steps": "-", "solute": "-"}
    numerical_method = "RK4 along the area; local fluxes by bracketed root with implicit-function gradients."

    _MW_W = 0.018015

    def __init__(self, params: ReverseOsmosisParams):
        """Initialize.

        Args:
            params: :class:`ReverseOsmosisParams`.
        """
        self.params = params

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Desalinate a feed.

        Args:
            feed: Feed stream with ``H2O`` and the solute (mol/s), at the
                feed pressure.

        Returns:
            ``(concentrate, permeate, info)``; ``info`` has ``recovery``
            (permeate/feed water volume), ``rejection`` (``1 - c_p/c_f``,
            mixed permeate), ``c_feed``, ``c_concentrate``,
            ``c_permeate`` (mol/m^3), ``pi_feed``, ``pi_concentrate`` (Pa),
            ``recovery_limit`` (``1 - pi_f/dP``) and ``profile``.
        """
        p = self.params
        flows = get_flows(feed)
        if "H2O" not in flows or p.solute not in flows:
            raise ValueError(f"the RO feed needs 'H2O' and {p.solute!r}")
        T = feed["T"]
        dP = jnp.asarray(feed["P"], float) - jnp.asarray(p.permeate_pressure, float)
        Qf = jnp.asarray(flows["H2O"], float) * self._MW_W / p.water_density
        Nf = jnp.asarray(flows[p.solute], float)
        kw = dict(k=p.k, T=T, i=p.vant_hoff_i, phi=p.osmotic_coefficient)

        def rhs(state):
            Qv, N = state[0], state[1]
            c = N / jnp.maximum(Qv, 1e-300)
            f = ro_local_flux(dP, c, p.A_w, p.B, **kw)
            return jnp.stack([-f["J_v"], -f["J_s"]])

        h = jnp.asarray(p.area, float) / p.n_steps
        states = _rk4(rhs, jnp.stack([Qf, Nf]), h, p.n_steps)
        Qc, Nc = states[-1, 0], states[-1, 1]
        Qp, Np = Qf - Qc, Nf - Nc
        rho = p.water_density
        others = {s: v for s, v in flows.items() if s not in ("H2O", p.solute)}
        conc = make_stream({"H2O": Qc * rho / self._MW_W, p.solute: Nc, **others}, T, feed["P"])
        perm = make_stream({"H2O": Qp * rho / self._MW_W, p.solute: Np,
                            **{s: 0.0 * v for s, v in others.items()}}, T, p.permeate_pressure)
        c_f = Nf / Qf
        c_p = Np / jnp.maximum(Qp, 1e-300)
        slope = osmotic_pressure(1.0, T, p.vant_hoff_i, p.osmotic_coefficient)
        local = jax.vmap(lambda s: ro_local_flux(dP, s[1] / s[0], p.A_w, p.B, **kw))(states)
        info = {
            "recovery": Qp / Qf,
            "rejection": 1.0 - c_p / c_f,
            "c_feed": c_f,
            "c_concentrate": Nc / Qc,
            "c_permeate": c_p,
            "pi_feed": slope * c_f,
            "pi_concentrate": slope * Nc / Qc,
            "recovery_limit": 1.0 - slope * c_f / dP,
            "delta_P": dP,
            "permeate_flow_m3s": Qp,
            "profile": {"area": jnp.linspace(0.0, 1.0, p.n_steps + 1) * p.area,
                        "Q": states[:, 0], "c": states[:, 1] / states[:, 0],
                        "J_v": local["J_v"], "c_w": local["c_w"], "c_p": local["c_p"]},
        }
        return conc, perm, info


# =============================================================================
# Ultrafiltration / microfiltration
# =============================================================================

def polarization_modulus(J, k, R=1.0):
    """Concentration polarization modulus ``c_w/c_b`` (film theory).

    From ``(c_w - c_p)/(c_b - c_p) = E`` with ``E = exp(J/k)`` and the
    intrinsic rejection ``c_p = (1 - R) c_w``:
    ``c_w/c_b = E/(R + (1 - R) E)``, which is ``exp(J/k)`` for a fully
    retained solute (``R = 1``) and 1 for a freely passing one.

    Args:
        J: Volumetric flux (m/s).
        k: Mass-transfer coefficient (m/s).
        R: Intrinsic rejection (0-1).

    Returns:
        ``c_w/c_b``.
    """
    E = jnp.exp(jnp.asarray(J) / jnp.asarray(k))
    return E / (R + (1.0 - R) * E)


def gel_polarization_flux(k, c_gel, c_bulk):
    """Gel-polarization (mass-transfer-limited) flux ``J = k ln(c_g/c_b)``.

    Args:
        k: Mass-transfer coefficient (m/s, or any flux unit).
        c_gel: Gel (wall) concentration.
        c_bulk: Bulk concentration (same unit as ``c_gel``).

    Returns:
        Limiting flux in the unit of ``k``.
    """
    return jnp.asarray(k) * jnp.log(jnp.asarray(c_gel) / jnp.asarray(c_bulk))


def uf_flux(TMP, R_m, mu, k, c_bulk, c_gel):
    """UF/MF flux against transmembrane pressure: linear, then capped.

    At low pressure the flux follows the clean-membrane line
    ``TMP/(mu R_m)``; once the wall concentration reaches the gel
    concentration it stops rising (pressure-independent region) at the
    gel-polarization flux.

    Args:
        TMP: Transmembrane pressure (Pa).
        R_m: Membrane resistance (1/m).
        mu: Permeate viscosity (Pa s).
        k: Mass-transfer coefficient (m/s).
        c_bulk, c_gel: Bulk and gel concentrations.

    Returns:
        Flux (m/s).
    """
    return jnp.minimum(jnp.asarray(TMP) / (jnp.asarray(mu) * jnp.asarray(R_m)),
                       gel_polarization_flux(k, c_gel, c_bulk))


def batch_concentration_time(V0, VF, area, k, c0, c_gel, n=48):
    """Time to concentrate a batch from ``V0`` to ``VF`` at the gel-limited flux.

    For a fully retained solute ``c = c0 V0/V`` and
    ``t = int_{VF}^{V0} dV / (area k ln(c_gel V/(c0 V0)))`` (Gauss-Legendre).

    Args:
        V0, VF: Initial and final volumes (m^3).
        area: Membrane area (m^2).
        k: Mass-transfer coefficient (m/s).
        c0: Initial concentration.
        c_gel: Gel concentration (must exceed ``c0 V0/VF``).
        n: Quadrature points.

    Returns:
        Time (s).
    """
    xg, wg = np.polynomial.legendre.leggauss(n)
    V0, VF = jnp.asarray(V0, float), jnp.asarray(VF, float)
    V = 0.5 * (V0 - VF) * xg + 0.5 * (V0 + VF)
    integrand = 1.0 / (area * gel_polarization_flux(k, c_gel, c0 * V0 / V))
    return 0.5 * (V0 - VF) * jnp.sum(wg * integrand)
