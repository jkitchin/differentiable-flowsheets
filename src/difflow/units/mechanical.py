"""Mechanical separations: gravity settlers and thickeners, cake filters and gas cyclones.

Three palette units (``Settler``, ``CakeFilter``, ``Cyclone``) and the pure-JAX
library functions behind them (SI units, differentiable).

Streams
-------
All streams are in **mol/s**, K, Pa. A solid is carried as the pseudo-species
``F_<solid_species>`` (default ``F_solid``) with ``solid_molar_mass`` in kg/mol
(default 1), so ``F_solid`` is numerically **kg/s** of dry solid. This is the
convention of :class:`~difflow.units.dryer.Dryer` (its ``info["fines"]`` stream)
and the other solids units. Every other species in a slurry is a liquid-phase
species that rides with the liquid; the first of them (``liquid_species``,
default ``H2O``) fixes the liquid mass flow through ``liquid_MW``.

Settling: ``Settler``
---------------------
*Overflow-rate (clarifier) design.* A particle whose terminal velocity ``v_t``
exceeds the overflow rate ``v_o = Q_o / A`` is captured (ideal Hazen basin); a
smaller one is captured in the proportion ``v_t / v_o``. The capture
``min(1, v_t/v_o)`` is evaluated with a smooth minimum.

*Thickener area.* Coe and Clevenger (1916): with feed solids rate ``F_s`` and
underflow concentration ``C_u`` the area is set by the concentration ``C``
between the feed and underflow concentrations with the smallest solids flux::

    A = F_s  max_C  (1/C - 1/C_u) / v(C)                            (Coe-Clevenger)

Kynch's theory (1952) makes the same statement in terms of the limiting flux
``G_L = min_C  C v(C) / (1 - C/C_u)`` (``A = F_s / G_L``, identical algebra);
:func:`solids_flux` is the total flux ``C (v + u)`` with the underflow bulk
velocity ``u``. A single batch test (interface height ``z`` against ``t``)
gives ``v(C)`` by Kynch's construction (:func:`kynch_construction`): the
tangent at ``(t, z)`` has slope ``-v`` and meets ``t = 0`` at
``z_i = z + v t``; the concentration at that point is ``C = C_0 z_0 / z_i``.
Talmage and Fitch (1955) read the underflow time ``t_u`` from the tangent
that meets the line ``z_u = C_0 z_0 / C_u`` latest and take
``A = Q_0 t_u / z_0`` (:func:`thickener_area_talmage_fitch`); on the same
tangent nodes this is algebraically the Coe-Clevenger area, which the tests
check.

**The smooth maximum.** ``max`` over the concentration grid is replaced by a
log-sum-exp ``tau ln sum exp(a_i/tau)`` with ``tau = mean(|a|)/sharpness``
(:func:`lse_max`). It always satisfies ``max <= LSE <= max + tau ln N``
(a conservative bias of at most ``ln(N)/sharpness`` relative to the scale
``mean|a|``); the default ``sharpness = 1000`` gives well under 1 % on a
100-point grid and ``sharpness -> inf`` recovers the grid maximum. Grid
points are taken from a fixed ``n_grid`` linear grid between ``C_0`` and
``C_u``, so the area is differentiable in ``C_u``, ``Q_0`` and the settling
parameters.

**Hindered settling.** With a particle model the velocity is
``v(C) = v_t (1 - C/rho_p)^n`` (Richardson-Zaki, :func:`hindered_settling_velocity`
from :mod:`difflow.particles`). The exponent is piecewise in ``Re_t`` (small
jumps at its boundaries): pass ``richardson_zaki_n`` explicitly when
optimizing over particle size.

Filtration: ``CakeFilter``
--------------------------
Ruth (1935) at constant pressure, ``V`` the filtrate volume::

    t / V = (K_p / 2) V + B,    K_p = mu alpha c_s / (A^2 dP),   B = mu R_m / (A dP)

and at constant rate ``q`` the pressure rises as ``dP = (mu q / A)(alpha c_s q t / A + R_m)``.
The cycle of a batch filter (press, leaf) is filtration, wash and downtime;
the throughput ``V / (t_f + t_w + t_d)`` has a maximum in ``t_f``
(:func:`optimal_cycle`). With the wash volume ``r V`` at ``phi`` times the
final filtration rate, ``T(V) = (K_p/2)(1 + 2r/phi) V^2 + B (1 + r/phi) V + t_d``
and the optimum is ``a V^2 = t_d`` independent of ``B``: with ``R_m = 0``
and no wash, ``t_f = t_d`` (the oracle the tests use).

Cyclones: ``Cyclone``
---------------------
Lapple (1951): the cut size ``d50 = [9 mu W / (2 pi N_e v_i (rho_p - rho_g))]^(1/2)``,
grade efficiency ``eta(d) = 1/(1 + (d50/d)^2)`` and the Shepherd-Lapple pressure
drop ``dP = K (H W / D_e^2) rho v_i^2 / 2``. **Unverified, typed from memory:**
the constant ``K = 16``, the Lapple standard proportions (``H = D/2``,
``W = D/4``, ``D_e = D/2``, ``S = 5D/8``, ``L_b = 2 D``, ``L_c = 2 D``,
``J = D/4``) and the Stairmand high-efficiency set (``0.5, 0.2, 0.5, 0.5,
1.5, 2.5, 0.375`` times ``D``). Cyclone mass-loading effects, re-entrainment
and the Barth / Muschelknautz models are not included.

Centrifuges (Sigma theory, ``Centrifuge``, ``DiscStackCentrifuge``) stay in
:mod:`difflow_bio.units.centrifuge`.

Not included: hydrocyclones, screening and size reduction, electrostatic
precipitators and bag filters, a per-stream particle size distribution (the
size distribution is a unit parameter), a wash-water inlet (wash volume and
time are reported, the cake moisture is the post-wash value), cost
correlations (none are cited here).

References
----------
- Coe, H.S. and Clevenger, G.H., Trans. AIME, 55, 356-384 (1916).
- Kynch, G.J., Trans. Faraday Soc., 48, 166-176 (1952).
- Talmage, W.P. and Fitch, E.B., Ind. Eng. Chem., 47, 38-41 (1955).
- Ruth, B.F., Ind. Eng. Chem., 27, 708-723 (1935).
- Lapple, C.E., Chem. Eng. 58(5), 144-151 (1951) (recalled; constants unverified).
- Shepherd, C.B. and Lapple, C.E., Ind. Eng. Chem., 31, 972-984 (1939) (recalled).
- Geankoplis, Transport Processes and Separation Process Principles, Ch. 14.
- McCabe, Smith and Harriott, Unit Operations of Chemical Engineering, Ch. 29-30.
- Coulson and Richardson, Vol. 2, Ch. 1 and 5; Cooper and Alley, Air Pollution Control, Ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.scipy.special import logsumexp

from difflow.params_mixin import ParamsMixin
from difflow.particles import (
    G_STD,
    hindered_settling_velocity,
    kozeny_carman,
    terminal_reynolds,
    terminal_velocity,
)
from difflow.staged import _pchip_eval, _pchip_slopes, _solve_monotone
from difflow.streams import Stream

#: Universal gas constant, J/mol/K
R_GAS = 8.314462618

_TINY = 1e-300
_V_FLOOR = 1e-15


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _safe_div(a, b, fallback=0.0):
    """``a / b`` with a finite value where ``b == 0`` (and a finite gradient there)."""
    a, b = _f(a), _f(b)
    ok = b != 0.0
    return jnp.where(ok, a / jnp.where(ok, b, 1.0), fallback)


def _soft_min(a, b, width) -> Array:
    """Smooth ``min(a, b)``: ``(a + b - sqrt((a-b)^2 + width^2)) / 2`` (error <= ``width/2``)."""
    a, b = _f(a), _f(b)
    return 0.5 * (a + b - jnp.sqrt((a - b) ** 2 + _f(width) ** 2))


def lse_max(a, sharpness: float = 1000.0, mask=None) -> Array:
    r"""Log-sum-exp maximum of an array, with a relative temperature.

    :math:`\tau \ln\sum e^{a_i/\tau}` with :math:`\tau = \overline{|a|}/\text{sharpness}`
    (the mean taken over the unmasked entries). It satisfies
    :math:`\max a \le \text{LSE} \le \max a + \tau\ln N`, so it never
    under-predicts (a conservative bias for an area), and
    ``sharpness -> inf`` recovers the maximum. Differentiable everywhere.

    Args:
        a: Values.
        sharpness: Inverse relative temperature (dimensionless, > 0).
        mask: Optional boolean array; ``False`` entries are excluded.

    Returns:
        The smooth maximum (scalar).
    """
    a = _f(a)
    if mask is None:
        m = jnp.ones_like(a, dtype=bool)
    else:
        m = jnp.asarray(mask, dtype=bool)
    a_safe = jnp.where(m, a, 0.0)
    n_valid = jnp.maximum(jnp.sum(m), 1)
    scale = jnp.sum(jnp.where(m, jnp.abs(a_safe), 0.0)) / n_valid + _TINY
    tau = scale / _f(sharpness)
    return tau * logsumexp(jnp.where(m, a_safe / tau, -jnp.inf))


# =============================================================================
# Settling: flux and thickener area
# =============================================================================


def solids_flux(C: Array | float, v: Array | float, u: Array | float = 0.0) -> Array:
    """Solids flux ``G = C (v + u)`` (kg/m^2/s): the batch flux ``C v`` plus the bulk-transport flux ``C u``.

    Args:
        C: Solids concentration (kg/m^3).
        v: Hindered settling velocity at ``C`` (m/s).
        u: Bulk downward velocity from underflow withdrawal (m/s); 0 gives the batch flux.
    """
    return _f(C) * (_f(v) + _f(u))


def batch_flux_curve(C: Array, v: Array) -> Array:
    """Batch flux curve ``G_b(C) = C v(C)`` (kg/m^2/s) from settling data."""
    return solids_flux(C, v)


def _grid(C0, C_u, n_grid: int) -> Array:
    return _f(C0) + (_f(C_u) - _f(C0)) * jnp.linspace(0.0, 1.0, n_grid)


def hindered_velocity_model(
    C: Array,
    d_p,
    rho_p,
    rho_f,
    mu,
    sphericity=1.0,
    n=None,
    method: str = "haider_levenspiel",
) -> Array:
    r"""Particle-model settling velocity ``v(C) = v_t (1 - C/rho_p)^n`` (m/s) for a suspension of ``C`` kg/m^3.

    ``v_t`` from :func:`difflow.particles.terminal_velocity`, the voidage
    ``1 - C/rho_p`` and the Richardson-Zaki exponent ``n`` (piecewise in
    ``Re_t`` if ``None``; see :func:`difflow.particles.hindered_settling_velocity`).
    """
    v_t = terminal_velocity(d_p, rho_p, rho_f, mu, sphericity, method)
    eps = 1.0 - _f(C) / _f(rho_p)
    if n is None:
        Re_t = terminal_reynolds(d_p, rho_p, rho_f, mu, sphericity, method)
        return hindered_settling_velocity(v_t, eps, Re_t=Re_t)
    return hindered_settling_velocity(v_t, eps, n=n)


def thickener_area_coe_clevenger(
    Q0: Array | float,
    C0: Array | float,
    C_u: Array | float,
    C_grid: Array,
    v_grid: Array,
    sharpness: float = 1000.0,
    return_details: bool = False,
):
    r"""Coe-Clevenger thickener area :math:`A = Q_0 C_0 \max_C (1/C - 1/C_u)/v(C)` (m^2).

    The maximum is a smooth log-sum-exp (:func:`lse_max`) over the supplied
    grid; grid points with ``C`` outside ``[C0, C_u]`` are excluded.

    Args:
        Q0: Feed slurry volume flow (m^3/s).
        C0: Feed solids concentration (kg/m^3).
        C_u: Underflow solids concentration (kg/m^3), ``> C0``.
        C_grid: Concentrations (kg/m^3) at which ``v`` is known.
        v_grid: Hindered settling velocity at ``C_grid`` (m/s).
        sharpness: Smooth-max sharpness (see :func:`lse_max`).
        return_details: Also return a dict with ``area``, the unit-free
            ``limiting_concentration`` (softmax-weighted mean of the grid
            concentration over the near-maximal terms), ``limiting_flux``
            (kg/m^2/s), and the per-point areas.

    Returns:
        The area (m^2), or ``(area, details)``.
    """
    C = _f(C_grid)
    v = jnp.maximum(_f(v_grid), _V_FLOOR)
    C0, C_u, Q0 = _f(C0), _f(C_u), _f(Q0)
    a = (1.0 / C - 1.0 / C_u) / v
    mask = (C >= C0 * (1.0 - 1e-12)) & (C <= C_u * (1.0 + 1e-12))
    m = lse_max(a, sharpness, mask)
    area = Q0 * C0 * m
    if not return_details:
        return area
    a_safe = jnp.where(mask, a, 0.0)
    scale = jnp.sum(jnp.where(mask, jnp.abs(a_safe), 0.0)) / jnp.maximum(jnp.sum(mask), 1) + _TINY
    w = jax.nn.softmax(jnp.where(mask, a_safe / (scale / _f(sharpness)), -jnp.inf))
    details = {
        "area": area,
        "limiting_concentration": jnp.sum(w * C),
        "limiting_flux": Q0 * C0 / area,
        "per_point_area": Q0 * C0 * a,
    }
    return area, details


def kynch_construction(t_data: Array, z_data: Array, C0: Array | float):
    """Kynch's construction: the batch flux curve from one interface-height-versus-time test.

    At each data node the tangent to the ``z(t)`` curve (PCHIP slope, which is
    monotone and exact on straight segments) has slope ``-v`` and meets the
    ``t = 0`` axis at ``z_i = z + v t``. The suspension at the interface
    has the concentration ``C = C0 z0 / z_i`` and the interface falls at ``v``.

    Args:
        t_data: Times (s), increasing.
        z_data: Interface heights (m).
        C0: Initial concentration of the test (kg/m^3).

    Returns:
        ``(C, v, z_intercept)``: concentration (kg/m^3), settling velocity (m/s)
        and tangent intercept (m) at every node.
    """
    t, z = _f(t_data), _f(z_data)
    slope = _pchip_slopes(t, z)
    v = -slope
    z_int = z + v * t
    C = _f(C0) * z[0] / z_int
    return C, v, z_int


def thickener_area_kynch(
    F_s: Array | float,
    C_u: Array | float,
    t_data: Array,
    z_data: Array,
    C0_batch: Array | float,
    sharpness: float = 1000.0,
) -> Array:
    """Thickener area from the Kynch flux curve of one batch test: ``A = F_s max_i (1/C_i - 1/C_u)/v_i`` (m^2).

    Nodes with ``C_i >= C_u`` or a non-positive velocity are excluded.

    Args:
        F_s: Solids feed rate (kg/s).
        C_u: Underflow concentration (kg/m^3).
        t_data, z_data: Batch interface-height curve.
        C0_batch: Initial concentration of the batch test (kg/m^3).
        sharpness: Smooth-max sharpness.
    """
    C, v, _ = kynch_construction(t_data, z_data, C0_batch)
    C_u = _f(C_u)
    ok = (v > _V_FLOOR) & (C < C_u)
    v_safe = jnp.where(ok, v, 1.0)
    a = jnp.where(ok, (1.0 / C - 1.0 / C_u) / v_safe, 0.0)
    return _f(F_s) * lse_max(a, sharpness, ok)


def thickener_area_talmage_fitch(
    Q0: Array | float,
    C0: Array | float,
    z0: Array | float,
    C_u: Array | float,
    t_data: Array,
    z_data: Array,
    sharpness: float = 1000.0,
    return_details: bool = False,
):
    r"""Talmage-Fitch thickener area :math:`A = Q_0 t_u / z_0` (m^2) from one batch test at the feed concentration.

    The underflow height is :math:`z_u = C_0 z_0/C_u`. A tangent to the
    batch curve at a node above ``z_u`` meets ``z = z_u`` at
    :math:`t_u = t + (z - z_u)/v`; the controlling (latest) one, which is
    the tangent at the compression point of a settling curve, is the
    maximum over nodes (smooth, see :func:`lse_max`).

    Args:
        Q0: Feed volume flow (m^3/s) at the test concentration.
        C0: Test (feed) concentration (kg/m^3).
        z0: Initial interface height of the test (m); must equal ``z_data[0]``.
        C_u: Underflow concentration (kg/m^3).
        t_data, z_data: Batch interface-height curve.
        sharpness: Smooth-max sharpness.
        return_details: Also return ``{"t_u", "z_u"}``.
    """
    t, z = _f(t_data), _f(z_data)
    z0, C0, C_u = _f(z0), _f(C0), _f(C_u)
    slope = _pchip_slopes(t, z)
    v = -slope
    z_u = C0 * z0 / C_u
    ok = (v > _V_FLOOR) & (z > z_u)
    v_safe = jnp.where(ok, v, 1.0)
    t_u_all = jnp.where(ok, t + (z - z_u) / v_safe, 0.0)
    t_u = lse_max(t_u_all, sharpness, ok)
    area = _f(Q0) * t_u / z0
    if return_details:
        return area, {"t_u": t_u, "z_u": z_u}
    return area


def overflow_rate_area(Q_overflow: Array | float, d_p, rho_p, rho_f, mu, sphericity=1.0,
                       safety_factor: float = 1.0) -> Array:
    """Clarifier area ``A = SF Q_o / v_t(d_p)`` (m^2) for the design particle ``d_p`` (ideal Hazen basin)."""
    v_t = terminal_velocity(d_p, rho_p, rho_f, mu, sphericity)
    return _f(safety_factor) * _f(Q_overflow) / v_t


def overflow_rate_capture(
    v_overflow: Array | float, d_bins: Array, rho_p, rho_f, mu, sphericity=1.0, width: float = 1e-4
) -> Array:
    """Capture fraction per size bin ``min(1, v_t / v_o)`` of an ideal basin (smooth minimum with width 1e-4, so exactly-captured sizes give 1 - 5e-5)."""
    v_t = terminal_velocity(_f(d_bins), rho_p, rho_f, mu, sphericity)
    return _soft_min(1.0, v_t / _f(v_overflow), width)


# =============================================================================
# Filtration: Ruth
# =============================================================================


def ruth_constants(A, dP, mu, alpha, c_s, R_m=0.0):
    """Ruth constants ``(K_p, B)``: ``K_p = mu alpha c_s / (A^2 dP)`` (s/m^6), ``B = mu R_m / (A dP)`` (s/m^3).

    Args:
        A: Filter area (m^2).
        dP: Pressure drop (Pa).
        mu: Filtrate viscosity (Pa s).
        alpha: Specific cake resistance (m/kg).
        c_s: Mass of cake solids per volume of filtrate (kg/m^3).
        R_m: Filter-medium resistance (1/m).
    """
    A, dP = _f(A), _f(dP)
    return _f(mu) * _f(alpha) * _f(c_s) / (A**2 * dP), _f(mu) * _f(R_m) / (A * dP)


def ruth_constant_pressure(V: Array | float, K_p: Array | float, B: Array | float) -> Array:
    """Filtration time ``t = (K_p / 2) V^2 + B V`` (s) to collect the filtrate volume ``V`` at constant pressure."""
    V = _f(V)
    return 0.5 * _f(K_p) * V * V + _f(B) * V


def filtrate_volume(t: Array | float, K_p: Array | float, B: Array | float) -> Array:
    r"""Filtrate volume (m^3) after time ``t`` at constant pressure: the positive root of the Ruth quadratic.

    Written as :math:`V = 2t / (B + \sqrt{B^2 + 2 K_p t})` (the rationalized form
    of :math:`(-B + \sqrt{B^2 + 2K_p t})/K_p`), which has no cancellation as
    ``K_p -> 0`` (gives ``t/B``) and is exact at ``B = 0`` (gives
    ``sqrt(2t/K_p)``). ``V = 0`` at ``t = 0`` with a finite gradient.
    """
    t, K_p, B = _f(t), _f(K_p), _f(B)
    den = B + jnp.sqrt(B * B + 2.0 * K_p * t)
    ok = den > 0.0
    return jnp.where(ok, 2.0 * t / jnp.where(ok, den, 1.0), 0.0)


def filtrate_rate(V: Array | float, K_p: Array | float, B: Array | float) -> Array:
    """Instantaneous filtrate rate ``dV/dt = 1 / (K_p V + B)`` (m^3/s) at constant pressure."""
    return 1.0 / (_f(K_p) * _f(V) + _f(B))


def ruth_constant_rate(t: Array | float, q: Array | float, A, mu, alpha, c_s, R_m=0.0) -> Array:
    """Pressure drop (Pa) at time ``t`` of a constant-rate filtration: ``dP = (mu q / A)(alpha c_s q t / A + R_m)``.

    Args:
        t: Time (s).
        q: Constant filtrate rate (m^3/s).
        A, mu, alpha, c_s, R_m: As in :func:`ruth_constants`.
    """
    q, A = _f(q), _f(A)
    return _f(mu) * q / A * (_f(alpha) * _f(c_s) * q * _f(t) / A + _f(R_m))


def time_to_pressure(dP_max, q, A, mu, alpha, c_s, R_m=0.0) -> Array:
    """Time (s) at which a constant-rate filtration reaches ``dP_max`` (inverse of :func:`ruth_constant_rate`)."""
    q, A = _f(q), _f(A)
    return (_f(dP_max) * A / (_f(mu) * q) - _f(R_m)) * A / (_f(alpha) * _f(c_s) * q)


def fit_ruth(t, V, A=None, dP=None, mu=None, c_s=None) -> dict:
    """The Ruth plot: a linear least-squares fit of ``t/V`` against ``V``.

    ``t/V = (K_p/2) V + B``: the slope is ``K_p/2`` and the intercept ``B``. With ``A``,
    ``dP``, ``mu`` and ``c_s`` it also returns ``alpha = K_p A^2 dP / (mu c_s)``
    and ``R_m = B A dP / mu``. The fit is ordinary (unweighted) least squares in
    the plotted variables, as the textbook does; it is differentiable in the data.

    Args:
        t: Times (s).
        V: Cumulative filtrate volumes (m^3), positive.
        A, dP, mu, c_s: Optional (see :func:`ruth_constants`).

    Returns:
        dict with ``K_p``, ``B``, ``slope``, ``intercept``, ``r2`` and, when the
        optional arguments are given, ``alpha`` and ``R_m``.
    """
    t, V = _f(t), _f(V)
    y = t / V
    xm, ym = jnp.mean(V), jnp.mean(y)
    sxx = jnp.sum((V - xm) ** 2)
    slope = jnp.sum((V - xm) * (y - ym)) / sxx
    icpt = ym - slope * xm
    res = y - (slope * V + icpt)
    sst = jnp.sum((y - ym) ** 2)
    out = {"K_p": 2.0 * slope, "B": icpt, "slope": slope, "intercept": icpt,
           "r2": 1.0 - _safe_div(jnp.sum(res**2), sst, 0.0)}
    if None not in (A, dP, mu, c_s):
        out["alpha"] = 2.0 * slope * _f(A) ** 2 * _f(dP) / (_f(mu) * _f(c_s))
        out["R_m"] = icpt * _f(A) * _f(dP) / _f(mu)
    return out


def compressible_cake(alpha0, dP, s, dP_ref: float = 1.0) -> Array:
    """Compressible cake: ``alpha = alpha0 (dP / dP_ref)^s`` (m/kg); ``s = 0`` is incompressible."""
    return _f(alpha0) * (_f(dP) / _f(dP_ref)) ** _f(s)


def specific_cake_resistance(d_p, voidage, rho_p, sphericity=1.0, k0: float = 5.0) -> Array:
    """Kozeny-Carman specific cake resistance ``alpha = 36 k0 (1-eps)/(eps^3 rho_p (phi d_p)^2)`` (m/kg).

    The form given in :func:`difflow.particles.kozeny_carman`: the pressure
    gradient of that function divided by ``(1 - eps) rho_p mu u``.
    """
    eps = _f(voidage)
    return 36.0 * k0 * (1.0 - eps) / (eps**3 * _f(rho_p) * (_f(sphericity) * _f(d_p)) ** 2)


def cake_solids_per_filtrate(w: Array | float, rho_f: Array | float, m: Array | float) -> Array:
    """``c_s = rho_f w / (1 - m w)`` (kg/m^3): slurry solids mass fraction ``w``, filtrate density, wet/dry cake mass ratio ``m``."""
    w = _f(w)
    return _f(rho_f) * w / (1.0 - _f(m) * w)


def wash_time(V_wash: Array | float, final_rate: Array | float, rate_factor: Array | float = 1.0) -> Array:
    """Wash time (s): ``V_wash / (rate_factor x final filtration rate)``.

    ``rate_factor = 1`` is the standard assumption (wash at the final filtration
    rate); a plate-and-frame press washing through the whole cake is often taken at 1/4.
    """
    return _f(V_wash) / (_f(rate_factor) * _f(final_rate))


def cycle_time_total(t_f, V, K_p, B, t_down, wash_ratio=0.0, wash_rate_factor=1.0, t_wash_fn=None) -> tuple:
    """``(t_wash, t_cycle)`` of a batch cycle that collects ``V`` in filtration time ``t_f`` (s)."""
    if t_wash_fn is not None:
        t_w = _f(t_wash_fn(V))
    else:
        t_w = wash_time(_f(wash_ratio) * V, filtrate_rate(V, K_p, B), wash_rate_factor)
    return t_w, _f(t_f) + t_w + _f(t_down)


def cycle_throughput(t_f, K_p, B, t_down, wash_ratio=0.0, wash_rate_factor=1.0, t_wash_fn=None) -> Array:
    """Batch-cycle throughput ``V / (t_f + t_w + t_d)`` (m^3/s of filtrate), differentiable in ``t_f``.

    ``V = filtrate_volume(t_f, K_p, B)``. The wash volume is ``wash_ratio * V`` washed at
    ``wash_rate_factor`` times the final rate, unless ``t_wash_fn(V)`` (a JAX function
    returning seconds) is given.
    """
    V = filtrate_volume(t_f, K_p, B)
    _, T = cycle_time_total(t_f, V, K_p, B, t_down, wash_ratio, wash_rate_factor, t_wash_fn)
    return V / T


def optimal_cycle(K_p, B, t_down, wash_ratio=0.0, wash_rate_factor=1.0, t_wash_fn=None) -> dict:
    r"""Filtration time that maximizes the batch throughput.

    Writing the cycle as a function of the filtrate volume, ``T(V) = t_f(V) + t_w(V) + t_d``, the
    optimum satisfies ``T = V T'(V)``. With the standard wash model this is
    :math:`V^* = \sqrt{2 t_d / (K_p (1 + 2 r/\phi))}`, independent of ``B`` (closed form). With a
    user ``t_wash_fn(V)`` (convex, zero at ``V = 0``, JAX-traceable) the stationarity
    condition is solved by a bracketed bisection with an implicit-gradient Newton
    step, on ``(0, V_0]`` where ``V_0 = sqrt(2 t_d/K_p)`` is the no-wash optimum (an upper bound).
    Both routes are differentiable in ``K_p``, ``B`` and ``t_down``.

    Args:
        K_p, B: Ruth constants (:func:`ruth_constants`).
        t_down: Downtime per cycle (s), ``> 0``.
        wash_ratio: Wash volume per filtrate volume, ``r``.
        wash_rate_factor: Wash rate over the final filtration rate, ``phi``.
        t_wash_fn: Optional wash-time function of ``V`` that replaces the standard model.

    Returns:
        dict with ``V`` (m^3), ``t_f``, ``t_wash``, ``t_cycle`` (s) and ``throughput`` (m^3/s).
    """
    K_p, B, t_d = _f(K_p), _f(B), _f(t_down)
    if t_wash_fn is None:
        c = 1.0 + 2.0 * _f(wash_ratio) / _f(wash_rate_factor)
        V = jnp.sqrt(2.0 * t_d / (K_p * c))
    else:
        V0 = jnp.sqrt(2.0 * t_d / K_p)

        def T(v):
            return 0.5 * K_p * v * v + B * v + _f(t_wash_fn(v)) + t_d

        def g(v):
            return T(v) - v * jax.grad(T)(v)

        V = _solve_monotone(g, 1e-12 * V0, V0)
    t_f = ruth_constant_pressure(V, K_p, B)
    t_w, t_c = cycle_time_total(t_f, V, K_p, B, t_d, wash_ratio, wash_rate_factor, t_wash_fn)
    return {"V": V, "t_f": t_f, "t_wash": t_w, "t_cycle": t_c, "throughput": V / t_c}


def filter_area_for_cycle(V, t_on, mu, alpha, c_s, dP, R_m=0.0, c2=1.0, c1=1.0) -> Array:
    """Area (m^2) so that ``c2 a x^2 + c1 b x = t_on`` with ``x = 1/A``, ``a = mu alpha c_s V^2/(2 dP)``, ``b = mu R_m V/dP``.

    The constant-pressure filtration of ``V`` plus wash fits in ``t_on`` (``c2 = 1 + 2r/phi``,
    ``c1 = 1 + r/phi`` for wash ratio ``r``; both 1 without wash). Closed form, no iteration.
    """
    V, dP = _f(V), _f(dP)
    a = _f(mu) * _f(alpha) * _f(c_s) * V * V / (2.0 * dP)
    b = _f(mu) * _f(R_m) * V / dP
    t_on = _f(t_on)
    c1b = _f(c1) * b
    return (c1b + jnp.sqrt(c1b * c1b + 4.0 * _f(c2) * a * t_on)) / (2.0 * t_on)


# =============================================================================
# Cyclones: Lapple
# =============================================================================

#: geometry ratios ``[H, W, D_e, S, L_b, L_c, J] / D`` of the Lapple and Stairmand high-efficiency
#: cyclones. UNVERIFIED: typed from memory (Lapple 1951; Coulson & Richardson Vol. 2 / Towler & Sinnott).
CYCLONE_GEOMETRIES = {
    "lapple": [0.5, 0.25, 0.5, 0.625, 2.0, 2.0, 0.25],
    "stairmand_he": [0.5, 0.2, 0.5, 0.5, 1.5, 2.5, 0.375],
}
_GEOM_KEYS = ("H", "W", "D_e", "S", "L_b", "L_c", "J")


def cyclone_geometry(D, geometry: str = "lapple", ratios=None) -> dict:
    """Dimensions (m) of a cyclone of body diameter ``D``: ``H`` inlet height, ``W`` inlet width, ``D_e`` gas outlet
    diameter, ``S`` outlet-duct length, ``L_b`` body (cylinder) length, ``L_c`` cone length and ``J`` dust-outlet diameter.

    Args:
        D: Body diameter (m).
        geometry: ``"lapple"``, ``"stairmand_he"`` (see :data:`CYCLONE_GEOMETRIES`, unverified) or ``"custom"``.
        ratios: Seven ratios to ``D`` for ``"custom"``.
    """
    if geometry == "custom":
        if ratios is None or len(ratios) != 7:
            raise ValueError("custom geometry needs 7 ratios [H, W, D_e, S, L_b, L_c, J] / D")
        r = ratios
    elif geometry in CYCLONE_GEOMETRIES:
        r = CYCLONE_GEOMETRIES[geometry]
    else:
        raise ValueError(f"geometry must be one of {tuple(CYCLONE_GEOMETRIES)} or 'custom'")
    D = _f(D)
    out = {k: _f(x) * D for k, x in zip(_GEOM_KEYS, r)}
    out["D"] = D
    return out


def lapple_turns(H, L_b, L_c) -> Array:
    """Lapple's estimate of the effective number of turns ``N_e = (L_b + L_c/2) / H``."""
    return (_f(L_b) + 0.5 * _f(L_c)) / _f(H)


def lapple_cut_size(mu, W, N_e, v_i, rho_p, rho_g) -> Array:
    """Lapple cut size ``d50 = [9 mu W / (2 pi N_e v_i (rho_p - rho_g))]^(1/2)`` (m).

    Args:
        mu: Gas viscosity (Pa s).
        W: Inlet width (m).
        N_e: Effective number of turns.
        v_i: Inlet velocity (m/s).
        rho_p, rho_g: Particle and gas densities (kg/m^3).
    """
    return jnp.sqrt(9.0 * _f(mu) * _f(W) / (2.0 * jnp.pi * _f(N_e) * _f(v_i) * (_f(rho_p) - _f(rho_g))))


def lapple_efficiency(d: Array | float, d50: Array | float) -> Array:
    """Lapple grade efficiency ``1 / (1 + (d50/d)^2)`` (0.5 at ``d = d50``)."""
    d = _f(d)
    return 1.0 / (1.0 + (_f(d50) / d) ** 2)


def overall_efficiency(d_bins: Array, mass_fractions: Array, d50: Array | float) -> Array:
    """Overall mass efficiency ``sum_i w_i eta(d_i)`` over fixed size bins (``w`` normalized to 1)."""
    w = _f(mass_fractions)
    w = w / jnp.sum(w)
    return jnp.sum(w * lapple_efficiency(d_bins, d50))


def cyclone_pressure_drop(rho_g, v_i, H, W, D_e, K: float = 16.0, velocity_heads=None) -> Array:
    """Shepherd-Lapple pressure drop ``K (H W / D_e^2) rho_g v_i^2 / 2`` (Pa); ``K = 16`` is unverified, from memory.

    If ``velocity_heads`` is given the drop is that many inlet velocity heads ``N_H rho_g v_i^2 / 2`` instead.
    """
    head = 0.5 * _f(rho_g) * _f(v_i) ** 2
    if velocity_heads is not None:
        return _f(velocity_heads) * head
    return _f(K) * _f(H) * _f(W) / _f(D_e) ** 2 * head


# =============================================================================
# Units
# =============================================================================


def _slurry(feed: dict, solid: str, solid_MW: float, liquid: str, liquid_MW: float):
    """``(m_s, m_l)`` kg/s of dry solid and of the reference liquid in a slurry stream."""
    m_s = _f(feed[f"F_{solid}"]) * solid_MW
    m_l = _f(feed[f"F_{liquid}"]) * liquid_MW / 1000.0
    return m_s, m_l


def _split_outlets(feed: dict, solid: str, solid_MW: float, m_solid_a, frac_liquid_a, tag_a=None):
    """Split a slurry feed into two streams: ``a`` gets ``m_solid_a`` kg/s of solid and the fraction
    ``frac_liquid_a`` of every other species; ``b`` gets the rest. Returns ``(a, b)``."""
    ks = f"F_{solid}"
    a, b = {}, {}
    for k, v in feed.items():
        if not k.startswith("F_"):
            continue
        v = _f(v)
        if k == ks:
            a[k] = _f(m_solid_a) / solid_MW
            b[k] = v - a[k]
        else:
            a[k] = frac_liquid_a * v
            b[k] = v - a[k]
    for s in (a, b):
        s["T"], s["P"] = _f(feed["T"]), _f(feed["P"])
    return a, b


def _balance_error(feed: dict, *outs: dict) -> Array:
    err = jnp.asarray(0.0)
    for k, v in feed.items():
        if k.startswith("F_"):
            err = jnp.maximum(err, jnp.abs(_f(v) - sum(o[k] for o in outs)))
    return err


def _strictly_increasing(x) -> bool:
    """True if the (concrete) sequence ``x`` is strictly increasing; traced values pass unchecked."""
    try:
        a = np.asarray(x, dtype=float)
    except Exception:  # traced (jit/grad): the check is concrete-only
        return True
    return bool(np.all(np.diff(a) > 0))


@dataclass
class SettlerParams(ParamsMixin):
    """Parameters of a gravity settler / thickener.

    Attributes:
        underflow_concentration: Underflow solids concentration ``C_u`` (kg solid per m^3 of slurry).
        method: ``"overflow_rate"`` (clarifier, particle settling), ``"coe_clevenger"``
            (flux data ``C_data``/``v_data`` or the particle model), ``"talmage_fitch"``
            or ``"kynch"`` (batch interface curve ``t_data``/``z_data``).
        C_data: Concentrations (kg/m^3, ascending) of settling-velocity data.
        v_data: Hindered settling velocities (m/s) at ``C_data``.
        t_data: Times (s) of a batch interface-height curve.
        z_data: Interface heights (m) at ``t_data``.
        batch_C0: Initial concentration (kg/m^3) of the batch test (the feed concentration for Talmage-Fitch).
        d_p: Particle diameter (m): the design particle (overflow rate) or the particle-model size.
        rho_p: Particle (solid) density (kg/m^3).
        liquid_density: Liquid density (kg/m^3).
        viscosity: Liquid viscosity (Pa s).
        sphericity: Particle sphericity.
        richardson_zaki_n: Richardson-Zaki exponent; ``None`` selects the piecewise value from ``Re_t``.
        d_bins: Particle size bins (m) for the overflow-rate capture.
        mass_fractions: Mass fraction of each size bin.
        capture: Fraction of the solids sent to the underflow (thickener methods).
        safety_factor: Multiplier on the required area (design mode).
        area: Settler area (m^2) for rating; ``None`` designs it.
        sharpness: Smooth-maximum sharpness.
        n_grid: Concentration grid points for the particle model and tabulated flux data.
        solid_species: Solid pseudo-species (stream key ``F_<solid_species>``).
        solid_molar_mass: Molar mass of the solid pseudo-species (kg/mol).
        liquid_species: Liquid species that fixes the liquid mass flow.
        liquid_MW: Molar mass of the liquid species (g/mol).
    """

    underflow_concentration: float
    method: Literal["overflow_rate", "coe_clevenger", "talmage_fitch", "kynch"] = "coe_clevenger"
    C_data: list | None = None
    v_data: list | None = None
    t_data: list | None = None
    z_data: list | None = None
    batch_C0: float | None = None
    d_p: float | None = None
    rho_p: float = 2650.0
    liquid_density: float = 998.0
    viscosity: float = 1.0e-3
    sphericity: float = 1.0
    richardson_zaki_n: float | None = None
    d_bins: list | None = None
    mass_fractions: list | None = None
    capture: float = 1.0
    safety_factor: float = 1.0
    area: float | None = None
    sharpness: float = 1000.0
    n_grid: int = 100
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0
    liquid_species: str = "H2O"
    liquid_MW: float = 18.015

    def __post_init__(self):
        if self.method not in ("overflow_rate", "coe_clevenger", "talmage_fitch", "kynch"):
            raise ValueError("method must be 'overflow_rate', 'coe_clevenger', 'talmage_fitch' or 'kynch'")
        m = self.method
        if m == "overflow_rate":
            if self.d_p is None and self.d_bins is None:
                raise ValueError("overflow_rate needs d_p (design particle) and/or d_bins")
            if self.area is None and self.d_p is None:
                raise ValueError("overflow_rate design (area=None) needs the design particle d_p")
            if self.area is not None and self.d_bins is None and self.d_p is None:
                raise ValueError("overflow_rate rating needs d_bins")
        elif m == "coe_clevenger":
            if (self.C_data is None) == (self.d_p is None):
                raise ValueError("coe_clevenger needs either C_data/v_data or the particle model d_p (not both)")
            if self.C_data is not None and (self.v_data is None or len(self.v_data) != len(self.C_data)):
                raise ValueError("C_data and v_data must have the same length")
            if self.C_data is not None and not _strictly_increasing(self.C_data):
                raise ValueError("C_data must be strictly increasing (sort it and drop repeated "
                                 "concentrations, e.g. the constant-rate points of a Kynch construction)")
        else:
            if self.t_data is None or self.z_data is None or len(self.t_data) != len(self.z_data):
                raise ValueError(f"{m} needs t_data and z_data of equal length")
            if self.batch_C0 is None:
                raise ValueError(f"{m} needs batch_C0")
            if not _strictly_increasing(self.t_data):
                raise ValueError("t_data must be strictly increasing")
        if self.d_bins is not None and (self.mass_fractions is None
                                        or len(self.mass_fractions) != len(self.d_bins)):
            raise ValueError("d_bins and mass_fractions must have the same length")


class Settler:
    """Gravity settler: overflow-rate clarifier or continuous thickener (Coe-Clevenger, Talmage-Fitch, Kynch).

    ``Settler(params)(feed) -> (overflow, underflow, info)``. The feed carries the dry
    solid ``F_<solid_species>`` (kg/s with the default molar mass 1 kg/mol) and the liquid
    species (mol/s). The underflow is the slurry at ``C_u``; the overflow the clarified liquid
    plus any solids that are not captured.

    ``info``: ``area_required`` and ``area`` (m^2; design: ``safety_factor`` times the required),
    ``area_ratio`` (area over required, rating), ``limiting_concentration`` (kg/m^3) and
    ``limiting_flux`` (kg/m^2/s) for the thickener methods, ``overflow_rate`` (m/s),
    ``overflow_flow`` and ``underflow_flow`` (m^3/s), ``solids_capture`` (fraction to the underflow),
    ``C_feed`` and ``balance_error`` (largest species imbalance, mol/s).

    A thickener rated with ``area`` reports ``area_ratio``; only the overflow-rate mode changes the
    split with area (a thickener short of area is reported, not modelled). In the overflow-rate
    mode the overflow rate is based on the liquid overflow with full capture; the outlet
    flows follow from the capture and close the balance.
    """

    symbol = "Settler"
    equations = [
        r"A = F_s \max_{C_0\le C\le C_u}\frac{1/C - 1/C_u}{v(C)}\quad\text{(Coe-Clevenger)}",
        r"G_L = \min_C \frac{C\,v(C)}{1 - C/C_u},\qquad A = F_s/G_L\quad\text{(Kynch)}",
        r"A = \frac{Q_0\,t_u}{z_0},\qquad t_u = t + \frac{z - C_0 z_0/C_u}{v}\quad\text{(Talmage-Fitch)}",
        r"v(C) = v_t\,(1 - C/\rho_p)^{n}\quad\text{(Richardson-Zaki)}",
        r"A = \frac{SF\,Q_o}{v_t(d_p)},\qquad \eta_i = \min(1, v_{t,i}/v_o)",
    ]
    assumptions = [
        "Steady state; ideal (Hazen) clarification zone; a thickener captures the fraction 'capture' of the solids.",
        "The thickener area is the smooth maximum (log-sum-exp) of the Coe-Clevenger expression over a fixed grid; it over-predicts by at most ln(N)/sharpness of the scale.",
        "Talmage-Fitch and Kynch use the PCHIP tangents of the batch curve at its data nodes; the test must be at (or below) the feed concentration.",
        "Underflow volume from C_u and the solid density; non-solid species ride with the liquid. Solid volume counted in the feed and underflow slurry volumes.",
        "The Richardson-Zaki exponent is piecewise in Re_t unless richardson_zaki_n is given; a thickener optimized over particle size should pass n.",
        "Compression, rakes, scale-up factors and flocculation are not modelled; the user supplies a safety factor.",
    ]
    references = [
        "Coe, H.S. and Clevenger, G.H., Trans. AIME 55, 356 (1916).",
        "Kynch, G.J., Trans. Faraday Soc. 48, 166 (1952).",
        "Talmage, W.P. and Fitch, E.B., Ind. Eng. Chem. 47, 38 (1955).",
        "Richardson, J.F. and Zaki, W.N., Trans. Inst. Chem. Eng. 32, 35 (1954).",
        "McCabe, Smith and Harriott, Unit Operations of Chemical Engineering, Ch. 29.",
        "Coulson and Richardson, Chemical Engineering Vol. 2, Ch. 5.",
    ]
    parameter_symbols = {
        "underflow_concentration": "C_u", "rho_p": r"\rho_p", "viscosity": r"\mu",
        "d_p": "d_p", "safety_factor": "SF", "area": "A", "richardson_zaki_n": "n",
    }
    parameter_units = {
        "underflow_concentration": "kg/m^3", "C_data": "kg/m^3", "v_data": "m/s", "t_data": "s",
        "z_data": "m", "batch_C0": "kg/m^3", "d_p": "m", "rho_p": "kg/m^3",
        "liquid_density": "kg/m^3", "viscosity": "Pa s", "sphericity": "-",
        "richardson_zaki_n": "-", "d_bins": "m", "mass_fractions": "-", "capture": "-",
        "safety_factor": "-", "area": "m^2", "sharpness": "-", "n_grid": "-",
        "solid_molar_mass": "kg/mol", "liquid_MW": "g/mol",
    }
    numerical_method = "Closed-form areas; log-sum-exp smooth maximum over a fixed grid; PCHIP tangents of the batch curve."
    outlet_roles = ["overflow", "underflow"]

    def __init__(self, params: SettlerParams):
        self.params = params

    def _v_grid(self, C0, C_u):
        p = self.params
        if p.C_data is not None:
            C = _grid(C0, C_u, p.n_grid)
            Cd, vd = _f(p.C_data), _f(p.v_data)
            v = _pchip_eval(Cd, vd, _pchip_slopes(Cd, vd), C)
            return C, jnp.maximum(v, _V_FLOOR)
        C = _grid(C0, C_u, p.n_grid)
        v = hindered_velocity_model(C, p.d_p, p.rho_p, p.liquid_density, p.viscosity, p.sphericity,
                                    p.richardson_zaki_n)
        return C, v

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        """Settle the feed.

        Args:
            feed: Slurry stream with ``F_<solid_species>``, ``F_<liquid_species>`` (mol/s), ``T``, ``P``.

        Returns:
            overflow: clarified liquid (plus uncaptured solids).
            underflow: thickened slurry at ``underflow_concentration``.
            info: see the class docstring.
        """
        p = self.params
        Ms = p.solid_molar_mass
        m_s, m_l = _slurry(feed, p.solid_species, Ms, p.liquid_species, p.liquid_MW)
        rho_l, rho_s, C_u = _f(p.liquid_density), _f(p.rho_p), _f(p.underflow_concentration)
        Q0 = m_l / rho_l + m_s / rho_s
        C0 = m_s / Q0
        info = {"C_feed": C0, "feed_flow": Q0}
        # liquid with the underflow if all the solids were captured: the nominal overflow flow
        liq_u_full = rho_l * (m_s / C_u - m_s / rho_s)
        Q_o_nom = (m_l - liq_u_full) / rho_l

        if p.method == "overflow_rate":
            if p.area is None:
                A_req = overflow_rate_area(Q_o_nom, p.d_p, rho_s, rho_l, p.viscosity, p.sphericity, 1.0)
                A = _f(p.safety_factor) * A_req
            else:
                A = _f(p.area)
                A_req = jnp.nan * A if p.d_p is None else overflow_rate_area(
                    Q_o_nom, p.d_p, rho_s, rho_l, p.viscosity, p.sphericity, 1.0)
            v_o = Q_o_nom / A
            if p.d_bins is not None:
                w = _f(p.mass_fractions)
                w = w / jnp.sum(w)
                eta_bins = overflow_rate_capture(v_o, p.d_bins, rho_s, rho_l, p.viscosity, p.sphericity)
                capture = jnp.sum(w * eta_bins)
                info["grade_efficiency"] = eta_bins
            elif p.d_p is not None:
                capture = overflow_rate_capture(v_o, jnp.asarray([p.d_p]), rho_s, rho_l, p.viscosity,
                                                p.sphericity)[0]
            else:  # unreachable (validated)
                capture = _f(1.0)
            info["overflow_rate"] = v_o
            info["terminal_velocity_design"] = (terminal_velocity(p.d_p, rho_s, rho_l, p.viscosity,
                                                                  p.sphericity) if p.d_p is not None else jnp.nan)
        else:
            capture = _f(p.capture)
            if p.method == "coe_clevenger":
                C, v = self._v_grid(C0, C_u)
                A_req, det = thickener_area_coe_clevenger(Q0, C0, C_u, C, v, p.sharpness, True)
                info["limiting_concentration"] = det["limiting_concentration"]
                info["limiting_flux"] = det["limiting_flux"]
            elif p.method == "kynch":
                A_req = thickener_area_kynch(m_s, C_u, p.t_data, p.z_data, p.batch_C0, p.sharpness)
                info["limiting_flux"] = m_s / A_req
            else:
                z0 = _f(p.z_data)[0]
                A_req, det = thickener_area_talmage_fitch(
                    m_s / _f(p.batch_C0), p.batch_C0, z0, C_u, p.t_data, p.z_data, p.sharpness, True)
                info["t_u"] = det["t_u"]
                info["limiting_flux"] = m_s / A_req
            A = _f(p.safety_factor) * A_req if p.area is None else _f(p.area)
            info["overflow_rate"] = Q_o_nom / A

        m_u = capture * m_s
        Q_u = m_u / C_u
        liq_u = rho_l * (Q_u - m_u / rho_s)
        frac_u = liq_u / m_l
        underflow, overflow = _split_outlets(feed, p.solid_species, Ms, m_u, frac_u)
        info.update(
            area_required=A_req, area=A, area_ratio=A / A_req, solids_capture=capture,
            underflow_flow=Q_u, overflow_flow=(m_l - liq_u) / rho_l + (m_s - m_u) / rho_s,
            underflow_liquid_fraction=frac_u,
            balance_error=_balance_error(feed, overflow, underflow),
        )
        return overflow, underflow, info


@dataclass
class CakeFilterParams(ParamsMixin):
    """Parameters of a cake filter (Ruth, constant pressure).

    Attributes:
        delta_P: Filtration pressure drop (Pa).
        mode: ``"batch"`` (press, leaf: filtration, wash, downtime) or ``"rotary"`` (drum).
        area: Filter area (m^2) for rating; ``None`` designs it from the feed and ``cycle_time``.
        alpha: Specific cake resistance (m/kg).
        alpha0: Compressible-cake coefficient: ``alpha = alpha0 delta_P^s`` (SI).
        s: Cake compressibility exponent (0 = incompressible).
        d_p: Particle diameter (m) for the Kozeny-Carman specific resistance.
        voidage: Cake voidage (Kozeny-Carman and cake thickness).
        rho_p: Solid density (kg/m^3); with ``voidage`` it gives the cake thickness.
        sphericity: Particle sphericity (Kozeny-Carman).
        k0: Kozeny constant.
        R_m: Filter-medium resistance (1/m).
        viscosity: Filtrate viscosity (Pa s).
        liquid_density: Filtrate density (kg/m^3).
        cake_moisture: Liquid retained in the discharged cake (kg liquid per kg dry cake).
        filtration_time: Filtration time per cycle (s), rating of a batch filter.
        optimize_cycle: Rating a batch filter: use the throughput-maximizing filtration time.
        cycle_time: Total cycle time (s): the design value (batch) or the drum revolution time (rotary).
        wash_ratio: Wash volume per filtrate volume.
        wash_rate_factor: Wash rate over the final filtration rate (1 = standard assumption).
        downtime: Downtime per cycle (s): discharge, cleaning, reassembly.
        submergence: Fraction of a drum revolution spent forming cake (rotary).
        solid_species: Solid pseudo-species (stream key ``F_<solid_species>``).
        solid_molar_mass: Molar mass of the solid pseudo-species (kg/mol).
        liquid_species: Liquid species that fixes the liquid mass flow.
        liquid_MW: Molar mass of the liquid species (g/mol).
    """

    delta_P: float
    mode: Literal["batch", "rotary"] = "batch"
    area: float | None = None
    alpha: float | None = None
    alpha0: float | None = None
    s: float = 0.0
    d_p: float | None = None
    voidage: float | None = None
    rho_p: float | None = None
    sphericity: float = 1.0
    k0: float = 5.0
    R_m: float = 0.0
    viscosity: float = 1.0e-3
    liquid_density: float = 998.0
    cake_moisture: float = 0.0
    filtration_time: float | None = None
    optimize_cycle: bool = False
    cycle_time: float | None = None
    wash_ratio: float = 0.0
    wash_rate_factor: float = 1.0
    downtime: float = 0.0
    submergence: float = 0.3
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0
    liquid_species: str = "H2O"
    liquid_MW: float = 18.015

    def __post_init__(self):
        if self.mode not in ("batch", "rotary"):
            raise ValueError("mode must be 'batch' or 'rotary'")
        n_src = (self.alpha is not None) + (self.alpha0 is not None) + (self.d_p is not None)
        if n_src != 1:
            raise ValueError("give exactly one of alpha, alpha0 (with s) or d_p (with voidage)")
        if self.d_p is not None and self.voidage is None:
            raise ValueError("d_p needs voidage and rho_p (Kozeny-Carman)")
        if self.d_p is not None and self.rho_p is None:
            raise ValueError("d_p needs voidage and rho_p (Kozeny-Carman)")
        if self.area is None and self.cycle_time is None:
            raise ValueError("design (area=None) needs cycle_time")
        if self.mode == "batch":
            if self.area is not None:
                if self.cycle_time is not None:
                    raise ValueError("rating (area given) takes filtration_time, not cycle_time")
                if self.filtration_time is None and not self.optimize_cycle:
                    raise ValueError("batch rating needs filtration_time or optimize_cycle=True")
            elif self.filtration_time is not None:
                raise ValueError("batch design (area=None) takes cycle_time, not filtration_time")
        else:
            if self.cycle_time is None:
                raise ValueError("rotary mode needs cycle_time (the revolution time)")
            if self.wash_ratio != 0.0 or self.downtime != 0.0:
                raise ValueError("rotary mode has no wash volume or downtime in the time budget; use wash_ratio=0")


class CakeFilter:
    """Cake filter: Ruth constant-pressure filtration of a slurry, batch cycle or rotary drum.

    ``CakeFilter(params)(slurry) -> (cake, filtrate, info)``. The slurry carries the dry solid
    ``F_<solid_species>`` (kg/s with the default molar mass 1 kg/mol) and the liquid species
    (mol/s); the stream is the time-averaged feed. The cake carries all the solid and
    ``cake_moisture`` kg of liquid per kg dry solid (every non-solid species in the
    liquid proportion); the filtrate is the rest.

    Design (``area=None``): the area for which the cycle of ``cycle_time`` (filtration + wash + downtime;
    drum revolution) passes the feed, in closed form. Rating (``area`` given): the cycle's filtrate volume
    and throughput; ``utilization`` is the feed over the capacity (the streams follow the feed).

    ``info``: ``area``, ``K_p``, ``B``, ``alpha``, ``c_s`` (kg/m^3 filtrate), ``V_cycle`` (m^3),
    ``t_filtration``, ``t_wash``, ``t_down``, ``t_cycle`` (s), ``wash_volume``, ``throughput``
    (filtrate capacity of the cycle, m^3/s), ``solids_capacity`` (kg/s), ``utilization``, ``final_rate``
    (m^3/s), ``cake_mass_per_cycle`` (kg), ``cake_thickness`` (m, with ``voidage`` and ``rho_p``),
    ``filtrate_flow`` (m^3/s), ``area_for_cost`` (m^2) and ``balance_error``.
    """

    symbol = "CakeFilter"
    equations = [
        r"\frac{t}{V} = \frac{K_p}{2}V + B,\quad K_p = \frac{\mu\,\alpha\,c_s}{A^2\,\Delta P},\quad B = \frac{\mu R_m}{A\,\Delta P}",
        r"V = \frac{2t}{B + \sqrt{B^2 + 2K_p t}}",
        r"\Delta P = \frac{\mu q}{A}\left(\frac{\alpha c_s q t}{A} + R_m\right)\quad\text{(constant rate)}",
        r"\text{throughput} = \frac{V}{t_f + t_w + t_d},\quad t_w = \frac{r V}{\phi\,(dV/dt)_f}",
        r"c_s = \frac{\rho_f w}{1 - m w},\quad \alpha_{KC} = \frac{36 k_0 (1-\varepsilon)}{\varepsilon^3 \rho_p (\phi d_p)^2}",
    ]
    assumptions = [
        "Constant pressure drop; incompressible cake (or alpha = alpha0 dP^s evaluated at the filtration dP); no medium blinding.",
        "c_s follows from the slurry composition and the cake moisture by a mass balance; the filtrate density is a constant.",
        "Wash at phi times the final filtration rate (phi = 1 unless set); the wash liquor is not a stream: wash volume and time are reported, cake_moisture is the post-wash value.",
        "Rotary drum: the cake forms during the submerged fraction of each revolution at constant dP; no wash or dewatering time in the budget; the area is the whole drum surface.",
        "Streams are time-averaged; a rated filter that is over-loaded (utilization > 1) still passes the feed and reports it.",
    ]
    references = [
        "Ruth, B.F., Ind. Eng. Chem. 27, 708 (1935).",
        "Geankoplis, Transport Processes and Separation Process Principles, Sec. 14.2.",
        "McCabe, Smith and Harriott, Unit Operations of Chemical Engineering, Ch. 30.",
    ]
    parameter_symbols = {
        "delta_P": r"\Delta P", "alpha": r"\alpha", "R_m": "R_m", "viscosity": r"\mu",
        "area": "A", "cake_moisture": "X_c", "wash_ratio": "r", "wash_rate_factor": r"\phi",
        "downtime": "t_d", "submergence": r"\psi",
    }
    parameter_units = {
        "delta_P": "Pa", "area": "m^2", "alpha": "m/kg", "alpha0": "SI", "s": "-", "d_p": "m",
        "voidage": "-", "rho_p": "kg/m^3", "sphericity": "-", "k0": "-", "R_m": "1/m",
        "viscosity": "Pa s", "liquid_density": "kg/m^3", "cake_moisture": "kg/kg",
        "filtration_time": "s", "cycle_time": "s", "wash_ratio": "-", "wash_rate_factor": "-",
        "downtime": "s", "submergence": "-", "solid_molar_mass": "kg/mol", "liquid_MW": "g/mol",
    }
    numerical_method = "Closed-form Ruth quadratic, cycle area quadratic and optimum cycle (no iteration)."
    outlet_roles = ["cake", "filtrate"]

    def __init__(self, params: CakeFilterParams):
        self.params = params

    def specific_resistance(self) -> Array:
        """The specific cake resistance alpha (m/kg) the parameters describe."""
        p = self.params
        if p.alpha is not None:
            return _f(p.alpha)
        if p.alpha0 is not None:
            return compressible_cake(p.alpha0, p.delta_P, p.s)
        return specific_cake_resistance(p.d_p, p.voidage, p.rho_p, p.sphericity, p.k0)

    def __call__(self, slurry: Stream) -> tuple[Stream, Stream, dict]:
        """Filter the slurry.

        Args:
            slurry: Stream with ``F_<solid_species>``, ``F_<liquid_species>`` (mol/s), ``T``, ``P``.

        Returns:
            cake: all the solid plus the retained liquid.
            filtrate: the liquid that passes.
            info: see the class docstring.
        """
        p = self.params
        Ms = p.solid_molar_mass
        m_s, m_l = _slurry(slurry, p.solid_species, Ms, p.liquid_species, p.liquid_MW)
        rho_f, mu, dP = _f(p.liquid_density), _f(p.viscosity), _f(p.delta_P)
        X = _f(p.cake_moisture)
        liq_ret = X * m_s
        Q_f = (m_l - liq_ret) / rho_f                       # filtrate flow, m^3/s
        c_s = m_s / Q_f                                      # kg solid per m^3 filtrate
        alpha = self.specific_resistance()
        r, phi, t_d = _f(p.wash_ratio), _f(p.wash_rate_factor), _f(p.downtime)
        rotary = p.mode == "rotary"
        c2 = 1.0 if rotary else 1.0 + 2.0 * r / phi
        c1 = 1.0 if rotary else 1.0 + r / phi
        info = {}

        if p.area is None:
            T_c = _f(p.cycle_time)
            V = Q_f * T_c
            t_on = _f(p.submergence) * T_c if rotary else T_c - t_d
            A = filter_area_for_cycle(V, t_on, mu, alpha, c_s, dP, p.R_m, c2, c1)
            K_p, B = ruth_constants(A, dP, mu, alpha, c_s, p.R_m)
            t_f = ruth_constant_pressure(V, K_p, B)
        else:
            A = _f(p.area)
            K_p, B = ruth_constants(A, dP, mu, alpha, c_s, p.R_m)
            if rotary:
                T_c = _f(p.cycle_time)
                t_f = _f(p.submergence) * T_c
            elif p.optimize_cycle:
                opt = optimal_cycle(K_p, B, t_d, r, phi)
                t_f = opt["t_f"]
            else:
                t_f = _f(p.filtration_time)
            V = filtrate_volume(t_f, K_p, B)
        final_rate = filtrate_rate(V, K_p, B)
        V_w = r * V
        t_w = wash_time(V_w, final_rate, phi)
        if rotary:
            t_cycle = _f(p.cycle_time)
            t_down = _f(0.0)
        else:
            t_down = t_d
            t_cycle = t_f + t_w + t_down
        throughput = V / t_cycle
        frac_ret = liq_ret / m_l
        cake, filtrate = _split_outlets(slurry, p.solid_species, Ms, m_s, frac_ret)
        info.update(
            area=A, K_p=K_p, B=B, alpha=alpha, c_s=c_s, V_cycle=V, t_filtration=t_f, t_wash=t_w,
            t_down=t_down, t_cycle=t_cycle, wash_volume=V_w, throughput=throughput,
            solids_capacity=throughput * c_s, utilization=Q_f / throughput, final_rate=final_rate,
            cake_mass_per_cycle=V * c_s, filtrate_flow=Q_f, cake_moisture=X, mode=p.mode,
            area_for_cost=A, balance_error=_balance_error(slurry, cake, filtrate),
        )
        if p.voidage is not None and p.rho_p is not None:
            info["cake_thickness"] = V * c_s / (A * _f(p.rho_p) * (1.0 - _f(p.voidage)))
        return cake, filtrate, info


@dataclass
class CycloneParams(ParamsMixin):
    """Parameters of a gas cyclone (Lapple).

    Attributes:
        D: Body diameter (m).
        rho_p: Particle density (kg/m^3).
        d_bins: Particle size bins (m), fixed length.
        mass_fractions: Mass fraction of the dust in each bin (normalized to 1).
        geometry: ``"lapple"`` or ``"stairmand_he"`` standard proportions, or ``"custom"``.
        ratios: ``[H, W, D_e, S, L_b, L_c, J] / D`` for ``"custom"``.
        n_turns: Effective number of turns ``N_e``; ``None`` is the Lapple estimate ``(L_b + L_c/2)/H``.
        gas_MW: Gas molar mass (g/mol).
        viscosity: Gas viscosity (Pa s), at the operating temperature.
        K_pressure: Shepherd-Lapple constant (16 for a tangential inlet; unverified).
        delta_P_velocity_heads: Pressure drop as this many inlet velocity heads, instead of Shepherd-Lapple.
        solid_species: Solid pseudo-species (stream key ``F_<solid_species>``).
        solid_molar_mass: Molar mass of the solid pseudo-species (kg/mol).
    """

    D: float
    rho_p: float
    d_bins: list
    mass_fractions: list
    geometry: Literal["lapple", "stairmand_he", "custom"] = "lapple"
    ratios: list | None = None
    n_turns: float | None = None
    gas_MW: float = 28.96
    viscosity: float = 1.8e-5
    K_pressure: float = 16.0
    delta_P_velocity_heads: float | None = None
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0

    def __post_init__(self):
        if self.geometry not in ("lapple", "stairmand_he", "custom"):
            raise ValueError("geometry must be 'lapple', 'stairmand_he' or 'custom'")
        if self.geometry == "custom" and (self.ratios is None or len(self.ratios) != 7):
            raise ValueError("custom geometry needs 7 ratios [H, W, D_e, S, L_b, L_c, J] / D")
        if len(self.d_bins) != len(self.mass_fractions):
            raise ValueError("d_bins and mass_fractions must have the same length")


class Cyclone:
    """Gas cyclone with the Lapple cut size and grade efficiency.

    ``Cyclone(params)(gas_with_solids) -> (clean_gas, solids, info)``. The inlet is a gas stream
    (mol/s per species) that also carries the dust as ``F_<solid_species>`` (kg/s with the default
    molar mass 1 kg/mol): the stream convention of the dryer's ``info["fines"]``, which
    :func:`dust_laden_gas` merges into the dryer's ``air_out``. The size distribution is a unit
    parameter (fixed bins). ``solids`` is the collected dust (only ``F_<solid_species>``); ``clean_gas`` the gas
    and the dust that escapes.

    ``info``: ``d50`` (m), ``grade_efficiency`` (per bin), ``overall_efficiency``, ``inlet_velocity``
    (m/s), ``delta_P`` (Pa), ``gas_flow`` (m^3/s), ``gas_density`` (kg/m^3), ``dust_loading`` (kg/m^3),
    ``N_e``, ``dimensions`` and ``balance_error``.
    """

    symbol = "Cyclone"
    equations = [
        r"d_{50} = \left[\frac{9\mu W}{2\pi N_e v_i(\rho_p - \rho_g)}\right]^{1/2}",
        r"\eta(d) = \frac{1}{1 + (d_{50}/d)^2},\qquad \eta_{tot} = \sum_i w_i\,\eta(d_i)",
        r"\Delta P = K\,\frac{H W}{D_e^2}\,\frac{\rho_g v_i^2}{2},\qquad v_i = \frac{Q}{H W}",
        r"N_e = \frac{1}{H}\left(L_b + \frac{L_c}{2}\right)",
    ]
    assumptions = [
        "Lapple's theory: uniform flow in the inlet, N_e turns of a particle in the outer vortex, a particle is captured if it reaches the wall in the residence time.",
        "Standard proportions, K = 16 and the efficiency curve are empirical and typed from memory (unverified); mass-loading and re-entrainment effects are not included.",
        "Ideal gas for the volumetric flow; gas molar mass and viscosity are parameters (not computed from composition).",
        "Gas flows to the clean gas outlet; the dust is split by the grade efficiency; the size distribution is a fixed set of bins (a unit parameter, not a stream property).",
    ]
    references = [
        "Lapple, C.E., Chem. Eng. 58(5), 144 (1951); Shepherd, C.B. and Lapple, C.E., Ind. Eng. Chem. 31, 972 (1939) (recalled).",
        "Cooper, C.D. and Alley, F.C., Air Pollution Control: A Design Approach, Ch. 4.",
        "Coulson and Richardson, Chemical Engineering Vol. 2, Ch. 1.",
        "Towler and Sinnott, Chemical Engineering Design, Ch. 10 (Stairmand design).",
    ]
    parameter_symbols = {
        "D": "D", "rho_p": r"\rho_p", "n_turns": "N_e", "viscosity": r"\mu", "K_pressure": "K",
        "delta_P_velocity_heads": "N_H",
    }
    parameter_units = {
        "D": "m", "rho_p": "kg/m^3", "d_bins": "m", "mass_fractions": "-", "ratios": "-",
        "n_turns": "-", "gas_MW": "g/mol", "viscosity": "Pa s", "K_pressure": "-",
        "delta_P_velocity_heads": "-", "solid_molar_mass": "kg/mol",
    }
    numerical_method = "Closed-form Lapple relations."
    outlet_roles = ["clean_gas", "solids"]

    def __init__(self, params: CycloneParams):
        self.params = params

    def __call__(self, gas: Stream) -> tuple[Stream, Stream, dict]:
        """Separate the dust from the gas.

        Args:
            gas: Stream with the gas species ``F_*`` (mol/s), the dust ``F_<solid_species>``
                (kg/s with the default molar mass), ``T`` and ``P``.

        Returns:
            clean_gas: the gas plus the escaping dust.
            solids: the collected dust.
            info: see the class docstring.
        """
        p = self.params
        ks = f"F_{p.solid_species}"
        Ms = p.solid_molar_mass
        F_gas = sum(_f(v) for k, v in gas.items() if k.startswith("F_") and k != ks)
        T, P = _f(gas["T"]), _f(gas["P"])
        Q = F_gas * R_GAS * T / P
        rho_g = F_gas * p.gas_MW / 1000.0 / Q
        dims = cyclone_geometry(p.D, p.geometry, p.ratios)
        N_e = lapple_turns(dims["H"], dims["L_b"], dims["L_c"]) if p.n_turns is None else _f(p.n_turns)
        v_i = Q / (dims["H"] * dims["W"])
        d50 = lapple_cut_size(p.viscosity, dims["W"], N_e, v_i, p.rho_p, rho_g)
        eta_bins = lapple_efficiency(p.d_bins, d50)
        eta = overall_efficiency(p.d_bins, p.mass_fractions, d50)
        dP = cyclone_pressure_drop(rho_g, v_i, dims["H"], dims["W"], dims["D_e"], p.K_pressure,
                                   p.delta_P_velocity_heads)
        m_dust = _f(gas[ks]) * Ms if ks in gas else _f(0.0)
        collected = eta * m_dust
        solids, clean = {}, {}
        for k, v in gas.items():
            if not k.startswith("F_"):
                continue
            v = _f(v)
            if k == ks:
                solids[k] = collected / Ms
                clean[k] = v - solids[k]
            else:
                solids[k] = 0.0 * v
                clean[k] = v
        if ks not in gas:
            solids[ks] = _f(0.0)
            clean[ks] = _f(0.0)
        for s in (solids, clean):
            s["T"], s["P"] = T, P
        # the collected dust leaves at the inlet pressure; the pressure drop is reported, not applied
        info = {
            "d50": d50, "grade_efficiency": eta_bins, "overall_efficiency": eta, "inlet_velocity": v_i,
            "delta_P": dP, "gas_flow": Q, "gas_density": rho_g, "dust_loading": m_dust / Q, "N_e": N_e,
            "dimensions": dims, "collected": collected,
            "balance_error": _balance_error(gas if ks in gas else {**gas, ks: _f(0.0)}, clean, solids),
        }
        return clean, solids, info


def dust_laden_gas(gas: dict, fines: dict, solid_species: str = "solid") -> dict:
    """Merge a gas stream and an entrained-solids stream into a cyclone inlet (the Dryer to Cyclone hand-off).

    The dryer puts the entrained fraction ``entrainment`` of its dry solids in ``info["fines"]``
    (a solids stream ``{F_solid, F_H2O, ...}``, ``F_solid`` in kg/s with the default molar mass)
    and not in ``air_out``. This adds every ``F_*`` of ``fines`` to ``gas`` (the fines' water becomes vapor
    in the gas) and keeps the gas ``T`` and ``P``: ``Cyclone(params)(dust_laden_gas(air_out, info["fines"]))``.
    Pure JAX; the species of the result are the union of the two streams.
    """
    out = {k: _f(v) for k, v in gas.items()}
    for k, v in fines.items():
        if k.startswith("F_"):
            out[k] = out.get(k, _f(0.0)) + _f(v)
    return out
