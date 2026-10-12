"""Drying: moisture bookkeeping, equilibrium isotherms, drying-time functions and the ``Dryer`` unit.

Library (pure JAX, SI, differentiable)
--------------------------------------
Moisture contents are on the **dry basis** ``X`` (kg water per kg bone-dry solid);
:func:`dry_basis`, :func:`wet_basis` and :func:`free_moisture` convert. The
*equilibrium moisture* ``X_eq`` is what the solid holds at the water activity
(``a_w``, the relative humidity of the air in equilibrium with it) of the drying
air, from a GAB or BET isotherm or a tabulated one
(:func:`equilibrium_moisture`, inverse :func:`water_activity`). The *free
moisture* ``X - X_eq`` is what can be removed.

The drying rate ``R`` (kg water per m^2 of drying surface per s) of a solid
with loading ``Ls/A`` (kg dry solid per m^2 of exposed surface) follows the
classical curve::

    R = R_c                                   X >= X_c       constant-rate period
    R = R_c (X - X*) / (X_c - X*)             X <  X_c       falling rate (linear)

with the constant rate set by heat transfer from the air to a surface at the
wet-bulb temperature, ``R_c = h (T - T_wb) / lambda_wb``
(:func:`constant_drying_rate`; ``h`` from :func:`heat_transfer_coefficient_tray`
or :mod:`difflow.heat_transfer`) and ``X* = X_eq`` (:func:`drying_rate_curve`).
The drying time from ``X1`` to ``X2`` is the sum of the two periods
(:func:`drying_time`, :func:`drying_periods`)::

    t_c = (Ls/A) (X1 - X_c) / R_c
    t_f = (Ls/A) (X_c - X*) / R_c  ln[(X_c - X*) / (X2 - X*)]      (linear falling rate)
    t_f = (Ls/A) int_{X2}^{X_c} dX / R(X)                          (tabulated rate curve)

**The smooth switch.** Which period a moisture falls in is decided by ``X_c``,
so ``X1 > X_c > X2``, ``X1 < X_c`` and ``X2 > X_c`` are three branches. They are
written as one expression with no Python ``if``: with the softplus
``sp(z) = s ln(1 + exp(z/s))`` (a smooth ``max(z, 0)``) the endpoints are
clipped against ``X_c``::

    t_c-range = sp(X1 - X_c) - sp(X2 - X_c)
    X_a = X1 - sp(X1 - X_c)            (smooth min(X1, X_c): where the falling period starts)
    X_b = X_a - sp(X_a - X2)           (smooth min(X2, X_a): where it ends)

and the falling-rate integral runs from ``X_b`` to ``X_a``. The smoothing width
``s = (X_c - X_eq) / sharpness`` is a fraction of the free-moisture range at
the critical point, set by ``sharpness`` (default 200: the result is within
``exp(-200 |X - X_c| / (X_c - X_eq))`` of the exact piecewise answer away from
the corner; exactly at the corner the first-order offsets of the two periods cancel, leaving a
second-order error, about 5e-6 of the time for the default on the test problem). The time and
its gradient with respect to ``X1``, ``X2``, ``X_c``, ``X_eq``, ``R_c`` and
``Ls/A`` therefore exist everywhere; ``sharpness -> inf`` recovers the piecewise
answer. A moisture at or below ``X_eq`` has no finite drying time; its
``X - X_eq`` is clipped at 1e-12 so the time is huge but finite, not NaN.

Streams
-------
All streams of this module are in **mol/s**, K, Pa, like the rest of difflow.

* Air is the humid-air stream of :mod:`difflow.psychrometrics` (``F_H2O`` vapor,
  ``F_air`` dry air).
* Solids are a stream with the dry-solid pseudo-species ``F_<solid_species>``
  (default ``F_solid``) and ``F_H2O`` for the water held. The dry solid has no
  molar mass, so ``DryerParams.solid_molar_mass`` (kg/mol, default 1) converts:
  with the default, ``F_solid`` is numerically **kg/s** of dry solid, while
  ``F_H2O`` is mol/s like every other water stream (the moisture content is
  ``X = F_H2O M_w / (F_solid M_s)``). This differs from the all-kg/s ``Leacher``.
  Keeping water in mol/s lets the dryer's water chain into pumps, condensers and
  the air streams without a conversion.

References
----------
- Geankoplis, Transport Processes and Separation Process Principles, Ch. 9
  (equations 9.6-x; the tray-dryer correlations are recalled, see
  :func:`heat_transfer_coefficient_tray`).
- McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 24.
- Mujumdar (ed.), Handbook of Industrial Drying (GAB / BET isotherm data).
- Brunauer, Emmett & Teller, JACS 60, 309 (1938); Guggenheim-Anderson-de Boer (GAB).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.psychrometrics import (
    CP_DRY_AIR,
    CP_VAPOR,
    MW_WATER,
    P_ATM,
    dry_air_flow,
    air_stream_from_state,
    humid_enthalpy,
    humid_temperature,
    latent_heat,
    rh_from_humidity_ratio,
    saturation_humidity_ratio,
    state,
    state_from_air_stream,
    wet_bulb_temperature,
)
from difflow.staged import _pchip_eval, _pchip_slopes, _solve_monotone
from difflow.streams import Stream
from difflow.water import T_REF

#: Universal gas constant, J/mol/K (temperature dependence of the isotherm constants)
R_GAS = 8.314462618

_TINY = 1e-12


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


# =============================================================================
# Moisture bookkeeping
# =============================================================================


def dry_basis(w_wet: Array | float) -> Array:
    """Dry-basis moisture ``X = w / (1 - w)`` (kg water / kg dry solid) from the wet-basis mass fraction ``w``."""
    w = _f(w_wet)
    return w / (1.0 - w)


def wet_basis(X: Array | float) -> Array:
    """Wet-basis moisture ``w = X / (1 + X)`` (kg water / kg wet solid) from the dry-basis ``X``."""
    X = _f(X)
    return X / (1.0 + X)


def free_moisture(X: Array | float, X_eq: Array | float) -> Array:
    """Free moisture ``X - X_eq`` (kg/kg dry solid): the part that can be removed in the given air."""
    return _f(X) - _f(X_eq)


def solids_moisture(stream: dict, solid: str = "solid", solid_molar_mass: float = 1.0) -> Array:
    """Dry-basis moisture ``X`` of a solids stream (``F_<solid>`` and ``F_H2O`` in mol/s)."""
    m_s = _f(stream[f"F_{solid}"]) * solid_molar_mass
    m_w = _f(stream.get("F_H2O", 0.0)) * MW_WATER
    return m_w / jnp.where(m_s > 0.0, m_s, 1.0)


# =============================================================================
# Equilibrium moisture isotherms
# =============================================================================


def _gab(a_w, X_m, C, K):
    # a_w K is kept below 1 (the GAB isotherm diverges at a_w = 1/K)
    Ka = jnp.clip(K * jnp.maximum(a_w, 0.0), 0.0, 1.0 - 1e-9)
    return X_m * C * Ka / ((1.0 - Ka) * (1.0 - Ka + C * Ka))


def _gab_constants(params: dict, T):
    """``C`` and ``K`` at temperature ``T``: ``C = C0 exp(Hc / R T)``, ``K = K0 exp(Hk / R T)``."""
    C = _f(params["C"]) * jnp.exp(_f(params.get("Hc", 0.0)) / (R_GAS * _f(T)))
    K = _f(params.get("K", 1.0)) * jnp.exp(_f(params.get("Hk", 0.0)) / (R_GAS * _f(T)))
    return C, K


def isotherm_params(model: str, params) -> dict:
    """Normalise the isotherm specification used by :class:`DryerParams`.

    Args:
        model: ``"gab"``, ``"bet"`` or ``"table"``.
        params: ``[X_m, C, K]`` (GAB), ``[X_m, C]`` (BET) or
            ``[[a_w ...], [X ...]]`` (a table, ``a_w`` increasing).

    Returns:
        The ``params`` dict for :func:`equilibrium_moisture`.
    """
    if model == "gab":
        return {"X_m": params[0], "C": params[1], "K": params[2]}
    if model == "bet":
        return {"X_m": params[0], "C": params[1]}
    if model == "table":
        return {"a_w": params[0], "X": params[1]}
    raise ValueError("isotherm model must be 'gab', 'bet' or 'table'")


def equilibrium_moisture(a_w: Array | float, T: Array | float = 298.15, model="gab",
                         params: dict | None = None) -> Array:
    """Equilibrium moisture ``X_eq(a_w, T)`` (kg water / kg dry solid) of a solid.

    Models (``params`` is data, never built in; take it from the literature for
    your solid, e.g. Mujumdar, *Handbook of Industrial Drying*):

    ``"gab"`` (Guggenheim-Anderson-de Boer), ``params = {"X_m", "C", "K"}``::

        X = X_m C K a / [(1 - K a)(1 - K a + C K a)]

    ``"bet"`` is GAB with ``K = 1`` (``params = {"X_m", "C"}``), valid to
    ``a_w`` of about 0.4. The monolayer moisture ``X_m`` is the one parameter
    that does not depend on temperature here. The optional keys ``Hc`` and
    ``Hk`` (J/mol, default 0) give the usual temperature dependence
    ``C = C0 exp(Hc / R T)``, ``K = K0 exp(Hk / R T)`` (``C`` and ``K`` are then
    the pre-exponential factors); without them the isotherm is
    temperature-independent and ``T`` is ignored.

    ``"table"``: ``params = {"a_w": [...], "X": [...]}``, monotone (PCHIP)
    interpolation, linear extrapolation at the ends (``X`` clipped at 0).

    A callable ``model(a_w, T)`` is used as is.

    Args:
        a_w: Water activity (0-1), the equilibrium relative humidity.
        T: Temperature (K).
        model: ``"gab"``, ``"bet"``, ``"table"`` or a callable.
        params: Model data, see above.

    Returns:
        ``X_eq`` (kg/kg dry solid). For GAB, ``K a_w`` is limited to ``1 - 1e-9``.
    """
    a = _f(a_w)
    if callable(model):
        return _f(model(a, _f(T)))
    if params is None:
        raise ValueError("equilibrium_moisture needs params for model " + str(model))
    if model == "gab":
        C, K = _gab_constants(params, T)
        return _gab(a, _f(params["X_m"]), C, K)
    if model == "bet":
        C, _ = _gab_constants(params, T)
        return _gab(a, _f(params["X_m"]), C, 1.0)
    if model == "table":
        x = _f(params["a_w"])
        y = _f(params["X"])
        return jnp.maximum(_pchip_eval(x, y, _pchip_slopes(x, y), a), 0.0)
    raise ValueError("model must be 'gab', 'bet', 'table' or a callable")


def water_activity(X: Array | float, T: Array | float = 298.15, model="gab",
                   params: dict | None = None) -> Array:
    """Water activity ``a_w`` in equilibrium with moisture ``X``: the inverse of :func:`equilibrium_moisture`.

    GAB and BET are inverted in closed form (a quadratic in ``K a_w``); a table
    or a callable by bisection with an implicit-function gradient. ``X`` above
    the isotherm's range returns the upper limit (``1/K`` for GAB, 1 otherwise).
    """
    X = jnp.maximum(_f(X), 0.0)
    if model in ("gab", "bet") and not callable(model):
        C, K = _gab_constants(params, T)
        if model == "bet":
            K = jnp.ones_like(C)
        r = X / _f(params["X_m"])
        # r (1 - y)(1 + (C - 1) y) = C y,   y = K a:  A y^2 + B y + r = 0
        A = -r * (C - 1.0)
        B = r * (C - 2.0) - C
        disc = jnp.maximum(B * B - 4.0 * A * r, 0.0)
        y = 2.0 * r / (-B + jnp.sqrt(disc))
        return jnp.clip(y, 0.0, 1.0) / K
    if model == "table":
        x = _f(params["a_w"])
        lo, hi = x[0], x[-1]
    else:
        lo, hi = _f(1e-9), _f(1.0 - 1e-9)
    # the bisection is scalar: map it over an array of moistures
    return jnp.vectorize(
        lambda x: _solve_monotone(lambda a: equilibrium_moisture(a, T, model, params) - x, lo, hi))(X)


# =============================================================================
# Constant rate and heat-transfer coefficient
# =============================================================================


def heat_transfer_coefficient_tray(G: Array | float, flow: str = "parallel") -> Array:
    """Air-to-surface heat-transfer coefficient (W/m^2/K) of a tray dryer, from the air mass velocity.

    Args:
        G: Air mass velocity (kg/m^2/s).
        flow: ``"parallel"`` (air across the surface) or ``"perpendicular"`` (air
            impinging on it).

    The correlations in Geankoplis' units (``G`` in kg/(m^2 h), ``h`` in W/m^2/K)::

        parallel:       h = 0.0204 G^0.8      (G = 2450-29300 kg/m^2/h, 0.7-8.5 m/s)
        perpendicular:  h = 1.17   G^0.37     (G = 3900-19500 kg/m^2/h, 1.0-5.1 m/s)

    **Unverified: both correlations and their ranges are typed from memory of
    Geankoplis Eqs. 9.6-7 and 9.6-8** (the 0.0204 coefficient is the one the
    issue also quotes); check them against the book before relying on the
    numbers. The parallel-flow form reproduces the h = 62.45 W/m^2/K of
    Geankoplis Example 9.6-1 (also recalled; see ``tests/test_dryer.py``).
    ``G`` is converted from kg/m^2/s here. No range check is made.
    """
    Gh = 3600.0 * _f(G)
    if flow == "parallel":
        return 0.0204 * Gh ** 0.8
    if flow == "perpendicular":
        return 1.17 * Gh ** 0.37
    raise ValueError("flow must be 'parallel' or 'perpendicular'")


def constant_drying_rate(T_air: Array | float, W_air: Array | float, h: Array | float,
                         P: Array | float = P_ATM, lewis: float = 1.0,
                         k_W: Array | float | None = None) -> Array:
    """Constant-rate-period drying rate ``R_c`` (kg water / m^2 / s) of a surface at the wet-bulb temperature.

    The heat-transfer form ``R_c = h (T - T_wb) / lambda_wb`` with ``T_wb`` from
    :func:`difflow.psychrometrics.wet_bulb_temperature` and ``lambda_wb`` the
    latent heat there; or, with ``k_W`` (kg/m^2/s per unit humidity-ratio
    difference), the mass-transfer form ``R_c = k_W (W_s(T_wb) - W)``. Radiation
    and conduction from the tray are neglected.

    Args:
        T_air: Air dry-bulb temperature (K).
        W_air: Air humidity ratio (kg/kg dry air).
        h: Heat-transfer coefficient (W/m^2/K); ignored when ``k_W`` is given.
        P: Total pressure (Pa).
        lewis: Lewis factor of the wet-bulb balance (1 is the Lewis relation).
        k_W: Mass-transfer coefficient (kg/m^2/s per kg/kg), optional.
    """
    T_air, W_air = _f(T_air), _f(W_air)
    T_wb = wet_bulb_temperature(T_air, W_air, P, lewis)
    if k_W is not None:
        return _f(k_W) * (saturation_humidity_ratio(T_wb, P) - W_air)
    return _f(h) * (T_air - T_wb) / latent_heat(T_wb)


# =============================================================================
# Drying rate curve and drying time (the smooth switch)
# =============================================================================


def _sp(z, s):
    """Softplus ``s ln(1 + exp(z / s))``: a smooth ``max(z, 0)`` of width ``s``."""
    return s * jax.nn.softplus(z / s)


def _scale(X_c, X_eq, sharpness):
    return jnp.maximum(_f(X_c) - _f(X_eq), 1e-9) / _f(sharpness)


def _check_falling(falling, rate_curve, interpolate="rate"):
    if interpolate not in ("rate", "inverse"):
        raise ValueError("interpolate must be 'rate' (R linear between table points) or 'inverse' (1/R linear)")
    if falling not in ("linear", "integral"):
        raise ValueError(
            "falling must be 'linear' or 'integral' (the Fick-diffusion series is not implemented)")
    if falling == "integral" and rate_curve is None:
        raise ValueError("falling='integral' needs rate_curve=(X, R)")


def _table_rate(X, Xt, Rt, interpolate):
    """Rate from the table ``(Xt, Rt)``: ``R`` (``"rate"``) or ``1/R`` (``"inverse"``) linear between points."""
    if interpolate == "inverse":
        return 1.0 / jnp.interp(X, Xt, 1.0 / jnp.maximum(Rt, _TINY))
    return jnp.interp(X, Xt, Rt)


def drying_rate_curve(X: Array | float, X_c: Array | float, X_eq: Array | float, R_c: Array | float,
                      falling: str = "linear", sharpness: float = 200.0,
                      rate_curve: tuple | None = None, interpolate: str = "rate") -> Array:
    """Drying rate ``R(X)`` (kg/m^2/s) against the dry-basis moisture, for plotting and for the unit.

    ``falling="linear"``: ``R_c`` above ``X_c`` and ``R_c (X - X_eq)/(X_c - X_eq)``
    below, joined and floored at zero with the same softplus (width
    ``(X_c - X_eq)/sharpness``) the drying time uses. ``falling="integral"``:
    ``R_c`` above ``X_c`` and the tabulated ``rate_curve = (X_table, R_table)``
    (``X_table`` increasing) below, with ``R`` (``interpolate="rate"``) or ``1/R``
    (``"inverse"``) linear between the table points.
    """
    _check_falling(falling, rate_curve, interpolate)
    X, X_c, X_eq, R_c = _f(X), _f(X_c), _f(X_eq), _f(R_c)
    s = 1.0 / _f(sharpness)
    if falling == "linear":
        fr = (X - X_eq) / jnp.maximum(X_c - X_eq, 1e-9)
        return R_c * _sp(1.0 - _sp(1.0 - fr, s), s)
    Xt, Rt = _f(rate_curve[0]), _f(rate_curve[1])
    below = _table_rate(X, Xt, Rt, interpolate)
    sig = jax.nn.sigmoid((X - X_c) / _scale(X_c, X_eq, sharpness))  # smooth step at X_c
    return below + (R_c - below) * sig


def drying_periods(X1, X2, X_c, X_eq, R_c, Ls_over_A, falling: str = "linear",
                   sharpness: float = 200.0, rate_curve: tuple | None = None,
                   n_quad: int = 12, interpolate: str = "rate") -> tuple[Array, Array]:
    """Constant- and falling-rate times ``(t_c, t_f)`` (s) from ``X1`` to ``X2``; see :func:`drying_time`."""
    _check_falling(falling, rate_curve, interpolate)
    X1, X2, X_c, X_eq, R_c, LsA = (_f(v) for v in (X1, X2, X_c, X_eq, R_c, Ls_over_A))
    s = _scale(X_c, X_eq, sharpness)
    c1 = _sp(X1 - X_c, s)
    c2 = _sp(X2 - X_c, s)
    t_c = LsA * (c1 - c2) / R_c
    X_a = X1 - c1                      # smooth min(X1, X_c)
    X_b = X_a - _sp(X_a - X2, s)       # smooth min(X2, X_a)
    if falling == "linear":
        num = jnp.maximum(X_a - X_eq, _TINY)
        den = jnp.maximum(X_b - X_eq, _TINY)
        t_f = LsA * (X_c - X_eq) / R_c * jnp.log(num / den)
    else:
        # 1/R is smooth on each segment of a piecewise-linear table, so integrate segment by segment
        # (Gauss-Legendre, n_quad points each) over the part of it inside [X_b, X_a]; the two tails
        # beyond the table hold its end values (jnp.interp clamps)
        xi, w = np.polynomial.legendre.leggauss(int(n_quad))
        xi, w = jnp.asarray(0.5 * (xi + 1.0)), jnp.asarray(0.5 * w)
        Xt, Rt = _f(rate_curve[0]), _f(rate_curve[1])
        edges = jnp.concatenate([jnp.array([-1e3]), Xt, jnp.array([1e3])])
        lo = jnp.clip(edges[:-1], X_b, X_a)
        hi = jnp.clip(edges[1:], X_b, X_a)
        X = lo[:, None] + (hi - lo)[:, None] * xi[None, :]
        R = jnp.maximum(_table_rate(X, Xt, Rt, interpolate), _TINY)
        t_f = LsA * jnp.sum((hi - lo) * jnp.sum(w / R, axis=1))
    return t_c, t_f


def drying_time(X1, X2, X_c, X_eq, R_c, Ls_over_A, falling: str = "linear",
                sharpness: float = 200.0, rate_curve: tuple | None = None,
                n_quad: int = 12, interpolate: str = "rate") -> Array:
    """Drying time (s) from moisture ``X1`` down to ``X2`` across the constant- and falling-rate periods.

    Args:
        X1: Initial moisture (kg/kg dry solid, dry basis).
        X2: Final moisture (kg/kg dry solid).
        X_c: Critical moisture, where the falling rate starts.
        X_eq: Equilibrium moisture ``X*`` in the drying air.
        R_c: Constant drying rate (kg/m^2/s).
        Ls_over_A: Dry solid per exposed drying surface (kg/m^2).
        falling: ``"linear"`` (rate proportional to free moisture, the closed
            form ``t_f = (Ls/A)(X_c - X*)/R_c ln[(X_c - X*)/(X2 - X*)]``) or
            ``"integral"`` (numerical ``(Ls/A) int dX/R`` of the tabulated
            ``rate_curve = (X, R)``, linearly interpolated, Gauss-Legendre with ``n_quad`` points per table segment). The
            Fick-diffusion series for a slab is **not** implemented.
        sharpness: Smooth-switch sharpness (default 200); the softplus width is
            ``(X_c - X_eq)/sharpness``. See the module docstring.
        rate_curve: ``(X_table, R_table)`` for ``falling="integral"``, ``X_table``
            increasing.
        n_quad: Gauss-Legendre points per table segment for ``"integral"``.
        interpolate: How the table is read between its points for
            ``"integral"``: ``"rate"`` (``R`` linear, the default) or
            ``"inverse"`` (``1/R`` linear, so the integral is exactly the
            trapezoid rule on ``1/R`` at the table points, the textbook's
            graphical integration). The two differ by a few percent where the
            curve bends; neither is "right" between measured points.

    Returns:
        ``t_c + t_f`` (s). The three cases ``X1 > X_c > X2``, falling only
        (``X1 < X_c``) and constant only (``X2 > X_c``) are one smooth expression,
        differentiable in every argument.
    """
    t_c, t_f = drying_periods(X1, X2, X_c, X_eq, R_c, Ls_over_A, falling, sharpness, rate_curve, n_quad,
                              interpolate)
    return t_c + t_f


def _rate_fraction(X, X_c, X_eq, falling, sharpness, rate_table, interpolate="rate"):
    """``R(X)/R_c`` (0-1), smooth; ``rate_table`` is the *relative* table ``(X, R/R_c)``."""
    if falling == "linear":
        return drying_rate_curve(X, X_c, X_eq, 1.0, "linear", sharpness)
    return drying_rate_curve(X, X_c, X_eq, 1.0, "integral", sharpness, rate_table, interpolate)


# =============================================================================
# Rotary-dryer heat transfer (McCabe, Smith & Harriott, Ch. 24)
# =============================================================================


def rotary_dryer_ua(G: Array | float, D: Array | float, coefficient: float = 237.0,
                    exponent: float = 0.67) -> Array:
    """Volumetric heat-transfer coefficient ``Ua`` (W/m^3/K) of a direct-heat rotary dryer.

    ``Ua = coefficient G^exponent / D`` with the air mass velocity ``G``
    (kg/m^2/s) and the shell diameter ``D`` (m). The defaults are the SI form of
    the McCabe, Smith & Harriott (Ch. 24) correlation ``Ua = 0.5 G^0.67 / D``
    (Btu/ft^3/h/F, ``G`` in lb/ft^2/h, ``D`` in ft); the factor 237 is that
    unit conversion (18.63 x 0.5 x 0.3048 x 737.3^0.67). **Recalled from
    memory, not checked against the printed book**: verify before relying on
    it. Rotary dryers run at ``G`` of about 0.5-5 kg/m^2/s.

    Args:
        G: Air mass velocity (kg/m^2/s) through the shell cross-section.
        D: Shell diameter (m).
        coefficient: Prefactor in SI units.
        exponent: Exponent on ``G``.

    Returns:
        ``Ua`` (W/m^3/K).

    Example:
        >>> float(rotary_dryer_ua(1.0, 1.0))
        237.0
    """
    return coefficient * _f(G) ** exponent / _f(D)


def heat_transfer_units(T_air_in: Array | float, T_air_out: Array | float,
                        T_surface: Array | float) -> Array:
    """Number of heat-transfer units ``N_t = ln[(T_in - T_s)/(T_out - T_s)]`` of the air in a dryer.

    The air cools from ``T_air_in`` to ``T_air_out`` against solids at
    ``T_surface`` (the wet-bulb temperature in the constant-rate zone). Rotary
    dryers are usually designed for ``N_t`` of 1.5-2.5 (McCabe, Smith &
    Harriott, Ch. 24).

    Args:
        T_air_in: Air inlet temperature (K).
        T_air_out: Air outlet temperature (K), above ``T_surface``.
        T_surface: Solids surface temperature (K).

    Returns:
        ``N_t`` (dimensionless).
    """
    T_s = _f(T_surface)
    return jnp.log(jnp.maximum(_f(T_air_in) - T_s, _TINY) / jnp.maximum(_f(T_air_out) - T_s, _TINY))


# =============================================================================
# The unit
# =============================================================================


@dataclass(repr=False)
class DryerParams(ParamsMixin):
    """Parameters of a dryer.

    Attributes:
        mode: ``"batch"`` (a tray dryer: the time to dry a load) or
            ``"continuous"`` (steady countercurrent or cocurrent air/solids
            balances, e.g. a rotary dryer).
        flow: ``"countercurrent"`` or ``"cocurrent"`` (continuous mode).
        X_c: Critical moisture (kg/kg dry solid) where the falling-rate period starts.
        X_eq: Equilibrium moisture (kg/kg dry solid) in the drying air; if
            ``None`` it comes from the isotherm at the air's relative humidity
            (or, with neither, 0).
        isotherm_model: ``"gab"``, ``"bet"`` or ``"table"`` for ``X_eq``.
        isotherm_params: ``[X_m, C, K]`` (GAB), ``[X_m, C]`` (BET) or
            ``[[a_w ...], [X ...]]`` (table).
        R_c: Constant drying rate (kg/m^2/s); if ``None`` it is ``h (T - T_wb)/lambda``.
        h: Heat-transfer coefficient (W/m^2/K); if ``None``, from
            ``air_mass_velocity`` and the tray correlation.
        air_mass_velocity: Air mass velocity ``G`` (kg/m^2/s) for the tray correlation.
        tray_flow: ``"parallel"`` or ``"perpendicular"`` air flow for that correlation.
        falling: ``"linear"`` or ``"integral"`` falling-rate model.
        rate_curve: ``[[X ...], [R/R_c ...]]``: the falling-rate curve relative to the
            constant rate, for ``falling="integral"``.
        rate_interpolation: ``"rate"`` (``R`` linear between table points) or
            ``"inverse"`` (``1/R`` linear, the trapezoid rule on ``1/R``).
        sharpness: Smooth-switch sharpness (see :func:`drying_time`).
        product_moisture: Product moisture ``X_out`` (kg/kg dry solid), the spec.
        residence_time: Drying/residence time (s): the product moisture is
            solved for when ``product_moisture`` is ``None``.
        load: Dry solid per batch (kg), batch mode.
        area: Tray area (m^2), batch mode; with ``load`` it gives ``Ls/A``.
        solids_per_area: ``Ls/A`` (kg dry solid / m^2 of drying surface); in batch
            mode an alternative to ``area`` (the area is then computed), in
            continuous mode it lets the unit report the residence time.
        turnaround_time: Batch loading/unloading time (s) added to the cycle.
        solid_species: Name of the dry-solid pseudo-species (stream key ``F_<name>``).
        solid_molar_mass: Pseudo molar mass of the dry solid (kg/mol); the
            default 1 makes ``F_<solid>`` numerically kg/s.
        T_solids_out: Specified product temperature (K). If ``None`` (default)
            it is estimated as ``T_a - (T_a - T_wb) R(X_out)/R_c`` in the air at
            the solids' exit end; a value overrides that estimate (the
            textbook balance problems state it).
        diameter: Shell diameter (m) of a continuous rotary dryer (either flow). When
            given, the unit sizes the dryer by the three-zone method (solids
            preheat to the wet bulb, evaporation at the wet bulb, heating of
            the dried solids), ``q_i = Ua S L_i dT_lm,i`` with ``Ua`` from
            :func:`rotary_dryer_ua`.
        ua_coefficient: Prefactor of :func:`rotary_dryer_ua` (SI).
        solids_cp: Specific heat of the dry solid (J/kg/K).
        cp_water: Specific heat of liquid water in the solid (J/kg/K).
        max_solids_T: Product-quality limit on the solids temperature (K); the
            margin is reported.
        heat_loss: Heat loss to the surroundings (W), used when ``adiabatic`` is False.
        adiabatic: ``True`` ignores ``heat_loss``.
        T_ambient: Temperature (K) the air is heated from, the reference of
            the energy per kg and the thermal efficiency.
        entrainment: Fraction of the dry solids carried out with the exhaust as fines
            (continuous mode); reported in ``info["fines"]``.
        lewis: Lewis factor of the wet-bulb relation.
    """

    mode: Literal["batch", "continuous"] = "batch"
    flow: Literal["countercurrent", "cocurrent"] = "countercurrent"
    X_c: float = 0.2
    X_eq: float | None = None
    isotherm_model: Literal["gab", "bet", "table"] | None = None
    isotherm_params: list | None = None
    R_c: float | None = None
    h: float | None = None
    air_mass_velocity: float | None = None
    tray_flow: Literal["parallel", "perpendicular"] = "parallel"
    falling: Literal["linear", "integral"] = "linear"
    rate_curve: list | None = None
    rate_interpolation: Literal["rate", "inverse"] = "rate"
    sharpness: float = 200.0
    product_moisture: float | None = None
    residence_time: float | None = None
    load: float | None = None
    area: float | None = None
    solids_per_area: float | None = None
    turnaround_time: float = 0.0
    solid_species: str = "solid"
    solid_molar_mass: float = 1.0
    T_solids_out: float | None = None
    diameter: float | None = None
    ua_coefficient: float = 237.0
    solids_cp: float = 1500.0
    cp_water: float = 4186.0
    max_solids_T: float | None = None
    heat_loss: float = 0.0
    adiabatic: bool = True
    T_ambient: float = 293.15
    entrainment: float = 0.0
    lewis: float = 1.0

    def __post_init__(self):
        if self.mode not in ("batch", "continuous"):
            raise ValueError("mode must be 'batch' or 'continuous'")
        if self.flow not in ("countercurrent", "cocurrent"):
            raise ValueError("flow must be 'countercurrent' or 'cocurrent'")
        if self.falling not in ("linear", "integral"):
            raise ValueError("falling must be 'linear' or 'integral'")
        if self.falling == "integral" and self.rate_curve is None:
            raise ValueError("falling='integral' needs rate_curve")
        if self.rate_interpolation not in ("rate", "inverse"):
            raise ValueError("rate_interpolation must be 'rate' or 'inverse'")
        if self.diameter is not None and self.mode != "continuous":
            raise ValueError("diameter (three-zone rotary sizing) needs mode='continuous'")


class Dryer:
    """Tray (batch) or continuous (countercurrent / cocurrent) convective dryer with air balances.

    Two inlets, wet solids and hot air; two outlets, dried solids and the humid
    exhaust air. See the module docstring for the rate model, the smooth switch
    and the stream convention (dry solid as ``F_<solid>``, water in mol/s).

    **Batch** (``mode="batch"``): the air at the inlet state flows over a tray
    of loading ``Ls/A = load/area`` at the constant rate
    ``R_c = h (T - T_wb)/lambda`` (``R_c`` or ``h`` given, or ``h`` from the
    air mass velocity); the unit returns the drying time to ``product_moisture``
    (or the moisture reached in ``residence_time``) and the air leaving the
    trays from the water and energy balances over the drying step (the air
    flow is the flow while the fan is on; the humidity rise across the trays is
    that of the average over the tray, which the unit assumes small: check
    ``info["air_humidity_rise"]``). The solids stream's flow is the batch
    *throughput* and passes through; ``area_required`` is the tray area for it.

    **Continuous**: overall water balance ``G (W_out - W_in) = m_s (X_in -
    X_out)`` and energy balance ``G h_in + m_s h_s,in = G h_out + m_s h_s,out +
    loss`` in humid and solids enthalpy. The solids leave at
    ``T_s,out = T_a - (T_a - T_wb) R(X_out)/R_c`` evaluated in the air at the
    solids' exit end (the air inlet for countercurrent, the air outlet for
    cocurrent), which fixes ``T_a,out`` (a bisection for cocurrent). If
    ``solids_per_area`` is given the unit also reports the residence time
    from :func:`drying_time` with the constant rate taken in the air at the wet
    end of the dryer. ``T_solids_out`` overrides the product-temperature
    estimate. With ``diameter`` (continuous, either flow) the unit sizes a rotary
    dryer by the three-zone method of McCabe, Smith & Harriott (Ch. 24):
    zone I preheats the wet solids to the wet bulb ``T_wb`` of the inlet air,
    zone II evaporates all the water at ``T_wb`` (duty ``m_s (X_in - X_out)
    lambda(T_wb)``), zone III heats the dried solids to ``T_s,out``; the air
    temperatures between zones follow from the zone I and III duties, and each zone's length is ``q_i / (Ua S dT_lm,i)`` with ``Ua``
    from :func:`rotary_dryer_ua` and ``S`` the shell cross-section.
    """

    symbol = "Dryer"
    equations = [
        r"R_c = \frac{h\,(T - T_{wb})}{\lambda_{wb}},\qquad R = R_c\,\frac{X - X^*}{X_c - X^*}\ (X < X_c)",
        r"t = \frac{L_s}{A R_c}\left[(X_1 - X_c) + (X_c - X^*)\ln\frac{X_c - X^*}{X_2 - X^*}\right]",
        r"G\,(W_{out} - W_{in}) = L_s\,(X_{in} - X_{out})",
        r"G\,h_{a,in} + L_s\,h_{s,in} = G\,h_{a,out} + L_s\,h_{s,out} + Q_{loss}",
        r"T_{s,out} = T_a - (T_a - T_{wb})\,\frac{R(X_{out})}{R_c}",
    ]
    assumptions = [
        "Moisture on the dry basis; free moisture X - X_eq; equilibrium from a GAB/BET/tabulated isotherm at the air's relative humidity.",
        "Constant-rate period at the wet-bulb temperature, heat-transfer limited, no radiation or conduction from the tray; linear (in free moisture) or tabulated falling rate.",
        "The constant/falling switch at X_c is a softplus (width (X_c - X_eq)/sharpness), so the time is differentiable; sharpness -> inf is the exact piecewise answer.",
        "Batch: air at the inlet state over the whole tray (humidity rise across the tray neglected for the rate, reported); the solids stream is the throughput.",
        "Continuous: solids leave at the surface-temperature estimate T_a - (T_a - T_wb) R/R_c in the air at their exit end; heat of wetting and bound-water heats are neglected.",
        "Ideal-gas humid air (difflow.psychrometrics); constant solid and liquid-water specific heats; entrained fines are split off at the product moisture and temperature.",
    ]
    references = [
        "Geankoplis, C.J. Transport Processes and Separation Process Principles, Ch. 9 (drying).",
        "McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 24.",
        "Mujumdar, A.S. (ed.), Handbook of Industrial Drying (isotherm data).",
        "Brunauer, Emmett, Teller, J. Am. Chem. Soc. 60, 309 (1938) (BET); Guggenheim, Anderson, de Boer (GAB).",
    ]
    parameter_symbols = {
        "X_c": "X_c", "X_eq": "X^*", "R_c": "R_c", "h": "h", "air_mass_velocity": "G",
        "product_moisture": "X_{out}", "residence_time": "t", "solids_per_area": "L_s/A",
        "solids_cp": "c_{ps}", "max_solids_T": r"T_{s,max}", "heat_loss": r"Q_{loss}",
    }
    parameter_units = {
        "X_c": "kg/kg", "X_eq": "kg/kg", "R_c": "kg/(m^2 s)", "h": "W/(m^2 K)",
        "air_mass_velocity": "kg/(m^2 s)", "sharpness": "-", "product_moisture": "kg/kg",
        "residence_time": "s", "load": "kg", "area": "m^2", "solids_per_area": "kg/m^2",
        "turnaround_time": "s", "solid_molar_mass": "kg/mol", "solids_cp": "J/(kg K)",
        "cp_water": "J/(kg K)", "max_solids_T": "K", "heat_loss": "W", "T_ambient": "K",
        "entrainment": "-", "lewis": "-",
    }
    numerical_method = (
        "Closed-form drying-time integrals with softplus-smoothed regime switch; continuous air outlet by "
        "closed-form enthalpy balance (countercurrent) or bisection with implicit gradient (cocurrent)."
    )
    outlet_roles = ["dry_solids", "air_out"]

    def __init__(self, params: DryerParams):
        self.params = params

    # ------------------------------------------------------------------ helpers

    def _Xeq(self, RH, T):
        p = self.params
        if p.X_eq is not None:
            return _f(p.X_eq)
        if p.isotherm_model is not None:
            return equilibrium_moisture(RH, T, p.isotherm_model,
                                        isotherm_params(p.isotherm_model, p.isotherm_params))
        return _f(0.0)

    def _rate_table(self):
        p = self.params
        return None if p.rate_curve is None else (_f(p.rate_curve[0]), _f(p.rate_curve[1]))

    def _Rc(self, T, T_wb):
        """Constant drying rate (kg/m^2/s) for air at ``T`` with wet bulb ``T_wb``."""
        p = self.params
        lam = latent_heat(T_wb)
        if p.R_c is not None:
            return jnp.maximum(_f(p.R_c), _TINY)
        if p.h is not None:
            h = _f(p.h)
        elif p.air_mass_velocity is not None:
            h = heat_transfer_coefficient_tray(p.air_mass_velocity, p.tray_flow)
        else:
            raise ValueError("Dryer needs R_c, h or air_mass_velocity for the constant rate")
        return jnp.maximum(h * (T - T_wb) / lam, _TINY)

    def _times(self, X1, X2, X_eq, R_c, LsA):
        p = self.params
        table = None
        if p.falling == "integral":
            rt = self._rate_table()
            table = (rt[0], R_c * rt[1])
        return drying_periods(X1, X2, p.X_c, X_eq, R_c, LsA, p.falling, p.sharpness, table,
                              interpolate=p.rate_interpolation)

    def _frac(self, X, X_eq):
        p = self.params
        return _rate_fraction(X, p.X_c, X_eq, p.falling, p.sharpness, self._rate_table(), p.rate_interpolation)

    def _Ts_out(self, T_a, T_wb, X, X_eq):
        """Product temperature: the specification, or the surface estimate in air at ``T_a``."""
        p = self.params
        if p.T_solids_out is not None:
            return _f(p.T_solids_out) + 0.0 * T_a
        return T_a - (T_a - T_wb) * self._frac(X, X_eq)

    def _hs(self, X, T):
        p = self.params
        return (p.solids_cp + _f(X) * p.cp_water) * (_f(T) - T_REF)

    def _solids_in(self, wet):
        p = self.params
        m_s = _f(wet[f"F_{p.solid_species}"]) * p.solid_molar_mass
        X = _f(wet.get("F_H2O", 0.0)) * MW_WATER / m_s
        return m_s, X, _f(wet["T"])

    def _loss(self):
        p = self.params
        return 0.0 if p.adiabatic else _f(p.heat_loss)

    def _solids_stream(self, wet, m_s, X, T, P):
        p = self.params
        out = dict(wet)
        out[f"F_{p.solid_species}"] = m_s / p.solid_molar_mass
        out["F_H2O"] = m_s * X / MW_WATER
        out["T"] = T
        out["P"] = P
        return out

    def _balances(self, wet_in, dry_out, fines, air_in, air_out):
        """Water (kg/s) and energy (W) imbalances computed from the streams themselves."""
        p = self.params

        def air_h(s):
            st = state_from_air_stream(s)
            return dry_air_flow(s) * humid_enthalpy(st["T"], st["W"])

        def sol_h(s):
            ms = _f(s[f"F_{p.solid_species}"]) * p.solid_molar_mass
            mw = _f(s.get("F_H2O", 0.0)) * MW_WATER
            return (ms * p.solids_cp + mw * p.cp_water) * (_f(s["T"]) - T_REF)

        streams_out = [dry_out] + ([fines] if fines is not None else [])
        w_in = (_f(wet_in.get("F_H2O", 0.0)) + _f(air_in["F_H2O"])) * MW_WATER
        w_out = (sum(_f(s.get("F_H2O", 0.0)) for s in streams_out) + _f(air_out["F_H2O"])) * MW_WATER
        return w_out - w_in, streams_out, sol_h, air_h

    # --------------------------------------------------------------------- call

    def __call__(self, wet_solids: Stream, air_in: Stream) -> tuple[Stream, Stream, dict]:
        """Dry the solids in the air.

        Args:
            wet_solids: Wet solids: ``F_<solid>`` (dry solid, mol/s; kg/s for the
                default pseudo molar mass), ``F_H2O`` (mol/s), ``T`` (K), ``P``.
            air_in: Humid air (``F_H2O``, ``F_air``, ``T``, ``P``), see
                :mod:`difflow.psychrometrics`.

        Returns:
            dry_solids: Product solids at the product moisture and temperature.
            air_out: Exhaust air (same dry-air flow; extra non-water species of
                ``air_in`` are passed through).
            info: ``mode``, ``X_in``, ``X_out``, ``X_eq``, ``R_c``, ``T_solids_out``,
                ``T_air_out``, ``air_out`` (state of the exhaust), ``evaporation``
                (kg/s), ``W_in``, ``W_out``, ``dew_point_margin`` (K, exhaust
                dry bulb minus its dew point; > 0 means not saturated),
                ``supersaturated`` (bool), ``solids_T_margin`` (K,
                ``max_solids_T - T_solids_out``; ``inf`` when unset),
                ``Q_air`` (W, heating the air from ``T_ambient``),
                ``energy_per_kg`` (J per kg water evaporated),
                ``thermal_efficiency`` (``evaporation lambda(T_amb) / Q_air``),
                ``t_constant`` and ``t_falling`` (s), ``water_balance_residual``
                (kg/s) and ``energy_balance_residual`` (W) recomputed from the
                streams, ``fines`` (the entrained solids stream, continuous).
                Batch adds ``drying_time``, ``cycle_time``, ``area_required``
                (m^2), ``Ls_over_A``, ``air_humidity_rise``; continuous adds
                ``residence_time_required`` (with ``solids_per_area``),
                ``N_t`` (air heat-transfer units against the inlet wet bulb) and, with ``diameter``, ``zone_duties`` (W),
                ``zone_lengths`` (m), ``zone_dT_lm`` (K) in the solids' order,
                ``zone_air_T`` (K, the air at the I/II and II/III boundaries), ``length``,
                ``volume``, ``L_over_D``, ``Ua``, ``air_mass_velocity``.
        """
        if self.params.mode == "batch":
            return self._batch(wet_solids, air_in)
        return self._continuous(wet_solids, air_in)

    # -------------------------------------------------------------------- batch

    def _batch(self, wet, air_in):
        p = self.params
        st = state_from_air_stream(air_in, lewis=p.lewis)
        P = _f(air_in["P"])
        G = dry_air_flow(air_in)
        T_a, W_a, T_wb = st["T"], st["W"], st["T_wb"]
        m_s, X1, T_s_in = self._solids_in(wet)
        if p.load is None or (p.area is None and p.solids_per_area is None):
            raise ValueError("batch Dryer needs load and area (or solids_per_area)")
        load = _f(p.load)
        LsA = _f(p.solids_per_area) if p.solids_per_area is not None else load / _f(p.area)
        X_eq = self._Xeq(st["RH"], T_a)
        R_c = self._Rc(T_a, T_wb)

        def t_of(X2):
            return sum(self._times(X1, X2, X_eq, R_c, LsA))

        if p.product_moisture is not None:
            X2 = _f(p.product_moisture)
        elif p.residence_time is not None:
            X2 = _solve_monotone(lambda x: t_of(x) - _f(p.residence_time), X_eq + 1e-6, X1)
        else:
            raise ValueError("batch Dryer needs product_moisture or residence_time")
        t_c, t_f = self._times(X1, X2, X_eq, R_c, LsA)
        t_dry = t_c + t_f

        T_s_out = self._Ts_out(T_a, T_wb, X2, X_eq)
        rate = load / jnp.maximum(t_dry, _TINY)           # dry solid processed while drying (kg/s)
        evap = rate * (X1 - X2)
        W_out = W_a + evap / G
        h_in = st["h"]
        h_out = h_in + (rate * (self._hs(X1, T_s_in) - self._hs(X2, T_s_out)) - self._loss()) / G
        T_out = humid_temperature(h_out, W_out)
        st_out = state(T=T_out, W=W_out, P=P, lewis=p.lewis)
        air_out = self._air_out(air_in, st_out, G)
        dry = self._solids_stream(wet, m_s, X2, T_s_out, _f(wet.get("P", P)))

        cycle = t_dry + _f(p.turnaround_time)
        info = self._common_info(X1, X2, X_eq, R_c, T_s_out, st_out, evap, G, st, t_c, t_f)
        info.update({
            "mode": "batch",
            "drying_time": t_dry,
            "cycle_time": cycle,
            "Ls_over_A": LsA,
            "area_required": m_s * cycle / LsA,
            "air_humidity_rise": W_out - W_a,
        })
        # the balances are over the drying step: solids at the batch rate
        wet_b = self._solids_stream(wet, rate, X1, T_s_in, _f(wet.get("P", P)))
        dry_b = self._solids_stream(wet, rate, X2, T_s_out, _f(wet.get("P", P)))
        self._add_balances(info, wet_b, dry_b, None, air_in, air_out)
        return dry, air_out, info

    # --------------------------------------------------------------- continuous

    def _air_out(self, air_in, st_out, G):
        out = air_stream_from_state(st_out, G)
        for k, v in air_in.items():
            if k.startswith("F_") and k not in out:
                out[k] = v
        return out

    def _common_info(self, X1, X2, X_eq, R_c, T_s_out, st_out, evap, G, st_in, t_c, t_f):
        p = self.params
        T_amb = _f(p.T_ambient)
        Q_air = G * (st_in["h"] - humid_enthalpy(T_amb, st_in["W"]))
        epk = Q_air / jnp.maximum(evap, _TINY)
        margin = (_f(p.max_solids_T) - T_s_out) if p.max_solids_T is not None else _f(jnp.inf)
        return {
            "X_in": X1, "X_out": X2, "X_eq": X_eq, "R_c": R_c,
            "T_solids_out": T_s_out, "T_air_out": st_out["T"], "air_out": st_out,
            "evaporation": evap, "G": G, "W_in": st_in["W"], "W_out": st_out["W"],
            "dew_point_margin": st_out["T"] - st_out["T_dp"],
            "supersaturated": st_out["RH"] > 1.0,
            "solids_T_margin": margin,
            "Q_air": Q_air,
            "energy_per_kg": epk,
            "thermal_efficiency": evap * latent_heat(T_amb) / jnp.where(Q_air > 0, Q_air, jnp.inf),
            "t_constant": t_c, "t_falling": t_f,
        }

    def _add_balances(self, info, wet_in, dry_out, fines, air_in, air_out):
        dw, streams_out, sol_h, air_h = self._balances(wet_in, dry_out, fines, air_in, air_out)
        e_in = air_h(air_in) + sol_h(wet_in)
        e_out = air_h(air_out) + sum(sol_h(s) for s in streams_out) + self._loss()
        info["water_balance_residual"] = dw
        info["energy_balance_residual"] = e_out - e_in

    def _three_zone(self, m_s, X_in, X_out, T_s_in, T_s_out, st_in, st_out, G):
        """Three-zone length of a rotary dryer (McCabe, Smith & Harriott, Ch. 24), either flow direction.

        Zones in the solids' order: I preheats the wet solids to the wet bulb, II
        evaporates at the wet bulb, III heats the dried solids. ``zone_air_T`` is
        the air temperature at the I/II and II/III boundaries.
        """
        from difflow.units.heat_exchanger import log_mean_temperature_difference as lmtd

        p = self.params
        T_wb = st_in["T_wb"]
        cs_in = CP_DRY_AIR + CP_VAPOR * st_in["W"]
        cs_out = CP_DRY_AIR + CP_VAPOR * st_out["W"]
        # heat passed from the air to the solids in each zone; the latent heat leaves the solids as vapor
        q3 = m_s * (p.solids_cp + X_out * p.cp_water) * (T_s_out - T_wb)    # heat the dried solids (hot end)
        q1 = m_s * (p.solids_cp + X_in * p.cp_water) * (T_wb - T_s_in)      # preheat the wet solids (cold end)
        q2 = m_s * (X_in - X_out) * latent_heat(T_wb)                       # evaporate at the wet bulb
        T_in, T_out = st_in["T"], st_out["T"]
        if p.flow == "countercurrent":
            T_b = T_in - q3 / (G * cs_in)         # air leaving zone III (II/III boundary)
            T_c = T_out + q1 / (G * cs_out)       # air entering zone I (I/II boundary)
            dT3 = lmtd(T_in - T_s_out, T_b - T_wb)
            dT2 = lmtd(T_b - T_wb, T_c - T_wb)
            dT1 = lmtd(T_c - T_wb, T_out - T_s_in)
            T_12, T_23 = T_c, T_b
        else:                                     # cocurrent: the hot air meets the wet feed
            T_b = T_in - q1 / (G * cs_in)         # air leaving zone I
            T_c = T_out + q3 / (G * cs_out)       # air entering zone III
            dT1 = lmtd(T_in - T_s_in, T_b - T_wb)
            dT2 = lmtd(T_b - T_wb, T_c - T_wb)
            dT3 = lmtd(T_c - T_wb, T_out - T_s_out)
            T_12, T_23 = T_b, T_c
        D = _f(p.diameter)
        S = jnp.pi * D ** 2 / 4.0
        G_mass = G / S
        Ua = rotary_dryer_ua(G_mass, D, p.ua_coefficient)
        L1, L2, L3 = (q / (Ua * S * dT) for q, dT in ((q1, dT1), (q2, dT2), (q3, dT3)))
        length = L1 + L2 + L3
        return {
            "air_mass_velocity": G_mass, "Ua": Ua, "cross_section": S,
            "zone_duties": jnp.stack([q1, q2, q3]), "zone_lengths": jnp.stack([L1, L2, L3]),
            "zone_dT_lm": jnp.stack([dT1, dT2, dT3]), "zone_air_T": jnp.stack([T_12, T_23]),
            "length": length, "volume": S * length, "L_over_D": length / D,
        }

    def _continuous(self, wet, air_in):
        p = self.params
        counter = p.flow == "countercurrent"
        st = state_from_air_stream(air_in, lewis=p.lewis)
        P = _f(air_in["P"])
        G = dry_air_flow(air_in)
        T_a, W_a = st["T"], st["W"]
        m_s, X_in, T_s_in = self._solids_in(wet)
        loss = self._loss()
        h_in = st["h"]
        hs_in = self._hs(X_in, T_s_in)
        LsA = None if p.solids_per_area is None else _f(p.solids_per_area)
        Xeq_in = self._Xeq(st["RH"], T_a)

        def core(X_out):
            evap = m_s * (X_in - X_out)
            W_out = W_a + evap / G
            if counter:
                Xeq = Xeq_in
                T_s_out = self._Ts_out(T_a, st["T_wb"], X_out, Xeq)
                h_out = h_in + (m_s * (hs_in - self._hs(X_out, T_s_out)) - loss) / G
                T_out = humid_temperature(h_out, W_out)
            else:
                def ends(T):
                    Twb = jnp.minimum(wet_bulb_temperature(T, W_out, P, p.lewis), T)
                    Xe = self._Xeq(rh_from_humidity_ratio(T, W_out, P), T)
                    return self._Ts_out(T, Twb, X_out, Xe), Xe

                def resid(T):
                    return (G * humid_enthalpy(T, W_out) + m_s * self._hs(X_out, ends(T)[0]) + loss
                            - G * h_in - m_s * hs_in)

                T_out = _solve_monotone(resid, 200.0, jnp.maximum(T_a, T_s_in) + 1.0)
                T_s_out, Xeq = ends(T_out)
            out = {"evap": evap, "W_out": W_out, "T_out": T_out, "T_s_out": T_s_out, "Xeq": Xeq}
            if LsA is not None or p.residence_time is not None:
                # constant rate in the air at the wet end: the outlet air (countercurrent) or the inlet air
                if counter:
                    T_wet = T_out
                    Twb_wet = jnp.minimum(wet_bulb_temperature(T_out, W_out, P, p.lewis), T_out)
                else:
                    T_wet, Twb_wet = T_a, st["T_wb"]
                R_c = self._Rc(T_wet, Twb_wet)
                out["R_c"] = R_c
                if LsA is not None:
                    out["t_c"], out["t_f"] = self._times(X_in, X_out, Xeq, R_c, LsA)
            return out

        if p.product_moisture is not None:
            X_out = _f(p.product_moisture)
        elif p.residence_time is not None and LsA is not None:
            def g(x):
                o = core(x)
                return o["t_c"] + o["t_f"] - _f(p.residence_time)

            X_out = _solve_monotone(g, Xeq_in + 1e-6, X_in)
        else:
            raise ValueError("continuous Dryer needs product_moisture (or residence_time with solids_per_area)")
        o = core(X_out)

        st_out = state(T=o["T_out"], W=o["W_out"], P=P, lewis=p.lewis)
        air_out = self._air_out(air_in, st_out, G)
        e = _f(p.entrainment)
        P_s = _f(wet.get("P", P))
        dry = self._solids_stream(wet, (1.0 - e) * m_s, X_out, o["T_s_out"], P_s)
        fines = self._solids_stream(wet, e * m_s, X_out, o["T_s_out"], P_s)
        t_c = o.get("t_c", _f(jnp.nan))
        t_f = o.get("t_f", _f(jnp.nan))
        info = self._common_info(X_in, X_out, o["Xeq"], o.get("R_c", _f(jnp.nan)), o["T_s_out"], st_out,
                                 o["evap"], G, st, t_c, t_f)
        info["mode"] = "continuous"
        info["fines"] = fines
        info["N_t"] = heat_transfer_units(T_a, o["T_out"], st["T_wb"])
        if p.diameter is not None:
            info.update(self._three_zone(m_s, X_in, X_out, T_s_in, o["T_s_out"], st, st_out, G))
        if LsA is not None:
            info["residence_time_required"] = t_c + t_f
            info["Ls_over_A"] = LsA
        self._add_balances(info, wet, dry, fines, air_in, air_out)
        return dry, air_out, info
