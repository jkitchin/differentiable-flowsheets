"""Compressible (ideal-gas) flow: Mach number, nozzles, and gas pipelines.

The incompressible pipe-flow tools are in :mod:`difflow.fluids`; this module
covers gas flow where the density changes along the flow path. Everything is
JAX, SI and differentiable.

Contents
--------
* :func:`speed_of_sound`, :func:`mach_number`
* :func:`stagnation_ratios` -- isentropic ``T0/T``, ``P0/P``, ``rho0/rho``
* :func:`critical_pressure_ratio`, :func:`choked_mass_flux`,
  :func:`nozzle_mass_flow` -- a converging nozzle (or orifice, relief valve)
  from subsonic to choked flow
* :func:`isothermal_pipe_mass_flux`, :func:`isothermal_outlet_pressure` -- the
  isothermal gas-line equation with the acceleration term, and its choking
  limit ``M = 1/sqrt(gamma)``
* :func:`isothermal_pipe_flow` -- the same line with Reynolds-dependent
  friction solved together with the flow; :func:`isothermal_distance`,
  :func:`isothermal_pressure_profile` -- the pressure profile ``x(P)``/``P(x)``
* :func:`fanno_parameter`, :func:`fanno_ratios`, :func:`adiabatic_outlet_mach`
  -- adiabatic (Fanno) flow in a constant-area duct with friction
* :func:`polytropic_exponent`, :func:`isentropic_efficiency_from_polytropic`,
  :func:`polytropic_efficiency_from_isentropic`, :func:`optimal_stage_ratios`
  -- ideal-gas compressor/expander efficiency conversions and staging

All of it assumes an ideal gas with constant ``gamma = Cp/Cv`` and molar mass
``MW`` (kg/mol); the sound-speed and isothermal-line functions take an optional
mean compressibility factor ``Z`` (default 1), which replaces ``R T / MW`` by
``Z R T / MW``. Friction factors are **Darcy** factors, as in
:mod:`difflow.fluids` (the Fanning factor is ``f/4``); for a gas line,
``Re = G D / mu`` is constant along an isothermal pipe because the mass flux
``G`` and the viscosity are, so ``f`` is a constant from
:func:`difflow.fluids.friction_factor`.

Example:
    >>> from difflow.compressible import critical_pressure_ratio
    >>> round(float(critical_pressure_ratio(1.4)), 4)   # air
    0.5283
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

#: Molar gas constant (J/mol/K).
R_GAS = 8.314462618

_BISECTION_STEPS = 80


# =============================================================================
# Speed of sound and stagnation properties
# =============================================================================


def speed_of_sound(
    T: Array | float, MW: Array | float, gamma: Array | float = 1.4, Z: Array | float = 1.0
) -> Array:
    """Speed of sound ``c = sqrt(gamma Z R T / MW)`` (m/s).

    ``Z = 1`` is the ideal gas. With a compressibility factor this is the usual
    real-gas estimate (it neglects the ``(dZ/dT)`` and ``(dZ/dP)`` terms of the
    exact expression), adequate for a Mach-number check on a gas line.

    Args:
        T: Temperature (K).
        MW: Molar mass (kg/mol), e.g. 0.02897 for air.
        gamma: Heat-capacity ratio ``Cp/Cv`` (-).
        Z: Compressibility factor (-).

    Returns:
        Speed of sound (m/s).

    Example:
        >>> round(float(speed_of_sound(293.15, 0.02897)))
        343
    """
    return jnp.sqrt(jnp.asarray(gamma) * jnp.asarray(Z) * R_GAS * jnp.asarray(T) / jnp.asarray(MW))


def mach_number(
    v: Array | float, T: Array | float, MW: Array | float, gamma: Array | float = 1.4,
    Z: Array | float = 1.0,
) -> Array:
    """Mach number ``M = v / c`` (-).

    Args:
        v: Velocity (m/s).
        T: Static temperature (K).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-).
        Z: Compressibility factor (-), see :func:`speed_of_sound`.

    Returns:
        Mach number (-).
    """
    return jnp.asarray(v) / speed_of_sound(T, MW, gamma, Z)


class StagnationRatios(NamedTuple):
    """Isentropic stagnation-to-static ratios at Mach ``M``."""

    T0_T: Array
    P0_P: Array
    rho0_rho: Array


def stagnation_ratios(M: Array | float, gamma: Array | float = 1.4) -> StagnationRatios:
    """Isentropic ``T0/T``, ``P0/P`` and ``rho0/rho`` at Mach ``M``.

    ``T0/T = 1 + (gamma-1)/2 M^2``; ``P0/P = (T0/T)^(gamma/(gamma-1))``;
    ``rho0/rho = (T0/T)^(1/(gamma-1))``.

    Args:
        M: Mach number (-).
        gamma: Heat-capacity ratio (-).

    Returns:
        :class:`StagnationRatios`.

    Example:
        >>> r = stagnation_ratios(1.0, 1.4)
        >>> round(float(r.P0_P), 4)
        1.8929
    """
    M = jnp.asarray(M)
    g = jnp.asarray(gamma)
    t = 1.0 + 0.5 * (g - 1.0) * M**2
    return StagnationRatios(t, t ** (g / (g - 1.0)), t ** (1.0 / (g - 1.0)))


# =============================================================================
# Nozzles: subsonic to choked
# =============================================================================


def critical_pressure_ratio(gamma: Array | float = 1.4) -> Array:
    """Back-to-stagnation pressure ratio at which a converging nozzle chokes.

    ``P*/P0 = (2/(gamma+1))^(gamma/(gamma-1))``: 0.528 for air, 0.546 for
    steam-like ``gamma = 1.3``. At any lower back pressure the throat stays at
    Mach 1 and the mass flow no longer increases.

    Args:
        gamma: Heat-capacity ratio (-).

    Returns:
        ``P*/P0`` (-).
    """
    g = jnp.asarray(gamma)
    return (2.0 / (g + 1.0)) ** (g / (g - 1.0))


def choked_mass_flux(
    P0: Array | float, T0: Array | float, MW: Array | float, gamma: Array | float = 1.4
) -> Array:
    """Mass flux through a sonic throat (kg/m^2/s).

    ``G* = P0 sqrt(gamma MW / (R T0)) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))``.

    Args:
        P0: Upstream stagnation pressure (Pa).
        T0: Upstream stagnation temperature (K).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-).

    Returns:
        Choked mass flux ``G*`` (kg/m^2/s), linear in ``P0``.
    """
    g = jnp.asarray(gamma)
    return (
        jnp.asarray(P0)
        * jnp.sqrt(g * jnp.asarray(MW) / (R_GAS * jnp.asarray(T0)))
        * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    )


def nozzle_mass_flow(
    P0: Array | float,
    T0: Array | float,
    P_back: Array | float,
    area: Array | float,
    MW: Array | float,
    gamma: Array | float = 1.4,
    Cd: Array | float = 1.0,
) -> Array:
    """Mass flow through a converging nozzle, subsonic or choked (kg/s).

    For an isentropic nozzle discharging to ``P_back``, with ``r = P_back/P0``
    held at no less than :func:`critical_pressure_ratio`::

        m = Cd A P0 sqrt( 2 gamma MW / ((gamma-1) R T0)
                          * (r^(2/gamma) - r^((gamma+1)/gamma)) )

    Below the critical ratio the throat is sonic and ``m`` equals
    ``Cd A`` :func:`choked_mass_flux`. The isentropic flux has its maximum
    exactly at the critical ratio, so clamping ``r`` there is continuous with a
    continuous first derivative (zero on both sides): ``jax.grad`` with
    respect to the back pressure is 0 in choked flow, as it physically is.

    Args:
        P0: Upstream stagnation pressure (Pa).
        T0: Upstream stagnation temperature (K).
        P_back: Downstream (back) pressure (Pa), at most ``P0``.
        area: Throat area (m^2).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-).
        Cd: Discharge coefficient (-); about 0.97 for a well-rounded nozzle,
            0.6-0.85 for an orifice or relief valve (use the vendor's value).

    Returns:
        Mass flow (kg/s).

    Example:
        >>> m = nozzle_mass_flow(10e5, 300.0, 1e5, 1e-4, 0.02897)  # choked
        >>> round(float(m), 4)
        0.2334
    """
    g = jnp.asarray(gamma)
    P0 = jnp.asarray(P0)
    r = jnp.clip(jnp.asarray(P_back) / P0, critical_pressure_ratio(g), 1.0)
    term = r ** (2.0 / g) - r ** ((g + 1.0) / g)
    flux = P0 * jnp.sqrt(2.0 * g * jnp.asarray(MW) / ((g - 1.0) * R_GAS * jnp.asarray(T0)) * jnp.maximum(term, 0.0))
    return jnp.asarray(Cd) * jnp.asarray(area) * flux


class NozzleFlow(NamedTuple):
    """Converging-nozzle result (:func:`nozzle_flow`)."""

    m_dot: Array
    """Mass flow (kg/s), equal to :func:`nozzle_mass_flow`."""
    choked: Array
    """True when ``P_back/P0`` is at or below :func:`critical_pressure_ratio`."""
    P_throat: Array
    """Throat static pressure (Pa): ``P_back`` if subsonic, ``r* P0`` if choked."""
    mach_throat: Array
    """Throat Mach number (-): 1 if choked."""


def nozzle_flow(
    P0: Array | float,
    T0: Array | float,
    P_back: Array | float,
    area: Array | float,
    MW: Array | float,
    gamma: Array | float = 1.4,
    Cd: Array | float = 1.0,
) -> NozzleFlow:
    """:func:`nozzle_mass_flow` plus the choked flag and the throat state.

    Same arguments and the same mass flow; additionally reports whether the
    throat is sonic and its static pressure and Mach number,
    ``M_t = sqrt(2/(gamma-1) ((P0/P_t)^((gamma-1)/gamma) - 1))``.

    Args:
        P0: Upstream stagnation pressure (Pa).
        T0: Upstream stagnation temperature (K).
        P_back: Back pressure (Pa), at most ``P0``.
        area: Throat area (m^2).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-).
        Cd: Discharge coefficient (-).

    Returns:
        :class:`NozzleFlow`.

    Example:
        >>> r = nozzle_flow(10e5, 293.15, 8e5, 3.1416e-4 / 4, 0.02897, 1.4, Cd=0.85)
        >>> bool(r.choked), round(float(r.m_dot), 4)
        (False, 0.129)
    """
    g = jnp.asarray(gamma)
    P0 = jnp.asarray(P0)
    r_star = critical_pressure_ratio(g)
    r_back = jnp.asarray(P_back) / P0
    choked = r_back <= r_star
    r = jnp.clip(r_back, r_star, 1.0)
    mach = jnp.sqrt(jnp.maximum(2.0 / (g - 1.0) * (r ** (-(g - 1.0) / g) - 1.0), 0.0))
    m = nozzle_mass_flow(P0, T0, P_back, area, MW, gamma, Cd)
    return NozzleFlow(m, choked, r * P0, mach)


# =============================================================================
# Root finding with implicit-function gradients
# =============================================================================


def _bisect_log(F, lo, hi):
    """Root of ``F`` in ``[lo, hi]`` (positive, F(lo) and F(hi) of opposite sign) by log bisection.

    Pure value: the caller attaches gradients with a Newton step.
    """
    f_lo = F(lo)

    def body(_, bounds):
        a, b = bounds
        m = jnp.sqrt(a * b)
        same = jnp.sign(F(m)) == jnp.sign(f_lo)
        return jnp.where(same, m, a), jnp.where(same, b, m)

    a, b = jax.lax.fori_loop(0, _BISECTION_STEPS, body, (lo, hi))
    return jnp.sqrt(a * b)


def _implicit(F, x_star):
    """Attach implicit-function-theorem gradients to a root ``x_star`` of ``F``.

    ``x = x* - F(x*)/F'(x*)`` with ``x*`` stopped: the value is unchanged (F(x*)
    ~ 0), and the derivative with respect to anything ``F`` closes over is
    ``-F_theta / F_x``.
    """
    x0 = jax.lax.stop_gradient(x_star)
    Fx = jax.grad(F)(x0)
    return x0 - F(x0) / Fx


# =============================================================================
# Isothermal pipe flow
# =============================================================================


def isothermal_pipe_mass_flux(
    P1: Array | float,
    P2: Array | float,
    T: Array | float,
    L: Array | float,
    D: Array | float,
    f: Array | float,
    MW: Array | float,
    Z: Array | float = 1.0,
) -> Array:
    """Mass flux ``G`` in an isothermal gas line from its end pressures (kg/m^2/s).

    Integrating the momentum balance for an ideal gas at constant ``T``
    (McCabe, Smith & Harriott 7e, Eq. 6.38; Coulson & Richardson Vol 1,
    Eq. 4.55, with the Darcy factor)::

        P1^2 - P2^2 = G^2 (R T / MW) [ f L / D + 2 ln(P1/P2) ]

    The log term is the acceleration of the expanding gas; dropping it gives
    the Weymouth-type "long line" form. The result is valid only while the
    exit Mach number is below the isothermal limit ``1/sqrt(gamma)``; check
    with :func:`isothermal_outlet_pressure` or the exit velocity
    ``G R T / (MW P2)``.

    Args:
        P1: Inlet pressure (Pa).
        P2: Outlet pressure (Pa), ``0 < P2 < P1``.
        T: Gas temperature (K).
        L: Pipe length (m).
        D: Inside diameter (m).
        f: Darcy friction factor (-).
        MW: Molar mass (kg/mol).
        Z: Mean compressibility factor (-); ``R T / MW`` becomes ``Z R T / MW``.

    Returns:
        Mass flux ``G`` (kg/m^2/s); the mass flow is ``G pi D^2 / 4``.
    """
    P1 = jnp.asarray(P1)
    P2 = jnp.asarray(P2)
    RT_M = jnp.asarray(Z) * R_GAS * jnp.asarray(T) / jnp.asarray(MW)
    resistance = jnp.asarray(f) * jnp.asarray(L) / jnp.asarray(D) + 2.0 * jnp.log(P1 / P2)
    return jnp.sqrt((P1**2 - P2**2) / (RT_M * resistance))


class IsothermalPipeResult(NamedTuple):
    """Outlet of an isothermal gas line (:func:`isothermal_outlet_pressure`)."""

    P2: Array
    """Outlet pressure (Pa); the choke pressure ``P_choke`` if choked."""
    mach2: Array
    """Outlet Mach number (-), based on ``gamma``."""
    choked: Array
    """True if the line cannot pass ``G``: the length needed exceeds ``L`` at the limit."""
    P_choke: Array
    """Pressure at which the velocity reaches ``sqrt(Z R T / MW)`` (the isothermal limit, Pa)."""


def isothermal_outlet_pressure(
    G: Array | float,
    P1: Array | float,
    T: Array | float,
    L: Array | float,
    D: Array | float,
    f: Array | float,
    MW: Array | float,
    gamma: Array | float = 1.4,
    Z: Array | float = 1.0,
) -> IsothermalPipeResult:
    """Outlet pressure of an isothermal gas line carrying mass flux ``G``.

    Solves the equation of :func:`isothermal_pipe_mass_flux` for ``P2``. The
    residual ``F(P2) = P1^2 - P2^2 - G^2 (RT/MW)(fL/D + 2 ln(P1/P2))`` rises
    from ``F(P1) < 0`` to a maximum at ``P_choke = G sqrt(RT/MW)``, where the
    velocity is ``sqrt(RT/MW)``, i.e. Mach ``1/sqrt(gamma)``; the subsonic
    root lies in ``[P_choke, P1]`` and is found by bisection with gradients by
    the implicit function theorem. If ``F(P_choke) < 0`` the pipe is too long
    for this flow at this inlet pressure: ``choked`` is True and ``P2`` is
    returned as ``P_choke`` (the real flow would be lower).

    Args:
        G: Mass flux (kg/m^2/s).
        P1: Inlet pressure (Pa).
        T: Gas temperature (K).
        L: Pipe length (m).
        D: Inside diameter (m).
        f: Darcy friction factor (-).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-), only for the reported Mach number.
        Z: Mean compressibility factor (-).

    Returns:
        :class:`IsothermalPipeResult`.

    Example:
        >>> G = isothermal_pipe_mass_flux(60e5, 40e5, 288.0, 5e4, 0.5, 0.012, 0.0175)
        >>> r = isothermal_outlet_pressure(G, 60e5, 288.0, 5e4, 0.5, 0.012, 0.0175)
        >>> round(float(r.P2) / 1e5, 6)
        40.0
    """
    G = jnp.asarray(G, dtype=jnp.float64)
    P1 = jnp.asarray(P1, dtype=jnp.float64)
    RT_M = jnp.asarray(Z) * R_GAS * jnp.asarray(T) / jnp.asarray(MW)
    fLD = jnp.asarray(f) * jnp.asarray(L) / jnp.asarray(D)

    def F(P2):
        return P1**2 - P2**2 - G**2 * RT_M * (fLD + 2.0 * jnp.log(P1 / P2))

    P_choke = G * jnp.sqrt(RT_M)
    choked = F(P_choke) < 0.0
    lo = jax.lax.stop_gradient(jnp.minimum(P_choke, P1))
    root = _bisect_log(F, lo, jax.lax.stop_gradient(P1))
    # Away from the choke F'(P2) = -2 P2 + 2 G^2 RT/(MW P2) < 0; at the choke it is 0.
    P2_free = _implicit(F, jnp.where(choked, P1, root))
    P2 = jnp.where(choked, P_choke, P2_free)
    mach2 = G * RT_M / P2 / speed_of_sound(T, MW, gamma, Z)
    return IsothermalPipeResult(P2, mach2, choked, P_choke)


def isothermal_limit_mach(gamma: Array | float = 1.4) -> Array:
    """Limiting (choking) Mach number of isothermal pipe flow, ``1/sqrt(gamma)``.

    An isothermal line chokes when its velocity reaches ``sqrt(Z R T / MW)``,
    the *isothermal* sound speed, i.e. at Mach ``1/sqrt(gamma)`` on the
    adiabatic sound speed (0.845 for ``gamma = 1.4``), not at Mach 1.

    Args:
        gamma: Heat-capacity ratio (-).

    Returns:
        Limiting Mach number (-).
    """
    return 1.0 / jnp.sqrt(jnp.asarray(gamma))


def isothermal_distance(
    P: Array | float,
    G: Array | float,
    P1: Array | float,
    T: Array | float,
    D: Array | float,
    f: Array | float,
    MW: Array | float,
    Z: Array | float = 1.0,
) -> Array:
    """Distance from the inlet at which an isothermal line reaches pressure ``P`` (m).

    The isothermal line equation solved explicitly for the length::

        x(P) = (D / f) [ (P1^2 - P^2) MW / (G^2 Z R T) - 2 ln(P1/P) ]

    Valid for ``P`` between the inlet pressure and the choke pressure
    ``G sqrt(Z R T / MW)``; plotting ``x(P)`` against ``P`` is the cheapest way
    to draw the pressure profile. The inverse, ``P(x)``, is
    :func:`isothermal_pressure_profile`.

    Args:
        P: Pressure(s) along the line (Pa).
        G: Mass flux (kg/m^2/s).
        P1: Inlet pressure (Pa).
        T: Gas temperature (K).
        D: Inside diameter (m).
        f: Darcy friction factor (-).
        MW: Molar mass (kg/mol).
        Z: Mean compressibility factor (-).

    Returns:
        Distance(s) ``x`` (m), the shape of ``P``.
    """
    P = jnp.asarray(P)
    P1 = jnp.asarray(P1)
    RT_M = jnp.asarray(Z) * R_GAS * jnp.asarray(T) / jnp.asarray(MW)
    G = jnp.asarray(G)
    return jnp.asarray(D) / jnp.asarray(f) * ((P1**2 - P**2) / (G**2 * RT_M) - 2.0 * jnp.log(P1 / P))


def isothermal_pressure_profile(
    x: Array | float,
    G: Array | float,
    P1: Array | float,
    T: Array | float,
    D: Array | float,
    f: Array | float,
    MW: Array | float,
    gamma: Array | float = 1.4,
    Z: Array | float = 1.0,
) -> IsothermalPipeResult:
    """Pressure ``P(x)`` along an isothermal gas line, at one or many positions.

    :func:`isothermal_outlet_pressure` evaluated with ``L = x``, vectorised
    over ``x`` (``jax.vmap``). Positions past the choke point come back with
    ``choked = True`` and ``P2 = P_choke``.

    Args:
        x: Distance(s) from the inlet (m), scalar or 1-D.
        G: Mass flux (kg/m^2/s).
        P1: Inlet pressure (Pa).
        T: Gas temperature (K).
        D: Inside diameter (m).
        f: Darcy friction factor (-).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-), for the Mach number.
        Z: Mean compressibility factor (-).

    Returns:
        :class:`IsothermalPipeResult` whose fields have the shape of ``x``
        (``P2`` is the pressure at each ``x``).

    Example:
        >>> import jax.numpy as jnp
        >>> prof = isothermal_pressure_profile(jnp.linspace(0.0, 5e4, 6), 300.0, 60e5,
        ...                                    288.0, 0.5, 0.012, 0.0175)
        >>> prof.P2.shape
        (6,)
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    def one(xi):
        return isothermal_outlet_pressure(G, P1, T, xi, D, f, MW, gamma, Z)

    if x.ndim == 0:
        return one(x)
    return jax.vmap(one)(x)


class IsothermalPipeFlow(NamedTuple):
    """Flow in an isothermal gas line with Reynolds-dependent friction (:func:`isothermal_pipe_flow`)."""

    G: Array
    """Mass flux (kg/m^2/s)."""
    m_dot: Array
    """Mass flow ``G pi D^2 / 4`` (kg/s)."""
    f: Array
    """Darcy friction factor at the converged Reynolds number (-)."""
    Re: Array
    """Reynolds number ``G D / mu`` (-), constant along the line."""


def isothermal_pipe_flow(
    P1: Array | float,
    P2: Array | float,
    T: Array | float,
    L: Array | float,
    D: Array | float,
    roughness: Array | float,
    mu: Array | float,
    MW: Array | float,
    Z: Array | float = 1.0,
    method: str = "colebrook",
) -> IsothermalPipeFlow:
    """Mass flow of an isothermal gas line from its end pressures, friction included.

    :func:`isothermal_pipe_mass_flux` needs ``f``, and ``f`` needs
    ``Re = G D / mu``. This solves the pair: the fixed point
    ``f = friction_factor(G(f) D / mu, roughness / D)`` (``f`` depends on
    ``Re`` only logarithmically, so plain substitution contracts fast), with
    the gradient attached by the implicit function theorem on
    ``F(f) = f - friction_factor(Re(G(f)))``.

    Args:
        P1: Inlet pressure (Pa).
        P2: Outlet pressure (Pa), ``0 < P2 < P1``.
        T: Gas temperature (K).
        L: Pipe length (m).
        D: Inside diameter (m).
        roughness: Absolute roughness (m).
        mu: Gas viscosity (Pa s), constant along an isothermal line.
        MW: Molar mass (kg/mol).
        Z: Mean compressibility factor (-).
        method: Friction correlation of :func:`difflow.fluids.friction_factor`.

    Returns:
        :class:`IsothermalPipeFlow`.

    Example:
        >>> r = isothermal_pipe_flow(60e5, 40e5, 288.15, 50e3, 0.6, 45e-6, 1.2e-5,
        ...                          0.0180042, Z=0.851)
        >>> round(float(r.m_dot), 1), round(float(r.f), 5)
        (121.5, 0.01146)
    """
    from difflow.fluids import friction_factor

    D = jnp.asarray(D, dtype=jnp.float64)
    e = jnp.asarray(roughness) / D

    def G_of(f):
        return isothermal_pipe_mass_flux(P1, P2, T, L, D, f, MW, Z)

    def ff(f):
        return friction_factor(G_of(f) * D / jnp.asarray(mu), e, method)

    def F(f):
        return f - ff(f)

    sg = jax.lax.stop_gradient
    f_star = jax.lax.fori_loop(0, 30, lambda _, f: sg(ff(f)), jnp.asarray(0.02))
    f = _implicit(F, f_star)
    G = G_of(f)
    return IsothermalPipeFlow(G, G * jnp.pi * D**2 / 4.0, f, G * D / jnp.asarray(mu))


# =============================================================================
# Adiabatic (Fanno) flow
# =============================================================================


def fanno_parameter(M: Array | float, gamma: Array | float = 1.4) -> Array:
    """Fanno ``f L* / D``: the duct length (in units of D/f) that brings Mach ``M`` to 1.

    ``fL*/D = (1 - M^2)/(gamma M^2) + (gamma+1)/(2 gamma) ln[(gamma+1) M^2 / (2 + (gamma-1) M^2)]``
    with the **Darcy** ``f``; Fanning-based tables list the same number as
    ``4 f_F L*/D``. Decreases from infinity at ``M -> 0`` to 0 at ``M = 1``.

    Args:
        M: Mach number (-), positive.
        gamma: Heat-capacity ratio (-).

    Returns:
        ``f L*/D`` (-).

    Example:
        >>> round(float(fanno_parameter(0.5, 1.4)), 4)   # Fanning tables: 4fL*/D = 1.0691
        1.0691
    """
    M2 = jnp.asarray(M) ** 2
    g = jnp.asarray(gamma)
    return (1.0 - M2) / (g * M2) + (g + 1.0) / (2.0 * g) * jnp.log((g + 1.0) * M2 / (2.0 + (g - 1.0) * M2))


class FannoRatios(NamedTuple):
    """Fanno-flow ratios to the sonic (``*``) state at Mach ``M``."""

    P_Pstar: Array
    T_Tstar: Array
    P0_P0star: Array


def fanno_ratios(M: Array | float, gamma: Array | float = 1.4) -> FannoRatios:
    """Static pressure, temperature and stagnation-pressure ratios to the sonic state.

    ``T/T* = (gamma+1)/(2 + (gamma-1) M^2)``; ``P/P* = sqrt(T/T*)/M``;
    ``P0/P0* = (1/M) [(2 + (gamma-1) M^2)/(gamma+1)]^((gamma+1)/(2(gamma-1)))``.

    Args:
        M: Mach number (-).
        gamma: Heat-capacity ratio (-).

    Returns:
        :class:`FannoRatios`.
    """
    M = jnp.asarray(M)
    g = jnp.asarray(gamma)
    TT = (g + 1.0) / (2.0 + (g - 1.0) * M**2)
    P0 = (1.0 / M) * (1.0 / TT) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    return FannoRatios(jnp.sqrt(TT) / M, TT, P0)


class FannoOutlet(NamedTuple):
    """Outlet of an adiabatic duct (:func:`adiabatic_outlet_mach`)."""

    M2: Array
    """Outlet Mach number (-); 1 if choked."""
    P2_P1: Array
    """Static pressure ratio outlet/inlet (-)."""
    T2_T1: Array
    """Static temperature ratio outlet/inlet (-)."""
    choked: Array
    """True if ``f L/D`` exceeds ``fanno_parameter(M1)``: the inlet Mach cannot be sustained."""


def adiabatic_outlet_mach(
    M1: Array | float, fL_D: Array | float, gamma: Array | float = 1.4
) -> FannoOutlet:
    """Outlet state of subsonic adiabatic flow with friction in a constant-area duct.

    ``fanno_parameter(M2) = fanno_parameter(M1) - f L/D`` solved on the
    subsonic branch ``M1 <= M2 <= 1`` by bisection, with gradients by the
    implicit function theorem. If the duct is longer than ``L*`` for ``M1``
    (``choked``), ``M2`` is reported as 1; physically the inlet flow would
    fall until the duct just chokes.

    Args:
        M1: Inlet Mach number (-), subsonic.
        fL_D: ``f L / D`` with the Darcy ``f`` (-).
        gamma: Heat-capacity ratio (-).

    Returns:
        :class:`FannoOutlet`.
    """
    M1 = jnp.asarray(M1, dtype=jnp.float64)
    g = jnp.asarray(gamma)
    target = fanno_parameter(M1, g) - jnp.asarray(fL_D)
    choked = target <= 0.0

    def F(M):
        return fanno_parameter(M, g) - target

    root = _bisect_log(F, jax.lax.stop_gradient(M1), jnp.asarray(1.0 - 1e-12))
    safe = jnp.where(choked, 0.5 * (M1 + 1.0), root)  # keep F' away from its zero at M = 1
    M2 = jnp.where(choked, 1.0, _implicit(F, safe))
    r1, r2 = fanno_ratios(M1, g), fanno_ratios(M2, g)
    return FannoOutlet(M2, r2.P_Pstar / r1.P_Pstar, r2.T_Tstar / r1.T_Tstar, choked)


# =============================================================================
# Compressors: polytropic efficiency and staging
# =============================================================================


def polytropic_exponent(
    eta_p: Array | float, gamma: Array | float = 1.4, compression: bool = True
) -> Array:
    """Polytropic exponent ``n`` of an ideal-gas machine with polytropic efficiency ``eta_p``.

    The path ``P v^n = const`` of a machine whose every infinitesimal stage has
    efficiency ``eta_p`` has::

        compression: (n - 1)/n = (gamma - 1) / (gamma eta_p)
        expansion:   (n - 1)/n = eta_p (gamma - 1) / gamma

    so ``T2/T1 = (P2/P1)^((n-1)/n)``. ``eta_p = 1`` gives ``n = gamma``.
    (Saravanamuttoo, Rogers & Cohen, *Gas Turbine Theory*, Sec. 2.2; Dixon &
    Hall, *Fluid Mechanics and Thermodynamics of Turbomachinery*, Ch. 2.)

    Args:
        eta_p: Polytropic (small-stage) efficiency in (0, 1].
        gamma: Heat-capacity ratio (-).
        compression: True for a compressor, False for a turbine/expander.

    Returns:
        ``n`` (-). The temperature exponent is ``(n - 1)/n = 1 - 1/n``.

    Example:
        >>> n = polytropic_exponent(0.80, 1.4)
        >>> round(float((n - 1) / n), 5)    # 0.4 / (1.4 * 0.8)
        0.35714
    """
    g = jnp.asarray(gamma)
    k = (g - 1.0) / g
    m = k / jnp.asarray(eta_p) if compression else k * jnp.asarray(eta_p)
    return 1.0 / (1.0 - m)


def isentropic_efficiency_from_polytropic(
    pressure_ratio: Array | float,
    eta_p: Array | float,
    gamma: Array | float = 1.4,
    compression: bool = True,
) -> Array:
    """Overall isentropic efficiency of an ideal-gas machine of polytropic efficiency ``eta_p``.

    With ``k = (gamma-1)/gamma`` and ``r`` the pressure ratio (> 1, high over
    low)::

        compression: eta_s = (r^k - 1) / (r^(k/eta_p) - 1)
        expansion:   eta_s = (1 - r^(-eta_p k)) / (1 - r^(-k))

    For a compressor ``eta_s < eta_p`` and the gap grows with ``r`` (the
    preheat effect); for a turbine ``eta_s > eta_p``.

    Args:
        pressure_ratio: ``r`` = high/low pressure (-), > 1.
        eta_p: Polytropic efficiency (-).
        gamma: Heat-capacity ratio (-).
        compression: True for a compressor, False for an expander.

    Returns:
        Isentropic efficiency (-).

    Example:
        >>> round(float(isentropic_efficiency_from_polytropic(4.0, 0.80, 1.4)), 3)
        0.759
    """
    r = jnp.asarray(pressure_ratio)
    g = jnp.asarray(gamma)
    k = (g - 1.0) / g
    e = jnp.asarray(eta_p)
    if compression:
        return (r**k - 1.0) / (r ** (k / e) - 1.0)
    return (1.0 - r ** (-e * k)) / (1.0 - r ** (-k))


def polytropic_efficiency_from_isentropic(
    pressure_ratio: Array | float,
    eta_s: Array | float,
    gamma: Array | float = 1.4,
    compression: bool = True,
) -> Array:
    """Polytropic efficiency matching an overall isentropic efficiency (ideal gas).

    The closed-form inverse of :func:`isentropic_efficiency_from_polytropic`::

        compression: eta_p = k ln r / ln(1 + (r^k - 1)/eta_s)
        expansion:   eta_p = -ln(1 - eta_s (1 - r^(-k))) / (k ln r)

    Args:
        pressure_ratio: ``r`` = high/low pressure (-), > 1.
        eta_s: Isentropic efficiency (-).
        gamma: Heat-capacity ratio (-).
        compression: True for a compressor, False for an expander.

    Returns:
        Polytropic efficiency (-).
    """
    r = jnp.asarray(pressure_ratio)
    g = jnp.asarray(gamma)
    k = (g - 1.0) / g
    e = jnp.asarray(eta_s)
    if compression:
        return k * jnp.log(r) / jnp.log1p((r**k - 1.0) / e)
    return -jnp.log1p(-e * (1.0 - r ** (-k))) / (k * jnp.log(r))


def optimal_stage_ratios(P_in: Array | float, P_out: Array | float, n_stages: int) -> Array:
    """Per-stage pressure ratios minimizing the work of an intercooled train.

    For an ideal gas with intercooling back to the suction temperature, equal
    stage efficiencies and no interstage pressure loss, the total work is
    minimized when every stage has the same ratio ``(P_out/P_in)^(1/N)``
    (McCabe, Smith & Harriott, *Unit Operations of Chemical Engineering*, 7e,
    Ch. 8). With a real gas, unequal intercooler temperatures or interstage
    losses the optimum shifts; optimize the stage pressures directly (the
    work is differentiable) and use these as the starting point.

    Args:
        P_in: Suction pressure (Pa).
        P_out: Final discharge pressure (Pa).
        n_stages: Number of stages ``N`` (a Python int).

    Returns:
        Array of ``N`` equal ratios. Stage discharge pressures are
        ``P_in * jnp.cumprod(ratios)``.

    Example:
        >>> round(float(optimal_stage_ratios(1e5, 110e5, 4)[0]), 4)
        3.2385
    """
    n = int(n_stages)
    if n < 1:
        raise ValueError(f"n_stages must be >= 1, got {n_stages}")
    r = (jnp.asarray(P_out, dtype=jnp.float64) / jnp.asarray(P_in)) ** (1.0 / n)
    return jnp.full((n,), r)


__all__ = [
    "R_GAS",
    "speed_of_sound",
    "mach_number",
    "StagnationRatios",
    "stagnation_ratios",
    "critical_pressure_ratio",
    "choked_mass_flux",
    "nozzle_mass_flow",
    "NozzleFlow",
    "nozzle_flow",
    "isothermal_pipe_mass_flux",
    "IsothermalPipeResult",
    "isothermal_outlet_pressure",
    "isothermal_limit_mach",
    "isothermal_distance",
    "isothermal_pressure_profile",
    "IsothermalPipeFlow",
    "isothermal_pipe_flow",
    "fanno_parameter",
    "FannoRatios",
    "fanno_ratios",
    "FannoOutlet",
    "adiabatic_outlet_mach",
    "polytropic_exponent",
    "isentropic_efficiency_from_polytropic",
    "polytropic_efficiency_from_isentropic",
    "optimal_stage_ratios",
]
