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
* :func:`fanno_parameter`, :func:`fanno_ratios`, :func:`adiabatic_outlet_mach`
  -- adiabatic (Fanno) flow in a constant-area duct with friction

All of it assumes an ideal gas with constant ``gamma = Cp/Cv`` and molar mass
``MW`` (kg/mol). Friction factors are **Darcy** factors, as in
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


def speed_of_sound(T: Array | float, MW: Array | float, gamma: Array | float = 1.4) -> Array:
    """Ideal-gas speed of sound ``c = sqrt(gamma R T / MW)`` (m/s).

    Args:
        T: Temperature (K).
        MW: Molar mass (kg/mol), e.g. 0.02897 for air.
        gamma: Heat-capacity ratio ``Cp/Cv`` (-).

    Returns:
        Speed of sound (m/s).

    Example:
        >>> round(float(speed_of_sound(293.15, 0.02897)))
        343
    """
    return jnp.sqrt(jnp.asarray(gamma) * R_GAS * jnp.asarray(T) / jnp.asarray(MW))


def mach_number(v: Array | float, T: Array | float, MW: Array | float, gamma: Array | float = 1.4) -> Array:
    """Mach number ``M = v / c`` (-).

    Args:
        v: Velocity (m/s).
        T: Static temperature (K).
        MW: Molar mass (kg/mol).
        gamma: Heat-capacity ratio (-).

    Returns:
        Mach number (-).
    """
    return jnp.asarray(v) / speed_of_sound(T, MW, gamma)


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

    Returns:
        Mass flux ``G`` (kg/m^2/s); the mass flow is ``G pi D^2 / 4``.
    """
    P1 = jnp.asarray(P1)
    P2 = jnp.asarray(P2)
    RT_M = R_GAS * jnp.asarray(T) / jnp.asarray(MW)
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
    """Pressure at which the velocity reaches ``sqrt(R T / MW)`` (the isothermal limit, Pa)."""


def isothermal_outlet_pressure(
    G: Array | float,
    P1: Array | float,
    T: Array | float,
    L: Array | float,
    D: Array | float,
    f: Array | float,
    MW: Array | float,
    gamma: Array | float = 1.4,
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
    RT_M = R_GAS * jnp.asarray(T) / jnp.asarray(MW)
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
    mach2 = G * RT_M / P2 / speed_of_sound(T, MW, gamma)
    return IsothermalPipeResult(P2, mach2, choked, P_choke)


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


__all__ = [
    "R_GAS",
    "speed_of_sound",
    "mach_number",
    "StagnationRatios",
    "stagnation_ratios",
    "critical_pressure_ratio",
    "choked_mass_flux",
    "nozzle_mass_flow",
    "isothermal_pipe_mass_flux",
    "IsothermalPipeResult",
    "isothermal_outlet_pressure",
    "fanno_parameter",
    "FannoRatios",
    "fanno_ratios",
    "FannoOutlet",
    "adiabatic_outlet_mach",
]
