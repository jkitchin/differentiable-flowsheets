"""Threshold fouling: the Ebert-Panchal rate model, in core and differentiable.

The fouling resistance of a heat-exchanger tube grows by deposition (a chemical
reaction in the thermal boundary layer, so Arrhenius in the *film* temperature)
and is limited by removal, which goes with the wall shear stress (Ebert and
Panchal, 1995)::

    dR_f/dt = alpha Re^beta Pr^(-0.33) exp(-E / (R T_film)) - gamma tau_w

with ``beta = -0.66``. Below the *threshold* (the film temperature and wall
shear at which the two terms cancel) the surface does not foul. ``alpha``,
``gamma`` and ``E`` are **user-supplied and fluid-specific**; there are no
defaults here. Any number in the docs or tests is illustrative and is
labelled so.

This is the same form as :mod:`difflow_refinery.preheat.fouling` (the crude
preheat plugin keeps its own copy, with its own illustrative constants, and is
not modified by this module); for the same constants
:func:`ebert_panchal_rate` returns the same number, which a test checks when the
plugin is importable. Core heat-transfer work can use this module without
importing a refinery plugin.

Contents
--------
* Inputs: :func:`wall_shear_stress`, :func:`film_temperature`
* Rate and threshold: :func:`ebert_panchal_rate`, :func:`threshold_wall_shear`,
  :func:`threshold_film_temperature`
* Time: :func:`fouling_time`, :func:`integrate_fouling`

Units: ``dR_f/dt`` is m^2 K/W per second, ``alpha`` then has units
m^2 K/W/s (for ``Re`` and ``Pr`` dimensionless), ``gamma`` m^2 K/W/s/Pa,
``E`` J/mol. Everything is JAX, differentiable in every numeric argument.

Reference:
    Ebert, W. and Panchal, C. B. (1995). Analysis of Exxon crude-oil slip
    stream coking data. In *Fouling Mitigation of Industrial Heat-Exchange
    Equipment*, Begell House. Recalled from the literature and not re-checked
    against the paper; the form of the rate equation is the one the refinery
    plugin also uses. The ``0.55`` film-temperature weight of
    :func:`film_temperature` is likewise a convention taken from that plugin,
    **unverified**.

Example:
    >>> from difflow.fouling import ebert_panchal_rate
    >>> r = ebert_panchal_rate(Re=3.0e4, Pr=10.0, T_film=600.0, tau_w=2.0,
    ...                        alpha=2.0e-3, gamma=4.0e-13, E=48.0e3)  # illustrative constants
    >>> float(r) > 0
    True
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

#: Universal gas constant (J/mol/K).
R_GAS = 8.314462618

#: Reynolds-number exponent of the deposition term (Ebert-Panchal).
BETA_RE = -0.66
#: Prandtl-number exponent of the deposition term (Ebert-Panchal).
EXPONENT_PR = -0.33
#: Weight of the wall in the film temperature ``T_b + w (T_w - T_b)``
#: (the value the refinery plugin uses; unverified).
FILM_WEIGHT = 0.55

_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731


def wall_shear_stress(f: Array | float, rho: Array | float, v: Array | float) -> Array:
    """Wall shear stress ``tau_w = (f/8) rho v^2`` (Pa) from the **Darcy** friction factor.

    ``f`` is e.g. from :func:`difflow.fluids.friction_factor` (the same factor as
    the tube-side pressure drop), ``v`` the mean velocity.

    Args:
        f: Darcy friction factor (-).
        rho: Density (kg/m^3).
        v: Mean velocity (m/s).

    Returns:
        Wall shear stress (Pa).
    """
    return _f64(f) / 8.0 * _f64(rho) * _f64(v) ** 2


def film_temperature(T_bulk: Array | float, T_wall: Array | float, weight: float = FILM_WEIGHT) -> Array:
    """Film temperature ``T_b + weight (T_w - T_b)`` (K) that the deposition term sees.

    Args:
        T_bulk: Bulk fluid temperature (K).
        T_wall: Wall (fluid-side surface) temperature (K).
        weight: Position in the film; 0.55 is the convention of the refinery
            plugin (unverified).

    Returns:
        Film temperature (K).
    """
    return _f64(T_bulk) + weight * (_f64(T_wall) - _f64(T_bulk))


def _deposition(Re, Pr, T_film, alpha, E, beta):
    return _f64(alpha) * _f64(Re) ** beta * _f64(Pr) ** EXPONENT_PR * jnp.exp(-_f64(E) / (R_GAS * _f64(T_film)))


def ebert_panchal_rate(
    Re: Array | float,
    Pr: Array | float,
    T_film: Array | float,
    tau_w: Array | float,
    alpha: Array | float,
    gamma: Array | float,
    E: Array | float,
    beta: float = BETA_RE,
    floor_at_zero: bool = False,
) -> Array:
    """Ebert-Panchal threshold fouling rate ``dR_f/dt`` (m^2 K/W per s).

    ``dR_f/dt = alpha Re^beta Pr^(-0.33) exp(-E/(R T_film)) - gamma tau_w``.
    The value is negative below the threshold (removal exceeds deposition).
    With ``floor_at_zero=True`` it is ``max(., 0)``, the usual "a clean wall
    does not foul" reading; the floor has a kink at the threshold, so keep it
    off when you differentiate through the threshold.

    Args:
        Re: Reynolds number (-).
        Pr: Prandtl number (-).
        T_film: Film temperature (K), see :func:`film_temperature`.
        tau_w: Wall shear stress (Pa), see :func:`wall_shear_stress`.
        alpha: Deposition pre-factor (user-supplied, fluid-specific).
        gamma: Removal constant (user-supplied).
        E: Activation energy (J/mol, user-supplied).
        beta: Reynolds exponent (default -0.66).
        floor_at_zero: Return ``max(rate, 0)``.

    Returns:
        ``dR_f/dt`` (m^2 K/W/s).
    """
    rate = _deposition(Re, Pr, T_film, alpha, E, beta) - _f64(gamma) * _f64(tau_w)
    return jnp.maximum(rate, 0.0) if floor_at_zero else rate


def threshold_wall_shear(
    Re: Array | float,
    Pr: Array | float,
    T_film: Array | float,
    alpha: Array | float,
    gamma: Array | float,
    E: Array | float,
    beta: float = BETA_RE,
) -> Array:
    """Wall shear stress (Pa) at which the rate is zero: ``tau_w* = deposition / gamma``.

    Above it the surface does not foul at these ``Re``, ``Pr`` and ``T_film``.
    For a velocity, solve ``wall_shear_stress(f, rho, v) = tau_w*`` (with ``Re``
    consistent with ``v``) for ``v``, for instance with
    :func:`optimistix.root_find`; ``tau_w* = f rho v^2 / 8`` gives the
    fixed-``Re`` approximation ``v* = sqrt(8 tau_w*/(f rho))``.
    """
    return _deposition(Re, Pr, T_film, alpha, E, beta) / _f64(gamma)


def threshold_film_temperature(
    Re: Array | float,
    Pr: Array | float,
    tau_w: Array | float,
    alpha: Array | float,
    gamma: Array | float,
    E: Array | float,
    beta: float = BETA_RE,
) -> Array:
    """Film temperature (K) at which the rate is zero, for a given ``Re``, ``Pr`` and ``tau_w``.

    ``T* = -E / (R ln(gamma tau_w / (alpha Re^beta Pr^-0.33)))``. Below it the
    surface does not foul. Requires ``gamma tau_w < alpha Re^beta Pr^-0.33``
    (otherwise removal exceeds deposition at every temperature and there is no
    threshold, and the result is NaN).
    """
    pre = _f64(alpha) * _f64(Re) ** beta * _f64(Pr) ** EXPONENT_PR
    ratio = _f64(gamma) * _f64(tau_w) / pre
    ok = ratio < 1.0
    safe = jnp.where(ok, ratio, 0.5)
    return jnp.where(ok, -_f64(E) / (R_GAS * jnp.log(safe)), jnp.nan)


def fouling_time(R_f_target: Array | float, rate: Array | float, R_f0: Array | float = 0.0) -> Array:
    """Time (s) for a constant fouling rate to reach ``R_f_target`` from ``R_f0``.

    ``t = (R_target - R_0)/rate`` for ``rate > 0``; ``inf`` when the rate is not
    positive (below the threshold the surface never reaches the target). The
    rate is taken constant (the usual first estimate); use
    :func:`integrate_fouling` when it changes with ``R_f``.

    Args:
        R_f_target: Target fouling resistance (m^2 K/W).
        rate: ``dR_f/dt`` (m^2 K/W/s).
        R_f0: Starting resistance (m^2 K/W).

    Returns:
        Time (s), possibly ``inf``.
    """
    rate = _f64(rate)
    safe = jnp.where(rate > 0.0, rate, 1.0)
    return jnp.where(rate > 0.0, (_f64(R_f_target) - _f64(R_f0)) / safe, jnp.inf)


def integrate_fouling(
    rate_fn: Callable[[Array], Array],
    t_end: Array | float,
    R_f0: Array | float = 0.0,
    n_steps: int = 200,
) -> dict[str, Array]:
    """Integrate ``dR_f/dt = rate_fn(R_f)`` from 0 to ``t_end`` (fixed-step RK4, ``lax.scan``).

    ``R_f`` is held at or above zero (a clean wall cannot lose more deposit).
    ``rate_fn`` may close over any traced parameters, so the final resistance
    is differentiable in them. Use it for the feedback a constant-rate
    estimate misses: a growing deposit lowers ``U``, which raises the wall
    temperature and the rate.

    Args:
        rate_fn: ``R_f -> dR_f/dt`` (m^2 K/W/s), JAX-traceable.
        t_end: End time (s).
        R_f0: Initial resistance (m^2 K/W).
        n_steps: Number of RK4 steps (a Python int).

    Returns:
        Dict with ``t`` (``n_steps + 1`` times) and ``R_f`` (the history, same length),
        and ``R_f_final``.
    """
    t_end = _f64(t_end)
    dt = t_end / n_steps

    def f(R):
        return rate_fn(R)

    def step(R, _):
        k1 = f(R)
        k2 = f(R + 0.5 * dt * k1)
        k3 = f(R + 0.5 * dt * k2)
        k4 = f(R + dt * k3)
        Rn = jnp.maximum(R + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0)
        return Rn, Rn

    R0 = _f64(R_f0)
    Rend, hist = jax.lax.scan(step, R0, None, length=n_steps)
    return {
        "t": jnp.linspace(0.0, 1.0, n_steps + 1) * t_end,
        "R_f": jnp.concatenate([R0[None], hist]),
        "R_f_final": Rend,
    }


__all__ = [
    "R_GAS",
    "BETA_RE",
    "EXPONENT_PR",
    "FILM_WEIGHT",
    "ebert_panchal_rate",
    "film_temperature",
    "fouling_time",
    "integrate_fouling",
    "threshold_film_temperature",
    "threshold_wall_shear",
    "wall_shear_stress",
]
