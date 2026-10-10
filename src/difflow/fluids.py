"""Incompressible (liquid) pipe flow: friction factors, fittings, flow meters.

Everything here is JAX, SI and differentiable: ``jax.grad`` of a pressure drop
with respect to the diameter, the roughness, the flow or the fluid properties
works, including through the laminar-turbulent transition.

Contents
--------
* :func:`reynolds_number`, :func:`hydraulic_diameter`
* :func:`friction_factor` -- the **Darcy** friction factor by the Colebrook-White,
  Churchill, Haaland or Swamee-Jain correlation
* :func:`darcy_pressure_drop`, :func:`minor_loss`, :func:`fully_rough_friction_factor`
* :data:`FITTINGS` -- standard K / L/D values with sources, :func:`fitting_K`,
  :func:`equivalent_length`
* :func:`orifice_flow`, :func:`orifice_dp`, :func:`venturi_flow`, :func:`venturi_dp`

Liquid density and viscosity are *inputs* (core ``SpeciesData`` carries no
liquid-phase transport data); see :class:`difflow.units.pipe.PipeParams`.

Example:
    >>> import jax.numpy as jnp
    >>> from difflow.fluids import friction_factor
    >>> round(float(friction_factor(1e5, 0.0)), 4)      # smooth pipe, Re = 1e5
    0.018
    >>> float(friction_factor(1000.0, 0.01)) == 64.0 / 1000.0   # laminar
    True
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_sqrt

#: Standard gravity (m/s^2).
GRAVITY = 9.80665

#: Reynolds number below which the Colebrook/Haaland/Swamee-Jain methods are
#: purely laminar (64/Re); the blend to turbulent starts here.
RE_LAMINAR = 2100.0
#: Reynolds number above which those methods are purely turbulent.
RE_TURBULENT = 4000.0

FRICTION_METHODS = ("colebrook", "churchill", "haaland", "swamee_jain")

_LN10 = math.log(10.0)
_NEWTON_ITERATIONS = 8


# =============================================================================
# Dimensionless groups and geometry
# =============================================================================


def reynolds_number(rho: Array | float, v: Array | float, D: Array | float, mu: Array | float) -> Array:
    """Reynolds number ``Re = rho v D / mu``.

    Args:
        rho: Density (kg/m^3).
        v: Mean velocity (m/s).
        D: Hydraulic diameter (m).
        mu: Dynamic viscosity (Pa s).

    Returns:
        Reynolds number (-).
    """
    return jnp.asarray(rho) * jnp.asarray(v) * jnp.asarray(D) / jnp.asarray(mu)


def hydraulic_diameter(area: Array | float, wetted_perimeter: Array | float) -> Array:
    """Hydraulic diameter ``D_h = 4 A / P_wetted`` (m).

    Equals the diameter for a full circular pipe and ``D_o - D_i`` for an
    annulus (``4 * pi/4 (D_o^2 - D_i^2) / (pi (D_o + D_i))``).

    Args:
        area: Flow cross-section (m^2).
        wetted_perimeter: Wetted perimeter (m).

    Returns:
        Hydraulic diameter (m).
    """
    return 4.0 * jnp.asarray(area) / jnp.asarray(wetted_perimeter)


# =============================================================================
# Friction factor (Darcy)
# =============================================================================


def _haaland_inv_sqrt(Re: Array, rel_roughness: Array) -> Array:
    """Haaland (1983): ``1/sqrt(f) = -1.8 log10[(e/D/3.7)^1.11 + 6.9/Re]``."""
    return -1.8 * jnp.log10((rel_roughness / 3.7) ** 1.11 + 6.9 / Re)


def _colebrook_turbulent(Re: Array, rel_roughness: Array) -> Array:
    """Colebrook-White friction factor, solved implicitly.

    Solves ``g(x) = x + 2 log10(e/D/3.7 + 2.51 x/Re) = 0`` for ``x = 1/sqrt(f)``
    by a fixed number of Newton steps from the Haaland estimate (quadratic
    convergence from a start within a few percent: the residual is below 1e-14
    after 8 steps over Re 2e3..1e9, e/D 0..0.05). The iteration runs on
    stop-gradient inputs; the returned value is then corrected by one
    implicit-function-theorem step ``x = x* - g(x*; theta)/g_x`` so that
    derivatives with respect to ``Re`` and ``e/D`` are the exact implicit
    derivatives (not derivatives of the unrolled iteration).
    """
    sg = jax.lax.stop_gradient
    Re_s, e_s = sg(Re), sg(rel_roughness)

    def g(x, Re_, e_):
        return x + 2.0 * jnp.log10(e_ / 3.7 + 2.51 * x / Re_)

    def gx(x, Re_, e_):
        return 1.0 + (2.0 / _LN10) * (2.51 / Re_) / (e_ / 3.7 + 2.51 * x / Re_)

    x = jnp.maximum(_haaland_inv_sqrt(Re_s, e_s), 1.0)
    for _ in range(_NEWTON_ITERATIONS):
        x = jnp.maximum(x - g(x, Re_s, e_s) / gx(x, Re_s, e_s), 1.0)
    x = sg(x)
    x = x - g(x, Re, rel_roughness) / sg(gx(x, Re_s, e_s))
    return 1.0 / x**2


def _churchill(Re: Array, rel_roughness: Array) -> Array:
    """Churchill (1977), valid for laminar, transition and turbulent flow."""
    A = (2.457 * jnp.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * rel_roughness))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)


def _swamee_jain(Re: Array, rel_roughness: Array) -> Array:
    """Swamee-Jain (1976): ``f = 0.25 / [log10(e/D/3.7 + 5.74/Re^0.9)]^2``."""
    return 0.25 / jnp.log10(rel_roughness / 3.7 + 5.74 / Re**0.9) ** 2


def _laminar_weight(Re: Array) -> Array:
    """Weight of the turbulent branch: a C2 smootherstep in ``ln Re``.

    Exactly 0 for ``Re <= RE_LAMINAR`` and exactly 1 for ``Re >= RE_TURBULENT``,
    with zero first and second derivatives at both ends, so the blend has finite
    gradients everywhere and the laminar result is *exactly* ``64/Re``.
    """
    t = (jnp.log(Re) - math.log(RE_LAMINAR)) / (math.log(RE_TURBULENT) - math.log(RE_LAMINAR))
    t = jnp.clip(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


def friction_factor(
    Re: Array | float,
    rel_roughness: Array | float = 0.0,
    method: str = "colebrook",
) -> Array:
    """Darcy friction factor ``f`` for a full circular pipe.

    ``Delta P = f (L/D) (rho v^2 / 2)``. (The Fanning factor is ``f/4``.)

    Methods:

    * ``"colebrook"`` -- the implicit Colebrook-White equation
      ``1/sqrt(f) = -2 log10(e/D/3.7 + 2.51/(Re sqrt(f)))`` solved by Newton
      iteration with implicit-function-theorem gradients, blended with laminar
      ``64/Re``.
    * ``"haaland"``, ``"swamee_jain"`` -- explicit approximations to Colebrook,
      blended with laminar ``64/Re`` the same way.
    * ``"churchill"`` -- the single explicit Churchill (1977) expression, valid
      across laminar, transition and turbulent flow with no blend.

    The laminar-turbulent blend for the first three is
    ``f = (1 - w) 64/Re + w f_turb`` where ``w`` is a C2 smootherstep of
    ``ln Re`` between ``Re = 2100`` and ``Re = 4000`` (a compact-support
    sigmoid in log Re). Below 2100 the result is exactly ``64/Re``. Between
    2100 and 4000 real flow is intermittent and no correlation is accurate;
    the blend is a smooth interpolation, not a model of transition.

    Args:
        Re: Reynolds number (-), positive.
        rel_roughness: Relative roughness ``epsilon / D`` (-), in [0, 0.05].
        method: One of ``FRICTION_METHODS``.

    Returns:
        Darcy friction factor (-).

    Raises:
        ValueError: Unknown ``method`` (a Python string, not a traced value).

    Example:
        >>> float(friction_factor(1e5, 0.0, "churchill")) > 0
        True
    """
    if method not in FRICTION_METHODS:
        raise ValueError(f"Unknown friction method {method!r}; use one of {FRICTION_METHODS}")
    Re = jnp.maximum(jnp.asarray(Re, dtype=jnp.float64), 1e-12)
    e = jnp.asarray(rel_roughness, dtype=jnp.float64)
    Re, e = jnp.broadcast_arrays(Re, e)

    if method == "churchill":
        return _churchill(Re, e)

    # The turbulent branch is evaluated on Re >= RE_LAMINAR only; below that its
    # weight is exactly zero and this keeps it finite (no 0 * nan in gradients).
    Re_t = jnp.maximum(Re, RE_LAMINAR)
    if method == "colebrook":
        f_t = _colebrook_turbulent(Re_t, e)
    elif method == "haaland":
        f_t = 1.0 / _haaland_inv_sqrt(Re_t, e) ** 2
    else:
        f_t = _swamee_jain(Re_t, e)
    w = _laminar_weight(Re)
    return (1.0 - w) * 64.0 / Re + w * f_t


def fully_rough_friction_factor(rel_roughness: Array | float) -> Array:
    """Fully rough (Re -> infinity) Darcy factor, von Karman / Nikuradse.

    ``1/sqrt(f_T) = -2 log10(e/D / 3.7)``. This is the "fully turbulent" ``f_T``
    that Crane TP-410 uses to turn a fitting's ``K = (L/D) f_T`` into a loss
    coefficient for a given pipe.

    Args:
        rel_roughness: Relative roughness ``epsilon / D`` (-), positive.

    Returns:
        ``f_T`` (-).
    """
    return 1.0 / (2.0 * jnp.log10(3.7 / jnp.asarray(rel_roughness))) ** 2


# =============================================================================
# Pressure drops
# =============================================================================


def darcy_pressure_drop(
    f: Array | float,
    L: Array | float,
    D: Array | float,
    rho: Array | float,
    v: Array | float,
) -> Array:
    """Darcy-Weisbach frictional pressure drop ``f (L/D) rho v^2 / 2`` (Pa).

    Args:
        f: Darcy friction factor (-).
        L: Pipe length (m).
        D: Inside diameter (m).
        rho: Density (kg/m^3).
        v: Mean velocity (m/s).

    Returns:
        Pressure drop (Pa), positive in the flow direction.
    """
    return jnp.asarray(f) * jnp.asarray(L) / jnp.asarray(D) * 0.5 * jnp.asarray(rho) * jnp.asarray(v) ** 2


def minor_loss(K: Array | float, rho: Array | float, v: Array | float) -> Array:
    """Minor (fitting) loss ``K rho v^2 / 2`` (Pa).

    Args:
        K: Loss coefficient (-), based on the velocity ``v``.
        rho: Density (kg/m^3).
        v: Mean velocity (m/s).

    Returns:
        Pressure drop (Pa).
    """
    return jnp.asarray(K) * 0.5 * jnp.asarray(rho) * jnp.asarray(v) ** 2


# =============================================================================
# Fittings
# =============================================================================


class Fitting(NamedTuple):
    """A standard fitting's loss data.

    Attributes:
        kind: ``"LD"`` -- ``value`` is an equivalent length ratio ``L/D`` and
            ``K = (L/D) f_T`` (Crane's "2-K-free" form: ``f_T`` is the fully
            rough friction factor of the connected pipe); ``"K"`` -- ``value``
            is a constant K.
        value: ``L/D`` or ``K``.
        source: Where the number comes from.
    """

    kind: str
    value: float
    source: str


_CRANE = "Crane Co., Flow of Fluids Through Valves, Fittings, and Pipe, Technical Paper 410 (TP-410), "

#: Standard fitting data, fully-open valves. Entries of kind ``"LD"`` scale with
#: the pipe's fully rough friction factor; see :func:`fitting_K`.
#:
#: Provenance: these are the widely reprinted Crane TP-410 values (and the
#: tables derived from them in Perry's Chemical Engineers' Handbook, 8e/9e,
#: Sec. 6, "Losses in Fittings and Valves"). They were transcribed from those
#: reprints, not machine-checked against the TP-410 scan; verify against your
#: edition before using them for a design with tight margins. K-values for
#: fittings are accurate to roughly +/-25 percent at best (Crane's own statement).
FITTINGS: dict[str, Fitting] = {
    "elbow_90_standard": Fitting("LD", 30.0, _CRANE + "90 deg standard elbow, K = 30 f_T"),
    "elbow_90_long_radius": Fitting("LD", 14.0, _CRANE + "90 deg bend r/d = 1.5, K = 14 f_T"),
    "elbow_45_standard": Fitting("LD", 16.0, _CRANE + "45 deg standard elbow, K = 16 f_T"),
    "return_bend_180": Fitting("LD", 50.0, _CRANE + "180 deg close return bend, K = 50 f_T"),
    "tee_through": Fitting("LD", 20.0, _CRANE + "tee, flow through run, K = 20 f_T"),
    "tee_branch": Fitting("LD", 60.0, _CRANE + "tee, flow through branch, K = 60 f_T"),
    "gate_valve": Fitting("LD", 8.0, _CRANE + "gate valve, fully open, K = 8 f_T"),
    "globe_valve": Fitting("LD", 340.0, _CRANE + "globe valve, fully open, K = 340 f_T"),
    "angle_valve": Fitting("LD", 150.0, _CRANE + "angle valve, fully open, K = 150 f_T"),
    "ball_valve": Fitting("LD", 3.0, _CRANE + "ball valve, fully open, K = 3 f_T"),
    "check_valve_swing": Fitting("LD", 50.0, _CRANE + "swing check valve, fully open, K = 50 f_T (L/D)"),
    "entrance_sharp": Fitting("K", 0.5, _CRANE + "pipe entrance, sharp-edged flush, K = 0.5; also Perry's 9e Sec. 6"),
    "entrance_rounded": Fitting("K", 0.04, _CRANE + "pipe entrance, well-rounded (r/d >= 0.15), K = 0.04"),
    "exit": Fitting("K", 1.0, _CRANE + "pipe exit into a large volume, K = 1.0 (all kinetic energy lost)"),
}


def fitting_K(
    name: str,
    rel_roughness: Array | float | None = None,
    f_T: Array | float | None = None,
) -> Array:
    """Loss coefficient K of a standard fitting.

    For ``"K"`` entries the constant is returned. For ``"LD"`` entries
    ``K = (L/D) f_T`` where ``f_T`` is the fully rough friction factor, given
    directly or computed from ``rel_roughness`` by
    :func:`fully_rough_friction_factor` (so K depends on the pipe size through
    ``epsilon / D``, which keeps it differentiable in ``D``).

    Args:
        name: Key of :data:`FITTINGS`.
        rel_roughness: ``epsilon / D`` of the connected pipe (-).
        f_T: Fully turbulent friction factor; overrides ``rel_roughness``.

    Returns:
        K (-).

    Raises:
        KeyError: Unknown fitting name.
        ValueError: An ``"LD"`` fitting with neither ``rel_roughness`` nor ``f_T``.
    """
    if name not in FITTINGS:
        raise KeyError(f"Unknown fitting {name!r}; known: {sorted(FITTINGS)}")
    fit = FITTINGS[name]
    if fit.kind == "K":
        return jnp.asarray(fit.value, dtype=jnp.float64)
    if f_T is None:
        if rel_roughness is None:
            raise ValueError(f"fitting {name!r} is given as L/D; pass rel_roughness or f_T")
        f_T = fully_rough_friction_factor(rel_roughness)
    return fit.value * jnp.asarray(f_T)


def equivalent_length(K: Array | float, D: Array | float, f: Array | float) -> Array:
    """Equivalent pipe length of a fitting, ``L_eq = K D / f`` (m).

    The L/D method: a fitting with loss coefficient ``K`` produces the same
    pressure drop as a straight run ``L_eq`` of pipe with friction factor ``f``.

    Args:
        K: Loss coefficient (-).
        D: Pipe inside diameter (m).
        f: Darcy friction factor of the pipe (-).

    Returns:
        Equivalent length (m).
    """
    return jnp.asarray(K) * jnp.asarray(D) / jnp.asarray(f)


# =============================================================================
# Differential-pressure flow meters (incompressible)
# =============================================================================


def _dp_meter_flow(dP, D_pipe, d, rho, Cd):
    beta = jnp.asarray(d) / jnp.asarray(D_pipe)
    area = jnp.pi / 4.0 * jnp.asarray(d) ** 2
    return Cd * area * safe_sqrt(2.0 * jnp.asarray(dP) / (jnp.asarray(rho) * (1.0 - beta**4)))


def _dp_meter_dp(Q, D_pipe, d, rho, Cd):
    beta = jnp.asarray(d) / jnp.asarray(D_pipe)
    area = jnp.pi / 4.0 * jnp.asarray(d) ** 2
    return 0.5 * jnp.asarray(rho) * (1.0 - beta**4) * (jnp.asarray(Q) / (Cd * area)) ** 2


def orifice_flow(
    dP: Array | float,
    D_pipe: Array | float,
    d_orifice: Array | float,
    rho: Array | float,
    Cd: float = 0.61,
) -> Array:
    """Volumetric flow through a sharp-edged orifice meter (m^3/s).

    ``Q = Cd (pi/4) d^2 sqrt(2 dP / (rho (1 - beta^4)))`` with ``beta = d/D``:
    incompressible, with the velocity-of-approach factor ``1/sqrt(1 - beta^4)``
    and ``Cd`` the discharge coefficient referred to the *orifice* area.
    ``Cd = 0.61`` is the usual high-Reynolds-number value for a sharp-edged
    orifice in textbooks (Perry's 9e Sec. 10);
    ISO 5167 / Reader-Harris-Gallagher gives ``Cd`` as a function of ``beta``,
    ``Re_D`` and tap geometry (0.59-0.62 for ``beta`` 0.2-0.6) -- pass that
    when accuracy matters. ``dP`` is the measured tap-to-tap differential, not
    the (smaller) permanent pressure loss.

    Args:
        dP: Differential pressure across the taps (Pa); negative values give 0.
        D_pipe: Pipe inside diameter (m).
        d_orifice: Orifice bore diameter (m), ``< D_pipe``.
        rho: Liquid density (kg/m^3).
        Cd: Discharge coefficient (-).

    Returns:
        Volumetric flow (m^3/s).
    """
    return _dp_meter_flow(dP, D_pipe, d_orifice, rho, Cd)


def orifice_dp(
    Q: Array | float,
    D_pipe: Array | float,
    d_orifice: Array | float,
    rho: Array | float,
    Cd: float = 0.61,
) -> Array:
    """Differential pressure (Pa) across an orifice for a known flow ``Q`` (m^3/s).

    Inverse of :func:`orifice_flow`.
    """
    return _dp_meter_dp(Q, D_pipe, d_orifice, rho, Cd)


def venturi_flow(
    dP: Array | float,
    D_pipe: Array | float,
    d_throat: Array | float,
    rho: Array | float,
    Cd: float = 0.98,
) -> Array:
    """Volumetric flow through a classical Venturi meter (m^3/s).

    Same equation as :func:`orifice_flow` with the throat as ``d``; the
    discharge coefficient of a machined/cast Venturi is close to 1
    (``Cd = 0.98`` default; ISO 5167-4 gives 0.984 for a cast convergent
    section, 0.995 machined, for ``Re`` 2e5-2e6).

    Args:
        dP: Differential pressure, inlet tap to throat tap (Pa).
        D_pipe: Pipe inside diameter (m).
        d_throat: Throat diameter (m).
        rho: Liquid density (kg/m^3).
        Cd: Discharge coefficient (-).

    Returns:
        Volumetric flow (m^3/s).
    """
    return _dp_meter_flow(dP, D_pipe, d_throat, rho, Cd)


def venturi_dp(
    Q: Array | float,
    D_pipe: Array | float,
    d_throat: Array | float,
    rho: Array | float,
    Cd: float = 0.98,
) -> Array:
    """Differential pressure (Pa) across a Venturi for a known flow (m^3/s).

    Inverse of :func:`venturi_flow`.
    """
    return _dp_meter_dp(Q, D_pipe, d_throat, rho, Cd)


__all__ = [
    "GRAVITY",
    "RE_LAMINAR",
    "RE_TURBULENT",
    "FRICTION_METHODS",
    "FITTINGS",
    "Fitting",
    "reynolds_number",
    "hydraulic_diameter",
    "friction_factor",
    "fully_rough_friction_factor",
    "darcy_pressure_drop",
    "minor_loss",
    "fitting_K",
    "equivalent_length",
    "orifice_flow",
    "orifice_dp",
    "venturi_flow",
    "venturi_dp",
]
