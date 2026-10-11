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
* :func:`orifice_flow`, :func:`orifice_dp`, :func:`venturi_flow`, :func:`venturi_dp`,
  :func:`orifice_cd` (ISO 5167-2 Reader-Harris/Gallagher),
  :func:`orifice_permanent_loss`, :func:`venturi_permanent_loss`
* :data:`NPS_SCHEDULES` (ASME B36.10M), :func:`pipe_inside_diameter` /
  :func:`pipe_id`, :func:`pipe_outside_diameter`, :func:`pipe_wall_thickness`,
  :func:`standard_sizes`, :func:`snap_to_standard_size`,
  :func:`pipe_weight_per_length`
* Non-Newtonian friction (Darcy, like :func:`friction_factor`): power law
  (:func:`metzner_reed_reynolds`, :func:`power_law_friction_factor` --
  laminar 64/Re_MR and Dodge-Metzner -- :func:`power_law_pressure_drop`) and
  Bingham plastic (:func:`hedstrom_number`, :func:`bingham_friction_factor` --
  Buckingham-Reiner and Darby et al. 1992 -- :func:`bingham_pressure_drop`)
* Rotameter: :func:`rotameter_flow`, :func:`rotameter_height`,
  :func:`rotameter_float_dp`

Liquid density and viscosity are *inputs* to these functions. Look them up
with :mod:`difflow.liquid_properties` (``liquid_density(name, T)``,
``liquid_viscosity(name, T)``, ``stream_liquid_properties(stream)``);
:class:`difflow.units.pipe.Pipe` does that itself when ``rho``/``mu`` are
left as None.

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
    ``ln Re`` between ``Re = 2100`` and ``Re = 4000`` (``RE_LAMINAR``,
    ``RE_TURBULENT``; a compact-support sigmoid in log Re). Below 2100 the
    result is exactly ``64/Re``. Between 2100 and 4000 real flow is
    intermittent and no correlation is accurate; the blend is a smooth
    interpolation, not a model of transition, and results there are
    uncertain by tens of percent.

    Inside the blend ``f`` *rises* with ``Re`` (64/2100 = 0.030 at the start,
    about 0.040 at 4000), so a pressure drop at fixed flow can **fall as the
    viscosity rises**: 12 L/s of a 870 kg/m^3 crude through 5 km of 6-in
    Sch 40 steel (D = 154.05 mm) drops 217 kPa at 50 cP (Re = 1726, laminar)
    but 234 kPa at 25 cP (Re = 3450, inside the blend). That is the
    blend, not physics to rely on; check whether a design point sits in
    2100-4000 before trusting a gradient there.

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
#:
#: The butterfly, plug and foot valves were added from secondary reprints of
#: TP-410 (SimuPipe's K-factor table, https://simupipe.com/resources/k-factor-table,
#: and https://www.fluidnetworkstudio.com/reference/pipe-fitting-k-factors);
#: where a reprint gives only K at a small size, the L/D multiple stored here is
#: the one consistent with it at f_T = 0.018-0.019. The two strainers are not in
#: TP-410 at all and are typical clean-condition values.
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
    "butterfly_valve": Fitting("LD", 45.0, _CRANE + "butterfly valve, 2-8 in, K = 45 f_T; "
                               "reprinted by SimuPipe's TP-410 K-factor table (n = 45)"),
    "butterfly_valve_10_14": Fitting("LD", 35.0, _CRANE + "butterfly valve, 10-14 in, K = 35 f_T; "
                                     "fluidnetworkstudio.com reprints K = 0.55 for DN200-350"),
    "butterfly_valve_16_24": Fitting("LD", 25.0, _CRANE + "butterfly valve, 16-24 in, K = 25 f_T; "
                                     "fluidnetworkstudio.com reprints K = 0.3 for DN400+"),
    "plug_valve": Fitting("LD", 18.0, _CRANE + "plug valve, straightway, K = 18 f_T; "
                          "reprinted by SimuPipe's TP-410 K-factor table (n = 18)"),
    "foot_valve": Fitting("LD", 420.0, _CRANE + "foot valve with strainer, poppet disc, K = 420 f_T; "
                          "fluidnetworkstudio.com reprints K = 7.5 (= 420 x 0.018)"),
    "foot_valve_hinged": Fitting("LD", 75.0, _CRANE + "foot valve with strainer, hinged disc, K = 75 f_T; "
                                 "fluidnetworkstudio.com reprints K = 1.4 (= 75 x 0.019)"),
    "strainer_y": Fitting("K", 2.0, "NOT Crane (TP-410 has no strainers): typical clean Y-strainer K = 2, "
                          "fluidnetworkstudio.com 'manufacturer typical'. A partly blocked strainer is "
                          "several times worse; use the vendor's figure."),
    "strainer_basket": Fitting("K", 1.5, "NOT Crane (TP-410 has no strainers): typical clean basket "
                               "strainer K = 1.5, fluidnetworkstudio.com 'manufacturer typical'; use the "
                               "vendor's figure."),
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

    See also :func:`fittings_K`, which sums a ``{name: count}`` dict and
    accepts a numeric ``"K"`` entry for anything not in the table.
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


#: Keys of a fittings dict that hold a plain extra loss coefficient (a number
#: added to the sum, not a count): equipment such as an exchanger jacket or a
#: strainer from a vendor sheet.
EXTRA_K_KEYS = ("K", "K_extra", "other")


def fittings_K(
    fittings: dict | float | None,
    rel_roughness: Array | float | None = None,
    f_T: Array | float | None = None,
) -> Array:
    """Total loss coefficient of a set of fittings.

    Args:
        fittings: ``{name: count}`` with names from :data:`FITTINGS`, plus
            optionally a numeric ``"K"`` (or ``"K_extra"``/``"other"``) entry
            holding an extra loss coefficient that is added as is; or a single
            number (the total K); or None (no fittings).
        rel_roughness: ``epsilon / D`` of the connected pipe, for ``"LD"`` entries.
        f_T: Fully turbulent friction factor; overrides ``rel_roughness``.

    Returns:
        Total K (-).

    Example:
        >>> K = fittings_K({"elbow_90_standard": 4, "K": 8.0}, f_T=0.02)
        >>> round(float(K), 6)          # 4 x 30 x 0.02 + 8
        10.4
    """
    if fittings is None:
        return jnp.asarray(0.0)
    if not isinstance(fittings, dict):
        return jnp.asarray(fittings, dtype=jnp.float64)
    K = jnp.asarray(0.0)
    for name, count in fittings.items():
        if name in EXTRA_K_KEYS:
            K = K + jnp.asarray(count, dtype=jnp.float64)
        else:
            K = K + count * fitting_K(name, rel_roughness=rel_roughness, f_T=f_T)
    return K


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


ORIFICE_TAPS = ("corner", "flange", "D-D/2")


def orifice_cd(
    beta: Array | float,
    Re_D: Array | float,
    D: Array | float,
    taps: str = "flange",
) -> Array:
    """Orifice discharge coefficient, Reader-Harris/Gallagher (ISO 5167-2:2003).

    ``C = 0.5961 + 0.0261 b^2 - 0.216 b^8 + 0.000521 (1e6 b/Re_D)^0.7
    + (0.0188 + 0.0063 A) b^3.5 (1e6/Re_D)^0.3
    + (0.043 + 0.080 e^{-10 L1} - 0.123 e^{-7 L1})(1 - 0.11 A) b^4/(1 - b^4)
    - 0.031 (M2' - 0.8 M2'^1.1) b^1.3``, with ``A = (19000 b/Re_D)^0.8``,
    ``M2' = 2 L2'/(1 - b)``, and, for ``D < 71.12 mm``, the extra term
    ``+0.011 (0.75 - b)(2.8 - D/25.4 mm)``. Tap positions: corner
    ``L1 = L2' = 0``; D and D/2 ``L1 = 1, L2' = 0.47``; flange
    ``L1 = L2' = 25.4 mm / D``.

    Source: ISO 5167-2:2003 Eq. (4), as transcribed in the documentation and
    code of the open-source ``fluids`` library (C. Bell,
    ``fluids.flow_meter.C_Reader_Harris_Gallagher``; its docstring prints
    0.11 for the small-pipe term, its code and the standard use 0.011). This
    function is the plain ISO form, without the low-Re ``max`` terms
    ``fluids`` adds; the two agree inside the standard's range.

    Validity (ISO 5167-2): ``50 mm <= D <= 1000 mm``, ``d >= 12.5 mm``,
    ``0.1 <= beta <= 0.75``, ``Re_D >= 5000`` (and ``>= 16000 beta^2`` for
    ``beta > 0.56``, corner and D-D/2 taps; ``>= 170000 beta^2 D[m]`` for
    flange taps). Outside that it is an extrapolation and no warning is given.

    Args:
        beta: Diameter ratio ``d/D`` (-).
        Re_D: Pipe Reynolds number ``4 rho Q / (pi D mu)`` (-).
        D: Pipe inside diameter (m).
        taps: ``"flange"``, ``"corner"`` or ``"D-D/2"``.

    Returns:
        Discharge coefficient C (-), referred to the orifice area, for use
        with the velocity-of-approach factor as in :func:`orifice_flow`.

    Example:
        >>> # fluids' example: D = 73.91 mm, d = 22.2 mm, air, 0.12 kg/s, flange taps
        >>> import math
        >>> Re = 4 * 0.12 / (math.pi * 0.07391 * 1.85e-5)
        >>> round(float(orifice_cd(0.0222 / 0.07391, Re, 0.07391, "flange")), 6)
        0.599033
    """
    if taps not in ORIFICE_TAPS:
        raise ValueError(f"taps must be one of {ORIFICE_TAPS}, got {taps!r}")
    b = jnp.asarray(beta, dtype=jnp.float64)
    Re = jnp.asarray(Re_D, dtype=jnp.float64)
    D = jnp.asarray(D, dtype=jnp.float64)
    if taps == "corner":
        L1 = L2 = jnp.asarray(0.0)
    elif taps == "D-D/2":
        L1, L2 = jnp.asarray(1.0), jnp.asarray(0.47)
    else:
        L1 = L2 = 0.0254 / D
    A = (19000.0 * b / Re) ** 0.8
    M2 = 2.0 * L2 / (1.0 - b)
    C = (
        0.5961 + 0.0261 * b**2 - 0.216 * b**8
        + 0.000521 * (1e6 * b / Re) ** 0.7
        + (0.0188 + 0.0063 * A) * b**3.5 * (1e6 / Re) ** 0.3
        + (0.043 + 0.080 * jnp.exp(-10.0 * L1) - 0.123 * jnp.exp(-7.0 * L1))
        * (1.0 - 0.11 * A) * b**4 / (1.0 - b**4)
        - 0.031 * (M2 - 0.8 * M2**1.1) * b**1.3
    )
    small = 0.011 * (0.75 - b) * (2.8 - D / 0.0254)
    return C + jnp.where(D < 0.07112, small, 0.0)


def _orifice_flow_iso(dP, D_pipe, d, rho, mu, taps):
    """Orifice flow with the Reader-Harris/Gallagher C, solved for Re_D.

    The flow and C are coupled through Re_D; this solves
    ``r(C) = C - C_RHG(Re_D(Q(C))) = 0`` by Newton (optimistix, implicit
    adjoint), so derivatives are implicit-function-theorem derivatives.
    """
    import optimistix as optx

    D_pipe = jnp.asarray(D_pipe, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    beta = d / D_pipe
    args = (jnp.asarray(dP, dtype=jnp.float64), D_pipe, d, jnp.asarray(rho, dtype=jnp.float64),
            jnp.asarray(mu, dtype=jnp.float64))

    def flow(C, a):
        dP_, D_, d_, rho_, _ = a
        return _dp_meter_flow(dP_, D_, d_, rho_, C)

    def resid(C, a):
        _, D_, _, rho_, mu_ = a
        Q = flow(C, a)
        Re = 4.0 * rho_ * jnp.maximum(Q, 1e-300) / (jnp.pi * D_ * mu_)
        return C - orifice_cd(beta, Re, D_, taps)

    sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-12), jnp.asarray(0.6), args,
                         max_steps=50, throw=False)
    return flow(sol.value, args)


def orifice_flow(
    dP: Array | float,
    D_pipe: Array | float,
    d_orifice: Array | float,
    rho: Array | float,
    Cd: float | str = 0.61,
    mu: Array | float | None = None,
    taps: str = "flange",
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
    the (smaller) permanent pressure loss (:func:`orifice_permanent_loss`).

    ``Cd="iso5167"`` takes ``C`` from :func:`orifice_cd` (Reader-Harris/
    Gallagher) at the flow's own Reynolds number, which needs ``mu`` and
    ``taps``; the coupled equations are solved by Newton with implicit
    differentiation.

    Args:
        dP: Differential pressure across the taps (Pa); negative values give 0.
        D_pipe: Pipe inside diameter (m).
        d_orifice: Orifice bore diameter (m), ``< D_pipe``.
        rho: Liquid density (kg/m^3).
        Cd: Discharge coefficient (-), or ``"iso5167"``.
        mu: Viscosity (Pa s); required with ``Cd="iso5167"``.
        taps: Tap arrangement for ``Cd="iso5167"``: ``"flange"``,
            ``"corner"`` or ``"D-D/2"``.

    Returns:
        Volumetric flow (m^3/s).
    """
    if isinstance(Cd, str):
        if Cd.lower() != "iso5167":
            raise ValueError(f"Cd must be a number or 'iso5167', got {Cd!r}")
        if mu is None:
            raise ValueError("Cd='iso5167' needs the viscosity mu (Reynolds-number dependent C)")
        return _orifice_flow_iso(dP, D_pipe, d_orifice, rho, mu, taps)
    return _dp_meter_flow(dP, D_pipe, d_orifice, rho, Cd)


def orifice_dp(
    Q: Array | float,
    D_pipe: Array | float,
    d_orifice: Array | float,
    rho: Array | float,
    Cd: float | str = 0.61,
    mu: Array | float | None = None,
    taps: str = "flange",
) -> Array:
    """Differential pressure (Pa) across an orifice for a known flow ``Q`` (m^3/s).

    Inverse of :func:`orifice_flow`. With ``Cd="iso5167"`` the
    Reader-Harris/Gallagher C is evaluated explicitly at ``Re_D`` of ``Q``
    (needs ``mu``).
    """
    if isinstance(Cd, str):
        if Cd.lower() != "iso5167":
            raise ValueError(f"Cd must be a number or 'iso5167', got {Cd!r}")
        if mu is None:
            raise ValueError("Cd='iso5167' needs the viscosity mu (Reynolds-number dependent C)")
        D = jnp.asarray(D_pipe, dtype=jnp.float64)
        Re = 4.0 * jnp.asarray(rho) * jnp.asarray(Q) / (jnp.pi * D * jnp.asarray(mu))
        Cd = orifice_cd(jnp.asarray(d_orifice) / D, Re, D, taps)
    return _dp_meter_dp(Q, D_pipe, d_orifice, rho, Cd)


def orifice_permanent_loss(dP: Array | float, beta: Array | float, Cd: Array | float = 0.61) -> Array:
    """Unrecovered (permanent) pressure loss of an orifice meter (Pa).

    ISO 5167-2:2003 clause 5.4.1:
    ``dw = (sqrt(1 - b^4 (1 - C^2)) - C b^2) / (sqrt(1 - b^4 (1 - C^2)) + C b^2) dP``,
    measured between about 1 D upstream and 6 D downstream. This is the loss
    that costs pumping energy; the tap differential ``dP`` is larger.

    Args:
        dP: Measured differential across the taps (Pa).
        beta: Diameter ratio ``d/D`` (-).
        Cd: Discharge coefficient (-).

    Returns:
        Permanent pressure loss (Pa).

    Example:
        >>> round(float(orifice_permanent_loss(1.0, 0.5, 0.61)), 3)
        0.731
    """
    b2 = jnp.asarray(beta, dtype=jnp.float64) ** 2
    C = jnp.asarray(Cd, dtype=jnp.float64)
    root = jnp.sqrt(1.0 - b2**2 * (1.0 - C**2))
    return (root - C * b2) / (root + C * b2) * jnp.asarray(dP)


#: Venturi-tube permanent loss as a fraction of the differential, ISO 5167-4
#: (pressure-loss figure; also ASME MFC-3M), at the two ends of the standard's
#: size range: ``(beta, ratio at D = 65 mm, ratio at D = 500 mm)``. Digitized
#: by the open-source ``fluids`` library (``fluids.flow_meter.dP_venturi_tube``,
#: C. Bell, MIT licence) and resampled here at beta steps of 0.05.
VENTURI_LOSS_TABLE = (
    (0.30, 0.1645, 0.0892),
    (0.35, 0.1617, 0.0872),
    (0.40, 0.1579, 0.0845),
    (0.45, 0.1515, 0.0804),
    (0.50, 0.1429, 0.0751),
    (0.55, 0.1343, 0.0679),
    (0.60, 0.1243, 0.0606),
    (0.65, 0.1135, 0.0540),
    (0.70, 0.1010, 0.0475),
    (0.75, 0.0876, 0.0399),
)


def venturi_permanent_loss(dP: Array | float, beta: Array | float, D_pipe: Array | float) -> Array:
    """Unrecovered (permanent) pressure loss of a classical Venturi tube (Pa).

    The ratio of permanent loss to tap differential is interpolated in
    :data:`VENTURI_LOSS_TABLE`, linearly in ``beta`` (0.3-0.75, clamped) and in
    ``D`` between 65 mm (rougher, higher loss) and 500 mm (clamped outside).
    It is 4-16 % of the differential, against 40-90 % for an orifice
    (:func:`orifice_permanent_loss`). The standard's Reynolds-number effect
    (higher loss below Re about 6e5) is not included.

    Args:
        dP: Measured differential, inlet to throat (Pa).
        beta: Throat-to-pipe diameter ratio (-).
        D_pipe: Pipe inside diameter (m).

    Returns:
        Permanent pressure loss (Pa).
    """
    tab = jnp.asarray(VENTURI_LOSS_TABLE)
    b = jnp.asarray(beta, dtype=jnp.float64)
    hi = jnp.interp(b, tab[:, 0], tab[:, 1])
    lo = jnp.interp(b, tab[:, 0], tab[:, 2])
    w = jnp.clip((jnp.asarray(D_pipe, dtype=jnp.float64) - 0.065) / (0.5 - 0.065), 0.0, 1.0)
    return (hi + w * (lo - hi)) * jnp.asarray(dP)


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


# =============================================================================
# Nominal pipe sizes (ASME B36.10M)
# =============================================================================

#: Welded and seamless wrought steel pipe dimensions, ASME B36.10M:
#: ``{NPS: (outside diameter in, {schedule: wall thickness in})}`` for NPS
#: 1/8-24 and schedules 10, 40, 80, 160, STD and XS. Transcribed from the
#: tables of the Wikipedia article "Nominal Pipe Size" (which reproduces
#: B36.10M/B36.19M), cross-checked against Engineering ToolBox (d_42) and,
#: for NPS 10 and up, projectmaterials.com's B36.10M chart. For NPS <= 12 the
#: schedule 10 walls are the values those tables list, which equal the ASME
#: B36.19M 10S walls (stainless). NPS 22 has no schedule 40. Inside diameter
#: = OD - 2 wall.
NPS_SCHEDULES: dict[float, tuple[float, dict[str, float]]] = {
    0.125: (0.405, {'10': 0.049, '40': 0.068, '80': 0.095, 'STD': 0.068, 'XS': 0.095}),
    0.25: (0.54, {'10': 0.065, '40': 0.088, '80': 0.119, 'STD': 0.088, 'XS': 0.119}),
    0.375: (0.675, {'10': 0.065, '40': 0.091, '80': 0.126, 'STD': 0.091, 'XS': 0.126}),
    0.5: (0.84, {'10': 0.083, '40': 0.109, '80': 0.147, '160': 0.188, 'STD': 0.109, 'XS': 0.147}),
    0.75: (1.05, {'10': 0.083, '40': 0.113, '80': 0.154, '160': 0.219, 'STD': 0.113, 'XS': 0.154}),
    1.0: (1.315, {'10': 0.109, '40': 0.133, '80': 0.179, '160': 0.25, 'STD': 0.133, 'XS': 0.179}),
    1.25: (1.66, {'10': 0.109, '40': 0.14, '80': 0.191, '160': 0.25, 'STD': 0.14, 'XS': 0.191}),
    1.5: (1.9, {'10': 0.109, '40': 0.145, '80': 0.2, '160': 0.281, 'STD': 0.145, 'XS': 0.2}),
    2.0: (2.375, {'10': 0.109, '40': 0.154, '80': 0.218, '160': 0.344, 'STD': 0.154, 'XS': 0.218}),
    2.5: (2.875, {'10': 0.12, '40': 0.203, '80': 0.276, '160': 0.375, 'STD': 0.203, 'XS': 0.276}),
    3.0: (3.5, {'10': 0.12, '40': 0.216, '80': 0.3, '160': 0.438, 'STD': 0.216, 'XS': 0.3}),
    3.5: (4.0, {'10': 0.12, '40': 0.226, '80': 0.318, 'STD': 0.226, 'XS': 0.318}),
    4.0: (4.5, {'10': 0.12, '40': 0.237, '80': 0.337, '160': 0.531, 'STD': 0.237, 'XS': 0.337}),
    5.0: (5.563, {'10': 0.134, '40': 0.258, '80': 0.375, '160': 0.625, 'STD': 0.258, 'XS': 0.375}),
    6.0: (6.625, {'10': 0.134, '40': 0.28, '80': 0.432, '160': 0.719, 'STD': 0.28, 'XS': 0.432}),
    8.0: (8.625, {'10': 0.148, '40': 0.322, '80': 0.5, '160': 0.906, 'STD': 0.322, 'XS': 0.5}),
    10.0: (10.75, {'10': 0.165, '40': 0.365, '80': 0.594, '160': 1.125, 'STD': 0.365, 'XS': 0.5}),
    12.0: (12.75, {'10': 0.18, '40': 0.406, '80': 0.688, '160': 1.312, 'STD': 0.375, 'XS': 0.5}),
    14.0: (14.0, {'10': 0.25, '40': 0.438, '80': 0.75, '160': 1.406, 'STD': 0.375, 'XS': 0.5}),
    16.0: (16.0, {'10': 0.25, '40': 0.5, '80': 0.844, '160': 1.594, 'STD': 0.375, 'XS': 0.5}),
    18.0: (18.0, {'10': 0.25, '40': 0.562, '80': 0.938, '160': 1.781, 'STD': 0.375, 'XS': 0.5}),
    20.0: (20.0, {'10': 0.25, '40': 0.594, '80': 1.031, '160': 1.969, 'STD': 0.375, 'XS': 0.5}),
    22.0: (22.0, {'10': 0.25, '80': 1.125, '160': 2.125, 'STD': 0.375, 'XS': 0.5}),
    24.0: (24.0, {'10': 0.25, '40': 0.688, '80': 1.219, '160': 2.344, 'STD': 0.375, 'XS': 0.5}),
}

_INCH = 0.0254


def _parse_nps(nps) -> float:
    """NPS as a float: 2, 2.0, "2", '2"', "1/2", "1-1/2", "1 1/2", "1.5"."""
    if isinstance(nps, (int, float)):
        x = float(nps)
    else:
        t = str(nps).strip().replace('"', "").replace("in", "").strip()
        t = t.replace("-", " ")
        parts = t.split()
        x = 0.0
        try:
            for part in parts:
                if "/" in part:
                    num, den = part.split("/")
                    x += float(num) / float(den)
                else:
                    x += float(part)
        except ValueError:
            raise KeyError(f"cannot read a nominal pipe size from {nps!r}") from None
    for k in NPS_SCHEDULES:
        if abs(k - x) < 1e-9:
            return k
    raise KeyError(f"NPS {nps!r} is not in the table; sizes: {list(NPS_SCHEDULES)}")


def _parse_schedule(schedule) -> str:
    t = str(schedule).strip().upper().replace("SCHEDULE", "").replace("SCH", "").strip()
    t = {"STANDARD": "STD", "XH": "XS", "EXTRA STRONG": "XS"}.get(t, t)
    return t


def pipe_outside_diameter(nps) -> float:
    """Outside diameter (m) of a nominal pipe size (ASME B36.10M).

    Args:
        nps: Nominal size in inches: a number or a string such as ``"1-1/2"``.

    Returns:
        OD (m).
    """
    return NPS_SCHEDULES[_parse_nps(nps)][0] * _INCH


def pipe_wall_thickness(nps, schedule="40") -> float:
    """Wall thickness (m) of a nominal pipe size and schedule (ASME B36.10M).

    Args:
        nps: Nominal size (in).
        schedule: ``"10"``, ``"40"``, ``"80"``, ``"160"``, ``"STD"`` or
            ``"XS"`` (``40``, ``"Sch 40"`` also work).

    Returns:
        Wall (m).

    Raises:
        KeyError: Size or schedule not tabulated (e.g. NPS 22 Sch 40).
    """
    key = _parse_nps(nps)
    sch = _parse_schedule(schedule)
    walls = NPS_SCHEDULES[key][1]
    if sch not in walls:
        raise KeyError(f"NPS {key:g} has no schedule {schedule!r}; available: {sorted(walls)}")
    return walls[sch] * _INCH


def pipe_inside_diameter(nps, schedule="40") -> float:
    """Inside diameter (m) of a nominal pipe size and schedule (ASME B36.10M).

    ``ID = OD - 2 wall``. Use it instead of the nominal size: 2-in Sch 40 is
    52.50 mm inside, not 50.8 mm, an 18 % difference in turbulent pressure drop.

    Args:
        nps: Nominal size (in): ``2``, ``"2"``, ``"1/2"``, ``"1-1/2"``.
        schedule: ``"10"``, ``"40"``, ``"80"``, ``"160"``, ``"STD"``, ``"XS"``.

    Returns:
        ID (m). A plain float (table data); differentiate with respect to a
        continuous ``D`` instead and snap with :func:`snap_to_standard_size`.

    Example:
        >>> round(pipe_inside_diameter(2) * 1e3, 2)
        52.5
        >>> round(pipe_inside_diameter("1-1/2", "80") * 1e3, 2)
        38.1
    """
    key = _parse_nps(nps)
    return NPS_SCHEDULES[key][0] * _INCH - 2.0 * pipe_wall_thickness(key, schedule)


#: Short alias of :func:`pipe_inside_diameter`.
pipe_id = pipe_inside_diameter


def standard_sizes(schedule="40") -> list[tuple[float, float]]:
    """All tabulated ``(NPS, inside diameter m)`` for a schedule, smallest first.

    Example:
        >>> [n for n, _ in standard_sizes("40")][:4]
        [0.125, 0.25, 0.375, 0.5]
    """
    sch = _parse_schedule(schedule)
    out = [(n, pipe_inside_diameter(n, sch)) for n, (_, w) in NPS_SCHEDULES.items() if sch in w]
    if not out:
        raise KeyError(f"unknown schedule {schedule!r}")
    return sorted(out, key=lambda t: t[1])


def snap_to_standard_size(D: float, schedule="40", direction: str = "up") -> tuple[float, float]:
    """Round a continuous inside diameter to a standard pipe size.

    Args:
        D: Inside diameter (m), e.g. a continuous economic optimum.
        schedule: Pipe schedule.
        direction: ``"up"`` (smallest size with ID >= D; the usual design
            choice), ``"down"`` (largest with ID <= D) or ``"nearest"``.

    Returns:
        ``(NPS, inside diameter m)``.

    Raises:
        ValueError: ``D`` is outside the table in the requested direction.

    Example:
        >>> n, d = snap_to_standard_size(0.095)
        >>> n, round(d * 1e3, 2)
        (4.0, 102.26)
    """
    D = float(D)
    sizes = standard_sizes(schedule)
    if direction == "up":
        for n, d in sizes:
            if d >= D:
                return n, d
        raise ValueError(f"D = {D} m is larger than the largest schedule {schedule} size")
    if direction == "down":
        below = [(n, d) for n, d in sizes if d <= D]
        if not below:
            raise ValueError(f"D = {D} m is smaller than the smallest schedule {schedule} size")
        return below[-1]
    if direction == "nearest":
        return min(sizes, key=lambda t: abs(t[1] - D))
    raise ValueError(f"direction must be 'up', 'down' or 'nearest', got {direction!r}")


# =============================================================================
# Non-Newtonian pipe flow (power-law and Bingham plastic)
# =============================================================================
#
# Convention: the public functions below return the **Darcy** friction factor,
# like :func:`friction_factor`, so ``darcy_pressure_drop(f, L, D, rho, v)``
# applies unchanged. The literature correlations are all written for the
# **Fanning** factor ``f_F = f_D / 4``; the docstrings quote them in that form.

_NN_NEWTON_ITERATIONS = 40


def metzner_reed_reynolds(
    rho: Array | float,
    v: Array | float,
    D: Array | float,
    K: Array | float,
    n: Array | float,
) -> Array:
    """Metzner-Reed generalized Reynolds number of a power-law fluid.

    ``Re_MR = rho v^(2-n) D^n / (K 8^(n-1) ((3n+1)/(4n))^n)`` for a fluid
    with ``tau = K gamma_dot^n``. It is defined so that the laminar Fanning
    friction factor is ``16/Re_MR`` (Darcy ``64/Re_MR``) for every ``n``, and
    it reduces to ``rho v D / mu`` at ``n = 1`` (``K = mu``).

    Source: Metzner, A.B. and Reed, J.C. (1955), AIChE J. 1, 434-440; also
    Perry's Chemical Engineers' Handbook, 9e, Sec. 6 (non-Newtonian flow).

    Args:
        rho: Density (kg/m^3).
        v: Mean velocity (m/s).
        D: Inside diameter (m).
        K: Consistency index (Pa s^n).
        n: Flow behaviour index (-); n < 1 shear-thinning, n > 1 shear-thickening.

    Returns:
        ``Re_MR`` (-).

    Example:
        >>> round(float(metzner_reed_reynolds(1000.0, 1.0, 0.05, 1e-3, 1.0)), 6)
        50000.0
    """
    rho, v, D, K, n = (jnp.asarray(x, dtype=jnp.float64) for x in (rho, v, D, K, n))
    return rho * v ** (2.0 - n) * D**n / (K * 8.0 ** (n - 1.0) * ((3.0 * n + 1.0) / (4.0 * n)) ** n)


def _dodge_metzner_fanning(Re: Array, n: Array) -> Array:
    """Dodge-Metzner turbulent Fanning factor, solved implicitly.

    ``1/sqrt(f) = (4/n^0.75) log10(Re f^(1-n/2)) - 0.4/n^1.2``. With
    ``x = 1/sqrt(f)`` the residual ``g(x) = x - A[log10 Re - (2-n) log10 x] + B``
    is increasing and concave, so Newton from any ``x >= 1`` lands left of the
    root and then converges monotonically. The iteration runs on
    stop-gradient inputs and one implicit-function-theorem step restores the
    exact derivatives in ``Re`` and ``n`` (as in the Colebrook solver).
    """
    sg = jax.lax.stop_gradient
    Re_s, n_s = sg(Re), sg(n)

    def g(x, Re_, n_):
        A = 4.0 / n_**0.75
        return x - A * (jnp.log10(Re_) - (2.0 - n_) * jnp.log10(x)) + 0.4 / n_**1.2

    def gx(x, Re_, n_):
        return 1.0 + (4.0 / n_**0.75) * (2.0 - n_) / (x * _LN10)

    x = jnp.full_like(Re_s, 10.0)
    for _ in range(_NN_NEWTON_ITERATIONS):
        x = jnp.maximum(x - g(x, Re_s, n_s) / gx(x, Re_s, n_s), 1.0)
    x = sg(x)
    x = x - g(x, Re, n) / sg(gx(x, Re_s, n_s))
    return 1.0 / x**2


def power_law_friction_factor(Re_MR: Array | float, n: Array | float) -> Array:
    """**Darcy** friction factor of a power-law fluid in a smooth pipe.

    Laminar (exact for a power-law fluid): Fanning ``f_F = 16/Re_MR``, i.e.
    Darcy ``f = 64/Re_MR``. Turbulent: Dodge and Metzner (1959), Fanning form

    ``1/sqrt(f_F) = (4.0/n^0.75) log10(Re_MR f_F^(1-n/2)) - 0.4/n^1.2``,

    solved by Newton iteration with implicit-function-theorem gradients and
    returned as Darcy ``f = 4 f_F``. At ``n = 1`` it is Prandtl's smooth-pipe
    law ``1/sqrt(f_F) = 4 log10(Re sqrt(f_F)) - 0.4``, which is within about
    1 % of Colebrook at zero roughness for Re 1e4-1e7.

    The laminar and turbulent branches are joined with the same C2 smootherstep
    in ``ln Re_MR`` between 2100 and 4000 that :func:`friction_factor` uses
    (exactly ``64/Re_MR`` below 2100). This is a convention, not a transition
    model: the critical ``Re_MR`` of a shear-thinning fluid is somewhat higher
    than 2100 (Ryan and Johnson 1959), and that shift is not modelled.

    Validity of Dodge-Metzner: smooth pipes; their data covered
    ``0.36 <= n <= 1.0`` and ``Re_MR`` up to about 1e5. Roughness is not
    included.

    Sources: Dodge, D.W. and Metzner, A.B. (1959), AIChE J. 5, 189-204;
    Metzner and Reed (1955), AIChE J. 1, 434; Perry's 9e Sec. 6.

    Args:
        Re_MR: Metzner-Reed Reynolds number (:func:`metzner_reed_reynolds`), positive.
        n: Flow behaviour index (-), positive.

    Returns:
        Darcy friction factor (-). Divide by 4 for Fanning.

    Example:
        >>> float(power_law_friction_factor(1000.0, 0.5)) == 64.0 / 1000.0
        True
    """
    Re = jnp.maximum(jnp.asarray(Re_MR, dtype=jnp.float64), 1e-12)
    n = jnp.asarray(n, dtype=jnp.float64)
    Re, n = jnp.broadcast_arrays(Re, n)
    f_t = 4.0 * _dodge_metzner_fanning(jnp.maximum(Re, RE_LAMINAR), n)
    w = _laminar_weight(Re)
    return (1.0 - w) * 64.0 / Re + w * f_t


def power_law_pressure_drop(
    Q: Array | float,
    L: Array | float,
    D: Array | float,
    rho: Array | float,
    K: Array | float,
    n: Array | float,
) -> Array:
    """Frictional pressure drop (Pa) of a power-law fluid in a smooth round pipe.

    ``dP = f (L/D) rho v^2 / 2`` with ``f`` from :func:`power_law_friction_factor`
    at ``Re_MR`` of the flow. In laminar flow this is exactly the
    Rabinowitsch-Mooney result
    ``dP = (4 K L / D) (8 v / D)^n ((3n+1)/(4n))^n``.

    Args:
        Q: Volumetric flow (m^3/s), positive.
        L: Length (m).
        D: Inside diameter (m).
        rho: Density (kg/m^3).
        K: Consistency index (Pa s^n).
        n: Flow behaviour index (-).

    Returns:
        Pressure drop (Pa).

    Example:
        >>> dP = power_law_pressure_drop(1e-4, 10.0, 0.05, 1000.0, 1.0, 0.5)   # laminar
        >>> v = 1e-4 / (3.141592653589793 / 4 * 0.05**2)
        >>> exact = 4 * 1.0 * 10.0 / 0.05 * (8 * v / 0.05) ** 0.5 * (2.5 / 2.0) ** 0.5
        >>> abs(float(dP) / exact - 1.0) < 1e-12
        True
    """
    D = jnp.asarray(D, dtype=jnp.float64)
    v = jnp.asarray(Q, dtype=jnp.float64) / (jnp.pi / 4.0 * D**2)
    f = power_law_friction_factor(metzner_reed_reynolds(rho, v, D, K, n), n)
    return darcy_pressure_drop(f, L, D, rho, v)


def hedstrom_number(
    rho: Array | float,
    tau_y: Array | float,
    D: Array | float,
    mu_p: Array | float,
) -> Array:
    """Hedstrom number of a Bingham plastic, ``He = rho tau_y D^2 / mu_p^2``.

    Args:
        rho: Density (kg/m^3).
        tau_y: Yield stress (Pa).
        D: Inside diameter (m).
        mu_p: Plastic (Bingham) viscosity (Pa s).

    Returns:
        ``He`` (-). Zero for a Newtonian fluid.

    Example:
        >>> round(float(hedstrom_number(1000.0, 10.0, 0.1, 0.01)), 6)
        1000000.0
    """
    return jnp.asarray(rho) * jnp.asarray(tau_y) * jnp.asarray(D) ** 2 / jnp.asarray(mu_p) ** 2


def _buckingham_reiner_fanning(Re: Array, He: Array) -> Array:
    """Laminar Fanning factor of a Bingham plastic (Buckingham-Reiner).

    Solves ``r(f) = f - (16/Re)(1 + He/(6 Re)) + (16/3) He^4 / (f^3 Re^8) = 0``
    (the Buckingham-Reiner equation multiplied out). ``r`` is convex, and
    increasing at the physical root (``dr/df = 1 - xi^4`` with
    ``xi = tau_y/tau_w < 1``); Newton started from the upper bound
    ``(16/Re)(1 + He/(6Re))`` converges monotonically from the right.
    Derivatives come from one implicit-function-theorem step.
    """
    sg = jax.lax.stop_gradient
    Re_s, He_s = sg(Re), sg(He)

    def r(f, Re_, He_):
        return f - 16.0 / Re_ * (1.0 + He_ / (6.0 * Re_)) + (16.0 / 3.0) * He_**4 / (f**3 * Re_**8)

    def rf(f, Re_, He_):
        return 1.0 - 16.0 * He_**4 / (f**4 * Re_**8)

    f = 16.0 / Re_s * (1.0 + He_s / (6.0 * Re_s))
    for _ in range(_NN_NEWTON_ITERATIONS):
        f = f - r(f, Re_s, He_s) / rf(f, Re_s, He_s)
    f = sg(f)
    return f - r(f, Re, He) / sg(rf(f, Re_s, He_s))


BINGHAM_METHODS = ("darby1992", "laminar")


def bingham_friction_factor(
    Re_B: Array | float,
    He: Array | float,
    method: str = "darby1992",
) -> Array:
    """**Darcy** friction factor of a Bingham plastic in a smooth round pipe.

    ``Re_B = rho v D / mu_p`` (plastic-viscosity Reynolds number) and ``He``
    from :func:`hedstrom_number`. In the Fanning form of the sources:

    * Laminar, Buckingham-Reiner (exact for a Bingham plastic):
      ``f_L = (16/Re_B) [1 + He/(6 Re_B) - He^4/(3 f_L^3 Re_B^7)]``, solved by
      Newton iteration with implicit-function-theorem gradients. At
      ``He = 0`` it is exactly ``16/Re`` (Darcy ``64/Re``).
    * Turbulent, Darby, Mun and Boger (1992):
      ``f_T = 10^a Re_B^-0.193``, ``a = -1.47 [1 + 0.146 exp(-2.9e-5 He)]``.
    * All regimes (``method="darby1992"``, default), Darby et al. (1992):
      ``f = (f_L^m + f_T^m)^(1/m)``, ``m = 1.7 + 40000/Re_B``, evaluated in
      log space so a large ``m`` neither overflows nor underflows.

    ``method="laminar"`` returns the Buckingham-Reiner factor alone.

    The turbulent fit is for slurries and pastes and does **not** reduce to
    the Newtonian smooth-pipe curve at ``He = 0`` (it is about half of it at
    ``Re = 1e5``); only the laminar branch has an exact Newtonian limit. The
    original Darby and Melson (1981) fit used ``-1.378`` in place of
    ``-1.47``; the 1992 revision is used here.

    Sources: Buckingham, E. (1921), Proc. ASTM 21, 1154; Darby, R. and
    Melson, J. (1981), Chem. Eng. 88(26), 59-61; Darby, R., Mun, R. and
    Boger, D.V. (1992), "Predict friction loss in slurry pipes", Chem. Eng.
    99(9), 116-119; Darby, *Chemical Engineering Fluid Mechanics*, 2e (2001).
    The 1992 constants were checked against secondary reprints (e.g.
    arXiv:2501.08999, which quotes the Darcy form ``4 x 10^a Re^-0.193``),
    not against the original article.

    Args:
        Re_B: Bingham Reynolds number ``rho v D / mu_p`` (-), positive.
        He: Hedstrom number (-), non-negative.
        method: ``"darby1992"`` or ``"laminar"``.

    Returns:
        Darcy friction factor (-). Divide by 4 for Fanning.

    Example:
        >>> float(bingham_friction_factor(1000.0, 0.0, "laminar")) == 64.0 / 1000.0
        True
    """
    if method not in BINGHAM_METHODS:
        raise ValueError(f"Unknown Bingham method {method!r}; use one of {BINGHAM_METHODS}")
    Re = jnp.maximum(jnp.asarray(Re_B, dtype=jnp.float64), 1e-12)
    He = jnp.asarray(He, dtype=jnp.float64)
    Re, He = jnp.broadcast_arrays(Re, He)
    f_L = _buckingham_reiner_fanning(Re, He)
    if method == "laminar":
        return 4.0 * f_L
    a = -1.47 * (1.0 + 0.146 * jnp.exp(-2.9e-5 * He))
    log_fT = a * _LN10 - 0.193 * jnp.log(Re)
    m = 1.7 + 40000.0 / Re
    log_f = jnp.logaddexp(m * jnp.log(f_L), m * log_fT) / m
    return 4.0 * jnp.exp(log_f)


def bingham_pressure_drop(
    Q: Array | float,
    L: Array | float,
    D: Array | float,
    rho: Array | float,
    tau_y: Array | float,
    mu_p: Array | float,
    method: str = "darby1992",
) -> Array:
    """Frictional pressure drop (Pa) of a Bingham plastic in a smooth round pipe.

    ``dP = f (L/D) rho v^2 / 2`` with ``f`` from :func:`bingham_friction_factor`.

    Args:
        Q: Volumetric flow (m^3/s), positive.
        L: Length (m).
        D: Inside diameter (m).
        rho: Density (kg/m^3).
        tau_y: Yield stress (Pa).
        mu_p: Plastic viscosity (Pa s).
        method: ``"darby1992"`` (all regimes) or ``"laminar"``.

    Returns:
        Pressure drop (Pa).

    Example:
        >>> dP0 = bingham_pressure_drop(1e-4, 10.0, 0.05, 1000.0, 0.0, 0.1)
        >>> dP1 = bingham_pressure_drop(1e-4, 10.0, 0.05, 1000.0, 5.0, 0.1)
        >>> bool(dP1 > dP0)          # a yield stress adds to the laminar drop
        True
    """
    D = jnp.asarray(D, dtype=jnp.float64)
    v = jnp.asarray(Q, dtype=jnp.float64) / (jnp.pi / 4.0 * D**2)
    Re = reynolds_number(rho, v, D, mu_p)
    f = bingham_friction_factor(Re, hedstrom_number(rho, tau_y, D, mu_p), method)
    return darcy_pressure_drop(f, L, D, rho, v)


# =============================================================================
# Rotameter (variable-area meter)
# =============================================================================


def rotameter_float_dp(
    V_float: Array | float,
    rho_float: Array | float,
    rho: Array | float,
    A_float: Array | float,
) -> Array:
    """Pressure difference across a rotameter float (Pa), from its force balance.

    A float at rest carries its weight less its buoyancy on the pressure
    difference over its largest cross-section:
    ``dP A_f = V_f (rho_f - rho) g``, so ``dP = V_f (rho_f - rho) g / A_f``,
    independent of the flow (McCabe, Smith and Harriott, *Unit Operations of
    Chemical Engineering*, 7e, Ch. 8, area meters; Perry's 9e Sec. 10).

    Args:
        V_float: Float volume (m^3).
        rho_float: Float density (kg/m^3).
        rho: Fluid density (kg/m^3).
        A_float: Float cross-section at its largest diameter (m^2).

    Returns:
        Pressure difference (Pa).

    Example:
        >>> round(float(rotameter_float_dp(4e-6, 8000.0, 1000.0, 1e-4)), 4)
        2745.862
    """
    return (jnp.asarray(V_float) * (jnp.asarray(rho_float) - jnp.asarray(rho)) * GRAVITY
            / jnp.asarray(A_float))


def _rotameter_geometry(D_float, taper, D_tube_0):
    D_f = jnp.asarray(D_float, dtype=jnp.float64)
    D_0 = D_f if D_tube_0 is None else jnp.asarray(D_tube_0, dtype=jnp.float64)
    return D_f, D_0, jnp.asarray(taper, dtype=jnp.float64), jnp.pi / 4.0 * D_f**2


def rotameter_flow(
    h: Array | float,
    D_float: Array | float,
    V_float: Array | float,
    rho_float: Array | float,
    rho: Array | float,
    C_R: Array | float,
    taper: Array | float,
    D_tube_0: Array | float | None = None,
) -> Array:
    """Volumetric flow (m^3/s) through a rotameter with the float at height ``h``.

    The tube is a linear taper, inside diameter ``D_t(h) = D_tube_0 + taper h``
    (``D_tube_0`` defaults to the float diameter: the float seats at
    ``h = 0``). The annulus around the float, ``A_a = (pi/4)(D_t^2 - D_f^2)``,
    passes the flow like an orifice driven by the force-balance pressure
    difference :func:`rotameter_float_dp`:

    ``Q = C_R A_a / sqrt(1 - (A_a/A_t)^2) * sqrt(2 g V_f (rho_f - rho) / (rho A_f))``

    with ``A_t = (pi/4) D_t^2`` (velocity-of-approach factor) and
    ``A_f = (pi/4) D_f^2``. Source: McCabe, Smith and Harriott, 7e, Ch. 8
    (area meters); the same force balance is in Perry's 9e Sec. 10. ``C_R``
    depends on the float shape and the annulus Reynolds number; take it from
    the meter's calibration (there is deliberately no default).

    The mass flow ``rho Q`` scales with ``sqrt(rho (rho_f - rho))``, the
    classic density correction of a rotameter reading.

    Args:
        h: Float height above the seat (m), non-negative.
        D_float: Float diameter at its widest (m).
        V_float: Float volume (m^3).
        rho_float: Float density (kg/m^3), greater than ``rho``.
        rho: Fluid density (kg/m^3).
        C_R: Rotameter discharge coefficient (-).
        taper: Tube taper ``dD_t/dh`` (m of diameter per m of height).
        D_tube_0: Tube inside diameter at ``h = 0`` (m); default ``D_float``.

    Returns:
        Volumetric flow (m^3/s).

    Example:
        >>> Q = rotameter_flow(0.1, 0.02, 4e-6, 8000.0, 1000.0, 0.7, 0.05)
        >>> round(float(rotameter_height(Q, 0.02, 4e-6, 8000.0, 1000.0, 0.7, 0.05)), 9)
        0.1
    """
    D_f, D_0, taper, A_f = _rotameter_geometry(D_float, taper, D_tube_0)
    D_t = D_0 + taper * jnp.asarray(h, dtype=jnp.float64)
    A_t = jnp.pi / 4.0 * D_t**2
    A_a = jnp.maximum(A_t - A_f, 0.0)
    dP = rotameter_float_dp(V_float, rho_float, rho, A_f)
    return (C_R * A_a / jnp.sqrt(1.0 - (A_a / A_t) ** 2)
            * safe_sqrt(2.0 * dP / jnp.asarray(rho)))


def rotameter_height(
    Q: Array | float,
    D_float: Array | float,
    V_float: Array | float,
    rho_float: Array | float,
    rho: Array | float,
    C_R: Array | float,
    taper: Array | float,
    D_tube_0: Array | float | None = None,
) -> Array:
    """Float height (m) for a volumetric flow ``Q`` (m^3/s), the inverse of
    :func:`rotameter_flow`.

    Solves ``A_a / sqrt(1 - (A_a/(A_a + A_f))^2) = Q / (C_R sqrt(2 dP/rho))``
    for the annulus area ``A_a`` by Newton's method in ``ln A_a`` (the left
    side is increasing in ``A_a``), with one implicit-function-theorem step
    for exact derivatives, then ``h = (D_t - D_tube_0)/taper`` with
    ``D_t = sqrt(4 (A_a + A_f)/pi)``. Arguments as :func:`rotameter_flow`,
    with ``Q`` (m^3/s, positive) in place of ``h``.

    Returns:
        Float height (m).
    """
    D_f, D_0, taper, A_f = _rotameter_geometry(D_float, taper, D_tube_0)
    dP = rotameter_float_dp(V_float, rho_float, rho, A_f)
    k = jnp.asarray(Q, dtype=jnp.float64) / (C_R * safe_sqrt(2.0 * dP / jnp.asarray(rho)))

    def g(y, k_, A_f_):  # residual in y = ln A_a
        A = jnp.exp(y)
        return y - 0.5 * jnp.log(1.0 - (A / (A + A_f_)) ** 2) - jnp.log(k_)

    def gy(y, A_f_):
        A = jnp.exp(y)
        s = A / (A + A_f_)
        return 1.0 + s**2 / (1.0 - s**2) * A_f_ / (A + A_f_)

    sg = jax.lax.stop_gradient
    k_s, A_f_s = sg(k), sg(A_f)
    y = jnp.log(k_s)  # A_a <= k: start at the upper bound
    for _ in range(_NN_NEWTON_ITERATIONS):
        y = y - g(y, k_s, A_f_s) / gy(y, A_f_s)
    y = sg(y)
    y = y - g(y, k, A_f) / sg(gy(y, A_f_s))
    D_t = jnp.sqrt(4.0 * (jnp.exp(y) + A_f) / jnp.pi)
    return (D_t - D_0) / taper


# =============================================================================
# Pipe metal mass
# =============================================================================

#: Carbon-steel density (kg/m^3) behind the ASME B36.10M plain-end weights.
STEEL_DENSITY = 7850.0


def pipe_weight_per_length(nps, schedule="40", rho_metal: float = STEEL_DENSITY) -> float:
    """Plain-end metal mass per metre (kg/m) of a nominal pipe size and schedule.

    ``W = rho_metal pi (OD - t) t``. With ``rho_metal = 7850 kg/m^3`` this is
    the ASME B36.10M plain-end weight formula ``W = 0.0246615 (OD - t) t``
    (``OD``, ``t`` in mm, ``W`` in kg/m): 2-in Sch 40 gives 5.44 kg/m.

    Args:
        nps: Nominal size (in), as for :func:`pipe_inside_diameter`.
        schedule: Pipe schedule.
        rho_metal: Metal density (kg/m^3); 7850 for carbon steel.

    Returns:
        Mass per metre (kg/m), a plain float (table data).

    Example:
        >>> round(pipe_weight_per_length(2, "40"), 2)
        5.44
    """
    OD = pipe_outside_diameter(nps)
    t = pipe_wall_thickness(nps, schedule)
    return float(rho_metal * math.pi * (OD - t) * t)


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
    "EXTRA_K_KEYS",
    "fittings_K",
    "ORIFICE_TAPS",
    "orifice_cd",
    "orifice_permanent_loss",
    "VENTURI_LOSS_TABLE",
    "venturi_permanent_loss",
    "NPS_SCHEDULES",
    "pipe_outside_diameter",
    "pipe_wall_thickness",
    "pipe_inside_diameter",
    "pipe_id",
    "standard_sizes",
    "snap_to_standard_size",
    "metzner_reed_reynolds",
    "power_law_friction_factor",
    "power_law_pressure_drop",
    "hedstrom_number",
    "BINGHAM_METHODS",
    "bingham_friction_factor",
    "bingham_pressure_drop",
    "rotameter_float_dp",
    "rotameter_flow",
    "rotameter_height",
    "STEEL_DENSITY",
    "pipe_weight_per_length",
]
