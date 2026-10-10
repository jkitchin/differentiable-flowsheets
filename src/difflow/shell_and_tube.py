"""Shell-and-tube exchanger design: Kern shell-side method, geometry, U iteration.

Pure JAX, SI, differentiable. Given a duty and the stream data it sizes a
1-2N shell-and-tube exchanger: it assumes an overall coefficient ``U``, gets the
area from ``Q = U A F LMTD``, builds a geometry (tube count, shell diameter,
baffle spacing), computes the tube-side film coefficient
(:func:`difflow.heat_transfer.internal_h`, Gnielinski with the friction factor of
:func:`difflow.fluids.friction_factor`), the shell-side film coefficient by the
Kern method, a new ``U`` by :func:`difflow.heat_transfer.overall_U` with
fouling, and repeats to a fixed point. The fixed point is an
``optimistix.fixed_point``, so ``jax.grad`` of the area (or of a cost built
from it) with respect to baffle spacing, tube velocity, flow rates, fouling or
any property works, by the implicit function theorem.

Not a palette unit operation (same policy as :mod:`difflow.heat_transfer`): the
design takes explicit mass flows and properties (``k``, ``mu``, ``Cp``,
``rho``) because core ``SpeciesData`` has no liquid transport properties, and
it returns a *geometry*, not a stream. The rating exchanger
:class:`difflow.units.heat_exchanger.ShellAndTubeHX` takes ``UA``; the design
returns it.

Contents
--------
* Geometry data: :data:`BWG_WALL_THICKNESS`, :data:`TEMA_TUBE_PITCH`,
  :func:`tube_inside_diameter`, :data:`BUNDLE_CONSTANTS`
* Geometry: :func:`tube_count`, :func:`bundle_diameter`
* Kern method: :func:`equivalent_diameter`, :func:`kern_shell_side`,
  :func:`kern_shell_pressure_drop`
* Tube side: :func:`tube_side_pressure_drop`
* Design: :class:`ShellAndTubeDesignParams`, :class:`ShellAndTubeDesign`,
  :func:`difflow.units.heat_exchanger.effectiveness_shell_and_tube`, :func:`round_design`

**Verification status** is stated in ``docs`` and ``tests/test_shell_and_tube.py``.
Short version: the Kern correlations below are *transcribed from memory* and
the published Kern kerosene-crude worked example is **not** reproduced; every
constant marked "unverified" must be checked against the book before a real
design.

Example:
    >>> from difflow.shell_and_tube import kern_shell_side
    >>> r = kern_shell_side(m_dot=5.0, rho=800.0, mu=3e-4, k=0.13, Cp=2200.0,
    ...                     shell_ID=0.5, tube_OD=0.019, pitch=0.0238,
    ...                     baffle_spacing=0.2)
    >>> float(r["h_o"]) > 0
    True
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow import fluids
from difflow import heat_transfer as ht
from difflow.params_mixin import ParamsMixin
from difflow.units.heat_exchanger import (
    effectiveness_shell_and_tube,
    lmtd_correction_factor,
    log_mean_temperature_difference,
)

_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731

LAYOUTS = ("triangular", "square")

#: Kern shell-side Nusselt correlation ``Nu = C Re^n Pr^(1/3) (mu/mu_w)^0.14``,
#: ``C = 0.36``, ``n = 0.55`` for ``2e3 < Re < 1e6`` and 25 percent cut
#: segmental baffles (Kern, *Process Heat Transfer*, 1950, Eq. 7.4; the same
#: line as the ``j_h`` chart of Coulson & Richardson Vol 6, Fig. 12.29).
#: **Transcribed from memory; unverified.**
KERN_NU_COEFFICIENT = 0.36
KERN_NU_EXPONENT = 0.55
#: Kern shell-side friction factor ``f = exp(a - b ln Re)`` with ``a = 0.576``,
#: ``b = 0.19`` (a widely quoted fit to Kern's Fig. 29 for 25 percent cut
#: baffles, ``400 < Re < 1e6``), dimensionless so that
#: ``dP = f (D_s/D_e)(L/l_B) rho u_s^2 / 2``. **From memory; unverified.**
KERN_FRICTION_A = 0.576
KERN_FRICTION_B = 0.19

#: Tube-count-correction (``CTP``) and layout (``CL``) constants of the
#: Kakac / Phadke ideal tube-count form
#: ``Nt = CTP (pi/4) D_s^2 / (CL Pt^2)``: CTP = 0.93, 0.90, 0.85 for one, two,
#: three tube passes; CL = 1 (square) or 0.866 (triangular). **From memory.**
CTP_PASSES = (1.0, 2.0, 3.0)
CTP_VALUES = (0.93, 0.90, 0.85)
LAYOUT_CL = {"square": 1.0, "triangular": 0.866}

#: Coulson & Richardson Vol 6, Table 12.4: ``Db = d_o (Nt/K1)^(1/n1)`` for
#: pitch ``1.25 d_o``; keyed ``layout -> {passes: (K1, n1)}``. **Recalled from
#: memory of Sinnott's table; unverified**, and valid only for ``Pt = 1.25 d_o``.
BUNDLE_CONSTANTS: dict[str, dict[int, tuple[float, float]]] = {
    "triangular": {
        1: (0.319, 2.142),
        2: (0.249, 2.207),
        4: (0.175, 2.285),
        6: (0.0743, 2.499),
        8: (0.0365, 2.675),
    },
    "square": {
        1: (0.215, 2.207),
        2: (0.156, 2.291),
        4: (0.158, 2.263),
        6: (0.0402, 2.617),
        8: (0.0331, 2.643),
    },
}

#: Tube wall thickness (m) by Birmingham wire gauge. **From memory of the
#: standard tube tables (Kern Table 10); unverified.**
BWG_WALL_THICKNESS: dict[int, float] = {
    10: 0.134 * 0.0254,
    12: 0.109 * 0.0254,
    14: 0.083 * 0.0254,
    16: 0.065 * 0.0254,
    18: 0.049 * 0.0254,
    20: 0.035 * 0.0254,
}

#: Common TEMA-style tube OD (in m) -> (layout, pitch in m): 3/4 in on 15/16 in
#: triangular and 1 in on 1-1/4 in triangular. **From memory; unverified.**
TEMA_TUBE_PITCH: dict[tuple[float, str], float] = {
    (0.75 * 0.0254, "triangular"): 0.9375 * 0.0254,
    (0.75 * 0.0254, "square"): 1.0 * 0.0254,
    (1.0 * 0.0254, "triangular"): 1.25 * 0.0254,
    (1.0 * 0.0254, "square"): 1.25 * 0.0254,
}


def _check_layout(layout: str) -> None:
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def tube_inside_diameter(tube_OD: float, bwg: int) -> float:
    """Tube inside diameter (m) from the outside diameter and BWG gauge.

    ``D_i = D_o - 2 t(BWG)`` with :data:`BWG_WALL_THICKNESS` (unverified).

    Args:
        tube_OD: Outside diameter (m).
        bwg: Birmingham wire gauge, one of the keys of :data:`BWG_WALL_THICKNESS`.

    Returns:
        Inside diameter (m).

    Raises:
        KeyError: Unknown gauge.
    """
    try:
        t = BWG_WALL_THICKNESS[bwg]
    except KeyError:
        raise KeyError(f"Unknown BWG {bwg!r}; use one of {sorted(BWG_WALL_THICKNESS)}") from None
    return tube_OD - 2.0 * t


# =============================================================================
# Geometry
# =============================================================================


def _ctp(n_passes: Array) -> Array:
    return jnp.interp(n_passes, jnp.asarray(CTP_PASSES), jnp.asarray(CTP_VALUES))


def tube_count(
    shell_ID: Array | float,
    tube_OD: Array | float,
    pitch: Array | float,
    layout: str = "triangular",
    n_passes: Array | float = 2.0,
    integer: bool = False,
) -> Array:
    """Tube count in a shell of inside diameter ``shell_ID`` (Kakac / Phadke form).

    ``Nt = CTP (pi/4) D_s^2 / (CL Pt^2)`` (the ideal packing of the full shell
    cross-section, scaled by the tube-count correction ``CTP`` for the pass
    partition lanes and the bundle-to-shell clearance). The value is
    **continuous** so it can be differentiated; ``CTP`` is interpolated
    linearly in the number of passes (0.93, 0.90, 0.85 at 1, 2, 3) and held
    constant outside 1 to 3, a crude extrapolation that overstates the tube
    count for four or more passes. ``tube_OD`` does not enter (it only
    constrains the ratio ``pitch/tube_OD`` >= 1.25 in practice); it is accepted
    to keep the signature parallel to :func:`bundle_diameter`.

    Constants are **from memory and unverified**; real shells are counted from
    tube-sheet layouts (tube-count tables), which this approximates within
    roughly 10 percent at best.

    Args:
        shell_ID: Shell inside diameter (m).
        tube_OD: Tube outside diameter (m) (unused, documented above).
        pitch: Tube pitch (m).
        layout: ``"triangular"`` or ``"square"``.
        n_passes: Number of tube passes (continuous allowed).
        integer: True to return ``floor`` (a concrete integer, not differentiable).

    Returns:
        Number of tubes (continuous unless ``integer``).

    Raises:
        ValueError: Unknown layout.
    """
    _check_layout(layout)
    del tube_OD
    Ds, Pt = _f64(shell_ID), _f64(pitch)
    Nt = _ctp(_f64(n_passes)) * (jnp.pi / 4.0) * Ds**2 / (LAYOUT_CL[layout] * Pt**2)
    return jnp.floor(Nt) if integer else Nt


def _bundle_constants(layout: str, n_passes: Array) -> tuple[Array, Array]:
    """``(K1, n1)`` interpolated in ``ln`` of the pass count between table rows."""
    table = BUNDLE_CONSTANTS[layout]
    passes = sorted(table)
    x = jnp.log(jnp.asarray(passes, dtype=jnp.float64))
    K1 = jnp.interp(jnp.log(jnp.maximum(n_passes, 1.0)), x, jnp.asarray([table[p][0] for p in passes]))
    n1 = jnp.interp(jnp.log(jnp.maximum(n_passes, 1.0)), x, jnp.asarray([table[p][1] for p in passes]))
    return K1, n1


def bundle_diameter(
    n_tubes: Array | float,
    tube_OD: Array | float,
    n_passes: Array | float = 2.0,
    layout: str = "triangular",
) -> Array:
    """Bundle diameter ``D_b = d_o (N_t / K_1)^(1/n_1)`` (Coulson & Richardson Vol 6, Eq. 12.3b).

    ``K1`` and ``n1`` come from :data:`BUNDLE_CONSTANTS` (pitch ``1.25 d_o``,
    **unverified, from memory**). For a non-integer number of passes the
    constants are interpolated linearly in ``ln(passes)`` between the tabulated
    1, 2, 4, 6, 8 (a smoothing choice for gradient-based design, not a
    published correlation); outside 1 to 8 they are held at the end values.
    Pitch ratios other than 1.25 are *not* covered by the table.

    Args:
        n_tubes: Number of tubes (continuous allowed).
        tube_OD: Tube outside diameter (m).
        n_passes: Number of tube passes.
        layout: ``"triangular"`` or ``"square"``.

    Returns:
        Bundle diameter (m).

    Raises:
        ValueError: Unknown layout.
    """
    _check_layout(layout)
    K1, n1 = _bundle_constants(layout, _f64(n_passes))
    return _f64(tube_OD) * (_f64(n_tubes) / K1) ** (1.0 / n1)


# =============================================================================
# Kern method
# =============================================================================


def equivalent_diameter(pitch: Array | float, tube_OD: Array | float, layout: str = "triangular") -> Array:
    """Shell-side equivalent (hydraulic) diameter ``D_e`` (m), Kern.

    Square pitch: ``D_e = 1.27 (Pt^2 - 0.785 d_o^2) / d_o``; triangular:
    ``D_e = 1.10 (Pt^2 - 0.917 d_o^2) / d_o`` (Coulson & Richardson Vol 6,
    Eqs. 12.21-12.22 / Kern Fig. 28). The constants are the four-times-flow-area
    over wetted-perimeter of the pitch cell: ``4 (Pt^2 - pi d_o^2/4)/(pi d_o)``
    for square pitch (1.273, 0.785) and
    ``4 (Pt^2 sqrt(3)/4 - pi d_o^2/8)/(pi d_o/2)`` for triangular
    (1.103, 0.9069 -- the textbook rounds the latter to 0.917). Both forms are
    in the tests.

    Args:
        pitch: Tube pitch (m).
        tube_OD: Tube outside diameter (m).
        layout: ``"triangular"`` or ``"square"``.

    Returns:
        Equivalent diameter (m).
    """
    _check_layout(layout)
    Pt, do = _f64(pitch), _f64(tube_OD)
    if layout == "square":
        return 1.27 / do * (Pt**2 - 0.785 * do**2)
    return 1.10 / do * (Pt**2 - 0.917 * do**2)


def _cross_flow_area(shell_ID, tube_OD, pitch, baffle_spacing):
    """Kern bundle cross-flow area ``A_s = (Pt - d_o)/Pt D_s l_B`` (m^2)."""
    return (pitch - tube_OD) / pitch * shell_ID * baffle_spacing


def kern_shell_side(
    m_dot: Array | float,
    rho: Array | float,
    mu: Array | float,
    k: Array | float,
    Cp: Array | float,
    shell_ID: Array | float,
    tube_OD: Array | float,
    pitch: Array | float,
    baffle_spacing: Array | float,
    layout: str = "triangular",
    mu_wall: Array | float | None = None,
) -> dict[str, Array]:
    """Kern shell-side film coefficient for segmental baffles.

    ``A_s = (Pt - d_o) D_s l_B / Pt``, ``G_s = m/A_s``, ``Re = G_s D_e / mu``,
    ``Nu = h_o D_e / k = 0.36 Re^0.55 Pr^(1/3) (mu/mu_w)^0.14`` (see
    :data:`KERN_NU_COEFFICIENT`; **unverified**; this is the ``j_h`` line of the
    25 percent baffle-cut chart in power-law form, valid ``2e3 < Re < 1e6``
    and used outside as an extrapolation). Kern ignores leakage and bypass
    streams (Bell-Delaware does not), so ``h_o`` is typically conservative
    for clean, well-sealed bundles.

    Args:
        m_dot: Shell-side mass flow (kg/s).
        rho: Density (kg/m^3).
        mu: Bulk viscosity (Pa s).
        k: Conductivity (W/m/K).
        Cp: Specific heat (J/kg/K).
        shell_ID: Shell inside diameter (m).
        tube_OD: Tube outside diameter (m).
        pitch: Tube pitch (m).
        baffle_spacing: Baffle spacing ``l_B`` (m).
        layout: ``"triangular"`` or ``"square"``.
        mu_wall: Viscosity at the wall temperature (Pa s), or None for 1.

    Returns:
        Dict with ``De`` (m), ``As`` (m^2), ``Gs`` (kg/m^2/s), ``us`` (m/s),
        ``Re``, ``Pr``, ``Nu`` and ``h_o`` (W/m^2/K).
    """
    De = equivalent_diameter(pitch, tube_OD, layout)
    As = _cross_flow_area(_f64(shell_ID), _f64(tube_OD), _f64(pitch), _f64(baffle_spacing))
    Gs = _f64(m_dot) / As
    Re = Gs * De / _f64(mu)
    Pr = ht.prandtl(Cp, mu, k)
    ratio = 1.0 if mu_wall is None else _f64(mu) / _f64(mu_wall)
    Nu = KERN_NU_COEFFICIENT * Re**KERN_NU_EXPONENT * Pr ** (1.0 / 3.0) * ratio**0.14
    return {
        "De": De, "As": As, "Gs": Gs, "us": Gs / _f64(rho), "Re": Re, "Pr": Pr,
        "Nu": Nu, "h_o": Nu * _f64(k) / De,
    }


def kern_friction_factor(Re: Array | float) -> Array:
    """Kern shell-side friction factor ``f = exp(0.576 - 0.19 ln Re)`` (dimensionless, **unverified**)."""
    return jnp.exp(KERN_FRICTION_A - KERN_FRICTION_B * jnp.log(_f64(Re)))


def kern_shell_pressure_drop(
    m_dot: Array | float,
    rho: Array | float,
    mu: Array | float,
    shell_ID: Array | float,
    tube_OD: Array | float,
    pitch: Array | float,
    baffle_spacing: Array | float,
    tube_length: Array | float,
    layout: str = "triangular",
    mu_wall: Array | float | None = None,
) -> dict[str, Array]:
    """Kern shell-side pressure drop.

    ``dP_s = f (D_s/D_e) (N_b + 1) (rho u_s^2 / 2) (mu/mu_w)^(-0.14)`` with
    ``N_b + 1 = L / l_B`` baffle crossings (continuous: the integer rounding is
    left to :func:`round_design`) and ``f`` from :func:`kern_friction_factor`.
    The same as Coulson & Richardson Vol 6 Eq. 12.26,
    ``8 j_f (D_s/D_e)(L/l_B)(rho u_s^2/2)(mu/mu_w)^-0.14``, with ``8 j_f = f``.
    Nozzle and baffle-window losses are not included.

    Args:
        m_dot: Shell-side mass flow (kg/s).
        rho: Density (kg/m^3).
        mu: Viscosity (Pa s).
        shell_ID: Shell inside diameter (m).
        tube_OD: Tube outside diameter (m).
        pitch: Tube pitch (m).
        baffle_spacing: Baffle spacing (m).
        tube_length: Tube length ``L`` (m).
        layout: ``"triangular"`` or ``"square"``.
        mu_wall: Wall viscosity (Pa s), or None.

    Returns:
        Dict with ``dP`` (Pa), ``f``, ``n_crossings`` (``L / l_B``), plus ``Re``, ``us``, ``De``.
    """
    side = kern_shell_side(m_dot, rho, mu, 1.0, 1.0, shell_ID, tube_OD, pitch, baffle_spacing, layout, mu_wall)
    f = kern_friction_factor(side["Re"])
    n_cross = _f64(tube_length) / _f64(baffle_spacing)
    ratio = 1.0 if mu_wall is None else _f64(mu) / _f64(mu_wall)
    dP = f * (_f64(shell_ID) / side["De"]) * n_cross * 0.5 * _f64(rho) * side["us"] ** 2 * ratio ** (-0.14)
    return {"dP": dP, "f": f, "n_crossings": n_cross, "Re": side["Re"], "us": side["us"], "De": side["De"]}


# =============================================================================
# Tube side
# =============================================================================


def tube_side_pressure_drop(
    m_dot: Array | float,
    rho: Array | float,
    mu: Array | float,
    n_tubes: Array | float,
    n_passes: Array | float,
    tube_ID: Array | float,
    tube_length: Array | float,
    roughness: Array | float = 4.6e-5,
    return_loss_heads: float = 4.0,
) -> dict[str, Array]:
    """Tube-side pressure drop: friction plus return losses.

    ``dP_t = N_p (f L/d_i + K_r) rho v^2 / 2`` with ``v = m N_p / (rho N_t pi d_i^2 / 4)``
    (``N_t / N_p`` tubes in parallel per pass), ``f`` the Darcy factor of
    :func:`difflow.fluids.friction_factor` (Colebrook, laminar blend) and
    ``K_r = return_loss_heads`` (4, the value of the issue / Kern, per pass;
    Coulson & Richardson use 2.5). No viscosity-ratio correction.

    Args:
        m_dot: Tube-side mass flow (kg/s).
        rho: Density (kg/m^3).
        mu: Viscosity (Pa s).
        n_tubes: Total number of tubes (continuous allowed).
        n_passes: Number of tube passes.
        tube_ID: Tube inside diameter (m).
        tube_length: Tube length (m), one pass.
        roughness: Absolute roughness (m), default commercial steel.
        return_loss_heads: Velocity heads lost per pass in the return.

    Returns:
        Dict with ``dP`` (Pa), ``dP_friction``, ``dP_return``, ``v`` (m/s), ``Re``, ``f``.
    """
    d = _f64(tube_ID)
    area = jnp.pi / 4.0 * d**2 * _f64(n_tubes) / _f64(n_passes)
    v = _f64(m_dot) / (_f64(rho) * area)
    Re = fluids.reynolds_number(rho, v, d, mu)
    f = fluids.friction_factor(Re, _f64(roughness) / d)
    dyn = 0.5 * _f64(rho) * v**2
    dP_f = _f64(n_passes) * f * _f64(tube_length) / d * dyn
    dP_r = _f64(n_passes) * return_loss_heads * dyn
    return {"dP": dP_f + dP_r, "dP_friction": dP_f, "dP_return": dP_r, "v": v, "Re": Re, "f": f}


# =============================================================================
# Rating helper and design
# =============================================================================


@dataclass(repr=False)
class ShellAndTubeDesignParams(ParamsMixin):
    """Parameters for :class:`ShellAndTubeDesign`.

    Hot and cold sides carry explicit properties (SI); mass flows come with
    the streams passed to the call.

    Attributes:
        tube_OD: Tube outside diameter (m).
        tube_ID: Tube inside diameter (m).
        rho_hot: Hot-side density (kg/m^3).
        mu_hot: Hot-side viscosity (Pa s).
        k_hot: Hot-side conductivity (W/m/K).
        Cp_hot: Hot-side specific heat (J/kg/K).
        rho_cold: Cold-side density (kg/m^3).
        mu_cold: Cold-side viscosity (Pa s).
        k_cold: Cold-side conductivity (W/m/K).
        Cp_cold: Cold-side specific heat (J/kg/K).
        pitch_ratio: Tube pitch over outside diameter (-). The bundle-diameter constants are for 1.25.
        layout: ``"triangular"`` or ``"square"`` (Python string).
        tube_side: ``"hot"`` or ``"cold"``: which fluid is in the tubes (Python string).
        n_tube_passes: Number of tube passes (continuous allowed).
        tube_length: Tube length (m); used when ``tube_velocity`` is None.
        tube_velocity: Tube-side velocity (m/s). If given, the tube count follows from
            the flow and the tube length from the area; otherwise the tube count
            follows from the area and ``tube_length``.
        baffle_spacing: Baffle spacing (m), or None to use ``baffle_spacing_ratio * D_s``.
        baffle_spacing_ratio: Baffle spacing over shell diameter, used if ``baffle_spacing`` is None.
        shell_clearance: Shell inside diameter minus bundle diameter (m). The
            default 0.05 is an illustrative value for a floating head, not a standard.
        k_wall: Tube wall conductivity (W/m/K).
        R_fi: Tube-side fouling resistance (m^2 K/W, inside area).
        R_fo: Shell-side fouling resistance (m^2 K/W, outside area).
        mu_wall_hot: Hot-side viscosity at the wall (Pa s), or None.
        mu_wall_cold: Cold-side viscosity at the wall (Pa s), or None.
        roughness: Tube roughness (m).
        hot_isothermal: True for a condensing hot side (T_out = T_in, F = 1).
        cold_isothermal: True for a boiling cold side.
        shell_h: ``None`` for Kern, a number (W/m^2/K) or a callable ``shell_h(geom)``
            (``geom`` is a dict with ``tube_OD``, ``n_tubes``, ``tube_length``, ``shell_ID``,
            ``area``) for a phase-change shell side. Shell dP is then zero (not modelled).
        dP_tube_max: Tube-side pressure-drop limit (Pa), for the warning flag.
        dP_shell_max: Shell-side pressure-drop limit (Pa).
        U_guess: Starting overall coefficient (W/m^2/K).
        damping: Damping of the fixed-point map (1 is plain substitution).
        tol: Relative tolerance of the fixed point.
        max_iter: Maximum fixed-point steps.
    """

    tube_OD: float
    tube_ID: float
    rho_hot: float
    mu_hot: float
    k_hot: float
    Cp_hot: float
    rho_cold: float
    mu_cold: float
    k_cold: float
    Cp_cold: float
    pitch_ratio: float = 1.25
    layout: str = "triangular"
    tube_side: str = "cold"
    n_tube_passes: float = 2.0
    tube_length: float | None = 4.88
    tube_velocity: float | None = None
    baffle_spacing: float | None = None
    baffle_spacing_ratio: float = 0.4
    shell_clearance: float = 0.05
    k_wall: float = 45.0
    R_fi: float = 0.0
    R_fo: float = 0.0
    mu_wall_hot: float | None = None
    mu_wall_cold: float | None = None
    roughness: float = 4.6e-5
    hot_isothermal: bool = False
    cold_isothermal: bool = False
    shell_h: float | Callable | None = None
    dP_tube_max: float = float("inf")
    dP_shell_max: float = float("inf")
    U_guess: float = 500.0
    damping: float = 0.6
    tol: float = 1e-10
    max_iter: int = 200


class ShellAndTubeDesign:
    """Size a 1-2N shell-and-tube exchanger by iterating the overall coefficient.

    ``design(hot_in, cold_in, Q=...)`` (or ``T_hot_out=`` / ``T_cold_out=``)
    solves the fixed point ``U = overall_U(h_i(A(U)), h_o(A(U)))`` for the
    area. The iteration is :func:`optimistix.fixed_point`, so the whole
    result is differentiable (implicit function theorem) in every numeric
    argument, which is how baffle spacing and tube velocity enter a gradient
    based cost minimization. Integer quantities (tube count, passes, baffle
    count) are continuous here; :func:`round_design` and :meth:`rate` turn the
    result into a buildable exchanger and re-rate it.

    **The ``U`` fixed point is not unique in general.** With a fixed tube length
    a larger area means more tubes, hence a lower tube velocity, hence a lower
    ``h_i``: positive feedback. Besides the practical solution there can be a
    second, large-area, laminar-tube-side one (in the test case 211 m^2 at
    ``Re_tube`` ~ 1500 against 13 m^2 at ~23000, reached from ``U_guess`` of
    100 or less). A start at the typical ``U`` of the service (or fixing
    ``tube_velocity``, which removes the feedback) finds the practical one;
    ``tube_laminar`` and a warning flag the other.

    Not a palette operation: it consumes mass flows and explicit properties
    and returns geometry, so it is a library like :mod:`difflow.heat_transfer`.
    """

    symbol = "S&T Design"
    equations = [
        r"Q = U A F\,\mathrm{LMTD}_{\mathrm{cc}},\qquad A = \frac{Q}{U F\,\mathrm{LMTD}_{\mathrm{cc}}}",
        r"\frac{1}{U_o} = \frac{1}{h_o} + R_{fo} + \frac{d_o\ln(d_o/d_i)}{2k_w} + R_{fi}\frac{d_o}{d_i} + \frac{d_o}{d_i h_i}",
        r"D_e = \frac{1.10}{d_o}\left(P_t^2 - 0.917\,d_o^2\right)\ (\triangle),\quad A_s = \frac{(P_t - d_o)D_s l_B}{P_t},\quad G_s = \frac{\dot m_s}{A_s}",
        r"\mathrm{Nu}_s = \frac{h_o D_e}{k} = 0.36\,\mathrm{Re}_s^{0.55}\mathrm{Pr}^{1/3}\left(\frac{\mu}{\mu_w}\right)^{0.14},\quad \mathrm{Re}_s = \frac{G_s D_e}{\mu}",
        r"\Delta P_s = f_s\,\frac{D_s}{D_e}\,\frac{L}{l_B}\,\frac{\rho u_s^2}{2}\left(\frac{\mu}{\mu_w}\right)^{-0.14},\quad f_s = e^{0.576 - 0.19\ln\mathrm{Re}_s}",
        r"\Delta P_t = N_p\left(f\frac{L}{d_i} + 4\right)\frac{\rho v^2}{2},\qquad D_b = d_o\left(\frac{N_t}{K_1}\right)^{1/n_1}",
    ]
    assumptions = [
        "1-2N exchanger (one shell pass, even tube passes): F from the Bowman/Underwood formula; countercurrent LMTD basis.",
        "Single-phase Newtonian liquids (or gases) with constant properties on each side, evaluated by the caller; a condensing or boiling side is the isothermal mode.",
        "Kern shell side: segmental baffles (25 percent cut), no leakage or bypass streams, Re 2e3 to 1e6 (extrapolated outside).",
        "Tube side: Gnielinski / laminar blend of difflow.heat_transfer.internal_h, Darcy friction from difflow.fluids.friction_factor, 4 velocity heads per pass for the return.",
        "Tube count, passes and baffle count are continuous in the design; the geometry must be rounded and re-rated (round_design, rate) before it is built.",
        "Bundle diameter from the Coulson & Richardson constants for pitch 1.25 d_o; shell diameter = bundle + a fixed clearance.",
    ]
    references = [
        "Kern, D.Q. (1950). Process Heat Transfer. McGraw-Hill.",
        "Coulson, J.M. & Richardson, J.F. Chemical Engineering Vol. 6 (Sinnott), Ch. 12.",
        "Kakac, S., Liu, H. & Pramuanjaroenkij, A. Heat Exchangers: Selection, Rating, and Thermal Design.",
        "Bowman, R.A., Mueller, A.C. & Nagle, W.M. (1940). Trans. ASME 62, 283-294.",
        "Gnielinski, V. (1976). Int. Chem. Eng. 16, 359-368.",
    ]
    parameter_symbols = {"tube_OD": "d_o", "tube_ID": "d_i", "pitch_ratio": "P_t/d_o", "n_tube_passes": "N_p",
                         "tube_length": "L", "baffle_spacing": "l_B", "k_wall": "k_w"}
    parameter_units = {
        "tube_OD": "m", "tube_ID": "m", "rho_hot": "kg/m^3", "mu_hot": "Pa s", "k_hot": "W/m/K",
        "Cp_hot": "J/kg/K", "rho_cold": "kg/m^3", "mu_cold": "Pa s", "k_cold": "W/m/K",
        "Cp_cold": "J/kg/K", "pitch_ratio": "-", "layout": "-", "tube_side": "-", "n_tube_passes": "-",
        "tube_length": "m", "tube_velocity": "m/s", "baffle_spacing": "m", "baffle_spacing_ratio": "-",
        "shell_clearance": "m", "k_wall": "W/m/K", "R_fi": "m^2 K/W", "R_fo": "m^2 K/W",
        "dP_tube_max": "Pa", "dP_shell_max": "Pa", "U_guess": "W/m^2/K",
    }
    numerical_method = "optimistix.fixed_point (damped substitution) on U with implicit-function-theorem gradients."

    def __init__(self, params: ShellAndTubeDesignParams):
        if params.tube_side not in ("hot", "cold"):
            raise ValueError(f"tube_side must be 'hot' or 'cold', got {params.tube_side!r}")
        _check_layout(params.layout)
        self.params = params

    # ------------------------------------------------------------------
    def _temperatures(self, hot_in, cold_in, Q, T_hot_out, T_cold_out):
        p = self.params
        given = [x is not None for x in (Q, T_hot_out, T_cold_out)]
        if sum(given) != 1:
            raise ValueError("give exactly one of Q, T_hot_out, T_cold_out")
        Th, Tc = _f64(hot_in["T"]), _f64(cold_in["T"])
        Ch = _f64(hot_in["m_dot"]) * _f64(p.Cp_hot)
        Cc = _f64(cold_in["m_dot"]) * _f64(p.Cp_cold)
        if Q is not None:
            Q = _f64(Q)
        elif T_hot_out is not None:
            Q = Ch * (Th - _f64(T_hot_out))
        else:
            Q = Cc * (_f64(T_cold_out) - Tc)
        Tho = Th if p.hot_isothermal else Th - Q / Ch
        Tco = Tc if p.cold_isothermal else Tc + Q / Cc
        return Q, Tho, Tco

    def _args(self, hot_in, cold_in, Q, T_hot_out, T_cold_out, over):
        p = self.params
        Q, Tho, Tco = self._temperatures(hot_in, cold_in, Q, T_hot_out, T_cold_out)
        Th, Tc = _f64(hot_in["T"]), _f64(cold_in["T"])
        dT1, dT2 = Th - Tco, Tho - Tc
        lmtd = log_mean_temperature_difference(dT1, dT2)
        if p.hot_isothermal or p.cold_isothermal:
            F = jnp.asarray(1.0)
            R = P = jnp.asarray(jnp.nan)
        else:
            R = (Th - Tho) / jnp.maximum(Tco - Tc, 1e-12)
            P = (Tco - Tc) / jnp.maximum(Th - Tc, 1e-12)
            F = lmtd_correction_factor(R, P, 1)

        def pick(name):
            return over[name] if over.get(name) is not None else getattr(p, name)

        tube_hot = p.tube_side == "hot"
        side = lambda hot, cold: hot if tube_hot else cold  # noqa: E731
        sh = lambda hot, cold: cold if tube_hot else hot  # noqa: E731
        a = {
            "Q": Q, "F": F, "lmtd": lmtd, "R": R, "P": P,
            "T_hot_in": Th, "T_hot_out": Tho, "T_cold_in": Tc, "T_cold_out": Tco,
            "m_t": _f64(side(hot_in, cold_in)["m_dot"]), "m_s": _f64(sh(hot_in, cold_in)["m_dot"]),
            "rho_t": _f64(side(p.rho_hot, p.rho_cold)), "mu_t": _f64(side(p.mu_hot, p.mu_cold)),
            "k_t": _f64(side(p.k_hot, p.k_cold)), "Cp_t": _f64(side(p.Cp_hot, p.Cp_cold)),
            "rho_s": _f64(sh(p.rho_hot, p.rho_cold)), "mu_s": _f64(sh(p.mu_hot, p.mu_cold)),
            "k_s": _f64(sh(p.k_hot, p.k_cold)), "Cp_s": _f64(sh(p.Cp_hot, p.Cp_cold)),
            "mu_wt": side(p.mu_wall_hot, p.mu_wall_cold), "mu_ws": sh(p.mu_wall_hot, p.mu_wall_cold),
            "tube_OD": _f64(pick("tube_OD")), "tube_ID": _f64(pick("tube_ID")),
            "pitch": _f64(pick("pitch_ratio")) * _f64(pick("tube_OD")),
            "n_passes": _f64(pick("n_tube_passes")),
            "clearance": _f64(p.shell_clearance), "k_wall": _f64(pick("k_wall")),
            "R_fi": _f64(pick("R_fi")), "R_fo": _f64(pick("R_fo")), "roughness": _f64(p.roughness),
            "bs_ratio": _f64(p.baffle_spacing_ratio),
        }
        v, L, lB = pick("tube_velocity"), pick("tube_length"), pick("baffle_spacing")
        a["v"] = None if v is None else _f64(v)
        a["L"] = None if L is None else _f64(L)
        a["lB"] = None if lB is None else _f64(lB)
        if a["v"] is not None:
            a["L"] = None  # length follows from the area
        return a

    def _geometry(self, A, a):
        """Geometry for an area ``A`` (m^2): tube count, length, shell, baffles."""
        p = self.params
        do, di, Np = a["tube_OD"], a["tube_ID"], a["n_passes"]
        if a["v"] is not None:
            per_pass = a["m_t"] / (a["rho_t"] * a["v"] * jnp.pi / 4.0 * di**2)
            Nt = Np * per_pass
            L = A / (jnp.pi * do * Nt)
        else:
            L = a["L"]
            Nt = A / (jnp.pi * do * L)
        Db = bundle_diameter(Nt, do, Np, p.layout)
        Ds = Db + a["clearance"]
        lB = a["bs_ratio"] * Ds if a["lB"] is None else a["lB"]
        return {"n_tubes": Nt, "tube_length": L, "bundle_ID": Db, "shell_ID": Ds, "baffle_spacing": lB, "area": A}

    def _film_and_U(self, g, a):
        """Film coefficients, U and pressure drops for a geometry dict ``g``."""
        p = self.params
        do, di, Np = a["tube_OD"], a["tube_ID"], a["n_passes"]
        tube = tube_side_pressure_drop(a["m_t"], a["rho_t"], a["mu_t"], g["n_tubes"], Np, di, g["tube_length"], a["roughness"])
        Pr_t = ht.prandtl(a["Cp_t"], a["mu_t"], a["k_t"])
        mu_ratio = 1.0 if a["mu_wt"] is None else a["mu_t"] / _f64(a["mu_wt"])
        h_i = ht.internal_h(tube["Re"], Pr_t, a["k_t"], di, L=g["tube_length"] * Np, mu_ratio=mu_ratio,
                            rel_roughness=a["roughness"] / di, heating=(p.tube_side == "cold"))
        if p.shell_h is None:
            shell = kern_shell_side(a["m_s"], a["rho_s"], a["mu_s"], a["k_s"], a["Cp_s"], g["shell_ID"], do,
                                    a["pitch"], g["baffle_spacing"], p.layout, a["mu_ws"])
            h_o = shell["h_o"]
            dPs = kern_shell_pressure_drop(a["m_s"], a["rho_s"], a["mu_s"], g["shell_ID"], do, a["pitch"],
                                           g["baffle_spacing"], g["tube_length"], p.layout, a["mu_ws"])
        else:
            h_o = p.shell_h({"tube_OD": do, "n_tubes": g["n_tubes"], "tube_length": g["tube_length"],
                             "shell_ID": g["shell_ID"], "area": g["area"]}) if callable(p.shell_h) else _f64(p.shell_h)
            shell = {"Re": jnp.asarray(jnp.nan), "us": jnp.asarray(jnp.nan), "De": jnp.asarray(jnp.nan)}
            dPs = {"dP": jnp.asarray(0.0)}
        U = ht.overall_U(h_i, h_o, di, do, a["k_wall"], a["R_fi"], a["R_fo"], "outer")
        return U, {"h_i": h_i, "h_o": _f64(h_o), "tube": tube, "shell": shell, "dP_shell": dPs["dP"]}

    def _U_map(self, U, a):
        A = a["Q"] / (U * a["F"] * a["lmtd"])
        g = self._geometry(A, a)
        U_new, _ = self._film_and_U(g, a)
        p = self.params
        return (1.0 - p.damping) * U + p.damping * U_new

    def __call__(
        self,
        hot_in: dict,
        cold_in: dict,
        Q: Array | float | None = None,
        T_hot_out: Array | float | None = None,
        T_cold_out: Array | float | None = None,
        warn: bool = True,
        **overrides,
    ) -> dict[str, Array]:
        """Design the exchanger.

        Args:
            hot_in: ``{"T": K, "m_dot": kg/s}`` for the hot stream.
            cold_in: Same for the cold stream.
            Q: Duty (W), or give exactly one of the next two.
            T_hot_out: Hot outlet temperature (K).
            T_cold_out: Cold outlet temperature (K).
            warn: Emit ``UserWarning`` for ``F < 0.75``, exceeded pressure-drop limits
                or an unconverged iteration (only when the values are concrete).
            **overrides: Replacement values for ``tube_velocity``, ``tube_length``,
                ``baffle_spacing``, ``n_tube_passes``, ``tube_OD``, ``tube_ID``,
                ``pitch_ratio``, ``k_wall``, ``R_fi``, ``R_fo``. These may be traced:
                differentiate the result with respect to them.

        Returns:
            Dict with ``Q``, ``U``, ``area``, ``UA``, ``F``, ``LMTD``, the four
            temperatures, ``n_tubes``, ``tube_length``, ``shell_ID``,
            ``bundle_ID``, ``baffle_spacing``, ``n_baffles`` (``L/l_B - 1``),
            ``n_tube_passes``, ``tube_velocity``, ``h_i``, ``h_o``,
            ``Re_tube``, ``Re_shell``, ``dP_tube``, ``dP_shell`` (Pa),
            ``U_residual`` (relative fixed-point residual), flags
            ``converged``, ``F_too_low``, ``tube_laminar`` (``Re_tube < 2300``),
            ``dP_tube_exceeded`` and ``dP_shell_exceeded``.

        Raises:
            ValueError: Not exactly one of ``Q``/``T_*_out``; neither
                ``tube_velocity`` nor ``tube_length``; unknown override.
        """
        p = self.params
        known = {"tube_velocity", "tube_length", "baffle_spacing", "n_tube_passes", "tube_OD", "tube_ID",
                 "pitch_ratio", "k_wall", "R_fi", "R_fo"}
        bad = set(overrides) - known
        if bad:
            raise ValueError(f"unknown override(s) {sorted(bad)}; use {sorted(known)}")
        a = self._args(hot_in, cold_in, Q, T_hot_out, T_cold_out, overrides)
        if a["v"] is None and a["L"] is None:
            raise ValueError("set tube_velocity or tube_length")

        solver = optx.FixedPointIteration(rtol=p.tol, atol=1e-8)
        sol = optx.fixed_point(lambda U, args: self._U_map(U, args), solver, _f64(p.U_guess), args=a,
                               max_steps=p.max_iter, throw=False)
        U = sol.value
        A = a["Q"] / (U * a["F"] * a["lmtd"])
        g = self._geometry(A, a)
        U_check, d = self._film_and_U(g, a)
        resid = jnp.abs(U_check - U) / U
        out = {
            "Q": a["Q"], "U": U, "area": A, "UA": U * A, "F": a["F"], "LMTD": a["lmtd"], "R": a["R"], "P": a["P"],
            "T_hot_in": a["T_hot_in"], "T_hot_out": a["T_hot_out"],
            "T_cold_in": a["T_cold_in"], "T_cold_out": a["T_cold_out"],
            "n_tubes": g["n_tubes"], "tube_length": g["tube_length"], "shell_ID": g["shell_ID"],
            "bundle_ID": g["bundle_ID"], "baffle_spacing": g["baffle_spacing"],
            "n_baffles": g["tube_length"] / g["baffle_spacing"] - 1.0, "n_tube_passes": a["n_passes"],
            "tube_velocity": d["tube"]["v"], "h_i": d["h_i"], "h_o": d["h_o"],
            "Re_tube": d["tube"]["Re"], "Re_shell": d["shell"]["Re"],
            "dP_tube": d["tube"]["dP"], "dP_shell": d["dP_shell"], "U_residual": resid,
        }
        out["converged"] = resid < 1e-6
        out["tube_laminar"] = out["Re_tube"] < ht.RE_LAMINAR
        out["F_too_low"] = out["F"] < 0.75
        out["dP_tube_exceeded"] = out["dP_tube"] > p.dP_tube_max
        out["dP_shell_exceeded"] = out["dP_shell"] > p.dP_shell_max
        if warn:
            self._warn(out)
        return out

    @staticmethod
    def _warn(out):
        keys = ("F", "dP_tube", "dP_shell", "Re_tube", "U_residual")
        if any(isinstance(out[k], jax.core.Tracer) for k in keys):
            return  # under jit / grad: no concrete values to warn about
        if bool(out["F_too_low"]):
            warnings.warn(f"LMTD correction factor F = {float(out['F']):.3f} < 0.75: use more shells in series "
                          "or a different layout.", UserWarning, stacklevel=3)
        if bool(out["dP_tube_exceeded"]):
            warnings.warn(f"tube-side pressure drop {float(out['dP_tube']):.4g} Pa exceeds the limit.",
                          UserWarning, stacklevel=3)
        if bool(out["dP_shell_exceeded"]):
            warnings.warn(f"shell-side pressure drop {float(out['dP_shell']):.4g} Pa exceeds the limit.",
                          UserWarning, stacklevel=3)
        if not bool(out["converged"]):
            warnings.warn(f"U iteration did not converge (relative residual {float(out['U_residual']):.2e}).",
                          UserWarning, stacklevel=3)
        if bool(out["tube_laminar"]):
            warnings.warn(f"tube-side Re = {float(out['Re_tube']):.0f} < 2300: the iteration may have settled on the "
                          "large-area, low-velocity fixed point; try a larger U_guess or a tube_velocity.",
                          UserWarning, stacklevel=3)

    # ------------------------------------------------------------------
    def rate(
        self,
        hot_in: dict,
        cold_in: dict,
        n_tubes: Array | float,
        tube_length: Array | float,
        shell_ID: Array | float,
        baffle_spacing: Array | float,
        n_tube_passes: Array | float | None = None,
        U: Array | float | None = None,
    ) -> dict[str, Array]:
        """Rate a *given* geometry (e.g. after :func:`round_design`).

        Computes ``h_i``, ``h_o``, ``U`` (or uses the ``U`` supplied), the area
        ``pi d_o N_t L``, ``UA``, then the outlet temperatures and duty from the
        1-2N effectiveness :func:`difflow.units.heat_exchanger.effectiveness_shell_and_tube` (an isothermal side
        has ``Cr = 0``). Also returns both pressure drops and the area margin
        relative to a duty-fixed requirement is left to the caller
        (``area / A_required - 1``).

        Args:
            hot_in: ``{"T", "m_dot"}`` of the hot stream.
            cold_in: Same for the cold stream.
            n_tubes: Number of tubes.
            tube_length: Tube length (m).
            shell_ID: Shell inside diameter (m).
            baffle_spacing: Baffle spacing (m).
            n_tube_passes: Tube passes (default ``params.n_tube_passes``).
            U: Overall coefficient (W/m^2/K) to use instead of the correlations.

        Returns:
            Dict with ``Q``, ``T_hot_out``, ``T_cold_out``, ``U``, ``area``, ``UA``,
            ``h_i``, ``h_o``, ``dP_tube``, ``dP_shell``, ``effectiveness``, ``NTU``, ``F``.
        """
        p = self.params
        Th, Tc = _f64(hot_in["T"]), _f64(cold_in["T"])
        a = self._args(hot_in, cold_in, 1.0, None, None, {"tube_velocity": None})
        a["n_passes"] = _f64(p.n_tube_passes if n_tube_passes is None else n_tube_passes)
        a["v"] = None
        Nt = _f64(n_tubes)
        L = _f64(tube_length)
        area = jnp.pi * a["tube_OD"] * Nt * L
        g = {"n_tubes": Nt, "tube_length": L, "shell_ID": _f64(shell_ID), "baffle_spacing": _f64(baffle_spacing),
             "area": area}
        U_corr, d = self._film_and_U(g, a)
        Uuse = U_corr if U is None else _f64(U)
        Ch = _f64(hot_in["m_dot"]) * _f64(p.Cp_hot)
        Cc = _f64(cold_in["m_dot"]) * _f64(p.Cp_cold)
        if p.hot_isothermal:
            Ch = jnp.asarray(1e30)
        if p.cold_isothermal:
            Cc = jnp.asarray(1e30)
        Cmin, Cmax = jnp.minimum(Ch, Cc), jnp.maximum(Ch, Cc)
        NTU = Uuse * area / Cmin
        eps = effectiveness_shell_and_tube(NTU, Cmin / Cmax, 1)
        Q = eps * Cmin * (Th - Tc)
        Tho = Th if p.hot_isothermal else Th - Q / Ch
        Tco = Tc if p.cold_isothermal else Tc + Q / Cc
        return {
            "Q": Q, "T_hot_out": Tho, "T_cold_out": Tco, "U": Uuse, "U_correlation": U_corr, "area": area,
            "UA": Uuse * area, "h_i": d["h_i"], "h_o": d["h_o"], "dP_tube": d["tube"]["dP"],
            "dP_shell": d["dP_shell"], "effectiveness": eps, "NTU": NTU,
        }


#: Even tube-pass counts built in practice.
STANDARD_TUBE_PASSES = (1, 2, 4, 6, 8)


def round_design(design: dict, baffle_step: float = 0.025, shell_step: float = 0.0127) -> dict[str, float]:
    """Round a continuous design to a buildable geometry (concrete values only).

    The continuous design treats tube count, pass count and baffle count as real
    numbers. To build it: round the **tube count up** to the next integer
    (``ceil``), the **passes** to the nearest standard even count in
    :data:`STANDARD_TUBE_PASSES`, the **baffle spacing** *down* to a multiple of
    ``baffle_step`` (closer baffles raise ``h_o`` and ``dP``), and the **shell
    ID** up to a multiple of ``shell_step`` (0.5 in). Then call
    :meth:`ShellAndTubeDesign.rate` on the result: the area margin
    ``rate(...)["area"] / design["area"] - 1`` and the pressure drops must be
    re-checked, since rounding changes the tube velocity, ``h_i``, ``h_o`` and
    both pressure drops. The steps are illustrative, not TEMA standards.

    Args:
        design: Result of :meth:`ShellAndTubeDesign.__call__` (concrete).
        baffle_step: Baffle-spacing resolution (m).
        shell_step: Shell-diameter resolution (m).

    Returns:
        Dict of Python floats with ``n_tubes``, ``n_tube_passes``,
        ``tube_length``, ``shell_ID`` and ``baffle_spacing``.
    """
    passes = min(STANDARD_TUBE_PASSES, key=lambda n: abs(n - float(design["n_tube_passes"])))
    return {
        "n_tubes": float(math.ceil(float(design["n_tubes"]))),
        "n_tube_passes": float(passes),
        "tube_length": float(design["tube_length"]),
        "shell_ID": math.ceil(float(design["shell_ID"]) / shell_step) * shell_step,
        "baffle_spacing": max(math.floor(float(design["baffle_spacing"]) / baffle_step), 1) * baffle_step,
    }


__all__ = [
    "BUNDLE_CONSTANTS",
    "BWG_WALL_THICKNESS",
    "TEMA_TUBE_PITCH",
    "KERN_NU_COEFFICIENT",
    "KERN_NU_EXPONENT",
    "KERN_FRICTION_A",
    "KERN_FRICTION_B",
    "STANDARD_TUBE_PASSES",
    "ShellAndTubeDesign",
    "ShellAndTubeDesignParams",
    "bundle_diameter",
    "equivalent_diameter",
    "kern_friction_factor",
    "kern_shell_pressure_drop",
    "kern_shell_side",
    "round_design",
    "tube_count",
    "tube_inside_diameter",
    "tube_side_pressure_drop",
]
