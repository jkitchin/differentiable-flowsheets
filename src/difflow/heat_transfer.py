"""Heat-transfer correlations: film coefficients, phase change and overall U.

Pure functions, JAX, SI units, differentiable: ``jax.grad`` of an overall
coefficient with respect to the tube velocity, the diameter or any property
works, including through the laminar-transition-turbulent blend. Nothing here
is a palette unit; the heat exchangers (:mod:`difflow.units.heat_exchanger`)
still take ``U`` / ``UA`` as numbers, and these functions compute the number.

Liquid and gas properties are *inputs* (core ``SpeciesData`` has no liquid
thermal conductivity or viscosity): every function takes ``k``, ``mu``,
``Cp`` and ``rho`` explicitly, as :mod:`difflow.fluids` does.

Contents
--------
* Dimensionless groups: :func:`reynolds`, :func:`prandtl`, :func:`nusselt_to_h`,
  :func:`grashof`, :func:`rayleigh`
* Conduction: :func:`slab_resistance`, :func:`cylinder_resistance`,
  :func:`sphere_resistance`, :func:`composite_wall`,
  :func:`critical_insulation_radius`, :func:`fin_efficiency_straight`
* Internal forced convection: :data:`NU_LAMINAR_TW`, :data:`NU_LAMINAR_QS`,
  :func:`sieder_tate_laminar`, :func:`dittus_boelter`, :func:`sieder_tate`,
  :func:`gnielinski`, :func:`internal_nusselt`, :func:`internal_h`
* External forced convection: :func:`churchill_bernstein`,
  :func:`flat_plate_nusselt`
* Natural convection: :func:`churchill_chu_vertical_plate`,
  :func:`churchill_chu_horizontal_cylinder`
* Phase change: :func:`nusselt_film_condensation`,
  :func:`rohsenow_heat_flux`, :func:`rohsenow_superheat`, :func:`mostinski`,
  :func:`critical_heat_flux`
* Overall coefficient: :func:`overall_U`, :data:`FOULING_RESISTANCES`,
  :data:`TYPICAL_U_RANGES`, :func:`typical_U_range`

Not included (out of scope or optional, see ``docs``): annular fins, tube banks
(Zukauskas), the Bell-Delaware shell-side method (Kern is in :mod:`difflow.shell_and_tube`), radiation, transient
conduction.

Example:
    >>> from difflow.heat_transfer import internal_h, overall_U
    >>> # water at ~40 C in a 25 mm tube at 1.5 m/s
    >>> h = internal_h(Re=1.5 * 992.0 * 0.025 / 653e-6, Pr=4.3, k=0.631, D=0.025)
    >>> 5000.0 < float(h) < 10000.0
    True
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array

from difflow import fluids

#: Standard gravity (m/s^2).
GRAVITY = fluids.GRAVITY

#: Fully developed laminar Nusselt number in a circular tube, uniform wall
#: temperature (Incropera & DeWitt Eq. 8.55).
NU_LAMINAR_TW = 3.66
#: Same, uniform wall heat flux (Incropera & DeWitt Eq. 8.53).
NU_LAMINAR_QS = 4.36

#: Reynolds numbers bounding the laminar-turbulent blend in tubes. Below
#: ``RE_LAMINAR`` the result is purely laminar, above ``RE_TURBULENT`` purely
#: turbulent. (2300 / 4000 are the limits of Gnielinski's / VDI's interpolation.)
RE_LAMINAR = 2300.0
RE_TURBULENT = 4000.0

#: Flat-plate critical Reynolds number (Incropera & DeWitt Sec. 7.2).
RE_CRITICAL_PLATE = 5.0e5
_RE_PLATE_BLEND_HI = 5.5e5

_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)  # noqa: E731


def _smooth_weight(x: Array, lo: float, hi: float) -> Array:
    """C2 smootherstep of ``ln x``: exactly 0 for ``x <= lo`` and 1 for ``x >= hi``."""
    t = (jnp.log(jnp.maximum(x, 1e-300)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    t = jnp.clip(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


# =============================================================================
# Dimensionless groups
# =============================================================================


def reynolds(rho: Array | float, v: Array | float, D: Array | float, mu: Array | float) -> Array:
    """Reynolds number ``Re = rho v D / mu`` (alias of :func:`difflow.fluids.reynolds_number`).

    Args:
        rho: Density (kg/m^3).
        v: Velocity (m/s).
        D: Characteristic length, e.g. tube diameter (m).
        mu: Dynamic viscosity (Pa s).

    Returns:
        Reynolds number (-).
    """
    return fluids.reynolds_number(rho, v, D, mu)


def prandtl(Cp: Array | float, mu: Array | float, k: Array | float) -> Array:
    """Prandtl number ``Pr = Cp mu / k``.

    Args:
        Cp: Specific heat (J/kg/K) -- mass basis.
        mu: Dynamic viscosity (Pa s).
        k: Thermal conductivity (W/m/K).

    Returns:
        Prandtl number (-).
    """
    return _f64(Cp) * _f64(mu) / _f64(k)


def nusselt_to_h(Nu: Array | float, k: Array | float, L: Array | float) -> Array:
    """Film coefficient ``h = Nu k / L`` (W/m^2/K).

    Args:
        Nu: Nusselt number (-).
        k: Fluid thermal conductivity (W/m/K).
        L: Characteristic length the Nusselt number is based on (m).

    Returns:
        Heat-transfer coefficient (W/m^2/K).
    """
    return _f64(Nu) * _f64(k) / _f64(L)


def grashof(
    beta: Array | float,
    dT: Array | float,
    L: Array | float,
    nu: Array | float,
    g: float = GRAVITY,
) -> Array:
    """Grashof number ``Gr = g beta |dT| L^3 / nu^2``.

    Args:
        beta: Volumetric expansion coefficient (1/K); ``1/T`` for an ideal gas.
        dT: Surface-to-fluid temperature difference (K).
        L: Characteristic length (m).
        nu: Kinematic viscosity ``mu / rho`` (m^2/s).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Grashof number (-).
    """
    return g * _f64(beta) * jnp.abs(_f64(dT)) * _f64(L) ** 3 / _f64(nu) ** 2


def rayleigh(
    beta: Array | float,
    dT: Array | float,
    L: Array | float,
    nu: Array | float,
    alpha: Array | float,
    g: float = GRAVITY,
) -> Array:
    """Rayleigh number ``Ra = Gr Pr = g beta |dT| L^3 / (nu alpha)``.

    Args:
        beta: Volumetric expansion coefficient (1/K).
        dT: Surface-to-fluid temperature difference (K).
        L: Characteristic length (m).
        nu: Kinematic viscosity (m^2/s).
        alpha: Thermal diffusivity ``k / (rho Cp)`` (m^2/s).
        g: Gravitational acceleration (m/s^2).

    Returns:
        Rayleigh number (-).
    """
    return grashof(beta, dT, L, nu, g) * _f64(nu) / _f64(alpha)


# =============================================================================
# Conduction
# =============================================================================


def slab_resistance(thickness: Array | float, k: Array | float, area: Array | float = 1.0) -> Array:
    """Thermal resistance of a plane wall ``R = t / (k A)`` (K/W).

    Args:
        thickness: Wall thickness (m).
        k: Thermal conductivity (W/m/K).
        area: Heat-flow area (m^2). With the default 1 the result is per unit
            area (m^2 K/W), directly comparable with fouling resistances.

    Returns:
        Resistance (K/W, or m^2 K/W for unit area).
    """
    return _f64(thickness) / (_f64(k) * _f64(area))


def cylinder_resistance(r_in: Array | float, r_out: Array | float, k: Array | float, L: Array | float) -> Array:
    """Radial resistance of a cylindrical shell ``R = ln(r_out/r_in) / (2 pi k L)`` (K/W).

    Args:
        r_in: Inner radius (m).
        r_out: Outer radius (m).
        k: Thermal conductivity (W/m/K).
        L: Length (m).

    Returns:
        Resistance (K/W).
    """
    return jnp.log(_f64(r_out) / _f64(r_in)) / (2.0 * jnp.pi * _f64(k) * _f64(L))


def sphere_resistance(r_in: Array | float, r_out: Array | float, k: Array | float) -> Array:
    """Radial resistance of a spherical shell ``R = (1/r_in - 1/r_out) / (4 pi k)`` (K/W).

    Args:
        r_in: Inner radius (m).
        r_out: Outer radius (m).
        k: Thermal conductivity (W/m/K).

    Returns:
        Resistance (K/W).
    """
    return (1.0 / _f64(r_in) - 1.0 / _f64(r_out)) / (4.0 * jnp.pi * _f64(k))


def composite_wall(resistances: list | tuple | Array) -> Array:
    """Series resistance: the sum of the individual resistances (K/W).

    Convenience for layered walls and film + wall + fouling stacks. All
    entries must be on the same area basis.

    Args:
        resistances: Sequence of resistances (K/W, or all per unit area).

    Returns:
        Total resistance.

    Example:
        >>> round(float(composite_wall([0.1, 0.2, 0.3])), 12)
        0.6
    """
    return sum((_f64(r) for r in resistances), start=_f64(0.0))


def critical_insulation_radius(k_ins: Array | float, h_out: Array | float, geometry: str = "cylinder") -> Array:
    """Critical insulation radius: maximum heat loss when the outer radius equals it.

    ``r_cr = k / h`` for a cylinder, ``2 k / h`` for a sphere. Insulating
    below this radius *increases* the heat loss (Incropera & DeWitt Sec. 3.3).

    Args:
        k_ins: Insulation conductivity (W/m/K).
        h_out: Outer (convective) coefficient (W/m^2/K).
        geometry: ``"cylinder"`` or ``"sphere"``.

    Returns:
        Critical radius (m).

    Raises:
        ValueError: Unknown ``geometry``.
    """
    if geometry == "cylinder":
        return _f64(k_ins) / _f64(h_out)
    if geometry == "sphere":
        return 2.0 * _f64(k_ins) / _f64(h_out)
    raise ValueError(f"geometry must be 'cylinder' or 'sphere', got {geometry!r}")


def fin_efficiency_straight(h: Array | float, k: Array | float, t: Array | float, L: Array | float) -> Array:
    """Efficiency of a straight rectangular fin, corrected-length (convecting tip) form.

    ``eta = tanh(m L_c) / (m L_c)``, ``m = sqrt(2 h / (k t))``, ``L_c = L + t/2``
    (fin wide compared with its thickness ``t``). Annular fins need Bessel
    functions of the second kind, which JAX does not provide, and are not
    implemented.

    Args:
        h: Film coefficient (W/m^2/K).
        k: Fin conductivity (W/m/K).
        t: Fin thickness (m).
        L: Fin length from base to tip (m).

    Returns:
        Fin efficiency in (0, 1].
    """
    t = _f64(t)
    m = jnp.sqrt(2.0 * _f64(h) / (_f64(k) * t))
    mLc = jnp.maximum(m * (_f64(L) + 0.5 * t), 1e-12)
    return jnp.tanh(mLc) / mLc


# =============================================================================
# Internal forced convection (circular tubes)
# =============================================================================


def sieder_tate_laminar(
    Re: Array | float,
    Pr: Array | float,
    D_over_L: Array | float,
    mu_ratio: Array | float = 1.0,
) -> Array:
    """Sieder-Tate laminar entry-region Nusselt number (mean over the tube length).

    ``Nu = 1.86 (Re Pr D/L)^(1/3) (mu_b/mu_w)^0.14`` -- Sieder & Tate (1936);
    Incropera & DeWitt Eq. 8.57 (``Pr`` 0.48-16700, constant wall temperature).
    It has no fully developed limit; :func:`internal_nusselt` joins it to the
    fully developed value.

    Args:
        Re: Reynolds number based on the diameter (-).
        Pr: Prandtl number (-).
        D_over_L: Diameter over heated length (-).
        mu_ratio: Bulk over wall viscosity ``mu_b / mu_w`` (-).

    Returns:
        Mean Nusselt number (-).
    """
    Gz = jnp.maximum(_f64(Re) * _f64(Pr) * _f64(D_over_L), 1e-300)
    return 1.86 * Gz ** (1.0 / 3.0) * _f64(mu_ratio) ** 0.14


def dittus_boelter(Re: Array | float, Pr: Array | float, heating: bool | Array = True) -> Array:
    """Dittus-Boelter: ``Nu = 0.023 Re^0.8 Pr^n``, ``n = 0.4`` heating, ``0.3`` cooling.

    Valid for fully developed turbulent flow, ``Re >= 1e4``, ``0.6 <= Pr <= 160``,
    ``L/D >= 10``, moderate wall-to-fluid temperature difference
    (Incropera & DeWitt Eq. 8.60).

    Args:
        Re: Reynolds number (-).
        Pr: Prandtl number (-).
        heating: True if the fluid is heated (wall hotter than the fluid).

    Returns:
        Nusselt number (-).
    """
    n = jnp.where(jnp.asarray(heating), 0.4, 0.3)
    return 0.023 * jnp.maximum(_f64(Re), 1e-12) ** 0.8 * _f64(Pr) ** n


def sieder_tate(Re: Array | float, Pr: Array | float, mu_ratio: Array | float = 1.0) -> Array:
    """Sieder-Tate turbulent: ``Nu = 0.027 Re^0.8 Pr^(1/3) (mu_b/mu_w)^0.14``.

    For large property variation (viscous liquids); ``Re >= 1e4``,
    ``0.7 <= Pr <= 16700`` (Incropera & DeWitt Eq. 8.61).

    Args:
        Re: Reynolds number (-).
        Pr: Prandtl number (-).
        mu_ratio: Bulk over wall viscosity (-).

    Returns:
        Nusselt number (-).
    """
    return 0.027 * jnp.maximum(_f64(Re), 1e-12) ** 0.8 * _f64(Pr) ** (1.0 / 3.0) * _f64(mu_ratio) ** 0.14


def gnielinski(Re: Array | float, Pr: Array | float, f: Array | float) -> Array:
    """Gnielinski (1976): turbulent and transitional tube flow, given the Darcy ``f``.

    ``Nu = (f/8)(Re - 1000) Pr / [1 + 12.7 sqrt(f/8) (Pr^(2/3) - 1)]``;
    ``3000 <= Re <= 5e6``, ``0.5 <= Pr <= 2000`` (Incropera & DeWitt Eq. 8.62).
    ``f`` is the **Darcy** friction factor, e.g. from
    :func:`difflow.fluids.friction_factor`. (Below Re = 1000 the formula goes
    negative; use :func:`internal_nusselt`, which never evaluates it there.)

    Args:
        Re: Reynolds number (-).
        Pr: Prandtl number (-).
        f: Darcy friction factor (-).

    Returns:
        Nusselt number (-).
    """
    f8 = _f64(f) / 8.0
    Pr = _f64(Pr)
    return f8 * (_f64(Re) - 1000.0) * Pr / (1.0 + 12.7 * jnp.sqrt(f8) * (Pr ** (2.0 / 3.0) - 1.0))


_TURBULENT_METHODS = ("gnielinski", "dittus_boelter", "sieder_tate")
_BOUNDARIES = ("T", "q")


def internal_nusselt(
    Re: Array | float,
    Pr: Array | float,
    D_over_L: Array | float | None = None,
    mu_ratio: Array | float = 1.0,
    rel_roughness: Array | float = 0.0,
    method: str = "gnielinski",
    heating: bool | Array = True,
    boundary: str = "T",
) -> Array:
    """Nusselt number in a circular tube, laminar through turbulent, smooth in Re.

    * Laminar (``Re <= 2300``): fully developed ``Nu = 3.66`` (``boundary="T"``,
      constant wall temperature) or ``4.36`` (``"q"``, constant flux). If
      ``D_over_L`` is given, the entry-region
      :func:`sieder_tate_laminar` is joined by a smooth asymptotic combination
      ``Nu = (Nu_fd^3 + Nu_entry^3)^(1/3)`` (Churchill-Usagi; *not* a published
      correlation -- it just enforces the fully developed floor and the
      Sieder-Tate entry behaviour).
    * Turbulent (``Re >= 4000``): ``method`` is ``"gnielinski"`` (friction factor
      from :func:`difflow.fluids.friction_factor`, Colebrook), ``"dittus_boelter"``
      or ``"sieder_tate"`` (``mu_ratio`` applies to the latter).
    * Between 2300 and 4000 the two are blended by a C2 smootherstep of
      ``ln Re`` (zero first and second derivative at both ends; compare the
      linear interpolation of Gnielinski / VDI). Both branches are evaluated
      at the actual ``Re`` there (clamped to the blend window outside it, where
      their weight and its derivatives are exactly zero, so the clamp leaves no
      kink in the gradient). Real transitional flow is intermittent and no
      correlation is accurate there; the blend only makes ``h`` continuous
      and differentiable.

    No Python branching on ``Re``: ``Re``, ``Pr`` etc. may be traced.

    Args:
        Re: Reynolds number (-).
        Pr: Prandtl number (-).
        D_over_L: Diameter over length, or None for fully developed laminar flow.
        mu_ratio: Bulk over wall viscosity ``mu_b / mu_w`` (-).
        rel_roughness: Relative roughness for the Gnielinski friction factor.
        method: Turbulent correlation (Python string).
        heating: Dittus-Boelter exponent selector (True: heating).
        boundary: ``"T"`` or ``"q"`` laminar fully developed value.

    Returns:
        Nusselt number based on the tube diameter (-).

    Raises:
        ValueError: Unknown ``method`` or ``boundary``.
    """
    if method not in _TURBULENT_METHODS:
        raise ValueError(f"Unknown method {method!r}; use one of {_TURBULENT_METHODS}")
    if boundary not in _BOUNDARIES:
        raise ValueError(f"boundary must be one of {_BOUNDARIES}, got {boundary!r}")
    Re = jnp.maximum(_f64(Re), 1e-12)
    Pr = _f64(Pr)
    Nu_fd = NU_LAMINAR_TW if boundary == "T" else NU_LAMINAR_QS

    Re_l = jnp.minimum(Re, RE_TURBULENT)
    if D_over_L is None:
        Nu_lam = jnp.full_like(Re_l * Pr, Nu_fd)
    else:
        Nu_en = sieder_tate_laminar(Re_l, Pr, D_over_L, mu_ratio)
        Nu_lam = (Nu_fd**3 + Nu_en**3) ** (1.0 / 3.0)

    Re_t = jnp.maximum(Re, RE_LAMINAR)
    if method == "gnielinski":
        f = fluids.friction_factor(Re_t, rel_roughness)
        Nu_turb = gnielinski(Re_t, Pr, f)
    elif method == "dittus_boelter":
        Nu_turb = dittus_boelter(Re_t, Pr, heating)
    else:
        Nu_turb = sieder_tate(Re_t, Pr, mu_ratio)

    w = _smooth_weight(Re, RE_LAMINAR, RE_TURBULENT)
    return (1.0 - w) * Nu_lam + w * Nu_turb


def internal_h(
    Re: Array | float,
    Pr: Array | float,
    k: Array | float,
    D: Array | float,
    L: Array | float | None = None,
    mu_ratio: Array | float = 1.0,
    rel_roughness: Array | float = 0.0,
    method: str = "gnielinski",
    heating: bool | Array = True,
    boundary: str = "T",
) -> Array:
    """Tube-side film coefficient ``h = Nu k / D`` (W/m^2/K); see :func:`internal_nusselt`.

    Args:
        Re: Reynolds number ``rho v D / mu`` (-).
        Pr: Prandtl number ``Cp mu / k`` (-).
        k: Fluid thermal conductivity (W/m/K).
        D: Tube inside diameter (m).
        L: Tube length (m) for the laminar entry effect, or None.
        mu_ratio: Bulk over wall viscosity (-).
        rel_roughness: Relative roughness ``eps / D`` (-).
        method: ``"gnielinski"``, ``"dittus_boelter"`` or ``"sieder_tate"``.
        heating: True if the fluid is being heated.
        boundary: ``"T"`` (constant wall temperature) or ``"q"`` (constant flux).

    Returns:
        Heat-transfer coefficient (W/m^2/K).
    """
    D_over_L = None if L is None else _f64(D) / _f64(L)
    Nu = internal_nusselt(Re, Pr, D_over_L, mu_ratio, rel_roughness, method, heating, boundary)
    return nusselt_to_h(Nu, k, D)


# =============================================================================
# External forced convection
# =============================================================================


def churchill_bernstein(Re: Array | float, Pr: Array | float) -> Array:
    """Churchill-Bernstein (1977): mean Nusselt number, cylinder in cross flow.

    ``Nu_D = 0.3 + 0.62 Re^(1/2) Pr^(1/3) / [1 + (0.4/Pr)^(2/3)]^(1/4)
    * [1 + (Re/282000)^(5/8)]^(4/5)``; all ``Re`` with ``Re Pr > 0.2``;
    properties at the film temperature (Incropera & DeWitt Eq. 7.57).

    Args:
        Re: Reynolds number based on the cylinder diameter (-).
        Pr: Prandtl number (-).

    Returns:
        Mean Nusselt number (-).
    """
    Re = jnp.maximum(_f64(Re), 1e-12)
    Pr = _f64(Pr)
    return 0.3 + 0.62 * Re**0.5 * Pr ** (1.0 / 3.0) / (1.0 + (0.4 / Pr) ** (2.0 / 3.0)) ** 0.25 * (
        1.0 + (Re / 282000.0) ** (5.0 / 8.0)
    ) ** 0.8


def flat_plate_nusselt(Re_L: Array | float, Pr: Array | float, regime: str = "mixed") -> Array:
    """Mean Nusselt number over a flat plate of length ``L`` at uniform temperature.

    * ``"laminar"``: ``0.664 Re_L^(1/2) Pr^(1/3)`` (``Pr >= 0.6``, Incropera
      Eq. 7.31).
    * ``"turbulent"``: ``0.037 Re_L^(4/5) Pr^(1/3)`` (fully turbulent from the
      leading edge, Eq. 7.41).
    * ``"mixed"``: ``(0.037 Re_L^(4/5) - 871) Pr^(1/3)`` for ``Re_L > 5e5``
      (laminar then turbulent, Eq. 7.38 with ``Re_c = 5e5``), joined to the
      laminar form by a smooth weight over ``Re_L`` 4e5 to 5e5 (the two agree
      to 0.6% at 5e5, so the join barely moves the value).

    Args:
        Re_L: Reynolds number based on the plate length (-).
        Pr: Prandtl number (-).
        regime: ``"laminar"``, ``"turbulent"`` or ``"mixed"``.

    Returns:
        Mean Nusselt number ``h L / k`` (-).

    Raises:
        ValueError: Unknown ``regime``.
    """
    Re = jnp.maximum(_f64(Re_L), 1e-12)
    P13 = _f64(Pr) ** (1.0 / 3.0)
    if regime == "laminar":
        return 0.664 * Re**0.5 * P13
    if regime == "turbulent":
        return 0.037 * Re**0.8 * P13
    if regime == "mixed":
        lam = 0.664 * Re**0.5 * P13
        mix = (0.037 * jnp.maximum(Re, RE_CRITICAL_PLATE) ** 0.8 - 871.0) * P13
        w = _smooth_weight(Re, RE_CRITICAL_PLATE, _RE_PLATE_BLEND_HI)
        return (1.0 - w) * lam + w * mix
    raise ValueError("regime must be 'laminar', 'turbulent' or 'mixed'")


# =============================================================================
# Natural convection
# =============================================================================


def churchill_chu_vertical_plate(Ra: Array | float, Pr: Array | float) -> Array:
    """Churchill-Chu (1975) mean Nusselt number, isothermal vertical plate, all ``Ra_L``.

    ``Nu_L = {0.825 + 0.387 Ra^(1/6) / [1 + (0.492/Pr)^(9/16)]^(8/27)}^2``
    (Incropera & DeWitt Eq. 9.26).

    Args:
        Ra: Rayleigh number based on the plate height (-).
        Pr: Prandtl number (-).

    Returns:
        Mean Nusselt number (-).
    """
    Ra = jnp.maximum(_f64(Ra), 1e-12)
    return (0.825 + 0.387 * Ra ** (1.0 / 6.0) / (1.0 + (0.492 / _f64(Pr)) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2


def churchill_chu_horizontal_cylinder(Ra: Array | float, Pr: Array | float) -> Array:
    """Churchill-Chu (1975) mean Nusselt number, isothermal horizontal cylinder.

    ``Nu_D = {0.60 + 0.387 Ra_D^(1/6) / [1 + (0.559/Pr)^(9/16)]^(8/27)}^2``,
    ``Ra_D <= 1e12`` (Incropera & DeWitt Eq. 9.34).

    Args:
        Ra: Rayleigh number based on the diameter (-).
        Pr: Prandtl number (-).

    Returns:
        Mean Nusselt number (-).
    """
    Ra = jnp.maximum(_f64(Ra), 1e-12)
    return (0.60 + 0.387 * Ra ** (1.0 / 6.0) / (1.0 + (0.559 / _f64(Pr)) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2


# =============================================================================
# Phase change
# =============================================================================

_CONDENSATION_COEFFS = {"vertical_plate": 0.943, "horizontal_tube": 0.729}


def nusselt_film_condensation(
    k_l: Array | float,
    rho_l: Array | float,
    rho_v: Array | float,
    mu_l: Array | float,
    h_fg: Array | float,
    Cp_l: Array | float,
    dT: Array | float,
    L: Array | float,
    geometry: str = "horizontal_tube",
    n_tubes: Array | float = 1.0,
) -> Array:
    """Nusselt laminar film condensation: mean coefficient (W/m^2/K).

    ``h = C [g rho_l (rho_l - rho_v) k_l^3 h_fg' / (mu_l dT L)]^(1/4)`` with the
    corrected latent heat ``h_fg' = h_fg + 0.68 Cp_l dT`` (Rohsenow) and
    ``dT = T_sat - T_wall``. ``C = 0.943`` and ``L`` the plate/tube height for
    ``"vertical_plate"``; ``C = 0.729`` and ``L = D`` the outside diameter for
    ``"horizontal_tube"`` (Incropera & DeWitt Eqs. 10.26, 10.40). A vertical
    column of ``n_tubes`` horizontal tubes (condensate draining from tube to
    tube, no splashing) has the mean coefficient ``h_N = h_1 N^(-1/4)``
    (Eq. 10.42, i.e. ``D -> N D``). Liquid properties at the film temperature
    ``(T_sat + T_wall)/2``; vapour density at saturation.

    Laminar, wave-free film only (condensate Reynolds number below ~30; no
    vapour shear, no non-condensables, no desuperheating).

    Args:
        k_l: Liquid conductivity (W/m/K).
        rho_l: Liquid density (kg/m^3).
        rho_v: Vapour density (kg/m^3).
        mu_l: Liquid viscosity (Pa s).
        h_fg: Latent heat of vaporization (J/kg).
        Cp_l: Liquid specific heat (J/kg/K).
        dT: ``T_sat - T_wall`` (K), positive.
        L: Plate height or tube outside diameter (m).
        geometry: ``"vertical_plate"`` or ``"horizontal_tube"``.
        n_tubes: Number of horizontal tubes in a vertical column.

    Returns:
        Mean heat-transfer coefficient (W/m^2/K).

    Raises:
        ValueError: Unknown ``geometry``.
    """
    if geometry not in _CONDENSATION_COEFFS:
        raise ValueError(f"geometry must be one of {tuple(_CONDENSATION_COEFFS)}, got {geometry!r}")
    dT = jnp.maximum(_f64(dT), 1e-12)
    rho_l, rho_v = _f64(rho_l), _f64(rho_v)
    h_fg_c = _f64(h_fg) + 0.68 * _f64(Cp_l) * dT
    L_eff = _f64(L) * (_f64(n_tubes) if geometry == "horizontal_tube" else 1.0)
    inner = GRAVITY * rho_l * (rho_l - rho_v) * _f64(k_l) ** 3 * h_fg_c / (_f64(mu_l) * dT * L_eff)
    return _CONDENSATION_COEFFS[geometry] * inner**0.25


def rohsenow_heat_flux(
    dT_e: Array | float,
    mu_l: Array | float,
    h_fg: Array | float,
    rho_l: Array | float,
    rho_v: Array | float,
    sigma: Array | float,
    Cp_l: Array | float,
    Pr_l: Array | float,
    C_sf: Array | float = 0.013,
    n: float = 1.0,
) -> Array:
    """Rohsenow (1952) nucleate pool boiling heat flux (W/m^2) at excess temperature ``dT_e``.

    ``q'' = mu_l h_fg [g (rho_l - rho_v)/sigma]^(1/2)
    [Cp_l dT_e / (C_sf h_fg Pr_l^n)]^3`` (Incropera & DeWitt Eq. 10.5).
    ``n = 1`` for water, ``1.7`` for other liquids. ``C_sf`` depends on the
    surface-fluid pair (see Note); the default 0.013 is the value usually
    quoted for water on polished copper or platinum, **transcribed from memory
    of Incropera & DeWitt Table 10.1 -- check it against the book for any real
    design.** Valid in the nucleate regime, below :func:`critical_heat_flux`.

    Args:
        dT_e: Excess temperature ``T_s - T_sat`` (K).
        mu_l: Liquid viscosity (Pa s).
        h_fg: Latent heat (J/kg).
        rho_l: Liquid density (kg/m^3).
        rho_v: Vapour density (kg/m^3).
        sigma: Surface tension (N/m).
        Cp_l: Liquid specific heat (J/kg/K).
        Pr_l: Liquid Prandtl number (-).
        C_sf: Surface-fluid constant (-).
        n: Prandtl exponent (1 for water, 1.7 otherwise).

    Returns:
        Heat flux (W/m^2).
    """
    cap = GRAVITY * (_f64(rho_l) - _f64(rho_v)) / _f64(sigma)
    x = _f64(Cp_l) * _f64(dT_e) / (_f64(C_sf) * _f64(h_fg) * _f64(Pr_l) ** n)
    return _f64(mu_l) * _f64(h_fg) * jnp.sqrt(cap) * x**3


def rohsenow_superheat(
    q: Array | float,
    mu_l: Array | float,
    h_fg: Array | float,
    rho_l: Array | float,
    rho_v: Array | float,
    sigma: Array | float,
    Cp_l: Array | float,
    Pr_l: Array | float,
    C_sf: Array | float = 0.013,
    n: float = 1.0,
) -> Array:
    """Excess temperature (K) giving heat flux ``q`` -- the closed-form inverse of
    :func:`rohsenow_heat_flux`.

    The boiling coefficient is then ``h = q / dT_e`` (Rohsenow: ``h ~ q^(2/3)``).

    Args:
        q: Heat flux (W/m^2).
        mu_l, h_fg, rho_l, rho_v, sigma, Cp_l, Pr_l, C_sf, n: As in
            :func:`rohsenow_heat_flux`.

    Returns:
        Excess temperature ``T_s - T_sat`` (K).
    """
    cap = GRAVITY * (_f64(rho_l) - _f64(rho_v)) / _f64(sigma)
    base = jnp.maximum(_f64(q), 1e-300) / (_f64(mu_l) * _f64(h_fg) * jnp.sqrt(cap))
    return _f64(C_sf) * _f64(h_fg) * _f64(Pr_l) ** n / _f64(Cp_l) * base ** (1.0 / 3.0)


def mostinski(q: Array | float, P_c: Array | float, P: Array | float) -> Array:
    """Mostinski (1963) nucleate pool-boiling coefficient from reduced pressure (W/m^2/K).

    ``h = 0.00417 P_c^0.69 q^0.7 F_p``, ``F_p = 1.8 p_r^0.17 + 4 p_r^1.2 + 10 p_r^10``,
    ``p_r = P / P_c``, with ``P_c`` in **kPa** and ``q`` in W/m^2. Needs only the
    critical pressure; accuracy is typically +-30% or worse. The constants are
    **recalled from memory of the standard form (Mostinski; Perry's / Coulson &
    Richardson) and were not checked against the source**; the tests verify
    only the arithmetic and agreement with Rohsenow for water to within a
    factor stated there.

    Args:
        q: Heat flux (W/m^2).
        P_c: Critical pressure (kPa).
        P: Operating pressure (kPa).

    Returns:
        Boiling coefficient (W/m^2/K).
    """
    pr = _f64(P) / _f64(P_c)
    Fp = 1.8 * pr**0.17 + 4.0 * pr**1.2 + 10.0 * pr**10
    return 0.00417 * _f64(P_c) ** 0.69 * jnp.maximum(_f64(q), 1e-300) ** 0.7 * Fp


def critical_heat_flux(
    h_fg: Array | float,
    rho_v: Array | float,
    rho_l: Array | float,
    sigma: Array | float,
    C: float = math.pi / 24.0,
) -> Array:
    """Zuber critical (peak) pool-boiling heat flux (W/m^2).

    ``q''_max = C h_fg rho_v^(1/2) [sigma g (rho_l - rho_v)]^(1/4)`` with
    ``C = pi/24 = 0.131`` (the usual value for large flat plates; Zuber's
    original constant was 0.149 and some texts quote that instead, so ``C``
    is a parameter).

    Args:
        h_fg: Latent heat (J/kg).
        rho_v: Vapour density (kg/m^3).
        rho_l: Liquid density (kg/m^3).
        sigma: Surface tension (N/m).
        C: Zuber constant.

    Returns:
        Critical heat flux (W/m^2).
    """
    return (
        C
        * _f64(h_fg)
        * jnp.sqrt(_f64(rho_v))
        * (_f64(sigma) * GRAVITY * (_f64(rho_l) - _f64(rho_v))) ** 0.25
    )


# =============================================================================
# Overall coefficient
# =============================================================================


def overall_U(
    h_i: Array | float,
    h_o: Array | float,
    D_i: Array | float,
    D_o: Array | float,
    k_wall: Array | float,
    R_fi: Array | float = 0.0,
    R_fo: Array | float = 0.0,
    basis: str = "outer",
) -> Array:
    """Overall coefficient of a tube wall from film, wall and fouling resistances.

    Referenced to the outer area (``basis="outer"``)::

        1/U_o = 1/h_o + R_fo + D_o ln(D_o/D_i) / (2 k) + R_fi D_o/D_i + D_o / (D_i h_i)

    and to the inner area (``"inner"``)::

        1/U_i = 1/h_i + R_fi + D_i ln(D_o/D_i) / (2 k) + (R_fo + 1/h_o) D_i/D_o

    so that ``U_o A_o = U_i A_i = UA``. (Incropera & DeWitt Eqs. 11.1-11.5.)
    ``k_wall = inf`` removes the wall resistance; ``D_i = D_o`` is allowed.

    Args:
        h_i: Inside film coefficient (W/m^2/K).
        h_o: Outside film coefficient (W/m^2/K).
        D_i: Tube inside diameter (m).
        D_o: Tube outside diameter (m).
        k_wall: Wall conductivity (W/m/K).
        R_fi: Inside fouling resistance, per inside area (m^2 K/W).
        R_fo: Outside fouling resistance, per outside area (m^2 K/W).
        basis: ``"outer"`` or ``"inner"`` reference area.

    Returns:
        Overall coefficient (W/m^2/K) on the chosen area basis.

    Raises:
        ValueError: Unknown ``basis``.

    Example:
        >>> round(float(overall_U(1000.0, 1000.0, 0.02, 0.02, 400.0)), 1)
        500.0
    """
    if basis not in ("outer", "inner"):
        raise ValueError(f"basis must be 'outer' or 'inner', got {basis!r}")
    h_i, h_o, D_i, D_o = _f64(h_i), _f64(h_o), _f64(D_i), _f64(D_o)
    wall = jnp.log(D_o / D_i) / (2.0 * _f64(k_wall))
    if basis == "outer":
        inv = 1.0 / h_o + _f64(R_fo) + D_o * wall + _f64(R_fi) * D_o / D_i + D_o / (D_i * h_i)
    else:
        inv = 1.0 / h_i + _f64(R_fi) + D_i * wall + (_f64(R_fo) + 1.0 / h_o) * D_i / D_o
    return 1.0 / inv


#: Typical fouling resistances ``R_f`` (m^2 K/W). **Transcribed from memory of
#: Incropera & DeWitt Table 11.1 (which cites the TEMA standards) and NOT
#: re-checked against the book or TEMA**; treat as order-of-magnitude placeholders
#: and replace with the values of your own standard for design work.
FOULING_RESISTANCES: dict[str, float] = {
    "seawater_below_50C": 0.0001,
    "seawater_above_50C": 0.0002,
    "treated_boiler_feedwater_above_50C": 0.0002,
    "fuel_oil": 0.0009,
    "refrigerating_liquids": 0.0002,
    "steam_oil_free": 0.0001,
}

#: Typical overall U ranges ``(low, high)`` in W/m^2/K by service. **Transcribed
#: from memory of Incropera & DeWitt Table 11.2 and NOT re-checked**; use only
#: as a sanity band for a computed U.
TYPICAL_U_RANGES: dict[str, tuple[float, float]] = {
    "water_to_water": (850.0, 1700.0),
    "water_to_oil": (100.0, 350.0),
    "steam_condenser_water_in_tubes": (1000.0, 6000.0),
    "ammonia_condenser_water_in_tubes": (800.0, 1400.0),
    "alcohol_condenser_water_in_tubes": (250.0, 700.0),
    "finned_tube_water_in_tubes_air_crossflow": (25.0, 50.0),
}


def typical_U_range(service: str) -> tuple[float, float]:
    """Typical overall ``(U_low, U_high)`` (W/m^2/K) for a service in :data:`TYPICAL_U_RANGES`.

    Args:
        service: Key of :data:`TYPICAL_U_RANGES`.

    Returns:
        ``(low, high)``.

    Raises:
        KeyError: Unknown service (the message lists the keys).
    """
    try:
        return TYPICAL_U_RANGES[service]
    except KeyError:
        raise KeyError(f"Unknown service {service!r}; use one of {sorted(TYPICAL_U_RANGES)}") from None


__all__ = [
    "GRAVITY",
    "NU_LAMINAR_TW",
    "NU_LAMINAR_QS",
    "RE_LAMINAR",
    "RE_TURBULENT",
    "FOULING_RESISTANCES",
    "TYPICAL_U_RANGES",
    "reynolds",
    "prandtl",
    "nusselt_to_h",
    "grashof",
    "rayleigh",
    "slab_resistance",
    "cylinder_resistance",
    "sphere_resistance",
    "composite_wall",
    "critical_insulation_radius",
    "fin_efficiency_straight",
    "sieder_tate_laminar",
    "dittus_boelter",
    "sieder_tate",
    "gnielinski",
    "internal_nusselt",
    "internal_h",
    "churchill_bernstein",
    "flat_plate_nusselt",
    "churchill_chu_vertical_plate",
    "churchill_chu_horizontal_cylinder",
    "nusselt_film_condensation",
    "rohsenow_heat_flux",
    "rohsenow_superheat",
    "mostinski",
    "critical_heat_flux",
    "overall_U",
    "typical_U_range",
]
