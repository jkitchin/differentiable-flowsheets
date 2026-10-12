"""Heat-transfer correlations: film coefficients, phase change and overall U.

Pure functions, JAX, SI units, differentiable: ``jax.grad`` of an overall
coefficient with respect to the tube velocity, the diameter or any property
works, including through the laminar-transition-turbulent blend. Nothing here
is a palette unit; the heat exchangers (:mod:`difflow.units.heat_exchanger`)
still take ``U`` / ``UA`` as numbers, and these functions compute the number.

Liquid and gas properties are *inputs*: every function takes ``k``, ``mu``,
``Cp`` and ``rho`` explicitly, as :mod:`difflow.fluids` does. Look them up
with :mod:`difflow.liquid_properties` or :mod:`difflow.gas_properties`.

Contents
--------
* Dimensionless groups: :func:`reynolds`, :func:`prandtl`, :func:`nusselt_to_h`,
  :func:`grashof`, :func:`rayleigh`
* Conduction: :func:`slab_resistance`, :func:`cylinder_resistance`,
  :func:`sphere_resistance`, :func:`composite_wall`,
  :func:`critical_insulation_radius`, :func:`insulated_pipe_heat_loss`,
  :func:`fin_efficiency_straight` (insulation cost and the economic
  thickness: :mod:`difflow.economics.insulation`)
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
* Annulus: :func:`annulus_nusselt` (laminar, Incropera Table 8.2)
* Overall coefficient: :func:`overall_U`, :data:`FOULING_RESISTANCES`,
  :data:`FOULING_RESISTANCE_RANGES`, :data:`TYPICAL_U_RANGES`,
  :func:`typical_U_range`
* Radiation: :data:`STEFAN_BOLTZMANN`, :func:`blackbody_emissive_power`,
  :func:`radiation_h`, :func:`net_radiation`, :func:`two_surface_radiation`
* Lumped-capacitance transients: :func:`biot_number`,
  :func:`lumped_time_constant`, :func:`lumped_temperature`,
  :func:`lumped_heating_time`
* Agitated vessels: :data:`AGITATED_VESSEL_CONSTANTS`, :func:`agitator_reynolds`,
  :func:`agitated_vessel_nusselt`, :func:`jacketed_vessel_h` (jacketed wall),
  :func:`agitated_coil_h` (immersed coil)

Gas properties for the dimensionless groups are in :mod:`difflow.gas_properties`
(air, N2, O2, CO2, CH4, H2, steam, ...) and liquid ones in
:mod:`difflow.liquid_properties`.

Not included (out of scope or optional, see ``docs``): annular fins, tube banks
(Zukauskas), the Bell-Delaware shell-side method (Kern is in :mod:`difflow.shell_and_tube`),
multi-dimensional transient conduction (Heisler charts).

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
#: Same, uniform wall heat flux (Incropera & DeWitt Eq. 8.53): exactly 48/11
#: = 4.3636 (the book rounds it to 4.36).
NU_LAMINAR_QS = 48.0 / 11.0

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


def insulated_pipe_heat_loss(
    D_o: Array | float,
    thickness: Array | float,
    k_ins: Array | float,
    h_out: Array | float,
    T_pipe: Array | float,
    T_amb: Array | float,
) -> Array:
    r"""Heat loss per metre (W/m) of a pipe with a layer of insulation.

    .. math:: q' = \frac{T_{pipe} - T_{amb}}{\dfrac{\ln[(D_o+2t)/D_o]}{2\pi k_{ins}}
              + \dfrac{1}{h_{out}\,\pi (D_o + 2t)}}

    The pipe's outer surface is taken at ``T_pipe`` (the inside film and the
    steel wall are small next to the insulation); ``h_out`` is the combined
    convective and radiative coefficient of the outer surface (cladding).
    ``thickness = 0`` is the bare pipe. Below the critical radius
    (:func:`critical_insulation_radius`) adding insulation increases the loss.

    Args:
        D_o: Pipe outside diameter (m).
        thickness: Insulation thickness (m).
        k_ins: Insulation conductivity (W/m/K).
        h_out: Outer surface coefficient (W/m^2/K).
        T_pipe: Pipe outer-surface temperature (K).
        T_amb: Ambient temperature (K).

    Returns:
        q' (W per metre of pipe).

    Example:
        >>> from difflow.heat_transfer import insulated_pipe_heat_loss
        >>> round(float(insulated_pipe_heat_loss(0.1143, 0.05, 0.04, 10.0, 450.0, 293.15)), 1)
        59.2
    """
    D_o, t = _f64(D_o), _f64(thickness)
    r_in = D_o / 2.0
    r_out = r_in + t
    R = cylinder_resistance(r_in, r_out, k_ins, 1.0) + 1.0 / (_f64(h_out) * 2.0 * jnp.pi * r_out)
    return (_f64(T_pipe) - _f64(T_amb)) / R


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

#: Exponents ``n`` of the optional property-ratio factor ``(mu_b/mu_w)^n`` on the
#: Gnielinski and Dittus-Boelter turbulent Nusselt numbers (liquids): 0.11 when
#: the fluid is heated, 0.25 when it is cooled (the Petukhov form; the VDI Heat
#: Atlas / Gnielinski form is ``(Pr/Pr_w)^0.11``). **FROM MEMORY, UNVERIFIED**:
#: written from recollection of the published form, not checked against
#: Petukhov, Gnielinski or the VDI Heat Atlas. Check them before a real design;
#: that is why the factor is opt-in (``viscosity_correction=True``). The factor
#: uses the viscosity ratio as a stand-in for ``Pr/Pr_w`` (``Cp`` and ``k``
#: taken constant across the film).
VISCOSITY_EXPONENT_HEATING = 0.11
VISCOSITY_EXPONENT_COOLING = 0.25


def internal_nusselt(
    Re: Array | float,
    Pr: Array | float,
    D_over_L: Array | float | None = None,
    mu_ratio: Array | float = 1.0,
    rel_roughness: Array | float = 0.0,
    method: str = "gnielinski",
    heating: bool | Array = True,
    boundary: str = "T",
    viscosity_correction: bool = False,
) -> Array:
    """Nusselt number in a circular tube, laminar through turbulent, smooth in Re.

    * Laminar (``Re <= 2300``): fully developed ``Nu = 3.66`` (``boundary="T"``,
      constant wall temperature) or ``48/11 = 4.364`` (``"q"``, constant flux). If
      ``D_over_L`` is given, the entry-region
      :func:`sieder_tate_laminar` is joined by a smooth asymptotic combination
      ``Nu = (Nu_fd^3 + Nu_entry^3)^(1/3)`` (Churchill-Usagi; *not* a published
      correlation -- it just enforces the fully developed floor and the
      Sieder-Tate entry behaviour).
    * Turbulent (``Re >= 4000``): ``method`` is ``"gnielinski"`` (friction factor
      from :func:`difflow.fluids.friction_factor`, Colebrook), ``"dittus_boelter"``
      or ``"sieder_tate"`` (``mu_ratio`` applies to the latter).
    * Optional viscosity correction (``viscosity_correction=True``): the
      Gnielinski and Dittus-Boelter turbulent values are multiplied by
      ``mu_ratio^n``, ``n = 0.11`` (``heating``) or ``0.25`` (cooling), the
      liquid form of Petukhov (Gnielinski / VDI: ``(Pr/Pr_w)^0.11``).
      **The exponents are from memory and unverified**
      (:data:`VISCOSITY_EXPONENT_HEATING`); the factor is off by default so
      that no result changes silently. ``"sieder_tate"`` already carries its
      own 0.14 factor and gets no second one. ``mu_ratio**n`` is smooth, so the
      correction is differentiable.
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
        mu_ratio: Bulk over wall viscosity ``mu_b / mu_w`` (-). Used by the
            laminar entry term and by ``method="sieder_tate"``; Gnielinski and
            Dittus-Boelter, as published, have no viscosity correction and
            ignore it unless ``viscosity_correction`` is True.
        rel_roughness: Relative roughness for the Gnielinski friction factor.
        method: Turbulent correlation (Python string).
        heating: Dittus-Boelter exponent selector (True: heating).
        boundary: ``"T"`` or ``"q"`` laminar fully developed value.
        viscosity_correction: True to apply ``mu_ratio^n`` to the Gnielinski /
            Dittus-Boelter turbulent value (exponents unverified, see above).

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

    if viscosity_correction and method != "sieder_tate":
        n = jnp.where(jnp.asarray(heating), VISCOSITY_EXPONENT_HEATING, VISCOSITY_EXPONENT_COOLING)
        Nu_turb = Nu_turb * _f64(mu_ratio) ** n

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
    viscosity_correction: bool = False,
) -> Array:
    """Tube-side film coefficient ``h = Nu k / D`` (W/m^2/K); see :func:`internal_nusselt`.

    Args:
        Re: Reynolds number ``rho v D / mu`` (-).
        Pr: Prandtl number ``Cp mu / k`` (-).
        k: Fluid thermal conductivity (W/m/K).
        D: Tube inside diameter (m).
        L: Tube length (m) for the laminar entry effect, or None.
        mu_ratio: Bulk over wall viscosity (-); ignored by Gnielinski and
            Dittus-Boelter (see :func:`internal_nusselt`).
        rel_roughness: Relative roughness ``eps / D`` (-).
        method: ``"gnielinski"``, ``"dittus_boelter"`` or ``"sieder_tate"``.
        heating: True if the fluid is being heated.
        boundary: ``"T"`` (constant wall temperature) or ``"q"`` (constant flux).
        viscosity_correction: Apply the optional ``mu_ratio^n`` factor to the
            Gnielinski / Dittus-Boelter value (unverified exponents).

    Returns:
        Heat-transfer coefficient (W/m^2/K).
    """
    D_over_L = None if L is None else _f64(D) / _f64(L)
    Nu = internal_nusselt(Re, Pr, D_over_L, mu_ratio, rel_roughness, method, heating, boundary,
                          viscosity_correction)
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

    **Single-surface value only.** Tube bundles (kettle reboilers) have a
    lower critical flux than this because vapour from the lower tubes blankets
    the upper ones. The published bundle factor (Palen's) is NOT implemented:
    its form and constants could not be checked against a source, so none was
    invented. Treat this as an upper bound for a bundle and derate by a
    sourced bundle correction before a real reboiler design.

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


#: Typical fouling resistances ``R_f`` (m^2 K/W), one value per service.
#:
#: * The first six entries plus ``treated_boiler_feedwater_below_50C`` and
#:   ``river_water_below_50C`` are Incropera & DeWitt, *Fundamentals of Heat
#:   and Mass Transfer*, Table 11.1 (from TEMA). The six were checked against
#:   the table while the unit-operations book was written.
#: * The rest are Sinnott, *Coulson & Richardson's Chemical Engineering*
#:   Vol. 6, 4th ed., Table 12.2. That table lists fouling
#:   *coefficients* as ranges; the value here is the larger (conservative)
#:   resistance of the range, and :data:`FOULING_RESISTANCE_RANGES` has both
#:   ends. (Its "steam, oil-free" row prints 0.0025; the coefficient,
#:   4000-10000 W/m^2/K, makes it 0.00025.)
#:
#: Crude oil by temperature and velocity, lube oils, cooling waters by
#: temperature and velocity and the other TEMA RGP-T-2.4 entries are in
#: :mod:`difflow.fouling` (:data:`difflow.fouling.TEMA_FOULING`,
#: :func:`difflow.fouling.tema_crude_oil_fouling`,
#: :func:`difflow.fouling.tema_water_fouling`), with their own provenance
#: note. Use your own standard for design.
FOULING_RESISTANCES: dict[str, float] = {
    # Incropera & DeWitt Table 11.1
    "seawater_below_50C": 0.0001,
    "seawater_above_50C": 0.0002,
    "treated_boiler_feedwater_below_50C": 0.0001,
    "treated_boiler_feedwater_above_50C": 0.0002,
    "river_water_below_50C": 0.001,
    "fuel_oil": 0.0009,
    "refrigerating_liquids": 0.0002,
    "steam_oil_free": 0.0001,
    # Sinnott (Coulson & Richardson Vol. 6) Table 12.2, larger end of the range
    "river_water": 0.0003,
    "sea_water": 0.001,
    "cooling_tower_water": 0.0003,
    "town_water_soft": 0.0003,
    "town_water_hard": 0.001,
    "steam_condensate": 0.00067,
    "steam_oil_traces": 0.0005,
    "refrigerated_brine": 0.0003,
    "aqueous_salt_solutions": 0.0003,
    "air_and_industrial_gases": 0.0002,
    "flue_gases": 0.0005,
    "organic_vapours": 0.0002,
    "organic_liquids": 0.0002,
    "light_hydrocarbons": 0.0002,
    "heavy_hydrocarbons": 0.0005,
    "boiling_organics": 0.0004,
    "condensing_organics": 0.0002,
    "heat_transfer_fluids": 0.0002,
}

#: ``(low, high)`` fouling resistance (m^2 K/W) where the source gives a range
#: (Incropera Table 11.1 river water; Sinnott Table 12.2). Single-valued
#: entries of :data:`FOULING_RESISTANCES` have ``low == high``.
FOULING_RESISTANCE_RANGES: dict[str, tuple[float, float]] = {
    **{k: (v, v) for k, v in FOULING_RESISTANCES.items()},
    "river_water_below_50C": (0.0002, 0.001),
    "river_water": (0.0001, 0.0003),
    "sea_water": (0.0003, 0.001),
    "cooling_tower_water": (0.00017, 0.0003),
    "town_water_soft": (0.0002, 0.0003),
    "town_water_hard": (0.0005, 0.001),
    "steam_condensate": (0.0002, 0.00067),
    "steam_oil_traces": (0.0002, 0.0005),
    "refrigerated_brine": (0.0002, 0.0003),
    "aqueous_salt_solutions": (0.0002, 0.0003),
    "air_and_industrial_gases": (0.0001, 0.0002),
    "flue_gases": (0.0002, 0.0005),
}

#: Source of each :data:`FOULING_RESISTANCES` entry.
FOULING_SOURCES: dict[str, str] = {
    k: ("Incropera & DeWitt, Fundamentals of Heat and Mass Transfer, Table 11.1"
        if i < 8 else
        "Sinnott, Coulson & Richardson's Chemical Engineering Vol. 6, 4th ed., Table 12.2")
    for i, k in enumerate(FOULING_RESISTANCES)
}

#: Typical overall U ranges ``(low, high)`` in W/m^2/K by service, from
#: Incropera & DeWitt Table 11.2 (checked against the table while the
#: unit-operations book was written; water-to-oil corrected from 100 to 110).
#: Use only as a sanity band for a computed U.
TYPICAL_U_RANGES: dict[str, tuple[float, float]] = {
    "water_to_water": (850.0, 1700.0),
    "water_to_oil": (110.0, 350.0),
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


# ----------------------------------------------------------------------
# Laminar annulus
# ----------------------------------------------------------------------
#: Incropera & DeWitt Table 8.2: fully developed laminar flow in a concentric
#: annulus with one surface insulated and the other at constant temperature
#: (after Kays & Perkins). Columns ``Di/Do``, ``Nu_i`` (inner surface heated),
#: ``Nu_o`` (outer surface heated); Nu based on ``D_h = Do - Di``. The values
#: were reproduced independently, to +-0.01, by a finite-difference solution
#: of the Graetz eigenproblem (``tests/test_heat_transfer.py::TestAnnulus``).
ANNULUS_NUSSELT_TABLE = (
    (0.0, float("nan"), 3.66),
    (0.05, 17.46, 4.06),
    (0.10, 11.56, 4.11),
    (0.25, 7.37, 4.23),
    (0.50, 5.74, 4.43),
    (1.00, 4.86, 4.86),
)


def annulus_nusselt(Di_over_Do: Array | float, surface: str = "inner") -> Array:
    """Fully developed laminar Nusselt number in a concentric annulus.

    Linear interpolation in Incropera & DeWitt Table 8.2 (one surface at
    constant temperature, the other insulated), as Incropera's Example 11.1
    does (``Nu_i = 5.63`` at ``Di/Do = 0.56``). Use it with the hydraulic
    diameter ``D_h = Do - Di``: ``h = Nu k / D_h``.

    For **turbulent** flow in an annulus, use the tube correlations
    (:func:`internal_nusselt`, Gnielinski or Dittus-Boelter) with
    ``Re`` and ``Nu`` based on ``D_h`` (Incropera Sec. 8.7). Laminar flow is
    where the annulus needs its own numbers.

    Args:
        Di_over_Do: Inner over outer diameter, 0-1 (``"inner"``: 0.05-1;
            below 0.05 the inner value is held at 17.46, and it diverges as
            Di/Do -> 0).
        surface: ``"inner"`` (inner tube heated or cooled, outer insulated:
            a double-pipe exchanger's annulus) or ``"outer"``.

    Returns:
        Nu based on ``D_h`` (-). Piecewise linear in ``Di/Do``, so the
        gradient jumps at the table points.

    Example:
        >>> round(float(annulus_nusselt(0.56, "inner")), 2)   # Incropera Ex 11.1
        5.63
    """
    if surface not in ("inner", "outer"):
        raise ValueError(f"surface must be 'inner' or 'outer', got {surface!r}")
    r = _f64(Di_over_Do)
    tab = ANNULUS_NUSSELT_TABLE if surface == "outer" else ANNULUS_NUSSELT_TABLE[1:]
    xs = jnp.asarray([row[0] for row in tab])
    ys = jnp.asarray([row[1] if surface == "inner" else row[2] for row in tab])
    return jnp.interp(r, xs, ys)


# ----------------------------------------------------------------------
# Radiation
# ----------------------------------------------------------------------
#: Stefan-Boltzmann constant (W/m^2/K^4), CODATA 2018 (exact in the 2019 SI).
STEFAN_BOLTZMANN = 5.670374419e-8


def blackbody_emissive_power(T: Array | float) -> Array:
    """Blackbody emissive power ``E_b = sigma T^4`` (W/m^2).

    Args:
        T: Absolute temperature (K).

    Returns:
        E_b (W/m^2).
    """
    return STEFAN_BOLTZMANN * _f64(T) ** 4


def radiation_h(Ts: Array | float, Tsur: Array | float, emissivity: Array | float) -> Array:
    """Linearized radiation coefficient ``h_r = eps sigma (Ts + Tsur)(Ts^2 + Tsur^2)``.

    For a small gray surface in large isothermal surroundings,
    ``q'' = h_r (Ts - Tsur)`` exactly (Incropera & DeWitt Eq. 1.9), so ``h_r``
    adds in parallel with a convection coefficient.

    Args:
        Ts: Surface temperature (K).
        Tsur: Surroundings temperature (K).
        emissivity: Surface emissivity (0-1).

    Returns:
        h_r (W/m^2/K).

    Example:
        >>> round(float(radiation_h(438.15, 296.15, 0.85)), 1)
        9.9
    """
    Ts, Tsur = _f64(Ts), _f64(Tsur)
    return _f64(emissivity) * STEFAN_BOLTZMANN * (Ts + Tsur) * (Ts**2 + Tsur**2)


def net_radiation(
    Ts: Array | float, Tsur: Array | float, emissivity: Array | float, area: Array | float = 1.0
) -> Array:
    """Net radiation from a small gray body to large surroundings (W).

    ``q = eps A sigma (Ts^4 - Tsur^4)`` (Incropera & DeWitt Eq. 1.7): the
    surroundings are large and isothermal (so their emissivity drops out) and
    the body does not see itself (convex surface).

    Args:
        Ts: Surface temperature (K).
        Tsur: Surroundings temperature (K).
        emissivity: Surface emissivity (0-1).
        area: Radiating area (m^2); per unit length for a cylinder, ``pi D``.

    Returns:
        q (W, or W/m with a per-length area); positive when the surface loses heat.

    Example:
        >>> import math
        >>> round(float(net_radiation(438.15, 296.15, 0.85, math.pi * 0.1)))  # Incropera Ex 9.4
        442
    """
    return _f64(emissivity) * _f64(area) * STEFAN_BOLTZMANN * (_f64(Ts) ** 4 - _f64(Tsur) ** 4)


def two_surface_radiation(
    T1: Array | float,
    T2: Array | float,
    eps1: Array | float,
    eps2: Array | float,
    A1: Array | float,
    A2: Array | float | None = None,
    F12: Array | float = 1.0,
) -> Array:
    """Net radiation exchange in a two-surface gray, diffuse enclosure (W).

    ``q12 = sigma (T1^4 - T2^4) / [(1 - eps1)/(eps1 A1) + 1/(A1 F12)
    + (1 - eps2)/(eps2 A2)]`` (Incropera & DeWitt Sec. 13.3, two-surface enclosure). Special cases:

    * large parallel plates: ``A1 = A2 = A``, ``F12 = 1``;
    * long concentric cylinders or concentric spheres: ``F12 = 1``,
      ``A2 = A1 r2/r1`` (cylinders) or ``A1 (r2/r1)^2`` (spheres);
    * small convex body in a large cavity: ``A2=None`` (``A2 -> inf``),
      ``F12 = 1``, which reduces to :func:`net_radiation`.

    Args:
        T1, T2: Surface temperatures (K).
        eps1, eps2: Emissivities (0-1).
        A1: Area of surface 1 (m^2).
        A2: Area of surface 2 (m^2); None for an enclosing surface much larger than A1.
        F12: View factor from 1 to 2.

    Returns:
        q12 (W), positive from 1 to 2.
    """
    A1, eps1, eps2 = _f64(A1), _f64(eps1), _f64(eps2)
    R = (1.0 - eps1) / (eps1 * A1) + 1.0 / (A1 * _f64(F12))
    if A2 is not None:
        R = R + (1.0 - eps2) / (eps2 * _f64(A2))
    return STEFAN_BOLTZMANN * (_f64(T1) ** 4 - _f64(T2) ** 4) / R


# ----------------------------------------------------------------------
# Agitated vessels: jacket (vessel wall) and immersed-coil coefficients
# ----------------------------------------------------------------------
#: Constants of the vessel-side (process-side) coefficient at the wall of a
#: jacketed, agitated vessel,
#: ``Nu = h D_T / k = a Re^b Pr^(1/3) (mu/mu_w)^m`` with
#: ``Re = rho N D_a^2 / mu`` (``N`` in rev/s, ``D_a`` the impeller diameter,
#: ``D_T`` the tank diameter). Each entry: ``(a, b, m, (Re_lo, Re_hi), source)``.
#:
#: * ``paddle`` is Chilton, Drew & Jebens (1944), as given by Kern (1950,
#:   Ch. 20); the constants (0.36, 2/3, 0.14) were not re-checked against
#:   the paper, but are the form reproduced throughout the textbooks.
#: * ``flat_blade_turbine`` (baffled) and the two ``anchor`` rows are Uhl's
#:   (1955) constants as tabulated in Uhl & Gray, *Mixing* Vol. I (1966) and
#:   reproduced in Perry's handbook; **recalled from those secondary
#:   tabulations and not checked against a printed copy**. Geometry
#:   corrections (``D_a/D_T``, blade width, clearance) of the fuller
#:   correlations are not included.
#:
#: Pass ``a``, ``b``, ``m`` to :func:`agitated_vessel_nusselt` to use any
#: other published set.
AGITATED_VESSEL_CONSTANTS: dict[str, tuple[float, float, float, tuple[float, float], str]] = {
    "paddle": (0.36, 2.0 / 3.0, 0.14, (300.0, 3.0e5),
               "Chilton, Drew & Jebens, Ind. Eng. Chem. 36, 510 (1944); Kern (1950) Ch. 20"),
    "flat_blade_turbine": (0.74, 0.67, 0.14, (200.0, 1.0e5),
                           "Uhl (1955), baffled vessel, via Uhl & Gray (1966) / Perry's (recalled)"),
    "anchor": (0.36, 2.0 / 3.0, 0.18, (300.0, 4.0e4),
               "Uhl (1955), via Uhl & Gray (1966) / Perry's (recalled)"),
    "anchor_low_re": (1.0, 0.5, 0.18, (30.0, 300.0),
                      "Uhl (1955), via Uhl & Gray (1966) / Perry's (recalled)"),
}

#: Methods of :func:`agitated_coil_h`.
AGITATED_COIL_METHODS = ("chilton_drew_jebens", "oldshue_gretton")


def agitator_reynolds(rho: Array | float, N: Array | float, D_a: Array | float, mu: Array | float) -> Array:
    """Impeller Reynolds number ``Re = rho N D_a^2 / mu`` (-).

    Args:
        rho: Liquid density (kg/m^3).
        N: Impeller speed (rev/s, **not** rad/s).
        D_a: Impeller diameter (m).
        mu: Bulk liquid viscosity (Pa s).

    Returns:
        Re (-).
    """
    return _f64(rho) * _f64(N) * _f64(D_a) ** 2 / _f64(mu)


def agitated_vessel_nusselt(
    Re: Array | float,
    Pr: Array | float,
    mu_ratio: Array | float = 1.0,
    impeller: str = "paddle",
    a: float | None = None,
    b: float | None = None,
    m: float | None = None,
) -> Array:
    r"""Vessel-side Nusselt number at the jacketed wall of an agitated vessel.

    .. math:: \mathrm{Nu} = \frac{h D_T}{k}
              = a\,\mathrm{Re}^b\,\mathrm{Pr}^{1/3}\,(\mu/\mu_w)^m,
              \qquad \mathrm{Re} = \frac{\rho N D_a^2}{\mu}

    with the constants of :data:`AGITATED_VESSEL_CONSTANTS` for ``impeller``
    (Chilton-Drew-Jebens paddle: 0.36, 2/3, 0.14). ``Nu`` is based on the
    **tank** diameter and ``Re`` on the **impeller** diameter. The fit's
    Reynolds range is in the table; nothing is clipped outside it.

    Args:
        Re: Impeller Reynolds number (see :func:`agitator_reynolds`).
        Pr: Prandtl number of the bulk liquid.
        mu_ratio: ``mu_bulk / mu_wall`` (-).
        impeller: Key of :data:`AGITATED_VESSEL_CONSTANTS`.
        a, b, m: Override the tabulated constants (each optional).

    Returns:
        Nu (-), on the tank diameter.

    Raises:
        KeyError: Unknown impeller.

    Example:
        >>> from difflow.heat_transfer import agitated_vessel_nusselt
        >>> round(float(agitated_vessel_nusselt(1.0e5, 5.0)), 1)   # 0.36 * 1e5^(2/3) * 5^(1/3)
        1326.3
    """
    try:
        a0, b0, m0, _, _ = AGITATED_VESSEL_CONSTANTS[impeller]
    except KeyError:
        raise KeyError(f"Unknown impeller {impeller!r}; use one of {sorted(AGITATED_VESSEL_CONSTANTS)}") from None
    a = a0 if a is None else a
    b = b0 if b is None else b
    m = m0 if m is None else m
    return a * _f64(Re) ** b * _f64(Pr) ** (1.0 / 3.0) * _f64(mu_ratio) ** m


def jacketed_vessel_h(
    rho: Array | float,
    N: Array | float,
    D_a: Array | float,
    D_T: Array | float,
    mu: Array | float,
    Cp: Array | float,
    k: Array | float,
    mu_wall: Array | float | None = None,
    impeller: str = "paddle",
) -> Array:
    """Vessel-side film coefficient (W/m^2/K) at the jacketed wall of an agitated vessel.

    ``h = Nu k / D_T`` with :func:`agitated_vessel_nusselt`. Combine with the
    jacket-side coefficient, the wall and fouling in :func:`overall_U`.

    Args:
        rho: Liquid density (kg/m^3).
        N: Impeller speed (rev/s).
        D_a: Impeller diameter (m).
        D_T: Tank diameter (m).
        mu: Bulk viscosity (Pa s).
        Cp: Heat capacity (J/kg/K).
        k: Thermal conductivity (W/m/K).
        mu_wall: Viscosity at the wall temperature (Pa s); None for ``mu/mu_w = 1``.
        impeller: Key of :data:`AGITATED_VESSEL_CONSTANTS`.

    Returns:
        h (W/m^2/K).

    Example:
        >>> from difflow.heat_transfer import jacketed_vessel_h
        >>> # water-like liquid, 1.5 m tank, 0.5 m paddle at 2 rev/s
        >>> round(float(jacketed_vessel_h(1000.0, 2.0, 0.5, 1.5, 1.0e-3, 4180.0, 0.6)))
        1733
    """
    Re = agitator_reynolds(rho, N, D_a, mu)
    Pr = prandtl(Cp, mu, k)
    ratio = 1.0 if mu_wall is None else _f64(mu) / _f64(mu_wall)
    return agitated_vessel_nusselt(Re, Pr, ratio, impeller) * _f64(k) / _f64(D_T)


def agitated_coil_h(
    Re: Array | float,
    Pr: Array | float,
    k: Array | float,
    D_T: Array | float,
    mu_ratio: Array | float = 1.0,
    method: str = "chilton_drew_jebens",
    D_a: Array | float | None = None,
    d_o: Array | float | None = None,
) -> Array:
    r"""Vessel-side coefficient (W/m^2/K) on the outside of an immersed helical coil.

    * ``"chilton_drew_jebens"`` (paddle agitator; Chilton, Drew & Jebens,
      1944, as given by Kern, 1950, Ch. 20), on the tank diameter:

      .. math:: \frac{h D_T}{k} = 0.87\,\mathrm{Re}^{0.62}\mathrm{Pr}^{1/3}(\mu/\mu_w)^{0.14}

    * ``"oldshue_gretton"`` (flat-blade turbine, baffled vessel; Oldshue &
      Gretton, Chem. Eng. Prog. 50, 615, 1954), on the coil-tube outside
      diameter :math:`d_o`:

      .. math:: \frac{h d_o}{k} = 0.17\,\mathrm{Re}^{0.67}\mathrm{Pr}^{0.37}
                (D_a/D_T)^{0.1}(d_o/D_T)^{0.5}

      (no viscosity-ratio term; ``mu_ratio`` is ignored).

    ``Re = rho N D_a^2 / mu`` in both (:func:`agitator_reynolds`).

    Args:
        Re: Impeller Reynolds number (-).
        Pr: Prandtl number (-).
        k: Liquid conductivity (W/m/K).
        D_T: Tank diameter (m).
        mu_ratio: ``mu_bulk / mu_wall`` (Chilton-Drew-Jebens only).
        method: One of :data:`AGITATED_COIL_METHODS`.
        D_a: Impeller diameter (m), Oldshue-Gretton only.
        d_o: Coil tube outside diameter (m), Oldshue-Gretton only.

    Returns:
        h (W/m^2/K) on the coil's outside surface.

    Raises:
        ValueError: Unknown method, or ``D_a``/``d_o`` missing for Oldshue-Gretton.

    Example:
        >>> from difflow.heat_transfer import agitated_coil_h
        >>> round(float(agitated_coil_h(1.0e5, 5.0, 0.6, 1.5)))   # 0.87 Re^0.62 Pr^(1/3) k/D_T
        749
    """
    if method == "chilton_drew_jebens":
        Nu = 0.87 * _f64(Re) ** 0.62 * _f64(Pr) ** (1.0 / 3.0) * _f64(mu_ratio) ** 0.14
        return Nu * _f64(k) / _f64(D_T)
    if method == "oldshue_gretton":
        if D_a is None or d_o is None:
            raise ValueError("oldshue_gretton needs the impeller diameter D_a and the coil tube d_o")
        D_T = _f64(D_T)
        Nu = (0.17 * _f64(Re) ** 0.67 * _f64(Pr) ** 0.37
              * (_f64(D_a) / D_T) ** 0.1 * (_f64(d_o) / D_T) ** 0.5)
        return Nu * _f64(k) / _f64(d_o)
    raise ValueError(f"unknown method {method!r}; choose from {AGITATED_COIL_METHODS}")


# ----------------------------------------------------------------------
# Lumped capacitance (transient)
# ----------------------------------------------------------------------
def biot_number(h: Array | float, L_c: Array | float, k: Array | float) -> Array:
    """Biot number ``Bi = h L_c / k`` (-).

    The lumped-capacitance model (uniform temperature in the solid) is good
    to about 5 % when ``Bi < 0.1`` with ``L_c = V / A_s`` (Incropera & DeWitt
    Eq. 5.10).

    Args:
        h: Surface heat-transfer coefficient (W/m^2/K).
        L_c: Characteristic length ``V / A_s`` (m).
        k: Solid thermal conductivity (W/m/K).

    Returns:
        Bi (-).
    """
    return _f64(h) * _f64(L_c) / _f64(k)


def lumped_time_constant(m_cp: Array | float, UA: Array | float) -> Array:
    """Thermal time constant ``tau = m cp / (U A)`` (s).

    Args:
        m_cp: Heat capacity of the lumped body ``m cp`` (J/K), e.g.
            ``rho V c`` for a solid or the batch contents of a vessel.
        UA: Overall conductance to the surroundings or heating medium (W/K).

    Returns:
        tau (s).
    """
    return _f64(m_cp) / _f64(UA)


def lumped_temperature(
    t: Array | float, T0: Array | float, T_inf: Array | float, m_cp: Array | float, UA: Array | float
) -> Array:
    """Temperature of a lumped body after time ``t`` (K).

    ``(T - T_inf) / (T0 - T_inf) = exp(-t / tau)`` with ``tau = m cp / UA``
    (Incropera & DeWitt Eq. 5.6). The same equation governs a well-mixed batch
    heated through a jacket or coil at a constant medium temperature
    ``T_inf`` (condensing steam), with ``UA`` the jacket's overall conductance.

    Args:
        t: Time (s).
        T0: Initial temperature (K).
        T_inf: Surroundings / heating-medium temperature (K).
        m_cp: Heat capacity ``m cp`` (J/K).
        UA: Conductance ``U A`` (W/K).

    Returns:
        T(t) (K).
    """
    return _f64(T_inf) + (_f64(T0) - _f64(T_inf)) * jnp.exp(-_f64(t) / lumped_time_constant(m_cp, UA))


def lumped_heating_time(
    m: Array | float,
    cp: Array | float,
    UA: Array | float,
    T_st: Array | float,
    T0: Array | float,
    T: Array | float,
) -> Array:
    """Time for a lumped body or batch to go from ``T0`` to ``T`` (s).

    ``t = (m cp / UA) ln[(T_st - T0) / (T_st - T)]``, the inverse of
    :func:`lumped_temperature`. Heating (``T0 < T < T_st``) and cooling
    (``T_st < T < T0``) both work; ``T`` must lie between ``T0`` and ``T_st``.

    Args:
        m: Mass (kg).
        cp: Specific heat capacity (J/kg/K).
        UA: Overall conductance (W/K).
        T_st: Heating/cooling medium temperature (K), e.g. jacket steam.
        T0: Initial temperature (K).
        T: Target temperature (K).

    Returns:
        t (s).

    Example:
        >>> # 1000 kg of water, UA = 2 kW/K, steam at 120 C, 20 C -> 80 C
        >>> round(float(lumped_heating_time(1000.0, 4180.0, 2000.0, 393.15, 293.15, 353.15)))
        1915
    """
    return _f64(m) * _f64(cp) / _f64(UA) * jnp.log((_f64(T_st) - _f64(T0)) / (_f64(T_st) - _f64(T)))


__all__ = [
    "GRAVITY",
    "NU_LAMINAR_TW",
    "NU_LAMINAR_QS",
    "RE_LAMINAR",
    "RE_TURBULENT",
    "FOULING_RESISTANCES",
    "FOULING_RESISTANCE_RANGES",
    "FOULING_SOURCES",
    "TYPICAL_U_RANGES",
    "ANNULUS_NUSSELT_TABLE",
    "STEFAN_BOLTZMANN",
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
    "VISCOSITY_EXPONENT_HEATING",
    "VISCOSITY_EXPONENT_COOLING",
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
    "annulus_nusselt",
    "blackbody_emissive_power",
    "radiation_h",
    "net_radiation",
    "two_surface_radiation",
    "biot_number",
    "lumped_time_constant",
    "lumped_temperature",
    "lumped_heating_time",
    "insulated_pipe_heat_loss",
    "AGITATED_VESSEL_CONSTANTS",
    "AGITATED_COIL_METHODS",
    "agitator_reynolds",
    "agitated_vessel_nusselt",
    "jacketed_vessel_h",
    "agitated_coil_h",
]
