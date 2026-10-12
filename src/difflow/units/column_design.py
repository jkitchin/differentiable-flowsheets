"""Column hardware: tray and packed-column flooding, diameter, hydraulics.

A **library** module (not a palette operation): sizing functions that read the
vapor and liquid rates of a column (a :class:`~difflow.units.distillation.DistillationColumn`
result, a McCabe-Thiele design, an absorber) and return a diameter, a pressure
drop, margins to the weeping / entrainment / flooding limits, and the height
and shell volume that :func:`difflow.economics.capital.column_cost` needs.
Pure JAX, SI units (molecular weights in g/mol, as in ``SpeciesData``), float64,
differentiable in every numeric argument.

.. warning::
   **The correlation constants below were written from memory, with no book
   open, and are unverified.** Each one is marked "from memory, unverified" in
   its docstring together with the source it is believed to come from. They are
   closed-form fits of *charts* and carry chart-reading accuracy, roughly
   10-20 % on a diameter or velocity, worse on a pressure drop and an
   entrainment. Check against the cited source before a design is relied on.

   Checked against published worked examples (``tests/test_column_design.py``,
   ``TestPublishedExamples``): the Fair fit, the dry-plate drop and the weir
   crest against Dutta, *Principles of Mass Transfer and Separation
   Processes* (2007), Example 5.1; the Eckert flood line against Dutta
   Example 5.2; the Robbins pressure drop against Perry's 8e Example 13
   (Sec. 14). The weep constant ``K2`` (a chart reading; pass your own as a
   callable) and :func:`fractional_entrainment` (a placeholder, not Fair's
   chart; it warns) remain unverified, and there is no structured-packing
   capacity model (see :func:`packed_column_diameter`).

Contents
--------
* Trays: :func:`flow_parameter`, :func:`fair_capacity_factor`,
  :func:`flooding_velocity`, :func:`tray_column_diameter`,
  :func:`tray_percent_flood`, :func:`weir_crest`, :func:`tray_pressure_drop`,
  :func:`weep_point`, :func:`downcomer_backup`, :func:`fractional_entrainment`,
  :func:`oconnell_efficiency`.
* Packing: :func:`gpdc_flood`, :func:`gpdc_pressure_drop`,
  :func:`kister_gill_flood_dp`, :func:`packed_column_diameter`, :func:`hetp`
  (the packing records are :class:`difflow.mass_transfer.PackingData` and
  :data:`difflow.mass_transfer.PACKINGS`).
* Whole column: :func:`size_column`, :func:`column_hardware_cost`.
* Flash drums: :func:`flash_drum_size` (Souders-Brown, Walas rules of thumb).

Limits (weeping, entrainment, flooding) are returned as **continuous margins**
and are never raised on traced values, so an optimizer can use them as smooth
constraints.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp

from difflow.mass_transfer import PACKINGS, PackingData  # noqa: F401  (re-exported)

Array = jax.Array

G_ACC = 9.80665  # m/s^2

__all__ = [
    "flow_parameter", "fair_capacity_factor", "flooding_velocity",
    "tray_column_diameter", "tray_percent_flood", "weir_crest",
    "tray_pressure_drop", "weep_point", "downcomer_backup", "fractional_entrainment",
    "EntrainmentPlaceholderWarning", "oconnell_efficiency", "gpdc_flood", "gpdc_pressure_drop",
    "kister_gill_flood_dp", "robbins_pressure_drop", "robbins_flood",
    "DRY_PACKING_FACTORS", "PackingCapacityWarning", "packed_column_diameter", "hetp", "size_column",
    "column_hardware_cost", "SHELL_MATERIAL_FACTORS", "TRAY_MATERIAL_FACTORS",
    "PACKING_MATERIAL_FACTORS", "HARDWARE_COST_BASIS", "flash_drum_size", "PackingData", "PACKINGS",
]


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _softfloor(x: Array, eps: float) -> Array:
    """Smooth ``max(x, 0)`` (shift ``eps^2/(4x)`` for ``x >> eps``)."""
    return 0.5 * (x + jnp.sqrt(x * x + eps * eps))


# =============================================================================
# Trays: Fair flooding and diameter
# =============================================================================


def flow_parameter(L_mass, V_mass, rho_L, rho_V) -> Array:
    """Flow parameter ``F_LV = (L/V) sqrt(rho_V/rho_L)`` (-).

    Args:
        L_mass: Liquid mass flow (kg/s).
        V_mass: Vapor mass flow (kg/s).
        rho_L: Liquid density (kg/m^3).
        rho_V: Vapor density (kg/m^3).
    """
    return _f(L_mass) / _f(V_mass) * jnp.sqrt(_f(rho_V) / _f(rho_L))


def fair_capacity_factor(
    F_LV, tray_spacing, sigma, foaming_factor=1.0, hole_area_factor=1.0
) -> Array:
    """Souders-Brown capacity factor ``C_SB`` at flood from Fair's chart (m/s).

    ``C_SB = [alpha log10(1/F_LV) + beta] (sigma/0.020)^0.2``, with
    ``alpha = 0.0744 t + 0.01173``, ``beta = 0.0304 t + 0.015`` and ``t`` the
    tray spacing in metres. ``C_SB`` is on the **net** area (column minus
    downcomer) and is used with ``u_f = C_SB sqrt((rho_L - rho_V)/rho_V)``.

    **From memory, unverified.** This is the closed-form fit of Fair's (1961)
    flooding chart given by Treybal, *Mass-Transfer Operations* (3rd ed.), in
    the chapter on tray towers (believed; the constants were recalled, not
    looked up). Checked against Dutta (2007) Example 5.1, which reads
    ``C_SB = 0.285 ft/s`` (0.0869 m/s) off Fair's chart at ``F_LV = 0.036``,
    18-in spacing and 20.5 mN/m; the fit gives 0.0956 m/s, 10 % higher, which
    is chart-reading accuracy. Believed valid for ``0.01 <= F_LV <= 0.1`` and tray spacing
    0.15-0.9 m, and it is a fit of chart *reading*: expect 10-20 %. Outside
    that range the logarithm extrapolates linearly in ``log F_LV`` (it reaches
    zero near ``F_LV ~ 4``); a smooth floor of 1 mm/s keeps the result
    positive. The ``(sigma/0.02)^0.2`` correction is the usual one, and
    ``foaming_factor`` (about 0.5-0.9 for foaming systems) and
    ``hole_area_factor`` (1 for hole/active area >= 0.1, about 0.9 at 0.08 and
    0.8 at 0.06 in the usual tables) are plain multipliers.

    Args:
        F_LV: Flow parameter (-), see :func:`flow_parameter`.
        tray_spacing: Tray spacing (m).
        sigma: Liquid surface tension (N/m). Required: there is no default.
        foaming_factor: Multiplier for foaming systems (1 = non-foaming).
        hole_area_factor: Multiplier for a small perforated-area fraction.

    Returns:
        ``C_SB`` (m/s).
    """
    t = _f(tray_spacing)
    alpha = 0.0744 * t + 0.01173
    beta = 0.0304 * t + 0.015
    C = (alpha * jnp.log10(1.0 / _f(F_LV)) + beta) * (_f(sigma) / 0.020) ** 0.2
    C = C * _f(foaming_factor) * _f(hole_area_factor)
    return _softfloor(C, 1e-3)


def flooding_velocity(C_SB, rho_L, rho_V) -> Array:
    """``u_f = C_SB sqrt((rho_L - rho_V)/rho_V)`` (m/s, on the net area)."""
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    return _f(C_SB) * jnp.sqrt((rho_L - rho_V) / rho_V)


def _tray_flood(V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, sigma, tray_spacing,
                foaming_factor, hole_area_factor):
    V_mass = _f(V_molar) * _f(MW_V) / 1000.0
    L_mass = _f(L_molar) * _f(MW_L) / 1000.0
    F_LV = flow_parameter(L_mass, V_mass, rho_L, rho_V)
    C_SB = fair_capacity_factor(F_LV, tray_spacing, sigma, foaming_factor, hole_area_factor)
    u_f = flooding_velocity(C_SB, rho_L, rho_V)
    Q_V = V_mass / _f(rho_V)
    return V_mass, L_mass, F_LV, C_SB, u_f, Q_V


def tray_column_diameter(
    V_molar,
    MW_V,
    rho_V,
    L_molar,
    MW_L,
    rho_L,
    sigma,
    tray_spacing=0.6,
    fraction_flood=0.8,
    downcomer_fraction=0.1,
    foaming_factor=1.0,
    hole_area_factor=1.0,
) -> tuple[Array, dict[str, Array]]:
    """Diameter of a trayed column at a fraction of Fair flooding.

    The operating vapor velocity on the net area ``A_n = A_c - A_d`` is
    ``fraction_flood * u_f`` (:func:`fair_capacity_factor`, :func:`flooding_velocity`),
    so ``A_c = Q_V / (fraction_flood u_f (1 - downcomer_fraction))`` and
    ``D = sqrt(4 A_c / pi)``. The active (bubbling) area is
    ``A_a = A_c - 2 A_d``. Use the vapor and liquid rates of the section with
    the largest ``V`` (usually the bottom of a stripping section or the top of
    the rectifying section; take the larger diameter).

    Args:
        V_molar: Vapor molar flow (mol/s).
        MW_V: Vapor molecular weight (g/mol).
        rho_V: Vapor density (kg/m^3).
        L_molar: Liquid molar flow (mol/s).
        MW_L: Liquid molecular weight (g/mol).
        rho_L: Liquid density (kg/m^3).
        sigma: Liquid surface tension (N/m); required. Estimate it with
            :func:`difflow.liquid_properties.surface_tension` /
            :func:`~difflow.liquid_properties.mixture_surface_tension` when
            there are no data.
        tray_spacing: Tray spacing (m).
        fraction_flood: Fraction of flooding velocity (0.7-0.85 typical).
        downcomer_fraction: ``A_d / A_c`` (0.1 typical; 0.08-0.2).
        foaming_factor: See :func:`fair_capacity_factor`.
        hole_area_factor: See :func:`fair_capacity_factor`.

    Returns:
        ``(D, info)`` with ``info`` keys ``F_LV``, ``C_SB``, ``u_flood`` (m/s,
        net area), ``u_operating``, ``Q_V`` (m^3/s), ``A_column``, ``A_net``,
        ``A_downcomer``, ``A_active`` (m^2) and ``percent_flood``.

    Example:
        >>> D, info = tray_column_diameter(100.0, 80.0, 2.9, 80.0, 85.0, 790.0, 0.02)
    """
    _, _, F_LV, C_SB, u_f, Q_V = _tray_flood(
        V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, sigma, tray_spacing,
        foaming_factor, hole_area_factor)
    ff, dc = _f(fraction_flood), _f(downcomer_fraction)
    u_op = ff * u_f
    A_net = Q_V / u_op
    A_c = A_net / (1.0 - dc)
    D = jnp.sqrt(4.0 * A_c / jnp.pi)
    info = {
        "F_LV": F_LV, "C_SB": C_SB, "u_flood": u_f, "u_operating": u_op, "Q_V": Q_V,
        "A_column": A_c, "A_net": A_net, "A_downcomer": dc * A_c,
        "A_active": (1.0 - 2.0 * dc) * A_c, "percent_flood": 100.0 * ff,
    }
    return D, info


def tray_percent_flood(
    V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, sigma, diameter,
    tray_spacing=0.6, downcomer_fraction=0.1, foaming_factor=1.0, hole_area_factor=1.0,
) -> Array:
    """Percent of Fair flood of an existing column (rating / debottlenecking).

    ``100 * u_net / u_f`` for the given diameter. The inverse of
    :func:`tray_column_diameter`; a continuous margin to the flooding limit
    (``100 - percent_flood``).
    """
    *_, u_f, Q_V = _tray_flood(
        V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, sigma, tray_spacing,
        foaming_factor, hole_area_factor)
    A_c = jnp.pi * _f(diameter) ** 2 / 4.0
    return 100.0 * Q_V / (A_c * (1.0 - _f(downcomer_fraction))) / u_f


# =============================================================================
# Tray hydraulics
# =============================================================================


def weir_crest(Q_L, weir_length) -> Array:
    """Liquid crest over the weir (m): Francis, ``Q = 1.84 l_w h_ow^1.5``.

    ``h_ow = (Q_L / (1.84 l_w))^(2/3)`` with ``Q_L`` in m^3/s and lengths in m
    (the classical metric Francis formula without end contraction). This is
    the same as the US form ``h_ow = 0.48 (q/l_w)^(2/3)`` in inches with ``q``
    in gpm and ``l_w`` in inches used by Dutta (2007) Example 5.1. Towler &
    Sinnott's closely related form uses a constant of about 750 mm, i.e. a crest
    roughly 13 % higher; **from memory, unverified**. A tiny regularisation
    keeps the gradient finite at ``Q_L = 0``.
    """
    q = _f(Q_L) / (1.84 * _f(weir_length))
    return (q * q + 1e-30) ** (1.0 / 3.0)


def _hole_velocity(V_vol, diameter, downcomer_fraction, hole_area_fraction):
    A_c = jnp.pi * _f(diameter) ** 2 / 4.0
    A_a = (1.0 - 2.0 * _f(downcomer_fraction)) * A_c
    return _f(V_vol) / (_f(hole_area_fraction) * A_a)


def tray_pressure_drop(
    V_vol, L_vol, rho_V, rho_L, diameter,
    downcomer_fraction=0.1, hole_area_fraction=0.1, weir_height=0.05,
    weir_length_fraction=0.7, orifice_coefficient=0.8,
) -> tuple[Array, dict[str, Array]]:
    """Pressure drop of one sieve tray (Pa): dry + liquid head + residual.

    ``Delta P = rho_L g (h_d + h_w + h_ow + h_r)`` in the standard textbook
    breakdown:

    * dry plate (orifice) ``h_d = rho_V u_h^2 / (2 C_0^2 rho_L g)`` with the hole
      velocity ``u_h = V / (a_h A_a)`` (identical to the US form
      ``h_d = 0.186 (rho_V/rho_L) u_h^2 / C_0^2`` in inches, ``u_h`` in ft/s,
      of Dutta (2007) Example 5.1: 1.75 in at ``u_h = 39.1 ft/s``, ``C_0 = 0.73``);
    * clear-liquid head: weir height ``h_w`` plus the Francis crest
      :func:`weir_crest` (``l_w = weir_length_fraction D``);
    * residual head ``h_r = 12.5e-3 m * (1000 / rho_L)`` i.e. a constant
      12.5 kg/m^2 liquid-equivalent (Hunt et al., as quoted by Sinnott/Towler;
      **from memory, unverified**).

    ``orifice_coefficient`` (``C_0`` ~ 0.65-0.85 depending on hole size and
    plate thickness; the chart value is not reproduced here, 0.8 is a typical
    mid-range value, **from memory, unverified**) is a geometry input. Accuracy
    of the total is about 20 %.

    Args:
        V_vol: Vapor volumetric flow (m^3/s).
        L_vol: Liquid volumetric flow (m^3/s).
        rho_V: Vapor density (kg/m^3).
        rho_L: Liquid density (kg/m^3).
        diameter: Column diameter (m).
        downcomer_fraction: ``A_d / A_c``.
        hole_area_fraction: Hole area / active area.
        weir_height: Outlet weir height (m).
        weir_length_fraction: ``l_w / D``.
        orifice_coefficient: ``C_0``.

    Returns:
        ``(dP, info)``; ``info`` has the heads ``h_dry``, ``h_weir``, ``h_crest``,
        ``h_residual`` (m of liquid), the components ``dP_dry``, ``dP_liquid``,
        ``dP_residual`` (Pa, summing to ``dP``) and ``u_hole`` (m/s).
    """
    rho_V, rho_L = _f(rho_V), _f(rho_L)
    u_h = _hole_velocity(V_vol, diameter, downcomer_fraction, hole_area_fraction)
    h_d = rho_V * u_h ** 2 / (2.0 * _f(orifice_coefficient) ** 2 * rho_L * G_ACC)
    h_ow = weir_crest(L_vol, _f(weir_length_fraction) * _f(diameter))
    h_w = _f(weir_height)
    h_r = 12.5 / rho_L
    dP_dry = rho_L * G_ACC * h_d
    dP_liq = rho_L * G_ACC * (h_w + h_ow)
    dP_res = rho_L * G_ACC * h_r
    info = {
        "h_dry": h_d, "h_weir": h_w, "h_crest": h_ow, "h_residual": h_r,
        "dP_dry": dP_dry, "dP_liquid": dP_liq, "dP_residual": dP_res, "u_hole": u_h,
    }
    return dP_dry + dP_liq + dP_res, info


def weep_point(
    V_vol, L_vol, rho_V, diameter, hole_diameter=0.005,
    downcomer_fraction=0.1, hole_area_fraction=0.1, weir_height=0.05,
    weir_length_fraction=0.7, K2=30.0,
) -> tuple[Array, dict[str, Array]]:
    """Weeping margin of a sieve tray: ``u_hole / u_min - 1`` (> 0 means no weeping).

    Minimum hole velocity from the Eduljee relation as quoted by Sinnott &
    Towler, ``u_min = (K2 - 0.90 (25.4 - d_h)) / sqrt(rho_V)`` with ``d_h`` in
    mm, ``u_min`` in m/s, ``rho_V`` in kg/m^3.

    **``K2`` is a chart reading, and the chart is not reproduced here.** In
    the source (Coulson & Richardson Vol. 6 / Sinnott, Fig. 11.30, after
    Eduljee, Brit. Chem. Eng. 3, 14 (1958)) ``K2`` is read against the clear
    liquid depth ``h_w + h_ow`` in mm (a slowly rising curve, roughly 28-32
    over 10-100 mm of liquid, recalled imprecisely). No published closed-form
    fit of that curve could be verified, so none is built in. ``K2`` is
    therefore either

    * a number (default 30, **from memory, unverified**; the old behaviour),
      or
    * a callable ``K2(h_mm) -> K2`` of the clear liquid depth
      ``h_w + h_ow`` in **mm**, evaluated here with the Francis crest of
      :func:`weir_crest`. Pass your own chart reading (for instance a
      ``jnp.interp`` over points read off Fig. 11.30) to make the weep point
      depend on the liquid load as in the source; a JAX-traceable callable
      keeps the margin differentiable.

    With 5 mm holes and ``K2 = 30``, ``u_min`` is about ``11.6/sqrt(rho_V)``,
    so a +-1.5 error in ``K2`` is about +-13 % in ``u_min``. Treat the sign
    change as good to roughly 15 %. The margin is continuous and never raises.

    Args:
        V_vol: Vapor volumetric flow (m^3/s) at the *turndown* condition of
            interest.
        L_vol: Liquid volumetric flow (m^3/s); sets the crest ``h_ow`` (used
            by a callable ``K2`` and reported as ``h_liquid``).
        rho_V: Vapor density (kg/m^3).
        diameter: Column diameter (m).
        hole_diameter: Hole diameter (m).
        downcomer_fraction: ``A_d / A_c``.
        hole_area_fraction: Hole area / active area.
        weir_height: Outlet weir height (m).
        weir_length_fraction: ``l_w / D``.
        K2: Eduljee constant, a number or a callable of ``h_w + h_ow`` (mm).

    Returns:
        ``(margin, info)`` with ``u_hole``, ``u_min`` (m/s), ``h_liquid``
        (``h_w + h_ow``, m) and ``K2`` (the value used).

    Example:
        >>> import jax.numpy as jnp
        >>> chart = lambda h_mm: jnp.interp(h_mm, jnp.array([0.0, 100.0]),
        ...                                 jnp.array([29.0, 31.0]))  # your reading
        >>> margin, info = weep_point(1.0, 0.005, 2.9, 1.5, K2=chart)
    """
    u_h = _hole_velocity(V_vol, diameter, downcomer_fraction, hole_area_fraction)
    d_mm = 1000.0 * _f(hole_diameter)
    h_liq = _f(weir_height) + weir_crest(L_vol, _f(weir_length_fraction) * _f(diameter))
    K2v = _f(K2(1000.0 * h_liq)) if callable(K2) else _f(K2)
    u_min = (K2v - 0.90 * (25.4 - d_mm)) / jnp.sqrt(_f(rho_V))
    return u_h / u_min - 1.0, {"u_hole": u_h, "u_min": u_min, "h_liquid": h_liq, "K2": K2v}


def downcomer_backup(
    V_vol, L_vol, rho_V, rho_L, diameter, tray_spacing=0.6,
    downcomer_fraction=0.1, hole_area_fraction=0.1, weir_height=0.05,
    weir_length_fraction=0.7, orifice_coefficient=0.8, apron_clearance=None,
    dP_tray=None, min_residence_time=3.0,
) -> tuple[Array, dict[str, Array]]:
    """Downcomer liquid back-up of a sieve tray and its flooding margin.

    Coulson & Richardson Vol. 6 (Sinnott), Sec. 11.13 (also Towler & Sinnott,
    Sec. 17.13): the clear-liquid back-up in the downcomer is

    ``h_b = (h_w + h_ow) + h_t + h_dc``

    with ``h_t`` the total pressure drop of the tray as liquid head and
    ``h_dc`` the head loss of the liquid flowing under the downcomer apron,

    ``h_dc = 166 (L_wd / (rho_L A_m))^2`` (mm, ``L_wd`` in kg/s, ``A_m`` in m^2),

    i.e. ``h_dc = 0.166 u_ap^2`` m with ``u_ap = Q_L / A_m`` the liquid
    velocity under the apron. ``A_m`` is the smaller of the downcomer area
    ``A_d`` and the clearance area ``A_ap = h_ap l_w``; the apron clearance
    ``h_ap`` is usually set 5-10 mm below the weir height (default here
    ``h_w - 10 mm``). To avoid downcomer flooding the source asks for
    ``h_b < 0.5 (l_t + h_w)`` (the 0.5 allows for the aerated liquid in the
    downcomer), and for a downcomer residence time
    ``t_r = A_d h_b rho_L / L_wd`` of at least 3 s so vapor can disengage.

    The heads use this module's pieces: ``h_ow`` is the Francis crest of
    :func:`weir_crest` (Sinnott's own constant gives a crest about 13 % higher)
    and ``h_t`` is :func:`tray_pressure_drop` (dry + ``h_w + h_ow`` + residual)
    unless ``dP_tray`` is given. Margins are continuous (``A_m`` uses
    ``jnp.minimum``, which has a kink where ``A_ap = A_d``) and never raise.

    Args:
        V_vol: Vapor volumetric flow (m^3/s).
        L_vol: Liquid volumetric flow (m^3/s).
        rho_V: Vapor density (kg/m^3).
        rho_L: Liquid density (kg/m^3).
        diameter: Column diameter (m).
        tray_spacing: Tray spacing ``l_t`` (m).
        downcomer_fraction: ``A_d / A_c``.
        hole_area_fraction: Hole area / active area (for ``h_t``).
        weir_height: Outlet weir height ``h_w`` (m).
        weir_length_fraction: ``l_w / D``.
        orifice_coefficient: ``C_0`` of the dry-plate drop (for ``h_t``).
        apron_clearance: ``h_ap`` (m); default ``weir_height - 0.010``.
        dP_tray: Total tray pressure drop (Pa); default
            :func:`tray_pressure_drop` with the same geometry.
        min_residence_time: Minimum downcomer residence time (s), default 3.

    Returns:
        ``(margin, info)`` with ``margin = 0.5 (l_t + h_w) / h_b - 1``
        (> 0: no downcomer flooding) and ``info`` keys ``h_backup``,
        ``h_crest``, ``h_tray``, ``h_apron`` (m of clear liquid),
        ``h_backup_max`` (m), ``A_apron``, ``A_m`` (m^2), ``u_apron`` (m/s),
        ``residence_time`` (s) and ``residence_margin``
        (``t_r / min_residence_time - 1``).

    Example:
        >>> margin, info = downcomer_backup(2.76, 0.0086, 2.9, 790.0, 1.67)
        >>> bool(margin > 0)
        True
    """
    rho_L = _f(rho_L)
    D = _f(diameter)
    A_c = jnp.pi * D ** 2 / 4.0
    A_d = _f(downcomer_fraction) * A_c
    l_w = _f(weir_length_fraction) * D
    h_w = _f(weir_height)
    h_ap = h_w - 0.010 if apron_clearance is None else _f(apron_clearance)
    h_ow = weir_crest(L_vol, l_w)
    if dP_tray is None:
        dP_tray, _ = tray_pressure_drop(
            V_vol, L_vol, rho_V, rho_L, diameter, downcomer_fraction,
            hole_area_fraction, weir_height, weir_length_fraction, orifice_coefficient)
    h_t = _f(dP_tray) / (rho_L * G_ACC)
    A_ap = h_ap * l_w
    A_m = jnp.minimum(A_d, A_ap)
    Q_L = _f(L_vol)
    u_ap = Q_L / A_m
    h_dc = 0.166 * u_ap ** 2
    h_b = h_w + h_ow + h_t + h_dc
    h_max = 0.5 * (_f(tray_spacing) + h_w)
    t_r = A_d * h_b / Q_L
    info = {
        "h_backup": h_b, "h_crest": h_ow, "h_tray": h_t, "h_apron": h_dc,
        "h_backup_max": h_max, "A_apron": A_ap, "A_m": A_m, "u_apron": u_ap,
        "residence_time": t_r,
        "residence_margin": t_r / _f(min_residence_time) - 1.0,
    }
    return h_max / h_b - 1.0, info


class EntrainmentPlaceholderWarning(UserWarning):
    """:func:`fractional_entrainment` is an assumed shape, not Fair's chart."""


def fractional_entrainment(F_LV, percent_flood, psi_ref=0.01, exponent_flood=4.0,
                           exponent_F=1.0) -> Array:
    """Fractional entrainment ``psi`` (kg liquid / kg liquid flow): a PLACEHOLDER.

    ``psi = psi_ref (percent_flood/80)^exponent_flood (F_LV/0.1)^exponent_F``.

    **This is not Fair's correlation and not any published fit.** Fair's
    entrainment chart (Fair, Petro/Chem Eng. 33(10), 45 (1961); ``psi``
    against ``F_LV`` with lines of percent flood, reproduced by Kister,
    Sinnott/Towler and Seader) was not available, and no closed-form fit of it
    could be recalled or found with confidence, so none is invented here. The
    default ``psi_ref = 0.01`` (an order of magnitude for 80 % flood at
    ``F_LV = 0.1``) and the two exponents are **assumed shapes** that
    reproduce only the qualitative trends (entrainment rises steeply with
    percent flood and with the liquid load). Expect a factor of 2-3 or worse;
    do not use the number for a design. Replace the three constants with a fit
    of the chart you are designing from (they are arguments). The limit
    conventionally applied is ``psi < 0.1`` (Sinnott/Towler: a significant
    efficiency penalty above); see the margin ``0.1 - psi``.

    Every call raises :class:`EntrainmentPlaceholderWarning` (once per call
    site under the default warning filters; at trace time under ``jit``).
    """
    warnings.warn(
        "fractional_entrainment is a placeholder with assumed constants, not a "
        "fit of Fair's entrainment chart; expect a factor of 2-3 error",
        EntrainmentPlaceholderWarning, stacklevel=2)
    return (_f(psi_ref) * (_f(percent_flood) / 80.0) ** _f(exponent_flood)
            * (_f(F_LV) / 0.1) ** _f(exponent_F))


def oconnell_efficiency(alpha, mu_cP, smooth: float | None = None) -> Array:
    """O'Connell's overall tray efficiency (fraction).

    ``E_o = 0.492 (alpha mu_L)^-0.245``, with ``alpha`` the key relative
    volatility and ``mu_L`` the liquid viscosity (cP), both at the average
    column conditions. Clipped to ``[0.1, 1]``: the **hard clip has a zero
    gradient outside that range** (the default, identical to the gas-plant
    function this was promoted from); pass ``smooth`` (a width in efficiency
    units, e.g. 0.02) for a softplus-based clip that stays differentiable.

    References:
        O'Connell, H. E. (1946) Trans. AIChE 42, 741-755 (the data and the
        curve). The power-law fit is the one quoted by Seader, Henley and
        Roper, *Separation Process Principles* (**from memory, unverified**: the
        form gives 0.492 at ``alpha mu = 1`` cP, and about 0.2 at 100).
    """
    E = 0.492 * (_f(alpha) * _f(mu_cP)) ** -0.245
    if smooth is None:
        return jnp.clip(E, 0.1, 1.0)
    w = float(smooth)
    softmax_lo = 0.1 + w * jax.nn.softplus((E - 0.1) / w)
    return 1.0 - w * jax.nn.softplus((1.0 - softmax_lo) / w)


# =============================================================================
# Packing: GPDC
# =============================================================================


def _ordinate(G, F_p, rho_L, rho_V, mu_cP):
    """Eckert ordinate ``Y = G^2 F_p mu^0.2 / (rho_V (rho_L - rho_V) g)``."""
    return G ** 2 * F_p * mu_cP ** 0.2 / (rho_V * (rho_L - rho_V) * G_ACC)


def _gpdc_Y_flood(F_LV):
    """Eckert flooding line, ``log10 Y = -1.6678 - 1.085 log10 X - 0.29655 (log10 X)^2``."""
    lx = jnp.log10(_f(F_LV))
    return 10.0 ** (-1.6678 - 1.085 * lx - 0.29655 * lx ** 2)


def gpdc_flood(F_LV, F_p, rho_L, rho_V, mu_L) -> Array:
    """Gas mass flux at flood of a packed column from the Eckert flood line (kg/m^2/s).

    With the abscissa ``X = F_LV = (L/G) sqrt(rho_V/(rho_L - rho_V))`` and the
    ordinate ``Y = G^2 F_p mu_L^0.2 / (rho_V (rho_L - rho_V) g)`` (the Eckert
    chart in consistent units; ``F_p`` in 1/m, ``mu_L`` in cP inside the 0.2
    power), the flooding line is the log-quadratic fit
    ``log10 Y = -1.6678 - 1.085 log10 X - 0.29655 (log10 X)^2``.

    The fit is the one quoted in Wankat, *Separation Process Engineering*
    (Kessler and Wankat, 1988), **from memory**; it is checked against the
    printed chart reading of Dutta, *Principles of Mass Transfer and
    Separation Processes*, Example 5.2 (``Y = 0.021`` at ``X = 0.90``; the fit
    gives 0.0241, 7 % on ``G``). The earlier fit
    ``ln Y = -3.5021 - 1.028 ln X - 0.11093 (ln X)^2`` gave 0.0335 there (26 %
    high on ``G``) and was replaced. Valid for ``0.01 < X < 5``.

    The Eckert flood line belongs to the old chart and to the high packing
    factors it was built from: Perry's 8e (Sec. 14, "Flood-point
    correlations") restricts it to ``F_p > 60 1/ft`` (about 200 1/m). For
    modern, low-``F_p`` packings use :func:`robbins_flood` (the flood point
    where the Robbins pressure drop reaches the Kister-Gill flood drop).

    Args:
        F_LV: Flow parameter ``(L/G) sqrt(rho_V/(rho_L - rho_V))``.
        F_p: Packing factor (1/m).
        rho_L: Liquid density (kg/m^3).
        rho_V: Vapor density (kg/m^3).
        mu_L: Liquid viscosity (Pa s).

    Returns:
        Superficial gas mass flux at flood (kg/m^2/s).
    """
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    mu_cP = 1000.0 * _f(mu_L)
    Y = _gpdc_Y_flood(F_LV)
    return jnp.sqrt(Y * rho_V * (rho_L - rho_V) * G_ACC / (_f(F_p) * mu_cP ** 0.2))


_INH2O_FT_TO_PA_M = 249.0889 / 0.3048  # 1 in H2O per ft of packing in Pa/m
_FLUX_SI_TO_LB_FT2_H = 3600.0 / 0.45359237 * 0.3048 ** 2  # kg/m^2/s -> lb/ft^2/h
_RHO_SI_TO_LB_FT3 = 0.3048 ** 3 / 0.45359237  # kg/m^3 -> lb/ft^3
_M_TO_FT = 1.0 / 0.3048


def kister_gill_flood_dp(F_p) -> Array:
    """Pressure drop at flood from the packing factor (Pa/m): Kister & Gill.

    ``Delta P_flood = 0.115 F_p^0.7`` in inches of water per foot of packing
    with ``F_p`` in 1/ft (Kister and Gill, Chem. Eng. Prog. 87(2), 32 (1991)),
    converted here to Pa/m with ``F_p`` in 1/m. Perry's 8e Eq. (14-142)
    prints the rounded ``0.12 F_p^0.7``; the 1991 constant is kept here (a
    4 % difference). A flood *definition* (about 1-2 in H2O/ft, i.e.
    800-1700 Pa/m for common random packings), good to about 20 %; for
    ``F_p < 9 1/ft`` Perry's caps it at 2 in H2O/ft (not applied here).
    """
    F_ft = _f(F_p) / _M_TO_FT
    return 0.115 * F_ft ** 0.7 * _INH2O_FT_TO_PA_M


class PackingCapacityWarning(UserWarning):
    """A packed-column capacity model is used outside what it was built for."""


#: Robbins dry packing factors ``F_pd`` (1/m) for the :data:`PACKINGS` keys,
#: from Perry's 8e Tables 14-13 and 14-14 (Robbins, Chem. Eng. Prog., May 1991).
#: Packings without a printed value (plastic Pall rings, Mellapak 250Y,
#: ceramic Intalox 38 mm) are absent; pass ``F_pd`` yourself for those.
DRY_PACKING_FACTORS: dict[str, float] = {
    "raschig_ceramic_25": 492.0,
    "berl_ceramic_25": 308.0,
    "intalox_ceramic_25": 308.0,
    "intalox_ceramic_50": 121.0,
    "pall_metal_25": 174.0,
    "pall_metal_38": 91.0,
    "pall_metal_50": 79.0,
}


def robbins_pressure_drop(G, L, rho_V, rho_L, mu_L, F_pd, pressure_correction: bool = False) -> Array:
    """Irrigated packed-bed pressure drop by the Robbins (1991) correlation (Pa/m).

    Perry's 8e Eqs. (14-145) to (14-151), in their US units internally:

    ``Delta P = C3 Gf^2 10^(C4 Lf) + 0.4 (Lf/20000)^0.1 [C3 Gf^2 10^(C4 Lf)]^4``
    (in H2O per ft), ``C3 = 7.4e-8``, ``C4 = 2.7e-5``, with the gas and
    liquid loading factors

    * ``Gf = G (0.075/rho_G)^0.5 (F_pd/20)^0.5`` (``= 986 F_s (F_pd/20)^0.5``),
      times ``10^(0.3 rho_G)`` if ``pressure_correction`` (Eq. 14-150, for
      operation above atmospheric pressure);
    * ``Lf = L (62.4/rho_L) (F_pd/20)^0.5 mu_L^0.1`` for
      ``15 <= F_pd <= 200 1/ft``; ``(20/F_pd)^0.5`` for ``F_pd < 15``
      (14-151b); ``mu_L^0.2`` for ``F_pd > 200`` (14-151a);

    ``G``, ``L`` in lb/ft^2/h, densities in lb/ft^3, ``mu_L`` in cP and
    ``F_pd`` in 1/ft. Near atmospheric pressure and under vacuum; Perry's
    says it is not suitable above 3 bar.

    Verified against Perry's 8e Example 13 (2-in metal Pall rings, air-water,
    ``G = 2.03``, ``L = 12.2 kg/m^2/s``, ``F_pd = 24 1/ft``): this function
    gives 0.380 in H2O/ft; the book reads 0.40 from the Robbins chart and 0.38
    from the GPDC chart.

    Args:
        G: Gas mass flux (kg/m^2/s).
        L: Liquid mass flux (kg/m^2/s).
        rho_V: Gas density (kg/m^3).
        rho_L: Liquid density (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        F_pd: Dry packing factor (1/m); see :data:`DRY_PACKING_FACTORS`.
        pressure_correction: Apply the ``10^(0.3 rho_G)`` factor of Eq. (14-150).

    Returns:
        Pressure drop per metre of packing (Pa/m).

    Example:
        >>> dp = robbins_pressure_drop(2.03, 12.2, 1.185, 999.5, 1e-3, 24 * 3.2808)
        >>> round(float(dp) / 817.2, 2)   # in H2O per ft
        0.38
    """
    G_lb = _f(G) * _FLUX_SI_TO_LB_FT2_H
    L_lb = _f(L) * _FLUX_SI_TO_LB_FT2_H
    rho_G_lb = _f(rho_V) * _RHO_SI_TO_LB_FT3
    rho_L_lb = _f(rho_L) * _RHO_SI_TO_LB_FT3
    mu_cP = 1000.0 * _f(mu_L)
    Fpd = _f(F_pd) / _M_TO_FT
    Gf = G_lb * jnp.sqrt(0.075 / rho_G_lb) * jnp.sqrt(Fpd / 20.0)
    if pressure_correction:
        Gf = Gf * 10.0 ** (0.3 * rho_G_lb)
    base = L_lb * (62.4 / rho_L_lb)
    Lf = jnp.where(
        Fpd < 15.0, base * jnp.sqrt(20.0 / Fpd) * mu_cP ** 0.1,
        jnp.where(Fpd > 200.0, base * jnp.sqrt(Fpd / 20.0) * mu_cP ** 0.2,
                  base * jnp.sqrt(Fpd / 20.0) * mu_cP ** 0.1))
    dry = 7.4e-8 * Gf ** 2 * 10.0 ** (2.7e-5 * Lf)
    wet = 0.4 * (_softfloor(Lf, 1e-9) / 20000.0) ** 0.1 * dry ** 4
    return (dry + wet) * _INH2O_FT_TO_PA_M


def robbins_flood(L_over_G, rho_V, rho_L, mu_L, F_p, F_pd=None, pressure_correction: bool = False) -> Array:
    """Gas mass flux at flood (kg/m^2/s): Robbins pressure drop = Kister-Gill flood drop.

    Perry's 8e (Sec. 14, flood-point correlations): "For flood and MOC
    predictions, Robbins recommends his pressure drop method together with
    Eq. (14-142)", the Kister-Gill flood pressure drop
    (:func:`kister_gill_flood_dp`). This solves
    ``robbins_pressure_drop(G, (L/G) G) = kister_gill_flood_dp(F_p)`` for
    ``G`` at a fixed liquid-to-gas mass ratio (a bisection on ``ln G``; the
    gradient comes from the implicit-function theorem through one Newton step,
    so it is exact and the solve is ``jit``/``grad`` safe). This is the flood
    model for modern, low-``F_p`` random packings, where the Eckert line
    (:func:`gpdc_flood`) is not recommended. It is **not** a structured-packing
    capacity model: Robbins' correlation is for random packings, and with
    ``F_pd = F_p`` for a structured packing the result is a rough screen only.

    Args:
        L_over_G: Liquid-to-gas mass flow ratio (-).
        rho_V, rho_L: Densities (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        F_p: Packing factor (1/m), for the flood pressure drop.
        F_pd: Dry packing factor (1/m) for Robbins; defaults to ``F_p``.
        pressure_correction: See :func:`robbins_pressure_drop`.

    Returns:
        Superficial gas mass flux at flood (kg/m^2/s).
    """
    F_pd = F_p if F_pd is None else F_pd
    r = _f(L_over_G)
    target = jnp.log(kister_gill_flood_dp(F_p))

    def resid(lnG, r, rho_V, rho_L, mu_L, F_pd, target):
        G = jnp.exp(lnG)
        return jnp.log(robbins_pressure_drop(G, r * G, rho_V, rho_L, mu_L, F_pd,
                                             pressure_correction)) - target

    args = (r, _f(rho_V), _f(rho_L), _f(mu_L), _f(F_pd), target)
    cargs = jax.tree_util.tree_map(jax.lax.stop_gradient, args)

    def body(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        hi = resid(m, *cargs) > 0.0
        return jnp.where(hi, a, m), jnp.where(hi, m, b)

    shape = jnp.broadcast_shapes(*(jnp.shape(a) for a in args))
    lo = jnp.full(shape, jnp.log(1e-5))
    a, b = jax.lax.fori_loop(0, 100, body, (lo, jnp.full(shape, jnp.log(1e3))))
    x0 = jax.lax.stop_gradient(0.5 * (a + b))
    # resid is elementwise in lnG, so a JVP with ones gives d resid / d lnG
    slope = jax.lax.stop_gradient(
        jax.jvp(lambda x: resid(x, *cargs), (x0,), (jnp.ones_like(x0),))[1])
    lnG = x0 - resid(x0, *args) / slope
    return jnp.exp(lnG)


def gpdc_pressure_drop(G, L, rho_L, rho_V, mu_L, F_p, exponent=1.7, *, method: str = "robbins",
                       F_pd=None) -> Array:
    """Pressure drop of an irrigated packed bed (Pa/m).

    ``method``:

    * ``"robbins"`` (default): :func:`robbins_pressure_drop` with the dry
      packing factor ``F_pd`` (defaults to ``F_p``; Perry's tables give both,
      and for metal rings they are within 10-30 %). Perry's 8e Example 13
      shows Robbins and the GPDC chart agree (0.40 vs 0.38 in H2O/ft).
    * ``"anchored"``: the old model,
      ``Delta P = Delta P_flood (Y / Y_flood)^n`` at fixed ``L/G``, anchored to
      the Kister-Gill flood drop at the Eckert flood line with an *assumed*
      exponent ``n = exponent``. It is kept only for comparison: for Perry's
      Example 13 it gives about a third of the published 0.38-0.40 in H2O/ft,
      because the Eckert line puts flood too high for a modern packing.

    Args:
        G: Gas mass flux (kg/m^2/s).
        L: Liquid mass flux (kg/m^2/s).
        rho_L, rho_V: Densities (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        F_p: Packing factor (1/m).
        exponent: ``n`` of the ``"anchored"`` model.
        method: ``"robbins"`` or ``"anchored"``.
        F_pd: Robbins dry packing factor (1/m); defaults to ``F_p``.
    """
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    if method == "robbins":
        return robbins_pressure_drop(G, L, rho_V, rho_L, mu_L, F_p if F_pd is None else F_pd)
    if method != "anchored":
        raise ValueError("method must be 'robbins' or 'anchored'")
    X = _f(L) / _f(G) * jnp.sqrt(rho_V / (rho_L - rho_V))
    Y = _ordinate(_f(G), _f(F_p), rho_L, rho_V, 1000.0 * _f(mu_L))
    Yf = _gpdc_Y_flood(X)
    return kister_gill_flood_dp(F_p) * (Y / Yf) ** _f(exponent)


def packed_column_diameter(
    V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, mu_L,
    packing: PackingData | None = None, F_p=None, a_p=None,
    fraction_flood=0.7, min_wetting_rate=2.2e-5, *, flood: str = "eckert", F_pd=None,
) -> tuple[Array, dict[str, Array]]:
    """Diameter of a packed column at a fraction of flood (default 70 %).

    The gas mass flux is ``fraction_flood * G_flood`` at the actual ``L/G``,
    ``D = sqrt(4 V_mass / (pi G))``. ``flood`` picks the flood model:

    * ``"eckert"`` (default): the Eckert flood line, :func:`gpdc_flood`; meant
      for high packing factors (Perry's: ``F_p > 60 1/ft``, about 200 1/m).
    * ``"robbins"``: Robbins pressure drop reaching the Kister-Gill flood drop,
      :func:`robbins_flood`; the choice for modern packings.

    Also reports the Robbins pressure drop per metre (:func:`robbins_pressure_drop`)
    and the wetting rate ``L / (rho_L a_p)`` (m^2/s, i.e. m^3 of liquid per s
    per m of perimeter) against ``min_wetting_rate`` (default 2.2e-5 m^2/s ~
    0.08 m^3/m/h, a commonly quoted minimum for random packing; **from memory,
    unverified**).

    **Structured packing: no capacity model.** Neither flood model is a
    structured-packing correlation. The Eckert line is for high-``F_p``
    random packings, and Robbins' pressure drop is a random-packing
    correlation used here with ``F_pd = F_p`` when no dry factor is known
    (Mellapak 250Y has none in :data:`DRY_PACKING_FACTORS`; a
    :class:`PackingCapacityWarning` says so). Only the flood pressure drop
    (:func:`kister_gill_flood_dp`) is believed to have been stated for
    structured packings as well. Treat a structured-packing diameter from
    this function as a rough screening number and size from the vendor's
    capacity chart (or a Kister-Gill / Billet-Schultes / SRP model, not
    implemented here).

    Args:
        V_molar, MW_V, rho_V: Vapor (mol/s, g/mol, kg/m^3).
        L_molar, MW_L, rho_L: Liquid (mol/s, g/mol, kg/m^3).
        mu_L: Liquid viscosity (Pa s); required.
        packing: A :class:`~difflow.mass_transfer.PackingData` (needs ``F_p``).
        F_p: Packing factor (1/m); overrides ``packing.F_p``.
        a_p: Specific area (m^2/m^3) for the wetting rate; overrides ``packing.a_p``.
        fraction_flood: Fraction of the flooding gas rate.
        min_wetting_rate: Minimum wetting rate (m^2/s).
        flood: ``"eckert"`` or ``"robbins"``.
        F_pd: Robbins dry packing factor (1/m); defaults to
            :data:`DRY_PACKING_FACTORS` for a listed ``packing``, else ``F_p``.

    Returns:
        ``(D, info)`` with ``F_LV``, ``G_flood``, ``G`` (kg/m^2/s), ``u_flood``,
        ``u_operating`` (m/s, superficial), ``dP_per_m``, ``dP_flood_per_m``
        (Pa/m), ``wetting_rate`` (m^2/s), ``wetting_margin``
        (``rate / minimum - 1``, > 0 means wetted) and ``percent_flood``.
    """
    if F_p is None:
        if packing is None or packing.F_p is None:
            raise ValueError("give `packing` with F_p, or F_p")
        F_p = packing.F_p
    if a_p is None:
        if packing is None:
            raise ValueError("give `packing` or a_p (needed for the wetting rate)")
        a_p = packing.a_p
    if F_pd is None:
        if packing is not None and packing.name not in DRY_PACKING_FACTORS:
            warnings.warn(
                f"no Robbins dry packing factor for {packing.name!r}; using F_pd = F_p. "
                "Robbins is a random-packing correlation and there is no "
                "structured-packing capacity model here (see packed_column_diameter)",
                PackingCapacityWarning, stacklevel=2)
        F_pd = DRY_PACKING_FACTORS.get(packing.name, F_p) if packing is not None else F_p
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    V_mass = _f(V_molar) * _f(MW_V) / 1000.0
    L_mass = _f(L_molar) * _f(MW_L) / 1000.0
    # abscissa in the sqrt(rho_V/(rho_L - rho_V)) form (flow_parameter uses rho_L)
    X = L_mass / V_mass * jnp.sqrt(rho_V / (rho_L - rho_V))
    if flood == "eckert":
        G_f = gpdc_flood(X, F_p, rho_L, rho_V, mu_L)
    elif flood == "robbins":
        G_f = robbins_flood(L_mass / V_mass, rho_V, rho_L, mu_L, F_p, F_pd)
    else:
        raise ValueError("flood must be 'eckert' or 'robbins'")
    ff = _f(fraction_flood)
    G = ff * G_f
    A = V_mass / G
    D = jnp.sqrt(4.0 * A / jnp.pi)
    L_flux = L_mass / A
    dP = robbins_pressure_drop(G, L_flux, rho_V, rho_L, mu_L, F_pd)
    wet = L_flux / (rho_L * _f(a_p))
    info = {
        "F_LV": X, "G_flood": G_f, "G": G, "u_flood": G_f / rho_V, "u_operating": G / rho_V,
        "dP_per_m": dP, "dP_flood_per_m": kister_gill_flood_dp(F_p), "wetting_rate": wet,
        "wetting_margin": wet / _f(min_wetting_rate) - 1.0, "percent_flood": 100.0 * ff,
    }
    return D, info


def hetp(method: str = "random", *, d_p=None, a_p=None, H_OG=None, lam=None, C_XY=1.0) -> Array:
    """Height equivalent to a theoretical plate (m).

    ``method``:

    * ``"random"``: ``HETP = 18 d_p`` (Perry's 8e Eq. 14-156a, the
      Frank/Harrison-France rule ``HETP (ft) = 1.5 d_p (in)``; ``d_p`` in m;
      for ``d_p`` of 25-50 mm). Perry's adds that HETP should not be taken
      below the column diameter for columns under 0.67 m.
    * ``"structured"``: ``HETP = 100 C_XY / a_p + 0.10`` m (Perry's 8e
      Eq. 14-159, Kister's rule; ``a_p`` in m^2/m^3). ``C_XY = 1`` for
      Y-type (45 degree), S-type and high-capacity packings; 1.45 for X-type
      (60 degree) packings with ``a_p < 300``. About 0.5 m for ``a_p = 250``.
    * ``"htu"``: ``H_OG ln(lam)/(lam - 1)`` from
      :func:`difflow.mass_transfer.hetp_from_htu`.

    Both rules are for organic and hydrocarbon systems with surface tension
    below 25 mN/m and ``lam`` near 1; Perry's warns that aqueous and
    high-surface-tension systems give larger HETPs. They are good to 20-30 %
    at best; prefer ``"htu"`` with the Onda coefficients when the system is
    known.
    """
    if method == "random":
        return 18.0 * _f(d_p)
    if method == "structured":
        return 100.0 * _f(C_XY) / _f(a_p) + 0.10
    if method == "htu":
        from difflow.mass_transfer import hetp_from_htu
        return hetp_from_htu(H_OG, lam)
    raise ValueError("method must be 'random', 'structured' or 'htu'")


# =============================================================================
# Whole column
# =============================================================================


def size_column(
    n_theoretical, efficiency, diameter, tray_spacing=None, hetp=None, extra_height=3.0,
    *, packed_height=None,
) -> dict[str, Array]:
    """Turn stages, efficiency, diameter and spacing into hardware sizes.

    Give exactly one of ``tray_spacing``, ``hetp`` or ``packed_height``:

    * Trays (``tray_spacing``): ``n_trays = n_theoretical / efficiency`` (kept
      continuous so the result is differentiable; round up for a drawing) and
      ``height = n_trays tray_spacing + extra_height``.
    * Packing from an HETP (``hetp``):
      ``packed_height = n_theoretical hetp`` and
      ``height = packed_height + extra_height``.
    * Packing of a known depth (``packed_height``, e.g. ``N_OG H_OG`` from a
      rate-based absorber design, or the installed bed of an existing column):
      ``height = packed_height + extra_height``; ``n_theoretical`` and
      ``efficiency`` are not used.

    Packed columns have ``n_trays = 0``. ``extra_height`` (default 3 m)
    covers the vapor disengagement space above the top tray or bed and the
    sump below the bottom one (add distributor and support space to it for a
    multi-bed packed column).

    Args:
        n_theoretical: Number of theoretical stages.
        efficiency: Overall tray efficiency (e.g. :func:`oconnell_efficiency`);
            ignored for packing.
        diameter: Column diameter (m).
        tray_spacing: Tray spacing (m), for a trayed column.
        hetp: HETP (m), for a packed column.
        extra_height: Disengagement plus sump height (m).
        packed_height: Total packed depth (m), for a packed column sized by
            height rather than by stages (keyword only).

    Returns:
        Dict with ``n_trays``, ``height`` (m), ``volume`` (shell, m^3),
        ``tray_area`` (m^2 per tray, the ``column_cost(..., "sieve_tray")``
        size; the cross-section for a packed column) and ``packed_height`` (m).

    Example:
        >>> s = size_column(None, None, 1.2, packed_height=6.0, extra_height=2.0)
        >>> float(s["height"])
        8.0
    """
    if sum(x is not None for x in (tray_spacing, hetp, packed_height)) != 1:
        raise ValueError("give exactly one of tray_spacing (trays), hetp or "
                         "packed_height (packing)")
    D = _f(diameter)
    area = jnp.pi * D ** 2 / 4.0
    if tray_spacing is not None:
        n_trays = _f(n_theoretical) / _f(efficiency)
        packed = _f(0.0)
        height = n_trays * _f(tray_spacing) + _f(extra_height)
    else:
        n_trays = _f(0.0)
        packed = _f(packed_height) if packed_height is not None else _f(n_theoretical) * _f(hetp)
        height = packed + _f(extra_height)
    return {"n_trays": n_trays, "height": height, "volume": area * height,
            "tray_area": area, "packed_height": packed}


#: Purchased-cost material factors ``F_M`` for the column shell, relative to
#: carbon steel: Turton et al. 5e, Table A.3 / Fig. A.18, "process vessels,
#: including towers" (**recalled, not looked up; check before relying**).
SHELL_MATERIAL_FACTORS: dict[str, float] = {
    "carbon_steel": 1.0,
    "stainless_clad": 1.7,
    "stainless": 3.1,
    "nickel_alloy_clad": 3.6,
    "nickel_alloy": 7.1,
    "titanium_clad": 4.7,
    "titanium": 9.4,
}

#: Material factors for trays, relative to carbon steel: Turton et al. 5e,
#: Table A.6, sieve trays (the tray ``F_BM``, which multiplies the purchased
#: tray cost directly); applied to valve trays too (**recalled, not looked up**).
TRAY_MATERIAL_FACTORS: dict[str, float] = {
    "carbon_steel": 1.0,
    "stainless": 1.8,
    "nickel_alloy": 5.6,
}

#: Material factors for random (loose) packing, relative to Turton's
#: ``packing_random`` line, which is the polyethylene basis: Turton et al. 5e,
#: Table A.6, tower packing (``F_BM`` polyethylene 1.0, ceramic 4.1, 304 SS
#: 7.1; **recalled, not looked up**). ``"plastic"`` is an alias of
#: polyethylene and ``"metal"`` of 304 stainless.
PACKING_MATERIAL_FACTORS: dict[str, float] = {
    "polyethylene": 1.0,
    "plastic": 1.0,
    "ceramic": 4.1,
    "stainless": 7.1,
    "metal": 7.1,
}

#: Dollar basis of each ``column_hardware_cost`` piece (before escalation).
HARDWARE_COST_BASIS: dict[str, str] = {
    "shell": "Turton 5e Table A.1 tower (2001 $, purchased, carbon steel, ~ambient P)",
    "sieve_tray": "Turton 5e Table A.1 sieve trays (2001 $, purchased, carbon steel, per tray)",
    "valve_tray": "Turton 5e Table A.1 valve trays (2001 $, purchased, carbon steel, per tray)",
    "packing_random": "Turton 5e Table A.1 loose packing (2001 $, purchased, polyethylene basis)",
    "packing_structured": ("Seider 4e Sec. 16.5 corrugated-sheet stainless, $285/ft^3 "
                           "(2013 $, INSTALLED, very approximate)"),
}


def _material_factor(material, table: dict[str, float], what: str) -> Array:
    if isinstance(material, str):
        if material not in table:
            raise ValueError(f"unknown {what} material {material!r}; one of "
                             f"{sorted(table)} or a numeric factor")
        return _f(table[material])
    return _f(material)


def column_hardware_cost(
    sizing: dict[str, Array], kind: str = "sieve_tray", *,
    material="carbon_steel", internals_material=None,
    target_year: int | None = None, structured_installation_factor=None,
) -> dict[str, Array]:
    """Cost ($) of a sized column via :func:`~difflow.economics.capital.column_cost`.

    ``kind``: ``"sieve_tray"`` or ``"valve_tray"`` (shell as
    ``tray_column_shell`` plus ``n_trays`` times the per-tray cost at
    ``tray_area``), or ``"packing_random"`` / ``"packing_structured"`` (shell as
    ``packed_column_shell`` plus the packing at its volume
    ``tray_area * packed_height``). The correlations' validity ranges are those
    of ``column_cost`` (a ``CostRangeWarning`` outside them).

    **Cost basis.** All pieces are escalated by CEPCI from their base year to
    ``target_year`` (default ``difflow.economics.indices.DEFAULT_CURRENT_YEAR``,
    returned as ``cost_year``):

    * shell, sieve and valve trays, random packing: Turton et al. 5e Table A.1,
      **2001 dollars, purchased cost** (free on board, not installed);
      carbon steel at near-ambient pressure; Turton's loose-packing line is
      the **polyethylene** basis;
    * structured packing: Seider et al. 4e Sec. 16.5, stainless corrugated
      sheet at $285/ft^3, **2013 dollars, an INSTALLED cost** ("very
      approximate"). It is the only installed number here and is not
      comparable with the purchased costs above. No source for the
      installation factor folded into it was available, so it is not
      converted by default: ``internals_installed`` is 1 in the result. Pass
      ``structured_installation_factor`` (installed / purchased, your own
      value) to divide it out; ``internals_installed`` is then 0. A pressure
      factor is not applied to anything (see
      :func:`~difflow.economics.capital.bare_module_cost`).

    **Materials** multiply the purchased cost (``F_M``; recalled from Turton
    5e Tables A.3 and A.6, not looked up: check before relying):

    * ``material`` (shell): a key of :data:`SHELL_MATERIAL_FACTORS`
      (carbon steel 1.0, stainless clad 1.7, stainless 3.1, ...), or a number.
    * ``internals_material``: for trays a key of
      :data:`TRAY_MATERIAL_FACTORS` (carbon steel 1.0, stainless 1.8,
      nickel alloy 5.6); for random packing a key of
      :data:`PACKING_MATERIAL_FACTORS` (polyethylene/plastic 1.0, ceramic
      4.1, stainless/metal 7.1). ``None`` (default) keeps the correlation's
      own basis: carbon-steel trays, **polyethylene** random packing (so
      metal rings must be asked for with ``"metal"``: about 7x),
      stainless structured packing. For structured packing only
      ``None``/``"stainless"`` or a numeric factor is accepted (Seider's price
      is for stainless; no other material ratio is tabulated here).

    Args:
        sizing: Output of :func:`size_column`.
        kind: ``"sieve_tray"``, ``"valve_tray"``, ``"packing_random"`` or
            ``"packing_structured"``.
        material: Shell material key or numeric ``F_M``.
        internals_material: Tray/packing material key or numeric factor.
        target_year: Dollar year of the result.
        structured_installation_factor: Installed/purchased ratio to convert
            Seider's installed structured-packing price to purchased.

    Returns:
        Dict of arrays: ``shell``, ``internals``, ``total`` ($ of
        ``cost_year``), ``cost_year``, ``shell_material_factor``,
        ``internals_material_factor`` and ``internals_installed`` (1.0 when
        ``internals`` is an installed cost, else 0.0). The text basis of each
        piece is in :data:`HARDWARE_COST_BASIS`.

    Example:
        >>> s = size_column(20, 0.6, 1.5, tray_spacing=0.6)
        >>> c = column_hardware_cost(s, material="stainless", internals_material="stainless")
        >>> float(c["internals_material_factor"])
        1.8
    """
    from difflow.economics.capital import column_cost
    from difflow.economics.indices import DEFAULT_CURRENT_YEAR
    year = DEFAULT_CURRENT_YEAR if target_year is None else int(target_year)
    F_shell = _material_factor(material, SHELL_MATERIAL_FACTORS, "shell")
    installed = 0.0
    if kind in ("sieve_tray", "valve_tray"):
        F_int = _material_factor(
            "carbon_steel" if internals_material is None else internals_material,
            TRAY_MATERIAL_FACTORS, "tray")
        shell = column_cost(sizing["volume"], "tray_column_shell", year)
        internals = sizing["n_trays"] * column_cost(sizing["tray_area"], kind, year)
    elif kind == "packing_random":
        F_int = _material_factor(
            "polyethylene" if internals_material is None else internals_material,
            PACKING_MATERIAL_FACTORS, "packing")
        shell = column_cost(sizing["volume"], "packed_column_shell", year)
        internals = column_cost(sizing["tray_area"] * sizing["packed_height"], kind, year)
    elif kind == "packing_structured":
        F_int = _material_factor(
            "stainless" if internals_material is None else internals_material,
            {"stainless": 1.0}, "structured packing")
        shell = column_cost(sizing["volume"], "packed_column_shell", year)
        internals = column_cost(sizing["tray_area"] * sizing["packed_height"], kind, year)
        if structured_installation_factor is None:
            installed = 1.0
        else:
            internals = internals / _f(structured_installation_factor)
    else:
        raise ValueError("kind must be sieve_tray, valve_tray, packing_random or packing_structured")
    shell = F_shell * shell
    internals = F_int * internals
    return {"shell": shell, "internals": internals, "total": shell + internals,
            "cost_year": _f(year), "shell_material_factor": F_shell,
            "internals_material_factor": F_int, "internals_installed": _f(installed)}


# =============================================================================
# Flash (knockout) drums: Souders-Brown sizing
# =============================================================================


def flash_drum_size(
    Q_V, Q_L, rho_L, rho_V, k=0.107, holdup_time=300.0, fill_fraction=0.5,
    L_over_D=3.0,
) -> tuple[Array, dict[str, Array]]:
    """Size a vertical vapor-liquid separator (flash drum) by Souders-Brown.

    The vapor may rise no faster than the Souders-Brown velocity

    ``u_max = k sqrt((rho_L - rho_V) / rho_V)``,

    at which a droplet of a given size just hangs in the up-flowing gas. The
    cross-section that keeps the vapor below ``u_max`` gives one diameter,
    ``D_vapor = sqrt(4 Q_V / (pi u_max))``. The liquid gives a second: the drum
    must hold ``Q_L holdup_time`` of liquid in ``fill_fraction`` of its volume,
    and with height ``H = L_over_D D`` that is
    ``D_liquid = (4 Q_L holdup_time / (pi fill_fraction L_over_D))^(1/3)``.
    The drum takes the larger, ``D = max(D_vapor, D_liquid)``, and
    ``H = L_over_D D``. Every output is differentiable in every argument
    (the ``max`` has a kink only where the two diameters are equal).

    The defaults are Walas's rules of thumb for drums and gas-liquid
    separators (S. M. Walas, *Chemical Process Equipment*, Butterworths 1988,
    "Rules of thumb"; also quoted at the front of Perry's Handbook, sec. 1 in
    the 7th and 8th editions): ``k = 0.35 ft/s = 0.107 m/s`` with a wire-mesh
    deentrainer (``0.1 ft/s = 0.0305 m/s`` without one), ``L/D = 3`` (range
    2.5-5) and a liquid holdup of 5 min with the drum half full
    (``holdup_time = 300 s``, ``fill_fraction = 0.5``); 5-10 min is usual for
    a drum feeding a column, about 30 min for one feeding a furnace. They are
    screening numbers, not a mechanical design.

    Args:
        Q_V: Vapor volumetric flow at drum conditions (m^3/s).
        Q_L: Liquid volumetric flow (m^3/s).
        rho_L: Liquid density (kg/m^3).
        rho_V: Vapor density (kg/m^3).
        k: Souders-Brown coefficient (m/s).
        holdup_time: Liquid residence time (s).
        fill_fraction: Fraction of the drum volume the held-up liquid fills.
        L_over_D: Height-to-diameter ratio.

    Returns:
        ``(D, info)``: the diameter (m) and a dict with ``u_max`` (m/s),
        ``D_vapor``, ``D_liquid``, ``height`` (m), ``volume`` (m^3),
        ``liquid_volume`` (m^3), ``u_vapor`` (actual superficial vapor
        velocity, m/s) and ``fraction_of_u_max`` (1 when the vapor sets the
        diameter, below 1 when the liquid holdup does).

    Example:
        >>> D, info = flash_drum_size(Q_V=0.5, Q_L=0.002, rho_L=600.0, rho_V=20.0)
        >>> bool(info["u_max"] > 0.5) and bool(D > info["D_liquid"] - 1e-12)
        True
    """
    Q_V, Q_L = _f(Q_V), _f(Q_L)
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    u_max = _f(k) * jnp.sqrt((rho_L - rho_V) / rho_V)
    D_vapor = jnp.sqrt(4.0 * Q_V / (jnp.pi * u_max))
    V_liq = Q_L * _f(holdup_time)
    D_liquid = (4.0 * V_liq / (jnp.pi * _f(fill_fraction) * _f(L_over_D))) ** (1.0 / 3.0)
    D = jnp.maximum(D_vapor, D_liquid)
    area = jnp.pi * D ** 2 / 4.0
    height = _f(L_over_D) * D
    u_vapor = Q_V / area
    return D, {
        "u_max": u_max, "D_vapor": D_vapor, "D_liquid": D_liquid,
        "height": height, "volume": area * height, "liquid_volume": V_liq,
        "u_vapor": u_vapor, "fraction_of_u_max": u_vapor / u_max,
    }
