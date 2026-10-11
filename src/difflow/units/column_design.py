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
   The tests check scaling laws, limiting cases, hand calculations and
   independent re-implementations -- **not** a published worked example.

Contents
--------
* Trays: :func:`flow_parameter`, :func:`fair_capacity_factor`,
  :func:`flooding_velocity`, :func:`tray_column_diameter`,
  :func:`tray_percent_flood`, :func:`weir_crest`, :func:`tray_pressure_drop`,
  :func:`weep_point`, :func:`fractional_entrainment`,
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

import jax
import jax.numpy as jnp

from difflow.mass_transfer import PACKINGS, PackingData  # noqa: F401  (re-exported)

Array = jax.Array

G_ACC = 9.80665  # m/s^2

__all__ = [
    "flow_parameter", "fair_capacity_factor", "flooding_velocity",
    "tray_column_diameter", "tray_percent_flood", "weir_crest",
    "tray_pressure_drop", "weep_point", "fractional_entrainment",
    "oconnell_efficiency", "gpdc_flood", "gpdc_pressure_drop",
    "kister_gill_flood_dp", "packed_column_diameter", "hetp", "size_column",
    "column_hardware_cost", "flash_drum_size", "PackingData", "PACKINGS",
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
    looked up). Believed valid for ``0.01 <= F_LV <= 0.1`` and tray spacing
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
        sigma: Liquid surface tension (N/m); required.
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
    (the classical metric Francis formula without end contraction). Towler &
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
      velocity ``u_h = V / (a_h A_a)``;
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

    **From memory, unverified.** In the source ``K2`` is read from a chart
    against ``h_w + h_ow`` (a slowly rising value, roughly 28-32 over 10-100 mm
    of liquid, recalled imprecisely); here it is a constant argument, default
    30. With 5 mm holes ``u_min`` is then about ``11.6/sqrt(rho_V)``, so a
    +-1.5 error in ``K2`` is about +-13 % in ``u_min``. Treat the sign change as
    good to roughly 15 %. The margin is continuous and never raises.

    Args:
        V_vol: Vapor volumetric flow (m^3/s) at the *turndown* condition of
            interest.
        L_vol: Liquid volumetric flow (m^3/s) (sets only the crest, unused if
            ``K2`` is held fixed; kept so the signature matches
            :func:`tray_pressure_drop`).
        rho_V: Vapor density (kg/m^3).
        diameter: Column diameter (m).
        hole_diameter: Hole diameter (m).
        K2: Eduljee constant (see above).

    Returns:
        ``(margin, info)`` with ``u_hole``, ``u_min`` (m/s) and ``h_liquid``
        (``h_w + h_ow``, m).
    """
    u_h = _hole_velocity(V_vol, diameter, downcomer_fraction, hole_area_fraction)
    d_mm = 1000.0 * _f(hole_diameter)
    u_min = (_f(K2) - 0.90 * (25.4 - d_mm)) / jnp.sqrt(_f(rho_V))
    h_liq = _f(weir_height) + weir_crest(L_vol, _f(weir_length_fraction) * _f(diameter))
    return u_h / u_min - 1.0, {"u_hole": u_h, "u_min": u_min, "h_liquid": h_liq}


def fractional_entrainment(F_LV, percent_flood, psi_ref=0.01, exponent_flood=4.0,
                           exponent_F=1.0) -> Array:
    """Fractional entrainment ``psi`` (kg liquid / kg liquid flow): a PLACEHOLDER.

    ``psi = psi_ref (percent_flood/80)^exponent_flood (F_LV/0.1)^exponent_F``.

    **This is not a published fit.** Fair's entrainment chart (``psi`` against
    ``F_LV`` with lines of percent flood) was not available and no fit of it
    could be recalled with confidence, so the default ``psi_ref = 0.01`` (the
    order of magnitude read from the chart at 80 % flood, ``F_LV = 0.1``) and
    the two exponents are **assumed, unverified shapes** that reproduce only
    the qualitative behavior (entrainment rises steeply with percent flood and
    with the liquid load). Expect a factor of 2-3 error. Replace the three
    constants with a fit from Kister, Seader or Fair's own chart before using
    it for a design; the function is built so that they are arguments. The
    limit conventionally applied is ``psi < 0.1`` (Sinnott/Towler: a
    significant efficiency penalty above); see the margin ``0.1 - psi``.
    """
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
    """Eckert flooding line, ``ln Y = -3.5021 - 1.028 ln X - 0.11093 (ln X)^2``."""
    lx = jnp.log(_f(F_LV))
    return jnp.exp(-3.5021 - 1.028 * lx - 0.11093 * lx ** 2)


def gpdc_flood(F_LV, F_p, rho_L, rho_V, mu_L) -> Array:
    """Gas mass flux at flood of a packed column from the Eckert/GPDC flood line (kg/m^2/s).

    With the abscissa ``X = F_LV = (L/G) sqrt(rho_V/(rho_L - rho_V))`` and the
    ordinate ``Y = G^2 F_p mu_L^0.2 / (rho_V (rho_L - rho_V) g)`` (Treybal's
    SI form of the Eckert chart; ``F_p`` in 1/m, ``mu_L`` in cP inside the
    0.2 power), the flooding line is the log-quadratic fit
    ``ln Y = -3.5021 - 1.028 ln X - 0.11093 (ln X)^2``.

    **From memory, unverified.** The fit constants were recalled from the
    literature fits of the Eckert flooding curve (the source is not certain; it
    is the curve in Treybal's Mass-Transfer Operations, Fig. 6.34, and Perry's)
    and checked only against chart orders of magnitude (Y ~ 0.03 at X = 1,
    ~0.2 at 0.1). Believed valid for ``0.01 < X < 5``. Chart-reading accuracy,
    **about 10-20 % on G** (the chart-reading tolerance, stated explicitly).

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


def kister_gill_flood_dp(F_p) -> Array:
    """Pressure drop at flood from the packing factor (Pa/m): Kister & Gill.

    ``Delta P_flood = 0.115 F_p^0.7`` in inches of water per foot of packing
    with ``F_p`` in 1/ft (Kister & Gill, Chem. Eng. Prog. 1991, believed;
    **from memory, unverified**), converted here to Pa/m with ``F_p`` in 1/m.
    A flood *definition* (about 1-2 in H2O/ft, i.e. 800-1700 Pa/m for common
    random packings), good to about 20 %.
    """
    F_ft = _f(F_p) / 3.280839895
    inH2O_ft_to_Pa_m = 249.0889 / 0.3048
    return 0.115 * F_ft ** 0.7 * inH2O_ft_to_Pa_m


def gpdc_pressure_drop(G, L, rho_L, rho_V, mu_L, F_p, exponent=1.7) -> Array:
    """Pressure drop of a packed bed (Pa/m) by anchoring to flood. NOT a GPDC-line fit.

    ``Delta P = Delta P_flood (Y / Y_flood)^n`` at fixed ``L/G``: ``Y`` is the
    Eckert ordinate of the operating ``G``, ``Y_flood`` that of the flood line
    (:func:`gpdc_flood`) at the same flow parameter, and ``Delta P_flood`` the
    Kister-Gill value (:func:`kister_gill_flood_dp`). It is exact at flood by
    construction and has the right dry-bed trend (``n = 1`` would be the
    dry-bed ``Delta P ~ G^2/rho_V`` law; liquid holdup steepens it).

    **The GPDC pressure-drop lines themselves could not be recalled**, so the
    exponent ``n = 1.7`` is an *assumption* chosen so that 70 % of flood
    velocity (``Y/Y_flood = 0.49``) gives ``0.49^1.7 = 0.30`` of the flood
    drop, consistent with the common design rule of 0.25-0.5 in H2O/ft; it is
    not from a published fit. Expect +-30-50 % away from flood. Fit ``exponent``
    to your own chart or vendor data if it matters.

    Args:
        G: Gas mass flux (kg/m^2/s).
        L: Liquid mass flux (kg/m^2/s).
        rho_L, rho_V: Densities (kg/m^3).
        mu_L: Liquid viscosity (Pa s).
        F_p: Packing factor (1/m).
        exponent: ``n``.
    """
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    X = _f(L) / _f(G) * jnp.sqrt(rho_V / (rho_L - rho_V))
    Y = _ordinate(_f(G), _f(F_p), rho_L, rho_V, 1000.0 * _f(mu_L))
    Yf = _gpdc_Y_flood(X)
    return kister_gill_flood_dp(F_p) * (Y / Yf) ** _f(exponent)


def packed_column_diameter(
    V_molar, MW_V, rho_V, L_molar, MW_L, rho_L, mu_L,
    packing: PackingData | None = None, F_p=None, a_p=None,
    fraction_flood=0.7, min_wetting_rate=2.2e-5,
) -> tuple[Array, dict[str, Array]]:
    """Diameter of a packed column at a fraction of GPDC flood (default 70 %).

    The gas mass flux is ``fraction_flood * G_flood`` at the actual ``L/G``
    (:func:`gpdc_flood`), ``D = sqrt(4 V_mass / (pi G))``. Also reports the
    pressure drop per metre (:func:`gpdc_pressure_drop`) and the wetting rate
    ``L / (rho_L a_p)`` (m^2/s, i.e. m^3 of liquid per s per m of perimeter)
    against ``min_wetting_rate`` (default 2.2e-5 m^2/s ~ 0.08 m^3/m/h, a
    commonly quoted minimum for random packing; **from memory, unverified**).

    Args:
        V_molar, MW_V, rho_V: Vapor (mol/s, g/mol, kg/m^3).
        L_molar, MW_L, rho_L: Liquid (mol/s, g/mol, kg/m^3).
        mu_L: Liquid viscosity (Pa s); required.
        packing: A :class:`~difflow.mass_transfer.PackingData` (needs ``F_p``).
        F_p: Packing factor (1/m); overrides ``packing.F_p``.
        a_p: Specific area (m^2/m^3) for the wetting rate; overrides ``packing.a_p``.
        fraction_flood: Fraction of the flooding gas rate.
        min_wetting_rate: Minimum wetting rate (m^2/s).

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
    rho_L, rho_V = _f(rho_L), _f(rho_V)
    V_mass = _f(V_molar) * _f(MW_V) / 1000.0
    L_mass = _f(L_molar) * _f(MW_L) / 1000.0
    # abscissa in the sqrt(rho_V/(rho_L - rho_V)) form (flow_parameter uses rho_L)
    X = L_mass / V_mass * jnp.sqrt(rho_V / (rho_L - rho_V))
    G_f = gpdc_flood(X, F_p, rho_L, rho_V, mu_L)
    ff = _f(fraction_flood)
    G = ff * G_f
    A = V_mass / G
    D = jnp.sqrt(4.0 * A / jnp.pi)
    L_flux = L_mass / A
    dP = gpdc_pressure_drop(G, L_flux, rho_L, rho_V, mu_L, F_p)
    wet = L_flux / (rho_L * _f(a_p))
    info = {
        "F_LV": X, "G_flood": G_f, "G": G, "u_flood": G_f / rho_V, "u_operating": G / rho_V,
        "dP_per_m": dP, "dP_flood_per_m": kister_gill_flood_dp(F_p), "wetting_rate": wet,
        "wetting_margin": wet / _f(min_wetting_rate) - 1.0, "percent_flood": 100.0 * ff,
    }
    return D, info


def hetp(method: str = "random", *, d_p=None, a_p=None, H_OG=None, lam=None) -> Array:
    """Height equivalent to a theoretical plate (m).

    ``method``:

    * ``"random"``: ``HETP = 18 d_p`` (Ludwig's rule of thumb, ``1.5 d_p`` with
      ``d_p`` in inches and HETP in feet; believed, **from memory, unverified**;
      needs ``d_p`` in m; sensible for ``d_p`` of 25-50 mm).
    * ``"structured"``: ``HETP = 100/a_p + 0.10`` m (Kister's rule for
      structured packing, ``a_p`` in m^2/m^3; **from memory, unverified**; about
      0.5 m for ``a_p = 250``).
    * ``"htu"``: ``H_OG ln(lam)/(lam - 1)`` from
      :func:`difflow.mass_transfer.hetp_from_htu`.

    Rules of thumb are good to 20-30 % at best; prefer ``"htu"`` with the Onda
    coefficients when the system is known.
    """
    if method == "random":
        return 18.0 * _f(d_p)
    if method == "structured":
        return 100.0 / _f(a_p) + 0.10
    if method == "htu":
        from difflow.mass_transfer import hetp_from_htu
        return hetp_from_htu(H_OG, lam)
    raise ValueError("method must be 'random', 'structured' or 'htu'")


# =============================================================================
# Whole column
# =============================================================================


def size_column(
    n_theoretical, efficiency, diameter, tray_spacing=None, hetp=None, extra_height=3.0,
) -> dict[str, Array]:
    """Turn stages, efficiency, diameter and spacing into hardware sizes.

    Trays: ``n_trays = n_theoretical / efficiency`` (kept continuous so the
    result is differentiable; round up for a drawing) and
    ``height = n_trays tray_spacing + extra_height``. Packing: pass ``hetp``
    instead of ``tray_spacing``: ``height = n_theoretical hetp + extra_height``
    (and ``n_trays = 0``). ``extra_height`` (default 3 m) covers the vapor
    disengagement space above the top tray and the sump below the bottom one.

    Args:
        n_theoretical: Number of theoretical stages.
        efficiency: Overall tray efficiency (e.g. :func:`oconnell_efficiency`);
            ignored for packing.
        diameter: Column diameter (m).
        tray_spacing: Tray spacing (m), for a trayed column.
        hetp: HETP (m), for a packed column.
        extra_height: Disengagement plus sump height (m).

    Returns:
        Dict with ``n_trays``, ``height`` (m), ``volume`` (shell, m^3),
        ``tray_area`` (m^2 per tray, the ``column_cost(..., "sieve_tray")``
        size) and ``packed_height`` (m).
    """
    if (tray_spacing is None) == (hetp is None):
        raise ValueError("give exactly one of tray_spacing (trays) or hetp (packing)")
    D = _f(diameter)
    area = jnp.pi * D ** 2 / 4.0
    if tray_spacing is not None:
        n_trays = _f(n_theoretical) / _f(efficiency)
        packed = _f(0.0)
        height = n_trays * _f(tray_spacing) + _f(extra_height)
    else:
        n_trays = _f(0.0)
        packed = _f(n_theoretical) * _f(hetp)
        height = packed + _f(extra_height)
    return {"n_trays": n_trays, "height": height, "volume": area * height,
            "tray_area": area, "packed_height": packed}


def column_hardware_cost(sizing: dict[str, Array], kind: str = "sieve_tray") -> dict[str, Array]:
    """Purchased cost ($) of a sized column via :func:`~difflow.economics.capital.column_cost`.

    ``kind``: ``"sieve_tray"`` or ``"valve_tray"`` (shell as
    ``tray_column_shell`` plus ``n_trays`` times the per-tray cost at
    ``tray_area``), or ``"packing_random"`` / ``"packing_structured"`` (shell as
    ``packed_column_shell`` plus the packing at its volume
    ``tray_area * packed_height``). The correlations' validity ranges are those
    of ``column_cost``.

    Returns:
        Dict with ``shell``, ``internals`` and ``total`` ($).
    """
    from difflow.economics.capital import column_cost
    if kind in ("sieve_tray", "valve_tray"):
        shell = column_cost(sizing["volume"], "tray_column_shell")
        internals = sizing["n_trays"] * column_cost(sizing["tray_area"], kind)
    elif kind in ("packing_random", "packing_structured"):
        shell = column_cost(sizing["volume"], "packed_column_shell")
        internals = column_cost(sizing["tray_area"] * sizing["packed_height"], kind)
    else:
        raise ValueError("kind must be sieve_tray, valve_tray, packing_random or packing_structured")
    return {"shell": shell, "internals": internals, "total": shell + internals}


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
