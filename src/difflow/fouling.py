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
* TEMA design values: :data:`TEMA_FOULING`, :data:`TEMA_CRUDE_OIL`,
  :data:`TEMA_WATER`, :func:`tema_fouling_resistance`,
  :func:`tema_crude_oil_fouling`, :func:`tema_water_fouling`

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


# ----------------------------------------------------------------------
# TEMA design fouling resistances (RGP-T-2.4)
# ----------------------------------------------------------------------
#: 1 h ft^2 degF / Btu in m^2 K / W. TEMA tabulates in h ft^2 degF/Btu.
HFT2F_PER_BTU = 0.17611

#: TEMA design fouling resistances, ``(low, high)`` in the TEMA unit
#: **h ft^2 degF/Btu** (single values have ``low == high``). Multiply by
#: :data:`HFT2F_PER_BTU` for m^2 K/W, or use :func:`tema_fouling_resistance`.
#:
#: Source: TEMA *Standards of the Tubular Exchanger Manufacturers
#: Association*, 7th ed. (1988), Recommended Good Practice Table RGP-T-2.4
#: ("Design fouling resistances"), the industrial-fluid, chemical-processing,
#: natural-gas/gasoline-processing and oil-refinery lists. **Provenance:**
#: the values were recalled from that table as it is reproduced in the
#: secondary literature (e.g. Serth, *Process Heat Transfer*, Ch. 3; Kakac &
#: Liu, *Heat Exchangers*); no printed copy of TEMA was available to check
#: them. They agree, after unit conversion and rounding to one significant
#: figure, with the TEMA-derived Incropera & DeWitt Table 11.1 entries in
#: :data:`difflow.heat_transfer.FOULING_RESISTANCES` (a test checks this);
#: entries that table does not cover are unchecked. TEMA itself calls these
#: guidance values; the design value is the purchaser's choice.
TEMA_FOULING: dict[str, tuple[float, float]] = {
    # Industrial fluids: oils
    "fuel_oil_no2": (0.002, 0.002),
    "fuel_oil_no6": (0.005, 0.005),
    "transformer_oil": (0.001, 0.001),
    "engine_lube_oil": (0.001, 0.001),
    "quench_oil": (0.004, 0.004),
    # Industrial fluids: gases and vapours
    "manufactured_gas": (0.010, 0.010),
    "engine_exhaust_gas": (0.010, 0.010),
    "steam_non_oil_bearing": (0.0005, 0.0005),
    "exhaust_steam_oil_bearing": (0.0015, 0.002),
    "refrigerant_vapors_oil_bearing": (0.002, 0.002),
    "compressed_air": (0.001, 0.001),
    "ammonia_vapor": (0.001, 0.001),
    "co2_vapor": (0.001, 0.001),
    "chlorine_vapor": (0.002, 0.002),
    "coal_flue_gas": (0.010, 0.010),
    "natural_gas_flue_gas": (0.005, 0.005),
    # Industrial fluids: liquids
    "molten_heat_transfer_salts": (0.0005, 0.0005),
    "refrigerant_liquids": (0.001, 0.001),
    "hydraulic_fluid": (0.001, 0.001),
    "industrial_organic_heat_transfer_media": (0.002, 0.002),
    "ammonia_liquid": (0.001, 0.001),
    "ammonia_liquid_oil_bearing": (0.003, 0.003),
    "calcium_chloride_solutions": (0.003, 0.003),
    "sodium_chloride_solutions": (0.003, 0.003),
    "co2_liquid": (0.001, 0.001),
    "chlorine_liquid": (0.002, 0.002),
    "methanol_solutions": (0.002, 0.002),
    "ethanol_solutions": (0.002, 0.002),
    "ethylene_glycol_solutions": (0.002, 0.002),
    # Chemical processing streams
    "acid_gases": (0.002, 0.003),
    "solvent_vapors": (0.001, 0.001),
    "stable_overhead_products": (0.001, 0.001),
    "mea_dea_solutions": (0.002, 0.002),
    "deg_teg_solutions": (0.002, 0.002),
    "stable_side_draw_and_bottom_products": (0.001, 0.002),
    "caustic_solutions": (0.002, 0.002),
    "vegetable_oils": (0.003, 0.003),
    # Natural gas - gasoline processing streams
    "natural_gas": (0.001, 0.002),
    "natural_gas_overhead_products": (0.001, 0.002),
    "lean_oil": (0.002, 0.002),
    "rich_oil": (0.001, 0.002),
    "natural_gasoline_and_lpg": (0.001, 0.002),
    # Oil refinery streams
    "gasoline": (0.002, 0.002),
    "naphtha_and_light_distillates": (0.002, 0.003),
    "kerosene": (0.002, 0.003),
    "light_gas_oil": (0.002, 0.003),
    "heavy_gas_oil": (0.003, 0.005),
    "heavy_fuel_oils": (0.005, 0.007),
    "asphalt_and_residuum": (0.010, 0.010),
    # Lube oil processing streams
    "lube_oil_feed_stock": (0.002, 0.002),
    "lube_oil_solvent_feed_mix": (0.002, 0.002),
    "lube_oil_solvent": (0.001, 0.001),
    "lube_oil_extract": (0.003, 0.003),
    "lube_oil_raffinate": (0.001, 0.001),
    "lube_oil_asphalt": (0.005, 0.005),
    "lube_oil_wax_slurries": (0.003, 0.003),
    "refined_lube_oil": (0.001, 0.001),
}

#: TEMA crude-oil fouling resistance (h ft^2 degF/Btu) by bulk temperature
#: (rows: 0-199, 200-299, 300-499, 500 degF and over) and tube velocity
#: (columns: below 2, 2-4, above 4 ft/s). Same source and provenance note as
#: :data:`TEMA_FOULING` (TEMA 7th ed. RGP-T-2.4, recalled from secondary
#: reproductions, unchecked against a printed copy). Older (pre-1988) TEMA
#: editions split crude into "dry" and "salt" columns with different values;
#: this is the later single table.
TEMA_CRUDE_OIL: tuple[tuple[float, float, float], ...] = (
    (0.003, 0.002, 0.002),
    (0.003, 0.002, 0.002),
    (0.004, 0.003, 0.002),
    (0.005, 0.004, 0.003),
)
#: Temperature bin edges (degF) and velocity bin edges (ft/s) of :data:`TEMA_CRUDE_OIL`.
TEMA_CRUDE_T_EDGES_F = (200.0, 300.0, 500.0)
TEMA_CRUDE_V_EDGES_FTS = (2.0, 4.0)

#: TEMA cooling-water fouling resistances (h ft^2 degF/Btu): for each water,
#: ``(T_w <= 125 F, v <= 3 ft/s; T_w <= 125 F, v > 3 ft/s; T_w > 125 F, v <= 3;
#: T_w > 125 F, v > 3)``. The two temperature columns are TEMA's "heating
#: medium up to 240 F" (water 125 F or less) and "240-400 F" (water over
#: 125 F). Same source and provenance note as :data:`TEMA_FOULING`.
TEMA_WATER: dict[str, tuple[float, float, float, float]] = {
    "sea_water": (0.0005, 0.0005, 0.001, 0.001),
    "brackish_water": (0.002, 0.001, 0.003, 0.002),
    "cooling_tower_treated": (0.001, 0.001, 0.002, 0.002),
    "cooling_tower_untreated": (0.003, 0.003, 0.005, 0.004),
    "city_or_well_water": (0.001, 0.001, 0.002, 0.002),
    "river_water_minimum": (0.002, 0.001, 0.003, 0.002),
    "river_water_average": (0.003, 0.002, 0.004, 0.003),
    "muddy_or_silty_water": (0.003, 0.002, 0.004, 0.003),
    "hard_water": (0.003, 0.003, 0.005, 0.005),
    "engine_jacket_water": (0.001, 0.001, 0.001, 0.001),
    "distilled_or_closed_cycle_condensate": (0.0005, 0.0005, 0.0005, 0.0005),
    "treated_boiler_feedwater": (0.001, 0.0005, 0.001, 0.001),
    "boiler_blowdown": (0.002, 0.002, 0.002, 0.002),
}

_FT = 0.3048


def _F_to_K(T_F: float) -> float:
    return (T_F - 32.0) * 5.0 / 9.0 + 273.15


def tema_fouling_resistance(service: str, which: str = "high") -> float:
    """TEMA design fouling resistance (m^2 K/W) of a service in :data:`TEMA_FOULING`.

    Args:
        service: Key of :data:`TEMA_FOULING`.
        which: ``"high"`` (default; the conservative end of a range),
            ``"low"`` or ``"mean"``.

    Returns:
        R_f (m^2 K/W), a Python float.

    Raises:
        KeyError: Unknown service (the message lists the keys).
        ValueError: Unknown ``which``.

    Example:
        >>> from difflow.fouling import tema_fouling_resistance
        >>> round(tema_fouling_resistance("fuel_oil_no6"), 6)   # 0.005 h ft2 F/Btu
        0.000881
    """
    try:
        lo, hi = TEMA_FOULING[service]
    except KeyError:
        raise KeyError(f"Unknown service {service!r}; use one of {sorted(TEMA_FOULING)}") from None
    if which == "high":
        v = hi
    elif which == "low":
        v = lo
    elif which == "mean":
        v = 0.5 * (lo + hi)
    else:
        raise ValueError(f"which must be 'high', 'low' or 'mean', got {which!r}")
    return v * HFT2F_PER_BTU


def tema_crude_oil_fouling(T: Array | float, v: Array | float) -> Array:
    """TEMA crude-oil fouling resistance (m^2 K/W) at bulk temperature ``T`` and velocity ``v``.

    Looks up :data:`TEMA_CRUDE_OIL`. Bin edges are converted from the TEMA
    units: 200, 300, 500 degF = 366.48, 422.04, 533.15 K; 2 and 4 ft/s =
    0.6096 and 1.2192 m/s. A value on an edge takes the upper bin. The result
    is piecewise constant (its derivative is zero almost everywhere), but it
    is jit-able and accepts arrays.

    Args:
        T: Bulk crude temperature (K).
        v: Tube-side velocity (m/s).

    Returns:
        R_f (m^2 K/W).

    Example:
        >>> from difflow.fouling import tema_crude_oil_fouling
        >>> round(float(tema_crude_oil_fouling(550.0, 1.0)), 6)   # >= 500 F, 2-4 ft/s: 0.004
        0.000704
    """
    T, v = _f64(T), _f64(v)
    table = jnp.asarray(TEMA_CRUDE_OIL) * HFT2F_PER_BTU
    edges_T = jnp.asarray([_F_to_K(t) for t in TEMA_CRUDE_T_EDGES_F])
    edges_v = jnp.asarray([x * _FT for x in TEMA_CRUDE_V_EDGES_FTS])
    i = jnp.searchsorted(edges_T, T, side="right")
    j = jnp.searchsorted(edges_v, v, side="right")
    return table[i, j]


def tema_water_fouling(water: str, T_water: Array | float, v: Array | float) -> Array:
    """TEMA cooling-water fouling resistance (m^2 K/W) of a water in :data:`TEMA_WATER`.

    The temperature split is at 125 degF (324.82 K) water temperature, the
    velocity split at 3 ft/s (0.9144 m/s); as in TEMA's "125 F or less" and
    "3 ft/s and less" columns, a value on an edge takes the lower bin.
    Piecewise constant, jit-able.

    Args:
        water: Key of :data:`TEMA_WATER`.
        T_water: Water temperature (K).
        v: Water velocity (m/s).

    Returns:
        R_f (m^2 K/W).

    Raises:
        KeyError: Unknown water (the message lists the keys).

    Example:
        >>> from difflow.fouling import tema_water_fouling
        >>> round(float(tema_water_fouling("sea_water", 300.0, 1.5)), 7)   # 0.0005 h ft2 F/Btu
        8.81e-05
    """
    try:
        row = TEMA_WATER[water]
    except KeyError:
        raise KeyError(f"Unknown water {water!r}; use one of {sorted(TEMA_WATER)}") from None
    table = jnp.asarray(row).reshape(2, 2) * HFT2F_PER_BTU
    i = (_f64(T_water) > _F_to_K(125.0)).astype(jnp.int32)
    j = (_f64(v) > 3.0 * _FT).astype(jnp.int32)
    return table[i, j]


__all__ = [
    "HFT2F_PER_BTU",
    "TEMA_FOULING",
    "TEMA_CRUDE_OIL",
    "TEMA_CRUDE_T_EDGES_F",
    "TEMA_CRUDE_V_EDGES_FTS",
    "TEMA_WATER",
    "tema_fouling_resistance",
    "tema_crude_oil_fouling",
    "tema_water_fouling",
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
