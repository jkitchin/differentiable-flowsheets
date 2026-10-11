"""Humid-air (psychrometric) properties and processes, differentiable.

Pure JAX, SI units: ``T`` in K, ``P`` in Pa, enthalpy in J per kg of **dry
air**, humidity ratio ``W`` in kg water per kg dry air, specific volume in m^3
per kg dry air. Everything is a smooth function of its inputs, so
``jax.grad`` / ``jax.jacobian`` work through every function here, including
the wet-bulb and dew-point inversions (``optimistix`` Newton with implicit
differentiation). There is no Python control flow on traced values; the only
Python-level branching is on *which* keyword arguments :func:`state` is given.

Model
-----
Ideal-gas mixture of dry air and water vapor, with the ASHRAE
*Handbook-Fundamentals* (Ch. 1) constants::

    W    = 0.621945 p_w / (P - p_w)
    h    = c_pa t + W (h_fg0 + c_pv t)            t in C
    v    = R_da T (1 + W / 0.621945) / P

with ``c_pa = 1006``, ``c_pv = 1860``, ``h_fg0 = 2501e3`` and ``R_da =
287.042``. The saturation pressure ``p_ws`` is IAPWS-IF97 over liquid water
above the triple point and the IAPWS-2008 sublimation formula over ice below
it, joined by a smooth blend (see :mod:`difflow.water`). **No enhancement
factor** is applied (real-gas correction, about 0.4 % in ``W_s`` at 1 atm);
tabulated ASHRAE ``W_s`` values therefore sit 0.3-0.5 % above this module's.

Wet bulb and adiabatic saturation
---------------------------------
*Adiabatic saturation* is the exact enthalpy balance of an adiabatic
saturator whose make-up liquid enters at the exit temperature ``T*``::

    h(T, W) = h_s(T*) - (W_s(T*) - W) h_liq(T*)          h_liq = c_pl t*

The *wet-bulb* temperature of a psychrometer comes from the heat/mass-transfer
balance on the wet wick ``h_c (T - T*) = k_y (W_s* - W) h_fg*``::

    Le_f (c_pa + W c_pv)(T - T*) = (W_s(T*) - W) h_fg(T*),
    Le_f = h_c / (k_y c_s)

where ``lewis`` is the Lewis *factor* ``Le_f`` (the psychrometer
coefficient). ``Le_f = 1`` is the Lewis relation, which for air-water is good
to a fraction of a kelvin: the two temperatures then coincide **exactly** (the
two equations are algebraically identical when ``h_fg = h_fg0 + (c_pv - c_pl)
t``). Below the triple point the wick carries ice, and the latent heat
includes the heat of fusion through the same smooth ice weight as the
saturation pressure (heat of fusion recalled, unverified).

Air stream convention
---------------------
Humid air flows through a flowsheet as an ordinary difflow stream with the
species ``H2O`` (vapor) and ``air`` (dry air as one pseudo-species, molar mass
28.966 g/mol): ``{"F_H2O", "F_air", "T", "P"}`` in mol/s, K, Pa.
:func:`air_stream_from_state` builds it from a state and a dry-air mass flow
(kg/s); :func:`state_from_air_stream` reads the state back. Any non-``H2O``
species of a stream is taken as dry air, so an ``N2``/``O2`` stream also
works. The flowsheet's ``species_order`` for such a loop is ``["H2O",
"air"]``.

State points
------------
A *state* is a plain ``dict`` (a pytree) with the keys ``T, W, RH, T_wb,
T_dp, h, v, P, p_w, W_sat, mu`` (``mu`` is the degree of saturation
``W / W_s``). :func:`state` builds one from any two independent properties.

References
----------
ASHRAE Handbook-Fundamentals, Ch. 1 (Psychrometrics); Hyland & Wexler (1983)
for the saturation-pressure background; Wagner et al., J. Phys. Chem. Ref.
Data 40, 023103 (2011) for the ice line; Geankoplis, *Transport Processes
and Separation Process Principles*, Ch. 9; McCabe, Smith & Harriott, *Unit
Operations of Chemical Engineering*, Ch. 19.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp
import numpy as np
import optimistix as optx
from jax import Array

from difflow.water import (
    H_FUSION,
    T_REF,
    ice_weight,
    water_saturation_temperature,
    water_vapor_pressure,
)

#: Standard atmosphere (Pa)
P_ATM = 101325.0
#: Molar-mass ratio water / dry air (ASHRAE)
EPS = 0.621945
#: Specific heat of dry air, J/kg/K
CP_DRY_AIR = 1006.0
#: Specific heat of water vapor, J/kg/K
CP_VAPOR = 1860.0
#: Specific heat of liquid water, J/kg/K
CP_LIQUID = 4186.0
#: Latent heat of vaporization of water at 0 C (the ASHRAE form), J/kg
H_FG0 = 2501.0e3
#: Specific gas constant of dry air, J/kg/K
R_DRY_AIR = 287.042
#: Molar mass of water, kg/mol
MW_WATER = 18.01528e-3
#: Molar mass of dry air, kg/mol (so that MW_WATER / MW_DRY_AIR = EPS)
MW_DRY_AIR = MW_WATER / EPS

#: Names of the state keys, in the order :func:`state` fills them
STATE_KEYS = ("T", "W", "RH", "T_wb", "T_dp", "h", "v", "P", "p_w", "W_sat", "mu")

_INPUTS = ("T", "W", "RH", "T_wb", "T_dp", "h")


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


def _w_ice(T, over_ice):
    if over_ice is None:
        return ice_weight(T)
    return jnp.where(bool(over_ice), 1.0, 0.0) * jnp.ones_like(_f(T))


def _newton(fn, x0, *args):
    """Scalar Newton root of ``fn(x, *args) = 0`` with implicit differentiation.

    ``fn`` must be scaled to be O(1) per kelvin of error so the fixed
    tolerances mean the same thing everywhere.
    """
    sol = optx.root_find(
        lambda x, a: fn(x, *a),
        optx.Newton(rtol=1e-12, atol=1e-10),
        _f(x0),
        args=args,
        max_steps=60,
        throw=False,
    )
    return sol.value


# =============================================================================
# Vapor pressure, humidity ratio, relative humidity
# =============================================================================


def saturation_pressure(T: Array | float, over_ice: bool | None = None) -> Array:
    """Saturation vapor pressure of water ``p_ws`` (Pa) at ``T`` (K).

    Over liquid above the triple point and over ice below (smooth blend over
    +-0.25 K); see :func:`difflow.water.water_vapor_pressure`. ``over_ice``:
    ``None`` automatic, ``True`` / ``False`` force ice / liquid.
    """
    return water_vapor_pressure(T, over_ice)


def humidity_ratio(p_w: Array | float, P: Array | float = P_ATM) -> Array:
    """Humidity ratio ``W = 0.621945 p_w / (P - p_w)`` (kg water / kg dry air).

    Args:
        p_w: Partial pressure of water vapor (Pa).
        P: Total pressure (Pa).
    """
    p_w = _f(p_w)
    return EPS * p_w / (_f(P) - p_w)


def vapor_pressure_from_humidity_ratio(W: Array | float, P: Array | float = P_ATM) -> Array:
    """Partial pressure of water vapor (Pa) for humidity ratio ``W``: ``P W / (0.621945 + W)``."""
    W = _f(W)
    return _f(P) * W / (EPS + W)


def saturation_humidity_ratio(T: Array | float, P: Array | float = P_ATM,
                              over_ice: bool | None = None) -> Array:
    """Humidity ratio of saturated air ``W_s(T, P)`` (kg/kg).

    Uses ``p_ws`` over liquid water, or over ice below 273.16 K (smooth
    blend; ``over_ice`` forces one). No enhancement factor.
    """
    return humidity_ratio(saturation_pressure(T, over_ice), P)


def humidity_ratio_from_rh(T: Array | float, RH: Array | float, P: Array | float = P_ATM,
                           over_ice: bool | None = None) -> Array:
    """Humidity ratio from relative humidity ``RH`` (0-1), ``W = 0.621945 RH p_ws / (P - RH p_ws)``."""
    return humidity_ratio(_f(RH) * saturation_pressure(T, over_ice), P)


def rh_from_humidity_ratio(T: Array | float, W: Array | float, P: Array | float = P_ATM,
                           over_ice: bool | None = None) -> Array:
    """Relative humidity ``p_w / p_ws`` (0-1) of air at ``T`` with humidity ratio ``W``."""
    return vapor_pressure_from_humidity_ratio(W, P) / saturation_pressure(T, over_ice)


def _T_from_psat(p: Array, over_ice: bool | None = None) -> Array:
    """Temperature (K) at which the saturation pressure equals ``p`` (Pa)."""
    p = _f(p)
    # IF97 backward equation: exact over liquid, a good seed over ice
    T0 = jnp.clip(water_saturation_temperature(jnp.maximum(p, 1.0)), 180.0, 640.0)

    def res(T, p_, over):
        # about one kelvin of error per unit
        return (jnp.log(saturation_pressure(T, over)) - jnp.log(p_)) * 15.0

    over = over_ice  # static flag, kept out of the traced args
    sol = optx.root_find(
        lambda T, a: res(T, a, over),
        optx.Newton(rtol=1e-12, atol=1e-10),
        T0,
        args=p,
        max_steps=60,
        throw=False,
    )
    return sol.value


def dew_point(W: Array | float, P: Array | float = P_ATM, over_ice: bool | None = None) -> Array:
    """Dew-point temperature (K) of air with humidity ratio ``W``.

    The temperature at which ``p_ws`` equals the vapor partial pressure
    (frost point below the triple point). Closed form over liquid water, via
    :func:`difflow.water.water_saturation_temperature`, with one implicit
    Newton solve that makes it exact and differentiable over ice too.
    """
    return _T_from_psat(vapor_pressure_from_humidity_ratio(W, P), over_ice)


# =============================================================================
# Enthalpy, heat capacity, volume
# =============================================================================


def humid_enthalpy(T: Array | float, W: Array | float) -> Array:
    """Specific enthalpy of humid air (J/kg dry air), ASHRAE form.

    ``h = c_pa t + W (h_fg0 + c_pv t)`` with ``t = T - 273.15`` and the
    reference state dry air and liquid water at 0 C.
    """
    t = _f(T) - T_REF
    return CP_DRY_AIR * t + _f(W) * (H_FG0 + CP_VAPOR * t)


def humid_heat(W: Array | float) -> Array:
    """Humid heat ``c_s = c_pa + W c_pv`` (J/kg dry air/K)."""
    return CP_DRY_AIR + _f(W) * CP_VAPOR


def humid_volume(T: Array | float, W: Array | float, P: Array | float = P_ATM) -> Array:
    """Specific volume of humid air (m^3/kg dry air), ``R_da T (1 + W/0.621945) / P``."""
    return R_DRY_AIR * _f(T) * (1.0 + _f(W) / EPS) / _f(P)


def saturation_enthalpy(T: Array | float, P: Array | float = P_ATM,
                        over_ice: bool | None = None) -> Array:
    """Enthalpy of saturated air ``h_s(T, P)`` (J/kg dry air); the Merkel driving-force curve."""
    return humid_enthalpy(T, saturation_humidity_ratio(T, P, over_ice))


def humid_temperature(h: Array | float, W: Array | float) -> Array:
    """Dry-bulb temperature (K) from enthalpy and humidity ratio (closed-form inverse of :func:`humid_enthalpy`)."""
    W = _f(W)
    return T_REF + (_f(h) - W * H_FG0) / (CP_DRY_AIR + W * CP_VAPOR)


def humidity_ratio_from_enthalpy(T: Array | float, h: Array | float) -> Array:
    """Humidity ratio from dry-bulb temperature and enthalpy (closed-form inverse of :func:`humid_enthalpy`)."""
    t = _f(T) - T_REF
    return (_f(h) - CP_DRY_AIR * t) / (H_FG0 + CP_VAPOR * t)


def condensed_enthalpy(T: Array | float, over_ice: bool | None = None) -> Array:
    """Enthalpy of the condensed phase (J/kg): ``c_pl t``, less the heat of fusion on the ice branch."""
    T = _f(T)
    return CP_LIQUID * (T - T_REF) - _w_ice(T, over_ice) * H_FUSION


def latent_heat(T: Array | float, over_ice: bool | None = None) -> Array:
    """Latent heat (J/kg) used by the wet-bulb balance, ``h_fg0 + (c_pv - c_pl) t`` (+ fusion on ice).

    This is the ideal-gas-consistent value ``h_v(t) - h_cond(t)`` (2454.5 kJ/kg at
    20 C against the steam-table 2453.5).
    """
    T = _f(T)
    return H_FG0 + CP_VAPOR * (T - T_REF) - condensed_enthalpy(T, over_ice)


# =============================================================================
# Wet bulb and adiabatic saturation
# =============================================================================


def _wb_residual(Ts, T, W, P, lewis, over_ice):
    """Wet-bulb balance in kelvin: zero at the wet-bulb temperature ``Ts``."""
    cs = humid_heat(W)
    return (T - Ts) - (saturation_humidity_ratio(Ts, P, over_ice) - W) * latent_heat(Ts, over_ice) / (lewis * cs)


def wet_bulb_temperature(T: Array | float, W: Array | float, P: Array | float = P_ATM,
                         lewis: Array | float = 1.0, over_ice: bool | None = None) -> Array:
    """Psychrometer wet-bulb temperature (K) of air at ``T`` with humidity ratio ``W``.

    Solves ``Le_f c_s (T - T*) = (W_s(T*) - W) h_fg(T*)`` by Newton from the
    dry-bulb temperature (the iteration is monotone from the right for
    ``W <= W_s(T)``), differentiable by the implicit function theorem.

    Args:
        T: Dry-bulb temperature (K).
        W: Humidity ratio (kg/kg); the result is meaningful for ``W <= W_s(T)``.
        P: Total pressure (Pa).
        lewis: Lewis *factor* ``h_c / (k_y c_s)``; 1 is the Lewis relation, for
            which the wet bulb equals the adiabatic-saturation temperature.
        over_ice: Force the ice (``True``) or liquid (``False``) saturation
            line; ``None`` switches at the triple point.
    """
    T, W, P, lewis = _f(T), _f(W), _f(P), _f(lewis)

    def res(Ts, T_, W_, P_, le_):
        return _wb_residual(Ts, T_, W_, P_, le_, over_ice)

    # a fog (W > W_s) has no wet bulb below the dry bulb; the seed keeps the iteration finite there
    return _newton(res, T, T, W, P, lewis)


def adiabatic_saturation_temperature(T: Array | float, W: Array | float, P: Array | float = P_ATM,
                                     over_ice: bool | None = None) -> Array:
    """Adiabatic-saturation temperature (K): air at ``(T, W)`` saturated adiabatically by liquid at that temperature.

    Solves ``h(T, W) = h_s(T*) - (W_s(T*) - W) h_liq(T*)``. Equal to
    :func:`wet_bulb_temperature` with ``lewis = 1`` for the ideal-gas model of
    this module.
    """
    T, W, P = _f(T), _f(W), _f(P)

    def res(Ts, T_, W_, P_):
        Ws = saturation_humidity_ratio(Ts, P_, over_ice)
        h_target = saturation_enthalpy(Ts, P_, over_ice) - (Ws - W_) * condensed_enthalpy(Ts, over_ice)
        return (humid_enthalpy(T_, W_) - h_target) / humid_heat(W_)

    return _newton(res, T, T, W, P)


# =============================================================================
# State points
# =============================================================================


def _assemble(T, W, P, lewis, over_ice) -> dict:
    """All properties of a state from its dry-bulb temperature and humidity ratio."""
    T, W, P = _f(T), _f(W), _f(P)
    p_ws = saturation_pressure(T, over_ice)
    p_w = vapor_pressure_from_humidity_ratio(W, P)
    W_sat = saturation_humidity_ratio(T, P, over_ice)
    return {
        "T": T,
        "W": W,
        "RH": p_w / p_ws,
        "T_wb": wet_bulb_temperature(T, W, P, lewis, over_ice),
        "T_dp": dew_point(W, P, over_ice),
        "h": humid_enthalpy(T, W),
        "v": humid_volume(T, W, P),
        "P": P,
        "p_w": p_w,
        "W_sat": W_sat,
        "mu": W / W_sat,
    }


def state(T=None, W=None, RH=None, T_wb=None, T_dp=None, h=None, P=P_ATM,
          lewis: float = 1.0, over_ice: bool | None = None) -> dict:
    """Fix a humid-air state point from any two independent properties.

    Exactly two of ``T`` (dry bulb, K), ``W`` (kg/kg), ``RH`` (0-1), ``T_wb``
    (wet bulb, K), ``T_dp`` (dew point, K) and ``h`` (J/kg dry air) must be
    given; the total pressure ``P`` (Pa) is the third state variable. ``W`` and
    ``T_dp`` are not independent (``T_dp`` fixes ``W`` at a given ``P``), so
    that pair raises ``ValueError``.

    Args:
        T, W, RH, T_wb, T_dp, h: The two known properties (the others ``None``).
        P: Total pressure (Pa).
        lewis: Lewis factor for the wet bulb (1 = Lewis relation).
        over_ice: Force the ice/liquid saturation line (``None`` automatic).

    Returns:
        A dict (a pytree) with ``T, W, RH, T_wb, T_dp, h, v, P, p_w, W_sat,
        mu``. Closed forms are used for the pairs that have one (``T``-``W``,
        ``T``-``RH``, ``T``-``T_wb``, ``T``-``T_dp``, ``T``-``h``, ``W``-``h``,
        ``W``-``RH``, ``W``-``T_wb``, ``RH``-``T_dp``, ``T_wb``-``T_dp``,
        ``T_dp``-``h``); ``RH``-``T_wb``, ``RH``-``h`` and ``T_wb``-``h`` take
        one implicit scalar Newton solve. All pairs are differentiable.

    Example:
        >>> from difflow.psychrometrics import state
        >>> s = state(T=303.15, RH=0.5)
        >>> round(float(s["T_dp"]) - 273.15, 1)
        18.4
    """
    given = {k: v for k, v in zip(_INPUTS, (T, W, RH, T_wb, T_dp, h)) if v is not None}
    if len(given) != 2:
        raise ValueError(
            f"state() needs exactly two of {_INPUTS}; got {sorted(given) or 'none'}")
    P = _f(P)
    lewis = _f(lewis)
    keys = frozenset(given)
    g = {k: _f(v) for k, v in given.items()}

    def W_of_Tdp(Td):
        return humidity_ratio(saturation_pressure(Td, over_ice), P)

    def W_of_wb(T_, Twb):
        # (W + ...) linear in W: Le_f (cpa + W cpv)(t - t*) = (Ws* - W) hfg*
        dt = T_ - Twb
        Ws = saturation_humidity_ratio(Twb, P, over_ice)
        hfg = latent_heat(Twb, over_ice)
        return (Ws * hfg - lewis * CP_DRY_AIR * dt) / (hfg + lewis * CP_VAPOR * dt)

    def T_of_wb(W_, Twb):
        Ws = saturation_humidity_ratio(Twb, P, over_ice)
        return Twb + (Ws - W_) * latent_heat(Twb, over_ice) / (lewis * humid_heat(W_))

    def T_from_p(p):
        return _T_from_psat(p, over_ice)

    if keys == {"T", "W"}:
        Tv, Wv = g["T"], g["W"]
    elif keys == {"T", "RH"}:
        Tv = g["T"]
        Wv = humidity_ratio_from_rh(Tv, g["RH"], P, over_ice)
    elif keys == {"T", "T_wb"}:
        Tv = g["T"]
        Wv = W_of_wb(Tv, g["T_wb"])
    elif keys == {"T", "T_dp"}:
        Tv = g["T"]
        Wv = W_of_Tdp(g["T_dp"])
    elif keys == {"T", "h"}:
        Tv = g["T"]
        Wv = humidity_ratio_from_enthalpy(Tv, g["h"])
    elif keys == {"W", "RH"}:
        Wv = g["W"]
        Tv = T_from_p(vapor_pressure_from_humidity_ratio(Wv, P) / g["RH"])
    elif keys == {"W", "T_wb"}:
        Wv = g["W"]
        Tv = T_of_wb(Wv, g["T_wb"])
    elif keys == {"W", "h"}:
        Wv = g["W"]
        Tv = humid_temperature(g["h"], Wv)
    elif keys == {"W", "T_dp"}:
        raise ValueError("W and T_dp are not independent at a given P; give one of them with another property")
    elif keys == {"RH", "T_dp"}:
        p_w = saturation_pressure(g["T_dp"], over_ice)
        Tv = T_from_p(p_w / g["RH"])
        Wv = humidity_ratio(p_w, P)
    elif keys == {"RH", "T_wb"}:
        def res(T_, RH_, Twb_, P_, le_):
            W_ = humidity_ratio_from_rh(T_, RH_, P_, over_ice)
            cs = humid_heat(W_)
            Ws = saturation_humidity_ratio(Twb_, P_, over_ice)
            return (T_ - Twb_) - (Ws - W_) * latent_heat(Twb_, over_ice) / (le_ * cs)
        Tv = _newton(res, g["T_wb"] + 5.0, g["RH"], g["T_wb"], P, lewis)
        Wv = humidity_ratio_from_rh(Tv, g["RH"], P, over_ice)
    elif keys == {"RH", "h"}:
        def res(T_, RH_, h_, P_):
            W_ = humidity_ratio_from_rh(T_, RH_, P_, over_ice)
            return (humid_enthalpy(T_, W_) - h_) / humid_heat(W_)
        Tv = _newton(res, 295.0, g["RH"], g["h"], P)
        Wv = humidity_ratio_from_rh(Tv, g["RH"], P, over_ice)
    elif keys == {"T_wb", "T_dp"}:
        Wv = W_of_Tdp(g["T_dp"])
        Tv = T_of_wb(Wv, g["T_wb"])
    elif keys == {"T_wb", "h"}:
        def res(T_, Twb_, h_, P_, le_):
            W_ = humidity_ratio_from_enthalpy(T_, h_)
            Ws = saturation_humidity_ratio(Twb_, P_, over_ice)
            return (T_ - Twb_) - (Ws - W_) * latent_heat(Twb_, over_ice) / (le_ * humid_heat(W_))
        Tv = _newton(res, g["T_wb"] + 5.0, g["T_wb"], g["h"], P, lewis)
        Wv = humidity_ratio_from_enthalpy(Tv, g["h"])
    else:  # {"T_dp", "h"}
        Wv = W_of_Tdp(g["T_dp"])
        Tv = humid_temperature(g["h"], Wv)
    return _assemble(Tv, Wv, P, lewis, over_ice)


# =============================================================================
# Air streams (the flowsheet convention)
# =============================================================================


def dry_air_flow(stream: dict, water: str = "H2O") -> Array:
    """Dry-air mass flow (kg/s) of a humid-air stream (every non-``water`` species counts as dry air)."""
    return sum(
        (v for k, v in stream.items() if k.startswith("F_") and k != f"F_{water}"),
        _f(0.0),
    ) * MW_DRY_AIR


def air_stream_from_state(st: dict, m_dry_air: Array | float, water: str = "H2O",
                          dry_air: str = "air") -> dict:
    """Humid-air state to a difflow stream.

    Args:
        st: A state from :func:`state` (uses ``T``, ``W``, ``P``).
        m_dry_air: Dry-air mass flow (kg/s).
        water: Name of the water-vapor species.
        dry_air: Name of the dry-air pseudo-species.

    Returns:
        ``{"F_<water>", "F_<dry_air>", "T", "P"}`` in mol/s, K, Pa.
    """
    m = _f(m_dry_air)
    return {
        f"F_{water}": m * st["W"] / MW_WATER,
        f"F_{dry_air}": m / MW_DRY_AIR,
        "T": _f(st["T"]),
        "P": _f(st["P"]),
    }


def state_from_air_stream(stream: dict, water: str = "H2O", lewis: float = 1.0,
                          over_ice: bool | None = None) -> dict:
    """Read the humid-air state of a stream (the inverse of :func:`air_stream_from_state`).

    The humidity ratio is the water mass per mass of everything that is not
    ``water``. A stream with no dry air gives ``W = 0``, not NaN.
    """
    n_dry = dry_air_flow(stream, water) / MW_DRY_AIR
    n_w = _f(stream.get(f"F_{water}", 0.0))
    safe = jnp.where(n_dry > 0.0, n_dry, 1.0)
    W = jnp.where(n_dry > 0.0, n_w * MW_WATER / (safe * MW_DRY_AIR), 0.0)
    return state(T=stream["T"], W=W, P=stream["P"], lewis=lewis, over_ice=over_ice)


# =============================================================================
# Processes
# =============================================================================


def mix_air(states: Sequence[dict], dry_air_flows: Sequence[Array | float]) -> dict:
    """Adiabatic mixing of humid-air streams.

    Dry air, water and enthalpy are conserved exactly: ``W`` and ``h`` of the
    mixture are the dry-air-flow-weighted averages, and ``T`` follows from
    them. The mixture pressure is the flow-weighted mean pressure.

    Args:
        states: States (dicts with ``T``, ``W``, ``P``).
        dry_air_flows: Dry-air mass flow of each (kg/s, or any consistent weight).

    Returns:
        The mixed state (see :func:`state`).
    """
    m = jnp.stack([_f(x) for x in dry_air_flows])
    tot = jnp.sum(m)
    W = jnp.sum(m * jnp.stack([_f(s["W"]) for s in states])) / tot
    hh = jnp.sum(m * jnp.stack([humid_enthalpy(s["T"], s["W"]) for s in states])) / tot
    P = jnp.sum(m * jnp.stack([_f(s["P"]) for s in states])) / tot
    return state(T=humid_temperature(hh, W), W=W, P=P)


def sensible_heat(st: dict, Q: Array | float, m_dry_air: Array | float = 1.0) -> dict:
    """Heat (``Q > 0``) or cool (``Q < 0``) at constant humidity ratio.

    Args:
        st: Entering state.
        Q: Heat added (W) for a dry-air flow ``m_dry_air`` (kg/s); with the
            default ``m_dry_air = 1`` it is J per kg dry air.
    """
    h = humid_enthalpy(st["T"], st["W"]) + _f(Q) / _f(m_dry_air)
    return state(T=humid_temperature(h, st["W"]), W=st["W"], P=st["P"])


def cool_dehumidify(st: dict, T_coil: Array | float, bypass_factor: Array | float = 0.0,
                    over_ice: bool | None = None) -> tuple[dict, dict]:
    """Cooling-and-dehumidifying coil with an apparatus dew point.

    The leaving air lies on the straight line (in ``W``-``h``) from the
    entering state to the saturated air at the coil temperature (the
    apparatus dew point): ``x_out = BF x_in + (1 - BF) x_sat(T_coil)``. When
    the leaving humidity would exceed the entering one (a coil warmer than the
    dew point) it simply cools sensibly, since ``W_out`` is clipped to
    ``W_in``.

    Args:
        st: Entering state.
        T_coil: Coil (apparatus dew point) temperature (K).
        bypass_factor: Fraction of the air that bypasses the coil (0-1).

    Returns:
        ``(out, info)``; ``info`` holds, per kg dry air, ``Q_total``,
        ``Q_sensible``, ``Q_latent`` (J/kg, positive = removed) and
        ``condensate`` (kg/kg), with the condensate leaving at ``T_coil``.
    """
    P = st["P"]
    bf = _f(bypass_factor)
    T_coil = _f(T_coil)
    W_s = saturation_humidity_ratio(T_coil, P, over_ice)
    h_s = humid_enthalpy(T_coil, W_s)
    h_in = humid_enthalpy(st["T"], st["W"])
    W_out = jnp.minimum(bf * st["W"] + (1.0 - bf) * W_s, st["W"])
    h_out = bf * h_in + (1.0 - bf) * h_s
    # when W is clipped, move along constant W instead (pure sensible cooling)
    clipped = bf * st["W"] + (1.0 - bf) * W_s > st["W"]
    T_sens = T_coil + bf * (st["T"] - T_coil)
    h_out = jnp.where(clipped, humid_enthalpy(T_sens, st["W"]), h_out)
    out = state(T=humid_temperature(h_out, W_out), W=W_out, P=P, over_ice=over_ice)
    cond = st["W"] - W_out
    Q_total = h_in - h_out - cond * condensed_enthalpy(T_coil, over_ice)
    # latent part: the condensate's latent heat at the entering temperature
    Q_latent = cond * (H_FG0 + CP_VAPOR * (st["T"] - T_REF))
    info = {"Q_total": Q_total, "Q_latent": Q_latent, "Q_sensible": Q_total - Q_latent,
            "condensate": cond, "T_coil": T_coil, "bypass_factor": bf}
    return out, info


def adiabatic_humidify(st: dict, effectiveness: Array | float = 1.0,
                       W_out: Array | float | None = None) -> dict:
    """Evaporative (adiabatic) humidification toward saturation.

    Air moves along its adiabatic-saturation line: ``h`` is constant apart
    from the make-up water enthalpy (liquid at the adiabatic-saturation
    temperature of the entering air).

    Args:
        st: Entering state.
        effectiveness: Fraction of the way to saturation, in humidity ratio,
            ``(W_out - W_in) / (W_s* - W_in)``; ignored when ``W_out`` is given.
        W_out: Leaving humidity ratio (kg/kg); overrides ``effectiveness``.
    """
    P = st["P"]
    Tas = adiabatic_saturation_temperature(st["T"], st["W"], P)
    Ws = saturation_humidity_ratio(Tas, P)
    Wo = st["W"] + _f(effectiveness) * (Ws - st["W"]) if W_out is None else _f(W_out)
    h = humid_enthalpy(st["T"], st["W"]) + (Wo - st["W"]) * condensed_enthalpy(Tas)
    return state(T=humid_temperature(h, Wo), W=Wo, P=P)


# =============================================================================
# Chart data
# =============================================================================


def _masked(T, W, P, over_ice=None):
    """NaN-mask a line to the physical region ``0 <= W <= W_s(T)``."""
    Ws = np.asarray(saturation_humidity_ratio(T, P, over_ice))
    W = np.asarray(W, dtype=float)
    ok = (W >= -1e-12) & (W <= Ws * (1.0 + 1e-9))
    return np.where(ok, np.clip(W, 0.0, None), np.nan)


def psychrometric_chart_data(
    T_range: tuple[float, float] = (258.15, 313.15),
    P: float = P_ATM,
    rh_lines: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    wb_lines: Sequence[float] | None = None,
    h_lines: Sequence[float] | None = None,
    v_lines: Sequence[float] | None = None,
    n: int = 121,
    lewis: float = 1.0,
) -> dict:
    """Data for a psychrometric chart: ``W`` against dry-bulb ``T``.

    Every family is a list of ``{"value", "T", "W"}`` dicts of NumPy arrays
    (K, kg/kg) with NaN where the line leaves the physical region
    ``0 <= W <= W_s(T)``, so they can be plotted directly. The saturation
    curve is the ``rh = 1.0`` line and is also returned as ``"saturation"``.

    Args:
        T_range: Dry-bulb range (K).
        P: Total pressure (Pa).
        rh_lines: Relative humidities (0-1).
        wb_lines: Wet-bulb temperatures (K); ``None`` gives every 5 K inside the range.
        h_lines: Enthalpies (J/kg dry air); ``None`` gives every 10 kJ/kg.
        v_lines: Specific volumes (m^3/kg dry air); ``None`` gives 0.01 steps
            across the range.
        n: Points per line.
        lewis: Lewis factor of the wet-bulb lines.

    Returns:
        A dict with ``P``, ``T_range``, ``saturation``, ``rh``, ``wb``, ``h``
        and ``v``.
    """
    T = np.linspace(T_range[0], T_range[1], n)
    Tj = jnp.asarray(T)
    ice = None
    if wb_lines is None:
        lo = np.ceil((T_range[0] - T_REF) / 5.0) * 5.0 + T_REF
        wb_lines = np.arange(lo, T_range[1] - 1.0, 5.0)
    if h_lines is None:
        h_hi = float(humid_enthalpy(T_range[1], saturation_humidity_ratio(T_range[1], P)))
        h_lo = float(humid_enthalpy(T_range[0], 0.0))
        h_lines = np.arange(np.ceil(h_lo / 1e4) * 1e4, h_hi, 1e4)
    if v_lines is None:
        v_lo = float(humid_volume(T_range[0], 0.0, P))
        v_hi = float(humid_volume(T_range[1], saturation_humidity_ratio(T_range[1], P), P))
        v_lines = np.arange(np.ceil(v_lo / 0.01) * 0.01, v_hi, 0.01)

    def line(value, W):
        return {"value": float(value), "T": T, "W": _masked(Tj, W, P, ice)}

    rh = [line(r, humidity_ratio_from_rh(Tj, r, P)) for r in rh_lines]
    wb = []
    for Twb in wb_lines:
        Ws = saturation_humidity_ratio(Twb, P)
        hfg = latent_heat(Twb)
        dt = Tj - Twb
        W = (Ws * hfg - lewis * CP_DRY_AIR * dt) / (hfg + lewis * CP_VAPOR * dt)
        wb.append(line(Twb, W))
    # a wet-bulb line only exists to the right of its saturation point
    for d in wb:
        d["W"] = np.where(d["T"] >= d["value"] - 1e-9, d["W"], np.nan)
    hh = [line(hv, humidity_ratio_from_enthalpy(Tj, hv)) for hv in h_lines]
    vv = [line(vv_, EPS * (vv_ * P / (R_DRY_AIR * Tj) - 1.0)) for vv_ in v_lines]
    sat = {"value": 1.0, "T": T, "W": np.asarray(saturation_humidity_ratio(Tj, P))}
    return {"P": float(P), "T_range": tuple(T_range), "saturation": sat,
            "rh": rh, "wb": wb, "h": hh, "v": vv}
