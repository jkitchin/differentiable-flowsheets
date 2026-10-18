"""Counterflow cooling tower: the Merkel equation, in rating and design modes.

A cooling tower cools process water by contacting it with air in a packed
fill. Merkel (1925) combined the sensible and latent transfer into one driving
force, the enthalpy difference between the air at the water surface (saturated
at the local water temperature) and the bulk air::

    KaV/L = int_{T_out}^{T_in} c_pw dT / (h_s(T) - h_a(T))

The air enthalpy follows the operating line fixed by the energy balance,
``h_a(T) = h_a,in + (L/G) c_pw (T - T_out)``, so the integral depends on
``T_in``, ``T_out``, the air inlet enthalpy (that is, the wet bulb) and the
ratio ``L/G`` of water to dry-air mass flow. ``KaV/L`` (dimensionless, the
*Merkel number*, or NTU) is the size of the fill.

Modes
-----
``mode="design"``: the water outlet temperature (``T_out``, or ``approach``
above the wet bulb) is given and ``KaV/L`` is computed by quadrature.

``mode="rating"``: ``KaV/L`` is given (directly, as ``Ka`` and ``V``, or from
the fill characteristic ``C (L/G)^-n``) and the outlet temperature is found by
a bisection with an implicit-function-theorem gradient.

Quadrature
----------
``quadrature="cti"`` is the Cooling Technology Institute four-point method,
the industry standard (CTI ATC-105): the integrand ``1/(h_s - h_a)`` is
averaged at 10, 40, 60 and 90 % of the range and multiplied by
``c_pw (T_in - T_out)`` (its "Chebyshev" integration). ``quadrature="gauss"`` is
Gauss-Legendre with ``n_quad`` points on the range, accurate where the
driving force gets small (a close approach).

Feasibility
-----------
Where ``L/G`` is too high for the approach the operating line touches the
saturation curve, ``h_s - h_a`` reaches zero and the integral diverges (an
infinite fill). The unit checks the driving force on a 65-point grid and
returns ``KaV/L = inf`` with ``info["feasible"] = False`` rather than a
negative or NaN number; every other output stays finite and the gradients
stay finite (the infeasible branch is masked before the quadrature).

Air and water streams
---------------------
Water is a stream with ``F_H2O`` (mol/s). Humid air is a stream with ``F_H2O``
(vapor) and ``F_air`` (dry air, one pseudo-species); build one with
:func:`difflow.psychrometrics.air_stream_from_state` and read it with
:func:`~difflow.psychrometrics.state_from_air_stream`. Without an air stream
the tower is given the wet bulb ``T_wb`` and ``L/G`` and takes the inlet air
to be saturated at the wet bulb, as the textbook Merkel examples do.

Merkel's assumptions
--------------------
The Lewis factor is 1; the air leaving is saturated at the exit enthalpy; the
water flow is constant in the enthalpy balance (the evaporated water, about 1-2 %
of ``L``, is neglected in ``h_a`` and added back in the loss bookkeeping). The
Poppe method and the e-NTU model remove the last two and are not included.

Air-side pressure drop and fan power
------------------------------------
The fan moves the air against the total air-side pressure drop, the fill plus
the inlet louvres, rain zone, sprays, drift eliminator and stack::

    dp_total = dp_fill + dp_other,        W_fan = V_dot dp_total / eta_fan

where ``V_dot`` (m^3/s) is the volumetric air flow at the fan, ``G v`` with
``v`` the humid volume (m^3/kg dry air) of the air the fan handles: the
leaving air for an induced-draft fan (``fan_location="induced"``, the
default), the entering air for a forced-draft one. ``W_fan`` is the air power
``V_dot dp_total`` divided by the efficiency (shaft power for a fan
efficiency, electrical power for an overall fan-motor-drive efficiency).

The fill pressure drop is an input: either directly (``fill_pressure_drop``,
Pa) or as a loss coefficient ``K_fill`` in velocity heads of the frontal air
flow (``fill_loss_coefficient`` with ``frontal_area``)::

    dp_fill = K_fill G_m^2 / (2 rho_in),   G_m = G (1 + W_in) / A_fr,   rho_in = (1 + W_in) / v_in

which is only the definition of a loss coefficient: ``dp_fill`` scales as the
square of the air rate, so the ``L/G`` trade-off (less air, a bigger fill)
reaches the fan. ``K_fill`` comes from fill test data. No published fill
correlation is built in. The intended one is the Kloppers & Kroger
splash/trickle/film fill loss-coefficient correlation (Kloppers, J.C., Kroger,
D.G., "Loss coefficient correlation for wet-cooling tower fills", Appl. Therm.
Eng. 23(17), 2201-2211, 2003, doi:10.1016/S1359-4311(03)00201-1; also Kroger,
*Air-Cooled Heat Exchangers and Cooling Towers*, PennWell, 2004), but its
equation and constants could not be read when this was written (neither the
paper nor the book was reachable), and this module takes no constant from
memory. The default ``fan_efficiency = 0.65`` is the design assumption of the
difflow book (Ch. 18, sec18-design), not a vendor or literature value; set it
from the fan curve. Without ``fill_pressure_drop``, ``fill_loss_coefficient``
or ``other_pressure_drop`` the unit reports no pressure drop or fan power, as
before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.psychrometrics import (
    CP_LIQUID,
    MW_DRY_AIR,
    MW_WATER,
    P_ATM,
    air_stream_from_state,
    dry_air_flow,
    humid_enthalpy,
    humid_volume,
    saturation_enthalpy,
    saturation_humidity_ratio,
    state,
    state_from_air_stream,
)
from difflow.staged import _solve_monotone
from difflow.streams import Stream

#: CTI four-point Chebyshev abscissae (fractions of the range from the cold end)
CTI_POINTS = (0.1, 0.4, 0.6, 0.9)

#: Points of the grid on which the driving force is checked for a pinch
_N_GRID = 65

#: A large finite stand-in for ``ln KaV/L`` where the tower is infeasible (keeps the rating bisection monotone)
_LN_BIG = 60.0


def _f(x) -> Array:
    return jnp.asarray(x, dtype=jnp.float64)


# =============================================================================
# The Merkel integral
# =============================================================================


def _nodes(method: str, n_quad: int):
    """Quadrature nodes and weights on [0, 1] (NumPy; static)."""
    if method == "cti":
        return np.array(CTI_POINTS), np.full(4, 0.25)
    if method == "gauss":
        x, w = np.polynomial.legendre.leggauss(int(n_quad))
        return 0.5 * (x + 1.0), 0.5 * w
    raise ValueError("quadrature must be 'cti' or 'gauss'")


def _driving_force(x, T_in, T_out, h_a_in, L_over_G, P, cp_w):
    """``h_s(T) - h_a(T)`` (J/kg dry air) at fractions ``x`` of the range from the cold end."""
    R = T_in - T_out
    T = T_out + x * R
    h_a = h_a_in + L_over_G * cp_w * R * x
    return saturation_enthalpy(T, P) - h_a


def merkel_min_driving_force(T_in, T_out, h_a_in, L_over_G, P=P_ATM, cp_w=CP_LIQUID) -> Array:
    """Smallest ``h_s - h_a`` (J/kg dry air) over the range, on a 65-point grid.

    Zero or negative means the operating line crosses the saturation curve:
    the tower cannot do this duty at any size.
    """
    x = jnp.linspace(0.0, 1.0, _N_GRID)
    return jnp.min(_driving_force(x, _f(T_in), _f(T_out), _f(h_a_in), _f(L_over_G), _f(P), cp_w))


def merkel_number(T_in, T_out, h_a_in, L_over_G, P=P_ATM, cp_w=CP_LIQUID,
                  quadrature: str = "cti", n_quad: int = 4) -> Array:
    """The Merkel number ``KaV/L = int c_pw dT / (h_s - h_a)``.

    Args:
        T_in: Hot-water temperature (K).
        T_out: Cold-water temperature (K), ``T_out < T_in``.
        h_a_in: Enthalpy of the inlet air (J/kg dry air); for air at the wet
            bulb ``T_wb`` that is ``saturation_enthalpy(T_wb)``.
        L_over_G: Water-to-dry-air mass flow ratio.
        P: Pressure (Pa).
        cp_w: Specific heat of the water (J/kg/K).
        quadrature: ``"cti"`` (the four-point method) or ``"gauss"``.
        n_quad: Gauss points (ignored for ``"cti"``).

    Returns:
        ``KaV/L`` (-), or ``inf`` where the duty is infeasible (the driving
        force reaches zero on the range, or ``T_out >= T_in``).
    """
    T_in, T_out, h_a_in, L_over_G, P = _f(T_in), _f(T_out), _f(h_a_in), _f(L_over_G), _f(P)
    x, w = _nodes(quadrature, n_quad)
    ok = (merkel_min_driving_force(T_in, T_out, h_a_in, L_over_G, P, cp_w) > 0.0) & (T_in > T_out)
    # mask before the quadrature so the infeasible branch cannot produce NaN gradients
    T_o = jnp.where(ok, T_out, T_in - 1.0)
    h_a = jnp.where(ok, h_a_in, saturation_enthalpy(T_in - 1.0, P) - 1.0e3)
    LG = jnp.where(ok, L_over_G, 0.0)
    df = _driving_force(jnp.asarray(x), T_in, T_o, h_a, LG, P, cp_w)
    M = (T_in - T_o) * cp_w * jnp.sum(jnp.asarray(w) / df)
    return jnp.where(ok, M, jnp.inf)


def maximum_L_over_G(T_in, T_out, h_a_in, P=P_ATM, cp_w=CP_LIQUID) -> Array:
    """Largest ``L/G`` for which the tower is feasible (the pinch limit).

    Beyond it the operating line crosses the saturation curve. The reciprocal
    is the minimum air-to-water ratio. Solved by bisection on the minimum
    driving force, differentiable by the implicit function theorem.
    """
    T_in, T_out, h_a_in, P = _f(T_in), _f(T_out), _f(h_a_in), _f(P)
    return _solve_monotone(
        lambda lg: merkel_min_driving_force(T_in, T_out, h_a_in, lg, P, cp_w), 1e-3, 100.0)


def saturation_temperature_from_enthalpy(h: Array | float, P: Array | float = P_ATM) -> Array:
    """Temperature (K) of saturated air of enthalpy ``h`` (J/kg dry air); inverse of ``saturation_enthalpy``."""
    h, P = _f(h), _f(P)
    return _solve_monotone(lambda T: saturation_enthalpy(T, P) - h, 200.0, 370.0)


def _fan(p, G, W_in, v_in, v_out) -> dict:
    """Air-side pressure drop and fan power; empty when no pressure drop (or area) is given."""
    out = {}
    if p.frontal_area is not None:
        A = _f(p.frontal_area)
        out["air_face_velocity"] = G * v_in / A
        out["air_mass_velocity"] = G * (1.0 + W_in) / A
    if p.fill_pressure_drop is None and p.fill_loss_coefficient is None and p.other_pressure_drop is None:
        return out
    if p.fill_pressure_drop is not None:
        dp_fill = _f(p.fill_pressure_drop)
    elif p.fill_loss_coefficient is not None:
        rho_in = (1.0 + W_in) / v_in
        dp_fill = _f(p.fill_loss_coefficient) * out["air_mass_velocity"] ** 2 / (2.0 * rho_in)
    else:
        dp_fill = _f(0.0)
    dp_total = dp_fill + _f(0.0 if p.other_pressure_drop is None else p.other_pressure_drop)
    V_dot = G * (v_out if p.fan_location == "induced" else v_in)
    out.update({
        "fill_pressure_drop": dp_fill,
        "total_pressure_drop": dp_total,
        "air_volumetric_flow": V_dot,
        "fan_power": V_dot * dp_total / _f(p.fan_efficiency),
    })
    return out


# =============================================================================
# The unit
# =============================================================================


@dataclass(repr=False)
class CoolingTowerParams(ParamsMixin):
    """Parameters of a counterflow cooling tower (Merkel).

    Attributes:
        mode: ``"design"`` (outlet water temperature given, ``KaV/L``
            computed) or ``"rating"`` (``KaV/L`` given, outlet temperature
            computed).
        L_over_G: Water-to-dry-air mass flow ratio; used when the unit is
            called without an air stream (otherwise it is computed from the
            two streams).
        KaV_L: Merkel number ``KaV/L`` (rating mode).
        Ka: Volumetric mass-transfer coefficient (kg of water per m^3 of fill
            per second, i.e. the Merkel coefficient in ``Ka V`` units of kg/s);
            with ``V`` it gives ``KaV/L = Ka V / L`` (rating mode).
        V: Fill volume (m^3).
        fill_characteristic: ``[C, n]``: the fill characteristic
            ``KaV/L = C (L/G)^-n`` (rating mode, when neither ``KaV_L`` nor
            ``Ka`` and ``V`` is given).
        T_out: Cold-water temperature (K) (design mode).
        approach: Approach to the wet bulb, ``T_out - T_wb`` (K) (design
            mode, instead of ``T_out``).
        P: Pressure (Pa); an air stream's own pressure wins when one is given.
        quadrature: ``"cti"`` (CTI four-point) or ``"gauss"``.
        n_quad: Gauss-Legendre points (``"gauss"`` only).
        drift_fraction: Drift (windage) loss as a fraction of the circulating
            water flow.
        cycles_of_concentration: Cycles of concentration of the basin water,
            for the blowdown: ``E + D + B = E C/(C - 1)`` for ``B >= 0``.
        cp_water: Specific heat of the circulating water (J/kg/K).
        makeup_in_outlet: ``True`` (default): the outlet water flow equals the
            inlet flow, i.e. the stream is the basin pump suction after make-up
            and blowdown; ``False``: the cold water leaving the fill, short by
            the evaporation and drift.
        fill_pressure_drop: Air-side pressure drop through the fill (Pa), given
            directly (from fill test data or a vendor curve). ``None``
            (default): computed from ``fill_loss_coefficient``, or absent.
        fill_loss_coefficient: Fill loss coefficient ``K_fill`` (-, velocity
            heads of the frontal air flow at the inlet density):
            ``dp_fill = K_fill G_m^2 / (2 rho_in)``, ``G_m = G (1 + W_in) / A_fr``.
            Needs ``frontal_area``; exclusive with ``fill_pressure_drop``.
        frontal_area: Fill plan (frontal) area ``A_fr`` (m^2), for the loss
            coefficient and the reported face velocity and mass velocity.
        other_pressure_drop: The rest of the air-side pressure drop (Pa): inlet
            louvres, rain zone, sprays, drift eliminator, stack. ``None``
            (default) counts as 0 when a fill pressure drop is given.
        fan_efficiency: Fan efficiency (-), ``0 < eta <= 1``; the fan power is
            ``V_dot dp_total / eta``. The default 0.65 is the difflow book's
            Ch. 18 design assumption, not a sourced value.
        fan_location: ``"induced"`` (default; the fan at the top handles the
            leaving air) or ``"forced"`` (the fan at the inlet handles the
            entering air); sets the volumetric flow ``V_dot = G v``.
    """

    mode: Literal["design", "rating"] = "design"
    L_over_G: float | None = None
    KaV_L: float | None = None
    Ka: float | None = None
    V: float | None = None
    fill_characteristic: list | None = None
    T_out: float | None = None
    approach: float | None = None
    P: float = 101325.0
    quadrature: Literal["cti", "gauss"] = "cti"
    n_quad: int = 8
    drift_fraction: float = 0.0002
    cycles_of_concentration: float = 4.0
    cp_water: float = 4186.0
    makeup_in_outlet: bool = True
    fill_pressure_drop: float | None = None
    fill_loss_coefficient: float | None = None
    frontal_area: float | None = None
    other_pressure_drop: float | None = None
    fan_efficiency: float = 0.65
    fan_location: Literal["induced", "forced"] = "induced"

    def __post_init__(self):
        if self.mode not in ("design", "rating"):
            raise ValueError("mode must be 'design' or 'rating'")
        if self.quadrature not in ("cti", "gauss"):
            raise ValueError("quadrature must be 'cti' or 'gauss'")
        if self.fan_location not in ("induced", "forced"):
            raise ValueError("fan_location must be 'induced' or 'forced'")
        if self.fill_pressure_drop is not None and self.fill_loss_coefficient is not None:
            raise ValueError("give fill_pressure_drop or fill_loss_coefficient, not both")
        if self.fill_loss_coefficient is not None and self.frontal_area is None:
            raise ValueError("fill_loss_coefficient needs frontal_area")
        # value checks on concrete numbers only (a value traced under jax.grad is not checked)
        for name in ("fill_pressure_drop", "fill_loss_coefficient", "other_pressure_drop"):
            x = getattr(self, name)
            if _concrete_number(x) and float(x) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if _concrete_number(self.frontal_area) and float(self.frontal_area) <= 0.0:
            raise ValueError("frontal_area must be > 0")
        if _concrete_number(self.fan_efficiency) and not 0.0 < float(self.fan_efficiency) <= 1.0:
            raise ValueError("fan_efficiency must be in (0, 1]")


def _concrete_number(x) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and not isinstance(x, bool)


class CoolingTower:
    """Counterflow cooling tower sized or rated with the Merkel equation.

    Two inlets (hot water, and optionally humid air) and two outlets (cold
    water, and the humid air leaving, saturated by Merkel's assumption). See
    the module docstring for the equation, the two modes and the quadratures;
    :func:`merkel_number` and :func:`maximum_L_over_G` are the library
    functions underneath.
    """

    symbol = "Cooling tower"
    equations = [
        r"\frac{K a V}{L} = \int_{T_{out}}^{T_{in}} \frac{c_{pw}\,dT}{h_s(T) - h_a(T)}",
        r"h_a(T) = h_{a,in} + \frac{L}{G}\,c_{pw}\,(T - T_{out})",
        r"\frac{K a V}{L} \approx c_{pw}\,(T_{in} - T_{out})\,\frac14\sum_{i=1}^{4}\frac{1}{(h_s - h_a)_i},\quad x_i = 0.1,\,0.4,\,0.6,\,0.9\ \text{(CTI)}",
        r"E = G\,(W_{out} - W_{in}),\quad h_s(T_{air,out}) = h_{a,out},\qquad M = E + D + B = E\,\frac{C}{C-1}",
        r"\Delta p_{fill} = K_{fill}\,\frac{G_m^2}{2\rho_{in}},\qquad \dot W_{fan} = \frac{G\,v\,(\Delta p_{fill} + \Delta p_{other})}{\eta_{fan}}",
    ]
    assumptions = [
        "Merkel: Lewis factor of 1, so sensible and latent transfer combine into one enthalpy driving force.",
        "The air leaves saturated; the evaporated water is neglected in the air enthalpy balance (Poppe's correction is not applied).",
        "Constant water specific heat and constant L/G through the fill; ideal-gas humid air (ASHRAE constants, no enhancement factor).",
        "Counterflow fill; no recirculation of exhaust air. The air-side pressure drop is an input "
        "(fill_pressure_drop, or a loss coefficient K_fill in velocity heads of the frontal flow, plus "
        "other_pressure_drop); fan power = V_dot dp_total / eta_fan.",
        "Without an air stream the inlet air is taken to be saturated at the wet bulb.",
    ]
    references = [
        "Merkel, F. Verdunstungskuhlung, VDI Forschungsarbeiten 275 (1925).",
        "Cooling Technology Institute, ATC-105 acceptance test code (four-point Chebyshev integration of the Merkel integral).",
        "Kloppers, J.C., Kroger, D.G. Int. J. Heat Mass Transfer 48, 765 (2005): Merkel, Poppe and e-NTU compared.",
        "McCabe, Smith & Harriott, Unit Operations of Chemical Engineering, Ch. 19.",
        "Kern, D.Q. Process Heat Transfer, Ch. 14 (cooling towers).",
    ]
    parameter_symbols = {
        "L_over_G": "L/G", "KaV_L": "KaV/L", "Ka": "Ka", "V": "V", "T_out": r"T_{out}",
        "approach": r"\Delta T_{app}", "drift_fraction": "D/L", "cycles_of_concentration": "C",
        "cp_water": r"c_{pw}", "fill_pressure_drop": r"\Delta p_{fill}",
        "fill_loss_coefficient": r"K_{fill}", "frontal_area": r"A_{fr}",
        "other_pressure_drop": r"\Delta p_{other}", "fan_efficiency": r"\eta_{fan}",
    }
    parameter_units = {
        "L_over_G": "kg/kg", "KaV_L": "-", "Ka": "kg/(m^3 s)", "V": "m^3", "fill_characteristic": "-",
        "T_out": "K", "approach": "K", "P": "Pa", "n_quad": "-", "drift_fraction": "-",
        "cycles_of_concentration": "-", "cp_water": "J/(kg K)", "fill_pressure_drop": "Pa",
        "fill_loss_coefficient": "-", "frontal_area": "m^2", "other_pressure_drop": "Pa",
        "fan_efficiency": "-",
    }
    numerical_method = (
        "CTI four-point (or Gauss-Legendre) quadrature of the Merkel integral; rating by bisection on the "
        "outlet temperature with an implicit-function-theorem gradient; pinch detected on a 65-point grid."
    )
    outlet_roles = ["water_out", "air_out"]

    def __init__(self, params: CoolingTowerParams):
        self.params = params

    def __call__(self, water_in: Stream, air_in: Stream = None, T_wb: float = None) -> tuple[Stream, Stream, dict]:
        """Cool the water against air.

        Args:
            water_in: Hot water (``F_H2O`` in mol/s, ``T`` in K).
            air_in: Humid inlet air (``F_H2O`` and ``F_air``); optional. When
                absent the air flow is ``L / L_over_G`` and the air is
                saturated at ``T_wb``.
            T_wb: Wet-bulb temperature (K) of the inlet air, for the case without
                an air stream (ignored when ``air_in`` is given).

        Returns:
            water_out: Cold water at ``T_out``, at the tower (basin) pressure.
            air_out: Humid air leaving the tower, saturated.
            info: ``KaV_L`` (Merkel number, ``inf`` when infeasible), ``feasible``,
                ``converged``, ``min_driving_force`` (J/kg), ``L_over_G``,
                ``L_over_G_max`` (pinch limit), ``T_in``, ``T_out``, ``T_wb``,
                ``range``, ``approach``, ``effectiveness`` (range/(range+approach)),
                ``Q`` (W), ``L`` and ``G`` (kg/s), ``evaporation``, ``drift``,
                ``blowdown``, ``makeup`` (kg/s), ``evaporation_fraction``,
                ``air_out`` (state of the leaving air), ``V`` (m^3, when ``Ka``
                is given). With a pressure drop given (``fill_pressure_drop``,
                ``fill_loss_coefficient`` or ``other_pressure_drop``) also
                ``fill_pressure_drop``, ``total_pressure_drop`` (Pa),
                ``air_volumetric_flow`` (m^3/s at the fan) and ``fan_power``
                (W); with ``frontal_area`` also ``air_face_velocity`` (m/s,
                inlet air) and ``air_mass_velocity`` (kg/(m^2 s), moist air).
        """
        p = self.params
        rating = p.mode == "rating"
        cp_w = p.cp_water
        T_in = _f(water_in["T"])
        L = _f(water_in["F_H2O"]) * MW_WATER

        if air_in is not None:
            st_in = state_from_air_stream(air_in)
            P = _f(air_in["P"])
            G = dry_air_flow(air_in)
            LG = L / G
            T_wb_v = st_in["T_wb"]
            h_ain = st_in["h"]
            W_in = st_in["W"]
            v_in = st_in["v"]
        else:
            if T_wb is None:
                raise ValueError("CoolingTower needs air_in or T_wb")
            if p.L_over_G is None:
                raise ValueError("without an air stream CoolingTower needs params.L_over_G")
            P = _f(p.P)
            T_wb_v = _f(T_wb)
            LG = _f(p.L_over_G)
            G = L / LG
            h_ain = saturation_enthalpy(T_wb_v, P)
            W_in = saturation_humidity_ratio(T_wb_v, P)
            v_in = humid_volume(T_wb_v, W_in, P)

        extras = {}

        if rating:
            if p.KaV_L is not None:
                target = _f(p.KaV_L)
            elif p.Ka is not None and p.V is not None:
                target = _f(p.Ka) * _f(p.V) / L
            elif p.fill_characteristic is not None:
                C, n = p.fill_characteristic
                target = _f(C) * LG ** (-_f(n))
            else:
                raise ValueError("rating mode needs KaV_L, Ka and V, or fill_characteristic")
            T_lo = saturation_temperature_from_enthalpy(h_ain, P) + 1e-6

            def f(T_o):
                M = merkel_number(T_in, T_o, h_ain, LG, P, cp_w, p.quadrature, p.n_quad)
                lnM = jnp.where(jnp.isfinite(M), jnp.log(jnp.where(jnp.isfinite(M), M, 1.0)), _LN_BIG)
                return lnM - jnp.log(target)

            T_out = _solve_monotone(f, T_lo, T_in - 1e-9)
            converged = jnp.abs(f(T_out)) < 1e-6
            M_val = target
        else:
            if p.T_out is not None:
                T_out = _f(p.T_out)
            elif p.approach is not None:
                T_out = T_wb_v + _f(p.approach)
            else:
                raise ValueError("design mode needs T_out or approach")
            M_val = merkel_number(T_in, T_out, h_ain, LG, P, cp_w, p.quadrature, p.n_quad)
            converged = jnp.asarray(True)

        # pinch limit and feasibility at the solution
        LG_max = maximum_L_over_G(T_in, T_out, h_ain, P, cp_w)
        dmin = merkel_min_driving_force(T_in, T_out, h_ain, LG, P, cp_w)
        feasible = (dmin > 0.0) & (T_in > T_out)
        if rating:
            feasible = feasible & converged

        # outlet air: saturated at the exit enthalpy (Merkel)
        R = T_in - T_out
        h_aout = h_ain + LG * cp_w * R
        T_ao = saturation_temperature_from_enthalpy(h_aout, P)
        W_out = saturation_humidity_ratio(T_ao, P)
        evap = G * (W_out - W_in)
        drift = p.drift_fraction * L
        C_oc = _f(p.cycles_of_concentration)
        blow = jnp.maximum(evap / (C_oc - 1.0) - drift, 0.0)
        makeup = evap + drift + blow

        water_out = dict(water_in)
        water_out["T"] = T_out
        water_out["P"] = P  # the cold-water basin is at the tower pressure: closes a pumped loop
        if not p.makeup_in_outlet:
            water_out["F_H2O"] = _f(water_in["F_H2O"]) - (evap + drift) / MW_WATER
        st_out = state(T=T_ao, W=W_out, P=P)
        air_out = air_stream_from_state(st_out, G)
        if air_in is not None:
            for k, v in air_in.items():
                if k.startswith("F_") and k not in air_out:
                    air_out[k] = v

        if not rating and p.Ka is not None:
            extras["V"] = M_val * L / _f(p.Ka)
        extras.update(_fan(p, G, W_in, v_in, st_out["v"]))
        approach = T_out - T_wb_v
        info = {
            "KaV_L": M_val,
            "feasible": feasible,
            "converged": converged,
            "min_driving_force": dmin,
            "L_over_G": LG,
            "L_over_G_max": LG_max,
            "T_in": T_in,
            "T_out": T_out,
            "T_wb": T_wb_v,
            "range": R,
            "approach": approach,
            "effectiveness": R / (R + approach),
            "Q": L * cp_w * R,
            "L": L,
            "G": G,
            "evaporation": evap,
            "evaporation_fraction": evap / L,
            "drift": drift,
            "blowdown": blow,
            "makeup": makeup,
            "air_out": st_out,
            **extras,
        }
        return water_out, air_out, info
