"""Single-stage mechanical vapor compression (MVC) desalination.

The plant is WaterTAP 1.8.0's ``flowsheets/mvc/mvc_single_stage.py``
layout, equation for equation, so the two compare (``benchmarks/watertap/
mvc.py``):

    feed -> feed pump -> split (fraction = recovery) -> distillate preheater (cold)
                                                     -> brine preheater (cold)
         -> mix -> evaporator -> brine -> brine pump -> brine preheater (hot) -> brine
                              -> vapor -> compressor -> condenser (in the evaporator)
                                       -> distillate pump -> distillate preheater (hot)

Equations (WaterTAP's; the thermodynamic content is standard, see
El-Dessouky & Ettouney [D2] for MVC and [T1] for the plant layout):

- evaporator: brine and vapor leave at the brine temperature ``T_b``; the
  brine at its vapor pressure ``p_sw(T_b, S_b)``; energy balance
  ``Q + H_feed = H_brine + H_vapor``; ``Q = U A LMTD`` with the condensing
  vapor on the other side (``dT_in = T_compressed - T_b``,
  ``dT_out = T_condensate - T_b``);
- compressor: ``P_out = r P_in``, isentropic outlet temperature
  ``T_s = T_in r^(1 - 1/gamma)`` with ``gamma = 1.3`` (WaterTAP's fixed
  value; an ideal-gas approximation), ``eta (h_out - h_in) = h_s - h_in``;
- condenser: complete condensation at the compressor outlet pressure,
  ``m_v (h_vap,in - h_liq,out) = Q``; the condensate must not be hotter than
  saturation at that pressure (``condensate_subcooling >= 0``);
- preheaters: countercurrent, ``Q = U A LMTD`` with Chen's approximation
  ``LMTD = (dT1 dT2 (dT1 + dT2)/2)^(1/3)`` (as WaterTAP);
- pumps: isothermal, ``W = V dP / eta``; the brine pump returns the brine to
  ``brine_outlet_pressure`` after the preheater's pressure drop.

Properties: :mod:`difflow_water.chemistry.seawater` (Sharqawy 2010 [S1] with
WaterTAP's Nayar 2016 [S2] precision where stated there).

Degrees of freedom: brine temperature and the preheater areas are inputs,
the evaporator area and duty are computed, and either the pressure ratio
is given (``mode="fixed_pressure_ratio"``, WaterTAP's simulation) or it is
solved so that the condensate leaves saturated (``"saturated_condensate"``,
the default). With the pressure ratio fixed, the compressor's energy must
match what the plant absorbs: most designs are then unphysical (condensate
hotter than saturation, or colder than the brine), which WaterTAP's solver
reports as infeasible and this model reports through
``condensate_subcooling`` and ``evaporator_dT_out``. The whole plant is one Newton solve
(``optimistix.root_find``), so gradients are implicit and exact at the
solution.

Default design values are those WaterTAP's example sets (50 % recovery,
70-75 C brine, ...) with preheater areas near its optimum: design inputs,
ILLUSTRATIVE. The preheater areas are absolute (WaterTAP's convention) and
must be sized to the feed: WaterTAP's optimum is about 4.5 (distillate) and
3.9 (brine) m^2 per kg/s of feed. Oversized preheaters leave too little
work for the compressor and the saturated-condensate solution has its
condensate colder than the brine (``feasible`` False).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.seawater import MW_SEA_SALT, MW_WATER
from difflow_water.economics import CostingParams, lcow
from difflow_water.streams import SALT, WATER, mass_flows

#: Chemical Engineering Plant Cost Index values IDAES uses for its currency
#: units (``idaes.core.base.costing_base``; second-hand, unverified).
CEPCI = {2001: 394.3, 2018: 603.1, 2020: 596.2}


def chen_lmtd(dT1, dT2) -> Array:
    """Chen's approximation of the log-mean temperature difference (K)."""
    return jnp.cbrt(dT1 * dT2 * 0.5 * (dT1 + dT2))


@dataclass(repr=False)
class MVCParams(ParamsMixin):
    """Design of a single-stage MVC plant (defaults: WaterTAP's example; ILLUSTRATIVE).

    Attributes:
        recovery: Vapor (product) mass / feed mass (-).
        T_brine: Evaporator brine (and vapor) temperature (K).
        pressure_ratio: Compressor outlet / inlet pressure (-).
        compressor_efficiency: Isentropic efficiency of the compressor (-).
        gamma: Heat-capacity ratio in the isentropic temperature (-).
        U_evaporator: Evaporator overall heat-transfer coefficient (W/(m^2 K)).
        A_hx_distillate: Distillate preheater area (m^2).
        A_hx_brine: Brine preheater area (m^2).
        U_hx: Preheater overall heat-transfer coefficient (W/(m^2 K)).
        feed_pump_dP: Feed pump pressure rise (Pa).
        hx_dP: Pressure drop on each side of each preheater (Pa; positive
            is a loss). WaterTAP's example fixes IDAES ``deltaP = +7 kPa``,
            a pressure GAIN: reproduce it with ``hx_dP = -7e3``.
        distillate_pump_dP: Distillate pump pressure rise (Pa).
        pump_efficiency: Pump efficiency (-).
        brine_outlet_pressure: Brine pressure after its preheater (Pa).
        mode: ``"saturated_condensate"`` (default): the pressure ratio is
            solved so that the condensate leaves saturated, which closes the
            energy balance for any feed (WaterTAP's LCOW optimum lies on
            this boundary); ``pressure_ratio`` is then only the initial
            guess. ``"fixed_pressure_ratio"``: WaterTAP's simulation degrees
            of freedom; a design is physical only if
            ``condensate_subcooling >= 0`` and ``evaporator_dT_out > 0``.
    """

    recovery: float | Array = 0.5
    T_brine: float | Array = 348.15
    pressure_ratio: float | Array = 1.6
    compressor_efficiency: float | Array = 0.8
    gamma: float | Array = 1.3
    U_evaporator: float | Array = 3000.0
    A_hx_distillate: float | Array = 200.0
    A_hx_brine: float | Array = 175.0
    U_hx: float | Array = 2000.0
    feed_pump_dP: float | Array = 7e3
    hx_dP: float | Array = 7e3
    distillate_pump_dP: float | Array = 4e4
    pump_efficiency: float | Array = 0.8
    brine_outlet_pressure: float | Array = 101325.0
    mode: str = "saturated_condensate"


@dataclass(repr=False)
class MVCCostingParams(ParamsMixin):
    """MVC unit costs as WaterTAP 1.8.0's MVC example sets them (ILLUSTRATIVE).

    Attributes:
        base_year: Currency year of the results. WaterTAP's example sets
            ``base_currency = USD_2020`` only after its costs are built, which
            leaves them in WaterTAP's default USD 2018: 2018 matches it.
        evaporator_unit_cost: Evaporator cost (USD 2020 per m^2).
        evaporator_material_factor: Material factor on the evaporator cost (-).
        hx_unit_cost: Preheater cost (USD 2020 per m^2).
        hx_material_factor: Material factor on the preheater cost (-).
        compressor_unit_cost: Compressor cost coefficient (USD 2001 per kg/s).
        compressor_exponent: Exponent on ``eta / (1 - eta)`` (-).
        pump_cost: Pump cost (USD 2018 per W of mechanical work).
        mixer_cost: Mixer cost (USD 2018 per L/s).
        electricity_cost: Electricity price (USD 2018 per kWh).
    """

    base_year: int = 2018
    evaporator_unit_cost: float | Array = 1000.0
    evaporator_material_factor: float | Array = 5.0
    hx_unit_cost: float | Array = 300.0
    hx_material_factor: float | Array = 5.0
    compressor_unit_cost: float | Array = 7364.0
    compressor_exponent: float | Array = 0.7
    pump_cost: float | Array = 53.0 / 1e5 * 3600.0
    mixer_cost: float | Array = 361.0
    electricity_cost: float | Array = 0.1


def _to_base(value, year, base_year):
    return value * CEPCI[base_year] / CEPCI[year]


def _temperature(h_fn, h_target, T0, n_iter=4):
    """Invert a temperature-enthalpy function by Newton (smooth, few steps)."""
    T = T0
    for _ in range(n_iter):
        f, df = jax.jvp(h_fn, (T,), (jnp.ones_like(T),))
        T = T - (f - h_target) / df
    return T


def _mvc_state(p: MVCParams, mw_f, ms_f, T_f, P_f):
    """Solve the plant; returns a dict of the converged state."""
    m_f = mw_f + ms_f
    S_f = ms_f / m_f
    r = p.recovery
    m_v = r * m_f
    m_b = m_f - m_v
    S_b = ms_f / m_b
    T_b = p.T_brine
    P_b = sw.vapor_pressure(T_b, S_b)
    P1 = P_f + p.feed_pump_dP              # after the feed pump
    P_cold_out = P1 - p.hx_dP
    P_bp = p.brine_outlet_pressure + p.hx_dP  # brine after its pump
    m_c1 = r * m_f                         # feed through the distillate preheater
    m_c2 = (1.0 - r) * m_f
    h_vb = sw.vapor_enthalpy(T_b, P_b)
    saturated = p.mode == "saturated_condensate"

    def compressor(pr):
        P_co = pr * P_b
        T_s = T_b * pr ** (1.0 - 1.0 / p.gamma)
        h_s = sw.vapor_enthalpy(T_s, P_co)
        h_co = h_vb + (h_s - h_vb) / p.compressor_efficiency
        T_co = _temperature(lambda T: sw.vapor_enthalpy(T, P_co), h_co, T_s + 10.0)
        return P_co, h_co, T_co

    h_c = lambda T, S, P: sw.enthalpy(T, S, P)  # noqa: E731

    def residual(x, _):
        T_c, Tc1, Th1, Tc2, Th2, T_mix, q, pr = x
        Q = q * 1e7
        P_co, h_co, _ = compressor(pr)
        P_d = P_co + p.distillate_pump_dP
        # condenser
        r1 = (m_v * (h_co - sw.enthalpy_water(T_c, P_co)) - Q) / 1e7
        # distillate preheater: hot = condensate at P_d, cold = feed at P1
        q1 = m_c1 * (h_c(Tc1, S_f, P_cold_out) - h_c(T_f, S_f, P1))
        r2 = (m_v * (h_c(T_c, 0.0, P_d) - h_c(Th1, 0.0, P_d - p.hx_dP)) - q1) / 1e6
        r3 = (q1 - p.U_hx * p.A_hx_distillate * chen_lmtd(T_c - Tc1, Th1 - T_f)) / 1e6
        # brine preheater: hot = brine at P_bp
        q2 = m_c2 * (h_c(Tc2, S_f, P_cold_out) - h_c(T_f, S_f, P1))
        r4 = (m_b * (h_c(T_b, S_b, P_bp) - h_c(Th2, S_b, P_bp - p.hx_dP)) - q2) / 1e6
        r5 = (q2 - p.U_hx * p.A_hx_brine * chen_lmtd(T_b - Tc2, Th2 - T_f)) / 1e6
        # mixer (outlet at the distillate-preheater cold outlet pressure)
        r6 = (m_f * h_c(T_mix, S_f, P_cold_out) - m_c1 * h_c(Tc1, S_f, P_cold_out)
              - m_c2 * h_c(Tc2, S_f, P_cold_out)) / 1e6
        # evaporator energy balance
        r7 = (Q + m_f * h_c(T_mix, S_f, P_cold_out) - m_b * h_c(T_b, S_b, P_b)
              - m_v * sw.vapor_enthalpy(T_b, P_b)) / 1e7
        # pressure ratio: given, or the one that leaves the condensate saturated
        r8 = (jnp.log(sw.vapor_pressure_water(T_c)) - jnp.log(P_co) if saturated
              else pr - p.pressure_ratio)
        return jnp.stack([r1, r2, r3, r4, r5, r6, r7, r8])

    q0 = m_v * sw.latent_heat_water(T_b) / 1e7
    pr0 = 1.6 if saturated else p.pressure_ratio
    # initial guess: preheaters from a balanced-countercurrent effectiveness
    # (NTU / (1 + NTU)), hot inlets at T_b + 3 (condensate) and T_b (brine)
    cp = sw.specific_heat(T_f, S_f)
    ntu1 = p.U_hx * p.A_hx_distillate / (m_c1 * cp)
    ntu2 = p.U_hx * p.A_hx_brine / (m_c2 * cp)
    e1, e2 = ntu1 / (1.0 + ntu1), ntu2 / (1.0 + ntu2)
    Tc1_0 = T_f + e1 * (T_b + 3.0 - T_f)
    Tc2_0 = T_f + e2 * (T_b - T_f)
    x0 = jnp.stack([T_b + 3.0, Tc1_0, T_b + 3.0 - (Tc1_0 - T_f), Tc2_0,
                    T_b - (Tc2_0 - T_f), r * Tc1_0 + (1.0 - r) * Tc2_0,
                    q0 + 0.0 * T_b, pr0 + 0.0 * T_b])
    sol = optx.root_find(residual, optx.Newton(rtol=1e-12, atol=1e-12), x0,
                         max_steps=100, throw=False)
    T_c, Tc1, Th1, Tc2, Th2, T_mix, q, pr = sol.value
    Q = q * 1e7
    P_co, h_co, T_co = compressor(pr)
    P_d = P_co + p.distillate_pump_dP
    res = residual(sol.value, None)
    lmtd = chen_lmtd(T_co - T_b, T_c - T_b)
    return dict(
        m_f=m_f, S_f=S_f, m_v=m_v, m_b=m_b, S_b=S_b, T_b=T_b, P_b=P_b, P_co=P_co,
        T_co=T_co, h_vb=h_vb, h_co=h_co, T_c=T_c, Tc1=Tc1, Th1=Th1, Tc2=Tc2, Th2=Th2,
        T_mix=T_mix, Q=Q, lmtd=lmtd, pressure_ratio=pr, P1=P1, P_d=P_d, P_bp=P_bp, P_cold_out=P_cold_out,
        converged=(sol.result == optx.RESULTS.successful) & (jnp.max(jnp.abs(res)) < 1e-6),
        residual=jnp.max(jnp.abs(res)),
    )


def mvc_performance(feed: Stream, params: MVCParams | None = None,
                    costing: MVCCostingParams | None = None,
                    global_costing: CostingParams | None = None) -> dict:
    """Solve a single-stage MVC plant and cost it as WaterTAP does.

    Args:
        feed: Seawater stream (``F_H2O``, ``F_TDS`` mol/s, ``T``, ``P``).
        params: :class:`MVCParams` (WaterTAP's example design when None).
        costing: :class:`MVCCostingParams` (WaterTAP's example when None).
        global_costing: :class:`~difflow_water.economics.CostingParams` for the
            plant-wide factors (WaterTAP's defaults when None; its
            ``electricity_cost`` is replaced by ``costing.electricity_cost``).

    Returns:
        dict with the state (``T_compressed_vapor``, ``T_condensate``,
        ``T_evaporator_feed``, ``P_brine``, ``P_compressed_vapor`` ...),
        ``evaporator_area`` (m^2), ``evaporator_heat`` (W),
        ``compressor_work`` and pump works (W), ``condensate_subcooling`` and
        ``evaporator_dT_out`` (K; both must be positive for a physical
        plant), ``feasible`` (converged and both hold), ``pressure_ratio``,
        ``SEC`` (kWh/m^3 of distillate), ``LCOW`` (USD/m^3),
        ``capital`` per unit (USD), ``distillate`` and ``brine`` streams,
        ``converged``.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_mass=44.444, salinity=0.1)
        >>> out = mvc_performance(feed, MVCParams())
        >>> bool(out["converged"])
        True
    """
    p = params or MVCParams()
    if p.mode not in ("saturated_condensate", "fixed_pressure_ratio"):
        raise ValueError("mode must be 'saturated_condensate' or 'fixed_pressure_ratio'")
    c = costing or MVCCostingParams()
    g = (global_costing or CostingParams()).update(electricity_cost=c.electricity_cost)
    mw_f, ms_f = mass_flows(feed)
    s = _mvc_state(p, mw_f, ms_f, feed["T"], feed["P"])
    eta = p.pump_efficiency
    rho = sw.density
    V_feed = (mw_f + ms_f) / rho(feed["T"], s["S_f"])
    W_feed = V_feed * p.feed_pump_dP / eta
    V_brine = s["m_b"] / rho(s["T_b"], s["S_b"])
    W_brine = V_brine * (s["P_bp"] - s["P_b"]) / eta
    V_dist = s["m_v"] / rho(s["T_c"], 0.0)
    W_dist = V_dist * p.distillate_pump_dP / eta
    W_comp = s["m_v"] * (s["h_co"] - s["h_vb"])
    area = s["Q"] / (p.U_evaporator * s["lmtd"])
    V_mix = s["m_f"] / rho(s["T_mix"], s["S_f"])
    V_product = s["m_v"] / rho(s["Th1"], 0.0)

    by = c.base_year
    TIC = g.TIC
    capital = {
        "evaporator": TIC * _to_base(
            c.evaporator_unit_cost * c.evaporator_material_factor * area, 2020, by),
        "hx_distillate": TIC * _to_base(
            c.hx_unit_cost * c.hx_material_factor * p.A_hx_distillate, 2020, by),
        "hx_brine": TIC * _to_base(
            c.hx_unit_cost * c.hx_material_factor * p.A_hx_brine, 2020, by),
        "compressor": TIC * _to_base(
            c.compressor_unit_cost * s["m_v"] * s["pressure_ratio"]
            * (p.compressor_efficiency / (1.0 - p.compressor_efficiency)) ** c.compressor_exponent,
            2001, by),
        "pump_feed": TIC * _to_base(c.pump_cost * W_feed, 2018, by),
        "pump_brine": TIC * _to_base(c.pump_cost * W_brine, 2018, by),
        "pump_distillate": TIC * _to_base(c.pump_cost * W_dist, 2018, by),
        "mixer_feed": TIC * _to_base(c.mixer_cost * V_mix * 1e3, 2018, by),
    }
    power = W_comp + W_feed + W_brine + W_dist
    g = g.update(electricity_cost=_to_base(c.electricity_cost, 2018, by))
    cost = lcow({k: {"capital": v, "fixed_opex": 0.0} for k, v in capital.items()},
                power, V_product, g)
    T_sat_co = sw.saturation_temperature(s["P_co"], 0.0, T0=s["T_b"] + 10.0)
    distillate = make_stream({WATER: s["m_v"] / MW_WATER, SALT: 0.0 * s["m_v"]},
                             s["Th1"], s["P_d"] - p.hx_dP)
    brine = make_stream({WATER: s["m_b"] * (1.0 - s["S_b"]) / MW_WATER,
                         SALT: s["m_b"] * s["S_b"] / MW_SEA_SALT},
                        s["Th2"], p.brine_outlet_pressure)
    return {
        "evaporator_area": area,
        "evaporator_heat": s["Q"],
        "evaporator_lmtd": s["lmtd"],
        "T_evaporator_feed": s["T_mix"],
        "P_brine": s["P_b"],
        "brine_salinity": s["S_b"],
        "T_compressed_vapor": s["T_co"],
        "P_compressed_vapor": s["P_co"],
        "pressure_ratio": s["pressure_ratio"],
        "compressor_work": W_comp,
        "T_condensate": s["T_c"],
        "T_distillate_out": s["Th1"],
        "T_brine_out": s["Th2"],
        "pump_feed_work": W_feed,
        "pump_brine_work": W_brine,
        "pump_distillate_work": W_dist,
        "product_flow_vol": V_product,
        "condensate_subcooling": T_sat_co - s["T_c"],
        "evaporator_dT_out": s["T_c"] - s["T_b"],
        "SEC": cost["SEC"],
        "LCOW": cost["LCOW"],
        "capital": capital,
        "total_capital": cost["total_capital"],
        "total_operating": cost["total_operating"],
        "distillate": distillate,
        "brine": brine,
        "converged": s["converged"],
        "feasible": s["converged"] & (T_sat_co - s["T_c"] >= -1e-6) & (s["T_c"] > s["T_b"]),
        "residual": s["residual"],
    }
