"""Levelized cost of water (LCOW) and specific energy, WaterTAP-compatible.

This reproduces the structure and the default parameters of WaterTAP
1.8.0's default costing package (``WaterTAPCosting``) and its unit costing
methods for RO membranes, high- and low-pressure pumps and pressure
exchangers, so an LCOW computed here and one computed by WaterTAP on the
same design differ only through the process models.

The default numbers are WaterTAP's, in USD (2018). WaterTAP's code does not
cite a source for them; they are used here as the benchmark's convention,
not as sourced cost data (status: second-hand, ILLUSTRATIVE for any real
plant). Change them through :class:`CostingParams`.

Structure (per WaterTAP)::

    capital_i       = TIC * unit_cost_i * size_i
    aggregate_cap   = sum_i capital_i
    total_capital   = total_investment_factor * aggregate_cap
    fixed_opex      = membrane_replacement * membrane_cost * area      (per year)
                      + maintenance_labor_chemical_factor * aggregate_cap
    electricity     = electricity_cost * P_kW * hours_per_year * utilization
    CRF             = wacc / (1 - (1 + wacc)^-lifetime)
    LCOW            = (total_capital * CRF + fixed_opex + electricity)
                      / (Q_product * seconds_per_year * utilization)

A year is 365.25 days (Pyomo's ``year`` unit).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin

HOURS_PER_YEAR = 365.25 * 24.0
SECONDS_PER_YEAR = HOURS_PER_YEAR * 3600.0


@dataclass(repr=False)
class CostingParams(ParamsMixin):
    """Global and unit costing parameters (WaterTAP 1.8.0 defaults, USD 2018).

    Attributes:
        utilization_factor: Fraction of the year the plant runs (-).
        electricity_cost: Electricity price (USD/kWh).
        plant_lifetime: Plant lifetime (years).
        wacc: Weighted average cost of capital (-).
        TIC: Total installed cost factor on equipment cost (-).
        total_investment_factor: Investment / installed equipment cost (-).
        maintenance_labor_chemical_factor: Fraction of capital per year (1/yr).
        membrane_cost: RO membrane cost (USD/m^2).
        membrane_replacement: Fraction of membrane replaced per year (1/yr).
        hp_pump_cost: High-pressure pump cost (USD/W of mechanical work).
        lp_pump_cost: Low-pressure pump cost (USD/(L/s) of inlet flow).
        px_cost: Pressure exchanger cost (USD/(m^3/h) of low-pressure flow).
    """

    utilization_factor: float | Array = 0.9
    electricity_cost: float | Array = 0.07
    plant_lifetime: float | Array = 30.0
    wacc: float | Array = 0.09307339771758532
    TIC: float | Array = 2.0
    total_investment_factor: float | Array = 1.0
    maintenance_labor_chemical_factor: float | Array = 0.03
    membrane_cost: float | Array = 30.0
    membrane_replacement: float | Array = 0.2
    hp_pump_cost: float | Array = 53.0 / 1e5 * 3600.0
    lp_pump_cost: float | Array = 889.0
    px_cost: float | Array = 535.0


def capital_recovery_factor(wacc, lifetime) -> Array:
    """``wacc / (1 - (1 + wacc)^-lifetime)`` (1/yr)."""
    wacc = jnp.asarray(wacc, dtype=jnp.float64)
    return wacc / (1.0 - (1.0 + wacc) ** (-jnp.asarray(lifetime, dtype=jnp.float64)))


def cost_ro(area, c: CostingParams) -> dict:
    """Capital (USD) and fixed operating cost (USD/yr) of RO membrane area (m^2)."""
    return {"capital": c.TIC * c.membrane_cost * area,
            "fixed_opex": c.membrane_replacement * c.membrane_cost * area}


def cost_hp_pump(work_mechanical_W, c: CostingParams) -> dict:
    """Capital (USD) of a high-pressure pump from its mechanical work (W)."""
    return {"capital": c.TIC * c.hp_pump_cost * work_mechanical_W, "fixed_opex": 0.0}


def cost_lp_pump(flow_vol_m3s, c: CostingParams) -> dict:
    """Capital (USD) of a low-pressure pump from its inlet flow (m^3/s)."""
    return {"capital": c.TIC * c.lp_pump_cost * flow_vol_m3s * 1e3, "fixed_opex": 0.0}


def cost_pressure_exchanger(flow_vol_m3s, c: CostingParams) -> dict:
    """Capital (USD) of a pressure exchanger from its low-pressure flow (m^3/s)."""
    return {"capital": c.TIC * c.px_cost * flow_vol_m3s * 3600.0, "fixed_opex": 0.0}


def lcow(unit_costs: dict[str, dict], electricity_W, product_flow_vol,
         c: CostingParams | None = None) -> dict:
    """Levelized cost of water and its parts.

    Args:
        unit_costs: ``{name: {"capital": USD, "fixed_opex": USD/yr}}``, e.g.
            from :func:`cost_ro` and the other ``cost_*`` helpers.
        electricity_W: Total electric power drawn (W): pump mechanical work
            (WaterTAP costs ``work_mechanical`` as electricity).
        product_flow_vol: Product water flow (m^3/s).
        c: :class:`CostingParams`; WaterTAP's defaults when None.

    Returns:
        dict with ``LCOW`` (USD/m^3), ``SEC`` (kWh/m^3), ``total_capital``,
        ``aggregate_capital`` (USD), ``total_operating`` (USD/yr),
        ``electricity_cost`` (USD/yr), ``CRF`` (1/yr) and
        ``annual_production`` (m^3/yr).
    """
    c = c or CostingParams()
    agg_cap = sum(v["capital"] for v in unit_costs.values())
    fixed = sum(v["fixed_opex"] for v in unit_costs.values())
    total_cap = c.total_investment_factor * agg_cap
    mlc = c.maintenance_labor_chemical_factor * agg_cap
    kW = jnp.asarray(electricity_W, dtype=jnp.float64) / 1e3
    elec = c.electricity_cost * kW * HOURS_PER_YEAR * c.utilization_factor
    opex = fixed + mlc + elec
    crf = capital_recovery_factor(c.wacc, c.plant_lifetime)
    annual = jnp.asarray(product_flow_vol, dtype=jnp.float64) * SECONDS_PER_YEAR * c.utilization_factor
    return {
        "LCOW": (total_cap * crf + opex) / annual,
        "SEC": kW / (jnp.asarray(product_flow_vol, dtype=jnp.float64) * 3600.0),
        "total_capital": total_cap,
        "aggregate_capital": agg_cap,
        "total_operating": opex,
        "electricity_cost": elec,
        "CRF": crf,
        "annual_production": annual,
    }
