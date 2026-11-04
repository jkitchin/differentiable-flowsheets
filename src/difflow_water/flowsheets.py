"""Seawater RO with a pressure exchanger: the phase-1 benchmark flowsheet.

The layout is WaterTAP 1.8.0's ``RO_with_energy_recovery`` (pressure
exchanger option)::

    seawater ──► PX (low-pressure side) ──bypass──► P1 (HP pump) ──┐
                  │  ▲                                           M1 ──► RO ──► permeate
                  │  └────────── RO concentrate ◄────────────────┘ │
                  └──hp_feed──► P2 (booster) ─────────────────────►┘
                  └──brine──► disposal

Two forms are provided:

- :func:`build_swro_flowsheet`: a difflow ``Flowsheet`` (units, streams and
  the concentrate recycle), for the session tools and the editor.
- :func:`swro_performance`: the same plant as one differentiable function
  of the design (RO pressure, area, permeabilities, ...), returning the
  performance and the LCOW. With no pressure-exchanger mixing the RO feed is
  the intake seawater at the RO pressure and the loop needs one pass;
  with mixing it is solved by substitution (the loop gain is about the
  mixing fraction, so a few passes converge).
"""

from __future__ import annotations

import jax.numpy as jnp

from difflow.flowsheet import Flowsheet, Unit
from difflow.streams import Stream
from difflow.units.flash import Mixer
from difflow_water.economics import (
    CostingParams,
    cost_hp_pump,
    cost_pressure_exchanger,
    cost_ro,
    lcow,
)
from difflow_water.membranes.erd import PressureExchanger, PressureExchangerParams
from difflow_water.membranes.ro import ROStage, ROStageParams
from difflow_water.pumps import WaterPump, WaterPumpParams
from difflow_water.streams import SALT, WATER, combine_seawater, volumetric_flow


def build_swro_flowsheet(ro: ROStageParams, ro_pressure: float,
                         px: PressureExchangerParams | None = None,
                         pump_efficiency: float = 0.8,
                         booster_efficiency: float = 0.8) -> Flowsheet:
    """The seawater RO + pressure exchanger plant as a difflow ``Flowsheet``.

    Feed it with ``fs.add_feed("seawater", stream)``; solve with
    ``fs.solve()``. The concentrate ``ro_conc_out`` is recycled to the
    exchanger's ``ro_conc`` inlet.

    Args:
        ro: RO stage parameters.
        ro_pressure: RO feed pressure (Pa), set by both pumps.
        px: Pressure exchanger parameters (defaults if None).
        pump_efficiency: High-pressure pump efficiency (-).
        booster_efficiency: Booster pump efficiency (-).

    Returns:
        The flowsheet.
    """
    px = px or PressureExchangerParams()
    fs = Flowsheet(species_order=[WATER, SALT])
    fs.add_unit(Unit("PX", PressureExchanger(px), ["ro_conc", "seawater"],
                     ["px_hp", "brine", "px_bypass"]))
    fs.add_unit(Unit("P1", WaterPump(WaterPumpParams(outlet_pressure=ro_pressure,
                                                     efficiency=pump_efficiency)),
                     ["px_bypass"], ["p1_out"]))
    fs.add_unit(Unit("P2", WaterPump(WaterPumpParams(outlet_pressure=ro_pressure,
                                                     efficiency=booster_efficiency)),
                     ["px_hp"], ["p2_out"]))
    fs.add_unit(Unit("M1", Mixer([WATER, SALT]), ["p1_out", "p2_out"], ["ro_feed"]))
    fs.add_unit(Unit("RO", ROStage(ro), ["ro_feed"], ["ro_conc_out", "permeate"]))
    fs.add_recycle("ro_conc_out", "ro_conc")
    return fs


def swro_performance(feed: Stream, ro: ROStageParams, ro_pressure,
                     px: PressureExchangerParams | None = None,
                     pump_efficiency=0.8, booster_efficiency=0.8,
                     costing: CostingParams | None = None,
                     n_passes: int | None = None) -> dict:
    """Performance and cost of the seawater RO + pressure exchanger plant.

    Differentiable in every numeric argument (``jax.grad`` with respect to
    ``ro_pressure``, ``ro.area``, ``ro.A_w``, ...).

    Args:
        feed: Intake seawater (low pressure).
        ro: RO stage parameters.
        ro_pressure: RO feed pressure (Pa).
        px: Pressure exchanger parameters.
        pump_efficiency: High-pressure pump efficiency.
        booster_efficiency: Booster pump efficiency.
        costing: Costing parameters (WaterTAP defaults if None).
        n_passes: Loop passes; 1 when the exchanger does not mix, else 6.

    Returns:
        dict with ``LCOW`` (USD/m^3), ``SEC`` (kWh/m^3), ``recovery``
        (``recovery_mass_water`` of the RO, equal to the plant's),
        ``recovery_vol``, ``salinity_permeate``, ``salinity_brine``,
        ``work_P1``/``work_P2`` (W), ``costing`` (the :func:`lcow` dict),
        ``ro`` (the RO info) and the streams ``permeate`` and ``brine``.
    """
    px = px or PressureExchangerParams()
    if n_passes is None:
        n_passes = 1 if (isinstance(px.mixing, (int, float)) and px.mixing == 0) else 6
    ro_unit = ROStage(ro)
    exch = PressureExchanger(px)
    p1 = WaterPump(WaterPumpParams(outlet_pressure=ro_pressure, efficiency=pump_efficiency))
    p2 = WaterPump(WaterPumpParams(outlet_pressure=ro_pressure, efficiency=booster_efficiency))
    ro_feed = dict(feed)
    ro_feed["P"] = jnp.asarray(ro_pressure, dtype=jnp.float64)
    for _ in range(n_passes):
        conc, perm, ro_info = ro_unit(ro_feed)
        hp, brine, bypass, px_info = exch(conc, feed)
        o1, i1 = p1(bypass)
        o2, i2 = p2(hp)
        ro_feed = combine_seawater(o1, o2)
    units = {
        "RO": cost_ro(ro.area, costing or CostingParams()),
        "P1": cost_hp_pump(i1["work_mechanical"], costing or CostingParams()),
        "P2": cost_hp_pump(i2["work_mechanical"], costing or CostingParams()),
        "PX": cost_pressure_exchanger(px_info["flow_vol_exchanged"], costing or CostingParams()),
    }
    work = i1["work_mechanical"] + i2["work_mechanical"]
    Qp = volumetric_flow(perm)
    cst = lcow(units, work, Qp, costing)
    return {
        "LCOW": cst["LCOW"], "SEC": cst["SEC"],
        "recovery": ro_info["recovery_mass_water"],
        "recovery_vol": ro_info["recovery_vol"],
        "salinity_permeate": ro_info["salinity_permeate"],
        "salinity_brine": ro_info["salinity_concentrate"],
        "work_P1": i1["work_mechanical"], "work_P2": i2["work_mechanical"],
        "costing": cst, "ro": ro_info, "px": px_info,
        "permeate": perm, "brine": brine,
    }
