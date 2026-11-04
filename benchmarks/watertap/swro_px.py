"""WaterTAP reference for the difflow_water seawater RO + pressure exchanger plant.

Run in an environment with WaterTAP and IPOPT (NOT a difflow dependency)::

    pip install watertap==1.8.0
    idaes get-extensions
    python benchmarks/watertap/swro_px.py      # writes swro_px_reference.json

The plant is WaterTAP's ``RO_with_energy_recovery`` layout (pressure
exchanger option) with the seawater property package and the 1-D RO model
(plug flow, calculated mass transfer and pressure drop), simulated at a
fixed design: feed pump outlet pressure and membrane area are fixed, the
recovery is computed. ``compare_swro_px.py`` runs difflow_water on the same
cases and reports the differences.

Every case also solves the RO stage alone (feed at the RO pressure), so the
stage comparison is not mixed up with the flowsheet's.
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path

import idaes.core.util.scaling as iscale
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Mixer, Product, Separator
from idaes.models.unit_models.mixer import MomentumMixingType
from pyomo.environ import (
    ConcreteModel,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
from pyomo.network import Arc
from watertap.core.solvers import get_solver
from watertap.costing import WaterTAPCosting
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump
from watertap.unit_models.pressure_exchanger import PressureExchanger
from watertap.unit_models.reverse_osmosis_1D import (
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
    ReverseOsmosis1D,
)

HERE = Path(__file__).parent

#: the benchmark cases; compare_swro_px.py reads them back from the JSON
CASES = {
    "base": dict(flow_vol=1e-3, salinity=0.035, T=298.15, P_feed=101325.0, P_ro=65e5,
                 area=50.0, A_w=4.2e-12, B=3.5e-8, channel_height=1e-3,
                 spacer_porosity=0.85, width=5.0, permeate_pressure=101325.0,
                 px_efficiency=0.95, pump_efficiency=0.8, booster_efficiency=0.8),
    "high_pressure": dict(flow_vol=1e-3, salinity=0.035, T=298.15, P_feed=101325.0,
                          P_ro=75e5, area=80.0, A_w=4.2e-12, B=3.5e-8,
                          channel_height=1e-3, spacer_porosity=0.85, width=5.0,
                          permeate_pressure=101325.0, px_efficiency=0.95,
                          pump_efficiency=0.8, booster_efficiency=0.8),
    "brackish_like": dict(flow_vol=1e-3, salinity=0.015, T=303.15, P_feed=101325.0,
                          P_ro=30e5, area=60.0, A_w=4.2e-12, B=3.5e-8,
                          channel_height=1e-3, spacer_porosity=0.85, width=5.0,
                          permeate_pressure=101325.0, px_efficiency=0.95,
                          pump_efficiency=0.8, booster_efficiency=0.8),
    "px_leak_mix": dict(flow_vol=1e-3, salinity=0.035, T=298.15, P_feed=101325.0, P_ro=65e5,
                        area=50.0, A_w=4.2e-12, B=3.5e-8, channel_height=1e-3,
                        spacer_porosity=0.85, width=5.0, permeate_pressure=101325.0,
                        px_efficiency=0.95, pump_efficiency=0.8, booster_efficiency=0.8,
                        px_leakage=0.01, px_mixing=0.035),
}

N_ELEMENTS = 160


def _ro(m, c):
    ro = ReverseOsmosis1D(
        property_package=m.fs.properties,
        has_pressure_change=True,
        pressure_change_type=PressureChangeType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.calculated,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
        finite_elements=N_ELEMENTS,
    )
    return ro


def _fix_ro(ro, c):
    ro.A_comp.fix(c["A_w"])
    ro.B_comp.fix(c["B"])
    ro.feed_side.channel_height.fix(c["channel_height"])
    ro.feed_side.spacer_porosity.fix(c["spacer_porosity"])
    ro.permeate.pressure[0].fix(c["permeate_pressure"])
    ro.width.fix(c["width"])
    ro.area.fix(c["area"])


def _ro_results(ro):
    t = 0
    x1 = ro.length_domain.last()
    out = ro.feed_side.properties[t, x1]
    inl = ro.feed_side.properties[t, ro.length_domain.first()]
    perm = ro.mixed_permeate[t]
    return {
        "recovery_mass_water": value(ro.recovery_mass_phase_comp[t, "Liq", "H2O"]),
        "recovery_vol": value(ro.recovery_vol_phase[t, "Liq"]),
        "salinity_permeate": value(perm.mass_frac_phase_comp["Liq", "TDS"]),
        "salinity_concentrate": value(out.mass_frac_phase_comp["Liq", "TDS"]),
        "deltaP": value(ro.deltaP[t]),
        "flux_mass_water_avg": value(ro.flux_mass_phase_comp_avg[t, "Liq", "H2O"]),
        "permeate_flow_vol": value(perm.flow_vol_phase["Liq"]),
        "osmotic_pressure_feed": value(inl.pressure_osm_phase["Liq"]),
        "Re_in": value(ro.feed_side.N_Re[t, ro.length_domain.first()]),
        "Re_out": value(ro.feed_side.N_Re[t, x1]),
    }


def ro_alone(c, solver):
    """The RO stage by itself, feed at the RO pressure."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.RO = _ro(m, c)
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.RO.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)
    m.fs.feed.properties[0].pressure.fix(c["P_ro"])
    m.fs.feed.properties[0].temperature.fix(c["T"])
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "TDS"))
    iscale.calculate_scaling_factors(m)
    m.fs.feed.properties.calculate_state(
        var_args={("flow_vol_phase", "Liq"): c["flow_vol"],
                  ("mass_frac_phase_comp", ("Liq", "TDS")): c["salinity"]},
        hold_state=True)
    _fix_ro(m.fs.RO, c)
    assert degrees_of_freedom(m) == 0
    m.fs.feed.initialize()
    propagate_state(m.fs.s01)
    m.fs.RO.initialize()
    assert_optimal_termination(solver.solve(m))
    return _ro_results(m.fs.RO)


def plant(c, solver):
    """The RO + pressure exchanger plant (WaterTAP's RO_with_energy_recovery layout)."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SeawaterParameterBlock()
    fs = m.fs
    fs.feed = Feed(property_package=fs.properties)
    fs.product = Product(property_package=fs.properties)
    fs.disposal = Product(property_package=fs.properties)
    fs.P1 = Pump(property_package=fs.properties)
    fs.RO = _ro(m, c)
    fs.S1 = Separator(property_package=fs.properties, outlet_list=["P1", "PXR"])
    leak, mix = c.get("px_leakage", 0.0), c.get("px_mixing", 0.0)
    fs.PXR = PressureExchanger(property_package=fs.properties,
                               has_leakage=leak > 0, has_mixing=mix > 0)
    fs.P2 = Pump(property_package=fs.properties)
    fs.M1 = Mixer(property_package=fs.properties,
                  momentum_mixing_type=MomentumMixingType.equality, inlet_list=["P1", "P2"])

    fs.costing = WaterTAPCosting()
    for u in (fs.P1, fs.RO, fs.PXR, fs.P2):
        u.costing = UnitModelCostingBlock(flowsheet_costing_block=fs.costing)
    fs.costing.cost_process()
    fs.costing.add_LCOW(fs.product.properties[0].flow_vol)
    fs.costing.add_specific_energy_consumption(fs.product.properties[0].flow_vol)

    fs.s01 = Arc(source=fs.feed.outlet, destination=fs.S1.inlet)
    fs.s02 = Arc(source=fs.S1.P1, destination=fs.P1.inlet)
    fs.s03 = Arc(source=fs.P1.outlet, destination=fs.M1.P1)
    fs.s04 = Arc(source=fs.M1.outlet, destination=fs.RO.inlet)
    fs.s05 = Arc(source=fs.RO.permeate, destination=fs.product.inlet)
    fs.s06 = Arc(source=fs.RO.retentate, destination=fs.PXR.brine_inlet)
    fs.s07 = Arc(source=fs.PXR.brine_outlet, destination=fs.disposal.inlet)
    fs.s08 = Arc(source=fs.S1.PXR, destination=fs.PXR.feed_inlet)
    fs.s09 = Arc(source=fs.PXR.feed_outlet, destination=fs.P2.inlet)
    fs.s10 = Arc(source=fs.P2.outlet, destination=fs.M1.P2)
    TransformationFactory("network.expand_arcs").apply_to(m)

    fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    fs.properties.set_default_scaling("flow_mass_phase_comp", 1e2, index=("Liq", "TDS"))
    for blk in (fs.P1.control_volume.work, fs.P2.control_volume.work,
                fs.PXR.feed_side.work, fs.PXR.brine_side.work):
        iscale.set_scaling_factor(blk, 1e-3)
    iscale.set_scaling_factor(fs.RO.area, 1e-2)
    fs.S1.mixed_state[0].mass_frac_phase_comp
    fs.S1.PXR_state[0].flow_vol_phase["Liq"]
    iscale.calculate_scaling_factors(m)

    fs.feed.properties[0].pressure.fix(c["P_feed"])
    fs.feed.properties[0].temperature.fix(c["T"])
    fs.feed.properties.calculate_state(
        var_args={("flow_vol_phase", "Liq"): c["flow_vol"],
                  ("mass_frac_phase_comp", ("Liq", "TDS")): c["salinity"]},
        hold_state=True)
    fs.P1.efficiency_pump.fix(c["pump_efficiency"])
    fs.P1.control_volume.properties_out[0].pressure.fix(c["P_ro"])
    fs.P2.efficiency_pump.fix(c["booster_efficiency"])
    fs.PXR.efficiency_pressure_exchanger.fix(c["px_efficiency"])
    if leak > 0:
        fs.PXR.leakage_vol.fix(leak)
    if mix > 0:
        fs.PXR.mixing_vol.fix(mix)
    _fix_ro(fs.RO, c)
    dof = degrees_of_freedom(m)
    assert dof == 0, dof

    # initialize: RO first on the intake at the RO pressure, then the loop
    fs.feed.initialize()
    x0 = fs.RO.length_domain.first()
    for j in ("H2O", "TDS"):
        fs.RO.feed_side.properties[0, x0].flow_mass_phase_comp["Liq", j] = value(
            fs.feed.properties[0].flow_mass_phase_comp["Liq", j])
    fs.RO.feed_side.properties[0, x0].temperature = c["T"]
    fs.RO.feed_side.properties[0, x0].pressure = c["P_ro"]
    fs.RO.initialize()
    propagate_state(fs.s06)
    fs.PXR.brine_side.properties_in.initialize()
    propagate_state(fs.s01)
    fs.S1.mixed_state.initialize()
    fs.S1.PXR_state.calculate_state(var_args={
        ("flow_vol_phase", "Liq"): value(fs.PXR.brine_side.properties_in[0].flow_vol_phase["Liq"]),
        ("mass_frac_phase_comp", ("Liq", "TDS")): value(
            fs.S1.mixed_state[0].mass_frac_phase_comp["Liq", "TDS"]),
        ("pressure", None): value(fs.S1.mixed_state[0].pressure),
        ("temperature", None): value(fs.S1.mixed_state[0].temperature)})
    fs.S1.PXR_state[0].flow_mass_phase_comp["Liq", "TDS"].fix()
    fs.S1.initialize()
    fs.S1.PXR_state[0].flow_mass_phase_comp["Liq", "TDS"].unfix()
    propagate_state(fs.s08)
    fs.PXR.initialize()
    propagate_state(fs.s02)
    fs.P1.initialize()
    propagate_state(fs.s09)
    fs.P2.control_volume.properties_out[0].pressure.fix(c["P_ro"])
    fs.P2.initialize()
    fs.P2.control_volume.properties_out[0].pressure.unfix()
    propagate_state(fs.s03)
    propagate_state(fs.s10)
    fs.M1.initialize()
    fs.costing.initialize()
    assert_optimal_termination(solver.solve(m))

    out = {"ro": _ro_results(fs.RO)}
    out.update({
        "LCOW": value(fs.costing.LCOW),
        "SEC": value(fs.costing.specific_energy_consumption),
        "total_capital": value(fs.costing.total_capital_cost),
        "total_operating": value(fs.costing.total_operating_cost),
        "work_P1": value(fs.P1.work_mechanical[0]),
        "work_P2": value(fs.P2.work_mechanical[0]),
        "px_hp_pressure": value(fs.PXR.feed_side.properties_out[0].pressure),
        "product_flow_vol": value(fs.product.properties[0].flow_vol),
        "salinity_product": value(fs.product.properties[0].mass_frac_phase_comp["Liq", "TDS"]),
        "px_flow_vol_lp": value(fs.PXR.feed_side.properties_in[0].flow_vol),
        "px_salinity_hp_out": value(
            fs.PXR.feed_side.properties_out[0].mass_frac_phase_comp["Liq", "TDS"]),
        "ro_feed_salinity": value(fs.M1.mixed_state[0].mass_frac_phase_comp["Liq", "TDS"]),
    })
    return out


def main(path: Path = HERE / "swro_px_reference.json") -> dict:
    solver = get_solver()
    ref = {
        "generated_by": "benchmarks/watertap/swro_px.py",
        "versions": {p: metadata.version(p) for p in ("watertap", "idaes-pse", "pyomo")},
        "solver": "ipopt (idaes get-extensions)",
        "ro_finite_elements": N_ELEMENTS,
        "cases": {},
    }
    for name, c in CASES.items():
        ref["cases"][name] = {"inputs": c, "ro_alone": ro_alone(c, solver), "plant": plant(c, solver)}
        print(name, "LCOW", ref["cases"][name]["plant"]["LCOW"], file=sys.stderr)
    path.write_text(json.dumps(ref, indent=2) + "\n")
    return ref


if __name__ == "__main__":
    main()
