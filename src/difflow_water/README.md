<p align="center">
  <img src="../../images/plugins/difflow-water-logo.svg" alt="difflow_water" width="420">
</p>

# difflow_water: Desalination and Water Treatment as Differentiable Flowsheets

`difflow_water` models water treatment and desalination plants in difflow.
It covers seawater reverse osmosis: seawater properties, Pitzer
speciation with mineral saturation indices (scaling), RO stages and trains,
a pressure exchanger, pumps and the levelized cost of water. Every result
is differentiable, so the gradient of the cost of water with respect to the
RO pressure, membrane area or membrane permeability is exact.

The units follow WaterTAP's physics and costing conventions and are checked
against WaterTAP 1.8.0 (`benchmarks/watertap/`): on three seawater and
brackish cases the RO stage, the pump energies and the LCOW agree to within
0.3 %.

```python
import jax
import difflow_water as dw

feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035)    # 1 L/s, 35 g/kg
ro = dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=50.0)
r = dw.swro_performance(feed, ro, ro_pressure=65e5)
r["recovery"], r["SEC"], r["LCOW"]       # ~0.42, ~2.47 kWh/m3, ~0.41 USD/m3

dLCOW_dP = jax.grad(lambda P: dw.swro_performance(feed, ro, P)["LCOW"])(65e5)
```

## The units

**Palette**: registered with the editor through `register()` and usable as
a `Flowsheet` operation. **Library**: called from Python; still differentiable.

| Unit | Kind | What it is |
|---|---|---|
| `seawater` | library | Density, viscosity, osmotic coefficient and pressure (Sharqawy et al. 2010) |
| `pitzer`, `speciation` | library | Pitzer activities (HMW/PHRQPITZ), speciation, pH, saturation indices; checked against PHREEQC |
| `ROStage` | palette | Seawater RO: solution-diffusion, film theory, plug flow, channel friction; optional scaling |
| `ROTrain` | palette | RO stages in series with interstage boosters |
| `PressureExchanger` | palette | Isobaric energy recovery, with a feed bypass outlet |
| `WaterPump` | palette | Isothermal pump, `V dP / eta` |
| `economics` | library | LCOW and specific energy, WaterTAP's default costing |
| `swro_performance`, `build_swro_flowsheet` | library | The seawater RO + pressure exchanger plant |

## Documentation

- [Water units at a glance](../../docs/water-summary.md)
- [Water Treatment and Desalination Unit Operations](../../docs/unit-operations-water.md):
  equations, parameters, the WaterTAP comparison and the references table
  with each source's verification status.
- Example: `examples/70_water_swro_design.ipynb` (LCOW minimization with exact gradients).
- Plan for the rest of the plugin: [DESIGN.md](DESIGN.md).
