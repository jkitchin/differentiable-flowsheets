# Water Units at a Glance

A one-page map of the `difflow_water` plugin: every unit, what it takes and
makes, what model it is, and how far it has been validated. The full
treatment (equations, parameters, references) is in
[Water Treatment and Desalination Unit Operations](unit-operations-water.md).

---

(water-summary-map)=
## How the units connect (phase 1)

```
 intake seawater (H2O + TDS)
        │
        ▼
 PressureExchanger ──bypass──► WaterPump (HP) ──┐
   ▲      │                                     ├─► ROStage ──► permeate (product)
   │      └──hp_feed──► WaterPump (booster) ────┘       │
   │                                                    │
   └──────────────── concentrate ◄──────────────────────┘
          │
          └──brine──► disposal

 economics.lcow(...) ◄── RO area, pump works, PX flow
```

---

(water-summary-units)=
## Units

| Unit | Kind | In → out | Model | Validated how |
|---|---|---|---|---|
| [`seawater` properties](unit-operations-water.md#water-seawater-properties) | library | (T, S) → ρ, μ, φ, π | Sharqawy et al. 2010 Eqs. 8, 22, 23, 49 | Coefficients read against the paper (content verified); WaterTAP uses the same equations |
| [`ROStage`](unit-operations-water.md#op-rostage) | palette | feed → concentrate, permeate | Solution-diffusion, film theory, plug flow; WaterTAP channel correlations | Within 0.3 % of WaterTAP RO1D (160 elements) on three cases |
| [`PressureExchanger`](unit-operations-water.md#op-pressureexchanger) | palette | brine, feed → hp_feed, brine, bypass | Efficiency form, equal volumes, optional mixing | Within 0.02 % (outlet pressure) of WaterTAP in the plant |
| [`WaterPump`](unit-operations-water.md#op-waterpump) | palette | inlet → outlet | Isothermal, `V dP / eta` | Within 0.24 % of WaterTAP in the plant |
| [`economics`](unit-operations-water.md#water-costing) | library | sizes, power, product flow → LCOW, SEC | WaterTAP 1.8.0 costing structure and defaults | LCOW within 0.16 % of WaterTAP |
| [`swro_performance` / `build_swro_flowsheet`](unit-operations-water.md#water-swro) | library | intake → LCOW, SEC, recovery, ... | The plant above | Against WaterTAP's `RO_with_energy_recovery` layout |

**Palette**: registered with the editor through the plugin's `register()`
and usable as a `Flowsheet` operation. **Library**: called from Python;
still differentiable.

---

(water-summary-trust)=
## What to trust

- **Seawater properties**: published correlations, coefficients checked
  against the source, inside their stated ranges.
- **RO and plant numbers**: they reproduce WaterTAP to 0.3 %. That checks
  the implementation against an independent code with the same physics,
  not the physics against a plant. The channel correlations behind the
  mass-transfer coefficient and pressure drop are WaterTAP's and are
  unverified against their papers.
- **Costs**: WaterTAP's default parameters, a benchmark convention, not a
  cost estimate for a real plant.

---

(water-summary-gaps)=
## Known gaps

- Seawater only; no ion speciation, carbonate chemistry or scaling limits yet.
- No distribution networks, conventional treatment, thermal desalination or
  wastewater yet (`src/difflow_water/DESIGN.md`, phases 2-4).
