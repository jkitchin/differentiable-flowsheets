# Water Units at a Glance

A one-page map of the `difflow_water` plugin: every unit, what it takes and
makes, what model it is, and how far it has been validated. The full
treatment (equations, parameters, references) is in
[Water Treatment and Desalination Unit Operations](unit-operations-water.md).

---

(water-summary-map)=
## How the units connect (phases 1 and 1b)

```
 intake seawater (H2O + TDS)
        │
        ▼
 PressureExchanger ──bypass──► WaterPump (HP) ──┐
   ▲      │                                     ├─► ROStage / ROTrain ──► permeate (product)
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
| [Pitzer chemistry, `speciate`](unit-operations-water.md#water-chemistry) | library | element totals, T → speciation, pH, saturation indices | Harvie-Møller-Weare Pitzer model as in PHRQPITZ; PHREEQC pitzer.dat parameters | Equations read against PHRQPITZ (content verified); SI within 0.0004 and pH within 3e-5 of PHREEQC, seawater to 3x concentrate, 15-35 C |
| [`ROStage`](unit-operations-water.md#op-rostage) | palette | feed → concentrate, permeate | Solution-diffusion, film theory, plug flow; WaterTAP channel correlations; optional saturation indices | Within 0.3 % of WaterTAP RO1D (160 elements) on four cases |
| [`ROTrain`](unit-operations-water.md#op-rotrain) | palette | feed → concentrate, permeate | ROStages in series with interstage boosters | No physics of its own |
| [`PressureExchanger`](unit-operations-water.md#op-pressureexchanger) | palette | brine, feed → hp_feed, brine, bypass | Efficiency form, optional leakage and mixing | Within 0.02 % (outlet pressure) of WaterTAP in the plant, with and without leakage/mixing |
| [`WaterPump`](unit-operations-water.md#op-waterpump) | palette | inlet → outlet | Isothermal, `V dP / eta` | Within 0.24 % of WaterTAP in the plant |
| [`economics`](unit-operations-water.md#water-costing) | library | sizes, power, product flow → LCOW, SEC | WaterTAP 1.8.0 costing structure and defaults | LCOW within 0.16 % of WaterTAP |
| [`network`: `WaterNetwork`, `NetworkModel`](unit-operations-water.md#water-network) | library | network, demands, leaks, settings → heads, pressures, flows | EPANET 2.2 equations, equation-oriented Newton; valves as complementarity (sharp or smooth); INP reader; extended-period simulation with controls | Within 2.7e-5 m (Net1, Net3) and 1.6e-3 m (ky10, 920 junctions) of EPANET 2.2; over 24 h / 7 days, tank levels within 1e-4 m (Net1) / 3.4e-2 m (Net3), statuses identical |
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
- **Scaling**: the Pitzer equations are checked against their source and
  the numbers against PHREEQC; the parameters themselves are PHREEQC's.
- **Costs**: WaterTAP's default parameters, a benchmark convention, not a
  cost estimate for a real plant.

---

(water-summary-gaps)=
## Known gaps

- RO transports sea salt as one solute; ions are resolved only for scaling
  (equal rejection assumed). Saturation indices are at 1 atm, without
  kinetics; no Sr, Ba, silica or boron.
- No nanofiltration yet.
- Networks: no water quality yet; no rule-based controls or volume curves.
- No conventional treatment, thermal desalination or wastewater yet
  (`src/difflow_water/DESIGN.md`, phases 3-4).
