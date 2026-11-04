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
| [`seawater` properties](unit-operations-water.md#water-seawater-properties) | library | (T, S) → ρ, μ, φ, π | Sharqawy et al. 2010 Eqs. 8, 22, 23, 49 | Coefficients read against the paper (content verified); within Sharqawy's stated accuracy of TEOS-10 (density 0.07 %, cp 0.15 %, osmotic pressure 0.6 %; notebook 77) |
| [Pitzer chemistry, `speciate`](unit-operations-water.md#water-chemistry) | library | element totals, T → speciation, pH, saturation indices | Harvie-Møller-Weare Pitzer model as in PHRQPITZ; PHREEQC pitzer.dat parameters | Equations read against PHRQPITZ (content verified); SI within 0.0004 and pH within 3e-5 of PHREEQC, seawater to 3x concentrate, 15-35 C |
| [`ROStage`](unit-operations-water.md#op-rostage) | palette | feed → concentrate, permeate | Solution-diffusion, film theory, plug flow; WaterTAP channel correlations; optional saturation indices | Within 0.3 % of WaterTAP RO1D (160 elements) on four cases |
| [`ROTrain`](unit-operations-water.md#op-rotrain) | palette | feed → concentrate, permeate | ROStages in series with interstage boosters | No physics of its own |
| [`PressureExchanger`](unit-operations-water.md#op-pressureexchanger) | palette | brine, feed → hp_feed, brine, bypass | Efficiency form, optional leakage and mixing | Within 0.02 % (outlet pressure) of WaterTAP in the plant, with and without leakage/mixing |
| [`WaterPump`](unit-operations-water.md#op-waterpump) | palette | inlet → outlet | Isothermal, `V dP / eta` | Within 0.24 % of WaterTAP in the plant |
| [`economics`](unit-operations-water.md#water-costing) | library | sizes, power, product flow → LCOW, SEC | WaterTAP 1.8.0 costing structure and defaults | LCOW within 0.16 % of WaterTAP |
| [`network`: `WaterNetwork`, `NetworkModel`](unit-operations-water.md#water-network) | library | network, demands, leaks, settings → heads, pressures, flows | EPANET 2.2 equations, equation-oriented Newton; valves as complementarity (sharp or smooth); INP reader; extended-period simulation with controls | Within 2.7e-5 m (Net1, Net3) and 1.6e-3 m (ky10, 920 junctions) of EPANET 2.2; over 24 h / 7 days, tank levels within 1e-4 m (Net1) / 3.4e-2 m (Net3), statuses identical; chlorine within 0.01 mg/L, age 0.16 h (Net1); pump energy and differentiable speed schedules |
| [`MVCDesalinator`, `mvc_performance`](unit-operations-water.md#op-mvcdesalinator) | palette | feed → distillate, brine | WaterTAP's single-stage MVC plant: evaporator, compressor, condenser, preheaters, pumps, costing; one Newton solve | Within 4e-7 of WaterTAP 1.8.0 on every output; its LCOW optimum reproduced with `jax.grad` + SLSQP (1e-7) |
| [`MEDDesalinator`, `med_performance`](unit-operations-water.md#op-meddesalinator) | palette | feed → distillate, brine | Parallel-feed MED: per-effect mass, salt and energy balances on Sharqawy properties | No WaterTAP counterpart; balances only (no fitted coefficients) |
| [`ActivatedSludgePlant`, `activated_sludge_plant`](unit-operations-water.md#op-activatedsludgeplant) | palette | influent → effluent, wastage | BSM1: ASM1 reactors + Takács settler, steady state, one equation set | BSM1 report Tables 6-7 within 3.2e-3 (printed digits); WaterTAP's BSM1 example within 1.3e-7 |
| [`AnaerobicDigester`, `anaerobic_digester`](unit-operations-water.md#op-anaerobicdigester) | palette | influent → effluent, biogas | ADM1 as implemented for BSM2 (Rosen & Jeppsson 2006), algebraic pH, steady state | Their 13-digit verification case within 1.7e-7; WaterTAP's ADM1 example within 2.5e-4 |
| [`ASM1Reactor`](unit-operations-water.md#op-asm1reactor) | palette | inlet → outlet | ASM1 CSTR at steady state; KLa or DO setpoint | BSM1 Table 6 reproduced reactor by reactor within 0.4 % (notebook 79) |
| [`TakacsSettler`](unit-operations-water.md#op-takacssettler) | palette | feed → effluent, underflow | 10-layer Takács settler, steady state | BSM1 Table 7 layers within 0.13 % from the printed feed (notebook 79) |
| [`Sedimentation`](unit-operations-water.md#op-sedimentation) | palette | feed → clarified, sludge | Ideal basin, overflow-rate removal, Stokes settling | Derived from the ideal-basin assumptions; Stokes as stated in Messina et al. 2015 (read) |
| [`Filtration`](unit-operations-water.md#op-filtration) | palette | feed → filtrate, retained | Yao colloid filtration (clean bed), Carman-Kozeny head loss | Yao's efficiency cross-checked against Messina et al. 2015; no independent numerical check |
| [`Disinfection`](unit-operations-water.md#op-disinfection) | palette | feed → treated | Chick-Watson in CSTRs in series; CT, log inactivation | Fitted to the EPA SWTR CT table for *Giardia* (within 3 %); reproduces 3 logs at the table's CT (notebook 78) |
| [`GACAdsorber`](unit-operations-water.md#op-gacadsorber) | palette | feed → treated, removed | WaterTAP's CPHSDM (Hand et al. 1984) | Within 2e-14 of WaterTAP 1.8.0 on four cases |
| [thermal properties](unit-operations-water.md#water-thermal-properties) | library | (T, S, P) → c_p, h, h_fg, p_v, BPE | Sharqawy 2010 Eqs. 9, 36, 37, 43, 53-55; Nayar 2016 digits via WaterTAP | Content verified to the printed digits; brine vapor pressure equal to WaterTAP's to 1e-9 |
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
- **MVC**: equal to WaterTAP's model to 4e-7: an implementation check
  (same equations, same properties), not a plant validation. MED rests on
  mass and energy balances only and has not been cross-checked.
- **Costs**: WaterTAP's default parameters, a benchmark convention, not a
  cost estimate for a real plant.

---

(water-summary-gaps)=
## Known gaps

Every unit is taken on its own, with its check, in notebooks 77-79
([per-unit examples and checks](unit-operations-water.md#water-unit-checks)).
Against WaterTAP 1.8.0 the plugin covers its seawater RO, MVC, GAC, ASM1
(BSM1) and ADM1 flowsheets, about 15-20 % of WaterTAP's unit models. The
full list of what is missing (nanofiltration, electrodialysis, ion
exchange, the zero-order library, BSM2, ...) is in
[Coverage relative to WaterTAP](unit-operations-water.md#water-watertap-coverage).

- RO transports sea salt as one solute; ions are resolved only for scaling
  (equal rejection assumed). Saturation indices are at 1 atm, without
  kinetics; no Sr, Ba, silica or boron.
- No nanofiltration yet.
- Networks: water quality is not differentiable; no rule-based controls,
  volume curves or quality sources.
- Thermal: no MSF; MED without inter-effect preheaters or flash boxes and
  without an independent cross-check.
- Treatment: no coagulation, softening, ion exchange, UV or UF/MF; filters
  are clean-bed; no treatment costing.
- Wastewater: steady state only, ASM1 at 15 C; no ASM2d or ASM3 yet; no
  ASM1-ADM1 interfaces; no costing.
