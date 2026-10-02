# Refinery Units at a Glance

A one-page map of the `difflow_refinery` plugin: every unit, what it takes
and makes, what model it is, and how far it has been validated. The full
treatment (equations, specs, references) is in
[Refinery Unit Operations](unit-operations-refinery.md); each row below links
to its section there.

---

(refinery-summary-map)=
## How the units connect

```
                 crude assay (TBP, SG, S, N, CCR, Ni+V)
                              │  characterize + composition
                              ▼
 tank ─► preheat train ─► desalter ─► preflash ─► furnace + atmospheric column (CDU)
                                                    │
     ┌──────────────┬──────────────┬────────────────┼────────────────┐
  light ends     naphtha        kerosene / diesel          atmospheric residue
     │              │                    │                          │
  gas plant     ┌───┴────────┐       hydrotreater             vacuum unit (VDU)
     │       light naphtha  heavy naphtha  │                  ┌─────┴─────┐
     │          │              │           │                 VGO       vacuum residue
     │    isomerization   reformer ◄─ H2 ─►│                ┌─┴────────┐
     │          │              │           │               FCC     hydrocracker
     │          │              │           │                │          │
     │          │              │           │      C3/C4 olefins    jet / diesel
     │          │              │           │           │
     │          │              │           │      alkylation
     ▼          ▼              ▼           ▼           ▼
   LPG     isomerate      reformate    ULSD / jet    alkylate ──► blend pools
                                                                 (gasoline, jet,
                                                                  ULSD, fuel oil)
```

Every arrow is a differentiable connection: a product property has an exact
gradient with respect to the assay data, the specs and the operating
variables upstream of it.

The hydroskimming part of the map (CDU, gas plant, naphtha hydrotreater,
reformer, distillate hydrotreater, residue to fuel oil, and the four pools)
runs end to end in `examples/40_refinery_flowsheet.ipynb`. Some of its
arrows are bridges written in that notebook rather than library
connections; they are listed under [Known gaps](#refinery-summary-gaps).

---

(refinery-summary-foundations)=
## Shared foundations

These are not units, but every unit reads them.

| Piece | Module | What it does |
|---|---|---|
| Assay and characterization | `Assay`, `characterize` | A TBP curve and gravity cut into pseudo-components (Twu 1984 critical properties by default). With a `HeavyEnd`, the curve extends into the vacuum range, closed by a residue lump, with sulfur, nitrogen, CCR, Ni+V and asphaltenes carried per cut. One characterization serves the CDU, the VDU and the blend pool. [Details](unit-operations-refinery.md#characterising-a-crude) |
| Composition | `difflow_refinery.composition` | Per pseudo-component: paraffin/naphthene/aromatic/olefin volume fractions, hydrogen content, five sulfur classes and basic/non-basic nitrogen. Estimated by Riazi-Daubert or n-d-M; measured PIONA, SARA or hydrogen data overrides the estimate per cut. [Details](unit-operations-refinery.md#hydrocarbon-type-hydrogen-and-heteroatom-classes) |
| Correlations | `difflow_refinery.correlations` | Twu, Riazi-Daubert, Lee-Kesler, Kesler-Lee and Maxwell-Bonnell, each written once. |
| Column thermodynamics | `ColumnThermo` | Raoult's law with Lee-Kesler vapour pressures and ideal-gas-path enthalpies, vectorised over stages and components. |
| Stage-network column | `vacuum.StageColumn` | The equation-oriented column the VDU, the gas plant and the hydrotreater's stripper are built on. [Details](unit-operations-refinery.md#the-stage-network-column) |
| Hydroprocessing blocks | `difflow_refinery.hydroprocessing` | A trickle-bed reactor around any kinetic model, a Peng-Robinson high-pressure separator, the recycle-gas loop and a steam stripper. Shared by the hydrotreater and the hydrocracker. [Details](unit-operations-refinery.md#hydroprocessing-building-blocks) |

---

(refinery-summary-units)=
## The units

**Palette** means the unit is registered with the editor and can be placed in
a `Flowsheet` as an operation. A **library** unit is called from Python (it is
still differentiable and usable inside a `Flowsheet` function); it has no
palette entry.

**Planning** names the function that wraps the unit as a block for
delta-base planning ([`difflow.planning`](planning.md)).

### Separation and feed preparation

| Unit | Main class (palette name) | In → out | Model | Planning | Validation | Example |
|---|---|---|---|---|---|---|
| [Preheat train](unit-operations-refinery.md#the-preheat-train) | `PreheatedCrudeUnit` (palette: `Desalter`, `PreflashDrum`, `CrudeUnitWithPreheat`) | crude from the tank → crude at the furnace inlet | Exchanger train heated by the column's pumparounds and products, solved together with the column; desalter; three-phase preflash drum; Ebert-Panchal fouling | `cdu_block` | Drum and exchangers against IDAES 2.10 unit models on the same thermo | 37 |
| [Crude distillation unit](unit-operations-refinery.md#the-crude-unit) | `CrudeUnit` (palette: `CrudeDistillationUnit`) | crude → naphtha, kerosene, diesel, AGO, residue | Fired heater solved with an equation-oriented MESH column; side strippers, pumparounds, steam | `cdu_block` | Against an independent Pyomo/IPOPT column and IDAES property packages ([details](unit-operations-refinery.md#validation)) | 35, 36, 40 |
| [Vacuum unit](unit-operations-refinery.md#the-vacuum-unit) | `VacuumColumn` (palette: `VacuumColumn`) | atmospheric residue → LVGO, HVGO, slop, vacuum residue | Stage-network column at vacuum with packed beds; contaminants carried per cut | — | Against an independent Pyomo/IPOPT model, equilibrium and Murphree beds ([details](unit-operations-refinery.md#validation-the-vacuum-unit)) | 34, 36 |
| [Saturated gas plant](unit-operations-refinery.md#the-saturated-gas-plant) | `GasPlantColumn`, `GasCompressor`, `AmineTreater` (all three on the palette) | light ends + naphtha → fuel gas, LPG, C3/C4 splits, stabilized naphtha | Cubic-EOS (PR or SRK) stage columns, staged compressor with knock-outs, amine treating as a removal fraction | `gasplant_block` | Debutanizer against IDAES `TrayColumn`; splitter against IDAES flashes ([details](unit-operations-refinery.md#validation-the-gas-plant)) | 38, 40 |

### Conversion

| Unit | Main class | In → out | Model | Planning | Validation | Example |
|---|---|---|---|---|---|---|
| [C5/C6 isomerization](unit-operations-refinery.md#c5c6-light-naphtha-isomerization) | `IsomerizationReactor`, `IsomerizationUnit` (both on the palette) | light naphtha → isomerate | Adiabatic approach-to-equilibrium bed on ideal-gas thermochemistry; optional deisopentanizer and deisohexanizer recycle | `isom_block` | Equilibrium layer against IDAES `GibbsReactor` ([details](unit-operations-refinery.md#validation-the-isomerization-unit)); rate constants illustrative | 39 |
| [Hydrotreater](unit-operations-refinery.md#the-hydrotreater) | `Hydrotreater` (library) | naphtha, kerosene or diesel → treated product, wild naphtha, off-gas; optionally jet / diesel (or light / heavy naphtha) | Trickle-bed HDS by sulfur class (LHHW, H2S-inhibited), HDN, aromatics saturation with equilibrium; charge-heater duty; HP separator, H2 recycle, stripper; optional TBP-split product fractionator (`res.fractionate`); diesel and naphtha (`NAPHTHA_HDT_PARAMS`) constant sets | `hdt_block` | Balances and gradients only; no literature cross-check; constants illustrative | 40 |
| [Hydrocracker](unit-operations-refinery.md#the-hydrocracker) | `Hydrocracker` (library) | VGO → LPG, naphtha, kerosene, diesel, unconverted oil | Pretreat bed (hydrotreating kinetics) then cracking bed on continuous lumping or discrete lumps, organic-N inhibition; TBP-split fractionator; UCO recycle | `hcu_block` | Balances and gradients only; no literature cross-check; constants illustrative | — |
| [Fluid catalytic cracker](unit-operations-refinery.md#the-fluid-catalytic-cracker) | `FCCUnit` (library) | VGO → dry gas, C3, C4, gasoline, LCO, slurry, flue gas | 3-, 4- or 5-lump riser and coke-burning regenerator solved together for the heat balance; TBP-split main fractionator | `fcc_block` | Balances and gradients only; no literature cross-check; constants illustrative ([details](unit-operations-refinery.md#fcc-what-is-tested-and-what-is-not)) | — |
| [Catalytic reformer](unit-operations-refinery.md#catalytic-reforming) | `CatalyticReformer` (library) | hydrotreated heavy naphtha (`NaphthaFeed.from_hydrotreater`, [#327](unit-operations-refinery.md#refinery-hydrotreated-naphtha-downstream)) → reformate, net H2, LPG, fuel gas | 29 lumps by carbon number, equilibrium from Gibbs energies; three adiabatic beds with fired heaters; PR separator and H2 recycle; component-split stabilizer; feed sulfur to H2S and reformate S (trace) | `reformer_block` | Balances and gradients only; no literature cross-check; constants illustrative | 40 |
| [Alkylation](unit-operations-refinery.md#alkylation) | `AlkylationUnit` (library) | C3-C5 olefins + isobutane → alkylate, propane, n-butane | Sauer-Colville-Burwick yield and octane correlations; per-olefin stoichiometry; shortcut DIB, depropanizer and debutanizer; isobutane recycle | `alky_block` | The correlation layer reproduces the GAMS `process.gms` optimum (profit 1161.3366) | — |
| [Hydrogen network](unit-operations-refinery.md#the-hydrogen-network) | `HydrogenNetwork` (library) | reformer net gas, H2 plant, import → hydrotreater / hydrocracker makeup, fuel gas, export | Header balance by species; optional PSA (recovery, product purity); ordered swing sources with capacities; purity and makeup partial-pressure specs; makeup purity fed back into the hydrotreaters by substitution | `h2_block` | Balances close by construction; gradients against finite differences; PSA defaults illustrative | — |
| [Residue desulfurizer](unit-operations-refinery.md#residue-desulfurization-and-fuel-oil) | `ResidueDesulfurizer` (library) | atmospheric residue → desulfurized residue (VLSFO base), distillate, gas | Trickle beds: HDS by sulfur class plus refractory residue sulfur (LHHW, H2S-inhibited), HDM of Ni+V onto the catalyst, CCR reduction, small 538 C+ conversion; once-through treat gas, ideal product split; `fuel_oil_blend` to a 0.5 wt% S pool | — | Balances and gradients only; no literature cross-check; constants illustrative | — |

### Products

| Unit | Main class | In → out | Model | Planning | Validation | Example |
|---|---|---|---|---|---|---|
| [Product blending](unit-operations-refinery.md#product-blending) | `BlendPool`, `BlendComponent` (library) | components → gasoline, jet, ULSD or fuel oil, with spec margins | Nonlinear blending rules (Ethyl RT-70 octane, RVP index, Refutas viscosity); distillation and cetane index computed from the blend; flash, freeze and smoke points, viscosity and straight-run octane estimated from a stream (`properties`, unverified) | `product_value_block` | Rules tested against published worked examples (RVP index, Refutas) | 33, 40 |

---

(refinery-summary-status)=
## What "illustrative" means here

Two kinds of number appear in these units, and they deserve different trust.

- **Physics and published correlations**: mass, element and energy balances,
  equilibrium from thermochemistry, the column equations, and the
  characterization and blending correlations. These are transferable, and
  where a unit has been cross-checked it is against an independent
  implementation of the same model (Pyomo/IPOPT, IDAES), not against a
  commercial simulator.
- **Kinetic and yield constants in the conversion units** (hydrotreater,
  residue desulfurizer, hydrocracker, FCC, reformer, isomerization rates, alkylation octane
  temperature terms): **illustrative**. They were chosen to give plausible
  behaviour, not taken from a published parameter set, and they must be
  fitted to the unit's own data (for example with `difflow.estimation`)
  before the yields are used to plan. The trends and the gradients are
  meaningful; the absolute yields are not predictions.

Every citation, equation number or coefficient that could not be checked
against its source is marked **(unverified)** in the full documentation and
in the code.

---

(refinery-summary-gaps)=
## Known gaps

- **Fractionation in the conversion units is simplified.** The FCC main
  fractionator, the hydrocracker fractionator and the hydrotreater's
  optional product fractionator are TBP splits, the
  reformer's stabilizer is a component split, and alkylation uses shortcut
  columns. The gas plant's rigorous columns reached `main` after these units
  were built and are not yet wired in.
- **No literature cross-check** for the hydrotreater, residue desulfurizer,
  hydrocracker, FCC or reformer: the papers named in their issues could not be obtained.
- **Not built:** example notebooks for the hydrocracker, FCC and alkylation
  (the hydrotreater and the reformer appear only inside the whole-refinery
  example 40); the 10-lump FCC scheme;
  mechanistic alkylation kinetics; catalyst-activity tracking wired into
  `difflow.reconciliation.tracking`.
- **Known model defect:** with the illustrative reformer constants, a rich
  (high-naphthene) naphtha makes less net H2 than a lean one, the reverse of
  commercial experience ([details](unit-operations-refinery.md#catalytic-reforming)).
- **Product property estimates are unverified** (#330): flash, smoke point,
  viscosity and straight-run octane come from correlations recalled but not
  checked against their sources; the freeze point is an n-paraffin
  solubility model on checked melting points
  ([details](unit-operations-refinery.md#estimated-product-properties)).
  A measured value overrides each.
- **Boiling ranges are TBP, not ASTM D86**, throughout.
- **Connections between units.** `examples/40_refinery_flowsheet.ipynb`
  joins the CDU, gas plant, naphtha and distillate hydrotreaters, reformer
  and four pools, and its last section lists the bridges it had to write:
  CDU streams into a `gas_components` table, the hydrotreater's product grid
  into a naphtha splitter or a `NaphthaFeed` (which reads the untreated
  composition), a jet/diesel split after a distillate hydrotreater, a
  hydrogen header, and a fuel-oil route for the atmospheric residue.
  The hydrogen header is now a library block, `difflow_refinery.hydrogen`
  (#329), and the fuel-oil route a library too (`difflow_refinery.residue`,
  #331: a residue desulfurizer and the VLSFO pool); example 40 does not use
  either yet ([the residue replacement](unit-operations-refinery.md#in-the-whole-refinery-example)).
