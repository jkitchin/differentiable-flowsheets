# difflow_refinery: Petroleum Refining as Differentiable Flowsheets

`difflow_refinery` models a refinery from the crude assay to the blended
products. A crude is characterized once, from its TBP curve and gravity,
into pseudo-components that carry their contaminants and hydrocarbon
types. Every unit downstream works on that one characterization, so a
product property has an exact gradient with respect to the assay data,
the specs and the operating variables upstream of it. Columns and
recycle loops are solved by Newton or accelerated substitution, and
their gradients are implicit-function gradients at the converged point.

```python
import jax.numpy as jnp
import difflow_refinery as dr

assay = dr.Assay(
    tbp_percent=[0, 5, 10, 30, 50, 70, 90, 95, 100],
    tbp_T=[t + 273.15 for t in [20, 60, 95, 205, 315, 430, 580, 640, 760]],
    sg=0.86,
    light_ends={"propane": 0.005, "n_butane": 0.01, "n_pentane": 0.015},
)
crude = dr.characterize(assay, composition=True)   # 29 components

whole = crude.composition.of_flows(jnp.ones(len(crude.names)))
whole.hc_type        # paraffins, naphthenes, aromatics, olefins (vol fractions)
whole.hydrogen_wt    # wt%
```

## The units

**Palette**: registered with the editor through this plugin's `register()`
and usable as a `Flowsheet` operation. **Library**: called from Python;
still differentiable, with no palette entry.

### Separation and feed preparation

| Unit | Module | Main class | Palette | In → out |
|---|---|---|---|---|
| Preheat train | `preheat` | `PreheatedCrudeUnit` | `Desalter`, `PreflashDrum`, `CrudeUnitWithPreheat` | crude from the tank → crude at the furnace inlet, with fouling |
| Crude distillation unit | `unit`, `column` | `CrudeUnit`, `CrudeColumn`, `Furnace` | `CrudeDistillationUnit` | crude → naphtha, kerosene, diesel, AGO, residue |
| Vacuum unit | `vacuum` | `VacuumColumn` | `VacuumColumn` | atmospheric residue → LVGO, HVGO, slop, vacuum residue |
| Saturated gas plant | `gasplant` | `GasPlantColumn`, `GasCompressor`, `AmineTreater` | all three | light ends → fuel gas, LPG, C3/C4 splits, stabilized naphtha; a hydrotreater product on a component table via `gasplant.hydroprocessed.hydroprocessed_feed` |

### Conversion

| Unit | Module | Main class | Palette | In → out |
|---|---|---|---|---|
| C5/C6 isomerization | `isomerization` | `IsomerizationReactor`, `IsomerizationUnit` | both | light naphtha → isomerate |
| Hydrotreater | `hydrotreating` | `Hydrotreater` | library | naphtha, kerosene or diesel → treated product, wild naphtha, off-gas; `res.fractionate(...)` → jet / diesel or light / heavy naphtha |
| Hydrocracker | `hydrocracking` | `Hydrocracker` | library | VGO → LPG, naphtha, kerosene, diesel, unconverted oil |
| Residue desulfurizer | `residue` | `ResidueDesulfurizer`, `fuel_oil_blend` | library | atmospheric residue → desulfurized residue, distillate, gas; VLSFO (0.5 wt% S) pool |
| Fluid catalytic cracker | `fcc` | `FCCUnit` | library | VGO → dry gas, C3, C4, gasoline, LCO, slurry, flue gas |
| Catalytic reformer | `reforming` | `CatalyticReformer` | library | heavy naphtha (from a hydrotreater: `NaphthaFeed.from_hydrotreater`) → reformate, net H2, LPG, fuel gas |
| Alkylation | `alkylation` | `AlkylationUnit` | library | C3-C5 olefins + isobutane → alkylate, propane, n-butane |
| Hydrogen network | `hydrogen` | `HydrogenNetwork` | library | reformer net gas, H2 plant, import → hydrotreater / hydrocracker makeup at the header purity, fuel gas, export |

### Products

| Unit | Module | Main class | Palette | In → out |
|---|---|---|---|---|
| Product blending | `blending` | `BlendPool`, `BlendComponent` | library | components → gasoline, jet, ULSD or fuel oil, with spec margins |

### Shared foundations

| Piece | Module | What it is |
|---|---|---|
| Assay and characterization | `assay` | `Assay`, `characterize`: TBP curve to pseudo-components, with an optional heavy end and contaminants per cut |
| Composition | `composition` | P/N/A/O fractions, hydrogen, sulfur and nitrogen classes per pseudo-component |
| Correlations | `correlations` | Twu, Riazi-Daubert, Lee-Kesler, Kesler-Lee, Maxwell-Bonnell |
| Column thermodynamics | `thermo` | `ColumnThermo`: Raoult with Lee-Kesler vapour pressures |
| CDU to gas plant | `gasplant.feed` | `gas_plant_feed`: crude-unit offgas and naphtha onto a `gas_components` table (cut selection, folding, water, H2S), with the folded and dropped mass reported (#326) |
| Stage-network column | `vacuum` | `StageColumn`, under the VDU, the gas plant and the hydrotreater's stripper |
| Hydroprocessing blocks | `hydroprocessing` | trickle-bed reactor, PR high-pressure separator, recycle-gas loop, stripper |
| Product properties | `products` | `product_properties`: yields, SG/API, TBP points |
| Product property estimates | `properties` | flash, freeze and smoke points, viscosity, straight-run RON/MON from a stream (#330, mostly unverified); used by `BlendComponent.from_stream` |
| Planning blocks | `planning`, and each unit's package or its `planning` submodule (e.g. `hydrotreating.planning.hdt_block`) | `cdu_block`, `gasplant_block`, `isom_block`, `hdt_block`, `hcu_block`, `fcc_block`, `reformer_block`, `alky_block`, `h2_block`, `product_value_block`, for `difflow.planning` |

## How far to trust it

The balances, the equilibrium thermochemistry, the column equations and
the published correlations are physics. Where a unit has been
cross-checked, it was against an independent implementation of the same
model: the CDU and VDU against Pyomo/IPOPT columns, and the preheat
train, gas plant and isomerization reactor against IDAES. Alkylation's
correlations reproduce the GAMS `process.gms` optimum.

The kinetic and yield constants of the hydrotreater, residue
desulfurizer, hydrocracker, FCC, reformer and the isomerization rates are **illustrative**. They give the
right trends and exact gradients, but the absolute yields are not
predictions until they are fitted to a unit's own data.

## Documentation

- [Refinery units at a glance](../../docs/refinery-summary.md): the
  process map, a table of every unit with its model and validation
  status, and the known gaps.
- [Refinery Unit Operations](../../docs/unit-operations-refinery.md):
  equations, specs, outputs and references for each unit.
- Examples: `examples/33_refinery_gasoline_blending.ipynb` through
  `examples/39_refinery_isomerization.ipynb`, and
  `examples/40_refinery_flowsheet.ipynb`, a small whole refinery (CDU,
  gas plant, naphtha and distillate hydrotreaters, reformer, product
  pools, hydrogen balance).
- Tests: `tests/refinery/`.
