# Refinery Unit Operations

This document covers the `difflow_refinery` plugin. It characterises a crude from its assay, separates it in a crude distillation unit (CDU): a fired heater and an atmospheric column, with side strippers, pumparounds and stripping steam. The products are reported the way a refinery reads them: yields, API gravities and TBP ranges. The atmospheric residue goes on to a vacuum distillation unit (VDU), and finished components are blended into products in a nonlinear blending pool.

---

(refinery-overview)=
## Overview

The `difflow_refinery` plugin provides:

- **Assay characterisation** (`Assay`, `characterize`): a TBP curve plus a gravity, cut into pseudo-components with the standard petroleum correlations. Light ends (C1--C6) are kept as real species. With a `HeavyEnd` the curve is carried into the vacuum range and closed by a residue lump, and sulfur, nitrogen, CCR, Ni+V and asphaltenes are carried per component. This **one characterization** is what the crude unit, the vacuum column and the blend pool all read.
- **Column thermodynamics** (`ColumnThermo`): vectorised over stages and components.
  - Raoult's law with Lee-Kesler vapour pressures.
  - Ideal-gas-path enthalpies.
  - Stripping steam as a vapour that condenses only as free water in the overhead drum.
- **The atmospheric column** (`CrudeColumn`): an equation-oriented MESH model, with side strippers, pumparounds, bottom and stripper steam, and a total or partial condenser.
- **The furnace** (`Furnace`): solved *with* the column, so an overflash spec sets the coil outlet temperature, as it does in operation.
- **Product properties** (`product_properties`, `products.gaps`): rates, volume and mass yields, SG/API, and TBP 5/10/50/90/95 points. Also the 5--95 gaps between neighbouring cuts.
- **`CrudeUnit`**: the assembly a planner means by "the CDU": assay in, yield table out.
- **`CrudeDistillationUnit`**: the same unit behind difflow's operation protocol, for a `Flowsheet`, JSON and the editor.
- **`VacuumColumn`** (`difflow_refinery.vacuum`): the vacuum unit, atmospheric residue to LVGO, HVGO, slop and vacuum residue, with contaminants carried per cut. It runs on the crude unit's own pseudo-components, so the CDU residue feeds it directly in a `Flowsheet`.
- **Correlations** (`difflow_refinery.correlations`): Twu, Riazi-Daubert, Lee-Kesler, Kesler-Lee and Maxwell-Bonnell, each written once, for all three of the above.
- **Product blending** (`BlendPool`, `BlendComponent`): gasoline, jet, ULSD and fuel-oil pools with the nonlinear blending rules, signed spec margins and LP back-off. A library for optimisation and planning, not a palette operation.

Everything is differentiable with `jax`. A product yield, a gravity or a furnace duty has an exact gradient with respect to:

- every spec value;
- the crude rate and preheat temperature;
- the assay data behind the thermodynamics (TBP points, gravity).

The column is solved by damped Newton. Gradients are implicit-function gradients at the converged point, not derivatives taped through the iterations.

---

(refinery-installation)=
## Installation

The plugin ships with difflow and registers itself through the `difflow.plugins` entry point:

```python
import difflow_refinery as dr
```

---

(refinery-characterisation)=
## Characterising a crude

A crude is too many molecules to list. It is described by its true boiling point (TBP) curve and a gravity, and the curve is cut into narrow boiling ranges, each of which becomes one pseudo-component:

```python
from difflow_refinery import Assay, characterize

assay = Assay(
    tbp_percent=[0, 5, 10, 30, 50, 70, 90, 95, 100],
    tbp_T=[t + 273.15 for t in [20, 60, 95, 205, 315, 430, 580, 640, 760]],
    sg=0.86,
    light_ends={"propane": 0.005, "n_butane": 0.01, "n_pentane": 0.015},
)
crude = characterize(assay)          # default cut points, Twu critical properties
crude.names, crude.Tb, crude.sg, crude.volume_fraction
```

- **Interpolation:** the curve is interpolated monotonically (PCHIP).
- **Default cut widths:** 20 K below 400 °C, 40 K to 600 °C, 100 K above that (`DEFAULT_CUT_WIDTHS`).
- **Gravity:** a bulk SG is distributed over the cuts at a constant Watson K. Alternatively, pass `sg_curve=` to give the gravity cut by cut.
- **Critical-property correlations** (`CRITICAL_METHODS`):
  - `"twu"` (the default; Twu 1984 as published, also named `"twu_1984"`);
  - `"twu_legacy"` (the crude unit's coding before #301; see below);
  - `"riazi_daubert_1987"`;
  - `"riazi_daubert_1980"`;
  - `"lee_kesler"`.

  Acentric factors are chosen so that each pseudo-component boils at its own `Tb`.

The cut points fix the number of pseudo-components and, with it, the shape of the column's equations. Keep them fixed when differentiating with respect to the assay.

This is not an assay library. Curated assays are proprietary data; the module characterises the curve the caller brings.

(refinery-heavy-end)=
### The heavy end and contaminants

An atmospheric column only needs the crude to its residue. A vacuum column needs pseudo-components to 750--800 °C, past where any TBP distillation stops. `HeavyEnd` adds them, and is opt-in. An assay without one characterizes exactly as before, and `tests/refinery/test_cdu_baseline.py` pins that.

```python
import difflow_refinery as dr

assay = dr.Assay([5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                 [t + 273.15 for t in (60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680)],
                 sg=0.86, light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(),                     # T_max 800 C, lump at 950 C, MW 1500
                 sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=6.0,
                 nickel_vanadium_wppm=60.0, asphaltenes_wt=3.0)
char = dr.characterize(assay)            # method "twu" by default
char.sulfur, char.ccr                    # per component, mass fraction
char.pseudo_components()                 # the vacuum column's property table
```

- **The curve.** With a heavy end the TBP curve is drawn on a probability scale, `z = Phi^-1(x)` against T. Inside the data it is a monotone C1 cubic in `z`; beyond it, the least-squares line through the last `n_tail` points. Because the curve is open at both ends, the percentages must lie strictly inside (0, 100), there must be at least 3 of them, and the last temperature must be below `T_max`.
- **The cuts** run to `T_max` (default 800 °C) on `DEFAULT_HEAVY_CUT_WIDTHS`. Everything above is one **residue lump** whose `Tb`, `MW` and optionally `SG` are set directly (`residue_Tb`, `residue_mw`, `residue_sg`), because Twu's n-alkane reference has no root past about 820--840 °C TBP. The lump's critical constants are still computed, so the EOS and the vapour pressure stay defined.
- **Contaminants** (`CONTAMINANTS`): bulk sulfur, nitrogen, CCR, Ni+V and asphaltenes are distributed over the cuts by a logistic in boiling point, scaled so they recombine *exactly* to the bulk. Measured curves can be given for S, N and CCR (`sulfur_curve=` and so on). Light ends carry none.
- **Differentiable.** Everything is a function of the assay data. A gradient with respect to one TBP point matches central differences to 1e-6 relative, with the cut points held fixed.

**Which Twu.** The heavy end made the correlations disagree visibly. Twu's molecular weight had three codings in the package. The crude unit's divided Twu's Rankine constants by `sqrt(1.8)` while taking the square root of `Tb` in Rankine, which under-corrects aromatics (naphthalene 10 % low, phenanthrene 15 % low). The vacuum unit's coding matches the 1984 paper and two independent implementations. On the reference set the published form has a molecular-weight AAD of 0.5 %, against 2.4 % for the old coding. `"twu"` is now the published form, for every unit. The old coding is kept as `"twu_legacy"`, which reproduces the crude unit's earlier results to round-off (`tests/refinery/test_cdu_baseline.py`). On the test crude below the change moves the coil outlet by 1.7 K and the fired duty by about 3 %. The yields do not move, because they are specs.

---

(refinery-column)=
## The atmospheric column

An atmospheric crude column is a main column with no reboiler:

- The crude arrives partly vaporised from the furnace and is stripped with steam.
- **Side products** are drawn as liquid from intermediate stages. Each is usually finished in a steam-stripped side stripper, whose vapour returns one stage above the draw.
- **Pumparounds** draw liquid, cool it outside the column and return it higher up. They remove heat part-way up, and so set the internal reflux in each section.

```python
from difflow_refinery import ColumnThermo, column as cc

thermo = ColumnThermo.from_characterization(crude)
params = cc.CrudeColumnParams(
    n_stages=30, feed_stage=27, P_top=1.5e5, P_bottom=1.9e5, P_condenser=1.3e5,
    bottom_steam=..., steam_T=533.15,               # steam in mol/s
    side_products=(cc.SideProduct("kero", 9, 4, steam=...),
                   cc.SideProduct("diesel", 16, 4, steam=...),
                   cc.SideProduct("ago", 22, 3, steam=...)),
    pumparounds=(cc.Pumparound("pa1", 12, 10), cc.Pumparound("pa2", 19, 17)),
    specs=(cc.product_rate("naphtha", ...), cc.product_rate("kero", ...),
           cc.product_rate("diesel", ...), cc.product_rate("ago", ...),
           cc.pumparound_duty("pa1", 15e6), cc.pumparound_delta_t("pa1", 60.0),
           cc.pumparound_duty("pa2", 20e6), cc.pumparound_delta_t("pa2", 60.0)),
)
col = cc.CrudeColumn(params, thermo)
col.degrees_of_freedom()
res = col.solve(feed)                # res.products, res.T, res.condenser_duty, ...
```

### Degrees of freedom and specs

The degrees of freedom follow a simulator's:

| Element | Freedoms |
|---|---|
| Total condenser | 1 (the distillate split) |
| Partial condenser | 2 |
| Each side product | 1 (its draw rate) |
| Each pumparound | 2 (rate and return temperature) |
| `Furnace` | 1 (its coil outlet temperature) |

Close them with specs, exactly as many as there are freedoms (the constructor says how many are missing):

| Spec | Meaning |
|---|---|
| `product_rate(name, value, basis="volume")` | rate of the distillate, a side product, `"residue"` or `"offgas"` |
| `reflux_ratio(value)` | reflux over overhead product, molar |
| `stage_temperature(stage, T)` | a main-column stage temperature (stage 0 is the condenser) |
| `pumparound_rate`, `pumparound_duty`, `pumparound_return_temperature`, `pumparound_delta_t` | the two pumparound freedoms |
| `overflash(value, basis="volume")` | liquid leaving the stage above the flash zone, as a fraction of crude |
| `coil_outlet_temperature(T)`, `furnace_duty(Q)` | the furnace, directly |

Spec values are traceable: the gradient with respect to one is the column's sensitivity to that spec. Rates are in mol/s, kg/s or standard m³/s at 60 °F; `cc.BARREL` converts to barrels.

### Model

MESH holds on every equilibrium stage of the main column and of every stripper, all solved simultaneously (the Naphtali--Sandholm approach). The bubble-point method fails on a feed that boils across 600 K.

- **Unknowns per stage:**
  - the log component liquid flows;
  - the temperature;
  - the log total vapour flow, steam included.
- **Vapour:** the hydrocarbon vapour follows from Raoult's law. The water vapour is what is left, so the water balance is the stage's summation equation.
- **Water** is never in the hydrocarbon liquid. It condenses only in the overhead drum, as free water.
- **Duties** are evaluated from the converged state; they are not unknowns.

The solve is a damped Newton from a bubble-point initialisation. It reports `converged` rather than raising: a set of specs with no solution comes back `converged=False`. Too little overflash for the heat a pumparound removes is one such set.

**Involatile components.** A heavy-end assay brings components that are essentially involatile: a 950 °C residue lump has a vapour pressure of about 3e-5 Pa at 600 K. From the bubble-point start, such a component stalls the damped Newton. When any component's vapour pressure at 600 K is below 0.05 Pa, the column is first solved with those vapour pressures raised to that floor. It is then solved again with the true ones, starting from the first answer. The answer, and its gradients, are those of the true vapour pressures. The default characterization's heaviest cut is at 7.6e-2 Pa, so it never takes this path.

(refinery-furnace)=
### The furnace

With a `Furnace` in the params, the column's feed is the furnace *inlet*: the crude as the preheat train delivers it. The coil outlet temperature becomes one more unknown, solved with the column:

- The absorbed duty is the enthalpy rise of the crude from inlet to coil outlet, each evaluated by a flash. The fired duty is the absorbed duty over `efficiency`.
- The coil outlet pressure defaults to the feed stage's.
- The extra freedom is usually closed by an `overflash`, which is how an operator runs the heater. Solving the furnace in front of the column instead would need the outlet temperature guessed and iterated by hand.

---

(refinery-products)=
## Products

`product_properties(products, thermo, feed)` returns, for each hydrocarbon product:

- its rates: `mole`, `mass`, `volume` and `bpd`;
- `yield_volume` and `yield_mass` against the crude;
- `sg`, `api` and `mw`;
- TBP points at 5, 10, 50, 90 and 95 % (`tbp_at(pct)`).

`products.gaps(properties, order)` gives the 5--95 gap between neighbouring cuts: `TBP5(heavier) - TBP95(lighter)`. A positive gap is a clean separation; a negative one is an overlap.

The TBP curves are built from the pseudo-components a product contains, so they are no finer than the cuts. A gap of a few kelvin is within that resolution. ASTM D86 conversion is not done; the gap is defined on TBP.

---

(refinery-crude-unit)=
## The crude unit

`CrudeUnit` puts characterisation, thermo, furnace, column and product properties together:

```python
import difflow_refinery as dr

unit = dr.CrudeUnit(assay, params)     # a default Furnace is added if params has none
res = unit.solve(95_000, T=273.15 + 240, P=6e5)   # bbl/d at the furnace inlet
print(res.table())
res.gaps()
```

On the test crude, a 30-stage column with three side strippers, two pumparounds and a 5 % overflash gives:

- a coil outlet of about 313 °C and about 50 MW fired;
- naphtha, kerosene, diesel, AGO and residue at 20, 11, 17, 5 and 47 vol %;
- API gravities from 64.5 down to 18.7.

`solve` takes `assay=` (same light ends and cut points) and `params=` (same layout), so derivatives with respect to the assay and the specs are a `jax.grad` of a function that calls it:

```python
import jax

def residue_api(sg):
    a = dr.Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=sg)
    return unit.solve(1.0, T=480.0, P=4e5, basis="mole", assay=a).properties["residue"].api

jax.grad(residue_api)(0.85)
```

## Planning with the crude unit

`difflow_refinery.planning.cdu_block` wraps a `CrudeUnit` as a `difflow.planning.Block`. Its delta vectors are the column's own derivatives, taken by AD through the Newton solve, and the trust-region planner refreshes them every cycle (see [Delta-base planning](planning.md)). The full walk-through is `examples/35_refinery_cdu_planning.ipynb`.

```python
from difflow_refinery.planning import available_levers, cdu_block, link_cdu, product_value_block
from difflow.planning import DeltaBasePlanner, Network, check_delta_vectors
from difflow.planning.lp import Spec

cdu = cdu_block(unit, ["crude.rate", "naphtha.yield", "kero.yield", "overflash"],
                ["kero.bpd", "kero.tbp95", "gap.kero_diesel", "furnace.fired", ...],
                rate=95_000, T=273.15 + 240, P=6e5,          # the base point, as for unit.solve
                bounds={"kero.yield": (0.08, 0.15), ...})
check_delta_vectors(cdu)["passed"]                           # AD against central differences
value = product_value_block({"naphtha": 70.0, "kero": 95.0, ...})   # $/bbl
net = Network([cdu, value], link_cdu(cdu, value))
plan = DeltaBasePlanner(net, prices={"value.revenue": 1.0, "cdu.crude.rate": -65.0,
                                     "cdu.furnace.fired": -700.0},
                        specs=[Spec("cdu.kero.tbp95", "<=", 235.0)]).solve()
```

**Levers come from the specs.** `available_levers(unit)` lists them:

- `crude.rate` (bbl/d) and `preheat.T` (°C) are always levers.
- Each volume product-rate spec offers `<product>.yield` (a fraction of the crude) or `<product>.bpd`.
- The other spec levers are `overflash`, `<pa>.duty` (MW), `<pa>.dT` (K), `<pa>.return_T` (°C), `<pa>.rate` (bbl/d), `furnace.cot` (°C), `furnace.duty` (MW absorbed), `reflux_ratio` and `stage<k>.T` (°C).
- The stripping steam rates are levers in kg/h.

A spec the column does not have is not a lever. A column closed by an overflash has no `furnace.cot`, because fixing the coil outlet as well would over-specify it.

A spec that is not chosen as a lever is held. A volume product rate is held **as its yield**, so it follows `crude.rate`. Holding it as an absolute rate would give the whole rate change to the residue.

**Outputs** are reported per product and for the unit as a whole:

| Output | Units | Notes |
|---|---|---|
| `<product>.bpd` | bbl/d | |
| `<product>.yield` | - | volume fraction of the crude |
| `<product>.yield_mass` | - | mass fraction of the crude |
| `<product>.sg` | - | |
| `<product>.api` | API | |
| `<product>.mw` | g/mol | |
| `<product>.tbp5` … `.tbp95` | °C | |
| `gap.<a>_<b>` | K | TBP5(b) less TBP95(a) |
| `cut.<a>_<b>` | °C | the effective cut point: the crude's TBP at the cumulative yield through `a` |
| `furnace.fired`, `furnace.absorbed` | MW | |
| `furnace.cot` | °C | |
| `furnace.vaporized` | - | |
| `condenser.duty`, `<pa>.duty` | MW | |
| `steam.total` | kg/h | |
| `water.saturation_max` | - | |

Temperatures are in °C and temperature differences in K. Every unit is recorded in `block.metadata["u_units"]` and `["y_units"]`, so an export is self-describing.

**Health.** `check_delta_health` passes on the default outputs. Three kinds of output would fail it and are left out of the defaults; ask for them by name if you want them.

- **A held product's yield** is a constant row.
- **`steam.total`** is a constant row unless a steam rate is a lever.
- **The lightest product's TBP5** has a kink. A TBP point is piecewise linear in the cumulative volume, with one node per component, and the nodes among the discrete light ends are tens of degrees apart. On the test crude the naphtha's 5 % point sits on the n-butane node at the base point, with a slope of 277.6 K per unit yield to the left and 146.2 to the right.

The `preheat.T` lever is not dead, but on a column closed by an overflash it moves only the fired duty. The overflash fixes the flash-zone vaporisation, so the preheat temperature changes how much heat the furnace must add and nothing about the products.

**Non-convergence.** Some spec sets have no solution. With 5 % overflash, taking more than about 24 MW out of PA1 on the test column dries out the section above it. The column then reports `converged=False` with a finite state. `cdu_block` returns NaN at such a point, and the planner rejects any proposal at which a block is not finite ([A block that cannot be evaluated](planning.md#a-block-that-cannot-be-evaluated)).

On the test column, a credit on PA1 duty drives the planner past the edge. It proposes 30, 27, 25.5, 24.75 and 24.38 MW (among others), rejects each, and settles at 23.62 MW, which converges. With `mask_nonconverged=False`, the same run ends at 30 MW on a column that did not converge and reports itself as converged.

**Yields as levers, cut points as outputs.** A cut-point target such as "kero TBP95 ≤ 235 °C" is a planner `Spec` on a CDU output, and the LP inverts the delta vector to find the yield that meets it. In the example plan, the kero end point binds at 235.000 °C. The planner trades naphtha yield, which pays \$70/bbl, against kero, which pays \$95/bbl, because a heavier naphtha cut also makes the kero heavier.

Making a cut point a column *spec* would describe the same feasible set, and it would cost more in two ways:

- it would put a piecewise-linear TBP point inside the column's Newton solve;
- or, as an alternative, it would wrap a root find around the column, which means several column solves per evaluation.

For the same reason, `product_value_block` prices products in bbl/d only. Folded into the revenue as a smooth penalty, a quality limit puts curvature in the objective that a linear model cannot see. On the test crude, that version crawled along the penalty's shoulder at a radius of 1e-4 and stopped at the iteration cap.

---

(refinery-unit-operations)=
## Unit operations

### CrudeDistillationUnit

The crude unit as a flowsheet operation. It is built from one `CrudeDistillationUnitParams` (`assay`, `column`, `cut_points`, `method`), called with the crude stream at the furnace inlet, and returns the product streams in `outlet_names` order:

1. the distillate;
2. each side product, in the order the column declares them;
3. `"residue"`;
4. `"water"`;
5. with a partial condenser, `"offgas"`.

```python
from difflow import Flowsheet
from difflow.flowsheet import Unit

cdu = dr.CrudeDistillationUnit(dr.CrudeDistillationUnitParams(assay=assay, column=params))
feed = cdu.feed(95_000, T=513.15, P=6e5)          # bbl/d; also basis="mass"/"mole"/"volume"

fs = Flowsheet(list(cdu.unit.thermo.names) + ["water"])
fs.add_feed("crude", feed)
fs.add_unit(Unit("cdu", cdu, ["crude"], list(cdu.outlet_names)))
streams = fs.solve()
cdu.last_result.table()                           # the full result of the last call
```

- **Inlet:** the inlet needs a flow for every component the assay characterises into; `cdu.feed(...)` makes one.
- **Full result:** `cdu.solve(feed)` returns the full `CrudeUnitResult` (column profiles, duties, product properties) rather than the streams.
- **Serialisation:** the assay, the column params and every nested spec, side product, pumparound and furnace are plain dataclasses, so the unit writes to and reads back from JSON with `difflow.serialize`.

---

(refinery-vacuum)=
## The vacuum unit

The vacuum distillation unit (VDU) takes the atmospheric residue to light and heavy vacuum gas oil (LVGO, HVGO), slop and vacuum residue. Its column lives in `difflow_refinery.vacuum`. Its components come from the **same characterization as the crude unit's** ([The heavy end and contaminants](#refinery-heavy-end)):

- **Its property table** is `Characterization.pseudo_components()`. Every pseudo-component of the crude unit is one of the vacuum column's, with the same Tb, SG, MW, critical constants and contaminants. The crude unit's residue therefore feeds the column as it is, with no re-cut.
- **Correlations** come from `difflow_refinery.correlations`: Twu critical properties and molecular weight, the Kesler-Lee acentric factor and liquid Cp, and Maxwell-Bonnell vapour pressure (the D1160 vacuum conversion). `difflow_refinery.vacuum.correlations` keeps the vacuum code's old names for them.
- **A stage-network column** (`StageColumn`, `ColumnLayout`, `Route`, `StageSpec`): Naphtali-Sandholm MESH equations with liquids routed to side draws, pumparounds and entrainment, Murphree efficiencies, and any column output specifiable in place of any knob.
- **`VacuumColumn`** is the registered operation. Feed species it does not model, such as the light ends and the crude unit's water dissolved in the residue, leave with its overhead, so a flowsheet still balances.
- **`vacuum.Assay` and `vacuum.characterize`** are kept as a compatibility view. The old `Assay` (Celsius, wt%) converts with `to_assay()`. `characterize` runs the shared characterization on a vacuum cut grid and returns the old `(components, yields, light_ends, Kw)` shape. `vacuum.atmospheric_residue` is an idealized TBP cut for running the column without a crude unit in front of it.

(refinery-crude-to-vacuum)=
### Crude unit into vacuum column

```python
from difflow import Flowsheet
from difflow.flowsheet import Unit
from difflow_refinery.vacuum import VacuumColumn, VacuumColumnParams

char = dr.characterize(assay)                         # one HeavyEnd assay, as above
cdu = dr.CrudeDistillationUnit(dr.CrudeDistillationUnitParams(assay=assay, column=params))
vdu = VacuumColumn(VacuumColumnParams(components=char.pseudo_components()))

fs = Flowsheet(list(char.names) + ["water", "H2O"])  # CDU water is F_water, VDU steam F_H2O
fs.add_feed("crude", cdu.feed(95_000, T=513.15, P=6e5))
fs.add_unit(Unit("cdu", cdu, ["crude"], list(cdu.outlet_names)))
fs.add_unit(Unit("vdu", vdu, ["residue"],             # renamed: "residue" is the CDU's
                 ["vac_overhead", "lvgo", "hvgo", "slop", "vac_residue", "vdu_info"]))
streams = fs.solve()
```

On the 95 000 bbl/d test crude (`tests/refinery/test_one_characterization.py`, `examples/36_crude_to_vacuum.ipynb`):

- Both units converge.
- The balance closes per component to round-off. It also closes in total, once the crude unit's and the vacuum unit's steam are counted, each at its own water molar mass (18.015 and 18.01528 g/mol).
- At the default 400 °C furnace, the vacuum column turns 0.74 of its feed into VGO.
- d(VGO yield)/d(VDU furnace T) is 2.8e-3 per K. d(VGO rate)/d(crude rate) runs through both units' implicit solves. Both match central differences.

(refinery-vacuum-quick-start)=
### Quick start: the vacuum unit

```python
import difflow_refinery as dr

char = dr.vacuum.characterize(dr.vacuum.heavy_crude())           # 20 cuts 300-800 C + lump
feed = dr.vacuum.atmospheric_residue(char, crude_rate_kg_s=100.0)

vdu = dr.VacuumColumn(dr.VacuumColumnParams(components=char.components))
overhead, lvgo, hvgo, slop, residue, info = vdu(feed)

info["converged"], info["iterations"]               # True, 6
info["properties"]["hvgo"]["T95"] - 273.15          # 566 C
info["properties"]["hvgo"]["ccr_wt"]                # 1.84
info["outputs"]["furnace.duty"]                     # 1.31e7 W
```

Streams use difflow's convention: `F_<pseudocomponent>` in mol/s, with `T`
in K and `P` in Pa. The overhead also carries the stripping steam as
`F_H2O`.

---

(refinery-vacuum-characterization)=
### Vacuum characterization

A `vacuum.Assay` holds what a lab reports: a TBP curve (cumulative wt% distilled
against temperature) and bulk properties. Two synthetic assays of the right
shape are included, `light_crude()` (38 API, 15% vacuum residue) and
`heavy_crude()` (20 API, 40%). They are not the assays of named crudes.

`vacuum.characterize(assay, cuts_C=...)` cuts the curve on a temperature grid. The
default is 300-800 C in 25 C steps. Each cut gets:

| Property | How |
|---|---|
| mass yield | difference of the TBP curve across the cut |
| `Tb` | the mean of the TBP curve over the cut |
| `SG` | from a constant Watson K, fitted over all the cuts so the whole crude matches its bulk SG |
| `MW`, `Tc`, `Pc` | Twu (1984) |
| `omega` | Kesler-Lee (1976) |
| S, N, CCR, Ni+V, asphaltenes | a logistic in boiling point, scaled to the bulk assay value (or a measured curve) |

**The heavy end.** A TBP curve stops at about 565 C, but a VDU needs
pseudocomponents to 750-800 C. The curve is drawn on a probability scale,
`z = Phi^-1(x)` against T, which is close to linear for crude oils. Inside
the data it is a monotone cubic Hermite in `z`, and beyond the data it is a
least-squares line through the last three points. The curve is C1
everywhere. That matters because the default grid puts a cut boundary *on*
every assay point (both fall on multiples of 50 C). A piecewise-linear
curve has a kink exactly there, so a derivative with respect to that TBP
point has two values, and central differences average them. That error
measured 1e-3 relative before the curve was made C1.

**The residue lump** is everything past the last cut. Twu's n-alkane
reference has no root past about 840 C, and Riazi-Daubert is fitted only to
about 580 C, so the lump's `Tb`, `SG` and `MW` are set directly
(`residue_Tb_C`, `residue_sg`, `residue_mw`). Its vapor pressure at VDU
conditions is then negligible, which is the property that matters.

---

(refinery-vacuum-thermodynamics)=
### Thermodynamics at vacuum

`ColumnThermo` makes three simplifications, each of which holds at vacuum:

- **K-values are `psat / P`.** At 1-10 kPa the vapor is ideal to far better
  than the correlations are accurate. Vapor pressure is Maxwell-Bonnell, the
  correlation ASTM D1160 uses to convert a boiling point measured under
  vacuum to its atmospheric equivalent. It is anchored at the boiling
  point the assay measures and fitted down to a fraction of a mmHg. A
  500 C cut boils at 331 C at 10 mmHg, as in the D1160 tables. Its three
  pressure branches are joined by a narrow logistic blend, so Newton sees a
  continuous derivative. Lee-Kesler with Twu criticals agrees near the
  boiling point and drifts at the reduced temperatures of a flash zone.
- **Steam does not condense.** At the coldest point in the column (60-80 C)
  water's vapor pressure is ten times the column pressure. Steam lowers the
  hydrocarbon partial pressure and carries enthalpy, and that is all it
  does.
- **The heat of vaporization is `R T^2 dln(psat)/dT`.** It is the
  Clausius-Clapeyron slope of the curve that sets the K-values, so the
  energy balance and the phase equilibrium cannot disagree about how
  volatile a cut is. Liquid enthalpy comes from Kesler-Lee's Cp.

`fit_antoine` fits Antoine constants to Maxwell-Bonnell over a range, for
handing a pseudocomponent to a simulator that takes nothing else.

---

(refinery-stage-network)=
### The stage-network column

`StageColumn` is the machinery any refinery column is built from. Stages
are numbered from the top. A furnace outlet flash sits in front of the feed
stage. The liquid leaving each stage is split between its **routes**:

| Route kind | Takes | Variable |
|---|---|---|
| `fixed` | a fraction set by the knobs (entrainment) | none |
| `reference` | what the share routes leave | none |
| `share` | a softmax share of the rest | one logit, with an equation `<route>.rate == knob` (kg/s) |

A route with `duty=` is a pumparound: its liquid returns to its destination
stage with that much heat removed. Draws are softmax shares rather than
rates, so a draw can never take more liquid than its stage has. That is
what makes a rate spec safe to give Newton.

**Specs.** Every share route has a default rate equation, and every other
knob (a duty, the furnace temperature, a pressure) is a fixed number. A
`StageSpec(output, target, replaces=...)` trades one of those for
`output == target`. The degrees of freedom always balance, and any output
can be specified by giving up the knob that controls it: a stage
temperature, a product's TBP point, the overflash, a pumparound return
temperature.

**Variables and residuals.** The variables are, for every stage, the log
of each component's liquid and vapor flow, plus the stage temperature. The
component balances are written in log form too, as
`ln(out) - logsumexp(ln ins)`. A residue lump in the top stage is 1e-200 of
its feed or less. In linear form its flows underflow, its balance rows go
to zero and the Jacobian is singular (condition number 1.7e17 was
measured). In log form every row is O(1) and exact. Equilibrium is Murphree
on the vapor, `y = eta K x + (1 - eta) y_in`, also in log form.

**Solve.** The solver runs damped Newton with a dense `jax.jacfwd`
Jacobian; a VDU has about 430 variables. Steps are capped at 30 K on
temperatures and 5 on log flows, and an Armijo backtracking line search runs
inside `lax.while_loop`. *Trace* flows, below 1e-7 of their component's
feed, are exempt from the cap and clipped individually instead. Otherwise
one residue-lump vapor that wants to fall by 600 log units scales every
step to nothing. Stage efficiencies are reached by a three-pass homotopy:
equilibrium stages first, then halfway, then the target.

**Gradients.** The solve finishes with one Newton step from the converged
point with the Jacobian frozen: `x* - J^-1 R(x*, theta)`. Its value is `x*`
and its derivative is `-J^-1 dR/dtheta`, the implicit-function derivative.
`jax.grad` and `jax.jacfwd` never differentiate the iteration. The Jacobian
the step uses is the last one the Newton loop evaluated, so it is traced
only once.

---

(refinery-vacuumcolumn)=
### VacuumColumn

```text
overhead vapor (steam + light HC) -> ejectors
+---------------------------+
| LVGO pumparound section   |  n_lvgo stages; LVGO PA returns to the top
|   LVGO draw (total draw)  |  -> LVGO product + LVGO PA
| HVGO pumparound section   |  n_hvgo stages; HVGO PA returns to its top
|   HVGO draw               |  -> HVGO product + HVGO PA + wash oil down
| wash zone                 |  n_wash stages
|   slop draw (total draw)  |  -> slop / overflash
| flash zone  <- furnace    |  1 stage; entrainment carried up
| stripping section         |  n_strip stages
+---------------------------+  <- bottom steam
vacuum residue
```

There is no condenser and no reflux drum. The top stage's vapor leaves as
overhead at the top-stage temperature, and the LVGO pumparound holds that
temperature. Each packed bed is `n` equilibrium stages with a Murphree
efficiency to fit. The defaults are two stages per bed, two stripping
stages and efficiency 1, which gives nine stages.

**Default specs** (`default_vacuum_specs()`):

| Spec | Replaces |
|---|---|
| `top.T = 70 C` | `lvgo_pa.duty` |
| `overflash = 0.03` (slop / feed, mass) | `hvgo.rate` |
| `lvgo.T95 = 450 C` | `hvgo_pa.duty` |

The pumparound circulations are fixed at 0.3 and 0.8 x feed unless given.
With the defaults, the operating levers are the furnace outlet temperature,
the flash-zone pressure and the stripping steam, and the yields and
qualities come out. That is the trade-off a planner wants to see. To ask
the reverse question, for example what furnace temperature a given HVGO end
point costs:

```python
specs = dr.default_vacuum_specs() + (
    dr.StageSpec("hvgo.T95", 570.0 + 273.15, replaces="furnace.T"),)
vdu = dr.VacuumColumn(params.update(specs=specs))
```

**Outputs** (`info["outputs"]`, SI units) include every stage `T` and `P`,
every route's `.rate`, each product's `.T05/.T10/.T50/.T90/.T95`, `.yield`
and draw temperature, pumparound `.duty` and `.return_T`, `furnace.duty`,
`furnace.vapor_fraction`, `furnace.cracking_margin`, `flash_zone.T`,
`overflash`, `vgo.yield` and `steam.rate`. `info["properties"]` gives SG,
API, S, N, CCR, Ni+V, asphaltenes and the TBP points of each product.

**Contaminants.** S, N, CCR, Ni+V and asphaltenes are carried per
pseudocomponent and follow the flows. CCR and metals reach HVGO by two
paths. One is vaporization of the heaviest cuts. The other is
entrainment: a fraction `entrainment` of the flash-zone liquid is carried
up with the vapor, and the wash bed removes `deentrainment` of it into the
slop. The rest reaches the HVGO draw. Both are parameters to fit.

**Furnace outlet.** `furnace_T_max` (415 C) is the cracking limit. It is
reported as `furnace.cracking_margin` and warned on (`CrackingWarning`),
not imposed, so a planner can see the trade-off rather than have it
hidden.

#### Results on the two included assays

Default specs, 100 kg/s of crude, furnace 400 C, flash zone 30 mmHg:

| | light: LVGO | HVGO | slop | residue | heavy: LVGO | HVGO | slop | residue |
|---|---|---|---|---|---|---|---|---|
| yield on feed | 0.155 | 0.487 | 0.030 | 0.328 | 0.071 | 0.296 | 0.030 | 0.604 |
| TBP T50 (C) | 393 | 484 | 605 | 674 | 391 | 473 | 595 | 753 |
| TBP T95 (C) | 450 | 579 | 699 | 868 | 450 | 566 | 834 | 888 |
| SG | 0.873 | 0.910 | 0.955 | 0.988 | 0.907 | 0.942 | 0.996 | 1.052 |
| S (wt%) | 0.76 | 1.13 | 1.42 | 1.49 | 2.76 | 4.01 | 5.18 | 5.55 |
| CCR (wt%) | 0.19 | 1.48 | 8.0 | 15.2 | 0.29 | 1.84 | 12.9 | 28.4 |
| Ni+V (wppm) | 0.00 | 0.14 | 8.1 | 57 | 0.01 | 1.05 | 100 | 495 |

These numbers moved slightly when the vacuum unit moved onto the shared characterization (#301). A cut's `Tb` is now the mean of the curve over it, not its mid-percent point, and the Watson K is fitted over the cuts themselves. Gravities fell by up to 0.004 and sulfur rose by up to 3 %. Yields and TBP points are unchanged to the figures shown.

Both converge from the default initialization in 6-7 Newton iterations.
The per-pseudocomponent mass balance closes to 1e-15. Implicit gradients
match central differences to better than 1e-5 relative, for product rates
and HVGO properties with respect to furnace T, flash-zone P, steam, and the
500 C point of the assay's TBP curve (`tests/refinery/test_vacuum.py`).

---

(refinery-vacuum-gotchas)=
### Vacuum unit gotchas

- **An end point the beds cannot make is infeasible, not hard.** Each
  Murphree stage passes `(1 - eta)` of the vapor through unchanged. An HVGO
  bed of two stages at 70% therefore sends 9% of the heavy vapor into the
  LVGO section. The default `lvgo.T95 = 450 C` then cannot be met at any
  HVGO pumparound duty, and Newton drives the duty to 1e13 W before it
  gives up. Loosen the spec (520 C converges) or raise the efficiency.
- **Specs must be feasible for the feed.** Overflash, draw rates and end
  points compete for the same vaporized material. A furnace that vaporizes
  25% of the feed cannot give 30% HVGO. When the solve does not converge,
  check that the specs add up before suspecting the solver.
- **The LVGO and slop draws are total draws.** No liquid passes from the
  LVGO section to the HVGO section, or from the wash bed to the flash zone,
  as on chimney trays. The LVGO yield is therefore whatever the HVGO section
  does not condense. It is set through the HVGO pumparound, by the
  `lvgo.T95` spec.
- **Compile once, solve many.** The first solve for a spec *structure*
  (which outputs replace which knobs) compiles in about 20 s. Every later
  solve with the same structure costs about 0.2 s, whatever the numbers,
  because spec targets, knobs, efficiencies and the whole assay are
  runtime inputs.

---

(refinery-vacuum-out-of-scope)=
### Vacuum unit: out of scope

Ejectors and vacuum-system modelling, dynamics, lube vacuum towers, and
D1160 (as opposed to TBP) product curves. A cross-check against an
independent simulator (DWSIM or IDAES on the same characterized feed) has
not been done; neither is available in the test environment.

---

(refinery-blending)=
## Product blending

`difflow_refinery` holds refinery models. Its first one is the **product blending pool**. It mixes component streams into finished products (gasoline, jet, ULSD, fuel oil), computes the properties that specs are written on, and reports a signed margin per spec. It uses the nonlinear blending rules refiners use, and it is differentiable in the recipe and in every component property.

```python
from difflow_refinery import BlendComponent, BlendPool

reformate = BlendComponent.from_properties(
    "reformate", SG=0.80, RON=98.0, MON=88.0, RVP_psi=3.5, S_ppm=1.0,
    olefins_vol=1.0, aromatics_vol=65.0)
# ... fcc, alkylate, butane likewise

pool = BlendPool("gasoline")                 # default specs: RON, MON, RVP, S
res = pool([reformate, fcc, alkylate, butane], recipe=[0.35, 0.35, 0.25, 0.05])
res.properties["MON"], res.margins["MON >= 82"]

pool.linear_blend_error(components, recipe)  # nonlinear minus linear-by-volume
pool.backoff(components, recipe, exact=("RVP_psi", "S_ppm"))
pool.spec_violations(components, recipe, temperature=0.05)   # smooth max(-m, 0)
pool.as_block(components)                    # a difflow.planning.Block
```

Like `difflow.planning`, this is a library rather than a set of GUI palette operations. It registers no `difflow.plugins` entry point. A pool is called with a recipe and returns properties and margins, which is not the stream-in/stream-out shape a palette unit has.

Example: `examples/33_refinery_gasoline_blending.ipynb`. Tests: `tests/refinery/test_blending.py`.

### Components: property mode and stream mode

A `BlendComponent` is built one of two ways:

- **`from_properties(name, SG=..., RON=..., ...)`**: measured or unit-reported properties only. This is enough for every *blended* property.
- **`from_stream(name, stream, characterization, **overrides)`**: a difflow stream of pseudocomponent molar flows on a shared `BlendCharacterization`. The composition gives `SG`, sulfur, nitrogen, PNA and a Raoult RVP. Overrides (a reformer's reported RON, say) take precedence. This mode adds:
  - the product **stream**, with an exact mass and volume balance;
  - the properties only composition can give: distillation and cetane index;
  - the Raoult RVP of the blend itself.

One pool takes one mode. Mixing the two raises an error rather than producing a product stream that is missing some components' mass.

**Volume basis.** Volumes are ideal-mixing volumes at 15 °C from `SG` (`rho = SG * 999.10 kg/m^3`). The product volume is the sum of the component volumes, and the product `SG` is the volume average. Both are tested to round-off against the product stream's own composition.

Recipes can be given as `basis="volume_fraction"` (the default), `"volume_flow"`, or `"split"`. A split is the fraction of each component stream's available volume sent to the pool, which is the natural lever when the pool sits in a flowsheet.

### Blending rules

| Property | Key | Rule (default first) | Source |
|---|---|---|---|
| RON, MON | `RON`, `MON` | Ethyl RT-70 interaction model; or linear by volume | Healy, Maassen & Peterson (1959), Ethyl report RT-70; coefficients as tabulated in Maples (2000) |
| RVP | `RVP_psi` | RVP^1.25 index by volume; Raoult on the pseudocomponents; or linear | Gary, Handwerk & Kaiser, product blending chapter; tested against a published worked example |
| Sulfur, nitrogen, CCR | `S_ppm`, `N_ppm`, `CCR_wt` | by mass | - |
| Aromatics, olefins, benzene, PNA, smoke point | `*_vol`, `smoke_mm` | by volume | - |
| Flash point | `flash_C` | Hu-Burns index, `log10 BI = -6.1188 + 2414/(T - 42.6)` (K), by volume | Hu & Burns (1970); identical to Wickey-Chittenden in °F |
| Cloud, pour point | `cloud_C`, `pour_C` | `BI = T^n` (K), n = 1/0.05 and 1/0.08 | Hu & Burns (1970) |
| Freeze point | `freeze_C` | `BI = T^n`, n = 20 | omsQlibs *Blending Quality Models Equations* (2016), generic freeze index; its Ethyl index defaults to n = 12.5 |
| CFPP | `CFPP_C` | `BI = T^n`, n = 12.5 (the pour-point exponent) | no published index found; an assumption, override with `rules={"CFPP_C_exponent": ...}` |
| Viscosity | `viscosity_cSt` | Refutas VBN `14.534 ln ln(nu + 0.8) + 10.975`, by mass | Refutas, as given in Maples (2000); tested against a published worked example |
| Distillation | `E70_tbp`, `E100_tbp`, `T{10,50,90,95}_d86_C` | from the blend's composition: smoothed TBP, then Riazi-Daubert TBP->D86 | Riazi & Daubert (1986) |
| Cetane index | `cetane_index` | ASTM D4737 (default) or D976 on the blend's density and D86 points; computed, never blended | ASTM D4737, D976 |

Choose rules per property with `BlendPool(rules={"octane": "volume", "RVP_psi": "raoult", "cetane": "d976"})`.

The RT-70 corrections are all *spreads*: covariances and variances across the components of sensitivity, olefins and aromatics. The model therefore reduces exactly to the linear blend when the components agree. The MON equation's first term is the MON×sensitivity covariance, the same form as RON's. The coefficients are the 75-blend fit (`a1 = 0.03224`, `a2 = 0.00101`, `a3 = 0`, `b1 = 0.04450`, `b2 = 0.00081`, `b3 = -0.00645`, the last on the squared aromatic spread / 1e4), as tabulated in Maples, *Petroleum Refinery Process Economics*, 2nd ed. (2000). The 1959 original was not reachable. They live in `EthylRT70` and can be refitted.

Until #301 the code had `b3 = -0.0645` and an olefin×MON first term. The factor of ten is a transcription error: Maples gives -0.00645, and the 135-blend fit written out in full (`-6.32e-7 (A^2 - A A)^2` against a tabulated -0.00632) fixes both the digit and the /1e4 scaling. Both errors made MON blend much worse than it does. On a reformate/FCC pool the MON correction is now a few tenths of a number. Check it against your own blend data.

The distillation points are on a TBP curve made differentiable by spreading each pseudocomponent with a logistic of `distillation_width` (default 5 K). Each point is then converted to D86 with the Riazi-Daubert correlation, because specs are written on D86. `E70`/`E100` stay on the TBP basis, and the key says so.

### Specs and margins

`BlendSpec(property, op, limit, scale=1.0)`. The margin is `value - limit` for `>=` and `limit - value` for `<=`, divided by `scale`, so **positive means on spec**. `PRODUCT_SPECS` holds illustrative defaults for each product: US regular summer gasoline, Jet A, ULSD S15, and VLSFO RMG 380. Pass your own specs for anything real.

- `spec_margins(components, recipe)` returns the margin vector, smooth in everything. With `weighted=True` it returns `V * margin` instead. A property is intensive, so it is 0/0 at an empty pool. The weighted form is the same constraint wherever `V > 0`, stays finite at zero, and is the form an LP's blending rows take. Use it as the constraint for an optimizer over volume flows.
- `spec_violations(..., temperature=t)` returns `max(-margin, 0)`, or its softplus `t * logaddexp(-margin/t, 0)`. The smooth form is within `t ln 2` of the kink and its derivative at an active spec is exactly `-1/2`. The branchless `max(u,0) + log1p(exp(-|u|))` has the same value and a derivative of zero there; see the CVaR note in `difflow.stochastic`.

### Linear blend error and back-off

`linear_properties` is the planning-LP view: every property linear by volume. `linear_blend_error` is nonlinear minus that view, and it is what a planner's back-off is meant to cover. Pass `exact=("RVP_psi", "S_ppm")` for properties the LP already models with the pool's own rule (the RVP index, sulfur by mass). Their error is then zero. `backoff` turns the error into a per-spec tightening, `max(linear margin - nonlinear margin, 0)`.

The example measures what this means for a gasoline LP:

- the example sets MON at 86, so that the MON row binds; at a regular grade's 82, MON is slack and the LP is exact;
- without back-off, the LP promises about 5% more margin than any feasible plan, and its recipe is about 0.6 MON numbers off spec;
- the back-off depends on the recipe, so one pass is not enough; successive back-off takes about six passes to settle, at about 0.98 MON against a first estimate of 0.58;
- on that pool every NLP start finds the converged back-off plan. That is not guaranteed in general: blending is nonconvex, and the back-off fixed point is a feasible vertex, not a certified optimum.

### Planning hook

`pool.as_block(components)` returns a `difflow.planning.Block`. Its levers are the component volume flows (`<name>_V`), bounded by availability in stream mode. Its outputs are the product `volume`, the spec properties, and `margin:<spec>` for every spec. `jax.jacobian` of the block is the delta-vector set: the volume row is the volume balance, all ones, and the property rows are the blend's sensitivities at the linearisation point.

### Blend characterization

`BlendCharacterization(names, Tb, SG, MW=, Tc=, Pc=, omega=, qualities=)` is the pseudocomponent grid. Molecular weight and critical constants come from Riazi-Daubert (1980), the acentric factor from Edmister, and vapor pressure from Lee-Kesler. Any of them can be overridden per pseudocomponent by passing a vector with NaN where the correlation should be used. That is how a defined component such as n-butane takes its own constants. `qualities` holds the per-pseudocomponent composition vectors (`S_ppm`, `aromatics_vol`, ...), each averaged on its own basis.

This is the minimum a blend pool needs to compute properties from composition. It is not an assay model: there is no TBP fitting and no heavy-end extrapolation.

For products of the crude and vacuum units, use `BlendCharacterization.from_characterization(char)` instead. It takes the crude characterization's Tb, SG, MW, its critical constants and its vapour-pressure acentric factor, so the pool's Raoult RVP sees `psat(Tb) = 1 atm` exactly as the columns do. It also takes the sulfur vector (and, with `contaminants=True`, nitrogen and CCR) as the `S_ppm`, `N_ppm` and `CCR_wt` qualities. Qualities the assay did not give are left out. A product stream from either column is then a `BlendComponent.from_stream` input; `F_water` and `F_H2O` are ignored. The pool's sulfur for LVGO or HVGO equals the vacuum column's own report to 1e-10, since both average the same per-component vector.

### Not in scope

- Tank inventory and multi-period scheduling. The pool is steady state, per period.
- Crude blending ahead of the CDU (assay mixing).
- A straight-run octane correlation from PNA. Octane is unit-reported or measured.

(refinery-validation)=
## Validation

The crude unit has been checked against an independent reference. The issue asked for a DWSIM (or HYSYS or PRO/II) crude case. None of those were available, so the reference is built with **IDAES** 2.10 (Pyomo 6.10, IPOPT 3.13.2): the U.S. DOE's open equation-oriented process modelling platform. That choice decides what the check can say. It is an independent *implementation*: different code, a different solver, a different formulation and a different starting point. It is not an independent *model*: the column and its property model are the ones difflow states, written again. A commercial simulator's crude case would also bring its own characterisation and its own thermodynamics. Here those are tested separately, against published numbers and against Peng-Robinson.

The generator is `tests/refinery/reference/generate.py`. It writes `cdu_reference.json`, which records the tool versions, the property methods, the date and the difflow commit. `tests/refinery/test_validation.py` checks difflow against that file, with each tolerance and the reason for it written beside the assertion. The test needs neither IDAES nor IPOPT. The case is the 95 000 bbl/d test crude in the 30-stage column used throughout this page: three side strippers, two pumparounds, bottom and stripper steam, a total condenser, and a 5 vol % overflash. The column builder (`reference/mesh.py`) takes a stage network, not a fixed layout, so the vacuum column (#294) can reuse it.

**Layer 1: characterisation, against published worked examples.**

| Source | Check | Agreement |
| --- | --- | --- |
| Riazi, ASTM MNL50 (2005), Ex. 2.5 and Ex. 2.7 / Table 2.11 | Riazi-Daubert 1980 and 1987, Lee-Kesler and Twu M, Tc, Pc | to the printed digits; Twu Pc within 1 % (below) |
| Ahmed, *Equations of State and PVT Analysis* (2016), Ex. 2.2 | Riazi-Daubert 1980/1987 and Kesler-Lee M, Tc, Pc | to the printed 3 figures |
| MNL50 Ex. 3.3 / Table 3.8 | TBP to ASTM D86 (Riazi-Daubert) | within 0.1 C at all six points |
| `chemicals` 1.5.2 docstring examples (API TDB 2B4.1; Lee-Kesler; Riedel) | Watson K, Lee-Kesler Psat, Riedel latent heat at Tb | to the printed digits |
| `chemicals` 1.5.2, on this crude's 28 components | Lee-Kesler Psat, Lee-Kesler omega, Riedel latent heat | 1e-10 |

Three of MNL50's printed values are not reproduced. None is asserted.

- Twu Pc for n-C36. The printed value is 6.02 bar (6.03 in Table 2.12); ours is 5.97. Our Vc reproduces the printed 2010 cm3/mol and Pc° = 6.015 bar, so the printed Pc appears to omit Twu's f_P correction. The test allows 1.5 %.
- The "API" Pc of 7.37 bar. The extended Riazi-Daubert 1987 equation gives 5.90.
- The Lee-Kesler Tc of 935.1 K. The Kesler-Lee equation gives 870.7, and our implementation reproduces Ahmed's Kesler-Lee example to three figures.

**Layer 2: thermodynamics, against IDAES.**

- *IDAES running difflow's property model.* The IDAES generic framework was given the same pure-component methods. The test flashes each stage's whole contents at the reference column's T and P:
  - K-values agree to 1e-6.
  - Liquid and vapour enthalpies agree to 1e-6.
  - Each stage splits back into its own L and V to 1e-5.
  - IDAES's crude bubble point (379 K) and dew point (845 K) at the flash-zone pressure satisfy difflow's sum z K = 1 and sum z / K = 1.

  This confirms that difflow's arrays implement the equations it states. It says nothing about whether those equations are right.
- *Peng-Robinson on the same Tc, Pc, omega and ideal-gas Cp* (kij = 0, hydrocarbons only). These are modelling differences. They are documented and pinned in the tests, not tuned away:
  - On every stage, and at the coil outlet, the vapour fraction agrees within 0.021. At the coil outlet the difference is 0.0033.
  - At the furnace inlet (240 C, 6 bar), Raoult vaporises 23 mol % of the crude and PR vaporises 17 %.
  - The crude's enthalpy rise from the furnace inlet to the coil outlet is **4.0 % higher under PR**. That is the furnace-duty difference a PR crude case would show from the property model alone.
  - For the cuts boiling 420-640 K, which make the side products, PR and Raoult K-values agree within -30 % / +25 % at the flash zone.
  - Raoult over Lee-Kesler badly overpredicts the supercritical light ends. Propane's K is about 60 times PR's. The light ends go overhead under either model, but do not read a light-ends K-value off this model.
  - For the heaviest residue cut (Tb 1033 K), PR's K is about 20 times Raoult's.
  - Liquid enthalpy at the same composition agrees within 1.5 kJ/mol above the flash zone. Where residue is in the liquid, PR's is 13-18 kJ/mol higher. Watson's latent heat and PR with an extrapolated omega are both extrapolations for a 1000 K cut, and nothing here says which is closer to the truth.
- *Water.* difflow's Wagner-Pruss Psat agrees with IAPWS-95 to 0.004 % and with IAPWS-IF97 to 0.02 %. Watson's latent heat for water is exact at Tb, 1.3 % high at 300 K and 2.6 % low at 550 K, compared with IAPWS-95 by Clausius-Clapeyron.

**Layer 3: the column, against an independent EO model.** `reference/mesh.py` writes every stage, stripper, pumparound, the condenser and the furnace flash as Pyomo equations with the same specs. It solves them with IPOPT from a linear 380-580 K profile and round-number flows; no difflow result is used to initialise it. It converges in about 17 s. The two solutions agree as follows:

- stage and stripper temperatures and the coil outlet (586.3 K): 5e-6 K
- condenser temperature: 3e-4 K, all of it the difference between the two water Psat formulations (below) in the bubble point of the condensate with free water
- condenser duty (31.39 MW): 6e-7 relative
- fired duty (49.89 MW): 3e-10 relative
- pumparound return temperatures: agree within 1e-3 K (the test tolerance)
- feed vaporised (0.696): agrees within 1e-6 (the test tolerance)
- volume yields: 2e-10
- API gravities: 1e-7
- TBP 5/10/50/90/95 % points: 4e-7 K
- 5-95 gaps: 5e-7 K
- steam saturation: within the 0.02 % that separates the two water Psat formulations

The test tolerances are set at the solvers' precision, not at engineering accuracy, so a transcription error in either column has nowhere to hide. The same column with Watson's floor unsmoothed (eps = 0.01 → 1e-6) moves the fired duty by 0.10 %, the condenser duty by 0.003 %, stage temperatures by at most 0.017 K and the API gravities by at most 2e-4. That is the price of the smoothing that gives difflow a derivative everywhere.

**Layer 4: gradients.** Central finite differences of the reference column (each spec ± 0.002, IPOPT warm-started) against `jax.grad` through difflow's implicit-function solve:

| Gradient | `jax.grad` | Reference FD | Agreement |
| --- | --- | --- | --- |
| d(diesel API)/d(diesel vol. yield) | -30.2248 | -30.2254 | 1.8e-5 |
| d(fired duty)/d(overflash) | 167.758 MW | 167.764 MW | 3.2e-5 |

The file also stores d(condenser duty)/d(diesel yield) and d(residue API)/d(overflash) for later use.

**What this does not validate.** It does not show that Raoult/Watson is the right property model for a given crude; layer 2 measures how far it is from PR, nothing more. It does not cover a commercial simulator's characterisation, its D86 interconversion defaults, or its tray-efficiency and hydraulics models, and it does not replace plant data. The 5-95 gaps of this case are negative (-28 to -54 K): equilibrium stages with these specs give overlapping products. Both implementations agree on that, which says nothing about whether a real column would overlap the same way. When a deliberate model change moves any number above, `TestReferenceIsCurrent` fails first and asks for the reference to be regenerated.

---

(refinery-limitations)=
## Limitations

- **The crude unit is the atmospheric column only.** The preflash drum and preheat train are not modelled; the inlet is the preheat train's outlet. The vacuum unit is a separate operation, fed from the crude unit's residue in a `Flowsheet` (above).
- **Thermodynamics:** Raoult's law and ideal-gas-path enthalpies. This is the usual model for an atmospheric column at one or two bar; it is not a cubic equation of state.
- **Equilibrium stages.** There are no tray efficiencies or hydraulics.
- **Boiling ranges are TBP, not ASTM D86.**
- **Validation:** against an independent equation-oriented model, IDAES property packages and published characterisation examples; not against a commercial simulator's crude case. See [Validation](#refinery-validation) for what that does and does not establish.
