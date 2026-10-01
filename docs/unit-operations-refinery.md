# Refinery Unit Operations

This document covers the `difflow_refinery` plugin. It characterises a crude from its assay, separates it in a crude distillation unit (CDU): a fired heater and an atmospheric column, with side strippers, pumparounds and stripping steam. The products are reported the way a refinery reads them: yields, API gravities and TBP ranges. The atmospheric residue goes on to a vacuum distillation unit (VDU), and finished components are blended into products in a nonlinear blending pool.

---

(refinery-overview)=
## Overview

The `difflow_refinery` plugin provides:

- **Assay characterisation** (`Assay`, `characterize`): a TBP curve plus a gravity, cut into pseudo-components with the standard petroleum correlations. Light ends (C1--C6) are kept as real species.
- **Column thermodynamics** (`ColumnThermo`): vectorised over stages and components.
  - Raoult's law with Lee-Kesler vapour pressures.
  - Ideal-gas-path enthalpies.
  - Stripping steam as a vapour that condenses only as free water in the overhead drum.
- **The atmospheric column** (`CrudeColumn`): an equation-oriented MESH model, with side strippers, pumparounds, bottom and stripper steam, and a total or partial condenser.
- **The furnace** (`Furnace`): solved *with* the column, so an overflash spec sets the coil outlet temperature, as it does in operation.
- **Product properties** (`product_properties`, `products.gaps`): rates, volume and mass yields, SG/API, and TBP 5/10/50/90/95 points. Also the 5--95 gaps between neighbouring cuts.
- **`CrudeUnit`**: the assembly a planner means by "the CDU": assay in, yield table out.
- **`CrudeDistillationUnit`**: the same unit behind difflow's operation protocol, for a `Flowsheet`, JSON and the editor.
- **`VacuumColumn`** (`difflow_refinery.vacuum`): the vacuum unit, atmospheric residue to LVGO, HVGO, slop and vacuum residue, with contaminants carried per cut.
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
  - `"twu"` (the default);
  - `"riazi_daubert_1987"`;
  - `"riazi_daubert_1980"`;
  - `"lee_kesler"`.

  Acentric factors are chosen so that each pseudo-component boils at its own `Tb`.

The cut points fix the number of pseudo-components and, with it, the shape of the column's equations. Keep them fixed when differentiating with respect to the assay.

This is not an assay library. Curated assays are proprietary data; the module characterises the curve the caller brings.

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

- a coil outlet of about 315 °C and about 52 MW fired;
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

**Non-convergence.** Some spec sets have no solution. With 5 % overflash, taking more than about 25 MW out of PA1 on the test column dries out the section above it. The column then reports `converged=False` with a finite state. `cdu_block` returns NaN at such a point, and the planner rejects any proposal at which a block is not finite ([A block that cannot be evaluated](planning.md#a-block-that-cannot-be-evaluated)).

On the test column, a credit on PA1 duty drives the planner past the edge. It proposes 30, 27, 25.5 and 24.75 MW, rejects each, and settles at 24.4 MW, which converges. With `mask_nonconverged=False`, the same run ends at 30 MW on a column that did not converge and reports itself as converged.

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

The vacuum distillation unit (VDU) takes the atmospheric residue to light and heavy vacuum gas oil (LVGO, HVGO), slop and vacuum residue. It is built on a stack of its own in `difflow_refinery.vacuum`:

- **Assays and characterization** (`vacuum.Assay`, `vacuum.characterize`): a TBP curve and bulk properties cut into pseudocomponents. The curve is extended past 565 C into the residue, and a residue lump gets its properties set directly. Sulfur, nitrogen, CCR, Ni+V and asphaltenes are carried per cut.
- **Heavy-end correlations** (`difflow_refinery.vacuum.correlations`): Twu critical properties and molecular weight, Kesler-Lee acentric factor and liquid Cp, Maxwell-Bonnell vapor pressure (the D1160 vacuum conversion).
- **A stage-network column** (`StageColumn`, `ColumnLayout`, `Route`, `StageSpec`): Naphtali-Sandholm MESH equations with liquids routed to side draws, pumparounds and entrainment, Murphree efficiencies, and any column output specifiable in place of any knob.
- **`VacuumColumn`**, the registered operation.

The vacuum stack does not yet share the crude unit's characterization: the two cut the crude on different grids. `vacuum.atmospheric_residue` stands in for the crude column's bottoms by applying an idealized TBP cut. The vacuum column only ever sees a difflow stream, so feeding it the `CrudeDistillationUnit`'s residue needs only a common component set.

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
| `Tb` | the mid-percent temperature of the cut |
| `SG` | from a constant Watson K, fitted so the whole crude matches its bulk SG |
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
| yield on feed | 0.154 | 0.488 | 0.030 | 0.328 | 0.070 | 0.296 | 0.030 | 0.603 |
| TBP T50 (C) | 393 | 484 | 605 | 674 | 391 | 473 | 595 | 753 |
| TBP T95 (C) | 450 | 579 | 700 | 868 | 450 | 566 | 834 | 888 |
| SG | 0.876 | 0.914 | 0.959 | 0.992 | 0.909 | 0.944 | 0.999 | 1.054 |
| S (wt%) | 0.74 | 1.10 | 1.38 | 1.45 | 2.73 | 3.98 | 5.14 | 5.50 |
| CCR (wt%) | 0.19 | 1.48 | 8.0 | 15.1 | 0.29 | 1.84 | 12.9 | 28.4 |
| Ni+V (wppm) | 0.00 | 0.14 | 8.1 | 57 | 0.01 | 1.05 | 100 | 495 |

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
D1160 (as opposed to TBP) product curves. The cross-check against an
independent simulator is against an equation-oriented re-implementation in
Pyomo/IPOPT on the same characterized feed, not against DWSIM; see
[Validation: the vacuum unit](#refinery-vacuum-validation) for what that does
and does not establish.

---

(refinery-gasplant)=
## The saturated gas plant

The gas plant (`difflow_refinery.gasplant`, #312) recovers the light ends.
Its feeds are the CDU overhead gas and unstabilised naphtha, and an FCC's
wet gas where there is one. Its products are fuel gas, LPG, and a
stabilised naphtha cut to a vapour-pressure spec:

```text
wet gas -> GasCompressor -> AmineTreater -> absorber-deethanizer -> fuel gas
                 | condensate                    | bottoms
                 +---------------------------->  +-> debutanizer -> LPG -> C3/C4 splitter
unstabilised naphtha ---------------------------^                \-> stabilised naphtha
```

At 10-20 bar, Raoult's law is no longer the right model, and the crude
column's thermodynamics would be wrong here by tens of percent in K.
The gas plant therefore runs on a cubic equation of state, Peng-Robinson
(default) or SRK. Real components and naphtha pseudocomponents go into
one `GasComponents` table, so one EOS covers the mixture:

```python
import difflow_refinery as dr
from difflow_refinery.gasplant import gas_components, debutanizer, GasPlantColumn

cuts = dr.characterize(dr.Assay([0, 50, 100], [360., 400., 470.], sg=0.74),
                       cut_points=[385., 420.])
comps = gas_components(["hydrogen_sulfide", "ethane", "propane", "isobutane",
                        "n_butane", "isopentane", "n_pentane", "n_hexane"], pseudo=cuts)
col = GasPlantColumn(debutanizer(comps, naphtha_rvp=80e3))
lpg, naphtha, info = col(feed)        # feed = {"F_propane": ..., "T": ..., "P": ...}
info["outputs"]["reboiler.duty"], info["outputs"]["bottoms.rvp"]
```

### Components and thermodynamics

`gas_components(light, pseudo=None, kij=None, cuts=None)` builds the
table from two sources. For the real species (hydrogen, H2S, N2, CO2, C1-C6 paraffins,
ethylene, propylene and the four butenes), it uses `difflow.database`
plus the tables in `gasplant/components.py`, whose sources are listed in
that module. For the pseudocomponents, it uses the refinery
characterisation's Tc, Pc, acentric factor and Watson-Nelson Cp. `kij`
is zero between hydrocarbons. The tabulated nonzero pairs (CO2, H2S and
N2 with the light paraffins) are recalled from the DECHEMA compilation
and are marked *verify* in the source. Each component also carries a
lower heating value, computed from its heat of formation.

`cuts=` keeps only the named cuts of the characterisation, in its own
order. A naphtha taken off a whole-crude characterisation carries the
first few cuts and almost nothing of the rest, and every cut in the
table is a column in every EOS call. `examples/38_refinery_gas_plant.ipynb`
keeps the cuts above 0.1 % of the naphtha and folds the remainder,
about 1e-4 of it, into the heaviest cut it keeps.

`CubicThermo` gives `ln K = ln phi_L - ln phi_V` at each stage's own
`(T, P, x, y)`, and residual enthalpies from the departure functions.
The cubic is solved in closed form. One Newton polish then carries the
root's exact implicit derivative, so no gradient passes through
`arccos`. Where the cubic has a single real root, both phases take it
and `K = 1`, as in any cubic-EOS package.

(refinery-gasplantcolumn)=
### GasPlantColumn

`GasPlantColumn` is the vacuum unit's [stage network](#refinery-stage-network)
on the EOS. The MESH equations use log flows, the same `StageSpec`
mechanism and the same implicit-function gradients. Trays are numbered
from 1 at the top. The column can have any of:

- a condenser that is `"total"` (the liquid at its bubble point, duty computed), `"partial"` (vapour product, duty a knob) or `None` (an absorber top);
- a kettle reboiler, or none;
- any number of feeds, each a stream at its own `(T, P)`, flashed once per solve;
- liquid side draws.

The factories return the parameters with the spec set each column is
normally run on:

| Factory | Products | Default specs (replacing) |
|---|---|---|
| `absorber_deethanizer` | `overhead` (fuel gas), `bottoms` | `bottoms.x.C2-` = 0.005 (reboiler duty); or `bottom.T` |
| `debutanizer` | `distillate` (LPG), `bottoms` (naphtha) | `distillate.x.C5+` = 0.01 (distillate rate); `bottoms.rvp` or `bottoms.x.C4` (reboiler duty) |
| `c3c4_splitter` | propane, butane | `distillate.x.C3` = 0.95; `bottoms.x.light` = 0.02 |
| `deisobutanizer` | isobutane, normal butane | `distillate.x.isobutane` = 0.95; `bottoms.x.isobutane` = 0.05 |
| `splitter` | any two-product cut | the cut placed by `light=`, specs as `(output, target)` pairs |

The absorber-deethanizer takes two feeds, `lean_oil` on tray 1 and
`feed`. Its lever is the lean-oil rate, which is the lean-oil stream's
own flow. Every factory accepts the `GasPlantColumnParams` fields as
keywords (`n_trays`, `top_P`, `side_draws=`, `eos="SRK"`, ...), so the
same factories serve a naphtha splitter (#311).

**Outputs** (`info["outputs"]`, SI) include:

- per product: `<p>.x.<component|group>`, `.recovery.<...>`, `.mol`, `.rate` (kg/s) and `.T`; for liquid products also `.rvp` and `.tvp`;
- `reflux_ratio`, `boilup_ratio`, `condenser.duty`, `reboiler.duty`, `energy_balance`;
- `top.T`, `bottom.T` and every `stage{j}.T`/`.P`;
- `<feed>.vapor_fraction`.

The groups are C2-, C3, C4, C3-, C4+ and C5+. Pseudocomponents count as
C5+. Any output can be specified.

**Tray efficiency.** By default the factories apply O'Connell's (1946)
correlation, `E_o = 0.492 (alpha mu_L)^-0.245`, as the Murphree vapour
efficiency of every tray. Alpha is the key components' relative
volatility at the feed. `mu_L` is `liquid_viscosity`, 0.1 cP by default,
which is a typical C3-C6 value and not a prediction. Two caveats apply:

- `E_MV = E_o` holds only at a stripping factor of one;
- the correlation is itself good to about 25%.

So the column is a rating model of that accuracy. `tray_efficiency=1.0`
turns the trays into theoretical stages.

**Vapour pressure.** `<product>.rvp` is the ASTM D323 construction on
the EOS: the liquid in contact with four times its volume of vapour, at
100 F. It is computed with the same EOS, not with a correlation.
`.tvp` is the bubble-point pressure at 100 F.

**Solve.** The solve runs in three passes, like the vacuum column's:

1. Equilibrium stages with easy specs. A rate is used for a rate slot, the reflux ratio for the distillate of a total condenser, and the boilup ratio for a reboiler duty.
2. Continuation of every target and efficiency to the user's values.
3. One implicit-function step.

The initial guess comes from the feed. Wilson K-values place the
temperature profile between the overhead's dew point and the bottoms'
bubble point. No user initialisation is needed.

(refinery-gascompressor)=
### GasCompressor

The wet-gas compressor has `n_stages` isentropic stages at equal
pressure ratios. Each stage's work is the isentropic enthalpy rise
divided by `efficiency`. An aftercooler and knockout drum follow each
stage. The condensate from all the drums leaves as one liquid stream,
which in a gas plant joins the absorber feed. `info` reports:

- `power` (W);
- `stage_power`;
- `discharge_T`;
- the stage pressure `ratio`.

Surge, choke and the compressor map are out of scope.

(refinery-aminetreater)=
### AmineTreater

H2S is a component throughout. The amine contactor is a fixed removal
fraction per component (`removal={"hydrogen_sulfide": 0.99}`), which
splits the gas into sweet gas and acid gas. Treating chemistry and
Merox are out of scope. For a rate-based contactor, see
`difflow_cc.AmineAbsorber`.

### Products

- `fuel_gas(flows, comps)` returns the rate, mass rate, MW, LHV (molar and mass), heat release and H2S ppm.
- `lpg_quality(flows, comps, grade)` checks the LPG against a GPA 2140 grade (`"HD-5"`, `"commercial_propane"`, `"commercial_butane"`). It returns `values`, signed `margins` (positive on spec) and `on_spec`. The vapour pressure is gauge, at 100 F, on the EOS.
- `reid_vapor_pressure(flows, comps)` and `true_vapor_pressure` give the same numbers the column reports.

The limits in `GPA_2140` are recalled values and are marked *verify*.
The standard writes its composition limits in liquid volume percent;
they are compared here as mole fractions, which differ by a few percent
of the value for C3/C4. The 95% evaporated, residue, copper strip,
sulfur and moisture tests are not computed.

### Planning with the gas plant

`gasplant_block(column, feeds, levers, outputs=None)` is the gas plant's
`cdu_block`. It returns a `difflow.planning.Block` whose delta vectors
are implicit-function Jacobians of the converged column. The levers are
of three kinds:

- every spec target by its own name (`distillate.x.C5+`, `bottoms.rvp`);
- every knob a spec has not replaced (`top.P`);
- per feed, `<feed>.mol`, `<feed>.T` and `<feed>.F_<component>`.

The outputs are in planner units (C, kPa, MW, kg/h, kmol/h).
Non-convergence is masked to NaN, as in `cdu_block`.

```python
blk = gasplant_block(col, [feed], ["distillate.x.C5+", "bottoms.rvp", "top.P", "feed.mol"],
                     outputs=["reboiler.duty", "condenser.duty", "distillate.rate"])
```

### Results

All four factories converge from the default initialisation, with no
warnings. They are run on two feeds: a straight-run feed (CDU light ends
with H2S, and two naphtha pseudocomponents) and an FCC feed (adding
hydrogen, ethylene, propylene and the four butenes). On both feeds:

- the total mass balance closes to 1e-15 relative;
- every component's balance closes to better than 1e-8;
- each column's energy balance closes to 1e-9 W on duties of order 1 MW.

On the deisobutanizer, the FCC butenes boil with the isobutane. The
olefin-rich case is therefore run at a 0.5 isobutane purity: a higher
purity is not available from that feed at any reflux, and the solve
says so by not converging.

The tests check the following, in `tests/refinery/test_gasplant.py`:

- The implicit gradients of LPG C5+, naphtha RVP and reboiler duty, with respect to the reflux ratio, the top pressure and a feed component, match central differences to 1e-5 relative.
- The reboiler duty rises monotonically as the naphtha RVP spec is tightened.
- The absorber-deethanizer's C2 slip falls monotonically as its bottoms temperature rises.

### The gas plant on a crude unit

`examples/38_refinery_gas_plant.ipynb` runs the whole chain on the CDU
of `examples/35_refinery_cdu_planning.ipynb`, with a partial condenser
held at 40 C:

- the offgas goes through a two-stage compressor to 14.5 bar, then the amine treater;
- an absorber-deethanizer takes the whole unstabilised naphtha as lean oil;
- a debutanizer makes the LPG;
- a naphtha splitter makes light and heavy naphtha.

Every column converges from the default initialisation. The material
balance across the plant closes to 1e-12 mol/s on 280 mol/s. The
debutanizer's implicit derivatives with respect to its C4 spec match
central differences to the digits printed.

This crude's offgas is mostly C3/C4. At 14.5 bar and 40 C almost all of
it condenses in the compressor's knock-out drums, so the amine treats a
few percent of what was compressed. The condensate carries most of the
H2S past it, into the fuel gas and the LPG.

The H2S figure starts from an assumption. The assay says nothing about
sulfur, so the offgas is given 2 mol % H2S.

### Gas plant gotchas

- **The naphtha sets a floor on its own RVP.** A stabiliser cannot bring the naphtha below the RVP of its C5+ part. The C5/C6 in the test feed alone sit near 70 kPa. A 60 kPa spec is infeasible, and Newton does not converge.
- **Purity specs must be reachable on the trays you gave.** At O'Connell efficiencies near 0.5, a 12-tray debutanizer is about six theoretical stages. That is not enough for 1% C5+ in the LPG and 1% C4 in the naphtha together.
- **A C2- spec has to be smaller than the C2- there is.** With the whole naphtha as lean oil, the absorber-deethanizer's bottoms are about 290 mol/s. At the factory's 0.5 %, that is 1.4 mol/s of C2-, but the CDU feeds bring in 1.07 mol/s. The spec cannot be met at any duty, so the solve does not converge. The example uses 0.2 %.
- **A hot feed sets a ceiling on the naphtha's RVP.** The deethanizer bottoms reach the debutanizer at 180 C. On that feed, an RVP spec of 40 or 50 kPa converges, at 2.7 and 1.7 MW. At 70 kPa the reboiler duty would have to go below zero, and the solve does not converge.
- **A heavy lean oil.** When the lean oil is a hundred times the gas, the guess's vapour profile is its 5 %-of-feed floor, and a pass-1 boilup ratio taken from it is a few percent: pass 1 then has almost no vapour and never converges. Pass 1's boilup ratio is therefore at least one (`test_deethanizer_with_a_lean_oil_a_hundred_times_the_gas`).
- **Compile once per spec structure.** The first solve compiles for 10-40 s. Later solves with the same structure (which specs replace which knobs) reuse the compiled solve for any numbers.

### Gas plant: out of scope

Treating chemistry, Merox, cryogenic C2 recovery, column hydraulics and
compressor surge.

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
| RON, MON | `RON`, `MON` | Ethyl RT-70 interaction model; or linear by volume | Healy, Maassen & Peterson (1959), Ethyl report RT-70 |
| RVP | `RVP_psi` | RVP^1.25 index by volume; Raoult on the pseudocomponents; or linear | Gary, Handwerk & Kaiser, product blending chapter |
| Sulfur, nitrogen, CCR | `S_ppm`, `N_ppm`, `CCR_wt` | by mass | - |
| Aromatics, olefins, benzene, PNA, smoke point | `*_vol`, `smoke_mm` | by volume | - |
| Flash point | `flash_C` | Hu-Burns index, `log10 BI = -6.1188 + 2414/(T - 42.6)` (K), by volume | Hu & Burns (1970); identical to Wickey-Chittenden in °F |
| Cloud, pour point | `cloud_C`, `pour_C` | `BI = T^n` (K), n = 1/0.05 and 1/0.08 | Hu & Burns (1970) |
| Freeze point, CFPP | `freeze_C`, `CFPP_C` | `BI = T^n`, cloud-type and pour-type exponents | approximation, override with `rules={"freeze_C_exponent": ...}` |
| Viscosity | `viscosity_cSt` | Refutas VBN `14.534 ln ln(nu + 0.8) + 10.975`, by mass | Refutas, as given in Maples (2000) |
| Distillation | `E70_tbp`, `E100_tbp`, `T{10,50,90,95}_d86_C` | from the blend's composition: smoothed TBP, then Riazi-Daubert TBP->D86 | Riazi & Daubert (1986) |
| Cetane index | `cetane_index` | ASTM D4737 (default) or D976 on the blend's density and D86 points; computed, never blended | ASTM D4737, D976 |

Choose rules per property with `BlendPool(rules={"octane": "volume", "RVP_psi": "raoult", "cetane": "d976"})`.

The RT-70 corrections are all *spreads*: covariances and variances across the components of sensitivity, olefins and aromatics. The model therefore reduces exactly to the linear blend when the components agree. The coefficients are the published 75-blend fit (`a1 = 0.03224`, `a2 = 0.00101`, `a3 = 0`, `b1 = 0.04450`, `b2 = 0.00081`, `b3 = -0.0645`, the last on the squared aromatic spread / 1e4), as restated in the gasoline-blending literature. They live in `EthylRT70` and can be refitted. On a reformate/FCC pool the MON correction is several octane numbers, so check them against your own blend data.

The distillation points are on a TBP curve made differentiable by spreading each pseudocomponent with a logistic of `distillation_width` (default 5 K). Each point is then converted to D86 with the Riazi-Daubert correlation, because specs are written on D86. `E70`/`E100` stay on the TBP basis, and the key says so.

### Specs and margins

`BlendSpec(property, op, limit, scale=1.0)`. The margin is `value - limit` for `>=` and `limit - value` for `<=`, divided by `scale`, so **positive means on spec**. `PRODUCT_SPECS` holds illustrative defaults for each product: US regular summer gasoline, Jet A, ULSD S15, and VLSFO RMG 380. Pass your own specs for anything real.

- `spec_margins(components, recipe)` returns the margin vector, smooth in everything. With `weighted=True` it returns `V * margin` instead. A property is intensive, so it is 0/0 at an empty pool. The weighted form is the same constraint wherever `V > 0`, stays finite at zero, and is the form an LP's blending rows take. Use it as the constraint for an optimizer over volume flows.
- `spec_violations(..., temperature=t)` returns `max(-margin, 0)`, or its softplus `t * logaddexp(-margin/t, 0)`. The smooth form is within `t ln 2` of the kink and its derivative at an active spec is exactly `-1/2`. The branchless `max(u,0) + log1p(exp(-|u|))` has the same value and a derivative of zero there; see the CVaR note in `difflow.stochastic`.

### Linear blend error and back-off

`linear_properties` is the planning-LP view: every property linear by volume. `linear_blend_error` is nonlinear minus that view, and it is what a planner's back-off is meant to cover. Pass `exact=("RVP_psi", "S_ppm")` for properties the LP already models with the pool's own rule (the RVP index, sulfur by mass). Their error is then zero. `backoff` turns the error into a per-spec tightening, `max(linear margin - nonlinear margin, 0)`.

The example measures what this means for a gasoline LP:

- without back-off, the LP promises about 50% more margin than any feasible plan, and its recipe is about two MON numbers off spec;
- the back-off depends on the recipe, so one pass is not enough; successive back-off takes about a dozen passes to settle;
- on that pool the converged plan coincides with the best local optimum of the nonlinear problem, but a single-start NLP stops at a worse one. Blending is nonconvex, so neither tool is safe alone.

### Planning hook

`pool.as_block(components)` returns a `difflow.planning.Block`. Its levers are the component volume flows (`<name>_V`), bounded by availability in stream mode. Its outputs are the product `volume`, the spec properties, and `margin:<spec>` for every spec. `jax.jacobian` of the block is the delta-vector set: the volume row is the volume balance, all ones, and the property rows are the blend's sensitivities at the linearisation point.

### Blend characterization

`BlendCharacterization(names, Tb, SG, MW=, Tc=, Pc=, omega=, qualities=)` is the pseudocomponent grid. Molecular weight and critical constants come from Riazi-Daubert (1980), the acentric factor from Edmister, and vapor pressure from Lee-Kesler. Any of them can be overridden per pseudocomponent by passing a vector with NaN where the correlation should be used. That is how a defined component such as n-butane takes its own constants. `qualities` holds the per-pseudocomponent composition vectors (`S_ppm`, `aromatics_vol`, ...), each averaged on its own basis.

This is the minimum a blend pool needs to compute properties from composition. It is not an assay model: there is no TBP fitting and no heavy-end extrapolation. The CDU/VDU work is where those belong.

### Not in scope

- Tank inventory and multi-period scheduling. The pool is steady state, per period.
- Crude blending ahead of the CDU (assay mixing).
- A straight-run octane correlation from PNA. Octane is unit-reported or measured.

(refinery-validation)=
## Validation

The crude unit has been checked against an independent reference. The issue asked for a DWSIM (or HYSYS or PRO/II) crude case. None of those were available, so the reference is built with **IDAES** 2.10 (Pyomo 6.10, IPOPT 3.13.2): the U.S. DOE's open equation-oriented process modelling platform. That choice decides what the check can say. It is an independent *implementation*: different code, a different solver, a different formulation and a different starting point. It is not an independent *model*: the column and its property model are the ones difflow states, written again. A commercial simulator's crude case would also bring its own characterisation and its own thermodynamics. Here those are tested separately, against published numbers and against Peng-Robinson.

The generator is `tests/refinery/reference/generate.py`. It writes `cdu_reference.json`, which records the tool versions, the property methods, the date and the difflow commit. `tests/refinery/test_validation.py` checks difflow against that file, with each tolerance and the reason for it written beside the assertion. The test needs neither IDAES nor IPOPT. The case is the 95 000 bbl/d test crude in the 30-stage column used throughout this page: three side strippers, two pumparounds, bottom and stripper steam, a total condenser, and a 5 vol % overflash. The column builder (`reference/mesh.py`) takes a stage network, not a fixed layout. The vacuum column has its own (below): its liquid routes, Murphree beds and Maxwell-Bonnell property model did not fit this one without rewriting most of it.

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
  - IDAES's crude bubble point (381 K) and dew point (847 K) at the flash-zone pressure satisfy difflow's sum z K = 1 and sum z / K = 1.

  This confirms that difflow's arrays implement the equations it states. It says nothing about whether those equations are right.
- *Peng-Robinson on the same Tc, Pc, omega and ideal-gas Cp* (kij = 0, hydrocarbons only). These are modelling differences. They are documented and pinned in the tests, not tuned away:
  - On every stage, and at the coil outlet, the vapour fraction agrees within 0.021. At the coil outlet the difference is 0.0035.
  - At the furnace inlet (240 C, 6 bar), Raoult vaporises 22 mol % of the crude and PR vaporises 15 %.
  - The crude's enthalpy rise from the furnace inlet to the coil outlet is **4.0 % higher under PR**. That is the furnace-duty difference a PR crude case would show from the property model alone.
  - For the cuts boiling 420-640 K, which make the side products, PR and Raoult K-values agree within -30 % / +25 % at the flash zone.
  - Raoult over Lee-Kesler badly overpredicts the supercritical light ends. Propane's K is about 60 times PR's. The light ends go overhead under either model, but do not read a light-ends K-value off this model.
  - For the heaviest residue cut (Tb 1033 K), PR's K is about 20 times Raoult's.
  - Liquid enthalpy at the same composition agrees within 1.5 kJ/mol above the flash zone. Where residue is in the liquid, PR's is 13-18 kJ/mol higher. Watson's latent heat and PR with an extrapolated omega are both extrapolations for a 1000 K cut, and nothing here says which is closer to the truth.
- *Water.* difflow's Wagner-Pruss Psat agrees with IAPWS-95 to 0.004 % and with IAPWS-IF97 to 0.02 %. Watson's latent heat for water is exact at Tb, 1.3 % high at 300 K and 2.6 % low at 550 K, compared with IAPWS-95 by Clausius-Clapeyron.

**Layer 3: the column, against an independent EO model.** `reference/mesh.py` writes every stage, stripper, pumparound, the condenser and the furnace flash as Pyomo equations with the same specs. It solves them with IPOPT from a linear 380-580 K profile and round-number flows; no difflow result is used to initialise it. It converges in about 25 s. The two solutions agree as follows:

- stage, stripper and condenser temperatures and the coil outlet (588.0 K): 6e-6 K
- condenser duty (32.39 MW): 6e-7 relative
- fired duty (51.59 MW): 3e-10 relative
- pumparound return temperatures: agree within 1e-3 K (the test tolerance)
- feed vaporised (0.691): agrees within 1e-6 (the test tolerance)
- volume yields: 4e-10
- API gravities: 2e-7
- TBP 5/10/50/90/95 % points: 5e-7 K
- 5-95 gaps: 6e-7 K
- steam saturation: within the 0.02 % that separates the two water Psat formulations

The test tolerances are set at the solvers' precision, not at engineering accuracy, so a transcription error in either column has nowhere to hide. The same column with Watson's floor unsmoothed (eps = 0.01 → 1e-6) moves the fired duty by 0.10 %, the condenser duty by 0.003 %, stage temperatures by at most 0.019 K and the API gravities by at most 2e-4. That is the price of the smoothing that gives difflow a derivative everywhere.

**Layer 4: gradients.** Central finite differences of the reference column (each spec ± 0.002, IPOPT warm-started) against `jax.grad` through difflow's implicit-function solve:

| Gradient | `jax.grad` | Reference FD | Agreement |
| --- | --- | --- | --- |
| d(diesel API)/d(diesel vol. yield) | -30.0685 | -30.0689 | 1.5e-5 |
| d(fired duty)/d(overflash) | 170.915 MW | 170.920 MW | 3.2e-5 |

The file also stores d(condenser duty)/d(diesel yield) and d(residue API)/d(overflash) for later use.

**What this does not validate.** It does not show that Raoult/Watson is the right property model for a given crude; layer 2 measures how far it is from PR, nothing more. It does not cover a commercial simulator's characterisation, its D86 interconversion defaults, or its tray-efficiency and hydraulics models, and it does not replace plant data. The 5-95 gaps of this case are negative (-27 to -55 K): equilibrium stages with these specs give overlapping products. Both implementations agree on that, which says nothing about whether a real column would overlap the same way. When a deliberate model change moves any number above, `TestReferenceIsCurrent` fails first and asks for the reference to be regenerated.

(refinery-vacuum-validation)=
### Validation: the vacuum unit

The vacuum column (#294) is checked the same way and with the same caveat. DWSIM was not available. IDAES has no vacuum-column model with pumparounds, a wash bed, entrainment routes and Murphree beds. So the column is written again in Pyomo, IDAES's modelling layer, as an equation-oriented MESH model (`tests/refinery/reference/vdu_mesh.py`) and solved with IDAES's IPOPT 3.13.2. Its property model (`vdu_formulas.py`) is transcribed from the published correlations:

- Maxwell-Bonnell vapour pressure in Rankine, with the Watson-K correction;
- Kesler-Lee liquid Cp;
- Clausius-Clapeyron latent heat;
- NIST Shomate steam.

This is an independent implementation of the model difflow states, not an independent model.

The two share only the input: difflow's characterised residue (the `heavy_crude` assay cut at 370 C, 21 pseudo-components, 66.2 kg/s). Otherwise the reference does each thing differently:

- mole fractions and total flows where difflow uses log component flows;
- absolute route flows where difflow uses softmax draws;
- an explicit summation equation;
- pumparound return temperatures as unknowns, where difflow makes the duties the knobs;
- the LVGO end point as a smooth cumulative-mass equation;
- IPOPT where difflow uses a damped Newton.

The reference starts from an engineering guess built from the feed flash and the specs, not from difflow's answer. It solves in two steps:

1. With the overflash spec relaxed and the HVGO draw held, so the wash bed stays wet.
2. With the spec restored.

Together they take about 12 s.

The case is the 2-2-2-2 bed layout of this page, at a 400 C furnace outlet, 30 mmHg at the flash zone and 0.5 wt % stripping steam. Its specs are a 70 C top, 3 wt % overflash and a 450 C LVGO T95. Regenerate it with

    PYTHONPATH=src:tests python -m refinery.reference.vdu_generate

which writes `vdu_reference.json`: provenance, the frozen component table, both columns, two model variants and the finite differences. `tests/refinery/test_vdu_validation.py` (release) compares against it. `test_vdu_validation_file.py` runs on every commit and checks two things: the file is intact, and difflow's characterisation is still the one the reference was built on.

| Quantity | difflow | Reference | Agreement (test tolerance) |
| --- | --- | --- | --- |
| LVGO / HVGO / slop / residue (kg/s) | 4.66199 / 19.6104 / 1.98588 / 39.9371 | same | ≤ 1e-7 rel (1e-6) |
| LVGO / HVGO pumparound duty (MW) | 2.8141 / 13.6700 | same | ≤ 1.3e-7 rel (1e-6) |
| Furnace duty (MW), vapour fraction | 13.1145, 0.30368 | same | 2e-10, 4e-8 rel (1e-6) |
| Stage temperatures, flash zone 669.52 K | | | ≤ 1e-5 K (1e-4 K) |
| Product TBP 5-95 % points; SGs | | | ≤ 1e-5 K (1e-4 K); ≤ 3e-9 (1e-8) |

The second case is Murphree beds: LVGO 80 %, HVGO 70 %, wash 50 %, stripping 40 %. It agrees just as closely: rates within 3e-7, duties within 1.6e-7, temperatures within 2e-5 K. With these beds the LVGO rate rises to 15.25 kg/s and HVGO falls to 8.01 kg/s, so the comparison exercises the efficiency path, not an unchanged column. The case needs a 520 C LVGO end point. With beds this poor, enough heavy vapour reaches the LVGO section that the 450 C spec cannot be met at any pumparound duty. Both implementations stop finding a column halfway from equilibrium to these efficiencies: difflow does not converge and IPOPT reports local infeasibility. That is consistent with the existing infeasible-spec test, although a local NLP verdict is not a proof. The reference reaches this case by continuation from its own equilibrium solution.

The remaining ~1e-7 differences come from one coefficient. The reference writes the Watson-K correction's coefficient as 2.5/1.8 in Rankine, and difflow uses 1.3889 in SI.

The reference also measures what two of difflow's numerical choices cost. These are not disagreements; each change applies to both implementations alike:

- **Four Watson-K passes instead of the exact fixed point.** The fixed point moves ln Psat at the top stage by 6e-3, but the products by under 1e-8.
- **Blending Maxwell-Bonnell's branches instead of switching at the published joins.** This keeps a derivative everywhere. It moves the LVGO rate by 8.7e-4, the pumparound and furnace duties by 8-9e-4, and the flash zone by 0.01 K.

**Sensitivities.** `jax.jacfwd` through difflow's solve was compared with central differences of the reference, with IPOPT warm-started at each perturbed point. The perturbations were furnace outlet ± 0.25 K, flash-zone pressure ± 10 Pa and steam ± 2.5e-5 kg/kg. Over the rates, duties, flash-zone temperature and HVGO T95 they agree within 4.1e-5 relative (tolerance 2e-4); the remaining difference is the finite differences' own truncation error. Two examples are d(HVGO rate)/d(furnace T) = 0.15520 kg/s/K and d(HVGO pumparound duty)/d(steam) = 234.39 MW per kg/kg.

**What this does not validate.** It does not test the property model. Maxwell-Bonnell with Raoult at 10-30 mmHg is a choice that nothing here tests against data or an equation of state; the crude unit's layer 2 is the nearest evidence. It does not cover a commercial simulator's vacuum characterisation, packing HETP and pressure-drop models, or the ejector system, and it does not replace plant data.

---

(refinery-limitations)=
## Limitations

- **The crude unit is the atmospheric column only.** The preflash drum and preheat train are not modelled; the inlet is the preheat train's outlet. The vacuum unit is separate (above) and is not yet fed from it.
- **Thermodynamics:** Raoult's law and ideal-gas-path enthalpies. This is the usual model for an atmospheric column at one or two bar; it is not a cubic equation of state.
- **Equilibrium stages.** There are no tray efficiencies or hydraulics.
- **Boiling ranges are TBP, not ASTM D86.**
- **Validation:** against an independent equation-oriented model, IDAES property packages and published characterisation examples; not against a commercial simulator's crude case. The vacuum column likewise, against an independent Pyomo/IPOPT model on the same residue (equilibrium and Murphree beds, and sensitivities); not against DWSIM. See [Validation](#refinery-validation) and [the vacuum unit's](#refinery-vacuum-validation) for what that does and does not establish.
