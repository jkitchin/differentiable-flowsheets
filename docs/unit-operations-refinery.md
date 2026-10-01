# Refinery Unit Operations

This document covers the `difflow_refinery` plugin: crude assay
characterization into pseudocomponents, and a differentiable vacuum
distillation unit (VDU) built on a general network-of-stages column.

---

(refinery-overview)=
## Overview

The plugin provides:

- **Assays and characterization** (`Assay`, `characterize`): a TBP curve and
  bulk properties cut into pseudocomponents. The curve is extended past 565 C
  into the residue, and a residue lump gets its properties set directly.
  Sulfur, nitrogen, CCR, Ni+V and asphaltenes are carried per cut.
- **Heavy-end correlations** (`difflow_refinery.correlations`): Twu critical
  properties and molecular weight, Kesler-Lee acentric factor and liquid Cp,
  Maxwell-Bonnell vapor pressure (the D1160 vacuum conversion), with
  Riazi-Daubert and Lee-Kesler kept for comparison.
- **A stage-network column** (`StageColumn`, `ColumnLayout`, `Route`, `Spec`):
  Naphtali-Sandholm MESH equations solved simultaneously. Liquids can be
  routed to side draws, pumparounds and entrainment, and any column output
  can be specified in place of any knob.
- **`VacuumColumn`**: atmospheric residue to LVGO, HVGO, slop and vacuum
  residue.

Everything is differentiable end to end. The gradient of an HVGO end point,
a residue yield or the HVGO sulfur content is exact, and it costs one linear
solve. That holds for the furnace outlet temperature, the flash-zone
pressure, the stripping steam, every spec target and stage efficiency, and
every number in the assay, down to a single TBP point.

There is no `CrudeColumn` yet. `atmospheric_residue` stands in for its
bottoms: it applies an idealized TBP cut with a realistic overlap. The
vacuum column only ever sees a difflow stream, so swapping the stand-in for
a real atmospheric column changes nothing downstream.

---

(refinery-installation)=
## Installation

The plugin ships with difflow and registers itself through the
`difflow.plugins` entry point:

```bash
pip install difflow[refinery]
```

---

(refinery-quick-start)=
## Quick start

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

(refinery-characterization)=
## Characterization

An `Assay` holds what a lab reports: a TBP curve (cumulative wt% distilled
against temperature) and bulk properties. Two synthetic assays of the right
shape are included, `light_crude()` (38 API, 15% vacuum residue) and
`heavy_crude()` (20 API, 40%). They are not the assays of named crudes.

`characterize(assay, cuts_C=...)` cuts the curve on a temperature grid. The
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

(refinery-thermodynamics)=
## Thermodynamics at vacuum

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
## The stage-network column

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
`Spec(output, target, replaces=...)` trades one of those for
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
## VacuumColumn

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

### Results on the two included assays

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

(refinery-gotchas)=
## Gotchas

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

(refinery-out-of-scope)=
## Out of scope

Ejectors and vacuum-system modelling, dynamics, lube vacuum towers, and
D1160 (as opposed to TBP) product curves. A cross-check against an
independent simulator (DWSIM or IDAES on the same characterized feed) has
not been done; neither is available in the test environment.
