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
- **The preheat train** (`difflow_refinery.preheat`): tank to furnace inlet. It covers the exchangers, the desalter and the preflash drum, and is solved together with the column whose products and pumparounds heat it (`PreheatedCrudeUnit`). It also provides the Ebert-Panchal fouling rates and a cleaning ranking from one gradient. The palette operations are `Desalter`, `PreflashDrum` and `CrudeUnitWithPreheat`.
- **`VacuumColumn`** (`difflow_refinery.vacuum`): the vacuum unit, atmospheric residue to LVGO, HVGO, slop and vacuum residue, with contaminants carried per cut. It runs on the crude unit's own pseudo-components, so the CDU residue feeds it directly in a `Flowsheet`.
- **Correlations** (`difflow_refinery.correlations`): Twu, Riazi-Daubert, Lee-Kesler, Kesler-Lee and Maxwell-Bonnell, each written once, for all three of the above.
- **The fluid catalytic cracker** (`difflow_refinery.fcc`): a lumped-kinetics riser (3-, 4- or 5-lump) and a coke-burning regenerator solved together as the unit's heat balance (catalyst circulation and regenerator temperature are unknowns, the riser outlet temperature the spec), with a simplified main fractionator; dry gas, C3/C4 olefin streams, gasoline, LCO and slurry. A library, not a palette operation; its kinetic constants are illustrative. See [The fluid catalytic cracker](#refinery-fcc).
- **C5/C6 isomerization** (`difflow_refinery.isomerization`): an adiabatic approach-to-equilibrium reactor on ideal-gas thermochemistry, with a shortcut stabilizer and optional DIP and DIH columns. The DIH recycle is converged by Anderson and differentiated implicitly. `isom_block` links the isomerate to a blend pool. The palette operations are `IsomerizationReactor` and `IsomerizationUnit`.
- **Product blending** (`BlendPool`, `BlendComponent`): gasoline, jet, ULSD and fuel-oil pools with the nonlinear blending rules, signed spec margins and LP back-off. A library for optimisation and planning, not a palette operation.
- **Catalytic reforming** (`difflow_refinery.reforming`): a semi-regen reactor train with fired heaters, a PR separator, H2 recycle through a `Flowsheet` tear and a stabilizer; naphtha P/N/A by carbon number in, reformate (with RON from composition), net H2, LPG and fuel gas out. A library and flowsheet, not a palette operation.
- **Hydroprocessing building blocks** (`difflow_refinery.hydroprocessing`) and the **hydrotreater** (`difflow_refinery.hydrotreating`): a trickle-bed reactor around any kinetic model, a Peng-Robinson HP separator, the recycle-gas loop and a steam stripper; HDS by sulfur class, HDN and aromatics saturation on the #305 composition. A library, not a palette operation. See [Hydroprocessing](#refinery-hydroprocessing) and [The hydrotreater](#refinery-hydrotreater).
- **Alkylation** (`difflow_refinery.alkylation`): C3-C5 olefins + isobutane over H2SO4 or HF, the Sauer-Colville-Burwick correlations (cross-checked against GAMS `process.gms`), shortcut DIB/depropanizer/debutanizer and the isobutane recycle as a `Flowsheet` tear; alkylate to `BlendPool`, `alky_block` for planning. See [Alkylation](#refinery-alkylation).
- **The VGO hydrocracker** (`difflow_refinery.hydrocracking`): the same building blocks with a pretreat bed (the hydrotreating kinetics, VGO constants), a cracking bed on continuous lumping over the pseudo-component grid (Laxminarasimhan et al. 1996, or discrete lumps) with organic-N inhibition, a simplified fractionator and a UCO recycle tear. A library, not a palette operation; its cracking constants are illustrative. See [The hydrocracker](#refinery-hydrocracker).
- **The hydrogen network** (`difflow_refinery.hydrogen`): producers (the reformer's net gas, an H2 plant, imports), consumers (hydrotreater and hydrocracker makeup with a purity or partial-pressure spec), an optional PSA, purge to fuel gas and export, on one or more headers; returns the balanced header and each consumer's makeup purity, feeds it back into the hydrotreaters (`close_hydrotreater_loop`), and `h2_block` for planning. A library. See [The hydrogen network](#refinery-hydrogen).
- **Residue desulfurization and fuel oil** (`difflow_refinery.residue`, #331): an atmospheric-residue desulfurizer on the same building blocks (HDS by sulfur class plus refractory residue sulfur, HDM of Ni+V, CCR reduction, a small 538 C+ conversion; once-through treat gas, ideal product split) and the VLSFO pool (`fuel_oil_blend`). It takes a 3 wt% S residue to a 0.5 wt% S fuel oil, which cutter blending alone cannot. A library; constants illustrative. See [Residue desulfurization and fuel oil](#refinery-residue).

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

(refinery-composition)=
### Hydrocarbon type, hydrogen and heteroatom classes

A boiling point and a gravity tell you how a cut distils. They do not tell you what it is made of. The conversion units need that. A hydrotreater's hydrogen demand follows the aromatics it saturates and the sulfur *classes* it has to remove: a thiol goes at any severity, a 4,6-dimethyldibenzothiophene only at the last ppm. A reformer converts naphthenes. An FCC cracks paraffins and condenses aromatics. `difflow_refinery.composition` (#305) puts these numbers on every component of a `Characterization`. They are per-component arrays that travel with the component flows, the way sulfur, nitrogen and CCR already do, so the composition of any stream of those components is a weighted average.

```python
import difflow_refinery as dr
from difflow_refinery import composition as cm

char = dr.characterize(assay, composition=True)        # or char.with_composition(...)
comp = char.composition                                # a cm.Composition
comp.hc_type          # (n, 4) vol fractions: paraffins, naphthenes, aromatics, olefins
comp.hydrogen         # (n,) mass fraction
comp.sulfur_classes   # (n, 5) mass fraction of S in each class
comp.of_stream(stream).as_dict()   # aromatics_vol, hydrogen_wt, S_<class>_wt, N_basic_wppm, ...

props = dr.product_properties(result.products, thermo, feed, composition=comp)
props["diesel"].composition.aromatics, props["diesel"].composition.sulfur_classes_wt
```

**What each component carries.** Every array has one row per component in `char.names` order: light ends first, then the cuts, residue lump last.

| Array | Shape | Units | Notes |
| --- | --- | --- | --- |
| `hc_type` | `(n, 4)` | volume fraction | columns `HC_TYPES` = paraffins, naphthenes, aromatics, olefins; each row sums to 1. Olefins are 0 in a straight run; a cracking unit sets them. |
| `hydrogen` | `(n,)` | mass fraction | |
| `sulfur`, `nitrogen` | `(n,)` | mass fraction | the characterization's own vectors |
| `sulfur_split` | `(n, 5)` | share of the component's S | columns `SULFUR_CLASSES` = sulfides (thiols and sulfides), thiophenes, benzothiophenes, dibenzothiophenes, hindered_dbts (4- and 4,6-alkyl DBTs); rows sum to 1 |
| `nitrogen_split` | `(n, 2)` | share of the component's N | columns `NITROGEN_CLASSES` = basic, non_basic |
| `refractive_index`, `d20` | `(n,)` | -, g/cm³ | n20 used for the estimate, density at 20 °C |
| `MW`, `SG` | `(n,)` | g/mol, 60/60 °F | the averaging weights |

Derived: `sulfur_classes` = `sulfur[:, None] * sulfur_split`, and likewise `nitrogen_classes`, `carbon` (`1 - H - S - N`) and `ch_ratio`. Light ends are exact: they are paraffins, carry their formula's hydrogen, and have no sulfur or nitrogen.

**A stream or a product.** `comp.of_flows(moles)` (mol/s in `names` order, or `basis="mass"` for kg/s) and `comp.of_stream(stream)` return a `StreamComposition`. It ignores `F_water`, `F_H2O` and any species not in `names`. The types are averaged by standard liquid volume, `m_i / SG_i`. Hydrogen, sulfur, nitrogen and their classes are averaged by mass. The fields are `hc_type` `(4,)`, `hydrogen_wt` (wt%), `sulfur_wt` (wt%), `nitrogen_wppm`, `sulfur_classes_wt` `(5,)` (wt% of the stream, summing to `sulfur_wt`), `nitrogen_classes_wppm` `(2,)`, and the `mass` and `volume` they were weighted on. `product_properties(..., composition=comp)` puts one on every `ProductProperties.composition`. `BlendCharacterization.from_characterization(char)` takes the types as the `paraffins_vol`, `naphthenes_vol`, `aromatics_vol` and `olefins_vol` qualities (vol%), so a crude-unit product reaches `BlendPool` with its aromatics. A unit that changes a component's composition, for example a hydrotreater saturating aromatics, builds its own `Composition` with `comp.replace(hc_type=..., hydrogen=..., sulfur=...)` and reports through the same `of_flows`.

**How each cut is estimated.** Each step is selectable in `CompositionData`, and each can be overridden by measurements:

1. **Refractive index n20.** Measured, or Riazi-Daubert's Huang index `I = (n² - 1)/(n² + 2)` from `(Tb, SG)`. Below 620 K this is the Riazi-Daubert (1987) six-constant form [R2]. Above it is Riazi's (2005) heavy-fraction form [R3]. The two disagree by about 0.01 in n20 at 620 K on gas-oil gravities, so they are joined by a logistic with a 20 K scale rather than switched. `d20` comes from SG (API).
2. **Hydrogen.**
   - `hydrogen_method="goossens"` (the default) uses Goossens (1997), from `(n20, d20, MW)`. It is the one meant for heavy cuts.
   - `"riazi_daubert"` uses the Riazi-Daubert (1987) C/H weight ratio from `(Tb, SG)` [R2], with `H = (1 - S - N)/(1 + C/H)`.
   - A measured hydrogen content replaces either.
3. **Hydrocarbon types.**
   - `pna_method="riazi_daubert"` (the default) is API procedure 2B4.1, Riazi-Daubert (1986), in the form that needs no viscosity. It takes the refractivity intercept `Ri = n20 - d20/2`, the C/H weight ratio, and `m = MW (n20 - 1.475)`.
   - `"ndm"` is the n-d-M method of ASTM D3238. It gives *carbon* types: the share of carbon atoms in aromatic rings, naphthenic rings and chains. An alkylbenzene is one aromatic molecule but mostly paraffinic carbon. Read its output as an approximation to molecule types, not as molecule types.
   - `riazi_daubert_pna_vgc` gives 2B4.1's viscosity-gravity-constant form for a caller who has a measured viscosity. A TBP assay does not give one, so the estimate does not use it.

   The C/H ratio that feeds the type correlation is computed *from the hydrogen content*, as `(1 - H - S - N)/H`. Hydrogen and type are therefore one estimate, and they move together. A cut measured richer in hydrogen comes out less aromatic.
4. **Projection.** A correlation can stray outside [0, 1] at the edge of its range. Its P, N and A are taken through a smooth positive part (`t·logaddexp(x/t, 0)`, `t = 1e-3`) and renormalized, so each cut's types are a composition and stay differentiable.

2B4.1's two branches, MW ≤ 200 and MW > 200, do **not** agree at 200. On straight-run kerosene and diesel cuts the heavy branch gives 0.1--0.2 less aromatics than the light one. The branches are joined by a logistic in MW with a 10 g/mol scale, so the composition stays smooth in the assay. Each branch is reproduced, to 1 % of the gap between them, beyond 200 ± 46. Near the join the estimate is a ramp between two published answers that disagree, and it is not more accurate than either. `blend=False` gives the published hard switch.

**Measured data overrides the estimate cut by cut.** `CutData(values, T=None, extrapolate="estimate")` holds either per-cut values (`n_cuts` entries, residue lump included, NaN where nothing was measured) or a curve over boiling point that is interpolated at each cut's Tb. For a curve, the default `extrapolate="estimate"` gives cuts outside the measured range the correlation, because a naphtha PIONA says nothing about the gas oil. `"flat"` holds the end values, as the contaminant curves do.

```python
data = cm.CompositionData(
    hc_types=cm.CutData({"paraffins": [0.55, 0.45], "naphthenes": [0.30, 0.33],
                         "aromatics": [0.15, 0.22]}, T=[340.0, 450.0]),   # PIONA of the naphtha
    refractive_index=n20_per_cut,              # NaN where not measured
    hydrogen_wt=cm.CutData([14.2, 12.1], T=[480.0, 650.0]),
    sulfur_split=(T_anchor_K, shares_k_by_5),  # or an (n_cuts, 5) array
    nitrogen_split=(T_anchor_K, basic_share),
)
char = char.with_composition(data)
```

- **Partial analyses fill in.** A cut takes the types measured for it. The remainder is shared among the unmeasured types in the estimate's proportions, and the row is renormalized. An FIA aromatics number alone therefore keeps the estimated P:N ratio.
- **SARA.** `saturates` is split into paraffins and naphthenes in the estimate's ratio. `resins` and `asphaltenes` count as aromatics, which their cores are. SARA is a mass-basis analysis, and it is used here on the volume basis as given.
- **Differentiable.** The override is a `jnp.where`, so a product property is differentiable with respect to the measured values. With a curve it is also differentiable with respect to the TBP data, through each cut's Tb.

**Sulfur and nitrogen classes are illustrative by default.** `DEFAULT_SULFUR_SPLIT` gives the shape the hydrodesulfurization literature describes:

- naphtha: thiols, sulfides and thiophenes;
- from kerosene: benzothiophenes (benzothiophene boils at 221 °C);
- from diesel: dibenzothiophene (332 °C) and the hindered alkyl-DBTs (4,6-DMDBT at about 365 °C);
- vacuum range: aliphatic sulfides again.

The numbers are not any crude's. `DEFAULT_NITROGEN_SPLIT` puts a quarter to a third of the nitrogen in the basic class, slightly more in the heavy end. Pass `sulfur_split=` and `nitrogen_split=` for real ones, as a `(T_K, shares)` table or a per-cut array.

**Range warnings.** Outside the data a correlation was fitted to, `estimate_composition` raises `CompositionRangeWarning`, which lists the cuts. It does this once per call and only with concrete inputs; under `jit`/`grad` the warning is skipped. The ranges it warns outside are:

| Correlation | Range |
| --- | --- |
| Huang index | 300--850 K |
| Riazi-Daubert 1987 C/H | 300--620 K |
| 2B4.1 types | MW 70--600 |
| n-d-M | MW below 200 |

The residue lump always warns, because its Tb and MW are set rather than correlated. The function still returns a number. A differentiable flowsheet needs one, and the warning tells you it is an extrapolation. `CompositionData(warn=False)` silences it.

**Model equations.** Every correlation in `(Tb, SG)` below has the Riazi-Daubert form `θ = a exp(b T + c SG + d T SG) T^e SG^f`. The constants, with the unit of T, are:

| Quantity | T unit | a | b | c | d | e | f | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Huang index I, Tb ≤ 620 K | °R | 2.2657e-2 | 3.9052e-4 | 2.468316 | -5.70425e-4 | 5.7209e-2 | -0.719895 | [R2] Table X (unverified) |
| Huang index I, heavy | K | 3.2709e-3 | 8.4377e-4 | 4.59487 | -1.0617e-3 | 0.03201 | -2.34887 | [R3] Eq. 2.46 / Table 2.9 (unverified) |
| C/H weight ratio | °R | 17.22022 | 8.24983e-3 | 16.9402 | -6.93931e-3 | -2.72522 | -6.79769 | [R2] Table XI (unverified) |

The two Huang-index forms are blended as `I = (1 - s) I_1987 + s I_heavy`, where `s = σ((Tb - 620 K)/20 K)`. The other equations are:

- `n20 = sqrt((1 + 2I)/(1 - I))`.
- `d20 = SG - 4.5e-3 (2.34 - 1.9 SG)` g/cm³ [R3, API TDB].
- `Ri = n20 - d20/2` (Kurtz-Ward refractivity intercept).
- `m = MW (n20 - 1.475)`.
- Goossens hydrogen [R4]: `H wt% = 30.346 + (82.952 - 65.341 n20)/d20 - 306/MW`.
- `C/H = (1 - H - S - N)/H` (mass fractions).
- 2B4.1 types, CH form [R1, R5]:
  - MW ≤ 200: `x_P = 2.57 - 2.877 SG + 0.02876 C/H` and `x_N = 0.52641 - 0.7494 x_P - 0.021811 m`;
  - MW > 200: `x_P = 1.9842 - 0.27722 Ri - 0.15643 C/H` and `x_N = 0.5977 - 0.761745 Ri + 0.068048 C/H`;
  - in both, `x_A = 1 - x_P - x_N`. The branches are blended with weight `σ((MW - 200)/10)`.
- 2B4.1 VGC form [R1, R5]:
  - MW ≤ 200: `x_P = -13.359 + 14.4591 Ri - 1.41344 VGC` and `x_N = 23.9825 - 23.333 Ri + 0.81517 VGC`;
  - MW > 200: `x_P = 2.5737 + 1.0133 Ri - 3.573 VGC` and `x_N = 2.464 - 3.6701 Ri + 1.96312 VGC`.
- n-d-M [R6, R7], with n and d at 20 °C, S in wt% and M = MW:
  - `v = 2.51 (n - 1.4750) - (d - 0.8510)` and `w = (d - 0.8510) - 1.11 (n - 1.4750)`;
  - `%C_A = 430 v + 3660/M` for v > 0, or `670 v + 3660/M` otherwise;
  - `%C_R = 820 w - 3 S + 10000/M` for w > 0, or `1440 w - 3 S + 10600/M` otherwise;
  - `%C_N = %C_R - %C_A` and `%C_P = 100 - %C_R`;
  - ring counts `R_A = 0.44 + 0.055 M v` (0.080 for v < 0) and `R_T = 1.33 + 0.146 M (w - 0.005 S)` (0.180 for w < 0). The ring counts are unverified.
- Projection onto the simplex: `x_i ← t·logaddexp(x_i/t, 0)` with `t = 1e-3`, then each row is normalized.
- Stream averages: `x_type = Σ φ_i x_type,i`, with standard-volume fractions `φ_i ∝ m_i/SG_i`, and `H = Σ w_i H_i` (likewise S, N and the classes), with mass fractions `w_i`.

**Assumptions.**

- 2B4.1's molecular fractions are used as volume fractions. MNL50's remark that the bases nearly coincide for a narrow cut is the justification, and it is not checked.
- n-d-M carbon types are used as molecule types when `pna_method="ndm"`.
- Oxygen and metals are neglected in `C/H`.
- Light ends are exact paraffins.
- Olefins are zero in a straight run.
- SARA (a mass-basis analysis) is used on the volume basis as given.

**Degrees of freedom.** None. This is a property estimate, not a unit. The choices are the two method switches and the measurements that override the estimate.

**References.**

| Key | Reference | Used for | How checked |
| --- | --- | --- | --- |
| R1 | Riazi, M.R.; Daubert, T.E. "Prediction of molecular-type analysis of petroleum fractions and coal liquids." *Ind. Eng. Chem. Process Des. Dev.* **1986**, 25(4), 1009--1015. doi:10.1021/i200035a027 | 2B4.1 type correlations (CH and VGC forms) | Title, journal, volume, issue, pages and year were confirmed by web search (publisher page title and listing). The **coefficients were not checked against the paper** (unverified). They match pychemqt's independent coding (`lib/petro.py`, `PNA_Riazi`), and its doctest is reproduced. |
| R2 | Riazi, M.R.; Daubert, T.E. "Characterization parameters for petroleum fractions." *Ind. Eng. Chem. Res.* **1987**, 26(4), 755--759. doi:10.1021/ie00064a023 | Huang index and C/H from (Tb, SG) | Citation as already used in `correlations.py`. The DOI and the table numbers (X, XI) come from pychemqt's reference list and were not checked (unverified). The constants are pychemqt's. Pure-compound checks: n20 within 0.004 for n-decane, benzene and toluene; C/H within 0.2 % for n-decane. |
| R3 | Riazi, M.R. *Characterization and Properties of Petroleum Fractions*, ASTM MNL50; ASTM International: West Conshohocken, PA, **2005**. doi:10.1520/MNL50-EB | Heavy Huang index (Eq. 2.46 / Table 2.9); d20 from SG | The DOI and publication details were confirmed by web search (ASTM store listing). Equation and table numbers, and the constants, come from pychemqt (unverified). The heavy form's constants give a physical n20 only with Tb in K, although pychemqt converts to °R; K is coded here. The d20 formula reproduces n-decane's measured d20 to 5e-5. |
| R4 | Goossens, A.G. "Prediction of the hydrogen content of petroleum fractions." *Ind. Eng. Chem. Res.* **1997**, 36(6), 2500--2504. doi:10.1021/ie960772x | Hydrogen content | Volume, pages and DOI come from pychemqt's reference list and were not found by web search (unverified). The equation is pychemqt's coding ("Eq. 3"); its Table 1 doctest for n-decane (15.45 wt%) is reproduced. |
| R5 | API *Technical Data Book -- Petroleum Refining*, procedure 2B4.1 (edition and page unverified) | Same correlations as R1, as the TDB tabulates them | Not checked (unverified). |
| R6 | van Nes, K.; van Westen, H.A. *Aspects of the Constitution of Mineral Oils*; Elsevier: New York, **1951** | n-d-M method | Not checked against the book (unverified). The carbon-type equations agree with pychemqt's `PNA_van_Nes`. |
| R7 | ASTM D3238, *Standard Test Method for Calculation of Carbon Distribution and Structural Group Analysis of Petroleum Oils by the n-d-M Method* (edition unverified) | n-d-M method | Not checked (unverified). |

The sulfur-class split `DEFAULT_SULFUR_SPLIT` and the basic-nitrogen split `DEFAULT_NITROGEN_SPLIT` cite **no source**. They are illustrative numbers chosen for this module. The boiling points quoted beside them (benzothiophene 221 °C, dibenzothiophene 332 °C, 4,6-DMDBT about 365 °C) and the rule of thumb of one quarter to one third basic nitrogen are handbook knowledge that was not checked here (unverified). The fitted ranges in the range-warning table come from pychemqt's input bounds for the same correlations (80--650 °F), from MNL50's "C20--C50" as pychemqt quotes it, or are conservative choices made here. All of them are unverified.

**What is checked** (`tests/refinery/test_composition.py`):

- **Coded constants.** Every exponential correlation's six constants are pinned to the values tabulated above.
- **Published coefficients.**
  - The P and N constants of 2B4.1 are the ones in pychemqt's independent coding of the procedure. The A constants, recalled separately, are exactly one minus their sum, which is what makes the three fractions sum to one for any input.
  - It reproduces pychemqt's doctests for 2B4.1 (`PNA_Riazi`: 0.606/0.275/0.118) and for Goossens (n-decane, 15.45 wt%).
  - The Huang index gives n20 within 0.004 for n-decane, benzene and toluene.
  - The C/H ratio is within 0.5 % for n-decane's formula and 7 % for benzene's.
- **Composition.** On two assays, one with a heavy end and one without, and with both type methods, every cut's PNA sums to 1.
- **Balances.** Hydrogen, sulfur and each sulfur class, nitrogen and each nitrogen class, and the volume-averaged types all balance to 1e-12 through `product_properties`, on both assays.
- **Gradients.** The gradients of product P, N, A and hydrogen with respect to one TBP point and the bulk SG match central differences to 1e-5. The gradient with respect to a measured aromatics value does too.

**Not checked.** Riazi's MNL50 worked examples for the composition correlations (molecular type, hydrogen, n-d-M) are not reproduced. Neither the book nor the papers could be reached (the publishers' sites are blocked here), so no agreement with them is claimed. The coefficient values were cross-checked against pychemqt's coding of the same procedures and against pure-compound data. That is weaker evidence than the source itself, and every such item is marked (unverified) above. The default sulfur and nitrogen splits are illustrative and not fitted to any crude.

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
- with `composition=char.composition`, its hydrocarbon types, hydrogen, sulfur and nitrogen and their classes (`.composition`, a `StreamComposition`; see [Hydrocarbon type, hydrogen and heteroatom classes](#refinery-composition)).

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

- `crude.rate` (bbl/d) is always a lever. So is `preheat.T` (°C), the furnace inlet temperature, except on a `PreheatedCrudeUnit`, where the train computes it (see [the preheat train](#refinery-preheat)).
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

On a unit with a preheat train, `preheat.T` is replaced by what sets it: the tank temperature and the exchangers ([below](#refinery-preheat-planning)). On a bare `CrudeUnit`, the `preheat.T` lever is not dead, but on a column closed by an overflash it moves only the fired duty. The overflash fixes the flash-zone vaporisation, so the preheat temperature changes how much heat the furnace must add and nothing about the products.

**Non-convergence.** Some spec sets have no solution. With 5 % overflash, taking more than about 24 MW out of PA1 on the test column dries out the section above it. The column then reports `converged=False` with a finite state. `cdu_block` returns NaN at such a point, and the planner rejects any proposal at which a block is not finite ([A block that cannot be evaluated](planning.md#a-block-that-cannot-be-evaluated)).

On the test column, a credit on PA1 duty drives the planner past the edge. It proposes 30, 27, 25.5, 24.75 and 24.38 MW (among others), rejects each, and settles at 23.62 MW, which converges. With `mask_nonconverged=False`, the same run ends at 30 MW on a column that did not converge and reports itself as converged.

**Yields as levers, cut points as outputs.** A cut-point target such as "kero TBP95 ≤ 235 °C" is a planner `Spec` on a CDU output, and the LP inverts the delta vector to find the yield that meets it. In the example plan, the kero end point binds at 235.000 °C. The planner trades naphtha yield, which pays \$70/bbl, against kero, which pays \$95/bbl, because a heavier naphtha cut also makes the kero heavier.

Making a cut point a column *spec* would describe the same feasible set, and it would cost more in two ways:

- it would put a piecewise-linear TBP point inside the column's Newton solve;
- or, as an alternative, it would wrap a root find around the column, which means several column solves per evaluation.

For the same reason, `product_value_block` prices products in bbl/d only. Folded into the revenue as a smooth penalty, a quality limit puts curvature in the objective that a linear model cannot see. On the test crude, that version crawled along the penalty's shoulder at a radius of 1e-4 and stopped at the iteration cap.

(refinery-preheat)=
## The preheat train

The crude reaches the furnace at 250 °C or so, from a tank at ambient temperature. Most of that heat is recovered from the column's own products and pumparounds in a train of exchangers. The furnace supplies the rest, so the train sets the fuel bill as much as the column does. `difflow_refinery.preheat` models the train from tank to furnace inlet. It has three kinds of item:

- **Exchangers** (`PreheatExchanger`): `UA = A / (1/U + R_f)`, with the duty from the LMTD equation of `difflow.units.heat_exchanger` (and its F-factor when `shells` is given). An optional `bypass` sends part of the hot stream around the exchanger.
- **A desalter** (`DesalterParams`): wash water at its own temperature is mixed in. The brine leaves at the desalter temperature, and the crude keeps `water_out` of its volume as water. Salt removal is a fixed `efficiency`. The 120-150 °C operating window is reported as two signed margins (`margin_low`, `margin_high`), not imposed.
- **A preflash drum** (`PreflashDrumParams`): an adiabatic flash at a set pressure, or at the pressure that flashes a set `vapor_fraction` of the hydrocarbons. The vapour goes to the column, on the stage above the flash zone by default (`vapor_stage`). The liquid is pumped on through the hot train. Free water is drawn off.

The crude side is a three-phase split on the column's own thermodynamics: hydrocarbon liquid, vapour, and free water. Water is immiscible with the hydrocarbon liquid, as it is on the column's trays. With free water present, water's partial pressure is its vapour pressure and the hydrocarbons see the rest. Without it, all the water is vapour. The two cases agree on the boundary. A dry crude's enthalpy computed this way equals the column's own feed enthalpy to round-off, which is what lets the energy balance close across the train and the column. The hot streams are liquid throughout.

The train and the column are coupled both ways:

- The hot streams are the column's products and pumparounds, so their rates and temperatures come from the column.
- The column's feed is the furnace inlet, which is the train's outlet. The drum vapour is a second, vapour feed.
- A pumparound cooled in the train returns at the temperature the train sends it back at.

`PreheatedCrudeUnit` solves the two together:

```python
import difflow_refinery as dr

tp = dr.PreheatTrainParams(
    exchangers=(dr.PreheatExchanger("E1", U=350.0, area=300.0), ..., dr.PreheatExchanger("E8", 350.0, 1200.0, Rf=2e-4)),
    hot_streams=(dr.HotStream("residue", ("E8", "E7", "E2")), dr.HotStream("pa2", ("E6",)), ...),
    crude_path=("E1", "E2", "E3", "desalter", "E4", "E5", "preflash", "E6", "E7", "E8"),
    desalter=dr.DesalterParams(), drum=dr.PreflashDrumParams(P=3e5))
unit = dr.PreheatedCrudeUnit(assay, column_params, tp)
res = unit.solve(95_000, T_tank=300.0)          # bbl/d from the tank
res.furnace_inlet_T, res.fired_duty, res.preflash_vapor
unit.balances(res)                               # tank to products: mass, water, energy
```

Each hot stream lists its exchangers hottest first. The crude path lists everything in the order the crude meets it, tank to furnace. Every exchanger must be on the crude path once and on exactly one hot stream; the train raises a `ValueError` when it is not. A pumparound in the train must have one `pumparound_return_temperature` spec on the column (its value only starts the loop) and one other spec, such as its rate. A pumparound not in the train keeps whatever spec it has.

**The solve.** The train alone is one damped Newton over its exchanger outlet temperatures, the desalter temperature and the drum temperature (and the drum pressure in `vapor_fraction` mode). Around it, an outer Newton works on the tear: the pumparound return temperatures, the drum state and the furnace inlet temperature. Each outer iteration solves the train and the column, and differentiates both with one forward-mode trace. Steps are clipped to 30 K. It starts from a default guess built from the specs, not from a previous solution.

Gradients are implicit-function gradients at the converged point. The last outer Jacobian is reused for the implicit step, so a `jax.grad` or `jax.jacfwd` with respect to an area, an `R_f`, the drum pressure or a TBP point costs one more linear solve, not a differentiated iteration history.

On the test crude (95 000 bbl/d from a 27 °C tank), with the column of [the crude unit](#refinery-crude-unit) and an eight-exchanger textbook layout (`tests/refinery/reference/preheat_case.py`, `examples/37_crude_preheat_train.ipynb`):

- the solve converges in 4 outer iterations;
- the train recovers 81.7 MW and delivers the crude to the furnace at 251.2 °C;
- the furnace fires 46.5 MW;
- the desalter runs at 140.8 °C;
- the drum flashes 8.9 mol % of the hydrocarbons at 150 °C and 3 bar;
- the mass, water and energy balances close to 1.5e-12.

A heavier invented crude (every TBP point 25 °C higher above 10 %, SG 0.885) converges from the same default start. It reaches the furnace at 272.9 °C with 48.8 MW fired. Its desalter sits at 154 °C, 4 K above the window, which the margin reports.

### Fouling and cleaning

`difflow_refinery.preheat.fouling` has the Ebert-Panchal (1995) threshold model. Fouling grows by deposition, which is Arrhenius in the crude-side film temperature and falls with Reynolds number. It shrinks by removal, which goes with the wall shear stress:

    dR_f/dt = alpha Re^beta Pr^(-0.33) exp(-E / (R T_film)) - gamma tau_w

Below the threshold an exchanger does not foul. `fouling_rates(result.train, train_params)` evaluates it for every exchanger of a solved train. The crude side's Re, Pr and wall shear are inputs, because the train carries no geometry beyond the area.

**The default constants (`EbertPanchal()`) are illustrative.** They are not fitted to any crude. They were chosen so that the hot end fouls at a few 1e-4 m²K/W a year and the cold end not at all, which is the right order for a crude train. Fit `alpha`, `E` and `gamma` to your own monitoring data before reading a cleaning date off them. On the test train they give E8 4.1e-4 and E7 2.0e-4 m²K/W a year, and zero for E1-E3.

Two methods turn a fouled train into a decision:

- `unit.fouling_sensitivity(rate, T_tank, train=...)` gives `d(fired duty)/d(R_f)` for every exchanger from one reverse-mode gradient.
- `unit.cleaning_ranking(rate, T_tank, train=...)` multiplies each sensitivity by its `R_f` (the linear estimate of the saving from cleaning). It then re-solves with each exchanger clean (the exact saving) and sorts by the exact saving.

After 18 months of the illustrative fouling, the fired duty has risen from 46.49 to 47.04 MW. The ranking is E8 (0.26 MW), E7 (0.18) and E6 (0.15). The linear estimates are within 10 % of the exact savings and in the same order.

The sensitivities alone tell a different story. Per unit of `R_f`, E6 costs the most (1.2 MW per 1e-3 m²K/W), and E2 at the cold end costs as much as E7. The hot end tops the ranking only because only the hot end fouls. Keeping the two apart is the point: the sensitivity says where fouling hurts, and the fouling model says where it happens.

(refinery-preheat-planning)=
### Planning with the preheat train

`cdu_block` accepts a `PreheatedCrudeUnit` (or a `CrudeUnitWithPreheat`). The base point is `rate=` and `T=` (the tank temperature); `P` is not needed. `preheat.T` is no longer a lever, because the train computes it. A pumparound cooled in the train has no `return_T` lever, because the train sets its return temperature. In their place the train offers these levers:

| Lever | Units | |
|---|---|---|
| `tank.T` | °C | |
| `<E>.Rf` | m²K/kW | the fouling resistance, so a delta vector reads per 1e-3 m²K/W |
| `<E>.area` | m² | |
| `<E>.bypass` | - | hot-stream fraction |
| `desalter.wash` | - | wash water, standard-volume fraction of the crude |
| `preflash.P` | bar | or `preflash.vapor_fraction` in that mode |

The train adds these outputs: `furnace.inlet_T` (°C) and `preheat.recovered` (MW); per exchanger `<E>.duty` (MW) and `<E>.approach` (K, negative for a temperature cross); `<source>.train_out_T` (°C) for each hot stream; `desalter.T` with its two margins; and `preflash.T`, `.P` and `.vapor_fraction`. The `furnace.fired` row of the delta vectors against the `<E>.Rf` levers is the fouling sensitivity above.

### Preheat train gotchas

- **A pinched exchanger may not solve.** The duty is `UA F LMTD`. When a small hot stream meets a large exchanger, its hot-side NTU is very large and the terminal difference at the cold end falls like `exp(-NTU)`. The LMTD of `difflow.units.heat_exchanger` floors each terminal difference at `MIN_DELTA_T` (1e-6 K), so below that floor the duty equation stops responding to the outlet temperature, and the Newton iteration fails. The case found while building the tests was 0.25 mol/s of naphtha against 40 m², where the NTU is about 240. Size the exchanger to the stream, or bypass most of it.
- **A temperature cross is not prevented.** The LMTD takes the absolute value of each terminal difference, so a solution with a cross is not a physical exchanger. `approach` reports it as negative. Check it, or hold it with a spec when planning.
- **The desalter window is a margin, not a constraint.** The heavy test crude runs its desalter outside the window, and the solve does not stop it.
- **The drum's vapour goes into the column, not past it.** It enters at `vapor_stage`, by default the stage above the flash zone, so the column's specs see it.

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

### Desalter

The desalter alone, for a flowsheet. It is built from `DesalterUnitParams` (`assay`, `desalter`, `cut_points`, `method`) and called with a wet crude stream. It returns `("crude", "brine")`. `desalter.feed(rate, T, P, water=0.002)` makes an inlet, and `desalter.solve(feed)` also returns the temperature and the window margins.

### PreflashDrum

The preflash drum alone. It is built from `PreflashDrumUnitParams` (`assay`, `drum`, ...) and returns `("vapor", "liquid", "water")`. With `drum.vapor_fraction` set, the pressure is solved for. `feed(rate, T, P, water=0.0)` makes an inlet at the train's pressure.

### CrudeUnitWithPreheat

The whole coupled unit (`PreheatedCrudeUnit`) as a flowsheet operation. It is built from `CrudeUnitWithPreheatParams` (`assay`, `column`, `train`, ...) and called with the tank crude (`op.feed(95_000, T=300.0)`). Its outlets are the column's products, then `"brine"` (with a desalter) and `"drum_water"` (with a drum). A product cooled in the train leaves at the train's outlet temperature, not the column's. `op.last_result` holds the full `PreheatedUnitResult`. The nested train params (exchangers, hot streams, desalter, drum) are plain dataclasses, so the unit round-trips through `difflow.serialize`.

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
D1160 (as opposed to TBP) product curves. The cross-check against an
independent simulator is against an equation-oriented re-implementation in
Pyomo/IPOPT on the same characterized feed, not against DWSIM; see
[Validation: the vacuum unit](#refinery-vacuum-validation) for what that does
and does not establish.

---

(refinery-fcc)=
## The fluid catalytic cracker

`difflow_refinery.fcc` (issue #308) is the gasoline-oriented refinery's VGO conversion unit: a **riser** with lumped cracking kinetics and catalyst deactivation, a **regenerator** that burns the coke, the two solved **together** as the unit's heat balance, and a **main fractionator** (simplified, see below) that splits the effluent into dry gas, C3, C4, gasoline, light cycle oil (LCO) and slurry. Like the blending pool it is a library, not a palette operation: nothing new is registered with the editor.

```python
import difflow_refinery as dr
from difflow_refinery.fcc import FCCFeed, FCCParams, FCCUnit, fcc_block
from difflow_refinery.vacuum import light_crude

char = dr.characterize(light_crude().to_assay(heavy_end=dr.HeavyEnd()))
feed = FCCFeed.from_characterization(char, rate=50.0,             # kg/s
                                     T_lo=616.15, T_hi=823.15)    # 343-550 C VGO
unit = FCCUnit(FCCParams(riser_outlet_T=793.15, feed_T=500.0))
res = unit.solve(feed)
res["outputs"]["conversion"], res["outputs"]["cat_oil"], res["outputs"]["regenerator_T"]
# (0.747, 5.82, 1003.9 K) -- with the ILLUSTRATIVE default constants
res["balances"]          # mass, C, H, S, N, energy: relative errors ~1e-16
```

The feed can also be a VDU outlet stream: `FCCFeed.from_stream(hvgo, vdu.params.components)`, or `FCCUnit(params, components=...)` called on the stream directly (it then returns the eight outlet streams and the result dict).

**What is physics and what is fitted.** The heat balance -- energy and mass conservation across riser and regenerator, coke combustion stoichiometry, ideal-gas flue-gas enthalpies -- is transferable. **Yields are not**: every published lumped parameter set is for one feed and one catalyst, and the defaults shipped here are not even that (below). Out of the box the yields are illustrative; the unit is predictive only after the kinetic constants and `activity` are fitted to the user's test-run data.

(refinery-fcc-model)=
### FCC model

**Riser** (`difflow_refinery.fcc.riser`). One-dimensional adiabatic plug flow in height `z`, integrated with `diffrax` (Tsit5, constant step, `DirectAdjoint`):

$$
\frac{dt_c}{dz} = \frac{s}{u_g},\qquad
\frac{dy}{dz} = \frac{C}{O}\,\frac{dt_c}{dz}\; a\, f_\text{feed}\; r(y, T, \phi),\qquad
u_g = \frac{n_\text{vap} R T}{P A},\qquad
T = T_\text{mix} - \frac{F_o\,\Delta H_c\,(1-y_\text{go})}{C_{p,\text{tot}}}
$$

- `y`: lump mass fractions (feed basis); `t_c`: catalyst residence time; `s`: slip factor; `a`: catalyst activity; `f_feed = exp(k_K (K_w - K_ref)) / (1 + N_basic/N_0)`: feed crackability and basic-nitrogen poisoning (illustrative forms).
- `dy/dz` is the catalyst-holdup formulation (`F_o dy = r F_c dt_c`), so rate constants are in **1/s per unit catalyst-to-oil ratio**.
- Rate law, all schemes: gas-oil cracking **second order** in the gas-oil mass fraction, every other reaction **first order** [F1]; Arrhenius about 500 °C, `k = k_ref exp(-Ea/R (1/T - 1/773.15))`.
- Energy: one constant vapour heat capacity for every hydrocarbon lump, and the heat of cracking `ΔH_c` (J per kg gas oil converted, endothermic) charged to every product including coke. Then the riser enthalpy is invariant along `z` and `T` is algebraic in the gas-oil fraction (above).
- `T_mix`: adiabatic mixing at the riser base of regenerated catalyst at `T_rg`, liquid feed at the preheat temperature (vaporising with latent heat `λ` at that temperature) and riser steam.
- Feed effects at the riser base, taken out of the gas oil before it cracks (all illustrative): additive coke `ccr_to_coke × CCR`; contaminant coke `metals_coke × (Ni+V)` and contaminant H2 `metals_h2 × (Ni+V)`.
- Deactivation: `phi = exp(-alpha t_c)` (time on stream, [F5]) or `phi = exp(-alpha C_c)` (coke on catalyst). `voorhies_coke(t_c, A, n) = A t_c^n` [F6] is provided as a diagnostic; the riser's coke is a kinetic lump.

| Scheme | Lumps | Reactions | Source |
| --- | --- | --- | --- |
| `weekman_nace_3` | gas oil, gasoline, gas+coke | GO→G, GO→C, G→C | [F1] |
| `lee_4` | gas oil, gasoline, gas (C1-C4), coke | GO→G, GO→gas, GO→coke, G→gas, G→coke | [F2] |
| `ancheyta_5` (default) | gas oil, gasoline, LPG (C3-C4), dry gas (H2, C1-C2), coke | GO→G, GO→LPG, GO→DG, GO→coke, G→LPG, G→DG, G→coke | [F3] |
| `jacob_10` | -- | -- | [F4]; **not implemented** |

The 3- and 4-lump schemes' combined lumps are mapped onto products by split parameters (`coke_share`, `dry_gas_share`).

**Regenerator** (`difflow_refinery.fcc.regenerator`). A well-mixed bed at `T_rg` burns coke of composition C/H/S/N (coke H `coke_hydrogen`; S at `coke_sulfur_factor` × feed S; `n_to_coke` of the feed N) to CO2, CO, H2O, SO2 and N2. CO/CO2 is specified (`co_co2`, 0 = full burn) or Arthur's primary-product ratio [F7], `CO/CO2 = 10^3.4 exp(-12400/RT)` (R in cal/mol/K) times `arthur_factor`; afterburn is not modelled, so `"arthur"` is a partial-burn model. Air (dry, 20.95 % O2) follows from the flue-gas O2 spec (`flue_o2`, wet mole fraction) -- or `air_rate` is given and the excess O2 is an output. Enthalpies are ideal-gas, `Hf(298.15) + ∫Cp dT` [F10, F11]; coke's enthalpy of formation is zero (elements), so its heat of combustion follows from its H content. The regenerated catalyst leaves clean.

**Heat balance** (`difflow_refinery.fcc.unit`). Unknowns `C/O` and `T_rg`; equations: riser outlet temperature = ROT spec, and regenerator energy in = out. Newton (`optimistix`) from `C/O = 6`, `T_rg = 980 K`; gradients of every output by the implicit function theorem (`optimistix`'s implicit adjoint, forward and reverse mode). The heat balance can have **more than one steady state** (multiplicity is a known property of FCC heat balances); with a very active catalyst the solve can land on a hot, low-circulation one, and `RegeneratorTemperatureWarning` fires above `regenerator_T_max` (760 °C, illustrative) -- the state is reported, not hidden.

**Products.** Real species for the gases; product pseudocomponents `fcc01`..`fcc22` for the liquids:

- **Dry gas**: hydrogen, methane, ethane, ethylene by a mass split (`dry_gas_split`), plus contaminant H2 and H2S.
- **LPG**: C3 share `c3_share`; propylene share of C3 `propylene_in_c3`; olefin share of C4 `olefins_in_c4`; isobutane share of C4 paraffins; the butenes split over 1-butene, isobutylene, cis- and trans-2-butene (`butene_split`). These are the "olefin split parameters" of the issue; all illustrative.
- **Gasoline** (C5-221 °C) and **cycle oil** (221 °C+, LCO + slurry) lumps are put on a fixed pseudocomponent grid by a fixed TBP distribution per lump (a logistic CDF, `(T50, width)` per lump), gravities from a Watson K per lump, MW from Twu [the plugin's `correlations`].
- Elemental bookkeeping: gasoline and coke carry hydrogen contents (parameters; gasoline H moves with feed H by `gasoline_hydrogen_slope`); sulfur in gasoline, coke and cycle oil at multiples of feed S (cycle oil `1 + cycle_oil_sulfur_slope × X`), **H2S takes the rest**; nitrogen `n_to_coke` to coke, the rest to cycle oil; **cycle-oil hydrogen by difference** (reported as `cycle_oil_hydrogen` -- 8.7 and 7.4 wt% on the two test feeds -- so an implausible value is visible); carbon is mass less H, S, N (feed metals and oxygen are counted as carbon).
- Gasoline RON/MON: `RON = ron_ref + ron_dT (ROT - T_ref) + ron_dX (X - X_ref)`, likewise MON -- fitted forms with illustrative defaults; there is no transferable open correlation. Gasoline PONA is a fixed parameter vector (olefins 27 vol%).
- LCO cetane index: ASTM D4737 (`blending.cetane_index_d4737`) on D86 points from the TBP curve; known to read high for aromatic cracked stocks [F12], so treat it as indicative.

**Main fractionator -- simplified.** The issue asks for a `StageColumn` layout with pumparounds, side strippers and a bottom quench. **That is not what is built.** The fractionator is a smooth TBP split of the product pseudocomponents at two cut points, `gasoline_cut` (221 °C) and `lco_cut` (343 °C): each pseudocomponent goes to the lighter product with fraction `sigmoid((T_cut - Tb)/w)` (`split_width` 6 K), which mimics real product overlap and keeps the cut points differentiable. The gases are split ideally into dry gas, C3 and C4 -- standing in for the gas plant (`difflow_refinery.gasplant`, #312), which reached `main` after this unit was built; routing through it is follow-up work. Mass is conserved exactly; there is no energy model of the fractionator (no condenser or pumparound duties), and the energy balance covers riser and regenerator only.

(refinery-fcc-specs)=
### FCC degrees of freedom and specs

| Spec / lever | Parameter | Notes |
| --- | --- | --- |
| Riser outlet temperature | `riser_outlet_T` (K) | the spec; catalyst circulation follows from the heat balance |
| Feed preheat | `feed_T` (K) | |
| Feed rate | `FCCFeed.mass` (kg/s) | `feed.with_rate(...)`; recycle of HCO/slurry is **not** modelled |
| Regenerator air | `flue_o2` (wet mole fraction) **or** `air_rate` (kg/s) | |
| CO/CO2 | `co_co2` or `"arthur"` (+ `arthur_factor`) | |
| Catalyst activity | `activity` | the parameter to track with `difflow.reconciliation.tracking` |
| Fractionator | `gasoline_cut`, `lco_cut` (K) | TBP cut points |
| Kinetics | `k_ref`, `Ea` (per reaction), `deactivation_alpha` | to be fitted |

Every numeric parameter is traceable: `unit.solve(feed, riser_outlet_T=jnp.asarray(800.0))`, or differentiate through `FCCFeed` fields and through `characterize` (pass `indices=` from `fcc.feed.cut_indices` when the assay itself is traced).

**Outlets** (`FCCUnit.OUTLETS`, difflow streams in mol/s; species are `difflow.database` names):

| Outlet | Species | Consumer |
| --- | --- | --- |
| `dry_gas` | `hydrogen`, `methane`, `ethane`, `ethylene`, `hydrogen_sulfide` (+ feed species not on the feed's property table, passed through) | fuel gas / amine |
| `c3` | `propane`, `propylene` | alkylation (#310), polymer-grade propylene |
| `c4` | `isobutane`, `n_butane`, `1_butene`, `isobutylene`, `cis_2_butene`, `trans_2_butene` | **alkylation (#310)**: C4= olefins and isobutane |
| `gasoline` | `fcc01`..`fcc22` | `BlendPool` (`BlendComponent` from properties: SG, RON, MON, S, olefins) |
| `lco` | `fcc01`..`fcc22` | diesel hydrotreating |
| `slurry` | `fcc01`..`fcc22` | fuel oil |
| `sour_water` | `water` | |
| `flue_gas` | `nitrogen`, `oxygen`, `carbon_dioxide`, `carbon_monoxide`, `water`, `sulfur_dioxide` | |

`carbon_monoxide` is not in `difflow.database`; it is defined (formula, Cp, Hf) in `fcc.species`.

**Outputs** (`res["outputs"]`): `conversion` (1 - unconverted 221 °C+ lump, coke included), `cat_oil`, `catalyst_circulation`, `regenerator_T`, `regenerator_margin`, `mix_T`, `residence_time`, `activity_out`, `coke_on_catalyst`, `air_rate`, `flue_o2`/`flue_co`/`flue_co2`, `co_co2`, `coke_burn`; `yield.<x>` for `dry_gas`, `lpg`, `c3`, `c4`, `gasoline_lump`, `cycle_oil`, `coke`, `h2s` and the fractionator products `gasoline`, `lco`, `slurry`; per liquid product `<p>.SG`, `.API`, `.sulfur`, `.hydrogen`, `.nitrogen`, `.tbp10/50/90`; `gasoline.RON`, `.MON`, `.olefins`, ...; `lco.cetane_index`; `c3_olefins`, `c4_olefins`, `cycle_oil_hydrogen`. `unit.profile(feed, res["x"])` gives the riser profiles.

**Planning.** `fcc_block(unit, feed, levers=["riser_outlet_T", "feed_T", "feed.ccr"], outputs=[...])` is a `difflow.planning.Block`, like `cdu_block`: its Jacobian carries the heat-balance coupling a fixed yield vector misses. Levers are numeric `FCCParams` fields and `feed.<field>`; outputs are `res["outputs"]` keys, in SI units; a non-converged solve returns NaN.

(refinery-fcc-validation)=
### FCC: what is tested, and what is not

Tested (`tests/refinery/test_fcc.py`):

- Converges from the default initialisation on the light and heavy synthetic VGOs (343-550 °C cuts of `vacuum.light_crude` / `heavy_crude`), heat balance closed (`|residual| < 1e-10`).
- Mass, C, H, S, N and energy (riser + regenerator) balances close to 1e-8 relative -- in practice to ~1e-16 -- for both feeds and for every scheme/deactivation/burn option.
- `jax.jacfwd` of conversion, gasoline yield, coke yield and regenerator T with respect to ROT, feed preheat and one VGO TBP point (450 °C, through `characterize`) matches central differences to 1e-5 relative (observed ~1e-7); reverse mode matches forward.
- Conversion rises with ROT; the gasoline lump passes through an interior maximum along a ROT sweep (overcracking); higher feed CCR raises the regenerator temperature at fixed ROT; the riser solution is converged in step size (the default 200 steps and 800 steps agree to 1e-8).
- Pinned constants: Arthur's as quoted, heats of formation, IUPAC atomic weights, N2/CO2 Cp equal to `difflow.database`'s.

**Not done** (stated plainly):

- **Literature cross-check: not done.** Reproducing the published steady state of Arbel et al. (1995) [F8] or McFarlane et al. (1993) Model IV [F9], or Ancheyta et al.'s (1999) predicted yields [F3], needs the papers' parameter tables and reported results. The papers could not be accessed for this implementation (network access to publishers was blocked), and no numbers from them are reproduced or claimed.
- **Default rate constants are not from any paper.** `ILLUSTRATIVE_5LUMP` was chosen to give commonly quoted VGO FCC yield ranges; the 3- and 4-lump defaults aggregate it. The networks follow the papers' descriptions; the 5-lump reaction set follows the abstract of [F3] (seven reaction constants plus deactivation, "eight kinetic constants") and was not checked against the paper's scheme figure.
- **Jacob et al.'s 10-lump** composition-aware scheme [F4] is not implemented (`get_scheme("jacob_10")` raises). #305's `Composition` is read only for feed hydrogen.
- **Main fractionator on `StageColumn`**: not built; the simplified split above stands in for it.
- **Riser hydrodynamics**: a constant slip factor (default 2, illustrative). The Han & Chung (2001) [F13] parameters named in the issue were not checked and are not used.
- **HCO/slurry recycle**, stripper (entrained hydrocarbons in coke), carbon on regenerated catalyst, NOx/NH3/HCN, pressure drop along the riser, and the gas plant: not modelled here (`difflow_refinery.gasplant`, #312, reached `main` after this unit was built; routing `c3`/`c4` through it is follow-up work).
- **Example notebook** (VDU → FCC → gas plant and gasoline pool): not written.
- No catalyst-vendor or licensor yield model: proprietary, out of reach by design.

**Performance.** The first solve of a configuration (scheme, deactivation kind, burn and air modes) compiles in about 7 s; later solves take ~20-30 ms. A Jacobian through `characterize` and the unit compiles in about 30 s.

(refinery-fcc-references)=
### FCC references

"Verified" below means the bibliographic data (authors, title, journal, volume, pages, year) were confirmed by web search; the papers' **contents** (equations, tables, constants) could not be read for this implementation, so no equation or table number is claimed from them unless stated.

| Key | Reference | Used for | How checked |
| --- | --- | --- | --- |
| F1 | Weekman, V.W., Jr.; Nace, D.M. "Kinetics of catalytic cracking selectivity in fixed, moving, and fluid bed reactors." *AIChE J.* **1970**, 16(3), 397--404. | 3-lump network; gas oil second order, gasoline first order | Citation verified (web search). Rate-law orders are the well-known form of this model, not re-read from the paper (unverified). No constants used. |
| F2 | Lee, L.-S.; Chen, Y.-W.; Huang, T.-N.; Pan, W.-Y. "Four-lump kinetic model for fluid catalytic cracking process." *Can. J. Chem. Eng.* **1989**, 67, 615--619. | 4-lump network | The 4-lump network (Weekman's gas+coke lump split into gas and coke) confirmed by web search of citing papers; **journal, volume, pages unverified**. No constants used. |
| F3 | Ancheyta-Juárez, J.; López-Isunza, F.; Aguilar-Rodríguez, E. "5-Lump kinetic model for gas oil catalytic cracking." *Appl. Catal. A: General* **1999**, 177(2), 227--235. | 5-lump network | Citation and abstract verified (web search: eight kinetic constants including deactivation; LPG C3-C4 and dry gas C2- lumps; MAT at 480/500/520 °C on one VGO and one equilibrium catalyst). Reaction set as coded unverified against the paper's figure. **No constants used; predicted yields not reproduced.** |
| F4 | Jacob, S.M.; Gross, B.; Voltz, S.E.; Weekman, V.W., Jr. "A lumping and reaction scheme for catalytic cracking." *AIChE J.* **1976**, 22(4), 701--713. | (10-lump scheme, not implemented) | Authors, title, journal, volume, first page verified; end page unverified. |
| F5 | Weekman, V.W., Jr. "A model of catalytic cracking conversion in fixed, moving, and fluid-bed reactors." *Ind. Eng. Chem. Process Des. Dev.* **1968**, 7(1), 90--95. | Exponential time-on-stream deactivation; C/O × t_c formulation | **Unverified** (web search did not return the paper); the exponential decay form is as commonly attributed to it. |
| F6 | Voorhies, A., Jr. "Carbon formation in catalytic cracking." *Ind. Eng. Chem.* **1945**, 37(4), 318--322. | `voorhies_coke` C = A t^n | Citation verified (web search). |
| F7 | Arthur, J.R. "Reactions between carbon and oxygen." *Trans. Faraday Soc.* **1951**, 47, 164--178. doi:10.1039/TF9514700164 | CO/CO2 ratio form | Citation verified (RSC listing; DOI from the RSC article URL). **Constants 10^3.4 and 12400 cal/mol are as quoted in the FCC literature, unverified against the paper.** |
| F8 | Arbel, A.; Huang, Z.; Rinard, I.H.; Shinnar, R.; Sapre, A.V. "Dynamic and control of fluidized catalytic crackers. 1. Modeling of the current generation of FCC's." *Ind. Eng. Chem. Res.* **1995**, 34(4), 1228--1243. | (cross-check target, not done) | Citation verified (web search). |
| F9 | McFarlane, R.C.; Reineman, R.C.; Bartee, J.F.; Georgakis, C. "Dynamic simulator for a model IV fluid catalytic cracking unit." *Comput. Chem. Eng.* **1993**, 17(3), 275--300. | (cross-check target, not done) | Citation verified (web search). |
| F10 | Reid, R.C.; Prausnitz, J.M.; Poling, B.E. *The Properties of Gases and Liquids*, 4th ed.; McGraw-Hill: New York, **1987**; Appendix A. | Ideal-gas Cp of N2, O2, CO2, CO, H2O, SO2 | N2 and CO2 equal the `difflow.database` entries (same source; pinned by test). **O2, CO, H2O, SO2 transcribed for this module, unverified against the book.** |
| F11 | Cox, J.D.; Wagman, D.D.; Medvedev, V.A. *CODATA Key Values for Thermodynamics*; Hemisphere: New York, **1989**. CO: Chase, M.W. *NIST-JANAF Thermochemical Tables*, 4th ed., *J. Phys. Chem. Ref. Data* Monograph 9, **1998**. | Hf(298.15) of CO2 (-393.51), H2O(g) (-241.826), SO2 (-296.81), CO (-110.53) kJ/mol | Values are the standard tabulated ones, from memory of the tables (unverified against the printed tables in this session); pinned by test. |
| F12 | ASTM D4737, *Standard Test Method for Calculated Cetane Index by Four Variable Equation* (edition unverified). | LCO cetane index | The plugin's existing `cetane_index_d4737`. |
| F13 | Han, I.-S.; Chung, C.-B. "Dynamic modeling and simulation of a fluidized catalytic cracking process. Part I: Process modeling." *Chem. Eng. Sci.* **2001**, 56(5), 1951--1971. | (riser slip parameters, not used) | Citation verified (web search). |
| F14 | Sadeghbeigi, R. *Fluid Catalytic Cracking Handbook*, 3rd ed.; Butterworth-Heinemann: Oxford, **2012**. | Orders of magnitude for the illustrative defaults (yield ranges, coke H 6-8 wt%, H2S share of feed S, CCR to coke) | **Not checked in this session (unverified)**; no number is attributed to a specific page. |
| F15 | IUPAC CIAAW, standard atomic weights (abridged/conventional values), Prohaska, T. et al. *Pure Appl. Chem.* **2022**, 94(5), 573--600. | Atomic weights C 12.011, H 1.008, N 14.007, O 15.999, S 32.06 | Values standard; page range unverified. |

**Illustrative parameters** (no source claimed for any number; each is a plausible order of magnitude to be fitted): all `k_ref` and `Ea`, `activity`, deactivation constants, `kw_sensitivity`, `nitrogen_poisoning`, `basic_nitrogen_fraction`, `ccr_to_coke` (0.6), `metals_coke`, `metals_h2`, all gas splits, product H/S/N factors and gradients, the lump TBP distributions and Watson Ks, `heat_of_cracking` (350 kJ/kg), `latent_heat` (250 kJ/kg), `cp_vapor` (3.0 kJ/kg/K), `cp_catalyst` (1.15 kJ/kg/K), `cp_steam` (2.1 kJ/kg/K), riser geometry, slip, steam ratio, the octane forms and the gasoline PONA, and `regenerator_T_max`.
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

### From crude-unit products to a gas-plant feed

The crude unit's products are on the crude's whole characterization
(`F_<light end>`, `F_pc01` ... `F_pcNN`, `F_water`); the gas plant works
on a `GasComponents` table holding only the light ends and the cuts the
naphtha carries. `gas_plant_feed` (#326) is the bridge:

```python
from difflow_refinery.gasplant import gas_plant_feed
gp = gas_plant_feed(cdu.products, char,
                    ["hydrogen_sulfide", "ethane", "propane", "isobutane",
                     "n_butane", "isopentane", "n_pentane"],
                    min_fraction=1e-3, h2s={"offgas": 0.02}, T=313.15, P=1.3e5)
comps, offgas, naphtha = gp.components, gp["offgas"], gp["naphtha"]
print(gp.summary())
```

It does three things, in this order:

1. **Selects the cuts.** Cut $i$ is kept when
   $F_i^{\mathrm{basis}} > f_{\min} \sum_j F_j^{\mathrm{basis}}$, the sum
   over every component of the basis stream (the naphtha by default) as the
   crude unit reports it, water included. The selection is a static choice:
   it is made on the concrete flows, or given as `cuts=` (required under
   `jax.jit`).
2. **Folds the rest.** Every cut not kept, in every converted stream, goes
   into the heaviest kept cut $d$: $F_d \leftarrow F_d + \sum_{i\ \mathrm{folded}} F_i$.
   Moles are conserved exactly. Mass is not: the fold changes it by
   $\sum_i F_i (M_d - M_i)$ (`fold_mass_change`, negative since the folded
   cuts are the heavy ones). `mass_change` is the whole difference between
   the stream on the gas-plant table and its water-free mass on the
   characterization, so it also holds the small difference between a light
   end's database molar mass and the characterization's.
3. **Drops the water** (and anything else in `drop=`), reported as
   `dropped_mol` / `dropped_kg`.

A light end in a stream that is neither in `light` nor in `drop=` raises,
rather than being lost. `folded_fraction` is the folded moles of all
streams over the basis stream's total (the figure example 38 prints);
`folded_fraction_of(name)` is per stream. The result is differentiable in
the stream flows (and the characterization's arrays): folding is a sum.

**H2S.** The crude unit makes none; the assay's sulfur stays on the cuts.
Two explicit ways to add it:

- `h2s={"offgas": r}`: an assumption, $r$ mol of H2S per mol of the
  water-free stream (examples 38 and 40 use $r = 0.02$).
- `h2s_flow={"offgas": evolved_h2s(cdu.products, char, fraction)}`: a flow
  from a sulfur balance,
  $F_{\mathrm{H_2S}} = \phi \sum_{\mathrm{streams}} \sum_i F_i M_i S_i / M_S$,
  with $S_i$ the characterization's sulfur mass fraction (an assay with
  `sulfur_wt`) and $M_S = 32.065$ g/mol. The fraction $\phi$ that evolves as
  H2S in the furnace and column depends on the crude and the severity; no
  value is sourced here, so it has no default, and any value is illustrative.
  The cuts' sulfur is not reduced, so that sulfur is counted twice (as gas and
  on the cuts); subtract it in a sulfur balance.

Tests: `tests/refinery/test_gasplant_feed.py` (against example 38's former
hand code, to 1e-12, and the folded-mass report).

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

(refinery-isomerization)=
## C5/C6 light naphtha isomerization

The isomerization unit (`difflow_refinery.isomerization`, #311) raises the
octane of a light straight-run naphtha. It does so by rearranging the
normal pentane and hexanes into their branched isomers, and saturating the
benzene on the way. The products are an isomerate for the gasoline pool,
an off-gas, and (with a DIH) a side draw sent back to the reactor:

```text
                      H2 make-up
                          |
fresh feed -> [DIP] -> reactor -> separator -> stabilizer -> [DIH] -> isomerate
                ^ iC5 round it      | H2         | C3-        | side draw (MP, nC6)
                +-> isomerate       +-> off-gas  +-> off-gas  +-> back to the reactor
```

```python
from difflow_refinery.isomerization import (
    IsomerizationUnit, IsomerizationUnitParams, IsomerizationReactorParams, constructed_feed)

feed = constructed_feed("paraffinic", 10.0)        # kg/s; the speciation is ASSUMED
unit = IsomerizationUnit(IsomerizationUnitParams(
    configuration="dih", T_in=413.15, H2_HC=0.3,
    reactor=IsomerizationReactorParams(LHSV=2.0), dih_side_draw=6.0, stabilizer_rvp=90e3))
isomerate, offgas, info = unit(feed)
info["outputs"]["RON"], info["outputs"]["dih_duty"], info["loop"]["iterations"]
```

Four configurations (`CONFIGURATIONS`) are built from the same pieces:

- `once_through`: the reactor, a product separator and a stabilizer.
- `dip`: a deisopentanizer ahead of the reactor sends the feed's isopentane (and butanes) round it.
- `dih`: a deisohexanizer after the stabilizer. Its overhead (the dimethylbutanes and the C5s) and bottoms (naphthenes and C7+) are isomerate. Its side draw (the methylpentanes and n-hexane, the low-octane C6s) goes back to the reactor inlet.
- `dip_dih`: both.

The specs are the reactor inlet temperature `T_in`, the reactor pressure,
`H2_HC`, `LHSV`, the configuration, the DIH side-draw rate
(`dih_side_draw`, kg/s) and the stabilizer RVP (`stabilizer_rvp`). The
DIH and DIP also take their purity specs and tray counts.

### Thermochemistry and feeds

The reactor carries sixteen species (`thermochem.SPECIES`): hydrogen,
ethane to the butanes, both pentanes, the five C6 paraffins, MCP,
cyclohexane, benzene and an inert C7+ lump. Every equilibrium constant
follows from the species' ideal-gas heats of formation, absolute
entropies and Cp:

- dHf from Prosen and Rossini (the API Project 44 values, as the NIST WebBook gives them);
- entropies from Yaws;
- Cp cubics fitted here to the NIST WebBook gas tables.

The module docstring lists the sources one by one. **No test yet compares
the free energies derived here with a tabulated set.** It matters: 0.5
kJ/mol in one isomer moves its equilibrium share by about 15 % at 420 K.

`equilibrium_table(T)` and `family_equilibrium(family, T)` give the
closed-form isomer equilibrium. The shares within the C6 paraffins are:

| T (C) | nC6 | 2MP | 3MP | 2,3-DMB | 2,2-DMB |
| --- | --- | --- | --- | --- | --- |
| 120 | 0.067 | 0.223 | 0.128 | 0.109 | 0.473 |
| 160 | 0.092 | 0.253 | 0.156 | 0.111 | 0.388 |
| 200 | 0.117 | 0.273 | 0.178 | 0.110 | 0.322 |
| 240 | 0.141 | 0.287 | 0.195 | 0.107 | 0.270 |

The isopentane share falls from 0.86 to 0.78 over the same range. The
branched isomers are favoured cold, which is why the catalysts that run
coldest make the best isomerate.

The octanes are the pure-hydrocarbon RON and MON of API Research Project
45 (ASTM STP 225). **They were recalled, not checked against the printed
tables, and are marked verify.** Benzene's MON and the C7+ lump's octanes
are assumptions. The isomerate's octane is the Ethyl RT-70 blend of the
species, as in the blend pool.

**The feed's speciation is constructed, not measured.** A TBP assay does
not say which C6 is n-hexane and which is 2,2-DMB. `constructed_feed` and
`light_naphtha_from_cdu` split the light-naphtha pseudo-components by an
assumed composition (`NaphthaSpeciation`). Two are provided:

- `PARAFFINIC`: about 1.5 wt % benzene in the C6 cut.
- `BENZENE_RICH`: a naphthenic crude's, about 5 wt % benzene and more MCP and cyclohexane.

Their numbers are in the range of the light straight-run analyses quoted in
refining texts; verify against a real PIONA before relying on them.

(refinery-isomerizationreactor)=
### IsomerizationReactor

A pseudo-homogeneous, adiabatic plug-flow bed. Every reversible reaction
`A (+ n H2) <=> B` runs at a first-order approach-to-equilibrium rate:

```text
r_j = theta k_j(T) (F_A - F_B / (K_j(T) p_H2^n)),   theta = 1/LHSV
```

The rate is zero at equilibrium whatever `k` is. So the equilibrium is set
by the thermochemistry alone, and the rate constants only set how close to
it the bed gets. The reactions are:

- `nC5 <=> iC5`;
- `nC6 <=> 2MP`, `2MP <=> 3MP`, `2MP <=> 23DMB`, `23DMB <=> 22DMB` (the slow step);
- `MCP <=> CH`;
- benzene saturation, `Bz + 3 H2 <=> CH`;
- ring opening, `MCP + H2 <=> 2MP`;
- hydrocracking to ethane and propane, irreversible.

**The rate constants are illustrative.** The catalyst presets (`CATALYSTS`:
chlorided alumina, sulfated zirconia, zeolite) give orders of magnitude and
each catalyst's temperature window. They are not fitted to any catalyst's
data. `k_scale` is the one factor to calibrate against a plant's measured
approach to equilibrium.

`info["approach"]` reports the approach to equilibrium of each reaction.
In an adiabatic bed it can exceed one: benzene saturation reaches about
1.016 on the benzene-rich feed. The saturation is fast and runs near
equilibrium at the hot outlet. The approach is measured against the
equilibrium at the outlet temperature, which the bed is still heating
towards.

The temperature is not integrated. At every point along the bed it is the
root of the energy balance, so the enthalpy is conserved exactly. The bed
is stiff (benzene saturation's rate constant is fifty times 2,2-DMB formation's), so it is integrated by a two-stage L-stable SDIRK. Each stage
is solved by Newton in a `lax.while_loop` to a residual tolerance. That
loop has no reverse-mode rule, so **the reactor is differentiable in
forward mode only** (`jax.jacfwd`, `jax.jvp`). Every block and test here
uses forward mode. `info["stage_residual"]` reports the worst stage
residual, and `IsomerizationConvergenceWarning` fires when it is not small.

Why not `difflow.kinetics`? Its rate laws are mass action in
concentrations with Arrhenius constants, written as data. These rates are
in molar flows against a temperature-dependent `K_eq` from the species'
free energies, with the hydrogen partial pressure in bar. Writing them as
mass action would need `K_eq(T)` as a rate-law term, which the module does
not have.

(refinery-isomerizationunit)=
### IsomerizationUnit

The unit around the reactor:

- **Hydrogen is once-through.** The charge is made up to `H2_HC` with pure hydrogen, and what is left leaves in the off-gas. There is no recycle-gas compressor.
- **The product separator** is one equilibrium stage at `separator_T` (a 1-tray `GasPlantColumn`). The effluent cooler is a specification and its duty is not reported.
- **The stabilizer is a shortcut, not a tray column.** Hydrogen, ethane and propane go overhead, the pentanes and heavier stay in the bottoms. The fraction of the butanes kept is solved so that the bottoms meet `stabilizer_rvp`. A rigorous stabilizer on the gas-plant column was tried in three layouts. None converged reliably over the compositions the DIH recycle produces, so no stabilizer duty is reported.
- **The DIP and DIH** are `GasPlantColumn` splitters: the gas plant's Peng-Robinson MESH model on the sixteen species. The DIH has a side draw at `dih_side_tray`, at the rate `dih_side_draw`.

With a DIH, the recycle is a `difflow.Flowsheet` recycle torn on the side
draw and converged by Anderson acceleration. `IsomerizationUnit.outputs(feed,
T_in, LHSV, x_nc6=None)` returns the output vector (`OUTPUT_NAMES`) and
differentiates the converged loop by the implicit function theorem through
a `jax.custom_jvp`:

```text
dy/du = Y_u + Y_x (I - G_x)^-1 G_u
```

Here `G` is one pass of the loop (recycle in, side draw out) and `Y` is the
outputs, both linearised by forward-mode AD at the solution. The once-through
and DIP configurations have no loop and are differentiated straight through.
`unit.last_solve["recycle"]` keeps the converged side draw, to warm-start
the next solve.

`unit.blend_component(info["outputs"])` returns the isomerate as a
`BlendComponent` for a `BlendPool`.

### Planning with the isomerization unit

`isom_block(unit, feed, levers, outputs=None)` returns a
`difflow.planning.Block`. The levers are `T_in` (C), `LHSV` (1/h) and
`x_nC6`, the fresh feed's n-hexane mole fraction. The outputs are any of
`OUTPUT_NAMES` in planner units, plus `isomerate_V` (m3/h). The block is
not jit-compiled and its AD mode is forward, because the DIH loop is a
Python loop.

`link_isom(isom_blk, pool_blk)` links the isomerate volume to the
`isomerate_V` lever of `BlendPool.as_block`, so the blend component must be
named `"isomerate"`:

```python
from difflow.planning import Network
from difflow_refinery.blending import BlendPool
from difflow_refinery.isomerization import isom_block, link_isom

blk = isom_block(unit, feed, levers=["T_in", "LHSV"])
iso = unit.blend_component(info["outputs"])        # properties at the base point
pool = BlendPool("gasoline").as_block([iso, reformate])   # reformate: another BlendComponent
net = Network([blk, pool], links=link_isom(blk, pool))
```

What crosses the link is the isomerate's volume. A `BlendComponent` has
fixed properties, so the isomerate's octane in the pool is the one at the
linearisation point. Rebuild it from `blend_component` at each new base point.

### Results

Both constructed feeds were run at 10 kg/s, `T_in` 140 C, LHSV 2, H2/HC
0.3 and 30 bar, on the chlorided-alumina preset. Every column converges.
The total and per-carbon-number balances close to 2e-11 or better.

| Configuration | Paraffinic: RON | Yield (vol) | DIH / DIP duty (MW) | Benzene-rich: RON | Yield (vol) | DIH / DIP duty (MW) |
| --- | --- | --- | --- | --- | --- | --- |
| once-through | 82.18 | 0.988 | - | 81.56 | 1.008 | - |
| DIP | 83.06 | 0.999 | - / 6.96 | 81.13 | 1.007 | - / 5.24 |
| DIH | 82.81 | 0.992 | 6.92 / - | 82.79 | 1.004 | 6.26 / - |
| DIP + DIH | 83.47 | | 7.11 / 6.96 | 83.20 | | 6.38 / 5.24 |

The volume yield exceeds one on the benzene-rich feed. Saturating benzene
and adding hydrogen makes a liquid of lower density.

The DIH loop converges in eight (paraffinic) or nine (benzene-rich) Anderson
iterations, one to three minutes on a laptop. A once-through solve takes
about 12 s the first time and 4 s after that.

What the numbers show:

- **The recycle gain is modest.** The DIH adds 0.6 RON on the paraffinic feed and 1.2 on the benzene-rich one. Licensors usually quote a larger gap between once-through and DIH units (verify). These rate constants and constructed feeds are not fitted to any unit, so neither number should be read as a prediction.
- **A DIP can lower the octane.** On the benzene-rich feed, taking the isopentane round the reactor leaves less mass to absorb the benzene exotherm. The bed runs hotter (a 91 K rise, against 70 K once through), and the hotter outlet equilibrium favours the less-branched isomers.
- **RON has a maximum in `T_in`.** Cold, the bed is short of equilibrium; hot, the equilibrium itself is worse. Once-through on the paraffinic feed, RON is 77.0 at 110 C, 82.5 at 150 C and 80.6 at 190 C. On the benzene-rich feed it peaks near 120 C, at 82.2.
- **The benzene-rich feed runs away.** At `T_in` of 160 C and above, the exotherm drives hydrocracking, which is itself exothermic and uses hydrogen. The bed then uses up its hydrogen. `IsomerizationHydrogenWarning` fires when the outlet H2/HC falls below 0.05, before the separator flash fails.
- **The stabilizer spec is not always met.** On the benzene-rich feed with a DIH, the isomerate's RVP is 80 kPa against a 90 kPa spec. The stabilizer keeps every butane and its C5+ alone is below the spec. `stabilizer_c4_recovery` reports this as 1 and `info["stabilizer"]["spec_met"]` as False.

### Isomerization gotchas

- **Forward mode only.** `jax.grad` through the reactor fails on the stage Newton's `while_loop`; use `jax.jacfwd` or `jax.jvp`.
- **Outputs at a spec have zero derivatives.** The isomerate RVP is held at its spec, the once-through H2 make-up does not depend on the reactor, and a stabilizer at its bound has a zero derivative. A finite-difference check of these compares zero with noise.
- **Keep `T_in` in the catalyst's window.** The presets carry their windows (`CATALYSTS[...]["window"]`). Outside them the constants mean nothing, and a hot benzene-rich charge runs away.

### Isomerization: out of scope

C4 isomerization, catalyst chloriding and its HCl/caustic scrubbing,
molecular-sieve (Ipsorb, TIP) separations, the recycle-gas loop, and
dynamics.

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
- **`from_stream(name, stream, characterization, **overrides)`**: a difflow stream of pseudocomponent molar flows on a shared `BlendCharacterization`. The composition gives `SG`, sulfur, nitrogen, PNA and a Raoult RVP, and flash, freeze and smoke points, viscosity and straight-run RON/MON are estimated (see {ref}`refinery-blend-estimates` below). Overrides (a reformer's reported RON, a measured flash point) take precedence, and an overridden property is not estimated. This mode adds:
  - the product **stream**, with an exact mass and volume balance;
  - the properties only composition can give: distillation and cetane index;
  - the Raoult RVP of the blend itself.

One pool takes one mode. Mixing the two raises an error rather than producing a product stream that is missing some components' mass.

**Volume basis.** Volumes are ideal-mixing volumes at 15 °C from `SG` (`rho = SG * 999.10 kg/m^3`). The product volume is the sum of the component volumes, and the product `SG` is the volume average. Both are tested to round-off against the product stream's own composition.

Recipes can be given as `basis="volume_fraction"` (the default), `"volume_flow"`, or `"split"`. A split is the fraction of each component stream's available volume sent to the pool, which is the natural lever when the pool sits in a flowsheet.

(refinery-blend-estimates)=
### Estimated product properties

`difflow_refinery.properties` (#330) estimates the properties a stream does not carry, and `from_stream` uses them whenever no measured value is given. `estimate=True` (default) estimates every property the characterization has the inputs for, `estimate=False` none (the pre-#330 behaviour), and a list names the ones wanted.

```python
from difflow_refinery import BlendComponent, BlendCharacterization, properties

bc = BlendCharacterization.from_characterization(char)        # char built with composition=True
jet = BlendComponent.from_stream("jet", jet_stream, bc)        # flash, freeze, smoke, viscosity estimated
fo = BlendComponent.from_stream("fuel oil", residue, bc, viscosity_T_C=50.0)
lsr = BlendComponent.from_stream("LSR", light_naphtha, bc)     # RON/MON from the P/N/A/O composition
jet_measured = BlendComponent.from_stream("jet", jet_stream, bc, flash_C=42.0)   # a measurement wins
properties.estimate_properties(bc, moles)                      # the estimates alone
```

| Property | Estimate | Inputs | Status |
|---|---|---|---|
| `flash_C` | `1/T_F = -0.024209 + 2.84947/T10 + 3.4254e-3 ln T10` (K), Riazi & Daubert (1987), API TDB 2B7.1, Riazi MNL50 Ch. 3 | D86 10 % point of the stream (smoothed TBP, Riazi-Daubert TBP->D86) | (unverified) |
| `freeze_C` | ideal solubility of each cut's n-paraffins, `1/T = 1/T_f - R ln x / dH_f`, Won (1986) `T_f = 374.5 + 0.02617 MW - 20172/MW`, `dH_f = 0.1426 MW T_f` cal/mol; the freeze point is the smooth maximum over cuts | `paraffins_vol`, MW, `n_paraffin_share` | `T_f` checked against n-C10..n-C24 melting points (within 7 K at C10, 3.5 K above); `dH_f` 4-7 % high (odd) / 25-32 % low (even) against CRC; the freeze point itself not validated |
| `smoke_mm` | `SP = exp(-1.028 + 0.474 Kw - 0.00168 Tb')`, Riazi MNL50 Ch. 3, `Tb'` read in degR (see below) | TBP 50 % point, SG | (unverified) |
| `viscosity_cSt` | Abbott, Kaufmann & Domash (1971), API TDB 11A4.2, at 100 and 210 °F; ASTM D341 (Walther) line to `viscosity_T_C` (default 50 °C) | TBP 50 % point, SG | (unverified); `PropertyRangeWarning` near the correlation's pole |
| `RON`, `MON` | each cut split into n-/iso-paraffins, naphthenes, aromatics and olefins; pure-compound octanes of the reformer's model compounds at the cut's Tb; Ethyl RT-70 over the sub-components (the reformate's rule) | `paraffins_vol`, `naphthenes_vol`, `aromatics_vol` (`olefins_vol`), `n_paraffin_share` | (unverified); method of this project |

**Unverified means unverified.** No primary source could be opened while this was written (the API Technical Data Book, ASTM MNL50 and the journals were unreachable), so the constants are as recalled and no published worked example is reproduced. The tests check what can be checked without one: Won's melting points against tabulated data, the Walther line on its two points, the octane of a single-compound stream, monotonicity in the direction the physics requires, plausible values on typical products, and gradients against central differences. Give a measured value whenever you have one.

Things to know about each estimate:

- **Flash point** comes from the D86 10 % point, which is the lightest material in the cut; it gives about -45 °C for a gasoline, 55-70 °C for a kerosene and around 100 °C for a diesel, kerosenes on the high side of measured values.
- **Freeze point** is a physical model, not a correlation. ASTM D2386 measures the temperature at which the last crystal disappears, and in a kerosene those crystals are n-paraffins. Each cut's n-paraffins (mole fraction `z_j * P_j * n_paraffin_share`) are treated as one n-alkane of the cut's molar mass crystallizing pure from an ideal solution. A coarse cut grid hides the heaviest members of a cut and so puts the estimate low, as does Won's low heat of fusion for the even n-alkanes. Cuts lighter than MW 100 are left out; a liquid with none heavier gets the floor `FREEZE_FLOOR_K` (150 K). `n_paraffin_share` (default 0.5) is ILLUSTRATIVE.
- **Smoke point**: with `Tb` in kelvin, as the constants were recalled, the equation puts an ordinary kerosene (Tb 473 K, SG 0.80) at 44 mm, far above the 20-30 mm such kerosenes measure. With `Tb` in degrees Rankine it gives 23 mm, and 15 mm for a naphthenic one. The Rankine reading is used because it gives measured magnitudes. That is a judgement, not a citation, and the first thing to check when the source is to hand.
- **Viscosity** is the Abbott correlation on the stream's bulk TBP 50 % point and SG, not a Refutas blend of per-cut values: per cut the heaviest lumps sit on the correlation's pole. The temperature of `viscosity_cSt` is `viscosity_T_C` (50 °C, the fuel-oil reference). Every component in one pool must be at one temperature for the Refutas rule. The D341 line drops the low-viscosity correction terms below about 2 cSt.
- **Octane** uses the reformer's pure-compound octanes (`reforming.species`, API Research Project 45 values as recalled, unverified, C9-C10 paraffins extrapolated), one model compound per type and carbon number, an ILLUSTRATIVE n-/iso-paraffin split, and the iso-paraffin octane as a placeholder for olefins. Straight-run C8+ iso-paraffins are many isomers, most of them higher in octane than the 2-methylalkane that stands for them, so heavy naphthas come out low. On the tests' crude it gives RON 74 for a C5-85 °C light naphtha and 41 for an 85-180 °C heavy naphtha; measured ranges are about 60-75 and 40-60. Because it uses the reformer's numbers and rule, a straight-run naphtha and a reformate in one pool are on one octane basis. It is meaningful for naphthas boiling below about 460 K.

Tests: `tests/refinery/test_properties.py`.

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

For products of the crude and vacuum units, use `BlendCharacterization.from_characterization(char)` instead. It takes the crude characterization's Tb, SG, MW, its critical constants and its vapour-pressure acentric factor, so the pool's Raoult RVP sees `psat(Tb) = 1 atm` exactly as the columns do. It also takes the sulfur vector (and, with `contaminants=True`, nitrogen and CCR) as the `S_ppm`, `N_ppm` and `CCR_wt` qualities. Qualities the assay did not give are left out. If the characterization carries a composition (`char.composition`), its hydrocarbon types become the `paraffins_vol`, `naphthenes_vol`, `aromatics_vol` and `olefins_vol` qualities (vol%); `composition=False` leaves them out. A product stream from either column is then a `BlendComponent.from_stream` input; `F_water` and `F_H2O` are ignored. The pool's sulfur for LVGO or HVGO equals the vacuum column's own report to 1e-10, since both average the same per-component vector.

### Not in scope

- Tank inventory and multi-period scheduling. The pool is steady state, per period.
- Crude blending ahead of the CDU (assay mixing).
- A *validated* straight-run octane correlation. The PNA-based estimate above is this project's method on recalled pure-compound octanes; measured or unit-reported octanes override it.

(refinery-reforming)=
## Catalytic reforming

`difflow_refinery.reforming` (#309) is a semi-regenerative catalytic reformer: hydrotreated heavy naphtha to reformate and hydrogen over three (or any number of) adiabatic reactors with fired interstage heaters, a high-pressure separator, hydrogen-rich recycle gas and a stabilizer. It is a `difflow.Flowsheet` with the recycle gas as its tear. Every output is differentiable in the reactor inlet temperatures (or WAIT), separator pressure, H2/HC ratio, space velocity and the naphtha's composition.

```python
from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha

reformer = CatalyticReformer(ReformerParams())           # 3 reactors, WAIT 500 C, 12 bar, H2/HC 5
res = reformer.solve(lean_naphtha())                     # 10 kg/s of an illustrative lean naphtha
print(res.summary())
res.outputs()["reformate.RON"], res.outputs()["h2.net_mol_s"], res.balances()

# Implicit gradients through the converged recycle (forward mode):
import jax
def ron(wait):
    p = ReformerParams().with_wait(wait)
    return reformer.solve(lean_naphtha(), p, tear_initial=res.tear).outputs()["reformate.RON"]
jax.jacfwd(ron)(773.15)
```

On the two illustrative feeds, at WAIT 500 °C, 12 bar separator, H2/HC 5, LHSV 1.5 h-1 (these are the model's own numbers with its illustrative kinetics, not data):

| | lean (N+2A 41) | rich (N+2A 75) |
|---|---|---|
| reactor ΔT (K) | -66, -62, -47 | -61, -56, -39 |
| C5+ reformate, vol% of feed | 79 | 81 |
| RON (RT-70) / MON | 92.5 / 82.6 | 98.1 / 87.4 |
| aromatics / benzene, vol% | 60 / 1.7 | 68 / 2.1 |
| net H2, wt% of feed / purity mol% | 2.8 / 87 | 2.4 / 85 |

`res.summary()` prints these; ten degrees more WAIT on the lean feed gives RON 98.8 at 76 vol% C5+ and 3.0 wt% H2.

**Known defect of the illustrative constants: the rich feed makes less hydrogen than the lean one.** With the default `ReformingKinetics()`, the rich (high-naphthene) feed makes 2.4 wt% net H2 and the lean (high-paraffin) feed 2.8 wt%. Commercial experience is the opposite: a naphthenic feed makes more hydrogen. The cause is the paraffin chemistry. The ring-opening pre-exponential (`A["ring_opening"] = 0.5`) and its carbon-number factors (up to 2.5 for C10) make dehydrocyclization of C7+ paraffins fast enough that the lean feed converts most of its paraffins to aromatics. Each one releases 4 H2, against the 3 H2 a naphthene gives. The rich feed has few paraffins to convert, and it also loses hydrogen to naphthene hydrocracking (`A["hydrocracking_N"]`, 2 H2 per event). The constants were tuned for octane and yield, not hydrogen. This is a defect of the illustrative parameter set, not a property of the model. Fitting the kinetics to plant or published data should remove it, and until then the hydrogen-versus-feed trend should not be relied on.

The module is a library plus a flowsheet, like the blend pool: it registers **no** palette operation.

### Feed: P/N/A by carbon number

The model cannot run on a boiling curve and a gravity. It needs paraffins, naphthenes and aromatics by carbon number, C6 to C10. A `NaphthaFeed` holds molar flows of the reformer's species and is built from:

- **`NaphthaFeed.from_piona(...)`**, a measured PIONA by carbon number (ASTM D5134 / D6730 grouped), on a mass, volume or mole basis. This is the preferred input.
- **`NaphthaFeed.from_characterization(char, flows)`**, the naphtha cuts of the unified characterization (#301) with the #305 hydrocarbon-type estimate (`characterize(assay, composition=True)`). The mapping:
  1. each cut's P/N/A/O **volume** fractions are converted to mass with the density of the type's model compound at the cut's carbon number, so the cut's mass is conserved exactly; olefins count as paraffins (the feed is hydrotreated);
  2. each type's mass is placed on C6..C10 by the cut's `Tb`, interpolated against the boiling points of the type's model compounds (n-paraffins; cyclohexane then the n-alkylcyclohexanes; benzene then the n-alkylbenzenes) and split linearly between the two neighbouring carbon numbers, clipped at C6 and C10;
  3. paraffins are split into normal and iso by `iso_fraction`, and C6 naphthenes into methylcyclopentane and cyclohexane by `mcp_fraction`. Neither split is in a Tb/SG characterization; the defaults (0.5, 0.6) are ILLUSTRATIVE and a PIONA replaces them;
  4. light ends kept as real species map to themselves (`n_hexane` -> `nP6`, `isopentane` -> `iC5`, ...); water is dropped.

  The #305 estimate is coarse at carbon-number resolution; the lumped feed's hydrogen content is the model compounds', not the #305 hydrogen estimate (`feed.hydrogen_wt()` reports it for comparison).
- `lean_naphtha()` and `rich_naphtha()` are two made-up, ILLUSTRATIVE feeds (not any crude's assay).

`feed.with_group_fraction("naphthenes", x)` is the naphthene-content lever (other groups rescaled at constant volume or mass); `feed.n_plus_2a()` is the reformability index.

### Species and model compounds

Every lump is one real compound, whose formula, thermochemistry, critical constants, density and octane it takes:

| C | n-paraffin `nP` | iso-paraffin `iP` | naphthene `N` | aromatic `A` |
|---|---|---|---|---|
| 6 | n-hexane | 2-methylpentane | cyclohexane `N6` and methylcyclopentane `N5_6` | benzene |
| 7 | n-heptane | 2-methylhexane | methylcyclohexane | toluene |
| 8 | n-octane | 2-methylheptane | ethylcyclohexane | ethylbenzene |
| 9 | n-nonane | 2-methyloctane | n-propylcyclohexane | n-propylbenzene |
| 10 | n-decane | 2-methylnonane | n-butylcyclohexane | n-butylbenzene |

plus H2 and the C1-C5 paraffins (`C1`, `C2`, `C3`, `iC4`, `nC4`, `iC5`, `nC5`): 29 species. Naphthene/aromatic pairs share their side chain, so each dehydrogenation equilibrium is that of a real reaction. The main simplification of the thermodynamic layer is that a lump's free energy is one isomer's, not that of the isomer distribution the catalyst holds (a C8 reformate aromatic is mostly xylenes, not ethylbenzene). C9 and C10 lumps are therefore pseudocomponents *by group*, represented by their n-alkyl member.

### Reaction network and rate laws

Smith's (1959) four reactions, per carbon number in the manner of Krane et al. (1959), plus three steps the carbon-number view needs. With `P` the total pressure and `p` partial pressures, both in bar, rates in mol/s per kg of catalyst:

| Family | Reaction | Rate | Form from |
|---|---|---|---|
| dehydrogenation | N_n = A_n + 3 H2 | k (p_N - p_A p_H2^3 / K) | Smith (1959) |
| ring opening | N_n + H2 = nP_n, N_n + H2 = iP_n | k (p_N p_H2 - p_P / K) | Smith (1959) (the reverse is dehydrocyclization) |
| hydrocracking of P | P_n + H2 -> lighter paraffins | k p_P / P | Smith (1959) |
| hydrocracking of N | N_n + 2 H2 -> lighter paraffins | k p_N / P | Smith (1959) |
| isomerization | nP_n = iP_n | k (p_nP - p_iP / K) | this module, ILLUSTRATIVE |
| ring expansion | MCP = CH (C6 only) | k (p_MCP - p_CH / K) | this module, ILLUSTRATIVE |
| dealkylation | A_n + H2 -> A_(n-1) + CH4 (n >= 7) | k p_A p_H2 / P | this module, ILLUSTRATIVE |

For C6 the ring a paraffin closes to is methylcyclopentane, which must expand to cyclohexane before it dehydrogenates: that is why benzene forms slowly. Hydrocracking splits a `C_n` at one C-C bond chosen uniformly, `2/(n-1)` mol of each `C_k`, `k = 1..n-1`, with `iso_fraction` of each C4+ product branched. That conserves carbon and hydrogen exactly (tested reaction by reaction) and is ILLUSTRATIVE.

`k = activity A f(n) exp(-E/R (1/T - 1/T_ref))`, `T_ref = 773.15 K`. The activation energies of Smith's reactions are his temperature coefficients, 34 750, 59 600 and 62 300 °R (dehydrogenation, ring opening, both hydrocrackings), divided by 1.8; they are as the reforming literature commonly reproduces Smith's model, and were not checked against the paper (unverified). The pre-exponentials `A` and carbon-number factors `f(n)` are **neither Smith's nor Krane's**: they were chosen for this module so that the unit behaves like a modern semi-regen reformer (first-reactor ΔT, octane and yields in the commonly quoted ranges) and are ILLUSTRATIVE until fitted. Their trends (heavier paraffins cyclize and crack faster; C6 paraffins barely cyclize) are the trends Krane et al. reported, not their values. `activity` scales every rate constant: it is the parameter `difflow.reconciliation.tracking` would estimate from plant data as the catalyst deactivates (the tracking loop itself is not wired in here).

**Equilibrium constants come from the Gibbs energies**, never from a kinetic paper: `ln K = -ΔG°(T)/(RT)`, with `ΔG° = ΔH°(T) - TΔS°(T)` from the species' ideal-gas `Hf`, `S0` and Cp, 1-bar standard state. Dehydrogenation therefore limits correctly at low pressure and high temperature. The heats of reaction come from the same data (cyclohexane to benzene: +206.06 kJ/mol at 298 K). Tests pin `ln K` to the Gibbs energy, check van 't Hoff against the coded heats of reaction, and check that a long bed relaxes the C7 dehydrogenation quotient to `K(T_out)`.

**Coke** (kg per kg catalyst per s) is `k_c (p_N + p_A) / p_H2` with an Arrhenius `k_c`. It rises with severity and falls with hydrogen partial pressure, the dependence the deactivation literature describes; the form and constants are this module's (ILLUSTRATIVE). The cycle length is the days to `coke_capacity` (default 0.15 kg/kg, ILLUSTRATIVE) of coke on catalyst. Coke is reported, not withdrawn from the balances (about 1e-5 of the feed).

### Reactors and heaters

Each bed is one-dimensional plug flow in catalyst mass, isobaric, adiabatic:

```
dF/dW = nu^T r(p, T),   H(F, T, P) = H_in,   d(coke)/dW = r_coke
```

The temperature is recovered from the enthalpy at every point by Newton's method rather than integrated as an ODE. Element balances are then exact (the stoichiometry conserves them, and Runge-Kutta methods preserve linear invariants), and the energy balance is exact to the Newton tolerance whatever the ODE tolerance. `diffrax` integrates it (Tsit5, adaptive, `rtol` 1e-8 by default) in the normalized bed coordinate. Gradients use `diffrax.ForwardMode`, because the reformer is differentiated through the recycle's implicit fixed point, which needs forward-mode derivatives of everything in the loop: **use `jax.jacfwd`**, not `jax.grad`, on a reformer (`ReformingReactor(..., adjoint="reverse")` exists for a stand-alone bed).

**One enthalpy basis** is used throughout: `H = sum F_i [Hf_i + int cp_i dT] + F h_dep(T, P, y)`. That is `CubicThermo`'s Peng-Robinson enthalpy (ideal-gas sensible plus PR departure) with the heats of formation added. `thermo.cubic_thermo()` builds the `CubicThermo` from the same Cp fits, so the reactor, the fired heaters, difflow's `EOSFlash` and its `Compressor` all share one reference state. A fired heater's absorbed duty is the outlet minus inlet enthalpy, and `fired = duty / efficiency` (as `Furnace` reports it). There is no feed-effluent exchanger: the charge-heater duty is the whole of heating the naphtha and recycle gas from separator to reactor temperature, and is larger than a real unit's for that reason.

### Separator, recycle and stabilizer

These are kept thin and local (`reforming/separation.py`) so they can be consolidated with the hydrotreater's shared separator and recycle module (#306) later:

- **`ProductSeparator`**: effluent cooler and Peng-Robinson flash (difflow's `EOSFlash`) at the separator temperature and pressure. The vapour is the flash's `V y`; the liquid is the feed minus the vapour, so the component balance closes to round-off. PR runs with all `k_ij = 0`, hydrogen-hydrocarbon included, so the hydrogen dissolved in the liquid is PR's unfitted prediction.
- **`RecycleSplitter`**: recycles enough of the vapour to carry `H2_HC` mol of hydrogen per mol of naphtha hydrocarbon (the spec); the rest is net gas.
- **Recycle compressor**: difflow's `eos_units.Compressor` (isentropic, efficiency 0.75) from separator to reactor pressure.
- **`Stabilizer`**: a **documented simplification**. This unit was built before the gas plant (`difflow_refinery.gasplant`, #312) reached `main`, so it does not use its debutanizer (wiring it in is follow-up work), and difflow's vacuum stage columns are not set up for a hydrogen-bearing feed. It is a component split instead: H2, C1 and C2 to fuel gas, C3 and `c4_recovery` of the butanes to LPG, the rest to stabilized reformate. `c4_recovery` stands in for the RVP / C4-in-reformate spec. Its duty is the net heat on the enthalpy basis, not a column design.

### Specs and outputs

| Spec (`ReformerParams`) | Default | |
|---|---|---|
| `inlet_T` (or `with_wait(...)`) | 773.15 K each | reactor inlet temperatures; WAIT is their catalyst-weighted mean |
| `catalyst_split` | 0.15, 0.30, 0.55 | number of reactors = its length |
| `LHSV`, `catalyst_density` | 1.5 1/h, 700 kg/m3 | catalyst mass = density x feed std volume per hour / LHSV (density ILLUSTRATIVE) |
| `P_separator`, `loop_dP` | 12 bar, 3 bar | reactors at `P_separator + loop_dP`, isobaric |
| `T_separator` | 311.15 K | |
| `H2_HC` | 5 | recycle H2 per naphtha hydrocarbon, mol/mol |
| `c4_recovery` | 0.95 | stabilizer butanes to LPG |
| `kinetics` | `ReformingKinetics()` | rate parameters, `activity` |

A reformate RON target in place of WAIT: `reformer.wait_for_ron(feed, ron)` solves for the WAIT by secant iteration (concrete). Its gradient with respect to any other input is `-(dRON/dx)/(dRON/dWAIT)`, both from `jax.jacfwd` at the returned point.

`res.outputs()` (units in `reforming.OUTPUT_UNITS`) includes reformate and C5+ yield (vol%, wt%), RON/MON (RT-70 and linear), aromatics, benzene, RVP, SG, net H2 (mol/s, wt% of feed, purity), LPG and fuel gas, every reactor's ΔT and outlet temperature, heater absorbed and fired duties, compressor power, separator duty, coke make, cycle length, WAIT and WABT. `res.balances()` returns the overall mass, carbon, hydrogen, energy and sulfur closures and each reactor's adiabatic residual.

**Reformate properties from composition** (`reforming.products`). RON and MON are the pure-compound octanes of the species, blended with the Ethyl RT-70 rule that `BlendPool` uses. Aromatics and benzene are standard liquid volume fractions. RVP is `raoult_rvp` (D323 geometry) on Lee-Kesler vapour pressures. `products.blend_component("reformate", res.flows("reformate"))` hands it to a `BlendPool`. Pure-component octanes are not blending octanes; the RT-70 rule with the large aromatic and sensitivity spreads of a reformate puts RON 8-12 above the linear average. The low-octane paraffins sit below the range RT-70 was fitted on, so the rule extrapolates there.

**Sulfur through the reformer** (#330, `reforming.sulfur`). The feed's organic sulfur, `ReformerParams.feed_sulfur_wppm` (default `None`: the feed's own `sulfur_wppm` if it has one, else zero), is carried as a trace element, outside the species list and the recycle tear. At sub-ppm levels it changes neither the chemistry nor the phase split, so its balance is solved on the converged streams:

- a share `sulfur_conversion` (default 0.95, ILLUSTRATIVE, not a desulfurization model) is hydrogenolysed to H2S over the reactors; the rest stays in the reformate;
- the separator sends a share `a = K V / (K V + L)` of the H2S to the vapour, with the ideal K-value `K = Psat(T)/P` (Lee-Kesler, H2S critical constants from the `chemicals` 1.5.2 PSRK table) against the separator's own vapour and liquid flows;
- with a share `s` of the vapour recycled, the steady state puts `(1 - s) a G / (1 - s a)` of an H2S make `G` in the net gas and `(1 - a) G / (1 - s a)` in the separator liquid. The two add up to `G`, so the balance closes exactly, and the recycle gas holds the same H2S concentration as the net gas;
- the stabilizer sends all dissolved H2S overhead with the fuel gas. H2S boils between ethane and propane; a real stabilizer puts some in the LPG.

`res.sulfur()` returns the split, `res.outputs()` adds `reformate.S_wppm`, `net_gas.H2S_ppmv`, `recycle.H2S_ppmv` and the H2S flows, `res.balances()["sulfur"]` is the closure, `res.reformate["S_ppm"]` is the reformate's sulfur and `res.blend_component()` hands the reformate to a pool with it. The hydrogen the H2S takes and the hydrocarbon part of the sulfur compound are not tracked, since both are ppm of their streams.

### Planning

`reformer_block(reformer, feed, levers, outputs)` gives a `difflow.planning.Block`, like `cdu_block`. The levers are `wait` (C), `P_separator` (bar), `H2_HC`, `LHSV`, `feed.rate` (kg/s), `feed.naphthenes` (vol%) and `c4_recovery`. The outputs are any `res.outputs()` keys; link `reformate.*` to a blend-pool block and `h2.net_mol_s` to a hydrogen balance. The block runs the traced recycle (optimistix fixed point with implicit differentiation, warm-started at the base solution) and defaults to `ad_mode="fwd"`.

### Validation: what is and is not checked

Checked (`tests/refinery/test_reforming.py`; flowsheet tests are marked `slow`):

- Converges with H2 recycle on the lean and rich feeds from the default initialization (Anderson, about 25-30 recycle iterations).
- Overall mass, carbon, hydrogen and energy balances close to 1e-8 relative (measured: 1e-11 to 1e-12), and every reactor is adiabatic to round-off.
- The first reactor has the largest temperature drop.
- RON and net H2 rise and C5+ yield falls with WAIT; aromatics rise as the separator pressure falls.
- `wait_for_ron` reaches a RON target (95 on the lean feed) to 1e-3.
- The reformate enters a `BlendPool` as a property-mode `BlendComponent`.
- Implicit gradients of reformate yield, RON, net H2 and first-reactor ΔT with respect to WAIT, separator pressure, H2/HC and naphthene content match central differences to 1e-5 (measured: 1e-7). A full `jax.jacfwd` of those 4x4 plus the eight finite-difference solves takes about eight minutes on one CPU core, mostly compilation of the traced recycle.
- Thermochemistry: coded `Hf`/`S0` are pinned to their sources; `Hf` agrees with `difflow.database` within 1 kJ/mol for the 16 species both hold; `ln K` is the Gibbs energy; van 't Hoff holds against the coded heats of reaction; a long bed reaches the Gibbs-energy equilibrium.

**Not done, and not claimed:**

- **No published commercial-reformer simulation is reproduced.** Neither Padmavathi & Chaudhuri (1997) nor Taskar & Riggs (1997) could be obtained to check which one tabulates feed, conditions and outlet data in full, so no cross-check against them is made. The kinetics are illustrative and would have to be replaced by either paper's parameters for such a comparison.
- **No IDAES `GibbsReactor` comparison** of the equilibrium layer: IDAES is not installed in this environment. The equilibrium layer is checked against its own Gibbs energies (above).
- **No dedicated example notebook.** The reformer runs inside the whole-refinery example, `examples/40_refinery_flowsheet.ipynb` (naphtha hydrotreater -> reformer -> gasoline pool), on a feed from `NaphthaFeed.from_characterization`.
- No Gary-Handwerk-Kaiser yield-versus-RON cross-check: the figure could not be consulted.

### References

| What | Source | Status |
|---|---|---|
| 4-reaction network, rate-law forms, activation energies (34 750, 59 600, 62 300 °R) | Smith, R.B., "Kinetic analysis of naphtha reforming with platinum catalyst", *Chem. Eng. Prog.* 55(6), 76-80 (1959) | Paper not consulted. The forms and coefficients are as commonly reproduced in later reforming papers (unverified); the title and pages are as cited there (unverified). |
| Carbon-number lumping, rate trends with carbon number | Krane, H.G., Groh, A.B., Schulman, B.L., Sinfelt, J.H., "Reactions in catalytic reforming of naphthas", *Proc. 5th World Petroleum Congress*, New York (1959), Sect. III | Not consulted; section and pages unverified. Only the lumping idea and the qualitative trends are used, none of its numbers. |
| Commercial reformer models with coking (not used numerically) | Padmavathi, G., Chaudhuri, K.K., *Can. J. Chem. Eng.* 75(5), 930-937 (1997); Taskar, U., Riggs, J.B., "Modeling and optimization of a semiregenerative catalytic naphtha reformer", *AIChE J.* 43(3), 740-753 (1997) | Not consulted (unverified). Cited for the coke dependence on severity and H2 partial pressure only, qualitatively. |
| KINPTR (background) | Ramage, M.P., Graziani, K.R., Schipper, P.H., Krambeck, F.J., Choi, B.C., *Adv. Chem. Eng.* 13, 193 (1987) | Not consulted (unverified); background only. |
| Ideal-gas Hf (298.15 K) | API Technical Data Book (as `API_TDB_G` in `chemicals` 1.5.2); 2-methylhexane from the CRC Handbook (as `CRC`) | Read from the `chemicals` tables; cross-checked against the CRC, ATcT and Yaws tables (spread within 1.5 kJ/mol except 2-methylnonane, CRC -260.2 against API -256.5 kJ/mol, unresolved; a sample is pinned in `HF_CROSSCHECK`). The primary tables were not opened. |
| Ideal-gas S0 (298.15 K, 1 bar) | Yaws ideal-gas entropy table (as `YAWS` in `chemicals` 1.5.2) | Read from `chemicals`; cross-checked against NIST WebBook values carried by `chemicals` (within 2.5 J/mol/K). Book edition unverified. |
| Ideal-gas Cp | TRC ideal-gas heat-capacity correlation (Thermodynamics Research Center), as tabulated in `chemicals.heat_capacity.TRC_gas_data`; cubic fit 298-1000 K by this module | Fit within 1.4 % of the correlation; values at 298 K pinned. |
| Tc, Pc, omega, Tb | First-ranked source in `chemicals` 1.5.2 (CoolProp reference EOS; IUPAC critical-property review; CRC; PSRK) | Read from `chemicals`; the primary tables were not opened. |
| Liquid density at 60 F | Perry's *Chemical Engineers' Handbook*, 8th ed., DIPPR-105 table (via `chemicals`); VDI Heat Atlas PPDS (n-propyl-, n-butylcyclohexane); COSTALD, Hankinson & Thomson, *AIChE J.* 25(4), 653-663 (1979) (2-methylhexane) | Table numbers and COSTALD pages unverified. 2-Methylheptane, -octane and -nonane are recalled handbook values (unverified). |
| Pure-compound RON/MON | API Research Project 45, ASTM STP 225, *Knocking Characteristics of Pure Hydrocarbons* (1958) | **Not consulted.** n-Heptane = 0 (and isooctane = 100) are exact by definition (ASTM D2699/D2700). Every other value is recalled (unverified). C9/C10 paraffins and n-butylcyclohexane are extrapolations by this module (`octane_source="estimate"`). Benzene's RON is the least certain. |
| Octane blending | Ethyl RT-70: Healy, Maassen & Peterson (1959), coefficients as in Maples (2000) | As in the blend pool (see [Blending rules](#refinery-blending)). |
| RVP | `raoult_rvp`, Lee-Kesler vapour pressure | As in the blend pool and `correlations`. |
| Separator VLE, departure enthalpy | Peng, D.-Y., Robinson, D.B., "A new two-constant equation of state", *Ind. Eng. Chem. Fundam.* 15(1), 59-64 (1976), doi:10.1021/i160057a011, via difflow's `PengRobinson` | `k_ij = 0`. |
| ΔG(T), ln K route | Smith, J.M., Van Ness, H.C., Abbott, M.M., *Introduction to Chemical Engineering Thermodynamics*, McGraw-Hill, chemical-reaction-equilibria chapter | Standard thermodynamics; chapter and equation numbers vary by edition (unverified). |
| Thermochemistry of model compounds (the issue's suggestion) | Stull, D.R., Westrum, E.F., Sinke, G.C., *The Chemical Thermodynamics of Organic Compounds*, Wiley (1969) | **Not used**: the values come from the `chemicals` tables above. |
| Reforming practice, yield vs. RON | Gary, J.H., Handwerk, G.E., Kaiser, M.J., *Petroleum Refining: Technology and Economics*, 5th ed., CRC Press (2007), catalytic reforming chapter | Not consulted; no cross-check made. |
(refinery-hydroprocessing)=
## Hydroprocessing building blocks

`difflow_refinery.hydroprocessing` holds the parts every hydroprocessing unit shares: the stream state, a Peng-Robinson flash, an adiabatic trickle-bed reactor that runs *any* kinetic model, the recycle-gas loop and a product stripper. The hydrotreater ([below](#refinery-hydrotreater)) is these parts plus its own kinetics (`difflow_refinery.hydrotreating`); a hydrocracker is meant to be these parts plus another kinetic model. Nothing in this module knows what HDS is.

(refinery-hydroprocessing-layout)=
### The stream: gases, cuts and attributes

A hydroprocessing stream carries real **gases** (H2, H2S, NH3, C1--C4, the light ends the characterization keeps as species, water) and the characterization's **cuts**. A reactor changes what a cut is made of, which a molecule flow cannot say, so each cut also carries **attribute flows**: extensive amounts that ride with its molecules. `Layout(gases, cuts, attributes, attribute_elements)` names them; `Flows(gas, cut, attr)` holds them as arrays of shape `(n_gas,)`, `(n_cut,)` and `(n_cut, n_attr)`.

- `"C"` and `"H"` (carbon and hydrogen atoms) are compulsory. A cut's mass is computed from its atoms, `m = 12.0107 n_C + 1.00794 n_H + 32.065 n_S + 14.0067 n_N` (g/s; S and N summed over every attribute that counts them), and its molecular weight is `m / F`. A saturated cut is heavier per molecule, a desulfurized one lighter, and the mass balance closes because the hydrogen came from the gas.
- Attributes are extensive. A mixer adds them, a splitter scales them, a phase split partitions them in proportion to their cut's molecules. So the element balances close to round-off through any sequence of units.
- As a difflow stream (`Flows.to_stream`): `F_<gas>`, `F_<cut>` and `F_<cut>@<attribute>`, all mol/s. difflow's own mixers and splitters therefore handle them correctly.
- `Flows.elements(layout)` gives the C, H, S and N atom flows; `Flows.mass(layout)` the total mass.

(refinery-hydroprocessing-thermo)=
### Thermodynamics

`hydroprocessing.thermo.Components` is the property table of the flashing components (the layout's gases except water, then its cuts), built with `Components.build(layout, Tb, SG, MW, Tc, Pc, omega, hvap_nb, cp_ig, kij=None)` from the characterization's own arrays, so a gradient reaches the assay through it. `difflow.thermo.CubicThermo` is not used because it is built from concrete species data (`Characterization.thermo("pr")` needs floats).

- **Peng-Robinson** (Peng & Robinson 1976, Eqs. 3, 4, 9--12 and the fugacity coefficient Eq. 17), van der Waals one-fluid mixing with a `k_ij` matrix. `kappa` is the 1976 quadratic up to `omega = 0.491` and the Robinson-Peng (1978) cubic above it. The liquid takes the smallest real root of the cubic, the vapour the largest; each root is polished by one implicit Newton step so its derivative is the implicit-function one.
- **k_ij** (`DEFAULT_KIJ`): the light-gas pairs from the ChemSep PR table (as distributed with the `thermo` package), H2S with every cut 0.0333 (ChemSep's H2S/n-decane), every other pair zero. Illustrative; pass `kij=`.
- **Enthalpy**: the ideal-gas path of the crude unit's `ColumnThermo`. Ideal-gas Cp integrated from 298.15 K; a liquid sits below it by Watson's heat of vaporisation through `dHvap(Tb)`. Gas constants: Tc, Pc and omega of H2, H2S and NH3 from `difflow.database`, of the hydrocarbons from the crude unit's light-end table (Poling, Prausnitz & O'Connell 5th ed., App. A); ideal-gas Cp of H2, H2S and NH3 from Reid, Prausnitz & Poling 4th ed., App. A; dHvap at the normal boiling point from the CRC Handbook as tabulated in `chemicals`.
- **Liquid molar volume** of a cut at temperature: the Rackett equation in the Spencer-Danner form, anchored to the cut's 60 °F density, with the Yamada-Gunn `Z_RA = 0.29056 - 0.08775 omega`; `Tr` capped smoothly at 0.95. Dissolved gases take no volume.

(refinery-hydroprocessing-flash)=
### The flash and the HP separator

`pr_flash(T, P, z, comps)` is an isothermal PR flash solved as `n` equations in `ln K`:

```
ln K_i - ln phi_i^L(x) + ln phi_i^V(y) = 0,    x = z / (1 + V (K - 1)),  y = K x
```

with `V` from Rachford-Rice solved as a **negative flash** (Whitson & Michelsen 1989): bracketed between the poles `1/(1 - K_max)` and `1/(1 - K_min)` instead of clipped to [0, 1]. The equations stay smooth through a phase boundary, and `x` and `y` remain the compositions of the (possibly incipient) phases. The reported split uses `beta = clip(V, 0, 1)`. Start: Wilson's K and 25 successive-substitution passes; finish: `optimistix` Newton. The derivative is the implicit-function one, `-J^-1 dR/dtheta` at the root, attached by a rule of the module's own (`_flash_root`, a `jax.custom_jvp` that calls itself for its primal, so second derivatives are exact too).

Two robustness rules (#332):

- **The Rachford-Rice bracket is set by the components present** (`z` above `TRACE = 1e-20` of the feed). A species that is absent -- the floored zero flow of hydrogen in a feed oil -- has a pole of its own that the physical root can lie beyond; bracketing on it left the root at the bracket's edge, the Newton polish crossed the pole, and every mole fraction came out NaN. A trace component beyond its pole gets `x = z` (`phase_denominator`), its own negligible amount.
- **A flash that fails reports, it does not raise.** Near a mixture's critical region Newton can diverge to a non-finite `ln K`. The flash then returns its successive-substitution start with a large `FlashResult.residual`. optimistix's implicit adjoint is not used because its linear solve raises on a non-finite Jacobian -- the "equinox NaN in linear solve" error a naphtha unit at 320 °C / 50 bar / LHSV 0.5 / 150 Nm³/m³ used to stop with. Callers read the residual: the hydrotreater's `converged` includes every flash (`outputs["flash.residual"]`), and `compress` returns NaN, not an exception, for a non-finite inlet.

`HPSeparator(layout)(flows, T, P, comps)` returns `(vapour, liquid, water, FlashResult)`: gases and cut molecules split by the flash, every attribute with its cut, water decanted whole (no free-water VLE).

At 50 °C and 47 bar the default diesel case dissolves 10.7 mol/s of H2 in about 230 mol/s of separator liquid (x_H2 about 0.045); that is the `h2.dissolved` loss.

(refinery-hydroprocessing-reactor)=
### The trickle-bed reactor and the kinetic-model interface

`TrickleBedReactor(layout, kinetics, options)` integrates adiabatic beds in series:

```
dF/dw = r(F, T, P)                   every gas, cut and attribute flow (w: kg of catalyst)
C_eff(F, T) dT/dw = q(F, T, P)        adiabatic
```

`r` (mol/s per kg) and `q` (W/kg, positive exothermic) come from the kinetic model. `C_eff = dH/dT` at fixed flows is the stream's heat capacity **including** the vaporisation a temperature rise causes, by forward-mode AD of the stream enthalpy.

**Phase model (pseudo-homogeneous).** Gas and liquid are in equilibrium along the bed. The K-values come from a PR flash at each bed's inlet and are carried down the bed linearised in temperature, `ln K(T) = ln K_in + (d ln K/dT)_in (T - T_in)` (the slope by forward-mode AD of the flash). Along a bed only the phase split moves (Rachford-Rice), not the K-values' composition dependence. Mass-transfer resistances, wetting and the effectiveness factor are multipliers the kinetic model applies; the Korsten-Hoffmann film model is **not** implemented.

**The interface.** A kinetic model is any object with

- `attributes` and `attribute_elements` -- the per-cut attributes it needs (the layout's must be these);
- `rates(ctx, params) -> Rates`.

`ReactionContext` is what it reads at a point: `flows`, `T`, `P`; the equilibrium `x`, `y` and vapour fraction `beta` of the flashing components; their **fugacity-equivalent liquid concentrations** `c = x / v_L` (mol/m³; `v_L` from the cuts' Rackett volumes); `c_attr`, the attribute concentrations in that liquid (`c_cut x attribute-per-molecule`); the partial pressures `p = y P`; `per_molecule`; and helpers `ctx.c_gas(name)`, `ctx.p_gas(name)`, `ctx.c_cut`. Inside the two-phase region (`0 < V < 1`) `x` and `y` are the equilibrium phases. Outside it they are the stream itself and the incipient phase in equilibrium with it, which is what the negative flash gives *at* the boundary: an all-vapour stream has `y = z` and `x = z / K` (its dew-point liquid, unnormalised, so the liquid's fugacities are the vapour's); an all-liquid one `x = z` and `y = K z`. So concentrations are defined through `x` even where the stream is all vapour, and a rate law written on them is continuous through a dry-out (`v_L` is the incipient liquid's molar volume). Before #332 the reactor used the negative flash's own fictitious split beyond the boundary; in a vapour-phase naphtha bed (`V = 9`) that put the hydrogen partial pressure at 1.9 bar instead of 11 (`y_H2 = 0.06` against `z_H2 = 0.37`), and the aromatics equilibrium ran backwards -- the dehydrogenation, negative chemical hydrogen and cooling beds first seen in example 40. A diesel bed is two-phase and is not affected. `Rates(gas, cut, attr, heat)` returns `d/dw` of every flow and the heat released. Element conservation is the kinetic model's job; `check_element_conservation(kinetics, ctx, params)` returns the net C, H, S, N production at a point.

**Beds and quench.** `reactor(oil, gas, T_in, T_gas, P, W, comps, params, quench=None)`. With `quench` fractions given, bed 1 gets the oil and the treat gas less every quench at `T_in[0]`, and each later bed's inlet temperature follows from the adiabatic mix (a scalar enthalpy balance, Newton). With `quench=None` every bed's inlet temperature is given and the quench each needs is solved (a scalar equation in the total quench, since the first bed's gas depends on it). The quench-mixing enthalpy uses the upstream bed's linearised K-values; the extra vaporisation caused by the quench gas itself is neglected, and the next bed's inlet flash resets the split. WABT is `sum_k W_k (T_in,k + 2 T_out,k)/3 / sum W`.

**Integration** is `diffrax` (Tsit5, PID step control on a state scaled by its inlet values; `ReactorOptions` sets tolerances, `fixed_steps`, `n_save` and the adjoint). Gradients are reverse mode through the solver with `RecursiveCheckpointAdjoint` (diffrax's default: the exact derivative of the discrete solution, checkpointed). `adjoint="backsolve"` selects the continuous adjoint; `adjoint="forward"` (`diffrax.ForwardMode`) makes the reactor forward-mode differentiable. Beds run under `lax.scan`, so the compiled program holds one copy of a bed whatever the bed count.

(refinery-hydroprocessing-recycle)=
### The recycle-gas loop

- `knockout(vapour, liquid)`: the recycle compressor's suction drum returns every cut in the separator vapour (and its attributes) to the separator liquid. A modelling choice: the recycle gas then carries only real species, and the tear does not need every cut attribute. At 40--60 °C the cuts in separator vapour are a few hundred ppm of it.
- `amine_scrub(gas, layout, h2s_removal, nh3_removal)`: fixed removal fractions. The NH3 fraction stands for the wash water a real unit injects upstream of the separator.
- `purge_split(gas, fraction)`.
- `compress(gas, layout, comps, T_in, P_in, P_out, eta)`: ideal-gas isentropic compression with temperature-dependent Cp (`sum z_i int Cp_i/T dT = R ln(P2/P1)`), `H_out = H_in + (H_s - H_in)/eta`, the convention of `difflow.units.eos_units.Compressor`. Ideal gas, not PR: the recycle gas is 80--95 % H2 at a compression ratio near 1.1, where the compressibility correction to the work is a few per cent (stated, not computed).
- `makeup_for_ratio(recycle, layout, makeup_y, h2_target)`: the makeup that brings the treat gas's H2 to a target, explicitly. The H2/oil ratio is a spec and the makeup rate an output.
- `solve_tear(g, x0, args, ...)`: Newton on `g(x, args) = x` over the recycle-gas flows (and the compressor outlet temperature). It is a difflow `Flowsheet` recycle in substance -- a tear on the recycle gas, converged and implicitly differentiated -- without the `Flowsheet` object, whose stream packing does not carry per-cut attributes.

**Newton through diffrax** (`hydroprocessing.solve.newton_solve`). optimistix's Newton forms its Jacobian in forward mode, and a diffrax solve with `RecursiveCheckpointAdjoint` supports only reverse mode. The tear, the quench balance and any target spec are therefore solved by a Newton loop of their own, inside `lax.while_loop` on stop-gradient inputs, with the Jacobian from `jax.jacfwd` of a copy of the residual built with the forward-mode diffrax adjoint (`f_iter`); then one step `x = x* - J^-1 f(x*, theta)` with `J` frozen at the solution and `f` the reverse-mode residual. Its value is `x*` and its derivative is `-J^-1 df/dtheta`, exactly the implicit-function one. So `g` must be a pure function of `(x, args)`: everything the loop depends on goes in `args`, never in a closure. Building the Jacobian with VJPs through the checkpointed adjoint instead compiled for six minutes on the default diesel case; the forward-mode copy, with the beds under `lax.scan`, brings the whole hydrotreater to about 70 s.

(refinery-hydroprocessing-stripper)=
### The product stripper

`strip(feed, layout, comps, feed_T, P_top, dP_stage, steam_rate, steam_T, spec)` runs a short steam-stripped column on the vacuum unit's `StageColumn`: `n_stages` equilibrium stages, the feed (heated to `feed_T`, the column's "furnace" knob) entering the top, liquid down, the bottoms leaving the last stage, steam under the last stage and (10 % by default, `feed_steam_fraction`) in the feed line. No condenser; the overhead goes to a drum. Thermodynamics are the vacuum unit's (Raoult with Maxwell-Bonnell vapour pressures, steam as a non-condensing vapour). Real gas species are not column components and leave with the overhead, as in the vacuum column. Cut attributes follow their cut, and the overhead is the feed less the bottoms, so the balances close exactly.

Why steam in the feed line: a separator liquid with its gases taken out is a subcooled liquid at the stripper's pressure, and `StageColumn`'s feed flash then has no vapour phase, which makes its Jacobian singular (condition number 1e16 on the default diesel). A tenth of the steam in the feed line gives the flash a vapour phase; the column then converges in four Newton iterations.

`overhead_drum(overhead, layout, comps, T, P)` is a PR flash: vapour is the sour off-gas, liquid the wild naphtha, all water decanted.

`cut_pseudo_components(...)` gives the stripper's property table, with each cut's own molecular weight (from its atoms) where it has flow.

---

(refinery-hydrotreater)=
## The hydrotreater

`difflow_refinery.hydrotreating.Hydrotreater` is a distillate hydrotreater -- naphtha, kerosene or diesel, straight-run or cracked -- on the shared building blocks: adiabatic trickle beds with quench, effluent cooler and HP separator, recycle-gas loop with amine scrubber, purge, compressor and makeup, product steam stripper and overhead drum. It is a library, not a palette operation (like the blend pool).

```python
import difflow_refinery as dr
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut

char = dr.characterize(assay, composition=True)              # #305 composition is required
feed = straight_run_cut(char, 230 + 273.15, 370 + 273.15, 50.0)   # or a CDU product stream
hdt = Hydrotreater(char, feed, HydrotreaterParams(T_in=(613.15,), P=50e5, lhsv=1.0, h2_oil=300.0))
res = hdt.solve(feed)
print(res.table())
res.outputs["product.S_wppm"], res.outputs["h2.chemical_nm3_m3"], res.balances
```

The unit carries every cut the feed has and every lighter pseudo-component, because the cracking leak puts molecules there. A crude-unit product stream works as it is (`F_<char.names>`); `straight_run_cut` is an idealized one (every cut boiling in a TBP range, in its crude proportion) for running without a crude unit.

(refinery-hydrotreater-feed)=
### The feed on the attribute layout

`hdt_feed(char, stream, layout)` turns a characterized stream into attribute flows (`HDT_ATTRIBUTES`): C and H atoms; S atoms in each of the five `SULFUR_CLASSES`; N atoms in the two `NITROGEN_CLASSES`; molecules that are mono-, di- and poly-aromatic, olefinic and naphthenic. All of it is read from `char.composition` (#305):

- carbon is `1 - H - S - N` by mass, so the cut's mass from its atoms equals `F x MW` exactly;
- a cut's **volume** fractions of types are taken as its **mole** fractions (the same assumption the composition module makes for procedure 2B4.1; product aromatics are reported back through the same rule, so an untreated cut reports exactly what it came in with);
- total aromatics are split into mono/di/poly by `DEFAULT_AROMATIC_SPLIT`, an **illustrative** split by boiling point (all mono in naphtha, about 60/30/10 in diesel, more polyaromatic in the vacuum range) from no source. Pass measured ones (`aromatic_split=`, e.g. from EN 12916 or IP 391).

(refinery-hydrotreater-kinetics)=
### Reactions and rate laws

Per cut, on the cut's own attribute concentrations (`c`, mol/m³ of fugacity-equivalent liquid), per kg of catalyst, with `f = activity x effectiveness x wetting`, Arrhenius `k(T) = k_ref exp(-E/R (1/T - 1/T_ref))`, `h = (c_H2/c_ref)^m` and adsorption constants `K(T) = K_ref exp(-dH_ads/R (1/T - 1/T_ref))`:

| Reaction | Rate | H2 per event | Heat per event (kJ/mol) |
|---|---|---|---|
| HDS, class j: S_j + nu_j H2 -> H2S | `f k_j c_Sj h / (1 + K_H2S c_H2S + K_N c_Nbasic)^2` | 2.0, 4.0, 3.0, 2.6, 3.95 | -104.7, -261.4, -157.0, -83.9, -173.1 |
| HDN, basic / non-basic | `f k_j c_Nj h / (1 + K_H2S c_H2S)` | 4.0, 5.0 | -238.2, -263.0 |
| poly + 2 H2 <-> di | `f k h (c_A3 - c_A2 / (K3 (pH2/1 bar)^2))` | 2 | -115.2 |
| di + 2 H2 <-> mono | `f k h (c_A2 - c_A1 / (K2 (pH2/1 bar)^2))` | 2 | -124.6 |
| mono + 3 H2 <-> naphthene | `f k h (c_A1 - c_Nn / (K1 (pH2/1 bar)^3))` | 3 | -205.3 |
| olefin + H2 -> paraffin | `f k c_O h` | 1 | -123.4 |
| cracking leak: molecule + H2 -> lighter molecule + C1--C4 | `f k c_cut` | 1 | -42.7 |

Sulfur classes in the order sulfides, thiophenes, benzothiophenes, dibenzothiophenes, hindered (4-/4,6-alkyl) DBTs.

- **HDS** is the Langmuir-Hinshelwood-Hougen-Watson rate with a squared H2S-inhibition denominator, the form of Korsten & Hoffmann (1996) and, with a richer denominator, of Froment, Depauw & Vanrysselberghe (1994) and Vanrysselberghe & Froment (1996). It is first order in each class. A sum of first-order classes with different constants is what gives a lumped total-sulfur rate its apparent order above one. Basic nitrogen adsorbs on the same sites and sits in the denominator. `hds_form="power"` gives the nth-order fallback `f k c_ref,S (c_Sj/c_ref,S)^n h`, with no inhibition.
- **Aromatics** saturate reversibly, first order, with equilibrium constants `K = exp(-(dH - T dS)/RT)` (pressures in bar) from model-compound thermochemistry. Saturation is exothermic and loses moles of gas, so equilibrium recedes as temperature rises and aromatics pass through a minimum.
- **H2 stoichiometry** per class is that of a model compound (below). The hydrogen not leaving as H2S or NH3 goes onto the cut (`H += 2 nu - 2` per S, `2 nu - 3` per N). A desulfurized molecule keeps its carbon skeleton; the cut's molecule count does not change. Element balances are exact.
- **The cracking leak** moves a molecule of cut `i` to the cut whose carbon number per molecule is nearest to `i`'s less the gas fragment's (fixed at construction from the composition), splitting off one C1--C4 molecule in the proportions `CRACK_GAS_SPLIT` (10/15/35/15/25 % C1/C2/C3/iC4/nC4, illustrative), with one H2.
- **Deactivation** is the `activity` multiplier a(t). It is a differentiable parameter, so `difflow.reconciliation.tracking` can track it from plant data as the drifting parameter that loop is built for (not wired up or tested here).

**Model compounds** behind the stoichiometry, heats and equilibrium (ideal gas, 298 K; formation enthalpies and entropies as tabulated in the `chemicals` package, which transcribes TRC/ATcT/CRC sources -- per-compound primary source unverified):

| Class | Model reaction | dH (kJ/mol) | dS (J/mol/K) |
|---|---|---|---|
| sulfides | diethyl sulfide + 2 H2 -> 2 ethane + H2S | -104.7 | |
| thiophenes | thiophene + 4 H2 -> n-butane + H2S | -261.4 | |
| benzothiophenes | benzothiophene + 3 H2 -> ethylbenzene + H2S | -157.0 | |
| DBTs | 80 % DBT + 2 H2 -> biphenyl + H2S (DDS), 20 % DBT + 5 H2 -> cyclohexylbenzene + H2S (HYD) | -83.9 | |
| hindered DBTs | 35 % DDS / 65 % HYD, DBT model compounds | -173.1 | |
| basic N | quinoline + 4 H2 -> propylbenzene + NH3 | -238.2 | |
| non-basic N | carbazole + 5 H2 -> cyclohexylbenzene + NH3 | -263.0 | |
| poly -> di | phenanthrene + 2 H2 -> 1,2,3,4-tetrahydrophenanthrene | -115.2 | -228.3 (taken from di; THP entropy not tabulated) |
| di -> mono | naphthalene + 2 H2 -> tetralin | -124.6 | -228.3 |
| mono -> naphthene | benzene + 3 H2 -> cyclohexane | -205.3 | -363.1 |
| olefins | 1-hexene + H2 -> n-hexane | -123.4 | |
| cracking | n-hexane + H2 -> n-butane + ethane | -42.7 | |

The DDS/HYD route shares of the two DBT classes are illustrative, set by the qualitative finding (Girgis & Gates 1991; Vanrysselberghe & Froment 1996) that CoMo removes DBT mainly by direct desulfurization and 4,6-DMDBT mainly after ring hydrogenation. Heats are gas-phase at 298 K: the heats of vaporisation of the reacting species and the temperature dependence are neglected. Benzene is a more favourable case than an alkylbenzene, so the mono-aromatic equilibrium is if anything too far to the right.

**Where the rate constants come from.** The forms are the literature's. The constants in `HDTKineticParams` (rate constants, activation energies, adsorption constants and enthalpies, H2 orders) are **illustrative**, chosen here so that a straight-run diesel at about 350 °C, LHSV 1 h⁻¹, 50 bar and 300 Nm³/m³ desulfurizes to a few hundred wppm and needs 370--380 °C for ULSD -- the right order of magnitude for a CoMo catalyst. They are not Korsten & Hoffmann's, not Froment's, and not any commercial catalyst's (those are proprietary). With these constants the model gives trends and orders of magnitude. Product sulfur to 10 ppm is predictive only after `activity` and the refractory-class constants are fitted to the unit's own data (`difflow.estimation`).

(refinery-hydrotreater-specs)=
### Degrees of freedom and specs

`HydrotreaterParams`:

| Spec | Default | |
|---|---|---|
| `T_in` | (613.15,) K | bed inlet temperatures: the first only when `quench` is given, every bed's when `quench=None` |
| `quench` | (0.15,) | quench into each later bed, fraction of the treat gas; `None` to solve it from `T_in` |
| `bed_fractions` | (0.4, 0.6) | catalyst split between beds |
| `P` | 50 bar | reactor pressure (no bed pressure drop) |
| `lhsv` | 1.0 h⁻¹ | feed standard liquid volume per hour per catalyst volume |
| `catalyst_density` | 800 kg/m³ | loaded density |
| `h2_oil` | 300 Nm³/m³ | treat-gas H2 to oil, including quench |
| `purge` | 0.05 | fraction of the scrubbed separator gas purged |
| `makeup` | 97 % H2, 3 % CH4 | makeup-gas composition, `{gas: mole fraction}`; the values may be JAX tracers, so outputs are differentiable in the makeup purity (the hydrogen network's `d h2.makeup / d purity`); an array on the layout's gases is accepted too |
| `heater_efficiency` | 0.85 | charge-heater absorbed / fired duty (illustrative) |
| `heater_inlet_T` | `None` | charge-heater inlet temperature after a feed/effluent exchanger; `None`: no exchanger |
| `h2s_removal`, `nh3_removal` | 0.99, 1.0 | amine and wash-water removal fractions |
| `hps_T`, `loop_dP` | 50 °C, 3 bar | separator temperature; pressure drop round the loop (the compressor makes it up) |
| `compressor_eta` | 0.75 | isentropic efficiency |
| `stripper_feed_T`, `stripper_P`, `stripper_dP` | 230 °C, 7 bar, 1 kPa/stage | stripper |
| `steam_ratio`, `steam_T` | 0.01 kg/kg, 250 °C | stripping steam |
| `drum_T`, `drum_dP` | 40 °C, 0.3 bar | overhead drum |
| `stripper_stages` | 6 | |
| `kinetics` | `HDTKineticParams()` | rate constants and catalyst activity |

`TargetSpec(output, target)` replaces the inlet temperatures by a target on any output -- `TargetSpec("wabt", 623.15)` or `TargetSpec("product.S_wppm", 10.0, scale=10.0)` -- by shifting every bed inlet by the same amount. It is a scalar Newton solve around the whole unit, differentiated implicitly (the shift is `outputs["T_shift"]`).

**Assumptions**, beyond those of the building blocks:

- the reactor pressure is uniform (no bed pressure drop); the separator sits `loop_dP` below it and the compressor makes that up;
- the makeup gas is delivered at the recycle compressor's discharge temperature, and the treat gas and quench are at that temperature;
- bed 1's inlet temperature is a spec; the charge heater's duty is computed from it (below), and the oil's own feed temperature enters only there;
- there is no wash water: the separator decants only the water the feed brought, and NH3 removal is the amine/wash fraction;
- the HP separator vapour's cuts are knocked out back into its liquid (`knockout`), so the recycle gas is real species only;
- a cut's gravity after treatment is computed from liquid molar-volume increments per saturation step (`VOLUME_INCREMENTS`: mono-aromatic -> naphthene +20.1, di -> mono +11.0, poly -> di +11.0, olefin -> paraffin +5.5 cm³/mol), from model-compound liquid densities at 60 °F (DIPPR 105 of Perry's 8th ed., via `chemicals`); heteroatom removal is taken to change the volume by nothing; the cut's boiling point and critical constants are the feed's;
- the stripper and the overhead drum run on the treated cuts' own molecular weights.

(refinery-hydrotreater-outputs)=
### Outputs

`HydrotreaterResult.outputs` (units in `OUTPUT_UNITS`):

- product: `product.S_wppm`, `.N_wppm`, `.sg`, `.api`, `.H_wt`, `.aromatics_vol` and its `mono_`/`di_`/`poly_aromatics_vol`, `.olefins_vol`, TBP `.T05` ... `.T95` (K), `.cetane_index` (ASTM D4737 from the TBP->D86 points and density, the blend pool's functions), `.rate`, `.yield` (mass), `.volume_yield`;
- wild naphtha `naphtha.rate`, `.yield`, `.sg`, `.S_wppm`, `.T50` (on the blend grid: liquid light ends and cuts);
- `gas.yield` (mass fraction of feed): everything that leaves as gas -- C1--C4, H2S and NH3 in the off-gas, purge, acid gas and waters, the vapour of the lightest cuts in the drum's off-gas, plus the real gases dissolved in the liquid products off their blend grid -- less the makeup's own hydrocarbons; H2 and water excluded. A feed light end counts where it leaves (a C5 in the wild naphtha, a C3 mostly in the gas). Before #332 it subtracted every feed light end, including the C5s that leave as liquid, and came out at -10 % on a naphtha. `yields.total` = product + wild naphtha + gas yields, which equals `1 - feed.h2_water_rate/feed.rate` plus the chemical hydrogen as a mass fraction (tested to 1e-9);
- dissolved gases (#333): `product.dissolved_gas_rate` (zero: the stripper sends every real gas overhead) and `naphtha.dissolved_gas_rate`, kg/s of H2, H2S, NH3, C1 and C2 in the wild naphtha;
- charge heater (#332, [below](#refinery-hydrotreater-heater)): `heater.duty` (absorbed, W), `heater.fired_duty`, `heater.inlet_T`, `feed_effluent.duty`;
- hydrogen: `h2.chemical` (mol/s), `h2.chemical_nm3_m3`, `h2.chemical_scf_bbl`, `h2.chemical_wt` -- chemical consumption **from the hydrogen balance on the characterized products** (H atoms in every outlet's cuts, H2S, NH3 and light hydrocarbons less those fed, halved); `h2.consumed_by_balance` (H2 in less H2 out, equal to it to round-off); `h2.makeup`, `h2.makeup_nm3_m3`, `h2.dissolved` (H2 in the separator liquid), `h2.purge`;
- loop: `recycle.rate`, `recycle.h2_purity`, `purge.rate`, `makeup.rate`, `compressor.power`, `compressor.T_out`, `reactor.pH2_in`;
- reactor: `wabt`, `reactor.T_out`, `reactor.dT_total`, per bed `bed<k>.T_in`, `.dT`, `.quench`; `catalyst.mass`; `hds.conversion`, `hdn.conversion`;
- convergence: `tear.residual`, `stripper.residual`, `flash.residual` (the largest PR-flash residual: bed inlets, HP separator, drum); `res.converged` requires all three (flashes below `FLASH_TOL = 1e-8`) and a finite product.

`res.balances` gives the relative closure of mass, C, H, S and N over the whole unit (feed + makeup + steam = product + wild naphtha + off-gas + purge + acid gas + separator water + sour water). `res.streams` has every internal stream as `Flows`; `res.reactor` the bed profiles. `res.product_char` is a `BlendCharacterization` of the treated cuts and `res.product_stream("product")` the product as a stream on it, so the product goes straight into `BlendComponent.from_stream` for the ULSD or jet pool.

(refinery-hydrotreater-streams)=
### Product streams and dissolved gases

`res.product_stream(name, T=298.15, P=101325.0, gases=True)` returns a liquid outlet (`"product"`, `"wild_naphtha"`) as `F_<name>` flows (mol/s) on `res.product_char`, **plus**, with `gases=True` (the default), the real gases dissolved in it that are not on that grid -- H2, H2S, NH3, methane, ethane -- under their own `F_<gas>` keys. With them the stream carries its whole mass, so a balance downstream closes to round-off: `res.stream_mass(stream)` (kg/s) equals `res.streams[name].mass(unit.layout)` to 1e-13, and the grid's light-end molar masses are now the layout's (IUPAC atomic weights), not the crude unit's table, which differed in the fifth figure (#333).

- The stripper bottoms carries no real gas by construction (the stripper sends them all overhead), so `product_stream("product")` is the blend grid only either way.
- The wild naphtha does carry them (in example 40, 0.022 t/h at the naphtha unit and 0.047 t/h at the distillate unit -- most of that refinery's mass imbalance). `BlendComponent.from_stream` refuses species outside its grid, so a pool takes `product_stream("wild_naphtha", gases=False)`, and `outputs["naphtha.dissolved_gas_rate"]` is exactly the mass that drops. Or send the wild naphtha through the fractionator, which puts the gases in an off-gas.
- Water is never in these streams: the drum decants all of it.

(refinery-hydrotreater-heater)=
### The charge heater

The bed-1 inlet temperature stays the spec; the heater duty follows from it (#332). The heater takes the oil and bed 1's share of the treat gas (the treat gas less every quench) to `T_in[0]`:

```
Q_absorbed = H(bed-1 inlet, T_in) - H(heater inlet)
Q_fired    = Q_absorbed / heater_efficiency
```

`H` is the reactor's own enthalpy (`stream_enthalpy`: the PR K-values and phase split at the bed inlet, ideal-gas and Watson-liquid enthalpies on one basis), so the heater and the bed agree on what the stream holds. The heater inlet is either

- `heater_inlet_T=None` (default): no feed/effluent exchanger. The oil enters at its feed temperature (`feed["T"]`) as liquid -- it is pumped to reactor pressure and holds no hydrogen yet, so it is below its bubble point -- and the treat gas at the recycle compressor's discharge temperature (ideal gas). The heater does all the heating. `feed_effluent.duty` is 0.
- `heater_inlet_T=<K>`: the oil and gas mixed at that temperature, after a feed/effluent exchanger. `feed_effluent.duty` is that exchanger's cold-side duty. Its hot side is **not** modelled (the reactor effluent goes to the separator at `hps_T`), so a temperature cross is not checked: keep `heater_inlet_T` below the reactor outlet less an approach.

`heater_efficiency` (default 0.85, absorbed over fired) is **illustrative**: a typical fired process heater with some heat recovery, from no cited source. Feed water, if any, is taken as vapour (the reactor's convention), which understates the duty by its latent heat. Not modelled: the heater's pressure drop, coil vaporisation profile, fuel and stack.

On the test diesel (50 kg/s from 25 °C, no exchanger) the absorbed duty is in the range a sensible-heat estimate gives (tested within 0.7--1.6 of `m cp dT` at 2.6 kJ/kg/K).

(refinery-hydrotreater-fractionator)=
### The product fractionator (optional, #328)

`res.fractionate(cut_points, products, width, feeds=("product",))` (or `hydrotreating.fractionate(res, ...)`) splits the liquid product -- or the product and the wild naphtha together, `feeds=("product", "wild_naphtha")` -- into named products at TBP cut points. Like the FCC and hydrocracker fractionators it is an idealized, differentiable split, **not** a stage column. Component `i` of the product grid (boiling point `Tb_i`) goes to product `k` (lightest first, cut points `T_1 < ... < T_{n-1}`) with the share

```
a_k(Tb) = sigmoid((Tb - T_k) / w),   a_0 = 1,  a_n = 0
share_k = a_{k-1} - a_k
```

the FCC main fractionator's logistic step generalised to `n` products. The shares telescope to one, so every component's moles, and the treated attributes they carry, are conserved: the mass closure is reported (`FractionationResult.balance`, below 1e-13 in the tests). The real gases the wild naphtha carries go whole to an `off_gas` stream (`fr.off_gas`, `fr.rates["off_gas"]`), so the liquid products are blendable as they are.

- Defaults: `products=("jet", "diesel")`, `cut_points=(240 °C,)`, `w = 8 K`. All three **illustrative**: `w` stands for the overlap of a real column's neighbouring products; a column calculation would set it. A logistic step rather than the hydrocracker's quintic because the quintic is flat beyond `+/- w`, which gives an exactly zero cut-point derivative when the cut point sits midway between two 30 K grid cuts.
- `fr.products[name]` is an `F_<product_char.names>` stream, straight into `BlendComponent.from_stream(name, stream, fr.char)`; `fr.rates` is kg/s per product.
- Cut points and width are traceable: `jax.grad` of a pool property with respect to a cut point is exact, checked against Richardson-extrapolated central differences at 1e-5 (release test).

```python
fr = res.fractionate(cut_points=(240 + 273.15,), products=("jet", "diesel"))
jet = BlendPool("jet")([BlendComponent.from_stream("jet", fr.products["jet"], fr.char,
                                                   flash_C=42.0, freeze_C=-47.0, smoke_mm=22.0)], [1.0])
ulsd = BlendPool("ulsd")([BlendComponent.from_stream("diesel", fr.products["diesel"], fr.char,
                                                     flash_C=60.0)], [1.0])
```

Not modelled: duties, reflux, side-stripper steam, flash points (the pools still take measured `flash_C`), and the D86 overlap a real column produces.

(refinery-hydrotreater-naphtha)=
### A naphtha hydrotreater

The unit runs a naphtha as it is (example 40's stabilised naphtha: C3--C5 light ends and four 30 °C cuts to 180 °C, 1070 wppm S). The bed is all vapour (`V` 9--14 in the negative flash), which is where the reactor's phase model was wrong until #332 (see [the reactor](#refinery-hydroprocessing-reactor)); with it fixed, the default (diesel) constants hydrogenate as they should, but they were chosen for diesel-range sulfur and reach only about 90 wppm at 30 bar, 320 °C, LHSV 4 h⁻¹ and 100 Nm³/m³.

`NAPHTHA_HDT_PARAMS` is an **illustrative** naphtha set: the diesel constants with HDS of the sulfide, thiophene and benzothiophene classes ten times faster (in a naphtha those are mercaptans, light sulfides and alkylthiophenes; the reactivity order of Girgis & Gates 1991, the factor chosen here), HDN a hundred times faster, and every aromatics-saturation step ten times slower (a CoMo naphtha catalyst at 20--35 bar passes benzene largely unsaturated). It is not any catalyst's.

```python
from difflow_refinery.hydrotreating import NAPHTHA_HDT_PARAMS
nht = Hydrotreater(char, naphtha, HydrotreaterParams(
    T_in=(320 + 273.15,), quench=None, bed_fractions=(1.0,), P=30e5, lhsv=4.0, h2_oil=100.0,
    stripper_feed_T=150 + 273.15, kinetics=NAPHTHA_HDT_PARAMS))
```

On example 40's naphtha (single bed, stripper feed at 150 °C; the stripper's default 230 °C does not converge on a naphtha and says so):

| Bed inlet, P, LHSV, H2/oil | product S (wppm) | product N (wppm) | chemical H2 (Nm³/m³) | bed dT (K) |
|---|---|---|---|---|
| 300 °C, 30 bar, 4 h⁻¹, 100 Nm³/m³ | 0.87 | 4.6 | 1.7 | 2.1 |
| 310 °C, 30 bar, 4 h⁻¹, 100 Nm³/m³ | 0.033 | 1.7 | 1.7 | 2.2 |
| 320 °C, 30 bar, 4 h⁻¹, 100 Nm³/m³ | < 0.001 | 0.34 | 1.8 | 2.3 |
| 340 °C, 30 bar, 4 h⁻¹, 100 Nm³/m³ | < 0.001 | < 0.001 | 2.1 | 2.8 |
| 320 °C, 30 bar, 6 h⁻¹, 100 Nm³/m³ | 0.015 | 1.2 | 1.7 | 2.2 |
| 320 °C, 20 bar, 4 h⁻¹, 100 Nm³/m³ | 0.068 | 2.0 | 1.6 | 2.0 |
| 320 °C, 30 bar, 4 h⁻¹, 100 Nm³/m³, **diesel** constants | 87 | 23 | 3.0 | 4.1 |

(Feed 1072 wppm S. Figures far below 1 wppm only say "removed": the illustrative first-order classes have no refractory tail.) So a reformer feed (below about 0.5 wppm S and N) is reached at 30 bar and LHSV 4 from about 310 °C for sulfur and 320 °C for nitrogen. The charge heater takes 23.6 MW absorbed (27.8 MW fired at 0.85) at 320 °C with no feed/effluent exchanger, and the wild naphtha carries 0.011 kg/s of dissolved gas. The reformer model carries neither sulfur nor nitrogen, so nothing downstream sees these numbers. Example 40's earlier conditions (60 bar, LHSV 0.7, 400 Nm³/m³), which were what the diesel constants needed under the old phase model, now run away on the diesel constants (the aromatics saturate, the integration does not finish, `converged=False`); on the naphtha set they converge.

**Where it fails.** At 320 °C, 50 bar, LHSV 0.5 and 150 Nm³/m³ the bed-inlet PR flash does not converge (the mixture is near its critical region; Newton diverges). The unit returns `converged=False` with `flash.residual` about 1e-3 and warns; it used to raise an equinox NaN-in-linear-solve error (`tests/refinery/test_hydrotreating_naphtha.py::test_a_failed_bed_inlet_flash_reports_a_residual_instead_of_raising`, per commit, on the flash itself; `test_a_failed_flash_is_reported_not_raised`, release, on the whole unit).

(refinery-hydrotreater-results)=
### Results on the test diesel

The test crude of the composition section (SG 0.86, 1.8 wt% S, 1500 wppm N, with a heavy end), its 230--370 °C straight-run diesel at 50 kg/s (11 286 wppm S, 406 wppm N), and the defaults above (two beds, 15 % quench to the second, bed 1 inlet 340 °C, 50 bar, LHSV 1 h⁻¹, 300 Nm³/m³):

| | |
|---|---|
| WABT | 351.4 °C; bed rises 13.2 and 7.2 K |
| product | 241 wppm S, 229 wppm N, SG 0.850, 14.2 vol% aromatics, cetane index 58.4 |
| yields (mass) | product 98.74 %, wild naphtha 0.39 %, gas (C1--C4, H2S, NH3) 1.21 % |
| charge heater | 47.4 MW absorbed, 55.8 MW fired (from a 25 °C feed, no feed/effluent exchanger) |
| hydrogen | chemical 32.4 Nm³/m³ (192 scf/bbl), makeup 49.7 Nm³/m³; recycle purity 95.1 % |
| loop | recycle compressor 153 kW; purge 36 mol/s |
| closure | mass, C, H, S and N to 1e-15 relative; tear residual 4e-14; stripper 2e-12 |

Compiling the unit takes about 70 s (more on a loaded machine); a solve then takes about 1.7 s, of which the recycle tear is five Newton steps. A reverse-mode gradient of all outputs costs one more compile and 30--140 s.

**Gradients** (`tests/refinery/test_hydrotreating.py::test_gradients_match_central_differences`): product S, chemical H2 consumption and liquid yield with respect to the bed inlet temperature, the pressure, the H2/oil ratio and the 50 % TBP point of the assay, AD against central differences. With plain central differences (steps 0.5 K, 0.5 bar, 3 Nm³/m³) they agree to 1e-5 -- 9e-4, and the larger figures are the differences' own O(h²) truncation; the test compares against Richardson-extrapolated differences at `rtol` 1e-5.

**Trends** (tested): product sulfur falls with inlet temperature and with pressure (H2 partial pressure); chemical H2 consumption rises with temperature; on a bed at 30 bar, total aromatics pass through a minimum between 300 and 480 °C as the saturation equilibrium recedes.

The numbers come from illustrative rate constants: read them as the shape of the answer, not a prediction for any catalyst.

**What the tests check** (`tests/refinery/test_hydrotreating.py`; the full-unit ones are marked `slow`):

- converges from the default initialization, recycle included, on the straight-run diesel above, on a straight-run kerosene (150--250 °C) of a second, lighter and low-sulfur assay (SG 0.83, 0.35 wt% S), and on the diesel side product of a 30-stage `CrudeDistillationUnit` as it comes (stripping water and light-end traces included);
- mass, C, H, S and N close to 1e-8 relative over reactor, separator, recycle and stripper (they close to about 1e-15); the H2 consumption by the hydrogen balance equals H2 in less H2 out;
- gradients against Richardson-extrapolated central differences at 1e-5 (above); a single bed's gradients, and `newton_solve` / `solve_tear` against closed forms;
- product S falls with temperature and pressure, H2 consumption rises with temperature, aromatics pass through a minimum in temperature;
- `TargetSpec("wabt", ...)` lands on its target; `hdt_block` delta vectors pass `check_delta_vectors`;
- the product enters `BlendPool("ulsd")` through `BlendComponent.from_stream`, with the same sulfur, gravity and cetane index the unit reports;
- the liquid outlets as streams carry their whole mass (dissolved gases included), and the unit's mass balance closes to 1e-13 over them; `gases=False` drops exactly `naphtha.dissolved_gas_rate`; the yields add up (`yields.total`); the charge heater's duty is positive and of the sensible-heat order;
- the makeup composition is a traced input (`jax.jacfwd` through `theta`), and (release) `d(h2.makeup, recycle purity, product S)/d(makeup purity)` through the whole unit against Richardson differences;
- the fractionator: shares sum to one, the jet/diesel split closes to 1e-13 and its two products go into the jet and ULSD pools with their sulfur averaging back to the unit's; three products over product and wild naphtha send the dissolved gas to the off-gas; the cut-point gradient of four pool properties against Richardson differences (release);
- on example 40's naphtha (`tests/refinery/test_hydrotreating_naphtha.py`): balances, `gas.yield` positive and consistent on a feed carrying light ends, the charge heater with a feed/effluent exchanger, positive chemical hydrogen in a vapour bed, the light/heavy naphtha split; the 320 °C / 50 bar / LHSV 0.5 / 150 Nm³/m³ point returning `converged=False` with a warning instead of raising; and (release) `NAPHTHA_HDT_PARAMS` reaching below 0.5 wppm S at 30 bar, 320 °C, LHSV 4;
- pins: PR fugacity coefficients against difflow's `PengRobinson` (1e-8), the H2/H2S/NH3 Cp polynomials against PPO 5th ed. (0.5 %), every heat of reaction and the aromatic-step entropies against the model-compound table, element conservation of the kinetics at a point, and the power-law fallback.

A crude-unit product carries every cut at some trace level. `Hydrotreater(..., trace=1e-9)` leaves out cuts heavier than the heaviest one above that mole fraction, and reports what that drops as `dropped_mass_fraction` (below 1e-6 on the crude-unit diesel).

(refinery-hydrotreater-planning)=
### Planning: `hdt_block`

`difflow_refinery.hydrotreating.planning.hdt_block(unit, feed, levers, outputs)` wraps the unit as a `difflow.planning.Block`, modelled on `cdu_block`. Levers: `feed.bpd` (or `<product>.bpd` with `feed_product=`, so `link_cdu` finds it; the feed composition is held at the base and, with the catalyst volume fixed, LHSV moves with the rate), `reactor.T_in` (°C; every bed inlet moves together -- WABT is an output, write a row on it), `h2_oil`, `pressure` (bar), `purge`. Outputs (`HDT_OUTPUTS`): product S and N, gravity, cetane index, aromatics, product and wild-naphtha bbl/d, chemical H2 and makeup in Nm³/h, WABT, temperature rise, compressor kW, inlet H2 partial pressure, HDS conversion. Reverse mode only (`ad_mode="rev"` is forced): the diffrax adjoint is a `custom_vjp`.

(refinery-hydrotreater-references)=
### References

| Key | Reference | Used for | How checked |
|---|---|---|---|
| H1 | Peng, D.-Y.; Robinson, D.B. "A new two-constant equation of state." *Ind. Eng. Chem. Fundam.* **1976**, 15(1), 59--64. doi:10.1021/i160057a011 | PR EOS, mixing rules, fugacity coefficient | The equations as coded match the standard form difflow's own `PengRobinson` uses and are pinned against it in the tests. The bibliographic details are as recalled; the paper was not reached (publisher blocked) -- volume/pages/DOI unverified. |
| H2 | Robinson, D.B.; Peng, D.-Y. *The characterization of the heptanes and heavier fractions for the GPA Peng-Robinson programs*, GPA Research Report RR-28, **1978** | `kappa` for omega > 0.49 | Not checked against the report (unverified); the cubic is the form in common use. |
| H3 | Rackett, H.G. *J. Chem. Eng. Data* **1970**, 15(4), 514--517, doi:10.1021/je60047a012; Spencer, C.F.; Danner, R.P. *J. Chem. Eng. Data* **1972**, 17(2), 236--241, doi:10.1021/je60053a012; Yamada, T.; Gunn, R.D. *J. Chem. Eng. Data* **1973**, 18(2), 234--236, doi:10.1021/je60057a006 | Liquid molar volume of the cuts | Equation form as in Poling, Prausnitz & O'Connell 5th ed. ch. 4 (recalled); citations unverified. |
| H4 | Whitson, C.H.; Michelsen, M.L. "The negative flash." *Fluid Phase Equilib.* **1989**, 53, 51--71. doi:10.1016/0378-3812(89)80072-X | Negative flash | Unverified (not reached). |
| H5 | Rachford, H.H.; Rice, J.D. *J. Petrol. Technol.* **1952**, 4(10), sec. 1 p. 19, sec. 2 p. 3 | Rachford-Rice | Unverified. |
| H6 | Michelsen, M.L. "The isothermal flash problem. Part II. Phase-split calculation." *Fluid Phase Equilib.* **1982**, 9(1), 21--40 | SS then Newton in ln K | Unverified. |
| H7 | Korsten, H.; Hoffmann, U. "Three-phase reactor model for hydrotreating in pilot trickle-bed reactors." *AIChE J.* **1996**, 42(5), 1350--1360. doi:10.1002/aic.690420515 | LHHW HDS form with squared H2S inhibition | Title, journal and year confirmed by web search (citing literature); volume/issue/pages/DOI as recalled (unverified). Their rate constants, solubility and density correlations are **not** used; their published profiles are **not** reproduced (below). |
| H8 | Froment, G.F.; Depauw, G.A.; Vanrysselberghe, V. "Kinetic modeling and reactor simulation in hydrodesulfurization of oil fractions." *Ind. Eng. Chem. Res.* **1994**, 33(12), 2975--2988 | LHHW HDS by sulfur class | Title, pages and DOI not verified (unverified). Form only; constants not used. |
| H9 | Vanrysselberghe, V.; Froment, G.F. "Hydrodesulfurization of dibenzothiophene on a CoMo/Al2O3 catalyst: reaction network and kinetics." *Ind. Eng. Chem. Res.* **1996**, 35(10), 3311--3318 | DBT network (DDS/HYD routes), LHHW form | Web search did not return the paper; details unverified. Qualitative use only. |
| H10 | Girgis, M.J.; Gates, B.C. "Reactivities, reaction networks, and kinetics in high-pressure catalytic hydroprocessing." *Ind. Eng. Chem. Res.* **1991**, 30(9), 2021--2058 | Class reactivity order; DBT routes | Unverified. Qualitative use only. |
| H11 | Mederos, F.S.; Elizalde, I.; Ancheyta, J. "Steady-state and dynamic reactor models for hydrotreatment of oil fractions: a review." *Catal. Rev. Sci. Eng.* **2009**, 51(4), 485--607 | Background: lumped HDS/HDN/HDA models | Title confirmed by web search (publisher listing); volume/pages as in the issue, DOI unverified. Not used for numbers. |
| H12 | Bell, C. et al., `chemicals` (Python package) v1.5.2 and `thermo` v0.6.1 | Model-compound Hf, S°; liquid densities (DIPPR 105 from Perry's 8th ed.); dHvap at Tb (CRC); ChemSep PR k_ij; cross-check of the RPP Cp polynomials | Values read from the packages' tables (offline). The packages transcribe primary sources; the primary source of each number was not checked (unverified). |
| H13 | Reid, R.C.; Prausnitz, J.M.; Poling, B.E. *The Properties of Gases and Liquids*, 4th ed., McGraw-Hill, **1987**, App. A | Ideal-gas Cp of H2, H2S, NH3 | Coefficients as recalled; checked against the PPO 5th ed. polynomials (via `chemicals`): within 0.5 % at 300--700 K (pinned in tests). |
| H14 | Poling, B.E.; Prausnitz, J.M.; O'Connell, J.P. *The Properties of Gases and Liquids*, 5th ed., McGraw-Hill, **2001**, App. A | Light-end constants (via the crude unit's table) | As the crude unit cites them. |
| H15 | ASTM D4737 (four-variable cetane index) | Product cetane index | The blend pool's function, as cited there. |
| H16 | Wieser, M.E.; Berglund, M. "Atomic weights of the elements 2007." *Pure Appl. Chem.* **2009**, 81(11), 2131--2156 | Atomic masses | Values are the standard ones; citation details unverified. |

(refinery-hydrotreater-not-done)=
### What is not done

- **The Korsten & Hoffmann (1996) profile cross-check is not done.** Their paper could not be reached from here (publisher sites are blocked), so neither their parameters nor their profiles could be verified, and no reproduction is claimed. Their gas-liquid / liquid-solid film model and their Henry's-law and Standing-Katz density correlations are not implemented either.
- **Smoke point** is not computed. The correlation the issue names (Riazi MNL50, Tb and SG) could not be verified; a jet pool takes a measured `smoke_mm` override.
- **No dedicated example notebook.** `examples/40_refinery_flowsheet.ipynb` runs the unit twice inside a whole refinery: as a naphtha hydrotreater ahead of the reformer, and as a distillate hydrotreater on the CDU's kerosene, diesel and AGO, whose product goes to the jet and ULSD pools through `res.product_stream()` and `res.product_char`.
- **Not a difflow `Flowsheet` object.** The recycle is solved by the unit's own tear (above); a `Flowsheet` wiring of the same pieces is not provided.
- **Commercial catalyst kinetics** are not reproduced and must not be implied: the rate constants are illustrative.
- **No dissolved-gas effect in the stripper**: the real gases leave with the overhead without taking part in the column's equations.
- **The product fractionator is a TBP split, not a column** (no duties, reflux or steam; the overlap width is illustrative). A `StageColumn` fractionator is not built.
- **The charge heater has no feed/effluent exchanger hot side**: `heater_inlet_T` is taken as given and a temperature cross is not checked; no heater pressure drop, coil profile or fuel balance.
- **Near a mixture's critical region** (a naphtha with little treat gas at 50--60 bar) the bed-inlet PR flash can fail; the unit then returns `converged=False`. A stability test and a critical-point-robust flash are not implemented.
- **Deactivation tracking** with `difflow.reconciliation.tracking` is possible (the activity is a parameter with a gradient) but not demonstrated.
- Out of scope (as the issue says): residue hydrotreating/HDM, countercurrent reactors, reactor internals and pressure drop, amine unit detail, dynamics.
(refinery-alkylation)=
## Alkylation

`difflow_refinery.alkylation` (#310) combines isobutane with C3-C5 olefins over sulfuric or hydrofluoric acid, settles the acid out, and fractionates the effluent: a deisobutanizer (DIB) whose overhead is the isobutane recycle, a depropanizer on a slipstream of that recycle, and a debutanizer. The products are alkylate (to `BlendPool`), propane and n-butane. It is differentiable in the isobutane/olefin ratio, reactor temperature, acid strength, space velocity and olefin feed rate and composition, through the recycle.

```python
import difflow_refinery.alkylation as al

unit = al.AlkylationUnit()                      # defaults: I/O 8, 10 C, 89 wt% H2SO4
res = unit.solve(al.c3c4_olefin_feed())         # or al.combine_feeds(fcc_c3, fcc_c4)
res.outputs["alkylate.MON"], res.outputs["dib.reboiler"]
alkylate = res.alkylate_component(S_ppm=5.0)    # a BlendComponent for BlendPool

blk = al.alky_block(unit, feed, levers=["io_ratio", "reactor.T", "acid_strength"])
al.solve_process_gms()                          # the Bracken-McCormick problem, profit 1161.3366
```

Like the blend pool, this is a library, not a palette operation: the plugin registers no new entry point. Tests: `tests/refinery/test_alkylation.py`. No example notebook was written (see [Alkylation: not done](#refinery-alkylation-not-done)).

**The correlations are not predictive.** The yield, octane and acid correlations are regressions on one 1960s sulfuric-acid plant. The defaults are illustrative. Refit them to the unit's own data (`difflow.estimation`) before the model is used to plan.

### Alkylation: the flowsheet

```
olefin feed --+
makeup iC4 ---+--> makeup --> reactor --> DIB --+--> bottoms --> DeC4 --> alkylate
              ^                               |  overhead          \--> n-butane
              |                               v  (tear)
              +----------- direct -------- splitter
              |                               | slipstream
              +------ DeC3 bottoms <------- DeC3 --> propane
```

A difflow `Flowsheet`. The tear is the DIB overhead (`dib_recycle`). It is solved by Anderson acceleration in a forward solve, and under `jax.grad`/`jax.jacfwd` by the flowsheet's `optimistix` fixed point, differentiated implicitly. Both default feeds converge from the default initialisation in 5-6 tear iterations. That initialisation is the isobutane the I/O ratio needs, less the feed's, plus the propane the purge will hold.

- **Makeup** (`IsobutaneMakeup`) mixes the fresh feed, both recycle branches and makeup isobutane. It adds the makeup that brings the reactor feed to the specified external I/O ratio. The ratio is a spec and the makeup rate is computed.
- **Reactor** (`AlkylationReactor`), with an ideal acid settler. The acid phase is not carried. The acid consumed is reported as an operating cost.
- **DIB** on the reactor effluent. Isobutane and the propane go overhead, n-butane and alkylate to the bottoms.
- **Depropanizer** on a fraction (`depropanizer_fraction`, 0.3) of the DIB overhead. It rejects propane, and its bottoms rejoin the recycle. At steady state the recycle carries `F_C3 / (fraction x recovery)` of propane.
- **Debutanizer** on the DIB bottoms. n-butane goes overhead, alkylate to the bottoms.

**Feeds.** The feed is a plain difflow stream of real species (mol/s). Any subset of `ALKYLATION_SPECIES` is accepted, with absent species counted as zero. The FCC of #308 emits its `c3` (propane, propylene) and `c4` (isobutane, n-butane, 1-butene, isobutylene, cis- and trans-2-butene) outlets in exactly these names. `al.combine_feeds(c3, c4)` makes them one feed, and a test runs the unit on that union. The two example feeds, `c3c4_olefin_feed` and `c4_olefin_feed`, have **illustrative** compositions that are not taken from a source.

### Alkylation: species and properties

All species are real molecules in `difflow.database`.

| Role | Species |
|---|---|
| Olefins | propylene; 1-butene, cis-2-butene, trans-2-butene, isobutylene; 1-pentene, 2-methyl-2-butene |
| Isoparaffin | isobutane |
| Inerts | propane, n-butane, isopentane, n-pentane |
| C7 alkylate (from propylene) | 2,3-dimethylpentane, 2,4-dimethylpentane |
| C8 alkylate (from butenes) | 2,2,4-trimethylpentane, 2,3,4-trimethylpentane, 2,5-dimethylhexane |
| C9 alkylate (from amylenes) | 2,2,5-trimethylhexane |
| Heavy ends | a C12 lump with n-dodecane's properties |

The heavy-end lump is a *property surrogate*. Dimer-alkylate heavy ends are C12 (2 C4= + iC4 → C12H26), and n-dodecane is the C12 paraffin that every property table carries. Its octane never enters the model.

Added to `difflow.database` for this issue: 2,3- and 2,4-dimethylpentane, 2,2,5-trimethylhexane and n-dodecane (both tables), and ideal-gas data for 1-butene. They were obtained and checked the same way as the #305 isomers. Each value is read from the data tables of the `chemicals` package (v1.5.2) and accepted where independent tables agree. Values on which the tables disagree are marked "(unverified)" in `SOURCE_CITATIONS`:

- 2,3-dimethylpentane Hf: CRC -198.7 kJ/mol, API TDB -194.1 kJ/mol.
- 2,2,5-trimethylhexane Pc: no IUPAC value.
- 2,2,5-trimethylhexane ω: 0.345 to 0.357.
- n-dodecane ω: 0.562 to 0.576.

What the alkylation model adds (`alkylation.species`):

- **Molar mass** from the formula and the IUPAC atomic weights. The database values are rounded to 0.01 g/mol, and with them a mass balance across C5= + iC4 → C9 would close only to about 1e-5.
- **Standard volumes at 60 °F** by COSTALD (Hankinson & Thomson 1979). The characteristic volumes `V*` and `ω_SRK` are the published fitted parameters. For 2,3,4-TMP, 2,5-DMH and 2,2,5-TMH, which have none, `V*` is fitted to the CRC density at 20 °C. Checked against GPA 2145: propane, isobutane and n-butane come out at 0.5073, 0.5625 and 0.5844 against 0.50736, 0.56293 and 0.58407, and isopentane and n-pentane agree within 0.4 %. Checked against CRC at 20 °C: within 1.5 % for every species with tabulated parameters, the worst being 2,3-DMP at 1.44 %.
- **Liquid heats of formation**, `Hf(l) = Hf(g) - ΔHvap(298 K)`. `Hf(g)` comes from the database and ΔHvap from CRC. Checked against CRC's own liquid Hf for 1-butene, 2,3-DMP, 2,4-DMP, 2,2,4-TMP and n-dodecane: all within 0.8 kJ/mol.

### Alkylation: the reactor

Every olefin `C_nH_2n` reacts with isobutane by one of two routes. Both conserve C and H exactly, so mass balances by construction:

| Route | Stoichiometry | Products |
|---|---|---|
| Alkylation | `C_nH_2n + iC4H10 → C_(n+4)H_(2n+10)` | per-olefin selectivity table (`DEFAULT_SELECTIVITY`) |
| Heavy ends | `(8/n) C_nH_2n + iC4H10 → C12H26` | the C12 lump |

Conversion is complete by default (`olefin_conversion = 1`). **The split between the routes is what the correlation sets.** The Sauer-Colville-Burwick yield `Y(r) = 1.12 + 0.13167 r - 0.00667 r²` (vol alkylate per vol olefin, `r` the external I/O ratio by 60 °F volume) peaks at `Y_max = 1.7698` at `r = 9.87`. The model reads `ε = Y(r)/Y_max` as an alkylation efficiency. For each olefin it then sends to heavy ends the fraction `h_j` that makes that olefin's volume yield `ε` times its all-alkylation yield:

```
(1 - h_j) Y_A,j + h_j Y_H,j = ε Y_A,j     =>     h_j = Y_A,j (1 - ε) / (Y_A,j - Y_H,j)
```

Here `Y_A,j` and `Y_H,j` are the stoichiometric 60 °F volume yields of the two routes. The butenes' `Y_A` are 1.73 to 1.81, within 2.5 % of `Y_max`, so on a butene feed the reactor reproduces the correlation's yield to about 1 % (tested). Propylene (1.80) and the amylenes (1.67 to 1.72) follow the correlation's *shape* about their own stoichiometric yields. **This mapping is this module's modelling choice, not part of the published correlation.** The default selectivities are illustrative too:

- isobutylene and 2-butenes go mostly to TMPs;
- 1-butene gives more 2,5-DMH;
- propylene gives 60/40 2,3-/2,4-DMP;
- amylenes give 2,2,5-TMH.

They are not from a source.

**Octane.** The motor octane number is the correlation's, plus two linear corrections:

```
MON = 86.35 + 1.098 r - 0.038 r² - 0.325 (89 - S) + c_T (T - T_ref) + c_SV (SV - SV_ref)
```

Here `S` is the acid strength (wt%), `T` the reactor temperature and `SV` the olefin space velocity. **The temperature and space-velocity terms are not in Sauer et al., and their defaults are assumptions of this module:**

- `c_T = -0.1` MON/K about `T_ref = 283.15 K` (about -0.55 per 10 °F);
- `c_SV = -2` MON per v/h/v about 0.3.

Set both to zero to recover the published correlation exactly; a test checks that. RON is `MON + 2.5`. The 2.5 is the sensitivity of the illustrative alkylate in `BlendComponent`'s example (RON 96, MON 93.5), and it is unverified.

**Acid.** For H2SO4, the acid consumed per bbl of alkylate is `dilute · S / (98 - S)`, with `dilute = 35.82 - 0.222 F4` and `F4 = -133 + 3 MON` (Sauer et al., as in `process.gms`). A temperature rise therefore raises acid consumption through the octane. HF has no open correlation. `acid="HF"` requires `hf_acid_lb_per_bbl` from the unit's data and raises without it. It uses the H2SO4 yield and octane correlations as they stand.

**Heat.** The heat of alkylation comes from Hess's law on the liquid heats of formation at 298.15 K. The temperature dependence of the heat of reaction between 298 K and the reactor is neglected. Values:

- isobutylene + iC4 → 2,2,4-TMP: -67.9 kJ/mol olefin;
- trans-2-butene: -71.5 kJ/mol;
- propylene: -86.3 kJ/mol.

The refrigeration duty is that heat plus the sensible heat of cooling the reactor feed to `T` (Peng-Robinson liquid enthalpy, `CubicThermo`, kij = 0). It is also reported as the isobutane vaporised to remove it (Watson's latent heat from the CRC value at Tb).

**Range.** `process.gms` bounds the regression variables to where the plant ran: `r` in [3, 12], `S` in [85, 93], MON in [90, 95]. Outside them the reactor raises `AlkylationRangeWarning`. The yield quadratic, for one, falls again above `r = 9.87`.

### Alkylation: fractionation

This unit was built before the gas plant's cubic-EOS columns (`difflow_refinery.gasplant`, #312) reached `main`; moving the fractionation onto them is follow-up work. All three columns are Fenske-Underwood-Gilliland **shortcut columns** (`KeySplitColumn`), each specified by its two key recoveries and its reflux ratio:

| Step | Equation | Source |
|---|---|---|
| Volatility | `α_i = Psat_i(T_col)/Psat_HK(T_col)`, Raoult, Lee-Kesler `Psat` from the database (Tc, Pc, ω); `T_col` where `Psat_LK Psat_HK = P²` | Lee & Kesler (1975) |
| Non-key split | `log(d_i/b_i) = A + C log α_i`, `A = log(d_HK/b_HK)`, `C = [log(d_LK/b_LK) - A]/log α_LK`; fraction overhead `sigmoid(A + C log α_i)` | Geddes (1958); Hengstebeck (1961) |
| Minimum stages | `N_min = log[(d/b)_LK (b/d)_HK]/log α_LK` | Fenske (1932) |
| Minimum reflux | `Σ α_i z_i/(α_i - θ) = 0` (q = 1), root just above the heavy key; `R_min + 1 = Σ α_i x_D,i/(α_i - θ)` | Underwood (1948) |
| Stages | `Y = 1 - exp[(1 + 54.4X)/(11 + 117.2X)·(X - 1)/√X]` | Gilliland (1940), Molokanov et al. (1972) form |
| Duties | CMO, saturated-liquid feed and products: `V = (R+1)D`; condenser `V·Σ x_D λ(T_top)`, reboiler `V·Σ x_B λ(T_bot)`; `T_top`, `T_bot` the Raoult bubble points of the products; λ by Watson from CRC ΔHvap(298 K) | Watson (1943) |

The only inner solves are two fixed-length bisections, each followed by one Newton step that attaches the implicit-function gradient. Column-level AD matches central differences to 1e-5 relative (tested).

| Column | Keys | Recoveries (LK up / HK down) | R | P (bar) | On the C3/C4 feed: N_min, R_min, N |
|---|---|---|---|---|---|
| Depropanizer | propane / isobutane | 0.95 / 0.99 | 40 | 17 | 8.8, 8.4, 9.9 |
| DIB | isobutane / n-butane | 0.97 / 0.85 | 2.5 | 7 | 16.6, 2.03, 35.6 |
| Debutanizer | n-butane / isopentane | 0.95 / 0.90 | 1.5 | 5 | 6.2, 0.53, 9.5 |

These defaults are illustrative design choices, made so that the reflux is above the Underwood minimum on both example feeds. On the C4-only feed the depropanizer's R_min is 34, because the slipstream is lean in propane. `columns_feasible` reports `R > R_min` for every column.

**Two things found on the way.**

1. **difflow's `ShortcutColumn` had a sign error in its Hengstebeck-Geddes constants (now fixed).** It used `A = log(d_LK/b_LK) - log(d_HK/b_HK)` and `C = log(d_LK/b_LK)/log α_LK`. That does not reproduce the heavy key's own split, and on a propane/isobutane depropanizer it sent 99.9 % of the n-butane overhead. This unit first worked around it with an override, `GeddesShortcutColumn`. The base class is now fixed (`A = log(d_HK/b_HK)`, `C = [log(d_LK/b_LK) - log(d_HK/b_HK)]/log α_LK`, the line through both keys). `GeddesShortcutColumn` remains as an alias, and `tests/test_distillation.py::TestShortcutColumnNonKeyDistribution` is the regression test.
2. **`ShortcutColumn` with Peng-Robinson is too expensive to differentiate through the recycle.** Three of them, each with nested Newton bubble-point solves inside the tear's implicit fixed point, exhausted this machine's memory. They remain available for forward cross-checks as `AlkylationUnitParams(fractionation="pr_shortcut")`.

Measured on the C3/C4 feed against `pr_shortcut` with the same specs (`test_peng_robinson_shortcut_cross_check`):

- product flows and the alkylate RVP agree to 0.2 %;
- condenser duties agree to 1 %;
- reboiler duties differ by up to 25 % (the DIB's: 33.6 MW CMO against 26.8 MW by the PR energy balance), because the CMO duty uses the bottoms' latent heat and neglects sensible heat;
- `R_min` and `N_min` differ by up to a factor of two between Raoult/Lee-Kesler and PR volatilities.

**Treat the reboiler duties as order-of-magnitude** until the fractionation is moved onto the gas plant's rigorous columns (#312).

### Alkylation: specs and outputs

| Degree of freedom | Where | Default |
|---|---|---|
| External I/O ratio (sets makeup, hence recycle) | `makeup.io_ratio` | 8 |
| Reactor temperature | `reactor.T` | 283.15 K |
| Acid strength | `reactor.acid_strength` | 89 wt% |
| Olefin space velocity | `reactor.space_velocity` | 0.3 1/h |
| Olefin feed rate and composition | the feed stream | |
| Key recoveries and reflux of each column | `deisobutanizer`, `depropanizer`, `debutanizer` | table above |
| Depropanizer slipstream | `depropanizer_fraction` | 0.3 |

The debutanizer is specified by its n-butane recovery, not by the alkylate RVP. RVP is an output, and an RVP target is a row on `alkylate.RVP_psi` (the same argument as `cdu_block`'s for cut points). The issue's other suggested specs are not implemented as specs: DIB overhead purity and acid/hydrocarbon ratio.

Outputs (`OUTPUT_UNITS`), from the two default feeds at the default specs:

| Output | Units | C4 feed | C3/C4 feed |
|---|---|---|---|
| `olefin.bpd` | bbl/d | 2709 | 2921 |
| `alkylate.bpd` | bbl/d | 4840 | 5234 |
| `alkylate.yield` (debutanized alkylate per vol olefin, incl. feed C5s) | - | 1.787 | 1.792 |
| `alkylate.yield_correlation` | - | 1.746 | 1.746 |
| `alkylate.MON` / `RON` | - | 92.7 / 95.2 | 92.7 / 95.2 |
| `alkylate.SG` | - | 0.704 | 0.700 |
| `alkylate.RVP_psi` (Raoult, D323 bomb, Lee-Kesler Psat) | psi | 2.39 | 2.66 |
| `alkylate.T10/50/90_tbp` | °C | 92 / 106 / 120 | 79 / 99 / 120 |
| `isobutane.consumed_bpd` / `makeup_bpd` / `recycle_bpd` | bbl/d | 2969 / 1730 / 18 140 | 3356 / 2778 / 19 410 |
| `propane.bpd`, `n_butane.bpd` | bbl/d | 102, 1083 | 437, 968 |
| `acid.lb_per_bbl`, `acid.klb_d` | lb/bbl, 1000 lb/d | 35.7, 173 | 35.7, 187 |
| `reactor.heat`, `refrigeration.duty` | MW | 4.0, 7.0 | 4.8, 8.0 |
| `dib.reboiler`, `dib.condenser` | MW | 29.9, 20.0 | 33.6, 22.8 |
| `dec3.*`, `dec4.*` reboiler/condenser | MW | 1.1/1.1, 1.3/0.9 | 5.0/4.8, 1.2/0.8 |

D86 points (`T10/50/90_d86`) come from the TBP points by Riazi-Daubert. That correlation was fitted on petroleum fractions. On this narrow-boiling alkylate it gives a D86 T10 above T50 (111 against 106 °C on the C4 feed), so read the TBP points. The alkylate here has no C5-C7 light ends from cracking or hydrogen transfer, which a real alkylate has, so its front end is too heavy.

**Trends** (tested, C4 feed):

- MON rises with I/O and falls with temperature;
- DIB duty rises with I/O;
- acid consumption falls with I/O.

**Gradients** (tested): the implicit gradients of alkylate yield, MON and DIB reboiler duty with respect to I/O ratio, temperature and acid strength, taken by `jax.jacfwd` through the recycle, match central differences to 1e-5 relative. At the C4 base point:

| Output | d/d(I/O) | d/dT (per K) | d/dS (per wt%) |
|---|---|---|---|
| Yield | 0.0250 | 0 | 0 |
| MON | 0.490 | -0.1 | 0.325 |
| DIB reboiler | 3.65 MW | 0 | 0 |

The zeros are structural: in the correlations neither yield nor flows depend on T or S.

### Alkylation: the correlation layer, cross-checked against process.gms

`alkylation.correlations` transcribes the Sauer-Colville-Burwick regressions from the GAMS model `process.gms`. The source was read from GAMS's own GAMSPy translation (`GAMS-dev/gamspy-examples`, `models/process/process.py`; gams.com itself was not reachable). Every coefficient, bound, price and starting level is pinned in the tests. `ratio` is labelled "isobutane makeup to olefin ratio" in `process.gms`, but it is *defined* as `(isor + isom)/olefin`, the external I/O ratio.

`solve_process_gms()` solves the problem with difflow's own gradient-based solver: the JAX primal-dual interior-point method of `difflow_power.ipm`, with exact Hessians, on scaled variables, from `process.gms`'s starting point.

| Model | Published | difflow | Notes |
|---|---|---|---|
| `process` (regressions exact) | 1161.33660200 (MINLPLib primal bound for instance `process`, minimisation sign) | **1161.336602**, converged in 10 iterations, residual < 1e-9 | reproduced to the digits given; olefin 1728.92, isor 16000 (bound), isom 2000 (bound), acid 98.161, alkylate 3056.49, strength 90.616, octane 94.188, ratio 10.411, dilute 2.617, f4 149.563 |
| `rproc` (each regression ±10 %) | none found | 2410.83 | not checked against a published number |

The MINLPLib value was read from a search-engine summary of minlplib.org, which itself was not reachable. The 1968 book was not opened.

### Alkylation: planning

`alky_block(unit, feed, levers, outputs)` is a `difflow.planning.Block`, like `cdu_block`. It has these levers:

- `io_ratio`;
- `reactor.T` (°C);
- `acid_strength` (wt%);
- `space_velocity` (1/h);
- `olefin.bpd`, the feed rate with composition held.

Any `OUTPUT_UNITS` name can be an output. The `alkylate.bpd`, `alkylate.RON`, `alkylate.MON` and `alkylate.RVP_psi` outputs link to a blend-pool component, and `result.alkylate_component()` gives the same properties as a `BlendComponent`. `jit=False` is the default: compiling the traced recycle takes minutes. A forward evaluation is about 20 s eager, and a 3-lever Jacobian by `jacfwd` about 35 s.

(refinery-alkylation-not-done)=
### Alkylation: not done, and why

- **Rigorous fractionation (#312).** Built before #312's gas-plant columns reached `main`; shortcut columns stand in (above) until they are wired in. The reboiler duties are uncertain to about 25 %.
- **Kinetic option.** The carbocation schemes of Langley & Pike (1972) and Lee & Harriott (1977) are not implemented. The issue lists them as non-default; the papers were not available to transcribe.
- **Per-olefin yields, isobutane consumption and octanes from Gary, Handwerk & Kaiser.** The table could not be consulted. The per-olefin selectivities and the octane corrections are labelled illustrative instead.
- **Pure-component octanes (API RP 45).** Not used. The alkylate octane is the correlation's.
- **Example notebook** (FCC LPG → alkylation → gasoline pool). Not written. A forward solve takes about 20 s, plus a minute of compilation on first use.
- **HF acid consumption, ASO make, and selectivity against mixing.** There is no open model (class (d)). The HF figure is a required input.
- Out of scope per the issue: acid regeneration, HF mitigation, solid-acid and ionic-liquid alkylation, contactor hydrodynamics, dynamics.

### Alkylation: references

| What | Source | Checked |
|---|---|---|
| Yield, MON, F-4, dilution, acid and makeup equations and coefficients; bounds; prices | GAMS Model Library `process.gms` (SEQ=20), "Alkylation Process Optimization", read via GAMS's GAMSPy translation `GAMS-dev/gamspy-examples/models/process/process.py` | transcribed and pinned in tests; optimum reproduced |
| The correlations' origin | Sauer, R.N., Colville, A.R., Burwick, C.W., "Computer points the way to more profits", *Hydrocarbon Processing* 43(3), 84 (1964) | **unverified**: not opened; volume, issue and page as given in the issue |
| Their restatement as an NLP | Bracken, J., McCormick, G.P., *Selected Applications of Nonlinear Programming*, Wiley, New York (1968), Ch. 4 | as cited by `process.gms`; book not opened |
| Reference optimum 1161.33660200 | MINLPLib, instance `process` (primal bound) | **from a search summary**; minlplib.org not reachable |
| Mechanistic kinetics (not implemented) | Langley, J.R., Pike, R.W., "The kinetics of alkylation of isobutane with propylene", *AIChE J.* 18(4), 698-705 (1972); Lee, L.M., Harriott, P., "The kinetics of isobutane alkylation in sulfuric acid", *Ind. Eng. Chem. Process Des. Dev.* 16(3), 282-287 (1977) | journal, volume, issue and pages confirmed by web search; DOIs not confirmed and so not given |
| COSTALD liquid volume | Hankinson, R.W., Thomson, G.H., "A new correlation for saturated densities of liquids and their mixtures", *AIChE J.* 25(4), 653-663 (1979), doi:10.1002/aic.690250412 | citation confirmed by web search; coefficients checked against the `chemicals` implementation and its API Technical Data Book propane example (530.30 kg/m³, reproduced to 1e-12) |
| COSTALD parameters V*, ω_SRK | Hankinson & Thomson (1979), as tabulated in `chemicals` 1.5.2 "COSTALD Parameters.tsv" | table number unverified; validated against GPA 2145 SGs (below) |
| Standard SGs of light ends (check) | GPA 2145, as in `difflow_refinery.assay.LIGHT_END_SG` | edition as in that module |
| Tb, ΔHvap(298 K), ΔHvap(Tb); liquid Hf and 20 °C densities (checks) | CRC Handbook of Chemistry and Physics, tables "Enthalpy of Vaporization", "Standard Thermodynamic Properties of Chemical Substances", "Physical Constants of Organic Compounds", as transcribed in `chemicals` 1.5.2 | edition unverified |
| Isobutylene ΔHvap | Perry's Chemical Engineers' Handbook, Table 2-150 (C1 = 32614 J/mol, C2 = 0.38073) | edition unverified |
| Gas Hf (check) | API Technical Data Book (Albahri), as in `chemicals` 1.5.2 "API TDB Albahri Hf (g).tsv"; ATcT 1.112 for 1-butene | all within 1.5 kJ/mol except 2,3-DMP (flagged) |
| New database species (Tc, Pc, ω, Cp, Antoine, ΔHvap, Hf) | as for the #305 isomers: IUPAC critical reviews / PPO 5e App. A, TRC Cp fits, PPO Antoine, CRC; see `difflow.database.SOURCE_CITATIONS` | Cp(298) against the Poling databank to 1.5 %, Antoine Tb to 2 %, Lee-Kesler Psat(Tb) to 6 %, all tested |
| Lee-Kesler vapour pressure | Lee, B.I., Kesler, M.G., *AIChE J.* 21(3), 510 (1975) (`difflow_refinery.correlations.vapor_pressure`) | as in that module |
| Watson latent heat | Watson, K.M., *Ind. Eng. Chem.* 35, 398 (1943) | as commonly cited; not opened |
| Fenske, Underwood, Gilliland/Molokanov, Geddes, Hengstebeck | Fenske, *Ind. Eng. Chem.* 24, 482 (1932); Underwood, *Chem. Eng. Prog.* 44, 603 (1948); Gilliland, *Ind. Eng. Chem.* 32, 1220 (1940); Molokanov et al., *Int. Chem. Eng.* 12, 209 (1972); Geddes, *AIChE J.* 4, 389 (1958); Hengstebeck, *Distillation*, Reinhold (1961) | as commonly cited; not opened; equations checked by their defining properties in the tests |
| Reid vapour pressure | `difflow_refinery.blending.raoult_rvp` (ASTM D323 bomb, V/L = 4, 100 °F) | as in that module |
| TBP → D86 | Riazi & Daubert (1986), `difflow_refinery.blending.TBP_D86` | as in that module |
| Temperature and space-velocity octane terms; per-olefin selectivities; RON - MON = 2.5; example feeds; column specs | **assumptions of this module, not from a source** | - |

---

(refinery-hydrocracker)=
## The hydrocracker

`difflow_refinery.hydrocracking.Hydrocracker` is a single-stage, series-flow VGO hydrocracker on the shared hydroprocessing building blocks ([above](#refinery-hydroprocessing-layout)): a pretreat reactor (the hydrotreating kinetics with a VGO parameter set), a cracking reactor (a new kinetic model plugged into the same `TrickleBedReactor`), effluent cooler and HP separator, the recycle-gas loop (knock-out, amine, purge, compressor, makeup to an H2/oil spec), a product fractionator, and an optional recycle of unconverted oil (UCO) to the cracking reactor. Like the hydrotreater it is a library, not a palette operation.

```python
import difflow_refinery as dr
from difflow_refinery.hydrocracking import Hydrocracker, HydrocrackerParams

char = dr.characterize(assay, composition=True)        # #305 composition is required; heavy end recommended
vgo = {**lvgo, **{k: lvgo.get(k, 0.0) + hvgo[k] for k in hvgo if k.startswith("F_")}}   # VDU LVGO + HVGO
hcu = Hydrocracker(char, vgo, HydrocrackerParams(T_crack=643.15, uco_recycle=0.5))
res = hcu.solve(vgo)
print(res.table())
res.outputs["conversion.per_pass"], res.outputs["kerosene.yield"], res.outputs["h2.chemical_nm3_m3"]
```

```
fresh VGO -> [pretreat beds] -> (+ UCO recycle) -> [cracking beds] -> cooler -> HPS -> fractionator
                 ^ quench                ^ quench to each bed inlet            |        |   off-gas, LPG, LN, HN,
treat gas -------+-----------------------+                      recycle gas <--+        |   kerosene, diesel, UCO
    ^ makeup H2  <- compressor <- purge <- amine <- KO drum                             +-> UCO bleed / recycle
```

The unit carries every pseudo-component from the lightest up to the heaviest one in the feed above `trace` of its pseudo-component mass (default 1e-4; a VDU product carries every cut at some trace level, and what is left out is `dropped_mass_fraction`). A VDU VGO goes in as it is (`F_<char.names>`); `hydrotreating.straight_run_cut(char, 370 + 273.15, 560 + 273.15, rate)` is an idealized one.

(refinery-hydrocracker-layout)=
### Stream, feed and pretreat bed

The layout is the hydrotreater's (`HDT_ATTRIBUTES`: C and H atoms, S in five classes, N in two, mono/di/poly-aromatic, olefinic and naphthenic molecule counts) plus one attribute, `"cracked"`: the number of molecules in the cut that are cracked *product*. It rides with the molecules like every attribute and is what lets the unit compute the gravity of a cut holding both feed and cracked molecules (below). The gases are the hydrotreater's plus isopentane and n-pentane, which cracking makes. The feed is read exactly as the hydrotreater reads one (`hcu_feed` = `hdt_feed` plus a zero `"cracked"` column).

The **pretreat bed** is `HDTKinetics` unchanged, with `VGO_PRETREAT_PARAMS`: the hydrotreater's illustrative constants with HDN made several times faster (a NiMo pretreat catalyst at 150 bar is chosen for HDN, which is what protects the cracking catalyst) and aromatics saturation slower. The only change to the hydrotreating code is that `HDTKinetics` now accepts a layout whose attributes *begin* with `HDT_ATTRIBUTES` (the cracking leak carries extra attributes with the molecule).

(refinery-hydrocracker-kinetics)=
### Cracking kinetics

`HCKinetics` is a kinetic model for `TrickleBedReactor` (the interface of the building blocks). Per cut `i`, molecules cracked per second per kg of catalyst:

```
r_i = f k_max exp(-E/R (1/T - 1/T_ref)) kappa_i c_i (c_H2/c_ref)^m / (1 + K_N(T) c_Norg + K_NH3 c_NH3)
```

with `f = activity x effectiveness x wetting`, `c_i` the cut's fugacity-equivalent liquid concentration, `c_Norg` the total **organic nitrogen** concentration of the liquid (both nitrogen classes, all cuts -- the nitrogen the pretreat bed left), `K_N(T) = K_N exp(-dH_N/R (1/T - 1/T_ref))` (Langmuir adsorption inhibition: basic nitrogen adsorbs on the acid sites) and `m = 0` by default (first order in hydrocarbon, as the lumping models are). The hydrotreating network runs alongside on the cracking catalyst (`HCKineticParams.hdt`, default `CRACK_BED_HDT_PARAMS`, its own cracking leak off).

**Continuous lumping** (`scheme="continuous"`, the default), after Laxminarasimhan, Verma & Ramachandran (1996). Each cut has a normalised boiling point `theta = (Tb - T_low)/(T_high - T_low)` and reactivity `kappa = theta^(1/alpha)` (`k = k_max theta^(1/alpha)`). The species-type distribution is `D(k) = dN/dk = N0/(alpha k_max^(1/alpha)) k^(1/alpha - 1)` and the yield distribution of species of reactivity `k` formed by cracking species of reactivity `K`

```
p(k, K) = [exp(-((k/K)^a0 - 0.5)^2 / a1) - exp(-0.25/a1) + delta (1 - k/K)] / (S0 sqrt(2 pi))
```

with `S0` from mass conservation, `int_0^K p(k, K) D(k) dk = 1`. `p(K, K) = 0` (a species does not crack to itself) and `p(0, K) ∝ delta` (light ends). On the grid the cracked mass of cut `i` is shared between the gas bin (below the lightest cut's lower edge) and every lighter cut `j` in proportion to `int_bin_j p(k(theta), K_i) D(k(theta)) dk/dtheta dtheta` (8-point Gauss-Legendre per bin), normalised over the bins -- the `S0` normalisation, discretised. Products of cut `i` stop at its lower cut edge: a cracked molecule always leaves its parent's cut. The weights distribute the parent's **carbon** (the paper's distribution is by mass; the two differ by the products' carbon fraction, 84--87 wt%).

```{note}
**These equations are not checked against the paper.** The paper (AIChE J. 42(9), 2645--2653, 1996) could not be reached from here; the forms above are the model as it is restated in the later literature that uses it, from recollection, so they are given **without equation numbers** and are marked unverified. Web-search snippets of citing papers corroborate parts of it -- five tuning parameters (`alpha`, `a0`, `a1`, `delta`, `k_max`) and an `exp(-(0.5)^2/a1)` term in the yield distribution -- which is not a check of the whole form. One consequence of the forms *as stated* is worth checking against the original: with `k = k_max theta^(1/alpha)` and that `D(k)`, the number of species per unit `theta` goes as `theta^(1/alpha^2 - 1)` -- uniform only at `alpha = 1`. The code follows the stated forms (`species_density` is computed from them, not assumed uniform). The paper's own parameter values are **not used** and their yield-versus-conversion results are **not reproduced** (below).
```

**Discrete lumps** (`scheme="discrete"`): TBP lumps (`lump_edges`, default naphtha < 165 °C < kerosene < 260 °C < diesel < 370 °C < VGO), a relative reactivity per lump (`lump_k`) and a selectivity row per lump (`lump_selectivity`: shares of a parent's cracked carbon to gas and each lighter lump), each product lump's share spread uniformly in `theta` over its cuts lighter than the parent (that spreading is this coding's assumption). The form is Stangeland's (1974) and the lump form of Mohanty, Saraf & Kunzru (1991); the default numbers are illustrative, not theirs. Both schemes reduce to one reactivity vector and one row-stochastic matrix, computed once per solve (`HCKinetics.prepare`) from the cut boiling points, so a TBP point of the assay moves them.

**What a cracked molecule becomes -- the property assignment (a modelling assumption).** The product landing in cut `j`:

- has cut `j`'s boiling point and the specific gravity of a **saturation-adjusted Watson K**, `SG_j = (1.8 Tb_j)^(1/3) / (Kw_feed + dKw)`, with `Kw_feed` the mass-average Watson K of the fresh feed's cuts and `dKw = +0.3` (illustrative; hydrocracked products are more paraffinic and naphthenic than the VGO they come from);
- has the molecular weight of Twu (1984) from `(Tb_j, SG_j)`, and the refractive index (Riazi-Daubert Huang index), hydrogen content (Goossens 1997) and hydrocarbon types (Riazi-Daubert 1986, API 2B4.1) of the composition module's own chain (#305) applied to those `(Tb, SG)`; aromatics split mono/di/poly by the hydrotreater's `DEFAULT_AROMATIC_SPLIT`; no olefins;
- carries no sulfur or nitrogen: a cracked molecule's heteroatoms leave as H2S and NH3 (the uncracked molecules keep theirs, and the hydrotreating network removes them);
- the gas bin's carbon becomes C1--C5 in the mole shares `HCU_GAS_SPLIT` (3/5/22/25/12/22/11 % C1/C2/C3/iC4/nC4/iC5/nC5; illustrative, iso-rich as hydrocracker light ends are).

The cut's critical constants and K-values stay the feed pseudo-component's (as in the hydrotreater): only its atoms, molecule count and the volume model below change.

**Hydrogen and heat.** Hydrogen consumption is the **hydrogen balance** of each event -- H atoms in the products (cuts, gas, H2S, NH3) less those of the parent, halved -- so it follows the conversion and the slate, not a separate correlation. C, S and N are conserved by construction and H through the H2 drawn; `check_element_conservation` is zero to round-off. Heat is per H2: an event making `n` molecules from one breaks `n - 1` C--C bonds, each with one H2 and the heat of n-hexane + H2 -> n-butane + ethane (`SCISSION_HEAT`, -42.7 kJ/mol); the rest of the H2 (saturation of the products, heteroatom removal) releases the benzene + 3 H2 -> cyclohexane heat per H2 (`SATURATION_HEAT_PER_H2`, -68.4 kJ/mol H2). Both are the hydrotreater's model-compound thermochemistry.

**Constants.** Every number in `HCKineticParams` (`k_max`, `E`, `alpha`, `a0`, `a1`, `delta`, `K_N`, `dH_N`, `dKw`, the lump tables) is **illustrative**: chosen here so that the default VGO cracks about 70 % per pass with 380 °C bed inlets (WABT near 395 °C), LHSV 1.5 h⁻¹ and 150 bar, with bed rises of 20--25 K and a middle-distillate-selective slate. Published hydrocracking parameters belong to one catalyst and one feed; a predictive slate needs the yield-distribution parameters fitted to the unit's own test runs (`difflow.estimation`). The commercial yield models (UOP Unicracking, Chevron Lummus ISOCRACKING, Shell, Axens) are proprietary; nothing here is equivalent to them.

(refinery-hydrocracker-fractionator)=
### Fractionator and UCO recycle

The fractionator is a **documented simplified split**, not a column (the issue allows either). Gases go whole to one product (H2, H2S, NH3, C1, C2 to off-gas; C3, C4 to LPG; C5, C6 to light naphtha; water to sour water). Cut `i` goes to the liquid products by a smooth step in its boiling point about each TBP cut point `T_c` (defaults 85, 165, 260, 370 °C): its share above `T_c` is `S((Tb_i - T_c)/w)`, `S` the quintic smootherstep on [-1, 1] (exactly 0 below, 1 above, C²), `w = 15 K`. `w` stands for the overlap between neighbouring products and is what makes a yield differentiable in its cut point on a grid of 20--25 K cuts. Attributes follow their cut, so the balances close exactly. Not computed: fractionator and side-stripper duties, steam, flash points; a `StageColumn` fractionator with side draws would replace `fractionate` without changing its callers.

**UCO recycle.** `uco_recycle` (a fraction of the bottoms) returns to the cracking reactor inlet; the rest is the UCO bleed. Because the step is exactly 1 above `T_uco + w`, the UCO is exactly empty below `T_uco - w`, so the recycle is torn on a fixed set of cuts (every cut boiling above `T_uco - w - uco_margin`, `uco_margin = 30 K`): their molecules and every attribute, a couple of hundred unknowns. Newton on that would need a tangent per unknown through the reactors; the loop is a contraction instead (its gain is the recycle fraction times the share of recycled oil that survives a pass; about 0.8 per pass measured on the default unit at 60 % recycle), so it is solved by **Anderson-accelerated substitution around the gas-loop Newton** (`hydrocracking.fixed_point`; depth 5, Walker & Ni 2011), and its gradient is the implicit one: the adjoint system `(I - (dG/dz)^T) w = v`, solved by GMRES on the scaled system with one vector-Jacobian product of the loop per operator application -- reverse mode only, which is what the reactor's diffrax adjoint supports. The iterations are never differentiated. A once-through unit (`uco_recycle = 0` at construction) builds no UCO tear; `recycle=True` builds one even at zero recycle.

(refinery-hydrocracker-specs)=
### Degrees of freedom and specs

`HydrocrackerParams`:

| Spec | Default | |
|---|---|---|
| `T_pretreat` | 370 °C | pretreat first-bed inlet |
| `quench_pretreat`, `pretreat_beds` | (0.06,), (0.5, 0.5) | quench to pretreat bed 2 (fraction of treat gas); catalyst split |
| `lhsv_pretreat` | 1.5 h⁻¹ | on fresh feed |
| `T_crack` | 380 °C | inlet of **every** cracking bed when `quench_crack=None` (the quench to each is solved) |
| `quench_crack` | None | or fixed quench fractions of the treat gas (a knife-edge: see below) |
| `crack_beds` | (0.15, 0.18, 0.20, 0.22, 0.25) | catalyst split, smaller beds first |
| `lhsv_crack` | 1.5 h⁻¹ | on fresh feed (a recycle loads the same catalyst harder) |
| `P` | 150 bar | uniform |
| `h2_oil` | 1500 Nm³/m³ | treat gas H2 per fresh feed, quench included |
| `purge`, `makeup` | 0.05, 99 % H2 / 1 % CH4 | |
| `h2s_removal`, `nh3_removal` | 0.99, 1.0 | amine / wash water |
| `hps_T`, `loop_dP`, `compressor_eta` | 50 °C, 8 bar, 0.75 | |
| `uco_recycle` | 0 | fraction of UCO recycled |
| `cut_points`, `cut_width` | 85/165/260/370 °C, 15 K | fractionator |
| `pretreat_kinetics`, `crack_kinetics` | `VGO_PRETREAT_PARAMS`, `HCKineticParams()` | activities are the deactivation handles |

`TargetSpec(output, target)` (the hydrotreater's) replaces the cracking inlet temperature with a target on any output -- `TargetSpec("conversion.per_pass", 0.7)` or `TargetSpec("wabt.crack", 653.15)` -- by a scalar Newton solve round the whole unit, differentiated implicitly (`outputs["T_shift"]`).

**Assumptions** beyond the hydrotreater's: series flow (the pretreat effluent, H2S and NH3 included, goes to the cracker); the cracking reactor's inlet temperature is a spec (an interstage exchanger is implied, its duty not computed); with `quench_crack=None` every cracking bed's inlet is held at `T_crack` and the quench into each later bed is the treat gas whose heating to `T_crack` absorbs the cooling of the bed above to it (the building blocks' `quench=None` enthalpy balance, explicit bed by bed); the first cracking bed takes the pretreat effluent with no gas of its own, and the pretreat reactor's inlet gets what the quenches leave (`crack.gas_left`, which must stay positive). The total quench share is one more unknown of the recycle-gas tear, so no inner Newton runs per pass. Fixed quench rates (`quench_crack=(...)`) are supported but are a knife-edge: with the quench fixed, a few kelvin on the inlet either runs the beds away or lets them die out, which is why units are run on bed-inlet temperature control; the UCO is recycled to the cracking reactor inlet at the cracking inlet temperature.

**Product gravity.** A cut holds feed molecules (treated) and cracked ones. Its molar volume is `(1 - f_cr) v0 + f_cr v0' + attr . dv`: `f_cr` the cracked share of its molecules, `v0` the feed molecule's volume with its aromatics and olefins taken out by the hydrotreater's `VOLUME_INCREMENTS`, `v0'` the same for the assigned cracked product, and `attr . dv` the current aromatic/olefin counts times those increments (so saturation after cracking still counts). Its SG is its mass (from its atoms) over that volume.

(refinery-hydrocracker-outputs)=
### Outputs

`HydrocrackerResult.outputs` (units in `OUTPUT_UNITS` and `PRODUCT_OUTPUT_UNITS`):

- per product (`off_gas`, `lpg`, `light_naphtha`, `heavy_naphtha`, `kerosene`, `diesel`, `uco`, `uco_bleed`): `.rate` (kg/s), `.yield` (mass, on fresh feed); for LPG and the liquids `.volume`, `.volume_yield`, `.sg`, `.api`, `.S_wppm`, `.N_wppm`, `.H_wt`, `.aromatics_vol`; for the liquids TBP `.T05` ... `.T95`; `diesel.cetane_index` (ASTM D4737 from TBP->D86 and density, the blend pool's functions); `uco.bmci` (US Bureau of Mines correlation index from the volume-average boiling point and SG);
- `conversion.per_pass` (`1 - UCO / (fresh 370+ + UCO recycled)`) and `conversion.overall` (`1 - UCO bleed / fresh 370+`), both on the fractionator's own UCO cut; `naphtha_to_middle_distillate` (mass); `liquid.volume_yield`;
- hydrogen: `h2.chemical` (mol/s; by the hydrogen balance over every outlet), `h2.chemical_nm3_m3`, `.chemical_scf_bbl`, `.chemical_wt`, `h2.consumed_by_balance` (H2 in less out, equal to it to round-off), `h2.makeup`, `h2.purge`, `h2.dissolved`;
- reactors: `wabt.pretreat`, `wabt.crack`, `pretreat.dT_total`, `crack.dT_total`, per bed `<reactor>.bed<k>.T_in`, `.dT`, cracking `.quench`; `crack.gas_left`; `pretreat.N_wppm` and `.S_wppm` (the organic N and S the cracking catalyst sees); catalyst masses;
- loop: `recycle.rate`, `recycle.h2_purity`, `purge.rate`, `makeup.rate`, `compressor.power`, `reactor.pH2_in`; `uco.recycle_rate`;
- convergence: `tear.residual`, `uco.residual`, `uco.steps`; `res.converged`.

`res.balances`: relative closure of mass, C, H, S, N over the unit (fresh feed + makeup = products + UCO bleed + purge + acid gas + separator water; nothing is added for the recycle, so an unconverged UCO tear shows here). `res.product_char` / `res.product_stream(name)` put any product into `BlendComponent.from_stream` (jet, ULSD pools).

Not computed: **jet smoke point and freeze point** (the (Tb, SG) correlations the issue names could not be verified, and are poor for highly saturated product; a jet pool takes measured overrides), fractionator duties, naphtha octane.

(refinery-hydrocracker-results)=
### Results on the test VGOs

Feeds: the LVGO + HVGO of a `VacuumColumn` on the idealized atmospheric residue (`vacuum.atmospheric_residue`, 150 kg/s of crude) of two test crudes, both characterized with a heavy end and the #305 composition -- the light crude of the composition section (SG 0.86, 1.8 wt% S, 1500 wppm N) and a heavy one (SG 0.93, 3.0 wt% S, 2500 wppm N). The VDU products carry every cut at some trace level; the unit keeps 28 (light) and 26 (heavy) cuts and drops 1.7e-4 and 2.4e-4 of the feed mass (`dropped_mass_fraction`). Defaults otherwise (the heavy VGO with 365 °C cracking inlets: at 380 °C its beds run away).

| | light VGO, once-through | light VGO, 60 % UCO recycle | heavy VGO, once-through |
|---|---|---|---|
| fresh feed | 45.1 kg/s; 2.91 wt% S, 2214 wppm N | same | 48.5 kg/s; 4.10 wt% S, 2948 wppm N |
| to the cracker | 947 wppm S, 42.7 wppm N | 844 wppm S, 40.0 wppm N | 315 wppm S, 14.5 wppm N |
| WABT pretreat / cracking | 394.1 / 395.2 °C | 393.9 / 391.0 °C | 416.8 / 383.4 °C |
| bed rises, cracking | 19.1, 21.4, 22.0, 23.1, 26.2 K | 13.3, 15.4, 16.2, 17.0, 18.8 K | 28.6, 30.3, 27.9, 26.0, 26.1 K |
| conversion (370 °C+), per pass / overall | 69.7 / 69.7 % | 46.9 / 68.8 % | 58.2 / 58.2 % |
| off-gas, LPG (wt%) | 1.07, 1.23 | 1.22, 1.06 | 1.43, 1.85 |
| light, heavy naphtha (wt%) | 2.55, 10.47 | 2.19, 9.22 | 1.47, 8.01 |
| kerosene, diesel (wt%) | 24.83, 31.48 | 23.03, 34.09 | 20.05, 27.13 |
| UCO bleed (wt%) | 28.60 | 29.47 | 39.81 |
| naphtha / middle distillate | 0.231 | 0.200 | 0.201 |
| chemical H2 | 278 Nm³/m³ (1649 scf/bbl, 2.69 wt%) | 264 Nm³/m³ (1565 scf/bbl) | 359 Nm³/m³ (2129 scf/bbl) |
| kerosene SG; diesel SG, cetane index | 0.790; 0.840, 65.5 | 0.790; 0.842, 65.3 | 0.829; 0.882, 47.4 |
| UCO BMCI | 33.1 | 34.1 | 50.1 |
| closure (worst of mass, C, H, S, N) | 2e-15 | 3e-12 | 1e-15 |
| tears | gas 1e-14 | gas 9e-12; UCO 1.8e-10 relative, 13 Anderson passes | gas 3e-13 |

Read these as the shape of the answer: every cracking constant is illustrative. A few things they show, all of which follow from the model rather than being tuned in: the recycle at the same catalyst and temperature *lowers* the per-pass conversion (a recycle reactor is less efficient than plug flow) and the overall conversion slightly, and buys selectivity -- 2.6 wt% more diesel, less naphtha per middle distillate, less H2; the heavy, aromatic VGO consumes more hydrogen, gives denser, lower-cetane products (its products inherit its lower Watson K through `Kw_feed + dKw`) and a higher-BMCI UCO; its pretreat bed rises 81 K, which a real unit would quench harder (the pretreat quench is a spec).

Compiling a once-through unit takes about 2.5 min and a solve about 7--9 s (the gas tear: 12 substitution passes, then Newton); with the UCO recycle the compile is about 6.5 min and a solve about 40 s. A reverse-mode gradient adds one compile: the once-through gradient test takes about 10 min, the recycle-ratio one about 30 min and 11 GB (its adjoint runs 60 vector-Jacobian products of the loop per cotangent), and both are marked `slow` and `release`.

**Gradients** (`test_gradients_match_central_differences`, `test_recycle_ratio_gradient`): kerosene yield, per-pass conversion and chemical H2 consumption with respect to the cracking inlet temperature, the reactor pressure and the 70 % TBP point of the assay (the feed's mass per cut held, so its moles move with the cut molecular weights), on the once-through light VGO; and overall conversion, diesel yield and chemical H2 with respect to the UCO recycle fraction, through the UCO tear's adjoint. One reverse-mode Jacobian each, against Richardson-extrapolated central differences (steps 0.5 K, 1 bar, 1 K; 0.02 in the recycle fraction) at `rtol` 1e-5.

**What the tests check** (`tests/refinery/test_hydrocracking.py`; full-unit tests `slow`):

- convergence from the default initialization on both VDU VGOs, once-through and with 60 % UCO recycle; mass, C, H, S and N closure to 1e-8 relative (they close to 1e-15 once-through and 3e-12 with the recycle); `h2.chemical` (H balance) equal to H2 in less H2 out; the UCO recycle identity `overall = 1 - (1 - rho)(1 - X)/(1 - rho (1 - X))`;
- conversion, naphtha/middle distillate and H2 consumption all rising with the cracking inlet temperature (360, 370, 380 °C), and the recycle lowering per-pass conversion and the naphtha/middle-distillate ratio;
- gradients against Richardson-extrapolated central differences (above);
- the discrete-lump scheme through the whole unit (converges, closes, conversion rises with temperature);
- the kerosene and diesel entering `BlendPool("jet")` / `BlendPool("ulsd")` through `BlendComponent.from_stream` with the unit's gravity, sulfur and cetane index; `hcu_block` delta vectors passing `check_delta_vectors`;
- the pieces: element conservation of both schemes at a point (1e-12 of the cracked flow), the distribution matrix row-stochastic with nothing landing in the parent's cut or heavier, reactivity rising with boiling point, the yield distribution's end points, `species_density` as derived, the product property chain (Watson K, Twu MW, H/C consistent with the H mass fraction), organic-N inhibition, the fractionator's shares (exactly zero UCO below `T_uco - w`), the adjoint fixed point against the implicit-function closed form, a cracking bed's conversion rising with temperature, the pretreat bed removing 95 %+ of the nitrogen; pins: the scission and saturation heats against the model-compound table, BMCI at its anchors (n-heptane 0, benzene 100).

(refinery-hydrocracker-planning)=
### Planning: `hcu_block`

`difflow_refinery.hydrocracking.planning.hcu_block(unit, feed, levers, outputs)` wraps the unit as a `difflow.planning.Block` in the style of `hdt_block`. Levers: `feed.bpd` (renamed `<product>.bpd` with `feed_product=`, e.g. `"vgo"`; composition held, both LHSVs move with the rate), `crack.T_in`, `pretreat.T_in` (°C), `h2_oil`, `pressure` (bar), `uco_recycle` (recycle units), `uco.cut_point` (°C). Outputs (`HCU_OUTPUTS`): product bbl/d (LPG, light and heavy naphtha, kerosene, diesel, UCO bleed), conversion per pass and overall, naphtha/middle-distillate ratio, chemical H2 and makeup (Nm³/h), cracking WABT, diesel cetane and sulfur, kerosene and diesel SG. Reverse mode only.

(refinery-hydrocracker-references)=
### References

| Key | Reference | Used for | How checked |
|---|---|---|---|
| C1 | Laxminarasimhan, C.S.; Verma, R.P.; Ramachandran, P.A. "Continuous lumping model for simulation of hydrocracking." *AIChE J.* **1996**, 42(9), 2645--2653. doi:10.1002/aic.690420925 | Continuous-lumping form: `theta`, `k(theta)`, `D(k)`, `p(k, K)`, the mass-conservation normalisation | Title, journal, volume, issue, pages and DOI confirmed by web search (publisher and index listings). The paper was **not reached**: the equations are as restated in the later literature, from recollection -- **equation numbers not given, forms unverified**; the published parameter values are **not used**. |
| C2 | Stangeland, B.E. "A kinetic model for the prediction of hydrocracker yields." *Ind. Eng. Chem. Process Des. Dev.* **1974**, 13(1), 71--76. doi:10.1021/i260049a013 | Discrete-lump form (reactivity rising with boiling point, a product-distribution rule) | Title and DOI confirmed by web search; issue and pages as in the issue (unverified). Form only, qualitative. |
| C3 | Mohanty, S.; Saraf, D.N.; Kunzru, D. "Modeling of a hydrocracking reactor." *Fuel Process. Technol.* **1991**, 29, 1--17 | Discrete-lump scheme on a commercial reactor | Not checked (unverified); qualitative only. |
| C4 | Quader, S.A.; Hill, G.R. *Ind. Eng. Chem. Process Des. Dev.* **1969**, 8, 98 | Early lumped hydrocracking kinetics | Listed by the issue; not checked (unverified); not used. |
| C5 | Twu, C.H. *Fluid Phase Equilib.* **1984**, 16, 137--150 | MW of the assigned products | As cited for `difflow_refinery.correlations`. |
| C6 | Riazi & Daubert (1986, 1987); Goossens (1997) | n20, H content and PNA of the assigned products | As cited for the composition module (#305). |
| C7 | Watson, K.M.; Nelson, E.F. *Ind. Eng. Chem.* **1933**, 25, 880 | Watson K | As cited for `correlations.watson_k` (unverified there). |
| C8 | Smith, H.M. "Correlation index to aid in interpreting crude-oil analyses." US Bureau of Mines Tech. Paper 610, **1940** | BMCI = 48640/VABP(K) + 473.7 SG - 456.8 | Formula as commonly given (e.g. Gary, Handwerk & Kaiser); source not checked (unverified). Pinned in the tests to the index's anchors (n-heptane 0, benzene 100). |
| C9 | Christianson, B. "Reverse accumulation and attractive fixed points." *Optim. Methods Softw.* **1994**, 3(4), 311--326 | Adjoint of a fixed-point iteration | Unverified (recalled); the construction is tested against the implicit-function closed form. |
| C10 | Walker, H.F.; Ni, P. "Anderson acceleration for fixed-point iterations." *SIAM J. Numer. Anal.* **2011**, 49(4), 1715--1735. doi:10.1137/10078356X | Anderson acceleration of the UCO loop | Recalled, unverified. |
| C11 | Saad, Y.; Schultz, M.H. "GMRES: a generalized minimal residual algorithm for solving nonsymmetric linear systems." *SIAM J. Sci. Stat. Comput.* **1986**, 7(3), 856--869 | The adjoint linear solve (`hydrocracking.fixed_point.gmres`, restarted GMRES(20) written out: `jax.scipy.sparse.linalg.gmres` differentiates its operator, a pullback through while loops) | Recalled, unverified. |
| -- | Hydrotreater references H1--H16 | PR flash, kinetics forms of the pretreat bed, model-compound heats, cetane index | See [the hydrotreater](#refinery-hydrotreater-references). |

(refinery-hydrocracker-not-done)=
### What is not done

- **The yield-versus-conversion cross-check is not done.** It needs the published numbers of Laxminarasimhan et al. (1996) (or Mohanty et al. 1991's plant comparison) with their parameters; neither paper could be reached, so nothing is reproduced and no agreement is claimed. The model's trends are tested (below); its absolute slate is illustrative.
- **No example notebook** (VDU -> hydrocracker -> jet/ULSD pools). The pieces are tested: a VDU VGO feeds the unit, and `product_stream`/`product_char` feed `BlendComponent.from_stream`.
- **The fractionator is not a column** (above), so no duties; jet smoke and freeze points are not computed.
- **Deactivation** is the `activity` multiplier of either catalyst, a differentiable parameter that `difflow.reconciliation.tracking` can track; not wired up or demonstrated.
- **Not a difflow `Flowsheet` object**: both tears are the unit's own.
- Out of scope (as the issue says): residue hydrocracking, hydrogen-network optimisation, cycle-length optimisation, dynamics; two-stage units are not built (the pieces would compose).

(refinery-residue)=
## Residue desulfurization and fuel oil

`difflow_refinery.residue` is the route from an atmospheric residue to a very-low-sulfur fuel oil (VLSFO, 0.50 wt% S). Its unit, `ResidueDesulfurizer`, is an atmospheric-residue desulfurizer (ARDS/RDS) built on the shared hydroprocessing blocks ([above](#refinery-hydroprocessing)): the hydrotreating kinetics with residue constants, plus refractory residue sulfur, hydrodemetallization (HDM) of Ni+V, CCR reduction and a small residue conversion, in the same `TrickleBedReactor`. `fuel_oil_blend` blends its products, and any cutter, in a `BlendPool` with the VLSFO specs. Like the hydrotreater it is a library, not a palette operation. Its rate constants are **illustrative**.

```python
import difflow_refinery as dr
from difflow_refinery.residue import ResidueDesulfurizer, RDSParams, fuel_oil_blend

char = dr.characterize(assay, composition=True)     # sulfur, CCR and Ni+V per cut; the #305 composition is required
residue = cdu.products["residue"]                     # or residue.atmospheric_residue_cut(char, kg_s)
rds = ResidueDesulfurizer(char, residue, RDSParams(T_in=(373.0 + 273.15,)))
res = rds.solve(residue)
print(res.table())

fo = fuel_oil_blend([res.blend_component("residue"), res.blend_component("distillate")],
                    [res.volume("residue"), res.volume("distillate")])
fo.properties["S_ppm"], fo.margins                    # 0.31 wt% S, every VLSFO spec met
```

```
atmospheric residue --+--> [bed 1] --quench--> [bed 2] --quench--> [bed 3] --> product separation --+--> gas: H2, H2S, NH3, C1-C5, water
                      |                                                                             +--> distillate (cuts below 350 C)
treat gas ------------+                                                                             +--> desulfurized residue
                                                                    fuel oil pool  <-- desulfurized residue + distillate (+ cutters)
```

(refinery-residue-route)=
### Which route, and why

The issue (#331) offered two routes: a residue hydrotreater, or the vacuum unit plus cutter-stock blending. The choice turns on one piece of arithmetic. Sulfur blends linearly by mass, so a base stock at `S_r` blended with a cutter at `S_c` reaches a spec `S_s` only with a cutter mass fraction of at least

$$
x_c = \frac{S_r - S_s}{S_r - S_c} \qquad \text{(\texttt{cutter\_fraction\_for\_sulfur})}
$$

For the test crude's atmospheric residue (3.27 wt% S) and a 15 ppm ULSD cutter, $x_c = 0.85$. The "fuel oil" would be 85 % diesel by mass. A refinery cannot make it: the crude unit of `examples/40` makes about a third as much diesel as residue. Nor would it want to: the blend sells for less than the diesel it consumed. An LCO cutter is worse, since an FCC on a high-sulfur feed makes LCO at around 1--2 wt% S (a typical figure, unverified). The vacuum residue is worse again (about 4 wt% S): the VDU sends the sulfur to its bottoms, so the VDU on its own moves the problem without solving it. Cutter blending is what sets a residual fuel's *viscosity*. It is not a way to take a high-sulfur crude's residue to 0.5 wt% S.

So the route here is (a), the residue desulfurizer. It is **credible**: it is how refineries that run high-sulfur crudes make VLSFO, and ARDS units report 85--92 % HDS on atmospheric residue (R2, R3; unverified). It is **affordable** in this code base: it adds a kinetic model and a thin unit around the reactor that #306 already built and tested, and changes none of it. It compiles in about 20 s and re-solves in 0.1--0.2 s. A gradient through it compiles in about 90 s. The cutter route stays available through the same pool (`fuel_oil_blend` takes any property-mode cutter), and `cutter_fraction_for_sulfur` gives the arithmetic for a low-sulfur crude, where cutting alone does work.

(refinery-residue-feed)=
### Feed and layout

The layout is the hydrotreater's (`HDT_ATTRIBUTES`: C and H atoms, S in five classes, N in two, mono/di/poly-aromatic, olefinic and naphthenic molecule counts) with three more per-cut attributes (`RDS_ATTRIBUTES`):

| Attribute | What it counts | Element in the balance | From the characterization |
|---|---|---|---|
| `S_residue` | sulfur atoms in asphaltene/resin molecules: the refractory residue sulfur | S | a share `residue_s_share(Tb)` of each cut's sulfur, taken from the five classes in proportion |
| `NiV` | Ni + V atoms, as moles of a nominal metal of `METAL_MW` (a 3:1 V:Ni mix, 52.9 g/mol) | none (metals are outside a cut's mass, as everywhere in the layout) | `char.nickel_vanadium` |
| `CCR` | carbon atoms the Conradson test would leave as coke | none (a subset of the cut's C; counting it as C would count it twice) | `char.ccr` |

`rds_feed(char, stream, layout)` is `hdt_feed` plus these three columns, so a cut's mass from its atoms is still `F * MW`. The refractory share (`DEFAULT_RESIDUE_S_SHARE`) is zero up to 450 C, 0.10 at 538 C, 0.25 at 600 C, 0.35 at 700 C and 0.50 at 800 C, linear in between. It is **illustrative**: it has the shape the residue-HDS literature describes, with asphaltene sulfur concentrated in the heaviest fraction, but the numbers are this project's. Pass `residue_s_share=` with measured shares, e.g. from the sulfur of the C7 asphaltenes. An assay without Ni+V or CCR gives zero columns. The unit carries every pseudo-component from the lightest one in the feed (or the lightest one conversion can reach, if that is lighter) up to the heaviest one. A crude-unit residue goes in as it is; `atmospheric_residue_cut(char, kg_s)` is an idealized one (every cut above 350 C, in crude proportion).

(refinery-residue-kinetics)=
### Reactions and rate laws

`RDSKinetics.rates` is `HDTKinetics.rates` with the residue constants (`residue_hdt_params`), plus four reactions per cut. The symbols are those of the hydrotreater ([rate laws](#refinery-hydrotreater-kinetics)): `c` are the fugacity-equivalent liquid concentrations, `f = activity * effectiveness * wetting`, `k(T)` is Arrhenius about `T_ref` = 380 C, and `h = (c_H2/c_ref)^m`:

$$
\begin{aligned}
\text{refractory HDS:}\quad & r_{S,res} = f\,k_{S,res}\,c_{S,res}\,h^{m}/D^2, \quad D = 1 + K_{H_2S} c_{H_2S} + K_N c_{N,basic} && (3\ \mathrm{H_2} \to \mathrm{H_2S}) \\
\text{HDM:}\quad & r_M = f\,k_M\,c_{NiV}\,h^{m_M} && \text{(metal onto the catalyst)} \\
\text{CCR reduction:}\quad & r_{CCR} = f\,k_{CCR}\,c_{CCR}\,h && (0.5\ \mathrm{H_2}\ \text{per CCR carbon}) \\
\text{conversion:}\quad & r_X = f\,k_X\,c_i \quad (T_{b,i} \ge T_{conv}) && (\text{molecule of } i + (m_i-1)\,\mathrm{H_2} \to m_i \text{ molecules of } j(i))
\end{aligned}
$$

- **HDS by class** keeps the hydrotreater's LHHW form: first order in each of the five classes, with the squared H2S and basic-nitrogen inhibition denominator (Korsten & Hoffmann 1996; Froment et al. 1994; hydrotreater refs H7, H8). The residue constants change four things. The effectiveness factor is 0.35, for the hindered diffusion of residue molecules into the pores. The class constants are lower. The basic-nitrogen adsorption constant is weaker, because most of a residue's basic nitrogen sits in molecules too large to reach the sites. The hydrotreater's cracking leak is switched off (`crack_k = 0`); conversion replaces it. The refractory class shares the inhibition denominator and reacts more slowly still. Being first order with its own constant, it is part of what gives the lumped total a high apparent order, as in residue HDS data.
- **HDM, CCR reduction and conversion** are first-order lumps with an H2 term, the forms reviewed for heavy-oil hydroprocessing by Ancheyta et al. (2005). Removed metal deposits on the catalyst, reported as `metals.deposit` (kg/s); that deposit is what sets a residue unit's cycle length (Rana et al. 2007). HDM takes no hydrogen and gives no heat: the metals are at ppm level, and their H2 would be about 1e-4 of the unit's.
- **Conversion** acts only on cuts boiling at or above `T_conv` (538 C), the "residue" of a conversion figure. Each converted molecule of cut `i` sends all of its atoms to `m_i = n_C,i/n_C,j` molecules of the lighter cut `j(i)` whose carbon number is nearest half of `i`'s (`conversion_targets`). The `m_i - 1` new chain ends take one H2 each, so carbon, hydrogen and every heteroatom balance exactly. The fragments inherit the parent's sulfur, nitrogen, metals, CCR and ring classes, per atom.
- **Heats** are the hydrotreater's model-compound reaction enthalpies: benzothiophene HDS for the refractory class, benzene to cyclohexane per H2 for CCR reduction, and n-hexane + H2 to n-butane + ethane per bond broken. The CCR stoichiometry (one aromatic ring of six carbons saturated by three H2) is **illustrative**.
- **Constants** (`RDSKineticParams`, `residue_hdt_params`) are **illustrative**, not fitted to any catalyst or unit. They were chosen so that this crude's residue at a WABT near 395 C, LHSV 0.25 1/h, 150 bar and 1000 Nm³/m³ lands in the ranges ARDS units report: 85--92 % HDS, 70--85 % HDM, 40--60 % CCR reduction and 10--20 % conversion of the 538 C+ (Speight 2000; Rana et al. 2007). The release tests pin those ranges. In a real unit the HDM catalyst grades into the HDS catalyst; here one average catalyst fills every bed.

(refinery-residue-specs)=
### Degrees of freedom and specs

`RDSParams`: `T_in` (bed-1 inlet with given quench fractions, default 373 C; or one per bed with `quench=None`, which solves each quench), `quench` (default 0.20 and 0.25 of the treat gas into beds 2 and 3), `bed_fractions` (0.25, 0.35, 0.40), `P` (150 bar), `lhsv` (0.25 1/h on 60 F feed volume), `catalyst_density` (800 kg/m³), `gas_oil` (treat gas, 1000 Nm³/m³), `treat_gas` (90 % H2 / 10 % CH4), `T_gas` (quench gas, 70 C), `product_cut_T` (350 C), `T_conv` (538 C), `kinetics`, `kij`, `reactor` (`ReactorOptions`, rtol 1e-8).

- **Once-through treat gas.** The treat gas is a given rate and composition, not a solved recycle loop. That equals a recycle loop with an ideal amine scrubber and makeup that holds the recycle purity. `hydroprocessing.recycle` has the pieces to close the loop, which matters for the hydrogen balance and little for the product sulfur. Chemical H2 consumption comes from the hydrogen balance.
- **Product separation** is an ideal component split of the effluent. Every gas and light end (H2, H2S, NH3, C1--C5, water) goes to `gas`. Every cut below `product_cut_T` goes to `distillate`, and the rest to `residue`. It stands in for the hot and cold separators and the fractionator, and the balances close through it exactly. No dissolved gas is carried into the liquids.
- No feed heater: the bed-1 inlet temperature is a spec, and oil and treat gas enter at it. No bed pressure drop.
- Fixed at construction (they shape the layout, or say which code path runs): the bed count, the quench mode, `T_conv`, `product_cut_T`, `kij`. Everything else can be traced, and `solve(feed, params=...)` re-solves without recompiling.

(refinery-residue-outputs)=
### Outputs and the fuel-oil pool

`RDSResult.outputs` (units in `residue.OUTPUT_UNITS`) holds:

- feed and product qualities: `feed.S_wt`, `feed.NiV_wppm`, `feed.CCR_wt`, `residue.S_wt`, `residue.NiV_wppm`, `residue.CCR_wt`, `residue.sg`, `distillate.S_wppm`;
- yields: `residue.yield`, `distillate.yield`, `gas.rate`;
- conversions: `hds.conversion`, `hdm.conversion`, `ccr.reduction`, `hdn.conversion`, and `conversion` (of the cuts at or above `T_conv`);
- hydrogen: `h2.chemical`, `h2.chemical_nm3_m3`, `h2.chemical_wt`, and `h2s.make`;
- the reactor: `metals.deposit`, `wabt`, `bed<k>.T_in`, `bed<k>.dT`, `bed<k>.quench`.

`balances` are the relative errors of mass, C, H, S, N and Ni+V across the unit; the Ni+V balance counts the catalyst deposit.

`product_char` is a `BlendCharacterization` of the treated cuts. Their molar mass comes from their atoms. Their gravity comes from the feed cut's molar volume plus the hydrotreater's per-molecule volume increments for the types saturated. Each cut carries `S_ppm`, `N_ppm`, `CCR_wt` and hydrocarbon types. `blend_component(name)` turns `"residue"` or `"distillate"` into a fuel-oil component. Its viscosity at 50 C is *estimated* by `difflow_refinery.properties` (Abbott at 100/210 F and Walther between, from the treated cuts' TBP 50 % point and gravity, **unverified** there). It is property mode by default, so it blends with cutters from any other unit. `volume(name)` gives the standard volume flow at 15 C.

`fuel_oil_blend(components, volumes)` is a fuel-oil `BlendPool` by volume flow with `VLSFO_SPECS`: S ≤ 5000 ppm, viscosity ≤ 380 cSt at 50 C, SG ≤ 0.991 and CCR ≤ 18 wt%. The sulfur limit is the MARPOL Annex VI 0.50 % m/m global cap of 2020. The other three are the ISO 8217 RMG 380 limits, **as recalled** and not checked against the standard's table. Sulfur, nitrogen and CCR blend by mass, SG by volume, and viscosity by Refutas. The pool's mass is the products' mass to round-off, so a route balance closes through it.

(refinery-residue-results)=
### Results on the test crude

These are the test crude of `examples/35`--`40` (1.8 wt% S, 1500 wppm N, 5 wt% CCR), given 40 wppm Ni+V, with the defaults and 50 kg/s of idealized 350 C+ residue (`tests/refinery/test_residue.py`):

| | feed (atm. residue) | desulfurized residue | fuel oil (residue + RDS distillate) |
|---|---|---|---|
| S | 3.27 wt% | 0.306 wt% | 0.31 wt% (spec 0.50) |
| Ni+V | 86 wppm | 16.8 wppm | |
| CCR | 10.8 wt% | 5.3 wt% | (spec 18) |
| SG | 0.953 | 0.928 | (spec 0.991) |
| viscosity at 50 C (estimated) | 225 cSt | 106 cSt | (spec 380) |

WABT is 394.7 C over a total bed rise of 71 K. HDS is 90.7 %, HDM 81.1 %, CCR reduction 51.6 % and 538 C+ conversion 13.1 %. The distillate yield is 1.8 % (at 0.50 wt% S: the fragments inherit their parent's sulfur, so it wants a distillate hydrotreater before the diesel pool). Chemical H2 is 121 Nm³/m³ (1.15 wt%). All balances close to 1e-15.

Over the route, the residue's sulfur equals the fuel oil's plus the H2S to 1e-10, and residue plus treat gas equals fuel oil plus gas.

The gradient of fuel-oil sulfur with respect to the bed-1 inlet temperature is −602 ppm/K, by reverse mode through the beds, the quench mixing, the product grid and the pool. A central difference at h = 0.5 K gives −602.0. The release test holds the two to 0.2 %.

On the CDU residue of `examples/40` (3.06 wt% S, no Ni+V given), the fuel oil comes out at 0.31 wt% S and 69 cSt, with every spec met (release test).

(refinery-residue-ex40)=
### In the whole-refinery example

`examples/40_refinery_flowsheet.ipynb` still sends the raw residue to the fuel-oil pool, with an assumed viscosity (its stand-in). The replacement is:

```python
from difflow_refinery.residue import ResidueDesulfurizer, fuel_oil_blend
rds = ResidueDesulfurizer(char, P["residue"])
rds_res = rds.solve(P["residue"])
fuel_oil = fuel_oil_blend([rds_res.blend_component("residue"), rds_res.blend_component("distillate")],
                          [rds_res.volume("residue"), rds_res.volume("distillate")])
```

The H2 consumer it adds is `rds_res.outputs["h2.chemical"]`, which the hydrogen balance of that example would need. Its H2S is `rds_res.outputs["h2s.make"]`, for the sulfur table. Give the assay `nickel_vanadium_wppm=` for HDM to have anything to remove.

(refinery-residue-references)=
### References

| Key | Reference | Used for | How checked |
|---|---|---|---|
| R1 | Ancheyta, J.; Sanchez, S.; Rodriguez, M.A. "Kinetic modeling of hydrocracking of heavy oil fractions: a review." *Catal. Today* **2005**, 109, 76--92. doi:10.1016/j.cattod.2005.08.015 | First-order lumped forms for heavy-oil HDS, HDM, CCR and conversion | Recalled; not reached (unverified). Forms only; no constants used. |
| R2 | Rana, M.S.; Samano, V.; Ancheyta, J.; Diaz, J.A.I. "A review of recent advances on process technologies for upgrading of heavy oils and residua." *Fuel* **2007**, 86, 1216--1231. doi:10.1016/j.fuel.2006.08.004 | ARDS severity ranges; metals deposit limits cycle length | Recalled (unverified). Qualitative ranges only. |
| R3 | Speight, J.G. *The Desulfurization of Heavy Oils and Residua*, 2nd ed., Marcel Dekker, **2000** | Residue HDS behaviour, refractory asphaltene sulfur, typical ARDS conditions | Recalled; chapter not checked (unverified). Qualitative only. |
| R4 | IMO, MARPOL Annex VI, Regulation 14 (0.50 % m/m global sulfur limit from 1 January 2020) | VLSFO sulfur spec | Widely reported limit; regulation text not opened (unverified). |
| R5 | ISO 8217:2017, *Petroleum products -- Fuels (class F) -- Specifications of marine fuels*, Table 2, grade RMG 380 | Viscosity 380 mm²/s at 50 C, density 991.0 kg/m³ at 15 C, CCR 18 % m/m | As recalled; the standard was not opened (unverified). |
| -- | Hydrotreater references H1--H16 | PR flash, HDS/HDN/HDA forms, model-compound heats | See [the hydrotreater](#refinery-hydrotreater-references). |
| -- | Refutas viscosity blending; Abbott-Kaufmann-Domash viscosity | Fuel-oil viscosity | As cited for the blend pool and `difflow_refinery.properties`. |

(refinery-residue-not-done)=
### What is not done

- **No literature or plant cross-check.** The constants are illustrative and were chosen to land in published severity ranges. They are not fitted, and no paper's profiles are reproduced. Product sulfur, metals and hydrogen are the shape of the answer, not a prediction for any catalyst.
- **No recycle-gas loop, hot/cold separators or fractionator column**: see the specs above for what stands in for each.
- **No catalyst deactivation.** `metals.deposit` is reported, and `activity` is a differentiable multiplier that `difflow.reconciliation.tracking` could follow, but neither metals-driven deactivation nor cycle length is modelled.
- **One average catalyst** for every bed (no HDM/HDS grading), and one `T_ref`.
- **The RDS distillate** leaves at about the parent's sulfur and is not hydrotreated. In the example it goes to the fuel oil.
- **No VDU in the route.** For this crude it is not needed, as explained above. The pieces compose: `VacuumColumn` on the RDS residue, or the RDS on a vacuum residue (VRDS), with the same unit.
- Out of scope: ebullated-bed and slurry residue hydrocracking, solvent deasphalting, coking, and IMO compatibility and stability (the P-value).

---

(refinery-hydrogen)=
## The hydrogen network

`difflow_refinery.hydrogen` (#329) balances the refinery's hydrogen: the
reformer's net gas, a hydrogen plant and imports on one side, the
hydrotreaters' and hydrocracker's makeup on the other, through one or more
headers, with an optional PSA, purge to fuel gas, and export. It returns the
balanced header and the purity each consumer receives, and
`close_hydrotreater_loop` feeds that purity back into the hydrotreaters. A
library, not a palette operation.

```python
import difflow_refinery.hydrogen as h2

net = h2.HydrogenNetwork(
    producers=[h2.Producer.from_reformer(ref)],              # the reformer's net gas, as it is
    consumers=[h2.Consumer("nht", 20.0, P=60e5, min_purity=0.85),
               h2.Consumer("dht", 40.0, P=60e5, min_pH2=50e5)],
    headers=[h2.Header("main", P=20e5, min_purge=1.0,
                       psa=None,                             # or h2.PSA(recovery=0.88, purity=0.999)
                       swing=[h2.Import(purity=0.999)])])     # fills a deficit
res = net.solve()
res.outputs["h2.surplus"], res.outputs["nht.purity"], res.balances

# feed the purity back into the hydrotreaters (substitution on the purity)
loop = h2.close_hydrotreater_loop(net, {"nht": (nht, nht_feed, nht_params),
                                        "dht": (dht, dht_feed, dht_params)})
loop.header.outputs["h2.surplus"], loop.params["nht"].makeup, loop.units["nht"].outputs["reactor.pH2_in"]
```

### Hydrogen network: model

Every stream is a vector of molar flows on `HEADER_GASES` (hydrogen,
methane to n-pentane, and n-hexane for anything heavier, on the
hydroprocessing gas names).

- **Producers** (`Producer`) have a fixed flow and composition:
  `Producer.from_reformer(result)` (the reformer's `net_gas`; C6+ traces
  lumped into n-hexane on a mole basis, so H2 and total moles are
  conserved), `Producer.of_purity(name, h2, purity, impurity=...)`, or
  `Producer.from_flows(name, {gas: mol/s})`. A share `to_psa` of each can
  go through the header's PSA.
- **PSA** (`PSA(recovery, purity)`): product H2 = `R s E_H2`; product
  impurities `R s E_H2 (1 - y_P)/y_P`, split like the feed's; tail gas the
  rest, to fuel gas. `s = 1`, or with `target_purity` the share for which the
  header purity equals the target, which is linear in `s`:
  `s = (y* B - A) / (E_H2 (R - 1) - y* (R E_H2/y_P - E))`, clipped to
  `[0, 1]` (`<h>.psa.target_error` is nonzero when the PSA cannot make the
  target). Impurities slip in the feed's proportions; there is no
  multicomponent adsorption model.
- **Swing sources** (`Import`, `H2Plant`) fill the deficit
  `need = sum_j d_j + min_purge - S_H2`, in order, each `clip(need, 0,
  capacity)`.
- **Consumers** (`Consumer`) take a makeup H2 flow `d_j` (the
  `Hydrotreater`'s `h2.makeup` output: chemical consumption, solution loss
  and the purge's H2) at the header's composition, so their total makeup is
  `d_j / y`. `Consumer.from_hydrotreater(name, result, params)` reads it. A
  demand may respond linearly to purity,
  `d_j(y) = d_j0 + (dd_j/dy)(y - y_ref)`; then the header purity is the
  fixed point of `y -> purity(d(y))` (30 iterations, residual reported as
  `<h>.loop_residual`).
- **Purge** is what is left, `G - sum_j M_j`, to fuel or (`purge_to="export"`)
  export. Its H2 is the **surplus**; a negative surplus is a deficit the
  swing could not cover. It is returned, not hidden, and
  `res.feasible["<h>.balanced"]` is `False`.

Consumers on one header all receive the header's purity. Consumers that
need different purities go on different headers (a header per pressure
level, say); a cascade from one header's purge into another is not built.

**Specs.** `min_purity` is on the makeup's H2 mole fraction;
`min_pH2` on its H2 partial pressure at the consumer's makeup pressure,
`y P`. Both report as `<c>.purity_margin` (mole fraction). Neither is the
reactor-inlet H2 partial pressure: that also depends on the unit's recycle
purity and is the `Hydrotreater`'s `reactor.pH2_in` output, which the
closed loop gives.

**Balances.** `res.balances` closes the network over producers + swing
against makeups + fuel gas + export, in total moles, H2 and mass, and checks
each consumer's makeup H2 against its demand. The purge is computed by
difference, so the first three close by construction (to round-off,
tested); the last is an independent check of the makeup bookkeeping.

### Hydrogen network: closing the loop on the hydrotreaters

`close_hydrotreater_loop(network, {consumer: (Hydrotreater, feed, params)})`
substitutes on the purity: balance the network; solve every hydrotreater
with `HydrotreaterParams.makeup` set to its header's composition (folded
onto the unit's gases by `fold_composition`, which moves a heavier gas into
the nearest lighter one so the purity is unchanged); put each `h2.makeup`
back on its consumer, with a secant `d_demand_d_purity` once two passes at
different purities exist; re-balance; stop when no purity moves by more than
`tol`. With purge as the only swing the purity does not depend on the
demands and one pass closes it; with an import or H2 plant as the swing,
the demands move the purity and a few more passes follow, each a re-solve
of a compiled unit.

**AD mode.** The loop is concrete Python: `HydrotreaterParams.makeup` is a
dict that `makeup_vector` reads with `float()`, so the makeup composition is
not a traced input of `Hydrotreater.solve`. The returned `loop.network`
carries each unit's purity response as the LINEAR model above (a delta
vector), so `loop.network.solve()` is differentiable in either mode with the
units' response to first order. The reformer is differentiated in forward
mode (`jax.jacfwd`; its beds are `diffrax.ForwardMode`): rebuild the producer
inside the function, `loop.network.replace(producers=[Producer.from_reformer(r)])`.
`response_step=` gets each slope from one extra solve when the loop itself
did not produce two purities.

### Hydrogen network: planning

`h2_block(network, levers, outputs)` wraps a network for `difflow.planning`.
Levers: `<producer>.h2` (mol/s), `<producer>.purity` (mol%),
`<consumer>.makeup` (Nm3/h), `<header>.min_purge` (mol/s),
`<header>.psa.recovery`. The units match `reformer_block`'s `h2.net_mol_s`
and `h2.purity` and `hdt_block`'s `h2.makeup`, so `link_reformer()` and
`link_hdt("nht")` give one-to-one links. Outputs are any network outputs
(default: surplus, purities and margins, fuel gas, swing H2). A planning
producer keeps the base impurity mix and moves only its H2 and purity.

### Hydrogen network: what is tested, and what is not

Per commit (`tests/refinery/test_hydrogen.py`, a few seconds): the
balances close to 1e-12 on every configuration (two producers, PSA,
purity target, ordered swing with capacity, deficit, two headers with
export, purity response); the PSA split, its target, and its failure to
reach an impossible one; the specs; the species mapping from the reformer;
`jit`, reverse and forward mode agree; `h2_block` reproduces the network.
Release: the network's Jacobian against central differences.
Slow (`tests/refinery/test_hydrogen_loop.py`): the acceptance case, the
reformer's net gas through an import-swing header into a kerosene and a
diesel hydrotreater with the makeup purity fed back, closing to `1e-7` in
purity with every unit solved at the purity the header delivers it; and
(release) d(surplus)/d(WAIT) through the reformer and the closed network
against central differences.

Not done: compression power and header pressure drop (supply pressures are
only checked to be at least the header's, `res.feasible["<h>.pressure"]`);
a cascade of one header's purge into another; consumer purges routed back
to the header (they can be added as a `Producer.from_flows` and closed by
the same substitution); a hydrogen-pinch targeting or a network
superstructure optimisation; the HDT's traced makeup composition (see AD mode).

### Hydrogen network: references

| What | Source | Status |
|---|---|---|
| Source-sink hydrogen network with purifier and purge to fuel (background) | Alves, J.J., Towler, G.P., "Analysis of refinery hydrogen distribution systems", *Ind. Eng. Chem. Res.* 41(23), 5759-5769 (2002) | Not consulted (unverified); the superstructure is the usual one and no number of the paper is used or reproduced. |
| PSA recovery 0.88, product 99.9 mol% | none | Illustrative defaults, inside the range usually quoted for refinery PSA units (unverified). Set them from the unit's data. |
| Nm3 at 0 C, 1 atm | `MOL_PER_NM3 = 101325/(R 273.15)` | As in the hydroprocessing blocks. |

---

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
| LVGO / HVGO / slop / residue (kg/s) | 4.67239 / 19.5862 / 1.98589 / 39.9512 | same | ≤ 1e-7 rel (1e-6) |
| LVGO / HVGO pumparound duty (MW) | 2.8208 / 13.6600 | same | ≤ 1.3e-7 rel (1e-6) |
| Furnace duty (MW), vapour fraction | 13.1119, 0.30336 | same | 2e-10, 4e-8 rel (1e-6) |
| Stage temperatures, flash zone 669.54 K | | | ≤ 1e-5 K (1e-4 K) |
| Product TBP 5-95 % points; SGs | | | ≤ 1e-5 K (1e-4 K); ≤ 3e-9 (1e-8) |

The second case is Murphree beds: LVGO 80 %, HVGO 70 %, wash 50 %, stripping 40 %. It agrees just as closely: rates within 3e-7, duties within 1.6e-7, temperatures within 2e-5 K. With these beds the LVGO rate rises to 15.27 kg/s and HVGO falls to 7.98 kg/s, so the comparison exercises the efficiency path, not an unchanged column. The case needs a 520 C LVGO end point. With beds this poor, enough heavy vapour reaches the LVGO section that the 450 C spec cannot be met at any pumparound duty. Both implementations stop finding a column halfway from equilibrium to these efficiencies: difflow does not converge and IPOPT reports local infeasibility. That is consistent with the existing infeasible-spec test, although a local NLP verdict is not a proof. The reference reaches this case by continuation from its own equilibrium solution.

The remaining ~1e-7 differences come from one coefficient. The reference writes the Watson-K correction's coefficient as 2.5/1.8 in Rankine, and difflow uses 1.3889 in SI.

The reference also measures what two of difflow's numerical choices cost. These are not disagreements; each change applies to both implementations alike:

- **Four Watson-K passes instead of the exact fixed point.** The fixed point moves ln Psat at the top stage by 6e-3, but the products by under 1e-8.
- **Blending Maxwell-Bonnell's branches instead of switching at the published joins.** This keeps a derivative everywhere. It moves the LVGO rate by 8.7e-4, the pumparound and furnace duties by 8-9e-4, and the flash zone by 0.01 K.

**Sensitivities.** `jax.jacfwd` through difflow's solve was compared with central differences of the reference, with IPOPT warm-started at each perturbed point. The perturbations were furnace outlet ± 0.25 K, flash-zone pressure ± 10 Pa and steam ± 2.5e-5 kg/kg. Over the rates, duties, flash-zone temperature and HVGO T95 they agree within 4.2e-5 relative (tolerance 2e-4); the remaining difference is the finite differences' own truncation error. Two examples are d(HVGO rate)/d(furnace T) = 0.15506 kg/s/K and d(HVGO pumparound duty)/d(steam) = 235.01 MW per kg/kg.

**What this does not validate.** It does not test the property model. Maxwell-Bonnell with Raoult at 10-30 mmHg is a choice that nothing here tests against data or an equation of state; the crude unit's layer 2 is the nearest evidence. It does not cover a commercial simulator's vacuum characterisation, packing HETP and pressure-drop models, or the ejector system, and it does not replace plant data.

(refinery-preheat-validation)=
### Validation: the preheat train

The drum and the exchangers are checked against IDAES 2.10 unit models (`tests/refinery/reference/preheat_generate.py` writes `preheat_reference.json`). `test_preheat_validation.py` (release) compares against the file. `test_preheat_validation_file.py` runs on every commit and checks that the file is intact and the characterisation unchanged. As for the column, this is **an independent implementation, not an independent model**. IDAES is given difflow's property model (Raoult over Lee-Kesler, the cubic ideal-gas Cp, Watson liquid enthalpy, water vapour-only) on the same pseudo-components. Its liquid `(P - P_ref)/rho` term is switched off, because difflow's model has none.

| Check | IDAES model | Agreement (test tolerance) |
| --- | --- | --- |
| Adiabatic drum: dry crude from 500 K and 15 bar to 3 bar | `Flash` | T within 2e-6 K (1e-5); vapour fraction 0.3052, 3e-8 rel (1e-6); vapour composition 2e-9 (1e-8) |
| Wet crude at drum states (470 K, 3 bar; 500 K, 2 bar), all water vapour | state block | vapour fraction and vapour water fraction 2e-11 rel (1e-9) |
| E7 and E8 at the base case's inlets (residue against drum liquid; E8's crude starts to boil) | `HeatExchanger`, counter-current, exact LMTD | duty 3.5e-10 rel (1e-8); outlet temperatures 8e-8 K (1e-6) |

What the check does not cover:

- **The free-water branch of the split.** IDAES's package carries water as vapour-only. That branch is checked by hand in the per-commit tests: the drum at 380 K and 3 bar leaves free water, at 430 K none, and the split is continuous between.
- **The F-factor.** IDAES's exchanger is pure counter-current. The F-factor is difflow's `lmtd_correction_factor`, which `difflow.units.heat_exchanger` already tests.
- **The coupled train and column.** That is checked against itself. The balances close to 1e-12, and the implicit gradients match central differences: `jax.jacfwd` of the furnace inlet temperature, the fired duty and the drum vapour with respect to E6's area, E8's `R_f`, the drum pressure and the 40 % TBP point agree within 1e-5 relative (`test_preheat.py`, release).

**Published case study: none found.** The issue named Polley, Wilson, Yeap and Pugh (2002) as a candidate. No preheat-train study was found that publishes a train's full data (assay, exchanger areas and U values, hot-stream rates) in a form that could be set up here, so nothing is reproduced. The eight-exchanger layout is a textbook one, and its numbers are not a validation.

(refinery-gasplant-validation)=
### Validation: the gas plant

The gas plant's columns are checked against IDAES 2.10 (`tests/refinery/reference/gasplant_generate.py` writes `gasplant_reference.json`). `test_gasplant_validation.py` (release) compares a fresh difflow solve against the file. `test_gasplant_validation_file.py` runs on every commit: it checks that the file is intact, that the component constants and cases are the ones it was built on, and the state-point comparison below, which needs no column solve. As for the crude unit, this is **an independent implementation, not an independent model**. IDAES's generic Peng-Robinson package (its `Cubic` EOS, `SmoothVLE`, log-fugacity equilibrium) is given difflow's constants (Tc, Pc, omega, ideal-gas Cp) with kij = 0. Both sides use equilibrium trays (`tray_efficiency=1.0`), a total condenser at the bubble point, a kettle reboiler and no pressure drop, with the reflux and boilup ratios fixed. IDAES signs the condenser duty negative; difflow reports the heat removed.

| Case | IDAES model | Agreement (test tolerance) |
| --- | --- | --- |
| Debutanizer: C3 to nC5, 10 bar, 10 trays, R = 2, boilup 2 | `TrayColumn`, IDAES's own initialization | product compositions 7.5e-8 rel (1e-5); condenser and reboiler duties 1.2e-8 and 2.4e-8 rel (1e-5); stage temperatures 6.8e-7 K (1e-4); K-values at IDAES's own (T, P, x, y), 1.4e-13 rel (1e-6) |
| C3/C4 splitter: C2 to nC4 with propylene, 17 bar, 20 trays, R = 5, boilup 3 | state block: a TP flash of each of difflow's 21 stage states (z, T, P), from IDAES's own initialization | K 7.1e-7 rel (1e-5); vapour fraction 3.3e-5 (1e-4); phase compositions 1.1e-6 (1e-5); phase enthalpies 2e-9 J/mol (1e-3) |

Two things the reference had to work around, both on the IDAES side:

- **SmoothVLE's smoothing.** At IDAES's default smoothing parameters the bubble-point condenser outlet is left 4e-4 vapour. The total condenser's ports carry the liquid composition at the total flow, so the condenser loses components while the total balances: 0.07 % of the propane, and a 0.7 % gap in the products. The generator tightens the parameters by continuation to eps_2 = 1e-9. The gap then falls to the 1e-7 in the table, which is the evidence it was all smoothing.
- **The splitter's column.** IDAES's `TrayColumn` was not converged on the C3/C4 splitter. Its initialization fails at the "column section + condenser" step for every variant tried: with and without ethane and propylene; 10 to 17 bar; 10 to 20 trays; reflux/boilup from 2/2 to 5/3. Starting every state block from difflow's profile did not help either. At the case's ratios IPOPT ends infeasible. At 4.0/2.5 it reports optimal on a spurious solution with two trays single phase (x = y on `SmoothVLE`'s branch). A reference seeded from difflow's answer would not have been independent of it anyway. So **the issue's 1 % column-level check is met for the debutanizer only**. For the splitter the comparison stops at the thermodynamics. The release test checks that difflow's column still puts its stages at the recorded states.

Regenerate (needs IDAES and IPOPT; `--case NAME` redoes one case):

```bash
PYTHONPATH=src:tests python -m refinery.reference.gasplant_generate
```

**What this does not validate.** It does not test how well PR with zero kij describes these mixtures. That is the propylene/propane split above all, where the relative volatility is near 1.1 and a small kij moves the trays needed. It does not test the O'Connell efficiency, the GPA 2140 limits, the RVP construction against measured RVPs, or the compressor. Those are tested against their definitions in `test_gasplant.py`, not against a second simulator or plant data.

(refinery-isomerization-validation)=
### Validation: the isomerization unit

The reactor's thermochemistry is checked against IDAES 2.10's
`GibbsReactor` (`tests/refinery/reference/isom_generate.py` writes
`isom_reference.json`). IDAES minimises the total Gibbs energy subject to
element balances, on an ideal-gas modular property package given difflow's
heats of formation, entropies and Cp cubics. As for the other units, this
is **an independent implementation, not an independent model**. It checks
how the free energies are assembled, the equilibrium-constant convention
(bar against a 1 bar standard state), the hydrogen-pressure dependence, the
reactor's energy balance, and that the rate law relaxes onto the
equilibrium it claims. It does not check the constants: both sides are
given the same ones.

A Gibbs minimiser given only C and H would turn pentanes into hexanes and
butanes, which no reaction here does. So each conserved carbon skeleton
gets its own element label (`C5`, `C6`, and one per species the network
holds fixed). The minimisation is then over exactly the reactor's reaction
space.

| Case | difflow side | Agreement (test tolerance) |
| --- | --- | --- |
| Each isomer family alone (C5, C6 paraffins, C6 naphthenes), 400-550 K | closed form `family_equilibrium` | 6.8e-13 (1e-10), per commit |
| C6 ring: H2, benzene, MCP, CH, n-hexane at 30 bar, 420 and 480 K | the reactor, isothermal, rate constants x 1e4, no cracking | mole fractions 1.4e-15 (1e-10) |
| Adiabatic: both feeds' reactor charge, 140 C, 30 bar | the reactor, adiabatic, rate constants x 10, LHSV 0.1, no cracking | outlet T 477.023 K (paraffinic) and 522.225 K (benzene-rich), within 1e-6 K; mole fractions 6.9e-14 (1e-10) |

`test_isomerization_validation.py` (release) runs difflow against the file.
`test_isomerization_validation_file.py` runs on every commit. It checks
that the file is intact and that the constants and feeds are the ones it
was built on. It also checks that the IDAES answers conserve atoms, close
difflow's own enthalpy balance, and match the closed-form families.

Regenerate (needs IDAES and IPOPT):

```bash
PYTHONPATH=src:tests python -m refinery.reference.isom_generate
```

The rest is checked against difflow itself, in `test_isomerization.py`,
`test_isomerization_dih.py` and `test_isomerization_dih_gradients.py`:

- The total mass and per-carbon-number balances close to 1e-8 or better, once-through and with the DIH, on both feeds. The reactor alone closes to 1e-12.
- The implicit gradients of RON, MON, volume yield, H2 consumption and gas make (once-through), and of RON, MON, yield, DIH duty and H2 make-up (DIH), with respect to `T_in`, `LHSV` and `x_nC6`, match central differences to 1e-5 relative. With the DIH, the worst entry is 4.4e-6 on the benzene-rich feed; the test runs the paraffinic one.
- At equilibrium, the 2,2-DMB and isopentane shares fall with temperature. RON has an interior maximum in `T_in` on both feeds. The DIH raises RON on both feeds.

**Published case study: none found.** No published isomerization case was
found that gives a feed analysis, catalyst, conditions and product analysis
complete enough to set up and reproduce here. The search was not
exhaustive. So the unit's absolute octanes and yields are not validated
against any plant or published simulation.

**What this does not validate.** It does not test the thermochemical
constants against a tabulated free-energy set. It does not test the
species octanes, which are recalled values marked verify. It does not test
the rate constants, which are illustrative, or the constructed feeds,
which are assumed.

---

(refinery-limitations)=
## Limitations

- **The reformer's kinetics are illustrative.** The rate-law forms and activation energies are Smith's (1959, unverified transcription); the pre-exponentials are this module's, chosen for plausible behaviour, and no published commercial-reformer simulation is reproduced. Pure-compound octanes other than the reference fuels are recalled, not checked against ASTM STP 225. The stabilizer is a component split, not a column. See [Catalytic reforming](#refinery-reforming).
- **The crude unit is the atmospheric column only.** The preflash drum and preheat train are not modelled; the inlet is the preheat train's outlet. The vacuum unit is a separate operation, fed from the crude unit's residue in a `Flowsheet` (above).
- **Thermodynamics:** Raoult's law and ideal-gas-path enthalpies. This is the usual model for an atmospheric column at one or two bar; it is not a cubic equation of state.
- **Equilibrium stages.** There are no tray efficiencies or hydraulics.
- **Boiling ranges are TBP, not ASTM D86.**
- **FCC:** illustrative kinetics (no published parameter set reproduced), a simplified main fractionator (a TBP split, not a `StageColumn`), no gas plant, no 10-lump scheme, no literature cross-check. See [FCC: what is tested, and what is not](#refinery-fcc-validation).
- **Composition is correlated, not measured.** Hydrocarbon types and hydrogen come from Riazi-Daubert / Goossens (or n-d-M) unless the caller gives PIONA, SARA or hydrogen data; the default sulfur- and nitrogen-class splits are illustrative. The MNL50 worked examples for those correlations are not reproduced (see [the composition section](#refinery-composition)).
- **Hydrotreater kinetics are illustrative** (rate forms from the literature, constants chosen for CoMo-like trends), and the Korsten-Hoffmann cross-check is not done; see [the hydrotreater](#refinery-hydrotreater-not-done).
- **Validation:** against an independent equation-oriented model, IDAES property packages and published characterisation examples; not against a commercial simulator's crude case. The vacuum column likewise, against an independent Pyomo/IPOPT model on the same residue (equilibrium and Murphree beds, and sensitivities); not against DWSIM. See [Validation](#refinery-validation) and [the vacuum unit's](#refinery-vacuum-validation) for what that does and does not establish.
- **The crude unit is the atmospheric column; the preheat train is optional.** `CrudeUnit` and `CrudeDistillationUnit` take the crude at the furnace inlet. `PreheatedCrudeUnit` and `CrudeUnitWithPreheat` add the train, desalter and preflash drum from the tank ([above](#refinery-preheat)). The vacuum unit is a separate operation, fed from the crude unit's residue in a `Flowsheet` (above).
- **The preheat train has no hydraulics or geometry.** Its exchangers are `U`, area and `R_f`. Film coefficients, pressure drops, and the Re, Pr and wall shear the fouling model needs are inputs, not computed. The fouling constants are illustrative. A pinched exchanger (hot-side NTU of a few hundred) cannot be solved ([gotchas](#refinery-preheat)).
- **Thermodynamics:** Raoult's law and ideal-gas-path enthalpies. This is the usual model for an atmospheric column at one or two bar; it is not a cubic equation of state.
- **Equilibrium stages.** There are no tray efficiencies or hydraulics.
- **Boiling ranges are TBP, not ASTM D86.**
- **Validation:** against an independent equation-oriented model, IDAES property packages and published characterisation examples; not against a commercial simulator's crude case. The vacuum column likewise, against an independent Pyomo/IPOPT model on the same residue (equilibrium and Murphree beds, and sensitivities); not against DWSIM. The gas plant's debutanizer against IDAES's `TrayColumn` on PR, its C3/C4 splitter at the thermodynamic level only. See [Validation](#refinery-validation), [the vacuum unit's](#refinery-vacuum-validation) and [the gas plant's](#refinery-gasplant-validation) for what that does and does not establish.
- **Isomerization:** the rate constants are illustrative, the feed speciation is constructed and the species octanes are recalled (verify). The stabilizer is a shortcut and the hydrogen is once-through. Validated against IDAES's `GibbsReactor` on the same thermochemistry; no published case study was found ([validation](#refinery-isomerization-validation)).
