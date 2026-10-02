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
- **The fluid catalytic cracker** (`difflow_refinery.fcc`): a lumped-kinetics riser (3-, 4- or 5-lump) and a coke-burning regenerator solved together as the unit's heat balance (catalyst circulation and regenerator temperature are unknowns, the riser outlet temperature the spec), with a simplified main fractionator; dry gas, C3/C4 olefin streams, gasoline, LCO and slurry. A library, not a palette operation; its kinetic constants are illustrative. See [The fluid catalytic cracker](#refinery-fcc).
- **Product blending** (`BlendPool`, `BlendComponent`): gasoline, jet, ULSD and fuel-oil pools with the nonlinear blending rules, signed spec margins and LP back-off. A library for optimisation and planning, not a palette operation.
- **Catalytic reforming** (`difflow_refinery.reforming`): a semi-regen reactor train with fired heaters, a PR separator, H2 recycle through a `Flowsheet` tear and a stabilizer; naphtha P/N/A by carbon number in, reformate (with RON from composition), net H2, LPG and fuel gas out. A library and flowsheet, not a palette operation.

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

**Main fractionator -- simplified.** The issue asks for a `StageColumn` layout with pumparounds, side strippers and a bottom quench. **That is not what is built.** The fractionator is a smooth TBP split of the product pseudocomponents at two cut points, `gasoline_cut` (221 °C) and `lco_cut` (343 °C): each pseudocomponent goes to the lighter product with fraction `sigmoid((T_cut - Tb)/w)` (`split_width` 6 K), which mimics real product overlap and keeps the cut points differentiable. The gases are split ideally into dry gas, C3 and C4 -- standing in for the gas plant of issue #312, which does not exist. Mass is conserved exactly; there is no energy model of the fractionator (no condenser or pumparound duties), and the energy balance covers riser and regenerator only.

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
- **HCO/slurry recycle**, stripper (entrained hydrocarbons in coke), carbon on regenerated catalyst, NOx/NH3/HCN, pressure drop along the riser, and the gas plant (#312): not modelled.
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

For products of the crude and vacuum units, use `BlendCharacterization.from_characterization(char)` instead. It takes the crude characterization's Tb, SG, MW, its critical constants and its vapour-pressure acentric factor, so the pool's Raoult RVP sees `psat(Tb) = 1 atm` exactly as the columns do. It also takes the sulfur vector (and, with `contaminants=True`, nitrogen and CCR) as the `S_ppm`, `N_ppm` and `CCR_wt` qualities. Qualities the assay did not give are left out. If the characterization carries a composition (`char.composition`), its hydrocarbon types become the `paraffins_vol`, `naphthenes_vol`, `aromatics_vol` and `olefins_vol` qualities (vol%); `composition=False` leaves them out. A product stream from either column is then a `BlendComponent.from_stream` input; `F_water` and `F_H2O` are ignored. The pool's sulfur for LVGO or HVGO equals the vacuum column's own report to 1e-10, since both average the same per-component vector.

### Not in scope

- Tank inventory and multi-period scheduling. The pool is steady state, per period.
- Crude blending ahead of the CDU (assay mixing).
- A straight-run octane correlation from PNA. Octane is unit-reported or measured.

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
- **`Stabilizer`**: a **documented simplification**. Issue #312's gas-plant debutanizer does not exist, and difflow's stage columns are not set up for a hydrogen-bearing feed. It is a component split instead: H2, C1 and C2 to fuel gas, C3 and `c4_recovery` of the butanes to LPG, the rest to stabilized reformate. `c4_recovery` stands in for the RVP / C4-in-reformate spec. Its duty is the net heat on the enthalpy basis, not a column design.

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

`res.outputs()` (units in `reforming.OUTPUT_UNITS`) includes reformate and C5+ yield (vol%, wt%), RON/MON (RT-70 and linear), aromatics, benzene, RVP, SG, net H2 (mol/s, wt% of feed, purity), LPG and fuel gas, every reactor's ΔT and outlet temperature, heater absorbed and fired duties, compressor power, separator duty, coke make, cycle length, WAIT and WABT. `res.balances()` returns the overall mass, carbon, hydrogen and energy closures and each reactor's adiabatic residual.

**Reformate properties from composition** (`reforming.products`). RON and MON are the pure-compound octanes of the species, blended with the Ethyl RT-70 rule that `BlendPool` uses. Aromatics and benzene are standard liquid volume fractions. RVP is `raoult_rvp` (D323 geometry) on Lee-Kesler vapour pressures. `products.blend_component("reformate", res.flows("reformate"))` hands it to a `BlendPool`. Pure-component octanes are not blending octanes; the RT-70 rule with the large aromatic and sensitivity spreads of a reformate puts RON 8-12 above the linear average. The low-octane paraffins sit below the range RT-70 was fitted on, so the rule extrapolates there.

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
- **No example notebook** (naphtha hydrotreater -> reformer -> gasoline pool). The hydrotreater (#306) is being built separately, and the notebook was not written.
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
- **Validation:** against an independent equation-oriented model, IDAES property packages and published characterisation examples; not against a commercial simulator's crude case. The vacuum column likewise, against an independent Pyomo/IPOPT model on the same residue (equilibrium and Murphree beds, and sensitivities); not against DWSIM. See [Validation](#refinery-validation) and [the vacuum unit's](#refinery-vacuum-validation) for what that does and does not establish.
