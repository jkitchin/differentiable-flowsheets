# Water Plugin (difflow_water) - Design Document

Status: **phase 1 implemented** (seawater properties, `ROStage`,
`PressureExchanger`, `WaterPump`, LCOW, the SWRO + PX plant; WaterTAP
comparison in `benchmarks/watertap/`, within 0.3 %). Phases 2-4 are plans.
See "Phase 1 as built" in Section 5. This document is the plan and the
contract for `difflow_water`: water distribution networks, drinking-water
treatment, membrane and thermal desalination, wastewater, and the shared
water chemistry they all need.

**Decisions made** (2026-11-03):

- **Wastewater lives in `difflow_water`** (not `difflow_bio`).
- **WaterTAP is the benchmark.** Every unit WaterTAP also models is
  cross-checked against WaterTAP on matched inputs, and the agreement is
  recorded in the docs (Section 2a). The aim is to be comparable to WaterTAP
  and differentiable end to end.
- **Streams are difflow-compatible**: per-component molar flows (`F_H2O`,
  `F_Na+`, ...) plus `T` and `P`, as `make_stream` produces, with mg/L, TDS
  and volumetric-flow helpers on top.
- **Network valves get both formulations**: smoothed switching and
  complementarity, selectable per valve (Section 3a).

It follows the conventions `difflow_refinery` established (see
`docs/unit-operations-refinery.md` and `docs/refinery-summary.md`):
every unit ships with **code, documentation, an example notebook and tests**,
and every equation, coefficient and data value is tied to a **reference whose
verification status is recorded**.

---

## 1. Requirements every unit must meet

A unit is not done until it has all six of the following.

1. **Code**: a `Params` dataclass on `ParamsMixin` with an `Attributes:`
   docstring and `parameter_units`, and a JAX-only `__call__`. Its NumPy-style
   docstring has a `References` section naming reference IDs from Section 6
   (e.g. `[S2]`, `[M4]`). A coefficient with no source carries the comment
   `# ILLUSTRATIVE` and is listed as illustrative in the docs.
2. **Documentation**: a section in `docs/unit-operations-water.md` under a
   heading naming the unit (or a `(op-<name>)=` label), so
   `tests/test_doclinks.py` passes after `make docs-index`. Each section
   gives the equations, inputs, outputs, validity range, known gaps and a
   **References** table (Section 2 format). A row in the one-page map
   `docs/water-summary.md` (modelled on `docs/refinery-summary.md`) links to
   it.
3. **Example**: a notebook in `examples/` (numbered after the current last
   one, `69_...`, or under `examples/water/` as `bio` does). It runs the
   unit, takes a gradient, and shows one design or sensitivity question.
   `make notebooks` executes it.
4. **Tests** in `tests/water/`, following the project policy: a smoke test,
   a finite `jax.grad` test, one test per input-validation path, and any
   fixed crash/NaN regression. **No comparisons to literature values.** The
   docs table records what has been cross-checked; tests do not.
5. **Registration**: palette units go in `register()` and `__all__`;
   library-only functions (e.g. the chemistry module, network builders) say
   so in the plugin `CLAUDE.md`, as the refinery does for `BlendPool`.
6. **Bibliography**: each reference is added to `docs/literature/references.bib`
   with the same key the docs use.

## 2. Citation verification protocol

The refinery docs separate *checking the citation* from *checking the
content*. A DOI that resolves does not mean the equation as coded matches
the paper. Every reference row records its status using these terms:

| Status | Meaning |
|---|---|
| **Citation verified (Crossref)** | Authors, title, journal, year, volume, pages and DOI were matched against the Crossref record for the DOI. Says nothing about the content. |
| **Content verified** | The equation, table or coefficient as coded was read against the source (with page, equation or table number recorded) by someone who had the paper. |
| **Cross-checked** | Output reproduced against an independent implementation (WaterTAP, WNTR/EPANET, PHREEQC, a published worked example), with the case and agreement stated. |
| **(unverified)** | Anything recalled, taken second-hand (from another code or a review), or not reached. Marked in the docs **and** in a code comment. |
| **ILLUSTRATIVE** | A value with no source, chosen to give plausible behaviour. Must be fitted before the model is used to predict. |

Rules:

- Never put a value in code without its reference ID or the word ILLUSTRATIVE.
- A second-hand constant (e.g. copied from WaterTAP or WNTR source code) is
  cited to the primary paper **and** to the code it came from, and stays
  (unverified) until read in the primary source. This mirrors the refinery's
  treatment of constants taken from pychemqt.
- Re-run the Crossref check (Section 6.1) whenever a reference is added.
  The check already caught one wrong DOI in this plan: the DOI recalled for
  Swamee & Jain (1976) belonged to a different paper.
- Software references (EPANET, WNTR, WaterTAP, PHREEQC) are used for
  cross-checks, not as the source of the physics, wherever a primary
  paper exists.

## 2a. WaterTAP comparisons

WaterTAP (the NAWI water treatment model library, built on Pyomo/IDAES
[B1, B2, B3]) is the benchmark this plugin must be comparable to.

- **Which units**: every unit that has a WaterTAP counterpart (RO and its
  0-D/1-D forms, NF/DSPM-DE, pressure exchanger, pumps, ED, MVC/MED,
  crystallizer, UV/AOP, ion exchange, GAC, the activated-sludge and ADM1
  models, the clarifier, LCOW costing). Units with no counterpart say so.
- **How**: a script under `benchmarks/watertap/` (repo root) builds the same
  case in WaterTAP (pinned version, recorded with the IPOPT version), solves
  it, and writes a JSON reference, as `isom_reference.json` does for the
  refinery's IDAES check. difflow runs the same inputs and the docs report
  the differences per output.
- **Not a test.** The test policy forbids reference-data comparisons, and
  WaterTAP is not a dev dependency. The comparison is a `make` target run by
  hand when a unit changes; a reference that a unit change invalidates is
  marked stale in the docs until regenerated (the refinery's
  `STALE_SINCE_339` pattern).
- **Where WaterTAP and the primary paper disagree**, the paper wins, and the
  docs say what WaterTAP does differently.
- **Property packages**: where WaterTAP uses a property correlation (e.g.
  seawater [S1, S2]), use the same primary source so differences come from
  the models, not the property data.
- **Stream mapping**: WaterTAP uses mass-flow-per-component state variables;
  `streams.py` has the converter so a comparison feeds identical streams.

## 3. Module structure

Modelled on `difflow_refinery` (package-level docs, subpackages each with
their own `CLAUDE.md` once they have invariants) and on `difflow_gas` (the
network graph and its decomposition).

```
src/difflow_water/
├── CLAUDE.md                # invariants, loaded when working here
├── DESIGN.md                # this file
├── README.md                # overview + palette/library table (as refinery)
├── __init__.py              # exports, __all__, register()
├── agent.py                 # agent support entry point
├── streams.py               # make_water_stream(ions=..., T, P) -> a plain difflow
│                            #   stream (F_<species> mol/s, T, P); mg/L, TDS, m3/s and
│                            #   WaterTAP mass-flow converters
├── chemistry/               # library
│   ├── seawater.py          # density, viscosity, osmotic pressure, Cp, BPE [S1, S2]
│   ├── activity.py          # Davies, Pitzer [C3, C5]; reuse difflow.activity
│   ├── carbonate.py         # CO2-HCO3-CO3 equilibrium, alkalinity [C2]
│   └── scaling.py           # LSI, saturation indices (calcite, gypsum) [C1, C2]
├── network/                 # distribution (library + palette units)
│   ├── network.py           # WaterNetwork graph, decomposition (from difflow_gas)
│   ├── residuals.py         # the one equation set
│   ├── hydraulics.py        # Hazen-Williams, Darcy-Weisbach, Colebrook [N4, N7]
│   ├── units.py             # WaterPipe, Pump, Tank, Reservoir, Junction, Leak
│   ├── valves.py            # PRV, PSV, FCV, TCV, check valve; smooth | complementarity
│   └── quality.py           # chlorine decay, water age [N5]
├── treatment/               # conventional drinking water
│   ├── coagulation.py, sedimentation.py, filtration.py
│   ├── disinfection.py      # Chick-Watson CT [K1, K2]
│   ├── adsorption.py        # GAC breakthrough [A1]
│   └── ion_exchange.py, softening.py, stripping.py
├── membranes/               # RO, NF, UF/MF, pressure exchanger
│   ├── ro.py                # wraps difflow.units.membranes.ReverseOsmosis [M1, M4]
│   ├── nf.py                # DSPM-DE [M2, M3]
│   └── erd.py               # pressure exchanger [M5]
├── thermal/                 # MSF, MED, MVC [D2]; reuse core evaporator.py
├── electro/                 # electrodialysis [E1, E2]
├── wastewater/              # ASM1/ASM2d [W3], Takacs settler [W1], ADM1 [W2]
├── economics/               # LCOW, specific energy; on difflow.economics
└── flowsheets.py            # SWRO train, conventional plant, BSM1 builder
```

Wastewater (ASM1/ASM2d, settler, ADM1, BSM1) is part of this plugin.

Docs: `docs/unit-operations-water.md` (full treatment, reference tables per
section), `docs/water-summary.md` (one-page map with a "how validated" column
and a "Known gaps" section). Tests: `tests/water/`. Plugin entry points in
`pyproject.toml` alongside `gas` and `refinery`.

## 3a. Network valves: two formulations

Valves switch mode (a PRV is open, active or closed; a check valve blocks
reverse flow), which puts kinks or jumps in the equations. Both standard
treatments are implemented, chosen per valve with
`formulation="smooth" | "complementarity"`:

- **Smoothed switching** (default for the sequential/fixed-point solve): each
  mode change is replaced by a smooth blend with a width parameter
  (e.g. a softplus or logistic in the pressure or flow error), so the
  residuals are differentiable everywhere and `jax.grad` works through any
  mode. The width is a documented parameter; the docs state how far the
  smoothed answer sits from the sharp one at the default width.
- **Complementarity** (for equation-oriented solves and optimization): the
  mode is written as complementarity conditions `0 <= a  perp  b >= 0`,
  solved through a smoothing or Fischer-Burmeister reformulation whose
  parameter is driven to zero. Gradients come from the implicit function
  theorem at the solution and are valid away from mode boundaries (strict
  complementarity); at a boundary the result is flagged in `info`.

Both share one residual definition in `network/residuals.py` and are
checked to agree on the same network away from switching points. Primary
references for each (beyond the EPANET manual) are still to find
(Section 6.3).

### What to reuse from core (do not duplicate)

- `difflow.units.membranes`: `ReverseOsmosis`, `osmotic_pressure`,
  `ro_local_flux` (concentration polarization), `uf_flux`.
- `difflow.units.pipe.Pipe`, `pump.py`, `evaporator.py`, `crystallizer.py`.
- `difflow.activity`; `difflow_ree.equilibrium.speciation` for electrolyte
  speciation patterns.
- `difflow_gas.network` / `residuals` / `reconcile` as the architecture
  template; `difflow.reconciliation` for calibration and leak localization.

## 4. Units and the references behind them

| Area | Unit / function | Model | Refs | Palette? |
|---|---|---|---|---|
| Chemistry | seawater properties | Sharqawy/Nayar correlations | S1, S2, S3 | library |
| Chemistry | activity coefficients | Davies; Pitzer for brines | C5, C3 | library |
| Chemistry | carbonate system, LSI, saturation index | calcite solubility; Langelier | C2, C1, C4 (cross-check) | library |
| Network | `WaterPipe` | Hazen-Williams; Darcy-Weisbach + Colebrook or Swamee-Jain | N4, N7, (HW: see 6.3) | yes |
| Network | network solve | global gradient algorithm | N2 | library |
| Network | `Junction` pressure-driven demand | Wagner-Shamir-Marks | N3 | yes |
| Network | `Pump`, `Tank`, `Reservoir`, valves | EPANET conventions | EPANET manual (6.3), N1 (cross-check) | yes |
| Network | `Leak` / emitter, calibration | Q = C p^beta; reconciliation | N6 | yes / library |
| Network | chlorine decay | first-order bulk + wall | N5 | library |
| Treatment | coagulation, sedimentation, filtration, softening, stripping | textbook design equations | D1 | yes |
| Treatment | `Disinfection` | Chick-Watson CT | K1, K2, D1 | yes |
| Treatment | `GACAdsorber` | HSDM, simplified | A1, D1 | yes |
| Membranes | `ROStage`, `ROTrain` | solution-diffusion + polarization; spiral-wound mass transfer and pressure loss | M1, M4 | yes |
| Membranes | `Nanofiltration` | DSPM-DE | M2, M3 | yes |
| Membranes | `PressureExchanger` | isobaric ERD efficiency + mixing | M5 | yes |
| Thermal | `MSF`, `MED`, `MVC` | stage/effect balances | D2, T1 | yes |
| Electro | `Electrodialysis` | current, ion flux, limiting current density | E1, E2 | yes |
| Wastewater | `ActivatedSludgeReactor` | ASM1, ASM2d | W3 | yes |
| Wastewater | `SecondaryClarifier` | Takacs layered settler | W1 | yes |
| Wastewater | `AnaerobicDigester` | ADM1 | W2 | yes |
| Wastewater | BSM1 flowsheet | benchmark plant | W4 | library |
| Context | desalination energy, review | | M6, T1 | docs only |

WaterTAP counterpart exists (cross-check per Section 2a) for: RO, NF,
pressure exchanger, pumps, ED, MVC/MED, crystallizer, GAC, ion exchange,
UV/AOP, ASM1/ASM2d, ADM1, clarifier, LCOW costing. Which WaterTAP models
match which units is confirmed against the pinned WaterTAP version when each
unit is written.

## 5. Phasing

Each phase finishes with its docs section, summary row, notebook and tests.

1. **Chemistry + SWRO** (the headline demo): `streams`, `chemistry/`,
   `ROStage`/`ROTrain`, `PressureExchanger`, high-pressure and booster pumps,
   LCOW. Notebook: minimise LCOW over feed pressure, stage areas and
   recovery, subject to a scaling limit; `jax.grad` through the ERD recycle.
   Set up `benchmarks/watertap/` here, with the WaterTAP SWRO flowsheet as
   the first comparison.
   **Phase 1 as built** (2026-11-03): `streams.py`, `chemistry/seawater.py`
   (Sharqawy 2010 Eqs. 8, 22, 23, 49, content verified; `M_S` from TEOS-10),
   `membranes/ro.py` (`ROStage`: one stage; flat-sheet or spiral-wound
   friction), `membranes/erd.py`, `pumps.py`, `economics.py` (WaterTAP 1.8.0
   default costing), `flowsheets.py`, notebook `examples/70_water_swro_design.ipynb`,
   docs `unit-operations-water.md` + `water-summary.md`, tests `tests/water/`.
   Deferred to **phase 1b**: carbonate chemistry, activity models and scaling
   indices (the notebook uses a brine-salinity limit instead of a scaling
   limit), `ROTrain` (multi-stage), NF, pressure-exchanger leakage.
2. **Distribution network** on the `difflow_gas` architecture. Notebooks:
   roughness calibration and leak localization with `reconcile_network`;
   pump scheduling with tanks. Cross-check target: WNTR/EPANET on the
   EPANET example networks.
3. **Conventional treatment train** and **thermal desalination**.
4. **Wastewater** (in this plugin): ASM1/ASM2d, Takacs, ADM1, BSM1 (adds a
   case to the convergence corpus). WaterTAP's activated-sludge and ADM1
   flowsheets are the comparison.

## 6. References

Status as of this document: the **citation** (bibliographic record) was
checked against Crossref by DOI for every row marked "Citation verified
(Crossref)". **No equation or coefficient has been content-verified yet**;
that happens when each unit is written.

### 6.1 How the check was done

Each DOI was fetched from `https://api.crossref.org/works/<doi>` and the first
author's family name, year, volume and first page were compared with the
values below. Re-run it for any new reference.

### 6.2 Reference table

| ID | Reference | Used for | Status |
|---|---|---|---|
| N1 | Klise, K.A.; Bynum, M.; Moriarty, D.; Murray, R. "A software framework for assessing the resilience of drinking water systems to disasters with an example earthquake case study." *Environ. Model. Softw.* **2017**, 95, 420--431. doi:10.1016/j.envsoft.2017.06.022 | WNTR (network cross-check target) | Citation verified (Crossref) |
| N2 | Todini, E.; Rossman, L.A. "Unified framework for deriving simultaneous equation algorithms for water distribution networks." *J. Hydraul. Eng.* **2013**, 139(5), 511--526. doi:10.1061/(ASCE)HY.1943-7900.0000703 | Global gradient network solve | Citation verified (Crossref) |
| N3 | Wagner, J.M.; Shamir, U.; Marks, D.H. "Water distribution reliability: simulation methods." *J. Water Resour. Plann. Manage.* **1988**, 114(3), 276--294. doi:10.1061/(ASCE)0733-9496(1988)114:3(276) | Pressure-driven demand | Citation verified (Crossref) |
| N4 | Colebrook, C.F. "Turbulent flow in pipes, with particular reference to the transition region between the smooth and rough pipe laws." *J. Inst. Civ. Eng.* **1939**, 11(4), 133--156. doi:10.1680/ijoti.1939.13150 | Colebrook friction factor | Citation verified (Crossref) |
| N5 | Rossman, L.A.; Clark, R.M.; Grayman, W.M. "Modeling chlorine residuals in drinking-water distribution systems." *J. Environ. Eng.* **1994**, 120(4), 803--820. doi:10.1061/(ASCE)0733-9372(1994)120:4(803) | Bulk + wall chlorine decay | Citation verified (Crossref) |
| N6 | Ostfeld, A.; Salomons, E.; Ormsbee, L.; et al. "Battle of the water calibration networks." *J. Water Resour. Plann. Manage.* **2012**, 138(5), 523--532. doi:10.1061/(ASCE)WR.1943-5452.0000191 | Calibration benchmark | Citation verified (Crossref) |
| N7 | Swamee, P.K.; Jain, A.K. "Explicit equations for pipe-flow problems." *J. Hydraul. Div.* **1976**, 102(5), 657--664. doi:10.1061/JYCEAJ.0004542 | Explicit friction factor | Citation verified (Crossref, by search; the DOI first recalled was wrong). An erratum exists: *J. Hydraul. Div.* 1978, 104, 300, doi:10.1061/JYCEAJ.0004941 -- read it before coding. |
| S1 | Sharqawy, M.H.; Lienhard, J.H.; Zubair, S.M. "Thermophysical properties of seawater: a review of existing correlations and data." *Desalin. Water Treat.* **2010**, 16(1-3), 354--380. doi:10.5004/dwt.2010.1079 | Seawater properties | Citation verified (Crossref) |
| S2 | Nayar, K.G.; Sharqawy, M.H.; Banchik, L.D.; Lienhard V, J.H. "Thermophysical properties of seawater: A review and new correlations that include pressure dependence." *Desalination* **2016**, 390, 1--24. doi:10.1016/j.desal.2016.02.024 | Seawater properties with pressure | Citation verified (Crossref) |
| S3 | Millero, F.J.; Feistel, R.; Wright, D.G.; McDougall, T.J. "The composition of Standard Seawater and the definition of the Reference-Composition Salinity Scale." *Deep-Sea Res. I* **2008**, 55(1), 50--72. doi:10.1016/j.dsr.2007.10.001 | Reference seawater ion composition | Citation verified (Crossref) |
| M1 | Wijmans, J.G.; Baker, R.W. "The solution-diffusion model: a review." *J. Membr. Sci.* **1995**, 107(1-2), 1--21. doi:10.1016/0376-7388(95)00102-I | RO water and salt flux | Citation verified (Crossref) |
| M2 | Bowen, W.R.; Mohammad, A.W.; Hilal, N. "Characterisation of nanofiltration membranes for predictive purposes -- use of salts, uncharged solutes and atomic force microscopy." *J. Membr. Sci.* **1997**, 126(1), 91--105. doi:10.1016/S0376-7388(96)00276-1 | DSPM | Citation verified (Crossref) |
| M3 | Bowen, W.R.; Welfoot, J.S. "Modelling the performance of membrane nanofiltration -- critical assessment and model development." *Chem. Eng. Sci.* **2002**, 57(7), 1121--1137. doi:10.1016/S0009-2509(01)00413-4 | DSPM-DE | Citation verified (Crossref) |
| M4 | Schock, G.; Miquel, A. "Mass transfer and pressure loss in spiral wound modules." *Desalination* **1987**, 64, 339--352. doi:10.1016/0011-9164(87)90107-X | Spiral-wound k and pressure drop | Citation verified (Crossref) |
| M5 | Stover, R.L. "Seawater reverse osmosis with isobaric energy recovery devices." *Desalination* **2007**, 203(1-3), 168--175. doi:10.1016/j.desal.2006.03.528 | Pressure exchanger | Citation verified (Crossref) |
| S4 | IOC, SCOR and IAPSO. *The International Thermodynamic Equation of Seawater -- 2010* (TEOS-10 manual), IOC Manuals and Guides 56, UNESCO, **2010** | `M_S = 31.4038218 g/mol` | Content verified (value read in the manual). No DOI. |
| M7 | Bartholomew, T.V.; Mauter, M.S. "Computational framework for modeling membrane processes without process and solution property simplifications." *J. Membr. Sci.* **2019**, 573, 682--693. doi:10.1016/j.memsci.2018.11.067 | Salt diffusivity (Eq. 6) | Citation verified (Crossref; the DOI first recalled was wrong). Coefficients via WaterTAP (unverified). |
| M8 | Bartholomew, T.V.; Siefert, N.S.; Mauter, M.S. "Cost optimization of osmotically assisted reverse osmosis." *Environ. Sci. Technol.* **2018**. doi:10.1021/acs.est.8b02771 | Flat-sheet friction factor (SI Eq. S27) | Title/authors/year/DOI verified (Crossref); volume/pages not in Crossref. Correlation via WaterTAP (unverified). |
| M6 | Elimelech, M.; Phillip, W.A. "The future of seawater desalination: energy, technology, and the environment." *Science* **2011**, 333(6043), 712--717. doi:10.1126/science.1200488 | Context, energy bounds | Citation verified (Crossref) |
| T1 | Mistry, K.H.; McGovern, R.K.; Thiel, G.P.; Summers, E.K.; Zubair, S.M.; Lienhard, J.H. "Entropy generation analysis of desalination technologies." *Entropy* **2011**, 13(10), 1829--1864. doi:10.3390/e13101829 | Thermal/RO energy comparison | Citation verified (Crossref) |
| D2 | El-Dessouky, H.T.; Ettouney, H.M. *Fundamentals of Salt Water Desalination*; Elsevier: Amsterdam, **2002**. doi:10.1016/B978-0-444-50810-2.X5000-3 | MSF, MED, MVC models | DOI and title/year verified (Crossref); Crossref lists no authors, so authorship is (unverified) there |
| E1 | Strathmann, H. "Electrodialysis, a mature technology with a multitude of new applications." *Desalination* **2010**, 264(3), 268--288. doi:10.1016/j.desal.2010.04.069 | Electrodialysis | Citation verified (Crossref) |
| E2 | McGovern, R.K.; Weiner, A.M.; Sun, L.; Chambers, C.G.; Zubair, S.M.; Lienhard V, J.H. "On the cost of electrodialysis for the desalination of high salinity feeds." *Appl. Energy* **2014**, 136, 649--661. doi:10.1016/j.apenergy.2014.09.050 | ED model and cost | Citation verified (Crossref) |
| W1 | Takács, I.; Patry, G.G.; Nolasco, D. "A dynamic model of the clarification-thickening process." *Water Res.* **1991**, 25(10), 1263--1271. doi:10.1016/0043-1354(91)90066-Y | Layered settler | Citation verified (Crossref); Crossref lists only the first author, co-authors confirmed from the 1990 conference version (doi:10.1016/B978-0-08-040776-0.50065-2) |
| W2 | Batstone, D.J.; Keller, J.; Angelidaki, I.; Kalyuzhnyi, S.V.; Pavlostathis, S.G.; Rozzi, A.; Sanders, W.T.M.; Siegrist, H.; Vavilin, V.A. "The IWA Anaerobic Digestion Model No 1 (ADM1)." *Water Sci. Technol.* **2002**, 45(10), 65--73. doi:10.2166/wst.2002.0292 | ADM1 | Citation verified (Crossref). The full parameter set is in the IWA Scientific and Technical Report No. 13 (not checked). |
| W3 | Henze, M.; Gujer, W.; Mino, T.; van Loosdrecht, M. *Activated Sludge Models ASM1, ASM2, ASM2d and ASM3*; IWA Scientific and Technical Report No. 9; IWA Publishing: London, 2000. doi:10.2166/9781780402369 | ASM1, ASM2d | DOI and authors verified (Crossref); Crossref dates the e-book reissue 2015, the 2000 print date is (unverified) |
| W4 | Gernaey, K.V.; Jeppsson, U.; Vanrolleghem, P.A.; Copp, J.B. *Benchmarking of Control Strategies for Wastewater Treatment Plants*; IWA Scientific and Technical Report No. 23; IWA Publishing, **2014**. doi:10.2166/9781780401171 | BSM1 definition | Citation verified (Crossref); report number (unverified) |
| D1 | Crittenden, J.C.; Trussell, R.R.; Hand, D.W.; Howe, K.J.; Tchobanoglous, G. *MWH's Water Treatment: Principles and Design*, 3rd ed.; Wiley: Hoboken, NJ, **2012**. doi:10.1002/9781118131473 | Conventional treatment design equations | Citation verified (Crossref); edition (unverified) |
| C1 | Langelier, W.F. "The analytical control of anti-corrosion water treatment." *J. AWWA* **1936**, 28(10), 1500--1521. doi:10.1002/j.1551-8833.1936.tb13785.x | Langelier saturation index | Citation verified (Crossref) |
| C2 | Plummer, L.N.; Busenberg, E. "The solubilities of calcite, aragonite and vaterite in CO2-H2O solutions between 0 and 90 °C, and an evaluation of the aqueous model for the system CaCO3-CO2-H2O." *Geochim. Cosmochim. Acta* **1982**, 46(6), 1011--1040. doi:10.1016/0016-7037(82)90056-4 | Carbonate constants, calcite Ksp(T) | Citation verified (Crossref) |
| C3 | Pitzer, K.S. "Thermodynamics of electrolytes. I. Theoretical basis and general equations." *J. Phys. Chem.* **1973**, 77(2), 268--277. doi:10.1021/j100621a026 | Pitzer activity model | Citation verified (Crossref) |
| C4 | Parkhurst, D.L.; Appelo, C.A.J. *Description of Input and Examples for PHREEQC Version 3*; U.S. Geological Survey Techniques and Methods 6-A43, **2013**. doi:10.3133/tm6A43 | Chemistry cross-check target | Citation verified (Crossref) |
| C5 | Davies, C.W. "The extent of dissociation of salts in water. Part VIII. ..." *J. Chem. Soc.* **1938**, 2093. doi:10.1039/JR9380002093 | Davies activity equation | Citation verified (Crossref) |
| K1 | Chick, H. "An investigation of the laws of disinfection." *J. Hyg.* **1908**, 8(1), 92--158. doi:10.1017/S0022172400006987 | Disinfection kinetics | Citation verified (Crossref) |
| K2 | Watson, H.E. "A note on the variation of the rate of disinfection with change in the concentration of the disinfectant." *J. Hyg.* **1908**, 8(4), 536--542. doi:10.1017/S0022172400015928 | Concentration exponent (CT) | Citation verified (Crossref) |
| B1 | Akkor, I.; Amusat, O.O.; Vecchiarelli, P.; Gounaris, C.E.; Knueven, B.; Dudchenko, A.V. "Optimization of desalination systems with detailed water chemistry through integration of Reaktoro in WaterTAP." *ACS ES&T Eng.* **2026**, 6(5), 1464--1477. doi:10.1021/acsestengg.5c01008 | WaterTAP with detailed chemistry (benchmark) | Citation verified (Crossref) |
| B2 | Dudchenko, A.; Atia, A.; Bartholomew, T. "WaterTAP: Tool for analysis of water treatment systems." NAWI Alliance Fall Meeting, **2023** (OSTI). doi:10.2172/2447624 | WaterTAP (benchmark) | Citation verified (Crossref). A presentation, not a paper: also cite the pinned software release (Zenodo/GitHub) used for each comparison. |
| B3 | Lee, A.; Ghouse, J.H.; Eslick, J.C.; Laird, C.D.; Siirola, J.D.; et al. "The IDAES process modeling framework and model library -- flexibility for process simulation and optimization." *J. Adv. Manuf. Process.* **2021**, 3(3), e10095. doi:10.1002/amp2.10095 | IDAES, which WaterTAP is built on | Citation verified (Crossref) |
| A1 | Hand, D.W.; Crittenden, J.C.; Thacker, W.E. "Simplified models for design of fixed-bed adsorption systems." *J. Environ. Eng.* **1984**, 110(2), 440--456. doi:10.1061/(ASCE)0733-9372(1984)110:2(440) | GAC breakthrough (HSDM) | Citation verified (Crossref) |

### 6.3 Still to find (no DOI in hand; do not cite until resolved)

- Hazen-Williams: Williams, G.S.; Hazen, A. *Hydraulic Tables* (1905 and later
  editions). A book with no DOI; cite the edition read.
- EPANET 2.2 user manual (Rossman et al., U.S. EPA, 2020); report number and
  URL to be confirmed from the EPA site.
- Todini & Pilati (1988), the original global gradient algorithm (book chapter);
  N2 covers the method in the meantime.
- A primary WaterTAP methods paper, if one exists beyond B1/B2, and the
  software release DOI to pin for the comparisons.
- Valve modelling: primary references for (a) the smoothed PRV/PSV/FCV/check
  valve forms and (b) complementarity formulations of network valves.

## 7. Open design questions

- Smoothing width defaults for the smoothed valves, and the solver for the
  complementarity form (smoothing continuation vs Fischer-Burmeister Newton).
- Which WaterTAP version to pin, and whether the comparison scripts live in
  this repo or a separate one (WaterTAP + IPOPT is a heavy dependency).
