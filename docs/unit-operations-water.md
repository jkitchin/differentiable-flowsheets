# Water Treatment and Desalination Unit Operations

![difflow_water](../images/plugins/difflow-water-logo.svg)

This page documents the `difflow_water` plugin: seawater properties, a
Pitzer activity model with seawater speciation and mineral saturation
indices (scaling), a seawater reverse-osmosis (RO) stage and multi-stage
train, a pressure exchanger, water pumps, a levelized-cost-of-water (LCOW)
model, and the seawater RO + pressure exchanger plant built from them. A one-page map is in
[Water units at a glance](water-summary.md); the plan for the rest of the
plugin (distribution networks, conventional treatment, thermal
desalination, wastewater) is `src/difflow_water/DESIGN.md`.

Two rules shape this plugin, the same ones `difflow_refinery` follows:

- **Every equation, coefficient and data value has a reference with a
  recorded verification status** (the [references table](water-references)
  at the end). "Citation verified" means the bibliographic record was
  checked; it says nothing about the content. "Content verified" means the
  coded equation and coefficients were read against the source. Anything
  taken second-hand (from another code or a review) or recalled is marked
  **(unverified)**, here and in the code.
- **WaterTAP is the benchmark.** The units model the same physics as
  WaterTAP's (same property correlations, same flux laws, same costing
  structure and defaults), and `benchmarks/watertap/` runs the same cases
  in WaterTAP so the differences are measured, not assumed
  ([comparison](water-watertap-comparison)). Water chemistry, which
  WaterTAP delegates to Reaktoro, is checked against PHREEQC instead
  ([comparison](water-phreeqc-comparison)).

All of it is differentiable with JAX: `jax.grad` of the LCOW with respect
to the RO pressure, membrane area or permeabilities is exact, through the
RO integration and its local flux solves.

---

(water-streams)=
## Seawater streams

A difflow_water stream is an ordinary difflow stream: molar flows in mol/s
under `F_<species>`, plus `T` (K) and `P` (Pa). Seawater carries two species:

- `H2O`, molar mass 18.01528 g/mol;
- `TDS`, sea salt, counted in moles of the Reference-Composition mean molar
  mass `M_S = 31.4038218 g/mol` [S4, from S3]. One mole of `TDS` is
  `M_S` grams of sea salt, and the molality `S / ((1 - S) M_S)` already
  counts every ion.

```python
import difflow_water as dw

feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, T=298.15, P=101325.0)
dw.salinity(feed)              # 0.035 kg/kg
dw.volumetric_flow(feed)       # 1e-3 m^3/s
dw.concentration_mg_per_L(feed)  # ~35 800 mg/L
dw.to_watertap_state(feed)     # WaterTAP's flow_mass_phase_comp, temperature, pressure
```

Mixing (`combine_seawater`) sums flows, takes the flow-weighted temperature
and the lowest pressure (difflow's `combine_streams`).

---

(water-seawater-properties)=
## Seawater properties

`difflow_water.seawater` (the module `difflow_water.chemistry.seawater`)
implements the correlations of Sharqawy, Lienhard and Zubair [S1]. T is in
K, salinity `S` is a mass fraction (kg/kg); the correlations use ITS-90
Celsius `t` internally.

| Function | Equation | Source | Validity | Stated accuracy |
|---|---|---|---|---|
| `density_water(T)` | `a1 + a2 t + a3 t^2 + a4 t^3 + a5 t^4` | [S1] Eq. 8, S = 0 | 0-180 C | +-0.1 % |
| `density(T, S)` | Eq. 8: pure water + `b1 S + b2 S t + b3 S t^2 + b4 S t^3 + b5 S^2 t^2` | [S1] Eq. 8 | 0-180 C, 0-0.16 kg/kg | +-0.1 % |
| `viscosity_water(T)` | `4.2844e-5 + (0.157 (t + 64.993)^2 - 91.296)^-1` | [S1] Eq. 23 | 0-180 C | +-1.5 % |
| `viscosity(T, S)` | `mu_w (1 + A S + B S^2)` | [S1] Eq. 22 | 0-180 C, 0-0.15 kg/kg | +-1.5 % |
| `osmotic_coefficient(T, S)` | 10-term polynomial in t and S | [S1] Eq. 49 | 0-200 C, 0.01-0.12 kg/kg | +-1.4 % |
| `osmotic_pressure(T, S)` | `phi m rho_w R T` | see below | as Eq. 49 | |
| `salinity_from_mass_concentration(C, T)` | Newton on `S rho(T, S) = C` | -- | | |

**Content verification.** The coefficients of Eqs. 8, 22, 23 and 49 were
read against the authors' corrected version of [S1] (MIT; it carries the
paper's five published corrections, none of which touches these four
equations). Spot values the code reproduces: pure water 996.9 kg/m^3 and
0.890 mPa s at 25 C; seawater (35 g/kg, 25 C) 1023.6 kg/m^3, osmotic
coefficient 0.907, osmotic pressure 25.9 bar.

**Osmotic pressure.** `pi = phi m rho_w R T`, with `m` the molality on the
`M_S` basis and `rho_w` the pure-water density, is the dilute form of the
thermodynamic definition `pi = -(R T / V_w) ln a_w` with
`ln a_w = -phi m M_w`. It is the form WaterTAP's seawater property package
codes, citing Nayar et al. 2016 [S2] Eq. 48 (that paper was not read here,
so the attribution is second-hand, **unverified**). Using the same form
keeps the two codes comparable.

**Limits.** Eq. 49 gives `phi = 0.8945` at `S = 0`, not 1, so the
osmotic pressure of a permeate (S ~ 1e-4) is an extrapolation. It is small
next to the feed's, and WaterTAP does the same. `seawater.validity(T, S)`
reports whether a concrete state is inside each correlation's range.

---

(water-chemistry)=
## Water chemistry: Pitzer activities, speciation and scaling

`difflow_water.chemistry.pitzer` and `difflow_water.chemistry.speciation`
(also `dw.speciate`, `dw.reference_seawater`, `dw.seawater_feed`) compute
the speciation of seawater and its brines and the saturation index
`SI = log10(IAP/K)` of the minerals that scale RO membranes. Everything is
in molality (mol/kg water) and differentiable: `jax.grad` of an SI with
respect to salinity, temperature or any element total is exact
(implicit-function gradient of the Newton solve).

**Aqueous model.** Species H+, Na+, K+, Mg2+, Ca2+, MgOH+, OH-, Cl-,
SO4^2-, HSO4-, HCO3-, CO3^2-, CO2(aq), MgCO3(aq): the
Na-K-Mg-Ca-H-Cl-SO4-OH-HCO3-CO3-CO2-H2O system of Harvie, Møller & Weare
[C7] as PHREEQC's `pitzer.dat` carries it (that file no longer includes
the CaCO3(aq) ion pair, so neither does this model).

| Piece | Equation | Source | Status |
|---|---|---|---|
| osmotic and activity coefficients | `phi`, `ln gamma_M`, `ln gamma_X`: B, B', B^phi, C, Z, F, Phi, Psi terms | [C6] Eqs. 1-3, 6-17 (Harvie & Weare form [C10]) | **Content verified** (read on the scanned report pages 2-5) |
| unsymmetrical mixing | `E-theta`, `E-theta'` from J0, J1, Chebyshev approximation (coefficients `AK`) | [C6] Eqs. 18-20 and the Fortran listing (`ETHETA`, `JAY`, `BDK`) | **Content verified** (coefficients read in the listing; Pitzer 1975 [C9] behind it) |
| neutral species | `ln gamma_N = sum_c 2 m_c lambda_Nc + sum_a 2 m_a lambda_Na` (+ PHREEQC's zeta term) | [C6] Eq. 21 | Content verified; the neutrals' effect on the ions and on `phi`, and lambda(CO2-CO2), are omitted (micromolal species, < 1e-5 in ln gamma) |
| Debye-Hückel slope | `A_phi = (1/3) sqrt(2 pi N_A rho_w) (e^2 / (4 pi eps0 D k T))^(3/2)` | [C6] Eq. 4 with CODATA constants | Content verified; see the note below |
| dielectric constant of water | Bradley & Pitzer, 1 atm | [C8] via the PHRQPITZ function `DC` | Coefficients content verified in the listing |
| parameter T dependence | `P = A0 + A1(1/T - 1/Tr) + A2 ln(T/Tr) + A3(T - Tr) + A4(T^2 - Tr^2) + A5(1/T^2 - 1/Tr^2)` | PHREEQC manual [C4], PITZER keyword | Content verified |
| log K(T) | analytic `A1 + A2 T + A3/T + A4 log10 T + A5/T^2 + A6 T^2`, else van't Hoff | [C4], SOLUTION_SPECIES / PHASES | Content verified (the form) |
| all parameter and log K values | 146 Pitzer parameters, 6 species, 8 minerals | PHREEQC `pitzer.dat` (phreeqpython 1.6.2), generated into `chemistry/data/pitzer_seawater.py` by `benchmarks/phreeqc/gen_pitzer_data.py` | **Second-hand** (pitzer.dat cites HMW 1984 and later work); cross-checked against PHREEQC below |

**A_phi note.** [C6] states that its Eq. 5, `A_phi = 1400684 (rho_w/(D T))^1.5`,
gives 0.39148 at 25 C. With D = 78.38 and rho_w = 0.99705 g/cm^3 the printed
constant gives 0.3904; the defining Eq. 4 with CODATA constants gives
0.39147, the value [C6] states, and PHREEQC's results match it (an NaCl
mean activity coefficient at 1 mol/kg moved from 0.2 % off PHREEQC to
7e-5 when Eq. 4 replaced Eq. 5). Eq. 4 is used.

**Speciation.** Unknowns are ln m of H+, Mg2+, Ca2+, SO4^2- and CO3^2-
(Na+, K+ and Cl- form no complexes in this model). Equations: the Mg, Ca,
SO4 and total-carbonate balances and either the alkalinity balance
(PHREEQC's definition: HCO3- 1, CO3^2- 2, OH- 1, MgOH+ 1, MgCO3 2, H+ -1,
HSO4- -1) or a fixed pH (`pH=`). Activity coefficients are iterated three
times inside the residual (only the minor species depend on them); Newton
runs a fixed 40 damped steps and ends with a value-neutral step for the
gradients.

**pH scale.** `pH` is `-log10 a_H` (unscaled Pitzer); `pH_macinnes` is on
the MacInnes scale (`ln gamma_i' = ln gamma_i + z_i (ln gamma_Cl - ln gamma_pm,KCl(I))`,
[C6]), which is what PHREEQC prints and reads. Saturation indices of
neutral salts do not depend on the scale.

**Seawater.** `reference_seawater(S)` gives the totals of
Reference-Composition seawater (Millero et al. 2008 [S3], TEOS-10 Table
D.3 [S4], content verified): Br- is counted as Cl-; Sr2+, F- and boron are
not in the aqueous model and are dropped (net charge 1e-5 of the total).
`seawater_feed(S, pH, T)` sets the alkalinity from a measured pH (default
8.1, ILLUSTRATIVE); the alkalinity is then conserved as the water is
concentrated (`concentrate(totals, factor)`).

**Minerals** (`MINERALS`): calcite, aragonite, gypsum, anhydrite, brucite,
magnesite, dolomite, halite. Saturation indices are at 1 atm (no pressure
correction of K) and say where precipitation becomes possible, not that it
happens (no kinetics, no antiscalant).

(water-phreeqc-comparison)=
### Comparison with PHREEQC

`benchmarks/phreeqc/seawater_brines.py` (run where `phreeqpython` is
installed) evaporates Reference-Composition seawater at pH 8.1 by
concentration factors 1-3 at 15, 25 and 35 C in PHREEQC 3 with
`pitzer.dat`, and `compare_seawater_brines.py` speciates the same totals
in difflow_water. Over the 15 cases the largest differences are:

| Quantity | max abs. difference |
|---|---|
| pH (MacInnes) | 3.3e-5 |
| ionic strength (relative) | 1.8e-7 |
| log10 water activity | 6.8e-7 |
| SI calcite, aragonite, magnesite | 0.0002 |
| SI gypsum, anhydrite, dolomite | 0.0004 |
| SI brucite, halite | 0.0001 |

With a fixed pH of 8.1, the computed alkalinity of the 25 C feed agrees
with PHREEQC's to 3e-6 (relative). Single-salt checks (1 mol/kg NaCl at 25
and 40 C, CaCl2, 0.5 mol/kg MgSO4): mean activity coefficients within
4e-4. The residual differences are consistent with the pure-water density
(Sharqawy Eq. 8 here; PHREEQC uses its own) entering `A_phi`.

---

(op-rostage)=
### ROStage

A seawater RO stage in plug flow, integrated along its membrane area.

| Inlet | Outlets |
|---|---|
| `feed`: seawater at the feed pressure | `concentrate` (retentate), `permeate` |

**Local fluxes** (solution-diffusion [M1], in WaterTAP's form so that the
same `A` and `B` mean the same thing in both codes):

```
J_w = A rho_w (P_f - P_p - (pi(S_m) - pi(S_p)))     water, kg/(m^2 s)
J_s = B (C_m - C_p)                                 salt,  kg/(m^2 s)
S_p = J_s / (J_s + J_w)                             local permeate
(C_m - C_p) / (C_b - C_p) = exp(J_v / k),  J_v = J_w / rho_w   (film theory)
```

with `C = S rho(T, S)` the salt mass concentration and `m` (wall), `b`
(bulk), `p` (permeate). For a given `J_w` the salt flux is the positive
root of a quadratic, so the local problem is one equation in `J_w`,
increasing on `[0, A rho_w (P_f - P_p)]`. It is solved by bisection plus a
value-neutral Newton step, which gives implicit-function gradients. The
permeate volume flux uses the pure-water density (`S_p` ~ 1e-4).

**Feed channel** (when `k=None` and/or `pressure_drop="calculated"`), as
coded in WaterTAP 1.8.0 (`membrane_channel_base.py`):

| Quantity | Equation | Source | Status |
|---|---|---|---|
| hydraulic diameter | `d_h = 4 eps / (2/h + (1 - eps) 8/h)` | [M4] Eq. 17 | **unverified** (via WaterTAP) |
| Reynolds number | `Re = m_dot d_h / (A_x mu)`, `A_x = h W eps` | definition | |
| Sherwood number | `Sh = 0.46 (Re Sc)^0.36`, `k = Sh D / d_h` | none identified | **unverified**; WaterTAP names no source |
| diffusivity | `D = 1.51e-9 - 2.00e-9 w + 3.01e-8 w^2 - 1.22e-7 w^3 + 1.53e-7 w^4` (NaCl, 25 C) | [M7] Eq. 6 | **unverified** (via WaterTAP) |
| friction, `module_type="flat_sheet"` (default) | `f = 0.42 + 189.3 / Re` | [M8] SI Eq. S27 | **unverified** (via WaterTAP) |
| friction, `module_type="spiral_wound"` | `f = 6.23 Re^-0.3` | [M4] Eq. 24 | **unverified** (via WaterTAP) |
| pressure gradient | `dP/dx = -f rho u^2 / (2 d_h)` | Darcy-Weisbach | |

**Parameters** (`ROStageParams`): `A_w` (m/(s Pa)), `B` (m/s), `area`
(m^2), `width` (m; length = area/width), `permeate_pressure` (Pa), `k`
(m/s, or None to compute it), `pressure_drop` (`"none"`, `"calculated"` or
a fixed total in Pa), `module_type`, `channel_height` (m),
`spacer_porosity`, `n_steps` (RK4 steps, default 40).

**Info**: `recovery_mass_water` (WaterTAP's water recovery),
`recovery_vol`, `salt_rejection`, `salinity_feed`/`_concentrate`/
`_permeate`, `deltaP`, `flux_mass_water_avg`, `max_cp_modulus`,
`min_driving_pressure` (outlet `P - P_p - pi_m`; near zero the stage is at
its osmotic limit and needs more `n_steps`), `permeate_flow_vol`,
`Re_in`/`Re_out`, and a `profile` along the area.

```python
ro = dw.ROStage(dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=50.0))
conc, perm, info = ro(dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035, P=65e5))
info["recovery_mass_water"], info["salinity_permeate"]
```

**Assumptions.** Isothermal; constant `A` and `B`; sea salt as one solute;
permeate fully mixed at `permeate_pressure`.

**Scaling** (`scaling=True`): the stage also speciates the concentrate and
the membrane-wall solution at the outlet ([water chemistry](water-chemistry))
and reports `SI_concentrate`, `SI_wall_outlet` (dict per mineral) and
`pH_concentrate`, `pH_wall_outlet`. The feed totals are `feed_totals`, or
Reference-Composition seawater at the feed salinity and `feed_pH` (default
8.1, ILLUSTRATIVE). Every solute is taken as rejected alike (totals scale
with the sea-salt molality) and CO2 stays in the concentrate.

The core `difflow.ReverseOsmosis` (in `difflow.units.membranes`) is the
single-solute, van't Hoff version of the same unit; `ROStage` adds the
seawater properties and the channel correlations.

---

(op-rotrain)=
### ROTrain

`ROTrain(ROTrainParams(stages=[...], boost=[...], booster_efficiency=0.8))`
(default: two 40 m^2 stages with WaterTAP's example membrane, ILLUSTRATIVE,
so the editor can place one) runs `ROStage`s in series on the concentrate side, with an optional booster
pump before each stage after the first (`boost`: pressure added, Pa; the
pump is a `WaterPump`), and mixes the permeates. Info: `stages` (each
stage's info), train `recovery_mass_water`, `salinity_permeate`,
`work_boosters` (W), `permeate_flow_vol`. No physics beyond the stage and
the pump.

---

(op-pressureexchanger)=
### PressureExchanger

An isobaric energy-recovery device [M5]: the pressure of the RO
concentrate is handed to an equal volume of incoming seawater. Equations as
in WaterTAP 1.8.0's `PressureExchanger` (efficiency form, optional leakage
and mixing):

```
P_feed,out - P_feed,in = eta (P_brine,in - P_brine,out)
P_brine,out = P_feed,in
V_feed,out  = (1 - L) V_brine,in                   (L: leakage fraction)
C_feed,out  = (1 - x) C_feed,in + x C_brine,in     (salt moves brine -> feed)
```

| Inlets | Outlets |
|---|---|
| `brine` (high-pressure concentrate), `feed` (low-pressure seawater) | `hp_feed`, `brine`, `bypass` |

In a sequential flowsheet the equal-flow condition is met by drawing the
brine's volume from the seawater fed to the exchanger; the rest leaves as
`bypass` (to the high-pressure pump). Feeding it the whole intake therefore
reproduces WaterTAP's splitter without a split fraction to iterate. The
feed must carry at least the brine's volume (`info["bypass_fraction"] >= 0`).

Defaults `efficiency=0.95`, `leakage=0`, `mixing=0` are the values WaterTAP's
`RO_with_energy_recovery` example fixes: design inputs, **ILLUSTRATIVE**,
not sourced performance data.

---

(op-waterpump)=
### WaterPump

Isothermal, incompressible pump: `W_fluid = V dP`, `W_mech = W_fluid / eta`,
with `V` the inlet volumetric flow from the seawater density. This is the
pump of WaterTAP's RO flowsheets (IDAES isothermal `Pump`), so pump energies
compare one for one. Give `outlet_pressure` (takes precedence) or `deltaP` (default 0, a placeholder), and `efficiency`
(default 0.8). Info: `deltaP`, `flow_vol`, `work_fluid`, `work_mechanical`.

---

(water-costing)=
## Levelized cost of water

`difflow_water.economics` reproduces the structure and default parameters
of WaterTAP 1.8.0's default costing package (`WaterTAPCosting`) and its RO,
pump and pressure-exchanger costing methods (read from WaterTAP's source):

```
capital_i      = TIC * unit_cost_i * size_i
total_capital  = total_investment_factor * sum_i capital_i
opex           = membrane_replacement * membrane_cost * area
                 + maintenance_labor_chemical_factor * sum_i capital_i
                 + electricity_cost * P_kW * 8766 h * utilization
CRF            = wacc / (1 - (1 + wacc)^-lifetime)
LCOW           = (total_capital * CRF + opex) / (Q_product * 1 yr * utilization)
SEC            = P_kW / Q_product[m^3/h]
```

| Parameter (`CostingParams`) | Default | Basis of size |
|---|---|---|
| `utilization_factor` | 0.9 | |
| `electricity_cost` | 0.07 USD/kWh | pump `work_mechanical` |
| `plant_lifetime`, `wacc` | 30 yr, 0.0930734 (CRF = 0.1) | |
| `TIC` | 2.0 | |
| `total_investment_factor` | 1.0 | |
| `maintenance_labor_chemical_factor` | 0.03 /yr | |
| `membrane_cost`, `membrane_replacement` | 30 USD/m^2, 0.2 /yr | membrane area |
| `hp_pump_cost` | 1.908 USD/W | mechanical work |
| `lp_pump_cost` | 889 USD/(L/s) | inlet flow |
| `px_cost` | 535 USD/(m^3/h) | low-pressure flow |

All USD 2018, a year of 365.25 days. WaterTAP's code cites no source for
these numbers; they are the benchmark's convention (**second-hand,
ILLUSTRATIVE** for any real plant). Change them through `CostingParams`.

---

(water-swro)=
## Seawater RO with a pressure exchanger

The phase-1 plant is WaterTAP's `RO_with_energy_recovery` layout:

```
seawater -> PX (low-pressure side) -bypass-> P1 (HP pump) --+
             |  ^                                           M1 -> RO -> permeate
             |  +---------- RO concentrate <----------------+ |
             +--hp_feed--> P2 (booster) --------------------->+
             +--brine--> disposal
```

- `dw.swro_performance(feed, ro, ro_pressure, px=..., costing=...)` is the
  plant as one differentiable function; it returns `LCOW`, `SEC`,
  `recovery`, permeate and brine salinities, pump works, the costing
  breakdown and the unit infos. Without exchanger mixing the RO feed is the
  intake at the RO pressure and one pass is exact; with mixing the loop is
  solved by substitution (`n_passes`).
- `dw.build_swro_flowsheet(ro, ro_pressure, ...)` is the same plant as a
  difflow `Flowsheet` (feed `"seawater"`, recycle `ro_conc_out -> ro_conc`)
  for the session tools and the editor.

```python
import jax
feed = dw.make_seawater_stream(flow_vol=1e-3, salinity=0.035)
ro = dw.ROStageParams(A_w=4.2e-12, B=3.5e-8, area=50.0)
r = dw.swro_performance(feed, ro, ro_pressure=65e5)
dLCOW_dP = jax.grad(lambda P: dw.swro_performance(feed, ro, P)["LCOW"])(65e5)
```

The example notebook `examples/70_water_swro_design.ipynb` sweeps the
membrane-wall saturation indices against recovery, minimizes the LCOW over
the RO pressure and membrane area with a calcite scaling limit (an
ILLUSTRATIVE stand-in for an antiscalant's tolerance), and runs a two-stage
train.

---

(water-editor)=
## In the editor

`ROStage`, `ROTrain`, `PressureExchanger` and `WaterPump` are palette
operations (category `water`) in the difflow editor (`difflow gui`, from the
difflow book), each with its own icon. **Examples > Seawater RO with a
pressure exchanger** opens the plant above (`10_water_swro`, written by
`scripts/make_gui_examples.py`), with membrane-wall saturation indices on.

- Dropped from the palette, `ROStage` gets placeholders (1.0) for `A_w`,
  `B` and `area`, flagged in the inspector; `ROTrain` starts as two 40 m^2
  stages with WaterTAP's example membrane (ILLUSTRATIVE); `WaterPump` starts
  with `deltaP = 0` until `outlet_pressure` or `deltaP` is set.
- The first solve in an editor session compiles the units (about 30 s with
  scaling on); later solves take about 1.5 s. The example turns the
  tear-error probe off (`error_probe: 0`): without exchanger mixing the
  loop gain is zero, and the probe would add a batched recompile of every
  unit (about 40 s) to the first solve. Turn it back on (Solve panel) when
  the exchanger mixes.

(water-watertap-comparison)=
## Comparison with WaterTAP

`benchmarks/watertap/swro_px.py` builds the plant above in WaterTAP 1.8.0
(IDAES 2.13.0, Pyomo 6.10.1, IPOPT from `idaes get-extensions`) with the
seawater property package and the 1-D RO model (calculated mass transfer
and pressure drop, 160 finite elements), at a fixed design (pump outlet
pressure and area fixed), and writes `swro_px_reference.json`.
`benchmarks/watertap/compare_swro_px.py` runs difflow_water on the same
inputs (RK4, 80 steps) and writes `swro_px_comparison.md`. Neither is a test:
WaterTAP is not a difflow dependency, and the test suite does not compare
against reference data. Run them by hand when a unit changes.

Four cases: `base` (35 g/kg, 65 bar, 50 m^2), `high_pressure` (35 g/kg,
75 bar, 80 m^2), `brackish_like` (15 g/kg, 30 bar, 60 m^2, 30 C) and
`px_leak_mix` (the base case with 1 % exchanger leakage and 3.5 % mixing,
which makes the concentrate loop a real recycle). Largest relative
difference over the four, per quantity:

| Quantity | max abs. rel. diff. | Quantity | max abs. rel. diff. |
|---|---|---|---|
| RO water recovery | 0.22 % | LCOW | 0.16 % |
| permeate salinity | 0.20 % | SEC | 0.10 % |
| concentrate salinity | 0.12 % | P1 work | 0.22 % |
| feed-side pressure drop | 0.28 % | P2 (booster) work | 0.24 % |
| average water flux | 0.22 % | PX outlet pressure | 0.02 % |
| outlet Reynolds number | 0.14 % | total capital / operating | 0.06 % / 0.07 % |
| PX low-pressure flow (leak/mix case) | 0.00 % | RO feed salinity (leak/mix case) | 0.00 % |

The base case in full:

| Quantity | WaterTAP | difflow | rel. diff |
|---|---|---|---|
| RO `recovery_mass_water` | 0.42383 | 0.423827 | -0.00 % |
| RO `salinity_permeate` | 2.42628e-4 | 2.42319e-4 | -0.13 % |
| RO `deltaP` (Pa) | -212208 | -212689 | -0.23 % |
| `LCOW` (USD/m^3) | 0.410379 | 0.410409 | +0.01 % |
| `SEC` (kWh/m^3) | 2.4696 | 2.46983 | +0.01 % |
| `work_P1` (W) | 3355.26 | 3355.23 | -0.00 % |
| `work_P2` (W) | 378.44 | 378.773 | +0.09 % |

What the comparison taught:

- **Friction factor.** WaterTAP's default RO module type is "flat sheet",
  `f = 0.42 + 189.3/Re`, not the spiral-wound `6.23 Re^-0.3`. With the
  latter the pressure drop was 2-14 % off; `module_type` now selects
  either, defaulting to WaterTAP's.
- **WaterTAP's mesh.** WaterTAP's RO1D is first-order (backward
  differences): its base-case recovery is 0.4216, 0.4231 and 0.4238 at 40,
  80 and 160 elements, approaching difflow's 0.4238 (RK4, converged by 40
  steps). The reference is generated at 160 elements for that reason.
- The remaining 0.1-0.3 % has not been traced. Candidates: difflow takes
  the permeate volume flux at pure-water density, and WaterTAP's
  pressure drop sums element values with a rectangle rule.

---

(water-gaps)=
## Known gaps

- The RO stage transports sea salt as one lumped solute; ions are resolved
  only for the scaling calculation, which assumes equal rejection of every
  solute and keeps CO2 in the concentrate.
- Saturation indices are at 1 atm, without kinetics or antiscalant
  effects. Sr2+ (celestite), barium, silica and boron are not modelled.
- Pitzer parameters and log K values are PHREEQC's (second-hand), not
  re-derived from the primary papers.
- The channel correlations (Sherwood, friction, diffusivity, hydraulic
  diameter) are WaterTAP's and are **unverified** against their papers; the
  Sherwood correlation has no identified source.
- Costing defaults are WaterTAP's convention, not sourced cost data.
- No nanofiltration yet.

---

(water-references)=
## References

Status terms: **Citation verified (Crossref)**: authors, title, journal,
year, volume, pages and DOI matched against the Crossref record.
**Content verified**: the coded equation/coefficients read against the
source. **(unverified)**: second-hand or recalled. **ILLUSTRATIVE**: no
source; a design input or placeholder.

| ID | Reference | Used for | Status |
|---|---|---|---|
| S1 | Sharqawy, M.H.; Lienhard, J.H.; Zubair, S.M. "Thermophysical properties of seawater: a review of existing correlations and data." *Desalin. Water Treat.* **2010**, 16(1-3), 354--380. doi:10.5004/dwt.2010.1079 | Density (Eq. 8), viscosity (Eqs. 22, 23), osmotic coefficient (Eq. 49) | Citation verified (Crossref). **Content verified**: coefficients read against the authors' corrected version (web.mit.edu/lienhard), including the five correction notes (none affect these equations). |
| S2 | Nayar, K.G.; Sharqawy, M.H.; Banchik, L.D.; Lienhard V, J.H. "Thermophysical properties of seawater: A review and new correlations that include pressure dependence." *Desalination* **2016**, 390, 1--24. doi:10.1016/j.desal.2016.02.024 | Osmotic pressure form `phi m rho_w R T` (Eq. 48, per WaterTAP) | Citation verified (Crossref). Content **unverified** (paper not reached; attribution from WaterTAP's code). |
| S3 | Millero, F.J.; Feistel, R.; Wright, D.G.; McDougall, T.J. "The composition of Standard Seawater and the definition of the Reference-Composition Salinity Scale." *Deep-Sea Res. I* **2008**, 55(1), 50--72. doi:10.1016/j.dsr.2007.10.001 | Origin of `M_S` | Citation verified (Crossref). Value read via S4. |
| S4 | IOC, SCOR and IAPSO. *The International Thermodynamic Equation of Seawater -- 2010: Calculation and Use of Thermodynamic Properties*. Intergovernmental Oceanographic Commission, Manuals and Guides No. 56, UNESCO, **2010** (TEOS-10 manual, teos-10.org) | `M_S = 31.4038218 g/mol` | **Content verified**: value read in the manual ("M_S = 31.403 821 8... g mol^-1, from Millero et al. (2008a)"). No DOI. |
| M1 | Wijmans, J.G.; Baker, R.W. "The solution-diffusion model: a review." *J. Membr. Sci.* **1995**, 107(1-2), 1--21. doi:10.1016/0376-7388(95)00102-I | Water and salt flux laws | Citation verified (Crossref). The flux forms are the standard ones; not re-read against the paper (unverified). |
| M4 | Schock, G.; Miquel, A. "Mass transfer and pressure loss in spiral wound modules." *Desalination* **1987**, 64, 339--352. doi:10.1016/0011-9164(87)90107-X | Spacer hydraulic diameter (Eq. 17), spiral-wound friction (Eq. 24) | Citation verified (Crossref). Equations and numbers as coded in WaterTAP 1.8.0 (unverified). |
| M5 | Stover, R.L. "Seawater reverse osmosis with isobaric energy recovery devices." *Desalination* **2007**, 203(1-3), 168--175. doi:10.1016/j.desal.2006.03.528 | Pressure exchanger | Citation verified (Crossref). Equations are WaterTAP's; no constants taken from the paper. |
| M7 | Bartholomew, T.V.; Mauter, M.S. "Computational framework for modeling membrane processes without process and solution property simplifications." *J. Membr. Sci.* **2019**, 573, 682--693. doi:10.1016/j.memsci.2018.11.067 | Salt diffusivity (Eq. 6) | Citation verified (Crossref; the DOI first recalled for it was wrong and was corrected). Coefficients as coded in WaterTAP 1.8.0 (unverified). |
| M8 | Bartholomew, T.V.; Siefert, N.S.; Mauter, M.S. "Cost optimization of osmotically assisted reverse osmosis." *Environ. Sci. Technol.* **2018**, 52(20), 11813--11821. doi:10.1021/acs.est.8b02771 | Flat-sheet friction factor (SI Eq. S27, per WaterTAP) | Title, authors, journal, year and DOI verified (Crossref; Crossref lists no volume/pages, so those are unverified). Correlation as coded in WaterTAP (unverified). |
| C4 | Parkhurst, D.L.; Appelo, C.A.J. *Description of Input and Examples for PHREEQC Version 3*; U.S. Geological Survey Techniques and Methods 6-A43, **2013**. doi:10.3133/tm6A43 | Pitzer parameter T form; log K forms; MacInnes and `pitzer.dat` conventions; PHREEQC itself is the chemistry cross-check | Citation verified (Crossref). The forms used were read in the manual (PITZER keyword, `-analytical_expression`): content verified. |
| C6 | Plummer, L.N.; Parkhurst, D.L.; Fleming, G.W.; Dunkle, S.A. *A Computer Program Incorporating Pitzer's Equations for Calculation of Geochemical Reactions in Brines*; U.S. Geological Survey Water-Resources Investigations Report 88-4153, **1988**. doi:10.3133/wri884153 | Pitzer equations (Eqs. 1-21), E-theta algorithm and coefficients, dielectric constant routine, MacInnes scale | Title, year and DOI verified (Crossref; Crossref lists no authors, which were read on the report's title page). **Content verified**: equations read on the scanned pages 2-5, code read in the Fortran listing (USGS public report). Its Eq. 5 constant is inconsistent with its Eq. 4 and its stated value; Eq. 4 is used. |
| C7 | Harvie, C.E.; Møller, N.; Weare, J.H. "The prediction of mineral solubilities in natural waters: The Na-K-Mg-Ca-H-Cl-SO4-OH-HCO3-CO3-CO2-H2O system to high ionic strengths at 25 °C." *Geochim. Cosmochim. Acta* **1984**, 48(4), 723--751. doi:10.1016/0016-7037(84)90098-X | The aqueous model and its 25 C parameters (via pitzer.dat) | Citation verified (Crossref). Not read; parameters taken from PHREEQC's pitzer.dat (second-hand). |
| C8 | Bradley, D.J.; Pitzer, K.S. "Thermodynamics of electrolytes. 12. Dielectric properties of water and Debye-Hückel parameters to 350 °C and 1 kbar." *J. Phys. Chem.* **1979**, 83(12), 1599--1603. doi:10.1021/j100475a009 | Dielectric constant of water | Citation verified (Crossref). Coefficients read in the PHRQPITZ listing [C6], not in the paper. |
| C9 | Pitzer, K.S. "Thermodynamics of electrolytes. V. Effects of higher-order electrostatic terms." *J. Solution Chem.* **1975**, 4, 249--265. doi:10.1007/BF00646562 | Origin of E-theta | Citation **unverified** (Crossref lookups rate-limited; as recalled and as cited in [C6]). The code follows [C6]'s listing, not this paper. |
| C10 | Harvie, C.E.; Weare, J.H. "The prediction of mineral solubilities in natural waters: the Na-K-Mg-Ca-Cl-SO4-H2O system from zero to high concentration at 25 °C." *Geochim. Cosmochim. Acta* **1980**, 44(7), 981--997. doi:10.1016/0016-7037(80)90287-2 | Notation of the Pitzer equations | Citation verified (Crossref). Equations read in [C6], which follows it. |
| PZ | PHREEQC `pitzer.dat`, as shipped with phreeqpython 1.6.2 (sha256 eb5051704fad461c...) | Every Pitzer parameter and log K value | Read directly and transcribed by script; second-hand to the papers it cites. |
| B1 | Akkor, I.; Amusat, O.O.; Vecchiarelli, P.; Gounaris, C.E.; Knueven, B.; Dudchenko, A.V. "Optimization of desalination systems with detailed water chemistry through integration of Reaktoro in WaterTAP." *ACS ES&T Eng.* **2026**, 6(5), 1464--1477. doi:10.1021/acsestengg.5c01008 | WaterTAP (benchmark context) | Citation verified (Crossref) |
| B2 | Dudchenko, A.; Atia, A.; Bartholomew, T. "WaterTAP: Tool for analysis of water treatment systems." NAWI Alliance Fall Meeting, **2023** (OSTI). doi:10.2172/2447624 | WaterTAP (benchmark) | Citation verified (Crossref). The comparison used WaterTAP 1.8.0 from PyPI. |
| B3 | Lee, A.; Ghouse, J.H.; Eslick, J.C.; Laird, C.D.; Siirola, J.D.; et al. "The IDAES process modeling framework and model library -- flexibility for process simulation and optimization." *J. Adv. Manuf. Process.* **2021**, 3(3), e10095. doi:10.1002/amp2.10095 | IDAES (pump model, WaterTAP's base) | Citation verified (Crossref) |
| WT | WaterTAP 1.8.0 source (`seawater_prop_pack.py`, `membrane_channel_base.py`, `reverse_osmosis_1D.py`, `pressure_exchanger.py`, `costing/`), read from the PyPI wheel | Equations as coded; costing structure and defaults | Read directly (the code, not the papers behind it). |
