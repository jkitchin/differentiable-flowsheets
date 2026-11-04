# Water Treatment and Desalination Unit Operations

![difflow_water](../images/plugins/difflow-water-logo.svg)

This page documents the `difflow_water` plugin: seawater properties, a
Pitzer activity model with seawater speciation and mineral saturation
indices (scaling), a seawater reverse-osmosis (RO) stage and multi-stage
train, a pressure exchanger, water pumps, a levelized-cost-of-water (LCOW)
model, the seawater RO + pressure exchanger plant built from them,
distribution networks, thermal desalination (MVC, MED), and conventional
treatment (sedimentation, filtration, disinfection, GAC), and activated
sludge wastewater treatment (ASM1, Takács settler, BSM1), and anaerobic
digestion (ADM1). A one-page map is in
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

(water-seawater-teos10)=
**Check against TEOS-10.** `benchmarks/teos10/seawater_teos10.py`
evaluates TEOS-10 [S4], the seawater standard whose Gibbs function is
IAPWS-2008, with its reference code (GSW-Python 3.6.23 [GSW]). The grid is
5-40 C and 0-70 g/kg at 1 atm, and notebook 77 compares the results.
Largest deviations, against the accuracy Sharqawy et al. [S1] state for
each correlation:

| Property | max deviation from TEOS-10 | stated accuracy [S1] |
|---|---|---|
| density (Eq. 8) | 0.07 % | 0.1 % |
| specific heat (Eq. 9) | 0.15 % | 0.28 % |
| osmotic pressure (Eq. 49 coefficient) | 0.62 % | 1.4 % |
| enthalpy rise from 15 C (Eq. 43) | 0.55 % | 0.5 % |
| latent heat of water (Eq. 54) | 0.15 % | 0.01 % (vs IAPWS-95) |

Eq. 43 is valid from 10 C. Below that its salinity term drifts: at 5 C and
35 g/kg it gives -882 J/kg for the effect of salt, against TEOS-10's -1178.
The latent-heat comparison is not like for like, since TEOS-10's value is
for evaporation from seawater and Eq. 54 is for pure water at saturation.

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

(water-network)=
## Water distribution networks

`difflow_water.network` models a distribution network (junctions,
reservoirs, tanks, pipes, pumps, valves) and solves its steady state the
way EPANET does: junction heads and link flows together, by Newton
(EPANET's gradient method is the same Newton step in a reduced form). It is
a library, not palette operations: a network is one equation set, solved
equation-oriented, so there are no tears. Everything is in JAX: pressures
and flows have exact (implicit-function) gradients with respect to demands,
leak (emitter) coefficients, roughness, diameters, reservoir heads, pump
speeds and valve settings.

```python
from difflow_water.network import NetworkModel, WaterNetwork, load_example

net = WaterNetwork()
net.add_reservoir("R", head=60.0)
net.add_junction("J1", elevation=10.0, demand=0.01, emitter=1e-4)   # m, m^3/s
net.add_pipe("P1", "R", "J1", length=1000.0, diameter=0.2, roughness=120.0)
res = NetworkModel(net).solve()            # head, pressure, flow, demand, emitter_flow, ...

net3, epanet = load_example("Net3")        # EPANET example 3 + EPANET 2.2's own solution
grad = jax.grad(lambda c: NetworkModel(net3).solve({"emitter": c})["pressure"].sum())
```

**Equations** (EPANET 2.2 manual [N8], Section 13.1 and Table 3.1; read
in the manual, content verified; US-unit constants converted to SI in code):

| Piece | Equation | Notes |
|---|---|---|
| link | `H_start - H_end = h_loss(Q) - h_gain(Q) + t - u` | `t`: valve throttling, `u`: blocking head (both >= 0) |
| junction | `sum Q_in - sum Q_out - demand(p) - C p^gamma = 0` | emitter (leak) `q = C p^gamma` |
| Hazen-Williams | `h = 10.667 C^-1.852 d^-4.871 L Q|Q|^0.852` | 4.727 in ft/cfs |
| Darcy-Weisbach | `h = 0.0827 f L d^-5 Q|Q|`; f: 64/Re (Re < 2000), Swamee-Jain [N7] (Re > 4000), EPANET's cubic between | 0.0252 in ft/cfs; see the log10 note |
| minor loss | `0.0826 K d^-4 Q|Q|` | 0.02517 in ft units |
| pump | `h_gain = w^2 (A - B (Q/w)^C)`; one-point curve: shutoff 4/3 of design head, zero head at twice design flow | constant-power pump `P/(rho g Q)`, rho g = 9802 N/m^3 (unverified, see below) |
| open valve | smooth pipe, f = 0.02, length twice the diameter, plus minor loss; TCV: setting is K | |
| closed link | `h = R Q`, R = 1e8 (ft/cfs) | keeps heads defined behind closed links |
| pressure-driven demand | `d = D ((p - P0)/(Pf - P0))^e` clipped to [0, D] (Wagner et al. [N3]) | `demand_model="PDD"` |

**Valves and check valves: two formulations** (DESIGN.md Section 3a).
Every switch is a complementarity pair, `u >= 0 perp Q >= 0` for check
valves, pumps, PRVs and PSVs; `t >= 0 perp (H_set - H_end) >= 0` for a PRV,
`(H_start - H_set)` for a PSV, `(Q_set - Q)` for an FCV; each through the
smoothed Fischer-Burmeister function
`phi(a, b) = a + b - sqrt(a^2 + b^2 + 2 mu^2)` (scaled: 1 m of head,
1 L/s of flow).

- `valve_formulation="complementarity"` (default): `mu` is continued down
  to 1e-9, so valves switch sharply, as EPANET's status logic does;
  gradients are exact away from a switching point.
- `valve_formulation="smooth"`: `mu` stops at `smoothing` (default 1e-3),
  so the equations are differentiable everywhere and a switch is rounded
  over about `smoothing`; useful when an optimizer has to cross a switch.

Both share one residual (`network/hydraulics.py`) and gave the same
answers on every benchmark network.

**Smooth where EPANET switches.** `|Q|^(n-1)` is `(Q^2 + delta^2)^((n-1)/2)`
with delta = 1e-7 m^3/s (EPANET instead linearizes below a flow threshold);
`max(p, 0)` in emitters and pressure-driven demand is a smooth maximum
(1 mm). A pump whose outlet side holds no demand, emitter, reservoir or
tank once closed links are removed is compiled as closed (EPANET's status
check closes it; a constant-power pump would otherwise produce an unbounded
head); `solve()["auto_closed"]` lists them.

**Solver.** Newton on the scaled residual with a line search over step
fractions, per continuation stage until the residual is below 1e-10 (at
most `n_iter`); `converged` and a `NetworkConvergenceWarning` report a
stop short of it. A last value-neutral Newton step gives the implicit
gradients. Dense linear algebra: ky10 (920 junctions, 1061 links) solves in
about 6 s on one CPU.

**Reading EPANET files.** `read_inp(path)` returns the network (SI, at
its initial state) and a `Schedule` (patterns, demands with their
patterns, reservoir and pump-speed patterns, simple controls, times),
following the manual's input format and units (Appendix A; content
verified). US units are converted with exact unit definitions; psi to
metres of head with EPANET's 0.4333 psi/ft (as WNTR converts; EPANET's
source not read, unverified). On EPANET's Net1, Net3 and ky10 every numeric
field matches WNTR's reading to 1e-6. Not read (an error, not a silent
skip): rule-based controls, tank volume curves, multi-point pump curves,
GPV and PBV valves, Chezy-Manning. `example_inp("Net1" | "Net3")` gives the
shipped EPANET example files.

**Extended-period simulation.** `simulate(network, schedule)` steps as
EPANET does [N8]: demands, reservoir heads and pump speeds follow their
patterns, simple controls (link open/closed or setting, on a tank level, a
junction pressure, a time or a clock time) switch links, tank levels
advance with their net inflow (`dh/dt = Q_net/A`), and each step ends at
the earliest of the next hydraulic step, pattern period, timed control,
report time, or a tank reaching a control level or a limit. A tank at a
limit closes the pumps that would overfill or drain it.
*Gradients:* a run records a `plan` (statuses of every step, and what set
its length). `simulate(..., plan=plan, params=...)` replays it with traced
parameters; steps cut to land a tank on a level are recomputed in JAX, so
the derivative includes how the switch times move (the derivative of the
trajectory with the same switch sequence). On Net1 over 24 h the
derivative of the final tank level with respect to pipe roughness matches
central differences to 1e-9.
*Energy and pump scheduling:* `simulate` returns each step's pump power
`rho g Q h_gain / efficiency` (`step_power`, W; efficiencies from the
input file's [ENERGY] section, EPANET's default 75 %) with `step_time`,
`step_dt` and `pump_energy_kwh`, all JAX values, and takes a
`speed_profile` (pump-speed multipliers per period) that may be traced, so
an energy cost under any tariff can be minimized over a schedule with exact
derivatives. On Net1 the day's pump energy is 1333 kWh; rebuilt from
EPANET's hourly flows and heads with the same formula it is 1348 kWh (the
hourly rectangle rule counts the pump as on for the whole hour in which it
switches off). `examples/72_water_pump_scheduling.ipynb` replaces Net1's
level controls by a 12-block speed schedule under an ILLUSTRATIVE
time-of-use tariff and cuts the day's cost from 171 to 149 USD (a local
stop of SLSQP on a non-convex problem; the notebook says so).

**Water quality.** `water_quality(network, schedule, simulate(...))` runs
EPANET's Lagrangian transport (Section 13.2 of [N8], content verified):
parcels move through pipes without dispersion, blend completely at
junctions (Eq. 13.9) and in tanks (Eq. 13.10), nodes are visited upstream
to downstream each quality step, a new parcel of the node's quality enters
each outflow pipe (merged into the last one within `tolerance`), and a
reversing pipe reverses its parcel train. Modes: CHEMICAL (bulk reaction
order 0 or 1; first-order wall reaction with the mass-transfer
coefficient from the laminar Edwards et al. or turbulent Notter-Sleicher
Sherwood number, both as given in [N8]), AGE (1 h per hour) and TRACE (% of
water from a node). Closed links and flows below 1e-6 cfs carry no water
for quality. Taken as EPANET's, not stated in the manual (unverified):
chlorine diffusivity 1.3e-8 ft^2/s, laminar below Re = 2300, `|k_w|` in
the wall-rate denominator, the 1e-6 cfs zero-flow threshold. NumPy, **not
differentiable** (the parcel bookkeeping is discrete). Not supported (an
error): water-quality sources ([SOURCES]), tank mixing models other than
complete mixing ([MIXING]), bulk orders other than 0 and 1, roughness-
correlated wall coefficients, limiting potentials.

**Not yet:** rule-based controls, volume curves, Chezy-Manning,
differentiable water quality, energy cost from [ENERGY] price patterns and
demand charges (the notebook applies its own tariff).

**Two notes from the EPANET comparison.**

- *Swamee-Jain uses log10.* The EPANET 2.2 manual prints Swamee-Jain with
  `ln`; coded that way the friction factor came out (ln 10)^2 = 5.3 times
  small and heads 70 m off EPANET. With `log10` (Swamee & Jain's form; the
  manual's own transition constant 0.86859 = 2/ln 10 is the log10 form
  written with ln) the Darcy-Weisbach network agrees to 1.7 mm.
- *rho g for power pumps.* The manual does not state the unit weight EPANET
  uses for constant-power pumps. 9802 N/m^3 (62.4 lbf/ft^3) brought ky10's
  median head difference from 1.3e-3 to 4e-5 m, against 9810; it is used,
  marked unverified.

(water-epanet-comparison)=
### Comparison with EPANET 2.2

`benchmarks/wntr/export_networks.py` (run where WNTR is installed) reads
EPANET example networks with WNTR 1.5.0, solves them in EPANET 2.2 (WNTR's
`EpanetSimulator`, one period: t = 0 demands, levels, controls and link
statuses) and writes each network and EPANET's solution to JSON;
`compare_networks.py` solves the same networks in difflow_water. Hand-run,
not a test. Largest differences (both valve formulations gave the same):

| Network | Junctions / links | Head loss | max abs head diff. | max abs flow diff. | solve |
|---|---|---|---|---|---|
| Net1 | 9 / 13 (1 pump) | H-W | 2.1e-5 m | 1.2e-4 L/s | 1-2 s |
| Net1, D-W (roughness 0.26 mm, ILLUSTRATIVE) | 9 / 13 | D-W | 1.7e-3 m | 4.0e-3 L/s | 2-3 s |
| Net3 | 92 / 119 (2 pumps) | H-W | 2.7e-5 m | 0.020 L/s | 1-2 s |
| ky10 | 920 / 1061 (13 constant-power pumps, 5 PRVs) | H-W | 1.6e-3 m (median 4e-5 m); 0.075 m in the closed pocket behind pump 11 | 7.2e-3 L/s | 5-6 s |

Extended-period (`benchmarks/wntr/export_eps.py`, `compare_eps.py`: the
INP files read by difflow's own `read_inp`, the whole duration, every
report time):

| Network | Duration | Solves | max tank level diff. | max head diff. | median head diff. | link-status mismatches |
|---|---|---|---|---|---|---|
| Net1 | 24 h | 27 | 1.1e-4 m | 1.2e-4 m | 2.7e-5 m | 0 |
| Net3 | 168 h | 204 | 3.4e-2 m | 3.4e-2 m | 6.9e-4 m | 0 |

Water quality (`compare_quality.py`, same runs):

| Network | Quality | Report times | max abs diff. | median abs diff. | EPANET range |
|---|---|---|---|---|---|
| Net1 | chlorine (bulk + wall decay) | 25 | 0.0098 mg/L | 0.0007 mg/L | 0.15-1 mg/L |
| Net1 | water age | 25 | 0.16 h | 1e-6 h | 0-23.5 h |
| Net3 | trace of the Lake source | 169 | 22.9 % | 0.0008 % | 0-100 % |

The Net3 trace's large differences are at moving fronts, where a small
difference in arrival time is a large difference at one report time, and
cluster in the hours where Net3's hydraulics differ most (below): 1.7 % of
the 16 393 node-times differ by more than 1 percentage point, 0.1 % by more
than 10. Closed links once carried their modelled `h = R Q` trickle in the
transport, which made a dead-end node take its source's quality a step
early (100 % off at Net3's node 10); they now carry no water, as in EPANET.

The Net3 differences are episodic: up to 3.4 cm in a tank level in the
hours after control switches (hours 23-26 of the first day), decaying again
as the tanks rebalance; every pump and pipe status agrees with EPANET at
every report time. Not traced further.

EPANET writes heads in single precision (about 1e-5 m at 300 m), and its
convergence accuracy is 0.001 on flow, so differences of this size are
within EPANET's own resolution. ky10's pocket behind the auto-closed pump
11 and the closed RV-4 has no flow; there difflow ties the heads to both
closed links and EPANET to the pump's suction.

---

(water-wastewater)=
## Wastewater: activated sludge (ASM1, BSM1)

`difflow_water.wastewater` covers municipal activated sludge: the ASM1
kinetics, a steady-state ASM1 reactor, the Takács secondary settler, and
the Benchmark Simulation Model no. 1 (BSM1) plant. All of it is
transcribed from the BSM1 technical report [W5], which states ASM1 in full
(variables, the eight process rates, the conversion rates, the
parameters), the settler and the plant. The report was read, so the
content is verified. ASM1 itself is Henze et al. 1987, restated in [W3],
which was not read.

Stream convention: plain streams, with `F_H2O` (it carries the volume,
`Q = F_H2O M_w / 1000`) and one `F_<state>` per ASM1 state with
`1 mol = 1 g` of COD or N (`S_ALK`: a real mole), so that
`F_<state> = Q C`. `bsm1_influent()` is the BSM1 constant influent ([W5]
Table 5).

| Function / unit | What it is |
|---|---|
| `conversion_rates(Z, ASM1Params())` | ASM1's 13 conversion rates (g/m^3/d), BSM1 parameters at about 15 C |
| `ASM1Reactor` | one completely mixed ASM1 reactor at steady state; aeration by `KLa` or a DO setpoint |
| `TakacsSettler` | 10-layer non-reactive settler with the Takács double-exponential settling velocity [W1, W5]; given underflow |
| `ActivatedSludgePlant` / `activated_sludge_plant` | the BSM1 plant (2 anoxic + 3 aerobic reactors, internal and external recycle, Takács settler) as one equation set |
| `build_bsm1_flowsheet()` | the same plant as a difflow `Flowsheet` of the palette units, with two recycle tears |

(op-asm1reactor)=
### ASM1Reactor

`ASM1Reactor(ASM1ReactorParams(volume=..., KLa=...))(inlet) -> (outlet, info)`:
`0 = Q (Z_in - Z)/V + r(Z) + KLa (S_O* - S_O)` (for oxygen; `S_O* = 8
g/m^3`), or `S_O` held at `DO_setpoint` with the aeration it needs
reported. The kinetic parameters are fields of the params (BSM1 values by
default). `info` has the state, the conversion rates and the aeration.

Checked against the BSM1 report on its own (notebook 79). Fed the printed
Table 6 state of reactor 4 (or 1), the reactor gives reactor 5 (or 2)
within 0.4 % on every state: dissolved oxygen 0.489 against 0.491 g/m^3,
the rest within 0.2 %. That is the precision of the 3-4-digit printed
inlet.

(op-takacssettler)=
### TakacsSettler

`TakacsSettler(TakacsSettlerParams(Q_underflow=...))(feed) -> (effluent,
underflow, info)`: the BSM1 settler ([W5] 2.3.3) at steady state. The
settling velocity is `v_s = max(0, min(v0', v0 (e^{-r_h (X - X_min)} -
e^{-r_p (X - X_min)})))` with `X_min = f_ns X_f`. Fluxes between layers
are limited by `min(J_s,m, J_s,m+-1)`, and above the feed layer by the
threshold `X_t`. Solubles pass unchanged, and particulates keep the feed's
fractions. `info["solids"]` is the layer profile, bottom first. Fed the
printed Table 6 reactor 5 state, it reproduces the report's Table 7
layers within 0.13 % (notebook 79).

(op-activatedsludgeplant)=
### ActivatedSludgePlant

`ActivatedSludgePlant(ActivatedSludgePlantParams())(influent) -> (effluent,
wastage, info)`, or `activated_sludge_plant(influent,
ActivatedSludgeParams(...))` for any number of reactors, DO setpoints and
the split clarifier. The result has the reactor states, the settler
profile, the flows, the aeration per reactor, the sludge age (on biomass,
[W5] 2.3.3) and the effluent TSS.

**Solving.** A steady state is found the way BSM1 prescribes ([W5]
Section 3). The dynamics are integrated with constant influent from a
generic sludge state (diffrax Kvaerno5, adaptive): 200 days for the plant,
20 residence times for a reactor, 50 days for the settler. Newton
(optimistix) then polishes the result, so gradients come from the
implicit-function theorem at the steady state. Starting Newton directly
from the guess can find the *autotroph washout* steady state (no
nitrification), which also solves the equations; following the dynamics
avoids it. The settler's feed-layer balance has `(v_up + v_dn) X` leaving
the layer. This is the mass-conservative form (the sign in the extracted
report text is garbled), and the agreement with [W5] Table 7 confirms it.

**Performance indices.** `bsm1_indices(result, params)` gives the BSM1
report's effluent COD, BOD5, TSS, Kjeldahl and total nitrogen, the
effluent quality index EQ (Table 10 weights), and the aeration energy
`AE = S_O,sat / 1.8e3 sum V_i KLa_i` and pumping energy
`PE = 0.004 Q_a + 0.008 Q_r + 0.05 Q_w` (kWh/d). These are [W5] Section 6
at steady state; reactors on a DO setpoint are left out of AE. It also
gives the margins to the Table 9 limits.

**As a sequential flowsheet: a hard recycle.** `build_bsm1_flowsheet()`
builds the same plant from the palette units: a mixer, five
`ASM1Reactor`s, splitters and the `TakacsSettler`, with two tear streams.
From the default tear guesses none of `Flowsheet.solve`'s accelerations
reaches the steady state within 150 iterations:

| Acceleration | After 150 iterations |
|---|---|
| Wegstein | NaN |
| Anderson | the autotroph-washout state (X_BA about 1e-40) |
| plain substitution | still creeping (X_BA 0.075 against 150) |

The sludge recycle has a loop gain close to one, since the sludge age is
about 9 days and each pass is about 0.6 days, and the washout state is a
second fixed point. Started from the equation-oriented answer, the
sequential flowsheet reproduces it to 1e-15. Use `ActivatedSludgePlant` /
`activated_sludge_plant`, or pass its states as `tear_initial`. This
case is documented here and not added to the core convergence corpus,
which would then have to import a plugin.

### Comparison: the BSM1 report and WaterTAP

`benchmarks/watertap/compare_bsm1.py` checks the plant against two
references:

- **The BSM1 report** ([W5] Tables 6-7, open-loop steady state:
  `Q_a = 55338`, `KLa = 0, 0, 240, 240, 84 1/d`; transcribed in
  `benchmarks/bsm1/report_tables.json`). All 13 states of the 5 reactors,
  the reactor TSS and the 10 settler layers agree to **3.2e-3** relative,
  within the tables' 3-4 printed digits. The sludge age is 9.14 d
  (report: "about 9 days").
- **WaterTAP 1.8.0's BSM1 example** (`flowsheets/activated_sludge/
  ASM1_flowsheet.py`). WaterTAP's clarifier is a component-split separator
  rather than a settler model, so difflow runs it with `settler="split"`,
  WaterTAP's split fractions and its recycle fractions converted to flows,
  `KLa = 240 1/d` in reactors 3-4 and DO held at 0.449 g/m^3 in reactor 5.
  Every state of the 5 reactors, the effluent and the wastage agrees to
  **1.3e-7**.

---

(water-adm1)=
## Wastewater: anaerobic digestion (ADM1)

(op-anaerobicdigester)=
### AnaerobicDigester

`AnaerobicDigester(ADM1Params())(influent) -> (effluent, biogas, info)`,
or `anaerobic_digester(influent, params)`. This is ADM1 [W2] as
implemented for BSM2 by Rosen & Jeppsson [W6]. The report was read, so the
content is verified. Its features:

- 19 biochemical processes and 24 water-phase states;
- carbon and nitrogen balances closed through the inorganic carbon and
  nitrogen states ([W6] 3.1.1, including its revised fractionation and
  C and N contents);
- temperature-dependent `K_w`, `K_a,co2`, `K_a,IN`, Henry constants and
  water vapour pressure;
- the three gas-phase states.

The formulation is the report's DAE with an algebraic pH: acid-base pairs
are at equilibrium, and `S_H+` solves electroneutrality. The left side
increases monotonically in `S_H+`, so the code bisects on `ln S_H+` and
then takes a value-neutral Newton step for implicit gradients. Gas
transfer is `kLa (S - K_H p)`, and the gas flow comes from the head-space
overpressure.

Options, each a choice [W6] discusses:

| Option | Default | Alternative |
|---|---|---|
| `gas_flow` | `"compensated"`: `k_p (P_gas - P_atm) P_gas/P_atm`, as [W6] 3.2.4 states for BSM2 and WaterTAP codes it | `"uncompensated"`: `k_p (P_gas - P_atm)`, the form that reproduces [W6]'s own Section 7 results |
| `pH_inhibition` | `"hill"`: hydrogen-ion Hill functions, the BSM2 choice ([W6] Expression 3) | `"switch"`: Batstone et al.'s `exp(-3((pH - pH_UL)/(pH_UL - pH_LL))^2)` below `pH_UL`, as WaterTAP codes it |

**Start-up and the soured state.** ADM1 has a second steady state: low
pH, accumulated volatile fatty acids, and acetogens and methanogens washed
out. The solver integrates 300 days of dynamics before Newton. It starts
from an inoculum (each biomass group at `inoculum`, default 1 kg COD/m^3)
with low substrate. A start that copies the influent into the reactor
(20+ kg COD/m^3 of solids hydrolysing at once) sours, whatever the seed.
Check `pH` and `gas_fraction_ch4`.

(water-adm1-comparison)=
### Comparison: Rosen & Jeppsson and WaterTAP

`benchmarks/watertap/compare_adm1.py`:

- **Rosen & Jeppsson [W6] Section 7**, the steady-state verification case
  (170 m^3/d, 35 C; outputs printed to 13 digits, transcribed in
  `benchmarks/adm1/report_steady_state.json`). All 24 states, pH, ions and
  gas states agree to **1.7e-7**, with pH 7.465537770 against 7.465537770,
  but only with the *uncompensated* gas flow. With the compensated form
  that the report says BSM2 uses, its own gas balances are off by 5 % at
  its printed state (q_gas 2956 against 2801 m^3/d). Methane is 60.9 % of
  the gas (report: "about 61 %").
- **WaterTAP 1.8.0's ADM1 example** (`flowsheets/anaerobic_digester/
  ADM1_flowsheet.py`, the same influent). Run with WaterTAP's choices:
  compensated gas flow, `P_atm = 101325 Pa` and the Batstone pH switch.
  The largest difference is **2.5e-4**. Most of it is a uniform 1.7e-4
  offset in the conserved states (`S_I`, `X_I`, `S_cat`), because WaterTAP
  takes the evaporated water out of the liquid outflow (its outflow is
  1.7e-4 below its inflow) and difflow does not. A trial of that term gave
  3.7 times WaterTAP's evaporation, so it was not kept. With the BSM2 Hill
  function instead, acetate differs by 2.6e-3: at pH 7.47 the Hill
  function still inhibits acetoclasts by 0.13 %.

---

(water-treatment)=
## Conventional treatment

`difflow_water.treatment` has four palette units for a conventional
drinking-water train. They use plain streams: `F_H2O` plus one
`F_<name>` per solute, particle class or organism. Particle classes count
`1 mol = 1 g` of solids, organisms `1 mol = 1 organism`, and volumetric flow
is mass over a constant 1000 kg/m^3 (WaterTAP's MCAS default).

(op-sedimentation)=
### Sedimentation

`Sedimentation(SedimentationParams(...))(feed) -> (clarified, sludge, info)`.
This is the ideal basin of Hazen [D3] and Camp [D4]: plug flow, discrete
particles settling at constant velocity. Each class is removed in the
fraction `r = min(1, v_s / v_o)`, where `v_o = Q / A` is the overflow rate,
independent of depth. That follows directly from the ideal-basin
assumptions; neither paper was read. Settling velocities are given, or come
from Stokes' law `v_s = (2/9) a_p^2 (rho_p - rho_w) g / mu` (as stated in
[F2], read), using the water viscosity of [S1]. The sludge carries the
removed solids at a set solids fraction. Flocculent and hindered settling
are not modelled.

**Check** (notebook 78). The ideal-basin law was re-derived by tracking
20,000 particles through a rectangular basin of three depths, and the
removal matches the unit's at every depth. No published settling-column
or basin data set was reachable.

(op-filtration)=
### Filtration

`Filtration(FiltrationParams(...))(feed) -> (filtrate, retained, info)`, a
clean-bed rapid granular filter:

| Quantity | Equation | Source and status |
|---|---|---|
| removal | `ln(C/C0) = -(3/2)(1 - eps) alpha eta L / d_c` | derived here (collector mass balance); the form of Yao et al. [F1] |
| single collector efficiency | `eta = 4.04 N_Pe^(-2/3) + N_G + (3/2) N_R^2`, `N_Pe = d_c U / D`, `D = k_B T / (3 pi mu d_p)`, `N_G = v_s / U`, `N_R = d_p / d_c` | Yao [F1] as restated by Messina et al. [F2] (read): cross-checked, secondary |
| head loss | `h_L / L = 180 mu (1 - eps)^2 U / (rho g eps^3 (phi d_c)^2)` | Carman-Kozeny [F3], Kozeny constant 5 (unverified) |

Yao's efficiency is not capped at 1 (it can exceed 1 at the extremes,
[F2]). Not modelled: clogging, ripening, breakthrough and backwash.
Later correlations (Rajagopalan-Tien, Tufenkji-Elimelech) are not
implemented.

**Check** (notebook 78). Integrating the collector balance slice by slice
reproduces the closed-form removal to 2e-5. The least-removed particle,
about 1.6 um at 12 m/h through 0.5 mm sand, is the classical minimum of
Yao's efficiency. The grain Reynolds number at 12 m/h is 1.5, at the edge
of Carman-Kozeny's laminar range. No published filter-run data were
reachable.

(op-disinfection)=
### Disinfection

`Disinfection(DisinfectionParams(...))(feed) -> (treated, info)`. The
kinetics are Chick-Watson, `dN/dt = -k C^n N` [K1, K2], with first-order
residual decay `dC/dt = -k_d C`. The contact tank is `n_tanks` equal CSTRs
in series, so each tank gives `C_i = C_{i-1}/(1 + k_d tau)` and
`N_i = N_{i-1}/(1 + k C_i^n tau)`. `info` reports the log inactivation, the
CT product `sum C_i tau` (mg min/L), the outlet residual and the contact
time. Rate constants are inputs; the defaults are ILLUSTRATIVE, since real
values depend on organism, disinfectant, pH and temperature. The 1908
papers were not read (their PMC copies are behind a bot check), so the
forms are (unverified) against them.

**Check against the EPA CT table** (notebook 78). The US EPA's Surface
Water Treatment Rule guidance [EP1] tabulates the CT for 3-log *Giardia*
inactivation by free chlorine against residual, temperature and pH. The
values used are as reproduced in [TN] Table 1, at 5 and 15 C, pH 7.0 and
residuals 0.6-2.8 mg/L. In plug flow, Chick-Watson needs
`CT = 3 ln 10 C^(1-n) / k`. Two constants per temperature fit the
tabulated CT within 2.8 %: `n` = 0.854 at 5 C and 0.855 at 15 C, with `k`
= 0.046 and 0.092 (L/mg)^n/min. At the table's CT the fitted law gives
2.95-3.09 logs, and the unit as 200 tanks in series 0.05 log less. One
mixed tank at the same CT gives 0.9 log, which is why the regulations
credit only a fraction of the residence time. The pH and temperature
dependence of `k` is not modelled: take `k` and `n` from the table column
for the conditions.

(op-gacadsorber)=
### GACAdsorber

`GACAdsorber(GACParams(...))(feed) -> (treated, removed, info)`, or
`gac_performance(feed, params)`. It is WaterTAP 1.8.0's `GAC`, the
empirical constant-pattern homogeneous surface diffusion model (CPHSDM) of
Hand et al. [A1], transcribed from WaterTAP's code. [A1] was not read, so
the equations and the regression constants are (unverified) against it.

| Quantity | Equation |
|---|---|
| equilibrium, distribution, Biot | `q_e = K C0^(1/n)`, `D_g = rho_a q_e (1 - eps)/(eps C0)`, `Bi = k_f (d_p/2)(1 - eps)/(D_s D_g eps)` |
| constant pattern | `St_min = a0 Bi + a1`, `EBCT_min = St_min (d_p/2)/((1 - eps) k_f)` |
| throughput, time to ratio `c` | `T(c) = b0 + b1 c^b2 + b3/(1.01 - c^b4)`; `t(c) = eps (D_g + 1)[EBCT_min T(c) + EBCT - EBCT_min]` |
| bed | `L = v EBCT`, `A = Q/v`, carbon `rho_b L A`, usage `rho_b L A / t(c_replace)` |
| steady-state removal | `1 - c_avg`, with `c_avg` the trapezoidal average of `c` over a bed life, from 0.01 to `c_replace` in `n_elements` points plus (0, 0) |

The regression constants depend on Bi and 1/n (tables in [A1]); the
defaults are WaterTAP's example values. The model holds only for
`EBCT >= EBCT_min` (`info["constant_pattern"]`). Fixed `k_f` and `D_s`
only: WaterTAP's calculated options (Gnielinski film transfer, a
surface-diffusion correlation) are not implemented. GAC costing is not
implemented either.

(water-gac-comparison)=
### Comparison with WaterTAP (GAC)

`benchmarks/watertap/gac.py` runs WaterTAP's example GAC flowsheet
(`flowsheets/gac/gac.py`: fixed `k_f`, `D_s`, trapezoidal effluent) at the
example inputs and at three variations: EBCT 900 s, replacement at 20 %
breakthrough, and `1/n = 0.6`. `compare_gac.py` compares every model
quantity (`q_e`, `D_g`, `Bi`, `St_min`, `EBCT_min`, throughput, operating
time, bed volumes, bed geometry, carbon mass and usage rate, `c_avg`, mass
adsorbed) and the outlet solute flow. Largest relative difference over
everything: **2e-14** (the same explicit equations, evaluated in another
code).

---

(water-thermal)=
## Thermal desalination

`difflow_water.thermal` has two whole-plant blocks: single-stage
mechanical vapor compression (MVC), built equation for equation as
WaterTAP's MVC flowsheet and cross-checked against it, and parallel-feed
multi-effect distillation (MED), a first-principles model with no WaterTAP
counterpart (WaterTAP 1.8.0 has none).

(water-thermal-properties)=
### Thermal seawater properties

Added to `difflow_water.seawater` for the evaporators:

| Function | Equation | Status |
|---|---|---|
| `specific_heat(T, S)` | [S1] Eq. 9, T68 by Eq. 4, `S_P = 1000 S` (as WaterTAP) | content verified |
| `enthalpy(T, S, P)` | [S1] Eq. 43 with Eq. 55 for water; 6-digit coefficients and the pressure term of [S2] Eqs. 25-26 as WaterTAP codes them | Eq. 43 content verified at its printed digits; [S2] values (unverified) |
| `latent_heat(T, S)` | [S1] Eqs. 37 and 54 | content verified |
| `vapor_pressure_water(T)` | [S1] Eq. 53 (Hyland-Wexler), full coefficients of [S2] Eq. 5 via WaterTAP | Eq. 53 content verified at its printed digits; full digits (unverified) |
| `vapor_pressure(T, S)` | `p_w exp(B1 s + B2 s^2)`, [S2] Eq. 6 via WaterTAP | (unverified) |
| `boiling_point_elevation(T, S)` | [S1] Eq. 36 | content verified |
| `vapor_enthalpy(T, P)` | `h_w(T, P) + h_fg(T)`: WaterTAP's water property package (ideal-gas vapor) | convention |

(op-mvcdesalinator)=
### MVCDesalinator

`MVCDesalinator(MVCParams(...))(feed) -> (distillate, brine, info)`, or
`mvc_performance(feed, params)` for the full result (state, works, capital
per unit, SEC, LCOW). The layout is WaterTAP's `mvc_single_stage`: feed
pump, a split (fraction = recovery) to a distillate and a brine
preheater, mixer, evaporator, compressor, condenser (the evaporator's hot
side), brine and distillate pumps.

| Part | Equations |
|---|---|
| Evaporator | brine and vapor at `T_b`; brine at `p_sw(T_b, S_b)`; `Q + H_feed = H_brine + H_vapor`; `Q = U A LMTD` with `dT_in = T_compressed - T_b`, `dT_out = T_condensate - T_b` |
| Compressor | `P_out = r P_in`; `T_s = T_in r^(1-1/gamma)`, `gamma = 1.3`; `eta (h_out - h_in) = h_s - h_in` |
| Condenser | complete condensation at the compressor pressure, `m_v (h_v,in - h_l,out) = Q` |
| Preheaters | countercurrent, `Q = U A LMTD`, Chen's `LMTD = (dT1 dT2 (dT1 + dT2)/2)^(1/3)` |
| Pumps | isothermal, `W = V dP / eta` |
| Costing | WaterTAP's MVC example: evaporator 1000 USD(2020)/m^2 and preheaters 300 USD(2020)/m^2, each x 5 (material) x TIC 2; compressor `7364 USD(2001) m_v r (eta/(1-eta))^0.7`; pumps 1.908 USD(2018)/W; mixer 361 USD(2018)/(L/s); electricity 0.1 USD(2018)/kWh; CEPCI 394.3 (2001), 603.1 (2018), 596.2 (2020) as IDAES has them (unverified) |

The plant is one Newton solve (`optimistix.root_find`, 8 unknowns:
condensate and preheater temperatures, mixed feed temperature, duty,
pressure ratio), so `jax.grad` is implicit and exact at the solution.

Degrees of freedom: brine temperature, recovery and the two preheater
areas are inputs; the evaporator area is computed. The pressure ratio is
either solved so that the condensate leaves saturated
(`mode="saturated_condensate"`, the default) or given
(`mode="fixed_pressure_ratio"`, WaterTAP's simulation). With a fixed
pressure ratio the compressor's energy has to match what the plant
absorbs, and most designs are then unphysical: the condensate would be
hotter than saturation (WaterTAP: infeasible) or colder than the brine.
`info` reports `condensate_subcooling`, `evaporator_dT_out` and
`feasible`. Preheater areas are absolute and must be sized to the feed:
WaterTAP's optimum is about 4.5 and 3.9 m^2 per kg/s of feed. Oversized
preheaters leave too little work for the compressor, and the
saturated-condensate solution is then infeasible too.

(op-meddesalinator)=
### MEDDesalinator

`MEDDesalinator(MEDParams(...))(feed) -> (distillate, brine, info)`, or
`med_performance(feed, params)`. Parallel feed, `N` effects with brine
temperatures falling linearly from `T_top` to `T_last`, each fed `F/N` at
`T_feed`:

- effect 1 heated by motive steam, `Q_1 = m_s h_fg(T_steam)`; effect `i`
  by the vapor of effect `i-1` condensing at `T_{i-1} - BPE_{i-1}`,
  `Q_i = D_{i-1} (h_v(T_{i-1}) - h_w(T_{i-1} - BPE_{i-1}))`;
- `Q_i + F_i h(T_f, S_f) = B_i h(T_i, S_i) + D_i h_v(T_i)`,
  `F_i S_f = B_i S_i`, solved for `D_i` (Newton, fixed count);
- `A_i = Q_i / (U (T_c,i - T_i))`; the final condenser preheats the feed
  from the intake temperature and rejects the rest to cooling seawater;
- `GOR = sum D_i / m_s`.

Everything is mass and energy balances on the properties above; the
configuration is the textbook one [D2] (book not read: the equations are
this module's own). Not modelled: feed preheaters between effects,
distillate flash boxes, non-condensables, pressure losses. Parallel feed
heats each effect's feed from `T_feed` inside the effect, which costs GOR
(about 4.6 for the default 8 effects). Defaults are ILLUSTRATIVE.

(water-mvc-comparison)=
### Comparison with WaterTAP (MVC)

`benchmarks/watertap/mvc.py` runs WaterTAP 1.8.0's `mvc_single_stage`
(its own build, operating conditions and initialization): its LCOW
optimum, and a simulation at a lower pressure ratio (1.58, preheaters 199 /
174 m^2). `compare_mvc.py` runs difflow on the same inputs with
`mode="fixed_pressure_ratio"`, and repeats the optimization (SLSQP with
`jax.grad`, WaterTAP's bounds) from the simulation design.

Largest relative difference over every reported output (evaporator area,
duty and LMTD, all temperatures and pressures, compressor and pump works,
product flow, SEC, LCOW, total capital and operating cost, and the capital
of each of the 8 units): **2.6e-7** at the optimum, **3.9e-7** at the
simulation point.

| Decision at the LCOW optimum | WaterTAP (IPOPT) | difflow (SLSQP + `jax.grad`) | rel. diff |
|---|---|---|---|
| brine temperature (K) | 348.15 (bound) | 348.15 | +1.5e-09 |
| pressure ratio | 1.604816 | 1.604816 | -7.5e-09 |
| distillate preheater (m^2) | 199.2295 | 199.2295 | -2.4e-08 |
| brine preheater (m^2) | 174.4172 | 174.4171 | -8.3e-08 |
| LCOW (USD/m^3) | 4.518535 | 4.518535 | +1.0e-07 |

What the comparison taught:

- **Currency year.** WaterTAP's example sets `base_currency = USD_2020`
  after its costs are built, so its results are in its default USD 2018;
  difflow's `MVCCostingParams.base_year` defaults to 2018 to match (with
  2020 every cost was 1.14 % lower).
- **Preheater pressure "drop".** The example fixes IDAES
  `deltaP = +7 kPa` on both sides of both preheaters: a pressure gain,
  which lowers the brine pump's work by 19 %. difflow's `hx_dP` is a loss
  (default +7 kPa); the comparison passes `hx_dP = -7e3`.
- **Feasibility.** At WaterTAP's optimum the condensate leaves exactly
  saturated (the condenser's `P >= P_sat` bound is active). Any higher
  pressure ratio at the same design is infeasible, and so were several
  lower-ratio designs tried; hence the saturated-condensate mode.

---

(water-editor)=
## In the editor

`ROStage`, `ROTrain`, `PressureExchanger`, `WaterPump`, `MVCDesalinator`,
`MEDDesalinator`, `GACAdsorber`, `Disinfection`, `Sedimentation`,
`Filtration`, `ASM1Reactor`, `TakacsSettler`, `ActivatedSludgePlant` and
`AnaerobicDigester` are palette operations (category `water`) in the difflow editor (`difflow gui`, from the
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

(water-unit-checks)=
## Per-unit examples and checks

Notebooks 77-79 take each unit on its own: how it works, then the check
behind it. The plant-level studies are notebooks 70-76.

| Unit | Notebook | Checked against | Result |
|---|---|---|---|
| seawater properties | 77 | TEOS-10 (GSW reference code) | within Sharqawy's stated accuracy except the enthalpy rise (0.55 % vs 0.5 %); see [above](water-seawater-teos10) |
| `ROStage`, `ROTrain` | 77 | WaterTAP RO1D; thermodynamic least work | within 0.3 %; plant SEC 2.5-2.7 times the least work |
| `PressureExchanger`, `WaterPump` | 77 | WaterTAP; the defining identities | 0.02 % / 0.24 % in the plant |
| `MVCDesalinator` | 77 | WaterTAP | 4e-7 |
| `MEDDesalinator` | 77 | nothing published; mass and salt balances | balances close |
| `Sedimentation` | 78 | the ideal-basin law, re-derived by tracking particles | equal at every depth |
| `Filtration` | 78 | the removal law, re-derived slice by slice; Yao's efficiency as [F2] states it | equal to 2e-5 |
| `Disinfection` | 78 | the US EPA SWTR CT table for 3-log *Giardia* by free chlorine [EP1, TN], 5 and 15 C, pH 7 | Chick-Watson fits it within about 3 % (`n` = 0.85); at the table's CT the fitted law gives 2.95-3.09 logs |
| `GACAdsorber` | 78 | WaterTAP | 2e-14 (WaterTAP's example is outside the constant-pattern regime) |
| `ASM1Reactor` | 79 | BSM1 Table 6: each reactor from the printed state of the one before | within 0.4 % (the printed inlet's precision) |
| `TakacsSettler` | 79 | BSM1 Table 7, fed the printed reactor 5 | within 0.13 % |
| `ActivatedSludgePlant` | 79 | BSM1 Tables 6-7; WaterTAP | 0.3 %; 1.3e-7 |
| `AnaerobicDigester` | 79 | Rosén & Jeppsson Section 7; WaterTAP | 1.7e-7; 2.5e-4 |

Where no published data set could be read (RO and MVC plant data, MED,
settling-column and filter-run data, GAC column tests), the table says
so. A check against WaterTAP is an implementation check (same equations,
another code), not a validation of the physics.

(water-watertap-coverage)=
## Coverage relative to WaterTAP

WaterTAP is the benchmark. This is what `difflow_water` covers of WaterTAP
1.8.0 (the version every comparison here was run against), and what it
does not. By unit-model count, the coverage is about 15-20 % of WaterTAP,
and much less if its zero-order library counts.

**Covered and cross-checked against WaterTAP:**

| WaterTAP | difflow_water | Agreement |
|---|---|---|
| RO1D + pressure exchanger + pumps (`RO_with_energy_recovery`) | `ROStage`, `ROTrain`, `PressureExchanger`, `WaterPump`, `swro_performance` | within 0.3 % |
| `mvc_single_stage` | `MVCDesalinator`, `mvc_performance` | 4e-7; LCOW optimum to 1e-7 |
| `GAC` (CPHSDM, fixed `k_f`/`D_s`) | `GACAdsorber` | 2e-14 |
| `ASM1_flowsheet` (BSM1 layout) | `ActivatedSludgePlant`, `activated_sludge_plant` | 1.3e-7 |
| `ADM1_flowsheet` | `AnaerobicDigester`, `anaerobic_digester` | 2.5e-4 |
| `WaterTAPCosting` | `economics` (RO, pumps, PX, MVC units) | equal for those units |

**In WaterTAP, not yet in difflow_water** (in rough order of priority for
comparability):

1. Nanofiltration: `nanofiltration_DSPMDE_0D`, `nanofiltration_0D`, `nanofiltration_ZO`.
2. Electrodialysis: `electrodialysis_0D`, `_1D`, `_bipolar_1D`.
3. Ion exchange: `ion_exchange_0D`.
4. The zero-order library (`unit_models/zero_order`), about a hundred
   removal-fraction units with their cost curves; most treatment-train
   screening uses it.
5. Plant-wide wastewater: BSM2 (`full_water_resource_recovery_facility`),
   the ASM-ADM translators, thickener, dewatering, the clarifier unit.
6. Other activated-sludge models: ASM2d, modified ASM2d, ASM3, modified
   ADM1. ASM2d is blocked on a readable primary source (see Known gaps).
7. Other membranes: osmotically assisted RO (0D/1D), low-salt-rejection RO
   (LSRRO), membrane distillation, RO0D.
8. Other units: crystallizer (and its surrogate), UV/AOP, electrocoagulation,
   electrolyzer, boron removal, coagulation/flocculation, steam ejector and
   heater, stoichiometric reactor, generic desalter/separation.
9. GAC options: calculated film transfer (Gnielinski) and surface diffusion,
   the surrogate CPHSDM, and GAC costing.
10. Property packages: MCAS (multicomponent aqueous solution), NaCl, and the
    water/vapor package beyond what MVC needs.
11. Costing for every unit beyond those listed above, and WaterTAP's
    parameter-sweep and UI tooling (difflow has its own sweeps and editor).

**In difflow_water, not in WaterTAP:** water distribution networks
(EPANET 2.2 hydraulics, extended-period simulation, water quality, pump
scheduling; cross-checked against EPANET), Pitzer speciation and scaling
(cross-checked against PHREEQC; WaterTAP uses Reaktoro), multi-effect
distillation, sedimentation, rapid filtration, disinfection, the Takács
settler, and exact JAX gradients through every model, including recycles
and steady-state solves.

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
- Treatment: no coagulation/flocculation, softening, ion exchange,
  stripping, UV or membrane filtration (UF/MF); clean-bed filtration only;
  Chick-Watson constants and GAC regression constants are inputs; no
  treatment costing yet.
- Wastewater: steady state only (no dynamic BSM1 influent files, no
  controllers or performance indices); ASM1 at BSM1's 15 C parameters with
  no temperature dependence; no ASM2d or ASM3 yet (ASM2d waits for a
  readable primary source: the only spec in hand, WaterTAP's code, has
  inconsistencies that cannot be resolved without Henze et al. 1999); no aeration energy
  costing. No ASM1-ADM1 interfaces (the digester takes an ADM1 stream).
- Thermal: no multi-stage flash (MSF) yet; MED is parallel feed without
  inter-effect preheaters or flash boxes, and has no WaterTAP counterpart to
  compare with; MVC is single-stage. The vapor pressure of seawater and the
  pressure term of the enthalpy are [S2]'s via WaterTAP (unverified).
- Networks: water quality is not differentiable yet; no rule-based
  controls, tank volume curves, multi-point pump curves, quality sources or
  Chezy-Manning.

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
| S1 | Sharqawy, M.H.; Lienhard, J.H.; Zubair, S.M. "Thermophysical properties of seawater: a review of existing correlations and data." *Desalin. Water Treat.* **2010**, 16(1-3), 354--380. doi:10.5004/dwt.2010.1079 | Density (Eq. 8), viscosity (Eqs. 22, 23), osmotic coefficient (Eq. 49); specific heat (Eq. 9), BPE (Eq. 36), latent heat (Eqs. 37, 54), enthalpy (Eqs. 43, 55), water vapor pressure (Eq. 53) | Citation verified (Crossref). **Content verified**: coefficients read against the authors' corrected version (web.mit.edu/lienhard), including the five correction notes (none affect these equations). |
| S2 | Nayar, K.G.; Sharqawy, M.H.; Banchik, L.D.; Lienhard V, J.H. "Thermophysical properties of seawater: A review and new correlations that include pressure dependence." *Desalination* **2016**, 390, 1--24. doi:10.1016/j.desal.2016.02.024 | Osmotic pressure form `phi m rho_w R T` (Eq. 48, per WaterTAP); enthalpy coefficients and pressure term (Eqs. 25-26), water vapor pressure coefficients (Eq. 5) and seawater vapor pressure (Eq. 6), all per WaterTAP | Citation verified (Crossref). Content **unverified** (paper not reached; attribution from WaterTAP's code). |
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
| N1 | Klise, K.A.; Bynum, M.; Moriarty, D.; Murray, R. "A software framework for assessing the resilience of drinking water systems to disasters with an example earthquake case study." *Environ. Model. Softw.* **2017**, 95, 420--431. doi:10.1016/j.envsoft.2017.06.022 | WNTR, used to run EPANET 2.2 for the network benchmark | Citation verified (Crossref) |
| N2 | Todini, E.; Rossman, L.A. "Unified framework for deriving simultaneous equation algorithms for water distribution networks." *J. Hydraul. Eng.* **2013**, 139(5), 511--526. doi:10.1061/(ASCE)HY.1943-7900.0000703 | Context: the gradient method as Newton on heads and flows | Citation verified (Crossref); not read |
| N3 | Wagner, J.M.; Shamir, U.; Marks, D.H. "Water distribution reliability: simulation methods." *J. Water Resour. Plann. Manage.* **1988**, 114(3), 276--294. doi:10.1061/(ASCE)0733-9496(1988)114:3(276) | Pressure-driven demand | Citation verified (Crossref). The equation is EPANET's form ([N8] Eq. 13.6, content verified there), not read in this paper. |
| N7 | Swamee, P.K.; Jain, A.K. "Explicit equations for pipe-flow problems." *J. Hydraul. Div.* **1976**, 102(5), 657--664. doi:10.1061/JYCEAJ.0004542 | Turbulent friction factor | Citation verified (Crossref; the DOI first recalled was wrong). Paper not read; the log10 form is the one that reproduces EPANET (see the network notes). |
| N8 | Rossman, L.; Woo, H.; Tryby, M.; Shang, F.; Janke, R.; Haxton, T. *EPANET 2.2 User Manual*; EPA/600/R-20/133; U.S. Environmental Protection Agency: Cincinnati, OH, **2020**. Online edition: epanet-manual.readthedocs.io | Network equations (Section 13.1, Table 3.1, pump curves, valve and emitter models, PDD) | Citation verified (report number, authors and title from the EPA PDF cover, via search; no DOI). **Content verified**: equations read in the online edition. Its Swamee-Jain `ln` is a typo (see the network notes). |
| W5 | Alex, J.; Benedetti, L.; Copp, J.; Gernaey, K.V.; Jeppsson, U.; Nopens, I.; Pons, M.-N.; Rieger, L.; Rosen, C.; Steyer, J.P.; Vanrolleghem, P.; Winkler, S. *Benchmark Simulation Model no. 1 (BSM1)*; Technical report LTH-IEA-7229 (CODEN LUTEDX/(TEIE-7229)/1-62/(2008)); Dept. of Industrial Electrical Engineering and Automation, Lund University, **2008**. No DOI; Lund University record lup.lub.lu.se/record/4391579 | ASM1 (variables, rates, conversion rates, Tables 2-3), the BSM1 plant, the Takács settler (Table 4), the open-loop steady state (Tables 6-7) | **Content verified** (read; PDF from www2.iea.lth.se). Title, authors and report number checked against the Lund record. |
| W6 | Rosén, C.; Jeppsson, U. *Aspects on ADM1 Implementation within the BSM2 Framework*; Report TEIE-7224 (CODEN LUTEDX/(TEIE-7224)/1-35/(2006)); Dept. of Industrial Electrical Engineering and Automation, Lund University, **2006**. No DOI; Lund University record lup.lub.lu.se/record/588079 | ADM1 as implemented for BSM2: rates, inhibition, balances, acid-base, gas phase, parameters (Section 6), the steady-state verification case (Section 7) | **Content verified** (read; PDF from www2.iea.lth.se). |
| W2 | Batstone, D.J.; Keller, J.; Angelidaki, I.; Kalyuzhnyi, S.V.; Pavlostathis, S.G.; Rozzi, A.; Sanders, W.T.M.; Siegrist, H.; Vavilin, V.A. "The IWA Anaerobic Digestion Model No 1 (ADM1)." *Water Sci. Technol.* **2002**, 45(10), 65--73. doi:10.2166/wst.2002.0292 | ADM1 origin | Citation verified (Crossref). Not read: ADM1 as [W6] states it. |
| W1 | Takács, I.; Patry, G.G.; Nolasco, D. "A dynamic model of the clarification-thickening process." *Water Res.* **1991**, 25(10), 1263--1271. doi:10.1016/0043-1354(91)90066-Y | Double-exponential settling velocity, layered settler | Citation verified (Crossref). Not read: the model as [W5] states it. |
| W3 | Henze, M.; Gujer, W.; Mino, T.; van Loosdrecht, M. *Activated Sludge Models ASM1, ASM2, ASM2d and ASM3*; IWA Scientific and Technical Report No. 9; IWA Publishing, 2000. doi:10.2166/9781780402369 | ASM1 origin | DOI and authors verified (Crossref). Not read: ASM1 as [W5] states it. |
| A1 | Hand, D.W.; Crittenden, J.C.; Thacker, W.E. "Simplified models for design of fixed-bed adsorption systems." *J. Environ. Eng.* **1984**, 110(2), 440--456. doi:10.1061/(ASCE)0733-9372(1984)110:2(440) | CPHSDM (GAC) | Citation verified (Crossref). Not read: equations and regression form via WaterTAP's code (unverified). |
| K1 | Chick, H. "An investigation of the laws of disinfection." *J. Hyg.* **1908**, 8(1), 92--158. doi:10.1017/S0022172400006987 | First-order inactivation | Citation verified (Crossref). Not read (unverified). |
| K2 | Watson, H.E. "A note on the variation of the rate of disinfection with change in the concentration of the disinfectant." *J. Hyg.* **1908**, 8(4), 536--542. doi:10.1017/S0022172400015928 | Concentration exponent | Citation verified (Crossref). Not read (unverified). |
| D3 | Hazen, A. "On sedimentation." *Trans. ASCE* **1904**, 53, 45--71. doi:10.1061/TACEAT.0001655 | Ideal basin, overflow rate | Citation verified (Crossref). Not read: the removal law is derived from the ideal-basin assumptions. |
| D4 | Camp, T.R. "Sedimentation and the design of settling tanks." *Trans. ASCE* **1946**, 111, 895--936. doi:10.1061/TACEAT.0005912 | Ideal basin (type I settling) | Citation verified (Crossref). Not read. |
| F1 | Yao, K.-M.; Habibian, M.T.; O'Melia, C.R. "Water and waste water filtration. Concepts and applications." *Environ. Sci. Technol.* **1971**, 5(11), 1105--1112. doi:10.1021/es60058a005 | Filtration theory, single collector efficiency | Citation verified (Crossref). Not read: the efficiency as restated in [F2]. |
| F2 | Messina, F.; Marchisio, D.L.; Sethi, R. "An extended and total flux normalized correlation equation for predicting single-collector efficiency." *J. Colloid Interface Sci.* **2015**, 446, 185--193. doi:10.1016/j.jcis.2015.01.024 | Yao's efficiency (Eq. 1), dimensionless groups, Stokes and Stokes-Einstein (Table 1) | Citation verified (Crossref). **Content verified** (author manuscript, polito.it). |
| F3 | Carman, P.C. "Fluid flow through granular beds." *Trans. IChemE* **1937**, 15, 150--166; reprinted *Chem. Eng. Res. Des.* **1997**, 75, S32--S48. doi:10.1016/S0263-8762(97)80003-2 | Clean-bed head loss | Citation verified (Crossref, the reprint). Not read: Kozeny constant 5 (unverified). |
| GSW | GSW-Python 3.6.23, the TEOS-10 Gibbs SeaWater reference software (teos-10.org; PyPI `gsw`) | TEOS-10 reference values for the seawater property check (`benchmarks/teos10/`) | Software, run directly (not a difflow dependency). |
| EP1 | U.S. Environmental Protection Agency. *Guidance Manual for Compliance with the Filtration and Disinfection Requirements for Public Water Systems Using Surface Water Sources*; March **1991** edition | CT values for *Giardia* inactivation by free chlorine | Not read in the original: the values are as reproduced in [TN]. |
| TN | Tennessee Department of Environment and Conservation, Division of Water Supply. *Guidance Manual for Determination of Disinfectant Contact Time and CT Requirements for Public Water Systems*; July **1991** | Table 1, reproducing [EP1]'s CT table for unfiltered systems (5 and 15 C, pH 7.0 columns used) | Read (scanned copy; a few OCR artifacts, none in the columns used). |
| D2 | El-Dessouky, H.T.; Ettouney, H.M. *Fundamentals of Salt Water Desalination*; Elsevier: Amsterdam, **2002**. doi:10.1016/B978-0-444-50810-2.X5000-3 | MED and MVC configurations | DOI, title, year and publisher verified (Crossref; it lists no authors, so authorship is (unverified)). Not read: no equation is taken from it. |
| B1 | Akkor, I.; Amusat, O.O.; Vecchiarelli, P.; Gounaris, C.E.; Knueven, B.; Dudchenko, A.V. "Optimization of desalination systems with detailed water chemistry through integration of Reaktoro in WaterTAP." *ACS ES&T Eng.* **2026**, 6(5), 1464--1477. doi:10.1021/acsestengg.5c01008 | WaterTAP (benchmark context) | Citation verified (Crossref) |
| B2 | Dudchenko, A.; Atia, A.; Bartholomew, T. "WaterTAP: Tool for analysis of water treatment systems." NAWI Alliance Fall Meeting, **2023** (OSTI). doi:10.2172/2447624 | WaterTAP (benchmark) | Citation verified (Crossref). The comparison used WaterTAP 1.8.0 from PyPI. |
| B3 | Lee, A.; Ghouse, J.H.; Eslick, J.C.; Laird, C.D.; Siirola, J.D.; et al. "The IDAES process modeling framework and model library -- flexibility for process simulation and optimization." *J. Adv. Manuf. Process.* **2021**, 3(3), e10095. doi:10.1002/amp2.10095 | IDAES (pump model, WaterTAP's base) | Citation verified (Crossref) |
| WT | WaterTAP 1.8.0 source (`seawater_prop_pack.py`, `water_prop_pack.py`, `membrane_channel_base.py`, `reverse_osmosis_1D.py`, `pressure_exchanger.py`, `mvc/`, `flowsheets/mvc/`, `gac.py`, `flowsheets/gac/`, `flowsheets/activated_sludge/`, `anaerobic_digester.py`, `flowsheets/anaerobic_digester/`, `costing/`), read from the PyPI wheel | Equations as coded; costing structure and defaults | Read directly (the code, not the papers behind it). |
