# Changelog

All notable changes to difflow are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org) (pre-1.0: a minor bump may break the API).

## [Unreleased]

### Added

- **Ternary-diagram helpers: `mix`, `lever_rule`, `tie_line_through`,
  `difference_point`, `hunter_nash`, `plot_hunter_nash` (#430, part 2).** In
  `difflow.phase_diagrams` (`mix` only there; the rest also top-level) and
  `difflow.visualization`. `mix` and `lever_rule` are pure jnp and
  differentiable; `tie_line_through`, `hunter_nash` (stage stepping with a real,
  fractionally interpolated stage count, pinch reported as `feasible=False`) and
  `difference_point` work on the conjugate curves of the `ternary_lle` tie lines
  (monotone cubic interpolation, numeric, not differentiated through the
  equilibrium); the operating lines use the extensive balance so a difference
  point at infinity is handled. Right-triangle coordinates for leaching
  diagrams were already provided by `ternary_xy(coords="right")`. No published
  Hunter-Nash example is reproduced; checked against the lever rule, component
  balances and the analytic Kremser stage count of an immiscible ternary.

- **`Leacher`: countercurrent leaching and washing (#430, part 1).** New palette
  unit in `difflow.units.leaching` (category `extraction`, mass flows in kg/s):
  ideal-stage solid-liquid cascade with constant-solution, constant-solvent or
  tabulated variable underflow. Constant underflow is a closed form (first stage
  separate when the feed solids carry a different solution amount) with a
  continuous, differentiable stage count and a smooth `r = L/V = 1` limit;
  variable underflow is a fixed-iteration Newton solve of the stage balances
  (integer stages). `recovery` with `design="stages"` returns N in closed form,
  with `design="solvent"` scales the solvent stream (implicit-gradient
  bisection). Library helpers `washing_recovery`, `washing_stages` (the Kremser
  form in the wash ratio) and `tabulate_underflow`. `stage_efficiency` is an
  overall efficiency (ideal stages = E x n_stages). No textbook worked example is
  reproduced (inputs not retrievable); the tests use an independent dense
  linear-system assembly, the closed washing form, a SciPy solve and
  `check_grads`. The variable-underflow table in the docs is synthetic.

- **`difflow.units.column_design`: column hardware (#429).** New core library
  module (no palette entry): `flow_parameter`, `fair_capacity_factor`,
  `flooding_velocity`, `tray_column_diameter` (Fair flooding, `fraction_flood`,
  downcomer fraction, optional foaming/hole-area multipliers, `(D, info)`),
  `tray_percent_flood` (rating), `weir_crest` (Francis), `tray_pressure_drop`
  (dry + liquid head + residual, components in `info`), `weep_point` (margin),
  `fractional_entrainment`, `oconnell_efficiency` (promoted from the gas plant,
  re-exported there; optional `smooth=` clip), `gpdc_flood`, `gpdc_pressure_drop`,
  `kister_gill_flood_dp`, `packed_column_diameter` (flood, dP/m, wetting
  margin), `hetp` (random, structured, from HTU), `size_column`,
  `column_hardware_cost` (into `column_cost`). `PACKINGS` gained Intalox
  ceramic 25/38/50, Pall metal 38 and Mellapak 250Y. Surface tension, viscosity
  and densities are required arguments. **Every correlation constant is from
  memory and unverified** (no book access): Fair's chart fit (Treybal form),
  Francis crest, the 12.5 kg/m^2 residual head, the Eduljee weep constant
  (`K2 = 30` fixed, the chart dependence is not reproduced), the Eckert
  flood-line log-quadratic, Kister-Gill flood dP, the minimum wetting rate, the
  HETP rules and the packing table; expect chart-reading accuracy (10-20 %).
  **Two pieces are placeholders, not literature fits:** `fractional_entrainment`
  (assumed shape, order of magnitude only) and the exponent of
  `gpdc_pressure_drop` (anchored to the flood dP, exponent assumed). **No
  published textbook worked example is reproduced**; tests use hand
  calculations, independent re-implementations, scaling laws, margin sign
  changes, `check_grads` and the DistillationColumn -> diameter -> size ->
  cost chain differentiated in R. Docs: `## Column Hardware` in
  `docs/unit-operations-chemical.md`; notebook `examples/53_column_hardware.ipynb`.

- **`difflow.mass_transfer`: diffusivities, film theory, Sherwood, Onda, HTU/NTU
  (#427).** New core library module (no palette entry): `fuller_diffusivity`,
  `wilke_chang_diffusivity`, `stokes_einstein_diffusivity` (native-unit
  implementations, SI at the boundary) with the tables `FULLER_VOLUMES`,
  `FULLER_ATOMIC_VOLUMES`, `LE_BAS_INCREMENTS`, `ASSOCIATION_FACTORS` and the
  helpers `diffusion_volume`, `molar_volume_nbp`; `equimolar_flux`,
  `stagnant_film_flux` (with a smooth `log_mean`), `mass_transfer_coefficient_conversions`;
  `ranz_marshall`, `frossling`, `linton_sherwood`, `gilliland_sherwood`,
  `pipe_sherwood`, `flat_plate_sherwood`, `chilton_colburn_jD`;
  `overall_gas_coefficient`, `overall_liquid_coefficient`, `overall_K_G`,
  `resistance_fractions` (optional enhancement factor); the shared
  `PackingData` record and `PACKINGS` table; `onda_wetted_area`,
  `onda_gas_coefficient`, `onda_liquid_coefficient`; `htu_og`, `htu_ol`,
  `htu_og_from_films`, `ntu_og_colburn` (analytic, smooth at lambda = 1),
  `ntu_og_integral` (Gauss-Legendre, optional `(1-y)_lm`), `hetp_from_htu`.
  Gas viscosity and surface tension are **required arguments** (no
  `liquid_properties` additions, no hidden `sigma=0.072`). Validated against
  closed forms, quadrature, independent NumPy transcriptions and two gas and
  two liquid diffusivity data points; **no published textbook worked example
  is reproduced**, and the diffusion-volume, Le Bas and packing tables are
  transcribed from memory (flagged in the docstrings). Docs: `## Mass
  Transfer` in `docs/unit-operations-chemical.md`; example notebook
  `examples/51_mass_transfer.ipynb`.
- **Generic `Absorber` and `Stripper` (#428).** New `difflow.units.absorption`:
  physical-solvent gas absorber and stripper, two inlets (`gas_in`, `liquid_in`)
  and two outlets, sharing one implementation and one `AbsorptionParams`
  (deliberately not `AbsorberParams`/`StripperParams`, which `difflow_cc` and
  `difflow_ree` already export and `difflow.serialize` looks up by name).
  Henry's law per solute with a van 't Hoff temperature dependence
  (`henry_constant`; `HENRY_WATER`, a small table **from memory, unverified**),
  three models: `"kremser"` (closed form per solute for any `A > 0`, solute in
  the entering solvent, Edmister-style geometric-mean `A` iterated with the
  outlet flows, Murphree vapor efficiency by the Lewis relation, real-valued
  `n_stages`), `"stages"` (tray-to-tray, one linear system per solute with Picard
  sweeps on the stage flows, concrete integer `n_stages`) and `"packed"` (HTU/NTU:
  Onda films from `difflow.mass_transfer` or a user `H_OG`; Colburn, or the
  numerical `N_OG` integral with the `(1-y)_lm` option; rating by `height`,
  design by `recovery`). `recovery` also makes the Kremser stage count an
  output. `info` carries per-solute recovery, `A`, `S`, `K`, `N_eq`/`N_OG`/`H_OG`,
  `L_min` (absorber) or `G_min` (stripper) and the ratio to it, the solute-free
  operating line and curve, the heat of absorption and an adiabatic temperature
  rise (reported, not applied), `converged` and `feasible` (an unreachable spec
  gives `inf`, never NaN). Library helpers `solute_free_operating_line`,
  `minimum_solvent_rate` (rich-end or tangent pinch on any
  `difflow.staged.EquilibriumCurve`) and `minimum_stripping_gas_rate`.
  Registered in the catalog (`absorption` module in the `separations` category)
  and the GUI palette (packed-column symbol). Validated against the Kremser
  closed form, a separate SciPy `fsolve` of the tray equations, dilute-limit
  Kremser/stages agreement, Colburn against the NTU integral, a dense chord-slope
  scan for the minimum rate, mass balances, `check_grads` and an absorber-stripper
  recycle flowsheet through `fs.solve` differentiated with respect to the solvent
  rate; **no published textbook example is reproduced** (the issue's numbers are
  unverified recollections). Not implemented: modified-Raoult K from a `thermo`
  object, `fraction_flood` sizing (needs the hardware model of #429), solvent
  volatility, a rate-based Maxwell-Stefan model. Example notebook
  `examples/52_absorption_stripping.ipynb`.

- **`BatchDistillation`: Rayleigh, constant-reflux and constant-composition
  batch distillation (#426).** New `difflow.units.batch_distillation`: a binary
  batch rectifier (`n_stages` equilibrium stages above the still, total
  condenser, constant boil-up, quasi-steady, no holdup) in constant-reflux and
  constant-distillate-composition modes, built on `difflow.staged`'s
  equilibrium curve and stage relation. The charge is a stream read as moles
  (as `FedBatchReactor` reads its initial state). Stopping is data
  (`stop={"x_W" | "x_D_avg" | "distillate" | "time": value}`, first reached
  wins), solved by bisection plus an implicit-function Newton step so every
  result is differentiable in `R`, `boilup`, the charge, the efficiency and
  the spec. `info` carries trajectories (`t`, `W`, `x_W`, `x_D`, `R`,
  `T_still`), accumulated distillate, batch time, `int V dt`, reboiler energy,
  recovery and flags `feasible`, `azeotrope_limited`, `stop_reached` (no NaN
  past an azeotrope). Library functions `rayleigh` (closed form for constant
  alpha, quadrature otherwise, solves for `x_W` or `W/F`),
  `rayleigh_multicomponent`, `equilibrium_curve` and `batch_cuts`
  (heads/hearts/tails with per-cut reflux overrides). Registered in the
  catalog (`distillation` category) and GUI palette (column symbol).
  Validated against the Rayleigh closed form, `scipy` and an independent NumPy
  time integration; **no published textbook example is reproduced** (the
  issue's numbers are unverified recollections). Not implemented:
  multicomponent tray-to-tray batch columns, holdup.

- **`DistillationColumn`: partial condenser and Murphree vapour efficiency
  (#425).** `condenser_type="partial"` makes the condenser an equilibrium stage
  outside `n_stages` (vapour distillate `y_D = K(T_c) x_R` at its dew point,
  liquid reflux `x_R`), as one more row of the tridiagonal system so the
  balances still telescope; `info` gains `y_D`, `x_R`, `n_equilibrium_stages`
  and a partial-basis `Q_condenser`. `stage_efficiency` (scalar or
  `(n_stages,)`, same name as `CascadeParams.stage_efficiency`) is a Murphree
  vapour efficiency solved as a block-tridiagonal system (well posed at
  `E = 0`) with a fixed-count `E = 1 -> E` homotopy; the reboiler is always an
  equilibrium stage. Differentiable in `E` and `R` on the CMO and MESH paths.
  `info["overall_efficiency_equiv"]` is a Fenske-style separation proxy, not a
  theoretical-stage ratio. `stage_efficiency=1.0` (default) with a total
  condenser takes the original code path and reproduces previous results
  exactly. Checked against `difflow.staged.mccabe_thiele`, an independent
  flash and `check_grads`; no textbook example numbers are claimed.
  **Changed:** `condenser_type="partial"` no longer raises
  `NotImplementedError`, and an unknown `condenser_type` is now a
  `ValueError` (`"mixed"` still raises `NotImplementedError`).

- **Shell-and-tube sizing and phase-change warning for the heat-exchanger
  helpers (#422).** `design_heat_exchanger` and `size_heat_exchanger` accept
  `flow_config="shell_and_tube"` with `n_shell_passes`, size with
  `A = Q/(U F LMTD)` (the sizer inverts the exact 1-2N effectiveness) and
  return `F` (1 for the other configurations). `LowCorrectionFactorWarning`
  when `F < 0.75` or `F = 0` (duty past the per-shell limit: `A = inf`, add
  shells); skipped for traced values. An unknown `flow_config` now raises
  `ValueError` (it used to be treated as co-current). The constant-Cp units
  (`CounterCurrentHX`, `CoCurrentHX`, `CrossFlowHX`, `ShellAndTubeHX`,
  `Heater`/`Cooler` without a thermo) emit `PhaseChangeWarning` when a
  stream's temperature span crosses a database species' boiling point at the
  stream pressure (Antoine inversion, species at least 1% of the flow, concrete
  values only, so `jit`/`grad` are unaffected) and point to
  `EnthalpyCounterCurrentHX`.

### Breaking

- **`column_diameter` is deprecated and its signature changed (#429).** The old
  body ignored `sigma` and `tray_spacing`, hard-coded a vapor molecular weight
  of 30 g/mol, `F_LV = 0.1` and `C_SB = 0.1`, and 80 % flood. It now emits a
  `DeprecationWarning` and calls `tray_column_diameter`; `MW_V`, `L` and `MW_L`
  are required keyword arguments and `sigma` is required (no `0.02` default).
  Results change accordingly (they now respond to the molecular weight, the
  surface tension and the spacing).

- **Onda mass-transfer values corrected; `difflow_cc` functions deprecated
  (#427).** `difflow_cc.kinetics.liquid_film_coefficient` used
  `(rho g/mu)^(1/3)` where Onda (1968) has `(mu g/rho)^(1/3)` (about 4600x too
  large: 0.487 m/s instead of order 1e-4 m/s) and a Reynolds number on the dry
  area instead of the wetted area; `gas_film_coefficient` was too large by
  `a_p d_p` and never used the 2.00 constant for packings under 15 mm.
  `gas_film_coefficient`, `liquid_film_coefficient`, `interfacial_area`,
  `height_of_transfer_unit` and `number_of_transfer_units` are now
  `DeprecationWarning` wrappers on `difflow.mass_transfer` and return the
  corrected values (`interfacial_area` is numerically unchanged apart from
  g = 9.80665). No unit or cc result depended on the old values (the
  absorber uses Kremser); the cc tests only checked positivity.
- **`minimum_reflux_ratio` honours `q`; `stages_kremser` fixed (#424).**
  `difflow.units.distillation.minimum_reflux_ratio` accepted `q` and ignored
  it (every feed was a saturated liquid) and floored the result at 0.1. It now
  returns the McCabe-Thiele pinch of a constant-alpha curve at that `q`
  (`difflow.staged.minimum_reflux`) and is not floored; a vapor feed returns
  a larger value than before, a subcooled one a smaller. `q = 1` agrees with
  the old closed form except where that was floored. `ShortcutColumn` was not
  affected. `difflow_ree.stages_kremser` computed `ln((R(E-1)+1)/E)/ln E`,
  negative for every `E > 1`, so its floor returned 1.0 for every input
  (E = 1.5, 2, 5 at 99 % recovery now give 8.70, 5.66, 2.72); it is now a
  wrapper on `difflow.staged.kremser_stages` (still floored at one stage;
  the core function is not).

- **Cubic EOS liquid root near 1 bar; results change (#417).**
  `PengRobinson`/`SRK` clamped every root at Z >= 0.01, which replaced the
  liquid root of water, benzene, heptane, pentane... near atmospheric
  pressure (true Z ~ 0.001-0.007). Liquid density, liquid fugacity, the
  `flash_TP_eos` phase split (cold liquid water at 3 bar flashed to vapor)
  and `CubicThermo.stream_enthalpy_flash` were wrong there. The bound is now
  the co-volume, Z > B. Cases above ~5 bar or near critical are unchanged.
- **Heat-exchanger formulas corrected; results change (#418).**
  `lmtd_correction_factor` used the R = 1 limit for every R (F = 0.969 for
  R = 2, P = 0.3; exact 0.883) and the multi-shell P conversion failed at
  R = 1; `effectiveness_crossflow_both_unmixed` returned the counter-current
  value; the Cmax/Cmin-mixed cross-flow forms blended toward NTU/(1+NTU)
  near Cr = 1 (15% high). `ShellAndTubeHX` now rates with the exact 1-2N
  effectiveness (new `effectiveness_shell_and_tube`) and reports
  F = Q/(UA LMTD_cc). `EnthalpyCounterCurrentHX` solves Q by bisection on
  [0, Q_max] (the damped fixed point diverged at high NTU; `damping` is
  now unused). Past the per-shell limit P_max = 2/(R + 1 + sqrt(R^2 + 1)),
  a temperature cross too large for the shell count, `lmtd_correction_factor`
  returns F = 0 (the infinite-area limit). Its log clamps used to return a
  plausible-looking value (F = 0.11 at R = 1.125, P = 0.615).
- **Equipment cost tables re-sourced; capital costs change (#420).** Every
  entry in `difflow.economics.capital` carried a "2019" label and `a + b S^n`
  coefficients with no traceable source. Below 100 m2 they were 2-2.5x below
  Turton's heat-exchanger curve, and 6.6x below it at 1000 m2. Each entry now
  holds the published correlation on its own basis year and names its
  `source`:
  - Turton 5e Table A.1, `log10 C = K1 + K2 log10 S + K3 (log10 S)^2`, 2001;
  - Towler & Sinnott Table 7.2, `a + b S^n`, 2010;
  - Seider, 2013.

  `CostParams` gains `form`, `source`, `base_cost()` and `exponent()`.
  `validate_cost_params` checks the local exponent, and there is a new
  differentiable `equipment_cost_log_quadratic`, which `tea()` writes for
  Turton entries. Size bases changed:
  - compressors are sized on driver power (Towler) or fluid power (screw,
    Turton);
  - `centrifuge_disk` is sized on bowl diameter in m;
  - `packing_structured` is Seider's installed cost.

  `cooling_water_cost` documents that `delta_T` does not affect a per-GJ
  price.

- **Core database heat capacities moved; results change (#393).** Every
  species now carries an explicit ideal-gas Cp cubic (`SpeciesData
  .Cp_vapor_coeffs`; Poling, Prausnitz & O'Connell 5th ed. Cp/R quartic,
  cubic-fitted over 250-800 K, within 1.2%; styrene by Joback) in addition
  to the liquid Cp (`Cp_coeffs`, where a liquid exists at ambient).
  `CubicThermo` and the entropy integral `IdealThermo.S_ig_T` read the
  ideal-gas field (new `IdealThermo.Cp_ig`, `Cp_mix_ig`, `H_ig`,
  `stream_enthalpy_ig`); `IdealThermo` keeps integrating the liquid one. This
  moves numbers: a `CubicThermo` flowsheet with water or an alcohol used a
  liquid Cp (water 75.3, methanol 81.0) as its ideal-gas Cp (about 33.6 and
  44), now fixed, and the 24 species that were still constants (n-heptane,
  n-octane, benzene, toluene, ...) are temperature dependent, so n-heptane at
  600 K gains about 40%. On `IdealThermo`, n-heptane and n-octane liquid Cp
  are now 225 and 254 (they held ideal-gas 166 and 189), and the species
  #172 gave ideal-gas cubics (C1-C6, N2, CO2, ...) keep that cubic for
  liquid enthalpy too (no ambient liquid exists). Data that stores the
  ideal-gas Cp in `Cp_coeffs` with no `Cp_vapor_coeffs` (the Cantera, DWSIM
  and PyGLenN importers) is unchanged. Downstream models with IDAES or other
  reference data on these species need that data regenerated.

- `REEDistribution(no_data=...)`, threaded as `no_data=` through the REE unit
  and circuit params, says so when there is no data (#384): `"nan"` answers
  NaN for an element with no coefficients (Ho to Lu on D2EHPA, #269) and for
  the elements a record lists in the new `unmeasured_outside_window` (D2EHPA:
  Gd, Tb, Dy, Y) below the fitted pH window, which is where heavy-REE
  stripping has to run; `"raise"` makes the latter an error. A NaN is not
  filtered by anything downstream, so a calculation that uses it returns NaN.
  The default `"warn"` is the previous behaviour.
- `Saponifier` reads a solvent stream's extractant on the record's own basis
  (dimers for D2EHPA, PC88A and Cyanex 272) and converts with
  `Extractant.monomers_per_basis_unit`, so it counts the same exchangeable
  equivalents as the extractor charge (it counted half on a dimeric extractant
  fed from an extract-scrub-strip module; #386). `Saponifier.to_formal_basis`
  hands its output to the formal-monomer `SaponifiedSection`.

- REE strip acid floor (stopgap for #384). A strip pH the code chooses from
  the D curves is floored at `-log10(max_strip_acid)`, default 6 M (pH
  -0.78), on `ExtractStripParams`, `ExtractScrubStripParams`,
  `GroupSeparator`, `SeparationTrainParams`, `design_extract_strip` and
  `design_extract_scrub_strip`. The cut put heavy-REE stripping on D2EHPA at
  pH -1.17 (14.6 M H+), beyond any real strip liquor and 1.2 pH units below
  the fitted window. At the floor the strip leaves REE on the solvent, now
  reported as `results["strip_retained"]` and the train's solvent holdup
  (default D2EHPA Gd/Tb/Dy/Y circuit: Y recovery 97.5 % -> 63.6 %, 34.7 % of
  the Y entering the strip stays on the barren organic; default train
  recovery 1.000 -> 0.972); the design helpers add strip stages at the floor
  (Nd/Dy: 2 -> 4) and raise `StripAcidLimitWarning` when no stage count meets
  the target. `StripperParams` refuses an `acid_conc` above
  `max_strip_acid` when it sets the pH. A pH you pass is never clamped;
  `max_strip_acid=None` restores the unbounded cut.
- REE extractant basis (closes #374, which kept the dimer basis): every
  extractant concentration is documented on the record's own basis (dimer
  for D2EHPA, PC88A, Cyanex272: 0.5 M dimer = 1.0 M formal; the molecule for
  TBP and naphthenic acid), and the boundary to the mass-action layer (formal
  monomer) now converts with the new `Extractant.monomers_per_basis_unit`.
  `log_K_from_correlation` / `MassActionSection` evaluated the correlation at
  the monomer number as if it were dimer (D2EHPA's K 2**2.38 too large,
  Cyanex272's 8x), and `REEExtractor(model="mass_action")` passed its
  record-basis `extractant_conc` and solvent stream unconverted (the closure
  saw half the extractant: twice the loading, half the free extractant).
  Mass-action numbers calibrated from the correlation move accordingly.

- Open-issue sweep (#373, #374, #376, #377, #378, #379, #380, #381). REE: the
  organic phase flow is the diluent volume alone (the extractant entry is a
  moles-per-volume charge), so a stated O/A of 1 runs at 1 and not 1 +
  `extractant_conc`; the loading capacity is counted in the concentration
  basis (`Extractant.basis_units_per_ree`, 3 dimers per REE for D2EHPA, PC88A
  and Cyanex 272), doubling it; `ExtractScrubStripParams.recycle_scrub_liquor`
  closes the scrub-liquor loop with a tear, and `GroupSeparator` /
  `FullSeparationTrain` recycle by default. Carbon capture: `AmineStripper`
  models the overhead condenser (`steam_ratio`, `T_condenser`; `reflux_ratio`
  is deprecated and ignored), and the amine and adsorbent degradation defaults
  are recalibrated. Pinned numbers moved with these. Also: `PlatformDSPParams.
  uf_concentration_factor`, `RadialFeederFlowsheet` refuses downstream PV
  buses, DIP feed hydrogen bypasses the column, SEC bypasses unloaded
  material to the product, the REE palette wrappers run under `jit`/`grad`,
  `tests/conftest.py` imports difflow (xdist deadlock), and
  `tests/test_doc_examples.py` runs every documentation example.

- Flowsheets read a unit's result with one shared reader that tells a Stream
  from an info dict by its keys and understands `((S, S), info)`; a unit that
  returns a different number of streams than it has outlets now raises
  instead of dropping the extras. The catalog counts streams inside a nested
  leading tuple, so Ultrafiltration, Diafiltration, the three chromatography
  units, PowerSplit and BranchFlow have their real outlet counts (was 0).
  `FedBatchReactor`/`SemiBatchReactor` no longer list `C0` as an inlet.
- REE circuits pick any section pH left unset from the extractant's own D
  curves (extraction, scrub and strip each at their cut for the circuit's
  elements and targets), replacing fixed fractions of the fitted window that
  no longer separated after the #270 refit; the design helpers size stages
  from the actual D. `REEStripper`'s `acid_conc` sets the strip pH; TBP
  circuits strip at their own `strip_nitrate_conc`; FullSeparationTrain runs
  its cerium oxidizer at the oxidizer's temperature (Ce removal 8% -> 81%).
  The ExtractStrip, ExtractScrubStrip, SplitShell and SeparationTrain palette
  operations return streams. Precipitators are limited by the reagent fed and
  pass non-REE species through.
- Carbon capture: `AmineAbsorber` capture is bounded by the absorption factor
  (A < 1 gave near-total capture) and uses `solvent_in` when given; the rich
  solvent carries total CO2. `MembraneSeparator` solves the per-species flux
  balance and `feed_pressure` defaults to the stream's; `CompressionTrain`
  compresses from the stream pressure (`P_inlet` defaults to None).
  `SolidSorbentDAC` capture is bounded by the air fed.
- `IsomerizationUnit` refuses feed species it does not model and keeps feed
  H2 above its target; `CrudeUnitWithPreheat` adds the inlet stream's water to
  the tank water; `build_ladder_flowsheet` carries root loads, shunts,
  generators and reversed taps.

- `Ultrafiltration`/`Diafiltration` set each species' rejection from the MWCO
  and its molecular weight (new `difflow_bio.database.BIO_SPECIES_MW_KDA`,
  `molecular_weights` parameter); with no `rejection` given they passed a
  150 kDa mAb freely. `rejection_from_mw` puts R = 0.9 at the cutoff (was
  0.5). `Diafiltration`'s buffer stream is the buffer consumed when
  `n_diavolumes` is omitted (now optional); with it given, the scaled buffer
  is `info['buffer_consumed']`. `TFF.uf_df_uf` returns
  `((product, perm1, perm2, perm3), info)`.
- Chromatography: `load_volume` needs `feed_volume` (it was divided by the
  stream amount), and `info['yield']` is product out over product in.
  `IonExchangeChromatography` uses `column_volume`, `q_max` and `K_d` as a
  binding capacity; Protein A uses `K_d` when the feed concentration is known.
- `mAbDSPTrain`, `PlatformDSP` and `ViralClearanceTrain` return every outlet
  in `side_streams`, so product + side streams = feed. `mAbDSPTrain` step
  yields are against each step's own inlet (adds `tff1`, `tff2`) and its
  final TFF concentration factor is a volume ratio. `PlatformDSP` rejects
  `capture_type="mmc"` and polish steps other than cex/aex, and
  `include_viral_filtration` adds a virus filtration step (default on, 0.97
  recovery). `ViralClearanceTrain` step methods return `((kept, lost), info)`
  and low-pH recovery no longer exceeds 0.98.

### Added

- **`difflow.staged`: McCabe-Thiele stepper, pinches and Kremser (#424).**
  A library (not a palette unit) of graphical equilibrium-stage design.
  Operating lines as `(a, b, c)` (`operating_line`, `rectifying_line`,
  `stripping_line` incl. open steam, `stripping_line_from_boilup`,
  `q_line` for any `q` including 1, `feed_intersection`, `to_ratio`),
  `EquilibriumCurve` (`constant_alpha`, `linear_equilibrium`, `tabulated` with
  monotone PCHIP and an inverse, optional Murphree efficiency),
  `mccabe_thiele` (a `lax.scan` stepper with a fractional, differentiable
  stage count, optimal or specified feed stage, partial condenser),
  `minimum_reflux` (feed-line and tangent pinch, not floored),
  `total_reflux_stages` / `minimum_stages` (Fenske at constant alpha),
  `minimum_solvent_ratio` for a curved equilibrium line, and
  `kremser_fraction` / `kremser_stages` (any `A > 0`, smooth through
  `A = 1`; re-exported from `difflow.units.lle` and `difflow_ree`). Plot
  helper `difflow.visualization.plot_mccabe_thiele`. Verified against hand
  loops, closed forms, SciPy PCHIP and the Lewis relation; no published
  textbook example is reproduced (see the docs). Example notebook
  `examples/48_mccabe_thiele_staged.ipynb`.

- **Phase-diagram helpers and Wilson / Margules / van Laar activity models
  (#398).** `difflow.activity` defines the `ActivityModel` protocol (a
  `gamma(x, T)` method; `activity_gamma` dispatches) with `WilsonParams`,
  `MargulesParams` (two-/three-suffix) and `VanLaarParams`; `NRTLParams` and
  `UNIQUACParams` gained `gamma` adapters, and `Flash(activity_model=...)` now
  accepts any of them (NRTL results unchanged). `difflow.phase_diagrams` adds
  `txy`, `pxy`, `xy_curve`, `find_azeotrope` (masked, no Python branch on
  traced values) and `ternary_lle`; `difflow.visualization` adds `plot_txy`,
  `plot_pxy`, `plot_xy`, `plot_ternary`. Example notebook
  `examples/41_phase_diagrams.ipynb`. UNIFAC is a follow-up.
- **Centrifugal pump (#400).** New `CentrifugalPump` / `CentrifugalPumpParams`
  unit (`difflow.units.pump`): polynomial head curve, BEP-centred or
  polynomial efficiency, `NPSHr` curve, affinity-law scaling for operating
  speed `N` and impeller diameter `D` (Q ~ N D, H ~ N^2 D^2, P ~ N^3 D^3),
  outlet `P = P_in + rho g H(Q)`, `info` with head, efficiency, shaft and
  electric power, `NPSHr`, `NPSHa` and `npsh_margin`, optional temperature
  rise (`cp`) and `eo_residuals`. Helpers: `PumpCurve`, `fit_pump_curve`,
  `npsh_available`, `pumps_in_series` / `pumps_in_parallel` (combined curves),
  `system_curve(static_head, *pipes)` built from `Pipe` runs, and
  `operating_point(pump, system)` solving the pump/system intersection with
  implicit-function-theorem gradients (w.r.t. D, N, pipe diameter, ...).
  Liquid `rho` is an explicit input, as for `Pipe`. `Pipe` gained
  `hydraulics_at_flow(Q)`, and `evaluate_property` / `stream_mass_flow` helpers
  that the pump shares. GUI palette symbol, docs (`### CentrifugalPump`, with
  a cross-reference to the CO2 `difflow_cc.Pump`, which is intentionally left
  as a separate fixed-outlet-pressure model), tests and
  `examples/43_centrifugal_pump.ipynb` (operating point, affinity laws,
  series/parallel, NPSH, economic pipe diameter with a VFD-speed constraint).
  The vendor curve in tests and notebook is illustrative, not a published
  datasheet; the operating point is verified against an independent
  numpy/scipy calculation rather than a textbook figure.
- **Shell-and-tube design (#403).** New `difflow.shell_and_tube` (JAX, SI,
  differentiable; a library, not a palette unit): the Kern shell side
  (`equivalent_diameter`, `kern_shell_side`, `kern_shell_pressure_drop`),
  `tube_side_pressure_drop` (friction from `fluids.friction_factor` plus return
  losses), continuous `tube_count` and `bundle_diameter`, BWG / TEMA tables,
  and `ShellAndTubeDesign`, which iterates `U` (area from `Q/(U F LMTD)`,
  geometry, `heat_transfer.internal_h` and Kern films, `overall_U` with
  fouling) as an `optx.fixed_point`, so the area is differentiable in baffle
  spacing, tube velocity, passes, flows, fouling and properties. Returns the
  geometry, `U`, area, both `dP`, `F` and flags (`F < 0.75`, `dP` limits,
  convergence, laminar tubes), with a simple isothermal mode for a condensing
  or boiling side, `rate()` to re-rate a geometry and `round_design()` to
  turn the continuous tube count, passes and baffle spacing into a buildable
  one. Docs (`### ShellAndTubeDesign`), tests and example notebook 46
  (gradient-based cost minimisation; placeholder prices). **The Kern
  kerosene-crude textbook example is not reproduced** and the Kern constants,
  bundle-diameter constants and tube tables are recalled from memory and
  unverified; the docs and tests say what was and was not checked. The `U`
  fixed point is not unique with a fixed tube length (a second, large-area
  laminar solution exists); it is flagged.
- `effectiveness_shell_and_tube(NTU, Cr, n_shells)`: exact 1-2N effectiveness
  for one or several shells in series.
- **Heat-transfer coefficients (#402).** New `difflow.heat_transfer` (pure JAX
  functions, SI, differentiable; not a palette unit): `reynolds`, `prandtl`,
  `grashof`, `rayleigh`, `nusselt_to_h`; conduction (`slab_`/`cylinder_`/
  `sphere_resistance`, `composite_wall`, `critical_insulation_radius`,
  `fin_efficiency_straight`); tube-side `dittus_boelter`, `sieder_tate`,
  `sieder_tate_laminar`, `gnielinski` (takes the Darcy factor from
  `difflow.fluids.friction_factor`) and the dispatcher `internal_h` /
  `internal_nusselt` with a C2 laminar-transition-turbulent blend over Re
  2300-4000 and no Python branching on Re; `churchill_bernstein`,
  `flat_plate_nusselt`, Churchill-Chu natural convection; Nusselt film
  condensation (plate, tube, n-tube column), Rohsenow boiling (flux and closed
  form inverse), `mostinski`, Zuber `critical_heat_flux`; and
  `overall_U(h_i, h_o, D_i, D_o, k_wall, R_fi, R_fo, basis)`. Docs section
  "Heat-Transfer Coefficients" in `unit-operations-chemical.md`, tests in
  `tests/test_heat_transfer.py` and `examples/45_heat_transfer_coefficients.ipynb`
  (U to `CounterCurrentHX`, optimal tube velocity with `difflow.economics`;
  its prices are illustrative placeholders). Validation is against
  independent evaluations of the stated formulas and derivations (Nusselt's
  0.943, fin, cylinder resistance), **not** textbook worked-example numbers,
  which were not reproduced; the `FOULING_RESISTANCES` and `TYPICAL_U_RANGES`
  tables, the Rohsenow `C_sf` default and the Mostinski constants are from
  memory and flagged unverified. Not done: callable `U`/`UA` on the exchangers
  (compute `U` and pass the number), annular fins, tube banks, Kern /
  Bell-Delaware (#403).

- **Liquid pipe flow (#399).** New `difflow.fluids` (JAX, SI, differentiable):
  `reynolds_number`, `hydraulic_diameter`, the Darcy `friction_factor` by
  Colebrook-White (Newton solve, implicit-function-theorem gradients),
  Churchill, Haaland or Swamee-Jain with a C2 laminar blend (exactly 64/Re
  below Re 2100), `darcy_pressure_drop`, `minor_loss`, a `FITTINGS` table of
  Crane TP-410 K / L/D values with `fitting_K` and `equivalent_length`, and
  `orifice_flow`/`orifice_dp`/`venturi_flow`/`venturi_dp`. New `Pipe` /
  `PipeParams` unit (`L`, `D`, `roughness`, `dz`, `fittings`,
  `friction_method`; liquid `rho` and `mu` are explicit numbers or callables of
  the stream, molar mass via `MW` or a `thermo`) with `info` (`v`, `Re`, `f`,
  `dP_friction`, `dP_minor`, `dP_static`, `head_loss`), `eo_residuals`, GUI
  palette symbol, docs (`### Pipe`, friction methods, K table) and
  `examples/42_liquid_pipe_flow.ipynb` (Moody chart, economic diameter). The
  K-values were transcribed from the Crane/Perry's reprints, not checked against
  the TP-410 scan; the notebook's $/m piping cost is an illustrative
  placeholder, not a cited correlation.
- `difflow.particles` (#401): pure JAX functions for flow past particles and
  through beds. `drag_coefficient` (Haider-Levenspiel 1989, Turton-Levenspiel
  1986, Schiller-Naumann), `terminal_velocity` (implicit root find in ln Re,
  implicit differentiation), `hindered_settling_velocity` and
  `richardson_zaki_exponent`, `ergun_pressure_gradient` (viscous and inertial
  terms in `info`), `kozeny_carman`, `minimum_fluidization_velocity` (Ergun
  quadratic or Wen-Yu), `bed_expansion`, `fluidization_window`, and
  `geldart_group` (approximate boundaries; not differentiable). Verified
  against first principles and independent scipy solves, not against copied
  textbook tables (see `docs/unit-operations-chemical.md`). `GasPFRParams`
  also accepts a physical bed (`d_p`, `voidage`, `mu`, `rho_gas`, `u_s0`,
  `bed_area`, `sphericity`) from which alpha is computed by the Ergun
  equation (`effective_alpha`); the lumped `alpha` input is unchanged.
  Example notebook `examples/44_particles_and_beds.ipynb`.
- **Evaporators (#404).** `Evaporator` (single effect, rating by area or design
  by product concentration), `MultiEffectEvaporator` (forward, backward and
  parallel feed; equal-area design solved as one Newton system, or rating with
  given areas; per-effect `U`) and `MechanicalVaporRecompression` (ideal-gas
  isentropic compression, compressor work against steam saved), in
  `difflow/units/evaporator.py`. Each returns `(concentrate, vapor, info)` with
  steam rate, steam economy, per-effect T, P, BPR, vapor, area and Q, and
  closure residuals; `info` is differentiable with respect to `U`, the feed and
  the steam pressure (implicit gradients through `optimistix`). The
  non-volatile solute is an ordinary stream species whose vapor flow is exactly
  zero. `boiling_point_rise` offers Raoult (`ideal`), `colligative`, user
  Duhring lines, polynomial, callable and built-in `naoh`/`nacl`/`sucrose`
  models, plus `bpr_fn=` and `enthalpy_fn=` hooks (heat of dilution). Water
  saturation uses IAPWS-IF97; the enthalpy fits are to recalled steam-table
  points. **Not verified**: the built-in NaOH, NaCl and sucrose BPR data are
  approximate recollections (they raise `UnverifiedDataWarning`), and no
  textbook worked example is claimed to be reproduced. Registered in the
  palette under a new `evaporation` category with its own symbol (GUI bundle
  rebuilt), documented in `docs/unit-operations-chemical.md`, example notebook
  `examples/47_evaporators.ipynb` (optimal number of effects, with illustrative
  placeholder prices). No evaporator cost curve was added to
  `economics/capital.py` (no citable constants at hand); use
  `heat_exchanger_cost` as a stand-in.

- `difflow.solvers.pounce_problem` builds the configured pounce Problem once
  so repeated solves reuse the compiled residual and Jacobian (about 0.1 s
  against 30 s on a nested-EOS flowsheet, #394). `solve_with_pounce` is now a
  thin wrapper over it and still rebuilds on every call; its docstring and
  `docs/external-solvers.md` say so.
- `InstantaneousUnit` sees time and the integrator `args` (#396): an `fn`
  taking `(t, inputs, params)` or `(t, inputs, params, args)`, and
  `call_params` callables taking `(t, params)` or `(t, params, args)`, chosen
  by arity (the 2-argument `fn` and 1-argument callable forms are unchanged).
  `DynamicFlowsheet.simulate(args=...)` is documented as the one place for
  time-varying disturbances, read by feeds and units alike.

### Fixed

- `CubicThermo` treats a species absent from a stream as zero flow, as
  `IdealThermo` already did. A condenser whose vapour listed only n-pentane
  against a water-only coolant raised a bare `KeyError: 'water'` from
  `stream_enthalpy_flash`; a species the EOS does not model still raises.
- **NaN gradient through an accelerated recycle whose parameter hid in a
  closure.** `Flowsheet.solve` routes a traced solve to the optimistix fixed
  point, but `_is_traced` checked only the feeds. A tracer captured in a
  unit's closure, or passed through `Unit(params=...)`, therefore went to the
  Python Anderson/Wegstein loop. There the unrolled step is 0/0 under
  `jax.grad`, and the gradient came back NaN. `_is_traced` now also checks
  unit params. A loop that still finds a traced residual falls back to the
  traced fixed point instead of returning NaN.

- `lmtd_correction_factor` returned its `R = 1` value for every `R` beyond
  1 +/- 0.075 (the blend weight was a cubic that clipped to 0 instead of
  saturating at 1), e.g. F = 0.969 for both R = 0.5 and R = 2 at P = 0.3 where
  the Bowman formula gives 0.987 and 0.883 (#403).
- `ShellAndTubeHX` computed `Q = F * Q_counter-current(UA)`, which is not
  `Q = UA F LMTD` and under-predicted the duty by a few percent (3 % in the
  #403 test case). It now uses the exact 1-2N (and n-shell) effectiveness.
  Results of existing `ShellAndTubeHX` flowsheets change accordingly (#403).
- A thermo or EOS object built inside `jax.jit` no longer gets a value key
  (#395): its derived arrays are tracers, and a later equal-valued eager
  object hit a cache entry holding dead tracers (`UnexpectedTracerError`).
  It falls back to identity, a missed cache rather than a wrong hit.
- The missing-dependency message names the PyPI distribution only as
  different from the import name when it is (`asdex`, #394).

## [0.3.0] - 2026-10-06

### Breaking

- The operation registry refuses a second class under a name that is already
  registered (`ValueError`), as its docstring always said; it used to log a
  warning and overwrite. The refinery wet-gas compressor, which silently
  replaced the core `GasCompressor` whenever the refinery plugin was installed,
  is registered as `WetGasCompressor` (the class is still
  `difflow_refinery.GasCompressor`) (#359).
- `FedBatchBioreactor`: with `m_s > 0`, maintenance uptake now tapers as
  `S/(K_m + S)` near substrate exhaustion (new `FedBatchParams.K_m`, default
  0.05 g/L), so results with maintenance change slightly (about 5% at
  S = 1 g/L) (#362, #366).
- `mAbDSPTrain` and `PlatformDSP` load the whole batch onto each column, with
  the column volume acting as a capacity; they used to load one column volume
  of the harvest, which gave yields of a few percent. Purity is now realistic
  (99.4% for the mAb train, not 1.0000) (#360, #366).
- `difflow_bio.economics.estimate_total_opex` prices each chromatography step
  with its own resin (it priced all of them as Protein A), and it and
  `cost_per_gram` emit a `DeprecationWarning` pointing to `cogs_breakdown`
  (#358, #363).

### Added

- **`difflow mcp`**, an MCP server that lets an AI agent (Claude Code, Claude
  Desktop or any stdio MCP client) use difflow: install with
  `pip install "difflow[mcp]"` and register with
  `claude mcp add difflow -- difflow mcp`. The tools are plain Python in
  `difflow.agent` (`Workbench`), with a thin adapter in `difflow.mcp` over the
  MCP SDK 2.x. See `docs/agents.md` (#359, #364, #365, #367, #368, #369).
  - Discovery from the installed code, with no hand-written lists: operations
    from the registry and catalog, library APIs from each package's `__all__`,
    species, documentation search, examples and plugin status.
  - Building and solving: named sessions, units with parameters in one call,
    wiring (a loop closes as a recycle), feeds, every solver option, and solve
    results with the loop gain, error estimate and captured warnings.
  - `diagnose` and `converge`: findings with remedies, and a remedy search
    judged "converged AND correct" that applies only numerics remedies and only
    when asked, reports multiple steady states, and passes every case of the
    convergence corpus.
  - An expression language for objectives (`vapor.F_x / vapor.total_flow`),
    named quantities, and `sensitivity`, `sweep`, `optimize`, `uncertainty`,
    `linearize`, `tea` and `report`.
  - Each plugin contributes its own agent support through a `difflow.agent`
    entry point; `power_flow` and `power_opf` reproduce MATPOWER's benchmarks.
  - Progress notifications for long calls, three workflow prompts, and
    `open_in_editor`. `--no-exec` leaves out the tools that run Python.
- `Flowsheet.last_solve_history` (the tear residual per iteration on the
  Anderson and Wegstein paths) and `Flowsheet.last_solve_unit_info` (each
  unit's info dict from the last concrete evaluation) (#364).
- CSTR and Flash report `balance_residual` and `converged` for their own inner
  solves; the editor's solve audit flags a unit whose balance did not close
  (#364).
- `difflow.diagnostics`: symptom cards, a remedy table, a residual-history
  classifier and `solve_findings`, shared by the editor and the agent tools
  (#364).
- The catalog says what each outlet is (`PortSpec.outlet_roles`), read from the
  unit's docstring (#368), and units may declare a `cost_basis` that the
  catalog carries (#369).
- `difflow.plugins.plugin_status()` reports installed plugins and why any failed
  to load (#359).
- `difflow_bio.economics.cogs_breakdown`: cost of goods by category and step,
  with per-step resins, cycles from load / (DBC x CV), labor from batches and
  steps, buffers, media, QC, single-use items and failed batches, all from data
  (`load_cost_model`, YAML or JSON). Benchmarked against the published mAb
  process of Petrides (Intelligen, 2015, section 11.6.3) in
  `data/petrides2015_mab.yaml` (#358, #363).

### Changed

- The editor's session exposes every setting of `Flowsheet.solve` (not just
  four) and returns the gain, error estimate, clipping and warnings with a
  solve; its messages no longer assume a canvas. `difflow.gui` imports the HTTP
  server lazily (#359).

### Fixed

- diffrax integrations can be jitted and vmapped (`int()` on solver statistics)
  (#361, #366).
- Fed-batch substrate no longer goes negative under starvation; the diffrax and
  RK4 paths agree (#362, #366).
- `mAbDSPTrain`, `PlatformDSP` and `ViralClearanceTrain` are differentiable
  (they converted results with `float()`), and `PlatformDSP(include_sec=True)`
  no longer raises (#360, #366).

## [0.2.2] - 2026-10-05

### Fixed

- Gas plant columns: 0.2.1's retry of a non-converging first pass compiled
  that pass three times (a `lax.cond` per retry). Every jit or `jacfwd`
  through a gas plant column grew by about a quarter -- a `jacfwd` through
  the isomerization DIH went from 10.7 to 13.3 GB -- and the shard of the
  Publish gate holding the DIH tests ran its runner out of memory, so 0.2.1
  never reached PyPI. The retries are now one `while_loop` that traces the
  pass once; same refluxes, same answers.
- C5/C6 isomerization: one pass of the DIH recycle is jitted (it ran eagerly,
  op by op, on every Anderson iteration and every `jvp`), and its implicit
  derivative linearizes the pass once instead of compiling three `jvp`s. A
  recycle solve went from about 390 s to 17 s; a `jacfwd` through the unit
  from 213 s and 13.3 GB to 103 s and 7.1 GB.

## [0.2.1] - 2026-10-05

### Fixed

- Gas plant columns: the initial guess solved its component balances by an LU
  solve, which returns round-off for a heavy cut at the top of a long
  column (true ~e^-600). A 1e-12 change of a feed, or a different CPU's
  rounding, moved those log flows by hundreds and the deisobutanizer's first
  Newton pass could stall on a near-singular Jacobian (this failed the 0.2.0
  Publish gate, so 0.2.0 never reached PyPI). The guess now eliminates the
  M-matrix without pivoting in logs: every gas plant factory case converges in
  the same iterations under feed perturbations. A first pass that still does
  not converge is retried from guesses at half and twice the guessed reflux
  (the C3/C4 splitter on an FCC feed needs it).

## [0.2.0] - 2026-10-05

### Breaking

- Python 3.10 is no longer supported; difflow requires Python >= 3.11 (#261).

### Added

- **`difflow_refinery`**, a new plugin for petroleum refining: crude assay
  characterisation (Twu 1984), differentiable crude and vacuum distillation
  units, a crude preheat train with a desalter and preflash drum, a saturated gas
  plant, C5/C6 isomerization (once-through and with DIH/DIP recycle), a
  hydrotreater, a hydrocracker, an FCC, a reformer, alkylation, a residue
  desulfurizer, a hydrogen network and a product blending pool with nonlinear
  blending rules. The crude unit also works as a delta-base planning block. It is
  validated against independent IDAES, Pyomo/IPOPT and DWSIM 9.0.5 references,
  and there is a whole-refinery example (`examples/40`)
  (#296, #299, #300, #302, #303, #316, #317, #319-#322, #324, #335, #336, #345, #351).
- **`difflow_power`**, a new plugin for electrical grids: AC power flow, DC-OPF
  and AC-OPF (#205).
- **`difflow.planning`**: delta-base planning from differentiable flowsheets,
  with second-order subproblems, feasibility restoration, multi-period inventory,
  delta-vector health checks, delta attribution estimated from plant history,
  and Pyomo export (#184, #185, #187, #252, #293).
- **`difflow.stochastic`**: design under uncertainty by sample average
  approximation (#236).
- **`difflow.reconciliation`**: data reconciliation, monitoring over time,
  pooled parameter estimation, and gating and filtering of parameter updates
  from plant data (#175, #183, #244).
- **Flowsheets as data**: kinetics, a unit catalog, JSON serialization and code
  generation (#182). `Params` field descriptions now appear in the catalog
  schema (#218).
- **Local browser editor** (`difflow.gui`): build, wire, rename and edit
  flowsheets, open Python-script flowsheets, browse example flowsheets, and
  use a code editor, hover cards and per-unit documentation
  (#186, #216, #229, #230, #233, #240-#243, #337, #340-#344).
- **Recycle convergence**: automatic tear-stream selection
  (`fs.solve(tears="auto")`, `fs.tear_analysis()`), loop-gain and error
  estimates for the step tolerance (`TearToleranceWarning`,
  `tol_basis="error"`), a corpus of hard flowsheets for measuring the solver's
  pass rate, and a working `solve(damping=...)` (#258, #260, #262, #273, #274).
- The distillation columns now run on a cubic EOS (#214).
- Heater and Cooler now use a real enthalpy balance (#232).
- Gas network schematics and a worked model-updating example (#176).
- REE: a mass-action closure, saponification, a train graph and four analysis
  modules (#207).
- An "Ask" docs assistant in the Jupyter Book (#325), a refinery units summary
  page, and logos for the package and every plugin (#348, #350).

### Fixed

- Distillation: the feed-stage section convention is set in one place, the
  energy balance uses the feed's thermal condition q, and component balances
  close on the CMO path (#215, #217, #225). Fixed the Hengstebeck-Geddes non-key
  split in `ShortcutColumn` (#323).
- Flowsheet: a recycle starts from the stream being recycled, and a solve that
  did not converge now says so (#254, #256).
- CSTR: the molar density that sets residence time now has a name, and the
  water fallback raises `CSTRDensityWarning` (#234).
- REE: extraction loading, capacity, mechanism and activity models were
  corrected (#204). Extractant correlations were refit against named sources
  (#270, #282, #283, #290). The Kremser two-inlet boundary condition was fixed
  (#285). The capital estimate is now anchored to disclosed project costs
  (#272). Several smaller fixes (#275-#280, #287, #289).
- Solvers: sparsity is derived from the graph instead of defaulting to dense
  (#209).
- Fixed eight defects that adversarial testing found across numerics, streams,
  pytrees, serialization, economics, bio and carbon capture (#292).
- Fact-checked the example notebooks against the literature (#281, #291).

### Changed

- Physics validation (the `release` test marker) runs nightly and before every
  publish, instead of on every commit. CI shards the suite by measured test
  duration (#245, #304, #318, #347, #352, #353).
- The PyPI wheel check covers all seven packages.

## [0.1.0] - 2026-08-10

First public release.

[Unreleased]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/jkitchin/differentiable-flowsheets/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jkitchin/differentiable-flowsheets/releases/tag/v0.1.0
