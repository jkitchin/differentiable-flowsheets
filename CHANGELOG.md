# Changelog

All notable changes to difflow are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org) (pre-1.0: a minor bump may break the API).

## [Unreleased]

### Added

- **Unit-ops book Part VI gaps (issue 451).**
  *Costs.* `difflow.economics.dryer_cost` (rotary direct, drum and tray from Turton et al. Table A.1; spray from
  Towler & Sinnott Table 7.2), `crystallizer_cost` (forced-circulation and draft-tube-baffled from Seider et al.
  Table 16.32, in t/day; batch from Turton), `cooling_tower_cost` (Towler & Sinnott cooling tower and pumps, L/s),
  and `separator_cost(area, "filter_press")` plus leaf, cartridge, gravity, pan, table and tube filters (Turton).
  New tables `DRYER_COSTS`, `CRYSTALLIZER_COSTS`, `COOLING_TOWER_COSTS`. The constants were read from the
  openpytea 3.2.0 and ecoana 0.0.1 transcriptions of those tables (cited in the code), not the printed books.
  *Cooling tower.* `CoolingTower` fan power: `fill_pressure_drop`, or `fill_loss_coefficient` + `frontal_area`,
  plus `other_pressure_drop`, `fan_efficiency` (default 0.65) and `fan_location` add
  `info["fill_pressure_drop"]`, `["total_pressure_drop"]`, `["air_volumetric_flow"]` and `["fan_power"]` (W). No
  published fill correlation is built in yet; outputs are unchanged when the new parameters are not given.
  *Drying.* `DryerParams.T_heater_in` sets the heater-inlet air for the energy basis (the mixed air with
  recirculation), and `info["heater_duty"]`.
  *Crystallization.* Optional Arrhenius kinetics (`kinetics_Ea=[E_g, E_b]`, `kinetics_T_ref`;
  `arrhenius_factor`), so dL50/dT is no longer zero at fixed residence time. Optional Randolph-Larson R-z fines
  destruction and classified product removal (`fines_ratio`, `fines_cut_size`, `product_classification`,
  `product_cut_size`; `rz_distribution`, `rz_moments`): a piecewise-exponential population density, with the
  moments, closure and L_50 taken from it, and new info keys `M_product`, `product_moments`, `fines_dissolved`.
  *Mechanical.* `DiscStackParams(d_bins=, mass_fractions=)`, as for `Settler`/`Cyclone` (per-bin
  `grade_efficiency`, `overall_efficiency`).
  *Solvers.* `difflow.solve_scalar`: a differentiable scalar root finder (Newton with an optional
  bisection-safeguarded bracket) that works under `jit`, differentiates by the implicit function theorem with
  respect to `args` and closed-over values, and returns a converged flag with `return_info=True`.
  All new options default off.
- **Gaspype species data in `GibbsReactor` (#492).** New `difflow.gaspype_import`: `GaspypeThermo(species)`
  adapts [gaspype](https://www.gaspype.org/)'s ~1000 NASA-9 gas-phase species to `GibbsReactor`'s thermo
  interface, and `list_available_species` searches its database. Gaspype's own JAX equilibrium cannot be
  differentiated under `jax.jit` (a `while_loop`), but through the adapter the reactor's implicit gradients
  compile. `GaspypeThermo` is exported flat (gaspype is imported only when one is built), and a flowsheet
  using it saves, reloads and goes through `codegen`. Optional extra `difflow[gaspype]`; also in `dev`, so
  the tests run in CI. Example notebook `examples/65_gaspype_thermo.ipynb`.
- **The editor asks before a file runs code.** Opening a flowsheet ran its code context (and a `.py` script, and
  now modules named by import path) with no question asked. A file that would run code now opens nothing until
  trusted: the page shows the code and the modules and asks *Trust it and open?*; declining leaves no file open,
  so Save cannot write an empty canvas over it. `difflow gui --trust` skips the question; bundled examples, files
  saved in the session and code typed into the editor are never asked about. Python: `FlowsheetSession(...,
  trust=True)`, `open_file(path, trust=True)`, `needs_trust`/`blocked`, and `serialize.untrusted_modules`. Undo and
  redo now rebuild with the trust their model was opened with (they failed on a model naming a module of yours),
  and `Workbench(allow_exec=False)` refuses a JSON with a code context, as it did a `.py` file.
- **`serialize` covers every difflow model.** `save`/`load` now also take a `DynamicFlowsheet` (units as their
  class and constructor call, time-dependent feeds), a `PowerNetwork` and a `RadialFeederFlowsheet`; a
  `Flowsheet` subclass (`GasNetworkFlowsheet`) keeps its class. New: `CubicThermo`, `ActivityThermo` and
  dataclass thermos (`IdealGasThermo`, `GibbsThermo`); functions and classes by import path, difflow's
  (`kinetic_fn=monod_kinetics`, which left every bioreactor flowsheet unsaveable) and yours, the latter loaded
  only with `trust=True`; any object whose class keeps its constructor arguments by name (checked by
  rebuilding it on write) or records its call (`difflow.init_args.record_init_args`); saved results
  (`save(..., results=streams)`, `fs.saved_results`, `serialize.warm_start`); and single objects
  (`save_object`/`load_object`, `dumps_object`/`loads_object`, `to_value`/`from_value`). Format version 3,
  written only when a file needs it: a plain flowsheet is still version 2.
- **`serialize`/`codegen` write `PengRobinson` and `SRK`.** An equation of state is written as the critical
  properties and `k_ij` that build it (`$eos`; a `{(i, j): value}` dict as `[i, j, value]` triples), wherever it
  appears: `EOSFlash`'s required `eos` (such a flowsheet could not be saved at all), `Flash(..., eos=...)`
  (which warned and reloaded as an ideal flash), or a `Params` field such as `CSTRParams.eos`. `codegen` writes
  it as `get_critical_props` lookups when the properties match the database. A subclass is refused rather than
  rebuilt as its base.
- **Example notebook `examples/64_serialization.ipynb`.** Saving and reloading a recycle flowsheet with
  `difflow.serialize` (exact answer and gradient after reload), what the file holds, `refs=` and `extras=`,
  format versions, and export through `difflow.codegen`.
- **Core membrane separations (unit-ops book Ch. 22).** New `difflow.units.membranes`: `GasPermeator` (any
  number of species; complete mixing, cross flow, cocurrent or countercurrent; given area or stage cut;
  implicit-function gradients), `ReverseOsmosis` (plug-flow RO with solution-diffusion fluxes, film-theory
  polarization, van't Hoff osmotic pressure and the recovery limit), and library functions: the binary
  complete-mixing closed forms (Geankoplis 5e §13.4), `permeation_stage`, `area_for_stage_cut`,
  `osmotic_pressure`, `ro_local_flux`, `polarization_modulus`, `gel_polarization_flux`, `uf_flux`,
  `batch_concentration_time`, `BARRER`, `GPU`. The carbon-capture `MembraneSeparator`/`MultistageMembrane`
  use the core solver and gain `flow_pattern=` and `n_steps=`; bio `gel_layer_flux` uses the core gel model.
  The plug-flow models are tested against hand integrations in the feed-side mole fraction.
- **`difflow.adsorption` (unit-ops book Ch 23).** A core module for adsorption, ion exchange and chromatography,
  independent of the CO2-only cycle models in `difflow_cc` and the mAb column in `difflow_bio`: linear, Langmuir
  (affinity form) and Freundlich isotherms; `fit_isotherm` (linearized start, Levenberg-Marquardt polish);
  `batch_adsorption`; the length-of-unused-bed method (`lub_analysis`, `lub_break_time`, `stoichiometric_time`);
  `fixed_bed_breakthrough`, an isothermal LDF bed. A clean bed without dispersion is solved in corrected time
  theta = t - eps z/v_s, where the fluid hold-up drops out exactly, so a liquid bed that lasts months costs a few
  hundred steps; otherwise a method-of-lines form. Both use adaptive Tsit5 with the step capped inside the stability
  region so reverse-mode gradients are right (uncapped, the gradient of the stoichiometric time came out -1010
  against +50); `klinkenberg_breakthrough`; `ion_exchange_equilibrium` (mass action, any valences);
  and `retention_factor`, `plate_count`, `resolution`, `purnell_resolution`, `plates_for_resolution`, `van_deemter`,
  `van_deemter_optimum`, `gaussian_peak`. Tests reproduce Geankoplis 5e Examples 12.2-1 and 12.3-1 (data recalled
  from memory) and the Klinkenberg solution (`tests/test_adsorption.py`).
- **`create_objective` / `Flowsheet.make_objective_fn` forward solve keywords and reach `Unit.params`.** Extra
  keyword arguments go to `fs.solve` (e.g. `max_iter` for a slow loop under `jax.grad`, which takes plain
  substitution), and a `"unit.field"` key that names an entry of the unit's call-keyword dict (a splitter's
  `split_frac`, a reactor's `T_spec`) now sets it; before, only fields of the operation's `Params` were reachable
  and `split_frac` raised `AttributeError`. A field that is neither still raises.

- **Unit-ops book Part V gaps (issue 448).**
  *Extraction.* `CascadeParams.stage_efficiency` (Murphree, raffinate phase) now applies to counter-current
  (Lewis overall efficiency, exact for straight lines) and crossflow (stage by stage), not only co-current; its
  default is `None` (ideal stages; co-current keeps 0.8), and the rigorous model rejects anything but 1.
  `CascadeParams.profile_stages` tabulates profiles under `jit` over a traced `n_stages` (NaN past `n_stages`;
  without it a warning says the profiles hold only the inlet row). `hunter_nash` and `tie_line_through` are
  differentiable (roots replayed in `jnp` through the implicit function theorem; eager only, not under `jit`);
  `tie_line_through(..., on_outside="flag")` returns `two_phase=False` instead of raising.
  *Absorption.* `Absorber`/`Stripper` `model="stages"` gain an opt-in stage energy balance
  (`energy_balance`, `cp_gas`, `intercooler_stages`, `intercooler_T`); `Stripper` gains live steam that
  condenses (`steam_species`, `steam_latent_heat`; `steam_condensed`, `steam_duty` in `info`). CH4 in
  `HENRY_WATER` (Sander 2015). `wilke_chang_diffusivity` raises `WilkeChangAccuracyWarning` for small solutes
  (`V_A` < 40 cm^3/mol: NH3, CO2, O2, ...).
  *Column hardware.* `downcomer_backup` (Sinnott Sec. 11.13); `size_column(packed_height=...)`;
  `column_hardware_cost(material=, internals_material=, target_year=, structured_installation_factor=)` with
  `SHELL_/TRAY_/PACKING_MATERIAL_FACTORS` and `HARDWARE_COST_BASIS` (cost year and basis now stated; structured
  packing flagged as an installed price); `weep_point(K2=callable)`; `fractional_entrainment` raises
  `EntrainmentPlaceholderWarning` (still not Fair's chart); `packed_column_diameter` raises
  `PackingCapacityWarning` for a packing without a Robbins dry factor.
  *Carbon capture.* `AmineAbsorber` rich loading is limited by Kent-Eisenberg VLE at the rich end
  (`rich_loading_limit="equilibrium"`, the default; `"stoichiometric"` restores the 0.5 cap; `rich_approach`).
  This lowers default MEA capture (0.673 -> 0.626 at the defaults) and makes MDEA/AMP at lean loading 0.2
  capture nothing (they were already above their own VLE). New `solve_amine_loop` closes the absorber-stripper
  loop on the lean loading.
- **Unit-ops book Ch. 10-13 gaps (#444).** (1) `DistillationColumn.solve_specs(feed, specs)` solves for `R` and
  `D` from any two of `x_D`/`x_B` purities, `recovery_D`/`recovery_B`, `R`, `D`, `B` (damped Newton, implicit
  derivative; an infeasible pair ends with `spec_converged` False). (2) `underwood_class2`: Underwood's general
  method with distributing non-keys (one root per volatility gap, linear solve for `V_min` and their distillate
  flows). (3) `staged.mccabe_thiele(..., lines=[...], switch_x=[...] | switch_stages=[...])` steps any number of
  operating lines, for side draws, side feeds and several feeds; returns `switch_stage` and `N_sections`.
  (4) `EVAPORATOR_COSTS` / `evaporator_cost` (Turton Table A.1 evaporator rows). (5) `MVRParams.steam_T`.
  (6) The CMO path of `DistillationColumn` reports `L_profile`/`V_profile`. Tests in
  `tests/test_distillation.py`, `test_staged.py`, `test_economics.py`, `test_evaporator.py`.

- **Dryer: rotary sizing and textbook checks (unit-ops book Ch 19).** `rotary_dryer_ua` (MSH Ch. 24
  `Ua = 237 G^0.67 / D`, SI, recalled from memory) and `heat_transfer_units` (`N_t`); a continuous (countercurrent or
  cocurrent) `Dryer` with `diameter=` sizes the shell by the three-zone method (preheat, constant rate at the wet bulb,
  product heating: `zone_duties`, `zone_lengths`, `length`, `L_over_D`) and reports `N_t`. `T_solids_out=` specifies
  the product temperature instead of estimating it. `interpolate="inverse"` on `drying_rate_curve`/`drying_periods`/
  `drying_time` (and `DryerParams.rate_interpolation`) interpolates `1/R` linearly, so the falling-rate integral is the
  trapezoid rule on the graphical `1/R` plot. Tests reproduce Geankoplis Examples 9.6-1, 9.7-1, 9.7-2 and 9.10-1
  (inputs recalled from memory).
- **Unit-ops book Ch 20 (crystallization) gaps.** `Crystallizer(mode="adiabatic")`, a vacuum
  crystallizer: the evaporation is solved from the heat balance (feed sensible heat plus heat of
  crystallization, `duty=` for any jacket heat) in closed form, differentiable.
  `sieve_population_density` converts a screen analysis into the MSMPR population density for
  `fit_msmpr`. `TestPublishedExamples` reproduces Geankoplis Example 12.11-1 (Na2CO3.10H2O
  yield, with and without evaporation).
- **Published-example tests for cake filtration (book Ch. 21).** `TestGeankoplisFiltration` reproduces
  Geankoplis 5e Examples 14.2-1 (alpha and R_m from the CaCO3 t-V data with `fit_ruth`), 14.2-2 (the 20-frame
  press, 269.7 s) and 14.2-3 (wash at a quarter of the final rate, 194 s), also through the `CakeFilter` unit.

- **Unit-ops book Parts I-II gaps (issue 442).**
  *Fixes.* `phase_identification_parameter` uses closed-form derivatives of the cubic (cheaper to compile inside
  every TP flash); `eos_pressure_TV` gives the cubic's P(T, v). `eos_bubble_*`/`eos_dew_*` fall back to continuation (from 0.75 T or P/4) when no Wilson start converges
  (90-95% CH4 in C3H8 at 200-215 K), differentiate only the final Newton solve, and are compiled once per EOS.
  `CubicThermo.K_values_array`'s doctest runs. `lineax>=0.1.0` (0.0.8 broke implicit diffrax solvers under
  current jax: the stiff Robertson example). EOS-unit compile time roughly halved.
  *Fluids.* `PipeNetwork` (looped networks; Newton global-gradient or Hardy Cross), power-law and Bingham
  friction (`power_law_friction_factor`, `bingham_friction_factor`, ...), `PipeParams.profile` for crest
  cavitation, `pipe_cost_per_length`/`pipe_cost_nps`, `rotameter_flow`/`rotameter_height`.
  *Compressible.* `MultistageCompressor` (EOS stages, intercooler dP), `compressible.stage_pressures`;
  `difflow_gas` friction `"panhandle_a"`, `"panhandle_b"`, `"aga"`; `dense_gas_viscosity` (Jossi-Stiel-Thodos),
  `dense_gas_conductivity` (Stiel-Thodos), `mixture_viscosity` (Wilke, Herning-Zipperer),
  `mixture_conductivity` (Mason-Saxena), `gas_mixture_transport`.
  *Particles.* Mori-Wen and Darton bubble sizes, `bubbling_bed_expansion`, `geldart_group(method="geldart1973")`,
  `GasPFR` density from the inlet stream (`GasPFRDensityWarning`).
  *Heat transfer.* `jacketed_vessel_h`, `agitated_coil_h`, `economic_insulation_thickness` and the insulation
  cost functions, `TEMA_FOULING`, `tema_crude_oil_fouling`, `tema_water_fouling`.
  *Thermodynamics.* `difflow.binary_params` (`get_nrtl_params`, `get_uniquac_params`: DECHEMA/ChemSep sets, each
  checked against a known azeotrope or its absence); `IdealThermo.latent_heat` (the path-consistent `H_V - H_L`,
  authoritative for duties; Watson `Hvap` is the anchor); Antoine/Cp/Hvap for 16 critical-only species; Wagner
  vapour pressure for 12 light gases (opt-in: `get_species_data(name, vapor_pressure="wagner")`);
  `saturation_temperature` / `saturation_pressure` (ideal or EOS, differentiable).
- **Mass-transfer additions for the unit-ops book Ch. 14.** `difflow.mass_transfer` gains the original 1966
  Fuller diffusion volumes (`FULLER_VOLUMES_1966`, `FULLER_ATOMIC_VOLUMES_1966`, `diffusion_volume(...,
  table="1966")`) so Geankoplis/Treybal/Welty examples reproduce; Higbie and Danckwerts coefficients
  (`penetration_coefficient`, `surface_renewal_coefficient`); and the laminar falling film of a wetted-wall column
  (`falling_film_thickness`, `falling_film_liquid_coefficient`). Fuller and Wilke-Chang are now tested against
  published worked examples (Poling et al. 5e Ex. 11-3; Geankoplis Ex. 6.2-5 and 6.3-2) in
  `tests/test_mass_transfer.py::TestPublishedExamples`.
- **Packed `Absorber`/`Stripper` report the liquid-phase basis (book Ch. 15 gaps).** `info["N_OL"]` and
  `info["H_OL"]` (per solute, `N_OL = N_OG/A`, `H_OL = A H_OG`, `N_OL H_OL = Z`) join `N_OG`/`H_OG`, so a
  stripper can be read on the basis it is designed on (a TCE air stripper has `H_OG` of several metres and
  `N_OG < 1`, which is correct but not what a textbook shows). Tests check `N_OL` against Colburn's
  liquid-phase form and a packed acetone absorber against the log-mean hand method.
- **Crossflow LLE, tabulated ternary data (book Ch. 17).** `CascadeParams(flow_config="crossflow")` splits
  fresh solvent equally over the stages, `(X_N - X*)/(X0 - X*) = (1 + E/N)^-N`, smooth in `N`.
  `difflow.tabulated_lle(tie_lines, binodal=None, plait=None, T=None)` packs measured tie lines into the
  `ternary_lle` dict, so `hunter_nash`, `tie_line_through` and `plot_ternary` run on published data
  (tested on water-MIBK-acetic acid tie lines from NIST ThermoML). `Leacher` now reproduces
  McCabe-Smith-Harriott 7e Example 23.1 (variable underflow, four stages; data as recalled).
- **Robbins packed-bed pressure drop, Robbins/Kister-Gill flooding and liquid surface tension (book Ch. 16 gaps).**
  `column_design.robbins_pressure_drop` is Perry's 8e Eqs. (14-145)-(14-151); it reproduces Perry's Example 13
  (2-in metal Pall rings, 0.38 in H2O/ft vs the printed 0.38-0.40). `robbins_flood` solves for the gas flux at which
  Robbins reaches the Kister-Gill flood drop (an implicit-gradient bisection), and
  `packed_column_diameter(..., flood="robbins")` uses it for modern packings, where Perry's does not recommend the
  Eckert flood line. `DRY_PACKING_FACTORS` holds Perry's dry packing factors. `hetp("structured", C_XY=1.45)` adds
  Perry's X-type factor (Eq. 14-159). `liquid_properties.surface_tension` (IAPWS for water, Brock-Bird otherwise,
  `SurfaceTensionWarning` for alcohols and acids) and `mixture_surface_tension` supply the Fair chart's sigma.

- **`ActivityThermo` and an equilibrium reboiler in the McCabe-Thiele stepper (book Ch. 12 gaps).**
  (1) `difflow.ActivityThermo(ideal, activity_model)` gives modified-Raoult K-values
  `gamma_i(x, T) Psat_i / P` with the wrapped `IdealThermo`'s enthalpies, so `DistillationColumn` (CMO and MESH)
  runs on a nonideal liquid such as ethanol-water; before, it took only `IdealThermo` (Raoult) or `CubicThermo`.
  (2) `staged.mccabe_thiele(..., reboiler_equilibrium=True)` treats the last stage as an equilibrium stage when the
  curve carries a Murphree efficiency, matching `DistillationColumn` and the textbook construction; the default still
  applies `E` to every step. Tests `tests/test_activity_thermo.py`, `tests/test_staged.py`.

- **Multicomponent FUG functions (unit-ops book Ch. 13).** `fenske_stages_multicomponent` (key-ratio Fenske),
  `fenske_distribution` (non-key split at total reflux), `underwood_theta` / `underwood_minimum_reflux` (bracketed
  root between the keys, implicit gradient), `kirkbride_feed_stage` (continuous `N_R`, `N_S`), and
  `gilliland_stages(..., method="eduljee"|"molokanov")`. `ShortcutColumn` reports `info['N_rect']` and
  `info['N_strip']`. Checked against Geankoplis 5e Examples 11.7-2 and 11.7-3 in `tests/test_distillation.py`.

- **`phase_identification_parameter` (book Ch. 11 gap).** `difflow.eos.phase_identification_parameter(eos, T, P,
  y, phase)` is the Venkatarathnam-Oellrich parameter Pi (Pi > 1 liquid-like), from autodiff derivatives of the
  cubic's P(T, V). `flash_TP_eos` uses it to label a stable feed (see Fixed, #442). Test
  `tests/test_eos.py::TestPhaseIdentification`.

- **`flash_drum_size` (book Ch. 11 gap).** `difflow.units.column_design.flash_drum_size(Q_V, Q_L, rho_L, rho_V,
  ...)` sizes a vertical flash drum by Souders-Brown with Walas's rules of thumb (k = 0.107 m/s with a mesh pad,
  L/D = 3, 5 min holdup half full); the larger of the vapor and liquid diameters; differentiable. Test
  `tests/test_column_design.py::TestFlashDrumSize`.

- **`OperatingPointWarning`.** `operating_point` warns on a concrete negative `Q*` (no forward-flow operating
  point, e.g. a valveless parallel pair with a weak pump running backwards) or an unconverged Newton solve; silent
  under `jit`/`grad`. `PipeCavitationWarning` is exported from `difflow`. Test `tests/test_pump.py`.
- **Compressible flow, compressors, expanders, valves (book Ch. 5 gaps).**
  (1) EOS `Turboexpander`/`Compressor`/`JTValve` solve the outlet temperature with a bracketed (P, S)/(P, H) flash,
  also public as `flash_PS_eos`/`flash_PH_eos`. Two-phase, pure-component jump handled by the lever rule. Implicit
  gradients use the saturation condition inside a dome. The old unbracketed Newton gave methane 300 K, 100 -> 10 bar
  `T_isen = 261.5 K` (true 152.1 K) with negative work, and JT letdowns into the dome returned the 100 K bound, all
  unflagged. `info` gains `vapor_fraction`, `x`, `y`, `residual` and `converged`; an unbracketed solve warns
  `EOSUnitConvergenceWarning`. (2) `PHFlash` with a `CubicThermo` uses `flash_PH_eos`. (3) `difflow.compressible`:
  `Z` keyword on `speed_of_sound`, `mach_number`, `isothermal_pipe_mass_flux` and `isothermal_outlet_pressure`;
  new `nozzle_flow` (with `choked`), `isothermal_pipe_flow` (G and Colebrook f together), `isothermal_distance`,
  `isothermal_pressure_profile`, `isothermal_limit_mach`, `polytropic_exponent`,
  `isentropic_efficiency_from_polytropic`, `polytropic_efficiency_from_isentropic` and `optimal_stage_ratios`.
  (4) `difflow_gas.isothermal_pipe_beta` (alias `weymouth_beta`) is jnp and differentiable, with
  `friction="nikuradse" | "colebrook" | "weymouth"`. **It now returns a JAX scalar, not a Python float.** The
  docs say it is the isothermal pipe law without the acceleration term. (5) `equipment_cost` warns
  `difflow.economics.CostRangeWarning` for a concrete size outside the entry's stored `S_min`..`S_max`.
  Tests `tests/test_eos_units_two_phase.py`, `tests/test_compressible.py`, `tests/gas/test_physics.py`,
  `tests/test_economics.py`.
- **Chapter 1 flowsheet gaps.** `StoichiometricReactor` / `StoichiometricReactorParams(stoich, key, conversion,
  T_out=None, pressure_drop=0.0)` (fixed conversion of a key reactant, parallel reactions, no thermo;
  `difflow/units/stoichiometric.py`) and `ComponentSplitter` / `ComponentSplitterParams(split_fractions,
  default_fraction=0.0)` (ideal per-species split, returns `(first, second, info)`), both palette operations.
  `Flowsheet.degrees_of_freedom()` returns a `DegreesOfFreedom` (variables, specified feeds, unit and recycle
  equations, unit parameters, unknowns, `problems`; zero for a well-posed flowsheet).
- **Thermo gaps from the textbook's Ch. 2.** EOS saturation points
  `eos_bubble_P(eos, x, T, k_ij=None, P_guess=None)`, `eos_dew_P(eos, y, T, ...)`,
  `eos_bubble_T(eos, x, P, k_ij=None, T_guess=None)` and `eos_dew_T(eos, y, P, ...)`
  in `difflow.eos`. They solve phi-phi with Newton (optimistix, implicit
  differentiation) from Wilson K-values and safeguarded substitution, with four
  starts and the first converged non-trivial root kept. They return a dict with
  `T`, `P`, `x`, `y`, `K`, `converged` and `trivial`. Modified-Raoult
  `bubble_P`, `dew_P(thermo, species, y, T, activity_model=None)` and
  `dew_T(thermo, species, y, P, activity_model=None, T0=None)` join
  `bubble_T(..., activity_model=None, T0=None)` in `difflow.phase_diagrams`.
  New warnings: `AntoineRangeWarning` (concrete T outside a species' fitted
  range; once per species and side; silent under `jit`/`grad`; from
  `IdealThermo.Psat`, `bubble_T`, `txy`, `pxy` and the flash helpers) and
  `BubbleDewConvergenceWarning` (unconverged or trivial). Also added:
  `thermo.check_antoine_range`, `database.ANTOINE_SOURCES` (a source for every
  ideal-thermo species), `list_species(kind="ideal"|"critical"|"any")` and
  `IdealThermo.vaporization_T`. Tests: `tests/test_antoine_database.py`,
  `tests/test_bubble_dew.py`.

- **Shell-and-tube gaps (#435).** (1) `ShellAndTubeDesignParams.shell_h` callables now receive the wall state of the
  current `U` iterate (`q_flux`, `T_wall`, `dT_film`, `T_shell`, `T_tube`, `area_total`, besides the geometry keys),
  closed inside the existing `optimistix.fixed_point` (and on the heat flux in `rate()`), so condensing and boiling
  coefficients need no outer loop on the film dT; still differentiable. A callable that reads only the old keys is
  unchanged. (2) `internal_nusselt`/`internal_h(..., viscosity_correction=True)` and
  `ShellAndTubeDesignParams(tube_viscosity_correction=True)` multiply the Gnielinski / Dittus-Boelter value by
  `(mu_b/mu_w)^n`, `n = 0.11` heating, `0.25` cooling (`VISCOSITY_EXPONENT_HEATING/COOLING`). **The exponents are from
  memory (Petukhov / VDI form) and unverified**, so the factor is OFF by default; nothing changes unless it is asked
  for. (4) `difflow.fouling`: the Ebert-Panchal threshold model in core (`ebert_panchal_rate`, `wall_shear_stress`,
  `film_temperature`, `threshold_wall_shear`, `threshold_film_temperature`, `fouling_time`, `integrate_fouling`),
  user-supplied constants, differentiable, equal to the refinery plugin's `fouling_rate` for the same constants (that
  plugin is untouched). The deposition form and the 0.55 film weight are recalled, not re-checked against the paper;
  constants in docs and tests are illustrative. (6) `ShellAndTubeDesignParams.n_shell_passes` (shells in series, default
  1): `F = lmtd_correction_factor(R, P, N)`, per-shell geometry, total `area`/`UA`/pressure drops, `rate()` uses the
  N-shell effectiveness; a cross one shell cannot make (`F = 0`) is sizable with two. New result keys `n_shell_passes`,
  `area_per_shell`, `q_flux`. (7) Result flags `shell_laminar` (`Re_s < 2e3`), `shell_Re_out_of_range` (outside Kern's
  `2e3 < Re_s < 1e6`, constants `KERN_RE_MIN/MAX`) and `F_zero`, with warnings for concrete values only (silent under
  `jit`/`grad`); an `F = 0` or `F < 0.75` design now warns with `LowCorrectionFactorWarning` (a `UserWarning`).
  Docs (`### ShellAndTubeDesign`, `### Threshold Fouling`, heat-transfer section), tests
  `tests/test_shell_and_tube_gaps.py` and `tests/test_fouling.py`.
  **Not implemented:** (3) Palen's bundle correction for boiling CHF (no sourced form or constants; the limitation of
  the single-surface Zuber value is documented in `critical_heat_flux` and the docs). **Not verified:** (5) the `K1`/`n1`
  and CTP/CL constants stay unverified (an independent recollection of the triangular-pitch rows agrees with the code,
  which is memory against memory, not a check against Sinnott Table 12.4).
  **Numerical results:** unchanged for every existing call with defaults (`n_shell_passes=1`, no viscosity correction,
  geometry-only `shell_h`). Behaviour changes: the `F < 0.75` warning is now a `LowCorrectionFactorWarning` with new
  wording, and a design whose Kern `Re_s` is outside 2e3 to 1e6 now warns (the value itself is unchanged).

- **`Settler`, `CakeFilter` and `Cyclone` units and the mechanical-separation library (#434).**
  `difflow.units.mechanical` (one module, catalog category `mechanical`, GUI symbols `settler`, `cake_filter`,
  `cyclone`). *Settler*: methods `overflow_rate` (Hazen clarifier, capture `min(1, v_t/v_o)` per size bin with a smooth
  minimum), `coe_clevenger` (flux data `C_data`/`v_data`, or the particle model `terminal_velocity` x Richardson-Zaki
  `(1 - C/rho_p)^n`), `kynch` and `talmage_fitch` (a batch interface-height curve). Library: `solids_flux`,
  `batch_flux_curve`, `thickener_area_coe_clevenger`, `kynch_construction`, `thickener_area_kynch`,
  `thickener_area_talmage_fitch`, `lse_max` (the **smooth maximum**: log-sum-exp with temperature
  `mean|a|/sharpness`, never under-predicts, within `ln(N)/sharpness` of the grid maximum; default sharpness 1000),
  `overflow_rate_area`/`overflow_rate_capture`. The Richardson-Zaki exponent is piecewise in `Re_t`; pass
  `richardson_zaki_n` to avoid the jumps when optimizing over size. *CakeFilter*: Ruth constant pressure
  (`ruth_constants`, `ruth_constant_pressure`, cancellation-safe `filtrate_volume`, `filtrate_rate`), constant rate
  (`ruth_constant_rate`, `time_to_pressure`), `fit_ruth` (the Ruth plot: K_p, B, alpha, R_m), `compressible_cake`,
  `specific_cake_resistance` (Kozeny-Carman, consistent with `particles.kozeny_carman`),
  `cake_solids_per_filtrate`, `wash_time`, `cycle_throughput` (differentiable in `t_f`), `optimal_cycle` (closed form
  `K_p/2 (1 + 2r/phi) V^2 = t_down`, independent of B; a bracketed implicit-gradient solve for a user wash function),
  `filter_area_for_cycle`; the unit has `batch` (design from `cycle_time`, or rating with `filtration_time` /
  `optimize_cycle`) and `rotary` (drum, submergence fraction) modes and reports `area_for_cost`. *Cyclone*: Lapple
  `lapple_cut_size`, `lapple_efficiency`, `overall_efficiency` over fixed size bins, `lapple_turns`,
  `cyclone_geometry` (Lapple, Stairmand HE, custom ratios), Shepherd-Lapple `cyclone_pressure_drop`. **Solids
  convention:** `F_<solid_species>` in kg/s (`solid_molar_mass` 1 kg/mol), as the `Dryer` fines stream. **Dryer to
  Cyclone:** a `Dryer`'s fines are in `info["fines"]` (not in `air_out`); the documented hand-off is
  `Cyclone(...)(dust_laden_gas(air_out, info["fines"]))` in Python (a `Flowsheet` cannot route `info`), tested.
  Docs `## Mechanical Separations` (`### Settler`, `### CakeFilter`, `### Cyclone`, with a pointer to the `difflow_bio`
  centrifuges), notebook 58, `particles.py` docstring updated, tests. No existing behaviour changed (`difflow.numerics`
  already exports `smooth_max`/`smooth_min`, so the new log-sum-exp maximum is exported as `lse_max`).
  **Not implemented / skipped:** cost entries for cyclones and thickeners (no correlation could be cited with
  confidence, so none was added to `economics/capital.py`), a wash-water inlet (wash volume and time are reported),
  a wash or dewatering arc for the rotary drum, thickener compression and rakes, cyclone mass-loading effects,
  hydrocyclones, screening, precipitators and bag filters. **Unverified/not reproduced:** no Geankoplis,
  McCabe, Coulson & Richardson or Cooper & Alley worked example was reproduced (numbers "from recollection"), so the
  tests use independent oracles (`scipy` quadrature and scalar optimization, a NumPy grid maximum, the analytic
  Coe-Clevenger area of `v = v0 (C0/C)^2`, a batch curve generated from a known `v(C)` as the envelope of its
  tangents, hand calculations, `check_grads`); the Lapple constants (K = 16, the standard proportions, the Stairmand
  set) are typed from memory and flagged unverified; all data are synthetic and labelled illustrative.
- **`Crystallizer` unit and the crystallization library (#433).** `difflow.units.crystallizer` (catalog category
  `crystallization`, one category for the unit as `evaporation` and `drying` are for theirs; GUI symbol `crystallizer`):
  `Solubility` pytree (polynomial in degrees C, van 't Hoff `ln c = A + B/T`, or PCHIP table; kg solute per kg
  solvent, `per_100g=True` for g/100 g) carrying the hydrate form (`hydrate_waters`, molar masses) and an optional
  second solid form with `T_transition`, joined by a **smooth logistic blend** in `T` (never a Python `if`);
  `supersaturation` (ratio, absolute, relative); `crystal_yield` / `crystal_balance` (the two-balance hydrate yield
  `C = (S0 - c_sat W)/(f_s - c_sat f_w)`, evaporation allowed, soft-clipped at zero so undersaturated feeds give
  zero yield and a finite gradient); the MSMPR functions `msmpr_distribution`, `msmpr_moments`,
  `msmpr_dominant_size` (3 G tau), `msmpr_mass_median_size` (the gamma(4) half-point, computed by Newton, not
  typed), `msmpr_mass_cv`, `msmpr_from_magma`, `fit_msmpr`; `growth_rate`, `nucleation_rate` and
  `msmpr_steady_state`, which closes the power-law kinetics with the third moment by a bisection with
  implicit-gradient Newton step (`difflow.staged._solve_monotone`, `j <= 1`) so `L_D`, `L_50` are differentiable in
  tau, `k_g`, `k_b`. `Crystallizer(CrystallizerParams)(feed) -> (crystals, mother_liquor, info)`: modes `cooling`,
  `evaporative`, `cooling_evaporative`; equilibrium or kinetic mode; crystal stream convention **solute plus bound
  water** (`F_<solute>`, `F_<solvent> = nu F_<solute>`, mol/s, every species in every outlet); duty from sensible
  heat, `water_latent_heat` and a user heat of crystallization; takes an `Evaporator` concentrate (tested in a
  `Flowsheet`, with a mother-liquor recycle). Parameters are flat data (solubility model/coefficients/table), so the
  unit is declarative and buildable in the GUI. Docs `## Crystallization` (`### Crystallizer`, `### MSMPR Functions`),
  notebook 57, tests. **Not implemented:** an adiabatic vacuum-flash mode (evaporation is a specification),
  size-dependent growth, fines destruction, classified removal, agglomeration/breakage, polymorphs, batch cooling
  profiles. **Unverified/not reproduced:** no Geankoplis, McCabe or Wankat worked example was reproduced (numbers
  "from recollection" could not be recalled with confidence) and no sucrose, Na2SO4, Na2CO3 or MgSO4 solubility data,
  decahydrate transition temperature or kinetic constant is included; every curve in tests, docs and the notebook
  is synthetic and labelled illustrative. Tests use independent oracles (3x3 NumPy balance solve, `scipy` quadrature
  of the moments, argmax of `L^3 n`, a `brentq` root of the kinetic closure, `k_g -> infinity` equilibrium limit,
  `check_grads`).
- **`Dryer` unit and the drying library (#432).** `difflow.units.dryer` (category `drying`, GUI symbol `dryer`):
  moisture bookkeeping (`dry_basis`, `wet_basis`, `free_moisture`, `solids_moisture`), equilibrium-moisture
  isotherms `equilibrium_moisture` / `water_activity` (GAB, BET, PCHIP table or a callable; closed-form inverse for
  GAB/BET; optional Arrhenius `Hc`/`Hk` temperature dependence; parameters are data, none built in),
  `constant_drying_rate` (`h (T - T_wb)/lambda` or `k_W (W_s - W)`), `heat_transfer_coefficient_tray`,
  `drying_rate_curve`, and `drying_time` / `drying_periods` across the constant- and falling-rate periods
  (linear falling rate in closed form, or a tabulated rate curve integrated segment by segment). The
  constant/falling switch at `X_c` is a **softplus** of width `(X_c - X_eq)/sharpness` (default 200), so the time is
  differentiable in `X_c`, the endpoints, the rate and the loading with no Python `if`; `sharpness -> inf`
  recovers the piecewise answer. `Dryer` is a batch tray dryer (drying time from `load`/`area`, or the moisture
  reached in a given time, cycle time and tray area for a throughput) and a continuous countercurrent or
  cocurrent dryer (water and humid-enthalpy balances closing to rounding, exhaust dew-point margin and
  `supersaturated` flag, product-temperature margin against `max_solids_T`, energy per kg of water, thermal
  efficiency, residence time with `solids_per_area`, entrained `fines` split). Streams: air is the
  `difflow.psychrometrics` stream; solids carry `F_solid` (dry-solid pseudo-species, mol/s, numerically kg/s with
  the default `solid_molar_mass=1`) and `F_H2O` (mol/s). **Not implemented:** the three-zone McCabe rotary-dryer
  length, the Fick-diffusion falling-rate series, deep-bed humidity profiles. **Unverified:** no Geankoplis or
  McCabe drying example was reproduced (the issue's numbers are from recollection and could not be recalled with
  confidence), the tray correlations `h = 0.0204 G^0.8` / `1.17 G^0.37` are typed from memory, and no GAB data of a
  real material is included; the tests use closed-form, numerical-integral, balance and independent-NumPy/SciPy
  oracles instead. Docs: `## Drying` in `docs/unit-operations-chemical.md`; notebook `examples/56_drying.ipynb`.
  **Behaviour change (bug fix):** `psychrometrics.wet_bulb_temperature` and `adiabatic_saturation_temperature`
  started their Newton solve at the dry-bulb temperature, where `W_s` diverges at and is negative above the boiling
  point at `P`, so air above about 100 C at 1 atm (a dryer's inlet) gave NaN wet bulbs (and `state(...)["T_wb"]`);
  the seed is now capped at 350 K. Results below 350 K are unchanged; hot air now returns the correct wet bulb.
  `W_sat` and `mu` of a state above the boiling point remain meaningless (negative); `RH`, `T_dp` and `h` are fine.

- **`difflow.psychrometrics`, `difflow.water` and the `CoolingTower` unit (#431).**
  `difflow.psychrometrics` (library): ideal-gas humid air with the ASHRAE constants,
  `state(T=, W=, RH=, T_wb=, T_dp=, h=, P=)` from any two independent properties (closed form
  where one exists, one implicit Newton solve for `RH`-`T_wb`, `RH`-`h`, `T_wb`-`h`),
  `humidity_ratio`, `saturation_humidity_ratio`, `dew_point`, `humid_enthalpy`, `humid_heat`,
  `humid_volume`, `wet_bulb_temperature(lewis=)`, `adiabatic_saturation_temperature`, processes
  (`mix_air`, `sensible_heat`, `cool_dehumidify`, `adiabatic_humidify`), the flowsheet air stream
  convention (`H2O` + `air` species; `air_stream_from_state`, `state_from_air_stream`,
  `dry_air_flow`) and `psychrometric_chart_data` with
  `difflow.visualization.plot_psychrometric_chart`. Saturation is over liquid above and over ice
  below the triple point, with a C2 blend; no enhancement factor is applied (`W_s` is 0.3-0.5 %
  below the real-gas ASHRAE table). `CoolingTower` (category `heat_transfer`, GUI symbol
  `cooling_tower`): counterflow Merkel tower in rating and design modes, CTI four-point or
  Gauss-Legendre quadrature, fill characteristic, evaporation / drift / blowdown / make-up,
  pinch detection (`feasible=False`, `KaV_L = inf`, no NaNs; `maximum_L_over_G`), implicit
  gradients. The wex18-1 Merkel numbers of the issue (1.24, 1.39, 1.59, 1.89) are reproduced
  to 1 %; they are the issue's own illustration, not published values. **Unverified:** no
  ASHRAE table or textbook example could be transcribed; saturated-air values and the ice-line
  coefficients, the heat of fusion and the 2500.9 kJ/kg triple-point latent heat are recalled
  (see the test docstrings). Docs: `Humidification and Cooling Towers` in
  `docs/unit-operations-chemical.md`; example notebook
  `examples/55_psychrometrics_cooling_tower.ipynb`.
  **Behaviour change:** the water correlations (`water_saturation_pressure`,
  `water_saturation_temperature`, `water_latent_heat`, `water_liquid_enthalpy`,
  `water_vapor_enthalpy`) moved to the new `difflow.water` and are re-exported from
  `difflow.units.evaporator` (old imports unchanged, values unchanged at 20-200 C). Below 20 C
  `water_latent_heat` and `water_liquid_enthalpy` now include a smooth correction that restores
  the triple-point values (2500.9 kJ/kg and 0; the raw fit was 0.3 % low and -0.19 kJ/kg
  there); below 0 C they remain extrapolations.

- **Partially miscible extraction: `ternary_lle_flash` and `CascadeParams(model="rigorous")` (#430, part 3).**
  `difflow.units.lle.ternary_lle_flash(lle_model, z, T, guess=None, pair=(0, 1))`: a
  differentiable two-liquid-phase flash for any number of species (Rachford-Rice in the phase
  fraction around isoactivity K updates, three seeds, then damped Newton on the full system,
  all fixed iteration counts; implicit-function gradients in `z`, `T` and the NRTL/UNIQUAC
  parameters; a single-phase feed returns a masked result with `two_phase=False`).
  `CascadeParams` gains `model: "immiscible" | "rigorous"` (default `"immiscible"`, so existing
  code is unchanged; an unknown value now raises `ValueError` at construction). `"rigorous"`
  solves `n_stages` countercurrent ideal stages with the full equations (mutual solubility,
  variable phase flows) by damped Newton on the flat stage system, seeded from a Kremser
  profile, with implicit gradients. It supersedes `mutual_solubility` (giving both is an
  error), needs an integer `n_stages`, `flow_config="counter_current"`, an NRTL/UNIQUAC
  `equilibrium` and a positive total flow of every species of the activity model, and
  ignores `stage_efficiency`. Stages that mix into a single liquid (too little solvent) cannot
  be represented and are reported as `info["converged"] = False`. Checked against `ternary_lle`
  tie lines, the binodal on random feeds, the Kremser cascade in the immiscible dilute limit, a
  Gauss-Seidel stage-flash sweep, the Hunter-Nash stage count (agreement ~1e-4 stage) and
  `check_grads`. **No published acetic acid-water-MIBK data or NRTL set is used** (the
  recalled values could not be verified): the tests and docs use a synthetic type-I
  ternary. Docs: `Ternary Diagrams and the Lever Rule`, `Leaching and Washing` and the
  `MultistageCascade` rigorous section of `docs/unit-operations-chemical.md`; example
  notebook `examples/54_leaching_ternary.ipynb`.

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

### Changed

- **`serialize`: a `Flowsheet` subclass is no longer written as a plain `Flowsheet`.** It reloaded as the base
  class and lost any solve it overrides. It is now written by its class (loading one from outside difflow needs
  `trust=True`), and refused when defined in a script or notebook, where no import path finds it again.

- **The test suite checks the code, not the physics.** ~7300 tests (~106k lines) became 1700 (~23k): per unit a
  smoke test, a finite-gradient test, one test per validation path and the simplest form of each fixed crash/NaN
  regression, plus the API contracts. Comparisons to literature and reference data, rederived answers, trend and
  finite-difference checks are gone, and with them the `slow`/`release` markers, the nightly Full suite workflow
  and the 12-shard gate in `publish.yml` (one job now). CI runs the whole suite on every commit in two shards.
- **Per-commit tests check correctness, not expensive solves.** ~990 tests whose recorded cost was the computation
  (over 2-3 s; the 350 over 10 s were 85% of the ~490 runner-minutes) are marked `slow`, and the Tests workflow now
  runs `-m "not release and not slow"` on 2 shards instead of 4 (timeout 40 -> 25 min). The nightly Full suite still
  runs everything. `DIFFLOW_TEST_BUDGET` (set to 180 s in CI) fails a non-slow test whose call runs past it, and CI
  prints `--durations=15`.
- **Recycle tears no longer lock a pressurized loop at 1 atm.** `Flowsheet(default_T=None, default_P=None)` is the
  new default: a tear with no better guess starts at the highest pressure and flow-weighted temperature of the
  other inlets of the unit it feeds (feeds, or units upstream of every tear), with `stop_gradient`. Before, the
  101325 Pa start was carried round the loop by the Mixer's minimum-pressure rule and the solve converged at 1 atm
  with `dP/dP_feed = 0`. A number passed explicitly still pins the guess; inference survives `to_dict`/`from_dict`,
  codegen and `_apply_params` (`infer_tear_T` / `infer_tear_P`).
- **An unconverged recycle under `jax.grad`/`jit` warns.** The traced optimistix fixed point now carries a
  `jax.debug.callback` that issues `ConvergenceWarning` at run time when the loop stopped short of `tol` (its
  implicit gradient is then wrong). It stays a warning for `on_nonconvergence="raise"` (a traced solve never
  raises); `"ignore"` compiles no check.
- **`last_solve_gain` after Anderson.** When the post-solve probe finds the residual at round-off, the gain is the
  dominant eigenvalue of the tear map's Jacobian (`jax.jacfwd`, tears up to 400 variables; variables the loop only
  carries through unchanged are excluded) instead of `None`.
- **`Mixer` with thermo solves the enthalpy balance exactly** (Newton from the flow-weighted mean T, bracketed
  fallback, implicit gradient) instead of using one Cp at the mean T (752.7 K vs the exact 763.6 K for 1 mol/s CH4 at
  300 K + 10 mol/s at 800 K). A species missing from an inlet is now zero flow instead of a `KeyError`.

### Breaking

- **Antoine constants corrected and sourced; flash bubble/dew helpers honour
  the thermo model; `IdealThermo` vapour enthalpy made monotone.** Results that
  depend on these move, by design:
  - **Antoine constants.** 18 species had constants whose implied normal
    boiling point was 2.5-100 K off: O2, NH3, CO2, methane, ethane, propane,
    n-butane, ethylene, propylene, styrene, the propanols, 1-butanol, formic
    acid, diethyl and dimethyl ether, CHCl3 and CCl4. Five more were 0.2-1 K
    off: N2, n-pentane, isobutane, isopentane and acetic acid. They were
    replaced with NIST WebBook fits, or with the PPO 5e and Landolt IV/20 sets
    where no NIST fit works. Every species now has a `T_antoine_min/max` and a
    source in `ANTOINE_SOURCES`. Water keeps its constants, but its range is
    now 273.16-402.42 K, not the unsourced 273-473 K. Ethyl acetate's range
    runs to 493.15 K, on a 1% check against the McGarry Wagner equation.
  - **`list_species()`** now returns the species `get_species_data` accepts
    (`kind="ideal"`). Pass `kind="any"` for the old union with
    critical-only species. `agent.discovery`, `tests/refinery/test_fcc.py`,
    `tests/test_dwsim_import.py` and `tests/test_liquid_properties.py` changed
    only for this.
  - **`Flash.bubble_point_*` / `dew_point_*`** use the flash's `eos`
    (`eos_bubble_P` etc.) or `activity_model` (modified Raoult), where they
    used Raoult before. The silent clip to 150-700 K is removed: an
    unconverged result warns instead.
  - **`IdealThermo.H_pure(phase="vapor")`** used to be
    `int Cp_liq (Tref->T) + Watson Hvap(T)`, which falls towards Tc
    (ethane: 9.5 kJ/mol at 200 K, 1.9 at 305 K) and gave energy balances
    spurious roots. It is now `int Cp_liq (Tref->T_v) + Hvap(T_v) +
    int Cp_ig (T_v->T)` with `T_v = min(Tref, Antoine Tb)`, so
    `dH/dT = Cp_ig`. The docstring promised `Cp_vap`; the code now uses it.
    Liquid enthalpies are unchanged, and so is every vapour enthalpy at `T_v`.
    `tests/test_core_unit_bugs.py::test_issue_73` pinned the old path and was
    rewritten.
  - **Tests whose numbers moved:**
    - `test_heat_exchanger` (n-pentane boils at 321.2 K at 1.5 bar, not 320.x;
      the NIST reference EOS gives 321.24 K).
    - `test_planning` and `test_planning_diagram`, and section 9 of
      `examples/30_delta_base_planning.ipynb`. The NGL cold-box gas (75 %
      methane) bubbles at about 196 K at 40 bar (Peng-Robinson: 198.5 K), so it
      is two-phase across the whole 218-244 K envelope. The old methane
      constants put methane's Psat 2.3x too low and made a spurious bubble
      point near 224 K. The phase-crossing tests and the notebook now open the
      envelope to 185 K and start at 190 K. The reference chain run no longer
      crosses a phase boundary.
    - `test_distillation`. The ">10 K Raoult vs PR at the light end at 10 bar"
      claim came from propane and n-butane vapour pressures that were 45 % and
      30 % high. Raoult and PR now agree to 1.4 K (light end) and 0.7 K (heavy
      end); the test pins that and checks it against `eos_bubble_T`. The docs
      claim is corrected.
    - `test_tier3_enhancements` (the water range) and `test_database` (ethyl
      acetate's Tb is now inside its recorded range).

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

### Fixed

- **Drying-rate tables in R/R_c units were silently misread (#451).** `drying_rate_curve`, `drying_periods` and
  `drying_time` take `rate_basis="absolute"|"relative"` and raise `RateTableWarning` when a table's scale
  contradicts its basis. A relative (R/R_c) table passed to them gave a falling time about 1/R_c times too short
  (7 s instead of 4.06 h).
- **A reloaded `IdealThermo` never equalled the original.** JSON turned `SpeciesData`'s tuples into lists, so
  every object keyed on its construction inputs (the JAX caches included) took the reload for a new one. Fields
  declared as tuples are restored on read. Also: an `$eos` written by the previous release was read by version-2
  readers as a plain dict (files using it are now version 3); a dataclass written by bare name could rebuild as
  another package's class of the same name (core vs gas `PipeParams`) and is now written by path; `codegen`
  wrote edited `SpeciesData` with database names as plain database lookups, and NumPy floats as `np.float64(...)`;
  a traced value now fails with a message saying so instead of JAX's `UnexpectedTracerError`.

- **`codegen` scripts for plugin flowsheets did not run.** Every name was imported from `difflow`, so a script
  with a plugin unit failed on its first line (37 of the 60 units the editor can place, and all nine bundled
  examples). Names are now imported from the package that exports them, matched by class (the gas `Compressor`
  is not `difflow.Compressor`). Units that build their own `Params` (`GasPipe(beta)`) get keyword arguments
  rather than a wrapped `Params`; a `Params` is written as the instance's own class, not one found by name
  (`AmineAbsorber` was given `StripperParams`); and a scalar `inf` renders as `float('inf')`.

- **`serialize`/`codegen` dropped optional constructor arguments.** Only required ones were written, so
  `Mixer(species, thermo)` reloaded as `Mixer(species)` and mixed temperatures without an enthalpy balance:
  the documented isomerisation recycle reloaded 1.8e-3 K off in its reactor feed. An optional argument set
  away from its default is now written (`serialize.optional_extras`); one the format cannot hold raises
  `SerializationWarning` instead of vanishing. The editor's parameter edit (`gui.edit.rebuild`) had the same
  gap: changing `V` on an adiabatic `CSTR` rebuilt it isothermal and without its thermo. `Evaporator`,
  `MultiEffectEvaporator`, `MechanicalVaporRecompression` (`bpr_fn`, `enthalpy_fn`) and `MultistageMembrane`
  (`stage_params`) now keep those arguments under their own names, so they are written or warned about
  rather than dropped unseen.

- Reverse-mode gradients through a single-phase cubic-EOS flash were NaN (since the issue #444
  Newton polish): at the trivial K = 1 point the polish Jacobian is singular, and a zero
  cotangent pulled through `solve` gave 0 * inf. The polish now decides acceptance on
  stopped-gradient values and never differentiates a rejected solve. This broke, for example,
  `jax.grad` of a `JTValve` outlet temperature for a supercritical natural gas.
- The EOS-unit convergence warning formatted traced values with `float()` and raised
  `ConcretizationTypeError` under `jax.grad`; it now prints them only when concrete.

- `tests/test_docs_examples.py` (release) honours `<!-- doc-test: skip -->` like
  `test_doc_examples.py`, and the `underwood_class2` docs example imports `jnp`: the nightly
  Distillation docs examples ran a call fragment and a block without its import.

- **Bio `concentration_polarization` swapped R and 1-R (unit-ops book Ch. 22).** It returned
  `C_b exp(J/k)/(1-R+R exp(J/k))`, which gives `C_w = C_b` for a fully retained solute (R = 1), the case that
  polarizes most. It now returns `C_b exp(J/k)/(R+(1-R) exp(J/k))` (film theory; `C_b exp(J/k)` at R = 1) via
  the core `polarization_modulus`.
- **Protein A load volume used the resin's saturation capacity as its dynamic capacity.**
  `ProteinAChromatography.calculate_load_volume` took the dynamic binding capacity as `0.9 q_max` whatever the
  feed titre; it now uses the Langmuir loading at the feed concentration times `(1 - breakthrough_limit)`, as the
  column call does, and the unit's `equations` show the dissociation-constant isotherm `q_max C/(K_d + C)` it
  actually evaluates.
- **`grad` of a recycle solve after a jitted `grad` (`difflow.flowsheet.Unit`).** `Unit` was a dataclass with a
  value `__eq__` over `params`, which holds tracers when a design variable goes in through `Unit.params`. The
  Units ride in the `optx.fixed_point` args, so equinox's trace cache compared them and raised "Exception raised
  while checking equality of metadata fields ... _ClosureConvert": once `jit(grad(f))` had run, `grad(jit(f))`
  and `grad(f)` failed in the same process (this broke `sensitivity_analysis` on a jitted plant model). `Unit`
  now compares by identity (`@dataclass(eq=False)`); its hash is unchanged.

- **Bisection with a NaN bracket end (`difflow.staged._solve_monotone`).** When the residual was NaN at one end of
  the bracket (humid air below its dew point at the 200 K lower bound of the cocurrent `Dryer` outlet solve), the
  direction of monotonicity came out wrong, the bisection walked to the wrong end and the final Newton step gave a
  nearly right value with a wrong gradient: `d T_air_out / d G` was -0.012 against +3.14 K per kg/s by finite
  differences at 60 kg/s of 400 C air. The direction is now read from the finite end and a NaN midpoint counts as
  lying on the NaN end's side. Used by the dryer, leaching, crystallizer, batch-distillation and filtration solves.
- **`total_utility_cost` is differentiable.** It called `float()` and branched in Python on the duties, so
  `jax.grad`/`jit`/`vmap` over it raised; it is now pure JAX (negative duties still cost nothing).

- **Flash derivatives and near-boundary values (#444).** (1) `flash_TP_eos` second derivatives were wrong (the
  Hessian of a two-flash PR train had the wrong sign): derivatives came from unrolling 50 damped substitution steps.
  The substitution now runs under `stop_gradient`, Newton finishes the split, and two chord steps on `[ln K, V]`
  carry derivatives exact through second order; `_solve_rachford_rice` takes two Newton steps. (2) Near the
  critical region (0.8 CH4 / 0.2 C3H8, 250 K) the split stopped at 0.993 of the bubble pressure and V was off by
  60 %; a converged split that lowers the Gibbs energy now counts as two-phase. (3) `Flash` (Raoult / gamma-phi)
  returns the exact Rachford-Rice root: the `PHASE_TRANSITION_WIDTH` sigmoid biased V by 10-30 % within a few per
  cent of the bubble and dew points and did nothing outside them. (4) The ideal `PHFlash` took every inlet's
  enthalpy as liquid; it now flashes the inlet at its own (T, P) (`inlet_phase=` to name it). (5) The ethanol-water
  NRTL set in `tests/test_flash_azeotrope.py` and `tests/test_bubble_dew.py` was transposed relative to
  `NRTLParams`; a bubble-point test now pins the orientation.

- **`cool_dehumidify` no longer delivers supersaturated air (book Ch. 18).** With a cold coil and a small bypass
  factor the straight line to the apparatus dew point crosses the saturation curve: 30 C / 60 % air on a 2 C coil
  at BF = 0.1 left at RH = 1.034. The coil now delivers saturated air at the same total enthalpy and adds the fog
  to the condensate (`info["fog"]`); the duty is unchanged. `adiabatic_humidify` gains `RH_out=` and `mu_out=`
  (degree of saturation) targets; Geankoplis Ex. 9.3-1 to 9.3-3 and the cooling tower of Ex. 10.5-1 (z = 6.0 m
  against about 6.1 m printed) are now tests.
- **`Crystallizer` residence time from `volume`** now divides by the product slurry flow
  `(m_F - E)/rho_L` (the MSMPR definition), not the feed flow; an evaporative crystallizer that
  boils off half its feed water had half the true `tau`.
- **`Crystallizer` kinetic mode under `jax.jit`** raised `ConcretizationTypeError` (the
  `j <= 1` check called `float()` on `jnp.asarray(j)`, a tracer under jit); it now checks the raw
  value. Regression test `test_kinetic_mode_under_jit`. `CrystallizerParams` made the same
  check, so differentiating with respect to the kinetic constants (or passing them as a jnp
  array inside `jit`) failed; it now skips the check for traced values. Regression test
  `test_gradient_with_respect_to_kinetics` (checks the `j = 1` closed form
  `d ln L50/d ln k_b = -g/(b+3g)`).
- **Sigma theory in the bio centrifuges was off by factors of 2 and sqrt(pi) (book Ch. 21).** `Centrifuge`
  captured `min(1, 2 v_s Sigma/Q)`, twice the ideal-settler value, so a disc stack (whose Sigma is the standard
  Ambler form) over-predicted the recovery by up to 2x below complete capture. It now captures
  `min(1, v_s e Sigma/Q)`: complete at `Q = v_s e Sigma`, half at `Q_critical = 2 v_s e Sigma`
  (Geankoplis 5e Sec. 14.4C). `efficiency` is now Ambler's correction on Sigma (the machine is an ideal settler
  of area `efficiency * Sigma`); it used to multiply the capture, so a disc stack at the default 0.7 could never
  recover more than 70 % of the cells however slow the feed. `critical_particle_diameter` had a spurious pi in the denominator (cut size 1.77x
  too small): `d50 = sqrt(9 mu Q/(Delta rho g Sigma))`. `tubular_bowl_sigma` returned half the thin-layer value
  (`pi` for `2 pi omega^2 L r^2/g`); it is now the thin-layer value and takes `r_inner=` for the thick-layer
  form `omega^2 pi L (r2^2 - r1^2)/(2 g ln(2 r2/(r1 + r2)))`. The `equations` entry (a tanh that the code never
  used) and the bio docs now match the code. Tests: `TestSigmaTheory` (Hazen capture, cut size, the
  Stokes-trajectory oracle for a thick layer, a gradient in rpm).
- **Lapple cyclone outlet-duct length.** `CYCLONE_GEOMETRIES["lapple"]` had `S = D/8`; Lapple's standard is
  `S = 5D/8` (Cooper & Alley Table 4.1, recalled). `S` enters no calculation, so no result changes.
- **`SettlerParams` accepted unsorted or repeated settling data (book Ch. 21).** The PCHIP interpolation behind
  `coe_clevenger` needs strictly increasing `C_data`; a Kynch table passed in descending order, or with the
  repeated `C0` of the constant-rate points, silently gave a thickener area of 1442 m^2 instead of 63. It now
  raises `ValueError` for `C_data` (and for `t_data` in `kynch`/`talmage_fitch`) that is not strictly
  increasing. Concrete-only: traced values pass unchecked. Test:
  `TestSettlerUnit.test_unsorted_or_repeated_data_is_rejected`.

- **Le Bas oxygen increment for ethers (book Ch. 14).** `LE_BAS_INCREMENTS["O_ether"]` was 9.1, which is the
  methyl-ester value; it is now 9.9 (methyl ethers), and the table gains `O_methyl_ester` (9.1), `O_methyl_ether`
  (9.9), `O_ester_ether` (11.0, higher esters and ethers), `O_SPN` (8.3) and `Cl_medial` (24.6).
- **`Absorber` docs misattributed the geometric-mean absorption factor to Edmister.** The Kremser mode's
  `A = sqrt(A_top A_bottom)` is the textbook simplification (Geankoplis 5e Sec. 10.6, Wankat 4e Ch. 12), not
  Edmister's effective factor `A_e = sqrt(A_N (A_1 + 1) + 0.25) - 0.5`; the module docstring, the
  `variable_flow` field, the catalog assumptions and `docs/unit-operations-chemical.md` now say so. No
  numerical change.

- **`Absorber`/`Stripper` reported `L_min`/`G_min` equal to the given flow when a design was infeasible.** In
  design mode (`recovery=`) below the minimum rate the outlets sit at the A-limited removal, and the minimum was
  computed from those outlets instead of the specification, so `info["L_min"]` tracked the solvent rate given
  (e.g. 500 -> 500, 1000 -> 1000 mol/s for a 95 % SO2 scrubber whose true minimum is 1024). The minimum now uses
  the target end from `recovery`; feasible designs and rating mode are unchanged.
- **LLE cascade and rate-based contactor gave wrong answers (book Ch. 17 audit).**
  (1) `MultistageCascade` (counter-current) dropped any solute entering with the solvent: with 0.5 mol/s of
  solute in the solvent the extract reported only what came from the feed, so 0.5 mol/s vanished. It now uses
  the Kremser equation with `X* = Y_in/K` and closes the balance (extract = feed + solvent - raffinate).
  (2) Its stage profile used the n-stage Kremser fraction at every stage instead of the profile of one
  N-stage cascade; it is now `X_n = X* + (X0 - X*)(E^(N+1-n) - 1)/(E^(N+1) - 1)` and every stage balance
  closes. (3) `DifferentialContactor(mass_transfer_model="rate_based")` created solute (a 1 mol/s feed came
  out as 13.7 mol/s counter-current, 79.8 co-current): a softplus "clip" added ~0.07 to every ratio
  concentration, the counter-current organic equation had the co-current sign, and `Kla * area / F` was
  missing a molar density. It is rewritten as the exact matrix-exponential solution of the transfer-unit
  equations with `alpha = Kla * aqueous_molar_density * area` (new field, default 55500 mol/m^3), reports
  `HTU_OR`/`NTU_OR`, and matches the closed-form `N_OR` to 1e-8.
- **Packed-column pressure drop was about a third of the published value, and the Eckert flood line ran high.**
  `gpdc_pressure_drop` anchored an assumed power law to the Kister-Gill flood drop at the Eckert flood line and gave
  0.13 in H2O/ft for Perry's 8e Example 13 (printed 0.38-0.40); it now defaults to the Robbins correlation
  (`method="anchored"` keeps the old model). The Eckert flood-line fit `ln Y = -3.5021 - 1.028 ln X - 0.11093 ln^2 X`
  gave Y = 0.034 where Dutta's Example 5.2 reads 0.021 off the chart (26 % high on G); it is now the Kessler-Wankat
  fit `log10 Y = -1.6678 - 1.085 log10 X - 0.29655 log10^2 X` (0.024; Dutta's column comes out 0.87 m vs 0.90 m).
  `packed_column_diameter` reports the Robbins pressure drop.

- **`MechanicalVaporRecompression` compressor work was 6 % low.** The
  isentropic work used `Cp_vapor` (1880 J/kg/K) with the exponent of
  `gamma = 1.3`, a pair no ideal gas has. It now uses
  `cp = gamma R/(gamma - 1)`; compressing saturated steam from 100 C to
  200 kPa gives 127 kJ/kg against 126.2 kJ/kg from the steam tables
  (before: 119 kJ/kg).

- **Built-in NaOH boiling-point rise was 12 K low at evaporator vacuum.**
  The `naoh` Dühring lines were drawn from the 1 atm table with the
  colligative slope (`BPR ~ R T^2/lambda`), but concentrated NaOH lines are
  nearly parallel to the water line. At 11.7 kPa the 50 wt % solution boiled
  at 77 C instead of the chart's 89.5 C, so Geankoplis Ex. 8.4-3 came out
  with 33.0 m^2 instead of 49.2 m^2. The lines now use an empirical slope
  fitted to cited 50 % values (`DUHRING_SLOPE_COEFF`); the 1 atm 50 % point is
  141.7 C. Geankoplis Ex. 8.4-2/8.4-3 and McCabe Ex. 16.1 are reproduced
  (`TestPublishedExamples`, issue #421). The sucrose BPR is credited to
  Geankoplis Ex. 8.5-1 (not 8.4-1) and no longer warns as unverified; the
  former regression pin on those inputs is now a textbook-agreement test.

- **`BatchDistillation` took 10-30 s a call.** The body ran eagerly, dispatching every bisection of its nested root
  solves op by op. It is now jitted like `ShortcutColumn` (cached on the value key, the stop-criterion names and
  `n_traj`; the curve is an argument because it is not in the value key): about 1-3 s, same answers. `rayleigh`
  is checked against Geankoplis 5e Example 11.3-2 (`TestGeankoplis` in `tests/test_batch_distillation.py`).

- **`ShortcutColumn` used the pseudo-binary Fenske equation.** `N_min` took `1 - x_LK` for `x_HK`, so a light
  non-key in the distillate understated it (8.7 instead of 10.4 on a C2-nC5 depropanizer). It is now the
  key-ratio form. The Underwood root was a Newton step from `sqrt(alpha_LK)` clipped to `(1.001, alpha_LK-0.001)`;
  it left the interval and returned 1.001 in about a quarter of random feeds. It is now bracketed bisection.
  `R_min` is floored at 0 rather than 0.1.

- **Ideal-thermo `PHFlash` could not flash a pure liquid into its two-phase region.** The Newton on T cannot land
  on a boiling point, where the flashed enthalpy jumps by the latent heat; saturated water let down from 180 C,
  12 bar to 1.5 bar came back at the 150 K clip with `V_frac = 0`. The ideal path now brackets T by bisection
  on `T_bounds` over the exact (unsmoothed) Rachford-Rice split, applies the lever rule at a jump (111.2 C,
  12.6 % vapour; steam tables 111.4 C, 13.3 %), and attaches implicit-function gradients (the Clausius-Clapeyron
  slope at a jump). `info` gains `residual`; an unbracketed solve warns `EOSUnitConvergenceWarning`; `T_guess`
  is ignored. Test `tests/test_flash.py::TestPHFlashPureComponent`.

- **`flash_TP_eos` labelled a single-root compressed liquid as vapor (#442).** A stable feed was called vapor
  when its lower-Gibbs root was the "vapor" root; above the bubble point of a liquid the cubic has one real root,
  so both selectors return it and the label was wrong. CH4/C3H8 (z = 0.2/0.8, PR) at 277.6 K gave V = 1 for every
  P above its 33.4 bar bubble point, with V jumping from 0.04 at 30 bar to 1. The stable phase is now liquid when
  the phase identification parameter Pi > 1 and T is below the Kay's-rule pseudo-critical temperature, vapor
  otherwise; V now falls continuously to 0 at the bubble point. Affects `EOSFlash`, `Flash(eos=...)` and
  everything that calls `flash_TP_eos`. Test `tests/test_eos.py::TestPhaseIdentification`.

- **A named acceleration survives `jax.jit` / `jax.jacfwd` (#440).** `Flowsheet.solve(acceleration="anderson"|"wegstein")`
  used to be replaced by plain substitution whenever a feed or unit parameter was traced. It now runs
  `difflow.initialization.AcceleratedFixedPoint` inside `optimistix.fixed_point`: the eager Anderson/Wegstein arithmetic
  in `lax.while_loop`, fixed-size `(anderson_depth, n)` history buffers (one compile, no per-call retrace), the stop
  test `max|g(x) - x| < tol` as an array predicate, `clip_negative_flows` on the extrapolated guess only, and the same
  implicit derivative (no differentiation through the iterations). New `Flowsheet.last_solve_acceleration`
  (`"anderson"`, `"wegstein"`, `"none"`) records what ran; `last_solve_method` stays `"fixed_point (traced)"`.
  **Behaviour change:** only a NAMED acceleration takes the new path. The unnamed default is still plain substitution
  under tracing (`tests/test_flowsheet_damping.py` pins that, and changing every jitted default solve is a larger
  change); eager solves are untouched (`python -m difflow.convergence` output is byte-identical before and after:
  pass rate 46.5%, convergence rate 49.5%). Callers that pass `acceleration=` explicitly and are traced (the
  catalytic reformer, `difflow_ree` trains, the isomerization plant) now take the accelerated path; the reformer agrees with the
  old answers to every printed digit and the `tests/ree`, isomerization and reforming suites pass. `CatalyticReformer.solve` results and `ReformerPlanningModel` expose
  `solve_method` / `acceleration` (`last_solve_method` / `last_solve_acceleration`), so a traced plain-substitution
  fallback is visible. Docs: `docs/convergence.md` ("Naming an acceleration under jit and jacfwd"); tests
  `tests/test_traced_acceleration.py`.
  **Measured** on the catalytic reformer (lean naphtha, `reformer_block(..., ["wait"], ...)`, tol 1e-10), on a 4-core
  2.8 GHz Xeon, not the issue's M2 (baseline then after, seconds): eager solve at WAIT+5 C 11.3 then 10.6 (22
  iterations, unchanged); jitted evaluation 5 C from the stored tear **18.6 then 6.6** (63 then 23 passes); jit
  compile+first 51.6 then 45.5; jitted evaluation from its own tear 0.5 then 0.6; `jacfwd` compile+first 340.6 then
  322.0; `jacfwd` warm at +5 C **52.3 then 39.9**. Primal results and the Jacobian agree to every printed digit.
  **The Jacobian half (second commit).** The first measurement above left `jacfwd` at 40-52 s warm, and profiling
  showed why: not the iteration count but the dense tear Jacobian the implicit rule materialises (one forward-mode
  tangent per tear entry, 31 here), and a tangent through the three diffrax beds cost 1.5 s against 0.27 s for a pass,
  because every right-hand-side evaluation solved the bed temperature from the enthalpy with eight unrolled Newton
  steps and the tangent went through all eight. `reforming.thermo.temperature_from_enthalpy` is now a `custom_jvp`
  carrying the implicit-function derivative (primal unchanged; derivative matches the unrolled one to 5e-15 in
  forward and reverse mode). Measured on one tear map `g` (before then after, seconds): one tangent warm 1.53 then
  0.52; dense 31-tangent Jacobian warm 28.0 then 1.7; compile of that Jacobian 107 then 75.
  Reformer end to end (same machine): `jacfwd` warm at +5 C, 1 lever, **52.3 (original code) / 39.9 (traced Anderson
  only) then 12.3** (jit eval of the same point 7.4 in that run, so the Jacobian costs about 0.7 of a primal solve);
  `jacfwd` compile+first 340.6 / 322.0 then 250.9; 2 levers (WAIT, H2/HC; measured after only): jit eval 9.8,
  `jacfwd` warm 11.8, compile+first 235.5. Jacobians are identical to every printed digit. Tried and dropped: a
  matrix-free GMRES in the implicit rule (warm 51.9, compile 457.6; the tear Jacobian needs about 13 Krylov steps
  for 1e-10), and a `fori_loop` Newton (no compile gain, slower warm). **Still slow:** compile of the differentiated
  reformer is 4 minutes (primal map 28 s, tangent 60 s, vmapped Jacobian 75 s, plus the loop); a persistent JAX
  compilation cache (opt-in, config lines in `docs/convergence.md`; nothing is enabled by default) removes most of
  it for a repeat process. Measured, three separate processes with a warm cache against two cold ones, same machine
  (shared with another job, so noisy): `jit` compile+first 63.5 and 51.6 s cold, 6.0, 7.4 and 8.9 s cached; `jacfwd`
  compile+first 223.4 and 249.5 s cold (250.9 in an earlier uncached run), 56.0, 61.9 and 73.5 s cached; building the
  planning block 54.4 and 46.4 s cold, 21.5-24.1 s cached; warm evaluations and the Jacobian values unchanged; cache
  directory 14 MB. A different feed misses once (54.7 s against 57.1 s with no cache), then hits (14.6 s), and the
  cached program returns the eager answer to 1e-11. Looked at and not kept, because they do not pay: per-unit XLA compile
  of the primal map is 3 x 5.5 s for the identical beds and 12.3 s for the (core) compressor's EOS solves, out of 25 s of
  XLA; lowering to HLO and tracing are 7 s; `XLA_FLAGS=--xla_backend_optimization_level` 0 or 1 does not change compile
  time (31.5 and 30.5 s against 33.0 s); a `fori_loop` bed Newton cuts the primal compile 32 to 25 s but makes every pass
  about 20% slower.

- **`plan_sensitivity` reported half the elastic penalty as a shadow price.**
  The stationarity gradient that the multipliers are solved from included
  the elastic penalty of the active specs as well. When the plan meets a spec
  exactly, that penalty sits on the kink of `relu(v - rhs)`. JAX's
  subgradient there is 0.5, so `penalty/2` leaked into the multiplier.
  Notebook 30, section 8, reported 4999.64 $/h per K on `T_colfeed`,
  while a re-solve gave 0.35. The fix leaves the penalty of an active spec
  out of the gradient, because its multiplier already carries it. Now the
  multiplier is the constraint's shadow price (0.359 against a 0.1 K
  re-solve of 0.358). Regression tests are in `tests/test_planning.py`
  (`TestPlanSensitivity`).
- `eos_bubble_P`/`eos_dew_P`/`_T` no longer flag a pure component (or a mixture with one species present) as
  `trivial` and warn: K = 1 is the answer there. Test `tests/test_bubble_dew.py`.
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
