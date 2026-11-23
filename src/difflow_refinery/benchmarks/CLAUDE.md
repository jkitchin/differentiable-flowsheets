# difflow_refinery.benchmarks

Published refinery benchmarks, implemented against their own data. A library,
not a palette operation.

## `du2025` -- Du et al. (2025), arXiv:2503.22057 (#530, tracking #538)

- NEVER vendor the data (paper CC BY-NC-SA, data repo unlicensed). `fetch()`
  downloads to `$DIFFLOW_DU2025_DATA` (default `~/.cache/difflow/du2025`).
  The suite runs on the SYNTHETIC case `tests/refinery/du2025_mini/` (format
  only, none of the benchmark's numbers); the published optimum is checked by
  hand: `tests/reference/du2025_case1_check.py`.
- `build_model` is the paper's Section 3 as residuals. Every equation is a
  polynomial of degree <= 2 after clearing denominators; keep it that way
  (`PolySystem` COO: exact Jacobian, pattern without probing). A new term of
  higher degree belongs in a new variable + bilinear equation, as GAMS has it.
- Rules READ from the published solution's variables/bounds, not from the
  paper -- each one is needed for every equation to close at the published
  point (max residual 8.5e-8); do not "fix" them toward the paper's text:
  SPG <= 10; FIX without FQ0 -> 0; an untraced (s, q) an equation references
  is a FREE nonnegative variable (`model.untracked`; unphysical values at
  the optimum, e.g. SPG(s163) = 4.9e-7); Eq. 13a skipped when the mixer
  outlet's SPG is fixed; Eq. 18 / V only for blenders with an SPG spec.
- Case 1 only: one flow `F_s` = FVI = FVO (Eq. 21, no inventory). Features
  Case 1 leaves empty (inventory, periods > 1, CDUMQ, CRU, RB, Qv) RAISE --
  never ignore them silently.
- Swing-cut ratios `phi` are fixed parameters in `build_model`; `crude.cut_table`
  + `difflow_refinery.swing_cut` makes them decisions (#531). Q45 is dropped
  from the cut table: traced but never blended, its published values are free.
- The model is nonconvex (pooling): `solve_local` is a local solve. Never
  present it as the global optimum. From the published point use
  `WARM_START_OPTIONS` (mu_init 1e-6); the default barrier ends 68 lower.
- `blocks` = the same equations solved forward, pure JAX; the replacement
  interface for nonlinear units (#531, #533-#537). `test_blocks_satisfy_model`
  ties the two together.
- The unit identification table in the docs is "consistent with the data",
  not proven; keep the distinction.

### Property data (#532: `properties`, `assays`)

- `PROPERTY_MAP` confidences are evidence-based; never upgrade one without new
  evidence, and keep "unknown" indices listed. Q21 is CONRADSON CARBON (not
  olefins, the issue's guess); Q22 + Q23 = Q25 (Ni, V, Ni+V) on 324/325 cuts.
- The benchmark blends all but SG by MASS; difflow's blend pool by volume /
  RT-70. Mapping an index to a `blend_key` does not make the rules equal.
- `fit_tbp`: cut temperatures are fitted (shared across crudes) from Watson-K
  smoothness + a typical-cut prior. The data cannot fix the absolute
  temperature scale -- the prior does. UCDU0 is the reference and is
  reproduced (yield 3e-4, SG 0 through `characterize`); UCDU1 is NOT
  consistent with the same oils (yield up to 13.6 wt%, SG 0.16) -- a data
  property, recorded in `yield_error`, not a fit bug. Do not "fix" it by
  per-unit curves for one crude.
- Gravity knots are corrected to `characterize`'s ARITHMETIC cut mean, not the
  volume average (they differ by ~0.005 over the residue cut).
- ILLUSTRATIVE: `ASPHALTENES_PER_CCR`, `LIGHT_NITROGEN_WPPM` (naphtha N comes
  out at tens of wppm, too high), `TYPICAL_CUTS_C`.

### Calibration (#533: `calibration`, `units`)

- A calibrated unit goes back into the benchmark as ITS OWN delta-base
  linearization (`substitute`): gamma from the unit, delta/Del from its AD
  Jacobian wrt feed SG/S. This keeps `build_model` bilinear -- never put a
  unit's nonlinear solve inside the PolySystem.
- `calibrate` is a Python LM loop on purpose: a trial point where a unit
  fails (NaN, or a solver that raises) is rejected. Do not swap in
  optimistix LM -- it dies on the first non-finite trial (seen with the FCC).
- FCC: ALWAYS calibrate with `with_regenerator=True` (soft 700 C target).
  Without it the fit landed on the hot branch (1016 C regenerator, cat/oil
  2.4) next to a turning point where FD fails. Knobs end at bounds; the
  kinetics stay ILLUSTRATIVE (`fcc/CLAUDE.md`).
- The FCC's sour-gas outlet (s137/s138) is NOT pure H2S (yield > the feed S
  can make; the S-free batch m10 sends 24 % there): fitted summed with dry gas.
- Batch yields of co-processed FCC feeds (m11 DAO, m12 UCO) have coke 0.4-
  0.65 %: no heat-balanced riser makes that. They are increments, not
  stand-alone yields -- report, do not tune toward them.
- The rigorous hydrotreater does NOT reach Upd1 (naphtha 0.4 % at every bound
  vs 6.4 %) and only just reaches Upd0 (0.99 wt%, knobs at bounds); its SG
  sensitivities disagree in sign with the published delta, and Upd0's
  published S deltas are ~10x below H2S stoichiometry (Del = 10 vs Upd1's 1).
  Recorded in the docs; do not tune away (#533 says so).

### Fixed-yield blocks and the upgrade ranking (#537)

- `difflow_refinery.fixed_yield.FixedYieldBlock` is generic (not benchmark
  specific); `du2025.units.fixed_yield_block` builds one from the data. An
  outlet property that is neither transferred nor fixed is NOT reported --
  the benchmark leaves it free; do not invent one. Yields that do not sum to
  the inlets become an explicit `loss` outlet (Upf19: 1 %).
- `calibration.yield_sensitivity` is the envelope theorem with pounce's
  multipliers (sign: pounce minimizes -P * PROFIT_SCALE, so dP/dp =
  -(mult_g . dg/dp) / PROFIT_SCALE). It matched a finite re-solve to 1e-5
  (Upf10). Keep that check when changing either.
- Ranking at the Case 1 optimum: ARU on reformate (Upf13) >> GFU > MTBE >
  SDA; only the ARU warrants a mechanistic model (docs). Local, one optimum.
- The GDX stores 14 FIXED SPG levels at 0.01 (vars in no equation: GAMS's
  initial level). Our model uses FQ0 constants; do not read those levels.
