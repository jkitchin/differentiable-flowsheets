# difflow_refinery

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_refinery` -- Petroleum refining

- Assay: `Assay` (TBP curve, SG or SG curve, light ends) -> `characterize` ->
  pseudo-components (Twu / Riazi-Daubert / Lee-Kesler critical properties)
- Thermo: `ColumnThermo` -- vectorised, Raoult + Lee-Kesler Psat, ideal-gas-path
  enthalpy; water is steam or free water in the drum, never in the HC liquid
- Column: `CrudeColumn` -- EO MESH (Naphtali-Sandholm), side strippers,
  pumparounds, steam; specs from `column.*` builders, one per degree of freedom
- `Furnace`: solved WITH the column (coil outlet T is an unknown), so an
  `overflash` spec sets the furnace. A spec set with no solution (e.g. too little
  overflash for a big pumparound) returns `converged=False`, not an answer
- `CrudeUnit` (assay in, yield table out) and `CrudeDistillationUnit` (the
  registered operation; outlets in `outlet_names` order)
- Products: `product_properties`, `products.gaps` -- TBP, not D86
- Alkylation (#310, `difflow_refinery.alkylation`, a library): Sauer-Colville-
  Burwick correlations transcribed from GAMS `process.gms` (`solve_process_gms`
  reproduces its 1161.3366 optimum); reactor = per-olefin stoichiometry whose
  heavy-end split is set by the correlation's yield; DIB overhead is the
  `Flowsheet` tear. Columns are `KeySplitColumn` shortcuts (Lee-Kesler
  Raoult), NOT difflow's PR `ShortcutColumn`: AD through the recycle with that
  one runs out of memory (its Geddes non-key split is fixed in the base class;
  `GeddesShortcutColumn` is now an alias). The temperature/space-velocity octane terms and the
  selectivities are illustrative, not sourced. FCC (#308) `c3`/`c4` outlets
  feed it via `combine_feeds`.
- Composition (#305, `difflow_refinery.composition`): `characterize(assay,
  composition=True)` / `char.with_composition(CompositionData(...))` puts a
  `Composition` on `char.composition` -- per component, `char.names` order:
  `hc_type` (n,4) vol fractions of `HC_TYPES` (P, N, A, O; rows sum to 1,
  O = 0 straight run), `hydrogen` (n,) mass fraction, `sulfur`/`nitrogen`
  (n,) mass fractions with `sulfur_split` (n,5) over `SULFUR_CLASSES` and
  `nitrogen_split` (n,2) over `NITROGEN_CLASSES`. `comp.of_flows(moles)` /
  `of_stream(s)` -> `StreamComposition` (types by std volume, H/S/N by mass);
  `product_properties(..., composition=comp)`; `BlendCharacterization.
  from_characterization` picks the types up as `*_vol` qualities (vol%).
  A conversion unit edits a copy with `comp.replace(...)`.
  Invariants: C/H for the type correlation is DERIVED from the hydrogen
  (`(1-H-S-N)/H`), so H and PNA are one estimate; correlation branch joins
  (Huang index at 620 K, 2B4.1 at MW 200) are logistic BLENDS because the
  published branches disagree there -- never hard switches; measured
  PIONA/SARA/n20/H override per cut via `jnp.where` (differentiable); the
  default S/N class splits are ILLUSTRATIVE; out-of-range use warns
  (`CompositionRangeWarning`), never clips the answer. MNL50 worked examples
  for these correlations are NOT reproduced -- do not claim they are.

Docs: `docs/unit-operations-refinery.md`. Tests: `tests/refinery/`.

**One characterization (#301).** `dr.Assay(..., heavy_end=dr.HeavyEnd(),
sulfur_wt=..., ccr_wt=...)` -> `dr.characterize` is read by all three
consumers: the CDU, the VDU (`VacuumColumnParams(components=
char.pseudo_components())`) and the blend pool
(`BlendCharacterization.from_characterization(char)`). The CDU's `"residue"`
feeds the VDU in a `Flowsheet` with no re-cut (rename the VDU's outlets:
`"residue"` is taken). Correlations live once, in
`difflow_refinery.correlations`; `vacuum.correlations` and the
`characterization` helpers are re-export shims.
- Opt-in, and pinned: an assay without a heavy end characterizes bit for
  bit as before UNDER `method="twu_legacy"` (`tests/refinery/test_cdu_baseline.py`).
  The default `"twu"` is Twu (1984) as published (alias `"twu_1984"`);
  `"twu_legacy"` is the old crude-unit MW coding (Rankine constants /
  sqrt(1.8), aromatics up to 15% low), kept only to reproduce old numbers.
  Do not make it the default again.
- Water rides under TWO keys: CDU water/steam is `F_water` (WATER_MW
  18.015), VDU steam is `F_H2O` (MW_WATER 18.01528). A flowsheet species list
  is `char.names + ["water", "H2O"]`; a total balance counts each at its own
  molar mass.
- `CrudeColumn` solves a column with a near-involatile component (psat at
  600 K < 0.05 Pa, e.g. the 950 C lump) by a volatility continuation: first
  with those psats floored, then the true ones from there. Without it Newton
  stalls from the bubble-point start.
- `VacuumColumn` sends feed species it does not model (light ends, CDU
  water) out the overhead, so the flowsheet balance closes.

## Thermochemical Data (`difflow_refinery.thermochemistry`, #339)

ONE table of ideal-gas Hf(298), S0(298, 1 bar) and a Cp cubic for every
refinery model compound (76 species: C1-C12 paraffins and isomers, olefins,
naphthenes, aromatics, the HDS/HDN model compounds, H2/N2/O2/H2O/CO/CO2/SO2/
H2S/NH3). `tc.species(k)` -> `FormationData` (`Hf` J/mol, `S0` or None, `cp`,
`cp_range`, sources, `status`, `note`); `tc.Hf/S0/cp/enthalpy/entropy/gibbs`,
`tc.reaction_enthalpy/entropy/gibbs/ln_K({species: nu}, T)` (element balance
checked, K on 1 bar, Cp-integrated), `tc.IdealGasSet(names)` for arrays.
- Rule: CODATA for inorganics, API TDB for organics (one evaluation, so isomer
  differences are consistent), CRC only where API TDB is absent or an outlier
  a third source confirms; Yaws S0; TRC/JANAF/Joback Cp fits. Every override
  has a `note`; never edit a value without its source.
- No module keeps its own copy: `tests/refinery/test_thermochemistry.py`
  AST-scans every `difflow_refinery` module (allowlist: only the
  separation Cp of the gas plant and HP separator; the hydrotreater moved
  onto the table in #338). `difflow.database` is core's table, not read by the refinery.
- Benzothiophene Hf is 166.3 (calorimetric, Sabbah 1979 / Good 1972), not
  ChemSep's 137.0. Moving isomerization onto it moved the C5/C6 equilibria
  (iC5/C5 at 450 K 0.820 -> 0.772); `isom_reference.json` (IDAES) is STALE
  (strict xfail, `STALE_SINCE_339`) until regenerated with IDAES + IPOPT.

## Refinery Blending (`difflow_refinery`)

`BlendPool(product, specs, rules)` blends `BlendComponent`s (from properties,
or from pseudocomponent streams on a shared `BlendCharacterization`) and returns
properties, signed spec margins and, in stream mode, the product stream. Like
`difflow.planning` it is a library, not a palette operation: the plugin's
entry point registers `CrudeDistillationUnit` and `VacuumColumn` only.

Invariants encoded in the module (do not weaken them):
- Volumes are ideal-mixing volumes at 15 degC from SG; product SG is the
  volume average and the stream-mode mass and volume balances close to
  round-off. Both are tested.
- Ethyl RT-70 corrections are spreads, so the rule reduces exactly to the
  linear blend when the components agree -- tested, keep it that way. The MON
  interaction is on MON x SENSITIVITY and b3 = -0.00645 (Maples 2000); the
  pre-#301 code had MON x olefins and -0.0645, which inflated the MON penalty
  ten-fold. RVP index and Refutas are tested against published worked examples.
- Distillation and cetane index are COMPUTED from the blend's composition,
  never blended; the linear view averages each component's own value, and the
  difference is real (T10 especially).
- The smooth violation is `t * logaddexp(-m/t, 0)` (derivative -1/2 at an
  active spec), never the branchless form -- same reason as `difflow.stochastic`.
- Margins for an optimizer over volume flows are `weighted=True` (`V * m`):
  properties are 0/0 at an empty pool.
- `exact=` in `linear_properties`/`backoff` accepts only rules linear in the
  volumes; it describes an LP.
- Blending is nonconvex (with the corrected RT-70 every start in the example
  happens to find one plan, which is not a guarantee); do not present a
  single-start NLP as "the" optimum.

- Property estimates (#330, `difflow_refinery.properties`): `from_stream`
  estimates flash (Riazi-Daubert from D86 T10), freeze (n-paraffin ideal
  solubility on Won 1986), smoke (Riazi), viscosity (Abbott + D341 at
  `viscosity_T_C`) and straight-run RON/MON (the reformer's pure-compound
  octanes by P/N/A/O, RT-70) unless given; `estimate=False` is the old
  behaviour. All but Won's melting points are UNVERIFIED against their
  sources -- keep them marked so, and let a measured value override.

Docs: `docs/unit-operations-refinery.md`. Example:
`examples/33_refinery_gasoline_blending.ipynb`. Tests: `tests/refinery/`.

## Chaining Units (`difflow_refinery.plant`, #334)

`Chain(Stage("nht", f, modes="rev"), Stage("reformer", g, modes="fwd", jit=True)).jacobian(x)`
composes library units (and the pure-JAX adapters between them) into one
differentiable function. `method="auto"`: one `jax.jacfwd`/`jacrev` when every
stage shares the mode, else `"chain"` (the chain rule by unit Jacobians, forward
accumulation). `AD_MODES` / `ad_mode_table()` is the one place each unit's AD mode
is written down. Example 40, section 10, uses it. A library.
- A mixed chain (forward-only reformer, reverse-only default hydrotreater) can NOT
  be traced end to end in either mode; `tests/refinery/test_plant.py` pins both
  failures on toy stages. `HydrotreaterParams(reactor=ReactorOptions(adjoint="forward"))`
  makes the hydrotreater forward-capable (same values).
- `jit=True` on any stage holding a Python-level recycle (the reformer's
  `Flowsheet`): unjitted, it is traced and compiled anew on every JVP and every
  call (measured 671 s / 511 s vs 487 s / 34 s jitted).
- Memory, not time, limits a chain on a 15 GB box: example 40 calls
  `jax.clear_caches()` before differentiating. Keep interfaces after a
  reverse-only stage narrow (one cotangent per output of that stage).

Docs: `docs/unit-operations-refinery.md` ("Chaining units"). Tests:
`tests/refinery/test_plant.py` (per commit), `tests/refinery/test_plant_chain.py` (release, slow).

Subpackages carry their own CLAUDE.md: `alkylation/` is above; `vacuum/`,
`reforming/`, `hydroprocessing/` (shared by `hydrotreating/`, `hydrocracking/`,
`residue/`), `hydrocracking/`, `hydrogen/`, `residue/`, `preheat/`, `gasplant/`, `fcc/`.
