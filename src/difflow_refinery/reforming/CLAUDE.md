# difflow_refinery.reforming

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Catalytic Reforming (`difflow_refinery.reforming`, #309)

`CatalyticReformer(ReformerParams()).solve(NaphthaFeed)` -- semi-regen train
(adiabatic beds in diffrax, fired heaters, PR separator, H2 recycle as a
`Flowsheet` tear, component-split stabilizer). Library, not a palette op.
- Lumps C6-C10 nP/iP/N/A (+ MCP, H2, C1-C5), each ONE model compound
  (`species.SPECIES`); Hf/S0/Cp are the shared `difflow_refinery.thermochemistry`
  rows (API TDB Hf, Yaws S0, TRC Cp fits), K from Gibbs energies -- never from
  a kinetic paper.
- One enthalpy basis: Hf + `CubicThermo` (ideal-gas Cp fit + PR departure).
  Reactor T comes from the conserved enthalpy by Newton, so balances close
  to round-off at any ODE tolerance. Keep it that way. Its derivative is the
  implicit one (`custom_jvp`, #440): differentiating the unrolled steps made
  a bed tangent 3x dearer and the reformer's Jacobian 5x slower.
- Beds use `diffrax.ForwardMode`: differentiate a reformer with `jax.jacfwd`
  (the recycle's implicit fixed point needs JVPs), warm-start with
  `tear_initial=res.tear`.
- Kinetic pre-exponentials are ILLUSTRATIVE (this project's); octanes other
  than n-heptane are recalled (unverified). Feed sulfur (#330,
  `reforming.sulfur`) is a TRACE element solved on the converged streams,
  outside the species list and the tear: H2S to net/fuel gas, unconverted S
  to reformate, balance exact. No Padmavathi/Taskar-Riggs
  cross-check is claimed. Separator/recycle are thin and local, to merge with
  #306's shared module later.

## CCR (`reforming.ccr`, #474)

`CCRReformer(ccr_reformer_params(), CCRParams()).solve(feed)` -> `CCRResult`
(`.reformer` is the gas-side `ReformerResult`; `outputs()` adds `ccr.*`,
`h2.export_mol_s`). `ccr_block` (planning, levers + `circulation` kg/h). A
library. Semi-regen is untouched: `ReformerParams.activity=None` is the old
reformer bit for bit; the CCR sets that per-reactor tuple itself.
- Activity `exp(-alpha C)`, bed at the PATH-MEAN activity of a linear coke
  rise (`mean_activity`, series branch near zero pick-up; tested vs quad).
  `alpha` and the form are ILLUSTRATIVE: no coke-basis alpha was verifiable.
- The coke-activity fixed point is an OUTER Python loop of warm-started
  reformer solves (coke make per unit activity held, beds re-solved by
  bisection), ~5 passes. Gradients: eager `jax.jacfwd` only -- primal
  values by `stop_gradient`, then ONE implicit step `(I-J)^-1` with J by
  jacfwd. Never `jit` it (`ccr_block` forces `jit=False`).
  The jacfwd-vs-FD check is `tests/reference/ccr_gradient_check.py`
  (~10 min, outside the suite's per-test budget); rerun it after touching
  the implicit step.
- Coke made = coke burned and each bed's `C_k - C_{k-1} = m_k/G` close to
  the loop tolerance (tested). Circulation -> inf with zero regenerated coke
  IS the fresh-catalyst reformer (tested exactly).
- The default circulation is the DESIGN circulation of `lean_naphtha()`:
  5.25 wt% spent coke, UOP's design basis (`design_circulation`). The coke
  rate law is the reformer's ILLUSTRATIVE one, so the absolute circulation
  (catalyst turnover) is not validated -- do not quote it as a plant number.
- Regenerator: O2-limited full burn (combustion module), recycle sets
  `y_O2`; `T_peak` is the gas-only adiabatic front (upper bound), so it is
  independent of the burn rate (tested). Sourced: coke_regen, coke_HC, y_O2,
  T_burn_max, Cl wt%, Pt wt%, design burn ratio 52/1000 (AFPM/UOP patents,
  quotes in the docs). ILLUSTRATIVE: alpha, T_burn_in, chloride strip,
  oxy/calc duty, PtO2 reduction stoichiometry. No "K per 0.1% O2" rule was
  found; the rise is first principles only.

Docs: `docs/unit-operations-refinery.md` (Catalytic reforming; "CCR reformer"). Tests:
`tests/refinery/test_reforming.py`, `tests/refinery/test_ccr.py`.
