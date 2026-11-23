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
- Swing-cut ratios `phi` are fixed parameters here; #531 makes them decisions.
- The model is nonconvex (pooling): `solve_local` is a local solve. Never
  present it as the global optimum. From the published point use
  `WARM_START_OPTIONS` (mu_init 1e-6); the default barrier ends 68 lower.
- `blocks` = the same equations solved forward, pure JAX; the replacement
  interface for nonlinear units (#531, #533-#537). `test_blocks_satisfy_model`
  ties the two together.
- The unit identification table in the docs is "consistent with the data",
  not proven; keep the distinction.
