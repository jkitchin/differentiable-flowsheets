# difflow_refinery.vacuum

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_refinery.vacuum` -- the vacuum unit (stage-network column) and a

compatibility view of the shared characterization:
- Assays: `Assay` (Celsius, wt%; `to_assay()` converts), `characterize`
  (the shared characterization on a 300-800 C grid + a residue lump, returned
  in the old shape), `light_crude`/`heavy_crude` (synthetic),
  `atmospheric_residue` (an idealized CDU cut, for running without a CDU)
- Correlations: names re-exported from `difflow_refinery.correlations` --
  Twu (Tc, Pc, MW), Kesler-Lee (omega, liquid Cp), Maxwell-Bonnell psat (the
  D1160 conversion), Riazi-Daubert/Lee-Kesler for comparison, `fit_antoine`
- Column: `StageColumn` over a `ColumnLayout` of `Route`s, `StageSpec` trades a
  knob or draw rate for a target on any output; `VacuumColumn` builds the VDU

Invariants encoded in the vacuum stack (do not weaken them):
- Component balances are in LOG form (`ln(l+v) - logsumexp(ln ins)`). The
  residue lump in the top stage is 1e-200 of its feed; in linear form its
  rows underflow to zero and the Jacobian is singular (cond 1.7e17).
- Trace log flows (< 1e-7 of their feed) are exempt from the Newton step
  cap and clipped on their own; in the cap, one of them scales every step
  to nothing.
- The TBP curve is C1 (monotone Hermite in `Phi^-1(x)`). The default cut
  grid puts a cut boundary on every assay point; a piecewise-linear curve
  kinks there and the TBP-point gradient had two values (1e-3 off FD).
- The heat of vaporization is the Clausius-Clapeyron slope of the SAME
  Maxwell-Bonnell psat that sets K, so energy and VLE agree on volatility.
- Twu has no root past ~840 C TBP: the residue lump's Tb, SG, MW are SET,
  never correlated.
- Draws are softmax SHARES of stage liquid, never raw rates, so a rate spec
  can never overdraw a stage. Specs replace one knob or rate equation each,
  so degrees of freedom always balance.
- The implicit step reuses the Jacobian the Newton loop evaluated at the
  converged point; do not trace a second Jacobian for it.
- A non-converging solve is usually an infeasible spec (an LVGO end point a
  low-efficiency HVGO bed cannot make), not a solver failure.

Tests: `tests/refinery/test_vacuum*.py`, `test_heavy_end.py`, `test_one_characterization.py`.
Examples: `examples/34_vacuum_distillation.ipynb`, `examples/36_crude_to_vacuum.ipynb`.
