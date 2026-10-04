# difflow_power

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_power` -- Electrical grids

- Network model: `PowerNetwork` (buses, branches, generators, loads); one branch
  model serves lines, transformers and phase shifters
  (`Yff = (ys + jb/2)/tau^2`, `Yft = -ys/conj(t)`, tap at the FROM end)
- Equations: `difflow_power.residuals.power_flow_residuals` is the single
  JAX-traceable definition of the equation set (2 balance rows per bus plus one
  angle-reference row); `powerflow`, `opf`, `estimation` and `verify` are all
  consumers of it and restate no physics
- Power flow: `solve_power_flow` (Newton via optimistix, implicit-diff gradients);
  the bus-type specification is written as `2 n_gen - 1` EQUATIONS, not by
  eliminating variables
- AC-OPF: `solve_acopf` over `difflow_power.ipm`, a primal-dual interior-point
  NLP solver written in JAX (no IPOPT -- that would end the differentiability).
  LMPs from the equality multipliers; `check_prices()` verifies them against
  `jax.grad` of the optimal cost through the KKT system
- DC: `solve_dcopf`, `ptdf`, `lodf`, `contingency_flows` (same IPM; a QP is an
  NLP with a constant Hessian, so DC and AC prices are comparable)
- Feeders: `RadialFeederFlowsheet` (backward/forward sweep, the voltage profile
  as the tear), `build_ladder_flowsheet` (a real `difflow.Flowsheet`)
- Sensitivity: `loss_sensitivity`, `branch_flow_sensitivity`,
  `demand_sensitivity`, `parameter_sensitivity`, `voltage_stability_margin`
- Estimation: `estimate_state` over `difflow.reconciliation` (state estimation
  and data reconciliation are the same computation)
- Cases: `case3`, `case5` (PJM), `case9` (WSCC), `case14` (IEEE),
  `radial_feeder`, plus `from_matpower`

Invariants encoded in the plugin (do not weaken them):
- Thermal limits are posed on `|S|^2`, never `|S|`: the modulus has curvature
  going as `1/|S|` and early interior-point iterates sit on lightly loaded
  branches.
- The angle-reference row stays in the residual set. Without it the Jacobian is
  one rank short for structural reasons and everything that inverts it fails.
- Limits are NOT equations. Voltage/generator/thermal/angle bounds live in
  `opf.py`, not in `residuals.py`.
- The IPM's inertia test runs on the RUIZ-EQUILIBRATED KKT matrix and uses a
  band (`n_neg <= m_eq <= n_nonpos`), because `Sigma = z/s` spans 12 decades
  near the solution and a weakly convex problem has genuine zero eigenvalues.
- `mu` follows IPOPT's monotone schedule judged on the subproblem, and its floor
  is `tol_comp / (10 m_in)`. Tying `mu` to `s.z` deadlocks on a degenerate
  problem; a floor of `tol_comp` makes the tolerance unreachable.
- Every benchmark number is asserted against MATPOWER's published answer. A
  self-consistent implementation with the phase-shift sign backwards converges
  beautifully to the wrong result.
- Bus/branch/generator order is INSERTION order, not sorted: numeric labels
  sorted as strings interleave "10" between "1" and "2".
- Out of scope by design: unit commitment (integer), security-constrained OPF,
  dynamics/transient stability, unbalanced three-phase. Do not add them.

Docs: `docs/unit-operations-power.md`. Tests: `tests/power/`.
