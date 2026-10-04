# difflow.planning

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Delta-Base Planning (`difflow.planning`)

Turns flowsheets into a planning LP/MILP whose unit submodels are AD Jacobians
("delta vectors"), kept honest with a trust region. It is a module alongside
`eo_solver.py` and `estimation/`, **not** a `difflow.plugins` entry point.

```python
from difflow.planning import Block, Network, DeltaBasePlanner

blk = Block(name="ngl", fn=flowsheet_fn,          # any pure JAX u -> y callable
            u_names=[...], y_names=[...], lb=[...], ub=[...], jit=True)
net = Network([blk, power], links=[("ngl.residue_F", "power.fuel_F")])
res = DeltaBasePlanner(net, prices={...}, specs=[("ngl.T_colfeed", "<=", 236.0)],
                       radius=0.3).solve()
res.plan, res.delta_vectors, res.pyomo_model, res.plan_sensitivity(wrt="prices")
```

Invariants encoded in the module (do not weaken them):
- Trust-region proposals are accepted only after evaluating the caller's own
  *nonlinear* blocks (`accept_test=True`); `accept_test=False` exists only to
  demonstrate the failure mode.
- Constraint violations are scored from the nonlinear model, never from LP slacks.
- Bang-bang levers get vertex-seeded starts; use `price_switch_point` for the
  finite price at which a corner flips.
- AD mode is chosen by shape (`choose_ad_mode`), never hard-coded.
- Blocks with a `phase_fn` raise `PhaseBoundaryWarning` when a proposal crosses
  a phase boundary.
- Inter-block recycles are rejected — merge them into one flowsheet.
- Out of scope by design: pooling/blending bilinearity, assay libraries,
  blending correlations, scheduling. Do not add them.

Scaling to large flowsheets (`difflow.planning.health`): a small *absolute*
composed sensitivity is physics, not a vanishing gradient — difflow runs in
float64 and relative sensitivity survives arbitrary depth. What does degrade
with size is (1) dead levers, where a `clip`/`where`/`minimum` on an active
spec gives an exactly-zero delta column so the LP never moves that lever,
(2) `(I - A)^-1` amplification from recycle loop gains near one, and (3)
constraint-matrix scale spread from mixed units. `check_delta_health(block or
network)` and `DeltaBasePlanner.check_health()` report all three; they never
raise during a solve. Keep blocks small — linearising a whole plant as one
block *does* form the deep chain-rule product and its entries do collapse.

Second order, and multi-period (`difflow.planning.curvature`, and the `Link`
machinery you already have):
- A Hessian-vector product is one `jvp` through `grad`, so the exact Hessian of
  a scalar output costs `O(n_u)` HVPs -- what a central-difference *Jacobian*
  costs. `check_model_order(block, output, sense=...)` measures the linear and
  quadratic models against the block itself and says which earns its cost.
- It recommends `"quadratic"` only when the fit is better AND the Hessian is
  definite in the direction of optimisation. An indefinite Hessian makes the
  subproblem a nonconvex QP, which forfeits the global optimality that the
  duals-as-prices reading and the Eason-Biegler theory both rest on -- so a
  perfect fit is refused, and `.caveat` names the convexification to apply.
- Definiteness is a property of the POINT, not of the model: the same reduced
  AC cost is convex at an incumbent schedule and indefinite under heavy load.
  Check it at the linearisation point each cycle; never cache the verdict.
- Multi-period inventory needs no new machinery. A `Link` is output-to-input
  and the network rejects only cycles, so `tank@t0.level_out ->
  tank@t1.level_in` is an ordinary DAG edge. Make the first period a block with
  no `level_in` so the model starts feasible.
- `Spec` is `elastic=True` by default, which is right for a commercial spec and
  WRONG for a mass balance: an elastic inventory balance lets the planner
  report a better objective by selling from an empty tank, and it converges
  without complaint. Physical constraints are `elastic=False`.
- `model_order="quadratic"` puts that curvature in the subproblem, which
  becomes a QP (`difflow.planning.quadratic`). It is what makes the loop
  TERMINATE: on case9 AC-OPF, linear runs 40 iterations and ends on the
  iteration cap, quadratic ends on its own radius test in 12, same optimum.
  `"auto"` takes curvature only where the Hessian is already definite.
- The subproblem stays a QP, never a QCQP: only the OBJECTIVE gets curvature,
  the constraint rows stay first order. Quadratic rows would make each
  subproblem a nonconvex QCQP and void the global-optimality guarantee.
- `convexify` clips wrong-signed eigenvalues and RECORDS it in
  `qp.convexification`; a convexified model is not the true second-order
  model, and the trust region plus the acceptance test are what keep it
  honest. A block with exactly zero curvature is skipped, never floored --
  flooring hands the solver curvature the model does not have.
- `QPModel.minimised` vs `.objective`: the first is the solver's convention
  (lower is better), the second the caller's. The expansion constant lives in
  the minimised convention and `objective_offset` in the caller's; conflating
  them shifts the reported value by twice the constant.
- Integer columns (piecewise SOS2) would make it a MIQP: those networks fall
  back to linear automatically.
- Feasibility restoration (`difflow.planning.restoration`) handles an
  inelastic spec violated at the start, which otherwise dead-ends: shrinking
  the radius cannot restore feasibility. Phase one relaxes only the SPEC
  rows -- model and link rows are definitional, so an equality-infeasible
  subproblem is a broken model and must be reported, not absorbed. The row
  taxonomy is total and asserted: a label matching neither `RELAXABLE_PREFIXES`
  nor `STRUCTURAL_PREFIXES` raises, because the match is positive and a new
  kind left unclassified would be silently left hard.
- Restoration has its OWN trust region and acceptance test, judged on the
  nonlinear blocks: a phase-one LP given a big enough region proposes points
  it predicts feasible and the blocks are not (measured: predicted violation
  to zero while true violation ROSE). It also keeps its own radius -- the
  search for a feasible point says nothing about where the objective model is
  trustworthy, and resuming from it makes the planner crawl.

Modifiers from plant history (`difflow.planning.attribution`):
`attribute_deltas(block, U, {output: y}, sigma_y=...)` estimates the level and
slope corrections from logged data. Slope estimability is decided by a
column-pivoted QR of the *design* only, never by the fitted answer; most
slopes are not estimable from routine closed-loop data, and saying so is the
point. Do not drop the autocorrelation inflation or the alias report -- both
exist because their absence produced confident false flags.

Reporting and drawings (use these rather than re-deriving them in a notebook):
- `planner.describe()` states the problem — objective, decisions, bounds, links, specs.
- `lp_model.as_text()` writes the assembled LP out row by row.
- `difflow.planning.diagram`: `draw_chain` (process flow diagram of the reference
  chain), `draw_planning_network` (any network as the LP holds it),
  `draw_delta_vectors`, `draw_taylor_model`, `draw_trust_region`. matplotlib is
  imported inside the functions.

From a flowsheet, and out to someone else's LP:
- `Block.from_flowsheet(fs, u=["reactor.V", "feed:feed.total_flow"],
  y=["purge.F_B", ...])` is the bridge. Lever keys are `_apply_params` notation;
  feed streams are levers via the `feed:` prefix (`T`, `P`, `total_flow`,
  `F_<species>`, `x_<species>`). Run `check_delta_vectors` before exporting.
- Under `jax.jacobian` a recycle solve routes to the optimistix fixed-point
  path automatically — the Anderson/Wegstein loops are Python and cannot be
  traced. Never record a solve diagnostic with a bare `float()`; use
  `flowsheet._concrete()`, which returns `None` under tracing.
- `difflow.planning.export`: `DeltaVectorSet.from_result` / `.from_block`, then
  `write_json` / `write_csv` / `write_lp` / `write_mps` /
  `write_iterations_csv`. Also `difflow plan-export`. Units come from
  `Block.metadata["u_units"]`/`["y_units"]`; LP symbols are sanitised and the
  map is in `meta["lp_symbols"]`. The export is one-way — no importer.

Reference model: `difflow.planning.chain.two_plant_chain()`. Docs: `docs/planning.md`.
Example: `examples/30_delta_base_planning.ipynb`. Tests: `tests/test_planning.py`,
`tests/test_planning_export.py`, `tests/test_planning_curvature.py`,
`tests/test_planning_multiperiod.py`, `tests/test_planning_quadratic.py`,
`tests/test_planning_restoration.py`, `tests/test_planning_attribution.py`, `tests/power/test_planning_opf.py` (the
accuracy claim: SLP over AD delta vectors reaches the AC-OPF optimum and beats
DC-OPF, all three dispatches scored in the full AC model; and the termination
claim, linear against quadratic).
