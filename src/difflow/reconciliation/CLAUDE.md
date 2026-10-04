# difflow.reconciliation

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Keeping a Model Current (`difflow.reconciliation.tracking`)

The digital-twin loop: a parameter updated from plant data as it arrives,
gated on the monitoring verdict and filtered onto a random walk. Four public,
stateless steps — `update_gate`, `time_update`, `parameter_measurement`,
`measurement_update` — plus `track_parameters`, the offline driver over a
campaign, which doubles as the backtest for choosing `drift_std`.

```python
run = track_parameters(F, daily, sigma, names=layout.names,
                       state=TrackerState.initial(["eta"], [1.0], std=[0.02]),
                       drift_std=drift_std_from_time_constant(0.05, 30.0))
run.final.as_params()          # {'eta': 0.71}, the shape a Block takes as theta
```

Invariants encoded in the module (do not weaken them):
- The verdict **gates** the update; it does not advise it. Only `model drift`
  opens it. Re-estimating on a concentrated suspect converts a meter fault into
  a confident statement about equipment, and the chi-squared statistic *falls*
  while it happens — `tests/test_tracking.py::TestGate::test_an_ungated_loop_would_have_invented_a_fouling_factor`
  is the regression. Widening `allow=` is a deliberate act with a known cost.
- Holding on `consistent` is the deadband, not an oversight. A parameter
  re-estimated every period tracks that period's noise and the twin stops being
  a model.
- The **time update always runs**, on held periods too: a held parameter widens
  its error bar at the drift rate, so the next permitted update steps further.
- `drift_std` is required and never defaulted — it is the bandwidth of the twin.
  Same knob, same warning as `process_std` in `difflow.mhe`.
- The covariance is **full**, never diagonal (correlated parameters have a
  difference variance a diagonal gets wrong by a factor of a few), and is
  propagated in **Joseph form** — `(I-K)P` loses symmetry over a long run, and a
  twin is a long run.
- `parameter_measurement` leaves the parameters at `sigma = inf` (free), so the
  prior stays outside `reconcile`: a prior smuggled in through `sigma` would be
  diagonal, and would inflate the objective that is supposed to test data
  against model.
- A weakly informative period is not a special case — large `R`, zero gain.
- Out of scope: this filters parameters, not model *form*. Under structural
  mismatch the statistic never comes back down; the answer is
  `difflow.planning.modifiers`, not a faster filter.

Docs: `docs/data-reconciliation.md`. Tests: `tests/test_tracking.py`.
