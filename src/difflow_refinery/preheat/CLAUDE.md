# difflow_refinery.preheat

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Crude Preheat Train (`difflow_refinery.preheat`)

`PreheatedCrudeUnit(assay, column, train)` puts a heat-exchanger train,
`Desalter` and `PreflashDrum` in front of the atmospheric column and solves
the coupled problem: an outer Newton on the tear (pumparound return
temperatures, drum temperature, furnace inlet temperature) around the train's
own Newton and the EO column, steps clipped to 30 K, implicit gradients.
`CrudeUnitWithPreheat` is the flowsheet operation. Fouling is a per-exchanger
`Rf`; `fouling_sensitivity` and `cleaning_ranking` give d(fired duty)/dRf and
the exchangers ranked by what cleaning them saves.

Invariants (do not weaken them):
- A pumparound that runs through the train is specified by its rate and its
  RETURN TEMPERATURE; the spec's value is only the loop's starting guess, the
  train sets the answer.
- The LMTD uses `abs()` and a `MIN_DELTA_T` floor (1e-6 K), so a temperature
  cross is NOT prevented -- an undersized hot stream on a large area pinches
  and the answer is the floor, not an error. Check the approach temperatures.
- Free water in the drum is a third phase (all water to vapour or liquid
  water, never dissolved in the oil).
- The Ebert-Panchal fouling constants are illustrative, not fitted.

Validation: `tests/refinery/test_preheat_validation.py` (release) against IDAES
`Flash` and `HeatExchanger` on the same ideal thermo -- an independent
implementation, not an independent model. Tests: `tests/refinery/test_preheat*.py`.
Example: `examples/37_crude_preheat_train.ipynb`.
