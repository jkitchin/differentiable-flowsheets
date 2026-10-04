# difflow_refinery.gasplant

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_refinery.gasplant` -- the saturated gas plant (#312), on a cubic EOS

(PR default, SRK) because Raoult is tens of percent off in K at 10-20 bar:
- `gas_components(light, pseudo=, kij=, cuts=)`: real light ends + naphtha
  pseudocomponents in one table; `cuts=` keeps only the named cuts
- `GasPlantColumn` (the vacuum `StageColumn` machinery on `CubicThermo`:
  total/partial/no condenser, reboiler, any feeds, side draws); factories
  `absorber_deethanizer`, `debutanizer`, `splitter`, `c3c4_splitter`,
  `deisobutanizer` (general enough for the naphtha splitter, #311)
- `GasCompressor` (stages + knock-out condensate), `AmineTreater` (a removal
  FRACTION -- treating chemistry is out of scope), `fuel_gas`, `lpg_quality`
  (GPA 2140, limits marked verify), `reid_vapor_pressure`, `gasplant_block`

Invariants encoded in the gas plant (do not weaken them):
- Three passes: easy specs on equilibrium stages, continuation of targets AND
  Murphree efficiencies, one implicit step. Pass 1's boilup ratio is at least
  one: from the guess's 5 %-of-feed vapor floor a heavy lean oil gets a few
  percent, and pass 1 never converges (the CDU-naphtha absorber of example 38).
- O'Connell is the factories' default tray efficiency; `tray_efficiency=1.0`
  gives theoretical stages. The IDAES cross-check uses 1.0 on both sides.
- RVP is the D323 construction on the same EOS, never a correlation.
- A non-converging solve is usually an infeasible spec: a C2- spec larger than
  the C2- fed, an RVP above what a hot feed allows, an olefin-limited iC4 purity.
- The IDAES reference (`tests/refinery/reference/gasplant_reference.json`) is
  an independent IMPLEMENTATION of the same model (PR, kij 0, same constants),
  not an independent model. Regenerate it, never loosen the staleness checks.
  The debutanizer is a full `TrayColumn` comparison (agreement 1e-7, but only
  after the generator tightens SmoothVLE's eps: at IDAES's defaults the total
  condenser leaks 0.07 % of a component). IDAES's TrayColumn does not converge
  the C3/C4 splitter, so that case is IDAES flashes at difflow's stage states.

Docs: `docs/unit-operations-refinery.md` ("The saturated gas plant").
Tests: `tests/refinery/test_gasplant*.py`. Example: `examples/38_refinery_gas_plant.ipynb`.
