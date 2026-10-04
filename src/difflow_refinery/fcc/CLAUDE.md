# difflow_refinery.fcc

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Fluid Catalytic Cracker (`difflow_refinery.fcc`, #308)

`FCCUnit(FCCParams(...)).solve(FCCFeed.from_characterization(char, rate, T_lo, T_hi))`:
lumped riser (`ancheyta_5` default, `lee_4`, `weekman_nace_3`; diffrax,
constant-step Tsit5 + `DirectAdjoint`) and coke-burning regenerator solved
TOGETHER by optimistix Newton -- unknowns C/O and T_rg, ROT the spec --
with implicit gradients; simplified main fractionator (TBP sigmoid split,
NOT a StageColumn). Outlets `dry_gas, c3, c4, gasoline, lco, slurry,
sour_water, flue_gas` (C3=/C4= for alkylation #310). `fcc_block(...)` for
planning. A library, not registered.
- Kinetic constants (`ILLUSTRATIVE_5LUMP`) are NOT from any paper; no
  literature cross-check was reproduced. Do not present yields as predictions.
- Balances (mass, C, H, S, N, energy) close by construction: cycle-oil H and
  H2S are BY DIFFERENCE; keep it that way, and keep `cycle_oil_hydrogen`
  reported so an implausible by-difference value is visible.
- The heat balance has multiple steady states; `RegeneratorTemperatureWarning`
  flags a hot one rather than hiding it.
- Under `jax.grad` w.r.t. the assay pass `indices=` (`fcc.feed.cut_indices`).

Docs: `docs/unit-operations-refinery.md` ("The fluid catalytic cracker").
Tests: `tests/refinery/test_fcc.py`.
