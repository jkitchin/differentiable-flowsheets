# difflow_refinery.residue

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

Built on `difflow_refinery.hydroprocessing`: read
`src/difflow_refinery/hydroprocessing/CLAUDE.md` before changing anything here;
its invariants hold for this unit too.

## Residue Desulfurizer and Fuel Oil (`difflow_refinery.residue`, #331)

`ResidueDesulfurizer(char, residue).solve(residue)` -> `RDSResult`; fuel oil is
`fuel_oil_blend([res.blend_component("residue"), ...], [res.volume("residue"), ...])`
(a property-mode `BlendPool`, `VLSFO_SPECS`: 0.5 wt% S, 380 cSt, SG 0.991, CCR 18).
`RDSKinetics` = `HDTKinetics` with residue constants + refractory `S_residue`,
`NiV` (HDM onto the catalyst), `CCR` reduction, 538 C+ conversion. Once-through
treat gas, ideal product split (gas / distillate / residue). A library.
- Route (a) chosen over VDU + cutters on the lever rule
  (`cutter_fraction_for_sulfur`): a 3.3 wt% residue needs 85 % ULSD by mass to
  reach 0.5 wt%. Keep that argument in the docs if the route changes.
- `NiV` and `CCR` attributes have NO element (metals outside a cut's mass, CCR a
  subset of C); `S_residue` counts S. Balances incl. Ni+V (with the deposit) close
  to round-off -- tested at 1e-10, keep it that way.
- Conversion moves ALL of a parent's atoms to `m = nC_i/nC_j` lighter molecules
  with `m - 1` H2; the HDT cracking leak is off (`crack_k=0`) so nothing double counts.
- Constants and the refractory-S share table are ILLUSTRATIVE (ARDS ranges, pinned
  by release tests); R1-R5 references are unverified.

Docs: `docs/unit-operations-refinery.md` ("Residue desulfurization and fuel oil").
Tests: `tests/refinery/test_residue.py` (gradient and CDU route: release + slow).
