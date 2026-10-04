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
  to round-off at any ODE tolerance. Keep it that way.
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

Docs: `docs/unit-operations-refinery.md` (Catalytic reforming). Tests:
`tests/refinery/test_reforming.py` (flowsheet tests `slow`).
