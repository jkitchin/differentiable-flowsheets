# difflow_refinery.hydrogen

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## The Hydrogen Network (`difflow_refinery.hydrogen`, #329)

`HydrogenNetwork(producers, consumers, headers).solve()` -> `H2NetworkResult`
(`outputs["h2.surplus"]`, `<consumer>.purity`, `<consumer>.purity_margin`,
`balances`, `makeup_composition(c, gases)`). `Producer.from_reformer(res)` /
`.of_purity`, `Consumer.from_hydrotreater(name, res, params)`, `PSA(recovery,
purity, target_purity=)`, swing `Import`/`H2Plant` (filled in order, capped),
`Header(min_purge=, purge_to="fuel"|"export")`. `close_hydrotreater_loop(net,
{c: (Hydrotreater, feed, params)})` feeds the header composition into
`HydrotreaterParams.makeup` by substitution on purity; `h2_block` for planning.
A library, not a palette operation.
- Every consumer on a header gets the header's purity; a consumer's demand is
  its makeup H2 FLOW (`h2.makeup`), the impurities ride along at `d/y`.
- The purge is by difference, so total/H2/mass balances close by
  construction; `balances["makeup_h2"]` is the independent check. A deficit
  is returned as a negative surplus, never clipped (`feasible` says so).
- `HydrotreaterParams.makeup` is concrete (`makeup_vector` calls `float()`):
  the loop is Python, and the returned network carries each unit's purity
  response as a LINEAR secant (`d_demand_d_purity`). Reformer gradients go
  through it in forward mode (`jax.jacfwd`).
- `min_pH2` is the MAKEUP's `y P`, not the reactor-inlet pH2 (that is the
  HDT's `reactor.pH2_in`). PSA defaults are illustrative.

Docs: `docs/unit-operations-refinery.md` ("The hydrogen network"). Tests:
`tests/refinery/test_hydrogen.py`, `tests/refinery/test_hydrogen_loop.py` (slow).
