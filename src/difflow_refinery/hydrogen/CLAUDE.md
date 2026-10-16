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

## Hydrogen PSA model (`hydrogen.psa`, #469)

`HydrogenPSA(HydrogenPSAParams())(feed) -> {product, tail_gas, purity,
recovery, gamma, heel_time, leak, breakthrough_time, swept, balance}`. A
library. `PSA.from_model(result)` bridges to the network shortcut.

- The isotherms are Zhang et al. (2025) Table 2 read as **mol/g**, even
  though the header says mmol/g. The pins and affinity orders in
  `test_psa.py` hold that reading.
- The Klinkenberg form is `+ 1/(8 sqrt tau) + 1/(8 sqrt xi)`.
  `test_klinkenberg_matches_exact` checks it against the exact
  Anzelius/Thomas integral. Do not loosen that test.
- The purge heel is the simple-wave integral. Its linear limit is
  Skarstrom's `(1 - gamma)+` (tested).
- Unknown species raise. Add an isotherm, or put the species in `TO_TAIL`
  or `TO_PRODUCT`.
- The LDF constants are ILLUSTRATIVE.
- The absolute numbers do not reproduce Zhang's three-layer bed. Only the
  Table 6 trend directions and bands are claimed.

## SMR hydrogen plant (`hydrogen.smr`, #473)

`smr_plant(ng, SMRParams())`: ZnO -> steam -> prereformer (adiabatic Gibbs) ->
reformer tubes (Xu-Froment, diffrax) -> HTS (-> LTS) -> knock-out -> PSA ->
furnace (supplementary fuel closes the duty). `H2Plant.from_smr`, `smr_block` /
`link_smr` (planning). A library.

- Xu-Froment constants are Part I Table 6 (REFERENCE activity) + Table 5,
  pinned as printed. Fresh catalyst is x2.246 (k1 9.49e15); "9.49e16" seen
  elsewhere is a copying error. The K1-K3 expressions are NOT in Part I
  (`xu_froment_equilibrium` cites Hou & Hughes 2001 Table 3); the rates use
  the formation table's K, so a fast tube lands on GibbsReactor (tested).
  The two K bases differ by up to 0.098 in ln K -- that is real, not a bug.
- Defaults are the IEAGHG 2017/02 base case; `TestIEAGHG` compares H2,
  slip, radiant duty, HTS, PSA-inlet H2. Supplementary FUEL is NOT
  benchmarked (air preheat T not given): do not tune T_air/T_bridgewall to
  hit 240 kmol/h and call it validation.
- The tube's rhs clamps T to [400, 1800] and floors n: rejected PID trial
  steps otherwise poison the reverse pass with NaN. The duty secant's tol
  (1e-4 K) sits above the ODE noise; tighter divides by ~0.
- Kvaerno5 fails (lineax TracerBoolConversionError); Tsit5 is used, so
  eta >~ 10 is too stiff (`converged=False`) -- the equilibrium-limit test
  uses eta = 1.
- The prereformer's equilibrium C2+ (ppm) rides on as element-equivalent
  CH4 + H2; dropping it breaks the plant C balance (tested).
- The PSA cycle model is Zhang's ~7 bar bed: at 25-30 bar it recovers ~0.57
  and the supplementary fuel goes NEGATIVE. Plants use the `PSA` shortcut.
- ILLUSTRATIVE: rho_bed, U / wall profile, flux shape, T_air, T_stack,
  T_reformer_in.

Docs: `docs/unit-operations-refinery.md` ("The hydrogen network", "Hydrogen PSA",
"SMR hydrogen plant"). Tests:
`tests/refinery/test_hydrogen.py`,
`tests/refinery/test_psa.py`, `tests/refinery/test_smr.py`.
