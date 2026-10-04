# difflow_refinery.hydrocracking

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

Built on `difflow_refinery.hydroprocessing`: read
`src/difflow_refinery/hydroprocessing/CLAUDE.md` before changing anything here;
its invariants hold for this unit too.

## The Hydrocracker (`difflow_refinery.hydrocracking`, #307)

The same building blocks: pretreat bed (`HDTKinetics` + `VGO_PRETREAT_PARAMS`)
-> cracking bed (`HCKinetics`, a new kinetic model: continuous lumping on the
cut grid after Laxminarasimhan et al. 1996, or discrete lumps; organic-N
inhibition; H2 from the H balance of each event; heat per H2) -> HPS and
recycle gas -> simplified fractionator (smooth TBP split, NOT a column) -> UCO
recycle. Layout attributes are `HDT_ATTRIBUTES + ("cracked",)`. `hcu_block`
for planning. A library, not a palette operation.

Invariants (do not weaken them):
- The Laxminarasimhan forms are UNVERIFIED against the paper (not reachable);
  no equation numbers are given and its parameters are not used. The
  yield-vs-conversion cross-check is NOT done. Constants are illustrative.
- Cracking quench is held to bed-inlet temperature (`quench_crack=None`); its
  total share is one more unknown of the gas tear (no inner Newton). Fixed
  quench rates are a knife-edge (runaway or die-out within a few K).
- The UCO tear (~200 unknowns) is Anderson substitution with a GMRES adjoint
  (`hydrocracking.fixed_point`), never Newton; its test is relative (the bed
  integration's rtol is the noise floor). Balances add nothing for the
  recycle, so an unconverged tear shows in them.
- Recycle at fixed catalyst and T LOWERS per-pass conversion (a recycle
  reactor is less efficient than plug flow); what it buys is selectivity.
- Full-unit tests compile 2-6 min each: slow.
