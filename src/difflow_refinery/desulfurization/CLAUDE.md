# difflow_refinery.desulfurization -- S Zorb gasoline desulfurizer (#535)

A library, not a palette operation. `SZorbUnit().solve(SZorbFeed, **overrides)`
-> flat dict. Pure JAX; one scalar root (sorbent loading) by optimistix
Newton, differentiated implicitly.

Invariants (do not weaken them):
- EVERY rate constant, activation energy, sorbent capacity/regen loading/
  circulation, H2-per-S stoichiometry, octane-loss slope and light-ends
  factor is ILLUSTRATIVE, chosen to land in the published S Zorb RANGES
  (500 wppm -> <10 wppm, RON loss ~1). No plant data are reproduced (#535
  asks for one case; none was at hand). Never quote outputs as predictions.
- Process description only is sourced: Babich & Moulijn, Fuel 82 (2003) 607.
- S balance (feed = product + captured) and mass balance (feed + H2 =
  product + light ends + captured S + H2 to water) close EXACTLY by
  construction -- keep every new term inside both.
- The sorbent loop is the model's nonlinearity: removal lowers activity,
  activity lowers removal. Keep it a root solve (implicit diff); a fixed
  number of substitution passes would make the Jacobian wrong.
- Activity is a SMOOTH floor (softplus), never `jnp.maximum`: a kink at
  saturation breaks Newton and the gradient. Near saturation it WARNS
  (`SZorbRangeWarning`).
- Ea_olefin > Ea_s on purpose: a hotter bed costs octane faster than it
  removes sulfur (the sulfur-octane trade-off the planner sees).
- Sulfur classes are #305's SULFUR_CLASSES (sulfides incl. mercaptans, ...);
  `FCC_GASOLINE_SULFUR_SPLIT` is ILLUSTRATIVE.
- Not implemented: the Prime-G selective-HDS variant (#535: separate PR).
Tests: tests/refinery/test_szorb.py (fast); tests/reference/szorb_check.py
(by hand: balances, trends, FD, Du Upd4).
