# difflow_refinery.treating

Amine treating (#471): `absorb` (Murphree stages, KE equilibrium), `regenerate`
(equilibrium-stage stripper with condenser/reflux), `rich_flash`, and the closed
loop `solve_amine_unit` (tear = lean ln alphas + reboiler T). A library, NOT a
palette operation; the gas plant's `AmineTreater` stays the fixed-fraction shortcut.

Invariants (each has a test in `tests/refinery/test_treating.py`; do not weaken them):
- Every constant in `properties.py` names its source; anything not read in a
  source is marked ILLUSTRATIVE: the heats of absorption, the amine Cp, the
  density slope, N2's Henry constant, CO2's physical solubility in the solution
  (taken as water's; the N2O-analogy table was not read) and DEA's viscosity
  T-dependence. The heats are held inside the KE model's own van 't Hoff heat
  (313-383 K) by a test: change one, re-run it.
- MDEA k2 is Versteeg & van Swaaij 1988 (NOT the Ko & Li 4.01e8 exp(-5400/T),
  which an earlier draft misattributed to Versteeg 1996); DEA k2 is Blanc &
  Demarais 1981 (an APPARENT first-order constant; DEA is zwitterionic), not Hikita.
- Balances close to round-off: the trace that holds an absent acid gas's log
  finite is taken back out. Never leave it in. The absorber scales BOTH
  outlets by fed/(fed + trace) (a CO2-free lean does not absorb the trace, so
  taking it from the rich alone went negative, #475); the flash takes it out
  of the liquid.
- An absent acid gas (refinery fuel gas has H2S, no CO2) must converge
  (`test_gas_without_co2`, #475). Three things make it work; keep all three:
  KE `TRACE` is 1e-30, far below a stripped bottoms' ~1e-17 loading (at 1e-12
  the pressure froze there and the regenerator had no solution); the
  regenerator scales each acid-gas balance by ITS OWN feed (+1e-6 of the
  total), not the total; the loop tears ln(alpha + LEAN_FLOOR) and starts each
  loading at min(fed/L, 1e-3 or 5e-3). With no source the loop map of ln alpha
  is a pure shift, flat in the residual, so starting high never converged.
  A floor held by max() in the tear would circulate floor*L, and so create it.
- The regenerator is pinned to the J.M. Campbell MDEA case (Moshfeghian, Tip of
  the Month July 2018: 1.1 lb/gal -> ~126 C, lean 0.002-0.004). A trade article,
  not a handbook; GPSA/Kohl & Nielsen were not available.
- Tray k_G, k_L, area are inputs (no defaults): the AIChE/Chan-Fair correlations
  were not verified. Test trays are ILLUSTRATIVE.
- Solves: damped Newton on stop-gradient inputs, then ONE implicit step
  (`ag._implicit`), so derivatives are exact. The loop's outer Newton is
  `_newton_seq` (sequential backtracking): the vmapped 16-alpha line search of
  `ag._newton` batches every nested solve 16 ways. `solve_amine_unit` is jitted
  and cached on the params with circulation/Q zeroed; pass those as arguments
  to differentiate. First call compiles ~60 s; loop tests are `slow`.
- Absorber is isothermal; water through it is fixed (sweet gas leaves saturated).
