# difflow_refinery.sulfur

Claus SRU + SCOT TGTU (#472): `vapor` (S2-S8, saturation, dew point), `claus`
(furnace, WHB/condensers, converters, `solve_air`), `tgtu` (hydrogenation,
quench, MDEA absorber, `solve_sru` with the acid-gas recycle). A library, NOT a
palette operation; the gas plant's sulfur shortcut is untouched.

Invariants (do not weaken them). Per the test policy `tests/refinery/test_sulfur.py` keeps only
the balance, convergence, gradient and regression ones; the comparisons with
sources below were made when the unit was written and are not per-commit tests:
- The measured TOTAL vapour pressure is Ferreira & Lobo 2011 eq 4
  (doi:10.1016/j.jct.2010.07.007; title "The low-pressure phase diagram of
  sulfur" -- not "vapour pressure ..."). The correction goes on the LIQUID's
  Gibbs energy (`liquid_correction`), never as one scale factor on every p_n:
  that would break the S2<->Sn gas equilibrium and make the condenser jump at
  the dew point. Saturated speciation must equal gas-phase speciation at s_sat.
- JANAF's raw liquid is 24% HIGH at 400 K and 30% LOW at 700 K vs Ferreira &
  Lobo: it is not "low" everywhere.
- Gamson-Elkins is checked by its features (minimum 760-880 K vs K&N's ~811 K,
  >99% at 400 K, 60-78% hot). No digitised G-E points were read: do not add
  recalled numbers ("~60% at the minimum") as data.
- Recoveries are EQUILIBRIUM, an upper bound: 2/3/4 beds sit at or just above
  the AP-42 ranges (2015: 92-95/95-96/96-97; 1995: 94-96/96-97.5/97-98.5).
  Don't tune approach_dT to land inside the practical ranges and call it validation.
- Sames-Paskall furnace correlations are NOT implemented: K&N and Pollock
  transcribe CS2/COS differently (exp(-0.965A) vs exp(-9.65A)); the original
  was not read.
- Balances: S/H/O/C/N close through the train; liquid S + tail S = S in;
  closed SRU: liquid S + SO2 emitted = S in (recycle internal).
- Solves: scalar roots by bisection (`bracket_root`; NaN counts as positive)
  then ONE implicit Newton step; air by secant (lax.while_loop on stop-gradient
  inputs) + implicit step; recycle by substitution + one `(I-J)^-1` step. All
  derivatives are exact (tests vs FD). `solve_sru` compiles in ~65 s: its tests
  and the solve_air tests are `slow`.
- The regenerator is NOT solved in `solve_sru` (solvent returned to its lean
  loading); quench water dissolves no H2S/CO2. Tray coefficients ILLUSTRATIVE.
- Steam make uses `difflow.water` fits valid 20-200 C, so steam saturation T
  stays <= 473.15 K.
