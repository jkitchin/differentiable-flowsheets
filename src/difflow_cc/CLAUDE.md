# difflow_cc

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_cc` - Carbon capture

- Amine absorption: `AmineAbsorber`, `AmineStripper` (MEA, DEA, MDEA, PZ, AMP)
- Membrane: `MembraneSeparator`, `MultistageMembrane` (9 membrane materials)
- Adsorption: `PSAUnit`, `TSAUnit`, `VSAUnit`, `TVSAUnit` (8 adsorbent materials)
- Direct air capture: `SolidSorbentDAC`, `LiquidSolventDAC`
- Heat integration: `LeanRichExchanger`, `HeatRecoverySystem`
- CO2 compression: `CompressionTrain`, `Pump`
- Economics: CAPEX/OPEX estimation, levelized cost of capture
- Degradation: Amine oxidation, adsorbent capacity fade, membrane aging

## Invariants (carbon-capture audit; tests in `tests/cc/test_audit_*.py`)

Do not weaken them; each has a regression test that failed before the fix.

- Amine solvent streams: `CO2_absorbed` is the TOTAL CO2 in the solvent
  (lean + captured). Absorber writes it, stripper reads/writes it, and a
  `solvent_in` stream overrides L/G, wt% and lean loading.
- Kremser must hold for A < 1 (capture <= A); no flooring of `log(A)`.
- Stripper lean loading is bounded below by the reboiler equilibrium
  (`equilibrium_loading` at `P - x_w Psat(T_reb)`).
- Membranes: exact complete-mixing balance per species (no reversed driving
  force, no ad-hoc caps); feed side at the stream pressure unless
  `feed_pressure` is given; `pressure_ratio > 1`.
- Adsorption outlets conserve every species; `info['purity']` is the
  product stream's; infeasible points report `feasible=False`, not NaN.
- DAC capture never exceeds the CO2 in the air processed.
- Heat exchangers: one duty for both sides. Compression starts at the
  inlet stream pressure.
