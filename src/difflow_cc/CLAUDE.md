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
