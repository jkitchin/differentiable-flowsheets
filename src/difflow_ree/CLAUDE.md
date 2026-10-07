# difflow_ree

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## `difflow_ree` -- Rare earth element extraction

- Unit operations: `REEExtractor`, `REEMixerSettler`, `REEScrubber`, `REEStripper`
- Precipitation: `OxalatePrecipitator`, `CarbonatePrecipitator`, `HydroxidePrecipitator`
- Flowsheets: `ExtractStripCircuit`, `ExtractScrubStripCircuit`, `SplitShellCascade`, `FullSeparationTrain`
  return ONE results dict, so the palette registers stream-returning wrappers
  under their names (`flowsheets/palette.py`: `ExtractStripUnit`, ...). Keep
  the circuits' dict return (callers read it by key) and keep the palette
  pointing at the wrappers: a results dict wired as a unit went downstream as
  a "stream" (KeyError 'T'), and `_split_result` now refuses it.
- Database: 15 REE elements (the 14 stable lanthanides -- no Pm -- plus Y), 5 extractant systems (D2EHPA, PC88A,
  Cyanex 272, TBP, naphthenic acid). Coverage is UNEVEN and `ext_db.coverage()`
  reports it: only naphthenic_acid has coefficients for all fifteen; the other
  four cover ten (no Ho, Er, Tm, Yb, Lu).
- Free extractant (#267): the correlation's `[HA]` is FREE (Q1 Eq. 2.88/2.89),
  not total. `solve_free_extractant(dist, el, c_aq, pH=...)` closes
  `c_org = D([HA]_free) c_aq`, `[HA]_free = [HA]_0 - m c_org` as a monotone
  scalar root through optimistix (implicit diff, so gradients survive). Total
  overpredicts D where the cascade works hardest -- 1.34x at the naphthenic
  anchor. `check_loading_capacity` / `implied_loading_fraction` reject a
  loading past `1/basis_units_per_ree` (3 dimers for D2EHPA, on the same dimer
  basis as `extractant_conc` -- NOT `monomers_per_ree` = 6, which halved the
  capacity until #374), which a total-basis correlation returns a
  finite D for. Do NOT compose with `LoadingIsotherm.apparent_D`: that caps
  the answer, this changes the input (#190/#204's double count).
- Langmuir constants are DERIVED (#268): `typical_K_L` was a second extractant
  table hand-synced with the YAML, and three of four entries matched the
  coefficients at NO pH (rms log10 residual 0.62/0.85/0.92 at best fit). Now
  `K_L = D(reference)/q_max` (`q_max = [HA]_ref / basis_units_per_ree`) computed on access; `EXTRACTANT_CAPACITIES` is a
  derived Mapping, not a dict. Every record declares its basis:
  `reference_concentration` plus `reference_pH` (cation exchange) or
  `reference_nitrate` (solvating). A missing extractant was already tested for;
  a STALE one was not, which is why it drifted silently. `naphthenic_acid` is
  the check that the derivation is right: at its declared `reference_pH` of 4.5
  it reproduces all fifteen of the deleted literals to four figures, which the
  other three could not be made to do at any pH. The memo in front of it is
  keyed on a FINGERPRINT of the record, never on the extractant name --
  `add_element_to_extractant` mutates a record IN PLACE, so identity and
  equality both say "unchanged" while the basis of every derived constant has
  moved, and a name-keyed memo puts the staleness back in memory where no
  drifted number in a file gives it away.
- Uncertain D: `REEDistribution(..., coefficient_overrides={"Nd": {"a": ...}})`
  replaces tabulated log10(D) correlation coefficients, and accepts JAX tracers,
  so a distribution can be put on D and differentiated through. Passed through by
  `REEExtractorParams`, `MixerSettlerParams`, `ScrubberParams`, `StripperParams`.
  `n_stages` is likewise a continuous, traceable decision (Kremser is `E**(N+1)`)
- Operating points (2026 audit): a circuit section pH left `None`
  (`ExtractStripParams`, `ExtractScrubStripParams`, `GroupSeparator`,
  `SplitShellParams` with `product_groups`) comes from
  `equilibrium/operating_points.cut_pHs`, the D x (O/A) = 1 / 10 / 0.1 cut
  at the phase ratios the units really run at (`circuit_phase_ratios`). The
  organic carrier flow is the DILUENT volume only (#373): the extractant entry
  of a solvent stream is a moles-per-volume charge, and counting it made a
  stated O/A of 1 run at 1.5 with 0.5 M. `_phase_flows` therefore keys the
  organic phase on the diluent.
  Do NOT go back to window fractions (`default_pH`) for circuits: they left
  99.8 % of the Y on a D2EHPA solvent. Strip cuts for heavy REE on D2EHPA lie
  below the fitted window; the warning is the policy, never clamp.
  `StripperParams.pH=None` means `-log10(acid_conc)`; TBP strips at
  `strip_nitrate_conc`. Sections carry non-REE species through
  (`units/carry.py`); precipitators are reagent-capped.
