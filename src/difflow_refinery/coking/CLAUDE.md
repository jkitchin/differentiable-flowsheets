# difflow_refinery.coking -- delayed coker (#534)

A library, not a palette operation (like `fcc`). `DelayedCoker.solve(feed |
[feeds], **overrides)` -> flat dict; `coker_block` for `difflow.planning`.

Invariants (do not weaken them):
- Yields are Gary, Handwerk & Kaiser (5th ed., ch. 5) delayed-coking
  correlations in CCR, volume yields in API. The constants are TRANSCRIBED,
  NOT re-checked against the book here (UNVERIFIED): mark them so until
  someone checks. Du et al. Case 2's coke delta (+0.0156/wt% CCR) agrees
  with 1.6.
- The product SGs FOLLOW from the volume formulas (SG = SG_feed * w / v);
  never set them independently or the volume balance breaks.
- H2S and NH3 are carved out of the C4- gas, so the mass balance closes
  exactly; S and N splits sum to 1 (validated); all Ni+V goes to coke.
- ILLUSTRATIVE: dry gas/LPG split, diesel/HCGO split and SG offset, S/N
  splits, olefins, every operating-lever sensitivity (drum P, COT, recycle).
- Weight yields do NOT depend on SG (the correlation has none): every AD
  dY/dSG is zero. The published Du deltas have SG terms; that disagreement
  is a property of the correlation, recorded in the docs -- do not invent an
  SG term without a source.
- Out-of-range CCR (outside 4-30 wt%) WARNS (`CokerRangeWarning`), never clips.
- Feeds pool before the yield model (SG by volume, rest by mass), as Du et
  al. Fig. 5 does; a per-batch (slurry fixed-yield) form is not implemented.
- Not implemented: the kinetic upgrade path (Del Bianco 1993 lumped thermal
  cracking); published yield data for named residues are not reproduced
  (none at hand). Tests: tests/refinery/test_coker.py (fast);
  tests/reference/coker_check.py (by hand, incl. the Du Case 2 comparison).
