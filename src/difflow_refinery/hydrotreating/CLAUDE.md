# difflow_refinery.hydrotreating

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

Built on `difflow_refinery.hydroprocessing`: read
`src/difflow_refinery/hydroprocessing/CLAUDE.md` before changing anything here;
its invariants hold for this unit too.

The hydrotreater is documented there, with the building blocks.

## Jet and VGO services (#536, `hydrotreating/services.py`)

- Both services run the existing `Hydrotreater` UNCHANGED (reactor, HPS,
  loop, stripper); they add operating points (`jet_params`, `vgo_params`
  with `VGO_PRETREAT_PARAMS`), product wiring, and nothing inside the solve.
- `jet_properties`: smoke point (properties.smoke_point, UNVERIFIED Rankine
  reading), flash from the TBP T10 (conservative vs D86), aromatics, S.
  Freeze point is NaN (needs per-component n-paraffins the outputs lack) and
  its margin stays NaN -- never count it as met. Mercaptan S is reported as
  the TOTAL S (upper bound; mercaptans are not tracked).
- `heavy_ends_removal` (HDCCR, HDM) is a POST-REACTOR first-order
  correlation; constants ILLUSTRATIVE. The reactor does not track CCR or
  metals -- do not claim it does.
- `fcc_feed_from_hydrotreater` maps outputs to `FCCFeed` (Tb = product T50
  standing in for the mass-average Tb; MW Riazi-Daubert 1987).
- Published kerosene-HDT and VGO HDS/HDN data are NOT reproduced (none at
  hand). Full solves are by hand: tests/reference/hdt_services_check.py.
