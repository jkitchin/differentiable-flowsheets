# difflow_refinery.hydroprocessing

Notes for Claude Code, loaded when working in this directory (moved out of
the root CLAUDE.md, which keeps only what every session needs).

## Hydroprocessing and the Hydrotreater (`difflow_refinery.hydroprocessing`, `.hydrotreating`)

`hydroprocessing/` is kinetics-agnostic and shared (the hydrocracker of #307
is meant to reuse it): `layout` (`Layout`/`Flows`: gases, cuts, and per-cut
ATTRIBUTE flows `F_<cut>@<attr>` -- C and H atoms compulsory, a cut's mass is
computed from its atoms), `thermo` (vectorised PR on traceable cut constants),
`separator` (`pr_flash`, negative flash, `HPSeparator`), `reactor`
(`TrickleBedReactor` around any object with `attributes`,
`attribute_elements` and `rates(ctx: ReactionContext, params) -> Rates`),
`recycle` (knock-out, amine, purge, ideal-gas compressor, makeup, `solve_tear`),
`stripper` (`StageColumn` steam stripper + PR overhead drum), `solve`
(`newton_solve`). `hydrotreating/` adds `HDTKinetics` (HDS by sulfur class,
LHHW; HDN; reversible aromatics; olefins; cracking leak), `Hydrotreater`,
`hdt_block`. A library, not a palette operation.

Invariants (do not weaken them):
- Attributes are extensive and follow their cut through every split; element
  balances (C, H, S, N) and mass close to round-off -- tested at 1e-8.
- A diffrax solve with `RecursiveCheckpointAdjoint` is reverse-mode only.
  Newton loops around the reactor (tear, quench, targets) use
  `hydroprocessing.solve.newton_solve`: Jacobian from a FORWARD-adjoint copy
  (`f_iter`, `jac="fwd"`), then one implicit step with the reverse-mode
  residual. Residuals must be pure in `(x, args)` -- no traced closures.
  Reverse-mode Jacobians through the checkpointed adjoint compiled for 6 min.
- Root polishing (Rachford-Rice, the PR cubic) takes TWO Newton steps on live
  inputs: the reactor differentiates derivatives (`C_eff = dH/dT`,
  `d ln K/dT`); one step left the second derivatives wrong and the loop's
  implicit gradient was 1-80 % off (high loop gain amplifies it).
- `Flows.per_molecule` uses a safe inverse (`F/(F^2+eps^2)`): `attr/max(F,
  1e-300)` gives a 1e300 cotangent for an empty cut and NaN gradients.
- The stripper feeds 10 % of its steam with the feed: a degassed separator
  liquid is subcooled and `StageColumn`'s feed flash is otherwise singular.
- Outside the two-phase region `reactor.phase_state` returns the stream itself
  and its incipient phase (`y = z, x = z/K` for a vapour), never the negative
  flash's fictitious split: that put a vapour naphtha bed's pH2 6x low and ran
  the aromatics equilibrium backwards (#332).
- `pr_flash` never raises: the Rachford-Rice bracket ignores absent (trace)
  species, a diverged Newton returns its start with a large `residual`, and
  the implicit derivative is the module's own `custom_jvp` (optimistix's
  implicit adjoint raises on a NaN Jacobian). Callers fold the residual into
  `converged` (the hydrotreater's `flash.residual`).
- `Hydrotreater.product_stream()` includes the dissolved real gases by
  default (mass closes downstream, #333); blend the wild naphtha with
  `gases=False` or through `res.fractionate(...)` (#328, a TBP sigmoid split).
  `NAPHTHA_HDT_PARAMS` is the illustrative naphtha constant set.
- Rate constants are ILLUSTRATIVE; thermochemistry is model-compound data from
  the shared table (`difflow_refinery.thermochemistry`; `MODEL_COMPOUNDS` is a
  view). The aromatics K is `aromatic_ln_K(T)` -- Cp-integrated, 1 bar standard
  state, pH2 in bar -- and its heat `aromatic_heat(T)` at the same T, so energy
  and equilibrium agree (#338: constant 298 K dH/dS made K 3-5x too large).
  Irreversible heats are 298 K values; the hydrocracker/residue per-H2
  saturation heats are `AROMATIC_DH298` (298 K, deliberate). The
  Korsten-Hoffmann profile cross-check is NOT done.

Docs: `docs/unit-operations-refinery.md` (Hydroprocessing building blocks; The
hydrotreater). Tests: `tests/refinery/test_hydrotreating.py`.
