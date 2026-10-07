"""Carrying spectator species through a contactor section.

The scrubber and stripper rebuilt their outlets from the extractant, the
diluent, the aqueous carrier and ``elements``. Anything else on an inlet, a
saponification counter-ion such as ``Na_org``, a modifier, acid or an
impurity metal in the scrub or strip liquor, left through no outlet, so an
``ExtractScrubStripModule`` fed solvent with ``Na_org = 3.0`` returned no Na
anywhere and a closed saponification loop only balanced because its test
passed ``allow_species_loss=True`` (2026 conservation audit, b). These
species do not partition in the correlation model, so they leave with the
phase they came in.
"""

from __future__ import annotations

import jax.numpy as jnp


def carry_through(target: dict, source: dict, elements) -> tuple[str, ...]:
    """Add ``source``'s spectator species to ``target`` (an outlet's flows).

    A spectator is any species that is not already an outlet key and not one
    of ``elements``. An REE that is not in ``elements`` is NOT carried: it
    would leave unpartitioned, as if its D were 0 or infinite depending on
    the phase, which is a wrong answer rather than a spectator. It is
    returned instead, so the unit can report it (#288).

    Args:
        target: Outlet flows to extend, in place.
        source: Inlet flows of the same phase.
        elements: The elements the section partitions.

    Returns:
        The untracked REE found in ``source``, sorted.
    """
    from difflow_ree.database import list_ree_elements

    ree = set(list_ree_elements())
    untracked = []
    for key, value in source.items():
        if key in target or key in elements:
            continue
        if key in ree:
            untracked.append(key)
            continue
        target[key] = jnp.asarray(value)
    return tuple(sorted(untracked))
