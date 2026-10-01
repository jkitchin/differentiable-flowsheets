"""FCC product pseudocomponents and the main fractionator.

Lumps to pseudocomponents. The kinetic lumps are boiling ranges, not
species. The gasoline lump (C5 to 221 C) and the unconverted "cycle oil"
lump (221 C+, i.e. LCO plus slurry) are put on a fixed grid of product
pseudocomponents by a fixed TBP distribution per lump: a logistic
cumulative curve in TBP, truncated to the lump's range and renormalised,
``(T50, width)`` per lump (:data:`DEFAULT_LUMP_TBP`). Each grid cut's
gravity follows from a Watson K per lump, its molar mass from Twu (1984)
through :func:`difflow_refinery.correlations.critical_properties` -- the
same correlation the rest of the plugin uses. Dry gas and LPG are real
species (:mod:`difflow_refinery.fcc.species`).

The main fractionator -- SIMPLIFIED. The issue asks for a ``StageColumn``
layout with pumparounds and side strippers. That is NOT what this is: the
fractionator here is a smooth TBP split of the effluent's pseudocomponents
at two cut points (gasoline/LCO and LCO/slurry), each pseudocomponent going
to the lighter product with the fraction ``sigmoid((T_cut - Tb)/w)``,
which mimics the overlap of a real column's products. The light species
are split ideally: dry gas, C3, C4 (the gas plant's job, which here stands
in for issue #312, not built). Mass is conserved exactly. No energy model:
the split is at riser outlet conditions, and condenser/pumparound duties
are not computed. Building the main fractionator on the vacuum unit's
``StageColumn`` machinery is listed as not done in the documentation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery import correlations as corr

C = 273.15

#: Product grid cut edges (C): gasoline range C5 (27 C, isopentane's
#: boiling point) to 221 C (430 F, the conventional FCC gasoline end point
#: on which "conversion" is defined), cycle-oil range 221 C to 600 C.
GASOLINE_EDGES_C = (27.0, 45.0, 65.0, 85.0, 105.0, 125.0, 145.0, 165.0, 185.0, 205.0, 221.0)
CYCLE_EDGES_C = (221.0, 245.0, 270.0, 295.0, 320.0, 343.0, 370.0, 400.0, 430.0,
                 460.0, 500.0, 550.0, 600.0)

#: Illustrative TBP distribution of each liquid lump on the grid: logistic
#: CDF ``(T50 [C], width [K])``. Not from any source; set to put FCC
#: gasoline's T50 near 100-120 C and LCO/slurry split near the usual
#: 343 C (650 F) cut. Fit to product distillations.
DEFAULT_LUMP_TBP = {"gasoline": (110.0, 32.0), "cycle_oil": (330.0, 55.0)}

#: Illustrative Watson K of each liquid lump. FCC products are more
#: aromatic than straight-run cuts of the same boiling range: K = 11.7 puts
#: a 110 C gasoline at SG ~0.74, K = 10.6 a 290 C LCO at SG ~0.94 and a
#: 450 C slurry at SG ~1.03 -- the usual ranges (unverified against any one
#: source).
DEFAULT_LUMP_KW = {"gasoline": 11.7, "cycle_oil": 10.6}


def grid() -> dict:
    """The fixed product grid: names, edges (K), and which lump owns each cut."""
    g = np.asarray(GASOLINE_EDGES_C) + C
    c = np.asarray(CYCLE_EDGES_C) + C
    n_g, n_c = len(g) - 1, len(c) - 1
    names = tuple(f"fcc{i + 1:02d}" for i in range(n_g + n_c))
    return {"names": names, "lo": np.concatenate([g[:-1], c[:-1]]),
            "hi": np.concatenate([g[1:], c[1:]]), "n_gasoline": n_g, "n_cycle": n_c}


GRID = grid()
PSEUDO_NAMES = GRID["names"]


def lump_distribution(lo: Array, hi: Array, T50: Array, width: Array) -> Array:
    """Mass share of each cut ``[lo, hi]`` under a logistic TBP CDF, renormalised."""
    F = jax.nn.sigmoid((jnp.concatenate([lo[:1], hi]) - T50) / width)
    d = jnp.diff(F)
    return d / jnp.sum(d)


def pseudo_properties(Kw: dict) -> dict[str, Array]:
    """Tb (cut mid-point, K), SG, MW of the grid cuts for given lump Watson Ks."""
    lo, hi = jnp.asarray(GRID["lo"]), jnp.asarray(GRID["hi"])
    Tb = 0.5 * (lo + hi)
    n_g = GRID["n_gasoline"]
    Kw_cut = jnp.concatenate([jnp.full(n_g, 1.0) * Kw["gasoline"],
                              jnp.full(GRID["n_cycle"], 1.0) * Kw["cycle_oil"]])
    SG = (1.8 * Tb) ** (1.0 / 3.0) / Kw_cut
    MW, _, _ = corr.critical_properties(Tb, SG, "twu")
    return {"Tb": Tb, "SG": SG, "MW": MW, "lo": lo, "hi": hi}


def split_fractions(Tb: Array, cut1: Array, cut2: Array, width: Array) -> Array:
    """``(n, 3)``: fraction of each pseudocomponent to gasoline, LCO, slurry.

    ``s1 = sigmoid((cut1 - Tb)/w)``, ``s2 = sigmoid((cut2 - Tb)/w)``;
    gasoline ``s1``, LCO ``s2 - s1``, slurry ``1 - s2``. Rows sum to one
    exactly; LCO is non-negative while ``cut2 > cut1``.
    """
    s1 = jax.nn.sigmoid((cut1 - Tb) / width)
    s2 = jax.nn.sigmoid((cut2 - Tb) / width)
    return jnp.stack([s1, s2 - s1, 1.0 - s2], axis=1)
