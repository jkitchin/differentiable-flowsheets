"""Split-shell cascade for multi-product REE separation.

A split-shell design produces multiple products from a single cascade
by withdrawing intermediate streams at different points.

          Feed
            ↓
    ┌───────────────────────────────────────┐
    │                                       │
    │     ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐│
    │ Org │  1  │──│  2  │──│  3  │──│  4  ││ Org
    │ ◀───│     │  │     │  │     │  │     │◀───
    │     └─────┘  └─────┘  └─────┘  └─────┘│
    │        │        │        │        │   │
    └────────┼────────┼────────┼────────┼───┘
             ↓        ↓        ↓        ↓
          Heavy    Mid-H    Mid-L    Light
          Product  Product  Product  Raffinate

This allows simultaneous separation of multiple groups.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.units.kremser import kremser_two_inlet


@dataclass(repr=False)
class SplitShellParams(ParamsMixin):
    """Parameters for split-shell cascade.

    Attributes:
        extractant: Extractant name
        elements: All REE elements
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_stages: Total number of stages
        split_points: Stage numbers where products are withdrawn. Strictly
            increasing, each in ``[1, n_stages - 1]``; anything else is
            rejected (a duplicate or out-of-range split used to run extra
            stages and report reversed stage ranges, audit a).
        product_groups: Element groups, one per cascade section in the order
            the aqueous meets them (most extractable first), optionally with
            one more group for the raffinate. When given and neither ``pH``
            nor ``section_pHs`` is, each section runs at its own pH, read off
            the D curves at the cascade's actual O/A when it is called: the
            pH where the geometric-mean ``D * (O/A)`` of the boundary pair
            (the section's least extractable member, the most extractable
            element still to come) is one, or, for a final group with no
            raffinate group after it, where its least extractable member has
            ``D * (O/A) = 10`` (:mod:`difflow_ree.equilibrium.operating_points`).
            This field used to be stored and never read, so every section ran
            at one pH and the cascade made one useful split (audit R7).
        pH: Operating pH for every section. None (the default) uses the
            ``product_groups`` cuts when groups are given, and otherwise
            resolves through :func:`difflow_ree.database.default_pH` to the
            extractant record's own scrubbing pH -- a quarter of the way up
            its fitted validity window (#270). One pH puts one D crossing in
            the cascade, so without groups expect one useful split.
        section_pHs: Explicit pH per section (``len(split_points) + 1``
            values); overrides both of the above.
        extractant_conc: Extractant concentration
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195)
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195).
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    n_stages: int = 20
    split_points: tuple[int, ...] = (5, 10, 15)  # Withdraw at these stages
    product_groups: dict = None  # Maps product name to elements
    # (#270) None means "the extractant record's own scrubbing pH", a quarter
    # of the way up its fitted validity window -- the fractionating point,
    # where D straddles one across the element set. The literal 3.5 it
    # replaced was 1.5 pH units outside D2EHPA's refitted window and put every
    # element at 100% extraction, so the cascade separated nothing.
    pH: float | None = None
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    nitrate_conc: float | None = None  # see #195
    mechanism: str | None = None  # see #195
    section_pHs: tuple[float, ...] | None = None  # audit R7

    def __post_init__(self):
        """Check the splits and groups; resolve the pH default (#270, R7)."""
        self.split_points = tuple(int(sp) for sp in self.split_points)
        sp = self.split_points
        if any(b <= a for a, b in zip(sp, sp[1:])) or \
                any(not 1 <= x <= self.n_stages - 1 for x in sp):
            raise ValueError(
                f"split_points {sp} must be strictly increasing and each in "
                f"[1, n_stages - 1] = [1, {self.n_stages - 1}]: a duplicate "
                f"makes an empty section and a split at or past n_stages runs "
                f"stages the cascade does not have.")
        n_sections = len(sp) + 1
        if self.section_pHs is not None:
            self.section_pHs = tuple(float(x) for x in self.section_pHs)
            if len(self.section_pHs) != n_sections:
                raise ValueError(
                    f"section_pHs has {len(self.section_pHs)} values for "
                    f"{n_sections} sections.")
        if self.product_groups is not None:
            groups = {k: tuple(v) for k, v in dict(self.product_groups).items()}
            unknown = sorted({e for g in groups.values() for e in g}
                             - set(self.elements))
            if unknown:
                raise ValueError(
                    f"product_groups name {unknown}, which are not in "
                    f"elements {tuple(self.elements)}.")
            if len(groups) not in (n_sections, n_sections + 1):
                raise ValueError(
                    f"product_groups has {len(groups)} groups for "
                    f"{n_sections} sections: give one per section, or one "
                    f"more for the raffinate.")
            self.product_groups = groups
        if self.pH is None and (self.product_groups is None
                                or self.section_pHs is not None):
            from difflow_ree.database import default_pH

            self.pH = default_pH(self.extractant, "scrubbing")

    def resolve_section_pHs(self, phase_ratio: float) -> tuple[float, ...]:
        """The pH of every section at the cascade's ``O/A``.

        Args:
            phase_ratio: ``F_org / F_aq`` as the cascade computes it.

        Returns:
            One pH per section; see ``product_groups`` for the rule.
        """
        n_sections = len(self.split_points) + 1
        if self.section_pHs is not None:
            return self.section_pHs
        if self.pH is not None:
            return (float(self.pH),) * n_sections
        from difflow_ree.equilibrium.distribution import REEDistribution
        from difflow_ree.equilibrium.operating_points import CUT_FACTOR, pH_where

        dist = REEDistribution(
            self.extractant, tuple(self.elements),
            concentration=self.extractant_conc,
            nitrate_conc=self.nitrate_conc, mechanism=self.mechanism,
            on_out_of_range="ignore")
        lo, hi = dist._ext_data.valid_ph_range
        D_mid = dist.get_D_all(pH=0.5 * (lo + hi))
        groups = list(self.product_groups.values())
        pHs = []
        for k in range(n_sections):
            group = groups[k]
            later = tuple(e for g in groups[k + 1:] for e in g)
            weakest = min(group, key=lambda e: float(D_mid[e]))
            if later:
                strongest = max(later, key=lambda e: float(D_mid[e]))
                pHs.append(pH_where(dist, (weakest, strongest),
                                    CUT_FACTOR["extraction"], phase_ratio))
            else:
                pHs.append(pH_where(dist, (weakest,),
                                    CUT_FACTOR["bulk_extraction"], phase_ratio))
        return tuple(pHs)


class SplitShellCascade:
    """Split-shell cascade for multi-product separation.

    Produces multiple REE products by withdrawing streams
    at different points along a counter-current cascade.

    Each section receives the raffinate of the previous one as its aqueous
    feed and is solved once, in order, with the two-inlet Kremser equation.
    With ``product_groups`` each section runs at the pH that cuts its group
    from the rest (audit R7).

    Example:
        >>> params = SplitShellParams(
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Sm", "Gd", "Dy"),
        ...     n_stages=20,
        ...     split_points=(7, 14),
        ...     product_groups={
        ...         "heavy": ("Dy", "Gd"),
        ...         "middle": ("Sm", "Nd"),
        ...         "light": ("Ce", "La"),
        ...     },
        ... )
        >>> cascade = SplitShellCascade(params)
        >>> results = cascade(feed, solvent)
    """

    symbol = "Split-Shell Cascade"
    equations = [
        r"\text{section }k:\quad N_k\text{ stages with side-draw into product group }k",
        r"\mathrm{recovery}_{i,k} = \sum_{\text{stages } \in k} (\text{extracted}_{i})",
    ]
    assumptions = [
        "Cascade segmented at user-specified split points.",
        "Equilibrium stages with adjusted distribution ratios per section.",
    ]
    references = ["Perry's Chemical Engineers' Handbook, 9e, Sec. 15."]
    parameter_symbols = {"n_stages": "N", "split_points": r"\{N_k\}"}
    parameter_units = {
        "n_stages": "-",
        "split_points": "-",
        "pH": "-",
        "section_pHs": "-",  # audit R7
        "extractant_conc": "mol/L",
        "solvent_to_feed_ratio": "-",
        "nitrate_conc": "mol/L",
    }

    def __init__(self, params: SplitShellParams):
        """Initialize cascade.

        Args:
            params: Cascade parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,
            mechanism=params.mechanism,
        )

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float = 298.15,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> dict:
        """Run split-shell cascade with convergence iteration.

        The cascade is divided into sections by the split points.  Each
        section is modelled with the Kremser equation.  Aqueous flow passes
        sequentially from the first section to the last (raffinate of section
        i feeds section i+1).  The organic solvent flows counter-currently
        through all sections as a single stream; after extracting material
        from each section it is collected as the product for that section.

        Each section is solved once, in order, since its aqueous inlet is
        the previous section's raffinate. REE carried in on the solvent
        enter the last section, where the solvent enters.

        Args:
            feed: Aqueous feed
            solvent: Organic solvent
            T: Temperature (K)
            max_iter: Unused; kept for compatibility (one pass suffices).
            tol: Unused; kept for compatibility.

        Returns:
            Dictionary with products and stage profiles
        """
        p = self.params
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        F_aq = feed_flows.get("H2O", 1.0)
        # Organic carrier flow = the diluent's; the extractant entry is a
        # moles-per-volume charge, not a volume (#373).
        F_org = solvent_flows.get(p.diluent, 1.0)

        # Each section at its own pH (audit R7): from product_groups cuts at
        # this O/A, from section_pHs, or one pH for all.
        # The phase ratio is only needed to cut pHs from product_groups; with
        # explicit pHs nothing is concretised and the cascade traces.
        needs_ratio = p.section_pHs is None and p.pH is None
        section_pHs = p.resolve_section_pHs(
            float(F_org) / float(F_aq) if needs_ratio else 1.0)
        D_sections = [self._distribution.get_D_all(jnp.asarray(ph), T)
                      for ph in section_pHs]
        D_values = D_sections[0]

        # Section sizes from the (validated, strictly increasing) splits.
        n_total = p.n_stages
        splits = list(p.split_points) + [n_total]
        n_sections = len(splits)
        section_stages = []
        stage_start = 0
        for split_stage in splits:
            section_stages.append(split_stage - stage_start)
            stage_start = split_stage

        # ----------------------------------------------------------------
        # The aqueous passes the sections in order; section k's organic
        # extract is product k. The solvent enters counter-currently at the
        # raffinate end, so REE it already carries enter the LAST section
        # as its organic inlet and are partitioned there (two-inlet
        # Kremser): what the aqueous takes back leaves in the raffinate,
        # the rest with that section's product. Solvent REE used to be
        # ignored while the closure still read 1.0 (audit a).
        #
        # The sections are solved in one sequential pass: each one's
        # aqueous inlet is the previous one's raffinate, which is known, so
        # the fixed-point iteration this replaced always converged on its
        # second pass. ``converged_in_iter`` is kept and reports 1.
        # ----------------------------------------------------------------
        products = {}
        aq = {elem: jnp.asarray(feed_flows.get(elem, 0.0)) for elem in p.elements}
        stage_start = 0
        for i, (split_stage, n_sec) in enumerate(zip(splits, section_stages)):
            last = i == n_sections - 1
            section_flows = {}
            for elem in p.elements:
                D = D_sections[i][elem]
                F_org_in = jnp.asarray(solvent_flows.get(elem, 0.0)) if last else 0.0
                F_raff, F_ext = kremser_two_inlet(
                    D, F_aq, F_org, aq[elem], F_org_in, float(n_sec))
                section_flows[elem] = F_ext
                aq[elem] = F_raff
            products[f"product_{i + 1}"] = {
                "stage_range": (stage_start, split_stage),
                "flows": section_flows,
                "pH": section_pHs[i],
            }
            stage_start = split_stage

        products["raffinate"] = {
            "stage_range": (splits[-1], splits[-1]),
            "flows": dict(aq),
        }

        # Calculate product compositions
        for name, prod in products.items():
            total = sum(prod["flows"].values())
            prod["composition"] = {
                elem: safe_divide(flow, total)
                for elem, flow in prod["flows"].items()
            }

        # Mass balance: everything that entered, feed AND solvent.
        feed_total = {
            elem: (jnp.asarray(feed_flows.get(elem, 0.0))
                   + jnp.asarray(solvent_flows.get(elem, 0.0)))
            for elem in p.elements
        }
        product_total = {
            elem: sum(
                prod["flows"].get(elem, 0.0) for prod in products.values()
            )
            for elem in p.elements
        }
        closure = {
            elem: safe_divide(product_total[elem], feed_total[elem])
            for elem in p.elements
        }

        return {
            "products": products,
            "D_values": D_values,
            "section_pHs": section_pHs,
            "n_stages": n_total,
            "split_points": list(p.split_points),
            "converged_in_iter": 1,
            "mass_balance": {
                "feed": feed_total,
                "product": product_total,
                "closure": closure,
            },
        }


def optimize_split_points(
    elements: tuple[str, ...],
    extractant: str,
    n_stages: int,
    n_products: int,
    pH: float | None = None,  # (#270) record's own; see default_pH
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> tuple[int, ...]:
    """Find split points that maximise inter-group separation.

    Elements are ordered by their distribution coefficient D at the
    given pH.  The cascade is then split so that the *largest gaps* in
    log(D) between adjacent elements fall at section boundaries.  This
    places the split points where the natural separation between groups
    is greatest, which is a D-value-ratio-based optimisation rather than
    the naive equal-spacing heuristic.

    Specifically, the algorithm:

    1. Sorts elements in increasing order of D (hardest-to-extract first).
    2. Computes log10(D[i+1] / D[i]) for each adjacent pair.
    3. Selects the (n_products - 1) largest gaps as split boundaries.
    4. Assigns stages proportionally to the number of elements in each
       group (more elements → more stages needed for that group).

    Note: This function uses a heuristic that is informed by the actual
    D-value landscape of the system.  It does not solve a full NLP
    optimisation problem, but it consistently outperforms equal spacing
    because it places split points at the most separable boundaries.

    Args:
        elements: REE elements to separate (any order)
        extractant: Extractant name (e.g., "D2EHPA")
        n_stages: Total stages available
        n_products: Number of products desired
        pH: Operating pH. None reads the extractant record's own scrubbing
            pH; see :func:`difflow_ree.database.default_pH`.
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195)
        mechanism: Explicit mechanism override; see REEDistribution (#195)

    Returns:
        Tuple of split stage numbers (length n_products - 1), sorted
        in ascending order.
    """
    from difflow_ree.equilibrium.distribution import REEDistribution

    if n_products <= 1:
        return ()

    if pH is None:
        from difflow_ree.database import default_pH

        pH = default_pH(extractant, "scrubbing")

    n_splits = n_products - 1

    # ------------------------------------------------------------------
    # Step 1: compute D values and sort elements by D
    # ------------------------------------------------------------------
    dist = REEDistribution(
        extractant=extractant,
        elements=tuple(elements),
        concentration=0.5,
        nitrate_conc=nitrate_conc,
        mechanism=mechanism,
    )
    D_vals = dist.get_D_all(pH)
    # Sort elements from lowest D (hardest to extract) to highest D
    sorted_elems = sorted(elements, key=lambda e: float(D_vals[e]))
    n_elem = len(sorted_elems)

    # ------------------------------------------------------------------
    # Step 2: compute log-ratio gaps between adjacent D values
    # ------------------------------------------------------------------
    if n_elem <= 1 or n_splits >= n_elem:
        # Fallback to equal spacing if there is nothing to split on
        spacing = n_stages // n_products
        return tuple(spacing * (i + 1) for i in range(n_splits))

    log_gaps = []
    for i in range(n_elem - 1):
        D_lo = float(D_vals[sorted_elems[i]])
        D_hi = float(D_vals[sorted_elems[i + 1]])
        gap = abs(jnp.log10(jnp.asarray(D_hi / (D_lo + 1e-30))))
        log_gaps.append((float(gap), i))  # (gap_size, boundary_after_index_i)

    # Pick the n_splits largest gaps as group boundaries
    log_gaps.sort(key=lambda x: -x[0])
    boundary_indices = sorted(idx for _, idx in log_gaps[:n_splits])

    # boundary_indices[k] means: split after sorted_elems[boundary_indices[k]]
    # Group sizes: number of elements in each group
    group_sizes = []
    prev = 0
    for bi in boundary_indices:
        group_sizes.append(bi - prev + 1)
        prev = bi + 1
    group_sizes.append(n_elem - prev)

    # ------------------------------------------------------------------
    # Step 3: allocate stages proportional to group size
    # ------------------------------------------------------------------
    total_elems = sum(group_sizes)
    split_points = []
    cumulative = 0
    for gs in group_sizes[:-1]:
        cumulative += max(1, round(n_stages * gs / total_elems))
        # Clamp so split points stay within [1, n_stages - 1]
        sp = max(1, min(cumulative, n_stages - 1))
        split_points.append(sp)

    # Ensure strictly increasing split points
    for k in range(1, len(split_points)):
        if split_points[k] <= split_points[k - 1]:
            split_points[k] = split_points[k - 1] + 1

    # Final clamp
    split_points = [min(sp, n_stages - 1) for sp in split_points]

    return tuple(split_points)
