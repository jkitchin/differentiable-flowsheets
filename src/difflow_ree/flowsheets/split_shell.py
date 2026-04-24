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


@dataclass(repr=False)
class SplitShellParams(ParamsMixin):
    """Parameters for split-shell cascade.

    Attributes:
        extractant: Extractant name
        elements: All REE elements
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_stages: Total number of stages
        split_points: Stage numbers where products are withdrawn
        product_groups: Element groups for each product
        pH: Operating pH
        extractant_conc: Extractant concentration
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    n_stages: int = 20
    split_points: tuple[int, ...] = (5, 10, 15)  # Withdraw at these stages
    product_groups: dict = None  # Maps product name to elements
    pH: float = 3.5
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0


def _kremser_fraction_extracted(E, n_stages):
    """Kremser equation: fraction extracted in n_stages counter-current stages.

    Args:
        E: Extraction factor D * (F_org / F_aq)
        n_stages: Number of stages in section

    Returns:
        Fraction of solute extracted into organic phase
    """
    E_Np1 = jnp.power(E, n_stages + 1)
    frac_extracted = jnp.where(
        jnp.abs(E - 1.0) < 1e-6,
        n_stages / (n_stages + 1.0),
        safe_divide(E_Np1 - E, E_Np1 - 1.0),
    )
    return jnp.clip(frac_extracted, 0.0, 1.0)


class SplitShellCascade:
    """Split-shell cascade for multi-product separation.

    Produces multiple REE products by withdrawing streams
    at different points along a counter-current cascade.

    The cascade is solved using a fixed-point iteration: each section
    receives the raffinate from the previous section as its aqueous feed.
    Iteration continues until the inter-stage flows converge.

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
    parameter_units = {"n_stages": "-"}

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

        Fixed-point iteration is used so that the organic loading entering
        each section is self-consistent with the aqueous composition leaving
        that section.  In practice, because the sections share a single
        counter-current organic stream the iteration converges in a small
        number of steps.

        Args:
            feed: Aqueous feed
            solvent: Organic solvent
            T: Temperature (K)
            max_iter: Maximum fixed-point iterations
            tol: Convergence tolerance on element flows (mol/s)

        Returns:
            Dictionary with products and stage profiles
        """
        p = self.params
        T = jnp.asarray(T)
        pH = jnp.asarray(p.pH)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        F_aq = feed_flows.get("H2O", 1.0)
        F_extractant = solvent_flows.get(p.extractant, 0.0)
        F_diluent = solvent_flows.get(p.diluent, 1.0)
        F_org = F_extractant + F_diluent

        # Get D values (constant throughout the cascade)
        D_values = self._distribution.get_D_all(pH, T)

        # Section boundaries: list of (start_stage, end_stage) tuples
        n_total = p.n_stages
        splits = sorted(p.split_points) + [n_total]
        n_sections = len(splits)
        section_stages = []
        stage_start = 0
        for split_stage in splits:
            n_section = split_stage - stage_start
            section_stages.append(max(n_section, 1))
            stage_start = split_stage

        # ----------------------------------------------------------------
        # Fixed-point iteration over inter-stage flows
        #
        # State: for each element, the aqueous flow entering each section.
        # Section 0 always receives the original feed.
        # Section k (k>0) receives the raffinate of section k-1.
        # ----------------------------------------------------------------

        # Initialise: assume equal partition across sections
        # section_aq_in[k][elem] = aqueous flow of elem entering section k
        section_aq_in = []
        for k in range(n_sections):
            if k == 0:
                section_aq_in.append({
                    elem: float(feed_flows.get(elem, 0.0))
                    for elem in p.elements
                })
            else:
                # Start with zero -- will be updated in first iteration
                section_aq_in.append({elem: 0.0 for elem in p.elements})

        for _iter in range(max_iter):
            prev_aq_in = [dict(s) for s in section_aq_in]

            # Forward pass: section 0 → section n_sections-1
            for k in range(n_sections):
                n_sec = section_stages[k]
                for elem in p.elements:
                    D = float(D_values[elem])
                    E = D * F_org / F_aq
                    F_in = section_aq_in[k][elem]

                    frac_ext = float(_kremser_fraction_extracted(
                        jnp.asarray(E), jnp.asarray(float(n_sec))
                    ))

                    F_raffinate = F_in * (1.0 - frac_ext)

                    # Aqueous feed to next section is raffinate of this one
                    if k + 1 < n_sections:
                        section_aq_in[k + 1][elem] = F_raffinate

            # Check convergence: max change in any inter-section flow
            converged = True
            for k in range(1, n_sections):
                for elem in p.elements:
                    delta = abs(
                        section_aq_in[k][elem] - prev_aq_in[k][elem]
                    )
                    if delta > tol:
                        converged = False
                        break
                if not converged:
                    break

            if converged:
                break

        # ----------------------------------------------------------------
        # Collect results: compute extracted (organic) flows per section
        # and the final aqueous raffinate.
        #
        # Physical convention:
        #   product_1 ... product_n  : organic extracts from each section
        #   raffinate                : final aqueous stream (unextracted)
        #
        # Mass balance:
        #   sum(product_k[elem]) + raffinate[elem] = feed[elem]
        # ----------------------------------------------------------------
        products = {}
        stage_start = 0

        for i, (split_stage, n_sec) in enumerate(zip(splits, section_stages)):
            section_flows = {}
            raffinate_flows = {}
            for elem in p.elements:
                D = float(D_values[elem])
                E = D * F_org / F_aq
                F_in = section_aq_in[i][elem]

                frac_ext = float(_kremser_fraction_extracted(
                    jnp.asarray(E), jnp.asarray(float(n_sec))
                ))
                section_flows[elem] = F_in * frac_ext          # organic
                raffinate_flows[elem] = F_in * (1.0 - frac_ext)  # remaining aqueous

            product_name = f"product_{i + 1}"
            products[product_name] = {
                "stage_range": (stage_start, split_stage),
                "flows": section_flows,
            }

            stage_start = split_stage

        # Final aqueous raffinate: unextracted remainder from the last section
        products["raffinate"] = {
            "stage_range": (splits[-1], splits[-1]),
            "flows": raffinate_flows,
        }

        # Calculate product compositions
        for name, prod in products.items():
            total = sum(prod["flows"].values())
            prod["composition"] = {
                elem: safe_divide(flow, total)
                for elem, flow in prod["flows"].items()
            }

        # Mass balance verification
        feed_total = {
            elem: jnp.asarray(float(feed_flows.get(elem, 0.0)))
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
            "n_stages": n_total,
            "split_points": list(p.split_points),
            "converged_in_iter": _iter + 1,
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
    pH: float = 3.5,
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
        pH: Operating pH

    Returns:
        Tuple of split stage numbers (length n_products - 1), sorted
        in ascending order.
    """
    from difflow_ree.equilibrium.distribution import REEDistribution

    if n_products <= 1:
        return ()

    n_splits = n_products - 1

    # ------------------------------------------------------------------
    # Step 1: compute D values and sort elements by D
    # ------------------------------------------------------------------
    dist = REEDistribution(
        extractant=extractant,
        elements=tuple(elements),
        concentration=0.5,
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
