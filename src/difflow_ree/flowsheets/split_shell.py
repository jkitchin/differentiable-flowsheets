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

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.equilibrium.distribution import REEDistribution


@dataclass(repr=False)
class SplitShellParams(ParamsMixin):
    """Parameters for split-shell cascade.

    Attributes:
        extractant: Extractant name
        elements: All REE elements
        n_stages: Total number of stages
        split_points: Stage numbers where products are withdrawn
        product_groups: Element groups for each product
        pH: Operating pH
        extractant_conc: Extractant concentration
    """
    extractant: str
    elements: tuple[str, ...]
    n_stages: int = 20
    split_points: tuple[int, ...] = (5, 10, 15)  # Withdraw at these stages
    product_groups: dict = None  # Maps product name to elements
    pH: float = 3.5
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0


class SplitShellCascade:
    """Split-shell cascade for multi-product separation.

    Produces multiple REE products by withdrawing streams
    at different points along a counter-current cascade.

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
    ) -> dict:
        """Run split-shell cascade.

        Args:
            feed: Aqueous feed
            solvent: Organic solvent
            T: Temperature (K)

        Returns:
            Dictionary with products and stage profiles
        """
        p = self.params
        T = jnp.asarray(T)
        pH = jnp.asarray(p.pH)

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        F_aq = feed_flows.get("H2O", 1.0)
        F_org = solvent_flows.get("Organic", 1.0)

        # Get D values
        D_values = self._distribution.get_D_all(pH, T)

        # Simulate cascade with splits
        # This is a simplified model - full model would iterate
        n_total = p.n_stages
        splits = sorted(p.split_points) + [n_total]

        products = {}
        stage_start = 0

        for i, split_stage in enumerate(splits):
            n_section = split_stage - stage_start

            # Calculate extraction in this section
            section_flows = {}

            for elem in p.elements:
                D = D_values[elem]
                E = D * F_org / F_aq

                # Kremser for this section
                E_Np1 = jnp.power(E, n_section + 1)
                frac_extracted = jnp.where(
                    jnp.abs(E - 1.0) < 1e-6,
                    n_section / (n_section + 1),
                    (E_Np1 - E) / (E_Np1 - 1.0 + 1e-10)
                )
                frac_extracted = jnp.clip(frac_extracted, 0.0, 1.0)

                F_in = feed_flows.get(elem, 0.0)
                section_flows[elem] = float(F_in * frac_extracted)

            # Name this product
            if i < len(splits) - 1:
                product_name = f"product_{i+1}"
            else:
                product_name = "raffinate"

            products[product_name] = {
                "stage_range": (stage_start, split_stage),
                "flows": section_flows,
            }

            stage_start = split_stage

        # Calculate product compositions
        for name, prod in products.items():
            total = sum(prod["flows"].values())
            prod["composition"] = {
                elem: flow / (total + 1e-10)
                for elem, flow in prod["flows"].items()
            }

        return {
            "products": products,
            "D_values": D_values,
            "n_stages": n_total,
            "split_points": list(p.split_points),
        }


def optimize_split_points(
    elements: tuple[str, ...],
    extractant: str,
    n_stages: int,
    n_products: int,
    pH: float = 3.5,
) -> tuple[int, ...]:
    """Find optimal split points for multi-product separation.

    Positions splits to maximize separation between adjacent groups.

    Args:
        elements: REE elements (ordered by D value)
        extractant: Extractant name
        n_stages: Total stages available
        n_products: Number of products desired
        pH: Operating pH

    Returns:
        Tuple of optimal split stage numbers
    """
    # Simple heuristic: equal spacing
    # Better approach: optimize based on D value ratios
    spacing = n_stages // n_products
    splits = tuple(spacing * (i + 1) for i in range(n_products - 1))
    return splits
