"""REE scrubbing section unit operation.

Scrubbing removes unwanted elements (impurities or lighter REE)
from the loaded organic phase by contacting with a scrub solution.

The scrub solution is typically:
- Dilute acid (to maintain pH)
- REE solution (to displace lighter REE)
- Pure water (simple wash)

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.equilibrium.distribution import REEDistribution


@dataclass(repr=False)
class ScrubberParams(ParamsMixin):
    """Parameters for REE scrubbing section.

    Attributes:
        n_stages: Number of scrubbing stages
        extractant: Extractant name
        elements: REE elements to track
        target_elements: Elements to retain in organic (others scrubbed)
        pH: Scrub solution pH (lower pH strips more)
        extractant_conc: Extractant concentration (M)
        scrub_type: Type of scrubbing (acid, REE, water)
    """
    n_stages: int | float | Array
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...]  # Elements to KEEP in organic
    pH: float | Array = 2.0  # Lower pH than extraction to strip impurities
    extractant_conc: float = 0.5
    scrub_type: Literal["acid", "ree", "water"] = "acid"


class REEScrubber:
    """Multi-stage REE scrubbing section.

    Removes unwanted (typically lighter) REE from the loaded organic
    by contacting with an acidic scrub solution.

    In a typical flowsheet:
    - Extract at pH 3-4: All REE go to organic
    - Scrub at pH 2: Light REE return to aqueous, heavy REE stay

    Example:
        >>> params = ScrubberParams(
        ...     n_stages=5,
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     target_elements=("Nd", "Dy"),  # Keep these
        ...     pH=2.0,
        ... )
        >>> scrubber = REEScrubber(params)
        >>> scrub_liquor, scrubbed_org, info = scrubber(loaded_org, scrub_soln)
    """

    def __init__(self, params: ScrubberParams):
        """Initialize scrubber.

        Args:
            params: Scrubber parameters
        """
        self.params = params
        self._distribution = REEDistribution(
            extractant=params.extractant,
            elements=params.elements,
            concentration=params.extractant_conc,
        )

    def __call__(
        self,
        loaded_organic: Stream,
        scrub_solution: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage scrubbing.

        Counter-current: Organic flows from extraction → stripping
                        Scrub solution flows opposite direction

        Args:
            loaded_organic: Organic from extraction (loaded with REE)
            scrub_solution: Aqueous scrub solution
            T: Temperature (K)
            pH: Scrub pH (overrides params if provided)

        Returns:
            scrub_liquor: Aqueous outlet (contains scrubbed impurities)
            scrubbed_organic: Organic outlet (purified, target REE retained)
            info: Scrubbing diagnostics
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        org_flows = get_flows(loaded_organic)
        scrub_flows = get_flows(scrub_solution)

        F_org = org_flows.get("Organic", 1.0)
        F_scrub = scrub_flows.get("H2O", 1.0)

        # Get D values at scrub pH (lower than extraction)
        D_values = self._distribution.get_D_all(pH, T)

        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        scrub_liquor_flows = {"H2O": F_scrub}
        scrubbed_org_flows = {"Organic": F_org}
        scrub_efficiency = {}

        for elem in p.elements:
            D = D_values[elem]
            F_org_in = jnp.asarray(org_flows.get(elem, 0.0))
            F_scrub_in = jnp.asarray(scrub_flows.get(elem, 0.0))

            # For scrubbing, we want impurities to go to aqueous
            # Use stripping factor S = 1/(D * F_org/F_scrub) = F_scrub/(D * F_org)
            E = D * F_org / F_scrub  # Extraction factor

            # Kremser equation
            E_Np1 = jnp.power(E, n_stages + 1)

            # Fraction remaining in organic
            frac_in_org = jnp.where(
                jnp.abs(E - 1.0) < 1e-6,
                n_stages / (n_stages + 1),
                (E_Np1 - E) / (E_Np1 - 1.0 + 1e-10)
            )
            frac_in_org = jnp.clip(frac_in_org, 0.0, 1.0)

            F_total = F_org_in + F_scrub_in
            F_org_out = F_total * frac_in_org
            F_scrub_out = F_total * (1 - frac_in_org)

            scrubbed_org_flows[elem] = jnp.maximum(F_org_out, 0.0)
            scrub_liquor_flows[elem] = jnp.maximum(F_scrub_out, 0.0)

            # Track scrubbing efficiency (how much was removed)
            scrub_efficiency[elem] = {
                "D": D,
                "fraction_scrubbed": 1 - frac_in_org,
                "is_target": elem in p.target_elements,
            }

        P = loaded_organic["P"]
        scrub_liquor = make_stream(scrub_liquor_flows, T, P)
        scrubbed_organic = make_stream(scrubbed_org_flows, T, P)

        # Calculate selectivity metrics
        target_retained = {}
        impurity_removed = {}
        for elem, eff in scrub_efficiency.items():
            if eff["is_target"]:
                target_retained[elem] = 1 - float(eff["fraction_scrubbed"])
            else:
                impurity_removed[elem] = float(eff["fraction_scrubbed"])

        info = {
            "n_stages": n_stages,
            "pH": pH,
            "T": T,
            "D_values": D_values,
            "scrub_efficiency": scrub_efficiency,
            "target_retained": target_retained,
            "impurity_removed": impurity_removed,
        }

        return scrub_liquor, scrubbed_organic, info


def optimal_scrub_pH(
    extractant: str,
    target_element: str,
    impurity_element: str,
    min_target_retention: float = 0.95,
    pH_range: tuple[float, float] = (1.0, 4.0),
    n_points: int = 50,
) -> tuple[float, float, float]:
    """Find optimal scrub pH for separation.

    Finds pH that maximizes impurity removal while retaining target.

    Args:
        extractant: Extractant name
        target_element: Element to keep in organic
        impurity_element: Element to remove
        min_target_retention: Minimum fraction of target to retain
        pH_range: pH range to search
        n_points: Number of evaluation points

    Returns:
        Tuple of (optimal_pH, target_D, impurity_D)
    """
    dist = REEDistribution(
        extractant=extractant,
        elements=(target_element, impurity_element),
    )

    best_pH = pH_range[0]
    best_ratio = 0.0

    for pH in jnp.linspace(pH_range[0], pH_range[1], n_points):
        D_target = float(dist.get_D(target_element, pH))
        D_impurity = float(dist.get_D(impurity_element, pH))

        # Want high D for target (stays in organic)
        # Want low D for impurity (goes to aqueous)
        # Ratio = D_target / D_impurity should be maximized

        if D_target > 1.0:  # Target should stay in organic
            ratio = D_target / (D_impurity + 0.01)
            if ratio > best_ratio:
                best_ratio = ratio
                best_pH = float(pH)

    D_target_opt = float(dist.get_D(target_element, best_pH))
    D_impurity_opt = float(dist.get_D(impurity_element, best_pH))

    return best_pH, D_target_opt, D_impurity_opt
