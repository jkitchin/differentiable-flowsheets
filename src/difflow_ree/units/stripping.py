"""REE stripping section unit operation.

Stripping recovers REE from the loaded organic phase into
an aqueous strip solution, regenerating the solvent for reuse.

Strip solutions are typically:
- Strong acid (HCl, H2SO4, HNO3)
- Low pH drives REE back to aqueous phase

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide, safe_log
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.equilibrium.distribution import REEDistribution


@dataclass(repr=False)
class StripperParams(ParamsMixin):
    """Parameters for REE stripping section.

    Attributes:
        n_stages: Number of stripping stages
        extractant: Extractant name
        elements: REE elements to track
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        pH: Strip solution pH (very low, typically < 1)
        extractant_conc: Extractant concentration (M)
        acid_type: Type of strip acid
        acid_conc: Acid concentration (M)
    """
    n_stages: int | float | Array
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    pH: float | Array = 0.5  # Very low pH for complete stripping
    extractant_conc: float = 0.5
    acid_type: Literal["HCl", "H2SO4", "HNO3"] = "HCl"
    acid_conc: float = 4.0  # M


class REEStripper:
    """Multi-stage REE stripping section.

    Strips REE from loaded organic back to aqueous phase
    using strong acid solution.

    In a typical flowsheet:
    - Load organic at pH 3-4
    - Scrub at pH 2 to remove impurities
    - Strip at pH < 1 to recover product

    Example:
        >>> params = StripperParams(
        ...     n_stages=5,
        ...     extractant="D2EHPA",
        ...     elements=("Nd", "Dy"),
        ...     pH=0.5,
        ... )
        >>> stripper = REEStripper(params)
        >>> product, barren_org, info = stripper(scrubbed_org, strip_acid)
    """

    symbol = "REE Stripping"
    equations = [
        r"\overline{\mathrm{REA}_3} + 3\,\mathrm{H}^+ \rightleftharpoons \mathrm{RE}^{3+}_{(aq)} + 3\,\overline{\mathrm{HA}}",
        r"D_i = K_{\mathrm{ex},i}\,\frac{[\overline{\mathrm{HA}}]^3}{[\mathrm{H}^+]^3}\quad\Rightarrow\quad D_i \downarrow \text{ as pH} \downarrow",
        r"\frac{y_{N+1}}{y_1} = \frac{S-1}{S^{N+1}-1},\qquad S = 1/E\qquad \text{(Kremser, stripping form)}",
    ]
    assumptions = [
        "Counter-current operation with strong acid stripping solution.",
        "Constant phase flows and extractant concentration.",
    ]
    references = [
        "Rydberg, J., Musikas, C., Choppin, G.R. Principles and Practices of Solvent Extraction, Marcel Dekker, 1992.",
    ]
    parameter_symbols = {
        "n_stages": "N",
        "pH": r"\mathrm{pH}",
        "acid_conc": "[H^+]",
    }
    parameter_units = {
        "pH": "-",
        "acid_conc": "mol/L",
    }
    numerical_method = "Kremser applied in the stripping direction with acid-loading boundary."

    def __init__(self, params: StripperParams):
        """Initialize stripper.

        Args:
            params: Stripper parameters
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
        strip_solution: Stream,
        T: Array | float = 298.15,
        pH: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage stripping.

        Counter-current: Organic flows through, strip acid flows opposite

        Args:
            loaded_organic: Organic from scrubbing (contains target REE)
            strip_solution: Acidic strip solution
            T: Temperature (K)
            pH: Strip pH (overrides params if provided)

        Returns:
            product: Aqueous REE product solution
            barren_organic: Stripped organic (for recycle to extraction)
            info: Stripping diagnostics
        """
        p = self.params
        pH = pH if pH is not None else p.pH
        pH = jnp.asarray(pH)
        T = jnp.asarray(T)

        org_flows = get_flows(loaded_organic)
        strip_flows = get_flows(strip_solution)

        F_extractant = org_flows.get(p.extractant, 0.0)
        F_diluent = org_flows.get(p.diluent, 1.0)
        F_org = F_extractant + F_diluent
        F_strip = strip_flows.get("H2O", 1.0)

        # Get D values at strip pH (very low - D << 1)
        D_values = self._distribution.get_D_all(pH, T)

        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        product_flows = {"H2O": F_strip}
        barren_org_flows = {p.extractant: F_extractant, p.diluent: F_diluent}
        strip_efficiency = {}

        for elem in p.elements:
            D = D_values[elem]
            F_org_in = jnp.asarray(org_flows.get(elem, 0.0))
            F_strip_in = jnp.asarray(strip_flows.get(elem, 0.0))

            # Stripping factor S = F_strip / (D * F_org): S > 1 favors stripping
            S = safe_divide(F_strip, D * F_org)  # Stripping factor

            # Kremser equation - fraction remaining in organic
            S_Np1 = jnp.power(S, n_stages + 1)

            frac_in_org = jnp.where(
                jnp.abs(S - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                safe_divide(S - 1.0, S_Np1 - 1.0)
            )

            # At very low pH with many stages, almost complete stripping
            # S >> 1 means frac_in_org → 0
            frac_in_org = jnp.clip(frac_in_org, 0.0, 1.0)

            F_total = F_org_in + F_strip_in
            F_org_out = F_total * frac_in_org
            F_strip_out = F_total * (1 - frac_in_org)

            barren_org_flows[elem] = jnp.maximum(F_org_out, 0.0)
            product_flows[elem] = jnp.maximum(F_strip_out, 0.0)

            strip_efficiency[elem] = {
                "D": D,
                "stripping_factor": S,
                "recovery": 1 - frac_in_org,
            }

        P = loaded_organic["P"]
        product = make_stream(product_flows, T, P)
        barren_organic = make_stream(barren_org_flows, T, P)

        # Calculate overall strip performance
        total_in = sum(float(org_flows.get(e, 0.0)) for e in p.elements)
        total_product = sum(float(product_flows.get(e, 0.0)) for e in p.elements)
        overall_recovery = safe_divide(total_product, total_in)

        info = {
            "n_stages": n_stages,
            "pH": pH,
            "T": T,
            "acid_type": p.acid_type,
            "acid_conc": p.acid_conc,
            "D_values": D_values,
            "strip_efficiency": strip_efficiency,
            "overall_recovery": overall_recovery,
        }

        return product, barren_organic, info


def minimum_strip_stages(
    D: float,
    org_to_strip_ratio: float,
    target_recovery: float = 0.999,
) -> float:
    """Calculate minimum stages for target stripping recovery.

    Args:
        D: Distribution coefficient at strip pH
        org_to_strip_ratio: O/A ratio
        target_recovery: Desired recovery fraction

    Returns:
        Minimum number of stages
    """
    E = D * org_to_strip_ratio

    if E >= 1.0:
        # Stripping not favored - need many stages
        # or adjust conditions
        return float('inf')

    # From Kremser: recovery = (1 - E^(N+1)) / (1 - E^(N+1) + E - E^N)
    # Solve for N given recovery target

    # Simplified approximation for E << 1:
    # N ≈ log(1 - recovery) / log(E)
    N = safe_divide(jnp.log(1 - target_recovery), safe_log(E))

    return float(jnp.maximum(N, 1.0))


def acid_consumption(
    ree_moles: float,
    acid_type: str = "HCl",
    stoichiometry: int = 3,
) -> float:
    """Calculate acid consumption for stripping.

    REE extraction releases H+ (acidic extractants).
    Stripping consumes H+ to reverse the reaction.

    REE(HA)3 + 3H+ → REE³+ + 3HA

    Args:
        ree_moles: Moles of REE stripped
        acid_type: Type of acid
        stoichiometry: Moles H+ per mole REE

    Returns:
        Moles of acid consumed
    """
    return ree_moles * stoichiometry


def strip_solution_concentration(
    ree_flow: float,  # mol/s
    strip_flow: float,  # L/s
    elements: list[str],
    element_flows: dict[str, float],
) -> dict[str, float]:
    """Calculate REE concentration in strip product.

    Args:
        ree_flow: Total REE molar flow
        strip_flow: Strip solution volumetric flow
        elements: List of elements
        element_flows: Flow of each element (mol/s)

    Returns:
        Concentration of each element (M)
    """
    conc = {}
    for elem in elements:
        conc[elem] = safe_divide(element_flows.get(elem, 0.0), strip_flow)
    return conc
