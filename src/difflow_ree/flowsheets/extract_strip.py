"""Basic extract-strip circuit for REE separation.

Two-section flowsheet:
1. Extraction: Load REE onto organic from aqueous feed
2. Stripping: Recover REE from organic into product solution

      Feed                    Product
        ↓                        ↑
    ┌───────────┐         ┌───────────┐
    │           │         │           │
    │ EXTRACTION│ ──Org──▶│ STRIPPING │
    │           │         │           │
    └───────────┘         └───────────┘
        ↓           ◀──Org──    ↓
    Raffinate              Strip Acid

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@dataclass(repr=False)
class ExtractStripParams(ParamsMixin):
    """Parameters for extract-strip circuit.

    Attributes:
        extractant: Extractant name
        elements: REE elements to track
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_extraction_stages: Number of extraction stages
        n_stripping_stages: Number of stripping stages
        extraction_pH: pH in extraction section
        stripping_pH: pH in stripping section
        extractant_conc: Extractant concentration (M)
        solvent_to_feed_ratio: Organic/aqueous ratio in extraction
        strip_to_solvent_ratio: Strip acid/organic ratio
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195). Threaded to both sections.
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195). Threaded to both
            sections, so a circuit never mixes mechanisms.
        capacity_sharpness: Sharpness k of the extraction section's smooth
            loading limiters; see REEExtractorParams (#193).
    """
    extractant: str
    elements: tuple[str, ...]
    diluent: str = "kerosene"
    n_extraction_stages: int = 10
    n_stripping_stages: int = 5
    extraction_pH: float = 3.5
    stripping_pH: float = 0.5
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    strip_to_solvent_ratio: float = 0.5
    nitrate_conc: float | None = None  # see #195
    mechanism: str | None = None  # see #195
    capacity_sharpness: int = 8  # see REEExtractorParams (#193)


class ExtractStripCircuit:
    """Basic extract-strip circuit.

    Simple two-section solvent extraction circuit for
    recovering REE from aqueous solution.

    Example:
        >>> params = ExtractStripParams(
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     n_extraction_stages=10,
        ...     n_stripping_stages=5,
        ... )
        >>> circuit = ExtractStripCircuit(params)
        >>> results = circuit(feed)
        >>> print(results["recovery"])
    """

    symbol = "Extract-Strip"
    equations = [
        r"\text{Aqueous feed} \xrightarrow{\text{Extract (pH 3-4)}} \text{loaded organic} \xrightarrow{\text{Strip (pH<1)}} \text{pregnant aqueous}",
        r"\mathrm{recovery} = 1 - \frac{[\mathrm{RE}]_\mathrm{raffinate}}{[\mathrm{RE}]_\mathrm{feed}}",
    ]
    assumptions = [
        "Two counter-current sections (extraction + stripping) in series.",
        "Constant phase ratios; no bleed/recycle.",
    ]
    references = ["Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014)."]
    parameter_symbols = {
        "n_extraction_stages": "N_E",
        "n_stripping_stages": "N_S",
        "solvent_to_feed_ratio": "O/A_E",
        "strip_to_solvent_ratio": "A/O_S",
    }
    parameter_units = {"n_extraction_stages": "-", "n_stripping_stages": "-"}

    def __init__(self, params: ExtractStripParams):
        """Initialize circuit.

        Args:
            params: Circuit parameters
        """
        self.params = params

        # Create extraction section
        self._extractor = REEExtractor(REEExtractorParams(
            n_stages=params.n_extraction_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.extraction_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
            capacity_sharpness=params.capacity_sharpness,  # see #193
        ))

        # Create stripping section
        self._stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
            nitrate_conc=params.nitrate_conc,  # see #195
            mechanism=params.mechanism,  # see #195
        ))

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> dict:
        """Run extract-strip circuit.

        Args:
            feed: Aqueous REE feed solution
            T: Operating temperature (K)

        Returns:
            Dictionary with:
            - raffinate: Depleted aqueous from extraction
            - product: REE product solution from stripping
            - barren_organic: Stripped organic (for recycle)
            - recovery: Overall REE recovery
            - extraction_info: Extraction section details
            - stripping_info: Stripping section details
        """
        p = self.params
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        F_aq = feed_flows.get("H2O", 1.0)

        # Create solvent stream
        F_org = F_aq * p.solvent_to_feed_ratio
        solvent_flows = {
            p.diluent: F_org,
            p.extractant: p.extractant_conc * F_org,
        }
        for elem in p.elements:
            solvent_flows[elem] = 0.0
        solvent = make_stream(solvent_flows, T, feed["P"])

        # Extraction
        raffinate, loaded_org, ext_info = self._extractor(
            feed, solvent, T, pH=p.extraction_pH
        )

        # Create strip acid stream
        F_strip = F_org * p.strip_to_solvent_ratio
        strip_flows = {"H2O": F_strip}
        for elem in p.elements:
            strip_flows[elem] = 0.0
        strip_acid = make_stream(strip_flows, T, feed["P"])

        # Stripping
        product, barren_org, strip_info = self._stripper(
            loaded_org, strip_acid, T, pH=p.stripping_pH
        )

        # Calculate overall recovery
        product_flows = get_flows(product)
        total_feed = sum(float(feed_flows.get(e, 0.0)) for e in p.elements)
        total_product = sum(float(product_flows.get(e, 0.0)) for e in p.elements)
        overall_recovery = safe_divide(total_product, total_feed)

        # Element-wise recovery
        element_recovery = {}
        for elem in p.elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = float(product_flows.get(elem, 0.0))
            element_recovery[elem] = safe_divide(f_out, f_in)

        # Mass balance verification
        raff_flows = get_flows(raffinate)
        feed_total = {
            elem: jnp.asarray(float(feed_flows.get(elem, 0.0)))
            for elem in p.elements
        }
        product_total = {
            elem: (
                float(product_flows.get(elem, 0.0))
                + float(raff_flows.get(elem, 0.0))
            )
            for elem in p.elements
        }
        mass_closure = {
            elem: safe_divide(product_total[elem], feed_total[elem])
            for elem in p.elements
        }

        return {
            "raffinate": raffinate,
            "product": product,
            "barren_organic": barren_org,
            "recovery": overall_recovery,
            "element_recovery": element_recovery,
            "extraction_info": ext_info,
            "stripping_info": strip_info,
            "mass_balance": {
                "feed": feed_total,
                "output": product_total,
                "closure": mass_closure,
            },
        }

    def optimize_stages(
        self,
        feed: Stream,
        target_recovery: float = 0.99,
        max_extraction_stages: int = 20,
        max_stripping_stages: int = 10,
    ) -> dict:
        """Find minimum stages for target recovery.

        Args:
            feed: Feed stream
            target_recovery: Desired overall recovery
            max_extraction_stages: Maximum extraction stages to try
            max_stripping_stages: Maximum stripping stages to try

        Returns:
            Optimal configuration
        """
        best_config = None
        min_total_stages = float('inf')

        for n_ext in range(3, max_extraction_stages + 1):
            for n_strip in range(2, max_stripping_stages + 1):
                # Update params temporarily
                old_ext = self.params.n_extraction_stages
                old_strip = self.params.n_stripping_stages
                self.params.n_extraction_stages = n_ext
                self.params.n_stripping_stages = n_strip

                # Rebuild units
                self.__init__(self.params)

                # Run circuit
                results = self(feed)

                # Check if target met
                if results["recovery"] >= target_recovery:
                    total = n_ext + n_strip
                    if total < min_total_stages:
                        min_total_stages = total
                        best_config = {
                            "n_extraction_stages": n_ext,
                            "n_stripping_stages": n_strip,
                            "total_stages": total,
                            "recovery": results["recovery"],
                        }

                # Restore
                self.params.n_extraction_stages = old_ext
                self.params.n_stripping_stages = old_strip

        return best_config


def design_extract_strip(
    feed_composition: dict[str, float],
    extractant: str,
    target_recovery: float = 0.99,
    extraction_pH: float = 3.5,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> ExtractStripParams:
    """Design extract-strip circuit for given feed.

    Args:
        feed_composition: Element flows in feed (mol/s)
        extractant: Extractant to use
        target_recovery: Target recovery fraction
        extraction_pH: Operating pH
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP (#195)
        mechanism: Explicit mechanism override; see REEDistribution (#195)

    Returns:
        Recommended ExtractStripParams
    """
    from difflow_ree.equilibrium.distribution import REEDistribution, stages_kremser

    elements = tuple(feed_composition.keys())

    # Get D values at operating pH
    dist = REEDistribution(
        extractant=extractant,
        elements=elements,
        nitrate_conc=nitrate_conc,  # see #195
        mechanism=mechanism,  # see #195
    )
    D_values = dist.get_D_all(extraction_pH)

    # Find element with lowest D (hardest to extract)
    min_D = min(float(D_values[e]) for e in elements)

    # Calculate stages needed for target recovery
    # Assume S/F = 1.0
    n_ext = int(stages_kremser(min_D, 1.0, target_recovery)) + 2  # Safety margin

    # Stripping at low pH, D << 1, fewer stages needed
    n_strip = max(3, n_ext // 2)

    return ExtractStripParams(
        extractant=extractant,
        elements=elements,
        n_extraction_stages=n_ext,
        n_stripping_stages=n_strip,
        extraction_pH=extraction_pH,
        nitrate_conc=nitrate_conc,  # see #195
        mechanism=mechanism,  # see #195
    )
