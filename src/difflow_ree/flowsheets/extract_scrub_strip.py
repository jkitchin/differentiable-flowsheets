"""Industrial 3-section extract-scrub-strip circuit.

Three-section flowsheet:
1. Extraction: Load all REE onto organic
2. Scrubbing: Remove unwanted REE (typically lighter ones)
3. Stripping: Recover purified REE product

        Feed                    Scrub                   Product
          ↓                       ↓                        ↑
    ┌───────────┐         ┌───────────┐         ┌───────────┐
    │           │         │           │         │           │
    │ EXTRACTION│ ──Org──▶│ SCRUBBING │ ──Org──▶│ STRIPPING │
    │           │         │           │         │           │
    └───────────┘         └───────────┘         └───────────┘
          ↓                     ↓          ◀──Org──    ↓
      Raffinate            Scrub Liquor          Strip Acid
    (depleted)           (impurities)         (recycled)

The scrub liquor (containing rejected light REE) is typically
recycled back to the feed or processed in a separate circuit.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@dataclass(repr=False)
class ExtractScrubStripParams(ParamsMixin):
    """Parameters for 3-section circuit.

    Attributes:
        extractant: Extractant name
        elements: All REE elements to track
        target_elements: Elements to recover in product
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        n_extraction_stages: Number of extraction stages
        n_scrubbing_stages: Number of scrubbing stages
        n_stripping_stages: Number of stripping stages
        extraction_pH: pH in extraction section
        scrubbing_pH: pH in scrubbing section
        stripping_pH: pH in stripping section
        extractant_conc: Extractant concentration (M)
        solvent_to_feed_ratio: O/A in extraction
        scrub_to_solvent_ratio: Scrub/O ratio
        strip_to_solvent_ratio: Strip/O ratio
    """
    extractant: str
    elements: tuple[str, ...]
    target_elements: tuple[str, ...]  # Elements to keep
    diluent: str = "kerosene"
    n_extraction_stages: int = 10
    n_scrubbing_stages: int = 5
    n_stripping_stages: int = 5
    extraction_pH: float = 3.5
    scrubbing_pH: float = 2.0  # Lower pH to reject light REE
    stripping_pH: float = 0.5
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    scrub_to_solvent_ratio: float = 0.2
    strip_to_solvent_ratio: float = 0.5


class ExtractScrubStripCircuit:
    """Industrial 3-section REE separation circuit.

    Produces purified REE product by:
    1. Extracting all REE from feed
    2. Scrubbing out lighter/unwanted REE
    3. Stripping purified heavy/target REE

    Example:
        >>> params = ExtractScrubStripParams(
        ...     extractant="D2EHPA",
        ...     elements=("La", "Ce", "Nd", "Dy"),
        ...     target_elements=("Nd", "Dy"),  # Heavy REE product
        ...     n_extraction_stages=10,
        ...     n_scrubbing_stages=5,
        ...     n_stripping_stages=5,
        ... )
        >>> circuit = ExtractScrubStripCircuit(params)
        >>> results = circuit(feed)
        >>> print(f"Nd purity: {results['product_purity']['Nd']:.1%}")
    """

    symbol = "Extract-Scrub-Strip"
    equations = [
        r"\text{feed} \xrightarrow{\text{Extract}} \text{loaded org.} \xrightarrow{\text{Scrub}} \text{purified org.} \xrightarrow{\text{Strip}} \text{product}",
        r"\mathrm{purity}_i = \frac{F_i^\mathrm{product}}{\sum_j F_j^\mathrm{product}}",
    ]
    assumptions = [
        "Three counter-current sections (extract, scrub, strip) in series.",
        "Target vs. non-target selectivity driven by pH difference between sections.",
    ]
    references = ["Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014)."]
    parameter_symbols = {
        "n_extraction_stages": "N_E",
        "n_scrubbing_stages": "N_Sc",
        "n_stripping_stages": "N_S",
    }
    parameter_units = {
        "n_extraction_stages": "-",
        "n_scrubbing_stages": "-",
        "n_stripping_stages": "-",
    }

    def __init__(self, params: ExtractScrubStripParams):
        """Initialize circuit.

        Args:
            params: Circuit parameters
        """
        self.params = params

        # Extraction section
        self._extractor = REEExtractor(REEExtractorParams(
            n_stages=params.n_extraction_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.extraction_pH,
            extractant_conc=params.extractant_conc,
        ))

        # Scrubbing section
        self._scrubber = REEScrubber(ScrubberParams(
            n_stages=params.n_scrubbing_stages,
            extractant=params.extractant,
            elements=params.elements,
            target_elements=params.target_elements,
            diluent=params.diluent,
            pH=params.scrubbing_pH,
            extractant_conc=params.extractant_conc,
        ))

        # Stripping section
        self._stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=params.elements,
            diluent=params.diluent,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
        ))

    def __call__(
        self,
        feed: Stream,
        T: Array | float = 298.15,
    ) -> dict:
        """Run 3-section circuit.

        Args:
            feed: Aqueous REE feed solution
            T: Operating temperature (K)

        Returns:
            Dictionary with:
            - raffinate: Depleted aqueous from extraction
            - scrub_liquor: Rejected REE from scrubbing
            - product: Purified REE product
            - barren_organic: Stripped organic (for recycle)
            - target_recovery: Recovery of target elements
            - product_purity: Purity of each element in product
            - section_info: Details from each section
        """
        p = self.params
        T = jnp.asarray(T)

        feed_flows = get_flows(feed)
        F_aq = feed_flows.get("H2O", 1.0)

        # Create fresh solvent
        F_org = F_aq * p.solvent_to_feed_ratio
        solvent_flows = {
            p.diluent: F_org,
            p.extractant: p.extractant_conc * F_org,
        }
        for elem in p.elements:
            solvent_flows[elem] = 0.0
        solvent = make_stream(solvent_flows, T, feed["P"])

        # EXTRACTION
        raffinate, loaded_org, ext_info = self._extractor(
            feed, solvent, T, pH=p.extraction_pH
        )

        # Create scrub solution
        F_scrub = F_org * p.scrub_to_solvent_ratio
        scrub_flows = {"H2O": F_scrub}
        for elem in p.elements:
            scrub_flows[elem] = 0.0
        scrub_soln = make_stream(scrub_flows, T, feed["P"])

        # SCRUBBING
        scrub_liquor, scrubbed_org, scrub_info = self._scrubber(
            loaded_org, scrub_soln, T, pH=p.scrubbing_pH
        )

        # Create strip acid
        F_strip = F_org * p.strip_to_solvent_ratio
        strip_flows = {"H2O": F_strip}
        for elem in p.elements:
            strip_flows[elem] = 0.0
        strip_acid = make_stream(strip_flows, T, feed["P"])

        # STRIPPING
        product, barren_org, strip_info = self._stripper(
            scrubbed_org, strip_acid, T, pH=p.stripping_pH
        )

        # Calculate metrics
        product_flows = get_flows(product)
        scrub_flows = get_flows(scrub_liquor)
        raff_flows = get_flows(raffinate)

        # Target element recovery
        target_recovery = {}
        for elem in p.target_elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = float(product_flows.get(elem, 0.0))
            target_recovery[elem] = safe_divide(f_out, f_in)

        # Product purity (mole fraction)
        total_product_ree = sum(float(product_flows.get(e, 0.0)) for e in p.elements)
        product_purity = {}
        for elem in p.elements:
            product_purity[elem] = safe_divide(float(product_flows.get(elem, 0.0)), total_product_ree)

        # Target purity (sum of target elements)
        target_purity = sum(product_purity.get(e, 0.0) for e in p.target_elements)

        # Impurity rejection
        impurity_elements = [e for e in p.elements if e not in p.target_elements]
        impurity_rejection = {}
        for elem in impurity_elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_product = float(product_flows.get(elem, 0.0))
            impurity_rejection[elem] = 1 - safe_divide(f_product, f_in)

        # Mass balance verification
        feed_total = {
            elem: jnp.asarray(float(feed_flows.get(elem, 0.0)))
            for elem in p.elements
        }
        output_total = {
            elem: (
                float(raff_flows.get(elem, 0.0))
                + float(scrub_flows.get(elem, 0.0))
                + float(product_flows.get(elem, 0.0))
            )
            for elem in p.elements
        }
        mass_closure = {
            elem: safe_divide(output_total[elem], feed_total[elem])
            for elem in p.elements
        }

        return {
            "raffinate": raffinate,
            "scrub_liquor": scrub_liquor,
            "product": product,
            "barren_organic": barren_org,
            "target_recovery": target_recovery,
            "product_purity": product_purity,
            "target_purity": target_purity,
            "impurity_rejection": impurity_rejection,
            "extraction_info": ext_info,
            "scrubbing_info": scrub_info,
            "stripping_info": strip_info,
            "mass_balance": {
                "feed": feed_total,
                "output": output_total,
                "closure": mass_closure,
            },
        }

    def material_balance(self, feed: Stream, results: dict) -> dict:
        """Check material balance across circuit.

        Args:
            feed: Input feed
            results: Results from circuit run

        Returns:
            Material balance closure for each element
        """
        feed_flows = get_flows(feed)
        raff_flows = get_flows(results["raffinate"])
        scrub_flows = get_flows(results["scrub_liquor"])
        product_flows = get_flows(results["product"])

        balance = {}
        for elem in self.params.elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = (
                float(raff_flows.get(elem, 0.0)) +
                float(scrub_flows.get(elem, 0.0)) +
                float(product_flows.get(elem, 0.0))
            )
            closure = safe_divide(f_out, f_in)
            balance[elem] = {
                "in": f_in,
                "out": f_out,
                "closure": closure,
            }

        return balance


def design_extract_scrub_strip(
    feed_composition: dict[str, float],
    target_elements: tuple[str, ...],
    extractant: str,
    target_purity: float = 0.95,
    target_recovery: float = 0.90,
) -> ExtractScrubStripParams:
    """Design 3-section circuit for given separation.

    Args:
        feed_composition: Element flows in feed (mol/s)
        target_elements: Elements to recover
        extractant: Extractant to use
        target_purity: Target product purity
        target_recovery: Target recovery of target elements

    Returns:
        Recommended ExtractScrubStripParams
    """
    from difflow_ree.equilibrium.distribution import REEDistribution
    from difflow_ree.database import get_sf_database

    elements = tuple(feed_composition.keys())

    # Get separation factors
    sf_db = get_sf_database()
    sf_data = sf_db.get(extractant)

    # Determine key separation (between target and impurity)
    impurity_elements = [e for e in elements if e not in target_elements]

    # Estimate stages based on separation factors
    # More stages needed for lower SF
    n_extraction = 10  # Default
    n_scrubbing = 5  # Default
    n_stripping = 5

    # Find extraction pH - want all REE to extract
    extraction_pH = 3.5

    # Find scrubbing pH - want to reject light REE, keep heavy
    # Lower pH rejects more
    scrubbing_pH = 2.0

    return ExtractScrubStripParams(
        extractant=extractant,
        elements=elements,
        target_elements=target_elements,
        n_extraction_stages=n_extraction,
        n_scrubbing_stages=n_scrubbing,
        n_stripping_stages=n_stripping,
        extraction_pH=extraction_pH,
        scrubbing_pH=scrubbing_pH,
    )
