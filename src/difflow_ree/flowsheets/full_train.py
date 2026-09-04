"""Complete REE separation train.

A full separation train processes mixed REE concentrate through
multiple circuits to produce individual high-purity REE products.

Typical sequence:
1. Ce removal (oxidation)
2. Group separation (Light / Middle / Heavy)
3. Individual element separation within groups

    Mixed REE
        ↓
    ┌─────────┐
    │Ce Oxidn │──▶ CeO2
    └────┬────┘
         ↓
    ┌─────────┐
    │ Group   │──▶ Heavy (Dy, Tb, Gd, Y)
    │ Sep     │──▶ Middle (Sm, Eu)
    └────┬────┘──▶ Light (La, Pr, Nd)
         ↓
    Individual separations...
"""

from dataclasses import dataclass, field

import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_divide
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)


@dataclass(repr=False)
class SeparationTrainParams(ParamsMixin):
    """Parameters for full separation train.

    Attributes:
        elements: All REE elements in feed
        extractant: Primary extractant
        diluent: Organic diluent name (e.g., "kerosene", "n-dodecane")
        include_ce_removal: Whether to include Ce oxidation step
        group_separation: Whether to separate into light/middle/heavy
        individual_separation: Whether to separate individual elements
        target_purities: Target purity for each element
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP whose D is nitrate- rather than pH-driven
            (#195). None for the acidic cation-exchange extractants. Threaded
            into every circuit the train builds.
        mechanism: Explicit extraction-mechanism override passed to
            REEDistribution ("cation_exchange" / "solvating"). None takes the
            mechanism from the extractant record (#195). Threaded into every
            section of every circuit the train builds, so no circuit mixes
            mechanisms.
        capacity_sharpness: Sharpness k of the smooth loading limiters in each
            circuit's extraction section; see REEExtractorParams (#193).
    """
    elements: tuple[str, ...] = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y")
    extractant: str = "D2EHPA"
    secondary_extractant: str = "PC88A"  # For Nd/Pr separation
    diluent: str = "kerosene"
    include_ce_removal: bool = True
    group_separation: bool = True
    individual_separation: bool = False  # Full individual sep is complex
    nitrate_conc: float | None = None  # see #195
    mechanism: str | None = None  # see #195
    capacity_sharpness: int = 8  # see REEExtractorParams (#193)
    target_purities: dict = field(default_factory=lambda: {
        "Nd": 0.99,
        "Dy": 0.99,
        "Y": 0.95,
    })


class GroupSeparator:
    """Separates REE into light, middle, and heavy groups.

    Uses extract-scrub-strip configuration to split:
    - Heavy (Dy, Tb, Gd, Y): High D values, extract at low pH
    - Middle (Sm, Eu): Intermediate D values
    - Light (La, Ce, Pr, Nd): Low D values, stay in raffinate

    Example:
        >>> separator = GroupSeparator(
        ...     elements=("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Y"),
        ...     extractant="D2EHPA",
        ... )
        >>> light, middle, heavy = separator(feed)
    """

    symbol = "Group Separator"
    equations = [
        r"\text{Two circuits in series: (i) heavy vs.\ light+middle, (ii) middle vs.\ light}",
        r"D_\mathrm{heavy} \gg D_\mathrm{middle} \gg D_\mathrm{light}\qquad \text{(typical D2EHPA selectivity)}",
    ]
    assumptions = [
        "Two extract-scrub-strip circuits partition the feed into L / M / H groups.",
        "Group boundaries selected by pH/extractant-concentration tuning.",
    ]
    references = ["Habashi, F. Handbook of Extractive Metallurgy, Vol. 3, Wiley-VCH, 1997."]
    parameter_symbols = {}
    parameter_units = {}

    def __init__(
        self,
        elements: tuple[str, ...],
        extractant: str = "D2EHPA",
        diluent: str = "kerosene",
        light_elements: tuple[str, ...] = ("La", "Ce", "Pr", "Nd"),
        middle_elements: tuple[str, ...] = ("Sm", "Eu"),
        heavy_elements: tuple[str, ...] = ("Gd", "Tb", "Dy", "Y"),
        nitrate_conc: float | None = None,
        mechanism: str | None = None,
        capacity_sharpness: int = 8,
    ):
        """Initialize separator.

        Args:
            elements: All elements to process
            extractant: Extractant to use
            diluent: Organic diluent name
            light_elements: Elements for light group
            middle_elements: Elements for middle group
            heavy_elements: Elements for heavy group
            nitrate_conc: Aqueous nitrate concentration (M), required for
                solvating extractants such as TBP (#195)
            mechanism: Explicit mechanism override; see REEDistribution (#195)
            capacity_sharpness: Sharpness k of the extraction sections' smooth
                loading limiters; see REEExtractorParams (#193)
        """
        self.elements = elements
        self.extractant = extractant
        self.diluent = diluent
        self.light_elements = light_elements
        self.middle_elements = middle_elements
        self.heavy_elements = heavy_elements
        self.nitrate_conc = nitrate_conc
        self.mechanism = mechanism
        self.capacity_sharpness = capacity_sharpness

        # Circuit 1: Separate heavy from light+middle
        self._heavy_circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=elements,
            target_elements=heavy_elements,
            diluent=diluent,
            n_extraction_stages=10,
            n_scrubbing_stages=8,
            n_stripping_stages=5,
            extraction_pH=3.0,  # All extract
            scrubbing_pH=2.0,   # Reject light+middle
            nitrate_conc=nitrate_conc,  # see #195
            mechanism=mechanism,  # see #195
            capacity_sharpness=capacity_sharpness,  # see #193
        ))

        # Circuit 2: Separate middle from light (on Circuit 1 scrub liquor)
        light_middle = light_elements + middle_elements
        self._middle_circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=light_middle,
            target_elements=middle_elements,
            diluent=diluent,
            n_extraction_stages=10,
            n_scrubbing_stages=6,
            n_stripping_stages=5,
            extraction_pH=3.5,
            scrubbing_pH=2.5,
            nitrate_conc=nitrate_conc,  # see #195
            mechanism=mechanism,  # see #195
            capacity_sharpness=capacity_sharpness,  # see #193
        ))

    def __call__(
        self,
        feed: Stream,
        T: float = 298.15,
    ) -> tuple[Stream, Stream, Stream, dict]:
        """Perform group separation.

        Args:
            feed: Mixed REE feed
            T: Temperature (K)

        Returns:
            light: Light REE stream (La, Ce, Pr, Nd)
            middle: Middle REE stream (Sm, Eu)
            heavy: Heavy REE stream (Gd, Tb, Dy, Y)
            info: Separation details
        """
        # Circuit 1: Extract heavy, reject light+middle
        results1 = self._heavy_circuit(feed, T)
        heavy = results1["product"]  # Heavy REE product
        light_middle_stream = results1["scrub_liquor"]  # Contains light + middle

        # Circuit 2: Extract middle, reject light
        results2 = self._middle_circuit(light_middle_stream, T)
        middle = results2["product"]  # Middle REE product
        light = results2["scrub_liquor"]  # Light REE

        info = {
            "heavy_circuit": results1,
            "middle_circuit": results2,
            "group_compositions": {
                "light": self._get_composition(light),
                "middle": self._get_composition(middle),
                "heavy": self._get_composition(heavy),
            },
        }

        return light, middle, heavy, info

    def _get_composition(self, stream: Stream) -> dict[str, float]:
        """Get REE composition of stream."""
        flows = get_flows(stream)
        total = sum(flows.get(e, 0.0) for e in self.elements)
        return {
            e: safe_divide(float(flows.get(e, 0.0)), float(total))
            for e in self.elements
        }


class FullSeparationTrain:
    """Complete REE separation plant.

    Processes mixed REE through:
    1. Optional Ce removal
    2. Group separation (Light/Middle/Heavy)
    3. Optional individual element separation

    Example:
        >>> params = SeparationTrainParams(
        ...     elements=("La", "Ce", "Pr", "Nd", "Sm", "Gd", "Dy", "Y"),
        ...     include_ce_removal=True,
        ...     group_separation=True,
        ... )
        >>> train = FullSeparationTrain(params)
        >>> results = train(feed)
    """

    symbol = "Full Separation Train"
    equations = [
        r"\text{feed} \rightarrow \text{Ce removal (opt.)} \rightarrow \text{group separation} \rightarrow \text{individual elements (opt.)}",
    ]
    assumptions = [
        "Pre-defined network of REE unit operations with optional Ce-removal and individual separation.",
    ]
    references = ["Xie, F., Zhang, T.A., Dreisinger, D., Doyle, F. Miner. Eng., 56, 10 (2014)."]
    parameter_symbols = {}
    parameter_units = {}

    def __init__(self, params: SeparationTrainParams):
        """Initialize separation train.

        Args:
            params: Train parameters
        """
        self.params = params

        # Ce removal unit (if Ce in elements)
        if params.include_ce_removal and "Ce" in params.elements:
            self._ce_oxidizer = CeriumOxidizer(CeriumOxidizerParams(
                elements=params.elements,
                oxidant="air",
                pH=8.0,
            ))
        else:
            self._ce_oxidizer = None

        # Group separator
        if params.group_separation:
            # Determine groups based on elements present
            light = tuple(e for e in ("La", "Ce", "Pr", "Nd") if e in params.elements)
            middle = tuple(e for e in ("Sm", "Eu") if e in params.elements)
            heavy = tuple(e for e in ("Gd", "Tb", "Dy", "Y") if e in params.elements)

            self._group_separator = GroupSeparator(
                elements=params.elements,
                extractant=params.extractant,
                diluent=params.diluent,
                light_elements=light,
                middle_elements=middle,
                heavy_elements=heavy,
                nitrate_conc=params.nitrate_conc,  # see #195
                mechanism=params.mechanism,  # see #195
                capacity_sharpness=params.capacity_sharpness,  # see #193
            )
        else:
            self._group_separator = None

    def __call__(
        self,
        feed: Stream,
        T: float = 298.15,
    ) -> dict:
        """Run separation train.

        Args:
            feed: Mixed REE feed
            T: Temperature (K)

        Returns:
            Dictionary with all products and process info
        """
        results = {
            "feed": feed,
            "products": {},
            "intermediates": {},
            "info": {},
        }

        current_stream = feed

        # Step 1: Ce removal
        if self._ce_oxidizer is not None:
            ce_depleted, ceo2_solid, ce_info = self._ce_oxidizer(current_stream, T)
            results["products"]["CeO2"] = ceo2_solid
            results["info"]["ce_removal"] = ce_info
            current_stream = ce_depleted
            results["intermediates"]["ce_depleted"] = ce_depleted

        # Step 2: Group separation
        if self._group_separator is not None:
            light, middle, heavy, group_info = self._group_separator(current_stream, T)
            results["products"]["light_REE"] = light
            results["products"]["middle_REE"] = middle
            results["products"]["heavy_REE"] = heavy
            results["info"]["group_separation"] = group_info

        # Calculate overall mass balance
        feed_flows = get_flows(feed)
        total_in = sum(float(feed_flows.get(e, 0.0)) for e in self.params.elements)

        total_out = 0.0
        for product_name, product in results["products"].items():
            if isinstance(product, dict):  # Solid product
                total_out += sum(float(product.get(e, 0.0)) for e in self.params.elements)
            else:  # Stream
                prod_flows = get_flows(product)
                total_out += sum(float(prod_flows.get(e, 0.0)) for e in self.params.elements)

        results["mass_balance"] = {
            "total_in": total_in,
            "total_out": total_out,
            "closure": safe_divide(total_out, total_in),
        }

        return results


def design_separation_train(
    feed_analysis: dict[str, float],
    target_products: list[str],
    annual_capacity_tonnes: float = 1000,
    nitrate_conc: float | None = None,
    mechanism: str | None = None,
) -> SeparationTrainParams:
    """Design separation train for given feed and products.

    Args:
        feed_analysis: REE composition in feed (wt%)
        target_products: List of target product elements
        annual_capacity_tonnes: Annual REE production capacity
        nitrate_conc: Aqueous nitrate concentration (M), required for solvating
            extractants such as TBP (#195)
        mechanism: Explicit mechanism override; see REEDistribution (#195)

    Returns:
        Recommended SeparationTrainParams
    """
    elements = tuple(feed_analysis.keys())

    # Determine if Ce removal beneficial
    ce_fraction = feed_analysis.get("Ce", 0) / 100
    include_ce = ce_fraction > 0.3  # If Ce > 30%, remove it first

    # Determine if group separation sufficient or need individual
    individual_sep = any(
        e in target_products
        for e in ["Nd", "Pr", "Eu", "Tb"]  # High-value elements needing purity
    )

    return SeparationTrainParams(
        elements=elements,
        include_ce_removal=include_ce,
        group_separation=True,
        individual_separation=individual_sep,
        nitrate_conc=nitrate_conc,  # see #195
        mechanism=mechanism,  # see #195
    )
