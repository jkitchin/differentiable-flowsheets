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
from difflow.streams import Stream, combine_streams, make_stream, get_flows
from difflow_ree.database import get_extractant
from difflow_ree.equilibrium.distribution import REEDistribution
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
        # Only the members of each group that are in the feed: a circuit
        # run for a group with nothing in it still extracts at its pH, and
        # whatever it extracts would be reported as that group's product.
        self.light_elements = tuple(e for e in light_elements if e in elements)
        self.middle_elements = tuple(e for e in middle_elements if e in elements)
        self.heavy_elements = tuple(e for e in heavy_elements if e in elements)
        self.nitrate_conc = nitrate_conc
        self.mechanism = mechanism
        self.capacity_sharpness = capacity_sharpness

        # Where each section runs. An element leaves for the other phase of a
        # section when D * (O/A) crosses one, so a section separates two
        # groups only at a pH where that crossing falls between them: between
        # the lightest target and the heaviest element it must reject. These
        # pHs used to be fixed fractions of the extractant's fitted window
        # (#270). After the refit moved D2EHPA's window to [0, 2] they no
        # longer sat between the groups -- circuit 1 extracted the light REE
        # at D >= 10, and stripping at the bottom of the window left the heavy
        # REE on the solvent at D ~ 100 -- so the "heavy" product was light
        # REE and the heavies never came off. Reading them off the D curves
        # puts every section at its cut, wherever a refit moves the curves.
        lo, hi = get_extractant(extractant).valid_ph_range
        span = hi - lo
        self._distribution = REEDistribution(
            extractant, elements, nitrate_conc=nitrate_conc,
            mechanism=mechanism, on_out_of_range="ignore")
        heavy_rest = tuple(e for e in elements if e not in self.heavy_elements)
        c1 = self._cut_pHs(self.heavy_elements, heavy_rest,
                           fallback=(lo + 0.500 * span, lo + 0.250 * span, lo))
        c2 = self._cut_pHs(self.middle_elements, self.light_elements,
                           fallback=(lo + 0.625 * span, lo + 0.375 * span, lo))
        c1_extraction_pH, c1_scrubbing_pH, c1_stripping_pH = c1
        c2_extraction_pH, c2_scrubbing_pH, c2_stripping_pH = c2
        self.operating_pH = {
            "heavy_circuit": dict(zip(("extraction", "scrubbing", "stripping"), c1)),
            "middle_circuit": dict(zip(("extraction", "scrubbing", "stripping"), c2)),
        }

        # Circuit 1: Separate heavy from light+middle (skipped without heavies)
        self._heavy_circuit = None if not self.heavy_elements else \
            ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=elements,
            target_elements=self.heavy_elements,
            diluent=diluent,
            n_extraction_stages=10,
            n_scrubbing_stages=8,
            n_stripping_stages=5,
            extraction_pH=c1_extraction_pH,  # the heavy / rest cut
            scrubbing_pH=c1_scrubbing_pH,    # wash co-extracted light + middle
            stripping_pH=c1_stripping_pH,    # strip every heavy
            nitrate_conc=nitrate_conc,  # see #195
            mechanism=mechanism,  # see #195
            capacity_sharpness=capacity_sharpness,  # see #193
        ))

        # Circuit 2: Separate middle from light, on circuit 1's non-heavy
        # outlets (skipped without middles)
        light_middle = tuple(e for e in elements if e not in self.heavy_elements)
        self._middle_circuit = None if not self.middle_elements else \
            ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=light_middle,
            target_elements=self.middle_elements,
            diluent=diluent,
            n_extraction_stages=10,
            n_scrubbing_stages=6,
            n_stripping_stages=5,
            extraction_pH=c2_extraction_pH,
            scrubbing_pH=c2_scrubbing_pH,
            stripping_pH=c2_stripping_pH,
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
        # Each circuit has four outlets. What it does not extract leaves in
        # the raffinate, what scrubbing returns to the aqueous leaves in the
        # scrub liquor, the targets leave in the product, and what stripping
        # leaves on the solvent stays on the barren organic. The rejected
        # stream passed on is raffinate + scrub liquor: it used to be the
        # scrub liquor alone, which dropped the raffinate -- where most of
        # the light REE goes -- on the floor (#371 follow-up).
        losses = []

        def run(circuit, stream):
            if circuit is None:
                return None, stream
            results = circuit(stream, T)
            losses.append(results["barren_organic"])
            return results, combine_streams(results["raffinate"],
                                            results["scrub_liquor"])

        # Circuit 1: extract heavy, reject light + middle
        results1, light_middle_stream = run(self._heavy_circuit, feed)
        heavy = results1["product"] if results1 else self._empty_like(feed)

        # Circuit 2: extract middle, reject light
        results2, light = run(self._middle_circuit, light_middle_stream)
        middle = results2["product"] if results2 else self._empty_like(feed)

        info = {
            "heavy_circuit": results1,
            "middle_circuit": results2,
            # REE still on the stripped solvent: not a product, not lost
            # either (in a plant it recycles with the solvent).
            "solvent_holdup": self._ree_only(losses, feed),
            "group_compositions": {
                "light": self._get_composition(light),
                "middle": self._get_composition(middle),
                "heavy": self._get_composition(heavy),
            },
        }

        return light, middle, heavy, info

    #: D x (O/A) at the boundary of each section: the extraction cut (O/A
    #: 1), the scrub cut (scrub/organic 0.2, the circuit's default ratio) and
    #: the strip, where the most strongly held target must come off with
    #: D x (O/A) = 0.1 at the circuit's strip O/A of 2.
    CUT_D = {"extraction": 1.0, "scrubbing": 0.2, "stripping": 0.05}

    def _pH_where(self, elements: tuple[str, ...], D_target: float,
                  how: str = "mean") -> float:
        """The pH at which D of ``elements`` (their geometric mean, or the
        largest) equals ``D_target``, by bisection; D rises with pH."""
        import math

        def log_D(pH):
            D = self._distribution.get_D_all(pH=pH)
            logs = [math.log10(max(float(D[e]), 1e-300)) for e in elements]
            return max(logs) if how == "max" else sum(logs) / len(logs)

        target = math.log10(D_target)
        a, b = -4.0, 10.0
        for _ in range(80):
            m = 0.5 * (a + b)
            if log_D(m) < target:
                a = m
            else:
                b = m
        return 0.5 * (a + b)

    def _cut_pHs(self, targets: tuple[str, ...], rejected: tuple[str, ...],
                 fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        """Extraction, scrubbing and stripping pH for one circuit.

        The boundary pair is the least extractable target and the most
        extractable element to reject; each section sits where their mean D
        meets that section's cut. A D that does not move with pH (a
        solvating extractant such as TBP, driven by nitrate) has no pH cut,
        and the window fractions are kept.
        """
        if not targets or not rejected:
            return fallback
        D_lo = self._distribution.get_D_all(pH=0.0)
        D_hi = self._distribution.get_D_all(pH=2.0)
        if all(abs(float(D_hi[e]) - float(D_lo[e])) <= 1e-12 * max(abs(float(D_lo[e])), 1.0)
               for e in targets + rejected):
            return fallback
        weakest_target = min(targets, key=lambda e: float(D_hi[e]))
        strongest_rejected = max(rejected, key=lambda e: float(D_hi[e]))
        pair = (weakest_target, strongest_rejected)
        return (self._pH_where(pair, self.CUT_D["extraction"]),
                self._pH_where(pair, self.CUT_D["scrubbing"]),
                self._pH_where(targets, self.CUT_D["stripping"], how="max"))

    def _empty_like(self, feed: Stream) -> Stream:
        """A stream at the feed's conditions carrying no REE."""
        return make_stream({e: 0.0 for e in self.elements}, feed["T"], feed["P"])

    def _ree_only(self, streams: list[Stream], feed: Stream) -> Stream:
        """The REE in ``streams``, summed, as one stream."""
        total = {e: 0.0 for e in self.elements}
        for stream in streams:
            flows = get_flows(stream)
            for e in self.elements:
                total[e] = total[e] + flows.get(e, 0.0)
        return make_stream(total, feed["T"], feed["P"])

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
    parameter_units = {"nitrate_conc": "mol/L", "capacity_sharpness": "-"}

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
            # REE on the stripped solvent: neither product nor lost
            "holdup": {},
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
            results["holdup"] = {"solvent": group_info["solvent_holdup"]}

        # Calculate overall mass balance
        feed_flows = get_flows(feed)
        total_in = sum(float(feed_flows.get(e, 0.0)) for e in self.params.elements)

        total_out = 0.0
        for product_name, product in results["products"].items():
            # A Stream is a dict too, so the type cannot tell a stream from a
            # bare flow dict; the keys can. Streams carry F_<element>, a bare
            # flow dict carries the element names themselves. Testing the type
            # sent every product down the bare-dict branch and counted every
            # stream as zero (#371).
            prod_flows = get_flows(product) or product
            total_out += sum(float(prod_flows.get(e, 0.0)) for e in self.params.elements)
        holdup = sum(float(get_flows(stream).get(e, 0.0))
                     for stream in results["holdup"].values()
                     for e in self.params.elements)

        # closure: every outlet accounted for, products and holdup (should be
        # 1); recovery: the share of the feed that reaches a product.
        results["mass_balance"] = {
            "total_in": total_in,
            "total_out": total_out,
            "holdup": holdup,
            "closure": safe_divide(total_out + holdup, total_in),
            "recovery": safe_divide(total_out, total_in),
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
