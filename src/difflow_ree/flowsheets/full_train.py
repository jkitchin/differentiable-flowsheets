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

from dataclasses import dataclass, field, replace, fields, asdict as dc_asdict
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.cerium import CeriumOxidizer, CeriumOxidizerParams
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)


@dataclass
class SeparationTrainParams:
    """Parameters for full separation train.

    Attributes:
        elements: All REE elements in feed
        extractant: Primary extractant
        include_ce_removal: Whether to include Ce oxidation step
        group_separation: Whether to separate into light/middle/heavy
        individual_separation: Whether to separate individual elements
        target_purities: Target purity for each element
    """
    elements: tuple[str, ...] = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y")
    extractant: str = "D2EHPA"
    secondary_extractant: str = "PC88A"  # For Nd/Pr separation
    include_ce_removal: bool = True
    group_separation: bool = True
    individual_separation: bool = False  # Full individual sep is complex
    target_purities: dict = field(default_factory=lambda: {
        "Nd": 0.99,
        "Dy": 0.99,
        "Y": 0.95,
    })

    def update(self, **kwargs) -> "SeparationTrainParams":
        """Return a new SeparationTrainParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., include_ce_removal=False)

        Returns:
            New SeparationTrainParams with updated fields
        """
        return replace(self, **kwargs)

    def __getitem__(self, key: str):
        """Get parameter value by name for dict-like access."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        """Check if a field exists in the params."""
        return key in {f.name for f in fields(self)}

    def keys(self):
        """Return field names for dict-like iteration."""
        return (f.name for f in fields(self))

    def values(self):
        """Return field values for dict-like iteration.

        Returns:
            Iterator over field values
        """
        return (getattr(self, f.name) for f in fields(self))

    def items(self):
        """Return (name, value) pairs for dict-like iteration.

        Returns:
            Iterator over (field_name, value) tuples
        """
        return ((f.name, getattr(self, f.name)) for f in fields(self))

    def __iter__(self):
        """Iterate over field names (like dict)."""
        return (f.name for f in fields(self))

    def __len__(self) -> int:
        """Return number of fields."""
        return len(fields(self))

    def asdict(self) -> dict:
        """Convert params to a dictionary."""
        return dc_asdict(self)

    def __repr__(self) -> str:
        """Concise string representation."""
        def fmt(v):
            if v is None:
                return "None"
            if callable(v) and hasattr(v, '__name__'):
                return v.__name__
            if hasattr(v, 'shape'):
                if v.ndim == 0:
                    return f"{float(v):.4g}"
                return f"Array{list(v.shape)}"
            if isinstance(v, dict):
                items = ", ".join(f"{k}: {fmt(val)}" for k, val in v.items())
                return "{" + items + "}"
            if isinstance(v, (list, tuple)) and len(v) > 5:
                return f"{type(v).__name__}[{len(v)}]"
            return repr(v)
        items = ", ".join(f"{f.name}={fmt(getattr(self, f.name))}" for f in fields(self))
        return f"{self.__class__.__name__}({items})"


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

    def __init__(
        self,
        elements: tuple[str, ...],
        extractant: str = "D2EHPA",
        light_elements: tuple[str, ...] = ("La", "Ce", "Pr", "Nd"),
        middle_elements: tuple[str, ...] = ("Sm", "Eu"),
        heavy_elements: tuple[str, ...] = ("Gd", "Tb", "Dy", "Y"),
    ):
        """Initialize separator.

        Args:
            elements: All elements to process
            extractant: Extractant to use
            light_elements: Elements for light group
            middle_elements: Elements for middle group
            heavy_elements: Elements for heavy group
        """
        self.elements = elements
        self.extractant = extractant
        self.light_elements = light_elements
        self.middle_elements = middle_elements
        self.heavy_elements = heavy_elements

        # Circuit 1: Separate heavy from light+middle
        self._heavy_circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=elements,
            target_elements=heavy_elements,
            n_extraction_stages=10,
            n_scrubbing_stages=8,
            n_stripping_stages=5,
            extraction_pH=3.0,  # All extract
            scrubbing_pH=2.0,   # Reject light+middle
        ))

        # Circuit 2: Separate middle from light (on Circuit 1 scrub liquor)
        light_middle = light_elements + middle_elements
        self._middle_circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
            extractant=extractant,
            elements=light_middle,
            target_elements=middle_elements,
            n_extraction_stages=10,
            n_scrubbing_stages=6,
            n_stripping_stages=5,
            extraction_pH=3.5,
            scrubbing_pH=2.5,
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
            e: float(flows.get(e, 0.0)) / (float(total) + 1e-10)
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
                light_elements=light,
                middle_elements=middle,
                heavy_elements=heavy,
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
            "closure": total_out / (total_in + 1e-10),
        }

        return results


def design_separation_train(
    feed_analysis: dict[str, float],
    target_products: list[str],
    annual_capacity_tonnes: float = 1000,
) -> SeparationTrainParams:
    """Design separation train for given feed and products.

    Args:
        feed_analysis: REE composition in feed (wt%)
        target_products: List of target product elements
        annual_capacity_tonnes: Annual REE production capacity

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
    )
