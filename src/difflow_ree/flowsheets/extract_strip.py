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

from dataclasses import dataclass, replace, fields, asdict as dc_asdict
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_ree.units.extraction import REEExtractor, REEExtractorParams
from difflow_ree.units.stripping import REEStripper, StripperParams


@dataclass
class ExtractStripParams:
    """Parameters for extract-strip circuit.

    Attributes:
        extractant: Extractant name
        elements: REE elements to track
        n_extraction_stages: Number of extraction stages
        n_stripping_stages: Number of stripping stages
        extraction_pH: pH in extraction section
        stripping_pH: pH in stripping section
        extractant_conc: Extractant concentration (M)
        solvent_to_feed_ratio: Organic/aqueous ratio in extraction
        strip_to_solvent_ratio: Strip acid/organic ratio
    """
    extractant: str
    elements: tuple[str, ...]
    n_extraction_stages: int = 10
    n_stripping_stages: int = 5
    extraction_pH: float = 3.5
    stripping_pH: float = 0.5
    extractant_conc: float = 0.5
    solvent_to_feed_ratio: float = 1.0
    strip_to_solvent_ratio: float = 0.5

    def update(self, **kwargs) -> "ExtractStripParams":
        """Return a new ExtractStripParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., n_extraction_stages=12)

        Returns:
            New ExtractStripParams with updated fields
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
            pH=params.extraction_pH,
            extractant_conc=params.extractant_conc,
        ))

        # Create stripping section
        self._stripper = REEStripper(StripperParams(
            n_stages=params.n_stripping_stages,
            extractant=params.extractant,
            elements=params.elements,
            pH=params.stripping_pH,
            extractant_conc=params.extractant_conc,
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
        solvent_flows = {"Organic": F_org}
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
        overall_recovery = total_product / (total_feed + 1e-10)

        # Element-wise recovery
        element_recovery = {}
        for elem in p.elements:
            f_in = float(feed_flows.get(elem, 0.0))
            f_out = float(product_flows.get(elem, 0.0))
            element_recovery[elem] = f_out / (f_in + 1e-10)

        return {
            "raffinate": raffinate,
            "product": product,
            "barren_organic": barren_org,
            "recovery": overall_recovery,
            "element_recovery": element_recovery,
            "extraction_info": ext_info,
            "stripping_info": strip_info,
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
) -> ExtractStripParams:
    """Design extract-strip circuit for given feed.

    Args:
        feed_composition: Element flows in feed (mol/s)
        extractant: Extractant to use
        target_recovery: Target recovery fraction
        extraction_pH: Operating pH

    Returns:
        Recommended ExtractStripParams
    """
    from difflow_ree.equilibrium.distribution import REEDistribution, stages_kremser

    elements = tuple(feed_composition.keys())

    # Get D values at operating pH
    dist = REEDistribution(extractant=extractant, elements=elements)
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
    )
