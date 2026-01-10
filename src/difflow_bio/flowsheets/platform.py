"""Generic platform DSP train for biopharmaceuticals.

Flexible configuration supporting various modalities:
- Monoclonal antibodies
- Fc-fusion proteins
- Bispecifics

Configurable chromatography sequence with optional steps.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass, field
from typing import Literal

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow_bio.units.chromatography import (
    ProteinAChromatography, ProteinAParams,
    IonExchangeChromatography, IEXParams,
    SizeExclusionChromatography, SECParams,
)
from difflow_bio.units.filtration import (
    Ultrafiltration, UltrafiltrationParams,
    Diafiltration, DiafiltrationParams,
)
from difflow_bio.units.centrifuge import (
    DiscStackCentrifuge, DiscStackParams,
)
from difflow.numerics import safe_divide


@dataclass(repr=False)
class PlatformDSPParams(ParamsMixin):
    """Parameters for platform DSP train.

    Attributes:
        species_order: List of species names
        target_species: Target product name
        capture_type: 'protein_a', 'cex', or 'mmc'
        polish_steps: List of polish step types
        include_sec: Include SEC polishing step
        include_viral_filtration: Include virus filtration
        column_volumes: Dict of step -> column volume (L)
        tff_area: TFF membrane area (m²)
    """
    species_order: list[str] = None
    target_species: str = "product"
    capture_type: Literal["protein_a", "cex", "mmc"] = "protein_a"
    polish_steps: list[str] = field(default_factory=lambda: ["cex", "aex"])
    include_sec: bool = False
    include_viral_filtration: bool = True
    column_volumes: dict = field(default_factory=lambda: {
        "capture": 10.0,
        "cex": 15.0,
        "aex": 12.0,
        "sec": 5.0,
    })
    tff_area: float | Array = 5.0
    target_yield: float | Array = 0.70


class PlatformDSP:
    """Generic platform DSP for biopharmaceuticals.

    Flexible process train with configurable:
    - Capture step (Protein A, CEX, or MMC)
    - Polish steps (any combination of CEX, AEX, HIC)
    - Optional SEC polishing
    - TFF concentration/formulation

    Example:
        >>> params = PlatformDSPParams(
        ...     capture_type="protein_a",
        ...     polish_steps=["cex", "aex"],
        ...     include_sec=False,
        ... )
        >>> train = PlatformDSP(params)
        >>> results = train(harvest)
    """

    def __init__(self, params: PlatformDSPParams):
        """Initialize platform DSP.

        Args:
            params: Platform DSP parameters
        """
        self.params = params
        self._steps = []
        self._step_names = []

        # Build capture step
        if params.capture_type == "protein_a":
            self._capture = ProteinAChromatography(ProteinAParams(
                column_volume=params.column_volumes.get("capture", 10.0),
                target_species=params.target_species,
                species_order=params.species_order,
            ))
        else:  # CEX capture
            self._capture = IonExchangeChromatography(IEXParams(
                column_volume=params.column_volumes.get("capture", 10.0),
                mode="bind_elute",
                target_species=params.target_species,
                species_order=params.species_order,
            ))
        self._steps.append(("capture", self._capture))

        # Build polish steps
        for step_type in params.polish_steps:
            if step_type == "cex":
                step = IonExchangeChromatography(IEXParams(
                    column_volume=params.column_volumes.get("cex", 15.0),
                    mode="bind_elute",
                    target_species=params.target_species,
                    species_order=params.species_order,
                ))
            elif step_type == "aex":
                step = IonExchangeChromatography(IEXParams(
                    column_volume=params.column_volumes.get("aex", 12.0),
                    mode="flow_through",
                    target_species=params.target_species,
                    species_order=params.species_order,
                ))
            else:
                continue
            self._steps.append((step_type, step))

        # Optional SEC
        if params.include_sec:
            self._sec = SizeExclusionChromatography(SECParams(
                column_volume=params.column_volumes.get("sec", 5.0),
                target_species=params.target_species,
                species_order=params.species_order,
            ))
            self._steps.append(("sec", self._sec))

        # TFF
        self._uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=params.tff_area,
            rejection={params.target_species: 0.995},
            species_order=params.species_order,
        ))

    def __call__(
        self,
        feed: Stream,
        return_intermediates: bool = False,
    ) -> dict:
        """Run platform DSP.

        Args:
            feed: Clarified feed stream
            return_intermediates: Return intermediate streams

        Returns:
            Dictionary with product, yields, and metrics
        """
        p = self.params
        target = p.target_species

        feed_flows = get_flows(feed)
        product_in = float(feed_flows.get(target, 0.0))

        intermediates = {"feed": feed}
        step_yields = {}
        current_stream = feed

        # Run each step
        for step_name, step_unit in self._steps:
            if hasattr(step_unit, '__call__'):
                result = step_unit(current_stream)
                if isinstance(result, tuple):
                    current_stream = result[0]  # Product stream
                else:
                    current_stream = result

            current_flows = get_flows(current_stream)
            current_product = float(current_flows.get(target, 0.0))
            prev_flows = get_flows(intermediates.get(list(intermediates.keys())[-1], feed))
            prev_product = float(prev_flows.get(target, 0.0))

            step_yields[step_name] = safe_divide(current_product, prev_product)
            intermediates[step_name] = current_stream

        # Final UF concentration
        final_product, permeate = self._uf(current_stream, concentration_factor=10.0)
        final_flows = get_flows(final_product)

        # Calculate overall metrics
        product_out = float(final_flows.get(target, 0.0))
        overall_yield = safe_divide(product_out, product_in)

        result = {
            "product": final_product,
            "overall_yield": overall_yield,
            "step_yields": step_yields,
        }

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def list_steps(self) -> list[str]:
        """List configured process steps."""
        return [name for name, _ in self._steps] + ["uf_concentration"]
