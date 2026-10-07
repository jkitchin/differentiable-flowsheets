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
    TYPICAL_CEX_CLEARANCE, TYPICAL_AEX_CLEARANCE,
)
from difflow_bio.units.filtration import (
    Ultrafiltration, UltrafiltrationParams,
    Diafiltration, DiafiltrationParams,
)
from difflow_bio.units.centrifuge import (
    DiscStackCentrifuge, DiscStackParams,
)
from difflow_bio.flowsheets.viral_clearance import (
    ViralClearanceTrain, ViralClearanceParams,
)
from difflow.numerics import safe_divide


#: Options PlatformDSP builds. Anything else is rejected at construction
#: (bio audit (b): "mmc" capture silently built a CEX column and an "hic"
#: polish step was silently skipped).
SUPPORTED_CAPTURE = ("protein_a", "cex")
SUPPORTED_POLISH = ("cex", "aex")


@dataclass(repr=False)
class PlatformDSPParams(ParamsMixin):
    """Parameters for platform DSP train.

    Attributes:
        species_order: List of species names
        target_species: Target product name
        capture_type: 'protein_a' or 'cex'. Mixed-mode capture is not
            modelled; any other value raises ValueError.
        polish_steps: Polish step types, in order, each 'cex'
            (bind-elute) or 'aex' (flow-through). Other types (e.g. HIC)
            are not modelled and raise ValueError.
        include_sec: Include SEC polishing step
        include_viral_filtration: Include a virus filtration step after the
            polishing (recovery and LRV from
            :meth:`ViralClearanceTrain.virus_filtration`).
        column_volumes: Dict of step -> column volume (L)
        tff_area: UF membrane area (m²). Sets the UF processing time
            reported when the call is given ``uf_feed_volume_L``; the split
            does not depend on it.
        uf_concentration_factor: Volume concentration factor of the final
            UF step (default 10).
        target_yield: Overall yield the process is expected to reach;
            reported as ``meets_target_yield``.
    """
    species_order: list[str] = None
    target_species: str = "product"
    capture_type: Literal["protein_a", "cex"] = "protein_a"
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
    uf_concentration_factor: float | Array = 10.0
    target_yield: float | Array = 0.70


class PlatformDSP:
    """Generic platform DSP for biopharmaceuticals.

    Flexible process train with configurable:
    - Capture step (Protein A or CEX)
    - Polish steps (any sequence of CEX and AEX)
    - Optional SEC polishing
    - Optional virus filtration
    - UF concentration/formulation

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

        Raises:
            ValueError: An unsupported capture type or polish step.
        """
        self.params = params
        self._steps = []

        if params.capture_type not in SUPPORTED_CAPTURE:
            raise ValueError(
                f"PlatformDSP: capture_type {params.capture_type!r} is not "
                f"modelled; choose one of {SUPPORTED_CAPTURE}."
            )
        unsupported = [s for s in params.polish_steps if s not in SUPPORTED_POLISH]
        if unsupported:
            raise ValueError(
                f"PlatformDSP: polish step(s) {unsupported} are not modelled; "
                f"each polish step must be one of {SUPPORTED_POLISH}."
            )

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

        # Build polish steps. A repeated type gets a numbered name so its
        # yield and side stream are not overwritten by the next one.
        for i, step_type in enumerate(params.polish_steps):
            if step_type == "cex":
                step = IonExchangeChromatography(IEXParams(
                    column_volume=params.column_volumes.get("cex", 15.0),
                    mode="bind_elute",
                    target_species=params.target_species,
                    impurity_clearance=dict(TYPICAL_CEX_CLEARANCE),
                    species_order=params.species_order,
                ))
            else:  # "aex"
                step = IonExchangeChromatography(IEXParams(
                    column_volume=params.column_volumes.get("aex", 12.0),
                    mode="flow_through",
                    target_species=params.target_species,
                    impurity_clearance=dict(TYPICAL_AEX_CLEARANCE),
                    species_order=params.species_order,
                ))
            name = step_type if params.polish_steps.count(step_type) == 1 else f"{step_type}_{i + 1}"
            self._steps.append((name, step))

        # Optional SEC
        if params.include_sec:
            self._sec = SizeExclusionChromatography(SECParams(
                column_volume=params.column_volumes.get("sec", 5.0),
                target_species=params.target_species,
                species_order=params.species_order,
            ))
            self._steps.append(("sec", self._sec))

        # Optional virus filtration (bio audit C10: the flag was never read).
        if params.include_viral_filtration:
            self._vf = ViralClearanceTrain(ViralClearanceParams(
                species_order=params.species_order,
                target_species=params.target_species,
            ))
            self._steps.append(("viral_filtration", self._vf.virus_filtration))

        # TFF
        self._uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=params.tff_area,
            # UF/DF is not credited with impurity clearance: HCP and DNA are retained
            # like the product, and aggregates, larger than the monomer, slightly
            # better. Unlisted species default to zero rejection, which washed
            # 90% of the HCP and aggregates out in the 10x concentration.
            rejection={params.target_species: 0.995, "HCP": 0.995, "DNA": 0.995,
                       "aggregates": 0.999},
            species_order=params.species_order,
        ))

    def __call__(
        self,
        feed: Stream,
        return_intermediates: bool = False,
        uf_feed_volume_L: float | Array = None,
    ) -> dict:
        """Run platform DSP.

        Args:
            feed: Clarified feed stream
            return_intermediates: Return intermediate streams
            uf_feed_volume_L: Volume entering the final UF (L). When given,
                the UF processing time over ``tff_area`` is reported as
                ``uf_process_time_h``.

        Returns:
            Dictionary with ``product``, ``side_streams`` (every other
            outlet, so product + side streams = feed for every species),
            ``overall_yield``, ``step_yields`` (each step's product out over
            its own product in), ``meets_target_yield`` and, with
            ``include_viral_filtration``, ``viral_filtration_lrv``.
        """
        p = self.params
        target = p.target_species

        def amount(stream):
            return get_flows(stream).get(target, jnp.asarray(0.0))

        # Values stay JAX arrays so the train can be traced (#360).
        product_in = amount(feed)

        intermediates = {"feed": feed}
        side_streams = {}
        step_yields = {}
        result = {}
        current_stream = feed

        for step_name, step_unit in self._steps:
            # Load the whole stream: the column volume sets capacity,
            # not how much of the feed is processed. Every unit returns
            # ((product, *side streams), info); SEC has two side streams
            # (aggregates, fragments). Bio audit (b): the side streams were
            # dropped, so the train's outputs did not close.
            (out, *sides), info = step_unit(current_stream)
            if len(sides) == 1:
                side_streams[f"{step_name}_waste"] = sides[0]
            else:
                side_streams["sec_aggregates"], side_streams["sec_fragments"] = sides
            if step_name == "viral_filtration":
                result["viral_filtration_lrv"] = info["lrv"]

            step_yields[step_name] = safe_divide(amount(out), amount(current_stream))
            intermediates[step_name] = out
            current_stream = out

        # Final UF concentration
        (final_product, permeate), uf_info = self._uf(
            current_stream, concentration_factor=p.uf_concentration_factor,
            feed_volume=uf_feed_volume_L)
        side_streams["uf_permeate"] = permeate
        step_yields["uf"] = safe_divide(amount(final_product), amount(current_stream))

        # Calculate overall metrics
        overall_yield = safe_divide(amount(final_product), product_in)

        result.update({
            "product": final_product,
            "side_streams": side_streams,
            "overall_yield": overall_yield,
            "step_yields": step_yields,
            # Bio audit C10: target_yield was documented and never read.
            "meets_target_yield": overall_yield >= p.target_yield,
        })
        if "process_time_h" in uf_info:
            result["uf_process_time_h"] = uf_info["process_time_h"]

        if return_intermediates:
            result["intermediates"] = intermediates

        return result

    def list_steps(self) -> list[str]:
        """List configured process steps."""
        return [name for name, _ in self._steps] + ["uf_concentration"]
