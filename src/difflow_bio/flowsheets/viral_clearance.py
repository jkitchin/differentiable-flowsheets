"""Viral clearance focused purification train.

Designed to maximize viral safety with dedicated clearance steps:
1. Low pH hold (inactivation)
2. Nanofiltration (size exclusion removal)
3. Orthogonal chromatography

Provides log reduction value (LRV) tracking throughout.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows
from difflow.numerics import safe_divide


def _split_target(feed: Stream, target: str, recovery) -> tuple[Stream, Stream]:
    """Split ``feed`` into (kept, lost): ``recovery`` of the target is kept.

    Other species pass with the product. The lost stream carries every
    species (zeros but the target) so the two outlets have the same keys.

    Args:
        feed: Inlet stream.
        target: Product species name.
        recovery: Fraction of the target kept, in [0, 1].

    Returns:
        (kept, lost) streams with kept + lost = feed.
    """
    flows = get_flows(feed)
    kept, lost = {}, {}
    for species, flow in flows.items():
        if species == target:
            kept[species] = flow * recovery
            lost[species] = flow - kept[species]
        else:
            kept[species] = flow
            lost[species] = jnp.zeros_like(jnp.asarray(flow, dtype=float))
    return (make_stream(kept, feed["T"], feed["P"]),
            make_stream(lost, feed["T"], feed["P"]))


@dataclass(repr=False)
class ViralClearanceParams(ParamsMixin):
    """Parameters for viral clearance train.

    Attributes:
        species_order: List of species names
        target_species: Target product name
        low_pH_hold_time: Low pH hold duration (min)
        low_pH_value: pH for viral inactivation
        vf_membrane_pore_nm: Virus filter pore size (nm)
        vf_area_m2: Virus filter area (m²). Sizing only: it sets the
            reported mass loading (``info['loading_g_per_m2']``, product
            per m² of filter) for comparison with a filter's rated
            capacity. The model's recovery and LRV do not depend on it.
        target_lrv: Target total log reduction value
    """
    species_order: list[str] = None
    target_species: str = "mAb"
    low_pH_hold_time: float | Array = 60.0  # min
    low_pH_value: float | Array = 3.6
    vf_membrane_pore_nm: float | Array = 20.0  # nm
    vf_area_m2: float | Array = 1.0
    target_lrv: float | Array = 12.0  # Total LRV target


class ViralClearanceTrain:
    """Viral clearance focused purification.

    Implements key viral safety steps:

    1. Low pH inactivation (3.5-3.7 pH, 30-60 min)
       - Effective against enveloped viruses
       - LRV: 4-6 for MuLV, XMuLV

    2. Nanofiltration (20nm pore size)
       - Size-based removal
       - LRV: 4-6 for small non-enveloped viruses

    3. Chromatography steps provide additional clearance
       - Protein A: LRV 2-4
       - IEX: LRV 2-4

    Example:
        >>> params = ViralClearanceParams(
        ...     low_pH_hold_time=60.0,
        ...     vf_area_m2=1.0,
        ... )
        >>> train = ViralClearanceTrain(params)
        >>> results = train(feed)
        >>> print(f"Total LRV: {results['total_lrv']}")
    """

    def __init__(self, params: ViralClearanceParams):
        """Initialize viral clearance train.

        Args:
            params: Viral clearance parameters
        """
        self.params = params

    def low_ph_inactivation(
        self,
        feed: Stream,
        target_virus: str = "MuLV",
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Perform low pH viral inactivation.

        Low pH (3.5-3.7) denatures enveloped virus proteins,
        providing robust inactivation of retrovirus-like particles.

        Args:
            feed: Input stream
            target_virus: Virus model for LRV calculation

        Returns:
            ((output, loss), info): the held product, the product lost in
            the hold (so output + loss = feed), and inactivation info.
        """
        p = self.params

        # LRV depends on pH, time, and temperature
        # Typical: 4-6 LRV for MuLV at pH 3.6, 60 min, 20°C
        base_lrv = 4.0

        # pH factor (lower pH = more inactivation)
        ph_factor = jnp.exp(-0.5 * (p.low_pH_value - 3.6))

        # Time factor (longer = more complete)
        time_factor = 1.0 - jnp.exp(-p.low_pH_hold_time / 30.0)

        lrv = base_lrv * ph_factor * time_factor

        # Product recovery: 98% at the reference pH 3.6, falling by one point
        # per pH unit below it (acid-induced aggregation and precipitation
        # grow as the pH drops). Above pH 3.6 the hold is milder, but no
        # hold returns more than goes in, so recovery stays on the 98%
        # plateau (handling losses), and it is never negative. Bio audit
        # C9: the unbounded line gave 1.014 at pH 7, creating product.
        pH = jnp.asarray(p.low_pH_value)
        recovery = jnp.clip(0.98 - 0.01 * jnp.maximum(3.6 - pH, 0.0), 0.0, 1.0)

        output, loss = _split_target(feed, p.target_species, recovery)

        # Values stay JAX arrays so the train can be traced (#360).
        info = {
            "lrv": lrv,
            "recovery": recovery,
            "pH": p.low_pH_value,
            "hold_time_min": p.low_pH_hold_time,
        }

        return (output, loss), info

    def virus_filtration(
        self,
        feed: Stream,
        target_virus: str = "PPV",
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Perform nanofiltration for virus removal.

        20nm filters provide size-based removal of:
        - Small non-enveloped viruses (PPV, MVM)
        - Large viruses filtered completely

        Args:
            feed: Input stream
            target_virus: Virus model for LRV calculation

        Returns:
            ((output, loss), info): the filtrate, the product held up in
            the filter (output + loss = feed), and filtration info,
            including the mass loading per m² of ``vf_area_m2``.
        """
        p = self.params

        # LRV depends on virus size vs pore size
        # 20nm filter: PPV (18-24nm) -> LRV 4-6
        # Larger viruses: higher LRV
        virus_sizes = {
            "PPV": 22.0,  # nm
            "MVM": 20.0,
            "MuLV": 100.0,  # Enveloped
            "XMuLV": 100.0,
        }

        virus_size = virus_sizes.get(target_virus, 25.0)

        # Size-based LRV. jnp.where, not if/elif: the pore size may be a
        # traced value (#360).
        ratio = virus_size / p.vf_membrane_pore_nm
        lrv = jnp.where(
            ratio > 1.5,
            6.0,  # Complete removal
            jnp.where(ratio > 1.0, 4.0 + 2.0 * (ratio - 1.0), 2.0 * ratio),
        )

        lrv = jnp.clip(lrv, 0.0, 6.0)

        # Recovery (typically 95-99%)
        recovery = 0.97

        output, loss = _split_target(feed, p.target_species, recovery)

        # The area does not enter recovery or LRV in this model (bio audit
        # C9 found it had no effect and was undocumented); it sizes the
        # filter, so report the product loading it implies.
        product_in = get_flows(feed).get(p.target_species, jnp.asarray(0.0))
        info = {
            "lrv": lrv,
            "recovery": recovery,
            "pore_size_nm": p.vf_membrane_pore_nm,
            "area_m2": p.vf_area_m2,
            "loading_g_per_m2": safe_divide(product_in, jnp.asarray(p.vf_area_m2)),
        }

        return (output, loss), info

    def __call__(
        self,
        feed: Stream,
        return_details: bool = True,
    ) -> dict:
        """Run viral clearance train.

        Args:
            feed: Input stream (post-capture)
            return_details: Return detailed step information

        Returns:
            Dictionary with product, LRV totals, step details, and
            ``side_streams`` (the product lost in each step), so that
            product + side streams = feed for every species.
        """
        p = self.params
        target = p.target_species

        feed_flows = get_flows(feed)
        product_in = feed_flows.get(target, 0.0)

        # Step 1: Low pH inactivation
        (post_low_ph, low_ph_loss), low_ph_info = self.low_ph_inactivation(feed)

        # Step 2: Virus filtration
        (post_vf, vf_loss), vf_info = self.virus_filtration(post_low_ph)

        # Calculate totals
        final_flows = get_flows(post_vf)
        product_out = final_flows.get(target, 0.0)
        overall_recovery = safe_divide(product_out, product_in)

        total_lrv = low_ph_info["lrv"] + vf_info["lrv"]

        result = {
            "product": post_vf,
            "side_streams": {
                "low_pH_loss": low_ph_loss,
                "virus_filtration_loss": vf_loss,
            },
            "overall_recovery": overall_recovery,
            "total_lrv": total_lrv,
            "meets_target": total_lrv >= p.target_lrv,
            "lrv_breakdown": {
                "low_pH": low_ph_info["lrv"],
                "nanofiltration": vf_info["lrv"],
            },
        }

        if return_details:
            result["step_details"] = {
                "low_pH_inactivation": low_ph_info,
                "virus_filtration": vf_info,
            }

        return result

    def calculate_lrv_budget(
        self,
        chromatography_lrv: dict = None,
    ) -> dict:
        """Calculate total LRV budget including chromatography.

        Args:
            chromatography_lrv: Dict of step -> LRV contribution

        Returns:
            Complete LRV budget
        """
        if chromatography_lrv is None:
            chromatography_lrv = {
                "protein_a": 3.0,
                "cex": 2.0,
                "aex": 2.0,
            }

        # Dedicated viral clearance
        low_ph_lrv = 4.5  # Typical
        vf_lrv = 4.5  # Typical for 20nm

        total_dedicated = low_ph_lrv + vf_lrv
        total_chromatography = sum(chromatography_lrv.values())
        grand_total = total_dedicated + total_chromatography

        return {
            "low_pH_inactivation": low_ph_lrv,
            "nanofiltration": vf_lrv,
            "dedicated_total": total_dedicated,
            "chromatography": chromatography_lrv,
            "chromatography_total": total_chromatography,
            "grand_total": grand_total,
            "meets_12_lrv": grand_total >= 12.0,
        }
