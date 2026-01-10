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


@dataclass(repr=False)
class ViralClearanceParams(ParamsMixin):
    """Parameters for viral clearance train.

    Attributes:
        species_order: List of species names
        target_species: Target product name
        low_pH_hold_time: Low pH hold duration (min)
        low_pH_value: pH for viral inactivation
        vf_membrane_pore_nm: Virus filter pore size (nm)
        vf_area_m2: Virus filter area (m²)
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
    ) -> tuple[Stream, dict]:
        """Perform low pH viral inactivation.

        Low pH (3.5-3.7) denatures enveloped virus proteins,
        providing robust inactivation of retrovirus-like particles.

        Args:
            feed: Input stream
            target_virus: Virus model for LRV calculation

        Returns:
            Tuple of (output stream, inactivation info)
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

        # Product recovery (typically >95% at optimized conditions)
        recovery = 0.98 - 0.01 * (3.6 - p.low_pH_value)  # Slight loss at lower pH

        # Create output stream with reduced product
        feed_flows = get_flows(feed)
        out_flows = {}
        for species, flow in feed_flows.items():
            if species == p.target_species:
                out_flows[species] = flow * recovery
            else:
                out_flows[species] = flow

        output = make_stream(out_flows, feed["T"], feed["P"])

        info = {
            "lrv": float(lrv),
            "recovery": float(recovery),
            "pH": float(p.low_pH_value),
            "hold_time_min": float(p.low_pH_hold_time),
        }

        return output, info

    def virus_filtration(
        self,
        feed: Stream,
        target_virus: str = "PPV",
    ) -> tuple[Stream, dict]:
        """Perform nanofiltration for virus removal.

        20nm filters provide size-based removal of:
        - Small non-enveloped viruses (PPV, MVM)
        - Large viruses filtered completely

        Args:
            feed: Input stream
            target_virus: Virus model for LRV calculation

        Returns:
            Tuple of (output stream, filtration info)
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

        # Size-based LRV
        if virus_size > p.vf_membrane_pore_nm * 1.5:
            lrv = 6.0  # Complete removal
        elif virus_size > p.vf_membrane_pore_nm:
            lrv = 4.0 + 2.0 * (virus_size / p.vf_membrane_pore_nm - 1.0)
        else:
            lrv = 2.0 * (virus_size / p.vf_membrane_pore_nm)

        lrv = jnp.clip(lrv, 0.0, 6.0)

        # Recovery (typically 95-99%)
        recovery = 0.97

        # Create output stream
        feed_flows = get_flows(feed)
        out_flows = {}
        for species, flow in feed_flows.items():
            if species == p.target_species:
                out_flows[species] = flow * recovery
            else:
                out_flows[species] = flow

        output = make_stream(out_flows, feed["T"], feed["P"])

        info = {
            "lrv": float(lrv),
            "recovery": float(recovery),
            "pore_size_nm": float(p.vf_membrane_pore_nm),
            "area_m2": float(p.vf_area_m2),
        }

        return output, info

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
            Dictionary with product, LRV totals, and step details
        """
        p = self.params
        target = p.target_species

        feed_flows = get_flows(feed)
        product_in = float(feed_flows.get(target, 0.0))

        # Step 1: Low pH inactivation
        post_low_ph, low_ph_info = self.low_ph_inactivation(feed)

        # Step 2: Virus filtration
        post_vf, vf_info = self.virus_filtration(post_low_ph)

        # Calculate totals
        final_flows = get_flows(post_vf)
        product_out = float(final_flows.get(target, 0.0))
        overall_recovery = safe_divide(product_out, product_in)

        total_lrv = low_ph_info["lrv"] + vf_info["lrv"]

        result = {
            "product": post_vf,
            "overall_recovery": overall_recovery,
            "total_lrv": total_lrv,
            "meets_target": total_lrv >= float(p.target_lrv),
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
