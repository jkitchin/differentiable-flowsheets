"""Membrane separation units for CO2 capture.

This module provides gas separation membrane models based on the
solution-diffusion mechanism. Both single-stage and multi-stage
configurations are supported.

The models are suitable for:
- Post-combustion CO2 capture (flue gas)
- Natural gas sweetening (CO2/CH4)
- Biogas upgrading
- Pre-combustion hydrogen purification

References:
    Baker RW (2012). Membrane Technology and Applications, 3rd ed.
        Wiley. Chapters 8-9.
    Robeson LM (2008). The upper bound revisited.
        J Membr Sci 320:390-400.
    Merkel TC et al. (2010). Power plant post-combustion carbon
        dioxide capture: An opportunity for membranes.
        J Membr Sci 359:126-139.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow_cc.database import get_membrane, Membrane


# Unit conversions
# 1 Barrer = 10^-10 cm³(STP)·cm / (cm²·s·cmHg)
# 1 GPU = 10^-6 cm³(STP) / (cm²·s·cmHg) = Barrer / thickness(cm)
# Converting to SI: 1 GPU = 3.35e-10 mol/(m²·s·Pa)

GPU_TO_SI = 3.35e-10  # mol/(m²·s·Pa)
BARRER_TO_SI = 3.35e-16  # mol·m/(m²·s·Pa)

# Gas constant
R = 8.314  # J/(mol*K)


# =============================================================================
# Membrane Parameters
# =============================================================================

@dataclass(repr=False)
class MembraneParams(ParamsMixin):
    """Parameters for membrane separator.

    Attributes:
        membrane_type: Membrane material name from database
        area: Membrane area (m²)
        thickness: Membrane thickness (μm), None uses database default
        pressure_ratio: Feed/permeate pressure ratio
        T_operation: Operating temperature (K)
        feed_pressure: Feed pressure (Pa)
        permeate_pressure: Permeate pressure (Pa), calculated if None

    Notes:
        For polymeric membranes, typical thicknesses are 0.1-1 μm
        for thin-film composites on supports.

        Pressure ratio typically 5-20 for gas separation.
        Higher ratios improve recovery but increase compression cost.

        Temperature affects permeability through Arrhenius relation:
            P = P0 * exp(-Ep/(R*T))
    """
    membrane_type: str
    area: float | Array = 1000.0  # m²
    thickness: float | Array | None = None  # μm (uses default if None)
    pressure_ratio: float | Array = 10.0
    T_operation: float | Array = 298.15  # K
    feed_pressure: float | Array = 1000000.0  # Pa (10 bar)
    permeate_pressure: float | Array | None = None  # Pa

    # Stage cut control
    stage_cut_target: float | Array | None = None  # If set, adjusts area






# =============================================================================
# Membrane Separator
# =============================================================================

class MembraneSeparator:
    """Single-stage membrane gas separator.

    Uses solution-diffusion model for gas transport:
        J_i = (P_i / δ) * (p_i,feed - p_i,permeate)

    where J_i is molar flux, P_i is permeability, δ is thickness,
    and p_i are partial pressures.

    Example:
        >>> params = MembraneParams(
        ...     membrane_type='Matrimid',
        ...     area=1000,  # m²
        ...     pressure_ratio=10,
        ... )
        >>> membrane = MembraneSeparator(params)
        >>> retentate, permeate, info = membrane(feed)

    The model outputs:
    - Retentate: Depleted stream (remaining feed side)
    - Permeate: Enriched stream (passed through membrane)
    - Stage cut: Fraction of feed that permeates

    For CO2 capture:
    - Permeate is CO2-enriched
    - Retentate is treated gas

    This model assumes perfect mixing on both sides (simplification).
    For counter-current or cross-flow, see extensibility hooks.

    References:
        Baker RW (2012). Membrane Technology and Applications.
        Wijmans JG, Baker RW (1995). The solution-diffusion model:
            a review. J Membr Sci 107:1-21.
    """

    def __init__(self, params: MembraneParams):
        """Initialize membrane separator.

        Args:
            params: MembraneParams dataclass
        """
        self.params = params
        self._membrane_data = get_membrane(params.membrane_type)

        # Get thickness (use default if not specified)
        if params.thickness is not None:
            self.thickness = params.thickness
        else:
            self.thickness = self._membrane_data.typical_thickness

    def _permeance(self, species: str, T: Array) -> Array:
        """Calculate permeance at temperature T.

        Permeance = Permeability / thickness

        Args:
            species: Gas species name
            T: Temperature (K)

        Returns:
            Permeance in mol/(m²·s·Pa)
        """
        mem = self._membrane_data
        T = jnp.asarray(T)
        T_ref = 298.15

        # Get base permeability
        P_base = mem.permeability.get(species, 0.0)  # Barrer

        # Temperature correction
        if species in mem.activation_energy:
            Ep = mem.activation_energy[species]  # J/mol
            T_factor = jnp.exp(-Ep / R * (1 / T - 1 / T_ref))
        else:
            T_factor = 1.0

        P_T = P_base * T_factor  # Barrer

        # Convert to permeance: mol/(m²·s·Pa)
        # Barrer → SI permeability, then divide by thickness
        thickness_m = jnp.asarray(self.thickness) * 1e-6  # μm to m
        permeance = P_T * BARRER_TO_SI / thickness_m

        return permeance

    def __call__(
        self,
        feed: Stream,
        P_feed: Array | float | None = None,
        P_permeate: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform membrane separation.

        Args:
            feed: Feed gas stream
            P_feed: Feed pressure (Pa), overrides params if provided
            P_permeate: Permeate pressure (Pa), overrides params if provided

        Returns:
            retentate: Retentate stream (feed side, depleted in CO2)
            permeate: Permeate stream (CO2 enriched)
            info: Dict with operation details:
                - stage_cut: Fraction permeated
                - CO2_recovery: CO2 recovery in permeate
                - CO2_purity: CO2 purity in permeate
                - area_used: Membrane area
        """
        p = self.params
        T = jnp.asarray(p.T_operation)
        area = jnp.asarray(p.area)

        # Pressures
        if P_feed is not None:
            P_feed = jnp.asarray(P_feed)
        else:
            P_feed = jnp.asarray(p.feed_pressure)

        if P_permeate is not None:
            P_permeate = jnp.asarray(P_permeate)
        elif p.permeate_pressure is not None:
            P_permeate = jnp.asarray(p.permeate_pressure)
        else:
            P_permeate = P_feed / jnp.asarray(p.pressure_ratio)

        # Get feed composition
        feed_flows = get_flows(feed)
        F_total = total_flow(feed)

        # Calculate mole fractions
        x_feed = {sp: flow / F_total for sp, flow in feed_flows.items()}

        # Calculate fluxes for each species
        fluxes = {}
        permeances = {}
        for species, x_i in x_feed.items():
            Q_i = self._permeance(species, T)  # mol/(m²·s·Pa)
            permeances[species] = Q_i

            # Partial pressure driving force
            # Simplified: assume permeate composition ~ selectivity-weighted
            # For more accurate: iterate to find permeate composition
            p_i_feed = x_i * P_feed

            # First estimate permeate composition (assume perfect mixing)
            # This is iteratively refined in practice
            # For simplified model, assume permeate is enriched by selectivity
            J_i = Q_i * p_i_feed * 0.9  # Approximate (90% of max)
            fluxes[species] = J_i

        # Total flux
        J_total = sum(fluxes.values())

        # Permeate flow
        F_permeate_total = J_total * area

        # Stage cut
        stage_cut = F_permeate_total / (F_total + 1e-10)
        stage_cut = jnp.clip(stage_cut, 0.0, 0.95)  # Physical limit

        # Permeate composition (from flux ratios)
        permeate_flows = {}
        retentate_flows = {}

        for species, flow in feed_flows.items():
            # Permeate flow proportional to flux
            y_perm = fluxes[species] / (J_total + 1e-10)
            F_perm = F_permeate_total * y_perm
            F_perm = jnp.minimum(F_perm, flow * 0.99)  # Can't permeate more than feed

            permeate_flows[species] = F_perm
            retentate_flows[species] = flow - F_perm

        # Calculate performance metrics
        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = permeate_flows.get("CO2", jnp.array(0.0))
        F_perm_total = sum(permeate_flows.values())

        CO2_recovery = F_CO2_perm / (F_CO2_feed + 1e-10)
        CO2_purity = F_CO2_perm / (F_perm_total + 1e-10)

        # Create output streams
        retentate = make_stream(retentate_flows, T, P_feed)
        permeate = make_stream(permeate_flows, T, P_permeate)

        info = {
            "stage_cut": stage_cut,
            "CO2_recovery": CO2_recovery,
            "CO2_purity": CO2_purity,
            "permeate_flow": F_perm_total,
            "retentate_flow": F_total - F_perm_total,
            "area_used": area,
            "pressure_ratio": P_feed / P_permeate,
            "permeances": permeances,
        }

        return retentate, permeate, info

    def required_area(
        self,
        feed: Stream,
        CO2_recovery_target: Array | float,
    ) -> Array:
        """Calculate membrane area for target CO2 recovery.

        Args:
            feed: Feed gas stream
            CO2_recovery_target: Target CO2 recovery (0-1)

        Returns:
            Required membrane area (m²)
        """
        p = self.params
        T = jnp.asarray(p.T_operation)
        P_feed = jnp.asarray(p.feed_pressure)
        P_permeate = P_feed / jnp.asarray(p.pressure_ratio)

        recovery = jnp.asarray(CO2_recovery_target)

        # Get CO2 feed
        feed_flows = get_flows(feed)
        F_CO2 = feed_flows.get("CO2", jnp.array(0.0))
        F_total = total_flow(feed)
        x_CO2 = F_CO2 / F_total

        # CO2 permeance
        Q_CO2 = self._permeance("CO2", T)

        # Required CO2 flux
        F_CO2_perm = F_CO2 * recovery

        # Approximate driving force
        p_CO2_feed = x_CO2 * P_feed
        driving_force = p_CO2_feed * 0.8  # Approximate average

        # Area = F / (Q * ΔP)
        area = F_CO2_perm / (Q_CO2 * driving_force + 1e-10)

        return area


# =============================================================================
# Multi-stage Membrane
# =============================================================================

class MultistageMembrane:
    """Multi-stage membrane cascade.

    Cascades multiple membrane stages for higher purity or recovery
    than achievable with a single stage.

    Common configurations:
    - Two-stage with permeate recycle (for high purity)
    - Two-stage with retentate recycle (for high recovery)
    - Three-stage for both high purity and recovery

    Example:
        >>> params = MembraneParams(membrane_type='Matrimid', area=500)
        >>> cascade = MultistageMembrane(params, n_stages=2)
        >>> retentate, permeate, info = cascade(feed)

    References:
        Merkel TC et al. (2010). Power plant post-combustion CO2
            capture: An opportunity for membranes.
            J Membr Sci 359:126-139.
    """

    def __init__(
        self,
        params: MembraneParams,
        n_stages: int = 2,
        configuration: Literal["series", "permeate_recycle"] = "series"
    ):
        """Initialize multi-stage membrane.

        Args:
            params: Base membrane parameters (area is per stage)
            n_stages: Number of stages
            configuration: 'series' or 'permeate_recycle'
        """
        self.params = params
        self.n_stages = n_stages
        self.configuration = configuration
        self._stages = [MembraneSeparator(params) for _ in range(n_stages)]

    def __call__(
        self,
        feed: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage separation.

        Args:
            feed: Feed gas stream

        Returns:
            retentate: Final retentate (treated gas)
            permeate: Final permeate (CO2 product)
            info: Dict with stage-by-stage results
        """
        if self.configuration == "series":
            return self._series_operation(feed)
        else:
            return self._permeate_recycle_operation(feed)

    def _series_operation(
        self,
        feed: Stream
    ) -> tuple[Stream, Stream, dict]:
        """Series configuration: each stage operates on previous retentate."""
        current_feed = feed
        stage_infos = []
        total_permeate_flows = {}

        for i, stage in enumerate(self._stages):
            retentate, permeate, info = stage(current_feed)
            stage_infos.append(info)

            # Accumulate permeate
            perm_flows = get_flows(permeate)
            for species, flow in perm_flows.items():
                if species in total_permeate_flows:
                    total_permeate_flows[species] = total_permeate_flows[species] + flow
                else:
                    total_permeate_flows[species] = flow

            current_feed = retentate

        # Final streams
        final_retentate = current_feed
        final_permeate = make_stream(
            total_permeate_flows,
            permeate["T"],
            permeate["P"]
        )

        # Overall metrics
        feed_flows = get_flows(feed)
        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = total_permeate_flows.get("CO2", jnp.array(0.0))
        F_perm_total = sum(total_permeate_flows.values())

        overall_info = {
            "n_stages": self.n_stages,
            "configuration": self.configuration,
            "overall_CO2_recovery": F_CO2_perm / (F_CO2_feed + 1e-10),
            "overall_CO2_purity": F_CO2_perm / (F_perm_total + 1e-10),
            "stage_info": stage_infos,
        }

        return final_retentate, final_permeate, overall_info

    def _permeate_recycle_operation(
        self,
        feed: Stream,
    ) -> tuple[Stream, Stream, dict]:
        """Permeate recycle: stage 2 permeate recycled to stage 1.

        This is a simplified implementation without full convergence.
        For rigorous modeling, iterative solution would be needed.
        """
        # Stage 1: feed + recycle
        ret_1, perm_1, info_1 = self._stages[0](feed)

        # Stage 2: operates on stage 1 permeate
        ret_2, perm_2, info_2 = self._stages[1](perm_1)

        # In full implementation, ret_2 would be recycled to stage 1 inlet
        # Simplified: ignore recycle for differentiability

        # Final permeate is stage 2 permeate (highest purity)
        final_permeate = perm_2

        # Final retentate is stage 1 retentate
        final_retentate = ret_1

        # Metrics
        feed_flows = get_flows(feed)
        perm_flows = get_flows(final_permeate)
        F_CO2_feed = feed_flows.get("CO2", jnp.array(0.0))
        F_CO2_perm = perm_flows.get("CO2", jnp.array(0.0))
        F_perm_total = total_flow(final_permeate)

        overall_info = {
            "n_stages": self.n_stages,
            "configuration": self.configuration,
            "overall_CO2_recovery": F_CO2_perm / (F_CO2_feed + 1e-10),
            "overall_CO2_purity": F_CO2_perm / (F_perm_total + 1e-10),
            "stage_info": [info_1, info_2],
        }

        return final_retentate, final_permeate, overall_info
