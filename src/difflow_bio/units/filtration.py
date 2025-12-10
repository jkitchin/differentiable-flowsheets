"""Membrane filtration unit operations for protein processing.

This module provides ultrafiltration and diafiltration models:
- Ultrafiltration: Concentration of proteins by membrane separation
- Diafiltration: Buffer exchange with UF membrane

Key equations:
    Flux: J = TMP / (μ * R_total)
    Rejection: R = 1 - C_permeate / C_retentate
    Concentration factor: CF = V_initial / V_final
    Diafiltration: C/C_0 = exp(-N_dv * (1-R)) for permeable solutes

where:
    J = permeate flux (L/m²/h or LMH)
    TMP = transmembrane pressure (bar)
    R = rejection coefficient
    N_dv = number of diavolumes
"""

from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, make_stream, get_flows


# =============================================================================
# Filtration Parameters
# =============================================================================

@dataclass
class UltrafiltrationParams:
    """Parameters for ultrafiltration.

    Attributes:
        membrane_area: Membrane area (m²)
        MWCO: Molecular weight cutoff (kDa)
        rejection: Dict of species name -> rejection coefficient (0-1)
                  Species not listed assumed to have R=0 (fully permeable)
        Lp: Membrane permeability (L/m²/h/bar), optional
        species_order: List of species names
    """
    membrane_area: float | Array
    MWCO: float | Array = 30.0  # kDa, typical for mAb
    rejection: dict = field(default_factory=dict)
    Lp: float | Array = 50.0  # L/m²/h/bar, typical for UF membrane
    species_order: list[str] = None


@dataclass
class DiafiltrationParams:
    """Parameters for diafiltration (buffer exchange).

    Attributes:
        membrane_area: Membrane area (m²)
        MWCO: Molecular weight cutoff (kDa)
        rejection: Dict of species -> rejection coefficient
        Lp: Membrane permeability (L/m²/h/bar)
        species_order: List of species names
    """
    membrane_area: float | Array
    MWCO: float | Array = 30.0
    rejection: dict = field(default_factory=dict)
    Lp: float | Array = 50.0
    species_order: list[str] = None


# =============================================================================
# Ultrafiltration
# =============================================================================

class Ultrafiltration:
    """Ultrafiltration for protein concentration.

    Uses a semi-permeable membrane to concentrate proteins (retained)
    while allowing water and small molecules to pass through (permeate).

    Operates in batch concentration mode or continuous mode.
    """

    def __init__(self, params: UltrafiltrationParams):
        """Initialize ultrafiltration unit.

        Args:
            params: UF parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        concentration_factor: float | Array,
        TMP: float | Array = 1.0,
        mode: str = "batch",
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform ultrafiltration.

        Args:
            inlet: Feed stream
            concentration_factor: Target CF = V_in / V_retentate
            TMP: Transmembrane pressure (bar)
            mode: "batch" for batch concentration, "continuous" for steady-state

        Returns:
            (retentate, permeate): Tuple of outlet streams
            info: Dictionary with:
                - 'flux': Permeate flux (L/m²/h)
                - 'recovery': Product recovery in retentate
                - 'volume_reduction': V_permeate / V_feed
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        CF = jnp.asarray(concentration_factor)

        # Calculate volumes (assuming unit density for simplicity)
        # Total inlet flow represents volume per unit time
        total_flow_in = sum(inlet_flows.values())

        # Volume fractions
        volume_reduction = 1.0 - 1.0 / CF
        retentate_volume_frac = 1.0 / CF
        permeate_volume_frac = volume_reduction

        # Permeate flux (simplified model)
        J = p.Lp * TMP  # L/m²/h

        # Split species based on rejection
        retentate_flows = {}
        permeate_flows = {}

        for species, flow in inlet_flows.items():
            R = p.rejection.get(species, 0.0)  # Default: fully permeable
            R = jnp.asarray(R)

            if mode == "batch":
                # Batch concentration: apply rejection with concentration
                # For batch: C_ret/C_0 = CF^R (for R=1, fully retained)
                # Mass balance: M_ret = M_0 - M_perm
                # Permeate gets (1-R) fraction of what passes through

                # Simplified: retained fraction = 1 - (1-R)*volume_reduction
                retained_frac = 1.0 - (1.0 - R) * volume_reduction
                retained_frac = jnp.clip(retained_frac, 0.0, 1.0)

                retentate_flows[species] = flow * retained_frac
                permeate_flows[species] = flow * (1.0 - retained_frac)

            else:  # continuous mode
                # Steady state: simple rejection split
                retentate_flows[species] = flow * retentate_volume_frac * (1.0 + R * (CF - 1.0))
                permeate_flows[species] = flow - retentate_flows[species]
                permeate_flows[species] = jnp.maximum(permeate_flows[species], 0.0)

        retentate = make_stream(retentate_flows, inlet["T"], inlet["P"])
        permeate = make_stream(permeate_flows, inlet["T"], inlet["P"])

        # Calculate recovery for species with R > 0
        recovery = {}
        for species, R in p.rejection.items():
            if species in inlet_flows:
                recovery[species] = retentate_flows[species] / inlet_flows[species]

        info = {
            "flux": J,
            "concentration_factor": CF,
            "volume_reduction": volume_reduction,
            "recovery": recovery,
            "retentate_volume_fraction": retentate_volume_frac,
        }

        return (retentate, permeate), info


class Diafiltration:
    """Diafiltration for buffer exchange.

    Adds buffer while removing permeate to exchange the buffer
    composition while maintaining constant volume.

    Two modes:
    - Constant volume diafiltration (CVD): Add buffer = Remove permeate
    - Discontinuous diafiltration: Batch dilution then concentration
    """

    def __init__(self, params: DiafiltrationParams):
        """Initialize diafiltration unit.

        Args:
            params: DF parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        buffer: Stream,
        n_diavolumes: float | Array,
        TMP: float | Array = 1.0,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform constant-volume diafiltration.

        Args:
            inlet: Feed stream (retentate side)
            buffer: Buffer stream composition (concentrations)
            n_diavolumes: Number of diavolumes (total buffer volume / initial volume)
            TMP: Transmembrane pressure (bar)

        Returns:
            (retentate, permeate): Tuple of outlet streams
            info: Dictionary with exchange efficiency for each species
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        buffer_flows = get_flows(buffer)

        n_dv = jnp.asarray(n_diavolumes)

        # Total volume (sum of flows as proxy)
        V_initial = sum(inlet_flows.values())

        # Permeate flux
        J = p.Lp * TMP

        # For CVD: C/C_0 = exp(-n_dv * (1-R)) for species being washed out
        # Buffer species: C = C_buffer * (1 - exp(-n_dv * (1-R)))

        retentate_flows = {}
        permeate_flows = {}  # Total permeate over diafiltration

        # Calculate total permeate volume = n_dv * V_initial
        permeate_volume = n_dv * V_initial

        for species, flow in inlet_flows.items():
            R = p.rejection.get(species, 0.0)
            R = jnp.asarray(R)

            # Fraction remaining after diafiltration
            remaining_frac = jnp.exp(-n_dv * (1.0 - R))

            # Initial contribution remaining
            from_initial = flow * remaining_frac

            # Buffer contribution (if species is in buffer)
            buffer_conc = buffer_flows.get(species, jnp.array(0.0)) / sum(buffer_flows.values())
            buffer_added = buffer_conc * n_dv * V_initial
            # Buffer that's retained follows same wash-in kinetics
            from_buffer = buffer_added * (1.0 - remaining_frac) / (1.0 - R + 1e-10)
            from_buffer = jnp.where(R < 0.99, from_buffer, buffer_added)

            retentate_flows[species] = from_initial + from_buffer

            # Permeate is what left
            total_in = flow + buffer_added
            permeate_flows[species] = jnp.maximum(total_in - retentate_flows[species], 0.0)

        # Add any buffer-only species
        for species, buffer_flow in buffer_flows.items():
            if species not in inlet_flows:
                R = p.rejection.get(species, 0.0)
                R = jnp.asarray(R)
                remaining_frac = jnp.exp(-n_dv * (1.0 - R))

                buffer_conc = buffer_flow / sum(buffer_flows.values())
                buffer_added = buffer_conc * n_dv * V_initial
                from_buffer = buffer_added * (1.0 - remaining_frac) / (1.0 - R + 1e-10)
                from_buffer = jnp.where(R < 0.99, from_buffer, buffer_added)

                retentate_flows[species] = from_buffer
                permeate_flows[species] = buffer_added - from_buffer

        retentate = make_stream(retentate_flows, inlet["T"], inlet["P"])
        permeate = make_stream(permeate_flows, inlet["T"], inlet["P"])

        # Exchange efficiency: fraction of original species removed
        exchange_efficiency = {}
        for species in inlet_flows:
            R = p.rejection.get(species, 0.0)
            exchange_efficiency[species] = 1.0 - jnp.exp(-n_dv * (1.0 - R))

        info = {
            "flux": J,
            "n_diavolumes": n_dv,
            "exchange_efficiency": exchange_efficiency,
            "buffer_volume_added": n_dv * V_initial,
        }

        return (retentate, permeate), info


# =============================================================================
# TFF (Tangential Flow Filtration) - Combined UF/DF
# =============================================================================

class TFF:
    """Tangential Flow Filtration system for UF and DF operations.

    Combines ultrafiltration and diafiltration in a single unit,
    as commonly used in bioprocessing for:
    1. Initial concentration
    2. Diafiltration (buffer exchange)
    3. Final concentration
    """

    def __init__(
        self,
        membrane_area: float | Array,
        MWCO: float | Array = 30.0,
        rejection: dict = None,
        Lp: float | Array = 50.0,
    ):
        """Initialize TFF system.

        Args:
            membrane_area: Membrane area (m²)
            MWCO: Molecular weight cutoff (kDa)
            rejection: Dict of species -> rejection coefficient
            Lp: Membrane permeability (L/m²/h/bar)
        """
        rejection = rejection or {}

        self.uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=membrane_area,
            MWCO=MWCO,
            rejection=rejection,
            Lp=Lp,
        ))

        self.df = Diafiltration(DiafiltrationParams(
            membrane_area=membrane_area,
            MWCO=MWCO,
            rejection=rejection,
            Lp=Lp,
        ))

    def concentrate(
        self,
        inlet: Stream,
        concentration_factor: float | Array,
        TMP: float | Array = 1.0,
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Concentrate feed by ultrafiltration.

        Args:
            inlet: Feed stream
            concentration_factor: Target CF
            TMP: Transmembrane pressure (bar)

        Returns:
            (retentate, permeate): Outlet streams
            info: Operation details
        """
        return self.uf(inlet, concentration_factor, TMP)

    def diafilter(
        self,
        inlet: Stream,
        buffer: Stream,
        n_diavolumes: float | Array,
        TMP: float | Array = 1.0,
    ) -> tuple[tuple[Stream, Stream], dict]:
        """Exchange buffer by diafiltration.

        Args:
            inlet: Feed stream
            buffer: Buffer composition
            n_diavolumes: Number of diavolumes
            TMP: Transmembrane pressure (bar)

        Returns:
            (retentate, permeate): Outlet streams
            info: Operation details
        """
        return self.df(inlet, buffer, n_diavolumes, TMP)

    def uf_df_uf(
        self,
        inlet: Stream,
        buffer: Stream,
        CF_initial: float | Array,
        n_diavolumes: float | Array,
        CF_final: float | Array,
        TMP: float | Array = 1.0,
    ) -> tuple[Stream, dict]:
        """Complete UF/DF/UF process.

        Standard process:
        1. Concentrate to CF_initial
        2. Diafilter with n_diavolumes
        3. Concentrate to CF_final

        Args:
            inlet: Feed stream
            buffer: Buffer for diafiltration
            CF_initial: Initial concentration factor
            n_diavolumes: Diavolumes for buffer exchange
            CF_final: Final concentration factor
            TMP: Transmembrane pressure

        Returns:
            final_retentate: Final product stream
            info: Aggregate information from all steps
        """
        # Step 1: Initial concentration
        (ret1, perm1), info1 = self.uf(inlet, CF_initial, TMP)

        # Step 2: Diafiltration
        (ret2, perm2), info2 = self.df(ret1, buffer, n_diavolumes, TMP)

        # Step 3: Final concentration
        (ret3, perm3), info3 = self.uf(ret2, CF_final, TMP)

        info = {
            "step1_uf": info1,
            "step2_df": info2,
            "step3_uf": info3,
            "total_CF": CF_initial * CF_final,
            "n_diavolumes": n_diavolumes,
        }

        return ret3, info


# =============================================================================
# Utility Functions
# =============================================================================

def concentration_polarization(
    C_bulk: Array,
    J: Array,
    k_m: Array,
    R: Array,
) -> Array:
    """Calculate wall concentration with concentration polarization.

    C_wall = C_bulk * exp(J / k_m) / (1 - R + R * exp(J / k_m))

    Args:
        C_bulk: Bulk concentration
        J: Permeate flux (m/s or consistent units with k_m)
        k_m: Mass transfer coefficient (m/s)
        R: Rejection coefficient

    Returns:
        Wall (membrane surface) concentration
    """
    exp_term = jnp.exp(J / k_m)
    return C_bulk * exp_term / (1.0 - R + R * exp_term)


def gel_layer_flux(
    C_bulk: Array,
    C_gel: Array,
    k_m: Array,
) -> Array:
    """Calculate flux limited by gel layer formation.

    J = k_m * ln(C_gel / C_bulk)

    Args:
        C_bulk: Bulk concentration
        C_gel: Gel concentration (limiting)
        k_m: Mass transfer coefficient

    Returns:
        Limiting permeate flux
    """
    return k_m * jnp.log(C_gel / C_bulk)


def diavolumes_required(
    initial_conc: Array,
    target_conc: Array,
    rejection: Array,
) -> Array:
    """Calculate diavolumes needed for target concentration.

    n_dv = -ln(C_target / C_initial) / (1 - R)

    Args:
        initial_conc: Initial concentration of species to remove
        target_conc: Target concentration
        rejection: Rejection coefficient of species

    Returns:
        Required number of diavolumes
    """
    ratio = target_conc / initial_conc
    return -jnp.log(ratio) / (1.0 - rejection + 1e-10)


def rejection_from_mw(
    MW: Array,
    MWCO: Array,
) -> Array:
    """Estimate rejection coefficient from molecular weight.

    Uses sigmoid approximation:
    R = 1 / (1 + exp(-k*(log(MW) - log(MWCO))))

    where k controls steepness (typically ~2-4)

    Args:
        MW: Molecular weight (Da or kDa, consistent with MWCO)
        MWCO: Molecular weight cutoff

    Returns:
        Estimated rejection coefficient (0-1)
    """
    k = 3.0  # Steepness factor
    log_ratio = jnp.log(MW) - jnp.log(MWCO)
    return 1.0 / (1.0 + jnp.exp(-k * log_ratio))
