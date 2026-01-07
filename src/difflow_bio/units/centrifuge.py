"""Centrifuge unit operation for cell separation.

This module provides centrifuge models for separating cells from broth:
- Centrifuge: General centrifuge model using Sigma factor theory

Key equations:
    Sigma factor: Σ = (ω² * V_bowl) / g * geometry_factor
    Throughput: Q = 2 * v_s * Σ
    Stokes velocity: v_s = d² * (ρ_p - ρ_f) * g / (18 * μ)

where:
    Σ = equivalent settling area (m²)
    ω = angular velocity (rad/s)
    v_s = Stokes settling velocity (m/s)
    d = particle diameter (m)
    ρ = density (kg/m³)
    μ = fluid viscosity (Pa·s)
"""

from dataclasses import dataclass, replace
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows


# =============================================================================
# Physical Constants
# =============================================================================

G_ACCEL = 9.81  # m/s²


# =============================================================================
# Centrifuge Parameters
# =============================================================================

@dataclass
class CentrifugeParams:
    """Parameters for centrifuge separation.

    Attributes:
        sigma: Sigma factor - equivalent settling area (m²)
               Can be calculated from geometry or specified directly
        efficiency: Separation efficiency factor (0-1), accounts for
                   non-ideal effects. Default 0.7.
        species_order: List of species names. First species assumed to be cells.
        cell_species: Name of cell species (default "cells")
    """
    sigma: float | Array
    efficiency: float | Array = 0.7
    species_order: list[str] = None
    cell_species: str = "cells"

    def update(self, **kwargs) -> "CentrifugeParams":
        """Return a new CentrifugeParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., sigma=1000.0)

        Returns:
            New CentrifugeParams with updated fields
        """
        return replace(self, **kwargs)


@dataclass
class DiscStackParams:
    """Parameters for disc-stack centrifuge geometry.

    Sigma = (2π * n * ω² * (r_o³ - r_i³)) / (3g)

    Attributes:
        n_discs: Number of discs
        r_outer: Outer radius of disc (m)
        r_inner: Inner radius of disc (m)
        half_angle: Half-angle of disc cone (radians)
        rpm: Rotational speed (rev/min)
        efficiency: Separation efficiency (0-1)
        species_order: List of species names
        cell_species: Name of cell species
    """
    n_discs: int | Array
    r_outer: float | Array
    r_inner: float | Array
    half_angle: float | Array = 0.698  # 40 degrees in radians
    rpm: float | Array = 6000.0
    efficiency: float | Array = 0.7
    species_order: list[str] = None
    cell_species: str = "cells"

    def update(self, **kwargs) -> "DiscStackParams":
        """Return a new DiscStackParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., n_discs=100, rpm=8000.0)

        Returns:
            New DiscStackParams with updated fields
        """
        return replace(self, **kwargs)


# =============================================================================
# Centrifuge Models
# =============================================================================

class Centrifuge:
    """Centrifuge for cell/particle separation.

    Separates particles from liquid based on density difference using
    centrifugal force. Uses Sigma factor theory for scale-up/design.

    Produces two outlet streams:
    - Concentrate (heavy phase): enriched in cells
    - Clarified (light phase): depleted in cells
    """

    def __init__(self, params: CentrifugeParams):
        """Initialize centrifuge.

        Args:
            params: Centrifuge parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        Q: float | Array,
        d_particle: float | Array = 5e-6,
        rho_particle: float | Array = 1050.0,
        rho_fluid: float | Array = 1000.0,
        viscosity: float | Array = 0.001,
        concentrate_fraction: float | Array = 0.1,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform centrifugal separation.

        Args:
            inlet: Feed stream
            Q: Volumetric flow rate (m³/s or L/h, specify units consistently)
            d_particle: Particle diameter (m), default 5 μm
            rho_particle: Particle density (kg/m³), default 1050 (cells)
            rho_fluid: Fluid density (kg/m³), default 1000 (water)
            viscosity: Dynamic viscosity (Pa·s), default 0.001 (water)
            concentrate_fraction: Fraction of flow going to concentrate (0-1)

        Returns:
            (concentrate, clarified): Tuple of outlet streams
            info: Dictionary with:
                - 'separation_efficiency': Actual separation efficiency
                - 'stokes_velocity': Particle settling velocity (m/s)
                - 'critical_diameter': Minimum separable particle size (m)
                - 'cell_recovery': Fraction of cells in concentrate
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        # Stokes settling velocity
        v_s = stokes_velocity(d_particle, rho_particle, rho_fluid, viscosity)

        # Theoretical separation: Q_crit = 2 * v_s * Σ
        # At Q = Q_crit, 50% of particles of size d are captured
        Q_crit = 2 * v_s * p.sigma

        # Separation efficiency based on Q/Q_crit
        # Using cumulative efficiency model
        Q_ratio = Q / Q_crit
        theoretical_sep = jnp.clip(1.0 / Q_ratio, 0.0, 1.0)

        # Apply efficiency factor
        actual_sep = theoretical_sep * p.efficiency

        # Critical particle diameter (smallest captured at 50% efficiency)
        d_crit = critical_particle_diameter(
            Q, p.sigma, rho_particle, rho_fluid, viscosity
        )

        # Split flows
        cell_species = p.cell_species
        cell_flow_in = inlet_flows.get(cell_species, jnp.array(0.0))

        # Cells going to concentrate
        cell_recovery = actual_sep
        cells_to_concentrate = cell_flow_in * cell_recovery
        cells_to_clarified = cell_flow_in * (1 - cell_recovery)

        # Other species split by volume fraction (assumed no separation)
        concentrate_flows = {}
        clarified_flows = {}

        for species, flow in inlet_flows.items():
            if species == cell_species:
                concentrate_flows[species] = cells_to_concentrate
                clarified_flows[species] = cells_to_clarified
            else:
                # Non-cell species split by volumetric ratio
                concentrate_flows[species] = flow * concentrate_fraction
                clarified_flows[species] = flow * (1 - concentrate_fraction)

        concentrate = make_stream(concentrate_flows, inlet["T"], inlet["P"])
        clarified = make_stream(clarified_flows, inlet["T"], inlet["P"])

        info = {
            "separation_efficiency": actual_sep,
            "stokes_velocity": v_s,
            "critical_diameter": d_crit,
            "cell_recovery": cell_recovery,
            "Q_critical": Q_crit,
            "concentration_factor": cell_recovery / concentrate_fraction,
        }

        return (concentrate, clarified), info


class DiscStackCentrifuge:
    """Disc-stack centrifuge with geometry-based Sigma calculation.

    Common in bioprocessing for continuous cell separation.
    """

    def __init__(self, params: DiscStackParams):
        """Initialize disc-stack centrifuge.

        Args:
            params: Disc-stack geometry and operating parameters
        """
        self.params = params

        # Calculate Sigma from geometry
        self.sigma = disc_stack_sigma(
            params.n_discs,
            params.r_outer,
            params.r_inner,
            params.half_angle,
            params.rpm,
        )

        # Create internal Centrifuge with calculated Sigma
        self._centrifuge = Centrifuge(
            CentrifugeParams(
                sigma=self.sigma,
                efficiency=params.efficiency,
                species_order=params.species_order,
                cell_species=params.cell_species,
            )
        )

    def __call__(
        self,
        inlet: Stream,
        Q: float | Array,
        d_particle: float | Array = 5e-6,
        rho_particle: float | Array = 1050.0,
        rho_fluid: float | Array = 1000.0,
        viscosity: float | Array = 0.001,
        concentrate_fraction: float | Array = 0.1,
    ) -> tuple[tuple[Stream, Stream], dict[str, Array]]:
        """Perform centrifugal separation.

        See Centrifuge.__call__ for details.
        """
        return self._centrifuge(
            inlet, Q, d_particle, rho_particle, rho_fluid, viscosity, concentrate_fraction
        )


# =============================================================================
# Sigma Factor Calculations
# =============================================================================

def disc_stack_sigma(
    n_discs: Array | float,
    r_outer: Array | float,
    r_inner: Array | float,
    half_angle: Array | float,
    rpm: Array | float,
) -> Array:
    """Calculate Sigma factor for disc-stack centrifuge.

    Σ = (2π * n * ω² * (r_o³ - r_i³) * cot(θ)) / (3g)

    Args:
        n_discs: Number of discs
        r_outer: Outer radius (m)
        r_inner: Inner radius (m)
        half_angle: Half-angle of cone (radians)
        rpm: Rotational speed (rev/min)

    Returns:
        Sigma factor (m²)
    """
    omega = rpm * 2 * jnp.pi / 60  # rad/s
    cot_theta = 1.0 / jnp.tan(half_angle)

    sigma = (
        2 * jnp.pi * n_discs * omega**2 *
        (r_outer**3 - r_inner**3) * cot_theta
    ) / (3 * G_ACCEL)

    return sigma


def tubular_bowl_sigma(
    r: Array | float,
    L: Array | float,
    rpm: Array | float,
) -> Array:
    """Calculate Sigma factor for tubular bowl centrifuge.

    Σ = (π * ω² * L * r²) / g

    Args:
        r: Bowl radius (m)
        L: Bowl length (m)
        rpm: Rotational speed (rev/min)

    Returns:
        Sigma factor (m²)
    """
    omega = rpm * 2 * jnp.pi / 60
    return jnp.pi * omega**2 * L * r**2 / G_ACCEL


# =============================================================================
# Physical Property Functions
# =============================================================================

def stokes_velocity(
    d: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
) -> Array:
    """Calculate Stokes settling velocity.

    v_s = d² * (ρ_p - ρ_f) * g / (18 * μ)

    Args:
        d: Particle diameter (m)
        rho_p: Particle density (kg/m³)
        rho_f: Fluid density (kg/m³)
        mu: Dynamic viscosity (Pa·s)

    Returns:
        Settling velocity (m/s)
    """
    d = jnp.asarray(d)
    rho_p = jnp.asarray(rho_p)
    rho_f = jnp.asarray(rho_f)
    mu = jnp.asarray(mu)

    return d**2 * (rho_p - rho_f) * G_ACCEL / (18 * mu)


def critical_particle_diameter(
    Q: Array | float,
    sigma: Array | float,
    rho_p: Array | float,
    rho_f: Array | float,
    mu: Array | float,
) -> Array:
    """Calculate critical particle diameter for separation.

    The smallest particle that can be separated at 50% efficiency.

    d_crit = sqrt(9 * μ * Q / (π * Σ * (ρ_p - ρ_f) * g))

    Args:
        Q: Volumetric flow rate (m³/s)
        sigma: Sigma factor (m²)
        rho_p: Particle density (kg/m³)
        rho_f: Fluid density (kg/m³)
        mu: Dynamic viscosity (Pa·s)

    Returns:
        Critical diameter (m)
    """
    Q = jnp.asarray(Q)
    sigma = jnp.asarray(sigma)
    rho_p = jnp.asarray(rho_p)
    rho_f = jnp.asarray(rho_f)
    mu = jnp.asarray(mu)

    d_sq = 9 * mu * Q / (jnp.pi * sigma * (rho_p - rho_f) * G_ACCEL)
    return jnp.sqrt(jnp.maximum(d_sq, 0.0))


def centrifuge_scale_up(
    sigma_1: Array | float,
    Q_1: Array | float,
    sigma_2: Array | float,
) -> Array:
    """Scale up centrifuge throughput using Sigma factor.

    At constant separation efficiency: Q_2/Q_1 = Σ_2/Σ_1

    Args:
        sigma_1: Sigma factor of reference centrifuge (m²)
        Q_1: Throughput of reference centrifuge (m³/s)
        sigma_2: Sigma factor of target centrifuge (m²)

    Returns:
        Throughput of target centrifuge (m³/s)
    """
    return Q_1 * sigma_2 / sigma_1


def g_force(
    r: Array | float,
    rpm: Array | float,
) -> Array:
    """Calculate relative centrifugal force (g-force).

    RCF = ω² * r / g = (rpm * 2π/60)² * r / g

    Args:
        r: Radius (m)
        rpm: Rotational speed (rev/min)

    Returns:
        Relative centrifugal force (dimensionless, multiples of g)
    """
    omega = rpm * 2 * jnp.pi / 60
    return omega**2 * r / G_ACCEL
