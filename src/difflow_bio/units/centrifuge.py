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

from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
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

@dataclass(repr=False)
class CentrifugeParams(ParamsMixin):
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
    # Cell lysis at high g-force (#103). Phenomenological, fully
    # user-parameterized model (no built-in constants): above
    # lysis_threshold_g (RCF), a fraction lysis_coefficient*(RCF - threshold)
    # of cells is disrupted and released as `lysate_species` into the
    # clarified (supernatant) stream, degrading its purity. Defaults leave
    # lysis disabled (backward compatible). See Doran, Bioprocess Engineering
    # Principles, 2e, 2013, Ch. 10 for the shear-damage phenomenon.
    lysis_threshold_g: float | Array | None = None  # RCF onset (x g)
    lysis_coefficient: float | Array = 0.0  # lysed fraction per unit RCF above threshold
    lysate_species: str = "lysate"


@dataclass(repr=False)
class DiscStackParams(ParamsMixin):
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
    # Cell lysis at high g-force (#103); forwarded to the internal Centrifuge.
    lysis_threshold_g: float | Array | None = None
    lysis_coefficient: float | Array = 0.0
    lysate_species: str = "lysate"


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

    symbol = "Centrifuge"
    equations = [
        r"v_s = \frac{(\rho_p - \rho_f)\,g\,d_p^2}{18\,\mu}\qquad \text{(Stokes settling)}",
        r"Q/\Sigma = 2\,v_s\qquad \text{(Ambler's sigma scale-up)}",
        r"\eta = \tanh\!\left(\frac{v_s\,\Sigma}{Q}\right)\qquad \text{(smooth separation efficiency)}",
    ]
    assumptions = [
        "Stokes flow regime (dilute suspension, low Re).",
        "Uniform particle size and density.",
        "Negligible particle-particle interactions.",
    ]
    references = [
        "Ambler, C.M. J. Biochem. Microbiol. Tech. Eng., 1, 185 (1959).",
        "Doran, P.M. Bioprocess Engineering Principles, 2e, Academic Press, 2013.",
    ]
    parameter_symbols = {}
    parameter_units = {}
    numerical_method = "Closed-form Stokes / sigma-factor relations with smooth efficiency."

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
        rcf: float | Array = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform centrifugal separation.

        Args:
            inlet: Feed stream
            Q: Volumetric flow rate (m³/s or L/h, specify units consistently)
            d_particle: Particle diameter (m), default 5 μm
            rho_particle: Particle density (kg/m³), default 1050 (cells)
            rho_fluid: Fluid density (kg/m³), default 1000 (water)
            viscosity: Dynamic viscosity (Pa·s), default 0.001 (water)
            concentrate_fraction: Fraction of flow going to concentrate (0-1)
            rcf: Relative centrifugal force (x g) at the separation zone. Used
                only when ``params.lysis_threshold_g`` is set, to compute the
                g-force-dependent cell lysis fraction (#103).

        Returns:
            concentrate: Concentrate stream (enriched in cells)
            clarified: Clarified stream (depleted in cells)
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

        # Apply efficiency factor (clamp to prevent mass creation)
        actual_sep = jnp.clip(theoretical_sep * p.efficiency, 0.0, 1.0)

        # Critical particle diameter (smallest captured at 50% efficiency)
        d_crit = critical_particle_diameter(
            Q, p.sigma, rho_particle, rho_fluid, viscosity
        )

        # Split flows
        cell_species = p.cell_species
        cell_flow_in = inlet_flows.get(cell_species, jnp.array(0.0))

        # Cell lysis at high g-force (#103): a fraction of cells is disrupted
        # and released as lysate into the supernatant. Intact cells then
        # separate as usual; lysate reports to the clarified stream.
        lysis_active = p.lysis_threshold_g is not None and rcf is not None
        if lysis_active:
            excess_g = jnp.maximum(jnp.asarray(rcf) - p.lysis_threshold_g, 0.0)
            lysis_fraction = jnp.clip(p.lysis_coefficient * excess_g, 0.0, 1.0)
        else:
            lysis_fraction = jnp.asarray(0.0)

        intact_cells_in = cell_flow_in * (1.0 - lysis_fraction)
        lysate_mass = cell_flow_in * lysis_fraction

        # Intact cells going to concentrate
        cell_recovery = actual_sep
        cells_to_concentrate = intact_cells_in * cell_recovery

        # Other species split by volume fraction (assumed no separation)
        concentrate_flows = {}
        clarified_flows = {}

        for species, flow in inlet_flows.items():
            if species == cell_species:
                concentrate_flows[species] = cells_to_concentrate
                # Only intact cells remain; lysed mass leaves as lysate below.
                clarified_flows[species] = intact_cells_in - cells_to_concentrate
            else:
                # Non-cell species split by volumetric ratio
                concentrate_flows[species] = flow * concentrate_fraction
                # Derive clarified from feed - concentrate to guarantee mass balance
                clarified_flows[species] = flow - flow * concentrate_fraction

        # Released intracellular content reports to the clarified (supernatant).
        # Only injected when lysis is active, so default outputs are unchanged.
        if lysis_active:
            if p.lysate_species in clarified_flows:
                clarified_flows[p.lysate_species] = clarified_flows[p.lysate_species] + lysate_mass
            else:
                clarified_flows[p.lysate_species] = lysate_mass
                concentrate_flows.setdefault(p.lysate_species, jnp.asarray(0.0))

        concentrate = make_stream(concentrate_flows, inlet["T"], inlet["P"])
        clarified = make_stream(clarified_flows, inlet["T"], inlet["P"])

        info = {
            "separation_efficiency": actual_sep,
            "stokes_velocity": v_s,
            "critical_diameter": d_crit,
            "cell_recovery": cell_recovery,
            "Q_critical": Q_crit,
            "concentration_factor": cell_recovery / concentrate_fraction,
            "lysis_fraction": lysis_fraction,
            "lysate_released": lysate_mass,
        }

        return concentrate, clarified, info


class DiscStackCentrifuge:
    """Disc-stack centrifuge with geometry-based Sigma calculation.

    Common in bioprocessing for continuous cell separation.
    """

    symbol = "Disc-Stack Centrifuge"
    equations = [
        r"\Sigma = \frac{2\pi\,n\,\omega^2}{3\,g\tan\theta}\,(r_o^3 - r_i^3)",
        r"Q/\Sigma = 2\,v_s",
    ]
    assumptions = [
        "Ideal disc-stack geometry; no short-circuiting.",
        "Stokes settling between discs.",
    ]
    references = [
        "Ambler, C.M. J. Biochem. Microbiol. Tech. Eng., 1, 185 (1959).",
        "Perry's Chemical Engineers' Handbook, 9e, Sec. 18.",
    ]
    parameter_symbols = {"n_discs": "n", "r_outer": "r_o", "r_inner": "r_i", "rpm": "RPM"}
    parameter_units = {"r_outer": "m", "r_inner": "m", "half_angle": "rad", "rpm": "rev/min"}
    numerical_method = "Sigma-factor from disc-stack geometry fed to Stokes settling model."

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

        # Relative centrifugal force at the outer radius (for lysis, #103)
        self.rcf = g_force(params.r_outer, params.rpm)

        # Create internal Centrifuge with calculated Sigma
        self._centrifuge = Centrifuge(
            CentrifugeParams(
                sigma=self.sigma,
                efficiency=params.efficiency,
                species_order=params.species_order,
                cell_species=params.cell_species,
                lysis_threshold_g=params.lysis_threshold_g,
                lysis_coefficient=params.lysis_coefficient,
                lysate_species=params.lysate_species,
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
        rcf: float | Array = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform centrifugal separation.

        See Centrifuge.__call__ for details. The relative centrifugal force
        defaults to the value computed from the disc-stack geometry
        (``g_force(r_outer, rpm)``) so the g-force-dependent lysis model (#103)
        works without a separately supplied ``rcf``.
        """
        return self._centrifuge(
            inlet, Q, d_particle, rho_particle, rho_fluid, viscosity,
            concentrate_fraction, rcf=self.rcf if rcf is None else rcf,
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
