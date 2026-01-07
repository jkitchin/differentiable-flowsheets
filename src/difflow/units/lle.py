"""Liquid-Liquid Extraction unit operations.

This module provides differentiable LLE operations:
- MultistageCascade: Counter-current or co-current extraction cascade
- DifferentialContactor: Packed column or similar continuous contactor

Supports both distribution coefficient (K-value) models and activity
coefficient models (NRTL, UNIQUAC) for computing equilibrium.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass, field, replace
from typing import Callable, NamedTuple, Literal
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream


# =============================================================================
# Utility Functions
# =============================================================================

def soft_clip_positive(x: Array, sharpness: float = 10.0) -> Array:
    """Soft clipping to ensure non-negative values with smooth gradients.

    Uses softplus function shifted to pass through origin:
        soft_clip(x) = softplus(x * sharpness) / sharpness

    This is approximately:
        - x when x >> 0
        - 0 when x << 0
        - smooth transition near x = 0

    Args:
        x: Input array
        sharpness: Controls transition sharpness (higher = closer to hard clip)

    Returns:
        Soft-clipped array with values >= 0
    """
    return jax.nn.softplus(x * sharpness) / sharpness


# =============================================================================
# Distribution Coefficient Models
# =============================================================================

class DistributionCoeffs(NamedTuple):
    """Simple distribution coefficient model.

    K = concentration in extract / concentration in raffinate

    For temperature-dependent K:
        K(T) = K0 * exp(-dH / R * (1/T - 1/Tref))

    Attributes:
        species: List of species that transfer between phases
        K0: Distribution coefficients at reference temperature
        dH: Heat of extraction (J/mol), for temperature dependence
        Tref: Reference temperature (K)
    """
    species: tuple[str, ...]
    K0: tuple[float, ...]  # Distribution coefficients at Tref
    dH: tuple[float, ...] | None = None  # Heat of extraction for T-dependence
    Tref: float = 298.15


def get_K_values(
    coeffs: DistributionCoeffs,
    T: Array,
) -> dict[str, Array]:
    """Calculate distribution coefficients at given temperature.

    Args:
        coeffs: Distribution coefficient parameters
        T: Temperature (K)

    Returns:
        Dictionary of K-values for each transferring species
    """
    R = 8.314  # J/(mol·K)
    K_dict = {}

    for i, species in enumerate(coeffs.species):
        K0 = coeffs.K0[i]
        if coeffs.dH is not None:
            dH = coeffs.dH[i]
            K = K0 * jnp.exp(-dH / R * (1/T - 1/coeffs.Tref))
        else:
            K = jnp.asarray(K0)
        K_dict[species] = K

    return K_dict


# =============================================================================
# Activity Coefficient Models
# =============================================================================

class NRTLParams(NamedTuple):
    """NRTL activity coefficient model parameters.

    For temperature-dependent parameters:
        tau_ij = a_ij + b_ij/T

    G_ij = exp(-alpha_ij * tau_ij)

    Attributes:
        species: List of all species (in order)
        a: Constant part of tau parameter matrix (n x n)
        b: Temperature-dependent part of tau (n x n), in K
        alpha: Non-randomness parameter matrix (n x n)
    """
    species: tuple[str, ...]
    a: Array  # (n, n) matrix of constant tau coefficients
    b: Array  # (n, n) matrix of temperature coefficients
    alpha: Array  # (n, n) non-randomness parameters


def nrtl_activity_coefficients(
    x: Array,
    T: Array,
    params: NRTLParams,
) -> Array:
    """Calculate activity coefficients using NRTL model.

    Fully vectorized implementation for JAX compatibility.

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: NRTL parameters

    Returns:
        Activity coefficients for each species
    """
    # Temperature-dependent tau
    tau = params.a + params.b / T

    # G matrix
    G = jnp.exp(-params.alpha * tau)

    # Compute activity coefficients (vectorized)
    # ln(gamma_i) = (sum_j tau_ji * G_ji * x_j) / (sum_k G_ki * x_k)
    #             + sum_j (x_j * G_ij / sum_k G_kj * x_k) * (tau_ij - sum_m x_m * tau_mj * G_mj / sum_k G_kj * x_k)

    # Term 1: (sum_j tau_ji * G_ji * x_j) / (sum_k G_ki * x_k)
    # Denominator for all i: G.T @ x, shape (n,)
    # Numerator for all i: (tau * G).T @ x, shape (n,)
    denom = G.T @ x  # shape (n,)
    numer_tau_G = (tau * G).T @ x  # shape (n,)
    term1 = numer_tau_G / denom

    # Term 2: sum_j (x_j * G_ij / sum_k G_kj * x_k) * (tau_ij - sum_m x_m * tau_mj * G_mj / sum_k G_kj * x_k)
    # denom[j] = sum_k G_kj * x_k (already computed above)
    # numer_tau_G[j] = sum_m x_m * tau_mj * G_mj (already computed above)
    # weight[i,j] = x_j * G_ij / denom[j]
    # inner[i,j] = tau_ij - numer_tau_G[j] / denom[j]
    weight = G * (x / denom)  # broadcasts (n,) over columns of (n,n)
    inner = tau - (numer_tau_G / denom)  # broadcasts (n,) over columns of (n,n)
    term2 = jnp.sum(weight * inner, axis=1)  # sum over j

    ln_gamma = term1 + term2
    return jnp.exp(ln_gamma)


class UNIQUACParams(NamedTuple):
    """UNIQUAC activity coefficient model parameters.

    For temperature-dependent parameters:
        tau_ij = exp(-(a_ij + b_ij/T))

    Attributes:
        species: List of all species (in order)
        r: Volume parameters for each species
        q: Surface area parameters for each species
        a: Constant part of interaction parameters (n x n)
        b: Temperature-dependent part (n x n), in K
        z: Coordination number (default 10, typical range 6-12)
    """
    species: tuple[str, ...]
    r: Array  # Volume parameters
    q: Array  # Surface parameters
    a: Array  # (n, n) interaction parameter matrix
    b: Array  # (n, n) temperature coefficients
    z: float = 10.0  # Coordination number


def uniquac_activity_coefficients(
    x: Array,
    T: Array,
    params: UNIQUACParams,
) -> Array:
    """Calculate activity coefficients using UNIQUAC model.

    Fully vectorized implementation for JAX compatibility.

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: UNIQUAC parameters

    Returns:
        Activity coefficients for each species
    """
    r = params.r
    q = params.q
    z = params.z

    # Segment and area fractions
    phi = x * r / jnp.sum(x * r)  # Segment fraction
    theta = x * q / jnp.sum(x * q)  # Area fraction

    # l parameter
    l = (z / 2) * (r - q) - (r - 1)

    # Temperature-dependent tau
    tau = jnp.exp(-(params.a + params.b / T))

    # Combinatorial contribution (already vectorized)
    ln_gamma_C = jnp.log(phi / x) + (z / 2) * q * jnp.log(theta / phi) + l - (phi / x) * jnp.sum(x * l)

    # Residual contribution (vectorized)
    # sum_theta_tau[i] = sum_j theta[j] * tau[j,i] = (tau.T @ theta)[i]
    sum_theta_tau = tau.T @ theta  # shape (n,)

    # Term 1: -q * log(sum_theta_tau)
    term1 = -q * jnp.log(sum_theta_tau)

    # Term 2: q * (1 - sum_j (theta[j] * tau[i,j] / sum_k theta[k] * tau[k,j]))
    # sum_theta_tau_j[j] = sum_k theta[k] * tau[k,j] = (tau.T @ theta)[j] (same as above)
    # inner[i,j] = theta[j] * tau[i,j] / sum_theta_tau[j]
    inner = tau * (theta / sum_theta_tau)  # broadcasts (n,) over columns of (n,n)
    term2 = q * (1 - jnp.sum(inner, axis=1))  # sum over j

    ln_gamma_R = term1 + term2

    return jnp.exp(ln_gamma_C + ln_gamma_R)


# =============================================================================
# LLE Equilibrium Calculation
# =============================================================================

@dataclass
class LLEEquilibrium:
    """Liquid-liquid equilibrium calculator.

    Computes how solutes distribute between aqueous and organic phases
    based on either distribution coefficients or activity coefficient models.

    Attributes:
        solutes: Species that transfer between phases
        aqueous_carrier: Species that stays in aqueous phase
        organic_carrier: Species that stays in organic phase
        K_coeffs: Distribution coefficient model (optional)
        nrtl_params: NRTL parameters (optional)
        uniquac_params: UNIQUAC parameters (optional)
    """
    solutes: list[str]
    aqueous_carrier: str
    organic_carrier: str
    K_coeffs: DistributionCoeffs | None = None
    nrtl_params: NRTLParams | None = None
    uniquac_params: UNIQUACParams | None = None
    activity_model: Literal["K", "NRTL", "UNIQUAC"] = "K"

    def get_distribution_coefficients(
        self,
        x_aq: dict[str, Array],
        x_org: dict[str, Array],
        T: Array,
    ) -> dict[str, Array]:
        """Calculate distribution coefficients at equilibrium.

        For K-value model: Returns constant (or T-dependent) K values
        For activity models: K = gamma_aq / gamma_org

        Args:
            x_aq: Mole fractions in aqueous phase
            x_org: Mole fractions in organic phase
            T: Temperature (K)

        Returns:
            Distribution coefficients for each solute
        """
        if self.activity_model == "K":
            if self.K_coeffs is None:
                raise ValueError("K_coeffs required for K model")
            return get_K_values(self.K_coeffs, T)

        elif self.activity_model == "NRTL":
            if self.nrtl_params is None:
                raise ValueError("nrtl_params required for NRTL model")

            # Convert dicts to arrays
            species = self.nrtl_params.species
            x_aq_arr = jnp.array([x_aq.get(s, 0.0) for s in species])
            x_org_arr = jnp.array([x_org.get(s, 0.0) for s in species])

            gamma_aq = nrtl_activity_coefficients(x_aq_arr, T, self.nrtl_params)
            gamma_org = nrtl_activity_coefficients(x_org_arr, T, self.nrtl_params)

            K_dict = {}
            for i, s in enumerate(species):
                if s in self.solutes:
                    # K = gamma_aq / gamma_org (equilibrium condition)
                    K_dict[s] = gamma_aq[i] / gamma_org[i]
            return K_dict

        elif self.activity_model == "UNIQUAC":
            if self.uniquac_params is None:
                raise ValueError("uniquac_params required for UNIQUAC model")

            species = self.uniquac_params.species
            x_aq_arr = jnp.array([x_aq.get(s, 0.0) for s in species])
            x_org_arr = jnp.array([x_org.get(s, 0.0) for s in species])

            gamma_aq = uniquac_activity_coefficients(x_aq_arr, T, self.uniquac_params)
            gamma_org = uniquac_activity_coefficients(x_org_arr, T, self.uniquac_params)

            K_dict = {}
            for i, s in enumerate(species):
                if s in self.solutes:
                    K_dict[s] = gamma_aq[i] / gamma_org[i]
            return K_dict

        else:
            raise ValueError(f"Unknown activity model: {self.activity_model}")


# =============================================================================
# Multi-Stage Cascade Extractor
# =============================================================================

@dataclass
class CascadeParams:
    """Parameters for multi-stage cascade extractor.

    Attributes:
        n_stages: Number of equilibrium stages (can be continuous for optimization)
        equilibrium: LLE equilibrium calculator
        flow_config: 'counter_current' or 'co_current'
        stage_efficiency: Murphree stage efficiency for co-current model (0-1).
                         Fraction of equilibrium achieved per stage. Default 0.8.
    """
    n_stages: int | float | Array
    equilibrium: LLEEquilibrium
    flow_config: Literal["counter_current", "co_current"] = "counter_current"
    stage_efficiency: float = 0.8

    def update(self, **kwargs) -> "CascadeParams":
        """Return a new CascadeParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., n_stages=10)

        Returns:
            New CascadeParams with updated fields
        """
        return replace(self, **kwargs)


class MultistageCascade:
    """Multi-stage liquid-liquid extraction cascade.

    Models a series of mixer-settler stages or a tray column.
    Supports both counter-current and co-current flow configurations.

    For optimization, n_stages can be treated as a continuous variable
    using stage efficiency interpolation.
    """

    def __init__(self, params: CascadeParams):
        """Initialize cascade.

        Args:
            params: Cascade parameters
        """
        self.params = params

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform multi-stage extraction.

        Args:
            feed: Aqueous feed stream (contains solutes + aqueous carrier)
            solvent: Organic solvent stream (fresh solvent)
            T: Operating temperature (K). If None, uses feed temperature.

        Returns:
            raffinate: Aqueous outlet (depleted in solutes)
            extract: Organic outlet (enriched in solutes)
            info: Stage-by-stage profiles and other information
        """
        p = self.params
        eq = p.equilibrium

        T = jnp.asarray(T) if T is not None else feed["T"]

        # Get flows
        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        # Total aqueous and organic flows (carriers don't transfer)
        F_aq = feed_flows[eq.aqueous_carrier]
        F_org = solvent_flows[eq.organic_carrier]

        # Convert n_stages to JAX array for differentiability
        n_stages = jnp.asarray(p.n_stages, dtype=jnp.float64)

        # Solve stage-by-stage using Kremser equation (continuous n_stages)
        if p.flow_config == "counter_current":
            raffinate, extract, profiles = self._solve_counter_current(
                feed_flows, solvent_flows, F_aq, F_org, T, n_stages
            )
        else:
            raffinate, extract, profiles = self._solve_co_current(
                feed_flows, solvent_flows, F_aq, F_org, T, n_stages
            )

        # Create output streams
        P = feed["P"]
        raffinate_stream = make_stream(raffinate, T, P)
        extract_stream = make_stream(extract, T, P)

        info = {
            "n_stages": n_stages,
            "profiles": profiles,
            "T": T,
        }

        return raffinate_stream, extract_stream, info

    def _solve_counter_current(
        self,
        feed_flows: dict,
        solvent_flows: dict,
        F_aq: Array,
        F_org: Array,
        T: Array,
        n_stages: Array,
    ) -> tuple[dict, dict, dict]:
        """Solve counter-current cascade using Kremser equation.

        Stage numbering: 1 = feed end, N = solvent end
        Aqueous flows: 1 -> 2 -> ... -> N
        Organic flows: N -> N-1 -> ... -> 1

        Uses the Kremser equation which is fully differentiable
        with respect to n_stages (continuous relaxation).

        Fully vectorized over solutes for JAX compatibility.
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        # Estimate phase compositions for activity coefficient models
        # Use feed composition for aqueous, solvent composition for organic
        F_in_arr = jnp.array([feed_flows.get(s, 0.0) for s in solutes])
        F_solvent_arr = jnp.array([solvent_flows.get(s, 0.0) for s in solutes])

        # Estimate mole fractions (including carriers)
        total_aq = F_aq + jnp.sum(F_in_arr)
        total_org = F_org + jnp.sum(F_solvent_arr)

        x_aq_est = {eq.aqueous_carrier: F_aq / total_aq}
        x_org_est = {eq.organic_carrier: F_org / total_org}
        for i, s in enumerate(solutes):
            x_aq_est[s] = F_in_arr[i] / total_aq
            x_org_est[s] = (F_solvent_arr[i] + 1e-10) / total_org  # Avoid zero

        # Get distribution coefficients with estimated compositions
        K_dict = eq.get_distribution_coefficients(x_aq_est, x_org_est, T)

        # Convert to arrays for vectorized computation
        K_arr = jnp.array([K_dict[s] for s in solutes])

        # Extraction factors for all solutes
        E_arr = K_arr * F_org / F_aq

        # Kremser equation with smooth handling of E ≈ 1 singularity
        # For E != 1: fraction_remaining = (E - 1) / (E^(N+1) - 1)
        # For E = 1: fraction_remaining = 1 / (N + 1)
        #
        # Use a fixed small offset to avoid the exact singularity.
        # This introduces a tiny error (~1e-6) but gives smooth gradients everywhere.

        # Add small fixed offset to ensure we're never exactly at E=1
        E_safe = E_arr + 1e-7

        E_Np1 = E_safe ** (n_stages + 1)
        delta_E = E_safe - 1.0

        # Now we can safely compute Kremser (no singularity)
        frac_remaining = delta_E / (E_Np1 - 1.0)

        # Clamp to physical bounds [0, 1]
        frac_remaining = jnp.clip(frac_remaining, 0.0, 1.0)

        frac_extracted = 1.0 - frac_remaining

        F_raffinate_arr = F_in_arr * frac_remaining
        F_extracted_arr = F_in_arr * frac_extracted

        # Convert back to dicts
        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}
        for i, s in enumerate(solutes):
            raffinate_flows[s] = F_raffinate_arr[i]
            extract_flows[s] = F_extracted_arr[i]

        # Stage profiles are approximate for continuous n_stages
        profiles = {"x": {s: [] for s in solutes}, "y": {s: [] for s in solutes}}

        return raffinate_flows, extract_flows, profiles

    def _solve_co_current(
        self,
        feed_flows: dict,
        solvent_flows: dict,
        F_aq: Array,
        F_org: Array,
        T: Array,
        n_stages: Array,
    ) -> tuple[dict, dict, dict]:
        """Solve co-current cascade.

        Both phases flow in same direction: stage 1 -> 2 -> ... -> N
        Less efficient than counter-current but simpler.

        For co-current flow, equilibrium is approached asymptotically.
        The extraction efficiency increases with number of stages but
        is limited by single-stage equilibrium.

        Fully vectorized over solutes for JAX compatibility.
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        # Convert to arrays for vectorized computation
        F_in_arr = jnp.array([feed_flows.get(s, 0.0) for s in solutes])
        F_solvent_arr = jnp.array([solvent_flows.get(s, 0.0) for s in solutes])

        # Estimate phase compositions for activity coefficient models
        total_aq = F_aq + jnp.sum(F_in_arr)
        total_org = F_org + jnp.sum(F_solvent_arr)

        x_aq_est = {eq.aqueous_carrier: F_aq / total_aq}
        x_org_est = {eq.organic_carrier: F_org / total_org}
        for i, s in enumerate(solutes):
            x_aq_est[s] = F_in_arr[i] / total_aq
            x_org_est[s] = (F_solvent_arr[i] + 1e-10) / total_org

        # Get distribution coefficients with estimated compositions
        K_dict = eq.get_distribution_coefficients(x_aq_est, x_org_est, T)
        K_arr = jnp.array([K_dict[s] for s in solutes])

        # Total solute amounts
        F_total_arr = F_in_arr + F_solvent_arr

        # Equilibrium concentrations
        # At equilibrium: y = K * x and mass balance
        # F_aq * x + F_org * y = F_total
        # x = F_total / (F_aq + K * F_org)
        x_eq_arr = F_total_arr / (F_aq + K_arr * F_org)

        # Approach to equilibrium with multiple stages
        # Each stage achieves stage_efficiency fraction of remaining driving force
        # Total efficiency = 1 - (1 - eff)^N (differentiable in N)
        eff_per_stage = self.params.stage_efficiency
        total_eff = 1.0 - (1.0 - eff_per_stage) ** n_stages

        x_feed_arr = F_in_arr / (F_aq + 1e-10)
        x_final_arr = x_feed_arr + total_eff * (x_eq_arr - x_feed_arr)

        F_raffinate_arr = x_final_arr * F_aq
        F_extracted_arr = F_total_arr - F_raffinate_arr

        # Convert back to dicts
        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}
        for i, s in enumerate(solutes):
            raffinate_flows[s] = F_raffinate_arr[i]
            extract_flows[s] = F_extracted_arr[i]

        profiles = {"x": {s: [] for s in solutes}, "y": {s: [] for s in solutes}}

        return raffinate_flows, extract_flows, profiles


# =============================================================================
# Differential Contactor
# =============================================================================

@dataclass
class ContactorParams:
    """Parameters for differential contactor.

    Attributes:
        length: Contactor length (m)
        area: Cross-sectional area (m^2)
        equilibrium: LLE equilibrium calculator
        n_segments: Number of discretization segments
        flow_config: 'counter_current' or 'co_current'
        mass_transfer_model: 'equilibrium' or 'rate_based'
        Kla: Overall mass transfer coefficient * interfacial area (1/s)
             Only used for rate_based model
        HETP: Height equivalent to theoretical plate (m)
              Only used for equilibrium model
    """
    length: float | Array
    area: float | Array
    equilibrium: LLEEquilibrium
    n_segments: int = 50
    flow_config: Literal["counter_current", "co_current"] = "counter_current"
    mass_transfer_model: Literal["equilibrium", "rate_based"] = "equilibrium"
    Kla: float | Array | dict[str, float] = 0.01  # 1/s, per solute or global
    HETP: float | Array = 0.5  # m

    def update(self, **kwargs) -> "ContactorParams":
        """Return a new ContactorParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., length=5.0, HETP=0.3)

        Returns:
            New ContactorParams with updated fields
        """
        return replace(self, **kwargs)


class DifferentialContactor:
    """Differential (packed column) liquid-liquid extractor.

    Models a continuous contacting device like a packed column.
    Supports both equilibrium stage equivalent (HETP) and rate-based
    mass transfer models.
    """

    def __init__(self, params: ContactorParams):
        """Initialize contactor.

        Args:
            params: Contactor parameters
        """
        self.params = params

    def __call__(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform extraction in differential contactor.

        Args:
            feed: Aqueous feed stream
            solvent: Organic solvent stream
            T: Operating temperature (K)

        Returns:
            raffinate: Aqueous outlet
            extract: Organic outlet
            info: Axial profiles and other information
        """
        p = self.params
        eq = p.equilibrium

        T = jnp.asarray(T) if T is not None else feed["T"]

        if p.mass_transfer_model == "equilibrium":
            return self._solve_equilibrium_model(feed, solvent, T)
        else:
            return self._solve_rate_based_model(feed, solvent, T)

    def _solve_equilibrium_model(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array,
    ) -> tuple[Stream, Stream, dict]:
        """Solve using equilibrium stage equivalent (HETP) model.

        Number of theoretical stages = L / HETP
        Then use cascade equations.
        """
        p = self.params
        eq = p.equilibrium

        # Equivalent number of stages
        n_stages = p.length / p.HETP

        # Use cascade solver with continuous stages
        cascade_params = CascadeParams(
            n_stages=n_stages,
            equilibrium=eq,
            flow_config=p.flow_config,
        )
        cascade = MultistageCascade(cascade_params)

        raffinate, extract, cascade_info = cascade(feed, solvent, T)

        # Add height profile info
        info = {
            "n_stages_equivalent": n_stages,
            "HETP": p.HETP,
            "length": p.length,
            **cascade_info,
        }

        return raffinate, extract, info

    def _solve_rate_based_model(
        self,
        feed: Stream,
        solvent: Stream,
        T: Array,
    ) -> tuple[Stream, Stream, dict]:
        """Solve using rate-based mass transfer model.

        Uses discretized ODEs with mass transfer rate:
        dF_i/dz = Kla * (c_i - c_i_eq) * A
        """
        p = self.params
        eq = p.equilibrium

        feed_flows = get_flows(feed)
        solvent_flows = get_flows(solvent)

        F_aq = feed_flows[eq.aqueous_carrier]
        F_org = solvent_flows[eq.organic_carrier]

        # Get K-values
        K_dict = eq.get_distribution_coefficients({}, {}, T)

        solutes = eq.solutes
        n_solutes = len(solutes)
        n_seg = p.n_segments
        dz = p.length / n_seg

        # Initialize concentrations
        # c_aq[z, s] = concentration in aqueous at position z
        c_aq_init = {s: feed_flows.get(s, 0.0) / F_aq for s in solutes}
        c_org_init = {s: solvent_flows.get(s, 0.0) / F_org for s in solutes}

        if p.flow_config == "counter_current":
            raffinate_flows, extract_flows, profiles = self._integrate_counter_current(
                c_aq_init, c_org_init, F_aq, F_org, K_dict, T, dz, n_seg
            )
        else:
            raffinate_flows, extract_flows, profiles = self._integrate_co_current(
                c_aq_init, c_org_init, F_aq, F_org, K_dict, T, dz, n_seg
            )

        # Add carriers to output
        raffinate_flows[eq.aqueous_carrier] = F_aq
        extract_flows[eq.organic_carrier] = F_org

        P = feed["P"]
        raffinate = make_stream(raffinate_flows, T, P)
        extract = make_stream(extract_flows, T, P)

        info = {
            "length": p.length,
            "n_segments": n_seg,
            "profiles": profiles,
            "T": T,
        }

        return raffinate, extract, info

    def _integrate_counter_current(
        self,
        c_aq_init: dict,
        c_org_init: dict,
        F_aq: Array,
        F_org: Array,
        K_dict: dict,
        T: Array,
        dz: float,
        n_seg: int,
    ) -> tuple[dict, dict, dict]:
        """Integrate rate equations for counter-current flow.

        Aqueous enters at z=0, organic enters at z=L.

        For constant K (linear system), solves the two-point BVP exactly
        using matrix exponential. Each solute is independent.

        System for each solute:
            dc_aq/dz = -Kla * (K*c_aq - c_org) * area / F_aq
            dc_org/dz = +Kla * (K*c_aq - c_org) * area / F_org

        Boundary conditions:
            c_aq(0) = c_aq_init (known)
            c_org(L) = c_org_init (known)

        Fully vectorized over solutes for JAX compatibility.
        """
        from jax.scipy.linalg import expm

        p = self.params
        eq = p.equilibrium
        solutes = eq.solutes
        n_solutes = len(solutes)
        L = p.length

        # Convert to arrays
        K_arr = jnp.array([K_dict[s] for s in solutes])
        if isinstance(p.Kla, dict):
            Kla_arr = jnp.array([p.Kla[s] for s in solutes])
        else:
            Kla_arr = jnp.full(n_solutes, p.Kla)

        c_aq_0 = jnp.array([c_aq_init[s] for s in solutes])
        c_org_L = jnp.array([c_org_init[s] for s in solutes])

        area = p.area

        # Build system matrices for each solute
        # For counter-current flow, organic flows in -z direction, so:
        #   dc_aq/dz = -Kla*(K*c_aq - c_org)*area/F_aq  (aqueous loses to organic)
        #   dc_org/dz = -Kla*(K*c_aq - c_org)*area/F_org  (organic gains, but flows backward)
        # Note: both have NEGATIVE sign because dc_org/dz = -rate/F_org for backward flow
        #
        # A = [[-a,  b],    where a = Kla*K*area/F_aq, b = Kla*area/F_aq
        #      [-c,  d]]          c = Kla*K*area/F_org, d = Kla*area/F_org
        alpha = Kla_arr * area
        a = alpha * K_arr / F_aq  # shape (n_solutes,)
        b = alpha / F_aq
        c = alpha * K_arr / F_org
        d = alpha / F_org

        # Stack into matrices: shape (n_solutes, 2, 2)
        A_matrices = jnp.stack([
            jnp.stack([-a, b], axis=-1),
            jnp.stack([-c, d], axis=-1)  # Note: [-c, d] for counter-current
        ], axis=-2).transpose(2, 0, 1)  # (n_solutes, 2, 2)

        # Compute matrix exponential at z=L for each solute
        # M = expm(A * L), shape (n_solutes, 2, 2)
        def compute_expm(A):
            return expm(A * L)
        M = jax.vmap(compute_expm)(A_matrices)

        # Extract matrix elements
        M00 = M[:, 0, 0]
        M01 = M[:, 0, 1]
        M10 = M[:, 1, 0]
        M11 = M[:, 1, 1]

        # Solve for c_org(0) using boundary conditions
        # c_org(L) = M10 * c_aq(0) + M11 * c_org(0)
        # c_org(0) = (c_org(L) - M10 * c_aq(0)) / M11
        c_org_0 = (c_org_L - M10 * c_aq_0) / (M11 + 1e-10)

        # Compute c_aq(L)
        c_aq_L = M00 * c_aq_0 + M01 * c_org_0

        # Compute profiles at each z position using lax.scan
        z_positions = jnp.linspace(0, L, n_seg + 1)

        def compute_profile_at_z(z):
            """Compute concentrations at position z for all solutes."""
            def expm_at_z(A):
                return expm(A * z)
            M_z = jax.vmap(expm_at_z)(A_matrices)
            c_aq_z = M_z[:, 0, 0] * c_aq_0 + M_z[:, 0, 1] * c_org_0
            c_org_z = M_z[:, 1, 0] * c_aq_0 + M_z[:, 1, 1] * c_org_0
            return c_aq_z, c_org_z

        # Vectorize over z positions
        c_aq_profile, c_org_profile = jax.vmap(compute_profile_at_z)(z_positions)
        # Shapes: (n_seg+1, n_solutes)

        # Ensure non-negative with soft clipping for smooth gradients
        # (should be automatic for physical systems, this is a safety net)
        c_aq_profile = soft_clip_positive(c_aq_profile)
        c_org_profile = soft_clip_positive(c_org_profile)

        # Final concentrations
        c_aq_final = c_aq_profile[-1]
        c_org_final = c_org_profile[0]  # Organic exits at z=0

        # Convert back to dicts
        raffinate_flows = {s: c_aq_final[i] * F_aq for i, s in enumerate(solutes)}
        extract_flows = {s: c_org_final[i] * F_org for i, s in enumerate(solutes)}

        profiles = {
            "z": z_positions,
            "c_aq": {s: c_aq_profile[:, i] for i, s in enumerate(solutes)},
            "c_org": {s: c_org_profile[:, i] for i, s in enumerate(solutes)},
        }

        return raffinate_flows, extract_flows, profiles

    def _integrate_co_current(
        self,
        c_aq_init: dict,
        c_org_init: dict,
        F_aq: Array,
        F_org: Array,
        K_dict: dict,
        T: Array,
        dz: float,
        n_seg: int,
    ) -> tuple[dict, dict, dict]:
        """Integrate rate equations for co-current flow.

        Both phases enter at z=0 and exit at z=L.
        Straightforward forward integration using lax.scan.

        Fully vectorized over solutes for JAX compatibility.
        """
        p = self.params
        eq = p.equilibrium
        solutes = eq.solutes

        # Convert to arrays for vectorized computation
        K_arr = jnp.array([K_dict[s] for s in solutes])
        if isinstance(p.Kla, dict):
            Kla_arr = jnp.array([p.Kla[s] for s in solutes])
        else:
            Kla_arr = jnp.full(len(solutes), p.Kla)

        c_aq_init_arr = jnp.array([c_aq_init[s] for s in solutes])
        c_org_init_arr = jnp.array([c_org_init[s] for s in solutes])

        area = p.area

        def step(state, _):
            """Single integration step (Euler method)."""
            c_aq, c_org = state

            # Equilibrium concentration in organic phase
            c_eq = c_aq * K_arr

            # Mass transfer rate (vectorized over solutes)
            rate = Kla_arr * (c_eq - c_org) * area * dz

            # Update concentrations with soft clipping for smooth gradients
            c_aq_new = soft_clip_positive(c_aq - rate / F_aq)
            c_org_new = soft_clip_positive(c_org + rate / F_org)

            return (c_aq_new, c_org_new), (c_aq_new, c_org_new)

        # Run integration with lax.scan
        init_state = (c_aq_init_arr, c_org_init_arr)
        (c_aq_final, c_org_final), (c_aq_history, c_org_history) = lax.scan(
            step, init_state, None, length=n_seg
        )

        # Build profiles (prepend initial values)
        # c_aq_history has shape (n_seg, n_solutes)
        c_aq_full = jnp.concatenate([c_aq_init_arr[None, :], c_aq_history], axis=0)
        c_org_full = jnp.concatenate([c_org_init_arr[None, :], c_org_history], axis=0)

        # Convert back to dicts
        raffinate_flows = {s: c_aq_final[i] * F_aq for i, s in enumerate(solutes)}
        extract_flows = {s: c_org_final[i] * F_org for i, s in enumerate(solutes)}

        profiles = {
            "z": jnp.linspace(0, p.length, n_seg + 1),
            "c_aq": {s: c_aq_full[:, i] for i, s in enumerate(solutes)},
            "c_org": {s: c_org_full[:, i] for i, s in enumerate(solutes)},
        }

        return raffinate_flows, extract_flows, profiles


# =============================================================================
# Convenience Functions
# =============================================================================

def separation_factor(K1: Array, K2: Array) -> Array:
    """Calculate separation factor between two solutes.

    SF = K1 / K2

    Higher SF means easier separation.
    """
    return K1 / K2


def minimum_solvent_ratio(K: Array, recovery: float = 0.99) -> Array:
    """Calculate minimum solvent-to-feed ratio for desired recovery.

    For counter-current extraction with infinite stages:
    (S/F)_min = recovery / K

    Args:
        K: Distribution coefficient
        recovery: Desired fraction of solute recovered

    Returns:
        Minimum molar solvent-to-feed ratio
    """
    return recovery / K


def stages_for_recovery(
    K: Array,
    SF_ratio: Array,
    recovery: float = 0.99,
) -> Array:
    """Calculate stages needed for desired recovery.

    Using Kremser equation for counter-current extraction.

    Args:
        K: Distribution coefficient
        SF_ratio: Actual solvent-to-feed ratio / minimum ratio
        recovery: Desired solute recovery

    Returns:
        Number of theoretical stages needed
    """
    E = K * SF_ratio  # Extraction factor

    # From Kremser: N = log((1-1/E) * (1-recovery) + 1/E) / log(1/E)
    # Simplified for E != 1
    N = jnp.log((recovery * (E - 1) + 1) / E) / jnp.log(E)

    return jnp.maximum(N, 1.0)
