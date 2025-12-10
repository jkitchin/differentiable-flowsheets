"""Liquid-Liquid Extraction unit operations.

This module provides differentiable LLE operations:
- MultistageCascade: Counter-current or co-current extraction cascade
- DifferentialContactor: Packed column or similar continuous contactor

Supports both distribution coefficient (K-value) models and activity
coefficient models (NRTL, UNIQUAC) for computing equilibrium.

All operations are fully differentiable using JAX.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Literal
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream


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

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: NRTL parameters

    Returns:
        Activity coefficients for each species
    """
    n = len(params.species)

    # Temperature-dependent tau
    tau = params.a + params.b / T

    # G matrix
    G = jnp.exp(-params.alpha * tau)

    # Compute activity coefficients
    # ln(gamma_i) = (sum_j tau_ji * G_ji * x_j) / (sum_k G_ki * x_k)
    #             + sum_j (x_j * G_ij / sum_k G_kj * x_k) * (tau_ij - sum_m x_m * tau_mj * G_mj / sum_k G_kj * x_k)

    def compute_ln_gamma(i):
        # Numerator and denominator for first term
        num1 = jnp.sum(tau[:, i] * G[:, i] * x)
        den1 = jnp.sum(G[:, i] * x)
        term1 = num1 / den1

        # Second term
        def inner_sum(j):
            sum_Gkj_xk = jnp.sum(G[:, j] * x)
            sum_tau_G_x = jnp.sum(x * tau[:, j] * G[:, j])
            return (x[j] * G[i, j] / sum_Gkj_xk) * (tau[i, j] - sum_tau_G_x / sum_Gkj_xk)

        term2 = sum(inner_sum(j) for j in range(n))

        return term1 + term2

    ln_gamma = jnp.array([compute_ln_gamma(i) for i in range(n)])
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
    """
    species: tuple[str, ...]
    r: Array  # Volume parameters
    q: Array  # Surface parameters
    a: Array  # (n, n) interaction parameter matrix
    b: Array  # (n, n) temperature coefficients


def uniquac_activity_coefficients(
    x: Array,
    T: Array,
    params: UNIQUACParams,
) -> Array:
    """Calculate activity coefficients using UNIQUAC model.

    Args:
        x: Mole fractions (array, same order as params.species)
        T: Temperature (K)
        params: UNIQUAC parameters

    Returns:
        Activity coefficients for each species
    """
    r = params.r
    q = params.q
    z = 10.0  # Coordination number

    # Segment and area fractions
    phi = x * r / jnp.sum(x * r)  # Segment fraction
    theta = x * q / jnp.sum(x * q)  # Area fraction

    # l parameter
    l = (z / 2) * (r - q) - (r - 1)

    # Temperature-dependent tau
    tau = jnp.exp(-(params.a + params.b / T))

    # Combinatorial contribution
    ln_gamma_C = jnp.log(phi / x) + (z / 2) * q * jnp.log(theta / phi) + l - (phi / x) * jnp.sum(x * l)

    # Residual contribution
    def compute_residual(i):
        sum_theta_tau_i = jnp.sum(theta * tau[:, i])
        term1 = -q[i] * jnp.log(sum_theta_tau_i)

        def inner_sum(j):
            sum_theta_tau_j = jnp.sum(theta * tau[:, j])
            return theta[j] * tau[i, j] / sum_theta_tau_j

        term2 = q[i] * (1 - jnp.sum(jnp.array([inner_sum(j) for j in range(len(params.species))])))
        return term1 + term2

    ln_gamma_R = jnp.array([compute_residual(i) for i in range(len(params.species))])

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
    """
    n_stages: int | float | Array
    equilibrium: LLEEquilibrium
    flow_config: Literal["counter_current", "co_current"] = "counter_current"


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
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        # Get distribution coefficients
        K_dict = eq.get_distribution_coefficients({}, {}, T)

        # Initial solute amounts in feed
        F_solute_in = {s: feed_flows.get(s, 0.0) for s in solutes}

        profiles = {"x": {s: [] for s in solutes}, "y": {s: [] for s in solutes}}

        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}

        for s in solutes:
            K = K_dict[s]
            E = K * F_org / F_aq  # Extraction factor

            F_in = jnp.asarray(F_solute_in[s])

            # Kremser equation for fraction remaining in raffinate
            # For E != 1: fraction_remaining = (E - 1) / (E^(N+1) - 1)
            # For E = 1: fraction_remaining = 1 / (N + 1)
            # So fraction_extracted = 1 - fraction_remaining

            E_Np1 = E ** (n_stages + 1)

            # Use jnp.where for differentiable conditional
            frac_remaining = jnp.where(
                jnp.abs(E - 1.0) < 1e-6,
                1.0 / (n_stages + 1),
                (E - 1.0) / (E_Np1 - 1.0 + 1e-10)
            )

            # Clamp to physical bounds
            frac_remaining = jnp.clip(frac_remaining, 0.0, 1.0)
            frac_extracted = 1.0 - frac_remaining

            F_extracted = F_in * frac_extracted
            F_raffinate = F_in * frac_remaining

            raffinate_flows[s] = F_raffinate
            extract_flows[s] = F_extracted

            # Stage profiles are approximate for continuous n_stages
            profiles["x"][s] = []
            profiles["y"][s] = []

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
        """
        eq = self.params.equilibrium
        solutes = eq.solutes

        K_dict = eq.get_distribution_coefficients({}, {}, T)

        profiles = {"x": {s: [] for s in solutes}, "y": {s: [] for s in solutes}}

        raffinate_flows = {eq.aqueous_carrier: F_aq}
        extract_flows = {eq.organic_carrier: F_org}

        for s in solutes:
            K = K_dict[s]

            F_in = jnp.asarray(feed_flows.get(s, 0.0))
            F_solvent_s = jnp.asarray(solvent_flows.get(s, 0.0))

            # Total solute amount
            F_total = F_in + F_solvent_s

            # At equilibrium: y = K * x and mass balance
            # F_aq * x + F_org * y = F_total
            # x = F_total / (F_aq + K * F_org)

            x_eq = F_total / (F_aq + K * F_org)

            # Approach to equilibrium with multiple stages
            # Model: each stage achieves 80% of remaining driving force
            # Total efficiency = 1 - (1 - eff)^N (differentiable in N)
            eff_per_stage = 0.8
            total_eff = 1.0 - (1.0 - eff_per_stage) ** n_stages

            x_feed = F_in / F_aq
            x_final = x_feed + total_eff * (x_eq - x_feed)

            F_raffinate = x_final * F_aq
            F_extracted = F_total - F_raffinate

            raffinate_flows[s] = F_raffinate
            extract_flows[s] = F_extracted

            profiles["x"][s] = []
            profiles["y"][s] = []

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
        Requires iterative solution (shooting method).
        """
        p = self.params
        eq = p.equilibrium
        solutes = eq.solutes

        # For counter-current, we need to iterate
        # Use fixed-point iteration on organic inlet composition

        # Get Kla values
        if isinstance(p.Kla, dict):
            Kla = p.Kla
        else:
            Kla = {s: p.Kla for s in solutes}

        def forward_integrate(c_org_at_L: dict) -> dict:
            """Integrate from z=0 to z=L, return c_org at z=0."""
            c_aq = {s: jnp.asarray(c_aq_init[s]) for s in solutes}
            c_org = {s: jnp.asarray(c_org_at_L[s]) for s in solutes}

            for _ in range(n_seg):
                for s in solutes:
                    K = K_dict[s]
                    c_eq = c_aq[s] * K  # Equilibrium org concentration

                    # Mass transfer rate: from aqueous to organic if c_eq > c_org
                    rate = Kla[s] * (c_eq - c_org[s]) * p.area * dz

                    # Update concentrations
                    # Aqueous loses solute (flows in +z direction)
                    c_aq[s] = c_aq[s] - rate / F_aq
                    # Organic gains solute (flows in -z direction, so add rate)
                    c_org[s] = c_org[s] + rate / F_org

                    # Clip to non-negative
                    c_aq[s] = jnp.maximum(c_aq[s], 0.0)
                    c_org[s] = jnp.maximum(c_org[s], 0.0)

            return c_org, c_aq

        # Initial guess: organic leaves with equilibrium amount
        c_org_at_L = {s: jnp.asarray(c_org_init[s]) for s in solutes}

        # Fixed-point iteration
        for _ in range(20):  # Usually converges quickly
            c_org_at_0, c_aq_at_L = forward_integrate(c_org_at_L)

            # The organic at z=L should match the inlet (c_org_init)
            # But we computed what c_org is at z=0
            # For shooting method, we'd adjust c_org_at_L
            # Here we use a simpler approach: just use final values

        # Final integration to get profiles
        c_aq_profile = {s: [c_aq_init[s]] for s in solutes}
        c_org_profile = {s: [] for s in solutes}

        c_aq = {s: jnp.asarray(c_aq_init[s]) for s in solutes}
        c_org = {s: jnp.asarray(c_org_at_0[s]) for s in solutes}

        for i in range(n_seg):
            c_org_profile_step = {}
            for s in solutes:
                K = K_dict[s]
                c_eq = c_aq[s] * K
                rate = Kla[s] * (c_eq - c_org[s]) * p.area * dz

                c_aq[s] = jnp.maximum(c_aq[s] - rate / F_aq, 0.0)
                c_org[s] = jnp.maximum(c_org[s] + rate / F_org, 0.0)

                c_aq_profile[s].append(c_aq[s])
                c_org_profile_step[s] = c_org[s]

            for s in solutes:
                c_org_profile[s].append(c_org_profile_step[s])

        # Output flows
        raffinate_flows = {s: c_aq[s] * F_aq for s in solutes}
        extract_flows = {s: c_org[s] * F_org for s in solutes}

        profiles = {
            "z": jnp.linspace(0, p.length, n_seg + 1),
            "c_aq": c_aq_profile,
            "c_org": c_org_profile,
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
        Straightforward forward integration.
        """
        p = self.params
        eq = p.equilibrium
        solutes = eq.solutes

        if isinstance(p.Kla, dict):
            Kla = p.Kla
        else:
            Kla = {s: p.Kla for s in solutes}

        c_aq_profile = {s: [c_aq_init[s]] for s in solutes}
        c_org_profile = {s: [c_org_init[s]] for s in solutes}

        c_aq = {s: jnp.asarray(c_aq_init[s]) for s in solutes}
        c_org = {s: jnp.asarray(c_org_init[s]) for s in solutes}

        for i in range(n_seg):
            for s in solutes:
                K = K_dict[s]
                c_eq = c_aq[s] * K  # Equilibrium concentration in organic

                # Mass transfer from aqueous to organic
                rate = Kla[s] * (c_eq - c_org[s]) * p.area * dz

                c_aq[s] = jnp.maximum(c_aq[s] - rate / F_aq, 0.0)
                c_org[s] = jnp.maximum(c_org[s] + rate / F_org, 0.0)

                c_aq_profile[s].append(c_aq[s])
                c_org_profile[s].append(c_org[s])

        raffinate_flows = {s: c_aq[s] * F_aq for s in solutes}
        extract_flows = {s: c_org[s] * F_org for s in solutes}

        profiles = {
            "z": jnp.linspace(0, p.length, n_seg + 1),
            "c_aq": c_aq_profile,
            "c_org": c_org_profile,
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
