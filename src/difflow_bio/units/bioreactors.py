"""Bioreactor unit operations for bio manufacturing.

This module provides bioreactor models for cell culture and fermentation:
- ContinuousBioreactor (Chemostat): Steady-state continuous culture
- FedBatchBioreactor: Fed-batch cultivation with substrate feeding

Key equations:
    Cell growth: dX/dt = (μ - D - k_d) * X
    Substrate: dS/dt = D*(S_f - S) - μ*X/Y_xs - m_s*X
    Product: dP/dt = (α*μ + β)*X - D*P

where:
    X = cell concentration (g/L)
    S = substrate concentration (g/L)
    P = product concentration (g/L)
    μ = specific growth rate (1/h)
    D = dilution rate (1/h)
    Y_xs = yield coefficient (g cells / g substrate)
"""

from typing import Callable, Literal
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, make_stream, get_flows


# =============================================================================
# Kinetic Models
# =============================================================================

def monod_kinetics(S: Array, params: dict) -> Array:
    """Monod growth kinetics.

    μ = μ_max * S / (K_s + S)

    Args:
        S: Substrate concentration (g/L)
        params: Dict with 'mu_max' and 'K_s'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    return mu_max * S / (K_s + S)


def substrate_inhibition_kinetics(S: Array, params: dict) -> Array:
    """Monod kinetics with substrate inhibition (Andrews model).

    μ = μ_max * S / (K_s + S + S²/K_i)

    Args:
        S: Substrate concentration (g/L)
        params: Dict with 'mu_max', 'K_s', 'K_i'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    K_i = params["K_i"]
    return mu_max * S / (K_s + S + S**2 / K_i)


def product_inhibition_kinetics(S: Array, P: Array, params: dict) -> Array:
    """Monod kinetics with product inhibition.

    μ = μ_max * S / (K_s + S) * (1 - P/P_max)^n

    Args:
        S: Substrate concentration (g/L)
        P: Product concentration (g/L)
        params: Dict with 'mu_max', 'K_s', 'P_max', 'n'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    P_max = params["P_max"]
    n = params.get("n", 1.0)

    monod_term = mu_max * S / (K_s + S)
    inhibition_term = jnp.maximum(1 - P / P_max, 0.0) ** n
    return monod_term * inhibition_term


def contois_kinetics(S: Array, X: Array, params: dict) -> Array:
    """Contois growth kinetics (cell-density dependent).

    μ = μ_max * S / (K_s*X + S)

    Args:
        S: Substrate concentration (g/L)
        X: Cell concentration (g/L)
        params: Dict with 'mu_max', 'K_s'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    return mu_max * S / (K_s * X + S)


# =============================================================================
# Bioreactor Parameters
# =============================================================================

@dataclass
class BioreactorParams:
    """Parameters for bioreactor models.

    Attributes:
        V: Reactor volume (L)
        Y_xs: Yield coefficient, cells per substrate (g/g)
        kinetic_fn: Growth kinetics function μ(S, params) or μ(S, X, params)
        kinetic_params: Parameters passed to kinetic function
        k_d: Cell death rate constant (1/h), default 0
        m_s: Maintenance coefficient (g substrate / g cells / h), default 0
        alpha: Growth-associated product formation (g product / g cells)
        beta: Non-growth-associated product formation (g product / g cells / h)
        species_order: List of species names for stream conversion
    """
    V: float | Array
    Y_xs: float | Array
    kinetic_fn: Callable
    kinetic_params: dict
    k_d: float | Array = 0.0
    m_s: float | Array = 0.0
    alpha: float | Array = 0.0
    beta: float | Array = 0.0
    species_order: list[str] = field(default_factory=lambda: ["cells", "substrate", "product"])


# =============================================================================
# Continuous Bioreactor (Chemostat)
# =============================================================================

class ContinuousBioreactor:
    """Continuous stirred-tank bioreactor (chemostat) at steady state.

    Solves steady-state material balances:
        0 = D*(X_in - X) + (μ - k_d)*X
        0 = D*(S_f - S) - μ*X/Y_xs - m_s*X
        0 = D*(P_in - P) + (α*μ + β)*X

    where D = F/V is the dilution rate.

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(self, params: BioreactorParams):
        """Initialize continuous bioreactor.

        Args:
            params: Bioreactor parameters
        """
        self.params = params

    def __call__(
        self,
        inlet: Stream,
        D: float | Array | None = None,
        F: float | Array | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Solve steady-state chemostat balances.

        Args:
            inlet: Feed stream with F_cells, F_substrate, F_product (mass flow, g/h)
                   or concentrations if using D directly
            D: Dilution rate (1/h). If None, calculated from F/V.
            F: Volumetric flow rate (L/h). Used if D is None.

        Returns:
            outlet: Outlet stream
            info: Dictionary with:
                - 'mu': Specific growth rate (1/h)
                - 'X': Cell concentration (g/L)
                - 'S': Substrate concentration (g/L)
                - 'P': Product concentration (g/L)
                - 'D': Dilution rate (1/h)
                - 'productivity': Volumetric productivity (g/L/h)
        """
        p = self.params

        # Get dilution rate
        if D is not None:
            D = jnp.asarray(D)
            F = D * p.V
        elif F is not None:
            F = jnp.asarray(F)
            D = F / p.V
        else:
            raise ValueError("Either D or F must be specified")

        # Get inlet concentrations
        inlet_flows = get_flows(inlet)

        # Convert mass flows to concentrations
        # Assuming inlet flows are g/h, divide by volumetric flow
        X_in = inlet_flows.get("cells", jnp.array(0.0)) / F
        S_f = inlet_flows.get("substrate", jnp.array(0.0)) / F
        P_in = inlet_flows.get("product", jnp.array(0.0)) / F

        # Solve for steady state using fixed-point iteration
        from difflow.solvers import fixed_point_solve

        # Initial guess
        x0 = jnp.array([1.0, S_f * 0.1, 0.1])  # [X, S, P]

        # Capture parameters for closure
        kinetic_fn = p.kinetic_fn
        kinetic_params = p.kinetic_params
        Y_xs = p.Y_xs
        k_d = p.k_d
        m_s = p.m_s
        alpha = p.alpha
        beta = p.beta

        def chemostat_fp(x, args):
            D_, X_in_, S_f_, P_in_ = args

            X, S, P = x[0], x[1], x[2]

            # Ensure positive concentrations
            X = jnp.maximum(X, 1e-10)
            S = jnp.maximum(S, 1e-10)
            P = jnp.maximum(P, 0.0)

            # Growth rate (try both signatures)
            try:
                mu = kinetic_fn(S, kinetic_params)
            except TypeError:
                mu = kinetic_fn(S, X, kinetic_params)

            # Steady-state balances rearranged for fixed-point
            # X: D*(X_in - X) + (μ - k_d)*X = 0
            #    X = D*X_in / (D - μ + k_d) ... but this can be unstable
            # Better: use derivative form with damping

            # Update equations (relaxed)
            dX = D_ * (X_in_ - X) + (mu - k_d) * X
            dS = D_ * (S_f_ - S) - mu * X / Y_xs - m_s * X
            dP = D_ * (P_in_ - P) + (alpha * mu + beta) * X

            # Simple Euler-like update with small step
            dt = 0.5  # pseudo time step for relaxation
            X_new = X + dt * dX
            S_new = S + dt * dS
            P_new = P + dt * dP

            # Enforce bounds
            X_new = jnp.maximum(X_new, 1e-10)
            S_new = jnp.maximum(S_new, 1e-10)
            S_new = jnp.minimum(S_new, S_f_)  # Can't exceed feed
            P_new = jnp.maximum(P_new, 0.0)

            return jnp.array([X_new, S_new, P_new])

        args = (D, X_in, S_f, P_in)
        x_sol = fixed_point_solve(chemostat_fp, x0, args, max_iter=200, damping=0.8)

        X_out, S_out, P_out = x_sol[0], x_sol[1], x_sol[2]

        # Calculate final growth rate
        try:
            mu = kinetic_fn(S_out, kinetic_params)
        except TypeError:
            mu = kinetic_fn(S_out, X_out, kinetic_params)

        # Convert back to mass flows (g/h)
        outlet_flows = {
            "cells": X_out * F,
            "substrate": S_out * F,
            "product": P_out * F,
        }

        outlet = make_stream(outlet_flows, inlet["T"], inlet["P"])

        info = {
            "mu": mu,
            "X": X_out,
            "S": S_out,
            "P": P_out,
            "D": D,
            "productivity": P_out * D,  # g/L/h
            "cell_productivity": X_out * D,
        }

        return outlet, info

    def washout_dilution_rate(self) -> Array:
        """Calculate critical dilution rate for washout.

        At washout, D_crit = μ_max - k_d

        Returns:
            Critical dilution rate (1/h)
        """
        p = self.params
        mu_max = p.kinetic_params.get("mu_max", 1.0)
        return jnp.asarray(mu_max) - jnp.asarray(p.k_d)


# =============================================================================
# Fed-Batch Bioreactor
# =============================================================================

@dataclass
class FedBatchParams:
    """Parameters for fed-batch bioreactor.

    Attributes:
        V0: Initial volume (L)
        Y_xs: Yield coefficient (g cells / g substrate)
        kinetic_fn: Growth kinetics function
        kinetic_params: Parameters for kinetic function
        k_d: Death rate constant (1/h)
        m_s: Maintenance coefficient (g/g/h)
        alpha: Growth-associated product formation (g/g)
        beta: Non-growth-associated product formation (g/g/h)
        species_order: Species names for stream output
    """
    V0: float | Array
    Y_xs: float | Array
    kinetic_fn: Callable
    kinetic_params: dict
    k_d: float | Array = 0.0
    m_s: float | Array = 0.0
    alpha: float | Array = 0.0
    beta: float | Array = 0.0
    species_order: list[str] = field(default_factory=lambda: ["cells", "substrate", "product"])


class FedBatchBioreactor:
    """Fed-batch bioreactor with substrate feeding.

    Simulates batch or fed-batch cultivation by integrating ODEs:
        dV/dt = F(t)
        d(VX)/dt = μ*V*X - k_d*V*X
        d(VS)/dt = F*S_f - μ*V*X/Y_xs - m_s*V*X
        d(VP)/dt = (α*μ + β)*V*X

    Uses JAX-compatible ODE integration via lax.scan.
    """

    def __init__(self, params: FedBatchParams):
        """Initialize fed-batch bioreactor.

        Args:
            params: Fed-batch parameters
        """
        self.params = params

    def __call__(
        self,
        X0: float | Array,
        S0: float | Array,
        P0: float | Array,
        t_final: float | Array,
        feed_rate_fn: Callable[[Array], Array] | None = None,
        S_feed: float | Array = 500.0,
        n_steps: int = 100,
        T: float | Array = 310.0,
        P_pressure: float | Array = 101325.0,
    ) -> tuple[Stream, dict[str, Array]]:
        """Simulate fed-batch cultivation.

        Args:
            X0: Initial cell concentration (g/L)
            S0: Initial substrate concentration (g/L)
            P0: Initial product concentration (g/L)
            t_final: Final time (h)
            feed_rate_fn: Function F(t) returning feed rate (L/h).
                         If None, batch mode (no feeding).
            S_feed: Substrate concentration in feed (g/L)
            n_steps: Number of integration steps
            T: Temperature (K) for output stream
            P_pressure: Pressure (Pa) for output stream

        Returns:
            outlet: Final state as stream (total mass of each species)
            info: Dictionary with time profiles:
                - 't': Time array (h)
                - 'X': Cell concentration profile (g/L)
                - 'S': Substrate concentration profile (g/L)
                - 'P': Product concentration profile (g/L)
                - 'V': Volume profile (L)
                - 'mu': Growth rate profile (1/h)
        """
        p = self.params

        # Default to batch (no feed)
        if feed_rate_fn is None:
            feed_rate_fn = lambda t: jnp.array(0.0)

        # Time step
        dt = t_final / n_steps
        t_array = jnp.linspace(0, t_final, n_steps + 1)

        # Initial state: [V, VX, VS, VP]
        V0 = jnp.asarray(p.V0)
        VX0 = V0 * jnp.asarray(X0)
        VS0 = V0 * jnp.asarray(S0)
        VP0 = V0 * jnp.asarray(P0)
        y0 = jnp.array([V0, VX0, VS0, VP0])

        # Capture parameters
        kinetic_fn = p.kinetic_fn
        kinetic_params = p.kinetic_params
        Y_xs = jnp.asarray(p.Y_xs)
        k_d = jnp.asarray(p.k_d)
        m_s = jnp.asarray(p.m_s)
        alpha = jnp.asarray(p.alpha)
        beta = jnp.asarray(p.beta)
        S_f = jnp.asarray(S_feed)

        def rhs(y, t):
            """Right-hand side of ODEs."""
            V, VX, VS, VP = y[0], y[1], y[2], y[3]

            # Concentrations
            X = VX / jnp.maximum(V, 1e-10)
            S = VS / jnp.maximum(V, 1e-10)
            P = VP / jnp.maximum(V, 1e-10)

            # Ensure positive
            X = jnp.maximum(X, 1e-10)
            S = jnp.maximum(S, 1e-10)

            # Growth rate
            try:
                mu = kinetic_fn(S, kinetic_params)
            except TypeError:
                mu = kinetic_fn(S, X, kinetic_params)

            # Feed rate
            F = feed_rate_fn(t)

            # ODEs in extensive form
            dV_dt = F
            dVX_dt = mu * V * X - k_d * V * X
            dVS_dt = F * S_f - mu * V * X / Y_xs - m_s * V * X
            dVP_dt = (alpha * mu + beta) * V * X

            return jnp.array([dV_dt, dVX_dt, dVS_dt, dVP_dt])

        def euler_step(y, t):
            """Simple Euler integration step."""
            dy = rhs(y, t)
            y_new = y + dt * dy
            # Enforce positivity
            y_new = jnp.maximum(y_new, 1e-10)
            return y_new, y

        # Integrate
        y_final, y_history = lax.scan(euler_step, y0, t_array[:-1])

        # Append final state to history
        y_all = jnp.vstack([y_history, y_final[None, :]])

        # Extract profiles
        V_profile = y_all[:, 0]
        X_profile = y_all[:, 1] / jnp.maximum(V_profile, 1e-10)
        S_profile = y_all[:, 2] / jnp.maximum(V_profile, 1e-10)
        P_profile = y_all[:, 3] / jnp.maximum(V_profile, 1e-10)

        # Calculate mu profile
        def calc_mu(S, X):
            try:
                return kinetic_fn(S, kinetic_params)
            except TypeError:
                return kinetic_fn(S, X, kinetic_params)

        mu_profile = jax_vmap_safe(calc_mu, S_profile, X_profile)

        # Final state as stream (total mass)
        V_final = y_final[0]
        outlet_flows = {
            "cells": y_final[1],      # Total cell mass (g)
            "substrate": y_final[2],  # Total substrate mass (g)
            "product": y_final[3],    # Total product mass (g)
        }

        outlet = make_stream(outlet_flows, T, P_pressure)

        info = {
            "t": t_array,
            "X": X_profile,
            "S": S_profile,
            "P": P_profile,
            "V": V_profile,
            "mu": mu_profile,
            "V_final": V_final,
            "X_final": X_profile[-1],
            "S_final": S_profile[-1],
            "P_final": P_profile[-1],
        }

        return outlet, info


def jax_vmap_safe(fn, S_arr, X_arr):
    """Vectorized map that handles kinetic function signatures."""
    import jax
    # Just compute element-wise in a scan to avoid vmap issues
    def step(_, args):
        S, X = args
        try:
            mu = fn(S, X)
        except:
            mu = fn(S, None)
        return None, mu

    _, mu_arr = lax.scan(step, None, (S_arr, X_arr))
    return mu_arr


# =============================================================================
# Utility Functions
# =============================================================================

def dilution_rate(F: Array, V: Array) -> Array:
    """Calculate dilution rate D = F/V.

    Args:
        F: Volumetric flow rate (L/h)
        V: Reactor volume (L)

    Returns:
        Dilution rate (1/h)
    """
    return F / V


def residence_time(D: Array) -> Array:
    """Calculate residence time τ = 1/D.

    Args:
        D: Dilution rate (1/h)

    Returns:
        Residence time (h)
    """
    return 1.0 / D


def optimal_dilution_rate(params: dict) -> Array:
    """Calculate optimal D for maximum productivity in chemostat.

    For Monod kinetics: D_opt = μ_max * (1 - sqrt(K_s / (K_s + S_f)))

    Args:
        params: Dict with 'mu_max', 'K_s', 'S_f'

    Returns:
        Optimal dilution rate (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    S_f = params["S_f"]

    return mu_max * (1 - jnp.sqrt(K_s / (K_s + S_f)))
