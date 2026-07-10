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

Numerical Considerations:
- Monod singularity: When K_s → 0 and S → 0, regularization prevents 0/0
- Kinetic dispatch: Uses inspect.signature instead of try/except for JAX compatibility
- Integration: Uses RK4 instead of Euler for better accuracy
- Fixed-point: Adaptive relaxation for robust convergence
"""

from typing import Callable, Literal
from dataclasses import dataclass, field

from difflow.params_mixin import ParamsMixin
import inspect
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, make_stream, get_flows


# =============================================================================
# Numerical Constants
# =============================================================================

# Minimum concentration to prevent division by zero in Monod-type kinetics.
# When S → 0 and K_s → 0 simultaneously, S/(K_s + S) → 0/0.
# Adding MIN_CONC to denominator ensures numerical stability.
MIN_CONC = 1e-10

# Relaxation factor range for fixed-point iteration.
# Smaller values → more stable but slower convergence.
# Larger values → faster but may oscillate.
MIN_RELAXATION = 0.1
MAX_RELAXATION = 0.8
DEFAULT_RELAXATION = 0.3


# =============================================================================
# Kinetic Models
# =============================================================================

def monod_kinetics(S: Array, params: dict) -> Array:
    """Monod growth kinetics.

    μ = μ_max * S / (K_s + S)

    Handles the singularity when K_s → 0 and S → 0:
    - Adds MIN_CONC to denominator to prevent 0/0
    - Physically: at very low S, growth rate is substrate-limited

    Args:
        S: Substrate concentration (g/L)
        params: Dict with 'mu_max' and 'K_s'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    # Regularize denominator to prevent 0/0 when K_s = 0 and S = 0
    denom = K_s + S + MIN_CONC
    return mu_max * S / denom


def substrate_inhibition_kinetics(S: Array, params: dict) -> Array:
    """Monod kinetics with substrate inhibition (Andrews model).

    μ = μ_max * S / (K_s + S + S²/K_i)

    Handles singularities:
    - K_s = S = 0: Regularized denominator
    - K_i → 0: Would cause S²/K_i → ∞, but K_i should be positive

    Args:
        S: Substrate concentration (g/L)
        params: Dict with 'mu_max', 'K_s', 'K_i'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    K_i = jnp.maximum(params["K_i"], MIN_CONC)  # Prevent division by zero
    # Regularize denominator
    denom = K_s + S + S**2 / K_i + MIN_CONC
    return mu_max * S / denom


def product_inhibition_kinetics(S: Array, P: Array, params: dict) -> Array:
    """Monod kinetics with product inhibition.

    μ = μ_max * S / (K_s + S) * (1 - P/P_max)^n

    Handles singularities:
    - K_s = S = 0: Regularized denominator
    - P_max → 0: Would be unphysical (no product tolerance)

    Args:
        S: Substrate concentration (g/L)
        P: Product concentration (g/L)
        params: Dict with 'mu_max', 'K_s', 'P_max', 'n'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    P_max = jnp.maximum(params["P_max"], MIN_CONC)  # Prevent division by zero
    n = params.get("n", 1.0)

    # Regularize Monod term
    denom = K_s + S + MIN_CONC
    monod_term = mu_max * S / denom
    inhibition_term = jnp.maximum(1 - P / P_max, 0.0) ** n
    return monod_term * inhibition_term


def contois_kinetics(S: Array, X: Array, params: dict) -> Array:
    """Contois growth kinetics (cell-density dependent).

    μ = μ_max * S / (K_s*X + S)

    Handles singularities:
    - K_s*X + S → 0: Regularized denominator

    Args:
        S: Substrate concentration (g/L)
        X: Cell concentration (g/L)
        params: Dict with 'mu_max', 'K_s'

    Returns:
        Specific growth rate μ (1/h)
    """
    mu_max = params["mu_max"]
    K_s = params["K_s"]
    # Regularize denominator
    denom = K_s * X + S + MIN_CONC
    return mu_max * S / denom


# =============================================================================
# Kinetic Function Dispatch
# =============================================================================

def get_kinetic_arity(kinetic_fn: Callable) -> int:
    """Determine number of state arguments for kinetic function.

    Inspects the function signature to determine if it takes:
    - 2 args: μ(S, params) - substrate-only kinetics (Monod, Andrews)
    - 3 args: μ(S, X, params) - cell-density dependent (Contois)
    - 4 args: μ(S, P, params) or μ(S, X, P, params) - product inhibition

    This is done at initialization time (not JIT time) to avoid
    try/except dispatch which breaks JAX tracing.

    Args:
        kinetic_fn: Kinetic rate function

    Returns:
        Number of positional arguments (excluding params dict)
    """
    sig = inspect.signature(kinetic_fn)
    params = list(sig.parameters.values())
    # Count positional parameters (excluding 'params' dict which is last)
    n_args = len([p for p in params if p.name != 'params'])
    return n_args


def call_kinetics(kinetic_fn: Callable, arity: int, S: Array, X: Array,
                  P: Array, params: dict) -> Array:
    """Call kinetic function with appropriate arguments based on arity.

    This avoids try/except dispatch which breaks JAX JIT.

    Args:
        kinetic_fn: Kinetic rate function
        arity: Number of state arguments (from get_kinetic_arity)
        S: Substrate concentration
        X: Cell concentration
        P: Product concentration
        params: Kinetic parameters dict

    Returns:
        Specific growth rate μ
    """
    if arity == 1:
        return kinetic_fn(S, params)
    elif arity == 2:
        # 2 state args: could be (S, P) for product inhibition
        # or (S, X) for Contois.  Inspect parameter names to decide.
        sig = inspect.signature(kinetic_fn)
        param_names = [p.name for p in sig.parameters.values() if p.name != 'params']
        if 'P' in param_names:
            return kinetic_fn(S, P, params)
        else:
            return kinetic_fn(S, X, params)
    else:
        # 3+ args: assume (S, X, P, params)
        return kinetic_fn(S, X, P, params)


# =============================================================================
# Bioreactor Parameters
# =============================================================================

@dataclass(repr=False)
class BioreactorParams(ParamsMixin):
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
        _kinetic_arity: Cached arity of kinetic function (auto-detected)
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
    _kinetic_arity: int = field(default=-1, repr=False)

    def __post_init__(self):
        """Auto-detect kinetic function arity."""
        if self._kinetic_arity < 0:
            object.__setattr__(self, '_kinetic_arity', get_kinetic_arity(self.kinetic_fn))


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

    symbol = "Chemostat"
    equations = [
        r"0 = D\,(X_\mathrm{in} - X) + (\mu - k_d)\,X",
        r"0 = D\,(S_f - S) - \mu\,X/Y_{xs} - m_s\,X",
        r"0 = D\,(P_\mathrm{in} - P) + (\alpha\,\mu + \beta)\,X",
        r"\mu = \mu_\mathrm{max}\,\frac{S}{K_S + S}\qquad \text{(Monod; kinetic\_fn override allowed)}",
    ]
    assumptions = [
        "Perfectly mixed liquid phase.",
        "Steady-state continuous operation.",
        "Constant Y_xs, alpha, beta over the operating window.",
        "Single growth-limiting substrate.",
    ]
    references = [
        "Monod, J. Ann. Rev. Microbiology, 3, 371 (1949).",
        "Bailey, J.E., Ollis, D.F. Biochemical Engineering Fundamentals, 2e, McGraw-Hill, 1986.",
    ]
    parameter_symbols = {
        "V": "V",
        "Y_xs": "Y_{xs}",
        "k_d": "k_d",
        "m_s": "m_s",
        "alpha": r"\alpha",
        "beta": r"\beta",
    }
    parameter_units = {
        "V": "L",
        "Y_xs": "g/g",
        "k_d": "1/h",
        "m_s": "g/g/h",
        "alpha": "g/g",
        "beta": "g/g/h",
    }
    numerical_method = "Fixed-point / Newton on the steady-state mass balances."

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
        import optimistix as optx

        # Initial guess
        x0 = jnp.array([1.0, S_f * 0.1, 0.1])  # [X, S, P]

        # Capture parameters for closure (ensure JAX arrays for consistent AD)
        kinetic_fn = p.kinetic_fn
        kinetic_params = p.kinetic_params
        Y_xs = jnp.asarray(p.Y_xs)
        k_d = jnp.asarray(p.k_d)
        m_s = jnp.asarray(p.m_s)
        alpha = jnp.asarray(p.alpha)
        beta = jnp.asarray(p.beta)

        # Get kinetic arity for proper dispatch (avoids try/except in JIT)
        kinetic_arity = p._kinetic_arity

        def chemostat_fp(x, args):
            D_, X_in_, S_f_, P_in_ = args

            X, S, P = x[0], x[1], x[2]

            # Ensure positive concentrations
            X = jnp.maximum(X, MIN_CONC)
            S = jnp.maximum(S, MIN_CONC)
            P = jnp.maximum(P, 0.0)

            # Growth rate using arity-based dispatch (JAX JIT compatible)
            mu = call_kinetics(kinetic_fn, kinetic_arity, S, X, P, kinetic_params)

            # Steady-state balances rearranged for fixed-point
            # X: D*(X_in - X) + (μ - k_d)*X = 0
            #    X = D*X_in / (D - μ + k_d) ... but this can be unstable
            # Better: use derivative form with damping

            # Update equations (relaxed)
            dX = D_ * (X_in_ - X) + (mu - k_d) * X
            dS = D_ * (S_f_ - S) - mu * X / Y_xs - m_s * X
            dP = D_ * (P_in_ - P) + (alpha * mu + beta) * X

            # Adaptive relaxation factor based on residual magnitude
            # Smaller steps when changes are large (more stable)
            # Larger steps when near convergence (faster)
            residual_norm = jnp.sqrt(dX**2 + dS**2 + dP**2)
            # Scale relaxation: small residual → larger step, large residual → smaller step
            dt = DEFAULT_RELAXATION / (1.0 + residual_norm)
            dt = jnp.clip(dt, MIN_RELAXATION, MAX_RELAXATION)

            X_new = X + dt * dX
            S_new = S + dt * dS
            P_new = P + dt * dP

            # Enforce bounds
            X_new = jnp.maximum(X_new, MIN_CONC)
            S_new = jnp.maximum(S_new, MIN_CONC)
            S_new = jnp.minimum(S_new, S_f_)  # Can't exceed feed
            P_new = jnp.maximum(P_new, 0.0)

            return jnp.array([X_new, S_new, P_new])

        args = (D, X_in, S_f, P_in)
        fp_solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
        sol = optx.fixed_point(chemostat_fp, fp_solver, x0, args=args, max_steps=200, throw=False)
        x_sol = sol.value

        X_out, S_out, P_out = x_sol[0], x_sol[1], x_sol[2]

        # Calculate final growth rate using arity-based dispatch
        mu = call_kinetics(kinetic_fn, kinetic_arity, S_out, X_out, P_out, kinetic_params)

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

@dataclass(repr=False)
class FedBatchParams(ParamsMixin):
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
        _kinetic_arity: Cached arity of kinetic function (auto-detected)
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
    # Oxygen transfer coupling (#101). When kLa is set, a dissolved-O2 balance
    # dC_O2/dt = kLa*(C* - C_O2) - OUR is added and growth is limited by O2
    # through a Monod factor C_O2/(K_O2 + C_O2), so a feed rate that outstrips
    # the oxygen transfer capacity drives the culture O2-limited. Defaults
    # (kLa=None) disable the balance (backward compatible).
    kLa: float | Array | None = None  # Volumetric O2 mass-transfer coeff (1/h)
    Y_xo: float | Array = 1.0  # Biomass yield on O2 (g cells/g O2); Doran 2013, Ch. 11
    m_o: float | Array = 0.0  # Maintenance O2 coeff (g O2/g cells/h)
    # Saturation dissolved O2 for air-water ~7 mg/L at 35 C
    # (Doran, Bioprocess Engineering Principles, 2e, 2013, Table 9.2).
    C_O2_star: float | Array = 7.0e-3  # g/L
    # Critical/half-saturation DO for aerobic growth ~0.1 mg/L
    # (Bailey & Ollis, Biochemical Engineering Fundamentals, 2e, 1986, Ch. 8).
    K_O2: float | Array = 1.0e-4  # g/L
    C_O2_0: float | Array | None = None  # Initial DO (g/L); default = C_O2_star
    _kinetic_arity: int = field(default=-1, repr=False)

    def __post_init__(self):
        """Auto-detect kinetic function arity."""
        if self._kinetic_arity < 0:
            object.__setattr__(self, '_kinetic_arity', get_kinetic_arity(self.kinetic_fn))


class FedBatchBioreactor:
    """Fed-batch bioreactor with substrate feeding.

    Simulates batch or fed-batch cultivation by integrating ODEs:
        dV/dt = F(t)
        d(VX)/dt = μ*V*X - k_d*V*X
        d(VS)/dt = F*S_f - μ*V*X/Y_xs - m_s*V*X
        d(VP)/dt = (α*μ + β)*V*X

    Integration methods:
    - "diffrax" (default): Adaptive step-size with Tsit5 solver
    - "diffrax:dopri5", "diffrax:kvaerno5", etc.: Specific diffrax solvers
    - "rk4": Fixed-step RK4 via lax.scan (fallback if diffrax unavailable)

    For stiff kinetics (e.g., substrate inhibition with high K_i),
    use "diffrax:kvaerno5" implicit solver.
    """

    symbol = "Fed-Batch Bioreactor"
    equations = [
        r"\frac{dV}{dt} = F(t)",
        r"\frac{d(VX)}{dt} = (\mu - k_d)\,V X",
        r"\frac{d(VS)}{dt} = F(t)\,S_f - \mu V X / Y_{xs} - m_s V X",
        r"\frac{d(VP)}{dt} = (\alpha\,\mu + \beta)\,V X",
    ]
    assumptions = [
        "Perfectly mixed liquid phase with time-varying volume.",
        "Single growth-limiting substrate and one lumped product.",
        "Isothermal, constant-pH operation.",
    ]
    references = [
        "Bailey, J.E., Ollis, D.F. Biochemical Engineering Fundamentals, 2e, McGraw-Hill, 1986.",
        "Shuler, M.L., Kargi, F. Bioprocess Engineering: Basic Concepts, 2e, Prentice Hall, 2002.",
    ]
    parameter_symbols = {
        "V0": "V_0",
        "Y_xs": "Y_{xs}",
        "k_d": "k_d",
        "m_s": "m_s",
        "alpha": r"\alpha",
        "beta": r"\beta",
    }
    parameter_units = {"V0": "L", "Y_xs": "g/g", "k_d": "1/h", "m_s": "g/g/h"}
    numerical_method = "Adaptive diffrax (Tsit5 / Kvaerno5) integration of (V, VX, VS, VP)."

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
        solver: str = "diffrax",
        rtol: float = 1e-5,
        atol: float = 1e-7,
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
            n_steps: Number of output points (for diffrax) or steps (for rk4)
            T: Temperature (K) for output stream
            P_pressure: Pressure (Pa) for output stream
            solver: Integration method:
                - "diffrax" or "diffrax:tsit5" (default): Adaptive Tsit5
                - "diffrax:dopri5": Dormand-Prince 5(4)
                - "diffrax:kvaerno5": Implicit (for stiff systems)
                - "rk4": Fixed-step RK4 (fallback)
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers

        Returns:
            outlet: Final state as stream (total mass of each species)
            info: Dictionary with time profiles:
                - 't': Time array (h)
                - 'X': Cell concentration profile (g/L)
                - 'S': Substrate concentration profile (g/L)
                - 'P': Product concentration profile (g/L)
                - 'V': Volume profile (L)
                - 'mu': Growth rate profile (1/h)
                - 'solver': Solver used
        """
        p = self.params

        # Default to batch (no feed)
        if feed_rate_fn is None:
            feed_rate_fn = lambda t: jnp.array(0.0)

        # Time step
        dt = t_final / n_steps
        t_array = jnp.linspace(0, t_final, n_steps + 1)

        # Oxygen transfer coupling (#101): add a dissolved-O2 state when kLa set
        oxygen_active = p.kLa is not None

        # Initial state: [V, VX, VS, VP] (+ VO2 when oxygen tracking is on)
        V0 = jnp.asarray(p.V0)
        VX0 = V0 * jnp.asarray(X0)
        VS0 = V0 * jnp.asarray(S0)
        VP0 = V0 * jnp.asarray(P0)
        if oxygen_active:
            C_O2_0 = p.C_O2_star if p.C_O2_0 is None else p.C_O2_0
            VO2_0 = V0 * jnp.asarray(C_O2_0)
            y0 = jnp.array([V0, VX0, VS0, VP0, VO2_0])
        else:
            y0 = jnp.array([V0, VX0, VS0, VP0])

        # Capture parameters
        kinetic_fn = p.kinetic_fn
        kinetic_params = p.kinetic_params
        kinetic_arity = p._kinetic_arity  # Pre-computed arity for JAX JIT
        Y_xs = jnp.asarray(p.Y_xs)
        k_d = jnp.asarray(p.k_d)
        m_s = jnp.asarray(p.m_s)
        alpha = jnp.asarray(p.alpha)
        beta = jnp.asarray(p.beta)
        S_f = jnp.asarray(S_feed)
        kLa = jnp.asarray(p.kLa) if oxygen_active else None
        Y_xo = jnp.asarray(p.Y_xo)
        m_o = jnp.asarray(p.m_o)
        C_O2_star = jnp.asarray(p.C_O2_star)
        K_O2 = jnp.asarray(p.K_O2)

        def rhs(y, t):
            """Right-hand side of ODEs."""
            V, VX, VS, VP = y[0], y[1], y[2], y[3]

            # Concentrations
            X = VX / jnp.maximum(V, MIN_CONC)
            S = VS / jnp.maximum(V, MIN_CONC)
            P = VP / jnp.maximum(V, MIN_CONC)

            # Ensure positive
            X = jnp.maximum(X, MIN_CONC)
            S = jnp.maximum(S, MIN_CONC)

            # Growth rate using arity-based dispatch (JAX JIT compatible)
            mu = call_kinetics(kinetic_fn, kinetic_arity, S, X, P, kinetic_params)

            # Feed rate
            F = feed_rate_fn(t)

            if oxygen_active:
                # Dissolved O2 concentration and O2-limitation of growth
                C_O2 = jnp.maximum(y[4] / jnp.maximum(V, MIN_CONC), 0.0)
                o2_factor = C_O2 / (K_O2 + C_O2)
                mu = mu * o2_factor
                # Oxygen uptake (growth + maintenance) and transfer
                OUR = (mu / Y_xo + m_o) * X            # g O2/L/h
                OTR = kLa * (C_O2_star - C_O2)          # g O2/L/h
                dVO2_dt = V * (OTR - OUR)               # feed O2 ~ 0

            dV_dt = F
            dVX_dt = mu * V * X - k_d * V * X
            dVS_dt = F * S_f - mu * V * X / Y_xs - m_s * V * X
            dVP_dt = (alpha * mu + beta) * V * X

            if oxygen_active:
                return jnp.array([dV_dt, dVX_dt, dVS_dt, dVP_dt, dVO2_dt])
            return jnp.array([dV_dt, dVX_dt, dVS_dt, dVP_dt])

        # Choose integration method
        use_diffrax = solver.startswith("diffrax")

        if use_diffrax:
            # Use diffrax for adaptive integration
            try:
                from difflow.dynamic.diffrax_backend import integrate_diffrax

                # Parse solver name (e.g., "diffrax:kvaerno5" -> "kvaerno5")
                if ":" in solver:
                    diffrax_solver = solver.split(":")[1]
                else:
                    diffrax_solver = "tsit5"  # Default

                # Wrap rhs for diffrax (expects f(t, y))
                def diffrax_rhs(t, y):
                    return rhs(y, t)

                result = integrate_diffrax(
                    diffrax_rhs,
                    y0,
                    t_span=(0.0, t_final),
                    solver=diffrax_solver,
                    rtol=rtol,
                    atol=atol,
                    saveat=t_array,
                )

                y_all = result.trajectory.y
                y_final = result.y_final
                solver_used = f"diffrax:{diffrax_solver}"

            except ImportError:
                # Fall back to RK4 if diffrax not available
                use_diffrax = False
                solver_used = "rk4 (diffrax unavailable)"

        if not use_diffrax:
            # Use fixed-step RK4 via lax.scan
            def rk4_step(y, t):
                """Fourth-order Runge-Kutta integration step."""
                k1 = rhs(y, t)
                k2 = rhs(y + 0.5 * dt * k1, t + 0.5 * dt)
                k3 = rhs(y + 0.5 * dt * k2, t + 0.5 * dt)
                k4 = rhs(y + dt * k3, t + dt)

                y_new = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

                # Enforce positivity
                y_new = jnp.maximum(y_new, MIN_CONC)
                return y_new, y

            y_final, y_history = lax.scan(rk4_step, y0, t_array[:-1])
            y_all = jnp.vstack([y_history, y_final[None, :]])
            solver_used = "rk4"

        # Extract profiles
        V_profile = y_all[:, 0]
        X_profile = y_all[:, 1] / jnp.maximum(V_profile, 1e-10)
        S_profile = y_all[:, 2] / jnp.maximum(V_profile, 1e-10)
        P_profile = y_all[:, 3] / jnp.maximum(V_profile, 1e-10)

        # Calculate mu profile using vectorized computation
        # Use vmap for efficient batch evaluation
        def calc_mu_single(S, X, P):
            return call_kinetics(kinetic_fn, kinetic_arity, S, X, P, kinetic_params)

        mu_profile = jax.vmap(calc_mu_single)(S_profile, X_profile, P_profile)

        # Final state as stream (total mass)
        # Use y_final directly for proper gradient flow (not interpolated trajectory)
        V_final = y_final[0]
        X_final = y_final[1] / jnp.maximum(V_final, MIN_CONC)
        S_final = y_final[2] / jnp.maximum(V_final, MIN_CONC)
        P_final = y_final[3] / jnp.maximum(V_final, MIN_CONC)

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
            "X_final": X_final,
            "S_final": S_final,
            "P_final": P_final,
            "solver": solver_used,
        }

        # Dissolved-O2 diagnostics (#101)
        if oxygen_active:
            C_O2_profile = y_all[:, 4] / jnp.maximum(V_profile, 1e-10)
            info["C_O2"] = C_O2_profile
            info["C_O2_final"] = y_final[4] / jnp.maximum(V_final, MIN_CONC)
            # O2 limitation factor along the trajectory
            info["o2_limitation"] = C_O2_profile / (K_O2 + C_O2_profile)
            info["OTR"] = kLa * (C_O2_star - C_O2_profile)

        return outlet, info


# Note: jax_vmap_safe removed - now using jax.vmap with call_kinetics directly


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
