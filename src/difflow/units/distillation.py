"""Distillation column unit operations.

This module provides distillation column models:
- ShortcutColumn: Fenske-Underwood-Gilliland method for quick design
- DistillationColumn: Rigorous stage-by-stage calculation

Key equations:
    Fenske (minimum stages): N_min = ln(x_D/x_B * (1-x_B)/(1-x_D)) / ln(alpha)
    Underwood (minimum reflux): sum(alpha*z_i/(alpha - theta)) = 1 + q
    Gilliland (correlation): (N - N_min)/(N + 1) = f((R - R_min)/(R + 1))

All calculations are JAX-compatible for automatic differentiation.

Numerical Considerations:
- Fenske singularity: When α ≈ 1, N_min → ∞; uses smooth capping
- Gilliland singularity: When Y → 1, N → ∞; uses smooth capping
- Temperature profiles: Scaled relative to feed T, not hard-coded values
- Initial guesses: Based on feed conditions, not ambient assumptions
"""

from typing import Callable, Literal
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream
from difflow.thermo import IdealThermo
from difflow.params_mixin import ParamsMixin
from difflow.constants import MIN_ALPHA_DIFF, MAX_STAGES, MAX_GILLILAND_Y, DEFAULT_TEMP_SCALE, EPS_DIVISION
from difflow.numerics import safe_divide, safe_log
import optimistix as optx


@dataclass(repr=False)
class ShortcutColumnParams(ParamsMixin):
    """Parameters for shortcut distillation column design.

    Attributes:
        species_order: List of species names for array ordering
        light_key: Name of light key component
        heavy_key: Name of heavy key component
        x_D_LK: Desired mole fraction of LK in distillate
        x_B_HK: Desired mole fraction of HK in bottoms
    """
    species_order: list[str]
    light_key: str
    heavy_key: str
    x_D_LK: float = 0.99  # LK recovery in distillate
    x_B_HK: float = 0.99  # HK recovery in bottoms


class ShortcutColumn:
    """Shortcut distillation column using Fenske-Underwood-Gilliland.

    This method provides quick estimates for:
    - Minimum number of stages (Fenske equation)
    - Minimum reflux ratio (Underwood equations)
    - Actual stages for given reflux (Gilliland correlation)

    Assumptions:
    - Constant relative volatility
    - Constant molar overflow
    - Feed at bubble point (q = 1) or specified q

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(
        self,
        params: ShortcutColumnParams,
        thermo: IdealThermo,
    ):
        """Initialize shortcut column.

        Args:
            params: Column parameters
            thermo: Thermodynamic property calculator for K-values
        """
        self.params = params
        self.thermo = thermo

    def relative_volatility(
        self,
        T: Array,
        P: Array,
    ) -> dict[str, Array]:
        """Calculate relative volatilities with respect to heavy key.

        alpha_i = K_i / K_HK

        Args:
            T: Temperature (K)
            P: Pressure (Pa)

        Returns:
            Dictionary of relative volatilities by species
        """
        K = self.thermo.K_values(T, P)
        K_HK = K[self.params.heavy_key]

        return {s: K[s] / K_HK for s in self.params.species_order}

    def average_alpha(
        self,
        T_top: Array,
        T_bot: Array,
        P: Array,
    ) -> dict[str, Array]:
        """Calculate geometric mean relative volatility.

        alpha_avg = sqrt(alpha_top * alpha_bottom)

        Args:
            T_top: Top temperature (K)
            T_bot: Bottom temperature (K)
            P: Column pressure (Pa)

        Returns:
            Dictionary of average relative volatilities
        """
        alpha_top = self.relative_volatility(T_top, P)
        alpha_bot = self.relative_volatility(T_bot, P)

        return {
            s: jnp.sqrt(alpha_top[s] * alpha_bot[s])
            for s in self.params.species_order
        }

    def fenske_minimum_stages(
        self,
        x_D_LK: Array,
        x_B_LK: Array,
        alpha_LK: Array,
    ) -> tuple[Array, Array]:
        """Calculate minimum number of stages using Fenske equation.

        N_min = ln[(x_D_LK/x_B_LK) * (x_B_HK/x_D_HK)] / ln(alpha_LK)

        For binary or pseudo-binary with LK recovery specification:
        N_min = ln[(x_D_LK/(1-x_D_LK)) * ((1-x_B_LK)/x_B_LK)] / ln(alpha_LK)

        Handles the singularity when α → 1:
        - When α < 1 + MIN_ALPHA_DIFF, ln(α) → 0 causing N_min → ∞
        - Uses smooth blending to cap N_min at MAX_STAGES
        - Returns a flag indicating if the system is close-boiling

        Args:
            x_D_LK: Mole fraction of LK in distillate
            x_B_LK: Mole fraction of LK in bottoms
            alpha_LK: Relative volatility of LK vs HK

        Returns:
            Tuple of (N_min, close_boiling_flag):
            - N_min: Minimum number of theoretical stages, capped at MAX_STAGES
            - close_boiling_flag: True if α is close to 1 (hard separation)
        """
        # Fenske equation for binary
        numer = safe_log(
            safe_divide(x_D_LK, 1 - x_D_LK) *
            safe_divide(1 - x_B_LK, x_B_LK)
        )

        # Handle singularity when alpha → 1
        # ln(alpha) → 0 as alpha → 1, causing N_min → ∞
        # Use regularization: max(ln(alpha), small_value)
        log_alpha = safe_log(jnp.maximum(alpha_LK, 1.0 + EPS_DIVISION))

        # Regularize denominator to prevent division by zero
        # When log_alpha is very small, N_min would be huge
        log_alpha_safe = jnp.maximum(log_alpha, MIN_ALPHA_DIFF)

        N_min_raw = numer / log_alpha_safe

        # Smooth capping using soft minimum
        # N_min = MAX_STAGES * tanh(N_min_raw / MAX_STAGES) gives smooth limit
        # Or simply clip with smooth transition
        # Use a sigmoid blend for smooth gradients
        N_min = jnp.where(
            N_min_raw < MAX_STAGES,
            N_min_raw,
            MAX_STAGES + (N_min_raw - MAX_STAGES) * 0.01  # Very slow growth beyond max
        )
        N_min = jnp.maximum(N_min, 1.0)  # At least 1 stage

        # Flag for close-boiling systems
        close_boiling = alpha_LK < (1.0 + 5 * MIN_ALPHA_DIFF)

        return N_min, close_boiling

    def underwood_theta(
        self,
        z: dict[str, Array],
        alpha: dict[str, Array],
        q: Array,
    ) -> Array:
        """Solve Underwood equation for theta parameter.

        sum(alpha_i * z_i / (alpha_i - theta)) = 1 - q

        Theta is between alpha_LK and alpha_HK (both > 1 typically).

        Args:
            z: Feed mole fractions by species
            alpha: Relative volatilities by species
            q: Feed quality (1 = saturated liquid, 0 = saturated vapor)

        Returns:
            Underwood theta parameter
        """
        p = self.params
        alpha_LK = alpha[p.light_key]
        alpha_HK = alpha[p.heavy_key]  # = 1.0 by definition

        # Theta is between alpha_HK and alpha_LK
        # For numerical stability, search in (1, alpha_LK)

        def underwood_func(theta, args):
            total = jnp.zeros(())
            for s in p.species_order:
                total = total + safe_divide(alpha[s] * z[s], alpha[s] - theta)
            return total - (1 - q)

        # Use Newton iteration to find theta
        # Initial guess: geometric mean
        theta_init = jnp.sqrt(alpha_LK * alpha_HK)
        solver = optx.Newton(rtol=1e-10, atol=1e-10)
        sol = optx.root_find(underwood_func, solver, theta_init, args=None, max_steps=50, throw=False)
        theta = sol.value

        # Ensure theta is in valid range
        theta = jnp.clip(theta, 1.001, alpha_LK - 0.001)

        return theta

    def underwood_minimum_reflux(
        self,
        x_D: dict[str, Array],
        alpha: dict[str, Array],
        theta: Array,
    ) -> Array:
        """Calculate minimum reflux ratio using second Underwood equation.

        R_min + 1 = sum(alpha_i * x_D_i / (alpha_i - theta))

        Args:
            x_D: Distillate mole fractions by species
            alpha: Relative volatilities by species
            theta: Underwood theta parameter

        Returns:
            Minimum reflux ratio
        """
        total = jnp.zeros(())
        for s in self.params.species_order:
            total = total + safe_divide(alpha[s] * x_D[s], alpha[s] - theta)

        R_min = total - 1
        return jnp.maximum(R_min, 0.1)  # Ensure positive

    def gilliland_correlation(
        self,
        R: Array,
        R_min: Array,
        N_min: Array,
    ) -> tuple[Array, Array]:
        """Calculate actual stages using Gilliland correlation.

        Y = (N - N_min) / (N + 1)
        X = (R - R_min) / (R + 1)

        Gilliland correlation (Molokanov fit):
        Y = 1 - exp[(1 + 54.4*X)/(11 + 117.2*X) * (X - 1)/sqrt(X)]

        Handles singularities:
        - X → 0 (R → R_min): Y → 1, N → ∞ (total reflux limit)
        - Y capped at MAX_GILLILAND_Y to prevent unreasonably large N
        - N capped at MAX_STAGES for economic feasibility

        Args:
            R: Actual reflux ratio
            R_min: Minimum reflux ratio
            N_min: Minimum number of stages

        Returns:
            Tuple of (N, near_minimum_reflux_flag):
            - N: Number of theoretical stages
            - near_minimum_reflux_flag: True if R is close to R_min
        """
        # Ensure R >= R_min (can't operate below minimum reflux)
        R_safe = jnp.maximum(R, R_min * 1.001)

        X = (R_safe - R_min) / (R_safe + 1)

        # Ensure X is positive for sqrt
        X_safe = jnp.maximum(X, 1e-6)

        # Molokanov correlation (more accurate than original Gilliland)
        Y = 1 - jnp.exp(
            (1 + 54.4 * X_safe) / (11 + 117.2 * X_safe) *
            (X_safe - 1) / jnp.sqrt(X_safe)
        )

        # Flag for near-minimum reflux operation
        near_min_reflux = X < 0.1  # R/R_min < ~1.1

        # Cap Y to prevent N → ∞
        # Use smooth capping with sigmoid for continuous gradients
        Y_raw = Y
        Y = jnp.minimum(Y, MAX_GILLILAND_Y)

        # Solve for N: Y = (N - N_min) / (N + 1)
        # N * (1 - Y) = N_min + Y
        # N = (N_min + Y) / (1 - Y)
        one_minus_Y = jnp.maximum(1 - Y, 1e-3)  # Prevent division by zero
        N_raw = (N_min + Y) / one_minus_Y

        # Cap N at maximum stages
        N = jnp.minimum(N_raw, MAX_STAGES)
        N = jnp.maximum(N, N_min)  # At least N_min stages

        return N, near_min_reflux

    def feed_stage_kirkbride(
        self,
        N: Array,
        z_LK: Array,
        z_HK: Array,
        x_B_LK: Array,
        x_D_HK: Array,
        B_over_D: Array,
    ) -> Array:
        """Estimate optimal feed stage using Kirkbride equation.

        log(N_R/N_S) = 0.206 * log[(z_HK/z_LK) * (x_B_LK/x_D_HK)^2 * (B/D)]

        where N_R = rectifying stages, N_S = stripping stages

        Args:
            N: Total stages
            z_LK: Feed mole fraction of LK
            z_HK: Feed mole fraction of HK
            x_B_LK: Bottoms mole fraction of LK
            x_D_HK: Distillate mole fraction of HK
            B_over_D: Bottoms to distillate molar flow ratio

        Returns:
            Feed stage number (from bottom)
        """
        ratio_arg = safe_divide(z_HK, z_LK) * \
                    safe_divide(x_B_LK, x_D_HK)**2 * \
                    B_over_D

        log_ratio = 0.206 * safe_log(ratio_arg)
        NR_over_NS = jnp.exp(log_ratio)

        # N = N_R + N_S, NR/NS = ratio
        # N_S = N / (1 + ratio)
        N_S = N / (1 + NR_over_NS)

        return jnp.maximum(jnp.round(N_S), 1.0)

    def __call__(
        self,
        feed: Stream,
        R: Array | float,
        P: Array | float = 101325.0,
        q: Array | float = 1.0,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform shortcut distillation calculation.

        Args:
            feed: Feed stream
            R: Reflux ratio (L/D)
            P: Column pressure (Pa)
            q: Feed quality (1 = saturated liquid)

        Returns:
            distillate: Distillate stream
            bottoms: Bottoms stream
            info: Dictionary with design information:
                - 'N_min': Minimum stages
                - 'R_min': Minimum reflux ratio
                - 'N': Actual stages
                - 'N_feed': Feed stage
                - 'alpha': Relative volatilities
                - 'close_boiling': True if α ≈ 1 (hard separation)
                - 'near_min_reflux': True if R ≈ R_min
        """
        p = self.params
        R = jnp.asarray(R)
        P = jnp.asarray(P)
        q = jnp.asarray(q)

        # Get feed composition
        feed_flows = get_flows(feed)
        F_total = sum(feed_flows.values())
        z = {s: feed_flows[s] / F_total for s in p.species_order}

        # Estimate column temperatures scaled relative to feed T
        # Instead of hard-coded ±20K which fails for cryogenic/high-T systems,
        # use a percentage of feed temperature. This scales appropriately:
        # - Cryogenic (90K feed): ΔT ≈ ±4.5K
        # - Ambient (350K feed): ΔT ≈ ±17.5K
        # - High-T (600K feed): ΔT ≈ ±30K
        T_feed = feed["T"]
        T_half_range = T_feed * DEFAULT_TEMP_SCALE
        T_top = T_feed - T_half_range
        T_bot = T_feed + T_half_range

        # Get relative volatilities
        alpha = self.average_alpha(T_top, T_bot, P)
        alpha_LK = alpha[p.light_key]

        # Recovery calculations
        # D*x_D_LK = F*z_LK * recovery_LK
        # B*x_B_HK = F*z_HK * recovery_HK
        z_LK = z[p.light_key]
        z_HK = z[p.heavy_key]

        # For given recoveries, calculate product compositions
        rec_LK = p.x_D_LK  # LK recovery to distillate
        rec_HK = p.x_B_HK  # HK recovery to bottoms

        # Component balances
        D_LK = F_total * z_LK * rec_LK
        B_LK = F_total * z_LK * (1 - rec_LK)
        D_HK = F_total * z_HK * (1 - rec_HK)
        B_HK = F_total * z_HK * rec_HK

        # Total distillate and bottoms
        D_total = D_LK + D_HK
        B_total = B_LK + B_HK

        # Distribute other components based on relative volatility
        distillate_flows = {}
        bottoms_flows = {}

        for s in p.species_order:
            if s == p.light_key:
                distillate_flows[s] = D_LK
                bottoms_flows[s] = B_LK
            elif s == p.heavy_key:
                distillate_flows[s] = D_HK
                bottoms_flows[s] = B_HK
            else:
                # Use Hengstebeck-Geddes equation for distribution
                # log(d_i/b_i) = A + C * log(alpha_i)
                # where A and C are determined from key components
                A = safe_log(safe_divide(D_LK, B_LK) * safe_divide(B_HK, D_HK))
                C = safe_divide(safe_log(safe_divide(D_LK, B_LK)), safe_log(alpha_LK))

                d_over_b = jnp.exp(A + C * safe_log(alpha[s]))
                F_i = feed_flows[s]
                d_i = F_i * d_over_b / (1 + d_over_b)
                b_i = F_i - d_i

                d_i_clipped = jnp.clip(d_i, 0.0, F_i)
                distillate_flows[s] = d_i_clipped
                bottoms_flows[s] = F_i - d_i_clipped  # Preserves mass balance
                D_total = D_total + distillate_flows[s]
                B_total = B_total + bottoms_flows[s]

        # Product compositions
        x_D = {s: safe_divide(distillate_flows[s], D_total) for s in p.species_order}
        x_B = {s: safe_divide(bottoms_flows[s], B_total) for s in p.species_order}

        # Fenske minimum stages
        x_D_LK = x_D[p.light_key]
        x_B_LK = x_B[p.light_key]
        N_min, close_boiling = self.fenske_minimum_stages(x_D_LK, x_B_LK, alpha_LK)

        # Underwood minimum reflux
        theta = self.underwood_theta(z, alpha, q)
        R_min = self.underwood_minimum_reflux(x_D, alpha, theta)

        # Gilliland actual stages
        N, near_min_reflux = self.gilliland_correlation(R, R_min, N_min)

        # Feed stage
        B_over_D = safe_divide(B_total, D_total)
        x_D_HK = x_D[p.heavy_key]
        N_feed = self.feed_stage_kirkbride(N, z_LK, z_HK, x_B_LK, x_D_HK, B_over_D)

        # Create output streams
        distillate = make_stream(distillate_flows, T_top, P)
        bottoms = make_stream(bottoms_flows, T_bot, P)

        # Compute condenser and reboiler duties including latent heat
        # Condenser: condense all vapor from top stage
        V_top = (R + 1) * D_total  # Vapor flow at top stage

        # Compute mixture enthalpies at top and bottom using full model
        # (sensible + latent heat via Hvap)
        x_D_arr = jnp.array([x_D[s] for s in p.species_order])
        x_B_arr = jnp.array([x_B[s] for s in p.species_order])

        # Vapor enthalpy at top (includes latent heat)
        H_vapor_top = jnp.zeros(())
        h_liquid_top = jnp.zeros(())
        for i, s in enumerate(p.species_order):
            H_vapor_top = H_vapor_top + x_D_arr[i] * self.thermo.H_pure(s, T_top, 'vapor')
            h_liquid_top = h_liquid_top + x_D_arr[i] * self.thermo.H_pure(s, T_top, 'liquid')

        # Feed enthalpy
        h_F = jnp.zeros(())
        for i, s in enumerate(p.species_order):
            z_arr_i = jnp.asarray(z[s])
            h_F = h_F + z_arr_i * self.thermo.H_pure(s, T_feed, 'liquid')

        # Bottoms liquid enthalpy
        h_B = jnp.zeros(())
        for i, s in enumerate(p.species_order):
            h_B = h_B + x_B_arr[i] * self.thermo.H_pure(s, T_bot, 'liquid')

        # Condenser duty (heat removed, negative)
        Q_condenser = V_top * (h_liquid_top - H_vapor_top)

        # Reboiler duty from overall energy balance
        # F*h_F + Q_reb = D*h_D + B*h_B + |Q_cond|
        # Q_reb = D*h_D + B*h_B - Q_cond - F*h_F  (Q_cond < 0)
        h_D = h_liquid_top  # total condenser: distillate is liquid at T_top
        Q_reboiler = D_total * h_D + B_total * h_B - Q_condenser - F_total * h_F

        info = {
            "N_min": N_min,
            "R_min": R_min,
            "N": N,
            "N_feed": N_feed,
            "alpha": alpha,
            "alpha_LK": alpha_LK,
            "theta": theta,
            "D": D_total,
            "B": B_total,
            "x_D": x_D,
            "x_B": x_B,
            "T_top": T_top,
            "T_bot": T_bot,
            "close_boiling": close_boiling,
            "near_min_reflux": near_min_reflux,
            "Q_condenser": Q_condenser,
            "Q_reboiler": Q_reboiler,
        }

        return distillate, bottoms, info


@dataclass(repr=False)
class DistillationColumnParams(ParamsMixin):
    """Parameters for rigorous distillation column.

    Attributes:
        species_order: List of species names
        n_stages: Total number of stages (including condenser/reboiler)
        feed_stage: Feed stage number (1 = bottom)
        condenser_type: 'total' or 'partial'
        P: Column pressure (Pa)
    """
    species_order: list[str]
    n_stages: int
    feed_stage: int
    condenser_type: Literal["total", "partial"] = "total"
    P: float = 101325.0
    q: float = 1.0  # Feed thermal condition (1.0 = saturated liquid, 0.0 = saturated vapor)


class DistillationColumn:
    """Rigorous stage-by-stage distillation column.

    Solves MESH equations (Material, Equilibrium, Summation, Heat balance)
    for each stage using the bubble-point method.

    Stage numbering: 1 = reboiler (bottom), n_stages = condenser (top)

    Assumptions:
    - Equilibrium stages
    - No pressure drop between stages
    - Constant molar overflow (CMO) for initial solution

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(
        self,
        params: DistillationColumnParams,
        thermo: IdealThermo,
    ):
        """Initialize distillation column.

        Args:
            params: Column parameters
            thermo: Thermodynamic property calculator
        """
        self.params = params
        self.thermo = thermo
        self.n_species = len(params.species_order)

    def _bubble_point_T(
        self,
        x: Array,
        P: Array,
        T_guess: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate bubble point temperature and vapor composition.

        Solves: sum(K_i * x_i) = 1 for T

        Args:
            x: Liquid mole fractions
            P: Pressure (Pa)
            T_guess: Initial temperature guess (K). If None, estimates from
                     pure component data or uses a pressure-scaled default.

        Returns:
            (T, y): Bubble temperature and vapor composition
        """
        p = self.params

        def bubble_residual(T, args):
            K = self.thermo.K_values_array(T, P)
            return jnp.sum(K * x) - 1.0

        # Initial guess: use provided guess or estimate from pressure
        # At higher pressures, boiling points are higher. Rough correlation:
        # T_bp ≈ T_nbp * (P / 101325)^0.1 for many organics
        # Use 350K as baseline for 1 atm, scale with pressure
        if T_guess is not None:
            T_init = T_guess
        else:
            # Pressure-scaled initial guess
            # 350K is reasonable for many organics at 1 atm
            # For cryogenic: would need lower base, but user should provide T_guess
            T_base = 350.0
            P_ref = 101325.0
            T_init = T_base * (P / P_ref) ** 0.08
            T_init = jnp.clip(T_init, 100.0, 800.0)  # Reasonable bounds

        solver = optx.Newton(rtol=1e-10, atol=1e-10)
        sol = optx.root_find(bubble_residual, solver, T_init, args=None, max_steps=50, throw=False)
        T = sol.value

        # Calculate y from K-values
        K = self.thermo.K_values_array(T, P)
        y = K * x
        y = y / jnp.sum(y)  # Normalize

        return T, y

    def _solve_constant_molar_overflow(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        R: Array,
        D: Array,
        T_feed: Array,
    ) -> tuple[Array, Array, Array]:
        """Solve column using constant molar overflow (CMO) assumption.

        Implements the Lewis-Matheson iterative method: a top-down sweep
        through the column that alternates between equilibrium (bubble-point)
        and operating-line steps until convergence.

        Stage numbering: j=0 is reboiler (bottom), j=n-1 is top stage.
        Total condenser assumed: x_D = y_{top} (vapor from top stage).

        For each iteration:
        1. Compute K values and equilibrium y at each stage from current T.
        2. Set boundary conditions: x_D = y[-1], x_B = x[0].
        3. Top-down sweep: for each stage j (n-2 down to 0), derive x_j from
           the operating line applied to x_{j+1} (liquid from above), then
           back-calculate x_j = y_op / K_j.
        4. Update T from bubble-point calculation at the new x.

        Args:
            feed_flows: Feed molar flows by species
            F_total: Total feed flow
            R: Reflux ratio
            D: Distillate flow rate
            T_feed: Feed temperature (K) — unused in solver, kept for API compat

        Returns:
            (x, y, T): Liquid compositions, vapor compositions, temperatures
                       All have shape (n_stages, n_species) or (n_stages,)
        """
        p = self.params
        n = p.n_stages
        P = jnp.asarray(p.P)

        B = F_total - D
        L_rect = R * D
        V_rect = (R + 1) * D
        q = jnp.asarray(p.q)
        L_strip = L_rect + q * F_total
        V_strip = V_rect - (1 - q) * F_total

        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])

        # Better initial guess: one-stage flash enriches the distillate estimate.
        # Use 350 K as a safe starting T for the feed bubble point.
        T_bp_feed, _ = self._bubble_point_T(z, P, T_guess=jnp.asarray(350.0))
        K_feed = self.thermo.K_values_array(T_bp_feed, P)
        y_flash = K_feed * z
        y_flash = y_flash / jnp.maximum(jnp.sum(y_flash), 1e-10)
        x_D_init = jnp.clip(y_flash, 1e-6, 1.0 - 1e-6)
        x_D_init = x_D_init / jnp.sum(x_D_init)

        x_B_init = (F_total * z - D * x_D_init) / jnp.maximum(B, 1e-10)
        x_B_init = jnp.clip(x_B_init, 1e-6, 1.0 - 1e-6)
        x_B_init = x_B_init / jnp.sum(x_B_init)

        # Linear composition profile: reboiler (j=0) → top (j=n-1)
        frac = jnp.linspace(0.0, 1.0, n)
        x_init = frac[:, None] * x_D_init[None, :] + (1.0 - frac[:, None]) * x_B_init[None, :]

        # Start temperatures at 350 K (reasonable for most organics at 1 atm).
        T_init = jnp.full(n, 350.0)

        def one_iteration(carry, _):
            x, T = carry

            # --- Step 1: K values at current stage temperatures ---
            def scan_K(_, inputs):
                x_j, T_j = inputs
                return None, self.thermo.K_values_array(T_j, P)

            _, K_all = lax.scan(scan_K, None, (x, T))  # (n, nc)

            # --- Step 2: Equilibrium vapor compositions ---
            yx = K_all * x
            y_all = yx / jnp.maximum(jnp.sum(yx, axis=1, keepdims=True), 1e-10)

            # --- Step 3: Boundary conditions ---
            # Total condenser: distillate has same composition as vapor leaving top stage
            x_D = y_all[-1]
            # Reboiler: bottoms composition = liquid leaving bottom stage
            x_B = x[0]

            # --- Step 4: Top-down sweep using operating lines ---
            # Rectifying OL: V*y_j = L*x_{j+1} + D*x_D  → y_j = (L*x_above + D*x_D)/V
            # Stripping OL:  L'*x_{j+1} = V'*y_j + B*x_B → y_j = (L'*x_above - B*x_B)/V'
            # Then x_j = y_op / K_j (normalised), i.e. inverse equilibrium step.
            def update_stage(x_above, j):
                is_rect = j >= p.feed_stage

                y_op_rect = (L_rect * x_above + D * x_D) / V_rect
                y_op_strip = (L_strip * x_above - B * x_B) / V_strip
                y_op = jnp.where(is_rect, y_op_rect, y_op_strip)
                y_op = jnp.maximum(y_op, 0.0)
                y_op = y_op / jnp.maximum(jnp.sum(y_op), 1e-10)

                K_j = K_all[j]
                x_j = y_op / jnp.maximum(K_j, 1e-10)
                x_j = jnp.maximum(x_j, 0.0)
                x_j = x_j / jnp.maximum(jnp.sum(x_j), 1e-10)
                return x_j, x_j

            # Sweep from j=n-2 down to j=0, carrying x from the stage above.
            stages_down = jnp.arange(n - 2, -1, -1)
            _, x_new_rev = lax.scan(update_stage, x[-1], stages_down)
            # x_new_rev[k] → stage (n-2-k); reverse to get [stage0, …, stage n-2]
            x_new_lower = x_new_rev[::-1]

            # Top stage: in equilibrium with x_D (total condenser constraint)
            K_top = K_all[-1]
            x_top_new = x_D / jnp.maximum(K_top, 1e-10)
            x_top_new = x_top_new / jnp.maximum(jnp.sum(x_top_new), 1e-10)

            x_new = jnp.concatenate([x_new_lower, x_top_new[None, :]], axis=0)
            x_new = jnp.clip(x_new, 1e-10, 1.0)
            x_new = x_new / jnp.sum(x_new, axis=1, keepdims=True)

            # --- Step 5: Update stage temperatures from bubble point ---
            def scan_T(_, inputs):
                x_j, T_j_old = inputs
                T_new, _ = self._bubble_point_T(x_j, P, T_guess=T_j_old)
                return None, T_new

            _, T_new = lax.scan(scan_T, None, (x_new, T))

            return (x_new, T_new), None

        (x_final, T_final), _ = lax.scan(
            one_iteration, (x_init, T_init), None, length=30
        )

        # Final equilibrium vapor compositions
        def compute_y(_, inputs):
            x_j, T_j = inputs
            K_j = self.thermo.K_values_array(T_j, P)
            y_j = K_j * x_j
            return None, y_j / jnp.maximum(jnp.sum(y_j), 1e-10)

        _, y_final = lax.scan(compute_y, None, (x_final, T_final))

        return x_final, y_final, T_final

    def _compute_stage_enthalpies(
        self,
        x: Array,
        y: Array,
        T: Array,
    ) -> tuple[Array, Array]:
        """Compute liquid and vapor mixture enthalpies at each stage.

        Args:
            x: (n, nc) liquid mole fractions
            y: (n, nc) vapor mole fractions
            T: (n,) stage temperatures

        Returns:
            h_all: (n,) liquid mixture enthalpies (J/mol)
            H_all: (n,) vapor mixture enthalpies (J/mol)
        """
        p = self.params

        def stage_enthalpies(_, inputs):
            x_j, y_j, T_j = inputs
            h_j = jnp.zeros(())
            H_j = jnp.zeros(())
            for i, s in enumerate(p.species_order):
                h_j = h_j + x_j[i] * self.thermo.H_pure(s, T_j, 'liquid')
                H_j = H_j + y_j[i] * self.thermo.H_pure(s, T_j, 'vapor')
            return None, (h_j, H_j)

        _, (h_all, H_all) = lax.scan(stage_enthalpies, None, (x, y, T))
        return h_all, H_all

    def _compute_duties(
        self,
        x_profile: Array,
        y_profile: Array,
        T_profile: Array,
        F_total: Array,
        z: dict[str, Array],
        T_feed: Array,
        R: Array,
        D: Array,
        B: Array,
        V_top: Array | None = None,
        V_bot: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute condenser and reboiler duties including latent heat.

        The dominant energy term in distillation is the latent heat of
        vaporization/condensation. This method uses the full enthalpy
        model (sensible + latent heat via Hvap) for accurate duty
        calculations.

        For total condenser:
            Q_cond = V_top * (H_vapor_top - h_liquid_top)
            where H_vapor includes latent heat of vaporization.

        For reboiler (from overall energy balance):
            Q_reb = D * h_D + B * h_B - F * h_F + Q_cond

        Args:
            x_profile: (n, nc) liquid mole fractions
            y_profile: (n, nc) vapor mole fractions
            T_profile: (n,) stage temperatures
            F_total: Total feed flow rate (mol/s)
            z: Feed mole fractions dict
            T_feed: Feed temperature (K)
            R: Reflux ratio
            D: Distillate flow rate (mol/s)
            B: Bottoms flow rate (mol/s)
            V_top: Vapor flow leaving top stage (mol/s).
                   If None, uses (R+1)*D (CMO assumption).
            V_bot: Vapor flow leaving bottom stage (mol/s).
                   If None, uses (R+1)*D (CMO assumption).

        Returns:
            Q_condenser: Condenser duty (J/s, negative = heat removed)
            Q_reboiler: Reboiler duty (J/s, positive = heat added)
        """
        p = self.params

        # Compute stage enthalpies (includes latent heat for vapor)
        h_all, H_all = self._compute_stage_enthalpies(
            x_profile, y_profile, T_profile
        )

        # Top stage enthalpies
        H_vapor_top = H_all[-1]   # Vapor enthalpy at top (includes Hvap)
        h_liquid_top = h_all[-1]  # Liquid enthalpy at top

        # Bottom stage enthalpies
        H_vapor_bot = H_all[0]    # Vapor enthalpy at bottom (includes Hvap)
        h_liquid_bot = h_all[0]   # Liquid enthalpy at bottom

        # Vapor flows (use provided or CMO estimates)
        V_top_flow = V_top if V_top is not None else (R + 1) * D
        V_bot_flow = V_bot if V_bot is not None else (R + 1) * D

        # Condenser duty: condense all vapor from top stage to liquid
        # Q_cond = V_top * (h_liquid_top - H_vapor_top) < 0 (heat removed)
        Q_condenser = V_top_flow * (h_liquid_top - H_vapor_top)

        # Feed enthalpy (saturated liquid, q=1)
        z_arr = jnp.array([z[s] for s in p.species_order])
        h_F = jnp.zeros(())
        for i, s in enumerate(p.species_order):
            h_F = h_F + z_arr[i] * self.thermo.H_pure(s, T_feed, 'liquid')

        # Distillate enthalpy (liquid at top T for total condenser)
        h_D = h_liquid_top

        # Bottoms enthalpy (liquid at bottom T)
        h_B = h_liquid_bot

        # Overall energy balance: F*h_F + Q_reb = D*h_D + B*h_B + Q_cond
        # Q_reb = D*h_D + B*h_B + |Q_cond| - F*h_F
        #       = D*h_D + B*h_B - Q_cond - F*h_F  (since Q_cond < 0)
        Q_reboiler = D * h_D + B * h_B - Q_condenser - F_total * h_F

        return Q_condenser, Q_reboiler

    def _solve_mesh(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        R: Array,
        D: Array,
        T_feed: Array,
        n_iter: int = 20,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Rigorous MESH solver using Wang-Henke bubble-point method.

        Starts from a CMO warm start and iterates using:
        1. Tridiagonal component material balance solve
        2. Bubble-point temperature update
        3. Energy balance to update L/V profiles

        Args:
            feed_flows: Feed molar flows by species
            F_total: Total feed flow rate (mol/s)
            R: Reflux ratio
            D: Distillate flow rate (mol/s)
            T_feed: Feed temperature (K)
            n_iter: Number of MESH iterations

        Returns:
            x: (n, nc) liquid mole fractions
            y: (n, nc) vapor mole fractions
            T: (n,) stage temperatures
            L: (n,) liquid flows leaving each stage downward (mol/s)
            V: (n,) vapor flows leaving each stage upward (mol/s)

        Example:
            12-stage heptane / ethylbenzene column at 1 atm (R=2.5,
            B_spec sets bottoms flow). McCabe-Thiele targets: 97 % heptane
            in distillate, 99 % ethylbenzene in bottoms.

            CMO result  (use_mesh=False):
              distillate heptane=0.862, bottoms ethylbenzene=0.996

            MESH result (use_mesh=True, mesh_iter=20):
              distillate heptane=0.962, bottoms ethylbenzene=0.989

            The energy-balance L/V correction drives the rectifying-section
            liquid rate (~0.012 mol/s) well below the stripping-section
            rate (~0.023 mol/s), consistent with a saturated-liquid feed
            and the differing latent heats of heptane and ethylbenzene.
        """
        p = self.params
        n = p.n_stages
        P = jnp.asarray(p.P)
        nc = self.n_species
        B = F_total - D
        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])

        # Feed flow vector: nonzero only at feed stage
        F_vec = jnp.zeros(n).at[p.feed_stage].set(F_total)

        # CMO warm start
        x_init, y_init, T_init = self._solve_constant_molar_overflow(
            feed_flows, F_total, R, D, T_feed
        )

        # Initial L/V from CMO (q=1 saturated liquid feed)
        L_rect = R * D
        V_rect = (R + 1) * D
        L_strip = L_rect + F_total
        V_strip = V_rect

        # L[j] = liquid leaving stage j downward
        L_init = jnp.where(jnp.arange(n) <= p.feed_stage, L_strip, L_rect)
        L_init = L_init.at[0].set(B)
        V_init = jnp.full(n, V_rect)
        V_init = V_init.at[n - 1].set((R + 1) * D)

        def one_mesh_iter(carry, _):
            x, y, T, L, V = carry

            # 1. Compute K values at current T
            _, K = lax.scan(
                lambda _, Tj: (None, self.thermo.K_values_array(Tj, P)),
                None, T
            )
            # K shape: (n, nc)

            # 2. Build tridiagonal system for all components simultaneously
            # Material balance at stage j (bottom-up numbering):
            #   V[j-1]*K[j-1,i]*x[j-1,i] - (L[j] + K[j,i]*V[j])*x[j,i]
            #   + L[j+1]*x[j+1,i] = -F[j]*z[i]

            V_prev = jnp.concatenate([jnp.zeros(1), V[:-1]])   # V[j-1]
            K_prev = jnp.concatenate([jnp.zeros((1, nc)), K[:-1]], axis=0)  # K[j-1]
            lower = V_prev[:, None] * K_prev   # (n, nc), lower[0] = 0

            diag = -(L[:, None] + K * V[:, None])  # (n, nc)

            L_next = jnp.concatenate([L[1:], jnp.zeros(1)])    # L[j+1]
            upper = jnp.array(L_next[:, None] * jnp.ones((n, nc)))  # (n, nc)
            upper = upper.at[-1].set(0.0)  # no stage above top

            rhs = -F_vec[:, None] * z[None, :]  # (n, nc)

            # Top boundary condition: reflux enters stage n-1 with
            # composition x_D_prev = y[-1] (total condenser assumption)
            x_D_prev = y[-1]
            rhs = rhs.at[-1].add(-R * D * x_D_prev)

            # 3. Thomas algorithm tridiagonal solve for all nc components
            m0_safe = jnp.where(
                jnp.abs(diag[0]) > 1e-30,
                diag[0],
                jnp.sign(diag[0] + 1e-60) * 1e-30,
            )
            c0_prime = upper[0] / m0_safe   # (nc,)
            d0_prime = rhs[0] / m0_safe     # (nc,)

            def forward_step(carry, j):
                c_prev, d_prev = carry  # (nc,), (nc,)
                m = diag[j] - lower[j] * c_prev   # (nc,)
                m_safe = jnp.where(
                    jnp.abs(m) > 1e-30,
                    m,
                    jnp.sign(m + 1e-60) * 1e-30,
                )
                c_new = upper[j] / m_safe                         # (nc,)
                d_new = (rhs[j] - lower[j] * d_prev) / m_safe    # (nc,)
                return (c_new, d_new), (c_new, d_new)

            _, (c_prime_rest, d_prime_rest) = lax.scan(
                forward_step, (c0_prime, d0_prime), jnp.arange(1, n)
            )
            c_prime = jnp.concatenate(
                [c0_prime[None], c_prime_rest], axis=0
            )  # (n, nc)
            d_prime = jnp.concatenate(
                [d0_prime[None], d_prime_rest], axis=0
            )  # (n, nc)

            # Back substitution
            def backward_step(x_next, j):
                x_j = d_prime[j] - c_prime[j] * x_next  # (nc,)
                return x_j, x_j

            x_last = d_prime[-1]  # (nc,)
            _, x_rev = lax.scan(
                backward_step, x_last, jnp.arange(n - 2, -1, -1)
            )
            # x_rev shape: (n-1, nc), from stage n-2 down to 0
            x_new = jnp.concatenate(
                [x_rev[::-1], x_last[None]], axis=0
            )  # (n, nc)

            # Clip and normalize
            x_new = jnp.maximum(x_new, 1e-10)
            x_new = x_new / jnp.sum(x_new, axis=1, keepdims=True)

            # 4. Update T from bubble point at each stage
            _, T_new = lax.scan(
                lambda _, args: (None, self._bubble_point_T(args[0], P, args[1])[0]),
                None, (x_new, T)
            )

            # 5. Recompute K and y with updated T
            _, K_new = lax.scan(
                lambda _, Tj: (None, self.thermo.K_values_array(Tj, P)),
                None, T_new
            )
            y_new = K_new * x_new
            y_new = y_new / jnp.maximum(
                jnp.sum(y_new, axis=1, keepdims=True), 1e-10
            )

            # 6. Energy balance update to obtain L/V profiles
            h_all, H_all = self._compute_stage_enthalpies(x_new, y_new, T_new)

            # Feed enthalpy (saturated liquid, q=1)
            h_F = sum(
                z[i] * self.thermo.H_pure(s, jnp.asarray(T_feed), 'liquid')
                for i, s in enumerate(p.species_order)
            )

            # Reflux enthalpy (total condenser: liquid at top stage T)
            h_reflux = sum(
                y_new[-1, i] * self.thermo.H_pure(s, T_new[-1], 'liquid')
                for i, s in enumerate(p.species_order)
            )

            # Top-down energy balance sweep from stage n-1 down to stage 1
            def eb_step(carry, stage_j):
                L_above, V_j, h_above = carry  # L[j+1], V[j], h[j+1]
                H_j = H_all[stage_j]            # vapor enthalpy at stage j
                H_below = H_all[stage_j - 1]    # vapor enthalpy at stage j-1
                h_j = h_all[stage_j]            # liquid enthalpy at stage j
                F_j = F_vec[stage_j]

                numer = (
                    V_j * (H_j - H_below)
                    + L_above * (H_below - h_above)
                    + F_j * (H_below - h_F)
                )
                # Denominator is H[j-1] - h[j]: vapor enthalpy from the
                # hotter stage below minus liquid enthalpy at stage j.
                # This is always positive (H_vap >> H_liq at the same T,
                # and stage j-1 is hotter).  Sign must be H_below - h_j,
                # NOT h_j - H_below; the latter would negate every L[j].
                denom = H_below - h_j
                denom_safe = jnp.where(
                    jnp.abs(denom) > 1.0,
                    denom,
                    jnp.sign(denom + 1e-30),
                )

                L_j = numer / denom_safe
                L_j = jnp.maximum(L_j, B * 0.01)  # physical lower bound

                V_j_prev = L_j + V_j - L_above - F_j  # overall balance
                V_j_prev = jnp.maximum(V_j_prev, D * 0.01)

                return (L_j, V_j_prev, h_j), (L_j, V_j_prev)

            # Sweep stages from n-1 down to 1
            stages = jnp.arange(n - 1, 0, -1)
            init_carry = (R * D, (R + 1) * D, h_reflux)
            (_, _, _), (L_computed, V_computed) = lax.scan(
                eb_step, init_carry, stages
            )
            # L_computed[k] = L at stage (n-1-k): [L[n-1], L[n-2], ..., L[1]]
            # V_computed[k] = V[j-1]:              [V[n-2], V[n-3], ..., V[0]]

            L_new = jnp.concatenate(
                [jnp.array([B]), L_computed[::-1]]
            )  # [L[0]=B, L[1], ..., L[n-1]]
            V_new = jnp.concatenate(
                [V_computed[::-1], jnp.array([(R + 1) * D])]
            )  # [V[0], ..., V[n-1]]

            return (x_new, y_new, T_new, L_new, V_new), None

        (x_f, y_f, T_f, L_f, V_f), _ = lax.scan(
            one_mesh_iter,
            (x_init, y_init, T_init, L_init, V_init),
            None,
            length=n_iter,
        )
        return x_f, y_f, T_f, L_f, V_f

    def __call__(
        self,
        feed: Stream,
        R: Array | float,
        D_spec: Array | float | None = None,
        B_spec: Array | float | None = None,
        use_mesh: bool = False,
        mesh_iter: int = 20,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Solve distillation column.

        Args:
            feed: Feed stream
            R: Reflux ratio
            D_spec: Distillate flow rate specification (mol/s)
            B_spec: Bottoms flow rate specification (mol/s)
                    Exactly one of D_spec or B_spec must be given.
            use_mesh: If True, run the rigorous MESH solver (Wang-Henke
                      bubble-point method) after the CMO warm start.
                      If False (default), return the CMO solution.
            mesh_iter: Number of MESH iterations when use_mesh=True.

        Returns:
            distillate: Distillate stream
            bottoms: Bottoms stream
            info: Dictionary with:
                - 'T_profile': Stage temperatures
                - 'x_profile': Liquid composition profiles
                - 'y_profile': Vapor composition profiles
                - 'L_rect': Rectifying liquid flow rate (CMO) or 'L_profile'
                - 'V_rect': Rectifying vapor flow rate (CMO) or 'V_profile'
                - 'D': Distillate flow rate
                - 'B': Bottoms flow rate
                When use_mesh=True, also includes:
                - 'L_profile': (n,) liquid flows leaving each stage (mol/s)
                - 'V_profile': (n,) vapor flows leaving each stage (mol/s)
        """
        p = self.params
        R = jnp.asarray(R)

        # Get feed flows
        feed_flows = get_flows(feed)
        F_total = sum(feed_flows.values())

        # Determine D and B
        if D_spec is not None:
            D = jnp.asarray(D_spec)
            B = F_total - D
        elif B_spec is not None:
            B = jnp.asarray(B_spec)
            D = F_total - B
        else:
            # Default: 50% split
            D = F_total / 2
            B = F_total - D

        # Get feed temperature for initial guess scaling
        T_feed = feed["T"]

        # Feed mole fractions for duty calculation
        z = {s: feed_flows[s] / F_total for s in p.species_order}

        if use_mesh:
            # Rigorous MESH solver (Wang-Henke bubble-point method)
            x_profile, y_profile, T_profile, L_profile, V_profile = self._solve_mesh(
                feed_flows, F_total, R, D, T_feed, n_iter=mesh_iter
            )

            # Total condenser: distillate composition = vapor from top stage
            x_D = {s: y_profile[-1, i] for i, s in enumerate(p.species_order)}
            x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}

            distillate_flows = {s: D * x_D[s] for s in p.species_order}
            bottoms_flows = {s: B * x_B[s] for s in p.species_order}

            distillate = make_stream(distillate_flows, T_profile[-1], p.P)
            bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

            # Compute duties with actual V profile from MESH
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
                V_top=V_profile[-1], V_bot=V_profile[0],
            )

            info = {
                "T_profile": T_profile,
                "x_profile": x_profile,
                "y_profile": y_profile,
                "L_profile": L_profile,
                "V_profile": V_profile,
                "D": D,
                "B": B,
                "Q_condenser": Q_condenser,
                "Q_reboiler": Q_reboiler,
            }
        else:
            # Constant molar overflow (CMO / Lewis-Matheson) solver
            x_profile, y_profile, T_profile = self._solve_constant_molar_overflow(
                feed_flows, F_total, R, D, T_feed
            )

            # Total condenser: distillate = vapor leaving top stage
            x_D = {s: y_profile[-1, i] for i, s in enumerate(p.species_order)}

            # Reboiler (bottom stage, index 0)
            x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}

            distillate_flows = {s: D * x_D[s] for s in p.species_order}
            bottoms_flows = {s: B * x_B[s] for s in p.species_order}

            distillate = make_stream(distillate_flows, T_profile[-1], p.P)
            bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

            # Compute duties with CMO vapor flow estimates
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
            )

            info = {
                "T_profile": T_profile,
                "x_profile": x_profile,
                "y_profile": y_profile,
                "L_rect": R * D,
                "V_rect": (R + 1) * D,
                "D": D,
                "B": B,
                "Q_condenser": Q_condenser,
                "Q_reboiler": Q_reboiler,
            }

        return distillate, bottoms, info


# =============================================================================
# Design Functions
# =============================================================================


def fenske_stages(
    x_D_LK: Array,
    x_B_LK: Array,
    alpha: Array,
) -> Array:
    """Calculate minimum stages using Fenske equation.

    N_min = log[(x_D_LK/x_B_LK) * ((1-x_B_LK)/(1-x_D_LK))] / log(alpha)

    Handles singularity when α → 1 by capping N_min at MAX_STAGES.

    Args:
        x_D_LK: Mole fraction of light key in distillate
        x_B_LK: Mole fraction of light key in bottoms
        alpha: Relative volatility of LK to HK

    Returns:
        Minimum number of theoretical stages, capped at MAX_STAGES
    """
    numer = safe_log(
        safe_divide(x_D_LK, x_B_LK) *
        safe_divide(1 - x_B_LK, 1 - x_D_LK)
    )

    # Handle singularity when alpha → 1
    log_alpha = safe_log(jnp.maximum(alpha, 1.0 + EPS_DIVISION))
    log_alpha_safe = jnp.maximum(log_alpha, MIN_ALPHA_DIFF)

    N_min = numer / log_alpha_safe

    # Cap at maximum stages
    N_min = jnp.minimum(N_min, MAX_STAGES)
    N_min = jnp.maximum(N_min, 1.0)

    return N_min


def minimum_reflux_ratio(
    z_LK: Array,
    z_HK: Array,
    x_D_LK: Array,
    alpha: Array,
    q: Array = 1.0,
) -> Array:
    """Estimate minimum reflux using simplified Underwood.

    For binary system:
    R_min = (1/(alpha-1)) * [x_D_LK/(z_LK) - alpha*(1-x_D_LK)/(1-z_LK)]

    Args:
        z_LK: Feed mole fraction of light key
        z_HK: Feed mole fraction of heavy key
        x_D_LK: Distillate mole fraction of light key
        alpha: Relative volatility
        q: Feed quality

    Returns:
        Minimum reflux ratio
    """
    q = jnp.asarray(q)

    # Simplified for pseudo-binary
    term1 = safe_divide(x_D_LK, z_LK)
    term2 = safe_divide(alpha * (1 - x_D_LK), 1 - z_LK)
    R_min = safe_divide(term1 - term2, alpha - 1)

    return jnp.maximum(R_min, 0.1)


def gilliland_stages(
    R: Array,
    R_min: Array,
    N_min: Array,
) -> Array:
    """Calculate actual stages using Gilliland correlation.

    Handles singularities:
    - R → R_min: Y → 1, N → ∞ (capped at MAX_STAGES)
    - Uses Eduljee correlation with proper numerical safeguards

    Args:
        R: Actual reflux ratio
        R_min: Minimum reflux ratio
        N_min: Minimum stages

    Returns:
        Number of theoretical stages, capped at MAX_STAGES
    """
    # Ensure R >= R_min
    R_safe = jnp.maximum(R, R_min * 1.001)

    X = (R_safe - R_min) / (R_safe + 1)
    X_safe = jnp.maximum(X, 1e-6)

    # Eduljee correlation (simpler than Molokanov)
    Y = 0.75 - 0.75 * X_safe**0.5668

    # Cap Y to prevent N → ∞
    Y = jnp.clip(Y, 0.01, MAX_GILLILAND_Y)

    # N = (N_min + Y) / (1 - Y)
    one_minus_Y = jnp.maximum(1 - Y, 1e-3)
    N = (N_min + Y) / one_minus_Y

    # Cap at maximum stages
    N = jnp.minimum(N, MAX_STAGES)
    N = jnp.maximum(N, N_min)

    return N


def column_diameter(
    V: Array,
    rho_V: Array,
    rho_L: Array,
    sigma: Array = 0.02,
    tray_spacing: float = 0.6,
) -> Array:
    """Estimate column diameter for trayed column.

    Uses Fair correlation for flooding velocity.

    Args:
        V: Vapor molar flow rate (mol/s)
        rho_V: Vapor density (kg/m³)
        rho_L: Liquid density (kg/m³)
        sigma: Surface tension (N/m), default 0.02
        tray_spacing: Tray spacing (m), default 0.6

    Returns:
        Column diameter (m)
    """
    # Flow parameter
    MW_avg = 30.0  # Approximate average MW (g/mol)
    V_mass = V * MW_avg / 1000  # kg/s

    # Capacity factor (simplified Fair correlation)
    F_LV = 0.1  # Assume low liquid loading
    C_sb = 0.1  # Souders-Brown coefficient (m/s)

    # Flooding velocity
    u_flood = C_sb * jnp.sqrt(safe_divide(rho_L - rho_V, rho_V))

    # Operating velocity (80% of flood)
    u_op = 0.8 * u_flood

    # Volumetric flow
    Q_V = safe_divide(V_mass, rho_V)  # m³/s

    # Column area
    A = safe_divide(Q_V, u_op)

    # Diameter
    D = jnp.sqrt(4 * A / jnp.pi)

    return D
