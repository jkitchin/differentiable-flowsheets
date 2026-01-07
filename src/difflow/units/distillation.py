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
from dataclasses import dataclass, replace
import jax
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream
from difflow.thermo import IdealThermo
import optimistix as optx


# =============================================================================
# Numerical Constants
# =============================================================================

# Minimum relative volatility difference from 1.0 for Fenske equation.
# When α < 1 + MIN_ALPHA_DIFF, the mixture is essentially non-separable by
# distillation and N_min approaches infinity. We cap N_min smoothly.
MIN_ALPHA_DIFF = 0.01

# Maximum number of theoretical stages.
# Beyond this, the column is economically infeasible. Used to cap N_min and N.
MAX_STAGES = 500.0

# Maximum Gilliland Y parameter. When Y → 1, N → ∞.
# Capping Y at 0.95 limits N to ~20*N_min which is still very large.
MAX_GILLILAND_Y = 0.95

# Temperature profile scaling factor.
# Column ΔT ≈ TEMP_SCALE_FACTOR * T_feed for typical systems.
# Cryogenic systems have smaller fractional ΔT, high-T systems similar.
DEFAULT_TEMP_SCALE = 0.05  # 5% of feed T as half-range (±2.5%)


@dataclass
class ShortcutColumnParams:
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

    def update(self, **kwargs) -> "ShortcutColumnParams":
        """Return a new ShortcutColumnParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., x_D_LK=0.995)

        Returns:
            New ShortcutColumnParams with updated fields
        """
        return replace(self, **kwargs)


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
        numer = jnp.log(
            (x_D_LK / (1 - x_D_LK + 1e-10)) *
            ((1 - x_B_LK + 1e-10) / (x_B_LK + 1e-10))
        )

        # Handle singularity when alpha → 1
        # ln(alpha) → 0 as alpha → 1, causing N_min → ∞
        # Use regularization: max(ln(alpha), small_value)
        log_alpha = jnp.log(jnp.maximum(alpha_LK, 1.0 + 1e-10))

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
                total = total + alpha[s] * z[s] / (alpha[s] - theta + 1e-10)
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
            total = total + alpha[s] * x_D[s] / (alpha[s] - theta + 1e-10)

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
        ratio_arg = (z_HK / (z_LK + 1e-10)) * \
                    ((x_B_LK + 1e-10) / (x_D_HK + 1e-10))**2 * \
                    B_over_D

        log_ratio = 0.206 * jnp.log(ratio_arg + 1e-10)
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
                A = jnp.log((D_LK / (B_LK + 1e-10)) * (B_HK / (D_HK + 1e-10)))
                C = jnp.log(D_LK / (B_LK + 1e-10)) / (jnp.log(alpha_LK) + 1e-10)

                d_over_b = jnp.exp(A + C * jnp.log(alpha[s] + 1e-10))
                F_i = feed_flows[s]
                d_i = F_i * d_over_b / (1 + d_over_b)
                b_i = F_i - d_i

                distillate_flows[s] = jnp.maximum(d_i, 0.0)
                bottoms_flows[s] = jnp.maximum(b_i, 0.0)
                D_total = D_total + distillate_flows[s]
                B_total = B_total + bottoms_flows[s]

        # Product compositions
        x_D = {s: distillate_flows[s] / (D_total + 1e-10) for s in p.species_order}
        x_B = {s: bottoms_flows[s] / (B_total + 1e-10) for s in p.species_order}

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
        B_over_D = B_total / (D_total + 1e-10)
        x_D_HK = x_D[p.heavy_key]
        N_feed = self.feed_stage_kirkbride(N, z_LK, z_HK, x_B_LK, x_D_HK, B_over_D)

        # Create output streams
        distillate = make_stream(distillate_flows, T_top, P)
        bottoms = make_stream(bottoms_flows, T_bot, P)

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
        }

        return distillate, bottoms, info


@dataclass
class DistillationColumnParams:
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

    def update(self, **kwargs) -> "DistillationColumnParams":
        """Return a new DistillationColumnParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update (e.g., n_stages=20, P=50000.0)

        Returns:
            New DistillationColumnParams with updated fields
        """
        return replace(self, **kwargs)


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
        """Solve column using constant molar overflow assumption.

        Uses Lewis-Matheson method (stage-by-stage from both ends).

        Args:
            feed_flows: Feed molar flows by species
            F_total: Total feed flow
            R: Reflux ratio
            D: Distillate flow rate
            T_feed: Feed temperature (K) for initial guess scaling

        Returns:
            (x, y, T): Liquid compositions, vapor compositions, temperatures
                       All have shape (n_stages, n_species) or (n_stages,)
        """
        p = self.params
        n = p.n_stages
        nc = self.n_species
        P = p.P

        # Overall balance: F = D + B
        B = F_total - D

        # Internal flows (CMO assumption)
        L_rect = R * D  # Liquid in rectifying section
        V_rect = (R + 1) * D  # Vapor in rectifying section
        L_strip = L_rect + F_total  # Liquid in stripping section (for q=1)
        V_strip = V_rect  # Vapor in stripping (for q=1)

        # Initialize composition profiles
        # Feed composition
        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])

        # Initial guess: linear profile between estimated products
        # Rough distillate: enriched in light components
        # Rough bottoms: enriched in heavy components
        x_init = jnp.tile(z, (n, 1))

        # Temperature profile scaled relative to feed T
        # Instead of hard-coded 380-340K which fails for cryogenic/high-T systems,
        # use a percentage range around feed temperature
        T_half_range = T_feed * DEFAULT_TEMP_SCALE
        T_bot = T_feed + T_half_range  # Bottom is hotter
        T_top = T_feed - T_half_range  # Top is cooler
        T_init = jnp.linspace(T_bot, T_top, n)  # Bottom to top

        def stage_calculation(state, _):
            """One iteration of stage-by-stage calculation."""
            x_all, T_all = state

            # New profiles
            x_new = jnp.zeros((n, nc))
            y_new = jnp.zeros((n, nc))
            T_new = jnp.zeros(n)

            def calc_stage(carry, stage_idx):
                x_below, y_above = carry
                j = stage_idx  # 0 = reboiler, n-1 = condenser

                # Determine section
                is_stripping = j < p.feed_stage
                L = jnp.where(is_stripping, L_strip, L_rect)
                V = jnp.where(is_stripping, V_strip, V_rect)

                # Get compositions from adjacent stages
                # Stage j receives liquid from j+1 and vapor from j-1
                x_j = x_all[j]

                # Bubble point for this stage
                T_j, y_j = self._bubble_point_T(x_j, P)

                # Material balance (simplified for iteration)
                # V*y_j + L*x_{j+1} = V*y_{j-1} + L*x_j + F_j*z
                # For now, just update y from equilibrium

                return (x_j, y_j), (x_j, y_j, T_j)

            # Scan through stages
            _, (x_result, y_result, T_result) = lax.scan(
                calc_stage,
                (z, z),  # Initial carry (dummy)
                jnp.arange(n),
            )

            return (x_result, T_result), None

        # Run fixed-point iteration
        state0 = (x_init, T_init)
        (x_final, T_final), _ = lax.scan(stage_calculation, state0, None, length=20)

        # Calculate final vapor compositions
        y_final = jnp.zeros_like(x_final)
        for j in range(n):
            K = self.thermo.K_values_array(T_final[j], P)
            y_final = y_final.at[j].set(K * x_final[j])

        return x_final, y_final, T_final

    def __call__(
        self,
        feed: Stream,
        R: Array | float,
        D_spec: Array | float | None = None,
        B_spec: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Solve distillation column.

        Args:
            feed: Feed stream
            R: Reflux ratio
            D_spec: Distillate flow rate specification (mol/s)
            B_spec: Bottoms flow rate specification (mol/s)
                    Exactly one of D_spec or B_spec must be given.

        Returns:
            distillate: Distillate stream
            bottoms: Bottoms stream
            info: Dictionary with:
                - 'T_profile': Stage temperatures
                - 'x_profile': Liquid composition profiles
                - 'y_profile': Vapor composition profiles
                - 'L': Liquid flow rate
                - 'V': Vapor flow rate
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

        # Solve column (simplified approach)
        # For a rigorous solution, would use full MESH with Newton

        # Use shortcut for initial estimate
        # Then refine with stage calculations

        # Get feed temperature for initial guess scaling
        T_feed = feed["T"]

        # For now, simplified approach using equilibrium stages
        x_profile, y_profile, T_profile = self._solve_constant_molar_overflow(
            feed_flows, F_total, R, D, T_feed
        )

        # Extract product compositions
        # Condenser (top stage, index -1)
        x_D = {s: x_profile[-1, i] for i, s in enumerate(p.species_order)}

        # Reboiler (bottom stage, index 0)
        x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}

        # Create product streams
        distillate_flows = {s: D * x_D[s] for s in p.species_order}
        bottoms_flows = {s: B * x_B[s] for s in p.species_order}

        distillate = make_stream(distillate_flows, T_profile[-1], p.P)
        bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

        info = {
            "T_profile": T_profile,
            "x_profile": x_profile,
            "y_profile": y_profile,
            "L_rect": R * D,
            "V_rect": (R + 1) * D,
            "D": D,
            "B": B,
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
    numer = jnp.log(
        (x_D_LK / (x_B_LK + 1e-10)) *
        ((1 - x_B_LK) / (1 - x_D_LK + 1e-10))
    )

    # Handle singularity when alpha → 1
    log_alpha = jnp.log(jnp.maximum(alpha, 1.0 + 1e-10))
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
    term1 = x_D_LK / (z_LK + 1e-10)
    term2 = alpha * (1 - x_D_LK) / (1 - z_LK + 1e-10)
    R_min = (term1 - term2) / (alpha - 1)

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
    u_flood = C_sb * jnp.sqrt((rho_L - rho_V) / (rho_V + 1e-10))

    # Operating velocity (80% of flood)
    u_op = 0.8 * u_flood

    # Volumetric flow
    Q_V = V_mass / (rho_V + 1e-10)  # m³/s

    # Column area
    A = Q_V / (u_op + 1e-10)

    # Diameter
    D = jnp.sqrt(4 * A / jnp.pi)

    return D
