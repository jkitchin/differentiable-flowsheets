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
from difflow.thermo import CubicThermo, IdealThermo
from difflow.params_mixin import ParamsMixin
from difflow.constants import MIN_ALPHA_DIFF, MAX_STAGES, MAX_GILLILAND_Y, EPS_DIVISION
from difflow.numerics import safe_divide, safe_log
import optimistix as optx


# Second pass of the rigorous column's bubble-point solve, used only when the
# thermo package's K-values depend on composition (an EOS). The step cap keeps
# an iterate inside the temperature window where the cubic has two real roots
# -- outside it the liquid and vapor roots coincide, K collapses to 1, and the
# residual goes flat.
_BUBBLE_T_REFINE_STEPS = 8
_BUBBLE_T_MAX_STEP = 25.0
# Below this the bubble-point residual is flat, i.e. the EOS is reporting one
# phase rather than two, and a Newton step there is 0/0.
_BUBBLE_T_FLAT_DR = 1e-8

# First pass of the bubble-point solve: damped Newton on log(sum(K x)), which
# is well scaled where the residual itself is exponentially flat.
_SEED_T_STEPS = 10
_SEED_T_MAX_STEP = 100.0
_SEED_T_FLOOR = 20.0

# Sweeps of the shortcut column's end-temperature fixed point: the relative
# volatilities are evaluated at the product bubble points, and the products
# come from the volatilities.
_SHORTCUT_END_T_ITERS = 3


def _mixture_molar_enthalpy(
    thermo: "IdealThermo | CubicThermo",
    species_order: list[str],
    mole_fracs: Array,
    T: Array,
    phase: str,
    P: Array,
) -> Array:
    """Molar enthalpy (J/mol) of one phase of a mixture.

    Goes through ``thermo.stream_enthalpy`` on a one-mole basis rather than
    summing ``H_pure`` so that a column is not tied to one thermodynamic model:
    for an :class:`~difflow.thermo.IdealThermo` this is exactly
    ``sum_i z_i H_pure_i`` (its enthalpy is a mole-fraction-weighted sum of
    pure-component enthalpies and it ignores ``P``), while for a
    :class:`~difflow.thermo.CubicThermo` it is the ideal-gas sensible enthalpy
    plus the EOS departure for this phase at the column pressure, so the latent
    heat comes from the same EOS as the K-values.

    Args:
        thermo: Thermodynamic property calculator.
        species_order: Species names, in the order ``mole_fracs`` is given.
        mole_fracs: (nc,) mole fractions of the phase.
        T: Temperature (K).
        phase: 'liquid' or 'vapor'.
        P: Pressure (Pa).

    Returns:
        Molar enthalpy (J/mol).
    """
    flows = {s: mole_fracs[i] for i, s in enumerate(species_order)}
    return thermo.stream_enthalpy(flows, T, phase=phase, P=jnp.asarray(P))


def _bubble_T(
    thermo: "IdealThermo | CubicThermo",
    x: Array,
    P: Array,
    T_guess: Array | None = None,
) -> tuple[Array, Array]:
    """Bubble-point temperature of a liquid mixture, and its incipient vapor.

    Solves ``sum(K_i x_i) = 1`` for T.

    With an ideal thermo package that is one Newton solve on Raoult K-values.
    With an EOS package it is that solve followed by a safeguarded refinement
    on the EOS K-values, which are a function of composition as well as of T;
    see the comments below for why the refinement is written by hand rather
    than handed to a root finder.

    Args:
        thermo: Thermodynamic property calculator.
        x: Liquid mole fractions.
        P: Pressure (Pa).
        T_guess: Initial temperature guess (K). If None, a pressure-scaled
            default is used.

    Returns:
        (T, y): The bubble temperature and the vapor in equilibrium with it.
    """
    x = jnp.asarray(x)

    def residual_from_K(K: Array) -> Array:
        return jnp.sum(K * x) - 1.0

    def K_at(T: Array) -> Array:
        """K-values at the composition this bubble point is posed at."""
        return thermo.K_values_array(T, P, x)

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

    # Pass 1: composition-free K-values -- Raoult's law either way, since a
    # CubicThermo hands this call back to the IdealThermo it wraps. They are
    # smooth and monotone in T, with a single root, so this pass owns getting
    # into the right neighbourhood from a cold start. That is also what pass 2
    # needs: away from the bubble point the EOS cubic has a single real root,
    # both phases then take that same root, and K collapses identically to 1 --
    # a flat-zero residual a Newton step cannot recover from. Pass 1 lands
    # inside the window where the EOS has two roots, so pass 2 never starts
    # outside it.
    def estimate_residual(T, args):
        return residual_from_K(thermo.K_values_array(T, P))

    # Pass 1a: a few damped Newton steps on log(sum(K x)) rather than on
    # sum(K x) - 1. Vapor pressure is exponential in -1/T, so far below the
    # bubble point sum(K x) is ~0 and so is its slope: Newton on the plain
    # residual divides one tiny number by another and steps tens of thousands
    # of degrees, into the region where the Antoine exponent overflows. Its
    # logarithm has a slope that stays O(1/K) over the whole range -- it is
    # nearly linear in 1/T -- so the same step is a sane one. Capped and
    # floored on top of that, because "nearly" is not "exactly".
    def log_residual(T):
        # Not safe_log: its 1e-15 floor is inside the range this pass has to
        # work over. Two hundred degrees under the bubble point sum(K x) is
        # ~1e-42 -- an ordinary float64 -- and clipping it there would flatten
        # the very slope the step is derived from, stalling the seed at its
        # starting point. Float64 goes down to ~1e-308, so the floor here only
        # has to stop a true underflow producing -inf.
        return jnp.log(jnp.maximum(jnp.sum(thermo.K_values_array(T, P) * x), 1e-300))

    def seed_step(T_k, _):
        r, dr = jax.value_and_grad(log_residual)(T_k)
        ok = jnp.isfinite(r) & jnp.isfinite(dr) & (jnp.abs(dr) > 1e-12)
        r_safe = jnp.where(ok, r, 0.0)
        dr_safe = jnp.where(ok, dr, 1.0)
        dT = jnp.clip(-r_safe / dr_safe, -_SEED_T_MAX_STEP, _SEED_T_MAX_STEP)
        T_next = jnp.maximum(T_k + dT, _SEED_T_FLOOR)
        return jnp.where(jnp.isfinite(T_next), T_next, T_k), None

    T_seed, _ = lax.scan(seed_step, T_init, None, length=_SEED_T_STEPS)

    # Pass 1b: finish on the residual itself. Seeded this close it converges in
    # a step or two, and optimistix differentiates it implicitly, so the seed
    # above never appears in the gradient.
    solver = optx.Newton(rtol=1e-10, atol=1e-10)
    sol = optx.root_find(
        estimate_residual, solver, T_seed, args=None, max_steps=50, throw=False,
    )
    T = sol.value

    # Pass 2: refine against the composition-dependent K-values. Skipped when
    # the thermo package's K-values are a function of (T, P) alone, where pass
    # 1 already solved the real residual.
    if getattr(thermo, "K_depends_on_composition", True):
        def refined_residual(T):
            return residual_from_K(K_at(T))

        # A safeguarded Newton written out rather than handed to a root finder.
        # An EOS K-value only means anything over the temperature window where
        # its cubic has two roots at this composition; outside it both phases
        # take the one root, K is identically 1, and the residual is a flat
        # zero that a root finder reads as converged wherever it happens to be
        # standing. So: cap the step, and when a step lands somewhere flat
        # (dr = 0), bisect back toward the last temperature that was not,
        # instead of taking 0/0. The pass-1 estimate is the first such
        # temperature, which makes the worst case "return the ideal-K bubble
        # point", never a NaN.
        def newton_step(carry, _):
            T_k, T_ok = carry
            r, dr = jax.value_and_grad(refined_residual)(T_k)
            in_window = (
                jnp.isfinite(r) & jnp.isfinite(dr)
                & (jnp.abs(dr) > _BUBBLE_T_FLAT_DR)
            )
            # Sanitise before the divide, not after: a NaN that has already
            # been formed is something a later jnp.where can hide from the
            # value but not from the derivative.
            r_safe = jnp.where(in_window, r, 0.0)
            dr_safe = jnp.where(in_window, dr, 1.0)
            dT = jnp.clip(
                -r_safe / dr_safe, -_BUBBLE_T_MAX_STEP, _BUBBLE_T_MAX_STEP
            )
            T_next = jnp.where(in_window, T_k + dT, 0.5 * (T_k + T_ok))
            return (T_next, jnp.where(in_window, T_k, T_ok)), None

        (T, _), _ = lax.scan(
            newton_step, (T, T), None, length=_BUBBLE_T_REFINE_STEPS
        )

    y = K_at(T) * x
    return T, y / jnp.sum(y)


@dataclass(repr=False)
class ShortcutColumnParams(ParamsMixin):
    """Parameters for shortcut distillation column design.

    Attributes:
        species_order: List of species names for array ordering
        light_key: Name of light key component
        heavy_key: Name of heavy key component
        x_D_LK: Fractional **recovery** of the light key in the distillate --
            0.99 sends 99 % of the feed's light key overhead. Despite the
            name, which is historical, it is not a mole fraction; the
            distillate's actual light-key mole fraction comes out in
            ``info["x_D"]``.
        x_B_HK: Fractional recovery of the heavy key in the bottoms, likewise.
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

    symbol = "Shortcut Column"
    equations = [
        r"N_\mathrm{min} = \frac{\ln\!\bigl[(x_{LK}/x_{HK})_D\,(x_{HK}/x_{LK})_B\bigr]}{\ln \alpha_{LK,HK}} \qquad \text{(Fenske)}",
        r"\sum_i \frac{\alpha_i\, z_i}{\alpha_i - \theta} = 1 - q \qquad \text{(Underwood 1st)}",
        r"R_\mathrm{min} + 1 = \sum_i \frac{\alpha_i\, x_{i,D}}{\alpha_i - \theta} \qquad \text{(Underwood 2nd)}",
        r"\frac{N - N_\mathrm{min}}{N+1} = f\!\left(\frac{R - R_\mathrm{min}}{R+1}\right) \qquad \text{(Gilliland)}",
    ]
    assumptions = [
        "Constant relative volatility between top and bottom.",
        "Constant molar overflow (CMO).",
        "Feed thermal condition q supplied or assumed saturated liquid (q=1).",
        "Key components define recovery specifications.",
    ]
    references = [
        "Fenske, M.R. Ind. Eng. Chem., 24, 482 (1932).",
        "Underwood, A.J.V. J. Inst. Petroleum, 32, 614 (1946).",
        "Gilliland, E.R. Ind. Eng. Chem., 32, 1220 (1940).",
    ]
    parameter_symbols = {
        "light_key": "LK",
        "heavy_key": "HK",
        "x_D_LK": "x_{D,LK}",
        "x_B_HK": "x_{B,HK}",
    }
    parameter_units = {"x_D_LK": "-", "x_B_HK": "-"}
    numerical_method = "Closed-form Fenske + Underwood + Gilliland correlation with smooth capping for singular cases."

    def __init__(
        self,
        params: ShortcutColumnParams,
        thermo: IdealThermo | CubicThermo,
    ):
        """Initialize shortcut column.

        Args:
            params: Column parameters. Supplies the K-values the relative
                volatilities are built from and the enthalpies the duties are
                built from, so it sets what the whole
                Fenske-Underwood-Gilliland chain is worth: an
                :class:`~difflow.thermo.IdealThermo` gives Raoult
                volatilities, a :class:`~difflow.thermo.CubicThermo` gives EOS
                ones. For light hydrocarbons at pressure the difference runs
                through to the answer -- on a C3-C8 cut at 10 bar the light
                key's alpha is 2.32 on Raoult against 1.81 on Peng-Robinson,
                which is 3.7 minimum stages against 5.6.
            thermo: Thermodynamic property calculator for K-values
        """
        self.params = params
        self.thermo = thermo

    def relative_volatility(
        self,
        T: Array,
        P: Array,
        x: Array | None = None,
    ) -> dict[str, Array]:
        """Calculate relative volatilities with respect to heavy key.

        alpha_i = K_i / K_HK

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            x: Liquid mole fractions in species order, for a thermo package
               whose K-values depend on composition (a
               :class:`~difflow.thermo.CubicThermo`). Ignored by Raoult
               K-values, which are a function of (T, P) alone. Omitting it on
               an EOS package falls back to that package's composition-free
               estimate, which is a worse relative volatility than one
               evaluated at the composition the column end actually has.

        Returns:
            Dictionary of relative volatilities by species
        """
        K = self.thermo.K_values(T, P, x)
        K_HK = K[self.params.heavy_key]

        return {s: K[s] / K_HK for s in self.params.species_order}

    def average_alpha(
        self,
        T_top: Array,
        T_bot: Array,
        P: Array,
        x_top: Array | None = None,
        x_bot: Array | None = None,
    ) -> dict[str, Array]:
        """Calculate geometric mean relative volatility.

        alpha_avg = sqrt(alpha_top * alpha_bottom)

        Args:
            T_top: Top temperature (K) -- the condenser, i.e. the bubble point
                of the distillate
            T_bot: Bottom temperature (K) -- the reboiler, i.e. the bubble
                point of the bottoms
            P: Column pressure (Pa)
            x_top: Distillate mole fractions, for a composition-dependent
                thermo package (see :meth:`relative_volatility`)
            x_bot: Bottoms mole fractions, likewise

        Returns:
            Dictionary of average relative volatilities
        """
        alpha_top = self.relative_volatility(T_top, P, x_top)
        alpha_bot = self.relative_volatility(T_bot, P, x_bot)

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

    def _split_products(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        z: dict[str, Array],
        alpha: dict[str, Array],
    ) -> tuple[dict, dict, Array, Array, dict, dict]:
        """Distribute the feed between the products at a given volatility set.

        The keys are placed by their specified recoveries; every other species
        is distributed by the Hengstebeck-Geddes equation,
        ``log(d_i/b_i) = A + C log(alpha_i)``, with A and C fixed by the keys.
        Split out of :meth:`__call__` because it has to be re-evaluated as the
        column-end temperatures converge: the split depends on ``alpha``, and
        ``alpha`` is evaluated at the ends, which are the bubble points of the
        products this returns.

        Args:
            feed_flows: Feed molar flows by species (mol/s).
            F_total: Total feed flow (mol/s).
            z: Feed mole fractions by species.
            alpha: Relative volatilities by species (vs the heavy key).

        Returns:
            (distillate_flows, bottoms_flows, D_total, B_total, x_D, x_B)
        """
        p = self.params
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

        return distillate_flows, bottoms_flows, D_total, B_total, x_D, x_B

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
            distillate: Distillate stream, at the condenser temperature --
                a total condenser delivers saturated liquid, so this is the
                bubble point of the distillate
            bottoms: Bottoms stream, at the reboiler temperature -- the
                bubble point of the bottoms
            info: Dictionary with design information:
                - 'N_min': Minimum stages
                - 'R_min': Minimum reflux ratio
                - 'N': Actual stages
                - 'N_feed': Feed stage
                - 'alpha': Relative volatilities, the geometric mean of the
                  values at the two column ends
                - 'T_top'/'T_condenser': Condenser temperature (K)
                - 'T_bot'/'T_reboiler': Reboiler temperature (K)
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

        # Column-end temperatures. These are not free estimates: the top of the
        # column is a total condenser, so it sits at the bubble point of the
        # distillate, and the bottom is the reboiler, at the bubble point of
        # the bottoms. Fenske-Underwood-Gilliland wants the relative
        # volatilities at those two states -- which makes them a small fixed
        # point, since the products' compositions come from the volatilities
        # in turn. Seed it from the feed's own bubble point and sweep a few
        # times; the products separate on the first pass and the temperatures
        # settle within a fraction of a degree after that.
        T_feed = feed["T"]
        z_arr = jnp.array([z[s] for s in p.species_order])
        T_end, _ = _bubble_T(self.thermo, z_arr, P, T_guess=T_feed)
        T_top = T_end
        T_bot = T_end
        x_D_arr = z_arr
        x_B_arr = z_arr

        def alpha_and_split(T_top, T_bot, x_D_arr, x_B_arr):
            """One sweep: volatilities at the ends, then the product split."""
            alpha_top = self.relative_volatility(T_top, P, x_D_arr)
            alpha_bot = self.relative_volatility(T_bot, P, x_B_arr)
            alpha = {
                s: jnp.sqrt(alpha_top[s] * alpha_bot[s])
                for s in p.species_order
            }
            split = self._split_products(feed_flows, F_total, z, alpha)
            return alpha_top, alpha_bot, alpha, split

        for _ in range(_SHORTCUT_END_T_ITERS):
            *_, (_, _, _, _, x_D, x_B) = alpha_and_split(
                T_top, T_bot, x_D_arr, x_B_arr
            )
            x_D_arr = jnp.array([x_D[s] for s in p.species_order])
            x_B_arr = jnp.array([x_B[s] for s in p.species_order])
            T_top, _ = _bubble_T(self.thermo, x_D_arr, P, T_guess=T_top)
            T_bot, _ = _bubble_T(self.thermo, x_B_arr, P, T_guess=T_bot)

        # A last sweep so the volatilities and the split that get reported are
        # the ones belonging to the converged end temperatures.
        alpha_top, alpha_bot, alpha, split = alpha_and_split(
            T_top, T_bot, x_D_arr, x_B_arr
        )
        distillate_flows, bottoms_flows, D_total, B_total, x_D, x_B = split
        x_D_arr = jnp.array([x_D[s] for s in p.species_order])
        x_B_arr = jnp.array([x_B[s] for s in p.species_order])
        alpha_LK = alpha[p.light_key]
        z_LK = z[p.light_key]
        z_HK = z[p.heavy_key]

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

        # Create output streams. The distillate leaves the condenser as a
        # saturated liquid (T_top is its bubble point); the bottoms leaves the
        # reboiler at its own bubble point.
        distillate = make_stream(distillate_flows, T_top, P)
        bottoms = make_stream(bottoms_flows, T_bot, P)

        # Compute condenser and reboiler duties including latent heat
        # Condenser: condense all vapor from the top stage
        V_top = (R + 1) * D_total  # Vapor flow at top stage

        def h_mix(mole_fracs, T, phase):
            return _mixture_molar_enthalpy(
                self.thermo, p.species_order, mole_fracs, T, phase, P
            )

        # Condensing vapor of the distillate's composition to liquid of the
        # same composition, both at the condenser temperature: the latent heat
        # of the distillate. This is the standard shortcut duty, and it is
        # where the shortcut and rigorous columns genuinely differ -- the
        # rigorous column knows its top *tray* temperature and so also charges
        # the condenser for cooling the vapor from the tray down to the
        # condenser, while a shortcut method has no tray profile to take that
        # temperature from. The dew point of x_D would supply one on paper, but
        # a dew point is set by the heaviest trace in the mixture, and the
        # heaviest trace in x_D is the least reliable number the
        # Hengstebeck-Geddes distribution produces.
        H_vapor_top = h_mix(x_D_arr, T_top, 'vapor')

        # Feed enthalpy
        h_F = h_mix(z_arr, T_feed, 'liquid')

        # Distillate liquid enthalpy (total condenser: saturated liquid)
        h_D = h_mix(x_D_arr, T_top, 'liquid')

        # Bottoms liquid enthalpy
        h_B = h_mix(x_B_arr, T_bot, 'liquid')

        # Condenser duty (heat removed, negative)
        Q_condenser = V_top * (h_D - H_vapor_top)

        # Reboiler duty from overall energy balance
        # F*h_F + Q_reb = D*h_D + B*h_B + |Q_cond|
        # Q_reb = D*h_D + B*h_B - Q_cond - F*h_F  (Q_cond < 0)
        Q_reboiler = D_total * h_D + B_total * h_B - Q_condenser - F_total * h_F

        # Alpha variation diagnostics (#87)
        alpha_variation = {
            s: jnp.abs(alpha_top[s] - alpha_bot[s]) / jnp.maximum(alpha[s], 1e-10)
            for s in p.species_order
        }
        max_alpha_variation = jnp.max(jnp.array([alpha_variation[s] for s in p.species_order]))

        # Feasibility checks (#158)
        alpha_insufficient = alpha_LK < (1.0 + MIN_ALPHA_DIFF)
        negative_flows_detected = jnp.any(jnp.array([
            jnp.any(jnp.array([distillate_flows[s] for s in p.species_order]) < 0),
            jnp.any(jnp.array([bottoms_flows[s] for s in p.species_order]) < 0),
        ]))
        reflux_ok = ~near_min_reflux
        alpha_ok = ~alpha_insufficient
        flows_ok = ~negative_flows_detected
        feasible = alpha_ok & reflux_ok & flows_ok

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
            "T_condenser": T_top,
            "T_reboiler": T_bot,
            "close_boiling": close_boiling,
            "near_min_reflux": near_min_reflux,
            "Q_condenser": Q_condenser,
            "Q_reboiler": Q_reboiler,
            # Alpha variation (#87)
            "alpha_top": alpha_top,
            "alpha_bot": alpha_bot,
            "alpha_variation": alpha_variation,
            "alpha_varies_significantly": max_alpha_variation > 0.3,
            # Feasibility (#158)
            "feasible": feasible,
            "alpha_insufficient": alpha_insufficient,
            "negative_flows_detected": negative_flows_detected,
        }

        return distillate, bottoms, info


@dataclass(repr=False)
class DistillationColumnParams(ParamsMixin):
    """Parameters for rigorous distillation column.

    Stages are indexed from the bottom, zero-based: stage 0 is the reboiler,
    stage ``n_stages - 1`` is the top tray. The condenser is not a stage -- it
    sits outside the cascade and is what ``condenser_type`` describes -- so
    ``n_stages`` counts the trays plus the reboiler, and every profile in
    ``info`` is in that same order (``T_profile[0]`` is the reboiler,
    ``T_profile[-1]`` the top tray).

    Attributes:
        species_order: List of species names. Sets the column order of every
            array in ``info``, and must match the thermo package's species.
        n_stages: Number of equilibrium stages: the trays plus the reboiler.
        feed_stage: Stage the feed enters, as a zero-based index from the
            bottom (0 = reboiler). Stages below it are the stripping section
            and stages above it the rectifying section. Which side the feed
            stage itself is counted on differs between the CMO sweep and the
            MESH flow initialisation, so do not read anything into the
            boundary stage.
        condenser_type: 'total' or 'partial'. Only 'total' is implemented;
            'partial' is rejected rather than silently treated as total.
        P: Column pressure (Pa). One pressure for the whole column -- there is
            no tray pressure drop.
        q: Feed thermal condition (1.0 = saturated liquid, 0.0 = saturated
            vapor).
    """
    species_order: list[str]
    n_stages: int
    feed_stage: int
    condenser_type: Literal["total", "partial"] = "total"
    P: float = 101325.0
    q: float = 1.0  # Feed thermal condition (1.0 = saturated liquid, 0.0 = saturated vapor)

    def __post_init__(self):
        if self.condenser_type != "total":
            raise NotImplementedError(
                f"condenser_type={self.condenser_type!r} is not implemented; "
                "the MESH solver models a total condenser (the distillate has "
                "the composition of the top tray's vapor and leaves as a "
                "saturated liquid). A partial condenser would be an extra "
                "equilibrium stage with a vapor product."
            )


class DistillationColumn:
    """Rigorous stage-by-stage distillation column.

    Solves MESH equations (Material, Equilibrium, Summation, Heat balance)
    for each stage using the bubble-point method.

    Stage numbering: zero-based from the bottom -- stage 0 is the reboiler,
    stage ``n_stages - 1`` is the top tray. The total condenser is outside the
    cascade, so the distillate is not a stage's product (see
    :meth:`_condenser_T`).

    Assumptions:
    - Equilibrium stages
    - No pressure drop between stages
    - Total condenser
    - Constant molar overflow (CMO) for the initial solution

    All calculations are JAX-compatible for automatic differentiation.
    """

    symbol = "MESH Column"
    equations = [
        r"\text{M: } L_{n+1} x_{i,n+1} + V_{n-1} y_{i,n-1} + F_n z_{i,n} - (L_n + U_n) x_{i,n} - (V_n + W_n) y_{i,n} = 0",
        r"\text{E: } y_{i,n} = K_i(T_n, P)\, x_{i,n}",
        r"\text{S: } \sum_i x_{i,n} = 1,\quad \sum_i y_{i,n} = 1",
        r"\text{H: } L_{n+1} h_{n+1} + V_{n-1} H_{n-1} + F_n h_{F,n} - (L_n + U_n) h_n - (V_n + W_n) H_n \pm Q_n = 0",
    ]
    assumptions = [
        "Equilibrium stages (Murphree efficiency implicitly 1; can be modified).",
        "Constant molar overflow for initialization.",
        "No pressure drop between stages.",
    ]
    references = [
        "King, C.J. Separation Processes, 2e, McGraw-Hill, 1980.",
        "Kister, H.Z. Distillation Design, McGraw-Hill, 1992.",
        "Seader, Henley, Roper. Separation Process Principles, 3e.",
    ]
    parameter_symbols = {
        "n_stages": "N",
        "feed_stage": "n_F",
        "P": "P",
        "q": "q",
    }
    parameter_units = {
        "n_stages": "-",
        "feed_stage": "-",
        "P": "Pa",
        "q": "-",
    }
    numerical_method = "MESH stage-by-stage with bubble-point method and simultaneous Newton update on T profile."

    def __init__(
        self,
        params: DistillationColumnParams,
        thermo: IdealThermo | CubicThermo,
    ):
        """Initialize distillation column.

        Args:
            params: Column parameters
            thermo: Thermodynamic property calculator. An
                :class:`~difflow.thermo.IdealThermo` gives Raoult K-values;
                a :class:`~difflow.thermo.CubicThermo` gives EOS K-values
                (``K_i = phi_i^L / phi_i^V``) and EOS enthalpies, which is what
                a light-hydrocarbon column at pressure needs -- Raoult's law is
                out by tens of degrees at the condenser there. The column code
                is the same either way; the composition dependence of the EOS
                K-values rides along inside the existing MESH iteration.
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

        Solves: sum(K_i * x_i) = 1 for T. A thin wrapper on the module-level
        :func:`_bubble_T`, which both columns share.

        Args:
            x: Liquid mole fractions
            P: Pressure (Pa)
            T_guess: Initial temperature guess (K). If None, estimates from
                     pure component data or uses a pressure-scaled default.

        Returns:
            (T, y): Bubble temperature and vapor composition
        """
        return _bubble_T(self.thermo, x, P, T_guess)

    def _condenser_T(
        self,
        x_D: Array,
        P: Array,
        T_guess: Array | None = None,
    ) -> Array:
        """Temperature of a total condenser's outlet (K).

        A total condenser condenses the whole of the top stage's vapor, so its
        outlet -- the distillate, and the reflux returned to the top stage --
        is a saturated liquid of composition ``x_D`` at the column pressure.
        Its temperature is therefore the bubble point of ``x_D``, which is
        **not** the top stage temperature: the top stage sits at the bubble
        point of its own liquid ``x_top``, equivalently the dew point of the
        vapor ``y_top = x_D`` it sends to the condenser, and a mixture's dew
        point is above its bubble point. The gap is the width of the cut --
        small for a sharp binary split, tens of degrees for a wide-boiling
        multicomponent distillate.

        Args:
            x_D: Distillate mole fractions (= the top stage vapor, for a total
                condenser).
            P: Column pressure (Pa).
            T_guess: Initial temperature guess (K); the top stage temperature
                is a reasonable one, being an upper bound.

        Returns:
            Condenser temperature (K).
        """
        return self._bubble_point_T(x_D, P, T_guess=T_guess)[0]

    def _cmo_flows(
        self,
        F_total: Array,
        R: Array,
        D: Array,
    ) -> tuple[Array, Array]:
        """Liquid and vapor flow profiles under constant molar overflow.

        Stage numbering: j=0 is the reboiler (bottom), j=n-1 the top stage.
        ``L[j]`` is the liquid leaving stage j downward and ``V[j]`` the vapor
        leaving stage j upward.  Above the feed these are the
        rectifying-section values; at and below the feed stage the liquid
        picks up the feed's liquid fraction q and the vapor loses its vapor
        fraction (1-q).

        Args:
            F_total: Total feed flow rate (mol/s)
            R: Reflux ratio
            D: Distillate flow rate (mol/s)

        Returns:
            (L, V): flow profiles, each of shape (n_stages,)
        """
        p = self.params
        n = p.n_stages
        q = jnp.asarray(p.q)
        B = F_total - D

        L_rect = R * D
        V_rect = (R + 1) * D
        L_strip = L_rect + q * F_total
        V_strip = V_rect - (1.0 - q) * F_total

        j = jnp.arange(n)
        L = jnp.where(j <= p.feed_stage, L_strip, L_rect)
        L = L.at[0].set(B)  # liquid leaving the reboiler is the bottoms product
        V = jnp.where(j < p.feed_stage, V_strip, V_rect)
        return L, V

    def _bubble_point_step(
        self,
        x_D: Array,
        x: Array,
        T: Array,
        L: Array,
        V: Array,
        F_vec: Array,
        z: Array,
        R: Array,
        D: Array,
    ) -> tuple[Array, Array, Array]:
        """One bubble-point (Wang-Henke) step at frozen L/V flows.

        Solves the component material balances for the whole column as a
        tridiagonal system, then updates the stage temperatures from the
        bubble point and recomputes the equilibrium vapor compositions.

        The tridiagonal system *is* the per-species material balance, so
        summing it over the stages telescopes to
        ``D x_{D,i} + B x_{B,i} = F z_i`` once the lagged reflux composition
        has converged.  Sweeps that instead march an operating line stage by
        stage have no such guarantee (issue #211).

        Args:
            x_D: (nc,) reflux composition.  With a total condenser this is the
                 vapor composition leaving the top stage, lagged one iteration.
            x: (n, nc) current liquid compositions.  The K-values are evaluated
               at these alongside T: an EOS K-value depends on the composition,
               a Raoult one ignores it.
            T: (n,) stage temperatures (K)
            L: (n,) liquid flows leaving each stage downward (mol/s)
            V: (n,) vapor flows leaving each stage upward (mol/s)
            F_vec: (n,) feed flow entering each stage (mol/s)
            z: (nc,) feed mole fractions
            R: Reflux ratio
            D: Distillate flow rate (mol/s)

        Returns:
            (x, y, T): updated liquid compositions (n, nc), vapor
                       compositions (n, nc) and temperatures (n,)
        """
        p = self.params
        n = p.n_stages
        nc = self.n_species
        P = jnp.asarray(p.P)

        # 1. K values at the current stage temperatures and compositions
        _, K = lax.scan(
            lambda _, args: (
                None, self.thermo.K_values_array(args[1], P, args[0])
            ),
            None, (x, T)
        )
        # K shape: (n, nc)

        # 2. Tridiagonal component balances for all components simultaneously.
        # Material balance at stage j (bottom-up numbering):
        #   V[j-1]*K[j-1,i]*x[j-1,i] - (L[j] + K[j,i]*V[j])*x[j,i]
        #   + L[j+1]*x[j+1,i] = -F[j]*z[i]
        V_prev = jnp.concatenate([jnp.zeros(1), V[:-1]])   # V[j-1]
        K_prev = jnp.concatenate([jnp.zeros((1, nc)), K[:-1]], axis=0)  # K[j-1]
        lower = V_prev[:, None] * K_prev   # (n, nc), lower[0] = 0

        diag = -(L[:, None] + K * V[:, None])  # (n, nc)

        L_next = jnp.concatenate([L[1:], jnp.zeros(1)])    # L[j+1]
        upper = jnp.broadcast_to(L_next[:, None], (n, nc))  # (n, nc)
        upper = upper.at[-1].set(0.0)  # no stage above top

        rhs = -F_vec[:, None] * z[None, :]  # (n, nc)

        # Top boundary condition: reflux enters stage n-1 with composition x_D
        rhs = rhs.at[-1].add(-R * D * x_D)

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
            c_new = upper[j] / m_safe                        # (nc,)
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

        # 4. Update T from the bubble point at each stage
        _, T_new = lax.scan(
            lambda _, args: (None, self._bubble_point_T(args[0], P, args[1])[0]),
            None, (x_new, T)
        )

        # 5. Recompute K and y with the updated T and compositions
        _, K_new = lax.scan(
            lambda _, args: (
                None, self.thermo.K_values_array(args[1], P, args[0])
            ),
            None, (x_new, T_new)
        )
        y_new = K_new * x_new
        y_new = y_new / jnp.maximum(
            jnp.sum(y_new, axis=1, keepdims=True), 1e-10
        )

        return x_new, y_new, T_new

    def _solve_constant_molar_overflow(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        R: Array,
        D: Array,
        T_feed: Array,
        n_iter: int = 30,
    ) -> tuple[Array, Array, Array]:
        """Solve column using the constant molar overflow (CMO) assumption.

        Runs the same bubble-point (Wang-Henke) iteration as the MESH solver
        but with the L/V profiles frozen at their CMO values instead of being
        corrected by stage energy balances.  What the CMO shortcut gives up is
        therefore the energy balance alone: because each iteration solves the
        component balances as a tridiagonal system, the converged profile
        still closes the per-species material balance over the column.

        Stage numbering: j=0 is reboiler (bottom), j=n-1 is top stage.
        Total condenser assumed: x_D = y_{top} (vapor from top stage).

        Args:
            feed_flows: Feed molar flows by species
            F_total: Total feed flow
            R: Reflux ratio
            D: Distillate flow rate
            T_feed: Feed temperature (K), used for the initial guess
            n_iter: Number of bubble-point iterations

        Returns:
            (x, y, T): Liquid compositions, vapor compositions, temperatures
                       All have shape (n_stages, n_species) or (n_stages,)
        """
        p = self.params
        n = p.n_stages
        P = jnp.asarray(p.P)

        B = F_total - D
        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])

        # Feed enters on a single stage; flows are fixed by the CMO assumption.
        F_vec = jnp.zeros(n).at[p.feed_stage].set(F_total)
        L, V = self._cmo_flows(F_total, R, D)

        # Initial guess: one-stage flash of the feed enriches the distillate
        # estimate, and the overall balance gives the matching bottoms.
        T_bp_feed, y_flash = self._bubble_point_T(
            z, P, T_guess=jnp.asarray(T_feed)
        )
        x_D_init = jnp.clip(y_flash, 1e-6, 1.0 - 1e-6)
        x_D_init = x_D_init / jnp.sum(x_D_init)

        x_B_init = (F_total * z - D * x_D_init) / jnp.maximum(B, 1e-10)
        x_B_init = jnp.clip(x_B_init, 1e-6, 1.0 - 1e-6)
        x_B_init = x_B_init / jnp.sum(x_B_init)

        # Linear composition profile: reboiler (j=0) -> top (j=n-1)
        frac = jnp.linspace(0.0, 1.0, n)
        x_init = frac[:, None] * x_D_init[None, :] + (1.0 - frac[:, None]) * x_B_init[None, :]

        # Start every stage at the feed bubble point: tracks the mixture and
        # the column pressure instead of assuming an ambient-organic value.
        T_init = jnp.full(n, T_bp_feed)

        # Stage vapor compositions from equilibrium at the initial profile.
        # The liquid composition goes into K too: an EOS K-value depends on it,
        # a Raoult K-value ignores it.
        def init_y(_, x_j):
            K_j = self.thermo.K_values_array(T_bp_feed, P, x_j)
            yx_j = K_j * x_j
            return None, yx_j / jnp.maximum(jnp.sum(yx_j), 1e-10)

        _, y_init = lax.scan(init_y, None, x_init)

        def one_iteration(carry, _):
            x, y, T = carry
            # Total condenser: the reflux carries the composition of the vapor
            # leaving the top stage.
            return self._bubble_point_step(
                y[-1], x, T, L, V, F_vec, z, R, D
            ), None

        (x_final, y_final, T_final), _ = lax.scan(
            one_iteration, (x_init, y_init, T_init), None, length=n_iter
        )

        return x_final, y_final, T_final

    def _molar_enthalpy(
        self,
        mole_fracs: Array,
        T: Array,
        phase: str,
    ) -> Array:
        """Molar enthalpy (J/mol) of one phase at a stage.

        A thin wrapper on the module-level :func:`_mixture_molar_enthalpy`,
        bound to this column's species order and pressure.

        Args:
            mole_fracs: (nc,) mole fractions of the phase, in species order.
            T: Stage temperature (K).
            phase: 'liquid' or 'vapor'.

        Returns:
            Molar enthalpy (J/mol).
        """
        p = self.params
        return _mixture_molar_enthalpy(
            self.thermo, p.species_order, mole_fracs, T, phase, jnp.asarray(p.P)
        )

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
        def stage_enthalpies(_, inputs):
            x_j, y_j, T_j = inputs
            h_j = self._molar_enthalpy(x_j, T_j, 'liquid')
            H_j = self._molar_enthalpy(y_j, T_j, 'vapor')
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
        T_condenser: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute condenser and reboiler duties including latent heat.

        The dominant energy term in distillation is the latent heat of
        vaporization/condensation. This method uses the full enthalpy
        model (sensible + latent heat via Hvap) for accurate duty
        calculations.

        For total condenser:
            Q_cond = V_top * (h_D - H_vapor_top)
            where H_vapor_top is the top stage vapor (latent heat included)
            and h_D is the condensed product: the same composition as that
            vapor, as a saturated liquid at the condenser temperature (see
            :meth:`_condenser_T`), not the top stage liquid at the top stage
            temperature.

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
            V_bot: Vapor flow leaving bottom stage (mol/s). Not used: the
                   reboiler duty comes from the overall energy balance, which
                   never needs it. Accepted so callers can pass the profile
                   they have.
            T_condenser: Condenser temperature (K). If None it is computed
                   here as the bubble point of the distillate; pass it when
                   the caller has already solved for it.

        Returns:
            Q_condenser: Condenser duty (J/s, negative = heat removed)
            Q_reboiler: Reboiler duty (J/s, positive = heat added)
        """
        p = self.params
        P = jnp.asarray(p.P)

        # Compute stage enthalpies (includes latent heat for vapor)
        h_all, H_all = self._compute_stage_enthalpies(
            x_profile, y_profile, T_profile
        )

        # Top stage enthalpies
        H_vapor_top = H_all[-1]   # Vapor enthalpy at top (includes Hvap)

        # Bottom stage enthalpies
        h_liquid_bot = h_all[0]   # Liquid enthalpy at bottom

        # Vapor flows (use provided or CMO estimates)
        V_top_flow = V_top if V_top is not None else (R + 1) * D

        # The condensed product: the top stage vapor, as a saturated liquid at
        # the condenser's own temperature.
        x_D = y_profile[-1]
        if T_condenser is None:
            T_condenser = self._condenser_T(x_D, P, T_guess=T_profile[-1])
        h_D = self._molar_enthalpy(x_D, T_condenser, 'liquid')

        # Condenser duty: condense all vapor from the top stage to liquid
        # Q_cond = V_top * (h_D - H_vapor_top) < 0 (heat removed)
        Q_condenser = V_top_flow * (h_D - H_vapor_top)

        # Feed enthalpy (saturated liquid, q=1)
        z_arr = jnp.array([z[s] for s in p.species_order])
        h_F = self._molar_enthalpy(z_arr, jnp.asarray(T_feed), 'liquid')

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
        cmo_iter: int = 30,
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
            cmo_iter: Number of CMO iterations for the warm start

        Returns:
            x: (n, nc) liquid mole fractions
            y: (n, nc) vapor mole fractions
            T: (n,) stage temperatures
            L: (n,) liquid flows leaving each stage downward (mol/s)
            V: (n,) vapor flows leaving each stage upward (mol/s)

        Example:
            12-stage heptane / ethylbenzene column at 1 atm, feed stage 6,
            equimolar feed of 0.02 mol/s, R=2.5, B_spec=0.01 mol/s.
            McCabe-Thiele targets: 97 % heptane in distillate, 99 %
            ethylbenzene in bottoms.

            CMO result  (use_mesh=False):
              distillate heptane=0.990, bottoms ethylbenzene=0.989

            MESH result (use_mesh=True, mesh_iter=20):
              distillate heptane=0.988, bottoms ethylbenzene=0.988

            The two agree closely here because both solve the same component
            balances; what MESH adds is the energy-balance L/V correction,
            which on this case pulls the rectifying-section liquid rate
            (~0.023 mol/s) a little below its CMO value (R*D = 0.025 mol/s)
            and the stripping rate (~0.042 mol/s) below L'=L+F (0.045 mol/s),
            reflecting the differing latent heats of heptane and
            ethylbenzene.  On a mixture with a wider latent-heat spread the
            correction is larger and the two paths separate.
        """
        p = self.params
        n = p.n_stages
        P = jnp.asarray(p.P)
        B = F_total - D
        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])

        # Feed flow vector: nonzero only at feed stage
        F_vec = jnp.zeros(n).at[p.feed_stage].set(F_total)

        # CMO warm start (component balances already closed)
        x_init, y_init, T_init = self._solve_constant_molar_overflow(
            feed_flows, F_total, R, D, T_feed, n_iter=cmo_iter
        )

        # Initial L/V from the CMO assumption; the energy balance corrects them.
        L_init, V_init = self._cmo_flows(F_total, R, D)

        def one_mesh_iter(carry, _):
            x, y, T, L, V = carry

            # 1-5. Component balances at the current flows, then the
            # bubble-point temperature and vapor-composition update.  Total
            # condenser: the reflux carries the top-stage vapor composition.
            x_new, y_new, T_new = self._bubble_point_step(
                y[-1], x, T, L, V, F_vec, z, R, D
            )

            # 6. Energy balance update to obtain L/V profiles
            h_all, H_all = self._compute_stage_enthalpies(x_new, y_new, T_new)

            # Feed enthalpy (saturated liquid, q=1)
            h_F = self._molar_enthalpy(z, jnp.asarray(T_feed), 'liquid')

            # Reflux enthalpy. A total condenser returns saturated liquid of
            # the distillate's composition at the condenser temperature, which
            # is below the top stage temperature -- so the reflux enters the
            # top stage subcooled, and the top stage has to boil some of it.
            # Evaluating it at the top stage temperature instead would credit
            # the balance with heat the condenser has already removed.
            T_cond = self._condenser_T(y_new[-1], P, T_guess=T_new[-1])
            h_reflux = self._molar_enthalpy(y_new[-1], T_cond, 'liquid')

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
        use_mesh: bool = True,
        mesh_iter: int = 20,
        cmo_iter: int = 30,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Solve distillation column.

        Args:
            feed: Feed stream
            R: Reflux ratio
            D_spec: Distillate flow rate specification (mol/s)
            B_spec: Bottoms flow rate specification (mol/s)
                    Exactly one of D_spec or B_spec must be given.
            use_mesh: If True (default), run the rigorous MESH solver
                      (Wang-Henke bubble-point method) after the CMO warm
                      start.  The MESH solver corrects L/V flows via energy
                      balance and gives much more accurate compositions for
                      systems with unequal latent heats.
                      If False, return the faster CMO solution, which solves
                      the same component material balances but holds the L/V
                      profiles at their constant-molar-overflow values instead
                      of correcting them with stage energy balances.  Both
                      paths close the per-species balance; check
                      ``info['balance_error']`` to see how well.
            mesh_iter: Number of MESH iterations when use_mesh=True.
            cmo_iter: Number of CMO bubble-point iterations (the warm start
                      when use_mesh=True, the whole solve when it is False).
                      Raise it if ``info['balance_error_rel']`` comes back
                      larger than the problem can tolerate.  The residual falls
                      geometrically with this count, but at a rate the column
                      sets: on a well-conditioned separation it drops several
                      decades over a few tens of iterations, while near minimum
                      reflux the rate approaches one and more iterations buy
                      almost nothing (a 12-stage binary at R = 1.2 moves only
                      1.0e-3 -> 9.5e-4 going from 30 to 100).  Read the
                      reported residual rather than assuming a count is enough.
                      Most of the MESH path's cost is this warm start, not the
                      MESH iterations -- do not trim it as an optimisation
                      without re-reading ``balance_error_rel``, because closure
                      is what it buys (0.41 ms CMO vs 0.53 ms MESH, jitted, on
                      a 20-stage ternary column).

        Returns:
            distillate: Distillate stream, at the condenser temperature --
                the bubble point of the distillate, which is below the top
                stage temperature (see :meth:`_condenser_T`)
            bottoms: Bottoms stream, at the reboiler temperature
            info: Dictionary with:
                - 'T_profile': Stage temperatures
                - 'x_profile': Liquid composition profiles
                - 'y_profile': Vapor composition profiles
                - 'T_condenser': Condenser temperature (K), = distillate T
                - 'T_reboiler': Reboiler temperature (K), = bottoms T
                - 'L_rect': Rectifying liquid flow rate (CMO) or 'L_profile'
                - 'V_rect': Rectifying vapor flow rate (CMO) or 'V_profile'
                - 'D': Distillate flow rate
                - 'B': Bottoms flow rate
                - 'balance_error': (n_species,) component balance residual
                  D x_D,i + B x_B,i - F z_i (mol/s), in species_order
                - 'balance_error_rel': max |balance_error| / F_total, a single
                  number a caller can compare against a tolerance
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
                feed_flows, F_total, R, D, T_feed,
                n_iter=mesh_iter, cmo_iter=cmo_iter,
            )

            # Total condenser: distillate composition = vapor from top stage
            x_D = {s: y_profile[-1, i] for i, s in enumerate(p.species_order)}
            x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}

            distillate_flows = {s: D * x_D[s] for s in p.species_order}
            bottoms_flows = {s: B * x_B[s] for s in p.species_order}

            # A total condenser delivers the distillate as a saturated liquid
            # at its own bubble point, which is below the top stage
            # temperature (see _condenser_T).
            T_condenser = self._condenser_T(
                y_profile[-1], jnp.asarray(p.P), T_guess=T_profile[-1]
            )

            distillate = make_stream(distillate_flows, T_condenser, p.P)
            bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

            # Compute duties with actual V profile from MESH
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
                V_top=V_profile[-1], V_bot=V_profile[0],
                T_condenser=T_condenser,
            )

            info = {
                "T_profile": T_profile,
                "x_profile": x_profile,
                "y_profile": y_profile,
                "L_profile": L_profile,
                "V_profile": V_profile,
                "D": D,
                "B": B,
                "T_condenser": T_condenser,
                "T_reboiler": T_profile[0],
                "Q_condenser": Q_condenser,
                "Q_reboiler": Q_reboiler,
            }
        else:
            # Constant molar overflow (CMO) solver: same component
            # balances as MESH, L/V frozen at their CMO values
            x_profile, y_profile, T_profile = self._solve_constant_molar_overflow(
                feed_flows, F_total, R, D, T_feed, n_iter=cmo_iter
            )

            # Total condenser: distillate = vapor leaving top stage
            x_D = {s: y_profile[-1, i] for i, s in enumerate(p.species_order)}

            # Reboiler (bottom stage, index 0)
            x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}

            distillate_flows = {s: D * x_D[s] for s in p.species_order}
            bottoms_flows = {s: B * x_B[s] for s in p.species_order}

            T_condenser = self._condenser_T(
                y_profile[-1], jnp.asarray(p.P), T_guess=T_profile[-1]
            )

            distillate = make_stream(distillate_flows, T_condenser, p.P)
            bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

            # Compute duties with CMO vapor flow estimates
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
                T_condenser=T_condenser,
            )

            info = {
                "T_profile": T_profile,
                "x_profile": x_profile,
                "y_profile": y_profile,
                "L_rect": R * D,
                "V_rect": (R + 1) * D,
                "D": D,
                "B": B,
                "T_condenser": T_condenser,
                "T_reboiler": T_profile[0],
                "Q_condenser": Q_condenser,
                "Q_reboiler": Q_reboiler,
            }

        # Component balance closure.  Both solvers satisfy the per-species
        # balances only to within their iteration count, so report the residual
        # rather than leaving a caller to discover it (issue #211).
        #
        # Taken from the reported product streams, not from the profiles the
        # solver converged: that is the number a caller can act on, and it also
        # catches anything the stream construction gets wrong between the
        # profile ends and the products.
        balance_error = jnp.array([
            distillate_flows[s] + bottoms_flows[s] - feed_flows[s]
            for s in p.species_order
        ])
        info["balance_error"] = balance_error
        info["balance_error_rel"] = jnp.max(jnp.abs(balance_error)) / jnp.maximum(
            F_total, EPS_DIVISION
        )

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
