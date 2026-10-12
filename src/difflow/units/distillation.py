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
from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
from jax import Array, lax

from difflow.streams import Stream, get_flows, make_stream
from difflow.thermo import CubicThermo, IdealThermo
from difflow.params_mixin import ParamsMixin
from difflow.cache_key import ValueKeyed
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

# Murphree efficiency homotopy (E = 1 -> target): a fixed number of steps, each
# with a fixed number of bubble-point iterations, so the loop is static.
_HOMOTOPY_STEPS = 4
_HOMOTOPY_ITERS = 10


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


def _feed_molar_enthalpy(
    thermo: "IdealThermo | CubicThermo",
    species_order: list[str],
    z: Array,
    T_feed: Array,
    q: Array,
    P: Array,
) -> Array:
    r"""Molar enthalpy (J/mol) of a feed of thermal condition ``q``.

    ``q`` is the liquid fraction of the feed, and it means the same thing in
    the energy balance as it does in the section flows: the feed arrives as
    ``q`` moles of liquid and ``1 - q`` moles of vapour, both at the feed
    stream's own temperature, so

    .. math::

        h_F = q\, h^L(z, T_F) + (1 - q)\, H^V(z, T_F)

    A saturated-vapour feed (``q = 0``) therefore carries its latent heat in
    with it, and the reboiler is not charged for it; at ``q = 1`` this is the
    plain liquid enthalpy.

    Args:
        thermo: Thermodynamic property calculator.
        species_order: Species names, in the order ``z`` is given.
        z: (nc,) feed mole fractions.
        T_feed: Feed temperature (K).
        q: Feed thermal condition, the liquid fraction of the feed
            (1 = saturated liquid, 0 = saturated vapour). Outside [0, 1] it is
            clamped for the mixture here, and only here -- see below.
        P: Pressure (Pa).

    Returns:
        Feed molar enthalpy (J/mol).

    Note:
        A subcooled feed is ``q > 1`` and a superheated one ``q < 0``, and
        both are ordinary things to model: on the CMO path ``q`` is the *only*
        place either can be said, since ``_cmo_section_rates`` is the whole
        model there and ``T_feed`` never reaches it.

        What must not happen is extrapolating this mixture past the two-phase
        range. Both enthalpies are already evaluated at the feed's own
        temperature, so at ``q = 1.3`` the honest answer is
        ``h_liquid(z, T_feed)`` -- an all-liquid feed, below its bubble point,
        with the subcooling carried by ``T_feed`` itself. Forming
        ``1.3 h_liq - 0.3 H_vap`` would subtract three tenths of a latent heat
        that is not there and count the departure from saturation twice.

        So the clamp is the physics, not a guard: it is confined to the
        enthalpy, and the section flows and Underwood's equation read the raw
        ``q``, which is what makes a subcooled feed expressible at all.
    """
    h_liq = _mixture_molar_enthalpy(thermo, species_order, z, T_feed, 'liquid', P)
    H_vap = _mixture_molar_enthalpy(thermo, species_order, z, T_feed, 'vapor', P)
    q = jnp.clip(jnp.asarray(q), 0.0, 1.0)
    return q * h_liq + (1.0 - q) * H_vap


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


class ShortcutColumn(ValueKeyed):
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
        # _solve is jitted with `self` static, so this is what decides whether
        # a second, identical column reuses the first one's executable.
        self._set_value_key(params, thermo)

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
        x_D_HK: Array | None = None,
        x_B_HK: Array | None = None,
    ) -> tuple[Array, Array]:
        """Calculate minimum number of stages using Fenske equation.

        N_min = ln[(x_D_LK/x_B_LK) * (x_B_HK/x_D_HK)] / ln(alpha_LK)

        For binary or pseudo-binary with LK recovery specification:
        N_min = ln[(x_D_LK/(1-x_D_LK)) * ((1-x_B_LK)/x_B_LK)] / ln(alpha_LK)

        Handles the singularity when α → 1:
        - When α < 1 + MIN_ALPHA_DIFF, ln(α) → 0 causing N_min → ∞
        - Uses smooth blending to cap N_min at MAX_STAGES
        - Returns a flag indicating if the system is close-boiling

        With ``x_D_HK`` and ``x_B_HK`` given this is the multicomponent
        Fenske equation, ``ln[(x_LK/x_HK)_D (x_HK/x_LK)_B] / ln(alpha_LK)``,
        which is what the column uses. The pseudo-binary form above (the
        default when the heavy-key fractions are omitted) treats every species
        that is not the light key as heavy key; with a light non-key in the
        distillate (ethane over a propane/isobutane split) it understates
        ``N_min`` -- by 16 % on the depropanizer of the unit-ops book's
        Chapter 13.

        Args:
            x_D_LK: Mole fraction of LK in distillate
            x_B_LK: Mole fraction of LK in bottoms
            alpha_LK: Relative volatility of LK vs HK
            x_D_HK: Mole fraction of HK in distillate (multicomponent form)
            x_B_HK: Mole fraction of HK in bottoms (multicomponent form)

        Returns:
            Tuple of (N_min, close_boiling_flag):
            - N_min: Minimum number of theoretical stages, capped at MAX_STAGES
            - close_boiling_flag: True if α is close to 1 (hard separation)
        """
        if x_D_HK is None or x_B_HK is None:
            # Pseudo-binary: everything that is not LK counts as HK
            x_D_HK = 1 - x_D_LK
            x_B_HK = 1 - x_B_LK
        numer = safe_log(
            safe_divide(x_D_LK, x_D_HK) *
            safe_divide(x_B_HK, x_B_LK)
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
        alpha_arr = jnp.stack([jnp.asarray(alpha[s_]) for s_ in p.species_order])
        z_arr = jnp.stack([jnp.asarray(z[s_]) for s_ in p.species_order])
        # Bracketed: the root lies between the heavy key and the next volatility
        # above it (the light key, when no species boils between the keys).
        # A Newton iteration from sqrt(alpha_LK) used to leave that interval
        # and be clipped to 1.001 in about a quarter of random feeds.
        return underwood_theta(alpha_arr, z_arr, q, alpha_HK=alpha[p.heavy_key])

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
        # Floored at zero (a non-negative reflux), not at 0.1 as it used to be:
        # an easy split has an R_min below 0.1, and the floor overstated it.
        return jnp.maximum(R_min, 0.0)

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

        # Distribute the non-keys by the Hengstebeck-Geddes equation,
        # log(d_i/b_i) = A + C log(alpha_i), the straight line in
        # (log alpha, log d/b) through both keys. alpha is relative to the
        # heavy key (alpha_HK = 1), so the heavy key fixes the intercept,
        # A = log(d/b)_HK, and the light key the slope,
        # C = [log(d/b)_LK - log(d/b)_HK] / log(alpha_LK).
        # Geddes, AIChE J. 4, 389 (1958); Hengstebeck, Distillation (1961).
        # Written from the recoveries, so a key absent from the feed still
        # gives a finite line.
        log_db_LK = jnp.log(rec_LK / (1 - rec_LK))
        log_db_HK = jnp.log((1 - rec_HK) / rec_HK)
        A = log_db_HK
        C = safe_divide(log_db_LK - log_db_HK, safe_log(alpha_LK))

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
                # d/(d+b) = sigmoid(log d/b): in [0, 1] by construction, so
                # the species balance closes exactly and needs no clip.
                F_i = feed_flows[s]
                distillate_flows[s] = F_i * jax.nn.sigmoid(A + C * safe_log(alpha[s]))
                bottoms_flows[s] = F_i - distillate_flows[s]
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
            q: Feed thermal condition: the liquid fraction of the feed
                (1 = saturated liquid, 0 = saturated vapour). It enters
                Underwood's equation and the feed enthalpy the duties come
                from. ``q > 1`` (subcooled) and ``q < 0`` (superheated) are
                allowed and reach Underwood as written; the feed enthalpy
                clamps them to [0, 1], because the subcooling is already in
                ``T_feed`` and mixing past the range would count it twice
                (see :func:`_feed_molar_enthalpy`).

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
        # The solve itself is jitted (see :meth:`_solve`); this wrapper is a
        # plain function so the catalog can still read the port names off the
        # signature -- difflow.catalog derives them with inspect.signature and
        # skips anything that is not a Python function, which a jitted method
        # is not.
        return self._solve(feed, R, P, q)

    @partial(jax.jit, static_argnums=(0,))
    def _solve(
        self,
        feed: Stream,
        R: Array | float,
        P: Array | float = 101325.0,
        q: Array | float = 1.0,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """The body of :meth:`__call__`, cached on (self, argument shapes)."""
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
        N_min, close_boiling = self.fenske_minimum_stages(
            x_D_LK, x_B_LK, alpha_LK,
            x_D_HK=x_D[p.heavy_key], x_B_HK=x_B[p.heavy_key],
        )

        # Underwood minimum reflux
        theta = self.underwood_theta(z, alpha, q)
        R_min = self.underwood_minimum_reflux(x_D, alpha, theta)

        # Gilliland actual stages
        N, near_min_reflux = self.gilliland_correlation(R, R_min, N_min)

        # Feed stage
        B_over_D = safe_divide(B_total, D_total)
        x_D_HK = x_D[p.heavy_key]
        N_feed = self.feed_stage_kirkbride(N, z_LK, z_HK, x_B_LK, x_D_HK, B_over_D)
        # The rounded feed stage has a zero derivative; report the continuous
        # Kirkbride split too, for gradients and for reading the correlation.
        N_rect, N_strip = kirkbride_feed_stage(
            N, z_LK, z_HK, x_B_LK, x_D_HK, B_total, D_total
        )

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

        # Feed enthalpy at the feed's thermal condition: q moles of liquid
        # and (1 - q) of vapour, both at the feed temperature. The same q that
        # Underwood's equation above uses, read the same way.
        h_F = _feed_molar_enthalpy(
            self.thermo, p.species_order, z_arr, T_feed, q, P
        )

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
            "N_rect": N_rect,
            "N_strip": N_strip,
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


# =============================================================================
# Column section convention
# =============================================================================
#
# Stages are numbered from the bottom: j = 0 is the reboiler and
# j = n_stages - 1 is the top stage.  The feed enters stage ``feed_stage``.
#
# Everything about where the rectifying section ends and the stripping section
# begins is decided by ``_is_rectifying_cut`` below, and the flows follow from
# it.  Both the CMO (Lewis-Matheson) sweep and the MESH flow initialisation
# read the section boundary from these functions, so the two solvers cannot
# drift apart.


def _is_rectifying_cut(j: Array | int, feed_stage: int) -> Array:
    """Is the horizontal cut between stage ``j`` and stage ``j+1`` rectifying?

    A cut is rectifying when the envelope above it -- stages ``j+1`` through
    the condenser -- contains no feed, so a balance over that envelope closes
    on the distillate alone.  The feed enters stage ``feed_stage``, which sits
    above the cut only when ``j < feed_stage``:

        cut j is rectifying  <=>  j >= feed_stage

    The cut at ``j = feed_stage`` is therefore rectifying: it passes just below
    the feed stage's vapour outlet and just above its liquid outlet.

    Args:
        j: Cut index (the cut above stage ``j``); may be an array.
        feed_stage: Index of the stage the feed enters (0 = reboiler).

    Returns:
        Boolean array, True where the cut is in the rectifying section.
    """
    return jnp.asarray(j) >= feed_stage


def _cmo_section_rates(
    R: Array,
    D: Array,
    F_total: Array,
    q: Array,
) -> tuple[Array, Array, Array, Array]:
    """Constant-molar-overflow liquid and vapour rates in each section.

    The feed splits into ``q F`` of liquid joining the down-flowing liquid and
    ``(1 - q) F`` of vapour joining the up-flowing vapour:

        L_rect = R D,          V_rect = (R + 1) D
        L_strip = L_rect + q F, V_strip = V_rect - (1 - q) F

    Args:
        R: Reflux ratio.
        D: Distillate flow rate (mol/s).
        F_total: Total feed flow rate (mol/s).
        q: Feed thermal condition (1 = saturated liquid, 0 = saturated vapour).

    Returns:
        (L_rect, L_strip, V_rect, V_strip) in mol/s.
    """
    L_rect = R * D
    V_rect = (R + 1) * D
    L_strip = L_rect + q * F_total
    V_strip = V_rect - (1.0 - q) * F_total
    return L_rect, L_strip, V_rect, V_strip


def _cmo_section_flows(
    j: Array | int,
    feed_stage: int,
    L_rect: Array,
    L_strip: Array,
    V_rect: Array,
    V_strip: Array,
) -> tuple[Array, Array]:
    """Constant-molar-overflow flows leaving stage ``j``.

    Both flows are read off ``_is_rectifying_cut`` -- the liquid leaving stage
    ``j`` downward crosses the cut *below* it (cut ``j-1``), the vapour leaving
    it upward crosses the cut *above* it (cut ``j``):

        L[j] = L_rect  if j > feed_stage  else L_strip
        V[j] = V_rect  if j >= feed_stage else V_strip

    The feed stage is thus counted with the stripping section on its liquid
    side and with the rectifying section on its vapour side.  That is one
    convention, not two: the feed splits into ``q F`` of liquid running down
    and ``(1 - q) F`` of vapour running up, so
    ``L[feed_stage] = L_rect + q F = L_strip`` while
    ``V[feed_stage] = V_strip + (1 - q) F = V_rect``.

    Args:
        j: Stage index (0 = reboiler); may be an array.
        feed_stage: Index of the stage the feed enters.
        L_rect: Liquid rate in the rectifying section (mol/s).
        L_strip: Liquid rate in the stripping section (mol/s).
        V_rect: Vapour rate in the rectifying section (mol/s).
        V_strip: Vapour rate in the stripping section (mol/s).

    Returns:
        (L_j, V_j): Liquid leaving stage ``j`` downward and vapour leaving it
        upward (mol/s). Boundary stages (reboiler, condenser) are not special
        cased here; callers that need the product flows override them.
    """
    L_j = jnp.where(_is_rectifying_cut(jnp.asarray(j) - 1, feed_stage), L_rect, L_strip)
    V_j = jnp.where(_is_rectifying_cut(j, feed_stage), V_rect, V_strip)
    return L_j, V_j


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
            and stages above it the rectifying section. The feed stage itself
            is split by the feed: the liquid leaving it downward carries
            ``q F``, so it is a stripping-section flow
            (``L[feed_stage] = L_rect + q F``), while the vapour leaving it
            upward carries ``(1 - q) F``, a rectifying-section flow
            (``V[feed_stage] = V_rect``). ``_is_rectifying_cut`` is the single
            definition; both the CMO sweep and the MESH flow initialisation
            read the boundary from it.
        condenser_type: ``'total'`` or ``'partial'``. A total condenser
            condenses all of the top tray's vapour: the distillate is a
            saturated liquid with the composition of that vapour. A partial
            condenser is an **extra equilibrium stage outside** ``n_stages``:
            its vapour is the distillate (``y_D = K(T_c) x_R``), its liquid is
            the reflux ``x_R``, and ``T_c`` is the dew point of ``y_D``. The
            profiles in ``info`` keep their ``n_stages`` rows (so nothing
            shifts when the condenser type changes); the condenser is
            reported separately and ``info['n_equilibrium_stages']`` counts
            it. ``'mixed'`` (vapour plus liquid distillate) is not
            implemented and is rejected.
        P: Column pressure (Pa). One pressure for the whole column -- there is
            no tray pressure drop.
        q: Feed thermal condition: the liquid fraction of the feed
            (1.0 = saturated liquid, 0.0 = saturated vapor). It sets the
            section flows above on both solver paths, and it is also what the
            energy balance forms the feed enthalpy from,
            ``q h_liquid(z, T_feed) + (1 - q) H_vapor(z, T_feed)``, so on the
            MESH path (``use_mesh=True``) the converged flows and duties
            follow it too -- a saturated-vapour feed arrives with its latent
            heat and does not charge the reboiler for it. ``q > 1`` is a
            subcooled feed and ``q < 0`` a superheated one; both are allowed,
            and on the CMO path ``q`` is the only place either can be said,
            since ``T_feed`` does not reach the section flows. The feed
            enthalpy clamps the mixture to [0, 1] --- the departure from
            saturation is carried by ``T_feed`` and must not be counted a
            second time (see :func:`_feed_molar_enthalpy`).
        stage_efficiency: Murphree **vapour** efficiency, a scalar or an
            ``(n_stages,)`` array (same name and 0-1 convention as
            ``CascadeParams.stage_efficiency`` in :mod:`difflow.units.lle`).
            ``y_j = y_{j-1} + E_j (K_j x_j - y_{j-1})`` with ``y_{j-1}`` the
            actual vapour entering stage ``j``. The reboiler (stage 0) and the
            partial condenser are equilibrium stages and always use 1,
            whatever is given for stage 0. ``1.0`` (default) is the
            equilibrium-stage column, and reproduces the results this class
            gave before the option existed. Differentiable in ``E``. Must be
            non-negative.
    """
    species_order: list[str]
    n_stages: int
    feed_stage: int
    condenser_type: Literal["total", "partial"] = "total"
    P: float = 101325.0
    q: float = 1.0  # Feed thermal condition (1.0 = saturated liquid, 0.0 = saturated vapor)
    stage_efficiency: float | Array = 1.0  # Murphree vapour efficiency (scalar or (n_stages,))

    def __post_init__(self):
        if self.condenser_type == "mixed":
            raise NotImplementedError(
                "condenser_type='mixed' (vapour plus liquid distillate) is not "
                "implemented; use 'total' or 'partial'."
            )
        if self.condenser_type not in ("total", "partial"):
            raise ValueError(
                f"condenser_type must be 'total' or 'partial', got "
                f"{self.condenser_type!r}"
            )
        try:
            eff = np.asarray(self.stage_efficiency, dtype=float)
        except Exception:        # a tracer: values are checked by the caller
            return
        if eff.ndim > 1 or (eff.ndim == 1 and eff.shape[0] != self.n_stages):
            raise ValueError(
                "stage_efficiency must be a scalar or have shape "
                f"(n_stages,) = ({self.n_stages},), got {eff.shape}"
            )
        if np.any(eff < 0.0):
            raise ValueError("stage_efficiency must be non-negative")


class DistillationColumn(ValueKeyed):
    """Rigorous stage-by-stage distillation column.

    Solves MESH equations (Material, Equilibrium, Summation, Heat balance)
    for each stage using the bubble-point method.

    Stage numbering: zero-based from the bottom -- stage 0 is the reboiler,
    stage ``n_stages - 1`` is the top tray. The total condenser is outside the
    cascade, so the distillate is not a stage's product (see
    :meth:`_condenser_T`). The feed enters stage ``params.feed_stage``;
    ``_is_rectifying_cut`` says which section each stage's flows belong to.

    Assumptions:
    - Equilibrium stages, or Murphree vapour efficiency ``stage_efficiency``
    - No pressure drop between stages
    - Total condenser, or a partial condenser (an extra equilibrium stage
      outside ``n_stages`` with a vapour distillate)
    - Constant molar overflow (CMO) for the initial solution

    All calculations are JAX-compatible for automatic differentiation.
    """

    symbol = "MESH Column"
    equations = [
        r"\text{M: } L_{n+1} x_{i,n+1} + V_{n-1} y_{i,n-1} + F_n z_{i,n} - (L_n + U_n) x_{i,n} - (V_n + W_n) y_{i,n} = 0",
        r"\text{E: } y_{i,n} = K_i(T_n, P)\, x_{i,n}",
        r"\text{S: } \sum_i x_{i,n} = 1,\quad \sum_i y_{i,n} = 1",
        r"\text{H: } L_{n+1} h_{n+1} + V_{n-1} H_{n-1} + F_n h_{F,n} - (L_n + U_n) h_n - (V_n + W_n) H_n \pm Q_n = 0",
        r"\text{Murphree: } y_{i,n} = y_{i,n-1} + E_{MV,n}\,(K_i x_{i,n} - y_{i,n-1})",
        r"\text{Partial condenser: } (R+1) D\, y_{i,N} = D\, y_{D,i} + R D\, x_{R,i},\quad y_{D,i} = K_i(T_c) x_{R,i}",
    ]
    assumptions = [
        "Equilibrium stages, or Murphree vapour efficiency stage_efficiency (default 1; the reboiler and a partial condenser are always equilibrium stages).",
        "Total condenser (distillate = saturated liquid of the top-tray vapour) or partial condenser (an extra equilibrium stage outside n_stages: vapour distillate, liquid reflux, T_c = dew point of the distillate).",
        "Constant molar overflow for initialization.",
        "No pressure drop between stages.",
    ]
    references = [
        "King, C.J. Separation Processes, 2e, McGraw-Hill, 1980.",
        "Kister, H.Z. Distillation Design, McGraw-Hill, 1992.",
        "Seader, Henley, Roper. Separation Process Principles, 3e.",
        "Murphree, E.V. Ind. Eng. Chem. 17 (1925) 747 (vapour efficiency).",
        "Lewis, W.K. Ind. Eng. Chem. 14 (1922) 492 (overall vs Murphree efficiency).",
    ]
    parameter_symbols = {
        "n_stages": "N",
        "feed_stage": "n_F",
        "P": "P",
        "q": "q",
        "stage_efficiency": "E_{MV}",
    }
    parameter_units = {
        "n_stages": "-",
        "feed_stage": "-",
        "P": "Pa",
        "q": "-",
        "stage_efficiency": "-",
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
                a light-hydrocarbon column near its critical region needs,
                where the vapor is far from ideal. The column code
                is the same either way; the composition dependence of the EOS
                K-values rides along inside the existing MESH iteration.
                An :class:`~difflow.thermo.ActivityThermo` gives
                modified-Raoult K-values (``K_i = gamma_i P_i^sat / P``) for a
                nonideal liquid such as ethanol-water, with the ideal
                enthalpies of the :class:`IdealThermo` it wraps.
        """
        self.params = params
        self.thermo = thermo
        self.n_species = len(params.species_order)
        self._set_value_key(params, thermo)

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
        rates = _cmo_section_rates(R, D, F_total, jnp.asarray(p.q))
        L, V = _cmo_section_flows(jnp.arange(p.n_stages), p.feed_stage, *rates)
        # Liquid leaving the reboiler is the bottoms product, not a section
        # rate.  The top stage's vapour needs no such fixup: it is already
        # V_rect, the whole of it condensed into reflux plus distillate.
        return L.at[0].set(F_total - D), V

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

    # ------------------------------------------------------------------
    # Murphree efficiency and partial condenser
    # ------------------------------------------------------------------

    def _murphree(self) -> tuple[Array, bool]:
        """Per-stage Murphree vapour efficiency and whether it is all 1.

        Returns:
            (E, is_unity): ``E`` has shape ``(n_stages,)`` with the reboiler
            forced to 1 (it is an equilibrium stage). ``is_unity`` is a
            *Python* bool, true only when the efficiency is concrete and equal
            to 1 on every stage; a traced efficiency (a gradient with respect
            to it) is never unity, so it takes the general path. Static, so it
            selects a code path without branching on a traced value.
        """
        p = self.params
        n = p.n_stages
        eff = p.stage_efficiency
        try:
            unity = bool(np.all(np.asarray(eff, dtype=float) == 1.0))
        except Exception:        # tracer
            unity = False
        E = jnp.broadcast_to(jnp.asarray(eff, dtype=float), (n,))
        return E.at[0].set(1.0), unity

    @staticmethod
    def _inv2(a: Array, b: Array, c: Array, d: Array):
        """Inverse of the 2x2 matrix ``[[a, b], [c, d]]`` (entries, any batch)."""
        det = a * d - b * c
        det = jnp.where(jnp.abs(det) > 1e-30, det, jnp.sign(det + 1e-60) * 1e-30)
        return d / det, -b / det, -c / det, a / det

    def _general_step(
        self,
        x: Array,
        T: Array,
        L: Array,
        V: Array,
        F_vec: Array,
        z: Array,
        reflux_term: Array,
        E: Array,
    ) -> tuple[Array, Array, Array]:
        """Wang-Henke step with Murphree vapour efficiency, any stage count.

        Works on ``m`` stages (the trays and reboiler, plus the partial
        condenser as the last row when there is one). The unknowns on stage
        ``j`` are the pair ``(x_j, y_j)`` and the two equations are the
        component balance and the Murphree relation

            L_{j+1} x_{j+1} + V_{j-1} y_{j-1} + F_j z - L_j x_j - V_j y_j = 0
            y_j - (1 - E_j) y_{j-1} - E_j K_j x_j = 0

        a block-tridiagonal system with 2x2 blocks, solved for every species
        at once by block Thomas elimination. Eliminating ``y`` instead would
        divide by ``E`` and fail at ``E = 0``; the block form does not.
        Because the balances are solved exactly, summing them telescopes to
        ``D y_D + B x_B = F z`` just as in the equilibrium-stage step.

        Args:
            x: (m, nc) current liquid compositions (K-values are evaluated at
                these and at ``T``, lagged one iteration).
            T: (m,) stage temperatures (K).
            L: (m,) liquid flow leaving each stage downward (mol/s); the last
                entry is the reflux when the last row is a partial condenser,
                and ignored for a total condenser.
            V: (m,) vapour flow leaving each stage upward (mol/s).
            F_vec: (m,) feed entering each stage (mol/s).
            z: (nc,) feed mole fractions.
            reflux_term: (nc,) component flow entering the top row from outside
                (``R D x_D`` for a total condenser, zeros for a partial one).
            E: (m,) Murphree efficiencies (stage 0 and a condenser row are 1).

        Returns:
            (x, y, T): updated liquid compositions, actual vapour compositions
            (after the Murphree recursion) and temperatures.
        """
        p = self.params
        m = x.shape[0]
        nc = self.n_species
        P = jnp.asarray(p.P)

        _, K = lax.scan(
            lambda _, a: (None, self.thermo.K_values_array(a[1], P, a[0])),
            None, (x, T),
        )  # (m, nc)

        V_prev = jnp.concatenate([jnp.zeros(1), V[:-1]])
        L_next = jnp.concatenate([L[1:], jnp.zeros(1)])
        rhs1 = -F_vec[:, None] * z[None, :]
        rhs1 = rhs1.at[-1].add(-reflux_term)

        def fwd(carry, inp):
            G1, G2, gy = carry                       # Gamma_{j-1} column, g_y
            L_j, V_j, Vp, Lc, E_j, K_j, r1 = inp
            M11 = -L_j - Vp * G2
            M12 = -V_j * jnp.ones_like(G2)
            M21 = -E_j * K_j + (1.0 - E_j) * G2
            M22 = jnp.ones_like(G2)
            i11, i12, i21, i22 = self._inv2(M11, M12, M21, M22)
            s1 = r1 - Vp * gy
            s2 = (1.0 - E_j) * gy
            gx_new = i11 * s1 + i12 * s2
            gy_new = i21 * s1 + i22 * s2
            G1n = i11 * Lc
            G2n = i21 * Lc
            return (G1n, G2n, gy_new), (G1n, gx_new)

        zero = jnp.zeros(nc)
        _, (G1, gx) = lax.scan(
            fwd, (zero, zero, zero),
            (L, V, V_prev, L_next, E, K, rhs1),
        )

        def back(x_next, inp):
            G1_j, gx_j = inp
            x_j = gx_j - G1_j * x_next
            return x_j, x_j

        _, x_rev = lax.scan(back, zero, (G1[::-1], gx[::-1]))
        x_new = x_rev[::-1]

        x_new = jnp.maximum(x_new, 1e-10)
        x_new = x_new / jnp.sum(x_new, axis=1, keepdims=True)

        _, T_new = lax.scan(
            lambda _, a: (None, self._bubble_point_T(a[0], P, a[1])[0]),
            None, (x_new, T),
        )
        _, K_new = lax.scan(
            lambda _, a: (None, self.thermo.K_values_array(a[1], P, a[0])),
            None, (x_new, T_new),
        )
        y_eq = K_new * x_new
        y_eq = y_eq / jnp.maximum(jnp.sum(y_eq, axis=1, keepdims=True), 1e-10)

        # Murphree recursion from the reboiler up: the vapour entering a stage
        # is the actual vapour that left the one below.
        def vapour(y_prev, inp):
            ye, E_j = inp
            y_j = (1.0 - E_j) * y_prev + E_j * ye
            return y_j, y_j

        _, y_rest = lax.scan(vapour, y_eq[0], (y_eq[1:], E[1:]))
        y_new = jnp.concatenate([y_eq[0][None], y_rest], axis=0)
        return x_new, y_new, T_new

    def _step(
        self,
        x: Array,
        y: Array,
        T: Array,
        L: Array,
        V: Array,
        F_vec: Array,
        z: Array,
        R: Array,
        D: Array,
        E: Array,
        legacy: bool,
    ) -> tuple[Array, Array, Array]:
        """One bubble-point step for whichever column this is.

        ``legacy`` (static) is the total-condenser, all-equilibrium column and
        calls :meth:`_bubble_point_step` untouched, so the default column is
        exactly what it was before efficiency and the partial condenser
        existed. Otherwise the general block step runs on the trays plus, for
        a partial condenser, one more equilibrium row.

        Args:
            x, y, T: current compositions and temperatures, ``m`` rows
                (``m = n_stages``, or ``n_stages + 1`` with a partial condenser).
            L, V, F_vec: (n_stages,) flow profiles (the condenser row is added
                here).
            z: (nc,) feed composition.
            R: Reflux ratio.
            D: Distillate flow (mol/s).
            E: (n_stages,) Murphree efficiencies.
            legacy: Use the original equilibrium-stage step.

        Returns:
            (x, y, T) updated, ``m`` rows each.
        """
        if legacy:
            return self._bubble_point_step(y[-1], x, T, L, V, F_vec, z, R, D)
        if self.params.condenser_type == "partial":
            one = jnp.ones(1)
            Lx = jnp.concatenate([L, (R * D)[None]])
            Vx = jnp.concatenate([V, jnp.asarray(D)[None]])
            Fx = jnp.concatenate([F_vec, jnp.zeros(1)])
            Ex = jnp.concatenate([E, one])
            reflux = jnp.zeros(self.n_species)
        else:
            Lx, Vx, Fx, Ex = L, V, F_vec, E
            reflux = R * D * y[-1]
        return self._general_step(x, T, Lx, Vx, Fx, z, reflux, Ex)

    def _separation_measure(
        self, z: Array, T_ref: Array, x_D: Array, x_B: Array
    ) -> Array:
        """ln of the key separation factor between the light and heavy end.

        The keys are the most and the least volatile species at ``T_ref`` (the
        feed bubble point): ``S = (x_D,a / x_D,b) / (x_B,a / x_B,b)``. A
        Fenske-style measure of how much separation a column delivers; the
        ratio of two of them is :meth:`_cmo_full`'s efficiency equivalent.
        """
        P = jnp.asarray(self.params.P)
        K = self.thermo.K_values_array(T_ref, P, z)
        a = jnp.argmax(K)
        b = jnp.argmin(K)
        tiny = 1e-300
        num = jnp.log(jnp.maximum(x_D[a], tiny)) - jnp.log(jnp.maximum(x_D[b], tiny))
        den = jnp.log(jnp.maximum(x_B[a], tiny)) - jnp.log(jnp.maximum(x_B[b], tiny))
        return num - den

    def _cmo_full(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        R: Array,
        D: Array,
        T_feed: Array,
        n_iter: int = 30,
    ) -> tuple[Array, Array, Array, Array]:
        """CMO solve returning every row, partial condenser included.

        Same iteration as :meth:`_solve_constant_molar_overflow`, which wraps
        this and returns the ``n_stages`` tray rows only. With a partial
        condenser there are ``n_stages + 1`` rows and the last one is the
        condenser (``x`` the reflux, ``y`` the distillate vapour, ``T`` its
        temperature).

        With a Murphree efficiency below 1 the iteration is a homotopy in
        ``E``: converge at ``E = 1``, then step ``E`` to its target in a
        fixed number of steps (``_HOMOTOPY_STEPS``), then iterate at the
        target. Fixed counts, so nothing branches on a traced value.

        Returns:
            (x, y, T, eff): rows as above and ``eff``, the ratio of the key
            separation factor's logarithm to that of the same column at
            ``E = 1`` (1 when ``E`` is 1). A Fenske-style *proxy* for the
            equivalent-theoretical-stage over actual-stage ratio.
        """
        p = self.params
        n = p.n_stages
        partial_cond = p.condenser_type == "partial"
        m = n + (1 if partial_cond else 0)
        P = jnp.asarray(p.P)
        E, unity = self._murphree()
        legacy = unity and not partial_cond

        B = F_total - D
        z = jnp.array([feed_flows[s] / F_total for s in p.species_order])
        F_vec = jnp.zeros(n).at[p.feed_stage].set(F_total)
        L, V = self._cmo_flows(F_total, R, D)

        T_bp_feed, y_flash = self._bubble_point_T(z, P, T_guess=jnp.asarray(T_feed))
        x_D_init = jnp.clip(y_flash, 1e-6, 1.0 - 1e-6)
        x_D_init = x_D_init / jnp.sum(x_D_init)
        x_B_init = (F_total * z - D * x_D_init) / jnp.maximum(B, 1e-10)
        x_B_init = jnp.clip(x_B_init, 1e-6, 1.0 - 1e-6)
        x_B_init = x_B_init / jnp.sum(x_B_init)

        frac = jnp.linspace(0.0, 1.0, m)
        x_init = frac[:, None] * x_D_init[None, :] + (1.0 - frac[:, None]) * x_B_init[None, :]
        T_init = jnp.full(m, T_bp_feed)

        def init_y(_, x_j):
            K_j = self.thermo.K_values_array(T_bp_feed, P, x_j)
            yx_j = K_j * x_j
            return None, yx_j / jnp.maximum(jnp.sum(yx_j), 1e-10)

        _, y_init = lax.scan(init_y, None, x_init)

        def run(carry, E_k, n_it):
            def one_iteration(c, _):
                return self._step(*c, L, V, F_vec, z, R, D, E_k, legacy), None
            c, _ = lax.scan(one_iteration, carry, None, length=n_it)
            return c

        carry = (x_init, y_init, T_init)
        if legacy:
            x_f, y_f, T_f = run(carry, E, n_iter)
            return x_f, y_f, T_f, jnp.asarray(1.0)

        ones = jnp.ones(n)
        carry = run(carry, ones, n_iter)
        ref = carry
        if not unity:
            for k in range(1, _HOMOTOPY_STEPS + 1):
                E_k = 1.0 + (E - 1.0) * (k / _HOMOTOPY_STEPS)
                carry = run(carry, E_k, _HOMOTOPY_ITERS)
            carry = run(carry, E, n_iter)

        def products(c):
            x_c, y_c, _ = c
            top = y_c[n] if partial_cond else y_c[-1]
            return top, x_c[0]

        if unity:
            eff = jnp.asarray(1.0)
        else:
            xD_a, xB_a = products(carry)
            xD_r, xB_r = products(ref)
            s_act = self._separation_measure(z, T_bp_feed, xD_a, xB_a)
            s_ref = self._separation_measure(z, T_bp_feed, xD_r, xB_r)
            eff = s_act / jnp.where(jnp.abs(s_ref) > 1e-12, s_ref, 1e-12)
        x_f, y_f, T_f = carry
        return x_f, y_f, T_f, eff

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
        With a total condenser the reflux and distillate are the vapour of the
        top stage (x_D = y_top). With a partial condenser see
        :meth:`_cmo_full`, which returns the condenser row too; this method
        returns the ``n_stages`` tray rows only.

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
        n = self.params.n_stages
        x, y, T, _ = self._cmo_full(feed_flows, F_total, R, D, T_feed, n_iter)
        return x[:n], y[:n], T[:n]

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

    def _feed_enthalpy(
        self,
        z: Array,
        T_feed: Array,
    ) -> Array:
        """Molar enthalpy (J/mol) of the feed at this column's ``q``.

        A thin wrapper on :func:`_feed_molar_enthalpy`, bound to this column's
        species order, pressure and feed thermal condition. Every energy
        balance that sees the feed -- the MESH stage sweep and the overall
        balance the duties come from -- goes through here, so ``q`` cannot
        mean one thing in one of them and something else in the other.

        Args:
            z: (nc,) feed mole fractions, in species order.
            T_feed: Feed temperature (K).

        Returns:
            Feed molar enthalpy (J/mol).
        """
        p = self.params
        return _feed_molar_enthalpy(
            self.thermo, p.species_order, z, T_feed,
            jnp.asarray(p.q), jnp.asarray(p.P),
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
            where h_F is the feed at the column's thermal condition
            ``params.q`` (see :meth:`_feed_enthalpy`), so a vaporised feed
            brings its own latent heat and the reboiler is not charged for it.

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

        # Feed enthalpy at the column's thermal condition p.q
        z_arr = jnp.array([z[s] for s in p.species_order])
        h_F = self._feed_enthalpy(z_arr, jnp.asarray(T_feed))

        # Bottoms enthalpy (liquid at bottom T)
        h_B = h_liquid_bot

        # Overall energy balance: F*h_F + Q_reb = D*h_D + B*h_B + Q_cond
        # Q_reb = D*h_D + B*h_B + |Q_cond| - F*h_F
        #       = D*h_D + B*h_B - Q_cond - F*h_F  (since Q_cond < 0)
        Q_reboiler = D * h_D + B * h_B - Q_condenser - F_total * h_F

        return Q_condenser, Q_reboiler

    def _compute_duties_partial(
        self,
        x_profile: Array,
        y_profile: Array,
        T_profile: Array,
        y_D: Array,
        x_R: Array,
        T_condenser: Array,
        F_total: Array,
        z: dict[str, Array],
        T_feed: Array,
        R: Array,
        D: Array,
        B: Array,
        V_top: Array | None = None,
    ) -> tuple[Array, Array]:
        """Condenser and reboiler duties with a partial condenser.

        The vapour ``V_top = (R + 1) D`` leaves the top tray with its actual
        composition and splits in the condenser stage into the vapour
        distillate ``D`` (``y_D``, enthalpy ``H_D``) and the reflux ``R D``
        (``x_R``, saturated liquid ``h_R``, both at ``T_c``):

            Q_cond = D H_D + R D h_R - V_top H_top          (< 0)
            Q_reb  = D H_D + B h_B - Q_cond - F h_F

        the second from the overall balance, with ``h_F`` at the column's
        ``q`` as in :meth:`_compute_duties`. The condenser duty is "on the
        partial basis": it removes the latent heat of the reflux only, not of
        the vapour distillate.

        Returns:
            (Q_condenser, Q_reboiler) in J/s.
        """
        p = self.params
        h_all, H_all = self._compute_stage_enthalpies(x_profile, y_profile, T_profile)
        H_top = H_all[-1]
        h_B = h_all[0]
        H_D = self._molar_enthalpy(y_D, T_condenser, 'vapor')
        h_R = self._molar_enthalpy(x_R, T_condenser, 'liquid')
        V_top_flow = V_top if V_top is not None else (R + 1) * D
        Q_condenser = D * H_D + R * D * h_R - V_top_flow * H_top
        z_arr = jnp.array([z[s] for s in p.species_order])
        h_F = self._feed_enthalpy(z_arr, jnp.asarray(T_feed))
        Q_reboiler = D * H_D + B * h_B - Q_condenser - F_total * h_F
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
        """MESH solve; the ``n_stages`` tray rows only (see :meth:`_mesh_full`)."""
        n = self.params.n_stages
        x, y, T, L, V, _ = self._mesh_full(
            feed_flows, F_total, R, D, T_feed, n_iter=n_iter, cmo_iter=cmo_iter
        )
        return x[:n], y[:n], T[:n], L, V

    def _mesh_full(
        self,
        feed_flows: dict[str, Array],
        F_total: Array,
        R: Array,
        D: Array,
        T_feed: Array,
        n_iter: int = 20,
        cmo_iter: int = 30,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
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
            x: (m, nc) liquid mole fractions; ``m = n_stages``, plus one row
                for a partial condenser (its reflux)
            y: (m, nc) vapor mole fractions (the condenser row is the
                distillate vapour)
            T: (m,) stage temperatures
            L: (n,) liquid flows leaving each stage downward (mol/s)
            V: (n,) vapor flows leaving each stage upward (mol/s)
            eff: Murphree efficiency equivalent (see :meth:`_cmo_full`)

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

        E, unity = self._murphree()
        partial_cond = p.condenser_type == "partial"
        legacy = unity and not partial_cond

        # CMO warm start (component balances already closed); with an
        # efficiency below 1 it already went through the E = 1 -> E homotopy.
        x_init, y_init, T_init, eff = self._cmo_full(
            feed_flows, F_total, R, D, T_feed, n_iter=cmo_iter
        )

        # Initial L/V from the CMO assumption; the energy balance corrects them.
        L_init, V_init = self._cmo_flows(F_total, R, D)

        def one_mesh_iter(carry, _):
            x, y, T, L, V = carry

            # 1-5. Component balances at the current flows, then the
            # bubble-point temperature and vapor-composition update.  Total
            # condenser: the reflux carries the top-stage vapor composition.
            x_new, y_new, T_new = self._step(
                x, y, T, L, V, F_vec, z, R, D, E, legacy
            )

            # 6. Energy balance update to obtain L/V profiles. The sweep runs
            # over the trays only; a partial condenser's row (index n) just
            # supplies the reflux enthalpy below. The vapour compositions are
            # the actual (Murphree) ones.
            h_all, H_all = self._compute_stage_enthalpies(x_new, y_new, T_new)
            h_cond = h_all[n:]
            h_all = h_all[:n]
            H_all = H_all[:n]

            # Feed enthalpy at the column's thermal condition: q moles of
            # liquid and (1 - q) of vapour, both at the feed temperature. A
            # saturated-vapour feed brings its latent heat in with it, so the
            # reboiler is not charged for it.
            h_F = self._feed_enthalpy(z, jnp.asarray(T_feed))

            # Reflux enthalpy. A total condenser returns saturated liquid of
            # the distillate's composition at the condenser temperature, which
            # is below the top stage temperature -- so the reflux enters the
            # top stage subcooled, and the top stage has to boil some of it.
            # Evaluating it at the top stage temperature instead would credit
            # the balance with heat the condenser has already removed.
            if partial_cond:
                # Partial condenser: the reflux is the condenser stage's own
                # liquid, at the condenser temperature.
                h_reflux = h_cond[0]
            else:
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
        return x_f, y_f, T_f, L_f, V_f, eff

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
                for a total condenser the bubble point of the distillate,
                which is below the top stage temperature (see
                :meth:`_condenser_T`); for a partial condenser the vapour
                distillate at its dew point
            bottoms: Bottoms stream, at the reboiler temperature
            info: Dictionary with:
                - 'T_profile': Stage temperatures
                - 'x_profile': Liquid composition profiles
                - 'y_profile': Vapor composition profiles
                - 'T_condenser': Condenser temperature (K), = distillate T
                - 'T_reboiler': Reboiler temperature (K), = bottoms T
                - 'L_profile': (n,) liquid flows leaving each stage (mol/s);
                  on the CMO path the constant-molar-overflow profile
                - 'V_profile': (n,) vapor flows leaving each stage (mol/s)
                - 'L_rect', 'V_rect': rectifying liquid and vapor flows (CMO
                  path only)
                - 'D': Distillate flow rate
                - 'B': Bottoms flow rate
                - 'balance_error': (n_species,) component balance residual
                  D x_D,i + B x_B,i - F z_i (mol/s), in species_order
                - 'balance_error_rel': max |balance_error| / F_total, a single
                  number a caller can compare against a tolerance
                - 'n_equilibrium_stages': ``n_stages``, plus one for a partial
                  condenser
                - 'overall_efficiency_equiv': ratio of the logarithm of the
                  light/heavy key separation factor (keys = most and least
                  volatile species at the feed bubble point) to that of the
                  same column at ``E = 1``. 1 at ``E = 1``. A Fenske-style
                  proxy for equivalent theoretical stages / actual stages (for
                  comparison with O'Connell), not that ratio itself. On the
                  MESH path it is the CMO warm start's value.
                With a partial condenser (profiles still have ``n_stages``
                rows) also:
                - 'y_D': distillate vapour composition (the condenser's vapour)
                - 'x_R': reflux composition (the condenser's liquid)
                  ``T_condenser`` is then the dew point of ``y_D`` and
                  ``Q_condenser`` the duty on the partial basis (reflux latent
                  heat only). The distillate stream is that vapour at
                  ``T_condenser``.
        """
        # Jitted body, plain-function wrapper: see :meth:`ShortcutColumn._solve`.
        # By keyword, not positionally: `static_argnames` only marks an
        # argument static when it arrives as a keyword, and a traced
        # `use_mesh` fails on the first `if` that reads it.
        return self._solve(feed, R, D_spec, B_spec, use_mesh=use_mesh,
                           mesh_iter=mesh_iter, cmo_iter=cmo_iter)

    @partial(jax.jit, static_argnums=(0,),
             static_argnames=("use_mesh", "mesh_iter", "cmo_iter"))
    def _solve(
        self,
        feed: Stream,
        R: Array | float,
        D_spec: Array | float | None = None,
        B_spec: Array | float | None = None,
        use_mesh: bool = True,
        mesh_iter: int = 20,
        cmo_iter: int = 30,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """The body of :meth:`__call__`, cached on (self, argument shapes)."""
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

        n = p.n_stages
        partial_cond = p.condenser_type == "partial"
        P_col = jnp.asarray(p.P)

        if use_mesh:
            # Rigorous MESH solver (Wang-Henke bubble-point method)
            x_full, y_full, T_full, L_profile, V_profile, eff = self._mesh_full(
                feed_flows, F_total, R, D, T_feed,
                n_iter=mesh_iter, cmo_iter=cmo_iter,
            )
        else:
            # Constant molar overflow (CMO) solver: same component
            # balances as MESH, L/V frozen at their CMO values
            x_full, y_full, T_full, eff = self._cmo_full(
                feed_flows, F_total, R, D, T_feed, n_iter=cmo_iter
            )

        # The profiles in info are the n_stages tray rows. A partial
        # condenser is one more row (index n), reported on its own.
        x_profile, y_profile, T_profile = x_full[:n], y_full[:n], T_full[:n]

        if partial_cond:
            # Distillate = the condenser stage's vapour, at its temperature
            # (the dew point of that vapour); reflux = its liquid.
            y_D = y_full[n]
            x_R = x_full[n]
            T_condenser = T_full[n]
        else:
            # Total condenser: distillate composition = vapor from top stage,
            # delivered as a saturated liquid at its own bubble point, which
            # is below the top stage temperature (see _condenser_T).
            y_D = y_profile[-1]
            T_condenser = self._condenser_T(
                y_profile[-1], P_col, T_guess=T_profile[-1]
            )

        x_D = {s: y_D[i] for i, s in enumerate(p.species_order)}
        x_B = {s: x_profile[0, i] for i, s in enumerate(p.species_order)}
        distillate_flows = {s: D * x_D[s] for s in p.species_order}
        bottoms_flows = {s: B * x_B[s] for s in p.species_order}

        distillate = make_stream(distillate_flows, T_condenser, p.P)
        bottoms = make_stream(bottoms_flows, T_profile[0], p.P)

        V_top = V_profile[-1] if use_mesh else None
        if partial_cond:
            Q_condenser, Q_reboiler = self._compute_duties_partial(
                x_profile, y_profile, T_profile, y_D, x_R, T_condenser,
                F_total, z, T_feed, R, D, B, V_top=V_top,
            )
        elif use_mesh:
            # Duties with the actual V profile from MESH
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
                V_top=V_profile[-1], V_bot=V_profile[0],
                T_condenser=T_condenser,
            )
        else:
            # Duties with CMO vapor flow estimates
            Q_condenser, Q_reboiler = self._compute_duties(
                x_profile, y_profile, T_profile,
                F_total, z, T_feed, R, D, B,
                T_condenser=T_condenser,
            )

        info = {
            "T_profile": T_profile,
            "x_profile": x_profile,
            "y_profile": y_profile,
            "D": D,
            "B": B,
            "T_condenser": T_condenser,
            "T_reboiler": T_profile[0],
            "Q_condenser": Q_condenser,
            "Q_reboiler": Q_reboiler,
            "n_equilibrium_stages": n + (1 if partial_cond else 0),
            "overall_efficiency_equiv": eff,
        }
        if use_mesh:
            info["L_profile"] = L_profile
            info["V_profile"] = V_profile
        else:
            info["L_rect"] = R * D
            info["V_rect"] = (R + 1) * D
            # The frozen CMO profiles the solve used (issue #444), so code
            # that reads L_profile/V_profile works on either path.
            info["L_profile"], info["V_profile"] = self._cmo_flows(F_total, R, D)
        if partial_cond:
            info["y_D"] = y_D
            info["x_R"] = x_R

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
    """Minimum reflux of a pseudo-binary split at constant relative volatility.

    The pinch of the McCabe-Thiele diagram: the q-line meets the equilibrium
    curve ``y = alpha x / (1 + (alpha - 1) x)`` and the rectifying line
    through ``(x_D, x_D)`` and that point has the minimum slope. For a
    saturated-liquid feed (``q = 1``) this is the closed form
    ``(1/(alpha-1)) [x_D/z - alpha (1-x_D)/(1-z)]``. It is routed through
    :func:`difflow.staged.minimum_reflux`, so ``q`` is honoured (a vapor feed
    needs more reflux, a subcooled feed less).

    Changed in #424: ``q`` used to be accepted and ignored (every feed was
    treated as saturated liquid), and the result was floored at 0.1; the floor
    is gone, so a value below 0.1 (or below 0, an ``x_D`` below the pinch) is
    returned as computed. ``z_HK`` is not used (the feed is taken as
    ``z_HK = 1 - z_LK``, pseudo-binary).

    Args:
        z_LK: Feed mole fraction of light key
        z_HK: Feed mole fraction of heavy key (unused; ``1 - z_LK``)
        x_D_LK: Distillate mole fraction of light key
        alpha: Relative volatility
        q: Feed quality (1 saturated liquid, 0 saturated vapor)

    Returns:
        Minimum reflux ratio
    """
    from difflow.staged import constant_alpha, minimum_reflux

    return minimum_reflux(constant_alpha(alpha), z_LK, q, x_D_LK)


def gilliland_stages(
    R: Array,
    R_min: Array,
    N_min: Array,
    method: str = "eduljee",
) -> Array:
    """Calculate actual stages using Gilliland correlation.

    ``X = (R - R_min)/(R + 1)``, ``Y = (N - N_min)/(N + 1)``, with ``Y(X)``
    from Eduljee's fit ``Y = 0.75 (1 - X^0.5668)`` (the default) or
    Molokanov's ``Y = 1 - exp[(1 + 54.4X)/(11 + 117.2X) (X - 1)/sqrt(X)]``,
    which is the one :class:`ShortcutColumn` uses. The two differ by a stage
    or so at moderate ``X``.

    Handles singularities:
    - R → R_min: Y → 1, N → ∞ (capped at MAX_STAGES)

    Args:
        R: Actual reflux ratio
        R_min: Minimum reflux ratio
        N_min: Minimum stages
        method: ``"eduljee"`` (default) or ``"molokanov"``.

    Returns:
        Number of theoretical stages, capped at MAX_STAGES
    """
    # Ensure R >= R_min
    R_safe = jnp.maximum(R, R_min * 1.001)

    X = (R_safe - R_min) / (R_safe + 1)
    X_safe = jnp.maximum(X, 1e-6)

    if method == "eduljee":
        Y = 0.75 - 0.75 * X_safe**0.5668
    elif method == "molokanov":
        Y = 1 - jnp.exp(
            (1 + 54.4 * X_safe) / (11 + 117.2 * X_safe)
            * (X_safe - 1) / jnp.sqrt(X_safe)
        )
    else:
        raise ValueError(
            f"gilliland_stages: method must be 'eduljee' or 'molokanov', "
            f"got {method!r}"
        )

    # Cap Y to prevent N → ∞
    Y = jnp.clip(Y, 0.01, MAX_GILLILAND_Y)

    # N = (N_min + Y) / (1 - Y)
    one_minus_Y = jnp.maximum(1 - Y, 1e-3)
    N = (N_min + Y) / one_minus_Y

    # Cap at maximum stages
    N = jnp.minimum(N, MAX_STAGES)
    N = jnp.maximum(N, N_min)

    return N


def fenske_stages_multicomponent(
    x_D_LK: Array,
    x_D_HK: Array,
    x_B_LK: Array,
    x_B_HK: Array,
    alpha_LK: Array,
) -> Array:
    """Minimum stages of a multicomponent split (Fenske, key-ratio form).

    ``N_min = ln[(x_LK/x_HK)_D (x_HK/x_LK)_B] / ln(alpha_LK)``, with
    ``alpha_LK`` the light-key volatility relative to the heavy key. Only the
    ratios enter, so product flows (``d_i``, ``b_i``) work as well as mole
    fractions. Unlike :func:`fenske_stages` (pseudo-binary, ``1 - x_LK`` in
    place of ``x_HK``) it is not thrown off by non-key species. ``N_min``
    counts equilibrium stages and includes a partial reboiler.

    Args:
        x_D_LK: Light key in the distillate (mole fraction or flow).
        x_D_HK: Heavy key in the distillate (same basis as ``x_D_LK``).
        x_B_LK: Light key in the bottoms (mole fraction or flow).
        x_B_HK: Heavy key in the bottoms (same basis as ``x_B_LK``).
        alpha_LK: Relative volatility of the light key to the heavy key.

    Returns:
        Minimum number of equilibrium stages (not capped below; capped at
        ``MAX_STAGES`` above).

    Example:
        Geankoplis (5e) Example 11.7-2, 90 % key recoveries:

        >>> fenske_stages_multicomponent(0.3489, 0.0310, 0.0704, 0.5068, 2.258)
        Array(5.39..., dtype=float64)
    """
    numer = safe_log(
        safe_divide(x_D_LK, x_D_HK) * safe_divide(x_B_HK, x_B_LK)
    )
    log_alpha = jnp.maximum(
        safe_log(jnp.maximum(alpha_LK, 1.0 + EPS_DIVISION)), MIN_ALPHA_DIFF
    )
    return jnp.minimum(numer / log_alpha, MAX_STAGES)


def fenske_distribution(
    feed: Array,
    alpha: Array,
    light_key: int,
    heavy_key: int,
    recovery_LK: Array,
    recovery_HK: Array,
) -> tuple[Array, Array, Array]:
    """Split every species between the products at total reflux (Fenske).

    The keys are placed by their recoveries (``recovery_LK`` of the light key
    to the distillate, ``recovery_HK`` of the heavy key to the bottoms);
    ``N_min`` follows from the key ratios, and each other species ``i`` is
    distributed by ``d_i/b_i = (d/b)_HK alpha_i^N_min`` (``alpha`` relative to
    the heavy key). This is the Fenske-Geddes estimate of where the non-keys
    go; it is the distribution :class:`ShortcutColumn` uses.

    Args:
        feed: Component feed flows, shape ``(n,)`` (mol/s).
        alpha: Relative volatilities to the heavy key, shape ``(n,)``.
        light_key: Index of the light key.
        heavy_key: Index of the heavy key.
        recovery_LK: Fraction of the light key recovered in the distillate.
        recovery_HK: Fraction of the heavy key recovered in the bottoms.

    Returns:
        Tuple ``(d, b, N_min)``: distillate and bottoms component flows and
        the Fenske minimum stages.

    Example:
        >>> d, b, N_min = fenske_distribution(
        ...     jnp.array([40., 25., 20., 15.]),
        ...     jnp.array([4.5, 2.0, 1.0, 0.45]), 1, 2, 0.9, 0.9)
    """
    feed = jnp.asarray(feed, dtype=float)
    alpha = jnp.asarray(alpha, dtype=float)
    alpha = alpha / alpha[heavy_key]
    dLK = recovery_LK * feed[light_key]
    bLK = feed[light_key] - dLK
    bHK = recovery_HK * feed[heavy_key]
    dHK = feed[heavy_key] - bHK
    N_min = fenske_stages_multicomponent(dLK, dHK, bLK, bHK, alpha[light_key])
    ratio_HK = safe_divide(dHK, bHK)
    # d_i/b_i = ratio_HK alpha_i^N; d_i = f_i r/(1+r), done in log space
    log_r = safe_log(ratio_HK) + N_min * safe_log(alpha)
    frac_d = jax.nn.sigmoid(log_r)
    idx = jnp.arange(feed.shape[0])
    frac_d = jnp.where(idx == light_key, recovery_LK, frac_d)
    frac_d = jnp.where(idx == heavy_key, 1.0 - recovery_HK, frac_d)
    d = feed * frac_d
    return d, feed - d, N_min


def underwood_theta(
    alpha: Array,
    z: Array,
    q: Array = 1.0,
    alpha_HK: Array = 1.0,
) -> Array:
    """Root of Underwood's first equation between the keys.

    Solves ``sum_i alpha_i z_i / (alpha_i - theta) = 1 - q`` for the root
    that lies between ``alpha_HK`` and the next volatility above it (the
    light key when no species boils between the keys). The function is
    monotone (increasing) between adjacent poles, so the root is found by
    bisection on that open interval and cannot escape it; the derivative is
    the implicit one, ``-(df/dp)/(df/dtheta)``.

    Args:
        alpha: Relative volatilities, shape ``(n,)`` (any reference species).
        z: Feed mole fractions, shape ``(n,)``.
        q: Feed thermal condition (1 saturated liquid, 0 saturated vapour).
        alpha_HK: Volatility of the heavy key on the same basis as ``alpha``.

    Returns:
        theta, with ``alpha_HK < theta < alpha_next``.

    Example:
        Geankoplis (5e) Example 11.7-3:

        >>> underwood_theta(jnp.array([11.14, 4.93, 2.14, 1.0]),
        ...                 jnp.array([0.40, 0.25, 0.20, 0.15]), 1.0, 2.14)
    """
    from difflow.staged import _solve_monotone

    alpha = jnp.asarray(alpha, dtype=float)
    z = jnp.asarray(z, dtype=float)
    alpha_HK = jnp.asarray(alpha_HK, dtype=float)
    above = alpha > alpha_HK * (1.0 + 1e-9)
    alpha_up = jnp.min(jnp.where(above, alpha, jnp.inf))
    gap = lax.stop_gradient(alpha_up - alpha_HK)
    lo = lax.stop_gradient(alpha_HK) + 1e-10 * gap
    hi = lax.stop_gradient(alpha_up) - 1e-10 * gap

    def f(theta):
        return jnp.sum(alpha * z / (alpha - theta)) - (1.0 - q)

    return _solve_monotone(f, lo, hi, n_iter=80)


def underwood_minimum_reflux(
    alpha: Array,
    z: Array,
    x_D: Array,
    q: Array = 1.0,
    alpha_HK: Array = 1.0,
) -> tuple[Array, Array]:
    """Minimum reflux ratio by Underwood's method (one root, class 1 split).

    ``theta`` from :func:`underwood_theta`, then
    ``R_min = sum_i alpha_i x_D,i / (alpha_i - theta) - 1``. ``x_D`` should
    hold the distillate composition at (or near) minimum reflux, including the
    non-keys -- a Fenske distribution is the usual estimate.

    Args:
        alpha: Relative volatilities, shape ``(n,)``.
        z: Feed mole fractions, shape ``(n,)``.
        x_D: Distillate mole fractions, shape ``(n,)``.
        q: Feed thermal condition.
        alpha_HK: Volatility of the heavy key on the basis of ``alpha``.

    Returns:
        Tuple ``(R_min, theta)``. ``R_min`` is not floored; a negative value
        means the specified distillate is reachable without reflux.

    Example:
        >>> R_min, theta = underwood_minimum_reflux(
        ...     jnp.array([11.14, 4.93, 2.14, 1.0]),
        ...     jnp.array([0.40, 0.25, 0.20, 0.15]),
        ...     jnp.array([0.6197, 0.3489, 0.0310, 0.0004]), 1.0, 2.14)
    """
    alpha = jnp.asarray(alpha, dtype=float)
    theta = underwood_theta(alpha, z, q, alpha_HK)
    R_min = jnp.sum(alpha * jnp.asarray(x_D, dtype=float) / (alpha - theta)) - 1.0
    return R_min, theta


def underwood_class2(
    alpha: Array,
    f: Array,
    d_LK: Array,
    d_HK: Array,
    light_key: int,
    heavy_key: int,
    q: Array = 1.0,
    distributing: tuple[int, ...] | list[int] | None = None,
) -> tuple[Array, Array, Array]:
    """Minimum reflux by Underwood's general method (class 2: distributing non-keys).

    When non-key species distribute between the products at minimum reflux
    (always for species boiling between the keys, sometimes for a non-key
    just outside them), the single-root form :func:`underwood_minimum_reflux`
    does not apply. Each root ``theta_k`` of
    ``sum_i alpha_i f_i / (alpha_i - theta) = (1 - q) F`` in the volatility
    gaps spanned by the keys and the distributing species gives one equation

        ``sum_i alpha_i d_i / (alpha_i - theta_k) = V_min``

    in the unknowns ``V_min = D (R_min + 1)`` and the distillate flows of the
    distributing species; there is one root more than distributing species,
    so the linear system is square (Shiras, Hanson and Gibson, *Ind. Eng.
    Chem.* 42 (1950) 871; Seader and Henley, Sec. 9.2). Species lighter than
    the span go wholly to the distillate and heavier ones wholly to the
    bottoms.

    Args:
        alpha: Relative volatilities, shape ``(n,)`` (any reference species).
        f: Feed molar flows, shape ``(n,)``.
        d_LK: Distillate flow of the light key (same basis as ``f``).
        d_HK: Distillate flow of the heavy key.
        light_key: Index of the light key.
        heavy_key: Index of the heavy key.
        q: Feed thermal condition (1 saturated liquid, 0 saturated vapour).
        distributing: Indices of the distributing non-keys. Default: every
            species with a volatility strictly between the keys. Every
            species inside the span of keys and distributing species must be
            listed (a species cannot sit inside the span and not distribute).

    Returns:
        ``(R_min, d, thetas)``: the minimum reflux ratio, the distillate
        flows of every species at minimum reflux (shape ``(n,)``), and the
        roots used, in increasing order. A distributing species whose ``d``
        comes out below 0 or above its feed does not in fact distribute;
        drop it from ``distributing`` and solve again.

    Example:
        >>> R_min, d, th = underwood_class2(
        ...     jnp.array([4.0, 2.0, 1.5, 1.0]), jnp.array([30.0, 30.0, 10.0, 30.0]),
        ...     d_LK=29.0, d_HK=1.0, light_key=1, heavy_key=3)
    """
    from difflow.staged import _solve_monotone

    alpha = jnp.asarray(alpha, dtype=float)
    f = jnp.asarray(f, dtype=float)
    n = int(alpha.shape[0])
    a_np = np.asarray(lax.stop_gradient(alpha))
    if not a_np[light_key] > a_np[heavy_key]:
        raise ValueError("the light key must be more volatile than the heavy key")
    if distributing is None:
        distributing = [i for i in range(n)
                        if a_np[heavy_key] < a_np[i] < a_np[light_key]]
    distributing = sorted(set(int(i) for i in distributing) - {light_key, heavy_key})
    span = [light_key, heavy_key, *distributing]
    a_lo, a_hi = min(a_np[i] for i in span), max(a_np[i] for i in span)
    inside = [i for i in range(n) if a_lo <= a_np[i] <= a_hi]
    stray = sorted(set(inside) - set(span))
    if stray:
        raise ValueError(
            f"species {stray} lie inside the span of the keys and distributing "
            "species; list them in `distributing`")
    order = sorted(span, key=lambda i: a_np[i])
    F = jnp.sum(f)

    def underwood(theta):
        return jnp.sum(alpha * f / (alpha - theta)) - (1.0 - q) * F

    thetas = []
    for lo_i, hi_i in zip(order[:-1], order[1:]):
        a, b = alpha[lo_i], alpha[hi_i]
        gap = lax.stop_gradient(b - a)
        thetas.append(_solve_monotone(underwood, lax.stop_gradient(a) + 1e-10 * gap,
                                      lax.stop_gradient(b) - 1e-10 * gap, n_iter=80))
    thetas = jnp.stack(thetas)

    # Known distillate flows: keys as given, lighter than the span all up,
    # heavier all down.
    idx = jnp.arange(n)
    d_known = jnp.where(alpha > a_hi, f, 0.0)
    d_known = jnp.where(idx == light_key, d_LK, d_known)
    d_known = jnp.where(idx == heavy_key, d_HK, d_known)
    m = len(distributing)
    # Rows k: sum_j alpha_j/(alpha_j - th_k) d_j - V = -sum_known alpha_i d_i/(alpha_i - th_k)
    W = alpha[None, :] / (alpha[None, :] - thetas[:, None])          # (m+1, n)
    known_mask = jnp.ones(n).at[jnp.asarray(distributing, dtype=int)].set(0.0) \
        if m else jnp.ones(n)
    rhs = -jnp.sum(W * (d_known * known_mask)[None, :], axis=1)
    cols = [W[:, j] for j in distributing] + [-jnp.ones(m + 1)]
    sol = jnp.linalg.solve(jnp.stack(cols, axis=1), rhs)
    d = d_known * known_mask
    if m:
        d = d.at[jnp.asarray(distributing, dtype=int)].set(sol[:m])
    V_min = sol[m]
    D = jnp.sum(d)
    return V_min / D - 1.0, d, thetas


def kirkbride_feed_stage(
    N: Array,
    z_LK: Array,
    z_HK: Array,
    x_B_LK: Array,
    x_D_HK: Array,
    B: Array,
    D: Array,
) -> tuple[Array, Array]:
    """Continuous Kirkbride split of ``N`` stages above and below the feed.

    ``N_R/N_S = [(B/D) (z_HK/z_LK) (x_B,LK/x_D,HK)^2]^0.206``, with
    ``N_R + N_S = N``. The values are not rounded (a rounded stage has a zero
    derivative); round ``N_R`` to place the feed on a real tray.

    Args:
        N: Total equilibrium stages.
        z_LK: Light key mole fraction in the feed.
        z_HK: Heavy key mole fraction in the feed.
        x_B_LK: Light key mole fraction in the bottoms.
        x_D_HK: Heavy key mole fraction in the distillate.
        B: Bottoms flow (any basis shared with ``D``).
        D: Distillate flow.

    Returns:
        Tuple ``(N_R, N_S)``: stages above the feed (rectifying) and the
        rest (stripping, feed stage included).

    Example:
        >>> N_R, N_S = kirkbride_feed_stage(11.0, 0.25, 0.20, 0.0704, 0.0310,
        ...                                 35.52, 64.48)
    """
    ratio = jnp.exp(0.206 * safe_log(
        safe_divide(B, D) * safe_divide(z_HK, z_LK)
        * safe_divide(x_B_LK, x_D_HK) ** 2
    ))
    N_R = N * ratio / (1.0 + ratio)
    return N_R, N - N_R


def column_diameter(
    V: Array,
    rho_V: Array,
    rho_L: Array,
    sigma: Array,
    tray_spacing: float = 0.6,
    *,
    MW_V: Array,
    L: Array,
    MW_L: Array,
) -> Array:
    """Deprecated: use :func:`difflow.units.column_design.tray_column_diameter`.

    Thin wrapper kept for compatibility. The old body ignored ``sigma`` and
    ``tray_spacing`` and hard-coded the vapor molecular weight (30 g/mol), the
    flow parameter and ``C_SB``. It now calls ``tray_column_diameter`` (Fair
    flooding at 80 % of flood, 10 % downcomer), so the vapor and liquid
    molecular weights, the liquid rate and the surface tension are **required**
    (``MW_V``, ``L``, ``MW_L`` keyword-only; ``sigma`` positional).

    Args:
        V: Vapor molar flow rate (mol/s).
        rho_V: Vapor density (kg/m^3).
        rho_L: Liquid density (kg/m^3).
        sigma: Surface tension (N/m).
        tray_spacing: Tray spacing (m).
        MW_V: Vapor molecular weight (g/mol).
        L: Liquid molar flow rate (mol/s).
        MW_L: Liquid molecular weight (g/mol).

    Returns:
        Column diameter (m).
    """
    import warnings
    from difflow.units.column_design import tray_column_diameter

    warnings.warn(
        "column_diameter is deprecated; use "
        "difflow.units.column_design.tray_column_diameter",
        DeprecationWarning,
        stacklevel=2,
    )
    D, _ = tray_column_diameter(
        V, MW_V, rho_V, L, MW_L, rho_L, sigma, tray_spacing=tray_spacing
    )
    return D
