"""Flash separator unit operation.

The flash separator performs vapor-liquid equilibrium calculations
to split a feed stream into vapor and liquid products.

Supports:
- TP flash: Temperature and pressure specified (ideal and non-ideal)
- PH flash: Pressure and enthalpy specified (isenthalpic)
- Bubble/dew point calculations (temperature and pressure)

The VLE is solved using the Rachford-Rice equation with implicit
differentiation through the converged solution.

For non-ideal systems, use EOSFlash with Peng-Robinson or SRK EOS.
"""

from typing import Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo
from difflow.params_mixin import ParamsMixin
from difflow.constants import K_MIN, K_MAX
from difflow.numerics import safe_divide
from difflow.eos import PengRobinson, SRK, flash_TP_eos
import optimistix as optx
from typing import Literal


@dataclass(repr=False)
class FlashParams(ParamsMixin):
    """Parameters for a flash separator.

    Attributes:
        species_order: List of species names for array ordering
    """
    species_order: list[str]

    def __post_init__(self):
        """Validate flash parameters."""
        if not self.species_order:
            raise ValueError("species_order cannot be empty")
        if len(self.species_order) != len(set(self.species_order)):
            raise ValueError("species_order contains duplicate species names")



#: Relative component-balance residual above which a flash's inner solve is
#: reported as not converged (``info["converged"]``).
INNER_BALANCE_TOL = 1e-6

class Flash:
    """Flash separator with TP specification.

    Performs isothermal flash calculation at specified T and P.
    Uses Rachford-Rice equation to find vapor fraction, then
    computes liquid and vapor compositions.

    All calculations are JAX-compatible for automatic differentiation.
    """

    symbol = "Flash"
    equations = [
        r"\sum_i \frac{z_i (K_i - 1)}{1 + \beta(K_i - 1)} = 0 \qquad \text{(Rachford-Rice)}",
        r"x_i = \frac{z_i}{1 + \beta(K_i - 1)},\qquad y_i = K_i\, x_i",
        r"K_i = \frac{P_i^{\mathrm{sat}}(T)}{P}\qquad \text{(ideal, Raoult's law)}",
    ]
    assumptions = [
        "Vapor-liquid equilibrium at specified T and P.",
        "Ideal thermodynamics via Raoult's law (or supplied EOS for the non-ideal variant).",
        "Single equilibrium stage with complete phase separation.",
    ]
    references = [
        "Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e, Ch. 10-14.",
        "Rachford, H.H., Rice, J.D. J. Petroleum Tech., 4(10), 1952.",
    ]
    parameter_symbols = {}
    parameter_units = {}
    numerical_method = "Rachford-Rice root-finding via optimistix; implicit differentiation through the converged root."

    def __init__(
        self,
        params: FlashParams,
        thermo: IdealThermo,
        eos: PengRobinson | SRK | None = None,
        k_ij: Array | None = None,
        activity_model=None,
    ):
        """Initialize flash separator.

        Args:
            params: Flash parameters
            thermo: Thermodynamic property calculator for K-values (used
                when ``eos`` is None)
            eos: Optional equation of state (PengRobinson or SRK). When
                provided, K-values are computed from fugacity coefficients
                via ``flash_TP_eos()`` instead of Raoult's law.
            k_ij: Binary interaction parameters for EOS (n x n matrix).
                Ignored when ``eos`` is None.
            activity_model: Optional liquid-phase activity-coefficient model
                implementing the :class:`~difflow.activity.ActivityModel`
                protocol (a ``gamma(x, T)`` method): ``NRTLParams``,
                ``UNIQUACParams``, ``WilsonParams``, ``MargulesParams`` or
                ``VanLaarParams``, with ``species`` matching
                ``params.species_order``. When provided (and no EOS),
                K-values follow the modified Raoult's law
                ``K_i = gamma_i(x, T) * Psat_i / P`` so non-ideal liquid
                behavior — including azeotropes — is captured (#90).
        """
        self.params = params
        self.thermo = thermo
        self.eos = eos
        self.k_ij = k_ij
        self.activity_model = activity_model

    def __call__(
        self,
        inlet: Stream,
        T: Array | float | None = None,
        P: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform TP flash calculation.

        Args:
            inlet: Feed stream
            T: Flash temperature (K). If None, uses inlet T.
            P: Flash pressure (Pa). If None, uses inlet P.

        Returns:
            liquid: Liquid outlet stream
            vapor: Vapor outlet stream
            info: Dictionary with additional information:
                - 'V_frac': Vapor fraction (molar)
                - 'K': K-values for each species
                - 'x': Liquid mole fractions
                - 'y': Vapor mole fractions
        """
        p = self.params

        # Use inlet T, P if not specified
        T = jnp.asarray(T) if T is not None else inlet["T"]
        P = jnp.asarray(P) if P is not None else inlet["P"]

        # Get feed flows and compute mole fractions
        inlet_flows = get_flows(inlet)
        F_feed = jnp.array([inlet_flows[s] for s in p.species_order])
        F_total = jnp.sum(F_feed)
        z = F_feed / F_total  # Feed mole fractions

        if self.eos is not None:
            # --- EOS path: K-values from fugacity coefficients ---
            V_frac, x, y = flash_TP_eos(self.eos, z, T, P, self.k_ij)
            K = jnp.where(x > 1e-15, y / x, jnp.ones_like(x))
        elif self.activity_model is not None:
            # --- Activity-coefficient (gamma-phi) path: modified Raoult's law
            # K_i = gamma_i(x, T) * Psat_i / P, capturing azeotropes (#90). ---
            from difflow.activity import activity_gamma
            # Ideal K = Psat/P; gamma multiplies it.
            K_ideal = jnp.clip(self.thermo.K_values_array(T, P), K_MIN, K_MAX)

            # gamma depends on the liquid composition, so iterate a fixed
            # number of times (unrolled -> JAX/JIT compatible).
            x_iter = z
            V_frac = jnp.asarray(0.5)
            for _ in range(30):
                gamma = activity_gamma(self.activity_model, x_iter, T)
                K = jnp.clip(gamma * K_ideal, K_MIN, K_MAX)
                bubble_check = jnp.sum(z * K)
                dew_check = jnp.sum(z / K)
                V_frac = self._solve_flash(z, K, bubble_check, dew_check)
                x_new = z / (1 + V_frac * (K - 1))
                x_iter = x_new / jnp.sum(x_new)

            x = x_iter
            gamma = activity_gamma(self.activity_model, x, T)
            K = jnp.clip(gamma * K_ideal, K_MIN, K_MAX)
            y = jnp.where(
                V_frac > 1e-10,
                (z - x * (1 - V_frac)) / jnp.maximum(V_frac, 1e-10),
                K * x,
            )
            y = jnp.maximum(y, 0.0)
            y = y / jnp.sum(y)
        else:
            # --- Ideal (Raoult's law) path ---
            # Get K-values from thermodynamics and clamp to avoid numerical issues.
            K_raw = self.thermo.K_values_array(T, P)
            K = jnp.clip(K_raw, K_MIN, K_MAX)

            # Check for subcooled liquid or superheated vapor
            bubble_check = jnp.sum(z * K)
            dew_check = jnp.sum(z / K)

            # Solve Rachford-Rice for vapor fraction
            V_frac = self._solve_flash(z, K, bubble_check, dew_check)

            # Compositions from Rachford-Rice
            x = z / (1 + V_frac * (K - 1))
            x = x / jnp.sum(x)
            y = jnp.where(
                V_frac > 1e-10,
                (z - x * (1 - V_frac)) / jnp.maximum(V_frac, 1e-10),
                K * x,
            )
            y = jnp.maximum(y, 0.0)
            y = y / jnp.sum(y)

        # Calculate outlet flows
        L = F_total * (1 - V_frac)  # Liquid molar flow
        V = F_total * V_frac  # Vapor molar flow

        liquid_flows = {s: L * x[i] for i, s in enumerate(p.species_order)}
        vapor_flows = {s: V * y[i] for i, s in enumerate(p.species_order)}

        # Create outlet streams
        liquid = make_stream(liquid_flows, T, P)
        vapor = make_stream(vapor_flows, T, P)

        # Phase detection flags
        # Recompute bubble/dew checks for the info dict (needed for both paths)
        bubble_check = jnp.sum(z * K)
        dew_check = jnp.sum(z / K)
        # 0 = two-phase, 1 = subcooled liquid, 2 = superheated vapor
        phase_flag = jnp.where(
            bubble_check < 1.0, 1,
            jnp.where(dew_check < 1.0, 2, 0),
        )

        # Component balance of the streams actually returned. Rachford-Rice
        # runs without a convergence check and y is clipped and renormalised
        # afterwards, so a failed solve shows up here and nowhere else; a
        # single-phase answer (V_frac at 0 or 1) closes it exactly.
        balance_residual = jnp.max(jnp.abs(L * x + V * y - F_total * z)) / \
            jnp.maximum(F_total, 1e-30)

        # Build info dict
        info = {
            "balance_residual": balance_residual,
            "converged": balance_residual < INNER_BALANCE_TOL,
            "V_frac": V_frac,
            "K": {s: K[i] for i, s in enumerate(p.species_order)},
            "x": {s: x[i] for i, s in enumerate(p.species_order)},
            "y": {s: y[i] for i, s in enumerate(p.species_order)},
            "L": L,
            "V": V,
            "phase_flag": phase_flag,
            "bubble_check": bubble_check,
            "dew_check": dew_check,
        }

        return liquid, vapor, info

    def _solve_flash(
        self,
        z: Array,
        K: Array,
        bubble_check: Array,
        dew_check: Array,
    ) -> Array:
        """Solve for vapor fraction handling edge cases.

        Args:
            z: Feed mole fractions
            K: K-values
            bubble_check: sum(z*K), < 1 means subcooled
            dew_check: sum(z/K), < 1 means superheated

        Returns:
            Vapor fraction in [0, 1]
        """
        # Rachford-Rice function: sum_i z_i * (K_i - 1) / (1 + V * (K_i - 1)) = 0
        #
        # Numerical stability considerations:
        # - When V = 1 and K < 1: denominator = 1 + (K-1) = K approaches K_MIN
        # - When V = 0 and K > 1: denominator = 1, always safe
        # - K-values are clamped to [K_MIN, K_MAX] upstream, preventing extreme values
        #
        # We add a small regularization to prevent division by exactly zero
        # in pathological cases where the clamped K still causes issues.
        def rr_func(V, args):
            z_, K_ = args
            denom = 1.0 + V * (K_ - 1.0)
            # Regularize denominator to prevent division by zero
            denom_safe = jnp.where(jnp.abs(denom) < 1e-10, 1e-10, denom)
            return jnp.sum(z_ * (K_ - 1.0) / denom_safe)

        # Initial guess: use a heuristic based on bubble/dew checks for faster convergence.
        # When bubble_check >> 1 and dew_check >> 1, we're well within two-phase region.
        # A better starting point than 0.5 can reduce iterations.
        # Simple estimate: V ≈ (bubble_check - 1) / (bubble_check + dew_check - 2)
        # This interpolates between 0 (at bubble point) and 1 (at dew point).
        # That interpolation is only a heuristic. g(V) is monotone on [0, 1]
        # but convex, and with one strongly volatile minor component (5%
        # ethanol in water, K ~ 8) Newton from the interpolated guess
        # overshoots past the pole at V = 1/(1 - K_max) and lands on a root
        # outside [0, 1], so the flash reported V = 0 inside the two-phase
        # region. Bisection on [0, 1] cannot leave the bracket; it only seeds
        # Newton (stop_gradient), which polishes the root and keeps the
        # implicit derivative of optx.root_find.
        Kc = jax.lax.stop_gradient(K)
        zc = jax.lax.stop_gradient(z)

        def _bisect(_, ab):
            a, b = ab
            m = 0.5 * (a + b)
            pos = rr_func(m, (zc, Kc)) > 0.0   # g falls with V: root lies above m
            return jnp.where(pos, m, a), jnp.where(pos, b, m)

        a, b = jax.lax.fori_loop(0, 40, _bisect, (jnp.asarray(0.0), jnp.asarray(1.0)))
        V0 = 0.5 * (a + b)
        args = (z, K)

        solver = optx.Newton(rtol=1e-10, atol=1e-10)
        sol = optx.root_find(rr_func, solver, V0, args=args, max_steps=50, throw=False)
        V_two_phase = jnp.clip(sol.value, 0.0, 1.0)

        # Single-phase feeds. On [0, 1] the clipped root is already continuous
        # in T and P: at the bubble point g(0) = bubble_check - 1 = 0 and at
        # the dew point g(1) = 1 - dew_check = 0, so V meets 0 and 1 there
        # with a kink (the physical one) and the clip's derivative is zero
        # outside. The explicit selects only guard against a Newton polish
        # that left the bracket. An earlier sigmoid blend of width
        # PHASE_TRANSITION_WIDTH did nothing outside the two-phase region but
        # biased V inside it by 10-30% within a few percent of either
        # boundary (benzene/toluene 50/50 at 1 atm, 366.9 K: 0.230 against the
        # exact 0.2499; issue #444), so V is now the exact root.
        V_frac = jnp.where(
            bubble_check <= 1.0, 0.0,
            jnp.where(dew_check <= 1.0, 1.0, V_two_phase),
        )
        return V_frac

    # =========================================================================
    # Equation-Oriented Interface
    # =========================================================================

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        For a TP flash with 1 inlet and 2 outlets (liquid, vapor):
            Material balance: F_in_i - F_liq_i - F_vap_i = 0   (n_species)
            Equilibrium: F_vap_i * L_total - K_i * F_liq_i * V_total = 0  (n_species)
            T_liq - T_flash = 0                                  (1)
            T_vap - T_flash = 0                                  (1)
            P_liq - P_in = 0                                     (1)
            P_vap - P_in = 0                                     (1)

        Total: 2*n_species + 4 residuals for 2*(n_species + 2) unknowns.

        Args:
            inlets: List of inlet streams (expects 1 inlet)
            outlets: List of outlet streams (expects 2: [liquid, vapor])
            **kwargs: Optional T, P overrides for flash conditions

        Returns:
            Flat array of residuals, length 2*n_species + 4
        """
        p = self.params
        inlet = inlets[0]
        liquid = outlets[0]
        vapor = outlets[1]

        # Flash conditions
        T_flash = kwargs.get('T')
        T_flash = jnp.asarray(T_flash) if T_flash is not None else inlet["T"]
        P_flash = kwargs.get('P')
        P_flash = jnp.asarray(P_flash) if P_flash is not None else inlet["P"]

        # Get flows
        inlet_flows = get_flows(inlet)
        liquid_flows = get_flows(liquid)
        vapor_flows = get_flows(vapor)

        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        F_liq = jnp.array([liquid_flows[s] for s in p.species_order])
        F_vap = jnp.array([vapor_flows[s] for s in p.species_order])

        # Material balance: F_in - F_liq - F_vap = 0
        mat_resid = F_in - F_liq - F_vap

        # K-values at flash conditions
        from difflow.constants import K_MIN, K_MAX
        K = self.thermo.K_values_array(T_flash, P_flash)
        K = jnp.clip(K, K_MIN, K_MAX)

        # Phase equilibrium: y_i = K_i * x_i
        # In terms of flows: F_vap_i / V_total = K_i * F_liq_i / L_total
        # Rearranged: F_vap_i * L_total - K_i * F_liq_i * V_total = 0
        L_total = jnp.sum(F_liq) + 1e-10
        V_total = jnp.sum(F_vap) + 1e-10
        equil_resid = F_vap * L_total - K * F_liq * V_total

        # Temperature residuals
        T_liq_resid = jnp.atleast_1d(liquid["T"] - T_flash)
        T_vap_resid = jnp.atleast_1d(vapor["T"] - T_flash)

        # Pressure residuals
        P_liq_resid = jnp.atleast_1d(liquid["P"] - P_flash)
        P_vap_resid = jnp.atleast_1d(vapor["P"] - P_flash)

        return jnp.concatenate([
            mat_resid, equil_resid,
            T_liq_resid, T_vap_resid,
            P_liq_resid, P_vap_resid,
        ])

    def _feed_composition(self, inlet: Stream) -> tuple[tuple[str, ...], Array]:
        """Feed mole fractions, ordered as the EOS (if any) or ``species_order``."""
        order = (tuple(self.eos.species_order) if self.eos is not None
                 else tuple(self.params.species_order))
        flows = get_flows(inlet)
        F = jnp.stack([jnp.asarray(flows.get(s, 0.0), dtype=float) for s in order])
        return order, F / jnp.sum(F)

    def bubble_point_pressure(
        self,
        inlet: Stream,
        T: Array | float | None = None,
    ) -> Array:
        """Bubble-point pressure of the feed (as a liquid) at temperature ``T``.

        Uses the same thermodynamics as the flash itself: the cubic EOS
        (:func:`~difflow.eos.eos_bubble_P`) when ``eos`` is set, the modified
        Raoult's law ``sum x_i gamma_i Psat_i`` when ``activity_model`` is set,
        otherwise Raoult's law.

        Args:
            inlet: Stream defining composition
            T: Temperature (K). If None, uses inlet T.

        Returns:
            Bubble point pressure (Pa)
        """
        import difflow.phase_diagrams as pd
        from difflow.eos import eos_bubble_P
        T = jnp.asarray(T) if T is not None else inlet["T"]
        species, z = self._feed_composition(inlet)
        if self.eos is not None:
            return eos_bubble_P(self.eos, z, T, self.k_ij)["P"]
        return pd.bubble_P(self.thermo, species, z, T, self.activity_model)

    def dew_point_pressure(
        self,
        inlet: Stream,
        T: Array | float | None = None,
    ) -> Array:
        """Dew-point pressure of the feed (as a vapor) at temperature ``T``.

        EOS (:func:`~difflow.eos.eos_dew_P`), modified Raoult
        (:func:`~difflow.phase_diagrams.dew_P`) or Raoult's law, matching the
        flash's own model.

        Args:
            inlet: Stream defining composition
            T: Temperature (K). If None, uses inlet T.

        Returns:
            Dew point pressure (Pa)
        """
        import difflow.phase_diagrams as pd
        from difflow.eos import eos_dew_P
        T = jnp.asarray(T) if T is not None else inlet["T"]
        species, z = self._feed_composition(inlet)
        if self.eos is not None:
            return eos_dew_P(self.eos, z, T, self.k_ij)["P"]
        return pd.dew_P(self.thermo, species, z, T, self.activity_model)

    def bubble_point_temperature(
        self,
        inlet: Stream,
        P: Array | float | None = None,
        T_guess: float | None = None,
    ) -> Array:
        """Bubble-point temperature of the feed at pressure ``P``.

        EOS (:func:`~difflow.eos.eos_bubble_T`) when ``eos`` is set, otherwise
        :func:`~difflow.phase_diagrams.bubble_T` (modified Raoult with the
        ``activity_model``, or Raoult). Newton with implicit differentiation.
        The result is never clipped: a solve that fails to converge (or an
        EOS solve that lands on the trivial K = 1 root) issues a
        :class:`~difflow.thermo.BubbleDewConvergenceWarning` (concrete values
        only), and an Antoine evaluation outside its fitted range issues an
        :class:`~difflow.thermo.AntoineRangeWarning`.

        Args:
            inlet: Stream defining composition
            P: Pressure (Pa). If None, uses inlet P.
            T_guess: Optional initial temperature guess (K); by default the
                composition-weighted Antoine (or Wilson, for an EOS) estimate.

        Returns:
            Bubble point temperature (K)
        """
        import difflow.phase_diagrams as pd
        from difflow.eos import eos_bubble_T
        P = jnp.asarray(P) if P is not None else inlet["P"]
        species, z = self._feed_composition(inlet)
        if self.eos is not None:
            return eos_bubble_T(self.eos, z, P, self.k_ij, T_guess)["T"]
        return pd.bubble_T(self.thermo, species, z, P, self.activity_model,
                           T0=T_guess)

    def dew_point_temperature(
        self,
        inlet: Stream,
        P: Array | float | None = None,
        T_guess: float | None = None,
    ) -> Array:
        """Dew-point temperature of the feed at pressure ``P``.

        EOS (:func:`~difflow.eos.eos_dew_T`) when ``eos`` is set, otherwise
        :func:`~difflow.phase_diagrams.dew_T` (modified Raoult or Raoult).
        Never clipped; warns as :meth:`bubble_point_temperature` does.

        Args:
            inlet: Stream defining composition
            P: Pressure (Pa). If None, uses inlet P.
            T_guess: Optional initial temperature guess (K).

        Returns:
            Dew point temperature (K)
        """
        import difflow.phase_diagrams as pd
        from difflow.eos import eos_dew_T
        P = jnp.asarray(P) if P is not None else inlet["P"]
        species, z = self._feed_composition(inlet)
        if self.eos is not None:
            return eos_dew_T(self.eos, z, P, self.k_ij, T_guess)["T"]
        return pd.dew_T(self.thermo, species, z, P, self.activity_model,
                        T0=T_guess)

    # =========================================================================
    # Initialization Interface
    # =========================================================================

    def initialize(
        self,
        inlet: Stream,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate initial guesses for flash outputs.

        Uses simplified Rachford-Rice calculation for quick estimates.

        Args:
            inlet: Inlet stream
            **kwargs: Additional parameters:
                - T: Flash temperature (K)
                - P: Flash pressure (Pa)
                - expected_vapor_fraction: Optional hint for vapor fraction

        Returns:
            Dictionary containing:
            - 'liquid': Initial guess for liquid outlet stream
            - 'vapor': Initial guess for vapor outlet stream
            - 'states': Dict with vapor fraction, compositions
            - 'info': Additional initialization information
        """
        p = self.params

        # Use inlet T, P if not specified
        T = kwargs.get('T', inlet["T"])
        P = kwargs.get('P', inlet["P"])
        T = jnp.asarray(T)
        P = jnp.asarray(P)

        # Get feed flows and compute mole fractions
        inlet_flows = get_flows(inlet)
        F_feed = jnp.array([inlet_flows[s] for s in p.species_order])
        F_total = jnp.sum(F_feed)
        z = safe_divide(F_feed, F_total)

        # Get K-values from thermodynamics
        K = self.thermo.K_values_array(T, P)
        K = jnp.clip(K, K_MIN, K_MAX)

        # Quick estimate of vapor fraction
        expected_V = kwargs.get('expected_vapor_fraction')
        if expected_V is not None:
            V_frac = jnp.asarray(expected_V)
        else:
            # Use bubble/dew check for initial estimate
            bubble_check = jnp.sum(z * K)
            dew_check = jnp.sum(z / K)

            # Simple heuristic
            if bubble_check < 1:
                V_frac = jnp.asarray(0.0)  # Subcooled liquid
            elif dew_check < 1:
                V_frac = jnp.asarray(1.0)  # Superheated vapor
            else:
                # Estimate based on relative volatility
                V_frac = safe_divide(bubble_check - 1.0, bubble_check + dew_check - 2.0)
                V_frac = jnp.clip(V_frac, 0.0, 1.0)

        # Estimate compositions
        x = safe_divide(z, 1 + V_frac * (K - 1))
        y = K * x
        x = safe_divide(x, jnp.sum(x))
        y = safe_divide(y, jnp.sum(y))

        # Calculate outlet flows
        L = F_total * (1 - V_frac)
        V = F_total * V_frac

        liquid_flows = {s: L * x[i] for i, s in enumerate(p.species_order)}
        vapor_flows = {s: V * y[i] for i, s in enumerate(p.species_order)}

        # Create outlet streams
        liquid = make_stream(liquid_flows, T, P)
        vapor = make_stream(vapor_flows, T, P)

        states = {
            'V_frac': float(V_frac),
            'K': {s: float(K[i]) for i, s in enumerate(p.species_order)},
            'x': {s: float(x[i]) for i, s in enumerate(p.species_order)},
            'y': {s: float(y[i]) for i, s in enumerate(p.species_order)},
        }

        info = {
            'method': 'simplified_rachford_rice',
            'bubble_point_estimate': float(jnp.sum(z * K)),
            'dew_point_estimate': float(jnp.sum(z / K)),
        }

        return {
            # Positional, in the order __call__ returns them: that is the
            # only thing a caller holding Unit.outlet_names can match the
            # guesses up by. 'liquid'/'vapor' stay for callers that know
            # they are talking to a flash.
            'outlets': (liquid, vapor),
            'liquid': liquid,
            'vapor': vapor,
            'states': states,
            'info': info,
        }


class Splitter:
    """Simple stream splitter (no phase equilibrium).

    Splits a stream into n streams with specified split fractions.
    Default is n=2 (two outlet streams).
    """

    symbol = "Splitter"
    equations = [
        r"F_{i,k}^{\mathrm{out}} = \phi_k\, F_{i}^{\mathrm{in}}\qquad \sum_k \phi_k = 1",
        r"T^{\mathrm{out}} = T^{\mathrm{in}},\quad P^{\mathrm{out}} = P^{\mathrm{in}}",
    ]
    assumptions = [
        "No phase change; each outlet has the same composition as the inlet.",
        "Isothermal, isobaric split.",
    ]
    references = ["Seider, Seader, Lewin, Widagdo. Product & Process Design Principles, 4e."]
    parameter_symbols = {}
    parameter_units = {}
    #: outlets to draw when dropped on a canvas: the count follows
    #: ``split_frac``, which the return annotation cannot say
    default_outlets = 2

    def __init__(self, species_order: list[str]):
        """Initialize splitter.

        Args:
            species_order: List of species names
        """
        self.species_order = species_order

    @staticmethod
    def _normalize_split_fracs(split_frac) -> list:
        """Convert split_frac input to a list of JAX arrays.

        A scalar value is treated as a 2-way split: [split_frac, 1 - split_frac].
        A sequence of n values defines an n-way split; the last fraction is
        recomputed as 1 - sum(others) to guarantee mass balance.

        Args:
            split_frac: Scalar or sequence of split fractions.

        Returns:
            List of JAX scalar arrays, one per outlet.
        """
        frac = jnp.asarray(split_frac)
        if frac.ndim == 0:
            # Scalar -> 2-way split (backward compatible)
            return [frac, 1 - frac]
        # Sequence -> n-way split; recompute last for exact mass balance
        fracs = [frac[i] for i in range(frac.shape[0] - 1)]
        fracs.append(1 - jnp.sum(jnp.stack(fracs)))
        return fracs

    def __call__(
        self,
        inlet: Stream,
        split_frac: Array | float | list | tuple,
    ) -> tuple:
        """Split a stream into n outlets.

        Args:
            inlet: Feed stream
            split_frac: Split fractions. A scalar gives a 2-way split
                (fraction to first outlet; second gets the remainder).
                A sequence of n values gives an n-way split (should sum to 1;
                the last fraction is recomputed as 1 - sum(others)).

        Returns:
            *outlets: n outlet streams
            info: Dictionary with split information
        """
        fracs = self._normalize_split_fracs(split_frac)
        inlet_flows = get_flows(inlet)

        outlets = []
        total_flow = sum(inlet_flows.values())
        flow_info = {}
        for k, f in enumerate(fracs):
            flows = {s: inlet_flows[s] * f for s in self.species_order}
            outlets.append(make_stream(flows, inlet["T"], inlet["P"]))
            flow_info[f"flow_to_outlet{k + 1}"] = total_flow * f

        info = {
            "split_fractions": jnp.stack(fracs),
            "n_outlets": len(fracs),
            **flow_info,
        }

        return (*outlets, info)

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        split_frac: Array | float | list | tuple = 0.5,
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        For each outlet k (n total):
            F_out_k_i - frac_k * F_in_i = 0   (n_species per outlet)
            T_out_k - T_in = 0                 (1 per outlet)
            P_out_k - P_in = 0                 (1 per outlet)

        Total: n * (n_species + 2) residuals

        Args:
            inlets: [inlet_stream]
            outlets: List of n outlet streams
            split_frac: Split fractions (scalar or sequence)

        Returns:
            Flat residual array
        """
        fracs = self._normalize_split_fracs(split_frac)
        inlet = inlets[0]
        inlet_flows = get_flows(inlet)

        all_resid = []
        for k, f in enumerate(fracs):
            out_flows = get_flows(outlets[k])
            for s in self.species_order:
                all_resid.append(jnp.atleast_1d(
                    out_flows[s] - f * inlet_flows[s]
                ))
            all_resid.append(jnp.atleast_1d(outlets[k]["T"] - inlet["T"]))
            all_resid.append(jnp.atleast_1d(outlets[k]["P"] - inlet["P"]))

        return jnp.concatenate(all_resid)


@dataclass(repr=False)
class ComponentSplitterParams(ParamsMixin):
    """Parameters of a :class:`ComponentSplitter`.

    Attributes:
        split_fractions: ``{species: fraction}`` of each species' inlet flow
            sent to the *first* outlet (its recovery there); the complement
            goes to the second. Values in ``[0, 1]``; may be JAX tracers.
        default_fraction: Fraction to the first outlet for any species not in
            ``split_fractions`` (default 0: unlisted species all leave in the
            second outlet).
    """

    split_fractions: dict
    default_fraction: float = 0.0

    def __post_init__(self):
        if not isinstance(self.split_fractions, dict):
            raise ValueError("split_fractions must be a {species: fraction} dict")
        for s, f in list(self.split_fractions.items()) + [("default_fraction", self.default_fraction)]:
            try:
                value = float(f)
            except Exception:      # a tracer: nothing to check yet
                continue
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"split fraction for {s!r} must lie in [0, 1], got {value}")


class ComponentSplitter:
    """Ideal component splitter: a specified fraction of each species to each outlet.

    ``ComponentSplitter(params)(inlet) -> (first, second, info)``. Species
    ``i`` leaves in the first outlet at ``phi_i F_i`` and in the second at
    ``(1 - phi_i) F_i``. It is the block-flow separator of a material-balance
    calculation ("all the methanol to the liquid, 98% of the CO overhead"):
    no equilibrium, no thermodynamic model, both outlets at the inlet
    temperature and pressure. Differentiable in the fractions and the feed.

    Unlike :class:`Splitter`, whose outlets share the inlet composition, each
    species here has its own split. (``ComponentSeparator`` in
    ``difflow.units.eos_units`` is the same balance with an enthalpy duty and
    needs a ``CubicThermo``.)

    ``info``: ``split_fractions`` (array in the order of the inlet's
    species), ``species`` (that order), ``flow_first`` and ``flow_second``
    (total mol/s).

    Example:
        >>> from difflow import make_stream, ComponentSplitter, ComponentSplitterParams
        >>> sep = ComponentSplitter(ComponentSplitterParams({"MeOH": 0.0}, default_fraction=1.0))
        >>> gas, liq, _ = sep(make_stream({"CO": 10.0, "MeOH": 5.0}, T=320.0, P=5e6))
        >>> float(gas["F_CO"]), float(liq["F_MeOH"])
        (10.0, 5.0)
    """

    symbol = "Component Splitter"
    equations = [
        r"F_i^{(1)} = \phi_i\, F_i^{\mathrm{in}},\qquad F_i^{(2)} = (1 - \phi_i)\, F_i^{\mathrm{in}}",
        r"T^{(1)} = T^{(2)} = T^{\mathrm{in}},\quad P^{(1)} = P^{(2)} = P^{\mathrm{in}}",
    ]
    assumptions = [
        "Specified per-species split fractions (recoveries); no phase equilibrium.",
        "Isothermal, isobaric; no energy balance.",
    ]
    references = [
        "Felder, Rousseau, Bullard. Elementary Principles of Chemical Processes, 4e, Ch. 4.",
    ]
    parameter_symbols = {"split_fractions": r"\phi_i", "default_fraction": r"\phi_0"}
    parameter_units = {"split_fractions": "-", "default_fraction": "-"}
    numerical_method = "Closed form (no iteration)."
    outlet_roles = ["first", "second"]

    def __init__(self, params: ComponentSplitterParams):
        """Build the splitter.

        Args:
            params: Per-species fractions to the first outlet.
        """
        self.params = params

    def __call__(self, inlet: Stream) -> tuple[Stream, Stream, dict]:
        """Split the inlet species by species.

        Args:
            inlet: Feed stream.

        Returns:
            first: Outlet receiving ``split_fractions`` of each species.
            second: Outlet receiving the complement.
            info: Fractions and outlet totals.
        """
        flows = get_flows(inlet)
        p = self.params
        phi = {
            s: jnp.asarray(p.split_fractions.get(s, p.default_fraction), dtype=float)
            for s in flows
        }
        first = {s: phi[s] * F for s, F in flows.items()}
        second = {s: (1.0 - phi[s]) * F for s, F in flows.items()}
        info = {
            "species": list(flows),
            "split_fractions": jnp.stack([phi[s] for s in flows]) if flows else jnp.zeros(0),
            "flow_first": sum(first.values()),
            "flow_second": sum(second.values()),
        }
        return (
            make_stream(first, inlet["T"], inlet["P"]),
            make_stream(second, inlet["T"], inlet["P"]),
            info,
        )


class Mixer:
    """Stream mixer.

    Combines multiple streams. For ideal mixing, outlet enthalpy
    equals sum of inlet enthalpies.
    """

    symbol = "Mixer"
    equations = [
        r"F_i^{\mathrm{out}} = \sum_k F_i^{(k)}",
        r"\sum_k \sum_i F_i^{(k)}\, h_i(T^{(k)}) = \sum_i F_i^{\mathrm{out}}\, h_i(T^{\mathrm{out}})\qquad \text{(enthalpy balance)}",
        r"P^{\mathrm{out}} = \min_k P^{(k)}",
    ]
    assumptions = [
        "Adiabatic mixing (no external heat duty).",
        "Negligible heat of mixing (ideal).",
        "Outlet pressure set by the lowest inlet pressure.",
        "With thermo, T_out solves the enthalpy balance exactly (Newton from the flow-weighted mean T, bracketed fallback; implicit gradient); without thermo it is the flow-weighted mean T (constant, equal heat capacities).",
        "A species absent from an inlet stream is taken as zero flow in that inlet.",
    ]
    references = ["Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e, Ch. 12."]
    parameter_symbols = {}
    parameter_units = {}

    def __init__(
        self,
        species_order: list[str],
        thermo: IdealThermo | None = None,
        phase: str = "liquid",
    ):
        """Initialize mixer.

        Args:
            species_order: List of species names
            thermo: Optional thermo for enthalpy-based T calculation.
                   If None, uses flow-weighted average temperature.
            phase: Phase for enthalpy calculations ("liquid" or "vapor")
        """
        self.species_order = species_order
        self.thermo = thermo
        self.phase = phase

    def __call__(self, *inlets: Stream) -> tuple[Stream, dict[str, Array]]:
        """Mix multiple streams.

        Args:
            *inlets: Input streams to mix

        Returns:
            outlet: Mixed outlet stream
            info: Dictionary with mixing information
        """
        if not inlets:
            raise ValueError("At least one inlet required")

        # Sum flows.  An inlet that does not carry a species contributes no
        # flow of it, as in combine_streams: a stream built without ethanol
        # is a stream with no ethanol in it, not an error.
        inlet_flows = [
            {s: inlet.get(f"F_{s}", 0.0) for s in self.species_order}
            for inlet in inlets
        ]
        outlet_flows = {}
        for s in self.species_order:
            outlet_flows[s] = sum(flows[s] for flows in inlet_flows)
        total_flow = sum(outlet_flows.values())

        # Flow-weighted average temperature: the answer without thermo, and
        # the starting point of the enthalpy solve with it.
        T_weighted = sum(
            sum(flows.values()) * inlet["T"]
            for flows, inlet in zip(inlet_flows, inlets)
        ) / total_flow

        if self.thermo is not None:
            # Energy balance: solve H_out(T) = sum_k H_in,k exactly.  The
            # phase is fixed, so H_out is smooth in T and Newton from the
            # flow-weighted mean converges to the root nearest it, which
            # matters because a fixed-phase enthalpy need not be monotone
            # (IdealThermo's vapor branch adds a Watson Hvap that vanishes
            # at Tc, so near Tc it has two roots).  Should Newton fail, the
            # bracketed solve of the heat exchangers is the fallback.  Both
            # are optimistix root finds, differentiated at the converged
            # root by the implicit function theorem.
            import optimistix as optx
            from difflow.units.heat_exchanger import invert_monotone_T

            H_total = jnp.zeros(())
            for flows, inlet in zip(inlet_flows, inlets):
                H_total = H_total + self.thermo.stream_enthalpy(
                    flows, inlet["T"], phase=self.phase
                )
            mole_fracs = {
                s: outlet_flows[s] / total_flow for s in self.species_order
            }
            scale = jnp.maximum(
                total_flow * self.thermo.Cp_mix(mole_fracs, T_weighted), 1e-12
            )

            def H_out(T):
                return self.thermo.stream_enthalpy(
                    outlet_flows, T, phase=self.phase
                )

            newton = optx.root_find(
                lambda T, _: (H_out(T) - H_total) / scale,
                optx.Newton(rtol=1e-12, atol=1e-9),
                T_weighted, max_steps=50, throw=False,
            )
            T_newton = newton.value
            ok = (
                (newton.result == optx.RESULTS.successful)
                & jnp.isfinite(T_newton)
                & (T_newton > 0.0)
            )
            T_out = jax.lax.cond(
                ok,
                lambda: T_newton,
                lambda: invert_monotone_T(H_out, H_total, T_weighted, scale),
            )
        else:
            T_out = T_weighted

        # Use minimum pressure across all inlets (consistent with combine_streams)
        P_out = inlets[0]["P"]
        for inlet in inlets[1:]:
            P_out = jnp.minimum(P_out, inlet["P"])

        outlet = make_stream(outlet_flows, T_out, P_out)

        info = {
            "n_inlets": len(inlets),
            "total_flow": total_flow,
            "T_out": T_out,
            "P_out": P_out,
        }

        return outlet, info

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        Residuals:
            F_out_i - sum(F_in_j_i) = 0  (n_species)
            T_out - T_mixed = 0           (1)
            P_out - P_in = 0              (1)

        Total: n_species + 2 residuals

        Args:
            inlets: List of inlet streams
            outlets: [outlet_stream]

        Returns:
            Flat residual array
        """
        outlet = outlets[0]
        outlet_flows = get_flows(outlet)

        # Material balance: F_out_i = sum of all inlet F_i
        mat_resid = []
        for s in self.species_order:
            F_in_total = sum(inlet.get(f"F_{s}", 0.0) for inlet in inlets)
            mat_resid.append(jnp.atleast_1d(outlet_flows[s] - F_in_total))

        # Temperature: flow-weighted average (or energy balance with thermo)
        if self.thermo is not None:
            # Compute expected mixed temperature
            mixed_stream, _ = self(*inlets)
            T_expected = mixed_stream["T"]
        else:
            total_flow = sum(outlet_flows.values())
            T_expected = sum(
                sum(inlet.get(f"F_{sp}", 0.0) for sp in self.species_order)
                * inlet["T"]
                for inlet in inlets
            ) / (total_flow + 1e-10)

        T_resid = jnp.atleast_1d(outlet["T"] - T_expected)

        # Pressure: use first inlet
        P_resid = jnp.atleast_1d(outlet["P"] - inlets[0]["P"])

        return jnp.concatenate(mat_resid + [T_resid, P_resid])


# =============================================================================
# EOSFlash - Non-ideal VLE using Cubic Equations of State
# =============================================================================


@dataclass(repr=False)
class EOSFlashParams(ParamsMixin):
    """Parameters for EOS-based flash separator.

    Attributes:
        species_order: List of species names for array ordering
        eos_type: Type of EOS ('PR' for Peng-Robinson, 'SRK' for Soave-Redlich-Kwong)
    """
    species_order: list[str]
    eos_type: Literal["PR", "SRK"] = "PR"

    def __post_init__(self):
        """Validate EOS flash parameters."""
        if not self.species_order:
            raise ValueError("species_order cannot be empty")
        if len(self.species_order) != len(set(self.species_order)):
            raise ValueError("species_order contains duplicate species names")
        if self.eos_type not in ("PR", "SRK"):
            raise ValueError(f"eos_type must be 'PR' or 'SRK', got '{self.eos_type}'")


class EOSFlash:
    """Flash separator using cubic equation of state for non-ideal VLE.

    For systems where ideal behavior (Raoult's law) is insufficient,
    this class uses Peng-Robinson or SRK equations of state to compute
    K-values from fugacity coefficients.

    Uses successive substitution to converge K-values:
    1. Initialize K from Wilson correlation
    2. Solve Rachford-Rice for vapor fraction V
    3. Calculate x, y from V
    4. Update K from fugacity coefficients
    5. Repeat until converged

    All calculations are JAX-compatible for automatic differentiation.

    Example:
        >>> from difflow.eos import CriticalProperties
        >>> species_data = {
        ...     "methane": CriticalProperties("methane", 190.6, 4.6e6, 0.011),
        ...     "ethane": CriticalProperties("ethane", 305.4, 4.9e6, 0.099),
        ... }
        >>> from difflow.eos import PengRobinson
        >>> eos = PengRobinson(species_data)
        >>> params = EOSFlashParams(species_order=["methane", "ethane"])
        >>> flash = EOSFlash(params, eos)
        >>> liquid, vapor, info = flash(inlet, T=250.0, P=2e6)
    """

    symbol = "EOS Flash"
    equations = [
        r"K_i = \frac{\hat{\varphi}_i^L(T,P,\mathbf{x})}{\hat{\varphi}_i^V(T,P,\mathbf{y})}\qquad \text{(EOS K-values)}",
        r"\sum_i \frac{z_i(K_i-1)}{1+\beta(K_i-1)} = 0 \qquad \text{(Rachford-Rice)}",
        r"P = \frac{RT}{v-b} - \frac{a(T)}{v^2 + u\,b\,v + w\,b^2} \qquad \text{(cubic EOS: PR / SRK)}",
    ]
    assumptions = [
        "Vapor-liquid equilibrium at specified T and P.",
        "Cubic equation of state (Peng-Robinson or SRK) with optional binary interaction parameters k_ij.",
        "Successive substitution until K-values converge.",
    ]
    references = [
        "Peng, D.-Y., Robinson, D.B. Ind. Eng. Chem. Fundam., 15, 59 (1976).",
        "Soave, G. Chem. Eng. Sci., 27, 1197 (1972).",
    ]
    parameter_symbols = {"eos_type": "EOS"}
    parameter_units = {}
    numerical_method = "Successive substitution on K-values, Rachford-Rice root inside the outer loop."

    def __init__(
        self,
        params: EOSFlashParams,
        eos: PengRobinson | SRK,
        k_ij: Array | None = None,
    ):
        """Initialize EOS flash separator.

        Args:
            params: Flash parameters
            eos: Equation of state object (PengRobinson or SRK)
            k_ij: Binary interaction parameters (n x n matrix).
                  If None, assumes k_ij = 0 for all pairs.
        """
        self.params = params
        self.eos = eos
        self.k_ij = k_ij

    def __call__(
        self,
        inlet: Stream,
        T: Array | float | None = None,
        P: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform TP flash calculation using EOS.

        Args:
            inlet: Feed stream
            T: Flash temperature (K). If None, uses inlet T.
            P: Flash pressure (Pa). If None, uses inlet P.

        Returns:
            liquid: Liquid outlet stream
            vapor: Vapor outlet stream
            info: Dictionary with additional information
        """
        p = self.params

        # Use inlet T, P if not specified
        T = jnp.asarray(T) if T is not None else inlet["T"]
        P = jnp.asarray(P) if P is not None else inlet["P"]

        # Get feed flows and compute mole fractions
        inlet_flows = get_flows(inlet)
        F_feed = jnp.array([inlet_flows[s] for s in p.species_order])
        F_total = jnp.sum(F_feed)
        z = safe_divide(F_feed, F_total)

        # Perform flash using EOS
        V_frac, x, y = flash_TP_eos(self.eos, z, T, P, self.k_ij)

        # Calculate outlet flows
        L = F_total * (1 - V_frac)  # Liquid molar flow
        V = F_total * V_frac  # Vapor molar flow

        liquid_flows = {s: L * x[i] for i, s in enumerate(p.species_order)}
        vapor_flows = {s: V * y[i] for i, s in enumerate(p.species_order)}

        # Create outlet streams
        liquid = make_stream(liquid_flows, T, P)
        vapor = make_stream(vapor_flows, T, P)

        # Compute K-values from final compositions
        K = self.eos.K_values(T, P, x, y, self.k_ij)

        # Build info dict
        info = {
            "V_frac": V_frac,
            "K": {s: K[i] for i, s in enumerate(p.species_order)},
            "x": {s: x[i] for i, s in enumerate(p.species_order)},
            "y": {s: y[i] for i, s in enumerate(p.species_order)},
            "L": L,
            "V": V,
            "eos_type": p.eos_type,
        }

        return liquid, vapor, info

    # =========================================================================
    # Equation-Oriented Interface
    # =========================================================================

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        For a TP flash with 1 inlet and 2 outlets (liquid, vapor):
            Material balance: F_in_i - F_liq_i - F_vap_i = 0        (n_species)
            Phase split:                                             (n_species)
                two-phase:     x_i phi_L_i(x) - y_i phi_V_i(y) = 0  (fugacity equality)
                liquid only:   F_vap_i = 0
                vapor only:    F_liq_i = 0
            T_liq - T_flash = 0, T_vap - T_flash = 0                 (2)
            P_liq - P_flash = 0, P_vap - P_flash = 0                 (2)

        Total: 2*n_species + 4 residuals for 2*(n_species + 2) unknowns.

        The compositions are ``x = F_liq / L`` and ``y = F_vap / V``, so the
        summations hold by construction and fugacity equality plus the
        component balances fix the split. Fugacity equality is written as
        ``x_i phi_L_i - y_i phi_V_i`` rather than in logs so a species absent
        from the feed still gives a well-posed row.

        Which rows apply is decided the way the sequential flash decides it:
        by the Michelsen stability test on the *feed* (z, T, P), under
        ``stop_gradient``. The choice depends only on the inlet, never on the
        outlet unknowns, so within a phase regime the residual is smooth and
        its Jacobian is nonsingular -- in the single-phase regime fugacity
        equality would otherwise leave the absent phase's composition
        undetermined. The switch moves where the sequential flash's does, so
        ``solve_eo`` and ``solve`` agree on both sides of a phase boundary.

        Args:
            inlets: List of inlet streams (expects 1 inlet)
            outlets: List of outlet streams (expects 2: [liquid, vapor])
            **kwargs: Optional T, P overrides for flash conditions

        Returns:
            Flat array of residuals, length 2*n_species + 4
        """
        from difflow.eos import _phase_stability

        p = self.params
        inlet = inlets[0]
        liquid = outlets[0]
        vapor = outlets[1]

        T_flash = kwargs.get('T')
        T_flash = jnp.asarray(T_flash) if T_flash is not None else inlet["T"]
        P_flash = kwargs.get('P')
        P_flash = jnp.asarray(P_flash) if P_flash is not None else inlet["P"]

        inlet_flows = get_flows(inlet)
        liquid_flows = get_flows(liquid)
        vapor_flows = get_flows(vapor)

        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        F_liq = jnp.array([liquid_flows[s] for s in p.species_order])
        F_vap = jnp.array([vapor_flows[s] for s in p.species_order])

        mat_resid = F_in - F_liq - F_vap

        # Phase regime from the feed, as flash_TP_eos decides it. A boolean
        # carries no derivative, so the test is kept off the tape.
        sg = jax.lax.stop_gradient
        z = sg(safe_divide(F_in, jnp.sum(F_in)))
        two_phase, feed_is_vapor = _phase_stability(
            self.eos, z, sg(T_flash), sg(P_flash), self.k_ij
        )

        # Compositions. In a single-phase regime one phase is empty; both
        # fugacity rows are then evaluated at the feed composition so the
        # unselected branch stays finite under jacobian (a NaN there would
        # survive the jnp.where below as NaN * 0).
        L = jnp.sum(F_liq)
        V = jnp.sum(F_vap)
        x = jnp.where(two_phase, F_liq / jnp.where(two_phase, L, 1.0), z)
        y = jnp.where(two_phase, F_vap / jnp.where(two_phase, V, 1.0), z)
        phi_L = self.eos.fugacity_coefficient(T_flash, P_flash, x, "liquid", self.k_ij)
        phi_V = self.eos.fugacity_coefficient(T_flash, P_flash, y, "vapor", self.k_ij)
        fug_resid = x * phi_L - y * phi_V

        single_resid = jnp.where(feed_is_vapor, F_liq, F_vap)
        split_resid = jnp.where(two_phase, fug_resid, single_resid)

        T_liq_resid = jnp.atleast_1d(liquid["T"] - T_flash)
        T_vap_resid = jnp.atleast_1d(vapor["T"] - T_flash)
        P_liq_resid = jnp.atleast_1d(liquid["P"] - P_flash)
        P_vap_resid = jnp.atleast_1d(vapor["P"] - P_flash)

        return jnp.concatenate([
            mat_resid, split_resid,
            T_liq_resid, T_vap_resid,
            P_liq_resid, P_vap_resid,
        ])


# =============================================================================
# PHFlash - Isenthalpic Flash (Constant P and H)
# =============================================================================


class PHFlash:
    """Isenthalpic flash separator (constant pressure and enthalpy).

    The PH flash finds the equilibrium temperature such that the
    outlet enthalpy equals the inlet enthalpy at specified pressure.

    Common applications:
    - Adiabatic flash drums
    - Valve expansion
    - Pressure reduction

    All calculations are JAX-compatible for automatic differentiation.

    Two thermodynamic paths:

    - :class:`~difflow.thermo.IdealThermo`: ideal-K (Raoult) inner flash, the
      inlet enthalpy from the inlet's own split at its (T, P) (or the phase
      named by ``inlet_phase``), and a bracketed bisection
      on ``T_bounds`` for the outlet T. A pure component that lands inside its
      two-phase region comes back at its boiling point with the lever-rule
      vapour fraction (the old Newton on T could not land there: it returned
      saturated water let down from 180 C to 1.5 bar at the 150 K clip with
      ``V_frac = 0``).
    - :class:`~difflow.thermo.CubicThermo`: a rigorous cubic-EOS PH flash
      (:func:`difflow.units.eos_units.flash_PH_eos`). The inlet enthalpy is the
      two-phase EOS enthalpy at the inlet (T, P); the outlet temperature comes
      from a bracketed bisection on ``T_bounds`` (no guess, no wrong root); a
      pure component inside its dome is returned at its saturation temperature
      with the lever-rule vapour fraction. ``T_guess`` is ignored on this path.
    """

    symbol = "PH Flash"
    equations = [
        r"H^{\mathrm{out}}(T,P) = H^{\mathrm{in}} \qquad \text{(isenthalpic constraint)}",
        r"\sum_i \frac{z_i(K_i-1)}{1+\beta(K_i-1)} = 0 \qquad \text{(Rachford-Rice at trial } T)",
    ]
    assumptions = [
        "Adiabatic operation (constant enthalpy).",
        "Constant outlet pressure P.",
        "Vapor-liquid equilibrium at the converged temperature.",
    ]
    references = ["Smith, Van Ness, Abbott. Introduction to Chemical Engineering Thermodynamics, 7e."]
    parameter_symbols = {}
    parameter_units = {}
    numerical_method = ("Bracketed bisection on T enforcing H(T,P) = H_in, inner TP flash at each trial T; "
                        "lever rule at a pure-component jump; implicit-function gradients.")

    def __init__(
        self,
        params: FlashParams,
        thermo: IdealThermo,
    ):
        """Initialize PH flash separator.

        Args:
            params: Flash parameters
            thermo: Thermodynamic property calculator
        """
        self.params = params
        self.thermo = thermo
        self._tp_flash = Flash(params, thermo)

    def __call__(
        self,
        inlet: Stream,
        P: Array | float | None = None,
        T_guess: float = 300.0,
        T_bounds: tuple[float, float] = (100.0, 1000.0),
        inlet_phase: str = "auto",
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform PH (isenthalpic) flash calculation.

        Args:
            inlet: Feed stream
            P: Flash pressure (Pa). If None, uses inlet P.
            T_guess: Ignored (kept for backward compatibility); both paths
                bracket the temperature with ``T_bounds``.
            T_bounds: (T_min, T_max) bracket (K) of the temperature search.
            inlet_phase: Phase state of the inlet for its enthalpy (ideal
                path). ``"auto"`` (default) flashes the inlet at its own
                (T, P); ``"liquid"`` or ``"vapor"`` takes the whole feed in
                that phase (needed for a pure feed exactly at its boiling
                point). The cubic-EOS path always flashes the inlet.

        Returns:
            liquid: Liquid outlet stream
            vapor: Vapor outlet stream
            info: Dictionary with additional information including T_flash
        """
        if inlet_phase not in ("auto", "liquid", "vapor"):
            raise ValueError(
                f"inlet_phase must be 'auto', 'liquid' or 'vapor', got {inlet_phase!r}")
        p = self.params

        # Use inlet P if not specified
        P_flash = jnp.asarray(P) if P is not None else inlet["P"]

        from difflow.thermo import CubicThermo
        if isinstance(self.thermo, CubicThermo):
            return self._eos_ph_flash(inlet, P_flash, T_bounds)

        return self._ideal_ph_flash(inlet, P_flash, T_bounds, inlet_phase)

    def _exact_split(self, z, T, P):
        """Unsmoothed Raoult TP split ``(V, x, y)`` at (T, P).

        Single phase where ``sum z K <= 1`` (liquid) or ``sum z / K <= 1``
        (vapour), otherwise the Rachford-Rice root found by bisection on
        [0, 1] with its implicit derivative attached. A pure component's
        split jumps exactly at its boiling point, which the (P, H) bisection
        needs to see.
        """
        sg = jax.lax.stop_gradient
        K = jnp.clip(jnp.array([self.thermo.K_value(s, T, P) for s in self.params.species_order]),
                     K_MIN, K_MAX)
        bub, dew = jnp.sum(z * K), jnp.sum(z / K)

        def rr(V, K_):
            return jnp.sum(z * (K_ - 1.0) / (1.0 + V * (K_ - 1.0)))

        Kc = sg(K)

        def body(_, ab):
            a, b = ab
            m = 0.5 * (a + b)
            pos = rr(m, Kc) > 0.0      # rr falls with V: root lies above m
            return jnp.where(pos, m, a), jnp.where(pos, b, m)

        a, b = jax.lax.fori_loop(0, 60, body, (jnp.asarray(0.0), jnp.asarray(1.0)))
        V_s = 0.5 * (a + b)
        val = rr(V_s, K)
        dV = sg(jax.grad(rr)(V_s, Kc))
        dV = jnp.where(jnp.abs(dV) > 0.0, dV, -1.0)
        V_rr = V_s - (val - sg(val)) / dV
        V = jnp.where(bub <= 1.0, 0.0, jnp.where(dew <= 1.0, 1.0, V_rr))
        x = z / (1.0 + V * (K - 1.0))
        x = x / jnp.sum(x)
        y = K * x
        y = y / jnp.sum(y)
        return V, x, y

    def _split_enthalpy(self, F_tot, z, T, P):
        """Total outlet enthalpy (W) of the unsmoothed split, and the split."""
        V, x, y = self._exact_split(z, T, P)
        order = self.params.species_order
        H_L = self.thermo.stream_enthalpy(
            {s: F_tot * (1.0 - V) * x[i] for i, s in enumerate(order)}, T, phase="liquid")
        H_V = self.thermo.stream_enthalpy(
            {s: F_tot * V * y[i] for i, s in enumerate(order)}, T, phase="vapor")
        return H_L + H_V, (V, x, y)

    def _ideal_ph_flash(self, inlet, P_flash, T_bounds, inlet_phase="auto"):
        """Ideal-K (P, H) flash: bracketed bisection plus the lever rule.

        At fixed P the outlet enthalpy of the Raoult split rises monotonically
        with T, so T is found by bisection on ``T_bounds`` (no guess, no wrong
        root). A mixture's flashed enthalpy is continuous in T and the
        bisection lands on the root. A pure component's enthalpy *jumps* by the
        latent heat at its boiling point: the bisection then converges on the
        jump and the vapour fraction comes from the lever rule,
        ``beta = (H_in - H_L(T)) / (H_V(T) - H_L(T))``. The derivative is
        attached by the implicit function theorem on the energy balance, or on
        the bubble condition ``ln sum z_i K_i = 0`` at a jump (so ``dT/dP``
        inside a pure fluid's two-phase region is the Clausius-Clapeyron slope).
        The split is the exact Rachford-Rice one.

        The inlet enthalpy is that of the inlet's own exact split at its
        (T, P) with ``inlet_phase="auto"`` (a subcooled liquid, a two-phase
        feed and a superheated vapour all come out right), or of the whole
        feed in the named phase with ``"liquid"`` / ``"vapor"``. A pure feed
        exactly at its boiling point is ambiguous; name its phase.
        """
        p = self.params
        order = p.species_order
        sg = jax.lax.stop_gradient
        inlet_flows = get_flows(inlet)
        flows = {s: inlet_flows[s] for s in order}
        T_in = inlet["T"]
        F_vec = jnp.array([flows[s] for s in order])
        F_tot = jnp.sum(F_vec)
        z = F_vec / jnp.maximum(F_tot, 1e-30)
        # Inlet enthalpy from the inlet's phase state at its own (T, P). It
        # used to be the liquid's whatever the inlet was, so a vapour feed
        # entered the balance short by its latent heat (issue #444).
        if inlet_phase == "auto":
            H_inlet = self._split_enthalpy(F_tot, z, T_in, inlet["P"])[0]
        else:
            H_inlet = self.thermo.stream_enthalpy(flows, T_in, phase=inlet_phase)

        F_c, z_c = sg(F_tot), sg(z)
        P_c, H_c = sg(jnp.asarray(P_flash, dtype=float)), sg(H_inlet)
        lo, hi = float(T_bounds[0]), float(T_bounds[1])

        def r(T):
            return self._split_enthalpy(F_c, z_c, T, P_c)[0] - H_c

        r_lo, r_hi = r(jnp.asarray(lo)), r(jnp.asarray(hi))
        bracketed = (r_lo <= 0.0) & (r_hi >= 0.0)

        def body(_, ab):
            a, b = ab
            m = 0.5 * (a + b)
            below = r(m) < 0.0
            return jnp.where(below, m, a), jnp.where(below, b, m)

        a, b = jax.lax.fori_loop(0, 60, body, (jnp.asarray(lo), jnp.asarray(hi)))
        T_s = jnp.where(r_lo > 0.0, lo, jnp.where(r_hi < 0.0, hi, 0.5 * (a + b)))
        scale = jnp.abs(H_c) + F_c * 8.314462618 * T_s
        jump = bracketed & (jnp.abs(r(T_s)) > 1e-7 * scale)

        def g_balance(T):
            return self._split_enthalpy(F_tot, z, T, P_flash)[0] - H_inlet

        def g_sat(T):
            K = jnp.array([self.thermo.K_value(s, T, P_flash) for s in order])
            return jnp.log(jnp.sum(z * K))

        def attach(g):
            val = g(T_s)
            dgdT = sg(jax.grad(g)(T_s))
            dgdT = jnp.where(jnp.abs(dgdT) > 0.0, dgdT, 1.0)
            return T_s - (val - sg(val)) / dgdT

        T = jnp.where(jump, attach(g_sat), attach(g_balance))
        T = jnp.where(bracketed, T, T_s)

        H_out, (V, x, y) = self._split_enthalpy(F_tot, z, T, P_flash)
        # Lever rule at a jump, on the phase enthalpies of the feed composition.
        h_L = self.thermo.stream_enthalpy(flows, T, phase="liquid")
        h_V = self.thermo.stream_enthalpy(flows, T, phase="vapor")
        dH = jnp.where(jnp.abs(h_V - h_L) > 1e-12, h_V - h_L, 1e-12)
        beta = jnp.where(jump, jnp.clip((H_inlet - h_L) / dH, 0.0, 1.0), V)
        x = jnp.where(jump, z, x)
        y = jnp.where(jump, z, y)
        H_out = jnp.where(jump, h_L + beta * dH, H_out)
        L_flow, V_flow = (1.0 - beta) * F_tot, beta * F_tot
        liquid = make_stream({s: L_flow * x[i] for i, s in enumerate(order)}, T, P_flash)
        vapor = make_stream({s: V_flow * y[i] for i, s in enumerate(order)}, T, P_flash)
        K = jnp.clip(jnp.array([self.thermo.K_value(s, T, P_flash) for s in order]), K_MIN, K_MAX)
        residual = H_out - H_inlet
        info = {
            "V_frac": beta,
            "K": {s: K[i] for i, s in enumerate(order)},
            "x": {s: x[i] for i, s in enumerate(order)},
            "y": {s: y[i] for i, s in enumerate(order)},
            "L": L_flow,
            "V": V_flow,
            "bubble_check": jnp.sum(z * K),
            "dew_check": jnp.sum(z / K),
            "T_flash": T,
            "T_out": T,
            "H_inlet": H_inlet,
            "T_inlet": T_in,
            "P_flash": P_flash,
            "residual": residual,
            "converged": bracketed & (jnp.abs(sg(residual)) <= 1e-7 * scale),
        }
        from difflow.units.eos_units import _warn_unconverged
        _warn_unconverged("PHFlash", info)
        return liquid, vapor, info

    def _eos_ph_flash(self, inlet, P_flash, T_bounds):
        """Cubic-EOS PH flash (``CubicThermo`` path of :meth:`__call__`)."""
        from difflow.units.eos_units import flash_PH_eos, _warn_unconverged

        order = list(self.thermo.eos.species_order)  # x, y come back in EOS order
        inlet_flows = get_flows(inlet)
        flows = {s: inlet_flows[s] for s in order}
        T_in = inlet["T"]
        H_inlet = self.thermo.stream_enthalpy_flash(flows, T_in, inlet["P"])
        res = flash_PH_eos(self.thermo, flows, P_flash, H_inlet, T_bounds)

        F_total = sum(flows[s] for s in order)
        beta = res["vapor_fraction"]
        x, y = res["x"], res["y"]
        # A pure component (or any single-phase TP result at a jump) has x = y
        # = z, so this split also covers the lever-rule case.
        liquid = make_stream({s: F_total * (1.0 - beta) * x[i] for i, s in enumerate(order)},
                             res["T"], P_flash)
        vapor = make_stream({s: F_total * beta * y[i] for i, s in enumerate(order)},
                            res["T"], P_flash)
        info = {
            "V_frac": beta,
            "x": {s: x[i] for i, s in enumerate(order)},
            "y": {s: y[i] for i, s in enumerate(order)},
            "T_flash": res["T"],
            "T_out": res["T"],
            "H_inlet": H_inlet,
            "T_inlet": T_in,
            "P_flash": P_flash,
            "residual": res["residual"],
            "converged": res["converged"],
        }
        _warn_unconverged("PHFlash", info)
        return liquid, vapor, info
