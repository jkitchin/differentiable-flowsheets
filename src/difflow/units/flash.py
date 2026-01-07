"""Flash separator unit operation.

The flash separator performs vapor-liquid equilibrium calculations
to split a feed stream into vapor and liquid products.

Currently supports:
- TP flash: Temperature and pressure specified

The VLE is solved using the Rachford-Rice equation with implicit
differentiation through the converged solution.
"""

from typing import Any
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo
from difflow.params_mixin import ParamsMixin
from difflow.constants import K_MIN, K_MAX, PHASE_TRANSITION_WIDTH
import optimistix as optx


@dataclass(repr=False)
class FlashParams(ParamsMixin):
    """Parameters for a flash separator.

    Attributes:
        species_order: List of species names for array ordering
    """
    species_order: list[str]


class Flash:
    """Flash separator with TP specification.

    Performs isothermal flash calculation at specified T and P.
    Uses Rachford-Rice equation to find vapor fraction, then
    computes liquid and vapor compositions.

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(
        self,
        params: FlashParams,
        thermo: IdealThermo,
    ):
        """Initialize flash separator.

        Args:
            params: Flash parameters
            thermo: Thermodynamic property calculator for K-values
        """
        self.params = params
        self.thermo = thermo

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

        # Get K-values from thermodynamics and clamp to avoid numerical issues.
        # Without clamping, extreme K-values cause problems:
        # - K → 0 for very heavy components causes division issues when V → 1
        # - K → ∞ for very volatile components causes overflow
        # - K → 1 near critical point makes (K-1) → 0, ill-conditioning Rachford-Rice
        K_raw = self.thermo.K_values_array(T, P)
        K = jnp.clip(K_raw, K_MIN, K_MAX)

        # Check for subcooled liquid or superheated vapor
        # sum(z*K) < 1 => subcooled liquid (all liquid)
        # sum(z/K) < 1 => superheated vapor (all vapor)
        bubble_check = jnp.sum(z * K)
        dew_check = jnp.sum(z / K)

        # Solve Rachford-Rice for vapor fraction
        V_frac = self._solve_flash(z, K, bubble_check, dew_check)

        # Get compositions (inlined from rachford_rice_compositions)
        x = z / (1 + V_frac * (K - 1))
        y = K * x

        # Normalize compositions to ensure they sum to 1.0.
        # Numerical errors in the Rachford-Rice solution can cause sum(x) and sum(y)
        # to deviate slightly from unity (e.g., 0.9999 or 1.0001). This normalization
        # ensures mass balance closure and prevents downstream numerical issues.
        x = x / jnp.sum(x)
        y = y / jnp.sum(y)

        # Calculate outlet flows
        L = F_total * (1 - V_frac)  # Liquid molar flow
        V = F_total * V_frac  # Vapor molar flow

        liquid_flows = {s: L * x[i] for i, s in enumerate(p.species_order)}
        vapor_flows = {s: V * y[i] for i, s in enumerate(p.species_order)}

        # Create outlet streams
        liquid = make_stream(liquid_flows, T, P)
        vapor = make_stream(vapor_flows, T, P)

        # Build info dict
        info = {
            "V_frac": V_frac,
            "K": {s: K[i] for i, s in enumerate(p.species_order)},
            "x": {s: x[i] for i, s in enumerate(p.species_order)},
            "y": {s: y[i] for i, s in enumerate(p.species_order)},
            "L": L,
            "V": V,
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
        V0_estimate = (bubble_check - 1.0) / (bubble_check + dew_check - 2.0 + 1e-10)
        V0 = jnp.clip(V0_estimate, 0.1, 0.9)  # Keep away from boundaries for initial guess
        args = (z, K)

        solver = optx.Newton(rtol=1e-10, atol=1e-10)
        sol = optx.root_find(rr_func, solver, V0, args=args, max_steps=50, throw=False)
        V_two_phase = jnp.clip(sol.value, 0.0, 1.0)

        # Handle edge cases with smooth transitions using sigmoid blending.
        # The original jnp.where approach creates discontinuous derivatives at
        # phase boundaries (bubble_check = 1, dew_check = 1), which breaks
        # gradient-based optimization. Sigmoid blending provides smooth gradients
        # while maintaining correct limiting behavior:
        # - bubble_check < 1: subcooled liquid (V → 0)
        # - dew_check < 1: superheated vapor (V → 1)
        # - otherwise: two-phase region (V = V_two_phase)
        #
        # The transition width (PHASE_TRANSITION_WIDTH) controls the sharpness:
        # smaller values → sharper transitions (closer to discontinuous)
        # larger values → smoother gradients but less accurate phase detection

        # Sigmoid functions for smooth blending
        # subcooled_weight → 1 when bubble_check < 1, → 0 when bubble_check > 1
        subcooled_weight = jax.nn.sigmoid(
            (1.0 - bubble_check) / PHASE_TRANSITION_WIDTH
        )
        # superheated_weight → 1 when dew_check < 1, → 0 when dew_check > 1
        superheated_weight = jax.nn.sigmoid(
            (1.0 - dew_check) / PHASE_TRANSITION_WIDTH
        )

        # Blend: subcooled pulls toward 0, superheated pulls toward 1
        # In two-phase region, both weights are ~0, so V_frac ≈ V_two_phase
        V_frac = (
            subcooled_weight * 0.0
            + superheated_weight * (1.0 - subcooled_weight) * 1.0
            + (1.0 - subcooled_weight) * (1.0 - superheated_weight) * V_two_phase
        )

        return V_frac

    def bubble_point_pressure(
        self,
        inlet: Stream,
        T: Array | float | None = None,
    ) -> Array:
        """Calculate bubble point pressure at given temperature.

        Args:
            inlet: Stream defining composition
            T: Temperature (K). If None, uses inlet T.

        Returns:
            Bubble point pressure (Pa)
        """
        T = jnp.asarray(T) if T is not None else inlet["T"]

        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        x = {s: inlet_flows[s] / F_total for s in self.params.species_order}

        return self.thermo.bubble_pressure(x, T)

    def dew_point_pressure(
        self,
        inlet: Stream,
        T: Array | float | None = None,
    ) -> Array:
        """Calculate dew point pressure at given temperature.

        Args:
            inlet: Stream defining composition
            T: Temperature (K). If None, uses inlet T.

        Returns:
            Dew point pressure (Pa)
        """
        T = jnp.asarray(T) if T is not None else inlet["T"]

        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        y = {s: inlet_flows[s] / F_total for s in self.params.species_order}

        return self.thermo.dew_pressure(y, T)

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
        z = F_feed / (F_total + 1e-10)

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
                V_frac = (bubble_check - 1.0) / (bubble_check + dew_check - 2.0 + 1e-10)
                V_frac = jnp.clip(V_frac, 0.0, 1.0)

        # Estimate compositions
        x = z / (1 + V_frac * (K - 1) + 1e-10)
        y = K * x
        x = x / (jnp.sum(x) + 1e-10)
        y = y / (jnp.sum(y) + 1e-10)

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
            'liquid': liquid,
            'vapor': vapor,
            'states': states,
            'info': info,
        }


class Splitter:
    """Simple stream splitter (no phase equilibrium).

    Splits a stream into two streams with specified split fraction.
    """

    def __init__(self, species_order: list[str]):
        """Initialize splitter.

        Args:
            species_order: List of species names
        """
        self.species_order = species_order

    def __call__(
        self,
        inlet: Stream,
        split_frac: Array | float,
    ) -> tuple[Stream, Stream]:
        """Split a stream.

        Args:
            inlet: Feed stream
            split_frac: Fraction of feed going to first outlet (0 to 1)

        Returns:
            outlet1: First outlet stream (split_frac of feed)
            outlet2: Second outlet stream (1 - split_frac of feed)
        """
        split_frac = jnp.asarray(split_frac)
        inlet_flows = get_flows(inlet)

        flows1 = {s: inlet_flows[s] * split_frac for s in self.species_order}
        flows2 = {s: inlet_flows[s] * (1 - split_frac) for s in self.species_order}

        outlet1 = make_stream(flows1, inlet["T"], inlet["P"])
        outlet2 = make_stream(flows2, inlet["T"], inlet["P"])

        return outlet1, outlet2


class Mixer:
    """Stream mixer.

    Combines multiple streams. For ideal mixing, outlet enthalpy
    equals sum of inlet enthalpies.
    """

    def __init__(
        self,
        species_order: list[str],
        thermo: IdealThermo | None = None,
    ):
        """Initialize mixer.

        Args:
            species_order: List of species names
            thermo: Optional thermo for enthalpy-based T calculation.
                   If None, uses flow-weighted average temperature.
        """
        self.species_order = species_order
        self.thermo = thermo

    def __call__(self, *inlets: Stream) -> Stream:
        """Mix multiple streams.

        Args:
            *inlets: Input streams to mix

        Returns:
            Mixed outlet stream
        """
        if not inlets:
            raise ValueError("At least one inlet required")

        # Sum flows
        outlet_flows = {}
        for s in self.species_order:
            outlet_flows[s] = sum(inlet[f"F_{s}"] for inlet in inlets)

        # Calculate outlet temperature
        if self.thermo is not None:
            # Energy balance: find T such that H_out = sum(H_in)
            H_total = jnp.zeros(())
            for inlet in inlets:
                flows = get_flows(inlet)
                H_total = H_total + self.thermo.stream_enthalpy(
                    {s: flows[s] for s in self.species_order},
                    inlet["T"],
                    phase="liquid"
                )

            # Estimate T from enthalpy (simplified - assume Cp is constant)
            total_flow = sum(outlet_flows.values())
            mole_fracs = {s: outlet_flows[s] / total_flow for s in self.species_order}

            # Use average inlet T as starting point
            T_avg = sum(inlet["T"] for inlet in inlets) / len(inlets)
            Cp_mix = self.thermo.Cp_mix(mole_fracs, T_avg)

            # H = n * Cp * (T - Tref), solve for T
            # This is approximate; could use Newton iteration for accuracy
            H_ref = self.thermo.stream_enthalpy(outlet_flows, 298.15, phase="liquid")
            T_out = 298.15 + (H_total - H_ref) / (total_flow * Cp_mix)
        else:
            # Flow-weighted average temperature
            total_flow = sum(outlet_flows.values())
            T_out = sum(
                sum(get_flows(inlet).values()) * inlet["T"]
                for inlet in inlets
            ) / total_flow

        # Use pressure from first inlet
        P_out = inlets[0]["P"]

        return make_stream(outlet_flows, T_out, P_out)
