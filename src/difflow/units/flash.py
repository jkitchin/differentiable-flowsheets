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
from difflow.constants import K_MIN, K_MAX, PHASE_TRANSITION_WIDTH
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
        eos: PengRobinson | SRK | None = None,
        k_ij: Array | None = None,
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
        """
        self.params = params
        self.thermo = thermo
        self.eos = eos
        self.k_ij = k_ij

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

        # Build info dict
        info = {
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
        V0_estimate = safe_divide(bubble_check - 1.0, bubble_check + dew_check - 2.0)
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

    def bubble_point_temperature(
        self,
        inlet: Stream,
        P: Array | float | None = None,
        T_guess: float = 350.0,
    ) -> Array:
        """Calculate bubble point temperature at given pressure.

        Solves: sum(x_i * Psat_i(T)) = P

        Args:
            inlet: Stream defining composition
            P: Pressure (Pa). If None, uses inlet P.
            T_guess: Initial temperature guess (K)

        Returns:
            Bubble point temperature (K)
        """
        P = jnp.asarray(P) if P is not None else inlet["P"]

        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        x = {s: inlet_flows[s] / F_total for s in self.params.species_order}

        # Solve: P_bubble(T) - P = 0
        def residual(T, args):
            P_target = args
            P_bubble = self.thermo.bubble_pressure(x, T)
            return P_bubble - P_target

        solver = optx.Newton(rtol=1e-8, atol=1e-8)
        T0 = jnp.array(T_guess)
        sol = optx.root_find(residual, solver, T0, args=P, max_steps=50, throw=False)

        return jnp.clip(sol.value, 150.0, 700.0)

    def dew_point_temperature(
        self,
        inlet: Stream,
        P: Array | float | None = None,
        T_guess: float = 350.0,
    ) -> Array:
        """Calculate dew point temperature at given pressure.

        Solves: 1/sum(y_i / Psat_i(T)) = P

        Args:
            inlet: Stream defining composition
            P: Pressure (Pa). If None, uses inlet P.
            T_guess: Initial temperature guess (K)

        Returns:
            Dew point temperature (K)
        """
        P = jnp.asarray(P) if P is not None else inlet["P"]

        inlet_flows = get_flows(inlet)
        F_total = sum(inlet_flows.values())
        y = {s: inlet_flows[s] / F_total for s in self.params.species_order}

        # Solve: P_dew(T) - P = 0
        def residual(T, args):
            P_target = args
            P_dew = self.thermo.dew_pressure(y, T)
            return P_dew - P_target

        solver = optx.Newton(rtol=1e-8, atol=1e-8)
        T0 = jnp.array(T_guess)
        sol = optx.root_find(residual, solver, T0, args=P, max_steps=50, throw=False)

        return jnp.clip(sol.value, 150.0, 700.0)

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
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Split a stream.

        Args:
            inlet: Feed stream
            split_frac: Fraction of feed going to first outlet (0 to 1)

        Returns:
            outlet1: First outlet stream (split_frac of feed)
            outlet2: Second outlet stream (1 - split_frac of feed)
            info: Dictionary with split information
        """
        split_frac = jnp.asarray(split_frac)
        inlet_flows = get_flows(inlet)

        flows1 = {s: inlet_flows[s] * split_frac for s in self.species_order}
        flows2 = {s: inlet_flows[s] * (1 - split_frac) for s in self.species_order}

        outlet1 = make_stream(flows1, inlet["T"], inlet["P"])
        outlet2 = make_stream(flows2, inlet["T"], inlet["P"])

        total_flow = sum(inlet_flows.values())
        info = {
            "split_fraction": split_frac,
            "flow_to_outlet1": total_flow * split_frac,
            "flow_to_outlet2": total_flow * (1 - split_frac),
        }

        return outlet1, outlet2, info

    def eo_residuals(
        self,
        inlets: list[Stream],
        outlets: list[Stream],
        split_frac: Array | float = 0.5,
        **kwargs,
    ) -> Array:
        """Compute residuals for the EO solver.

        Residuals:
            F_out1_i - split * F_in_i = 0      (n_species)
            F_out2_i - (1-split) * F_in_i = 0   (n_species)
            T_out1 - T_in = 0                    (1)
            T_out2 - T_in = 0                    (1)
            P_out1 - P_in = 0                    (1)
            P_out2 - P_in = 0                    (1)

        Total: 2*(n_species + 2) residuals

        Args:
            inlets: [inlet_stream]
            outlets: [outlet1, outlet2]
            split_frac: Split fraction for first outlet

        Returns:
            Flat residual array
        """
        split_frac = jnp.asarray(split_frac)
        inlet = inlets[0]
        out1 = outlets[0]
        out2 = outlets[1]

        inlet_flows = get_flows(inlet)
        out1_flows = get_flows(out1)
        out2_flows = get_flows(out2)

        resid_out1 = []
        resid_out2 = []
        for s in self.species_order:
            resid_out1.append(jnp.atleast_1d(
                out1_flows[s] - split_frac * inlet_flows[s]
            ))
            resid_out2.append(jnp.atleast_1d(
                out2_flows[s] - (1 - split_frac) * inlet_flows[s]
            ))

        T_resid1 = jnp.atleast_1d(out1["T"] - inlet["T"])
        P_resid1 = jnp.atleast_1d(out1["P"] - inlet["P"])
        T_resid2 = jnp.atleast_1d(out2["T"] - inlet["T"])
        P_resid2 = jnp.atleast_1d(out2["P"] - inlet["P"])

        return jnp.concatenate(
            resid_out1 + [T_resid1, P_resid1]
            + resid_out2 + [T_resid2, P_resid2]
        )


class Mixer:
    """Stream mixer.

    Combines multiple streams. For ideal mixing, outlet enthalpy
    equals sum of inlet enthalpies.
    """

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
                    phase=self.phase
                )

            # Estimate T from enthalpy (simplified - assume Cp is constant)
            total_flow = sum(outlet_flows.values())
            mole_fracs = {s: outlet_flows[s] / total_flow for s in self.species_order}

            # Use average inlet T as starting point
            T_avg = sum(inlet["T"] for inlet in inlets) / len(inlets)
            Cp_mix = self.thermo.Cp_mix(mole_fracs, T_avg)

            # H = n * Cp * (T - Tref), solve for T
            # This is approximate; could use Newton iteration for accuracy
            H_ref = self.thermo.stream_enthalpy(outlet_flows, 298.15, phase=self.phase)
            T_out = 298.15 + (H_total - H_ref) / (total_flow * Cp_mix)
        else:
            # Flow-weighted average temperature
            total_flow = sum(outlet_flows.values())
            T_out = sum(
                sum(get_flows(inlet).values()) * inlet["T"]
                for inlet in inlets
            ) / total_flow

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
            F_in_total = sum(inlet[f"F_{s}"] for inlet in inlets)
            mat_resid.append(jnp.atleast_1d(outlet_flows[s] - F_in_total))

        # Temperature: flow-weighted average (or energy balance with thermo)
        if self.thermo is not None:
            # Compute expected mixed temperature
            mixed_stream, _ = self(*inlets)
            T_expected = mixed_stream["T"]
        else:
            total_flow = sum(outlet_flows.values())
            T_expected = sum(
                sum(get_flows(inlet).values()) * inlet["T"]
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
    """

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
    ) -> tuple[Stream, Stream, dict[str, Array]]:
        """Perform PH (isenthalpic) flash calculation.

        Args:
            inlet: Feed stream
            P: Flash pressure (Pa). If None, uses inlet P.
            T_guess: Initial temperature guess (K)

        Returns:
            liquid: Liquid outlet stream
            vapor: Vapor outlet stream
            info: Dictionary with additional information including T_flash
        """
        p = self.params

        # Use inlet P if not specified
        P_flash = jnp.asarray(P) if P is not None else inlet["P"]

        # Get feed flows and compute inlet enthalpy
        inlet_flows = get_flows(inlet)
        T_in = inlet["T"]

        # Inlet enthalpy (assume liquid feed for simplicity)
        H_inlet = self.thermo.stream_enthalpy(
            {s: inlet_flows[s] for s in p.species_order},
            T_in,
            phase="liquid"
        )

        # Define enthalpy residual function
        def H_residual(T, args):
            P_op, H_target = args

            # Perform TP flash at trial temperature
            liquid, vapor, _ = self._tp_flash(inlet, T=T, P=P_op)

            # Get outlet flows
            liquid_flows = get_flows(liquid)
            vapor_flows = get_flows(vapor)

            # Calculate outlet enthalpy
            H_liquid = self.thermo.stream_enthalpy(
                {s: liquid_flows[s] for s in p.species_order},
                T,
                phase="liquid"
            )
            H_vapor = self.thermo.stream_enthalpy(
                {s: vapor_flows[s] for s in p.species_order},
                T,
                phase="vapor"
            )
            H_outlet = H_liquid + H_vapor

            return H_outlet - H_target

        # Solve for flash temperature
        # Use inlet T as initial guess if T_guess not specified sensibly
        T0 = jnp.where(T_guess > 0, jnp.array(T_guess), T_in)
        solver = optx.Newton(rtol=1e-6, atol=1e-6)
        args = (P_flash, H_inlet)
        sol = optx.root_find(H_residual, solver, T0, args=args, max_steps=100, throw=False)

        # Clip to reasonable temperature bounds
        T_flash = jnp.clip(sol.value, 150.0, 700.0)

        # Final flash at converged temperature
        liquid, vapor, flash_info = self._tp_flash(inlet, T=T_flash, P=P_flash)

        # Add PH-specific info
        flash_info["T_flash"] = T_flash
        flash_info["H_inlet"] = H_inlet
        flash_info["T_inlet"] = T_in
        flash_info["P_flash"] = P_flash
        flash_info["converged"] = sol.result == optx.RESULTS.successful

        return liquid, vapor, flash_info
