"""Flash separator unit operation.

The flash separator performs vapor-liquid equilibrium calculations
to split a feed stream into vapor and liquid products.

Currently supports:
- TP flash: Temperature and pressure specified

The VLE is solved using the Rachford-Rice equation with implicit
differentiation through the converged solution.
"""

from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo
from difflow.solvers import rachford_rice, rachford_rice_compositions


@dataclass
class FlashParams:
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

        # Get K-values from thermodynamics
        K = self.thermo.K_values_array(T, P)

        # Check for subcooled liquid or superheated vapor
        # sum(z*K) < 1 => subcooled liquid (all liquid)
        # sum(z/K) < 1 => superheated vapor (all vapor)
        bubble_check = jnp.sum(z * K)
        dew_check = jnp.sum(z / K)

        # Solve Rachford-Rice for vapor fraction
        V_frac = self._solve_flash(z, K, bubble_check, dew_check)

        # Get compositions
        x, y = rachford_rice_compositions(z, K, V_frac)

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
        # Use differentiable conditionals via jnp.where
        # This ensures smooth gradients near phase boundaries

        # Normal two-phase case
        V_two_phase = rachford_rice(z, K)

        # Handle edge cases with smooth transitions
        # If bubble_check < 1, we're subcooled (V → 0)
        # If dew_check < 1, we're superheated (V → 1)

        V_frac = jnp.where(
            bubble_check < 1.0,
            0.0,  # Subcooled liquid
            jnp.where(
                dew_check < 1.0,
                1.0,  # Superheated vapor
                V_two_phase,  # Two-phase region
            )
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
