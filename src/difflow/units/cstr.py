"""Continuous Stirred Tank Reactor (CSTR) unit operation.

The CSTR solves steady-state material and energy balances:
    0 = F_in - F_out + V * r  (for each species)
    0 = Q + H_in - H_out + V * sum(r_j * (-dH_rxn_j))  (energy balance)

where:
    F = molar flow rates (mol/s)
    V = reactor volume (m^3)
    r = reaction rates (mol/m^3/s)
    Q = heat duty (W), positive = heat added
    H = enthalpy flow (W)
    dH_rxn = heat of reaction (J/mol)

This module also supports dynamic simulation via the DynamicUnit interface:
    dn_i/dt = F_in_i - F_out_i + V * sum_j(nu_ij * r_j)
    d(n*Cp*T)/dt = F_in*Cp*(T_in - T) + V*sum_j(r_j*(-dH_j)) + Q

The CSTR class implements both the original steady-state __call__ interface
and the new DynamicUnit protocol for unified dynamic modeling.
"""

from typing import Callable, Literal, Any
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo
from difflow.dynamic.state import StateSpec, StateVar, StateVector


# Type alias for rate function
RateFunction = Callable[[dict[str, Array], Array, dict], Array]

# Type alias for parameters dict
Params = dict[str, Any]


@dataclass
class CSTRParams:
    """Parameters for a CSTR.

    Attributes:
        V: Reactor volume (m^3)
        rate_fn: Function computing reaction rates.
                 Signature: rate_fn(C, T, rate_params) -> r
                 where C is dict of concentrations (mol/m^3),
                 T is temperature (K),
                 rate_params is user-defined parameters,
                 r is array of reaction rates (mol/m^3/s)
        stoich: Stoichiometry matrix, shape (n_species, n_reactions).
                stoich[i, j] is the coefficient of species i in reaction j
                (negative for reactants, positive for products)
        rate_params: Parameters passed to rate_fn (can be any pytree)
        species_order: List of species names in order matching stoich rows
        dH_rxn: Heats of reaction (J/mol) for each reaction, shape (n_reactions,).
                Negative for exothermic. If None, assumes isothermal or uses
                thermo object for calculation.
        T_damping: Damping factor for temperature updates in adiabatic/specified_duty
                   modes (0 < T_damping <= 1). Smaller values = more stable but slower.
                   Default 0.3.
    """
    V: float | Array
    rate_fn: RateFunction
    stoich: Array
    rate_params: dict
    species_order: list[str]
    dH_rxn: Array | None = None
    T_damping: float = 0.3


class CSTR:
    """Continuous Stirred Tank Reactor with multiple reactions.

    Supports three energy balance modes:
    - isothermal: Temperature specified, Q calculated
    - adiabatic: Q = 0, temperature calculated
    - specified_duty: Q specified, temperature calculated

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(
        self,
        params: CSTRParams,
        thermo: IdealThermo | None = None,
        mode: Literal["isothermal", "adiabatic", "specified_duty"] = "isothermal",
    ):
        """Initialize CSTR.

        Args:
            params: CSTR parameters
            thermo: Thermodynamic property calculator (required for non-isothermal)
            mode: Energy balance mode
        """
        self.params = params
        self.thermo = thermo
        self.mode = mode

        if mode != "isothermal" and thermo is None:
            raise ValueError("Thermo object required for non-isothermal operation")

    def _compute_concentrations(
        self,
        flows: dict[str, Array],
        total_volumetric_flow: Array,
    ) -> dict[str, Array]:
        """Compute concentrations from molar flows.

        Args:
            flows: Molar flow rates (mol/s)
            total_volumetric_flow: Total volumetric flow (m^3/s)

        Returns:
            Concentrations (mol/m^3)
        """
        return {
            species: F / total_volumetric_flow
            for species, F in flows.items()
        }

    def __call__(
        self,
        inlet: Stream,
        T_spec: Array | float | None = None,
        Q_spec: Array | float | None = None,
        volumetric_flow: Array | float | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Solve CSTR material and energy balances.

        Args:
            inlet: Inlet stream
            T_spec: Specified outlet temperature (K) for isothermal mode
            Q_spec: Specified heat duty (W) for specified_duty mode
            volumetric_flow: Volumetric flow rate (m^3/s). If None, assumes
                            incompressible liquid with density 1000 kg/m^3.

        Returns:
            outlet: Outlet stream
            info: Dictionary with additional information:
                - 'Q': Heat duty (W)
                - 'rates': Reaction rates (mol/m^3/s)
                - 'conversion': Conversion of each species
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        # Determine volumetric flow
        if volumetric_flow is None:
            # Approximate as total molar flow / 50 for liquid (rough estimate)
            total_molar = sum(inlet_flows.values())
            volumetric_flow = total_molar / 50.0  # mol/s / (mol/m^3) = m^3/s

        volumetric_flow = jnp.asarray(volumetric_flow)

        # Compute outlet based on mode
        if self.mode == "isothermal":
            if T_spec is None:
                T_spec = inlet["T"]
            T_out = jnp.asarray(T_spec)
            outlet_flows, rates = self._solve_material_balance(
                inlet_flows, T_out, volumetric_flow
            )
            Q = self._compute_heat_duty(inlet, outlet_flows, T_out, rates)

        elif self.mode == "adiabatic":
            Q = jnp.asarray(0.0)
            outlet_flows, T_out, rates = self._solve_adiabatic(
                inlet, volumetric_flow
            )

        elif self.mode == "specified_duty":
            if Q_spec is None:
                raise ValueError("Q_spec required for specified_duty mode")
            Q = jnp.asarray(Q_spec)
            outlet_flows, T_out, rates = self._solve_with_duty(
                inlet, Q, volumetric_flow
            )

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Build outlet stream
        outlet = make_stream(outlet_flows, T_out, inlet["P"])

        # Calculate conversions (guard against zero inlet flow)
        conversion = {
            species: jnp.where(
                inlet_flows[species] > 0,
                (inlet_flows[species] - outlet_flows[species]) / (inlet_flows[species] + 1e-30),
                0.0
            )
            for species in p.species_order
        }

        info = {
            "Q": Q,
            "rates": rates,
            "conversion": conversion,
        }

        return outlet, info

    def _solve_material_balance(
        self,
        inlet_flows: dict[str, Array],
        T: Array,
        volumetric_flow: Array,
    ) -> tuple[dict[str, Array], Array]:
        """Solve material balance for given temperature.

        Material balance: F_out = F_in + V * sum_j(nu_ij * r_j)

        Uses Newton-Raphson with implicit differentiation for accurate
        gradients through the converged solution.

        Args:
            inlet_flows: Inlet molar flows by species (mol/s)
            T: Temperature (K)
            volumetric_flow: Volumetric flow rate (m^3/s)

        Returns:
            outlet_flows: Outlet molar flows by species
            rates: Reaction rates (mol/m^3/s)
        """
        p = self.params

        # Compute outlet concentrations from outlet flows
        # This requires solving implicitly since rates depend on C_out
        # For a CSTR: C_out = F_out / volumetric_flow
        #             F_out = F_in + V * stoich @ r(C_out, T)

        # Convert inlet flows to array
        F_in = jnp.array([inlet_flows[s] for s in p.species_order])

        # Use optimistix Newton solver for root finding
        import optimistix as optx

        # Capture non-JAX types in closure, pass only JAX arrays as args
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich

        def material_balance_residual(F_out, args):
            """Residual: F_out - F_in - V * stoich @ r = 0"""
            F_in_, V_, Q_v, T_, rate_params = args

            # Ensure non-negative flows for concentration calculation
            F_out_safe = jnp.maximum(F_out, 1e-10)

            # Concentrations from flows
            C = {s: F_out_safe[i] / Q_v for i, s in enumerate(species_order)}

            # Reaction rates
            r = rate_fn(C, T_, rate_params)

            # Residual: F_out - (F_in + V * stoich @ r) = 0
            residual = F_out - (F_in_ + V_ * stoich @ r)

            return residual

        # Only pass JAX-compatible args (arrays and dicts of arrays)
        args = (F_in, p.V, volumetric_flow, T, p.rate_params)

        solver = optx.Newton(rtol=1e-10, atol=1e-10)
        sol = optx.root_find(
            material_balance_residual,
            solver,
            F_in,  # Initial guess
            args=args,
            max_steps=50,
            throw=False,
        )
        F_out = sol.value

        # Ensure non-negative flows
        F_out = jnp.maximum(F_out, 0.0)

        # Calculate final rates
        C_out = {s: F_out[i] / volumetric_flow for i, s in enumerate(p.species_order)}
        rates = p.rate_fn(C_out, T, p.rate_params)

        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}

        return outlet_flows, rates

    def _compute_heat_duty(
        self,
        inlet: Stream,
        outlet_flows: dict[str, Array],
        T_out: Array,
        rates: Array,
    ) -> Array:
        """Compute heat duty from energy balance.

        Q = H_out - H_in + V * sum_j(r_j * dH_rxn_j)
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        if self.thermo is None:
            # Without thermo, use simplified calculation
            if p.dH_rxn is None:
                return jnp.asarray(0.0)

            # Heat from reactions only
            Q_rxn = p.V * jnp.sum(rates * p.dH_rxn)
            return -Q_rxn  # Convention: Q positive = heat added

        # Full energy balance with enthalpy
        H_in = self.thermo.stream_enthalpy(inlet_flows, inlet["T"], phase="liquid")
        H_out = self.thermo.stream_enthalpy(outlet_flows, T_out, phase="liquid")

        if p.dH_rxn is not None:
            Q_rxn = p.V * jnp.sum(rates * p.dH_rxn)
        else:
            Q_rxn = jnp.asarray(0.0)

        # Q = H_out - H_in + Q_rxn (heat of reaction, exothermic negative)
        Q = H_out - H_in + Q_rxn

        return Q

    def _solve_adiabatic(
        self,
        inlet: Stream,
        volumetric_flow: Array,
    ) -> tuple[dict[str, Array], Array, Array]:
        """Solve for outlet temperature and composition with Q=0.

        This requires simultaneous solution of material and energy balances.
        Uses an outer loop on temperature with inner Newton solve for material balance.
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        import optimistix as optx

        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        T_inlet = inlet["T"]

        # Capture non-JAX types in closure
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        thermo = self.thermo
        dH_rxn = p.dH_rxn
        V = p.V
        T_damping = p.T_damping

        def solve_material_balance_at_T(T, F_in_, Q_v, rate_params):
            """Solve material balance at fixed T using Newton's method."""
            def residual(F_out, args):
                F_in_inner, V_inner, Q_v_inner, T_inner, rp = args
                F_out_safe = jnp.maximum(F_out, 1e-10)
                C = {s: F_out_safe[i] / Q_v_inner for i, s in enumerate(species_order)}
                r = rate_fn(C, T_inner, rp)
                return F_out - (F_in_inner + V_inner * stoich @ r)

            args = (F_in_, V, Q_v, T, rate_params)
            solver = optx.Newton(rtol=1e-10, atol=1e-10)
            sol = optx.root_find(residual, solver, F_in_, args=args, max_steps=50, throw=False)
            return jnp.maximum(sol.value, 0.0)

        def adiabatic_T_iteration(T, args):
            """Fixed-point iteration on temperature only."""
            F_in_, Q_v, rate_params = args

            # Solve material balance at current T
            F_out = solve_material_balance_at_T(T, F_in_, Q_v, rate_params)

            # Compute reaction rates at solution
            C_out = {s: F_out[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C_out, T, rate_params)

            # Energy balance: Q = 0 for adiabatic
            # 0 = H_out - H_in + Q_rxn
            # H_out = H_in - Q_rxn
            inlet_fl = {s: F_in_[i] for i, s in enumerate(species_order)}
            outlet_fl = {s: F_out[i] for i, s in enumerate(species_order)}

            H_in = thermo.stream_enthalpy(inlet_fl, T_inlet, phase="liquid")

            if dH_rxn is not None:
                Q_rxn = V * jnp.sum(r * dH_rxn)
            else:
                Q_rxn = 0.0

            # Target enthalpy for outlet
            H_out_target = H_in - Q_rxn

            # Find T that gives this enthalpy
            # H_out = sum(F_i * Cp_i * (T - Tref)) approximately
            # Use current T to estimate Cp
            total_F = jnp.sum(F_out) + 1e-10
            mole_fracs = {s: F_out[i] / total_F for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            # Current H_out
            H_out_current = thermo.stream_enthalpy(outlet_fl, T, phase="liquid")

            # Update T: H_out_target = H_out_current + total_F * Cp * (T_new - T)
            # T_new = T + (H_out_target - H_out_current) / (total_F * Cp)
            dT = (H_out_target - H_out_current) / (total_F * Cp_mix + 1e-10)
            T_new = T + T_damping * dT  # Damped update for stability

            return T_new

        # Solve for temperature using fixed-point iteration
        args = (F_in, volumetric_flow, p.rate_params)
        fp_solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
        sol = optx.fixed_point(
            adiabatic_T_iteration,
            fp_solver,
            jnp.array(T_inlet),
            args=args,
            max_steps=200,
            throw=False,
        )
        T_out = sol.value

        # Final material balance solve at converged temperature
        F_out = solve_material_balance_at_T(T_out, F_in, volumetric_flow, p.rate_params)
        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}

        C_out = {s: F_out[i] / volumetric_flow for i, s in enumerate(p.species_order)}
        rates = p.rate_fn(C_out, T_out, p.rate_params)

        return outlet_flows, T_out, rates

    def _solve_with_duty(
        self,
        inlet: Stream,
        Q: Array,
        volumetric_flow: Array,
    ) -> tuple[dict[str, Array], Array, Array]:
        """Solve for outlet temperature and composition with specified Q.

        Similar to adiabatic but with Q != 0 in energy balance.
        Uses an outer loop on temperature with inner Newton solve for material balance.
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        import optimistix as optx

        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        T_inlet = inlet["T"]

        # Capture non-JAX types in closure
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        thermo = self.thermo
        dH_rxn = p.dH_rxn
        V = p.V
        T_damping = p.T_damping

        def solve_material_balance_at_T(T, F_in_, Q_v, rate_params):
            """Solve material balance at fixed T using Newton's method."""
            def residual(F_out, args):
                F_in_inner, V_inner, Q_v_inner, T_inner, rp = args
                F_out_safe = jnp.maximum(F_out, 1e-10)
                C = {s: F_out_safe[i] / Q_v_inner for i, s in enumerate(species_order)}
                r = rate_fn(C, T_inner, rp)
                return F_out - (F_in_inner + V_inner * stoich @ r)

            args = (F_in_, V, Q_v, T, rate_params)
            solver = optx.Newton(rtol=1e-10, atol=1e-10)
            sol = optx.root_find(residual, solver, F_in_, args=args, max_steps=50, throw=False)
            return jnp.maximum(sol.value, 0.0)

        def duty_T_iteration(T, args):
            """Fixed-point iteration on temperature only."""
            F_in_, Q_v, Q_spec, rate_params = args

            # Solve material balance at current T
            F_out = solve_material_balance_at_T(T, F_in_, Q_v, rate_params)

            # Compute reaction rates at solution
            C_out = {s: F_out[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C_out, T, rate_params)

            # Energy balance: Q = H_out - H_in + Q_rxn
            # H_out = H_in - Q_rxn + Q
            inlet_fl = {s: F_in_[i] for i, s in enumerate(species_order)}
            outlet_fl = {s: F_out[i] for i, s in enumerate(species_order)}

            H_in = thermo.stream_enthalpy(inlet_fl, T_inlet, phase="liquid")

            if dH_rxn is not None:
                Q_rxn = V * jnp.sum(r * dH_rxn)
            else:
                Q_rxn = 0.0

            # Target enthalpy for outlet (with specified Q)
            H_out_target = H_in - Q_rxn + Q_spec

            # Find T that gives this enthalpy
            total_F = jnp.sum(F_out) + 1e-10
            mole_fracs = {s: F_out[i] / total_F for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            # Current H_out
            H_out_current = thermo.stream_enthalpy(outlet_fl, T, phase="liquid")

            # Update T
            dT = (H_out_target - H_out_current) / (total_F * Cp_mix + 1e-10)
            T_new = T + T_damping * dT  # Damped update for stability

            return T_new

        # Solve for temperature using fixed-point iteration
        args = (F_in, volumetric_flow, Q, p.rate_params)
        fp_solver = optx.FixedPointIteration(rtol=1e-6, atol=1e-6)
        sol = optx.fixed_point(
            duty_T_iteration,
            fp_solver,
            jnp.array(T_inlet),
            args=args,
            max_steps=200,
            throw=False,
        )
        T_out = sol.value

        # Final material balance solve at converged temperature
        F_out = solve_material_balance_at_T(T_out, F_in, volumetric_flow, p.rate_params)
        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}

        C_out = {s: F_out[i] / volumetric_flow for i, s in enumerate(p.species_order)}
        rates = p.rate_fn(C_out, T_out, p.rate_params)

        return outlet_flows, T_out, rates

    # =========================================================================
    # DynamicUnit Interface Methods
    # =========================================================================

    def state_spec(self) -> StateSpec:
        """Return specification of state variables for dynamic simulation.

        State variables:
        - n_<species>: Moles of each species in the reactor (mol)
        - T: Temperature (K) - only for non-isothermal modes

        Returns:
            StateSpec describing all state variables
        """
        p = self.params
        variables = []

        # Molar holdup for each species
        for s in p.species_order:
            variables.append(StateVar(
                name=f"n_{s}",
                category="moles",
                units="mol",
                description=f"Moles of {s} in reactor",
                bounds=(0.0, None),
                scale=1.0,
            ))

        # Temperature for non-isothermal
        if self.mode != "isothermal":
            variables.append(StateVar(
                name="T",
                category="temperature",
                units="K",
                description="Reactor temperature",
                bounds=(200.0, 1000.0),
                scale=300.0,
                initial_value=300.0,
            ))

        return StateSpec(variables)

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial state from inlet streams.

        Initializes molar holdup based on inlet composition and an assumed
        residence time. For startup, the reactor is assumed to contain
        material at inlet composition.

        Args:
            inputs: Dictionary of inlet streams (expects "inlet" key)
            params: Optional parameter overrides

        Returns:
            Initial state array [n_species..., T?]
        """
        p = self.params
        V = p.V

        # Get inlet stream
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)

        # Total inlet flow and composition
        F_total = sum(inlet_flows.values())
        if F_total < 1e-10:
            F_total = 1e-10

        # Estimate residence time (or use provided)
        tau = params.get("tau_init", 60.0) if params else 60.0  # Default 1 min

        # Initial moles = F_in * tau (fills reactor to steady-state holdup)
        n0 = jnp.array([inlet_flows.get(s, 0.0) * tau for s in p.species_order])

        if self.mode != "isothermal":
            T0 = inlet["T"]
            return jnp.concatenate([n0, jnp.array([T0])])

        return n0

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives for dynamic simulation.

        Material balance:
            dn_i/dt = F_in_i - F_out_i + V * sum_j(nu_ij * r_j)

        Energy balance (non-isothermal):
            d(n*Cp*T)/dt = F_in*Cp*(T_in - T) + V*sum_j(r_j*(-dH_j)) + Q

        Args:
            t: Current time (not used, CSTR is autonomous)
            state: Current state array [n_species..., T?]
            inputs: Dictionary of inlet streams
            params: Optional parameter overrides (can include "Q_ext", "T_spec")

        Returns:
            Array of derivatives [dn/dt..., dT/dt?]
        """
        p = self.params
        species = p.species_order
        n_species = len(species)
        V = p.V

        # Extract state variables
        n = state[:n_species]
        n_total = jnp.sum(n) + 1e-10

        # Temperature
        if self.mode == "isothermal":
            # Use inlet T or specified T
            inlet = inputs.get("inlet") or list(inputs.values())[0]
            T = params.get("T_spec", inlet["T"]) if params else inlet["T"]
        else:
            T = state[n_species]

        # Concentrations in reactor
        C = {s: n[i] / V for i, s in enumerate(species)}

        # Get inlet stream
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in = jnp.array([inlet_flows.get(s, 0.0) for s in species])
        F_in_total = jnp.sum(F_in)

        # Outlet flow rate (constant volume assumption: F_out_total = F_in_total)
        # Outlet composition is reactor composition
        x_out = n / n_total
        F_out = F_in_total * x_out

        # Reaction rates
        r = p.rate_fn(C, T, p.rate_params)

        # Material balance: dn/dt = F_in - F_out + V * stoich @ r
        dn_dt = F_in - F_out + V * (p.stoich @ r)

        if self.mode == "isothermal":
            return dn_dt

        # Energy balance for non-isothermal
        # Simplified model: constant Cp, perfect mixing
        Cp = 75.0  # J/mol/K (approximate for liquid)

        # Heat from reaction
        if p.dH_rxn is not None:
            Q_rxn = -V * jnp.sum(r * p.dH_rxn)  # Positive for exothermic
        else:
            Q_rxn = 0.0

        # Heat from flow
        T_in = inlet["T"]
        Q_flow = F_in_total * Cp * (T_in - T)

        # External heat
        if self.mode == "adiabatic":
            Q_ext = 0.0
        else:  # specified_duty
            Q_ext = params.get("Q_ext", 0.0) if params else 0.0

        # dT/dt = (Q_flow + Q_rxn + Q_ext) / (n_total * Cp)
        dT_dt = (Q_flow + Q_rxn + Q_ext) / (n_total * Cp + 1e-10)

        return jnp.concatenate([dn_dt, jnp.array([dT_dt])])

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet stream from current state.

        Maps internal molar holdup to outlet molar flow rates based on
        perfect mixing (outlet composition = reactor composition).

        Args:
            t: Current time
            state: Current state array
            inputs: Dictionary of inlet streams
            params: Optional parameter overrides

        Returns:
            Dictionary with "outlet" stream
        """
        p = self.params
        species = p.species_order
        n_species = len(species)

        # Extract state
        n = state[:n_species]
        n_total = jnp.sum(n) + 1e-10
        x_out = n / n_total

        # Get inlet for flow rate basis
        inlet = inputs.get("inlet") or list(inputs.values())[0]
        inlet_flows = get_flows(inlet)
        F_in_total = sum(inlet_flows.values())

        # Outlet flows (constant volume: F_out = F_in)
        outlet_flows = {s: F_in_total * x_out[i] for i, s in enumerate(species)}

        # Temperature
        if self.mode == "isothermal":
            T_out = params.get("T_spec", inlet["T"]) if params else inlet["T"]
        else:
            T_out = state[n_species]

        # Pressure (constant)
        P = inlet["P"]

        return {"outlet": make_stream(outlet_flows, T_out, P)}

    def residual(
        self,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute residual for steady-state (derivatives = 0).

        This can be used to find steady-state by solving residual = 0.

        Args:
            state: State array
            inputs: Inlet streams
            params: Optional parameters

        Returns:
            Residual array (should be zero at steady state)
        """
        return self.derivatives(jnp.array(0.0), state, inputs, params)
