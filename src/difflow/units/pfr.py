"""Plug Flow Reactor (PFR) unit operation.

The PFR solves the differential material balance along the reactor length:
    dF_i/dV = sum_j(nu_ij * r_j)  (for each species)

For isothermal operation, temperature is constant along the reactor.
For adiabatic operation, the energy balance is also integrated:
    dT/dV = -sum_j(r_j * dH_rxn_j) / (F_total * Cp_mix)

**Gas-phase with pressure drop and mole change:**

For gas-phase reactions where the total number of moles changes, the
volumetric flow rate varies along the reactor:
    Q = Q0 * (F_total / F_total_0) * (P0 / P) * (T / T0)

Pressure drop in packed beds follows the Ergun equation:
    dP/dV = -alpha * (P0/P) * (T/T0) * (F_total/F_total_0)

where alpha is the pressure drop parameter (Pa/m^3).

Symbols:
    F = molar flow rates (mol/s)
    V = reactor volume (m^3) - integration variable
    r = reaction rates (mol/m^3/s)
    dH_rxn = heat of reaction (J/mol)
    Cp_mix = mixture heat capacity (J/mol/K)
    P = pressure (Pa)
    Q = volumetric flow rate (m^3/s)
    alpha = pressure drop parameter

The design equation is integrated from V=0 to V=V_total using diffrax,
a JAX-native differentiable ODE solver library.
"""

from typing import Callable, Literal, Any
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array
import diffrax

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo
from difflow.dynamic.state import StateSpec, StateVar
from difflow.params_mixin import ParamsMixin
from difflow.numerics import safe_divide
from difflow.units.base import (
    estimate_volumetric_flow,
    estimate_residence_time,
    estimate_pfr_conversion,
    estimate_outlet_composition,
    estimate_adiabatic_temperature,
)


# Type alias for rate function
RateFunction = Callable[[dict[str, Array], Array, dict], Array]

# Type alias for parameters
Params = dict[str, Any]


@dataclass(repr=False)
class PFRParams(ParamsMixin):
    """Parameters for a Plug Flow Reactor.

    Attributes:
        V: Total reactor volume (m^3)
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
                Negative for exothermic. Required for non-isothermal operation.
        rtol: Relative tolerance for ODE solver (default 1e-6)
        atol: Absolute tolerance for ODE solver (default 1e-8)
        n_save_points: Number of points to save in profile output (default 101)
    """
    V: float | Array
    rate_fn: RateFunction
    stoich: Array
    rate_params: dict
    species_order: list[str]
    dH_rxn: Array | None = None
    rtol: float = 1e-6
    atol: float = 1e-8
    n_save_points: int = 101

    def __post_init__(self):
        """Validate parameter consistency."""
        n_species = len(self.species_order)
        if self.stoich.shape[0] != n_species:
            raise ValueError(
                f"Stoichiometry matrix has {self.stoich.shape[0]} rows "
                f"but species_order has {n_species} species"
            )
        if self.dH_rxn is not None:
            n_reactions = self.stoich.shape[1]
            if hasattr(self.dH_rxn, 'shape') and self.dH_rxn.shape[0] != n_reactions:
                raise ValueError(
                    f"dH_rxn has {self.dH_rxn.shape[0]} values "
                    f"but stoich has {n_reactions} reactions"
                )


class PFR:
    """Plug Flow Reactor with multiple reactions.

    Supports two energy balance modes:
    - isothermal: Temperature constant along reactor
    - adiabatic: No heat transfer, temperature varies with reaction

    Uses diffrax for ODE integration, providing adaptive stepping and
    proper differentiability through the solver.
    """

    def __init__(
        self,
        params: PFRParams,
        thermo: IdealThermo | None = None,
        mode: Literal["isothermal", "adiabatic"] = "isothermal",
    ):
        """Initialize PFR.

        Args:
            params: PFR parameters
            thermo: Thermodynamic property calculator (required for adiabatic)
            mode: Energy balance mode
        """
        self.params = params
        self.thermo = thermo
        self.mode = mode

        if mode == "adiabatic":
            if thermo is None:
                raise ValueError("Thermo object required for adiabatic operation")
            if params.dH_rxn is None:
                raise ValueError("dH_rxn required for adiabatic operation")

    def __call__(
        self,
        inlet: Stream,
        volumetric_flow: Array | float,
        T_spec: Array | float | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Solve PFR material and energy balances.

        Args:
            inlet: Inlet stream
            volumetric_flow: Volumetric flow rate (m^3/s)
            T_spec: Specified temperature (K) for isothermal mode

        Returns:
            outlet: Outlet stream
            info: Dictionary with additional information:
                - 'conversion': Conversion of each species
                - 'profiles': Profiles along reactor {V, F, T}
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        volumetric_flow = jnp.asarray(volumetric_flow)

        # Initial conditions
        F0 = jnp.array([inlet_flows[s] for s in p.species_order])

        if self.mode == "isothermal":
            T = jnp.asarray(T_spec) if T_spec is not None else inlet["T"]
            F_out, profiles = self._integrate_isothermal(F0, T, volumetric_flow)
            T_out = T
        else:
            T0 = inlet["T"]
            F_out, T_out, profiles = self._integrate_adiabatic(
                F0, T0, volumetric_flow
            )

        # Build outlet stream
        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}
        outlet = make_stream(outlet_flows, T_out, inlet["P"])

        # Calculate conversions
        conversion = {
            s: safe_divide(inlet_flows[s] - outlet_flows[s], inlet_flows[s])
            for s in p.species_order
        }

        info = {
            "conversion": conversion,
            "profiles": profiles,
        }

        return outlet, info

    def _integrate_isothermal(
        self,
        F0: Array,
        T: Array,
        Q_v: Array,
    ) -> tuple[Array, dict]:
        """Integrate material balance for isothermal operation.

        dF/dV = stoich @ r(C, T)

        Uses diffrax ODE solver.
        """
        p = self.params
        V_total = jnp.asarray(p.V)

        # Capture closures
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        rate_params = p.rate_params

        def vector_field(V, F, args):
            """Right-hand side of ODE: dF/dV = stoich @ r"""
            C = {s: F[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)
            return stoich @ r

        # Set up diffrax solver
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, V_total, p.n_save_points))
        stepsize_controller = diffrax.PIDController(rtol=p.rtol, atol=p.atol)

        # Solve the ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=V_total,
            dt0=V_total / 100,
            y0=F0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )

        F_final = solution.ys[-1]
        F_profile = solution.ys
        V_profile = solution.ts

        profiles = {
            "V": V_profile,
            "F": F_profile,
            "T": jnp.broadcast_to(T, (p.n_save_points,)),
        }

        return F_final, profiles

    def _integrate_adiabatic(
        self,
        F0: Array,
        T0: Array,
        Q_v: Array,
    ) -> tuple[Array, Array, dict]:
        """Integrate material and energy balances for adiabatic operation.

        dF/dV = stoich @ r(C, T)
        dT/dV = -sum(r * dH_rxn) / (F_total * Cp_mix)

        Uses diffrax ODE solver.
        """
        p = self.params
        V_total = jnp.asarray(p.V)

        # Capture closures
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        rate_params = p.rate_params
        dH_rxn = p.dH_rxn
        thermo = self.thermo

        def vector_field(V, state, args):
            """Right-hand side: [dF/dV, dT/dV]"""
            F = state[:-1]
            T = state[-1]

            # Concentrations and rates
            C = {s: F[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)

            # Material balance
            dF = stoich @ r

            # Energy balance: dT/dV = -sum(r * dH) / (F_tot * Cp_mix)
            F_total = jnp.sum(F) + 1e-10
            mole_fracs = {s: F[i] / F_total for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            Q_rxn = jnp.sum(r * dH_rxn)  # Heat generated per unit volume
            dT = -Q_rxn / (F_total * Cp_mix / Q_v)  # Divide by molar flow density

            return jnp.concatenate([dF, jnp.array([dT])])

        # Initial state: [F0, T0]
        state0 = jnp.concatenate([F0, jnp.array([T0])])

        # Set up diffrax solver
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, V_total, p.n_save_points))
        stepsize_controller = diffrax.PIDController(rtol=p.rtol, atol=p.atol)

        # Solve the ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=V_total,
            dt0=V_total / 100,
            y0=state0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )

        state_final = solution.ys[-1]
        F_final = state_final[:-1]
        T_final = state_final[-1]

        # Build profiles
        V_profile = solution.ts
        F_profile = solution.ys[:, :-1]
        T_profile = solution.ys[:, -1]

        profiles = {
            "V": V_profile,
            "F": F_profile,
            "T": T_profile,
        }

        return F_final, T_final, profiles

    # =========================================================================
    # DynamicUnit Interface Methods (Pseudo-Steady-State)
    # =========================================================================
    #
    # The PFR is treated as always at spatial steady-state (fast dynamics).
    # The state represents outlet conditions, and derivatives are zero.
    # This is valid when reactor residence time << timescales of interest.

    def state_spec(self) -> StateSpec:
        """Return specification of state variables.

        For pseudo-steady-state PFR, state = outlet flows (for tracking).
        The spatial integration is done internally, not exposed as state.

        Returns:
            StateSpec with outlet flow states
        """
        p = self.params
        variables = []

        for s in p.species_order:
            variables.append(StateVar(
                name=f"F_out_{s}",
                category="moles",
                units="mol/s",
                description=f"Outlet flow of {s}",
                bounds=(0.0, None),
                scale=1.0,
            ))

        if self.mode == "adiabatic":
            variables.append(StateVar(
                name="T_out",
                category="temperature",
                units="K",
                description="Outlet temperature",
                scale=300.0,
            ))

        return StateSpec(variables)

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial state by running the spatial integration.

        Args:
            inputs: Dictionary with "inlet" stream
            params: Optional parameters (T_spec, volumetric_flow)

        Returns:
            Initial state array (outlet flows and T)
        """
        inlet = inputs.get("inlet") or list(inputs.values())[0]

        # Run the PFR calculation
        T_spec = params.get("T_spec") if params else None
        Q_v = params.get("volumetric_flow") if params else None

        # Calculate volumetric flow from total molar flow if not provided
        if Q_v is None:
            inlet_flows = get_flows(inlet)
            total_molar = sum(inlet_flows.values())
            Q_v = total_molar / 50.0  # Assume molar density ~50 mol/m³

        outlet, info = self(inlet, T_spec=T_spec, volumetric_flow=Q_v)

        # Extract outlet state
        outlet_flows = get_flows(outlet)
        F_out = jnp.array([outlet_flows[s] for s in self.params.species_order])

        if self.mode == "adiabatic":
            T_out = outlet["T"]
            return jnp.concatenate([F_out, jnp.array([T_out])])

        return F_out

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives (zero for pseudo-steady-state).

        The PFR is assumed to reach spatial steady-state instantaneously.
        For true dynamic PFR simulation, use a discretized model.

        Args:
            t: Current time
            state: Current state array
            inputs: Dictionary of inlet streams
            params: Optional parameters

        Returns:
            Array of zeros (steady state)
        """
        return jnp.zeros_like(state)

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet stream by running spatial integration.

        Note: For pseudo-steady-state, we re-run the spatial integration
        from the current inlet. The 'state' is updated but not directly
        used for output (the outlet depends on current inlet).

        Args:
            t: Current time
            state: Current state (not used directly)
            inputs: Dictionary of inlet streams
            params: Optional parameters

        Returns:
            Dictionary with "outlet" stream
        """
        inlet = inputs.get("inlet") or list(inputs.values())[0]

        T_spec = params.get("T_spec") if params else None
        Q_v = params.get("volumetric_flow") if params else None

        # Calculate volumetric flow from total molar flow if not provided
        if Q_v is None:
            inlet_flows = get_flows(inlet)
            total_molar = sum(inlet_flows.values())
            Q_v = total_molar / 50.0  # Assume molar density ~50 mol/m³

        outlet, info = self(inlet, T_spec=T_spec, volumetric_flow=Q_v)

        return {"outlet": outlet, "profiles": info.get("profiles")}

    # =========================================================================
    # Initialization Interface
    # =========================================================================

    def initialize(
        self,
        inlet: Stream,
        **kwargs,
    ) -> dict[str, Any]:
        """Generate initial guesses for PFR outputs and internal states.

        Uses analytical PFR conversion estimates for faster convergence.

        Args:
            inlet: Inlet stream
            **kwargs: Additional parameters:
                - volumetric_flow: Volumetric flow rate (m^3/s)
                - T_spec: Specified temperature for isothermal mode
                - expected_conversion: Optional hint for expected conversion

        Returns:
            Dictionary containing:
            - 'outlet': Initial guess for outlet stream
            - 'states': Dict with internal state guesses
            - 'info': Additional initialization information
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        # Get volumetric flow using helper
        volumetric_flow = kwargs.get('volumetric_flow')
        if volumetric_flow is None:
            volumetric_flow = estimate_volumetric_flow(inlet_flows)

        # Residence time using helper
        tau = estimate_residence_time(p.V, volumetric_flow)

        # Estimate conversion using helper
        expected_conversion = kwargs.get('expected_conversion')
        if expected_conversion is not None:
            X_est = expected_conversion
        else:
            k_est = p.rate_params.get('k', 0.1)
            X_est = estimate_pfr_conversion(k_est, tau)

        # Estimate outlet composition using helper
        outlet_flows = estimate_outlet_composition(
            inlet_flows, p.stoich, p.species_order, X_est
        )

        # Estimate outlet temperature
        T_spec = kwargs.get('T_spec')
        if self.mode == "isothermal":
            T_out = T_spec if T_spec is not None else float(inlet["T"])
        else:  # adiabatic
            if p.dH_rxn is not None:
                # Find limiting reactant and estimate extent
                reactant_species = [s for j, s in enumerate(p.species_order)
                                    if p.stoich[j, 0] < 0]
                if reactant_species:
                    limiting_species = reactant_species[0]
                    extent_est = inlet_flows.get(limiting_species, 1.0) * X_est / (
                        -float(p.stoich[p.species_order.index(limiting_species), 0])
                    )
                else:
                    extent_est = 0.0

                total_flow = sum(inlet_flows.values())
                T_out = estimate_adiabatic_temperature(
                    inlet["T"], p.dH_rxn, extent_est, total_flow
                )
            else:
                T_out = float(inlet["T"])

        # Build outlet stream guess
        outlet = make_stream(outlet_flows, T_out, inlet["P"])

        states = {
            'conversion': X_est,
            'residence_time': tau,
        }
        if self.mode == "adiabatic":
            states['T_out'] = T_out

        info = {
            'method': 'analytical_pfr_estimate',
            'assumptions': ['first-order kinetics', 'no pressure drop'],
        }

        return {
            'outlet': outlet,
            'states': states,
            'info': info,
        }


@dataclass(repr=False)
class GasPFRParams(ParamsMixin):
    """Parameters for a gas-phase PFR with pressure drop.

    Attributes:
        V: Total reactor volume (m^3)
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
                Negative for exothermic. Required for non-isothermal operation.
        alpha: Pressure drop parameter (Pa/m^3). If None, no pressure drop.
        rtol: Relative tolerance for ODE solver (default 1e-6)
        atol: Absolute tolerance for ODE solver (default 1e-8)
        n_save_points: Number of points to save in profile output (default 101)
    """
    V: float | Array
    rate_fn: RateFunction
    stoich: Array
    rate_params: dict
    species_order: list[str]
    dH_rxn: Array | None = None
    alpha: float | Array | None = None
    rtol: float = 1e-6
    atol: float = 1e-8
    n_save_points: int = 101

    def __post_init__(self):
        """Validate parameter consistency."""
        n_species = len(self.species_order)
        if self.stoich.shape[0] != n_species:
            raise ValueError(
                f"Stoichiometry matrix has {self.stoich.shape[0]} rows "
                f"but species_order has {n_species} species"
            )
        if self.dH_rxn is not None:
            n_reactions = self.stoich.shape[1]
            if hasattr(self.dH_rxn, 'shape') and self.dH_rxn.shape[0] != n_reactions:
                raise ValueError(
                    f"dH_rxn has {self.dH_rxn.shape[0]} values "
                    f"but stoich has {n_reactions} reactions"
                )


class GasPFR:
    """Gas-phase Plug Flow Reactor with pressure drop and mole change.

    Handles:
    - Variable volumetric flow due to mole change: Q = Q0 * (F_tot/F_tot0) * (P0/P) * (T/T0)
    - Pressure drop: dP/dV = -alpha * (P0/P) * (T/T0) * (F_tot/F_tot0)
    - Energy balance for adiabatic operation

    Supports two energy balance modes:
    - isothermal: Temperature constant along reactor
    - adiabatic: No heat transfer, temperature varies with reaction

    Uses diffrax for ODE integration, providing adaptive stepping and
    proper differentiability through the solver.
    """

    def __init__(
        self,
        params: GasPFRParams,
        thermo: IdealThermo | None = None,
        mode: Literal["isothermal", "adiabatic"] = "isothermal",
    ):
        """Initialize GasPFR.

        Args:
            params: GasPFR parameters
            thermo: Thermodynamic property calculator (required for adiabatic)
            mode: Energy balance mode
        """
        self.params = params
        self.thermo = thermo
        self.mode = mode

        if mode == "adiabatic":
            if thermo is None:
                raise ValueError("Thermo object required for adiabatic operation")
            if params.dH_rxn is None:
                raise ValueError("dH_rxn required for adiabatic operation")

    def __call__(
        self,
        inlet: Stream,
        T_spec: Array | float | None = None,
        volumetric_flow: Array | float | None = None,
    ) -> tuple[Stream, dict[str, Array]]:
        """Solve gas-phase PFR with pressure drop and mole change.

        Args:
            inlet: Inlet stream (must include P and T)
            T_spec: Specified temperature (K) for isothermal mode
            volumetric_flow: Inlet volumetric flow rate (m^3/s). If None,
                            uses ideal gas law: Q0 = F_tot * R * T0 / P0

        Returns:
            outlet: Outlet stream
            info: Dictionary with additional information:
                - 'conversion': Conversion of each species
                - 'profiles': Profiles along reactor {V, F, T, P, Q}
                - 'pressure_drop': Total pressure drop (Pa)
        """
        p = self.params
        inlet_flows = get_flows(inlet)
        P0 = inlet["P"]
        T0 = inlet["T"]

        # Initial total molar flow
        F_total_0 = sum(inlet_flows.values())

        # Determine inlet volumetric flow
        if volumetric_flow is None:
            # Ideal gas: Q = F_tot * R * T / P
            R = 8.314  # J/(mol·K)
            volumetric_flow = F_total_0 * R * T0 / P0
        Q0 = jnp.asarray(volumetric_flow)

        # Initial conditions
        F0 = jnp.array([inlet_flows[s] for s in p.species_order])

        if self.mode == "isothermal":
            T = jnp.asarray(T_spec) if T_spec is not None else T0
            F_out, P_out, profiles = self._integrate_isothermal_gas(
                F0, T, P0, Q0, F_total_0
            )
            T_out = T
        else:
            F_out, T_out, P_out, profiles = self._integrate_adiabatic_gas(
                F0, T0, P0, Q0, F_total_0
            )

        # Build outlet stream
        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}
        outlet = make_stream(outlet_flows, T_out, P_out)

        # Calculate conversions
        conversion = {
            s: safe_divide(inlet_flows[s] - outlet_flows[s], inlet_flows[s])
            for s in p.species_order
        }

        info = {
            "conversion": conversion,
            "profiles": profiles,
            "pressure_drop": P0 - P_out,
        }

        return outlet, info

    def _integrate_isothermal_gas(
        self,
        F0: Array,
        T: Array,
        P0: Array,
        Q0: Array,
        F_total_0: Array,
    ) -> tuple[Array, Array, dict]:
        """Integrate material and momentum balances for isothermal gas-phase PFR.

        State: [F_1, F_2, ..., F_n, P]

        dF/dV = stoich @ r(C, T)
        dP/dV = -alpha * (P0/P) * (T/T0) * (F_tot/F_tot0)

        where C = F / Q, and Q = Q0 * (F_tot/F_tot0) * (P0/P) * (T/T0)

        Uses diffrax ODE solver.
        """
        p = self.params
        V_total = jnp.asarray(p.V)

        # Capture closures
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        rate_params = p.rate_params
        alpha = p.alpha if p.alpha is not None else 0.0
        T0 = T  # For isothermal, T = T0

        def vector_field(V, state, args):
            """Right-hand side: [dF/dV, dP/dV]"""
            F = state[:-1]
            P = state[-1]

            # Total molar flow
            F_total = jnp.sum(F) + 1e-10

            # Volumetric flow accounting for mole change and pressure
            # Q = Q0 * (F_tot/F_tot0) * (P0/P) * (T/T0)
            Q = Q0 * (F_total / F_total_0) * safe_divide(P0, P) * (T / T0)

            # Concentrations and rates
            C = {s: F[i] / Q for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)

            # Material balance
            dF = stoich @ r

            # Pressure drop: dP/dV = -alpha * (P0/P) * (T/T0) * (F_tot/F_tot0)
            dP = -alpha * safe_divide(P0, P) * (T / T0) * (F_total / F_total_0)

            return jnp.concatenate([dF, jnp.array([dP])])

        # Initial state: [F0, P0]
        state0 = jnp.concatenate([F0, jnp.array([P0])])

        # Set up diffrax solver
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, V_total, p.n_save_points))
        stepsize_controller = diffrax.PIDController(rtol=p.rtol, atol=p.atol)

        # Solve the ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=V_total,
            dt0=V_total / 100,
            y0=state0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )

        state_final = solution.ys[-1]
        F_final = state_final[:-1]
        P_final = state_final[-1]

        # Build profiles
        V_profile = solution.ts
        F_profile = solution.ys[:, :-1]
        P_profile = solution.ys[:, -1]

        # Compute Q profile
        F_total_profile = jnp.sum(F_profile, axis=1) + 1e-10
        Q_profile = Q0 * (F_total_profile / F_total_0) * safe_divide(P0, P_profile) * (T / T0)

        profiles = {
            "V": V_profile,
            "F": F_profile,
            "T": jnp.broadcast_to(T, (p.n_save_points,)),
            "P": P_profile,
            "Q": Q_profile,
        }

        return F_final, P_final, profiles

    def _integrate_adiabatic_gas(
        self,
        F0: Array,
        T0: Array,
        P0: Array,
        Q0: Array,
        F_total_0: Array,
    ) -> tuple[Array, Array, Array, dict]:
        """Integrate material, energy, and momentum balances for adiabatic gas-phase PFR.

        State: [F_1, F_2, ..., F_n, T, P]

        dF/dV = stoich @ r(C, T)
        dT/dV = -sum(r * dH_rxn) / (F_total * Cp_mix)
        dP/dV = -alpha * (P0/P) * (T/T0) * (F_tot/F_tot0)

        where C = F / Q, and Q = Q0 * (F_tot/F_tot0) * (P0/P) * (T/T0)

        Uses diffrax ODE solver.
        """
        p = self.params
        V_total = jnp.asarray(p.V)

        # Capture closures
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        rate_params = p.rate_params
        dH_rxn = p.dH_rxn
        alpha = p.alpha if p.alpha is not None else 0.0
        thermo = self.thermo
        n_species = len(species_order)

        def vector_field(V, state, args):
            """Right-hand side: [dF/dV, dT/dV, dP/dV]"""
            F = state[:n_species]
            T = state[n_species]
            P = state[n_species + 1]

            # Total molar flow
            F_total = jnp.sum(F) + 1e-10

            # Volumetric flow accounting for mole change, pressure, and temperature
            Q = Q0 * (F_total / F_total_0) * safe_divide(P0, P) * (T / T0)

            # Concentrations and rates
            C = {s: F[i] / Q for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)

            # Material balance
            dF = stoich @ r

            # Energy balance: dT/dV = -sum(r * dH) / (F_tot * Cp_mix / Q)
            mole_fracs = {s: F[i] / F_total for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)
            Q_rxn = jnp.sum(r * dH_rxn)
            dT = -Q_rxn / (F_total * Cp_mix / Q)

            # Pressure drop
            dP = -alpha * safe_divide(P0, P) * (T / T0) * (F_total / F_total_0)

            return jnp.concatenate([dF, jnp.array([dT, dP])])

        # Initial state: [F0, T0, P0]
        state0 = jnp.concatenate([F0, jnp.array([T0, P0])])

        # Set up diffrax solver
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=jnp.linspace(0, V_total, p.n_save_points))
        stepsize_controller = diffrax.PIDController(rtol=p.rtol, atol=p.atol)

        # Solve the ODE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=0.0,
            t1=V_total,
            dt0=V_total / 100,
            y0=state0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=4096,
        )

        state_final = solution.ys[-1]
        F_final = state_final[:n_species]
        T_final = state_final[n_species]
        P_final = state_final[n_species + 1]

        # Build profiles
        V_profile = solution.ts
        F_profile = solution.ys[:, :n_species]
        T_profile = solution.ys[:, n_species]
        P_profile = solution.ys[:, n_species + 1]

        # Compute Q profile
        F_total_profile = jnp.sum(F_profile, axis=1) + 1e-10
        Q_profile = Q0 * (F_total_profile / F_total_0) * safe_divide(P0, P_profile) * (T_profile / T0)

        profiles = {
            "V": V_profile,
            "F": F_profile,
            "T": T_profile,
            "P": P_profile,
            "Q": Q_profile,
        }

        return F_final, T_final, P_final, profiles

    # =========================================================================
    # DynamicUnit Interface Methods (Pseudo-Steady-State)
    # =========================================================================

    def state_spec(self) -> StateSpec:
        """Return specification of state variables.

        For pseudo-steady-state GasPFR, state = outlet flows, T, P.

        Returns:
            StateSpec with outlet flow states
        """
        p = self.params
        variables = []

        for s in p.species_order:
            variables.append(StateVar(
                name=f"F_out_{s}",
                category="moles",
                units="mol/s",
                description=f"Outlet flow of {s}",
                bounds=(0.0, None),
                scale=1.0,
            ))

        variables.append(StateVar(
            name="T_out",
            category="temperature",
            units="K",
            description="Outlet temperature",
            scale=300.0,
        ))

        variables.append(StateVar(
            name="P_out",
            category="pressure",
            units="Pa",
            description="Outlet pressure",
            scale=101325.0,
        ))

        return StateSpec(variables)

    def initial_state(
        self,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute initial state by running the spatial integration."""
        inlet = inputs.get("inlet") or list(inputs.values())[0]

        T_spec = params.get("T_spec") if params else None
        Q_v = params.get("volumetric_flow") if params else None

        # Calculate volumetric flow from ideal gas law if not provided
        if Q_v is None:
            inlet_flows = get_flows(inlet)
            total_molar = sum(inlet_flows.values())
            T = inlet["T"]
            P = inlet["P"]
            R = 8.314  # J/mol/K
            Q_v = total_molar * R * T / P  # Ideal gas volumetric flow

        outlet, info = self(inlet, T_spec=T_spec, volumetric_flow=Q_v)

        outlet_flows = get_flows(outlet)
        F_out = jnp.array([outlet_flows[s] for s in self.params.species_order])
        T_out = outlet["T"]
        P_out = outlet["P"]

        return jnp.concatenate([F_out, jnp.array([T_out, P_out])])

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives (zero for pseudo-steady-state)."""
        return jnp.zeros_like(state)

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute outlet stream by running spatial integration."""
        inlet = inputs.get("inlet") or list(inputs.values())[0]

        T_spec = params.get("T_spec") if params else None
        Q_v = params.get("volumetric_flow") if params else None

        # Calculate volumetric flow from ideal gas law if not provided
        if Q_v is None:
            inlet_flows = get_flows(inlet)
            total_molar = sum(inlet_flows.values())
            T = inlet["T"]
            P = inlet["P"]
            R = 8.314  # J/mol/K
            Q_v = total_molar * R * T / P  # Ideal gas volumetric flow

        outlet, info = self(inlet, T_spec=T_spec, volumetric_flow=Q_v)

        return {"outlet": outlet, "profiles": info.get("profiles")}


def pfr_conversion_analytical(
    k: Array,
    tau: Array,
    order: int = 1,
) -> Array:
    """Analytical PFR conversion for simple kinetics (no pressure drop).

    For a first-order reaction A -> B:
        X = 1 - exp(-k * tau)

    For a second-order reaction 2A -> B:
        X = k * tau * C_A0 / (1 + k * tau * C_A0)

    Args:
        k: Rate constant (1/s for first-order, m^3/mol/s for second-order)
        tau: Residence time V/Q (s)
        order: Reaction order (1 or 2)

    Returns:
        Conversion X
    """
    if order == 1:
        return 1.0 - jnp.exp(-k * tau)
    elif order == 2:
        # For second order, need initial concentration
        # This is a simplified version assuming k*tau*C0 >> 1
        raise NotImplementedError("Second-order analytical solution not implemented")
    else:
        raise ValueError(f"Unsupported reaction order: {order}")
