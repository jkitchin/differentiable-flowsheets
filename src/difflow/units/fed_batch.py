"""Fed-Batch Reactor unit operation.

This module provides a general-purpose fed-batch reactor for chemical reactions,
distinct from the bio-specific FedBatchBioreactor in difflow_bio.

The fed-batch reactor solves dynamic material and energy balances:
    d(V*C_i)/dt = F_in*C_in_i + V*sum_j(nu_ij*r_j)
    d(V*rho*Cp*T)/dt = F_in*rho_in*Cp_in*T_in + V*sum_j(r_j*(-dH_rxn_j)) + Q

where:
    V = reactor volume (m³, time-varying)
    C = concentrations (mol/m³)
    r = reaction rates (mol/m³/s)
    Q = heat duty (W)
    F_in = inlet volumetric flow rate (m³/s)

All calculations are JAX-compatible for automatic differentiation.
"""

from typing import Callable, Literal, Any
from dataclasses import dataclass

from difflow.params_mixin import ParamsMixin
import jax
import jax.numpy as jnp
from jax import Array, lax
import numpy as np
import optimistix as optx

from difflow.streams import Stream, get_flows, make_stream
from difflow.thermo import IdealThermo
from difflow.dynamic.state import StateSpec, StateVar
from difflow.numerics import safe_divide

# Check for diffrax availability
try:
    from difflow.dynamic.diffrax_backend import integrate_diffrax, HAS_DIFFRAX
except ImportError:
    HAS_DIFFRAX = False


# Type alias for rate function
RateFunction = Callable[[dict[str, Array], Array, dict], Array]

# Type alias for feed profile function
FeedProfile = Callable[[Array], Array]  # t -> F(t)

# Type alias for parameters
Params = dict[str, Any]


@dataclass(repr=False)
class FedBatchParams(ParamsMixin):
    """Parameters for a fed-batch reactor.

    Attributes:
        V0: Initial reactor volume (m³)
        rate_fn: Function computing reaction rates.
                 Signature: rate_fn(C, T, rate_params) -> r
                 where C is dict of concentrations (mol/m³),
                 T is temperature (K),
                 rate_params is user-defined parameters,
                 r is array of reaction rates (mol/m³/s)
        stoich: Stoichiometry matrix, shape (n_species, n_reactions).
                stoich[i, j] is the coefficient of species i in reaction j
                (negative for reactants, positive for products)
        rate_params: Parameters passed to rate_fn
        species_order: List of species names in order matching stoich rows
        dH_rxn: Heats of reaction (J/mol) for each reaction, shape (n_reactions,).
                Negative for exothermic. If None, assumes isothermal.
    """
    V0: float | Array
    rate_fn: RateFunction
    stoich: Array
    rate_params: dict
    species_order: list[str]
    dH_rxn: Array | None = None


class FedBatchReactor:
    """Fed-batch reactor with dynamic volume and composition.

    Simulates semi-batch or fed-batch operation by integrating ODEs
    for volume, concentrations, and temperature over time.

    Supports three modes:
    - isothermal: Temperature specified, Q calculated
    - adiabatic: Q = 0, temperature calculated
    - specified_duty: Q specified as function of time

    Uses JAX-compatible ODE integration via lax.scan with RK4.

    All calculations are JAX-compatible for automatic differentiation.
    """

    def __init__(
        self,
        params: FedBatchParams,
        thermo: IdealThermo | None = None,
        mode: Literal["isothermal", "adiabatic", "specified_duty"] = "isothermal",
    ):
        """Initialize fed-batch reactor.

        Args:
            params: Fed-batch reactor parameters
            thermo: Thermodynamic property calculator (required for non-isothermal)
            mode: Energy balance mode
        """
        self.params = params
        self.thermo = thermo
        self.mode = mode

        if mode != "isothermal" and thermo is None:
            raise ValueError("Thermo object required for non-isothermal operation")

    def __call__(
        self,
        C0: dict[str, Array | float],
        T0: Array | float,
        P: Array | float,
        t_final: Array | float,
        feed_rate_fn: FeedProfile | None = None,
        feed_composition: dict[str, Array | float] | None = None,
        feed_T: Array | float | None = None,
        Q_fn: Callable[[Array], Array] | None = None,
        n_steps: int = 100,
        use_diffrax: bool = True,
        diffrax_solver: str = "tsit5",
        rtol: float = 1e-5,
        atol: float = 1e-7,
    ) -> tuple[Stream, dict[str, Array]]:
        """Simulate fed-batch reactor operation.

        Args:
            C0: Initial concentrations by species (mol/m³)
            T0: Initial temperature (K)
            P: Pressure (Pa, assumed constant)
            t_final: Final simulation time (s)
            feed_rate_fn: Function F(t) returning volumetric feed rate (m³/s).
                         If None, batch mode (no feed).
            feed_composition: Feed concentrations by species (mol/m³).
                             Required if feed_rate_fn is provided.
            feed_T: Feed temperature (K). Default is T0.
            Q_fn: Heat duty function Q(t) for specified_duty mode (W).
            n_steps: Number of integration steps.
            use_diffrax: If True and diffrax available, use diffrax for integration.
            diffrax_solver: Diffrax solver name (e.g., "tsit5", "dopri5", "kvaerno5").
            rtol: Relative tolerance for diffrax solver.
            atol: Absolute tolerance for diffrax solver.

        Returns:
            final_stream: Stream representing final reactor contents
            info: Dictionary with time profiles:
                - 't': Time array (s)
                - 'V': Volume profile (m³)
                - 'C': Concentration profiles dict {species: array}
                - 'T': Temperature profile (K)
                - 'Q': Heat duty profile (W) for isothermal mode
                - 'rates': Reaction rate profiles
                - 'conversion': Final conversion of each species
        """
        p = self.params
        n_species = len(p.species_order)

        # Convert inputs to arrays
        T0 = jnp.asarray(T0)
        P = jnp.asarray(P)
        t_final = jnp.asarray(t_final)
        V0 = jnp.asarray(p.V0)

        # Initial concentrations as array
        C0_arr = jnp.array([jnp.asarray(C0[s]) for s in p.species_order])

        # Initial moles
        n0 = V0 * C0_arr

        # Feed setup
        if feed_rate_fn is None:
            feed_rate_fn = lambda t: jnp.array(0.0)
            C_feed = jnp.zeros(n_species)
        else:
            if feed_composition is None:
                raise ValueError("feed_composition required when feed_rate_fn is provided")
            C_feed = jnp.array([jnp.asarray(feed_composition[s]) for s in p.species_order])

        feed_T = jnp.asarray(feed_T) if feed_T is not None else T0

        # Heat duty function
        if self.mode == "specified_duty":
            if Q_fn is None:
                raise ValueError("Q_fn required for specified_duty mode")
        elif Q_fn is None:
            Q_fn = lambda t: jnp.array(0.0)

        # Time discretization
        dt = t_final / n_steps
        t_array = jnp.linspace(0, t_final, n_steps + 1)

        # Initial state vector: [V, n_1, n_2, ..., n_ns, T]
        # (using moles instead of concentrations for better numerics)
        if self.mode == "isothermal":
            y0 = jnp.concatenate([jnp.array([V0]), n0])
        else:
            y0 = jnp.concatenate([jnp.array([V0]), n0, jnp.array([T0])])

        # Capture parameters for closure
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        rate_params = p.rate_params
        dH_rxn = p.dH_rxn
        thermo = self.thermo
        mode = self.mode

        def rhs(y, t, T_spec=None):
            """Right-hand side of ODEs.

            dy/dt = f(y, t)

            For isothermal: y = [V, n_1, ..., n_ns]
            For non-isothermal: y = [V, n_1, ..., n_ns, T]
            """
            V = y[0]
            n = y[1:n_species+1]

            # Concentrations
            C = n / jnp.maximum(V, 1e-10)
            C_dict = {s: C[i] for i, s in enumerate(species_order)}

            # Temperature
            if mode == "isothermal":
                T = T_spec if T_spec is not None else T0
            else:
                T = y[n_species + 1]

            # Feed rate at current time
            F_in = feed_rate_fn(t)

            # Reaction rates
            r = rate_fn(C_dict, T, rate_params)

            # Material balances: dn_i/dt = F_in*C_feed_i + V*sum_j(nu_ij*r_j)
            dn_dt = F_in * C_feed + V * (stoich @ r)

            # Volume balance: dV/dt = F_in (assuming incompressible)
            dV_dt = F_in

            if mode == "isothermal":
                return jnp.concatenate([jnp.array([dV_dt]), dn_dt])

            # Energy balance for non-isothermal
            # d(n_total * Cp_mix * T)/dt = F_in*Cp_feed*(T_feed - T) + V*sum_j(r_j*(-dH_rxn_j)) + Q
            # Simplified: rho*V*Cp * dT/dt = F_in*rho*Cp*(T_feed - T) + ...

            n_total = jnp.sum(n)
            x = n / jnp.maximum(n_total, 1e-10)
            mole_fracs = {s: x[i] for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            # Heat of reaction
            if dH_rxn is not None:
                Q_rxn = -V * jnp.sum(r * dH_rxn)  # Positive for exothermic
            else:
                Q_rxn = 0.0

            # Feed enthalpy contribution
            n_feed_rate = F_in * jnp.sum(C_feed)
            Q_feed = n_feed_rate * Cp_mix * (feed_T - T)

            # External heat
            if mode == "specified_duty":
                Q_ext = Q_fn(t)
            else:  # adiabatic
                Q_ext = 0.0

            # dT/dt
            dT_dt = safe_divide(Q_rxn + Q_feed + Q_ext, n_total * Cp_mix)

            return jnp.concatenate([jnp.array([dV_dt]), dn_dt, jnp.array([dT_dt])])

        # Diffrax-compatible RHS wrapper (f(t, y) -> dy/dt)
        def diffrax_rhs(t, y):
            return rhs(y, t, T0)

        # Choose integration method
        if use_diffrax and HAS_DIFFRAX:
            # Use diffrax for integration
            result = integrate_diffrax(
                diffrax_rhs,
                y0,
                t_span=(0.0, float(t_final)),
                solver=diffrax_solver,
                rtol=rtol,
                atol=atol,
                saveat=t_array,
            )
            y_all = result.trajectory.y
            # Ensure positivity
            y_all = jnp.maximum(y_all, 1e-20)
        else:
            # Fallback to RK4 via lax.scan
            def rk4_step(y, t):
                """Fourth-order Runge-Kutta step."""
                k1 = rhs(y, t, T0)
                k2 = rhs(y + 0.5*dt*k1, t + 0.5*dt, T0)
                k3 = rhs(y + 0.5*dt*k2, t + 0.5*dt, T0)
                k4 = rhs(y + dt*k3, t + dt, T0)
                y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
                # Ensure positivity
                y_new = jnp.maximum(y_new, 1e-20)
                return y_new, y

            # Integrate using lax.scan
            y_final_rk4, y_history = lax.scan(rk4_step, y0, t_array[:-1])

            # Append final state
            y_all = jnp.vstack([y_history, y_final_rk4[None, :]])

        # Extract profiles
        V_profile = y_all[:, 0]
        n_profiles = y_all[:, 1:n_species+1]
        C_profiles = n_profiles / jnp.maximum(V_profile[:, None], 1e-10)

        if mode == "isothermal":
            T_profile = jnp.full(n_steps + 1, T0)
        else:
            T_profile = y_all[:, n_species + 1]

        # Calculate Q profile for isothermal mode
        if mode == "isothermal":
            Q_profile = self._compute_Q_profile(
                t_array, V_profile, C_profiles, T0, feed_rate_fn, C_feed, feed_T
            )
        else:
            Q_profile = jnp.zeros(n_steps + 1)

        # Calculate rates at final time
        C_final = {s: C_profiles[-1, i] for i, s in enumerate(p.species_order)}
        T_final = T_profile[-1]
        rates_final = rate_fn(C_final, T_final, rate_params)

        # Conversions (based on limiting reactant)
        n_initial = n0
        n_final = n_profiles[-1]
        conversion = {}
        for i, s in enumerate(p.species_order):
            conversion[s] = jnp.where(
                n_initial[i] > 1e-10,
                (n_initial[i] - n_final[i]) / jnp.maximum(n_initial[i], 1e-10),
                jnp.array(0.0),
            )

        # Create final stream (in moles, not concentrations)
        V_final = V_profile[-1]
        final_flows = {s: n_final[i] for i, s in enumerate(p.species_order)}

        # Convert to molar flow rate (moles / batch time)
        flow_rate_basis = {s: n_final[i] / t_final for i, s in enumerate(p.species_order)}

        final_stream = make_stream(flow_rate_basis, T_final, P)

        # Build info dict
        C_dict_profiles = {s: C_profiles[:, i] for i, s in enumerate(p.species_order)}
        n_dict_profiles = {s: n_profiles[:, i] for i, s in enumerate(p.species_order)}

        info = {
            "t": t_array,
            "V": V_profile,
            "C": C_dict_profiles,
            "n": n_dict_profiles,
            "T": T_profile,
            "Q": Q_profile,
            "rates_final": rates_final,
            "conversion": conversion,
            "V_final": V_final,
            "n_final": {s: n_final[i] for i, s in enumerate(p.species_order)},
            "C_final": C_final,
            "T_final": T_final,
        }

        return final_stream, info

    def _compute_Q_profile(
        self,
        t_array: Array,
        V_profile: Array,
        C_profiles: Array,
        T: Array,
        feed_rate_fn: FeedProfile,
        C_feed: Array,
        feed_T: Array,
    ) -> Array:
        """Compute heat duty profile for isothermal operation."""
        p = self.params
        n_species = len(p.species_order)

        def calc_Q(state, inputs):
            t, V, C = inputs
            C_dict = {s: C[i] for i, s in enumerate(p.species_order)}
            r = p.rate_fn(C_dict, T, p.rate_params)
            F_in = feed_rate_fn(t)

            # Heat from reaction
            if p.dH_rxn is not None:
                Q_rxn = V * jnp.sum(r * p.dH_rxn)  # Heat released
            else:
                Q_rxn = 0.0

            # Heat from feed
            if self.thermo is not None:
                n_total = V * jnp.sum(C)
                x = safe_divide(C, jnp.sum(C))
                mole_fracs = {s: x[i] for i, s in enumerate(p.species_order)}
                Cp_mix = self.thermo.Cp_mix(mole_fracs, T)
                Q_feed = F_in * jnp.sum(C_feed) * Cp_mix * (feed_T - T)
            else:
                Q_feed = 0.0

            # Q required to maintain isothermal = -(heat generated)
            Q = -Q_rxn - Q_feed

            return None, Q

        _, Q_profile = lax.scan(calc_Q, None, (t_array, V_profile, C_profiles))

        return Q_profile

    # =========================================================================
    # DynamicUnit Interface Methods
    # =========================================================================

    def state_spec(self) -> StateSpec:
        """Return specification of state variables for dynamic simulation.

        State variables:
        - V: Reactor volume (m³)
        - n_<species>: Moles of each species (mol)
        - T: Temperature (K) - only for non-isothermal modes

        Returns:
            StateSpec describing all state variables
        """
        p = self.params
        variables = []

        # Volume
        variables.append(StateVar(
            name="V",
            category="volume",
            units="m³",
            description="Reactor volume",
            bounds=(0.0, None),
            scale=p.V0 if isinstance(p.V0, (int, float)) else 1.0,
            initial_value=float(p.V0) if isinstance(p.V0, (int, float)) else 1.0,
        ))

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
        """Compute initial state from provided initial conditions.

        Args:
            inputs: Dictionary that may contain:
                - "C0": Initial concentrations dict {species: mol/m³}
                - "T0": Initial temperature (K)
                - Or an "inlet" stream for composition reference
            params: Optional parameters with C0, T0

        Returns:
            Initial state array [V, n_species..., T?]
        """
        p = self.params
        V0 = jnp.asarray(p.V0)

        # Get initial concentrations
        if params and "C0" in params:
            C0 = params["C0"]
        elif "C0" in inputs:
            C0 = inputs["C0"]
        else:
            # Default: use inlet stream composition or zeros
            inlet = inputs.get("inlet")
            if inlet is not None:
                inlet_flows = get_flows(inlet)
                F_total = sum(inlet_flows.values())
                if F_total > 1e-10:
                    C0 = {s: inlet_flows.get(s, 0.0) / F_total * 50.0
                          for s in p.species_order}  # Assume 50 mol/m³ total
                else:
                    C0 = {s: 0.0 for s in p.species_order}
            else:
                C0 = {s: 0.0 for s in p.species_order}

        # Initial moles
        n0 = jnp.array([V0 * C0.get(s, 0.0) for s in p.species_order])

        # Initial temperature
        if params and "T0" in params:
            T0 = jnp.asarray(params["T0"])
        elif "T0" in inputs:
            T0 = jnp.asarray(inputs["T0"])
        else:
            inlet = inputs.get("inlet")
            T0 = inlet["T"] if inlet is not None else jnp.array(300.0)

        state0 = jnp.concatenate([jnp.array([V0]), n0])

        if self.mode != "isothermal":
            state0 = jnp.concatenate([state0, jnp.array([T0])])

        return state0

    def derivatives(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> Array:
        """Compute time derivatives for dynamic simulation.

        Material balance:
            dV/dt = F_in (volumetric)
            dn_i/dt = F_in * C_in_i + V * sum_j(nu_ij * r_j)

        Energy balance (non-isothermal):
            d(n*Cp*T)/dt = F_in*Cp*(T_in - T) + V*sum_j(r_j*(-dH_j)) + Q

        Args:
            t: Current time
            state: Current state [V, n_species..., T?]
            inputs: Dictionary with feed info or inlet stream
            params: Optional parameters (feed_rate_fn, feed_composition, etc.)

        Returns:
            Array of derivatives [dV/dt, dn/dt..., dT/dt?]
        """
        p = self.params
        species = p.species_order
        n_species = len(species)

        # Extract state
        V = state[0]
        n = state[1:n_species + 1]
        n_total = jnp.sum(n) + 1e-10

        # Concentrations
        C = n / jnp.maximum(V, 1e-10)
        C_dict = {s: C[i] for i, s in enumerate(species)}

        # Temperature
        if self.mode == "isothermal":
            T = params.get("T_spec", 300.0) if params else 300.0
            T = jnp.asarray(T)
        else:
            T = state[n_species + 1]

        # Feed rate and composition
        if params and "feed_rate_fn" in params:
            F_in = params["feed_rate_fn"](t)
            C_feed = jnp.array([
                params.get("feed_composition", {}).get(s, 0.0)
                for s in species
            ])
            T_feed = params.get("feed_T", T)
        else:
            # Check inputs for feed info
            F_in = jnp.array(0.0)  # Default: batch mode
            C_feed = jnp.zeros(n_species)
            T_feed = T

        # Reaction rates
        r = p.rate_fn(C_dict, T, p.rate_params)

        # Volume balance
        dV_dt = F_in

        # Material balance: dn/dt = F_in * C_feed + V * stoich @ r
        dn_dt = F_in * C_feed + V * (p.stoich @ r)

        derivs = jnp.concatenate([jnp.array([dV_dt]), dn_dt])

        if self.mode == "isothermal":
            return derivs

        # Energy balance
        Cp = 75.0  # J/mol/K

        # Heat from reaction
        if p.dH_rxn is not None:
            Q_rxn = -V * jnp.sum(r * p.dH_rxn)
        else:
            Q_rxn = 0.0

        # Heat from feed
        n_feed_rate = F_in * jnp.sum(C_feed)
        Q_feed = n_feed_rate * Cp * (T_feed - T)

        # External heat
        if self.mode == "adiabatic":
            Q_ext = 0.0
        else:
            Q_ext = params.get("Q_ext", 0.0) if params else 0.0

        dT_dt = safe_divide(Q_rxn + Q_feed + Q_ext, n_total * Cp)

        return jnp.concatenate([derivs, jnp.array([dT_dt])])

    def outputs(
        self,
        t: Array,
        state: Array,
        inputs: dict[str, Stream],
        params: Params | None = None,
    ) -> dict[str, Stream]:
        """Compute output information from current state.

        For fed-batch, there's typically no continuous outlet stream.
        Returns reactor contents as a "batch" stream.

        Args:
            t: Current time
            state: Current state
            inputs: Input dictionary
            params: Optional parameters

        Returns:
            Dictionary with "contents" stream representing reactor contents
        """
        p = self.params
        species = p.species_order
        n_species = len(species)

        V = state[0]
        n = state[1:n_species + 1]

        # Express as "equivalent flow" (moles / some basis time)
        # Using 1 hour basis for flow rate
        basis_time = 3600.0
        flows = {s: n[i] / basis_time for i, s in enumerate(species)}

        # Temperature
        if self.mode == "isothermal":
            T = params.get("T_spec", 300.0) if params else 300.0
        else:
            T = state[n_species + 1]

        # Pressure from inputs or default
        inlet = inputs.get("inlet")
        P = inlet["P"] if inlet is not None else 101325.0

        return {
            "contents": make_stream(flows, T, P),
            "V": V,
            "n": {s: n[i] for i, s in enumerate(species)},
            "C": {s: n[i] / V for i, s in enumerate(species)},
        }


class SemiBatchReactor(FedBatchReactor):
    """Semi-batch reactor (alias for FedBatchReactor).

    A semi-batch reactor is a fed-batch reactor where one reactant is added
    gradually to control the reaction rate, heat release, or selectivity.

    This is identical to FedBatchReactor but with a more descriptive name
    for certain applications.
    """
    pass


# =============================================================================
# Utility Functions
# =============================================================================


def batch_time_for_conversion(
    params: FedBatchParams,
    C0: dict[str, Array],
    T: Array,
    target_species: str,
    target_conversion: float,
    thermo: IdealThermo | None = None,
) -> Array:
    """Estimate batch time needed to achieve target conversion.

    Uses simple numerical integration to find batch time.

    Args:
        params: Fed-batch parameters (with V0, rate_fn, etc.)
        C0: Initial concentrations
        T: Reaction temperature (isothermal)
        target_species: Species to track for conversion
        target_conversion: Desired conversion (0-1)
        thermo: Thermodynamic calculator (optional)

    Returns:
        Estimated batch time (s)
    """
    # Create batch reactor (no feed)
    reactor = FedBatchReactor(params, thermo, mode="isothermal")

    # Try increasing batch times until target conversion is reached
    # This is a simple bisection approach

    t_min = 1.0
    t_max = 100000.0  # 100000 seconds = ~28 hours

    def get_conversion(t_batch):
        _, info = reactor(
            C0, T, 101325.0, t_batch,
            n_steps=100,
        )
        return info["conversion"][target_species]

    # Use optimistix bisection for root finding
    # Find t where conversion(t) - target = 0
    def residual(t, args):
        return get_conversion(t) - target_conversion

    solver = optx.Bisection(rtol=1e-4, atol=1e-4, expand_if_necessary=True)
    sol = optx.root_find(
        residual,
        solver,
        jnp.array((t_min + t_max) / 2),  # Initial guess (midpoint)
        args=None,
        options=dict(lower=t_min, upper=t_max),
        max_steps=50,
        throw=False,
    )

    return sol.value


def optimal_feed_profile(
    objective: Literal["max_selectivity", "min_time", "max_yield"],
    params: FedBatchParams,
    C0: dict[str, Array],
    T: Array,
    target_species: str,
    feed_composition: dict[str, Array],
    V_max: float,
    t_max: float,
    n_intervals: int = 10,
    n_sim_steps: int = 100,
) -> tuple[Callable[[Array], Array], float]:
    """Determine optimal feed profile for fed-batch reactor via gradient-based optimization.

    Parameterizes the feed as a piecewise-constant profile over ``n_intervals``
    control intervals and uses JAX automatic differentiation with BFGS to
    optimize the feed rates.

    Args:
        objective: Optimization objective:

            - ``"max_yield"``: maximize total moles of *target_species* at end of batch.
            - ``"max_selectivity"``: maximize ``C_target / C_total`` at end of batch.
            - ``"min_time"``: maximize time-weighted average yield of *target_species*,
              rewarding early production (proxy for minimizing time-to-target).

        params: Reactor parameters (must include ``V0``, ``rate_fn``, ``stoich``,
            ``rate_params``, ``species_order``).
        C0: Initial concentrations by species (mol/m³).
        T: Isothermal reaction temperature (K).
        target_species: Product species to optimize.
        feed_composition: Feed concentrations by species (mol/m³).
        V_max: Maximum final reactor volume (m³).
        t_max: Batch duration (s).
        n_intervals: Number of piecewise-constant feed intervals (control horizon).
        n_sim_steps: ODE integration steps per simulation call.

    Returns:
        ``(feed_fn, t_opt)``: Optimized piecewise-constant feed-rate function
        ``F(t) -> m³/s`` and the batch time ``t_opt`` (equal to ``t_max``).
    """
    V0 = float(params.V0)

    # Trivial case: no headroom to add feed
    if V_max <= V0 + 1e-10:
        def _no_feed(t):
            return jnp.array(0.0)
        return _no_feed, t_max

    dt_interval = t_max / n_intervals

    def make_feed_fn(feed_rates_arr: Array) -> Callable[[Array], Array]:
        """Build piecewise-constant feed-rate function from an array of rates."""
        def feed_fn(t: Array) -> Array:
            idx = jnp.clip(
                jnp.floor(t / dt_interval).astype(jnp.int32),
                0, n_intervals - 1,
            )
            return feed_rates_arr[idx]
        return feed_fn

    reactor = FedBatchReactor(params, mode="isothermal")

    V_budget = V_max - V0   # maximum volume that can be added

    def _constrained_rates(log_rates: Array) -> Array:
        """Convert unconstrained variables to volume-constraint-satisfying rates.

        Softplus maps to non-negative rates, then a differentiable rescaling
        ensures the total volume added never exceeds V_budget.
        """
        rates_raw = jax.nn.softplus(log_rates)
        V_added = jnp.sum(rates_raw) * dt_interval
        scale = jnp.minimum(1.0, V_budget / (V_added + 1e-10))
        return rates_raw * scale

    def loss(log_rates: Array) -> Array:
        """Loss function: negative objective. Volume constraint is exact."""
        feed_rates = _constrained_rates(log_rates)
        feed_fn = make_feed_fn(feed_rates)

        _, info = reactor(
            C0=C0,
            T0=T,
            P=101325.0,
            t_final=jnp.array(t_max),
            feed_rate_fn=feed_fn,
            feed_composition=feed_composition,
            n_steps=n_sim_steps,
            use_diffrax=False,  # lax.scan gives a fixed computational graph
        )

        if objective == "max_yield":
            obj = -info["n_final"][target_species]
        elif objective == "max_selectivity":
            C_total = sum(info["C_final"].values()) + 1e-10
            obj = -info["C_final"][target_species] / C_total
        else:  # "min_time"
            # Maximize time-weighted average of target moles;
            # earlier time steps get higher weight, rewarding fast production.
            n_tgt = info["n"][target_species]
            t_arr = info["t"]
            weights = (t_max - t_arr) + 1.0   # positive, decreasing
            weights = weights / jnp.sum(weights)
            obj = -jnp.dot(n_tgt, weights)

        return obj

    # Initial guess: inverse-softplus of the constant feed that
    # would exactly fill the remaining volume over t_max.
    F_avg = (V_max - V0) / t_max
    # softplus⁻¹(y) = log(expm1(y));  numerically safe for y > 0
    log_F_init = float(np.log(np.expm1(min(F_avg, 50.0)) + 1e-30))
    x0 = jnp.full(n_intervals, log_F_init)

    def _loss_for_optx(log_rates, _args):
        return loss(log_rates)

    solver = optx.BFGS(rtol=1e-5, atol=1e-5)
    result = optx.minimise(
        _loss_for_optx,
        solver,
        x0,
        args=None,
        max_steps=200,
        throw=False,
    )

    opt_rates = _constrained_rates(result.value)
    opt_feed_fn = make_feed_fn(opt_rates)

    return opt_feed_fn, t_max
