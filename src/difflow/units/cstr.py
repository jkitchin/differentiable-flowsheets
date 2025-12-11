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
"""

from typing import Callable, Literal
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, get_species, make_stream
from difflow.thermo import IdealThermo


# Type alias for rate function
RateFunction = Callable[[dict[str, Array], Array, dict], Array]


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
    """
    V: float | Array
    rate_fn: RateFunction
    stoich: Array
    rate_params: dict
    species_order: list[str]
    dH_rxn: Array | None = None


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

        # Calculate conversions
        conversion = {
            species: (inlet_flows[species] - outlet_flows[species]) / inlet_flows[species]
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

        # Fixed-point iteration: F_out^{k+1} = F_in + V * stoich @ r(F_out^k / Q_v, T)
        from difflow.solvers import fixed_point_solve

        # Capture non-JAX types in closure, pass only JAX arrays as args
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich

        def material_balance_fp(F_out, args):
            F_in_, V_, Q_v, T_, rate_params = args

            # Concentrations from flows
            C = {s: F_out[i] / Q_v for i, s in enumerate(species_order)}

            # Reaction rates
            r = rate_fn(C, T_, rate_params)

            # New flows from material balance
            F_out_new = F_in_ + V_ * stoich @ r

            # Ensure non-negative flows
            return jnp.maximum(F_out_new, 0.0)

        # Only pass JAX-compatible args (arrays and dicts of arrays)
        args = (F_in, p.V, volumetric_flow, T, p.rate_params)

        F_out = fixed_point_solve(
            material_balance_fp,
            F_in,  # Initial guess
            args,
            tol=1e-10,
            max_iter=100,
            damping=0.3,  # Lower damping for stability with stiff kinetics
        )

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
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        from difflow.solvers import fixed_point_solve

        # State vector: [T, F_1, F_2, ..., F_n]
        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        x0 = jnp.concatenate([jnp.array([inlet["T"]]), F_in])

        # Capture non-JAX types in closure
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        thermo = self.thermo
        dH_rxn = p.dH_rxn
        T_inlet = inlet["T"]

        def adiabatic_fp(x, args):
            F_in_, V_, Q_v, rate_params = args

            T = x[0]
            F_out = x[1:]

            # Material balance
            C = {s: F_out[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)
            F_new = F_in_ + V_ * stoich @ r
            F_new = jnp.maximum(F_new, 0.0)

            # Energy balance: find T such that H_out - H_in + Q_rxn = 0
            inlet_fl = {s: F_in_[i] for i, s in enumerate(species_order)}
            outlet_fl = {s: F_new[i] for i, s in enumerate(species_order)}

            H_in = thermo.stream_enthalpy(inlet_fl, T_inlet, phase="liquid")
            H_out = thermo.stream_enthalpy(outlet_fl, T, phase="liquid")

            if dH_rxn is not None:
                Q_rxn = V_ * jnp.sum(r * dH_rxn)
            else:
                Q_rxn = 0.0

            # Residual for energy balance
            # H_out - H_in + Q_rxn = 0 => H_out = H_in - Q_rxn
            # Update T based on enthalpy difference
            # Use Cp to estimate new T
            total_F = jnp.sum(F_new)
            mole_fracs = {s: F_new[i] / total_F for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            dH = H_in - Q_rxn - H_out  # Should be zero at solution
            dT = dH / (total_F * Cp_mix)  # Correction to T
            T_new = T + 0.5 * dT  # Damped update

            return jnp.concatenate([jnp.array([T_new]), F_new])

        args = (F_in, p.V, volumetric_flow, p.rate_params)

        x_sol = fixed_point_solve(adiabatic_fp, x0, args, tol=1e-8, max_iter=200)

        T_out = x_sol[0]
        F_out = x_sol[1:]
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
        """
        p = self.params
        inlet_flows = get_flows(inlet)

        from difflow.solvers import fixed_point_solve

        F_in = jnp.array([inlet_flows[s] for s in p.species_order])
        x0 = jnp.concatenate([jnp.array([inlet["T"]]), F_in])

        # Capture non-JAX types in closure
        rate_fn = p.rate_fn
        species_order = p.species_order
        stoich = p.stoich
        thermo = self.thermo
        dH_rxn = p.dH_rxn
        T_inlet = inlet["T"]

        def duty_fp(x, args):
            F_in_, V_, Q_v, Q_, rate_params = args

            T = x[0]
            F_out = x[1:]

            # Material balance
            C = {s: F_out[i] / Q_v for i, s in enumerate(species_order)}
            r = rate_fn(C, T, rate_params)
            F_new = F_in_ + V_ * stoich @ r
            F_new = jnp.maximum(F_new, 0.0)

            # Energy balance with specified Q
            inlet_fl = {s: F_in_[i] for i, s in enumerate(species_order)}
            outlet_fl = {s: F_new[i] for i, s in enumerate(species_order)}

            H_in = thermo.stream_enthalpy(inlet_fl, T_inlet, phase="liquid")
            H_out = thermo.stream_enthalpy(outlet_fl, T, phase="liquid")

            if dH_rxn is not None:
                Q_rxn = V_ * jnp.sum(r * dH_rxn)
            else:
                Q_rxn = 0.0

            # Q = H_out - H_in + Q_rxn
            # H_out = H_in - Q_rxn + Q
            total_F = jnp.sum(F_new)
            mole_fracs = {s: F_new[i] / total_F for i, s in enumerate(species_order)}
            Cp_mix = thermo.Cp_mix(mole_fracs, T)

            dH = H_in - Q_rxn + Q_ - H_out
            dT = dH / (total_F * Cp_mix)
            T_new = T + 0.5 * dT

            return jnp.concatenate([jnp.array([T_new]), F_new])

        args = (F_in, p.V, volumetric_flow, Q, p.rate_params)

        x_sol = fixed_point_solve(duty_fp, x0, args, tol=1e-8, max_iter=200)

        T_out = x_sol[0]
        F_out = x_sol[1:]
        outlet_flows = {s: F_out[i] for i, s in enumerate(p.species_order)}

        C_out = {s: F_out[i] / volumetric_flow for i, s in enumerate(p.species_order)}
        rates = p.rate_fn(C_out, T_out, p.rate_params)

        return outlet_flows, T_out, rates
