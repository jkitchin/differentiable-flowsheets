"""Amine absorption column for CO2 capture.

This module provides a simplified equilibrium-stage absorber model
for amine-based CO2 capture. The model uses the Kremser equation
with stage efficiency to calculate CO2 removal.

For rate-based modeling with mass transfer correlations, this can
be extended via subclassing or the extensibility hooks provided.

References:
    Kohl AL, Nielsen RB (1997). Gas Purification, 5th ed.
        Gulf Publishing Company. Chapter 2.
    Onda K et al. (1968). Mass transfer coefficients between
        gas and liquid phases in packed columns.
        J Chem Eng Japan 1:56-62.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, make_stream, get_flows, total_flow
from difflow.params_mixin import ParamsMixin
from difflow_cc.database import get_solvent
from difflow_cc.equilibrium.vle import AmineVLE


# =============================================================================
# Absorber Parameters
# =============================================================================

@dataclass(repr=False)
class AbsorberParams(ParamsMixin):
    """Parameters for amine absorber column.

    Attributes:
        solvent: Amine solvent name (e.g., 'MEA', 'PZ', 'MDEA')
        n_stages: Number of theoretical stages
        solvent_conc: Amine concentration (wt%)
        L_G_ratio: Liquid/gas molar ratio
        T_gas_in: Inlet gas temperature (K)
        T_liquid_in: Inlet liquid temperature (K)
        P_absorber: Operating pressure (Pa)
        stage_efficiency: Murphree stage efficiency (0-1)
        lean_loading: Lean solvent CO2 loading (mol CO2/mol amine)

    Notes:
        The simplified model assumes:
        - Isothermal operation at average temperature
        - Constant L/G ratio throughout column
        - Henry's law for physical solubility
        - Kent-Eisenberg VLE for chemical equilibrium

        For extensibility to rate-based models, additional parameters
        can be added: column_diameter, packing_type, packing_height,
        specific_area, void_fraction, etc.
    """
    solvent: str
    n_stages: int | float | Array = 10
    solvent_conc: float = 30.0  # wt%
    L_G_ratio: float | Array = 3.0  # mol/mol
    T_gas_in: float | Array = 313.15  # K (40°C)
    T_liquid_in: float | Array = 313.15  # K
    P_absorber: float | Array = 101325.0  # Pa (1 atm)
    stage_efficiency: float | Array = 0.25  # Murphree efficiency
    lean_loading: float | Array = 0.2  # mol CO2/mol amine

    # Extensibility for rate-based (not used in simplified model)
    column_diameter: float | None = None  # m
    packing_type: str | None = None
    packing_height: float | None = None  # m


# =============================================================================
# Amine Absorber
# =============================================================================

class AmineAbsorber:
    """Amine absorption column for CO2 capture.

    Simplified equilibrium-stage model using the Kremser equation
    with Murphree stage efficiency.

    The absorber removes CO2 from a gas stream using an amine solvent.
    Key outputs are:
    - Treated gas (CO2 depleted)
    - Rich solvent (CO2 loaded)
    - CO2 capture efficiency

    Example:
        >>> params = AbsorberParams(
        ...     solvent='MEA',
        ...     n_stages=10,
        ...     solvent_conc=30.0,
        ...     L_G_ratio=3.0,
        ... )
        >>> absorber = AmineAbsorber(params)
        >>> treated_gas, rich_solvent, info = absorber(flue_gas, lean_solvent)

    The model is fully differentiable with respect to:
    - Number of stages
    - L/G ratio
    - Temperatures
    - Lean loading
    - Inlet compositions

    This enables gradient-based optimization of capture efficiency
    and energy consumption.

    References:
        Kohl AL, Nielsen RB (1997). Gas Purification, 5th ed.
            Gulf Publishing Company.
        Treybal RE (1980). Mass Transfer Operations, 3rd ed.
            McGraw-Hill. Kremser equation derivation.
    """

    def __init__(self, params: AbsorberParams):
        """Initialize absorber.

        Args:
            params: AbsorberParams dataclass
        """
        self.params = params
        self._solvent_data = get_solvent(params.solvent)
        self._vle = AmineVLE(params.solvent)

    def __call__(
        self,
        gas_in: Stream,
        solvent_in: Stream | None = None,
        T_op: Array | float | None = None,
    ) -> tuple[Stream, Stream, dict]:
        """Perform CO2 absorption.

        Args:
            gas_in: Inlet gas stream (must contain F_CO2, F_N2 or other inerts)
            solvent_in: Inlet lean solvent (optional, created if not provided)
            T_op: Operating temperature (K), defaults to T_liquid_in

        Returns:
            gas_out: Treated gas stream
            solvent_out: Rich solvent stream
            info: Dict with operation details:
                - capture_efficiency: CO2 removal fraction
                - rich_loading: Rich solvent loading
                - n_stages_actual: Actual stages used
                - absorption_factor: A = L*m / G
        """
        p = self.params
        T_op = T_op if T_op is not None else p.T_liquid_in
        T_op = jnp.asarray(T_op)

        # Get gas flows
        gas_flows = get_flows(gas_in)
        F_CO2_in = jnp.asarray(gas_flows.get("CO2", 0.0))
        F_total_gas = total_flow(gas_in)

        # Calculate molar concentration of amine
        # solvent_conc in wt%, MW in g/mol, density ~ 1000 kg/m³
        MW_amine = self._solvent_data.MW
        rho_solvent = self._solvent_data.density
        C_amine = (p.solvent_conc / 100) * rho_solvent * 1000 / MW_amine  # mol/m³

        # Liquid flow rate from L/G ratio
        L_G = jnp.asarray(p.L_G_ratio)
        F_liquid = L_G * F_total_gas  # mol/s total liquid

        # Moles of amine in liquid
        # Assume liquid is ~solvent_conc% amine by mass
        # mol amine = F_liquid * (solvent_conc/100) * MW_water / MW_amine
        # Simplified: F_amine ~ F_liquid * solvent_conc / 100 (approximate)
        F_amine = F_liquid * p.solvent_conc / 100

        # VLE slope (m = dP_CO2/d_loading at operating conditions)
        # For simplified model, linearize around lean loading
        lean_loading = jnp.asarray(p.lean_loading)
        P_eq_lean = self._vle.equilibrium_pressure(lean_loading, T_op)

        # Approximate slope by finite difference
        d_alpha = 0.01
        P_eq_plus = self._vle.equilibrium_pressure(lean_loading + d_alpha, T_op)
        m = (P_eq_plus - P_eq_lean) / d_alpha  # Pa / (mol/mol)

        # Convert to molar basis: m' = m * C_amine / P_total
        P_total = jnp.asarray(p.P_absorber)
        m_prime = m * F_amine / (P_total * F_total_gas)

        # Absorption factor A = L / (m * G) where L, G are molar flows
        # In our units: A = F_amine / (m_prime * F_total_gas)
        A = L_G / (m_prime + 1e-10)

        # Kremser equation for absorption
        n_stages = jnp.asarray(p.n_stages)
        eta = jnp.asarray(p.stage_efficiency)

        # Effective stages accounting for efficiency
        # N_eff = N * eta (simplified approach)
        # More rigorous: use Murphree efficiency in stage-by-stage calc
        N_eff = n_stages * eta

        # Fraction of CO2 remaining in gas
        # phi = (A - 1) / (A^(N+1) - 1) for counter-current
        A_Np1 = jnp.power(A, N_eff + 1)
        phi = jnp.where(
            jnp.abs(A - 1.0) < 1e-6,
            1.0 / (N_eff + 1),
            (A - 1.0) / (A_Np1 - 1.0 + 1e-10)
        )
        phi = jnp.clip(phi, 0.001, 0.999)

        # CO2 in outlet gas
        F_CO2_out = F_CO2_in * phi
        capture_efficiency = 1.0 - phi

        # CO2 absorbed
        F_CO2_absorbed = F_CO2_in - F_CO2_out

        # Rich loading
        rich_loading = lean_loading + F_CO2_absorbed / (F_amine + 1e-10)
        rich_loading = jnp.clip(rich_loading, 0.0, self._solvent_data.loading_capacity)

        # Create output streams
        # Treated gas
        gas_out_flows = {}
        for species, flow in gas_flows.items():
            if species == "CO2":
                gas_out_flows[species] = F_CO2_out
            else:
                gas_out_flows[species] = flow

        gas_out = make_stream(gas_out_flows, T_op, P_total)

        # Rich solvent
        solvent_out_flows = {
            "H2O": F_liquid * (1 - p.solvent_conc / 100),
            "Amine": F_amine,
            "CO2_absorbed": F_CO2_absorbed,
        }
        solvent_out = make_stream(solvent_out_flows, T_op, P_total)

        info = {
            "capture_efficiency": capture_efficiency,
            "CO2_captured": F_CO2_absorbed,
            "rich_loading": rich_loading,
            "lean_loading": lean_loading,
            "n_stages": n_stages,
            "n_stages_effective": N_eff,
            "absorption_factor": A,
            "L_G_ratio": L_G,
            "T_operating": T_op,
            "C_amine": C_amine,
        }

        return gas_out, solvent_out, info

    def required_stages(
        self,
        capture_target: Array | float,
        gas_in: Stream,
    ) -> Array:
        """Calculate stages needed for target capture efficiency.

        Inverts the Kremser equation to find N for given capture.

        Args:
            capture_target: Target capture efficiency (0-1)
            gas_in: Inlet gas stream

        Returns:
            Required number of stages
        """
        p = self.params
        capture_target = jnp.asarray(capture_target)
        phi = 1.0 - capture_target

        # Need to recalculate A (absorption factor)
        # This is a simplified version
        T_op = jnp.asarray(p.T_liquid_in)
        gas_flows = get_flows(gas_in)
        F_total_gas = total_flow(gas_in)

        MW_amine = self._solvent_data.MW
        rho_solvent = self._solvent_data.density
        C_amine = (p.solvent_conc / 100) * rho_solvent * 1000 / MW_amine

        L_G = jnp.asarray(p.L_G_ratio)
        F_liquid = L_G * F_total_gas
        F_amine = F_liquid * p.solvent_conc / 100

        lean_loading = jnp.asarray(p.lean_loading)
        P_eq_lean = self._vle.equilibrium_pressure(lean_loading, T_op)
        d_alpha = 0.01
        P_eq_plus = self._vle.equilibrium_pressure(lean_loading + d_alpha, T_op)
        m = (P_eq_plus - P_eq_lean) / d_alpha

        P_total = jnp.asarray(p.P_absorber)
        m_prime = m * F_amine / (P_total * F_total_gas)
        A = L_G / (m_prime + 1e-10)

        # Solve Kremser for N:
        # phi = (A-1)/(A^(N+1)-1)
        # A^(N+1) = (A-1)/phi + 1
        # N+1 = log((A-1)/phi + 1) / log(A)
        eta = jnp.asarray(p.stage_efficiency)

        numerator = (A - 1) / (phi + 1e-10) + 1
        N_eff = jnp.log(numerator) / jnp.log(A + 1e-10) - 1
        N = N_eff / eta

        return jnp.maximum(N, 1.0)
