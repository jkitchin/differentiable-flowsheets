"""Vapor-liquid equilibrium models for amine-CO2-H2O systems.

This module provides simplified VLE models for amine-based CO2 capture.
The simplified approach uses Kent-Eisenberg type correlations that are
fast and differentiable, suitable for process optimization.

For rigorous thermodynamic calculations, the electrolyte NRTL (eNRTL)
model should be used. This module provides extensibility hooks for
integration with more rigorous models.

References:
    Kent RL, Eisenberg B (1976). Better data for amine treating.
        Hydrocarbon Processing 55(2):87-90.
    Chen CC, Evans LB (1986). A local composition model for the
        excess Gibbs energy of aqueous electrolyte systems.
        AIChE J 32:444-454.
    Kim I, Svendsen HF (2007). Heat of absorption of carbon dioxide
        (CO2) in monoethanolamine (MEA) and 2-(aminoethyl)ethanolamine
        (AEEA) solutions. Ind Eng Chem Res 46:5803-5809.
"""

__all__ = [
    "henry_constant",
    "co2_equilibrium_pressure",
    "co2_loading",
    "AmineVLE",
    "AmineVLEParams",
]

from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array
import optimistix as optx

from difflow.params_mixin import ParamsMixin
from difflow_cc.database import get_solvent, AmineSolvent

# Gas constant
R = 8.314  # J/(mol*K)


# =============================================================================
# Henry's Law Functions
# =============================================================================

def henry_constant(T: Array | float, solvent: str) -> Array:
    """Calculate Henry's constant for CO2 in amine solution.

    ln(H/Pa) = a + b/T + c*ln(T)

    Args:
        T: Temperature (K)
        solvent: Solvent name

    Returns:
        Henry's constant H (Pa)

    References:
        Versteeg GF, Van Swaaij WPM (1988). Solubility and diffusivity
            of acid gases (CO2, N2O) in aqueous alkanolamine solutions.
            J Chem Eng Data 33:29-34.
    """
    T = jnp.asarray(T)
    s = get_solvent(solvent)

    ln_H = s.henry_a + s.henry_b / T + s.henry_c * jnp.log(T)
    return jnp.exp(ln_H)


def physical_solubility(
    P_CO2: Array | float,
    T: Array | float,
    solvent: str
) -> Array:
    """Calculate physical (Henry's law) CO2 solubility.

    C_CO2_phys = P_CO2 / H(T)

    This is the free dissolved CO2 concentration, not accounting
    for chemical reaction with the amine.

    Args:
        P_CO2: CO2 partial pressure (Pa)
        T: Temperature (K)
        solvent: Solvent name

    Returns:
        Physical CO2 concentration (mol/m³)
    """
    P_CO2 = jnp.asarray(P_CO2)
    T = jnp.asarray(T)

    H = henry_constant(T, solvent)
    return P_CO2 / H


# =============================================================================
# Simplified VLE Model (Kent-Eisenberg type)
# =============================================================================

def co2_equilibrium_pressure(
    loading: Array | float,
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0
) -> Array:
    """Calculate CO2 equilibrium partial pressure for given loading.

    Uses a simplified Kent-Eisenberg type correlation:
        P_CO2 = K(T) * alpha^n / (1 - alpha/alpha_max)^m

    where alpha = loading (mol CO2 / mol amine)

    This correlation captures the key behavior:
    - Low P_CO2 at low loading (steep initial slope)
    - Rapid increase as loading approaches capacity
    - Temperature dependence through K(T)

    Args:
        loading: CO2 loading (mol CO2 / mol amine)
        T: Temperature (K)
        solvent: Solvent name
        C_amine: Amine concentration (mol/m³, default 5000 = ~30 wt% MEA)

    Returns:
        CO2 equilibrium partial pressure (Pa)

    References:
        Kent RL, Eisenberg B (1976). Hydrocarbon Processing.
    """
    loading = jnp.asarray(loading)
    T = jnp.asarray(T)
    C_amine = jnp.asarray(C_amine)

    s = get_solvent(solvent)

    # Maximum loading capacity
    alpha_max = s.loading_capacity

    # Temperature-dependent equilibrium constant
    # Fitted form: K = K0 * exp(-dH/(R*T))
    # dH is approximately the heat of absorption
    dH = s.heat_of_absorption * 1000.0  # Convert kJ to J
    K0 = 1e8  # Base equilibrium constant (Pa)
    K = K0 * jnp.exp(-dH / (R * T))

    # Exponents (typical values for MEA-like behavior)
    n = 2.0  # Loading exponent
    m = 1.5  # Capacity approach exponent

    # Avoid division by zero near saturation
    alpha_ratio = jnp.clip(loading / alpha_max, 0.0, 0.99)

    P_CO2 = K * jnp.power(loading + 1e-10, n) / jnp.power(1.0 - alpha_ratio, m)

    return jnp.maximum(P_CO2, 0.0)


def co2_loading(
    P_CO2: Array | float,
    T: Array | float,
    solvent: str,
    C_amine: Array | float = 5000.0
) -> Array:
    """Calculate CO2 loading for given partial pressure.

    Inverse of co2_equilibrium_pressure. Uses root finding to
    solve P_CO2 = f(loading) for loading.

    Args:
        P_CO2: CO2 partial pressure (Pa)
        T: Temperature (K)
        solvent: Solvent name
        C_amine: Amine concentration (mol/m³)

    Returns:
        CO2 loading (mol CO2 / mol amine)

    Notes:
        Uses optimistix for JAX-compatible root finding.
    """
    P_CO2 = jnp.asarray(P_CO2)
    T = jnp.asarray(T)

    s = get_solvent(solvent)
    alpha_max = s.loading_capacity

    def residual(alpha, args):
        P_calc = co2_equilibrium_pressure(alpha, T, solvent, C_amine)
        return P_calc - P_CO2

    # Initial guess based on linearization at low loading
    alpha_init = jnp.clip(P_CO2 / 1e6, 0.01, alpha_max * 0.9)

    solver = optx.Newton(rtol=1e-6, atol=1e-8)
    sol = optx.root_find(residual, solver, alpha_init, args=None)

    return jnp.clip(sol.value, 0.0, alpha_max)


# =============================================================================
# VLE Model Class
# =============================================================================

@dataclass(repr=False)
class AmineVLEParams(ParamsMixin):
    """Parameters for amine VLE model.

    Attributes:
        solvent: Solvent name from database
        C_amine: Amine concentration (mol/m³)
        T_ref: Reference temperature (K)
    """
    solvent: str
    C_amine: float = 5000.0  # mol/m³ (~30 wt% MEA)
    T_ref: float = 313.15  # K (40°C, typical absorber)


class AmineVLE:
    """Vapor-liquid equilibrium model for amine-CO2-H2O system.

    Provides a unified interface for VLE calculations with
    automatic differentiation support.

    Example:
        >>> vle = AmineVLE('MEA', C_amine=5000)
        >>> P_eq = vle.equilibrium_pressure(loading=0.3, T=313.15)
        >>> alpha = vle.loading(P_CO2=5000, T=313.15)

    The model can compute:
    - Equilibrium pressure from loading
    - Loading from partial pressure
    - Heat of absorption
    - Enhancement factors for mass transfer

    For rate-based models, this class can be extended to include
    rigorous eNRTL thermodynamics.
    """

    def __init__(
        self,
        solvent: str,
        C_amine: float = 5000.0,
    ):
        """Initialize VLE model.

        Args:
            solvent: Solvent name (e.g., 'MEA', 'PZ', 'MDEA')
            C_amine: Amine concentration (mol/m³)
        """
        self.solvent = solvent
        self.C_amine = C_amine
        self._solvent_data = get_solvent(solvent)

    @property
    def loading_capacity(self) -> float:
        """Maximum CO2 loading capacity (mol CO2 / mol amine)."""
        return self._solvent_data.loading_capacity

    @property
    def heat_of_absorption(self) -> float:
        """Heat of CO2 absorption (kJ/mol CO2)."""
        return self._solvent_data.heat_of_absorption

    def equilibrium_pressure(
        self,
        loading: Array | float,
        T: Array | float
    ) -> Array:
        """Calculate CO2 equilibrium partial pressure.

        Args:
            loading: CO2 loading (mol CO2 / mol amine)
            T: Temperature (K)

        Returns:
            CO2 partial pressure (Pa)
        """
        return co2_equilibrium_pressure(
            loading, T, self.solvent, self.C_amine
        )

    def loading(self, P_CO2: Array | float, T: Array | float) -> Array:
        """Calculate CO2 loading from partial pressure.

        Args:
            P_CO2: CO2 partial pressure (Pa)
            T: Temperature (K)

        Returns:
            CO2 loading (mol CO2 / mol amine)
        """
        return co2_loading(P_CO2, T, self.solvent, self.C_amine)

    def henry_constant(self, T: Array | float) -> Array:
        """Get Henry's constant at temperature T.

        Args:
            T: Temperature (K)

        Returns:
            Henry's constant (Pa)
        """
        return henry_constant(T, self.solvent)

    def cyclic_capacity(
        self,
        P_CO2_rich: Array | float,
        P_CO2_lean: Array | float,
        T_abs: Array | float,
        T_des: Array | float
    ) -> Array:
        """Calculate cyclic CO2 capacity.

        Cyclic capacity = alpha_rich - alpha_lean

        Args:
            P_CO2_rich: Rich loading equilibrium pressure (Pa)
            P_CO2_lean: Lean loading equilibrium pressure (Pa)
            T_abs: Absorber temperature (K)
            T_des: Stripper/desorber temperature (K)

        Returns:
            Cyclic capacity (mol CO2 / mol amine)
        """
        alpha_rich = self.loading(P_CO2_rich, T_abs)
        alpha_lean = self.loading(P_CO2_lean, T_des)
        return alpha_rich - alpha_lean

    def regeneration_energy(
        self,
        alpha_rich: Array | float,
        alpha_lean: Array | float,
        T_reboiler: Array | float,
        Cp_solvent: Array | float = 4000.0,
        T_feed: Array | float = 313.15
    ) -> Array:
        """Estimate specific regeneration energy.

        Q_regen = Q_sensible + Q_reaction + Q_vaporization

        This is a simplified estimate. For rigorous calculations,
        a full stripper model should be used.

        Args:
            alpha_rich: Rich loading (mol CO2 / mol amine)
            alpha_lean: Lean loading (mol CO2 / mol amine)
            T_reboiler: Reboiler temperature (K)
            Cp_solvent: Solvent heat capacity (J/(kg·K))
            T_feed: Feed temperature (K)

        Returns:
            Specific regeneration energy (GJ/tonne CO2)
        """
        alpha_rich = jnp.asarray(alpha_rich)
        alpha_lean = jnp.asarray(alpha_lean)
        T_reboiler = jnp.asarray(T_reboiler)

        # CO2 absorbed per kg solvent
        # Assume ~30 wt% amine, MW ~ 60 g/mol
        MW_amine = self._solvent_data.MW
        delta_alpha = alpha_rich - alpha_lean
        mol_CO2_per_kg = delta_alpha * 0.30 * 1000 / MW_amine  # mol CO2/kg solvent

        # Sensible heat (J/kg solvent)
        Q_sensible = Cp_solvent * (T_reboiler - T_feed)

        # Heat of reaction (J/mol CO2)
        dH_abs = self._solvent_data.heat_of_absorption * 1000  # J/mol

        # Heat of vaporization for steam (rough estimate, J/kg solvent)
        # Assume ~0.3 mol H2O per mol CO2 vaporized
        dH_vap = 2260000  # J/kg water
        steam_ratio = 0.3 * 18 / 1000  # kg water per mol CO2

        # Total energy per mol CO2
        Q_per_mol = (
            Q_sensible / (mol_CO2_per_kg + 1e-10) +
            dH_abs +
            steam_ratio * dH_vap
        )

        # Convert to GJ/tonne CO2
        # 1 mol CO2 = 44 g, 1 tonne = 1e6 g
        mol_per_tonne = 1e6 / 44
        Q_GJ_per_tonne = Q_per_mol * mol_per_tonne / 1e9

        return Q_GJ_per_tonne
