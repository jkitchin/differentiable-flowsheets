"""Base classes and helpers for unit operations.

This module provides:
- UnitBase: Optional base class for unit operations with common functionality
- Helper functions for initialization to reduce code duplication

The base class is optional - units can also implement the UnitOperation
protocol directly without inheriting from UnitBase.
"""

from abc import ABC, abstractmethod
from typing import Any, Literal
import jax.numpy as jnp
from jax import Array

from difflow.streams import Stream, get_flows, make_stream
from difflow.thermo import IdealThermo
from difflow.numerics import safe_divide


# =============================================================================
# Initialization Helpers
# =============================================================================

def estimate_volumetric_flow(
    inlet_flows: dict[str, float | Array],
    default_concentration: float = 55500.0,
) -> float:
    """Estimate volumetric flow from molar flows.

    Assumes a default molar concentration for liquids.

    Args:
        inlet_flows: Dict of species molar flow rates (mol/s)
        default_concentration: Assumed molar concentration (mol/m³), default 55500

    Returns:
        Estimated volumetric flow rate (m³/s)
    """
    total_molar = sum(inlet_flows.values())
    return float(total_molar) / default_concentration


def estimate_outlet_composition(
    inlet_flows: dict[str, float | Array],
    stoich: Array,
    species_order: list[str],
    conversion: float,
) -> dict[str, float]:
    """Estimate outlet composition based on stoichiometry and conversion.

    Assumes the first reaction is dominant and the first reactant
    in species_order with negative stoichiometry is limiting.

    Args:
        inlet_flows: Dict of species inlet molar flow rates (mol/s)
        stoich: Stoichiometry matrix, shape (n_species, n_reactions)
        species_order: List of species names matching stoich rows
        conversion: Estimated conversion (0-1)

    Returns:
        Dict of estimated outlet molar flow rates (mol/s)
    """
    outlet_flows = {}

    # Find limiting reactant for first reaction
    reactant_info = []
    for i, species in enumerate(species_order):
        nu = stoich[i, 0] if stoich.shape[1] > 0 else 0
        if nu < 0:
            F_in = float(inlet_flows.get(species, 0.0))
            reactant_info.append((species, i, F_in, -nu))

    # Find limiting reactant (minimum F/nu)
    if reactant_info:
        limiting = min(reactant_info, key=lambda x: safe_divide(x[2], x[3]))
        limiting_species, limiting_idx, F_limiting, nu_limiting = limiting
        extent = F_limiting * conversion / nu_limiting
    else:
        extent = 0.0

    # Calculate outlet flows
    for i, species in enumerate(species_order):
        F_in = float(inlet_flows.get(species, 0.0))
        nu = float(stoich[i, 0]) if stoich.shape[1] > 0 else 0.0
        outlet_flows[species] = F_in + nu * extent

    return outlet_flows


def estimate_adiabatic_temperature(
    inlet_T: float | Array,
    dH_rxn: Array,
    extent: float,
    total_flow: float,
    Cp_avg: float = 75.0,
    T_min: float = 250.0,
    T_max: float = 800.0,
) -> float:
    """Estimate outlet temperature for adiabatic operation.

    Args:
        inlet_T: Inlet temperature (K)
        dH_rxn: Heats of reaction (J/mol), negative for exothermic
        extent: Reaction extent (mol/s)
        total_flow: Total molar flow rate (mol/s)
        Cp_avg: Average heat capacity (J/mol/K), default 75 for liquids
        T_min: Minimum temperature bound (K)
        T_max: Maximum temperature bound (K)

    Returns:
        Estimated outlet temperature (K)
    """
    # Heat released (positive for exothermic)
    Q_rxn = -float(jnp.sum(dH_rxn)) * extent

    # Temperature change
    dT = safe_divide(Q_rxn, total_flow * Cp_avg)

    # Apply bounds
    T_out = float(inlet_T) + dT
    return float(jnp.clip(T_out, T_min, T_max))


def estimate_residence_time(
    volume: float | Array,
    volumetric_flow: float | Array,
) -> float:
    """Calculate residence time.

    Args:
        volume: Reactor volume (m³)
        volumetric_flow: Volumetric flow rate (m³/s)

    Returns:
        Residence time (s)
    """
    return float(safe_divide(volume, volumetric_flow))


def estimate_cstr_conversion(
    k: float,
    tau: float,
) -> float:
    """Estimate CSTR conversion for first-order reaction.

    X = k*tau / (1 + k*tau)

    Args:
        k: Rate constant (1/s)
        tau: Residence time (s)

    Returns:
        Estimated conversion (0-1)
    """
    X = k * tau / (1 + k * tau)
    return float(jnp.clip(X, 0.0, 0.99))


def estimate_pfr_conversion(
    k: float,
    tau: float,
) -> float:
    """Estimate PFR conversion for first-order reaction.

    X = 1 - exp(-k*tau)

    Args:
        k: Rate constant (1/s)
        tau: Residence time (s)

    Returns:
        Estimated conversion (0-1)
    """
    X = 1.0 - jnp.exp(-k * tau)
    return float(jnp.clip(X, 0.0, 0.99))


# =============================================================================
# Base Classes
# =============================================================================

class UnitBase(ABC):
    """Optional base class for unit operations.

    Provides common functionality for units with params and optional thermo.
    Units don't need to inherit from this - they can implement UnitOperation
    protocol directly.

    Subclasses should:
    1. Call super().__init__(params, thermo, mode) in their __init__
    2. Implement __call__ for the main calculation
    3. Optionally implement initialize() for initialization support

    Attributes:
        params: Unit parameters (dataclass inheriting from ParamsMixin)
        thermo: Optional thermodynamic property calculator
        mode: Optional operating mode (e.g., 'isothermal', 'adiabatic')
    """

    def __init__(
        self,
        params: Any,
        thermo: IdealThermo | None = None,
        mode: str | None = None,
    ):
        """Initialize unit.

        Args:
            params: Unit parameters
            thermo: Optional thermodynamic property calculator
            mode: Optional operating mode
        """
        self.params = params
        self.thermo = thermo
        self.mode = mode

        # Subclasses should add their own validation
        self._validate()

    def _validate(self) -> None:
        """Validate unit configuration.

        Override in subclasses to add specific validation.
        Called at end of __init__.
        """
        pass

    @abstractmethod
    def __call__(self, inlet: Stream, **kwargs) -> tuple[Any, dict]:
        """Process inlet stream.

        Args:
            inlet: Input stream
            **kwargs: Operation-specific parameters

        Returns:
            outlet: Output stream(s)
            info: Dictionary with operation results
        """
        pass

    @property
    def species_order(self) -> list[str]:
        """Get species order from params if available."""
        return getattr(self.params, 'species_order', [])

    def _get_volumetric_flow(
        self,
        inlet_flows: dict[str, float | Array],
        volumetric_flow: float | Array | None = None,
    ) -> float:
        """Get volumetric flow, estimating if not provided.

        Args:
            inlet_flows: Dict of species molar flows
            volumetric_flow: Optional provided volumetric flow

        Returns:
            Volumetric flow rate (m³/s)
        """
        if volumetric_flow is not None:
            return float(volumetric_flow)
        return estimate_volumetric_flow(inlet_flows)


class ReactorBase(UnitBase):
    """Base class for reactor units (CSTR, PFR).

    Extends UnitBase with reactor-specific functionality:
    - Stoichiometry handling
    - Rate function support
    - Common initialization patterns
    """

    def _validate(self) -> None:
        """Validate reactor configuration."""
        super()._validate()

        # Check stoich dimensions match species_order
        if hasattr(self.params, 'stoich') and hasattr(self.params, 'species_order'):
            n_species = len(self.params.species_order)
            if self.params.stoich.shape[0] != n_species:
                raise ValueError(
                    f"Stoichiometry matrix has {self.params.stoich.shape[0]} rows "
                    f"but species_order has {n_species} species"
                )

        # Check thermo for non-isothermal
        if self.mode not in (None, "isothermal"):
            if self.thermo is None:
                raise ValueError(
                    f"Thermo object required for {self.mode} operation"
                )

    def _compute_concentrations(
        self,
        flows: dict[str, Array],
        total_volumetric_flow: Array,
    ) -> dict[str, Array]:
        """Compute concentrations from molar flows.

        Args:
            flows: Molar flow rates (mol/s)
            total_volumetric_flow: Total volumetric flow (m³/s)

        Returns:
            Concentrations (mol/m³)
        """
        return {
            species: F / total_volumetric_flow
            for species, F in flows.items()
        }

    def _get_rate_constant_estimate(self) -> float:
        """Get rate constant estimate from params for initialization."""
        rate_params = getattr(self.params, 'rate_params', {})
        return rate_params.get('k', 0.1)
