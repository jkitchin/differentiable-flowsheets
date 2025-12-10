"""Thermodynamic property calculations for difflow.

This module provides ideal thermodynamic models:
- Ideal gas behavior
- Ideal liquid mixtures
- Antoine equation for vapor pressures
- Polynomial Cp correlations
"""

from typing import NamedTuple
import jax.numpy as jnp
from jax import Array


class SpeciesData(NamedTuple):
    """Thermodynamic data for a pure species.

    Attributes:
        name: Species identifier
        MW: Molecular weight (g/mol)
        Cp_coeffs: Heat capacity coefficients [a, b, c, d] for
                   Cp = a + b*T + c*T^2 + d*T^3 (J/mol/K)
        Hvap_coeffs: Heat of vaporization coefficients [A, n, Tc] for
                     Hvap = A * (1 - T/Tc)^n (J/mol)
        antoine_coeffs: Antoine equation coefficients [A, B, C] for
                        log10(Psat/Pa) = A - B/(T + C) where T in K
        Hf: Standard heat of formation (J/mol) at 298.15 K
        Tref: Reference temperature for Hf (K), default 298.15
    """
    name: str
    MW: float
    Cp_coeffs: tuple[float, float, float, float]
    Hvap_coeffs: tuple[float, float, float]  # A, n, Tc
    antoine_coeffs: tuple[float, float, float]  # A, B, C
    Hf: float = 0.0
    Tref: float = 298.15


class IdealThermo:
    """Ideal thermodynamic property calculator.

    Provides methods for computing:
    - Heat capacities
    - Enthalpies
    - Vapor pressures (Antoine equation)
    - K-values for VLE (Raoult's law)

    All methods are JAX-compatible for automatic differentiation.
    """

    def __init__(self, species_data: dict[str, SpeciesData]):
        """Initialize with species data.

        Args:
            species_data: Dictionary mapping species names to SpeciesData
        """
        self.species = species_data
        self._species_order = list(species_data.keys())

    @property
    def species_order(self) -> list[str]:
        """Ordered list of species names."""
        return self._species_order

    @property
    def n_species(self) -> int:
        """Number of species."""
        return len(self._species_order)

    def Cp(self, species: str, T: Array | float) -> Array:
        """Calculate heat capacity of a pure species.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Heat capacity Cp (J/mol/K)
        """
        a, b, c, d = self.species[species].Cp_coeffs
        T = jnp.asarray(T)
        return a + b * T + c * T**2 + d * T**3

    def Cp_mix(
        self,
        mole_fracs: dict[str, Array | float],
        T: Array | float,
    ) -> Array:
        """Calculate heat capacity of an ideal mixture.

        Args:
            mole_fracs: Dictionary of mole fractions by species
            T: Temperature (K)

        Returns:
            Mixture heat capacity (J/mol/K)
        """
        Cp_total = jnp.zeros(())
        for species, x in mole_fracs.items():
            Cp_total = Cp_total + x * self.Cp(species, T)
        return Cp_total

    def H_pure(
        self,
        species: str,
        T: Array | float,
        phase: str = "liquid",
    ) -> Array:
        """Calculate enthalpy of a pure species relative to reference.

        Args:
            species: Species name
            T: Temperature (K)
            phase: 'liquid' or 'vapor'

        Returns:
            Specific enthalpy (J/mol) relative to liquid at Tref
        """
        data = self.species[species]
        a, b, c, d = data.Cp_coeffs
        T = jnp.asarray(T)
        Tref = jnp.asarray(data.Tref)

        # Integral of Cp from Tref to T
        H = (
            a * (T - Tref)
            + b / 2 * (T**2 - Tref**2)
            + c / 3 * (T**3 - Tref**3)
            + d / 4 * (T**4 - Tref**4)
        )

        if phase == "vapor":
            # Add heat of vaporization at T
            H = H + self.Hvap(species, T)

        return H

    def Hvap(self, species: str, T: Array | float) -> Array:
        """Calculate heat of vaporization at temperature T.

        Uses Watson correlation: Hvap = A * (1 - T/Tc)^n

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Heat of vaporization (J/mol)
        """
        A, n, Tc = self.species[species].Hvap_coeffs
        T = jnp.asarray(T)
        Tc = jnp.asarray(Tc)
        # Avoid issues at T >= Tc
        Tr = jnp.clip(T / Tc, 0.0, 0.999)
        return A * (1 - Tr) ** n

    def Psat(self, species: str, T: Array | float) -> Array:
        """Calculate saturation pressure using Antoine equation.

        log10(Psat/Pa) = A - B/(T + C)

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Saturation pressure (Pa)
        """
        A, B, C = self.species[species].antoine_coeffs
        T = jnp.asarray(T)
        log10_P = A - B / (T + C)
        return 10.0 ** log10_P

    def K_value(
        self,
        species: str,
        T: Array | float,
        P: Array | float,
    ) -> Array:
        """Calculate VLE K-value using Raoult's law.

        K = y/x = Psat/P (ideal, Raoult's law)

        Args:
            species: Species name
            T: Temperature (K)
            P: Pressure (Pa)

        Returns:
            K-value (dimensionless)
        """
        Psat = self.Psat(species, T)
        return Psat / jnp.asarray(P)

    def K_values(
        self,
        T: Array | float,
        P: Array | float,
    ) -> dict[str, Array]:
        """Calculate K-values for all species.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)

        Returns:
            Dictionary of K-values by species name
        """
        return {
            species: self.K_value(species, T, P)
            for species in self._species_order
        }

    def K_values_array(
        self,
        T: Array | float,
        P: Array | float,
    ) -> Array:
        """Calculate K-values as an array in species order.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)

        Returns:
            Array of K-values in species_order
        """
        return jnp.array([
            self.K_value(s, T, P) for s in self._species_order
        ])

    def bubble_pressure(
        self,
        x: dict[str, Array | float],
        T: Array | float,
    ) -> Array:
        """Calculate bubble point pressure.

        P_bubble = sum(x_i * Psat_i)

        Args:
            x: Liquid mole fractions by species
            T: Temperature (K)

        Returns:
            Bubble pressure (Pa)
        """
        P_bubble = jnp.zeros(())
        for species, xi in x.items():
            P_bubble = P_bubble + xi * self.Psat(species, T)
        return P_bubble

    def dew_pressure(
        self,
        y: dict[str, Array | float],
        T: Array | float,
    ) -> Array:
        """Calculate dew point pressure.

        1/P_dew = sum(y_i / Psat_i)

        Args:
            y: Vapor mole fractions by species
            T: Temperature (K)

        Returns:
            Dew pressure (Pa)
        """
        inv_P_dew = jnp.zeros(())
        for species, yi in y.items():
            inv_P_dew = inv_P_dew + yi / self.Psat(species, T)
        return 1.0 / inv_P_dew

    def stream_enthalpy(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        phase: str = "liquid",
    ) -> Array:
        """Calculate total enthalpy of a stream.

        Args:
            flows: Molar flows by species (mol/s)
            T: Temperature (K)
            phase: 'liquid' or 'vapor'

        Returns:
            Total enthalpy flow (J/s = W)
        """
        H_total = jnp.zeros(())
        for species, F in flows.items():
            H_total = H_total + F * self.H_pure(species, T, phase)
        return H_total
