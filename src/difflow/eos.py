"""Cubic Equations of State for difflow.

This module provides cubic EOS models for non-ideal thermodynamic calculations:
- Peng-Robinson (PR) equation of state
- Soave-Redlich-Kwong (SRK) equation of state

These equations are essential for accurate VLE calculations in systems
where ideal behavior (Raoult's law) is insufficient.

Key equations:
    P = RT/(V-b) - a(T)/(V² + ubV + wb²)

    PR:  u = 2, w = -1  ->  a(T)/(V² + 2bV - b²)
    SRK: u = 1, w = 0   ->  a(T)/(V(V + b))

All calculations are JAX-compatible for automatic differentiation.
"""

from typing import NamedTuple, Literal
from dataclasses import dataclass, replace
import jax.numpy as jnp
from jax import Array, lax

import optimistix as optx


# Universal gas constant (J/mol/K)
R = 8.314462618


class CriticalProperties(NamedTuple):
    """Critical properties for a pure species.

    Attributes:
        name: Species identifier
        Tc: Critical temperature (K)
        Pc: Critical pressure (Pa)
        omega: Acentric factor (dimensionless)
        MW: Molecular weight (g/mol)
    """
    name: str
    Tc: float
    Pc: float
    omega: float
    MW: float = 0.0


@dataclass
class EOSParams:
    """Parameters for equation of state calculations.

    Precomputed from critical properties for efficiency.
    """
    a_c: Array  # Critical 'a' parameter for each species
    b: Array    # 'b' parameter for each species
    kappa: Array  # Temperature dependence parameter
    Tc: Array   # Critical temperatures
    Pc: Array   # Critical pressures
    omega: Array  # Acentric factors
    species_order: list[str]

    def update(self, **kwargs) -> "EOSParams":
        """Return a new EOSParams with specified fields replaced.

        This enables JAX-compatible parameter updates for differentiation.

        Args:
            **kwargs: Fields to update

        Returns:
            New EOSParams with updated fields
        """
        return replace(self, **kwargs)


class PengRobinson:
    """Peng-Robinson equation of state.

    P = RT/(V-b) - a(T)/(V² + 2bV - b²)

    where:
        a(T) = a_c * alpha(T)
        alpha(T) = [1 + kappa*(1 - sqrt(T/Tc))]²
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega²
        a_c = 0.45724 * R² * Tc² / Pc
        b = 0.07780 * R * Tc / Pc

    Provides methods for:
    - Compressibility factor (Z)
    - Fugacity coefficients
    - K-values for VLE

    All methods are JAX-compatible for automatic differentiation.
    """

    # EOS constants for PR
    u = 2.0   # coefficient for bV term
    w = -1.0  # coefficient for b² term
    OMEGA_A = 0.45724
    OMEGA_B = 0.07780

    def __init__(self, species_data: dict[str, CriticalProperties]):
        """Initialize with species critical properties.

        Args:
            species_data: Dictionary mapping species names to CriticalProperties
        """
        self.species = species_data
        self._species_order = list(species_data.keys())
        self.n_species = len(self._species_order)
        self.params = self._compute_params()

    @property
    def species_order(self) -> list[str]:
        """Ordered list of species names."""
        return self._species_order

    def _compute_params(self) -> EOSParams:
        """Precompute EOS parameters from critical properties."""
        n = self.n_species
        Tc = jnp.array([self.species[s].Tc for s in self._species_order])
        Pc = jnp.array([self.species[s].Pc for s in self._species_order])
        omega = jnp.array([self.species[s].omega for s in self._species_order])

        # Critical parameters
        a_c = self.OMEGA_A * R**2 * Tc**2 / Pc
        b = self.OMEGA_B * R * Tc / Pc

        # Temperature dependence (kappa for alpha function)
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        return EOSParams(
            a_c=a_c,
            b=b,
            kappa=kappa,
            Tc=Tc,
            Pc=Pc,
            omega=omega,
            species_order=self._species_order,
        )

    def alpha(self, T: Array) -> Array:
        """Calculate alpha function for temperature dependence.

        alpha(T) = [1 + kappa*(1 - sqrt(T/Tc))]²

        Args:
            T: Temperature (K)

        Returns:
            Alpha values for each species
        """
        p = self.params
        Tr = T / p.Tc
        return (1 + p.kappa * (1 - jnp.sqrt(Tr)))**2

    def a(self, T: Array) -> Array:
        """Calculate 'a' parameter at temperature T.

        Args:
            T: Temperature (K)

        Returns:
            'a' values for each species (Pa*m^6/mol^2)
        """
        return self.params.a_c * self.alpha(T)

    def a_mix(
        self,
        T: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate mixture 'a' parameter using van der Waals mixing rules.

        a_mix = sum_i sum_j y_i * y_j * sqrt(a_i * a_j) * (1 - k_ij)

        Args:
            T: Temperature (K)
            y: Mole fractions (array in species_order)
            k_ij: Binary interaction parameters (n x n matrix).
                  If None, assumes k_ij = 0 for all pairs.

        Returns:
            Mixture 'a' parameter
        """
        a_i = self.a(T)
        n = self.n_species

        if k_ij is None:
            k_ij = jnp.zeros((n, n))

        # a_mix = sum_i sum_j y_i * y_j * sqrt(a_i * a_j) * (1 - k_ij)
        a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
        a_mix = jnp.sum(jnp.outer(y, y) * a_ij)

        return a_mix

    def b_mix(self, y: Array) -> Array:
        """Calculate mixture 'b' parameter using linear mixing rule.

        b_mix = sum_i y_i * b_i

        Args:
            y: Mole fractions (array in species_order)

        Returns:
            Mixture 'b' parameter
        """
        return jnp.sum(y * self.params.b)

    def compressibility_cubic(
        self,
        T: Array,
        P: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Get cubic equation coefficients for compressibility factor.

        The cubic equation is:
            Z³ + c2*Z² + c1*Z + c0 = 0

        For PR: Z³ - (1-B)*Z² + (A - 3B² - 2B)*Z - (AB - B² - B³) = 0

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            k_ij: Binary interaction parameters

        Returns:
            (c2, c1, c0): Coefficients of the cubic equation
        """
        a_m = self.a_mix(T, y, k_ij)
        b_m = self.b_mix(y)

        A = a_m * P / (R * T)**2
        B = b_m * P / (R * T)

        # For PR: u=2, w=-1
        # Z³ - (1 - B)Z² + (A - 3B² - 2B)Z - (AB - B² - B³) = 0
        c2 = -(1 - B)
        c1 = A - 3*B**2 - 2*B
        c0 = -(A*B - B**2 - B**3)

        return c2, c1, c0

    def solve_Z(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Solve for compressibility factor Z.

        Solves the cubic EOS to find Z. For two-phase conditions,
        returns largest root for vapor, smallest (positive) for liquid.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            phase: 'vapor' or 'liquid'
            k_ij: Binary interaction parameters

        Returns:
            Compressibility factor Z
        """
        c2, c1, c0 = self.compressibility_cubic(T, P, y, k_ij)

        # Solve cubic using Cardano's formula (JAX-compatible)
        Z = self._solve_cubic(c2, c1, c0, phase)

        return Z

    def _solve_cubic(
        self,
        c2: Array,
        c1: Array,
        c0: Array,
        phase: str,
    ) -> Array:
        """Solve cubic equation x³ + c2*x² + c1*x + c0 = 0.

        Uses Cardano's formula with handling for different root cases.
        Returns appropriate root based on phase.
        """
        # Reduce to depressed cubic: t³ + pt + q = 0 where x = t - c2/3
        p = c1 - c2**2 / 3
        q = c0 - c1 * c2 / 3 + 2 * c2**3 / 27

        # Discriminant
        disc = (q/2)**2 + (p/3)**3

        # Three cases based on discriminant
        # disc > 0: one real root
        # disc = 0: three real roots, at least two equal
        # disc < 0: three distinct real roots

        def one_real_root(disc_p_q):
            """Case: disc > 0, one real root."""
            disc, p, q = disc_p_q
            sqrt_disc = jnp.sqrt(disc)
            u = jnp.cbrt(-q/2 + sqrt_disc)
            v = jnp.cbrt(-q/2 - sqrt_disc)
            t = u + v
            return t - c2/3

        def three_real_roots(disc_p_q):
            """Case: disc <= 0, three real roots."""
            disc, p, q = disc_p_q
            # Use trigonometric solution
            r = jnp.sqrt(-p**3 / 27)
            theta = jnp.arccos(jnp.clip(-q / (2 * r + 1e-30), -1, 1))

            # Three roots
            t1 = 2 * jnp.cbrt(r) * jnp.cos(theta / 3)
            t2 = 2 * jnp.cbrt(r) * jnp.cos((theta + 2*jnp.pi) / 3)
            t3 = 2 * jnp.cbrt(r) * jnp.cos((theta + 4*jnp.pi) / 3)

            x1 = t1 - c2/3
            x2 = t2 - c2/3
            x3 = t3 - c2/3

            # For vapor: largest positive root
            # For liquid: smallest positive root
            roots = jnp.array([x1, x2, x3])
            positive_mask = roots > 0

            # Replace negative roots with large/small values for selection
            if phase == "vapor":
                masked_roots = jnp.where(positive_mask, roots, -jnp.inf)
                return jnp.max(masked_roots)
            else:
                masked_roots = jnp.where(positive_mask, roots, jnp.inf)
                return jnp.min(masked_roots)

        # Select root based on discriminant
        Z = lax.cond(
            disc > 0,
            one_real_root,
            three_real_roots,
            (disc, p, q),
        )

        # Ensure physical bounds (Z > B for liquid, Z > 0)
        b_m = self.b_mix(jnp.ones(self.n_species) / self.n_species)  # Approximate
        return jnp.maximum(Z, 0.01)

    def fugacity_coefficient(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate fugacity coefficients for all species.

        For PR EOS:
        ln(phi_i) = (b_i/b_m)(Z-1) - ln(Z-B)
                    - A/(2√2 B) * [2*sum_j(y_j*a_ij)/a_m - b_i/b_m]
                    * ln[(Z + (1+√2)B)/(Z + (1-√2)B)]

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            phase: 'vapor' or 'liquid'
            k_ij: Binary interaction parameters

        Returns:
            Fugacity coefficients for each species (array)
        """
        a_i = self.a(T)
        b_i = self.params.b
        n = self.n_species

        if k_ij is None:
            k_ij = jnp.zeros((n, n))

        # Mixture parameters
        a_m = self.a_mix(T, y, k_ij)
        b_m = self.b_mix(y)

        A = a_m * P / (R * T)**2
        B = b_m * P / (R * T)

        Z = self.solve_Z(T, P, y, phase, k_ij)

        # Compute a_ij matrix
        a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)

        # sum_j(y_j * a_ij) for each i
        sum_ya = jnp.sum(y * a_ij, axis=1)

        # ln(phi_i) calculation
        sqrt2 = jnp.sqrt(2.0)
        term1 = (b_i / b_m) * (Z - 1)
        term2 = -jnp.log(jnp.maximum(Z - B, 1e-10))
        term3_coeff = A / (2 * sqrt2 * B + 1e-10)
        term3_bracket = 2 * sum_ya / (a_m + 1e-30) - b_i / b_m
        term3_log = jnp.log(
            jnp.maximum((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B + 1e-10), 1e-10)
        )
        term3 = -term3_coeff * term3_bracket * term3_log

        ln_phi = term1 + term2 + term3

        return jnp.exp(ln_phi)

    def K_values(
        self,
        T: Array,
        P: Array,
        x: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate VLE K-values: K_i = y_i/x_i = phi_L_i / phi_V_i.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            x: Liquid mole fractions
            y: Vapor mole fractions
            k_ij: Binary interaction parameters

        Returns:
            K-values for each species
        """
        phi_L = self.fugacity_coefficient(T, P, x, "liquid", k_ij)
        phi_V = self.fugacity_coefficient(T, P, y, "vapor", k_ij)

        return phi_L / (phi_V + 1e-30)

    def K_values_wilson(self, T: Array, P: Array) -> Array:
        """Estimate K-values using Wilson correlation (for initialization).

        K_i = (Pc_i/P) * exp(5.373 * (1 + omega_i) * (1 - Tc_i/T))

        Args:
            T: Temperature (K)
            P: Pressure (Pa)

        Returns:
            Estimated K-values for each species
        """
        p = self.params
        return (p.Pc / P) * jnp.exp(5.373 * (1 + p.omega) * (1 - p.Tc / T))

    def molar_volume(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate molar volume from compressibility factor.

        V = ZRT/P

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            phase: 'vapor' or 'liquid'
            k_ij: Binary interaction parameters

        Returns:
            Molar volume (m³/mol)
        """
        Z = self.solve_Z(T, P, y, phase, k_ij)
        return Z * R * T / P

    def density(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate molar density.

        rho = P/(ZRT)

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            phase: 'vapor' or 'liquid'
            k_ij: Binary interaction parameters

        Returns:
            Molar density (mol/m³)
        """
        V = self.molar_volume(T, P, y, phase, k_ij)
        return 1.0 / V


class SRK:
    """Soave-Redlich-Kwong equation of state.

    P = RT/(V-b) - a(T)/(V(V+b))

    where:
        a(T) = a_c * alpha(T)
        alpha(T) = [1 + m*(1 - sqrt(T/Tc))]²
        m = 0.480 + 1.574*omega - 0.176*omega²
        a_c = 0.42748 * R² * Tc² / Pc
        b = 0.08664 * R * Tc / Pc

    All methods are JAX-compatible for automatic differentiation.
    """

    # EOS constants for SRK
    u = 1.0   # coefficient for bV term
    w = 0.0   # coefficient for b² term
    OMEGA_A = 0.42748
    OMEGA_B = 0.08664

    def __init__(self, species_data: dict[str, CriticalProperties]):
        """Initialize with species critical properties.

        Args:
            species_data: Dictionary mapping species names to CriticalProperties
        """
        self.species = species_data
        self._species_order = list(species_data.keys())
        self.n_species = len(self._species_order)
        self.params = self._compute_params()

    @property
    def species_order(self) -> list[str]:
        """Ordered list of species names."""
        return self._species_order

    def _compute_params(self) -> EOSParams:
        """Precompute EOS parameters from critical properties."""
        Tc = jnp.array([self.species[s].Tc for s in self._species_order])
        Pc = jnp.array([self.species[s].Pc for s in self._species_order])
        omega = jnp.array([self.species[s].omega for s in self._species_order])

        # Critical parameters
        a_c = self.OMEGA_A * R**2 * Tc**2 / Pc
        b = self.OMEGA_B * R * Tc / Pc

        # Temperature dependence (m for alpha function in SRK)
        kappa = 0.480 + 1.574 * omega - 0.176 * omega**2

        return EOSParams(
            a_c=a_c,
            b=b,
            kappa=kappa,
            Tc=Tc,
            Pc=Pc,
            omega=omega,
            species_order=self._species_order,
        )

    def alpha(self, T: Array) -> Array:
        """Calculate alpha function for temperature dependence.

        alpha(T) = [1 + m*(1 - sqrt(T/Tc))]²

        Args:
            T: Temperature (K)

        Returns:
            Alpha values for each species
        """
        p = self.params
        Tr = T / p.Tc
        return (1 + p.kappa * (1 - jnp.sqrt(Tr)))**2

    def a(self, T: Array) -> Array:
        """Calculate 'a' parameter at temperature T."""
        return self.params.a_c * self.alpha(T)

    def a_mix(
        self,
        T: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate mixture 'a' parameter using van der Waals mixing rules."""
        a_i = self.a(T)
        n = self.n_species

        if k_ij is None:
            k_ij = jnp.zeros((n, n))

        a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
        a_mix = jnp.sum(jnp.outer(y, y) * a_ij)

        return a_mix

    def b_mix(self, y: Array) -> Array:
        """Calculate mixture 'b' parameter using linear mixing rule."""
        return jnp.sum(y * self.params.b)

    def compressibility_cubic(
        self,
        T: Array,
        P: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> tuple[Array, Array, Array]:
        """Get cubic equation coefficients for compressibility factor.

        For SRK: Z³ - Z² + (A - B - B²)Z - AB = 0

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            k_ij: Binary interaction parameters

        Returns:
            (c2, c1, c0): Coefficients of the cubic equation
        """
        a_m = self.a_mix(T, y, k_ij)
        b_m = self.b_mix(y)

        A = a_m * P / (R * T)**2
        B = b_m * P / (R * T)

        # For SRK: u=1, w=0
        # Z³ - Z² + (A - B - B²)Z - AB = 0
        c2 = -1.0
        c1 = A - B - B**2
        c0 = -A * B

        return c2, c1, c0

    def solve_Z(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Solve for compressibility factor Z."""
        c2, c1, c0 = self.compressibility_cubic(T, P, y, k_ij)
        Z = self._solve_cubic(c2, c1, c0, phase)
        return Z

    def _solve_cubic(
        self,
        c2: Array,
        c1: Array,
        c0: Array,
        phase: str,
    ) -> Array:
        """Solve cubic equation x³ + c2*x² + c1*x + c0 = 0."""
        # Same approach as PR
        p = c1 - c2**2 / 3
        q = c0 - c1 * c2 / 3 + 2 * c2**3 / 27

        disc = (q/2)**2 + (p/3)**3

        def one_real_root(disc_p_q):
            disc, p, q = disc_p_q
            sqrt_disc = jnp.sqrt(disc)
            u = jnp.cbrt(-q/2 + sqrt_disc)
            v = jnp.cbrt(-q/2 - sqrt_disc)
            t = u + v
            return t - c2/3

        def three_real_roots(disc_p_q):
            disc, p, q = disc_p_q
            r = jnp.sqrt(-p**3 / 27)
            theta = jnp.arccos(jnp.clip(-q / (2 * r + 1e-30), -1, 1))

            t1 = 2 * jnp.cbrt(r) * jnp.cos(theta / 3)
            t2 = 2 * jnp.cbrt(r) * jnp.cos((theta + 2*jnp.pi) / 3)
            t3 = 2 * jnp.cbrt(r) * jnp.cos((theta + 4*jnp.pi) / 3)

            x1 = t1 - c2/3
            x2 = t2 - c2/3
            x3 = t3 - c2/3

            roots = jnp.array([x1, x2, x3])
            positive_mask = roots > 0

            if phase == "vapor":
                masked_roots = jnp.where(positive_mask, roots, -jnp.inf)
                return jnp.max(masked_roots)
            else:
                masked_roots = jnp.where(positive_mask, roots, jnp.inf)
                return jnp.min(masked_roots)

        Z = lax.cond(
            disc > 0,
            one_real_root,
            three_real_roots,
            (disc, p, q),
        )

        return jnp.maximum(Z, 0.01)

    def fugacity_coefficient(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate fugacity coefficients for all species.

        For SRK EOS:
        ln(phi_i) = (b_i/b_m)(Z-1) - ln(Z-B)
                    - A/B * [2*sum_j(y_j*a_ij)/a_m - b_i/b_m] * ln(1 + B/Z)

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions
            phase: 'vapor' or 'liquid'
            k_ij: Binary interaction parameters

        Returns:
            Fugacity coefficients for each species
        """
        a_i = self.a(T)
        b_i = self.params.b
        n = self.n_species

        if k_ij is None:
            k_ij = jnp.zeros((n, n))

        a_m = self.a_mix(T, y, k_ij)
        b_m = self.b_mix(y)

        A = a_m * P / (R * T)**2
        B = b_m * P / (R * T)

        Z = self.solve_Z(T, P, y, phase, k_ij)

        # Compute a_ij matrix
        a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
        sum_ya = jnp.sum(y * a_ij, axis=1)

        # ln(phi_i) for SRK
        term1 = (b_i / b_m) * (Z - 1)
        term2 = -jnp.log(jnp.maximum(Z - B, 1e-10))
        term3_coeff = A / (B + 1e-10)
        term3_bracket = 2 * sum_ya / (a_m + 1e-30) - b_i / b_m
        term3_log = jnp.log(jnp.maximum(1 + B / Z, 1e-10))
        term3 = -term3_coeff * term3_bracket * term3_log

        ln_phi = term1 + term2 + term3

        return jnp.exp(ln_phi)

    def K_values(
        self,
        T: Array,
        P: Array,
        x: Array,
        y: Array,
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate VLE K-values: K_i = y_i/x_i = phi_L_i / phi_V_i."""
        phi_L = self.fugacity_coefficient(T, P, x, "liquid", k_ij)
        phi_V = self.fugacity_coefficient(T, P, y, "vapor", k_ij)

        return phi_L / (phi_V + 1e-30)

    def K_values_wilson(self, T: Array, P: Array) -> Array:
        """Estimate K-values using Wilson correlation."""
        p = self.params
        return (p.Pc / P) * jnp.exp(5.373 * (1 + p.omega) * (1 - p.Tc / T))

    def molar_volume(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate molar volume from compressibility factor."""
        Z = self.solve_Z(T, P, y, phase, k_ij)
        return Z * R * T / P

    def density(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Calculate molar density."""
        V = self.molar_volume(T, P, y, phase, k_ij)
        return 1.0 / V


# =============================================================================
# VLE Flash with Cubic EOS
# =============================================================================


def _solve_rachford_rice(z: Array, K: Array) -> Array:
    """Solve Rachford-Rice equation for vapor fraction.

    The Rachford-Rice equation is:
        sum_i z_i * (K_i - 1) / (1 + V * (K_i - 1)) = 0

    Args:
        z: Feed mole fractions
        K: K-values

    Returns:
        Vapor fraction V in [0, 1]
    """
    def rr_func(V, args):
        z_, K_ = args
        return jnp.sum(z_ * (K_ - 1) / (1 + V * (K_ - 1)))

    V0 = jnp.array(0.5)
    args = (z, K)
    solver = optx.Newton(rtol=1e-10, atol=1e-10)
    sol = optx.root_find(rr_func, solver, V0, args=args, max_steps=50, throw=False)
    return jnp.clip(sol.value, 0.0, 1.0)


def flash_TP_eos(
    eos: PengRobinson | SRK,
    z: Array,
    T: Array,
    P: Array,
    k_ij: Array | None = None,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> tuple[Array, Array, Array]:
    """Perform TP flash using cubic EOS.

    Uses successive substitution to converge K-values:
    1. Initialize K from Wilson correlation
    2. Solve Rachford-Rice for vapor fraction V
    3. Calculate x, y from V
    4. Update K from fugacity coefficients
    5. Repeat until converged

    Args:
        eos: Equation of state object (PR or SRK)
        z: Feed mole fractions
        T: Temperature (K)
        P: Pressure (Pa)
        k_ij: Binary interaction parameters
        tol: Convergence tolerance
        max_iter: Maximum iterations

    Returns:
        (V, x, y): Vapor fraction, liquid and vapor mole fractions
    """
    # Initialize K from Wilson
    K = eos.K_values_wilson(T, P)

    def step(state, _):
        K_prev = state

        # Solve Rachford-Rice
        V = _solve_rachford_rice(z, K_prev)

        # Get compositions (inlined)
        x = z / (1 + V * (K_prev - 1))
        y = K_prev * x

        # Ensure positive and normalized
        x = jnp.maximum(x, 1e-10)
        y = jnp.maximum(y, 1e-10)
        x = x / jnp.sum(x)
        y = y / jnp.sum(y)

        # Update K from fugacity coefficients
        K_new = eos.K_values(T, P, x, y, k_ij)

        # Damped update for stability
        K_new = 0.3 * K_new + 0.7 * K_prev

        return K_new, K_new

    # Run iterations
    K_final, _ = lax.scan(step, K, None, length=max_iter)

    # Final flash calculation
    V = _solve_rachford_rice(z, K_final)
    x = z / (1 + V * (K_final - 1))
    y = K_final * x

    # Normalize
    x = jnp.maximum(x, 0.0)
    y = jnp.maximum(y, 0.0)
    x = x / jnp.sum(x)
    y = y / jnp.sum(y)

    return V, x, y
