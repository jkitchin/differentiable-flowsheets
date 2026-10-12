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
from dataclasses import dataclass
import jax
import jax.numpy as jnp

from difflow.cache_key import ValueKeyed
from jax import Array, lax

import optimistix as optx

from difflow.params_mixin import ParamsMixin
from difflow.constants import EPS_FUGACITY, EPS_ARCCOS, EPS_DIVISION
from difflow.numerics import safe_divide, safe_log


# Universal gas constant (J/mol/K)
R = 8.314462618


# =============================================================================
# JIT-compiled helper functions for EOS calculations
# =============================================================================


@jax.jit
def _compute_alpha_pr(T: Array, Tc: Array, kappa: Array) -> Array:
    """JIT-compiled alpha function for Peng-Robinson."""
    Tr = T / Tc
    # Safe sqrt for numerical stability
    Tr_safe = jnp.maximum(Tr, 1e-10)
    return (1 + kappa * (1 - jnp.sqrt(Tr_safe)))**2


@jax.jit
def _compute_a_mix(a_i: Array, y: Array, k_ij: Array) -> Array:
    """JIT-compiled mixture 'a' parameter calculation."""
    a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
    return jnp.sum(jnp.outer(y, y) * a_ij)


@jax.jit
def _compute_b_mix(b: Array, y: Array) -> Array:
    """JIT-compiled mixture 'b' parameter calculation."""
    return jnp.sum(y * b)


@jax.jit
def _compute_cubic_coeffs_pr(A: Array, B: Array) -> tuple[Array, Array, Array]:
    """JIT-compiled cubic equation coefficients for Peng-Robinson."""
    c2 = -(1 - B)
    c1 = A - 3*B**2 - 2*B
    c0 = -(A*B - B**2 - B**3)
    return c2, c1, c0


@jax.jit
def _compute_cubic_coeffs_srk(A: Array, B: Array) -> tuple[Array, Array, Array]:
    """JIT-compiled cubic equation coefficients for SRK."""
    c2 = -1.0
    c1 = A - B - B**2
    c0 = -A * B
    return c2, c1, c0


@jax.jit
def _compute_fugacity_pr(
    T: Array,
    P: Array,
    y: Array,
    Z: Array,
    a_i: Array,
    b_i: Array,
    a_m: Array,
    b_m: Array,
    k_ij: Array,
) -> Array:
    """JIT-compiled fugacity coefficient calculation for PR EOS."""
    A = a_m * P / (R * T)**2
    B = b_m * P / (R * T)

    # Compute a_ij matrix
    a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
    sum_ya = jnp.sum(y * a_ij, axis=1)

    # ln(phi_i) calculation
    sqrt2 = jnp.sqrt(2.0)
    term1 = (b_i / b_m) * (Z - 1)
    term2 = -jnp.log(jnp.maximum(Z - B, EPS_FUGACITY))
    term3_coeff = A / (2 * sqrt2 * B + EPS_FUGACITY)
    term3_bracket = 2 * sum_ya / (a_m + EPS_FUGACITY) - b_i / b_m
    term3_log = jnp.log(
        jnp.maximum((Z + (1 + sqrt2) * B) / (Z + (1 - sqrt2) * B + EPS_FUGACITY), EPS_FUGACITY)
    )
    term3 = -term3_coeff * term3_bracket * term3_log

    ln_phi = term1 + term2 + term3
    return jnp.exp(ln_phi)


@jax.jit
def _compute_fugacity_srk(
    T: Array,
    P: Array,
    y: Array,
    Z: Array,
    a_i: Array,
    b_i: Array,
    a_m: Array,
    b_m: Array,
    k_ij: Array,
) -> Array:
    """JIT-compiled fugacity coefficient calculation for SRK EOS."""
    A = a_m * P / (R * T)**2
    B = b_m * P / (R * T)

    # Compute a_ij matrix
    a_ij = jnp.sqrt(jnp.outer(a_i, a_i)) * (1 - k_ij)
    sum_ya = jnp.sum(y * a_ij, axis=1)

    # ln(phi_i) for SRK
    term1 = (b_i / b_m) * (Z - 1)
    term2 = -jnp.log(jnp.maximum(Z - B, EPS_FUGACITY))
    term3_coeff = A / (B + EPS_FUGACITY)
    term3_bracket = 2 * sum_ya / (a_m + EPS_FUGACITY) - b_i / b_m
    term3_log = jnp.log(jnp.maximum(1 + B / Z, EPS_FUGACITY))
    term3 = -term3_coeff * term3_bracket * term3_log

    ln_phi = term1 + term2 + term3
    return jnp.exp(ln_phi)


@jax.jit
def _wilson_k_values(T: Array, P: Array, Tc: Array, Pc: Array, omega: Array) -> Array:
    """JIT-compiled Wilson K-value estimation."""
    return (Pc / P) * jnp.exp(5.373 * (1 + omega) * (1 - Tc / T))


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


@dataclass(repr=False)
class EOSParams(ParamsMixin):
    """Parameters for equation of state calculations.

    Precomputed from critical properties for efficiency.

    Attributes:
        a_c: Critical 'a' parameter for each species
        b: 'b' parameter for each species
        kappa: Temperature dependence parameter
        Tc: Critical temperatures (K)
        Pc: Critical pressures (Pa)
        omega: Acentric factors (dimensionless)
        species_order: List of species names for array ordering
    """
    a_c: Array
    b: Array
    kappa: Array
    Tc: Array
    Pc: Array
    omega: Array
    species_order: list[str]
    k_ij: Array | None = None

    def __post_init__(self):
        """Validate EOS parameters."""
        if not self.species_order:
            raise ValueError("species_order cannot be empty")
        n = len(self.species_order)
        # Check array dimensions match species count
        for name in ['a_c', 'b', 'kappa', 'Tc', 'Pc', 'omega']:
            arr = getattr(self, name)
            if hasattr(arr, 'shape') and arr.shape[0] != n:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected ({n},) for "
                    f"{n} species"
                )


def build_kij_matrix(
    species_order: list[str],
    kij_dict: dict[tuple[str, str], float],
) -> Array:
    """Build symmetric binary interaction parameter matrix from a dict.

    Args:
        species_order: Ordered list of species names
        kij_dict: Dictionary mapping species pairs to k_ij values,
                  e.g. {("methane", "ethane"): 0.02}. Order of the pair
                  does not matter (matrix is symmetric).

    Returns:
        n x n JAX array of binary interaction parameters
    """
    n = len(species_order)
    k = jnp.zeros((n, n))
    name_to_idx = {name: i for i, name in enumerate(species_order)}
    for (s1, s2), val in kij_dict.items():
        i = name_to_idx.get(s1)
        j = name_to_idx.get(s2)
        if i is not None and j is not None:
            k = k.at[i, j].set(val)
            k = k.at[j, i].set(val)
    return k


class PengRobinson(ValueKeyed):
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

    def __init__(
        self,
        species_data: dict[str, CriticalProperties],
        k_ij: dict[tuple[str, str], float] | Array | None = None,
    ):
        """Initialize with species critical properties.

        Args:
            species_data: Dictionary mapping species names to CriticalProperties
            k_ij: Binary interaction parameters. Can be:
                  - None: all k_ij = 0 (default)
                  - dict: {("species1", "species2"): value} pairs
                  - Array: n x n matrix directly
        """
        self.species = species_data
        self._species_order = list(species_data.keys())
        self.n_species = len(self._species_order)
        self._kij_input = k_ij
        self.params = self._compute_params()
        # Keyed on what it was built from, not on the arrays that came out of
        # it -- those can be tracers, and are a pure function of these anyway.
        self._set_value_key(species_data, k_ij, derived=self.params)

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

        # Binary interaction parameters
        k_ij_arr = None
        if self._kij_input is not None:
            if isinstance(self._kij_input, dict):
                k_ij_arr = build_kij_matrix(self._species_order, self._kij_input)
            else:
                k_ij_arr = jnp.asarray(self._kij_input)

        return EOSParams(
            a_c=a_c,
            b=b,
            kappa=kappa,
            Tc=Tc,
            Pc=Pc,
            omega=omega,
            species_order=self._species_order,
            k_ij=k_ij_arr,
        )

    def alpha(self, T: Array) -> Array:
        """Calculate alpha function for temperature dependence.

        alpha(T) = [1 + kappa*(1 - sqrt(T/Tc))]²

        Args:
            T: Temperature (K)

        Returns:
            Alpha values for each species
        """
        return _compute_alpha_pr(T, self.params.Tc, self.params.kappa)

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
                  If None, uses stored params.k_ij or zeros.

        Returns:
            Mixture 'a' parameter
        """
        a_i = self.a(T)
        if k_ij is None:
            k_ij = self.params.k_ij if self.params.k_ij is not None else jnp.zeros((self.n_species, self.n_species))
        return _compute_a_mix(a_i, y, k_ij)

    def b_mix(self, y: Array) -> Array:
        """Calculate mixture 'b' parameter using linear mixing rule.

        b_mix = sum_i y_i * b_i

        Args:
            y: Mole fractions (array in species_order)

        Returns:
            Mixture 'b' parameter
        """
        return _compute_b_mix(self.params.b, y)

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
        B = self.b_mix(y) * P / (R * T)

        # Solve cubic using Cardano's formula (JAX-compatible)
        Z = self._solve_cubic(c2, c1, c0, phase, Z_min=B)

        return Z

    def _solve_cubic(
        self,
        c2: Array,
        c1: Array,
        c0: Array,
        phase: str,
        Z_min: Array = 0.0,
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
            theta = jnp.arccos(jnp.clip(safe_divide(-q, 2 * r, EPS_ARCCOS), -1, 1))

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

        # Physical bound: Z > B (the co-volume). A fixed floor such as 0.01
        # replaced the liquid root of most liquids near 1 bar (issue #417).
        return jnp.maximum(Z, Z_min * (1.0 + 1e-8))

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
            k_ij = self.params.k_ij if self.params.k_ij is not None else jnp.zeros((n, n))

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
        term2 = -safe_log(jnp.maximum(Z - B, EPS_DIVISION))
        term3_coeff = safe_divide(A, 2 * sqrt2 * B)
        term3_bracket = safe_divide(2 * sum_ya, a_m, EPS_FUGACITY) - b_i / b_m
        term3_log = safe_log(
            jnp.maximum(safe_divide(Z + (1 + sqrt2) * B, Z + (1 - sqrt2) * B), EPS_DIVISION)
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
            k_ij: Binary interaction parameters (uses stored if None)

        Returns:
            K-values for each species
        """
        phi_L = self.fugacity_coefficient(T, P, x, "liquid", k_ij)
        phi_V = self.fugacity_coefficient(T, P, y, "vapor", k_ij)

        return safe_divide(phi_L, phi_V, EPS_FUGACITY)

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
        return _wilson_k_values(T, P, p.Tc, p.Pc, p.omega)

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

    def enthalpy_departure(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Molar enthalpy departure H(T,P) - H_ideal_gas(T) from Peng-Robinson.

        H_dep = R*T*(Z - 1)
                + (T*da_m/dT - a_m) / (2*sqrt(2)*b_m)
                  * ln[(Z + (1 + sqrt(2))*B) / (Z + (1 - sqrt(2))*B)]

        Added to an ideal-gas enthalpy this gives the true PR enthalpy -- the
        same ideal-gas-plus-departure decomposition a cubic-EOS property
        package (e.g. IDAES's) uses. Near the critical region this term and
        its temperature derivative are large, so for a near-critical vapor the
        real-gas effective heat capacity differs substantially from the
        ideal-gas value; that correction is what a departure-free model misses.

        The mixture-a temperature derivative da_m/dT is taken with forward-mode
        autodiff (jax.jvp), so the result stays exact and differentiable under
        an outer jax.grad of the flowsheet.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions (array in species_order)
            phase: 'vapor' or 'liquid' (selects the Z root)
            k_ij: Binary interaction parameters

        Returns:
            Molar enthalpy departure (J/mol). Negative for typical vapors.
        """
        T = jnp.asarray(T)
        b_m = self.b_mix(y)
        Z = self.solve_Z(T, P, y, phase, k_ij)
        B = b_m * P / (R * T)

        # a_m and its temperature derivative in one forward-mode pass.
        a_m, da_m_dT = jax.jvp(
            lambda t: self.a_mix(t, y, k_ij), (T,), (jnp.ones_like(T),)
        )

        sqrt2 = jnp.sqrt(2.0)
        log_arg = (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
        return R * T * (Z - 1.0) + (T * da_m_dT - a_m) / (
            2.0 * sqrt2 * b_m
        ) * safe_log(log_arg)

    def entropy_departure(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Molar entropy departure S(T,P) - S_ideal_gas(T,P) from Peng-Robinson.

        S_dep = R*ln(Z - B)
                + (da_m/dT) / (2*sqrt(2)*b_m)
                  * ln[(Z + (1 + sqrt(2))*B) / (Z + (1 - sqrt(2))*B)]

        This is the entropy analogue of :meth:`enthalpy_departure`, sharing its
        Z, B and the same forward-mode ``da_m/dT``; the ``R*ln(Z - B)`` term
        replaces enthalpy's ``R*T*(Z - 1)`` and the bracket coefficient drops the
        ``T*da/dT - a`` combination for a bare ``da/dT``. Added to a
        temperature-dependent ideal-gas entropy it gives the true PR entropy, so
        an isentropic unit (turboexpander, compressor, pump) can match entropy
        across a pressure change. Negative for a typical compressed vapor.

        The mixture-a temperature derivative da_m/dT is taken with forward-mode
        autodiff (jax.jvp), so the result stays exact and differentiable under
        an outer jax.grad of the flowsheet.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions (array in species_order)
            phase: 'vapor' or 'liquid' (selects the Z root)
            k_ij: Binary interaction parameters

        Returns:
            Molar entropy departure (J/mol/K). Negative for typical vapors.
        """
        T = jnp.asarray(T)
        b_m = self.b_mix(y)
        Z = self.solve_Z(T, P, y, phase, k_ij)
        B = b_m * P / (R * T)

        _, da_m_dT = jax.jvp(
            lambda t: self.a_mix(t, y, k_ij), (T,), (jnp.ones_like(T),)
        )

        sqrt2 = jnp.sqrt(2.0)
        log_arg = (Z + (1.0 + sqrt2) * B) / (Z + (1.0 - sqrt2) * B)
        # Z - B > 0 for a physical root; safe_log guards the cryogenic liquid root.
        return R * safe_log(Z - B) + da_m_dT / (2.0 * sqrt2 * b_m) * safe_log(log_arg)


class SRK(ValueKeyed):
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

    def __init__(
        self,
        species_data: dict[str, CriticalProperties],
        k_ij: dict[tuple[str, str], float] | Array | None = None,
    ):
        """Initialize with species critical properties.

        Args:
            species_data: Dictionary mapping species names to CriticalProperties
            k_ij: Binary interaction parameters. Can be:
                  - None: all k_ij = 0 (default)
                  - dict: {("species1", "species2"): value} pairs
                  - Array: n x n matrix directly
        """
        self.species = species_data
        self._species_order = list(species_data.keys())
        self.n_species = len(self._species_order)
        self._kij_input = k_ij
        self.params = self._compute_params()
        # Keyed on what it was built from, not on the arrays that came out of
        # it -- those can be tracers, and are a pure function of these anyway.
        self._set_value_key(species_data, k_ij, derived=self.params)

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

        # Binary interaction parameters
        k_ij_arr = None
        if self._kij_input is not None:
            if isinstance(self._kij_input, dict):
                k_ij_arr = build_kij_matrix(self._species_order, self._kij_input)
            else:
                k_ij_arr = jnp.asarray(self._kij_input)

        return EOSParams(
            a_c=a_c,
            b=b,
            kappa=kappa,
            Tc=Tc,
            Pc=Pc,
            omega=omega,
            species_order=self._species_order,
            k_ij=k_ij_arr,
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
            k_ij = self.params.k_ij if self.params.k_ij is not None else jnp.zeros((n, n))

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
        B = self.b_mix(y) * P / (R * T)
        return self._solve_cubic(c2, c1, c0, phase, Z_min=B)

    def _solve_cubic(
        self,
        c2: Array,
        c1: Array,
        c0: Array,
        phase: str,
        Z_min: Array = 0.0,
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
            theta = jnp.arccos(jnp.clip(safe_divide(-q, 2 * r, EPS_ARCCOS), -1, 1))

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

        # Physical bound: Z > B (the co-volume), not a fixed floor (issue #417).
        return jnp.maximum(Z, Z_min * (1.0 + 1e-8))

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
            k_ij = self.params.k_ij if self.params.k_ij is not None else jnp.zeros((n, n))

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
        term2 = -safe_log(jnp.maximum(Z - B, EPS_DIVISION))
        term3_coeff = safe_divide(A, B)
        term3_bracket = safe_divide(2 * sum_ya, a_m, EPS_FUGACITY) - b_i / b_m
        term3_log = safe_log(jnp.maximum(1 + safe_divide(B, Z), EPS_DIVISION))
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

        return safe_divide(phi_L, phi_V, EPS_FUGACITY)

    def K_values_wilson(self, T: Array, P: Array) -> Array:
        """Estimate K-values using Wilson correlation."""
        p = self.params
        return _wilson_k_values(T, P, p.Tc, p.Pc, p.omega)

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

    def enthalpy_departure(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Molar enthalpy departure H(T,P) - H_ideal_gas(T) from SRK.

        H_dep = R*T*(Z - 1)
                + (T*da_m/dT - a_m) / b_m * ln[(Z + B) / Z]

        This is the SRK (epsilon=0, sigma=1) form of the same generic-cubic
        departure ``PengRobinson.enthalpy_departure`` gives; the log term
        collapses from PR's ``ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]/(2 sqrt2 b)`` to
        ``ln[(Z+B)/Z]/b`` because SRK's second EOS constant is zero. Providing it
        here lets :class:`~difflow.thermo.CubicThermo` (and every consumer of it:
        the enthalpy-based CSTR and heat exchanger) work with an SRK EOS as well
        as a Peng-Robinson one.

        The mixture-a temperature derivative da_m/dT is taken with forward-mode
        autodiff (jax.jvp), so the result stays exact and differentiable under
        an outer jax.grad of the flowsheet.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions (array in species_order)
            phase: 'vapor' or 'liquid' (selects the Z root)
            k_ij: Binary interaction parameters

        Returns:
            Molar enthalpy departure (J/mol). Negative for typical vapors.
        """
        T = jnp.asarray(T)
        b_m = self.b_mix(y)
        Z = self.solve_Z(T, P, y, phase, k_ij)
        B = b_m * P / (R * T)

        # a_m and its temperature derivative in one forward-mode pass.
        a_m, da_m_dT = jax.jvp(
            lambda t: self.a_mix(t, y, k_ij), (T,), (jnp.ones_like(T),)
        )

        log_arg = (Z + B) / Z
        return R * T * (Z - 1.0) + (T * da_m_dT - a_m) / b_m * safe_log(log_arg)

    def entropy_departure(
        self,
        T: Array,
        P: Array,
        y: Array,
        phase: Literal["vapor", "liquid"] = "vapor",
        k_ij: Array | None = None,
    ) -> Array:
        """Molar entropy departure S(T,P) - S_ideal_gas(T,P) from SRK.

        S_dep = R*ln(Z - B) + (da_m/dT) / b_m * ln[(Z + B) / Z]

        This is the SRK (epsilon=0, sigma=1) form of the generic-cubic entropy
        departure that :meth:`PengRobinson.entropy_departure` gives; the log term
        collapses from PR's ``ln[(Z+(1+sqrt2)B)/(Z+(1-sqrt2)B)]/(2 sqrt2 b)`` to
        ``ln[(Z+B)/Z]/b`` because SRK's second EOS constant is zero. Providing it
        here lets an SRK EOS drive the same isentropic units (turboexpander,
        compressor) as a Peng-Robinson one.

        The mixture-a temperature derivative da_m/dT is taken with forward-mode
        autodiff (jax.jvp), so the result stays exact and differentiable under
        an outer jax.grad of the flowsheet.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            y: Mole fractions (array in species_order)
            phase: 'vapor' or 'liquid' (selects the Z root)
            k_ij: Binary interaction parameters

        Returns:
            Molar entropy departure (J/mol/K). Negative for typical vapors.
        """
        T = jnp.asarray(T)
        b_m = self.b_mix(y)
        Z = self.solve_Z(T, P, y, phase, k_ij)
        B = b_m * P / (R * T)

        _, da_m_dT = jax.jvp(
            lambda t: self.a_mix(t, y, k_ij), (T,), (jnp.ones_like(T),)
        )

        log_arg = (Z + B) / Z
        return R * safe_log(Z - B) + da_m_dT / b_m * safe_log(log_arg)


# =============================================================================
# VLE Flash with Cubic EOS
# =============================================================================


def _solve_rachford_rice(z: Array, K: Array, n_bisect: int = 60) -> Array:
    """Solve Rachford-Rice equation for vapor fraction.

    The Rachford-Rice equation is:
        sum_i z_i * (K_i - 1) / (1 + V * (K_i - 1)) = 0

    The function has poles at ``V = 1 / (1 - K_i)`` which always lie *outside*
    ``[0, 1]`` (negative for ``K_i > 1``, greater than 1 for ``K_i < 1``). On
    ``[0, 1]`` the function is therefore smooth and strictly decreasing, so a
    bracketed bisection converges robustly to the unique two-phase root. An
    unbracketed Newton iteration (the previous approach) could instead overshoot
    a pole just outside the interval and diverge, clipping to a spurious
    ``V = 1`` for wide-K-spread mixtures (see issue #169).

    Bisection runs under ``stop_gradient``; a single analytic Newton step then
    restores exact implicit-function-theorem gradients ``dV/dK`` without changing
    the (already-converged) value.

    Args:
        z: Feed mole fractions
        K: K-values
        n_bisect: Number of bisection iterations (2**-n resolution)

    Returns:
        Vapor fraction V in [0, 1]
    """
    def rr_func(V: Array) -> Array:
        return jnp.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0)))

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        # rr_func is decreasing: where it is positive the root lies to the right.
        go_right = rr_func(mid) > 0.0
        lo = jnp.where(go_right, mid, lo)
        hi = jnp.where(go_right, hi, mid)
        return (lo, hi), None

    (lo, hi), _ = lax.scan(body, (jnp.array(0.0), jnp.array(1.0)), None, length=n_bisect)
    V_star = jax.lax.stop_gradient(0.5 * (lo + hi))

    # One analytic Newton step for exact implicit gradients. At convergence
    # rr_func(V_star) ~ 0 so the value is unchanged, but the expression carries
    # dV/dK = -f_K / f_V through autodiff.
    fp = jax.grad(rr_func)(V_star)
    V = V_star - rr_func(V_star) / jnp.where(fp == 0.0, -1.0, fp)
    return jnp.clip(V, 0.0, 1.0)


def phase_identification_parameter(
    eos: "PengRobinson | SRK",
    T: Array,
    P: Array,
    y: Array,
    phase: Literal["vapor", "liquid"] = "vapor",
    k_ij: Array | None = None,
) -> Array:
    """Venkatarathnam-Oellrich phase identification parameter (PIP).

    .. math::

        \\Pi = V\\left[\\frac{\\partial^2 P/\\partial T\\partial V}
        {(\\partial P/\\partial T)_V}
        - \\frac{(\\partial^2 P/\\partial V^2)_T}{(\\partial P/\\partial V)_T}\\right]

    evaluated on the cubic ``P(T, V) = RT/(V-b) - a(T)/(V^2 + u b V + w b^2)``
    at the molar volume of the requested root. A single phase with
    ``Pi > 1`` is liquid-like and one with ``Pi < 1`` is vapour-like
    (Venkatarathnam and Oellrich, *Fluid Phase Equilib.* 301 (2011) 225).
    Unlike "which cubic root is it" the test still works when the cubic has
    only one real root, as for a compressed liquid above its bubble pressure
    or a supercritical fluid, where the "vapour" and "liquid" roots coincide.

    Args:
        eos: Peng-Robinson or SRK object.
        T: Temperature (K).
        P: Pressure (Pa).
        y: Mole fractions (array in ``eos.species_order``).
        phase: Which cubic root to evaluate (they coincide when the cubic
            has a single real root).
        k_ij: Binary interaction parameters (uses the stored ones if None).

    Returns:
        The dimensionless PIP (differentiable in T, P and y).

    Example:
        >>> from difflow.eos import PengRobinson, phase_identification_parameter
        >>> from difflow.database import get_critical_props
        >>> pr = PengRobinson({"propane": get_critical_props("propane")})
        >>> float(phase_identification_parameter(pr, 300.0, 50e5, jnp.array([1.0]))) > 1
        True
    """
    T = jnp.asarray(T, dtype=float)
    Vm = eos.molar_volume(T, P, y, phase, k_ij)
    b = eos.b_mix(y)
    u, w = eos.u, eos.w

    def pressure(TT, VV):
        a = eos.a_mix(TT, y, k_ij)
        return R * TT / (VV - b) - a / (VV**2 + u * b * VV + w * b**2)

    dP_dV = jax.grad(pressure, argnums=1)
    dP_dT = jax.grad(pressure, argnums=0)
    d2P_dV2 = jax.grad(dP_dV, argnums=1)(T, Vm)
    d2P_dTdV = jax.grad(dP_dV, argnums=0)(T, Vm)
    return Vm * (safe_divide(d2P_dTdV, dP_dT(T, Vm))
                 - safe_divide(d2P_dV2, dP_dV(T, Vm)))


def _phase_stability(
    eos: "PengRobinson | SRK",
    z: Array,
    T: Array,
    P: Array,
    k_ij: Array | None = None,
    n_iter: int = 30,
) -> tuple[Array, Array]:
    """Michelsen two-sided tangent-plane phase-stability test.

    Decides whether a feed z at (T, P) is single-phase or splits into two
    phases, without relying on a TP-flash that can drift to the spurious
    trivial root (all K_i -> 1) near the critical region. From a vapor-like
    and a liquid-like trial phase it minimizes the tangent-plane distance; if
    either trial finds a distinct phase with sum(W) > 1, a second phase lowers
    the Gibbs energy and the feed is two-phase.

    Returns (two_phase, feed_is_vapor):
      - two_phase: boolean array, True if the feed splits.
      - feed_is_vapor: boolean array, which single phase the feed is (used only
        when single-phase, to return V = 1 vs 0): the lower-Gibbs cubic root at
        the feed composition, labelled liquid when its phase identification
        parameter (:func:`phase_identification_parameter`) exceeds 1 and T is
        below the Kay's-rule pseudo-critical temperature sum(z_i Tc_i), and
        vapour otherwise (so a supercritical fluid is reported as vapour).

    Fixed-length ``lax.scan`` trial iterations keep this differentiable.
    """
    z = z / jnp.sum(z)
    lnz = safe_log(jnp.maximum(z, EPS_DIVISION))

    # Feed fugacity from the lower-Gibbs (stable) single-phase root.
    lnphi_zv = safe_log(eos.fugacity_coefficient(T, P, z, "vapor", k_ij))
    lnphi_zl = safe_log(eos.fugacity_coefficient(T, P, z, "liquid", k_ij))
    g_v = jnp.sum(z * (lnz + lnphi_zv))
    g_l = jnp.sum(z * (lnz + lnphi_zl))
    stable_is_vapor_root = g_v <= g_l
    # Which phase the stable root IS: by the phase identification parameter
    # of that root, not by which root it is. When the cubic has one real root
    # (a compressed liquid above its bubble pressure, a dense supercritical
    # fluid) the "vapour" and "liquid" roots coincide, g_v == g_l, and the
    # old ``g_v <= g_l`` test called every such fluid a vapour (issue #442).
    # Above the (Kay's rule) pseudo-critical temperature a single phase is
    # reported as vapour, the usual simulator convention: a hot dilute gas
    # is repulsion-dominated and has Pi slightly above 1 (an ideal gas has
    # Pi = 1 exactly), so PIP alone would call it a liquid.
    pip_v = phase_identification_parameter(eos, T, P, z, "vapor", k_ij)
    pip_l = phase_identification_parameter(eos, T, P, z, "liquid", k_ij)
    pip = jnp.where(stable_is_vapor_root, pip_v, pip_l)
    T_pc = jnp.sum(z * eos.params.Tc)
    feed_is_vapor = (T >= T_pc) | (pip < 1.0)
    lnphi_feed = jnp.where(feed_is_vapor, lnphi_zv, lnphi_zl)
    d = lnz + lnphi_feed  # tangent-plane reference

    K = eos.K_values_wilson(T, P)
    lnK = safe_log(jnp.maximum(K, EPS_DIVISION))

    def run_trial(lnW0, trial_phase):
        def step(lnW, _):
            w = jnp.exp(lnW)
            w = jnp.maximum(w, EPS_DIVISION)
            w = w / jnp.sum(w)
            lnphi_w = safe_log(eos.fugacity_coefficient(T, P, w, trial_phase, k_ij))
            return d - lnphi_w, None

        lnW, _ = lax.scan(step, lnW0, None, length=n_iter)
        W = jnp.exp(lnW)
        S = jnp.sum(W)
        w = W / S
        trivial = jnp.sum((w - z) ** 2) < 1e-8
        return S, trivial

    # Vapor-like trial (search for a lighter phase); liquid-like trial (heavier).
    Sv, triv_v = run_trial(lnz + lnK, "vapor")
    Sl, triv_l = run_trial(lnz - lnK, "liquid")

    tol = 1e-4
    unstable_v = (Sv > 1.0 + tol) & (~triv_v)
    unstable_l = (Sl > 1.0 + tol) & (~triv_l)
    two_phase = unstable_v | unstable_l
    return two_phase, feed_is_vapor


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
        (V, x, y): Vapor fraction, liquid and vapor mole fractions.

    A Michelsen phase-stability test (:func:`_phase_stability`) first decides
    whether the feed is single- or two-phase. This guards against the classic
    trivial-root failure of successive substitution near the critical region,
    where the K-value iteration can drift to the spurious all-vapor/all-liquid
    (K_i -> 1) solution and report a wrong split. When the feed is single-phase,
    V is returned as exactly 1 (vapor) or 0 (liquid); the two-phase iteration
    below is used only when a distinct second phase genuinely lowers the Gibbs
    energy.
    """
    z = z / jnp.sum(z)
    two_phase, feed_is_vapor = _phase_stability(eos, z, T, P, k_ij)

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

    # Two-phase flash result
    V_tp = _solve_rachford_rice(z, K_final)
    x_tp = z / (1 + V_tp * (K_final - 1))
    y_tp = K_final * x_tp
    x_tp = jnp.maximum(x_tp, 0.0)
    y_tp = jnp.maximum(y_tp, 0.0)
    x_tp = x_tp / jnp.sum(x_tp)
    y_tp = y_tp / jnp.sum(y_tp)

    # Select two-phase split vs. single-phase (V = 1 vapor, 0 liquid). When
    # single-phase, x and y both collapse to the feed composition.
    V_single = jnp.where(feed_is_vapor, 1.0, 0.0)
    V = jnp.where(two_phase, V_tp, V_single)
    x = jnp.where(two_phase, x_tp, z)
    y = jnp.where(two_phase, y_tp, z)

    return V, x, y


# =============================================================================
# Bubble and dew points with a cubic EOS
# =============================================================================

def _eos_saturation(eos, z, T, P, k_ij, kind: str, unknown: str, guess=None,
                    n_sub: int = 100) -> dict:
    """Bubble/dew T or P of a cubic EOS (phi-phi), differentiable.

    ``kind`` is ``"bubble"`` (``z`` is the liquid, the incipient phase is
    vapor ``y = K z``) or ``"dew"`` (``z`` is the vapor, incipient liquid
    ``x = z / K``); ``unknown`` is ``"P"`` or ``"T"``. The unknowns
    ``u = [ln K_1..ln K_n, ln s]`` (``s`` = P or T) solve

    - ``ln K_i + ln phi_i^V(y) - ln phi_i^L(x) = 0``
    - ``ln sum(K z) = 0`` (bubble) or ``ln sum(z / K) = 0`` (dew)

    by Newton's method (optimistix, implicit differentiation), started from
    Wilson K-values and ``n_sub`` successive-substitution steps (stopped
    gradient: only the converged root carries derivatives).
    """
    from difflow.thermo import BubbleDewConvergenceWarning
    import warnings
    import numpy as np

    z = jnp.asarray(z, dtype=float)
    z = z / jnp.sum(z)
    n = z.shape[0]
    p = eos.params
    a_w = 5.373 * (1.0 + p.omega) * p.Tc  # -d ln K_wilson / d(1/T)
    bubble = kind == "bubble"
    sgn = 1.0 if bubble else -1.0
    # K = 1 is the answer, not a failure, for a pure component (or a mixture
    # with one species present): x = y and the saturation point is Psat/Tsat.
    multicomp = jnp.sum(z > 1e-12) >= 2

    def phases(lnK):
        K = jnp.exp(lnK)
        if bubble:
            S = jnp.sum(K * z)
            return z, K * z / S, S
        S = jnp.sum(z / K)
        return (z / K) / S, z, S

    def TP(ls, args):
        T_, P_ = args
        return (jnp.exp(ls), P_) if unknown == "T" else (T_, jnp.exp(ls))

    def resid(u, args):
        lnK, ls = u[:n], u[n]
        T_, P_ = TP(ls, args)
        x, y, S = phases(lnK)
        lnphiL = safe_log(eos.fugacity_coefficient(T_, P_, x, "liquid", k_ij))
        lnphiV = safe_log(eos.fugacity_coefficient(T_, P_, y, "vapor", k_ij))
        return jnp.concatenate([lnK + lnphiV - lnphiL, jnp.atleast_1d(jnp.log(S))])

    args = (None if T is None else jnp.asarray(T, dtype=float),
            None if P is None else jnp.asarray(P, dtype=float))

    def s_step(ls, lnK, lnS, damp=1.0):
        # Newton-like update of ln P (K ~ 1/P) or of 1/T (Wilson slope).
        lnS = damp * jnp.clip(lnS, -0.5, 0.5)
        if unknown == "P":
            return ls + sgn * lnS
        x, y, _ = phases(lnK)
        w = y if bubble else x
        invT = jnp.exp(-ls) + sgn * lnS / jnp.sum(w * a_w)
        return -jnp.log(jnp.maximum(invT, 1e-6))

    # 1) Wilson K-values and a consistent unknown.
    if guess is not None:
        ls0 = jnp.log(jnp.asarray(guess, dtype=float))
    elif unknown == "P":
        Kw1 = _wilson_k_values(args[0], 1.0, p.Tc, p.Pc, p.omega)  # K * P
        ls0 = jnp.log(jnp.sum(Kw1 * z)) if bubble else -jnp.log(jnp.sum(z / Kw1))
    else:
        ls0 = jnp.log(jnp.sum(z * p.Tc) * 0.7)

    def lnK_wilson(ls):
        T_, P_ = TP(ls, args)
        return jnp.log(_wilson_k_values(T_, P_, p.Tc, p.Pc, p.omega))

    if unknown == "T" and guess is None:
        def wstep(ls, _):
            lnK = lnK_wilson(ls)
            S = phases(lnK)[2]
            return s_step(ls, lnK, jnp.log(S)), None
        ls0, _ = lax.scan(wstep, ls0, None, length=60)

    # 2) Successive substitution on K with phi, safeguarded against the
    #    trivial root: an iterate whose K-values collapse towards 1 (or go
    #    non-finite) is rejected; the unknown moves back halfway to the last
    #    good point, the K-values are restored, and the step is damped.
    def sstep(c, _):
        lnK, ls, lnK_g, ls_g, damp = c
        T_, P_ = TP(ls, args)
        x, y, _S = phases(lnK)
        lnK_new = (safe_log(eos.fugacity_coefficient(T_, P_, x, "liquid", k_ij))
                   - safe_log(eos.fugacity_coefficient(T_, P_, y, "vapor", k_ij)))
        S = phases(lnK_new)[2]
        bad = (~jnp.all(jnp.isfinite(lnK_new)) | ~jnp.isfinite(S)
               | (multicomp & (jnp.max(jnp.abs(lnK_new)) < 0.02)))
        lnK_new = jnp.where(jnp.all(jnp.isfinite(lnK_new)), lnK_new, lnK_g)
        ls_ok = s_step(ls, lnK_new, jnp.log(jnp.where(jnp.isfinite(S), S, 1.0)), damp)
        return (jnp.where(bad, lnK_g, lnK_new),
                jnp.where(bad, 0.5 * (ls + ls_g), ls_ok),
                jnp.where(bad, lnK_g, lnK_new),
                jnp.where(bad, ls_g, ls),
                jnp.where(bad, 0.5 * damp, jnp.minimum(1.0, 1.25 * damp))), None

    def attempt(ls_start):
        lnKw = lnK_wilson(ls_start)
        (lnK0, ls1, _, _, _), _ = lax.scan(
            sstep, (lnKw, ls_start, lnKw, ls_start, jnp.asarray(1.0)), None,
            length=n_sub)
        u0 = lax.stop_gradient(jnp.concatenate([lnK0, jnp.atleast_1d(ls1)]))
        # 3) Newton with implicit differentiation.
        sol = optx.root_find(resid, optx.Newton(rtol=1e-10, atol=1e-10), u0,
                             args=args, max_steps=60, throw=False)
        # A Newton step that left the domain (e.g. no saturation point: above
        # the mixture critical region) falls back to the substitution iterate,
        # so the flags below describe where the iteration actually went.
        u_ = jnp.where(jnp.all(jnp.isfinite(sol.value)), sol.value, u0)
        r_ = resid(u_, args)
        ok = (jnp.all(jnp.isfinite(u_)) & (jnp.max(jnp.abs(r_)) < 1e-8)
              & (~multicomp | (jnp.max(jnp.abs(u_[:n])) >= 1e-3)))
        return u_, ok

    if guess is not None:
        u, _ = attempt(ls0)
    else:
        # Start on the far side of the trivial root: a pressure well below
        # (or a temperature 5% below) the Wilson estimate, where the two phases
        # are distinct. Approached from above, substitution slides onto
        # K = 1 near the critical region (e.g. CH4/C3H8 at 300 K). Near the
        # critical region no one start serves every mixture (a C7/C8 liquid
        # at 20 bar needs a start closer to Wilson), so a few are tried and
        # the first converged, non-trivial root is kept; derivatives flow
        # through the selected root only.
        offsets = (0.25, 0.1, 0.5, 1.0) if unknown == "P" else (0.95, 0.9, 1.0, 0.85)
        us, oks = zip(*[attempt(ls0 + jnp.log(f)) for f in offsets])
        oks = jnp.stack(oks)
        pick = jnp.where(jnp.any(oks), jnp.argmax(oks), 0)
        u = jnp.stack(us)[pick]
    lnK, ls = u[:n], u[n]
    T_out, P_out = TP(ls, args)
    x, y, _ = phases(lnK)
    r = resid(u, args)
    converged = jnp.all(jnp.isfinite(u)) & (jnp.max(jnp.abs(r)) < 1e-8)
    trivial = multicomp & (jnp.max(jnp.abs(lnK)) < 1e-3)
    out = {"T": jnp.asarray(T_out), "P": jnp.asarray(P_out), "K": jnp.exp(lnK),
           "x": x, "y": y, "converged": converged, "trivial": trivial}

    try:  # concrete-only diagnostics
        conv_c, triv_c = bool(np.asarray(converged)), bool(np.asarray(trivial))
    except Exception:
        return out
    name = f"eos_{kind}_{unknown}"
    if triv_c:
        warnings.warn(f"{name} converged to the trivial solution (all K = 1): "
                      "the state is near or beyond the mixture critical point, "
                      "or the guess was poor", BubbleDewConvergenceWarning,
                      stacklevel=3)
    elif not conv_c:
        warnings.warn(f"{name} did not converge (residual "
                      f"{float(np.max(np.abs(np.asarray(r)))):.3g})",
                      BubbleDewConvergenceWarning, stacklevel=3)
    return out


def eos_bubble_P(eos: "PengRobinson | SRK", x: Array, T: Array | float,
                 k_ij: Array | None = None,
                 P_guess: Array | float | None = None) -> dict:
    """Bubble-point pressure of liquid ``x`` at ``T`` with a cubic EOS.

    Solves the phi-phi equilibrium ``y_i phi_i^V = x_i phi_i^L`` with
    ``sum y = 1`` (Newton from Wilson K-values and successive substitution;
    implicit differentiation, so ``jax.grad`` through the result is exact).

    Args:
        eos: ``PengRobinson`` or ``SRK``.
        x: Liquid mole fractions (in ``eos.species_order``).
        T: Temperature (K).
        k_ij: Binary interaction parameters (default: the EOS's own).
        P_guess: Optional initial pressure (Pa).

    Returns:
        dict with ``P`` (Pa), ``T``, ``y`` (incipient vapor), ``x``, ``K``,
        ``converged`` and ``trivial`` (bool arrays). ``trivial`` is True when
        the iteration collapsed onto ``K = 1`` (``y = x``), which happens near
        or above the mixture critical point; the ``P`` is then meaningless.

    Warns:
        BubbleDewConvergenceWarning: concrete result that is unconverged or
            trivial.

    Example:
        >>> from difflow import PengRobinson
        >>> from difflow.database import get_critical_props
        >>> eos = PengRobinson({s: get_critical_props(s) for s in ("methane", "propane")})
        >>> r = eos_bubble_P(eos, [0.2, 0.8], 277.6)
        >>> 30e5 < float(r["P"]) < 37e5 and bool(r["converged"])
        True
    """
    return _eos_saturation(eos, x, T, None, k_ij, "bubble", "P", P_guess)


def eos_dew_P(eos: "PengRobinson | SRK", y: Array, T: Array | float,
              k_ij: Array | None = None,
              P_guess: Array | float | None = None) -> dict:
    """Dew-point pressure of vapor ``y`` at ``T`` with a cubic EOS.

    As :func:`eos_bubble_P` with the roles swapped: solves for ``P`` and the
    incipient liquid ``x`` (``sum x = 1``). Returns the same dict (``x`` is
    the dew liquid). Mixtures above their cricondentherm have no dew point
    and come back ``trivial`` or unconverged.
    """
    return _eos_saturation(eos, y, T, None, k_ij, "dew", "P", P_guess)


def eos_bubble_T(eos: "PengRobinson | SRK", x: Array, P: Array | float,
                 k_ij: Array | None = None,
                 T_guess: Array | float | None = None) -> dict:
    """Bubble-point temperature of liquid ``x`` at ``P`` with a cubic EOS.

    As :func:`eos_bubble_P` with ``T`` the unknown (the start is the Wilson
    bubble temperature). Returns the same dict.
    """
    return _eos_saturation(eos, x, None, P, k_ij, "bubble", "T", T_guess)


def eos_dew_T(eos: "PengRobinson | SRK", y: Array, P: Array | float,
              k_ij: Array | None = None,
              T_guess: Array | float | None = None) -> dict:
    """Dew-point temperature of vapor ``y`` at ``P`` with a cubic EOS.

    As :func:`eos_dew_P` with ``T`` the unknown. Returns the same dict.
    """
    return _eos_saturation(eos, y, None, P, k_ij, "dew", "T", T_guess)
