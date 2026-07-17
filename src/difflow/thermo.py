"""Thermodynamic property calculations for difflow.

This module provides ideal thermodynamic models:
- Ideal gas behavior
- Ideal liquid mixtures
- Antoine equation for vapor pressures
- Polynomial Cp correlations

All key functions are JIT-compiled for performance.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from jax import Array

from difflow.numerics import safe_log

# Universal gas constant (J/mol/K) and the reference pressure for the ideal-gas
# entropy's -R ln(P/P_ref) term (1 atm). Kept local to this module so the
# entropy uses the same convention as difflow.eos (R = 8.314462618).
R_GAS = 8.314462618
P_REF_ENTROPY = 101325.0


# =============================================================================
# JIT-compiled helper functions for thermodynamic calculations
# =============================================================================


@jax.jit
def _compute_cp_poly(T: Array, a: float, b: float, c: float, d: float) -> Array:
    """JIT-compiled polynomial Cp calculation."""
    return a + b * T + c * T**2 + d * T**3


@jax.jit
def _compute_enthalpy_integral(
    T: Array, Tref: Array, a: float, b: float, c: float, d: float
) -> Array:
    """JIT-compiled enthalpy integral from Tref to T."""
    return (
        a * (T - Tref)
        + b / 2 * (T**2 - Tref**2)
        + c / 3 * (T**3 - Tref**3)
        + d / 4 * (T**4 - Tref**4)
    )


@jax.jit
def _compute_hvap_watson(T: Array, A: float, n: float, Tc: float) -> Array:
    """JIT-compiled Watson correlation for heat of vaporization."""
    Tc_arr = jnp.asarray(Tc)
    Tr = jnp.clip(T / Tc_arr, 0.0, 0.999)
    return A * (1 - Tr) ** n


@jax.jit
def _compute_psat_antoine(T: Array, A: float, B: float, C: float) -> Array:
    """JIT-compiled Antoine equation for saturation pressure."""
    log10_P = A - B / (T + C)
    return jnp.power(10.0, log10_P)


class SpeciesData(NamedTuple):
    """Thermodynamic data for a pure species.

    Attributes:
        name: Species identifier
        MW: Molecular weight (g/mol)
        Cp_coeffs: Liquid heat capacity coefficients [a, b, c, d] for
                   Cp = a + b*T + c*T^2 + d*T^3 (J/mol/K)
        Hvap_coeffs: Heat of vaporization coefficients [A, n, Tc] for
                     Hvap = A * (1 - T/Tc)^n (J/mol)
        antoine_coeffs: Antoine equation coefficients [A, B, C] for
                        log10(Psat/Pa) = A - B/(T + C) where T in K
        Hf: Standard heat of formation (J/mol) at 298.15 K
        Tref: Reference temperature for Hf (K), default 298.15
        Cp_vapor_coeffs: Vapor heat capacity coefficients [a, b, c, d] for
                         Cp = a + b*T + c*T^2 + d*T^3 (J/mol/K).
                         If None, falls back to Cp_coeffs (liquid).
        T_antoine_min: Minimum valid temperature for Antoine equation (K)
        T_antoine_max: Maximum valid temperature for Antoine equation (K)
    """
    name: str
    MW: float
    Cp_coeffs: tuple[float, float, float, float]
    Hvap_coeffs: tuple[float, float, float]  # A, n, Tc
    antoine_coeffs: tuple[float, float, float]  # A, B, C
    Hf: float = 0.0
    Tref: float = 298.15
    Cp_vapor_coeffs: tuple[float, float, float, float] | None = None
    T_antoine_min: float = 0.0
    T_antoine_max: float = 1e6


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

    def Cp(self, species: str, T: Array | float, phase: str = "liquid") -> Array:
        """Calculate heat capacity of a pure species.

        Args:
            species: Species name
            T: Temperature (K)
            phase: 'liquid' or 'vapor'. If vapor coefficients are not available,
                   falls back to liquid coefficients.

        Returns:
            Heat capacity Cp (J/mol/K)
        """
        data = self.species[species]
        if phase == "vapor" and data.Cp_vapor_coeffs is not None:
            a, b, c, d = data.Cp_vapor_coeffs
        else:
            a, b, c, d = data.Cp_coeffs
        return _compute_cp_poly(jnp.asarray(T), a, b, c, d)

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

        For liquid: H = integral(Cp_liq, Tref, T)
        For vapor:  H = integral(Cp_vap, Tref, T) + Hvap(T)

        When Cp_vapor_coeffs is not available, falls back to Cp_coeffs.

        Args:
            species: Species name
            T: Temperature (K)
            phase: 'liquid' or 'vapor'

        Returns:
            Specific enthalpy (J/mol) relative to liquid at Tref
        """
        data = self.species[species]
        T_arr = jnp.asarray(T)
        Tref = jnp.asarray(data.Tref)

        # Always use liquid Cp for the base enthalpy integral (reference state
        # is liquid at Tref). For vapor phase, we add Hvap(T) to account for
        # the phase change, giving the correct thermodynamic path:
        # H_vap(T) = integral(Cp_liq, Tref→T) + Hvap(T)
        a, b, c, d = data.Cp_coeffs

        # Integral of Cp_liquid from Tref to T
        H = _compute_enthalpy_integral(T_arr, Tref, a, b, c, d)

        if phase == "vapor":
            # Add heat of vaporization at T
            H = H + self.Hvap(species, T_arr)

        return H

    def S_ig_T(self, species: str, T: Array | float) -> Array:
        """Ideal-gas entropy temperature integral integral(Cp/T dT, Tref -> T).

        This is the T-dependent part of the ideal-gas molar entropy [J/mol/K]::

            integral(Cp/T dT) = a ln(T/Tref) + b (T - Tref)
                                + c/2 (T^2 - Tref^2) + d/3 (T^3 - Tref^3)

        for Cp = a + bT + cT^2 + dT^3 (the same ``Cp_coeffs`` used for the
        ideal-gas sensible enthalpy). The per-species reference entropy s_i^0 is
        not included; it cancels in any constant-composition process change (an
        expander, valve, cooler or compressor), which is what the entropy is used
        for here. The pressure (-R ln(P/Pref)) and mixing (-R sum y ln y) terms
        are added at the mixture level in :class:`CubicThermo`.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Ideal-gas entropy T-integral relative to Tref (J/mol/K).
        """
        data = self.species[species]
        T_arr = jnp.asarray(T)
        Tref = jnp.asarray(data.Tref)
        a, b, c, d = data.Cp_coeffs
        return (
            a * safe_log(T_arr / Tref)
            + b * (T_arr - Tref)
            + c / 2 * (T_arr**2 - Tref**2)
            + d / 3 * (T_arr**3 - Tref**3)
        )

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
        return _compute_hvap_watson(jnp.asarray(T), A, n, Tc)

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
        return _compute_psat_antoine(jnp.asarray(T), A, B, C)

    def validate_antoine(self, species: str, T: float) -> dict:
        """Check whether temperature is in valid range for Antoine equation.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Dict with 'in_range', 'T_min', and 'T_max' keys
        """
        data = self.species[species]
        return {
            'in_range': data.T_antoine_min <= T <= data.T_antoine_max,
            'T_min': data.T_antoine_min,
            'T_max': data.T_antoine_max,
        }

    def Psat_with_info(
        self, species: str, T: Array | float
    ) -> tuple[Array, dict]:
        """Calculate saturation pressure with Antoine range validation info.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            (Psat, info_dict) where info_dict contains 'antoine_in_range' flag
        """
        Psat = self.Psat(species, T)
        data = self.species[species]
        T_val = float(T)
        info = {
            'antoine_in_range': data.T_antoine_min <= T_val <= data.T_antoine_max,
            'T_antoine_min': data.T_antoine_min,
            'T_antoine_max': data.T_antoine_max,
        }
        return Psat, info

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
        P: Array | float | None = None,
    ) -> Array:
        """Calculate total enthalpy of a stream.

        Args:
            flows: Molar flows by species (mol/s)
            T: Temperature (K)
            phase: 'liquid' or 'vapor'
            P: Pressure (Pa). Ignored -- ideal-gas enthalpy is pressure
               independent. Accepted only so IdealThermo is call-compatible
               with CubicThermo (which does use P for its departure term),
               letting callers pass P unconditionally.

        Returns:
            Total enthalpy flow (J/s = W)
        """
        H_total = jnp.zeros(())
        for species, F in flows.items():
            H_total = H_total + F * self.H_pure(species, T, phase)
        return H_total


class CubicThermo:
    """Peng-Robinson-consistent enthalpy: ideal-gas sensible + EOS departure.

    Wraps an :class:`IdealThermo` (for the ideal-gas sensible enthalpy, using
    the constant ideal-gas Cp already in the species data) and a cubic EOS
    (for the enthalpy departure ``H - H_ideal_gas``). This mirrors the way a
    cubic-EOS property package (e.g. IDAES's Generic framework) builds every
    unit's enthalpy as ideal-gas + departure, so difflow's reactor and
    heat-exchanger energy balances become consistent with such a tool rather
    than relying on difflow's Watson-Hvap liquid/vapor enthalpy split.

    The ideal-gas sensible part is taken from ``IdealThermo`` via its
    ``phase="liquid"`` path, which is the bare ``integral(Cp, Tref -> T)`` with
    no heat-of-vaporization term. That is exactly the ideal-gas sensible
    enthalpy when ``Cp_coeffs`` holds the ideal-gas Cp (as difflow's database
    does for these components); the EOS departure then supplies the entire
    real-gas / phase-change correction, so both phases share one reference.

    ``stream_enthalpy`` keeps ``IdealThermo``'s calling convention but adds an
    optional ``P``: the departure is pressure-dependent, so a caller must pass
    the stream pressure for it to be included. With ``P=None`` the result
    degrades to the pure ideal-gas sensible enthalpy (no departure), which lets
    it stand in for an ``IdealThermo`` wherever pressure is not threaded
    through.
    """

    def __init__(self, ideal: "IdealThermo", eos):
        self.ideal = ideal
        self.eos = eos

    @property
    def species_order(self) -> list[str]:
        return self.ideal.species_order

    def Cp_mix(
        self,
        mole_fracs: dict[str, Array | float],
        T: Array | float,
    ) -> Array:
        """Ideal-gas mixture heat capacity (J/mol/K).

        Delegates to the wrapped IdealThermo. Used only as the Jacobian
        estimate for solvers' temperature updates (e.g. the CSTR's adiabatic
        fixed point), where the converged answer is set by the enthalpy
        balance, not by this Cp; the EOS departure's contribution to the true
        Cp therefore does not need to appear here.
        """
        return self.ideal.Cp_mix(mole_fracs, T)

    def stream_enthalpy(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        phase: str = "vapor",
        P: Array | float | None = None,
    ) -> Array:
        """Total stream enthalpy (W) = ideal-gas sensible + PR departure.

        Args:
            flows: Molar flows by species (mol/s).
            T: Temperature (K).
            P: Pressure (Pa). If None, the departure term is omitted and only
               the ideal-gas sensible enthalpy is returned.
            phase: 'vapor' or 'liquid'; selects the EOS Z root for the
               departure.

        Returns:
            Total enthalpy flow (J/s = W).
        """
        # Ideal-gas sensible part (constant-Cp integral, no Hvap): IdealThermo's
        # "liquid" path is exactly integral(Cp, Tref -> T) for this data.
        H_ideal = self.ideal.stream_enthalpy(flows, T, phase="liquid")
        if P is None:
            return H_ideal

        order = self.eos.species_order
        F = jnp.array([flows[s] for s in order])
        F_total = jnp.sum(F)
        y = F / jnp.maximum(F_total, 1e-30)
        h_dep = self.eos.enthalpy_departure(T, P, y, phase)
        return H_ideal + F_total * h_dep

    def stream_enthalpy_flash(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        P: Array | float,
    ) -> Array:
        """Two-phase-aware total stream enthalpy (W): flash at (T, P), then sum
        the phase enthalpies.

        Unlike :meth:`stream_enthalpy` (which assumes a single named phase),
        this determines the vapor/liquid split from the EOS and weights each
        phase's departure by its flow, so the result captures latent heat as the
        stream partially vaporizes or condenses with temperature. That makes a
        heat-exchanger energy balance built on this enthalpy consistent with a
        rigorous VLE tool through phase change (e.g. the cold naphtha+H2 feed
        boiling as it preheats).

        The ideal-gas enthalpy is composition-invariant across the split
        (sum_i z_i h_i = V sum_i y_i h_i + (1-V) sum_i x_i h_i), so only the
        departure is phase-weighted::

            H = H_ideal(flows, T)
                + F_total * [V * h_dep_vap(T, P, y) + (1-V) * h_dep_liq(T, P, x)]

        Args:
            flows: Molar flows by species (mol/s).
            T: Temperature (K).
            P: Pressure (Pa).

        Returns:
            Total enthalpy flow (J/s = W).
        """
        from difflow.eos import flash_TP_eos

        H_ideal = self.ideal.stream_enthalpy(flows, T, phase="liquid")
        order = self.eos.species_order
        F = jnp.array([flows[s] for s in order])
        F_total = jnp.sum(F)
        z = F / jnp.maximum(F_total, 1e-30)

        V, x, y = flash_TP_eos(self.eos, z, T, P)
        V = jnp.clip(V, 0.0, 1.0)
        h_dep_vap = self.eos.enthalpy_departure(T, P, y, "vapor")
        h_dep_liq = self.eos.enthalpy_departure(T, P, x, "liquid")
        return H_ideal + F_total * (V * h_dep_vap + (1.0 - V) * h_dep_liq)

    def _ideal_gas_entropy(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        P: Array | float,
    ) -> tuple[Array, Array, Array]:
        """Molar ideal-gas entropy of the mixture, relative to (Tref, P_ref).

        Returns ``(F_total, z, s_ig)`` where::

            s_ig = sum_i z_i integral(Cp_i/T dT)   (temperature integral)
                   - R ln(P / P_ref)               (pressure term)
                   - R sum_i z_i ln z_i            (entropy of mixing)

        The per-species reference entropy s_i^0 is omitted (it cancels in the
        constant-composition changes the isentropic units use). Shared by
        :meth:`stream_entropy` and :meth:`stream_entropy_flash`.
        """
        order = self.eos.species_order
        F = jnp.array([flows[s] for s in order])
        F_total = jnp.sum(F)
        z = F / jnp.maximum(F_total, 1e-30)

        s_T = jnp.sum(jnp.array([z[i] * self.ideal.S_ig_T(s, T)
                                 for i, s in enumerate(order)]))
        s_pressure = -R_GAS * safe_log(jnp.asarray(P) / P_REF_ENTROPY)
        s_mix = -R_GAS * jnp.sum(z * safe_log(jnp.maximum(z, 1e-30)))
        return F_total, z, s_T + s_pressure + s_mix

    def stream_entropy(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        phase: str = "vapor",
        P: Array | float | None = None,
    ) -> Array:
        """Total stream entropy (W/K) = ideal-gas entropy + PR entropy departure.

        The entropy analogue of :meth:`stream_enthalpy`: an ideal-gas part (the
        Cp/T temperature integral plus the -R ln(P/P_ref) pressure and
        -R sum y ln y mixing terms) plus the EOS entropy departure for a single
        named phase. With ``P=None`` no pressure or departure term can be formed,
        so only the temperature integral and mixing entropy are returned.

        Args:
            flows: Molar flows by species (mol/s).
            T: Temperature (K).
            phase: 'vapor' or 'liquid'; selects the EOS Z root for the departure.
            P: Pressure (Pa). If None, the pressure and departure terms are
               omitted.

        Returns:
            Total entropy flow (W/K = J/s/K).
        """
        if P is None:
            order = self.eos.species_order
            F = jnp.array([flows[s] for s in order])
            F_total = jnp.sum(F)
            z = F / jnp.maximum(F_total, 1e-30)
            s_T = jnp.sum(jnp.array([z[i] * self.ideal.S_ig_T(s, T)
                                     for i, s in enumerate(order)]))
            s_mix = -R_GAS * jnp.sum(z * safe_log(jnp.maximum(z, 1e-30)))
            return F_total * (s_T + s_mix)

        F_total, z, s_ig = self._ideal_gas_entropy(flows, T, P)
        s_dep = self.eos.entropy_departure(T, P, z, phase)
        return F_total * (s_ig + s_dep)

    def stream_entropy_flash(
        self,
        flows: dict[str, Array | float],
        T: Array | float,
        P: Array | float,
    ) -> Array:
        """Two-phase-aware total stream entropy (W/K): flash at (T, P), then sum
        the phase entropies.

        The entropy counterpart of :meth:`stream_enthalpy_flash`. The ideal-gas
        entropy is composition-invariant across the split (the mixing term uses
        the feed composition z), so only the departure is phase-weighted::

            S = S_ideal(flows, T, P)
                + F_total * [V * s_dep_vap(T, P, y) + (1-V) * s_dep_liq(T, P, x)]

        This makes an isentropic turboexpander or compressor built on this
        entropy consistent with a rigorous VLE tool through phase change.

        Args:
            flows: Molar flows by species (mol/s).
            T: Temperature (K).
            P: Pressure (Pa).

        Returns:
            Total entropy flow (W/K = J/s/K).
        """
        from difflow.eos import flash_TP_eos

        F_total, z, s_ig = self._ideal_gas_entropy(flows, T, P)
        V, x, y = flash_TP_eos(self.eos, z, T, P)
        V = jnp.clip(V, 0.0, 1.0)
        s_dep_vap = self.eos.entropy_departure(T, P, y, "vapor")
        s_dep_liq = self.eos.entropy_departure(T, P, x, "liquid")
        return F_total * (s_ig + V * s_dep_vap + (1.0 - V) * s_dep_liq)
