"""Thermodynamic property calculations for difflow.

This module provides ideal thermodynamic models:
- Ideal gas behavior
- Ideal liquid mixtures
- Antoine equation for vapor pressures
- Polynomial Cp correlations

All key functions are JIT-compiled for performance.
"""

from typing import NamedTuple
import warnings
import jax
import jax.numpy as jnp
import numpy as np

from difflow.cache_key import ValueKeyed
from jax import Array

from difflow.numerics import safe_log

# Universal gas constant (J/mol/K) and the reference pressure for the ideal-gas
# entropy's -R ln(P/P_ref) term (1 atm). Kept local to this module so the
# entropy uses the same convention as difflow.eos (R = 8.314462618).
R_GAS = 8.314462618
P_REF_ENTROPY = 101325.0


class AntoineRangeWarning(UserWarning):
    """A vapor-pressure correlation was evaluated outside its fitted range.

    Outside ``[T_antoine_min, T_antoine_max]`` the equation is extrapolated,
    and Antoine constants extrapolate badly (tens of percent within a few tens
    of kelvin is common). For a species whose :class:`SpeciesData` carries a
    Wagner set (``wagner_coeffs``) the range checked is the Wagner one,
    ``[T_wagner_min, Tc]``. Raised only for concrete temperatures; under
    ``jit``/``grad`` the check is skipped, as for difflow's other diagnostics.
    """


class BubbleDewConvergenceWarning(UserWarning):
    """A bubble/dew-point solve did not converge or found the trivial root.

    Raised only for concrete results, like difflow's other diagnostics; the
    value is returned unchanged (never silently clipped).
    """


def _concrete_range(T) -> tuple[float, float] | None:
    """``(min, max)`` of a concrete temperature (array), or None if traced."""
    try:
        Tv = np.asarray(T, dtype=float)
    except Exception:  # a tracer has no numeric value
        return None
    if Tv.size == 0 or not np.any(np.isfinite(Tv)):
        return None
    return float(np.nanmin(Tv)), float(np.nanmax(Tv))


def check_antoine_range(species: str, data, T, stacklevel: int = 3) -> bool:
    """Warn when concrete ``T`` falls outside the species' Antoine range.

    The message names the species and its range but not ``T``, so Python's
    warning registry shows it once per species and side (below/above) per
    call site rather than once per evaluation.

    Args:
        species: Species name (for the message).
        data: The species' :class:`SpeciesData`.
        T: Temperature (K): a scalar or an array (its min and max are tested).
            Traced values are not checked.
        stacklevel: Passed to :func:`warnings.warn`.

    Returns:
        True when a warning was issued.

    Example:
        >>> import warnings
        >>> from difflow.database import get_species_data
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     check_antoine_range("water", get_species_data("water"), 550.0)
        True
    """
    wagner = getattr(data, "wagner_coeffs", None)
    if wagner is not None:
        what = "Wagner"
        lo_ok, hi_ok = float(getattr(data, "T_wagner_min", 0.0)), float(wagner[4])
    else:
        what = "Antoine"
        lo_ok, hi_ok = data.T_antoine_min, data.T_antoine_max
    if lo_ok <= 0.0 and hi_ok >= 1e6:  # no range recorded
        return False
    rng = _concrete_range(T)
    if rng is None:
        return False
    lo, hi = rng
    warned = False
    for side, bad in (("below", lo < lo_ok), ("above", hi > hi_ok)):
        if bad:
            warnings.warn(
                f"{what} vapor pressure of {species!r} extrapolated {side} its "
                f"fitted range {lo_ok:g}-{hi_ok:g} K",
                AntoineRangeWarning, stacklevel=stacklevel)
            warned = True
    return warned


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


#: Exponents of the third and fourth terms of the two Wagner forms in use:
#: ``"2.5-5"`` (Poling, Prausnitz & O'Connell 5e, ch. 7 and App. A) and
#: ``"3-6"`` (Wagner's original form, as fitted by McGarry 1983).
WAGNER_FORMS: dict[str, tuple[float, float]] = {"2.5-5": (2.5, 5.0),
                                                 "3-6": (3.0, 6.0)}


def psat_wagner(T: Array | float, coeffs, form: str = "2.5-5") -> Array:
    """Wagner vapor pressure (Pa).

    ``ln(P/Pc) = (a tau + b tau^1.5 + c tau^e3 + d tau^e4) / Tr`` with
    ``tau = 1 - Tr``, ``Tr = T/Tc`` and ``(e3, e4)`` = (2.5, 5) for the
    ``"2.5-5"`` form or (3, 6) for the ``"3-6"`` form.

    Above ``Tc`` (``tau < 0``) the fractional powers have no real value; the
    equation is continued by its leading term, ``ln(P/Pc) = a (Tc/T - 1)``,
    which matches the value and slope of the full equation at ``Tc`` (the
    other terms and their first derivatives vanish there). A supercritical
    "Psat" has no physical meaning, but a Raoult K-value of a light gas
    still needs a smooth, monotone one.

    Args:
        T: Temperature (K).
        coeffs: ``(a, b, c, d, Tc, Pc)``, Tc in K and Pc in Pa.
        form: ``"2.5-5"`` or ``"3-6"`` (see :data:`WAGNER_FORMS`).

    Returns:
        Saturation pressure (Pa).

    Example:
        >>> # methane, Poling et al. 5e App. A: 1 atm at 111.67 K
        >>> P = psat_wagner(111.67, (-6.02242, 1.26652, -0.5707, -1.366,
        ...                          190.551, 4.5992e6))
        >>> abs(float(P) / 101325.0 - 1.0) < 0.002
        True
    """
    a, b, c, d, Tc, Pc = coeffs
    e3, e4 = WAGNER_FORMS[form]
    T = jnp.asarray(T)
    Tr = T / Tc
    tau = 1.0 - Tr
    tp = jnp.maximum(tau, 0.0)  # fractional powers only of tau >= 0
    ln_pr = (a * tau + b * tp ** 1.5 + c * tp ** e3 + d * tp ** e4) / Tr
    return Pc * jnp.exp(ln_pr)


class SpeciesData(NamedTuple):
    """Thermodynamic data for a pure species.

    Attributes:
        name: Species identifier
        MW: Molecular weight (g/mol)
        Cp_coeffs: Liquid heat capacity coefficients [a, b, c, d] for
                   Cp = a + b*T + c*T^2 + d*T^3 (J/mol/K). :class:`IdealThermo`
                   integrates this for liquid (and, via ``Hvap``, vapor)
                   enthalpy. For a species with no liquid at ambient
                   conditions (methane, N2, ...) the database stores the
                   ideal-gas cubic here, as there is nothing else to store.
        Hvap_coeffs: Heat of vaporization coefficients [A, n, Tc] for
                     Hvap = A * (1 - T/Tc)^n (J/mol)
        antoine_coeffs: Antoine equation coefficients [A, B, C] for
                        log10(Psat/Pa) = A - B/(T + C) where T in K
        Hf: Standard heat of formation (J/mol) at 298.15 K
        Tref: Reference temperature for Hf (K), default 298.15
        Cp_vapor_coeffs: **Ideal-gas** heat capacity coefficients [a, b, c, d]
                         for Cp = a + b*T + c*T^2 + d*T^3 (J/mol/K). This is
                         the field :class:`CubicThermo` (and every ideal-gas
                         enthalpy/entropy) reads; see
                         :meth:`IdealThermo.Cp_ig`. If None, falls back to
                         Cp_coeffs, which is right only when that already
                         holds the ideal-gas Cp (the Cantera, DWSIM and
                         PyGLenN importers store it there).
        T_antoine_min: Minimum valid temperature for Antoine equation (K)
        T_antoine_max: Maximum valid temperature for Antoine equation (K)
        wagner_coeffs: Optional Wagner vapor-pressure constants
                       ``(a, b, c, d, Tc, Pc)`` (Tc in K, Pc in Pa; the
                       Wagner set's own critical point, which may differ
                       slightly from the EOS one). When set,
                       :meth:`IdealThermo.Psat` uses it instead of the
                       Antoine equation; see :func:`psat_wagner`. The
                       Antoine constants are still used for the enthalpy
                       path's vaporization temperature and as a starting
                       guess.
        wagner_form: ``"2.5-5"`` (Poling et al. 5e, the default) or
                     ``"3-6"`` (McGarry 1983): the exponents of the third
                     and fourth Wagner terms.
        T_wagner_min: Lower end of the Wagner set's range (K); the upper
                      end is its Tc.
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
    wagner_coeffs: tuple[float, float, float, float, float, float] | None = None
    wagner_form: str = "2.5-5"
    T_wagner_min: float = 0.0


class IdealThermo(ValueKeyed):
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
        # Two IdealThermos over the same species data compile to the same
        # graph; keying on the data rather than on identity lets them share
        # the executable. See difflow.cache_key.
        self._set_value_key(species_data)

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

    def _ig_coeffs(self, species: str) -> tuple[float, float, float, float]:
        """Ideal-gas Cp cubic: ``Cp_vapor_coeffs``, else ``Cp_coeffs``."""
        data = self.species[species]
        if data.Cp_vapor_coeffs is not None:
            return tuple(data.Cp_vapor_coeffs)
        return tuple(data.Cp_coeffs)

    def Cp_ig(self, species: str, T: Array | float) -> Array:
        """Ideal-gas heat capacity of a pure species (J/mol/K).

        Reads ``Cp_vapor_coeffs`` (falling back to ``Cp_coeffs`` when that is
        unset). This is the Cp a cubic-EOS property package adds its departure
        function to, as distinct from the liquid :meth:`Cp`.
        """
        a, b, c, d = self._ig_coeffs(species)
        return _compute_cp_poly(jnp.asarray(T), a, b, c, d)

    def Cp_mix_ig(
        self,
        mole_fracs: dict[str, Array | float],
        T: Array | float,
    ) -> Array:
        """Ideal-gas mixture heat capacity (J/mol/K); see :meth:`Cp_ig`."""
        Cp_total = jnp.zeros(())
        for species, x in mole_fracs.items():
            Cp_total = Cp_total + x * self.Cp_ig(species, T)
        return Cp_total

    def H_ig(self, species: str, T: Array | float) -> Array:
        """Ideal-gas sensible enthalpy ``integral(Cp_ig, Tref -> T)`` (J/mol).

        No heat of vaporization: the ideal gas at ``Tref`` is the zero, so a
        cubic EOS departure supplies all of the real-fluid and phase-change
        correction.
        """
        data = self.species[species]
        a, b, c, d = self._ig_coeffs(species)
        return _compute_enthalpy_integral(
            jnp.asarray(T), jnp.asarray(data.Tref), a, b, c, d
        )

    def stream_enthalpy_ig(
        self, flows: dict[str, Array | float], T: Array | float
    ) -> Array:
        """Total ideal-gas sensible enthalpy flow (W) of a stream."""
        H_total = jnp.zeros(())
        for species, F in flows.items():
            H_total = H_total + F * self.H_ig(species, T)
        return H_total

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

        The zero is the liquid at ``Tref``. The vapor follows a fixed
        thermodynamic path -- heat the liquid to a vaporization temperature
        ``T_v``, vaporize there, then heat the ideal gas::

            liquid: H = integral(Cp_liq, Tref -> T)
            vapor:  H = integral(Cp_liq, Tref -> T_v) + Hvap(T_v)
                        + integral(Cp_ig, T_v -> T)

        ``T_v`` is the lower of ``Tref`` and the Antoine normal boiling point
        (see :meth:`vaporization_T`), where the Watson latent heat is well
        away from Tc.
        So ``dH_vap/dT = Cp_ig`` > 0 at every T: the vapor enthalpy is
        monotone, which an energy balance needs for a unique root. (Adding
        the Watson ``Hvap(T)`` at the stream's own T instead makes the vapor
        enthalpy *fall* towards Tc, where ``Hvap -> 0``.) The latent heat
        ``H_vap(T) - H_liq(T)`` then follows Kirchhoff's law from ``T_v``
        rather than the Watson correlation.

        ``Cp_ig`` is ``Cp_vapor_coeffs``; when that is unset it falls back to
        ``Cp_coeffs``, and the latent heat is the constant ``Hvap(T_v)``.

        **Which latent heat is authoritative.** This path is what every
        unit energy balance (flash, heater, column, ...) integrates, so the
        latent heat a duty actually contains at ``T`` is
        :meth:`latent_heat` ``= H_pure(vapor) - H_pure(liquid)``, not the
        Watson :meth:`Hvap` at ``T``. The two agree at ``T_v`` and drift
        apart away from it, because ``Cp_liq``, ``Cp_ig`` and the Watson
        constants are separate fits and Kirchhoff's relation
        ``d(dHvap)/dT = Cp_ig - Cp_liq`` is not imposed on the Watson curve
        (toluene at 374.64 K: 34 783 J/mol along the path vs a Watson 33 728,
        +3.1%). :meth:`Hvap` is the correlation the path is anchored to; read
        :meth:`latent_heat` when a number must be consistent with the duties.

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
        a, b, c, d = data.Cp_coeffs
        if phase != "vapor":
            return _compute_enthalpy_integral(T_arr, Tref, a, b, c, d)

        Tv = jnp.asarray(self.vaporization_T(species))
        ag, bg, cg, dg = self._ig_coeffs(species)
        return (_compute_enthalpy_integral(Tv, Tref, a, b, c, d)
                + self.Hvap(species, Tv)
                + _compute_enthalpy_integral(T_arr, Tv, ag, bg, cg, dg))

    def vaporization_T(self, species: str) -> float:
        """Temperature (K) at which :meth:`H_pure`'s vapor path vaporizes.

        The lower of ``Tref`` and the normal boiling point implied by the
        Antoine constants: a light species (ethane, Tc = 305 K) vaporizes at
        its Tb, where the Watson ``Hvap`` is a real latent heat, not at
        ``Tref`` = 298 K, where it has nearly vanished. A Python float:
        species data are constants, not traced.
        """
        data = self.species[species]
        A, B, C = (float(v) for v in data.antoine_coeffs)
        Tb = B / (A - np.log10(101325.0)) - C
        return min(float(data.Tref), Tb)

    def latent_heat(self, species: str, T: Array | float) -> Array:
        """Path-consistent latent heat ``H_vap(T) - H_liq(T)`` (J/mol).

        The difference of :meth:`H_pure`'s vapor and liquid enthalpies at
        the same ``T``: the latent heat that the energy balances of every
        unit using this thermo actually contain, and so the authoritative
        one for duties. It equals the Watson :meth:`Hvap` at
        :meth:`vaporization_T` and follows Kirchhoff's law from there::

            latent_heat(T) = Hvap(T_v) + integral(Cp_ig - Cp_liq, T_v -> T)

        whereas :meth:`Hvap` is the Watson correlation at ``T`` itself. They
        differ away from ``T_v`` because the Cp's and the Watson constants
        are fitted separately (toluene at 374.64 K: 34 783 vs 33 728 J/mol).

        Args:
            species: Species name.
            T: Temperature (K).

        Returns:
            Latent heat (J/mol), differentiable in ``T``.

        Example:
            >>> from difflow.database import get_species_data
            >>> th = IdealThermo({"toluene": get_species_data("toluene")})
            >>> L = th.latent_heat("toluene", 374.64)
            >>> float(L) == float(th.H_pure("toluene", 374.64, "vapor")
            ...                   - th.H_pure("toluene", 374.64, "liquid"))
            True
        """
        return (self.H_pure(species, T, "vapor")
                - self.H_pure(species, T, "liquid"))

    def saturation_T(self, species: str, P: Array | float,
                     T0: Array | float | None = None) -> Array:
        """Saturation temperature (K) at pressure ``P``: ``Psat(T) = P``.

        The inverse of :meth:`Psat`, so it uses the Wagner set when the
        species has one and the Antoine equation otherwise. For Antoine the
        inverse is explicit, ``T = B / (A - log10 P) - C``; for Wagner it is
        a Newton solve of ``ln Psat(T) = ln P`` from the Antoine value, with
        implicit differentiation. Either way ``dT/dP`` is exact and obeys
        Clausius-Clapeyron's sign (positive).

        Not to be confused with :meth:`vaporization_T`, the fixed temperature
        at which :meth:`H_pure`'s vapor path vaporizes.

        Args:
            species: Species name.
            P: Pressure (Pa).
            T0: Optional initial guess for the Wagner solve (K).

        Returns:
            Saturation temperature (K).

        Warns:
            AntoineRangeWarning: a concrete result outside the correlation's
                fitted range.

        Example:
            >>> from difflow.database import get_species_data
            >>> th = IdealThermo({"water": get_species_data("water")})
            >>> abs(float(th.saturation_T("water", 101325.0)) - 373.12) < 0.1
            True
        """
        import optimistix as optx

        data = self.species[species]
        P = jnp.asarray(P, dtype=float)
        A, B, C = data.antoine_coeffs
        T_ant = B / (A - jnp.log10(P)) - C
        if data.wagner_coeffs is None:
            T_out = T_ant
        else:
            coeffs, form = data.wagner_coeffs, data.wagner_form

            def resid(T, lnP):
                return jnp.log(psat_wagner(T, coeffs, form)) - lnP

            start = T_ant if T0 is None else jnp.asarray(T0, dtype=float)
            sol = optx.root_find(resid, optx.Newton(rtol=1e-12, atol=1e-12),
                                 jax.lax.stop_gradient(start),
                                 args=jnp.log(P), max_steps=50, throw=False)
            T_out = sol.value
            try:
                r = float(np.max(np.abs(np.asarray(resid(T_out, jnp.log(P))))))
            except Exception:  # traced: no diagnostic
                r = 0.0
            if not r < 1e-8:
                warnings.warn(f"saturation_T({species!r}) did not converge "
                              f"(residual {r:.3g})", BubbleDewConvergenceWarning,
                              stacklevel=2)
        check_antoine_range(species, data, T_out)
        return T_out

    def S_ig_T(self, species: str, T: Array | float) -> Array:
        """Ideal-gas entropy temperature integral integral(Cp/T dT, Tref -> T).

        This is the T-dependent part of the ideal-gas molar entropy [J/mol/K]::

            integral(Cp/T dT) = a ln(T/Tref) + b (T - Tref)
                                + c/2 (T^2 - Tref^2) + d/3 (T^3 - Tref^3)

        for Cp = a + bT + cT^2 + dT^3 with the ideal-gas Cp (:meth:`Cp_ig`). The per-species reference entropy s_i^0 is
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
        a, b, c, d = self._ig_coeffs(species)
        return (
            a * safe_log(T_arr / Tref)
            + b * (T_arr - Tref)
            + c / 2 * (T_arr**2 - Tref**2)
            + d / 3 * (T_arr**3 - Tref**3)
        )

    def Hvap(self, species: str, T: Array | float) -> Array:
        """Calculate heat of vaporization at temperature T.

        Uses Watson correlation: Hvap = A * (1 - T/Tc)^n

        This is the correlation :meth:`H_pure` is anchored to at
        :meth:`vaporization_T`, not the latent heat in the enthalpies at
        other temperatures; for a value consistent with unit duties use
        :meth:`latent_heat`.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Heat of vaporization (J/mol)
        """
        A, n, Tc = self.species[species].Hvap_coeffs
        return _compute_hvap_watson(jnp.asarray(T), A, n, Tc)

    def _psat(self, species: str, T: Array | float) -> Array:
        """Psat without the range check: Wagner if present, else Antoine."""
        data = self.species[species]
        if data.wagner_coeffs is not None:
            return psat_wagner(T, data.wagner_coeffs, data.wagner_form)
        A, B, C = data.antoine_coeffs
        return _compute_psat_antoine(jnp.asarray(T), A, B, C)

    def Psat(self, species: str, T: Array | float) -> Array:
        """Calculate saturation pressure (Pa).

        By default the Antoine equation, ``log10(Psat/Pa) = A - B/(T + C)``.
        When the species' :class:`SpeciesData` carries ``wagner_coeffs`` (the
        light gases, via ``get_species_data(name, vapor_pressure="wagner")``)
        the Wagner equation is used instead; see :func:`psat_wagner`.

        Args:
            species: Species name
            T: Temperature (K)

        Returns:
            Saturation pressure (Pa)

        Warns:
            AntoineRangeWarning: when a concrete ``T`` lies outside the
                species' fitted range (``T_antoine_min``-``T_antoine_max``,
                or ``T_wagner_min``-Tc for a Wagner set).
        """
        data = self.species[species]
        check_antoine_range(species, data, T)
        return self._psat(species, T)

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
            (Psat, info_dict) where info_dict contains 'antoine_in_range': a
            Python bool for a concrete scalar ``T``, otherwise a boolean array
            (so the call also works under ``jit``/``grad``). For a species
            with a Wagner set, 'antoine_in_range' tests the Wagner range
            (``T_wagner_min``-Tc), the correlation ``Psat`` actually used;
            'uses_wagner' says which correlation that is and 'T_min'/'T_max'
            give its range.
        """
        data = self.species[species]
        Psat = self._psat(species, T)
        if data.wagner_coeffs is not None:
            corr, lo, hi = "wagner", data.T_wagner_min, data.wagner_coeffs[4]
        else:
            corr, lo, hi = "antoine", data.T_antoine_min, data.T_antoine_max
        T_arr = jnp.asarray(T)
        in_range = (T_arr >= lo) & (T_arr <= hi)
        rng = _concrete_range(T) if jnp.ndim(T) == 0 else None
        if rng is not None:
            in_range = lo <= rng[0] <= hi
        info = {
            'antoine_in_range': in_range,
            'T_antoine_min': data.T_antoine_min,
            'T_antoine_max': data.T_antoine_max,
            'uses_wagner': corr == "wagner",
            'T_min': lo,
            'T_max': hi,
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

    #: Raoult K-values are a function of (T, P) alone. Callers that solve a
    #: composition fixed point (a column's MESH loop, a flash) read this to
    #: skip the composition iteration entirely; see
    #: :attr:`CubicThermo.K_depends_on_composition` for the other case.
    K_depends_on_composition = False

    def K_values(
        self,
        T: Array | float,
        P: Array | float,
        x: Array | None = None,
        y: Array | None = None,
    ) -> dict[str, Array]:
        """Calculate K-values for all species.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            x: Liquid mole fractions. Ignored -- Raoult K-values are
               composition independent. Accepted only so IdealThermo is
               call-compatible with CubicThermo (whose K-values come from
               fugacity coefficients and do depend on composition), letting
               callers pass the phase compositions unconditionally.
            y: Vapor mole fractions. Ignored, as for ``x``.

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
        x: Array | None = None,
        y: Array | None = None,
    ) -> Array:
        """Calculate K-values as an array in species order.

        Args:
            T: Temperature (K)
            P: Pressure (Pa)
            x: Liquid mole fractions. Ignored (see :meth:`K_values`).
            y: Vapor mole fractions. Ignored (see :meth:`K_values`).

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


class CubicThermo(ValueKeyed):
    """Peng-Robinson-consistent enthalpy: ideal-gas sensible + EOS departure.

    Wraps an :class:`IdealThermo` (for the ideal-gas sensible enthalpy, using
    the ideal-gas Cp in the species data, ``Cp_vapor_coeffs``) and a cubic EOS
    (for the enthalpy departure ``H - H_ideal_gas``). This mirrors the way a
    cubic-EOS property package (e.g. IDAES's Generic framework) builds every
    unit's enthalpy as ideal-gas + departure, so difflow's reactor and
    heat-exchanger energy balances become consistent with such a tool rather
    than relying on difflow's Watson-Hvap liquid/vapor enthalpy split.

    The ideal-gas sensible part is ``IdealThermo.stream_enthalpy_ig``, the bare
    ``integral(Cp_ig, Tref -> T)`` with no heat-of-vaporization term, where
    ``Cp_ig`` is ``SpeciesData.Cp_vapor_coeffs`` (``Cp_coeffs`` when that is
    unset). The liquid ``Cp_coeffs`` that :class:`IdealThermo` integrates is
    never read here; the EOS departure supplies the entire real-gas /
    phase-change correction, so both phases share one reference.

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
        self._set_value_key(ideal, eos)

    @property
    def species_order(self) -> list[str]:
        return self.ideal.species_order

    def _flow_vector(self, flows: dict[str, Array | float]) -> Array:
        """Molar flows in EOS species order; an absent species has zero flow.

        Matches :class:`IdealThermo`, which sums over the stream's own keys, so
        a condenser's vapour need not list the coolant. A species the EOS does
        not model is an error, not a silent zero.
        """
        order = self.eos.species_order
        unknown = sorted(set(flows) - set(order))
        if unknown:
            raise KeyError(f"species {unknown} not in the EOS species {list(order)}")
        return jnp.array([jnp.asarray(flows.get(s, 0.0), dtype=jnp.float64) for s in order])

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
        return self.ideal.Cp_mix_ig(mole_fracs, T)

    # ------------------------------------------------------------------
    # VLE K-values
    # ------------------------------------------------------------------

    #: EOS K-values are ratios of fugacity coefficients, so they depend on the
    #: liquid and vapor compositions as well as on (T, P). Callers solve that
    #: fixed point in their own outer loop; see :meth:`K_values_array`.
    K_depends_on_composition = True

    def K_values_array(
        self,
        T: Array | float,
        P: Array | float,
        x: Array | None = None,
        y: Array | None = None,
        n_inner: int = 3,
    ) -> Array:
        """EOS K-values as an array in species order.

        The rigorous VLE ratio built from the EOS fugacity coefficients::

            K_i = phi_i^L(T, P, x) / phi_i^V(T, P, y)

        This is the same K-value :class:`~difflow.eos.PengRobinson` uses in its
        own flash, exposed under ``IdealThermo``'s interface so that units
        written against ``K_values_array`` (the rigorous ``DistillationColumn``,
        for one) run on a cubic EOS instead of Raoult's law. For light
        hydrocarbons at pressure that is not a small correction: the difference
        shows up as tens of degrees in a condenser temperature.

        Unlike Raoult K-values these depend on the phase compositions, so they
        are a fixed point rather than a closed form. Pass whichever
        compositions you have; whatever is missing is filled in by a bounded
        successive substitution from a composition-free estimate, which is the
        standard bubble- or dew-point K calculation.

        Note the trivial solution. The EOS gives a two-phase K only over the
        temperature window where its cubic has two distinct roots for this
        (T, P, composition). Outside that window -- a subcooled liquid, a
        superheated vapor -- there is one root, both phases take it, and K
        comes back identically 1. That is the EOS reporting a single phase, but
        it makes ``sum(K x) - 1`` a flat zero, so a bubble-point solve built on
        this method needs an initial temperature inside the window and a step
        that can retreat when it leaves (see
        ``difflow.units.distillation._bubble_T``).

        Args:
            T: Temperature (K).
            P: Pressure (Pa).
            x: Liquid mole fractions in species order. If None it is taken
               from ``y`` and the current K-values (``x = y / K``).
            y: Vapor mole fractions in species order. If None it is taken from
               ``x`` and the current K-values (``y = K x``) -- the bubble-point
               K at composition ``x``, which is what a column stage wants.
            n_inner: Successive-substitution steps used to fill in a missing
               composition. Ignored when both ``x`` and ``y`` are given.

        Returns:
            Array of K-values in the EOS's species order (``eos.species_order``),
            which is also the order ``x`` and ``y`` are read in.

        Example:
            >>> import jax.numpy as jnp
            >>> from difflow import IdealThermo, PengRobinson
            >>> from difflow.database import get_critical_props, get_species_data
            >>> sp = ("propane", "n_butane")
            >>> ideal = IdealThermo({s: get_species_data(s) for s in sp})
            >>> thermo = CubicThermo(ideal, PengRobinson(
            ...     {s: get_critical_props(s) for s in sp}))
            >>> x = jnp.array([0.5, 0.5])
            >>> K = thermo.K_values_array(300.0, 6e5, x=x)  # bubble-point K at x
            >>> bool(K[0] > 1.0 > K[1])
            True
        """
        T = jnp.asarray(T)
        P = jnp.asarray(P)

        def _norm(v: Array) -> Array:
            v = jnp.maximum(jnp.asarray(v), 0.0)
            return v / jnp.maximum(jnp.sum(v), 1e-30)

        if x is None and y is None:
            # No composition to build a fugacity coefficient from. Fall back on
            # the wrapped ideal model's Raoult K-values rather than the EOS's
            # own Wilson correlation: both are composition-free estimates, but
            # Antoine coefficients are fitted vapor-pressure data while Wilson
            # is a two-constant fit off the critical point, and for the heavy
            # ends of a hydrocarbon cut Wilson is far enough out to land a
            # subsequent EOS iteration outside the temperature window where the
            # cubic has two roots at all (outside it the liquid and vapor roots
            # coincide and K collapses identically to 1).
            return self.ideal.K_values_array(T, P)

        # The successive substitutions below start from Wilson, as the EOS's
        # own flash does. Its job here is only to be in the basin of a
        # contraction that converges in two or three steps, which is a much
        # weaker requirement than the one that ruled it out above.
        if x is None:
            y = _norm(y)
            K = self.eos.K_values_wilson(T, P)
            for _ in range(n_inner):
                K = self.eos.K_values(T, P, _norm(y / jnp.maximum(K, 1e-30)), y)
            return K

        x = _norm(x)

        if y is None:
            K = self.eos.K_values_wilson(T, P)
            for _ in range(n_inner):
                K = self.eos.K_values(T, P, x, _norm(K * x))
            return K

        return self.eos.K_values(T, P, x, _norm(y))

    def K_values(
        self,
        T: Array | float,
        P: Array | float,
        x: Array | None = None,
        y: Array | None = None,
        n_inner: int = 3,
    ) -> dict[str, Array]:
        """EOS K-values for all species, keyed by species name.

        The dictionary form of :meth:`K_values_array`; see it for the meaning
        of ``x``, ``y`` and ``n_inner``.

        Args:
            T: Temperature (K).
            P: Pressure (Pa).
            x: Liquid mole fractions in species order (optional).
            y: Vapor mole fractions in species order (optional).
            n_inner: Successive-substitution steps for a missing composition.

        Returns:
            Dictionary of K-values by species name.
        """
        K = self.K_values_array(T, P, x, y, n_inner)
        return {s: K[i] for i, s in enumerate(self.eos.species_order)}

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
        # Ideal-gas sensible part, integral(Cp_ig, Tref -> T) with no Hvap.
        H_ideal = self.ideal.stream_enthalpy_ig(flows, T)
        if P is None:
            return H_ideal

        order = self.eos.species_order
        F = self._flow_vector(flows)
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

        H_ideal = self.ideal.stream_enthalpy_ig(flows, T)
        order = self.eos.species_order
        F = self._flow_vector(flows)
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
        F = self._flow_vector(flows)
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
            F = self._flow_vector(flows)
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
