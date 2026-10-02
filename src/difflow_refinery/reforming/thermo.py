"""Thermochemistry of the reformer species: enthalpy, Gibbs energy, equilibrium.

One enthalpy basis is used everywhere in the reformer, so that every energy
balance -- reactor, heater, cooler, compressor -- closes on the same numbers:

    H(F, T, P) = sum_i F_i [Hf_i + int_298.15^T cp_i dT] + F h_dep(T, P, y)

the ideal-gas enthalpy of formation path (Hf at 298.15 K plus the sensible
heat of the ideal gas) and the Peng-Robinson enthalpy departure of the
mixture. It is :class:`difflow.thermo.CubicThermo`'s enthalpy (ideal-gas
sensible + PR departure, built here by :func:`cubic_thermo` from the same
cubic Cp fits) with the heats of formation added, so the reactor and
difflow's own ``EOSFlash``/``Compressor`` share a reference state. Heats of
reaction therefore come out of the heats of formation and the Cp
polynomials: they are not separate parameters.

Equilibrium constants come from the standard Gibbs energy of reaction at the
reaction temperature, with the ideal-gas standard state at 1 bar::

    dG(T) = dH(T) - T dS(T)
    dH(T) = dH(298.15) + int dcp dT,   dS(T) = dS(298.15) + int dcp / T dT
    ln K  = -dG(T) / (R T)                       (K on a 1-bar basis)

which is the textbook route (e.g. Smith, Van Ness & Abbott, *Introduction to
Chemical Engineering Thermodynamics*, ch. 13 -- equation numbering differs by
edition, unverified). Partial pressures in the rate laws are therefore in bar.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery.reforming import species as sp

jax.config.update("jax_enable_x64", True)

#: Gas constant (J/mol/K).
R = 8.314462618
#: Reference temperature of Hf and S0 (K).
T_REF = 298.15
#: Standard-state pressure of S0 and of every equilibrium constant (Pa).
P_STD = 1.0e5

_CP = jnp.asarray(sp.CP)
_HF = jnp.asarray(sp.HF)
_S0 = jnp.asarray(sp.S0)


def cp(T: Array) -> Array:
    """Ideal-gas Cp of every species at ``T`` (J/mol/K), ``(N,)``."""
    T = jnp.asarray(T, dtype=float)
    a, b, c, d = _CP.T
    return a + b * T + c * T**2 + d * T**3


def sensible_enthalpy(T: Array) -> Array:
    """``int_298.15^T cp dT`` of every species (J/mol), ``(N,)``."""
    T = jnp.asarray(T, dtype=float)
    a, b, c, d = _CP.T

    def prim(t):
        return a * t + b * t**2 / 2 + c * t**3 / 3 + d * t**4 / 4

    return prim(T) - prim(T_REF)


def sensible_entropy(T: Array) -> Array:
    """``int_298.15^T cp / T dT`` of every species (J/mol/K), ``(N,)``."""
    T = jnp.asarray(T, dtype=float)
    a, b, c, d = _CP.T

    def prim(t):
        return a * jnp.log(t) + b * t + c * t**2 / 2 + d * t**3 / 3

    return prim(T) - prim(T_REF)


def enthalpy_ig(T: Array) -> Array:
    """Ideal-gas molar enthalpy ``Hf + int cp dT`` of every species (J/mol)."""
    return _HF + sensible_enthalpy(T)


def gibbs_ig(T: Array) -> Array:
    """Ideal-gas molar Gibbs energy ``H - T S`` at 1 bar of every species (J/mol).

    The absolute entropy ``S0`` is used, so this is the Gibbs energy on the
    "elements' enthalpy zero at 298.15 K, third-law entropy" basis. Element
    terms cancel in any balanced reaction, so differences of this are
    standard Gibbs energies of reaction.
    """
    T = jnp.asarray(T, dtype=float)
    return enthalpy_ig(T) - T * (_S0 + sensible_entropy(T))


def reaction_enthalpy(nu: Array, T: Array) -> Array:
    """Standard enthalpy of reaction (J/mol of reaction) for stoichiometry rows ``nu``."""
    return jnp.asarray(nu) @ enthalpy_ig(T)


def ln_K(nu: Array, T: Array) -> Array:
    """``ln K`` (1-bar standard state) of the reactions whose stoichiometry rows are ``nu``."""
    T = jnp.asarray(T, dtype=float)
    return -(jnp.asarray(nu) @ gibbs_ig(T)) / (R * T)


# -----------------------------------------------------------------------------
# Peng-Robinson for the real-gas departure, the separator and the compressor
# -----------------------------------------------------------------------------


@lru_cache(maxsize=None)
def peng_robinson():
    """difflow's :class:`~difflow.eos.PengRobinson` over the reformer species.

    No binary interaction parameters (``k_ij = 0`` for every pair, including
    hydrogen-hydrocarbon): the separator's hydrogen solubility is therefore
    the PR prediction without a fitted correction. Peng & Robinson, Ind. Eng.
    Chem. Fundam. 15(1), 59-64 (1976), doi:10.1021/i160057a011.
    """
    from difflow.eos import CriticalProperties, PengRobinson

    props = {k: CriticalProperties(k, s.Tc, s.Pc, s.omega, s.MW) for k, s in sp.SPECIES.items()}
    with jax.ensure_compile_time_eval():  # concrete parameter arrays even when first built under jit
        return PengRobinson(props)


@lru_cache(maxsize=None)
def cubic_thermo():
    """difflow's :class:`~difflow.thermo.CubicThermo` over the reformer species.

    The ideal-gas Cp in each :class:`~difflow.thermo.SpeciesData` is the same
    cubic the reactor integrates, so ``CubicThermo``'s enthalpy plus the heats
    of formation is :func:`total_enthalpy` exactly. The Antoine and Watson
    fields are required by ``SpeciesData`` but unused by ``CubicThermo``'s
    enthalpy and K-values (the PR departure carries the latent heat, the PR
    fugacities the K-values): the Antoine constants are a fit by this project
    to the Lee-Kesler vapour pressure (:func:`difflow_refinery.fit_antoine`),
    and the Watson heat of vaporization is a placeholder of zero.
    """
    from difflow.thermo import CubicThermo, IdealThermo, SpeciesData
    from difflow_refinery.assay import fit_antoine

    data = {}
    for k, s in sp.SPECIES.items():
        if s.Tc > 150.0:
            (A, B, C), (lo, hi) = fit_antoine(s.Tc, s.Pc, s.omega, (0.55 * s.Tc, 0.95 * s.Tc), Tb=s.Tb)
        else:  # hydrogen and methane: a nominal fit, never used
            A, B, C, lo, hi = 9.0, 100.0, 0.0, 0.0, 1e6
        data[k] = SpeciesData(name=k, MW=s.MW, Cp_coeffs=tuple(s.cp), Hvap_coeffs=(0.0, 0.38, s.Tc),
                              antoine_coeffs=(float(A), float(B), float(C)), Hf=s.Hf,
                              Cp_vapor_coeffs=tuple(s.cp), T_antoine_min=float(lo),
                              T_antoine_max=float(hi))
    with jax.ensure_compile_time_eval():
        return CubicThermo(IdealThermo(data), peng_robinson())


def vapor_departure(T: Array, P: Array, y: Array) -> Array:
    """PR molar enthalpy departure of a vapor of mole fractions ``y`` (J/mol)."""
    return peng_robinson().enthalpy_departure(T, P, y, "vapor")


def total_enthalpy(F: Array, T: Array, P: Array | None = None, phase: str = "vapor") -> Array:
    """Total enthalpy flow (W) of a single-phase stream of flows ``F`` (mol/s).

    ``Hf`` + ideal-gas sensible heat, plus the PR departure of the named
    phase when ``P`` is given.
    """
    F = jnp.asarray(F, dtype=float)
    H = F @ enthalpy_ig(T)
    if P is None:
        return H
    Ft = jnp.sum(F)
    y = F / Ft
    return H + Ft * peng_robinson().enthalpy_departure(T, P, y, phase)


def total_enthalpy_flash(F: Array, T: Array, P: Array) -> Array:
    """Two-phase-aware total enthalpy flow (W): ``CubicThermo`` flash enthalpy + Hf."""
    flows = {k: F[i] for i, k in enumerate(sp.NAMES)}
    return cubic_thermo().stream_enthalpy_flash(flows, T, P) + jnp.asarray(F) @ _HF


def temperature_from_enthalpy(F: Array, H: Array, P: Array, T_guess: Array,
                              n_iter: int = 8) -> Array:
    """Solve ``total_enthalpy(F, T, P) = H`` for a vapor by Newton's method.

    A fixed number of steps, unrolled, so it differentiates in both modes and
    traces under ``jit``. The enthalpy is smooth and nearly linear in ``T``
    over a reactor's temperature drop, so eight steps from the inlet
    temperature reach round-off.
    """
    def f(T):
        return total_enthalpy(F, T, P) - H

    T = jnp.asarray(T_guess, dtype=float)
    for _ in range(n_iter):
        r, dr = jax.jvp(f, (T,), (jnp.ones_like(T),))
        T = T - r / dr
    return T


def std_liquid_volume(F: Array) -> Array:
    """Standard liquid volume flow at 60 F (m^3/s) of flows ``F`` (mol/s).

    Ideal mixing by volume, the convention of a refinery volume yield. Species
    with no liquid density (hydrogen, methane) contribute nothing.
    """
    rho = np.where(np.isfinite(sp.RHO60), sp.RHO60, np.inf)
    return jnp.asarray(F) @ jnp.asarray(sp.MW / 1000.0 / rho)
