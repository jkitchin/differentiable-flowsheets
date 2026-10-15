"""Sulfur vapour: allotrope speciation, saturation over the liquid, dew point.

Gas-phase allotropes S2...S8 are in ideal-gas equilibrium with each other
(JANAF gases, :mod:`difflow_refinery.thermochemistry`): with ``p2`` the S2
partial pressure, ``p_n = K_n(T) P0 (p2/P0)^(n/2)``. Liquid sulfur
saturates the vapour when ``p2`` reaches ``p2_sat(T)``.

The JANAF liquid (S-004) does not reproduce the measured total vapour
pressure: against Ferreira & Lobo it is 24% high at 400 K and 30% low at
700 K (the 1 atm point at ~745 K against 717.75 K; pinned in the tests).
Scaling every ``p_n`` by one factor would break the gas-phase equilibrium
among the allotropes, so the correction goes on the LIQUID instead: its
Gibbs energy per S atom is shifted by ``delta(T)``, chosen so that the
saturated total equals the measured :func:`total_vapor_pressure`. Then
``p_n,sat = p_n,JANAF exp(n delta/RT)`` and gas-phase and saturated
speciation are one consistent model (``p2_sat`` is a point on the gas-phase
curve), which keeps the condenser continuous across the dew point.

The measured total is :data:`VAPOR_PRESSURE_SOURCE`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from difflow_refinery import thermochemistry as tc

#: allotrope sizes, in :data:`SULFUR_SPECIES` order
SIZES = jnp.arange(2.0, 9.0)
SULFUR_SPECIES = tuple(f"sulfur_s{n}" for n in range(2, 9))
_GAS = tc.IdealGasSet(list(SULFUR_SPECIES), cp_basis="wide")
P0 = tc.P_STD
R = tc.R


def scalar_root(f, x0, theta, iters: int = 40):
    """Root of a monotone scalar ``f(x, theta)``: Newton on stop-gradient
    ``theta``, then ONE Newton step with the slope held fixed, so the value is
    the root and its derivative in ``theta`` is the implicit one."""
    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    df = jax.grad(f)

    def body(_, x):
        return x - f(x, th0) / df(x, th0)

    x = lax.stop_gradient(lax.fori_loop(0, iters, body, jnp.asarray(x0, dtype=float)))
    return x - f(x, theta) / lax.stop_gradient(df(x, th0))


def bracket_root(f, lo, hi, theta, iters: int = 64):
    """Root of a scalar ``f(x, theta)`` that increases through zero on
    ``[lo, hi]``: bisection on stop-gradient ``theta`` (``f`` may return NaN
    where it is undefined; that side counts as positive), then ONE Newton
    step with the slope held fixed, for the implicit derivative."""
    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)

    def body(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        fm = f(m, th0)
        neg = fm < 0.0                       # False for NaN
        return jnp.where(neg, m, a), jnp.where(neg, b, m)

    a, b = lax.fori_loop(0, iters, body, (jnp.asarray(lo, dtype=float),
                                         jnp.asarray(hi, dtype=float)))
    x = lax.stop_gradient(a)                 # on the defined side
    return x - f(x, theta) / lax.stop_gradient(jax.grad(f)(x, th0))


def ln_K(T):
    """``ln K_n`` of ``(n/2) S2 = Sn`` (pressures in units of ``P0``), ``(..., 7)``."""
    T = jnp.asarray(T, dtype=float)
    G = _GAS.gibbs(T)
    return -(G - SIZES / 2 * G[..., :1]) / (R * T[..., None])


def janaf_saturation(T):
    """S2...S8 over the JANAF liquid (Pa), without the correction."""
    return tc.sulfur_vapor_pressures(T)


def liquid_correction(T):
    """``delta(T)/(R T)``: the shift of the liquid's Gibbs energy per S atom
    that makes the saturated total equal :func:`total_vapor_pressure`."""
    T = jnp.asarray(T, dtype=float)
    lnp = jnp.log(janaf_saturation(T))
    target = jnp.log(total_vapor_pressure(T))

    def f(u, th):
        return jax.nn.logsumexp(th[0] + SIZES * u) - th[1]

    return scalar_root(f, 0.0, (lnp, target), iters=30)


def saturation_pressures(T):
    """Partial pressures (Pa) of S2...S8 over liquid sulfur, ``(7,)``; they
    sum to :func:`total_vapor_pressure`."""
    return janaf_saturation(T) * jnp.exp(SIZES * liquid_correction(T))


def speciate(s_atoms, G, T, P):
    """Gas-phase allotrope moles (mol/s, ``(7,)``) holding ``s_atoms`` mol/s of
    S in ``G`` mol/s of other gas at ``T``, ``P`` (no liquid)."""
    lnK = ln_K(T)

    def atoms(x, th):           # x = ln(p2/P0); th = (s, G, P, lnK)
        s, G, P, lnK = th
        p = P0 * jnp.exp(lnK + SIZES / 2 * x)
        return jnp.log(G * jnp.sum(SIZES * p) / (P - jnp.sum(p))) - jnp.log(s)

    # every x with sum(p) < P lies below min_n (2/n) (ln(P/P0) - ln K_n)
    hi = jnp.min(2.0 / SIZES * (jnp.log(lax.stop_gradient(P) / P0) - lax.stop_gradient(lnK)))
    x = bracket_root(atoms, hi - 200.0, hi, (s_atoms, G, P, lnK), iters=80)
    p = P0 * jnp.exp(lnK + SIZES / 2 * x)
    return G * p / (P - jnp.sum(p))


def saturated(G, T, P):
    """Allotrope moles (mol/s) that saturate ``G`` mol/s of other gas."""
    p = saturation_pressures(T)
    return G * p / (P - jnp.sum(p))


def condense(s_atoms, G, T, P):
    """Equilibrium split of ``s_atoms`` mol/s of S at ``T``, ``P``.

    Returns:
        ``(vapour, liquid)``: allotrope moles ``(7,)`` and liquid S atoms.
        ``liquid`` is 0 above the dew point; below it the vapour is saturated.
        The two branches meet at the dew point (``p2_sat`` lies on the
        gas-phase curve), so the split is continuous there.
    """
    sat = saturated(G, T, P)
    s_sat = jnp.sum(SIZES * sat)
    wet = s_atoms > s_sat
    # the unsaturated branch is evaluated at an atom count it can always hold
    vap = speciate(jnp.where(wet, s_sat, s_atoms), G, T, P)
    vapour = jnp.where(wet, sat, vap)
    return vapour, jnp.where(wet, s_atoms - s_sat, 0.0)


def dew_point(s_atoms, G, P, lo: float = 300.0, hi: float = 1000.0):
    """Sulfur dew point (K): the T at which ``G`` mol/s of other gas at ``P``
    is saturated by ``s_atoms`` mol/s of S."""

    def f(T, th):
        s, G, P = th
        return jnp.log(jnp.sum(SIZES * saturated(G, T, P))) - jnp.log(s)

    return bracket_root(f, lo, hi, (s_atoms, G, P), iters=60)


# ---------------------------------------------------------------- measured total
#: Ferreira A.G.M., Lobo L.Q. (2011) "The low-pressure phase diagram of
#: sulfur", J. Chem. Thermodyn. 43:95-104, doi:10.1016/j.jct.2010.07.007,
#: eq 4 (Wagner form) fitted to 98 points (West & Menzies 1929, Baker 1971,
#: Rau et al. 1973, the normal boiling point); AAD 1.8%, triple point
#: (393.57 K) to Tc. Below 393.57 K it is the supercooled-liquid extrapolation.
VAPOR_PRESSURE_SOURCE = "Ferreira & Lobo 2011, doi:10.1016/j.jct.2010.07.007, eq 4"
_TC, _PC = 1313.0, 19811.0e3
_WAGNER = (-7.1742, -1.9154, 8.4830, -11.4867)


def total_vapor_pressure(T):
    """Measured total vapour pressure of liquid sulfur (Pa), :data:`VAPOR_PRESSURE_SOURCE`."""
    T = jnp.asarray(T, dtype=float)
    a1, a2, a3, a4 = _WAGNER
    tr = T / _TC
    tau = 1.0 - tr
    return _PC * jnp.exp((a1 * tau + a2 * tau**1.5 + a3 * tau**2.5 + a4 * tau**5) / tr)
