"""Peng-Robinson flash and the high-pressure separator.

At 30-150 bar and 40-60 C the separator's question is how much hydrogen,
H2S and light hydrocarbon dissolves in the oil and leaves with it (a loss of
hydrogen, and sour gas the stripper must remove), and how much light naphtha
goes with the recycle gas. Raoult's law with a Lee-Kesler vapour pressure
cannot answer it -- hydrogen is 300 K above its critical point -- so the
flash is Peng-Robinson on every component (:mod:`.thermo`).

:func:`pr_flash` is an isothermal flash solved as ``n`` equations in
``ln K``::

    ln K_i - ln phi_i^L(x) + ln phi_i^V(y) = 0,
    x = z / (1 + V (K - 1)),  y = K x,     V from Rachford-Rice

Rachford-Rice is solved as a *negative flash* (Whitson & Michelsen 1989):
``V`` is bracketed between the poles ``1/(1 - K_max)`` and ``1/(1 - K_min)``
rather than clipped to [0, 1], so the equations stay smooth through a phase
boundary, and ``x``, ``y`` remain the equilibrium compositions of the
(incipient) phases. The reported phase amounts clip ``V`` to [0, 1].

Successive substitution from Wilson's K-values (a fixed number of passes,
under ``stop_gradient``) gives the start, Newton (``optimistix``) finishes,
and its implicit adjoint gives the derivative: ``-J^-1 dR/dtheta`` at the
solution, never the iteration.

:class:`HPSeparator` applies it to a :class:`~.layout.Flows`: gases and cut
molecules split by the flash, every attribute with its cut, water decanted
(no free-water VLE).

References:
    Rachford, H.H. and Rice, J.D., "Procedure for use of electronic digital
        computers in calculating flash vaporization hydrocarbon equilibrium",
        J. Petrol. Technol. 4(10), sec. 1, 19 and sec. 2, 3 (1952),
        doi:10.2118/952327-G (unverified).
    Whitson, C.H. and Michelsen, M.L., "The negative flash", Fluid Phase
        Equilib. 53, 51-71 (1989), doi:10.1016/0378-3812(89)80072-X
        (unverified).
    Michelsen, M.L., "The isothermal flash problem. Part II. Phase-split
        calculation", Fluid Phase Equilib. 9(1), 21-40 (1982),
        doi:10.1016/0378-3812(82)85002-4 (unverified) -- successive
        substitution then Newton in ln K.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.thermo import Components, pr_lnphi, wilson_lnK

jax.config.update("jax_enable_x64", True)

_RR_BISECT = 120
_SS_PASSES = 25


def rachford_rice(z: Array, K: Array) -> Array:
    """Vapour fraction of the negative flash: the root of ``sum z (K-1)/(1+V(K-1))``.

    Bisection between the poles under ``stop_gradient``, then two Newton
    steps on the live inputs. The first attaches the implicit-function
    derivative ``dV/d(z, K)``; the second makes the *second* derivatives
    exact too (each step from the root doubles the order to which the
    derivatives are right), which matters because the reactor differentiates
    a quantity that is itself a derivative (``C_eff = dH/dT``, ``d ln K/dT``).
    Needs ``K_max > 1 > K_min`` (otherwise returns the nearer bound).
    """
    zs, Ks = jax.lax.stop_gradient(z), jax.lax.stop_gradient(K)
    lo = 1.0 / (1.0 - jnp.max(Ks))
    hi = 1.0 / (1.0 - jnp.min(Ks))
    span = hi - lo
    lo, hi = lo + 1e-14 * jnp.abs(span), hi - 1e-14 * jnp.abs(span)

    def f(V, z, K):
        return jnp.sum(z * (K - 1.0) / (1.0 + V * (K - 1.0)))

    def body(_, b):
        lo, hi = b
        mid = 0.5 * (lo + hi)
        right = f(mid, zs, Ks) > 0.0
        return jnp.where(right, mid, lo), jnp.where(right, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _RR_BISECT, body, (lo, hi))
    V = 0.5 * (lo + hi)
    for _ in range(2):
        V = V - f(V, z, K) / jax.grad(f)(V, z, K)
    return V


def _phases(z, lnK):
    K = jnp.exp(lnK)
    V = rachford_rice(z, K)
    x = z / (1.0 + V * (K - 1.0))
    y = K * x
    return V, x / jnp.sum(x), y / jnp.sum(y)


def _residual(lnK, args):
    T, P, z, comps = args
    _, x, y = _phases(z, lnK)
    lnphi_l, _ = pr_lnphi(T, P, x, comps, "liquid")
    lnphi_v, _ = pr_lnphi(T, P, y, comps, "vapor")
    return lnK - (lnphi_l - lnphi_v)


@dataclass(frozen=True)
class FlashResult:
    """An isothermal PR flash.

    Attributes:
        V: Vapour fraction (negative-flash value, may lie outside [0, 1]).
        beta: ``V`` clipped to [0, 1], the vapour fraction actually reported.
        x, y: Liquid and vapour mole fractions.
        lnK: ``ln(y/x)`` at the solution.
        residual: Largest |residual| of the K equations.
    """

    V: Array
    beta: Array
    x: Array
    y: Array
    lnK: Array
    residual: Array

    @property
    def vapor_fraction_by_component(self) -> Array:
        """Fraction of each component's feed that is vapour: ``beta K / (1 + beta (K-1))``."""
        K = jnp.exp(self.lnK)
        return self.beta * K / (1.0 + self.beta * (K - 1.0))


jax.tree_util.register_dataclass(FlashResult, data_fields=["V", "beta", "x", "y", "lnK", "residual"],
                                 meta_fields=[])


def pr_flash(T, P, z, comps: Components, lnK0: Array | None = None,
             max_steps: int = 60, rtol: float = 1e-12, atol: float = 1e-12) -> FlashResult:
    """Isothermal Peng-Robinson flash of feed ``z`` at ``(T, P)``.

    Args:
        T, P: Temperature (K) and pressure (Pa).
        z: Feed mole fractions (or flows; normalised here), in ``comps`` order.
        comps: The component table.
        lnK0: Starting ``ln K`` (default Wilson, then successive substitution).
        max_steps, rtol, atol: Newton controls.

    Differentiable in everything, by the implicit-function theorem.
    """
    T = jnp.asarray(T, dtype=float)
    P = jnp.asarray(P, dtype=float)
    z = jnp.asarray(z, dtype=float)
    z = jnp.maximum(z, 1e-300)
    z = z / jnp.sum(z)
    args = (T, P, z, comps)
    sargs = jax.lax.stop_gradient(args)
    lnK = wilson_lnK(sargs[0], sargs[1], sargs[3]) if lnK0 is None else jax.lax.stop_gradient(lnK0)

    def ss(_, lk):
        return lk - _residual(lk, sargs)

    lnK = jax.lax.fori_loop(0, _SS_PASSES, ss, lnK)
    sol = optx.root_find(_residual, optx.Newton(rtol=rtol, atol=atol), lnK, args=args,
                         max_steps=max_steps, throw=False)
    lnK = sol.value
    V, x, y = _phases(z, lnK)
    res = jnp.max(jnp.abs(_residual(jax.lax.stop_gradient(lnK), sargs)))
    return FlashResult(V=V, beta=jnp.clip(V, 0.0, 1.0), x=x, y=y, lnK=lnK, residual=res)


# =============================================================================
# The separator
# =============================================================================


def flash_components(flows: Flows, layout: Layout, comps: Components) -> Array:
    """Molar flows of the flashing components (gases except water, then cuts)."""
    gas = jnp.stack([flows.gas[layout.gas_index(g)] for g in comps.names[:comps.n_gas]]) \
        if comps.n_gas else jnp.zeros(0)
    return jnp.concatenate([gas, flows.cut])


def split_by_vapor_fraction(flows: Flows, layout: Layout, comps: Components, frac: Array):
    """``(vapour, liquid, water)`` of ``flows`` given each component's vapour fraction.

    Attributes follow their cut; water (if the layout has it) is the third
    stream, decanted whole.
    """
    gf = jnp.zeros(layout.n_gas)
    for i, g in enumerate(comps.names[:comps.n_gas]):
        gf = gf.at[layout.gas_index(g)].set(frac[i])
    cf = frac[comps.n_gas:]
    water_mask = jnp.asarray([1.0 if g == "water" else 0.0 for g in layout.gases]) \
        if layout.n_gas else jnp.zeros(0)
    vap = flows.split(gf, cf)
    water = Flows(flows.gas * water_mask, jnp.zeros(layout.n_cut), jnp.zeros((layout.n_cut, layout.n_attr)))
    liq = flows - vap - water
    return vap, liq, water


@dataclass(frozen=True)
class HPSeparator:
    """High-pressure separator: an isothermal PR flash of the reactor effluent.

    Attributes:
        layout: The stream layout. Temperature and pressure are arguments of
            :meth:`__call__`, so they can carry gradients.
    """

    layout: Layout

    def __call__(self, flows: Flows, T, P, comps: Components, lnK0=None):
        """Flash ``flows``; returns ``(vapour, liquid, water, FlashResult)``."""
        z = flash_components(flows, self.layout, comps)
        fr = pr_flash(T, P, z, comps, lnK0=lnK0)
        vap, liq, water = split_by_vapor_fraction(flows, self.layout, comps,
                                                  fr.vapor_fraction_by_component)
        return vap, liq, water, fr


__all__ = ["rachford_rice", "pr_flash", "FlashResult", "HPSeparator", "flash_components",
           "split_by_vapor_fraction"]
