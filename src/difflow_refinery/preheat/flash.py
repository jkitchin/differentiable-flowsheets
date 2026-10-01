"""Crude-side phase split and enthalpy on :class:`~difflow_refinery.thermo.ColumnThermo`.

The crude in a preheat train carries water -- the tank's BS&W, then the
desalter's wash water, then what the desalter leaves -- so the split has
three phases: a hydrocarbon liquid that holds no water, a vapour of both,
and free liquid water. On the column's model (Raoult over Lee-Kesler vapour
pressures, water immiscible with the hydrocarbon liquid) that is two cases:

* **Free water present.** Water's partial pressure is its own vapour
  pressure, so ``y_w = s = Psat_w / P``, and the hydrocarbons see the rest of
  the pressure: their Rachford-Rice equation is the ordinary one with
  ``K' = K / (1 - s)``. The water vapour is ``s / (1 - s)`` times the
  hydrocarbon vapour, and what is left of the water is liquid.
* **No free water.** All the water is vapour, the case the column's own feed
  flash treats (:meth:`~difflow_refinery.column.CrudeColumn._feed_split`):
  Rachford-Rice with an infinite-K term ``z_w / psi``.

The first holds when ``s < 1`` and it leaves a non-negative amount of free
water; otherwise the second. The two agree on the boundary, so the split is
continuous. Without water the second *is* the column's flash, computed the
same way, so a furnace inlet enthalpy computed here equals the one the
column computes to round-off -- which is what lets the energy balance close
across the train and the column.

Each Rachford-Rice root is bisected on stop-gradiented K-values and then
given one Newton step, the column's construction: the value is the root and
the derivative is the implicit one.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from difflow_refinery.thermo import ColumnThermo

_BISECT = 80


def _rr_root(z, K, extra=0.0):
    """Root ``psi`` of ``sum z (K-1)/(1+psi(K-1)) + extra/psi`` in (0, 1].

    Returns ``(psi, frac_v)`` with ``frac_v`` the fraction of each component
    in the vapour; the all-liquid and all-vapour ends are handled as the
    column handles them.
    """

    def rr(psi):
        return jnp.sum(z * (K - 1.0) / (1.0 + psi * (K - 1.0))) + extra / psi

    Ks, zs, es = (jax.lax.stop_gradient(a) for a in (K, z, jnp.asarray(extra, dtype=float)))

    def rr_s(psi):
        return jnp.sum(zs * (Ks - 1.0) / (1.0 + psi * (Ks - 1.0))) + es / psi

    def bisect(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        pos = rr_s(mid) > 0
        return jnp.where(pos, mid, lo), jnp.where(pos, hi, mid)

    lo, hi = jax.lax.fori_loop(0, _BISECT, bisect, (jnp.asarray(1e-12), jnp.asarray(1.0)))
    psi0 = 0.5 * (lo + hi)
    psi = psi0 - rr(psi0) / jax.grad(rr)(psi0)
    all_vap = rr_s(jnp.asarray(1.0)) >= 0
    all_liq = (es == 0) & (rr_s(jnp.asarray(1e-12)) <= 0)
    psi = jnp.where(all_vap, 1.0, jnp.where(all_liq, 0.0, psi))
    frac_v = jnp.where(all_vap, 1.0, jnp.where(all_liq, 0.0, psi * K / (1.0 + psi * (K - 1.0))))
    return psi, frac_v


def crude_split(th: ColumnThermo, f: Array, fw: Array, T: Array, P: Array) -> dict:
    """Three-phase split of a wet crude at ``T`` and ``P``.

    Args:
        th: The column thermo.
        f: Hydrocarbon molar flows ``(nc,)`` (mol/s).
        fw: Water (mol/s).
        T, P: Temperature (K) and pressure (Pa).

    Returns:
        ``{"liquid": (nc,), "vapor": (nc,), "water_vapor", "water_liquid"}``,
        all mol/s.
    """
    f = jnp.asarray(f, dtype=float)
    fw = jnp.asarray(fw, dtype=float)
    K = th.K(T, P)
    # no free water: the column's feed flash
    total = jnp.sum(f) + fw
    _, fv_a = _rr_root(f / total, K, fw / total)
    # free water: hydrocarbons against the pressure water leaves them
    s = th.water_psat(T) / P
    s_safe = jnp.minimum(s, 0.999)
    Fh = jnp.sum(f)
    _, fv_b = _rr_root(f / Fh, K / (1.0 - s_safe))
    v_b = f * fv_b
    wv_b = s_safe / (1.0 - s_safe) * jnp.sum(v_b)
    free = (s < 1.0) & (fw > wv_b)
    frac_v = jnp.where(free, fv_b, fv_a)
    wv = jnp.where(free, wv_b, fw)
    return {"liquid": f * (1.0 - frac_v), "vapor": f * frac_v,
            "water_vapor": wv, "water_liquid": fw - wv}


def crude_enthalpy(th: ColumnThermo, f: Array, fw: Array, T: Array, P: Array) -> Array:
    """Enthalpy flow (W) of a wet crude at ``T`` and ``P``, phases as :func:`crude_split`."""
    sp = crude_split(th, f, fw, T, P)
    return (jnp.sum(sp["liquid"] * th.h_liquid(T)) + jnp.sum(sp["vapor"] * th.h_vapor(T))
            + sp["water_vapor"] * th.water_h_vapor(T) + sp["water_liquid"] * th.water_h_liquid(T))


def liquid_enthalpy(th: ColumnThermo, f: Array, T: Array) -> Array:
    """Enthalpy flow (W) of a hydrocarbon stream held liquid: a column product
    or pumparound, as the column itself counts it."""
    return jnp.sum(jnp.asarray(f) * th.h_liquid(T))


class ColumnThermoEnthalpy:
    """A :class:`ColumnThermo` as the thermo of a difflow heat exchanger.

    :class:`~difflow.units.heat_exchanger.EnthalpyCounterCurrentHX` takes any
    thermo with ``stream_enthalpy_flash(flows, T, P)``; this supplies it from
    :func:`crude_enthalpy`, with ``flows`` a dict of ``F_<name>``-less
    component flows (and ``"water"``). Hashed by identity, as that unit's jit
    requires, so build one and reuse it.

    Args:
        thermo: The column thermo.
        liquid: Hold the stream liquid (a column product or pumparound)
            instead of flashing it.
    """

    def __init__(self, thermo: ColumnThermo, liquid: bool = False):
        self.thermo = thermo
        self.liquid = liquid

    def _arrays(self, flows: dict):
        th = self.thermo
        f = jnp.stack([jnp.asarray(flows.get(n, 0.0), dtype=float) for n in th.names])
        return f, jnp.asarray(flows.get("water", 0.0), dtype=float)

    def stream_enthalpy_flash(self, flows: dict, T, P) -> Array:
        f, fw = self._arrays(flows)
        if self.liquid:
            return liquid_enthalpy(self.thermo, f, T) + fw * self.thermo.water_h_liquid(T)
        return crude_enthalpy(self.thermo, f, fw, T, P)


__all__ = ["ColumnThermoEnthalpy", "crude_enthalpy", "crude_split", "liquid_enthalpy"]
