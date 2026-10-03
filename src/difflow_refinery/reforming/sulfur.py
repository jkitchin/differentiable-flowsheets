"""Sulfur through the reformer (#330): feed S -> H2S in the gases + residual S in reformate.

A reformer feed is hydrotreated to a fraction of a ppm of sulfur, because the
platinum catalyst is poisoned by it, and over that catalyst the organic sulfur
that is left is hydrogenolysed to H2S.  The H2S leaves the reactors with the
effluent, mostly with the separator vapour: part of it recirculates with the
recycle gas and the rest goes out with the net gas; what dissolves in the
separator liquid is stripped in the stabilizer and leaves with the fuel gas.
The organic sulfur that is not converted stays in the reformate.

Sulfur is carried as a TRACE ELEMENT, outside the reformer's species list and
outside its recycle tear.  At the sub-ppm levels a reformer runs at, the H2S
changes neither the hydrocarbon chemistry nor the phase split, so the sulfur
balance is solved after the hydrocarbon flowsheet, as a linear problem on its
converged streams.  Three simplifications, stated:

* ``conversion`` (the share of the feed's organic sulfur hydrogenolysed to
  H2S over the reactor train) is a parameter, ILLUSTRATIVE -- not a kinetic
  model of hydrodesulfurization over Pt.  The unconverted sulfur goes to the
  reformate.
* The separator splits H2S with an ideal K-value, ``K = Psat(T) / P``
  (Lee-Kesler vapour pressure from H2S's critical constants -- the
  ``chemicals`` 1.5.2 PSRK-table values ``Tc`` 373.53 K, ``Pc`` 89.63 bar,
  ``omega`` 0.0942), against the separator's own vapour and liquid molar
  flows.  Its share to the vapour is ``a = K V / (K V + L)``.  With a share
  ``s`` of that vapour recycled, the H2S into the separator is ``G / (1 - s
  a)`` for a make ``G``, and at steady state the net gas carries ``(1 - s) a
  G / (1 - s a)``, the liquid ``(1 - a) G / (1 - s a)``: the two add to
  ``G``, which is the balance (``tests/refinery/test_reforming.py``).
* The stabilizer sends all the dissolved H2S overhead with the fuel gas (it
  boils between ethane and propane; a real stabilizer puts some in the LPG,
  which is treated downstream).

The hydrogen the H2S takes from the hydrogen balance and the hydrocarbon
residue of the sulfur compound are not tracked: both are ppm of the stream
they come from.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from difflow_refinery.correlations import vapor_pressure

#: H2S constants (``chemicals`` 1.5.2, "Appendix to PSRK Revision 4" table).
H2S_TC = 373.53
H2S_PC = 89.63e5
H2S_OMEGA = 0.0942
H2S_MW = 34.081
S_MW = 32.065


def h2s_k_value(T: Array, P: Array) -> Array:
    """Ideal (Raoult) K-value of H2S at ``T`` (K), ``P`` (Pa)."""
    return vapor_pressure(jnp.asarray(T), H2S_TC, H2S_PC, H2S_OMEGA) / jnp.asarray(P)


def sulfur_balance(feed_mass: Array, feed_S_wppm: Array, conversion: Array,
                   V: Array, L: Array, T_sep: Array, P_sep: Array,
                   recycle_fraction: Array, reformate_mass: Array,
                   net_gas_moles: Array, recycle_moles: Array) -> dict[str, Array]:
    """Steady-state sulfur split of the reformer (see the module docstring).

    Args:
        feed_mass: Naphtha feed (kg/s).
        feed_S_wppm: Sulfur in the feed (wppm, organic).
        conversion: Share of it converted to H2S. ILLUSTRATIVE.
        V, L: Separator vapour and liquid (mol/s).
        T_sep, P_sep: Separator temperature (K) and pressure (Pa).
        recycle_fraction: Share of the separator vapour recycled.
        reformate_mass: Reformate (kg/s).
        net_gas_moles, recycle_moles: Net gas and recycle gas (mol/s).

    Returns:
        Sulfur flows (kg S/s) ``S.feed``, ``S.reformate``, ``S.net_gas``,
        ``S.fuel_gas``; H2S flows (mol/s) ``H2S.net_gas``, ``H2S.fuel_gas``,
        ``H2S.recycle``; ``reformate.S_wppm``, ``net_gas.H2S_ppmv``,
        ``recycle.H2S_ppmv`` and ``separator.H2S_to_vapor`` (the share ``a``).
    """
    S_in = jnp.asarray(feed_mass) * jnp.asarray(feed_S_wppm) * 1e-6     # kg S/s
    x = jnp.asarray(conversion)
    G = x * S_in / (S_MW * 1e-3)                                       # mol H2S/s made
    K = h2s_k_value(T_sep, P_sep)
    a = K * V / (K * V + L)
    s = jnp.asarray(recycle_fraction)
    into_sep = G / (1.0 - s * a)
    net = (1.0 - s) * a * into_sep
    liq = (1.0 - a) * into_sep
    rec = s * a * into_sep
    S_ref = (1.0 - x) * S_in
    kgS = S_MW * 1e-3
    return {
        "S.feed": S_in, "S.reformate": S_ref, "S.net_gas": net * kgS, "S.fuel_gas": liq * kgS,
        "H2S.net_gas": net, "H2S.fuel_gas": liq, "H2S.recycle": rec,
        "reformate.S_wppm": 1e6 * S_ref / reformate_mass,
        "net_gas.H2S_ppmv": 1e6 * net / jnp.asarray(net_gas_moles),
        "recycle.H2S_ppmv": 1e6 * rec / jnp.asarray(recycle_moles),
        "separator.H2S_to_vapor": a,
    }
