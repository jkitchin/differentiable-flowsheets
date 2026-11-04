"""Parallel-feed multi-effect distillation (MED).

A first-principles model: mass, salt and energy balances per effect with
the seawater properties of :mod:`difflow_water.chemistry.seawater` (content
verified against [S1] where stated there). No empirical design coefficient
enters except the heat-transfer coefficients the user gives. The
configuration (parallel feed, each effect heated by the previous effect's
vapor, a final condenser that preheats the feed) is the textbook one
(El-Dessouky & Ettouney [D2]; that book was not read here: the layout is
standard, the equations below are this module's own).

Effect ``i = 1..N`` (brine temperatures ``T_i`` falling linearly from
``T_top`` to ``T_last``):

- feed ``F/N`` at ``T_feed`` and salinity ``S_f``; brine ``B_i`` at ``T_i``
  and ``S_i = (F/N) S_f / B_i``; vapor ``D_i = F/N - B_i`` at ``T_i``
  (superheated by the boiling point elevation ``BPE(T_i, S_i)``);
- heat in: effect 1 from motive steam ``m_s`` condensing at ``T_steam``,
  ``Q_1 = m_s h_fg(T_steam)``; effect ``i > 1`` from ``D_{i-1}`` condensing
  at its saturation temperature ``T_{i-1} - BPE_{i-1}`` (superheat and latent
  heat), ``Q_i = D_{i-1} (h_v(T_{i-1}) - h_w(T_{i-1} - BPE_{i-1}))``;
- energy: ``Q_i + (F/N) h(T_feed, S_f) = B_i h(T_i, S_i) + D_i h_v(T_i)``,
  solved for ``D_i`` by Newton (``S_i`` depends on it);
- area: ``A_i = Q_i / (U (T_cond,i - T_i))`` with ``T_cond,i`` the
  condensing temperature on the hot side.

The last effect's vapor condenses in the final condenser at
``T_N - BPE_N``; it heats the whole feed from the intake temperature to
``T_feed``, and the rest of its heat goes to cooling seawater (also at the
intake temperature, warmed to ``T_feed``). Condensate leaves each effect
saturated; distillate flashing and flash boxes are not modelled (they
raise the output slightly). Pressures are not tracked (the effects run at
their vapor pressures).

Parallel feed heats every effect's feed from ``T_feed`` to the effect
temperature inside the effect, which costs GOR (about 4.6 for the default 8
effects); plants with feed preheaters between effects do better, and are
not modelled.

Outputs include the gain output ratio ``GOR = sum D_i / m_s``, the specific
thermal energy ``Q_1 / (product volume)`` and the total area. All
differentiable (fixed Newton counts; no Python control flow on values).
Default values are ILLUSTRATIVE.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream, make_stream
from difflow_water.chemistry import seawater as sw
from difflow_water.chemistry.seawater import MW_SEA_SALT, MW_WATER
from difflow_water.streams import SALT, WATER, mass_flows


@dataclass(repr=False)
class MEDParams(ParamsMixin):
    """Design of a parallel-feed MED plant (defaults ILLUSTRATIVE).

    Attributes:
        n_effects: Number of effects (-).
        steam_flow: Motive steam to the first effect (kg/s).
        T_steam: Condensing temperature of the motive steam (K).
        T_top: Brine temperature of the first effect (K).
        T_last: Brine temperature of the last effect (K).
        T_feed: Feed temperature after the final condenser (K).
        U_effect: Overall heat-transfer coefficient of the effects (W/(m^2 K)).
        U_condenser: Overall heat-transfer coefficient of the final condenser (W/(m^2 K)).
    """

    n_effects: int = 8
    steam_flow: float | Array = 1.0
    T_steam: float | Array = 343.15
    T_top: float | Array = 338.15
    T_last: float | Array = 313.15
    T_feed: float | Array = 308.15
    U_effect: float | Array = 2500.0
    U_condenser: float | Array = 2000.0


def _effect(Q, F, S_f, h_feed, T, n_iter=6):
    """Vapor ``D`` from one effect with heat ``Q``: Newton on the energy balance."""
    h_v = sw.vapor_enthalpy(T)

    def resid(D):
        B = F - D
        S = F * S_f / B
        return Q + F * h_feed - B * sw.enthalpy(T, S) - D * h_v

    D = Q / sw.latent_heat_water(T)
    for _ in range(n_iter):
        f, df = jax.jvp(resid, (D,), (jnp.ones_like(D),))
        D = D - f / df
    B = F - D
    return D, B, F * S_f / B


def med_performance(feed: Stream, params: MEDParams | None = None) -> dict:
    """Solve a parallel-feed MED plant.

    Args:
        feed: Seawater feed to the effects (``F_H2O``, ``F_TDS`` mol/s; its
            ``T`` is the intake temperature, below ``T_feed``).
        params: :class:`MEDParams`.

    Returns:
        dict with ``distillate_mass`` (kg/s), ``GOR`` (-), ``brine_salinity``
        (kg/kg, of the mixed brine), per-effect arrays ``T``, ``D``,
        ``salinity``, ``Q`` (W), ``area`` (m^2), ``area_total``,
        ``condenser_area``, ``cooling_water`` (kg/s), ``specific_thermal_energy``
        (kWh/m^3 of distillate), ``min_driving_dT`` (K; must be positive),
        and ``distillate``/``brine`` streams.

    Example:
        >>> from difflow_water.streams import make_seawater_stream
        >>> feed = make_seawater_stream(flow_mass=20.0, salinity=0.035)
        >>> out = med_performance(feed, MEDParams())
        >>> 4.0 < float(out["GOR"]) < 6.0
        True
    """
    p = params or MEDParams()
    N = int(p.n_effects)
    if N < 1:
        raise ValueError(f"n_effects must be at least 1, got {N}")
    mw, ms = mass_flows(feed)
    F_tot = mw + ms
    S_f = ms / F_tot
    F = F_tot / N
    T_in = feed["T"]
    h_feed = sw.enthalpy(p.T_feed, S_f)
    frac = jnp.linspace(0.0, 1.0, N) if N > 1 else jnp.zeros(1)
    Ts = p.T_top + (p.T_last - p.T_top) * frac

    Q = p.steam_flow * sw.latent_heat_water(p.T_steam)
    T_hot = jnp.asarray(p.T_steam, dtype=jnp.float64)
    D_l, B_l, S_l, Q_l, A_l, dT_l = [], [], [], [], [], []
    for i in range(N):
        T = Ts[i]
        D, B, S = _effect(Q, F, S_f, h_feed, T)
        dT = T_hot - T
        D_l.append(D), B_l.append(B), S_l.append(S), Q_l.append(Q), dT_l.append(dT)
        A_l.append(Q / (p.U_effect * dT))
        bpe = sw.boiling_point_elevation(T, S)
        T_hot = T - bpe
        Q = D * (sw.vapor_enthalpy(T) - sw.enthalpy_water(T_hot))
    # final condenser: last vapor (heat Q, at T_hot) preheats the feed, the rest to cooling
    q_feed = F_tot * (h_feed - sw.enthalpy(T_in, S_f))
    q_cool = Q - q_feed
    m_cool = q_cool / (sw.enthalpy(p.T_feed, S_f) - sw.enthalpy(T_in, S_f))
    lmtd = ((T_hot - T_in) - (T_hot - p.T_feed)) / jnp.log((T_hot - T_in) / (T_hot - p.T_feed))
    A_cond = Q / (p.U_condenser * lmtd)

    D = jnp.stack(D_l)
    B = jnp.stack(B_l)
    S = jnp.stack(S_l)
    D_tot = jnp.sum(D)
    B_tot = jnp.sum(B)
    S_brine = ms / B_tot
    T_brine = jnp.sum(B * Ts) / B_tot
    V_dist = D_tot / sw.density(Ts[-1], 0.0)
    Q1 = Q_l[0]
    distillate = make_stream({WATER: D_tot / MW_WATER, SALT: 0.0 * D_tot}, Ts[-1], 101325.0)
    brine = make_stream({WATER: B_tot * (1.0 - S_brine) / MW_WATER,
                         SALT: B_tot * S_brine / MW_SEA_SALT}, T_brine, 101325.0)
    return {
        "distillate_mass": D_tot,
        "recovery": D_tot / F_tot,
        "GOR": D_tot / p.steam_flow,
        "brine_salinity": S_brine,
        "T": Ts,
        "D": D,
        "salinity": S,
        "Q": jnp.stack(Q_l),
        "area": jnp.stack(A_l),
        "area_total": jnp.sum(jnp.stack(A_l)) + A_cond,
        "condenser_area": A_cond,
        "condenser_heat": Q,
        "cooling_water": m_cool,
        "specific_thermal_energy": Q1 / V_dist / 3.6e6,
        "min_driving_dT": jnp.min(jnp.stack(dT_l)),
        "distillate": distillate,
        "brine": brine,
    }
