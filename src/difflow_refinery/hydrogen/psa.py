r"""Hydrogen-purification PSA: a layered-bed cycle model (#469).

A library. :class:`HydrogenPSA` takes a feed (mol/s by species) and returns
the high-purity H2 product and the low-pressure tail gas (fuel for the SMR
furnace), with purity and recovery that follow from the bed, the isotherms
and the cycle -- not from given numbers. :meth:`PSA.from_model
<difflow_refinery.hydrogen.network.PSA.from_model>` hands the result to the
hydrogen network's shortcut ``PSA(recovery, purity)``.

The model (per bed, one cycle at cyclic steady state)
-----------------------------------------------------
Bed: an activated-carbon (AC) layer at the feed end and a zeolite 5A layer
at the product end (``ac_fraction`` of the length is AC), Skarstrom-type
cycle: adsorption at ``P_ads`` for ``t_ads``, ``n_equalizations`` pressure
equalizations (0 or 1) with the other bed, blowdown to ``P_purge``, purge
with product H2, repressurization with product.

1. **Equilibrium.** Competitive dual-site Langmuir on each adsorbent,

   .. math::

       q_i = \frac{k_{1,i} B_{1,i} p_i}{1 + \sum_j B_{1,j} p_j}
           + \frac{k_{4,i} B_{2,i} p_i}{1 + \sum_j B_{2,j} p_j},\quad
       B_1 = k_2 e^{k_3/T},\ B_2 = k_5 e^{k_6/T}

   (``p`` in atm, ``q`` in mol/g), parameters of Park, Kim & Cho (1998) as
   tabulated by Zhang, Hu & Xin (2025), Table 2 (:data:`DSL_PARAMETERS`).

2. **Breakthrough.** Each impurity's front loads the bed at the feed
   loading (the chord ``q_i(feed)/c_i(feed)``). Its stoichiometric time
   through the layers is ``t_s,i = sum_k rho_b,k V_k q_i,k / (F y_i)`` (the
   solid) plus the gas holdup. The effluent concentration is the Klinkenberg
   (1948) approximation to the linear-driving-force solution,

   .. math::

       \frac{c}{c_0} \approx \tfrac12\left[1 + \operatorname{erf}\left(
           \sqrt{\tau} - \sqrt{\xi} + \frac{1}{8\sqrt{\tau}} + \frac{1}{8\sqrt{\xi}}
       \right)\right],\quad \xi = \omega_i t_{s,i},\ \tau = \omega_i (t - t_{g,i}),

   averaged over the adsorption step by Gauss-Legendre quadrature. With a
   favourable (Langmuir) isotherm the real front self-sharpens, so the
   linear-isotherm spread is an upper bound on the leak.

3. **Purge.** Equilibrium theory: the purge desorbs the swept zone as a
   simple (dispersive) wave in which a concentration ``c`` travels at
   ``u / n'(c)``, with ``n(c) = eps_t c + rho_b q(c)`` the bed holdup
   (Ruthven 1984, ch. 8). A purge of actual volume ``V_p`` at ``P_purge``
   through a swept volume ``V_s`` therefore leaves the heel

   .. math::

       H = V_s \int_0^{c_L} \left(n'(c) - V_p/V_s\right)^+ \, dc ,

   with ``c_L`` the post-blowdown state, taken as the feed composition at
   ``P_purge``. For a linear isotherm ``H/n`` is ``(1 - gamma)+``, where
   ``gamma`` = purge volume / feed volume, which is Skarstrom's (1959)
   condition (tested). A favourable isotherm is steep at ``c -> 0``, so a
   heel stays even when ``gamma > 1``. The integral runs along the feed
   composition ray, ``p = s y P_purge`` with ``s`` in [0, 1], on the
   competitive isotherm, layer by layer, each layer seeing the whole purge.
   The heel then starts the next feed front further down the bed: ``t_s,i``
   loses ``H_i / F_i``. The ``(.)+`` is a softplus of width
   :data:`HEEL_SMOOTH` in units of ``n'``.

4. **Recovery.** H2 to the tail gas is the bed's H2 inventory at the end of
   adsorption (voids + adsorbed; the swept zone at the feed composition, the
   rest pure H2 at ``P_ads``), less what equalization passes to the other
   bed (the inventory above ``P_eq = (P_ads + P_purge)/2``), less the
   inventory of a clean bed at ``P_purge``, plus the purge H2
   (``purge_to_feed`` x feed H2 -- the P/F of Yang & Lee 1998 as used by
   Zhang et al.). Repressurization gas comes from product and returns, so it
   nets out.

Species: H2, CH4, CO, CO2 carry isotherms. Other species are mapped by
:data:`TO_TAIL` (C2+ hydrocarbons, H2S, water -- all adsorbed more strongly
than CH4 on activated carbon, so held wherever CH4 is held) or
:data:`TO_PRODUCT` (N2, Ar -- weaker than CO on 5A and not in the table,
passed conservatively to product). Anything else raises.

Balances: per species, feed = product + tail by construction, and the
product impurity is computed independently from the leak (tested).

What is NOT modelled: heat effects (isothermal at ``T``), roll-up of weaker
species by stronger ones, axial dispersion, pressure drop, step timing
beyond ``t_ads``. The LDF coefficients :data:`LDF_OMEGA` are ILLUSTRATIVE
(no source gives them for these pellets); purity depends on them, recovery
hardly at all.

Validation (``tests/refinery/test_psa.py``): the isotherms reproduce the
affinity orders Zhang et al. state (AC: CO2 > CH4 > CO > H2; 5A: CO2 > CO >
CH4 > H2) and the breakthrough order of their Fig. 4 (CO before CH4 in AC,
CH4 before CO in 5A, the two together in a 0.7 AC layered bed); the
Klinkenberg form agrees with the exact Anzelius/Thomas linear-LDF solution;
balances close to round-off; purity and recovery move with adsorption time,
P/F and pressure as in Zhang et al., Table 6, with recovery in the range
reported there; ``check_grads`` in the cycle parameters.

References
----------
- Zhang, N., Hu, S. & Xin, Q., "Optimization of pressure swing adsorption in
  a three-layered bed for hydrogen purification using machine learning
  model", Sci. Rep. 15, 14193 (2025), doi:10.1038/s41598-025-97139-4
  (open access; Tables 1, 2, 6 read from the full text, PMC12019191).
- Park, J.H., Kim, J.N. & Cho, S.H., "Adsorber dynamics and optimal design
  of layered beds for multicomponent gas adsorption", Chem. Eng. Sci. 53,
  3951-3963 (1998), doi:10.1016/S0009-2509(98)00196-1 (the isotherm
  source, read through Zhang et al.; not seen directly).
- Klinkenberg, A., "Numerical evaluation of equations describing transient
  heat and mass transfer in packed solids", Ind. Eng. Chem. 40, 1992-1994
  (1948).
- Ruthven, D.M., *Principles of Adsorption and Adsorption Processes*,
  Wiley, New York (1984), ch. 8 (equilibrium theory, simple waves).
- Skarstrom, C.W., "Use of adsorption phenomena in automatic plant-type gas
  analyzers", Ann. N.Y. Acad. Sci. 72, 751-763 (1959).
- Yang, J. & Lee, C.-H., "Adsorption dynamics of a layered bed PSA for H2
  recovery from coke oven gas", AIChE J. 44, 1325-1334 (1998) (P/F
  definition, via Zhang et al.).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin

R = 8.314462618        # J/mol/K (CODATA 2018 exact)
ATM = 101325.0         # Pa

#: Species with isotherms, in array order.
ADSORBED = ("hydrogen", "methane", "carbon_monoxide", "carbon_dioxide")

#: Dual-site Langmuir constants ``(k1 mol/g, k2 1/atm, k3 K, k4 mol/g, k5 1/atm,
#: k6 K)`` per adsorbent and species: Zhang, Hu & Xin (2025) Table 2, from Park,
#: Kim & Cho (1998). Zhang's column header reads "mmol/g" for k1 and k4; the
#: magnitudes are mol/g (CO2 on AC 3.2 mmol/g at 1 atm and 293 K; read as
#: mmol/g it would be 3.2 umol/g), and the mol/g reading reproduces the
#: published affinity orders, which ``test_psa.py`` checks.
DSL_PARAMETERS: dict[str, dict[str, tuple[float, ...]]] = {
    "activated_carbon": {
        "hydrogen": (2.4e-5, 9.0e-4, 1700.0, 4.8e-4, 6.0e-5, 1915.0),
        "carbon_monoxide": (3.65e-3, 8.7e-5, 2003.0, 4.37e-4, 2.0e-3, 2050.0),
        "methane": (5.8e-3, 1.0e-4, 2060.0, 7.4e-4, 3.5e-3, 2200.0),
        "carbon_dioxide": (8.0e-3, 8.0e-6, 3100.0, 1.4e-3, 9.6e-7, 4750.0),
    },
    "zeolite_5a": {
        "hydrogen": (7.8e-4, 9.0e-5, 1647.0, 7.0e-6, 1.2e-5, 3000.0),
        "carbon_monoxide": (2.3e-3, 9.7e-5, 2920.0, 9.5e-4, 7.9e-5, 3280.0),
        "methane": (2.6e-3, 2.1e-4, 2200.0, 7.5e-4, 3.0e-9, 4800.0),
        "carbon_dioxide": (3.0e-3, 1.1e-3, 2508.0, 1.4e-3, 1.0e-2, 3300.0),
    },
}

#: Pellet density (kg/m3), interparticle and total void fraction: Zhang et al.
#: (2025) Table 1.
ADSORBENT_PROPERTIES: dict[str, dict[str, float]] = {
    "activated_carbon": {"rho_p": 850.0, "eps_b": 0.433, "eps_t": 0.78},
    "zeolite_5a": {"rho_p": 1160.0, "eps_b": 0.357, "eps_t": 0.77},
}

#: LDF coefficients (1/s). ILLUSTRATIVE: of the order usual for these gases
#: on ~1-2 mm pellets; no source here gives them.
LDF_OMEGA: dict[str, float] = {"hydrogen": 0.7, "methane": 0.2, "carbon_monoxide": 0.15,
                               "carbon_dioxide": 0.04}

#: Species assumed held by the bed (to tail gas): C2+ hydrocarbons, H2S and
#: water adsorb more strongly than CH4 on activated carbon (and water on the
#: guard layer a real unit has).
TO_TAIL = frozenset({"ethane", "ethylene", "propane", "propylene", "isobutane", "n_butane",
                     "isopentane", "n_pentane", "n_hexane", "hydrogen_sulfide", "water"})
#: Species passed to product (weakly adsorbed, no isotherm here): conservative.
TO_PRODUCT = frozenset({"nitrogen", "argon"})

#: Softplus width of ``(n'(c) - V_p/V_s)+`` relative to ``V_p/V_s`` (numerical).
HEEL_SMOOTH = 0.01
_N_QUAD = 64


def dsl_loading(adsorbent: str, p_atm: Array, T: Array) -> Array:
    """Competitive dual-site Langmuir loading (mol/kg) of :data:`ADSORBED`.

    Args:
        adsorbent: ``"activated_carbon"`` or ``"zeolite_5a"``.
        p_atm: Partial pressures (atm), last axis over :data:`ADSORBED`.
        T: Temperature (K).

    Returns:
        Loadings (mol per kg adsorbent), same shape as ``p_atm``.

    Example:
        >>> q = dsl_loading("activated_carbon", jnp.array([0., 0., 0., 1.]), 293.15)
        >>> round(float(q[3]), 2)          # pure CO2, 1 atm, mol/kg
        3.19
    """
    k = jnp.asarray([DSL_PARAMETERS[adsorbent][s] for s in ADSORBED])
    p = jnp.asarray(p_atm, dtype=float)
    B1 = k[:, 1] * jnp.exp(k[:, 2] / T)
    B2 = k[:, 4] * jnp.exp(k[:, 5] / T)
    s1 = k[:, 0] * B1 * p / (1.0 + jnp.sum(B1 * p, axis=-1, keepdims=True))
    s2 = k[:, 3] * B2 * p / (1.0 + jnp.sum(B2 * p, axis=-1, keepdims=True))
    return 1000.0 * (s1 + s2)


def klinkenberg(xi: Array, tau: Array) -> Array:
    """``c/c0`` of a linear-isotherm LDF bed, Klinkenberg (1948) approximation.

    ``xi = omega x solid holdup time``, ``tau = omega x (t - gas holdup time)``.
    Zero for ``tau <= 0`` (the front has not arrived).
    """
    safe = jnp.where(tau > 1e-6, tau, 1e-6)
    st = jnp.sqrt(safe)
    sx = jnp.sqrt(xi)
    c = 0.5 * (1.0 + jax.scipy.special.erf(st - sx + 1.0 / (8.0 * st) + 1.0 / (8.0 * sx)))
    return jnp.where(tau > 1e-6, c, 0.0)


def purge_heel(adsorbent: str, y: Array, P: Array, T: Array, v: Array, eps_t: float,
               rho_b: float) -> Array:
    """Heel (mol per m3 of swept bed) the purge leaves, per species of :data:`ADSORBED`.

    ``v`` is the purge volume (at ``P``) per swept bed volume. Equilibrium
    theory, simple wave along ``p = s y P`` (see the module docstring)::

        H_i = int_0^1 (dn_i/ds - v dc_i/ds)+ ds,   n_i = eps_t c_i + rho_b q_i

    Linear isotherm: ``H_i / n_i(1) = (1 - v c_i / n_i)+`` (Skarstrom).
    """
    x, w = np.polynomial.legendre.leggauss(_N_QUAD)
    s = 0.5 * (jnp.asarray(x) + 1.0)
    c_full = y * P / (R * T)

    def n_of(si):
        return eps_t * si * c_full + rho_b * dsl_loading(adsorbent, si * y * P / ATM, T)

    dn = jax.vmap(jax.jacfwd(n_of))(s)                 # (n_s, n_species)
    excess = dn - v * c_full[None, :]
    scale = HEEL_SMOOTH * v * c_full[None, :] + 1e-30
    pos = scale * jax.nn.softplus(excess / scale)
    return 0.5 * jnp.sum(jnp.asarray(w)[:, None] * pos, axis=0)


@dataclass
class HydrogenPSAParams(ParamsMixin):
    """Bed, isotherm and cycle of a hydrogen PSA.

    Attributes:
        bed_length: Bed length (m). Default 1.68, Zhang et al. (2025) Table 1.
        bed_diameter: Bed inside diameter (m). Default 0.0384, Zhang et al.
            Table 1 (a laboratory bed; scale with ``n_beds``/``bed_diameter``).
        ac_fraction: Fraction of the length that is activated carbon (feed
            end); the rest is zeolite 5A. Default 0.5 (illustrative).
        n_beds_on_feed: Beds receiving feed at once (the feed is split).
        T: Bed temperature (K), isothermal. Default 298.15.
        P_ads: Adsorption pressure (Pa). Default 12 bar (Zhang et al.).
        P_purge: Blowdown/purge pressure (Pa). Default 1 atm.
        t_ads: Adsorption step time (s). Default 200 (Zhang et al. Table 4).
        purge_to_feed: Purge H2 / feed H2 per cycle (P/F of Yang & Lee
            1998). Default 0.11 (Zhang et al.).
        n_equalizations: Pressure equalizations (0 or 1). Default 1.
        ldf_scale: Multiplies :data:`LDF_OMEGA` (a fit knob).
    """

    bed_length: float = 1.68
    bed_diameter: float = 0.0384
    ac_fraction: float = 0.5
    n_beds_on_feed: int = 1
    T: float = 298.15
    P_ads: float = 12.0e5
    P_purge: float = ATM
    t_ads: float = 200.0
    purge_to_feed: float = 0.11
    n_equalizations: int = 1
    ldf_scale: float = 1.0


def _layers(p: HydrogenPSAParams):
    A = jnp.pi * p["bed_diameter"] ** 2 / 4.0
    out = []
    for name, frac in (("activated_carbon", p["ac_fraction"]),
                       ("zeolite_5a", 1.0 - p["ac_fraction"])):
        prop = ADSORBENT_PROPERTIES[name]
        V = A * p["bed_length"] * frac
        out.append((name, V, prop["rho_p"] * (1.0 - prop["eps_b"]), prop["eps_t"]))
    return out


def _inventory_h2(layers, y_ads: Array, P: Array, T: Array, swept: list) -> Array:
    """H2 (mol) in the bed at pressure ``P``: the swept part of each layer at
    the feed composition, the rest pure H2."""
    pure = jnp.zeros(len(ADSORBED)).at[0].set(1.0)
    tot = 0.0
    for (name, V, rho_b, eps_t), phi in zip(layers, swept):
        n_feed = eps_t * V * P * y_ads[0] / (R * T) + rho_b * V * dsl_loading(name, y_ads * P / ATM, T)[0]
        n_pure = eps_t * V * P / (R * T) + rho_b * V * dsl_loading(name, pure * P / ATM, T)[0]
        tot = tot + phi * n_feed + (1.0 - phi) * n_pure
    return tot


class HydrogenPSA:
    """A layered-bed H2 PSA (library). ``HydrogenPSA(params)(feed) -> dict``.

    ``feed`` is ``{species: mol/s}`` (difflow names). Returns ``product`` and
    ``tail_gas`` (``{species: mol/s}``), ``purity`` (H2 mole fraction of the
    product), ``recovery`` (product H2 / feed H2), ``gamma`` (purge/feed
    actual volume ratio; >= 1 is Skarstrom's condition), ``heel_time``
    (the purge heel per impurity, as seconds of feed), ``leak`` (fraction
    of each adsorbed impurity's feed that reaches product),
    ``breakthrough_time`` (s, stoichiometric, per impurity), ``swept``
    (fraction of each layer the fastest front swept) and ``balance``
    (feed - product - tail per species; zero to round-off).

    Example:
        >>> feed = {"hydrogen": 0.76e-2, "methane": 0.03e-2,
        ...         "carbon_monoxide": 0.04e-2, "carbon_dioxide": 0.17e-2}
        >>> out = HydrogenPSA(HydrogenPSAParams())(feed)
        >>> bool(0.5 < float(out["recovery"]) < 0.8)
        True
    """

    def __init__(self, params: HydrogenPSAParams | None = None):
        self.params = params if params is not None else HydrogenPSAParams()

    def __call__(self, feed: dict[str, Array]) -> dict:
        p = self.params
        names = list(feed)
        for n in names:
            if n not in ADSORBED and n not in TO_TAIL and n not in TO_PRODUCT:
                raise ValueError(f"PSA has no rule for species {n!r}: add an isotherm, or "
                                 "put it in TO_TAIL/TO_PRODUCT")
        F_ads = jnp.stack([jnp.asarray(feed.get(s, 0.0), dtype=float) for s in ADSORBED])
        F_ads = F_ads / p["n_beds_on_feed"]
        F_other = {n: jnp.asarray(feed[n], dtype=float) / p["n_beds_on_feed"]
                   for n in names if n not in ADSORBED}
        F_tot = jnp.sum(F_ads) + sum(F_other.values(), 0.0)
        # the gas the adsorbent sees: adsorbed species + product-going inerts
        # (strongly held species are taken off at the feed end)
        F_seen = jnp.sum(F_ads) + sum((F_other[n] for n in F_other if n in TO_PRODUCT), 0.0)
        y = F_ads / F_seen
        T, P_H, P_L, t = p["T"], p["P_ads"], p["P_purge"], p["t_ads"]
        layers = _layers(p)
        c_feed = y * P_H / (R * T)                     # mol/m3
        omega = jnp.asarray([LDF_OMEGA[s] for s in ADSORBED]) * p["ldf_scale"]

        # stoichiometric times (s) through the layers, per impurity
        t_solid = 0.0
        t_gas = 0.0
        q_layer = []
        for name, V, rho_b, eps_t in layers:
            q = dsl_loading(name, y * P_H / ATM, T)
            q_layer.append(q)
            t_solid = t_solid + rho_b * V * q / jnp.maximum(F_ads, 1e-30)
            t_gas = t_gas + eps_t * V * c_feed / jnp.maximum(F_ads, 1e-30)
        t_break = t_solid + t_gas

        # swept fraction of each layer on a clean bed: per impurity, the front
        # fills the AC layer in t_fill[0], then the 5A layer in t_fill[1]
        t_fill = []
        for (name, V, rho_b, eps_t), q in zip(layers, q_layer):
            cap = rho_b * V * q[1:] + eps_t * V * c_feed[1:]     # mol to fill the layer
            t_fill.append(jnp.maximum(cap / jnp.maximum(F_ads[1:], 1e-30), 1e-9))

        def swept_of(t_on):
            return [jnp.max(jnp.clip(t_on / t_fill[0], 0.0, 1.0)),
                    jnp.max(jnp.clip((t_on - t_fill[0]) / t_fill[1], 0.0, 1.0))]

        swept0 = swept_of(t)

        # purge: Skarstrom's volume ratio, and the heel left in the swept zone
        n_feed_cycle = F_tot * t
        n_purge = p["purge_to_feed"] * F_ads[0] * t
        V_purge = n_purge * R * T / P_L
        gamma = V_purge / (n_feed_cycle * R * T / P_H)
        heel = 0.0
        for (name, V, rho_b, eps_t), phi in zip(layers, swept0):
            Vs = jnp.maximum(phi * V, 1e-12)
            heel = heel + Vs * purge_heel(name, y, P_L, T, V_purge / Vs, eps_t, rho_b)
        t_heel = heel / jnp.maximum(F_ads, 1e-30)      # heel as feed time (s)
        t_front = t_solid - t_heel

        # breakthrough leak over the step (Klinkenberg, Gauss-Legendre)
        xg, wg = np.polynomial.legendre.leggauss(_N_QUAD)
        ts = 0.5 * t * (jnp.asarray(xg) + 1.0)
        tau = omega[:, None] * (ts[None, :] - t_gas[:, None])
        xi = omega * jnp.maximum(t_front, 1e-6 * t)
        c = klinkenberg(xi[:, None], tau)
        leak = 0.5 * jnp.sum(jnp.asarray(wg)[None, :] * c, axis=1)
        leak = leak.at[0].set(0.0)                     # H2 is the product, not a leak
        swept_ac, swept_5a = swept = swept_of(t + t_heel[1:])

        # H2 inventory and the tail-gas H2
        P_end = jnp.where(p["n_equalizations"] >= 1, 0.5 * (P_H + P_L), P_H)
        I_end = _inventory_h2(layers, y, P_end, T, swept)
        I_clean = _inventory_h2(layers, y, P_L, T, [0.0, 0.0])
        h2_tail_cycle = I_end - I_clean + n_purge
        h2_feed_cycle = F_ads[0] * t
        recovery = 1.0 - h2_tail_cycle / h2_feed_cycle

        nb = p["n_beds_on_feed"]
        prod = {"hydrogen": nb * F_ads[0] * recovery}
        tail = {"hydrogen": nb * F_ads[0] * (1.0 - recovery)}
        for i, s in enumerate(ADSORBED[1:], start=1):
            prod[s] = nb * F_ads[i] * leak[i]
            tail[s] = nb * F_ads[i] - prod[s]
        for n, f in F_other.items():
            to_prod = n in TO_PRODUCT
            prod[n] = nb * f if to_prod else 0.0 * f
            tail[n] = 0.0 * f if to_prod else nb * f
        tot_prod = sum(prod.values())
        purity = prod["hydrogen"] / tot_prod
        balance = {s: jnp.asarray(feed.get(s, 0.0), dtype=float) - prod[s] - tail[s] for s in prod}
        return {"product": prod, "tail_gas": tail, "purity": purity, "recovery": recovery,
                "gamma": gamma, "heel_time": dict(zip(ADSORBED[1:], t_heel[1:])), "leak": dict(zip(ADSORBED[1:], leak[1:])),
                "breakthrough_time": dict(zip(ADSORBED[1:], t_break[1:])),
                "swept": {"activated_carbon": swept_ac, "zeolite_5a": swept_5a},
                "balance": balance}


__all__ = ["ADSORBED", "DSL_PARAMETERS", "ADSORBENT_PROPERTIES", "LDF_OMEGA", "TO_TAIL",
           "TO_PRODUCT", "HEEL_SMOOTH", "purge_heel", "dsl_loading", "klinkenberg", "HydrogenPSAParams",
           "HydrogenPSA"]
