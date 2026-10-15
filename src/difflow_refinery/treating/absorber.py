"""Amine absorber (contactor): countercurrent stages with Murphree efficiencies.

Stages ``j = 1..N`` from the top; lean solvent enters stage 1, sour gas
stage ``N``. On every stage the gas leaving approaches the Kent-Eisenberg
equilibrium of the liquid leaving with a Murphree vapor efficiency
(Murphree 1925), species by species:

    ``y_j = y_{j+1} + E_i (y*_i(alpha_j) - y_{j+1})``,
    ``E_i = 1 - exp(-N_OG,i)``,   ``N_OG,i = K_OG,i a P / G``

(gas in plug flow across a well-mixed liquid; Lockett 1986, ch. 6), with
``a`` the interfacial area per stage, ``P`` the pressure and ``G`` the gas
flow. The overall coefficients are the two-film resistances in series:

* H2S reacts with any amine by proton transfer, instantaneous on the
  scale of diffusion, so its transfer is gas-film controlled:
  ``K_OG = k_G`` (Kohl & Nielsen 1997, ch. 2).
* CO2 must react with the amine (DEA: carbamate; MDEA: base-catalysed
  hydration, slow) and the liquid film resists:
  ``1/K_OG = 1/k_G + He/(E k_L)``, with the enhancement factor
  ``E = sqrt(1 + Ha^2)`` and Hatta number ``Ha = sqrt(k2 [Am] D)/k_L``
  (pseudo-first order in CO2; Danckwerts 1970, ch. 5; ``sqrt(1 + Ha^2)``
  joins the slow and fast regimes smoothly). ``[Am]`` is the FREE amine
  from the Kent-Eisenberg speciation of the stage liquid, so a loaded
  solution reacts more slowly.

This is what makes MDEA selective: its ``k2`` is two orders of magnitude
below DEA's, so CO2 slips through while H2S is held at the gas-film
rate. The absorber is isothermal (an assumption: the real column has a
temperature bulge from the heat of absorption) and the solvent's water is
held fixed through it; the sweet gas leaves saturated with water at the
column temperature and the water balance is closed at the outlets.

The unknowns per stage are ``ln alpha_CO2``, ``ln alpha_H2S`` (liquid
leaving) and the logs of the CO2 and H2S flows in the gas leaving; the
equations are the two liquid balances and the two Murphree relations.
Same Newton and implicit step as the regenerator.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from difflow.water import water_saturation_pressure
from difflow_cc.equilibrium import acid_gas as ag
from difflow_refinery.treating.properties import amine_properties, co2_diffusivity, co2_henry_solution

M_W = ag.M_WATER


@dataclass(frozen=True)
class TrayParams:
    """Mass-transfer parameters of one absorber stage.

    Attributes:
        area: Interfacial area per stage (m2).
        k_G: Gas-film coefficient (mol/(m2 s Pa)).
        k_L: Liquid-film coefficient (m/s).
    """

    area: float
    k_G: float
    k_L: float


def efficiencies(am, tray: TrayParams, T, P, G, m_free, w_water, rho):
    """Murphree vapor efficiencies ``(E_CO2, E_H2S)`` and the Hatta number.

    Args:
        am: Kent-Eisenberg amine.
        T, P: Temperature (K), pressure (Pa).
        G: Gas flow (mol/s).
        m_free: Free-amine molality (mol/kg water).
        w_water: Water mass fraction of the solution.
        rho: Solution density (kg/m3).
    """
    pr = amine_properties(am.name)
    c_am = m_free * w_water * rho                 # mol/m3
    D = co2_diffusivity(am.name, 1.0 - w_water, T)
    Ha = jnp.sqrt(pr.k2(T) * c_am * D) / tray.k_L
    E_enh = jnp.sqrt(1.0 + Ha**2)
    He = co2_henry_solution(am.name, 1.0 - w_water, T)           # Pa m3/mol
    K_C = 1.0 / (1.0 / tray.k_G + He / (E_enh * tray.k_L))
    N_C = K_C * tray.area * P / G
    N_S = tray.k_G * tray.area * P / G
    return 1.0 - jnp.exp(-N_C), 1.0 - jnp.exp(-N_S), Ha


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AbsorberResult:
    """What :func:`absorb` returns (stage arrays top first)."""

    sweet: dict          # gas flows by name (mol/s), T, P
    rich: dict           # L, W, CO2, H2S (mol/s), T; dissolved inerts in ``dissolved``
    dissolved: dict      # physically dissolved non-acid gases (mol/s)
    alpha_CO2: jax.Array
    alpha_H2S: jax.Array
    y_CO2: jax.Array
    y_H2S: jax.Array
    E_CO2: jax.Array
    E_H2S: jax.Array
    hatta: jax.Array
    converged: jax.Array
    residual: jax.Array


CO2, H2S, WATER = "carbon_dioxide", "hydrogen_sulfide", "water"


def absorb(amine, gas: dict, lean: dict, n_stages: int, tray: TrayParams, T=None,
           max_iter: int = 100) -> AbsorberResult:
    """Contact a sour gas with lean amine.

    Args:
        amine: ``"MDEA"``/``"DEA"`` or a ``KentEisenbergAmine``.
        gas: Sour gas stream: ``F_<name>`` (mol/s, refinery names, CO2 is
            ``carbon_dioxide``, H2S ``hydrogen_sulfide``), ``T``, ``P``.
        lean: ``{"L", "W", "CO2", "H2S", "T"}`` lean solvent (mol/s, K).
        n_stages: Number of actual stages (trays).
        tray: :class:`TrayParams`.
        T: Column temperature (K); default the lean solvent's.

    Returns:
        :class:`AbsorberResult`.
    """
    from difflow_refinery.treating.properties import dissolved_gas, solution_density

    am = ag.ke_amine(amine)
    N = int(n_stages)
    T = lean["T"] if T is None else T
    names = [k[2:] for k in gas if k.startswith("F_")]
    inert = [n for n in names if n not in (CO2, H2S, WATER)]
    theta = dict(L=lean["L"], W=lean["W"], aC0=lean["CO2"] / lean["L"],
                 aS0=lean["H2S"] / lean["L"], T=T, P=gas["P"],
                 gC=gas.get(f"F_{CO2}", 0.0), gS=gas.get(f"F_{H2S}", 0.0),
                 G0=sum(gas[f"F_{n}"] for n in inert))
    theta = {k: jnp.asarray(v, dtype=float) for k, v in theta.items()}
    # an absent acid gas is held at a trace (its log must stay finite); the
    # trace is taken back out of BOTH outlets in proportion (below), so the
    # balances are exact and no outlet goes negative (#475: a CO2-free lean
    # does not absorb the trace, so taking it from the rich alone went negative)
    trace = {}
    for k in ("gC", "gS"):
        floored = jnp.maximum(theta[k], 1e-12 * theta["G0"])
        trace[k] = floored - theta[k]
        theta[k] = floored
    floor = 1e-12

    def props(th):
        m = th["L"] / (th["W"] * M_W)
        mass_w = th["W"] * M_W
        w_water = mass_w / (mass_w + th["L"] * am.MW * 1e-3)
        rho = solution_density(am.name, 1.0 - w_water)
        return m, w_water, rho

    def evaluate(z, th):
        z = z.reshape(N, 4)
        aC, aS, gC, gS = (jnp.exp(z[:, i]) for i in range(4))
        m, w_water, rho = props(th)
        sp = jax.vmap(lambda c, s: ag.ke_speciate(am, th["T"], m, c, s))(aC, aS)
        G = th["G0"] + gC + gS
        E_C, E_S, Ha = jax.vmap(
            lambda g, mf: efficiencies(am, tray, th["T"], th["P"], g, mf, w_water, rho))(
                G, sp["species"]["Am"])
        return dict(aC=aC, aS=aS, gC=gC, gS=gS, G=G, pC=sp["p_CO2"], pS=sp["p_H2S"],
                    E_C=E_C, E_S=E_S, Ha=Ha, ok=sp["converged"])

    def res(z, th):
        s = evaluate(z, th)
        L = th["L"]
        # liquid from above (stage 1: lean), gas from below (stage N: feed)
        aC_in = jnp.concatenate([th["aC0"][None], s["aC"][:-1]])
        aS_in = jnp.concatenate([th["aS0"][None], s["aS"][:-1]])
        gC_in = jnp.concatenate([s["gC"][1:], th["gC"][None]])
        gS_in = jnp.concatenate([s["gS"][1:], th["gS"][None]])
        G_in = th["G0"] + gC_in + gS_in
        acid = th["gC"] + th["gS"] + 1e-6 * th["G0"]
        rC = (L * aC_in + gC_in - L * s["aC"] - s["gC"]) / acid
        rS = (L * aS_in + gS_in - L * s["aS"] - s["gS"]) / acid
        P = th["P"]

        def murphree(g, G, y_in, ystar, E):
            y_t = y_in + E * (ystar - y_in)
            return jnp.log(jnp.maximum(g / G, floor * 1e-6)) - jnp.log(jnp.maximum(y_t, floor * 1e-6))

        mC = murphree(s["gC"], s["G"], gC_in / G_in, s["pC"] / P, s["E_C"])
        mS = murphree(s["gS"], s["G"], gS_in / G_in, s["pS"] / P, s["E_S"])
        return jnp.stack([rC, rS, mC, mS], axis=1).ravel()

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    z0 = _guess(th0, N)
    z_star, k, err = ag._newton(lambda z: res(z, th0), z0, cap=2.0, max_iter=max_iter, tol=1e-11)
    z = ag._implicit(res, z_star, theta)
    s = evaluate(z, theta)
    L, W, P = theta["L"], theta["W"], theta["P"]
    Tt = theta["T"]

    # water: the sweet gas leaves saturated at T (Raoult on the solvent),
    # taking it from the solvent; the sour gas's water joins the solvent
    xw = W / (W + L)
    yw = xw * water_saturation_pressure(Tt) / P
    dry = s["G"][0]
    w_sweet = dry * yw / (1.0 - yw)
    # physically dissolved inerts, at the bottom-stage partial pressure
    m, w_water, rho = props(theta)
    vol = (W * M_W + L * am.MW * 1e-3) / rho      # m3/s of solvent
    G_bot = theta["G0"] + theta["gC"] + theta["gS"]
    dissolved = {n: jnp.minimum(dissolved_gas(n, Tt) * P * gas[f"F_{n}"] / G_bot * vol,
                                0.5 * gas[f"F_{n}"]) for n in inert}
    sweet = {f"F_{n}": gas[f"F_{n}"] - dissolved[n] for n in inert}
    # what came in over what was solved for: at a trace the acid gas is in
    # the Henry's-law limit (linear), so this scaling is also superposition
    keep = {}
    for k, a0 in (("gC", "aC0"), ("gS", "aS0")):
        fed = L * theta[a0] + theta[k] - trace[k]
        added = trace[k] > 0
        keep[k] = jnp.where(added, fed / jnp.where(added, fed + trace[k], 1.0), 1.0)
    sweet[f"F_{CO2}"] = s["gC"][0] * keep["gC"]
    sweet[f"F_{H2S}"] = s["gS"][0] * keep["gS"]
    sweet[f"F_{WATER}"] = w_sweet
    sweet.update(T=Tt, P=P)
    rich = dict(L=L, W=W - w_sweet + jnp.asarray(gas.get(f"F_{WATER}", 0.0), dtype=float),
                CO2=L * s["aC"][-1] * keep["gC"], H2S=L * s["aS"][-1] * keep["gS"], T=Tt)
    return AbsorberResult(sweet=sweet, rich=rich, dissolved=dissolved, alpha_CO2=s["aC"],
                          alpha_H2S=s["aS"], y_CO2=s["gC"] / s["G"], y_H2S=s["gS"] / s["G"],
                          E_CO2=s["E_C"], E_H2S=s["E_S"], hatta=s["Ha"],
                          converged=(err < 1e-8) & jnp.all(s["ok"]), residual=err)


def _guess(th, N):
    """Half of each acid gas absorbed, spread evenly down the column."""
    L = th["L"]
    frac = jnp.linspace(0.05, 1.0, N)
    aC = th["aC0"] + 0.5 * th["gC"] / L * frac + 1e-9
    aS = th["aS0"] + 0.9 * th["gS"] / L * frac + 1e-9
    gC = jnp.maximum(th["gC"] * (1.0 - 0.5 * frac[::-1] * 0 - 0.5 * (1.0 - frac)), 1e-9 * th["G0"])
    gS = jnp.maximum(th["gS"] * jnp.geomspace(1e-3, 1.0, N), 1e-9 * th["G0"])
    return jnp.stack([jnp.log(aC), jnp.log(aS), jnp.log(gC), jnp.log(gS)], axis=1).ravel()
