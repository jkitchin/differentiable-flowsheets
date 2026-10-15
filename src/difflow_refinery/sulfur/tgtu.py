"""SCOT-type tail-gas treating and the closed SRU (Claus + TGTU + recycle).

Claus tail gas -> reheat -> hydrogenation/hydrolysis reactor -> quench
-> MDEA absorber -> off-gas to the incinerator; the absorbed acid gas is
stripped and recycled to the furnace (the second tear of the SRU).

* Hydrogenation (:func:`hydrogenate`): adiabatic equilibrium restricted to
  ``SO2 + 3 H2 = H2S + 2 H2O``, ``S2 + 2 H2 = 2 H2S`` (and the allotropes),
  ``COS + H2O = CO2 + H2S``, ``CS2 + 2 H2O = CO2 + 2 H2S`` and the shift
  ``CO + H2O = CO2 + H2``. At 500-600 K the first two run to completion;
  COS leaves at its hydrolysis equilibrium.
* Quench (:func:`quench`): cooled to ``T`` with the water condensing to
  saturation (IAPWS-IF97, :func:`difflow.water.water_saturation_pressure`);
  H2S and CO2 dissolved in the sour water are neglected.
* Absorber: :func:`difflow_refinery.treating.absorb` (Kent-Eisenberg MDEA,
  Murphree trays with the CO2 kinetic slip that makes MDEA selective).
* Regenerator: NOT solved here. The solvent is taken back to its lean
  loading, so the recycled acid gas is the absorbed H2S and CO2, saturated
  with water at the reflux drum (``T_reflux``, ``P_reflux``). The full
  stripper is :func:`difflow_refinery.treating.regenerate`.
* Incinerator (:func:`incinerate`): every S species of the off-gas to SO2.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax

from difflow import GibbsReactor, GibbsReactorParams
from difflow.water import water_saturation_pressure
from difflow_refinery.sulfur import claus as cl
from difflow_refinery.sulfur.claus import (CO2, COS, CS2, GAS, H2O, H2S, SO2, SPECIES,
                                           enthalpy, to_stream, to_vec)
from difflow_refinery.treating import TrayParams, absorb, lean_water

_IDX = cl._IDX
#: molar mass of SO2 (kg/mol), from the IUPAC standard atomic weights
#: (:data:`difflow_refinery.thermochemistry.ATOMIC_WEIGHT`)
M_SO2 = (32.06 + 2 * 15.999) * 1e-3

HYDROGENATION = [
    {SO2: -1.0, "hydrogen": -3.0, H2S: 1.0, H2O: 2.0},
    {"sulfur_s2": -1.0, "hydrogen": -2.0, H2S: 2.0},
    {COS: -1.0, H2O: -1.0, CO2: 1.0, H2S: 1.0},
    {CS2: -1.0, H2O: -2.0, CO2: 1.0, H2S: 2.0},
    {"carbon_monoxide": -1.0, H2O: -1.0, CO2: 1.0, "hydrogen": 1.0},
    *cl.ALLOTROPES,
]


def hydrogenate(tail_gas, T_in, h2_makeup=0.0):
    """Reheat the Claus tail gas to ``T_in`` (with ``h2_makeup`` mol/s of H2
    added) and equilibrate the hydrogenation/hydrolysis reactor adiabatically.

    Returns:
        ``(outlet, info)``; ``info`` adds ``reheat`` (W) and ``H2_excess``
        (H2 left, mol/s; must stay positive or SO2 breaks through).
    """
    n = to_vec(tail_gas).at[_IDX["hydrogen"]].add(h2_makeup)
    T0, P = jnp.asarray(tail_gas["T"], dtype=float), jnp.asarray(tail_gas["P"], dtype=float)
    reheat = enthalpy(n, T_in) - enthalpy(n, T0)
    out, info = GibbsReactor(GibbsReactorParams(
        thermo=GAS, mode="adiabatic", reactions=HYDROGENATION))(to_stream(n, T_in, P))
    return out, dict(info, reheat=reheat, H2_excess=out["F_hydrogen"])


def quench(stream, T):
    """Cool to ``T``, condensing water to saturation.

    Returns:
        ``(gas, water_condensed, duty_removed)``.
    """
    n = to_vec(stream)
    P = jnp.asarray(stream["P"], dtype=float)
    y = water_saturation_pressure(T) / P
    dry = jnp.sum(n) - n[_IDX[H2O]]
    w_sat = dry * y / (1.0 - y)
    w = jnp.minimum(n[_IDX[H2O]], w_sat)
    cond = n[_IDX[H2O]] - w
    n_out = n.at[_IDX[H2O]].set(w)
    # condensate leaves as liquid at T: gas enthalpy less the latent heat,
    # from the difflow.water steam-table fit (kg -> mol)
    from difflow.water import water_latent_heat
    h_l = GAS.enthalpy(T)[_IDX[H2O]] - water_latent_heat(T) * 18.015e-3
    duty = enthalpy(n, stream["T"]) - enthalpy(n_out, T) - cond * h_l
    return to_stream(n_out, T, P), cond, duty


def incinerate(off_gas):
    """S emitted as SO2 (mol/s) when every S species of ``off_gas`` is burnt."""
    return cl.sulfur_in(to_vec(off_gas))


@dataclass(frozen=True)
class TGTUParams:
    """Tail-gas treating unit.

    Attributes:
        T_hydrogenation: Reactor inlet T after reheat (K).
        h2_makeup: External H2 (mol/s).
        T_quench: Quench-column outlet T (K).
        amine: ``"MDEA"`` (selective) or ``"DEA"``.
        w_amine: Amine mass fraction.
        L: Amine circulation (mol amine/s).
        lean_H2S, lean_CO2: Lean loadings (mol/mol amine).
        n_stages: Absorber trays.
        tray: :class:`difflow_refinery.treating.TrayParams`.
        T_reflux, P_reflux: Regenerator reflux drum (K, Pa): the recycled
            acid gas leaves it saturated with water.
        P_absorber: Absorber pressure (Pa); None keeps the quench outlet's.
    """

    tray: TrayParams
    L: float
    T_hydrogenation: float = 553.15
    h2_makeup: float = 0.0
    T_quench: float = 313.15
    amine: str = "MDEA"
    w_amine: float = 0.40
    lean_H2S: float = 0.002
    lean_CO2: float = 0.0005
    n_stages: int = 10
    T_reflux: float = 323.15
    P_reflux: float = 1.9e5
    P_absorber: float | None = None


def tgtu(tail_gas, p: TGTUParams):
    """Treat a Claus tail gas.

    Returns:
        dict: ``off_gas`` (to the incinerator), ``recycle`` (acid gas stream
        to the furnace), ``SO2_emitted`` (mol/s), ``hydrogenation``,
        ``quench_water``, ``absorber`` (:class:`AbsorberResult`), ``converged``.
    """
    h, hinfo = hydrogenate(tail_gas, p.T_hydrogenation, p.h2_makeup)
    q, cond, q_duty = quench(h, p.T_quench)
    if p.P_absorber is not None:
        q = dict(q, P=p.P_absorber)
    lean = dict(L=p.L, W=lean_water(p.amine, p.L, p.w_amine), CO2=p.L * p.lean_CO2,
                H2S=p.L * p.lean_H2S, T=p.T_quench)
    n = to_vec(q)
    gas = {f"F_{k}": n[_IDX[k]] for k in SPECIES}
    gas.update(T=q["T"], P=q["P"])
    ab = absorb(p.amine, gas, lean, p.n_stages, p.tray)
    off = dict(ab.sweet)
    # absorbed acid gas = rich - lean; leaves the reflux drum water-saturated
    r_h2s = ab.rich["H2S"] - lean["H2S"]
    r_co2 = ab.rich["CO2"] - lean["CO2"]
    y = water_saturation_pressure(p.T_reflux) / p.P_reflux
    r_w = (r_h2s + r_co2) * y / (1.0 - y)
    recycle = cl.to_stream(jnp.zeros(len(SPECIES)).at[_IDX[H2S]].set(r_h2s)
                           .at[_IDX[CO2]].set(r_co2).at[_IDX[H2O]].set(r_w),
                           p.T_reflux, p.P_reflux)
    return {"off_gas": off, "recycle": recycle, "SO2_emitted": incinerate(off),
            "hydrogenation": hinfo, "hydrogenated": h, "quench_water": cond,
            "quench_duty": q_duty, "absorber": ab,
            "converged": hinfo["converged"] & ab.converged}


_TEAR = (H2S, CO2, H2O)


def solve_sru(acid_gas, claus_params, tgtu_params, tol: float = 1e-10, max_iter: int = 50):
    """The closed sulfur recovery unit: Claus train under air-demand control,
    TGTU, and the TGTU acid gas recycled to the furnace.

    The tear is the recycle's (H2S, CO2, H2O). It is converged by direct
    substitution on stop-gradient inputs (the loop gain is the fraction of
    the furnace S and CO2 that comes back, a few percent), then ONE Newton
    step ``x - (I - J)^-1 (x - F(x))`` with ``J`` held fixed: the value is
    the fixed point and its derivative the implicit one.

    Returns:
        dict: ``claus`` (:func:`difflow_refinery.sulfur.claus.solve_air`
        result), ``tgtu`` (:func:`tgtu`), ``air``, ``recycle`` (tear, mol/s),
        ``recovery`` (1 - S emitted / S in), ``SO2_emitted`` (mol/s and
        ``SO2_emitted_kg_h``), ``tear_iterations``, ``converged``.
    """
    from difflow_refinery.sulfur.claus import solve_air

    def furnace_feed(x, acid):
        g = dict(acid)
        for k, v in zip(_TEAR, x):
            g[f"F_{k}"] = jnp.asarray(acid.get(f"F_{k}", 0.0), dtype=float) + v
        # the recycle enters at the acid gas's T (its sensible heat is small)
        return g

    def plant(x, acid):
        air, cr = solve_air(furnace_feed(x, acid), claus_params)
        tr = tgtu(cr["tail_gas"], tgtu_params)
        x_new = jnp.stack([tr["recycle"][f"F_{k}"] for k in _TEAR])
        return x_new, (air, cr, tr)

    acid0 = jax.tree_util.tree_map(lax.stop_gradient, dict(acid_gas))

    def F(x, acid):
        return plant(x, acid)[0]

    def cond(st):
        x, dx, k = st
        return (dx > tol) & (k < max_iter)

    def body(st):
        x, _, k = st
        xn = F(x, acid0)
        scale = jnp.maximum(jnp.sum(jnp.abs(xn)), 1e-30)
        return xn, jnp.sum(jnp.abs(xn - x)) / scale, k + 1

    x0 = jnp.zeros(len(_TEAR))
    x_star, dx, k = lax.while_loop(cond, body, (x0, jnp.inf, 0))
    x_star = lax.stop_gradient(x_star)
    J = lax.stop_gradient(jax.jacfwd(F)(x_star, acid0))
    x = x_star - jnp.linalg.solve(jnp.eye(len(_TEAR)) - J, x_star - F(x_star, acid_gas))
    x_out, (air, cr, tr) = plant(x, acid_gas)
    S_in = cl.sulfur_in(to_vec(acid_gas))
    return {"claus": cr, "tgtu": tr, "air": air, "recycle": x,
            "recovery": 1.0 - tr["SO2_emitted"] / S_in,
            "SO2_emitted": tr["SO2_emitted"],
            "SO2_emitted_kg_h": tr["SO2_emitted"] * M_SO2 * 3600.0,
            "tear_residual": jnp.sum(jnp.abs(x_out - x)), "tear_iterations": k,
            "gain": jnp.max(jnp.abs(jnp.linalg.eigvals(J))),
            "converged": (dx <= tol) & cr["converged"] & cr["air_converged"] & tr["converged"]}
