"""Modified-Claus sulfur recovery: furnace, waste-heat boiler, condensers,
catalytic converters and air-demand control.

The train (``claus_train``) is::

    acid gas + air -> furnace (adiabatic Gibbs) -> WHB -> condenser 1
        -> [reheat -> converter (adiabatic, Claus + allotropes) -> condenser] x n
        -> tail gas

* Furnace: adiabatic equilibrium over every species of :data:`SPECIES`
  (:class:`difflow.GibbsReactor`), with an optional ``approach_dT``. Real
  furnaces fall short of equilibrium in H2, COS and CS2; nothing here
  corrects that unless the caller does.
* WHB and condensers (:func:`cool`): the non-sulfur gas is FROZEN (the
  furnace composition is quenched, as in practice); the sulfur re-speciates
  among S2...S8 and condenses to saturation (:mod:`.vapor`). The duty raises
  steam.
* Converters (:func:`converter`): reheat to the bed inlet T, a specified COS
  and CS2 hydrolysis (bed 1 by default), then adiabatic equilibrium
  restricted to the Claus reaction ``4 H2S + 2 SO2 = 4 H2O + 3 S2`` and the
  allotrope reactions, with an approach to equilibrium on the Claus reaction.
* Air demand (:func:`solve_air`): the air flow at which the tail gas has
  ``H2S - 2 SO2 = 0``, solved by secant and returned with its implicit
  derivative.

Every heat is an enthalpy difference on one basis (JANAF formation basis,
:mod:`difflow_refinery.thermochemistry`); liquid sulfur is JANAF S-004.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jax import lax

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from difflow.water import water_liquid_enthalpy, water_vapor_enthalpy
from difflow_refinery import thermochemistry as tc
from difflow_refinery.sulfur import vapor
from difflow_refinery.sulfur.vapor import SIZES, SULFUR_SPECIES

H2S, SO2, H2O = "hydrogen_sulfide", "sulfur_dioxide", "water"
COS, CS2, CO2 = "carbonyl_sulfide", "carbon_disulfide", "carbon_dioxide"

#: species of the train (every stream carries all of them)
SPECIES = (H2S, SO2, *SULFUR_SPECIES, "hydrogen", "carbon_monoxide", CO2, COS, CS2, H2O,
           "nitrogen", "oxygen", "ammonia", "methane")
GAS = tc.IdealGasSet(list(SPECIES), cp_basis="wide")
_IDX = {n: i for i, n in enumerate(SPECIES)}
_S_IDX = jnp.array([_IDX[n] for n in SULFUR_SPECIES])
#: S atoms per molecule, in :data:`SPECIES` order
S_ATOMS = jnp.array([tc.elements(n).get("S", 0) for n in SPECIES], dtype=float)

#: dry air, mole fractions: O2 20.946%, the rest (N2 78.084, Ar 0.934,
#: CO2 0.04) lumped as N2 (U.S. Standard Atmosphere 1976, NOAA-S/T 76-1562,
#: Table 3; NASA-TM-X-74335)
AIR_O2 = 0.20946

#: Claus reaction (to S2) and the allotrope reactions S2 -> Sn
CLAUS = {H2S: -4.0, SO2: -2.0, H2O: 4.0, "sulfur_s2": 3.0}
ALLOTROPES = [{"sulfur_s2": -n / 2.0, f"sulfur_s{n}": 1.0} for n in range(3, 9)]


def to_vec(stream) -> jax.Array:
    """Molar flows in :data:`SPECIES` order (missing species are 0)."""
    return jnp.stack([jnp.asarray(stream.get(f"F_{n}", 0.0), dtype=float) for n in SPECIES])


def to_stream(n, T, P):
    """A stream over :data:`SPECIES` from flows ``n``."""
    return make_stream(dict(zip(SPECIES, n)), T, P)


def enthalpy(n, T):
    """Total enthalpy (W) of gas flows ``n`` at ``T``."""
    return jnp.sum(n * GAS.enthalpy(T))


def sulfur_in(n):
    """S atoms (mol/s) in gas flows ``n``."""
    return jnp.sum(S_ATOMS * n)


def steam_make(duty, T_steam, T_bfw):
    """Saturated steam (kg/s) at ``T_steam`` raised from boiler feedwater at
    ``T_bfw`` by ``duty`` W (difflow.water steam-table fits, 20-200 C)."""
    return duty / (water_vapor_enthalpy(T_steam) - water_liquid_enthalpy(T_bfw))


# ---------------------------------------------------------------- units
def cool(stream, T):
    """Cool a sulfur-bearing gas to ``T``: non-sulfur gas frozen, sulfur
    re-speciated and condensed to saturation.

    Returns:
        ``(gas, liquid_S, duty)``: the gas stream, liquid sulfur (mol S
        atoms/s) and the heat REMOVED (W, positive when cooling).
    """
    n = to_vec(stream)
    T_in, P = jnp.asarray(stream["T"], dtype=float), jnp.asarray(stream["P"], dtype=float)
    T = jnp.asarray(T, dtype=float)
    s_elem = jnp.sum(SIZES * n[_S_IDX])
    G = jnp.sum(n) - jnp.sum(n[_S_IDX])
    vap, liq = vapor.condense(s_elem, G, T, P)
    n_out = n.at[_S_IDX].set(vap)
    duty = enthalpy(n, T_in) - enthalpy(n_out, T) - liq * tc.liquid_sulfur_enthalpy(T)
    return to_stream(n_out, T, P), liq, duty


@dataclass(frozen=True)
class ConverterParams:
    """One catalytic converter.

    Attributes:
        T_in: Bed inlet temperature after reheat (K).
        hydrolysis: ``(X_COS, X_CS2)`` fractional hydrolysis of COS and CS2
            to H2S over the bed (titania or alumina in bed 1; ~0 downstream).
        approach_dT: Approach to Claus equilibrium (K): the Claus reaction's
            equilibrium is taken at ``T + approach_dT``, the allotropes' at
            ``T``. The reaction is written to S2, which is endothermic
            (about +96 kJ per 4 H2S at 500 K), so a NEGATIVE value falls
            short of equilibrium and a positive one overshoots it.
        pressure_drop: Bed pressure drop (Pa).
    """

    T_in: float
    hydrolysis: tuple = (0.0, 0.0)
    approach_dT: float = 0.0
    pressure_drop: float = 0.0


_BED_REACTIONS = [CLAUS, *ALLOTROPES]


def converter(stream, cp: ConverterParams):
    """Reheat, hydrolyse COS/CS2, and equilibrate one Claus bed adiabatically.

    Returns:
        ``(outlet, info)``; ``info`` has ``reheat`` (W), ``T_out``,
        ``converged``, and ``dew_margin`` (K; bed outlet T minus its sulfur
        dew point, which must stay positive: sulfur condensing in the bed
        deactivates it).
    """
    n = to_vec(stream)
    T0, P = jnp.asarray(stream["T"], dtype=float), jnp.asarray(stream["P"], dtype=float)
    T_in = jnp.asarray(cp.T_in, dtype=float)
    reheat = enthalpy(n, T_in) - enthalpy(n, T0)
    # COS + H2O -> CO2 + H2S ; CS2 + 2 H2O -> CO2 + 2 H2S
    x_cos, x_cs2 = cp.hydrolysis
    r1, r2 = x_cos * n[_IDX[COS]], x_cs2 * n[_IDX[CS2]]
    d = jnp.zeros_like(n)
    d = d.at[_IDX[COS]].add(-r1).at[_IDX[CS2]].add(-r2)
    d = d.at[_IDX[H2O]].add(-r1 - 2 * r2).at[_IDX[CO2]].add(r1 + r2)
    d = d.at[_IDX[H2S]].add(r1 + 2 * r2)
    n_h = n + d
    # the hydrolysis heat is released into the bed (adiabatic overall)
    q = enthalpy(n, T_in) - enthalpy(n_h, T_in)
    out, info = GibbsReactor(GibbsReactorParams(
        thermo=GAS, mode="adiabatic", reactions=_BED_REACTIONS, heat_duty=q,
        reaction_approach_dT=[cp.approach_dT] + [0.0] * len(ALLOTROPES),
        pressure_drop=cp.pressure_drop))(to_stream(n_h, T_in, P))
    n_out = to_vec(out)
    s_elem = jnp.sum(SIZES * n_out[_S_IDX])
    G = jnp.sum(n_out) - jnp.sum(n_out[_S_IDX])
    Tdew = vapor.dew_point(s_elem, G, out["P"])
    return out, {"reheat": reheat, "T_out": info["T"], "converged": info["converged"],
                 "dew_point": Tdew, "dew_margin": info["T"] - Tdew}



def furnace(acid_gas, air, T_air=None, approach_dT: float = 0.0, pressure_drop: float = 0.0):
    """Reaction furnace: acid gas burnt with ``air`` mol/s of dry air,
    adiabatic equilibrium over :data:`SPECIES`.

    Args:
        acid_gas: Feed stream (``F_<name>``, ``T``, ``P``).
        air: Dry air (mol/s).
        T_air: Air temperature (K); default the acid gas's.

    Returns:
        ``(outlet, info)`` of the :class:`difflow.GibbsReactor`.
    """
    n = to_vec(acid_gas)
    T = jnp.asarray(acid_gas["T"], dtype=float)
    a = jnp.zeros_like(n).at[_IDX["oxygen"]].set(AIR_O2 * air)
    a = a.at[_IDX["nitrogen"]].set((1.0 - AIR_O2) * air)
    # air at its own temperature: its sensible heat enters as a duty
    q = 0.0 if T_air is None else enthalpy(a, T_air) - enthalpy(a, T)
    return GibbsReactor(GibbsReactorParams(
        thermo=GAS, mode="adiabatic", heat_duty=q, approach_dT=approach_dT,
        pressure_drop=pressure_drop))(to_stream(n + a, T, acid_gas["P"]))


def oxygen_demand(acid_gas):
    """Stoichiometric O2 (mol/s) for the Claus furnace: a third of the H2S
    to SO2 (0.5 O2 per H2S overall), hydrocarbons, NH3, H2 and CO burnt out."""
    n = to_vec(acid_gas)
    g = {k: n[_IDX[k]] for k in SPECIES}
    return (0.5 * g[H2S] + 2.0 * g["methane"] + 0.75 * g["ammonia"]
            + 0.5 * g["hydrogen"] + 0.5 * g["carbon_monoxide"])


@dataclass(frozen=True)
class ClausParams:
    """A Claus train.

    Attributes:
        converters: One :class:`ConverterParams` per bed.
        T_whb: Waste-heat-boiler outlet (K).
        T_condensers: Outlet T (K) of each condenser: after the WHB, then
            after each bed (``len(converters) + 1`` values).
        T_air: Combustion air temperature (K); None = the acid gas's.
        furnace_approach_dT: Furnace approach to equilibrium (K).
        furnace_dP: Furnace + WHB pressure drop (Pa).
        condenser_dP: Pressure drop per condenser (Pa).
        T_steam_whb, T_steam_cond, T_bfw: Steam saturation T at the WHB and
            the condensers, and the boiler-feedwater T (K), for steam make.
    """

    converters: tuple = field(default_factory=tuple)
    T_whb: float = 589.15
    T_condensers: tuple = ()
    T_air: float | None = None
    furnace_approach_dT: float = 0.0
    furnace_dP: float = 0.0
    condenser_dP: float = 0.0
    T_steam_whb: float = 473.15
    T_steam_cond: float = 418.15
    T_bfw: float = 378.15

    def __post_init__(self):
        if len(self.T_condensers) != len(self.converters) + 1:
            raise ValueError("need one condenser T after the WHB and one per converter")


def claus_train(acid_gas, air, p: ClausParams):
    """Run the train at a given air flow.

    Returns:
        dict: ``tail_gas`` (stream), ``sulfur`` (liquid S, mol atoms/s, total),
        ``sulfur_by_condenser``, ``recovery`` (liquid S / S in), ``S_in``,
        ``ratio_residual`` (tail ``(H2S - 2 SO2)/S_in``), ``T_furnace``,
        ``whb_duty``, ``condenser_duty``, ``reheat``, ``steam_whb``,
        ``steam_condensers`` (kg/s), ``bed`` (per-bed info), ``converged``.
    """
    S_in = sulfur_in(to_vec(acid_gas))
    fout, finfo = furnace(acid_gas, air, p.T_air, p.furnace_approach_dT, p.furnace_dP)
    g, liq0, q_whb = cool(fout, p.T_whb)
    liquids, duties, reheats, beds = [liq0], [], [], []
    conv = finfo["converged"]
    for k, T_c in enumerate(p.T_condensers):
        if k > 0:
            g, info = converter(g, p.converters[k - 1])
            reheats.append(info["reheat"])
            beds.append(info)
            conv = conv & info["converged"]
        g = dict(g, P=g["P"] - p.condenser_dP)
        g, liq, q = cool(g, T_c)
        liquids.append(liq)
        duties.append(q)
    n_t = to_vec(g)
    liquids = jnp.stack(liquids)
    duties = jnp.stack(duties)
    sulfur = jnp.sum(liquids)
    return {
        "tail_gas": g, "sulfur": sulfur, "sulfur_by_condenser": liquids,
        "recovery": sulfur / S_in, "S_in": S_in,
        "ratio_residual": (n_t[_IDX[H2S]] - 2.0 * n_t[_IDX[SO2]]) / S_in,
        "T_furnace": finfo["T"], "furnace": fout, "whb_duty": q_whb,
        "condenser_duty": duties, "reheat": jnp.stack(reheats) if reheats else jnp.zeros(0),
        "steam_whb": steam_make(q_whb, p.T_steam_whb, p.T_bfw),
        "steam_condensers": steam_make(jnp.sum(duties), p.T_steam_cond, p.T_bfw),
        "bed": beds, "converged": conv,
    }


def solve_air(acid_gas, p: ClausParams, tol: float = 1e-12, max_iter: int = 30):
    """Air (mol/s) at which the tail gas has ``H2S = 2 SO2`` (air-demand control).

    Secant on stop-gradient inputs from the stoichiometric air; the train is
    evaluated once per pass inside the loop, so it is traced once there. The
    result is then taken at the root by :func:`difflow_refinery.implicit.solution`
    (one more trace, or one linearisation when differentiated): the value is
    the root and its derivative the implicit one. The solve is jitted, cached
    by the train's structure, when ``p`` holds no arrays.

    Returns:
        ``(air, result)``: the air and :func:`claus_train` at it, plus
        ``air_converged`` and ``air_iterations`` in ``result``.
    """
    from difflow_refinery.implicit import hashable, split_params

    skel, arrays = split_params(p)
    acid = {k: jnp.asarray(v, dtype=float) for k, v in dict(acid_gas).items()}
    if hashable(skel):
        return _solve_air_jit(acid, arrays, skel, tol, max_iter)
    return _solve_air(acid, arrays, skel, tol, max_iter)


def _solve_air(acid, arrays, skel, tol, max_iter):
    from difflow_refinery.implicit import merge_params, solution

    theta = (acid, arrays)

    def G(a, th):
        out = claus_train(th[0], a[0], merge_params(skel, th[1]))
        return out["ratio_residual"][None], out

    th0 = jax.tree_util.tree_map(lax.stop_gradient, theta)
    a0 = oxygen_demand(th0[0]) / AIR_O2

    # secant, with the first two passes evaluating a0 and 1.01 a0: one
    # evaluation of the train per pass, so one trace of it
    def cond(st):
        _, _, _, r_cur, _, k = st
        return (k < 2) | ((jnp.abs(r_cur) > tol) & (k - 2 < max_iter))

    def body(st):
        a_prev, a_cur, r_prev, r_cur, a_ev, k = st
        r = G(a_ev[None], th0)[0][0]
        first = k == 0
        a_prev, r_prev = jnp.where(first, a_ev, a_cur), jnp.where(first, r, r_cur)
        a_cur, r_cur = a_ev, r
        a_sec = a_cur - r_cur * (a_cur - a_prev) / (r_cur - r_prev)
        a_next = jnp.where(first, 1.01 * a_ev, jnp.clip(a_sec, 0.5 * a_cur, 2.0 * a_cur))
        return a_prev, a_cur, r_prev, r_cur, a_next, k + 1

    st0 = (a0, a0, jnp.zeros(()), jnp.asarray(jnp.inf), a0, 0)
    _, a_star, _, r_star, _, k = lax.while_loop(cond, body, st0)
    a, out = solution(G, a_star[None], theta, fixed_point=False)
    out = dict(out)
    out["air_converged"] = jnp.abs(r_star) <= tol * 10
    out["air_iterations"] = k - 2
    return a[0], out


_solve_air_jit = jax.jit(_solve_air, static_argnums=(2, 3, 4))
