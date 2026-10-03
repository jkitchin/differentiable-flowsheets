"""An equation-oriented MESH model of a steam-stripped crude column, in Pyomo.

Written from the physics -- Material balance, Equilibrium, Summation and
entHalpy balance on every stage -- and solved simultaneously by IPOPT. It
calls nothing in difflow: the inputs are the component table
(:func:`case.component_data`), a layout and the specs, all plain numbers.

The formulation is deliberately not difflow's. difflow's unknowns are
log component liquid flows, stage temperatures and log total vapour flows,
with draws as fractions of the stage liquid, closed by a damped Newton.
Here they are mole fractions ``x``, ``y``, the water vapour fraction ``yw``,
total flows ``L`` and ``V``, temperatures, and the absolute draw rates, with
the summation equations written out, and IPOPT's interior point does the
solving. Two formulations of the same physics agreeing to solver precision
is the check on the column; where they could differ by modelling choice
(the property model, water) they were made the same on purpose, and layer 2
measures those choices separately.

The network is general -- nodes with a pressure, where their liquid and
vapour go, and what enters them -- so a vacuum column (#294) is a different
:class:`Network`, not a different model.

Conventions
-----------
* Hydrocarbons are in both phases with Raoult K-values. Water is in the
  vapour only: it enters as steam and leaves overhead, where a total
  condenser decants it as free water. The model checks the assumption by
  reporting each stage's water saturation ``yw P / Psat_w(T)``.
* The total condenser's temperature is the bubble point of the condensate
  with free water present: ``sum K x0 + Psat_w(T0)/P_cond = 1``.
* The feed is flashed at the coil outlet temperature ``T_F`` and the
  feed-stage pressure; both phases enter the feed stage. ``T_F`` is free
  and the overflash spec fixes it.
* Pumparounds draw liquid at the stage composition, cool it to ``T_ret``
  and return it as liquid two stages up (or wherever the layout says).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import formulas as fm


@dataclass
class Network:
    """A stage network.

    Attributes:
        nodes: Node names, in an order the initialiser can sweep.
        P: Pressure of each node (Pa).
        liq_to: Where each node's (remaining) liquid goes: a node or a
            product name.
        vap_to: Where each node's vapour goes: a node, or ``"condenser"``.
        side_draws: ``name -> (source node, destination node)``: a liquid
            draw at the source's composition, at an absolute molar rate.
        pumparounds: ``name -> (source node, return node)``.
        steam: ``node -> mol/s`` of stripping steam.
        feed_node: Where the flashed feed enters.
        products: Liquid products ``name -> node`` whose remaining liquid it is,
            plus the distillate (from the condenser).
        P_condenser: Condenser pressure (Pa).
    """

    nodes: list
    P: dict
    liq_to: dict
    vap_to: dict
    side_draws: dict = field(default_factory=dict)
    pumparounds: dict = field(default_factory=dict)
    steam: dict = field(default_factory=dict)
    feed_node: str = ""
    P_condenser: float = 0.0
    distillate: str = "naphtha"

    @classmethod
    def atmospheric(cls, layout: dict) -> "Network":
        """The crude column of :data:`case.LAYOUT`: a main column, side
        strippers (draw to the stripper top, vapour back to the stage above
        the draw, steam to the stripper bottom), pumparounds, bottom steam."""
        N, F = layout["n_stages"], layout["feed_stage"]
        Pt, Pb = layout["P_top"], layout["P_bottom"]
        main = [f"s{k}" for k in range(1, N + 1)]
        P = {f"s{k}": Pt + (k - 1) * (Pb - Pt) / (N - 1) for k in range(1, N + 1)}
        liq_to = {f"s{k}": f"s{k + 1}" for k in range(1, N)}
        liq_to[f"s{N}"] = "residue"
        vap_to = {f"s{k}": f"s{k - 1}" for k in range(2, N + 1)}
        vap_to["s1"] = "condenser"
        nodes = list(main)
        side, steam = {}, {f"s{N}": layout["bottom_steam"]}
        for name, draw, n_st, st in layout["side_products"]:
            ss = [f"{name}{j}" for j in range(1, n_st + 1)]
            nodes += ss
            for j, s in enumerate(ss):
                P[s] = P[f"s{draw}"]
                liq_to[s] = ss[j + 1] if j + 1 < n_st else name
                vap_to[s] = ss[j - 1] if j > 0 else f"s{draw - 1}"
            side[name] = (f"s{draw}", ss[0])
            steam[ss[-1]] = st
        pas = {name: (f"s{d}", f"s{r}") for name, d, r in layout["pumparounds"]}
        return cls(nodes=nodes, P=P, liq_to=liq_to, vap_to=vap_to, side_draws=side,
                   pumparounds=pas, steam=steam, feed_node=f"s{F}",
                   P_condenser=layout["P_condenser"], distillate=layout["distillate"])

    @property
    def liquid_products(self):
        return sorted({d for d in self.liq_to.values() if d not in self.nodes})


def build(comp: dict, net: Network, feed: np.ndarray, specs: dict, steam_T: float,
          furnace_inlet_H: float, eps: float = 0.01):
    """Pose the column as a Pyomo model.

    Args:
        comp: Component table (:func:`case.component_data`).
        net: The stage network.
        feed: Crude molar flows (mol/s), per component.
        specs: ``volume_yield`` (product -> m^3/s standard volume),
            ``pa_duty`` (W), ``pa_delta_T`` (K), ``overflash`` (m^3/s of
            liquid leaving the stage above the feed).
        steam_T: Steam temperature (K).
        furnace_inlet_H: Enthalpy of the crude at the furnace inlet (W),
            from a flash computed outside (:func:`formulas.flash_enthalpy`).
        eps: Watson smoothing (:func:`formulas.watson_base`).
    """
    import pyomo.environ as pyo

    m = fm.pyomo_math()
    nc = len(comp["names"])
    C = range(nc)
    Tc, Pc, w = comp["Tc"], comp["Pc"], comp["omega_vp"]
    Tb, hv, cp = comp["Tb"], comp["hvap_nb"], comp["cp_ig"]
    MW, SG = comp["MW"], comp["SG"]
    vol = [MW[i] / (1000.0 * SG[i] * 999.016) for i in C]  # m^3/mol standard
    nodes = net.nodes
    S = 1e-6  # energy balances in MW

    def K(i, T, P):
        return fm.lee_kesler_psat(m, T, Tc[i], Pc[i], w[i]) / P

    def hV(i, T):
        return fm.h_ideal_gas(T, cp[i])

    def hL(i, T):
        return hV(i, T) - fm.latent_heat(m, T, Tb[i], Tc[i], hv[i], eps)

    M = pyo.ConcreteModel()
    M.N = pyo.Set(initialize=nodes, ordered=True)
    M.C = pyo.Set(initialize=list(C), ordered=True)
    M.T = pyo.Var(M.N, bounds=(250, 900), initialize=500)
    M.L = pyo.Var(M.N, bounds=(0, None), initialize=100)
    M.V = pyo.Var(M.N, bounds=(0, None), initialize=100)
    M.x = pyo.Var(M.N, M.C, bounds=(0, 1), initialize=1.0 / nc)
    M.y = pyo.Var(M.N, M.C, bounds=(0, 1), initialize=1.0 / nc)
    M.yw = pyo.Var(M.N, bounds=(0, 1), initialize=0.3)
    M.SD = pyo.Set(initialize=list(net.side_draws), ordered=True)
    M.PA = pyo.Set(initialize=list(net.pumparounds), ordered=True)
    M.D = pyo.Var(M.SD, bounds=(0, None), initialize=50)
    M.R = pyo.Var(M.PA, bounds=(0, None), initialize=200)
    M.T_ret = pyo.Var(M.PA, bounds=(250, 900), initialize=400)
    # condenser
    M.T0 = pyo.Var(bounds=(250, 600), initialize=320)
    M.x0 = pyo.Var(M.C, bounds=(0, 1), initialize=1.0 / nc)
    M.reflux = pyo.Var(bounds=(0, None), initialize=300)
    M.dist = pyo.Var(bounds=(0, None), initialize=200)
    M.Qc = pyo.Var(initialize=30)  # MW
    # feed flash
    Ft = float(np.sum(feed))
    z = feed / Ft
    M.TF = pyo.Var(bounds=(400, 800), initialize=590)
    M.beta = pyo.Var(bounds=(0, 1), initialize=0.6)
    M.xF = pyo.Var(M.C, bounds=(0, 1), initialize=lambda M, i: z[i])
    M.yF = pyo.Var(M.C, bounds=(0, 1), initialize=lambda M, i: z[i])
    PF = net.P[net.feed_node]

    M.feedflash_mb = pyo.Constraint(M.C, rule=lambda M, i: z[i] == (1 - M.beta) * M.xF[i] + M.beta * M.yF[i])
    M.feedflash_eq = pyo.Constraint(M.C, rule=lambda M, i: M.yF[i] == K(i, M.TF, PF) * M.xF[i])
    M.feedflash_sum = pyo.Constraint(expr=sum(M.yF[i] - M.xF[i] for i in C) == 0)

    # ---- inflows to each node, per component and as enthalpy ----
    def liquid_out(n):
        """Liquid leaving node n by its normal route (after draws)."""
        draws = [M.D[d] for d, (s, _) in net.side_draws.items() if s == n]
        pas = [M.R[p] for p, (s, _) in net.pumparounds.items() if s == n]
        return M.L[n] - sum(draws) - sum(pas)

    def comp_in(n, i):
        e = 0
        for k in nodes:
            if net.liq_to[k] == n:
                e += liquid_out(k) * M.x[k, i]
            if net.vap_to[k] == n:
                e += M.V[k] * M.y[k, i]
        for d, (s, t) in net.side_draws.items():
            if t == n:
                e += M.D[d] * M.x[s, i]
        for p, (s, r) in net.pumparounds.items():
            if r == n:
                e += M.R[p] * M.x[s, i]
        if n == nodes[0]:
            e += M.reflux * M.x0[i]
        if n == net.feed_node:
            e += Ft * ((1 - M.beta) * M.xF[i] + M.beta * M.yF[i])
        return e

    def water_in(n):
        e = net.steam.get(n, 0.0)
        for k in nodes:
            if net.vap_to[k] == n:
                e += M.V[k] * M.yw[k]
        return e

    hw_steam = fm.water_h_vapor(steam_T)

    def enthalpy_in(n):
        e = net.steam.get(n, 0.0) * hw_steam
        for k in nodes:
            if net.liq_to[k] == n:
                e += liquid_out(k) * sum(M.x[k, i] * hL(i, M.T[k]) for i in C)
            if net.vap_to[k] == n:
                e += M.V[k] * (sum(M.y[k, i] * hV(i, M.T[k]) for i in C)
                               + M.yw[k] * fm.water_h_vapor(M.T[k]))
        for d, (s, t) in net.side_draws.items():
            if t == n:
                e += M.D[d] * sum(M.x[s, i] * hL(i, M.T[s]) for i in C)
        for p, (s, r) in net.pumparounds.items():
            if r == n:
                e += M.R[p] * sum(M.x[s, i] * hL(i, M.T_ret[p]) for i in C)
        if n == nodes[0]:
            e += M.reflux * sum(M.x0[i] * hL(i, M.T0) for i in C)
        if n == net.feed_node:
            e += Ft * ((1 - M.beta) * sum(M.xF[i] * hL(i, M.TF) for i in C)
                       + M.beta * sum(M.yF[i] * hV(i, M.TF) for i in C))
        return e

    M.mb = pyo.Constraint(M.N, M.C, rule=lambda M, n, i:
                          comp_in(n, i) == M.L[n] * M.x[n, i] + M.V[n] * M.y[n, i])
    M.wb = pyo.Constraint(M.N, rule=lambda M, n: water_in(n) == M.V[n] * M.yw[n])
    M.eq = pyo.Constraint(M.N, M.C, rule=lambda M, n, i: M.y[n, i] == K(i, M.T[n], net.P[n]) * M.x[n, i])
    M.sx = pyo.Constraint(M.N, rule=lambda M, n: sum(M.x[n, i] for i in C) == 1)
    M.sy = pyo.Constraint(M.N, rule=lambda M, n: sum(M.y[n, i] for i in C) + M.yw[n] == 1)
    M.eb = pyo.Constraint(M.N, rule=lambda M, n: S * enthalpy_in(n) == S * (
        M.L[n] * sum(M.x[n, i] * hL(i, M.T[n]) for i in C)
        + M.V[n] * (sum(M.y[n, i] * hV(i, M.T[n]) for i in C) + M.yw[n] * fm.water_h_vapor(M.T[n]))))

    # ---- total condenser ----
    top = [k for k in nodes if net.vap_to[k] == "condenser"][0]
    Pcd = net.P_condenser
    M.cond_mb = pyo.Constraint(M.C, rule=lambda M, i: M.V[top] * M.y[top, i] == (M.reflux + M.dist) * M.x0[i])
    M.cond_sum = pyo.Constraint(expr=sum(M.x0[i] for i in C) == 1)
    M.cond_T = pyo.Constraint(expr=sum(K(i, M.T0, Pcd) * M.x0[i] for i in C)
                              + fm.water_psat(m, M.T0) / Pcd == 1)
    M.cond_eb = pyo.Constraint(expr=M.Qc == S * (
        M.V[top] * (sum(M.y[top, i] * hV(i, M.T[top]) for i in C) + M.yw[top] * fm.water_h_vapor(M.T[top]))
        - (M.reflux + M.dist) * sum(M.x0[i] * hL(i, M.T0) for i in C)
        - M.V[top] * M.yw[top] * fm.water_h_liquid(m, M.T0, eps)))

    # ---- pumparounds ----
    M.Qpa = pyo.Expression(M.PA, rule=lambda M, p: S * M.R[p] * sum(
        M.x[net.pumparounds[p][0], i] * (hL(i, M.T[net.pumparounds[p][0]]) - hL(i, M.T_ret[p])) for i in C))

    # ---- products ----
    def product_flow(name, i):
        if name == net.distillate:
            return M.dist * M.x0[i]
        n = [k for k in nodes if net.liq_to[k] == name][0]
        return liquid_out(n) * M.x[n, i]

    products = [net.distillate] + net.liquid_products
    M.P = pyo.Set(initialize=products, ordered=True)
    M.prod = pyo.Expression(M.P, M.C, rule=lambda M, p, i: product_flow(p, i))
    M.prod_vol = pyo.Expression(M.P, rule=lambda M, p: sum(M.prod[p, i] * vol[i] for i in C))

    # ---- furnace ----
    M.Qf = pyo.Expression(expr=S * (Ft * ((1 - M.beta) * sum(M.xF[i] * hL(i, M.TF) for i in C)
                                         + M.beta * sum(M.yF[i] * hV(i, M.TF) for i in C))
                                    - furnace_inlet_H))

    # ---- specs ----
    M.spec_yield = pyo.Constraint(list(specs["volume_yield"]), rule=lambda M, p:
                                  1e3 * M.prod_vol[p] == 1e3 * specs["volume_yield"][p])
    M.spec_pa_duty = pyo.Constraint(M.PA, rule=lambda M, p: M.Qpa[p] == S * specs["pa_duty"][p])
    M.spec_pa_dT = pyo.Constraint(M.PA, rule=lambda M, p:
                                  M.T[net.pumparounds[p][0]] - M.T_ret[p] == specs["pa_delta_T"][p])
    above_feed = [k for k in nodes if net.liq_to[k] == net.feed_node][0]
    M.spec_overflash = pyo.Constraint(expr=1e3 * liquid_out(above_feed) * sum(M.x[above_feed, i] * vol[i] for i in C)
                                      == 1e3 * specs["overflash"])
    M.obj = pyo.Objective(expr=0.0)
    M._meta = dict(net=net, comp=comp, products=products, vol=vol, S=S, top=top,
                   above_feed=above_feed, eps=eps)
    return M


def initialise(M, props: "fm.Props", feed: np.ndarray, guess: dict):
    """Set a starting point from a coarse guess, independent of difflow.

    ``guess``: ``T_top``, ``T_bottom`` (main column, linear), ``T_F``,
    ``L`` and ``V`` (mol/s, uniform), ``reflux``, ``draws`` and ``pa`` rates.
    Each stage's liquid starts as the feed's liquid-phase composition from an
    ideal flash at the stage temperature, its vapour as the equilibrium
    vapour of that liquid, so the equilibrium and summation equations start
    nearly satisfied and IPOPT is left the balances.
    """
    net = M._meta["net"]
    nodes = net.nodes
    z = feed / feed.sum()
    main = [k for k in nodes if k.startswith("s") and k[1:].isdigit()]
    n_main = len(main)
    Tmain = {k: guess["T_top"] + (guess["T_bottom"] - guess["T_top"]) * j / (n_main - 1)
             for j, k in enumerate(main)}
    for k in nodes:
        if k in Tmain:
            T = Tmain[k]
        else:
            src = [s for d, (s, t) in net.side_draws.items() if k.startswith(d)][0]
            T = Tmain[src] - 5.0
        P = net.P[k]
        Kv = props.K(T, P)
        b = fm.rachford_rice(z, Kv)
        x = z / (1 + b * (Kv - 1))
        x = x / x.sum()
        y = Kv * x
        yw = 0.3
        y = (1 - yw) * y / y.sum()
        M.T[k].set_value(T)
        for i in range(len(z)):
            M.x[k, i].set_value(max(x[i], 1e-12))
            M.y[k, i].set_value(max(y[i], 1e-12))
        M.yw[k].set_value(yw)
        M.L[k].set_value(guess["L"])
        M.V[k].set_value(guess["V"])
    for d in net.side_draws:
        M.D[d].set_value(guess["draws"][d])
    for p in net.pumparounds:
        M.R[p].set_value(guess["pa"][p])
        M.T_ret[p].set_value(M.T[net.pumparounds[p][0]].value - 60)
    M.TF.set_value(guess["T_F"])
    Kf = props.K(guess["T_F"], net.P[net.feed_node])
    b = fm.rachford_rice(z, Kf)
    xF = z / (1 + b * (Kf - 1))
    M.beta.set_value(b)
    for i in range(len(z)):
        M.xF[i].set_value(xF[i])
        M.yF[i].set_value(Kf[i] * xF[i])
    top = M._meta["top"]
    y1 = np.array([M.y[top, i].value for i in range(len(z))])
    for i in range(len(z)):
        M.x0[i].set_value(y1[i] / y1.sum())
    M.T0.set_value(guess["T_top"] - 60)
    M.reflux.set_value(guess["reflux"])
    M.dist.set_value(guess["dist"])


def solve(M, ipopt: str | None = None, tee: bool = False, options: dict | None = None):
    import pyomo.environ as pyo

    opt = pyo.SolverFactory("ipopt", executable=ipopt) if ipopt else pyo.SolverFactory("ipopt")
    o = {"tol": 1e-10, "max_iter": 3000, "nlp_scaling_method": "gradient-based",
         "bound_push": 1e-8, "mu_init": 1e-3}
    o.update(options or {})
    for k, v in o.items():
        opt.options[k] = v
    res = opt.solve(M, tee=tee)
    return str(res.solver.termination_condition), res


def results(M) -> dict:
    """Plain-number results of a solved model."""
    from pyomo.environ import value

    meta = M._meta
    net, comp = meta["net"], meta["comp"]
    nc = len(comp["names"])
    out = {"T": {k: value(M.T[k]) for k in net.nodes}, "T_condenser": value(M.T0),
           "T_F": value(M.TF), "feed_vaporized": value(M.beta),
           "condenser_duty": value(M.Qc) / meta["S"],
           "furnace_duty": value(M.Qf) / meta["S"],
           "pumparound_duty": {p: value(M.Qpa[p]) / meta["S"] for p in net.pumparounds},
           "pumparound_return_T": {p: value(M.T_ret[p]) for p in net.pumparounds},
           "pumparound_rate": {p: value(M.R[p]) for p in net.pumparounds},
           "side_draw_rate": {d: value(M.D[d]) for d in net.side_draws},
           "reflux": value(M.reflux),
           "products": {p: [value(M.prod[p, i]) for i in range(nc)] for p in meta["products"]},
           "free_water": value(M.V[meta["top"]] * M.yw[meta["top"]]),
           "water_saturation": {k: value(M.yw[k]) * net.P[k] / fm.water_psat(fm.NUMPY, value(M.T[k]))
                                for k in net.nodes},
           "L": {k: value(M.L[k]) for k in net.nodes}, "V": {k: value(M.V[k]) for k in net.nodes}}
    return out
