"""An equation-oriented MESH model of a vacuum column, in Pyomo.

Written from the physics -- Material balance, Equilibrium (Murphree on the
vapour), Summation and entHalpy balance on every stage -- and solved
simultaneously by IPOPT. It calls nothing in difflow: the inputs are the
frozen component table, the feed's molar flows and the plain numbers of
:mod:`.vdu_case`, and the property model is :mod:`.vdu_formulas`.

The formulation is deliberately not difflow's:

==========================  ===============================  ==============================
                            difflow (``StageColumn``)        here
==========================  ===============================  ==============================
unknowns                    ln of component flows, l and v   mole fractions ``x``, ``y``,
                                                             total flows ``L``, ``V``
balances                    log form, ``logsumexp`` of ins   linear, ``L x + V y = sum in``
draws                       softmax logits of stage liquid   absolute route flows (mol/s)
summation                   implied by the flows             written out, steam included
pumparound duty             a knob; return T recovered       return temperature an unknown;
                            afterwards by Newton             duty computed afterwards
Watson-K correction         4 unrolled passes, AD            one unknown + one constraint
                                                             per pass, chain rule by hand
feed flash                  in the Newton system             Rachford-Rice beforehand
LVGO T95 spec               ``jnp.interp`` of the TBP curve  cumulative mass up to the
                                                             target = 0.95
solver                      damped Newton (JAX)              IPOPT
==========================  ===============================  ==============================

The T95 spec: the TBP curve distils each cut uniformly between its edges,
so it is piecewise linear in cumulative mass fraction with knots at the cut
edges. For a target a fraction ``theta`` of the way through cut ``k``,
"T95 = target" and "the cuts below ``k`` plus ``theta`` of cut ``k`` hold
95 % of the LVGO by mass" have the same zero set (the curve is monotone).
The second is smooth, which IPOPT prefers, and gives the same implicit
derivatives. 450 C is the upper edge of PC425_450 (``theta = 1``).

Conventions (the same physics as difflow's, written from the docs):

* Stage 0 is the top. Pressures are linear from the top to the flash zone
  and rise by ``stripping_dP`` per stripping stage below it.
* Water is in the vapour only, as steam: injected below the bottom stage
  at ``steam_T``, it rises through every stage and leaves overhead. The
  summation is ``sum y + S/V = 1`` with ``y`` relative to the total vapour.
* Murphree vapour efficiency ``y = eta K x + (1 - eta) y_below`` with
  ``y_below = 0`` under the bottom stage; the flash zone is an equilibrium
  stage.
* The feed is flashed at the furnace outlet temperature and the flash-zone
  pressure plus the transfer-line drop, and both phases enter the flash zone.
  The furnace duty heats the feed from its inlet temperature as liquid.
* Liquid leaves a stage by routes: a pumparound returns to a stage above at
  ``T_ret`` (the cooler duty is what it gives up), entrained flash-zone
  liquid goes to the wash bed's slop stage (the de-entrained part) or past it
  to the HVGO draw, and a product route leaves the column.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import vdu_formulas as vf


@dataclass
class Network:
    """A vacuum column's stages and liquid routes.

    Attributes:
        N: Number of stages (0 is the top).
        feed_stage: The flash zone.
        P: Stage pressures (Pa).
        eta: Murphree vapour efficiency per stage.
        routes: ``name -> (source, dest or None, kind)``; kind is
            ``"rest"`` (whatever the other routes leave), ``"free"`` (an
            unknown rate, closed by a spec) or a float (fraction of the
            source stage's liquid).
        pumparounds: Routes that return cooled liquid.
        steam_in: Stage under which the stripping steam enters (mol/s).
        products: Product routes (dest None), plus ``"overhead"``.
    """

    N: int
    feed_stage: int
    P: list
    eta: list
    routes: dict
    pumparounds: tuple
    steam_in: dict = field(default_factory=dict)

    @classmethod
    def vacuum(cls, layout: dict, efficiency: dict, steam_mol_s: float) -> "Network":
        """The column of ``docs/unit-operations-refinery.md``: LVGO, HVGO and
        wash beds above the flash zone, stripping stages below."""
        nl, nh, nw, ns = (layout[k] for k in ("n_lvgo", "n_hvgo", "n_wash", "n_strip"))
        fz = nl + nh + nw
        N = fz + 1 + ns
        Pt, Pf = layout["top_P"], layout["flash_zone_P"]
        P = [Pt + (Pf - Pt) * j / fz for j in range(fz + 1)]
        P += [Pf + layout["stripping_dP"] * k for k in range(1, ns + 1)]
        eta = ([efficiency["lvgo"]] * nl + [efficiency["hvgo"]] * nh
               + [efficiency["wash"]] * nw + [1.0] + [efficiency["strip"]] * ns)
        lvgo, hvgo, slop = nl - 1, nl + nh - 1, fz - 1
        eps, dee = layout["entrainment"], layout["deentrainment"]
        routes = {}
        for j in range(N):
            if j == lvgo:
                routes["lvgo"] = (j, None, "rest")
                routes["lvgo_pa"] = (j, 0, "free")
            elif j == hvgo:
                routes["wash_oil"] = (j, j + 1, "rest")
                routes["hvgo"] = (j, None, "free")
                routes["hvgo_pa"] = (j, nl, "free")
            elif j == slop:
                routes["slop"] = (j, None, "rest")
            elif j == fz:
                routes["entrained"] = (j, slop, eps * dee)
                routes["entrained_bypass"] = (j, hvgo, eps * (1.0 - dee))
                routes[f"stage{j}.down"] = (j, j + 1, "rest")
            elif j == N - 1:
                routes["residue"] = (j, None, "rest")
            else:
                routes[f"stage{j}.down"] = (j, j + 1, "rest")
        return cls(N=N, feed_stage=fz, P=P, eta=eta, routes=routes,
                   pumparounds=("lvgo_pa", "hvgo_pa"), steam_in={N - 1: steam_mol_s})

    def routes_from(self, j):
        return [r for r, (s, _, _) in self.routes.items() if s == j]

    def routes_to(self, j):
        return [r for r, (_, d, _) in self.routes.items() if d == j]

    @property
    def products(self):
        return [r for r, (_, d, _) in self.routes.items() if d is None] + ["overhead"]

    def steam_up(self, j):
        """Steam in the vapour leaving stage ``j`` upward (mol/s)."""
        return sum(s for k, s in self.steam_in.items() if k >= j)


def feed_flash(props: vf.Props, feed, T, P):
    """Ideal flash of the feed at the furnace outlet: ``(lF, vF)`` in mol/s."""
    z = np.asarray(feed) / np.sum(feed)
    K = props.K(T, P)
    b = vf.rachford_rice(z, K)
    x = z / (1.0 + b * (K - 1.0))
    lF = (1.0 - b) * np.sum(feed) * x
    return lF, np.asarray(feed) - lF


def build(comp: dict, net: Network, feed, case: dict, n_pass=vf.N_PASS, published=False):
    """The Pyomo model.

    Args:
        comp: Component table (lists: ``Tb``, ``SG``, ``MW``, ``Kw``,
            ``T_lo``, ``T_hi``, ``names``).
        net: The :class:`Network`.
        feed: Feed molar flows (mol/s).
        case: ``feed_T``, ``furnace_T``, ``furnace_P``, ``steam_T``, and the
            spec values ``top_T``, ``overflash``, ``lvgo_T95``,
            ``lvgo_pa_kg_s``, ``hvgo_pa_kg_s``.
        n_pass: Watson-K passes (``None``: the exact fixed point).
        published: Maxwell-Bonnell's unblended piecewise branches.
    """
    import pyomo.environ as pyo

    m = vf.pyomo_math()
    props = vf.Props(comp, n_pass=n_pass, published=published)
    C, N = len(comp["names"]), net.N
    I, J = range(C), range(N)  # noqa: E741 -- the index sets, named as in the equations
    Tb, SG, MW, Kw = (np.asarray(comp[k], dtype=float) for k in ("Tb", "SG", "MW", "Kw"))
    f = vf.watson_f(Tb)
    feed = np.asarray(feed, dtype=float)
    F_kg = float(feed @ MW) / 1000.0

    M = pyo.ConcreteModel()
    M.T = pyo.Var(J, bounds=(250.0, 900.0), initialize=600.0)
    M.L = pyo.Var(J, bounds=(1e-8, None), initialize=50.0)
    M.V = pyo.Var(J, bounds=(1e-8, None), initialize=50.0)
    # unbounded fractions: the equilibrium and balances are linear in them, so
    # a trace component (the residue lump at the top, 1e-30) is solved for
    # exactly instead of being held off a bound by the barrier
    M.x = pyo.Var(J, I, initialize=1.0 / C)
    M.y = pyo.Var(J, I, initialize=1.0 / C)
    free = [r for r, (_, _, k) in net.routes.items() if k == "free"]
    M.R = pyo.Var(free, bounds=(0.0, None), initialize=10.0)
    M.T_ret = pyo.Var(list(net.pumparounds), bounds=(250.0, 900.0), initialize=400.0)

    # Watson-K passes: lp[j, i, k] = g(T_j, lp[j, i, k-1]) and its T-derivative
    K_pass = list(range(1, (n_pass or 1) + 1))
    M.lp = pyo.Var(J, I, K_pass, initialize=0.0)
    M.dlp = pyo.Var(J, I, K_pass, initialize=0.0)

    def pass_rule(M, j, i, k):
        lp0 = vf.LP_START if k == 1 else M.lp[j, i, k - 1]
        return vf.mb_pass(m, lp0, M.T[j], Tb[i], Kw[i], f[i], published)

    if n_pass is not None:
        def c_lp(M, j, i, k):
            return M.lp[j, i, k] == pass_rule(M, j, i, k)[0]

        def c_dlp(M, j, i, k):
            _, gT, glp = pass_rule(M, j, i, k)
            d0 = 0.0 if k == 1 else M.dlp[j, i, k - 1]
            return M.dlp[j, i, k] == gT + glp * d0
    else:
        def c_lp(M, j, i, k):
            g, _, _ = vf.mb_pass(m, M.lp[j, i, k], M.T[j], Tb[i], Kw[i], f[i], published)
            return M.lp[j, i, k] == g

        def c_dlp(M, j, i, k):
            _, gT, glp = vf.mb_pass(m, M.lp[j, i, k], M.T[j], Tb[i], Kw[i], f[i], published)
            return M.dlp[j, i, k] * (1.0 - glp) == gT
    M.c_lp = pyo.Constraint(J, I, K_pass, rule=c_lp)
    M.c_dlp = pyo.Constraint(J, I, K_pass, rule=c_dlp)
    kl = K_pass[-1]

    def K(j, i):
        return m.exp(vf.LN10 * M.lp[j, i, kl]) * vf.MMHG / net.P[j]

    def hL(i, T):
        return vf.h_liquid(T, SG[i], Kw[i], MW[i])

    def hV(j, i):
        return hL(i, M.T[j]) + vf.dh_vap(M.T[j], M.dlp[j, i, kl])

    hS = vf.h_steam

    # feed: flashed beforehand at fixed furnace conditions
    lF, vF = feed_flash(props, feed, case["furnace_T"], case["furnace_P"])
    T_fo = case["furnace_T"]
    h_feed = float(lF @ props.hL(T_fo) + vF @ props.hV(T_fo))

    def route_flow(r):
        s, _, kind = net.routes[r]
        if kind == "free":
            return M.R[r]
        if kind == "rest":
            return M.L[s] - sum(route_flow(o) for o in net.routes_from(s) if o != r)
        return kind * M.L[s]

    Sup = [net.steam_up(j) for j in J]
    fs = net.feed_stage

    def comp_in(j, i):
        e = sum(route_flow(r) * M.x[net.routes[r][0], i] for r in net.routes_to(j))
        if j + 1 < N:
            e = e + M.V[j + 1] * M.y[j + 1, i]
        if j == fs:
            e = e + float(lF[i] + vF[i])
        return e

    M.mb = pyo.Constraint(J, I, rule=lambda M, j, i:
                          M.L[j] * M.x[j, i] + M.V[j] * M.y[j, i] == comp_in(j, i))

    def eq(M, j, i):
        below = M.y[j + 1, i] if j + 1 < N else 0.0
        eta = net.eta[j]
        return M.y[j, i] == eta * K(j, i) * M.x[j, i] + (1.0 - eta) * below
    M.eq = pyo.Constraint(J, I, rule=eq)
    # a "rest" route is what the others leave: it cannot run backwards
    rest = [r for r, (_, _, k) in net.routes.items() if k == "rest"]
    M.rest_nonneg = pyo.Constraint(rest, rule=lambda M, r: route_flow(r) >= 0.0)
    M.sx = pyo.Constraint(J, rule=lambda M, j: sum(M.x[j, i] for i in I) == 1.0)
    M.sy = pyo.Constraint(J, rule=lambda M, j:
                          M.V[j] * (1.0 - sum(M.y[j, i] for i in I)) == Sup[j])

    E = 1e-6  # W -> MW: energy rows the size of the others

    def liquid_h(j, T):
        return sum(M.x[j, i] * hL(i, T) for i in I)

    def vapour_out_h(j):
        return M.V[j] * sum(M.y[j, i] * hV(j, i) for i in I) + Sup[j] * hS(M.T[j])

    def eb(M, j):
        out = M.L[j] * liquid_h(j, M.T[j]) + vapour_out_h(j)
        inn = 0.0
        for r in net.routes_to(j):
            s = net.routes[r][0]
            T_in = M.T_ret[r] if r in net.pumparounds else M.T[s]
            inn = inn + route_flow(r) * liquid_h(s, T_in)
        if j + 1 < N:
            inn = inn + vapour_out_h(j + 1)
        if j == fs:
            inn = inn + h_feed
        if j in net.steam_in:
            inn = inn + net.steam_in[j] * hS(case["steam_T"])
        return E * inn == E * out
    M.eb = pyo.Constraint(J, rule=eb)

    def mass(j, flow):
        return flow * sum(M.x[j, i] * MW[i] for i in I) / 1000.0

    def src(r):
        return net.routes[r][0]

    # specs
    M.spec_top_T = pyo.Constraint(expr=M.T[0] == case["top_T"])
    M.spec_lpa = pyo.Constraint(expr=mass(src("lvgo_pa"), M.R["lvgo_pa"]) == case["lvgo_pa_kg_s"])
    M.spec_hpa = pyo.Constraint(expr=mass(src("hvgo_pa"), M.R["hvgo_pa"]) == case["hvgo_pa_kg_s"])
    M.spec_overflash = pyo.Constraint(
        expr=mass(src("slop"), route_flow("slop")) == case["overflash"] * F_kg)
    # the cut k holding the target, and how far through it the target sits
    T_lo, T_hi = np.asarray(comp["T_lo"]), np.asarray(comp["T_hi"])
    k = int(np.searchsorted(T_hi, case["lvgo_T95"] - 1e-9))
    if not (T_lo[k] < case["lvgo_T95"] <= T_hi[k] + 1e-9):
        raise ValueError("the LVGO T95 target is outside the component table")
    theta = float((case["lvgo_T95"] - T_lo[k]) / (T_hi[k] - T_lo[k]))
    jl = src("lvgo")
    M.spec_t95 = pyo.Constraint(
        expr=(sum(M.x[jl, i] * MW[i] for i in range(k)) + theta * M.x[jl, k] * MW[k]) / 100.0
        == 0.95 * sum(M.x[jl, i] * MW[i] for i in I) / 100.0)

    M._meta = dict(net=net, comp=comp, feed=feed, case=case, n_pass=n_pass, props=props,
                   lF=lF, vF=vF, h_feed=h_feed, F_kg=F_kg, route_flow=route_flow,
                   hL=hL, liquid_h=liquid_h, kl=kl, Sup=Sup)
    return M


def set_passes(M, j, i, T):
    """Set stage ``j``'s Watson-K pass variables for component ``i`` at ``T``."""
    props, n_pass = M._meta["props"], M._meta["n_pass"]
    Tb, Kw, f = props.Tb[i], props.Kw[i], props.f[i]
    if n_pass is None:
        lp, d = vf.mb_log10p(T, Tb, Kw, n_pass=None, published=props.published)
        M.lp[j, i, 1].set_value(float(lp))
        M.dlp[j, i, 1].set_value(float(d))
        return
    lp, d = vf.LP_START, 0.0
    for k in range(1, n_pass + 1):
        g, gT, glp = vf.mb_pass(vf.NUMPY, lp, T, Tb, Kw, f, props.published)
        lp, d = g, gT + glp * d
        M.lp[j, i, k].set_value(float(lp))
        M.dlp[j, i, k].set_value(float(d))


def engineering_guess(M) -> dict:
    """A rough design-basis guess from the feed flash and the specs alone.

    What a designer would sketch before simulating: the furnace-outlet flash
    says how much of the feed is vapour; the LVGO is roughly the vapour that
    boils below the T95 target, the HVGO the rest of it less the slop. The
    vapour leaving a stage carries the products drawn above it, plus, between
    a (non-top) pumparound's return and its draw, the vapour that the cold
    return condenses; under the flash zone, a few percent of the liquid
    stripped. Temperatures are linear from the top spec to the furnace outlet
    less a few degrees, and fall slightly down the stripper (steam).
    """
    meta = M._meta
    net, case, props = meta["net"], meta["case"], meta["props"]
    lF, vF = meta["lF"], meta["vF"]
    Tb, MW = props.Tb, props.MW
    fs, N = net.feed_stage, net.N
    light = Tb < case["lvgo_T95"]
    lvgo = 0.5 * float(vF[light].sum())
    slop = case["overflash"] * meta["F_kg"] * 1000.0 / float(MW[np.argmin(np.abs(Tb - 850.0))])
    hvgo = max(float(vF.sum()) - lvgo - slop, 0.1 * float(vF.sum()))
    mw_l = float(vF[light] @ MW[light] / vF[light].sum())
    mw_h = float(vF[~light] @ MW[~light] / max(vF[~light].sum(), 1e-12))
    R = {"lvgo_pa": case["lvgo_pa_kg_s"] * 1000.0 / mw_l,
         "hvgo_pa": case["hvgo_pa_kg_s"] * 1000.0 / mw_h, "hvgo": hvgo}
    draws = {net.routes["lvgo"][0]: lvgo, net.routes["hvgo"][0]: hvgo,
             net.routes["slop"][0]: slop}
    V = np.zeros(N)
    for j in range(N):
        if j <= fs:
            V[j] = 1e-4 * float(vF.sum()) + sum(v for s, v in draws.items() if s < j)
        else:
            V[j] = 0.03 * float(lF.sum())
    s, d = net.routes["hvgo_pa"][0], net.routes["hvgo_pa"][1]
    for j in range(d + 1, s + 1):
        V[j] += 0.5 * R["hvgo_pa"]
    T_top, T_fz = case["top_T"], case["furnace_T"] - 5.0
    T = [T_top + (T_fz - T_top) * j / fs for j in range(fs + 1)]
    T += [T_fz - 3.0 * k for k in range(1, N - fs)]
    return dict(T=T, V=V.tolist(), R=R,
                T_ret={"lvgo_pa": T_top - 2.0, "hvgo_pa": T[s] - 100.0})


def initialise(M, guess: dict | None = None):
    """A starting point from an engineering guess, independent of difflow.

    ``guess``: ``T`` (per stage, K), ``V`` (hydrocarbon vapour per stage,
    mol/s), ``R`` (route rates, mol/s), ``T_ret``. Liquid flows follow from
    total molar balances with those vapour rates; compositions from solving
    each component's linear stage balances with the K-values at the guessed
    temperatures (the inner loop of a Wang-Henke bubble-point method), then
    a few bubble-point temperature updates. The equilibrium and balance rows
    start nearly satisfied and IPOPT is left the energy balances and specs.
    """
    meta = M._meta
    guess = guess or engineering_guess(M)
    net, props, feed = meta["net"], meta["props"], meta["feed"]
    lF, vF = meta["lF"], meta["vF"]
    N, C, fs = net.N, len(feed), net.feed_stage
    T = np.array(guess["T"], dtype=float)
    Vh = np.array(guess["V"], dtype=float)
    Sup = np.array(meta["Sup"])
    V = Vh + Sup

    # liquid totals from overall stage balances, top down: L_j = in_j - V_j
    def totals(Vh):
        L = np.zeros(N)
        rates = {}
        for j in range(N):
            inn = sum(rates[r] for r in net.routes_to(j) if r in rates)
            inn += Vh[j + 1] if j + 1 < N else 0.0
            inn += float(np.sum(feed)) if j == fs else 0.0
            # routes into j from below (pumparound returns) are known guesses
            for r in net.routes_to(j):
                if r not in rates and net.routes[r][2] == "free":
                    inn += guess["R"][r]
            L[j] = max(inn - Vh[j], 1e-3)
            for r in net.routes_from(j):
                kind = net.routes[r][2]
                if kind == "free":
                    rates[r] = guess["R"][r]
                elif kind != "rest":
                    rates[r] = kind * L[j]
            for r in net.routes_from(j):
                if net.routes[r][2] == "rest":
                    others = sum(rates[o] for o in net.routes_from(j) if o != r)
                    rates[r] = max(L[j] - others, 1e-3)
                    L[j] = others + rates[r]
        return L, rates

    L, rates = totals(Vh)
    for _ in range(30):
        Kv = np.array([props.K(T[j], net.P[j]) for j in range(N)])  # (N, C)
        lflow = np.zeros((N, C))
        for i in range(C):
            A = np.zeros((N, N))
            b = np.zeros(N)
            for j in range(N):
                # l_j + v_j with v_j = (V_j / L_j) K l_j (equilibrium stages)
                A[j, j] += 1.0 + V[j] / L[j] * Kv[j, i]
                for r in net.routes_to(j):
                    s = net.routes[r][0]
                    A[j, s] -= rates[r] / L[s]
                if j + 1 < N:
                    A[j, j + 1] -= V[j + 1] / L[j + 1] * Kv[j + 1, i]
                if j == fs:
                    b[j] += lF[i] + vF[i]
            lflow[:, i] = np.linalg.solve(A, b)
        x = lflow / lflow.sum(1, keepdims=True)
        # bubble point at each stage: sum K x = 1 - S/V (steam dilutes y)
        for j in range(N):
            target = 1.0 - Sup[j] / V[j]
            for _ in range(20):
                s = np.sum(props.K(T[j], net.P[j]) * x[j])
                T[j] = min(max(T[j] / (1.0 + 0.02 * np.log(s / target)), 300.0), 700.0)
        T[0] = M._meta["case"]["top_T"]
    Kv = np.array([props.K(T[j], net.P[j]) for j in range(N)])
    y = Kv * x
    for j in range(N):
        M.T[j].set_value(T[j])
        M.L[j].set_value(L[j])
        M.V[j].set_value(V[j])
        for i in range(C):
            M.x[j, i].set_value(x[j, i])
            M.y[j, i].set_value(y[j, i])
            # the pass variables, evaluated forward at the guess
            set_passes(M, j, i, T[j])
    for r in M.R:
        M.R[r].set_value(guess["R"][r])
    for p in net.pumparounds:
        M.T_ret[p].set_value(guess["T_ret"][p])


def solve(M, ipopt: str | None = None, tee: bool = False, options: dict | None = None):
    """IPOPT on the model; returns ``(termination, result)``."""
    import pyomo.environ as pyo

    opt = pyo.SolverFactory("ipopt", executable=ipopt) if ipopt else pyo.SolverFactory("ipopt")
    o = {"tol": 1e-10, "max_iter": 3000, "nlp_scaling_method": "gradient-based",
         "bound_push": 1e-8, "mu_init": 1e-3}
    o.update(options or {})
    for k, v in o.items():
        opt.options[k] = v
    res = opt.solve(M, tee=tee)
    return str(res.solver.termination_condition), res


def solve_staged(M, ipopt: str | None = None, tee: bool = False):
    """Solve from :func:`initialise`'s point in two steps.

    The overflash spec is the one that can strand IPOPT from a crude start:
    the slop is the whole liquid of the wash bed's bottom stage, so a guess
    that draws too much HVGO dries the wash bed, its down-flow sits on its
    zero bound and IPOPT reports a local infeasibility there. So the first
    solve holds the HVGO draw at the guess (a wet wash bed) and drops the
    overflash spec; the second restores the spec and frees the draw from
    the first solve's point. Returns ``(termination, info)``.
    """
    M.spec_overflash.deactivate()
    M.R["hvgo"].fix()
    t0 = time.time()
    first, _ = solve(M, ipopt, tee)
    M.spec_overflash.activate()
    M.R["hvgo"].unfix()
    term, res = solve(M, ipopt, tee)
    return term, {"status": term, "first_step": first,
                  "seconds": round(time.time() - t0, 1)}


def copy_values(src, dst):
    """Warm start ``dst`` from a solved ``src`` (same structure)."""
    import pyomo.environ as pyo

    for v in src.component_objects(pyo.Var, active=True):
        dv = dst.find_component(v.name)
        for k in v:
            if v[k].value is not None and k in dv:
                dv[k].set_value(v[k].value, skip_validation=True)


def results(M) -> dict:
    """Plain-number results of a solved model (SI; rates kg/s, duties W)."""
    from pyomo.environ import value

    meta = M._meta
    net, comp, props = meta["net"], meta["comp"], meta["props"]
    C, N = len(comp["names"]), net.N
    MW = np.asarray(comp["MW"])
    T = np.array([value(M.T[j]) for j in range(N)])
    x = np.array([[value(M.x[j, i]) for i in range(C)] for j in range(N)])
    y = np.array([[value(M.y[j, i]) for i in range(C)] for j in range(N)])
    V = np.array([value(M.V[j]) for j in range(N)])
    rf = meta["route_flow"]
    mol = {r: value(rf(r)) * x[net.routes[r][0]] for r in net.routes}
    prod = {r: mol[r] for r in net.products if r != "overhead"}
    prod["overhead"] = V[0] * y[0]
    rates = {r: float(mol[r] @ MW / 1000.0) for r in net.routes}
    rates["overhead"] = float(prod["overhead"] @ MW / 1000.0)
    duties, ret = {}, {}
    for p in net.pumparounds:
        s = net.routes[p][0]
        Tr = value(M.T_ret[p])
        duties[p] = float(mol[p] @ (props.hL(T[s]) - props.hL(Tr)))
        ret[p] = Tr
    case, feed = meta["case"], meta["feed"]
    vF = meta["vF"]
    furnace = meta["h_feed"] - float(feed @ props.hL(case["feed_T"]))
    inspections = {p: vf.product_inspection(comp, prod[p]) for p in prod}
    return {
        "T": T.tolist(), "P": list(net.P),
        "L": [value(M.L[j]) for j in range(N)], "V": V.tolist(),
        "rates": rates, "products_mol_s": {p: prod[p].tolist() for p in prod},
        "inspection": inspections,
        "pumparound_duty": duties, "pumparound_return_T": ret,
        "furnace_duty": furnace,
        "furnace_vapor_fraction": float(vF @ MW / (feed @ MW)),
        "feed_rate": float(feed @ MW / 1000.0),
        "overflash": rates["slop"] / float(feed @ MW / 1000.0),
    }
