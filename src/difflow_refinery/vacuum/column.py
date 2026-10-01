"""A network-of-stages distillation column for pseudocomponent mixtures.

This is the machinery shared by every refinery column: a stack of
equilibrium stages, numbered from the top, whose liquids are *routed* --
down to the next stage, out as a side product, round a pumparound to a stage
above, up with the vapor as entrainment -- and whose MESH equations are
solved simultaneously (Naphtali-Sandholm), with a furnace flash in front of
the feed stage. :mod:`difflow_refinery.vacuum` builds a vacuum column out of
it; a crude column is the same machinery with a condenser.

Variables
    For the furnace outlet and every stage: ``ln`` of each component's liquid
    and vapor flow (log flows cannot go negative, and span the twenty decades
    between a residue lump in the overhead and the same lump in the residue
    without losing precision), and each stage temperature. Plus one logit per
    *share* route (below) and one scaled value per freed knob.

Routes
    The liquid leaving a stage is split between its routes. ``fixed`` routes
    take a fraction set by the knobs (entrainment). The rest is shared out by
    a softmax over one ``reference`` route and any number of ``share`` routes,
    each of which carries a logit as a variable -- so a draw can never take
    more liquid than the stage has, which is what makes a draw *rate* spec
    safe to hand to Newton. A route with ``duty`` is a pumparound: its liquid
    comes back to its destination stage with that much heat removed.

Specs
    Every share route has a default equation, ``<route>.rate`` equal to the
    knob of that name (kg/s). Every other knob (a duty, the furnace outlet
    temperature, a pressure) is a fixed number. A :class:`StageSpec` *replaces*
    one of those -- a route's rate equation, or a knob's fixed value, which
    then becomes a variable -- with ``output == target``. So the degrees of
    freedom always balance, and any column output (a stage temperature, a
    product's TBP 95% point, the overflash) can be specified by giving up
    the knob that controls it.

Solve
    Damped Newton on the full residual with a dense Jacobian (``jax.jacfwd``;
    a VDU is a few hundred variables), step caps on temperatures and log
    flows, and an Armijo backtracking line search, inside ``lax.while_loop``
    on stop-gradient inputs. Then one Newton step with the Jacobian frozen
    at the solution: its value is the solution (to round-off) and its
    derivative is exactly the implicit-function-theorem derivative
    ``-J^-1 dR/dtheta``. So ``jax.grad`` and ``jax.jacfwd`` of anything
    computed from the result are exact, cost one linear solve, and never
    differentiate the iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from difflow_refinery.vacuum.assay import PseudoComponents, product_properties
from difflow_refinery.vacuum.thermo import ColumnThermo, MW_WATER


@dataclass(frozen=True)
class Route:
    """Where part of a stage's liquid goes.

    Attributes:
        name: Route name; ``<name>.rate`` is its mass flow (kg/s) output.
        source: Stage the liquid leaves (0 = top).
        dest: Stage it enters, or ``None`` for a product leaving the column.
        kind: ``"reference"`` (takes what the share routes leave),
            ``"share"`` (a logit variable with a rate equation) or ``"fixed"``.
        fraction: For ``fixed`` routes, ``knobs -> fraction`` of the liquid.
        duty: Knob naming the heat removed from the routed liquid (W) -- a
            pumparound.
    """

    name: str
    source: int
    dest: Optional[int]
    kind: str = "share"
    fraction: Optional[Callable] = None
    duty: Optional[str] = None


@dataclass(frozen=True)
class StageSpec:
    """``output == target``, in place of the equation or knob ``replaces``.

    Attributes:
        output: Name of a column output (see :meth:`StageColumn.output_names`).
        target: Its required value (SI: K, kg/s, W, Pa; fractions as fractions).
        replaces: A share route's ``<route>.rate`` or a knob name.
        scale: Residual scale; inferred from the output's kind if omitted.
    """

    output: str
    target: float
    replaces: str
    scale: Optional[float] = None


@dataclass(frozen=True)
class ColumnLayout:
    """Static description of a stage network (see module docstring).

    Attributes:
        n_stages: Number of equilibrium stages, numbered from the top.
        feed_stage: Stage the furnace outlet enters.
        routes: Every liquid route; each stage needs at least a reference.
        pressures: ``knobs -> (n_stages,)`` stage pressures (Pa).
        furnace_pressure: ``knobs -> `` furnace outlet pressure (Pa).
        steam_injection: ``knobs -> (n_stages,)`` steam injected (mol/s) at
            temperature ``knobs["steam.T"]``.
        coil_steam: ``knobs -> `` steam injected with the feed (mol/s).
        knob_scales: Scale of each knob, for scaling freed-knob variables.
        knob_caps: Largest Newton step for a freed knob, in its own units.
        init_guess: ``(column, ctx) -> (T0, L0, V0, z0)``: stage
            temperatures, total liquid and hydrocarbon vapor flows (mol/s)
            and a dict of share-route logits.
        extra_outputs: ``(ctx, outputs) -> dict`` of column-specific outputs.
    """

    n_stages: int
    feed_stage: int
    routes: tuple
    pressures: Callable
    furnace_pressure: Callable
    steam_injection: Callable
    coil_steam: Callable
    knob_scales: tuple = ()
    knob_caps: tuple = ()
    init_guess: Optional[Callable] = None
    extra_outputs: Optional[Callable] = None

    def routes_from(self, stage):
        return [r for r in self.routes if r.source == stage]


def _output_scale(name, mass_scale, h_scale):
    tail = name.rsplit(".", 1)[-1]
    if tail == "T" or tail.startswith("T0") or tail.startswith("T5") \
            or tail.startswith("T9") or tail.endswith("_T") or tail == "T_max":
        return 50.0
    if tail == "rate":
        return mass_scale
    if tail == "duty":
        return h_scale
    if tail == "P":
        return 1000.0
    return 1.0


class StageColumn:
    """Solver for a :class:`ColumnLayout` with a set of :class:`StageSpec`.

    Args:
        layout: The stage network.
        n_components: Number of pseudocomponents.
        knob_names: Every knob the layout reads.
        specs: The specs (each replaces a rate equation or frees a knob).
        max_iter: Newton iteration limit.
        tol: Convergence test on the largest scaled residual.
        homotopy_steps: Passes of the stage-efficiency homotopy (equilibrium
            stages first, then the Murphree efficiencies in equal steps).
    """

    def __init__(self, layout: ColumnLayout, n_components: int,
                 knob_names: Sequence[str], specs: Sequence[StageSpec] = (),
                 max_iter: int = 150, tol: float = 1e-10,
                 homotopy_steps: int = 3):
        self.layout = layout
        self.C = n_components
        self.N = layout.n_stages
        self.knob_names = tuple(knob_names)
        self.specs = tuple(specs)
        self.max_iter = max_iter
        self.tol = tol
        self.homotopy_steps = max(int(homotopy_steps), 2)

        self.share_routes = tuple(r.name for r in layout.routes if r.kind == "share")
        slots = {f"{n}.rate" for n in self.share_routes}
        replaced = [s.replaces for s in self.specs]
        if len(set(replaced)) != len(replaced):
            raise ValueError(f"two specs replace the same thing: {replaced}")
        for s in self.specs:
            if s.replaces not in slots and s.replaces not in self.knob_names:
                raise ValueError(
                    f"spec on {s.output!r} replaces {s.replaces!r}, which is "
                    f"neither a draw rate ({sorted(slots)}) nor a knob "
                    f"({sorted(self.knob_names)})")
        self.slot_spec = {s.replaces: s for s in self.specs if s.replaces in slots}
        self.freed = tuple(s.replaces for s in self.specs
                           if s.replaces not in slots)
        self.free_specs = tuple(s for s in self.specs if s.replaces not in slots)
        for r in layout.routes:
            kinds = [q.kind for q in layout.routes_from(r.source)]
            if kinds.count("reference") != 1:
                raise ValueError(f"stage {r.source} needs exactly one reference route")
        self._wanted = frozenset(s.output for s in self.specs)
        self._caps = dict(layout.knob_caps)
        self._kscale = dict(layout.knob_scales)

    # -- packing ---------------------------------------------------------

    @property
    def n_vars(self):
        C, N = self.C, self.N
        return 2 * C + 2 * N * C + N + len(self.share_routes) + len(self.freed)

    def unpack(self, x):
        C, N = self.C, self.N
        i = 0

        def take(n):
            nonlocal i
            out = x[i:i + n]
            i += n
            return out

        llF, lvF = take(C), take(C)
        ll = take(N * C).reshape(N, C)
        lv = take(N * C).reshape(N, C)
        T = take(N)
        z = take(len(self.share_routes))
        k = take(len(self.freed))
        return llF, lvF, ll, lv, T, z, k

    def pack(self, llF, lvF, ll, lv, T, z, k):
        return jnp.concatenate([llF, lvF, ll.ravel(), lv.ravel(), T, z, k])

    def _caps_vector(self):
        C, N = self.C, self.N
        caps = [jnp.full(2 * C + 2 * N * C, 5.0), jnp.full(N, 30.0),
                jnp.full(len(self.share_routes), 3.0)]
        caps.append(jnp.asarray([self._caps.get(n, jnp.inf) / self._kscale.get(n, 1.0)
                                 for n in self.freed], dtype=float))
        return jnp.concatenate(caps)

    # -- the model -------------------------------------------------------

    @staticmethod
    def feed_of(th):
        """Component feed (mol/s), floored at 1e-30 of the total.

        The balances are in log form, so a pseudocomponent absent from the
        feed would put ``ln 0`` into them; the floor is a mass error of one
        part in 1e30.
        """
        f = th["feed"]
        return jnp.maximum(f, 1e-30 * jnp.sum(f))

    def knobs_of(self, x, th):
        knobs = dict(th["knobs"])
        k = self.unpack(x)[-1]
        for i, name in enumerate(self.freed):
            knobs[name] = k[i] * self._kscale.get(name, 1.0)
        return knobs

    def route_weights(self, z, knobs):
        """Fraction of each stage's liquid on each route, as {name: scalar}."""
        zmap = dict(zip(self.share_routes, [z[i] for i in range(len(self.share_routes))]))
        weights = {}
        for j in range(self.N):
            rs = self.layout.routes_from(j)
            fixed = [r for r in rs if r.kind == "fixed"]
            rem = 1.0
            for r in fixed:
                f = r.fraction(knobs)
                weights[r.name] = f
                rem = rem - f
            group = [r for r in rs if r.kind == "reference"] + \
                    [r for r in rs if r.kind == "share"]
            logits = jnp.stack([jnp.zeros(())] + [zmap[r.name] for r in group[1:]])
            sm = jax.nn.softmax(logits)
            for r, w in zip(group, sm):
                weights[r.name] = rem * w
        return weights

    def context(self, x, th):
        """Every intermediate quantity of the model at ``x``, as a dict."""
        lay, N = self.layout, self.N
        comps: PseudoComponents = th["components"]
        thermo = ColumnThermo(comps)
        llF, lvF, ll, lv, T, z, _ = self.unpack(x)
        knobs = self.knobs_of(x, th)
        f = self.feed_of(th)
        P = lay.pressures(knobs)
        T_fo = knobs["furnace.T"]
        P_fo = lay.furnace_pressure(knobs)
        S_inj = lay.steam_injection(knobs)
        S_coil = lay.coil_steam(knobs)
        fs = lay.feed_stage
        # steam leaving each stage upward
        S_up = jnp.cumsum(S_inj[::-1])[::-1] + jnp.where(jnp.arange(N) <= fs, S_coil, 0.0)

        lF, vF = jnp.exp(llF), jnp.exp(lvF)
        l, v = jnp.exp(ll), jnp.exp(lv)
        weights = self.route_weights(z, knobs)
        flows = {r.name: weights[r.name] * l[r.source] for r in lay.routes}
        return dict(
            th=th, comps=comps, thermo=thermo, knobs=knobs, f=f, P=P, T=T,
            T_fo=T_fo, P_fo=P_fo, S_inj=S_inj, S_coil=S_coil, S_up=S_up,
            llF=llF, lvF=lvF, ll=ll, lv=lv, lF=lF, vF=vF, l=l, v=v, z=z,
            weights=weights, flows=flows,
            mass_scale=jnp.sum(f * comps.MW) / 1000.0,
            h_scale=jnp.sum(f) * 3e4,
        )

    def log_inflows(self, ctx):
        """ln of the component flow into each stage (liquid + vapor), (N, C)."""
        lay, N = self.layout, self.N
        terms = [[] for _ in range(N)]
        for r in lay.routes:
            if r.dest is not None:
                lw = jnp.log(jnp.maximum(ctx["weights"][r.name], 1e-300))
                terms[r.dest].append(lw + ctx["ll"][r.source])
        for j in range(N - 1):
            terms[j].append(ctx["lv"][j + 1])
        terms[lay.feed_stage].append(jnp.logaddexp(ctx["llF"], ctx["lvF"]))
        return jnp.stack([jax.scipy.special.logsumexp(jnp.stack(t), axis=0)
                          for t in terms])

    def energy_terms(self, ctx):
        """Stage energy residuals (W) for given ctx, before scaling."""
        lay, N = self.layout, self.N
        th = ctx["thermo"]
        T, knobs = ctx["T"], ctx["knobs"]
        hL = th.h_liquid(T)                 # (N, C)
        hV = th.h_vapor(T)
        hS = th.h_steam(T)                  # (N,)
        out = (jnp.sum(ctx["l"] * hL, 1) + jnp.sum(ctx["v"] * hV, 1)
               + ctx["S_up"] * hS)
        inn = jnp.zeros(N)
        for r in lay.routes:
            if r.dest is None:
                continue
            h = jnp.sum(ctx["flows"][r.name] * hL[r.source])
            if r.duty is not None:
                h = h - knobs[r.duty]
            inn = inn.at[r.dest].add(h)
        up = jnp.sum(ctx["v"][1:] * hV[1:], 1) + ctx["S_up"][1:] * hS[1:]
        inn = inn.at[:-1].add(up)
        T_fo = ctx["T_fo"]
        h_feed = (jnp.sum(ctx["lF"] * th.h_liquid(T_fo))
                  + jnp.sum(ctx["vF"] * th.h_vapor(T_fo))
                  + ctx["S_coil"] * th.h_steam(T_fo))
        inn = inn.at[lay.feed_stage].add(h_feed)
        inn = inn + ctx["S_inj"] * th.h_steam(knobs["steam.T"])
        return inn - out

    def residual(self, x, th):
        ctx = self.context(x, th)
        C = self.C
        f = ctx["f"]
        thermo = ctx["thermo"]

        # Component balances in LOG form: ln(out) - ln(sum of ins), with the
        # sum taken by logsumexp over log terms. A residue lump in the top
        # stage is 1e-200 of its feed or less; in linear form its flows
        # underflow, its balance rows go to zero and the Jacobian is
        # singular. In log form every row is O(1) and exact, and a converged
        # log balance is a relative balance, stage by stage.
        rF_bal = jnp.logaddexp(ctx["llF"], ctx["lvF"]) - jnp.log(f)
        LF, VF = jnp.sum(ctx["lF"]), jnp.sum(ctx["vF"]) + ctx["S_coil"]
        rF_eq = (ctx["lvF"] - jnp.log(VF) - thermo.log_K(ctx["T_fo"], ctx["P_fo"])
                 - ctx["llF"] + jnp.log(LF))
        r_bal = jnp.logaddexp(ctx["ll"], ctx["lv"]) - self.log_inflows(ctx)

        # stages: equilibrium (Murphree on the vapor)
        L = jnp.sum(ctx["l"], 1)
        V = jnp.sum(ctx["v"], 1) + ctx["S_up"]
        lnK = thermo.log_K(ctx["T"], ctx["P"])
        lnx = ctx["ll"] - jnp.log(L)[:, None]
        y_in = jnp.concatenate([ctx["v"][1:] / V[1:, None], jnp.zeros((1, C))])
        eta = ctx["th"]["eta"]
        a = jnp.log(eta)[:, None] + lnK + lnx
        b = (jnp.log(jnp.maximum(1.0 - eta, 1e-300))[:, None]
             + jnp.log(jnp.maximum(y_in, 1e-300)))
        r_eq = ctx["lv"] - jnp.log(V)[:, None] - jnp.logaddexp(a, b)

        r_E = self.energy_terms(ctx) / ctx["h_scale"]

        outs = self.outputs_from(ctx, want=self._wanted)
        r_slot = []
        for name in self.share_routes:
            slot = f"{name}.rate"
            spec = self.slot_spec.get(slot)
            if spec is None:
                r_slot.append((outs[slot] - ctx["knobs"][slot]) / ctx["mass_scale"])
            else:
                r_slot.append(self._spec_residual(spec, outs, ctx))
        r_free = [self._spec_residual(s, outs, ctx) for s in self.free_specs]
        return jnp.concatenate([rF_bal, rF_eq, r_bal.ravel(), r_eq.ravel(), r_E,
                                jnp.asarray(r_slot, dtype=float).reshape(-1),
                                jnp.asarray(r_free, dtype=float).reshape(-1)])

    def _spec_residual(self, spec, outs, ctx):
        if spec.output not in outs:
            raise KeyError(f"no output {spec.output!r}; have {sorted(outs)}")
        sc = spec.scale or _output_scale(spec.output, ctx["mass_scale"], ctx["h_scale"])
        return (outs[spec.output] - ctx["th"]["targets"][spec.output]) / sc

    # -- outputs ---------------------------------------------------------

    def product_mass(self, ctx):
        """Hydrocarbon mass flows (kg/s per component) of every product."""
        MW = ctx["comps"].MW
        prods = {r.name: ctx["flows"][r.name] * MW / 1000.0
                 for r in self.layout.routes if r.dest is None}
        prods["overhead"] = ctx["v"][0] * MW / 1000.0
        return prods

    def outputs_from(self, ctx, want=None):
        """Every column output (or, with ``want``, at least those named).

        The residual asks only for what its specs read, so a TBP point or a
        pumparound return temperature nobody specified stays out of the
        Jacobian the Newton loop compiles.
        """
        lay = self.layout
        comps, thermo, knobs = ctx["comps"], ctx["thermo"], ctx["knobs"]
        MW = comps.MW
        T = ctx["T"]
        out = {"top.T": T[0], "furnace.T": ctx["T_fo"], "furnace.P": ctx["P_fo"]}
        for j in range(self.N):
            out[f"stage{j}.T"] = T[j]
            out[f"stage{j}.P"] = ctx["P"][j]
        for r in lay.routes:
            out[f"{r.name}.rate"] = jnp.sum(ctx["flows"][r.name] * MW) / 1000.0
        for name, m in self.product_mass(ctx).items():
            out[f"{name}.rate"] = jnp.sum(m)
            if want is not None and not any(w.startswith(name + ".T") for w in want):
                continue
            props = product_properties(comps, m)
            for key in ("T05", "T10", "T50", "T90", "T95"):
                out[f"{name}.{key}"] = props[key]
        hL_all = thermo.h_liquid(T)
        for r in lay.routes:
            if r.duty is None:
                continue
            out[f"{r.name}.duty"] = knobs[r.duty]
            if want is not None and f"{r.name}.return_T" not in want:
                continue
            fl = ctx["flows"][r.name]
            h_ret = jnp.sum(fl * hL_all[r.source]) - knobs[r.duty]
            out[f"{r.name}.return_T"] = self._liquid_T(thermo, fl, h_ret, T[r.source])
        # furnace
        T_fo = ctx["T_fo"]
        th_ = ctx["th"]
        h_out = (jnp.sum(ctx["lF"] * thermo.h_liquid(T_fo))
                 + jnp.sum(ctx["vF"] * thermo.h_vapor(T_fo))
                 + ctx["S_coil"] * thermo.h_steam(T_fo))
        h_in = (jnp.sum(ctx["f"] * thermo.h_liquid(th_["feed_T"]))
                + ctx["S_coil"] * thermo.h_steam(knobs["steam.T"]))
        out["furnace.duty"] = h_out - h_in
        mF = jnp.sum(ctx["f"] * MW)
        out["furnace.vapor_fraction"] = jnp.sum(ctx["vF"] * MW) / mF
        out["feed.rate"] = mF / 1000.0
        out["steam.rate"] = (jnp.sum(ctx["S_inj"]) + ctx["S_coil"]) * MW_WATER / 1000.0
        if lay.extra_outputs is not None:
            out.update(lay.extra_outputs(ctx, out))
        return out

    @staticmethod
    def _liquid_T(thermo, flows, h_target, T0):
        """Temperature at which liquid ``flows`` has enthalpy ``h_target``."""
        T = T0
        for _ in range(8):
            h, dh = jax.jvp(lambda t: jnp.sum(flows * thermo.h_liquid(t)), (T,),
                            (jnp.ones_like(T),))
            T = T - (h - h_target) / dh
        return T

    def outputs(self, x, th):
        return self.outputs_from(self.context(x, th))

    # -- initialization ----------------------------------------------------

    def _furnace_flash(self, th, knobs):
        """Isothermal flash of the feed at the furnace outlet, by bisection."""
        thermo = ColumnThermo(th["components"])
        f = self.feed_of(th)
        K = jnp.exp(thermo.log_K(knobs["furnace.T"], self.layout.furnace_pressure(knobs)))
        S = self.layout.coil_steam(knobs)

        def h(lna):
            a = jnp.exp(lna)
            return S + a * jnp.sum(f * (K - 1.0) / (1.0 + K * a))

        def body(_, bounds):
            lo, hi = bounds
            mid = 0.5 * (lo + hi)
            pos = h(mid) > 0
            return (jnp.where(pos, mid, lo), jnp.where(pos, hi, mid))

        lo, hi = jax.lax.fori_loop(0, 200, body, (jnp.asarray(-30.0), jnp.asarray(30.0)))
        a = jnp.exp(0.5 * (lo + hi))
        lF = f / (1.0 + K * a)
        vF = f - lF
        return lF, jnp.maximum(vF, 1e-300 + 0 * vF)

    def initial_guess(self, th):
        lay, N, C = self.layout, self.N, self.C
        knobs = dict(th["knobs"])
        lF, vF = self._furnace_flash(th, knobs)
        ctx0 = dict(th=th, knobs=knobs, lF=lF, vF=vF, comps=th["components"])
        T0, L0, V0, z0 = lay.init_guess(self, ctx0)
        z = jnp.asarray([z0[n] for n in self.share_routes], dtype=float).reshape(-1)
        weights = self.route_weights(z, knobs)
        thermo = ColumnThermo(th["components"])
        P = lay.pressures(knobs)
        S_inj = lay.steam_injection(knobs)
        S_up = (jnp.cumsum(S_inj[::-1])[::-1]
                + jnp.where(jnp.arange(N) <= lay.feed_stage, lay.coil_steam(knobs), 0.0))
        K = jnp.exp(thermo.log_K(T0, P))                       # (N, C)
        A = K * ((V0 + S_up) / L0)[:, None]                    # v = A l

        # one linear system per component: balances with fixed routing
        M = jnp.zeros((C, N, N))
        idx = jnp.arange(N)
        M = M.at[:, idx, idx].set(1.0 + A.T)
        M = M.at[:, idx[:-1], idx[1:]].add(-A.T[:, 1:])
        for r in lay.routes:
            if r.dest is not None:
                M = M.at[:, r.dest, r.source].add(-weights[r.name])
        rhs = jnp.zeros((C, N)).at[:, lay.feed_stage].set(lF + vF)
        l = jnp.linalg.solve(M, rhs[..., None])[..., 0].T     # (N, C)
        l = jnp.maximum(l, 1e-250)
        v = A * l
        k0 = jnp.asarray([knobs[n] / self._kscale.get(n, 1.0) for n in self.freed],
                         dtype=float).reshape(-1)
        x0 = self.pack(jnp.log(lF), jnp.log(vF), jnp.log(l), jnp.log(v), T0, z, k0)

        # a freed pumparound duty starts where its return stage's energy closes
        duty_of = {r.duty: r for r in lay.routes if r.duty is not None}
        if any(n in duty_of for n in self.freed):
            ctx = self.context(x0, th)
            rE = self.energy_terms(ctx)
            k = []
            for n in self.freed:
                if n in duty_of:
                    k.append((ctx["knobs"][n] + rE[duty_of[n].dest])
                             / self._kscale.get(n, 1.0))
                else:
                    k.append(knobs[n] / self._kscale.get(n, 1.0))
            x0 = x0.at[x0.size - len(self.freed):].set(jnp.asarray(k))
        return x0

    # -- solve -------------------------------------------------------------

    def _trace(self, x, th):
        """Log-flow entries that are a negligible part of their component.

        Such a flow (a residue lump in the overhead, a light cut in the
        residue) often wants to move by tens of log units -- harmlessly,
        since it carries no mass. Left in the step cap it would scale every
        Newton step to nothing, so it is exempt from the cap and clipped on
        its own instead (falling freely, rising at most 5 per step); only
        the flows that carry mass limit the step as a whole.
        """
        C, N = self.C, self.N
        n_log = 2 * C + 2 * N * C
        lnf = jnp.log(self.feed_of(th))
        lnf_all = jnp.concatenate([lnf, lnf, jnp.tile(lnf, 2 * N)])
        trace = x[:n_log] < lnf_all + jnp.log(1e-7)
        return jnp.concatenate([trace, jnp.zeros(x.size - n_log, dtype=bool)])

    def newton(self, x0, th):
        """Damped Newton from ``x0`` (no gradients flow through this).

        Returns ``(x, J, iterations, max scaled residual)`` with ``J`` the
        Jacobian *at the returned* ``x``: each pass evaluates the Jacobian
        (and the residual, as its aux) first and stops before stepping once
        the residual is inside ``tol``, so the last Jacobian is the one the
        implicit-function step needs and is not traced a second time.
        """
        th = jax.lax.stop_gradient(th)
        x0 = jax.lax.stop_gradient(x0)
        R = lambda x: self.residual(x, th)
        RJ = jax.jacfwd(lambda x: (R(x), R(x)), has_aux=True)
        caps = self._caps_vector()
        n = x0.size

        def cond(s):
            x, J, it, rn, done = s
            return ~done

        def body(s):
            x, _, it, _, _ = s
            J, r = RJ(x)
            rn = jnp.max(jnp.abs(r))
            done = (rn < self.tol) | (it >= self.max_iter)

            dx = jnp.linalg.solve(J, -r)
            dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
            trace = self._trace(x, th)
            cap = jnp.where(trace, jnp.inf, caps)
            a0 = jnp.minimum(1.0, jnp.min(cap / jnp.maximum(jnp.abs(dx), 1e-300)))
            m0 = 0.5 * jnp.sum(r * r)

            def step(a):
                return x + jnp.where(trace, jnp.clip(a * dx, -60.0, 5.0), a * dx)

            def ls_cond(t):
                a, m, k = t
                return (k < 30) & ~(m <= (1.0 - 1e-4 * a) * m0)

            def ls_body(t):
                a, m, k = t
                a = 0.5 * a
                rr = R(step(a))
                mm = 0.5 * jnp.sum(rr * rr)
                return (a, jnp.where(jnp.isfinite(mm), mm, jnp.inf), k + 1)

            # starts at 2*a0 with m = inf, so the first pass evaluates a0
            a, _, _ = jax.lax.while_loop(ls_cond, ls_body, (2.0 * a0, jnp.inf, 0))
            xn = jnp.where(done, x, step(a))
            return (xn, J, it + jnp.where(done, 0, 1), rn, done)

        x, J, it, rn, _ = jax.lax.while_loop(
            cond, body, (x0, jnp.zeros((n, n)), 0, jnp.inf, False))
        return x, J, it, rn

    def solve(self, th, x0=None):
        """Solve; returns ``(x, iterations, max scaled residual)``.

        ``x`` carries implicit-function gradients with respect to ``th``:
        it is one Newton step from the converged point with the Jacobian
        frozen, ``x* - J^-1 R(x*, th)``, whose value is ``x*`` and whose
        derivative is ``-J^-1 dR/dth``.
        """
        th_s = jax.lax.stop_gradient(th)
        if x0 is None:
            x0 = self.initial_guess(th_s)

        # Homotopy on the stage efficiencies: the initial guess is built for
        # equilibrium stages, so solve those first and walk the Murphree
        # efficiencies to their values. With every efficiency at 1 the later
        # passes start converged and cost one Jacobian each.
        eta = th_s["eta"]
        n_h = self.homotopy_steps

        def pass_(k, carry):
            x, _, it, _ = carry
            lam = k / (n_h - 1)
            th_k = dict(th_s, eta=1.0 + lam * (eta - 1.0))
            x, J, it_k, rn = self.newton(x, th_k)
            return x, J, it + it_k, rn

        n = x0.size
        xs, J, it, rn = jax.lax.fori_loop(
            0, n_h, pass_, (x0, jnp.zeros((n, n)), 0, jnp.asarray(jnp.inf)))
        xs, J = jax.lax.stop_gradient(xs), jax.lax.stop_gradient(J)
        x = xs - jnp.linalg.solve(J, self.residual(xs, th))
        return x, it, rn

    def output_names(self, th):
        """Names of every output (evaluated at the initial guess)."""
        return sorted(self.outputs(self.initial_guess(th), th))
