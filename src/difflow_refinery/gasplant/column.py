"""A stage-network column on a cubic equation of state: condensers,
reboilers, absorbers.

This is the :mod:`difflow_refinery.vacuum.column` machinery -- log-flow
Naphtali-Sandholm MESH, liquids *routed* between stages by softmax shares,
any output specifiable in place of any knob -- carried over to light ends,
where the K-values depend on both phase compositions and the enthalpies are
the EOS's. What it adds:

Condensers
    ``condenser="total"``: stage 0 is a total condenser. Its liquid is at its
    bubble point, which the stage's equilibrium rows say through a *virtual*
    vapor (``y = K x``, so ``sum K x = 1``); that vapor is not an outflow,
    and since its size is free the stage's energy row is replaced by
    ``ln V_0 = ln L_0``. The condenser duty is then a computed output.
    ``condenser="partial"``: stage 0 is an equilibrium stage whose vapor is
    the ``overhead`` product, with a knob ``condenser.duty`` (heat removed,
    W). ``condenser=None``: stage 0 is a tray, its vapor the ``overhead``
    (an absorber, a stripper, an absorber-deethanizer).

Reboiler
    ``reboiler=True``: the last stage is a partial (kettle) reboiler, an
    equilibrium stage with a knob ``reboiler.duty`` (heat added, W).

Feeds
    Any number of feeds, each on its own stage, each a stream at its own
    ``(T, P)``. A feed is flashed at its own conditions (a two-phase feed
    by :func:`~difflow_refinery.gasplant.thermo.feed_state`, with implicit
    gradients) and enters its stage with that enthalpy. The flash runs once
    per solve, outside the Newton loop.

Pressures
    ``P_j = top.P + j * dP``, two knobs (freeable like any other).

Thermodynamics
    Pluggable (see :mod:`difflow_refinery.gasplant.thermo`): ``ln K`` at the
    stage's own ``(T, P, x, y)``, and molar phase enthalpies. With a Murphree
    efficiency ``eta < 1`` the equilibrium vapor is ``K(x, y) x`` with ``K``
    taken at the actual vapor rather than the (unknown) equilibrium one; the
    approximation vanishes at ``eta = 1`` and is of the order of the
    composition dependence of the vapor fugacity coefficients otherwise.

Solve
    Three passes, each a damped Newton (the vacuum column's, with a
    Levenberg-Marquardt fallback: see :meth:`GasColumn.newton`), then the
    implicit-function step:

    1. *Easy specs, equilibrium stages.* Each user spec is swapped for one
       that is nearly linear in the variables and has a value the initial
       guess can supply -- a draw rate for a rate slot, the reflux ratio for
       a condenser duty, the boilup ratio for a reboiler duty.
    2. *Continuation.* From that solution, every user spec's target and
       every Murphree efficiency are walked from where pass 1 left them to
       their values in ``continuation_steps`` equal steps.
    3. The last step lands on the user's problem; one more Newton step with
       the Jacobian frozen carries the exact ``-J^-1 dR/dtheta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from difflow_refinery.gasplant.components import GasComponents
from difflow_refinery.gasplant.thermo import (
    CubicThermo,
    PR,
    feed_state,
    vapor_pressure_vl,
    wilson_bubble_T,
    wilson_dew_T,
)
from difflow_refinery.vacuum.column import Route, StageColumn, StageSpec

#: 100 F, the temperature of the Reid and the GPA 2140 vapor pressures (K).
T_100F = 310.92777777777775
#: ASTM D323 vapor-to-liquid volume ratio.
RVP_VL_RATIO = 4.0
#: Floor on every component's feed, relative to the column's total feed
#: (see :meth:`GasColumn.feed_flows`).
FEED_FLOOR = 1e-30
#: Floor on an initial-guess log flow (1e-300).
_LOG_TINY = -690.7755278982137


def _log_mmatrix_solve(M, f):
    """``log(M^-1 f)`` for a batch of M-matrices and non-negative ``f``.

    ``M`` is ``(C, N, N)`` with a positive diagonal and non-positive
    off-diagonals, column diagonally dominant (the initial guess's component
    balances), and ``f`` is ``(C, N)``. Gaussian elimination without pivoting
    keeps every off-diagonal of such a matrix non-positive, so eliminating the
    right-hand side and back-substituting only ever ADD non-negative terms:
    done in logs, a heavy cut's flow at the top of a long column comes out as
    e^-600 to full relative accuracy instead of as the round-off of an LU
    solve (which a 1e-12 change of a feed moved by hundreds in the log, and
    with it the whole first Newton pass).
    """
    C, N, _ = M.shape
    idx = jnp.arange(N)
    logneg = lambda a: jnp.log(jnp.maximum(-a, 0.0))

    def elim(k, s):
        U, d = s
        piv = U[:, k, k]
        m = jnp.where(idx > k, U[:, :, k] / piv[:, None], 0.0)          # (C, N) <= 0
        U = U - m[:, :, None] * U[:, k, None, :]
        d = jnp.logaddexp(d, logneg(m) + d[:, k, None])
        return U, d

    U, d = jax.lax.fori_loop(0, N, elim, (M, jnp.log(f)))

    def back(j, x):
        i = N - 1 - j
        terms = jnp.where(idx > i, logneg(U[:, i, :]) + x, -jnp.inf)    # (C, N)
        xi = jnp.logaddexp(d[:, i], jax.nn.logsumexp(terms, axis=1)) - jnp.log(U[:, i, i])
        return x.at[:, i].set(xi)

    return jax.lax.fori_loop(0, N, back, jnp.full((C, N), -jnp.inf))


@dataclass(frozen=True)
class Feed:
    """A feed to a column.

    Attributes:
        name: Feed name (the key of its stream in ``th["feeds"]``).
        stage: Stage it enters (0 = top).
    """

    name: str
    stage: int


@dataclass(frozen=True)
class GasColumnLayout:
    """Static description of a gas-plant column (see module docstring).

    Attributes:
        n_stages: Stages, numbered from the top, condenser and reboiler
            included.
        feeds: :class:`Feed` entries.
        routes: Liquid :class:`~difflow_refinery.vacuum.column.Route`
            entries; every stage needs exactly one reference route.
        condenser: ``"total"``, ``"partial"`` or ``None``.
        reboiler: Whether the last stage is a reboiler.
        groups: ``((name, (component, ...)), ...)``: lumps reported as
            ``<product>.x.<name>`` and ``<product>.recovery.<name>``.
        overhead_keys: Components the initial guess sends overhead
            (everything else goes down): a perfect split at the keys.
        reflux_guess: Molar reflux ratio of the initial guess.
    """

    n_stages: int
    feeds: tuple
    routes: tuple
    condenser: Optional[str] = "total"
    reboiler: bool = True
    groups: tuple = ()
    overhead_keys: tuple = ()
    reflux_guess: float = 2.0
    knob_scales: tuple = ()
    knob_caps: tuple = ()

    def routes_from(self, stage):
        return [r for r in self.routes if r.source == stage]

    @property
    def products(self):
        """Names of the products: liquid routes leaving the column, plus
        ``overhead`` when stage 0's vapor leaves."""
        names = [r.name for r in self.routes if r.dest is None]
        if self.condenser != "total":
            names = ["overhead"] + names
        return tuple(names)


def _scale_of(name, ctx):
    tail = name.rsplit(".", 1)[-1]
    parts = name.split(".")
    if tail == "T" or tail.endswith("_T"):
        return 50.0
    if tail == "rate":
        return ctx["mass_scale"]
    if tail == "mol":
        return ctx["mol_scale"]
    if tail == "duty":
        return ctx["h_scale"]
    if tail in ("P", "rvp", "tvp"):
        return 1e4
    if len(parts) >= 3 and parts[-2] in ("x", "recovery"):
        return 1.0
    return 1.0


class GasColumn(StageColumn):
    """Solver for a :class:`GasColumnLayout` with a set of
    :class:`~difflow_refinery.vacuum.column.StageSpec`.

    Args:
        layout: The column.
        component_names: Names of the components, in array order.
        specs: Specs (each replaces a draw-rate equation or frees a knob).
        thermo: ``GasComponents -> thermo`` (see
            :mod:`~difflow_refinery.gasplant.thermo`); default Peng-Robinson.
        max_iter: Newton iteration limit per pass.
        tol: Convergence test on the largest scaled residual.
        continuation_steps: Steps of the spec/efficiency continuation.
    """

    def __init__(self, layout: GasColumnLayout, component_names: Sequence[str],
                 specs: Sequence[StageSpec] = (), thermo: Optional[Callable] = None,
                 max_iter: int = 60, tol: float = 1e-10,
                 continuation_steps: int = 4, _easy: bool = False):
        self.names = tuple(component_names)
        knobs = ["top.P", "dP"]
        if layout.condenser == "partial":
            knobs.append("condenser.duty")
        if layout.reboiler:
            knobs.append("reboiler.duty")
        knobs += [f"{r.name}.rate" for r in layout.routes if r.kind == "share"]
        knobs += [r.duty for r in layout.routes if r.duty is not None]
        if layout.condenser not in ("total", "partial", None):
            raise ValueError(f"condenser must be 'total', 'partial' or None, "
                             f"not {layout.condenser!r}")
        if layout.condenser == "total" and any(f.stage == 0 for f in layout.feeds):
            raise ValueError("a feed cannot enter a total condenser")
        super().__init__(layout, len(self.names), knobs, specs,
                         max_iter=max_iter, tol=tol, homotopy_steps=2)
        self.thermo_factory = thermo or (lambda c: CubicThermo(c, PR))
        self.continuation_steps = max(int(continuation_steps), 1)
        self._groups = {g: tuple(self.names.index(n) for n in members if n in self.names)
                        for g, members in layout.groups}
        self.easy = None if _easy else self._easy_column(max_iter, tol)

    # -- the easy problem --------------------------------------------------

    def _easy_specs(self):
        """Pass-1 specs: one per freed knob of the user's problem."""
        lay = self.layout
        out = []
        for name in self.freed:
            if name == "reboiler.duty":
                out.append(StageSpec("boilup_ratio", 0.0, replaces=name))
            elif name == "condenser.duty":
                out.append(StageSpec("reflux_ratio", 0.0, replaces=name))
        # a total condenser's distillate, if a user spec sets it, is set by
        # the reflux ratio instead: with the boilup ratio that fixes the
        # distillate through the energy balance, which is far better
        # conditioned than fixing the distillate rate and leaving the
        # profile to find the split (a guessed rate at the keys sits on a
        # pinch)
        if lay.condenser == "total" and "distillate.rate" in self.slot_spec:
            out.append(StageSpec("reflux_ratio", 0.0, replaces="distillate.rate"))
        return tuple(out)

    def _easy_column(self, max_iter, tol):
        return GasColumn(self.layout, self.names, self._easy_specs(),
                         thermo=self.thermo_factory, max_iter=max_iter, tol=tol,
                         _easy=True)

    # -- packing -----------------------------------------------------------

    @property
    def n_vars(self):
        return 2 * self.N * self.C + self.N + len(self.share_routes) + len(self.freed)

    def unpack(self, x):
        C, N = self.C, self.N
        i = 0

        def take(n):
            nonlocal i
            out = x[i:i + n]
            i += n
            return out

        ll = take(N * C).reshape(N, C)
        lv = take(N * C).reshape(N, C)
        T = take(N)
        z = take(len(self.share_routes))
        k = take(len(self.freed))
        return ll, lv, T, z, k

    def pack(self, ll, lv, T, z, k):
        return jnp.concatenate([ll.ravel(), lv.ravel(), T, z, k])

    def _caps_vector(self):
        C, N = self.C, self.N
        caps = [jnp.full(2 * N * C, 5.0), jnp.full(N, 20.0),
                jnp.full(len(self.share_routes), 3.0)]
        caps.append(jnp.asarray([self._caps.get(n, 1.0) for n in self.freed],
                                dtype=float).reshape(-1))
        return jnp.concatenate(caps)

    def _kscale_of(self, name, th):
        if name.endswith(".duty"):
            return th["_scale"]["h"]
        if name.endswith(".rate"):
            return th["_scale"]["mass"]
        if name.endswith("P"):
            return 1e5
        return self._kscale.get(name, 1.0)

    def knobs_of(self, x, th):
        knobs = dict(th["knobs"])
        k = self.unpack(x)[-1]
        for i, name in enumerate(self.freed):
            knobs[name] = k[i] * self._kscale_of(name, th)
        return knobs

    # -- feeds -------------------------------------------------------------

    def feed_flows(self, th, name):
        """A feed's component flows, each floored at ``FEED_FLOOR`` times the
        column's total feed.

        The log-flow equations need every component present somewhere: a
        component no feed carries would have ``ln l = -inf`` everywhere. The
        floor (1e-30 of the feed) is added to the feed itself, so the
        equations stay consistent and the balances close on the floored
        feed; what it adds to any product is below 1e-30 of the feed.
        """
        Ft = sum(jnp.sum(th["feeds"][f.name]["flows"]) for f in self.layout.feeds)
        return jnp.maximum(th["feeds"][name]["flows"], FEED_FLOOR * Ft)

    def feed_matrix(self, th):
        """``(N, C)`` component feed to each stage, and the total by component."""
        f = jnp.zeros((self.N, self.C))
        for fd in self.layout.feeds:
            f = f.at[fd.stage].add(self.feed_flows(th, fd.name))
        return f, jnp.sum(f, 0)

    def prepare(self, th):
        """Flash every feed at its own conditions; add scales. Carries
        gradients (it runs once, outside the Newton loop)."""
        thermo = self.thermo_factory(th["components"])
        H, beta = [], []
        for fd in self.layout.feeds:
            s = th["feeds"][fd.name]
            st = feed_state(thermo, s["flows"], s["T"], s["P"])
            H.append(st["H"])
            beta.append(st["beta"])
        _, ftot = self.feed_matrix(th)
        MW = th["components"].MW
        Ft = jnp.sum(ftot)
        scale = {"h": jax.lax.stop_gradient(Ft * 3e4),
                 "mass": jax.lax.stop_gradient(jnp.sum(ftot * MW) / 1000.0),
                 "mol": jax.lax.stop_gradient(Ft)}
        return dict(th, _feedH=jnp.stack(H), _feedbeta=jnp.stack(beta), _scale=scale)

    # -- the model ---------------------------------------------------------

    def context(self, x, th):
        lay, N = self.layout, self.N
        comps: GasComponents = th["components"]
        thermo = self.thermo_factory(comps)
        ll, lv, T, z, _ = self.unpack(x)
        knobs = self.knobs_of(x, th)
        fmat, ftot = self.feed_matrix(th)
        P = knobs["top.P"] + knobs["dP"] * jnp.arange(N)
        l, v = jnp.exp(ll), jnp.exp(lv)
        weights = self.route_weights(z, knobs)
        flows = {r.name: weights[r.name] * l[r.source] for r in lay.routes}
        L = jnp.sum(l, 1)
        V = jnp.sum(v, 1)
        return dict(
            th=th, comps=comps, thermo=thermo, knobs=knobs, fmat=fmat, f=ftot,
            P=P, T=T, ll=ll, lv=lv, l=l, v=v, L=L, V=V, z=z,
            x=l / L[:, None], y=v / V[:, None],
            weights=weights, flows=flows,
            mass_scale=th["_scale"]["mass"], h_scale=th["_scale"]["h"],
            mol_scale=th["_scale"]["mol"],
        )

    def _vapor_out(self):
        """1 where a stage's vapor is a real outflow, 0 for a total condenser."""
        m = jnp.ones(self.N)
        if self.layout.condenser == "total":
            m = m.at[0].set(0.0)
        return m

    def log_inflows(self, ctx):
        lay, N = self.layout, self.N
        terms = [[] for _ in range(N)]
        for r in lay.routes:
            if r.dest is not None:
                lw = jnp.log(jnp.maximum(ctx["weights"][r.name], 1e-300))
                terms[r.dest].append(lw + ctx["ll"][r.source])
        for j in range(N - 1):
            terms[j].append(ctx["lv"][j + 1])
        for fd in lay.feeds:
            terms[fd.stage].append(jnp.log(self.feed_flows(ctx["th"], fd.name)))
        return jnp.stack([jax.scipy.special.logsumexp(jnp.stack(t), axis=0)
                          for t in terms])

    def stage_enthalpies(self, ctx):
        th = ctx["thermo"]
        hL = th.h_liquid(ctx["T"], ctx["P"], ctx["x"])
        hV = th.h_vapor(ctx["T"], ctx["P"], ctx["y"])
        return hL, hV

    def energy_terms(self, ctx, hLV=None):
        """``in - out`` (W) for every stage; for a total condenser, row 0 is
        the heat the condenser must remove (positive)."""
        lay, N = self.layout, self.N
        hL, hV = hLV if hLV is not None else self.stage_enthalpies(ctx)
        knobs = ctx["knobs"]
        out = ctx["L"] * hL + self._vapor_out() * ctx["V"] * hV
        inn = jnp.zeros(N)
        for r in lay.routes:
            if r.dest is None:
                continue
            h = jnp.sum(ctx["flows"][r.name]) * hL[r.source]
            if r.duty is not None:
                h = h - knobs[r.duty]
            inn = inn.at[r.dest].add(h)
        inn = inn.at[:-1].add(ctx["V"][1:] * hV[1:])
        H = ctx["th"]["_feedH"]
        for i, fd in enumerate(lay.feeds):
            inn = inn.at[fd.stage].add(H[i])
        if lay.condenser == "partial":
            inn = inn.at[0].add(-knobs["condenser.duty"])
        if lay.reboiler:
            inn = inn.at[N - 1].add(knobs["reboiler.duty"])
        return inn - out

    def residual(self, x, th):
        ctx = self.context(x, th)
        thermo = ctx["thermo"]
        N, C = self.N, self.C
        vout = self._vapor_out()

        # component balances, log form; a total condenser's vapor is virtual
        out_log = jnp.where(vout[:, None] > 0, jnp.logaddexp(ctx["ll"], ctx["lv"]), ctx["ll"])
        r_bal = out_log - self.log_inflows(ctx)

        # equilibrium (Murphree on the vapor)
        lnK = thermo.log_K(ctx["T"], ctx["P"], ctx["x"], ctx["y"])
        lnx = ctx["ll"] - jnp.log(ctx["L"])[:, None]
        y_in = jnp.concatenate([ctx["y"][1:], jnp.zeros((1, C))])
        eta = ctx["th"]["eta"]
        a = jnp.log(eta)[:, None] + lnK + lnx
        b = (jnp.log(jnp.maximum(1.0 - eta, 1e-300))[:, None]
             + jnp.log(jnp.maximum(y_in, 1e-300)))
        r_eq = ctx["lv"] - jnp.log(ctx["V"])[:, None] - jnp.logaddexp(a, b)

        r_E = self.energy_terms(ctx) / ctx["h_scale"]
        if self.layout.condenser == "total":
            r_E = r_E.at[0].set(jnp.log(ctx["V"][0]) - jnp.log(ctx["L"][0]))

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
        return jnp.concatenate([r_bal.ravel(), r_eq.ravel(), r_E,
                                jnp.asarray(r_slot, dtype=float).reshape(-1),
                                jnp.asarray(r_free, dtype=float).reshape(-1)])

    def _spec_residual(self, spec, outs, ctx):
        if spec.output not in outs:
            raise KeyError(f"no output {spec.output!r}; have {sorted(outs)}")
        sc = spec.scale or _scale_of(spec.output, ctx)
        return (outs[spec.output] - ctx["th"]["targets"][spec.output]) / sc

    # -- outputs -------------------------------------------------------------

    def product_flows(self, ctx):
        """Component molar flows (mol/s) of every product."""
        prods = {}
        if self.layout.condenser != "total":
            prods["overhead"] = ctx["v"][0]
        for r in self.layout.routes:
            if r.dest is None:
                prods[r.name] = ctx["flows"][r.name]
        return prods

    def product_stage(self, name):
        if name == "overhead":
            return 0
        return next(r.source for r in self.layout.routes if r.name == name)

    def outputs_from(self, ctx, want=None):
        """Every column output (or, with ``want``, at least those named).

        Vapor pressures (``<product>.rvp``, ``.tvp``) are a nested solve
        each, so with ``want`` given they are computed only if asked for.
        """
        lay, N = self.layout, self.N
        comps, thermo, knobs = ctx["comps"], ctx["thermo"], ctx["knobs"]
        MW = comps.MW
        T, P = ctx["T"], ctx["P"]
        out = {"top.T": T[0], "bottom.T": T[N - 1], "top.P": P[0], "bottom.P": P[N - 1]}
        for j in range(N):
            out[f"stage{j}.T"] = T[j]
            out[f"stage{j}.P"] = P[j]
        for r in lay.routes:
            out[f"{r.name}.rate"] = jnp.sum(ctx["flows"][r.name] * MW) / 1000.0
            out[f"{r.name}.mol"] = jnp.sum(ctx["flows"][r.name])
        f = ctx["f"]
        prods = self.product_flows(ctx)
        top_mol = 0.0
        for name, fl in prods.items():
            mol = jnp.sum(fl)
            out[f"{name}.mol"] = mol
            out[f"{name}.rate"] = jnp.sum(fl * MW) / 1000.0
            out[f"{name}.T"] = T[self.product_stage(name)]
            xs = fl / mol
            rec = fl / f
            for i, n in enumerate(self.names):
                out[f"{name}.x.{n}"] = xs[i]
                out[f"{name}.recovery.{n}"] = rec[i]
            for g, idx in self._groups.items():
                ii = jnp.asarray(idx, dtype=int)
                out[f"{name}.x.{g}"] = jnp.sum(xs[ii]) if idx else jnp.zeros(())
                out[f"{name}.recovery.{g}"] = (jnp.sum(fl[ii]) / jnp.sum(f[ii])
                                               if idx else jnp.zeros(()))
            if self.product_stage(name) == 0:
                top_mol = top_mol + mol
            if name == "overhead":
                continue
            for kind, ratio in (("rvp", RVP_VL_RATIO), ("tvp", 0.0)):
                key = f"{name}.{kind}"
                if want is not None and key not in want:
                    continue
                out[key] = vapor_pressure_vl(thermo, fl, T_100F, ratio)
        if lay.condenser is not None:
            reflux = sum(jnp.sum(ctx["flows"][r.name]) for r in lay.routes
                         if r.source == 0 and r.dest is not None)
            out["reflux.mol"] = reflux
            out["reflux_ratio"] = reflux / top_mol
            out["condenser.T"] = T[0]
        if lay.reboiler:
            bottoms = sum(jnp.sum(ctx["flows"][r.name]) for r in lay.routes
                          if r.source == N - 1 and r.dest is None)
            out["boilup_ratio"] = ctx["V"][N - 1] / bottoms
            out["reboiler.T"] = T[N - 1]
            out["reboiler.duty"] = knobs["reboiler.duty"]
        need_e = (want is None or "condenser.duty" in want or "energy_balance" in want)
        if lay.condenser == "partial":
            out["condenser.duty"] = knobs["condenser.duty"]
        elif lay.condenser == "total" and need_e:
            out["condenser.duty"] = self.energy_terms(ctx)[0]
        for i, fd in enumerate(lay.feeds):
            out[f"{fd.name}.vapor_fraction"] = ctx["th"]["_feedbeta"][i]
        for r in lay.routes:
            if r.duty is not None:
                out[f"{r.name}.duty"] = knobs[r.duty]
        if need_e:
            # overall energy balance: feeds + reboiler - condenser - products
            hL, hV = self.stage_enthalpies(ctx)
            e = jnp.sum(ctx["th"]["_feedH"])
            if lay.reboiler:
                e = e + knobs["reboiler.duty"]
            if lay.condenser is not None:
                e = e - out["condenser.duty"]
            for r in lay.routes:
                if r.duty is not None:
                    e = e - knobs[r.duty]
            for name, fl in prods.items():
                j = self.product_stage(name)
                e = e - jnp.sum(fl) * (hV[j] if name == "overhead" else hL[j])
            out["energy_balance"] = e
        out["feed.mol"] = jnp.sum(f)
        out["feed.rate"] = jnp.sum(f * MW) / 1000.0
        return out

    # -- initialization ------------------------------------------------------

    def _guess_flows(self, th, reflux=None):
        """Constant-molar-overflow totals and product split for the guess
        (at reflux ratio ``reflux``, default the layout's ``reflux_guess``)."""
        lay, N, C = self.layout, self.N, self.C
        _, ftot = self.feed_matrix(th)
        MW = th["components"].MW
        knobs = th["knobs"]
        top_mask = jnp.asarray([1.0 if n in lay.overhead_keys else 0.0 for n in self.names])
        D_comp = ftot * jnp.where(top_mask > 0, 0.995, 0.005)
        Fmol = jnp.sum(ftot)
        # side draws (liquid products below the top, above the bottom)
        side = {}
        for r in lay.routes:
            if r.dest is None and r.source not in (0, N - 1):
                side[r.name] = knobs.get(f"{r.name}.rate", 0.0) * 1000.0 / (
                    jnp.sum(ftot * MW) / Fmol)
        D = jnp.maximum(jnp.sum(D_comp), 1e-3 * Fmol)
        q = 1.0 - th["_feedbeta"]
        R0 = lay.reflux_guess if reflux is None else reflux
        Lj, Vj = [None] * N, [None] * N
        draws = [0.0] * N
        for name, s in side.items():
            src = next(r.source for r in lay.routes if r.name == name)
            draws[src] = draws[src] + s
        feeds_at = [[] for _ in range(N)]
        for i, fd in enumerate(lay.feeds):
            F = jnp.sum(th["feeds"][fd.name]["flows"])
            feeds_at[fd.stage].append((F, q[i]))
        if lay.condenser == "total":
            Lj[0] = (R0 + 1.0) * D
            Vj[0] = Lj[0]
            draws[0] = D
            Vnext = Lj[0]
        elif lay.condenser == "partial":
            liq_top = [r for r in lay.routes if r.source == 0 and r.dest is None]
            Dv = D if not liq_top else 0.5 * D
            draws[0] = D - Dv
            Lj[0] = R0 * D + draws[0]
            Vj[0] = Dv
            Vnext = Lj[0] - draws[0] + Dv
        else:
            Vj[0] = D
            Lin = sum(F * qq for F, qq in feeds_at[0])
            Lj[0] = jnp.maximum(Lin + 0.0, 0.05 * Fmol)
            Vnext = Vj[0] - sum(F * (1 - qq) for F, qq in feeds_at[0])
        for j in range(1, N):
            Vj[j] = jnp.maximum(Vnext, 0.05 * Fmol)
            Lin = Lj[j - 1] - draws[j - 1]
            Ff = sum(F for F, _ in feeds_at[j])
            if j == N - 1:
                Lj[j] = jnp.maximum(Lin + Ff - Vj[j], 0.05 * Fmol)
            else:
                Lj[j] = jnp.maximum(Lin + sum(F * qq for F, qq in feeds_at[j]), 0.05 * Fmol)
                Vnext = Vj[j] - sum(F * (1 - qq) for F, qq in feeds_at[j])
        return jnp.stack(Lj), jnp.stack(Vj), D_comp, draws, side

    def initial_guess(self, th, reflux=None):
        """Bubble-point (Wang-Henke) passes on Wilson K over CMO flows."""
        lay, N, C = self.layout, self.N, self.C
        knobs = dict(th["knobs"])
        thermo = self.thermo_factory(th["components"])
        fmat, ftot = self.feed_matrix(th)
        P = knobs["top.P"] + knobs["dP"] * jnp.arange(N)
        L0, V0, D_comp, draws, side = self._guess_flows(th, reflux)
        B_comp = ftot - D_comp

        # share-route logits from the guessed flows
        zmap = {}
        for r in lay.routes:
            if r.kind != "share":
                continue
            j = r.source
            if r.dest is None and j == 0:
                g = draws[0]
            elif r.name in side:
                g = side[r.name]
            else:
                g = knobs.get(f"{r.name}.rate", 0.0) * 1000.0 / (
                    jnp.sum(ftot * th["components"].MW) / jnp.sum(ftot))
            zmap[r.name] = jnp.log(jnp.maximum(g, 1e-6 * L0[j])
                                   / jnp.maximum(L0[j] - g, 1e-6 * L0[j]))
        z = jnp.asarray([zmap[n] for n in self.share_routes], dtype=float).reshape(-1)
        weights = self.route_weights(z, knobs)

        T_top = (wilson_bubble_T(thermo, D_comp, P[0]) if lay.condenser == "total"
                 else wilson_dew_T(thermo, D_comp, P[0]))
        T_bot = wilson_bubble_T(thermo, B_comp, P[N - 1])
        T_bot = jnp.maximum(T_bot, T_top + 5.0)
        T = T_top + (T_bot - T_top) * jnp.arange(N) / max(N - 1, 1)
        vout = self._vapor_out()
        idx = jnp.arange(N)

        def sweep(T):
            K = jnp.exp(thermo.log_K_wilson(T, P))                 # (N, C)
            A = K * (V0 / L0)[:, None]
            M = jnp.zeros((C, N, N))
            M = M.at[:, idx, idx].set(1.0 + (A * vout[:, None]).T)
            M = M.at[:, idx[:-1], idx[1:]].add(-A.T[:, 1:])
            for r in lay.routes:
                if r.dest is not None:
                    M = M.at[:, r.dest, r.source].add(-weights[r.name])
            ll = _log_mmatrix_solve(M, fmat.T).T                 # (N, C)
            lv = jnp.log(A) + ll
            ll, lv = jnp.maximum(ll, _LOG_TINY), jnp.maximum(lv, _LOG_TINY)
            return ll, lv

        for _ in range(6):
            ll, _ = sweep(T)
            Tn = wilson_bubble_T(thermo, jnp.exp(ll), P)
            T = jnp.clip(Tn, T - 30.0, T + 30.0)
        ll, lv = sweep(T)
        # freed knobs start from their own values, freed duties at zero; then
        # a freed condenser or reboiler duty is set where its stage's energy
        # balance closes on the guess
        duties = ("reboiler.duty", "condenser.duty")
        k = jnp.asarray([0.0 if n in duties else knobs[n] / self._kscale_of(n, th)
                         for n in self.freed], dtype=float).reshape(-1)
        x0 = self.pack(ll, lv, T, z, k)
        if any(n in duties for n in self.freed):
            rE = self.energy_terms(self.context(x0, th))
            for i, n in enumerate(self.freed):
                if n == "reboiler.duty":
                    k = k.at[i].set(-rE[N - 1] / self._kscale_of(n, th))
                elif n == "condenser.duty":
                    k = k.at[i].set(rE[0] / self._kscale_of(n, th))
            x0 = self.pack(ll, lv, T, z, k)
        return x0

    def _trace_floor(self, x, th):
        """The log flow below which an entry of ``x`` is trace (1e-7 of
        its component's feed); ``-inf`` for the entries that are not flows."""
        C, N = self.C, self.N
        n_log = 2 * N * C
        _, ftot = self.feed_matrix(th)
        lnf = jnp.tile(jnp.log(ftot), 2 * N) + jnp.log(1e-7)
        return jnp.concatenate([lnf, jnp.full(x.size - n_log, -jnp.inf)])

    def _trace(self, x, th):
        return x < self._trace_floor(x, th)

    def newton(self, x0, th):
        """The vacuum column's damped Newton, with a Levenberg-Marquardt
        fallback; returns ``(x, J, iterations, max scaled residual)``.

        Each iteration first tries the Newton step under the same caps and
        Armijo search as :meth:`StageColumn.newton`, and takes it whenever
        the search succeeds within ten halvings -- so a column that solved
        before solves through the same iterates. Otherwise the Jacobian is
        near singular (a stage on the edge of the cubic's three-root region
        put cond(J) near 3e7 in the C3/C4 splitter's pass 1, and the Newton
        step was 1e9 K long) and the step is Levenberg-Marquardt's,
        ``-(J'J + mu diag(J'J))^-1 J'r``, with ``mu`` raised until the merit
        falls and relaxed after each success. ``J`` is the Jacobian at the
        returned ``x``, as the implicit step needs.
        """
        th = jax.lax.stop_gradient(th)
        x0 = jax.lax.stop_gradient(x0)
        R = lambda x: self.residual(x, th)
        RJ = jax.jacfwd(lambda x: (R(x), R(x)), has_aux=True)
        caps = self._caps_vector()
        n = x0.size

        def merit(x):
            r = R(x)
            m = 0.5 * jnp.sum(r * r)
            return jnp.where(jnp.isfinite(m), m, jnp.inf)

        def cond(s):
            return ~s[4]

        def body(s):
            x, _, it, _, _, mu = s
            J, r = RJ(x)
            rn = jnp.max(jnp.abs(r))
            done = (rn < self.tol) | (it >= self.max_iter)
            m0 = 0.5 * jnp.sum(r * r)
            floor = self._trace_floor(x, th)
            trace = x < floor
            cap = jnp.where(trace, jnp.inf, caps)
            # a trace flow may rise to the trace floor in one step: a heavy
            # cut's guess can sit e^-600 under its inflow on a light stage,
            # and the +5 clip alone spends 120 iterations climbing out
            up = jnp.maximum(5.0, floor - x)

            def step(dx, a):
                return x + jnp.where(trace, jnp.clip(a * dx, -60.0, up), a * dx)

            def capped(dx):
                dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
                a0 = jnp.minimum(1.0, jnp.min(cap / jnp.maximum(jnp.abs(dx), 1e-300)))
                return dx, a0

            # Newton under the vacuum column's caps and Armijo search
            dx, a0 = capped(jnp.linalg.solve(J, -r))

            def ls_cond(t):
                a, m, k = t
                return (k < 10) & ~(m <= (1.0 - 1e-4 * a) * m0)

            def ls_body(t):
                a, _, k = t
                a = 0.5 * a
                return (a, merit(step(dx, a)), k + 1)

            a, m_n, _ = jax.lax.while_loop(ls_cond, ls_body, (2.0 * a0, jnp.inf, 0))
            newton_ok = m_n <= (1.0 - 1e-4 * a) * m0

            # Levenberg-Marquardt, with diag(J'J) scaling
            def lm(mu):
                H = J.T @ J
                g = J.T @ r
                d = jnp.diag(H)
                d = jnp.where(d > 0, d, 1.0)

                def trial(mu):
                    dl, al = capped(jnp.linalg.solve(H + mu * jnp.diag(d), -g))
                    xn = step(dl, al)
                    return xn, merit(xn)

                def lm_cond(t):
                    _, m, mu, k = t
                    return (k < 40) & ~(m < m0)

                def lm_body(t):
                    _, _, mu, k = t
                    mu = jnp.where(k == 0, mu, 4.0 * mu)
                    xn, m = trial(mu)
                    return (xn, m, mu, k + 1)

                xn, m, mu, _ = jax.lax.while_loop(lm_cond, lm_body, (x, jnp.inf, mu, 0))
                ok = m < m0
                return jnp.where(ok, xn, x), jnp.where(ok, jnp.maximum(mu / 3.0, 1e-12), mu)

            xn, mu = jax.lax.cond(newton_ok | done,
                                  lambda mu: (step(dx, a), mu), lm, mu)
            xn = jnp.where(done, x, xn)
            return (xn, J, it + jnp.where(done, 0, 1), rn, done, mu)

        x, J, it, rn, _, _ = jax.lax.while_loop(
            cond, body, (x0, jnp.zeros((n, n)), 0, jnp.inf, False, jnp.asarray(1e-6)))
        return x, J, it, rn

    # -- solve ---------------------------------------------------------------

    def _transfer(self, x_from, col_from, th_from, col_to, th_to):
        """Map a solution of ``col_from`` to ``col_to``'s variables."""
        ll, lv, T, z, _ = col_from.unpack(x_from)
        knobs = col_from.knobs_of(x_from, th_from)
        k = jnp.asarray([knobs[n] / col_to._kscale_of(n, th_to) for n in col_to.freed],
                        dtype=float).reshape(-1)
        return col_to.pack(ll, lv, T, z, k), knobs

    def _pass1(self, th_s, th1, R):
        """Pass 1 from the guess at reflux ratio ``R``: easy specs (reflux
        ``R`` and the guess's boilup), guessed rates in every rate slot the
        user's specs replace; returns ``(x, iterations, max residual)``."""
        easy = self.easy
        x0 = self.initial_guess(th_s, R)
        L0g, V0g, _, draws, side = self._guess_flows(th_s, R)
        targets = {}
        for s in easy.specs:
            if s.output == "reflux_ratio":
                targets[s.output] = jnp.asarray(R, dtype=float)
            elif s.output == "boilup_ratio":
                # at least one: the guess's vapor floor (5% of the
                # feed) gives an absorber with a heavy lean oil a
                # boilup of a few percent, which strips nothing and
                # leaves pass 1 on a column with no vapor to speak of
                targets[s.output] = jnp.maximum(V0g[-1] / L0g[-1], 1.0)
        kn = dict(th1["knobs"])
        MWbar = th_s["_scale"]["mass"] / th_s["_scale"]["mol"] * 1000.0
        for slot in self.slot_spec:
            if slot in easy.slot_spec:
                continue
            r = slot[:-len(".rate")]
            src = next(q.source for q in self.layout.routes if q.name == r)
            g = side.get(r, draws[src] if src == 0 else 0.3 * L0g[src])
            kn[slot] = g * MWbar / 1000.0
        th_e = dict(th1, targets=targets, knobs=kn)
        x0e, _ = self._transfer(x0, self, th1, easy, th_e)
        xe, _, it, rn = easy.newton(x0e, th_e)
        x, _ = self._transfer(xe, easy, th_e, self, th_s)
        return x, it, rn

    def solve(self, th, x0=None):
        """Solve; returns ``(x, iterations, max scaled residual)``.

        ``x`` carries implicit-function gradients with respect to ``th``
        (see :class:`~difflow_refinery.vacuum.column.StageColumn`).
        """
        th = self.prepare(th)
        th_s = jax.lax.stop_gradient(th)
        eta = th_s["eta"]
        th1 = dict(th_s, eta=jnp.ones_like(eta))
        if x0 is None:
            easy = self.easy
            if easy is not None:
                # pass 1 from the guess at the layout's reflux; if it does not
                # converge, again from guesses at half and twice that reflux
                # (the C3/C4 splitter on an FCC feed stalls near the cubic's
                # three-root edge from reflux 2 and converges from 1 or 4).
                # A pass 1 that converges is never retried. One while_loop
                # over the refluxes, so _pass1 is traced once: a cond per
                # retry compiled three copies of it, and a jacfwd through
                # the isomerization DIH grew by 2.6 GB (to 13 GB).
                refluxes = self.layout.reflux_guess * jnp.asarray([1.0, 0.5, 2.0])
                shape = jax.eval_shape(self._pass1, th_s, th1, refluxes[0])

                def retry(c):
                    k, _, it, _ = c
                    x, it_k, rn = self._pass1(th_s, th1, refluxes[k])
                    return k + 1, x, it + it_k, rn

                _, x0, it0, _ = jax.lax.while_loop(
                    lambda c: (c[0] < refluxes.size) & ~(c[3] < easy.tol), retry,
                    (0, jnp.zeros(shape[0].shape, shape[0].dtype),
                     jnp.zeros(shape[1].shape, shape[1].dtype),
                     jnp.full(shape[2].shape, jnp.inf, shape[2].dtype)))
            else:
                x0 = self.initial_guess(th_s)
                it0 = 0
        else:
            it0 = 0

        # continuation from pass 1 to the user's targets and efficiencies
        outs1 = self.outputs_from(self.context(x0, th1), want=self._wanted)
        start = {k: outs1[k] for k in th_s["targets"]}
        n_c = self.continuation_steps

        def pass_(k, carry):
            x, _, it, _ = carry
            lam = (k + 1.0) / n_c
            targets = {n: start[n] + lam * (th_s["targets"][n] - start[n]) for n in start}
            th_k = dict(th_s, eta=1.0 + lam * (eta - 1.0), targets=targets)
            x, J, it_k, rn = self.newton(x, th_k)
            return x, J, it + it_k, rn

        n = x0.size
        xs, J, it, rn = jax.lax.fori_loop(
            0, n_c, pass_, (x0, jnp.zeros((n, n)), it0, jnp.asarray(jnp.inf)))
        xs, J = jax.lax.stop_gradient(xs), jax.lax.stop_gradient(J)
        x = xs - jnp.linalg.solve(J, self.residual(xs, th))
        return x, it, rn

    def outputs(self, x, th):
        if "_scale" not in th:
            th = self.prepare(th)
        return self.outputs_from(self.context(x, th))

    def output_names(self, th):
        th = self.prepare(th)
        return sorted(self.outputs(self.initial_guess(th), th))
