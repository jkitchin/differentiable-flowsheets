"""The crude unit with its preheat train: tank to products, solved as one.

The train heats the crude with the column's own products and pumparounds,
so the two are a recycle: the column's product temperatures and pumparound
rates set what the train recovers, and the train sets the furnace inlet
temperature, the preflash drum's split (and with it the column's feed and
its second, vapour feed) and the temperature every pumparound comes back
at. :class:`PreheatedCrudeUnit` closes that loop with a Newton iteration on
a short tear vector:

* the return temperature of every pumparound the train cools,
* the preflash drum's temperature (and its pressure, when the drum is run
  to a vapour fraction), and
* the furnace inlet temperature.

Given the tear, the drum is flashed, the column is solved (its pumparound
return temperatures specified as the tear's), the train is solved against
the column's products and pumparounds, and the train's answers are the new
tear. Each of the three inner solves is Newton with an implicit-function
derivative, and so is the loop around them, so a gradient of anything the
unit reports -- the fired duty, the furnace inlet temperature, the
preflash vapour -- with respect to anything it takes -- a fouling
resistance, an area, the drum pressure, a TBP point of the assay -- costs
one linear solve per level, not a pass through the iterations.

A pumparound the train cools has no trim cooler in this model: its
return temperature is the train's outlet, and the heat it removes from the
column is what the train recovers from it. The column spec that would
otherwise fix it (a duty or a draw-minus-return temperature) is replaced by
a ``pa_return_T`` spec whose value is only the starting guess; give the
pumparound one more spec (its rate, usually) as the second of its two.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow_refinery import products as prod
from difflow_refinery.assay import Assay, characterize, default_cut_points
from difflow_refinery.column import BARREL, CrudeColumn, CrudeColumnParams, CrudeColumnResult, Furnace
from difflow_refinery.preheat.flash import crude_split, liquid_enthalpy
from difflow_refinery.preheat.train import _LNP, PreheatTrain, PreheatTrainParams, PreheatTrainResult
from difflow_refinery.thermo import RHO_WATER_60F, ColumnThermo


@dataclass(frozen=True)
class PreheatedUnitResult:
    """A solved crude unit with its preheat train.

    Attributes:
        column: The column's :class:`~difflow_refinery.column.CrudeColumnResult`.
        train: The train's :class:`~difflow_refinery.preheat.train.PreheatTrainResult`.
        properties: Product properties (yields relative to the tank crude).
        feed: The tank crude (a stream dict); ``F_water`` is the water that
            entered with it (the ``tank_water`` BS&W plus any passed in).
        furnace_inlet_T: Crude temperature into the furnace (K).
        fired_duty: Fuel fired in the furnace (W).
        absorbed_duty: Heat the crude takes up in the furnace (W).
        recovered: Heat the train recovers (W).
        preflash_vapor: Hydrocarbon vapour from the preflash drum (mol/s); 0
            without a drum.
        tear_residual: Largest change of a tear temperature (K) at the
            returned solution.
        converged: The column, the train and the loop all converged.
        iterations: Outer Newton iterations.
    """

    column: CrudeColumnResult
    train: PreheatTrainResult
    properties: dict
    feed: dict
    furnace_inlet_T: Array
    fired_duty: Array
    absorbed_duty: Array
    recovered: Array
    preflash_vapor: Array
    tear_residual: Array
    converged: Array
    iterations: Array

    @property
    def products(self) -> dict:
        return self.column.products

    def table(self) -> str:
        """The train, exchanger by exchanger, then the furnace."""
        t = self.train
        lines = [f"{'exchanger':10s} {'Q MW':>7s} {'crude in':>9s} {'out':>7s} "
                 f"{'hot in':>8s} {'out':>7s} {'UA kW/K':>8s}"]
        for name, e in t.exchangers.items():
            lines.append(f"{name:10s} {float(e['Q']) / 1e6:7.2f} {float(e['T_cold_in']) - 273.15:9.1f} "
                         f"{float(e['T_cold_out']) - 273.15:7.1f} {float(e['T_hot_in']) - 273.15:8.1f} "
                         f"{float(e['T_hot_mixed']) - 273.15:7.1f} {float(e['UA']) / 1e3:8.1f}")
        if t.desalter:
            lines.append(f"desalter {float(t.desalter['T']) - 273.15:.1f} C")
        if t.drum:
            lines.append(f"preflash {float(t.drum['T']) - 273.15:.1f} C, {float(t.drum['P']) / 1e5:.2f} bar, "
                         f"{100 * float(t.drum['vapor_fraction']):.1f} mol% flashed")
        lines.append(f"recovered {float(self.recovered) / 1e6:.1f} MW; furnace inlet "
                     f"{float(self.furnace_inlet_T) - 273.15:.1f} C, fired {float(self.fired_duty) / 1e6:.1f} MW")
        return "\n".join(lines)


jax.tree_util.register_dataclass(
    PreheatedUnitResult,
    data_fields=["column", "train", "properties", "feed", "furnace_inlet_T", "fired_duty",
                 "absorbed_duty", "recovered", "preflash_vapor", "tear_residual", "converged",
                 "iterations"],
    meta_fields=[],
)


class PreheatedCrudeUnit:
    """Tank crude through the preheat train, preflash drum, furnace and column.

    Args:
        assay: The crude's assay (concrete: it fixes the pseudo-components).
        column: The column's :class:`~difflow_refinery.column.CrudeColumnParams`.
            A default :class:`~difflow_refinery.column.Furnace` is added if it
            has none, and with a preflash drum ``vapor_feed_stage`` is set
            from the drum's ``vapor_stage``. Each pumparound the train cools
            needs a ``pa_return_T`` spec (see the module docstring).
        train: The :class:`~difflow_refinery.preheat.train.PreheatTrainParams`.
            Its hot streams' sources are column products (side products or
            ``"residue"``) and pumparounds.
        cut_points, method: As for :class:`~difflow_refinery.unit.CrudeUnit`.
        tol: The loop is converged when no tear temperature moves by more
            than this (K).
        max_iter: Outer Newton iterations.
    """

    def __init__(self, assay: Assay, column: CrudeColumnParams, train: PreheatTrainParams,
                 cut_points=None, method: str | None = None, tol: float = 1e-8, max_iter: int = 30):
        self.assay = assay
        self.cut_points = tuple(default_cut_points(assay) if cut_points is None else cut_points)
        self.method = method
        self.tol, self.max_iter = tol, max_iter
        self.crude = characterize(assay, self.cut_points, method)
        self.thermo = ColumnThermo.from_characterization(self.crude)
        self.column_params = self._column_params(column, train)
        self.train_params = train
        self.column = CrudeColumn(self.column_params, self.thermo)
        self.train = PreheatTrain(train, self.thermo)
        self._hot = self._hot_sources()
        self._jit = None

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    @staticmethod
    def _column_params(column: CrudeColumnParams, train: PreheatTrainParams) -> CrudeColumnParams:
        if column.furnace is None:
            column = replace(column, furnace=Furnace())
        if train.drum is not None:
            stage = train.drum.vapor_stage
            stage = column.feed_stage - 1 if stage is None else stage
            column = replace(column, vapor_feed_stage=stage)
        elif column.vapor_feed_stage is not None:
            raise ValueError("the column has a vapor_feed_stage but the train has no preflash drum")
        return column

    def _hot_sources(self):
        """Per hot stream: ``("pa", k, draw node, spec index)`` or ``("product", name)``."""
        p = self.column_params
        pas = {pa.name: k for k, pa in enumerate(p.pumparounds)}
        products = {p.distillate_name, "residue", *(s.name for s in p.side_products)}
        out = []
        for src in self.train.sources:
            if src in pas:
                k = pas[src]
                idx = [i for i, s in enumerate(p.specs) if s.kind == "pa_return_T" and s.target == src]
                if len(idx) != 1:
                    raise ValueError(
                        f"pumparound {src!r} runs through the train, so its return temperature is "
                        "the train's outlet: give it one pa_return_T spec (its value is the starting "
                        "guess) and one other spec, such as its rate")
                out.append(("pa", k, p.pumparounds[k].draw_stage - 1, idx[0]))
            elif src in products:
                out.append(("product", src))
            else:
                raise ValueError(f"hot stream {src!r} is neither a product nor a pumparound of the column")
        return tuple(out)

    @property
    def has_drum(self) -> bool:
        return self.train_params.drum is not None

    @property
    def vf_mode(self) -> bool:
        return self.train.plan.vf_mode

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def tank_flows(self, rate, basis: Literal["volume", "mass", "mole", "bpd"] = "bpd", crude=None) -> Array:
        """Hydrocarbon flows (mol/s) of the tank crude at a rate."""
        crude = self.crude if crude is None else crude
        if basis == "bpd":
            rate, basis = rate * BARREL / 86400.0, "volume"
        if basis == "volume":
            rate, basis = rate * crude.bulk_sg * RHO_WATER_60F, "mass"
        if basis not in ("mass", "mole"):
            raise ValueError(f"basis must be 'bpd', 'volume', 'mass' or 'mole', not {basis!r}")
        s = crude.stream(rate, T=300.0, P=1e5, basis=basis)
        return jnp.stack([jnp.asarray(s[f"F_{n}"], dtype=float) for n in crude.names])

    def _inputs(self, f, T_tank, thermo, column: CrudeColumnParams | None,
                train: PreheatTrainParams | None, water=0.0):
        th = thermo
        tr_params = self.train_params if train is None else train
        if train is not None and (train.drum is None) != (self.train_params.drum is None):
            raise ValueError("train params must have the same structure as the unit's")
        dummy = {s: (jnp.zeros(th.n_components), jnp.asarray(400.0)) for s in self.train.sources}
        a = self.train._args(f, T_tank, dummy, th, train, water)
        col = self.column
        if column is not None:
            column = self._column_params(column, tr_params)
            col = CrudeColumn(column, th)
            if _structure(column) != _structure(self.column_params):
                raise ValueError("column params must have the same layout and spec kinds as the unit's")
        feed = {f"F_{n}": f[i] for i, n in enumerate(th.names)}
        feed.update(T=jnp.asarray(500.0), P=a["P_furnace"] if self.has_drum else a["P_crude"])
        vapor = {"T": jnp.asarray(400.0)} if self.has_drum else None
        c = col._args(feed, vapor)
        c["thermo"] = th
        return c, a

    def _x0(self, c, a):
        x = [c["specs"][h[3]] for h in self._hot if h[0] == "pa"]
        if self.has_drum:
            x.append(jnp.asarray(440.0))
            if self.vf_mode:
                x.append(_LNP * jnp.log(a["P_drum"]))
        x.append(jnp.asarray(500.0))
        return jnp.stack([jnp.asarray(v, dtype=float) for v in x])

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def _G(self, x, c, a):
        """One pass round the loop: the tear in, the train's answers out."""
        th = a["thermo"]
        n_pa = sum(1 for h in self._hot if h[0] == "pa")
        i = n_pa
        c = dict(c)
        f = a["f"]
        drum = None
        if self.has_drum:
            Td = x[i]
            i += 1
            if self.vf_mode:
                P_d = jnp.exp(x[i] / _LNP)
                i += 1
            else:
                P_d = a["P_drum"]
            fw = a["fw_desalted"] if "fw_desalted" in a else a["fw"]
            drum = crude_split(th, f, fw, Td, P_d)
            c.update(f=drum["liquid"], f_water=jnp.asarray(0.0), fx=drum["vapor"],
                     fx_water=drum["water_vapor"], T_x=Td, P_in=a["P_furnace"])
        else:
            fw = a["fw_desalted"] if "fw_desalted" in a else a["fw"]
            c.update(f=f, f_water=fw, P_in=a["P_crude"])
        T_fi = x[i]
        c.update(T_in=T_fi, T_F=T_fi)
        specs = c["specs"]
        j = 0
        for h in self._hot:
            if h[0] == "pa":
                specs = specs.at[h[3]].set(x[j])
                j += 1
        c["specs"] = specs
        col = self.column._solve_args(c)
        hot_f, hot_T = [], []
        for h in self._hot:
            if h[0] == "pa":
                _, k, node, _ = h
                hot_f.append(col.pumparound_rate[k] * col.x[node])
                hot_T.append(col.T[node])
            else:
                pr = col.products[h[1]]
                hot_f.append(jnp.stack([pr[f"F_{n}"] for n in th.names]))
                hot_T.append(pr["T"])
        a = dict(a, hot_f=hot_f, hot_T=hot_T)
        tr = self.train._core(a, None)
        new = [tr.hot_outlet_T[src] for src, h in zip(self.train.sources, self._hot) if h[0] == "pa"]
        if self.has_drum:
            new.append(tr.drum["T"])
            if self.vf_mode:
                new.append(_LNP * jnp.log(tr.drum["P"]))
        new.append(tr.T_out)
        return jnp.stack(new), (col, tr, c)

    def _core(self, c, a):
        theta = (c, a)
        frozen = jax.lax.stop_gradient(theta)

        def R(x, th):
            return self._G(x, *th)[0] - x

        # Newton on the tear. Each pass round the loop is three Newton solves
        # with their own derivatives, so the outer iteration is written to
        # trace the loop as few times as possible -- the residual and its
        # Jacobian come from one forward-mode pass, and the step is clipped
        # rather than line-searched -- because every trace is compiled.
        RJ = jax.jacfwd(lambda x: (R(x, frozen), R(x, frozen)), has_aux=True)

        def body(carry):
            x, _, _, k = carry
            J, r = RJ(x)
            dx = -jnp.linalg.solve(J, r)
            dx = jnp.where(jnp.isfinite(dx), dx, 0.0)
            dx = dx * jnp.minimum(1.0, 30.0 / jnp.maximum(jnp.max(jnp.abs(dx)), 1e-300))
            return x + dx, J, jnp.max(jnp.abs(r)), k + 1

        def cond(carry):
            _, _, norm, k = carry
            return (k < self.max_iter) & ~(norm <= self.tol)

        x0 = self._x0(*frozen)
        n = x0.shape[0]
        x1, J, _, iters = jax.lax.while_loop(cond, body, (x0, jnp.eye(n), jnp.asarray(jnp.inf), 0))
        # the implicit step, with the Jacobian of the last iteration
        x1 = jax.lax.stop_gradient(x1)
        dz = jnp.linalg.solve(jax.lax.stop_gradient(J), R(x1, theta))
        x = x1 - (dz - jax.lax.stop_gradient(dz))
        x_new, (col, tr, cargs) = self._G(x, c, a)
        tear = jnp.max(jnp.abs(jax.lax.stop_gradient(x_new - x)))
        return x, col, tr, cargs, tear, iters

    def solve(self, rate, T_tank, basis: Literal["volume", "mass", "mole", "bpd"] = "bpd",
              assay: Assay | None = None, column: CrudeColumnParams | None = None,
              train: PreheatTrainParams | None = None, flows=None,
              water=0.0) -> PreheatedUnitResult:
        """Solve the unit.

        Args:
            rate: Crude rate (``basis`` as for :meth:`CrudeUnit.feed
                <difflow_refinery.unit.CrudeUnit.feed>`).
            T_tank: Tank temperature (K).
            assay: A different assay (same light ends) -- for a gradient with
                respect to the assay data.
            column, train: Different params with the same structure: their
                numbers (spec values, steam, areas, fouling resistances,
                bypasses, drum pressure, wash water ...) may be JAX tracers.
            flows: The tank crude's hydrocarbon flows (mol/s), one per
                component, in place of ``rate``/``basis`` -- the flowsheet
                operation's inlet.
            water: Free water arriving with the crude (mol/s), added to the
                train's ``tank_water`` BS&W -- the flowsheet operation's
                inlet ``F_water``.
        """
        if assay is None:
            crude, thermo = self.crude, self.thermo
        else:
            crude = characterize(assay, self.cut_points, self.method)
            if crude.names != self.crude.names:
                raise ValueError("the assay characterises into different components; "
                                 "it needs the unit's light ends")
            thermo = ColumnThermo.from_characterization(crude)
        f = self.tank_flows(rate, basis, crude) if flows is None else jnp.asarray(flows, dtype=float)
        c, a = self._inputs(f, jnp.asarray(T_tank, dtype=float), thermo, column, train, water)
        if self._jit is None:
            self._jit = jax.jit(self._core)
        x, col, tr, cargs, tear, iters = self._jit(c, a)
        feed = {f"F_{n}": f[i] for i, n in enumerate(thermo.names)}
        # The water that actually entered (BS&W plus any passed in), so
        # balances() counts what the train saw rather than recomputing the
        # BS&W alone.
        feed.update(F_water=a["fw"], T=jnp.asarray(T_tank, dtype=float), P=a["P_crude"])
        props = prod.product_properties(col.products, thermo, feed)
        conv = col.converged & tr.converged & (tear <= 100.0 * self.tol)
        return PreheatedUnitResult(
            column=col, train=tr, properties=props, feed=feed,
            furnace_inlet_T=tr.T_out, fired_duty=col.furnace_fired_duty,
            absorbed_duty=col.furnace_duty, recovered=tr.recovered,
            preflash_vapor=jnp.sum(tr.drum["vapor"]) if tr.drum else jnp.asarray(0.0),
            tear_residual=tear, converged=conv, iterations=iters,
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def balances(self, result: PreheatedUnitResult) -> dict:
        """Overall mass and energy balances, tank to products.

        In: the tank crude, wash water, stripping steam, the furnace's
        absorbed duty and any drum duty. Out: brine, the drum's free water,
        every product at the temperature it leaves the unit (through the
        train for those that are cooled in it), the condenser duty and the
        duty of every pumparound the train does not cool.

        Returns:
            ``{"hydrocarbon", "water", "energy"}``: each ``(in, out, relative
            imbalance)``.
        """
        th, col, tr = self.thermo, result.column, result.train
        p = self.column_params
        f_tank = jnp.stack([result.feed[f"F_{n}"] for n in th.names])
        # The water that entered with the crude, as solve() recorded it;
        # recomputing it from the BS&W missed water passed in with the
        # stream (audit, 2026-10).
        fw_tank = result.feed["F_water"]
        steam = jnp.asarray(p.bottom_steam) + sum(jnp.asarray(s.steam) for s in p.side_products
                                                  if s.stripper_stages > 0)
        a = self.train._args(f_tank, result.feed["T"], {s: (f_tank, 400.0) for s in self.train.sources})
        wash = a.get("wash", 0.0)
        brine = tr.desalter.get("brine", 0.0) if tr.desalter else 0.0
        drum_water = tr.drum["free_water"] if tr.drum else 0.0

        hc_in = jnp.sum(f_tank)
        hc_out = sum(jnp.sum(jnp.stack([pr[f"F_{n}"] for n in th.names])) for pr in col.products.values())
        w_in = fw_tank + wash + steam
        w_out = brine + drum_water + sum(pr["F_water"] for pr in col.products.values())

        from difflow_refinery.preheat.flash import crude_enthalpy

        E_in = (crude_enthalpy(th, f_tank, fw_tank, result.feed["T"], a["P_crude"])
                + (wash * th.water_h_liquid(a["T_wash"]) if "wash" in a else 0.0)
                + steam * th.water_h_vapor(jnp.asarray(p.steam_T))
                + col.furnace_duty + (a["drum_duty"] if "drum_duty" in a else 0.0))
        E_out = col.condenser_duty
        if tr.desalter:
            E_out = E_out + brine * th.water_h_liquid(tr.desalter["T"])
        if tr.drum:
            E_out = E_out + drum_water * th.water_h_liquid(tr.drum["T"])
        cooled = {src for src in self.train.sources}
        for name, pr in col.products.items():
            flows = jnp.stack([pr[f"F_{n}"] for n in th.names])
            T = tr.hot_outlet_T[name] if name in cooled else pr["T"]
            if name == "offgas":
                E_out = E_out + jnp.sum(flows * th.h_vapor(T)) + pr["F_water"] * th.water_h_vapor(T)
            else:
                E_out = E_out + liquid_enthalpy(th, flows, T) + pr["F_water"] * th.water_h_liquid(T)
        for k, pa in enumerate(p.pumparounds):
            if pa.name not in cooled:
                E_out = E_out + col.pumparound_duty[k]

        def rel(i, o):
            return (i, o, (i - o) / jnp.maximum(jnp.abs(i), jnp.abs(o)))

        return {"hydrocarbon": rel(hc_in, hc_out), "water": rel(w_in, w_out), "energy": rel(E_in, E_out)}

    def fouling_sensitivity(self, rate, T_tank, basis="bpd", train: PreheatTrainParams | None = None):
        """``d(fired duty)/d(R_f)`` for every exchanger (W per m^2 K/W), by name.

        One reverse-mode gradient through the whole loop.
        """
        tp = self.train_params if train is None else train
        names = [e.name for e in tp.exchangers]

        def fired(rf):
            t = tp.update(exchangers=tuple(e.update(Rf=rf[i]) for i, e in enumerate(tp.exchangers)))
            return self.solve(rate, T_tank, basis, train=t).fired_duty

        rf0 = jnp.asarray([float(e.Rf) for e in tp.exchangers])
        g = jax.grad(fired)(rf0)
        return dict(zip(names, g))

    def cleaning_ranking(self, rate, T_tank, basis="bpd", train: PreheatTrainParams | None = None,
                         exact: bool = True) -> list[dict]:
        """Which exchanger to clean first: the fired duty each cleaning would save.

        For every exchanger, the linear estimate ``dQ_fired/dR_f * R_f`` (one
        gradient for all of them) and, with ``exact``, the saving from
        re-solving with that exchanger clean (``R_f = 0``). The two differ
        by the curvature of ``UA = A/(1/U + R_f)`` and of the train's
        response; the gradient ranks, the re-solve sizes.

        Returns:
            One dict per exchanger -- ``name``, ``Rf``, ``dfired_dRf``,
            ``linear_saving`` (W), ``saving`` (W, with ``exact``) -- sorted
            by the saving, largest first.
        """
        tp = self.train_params if train is None else train
        base = self.solve(rate, T_tank, basis, train=tp)
        sens = self.fouling_sensitivity(rate, T_tank, basis, train=tp)
        rows = []
        for e in tp.exchangers:
            row = {"name": e.name, "Rf": float(e.Rf), "dfired_dRf": float(sens[e.name]),
                   "linear_saving": float(sens[e.name]) * float(e.Rf)}
            if exact:
                clean = self.solve(rate, T_tank, basis, train=tp.with_exchanger(e.name, Rf=0.0))
                row["saving"] = float(base.fired_duty - clean.fired_duty)
            rows.append(row)
        key = "saving" if exact else "linear_saving"
        return sorted(rows, key=lambda r: -r[key])


def _structure(p: CrudeColumnParams):
    """Everything about column params that fixes the shape of its equations."""
    return (p.n_stages, p.feed_stage, p.condenser, p.distillate_name, p.vapor_feed_stage,
            tuple((s.name, s.draw_stage, s.stripper_stages, s.return_stage) for s in p.side_products),
            tuple(p.pumparounds), tuple((s.kind, s.target, s.basis) for s in p.specs),
            p.max_iter, p.tol)


__all__ = ["PreheatedCrudeUnit", "PreheatedUnitResult"]
