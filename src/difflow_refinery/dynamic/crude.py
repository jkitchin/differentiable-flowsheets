"""A dynamic crude unit: furnace lag and product inventories around the steady column.

The crude column's trays settle in minutes; what the plant downstream sees
is slower and comes from the inventories around it: the furnace (tube metal
and refractory) between a coil-outlet setpoint and the crude, and the drums
and strippers each product leaves through -- the overhead reflux drum, the
side strippers' sumps and the column bottoms. :class:`DynamicCrudeUnit`
keeps those and treats the column as quasi-steady:

* **Furnace** (:class:`~difflow_refinery.dynamic.hydroprocessing.FurnaceLag`):
  the coil outlet temperature lags its setpoint (signal ``COT``). The
  column's spec that closes the furnace becomes a coil-outlet-temperature
  spec held by the lag: a column specified by its overflash is re-specified
  with the coil outlet temperature it solved to, which has the same
  solution at rest.
* **Column**: the steady :class:`~difflow_refinery.column.CrudeColumn`,
  solved at every time on the current charge and coil outlet temperature
  (the Newton solve with implicit derivatives). Rate and duty specs follow
  the charge in proportion (ratio control of the draws and pumparounds), so
  a charge change keeps the cut yields.
* **Product inventories**
  (:class:`~difflow_refinery.dynamic.holdup.QuasiSteadyHoldup`): each
  product leaves through a well-mixed holdup ``tau`` (reflux drum, side
  stripper sump, column bottom) -- composition and temperature lag the
  column's.

Out of scope: tray-by-tray column dynamics (tray holdups and hydraulics),
pressure dynamics, and the preheat train (the crude reaches the furnace at
its given temperature). At rest the unit is the steady column. Every
component balance closes exactly: crude and steam in minus products out is
the product inventories' rate of change (:meth:`DynamicCrudeUnit.balance`).

Holdups are ILLUSTRATIVE (:class:`CDUHoldupParams`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.dynamic import DynamicFlowsheet
from difflow_refinery import column as cc
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.holdup import QuasiSteadyHoldup
from difflow_refinery.dynamic.hydroprocessing import FurnaceLag

jax.config.update("jax_enable_x64", True)

#: specs whose value scales with the charge (ratio control)
RATE_SPECS = ("product_rate", "pa_rate", "pa_duty", "furnace_duty")


@dataclass(frozen=True)
class CDUHoldupParams:
    """Furnace lag and product holdups of the dynamic crude unit (s). ILLUSTRATIVE.

    Attributes:
        tau_furnace: Coil-outlet temperature lag (fuel valve, firebox, tube metal).
        tau_distillate: Overhead reflux drum (distillate and free water).
        tau_side: Each side stripper's sump.
        tau_residue: Column bottom.
        tau_offgas: Overhead gas (a partial condenser's).
        tau: Per-product overrides ``{name: s}``.
    """

    tau_furnace: float = 600.0
    tau_distillate: float = 600.0
    tau_side: float = 300.0
    tau_residue: float = 300.0
    tau_offgas: float = 30.0
    tau: Mapping[str, float] = field(default_factory=dict)


class DynamicCrudeUnit:
    """The crude unit with a furnace lag and product inventories, built with :meth:`from_steady`.

    Feeds ``"crude"`` (the steady charge times the ``feed_rate`` signal)
    and ``"signals"``. Signals: ``feed_rate`` (-), ``COT`` (K).
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_steady(cls, column: cc.CrudeColumn, feed: Mapping, result=None,
                    holdup: CDUHoldupParams = CDUHoldupParams()) -> "DynamicCrudeUnit":
        """The dynamic unit of a steady column on ``feed`` (solved here unless ``result`` is given).

        ``column`` needs a furnace (:class:`~difflow_refinery.column.Furnace`).
        """
        res = column.solve(feed) if result is None else result
        p = column.params
        if p.furnace is None:
            raise ValueError("the dynamic crude unit needs a column with a furnace")
        COT0 = float(res.coil_outlet_T)
        # re-specify the furnace by its coil outlet temperature (the lag holds it)
        kinds = [s.kind for s in p.specs]
        if "furnace_T" in kinds:
            specs = p.specs
        else:
            closing = [i for i, k in enumerate(kinds) if k in ("overflash", "furnace_duty")]
            if not closing:
                raise ValueError("no overflash, furnace duty or coil outlet temperature spec closes the furnace")
            specs = tuple(cc.coil_outlet_temperature(COT0) if i == closing[0] else s for i, s in enumerate(p.specs))
        col = cc.CrudeColumn(dataclasses.replace(p, specs=specs), column.thermo)
        i_cot = [s.kind for s in specs].index("furnace_T")
        rate_mask = jnp.asarray([s.kind in RATE_SPECS for s in specs])
        spec0 = jnp.stack([jnp.asarray(s.value, dtype=float) for s in specs])
        names = list(column.thermo.names)
        F_keys = tuple(f"F_{n}" for n in names) + ("F_water",)
        F0 = sum(float(feed[f"F_{n}"]) for n in names)
        T_pre, P_in = jnp.asarray(feed["T"], dtype=float), jnp.asarray(feed["P"], dtype=float)
        lay = p
        products = (lay.distillate_name, *(s.name for s in lay.side_products), "residue", "water")
        if lay.condenser == "partial":
            products = products + ("offgas",)

        def solve_at(fo):
            """The steady column on the furnace outlet ``fo`` (its flows; COT = its T)."""
            fd = {k: fo.get(k, 0.0) for k in F_keys}
            fd.update(T=T_pre, P=P_in)
            ratio = sum(fd[f"F_{n}"] for n in names) / F0
            args = col._args(fd)
            sv = jnp.where(rate_mask, spec0 * ratio, spec0).at[i_cot].set(fo["T"])
            r = col._solve_args(dict(args, specs=sv))
            return r

        def op(fo):
            r = solve_at(fo)
            return tuple(dict(r.products[n]) for n in products)

        tau = {}
        for n in products:
            if n in holdup.tau:
                tau[n] = holdup.tau[n]
            elif n in (lay.distillate_name, "water"):
                tau[n] = holdup.tau_distillate
            elif n == "residue":
                tau[n] = holdup.tau_residue
            elif n == "offgas":
                tau[n] = holdup.tau_offgas
            else:
                tau[n] = holdup.tau_side

        fs = DynamicFlowsheet(species_order=list(F_keys))
        crude0 = dict(feed)

        def crude_feed(t, args):
            s = sig.evaluate(t, args)
            return {k: (v * s["feed_rate"] if k.startswith("F_") else v) for k, v in crude0.items()}

        fs.add_feed("crude", crude_feed)
        fs.add_feed("signals", lambda t, args: sig.evaluate(t, args))
        furnace = FurnaceLag(("crude",), "COT", holdup.tau_furnace, float(P_in), COT0, name="furnace")
        fs.add_unit(furnace, inlets={"crude": "crude", "signals": "signals"}, outlets={"outlet": "furnace_out"})
        fo0 = {k: jnp.asarray(feed.get(k, 0.0), dtype=float) for k in F_keys}
        fo0.update(T=jnp.asarray(COT0), P=P_in)
        prods = QuasiSteadyHoldup.at_steady(op, {"furnace_out": fo0}, ["furnace_out"], list(products), F_keys,
                                            tau, name="column")
        fs.add_unit(prods, inlets={"furnace_out": "furnace_out"}, outlets={n: n for n in products})
        base = {"feed_rate": 1.0, "COT": COT0}
        y0 = fs.initial_state(None, sig.signal_args(base))
        return cls(flowsheet=fs, column=col, steady_column=column, products=products, holdup_unit=prods,
                   furnace=furnace, y0=y0, base_signals=base, steady=res, feed=feed, solve_at=solve_at,
                   F_keys=F_keys, names=names, tau=tau)

    def args(self, steps=None, ramp=sig.DEFAULT_RAMP, **base) -> dict:
        b = dict(self.base_signals)
        b.update(base)
        return sig.signal_args(b, steps, ramp)

    def simulate(self, t_end: float, args: dict | None = None, y0: Array | None = None, n_save: int = 61,
                 saveat: Array | None = None, method: str = "diffrax:tsit5", rtol: float = 1e-6,
                 atol: float = 1e-8, max_steps: int = 100_000, dtmax: float = 120.0, **kwargs):
        """Integrate from ``y0`` (default :attr:`y0`) to ``t_end`` seconds (``kwargs`` to diffrax)."""
        args = self.args() if args is None else args
        y0 = self.y0 if y0 is None else y0
        ts = jnp.linspace(0.0, t_end, n_save) if saveat is None else jnp.asarray(saveat)
        return self.flowsheet.simulate((0.0, float(t_end)), y0=y0, method=method, args=args, rtol=rtol,
                                       atol=atol, saveat=ts, max_steps=max_steps, dtmax=dtmax,
                                       dt0=kwargs.pop("dt0", 1.0), **kwargs)

    def _split(self, y):
        fs = self.flowsheet
        fs._update_state_indices()
        return {e.name: y[e.state_slice] for e in fs.units}

    def _std_volume(self, s) -> Array:
        th = self.column.thermo
        return th.std_volume(jnp.stack([jnp.asarray(s.get(f"F_{n}", 0.0), dtype=float) for n in th.names]))

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Coil outlet T, and each product's standard volume rate (m3/s) and temperature as it leaves its holdup."""
        args = self.args() if args is None else args
        st = self.flowsheet.outputs(jnp.asarray(t), y, None, args)
        out = {"COT": st["furnace_out"]["T"]}
        for n in self.products:
            if n == "water":
                out["water"] = st["water"].get("F_water", 0.0)
                continue
            out[f"{n}.volume"] = self._std_volume(st[n])
            out[f"{n}.T"] = st[n]["T"]
        return out

    def trajectory(self, result, args: dict | None = None, keys: Sequence[str] | None = None) -> dict:
        args = self.args() if args is None else args
        f = jax.jit(lambda t, y, a: self.observe(t, y, a))
        obs = [f(t, y, args) for t, y in zip(result.trajectory.t, result.trajectory.y)]
        keys = keys or list(obs[0])
        out = {k: jnp.stack([o[k] for o in obs]) for k in keys}
        out["t"] = result.trajectory.t
        return out

    def steady_reference(self) -> dict:
        r = self.steady
        out = {"COT": r.coil_outlet_T}
        for n in self.products:
            if n == "water":
                out["water"] = r.products["water"].get("F_water", 0.0)
                continue
            out[f"{n}.volume"] = self._std_volume(r.products[n])
            out[f"{n}.T"] = r.products[n]["T"]
        return out

    def balance(self, t, y, args: dict | None = None) -> dict:
        """Component balances at ``(t, y)``: ``in - out`` against the product holdups' rate of change (mol/s).

        In: the crude and the stripping steam (as water). Out: the products.
        Returns the per-component arrays over :attr:`F_keys` (water last).
        """
        args = self.args() if args is None else args
        fs = self.flowsheet
        st = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)
        us, ds = self._split(y), self._split(dy)
        p = self.column.params
        steam = p.bottom_steam + sum(s.steam for s in p.side_products if s.stripper_stages > 0)
        vec = lambda s: jnp.stack([jnp.asarray(s.get(k, 0.0), dtype=float) for k in self.F_keys])  # noqa: E731
        inflow = vec(st["crude"]).at[-1].add(steam)
        outflow = sum(vec(st[n]) for n in self.products)

        def held(yy):
            inv = self.holdup_unit.inventory(yy)
            return sum(jnp.stack([v[k] for k in self.F_keys]) for v in inv.values())

        acc = jax.jvp(held, (us["column"],), (ds["column"],))[1]
        return {"in_minus_out": inflow - outflow, "accumulation": acc, "through": inflow}


__all__ = ["CDUHoldupParams", "DynamicCrudeUnit", "RATE_SPECS"]
