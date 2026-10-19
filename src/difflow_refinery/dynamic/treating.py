"""A dynamic amine unit: the solvent inventory carries the lean loading around the loop.

In the steady amine unit (:func:`difflow_refinery.treating.solve_amine_unit`)
the lean solvent's loadings and the regenerator's hot-end temperature are a
tear, closed by Newton. What makes the real unit slow is the solvent
inventory: the absorber sumps, the rich flash drum, the regenerator's sump
and the lean surge tank hold tens of minutes of circulation, so a sour-gas
upset reaches the regenerator, and the regenerated solvent comes back to
the contactors, only after that inventory has turned over.

:class:`DynamicAmineUnit` keeps that inventory and makes the tear its state:

* **Lean inventory** (regenerator sump + lean surge tank, ``tau_lean`` of
  circulation): its CO2 and H2S, ``n_L``. The contactors receive the
  solvent at the inventory's loading ``alpha = n_L / A_L`` (``A_L`` the
  amine it holds); the regenerator returns its lean solvent to it::

      dn_L/dt = (acid gas in the regenerated lean) - L(t) alpha

* **Rich inventory** (absorber sumps + flash drum, ``tau_rich``): its CO2,
  H2S and water ``n_R``; the absorbers' rich solvent enters it and the flash
  and regenerator receive it at its composition.
* **Regenerator hot end**: the lean/rich exchanger's hot-end temperature
  (the third tear variable) relaxes to the regenerator's lean temperature
  with ``tau_T`` (the reboiler and sump's thermal inventory).

The absorbers, rich flash and regenerator are the steady unit's own
functions (:func:`~difflow_refinery.treating.absorb`,
:func:`~difflow_refinery.treating.unit.rich_flash`,
:func:`~difflow_refinery.treating.regenerate`), evaluated at every time on
the current inventories: their trays settle in a minute or two, inside the
inventory's turnover. The amine itself is conserved (the inventories hold a
fixed amount of it; the circulation moves it). Makeup water holds the lean
solvent at its design strength, as in the steady unit, so the lean
inventory's water is constant; the rich inventory's water is a state.

At rest the inventories' loadings are the steady tear's fixed point: the
rest state is the steady unit's. Species balances (CO2, H2S, water, each
inert) close exactly at every instant: in (sour gases, makeup water) minus
out (sweet gases, flash gas, acid gas) is the inventories' rate of change
(:meth:`DynamicAmineUnit.balance`).

What this leaves out: the absorbers' and regenerator's own holdups (their
tray-by-tray dynamics), the rich solvent's temperature (it follows the
absorbers instantly), and the dissolved inerts' passage (instant).
``tau_lean``, ``tau_rich`` and ``tau_T`` are ILLUSTRATIVE
(:class:`AmineHoldupParams`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.state import StateSpec, StateVar
from difflow_cc.equilibrium import acid_gas as ag
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.treating.absorber import CO2, H2S, WATER, absorb
from difflow_refinery.treating.regenerator import M_W, liquid_enthalpy, regenerate
from difflow_refinery.treating.unit import AmineUnitParams, _heat_to, lean_water, rich_flash, solve_amine_unit

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class AmineHoldupParams:
    """Solvent inventories of the dynamic amine unit, as residence times at design circulation. ILLUSTRATIVE.

    Attributes:
        tau_lean: Lean inventory: regenerator sump and lean surge tank (s).
        tau_rich: Rich inventory: absorber sumps and rich flash drum (s).
        tau_T: Regenerator hot-end thermal lag (s).
    """

    tau_lean: float = 900.0
    tau_rich: float = 600.0
    tau_T: float = 600.0


class DynamicAmineUnit:
    """The amine loop with its solvent inventories (see the module docstring); build with :meth:`from_steady`.

    As a :class:`~difflow.dynamic.DynamicUnit`: ports, one per contactor,
    named after it (its sour gas), and ``"signals"`` (reads
    ``circulation``, a multiplier on every contactor's design amine flow,
    and ``Q_reboiler``, W). Outputs ``"<contactor>_sweet"``,
    ``"flash_gas"``, ``"acid_gas"`` (the SRU feed) and ``"makeup"`` (the
    makeup water, ``F_water``).
    """

    def __init__(self, params: AmineUnitParams, holdup: AmineHoldupParams, y0: Array, scale: Array,
                 name: str = "amine"):
        self.params = params
        self.holdup = holdup
        self.am = ag.ke_amine(params.amine)
        self.names = tuple(c.name for c in params.contactors)
        self.L_design = jnp.asarray([c.circulation for c in params.contactors], dtype=float)
        L = float(jnp.sum(self.L_design))
        self.A_lean = L * holdup.tau_lean       # mol amine held lean
        self.A_rich = L * holdup.tau_rich       # mol amine held rich
        self._y0 = y0
        self._scale = scale
        self.output_ports = tuple(f"{n}_sweet" for n in self.names) + ("flash_gas", "acid_gas", "makeup")
        self.name = name

    # ----- state ------------------------------------------------------------------------

    STATE = ("lean.CO2", "lean.H2S", "rich.CO2", "rich.H2S", "rich.W", "T_hot")

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(k, "generic", "mol" if k != "T_hot" else "K", "") for k in self.STATE])

    def initial_state(self, inputs=None, params=None) -> Array:
        return self._y0

    def unpack(self, state):
        x = state * self._scale
        return {"lean": {"CO2": x[0], "H2S": x[1]}, "rich": {"CO2": x[2], "H2S": x[3], "W": x[4]}, "T_hot": x[5]}

    # ----- the loop at the current inventories --------------------------------------------

    def _run(self, inputs, state):
        p, am = self.params, self.am
        x = self.unpack(state)
        s = inputs["signals"]
        Ls = self.L_design * s["circulation"]
        Lt = jnp.sum(Ls)
        aC, aS = x["lean"]["CO2"] / self.A_lean, x["lean"]["H2S"] / self.A_lean
        rich_in = dict(L=0.0, W=0.0, CO2=0.0, H2S=0.0)
        H_rich, dissolved, sweet, abss = 0.0, {}, {}, {}
        for c, L in zip(p.contactors, Ls):
            lean = dict(L=L, W=lean_water(am, L, p.w_amine), CO2=L * aC, H2S=L * aS, T=p.T_lean)
            r = absorb(am, inputs[c.name], lean, c.n_stages, c.tray, T=c.T)
            abss[c.name] = r
            sweet[c.name] = r.sweet
            for k in ("L", "W", "CO2", "H2S"):
                rich_in[k] = rich_in[k] + r.rich[k]
            H_rich = H_rich + liquid_enthalpy(am, r.rich["L"], r.rich["W"], r.rich["CO2"], r.rich["H2S"],
                                              r.rich["T"])
            for n, v in r.dissolved.items():
                dissolved[n] = dissolved.get(n, 0.0) + v
        # the rich inventory's outflow: the circulation at the inventory's composition, at the
        # temperature of the absorbers' mixed rich solvent (no thermal holdup on the rich side)
        T_rich = _heat_to(am, rich_in, jnp.asarray(p.T_lean, dtype=float), H_rich)
        rich_out = dict(L=Lt, W=Lt * x["rich"]["W"] / self.A_rich, CO2=Lt * x["rich"]["CO2"] / self.A_rich,
                        H2S=Lt * x["rich"]["H2S"] / self.A_rich, T=T_rich)
        flashed, flash_gas, _ = rich_flash(am, rich_out, dissolved, p.P_flash)
        feed = dict(flashed, T=x["T_hot"] - p.approach)
        reg = regenerate(am, feed, s["Q_reboiler"], p.P_regenerator, p.n_regenerator, p.T_condenser)
        W_design = lean_water(am, Lt, p.w_amine)
        return dict(Ls=Ls, Lt=Lt, aC=aC, aS=aS, rich_in=rich_in, rich_out=rich_out, flash_gas=flash_gas,
                    reg=reg, sweet=sweet, makeup=W_design - reg.lean["W"], absorbers=abss)

    def derivatives(self, t, state, inputs, params=None) -> Array:
        o = self._run(inputs, state)
        x = self.unpack(state)
        lean = o["reg"].lean
        d = jnp.stack([
            lean["CO2"] - o["Lt"] * o["aC"],
            lean["H2S"] - o["Lt"] * o["aS"],
            o["rich_in"]["CO2"] - o["rich_out"]["CO2"],
            o["rich_in"]["H2S"] - o["rich_out"]["H2S"],
            o["rich_in"]["W"] - o["rich_out"]["W"],
            (lean["T"] - x["T_hot"]) / self.holdup.tau_T,
        ])
        return d / self._scale

    def _streams(self, o) -> dict:
        p = self.params
        a = o["reg"].acid_gas
        out = {f"{n}_sweet": o["sweet"][n] for n in self.names}
        out["flash_gas"] = o["flash_gas"]
        out["acid_gas"] = {f"F_{CO2}": a["CO2"], f"F_{H2S}": a["H2S"], f"F_{WATER}": a["water"],
                           "T": a["T"], "P": a["P"]}
        out["makeup"] = {f"F_{WATER}": o["makeup"], "T": jnp.asarray(p.T_lean), "P": jnp.asarray(p.P_regenerator)}
        return out

    def outputs(self, t, state, inputs, params=None) -> dict:
        return self._streams(self._run(inputs, state))

    def report(self, state, inputs) -> dict:
        """Lean and rich loadings (mol/mol amine), the hot-end T, and the loop's duties."""
        o = self._run(inputs, state)
        x = self.unpack(state)
        return {"lean.alpha_CO2": o["aC"], "lean.alpha_H2S": o["aS"],
                "rich.alpha_CO2": x["rich"]["CO2"] / self.A_rich, "rich.alpha_H2S": x["rich"]["H2S"] / self.A_rich,
                "T_hot": x["T_hot"], "regenerator.lean_T": o["reg"].lean["T"]}

    def inventory(self, state) -> dict:
        """Species held (mol): CO2, H2S in both inventories, water in the rich one."""
        x = self.unpack(state)
        return {CO2: x["lean"]["CO2"] + x["rich"]["CO2"], H2S: x["lean"]["H2S"] + x["rich"]["H2S"],
                WATER: x["rich"]["W"]}

    # ----- construction from a steady solve -----------------------------------------------

    @classmethod
    def from_steady(cls, params: AmineUnitParams, gases: Mapping[str, Mapping], result=None,
                    holdup: AmineHoldupParams = AmineHoldupParams(), name: str = "amine") -> "DynamicAmineUnit":
        """The dynamic loop at the steady unit's state (solved here unless ``result`` is given)."""
        res = solve_amine_unit(params, gases) if result is None else result
        L = float(sum(c.circulation for c in params.contactors))
        A_lean, A_rich = L * holdup.tau_lean, L * holdup.tau_rich
        lean, rich = res.lean, res.rich
        x0 = jnp.stack([A_lean * lean["CO2"] / L, A_lean * lean["H2S"] / L, A_rich * rich["CO2"] / L,
                        A_rich * rich["H2S"] / L, A_rich * rich["W"] / L,
                        jnp.asarray(res.regenerator.lean["T"], dtype=float)])
        scale = jnp.maximum(jnp.abs(x0), jnp.asarray([1e-6 * A_lean] * 5 + [100.0]))
        scale = scale.at[5].set(100.0)
        return cls(params, holdup, x0 / scale, scale, name=name)


class DynamicAminePlant:
    """A :class:`DynamicAmineUnit` with its feeds, as a dynamic flowsheet (:meth:`from_steady`).

    Feeds: each contactor's sour gas (the steady one times the
    ``gas_rate`` signal) and ``"signals"``. Signals: ``gas_rate``,
    ``circulation`` (multiplier), ``Q_reboiler`` (W).
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_steady(cls, params: AmineUnitParams, gases: Mapping[str, Mapping], result=None,
                    holdup: AmineHoldupParams = AmineHoldupParams()) -> "DynamicAminePlant":
        res = solve_amine_unit(params, gases) if result is None else result
        unit = DynamicAmineUnit.from_steady(params, gases, res, holdup)
        fs = DynamicFlowsheet(species_order=[])
        for n in unit.names:
            g = dict(gases[n])

            def feed(t, args, g=g):
                s = sig.evaluate(t, args)
                return {k: (v * s["gas_rate"] if k.startswith("F_") else v) for k, v in g.items()}

            fs.add_feed(n, feed)
        fs.add_feed("signals", lambda t, args: sig.evaluate(t, args))
        fs.add_unit(unit, inlets={**{n: n for n in unit.names}, "signals": "signals"},
                    outlets={p: p for p in unit.output_ports})
        base = {"gas_rate": 1.0, "circulation": 1.0, "Q_reboiler": float(params.Q_reboiler)}
        y0 = fs.initial_state(None, sig.signal_args(base))
        return cls(flowsheet=fs, unit=unit, y0=y0, base_signals=base, steady=res, params=params, gases=gases,
                   holdup=holdup)

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

    def _inputs(self, t, args):
        st = self.flowsheet.outputs(jnp.asarray(t), self.y0, None, args)
        return {**{n: st[n] for n in self.unit.names}, "signals": st["signals"]}

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Sweet-gas H2S (ppm mol, each contactor), acid gas H2S and CO2 (mol/s), lean and rich loadings."""
        args = self.args() if args is None else args
        st = self.flowsheet.outputs(jnp.asarray(t), y, None, args)
        inputs = {**{n: st[n] for n in self.unit.names}, "signals": st["signals"]}
        rep = self.unit.report(y, inputs)
        out = {}
        for n in self.unit.names:
            sw = st[f"{n}_sweet"]
            tot = sum(v for k, v in sw.items() if k.startswith("F_"))
            out[f"{n}.H2S_ppm"] = 1e6 * sw.get(f"F_{H2S}", 0.0) / tot
        a = st["acid_gas"]
        out.update({"acid_gas.H2S": a[f"F_{H2S}"], "acid_gas.CO2": a[f"F_{CO2}"], **rep})
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
        out = {}
        for n in self.unit.names:
            sw = r.sweet[n]
            tot = sum(v for k, v in sw.items() if k.startswith("F_"))
            out[f"{n}.H2S_ppm"] = 1e6 * sw.get(f"F_{H2S}", 0.0) / tot
        L = float(sum(c.circulation for c in self.params.contactors))
        out.update({"acid_gas.H2S": r.acid_gas[f"F_{H2S}"], "acid_gas.CO2": r.acid_gas[f"F_{CO2}"],
                    "lean.alpha_CO2": r.lean["CO2"] / L, "lean.alpha_H2S": r.lean["H2S"] / L,
                    "rich.alpha_CO2": r.rich["CO2"] / L, "rich.alpha_H2S": r.rich["H2S"] / L,
                    "T_hot": r.regenerator.lean["T"], "regenerator.lean_T": r.regenerator.lean["T"]})
        return out

    def balance(self, t, y, args: dict | None = None) -> dict:
        """Species balances at ``(t, y)``: ``in - out`` against the inventories' rate of change (mol/s).

        Species: CO2, H2S, water, and the inerts together. In: sour gases,
        makeup water. Out: sweet gases, flash gas, acid gas.
        """
        args = self.args() if args is None else args
        fs = self.flowsheet
        st = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)

        def sp(stream, k):
            return jnp.asarray(stream.get(f"F_{k}", 0.0), dtype=float)

        def inert(stream):
            return sum(v for k, v in stream.items() if k.startswith("F_") and k[2:] not in (CO2, H2S, WATER))

        ins = [st[n] for n in self.unit.names] + [st["makeup"]]
        outs = [st[f"{n}_sweet"] for n in self.unit.names] + [st["flash_gas"], st["acid_gas"]]
        keys = (CO2, H2S, WATER)
        io = jnp.stack([sum(sp(s, k) for s in ins) - sum(sp(s, k) for s in outs) for k in keys]
                       + [sum(inert(s) for s in ins) - sum(inert(s) for s in outs)])
        inv = jax.jvp(lambda yy: jnp.stack([self.unit.inventory(yy)[k] for k in keys]), (y,), (dy,))[1]
        acc = jnp.concatenate([inv, jnp.zeros(1)])
        through = jnp.stack([sum(sp(s, k) for s in ins) for k in keys] + [sum(inert(s) for s in ins)])
        return {"species": keys + ("inerts",), "in_minus_out": io, "accumulation": acc, "through": through}


__all__ = ["AmineHoldupParams", "DynamicAmineUnit", "DynamicAminePlant"]
