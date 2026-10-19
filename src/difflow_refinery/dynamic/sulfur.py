"""A dynamic sulfur recovery unit: converter and hydrogenation beds with catalyst heat capacity, TGTU solvent inventory, air-demand control.

The steady SRU (:func:`difflow_refinery.sulfur.solve_sru`) is the Claus train
under air-demand control with the TGTU's acid gas recycled to the furnace (a
tear). The dynamic SRU keeps the parts that hold heat or material long
enough to matter:

* **Catalytic converters** and the **TGTU hydrogenation reactor**:
  :class:`~difflow_refinery.dynamic.beds.DynamicEquilibriumBed` -- the
  steady unit's restricted equilibria (Claus + allotropes with the bed's
  approach to equilibrium; the SCOT hydrogenation/hydrolysis set), taken in
  thermal cells with the catalyst's heat capacity. A Claus bed holds tens of
  minutes of thermal inventory; that is the slow part of an SRU's response
  to an acid-gas or reheat change. A bed's COS/CS2 hydrolysis is the steady
  converter's fixed conversion, applied as the gas enters the bed.
* **Reheaters** ahead of each bed:
  :class:`~difflow_refinery.dynamic.hydroprocessing.FurnaceLag` on the bed
  inlet setpoint (signals ``T_bed1``, ``T_bed2``, ..., ``T_hydrogenation``).
* **The TGTU solvent** (:class:`TGTUSolvent`): the acid gas the absorber
  takes up enters the solvent inventory (absorber sump, regenerator,
  lean tank: ``tau_solvent`` of the absorbed flow at rest) and is stripped
  back to the furnace from it, ``dn/dt = absorbed - n / tau``. The
  inventory turns the steady SRU's recycle tear into a state. As in the
  steady TGTU the regenerator is not solved: the solvent returns at its
  lean loading, the recycle is saturated with water at the reflux drum,
  and makeup water holds the solvent's strength.
* **Air-demand control** (:class:`AirDemandControl`): air =
  ``air_set (1 + u)``, with an integral trim on the tail gas's
  ``(H2S - 2 SO2) / S_in``, ``du/dt = -K_i r``. With ``K_i = 0`` (default)
  the air is the setpoint signal ``air`` (open loop).

The reaction furnace, waste-heat boiler, condensers, quench and absorber are
instantaneous (the steady functions at every time): their gas residence
times are seconds, and their walls' heat is not modelled.

At rest every state is the steady SRU's (the beds at their adiabatic
equilibria, the inventory's stripping rate equal to the absorption, the
air at the steady air demand). Element balances close exactly at every
instant: in (acid gas, air, makeup water, hydrogen) minus out (liquid
sulfur, TGTU off-gas, quench water, the inerts dissolved in the rich
solvent) is the TGTU inventory's rate of change
(:meth:`DynamicSRU.balance`).

Holdups are ILLUSTRATIVE (:class:`SRUHoldupParams`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from difflow import GibbsReactor, GibbsReactorParams
from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.instantaneous import InstantaneousUnit
from difflow.dynamic.state import StateSpec, StateVar
from difflow.water import water_saturation_pressure
from difflow_refinery import thermochemistry as tc
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.beds import NM3_PER_MOL, DynamicEquilibriumBed, catalyst_heat_capacity
from difflow_refinery.dynamic.hydroprocessing import FurnaceLag
from difflow_refinery.sulfur import claus as cl
from difflow_refinery.sulfur import tgtu as tg
from difflow_refinery.treating import absorb, lean_water

jax.config.update("jax_enable_x64", True)

H2S, SO2, CO2, H2O = cl.H2S, cl.SO2, cl.CO2, cl.H2O
_IDX = cl._IDX
CL_KEYS = tuple(f"F_{k}" for k in cl.SPECIES)


@dataclass(frozen=True)
class SRUHoldupParams:
    """Catalyst heat capacities, lags and the TGTU solvent inventory. ALL ILLUSTRATIVE.

    Attributes:
        claus_ghsv: Claus converter gas hourly space velocity (h^-1, wet gas
            at normal conditions) that sizes each bed's alumina.
        claus_density, claus_cp: Alumina bulk density (kg/m3), heat capacity (J/(kg K)).
        hydrogenation_ghsv: TGTU hydrogenation reactor GHSV (h^-1).
        hydrogenation_density, hydrogenation_cp: CoMo catalyst bulk density, heat capacity.
        tau_reheat: Reheater outlet-temperature lag (s).
        tau_solvent: TGTU solvent inventory turnover (s).
        n_cells: Cells per bed.
        air_Ki: Air-demand integral gain (1/s per unit of ratio residual); 0 = open loop.
    """

    claus_ghsv: float = 1000.0
    claus_density: float = 750.0
    claus_cp: float = 900.0
    hydrogenation_ghsv: float = 1500.0
    hydrogenation_density: float = 700.0
    hydrogenation_cp: float = 900.0
    tau_reheat: float = 120.0
    tau_solvent: float = 1200.0
    n_cells: int = 4
    air_Ki: float = 0.0


def _gibbs_equilibrium(reactions, approach=None):
    def eq(n, T, P):
        kw = {} if approach is None else {"reaction_approach_dT": approach}
        out, _ = GibbsReactor(GibbsReactorParams(thermo=cl.GAS, mode="isothermal", T_out=T, reactions=reactions,
                                                 **kw))(cl.to_stream(n, T, P))
        return cl.to_vec(out)
    return eq


def _hydrolysis(x_cos, x_cs2):
    """A Claus bed's COS/CS2 hydrolysis at the steady converter's fixed conversions, as the gas enters."""
    def pre(n, T_in):
        r1, r2 = x_cos * n[_IDX[cl.COS]], x_cs2 * n[_IDX[cl.CS2]]
        d = jnp.zeros_like(n)
        d = d.at[_IDX[cl.COS]].add(-r1).at[_IDX[cl.CS2]].add(-r2)
        d = d.at[_IDX[H2O]].add(-r1 - 2 * r2).at[_IDX[CO2]].add(r1 + r2)
        d = d.at[_IDX[H2S]].add(r1 + 2 * r2)
        return n + d, cl.enthalpy(n, T_in)               # its heat is released into the bed
    return pre


class AirDemandControl:
    """Combustion air with an integral trim on the tail gas's H2S - 2 SO2.

    Ports ``"signals"`` (reads ``air``, mol/s) and ``"tail_gas"``. Output
    ``"air"`` (``{"air": mol/s}``; from the state alone). State ``u``::

        air = air_set (1 + u),   du/dt = -K_i (H2S - 2 SO2)_tail / S_in

    ``S_in`` is the design sulfur feed (a fixed normalisation).
    """

    output_ports = ("air",)
    output_dependencies = {"air": ("signals",)}

    def __init__(self, Ki: float, S_in: float, name: str = "air_control"):
        self.Ki = jnp.asarray(Ki, dtype=float)
        self.S_in = jnp.asarray(S_in, dtype=float)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar("u", "generic", "-", "air trim")])

    def initial_state(self, inputs=None, params=None):
        return jnp.zeros(1)

    def derivatives(self, t, state, inputs, params=None):
        n = cl.to_vec(inputs["tail_gas"])
        r = (n[_IDX[H2S]] - 2.0 * n[_IDX[SO2]]) / self.S_in
        return jnp.reshape(-self.Ki * r, (1,))

    def outputs(self, t, state, inputs, params=None):
        return {"air": {"air": inputs["signals"]["air"] * (1.0 + state[0])}}

    partial_outputs = outputs


class TGTUSolvent:
    """The TGTU's quench-to-absorber and its solvent inventory (see the module docstring).

    Port ``"gas"`` (the hydrogenated gas). Outputs ``"off_gas"`` (to the
    incinerator), ``"quench_water"``, ``"dissolved"`` (inerts leaving in
    the rich solvent), ``"makeup"`` (water) -- these from the inlet -- and
    ``"recycle"`` (the stripped acid gas to the furnace, from the state
    alone, which is what breaks the furnace-TGTU loop). State: the
    inventory's H2S and CO2 (mol).
    """

    output_ports = ("off_gas", "quench_water", "dissolved", "makeup", "recycle")
    output_dependencies = {"off_gas": ("gas",), "quench_water": ("gas",), "dissolved": ("gas",),
                           "makeup": ("gas",), "recycle": ()}

    def __init__(self, p: tg.TGTUParams, tau: float, n0: Sequence[float], name: str = "tgtu_solvent"):
        self.p = p
        self.tau = jnp.asarray(tau, dtype=float)
        self.scale = jnp.maximum(jnp.abs(jnp.asarray(n0, dtype=float)), 1e-6)
        self._y0 = jnp.asarray(n0, dtype=float) / self.scale
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar("H2S", "generic", "mol", ""), StateVar("CO2", "generic", "mol", "")])

    def initial_state(self, inputs=None, params=None):
        return self._y0

    def _absorb(self, gas):
        p = self.p
        q, cond, _ = tg.quench(gas, p.T_quench)
        if p.P_absorber is not None:
            q = dict(q, P=p.P_absorber)
        lean = dict(L=p.L, W=lean_water(p.amine, p.L, p.w_amine), CO2=p.L * p.lean_CO2,
                    H2S=p.L * p.lean_H2S, T=p.T_quench)
        n = cl.to_vec(q)
        g = {f"F_{k}": n[_IDX[k]] for k in cl.SPECIES}
        g.update(T=q["T"], P=q["P"])
        ab = absorb(p.amine, g, lean, p.n_stages, p.tray)
        return q, cond, lean, ab

    def _recycle(self, state):
        p = self.p
        inv = state * self.scale
        r_h2s, r_co2 = inv[0] / self.tau, inv[1] / self.tau
        y = water_saturation_pressure(p.T_reflux) / p.P_reflux
        r_w = (r_h2s + r_co2) * y / (1.0 - y)
        return cl.to_stream(jnp.zeros(len(cl.SPECIES)).at[_IDX[H2S]].set(r_h2s).at[_IDX[CO2]].set(r_co2)
                            .at[_IDX[H2O]].set(r_w), p.T_reflux, p.P_reflux), r_w

    def derivatives(self, t, state, inputs, params=None):
        _, _, lean, ab = self._absorb(inputs["gas"])
        inv = state * self.scale
        absorbed = jnp.stack([ab.rich["H2S"] - lean["H2S"], ab.rich["CO2"] - lean["CO2"]])
        return (absorbed - inv / self.tau) / self.scale

    def outputs(self, t, state, inputs, params=None):
        q, cond, lean, ab = self._absorb(inputs["gas"])
        rec, r_w = self._recycle(state)
        diss = {f"F_{k}": v for k, v in ab.dissolved.items()}
        diss.update(T=q["T"], P=q["P"])
        return {"off_gas": dict(ab.sweet), "recycle": rec,
                "quench_water": {f"F_{H2O}": cond, "T": q["T"], "P": q["P"]},
                "dissolved": diss,
                "makeup": {f"F_{H2O}": lean["W"] - ab.rich["W"] + r_w, "T": q["T"], "P": q["P"]}}

    def partial_outputs(self, t, state, inputs, params=None):
        return {"recycle": self._recycle(state)[0]}

    def inventory(self, state) -> dict:
        inv = state * self.scale
        return {f"F_{H2S}": inv[0], f"F_{CO2}": inv[1]}


class DynamicSRU:
    """The SRU as a dynamic flowsheet, built with :meth:`from_steady`.

    Feeds ``"acid_gas"`` (the steady feed times the ``acid_rate`` signal)
    and ``"signals"``. Signals: ``acid_rate``, ``air`` (mol/s, the air
    setpoint; base the steady air demand), ``T_bed1``... (converter inlets,
    K), ``T_hydrogenation`` (K).
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_steady(cls, acid_gas, claus: cl.ClausParams, tgtu: tg.TGTUParams, result: dict | None = None,
                    holdup: SRUHoldupParams = SRUHoldupParams()) -> "DynamicSRU":
        """The dynamic SRU at the steady :func:`solve_sru` state (solved here unless ``result`` is given)."""
        res = tg.solve_sru(acid_gas, claus, tgtu) if result is None else result
        hp = holdup
        cr, tr = res["claus"], res["tgtu"]
        air0 = float(res["air"])
        S_in = float(cl.sulfur_in(cl.to_vec(acid_gas)))
        fs = DynamicFlowsheet(species_order=list(CL_KEYS))
        ag0 = dict(acid_gas)

        def acid_feed(t, args):
            s = sig.evaluate(t, args)
            return {k: (v * s["acid_rate"] if k.startswith("F_") else v) for k, v in ag0.items()}

        fs.add_feed("acid_gas", acid_feed)
        fs.add_feed("signals", lambda t, args: sig.evaluate(t, args))
        ctl = AirDemandControl(hp.air_Ki, S_in)
        fs.add_unit(ctl, inlets={"signals": "signals", "tail_gas": "tail_gas"}, outlets={"air": "air"})

        def front(inputs, params=None):
            g = dict(inputs["acid_gas"])
            for k in (H2S, CO2, H2O):
                g[f"F_{k}"] = jnp.asarray(g.get(f"F_{k}", 0.0), dtype=float) + inputs["recycle"][f"F_{k}"]
            fout, _ = cl.furnace(g, inputs["air"]["air"], claus.T_air, claus.furnace_approach_dT, claus.furnace_dP)
            w, liq0, _ = cl.cool(fout, claus.T_whb)
            w = dict(w, P=w["P"] - claus.condenser_dP)
            c1, liq1, _ = cl.cool(w, claus.T_condensers[0])
            return {"c1": c1, "sulfur0": {"S": liq0 + liq1}}

        fs.add_unit(InstantaneousUnit(fn=front, inlet_ports=("acid_gas", "recycle", "air"),
                                      output_ports=("c1", "sulfur0"), name="furnace_whb"),
                    inlets={"acid_gas": "acid_gas", "recycle": "recycle", "air": "air"},
                    outlets={"c1": "c1", "sulfur0": "sulfur0"})
        # the steady train's streams, to start the beds and size them
        g = res["claus"]["furnace"]
        g, _, _ = cl.cool(g, claus.T_whb)
        g = dict(g, P=g["P"] - claus.condenser_dP)
        g, _, _ = cl.cool(g, claus.T_condensers[0])
        beds, reheaters, sulfur_streams, bed_flows = [], [], ["sulfur0"], []
        prev = "c1"
        base = {"acid_rate": 1.0, "air": air0}
        for k, cp in enumerate(claus.converters):
            out_k, info_k = cl.converter(g, cp)
            P_in = float(g["P"])
            rh = FurnaceLag((prev,), f"T_bed{k + 1}", hp.tau_reheat, P_in, float(cp.T_in), name=f"reheater{k + 1}")
            fs.add_unit(rh, inlets={prev: prev, "signals": "signals"}, outlets={"outlet": f"bed{k + 1}_in"})
            Nm3_h = float(jnp.sum(cl.to_vec(g))) * NM3_PER_MOL * 3600.0
            C = catalyst_heat_capacity(Nm3_h, hp.claus_ghsv, hp.claus_density, hp.claus_cp)
            bed_flows.append(cl.to_vec(out_k))
            pre = _hydrolysis(*cp.hydrolysis) if any(cp.hydrolysis) else None
            bed = DynamicEquilibriumBed(_gibbs_equilibrium(cl._BED_REACTIONS,
                                                           [cp.approach_dT] + [0.0] * len(cl.ALLOTROPES)),
                                        cl.enthalpy, cl.to_vec, cl.to_stream, C, hp.n_cells, P=P_in,
                                        T0=float(info_k["T_out"]), pre=pre, dP=cp.pressure_drop, name=f"bed{k + 1}")
            fs.add_unit(bed, inlets={"inlet": f"bed{k + 1}_in"}, outlets={"outlet": f"bed{k + 1}_out"})
            T_c = claus.T_condensers[k + 1]

            def condenser(inputs, params=None, *, src=f"bed{k + 1}_out", T_c=T_c):
                w = dict(inputs[src], P=inputs[src]["P"] - claus.condenser_dP)
                c, liq, _ = cl.cool(w, T_c)
                return {"gas": c, "sulfur": {"S": liq}}

            fs.add_unit(InstantaneousUnit(fn=condenser, inlet_ports=(f"bed{k + 1}_out",),
                                          output_ports=("gas", "sulfur"), name=f"condenser{k + 2}"),
                        inlets={f"bed{k + 1}_out": f"bed{k + 1}_out"},
                        outlets={"gas": f"c{k + 2}", "sulfur": f"sulfur{k + 1}"})
            base[f"T_bed{k + 1}"] = float(cp.T_in)
            beds.append(bed)
            reheaters.append(rh)
            sulfur_streams.append(f"sulfur{k + 1}")
            g = dict(out_k, P=out_k["P"] - claus.condenser_dP)
            g, _, _ = cl.cool(g, T_c)
            prev = f"c{k + 2}"
        # the tail gas is the last condenser's gas; name it for the air control
        last = prev

        def tail(inputs, params=None):
            return {"tail_gas": inputs[last]}

        fs.add_unit(InstantaneousUnit(fn=tail, inlet_ports=(last,), output_ports=("tail_gas",), name="tail"),
                    inlets={last: last}, outlets={"tail_gas": "tail_gas"})
        # TGTU: reheat, hydrogenation bed (with the H2 makeup), solvent
        T_h = float(tgtu.T_hydrogenation)
        P_t = float(g["P"])
        h2 = float(tgtu.h2_makeup)
        rh = FurnaceLag(("tail_gas",), "T_hydrogenation", hp.tau_reheat, P_t, T_h, name="tgtu_reheater")
        fs.add_unit(rh, inlets={"tail_gas": "tail_gas", "signals": "signals"}, outlets={"outlet": "hyd_in"})

        def add_h2(n, T_in):
            m = n.at[_IDX["hydrogen"]].add(h2)
            return m, cl.enthalpy(m, T_in)

        Nm3_h = float(jnp.sum(cl.to_vec(g))) * NM3_PER_MOL * 3600.0
        C_h = catalyst_heat_capacity(Nm3_h, hp.hydrogenation_ghsv, hp.hydrogenation_density, hp.hydrogenation_cp)
        bed_flows.append(cl.to_vec(tr["hydrogenated"]))
        hyd = DynamicEquilibriumBed(_gibbs_equilibrium(tg.HYDROGENATION), cl.enthalpy, cl.to_vec, cl.to_stream,
                                    C_h, hp.n_cells, P=P_t, T0=float(tr["hydrogenated"]["T"]),
                                    pre=add_h2 if h2 else None, name="hydrogenation")
        fs.add_unit(hyd, inlets={"inlet": "hyd_in"}, outlets={"outlet": "hydrogenated"})
        base["T_hydrogenation"] = T_h
        ab = tr["absorber"]
        n0 = [float(ab.rich["H2S"] - tgtu.L * tgtu.lean_H2S) * hp.tau_solvent,
              float(ab.rich["CO2"] - tgtu.L * tgtu.lean_CO2) * hp.tau_solvent]
        solv = TGTUSolvent(tgtu, hp.tau_solvent, n0)
        fs.add_unit(solv, inlets={"gas": "hydrogenated"},
                    outlets={p: p for p in solv.output_ports})

        def sulfur_total(inputs, params=None):
            return {"sulfur": {"S": sum(inputs[s]["S"] for s in sulfur_streams)}}

        fs.add_unit(InstantaneousUnit(fn=sulfur_total, inlet_ports=tuple(sulfur_streams), output_ports=("sulfur",),
                                      name="sulfur_pit"),
                    inlets={s: s for s in sulfur_streams}, outlets={"sulfur": "sulfur"})
        y0 = fs.initial_state(None, sig.signal_args(base))
        taus = [float(b.thermal_time(n, b.T0[0])) for b, n in zip(beds + [hyd], bed_flows)]
        return cls(flowsheet=fs, beds=beds, hydrogenation=hyd, reheaters=reheaters, solvent=solv, control=ctl,
                   y0=y0, base_signals=base, steady=res, claus=claus, tgtu=tgtu, holdup=hp, acid_gas=acid_gas,
                   S_in=S_in, tau_beds=taus)

    # ----- running it --------------------------------------------------------------------

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

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Sulfur made, SO2 emitted (mol/s, and recovery), tail-gas ratio residual, bed outlet T, air, recycle."""
        args = self.args() if args is None else args
        st = self.flowsheet.outputs(jnp.asarray(t), y, None, args)
        S_in = cl.sulfur_in(cl.to_vec(st["acid_gas"]))
        so2 = tg.incinerate(st["off_gas"])
        n_t = cl.to_vec(st["tail_gas"])
        out = {"sulfur": st["sulfur"]["S"], "SO2_emitted": so2, "recovery": 1.0 - so2 / S_in,
               "ratio_residual": (n_t[_IDX[H2S]] - 2.0 * n_t[_IDX[SO2]]) / S_in, "air": st["air"]["air"],
               "recycle.H2S": st["recycle"][f"F_{H2S}"]}
        for k, b in enumerate(self.beds):
            out[f"bed{k + 1}.T_out"] = st[f"bed{k + 1}_out"]["T"]
        out["hydrogenation.T_out"] = st["hydrogenated"]["T"]
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
        cr, tr = r["claus"], r["tgtu"]
        S_in = cl.sulfur_in(cl.to_vec(self.acid_gas))
        out = {"sulfur": cr["sulfur"], "SO2_emitted": r["SO2_emitted"], "recovery": r["recovery"],
               "ratio_residual": cr["ratio_residual"] * cr["S_in"] / S_in, "air": r["air"],
               "recycle.H2S": r["recycle"][0]}
        for k, b in enumerate(cr["bed"]):
            out[f"bed{k + 1}.T_out"] = b["T_out"]
        out["hydrogenation.T_out"] = tr["hydrogenated"]["T"]
        return out

    def balance(self, t, y, args: dict | None = None) -> dict:
        """Element balance at ``(t, y)``: ``in - out`` against the TGTU inventory's rate of change (mol/s).

        In: acid gas, air, makeup water, the TGTU's hydrogen. Out: liquid
        sulfur, off-gas, quench water, the inerts dissolved in the rich
        solvent.
        """
        args = self.args() if args is None else args
        fs = self.flowsheet
        st = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)
        us, ds = self._split(y), self._split(dy)
        els = ("S", "H", "O", "C", "N")
        E = {k: tc.elements(k) for k in cl.SPECIES}

        def el(stream):
            return jnp.stack([sum(E[k].get(e, 0) * jnp.asarray(stream.get(f"F_{k}", 0.0), dtype=float)
                                  for k in cl.SPECIES) for e in els])

        air = st["air"]["air"]
        air_el = jnp.stack([0.0, 0.0, 2.0 * cl.AIR_O2 * air, 0.0, 2.0 * (1.0 - cl.AIR_O2) * air])
        h2 = jnp.stack([0.0, 2.0 * float(self.tgtu.h2_makeup), 0.0, 0.0, 0.0])
        inflow = el(st["acid_gas"]) + air_el + el(st["makeup"]) + h2
        sulfur = jnp.stack([st["sulfur"]["S"], 0.0, 0.0, 0.0, 0.0])
        outflow = sulfur + el(st["off_gas"]) + el(st["quench_water"]) + el(st["dissolved"])
        acc = jax.jvp(lambda yy: el(self.solvent.inventory(yy)), (us["tgtu_solvent"],), (ds["tgtu_solvent"],))[1]
        return {"elements": els, "in_minus_out": inflow - outflow, "accumulation": acc, "through": inflow}


__all__ = ["SRUHoldupParams", "AirDemandControl", "TGTUSolvent", "DynamicSRU"]
