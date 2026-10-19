"""A dynamic steam-methane-reforming hydrogen plant: tube and catalyst heat capacity, furnace lag, shift bed, PSA lag.

The steady plant (:func:`difflow_refinery.hydrogen.smr.smr_plant`) is a chain
of steady operations. In time, three things in it hold heat or material for
long enough to matter to the plant around it, and the dynamic plant keeps
exactly those as states:

* **The reformer tubes** (:class:`DynamicReformerTube`). A tube is ``N``
  cells along its length. A cell's state is its temperature ``T_k`` -- the
  tube metal's and the catalyst's, shared with the gas leaving it. The gas
  passes a tube in about a second, so it is quasi-steady: the gas entering
  cell ``k`` (the previous cell's outflow at ``T_{k-1}``) is carried across
  the cell's stretch of tube by the STEADY tube equations
  (Xu & Froment rates, the flux or wall heat input; :func:`smr._integrate`
  over ``z_k..z_{k+1}``) to ``(n*, T*)``, and leaves the cell at ``T_k``.
  The cell's energy balance::

      C_k dT_k/dt = H(n*, T*) - H(n*, T_k)

  At rest ``T_k = T*``, so the cells are the steady tube integrated a
  stretch at a time: the rest state is the steady tube's. ``C_k`` is the
  cell's tube metal plus catalyst (:class:`SMRHoldupParams`). With a flux
  heat input the heat the tubes absorb does not depend on the tube
  temperature (a radiant furnace, flue at about 1300 K against a tube
  wall at 1100 K, is close to that); with the wall mode it does.
* **The furnace** (:class:`FiringLag`): the heat the tubes absorb lags its
  setpoint (fuel valve, firebox and refractory), ``tau_fire dQ/dt = Q_set - Q``.
  The supplementary fuel is what closes that duty with the (lagged) PSA
  tail gas, as in the steady plant.
* **The HT shift bed** (and an LT bed if the plant has one): a
  :class:`~difflow_refinery.dynamic.beds.DynamicEquilibriumBed`, the steady
  shift's equilibrium in thermal cells.
* **The PSA** (:class:`~difflow_refinery.dynamic.holdup.QuasiSteadyHoldup`):
  the steady PSA on the knocked-out gas, its product and tail gas behind a
  cycle-averaged lag ``tau_psa`` (about one cycle). A cycle-resolved PSA --
  beds pressurising, adsorbing, blowing down and purging in sequence, with
  the product purity cycling within each cycle -- is out of scope: it is a
  hybrid system of tens of switching beds, its time scale (a minute per
  step) is below what the refinery around it sees, and the plant's
  hydrogen header (a gas inventory) smooths it anyway.

The ZnO guard, steam addition, prereformer, process-gas boiler and
knock-out are instantaneous (steady functions evaluated at every time). The
prereformer's catalyst heat capacity is not modelled.

Element balances close exactly: the tubes and beds hold no gas, and the
PSA holdup is a conserving lag. The process side's in (natural gas, steam)
minus out (product, tail gas, condensate, H2S taken by the guard) is the
PSA inventory's rate of change at every instant
(:meth:`DynamicHydrogenPlant.balance`).

Holdup parameters are ILLUSTRATIVE (:class:`SMRHoldupParams`); they set the
speeds, not the rest state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow import GibbsReactor, GibbsReactorParams
from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.instantaneous import InstantaneousUnit
from difflow.dynamic.state import StateSpec, StateVar
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.beds import NM3_PER_MOL, DynamicEquilibriumBed, catalyst_heat_capacity
from difflow_refinery.dynamic.holdup import QuasiSteadyHoldup
from difflow_refinery.hydrogen import smr

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class SMRHoldupParams:
    """Heat capacities and lags of the dynamic hydrogen plant. ALL ILLUSTRATIVE.

    Attributes:
        wall: Tube wall thickness (m). Centrifugally cast HP-type reformer
            tubes of about 100 mm bore have walls of 10-15 mm.
        metal_density: Tube alloy density (kg/m3; HP-40 Nb about 7900).
        metal_cp: Tube alloy heat capacity at 1000-1200 K (J/(kg K)).
        catalyst_cp: Reforming catalyst (Ni on alumina) heat capacity (J/(kg K)).
        tau_fire: Furnace lag (s): fuel valve, firebox and refractory.
        tau_psa: PSA product and tail-gas lag (s): about one PSA cycle.
        shift_ghsv: Shift bed gas hourly space velocity (h^-1, wet gas at
            normal conditions) that sizes its catalyst.
        shift_density: Shift catalyst bulk density (kg/m3).
        shift_cp: Shift catalyst heat capacity (J/(kg K)).
        n_tube_cells: Cells per tube.
        n_bed_cells: Cells per shift bed.
    """

    wall: float = 0.0127
    metal_density: float = 7900.0
    metal_cp: float = 600.0
    catalyst_cp: float = 1000.0
    tau_fire: float = 300.0
    tau_psa: float = 600.0
    shift_ghsv: float = 3500.0
    shift_density: float = 1250.0
    shift_cp: float = 850.0
    n_tube_cells: int = 8
    n_bed_cells: int = 4


def tube_heat_capacity(tp: smr.ReformerTubeParams, hp: SMRHoldupParams) -> float:
    """Heat capacity of one tube's metal and catalyst (J/K)."""
    d_out = tp.d_in + 2.0 * hp.wall
    metal = hp.metal_density * jnp.pi / 4.0 * (d_out ** 2 - tp.d_in ** 2) * tp.length
    cat = tp.rho_bed * jnp.pi / 4.0 * tp.d_in ** 2 * tp.length
    return float(metal * hp.metal_cp + cat * hp.catalyst_cp)


class FiringLag:
    """The reformer furnace: the heat the tubes absorb lags its setpoint.

    Port ``"signals"`` (reads ``signal``, W). Output ``"firing"``:
    ``{"Q": W}``. State: ``Q / Q0``::

        tau dQ/dt = Q_set(t) - Q
    """

    output_ports = ("firing",)
    output_dependencies = {"firing": ()}

    def __init__(self, Q0: float, tau: float, signal: str = "firing", name: str = "furnace_lag"):
        self.Q0 = float(Q0)
        self.tau = jnp.asarray(tau, dtype=float)
        self.signal = signal
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar("Q", "generic", "W", "heat absorbed by the tubes", scale=self.Q0)])

    def initial_state(self, inputs=None, params=None):
        return jnp.asarray([1.0])

    def derivatives(self, t, state, inputs, params=None):
        Q = self.Q0 * state[0]
        return jnp.reshape((inputs["signals"][self.signal] - Q) / self.tau / self.Q0, (1,))

    def outputs(self, t, state, inputs, params=None):
        return {"firing": {"Q": self.Q0 * state[0]}}

    partial_outputs = outputs


class DynamicReformerTube:
    """The plant's reformer tubes (``n_tubes`` identical tubes) as ``n_cells`` thermal cells each.

    Ports ``"inlet"`` (the plant's gas over :data:`smr.SYNGAS` at the tube
    inlet) and ``"firing"`` (``{"Q": W}``, all tubes; flux mode only).
    Output ``"outlet"`` (the plant's reformed gas). State: cell
    temperatures / 100 K. See the module docstring.
    """

    output_ports = ("outlet",)
    output_dependencies = {"outlet": ("inlet",)}

    def __init__(self, tp: smr.ReformerTubeParams, n_tubes: float, n_cells: int, C_tube: float, P_in: float,
                 T0: Sequence[float], name: str = "tubes"):
        self.tp = tp
        self.n_tubes = jnp.asarray(n_tubes, dtype=float)
        self.n_cells = int(n_cells)
        self.C_cell = jnp.asarray(C_tube / self.n_cells, dtype=float)
        self.P_in = jnp.asarray(P_in, dtype=float)
        self.T0 = np.asarray(T0, dtype=float)
        self.z = np.linspace(0.0, tp.length, self.n_cells + 1)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"cell{k + 1}.T", "temperature", "K", "", scale=100.0)
                          for k in range(self.n_cells)])

    def initial_state(self, inputs=None, params=None):
        return jnp.asarray(self.T0) / 100.0

    def _heat(self, inputs):
        if self.tp.mode == "flux":
            return smr._flux_heat(inputs["firing"]["Q"] / self.n_tubes, self.tp)
        return smr._wall_heat(self.tp)

    def _run(self, inputs, T):
        """Per tube: ``(n*, T*)`` of each cell's stretch from the upstream cell's gas."""
        inlet = inputs["inlet"]
        n0 = smr.to_vec(inlet) / self.n_tubes
        T_in = jnp.asarray(inlet["T"], dtype=float)
        q = self._heat(inputs)
        zs = jnp.asarray(self.z)

        def body(carry, xs):
            n, Tk = carry
            T_cell, z0, z1 = xs
            y, _ = smr._integrate(n, Tk, self.P_in, q, self.tp, z_span=(z0, z1))
            ns, Ts = y[-1, :-1], y[-1, -1]
            return (ns, T_cell), (ns, Ts)

        _, (ns, Ts) = jax.lax.scan(body, (n0, T_in), (T, zs[:-1], zs[1:]))
        return ns, Ts

    def derivatives(self, t, state, inputs, params=None):
        T = 100.0 * state
        ns, Ts = self._run(inputs, T)
        H_star = jax.vmap(smr.enthalpy)(ns, Ts)
        H_cell = jax.vmap(smr.enthalpy)(ns, T)
        return (H_star - H_cell) / self.C_cell / 100.0

    def outputs(self, t, state, inputs, params=None):
        T = 100.0 * state
        ns, _ = self._run(inputs, T)
        return {"outlet": smr.to_stream(ns[-1] * self.n_tubes, T[-1], self.P_in - self.tp.pressure_drop)}

    def report(self, state, inputs) -> dict:
        T = 100.0 * state
        ns, Ts = self._run(inputs, T)
        return {"T": T, "T_star": Ts, "n": ns}


def _shift_equilibrium(approach_dT):
    def eq(n, T, P):
        out, _ = GibbsReactor(GibbsReactorParams(thermo=smr.GAS, mode="isothermal", T_out=T,
                                                 reactions=[smr.WGS], approach_dT=approach_dT))(
            smr.to_stream(n, T, P))
        return smr.to_vec(out)
    return eq


def _species_stream(d: Mapping, T, P):
    """``{species: mol/s}`` (the PSA's dicts) as a stream over :data:`smr.SYNGAS`."""
    return smr.to_stream(jnp.stack([jnp.asarray(d.get(k, 0.0), dtype=float) for k in smr.SYNGAS]), T, P)


SYN_KEYS = tuple(f"F_{k}" for k in smr.SYNGAS)


class DynamicHydrogenPlant:
    """The SMR hydrogen plant as a dynamic flowsheet, built with :meth:`from_steady`.

    Feeds ``"natural_gas"`` (the steady feed times the ``feed_rate``
    signal) and ``"signals"``. Signals: ``feed_rate`` (-), ``S_C``
    (mol/mol), ``firing`` (W absorbed by all tubes, flux mode), ``T_hts``
    (HT shift inlet, K; the process-gas boiler holds it).
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_steady(cls, feed, p: smr.SMRParams = smr.SMRParams(), result: dict | None = None,
                    holdup: SMRHoldupParams = SMRHoldupParams()) -> "DynamicHydrogenPlant":
        """The dynamic plant of a steady :func:`smr.smr_plant` (solved here unless ``result`` is given).

        It starts at the steady plant's state: the tube cells at the steady
        tube's profile, the furnace at its duty, the shift bed at its outlet,
        the PSA holdup at its product and tail gas.
        """
        if p.tube.mode != "flux":
            raise NotImplementedError("the dynamic plant is built for the flux mode (the firing is a signal)")
        res = smr.smr_plant(feed, p) if result is None else result
        hp = holdup
        nt = float(p.n_tubes)
        ri = res["reformer_info"]
        Q_tube = float(ri["Q"])
        Q0 = Q_tube * nt
        per_tube = smr.to_stream(smr.to_vec(res["prereformer"]) / nt, p.T_reformer_in, p.P)
        _, prof = smr.reformer_tube(per_tube, p.tube, T_out=p.T_out, n_profile=hp.n_tube_cells + 1)
        T_tube0 = np.asarray(prof["profile"]["T"])[1:]
        fuel_comp = p.fuel
        if fuel_comp is None:
            tot = sum(float(feed.get(f"F_{k}", 0.0)) for k in smr.SPECIES)
            fuel_comp = {k: float(feed.get(f"F_{k}", 0.0)) / tot for k in smr.SPECIES}

        fs = DynamicFlowsheet(species_order=[f"F_{k}" for k in smr.SPECIES])
        ng = dict(feed)

        def ng_feed(t, args):
            s = sig.evaluate(t, args)
            return {k: (v * s["feed_rate"] if k.startswith("F_") else v) for k, v in ng.items()}

        fs.add_feed("natural_gas", ng_feed)
        fs.add_feed("signals", lambda t, args: sig.evaluate(t, args))

        def front_end(inputs, params=None):
            s = inputs["signals"]
            d, removed = smr.desulfurize(inputs["natural_gas"], p.h2s_removal)
            steam = s["S_C"] * smr.carbon_flow(d)
            mixed = dict(d)
            mixed[f"F_{smr.H2O}"] = jnp.asarray(d.get(f"F_{smr.H2O}", 0.0), dtype=float) + steam
            if p.T_prereformer is not None:
                pre, _ = smr.prereform(mixed, p.T_prereformer)
                n = smr.to_vec(pre)
            else:
                n = smr.to_vec(mixed)
            return {"reformer_feed": smr.to_stream(n, p.T_reformer_in, p.P),
                    "steam": {f"F_{smr.H2O}": steam, "T": jnp.asarray(p.T_steam), "P": jnp.asarray(p.P)},
                    "guard": {f"F_{smr.H2S}": removed, "T": jnp.asarray(feed["T"]), "P": jnp.asarray(p.P)}}

        fs.add_unit(InstantaneousUnit(fn=front_end, inlet_ports=("natural_gas", "signals"),
                                      output_ports=("reformer_feed", "steam", "guard"), name="front_end"),
                    inlets={"natural_gas": "natural_gas", "signals": "signals"},
                    outlets={"reformer_feed": "reformer_feed", "steam": "steam", "guard": "guard"})
        furnace_lag = FiringLag(Q0, hp.tau_fire)
        fs.add_unit(furnace_lag, inlets={"signals": "signals"}, outlets={"firing": "firing"})
        C_tube = tube_heat_capacity(p.tube, hp)
        tubes = DynamicReformerTube(p.tube, nt, hp.n_tube_cells, C_tube, p.P, T_tube0)
        fs.add_unit(tubes, inlets={"inlet": "reformer_feed", "firing": "firing"}, outlets={"outlet": "reformed"})

        def pgb(inputs, params=None):
            g, duty = smr.cool(inputs["reformed"], inputs["signals"]["T_hts"])
            return {"hts_in": g}

        fs.add_unit(InstantaneousUnit(fn=pgb, inlet_ports=("reformed", "signals"), output_ports=("hts_in",),
                                      name="process_gas_boiler"),
                    inlets={"reformed": "reformed", "signals": "signals"}, outlets={"hts_in": "hts_in"})

        def bed_C(stream):
            Nm3_h = float(jnp.sum(smr.to_vec(stream))) * NM3_PER_MOL * 3600.0
            return catalyst_heat_capacity(Nm3_h, hp.shift_ghsv, hp.shift_density, hp.shift_cp)

        hts = DynamicEquilibriumBed(_shift_equilibrium(p.hts_approach_dT), smr.enthalpy, smr.to_vec,
                                    smr.to_stream, bed_C(res["hts"]), hp.n_bed_cells,
                                    P=float(res["reformer"]["P"]), T0=float(res["hts"]["T"]), dP=p.shift_dP,
                                    name="hts")
        fs.add_unit(hts, inlets={"inlet": "hts_in"}, outlets={"outlet": "hts_out"})
        beds = [hts]
        last = "hts_out"
        if p.T_lts is not None:
            def lts_cool(inputs, params=None):
                g, _ = smr.cool(inputs["hts_out"], p.T_lts)
                return {"lts_in": g}

            fs.add_unit(InstantaneousUnit(fn=lts_cool, inlet_ports=("hts_out",), output_ports=("lts_in",),
                                          name="hts_cooler"),
                        inlets={"hts_out": "hts_out"}, outlets={"lts_in": "lts_in"})
            lts = DynamicEquilibriumBed(_shift_equilibrium(p.lts_approach_dT), smr.enthalpy, smr.to_vec,
                                        smr.to_stream, bed_C(res["lts"]), hp.n_bed_cells,
                                        P=float(res["hts"]["P"]), T0=float(res["lts"]["T"]), dP=p.shift_dP,
                                        name="lts")
            fs.add_unit(lts, inlets={"inlet": "lts_in"}, outlets={"outlet": "lts_out"})
            beds.append(lts)
            last = "lts_out"

        def ko(inputs, params=None):
            g, cond, _ = smr.knock_out(inputs[last], p.T_ko)
            return {"ko_gas": g, "condensate": {f"F_{smr.H2O}": cond, "T": jnp.asarray(p.T_ko), "P": g["P"]}}

        fs.add_unit(InstantaneousUnit(fn=ko, inlet_ports=(last,), output_ports=("ko_gas", "condensate"),
                                      name="knock_out"),
                    inlets={last: last}, outlets={"ko_gas": "ko_gas", "condensate": "condensate"})

        def psa_op(ko_gas):
            ps = smr.purify(ko_gas, p.psa)
            return (_species_stream(ps["product"], ko_gas["T"], ko_gas["P"]),
                    _species_stream(ps["tail_gas"], ko_gas["T"], ko_gas["P"]))

        psa = QuasiSteadyHoldup.at_steady(psa_op, {"ko_gas": res["ko"]}, ["ko_gas"], ["product", "tail_gas"],
                                          SYN_KEYS, hp.tau_psa, name="psa")
        fs.add_unit(psa, inlets={"ko_gas": "ko_gas"}, outlets={"product": "product", "tail_gas": "tail_gas"})

        def burner(inputs, params=None):
            tail = {k: inputs["tail_gas"].get(f"F_{k}", 0.0) for k in smr.SYNGAS}
            fz = smr.furnace(inputs["firing"]["Q"], tail, fuel_comp, p.y_O2, p.T_air,
                             T_bridgewall=p.T_bridgewall, T_stack=p.T_stack)
            fuel = {f"F_{k}": fz["fuel"] * v for k, v in fuel_comp.items()}
            fuel.update(T=jnp.asarray(p.T_air), P=jnp.asarray(101325.0))
            flue = {f"F_{k}": fz["flue"][i] for i, k in enumerate(smr.FLUE)}
            flue.update(T=jnp.asarray(p.T_stack), P=jnp.asarray(101325.0))
            return {"fuel": fuel, "flue": flue}

        fs.add_unit(InstantaneousUnit(fn=burner, inlet_ports=("tail_gas", "firing"), output_ports=("fuel", "flue"),
                                      name="burner"),
                    inlets={"tail_gas": "tail_gas", "firing": "firing"}, outlets={"fuel": "fuel", "flue": "flue"})

        base = {"feed_rate": 1.0, "S_C": float(p.S_C), "firing": Q0, "T_hts": float(p.T_hts)}
        y0 = fs.initial_state(None, sig.signal_args(base))
        # time scales: a tube's thermal transit (cells' heat capacity over the gas's heat flow)
        n_t = smr.to_vec(per_tube)
        C_flow = float(jax.jvp(lambda T: smr.enthalpy(n_t, T), (jnp.asarray(p.T_out),), (jnp.asarray(1.0),))[1])
        return cls(flowsheet=fs, tubes=tubes, beds=beds, psa=psa, furnace_lag=furnace_lag, y0=y0,
                   base_signals=base, steady=res, params=p, holdup=hp, feed=feed, fuel_comp=fuel_comp,
                   tau_tube=C_tube / C_flow, tau_beds=[float(b.thermal_time(smr.to_vec(res["hts"]), res["hts"]["T"]))
                                                       for b in beds])

    # ----- running it -----------------------------------------------------------------

    def args(self, steps: Mapping[str, Sequence[tuple[float, float]]] | None = None,
             ramp: float | Mapping[str, float] = sig.DEFAULT_RAMP, **base) -> dict:
        """Integrator args: the design signals, overridden by ``base``, with ``steps``."""
        b = dict(self.base_signals)
        b.update(base)
        return sig.signal_args(b, steps, ramp)

    def simulate(self, t_end: float, args: dict | None = None, y0: Array | None = None, n_save: int = 61,
                 saveat: Array | None = None, method: str = "diffrax:tsit5", rtol: float = 1e-6,
                 atol: float = 1e-8, max_steps: int = 100_000, dtmax: float | None = None, **kwargs):
        """Integrate from ``y0`` (default :attr:`y0`) to ``t_end`` seconds.

        ``dtmax`` defaults to the smallest thermal time constant of a tube
        cell (:attr:`tau_tube` over the cell count): the explicit step stays
        inside its stability region and does not grow across a quiet spell
        into a step. ``kwargs`` go to ``diffrax.diffeqsolve`` (``adjoint=``).
        """
        dtmax = self.tau_tube / self.tubes.n_cells if dtmax is None else dtmax
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
        """Product H2 (mol/s), purity, reformer outlet T, CH4 slip (dry), shift outlet T, fuel, CO2."""
        args = self.args() if args is None else args
        st = self.flowsheet.outputs(jnp.asarray(t), y, None, args)
        prod = smr.to_vec(st["product"])
        ref = smr.to_vec(st["reformed"])
        i = smr._I
        dry = jnp.sum(ref) - ref[i[smr.H2O]]
        flue = st["flue"]
        return {"H2": prod[i[smr.H2]], "purity": prod[i[smr.H2]] / jnp.sum(prod),
                "T_reformer_out": st["reformed"]["T"], "CH4_slip": ref[i[smr.CH4]] / dry,
                "T_hts_out": st["hts_out"]["T"], "firing": st["firing"]["Q"],
                "fuel": sum(v for k, v in st["fuel"].items() if k.startswith("F_")),
                "CO2": flue[f"F_{smr.CO2}"]}

    def trajectory(self, result, args: dict | None = None, keys: Sequence[str] | None = None) -> dict:
        """:meth:`observe` at every saved time, with a leading time axis."""
        args = self.args() if args is None else args
        f = jax.jit(lambda t, y, a: self.observe(t, y, a))
        obs = [f(t, y, args) for t, y in zip(result.trajectory.t, result.trajectory.y)]
        keys = keys or list(obs[0])
        out = {k: jnp.stack([o[k] for o in obs]) for k in keys}
        out["t"] = result.trajectory.t
        return out

    def steady_reference(self) -> dict:
        """The steady plant's values of what :meth:`observe` reports."""
        r = self.steady
        i = smr._I
        prod = jnp.stack([jnp.asarray(r["psa"]["product"].get(k, 0.0)) for k in smr.SYNGAS])
        return {"H2": r["H2"], "purity": prod[i[smr.H2]] / jnp.sum(prod),
                "T_reformer_out": r["reformer"]["T"], "CH4_slip": r["CH4_slip"],
                "T_hts_out": r["hts"]["T"], "firing": r["reformer_duty"],
                "fuel": r["fuel"], "CO2": r["CO2"]}

    # ----- balances ---------------------------------------------------------------------

    def balance(self, t, y, args: dict | None = None) -> dict:
        """Process-side element balance at ``(t, y)``: ``{element: (in - out, d inventory/dt)}`` (mol/s).

        In: natural gas and steam. Out: PSA product and tail gas, condensate,
        the guard's H2S. The tubes and beds hold no gas, so the difference is
        the PSA holdup's rate of change, exactly.
        """
        from difflow_refinery import thermochemistry as tc
        args = self.args() if args is None else args
        fs = self.flowsheet
        st = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)
        us, ds = self._split(y), self._split(dy)
        els = ("C", "H", "O", "N", "S")
        names = {k: tc.elements(k) for k in smr.SPECIES}

        def el(stream):
            return jnp.stack([sum(names[k].get(e, 0) * jnp.asarray(stream.get(f"F_{k}", 0.0), dtype=float)
                                  for k in smr.SPECIES) for e in els])

        inflow = el(st["natural_gas"]) + el(st["steam"])
        outflow = el(st["product"]) + el(st["tail_gas"]) + el(st["condensate"]) + el(st["guard"])

        def held(ypsa):
            inv = self.psa.inventory(ypsa)
            return sum(el({k: v for k, v in s.items()}) for s in inv.values())

        acc = jax.jvp(held, (us["psa"],), (ds["psa"],))[1]
        return {"elements": els, "in_minus_out": inflow - outflow, "accumulation": acc, "through": inflow}


__all__ = ["SMRHoldupParams", "FiringLag", "DynamicReformerTube", "DynamicHydrogenPlant", "tube_heat_capacity"]
