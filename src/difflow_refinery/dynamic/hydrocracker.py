"""A dynamic hydrocracker built from a solved steady one.

:class:`DynamicHydrocracker` wires the dynamic hydroprocessing units into a
:class:`~difflow.dynamic.DynamicFlowsheet` with the steady
:class:`~difflow_refinery.hydrocracking.Hydrocracker`'s layout, kinetics,
components and operating specs::

    fresh VGO --+                                  UCO recycle (holdup) --+
                +--> charge heater --> pretreat beds --+                  |
    treat gas --+      (lag)          (cells, quench)  +--> interstage --> cracking beds --> HP separator --> liquid
       ^   ^                                                 (lag)       (cells; quench    |  (holdup)      |
       |   +------------------- quench valves (controllers) ----------------- valves)     vapour          v
       |                                                                                    |       fractionator
       +----- loop inventory, compressor, makeup <-- purge <-- amine <---------------------+     (outlet holdups)

States: the charge-heater and interstage temperatures; every pretreat and
cracking cell's holdup and temperature; the cracking quench valves; the
separator's liquid holdup; the loop's gas inventory; the fractionator's
product and UCO holdups. Both tears of the steady unit are gone: the loop
inventory carries the recycle gas and the fractionator's UCO holdup the UCO
recycle, as ODE states.

**The cracking quench is controlled.** The steady unit holds every cracking
bed's inlet at ``T_crack`` and solves for the quench that does it
(``HydrocrackerParams.quench_crack=None``, the default; fixed quench rates
are a knife-edge, the docs say). The dynamic unit does what the plant's
quench controllers do: each later bed's quench valve is an integral
controller on that bed's mixed inlet temperature::

    dq_k/dt = (T_in,k - T_set) / (G_k tau_c)

``G_k`` is the design gain ``|dT_in,k/dq_k|`` (the cooling bed ``k-1``'s
outlet takes from a quench ``q_k``, from the steady solution) and ``tau_c``
the closed loop's time constant. At rest ``T_in,k = T_set``: the steady
unit's own specification, so the rest state is the steady unit's. A valve
cannot go below shut or above ``q_max`` (a multiple of its design opening):
the opening is a smooth clip of the integrator's state to ``[0, q_max]``,
and back-calculation anti-windup pulls the state back to the clip with time
constant ``tau_aw`` (``dq/dt = ... - (q - clip(q))/tau_aw``), so a valve
against its stop does not wind up and the right-hand side stays smooth (a
hard switch there made the adaptive integrator chatter at ~0.1 s steps).
Inside the range the clip is the identity to about ``exp(-100 x)`` of the
range (1e-14 a third of the way in). With a valve on its stop the bed inlet
floats -- which is how a bed runs away in this model: the quench valves
saturate.

The fractionator is :class:`~.holdup.QuasiSteadyHoldup` around the steady
fractionator's split (a documented TBP split, not a column), with residence
times on its products; the UCO bottoms (bleed and recycle) have the
bottoms-plus-surge-drum residence time.

Holdup parameters (:class:`HydrocrackerHoldupParams`) are ILLUSTRATIVE design
values; they set how fast things move, never where they settle.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.instantaneous import InstantaneousUnit
from difflow.dynamic.state import StateSpec, StateVar
from difflow_refinery.composition import NITROGEN_CLASSES, SULFUR_CLASSES
from difflow_refinery.hydrocracking.fractionator import PRODUCTS, cut_shares, fractionate
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.holdup import QuasiSteadyHoldup
from difflow_refinery.dynamic.hydroprocessing import (
    R_GAS, BedGeometry, DynamicHPSeparator, DynamicTrickleBedReactor, FurnaceLag, RecycleGasLoop,
    flows_scale, holdup_from_outflow, phase_flows)
from difflow_refinery.dynamic.hydrotreater import DEFAULT_ATOL, DEFAULT_RTOL, HoldupParams, _interp_profile

jax.config.update("jax_enable_x64", True)

#: The fractionator's outlets: its products less the whole UCO, plus the UCO split.
FRACTIONATOR_OUTLETS: tuple[str, ...] = tuple(p for p in PRODUCTS if p != "uco") + ("uco_bleed", "uco_recycle")


class DynamicRunawayWarning(UserWarning):
    """A dynamic hydroprocessing simulation did not reach its end time (a bed running away, typically)."""


@dataclass
class HydrocrackerHoldupParams(HoldupParams):
    """Holdups and time constants of the dynamic hydrocracker. ILLUSTRATIVE.

    The fields of :class:`~.hydrotreater.HoldupParams`, and:

    Attributes:
        interstage_tau: Lag of the exchanger that sets the first cracking
            bed's inlet temperature, s.
        quench_tau: Closed-loop time constant of the cracking quench
            controllers, s. Five minutes: with one minute the controllers and
            the beds' gas holdup made every disturbance stiff for the explicit
            integrator (5000 steps for 10 simulated minutes, against about 60
            at five minutes); a real quench loop is faster, so the inlet
            temperatures here recover more slowly than a plant's.
        quench_max: A quench valve's full opening as a multiple of its design
            opening.
        fractionator_tau: Residence time of the fractionator's products
            (trays, side strippers and drums lumped), s.
        uco_tau: Residence time of the UCO (column bottoms and surge drum), s.
        T_alarm: High-temperature alarm (K): :meth:`DynamicHydrocracker.excursion`
            reports when any cell crosses it. 460 C; illustrative.
        min_gas_tau: Floor on a cell's gas residence time, s. A
            hydrocracker's treat gas (1500 Nm^3/m^3 at 150 bar) crosses a
            cell in a few seconds, against tens of seconds for the liquid and
            about a minute for the catalyst's heat: the explicit integrator
            would step at the gas's pace. The floor slows the gas's transit
            through a bed (a few seconds become ``n_cells x min_gas_tau``);
            it does not move the rest state (holdups never do), and nothing
            the plant does on the minute scale depends on the gas's transit.
            ``None`` keeps the physical value.
    """

    interstage_tau: float = 300.0
    quench_tau: float = 300.0
    quench_max: float = 3.0
    fractionator_tau: float = 600.0
    uco_tau: float = 1800.0
    T_alarm: float = 733.15
    min_gas_tau: float | None = 20.0


def cells_from_steady(beds, Ws: Sequence[float], layout, comps, P: float, rho_cat: float, hp: HoldupParams,
                      n_cells: int):
    """``(geometry, y0, scale)`` of a dynamic bed series started on a steady one.

    The steady profile at each cell outlet, turned into the holdup that drains
    at that rate (:func:`~.hydroprocessing.holdup_from_outflow`); the design
    residence times from the bed-inlet gas and liquid flows and the bed's
    void volumes (as :meth:`~.hydrotreater.DynamicHydrotreater.from_steady`).
    """
    geo, y_rx, sc_rx = [], [], []
    xi_out = np.arange(1, n_cells + 1) / n_cells
    pack = jax.vmap(lambda f: f.ravel())
    for bed, W in zip(beds, Ws):
        W = float(W)
        km = bed.k_model
        F_V, F_L, v_L = phase_flows(bed.inlet, layout, comps, km, bed.T_in)
        V_cell = W / rho_cat / n_cells
        eps_G = hp.voidage - hp.liquid_holdup
        Q_V = float(F_V) * R_GAS * float(bed.T_in) / P
        Q_L = float(F_L) * float(v_L)
        tau_G = eps_G * V_cell / Q_V
        tau_L = hp.liquid_holdup * V_cell / Q_L if Q_L > 1e-12 * Q_V else tau_G
        floor = getattr(hp, "min_gas_tau", None)
        if floor is not None:
            tau_G = max(tau_G, float(floor))
        geo.append(BedGeometry(W=W, tau_G=tau_G, tau_L=tau_L))
        Fk, Tk = _interp_profile(bed, xi_out)
        hk = jax.vmap(lambda f, t: holdup_from_outflow(f, layout, comps, km, t, tau_G, tau_L))(Fk, Tk)
        sk = jax.vmap(flows_scale)(hk)
        y_rx.append(jnp.concatenate([pack(hk), Tk[:, None]], axis=1))
        sc_rx.append(jnp.concatenate([pack(sk), jnp.full((n_cells, 1), 100.0)], axis=1))
    scale = jnp.stack(sc_rx).reshape(-1)
    return geo, jnp.stack(y_rx).reshape(-1) / scale, scale


class QuenchControlledReactor(DynamicTrickleBedReactor):
    """Trickle beds whose quench valves are integral controllers on each later bed's inlet temperature.

    Ports: ``"feed"`` (the first bed's inlet stream, at its temperature),
    ``"quench"`` (the treat-gas header the valves draw from: a valve at
    ``q`` takes ``q`` of it) and ``"signals"`` (``T_set`` from the signal
    named ``setpoint``). Outputs: ``"outlet"``, and ``"quench_draw"`` (the
    gas the valves take, which depends on the header and the valve states
    only -- so the header's other users can be evaluated first).

    State: the beds' (as :class:`~.hydroprocessing.DynamicTrickleBedReactor`),
    then one valve position per later bed (scaled by its design opening).
    The module docstring gives the control law.
    """

    output_ports = ("outlet", "quench_draw")
    output_dependencies = {"outlet": ("feed", "quench", "signals"), "quench_draw": ("quench",)}

    #: Sharpness of the smooth clip (per unit of the valve's range).
    CLIP_SHARPNESS = 100.0

    def __init__(self, *args, q0: Sequence[float], gain: Sequence[float], tau_c: float, q_max: Sequence[float],
                 setpoint: str = "T_crack", tau_aw: float = 30.0, **kw):
        super().__init__(*args, **kw)
        self.n_q = self.n_beds - 1
        self.q0 = jnp.asarray(q0, dtype=float).reshape(self.n_q)
        self.q_scale = jnp.maximum(self.q0, 1e-3)
        self.gain = jnp.asarray(gain, dtype=float).reshape(self.n_q)
        self.tau_c = jnp.asarray(tau_c, dtype=float)
        self.q_max = jnp.asarray(q_max, dtype=float).reshape(self.n_q)
        self.setpoint = setpoint
        self.tau_aw = jnp.asarray(tau_aw, dtype=float)

    @property
    def n_bed_states(self) -> int:
        return self.n_beds * self.n_cells * self.n_per_cell

    @property
    def n_states(self) -> int:
        return self.n_bed_states + self.n_q

    def state_spec(self) -> StateSpec:
        base = super().state_spec()
        return StateSpec(list(base.variables) + [StateVar(f"bed{b + 2}.quench", "generic", "-",
                                                          "quench valve position (fraction of the header)")
                                                 for b in range(self.n_q)])

    def initial_state(self, inputs=None, params=None) -> Array:
        return jnp.concatenate([super().initial_state(inputs, params), self.q0 / self.q_scale])

    def unpack(self, state: Array):
        return super().unpack(state[: self.n_bed_states])

    def valves(self, state: Array) -> Array:
        """Valve positions (fractions of the header): the state smoothly clipped to ``[0, q_max]``."""
        x = state[self.n_bed_states:] * self.q_scale / self.q_max
        k = self.CLIP_SHARPNESS
        s = (jax.nn.softplus(k * x) - jax.nn.softplus(k * (x - 1.0))) / k
        return self.q_max * s

    def derivatives(self, t, state, inputs, params=None):
        q = self.valves(state)
        dns, dTs, _, bed_in = self._run(inputs, state, shares=q)
        dn = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *dns)
        dT = jnp.stack(dTs)
        v = jax.vmap(jax.vmap(lambda f: f.ravel()))(dn)
        x = jnp.concatenate([v, dT[..., None]], axis=-1).reshape(-1) / self.scale
        T_in = jnp.stack([b[1] for b in bed_in[1:]])
        raw = state[self.n_bed_states:] * self.q_scale
        dq = (T_in - inputs["signals"][self.setpoint]) / (self.gain * self.tau_c)
        dq = dq - (raw - q) / self.tau_aw            # back-calculation anti-windup (smooth)
        return jnp.concatenate([x, dq / self.q_scale])

    def _draw(self, state, header):
        q = self.valves(state)
        f = {k: v * jnp.sum(q) for k, v in header.items() if k.startswith("F_")}
        f["T"], f["P"] = header["T"], header["P"]
        return f

    def outputs(self, t, state, inputs, params=None):
        q = self.valves(state)
        _, _, outs, _ = self._run(inputs, state, shares=q)
        _, T = self.unpack(state)
        last = jax.tree_util.tree_map(lambda a: a[-1], outs[-1])
        return {"outlet": last.to_stream(self.layout, T[-1, -1], self.P),
                "quench_draw": self._draw(state, inputs["quench"])}

    def partial_outputs(self, t, state, inputs, params=None):
        if len(inputs) == 3:
            return self.outputs(t, state, inputs, params)
        return {"quench_draw": self._draw(state, inputs["quench"])}

    def profiles(self, state, inputs) -> dict:
        q = self.valves(state)
        _, _, outs, bed_in = self._run(inputs, state, shares=q)
        _, T = self.unpack(state)
        out = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *outs)
        return {"T": T, "T_in": jnp.stack([b[1] for b in bed_in]), "outflow": out, "quench": q}


def _gas_split(q_pre_total):
    """The treat-gas header less the cracking quench: the pretreat first bed's gas and its quench."""

    def fn(inputs, params):
        G, draw = inputs["gas"], inputs["draw"]
        quench = {k: v * q_pre_total for k, v in G.items() if k.startswith("F_")}
        bed1 = {k: v - quench[k] - draw.get(k, 0.0) for k, v in G.items() if k.startswith("F_")}
        for d in (quench, bed1):
            d["T"], d["P"] = G["T"], G["P"]
        return {"bed1": bed1, "quench": quench}

    return fn


class DynamicHydrocracker:
    """A dynamic hydrocracker (see the module docstring). Build with :meth:`from_steady`.

    Attributes:
        flowsheet: The :class:`~difflow.dynamic.DynamicFlowsheet`.
        pretreat, crack, separator, loop, heater, interstage, fractionator: Its units.
        y0: The initial state built from the steady solution.
        base_signals: The signals' design values.
        steady: The steady :class:`~difflow_refinery.hydrocracking.unit.HydrocrackerResult`.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # ----- construction ----------------------------------------------------------

    @classmethod
    def from_steady(cls, unit, feed: Mapping, result=None, *, n_cells: int = 5, n_sub: int = 1,
                    holdups: HydrocrackerHoldupParams | None = None) -> "DynamicHydrocracker":
        """The dynamic twin of a steady :class:`~difflow_refinery.hydrocracking.Hydrocracker`.

        Args:
            unit: The steady unit (its layout, kinetics and specs are used).
                Its cracking quench must be solved (``quench_crack=None``,
                the default): the dynamic unit's quench controllers hold the
                same inlet temperatures.
            feed: The feed stream it was solved for.
            result: Its solved result (solved here if omitted).
            n_cells: Cells per bed (pretreat and cracking).
            n_sub: RK4 steps of the steady bed equations per cell.
            holdups: :class:`HydrocrackerHoldupParams`.

        The initial state is the steady solution mapped onto the cells; it is
        close to, not exactly at, the dynamic model's own rest state --
        integrate a while (:meth:`settle`) to start at rest.
        """
        hp = holdups or HydrocrackerHoldupParams()
        p = unit.params
        if p.quench_crack is not None:
            raise NotImplementedError("the dynamic hydrocracker holds the cracking inlets with quench "
                                      "controllers: build the steady unit with quench_crack=None")
        if unit.target is not None:
            raise NotImplementedError("solve the steady unit without a target (give T_crack directly)")
        res = unit.solve(feed) if result is None else result
        th = unit.theta(feed)
        comps = unit._comps(th)
        cst = unit._crack_state(th)
        lay = unit.layout
        n = th["num"]
        P = float(n["P"])
        P_hps = float(n["P"] - n["loop_dP"])
        T_hps = float(n["hps_T"])
        rho_cat = float(n["catalyst_density"])
        o = res.outputs
        W_pre_tot, W_crk_tot = float(o["catalyst.pretreat"]), float(o["catalyst.crack"])
        W_pre = [W_pre_tot * float(s) for s in p.pretreat_beds]
        W_crk = [W_crk_tot * float(s) for s in p.crack_beds]

        # --- reactors
        geo1, y1, s1 = cells_from_steady(res.pretreat.beds, W_pre, lay, comps, P, rho_cat, hp, n_cells)
        q_pre = np.asarray(p.quench_pretreat, dtype=float)
        q_pre_tot = float(q_pre.sum()) if len(p.pretreat_beds) > 1 else 0.0
        shares1 = tuple(q_pre / q_pre_tot) if q_pre_tot > 0 else tuple(np.ones(len(p.pretreat_beds) - 1))
        pretreat = DynamicTrickleBedReactor(lay, unit.hdt_kinetics, th["pre"], comps, geo1, P,
                                            quench_shares=shares1 if len(p.pretreat_beds) > 1 else (),
                                            n_cells=n_cells, n_sub=n_sub, cp_cat=hp.cp_cat, scale=s1, y0=y1,
                                            name="pretreat")
        geo2, y2, s2 = cells_from_steady(res.crack.beds, W_crk, lay, comps, P, rho_cat, hp, n_cells)
        q_crk = np.asarray(res.crack.quench, dtype=float)[1:]          # into beds 2..n
        T_out = np.asarray([float(b.T_out) for b in res.crack.beds])
        T_set = float(n["T_crack"])
        # design gain |dT_in/dq|: the secant from a shut valve (inlet = bed above's outlet)
        gain = np.where(q_crk > 1e-6, (T_out[:-1] - T_set) / np.maximum(q_crk, 1e-6), 300.0)
        gain = np.maximum(gain, 10.0)
        crack = QuenchControlledReactor(lay, unit.kinetics, cst, comps, geo2, P,
                                        quench_shares=tuple(np.ones(len(W_crk) - 1)), n_cells=n_cells,
                                        n_sub=n_sub, cp_cat=hp.cp_cat, scale=s2, y0=y2, name="crack",
                                        q0=q_crk, gain=gain, tau_c=hp.quench_tau,
                                        q_max=hp.quench_max * np.maximum(q_crk, 0.02))

        # --- separator liquid holdup
        liq = res.streams["hps_liquid"]
        n_liq = liq.scale(hp.separator_tau)
        s_liq = flows_scale(n_liq)
        hps = DynamicHPSeparator(lay, comps, T_hps, P_hps, hp.separator_tau, s_liq,
                                 n_liq.ravel() / s_liq.ravel())

        # --- loop inventory (its own quench split unused: the valves draw from the header)
        rec = res.streams["recycle"]
        n_design = P_hps * hp.loop_volume / (R_GAS * T_hps)
        F_rec = float(jnp.sum(rec.gas))
        tau_loop = n_design / F_rec
        n_loop = rec.gas * tau_loop
        s_loop = jnp.maximum(jnp.abs(n_loop), 1e-10 * jnp.max(jnp.abs(n_loop)) + 1e-300)
        h2_target = float(n["h2_oil"]) * float(unit._feed_volume(th)) * MOL_PER_NM3
        M_design = float(o["h2.makeup"]) / float(th["makeup_y"][lay.gas_index("hydrogen")])
        loop = RecycleGasLoop(
            lay, comps, h2_target=h2_target, makeup_y=th["makeup_y"], purge=n["purge"],
            h2s_removal=n["h2s_removal"], nh3_removal=n["nh3_removal"], tau=tau_loop, P_suction=P_hps,
            P_discharge=P, T_suction=T_hps, eta=n["compressor_eta"], makeup_max=hp.makeup_capacity * M_design,
            relief_inventory=(1.0 + hp.relief_margin) * n_design, tau_relief=hp.tau_relief,
            scale=s_loop, y0=n_loop / s_loop)

        # --- fractionator: the steady split behind product holdups
        Tb = th["cut"]["Tb"]
        cps = [n["cut_points"][k] for k in range(4)]
        width = n["cut_width"]
        rho = n["uco_recycle"]

        def frac_op(liquid):
            prods = fractionate(Flows.from_stream(liquid, lay), lay, Tb, cps, width)
            outs = [prods[k] for k in PRODUCTS if k != "uco"] + [prods["uco"].scale(1.0 - rho),
                                                                  prods["uco"].scale(rho)]
            return tuple(f.to_stream(lay, T_hps, P_hps) for f in outs)

        uco_share = cut_shares(Tb, jnp.stack(cps), width)[:, -1]
        tau_f = {k: hp.fractionator_tau for k in FRACTIONATOR_OUTLETS}
        tau_f["uco_bleed"] = tau_f["uco_recycle"] = hp.uco_tau
        frac = QuasiSteadyHoldup.at_steady(frac_op, {"liquid": liq.to_stream(lay, T_hps, P_hps)}, ("liquid",),
                                           FRACTIONATOR_OUTLETS, lay.keys(), tau_f, name="fractionator")

        heater = FurnaceLag(("oil", "gas"), "T_pretreat", hp.heater_tau, P, float(n["T_pretreat"]),
                            name="heater")
        interstage = FurnaceLag(("pretreat", "uco"), "T_crack", hp.interstage_tau, P, T_set, name="interstage")
        split = InstantaneousUnit(fn=_gas_split(q_pre_tot), inlet_ports=("gas", "draw"),
                                  output_ports=("bed1", "quench"), name="gas_split")

        oil = th["oil"]
        feed_T = float(n["T_pretreat"])
        ai = {a: i for i, a in enumerate(lay.attributes)}
        s_cols = jnp.asarray([ai[f"S_{k}"] for k in SULFUR_CLASSES])
        n_cols = jnp.asarray([ai[f"N_{k}"] for k in NITROGEN_CLASSES])

        def oil_feed(t, args):
            s = sig.evaluate(t, args)
            f = oil.scale(s["feed_rate"])
            f = Flows(f.gas, f.cut, f.attr.at[:, s_cols].multiply(s["feed_sulfur"])
                      .at[:, n_cols].multiply(s["feed_nitrogen"]))
            return f.to_stream(lay, feed_T, P)

        def signals_feed(t, args):
            return sig.evaluate(t, args)

        fs = DynamicFlowsheet(species_order=lay.keys())
        fs.add_feed("oil", oil_feed)
        fs.add_feed("signals", signals_feed)
        fs.add_unit(loop, inlets={"vapor": "hps_vapor", "signals": "signals"},
                    outlets={"treat_gas": "header", "quench_gas": "loop_quench_unused", "makeup": "makeup",
                             "vent": "vent", "purge": "purge", "acid_gas": "acid_gas"})
        fs.add_unit(crack, inlets={"feed": "crack_in", "quench": "header", "signals": "signals"},
                    outlets={"outlet": "effluent", "quench_draw": "quench_draw"})
        fs.add_unit(split, inlets={"gas": "header", "draw": "quench_draw"},
                    outlets={"bed1": "pretreat_gas", "quench": "pretreat_quench"})
        fs.add_unit(heater, inlets={"oil": "oil", "gas": "pretreat_gas", "signals": "signals"},
                    outlets={"outlet": "pretreat_in"})
        fs.add_unit(pretreat, inlets={"feed": "pretreat_in", "quench": "pretreat_quench"},
                    outlets={"outlet": "pretreat_out"})
        fs.add_unit(frac, inlets={"liquid": "hps_liquid"},
                    outlets={k: k for k in FRACTIONATOR_OUTLETS})
        fs.add_unit(interstage, inlets={"pretreat": "pretreat_out", "uco": "uco_recycle", "signals": "signals"},
                    outlets={"outlet": "crack_in"})
        fs.add_unit(hps, inlets={"inlet": "effluent"},
                    outlets={"vapor": "hps_vapor", "liquid": "hps_liquid", "water": "hps_water"})
        base = {"T_pretreat": float(n["T_pretreat"]), "T_crack": T_set, "compressor": 1.0, "quench": 0.0,
                "feed_rate": 1.0, "feed_sulfur": 1.0, "feed_nitrogen": 1.0}
        fs._update_state_indices()
        parts = {"loop": loop.initial_state(), "crack": crack.initial_state(), "gas_split": jnp.zeros(0),
                 "heater": heater.initial_state(), "pretreat": pretreat.initial_state(),
                 "fractionator": frac.initial_state(), "interstage": interstage.initial_state(),
                 "hps": hps.initial_state()}
        y0 = jnp.concatenate([jnp.atleast_1d(parts[e.name]) for e in fs.units])
        return cls(flowsheet=fs, pretreat=pretreat, crack=crack, separator=hps, loop=loop, heater=heater,
                   interstage=interstage, fractionator=frac, split=split, y0=y0, base_signals=base, steady=res,
                   unit=unit, layout=lay, comps=comps, holdups=hp, n_cells=n_cells, n_sub=n_sub, oil=oil,
                   tau_loop=tau_loop, T_set=T_set, uco_share=uco_share)

    # ----- signals and simulation ------------------------------------------------

    def args(self, steps: Mapping[str, Sequence[tuple[float, float]]] | None = None,
             ramp: float | Mapping[str, float] = sig.DEFAULT_RAMP, **base) -> dict:
        """Integrator args: the design signals, overridden by ``base``, with ``steps``.

        Signals: ``T_pretreat`` (pretreat inlet setpoint, K), ``T_crack``
        (every cracking bed's inlet setpoint: the interstage exchanger's and
        the quench controllers'), ``compressor`` (1 running, 0 tripped),
        ``feed_rate``, ``feed_sulfur`` and ``feed_nitrogen`` (multipliers of
        the design feed and of its sulfur and nitrogen).
        """
        b = dict(self.base_signals)
        b.update(base)
        return sig.signal_args(b, steps, ramp)

    def simulate(self, t_end: float, args: dict | None = None, y0: Array | None = None, n_save: int = 61,
                 saveat: Array | None = None, method: str = "diffrax:tsit5", rtol: float = DEFAULT_RTOL,
                 atol: float = DEFAULT_ATOL, max_steps: int = 100_000, dtmax: float = 60.0, warn: bool = True,
                 **kwargs):
        """Integrate from ``y0`` (default :attr:`y0`) to ``t_end`` seconds.

        As :meth:`~.hydrotreater.DynamicHydrotreater.simulate`, except that a
        failed integration (the step size collapsing as a bed runs away, or
        ``max_steps``) does not raise: the result's ``info.success`` is
        False, the saved states after the failure are not finite, and (with
        ``warn``) a :class:`DynamicRunawayWarning` names the last finite
        saved time. :meth:`excursion` reports the temperatures.
        """
        args = self.args() if args is None else args
        y0 = self.y0 if y0 is None else y0
        ts = jnp.linspace(0.0, t_end, n_save) if saveat is None else jnp.asarray(saveat)
        kwargs.setdefault("throw", False)
        r = self.flowsheet.simulate((0.0, float(t_end)), y0=y0, method=method, args=args, rtol=rtol, atol=atol,
                                    saveat=ts, max_steps=max_steps, dtmax=dtmax, dt0=kwargs.pop("dt0", 1.0),
                                    **kwargs)
        if warn and not isinstance(r.info.success, jax.core.Tracer) and not bool(r.info.success):
            ok = np.all(np.isfinite(np.asarray(r.trajectory.y)), axis=1)
            t_ok = float(np.asarray(r.trajectory.t)[ok][-1]) if ok.any() else 0.0
            warnings.warn(f"the dynamic hydrocracker stopped before {t_end:.0f} s (last finite saved state "
                          f"at {t_ok:.0f} s): {r.info.message}; a bed running away is the usual cause "
                          "(see excursion())", DynamicRunawayWarning, stacklevel=2)
        return r

    def settle(self, t_end: float = 4 * 3600.0, args: dict | None = None, **kw) -> Array:
        """The state after ``t_end`` s at constant signals (a start at rest)."""
        return self.simulate(t_end, args=args, n_save=2, **kw).y_final

    # ----- what a trajectory shows ------------------------------------------------

    def _split(self, y):
        fs = self.flowsheet
        fs._update_state_indices()
        return {e.name: y[e.state_slice] for e in fs.units}

    def _cut_wppm(self, f: Flows, classes, mw) -> Array:
        lay = self.layout
        ai = {a: i for i, a in enumerate(lay.attributes)}
        X = jnp.sum(f.attr[:, [ai[k] for k in classes]]) * mw
        return 1e6 * X / jnp.sum(f.cut_mass(lay))

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Headline quantities at one time: temperatures, quench, nitrogen slip, conversion, hydrogen, loop."""
        args = self.args() if args is None else args
        fs = self.flowsheet
        lay = self.layout
        streams = fs.outputs(jnp.asarray(t), y, None, args)
        us = self._split(y)
        p1 = self.pretreat.profiles(us["pretreat"], {"feed": streams["pretreat_in"],
                                                     "quench": streams["pretreat_quench"]})
        p2 = self.crack.profiles(us["crack"], {"feed": streams["crack_in"], "quench": streams["header"],
                                               "signals": streams["signals"]})
        comp = self.loop.compressor(us["loop"], streams["signals"])
        F = lambda s: Flows.from_stream(streams[s], lay)
        W1 = jnp.asarray([g.W for g in self.pretreat.beds])
        W2 = jnp.asarray([g.W for g in self.crack.beds])
        wabt = lambda W, pr: jnp.sum(W * (pr["T_in"] + 2.0 * pr["T"][:, -1]) / 3.0) / jnp.sum(W)
        pre_out = F("pretreat_out")
        # conversion of the 370 C+ on the fractionator's own UCO cut, from the products' holdups
        uco = F("uco_bleed") + F("uco_recycle")
        heavy_in = self._heavy(F("oil")) + self._heavy(F("uco_recycle"))
        n_cols = [f"N_{k}" for k in NITROGEN_CLASSES]
        return {
            "T_heater": 100.0 * us["heater"][0], "T_interstage": 100.0 * us["interstage"][0],
            "pretreat.T_in": p1["T_in"], "pretreat.T_out": p1["T"][:, -1], "pretreat.cell_T": p1["T"],
            "crack.T_in": p2["T_in"], "crack.T_out": p2["T"][:, -1], "crack.cell_T": p2["T"],
            "crack.quench": p2["quench"], "T_max": jnp.maximum(jnp.max(p1["T"]), jnp.max(p2["T"])),
            "wabt.pretreat": wabt(W1, p1), "wabt.crack": wabt(W2, p2),
            "pretreat.N_wppm": self._cut_wppm(pre_out, n_cols, 14.0067e-3),
            "conversion.per_pass": 1.0 - uco.mass(lay) / jnp.maximum(heavy_in, 1e-300),
            "uco.recycle_rate": F("uco_recycle").mass(lay),
            "diesel.rate": F("diesel").mass(lay), "kerosene.rate": F("kerosene").mass(lay),
            "recycle.rate": comp["recycle"], "compressor.T_out": comp["T_out"], "h2.makeup": comp["makeup"],
            "vent.rate": jnp.sum(F("vent").gas), "loop.inventory": jnp.sum(self.loop.inventory(us["loop"])),
        }

    def _heavy(self, f: Flows) -> Array:
        """Mass (kg/s) of ``f`` in the fractionator's UCO cut (the conversion's 370 C+ basis)."""
        return jnp.sum(f.cut_mass(self.layout) * self.uco_share)

    def trajectory(self, result, args: dict | None = None, keys: Sequence[str] | None = None) -> dict:
        """:meth:`observe` at every saved time of a simulation, as arrays with a leading time axis."""
        ts, ys = result.trajectory.t, result.trajectory.y
        args = self.args() if args is None else args
        f = jax.jit(lambda t, y, a: self.observe(t, y, a))
        ok = np.all(np.isfinite(np.asarray(ys)), axis=1)
        obs = [f(t, y, args) for t, y, k in zip(ts, ys, ok) if k]
        keys = keys or list(obs[0])
        out = {k: jnp.stack([o[k] for o in obs]) for k in keys}
        out["t"] = jnp.asarray(np.asarray(ts)[ok])
        return out

    def excursion(self, result, args: dict | None = None) -> dict:
        """``{"T_max", "t_T_max", "t_alarm", "completed"}`` of a simulation.

        ``T_max`` is the hottest cell over the finite saved states,
        ``t_alarm`` the first saved time a cell is above
        :attr:`HydrocrackerHoldupParams.T_alarm` (None if never) and
        ``completed`` whether the integration reached its end.
        """
        tr = self.trajectory(result, args, keys=["T_max"])
        T = np.asarray(tr["T_max"])
        t = np.asarray(tr["t"])
        hot = np.nonzero(T > self.holdups.T_alarm)[0]
        return {"T_max": float(T.max()), "t_T_max": float(t[int(T.argmax())]),
                "t_alarm": float(t[hot[0]]) if hot.size else None, "completed": bool(result.info.success)}

    def steady_reference(self) -> dict:
        """The steady unit's values of the quantities :meth:`observe` reports."""
        r = self.steady
        o = r.outputs
        return {"wabt.pretreat": o["wabt.pretreat"], "wabt.crack": o["wabt.crack"],
                "pretreat.T_out": jnp.stack([b.T_out for b in r.pretreat.beds]),
                "crack.T_out": jnp.stack([b.T_out for b in r.crack.beds]),
                "crack.quench": r.crack.quench[1:], "pretreat.N_wppm": o["pretreat.N_wppm"],
                "conversion.per_pass": o["conversion.per_pass"], "uco.recycle_rate": o["uco.recycle_rate"],
                "diesel.rate": o["diesel.rate"], "kerosene.rate": o["kerosene.rate"],
                "recycle.rate": o["recycle.rate"], "compressor.T_out": o["compressor.T_out"],
                "h2.makeup": o["h2.makeup"]}

    # ----- balances -----------------------------------------------------------------

    def inventory(self, y) -> Flows:
        """Total material held (mol): every reactor cell, the separator liquid, the loop gas, the fractionator."""
        us = self._split(y)
        lay = self.layout
        held = Flows.zeros(lay)
        for rx in (self.pretreat, self.crack):
            n, _ = rx.unpack(us[rx.name])
            held = held + jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=(0, 1)), n)
        held = held + self.separator.holdup(us["hps"])
        held = held + Flows(self.loop.inventory(us["loop"]), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
        for port, inv in self.fractionator.inventory(us["fractionator"]).items():
            held = held + Flows.from_stream(inv, lay)
        return held

    def element_balance(self, t, y, args: dict | None = None) -> dict:
        """``{"accumulation", "in_minus_out"}``: (4,) mol/s of C, H, S, N at one state (equal by construction)."""
        args = self.args() if args is None else args
        fs = self.flowsheet
        lay = self.layout
        streams = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)
        acc = jax.jvp(lambda v: self.inventory(v).elements(lay), (y,), (dy,))[1]
        F = lambda s: Flows.from_stream(streams[s], lay)
        inflow = F("oil") + F("makeup")
        outflow = F("hps_water") + F("purge") + F("acid_gas") + F("vent")
        for k in FRACTIONATOR_OUTLETS:
            if k != "uco_recycle":
                outflow = outflow + F(k)
        return {"accumulation": acc, "in_minus_out": inflow.elements(lay) - outflow.elements(lay)}


__all__ = ["FRACTIONATOR_OUTLETS", "DynamicRunawayWarning", "HydrocrackerHoldupParams", "cells_from_steady",
           "QuenchControlledReactor", "DynamicHydrocracker"]
