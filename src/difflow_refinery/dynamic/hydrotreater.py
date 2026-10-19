"""A dynamic hydrotreater built from a solved steady one.

:class:`DynamicHydrotreater` wires the dynamic units of
:mod:`.hydroprocessing` into a :class:`~difflow.dynamic.DynamicFlowsheet`
with the steady :class:`~difflow_refinery.hydrotreating.Hydrotreater`'s
layout, kinetics, components and operating specs::

    oil ---------+
                 +--> charge heater --> bed 1 --(quench)--> bed 2 ... --> HP separator --> liquid
    treat gas ---+       (lag)          (cells)                             |   (holdup)
       ^                                                                   vapour
       +--------- loop inventory, compressor, makeup <-- purge <-- amine <-+

States: the heater temperature; every reactor cell's holdup and temperature;
the separator's liquid holdup; the loop's gas inventory. The recycle tear of
the steady unit is gone -- the loop inventory carries it as ODE states -- so
nothing is solved iteratively round the loop.

The stripper and products section is left steady: the separator liquid is
what the dynamic model returns, and :meth:`DynamicHydrotreater.observe`
reports its sulfur on the cuts (what the stripper does not change). Run the
steady stripper on a separator liquid if the product properties are needed
at a time.

Holdup parameters (:class:`HoldupParams`) are ILLUSTRATIVE design values:
see each field. They set how fast things move, never where they settle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.integrators import integrate
from difflow_refinery.composition import SULFUR_CLASSES
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.recycle import MOL_PER_NM3
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.hydroprocessing import (
    CATALYST_CP, R_GAS, BedGeometry, DynamicHPSeparator, DynamicTrickleBedReactor, FurnaceLag,
    RecycleGasLoop, flows_scale, holdup_from_outflow, phase_flows)

jax.config.update("jax_enable_x64", True)

#: Integration defaults of :meth:`DynamicHydrotreater.simulate` (on scaled states).
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-8


@dataclass
class HoldupParams:
    """Holdups and time constants of the dynamic hydrotreater. ILLUSTRATIVE.

    Attributes:
        voidage: Bed void fraction (interparticle). 0.35-0.45 for loaded
            extrudates; 0.40.
        liquid_holdup: Total (static + dynamic) liquid holdup as a fraction of
            the bed volume; 0.12, a round value of the usual order for the
            trickling regime (liquid holdup in trickle beds is discussed in
            Satterfield's review, AIChE J. 21, 209 (1975),
            doi:10.1002/aic.690210202; the value is not taken from it). The
            gas holds the rest of the voids.
        cp_cat: Catalyst heat capacity, J/(kg K) (see :data:`.hydroprocessing.CATALYST_CP`).
        heater_tau: Charge-heater outlet lag, s (firebox and tube metal).
        separator_tau: HP-separator liquid residence time, s (5-10 min is the
            usual design range; 300).
        loop_volume: Gas volume of the recycle loop (separator vapour space,
            knock-out drum, scrubber, piping), m^3.
        makeup_capacity: Makeup compressor capacity as a multiple of the
            design makeup rate.
        relief_margin: The loop vents to flare above this fraction over its
            design inventory (its pressure controller).
        tau_relief: Time constant of that vent, s.
    """

    voidage: float = 0.40
    liquid_holdup: float = 0.12
    cp_cat: float = CATALYST_CP
    heater_tau: float = 300.0
    separator_tau: float = 300.0
    loop_volume: float = 30.0
    makeup_capacity: float = 1.5
    relief_margin: float = 0.05
    tau_relief: float = 60.0


def _interp_profile(bed, xi_out: np.ndarray):
    """The steady bed's outflow and T interpolated at normalised positions ``xi_out``."""
    xi = np.asarray(bed.xi)
    T = jnp.asarray(np.interp(xi_out, xi, np.asarray(bed.T)))

    def at(a):
        a = np.asarray(a)
        flat = a.reshape(a.shape[0], -1)
        return jnp.asarray(np.stack([np.interp(xi_out, xi, flat[:, j]) for j in range(flat.shape[1])], axis=1)
                           .reshape((len(xi_out),) + a.shape[1:]))

    prof = bed.profile
    return Flows(at(prof.gas), at(prof.cut), at(prof.attr)), T


class DynamicHydrotreater:
    """A dynamic hydrotreater (see the module docstring). Build with :meth:`from_steady`.

    Attributes:
        flowsheet: The :class:`~difflow.dynamic.DynamicFlowsheet`.
        reactor, separator, loop, heater: Its units.
        y0: The initial state built from the steady solution.
        base_signals: The signals' design values.
        steady: The steady :class:`~difflow_refinery.hydrotreating.HydrotreaterResult`.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # ----- construction ----------------------------------------------------------

    @classmethod
    def from_steady(cls, unit, feed: Mapping, result=None, *, n_cells: int = 10, n_sub: int = 1,
                    holdups: HoldupParams | None = None) -> "DynamicHydrotreater":
        """The dynamic twin of a steady :class:`~difflow_refinery.hydrotreating.Hydrotreater`.

        Args:
            unit: The steady unit (its layout, kinetics and specs are used).
            feed: The feed stream it was solved for.
            result: Its solved result (solved here if omitted).
            n_cells: Cells per bed.
            n_sub: RK4 steps of the steady bed equations per cell.
            holdups: :class:`HoldupParams`.

        The initial state is the steady solution mapped onto the cells (the
        steady profile at each cell outlet, turned into the holdup that drains
        at that rate); it is close to, not exactly at, the dynamic model's own
        steady state -- integrate a while (:meth:`settle`) to start at rest.
        """
        hp = holdups or HoldupParams()
        p = unit.params
        if p.quench is None:
            raise NotImplementedError("the dynamic hydrotreater needs given quench rates "
                                      "(HydrotreaterParams.quench), not solved ones")
        res = unit.solve(feed) if result is None else result
        th = unit.theta(feed)
        comps = unit._comps(th)
        kin = unit._kin_params(th["kin"])
        lay = unit.layout
        n = th["num"]
        P = float(n["P"])
        P_hps = float(n["P"] - n["loop_dP"])
        T_hps = float(n["hps_T"])
        rho_cat = float(n["catalyst_density"])
        q = np.asarray(p.quench, dtype=float)
        q_tot = float(q.sum())

        # --- reactor: geometry, design residence times, initial holdups
        beds, y_rx, sc_rx = [], [], []
        xi_out = np.arange(1, n_cells + 1) / n_cells
        for b, bed in enumerate(res.reactor.beds):
            W = float(np.asarray(res.outputs["catalyst.mass"])) * float(p.bed_fractions[b])
            km = bed.k_model
            F_V, F_L, v_L = phase_flows(bed.inlet, lay, comps, km, bed.T_in)
            V_cell = W / rho_cat / n_cells
            eps_G = hp.voidage - hp.liquid_holdup
            Q_V = float(F_V) * R_GAS * float(bed.T_in) / P
            Q_L = float(F_L) * float(v_L)
            tau_G = eps_G * V_cell / Q_V
            tau_L = hp.liquid_holdup * V_cell / Q_L if Q_L > 1e-12 * Q_V else tau_G
            beds.append(BedGeometry(W=W, tau_G=tau_G, tau_L=tau_L))
            Fk, Tk = _interp_profile(bed, xi_out)
            hk = jax.vmap(lambda f, t: holdup_from_outflow(f, lay, comps, km, t, tau_G, tau_L))(Fk, Tk)
            sk = jax.vmap(flows_scale)(hk)
            pack = jax.vmap(lambda f: f.ravel())
            y_rx.append(jnp.concatenate([pack(hk), Tk[:, None]], axis=1))
            sc_rx.append(jnp.concatenate([pack(sk), jnp.full((n_cells, 1), 100.0)], axis=1))
        scale_rx = jnp.stack(sc_rx).reshape(-1)
        y0_rx = jnp.stack(y_rx).reshape(-1) / scale_rx
        shares = tuple(q / q_tot) if q_tot > 0 else tuple(np.ones(len(q)) / max(len(q), 1))
        reactor = DynamicTrickleBedReactor(lay, unit.kinetics, kin, comps, beds, P, quench_shares=shares,
                                           n_cells=n_cells, n_sub=n_sub, cp_cat=hp.cp_cat,
                                           scale=scale_rx, y0=y0_rx)

        # --- separator liquid holdup
        liq = res.streams["hps_liquid"]
        n_liq = liq.scale(hp.separator_tau)
        s_liq = flows_scale(n_liq)
        hps = DynamicHPSeparator(lay, comps, T_hps, P_hps, hp.separator_tau, s_liq,
                                 n_liq.ravel() / s_liq.ravel())

        # --- loop inventory
        rec = res.streams["recycle"]
        n_design = P_hps * hp.loop_volume / (R_GAS * T_hps)
        F_rec = float(jnp.sum(rec.gas))
        tau_loop = n_design / F_rec
        n_loop = rec.gas * tau_loop
        s_loop = jnp.maximum(jnp.abs(n_loop), 1e-10 * jnp.max(jnp.abs(n_loop)) + 1e-300)
        h2_target = float(n["h2_oil"]) * float(unit._feed_volume(th)) * MOL_PER_NM3
        M_design = float(res.outputs["h2.makeup"]) / float(th["makeup_y"][lay.gas_index("hydrogen")])
        loop = RecycleGasLoop(
            lay, comps, h2_target=h2_target, makeup_y=th["makeup_y"], purge=n["purge"],
            h2s_removal=n["h2s_removal"], nh3_removal=n["nh3_removal"], tau=tau_loop, P_suction=P_hps,
            P_discharge=P, T_suction=T_hps, eta=n["compressor_eta"], makeup_max=hp.makeup_capacity * M_design,
            relief_inventory=(1.0 + hp.relief_margin) * n_design, tau_relief=hp.tau_relief,
            scale=s_loop, y0=n_loop / s_loop)

        heater = FurnaceLag(("oil", "gas"), "T_heater", hp.heater_tau, P, float(p.T_in[0]))

        oil = th["oil"]
        feed_T = th["feed_T"]
        ai = {a: i for i, a in enumerate(lay.attributes)}
        s_cols = jnp.asarray([ai[f"S_{k}"] for k in SULFUR_CLASSES])

        def oil_feed(t, args):
            s = sig.evaluate(t, args)
            f = oil.scale(s["feed_rate"])
            f = Flows(f.gas, f.cut, f.attr.at[:, s_cols].multiply(s["feed_sulfur"]))
            return f.to_stream(lay, feed_T, P)

        def signals_feed(t, args):
            return sig.evaluate(t, args)

        fs = DynamicFlowsheet(species_order=lay.keys())
        fs.add_feed("oil", oil_feed)
        fs.add_feed("signals", signals_feed)
        fs.add_unit(heater, inlets={"oil": "oil", "gas": "treat_gas", "signals": "signals"},
                    outlets={"outlet": "rx_in"})
        fs.add_unit(reactor, inlets={"feed": "rx_in", "quench": "quench_gas"}, outlets={"outlet": "effluent"})
        fs.add_unit(hps, inlets={"inlet": "effluent"},
                    outlets={"vapor": "hps_vapor", "liquid": "hps_liquid", "water": "hps_water"})
        fs.add_unit(loop, inlets={"vapor": "hps_vapor", "signals": "signals"},
                    outlets={"treat_gas": "treat_gas", "quench_gas": "quench_gas", "makeup": "makeup",
                             "vent": "vent", "purge": "purge", "acid_gas": "acid_gas"})
        base = {"T_heater": float(p.T_in[0]), "compressor": 1.0, "quench": q_tot, "feed_rate": 1.0,
                "feed_sulfur": 1.0}
        y0 = jnp.concatenate([heater.initial_state(), y0_rx, hps.initial_state(), loop.initial_state()])
        return cls(flowsheet=fs, reactor=reactor, separator=hps, loop=loop, heater=heater, y0=y0,
                   base_signals=base, steady=res, unit=unit, layout=lay, comps=comps, holdups=hp,
                   n_cells=n_cells, n_sub=n_sub, oil=oil, tau_loop=tau_loop)

    # ----- signals and simulation ------------------------------------------------

    def args(self, steps: Mapping[str, Sequence[tuple[float, float]]] | None = None,
             ramp: float | Mapping[str, float] = sig.DEFAULT_RAMP, **base) -> dict:
        """Integrator args: the design signals, overridden by ``base``, with ``steps``.

        Signals: ``T_heater`` (bed-1 inlet setpoint, K), ``compressor``
        (recycle compressor availability, 1 running, 0 tripped), ``quench``
        (quench gas as a fraction of the treat gas), ``feed_rate`` and
        ``feed_sulfur`` (multipliers of the design feed and of its sulfur).
        """
        b = dict(self.base_signals)
        b.update(base)
        return sig.signal_args(b, steps, ramp)

    def simulate(self, t_end: float, args: dict | None = None, y0: Array | None = None, n_save: int = 61,
                 saveat: Array | None = None, method: str = "diffrax:tsit5", rtol: float = DEFAULT_RTOL,
                 atol: float = DEFAULT_ATOL, max_steps: int = 100_000, dtmax: float = 60.0, **kwargs):
        """Integrate from ``y0`` (default :attr:`y0`) to ``t_end`` seconds.

        Returns the flowsheet's :class:`~difflow.dynamic.flowsheet.DynamicFlowsheetResult`
        (``trajectory.t``, ``trajectory.y``). ``kwargs`` go to
        ``diffrax.diffeqsolve`` (``adjoint=`` for gradients: the default
        ``RecursiveCheckpointAdjoint`` is reverse mode, ``diffrax.ForwardMode()``
        forward mode). ``dtmax`` stops the step from growing across a quiet
        spell into a disturbance (at rest the error estimate is zero).
        """
        args = self.args() if args is None else args
        y0 = self.y0 if y0 is None else y0
        ts = jnp.linspace(0.0, t_end, n_save) if saveat is None else jnp.asarray(saveat)
        return self.flowsheet.simulate((0.0, float(t_end)), y0=y0, method=method, args=args,
                                       rtol=rtol, atol=atol, saveat=ts, max_steps=max_steps, dtmax=dtmax,
                                       dt0=kwargs.pop("dt0", 1.0), **kwargs)

    def settle(self, t_end: float = 4 * 3600.0, args: dict | None = None, **kw) -> Array:
        """The state after ``t_end`` s at constant signals (a start at rest)."""
        return self.simulate(t_end, args=args, n_save=2, **kw).y_final

    # ----- what a trajectory shows ------------------------------------------------

    def _split(self, y):
        fs = self.flowsheet
        fs._update_state_indices()
        return {e.name: y[e.state_slice] for e in fs.units}

    def cut_sulfur_wppm(self, f: Flows) -> Array:
        """Sulfur on the cuts of ``f`` (wppm of the cuts' mass): what the stripper leaves in the product."""
        lay = self.layout
        ai = {a: i for i, a in enumerate(lay.attributes)}
        S = jnp.sum(f.attr[:, [ai[f"S_{k}"] for k in SULFUR_CLASSES]]) * 32.065e-3
        return 1e6 * S / jnp.sum(f.cut_mass(lay))

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Headline quantities at one time: temperatures, liquid sulfur, hydrogen, loop."""
        args = self.args() if args is None else args
        fs = self.flowsheet
        streams = fs.outputs(jnp.asarray(t), y, None, args)
        us = self._split(y)
        inputs = {"feed": streams["rx_in"], "quench": streams["quench_gas"]}
        prof = self.reactor.profiles(us["reactor"], inputs)
        lay = self.layout
        liq = Flows.from_stream(streams["hps_liquid"], lay)
        comp = self.loop.compressor(us["loop"], streams["signals"])
        W = jnp.asarray([g.W for g in self.reactor.beds])
        T = prof["T"]
        return {
            "T_heater": 100.0 * us["heater"][0],
            "bed_T_in": prof["T_in"], "bed_T_out": T[:, -1], "cell_T": T,
            "wabt": jnp.sum(W * (prof["T_in"] + 2.0 * T[:, -1]) / 3.0) / jnp.sum(W),
            "liquid.S_wppm": self.cut_sulfur_wppm(liq),
            "reactor_out.S_wppm": self.cut_sulfur_wppm(jax.tree_util.tree_map(lambda a: a[-1, -1],
                                                                               prof["outflow"])),
            "recycle.rate": comp["recycle"], "compressor.T_out": comp["T_out"], "h2.makeup": comp["makeup"],
            "vent.rate": jnp.sum(Flows.from_stream(streams["vent"], lay).gas),
            "loop.inventory": jnp.sum(self.loop.inventory(us["loop"])),
        }

    def trajectory(self, result, args: dict | None = None, keys: Sequence[str] | None = None) -> dict:
        """:meth:`observe` at every saved time of a simulation, as arrays with a leading time axis."""
        ts, ys = result.trajectory.t, result.trajectory.y
        args = self.args() if args is None else args
        f = jax.jit(lambda t, y, a: self.observe(t, y, a))
        obs = [f(t, y, args) for t, y in zip(ts, ys)]
        keys = keys or list(obs[0])
        out = {k: jnp.stack([o[k] for o in obs]) for k in keys}
        out["t"] = ts
        return out

    def steady_reference(self) -> dict:
        """The steady unit's values of the quantities :meth:`observe` reports."""
        r = self.steady
        o = r.outputs
        beds = r.reactor.beds
        return {"wabt": o["wabt"], "bed_T_in": jnp.stack([b.T_in for b in beds]),
                "bed_T_out": jnp.stack([b.T_out for b in beds]),
                "liquid.S_wppm": self.cut_sulfur_wppm(r.streams["hps_liquid"]),
                "reactor_out.S_wppm": self.cut_sulfur_wppm(r.streams["reactor_out"]),
                "recycle.rate": jnp.sum(r.streams["recycle"].gas), "compressor.T_out": o["compressor.T_out"],
                "h2.makeup": o["h2.makeup"], "product.S_wppm": o["product.S_wppm"]}

    # ----- balances -----------------------------------------------------------------

    def inventory(self, y) -> Flows:
        """Total material held (mol): every reactor cell, the separator liquid, the loop gas."""
        us = self._split(y)
        n, _ = self.reactor.unpack(us["reactor"])
        held = jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=(0, 1)), n)
        held = held + self.separator.holdup(us["hps"])
        lay = self.layout
        return held + Flows(self.loop.inventory(us["loop"]), jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))

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
        outflow = F("hps_liquid") + F("hps_water") + F("purge") + F("acid_gas") + F("vent")
        return {"accumulation": acc, "in_minus_out": inflow.elements(lay) - outflow.elements(lay)}


__all__ = ["HoldupParams", "DynamicHydrotreater", "DEFAULT_RTOL", "DEFAULT_ATOL"]
