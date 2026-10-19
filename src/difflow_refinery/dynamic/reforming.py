"""A dynamic catalytic-reformer reactor train: catalyst heat capacity, furnace lags, coke on catalyst.

The reformer's beds are vapour phase: the gas passes a bed in a few seconds,
the catalyst's heat capacity takes the bed tens of minutes to swing, and the
coke on the catalyst moves over weeks. The dynamic model keeps the two slow
ones as states and treats the gas as quasi-steady:

* Each reactor is ``N`` cells. A cell has a **temperature** ``T`` (its
  catalyst's, shared with the gas leaving it) and a **coke** loading ``C``
  (kg coke / kg catalyst).
* The gas entering a cell -- the previous cell's outflow at its
  temperature -- reacts across the cell's catalyst adiabatically, by the
  **steady bed integration** :func:`difflow_refinery.reforming.reactor.integrate_bed`
  (``dF/dW = nu^T r``, temperature from the conserved total enthalpy;
  adaptive Tsit5), to ``(F*, T*)``; ``F*`` leaves the cell at ``T``. The cell's energy and coke balances::

      W_c c_p,cat dT/dt = H(F_in, T_in) - H(F*, T)        (= H(F*, T*) - H(F*, T))
      dC/dt = (coke made across the cell) / W_c

  ``H`` is the reformer's one enthalpy basis (heats of formation + ideal-gas
  sensible heat + PR vapour departure), so the cell's energy balance is a
  true one: what the gas loses (the reaction's endotherm) the catalyst
  gives up.
* With ``coke_alpha`` set, a cell's activity is ``exp(-alpha C)`` (the CCR
  module's form, :func:`difflow_refinery.reforming.ccr.activity`; ``alpha``
  ILLUSTRATIVE), multiplying the reformer's ``kinetics.activity``: the
  semi-regenerative cycle as a trajectory. Without it the coke is a
  bookkeeping state and the train has a steady state, which is the steady
  reformer's: at rest ``T = T*``, so the train is the steady beds, each
  integrated in ``N`` pieces.

The interstage heaters are :class:`~difflow_refinery.dynamic.hydroprocessing.FurnaceLag`
(an outlet-temperature lag on each reactor's inlet setpoint). By default the
recycle gas and the separator are not dynamic: the train takes the naphtha
and the steady unit's recycle as feeds, and the reformate is reported by
running the steady separator and stabilizer on the train's effluent at each
time. With ``recycle_loop=True`` (:meth:`DynamicReformerTrain.from_steady`)
the separator and the recycle loop are holdups too
(:class:`DynamicReformerLoop`): the recycle then responds to the effluent
and a compressor trip starves the reactors of hydrogen.

Stiffness: the gas is quasi-steady, so the fastest state is a cell's
temperature (``W_c c_p / (F c_p)``, about a minute per cell at the default
five cells), and the slowest the coke (weeks). Hours run with Tsit5
(:meth:`DynamicReformerTrain.simulate`); a cycle of weeks to months runs on
the slow time scale alone (:meth:`DynamicReformerTrain.cycle`: the
temperatures on their rest manifold, only the coke integrated), the
two-time-scale reduction of the same model.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.dynamic import DynamicFlowsheet
from difflow.dynamic.state import StateSpec, StateVar
from difflow_refinery.dynamic import signals as sig
from difflow_refinery.dynamic.hydroprocessing import CATALYST_CP, FurnaceLag
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.reactor import flows_of, integrate_bed, stream_of

jax.config.update("jax_enable_x64", True)

#: Coke state scale (kg/kg): a fresh-to-end-of-run loading is 0-0.2.
COKE_SCALE = 0.01


class DynamicReformingReactor:
    """One adiabatic reforming bed as ``n_cells`` cells (see the module docstring).

    Port ``"inlet"`` (a reformer-species stream), output ``"outlet"``.
    State: the cells' temperatures (scaled by 100 K) then their coke
    loadings (scaled by :data:`COKE_SCALE`).

    Args:
        kinetics: :class:`~difflow_refinery.reforming.kinetics.ReformingKinetics`.
        W: Catalyst mass, kg.
        P: Bed pressure, Pa.
        n_cells: Cells.
        rtol: Tolerance of each cell's bed integration.
        adjoint: The cells' bed-integration adjoint, ``"reverse"`` (default:
            ``jax.grad`` of a trajectory) or ``"forward"`` (``jax.jacfwd``).
        cp_cat: Catalyst heat capacity, J/(kg K).
        coke_alpha: ``a = exp(-alpha C)`` per cell, or ``None`` (coke does not
            deactivate: the steady reformer's assumption).
        T0, C0: Initial cell temperatures (K) and coke (kg/kg), ``(n_cells,)``.
    """

    output_ports = ("outlet",)

    def __init__(self, kinetics, W: float, P: float, n_cells: int = 5, rtol: float = 1e-6,
                 adjoint: str = "reverse", cp_cat: float = CATALYST_CP, coke_alpha: float | None = None,
                 T0: Sequence[float] | None = None, C0: Sequence[float] | None = None, name: str = "rx"):
        self.kinetics = kinetics
        self.W = jnp.asarray(W, dtype=float)
        self.P = jnp.asarray(P, dtype=float)
        self.n_cells = int(n_cells)
        self.rtol = float(rtol)
        self.adjoint = adjoint
        self.cp_cat = jnp.asarray(cp_cat, dtype=float)
        self.coke_alpha = coke_alpha
        self.T0 = np.full(self.n_cells, 773.15) if T0 is None else np.asarray(T0, dtype=float)
        self.C0 = np.zeros(self.n_cells) if C0 is None else np.asarray(C0, dtype=float)
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"cell{k + 1}.T", "temperature", "K", "", scale=100.0)
                          for k in range(self.n_cells)]
                         + [StateVar(f"cell{k + 1}.coke", "generic", "kg/kg", "", scale=COKE_SCALE)
                            for k in range(self.n_cells)])

    def initial_state(self, inputs=None, params=None):
        return jnp.concatenate([jnp.asarray(self.T0) / 100.0, jnp.asarray(self.C0) / COKE_SCALE])

    def unpack(self, state):
        return 100.0 * state[:self.n_cells], COKE_SCALE * state[self.n_cells:]

    def activity(self, C):
        if self.coke_alpha is None:
            return jnp.ones_like(C)
        return jnp.exp(-jnp.asarray(self.coke_alpha) * C)

    def _segment(self, F_in, T_in, Wc, a):
        """``(F*, T*, coke kg/s, H_in)`` across one cell at activity multiplier ``a``.

        The steady reformer's own adaptive bed integration
        (:func:`~difflow_refinery.reforming.reactor.integrate_bed`) over the
        cell's catalyst: a bed split into cells is the same bed, so the
        train's rest state is the steady reformer's to the integration
        tolerance. (Fixed RK4 steps are not: the dehydrogenation equilibria at
        a bed inlet need more than a thousand of them to be stable.)
        """
        kin = dataclasses.replace(self.kinetics, activity=jnp.asarray(self.kinetics.activity) * a)
        F, T, coke, _ = integrate_bed(F_in, T_in, self.P, Wc, kin, iso_fraction=float(self.kinetics.iso_fraction),
                                      adjoint=self.adjoint, rtol=self.rtol)
        return F, T, coke, th.total_enthalpy(F_in, T_in, self.P)

    def _run(self, inlet, state):
        T, C = self.unpack(state)
        F_in = flows_of(inlet)
        T_in = jnp.asarray(inlet["T"], dtype=float)
        Wc = self.W / self.n_cells
        a = self.activity(C)

        def body(carry, xs):                      # cells in sequence: each takes the last one's gas
            F, Tk = carry
            T_cell, a_k = xs
            Fk, Tsk, ck, Hk = self._segment(F, Tk, Wc, a_k)
            return (Fk, T_cell), (Fk, Tsk, ck, Hk)

        _, (Fs, Ts, cokes, Hs) = jax.lax.scan(body, (F_in, T_in), (T, a))
        return Fs, Ts, cokes, Hs

    def rest(self, inlet, C):
        """The cells at thermal rest at coke ``C``: ``(outlet F, outlet T, coke made per cell kg/s)``.

        Each cell at its own segment's outlet temperature (``T = T*``), the
        fast manifold the slow coke moves on (:meth:`DynamicReformerTrain.cycle`).
        """
        F_in = flows_of(inlet)
        T_in = jnp.asarray(inlet["T"], dtype=float)
        Wc = self.W / self.n_cells

        def body(carry, a_k):
            F, Tk = carry
            Fk, Tsk, ck, _ = self._segment(F, Tk, Wc, a_k)
            return (Fk, Tsk), ck

        (F, T), cokes = jax.lax.scan(body, (F_in, T_in), self.activity(C))
        return F, T, cokes

    def derivatives(self, t, state, inputs, params=None):
        T, _ = self.unpack(state)
        Fs, _, cokes, Hs = self._run(inputs["inlet"], state)
        Wc = self.W / self.n_cells
        H_out = jax.vmap(lambda F, t_: th.total_enthalpy(F, t_, self.P))(Fs, T)
        dT = (Hs - H_out) / (Wc * self.cp_cat)
        dC = cokes / Wc
        return jnp.concatenate([dT / 100.0, dC / COKE_SCALE])

    def outputs(self, t, state, inputs, params=None):
        T, _ = self.unpack(state)
        Fs, _, _, _ = self._run(inputs["inlet"], state)
        return {"outlet": stream_of(Fs[-1], T[-1], self.P)}

    def report(self, state, inlet) -> dict:
        """``{"T": cells K, "coke": cells kg/kg, "activity", "coke_make": kg/s, "T_out", "T_in"}``."""
        T, C = self.unpack(state)
        _, _, cokes, _ = self._run(inlet, state)
        return {"T": T, "coke": C, "activity": self.activity(C), "coke_make": jnp.sum(cokes),
                "T_out": T[-1], "T_in": jnp.asarray(inlet["T"], dtype=float)}


class DynamicReformerLoop:
    """The reformer's product separator and recycle-gas loop with holdups.

    Ports ``"effluent"`` (the last reactor's outlet) and ``"signals"``. The
    effluent is cooled and flashed at the separator's ``(T, P)`` by the
    steady :class:`~difflow_refinery.reforming.separation.ProductSeparator`
    (instantaneous). Two holdups, the state:

    * the separator **liquid** ``n_L`` (every species, mol), drained to the
      stabilizer with residence time ``tau_L`` (a level controller on the
      outlet valve, idealised as proportional drainage);
    * the loop's **gas** inventory ``n_G`` (separator vapour space, piping,
      compressor suction drum), fed by the separator vapour and drained by
      the recycle compressor and the net gas::

          dn_G/dt = V - R - N_net

      The compressor holds the steady unit's recycle spec, ``H2/HC`` mol of
      hydrogen per mol of naphtha hydrocarbon: ``R = c(t) min(need / y_H2,
      R_max) y`` (``y`` the inventory's composition, ``c`` the signal
      ``"compressor"``: 1 running, 0 tripped; ``R_max`` its capacity). The net
      gas leaves on the loop's pressure controller, proportional to the
      inventory above a set level: ``N_net = max(N - N_lo, 0) / tau_p``, with
      ``N_lo`` chosen so that the design inventory delivers the design net gas.

    At rest the inventory's composition is the separator vapour's and the
    recycle is ``need / H2_vapour`` of it: the steady splitter
    (:class:`~difflow_refinery.reforming.separation.RecycleSplitter`), so the
    rest state is the steady reformer's. Outputs ``"recycle"`` (compressed to
    the reactor pressure by the steady compressor), ``"net_gas"`` and
    ``"liquid"`` depend on the state and the signals alone, which breaks the
    loop for the flowsheet. Pressures are fixed (the controller's level is a
    moles inventory, not a pressure); the compressor's power is not modelled
    beyond its discharge temperature.
    """

    output_ports = ("recycle", "net_gas", "liquid")
    output_dependencies = {"recycle": ("signals",), "net_gas": (), "liquid": ()}

    def __init__(self, separator, compressor, *, H2_HC: float, hydrocarbon_feed: float, tau_L: float,
                 N_design: float, net_design: float, tau_p: float, R_max: float, y0: Array,
                 T_sep: float, P_sep: float, name: str = "loop"):
        self.separator = separator
        self.compressor = compressor
        self.H2_HC = jnp.asarray(H2_HC, dtype=float)
        self.hydrocarbon_feed = jnp.asarray(hydrocarbon_feed, dtype=float)
        self.tau_L = jnp.asarray(tau_L, dtype=float)
        self.tau_p = jnp.asarray(tau_p, dtype=float)
        self.N_lo = jnp.asarray(N_design - tau_p * net_design, dtype=float)
        self.R_max = jnp.asarray(R_max, dtype=float)
        self.T_sep = jnp.asarray(T_sep, dtype=float)
        self.P_sep = jnp.asarray(P_sep, dtype=float)
        y0 = jnp.asarray(y0, dtype=float)
        self._scale = jnp.maximum(jnp.abs(y0), 1e-10 * jnp.max(jnp.abs(y0)) + 1e-300)
        self._y0 = y0 / self._scale
        self.name = name

    def state_spec(self) -> StateSpec:
        return StateSpec([StateVar(f"liquid.{k}", "moles", "mol", "") for k in sp.NAMES]
                         + [StateVar(f"gas.{k}", "moles", "mol", "") for k in sp.NAMES])

    def initial_state(self, inputs=None, params=None):
        return self._y0

    def holdups(self, state):
        """``(liquid mol, gas mol)``, each ``(N_SPECIES,)``."""
        x = state * self._scale
        return x[:sp.N_SPECIES], x[sp.N_SPECIES:]

    def _composition(self, state):
        _, nG = self.holdups(state)
        nG = jnp.maximum(nG, 0.0)
        return nG / jnp.maximum(jnp.sum(nG), 1e-300)

    def _recycle_rate(self, state, signals):
        y = self._composition(state)
        yH2 = jnp.maximum(y[sp.INDEX["H2"]], 1e-12)
        need = self.H2_HC * self.hydrocarbon_feed * signals["feed_rate"]
        return signals["compressor"] * jnp.minimum(need / yH2, self.R_max)

    def _recycle_flows(self, state, signals):
        return self._recycle_rate(state, signals) * self._composition(state)

    def _net_flows(self, state):
        _, nG = self.holdups(state)
        nG = jnp.maximum(nG, 0.0)
        N = jnp.sum(nG)
        return jnp.maximum(N - self.N_lo, 0.0) / self.tau_p * nG / jnp.maximum(N, 1e-300)

    def _liquid_flows(self, state):
        nL, _ = self.holdups(state)
        return jnp.maximum(nL, 0.0) / self.tau_L

    def derivatives(self, t, state, inputs, params=None):
        liq, vap, _ = self.separator(inputs["effluent"])
        nL, _ = self.holdups(state)
        dL = flows_of(liq) - self._liquid_flows(state)
        dG = flows_of(vap) - self._recycle_flows(state, inputs["signals"]) - self._net_flows(state)
        return jnp.concatenate([dL, dG]) / self._scale

    def _state_outputs(self, state):
        return {"net_gas": stream_of(self._net_flows(state), self.T_sep, self.P_sep),
                "liquid": stream_of(self._liquid_flows(state), self.T_sep, self.P_sep)}

    def _recycle(self, state, signals):
        # the discharge temperature is intensive: compress one mol/s of the
        # inventory's gas (finite with the compressor tripped) and scale it
        y = self._composition(state)
        out, _ = self.compressor(stream_of(y, self.T_sep, self.P_sep))
        return stream_of(self._recycle_rate(state, signals) * y, out["T"], out["P"])

    def outputs(self, t, state, inputs, params=None):
        out = self._state_outputs(state)
        out["recycle"] = self._recycle(state, inputs["signals"])
        return out

    def partial_outputs(self, t, state, inputs, params=None):
        out = self._state_outputs(state)
        if "signals" in inputs:
            out["recycle"] = self._recycle(state, inputs["signals"])
        return out


class DynamicReformerTrain:
    """The reformer's heaters and reactors as a dynamic flowsheet, built with :meth:`from_steady`.

    Feeds: ``"naphtha"`` (the steady feed, times the ``feed_rate`` signal),
    ``"recycle"`` (the steady unit's converged recycle gas, held), and
    ``"signals"``. Signals: ``T_in1``, ``T_in2``, ... (each reactor's inlet
    setpoint, K) and ``feed_rate``.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_steady(cls, reformer, feed, result=None, *, n_cells: int = 5, rtol: float = 1e-6,
                    adjoint: str = "reverse",
                    cp_cat: float = CATALYST_CP, heater_tau: float = 300.0,
                    coke_alpha: float | None = None, recycle_loop: bool = False,
                    loop_volume: float = 30.0, separator_tau: float = 300.0, pressure_tau: float = 30.0,
                    compressor_capacity: float = 1.3) -> "DynamicReformerTrain":
        """The dynamic train of a solved :class:`~difflow_refinery.reforming.CatalyticReformer`.

        The cells start at the steady bed's temperature profile and with no
        coke (start of run).

        With ``recycle_loop`` the separator and the recycle-gas loop are
        dynamic too (:class:`DynamicReformerLoop`: separator liquid holdup,
        a gas inventory of ``loop_volume`` m^3 at the separator conditions,
        the compressor on the H2/HC spec up to ``compressor_capacity`` times
        its design rate, the net gas on a pressure controller with time
        constant ``pressure_tau``); without it the recycle is the steady
        unit's converged recycle, held. ILLUSTRATIVE holdups.
        """
        p = reformer.params
        res = reformer.solve(feed) if result is None else result
        P = float(p.P_reactor)
        Ws = [float(r["W"]) for r in res.reactors]
        nr = p.n_reactors
        fs = DynamicFlowsheet(species_order=[f"F_{s}" for s in sp.NAMES])
        naphtha = feed.stream()
        rec = res.streams["recycle"]

        def naphtha_feed(t, args):
            s = sig.evaluate(t, args)
            out = {k: (v * s["feed_rate"] if k.startswith("F_") else v) for k, v in naphtha.items()}
            return out

        fs.add_feed("naphtha", naphtha_feed)
        if not recycle_loop:
            fs.add_feed("recycle", dict(rec))
        fs.add_feed("signals", lambda t, args: sig.evaluate(t, args))
        reactors, heaters = [], []
        for k in range(nr):
            kin = p.kinetics
            if p.activity is not None:
                kin = dataclasses.replace(kin, activity=jnp.asarray(kin.activity, dtype=float) * p.activity[k])
            r_in = res.streams[f"rx{k + 1}_in"]
            _, _, _, prof = integrate_bed(flows_of(r_in), jnp.asarray(r_in["T"]), jnp.asarray(P),
                                          jnp.asarray(Ws[k]), kin, iso_fraction=float(kin.iso_fraction),
                                          n_save=n_cells + 1)
            T0 = np.asarray(prof[2])[1:]
            inl = ("naphtha", "recycle") if k == 0 else (f"rx{k}_out",)
            h = FurnaceLag(inl, f"T_in{k + 1}", heater_tau, P, float(p.inlet_T[k]), name=f"heater{k + 1}")
            r = DynamicReformingReactor(kin, Ws[k], P, n_cells=n_cells, rtol=rtol, adjoint=adjoint, cp_cat=cp_cat,
                                        coke_alpha=coke_alpha, T0=T0, name=f"rx{k + 1}")
            fs.add_unit(h, inlets={**{s: s for s in inl}, "signals": "signals"},
                        outlets={"outlet": f"rx{k + 1}_in"})
            fs.add_unit(r, inlets={"inlet": f"rx{k + 1}_in"}, outlets={"outlet": f"rx{k + 1}_out"})
            reactors.append(r)
            heaters.append(h)
        base = {f"T_in{k + 1}": float(p.inlet_T[k]) for k in range(nr)}
        base["feed_rate"] = 1.0
        loop = None
        if recycle_loop:
            from difflow.units.eos_units import Compressor, CompressorParams
            from difflow_refinery.reforming.separation import ProductSeparator, ProductSeparatorParams
            sep = ProductSeparator(ProductSeparatorParams(T=p.T_separator, P=p.P_separator))
            comp = Compressor(CompressorParams(P_out=P, eta_isentropic=p.compressor_efficiency,
                                               T_bounds=(200.0, 800.0)), th.cubic_thermo())
            T_sep, P_sep = float(p.T_separator), float(p.P_separator)
            F_v = flows_of(res.streams["sep_vapor"])
            F_l = flows_of(res.streams["sep_liquid"])
            N_design = P_sep * loop_volume / (8.314462618 * T_sep)
            net = float(jnp.sum(flows_of(res.streams["net_gas"])))
            R_design = float(jnp.sum(flows_of(rec)))
            nG0 = F_v / jnp.sum(F_v) * N_design
            loop = DynamicReformerLoop(sep, comp, H2_HC=float(p.H2_HC),
                                       hydrocarbon_feed=float(feed.hydrocarbon_moles), tau_L=separator_tau,
                                       N_design=N_design, net_design=net, tau_p=pressure_tau,
                                       R_max=compressor_capacity * R_design, y0=jnp.concatenate([F_l * separator_tau, nG0]),
                                       T_sep=T_sep, P_sep=P_sep)
            fs.add_unit(loop, inlets={"effluent": f"rx{nr}_out", "signals": "signals"},
                        outlets={"recycle": "recycle", "net_gas": "net_gas", "liquid": "sep_liquid"})
            base["compressor"] = 1.0
        y0 = fs.initial_state(None, sig.signal_args(base))
        # the fastest state: a cell's catalyst heat capacity over the heat flow of the gas through it
        taus = []
        for k, r in enumerate(reactors):
            s = res.streams[f"rx{k + 1}_in"]
            F, T = flows_of(s), jnp.asarray(s["T"])
            C_flow = jax.jvp(lambda t: th.total_enthalpy(F, t, r.P), (T,), (jnp.ones_like(T),))[1]
            taus.append(float(r.W / r.n_cells * r.cp_cat / C_flow))
        return cls(flowsheet=fs, reactors=reactors, heaters=heaters, loop=loop, y0=y0, base_signals=base,
                   steady=res, reformer=reformer, feed=feed, n_cells=n_cells, tau_thermal_min=min(taus))

    def args(self, steps: Mapping[str, Sequence[tuple[float, float]]] | None = None,
             ramp: float | Mapping[str, float] = sig.DEFAULT_RAMP, **base) -> dict:
        """Integrator args: the design signals, overridden by ``base``, with ``steps``."""
        b = dict(self.base_signals)
        b.update(base)
        return sig.signal_args(b, steps, ramp)

    def simulate(self, t_end: float, args: dict | None = None, y0: Array | None = None, n_save: int = 61,
                 saveat: Array | None = None, method: str = "diffrax:tsit5", rtol: float = 1e-5,
                 atol: float = 1e-6, max_steps: int = 100_000, dtmax: float | None = None, **kwargs):
        """Integrate from ``y0`` (default :attr:`y0`) to ``t_end`` seconds (see
        :meth:`DynamicHydrotreater.simulate <difflow_refinery.dynamic.hydrotreater.DynamicHydrotreater.simulate>`).

        The tolerances are on the scaled states (temperatures in 100 K):
        looser than the hydrotreater's because every cell's bed integration
        carries its own tolerance (``rtol`` of :meth:`from_steady`) into the
        right-hand side. ``dtmax`` (default twice the smallest cell's thermal
        time constant, :attr:`tau_thermal_min`) keeps the explicit step inside
        its stability region -- near the boundary the cells chatter at rest
        -- and stops the step from growing across a quiet spell into a
        disturbance (at rest the error estimate is zero).
        """
        dtmax = 2.0 * self.tau_thermal_min if dtmax is None else dtmax
        args = self.args() if args is None else args
        y0 = self.y0 if y0 is None else y0
        ts = jnp.linspace(0.0, t_end, n_save) if saveat is None else jnp.asarray(saveat)
        return self.flowsheet.simulate((0.0, float(t_end)), y0=y0, method=method, args=args, rtol=rtol,
                                       atol=atol, saveat=ts, max_steps=max_steps, dtmax=dtmax,
                                       dt0=kwargs.pop("dt0", 1.0), **kwargs)

    # ----- the slow time scale: a cycle of coking ------------------------------------

    def _rest_train(self, t, C_all, args):
        """The train at thermal rest at coke loadings ``C_all`` ``(n_reactors, n_cells)``."""
        s = sig.evaluate(t, args)
        nap = {k: (v * s["feed_rate"] if k.startswith("F_") else v) for k, v in self.feed.stream().items()}
        F = flows_of(nap) + flows_of(self.steady.streams["recycle"])
        dC, outs = [], []
        for k, r in enumerate(self.reactors):
            inlet = stream_of(F, s[f"T_in{k + 1}"], r.P)
            F, T, cokes = r.rest(inlet, C_all[k])
            dC.append(cokes / (r.W / r.n_cells))
            outs.append((F, T))
        return jnp.stack(dC), outs

    def cycle(self, t_end: float, args: dict | None = None, C0: Array | None = None, n_save: int = 31,
              method: str = "diffrax:tsit5", rtol: float = 1e-4, atol: float = 1e-6, **kwargs):
        """A run of coking on the slow time scale (``t_end`` in seconds; days to months).

        Two time scales: the temperatures relax in minutes, the coke moves over
        weeks. On the coke's time scale the temperatures sit on their rest
        manifold (each cell at its segment's adiabatic outlet, :meth:`DynamicReformingReactor.rest`)
        and the heaters at their setpoints, so only the coke is integrated --
        the singular-perturbation reduction of the full model, whose stiffness
        (a minute against weeks) an explicit solver could not take. Needs
        ``coke_alpha`` for the coke to matter.

        Returns ``(ts, C (n_save, n_reactors, n_cells) kg/kg, solver info)``.
        """
        from difflow.dynamic.integrators import integrate
        args = self.args() if args is None else args
        nr, nc = len(self.reactors), self.n_cells
        C0 = jnp.zeros((nr, nc)) if C0 is None else jnp.asarray(C0, dtype=float)

        def f(t, y, a):
            dC, _ = self._rest_train(t, y.reshape(nr, nc) * COKE_SCALE, a)
            return dC.reshape(-1) / COKE_SCALE

        res = integrate(f, C0.reshape(-1) / COKE_SCALE, (0.0, float(t_end)), method, args=args, rtol=rtol,
                        atol=atol, saveat=jnp.linspace(0.0, t_end, n_save), dt0=kwargs.pop("dt0", 3600.0),
                        **kwargs)
        return res.trajectory.t, res.trajectory.y.reshape(-1, nr, nc) * COKE_SCALE, res.info

    def cycle_observe(self, t, C_all, args: dict | None = None) -> dict:
        """Reactor outlet temperatures and the reformate on the rest manifold at coke ``C_all``."""
        from difflow_refinery.reforming.products import reformate_properties
        from difflow_refinery.reforming.separation import (
            ProductSeparator, ProductSeparatorParams, Stabilizer, StabilizerParams)
        args = self.args() if args is None else args
        p = self.reformer.params
        _, outs = self._rest_train(jnp.asarray(t), C_all, args)
        F, T = outs[-1]
        eff = stream_of(F, T, self.reactors[-1].P)
        liq, _, _ = ProductSeparator(ProductSeparatorParams(T=p.T_separator, P=p.P_separator))(eff)
        ref, _, _, _ = Stabilizer(StabilizerParams(c4_recovery=p.c4_recovery))(liq)
        return {"T_out": jnp.stack([o[1] for o in outs]), "reformate.RON": reformate_properties(flows_of(ref))["RON"],
                "activity_mean": jnp.stack([jnp.mean(r.activity(C_all[k])) for k, r in enumerate(self.reactors)])}

    def _split(self, y):
        fs = self.flowsheet
        fs._update_state_indices()
        return {e.name: y[e.state_slice] for e in fs.units}

    def observe(self, t, y, args: dict | None = None) -> dict:
        """Reactor temperatures and coke, and the reformate (steady separator + stabilizer on the effluent)."""
        from difflow_refinery.reforming.products import reformate_properties
        from difflow_refinery.reforming.separation import (
            ProductSeparator, ProductSeparatorParams, Stabilizer, StabilizerParams)
        args = self.args() if args is None else args
        streams = self.flowsheet.outputs(jnp.asarray(t), y, None, args)
        us = self._split(y)
        p = self.reformer.params
        out = {}
        reps = [r.report(us[r.name], streams[f"{r.name}_in"]) for r in self.reactors]
        out["T_in"] = jnp.stack([r["T_in"] for r in reps])
        out["T_out"] = jnp.stack([r["T_out"] for r in reps])
        out["coke_make"] = jnp.stack([r["coke_make"] for r in reps])
        out["coke_mean"] = jnp.stack([jnp.mean(r["coke"]) for r in reps])
        out["activity_mean"] = jnp.stack([jnp.mean(r["activity"]) for r in reps])
        w = jnp.asarray(p.catalyst_split, dtype=float)
        out["wait"] = jnp.sum(w * out["T_in"]) / jnp.sum(w)
        eff = streams[f"rx{len(self.reactors)}_out"]
        if self.loop is None:
            liq, vap, _ = ProductSeparator(ProductSeparatorParams(T=p.T_separator, P=p.P_separator))(eff)
        else:
            liq = streams["sep_liquid"]
            nL, nG = self.loop.holdups(us["loop"])
            out["loop.inventory"] = jnp.sum(nG)
            out["recycle.rate"] = jnp.sum(flows_of(streams["recycle"]))
            out["recycle.h2_purity"] = flows_of(streams["recycle"])[sp.INDEX["H2"]] / out["recycle.rate"]
            out["net_gas.h2"] = flows_of(streams["net_gas"])[sp.INDEX["H2"]]
        ref, _, _, _ = Stabilizer(StabilizerParams(c4_recovery=p.c4_recovery))(liq)
        props = reformate_properties(flows_of(ref))
        out["reformate.RON"] = props["RON"]
        MW = jnp.asarray(sp.MW) / 1000.0
        out["reformate.yield"] = flows_of(ref) @ MW / self.feed.mass_flow
        h2 = sp.INDEX["H2"]
        out["h2.make"] = flows_of(eff)[h2] - flows_of(streams["recycle"])[h2]
        return out

    def trajectory(self, result, args: dict | None = None, keys: Sequence[str] | None = None) -> dict:
        """:meth:`observe` at every saved time, with a leading time axis."""
        ts, ys = result.trajectory.t, result.trajectory.y
        f = jax.jit(lambda t, y, a: self.observe(t, y, a))
        args = self.args() if args is None else args
        obs = [f(t, y, args) for t, y in zip(ts, ys)]
        keys = keys or list(obs[0])
        out = {k: jnp.stack([o[k] for o in obs]) for k in keys}
        out["t"] = ts
        return out

    def steady_reference(self) -> dict:
        """The steady reformer's values of the quantities :meth:`observe` reports."""
        r = self.steady
        o = r.outputs()
        out = {"T_in": jnp.stack([x["T_in"] for x in r.reactors]),
               "T_out": jnp.stack([x["T_out"] for x in r.reactors]),
               "coke_make": jnp.stack([x["coke"] for x in r.reactors]),
               "reformate.RON": o["reformate.RON"]}
        if self.loop is not None:
            rec = flows_of(r.streams["recycle"])
            out["recycle.rate"] = jnp.sum(rec)
            out["recycle.h2_purity"] = rec[sp.INDEX["H2"]] / jnp.sum(rec)
            out["net_gas.h2"] = flows_of(r.streams["net_gas"])[sp.INDEX["H2"]]
        return out

    def element_balance(self, t, y, args: dict | None = None) -> dict:
        """``{"accumulation", "in_minus_out"}``: (2,) mol/s of C and H over the train and loop.

        Needs ``recycle_loop`` (the held recycle is not a balance). The cells
        hold no gas (quasi-steady); the coke is bookkeeping, not removed from
        the gas (as in the steady reformer), so it is neither.
        """
        if self.loop is None:
            raise ValueError("build with recycle_loop=True for an element balance")
        args = self.args() if args is None else args
        fs = self.flowsheet
        streams = fs.outputs(jnp.asarray(t), y, None, args)
        dy = fs.derivatives(jnp.asarray(t), y, None, args)
        E = jnp.asarray(sp.ELEMENTS)

        def held(v):
            nL, nG = self.loop.holdups(self._split(v)["loop"])
            return E @ (nL + nG)

        acc = jax.jvp(held, (y,), (dy,))[1]
        inflow = flows_of(streams["naphtha"])
        outflow = flows_of(streams["net_gas"]) + flows_of(streams["sep_liquid"])
        return {"accumulation": acc, "in_minus_out": E @ (inflow - outflow)}


__all__ = ["COKE_SCALE", "DynamicReformingReactor", "DynamicReformerLoop", "DynamicReformerTrain"]
