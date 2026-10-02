"""The catalytic reformer: a semi-regenerative reactor train with H2 recycle.

Flowsheet (a :class:`difflow.Flowsheet`, the recycle gas as its tear)::

    naphtha --+--> charge heater --> R1 --> heater 2 --> R2 --> heater 3 --> R3
              ^                                                              |
              |                                                    effluent cooler
              |                                                              v
       recycle compressor <-- recycle gas <-- splitter <-- vapour <-- separator
                                                 |                          |
                                              net gas                    liquid
                                                                            v
                                                    reformate, LPG, fuel gas <-- stabilizer

Any number of reactors (three by default, the classic semi-regen train; four
is common) is set by the length of ``inlet_T`` and ``catalyst_split``.

Degrees of freedom (all in :class:`ReformerParams`, all differentiable):

* reactor inlet temperatures ``inlet_T`` -- or their catalyst-weighted
  average, the WAIT, through :meth:`ReformerParams.with_wait`;
* separator pressure ``P_separator`` (the reactors run ``loop_dP`` above it);
* the recycle ``H2_HC`` ratio, mol H2 per mol naphtha hydrocarbon;
* the liquid hourly space velocity ``LHSV`` (with the feed rate it sets the
  catalyst inventory);
* the stabilizer's ``c4_recovery`` (the RVP / C4-in-reformate spec).

The reformate RON is an OUTPUT; :meth:`CatalyticReformer.wait_for_ron` turns
it into a target by solving for the WAIT (a scalar secant on the
flowsheet), the way a unit is operated to an octane.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flowsheet import Flowsheet, Unit
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow.units.eos_units import Compressor, CompressorParams
from difflow_refinery.reforming import species as sp
from difflow_refinery.reforming import thermo as th
from difflow_refinery.reforming.feed import NaphthaFeed
from difflow_refinery.reforming.kinetics import ReformingKinetics
from difflow_refinery.reforming.products import reformate_properties
from difflow_refinery.reforming.reactor import (
    FiredHeater, FiredHeaterParams, ReformingReactor, ReformingReactorParams, flows_of, stream_of)
from difflow_refinery.reforming.separation import (
    ProductSeparator, ProductSeparatorParams, RecycleSplitter, RecycleSplitterParams, Stabilizer,
    StabilizerParams)

_H_MASS = 1.00794
_C_MASS = 12.0107


@dataclass
class ReformerParams(ParamsMixin):
    """Design and operating specifications of a :class:`CatalyticReformer`.

    Attributes:
        inlet_T: Inlet temperature of each reactor (K), first reactor first.
        catalyst_split: Share of the catalyst in each reactor (sums to one).
        LHSV: Liquid hourly space velocity (1/h): feed standard liquid volume
            per hour over catalyst volume.
        catalyst_density: Catalyst bulk density (kg/m^3) converting the
            catalyst volume to the mass the rate laws are written on.
            ILLUSTRATIVE (typical of extruded Pt-Re/alumina).
        P_separator: Product separator pressure (Pa).
        loop_dP: Pressure of the reactors above the separator (Pa): the loop
            pressure drop the recycle compressor makes up. The reactors are
            isobaric at ``P_separator + loop_dP``.
        T_separator: Separator temperature (K).
        H2_HC: Recycle hydrogen per mol of naphtha hydrocarbon.
        kinetics: Rate parameters (:class:`ReformingKinetics`).
        heater_efficiency: Absorbed over fired duty of every heater.
        compressor_efficiency: Isentropic efficiency of the recycle compressor.
        c4_recovery: Share of the butanes the stabilizer takes to LPG.
        coke_capacity: Coke on catalyst at end of run (kg/kg) the cycle length
            is reckoned to. ILLUSTRATIVE.
        rtol: Relative tolerance of the reactor integrations.
    """

    inlet_T: tuple = (773.15, 773.15, 773.15)
    catalyst_split: tuple = (0.15, 0.30, 0.55)
    LHSV: float = 1.5
    catalyst_density: float = 700.0
    P_separator: float = 12.0e5
    loop_dP: float = 3.0e5
    T_separator: float = 311.15
    H2_HC: float = 5.0
    kinetics: ReformingKinetics = field(default_factory=ReformingKinetics)
    heater_efficiency: float = 0.85
    compressor_efficiency: float = 0.75
    c4_recovery: float = 0.95
    coke_capacity: float = 0.15
    rtol: float = 1e-8

    def __post_init__(self):
        if len(self.inlet_T) != len(self.catalyst_split):
            raise ValueError("inlet_T and catalyst_split need one entry per reactor")
        if len(self.inlet_T) < 1:
            raise ValueError("at least one reactor")

    @property
    def n_reactors(self) -> int:
        return len(self.inlet_T)

    @property
    def wait(self) -> Array:
        """Weighted average inlet temperature (K), weighted by catalyst."""
        w = jnp.asarray(self.catalyst_split, dtype=float)
        return jnp.sum(w * jnp.stack([jnp.asarray(t, dtype=float) for t in self.inlet_T])) / jnp.sum(w)

    def with_wait(self, wait: Array) -> "ReformerParams":
        """A copy with every inlet temperature shifted so the WAIT is ``wait`` (K)."""
        d = jnp.asarray(wait, dtype=float) - self.wait
        return replace(self, inlet_T=tuple(jnp.asarray(t, dtype=float) + d for t in self.inlet_T))

    @property
    def P_reactor(self) -> Array:
        return jnp.asarray(self.P_separator) + jnp.asarray(self.loop_dP)


@dataclass
class ReformerResult:
    """A solved reformer.

    Attributes:
        streams: Every flowsheet stream by name.
        feed: The :class:`NaphthaFeed`.
        params: The :class:`ReformerParams` it was solved at.
        reactors: Per reactor ``{"T_in", "T_out", "dT", "coke", "W", "H_in", "H_out"}``.
        heaters: Per heater ``{"duty", "fired", ...}`` (W).
        separator, splitter, compressor, stabilizer: Their info dicts.
        converged: Whether the recycle met its tolerance (``None`` when traced).
        iterations: Recycle iterations used.
    """

    streams: dict
    feed: NaphthaFeed
    params: ReformerParams
    reactors: list
    heaters: list
    separator: dict
    splitter: dict
    compressor: dict
    stabilizer: dict
    converged: bool | None = None
    iterations: int | None = None

    # ----- products ---------------------------------------------------------

    def flows(self, name: str) -> Array:
        return flows_of(self.streams[name])

    @property
    def tear(self) -> Stream:
        """The converged recycle gas: pass as ``tear_initial`` to warm-start a solve."""
        return self.streams["recycle"]

    @property
    def reformate(self) -> dict[str, Array]:
        """Reformate properties (:func:`~difflow_refinery.reforming.products.reformate_properties`)."""
        return reformate_properties(self.flows("reformate"))

    def outputs(self) -> dict[str, Array]:
        """The numbers a planner reads, as one flat dict (see :data:`OUTPUT_UNITS`)."""
        feed_mass = self.feed.mass_flow
        feed_vol = self.feed.volume_flow
        F_ref = self.flows("reformate")
        F_net = self.flows("net_gas")
        F_lpg = self.flows("lpg")
        F_fg = self.flows("fuel_gas")
        MW = jnp.asarray(sp.MW) / 1000.0
        props = reformate_properties(F_ref)
        iH = sp.INDEX["H2"]
        # C5+ in the reformate (the stabilizer leaves some butane in it).
        c5 = jnp.asarray([sp.SPECIES[k].carbon >= 5 for k in sp.NAMES], dtype=float)
        out = {
            "reformate.yield_vol": 100.0 * th.std_liquid_volume(F_ref) / feed_vol,
            "reformate.yield_wt": 100.0 * (F_ref @ MW) / feed_mass,
            "c5plus.yield_vol": 100.0 * th.std_liquid_volume(F_ref * c5) / feed_vol,
            "reformate.RON": props["RON"], "reformate.MON": props["MON"],
            "reformate.RON_linear": props["RON_linear"],
            "reformate.aromatics_vol": props["aromatics_vol"],
            "reformate.benzene_vol": props["benzene_vol"],
            "reformate.RVP_psi": props["RVP_psi"],
            "reformate.SG": props["SG"],
            "reformate.kg_s": F_ref @ MW,
            "h2.net_mol_s": F_net[iH],
            "h2.net_wt": 100.0 * F_net[iH] * MW[iH] / feed_mass,
            "h2.purity": 100.0 * F_net[iH] / jnp.sum(F_net),
            "net_gas.kg_s": F_net @ MW,
            "lpg.kg_s": F_lpg @ MW,
            "fuel_gas.kg_s": F_fg @ MW,
            "compressor.power_MW": self.compressor["W"] / 1e6,
            "separator.duty_MW": self.separator["duty"] / 1e6,
            "heaters.fired_MW": sum(h["fired"] for h in self.heaters) / 1e6,
            "coke.kg_h": sum(r["coke"] for r in self.reactors) * 3600.0,
            "cycle.days": self.cycle_length_days,
            "wait.C": self.params.wait - 273.15,
            "wabt.C": self.wabt - 273.15,
        }
        for k, r in enumerate(self.reactors, start=1):
            out[f"rx{k}.dT"] = r["dT"]
            out[f"rx{k}.T_out_C"] = r["T_out"] - 273.15
        for k, h in enumerate(self.heaters, start=1):
            out[f"heater{k}.duty_MW"] = h["duty"] / 1e6
            out[f"heater{k}.fired_MW"] = h["fired"] / 1e6
        return out

    @property
    def wabt(self) -> Array:
        """Weighted average bed temperature (K): mean of inlet and outlet, by catalyst."""
        w = jnp.asarray(self.params.catalyst_split, dtype=float)
        T = jnp.stack([0.5 * (r["T_in"] + r["T_out"]) for r in self.reactors])
        return jnp.sum(w * T) / jnp.sum(w)

    @property
    def catalyst_mass(self) -> Array:
        return sum(r["W"] for r in self.reactors)

    @property
    def cycle_length_days(self) -> Array:
        """Days on stream until the catalyst holds ``coke_capacity`` of coke."""
        coke = sum(r["coke"] for r in self.reactors)
        return self.params.coke_capacity * self.catalyst_mass / coke / 86400.0

    # ----- balances ----------------------------------------------------------

    def balances(self) -> dict[str, Array]:
        """Relative closures of the overall mass, carbon, hydrogen and energy balances.

        Products are reformate, LPG, fuel gas and net gas. ``energy`` closes
        the overall balance (heater duties + compressor work + cooler and
        stabilizer duties against the product-minus-feed enthalpy, every
        stream on the one basis) relative to the total fired-heater duty;
        ``adiabatic_<k>`` is reactor ``k``'s enthalpy change relative to its
        heater's duty. Coke is not withdrawn from the balance (it is
        reported, ~1e-6 of the feed).
        """
        Fin = self.feed.flows
        outs = [self.flows(n) for n in ("reformate", "lpg", "fuel_gas", "net_gas")]
        Fout = sum(outs)
        MW = jnp.asarray(sp.MW)
        C, H = jnp.asarray(sp.ELEMENTS[0]), jnp.asarray(sp.ELEMENTS[1])
        res = {
            "mass": (Fout - Fin) @ MW / (Fin @ MW),
            "carbon": (Fout - Fin) @ C / (Fin @ C),
            "hydrogen": (Fout - Fin) @ H / (Fin @ H),
        }
        H_feed = th.total_enthalpy_flash(Fin, self.feed.T, self.feed.P)
        H_prod = sum(th.total_enthalpy_flash(self.flows(n), self.streams[n]["T"], self.streams[n]["P"])
                     for n in ("reformate", "lpg", "fuel_gas", "net_gas"))
        Q = (sum(h["duty"] for h in self.heaters) + self.compressor["W"]
             + self.separator["duty"] + self.stabilizer["duty"]
             + sum(r["H_out"] - r["H_in"] for r in self.reactors))
        Q_ref = sum(h["duty"] for h in self.heaters)
        res["energy"] = (H_prod - H_feed - Q) / Q_ref
        for k, (r, h) in enumerate(zip(self.reactors, self.heaters), start=1):
            res[f"adiabatic_{k}"] = (r["H_out"] - r["H_in"]) / h["duty"]
        return res

    def summary(self) -> str:
        o = {k: float(v) for k, v in self.outputs().items()}
        lines = [
            f"WAIT {o['wait.C']:.1f} C, WABT {o['wabt.C']:.1f} C, "
            f"P_sep {float(self.params.P_separator) / 1e5:.1f} bar, H2/HC {float(self.params.H2_HC):.2f}",
            "reactor dT (K): " + ", ".join(f"{o[f'rx{k}.dT']:.1f}" for k in range(1, len(self.reactors) + 1)),
            f"reformate: {o['reformate.yield_vol']:.1f} vol% ({o['c5plus.yield_vol']:.1f} C5+), "
            f"RON {o['reformate.RON']:.1f} (linear {o['reformate.RON_linear']:.1f}), "
            f"MON {o['reformate.MON']:.1f}, aromatics {o['reformate.aromatics_vol']:.1f} vol%, "
            f"benzene {o['reformate.benzene_vol']:.2f} vol%, RVP {o['reformate.RVP_psi']:.1f} psi",
            f"net H2 {o['h2.net_wt']:.2f} wt% of feed at {o['h2.purity']:.1f} mol%; "
            f"LPG {o['lpg.kg_s']:.3f} kg/s, fuel gas {o['fuel_gas.kg_s']:.3f} kg/s",
            f"fired {o['heaters.fired_MW']:.2f} MW, compressor {o['compressor.power_MW']:.3f} MW, "
            f"coke {o['coke.kg_h']:.2f} kg/h, cycle {o['cycle.days']:.0f} d",
        ]
        return "\n".join(lines)


#: Units of :meth:`ReformerResult.outputs` (per-reactor and per-heater keys follow the pattern).
OUTPUT_UNITS: dict[str, str] = {
    "reformate.yield_vol": "vol% of feed", "reformate.yield_wt": "wt% of feed",
    "c5plus.yield_vol": "vol% of feed", "reformate.RON": "-", "reformate.MON": "-",
    "reformate.RON_linear": "-", "reformate.aromatics_vol": "vol%", "reformate.benzene_vol": "vol%",
    "reformate.RVP_psi": "psi", "reformate.SG": "-", "reformate.kg_s": "kg/s",
    "h2.net_mol_s": "mol/s", "h2.net_wt": "wt% of feed", "h2.purity": "mol%",
    "net_gas.kg_s": "kg/s", "lpg.kg_s": "kg/s", "fuel_gas.kg_s": "kg/s",
    "compressor.power_MW": "MW", "separator.duty_MW": "MW", "heaters.fired_MW": "MW",
    "coke.kg_h": "kg/h", "cycle.days": "d", "wait.C": "C", "wabt.C": "C",
    "rx.dT": "K", "rx.T_out_C": "C", "heater.duty_MW": "MW", "heater.fired_MW": "MW",
}


def output_units(name: str) -> str:
    if name in OUTPUT_UNITS:
        return OUTPUT_UNITS[name]
    head, tail = name.split(".", 1)
    return OUTPUT_UNITS[head.rstrip("0123456789") + "." + tail]


class CatalyticReformer:
    """Semi-regenerative catalytic reformer (see the module docstring).

    Example::

        from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha
        res = CatalyticReformer(ReformerParams()).solve(lean_naphtha())
        print(res.summary())
    """

    def __init__(self, params: ReformerParams | None = None, adjoint: str = "forward"):
        self.params = params or ReformerParams()
        self.adjoint = adjoint

    # ----- the flowsheet ----------------------------------------------------

    def catalyst_masses(self, feed: NaphthaFeed, params: ReformerParams | None = None) -> list[Array]:
        p = params or self.params
        V_cat = feed.volume_flow * 3600.0 / jnp.asarray(p.LHSV)          # m^3
        W = jnp.asarray(p.catalyst_density) * V_cat
        split = jnp.asarray(p.catalyst_split, dtype=float)
        split = split / jnp.sum(split)
        return [W * split[k] for k in range(p.n_reactors)]

    def build(self, feed: NaphthaFeed, params: ReformerParams | None = None):
        """The :class:`difflow.Flowsheet` and its unit objects (not solved)."""
        p = params or self.params
        n = p.n_reactors
        P_rx = p.P_reactor
        Ws = self.catalyst_masses(feed, p)
        fs = Flowsheet(species_order=list(sp.NAMES))
        fs.add_feed("naphtha", feed.stream())
        ops: dict[str, Any] = {}
        for k in range(n):
            h = FiredHeater(FiredHeaterParams(T_out=p.inlet_T[k], efficiency=p.heater_efficiency,
                                              P_out=P_rx))
            inlets = ["naphtha", "recycle"] if k == 0 else [f"rx{k}_out"]
            ops[f"heater{k + 1}"] = h
            fs.add_unit(Unit(f"heater{k + 1}", h, inlets, [f"rx{k + 1}_in"]))
            r = ReformingReactor(ReformingReactorParams(catalyst_mass=Ws[k], kinetics=p.kinetics,
                                                        P=P_rx, rtol=p.rtol), adjoint=self.adjoint)
            ops[f"rx{k + 1}"] = r
            fs.add_unit(Unit(f"rx{k + 1}", r, [f"rx{k + 1}_in"], [f"rx{k + 1}_out"]))
        sep = ProductSeparator(ProductSeparatorParams(T=p.T_separator, P=p.P_separator))
        spl = RecycleSplitter(RecycleSplitterParams(H2_HC=p.H2_HC, hydrocarbon_feed=feed.hydrocarbon_moles))
        comp = Compressor(CompressorParams(P_out=P_rx, eta_isentropic=p.compressor_efficiency,
                                           T_bounds=(200.0, 800.0)), th.cubic_thermo())
        stab = Stabilizer(StabilizerParams(c4_recovery=p.c4_recovery))
        ops.update(separator=sep, splitter=spl, compressor=comp, stabilizer=stab)
        fs.add_unit(Unit("separator", sep, [f"rx{n}_out"], ["sep_liquid", "sep_vapor"]))
        fs.add_unit(Unit("splitter", spl, ["sep_vapor"], ["recycle_suction", "net_gas"]))
        fs.add_unit(Unit("compressor", comp, ["recycle_suction"], ["recycle_out"]))
        fs.add_unit(Unit("stabilizer", stab, ["sep_liquid"], ["reformate", "lpg", "fuel_gas"]))
        fs.add_recycle("recycle_out", "recycle")
        return fs, ops, Ws

    def initial_recycle(self, feed: NaphthaFeed, params: ReformerParams | None = None) -> Stream:
        """A starting recycle gas: the H2/HC's hydrogen at 90 mol% purity, the rest methane."""
        p = params or self.params
        H2 = jnp.asarray(p.H2_HC) * feed.hydrocarbon_moles
        F = jnp.zeros(sp.N_SPECIES).at[sp.INDEX["H2"]].set(H2).at[sp.INDEX["C1"]].set(H2 / 9.0)
        return stream_of(F, jnp.asarray(330.0), p.P_reactor)

    def solve(self, feed: NaphthaFeed, params: ReformerParams | None = None, *,
              tear_initial: Stream | None = None, tol: float = 1e-9, max_iter: int = 200,
              on_nonconvergence: str = "warn") -> ReformerResult:
        """Solve the reformer for a feed.

        Concrete inputs use the flowsheet's Anderson-accelerated recycle
        solve. Under ``jax.jacfwd``/``jvp``/``jit`` the flowsheet switches to
        optimistix's fixed point with implicit differentiation (difflow's
        traced path), so gradients come from the converged recycle;
        warm-start it with ``tear_initial=previous_result.tear``. Use
        forward-mode transforms (``jax.jacfwd``): the reactor integrations
        use ``diffrax.ForwardMode`` (see :mod:`.reactor`).
        """
        p = params or self.params
        fs, ops, Ws = self.build(feed, p)
        guess = tear_initial if tear_initial is not None else self.initial_recycle(feed, p)
        guess = {k: guess.get(k, jnp.asarray(0.0)) for k in
                 [f"F_{s}" for s in sp.NAMES] + ["T", "P"]}
        streams = fs.solve(tear_initial={"recycle": guess}, tol=tol, max_iter=max_iter,
                           acceleration="anderson", on_nonconvergence=on_nonconvergence,
                           clip_negative_flows=True, error_probe=0)
        # One more pass over the converged streams for the units' info dicts
        # (the flowsheet keeps the streams only).
        info = {}
        for u in fs.units:
            out = u.operation(*[streams[n] for n in u.inlet_names])
            info[u.name] = out[-1]
        reactors = []
        for k in range(p.n_reactors):
            r = dict(info[f"rx{k + 1}"])
            r["W"] = Ws[k]
            reactors.append(r)
        heaters = [info[f"heater{k + 1}"] for k in range(p.n_reactors)]
        return ReformerResult(streams=streams, feed=feed, params=p, reactors=reactors,
                              heaters=heaters, separator=info["separator"],
                              splitter=info["splitter"], compressor=info["compressor"],
                              stabilizer=info["stabilizer"], converged=fs.last_solve_converged,
                              iterations=fs.last_solve_iterations)

    # ----- RON as a target ----------------------------------------------------

    def wait_for_ron(self, feed: NaphthaFeed, ron: float, params: ReformerParams | None = None,
                     wait0: float | None = None, tol: float = 1e-4, max_iter: int = 20):
        """The WAIT (K) at which the reformate RON is ``ron``, by secant iteration.

        Concrete only (a Python loop of full solves). Returns
        ``(wait, result)``. The gradient of that WAIT with respect to any
        other input follows from the implicit function theorem:
        ``dWAIT/dx = -(dRON/dx) / (dRON/dWAIT)``, both from ``jax.jacfwd``
        of the solve at the returned point.
        """
        p = params or self.params
        w0 = float(p.wait if wait0 is None else wait0)
        res0 = self.solve(feed, p.with_wait(w0))
        f0 = float(res0.outputs()["reformate.RON"]) - ron
        w1 = w0 + (2.0 if f0 < 0 else -2.0)
        tear = res0.tear
        for _ in range(max_iter):
            res1 = self.solve(feed, p.with_wait(w1), tear_initial=tear)
            f1 = float(res1.outputs()["reformate.RON"]) - ron
            if abs(f1) < tol:
                return w1, res1
            w0, w1, f0 = w1, w1 - f1 * (w1 - w0) / (f1 - f0), f1
            tear = res1.tear
        warnings.warn(f"wait_for_ron did not reach RON {ron} within {max_iter} iterations")
        return w1, res1
