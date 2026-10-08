"""The C5/C6 isomerization unit: reactor, stabilizer, DIP and DIH.

Four configurations (:data:`CONFIGURATIONS`), all on the same pieces:

* ``once_through`` -- fresh feed, hydrogen and the reactor, then a
  product separator (a cold flash drum, one equilibrium stage, that
  takes most of the hydrogen) and a stabilizer that takes the rest of
  the light gas overhead and sets the isomerate's RVP;
* ``dip`` -- a deisopentanizer ahead of the reactor takes the feed's
  isopentane (and the butanes) round it, so the reactor sees less of
  the product it is trying to make;
* ``dih`` -- a deisohexanizer after the stabilizer: its overhead (the
  dimethylbutanes and the C5s) and its bottoms (the naphthenes and C7+)
  are isomerate, and its side draw -- the methylpentanes and n-hexane,
  the low-octane C6s -- goes back to the reactor inlet. The recycle is a
  :class:`difflow.Flowsheet` recycle torn on the side draw;
* ``dip_dih`` -- both.

The hydrogen is once-through: the reactor charge is made up to
``H2_HC`` moles of hydrogen per mole of hydrocarbon with pure hydrogen
and what is left leaves in the off-gas (separator vapor plus stabilizer
overhead), as in a unit with no recycle-gas compressor. Hydrogen the
feed already carries counts towards that target; where it exceeds it,
the make-up is zero and the excess rides through the reactor with the
charge (``info["H2_HC_charge"]`` is then above ``H2_HC``).

The feed must carry only the reactor's species (:data:`.thermochem.NAMES`,
as :func:`.reactor.flows_of` and the registered ``IsomerizationReactor``
require); any other ``F_`` key, water included, is refused rather than
dropped. Dry the feed, or route what the unit does not model round it. The reactor
effluent is cooled to ``separator_T`` (the effluent cooler is a
specification, its duty is not reported) and flashed at ``separator_P``.

The stabilizer is a SHORTCUT, not a tray column: hydrogen, ethane and
propane go overhead, the pentanes and heavier stay in the bottoms, and
the fraction of the butanes left in the bottoms is solved so that the
bottoms meet ``stabilizer_rvp`` (:func:`stabilize`). A rigorous MESH
stabilizer was tried on the gas-plant column -- a reboiled stripper fed
hot and fed cold, a partial-condenser column with a vent, and the gas
plant's debutanizer on the separator liquid -- and none converged
reliably over the compositions the DIH recycle produces, so the unit
does not report a stabilizer duty.

The columns are :class:`~difflow_refinery.gasplant.GasPlantColumn` --
the same rigorous Peng-Robinson MESH model as the gas plant, on the
sixteen reactor species. Octanes are the Ethyl RT-70 blend of the
species octanes (:mod:`.thermochem`, marked verify there) and RVP is
the gas plant's ASTM D323 construction on the EOS.

Gradients. Once-through and DIP are differentiable straight through
(the columns carry implicit-function gradients). With a DIH the recycle
is converged by the flowsheet's Anderson iteration, a Python loop, and
:meth:`IsomerizationUnit.outputs` differentiates the converged loop by
the implicit function theorem::

    dy/du = Y_u + Y_x (I - G_x)^-1 G_u

with ``G`` one pass of the loop (recycle in -> side draw out) and ``Y``
the outputs, both linearised by forward-mode AD at the solution.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Optional

import jax
import jax.numpy as jnp

from difflow.flowsheet import Flowsheet, Unit
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_refinery.blending import BlendComponent, ethyl_rt70
from difflow_refinery.gasplant import GasPlantColumn, GasPlantColumnParams, gas_components, splitter
from difflow_refinery.gasplant.products import reid_vapor_pressure
from difflow_refinery.isomerization import thermochem as tc
from difflow_refinery.isomerization.feed import hydrocarbon_flows, with_nc6_fraction
from difflow_refinery.isomerization.reactor import (
    NU, IsomerizationReactor, IsomerizationReactorParams, carbon_number_flows, flows_of,
    stream_of)

#: The flowsheet configurations.
CONFIGURATIONS = ("once_through", "dip", "dih", "dip_dih")

#: Species lighter than isopentane (go overhead in the DIP).
C4_MINUS = ("hydrogen", "ethane", "propane", "isobutane", "n_butane")
#: Smallest reactor-outlet H2/HC (mol/mol) the unit accepts without an
#: :class:`IsomerizationHydrogenWarning`.
MIN_H2_HC_OUT = 0.05


class IsomerizationHydrogenWarning(RuntimeWarning):
    """The reactor has used up most of its hydrogen -- on these kinetics, the
    signature of a hydrocracking runaway (a hot, benzene-rich charge does it
    first) -- and the bed is outside the window the model is for."""


#: Light gas the stabilizer sends overhead entirely.
C3_MINUS = ("hydrogen", "ethane", "propane")
#: The butanes, split by the stabilizer's RVP spec.
BUTANES = ("isobutane", "n_butane")
#: Overhead of the deisohexanizer: everything up to 2,2-dimethylbutane
#: (2,3-dimethylbutane boils 2 K below 2-methylpentane and splits between
#: the overhead and the side draw).
DIH_LIGHT = C4_MINUS + ("isopentane", "n_pentane", "2_2_dimethylbutane")
#: What the DIH side draw is meant to carry back to the reactor.
DIH_RECYCLE = ("2_3_dimethylbutane", "2_methylpentane", "3_methylpentane", "n_hexane")
#: Overhead of the deisopentanizer.
DIP_LIGHT = C4_MINUS + ("isopentane",)

#: Names of :meth:`IsomerizationUnit.outputs`, in vector order.
OUTPUT_NAMES = (
    "RON", "MON", "RVP", "benzene_vol", "aromatics_vol", "SG", "isomerate_mass",
    "isomerate_volume", "yield_mass", "yield_volume", "H2_makeup", "H2_consumption",
    "gas_make", "stabilizer_c4_recovery", "dih_duty", "dip_duty", "recycle_mass", "reactor_dT",
    "iC5_C5", "22DMB_C6P", "DMB_C6P", "benzene_conversion",
)

OUTPUT_UNITS = {
    "RON": "-", "MON": "-", "RVP": "Pa", "benzene_vol": "vol%", "aromatics_vol": "vol%",
    "SG": "-", "isomerate_mass": "kg/s", "isomerate_volume": "m3/h", "yield_mass": "-",
    "yield_volume": "-", "H2_makeup": "kg/s", "H2_consumption": "kg/s", "gas_make": "kg/s",
    "stabilizer_c4_recovery": "-", "dih_duty": "W", "dip_duty": "W", "recycle_mass": "kg/s",
    "reactor_dT": "K", "iC5_C5": "-", "22DMB_C6P": "-", "DMB_C6P": "-",
    "benzene_conversion": "-",
}


@dataclass
class IsomerizationUnitParams(ParamsMixin):
    """Parameters for :class:`IsomerizationUnit`.

    Attributes:
        configuration: One of :data:`CONFIGURATIONS`.
        reactor: The reactor (LHSV, catalyst preset, activity, pressure).
        T_in: Reactor inlet temperature (K).
        H2_HC: Hydrogen to hydrocarbon mole ratio of the reactor charge.
        stabilizer_rvp: Stabilizer bottoms RVP (Pa, ASTM D323 on the EOS);
            the spec that sets how much of the butanes the isomerate keeps.
        stabilizer_P: Stabilizer pressure (Pa), the pressure the off-gas
            leaves at.
        separator_T: Product separator temperature (K): the reactor
            effluent is cooled to it and flashed.
        separator_P: Product separator pressure (Pa); ``None`` is the
            reactor outlet pressure less 1 bar.
        dih_trays: Deisohexanizer theoretical trays.
        dih_feed_tray: DIH feed tray (from the top).
        dih_side_tray: DIH side-draw tray (from the top).
        dih_side_draw: DIH side-draw (recycle) rate (kg/s).
        dih_P: DIH top pressure (Pa).
        dih_heavy_in_overhead: Mole fraction of methylpentanes and heavier
            in the DIH overhead (spec on the distillate rate).
        dih_light_in_bottoms: Mole fraction of 2,2-dimethylbutane and
            lighter in the DIH bottoms (spec on the reboiler duty).
        dip_trays: Deisopentanizer theoretical trays.
        dip_feed_tray: DIP feed tray (from the top).
        dip_P: DIP top pressure (Pa).
        dip_nc5_in_overhead: Mole fraction of n-pentane and heavier in the
            DIP overhead.
        dip_ic5_in_bottoms: Mole fraction of isopentane and lighter in the
            DIP bottoms.
        tol: Recycle tolerance (mol/s, largest change of a tear flow).
        max_iter: Recycle iteration limit.
    """

    configuration: str = "once_through"
    reactor: IsomerizationReactorParams = field(default_factory=IsomerizationReactorParams)
    T_in: float = 413.15
    H2_HC: float = 0.3
    stabilizer_rvp: float = 90e3
    stabilizer_P: float = 10e5
    separator_T: float = 313.15
    separator_P: Optional[float] = None
    dih_trays: int = 40
    dih_feed_tray: int = 15
    dih_side_tray: int = 36
    dih_side_draw: float = 6.0
    dih_P: float = 2.0e5
    dih_heavy_in_overhead: float = 0.15
    dih_light_in_bottoms: float = 0.02
    dip_trays: int = 50
    dip_feed_tray: int = 25
    dip_P: float = 2.5e5
    dip_nc5_in_overhead: float = 0.05
    dip_ic5_in_bottoms: float = 0.05
    tol: float = 1e-9
    max_iter: int = 60

    def __post_init__(self):
        if self.configuration not in CONFIGURATIONS:
            raise ValueError(f"configuration must be one of {CONFIGURATIONS}, "
                             f"not {self.configuration!r}")


def stabilize(F, rvp, components, n_bisect: int = 40):
    """Shortcut stabilizer: ``(overhead flows, bottoms flows, info)``.

    Hydrogen, ethane and propane go overhead, everything from isopentane
    up stays in the bottoms, and the bottoms keep the fraction ``f`` of
    the butanes that puts their Reid vapor pressure at ``rvp``. ``f`` is
    found by bisection on ``[0, 1]`` (the RVP rises monotonically with
    it) and then given its gradient by one Newton step at the root, which
    is the implicit-function derivative ``df = -dg / (dg/df)``. Where the
    pentanes alone are over ``rvp`` the spec cannot be met: ``f = 0`` and
    ``info["spec_met"]`` is False; where all the butanes still leave the
    bottoms under it, ``f = 1``.
    """
    light = jnp.asarray([1.0 if n in C3_MINUS else 0.0 for n in tc.NAMES])
    c4 = jnp.asarray([1.0 if n in BUTANES else 0.0 for n in tc.NAMES])
    heavy = 1.0 - light - c4

    def g(f, F):
        return (reid_vapor_pressure(F * (heavy + f * c4), components) - rvp) / 1e4

    Fs = jax.lax.stop_gradient(F)
    g0, g1 = g(0.0, Fs), g(1.0, Fs)

    def step(_, ab):
        a, b = ab
        m = 0.5 * (a + b)
        lo = g(m, Fs) < 0.0
        return jnp.where(lo, m, a), jnp.where(lo, b, m)

    a, b = jax.lax.fori_loop(0, n_bisect, step, (jnp.asarray(0.0), jnp.asarray(1.0)))
    fb = jax.lax.stop_gradient(0.5 * (a + b))
    interior = (g0 < 0.0) & (g1 > 0.0)
    slope = jax.grad(g)(fb, Fs)
    f_newton = fb - g(fb, F) / slope
    f = jnp.where(interior, f_newton, jnp.where(g0 >= 0.0, 0.0, 1.0))
    bottoms = F * (heavy + f * c4)
    info = {"c4_recovery": f, "spec_met": interior,
            "rvp": reid_vapor_pressure(bottoms, components)}
    return F - bottoms, bottoms, info


def _sum_streams(*streams, T=None, P=None):
    F = sum(hydrocarbon_flows(s) for s in streams)
    return stream_of(F, streams[0]["T"] if T is None else T,
                     streams[0]["P"] if P is None else P)


class IsomerizationUnit:
    """C5/C6 light-naphtha isomerization unit (see module docstring).

    ``__call__(feed)`` returns ``(isomerate, offgas, info)``. ``info`` has
    the scalar ``outputs`` (:data:`OUTPUT_NAMES`), every internal
    ``streams``, the reactor diagnostics (``reactor``: approach to
    equilibrium per reaction, isomer ratios, extents), the column infos and
    the recycle ``loop`` diagnostics.

    Example:
        >>> from difflow_refinery.isomerization import (IsomerizationUnit,
        ...     IsomerizationUnitParams, constructed_feed)
        >>> unit = IsomerizationUnit(IsomerizationUnitParams())  # once through
        >>> iso, gas, info = unit(constructed_feed("paraffinic", 10.0))  # doctest: +SKIP
        >>> float(info["outputs"]["RON"]) > 80                            # doctest: +SKIP
        True
    """

    symbol = "ISOM-U"
    equations = [
        r"\text{isomerate RON, MON} = \text{RT-70}(v_i, \mathrm{RON}_i, \mathrm{MON}_i, A_i)",
        r"x^* = G(x^*, u), \quad \frac{dy}{du} = Y_u + Y_x (I - G_x)^{-1} G_u",
    ]
    assumptions = [
        "Once-through hydrogen, pure make-up; no recycle-gas compressor",
        "Feed/effluent exchange not modelled: the separator temperature is set",
        "Product separator is one adiabatic equilibrium stage at separator_T",
        "Columns on Peng-Robinson with theoretical trays (gas-plant MESH model)",
        "Species octanes blended by Ethyl RT-70; species octanes to be verified",
        "LHSV is a specification on the reactor charge (fresh plus recycle)",
    ]
    references = [
        "Meyers, R. A. (ed.) (2004) Handbook of Petroleum Refining Processes, 3rd ed., "
        "McGraw-Hill, Part 9 (isomerization)",
        "Healy, W. C.; Maassen, C. W.; Peterson, R. T. (1959) A new approach to blending "
        "octanes, API Division of Refining 24th midyear meeting",
    ]
    parameter_units = {
        "T_in": "K", "H2_HC": "mol/mol", "stabilizer_rvp": "Pa", "stabilizer_P": "Pa",
        "separator_T": "K", "separator_P": "Pa", "dih_side_draw": "kg/s", "dih_P": "Pa", "dip_P": "Pa",
        "tol": "mol/s", "dih_trays": "-", "dih_feed_tray": "-", "dih_side_tray": "-",
        "dih_heavy_in_overhead": "mol/mol", "dih_light_in_bottoms": "mol/mol",
        "dip_trays": "-", "dip_feed_tray": "-", "dip_nc5_in_overhead": "mol/mol",
        "dip_ic5_in_bottoms": "mol/mol", "max_iter": "-",
    }

    def __init__(self, params: IsomerizationUnitParams):
        self.params = p = params
        self.components = comps = gas_components(list(tc.NAMES))
        self.reactor = IsomerizationReactor(p.reactor)
        self.separator = GasPlantColumn(GasPlantColumnParams(
            components=comps, n_trays=1, feed_trays={"feed": 1}, condenser=None,
            reboiler=False, top_P=self._separator_P(), overhead_keys=("hydrogen",)))
        self.has_dih = p.configuration in ("dih", "dip_dih")
        self.has_dip = p.configuration in ("dip", "dip_dih")
        self.dih = self.dip = None
        if self.has_dih:
            self.dih = GasPlantColumn(splitter(
                comps, DIH_LIGHT, heavy_spec=("bottoms.x.light", p.dih_light_in_bottoms),
                groups=(("recycle_C6", DIH_RECYCLE),),
                light_spec=("distillate.x.heavy", p.dih_heavy_in_overhead),
                n_trays=p.dih_trays, feed_tray=p.dih_feed_tray, top_P=p.dih_P,
                side_draws={"side": p.dih_side_tray}, draw_rates={"side": p.dih_side_draw}))
        if self.has_dip:
            self.dip = GasPlantColumn(splitter(
                comps, DIP_LIGHT, heavy_spec=("bottoms.x.light", p.dip_ic5_in_bottoms),
                light_spec=("distillate.x.heavy", p.dip_nc5_in_overhead),
                n_trays=p.dip_trays, feed_tray=p.dip_feed_tray, top_P=p.dip_P))

    def _separator_P(self):
        p = self.params
        return p.reactor.P - 1e5 if p.separator_P is None else p.separator_P

    # -- the pieces, as stream -> stream functions --------------------------

    def _charge(self, hc_streams, T_in, LHSV):
        F = sum(hydrocarbon_flows(s) for s in hc_streams)
        ih = tc.idx("hydrogen")
        hc = jnp.sum(F) - F[ih]
        # Make-up tops the charge up to H2_HC and never takes hydrogen
        # away: setting the charge H2 to the target outright destroyed any
        # feed hydrogen above it and reported a negative make-up (audit,
        # 2026-10: 630.2 mol/s H2 in the feed gave make-up -592.4 mol/s and
        # 11270.5 g/s in vs 10076.2 g/s out, while balances() added the
        # negative make-up back and reported closure).
        H2_charge = jnp.maximum(F[ih], self.params.H2_HC * hc)
        F_in = F.at[ih].set(H2_charge)
        return F_in, H2_charge - F[ih]

    def _pass_jit(self, fresh, recycle, T_in, LHSV):
        """:meth:`_pass`, compiled once per unit and recycle-or-not.

        Every caller -- each Anderson iteration of :meth:`solve`, and each
        ``jacfwd``/``jvp`` of :meth:`outputs` -- goes through here. Eager,
        a pass re-dispatches the reactor and its gas-plant columns op by
        op (about 5 s each, plus a 37 s first trace); the DIH gradient test
        made seven solves that way and took 28 minutes. The streams go in
        as their arrays (flows, T, P) and the result's non-array leaves
        (phase labels, Python constants) are kept outside the trace.
        """
        key = recycle is None
        cache = self.__dict__.setdefault("_pass_cache", {})
        if key not in cache:
            static = {}

            def arrays(Ff, Tf, Pf, rec, T_in, LHSV):
                r = None if rec is None else stream_of(*rec)
                leaves, tree = jax.tree_util.tree_flatten(
                    self._pass(stream_of(Ff, Tf, Pf), r, T_in, LHSV))
                dyn = [isinstance(x, jax.Array) for x in leaves]
                static["tree"], static["dyn"] = tree, dyn
                static["leaves"] = [None if d else x for x, d in zip(leaves, dyn)]
                return [x for x, d in zip(leaves, dyn) if d]

            cache[key] = (jax.jit(arrays), static)
        fn, static = cache[key]
        rec = None if recycle is None else (hydrocarbon_flows(recycle),
                                            jnp.asarray(recycle["T"], dtype=float),
                                            jnp.asarray(recycle["P"], dtype=float))
        out = iter(fn(hydrocarbon_flows(fresh), jnp.asarray(fresh["T"], dtype=float),
                      jnp.asarray(fresh["P"], dtype=float), rec,
                      jnp.asarray(T_in, dtype=float), jnp.asarray(LHSV, dtype=float)))
        leaves = [next(out) if d else x for x, d in zip(static["leaves"], static["dyn"])]
        return jax.tree_util.tree_unflatten(static["tree"], leaves)

    def _pass(self, fresh, recycle, T_in, LHSV):
        """One pass of the flowsheet, recycle given. Pure JAX."""
        p = self.params
        st, info = {}, {}
        if self.has_dip:
            # Hydrogen in the fresh feed does not go through the DIP: it is
            # a column for the liquid hydrocarbons, and hydrogen sent
            # through it left with the overhead, into the isomerate (#381).
            # It bypasses the column to the reactor charge instead, where
            # it counts against the make-up.
            ih = tc.idx("hydrogen")
            F_fresh = hydrocarbon_flows(fresh)
            fresh_h2 = jnp.zeros_like(F_fresh).at[ih].set(F_fresh[ih])
            dip_feed = stream_of(F_fresh.at[ih].set(0.0), fresh["T"], p.dip_P + 0.5e5)
            dip_ovhd, dip_btms, info["dip"] = self.dip(dip_feed)
            st.update(dip_overhead=dip_ovhd, dip_bottoms=dip_btms)
            to_reactor = [dip_btms, stream_of(fresh_h2, fresh["T"], fresh["P"])]
        else:
            to_reactor = [fresh]
        if self.has_dih:
            to_reactor.append(recycle)
        F_in, makeup = self._charge(to_reactor, T_in, LHSV)
        st["charge"] = stream_of(F_in, T_in, self.reactor.params.P)
        res = self.reactor.run(F_in, T_in, LHSV=LHSV)
        info["reactor"] = res["info"]
        info["reactor_extents"] = res["extents"]
        info["H2_makeup"] = makeup
        info["H2_HC_charge"] = F_in[tc.idx("hydrogen")] / (
            jnp.sum(F_in) - F_in[tc.idx("hydrogen")])
        st["effluent"] = stream_of(res["F"], res["T"], res["P"])
        sep_vap, sep_liq, info["separator"] = self.separator(
            stream_of(res["F"], p.separator_T, self._separator_P()))
        F_ovhd, F_btms, info["stabilizer"] = stabilize(
            hydrocarbon_flows(sep_liq), p.stabilizer_rvp, self.components)
        stab_ovhd = stream_of(F_ovhd, p.separator_T, p.stabilizer_P)
        stab_btms = stream_of(F_btms, p.separator_T, p.stabilizer_P)
        st.update(separator_vapor=sep_vap, separator_liquid=sep_liq,
                  stabilizer_overhead=stab_ovhd, stabilizer_bottoms=stab_btms)
        st["offgas"] = _sum_streams(sep_vap, stab_ovhd, T=p.separator_T, P=p.stabilizer_P)
        products = [stab_btms]
        if self.has_dih:
            dih_ovhd, side, dih_btms, info["dih"] = self.dih(dict(stab_btms, P=p.dih_P + 1.5e5))
            st.update(dih_overhead=dih_ovhd, side_draw=side, dih_bottoms=dih_btms)
            products = [dih_ovhd, dih_btms]
        if self.has_dip:
            products.append(st["dip_overhead"])
        st["isomerate"] = _sum_streams(*products, T=313.15, P=1e5 + 0.5e5)
        return st, info

    # -- solving -------------------------------------------------------------

    def _flowsheet(self, fresh, T_in, LHSV):
        """The recycle as a :class:`difflow.Flowsheet` (DIH configurations)."""
        fs = Flowsheet(list(tc.NAMES))
        fs.add_feed("fresh", fresh)
        self._last_info = {}

        def loop(recycle):
            st, info = self._pass_jit(fresh, recycle, T_in, LHSV)
            self._last_info = (st, info)
            side = st["side_draw"]
            return stream_of(hydrocarbon_flows(side), side["T"], side["P"])

        fs.add_unit(Unit("isomerization_loop", loop, ["recycle"], ["side_draw"]))
        fs.add_recycle("side_draw", "recycle")
        return fs

    def initial_recycle(self, fresh) -> dict:
        """Tear initial guess: the side-draw rate at the fresh feed's C6
        composition (methylpentanes and n-hexane)."""
        F = hydrocarbon_flows(fresh)
        mask = jnp.asarray([1.0 if n in ("2_methylpentane", "3_methylpentane", "n_hexane")
                            else 0.0 for n in tc.NAMES])
        Fm = F * mask
        kg = jnp.dot(Fm, tc.MW) / 1e3
        return stream_of(Fm * self.params.dih_side_draw / kg, 340.0, self.params.dih_P)

    def solve(self, fresh: Mapping, T_in=None, LHSV=None, recycle_guess=None) -> dict:
        """Solve the unit. Returns ``{"streams", "info", "recycle", "loop"}``.

        ``recycle_guess`` is the tear's starting point: a stream, or a flow
        vector such as a previous solve's ``"recycle"``; default
        :meth:`initial_recycle`.
        """
        p = self.params
        T_in = p.T_in if T_in is None else T_in
        LHSV = p.reactor.LHSV if LHSV is None else LHSV
        # flows_of refuses a species the unit does not model; rebuilding the
        # feed through hydrocarbon_flows dropped it silently (audit, 2026-10:
        # 2 mol/s water and 1 mol/s of an unknown species vanished in all
        # configurations, and balances() never saw them).
        fresh = stream_of(flows_of(fresh), fresh["T"], fresh["P"])
        if not self.has_dih:
            st, info = self._pass_jit(fresh, None, T_in, LHSV)
            self._check_hydrogen(info)
            return {"streams": st, "info": info, "recycle": None,
                    "loop": {"converged": True, "iterations": 0, "residual": 0.0}}
        fs = self._flowsheet(fresh, T_in, LHSV)
        if recycle_guess is None:
            guess = self.initial_recycle(fresh)
        elif isinstance(recycle_guess, Mapping):
            guess = recycle_guess
        else:  # a flow vector, e.g. a previous solve's "recycle"
            guess = stream_of(jnp.asarray(recycle_guess), 340.0, p.dih_P)
        out = fs.solve(tear_initial={"recycle": guess}, tol=p.tol, max_iter=p.max_iter,
                       acceleration="anderson", on_nonconvergence="warn")
        x = hydrocarbon_flows(out["side_draw"])
        st, info = self._pass_jit(fresh, stream_of(x, out["side_draw"]["T"], p.dih_P),
                              T_in, LHSV)
        self._check_hydrogen(info)
        loop = {"converged": fs.last_solve_converged, "iterations": fs.last_solve_iterations,
                "residual": fs.last_solve_residual,
                "closure": float(jnp.max(jnp.abs(hydrocarbon_flows(st["side_draw"]) - x)))}
        return {"streams": st, "info": info, "recycle": x, "loop": loop}

    @staticmethod
    def _check_hydrogen(info) -> None:
        try:
            r = float(info["reactor"]["H2_HC_out"])
        except jax.errors.ConcretizationTypeError:   # traced: the caller reads info
            return
        if not r >= MIN_H2_HC_OUT:
            warnings.warn(
                f"hydrogen-starved reactor: H2/HC at the outlet is {r:.2g} (< {MIN_H2_HC_OUT}). "
                "Hydrocracking is running away (it is exothermic and uses hydrogen, and the "
                "hotter bed cracks faster); past this the separator flash can fail. Lower T_in, "
                "raise LHSV or raise H2_HC.",
                IsomerizationHydrogenWarning, stacklevel=3)

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, dict]:
        res = self.solve(feed)
        out = self.summarize(feed, res)
        info = dict(res["info"], outputs=out, streams=res["streams"], loop=res["loop"])
        return res["streams"]["isomerate"], res["streams"]["offgas"], info

    # -- outputs ---------------------------------------------------------------

    def octane(self, F):
        """``(RON, MON, aromatics vol%)`` of a product, Ethyl RT-70 over the
        species (hydrogen excluded)."""
        v = tc.liquid_volume(F).at[tc.idx("hydrogen")].set(0.0)
        arom = 100.0 * tc.AROMATIC
        ron, mon = ethyl_rt70(v, tc.RON, tc.MON, jnp.zeros_like(v), arom)
        return ron, mon, jnp.sum(v * arom) / jnp.sum(v)

    def summarize(self, fresh, res) -> dict:
        """The scalar outputs (:data:`OUTPUT_NAMES`) of a solved unit."""
        st, info = res["streams"], res["info"]
        Fi = hydrocarbon_flows(st["isomerate"])
        Fi = Fi.at[tc.idx("hydrogen")].set(0.0)
        Ff = hydrocarbon_flows(fresh)
        Ff = Ff.at[tc.idx("hydrogen")].set(0.0)
        ron, mon, arom = self.octane(Fi)
        v = tc.liquid_volume(Fi)
        vol = jnp.sum(v)
        mass = jnp.dot(Fi, tc.MW) / 1e3
        fresh_mass = jnp.dot(Ff, tc.MW) / 1e3
        fresh_vol = jnp.sum(tc.liquid_volume(Ff))
        Fg = hydrocarbon_flows(st["offgas"])
        h2 = tc.idx("hydrogen")
        H2w = tc.MW[h2] / 1e3
        rx = info["reactor"]
        out = {
            "RON": ron, "MON": mon,
            "RVP": reid_vapor_pressure(Fi, self.components),
            "benzene_vol": 100.0 * v[tc.idx("benzene")] / vol,
            "aromatics_vol": arom,
            "SG": mass / vol / 999.0,
            "isomerate_mass": mass,
            "isomerate_volume": 3600.0 * vol,
            "yield_mass": mass / fresh_mass,
            "yield_volume": vol / fresh_vol,
            "H2_makeup": info["H2_makeup"] * H2w,
            "H2_consumption": rx["H2_consumption"] * H2w,
            "gas_make": (jnp.dot(Fg, tc.MW) - Fg[h2] * tc.MW[h2]) / 1e3,
            "stabilizer_c4_recovery": info["stabilizer"]["c4_recovery"],
            "dih_duty": info["dih"]["outputs"]["reboiler.duty"] if self.has_dih
            else jnp.asarray(0.0),
            "dip_duty": info["dip"]["outputs"]["reboiler.duty"] if self.has_dip
            else jnp.asarray(0.0),
            "recycle_mass": (jnp.dot(hydrocarbon_flows(st["side_draw"]), tc.MW) / 1e3
                             if self.has_dih else jnp.asarray(0.0)),
            "reactor_dT": rx["delta_T"],
            "iC5_C5": rx["iC5/C5"], "22DMB_C6P": rx["22DMB/C6P"], "DMB_C6P": rx["DMB/C6P"],
            "benzene_conversion": rx["benzene_conversion"],
        }
        return out

    def output_vector(self, fresh, res):
        o = self.summarize(fresh, res)
        return jnp.stack([jnp.asarray(o[k], dtype=float) for k in OUTPUT_NAMES])

    def balances(self, fresh, res) -> dict:
        """Relative closure of the whole unit: total mass, and the moles of
        each carbon number against what the reactor extents say was made
        or cracked. In: fresh feed and make-up hydrogen; out: isomerate and
        off-gas."""
        st, info = res["streams"], res["info"]
        Ff = flows_of(fresh)    # the feed as given: refuses what solve() would
        Fh = jnp.zeros_like(Ff).at[tc.idx("hydrogen")].set(info["H2_makeup"])
        Fout = hydrocarbon_flows(st["isomerate"]) + hydrocarbon_flows(st["offgas"])
        m_in = jnp.dot(Ff + Fh, tc.MW)
        out = {"mass": (jnp.dot(Fout, tc.MW) - m_in) / m_in}
        change = NU.T @ info["reactor_extents"]
        n_in, n_out, dn = (carbon_number_flows(Ff), carbon_number_flows(Fout),
                           carbon_number_flows(change))
        tot = jnp.sum(Ff)
        for n in n_in:
            out[f"C{n}"] = (n_in[n] + dn[n] - n_out[n]) / jnp.maximum(n_in[n], 1e-3 * tot)
        return out

    # -- gradients -------------------------------------------------------------

    def outputs(self, fresh: Mapping, T_in, LHSV, x_nc6=None, recycle_guess=None):
        """Output vector (:data:`OUTPUT_NAMES`) as a differentiable function
        of the inlet temperature, the LHSV and (optionally) the fresh feed's
        n-hexane mole fraction (:func:`~.feed.with_nc6_fraction`).

        Use under ``jax.jacfwd`` / ``jax.jvp`` (forward mode), not ``jit``:
        a DIH recycle is converged by a concrete Anderson loop and
        differentiated implicitly at its solution. ``recycle_guess`` (a
        stream) warm-starts that loop, e.g. from ``last_solve["recycle"]``
        of a nearby point.
        """
        fresh = stream_of(flows_of(fresh), fresh["T"], fresh["P"])
        use_x6 = x_nc6 is not None
        x6 = jnp.asarray(x_nc6 if use_x6 else 0.0, dtype=float)
        return self._outputs_fn(fresh, use_x6, recycle_guess)(jnp.asarray(T_in, dtype=float),
                                               jnp.asarray(LHSV, dtype=float), x6)

    def _outputs_fn(self, fresh, use_x6: bool, recycle_guess=None):
        unit = self

        def feed_of(x6):
            return with_nc6_fraction(fresh, x6) if use_x6 else fresh

        def direct(T_in, LHSV, x6, recycle):
            f = feed_of(x6)
            rec = None if recycle is None else stream_of(recycle, 340.0, unit.params.dih_P)
            st, info = unit._pass_jit(f, rec, T_in, LHSV)
            y = unit.output_vector(f, {"streams": st, "info": info})
            g = hydrocarbon_flows(st["side_draw"]) if unit.has_dih else None
            return y, g

        if not self.has_dih:
            def f_open(T_in, LHSV, x6):
                return direct(T_in, LHSV, x6, None)[0]
            return f_open

        def primal(T_in, LHSV, x6):
            f = feed_of(x6)
            res = unit.solve(f, T_in=T_in, LHSV=LHSV, recycle_guess=recycle_guess)
            unit.last_solve = res
            return unit.output_vector(f, res), res["recycle"]

        @jax.custom_jvp
        def f_loop(T_in, LHSV, x6):
            return primal(T_in, LHSV, x6)[0]

        @f_loop.defjvp
        def f_loop_jvp(primals, tangents):
            T_in, LHSV, x6 = primals
            y, x = primal(T_in, LHSV, x6)
            # one linearization of a pass at the solution, reused for the
            # tear Jacobian G_x, for Y_u du and G_u du, and for the loop's
            # response (three separate jvps compiled the pass three times)
            _, lin = jax.linearize(direct, T_in, LHSV, x6, x)
            zu = jnp.zeros_like(T_in)
            Gx = jax.vmap(lambda e: lin(zu, zu, zu, e)[1])(jnp.eye(x.size)).T
            Yu, Gu = lin(*tangents, jnp.zeros_like(x))
            dx = jnp.linalg.solve(jnp.eye(x.size) - Gx, Gu)
            Yx = lin(zu, zu, zu, dx)[0]
            return y, Yu + Yx

        return f_loop

    def blend_component(self, outputs: Mapping, name: str = "isomerate") -> BlendComponent:
        """The isomerate as a :class:`~difflow_refinery.blending.BlendComponent`."""
        return BlendComponent.from_properties(
            name, SG=outputs["SG"], RON=outputs["RON"], MON=outputs["MON"],
            RVP_psi=outputs["RVP"] / 6894.757, olefins_vol=0.0,
            aromatics_vol=outputs["aromatics_vol"], benzene_vol=outputs["benzene_vol"])


__all__ = [
    "CONFIGURATIONS", "OUTPUT_NAMES", "OUTPUT_UNITS", "IsomerizationUnitParams",
    "IsomerizationHydrogenWarning", "MIN_H2_HC_OUT",
    "IsomerizationUnit",
]
