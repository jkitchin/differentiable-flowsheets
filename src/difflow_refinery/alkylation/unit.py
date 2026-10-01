"""The alkylation unit: reactor, fractionation and the isobutane recycle.

A difflow :class:`~difflow.flowsheet.Flowsheet`::

    olefin feed ---+
    makeup iC4 ----+--> makeup --> reactor --> DIB --+--> bottoms --> DeC4 --> alkylate
                   ^                               |  overhead          \\--> n-butane
                   |                               v  (tear)
                   +----------- direct -------- splitter
                   |                               | slipstream
                   +------ DeC3 bottoms <------- DeC3 --> propane

* **Makeup** (:class:`IsobutaneMakeup`) mixes the fresh olefin feed, the
  recycle (both branches) and makeup isobutane, adding the makeup that
  brings the reactor feed to the specified external I/O ratio. The ratio is
  therefore a spec (the issue's first degree of freedom); the makeup rate is
  computed.
* **Reactor** (:class:`~difflow_refinery.alkylation.reactor.AlkylationReactor`)
  with an ideal acid settler.
* **Deisobutanizer** on the reactor effluent: isobutane (with the propane)
  overhead, recycled; n-butane and alkylate to the bottoms.
* **Depropanizer** on a slipstream (``depropanizer_fraction``) of the DIB
  overhead, rejecting propane; its bottoms (isobutane) rejoin the recycle.
  The propane purge is what keeps propane from accumulating: at steady state
  the recycle carries ``F_C3 / (fraction * recovery)`` of it.
* **Debutanizer** on the DIB bottoms: n-butane overhead, alkylate bottoms.

The three columns are difflow's :class:`~difflow.units.distillation.ShortcutColumn`
-- Fenske minimum stages, Underwood minimum reflux, Gilliland stages, duties
from an energy balance -- on Peng-Robinson K-values and enthalpies
(:func:`~difflow_refinery.alkylation.reactor.alkylation_thermo`), with the
non-key distribution corrected (:class:`GeddesShortcutColumn`). Each column is
specified by its two key recoveries and its reflux ratio. This is a SHORTCUT
simplification: the rigorous gas-plant cubic-EOS stage columns the issue
points to (#312) do not exist, and the shortcut method is the existing
difflow column machinery that fits.

**Recycle**: the DIB overhead is the tear, solved by the flowsheet (Anderson
forward; ``optimistix`` fixed point with implicit differentiation under
``jax.grad``).

The debutanizer is specified by its n-butane recovery, not by the alkylate
RVP: the RVP is an output, and a planner that wants an RVP target writes a row
on the ``alkylate.RVP_psi`` output (the same argument as ``cdu_block``'s for
cut points).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.flowsheet import Flowsheet, Unit
from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow.units.distillation import ShortcutColumn, ShortcutColumnParams

from difflow_refinery.alkylation.reactor import (
    AlkylationReactor, AlkylationReactorParams, alkylation_thermo, flows_array,
    stream_from_array, to_bpd,
)
from difflow_refinery.alkylation.species import (
    ALKYLATION_SPECIES, OLEFINS, RHO_WATER_60F, property_array,
)
from difflow_refinery.blending import (
    BlendComponent, T_RVP, raoult_rvp, tbp_temperature, tbp_to_d86,
)
from difflow_refinery.correlations import vapor_pressure

_IDX = {s: i for i, s in enumerate(ALKYLATION_SPECIES)}
_V60 = property_array(ALKYLATION_SPECIES, "v60")
_MW = property_array(ALKYLATION_SPECIES, "MW")
_TB = property_array(ALKYLATION_SPECIES, "Tb")
_TC = property_array(ALKYLATION_SPECIES, "Tc")
_PC = property_array(ALKYLATION_SPECIES, "Pc")
_OMEGA = property_array(ALKYLATION_SPECIES, "omega")
_NC = property_array(ALKYLATION_SPECIES, "n_C")
_NH = property_array(ALKYLATION_SPECIES, "n_H")
_OLEFIN_IDX = np.array([_IDX[o] for o in OLEFINS])
_PSI = 6894.757293168
_C0 = 273.15

#: Smoothing width (K) of the alkylate TBP curve: each species' volume is
#: spread over a logistic of this scale about its boiling point, which
#: turns the discrete-component staircase into a differentiable curve
#: (:func:`difflow_refinery.blending.tbp_evaporated`). A modelling choice.
TBP_WIDTH = 4.0


# =============================================================================
# Makeup mixer and recycle splitter
# =============================================================================


@dataclass(repr=False)
class IsobutaneMakeupParams(ParamsMixin):
    """Parameters of the makeup mixer.

    Attributes:
        io_ratio: External isobutane/olefin ratio at the reactor inlet, by
            standard (60 F) liquid volume -- the Sauer et al. ``ratio``.
        makeup: Makeup stream composition, ``{species: mole fraction}``.
        T: Makeup temperature (K).
        P: Mixed-feed pressure (Pa).
    """

    io_ratio: float = 8.0
    makeup: dict = field(default_factory=lambda: {"isobutane": 1.0})
    T: float = 311.0
    P: float = 6.0e5


class IsobutaneMakeup:
    """Mix feed, recycles and the makeup isobutane that meets the I/O ratio.

    ``mixer(feed, *recycles) -> (reactor_feed, makeup)``. The makeup is
    ``max(0, r V_olefin - V_iC4) / (x_iC4 v_iC4)`` mol/s, the volumes those
    of the feed plus recycles at 60 F. A recycle carrying more isobutane than
    the ratio wants cannot be corrected by makeup, and the reactor feed is
    then above the ratio (the reactor reports the ratio it actually sees).
    The mixed temperature is the constant-Cp adiabatic mixing temperature
    (ideal-gas Cp at 298.15 K).
    """

    outlet_names = ("reactor_feed", "makeup")

    def __init__(self, params: IsobutaneMakeupParams | None = None):
        self.params = params if params is not None else IsobutaneMakeupParams()
        mk = self.params.makeup
        total = sum(mk.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"makeup composition sums to {total}, not 1")
        if mk.get("isobutane", 0.0) <= 0.0:
            raise ValueError("the makeup must contain isobutane")
        self._x = jnp.asarray([mk.get(s, 0.0) for s in ALKYLATION_SPECIES])
        th = alkylation_thermo().ideal
        self._cp = jnp.asarray([th.Cp(s, 298.15) for s in ALKYLATION_SPECIES])

    def __call__(self, feed: Stream, *recycles: Stream):
        p = self.params
        parts = [flows_array(feed)] + [flows_array(r) for r in recycles]
        F = sum(parts)
        V_ol = jnp.sum(F[_OLEFIN_IDX] * _V60[_OLEFIN_IDX])
        need = p.io_ratio * V_ol - F[_IDX["isobutane"]] * _V60[_IDX["isobutane"]]
        n_mk = jnp.maximum(need, 0.0) / (self._x[_IDX["isobutane"]] * _V60[_IDX["isobutane"]])
        Fm = n_mk * self._x
        Fout = F + Fm
        w = jnp.stack([jnp.sum(x * self._cp) for x in parts + [Fm]])
        Ts = jnp.stack([feed["T"]] + [r["T"] for r in recycles]
                       + [jnp.asarray(p.T, dtype=jnp.float64)])
        T_mix = jnp.sum(w * Ts) / jnp.sum(w)
        return stream_from_array(Fout, T_mix, p.P), stream_from_array(Fm, p.T, p.P)


class RecycleSplitter:
    """Split the DIB overhead: a slipstream to the depropanizer, the rest direct."""

    outlet_names = ("dec3_feed", "recycle_direct")

    def __init__(self, fraction: float):
        self.fraction = fraction

    def __call__(self, stream: Stream):
        F = flows_array(stream)
        return (stream_from_array(self.fraction * F, stream["T"], stream["P"]),
                stream_from_array((1.0 - self.fraction) * F, stream["T"], stream["P"]))


# =============================================================================
# Columns
# =============================================================================


class GeddesShortcutColumn(ShortcutColumn):
    """:class:`~difflow.units.distillation.ShortcutColumn` with the Geddes non-key split.

    The Hengstebeck-Geddes distribution is ``log(d_i/b_i) = A + C log
    alpha_i`` with ``alpha`` relative to the heavy key, so the two key
    conditions give ``A = log(d_HK/b_HK)`` and ``C = [log(d_LK/b_LK) -
    log(d_HK/b_HK)] / log alpha_LK`` (Geddes, AIChE J. 4, 389 (1958);
    Hengstebeck, Distillation, Reinhold (1961); as in Seader, Henley &
    Roper, Separation Process Principles, Fenske-Underwood-Gilliland
    chapter -- page and equation numbers unverified). The base class has
    ``A = log(d_LK/b_LK) - log(d_HK/b_HK)`` and ``C = log(d_LK/b_LK) / log
    alpha_LK``, which does not reproduce the heavy key's own split and sends
    a non-key slightly heavier than the heavy key to the distillate: on a
    propane/isobutane depropanizer it put 99.9 % of the n-butane overhead.
    Only the non-key distribution is overridden; the keys, Fenske,
    Underwood, Gilliland and the duties are the base class's.
    """

    def _split_products(self, feed_flows, F_total, z, alpha):
        p = self.params
        rec_LK, rec_HK = p.x_D_LK, p.x_B_HK
        D_LK = F_total * z[p.light_key] * rec_LK
        B_LK = F_total * z[p.light_key] * (1 - rec_LK)
        D_HK = F_total * z[p.heavy_key] * (1 - rec_HK)
        B_HK = F_total * z[p.heavy_key] * rec_HK
        log_hk = jnp.log((1 - rec_HK) / rec_HK)
        log_lk = jnp.log(rec_LK / (1 - rec_LK))
        A = log_hk
        C = (log_lk - log_hk) / jnp.log(alpha[p.light_key])
        dist, bott = {}, {}
        D_total, B_total = D_LK + D_HK, B_LK + B_HK
        for s in p.species_order:
            if s == p.light_key:
                dist[s], bott[s] = D_LK, B_LK
            elif s == p.heavy_key:
                dist[s], bott[s] = D_HK, B_HK
            else:
                # d/(d+b) = sigmoid(log(d/b)): bounded in [0, 1], so the split
                # conserves the species exactly and needs no clip.
                frac = jax.nn.sigmoid(A + C * jnp.log(alpha[s]))
                dist[s] = feed_flows[s] * frac
                bott[s] = feed_flows[s] - dist[s]
                D_total = D_total + dist[s]
                B_total = B_total + bott[s]
        x_D = {s: dist[s] / D_total for s in p.species_order}
        x_B = {s: bott[s] / B_total for s in p.species_order}
        return dist, bott, D_total, B_total, x_D, x_B


@dataclass(repr=False)
class ColumnSpec(ParamsMixin):
    """Specs of one shortcut column.

    Attributes:
        light_key: Light key species.
        heavy_key: Heavy key species. Must be present in the column feed.
        lk_recovery: Fraction of the light key to the distillate.
        hk_recovery: Fraction of the heavy key to the bottoms.
        reflux_ratio: Reflux ratio L/D. Must exceed the column's Underwood
            minimum, which ``columns_feasible`` checks.
        P: Column pressure (Pa).
    """

    light_key: str
    heavy_key: str
    lk_recovery: float
    hk_recovery: float
    reflux_ratio: float
    P: float


def _default_dec3():
    return ColumnSpec("propane", "isobutane", 0.95, 0.98, 20.0, 17.0e5)


def _default_dib():
    return ColumnSpec("isobutane", "n_butane", 0.97, 0.85, 1.0, 7.0e5)


def _default_dec4():
    return ColumnSpec("n_butane", "isopentane", 0.95, 0.90, 1.5, 5.0e5)


@dataclass(repr=False)
class AlkylationUnitParams(ParamsMixin):
    """Parameters of the alkylation unit.

    Attributes:
        reactor: The reactor's parameters.
        makeup: The makeup mixer's parameters (the I/O ratio spec).
        deisobutanizer: DIB specs (isobutane overhead, recycled).
        depropanizer: Depropanizer specs (propane overhead, on the slipstream).
        debutanizer: Debutanizer specs (n-butane overhead, alkylate bottoms).
        depropanizer_fraction: Fraction of the DIB overhead sent to the
            depropanizer.
        tol: Recycle tolerance (mol/s, max-norm of the tear step).
        max_iter: Recycle iteration cap.
    """

    reactor: AlkylationReactorParams = field(default_factory=AlkylationReactorParams)
    makeup: IsobutaneMakeupParams = field(default_factory=IsobutaneMakeupParams)
    deisobutanizer: ColumnSpec = field(default_factory=_default_dib)
    depropanizer: ColumnSpec = field(default_factory=_default_dec3)
    debutanizer: ColumnSpec = field(default_factory=_default_dec4)
    depropanizer_fraction: float = 0.3
    tol: float = 1e-9
    max_iter: int = 200


def _column(spec: ColumnSpec) -> GeddesShortcutColumn:
    return GeddesShortcutColumn(ShortcutColumnParams(
        species_order=list(ALKYLATION_SPECIES), light_key=spec.light_key,
        heavy_key=spec.heavy_key, x_D_LK=spec.lk_recovery, x_B_HK=spec.hk_recovery),
        alkylation_thermo())


#: Outputs of :meth:`AlkylationUnit.solve`, and their units.
OUTPUT_UNITS: dict[str, str] = {
    "olefin.bpd": "bbl/d", "io_ratio": "-",
    "alkylate.bpd": "bbl/d", "alkylate.yield": "-", "alkylate.yield_correlation": "-",
    "alkylate.MON": "-", "alkylate.RON": "-", "alkylate.SG": "-",
    "alkylate.RVP_psi": "psi",
    "alkylate.T10_tbp": "C", "alkylate.T50_tbp": "C", "alkylate.T90_tbp": "C",
    "alkylate.T10_d86": "C", "alkylate.T50_d86": "C", "alkylate.T90_d86": "C",
    "alkylate.heavy_fraction": "-",
    "isobutane.consumed_bpd": "bbl/d", "isobutane.makeup_bpd": "bbl/d",
    "isobutane.recycle_bpd": "bbl/d",
    "propane.bpd": "bbl/d", "n_butane.bpd": "bbl/d",
    "acid.lb_per_bbl": "lb/bbl", "acid.klb_d": "1000 lb/d",
    "reactor.heat": "MW", "refrigeration.duty": "MW", "refrigerant.isobutane": "mol/s",
    "dec3.reboiler": "MW", "dib.reboiler": "MW", "dec4.reboiler": "MW",
    "dec3.condenser": "MW", "dib.condenser": "MW", "dec4.condenser": "MW",
}


@dataclass
class AlkylationResult:
    """A converged (or not) alkylation unit.

    Attributes:
        outputs: ``{name: value}``, names and units in :data:`OUTPUT_UNITS`.
        streams: Every flowsheet stream, by name (``alkylate``,
            ``propane_product``, ``butane_product``, ``makeup``,
            ``dib_overhead`` -- the tear -- and the rest).
        converged: The recycle solve's verdict (``None`` under tracing).
        iterations: Recycle iterations used.
        columns_feasible: ``{column: bool}``, ShortcutColumn's feasibility
            flag (reflux above minimum, volatility above one, flows
            non-negative).
    """

    outputs: dict[str, Array]
    streams: dict[str, Stream]
    converged: bool | None
    iterations: int | None
    columns_feasible: dict[str, bool | None]

    def alkylate_component(self, name: str = "alkylate", **overrides) -> BlendComponent:
        """The alkylate as a :class:`~difflow_refinery.blending.BlendComponent`.

        SG, RON, MON and RVP from the unit; olefins, aromatics and benzene
        zero (alkylate is all paraffin). ``overrides`` add or replace
        properties (sulfur, say, which the model does not track).
        """
        o = self.outputs
        props = dict(SG=o["alkylate.SG"], RON=o["alkylate.RON"], MON=o["alkylate.MON"],
                     RVP_psi=o["alkylate.RVP_psi"], olefins_vol=0.0,
                     aromatics_vol=0.0, benzene_vol=0.0)
        props.update(overrides)
        return BlendComponent.from_properties(name, **props)


def alkylate_properties(F: Array) -> dict[str, Array]:
    """SG, RVP and distillation points of a product stream (flow array).

    * SG: volume-weighted, ideal mixing at 60 F.
    * RVP: :func:`~difflow_refinery.blending.raoult_rvp` (ASTM D323 bomb,
      V/L = 4, 100 F) with Lee-Kesler vapour pressures from the database
      critical constants.
    * TBP: the boiling-point curve of the species (CRC Tb) by volume,
      smoothed over :data:`TBP_WIDTH`; D86 from it by Riazi-Daubert
      (:func:`~difflow_refinery.blending.tbp_to_d86`). Riazi-Daubert was
      fitted on petroleum fractions; on a narrow-boiling alkylate whose TBP
      10 and 50 % points are a few kelvin apart it can return a D86 T10
      above T50 -- read the TBP points there.
    """
    vol = F * _V60
    V = jnp.sum(vol)
    phi = vol / V
    mass = jnp.sum(F * _MW) / 1000.0
    sg = mass / V / RHO_WATER_60F
    psat = vapor_pressure(jnp.asarray(T_RVP), _TC, _PC, _OMEGA)
    rvp = raoult_rvp(F, psat, _V60)
    out = {"SG": sg, "RVP_psi": rvp / _PSI}
    for pct in (10, 50, 90):
        tbp = tbp_temperature(jnp.asarray(pct / 100.0), jnp.asarray(_TB), phi, TBP_WIDTH)
        out[f"T{pct}_tbp"] = tbp - _C0
        out[f"T{pct}_d86"] = tbp_to_d86(tbp, pct) - _C0
    return out


class AlkylationUnit:
    """The alkylation unit as a flowsheet with the isobutane recycle.

    Example:
        >>> from difflow_refinery.alkylation import AlkylationUnit, c4_olefin_feed
        >>> unit = AlkylationUnit()                       # doctest: +SKIP
        >>> res = unit.solve(c4_olefin_feed())            # doctest: +SKIP
        >>> res.outputs["alkylate.MON"]                   # doctest: +SKIP
    """

    def __init__(self, params: AlkylationUnitParams | None = None):
        self.params = params if params is not None else AlkylationUnitParams()
        p = self.params
        self.makeup = IsobutaneMakeup(p.makeup)
        self.reactor = AlkylationReactor(p.reactor)
        self.splitter = RecycleSplitter(p.depropanizer_fraction)
        self.columns = {"dec3": _column(p.depropanizer), "dib": _column(p.deisobutanizer),
                        "dec4": _column(p.debutanizer)}

    def flowsheet(self, feed: Stream) -> Flowsheet:
        """The unit's :class:`~difflow.flowsheet.Flowsheet` on ``feed``."""
        p = self.params
        fs = Flowsheet(list(ALKYLATION_SPECIES), default_T=p.makeup.T, default_P=p.makeup.P)
        fs.add_feed("olefin_feed", feed)

        def col(name, spec, inlet, outs):
            fs.add_unit(Unit(name, self.columns[name], [inlet], outs,
                             params={"R": spec.reflux_ratio, "P": spec.P, "q": 1.0}))

        fs.add_unit(Unit("splitter", self.splitter, ["dib_recycle"],
                         ["dec3_feed", "recycle_direct"]))
        col("dec3", p.depropanizer, "dec3_feed", ["propane_product", "dec3_bottoms", "dec3_info"])
        fs.add_unit(Unit("makeup", self.makeup, ["olefin_feed", "recycle_direct", "dec3_bottoms"],
                         ["reactor_feed", "makeup"]))
        fs.add_unit(Unit("reactor", self.reactor, ["reactor_feed"], ["effluent", "reactor_info"]))
        col("dib", p.deisobutanizer, "effluent", ["dib_overhead", "dib_bottoms", "dib_info"])
        col("dec4", p.debutanizer, "dib_bottoms", ["butane_product", "alkylate", "dec4_info"])
        fs.add_recycle("dib_overhead", "dib_recycle")
        return fs

    def recycle_guess(self, feed: Stream) -> Stream:
        """A starting tear: the isobutane the I/O ratio needs, less the feed's,
        and the propane a purge at the spec'd fraction and recovery would hold.
        """
        p = self.params
        F = flows_array(feed)
        V_ol = jnp.sum(F[_OLEFIN_IDX] * _V60[_OLEFIN_IDX])
        n = jnp.maximum(p.makeup.io_ratio * V_ol - F[_IDX["isobutane"]] * _V60[_IDX["isobutane"]],
                        0.0) / _V60[_IDX["isobutane"]]
        c3 = F[_IDX["propane"]] / (p.depropanizer_fraction * p.depropanizer.lk_recovery)
        guess = (jnp.zeros(len(ALKYLATION_SPECIES)).at[_IDX["isobutane"]].set(0.95 * n)
                 .at[_IDX["propane"]].set(c3))
        return stream_from_array(guess, p.makeup.T, p.deisobutanizer.P)

    def solve(self, feed: Stream, **solve_kwargs) -> AlkylationResult:
        """Solve the unit on ``feed`` (a difflow stream of alkylation species).

        ``solve_kwargs`` go to :meth:`Flowsheet.solve` (``tol`` and
        ``max_iter`` default to the params'). Differentiable: under
        ``jax.grad`` the recycle is solved by ``optimistix``'s fixed point
        and differentiated implicitly.
        """
        p = self.params
        fs = self.flowsheet(feed)
        kw = dict(tol=p.tol, max_iter=p.max_iter,
                  tear_initial={"dib_recycle": self.recycle_guess(feed)},
                  on_nonconvergence="warn")
        kw.update(solve_kwargs)
        st = fs.solve(**kw)
        return self._result(feed, st, fs)

    def _result(self, feed: Stream, st: dict, fs: Flowsheet) -> AlkylationResult:
        rinfo = st["reactor_info"]
        F_feed = flows_array(feed)
        F_alk = flows_array(st["alkylate"])
        V_ol_feed = jnp.sum(F_feed[_OLEFIN_IDX] * _V60[_OLEFIN_IDX])
        V_alk = jnp.sum(F_alk * _V60)
        props = alkylate_properties(F_alk)
        F_mk = flows_array(st["makeup"])
        F_rec = flows_array(st["dib_overhead"])
        mw = 1e-6
        o = {
            "olefin.bpd": to_bpd(V_ol_feed),
            "io_ratio": rinfo["io_ratio"],
            "alkylate.bpd": to_bpd(V_alk),
            "alkylate.yield": V_alk / V_ol_feed,
            "alkylate.yield_correlation": rinfo["yield_correlation"],
            "alkylate.MON": rinfo["MON"],
            "alkylate.RON": rinfo["RON"],
            "alkylate.SG": props["SG"],
            "alkylate.RVP_psi": props["RVP_psi"],
            **{f"alkylate.T{p}_{b}": props[f"T{p}_{b}"] for p in (10, 50, 90) for b in ("tbp", "d86")},
            "alkylate.heavy_fraction": rinfo["heavy_fraction"],
            "isobutane.consumed_bpd": rinfo["isobutane_consumed_bpd"],
            "isobutane.makeup_bpd": to_bpd(F_mk[_IDX["isobutane"]] * _V60[_IDX["isobutane"]]),
            "isobutane.recycle_bpd": to_bpd(F_rec[_IDX["isobutane"]] * _V60[_IDX["isobutane"]]),
            "propane.bpd": to_bpd(jnp.sum(flows_array(st["propane_product"]) * _V60)),
            "n_butane.bpd": to_bpd(jnp.sum(flows_array(st["butane_product"]) * _V60)),
            "acid.lb_per_bbl": rinfo["acid_lb_per_bbl"],
            "acid.klb_d": rinfo["acid_lb_per_bbl"] * to_bpd(V_alk) / 1000.0,
            "reactor.heat": rinfo["heat_of_reaction"] * mw,
            "refrigeration.duty": rinfo["refrigeration_duty"] * mw,
            "refrigerant.isobutane": rinfo["refrigerant_isobutane"],
        }
        feasible = {}
        for name in ("dec3", "dib", "dec4"):
            info = st[f"{name}_info"]
            o[f"{name}.reboiler"] = info["Q_reboiler"] * mw
            o[f"{name}.condenser"] = -info["Q_condenser"] * mw
            try:
                feasible[name] = bool(info["feasible"])
            except Exception:
                feasible[name] = None
        return AlkylationResult(outputs=o, streams=st, converged=fs.last_solve_converged,
                                iterations=fs.last_solve_iterations, columns_feasible=feasible)

    def with_params(self, **changes) -> "AlkylationUnit":
        """A unit with top-level params replaced (``reactor=...``, ``makeup=...``)."""
        return AlkylationUnit(replace(self.params, **changes))


#: The plant's product streams (its outlets, besides the acid not carried).
PRODUCTS = ("propane_product", "butane_product", "alkylate")


def atom_balance(streams_in, streams_out) -> dict[str, tuple[Array, Array]]:
    """``{"C": (in, out), "H": (in, out), "mass": (in, out)}`` in mol/s and kg/s."""
    def tot(streams, w):
        return sum(jnp.sum(flows_array(s) * w) for s in streams)
    return {"C": (tot(streams_in, _NC), tot(streams_out, _NC)),
            "H": (tot(streams_in, _NH), tot(streams_out, _NH)),
            "mass": (tot(streams_in, _MW / 1000.0), tot(streams_out, _MW / 1000.0))}


__all__ = [
    "AlkylationResult", "AlkylationUnit", "AlkylationUnitParams", "ColumnSpec",
    "GeddesShortcutColumn", "IsobutaneMakeup", "IsobutaneMakeupParams", "OUTPUT_UNITS",
    "PRODUCTS", "RecycleSplitter", "TBP_WIDTH", "alkylate_properties", "atom_balance",
]
