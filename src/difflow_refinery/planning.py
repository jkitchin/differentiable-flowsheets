"""The crude unit as a delta-base planning block.

A refinery LP's crude-unit rows are base yields plus *shift vectors*: how
each product's rate and quality moves when a cut point, a pumparound or the
furnace is moved. Those vectors are usually hand-maintained and go stale.
:func:`cdu_block` computes them exactly, as the AD Jacobian of a rigorous
column at the current operating point, and the planner refreshes them every
trust-region cycle (:mod:`difflow.planning`).

Levers and outputs are named by what they mean -- ``naphtha.yield``,
``pa1.duty``, ``kero.tbp95``, ``gap.kero_diesel`` -- and carried in the units
a planner writes rows in, not the column's SI internals: bbl/d, MW, deg C
and volume fractions. That is also a conditioning choice: in m^3/s and W the
same block's delta vectors span fourteen orders of magnitude, in these units
about six.

Units, everywhere in this module (also in ``Block.metadata["u_units"]`` and
``["y_units"]``, which :mod:`difflow.planning.export` writes out):

=================  ===========================================================
``bbl/d``          standard barrels (60 F) per day
``-``              a volume (or mass, where named) fraction of the crude
``MW``             10^6 W; duties are positive as heat absorbed or removed
``C``              degrees Celsius -- every *temperature*: TBP points, cut
                   points, coil outlet, preheat, pumparound return, stages
``K``              a temperature *difference*: gaps, pumparound delta-T
``kg/h``           stripping steam
``API``            API gravity
``m2``             exchanger area (preheat train)
``m2K/kW``         fouling resistance: 1 m2K/kW is 1e-3 m2 K/W
``bar``            preflash drum pressure (absolute)
=================  ===========================================================

The builder's own keyword arguments -- the base crude rate and preheat
condition -- follow :meth:`~difflow_refinery.unit.CrudeUnit.solve` (bbl/d,
K, Pa), because they configure the unit; only the block's levers and outputs
are in planner units.

Non-convergence. A spec set can have no solution -- too much pumparound duty
for the overflash, say, and the column dries out. The column then reports
``converged=False`` with a finite, meaningless state. The block returns NaN
for every output at such a point, and :class:`~difflow.planning.DeltaBasePlanner`
rejects any proposal at which a block returns a non-finite value (and
refuses to start from one). A non-converged column is never scored.

Yields as levers, qualities as outputs. A ``<product>.yield`` lever gives an
identity column on that product's own yield; the information is in what the
yield does to everything else -- the TBP points, gaps, gravities and duties.
A planner that wants a cut-point *target* ("kero TBP95 <= 270 C") writes it
as a row on the ``kero.tbp95`` output, or on the effective cut point
``cut.kero_diesel``, and the LP inverts the delta vector to find the yield
that meets it. That is the same feasible set a TBP-cut-point lever would
describe, without a kinked spec inside the column's Newton solve (a TBP
point is piecewise linear in the flows) or a root find wrapped around it.

Example:
    >>> blk = cdu_block(unit, levers=["naphtha.yield", "pa1.duty"],   # doctest: +SKIP
    ...                 outputs=["naphtha.tbp95", "gap.naphtha_kero", "furnace.fired"],
    ...                 rate=95_000.0, T=513.15, P=6e5)
    >>> check_delta_vectors(blk)["passed"]                             # doctest: +SKIP
    True
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.planning.block import Block

from difflow_refinery import products as prod
from difflow_refinery.column import BARREL
from difflow_refinery.thermo import WATER_MW
from difflow_refinery.preheat.unit import PreheatedCrudeUnit
from difflow_refinery.unit import CrudeUnit, CrudeUnitResult

_DAY = 86400.0
_C = 273.15
_MW = 1e6

#: Per-product outputs and their units. ``tbp<p>`` for each TBP point.
PRODUCT_OUTPUTS: dict[str, str] = {
    "bpd": "bbl/d", "yield": "-", "yield_mass": "-", "sg": "-", "api": "API",
    "mw": "g/mol",
    **{f"tbp{int(p)}": "C" for p in prod.TBP_POINTS},
}

#: Unit-level outputs and their units (``<pa>.duty`` for each pumparound too).
UNIT_OUTPUTS: dict[str, str] = {
    "furnace.fired": "MW", "furnace.absorbed": "MW", "furnace.cot": "C",
    "furnace.vaporized": "-", "condenser.duty": "MW", "steam.total": "kg/h",
    "water.saturation_max": "-",
}

#: Outputs of a unit with a preheat train (and, per exchanger, ``<E>.duty``
#: in MW and ``<E>.approach`` in K; per hot stream ``<source>.train_out_T``;
#: ``desalter.T``/``.margin_low``/``.margin_high``; ``preflash.T``/``.P``
#: (bar)/``.vapor_fraction``).
PREHEAT_OUTPUTS: dict[str, str] = {"furnace.inlet_T": "C", "preheat.recovered": "MW"}

#: Per-product outputs a block reports when ``outputs`` is not given (see
#: :meth:`CDUPlanningModel.default_outputs` for the two exceptions).
DEFAULT_PRODUCT_OUTPUTS = ("bpd", "yield", "api", "tbp5", "tbp95")


# =============================================================================
# Levers
# =============================================================================


@dataclass(frozen=True)
class Lever:
    """One planning lever on the crude unit.

    Attributes:
        name: The lever's name, e.g. ``"naphtha.yield"``.
        units: Its planner unit (see the module table).
        kind: ``"rate"`` (crude rate), ``"preheat"`` (furnace inlet T),
            ``"spec"`` (the value of a column spec) or ``"steam"``; on a
            unit with a preheat train also ``"tank"`` (tank temperature),
            ``"exchanger"`` (an exchanger's ``Rf``, ``area`` or ``bypass``,
            named by the lever's suffix), ``"desalter"`` (wash water) and
            ``"drum"`` (preflash pressure or vapour fraction).
        index: The spec's position in the column's specs, the steam
            point (0 = bottom, then the side products in order), or the
            exchanger's position in the train.
        to_si: Planner value -> the column's own value.
        description: What it moves, in words.
    """

    name: str
    units: str
    kind: str
    index: int | None
    description: str

    def to_si(self, value, feed_volume):
        """Planner value -> column value (spec value, K, mol/s or bbl/d)."""
        u = self.units
        if self.kind == "spec":
            if u == "-" and self.name.endswith(".yield"):
                return value * feed_volume
            if u == "bbl/d":
                return value * BARREL / _DAY
            if u == "MW":
                return value * _MW
            if u == "C":
                return value + _C
            return value
        if self.kind == "steam":
            return value / 3600.0 / (WATER_MW / 1000.0)
        if self.kind == "preheat" or self.kind == "tank":
            return value + _C
        if self.kind == "exchanger" and self.units == "m2K/kW":
            return value / 1000.0
        if self.kind == "drum" and self.units == "bar":
            return value * 1e5
        return value

    def from_si(self, value, feed_volume):
        """Column value -> planner value; the inverse of :meth:`to_si`."""
        u = self.units
        if self.kind == "spec":
            if u == "-" and self.name.endswith(".yield"):
                return value / feed_volume
            if u == "bbl/d":
                return value * _DAY / BARREL
            if u == "MW":
                return value / _MW
            if u == "C":
                return value - _C
            return value
        if self.kind == "steam":
            return value * 3600.0 * (WATER_MW / 1000.0)
        if self.kind == "preheat" or self.kind == "tank":
            return value - _C
        if self.kind == "exchanger" and self.units == "m2K/kW":
            return value * 1000.0
        if self.kind == "drum" and self.units == "bar":
            return value / 1e5
        return value


def _column_params(unit):
    """The column params of a :class:`CrudeUnit` or a :class:`PreheatedCrudeUnit`."""
    return unit.column_params if isinstance(unit, PreheatedCrudeUnit) else unit.params


def _train_cooled(unit) -> set[str]:
    """Pumparounds whose return temperature the preheat train sets."""
    return set(unit.train.sources) if isinstance(unit, PreheatedCrudeUnit) else set()


def _spec_levers(unit) -> list[Lever]:
    """The levers the column's own specs offer: one (or two) per spec.

    A pumparound the preheat train cools has no ``return_T`` lever: its
    return temperature is the train's answer, and its spec value only the
    loop's starting guess.
    """
    out = []
    p = _column_params(unit)
    cooled = _train_cooled(unit)
    for i, s in enumerate(p.specs):
        k, t = s.kind, s.target
        if k == "product_rate":
            if s.basis == "volume":
                out.append(Lever(f"{t}.yield", "-", "spec", i,
                                 f"{t} rate as a volume fraction of the crude"))
                out.append(Lever(f"{t}.bpd", "bbl/d", "spec", i, f"{t} rate"))
            # a mole or mass rate spec is held, not offered: a planner has no
            # row in mol/s, and converting it would need the product's MW,
            # which is an output
        elif k == "overflash":
            out.append(Lever("overflash", "-", "spec", i,
                             f"overflash, {s.basis} fraction of the crude"))
        elif k == "pa_duty":
            out.append(Lever(f"{t}.duty", "MW", "spec", i, f"heat removed by pumparound {t}"))
        elif k == "pa_delta_T":
            out.append(Lever(f"{t}.dT", "K", "spec", i, f"pumparound {t} draw less return temperature"))
        elif k == "pa_return_T" and t not in cooled:
            out.append(Lever(f"{t}.return_T", "C", "spec", i, f"pumparound {t} return temperature"))
        elif k == "pa_rate" and s.basis == "volume":
            out.append(Lever(f"{t}.rate", "bbl/d", "spec", i, f"pumparound {t} circulation"))
        elif k == "furnace_T":
            out.append(Lever("furnace.cot", "C", "spec", i, "furnace coil outlet temperature"))
        elif k == "furnace_duty":
            out.append(Lever("furnace.duty", "MW", "spec", i, "heat absorbed in the furnace"))
        elif k == "reflux_ratio":
            out.append(Lever("reflux_ratio", "-", "spec", i, "molar reflux ratio"))
        elif k == "stage_T":
            out.append(Lever(f"stage{int(t)}.T", "C", "spec", i, f"temperature of stage {t}"))
    return out


def available_levers(unit) -> dict[str, Lever]:
    """Every lever a :func:`cdu_block` can take on this unit, by name.

    ``crude.rate`` (bbl/d) and ``preheat.T`` (C, the furnace inlet), the
    stripping steam (``bottom.steam`` and ``<side product>.steam``, kg/h,
    for each side product with a stripper), and one lever per column spec:
    a volume product-rate spec offers both ``<product>.yield`` (fraction of
    crude) and ``<product>.bpd``; ``overflash``; ``<pa>.duty`` (MW),
    ``<pa>.dT`` (K), ``<pa>.return_T`` (C), ``<pa>.rate`` (bbl/d);
    ``furnace.cot`` (C), ``furnace.duty`` (MW absorbed); ``reflux_ratio``;
    ``stage<k>.T`` (C).

    A spec can only be a lever if the column has it: the specs fix the
    shape of the column's equations, so a column closed by an overflash has
    an ``overflash`` lever and no ``furnace.cot``.

    On a :class:`~difflow_refinery.preheat.PreheatedCrudeUnit` the furnace
    inlet is an answer, so ``preheat.T`` is replaced by the train's own
    levers: ``tank.T`` (C); per exchanger ``<E>.Rf`` (m2K/kW),
    ``<E>.area`` (m2) and ``<E>.bypass`` (-); ``desalter.wash`` (-, fraction
    of the crude) with a desalter; ``preflash.P`` (bar), or
    ``preflash.vapor_fraction`` (-) for a drum specified that way. A
    pumparound the train cools loses its ``return_T`` lever.
    """
    levers = [Lever("crude.rate", "bbl/d", "rate", None, "crude charge")]
    if isinstance(unit, PreheatedCrudeUnit):
        levers += _train_levers(unit)
    else:
        levers.append(Lever("preheat.T", "C", "preheat", None,
                            "furnace inlet temperature (preheat train outlet)"))
    levers.append(Lever("bottom.steam", "kg/h", "steam", 0, "bottom stripping steam"))
    j = 1
    for sp in _column_params(unit).side_products:
        if sp.stripper_stages > 0:
            levers.append(Lever(f"{sp.name}.steam", "kg/h", "steam", j, f"{sp.name} stripper steam"))
            j += 1
    levers += _spec_levers(unit)
    return {lv.name: lv for lv in levers}


def _train_levers(unit) -> list[Lever]:
    tp = unit.train_params
    out = [Lever("tank.T", "C", "tank", None, "tank (train inlet) temperature")]
    for i, ex in enumerate(tp.exchangers):
        out += [Lever(f"{ex.name}.Rf", "m2K/kW", "exchanger", i, f"fouling resistance of {ex.name}"),
                Lever(f"{ex.name}.area", "m2", "exchanger", i, f"area of {ex.name}"),
                Lever(f"{ex.name}.bypass", "-", "exchanger", i, f"hot-side bypass of {ex.name}")]
    if tp.desalter is not None:
        out.append(Lever("desalter.wash", "-", "desalter", None, "desalter wash water, fraction of the crude"))
    if tp.drum is not None:
        if tp.drum.vapor_fraction is None:
            out.append(Lever("preflash.P", "bar", "drum", None, "preflash drum pressure"))
        else:
            out.append(Lever("preflash.vapor_fraction", "-", "drum", None,
                             "fraction of the crude's hydrocarbons flashed (moles)"))
    return out


# =============================================================================
# The model
# =============================================================================


class CDUPlanningModel:
    """The pure ``u -> y`` map behind :func:`cdu_block`.

    Callable on a lever array (planner units) and returning the output array
    (planner units), NaN everywhere when the column does not converge.
    :meth:`solve` returns the whole :class:`CrudeUnitResult` at a lever
    array, for looking at a point the block reports as NaN.

    Specs the block does not lever are held: a volume product rate as its
    *yield* (so it follows ``crude.rate``), everything else at its value.
    """

    def __init__(self, unit, levers: Sequence[str], outputs: Sequence[str] | None,
                 rate: float, T: float, P: float | None, mask_nonconverged: bool = True):
        self.unit = unit
        self.preheated = isinstance(unit, PreheatedCrudeUnit)
        if P is None and not self.preheated:
            raise ValueError("P (the furnace inlet pressure) is needed for a CrudeUnit")
        self.rate, self.T = float(rate), float(T)
        self.P = None if P is None else float(P)
        self.mask_nonconverged = bool(mask_nonconverged)
        avail = available_levers(unit)
        unknown = [n for n in levers if n not in avail]
        if unknown:
            raise ValueError(f"no lever(s) {unknown} on this unit; available: {sorted(avail)}")
        if len(set(levers)) != len(levers):
            raise ValueError("duplicate levers")
        self.levers = [avail[n] for n in levers]
        idx = [lv.index for lv in self.levers if lv.kind == "spec"]
        if len(set(idx)) != len(idx):
            raise ValueError("two levers move the same spec (a product's .yield and .bpd?); pick one")

        self.base_feed_volume = self.rate * BARREL / _DAY
        # the base solve fixes the product order (by TBP50, for gaps and cut
        # points) and is where u0 is read from
        base = (unit.solve(self.rate, self.T) if self.preheated
                else unit.solve(self.rate, self.T, self.P))
        if not bool(base.converged):
            raise ValueError("the unit does not converge at its base point; "
                             "a planning block has to start from a solution")
        self.base = base
        names = list(base.properties)
        self.order = sorted(names, key=lambda n: float(base.properties[n].tbp_at(50)))
        self.products = [n for n in self.order if n != "offgas"]
        self.pumparounds = [pa.name for pa in _column_params(unit).pumparounds]
        self.output_units = self._output_table()
        if outputs is None:
            # a lever as an output is an identity row; the defaults leave it out
            outputs = [n for n in self.default_outputs() if n not in set(levers)]
        outputs = list(outputs)
        unknown = [n for n in outputs if n not in self.output_units]
        if unknown:
            raise ValueError(f"no output(s) {unknown}; available: {sorted(self.output_units)}")
        both = sorted(set(outputs) & set(levers))
        if both:
            raise ValueError(f"{both} are levers: as outputs they would be the lever itself")
        self.outputs = outputs
        self.u0 = np.array([self._base_value(lv) for lv in self.levers])

    # -- tables ----------------------------------------------------------

    def _output_table(self) -> dict[str, str]:
        out = {}
        for n in self.order:
            for k, u in PRODUCT_OUTPUTS.items():
                out[f"{n}.{k}"] = u
        for a, b in zip(self.products[:-1], self.products[1:]):
            out[f"gap.{a}_{b}"] = "K"
            out[f"cut.{a}_{b}"] = "C"
        out.update(UNIT_OUTPUTS)
        for pa in self.pumparounds:
            out[f"{pa}.duty"] = "MW"
        if self.preheated:
            out.update(PREHEAT_OUTPUTS)
            tp = self.unit.train_params
            for ex in tp.exchangers:
                out[f"{ex.name}.duty"] = "MW"
                out[f"{ex.name}.approach"] = "K"
            for src in self.unit.train.sources:
                out[f"{src}.train_out_T"] = "C"
            if tp.desalter is not None:
                out.update({"desalter.T": "C", "desalter.margin_low": "K", "desalter.margin_high": "K"})
            if tp.drum is not None:
                out.update({"preflash.T": "C", "preflash.P": "bar", "preflash.vapor_fraction": "-"})
        return out

    def default_outputs(self) -> list[str]:
        """The outputs a block reports by default.

        Per product bbl/d, yield, API and TBP 5/95; every gap and cut
        point; the fired duty; the steam when a steam rate is a lever (it
        is a constant otherwise). Two more kinds are left out,
        because :func:`~difflow.planning.check_delta_health` would rightly
        flag them: the *yield* of a product whose volume rate spec is held
        (a held spec holds the yield, so the row is structurally zero), and
        the TBP5 of the lightest product, which lies among the discrete
        light ends -- a TBP point is piecewise linear in the product's
        cumulative volume, with one node per component, and there the
        segments are tens of degrees wide (the test crude's naphtha TBP5
        sits *on* n-butane's node at its base point, slope 277 K per unit
        yield to the left and 146 to the right). Ask for either explicitly
        if it is wanted.
        """
        held = {s.target for i, s in enumerate(_column_params(self.unit).specs)
                if s.kind == "product_rate" and s.basis == "volume"
                and i not in {lv.index for lv in self.levers if lv.kind == "spec"}}
        out = [f"{n}.{k}" for n in self.products for k in DEFAULT_PRODUCT_OUTPUTS
               if not (k == "yield" and n in held)
               and not (k == "tbp5" and n == self.products[0])]
        out += [f"gap.{a}_{b}" for a, b in zip(self.products[:-1], self.products[1:])]
        out += [f"cut.{a}_{b}" for a, b in zip(self.products[:-1], self.products[1:])]
        out.append("furnace.fired")
        if any(lv.kind == "steam" for lv in self.levers):
            out.append("steam.total")       # a constant unless steam is a lever
        if self.preheated:
            out += ["furnace.inlet_T", "preheat.recovered"]
            tp = self.unit.train_params
            if tp.desalter is not None:
                out.append("desalter.T")
            if tp.drum is not None:
                out += ["preflash.T"] + (["preflash.vapor_fraction"] if tp.drum.vapor_fraction is None
                                         else ["preflash.P"])
        return out

    def _base_value(self, lv: Lever) -> float:
        p = _column_params(self.unit)
        if lv.kind == "rate":
            return self.rate
        if lv.kind in ("preheat", "tank"):
            return lv.from_si(self.T, None)
        if lv.kind == "exchanger":
            ex = self.unit.train_params.exchangers[lv.index]
            return float(lv.from_si(float(getattr(ex, lv.name.rsplit(".", 1)[1])), None))
        if lv.kind == "desalter":
            return float(self.unit.train_params.desalter.wash_water)
        if lv.kind == "drum":
            d = self.unit.train_params.drum
            return float(lv.from_si(float(d.P if d.vapor_fraction is None else d.vapor_fraction), None))
        if lv.kind == "steam":
            steam = [p.bottom_steam] + [sp.steam for sp in p.side_products if sp.stripper_stages > 0]
            return float(lv.from_si(float(steam[lv.index]), None))
        return float(lv.from_si(float(p.specs[lv.index].value), self.base_feed_volume))

    # -- evaluation ------------------------------------------------------

    def _params_and_feed(self, u):
        vals = {lv.name: u[i] for i, lv in enumerate(self.levers)}
        rate = vals.get("crude.rate", self.rate)
        T = vals["preheat.T"] + _C if "preheat.T" in vals else self.T
        T = vals["tank.T"] + _C if "tank.T" in vals else T
        feed_volume = rate * BARREL / _DAY
        p = _column_params(self.unit)
        specs = list(p.specs)
        # a held volume product rate keeps its yield, so a crude-rate move
        # scales it rather than handing the whole change to the residue
        for i, s in enumerate(specs):
            if s.kind == "product_rate" and s.basis == "volume":
                specs[i] = replace(s, value=s.value / self.base_feed_volume * feed_volume)
        steam = [p.bottom_steam] + [sp.steam for sp in p.side_products if sp.stripper_stages > 0]
        for i, lv in enumerate(self.levers):
            if lv.kind == "spec":
                specs[lv.index] = replace(specs[lv.index], value=lv.to_si(u[i], feed_volume))
            elif lv.kind == "steam":
                steam[lv.index] = lv.to_si(u[i], feed_volume)
        sides, j = [], 1
        for sp in p.side_products:
            if sp.stripper_stages > 0:
                sides.append(replace(sp, steam=steam[j]))
                j += 1
            else:
                sides.append(sp)
        params = replace(p, specs=tuple(specs), bottom_steam=steam[0], side_products=tuple(sides))
        return params, rate, T

    def _train_of(self, u):
        """The preheat train's params with the train levers applied."""
        tp = self.unit.train_params
        exs = list(tp.exchangers)
        des, drum = tp.desalter, tp.drum
        for i, lv in enumerate(self.levers):
            if lv.kind == "exchanger":
                field = lv.name.rsplit(".", 1)[1]
                exs[lv.index] = replace(exs[lv.index], **{field: lv.to_si(u[i], None)})
            elif lv.kind == "desalter":
                des = replace(des, wash_water=u[i])
            elif lv.kind == "drum":
                key = "P" if drum.vapor_fraction is None else "vapor_fraction"
                drum = replace(drum, **{key: lv.to_si(u[i], None)})
        return replace(tp, exchangers=tuple(exs), desalter=des, drum=drum)

    def _solve(self, u):
        params, rate, T = self._params_and_feed(u)
        if self.preheated:
            return self.unit.solve(rate, T, basis="bpd", column=params, train=self._train_of(u)), params
        return self.unit.solve(rate, T, self.P, basis="bpd", params=params), params

    def solve(self, u) -> CrudeUnitResult:
        """The unit's full result at a lever array (planner units)."""
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        return self._solve(u)[0]

    def outputs_of(self, res: CrudeUnitResult, params=None) -> dict[str, Array]:
        """Every available output, in planner units, from a result.

        ``params`` are the column parameters ``res`` was solved with (for
        the steam fed); default the unit's own.
        """
        props, col = res.properties, res.column
        out = {}
        for n, pp in props.items():
            out[f"{n}.bpd"] = pp.bpd
            out[f"{n}.yield"] = pp.yield_volume
            out[f"{n}.yield_mass"] = pp.yield_mass
            out[f"{n}.sg"] = pp.sg
            out[f"{n}.api"] = pp.api
            out[f"{n}.mw"] = pp.mw
            for k, T in zip(prod.TBP_POINTS, pp.tbp):
                out[f"{n}.tbp{int(k)}"] = T - _C
        th = self.unit.thermo
        feed_flows = jnp.stack([jnp.asarray(res.feed[f"F_{c}"], dtype=float) for c in th.names])
        cum = jnp.cumsum(jnp.stack([props[n].yield_volume for n in self.order]))
        cuts = prod.tbp_curve(feed_flows, th, percents=100.0 * cum)
        pos = {n: i for i, n in enumerate(self.order)}
        for a, b in zip(self.products[:-1], self.products[1:]):
            out[f"gap.{a}_{b}"] = props[b].tbp_at(5) - props[a].tbp_at(95)
            # the effective cut point: the crude's own TBP at the cumulative
            # yield taken through the lighter product -- the temperature a
            # planner means by "the kero/diesel cut"
            out[f"cut.{a}_{b}"] = cuts[pos[a]] - _C
        out["furnace.fired"] = col.furnace_fired_duty / _MW
        out["furnace.absorbed"] = col.furnace_duty / _MW
        out["furnace.cot"] = col.coil_outlet_T - _C
        out["furnace.vaporized"] = col.feed_vaporized
        out["condenser.duty"] = col.condenser_duty / _MW
        # the steam the unit is fed -- what the utility bill counts -- from
        # the parameters the column was solved with, so it follows the
        # steam levers
        params = _column_params(self.unit) if params is None else params
        out["steam.total"] = (jnp.asarray(params.bottom_steam, dtype=float) + sum(
            jnp.asarray(sp.steam, dtype=float) for sp in params.side_products
            if sp.stripper_stages > 0)) * 3600.0 * (WATER_MW / 1000.0)
        out["water.saturation_max"] = jnp.max(col.water_saturation)
        for k, pa in enumerate(self.pumparounds):
            out[f"{pa}.duty"] = col.pumparound_duty[k] / _MW
        if self.preheated:
            tr = res.train
            out["furnace.inlet_T"] = res.furnace_inlet_T - _C
            out["preheat.recovered"] = res.recovered / _MW
            for name, e in tr.exchangers.items():
                out[f"{name}.duty"] = e["Q"] / _MW
                out[f"{name}.approach"] = e["approach"]
            for src, T in tr.hot_outlet_T.items():
                out[f"{src}.train_out_T"] = T - _C
            if tr.desalter:
                d = tr.desalter
                out.update({"desalter.T": d["T"] - _C, "desalter.margin_low": d["margin_low"],
                            "desalter.margin_high": d["margin_high"]})
            if tr.drum:
                out.update({"preflash.T": tr.drum["T"] - _C, "preflash.P": tr.drum["P"] / 1e5,
                            "preflash.vapor_fraction": tr.drum["vapor_fraction"]})
        return out

    def __call__(self, u) -> Array:
        u = jnp.atleast_1d(jnp.asarray(u, dtype=float))
        res, params = self._solve(u)
        allv = self.outputs_of(res, params)
        y = jnp.stack([jnp.asarray(allv[n], dtype=float) for n in self.outputs])
        if not self.mask_nonconverged:
            return y
        return jnp.where(res.converged, y, jnp.nan)


# =============================================================================
# The builder
# =============================================================================


def cdu_block(unit: CrudeUnit | PreheatedCrudeUnit, levers: Sequence[str],
              outputs: Sequence[str] | None = None, *,
              rate: float, T: float, P: float | None = None, name: str = "cdu",
              bounds: Mapping[str, tuple[float, float]] | None = None, span: float = 0.1,
              jit: bool = True, mask_nonconverged: bool = True, **kwargs: Any) -> Block:
    """A :class:`~difflow.planning.Block` for a crude unit.

    Args:
        unit: The :class:`~difflow_refinery.unit.CrudeUnit`, or a
            :class:`~difflow_refinery.preheat.PreheatedCrudeUnit` -- the
            column behind its preheat train, whose furnace inlet is then an
            output and whose exchangers are levers. Its specs fix which
            levers exist (see :func:`available_levers`) and their base
            values.
        levers: Lever names, e.g. ``["naphtha.yield", "kero.yield",
            "overflash", "pa1.duty", "crude.rate", "preheat.T"]``.
        outputs: Output names; default :meth:`CDUPlanningModel.default_outputs`.
            Per product ``<p>.bpd``, ``.yield``, ``.yield_mass``, ``.sg``,
            ``.api``, ``.mw``, ``.tbp5`` ... ``.tbp95``; per neighbouring
            pair ``gap.<a>_<b>`` (TBP5 of the heavier less TBP95 of the
            lighter, K) and ``cut.<a>_<b>`` (the effective cut point, C);
            ``furnace.fired``, ``furnace.absorbed``, ``furnace.cot``,
            ``furnace.vaporized``, ``condenser.duty``, ``<pa>.duty``,
            ``steam.total``, ``water.saturation_max``. With a preheat
            train also ``furnace.inlet_T`` (C), ``preheat.recovered`` (MW),
            ``<E>.duty`` (MW), ``<E>.approach`` (K),
            ``<source>.train_out_T`` (C), ``desalter.T``,
            ``desalter.margin_low``/``_high`` (K), ``preflash.T``,
            ``preflash.P`` (bar), ``preflash.vapor_fraction``. The fouling
            sensitivities are the ``furnace.fired`` row of the delta vectors
            against the ``<E>.Rf`` levers.
        rate: Base crude rate (bbl/d), the linearisation point's.
        T, P: Base furnace inlet temperature (K) and pressure (Pa), as for
            :meth:`~difflow_refinery.unit.CrudeUnit.solve`. For a
            :class:`~difflow_refinery.preheat.PreheatedCrudeUnit`, ``T`` is
            the tank temperature and ``P`` is not used.
        name: Block name.
        bounds: ``{lever: (lo, hi)}`` in planner units. A lever without one
            gets ``u0 * (1 -/+ span)``. Bounds matter: the trust region
            steps in fractions of them.
        span: Default relative half-width of a lever's bounds.
        jit: Compile the block (recommended: a 30-stage solve is ~0.2 s
            compiled against ~2 s eager, after a compile of a few seconds).
        mask_nonconverged: Return NaN when the column does not converge.
            Leave it on. ``False`` reports the non-converged state as if it
            were an answer, and exists only to show what the planner would
            then accept.
        **kwargs: Passed to :class:`~difflow.planning.Block` (``ad_mode``,
            ``phase_fn``, ...). ``ad_mode`` defaults to ``"auto"``, which
            picks forward mode for the usual few levers and many outputs.

    Returns:
        A :class:`~difflow.planning.Block` with ``fn`` a
        :class:`CDUPlanningModel` (``block.fn.solve(u)`` gives the full
        result at a point) and ``metadata`` carrying ``u_units``, ``y_units``
        and a description of every lever.
    """
    model = CDUPlanningModel(unit, list(levers), outputs, rate, T, P, mask_nonconverged)
    u0 = model.u0
    lb, ub = [], []
    bounds = dict(bounds or {})
    unknown = sorted(set(bounds) - set(levers))
    if unknown:
        raise ValueError(f"bounds given for {unknown}, which are not levers")
    for lv, x in zip(model.levers, u0):
        if lv.name in bounds:
            lo, hi = bounds[lv.name]
        else:
            w = abs(x) * span if x != 0 else span
            lo, hi = x - w, x + w
        if not lo <= x <= hi:
            raise ValueError(f"{lv.name}: base value {x:g} is outside its bounds ({lo:g}, {hi:g})")
        lb.append(lo)
        ub.append(hi)
    metadata = dict(kwargs.pop("metadata", {}) or {})
    metadata.setdefault("source", "difflow_refinery.cdu_block")
    metadata.setdefault("u_units", [lv.units for lv in model.levers])
    metadata.setdefault("y_units", [model.output_units[n] for n in model.outputs])
    metadata.setdefault("u_descriptions", [lv.description for lv in model.levers])
    return Block(name=name, fn=model, u_names=list(levers), y_names=list(model.outputs),
                 lb=lb, ub=ub, u0=u0, jit=jit, metadata=metadata, **kwargs)


# =============================================================================
# A downstream block: what the products are worth
# =============================================================================


def product_value_block(prices: Mapping[str, float], *, name: str = "value",
                        rates: Mapping[str, float] | None = None, jit: bool = True) -> Block:
    """Product revenue ($/d) from CDU product rates.

    The downstream block a crude unit links into: the first thing a
    refinery LP does with CDU products is price them. Inputs are
    ``<product>.bpd`` (bbl/d), named as the CDU names them so
    :func:`link_cdu` wires them by name; the one output is ``revenue``
    ($/d), priced at 1.0 in the planner.

    Product *qualities* are not in this block on purpose. A quality limit
    ("kero TBP95 <= 235 C") is a planner :class:`~difflow.planning.lp.Spec`
    on the CDU's own output, which the LP holds exactly at every step. A
    smooth regrade penalty folded into the revenue instead puts the
    limit's curvature in the objective, where a linear model cannot see
    it: tried on the test crude, the trust region crawled along the
    penalty's shoulder at radius 1e-4 and hit its iteration cap.

    Args:
        prices: $/bbl by product name.
        name: Block name.
        rates: Base bbl/d by product, the inputs' nominal point; bounds are
            ``[0, 2 rate]`` (default rate 1e4). Linked inputs are pinned by
            their links, so the bounds only have to contain them.
        jit: Compile the block.

    Returns:
        A :class:`~difflow.planning.Block`.
    """
    products = list(prices)
    rates = dict(rates or {})
    base = [float(rates.get(p, 1e4)) for p in products]
    price = jnp.asarray([float(prices[p]) for p in products])

    def fn(u):
        return jnp.atleast_1d(jnp.sum(price * u))

    return Block(name=name, fn=fn, u_names=[f"{p}.bpd" for p in products], y_names=["revenue"],
                 lb=[0.0] * len(products), ub=[2.0 * b for b in base], u0=base, jit=jit,
                 metadata={"source": "difflow_refinery.product_value_block",
                           "u_units": ["bbl/d"] * len(products), "y_units": ["$/d"]})


def link_cdu(cdu: Block, downstream: Block) -> list[tuple[str, str]]:
    """Links from every CDU output to the downstream input of the same name."""
    shared = [n for n in downstream.u_names if n in cdu.y_names]
    missing = sorted(set(downstream.u_names) - set(shared))
    if missing:
        raise ValueError(f"{downstream.name} needs {missing}, which {cdu.name} does not report; "
                         "add them to the CDU block's outputs")
    return [(f"{cdu.name}.{n}", f"{downstream.name}.{n}") for n in shared]


__all__ = ["CDUPlanningModel", "DEFAULT_PRODUCT_OUTPUTS", "Lever", "PRODUCT_OUTPUTS", "UNIT_OUTPUTS",
           "available_levers", "cdu_block", "link_cdu",
           "product_value_block"]
