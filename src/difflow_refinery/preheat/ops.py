"""The preheat train's pieces as flowsheet operations.

:class:`Desalter` and :class:`PreflashDrum` are the two non-exchanger
pieces of the train on their own, for a flowsheet that wires its own
exchangers; :class:`CrudeUnitWithPreheat` is the whole of
:class:`~difflow_refinery.preheat.unit.PreheatedCrudeUnit` -- tank crude in,
column products out. Like :class:`~difflow_refinery.unit.CrudeDistillationUnit`
each is built on an assay, which fixes the pseudo-components and so the
streams' species.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream

from difflow_refinery.assay import Assay, characterize, default_cut_points
from difflow_refinery.column import CrudeColumnParams
from difflow_refinery.preheat._newton import damped_newton, implicit_step
from difflow_refinery.preheat.flash import crude_enthalpy, crude_split
from difflow_refinery.preheat.train import (
    _LNP,
    DesalterParams,
    PreflashDrumParams,
    PreheatTrainParams,
    _water_moles,
)
from difflow_refinery.preheat.unit import PreheatedCrudeUnit
from difflow_refinery.thermo import ColumnThermo


def _thermo(assay, cut_points, method):
    cut = tuple(default_cut_points(assay) if cut_points is None else cut_points)
    crude = characterize(assay, cut, method)
    return crude, ColumnThermo.from_characterization(crude)


def _flows(stream, th):
    return (jnp.stack([jnp.asarray(stream[f"F_{n}"], dtype=float) for n in th.names]),
            jnp.asarray(stream.get("F_water", 0.0), dtype=float))


def _stream(th, f, fw, T, P):
    out = {f"F_{n}": f[i] for i, n in enumerate(th.names)}
    out.update(F_water=jnp.asarray(fw, dtype=float), T=T, P=P)
    return out


def _solve_1d(residual, z0, args):
    """Newton with an implicit derivative, for the small solves below."""
    frozen = jax.lax.stop_gradient(args)
    z1, norm, _ = damped_newton(lambda z: residual(z, frozen), z0, 50, 1e-10, 30.0)
    return implicit_step(residual, z1, args, frozen), norm


# =============================================================================
# Desalter
# =============================================================================


@dataclass
class DesalterUnitParams(ParamsMixin):
    """Parameters for :class:`Desalter`.

    Attributes:
        assay: The crude's assay; it fixes the pseudo-components.
        desalter: The desalter's :class:`~difflow_refinery.preheat.train.DesalterParams`.
        cut_points: Interior cut boundaries (K); default from the assay.
        method: Critical-property correlation for the pseudo-components.
    """

    assay: Assay
    desalter: DesalterParams
    cut_points: tuple | None = None
    method: str | None = None

    def __post_init__(self):
        if self.cut_points is not None:
            self.cut_points = tuple(self.cut_points)


class Desalter:
    """Crude desalter: wash water in, brine out, adiabatic.

    The inlet is a wet crude (``F_<component>``, ``F_water``, ``T``, ``P``);
    the outlets are the desalted crude, carrying the BS&W
    ``desalter.water_out``, and the brine -- the inlet water plus the wash
    water less what stays in the crude. The outlet temperature closes the
    energy balance; whether it lies in the desalter's window is reported by
    :meth:`solve`, not imposed.

    Key equations:
        Water: brine = F_water,in + wash - water_out
        Energy: H_crude(T_in) + wash h_w(T_wash) = H_crude(T) + brine h_w(T)

    Assumptions:
        Electrostatic separation is not modelled: the efficiency and the
        outlet BS&W are specifications
        Water immiscible with the hydrocarbon liquid; three-phase crude enthalpy
        No pressure drop

    Example:
        >>> from difflow_refinery import Assay
        >>> from difflow_refinery.preheat import Desalter, DesalterUnitParams, DesalterParams
        >>> op = Desalter(DesalterUnitParams(
        ...     assay=Assay([0, 30, 70, 100], [320., 480., 640., 900.], sg=0.85),
        ...     desalter=DesalterParams()))
        >>> crude, brine = op(op.feed(100.0, T=400.0, P=10e5))
    """

    symbol = "DS"
    numerical_method = "Newton on the outlet temperature; implicit-function gradients"
    parameter_units = {"cut_points": "K"}
    outlet_names = ("crude", "brine")

    def __init__(self, params: DesalterUnitParams):
        self.params = params
        self.crude, self.thermo = _thermo(params.assay, params.cut_points, params.method)

    def feed(self, rate, T, P, water=0.002):
        """A wet crude: ``rate`` mol/s of hydrocarbons, ``water`` BS&W."""
        s = self.crude.stream(rate, T=T, P=P, basis="mole")
        f, _ = _flows(s, self.thermo)
        return _stream(self.thermo, f, _water_moles(self.thermo, f, water), jnp.asarray(T, float), jnp.asarray(P, float))

    def solve(self, feed: dict) -> dict:
        """The outlets and the desalter's state: ``crude``, ``brine``, ``T``,
        ``margin_low``, ``margin_high``, ``salt_out``."""
        th, d = self.thermo, self.params.desalter
        f, fw = _flows(feed, th)
        T_in, P = jnp.asarray(feed["T"], float), jnp.asarray(feed["P"], float)
        args = {"th": th, "f": f, "fw": fw, "T_in": T_in, "P": P,
                "wash": _water_moles(th, f, jnp.asarray(d.wash_water, float)),
                "out": _water_moles(th, f, jnp.asarray(d.water_out, float)),
                "T_wash": jnp.asarray(d.T_wash, float)}

        def r(z, a):
            brine = a["fw"] + a["wash"] - a["out"]
            H_in = crude_enthalpy(a["th"], a["f"], a["fw"], a["T_in"], a["P"]) + a["wash"] * a["th"].water_h_liquid(a["T_wash"])
            H_out = crude_enthalpy(a["th"], a["f"], a["out"], z[0], a["P"]) + brine * a["th"].water_h_liquid(z[0])
            return jnp.stack([(H_in - H_out) / (400.0 * jnp.sum(a["f"]))])

        z, _ = _solve_1d(r, jnp.stack([T_in]), args)
        T = z[0]
        brine = fw + args["wash"] - args["out"]
        return {
            "crude": _stream(th, f, args["out"], T, P),
            "brine": _stream(th, jnp.zeros_like(f), brine, T, P),
            "T": T, "margin_low": T - jnp.asarray(d.T_min), "margin_high": jnp.asarray(d.T_max) - T,
            "salt_out": (1.0 - jnp.asarray(d.efficiency)) * jnp.asarray(d.salt_in),
        }

    def __call__(self, feed: Stream) -> tuple[Stream, Stream]:
        out = self.solve(feed)
        return out["crude"], out["brine"]


# =============================================================================
# Preflash drum
# =============================================================================


@dataclass
class PreflashDrumUnitParams(ParamsMixin):
    """Parameters for :class:`PreflashDrum`.

    Attributes:
        assay: The crude's assay; it fixes the pseudo-components.
        drum: The drum's :class:`~difflow_refinery.preheat.train.PreflashDrumParams`
            (its ``vapor_stage`` is unused here: the vapour is an outlet).
        cut_points: Interior cut boundaries (K); default from the assay.
        method: Critical-property correlation for the pseudo-components.
    """

    assay: Assay
    drum: PreflashDrumParams
    cut_points: tuple | None = None
    method: str | None = None

    def __post_init__(self):
        if self.cut_points is not None:
            self.cut_points = tuple(self.cut_points)


class PreflashDrum:
    """Preflash drum: an adiabatic three-phase flash of the wet crude.

    The crude is let down to the drum pressure (or to the pressure that
    flashes ``drum.vapor_fraction`` of its hydrocarbons) and separates into
    a vapour of hydrocarbons and water, a hydrocarbon liquid, and free
    water, decanted. On the column's thermodynamics, so the drum and the
    column agree on every K-value and enthalpy.

    Key equations:
        H(T_in, P_in) + duty = H(T, P), three-phase
        Hydrocarbons: Rachford-Rice with K_i = Psat_i/P, against P - Psat_w when free water is present

    Assumptions:
        Raoult's law (Lee-Kesler vapour pressures); water immiscible with the hydrocarbon liquid
        Equilibrium; no entrainment

    Example:
        >>> from difflow_refinery import Assay
        >>> from difflow_refinery.preheat import PreflashDrum, PreflashDrumUnitParams, PreflashDrumParams
        >>> op = PreflashDrum(PreflashDrumUnitParams(
        ...     assay=Assay([0, 30, 70, 100], [320., 480., 640., 900.], sg=0.85),
        ...     drum=PreflashDrumParams(P=2.5e5)))
        >>> vapor, liquid, water = op(op.feed(100.0, T=480.0, P=12e5))
    """

    symbol = "PF"
    numerical_method = "Newton on the drum temperature (and pressure); implicit-function gradients"
    parameter_units = {"cut_points": "K"}
    outlet_names = ("vapor", "liquid", "water")

    def __init__(self, params: PreflashDrumUnitParams):
        self.params = params
        self.crude, self.thermo = _thermo(params.assay, params.cut_points, params.method)

    def feed(self, rate, T, P, water=0.0):
        """A crude: ``rate`` mol/s of hydrocarbons, ``water`` BS&W."""
        s = self.crude.stream(rate, T=T, P=P, basis="mole")
        f, _ = _flows(s, self.thermo)
        return _stream(self.thermo, f, _water_moles(self.thermo, f, water), jnp.asarray(T, float), jnp.asarray(P, float))

    def solve(self, feed: dict) -> dict:
        """The outlets and ``T``, ``P``, ``vapor_fraction``."""
        th, d = self.thermo, self.params.drum
        f, fw = _flows(feed, th)
        vf_mode = d.vapor_fraction is not None
        args = {"th": th, "f": f, "fw": fw, "T_in": jnp.asarray(feed["T"], float),
                "P_in": jnp.asarray(feed["P"], float), "P": jnp.asarray(d.P, float),
                "duty": jnp.asarray(d.duty, float)}
        if vf_mode:
            args["vf"] = jnp.asarray(d.vapor_fraction, float)

        def unpack(z, a):
            return z[0], (jnp.exp(z[1] / _LNP) if vf_mode else a["P"])

        def r(z, a):
            T, P = unpack(z, a)
            th_ = a["th"]
            H_in = crude_enthalpy(th_, a["f"], a["fw"], a["T_in"], a["P_in"]) + a["duty"]
            out = [(H_in - crude_enthalpy(th_, a["f"], a["fw"], T, P)) / (400.0 * jnp.sum(a["f"]))]
            if vf_mode:
                out.append(jnp.sum(crude_split(th_, a["f"], a["fw"], T, P)["vapor"]) / jnp.sum(a["f"]) - a["vf"])
            return jnp.stack(out)

        z0 = [args["T_in"] - 5.0] + ([_LNP * jnp.log(args["P"])] if vf_mode else [])
        z, _ = _solve_1d(r, jnp.stack(z0), args)
        T, P = unpack(z, args)
        sp = crude_split(th, f, fw, T, P)
        return {
            "vapor": _stream(th, sp["vapor"], sp["water_vapor"], T, P),
            "liquid": _stream(th, sp["liquid"], 0.0, T, P),
            "water": _stream(th, jnp.zeros_like(f), sp["water_liquid"], T, P),
            "T": T, "P": P, "vapor_fraction": jnp.sum(sp["vapor"]) / jnp.sum(f),
        }

    def __call__(self, feed: Stream) -> tuple[Stream, Stream, Stream]:
        out = self.solve(feed)
        return out["vapor"], out["liquid"], out["water"]


# =============================================================================
# The crude unit with its preheat train
# =============================================================================


@dataclass
class CrudeUnitWithPreheatParams(ParamsMixin):
    """Parameters for :class:`CrudeUnitWithPreheat`.

    Attributes:
        assay: The crude's assay; it fixes the pseudo-components.
        column: The atmospheric column. Each pumparound the train cools
            needs a ``pa_return_T`` spec, whose value is only a starting guess.
        train: The preheat train, desalter and preflash drum.
        cut_points: Interior cut boundaries (K); default from the assay.
        method: Critical-property correlation for the pseudo-components.
    """

    assay: Assay
    column: CrudeColumnParams
    train: PreheatTrainParams
    cut_points: tuple | None = None
    method: str | None = None

    def __post_init__(self):
        if self.cut_points is not None:
            self.cut_points = tuple(self.cut_points)


class CrudeUnitWithPreheat:
    """Crude unit from the tank: preheat train, desalter, preflash drum, furnace, column.

    The inlet is the tank crude (``F_<component>`` and ``T``, the tank
    temperature; its water is the train's ``tank_water`` BS&W, not the
    stream's). The outlets are the column's products, as
    :class:`~difflow_refinery.unit.CrudeDistillationUnit` names them, then
    ``"brine"`` with a desalter and ``"drum_water"`` with a preflash drum.
    Products the train cools leave at the train's outlet temperature.

    Key equations:
        Tear on the pumparound return temperatures, the drum temperature and
        the furnace inlet temperature, closed by Newton
        Exchangers: Q = UA F LMTD with UA = A / (1/U + R_f)
        Furnace: Q_f = H_feed(T_coil, P_flash) - H_feed(T_furnace_in, P)

    Assumptions:
        The column's: equilibrium stages, Raoult with Lee-Kesler vapour pressures
        Pumparounds cooled in the train have no trim cooler
        No pressure drop across exchangers; pumps add no enthalpy

    Example:
        >>> # see examples/37_crude_preheat_train.ipynb
    """

    symbol = "PHT"
    numerical_method = ("Newton on a tear of the pumparound returns, drum and furnace inlet "
                        "temperatures around damped-Newton column and train solves; "
                        "implicit-function gradients at both levels")
    parameter_units = {"cut_points": "K"}

    def __init__(self, params: CrudeUnitWithPreheatParams):
        self.params = params
        self.unit = PreheatedCrudeUnit(params.assay, params.column, params.train,
                                       params.cut_points, params.method)
        self.last_result = None

    @property
    def outlet_names(self) -> tuple[str, ...]:
        p = self.unit.column_params
        names = (p.distillate_name, *(s.name for s in p.side_products), "residue", "water")
        names += ("offgas",) if p.condenser == "partial" else ()
        names += ("brine",) if self.params.train.desalter is not None else ()
        names += ("drum_water",) if self.params.train.drum is not None else ()
        return names

    def feed(self, rate, T, basis="bpd") -> dict:
        """A tank crude stream at ``rate`` and tank temperature ``T``."""
        f = self.unit.tank_flows(rate, basis)
        return _stream(self.unit.thermo, f, 0.0, jnp.asarray(T, float), jnp.asarray(1e5))

    def solve(self, feed: dict):
        """The full :class:`~difflow_refinery.preheat.unit.PreheatedUnitResult`."""
        f, _ = _flows(feed, self.unit.thermo)
        return self.unit.solve(jnp.sum(f), feed["T"], basis="mole", flows=f)

    def __call__(self, feed: Stream) -> tuple:
        res = self.solve(feed)
        self.last_result = res
        th = self.unit.thermo
        out = []
        cooled = set(self.unit.train.sources)
        for name in self.outlet_names:
            if name == "brine":
                d = res.train.desalter
                out.append(_stream(th, jnp.zeros(th.n_components), d["brine"], d["T"], res.feed["P"]))
            elif name == "drum_water":
                d = res.train.drum
                out.append(_stream(th, jnp.zeros(th.n_components), d["free_water"], d["T"], d["P"]))
            else:
                s = dict(res.column.products[name])
                if name in cooled:
                    s["T"] = res.train.hot_outlet_T[name]
                out.append(s)
        return tuple(out)


__all__ = ["CrudeUnitWithPreheat", "CrudeUnitWithPreheatParams", "Desalter", "DesalterUnitParams",
           "PreflashDrum", "PreflashDrumUnitParams"]
