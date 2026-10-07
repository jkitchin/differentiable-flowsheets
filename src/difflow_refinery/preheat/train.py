"""The preheat train: exchangers, desalter and preflash drum on the crude's way to the furnace.

The crude leaves the tank cold and wet, picks up heat from the column's
products and pumparounds in a network of shell-and-tube exchangers, is
washed in the desalter, partly flashed in the preflash drum, pumped up again
and reaches the furnace. :class:`PreheatTrain` solves that network for given
hot streams; :class:`~difflow_refinery.preheat.unit.PreheatedCrudeUnit` closes
it against the column that supplies them.

The model, equation by equation (all solved simultaneously by damped Newton):

* **Exchanger.** ``Q = H_c(T_c,out) - H_c(T_c,in)`` (the crude's enthalpy,
  three-phase, :func:`~difflow_refinery.preheat.flash.crude_enthalpy`),
  ``Q = (1 - b) [H_h(T_h,in) - H_h(T_h,out)]`` (the hot side held liquid, as
  the column counts its products), ``Q = UA F LMTD``, and the hot outlet
  mixed with the bypassed fraction ``b``: ``H_h(T_mix) = (1 - b) H_h(T_h,out)
  + b H_h(T_h,in)``. ``UA = A / (1/U_clean + R_f)``. ``F = 1`` for a pure
  counter-current exchanger, the 1-2N shell-and-tube correction otherwise.
* **Desalter.** Wash water at ``T_wash`` is mixed in, the brine drains at the
  desalter temperature, and what is left is the outlet BS&W. Adiabatic:
  ``H_crude(T_in) + H_wash = H_crude(T_d) + H_brine(T_d)``.
* **Preflash drum.** An adiabatic three-phase flash from the train pressure
  to the drum pressure (or, given a vapour fraction, the pressure that makes
  it). The vapour goes to the column, the free water is decanted, the
  hydrocarbon liquid is pumped on, dry.

Pressure drop and hydraulics are out of scope: the crude side sits at
``P_crude`` up to the drum and ``P_furnace`` after it, and pumps add no
enthalpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from difflow.params_mixin import ParamsMixin
from difflow.units.heat_exchanger import lmtd_correction_factor, log_mean_temperature_difference
from difflow_refinery.preheat._newton import damped_newton, implicit_step
from difflow_refinery.preheat.flash import crude_enthalpy, crude_split, liquid_enthalpy
from difflow_refinery.thermo import RHO_WATER_60F, WATER_MW, ColumnThermo

#: Temperature-like scale of an enthalpy residual: W over (crude mol/s x this).
_CP_SCALE = 400.0
#: The drum pressure is solved for as ``100 ln P``, so a Newton step limit in
#: kelvin is also a sensible one for it.
_LNP = 100.0

DESALTER = "desalter"
PREFLASH = "preflash"


@dataclass
class PreheatExchanger(ParamsMixin):
    """One crude preheat exchanger: crude on one side, a hot stream on the other.

    Which hot stream it is on is set by the :class:`HotStream` that lists it;
    where it sits on the crude's way by
    :attr:`PreheatTrainParams.crude_path`.

    Attributes:
        name: The exchanger's name (unique in the train).
        U: Clean overall heat-transfer coefficient (W/m^2/K).
        area: Heat-transfer area (m^2).
        Rf: Fouling resistance (m^2 K/W), lumped over both sides and referred
            to ``area``: ``UA = area / (1/U + Rf)``.
        bypass: Fraction of the hot stream that bypasses the exchanger and is
            mixed back in after it (0 to below 1).
        shells: ``None`` for a pure counter-current exchanger; ``n`` for a
            1-2n shell-and-tube exchanger, whose LMTD is reduced by the
            correction factor ``F``.
        h_crude: Crude-side film coefficient (W/m^2/K), used only to estimate
            the wall temperature a fouling model sees.
    """

    name: str
    U: float | Array
    area: float | Array
    Rf: float | Array = 0.0
    bypass: float | Array = 0.0
    shells: int | None = None
    h_crude: float | Array = 1000.0

    @property
    def UA(self) -> Array:
        """``area / (1/U + Rf)`` (W/K)."""
        return jnp.asarray(self.area) / (1.0 / jnp.asarray(self.U) + jnp.asarray(self.Rf))


@dataclass
class HotStream(ParamsMixin):
    """A column product or pumparound routed through the train.

    Attributes:
        source: The column product (``"residue"``, a side product) or
            pumparound it is.
        exchangers: The exchangers it passes through, hottest first.
    """

    source: str
    exchangers: tuple[str, ...]

    def __post_init__(self):
        self.exchangers = tuple(self.exchangers)


@dataclass
class DesalterParams(ParamsMixin):
    """The desalter: wash water in, brine out, salt removed.

    Water amounts are standard-volume fractions of the crude (BS&W).

    Attributes:
        wash_water: Wash water rate, as a fraction of the crude.
        T_wash: Wash water temperature (K).
        water_out: BS&W of the desalted crude: the water the desalter leaves
            in it. The rest leaves as brine.
        efficiency: Fraction of the salt removed.
        salt_in: Salt in the tank crude (pounds per thousand barrels).
        T_min: Bottom of the temperature window the desalter works in (K).
        T_max: Top of it (K). The window is reported as margins, not
            imposed: a train that misses it still solves, and says so.
    """

    wash_water: float | Array = 0.05
    T_wash: float | Array = 393.15
    water_out: float | Array = 0.002
    efficiency: float | Array = 0.95
    salt_in: float | Array = 20.0
    T_min: float | Array = 393.15
    T_max: float | Array = 423.15


@dataclass
class PreflashDrumParams(ParamsMixin):
    """The preflash drum.

    Attributes:
        P: Drum pressure (Pa); with ``vapor_fraction`` set, the starting
            guess for the pressure that delivers it.
        vapor_fraction: If given, the molar fraction of the hydrocarbons to
            flash, and the pressure is solved for.
        duty: Heat added in the drum (W); 0 for the adiabatic drum.
        vapor_stage: Where the vapour enters the column: a stage number, or 0
            for the overhead line (the condenser). ``None``: the stage above
            the flash zone.
    """

    P: float | Array = 3.0e5
    vapor_fraction: float | Array | None = None
    duty: float | Array = 0.0
    vapor_stage: int | None = None


@dataclass
class PreheatTrainParams(ParamsMixin):
    """A preheat train.

    Attributes:
        exchangers: The :class:`PreheatExchanger` s.
        hot_streams: The :class:`HotStream` s; together they must list every
            exchanger exactly once.
        crude_path: The order the crude meets things, tank to furnace:
            exchanger names, and ``"desalter"`` and ``"preflash"`` if there
            are those (the desalter first).
        desalter: :class:`DesalterParams`, needed if the path has one.
        drum: :class:`PreflashDrumParams`, needed if the path has one.
        tank_water: BS&W of the tank crude (standard-volume fraction). Free
            water a caller passes with the crude (``water=`` on
            :meth:`PreheatTrain.solve`, the inlet stream's ``F_water`` on
            :class:`~difflow_refinery.preheat.ops.CrudeUnitWithPreheat`) is
            added to it.
        P_crude: Crude-side pressure up to the drum (Pa), and to the furnace
            inlet without one.
        P_furnace: Crude-side pressure after the drum, the drum liquid's
            pump discharge, and so the furnace inlet pressure (Pa). Unused
            without a drum.
        max_iter: Damped-Newton iterations.
        tol: Converged when the scaled residual's infinity norm (kelvin-like)
            is below this.
    """

    exchangers: tuple[PreheatExchanger, ...]
    hot_streams: tuple[HotStream, ...]
    crude_path: tuple[str, ...]
    desalter: DesalterParams | None = None
    drum: PreflashDrumParams | None = None
    tank_water: float | Array = 0.002
    P_crude: float | Array = 15.0e5
    P_furnace: float | Array = 12.0e5
    max_iter: int = 60
    tol: float = 1e-9

    def __post_init__(self):
        self.exchangers = tuple(self.exchangers)
        self.hot_streams = tuple(self.hot_streams)
        self.crude_path = tuple(self.crude_path)

    def exchanger(self, name: str) -> PreheatExchanger:
        """The exchanger called ``name``."""
        for e in self.exchangers:
            if e.name == name:
                return e
        raise KeyError(name)

    def with_exchanger(self, name: str, **changes) -> "PreheatTrainParams":
        """A copy with one exchanger's fields changed (``Rf=``, ``area=``, ...)."""
        ex = tuple(e.update(**changes) if e.name == name else e for e in self.exchangers)
        if all(e.name != name for e in self.exchangers):
            raise KeyError(name)
        return self.update(exchangers=ex)


def _water_moles(th: ColumnThermo, f: Array, fraction) -> Array:
    """Water (mol/s) at a standard-volume fraction of the hydrocarbons ``f``."""
    return fraction * th.std_volume(f) * RHO_WATER_60F / (WATER_MW / 1000.0)


@dataclass(frozen=True)
class _Plan:
    """The train's structure as indices."""

    items: tuple  # (kind, name, slot) along the crude path; slot = first unknown
    hot_of: tuple  # (exchanger, hot stream index, slot of the T_mix upstream or -1)
    shells: tuple  # (exchanger, shells or None)
    n: int  # unknowns
    vf_mode: bool

    def hot(self, name):
        """``(hot stream index, slot of the upstream exchanger's T_mix or -1)``."""
        for e, k, up in self.hot_of:
            if e == name:
                return k, up
        raise KeyError(name)


def _plan(p: PreheatTrainParams) -> _Plan:
    names = [e.name for e in p.exchangers]
    if len(set(names)) != len(names):
        raise ValueError(f"exchanger names repeat: {names}")
    if DESALTER in names or PREFLASH in names:
        raise ValueError(f"{DESALTER!r} and {PREFLASH!r} are reserved names")
    path = list(p.crude_path)
    if len(set(path)) != len(path):
        raise ValueError(f"the crude path visits something twice: {path}")
    unknown = [s for s in path if s not in names and s not in (DESALTER, PREFLASH)]
    if unknown:
        raise ValueError(f"the crude path names {unknown}, which are not exchangers of the train")
    missing = [n for n in names if n not in path]
    if missing:
        raise ValueError(f"exchangers {missing} are not on the crude path")
    if DESALTER in path and p.desalter is None:
        raise ValueError("the crude path has a desalter but the train has no DesalterParams")
    if PREFLASH in path and p.drum is None:
        raise ValueError("the crude path has a preflash drum but the train has no PreflashDrumParams")
    if DESALTER in path and PREFLASH in path and path.index(DESALTER) > path.index(PREFLASH):
        raise ValueError("the desalter must come before the preflash drum")
    hot_of = {}
    for k, hs in enumerate(p.hot_streams):
        for j, e in enumerate(hs.exchangers):
            if e not in names:
                raise ValueError(f"hot stream {hs.source!r} lists {e!r}, not an exchanger of the train")
            if e in hot_of:
                raise ValueError(f"exchanger {e!r} is on two hot streams")
            hot_of[e] = (k, j)
    lone = [n for n in names if n not in hot_of]
    if lone:
        raise ValueError(f"exchangers {lone} are on no hot stream")
    vf_mode = PREFLASH in path and p.drum.vapor_fraction is not None
    items, slot = [], 0
    for s in path:
        if s == DESALTER:
            items.append(("desalter", s, slot))
            slot += 1
        elif s == PREFLASH:
            items.append(("drum", s, slot))
            slot += 2 if vf_mode else 1
        else:
            items.append(("hx", s, slot))
            slot += 3
    slot_of = {name: sl for kind, name, sl in items if kind == "hx"}
    hot = []
    for e, (k, j) in hot_of.items():
        up = p.hot_streams[k].exchangers[j - 1] if j > 0 else None
        hot.append((e, k, -1 if up is None else slot_of[up] + 2))
    return _Plan(items=tuple(items), hot_of=tuple(hot),
                 shells=tuple((e.name, e.shells) for e in p.exchangers), n=slot, vf_mode=vf_mode)


@dataclass(frozen=True)
class PreheatTrainResult:
    """A solved preheat train.

    Attributes:
        exchangers: Per exchanger (by name): ``Q`` (W), ``T_cold_in``,
            ``T_cold_out``, ``T_hot_in``, ``T_hot_out`` (leaving the
            exchanger), ``T_hot_mixed`` (after the bypass rejoins), ``LMTD``,
            ``F``, ``UA`` (W/K), ``approach`` (the smaller terminal
            difference, K; negative is a temperature cross).
        hot_outlet_T: Each hot stream's temperature leaving the train, by source.
        desalter: ``T``, ``wash_water``, ``brine``, ``water_out`` (mol/s),
            ``salt_out`` (PTB), ``margin_low`` and ``margin_high`` (K inside
            the window; negative is outside). Empty without a desalter.
        drum: ``T``, ``P``, ``T_in``, ``vapor`` (component flows),
            ``vapor_water``, ``liquid`` (component flows), ``free_water``
            (mol/s), ``vapor_fraction`` (of the hydrocarbons). Empty without one.
        T_out: Crude temperature at the furnace inlet (K).
        P_out: Its pressure (Pa).
        crude_out: Hydrocarbon flows at the furnace inlet (mol/s).
        water_out: Water in the crude at the furnace inlet (mol/s).
        recovered: Heat recovered, the sum of the exchanger duties (W).
        z: The solved unknowns (for a warm start).
        residual_norm: Scaled residual infinity norm.
        converged: Whether it is below the tolerance.
        iterations: Newton iterations.
    """

    exchangers: dict
    hot_outlet_T: dict
    desalter: dict
    drum: dict
    T_out: Array
    P_out: Array
    crude_out: Array
    water_out: Array
    recovered: Array
    z: Array
    residual_norm: Array
    converged: Array
    iterations: Array


jax.tree_util.register_dataclass(
    PreheatTrainResult,
    data_fields=["exchangers", "hot_outlet_T", "desalter", "drum", "T_out", "P_out", "crude_out",
                 "water_out", "recovered", "z", "residual_norm", "converged", "iterations"],
    meta_fields=[],
)


class PreheatTrain:
    """Solve a preheat train for given hot streams.

    Args:
        params: :class:`PreheatTrainParams`.
        thermo: The crude's :class:`~difflow_refinery.thermo.ColumnThermo`.

    :meth:`solve` takes the crude (hydrocarbon flows and tank temperature)
    and each hot stream's flows and inlet temperature. Differentiable with
    respect to all of them and every number in ``params``, by the implicit
    function theorem.
    """

    def __init__(self, params: PreheatTrainParams, thermo: ColumnThermo):
        self.params = params
        self.thermo = thermo
        self.plan = _plan(params)
        self._jit = None

    @property
    def sources(self) -> tuple[str, ...]:
        """The hot streams' sources, in order."""
        return tuple(h.source for h in self.params.hot_streams)

    # ------------------------------------------------------------------

    def _args(self, f, T_tank, hot: dict, th=None, params=None, water=0.0) -> dict:
        p = self.params if params is None else params
        th = self.thermo if th is None else th
        missing = [s for s in self.sources if s not in hot]
        if missing:
            raise ValueError(f"no inlet given for hot streams {missing}")
        f = jnp.asarray(f, dtype=float)
        a = {
            "thermo": th,
            "f": f,
            "T_tank": jnp.asarray(T_tank, dtype=float),
            # BS&W plus the water the crude arrives with (mol/s): replacing
            # the latter by the BS&W dropped it (audit, 2026-10).
            "fw": (_water_moles(th, f, jnp.asarray(p.tank_water, dtype=float))
                   + jnp.asarray(water, dtype=float)),
            "hot_f": [jnp.asarray(hot[s][0], dtype=float) for s in self.sources],
            "hot_T": [jnp.asarray(hot[s][1], dtype=float) for s in self.sources],
            "UA": {e.name: e.UA for e in p.exchangers},
            "bypass": {e.name: jnp.asarray(e.bypass, dtype=float) for e in p.exchangers},
            "P_crude": jnp.asarray(p.P_crude, dtype=float),
            "P_furnace": jnp.asarray(p.P_furnace, dtype=float),
        }
        if p.desalter is not None:
            d = p.desalter
            a["wash"] = _water_moles(th, f, jnp.asarray(d.wash_water, dtype=float))
            a["T_wash"] = jnp.asarray(d.T_wash, dtype=float)
            a["fw_desalted"] = _water_moles(th, f, jnp.asarray(d.water_out, dtype=float))
            a["salt_out"] = (1.0 - jnp.asarray(d.efficiency, dtype=float)) * jnp.asarray(d.salt_in, dtype=float)
            a["T_window"] = jnp.stack([jnp.asarray(d.T_min, dtype=float), jnp.asarray(d.T_max, dtype=float)])
        if p.drum is not None:
            a["P_drum"] = jnp.asarray(p.drum.P, dtype=float)
            a["drum_duty"] = jnp.asarray(p.drum.duty, dtype=float)
            if p.drum.vapor_fraction is not None:
                a["drum_vf"] = jnp.asarray(p.drum.vapor_fraction, dtype=float)
        return a

    def _walk(self, z, a):
        """Everything the unknowns imply, in path order, with the residuals.

        The crude-side enthalpies are gathered first and evaluated in one
        vectorised call: each is a three-phase flash, and written out one by
        one they multiply the compiled program (and its compile time) by the
        number of exchangers.
        """
        plan, th = self.plan, a["thermo"]
        shells = dict(plan.shells)
        Cs = jnp.sum(a["f"]) * _CP_SCALE
        pts = []

        def pt(f, fw, T, P):
            pts.append((f, jnp.asarray(fw, dtype=float), T, P))
            return len(pts) - 1

        # pass 1: the crude's state along the path, as points to evaluate
        f, fw, P, T = a["f"], a["fw"], a["P_crude"], a["T_tank"]
        here = pt(f, fw, T, P)
        idx, drum = {}, None
        for kind, name, s in plan.items:
            if kind == "hx":
                out_ = pt(f, fw, z[s], P)
                idx[name] = (here, out_, T)
                here, T = out_, z[s]
            elif kind == "desalter":
                fw_out = a["fw_desalted"]
                out_ = pt(f, fw_out, z[s], P)
                idx[name] = (here, out_, T, fw)
                here, fw, T = out_, fw_out, z[s]
            else:
                Td = z[s]
                P_d = jnp.exp(z[s + 1] / _LNP) if plan.vf_mode else a["P_drum"]
                out_ = pt(f, fw, Td, P_d)
                sp = crude_split(th, f, fw, Td, P_d)
                drum = (here, out_, T, Td, P_d, sp, f)
                f, fw, P, T = sp["liquid"], jnp.asarray(0.0), a["P_furnace"], Td
                here = pt(f, fw, T, P)
        H = jax.vmap(lambda f_, fw_, T_, P_: crude_enthalpy(th, f_, fw_, T_, P_))(
            *(jnp.stack(c) for c in zip(*pts)))

        # pass 2: the residuals
        res, ex, out = [], {}, {"desalter": {}, "drum": {}}
        for kind, name, s in plan.items:
            if kind == "hx":
                i_in, i_out, Tc_in = idx[name]
                Tc_out, Th_out, T_mix = z[s], z[s + 1], z[s + 2]
                k, up = plan.hot(name)
                hf = a["hot_f"][k]
                Th_in = a["hot_T"][k] if up < 0 else z[up]
                b = a["bypass"][name]
                Q = H[i_out] - H[i_in]
                Hh_in = liquid_enthalpy(th, hf, Th_in)
                Hh_out = liquid_enthalpy(th, hf, Th_out)
                dT1, dT2 = Th_in - Tc_out, Th_out - Tc_in
                lmtd = log_mean_temperature_difference(dT1, dT2)
                if shells[name]:
                    dTc = Tc_out - Tc_in
                    R = (Th_in - Th_out) / jnp.where(jnp.abs(dTc) > 1e-9, dTc, 1e-9)
                    F = lmtd_correction_factor(R, dTc / (Th_in - Tc_in), int(shells[name]))
                else:
                    F = jnp.asarray(1.0)
                Ch = jnp.sum(hf) * _CP_SCALE
                res += [(Q - (1.0 - b) * (Hh_in - Hh_out)) / Cs,
                        (Q - a["UA"][name] * F * lmtd) / Cs,
                        (liquid_enthalpy(th, hf, T_mix) - (1.0 - b) * Hh_out - b * Hh_in) / Ch]
                ex[name] = {"Q": Q, "T_cold_in": Tc_in, "T_cold_out": Tc_out, "T_hot_in": Th_in,
                            "T_hot_out": Th_out, "T_hot_mixed": T_mix, "LMTD": lmtd, "F": F,
                            "UA": a["UA"][name], "approach": jnp.minimum(dT1, dT2)}
            elif kind == "desalter":
                i_in, i_out, T_in, fw_in = idx[name]
                Td = z[s]
                fw_out = a["fw_desalted"]
                brine = fw_in + a["wash"] - fw_out
                res.append((H[i_in] + a["wash"] * th.water_h_liquid(a["T_wash"])
                            - H[i_out] - brine * th.water_h_liquid(Td)) / Cs)
                out["desalter"] = {
                    "T": Td, "T_in": T_in, "wash_water": a["wash"], "brine": brine, "water_out": fw_out,
                    "salt_out": a["salt_out"],
                    "margin_low": Td - a["T_window"][0], "margin_high": a["T_window"][1] - Td,
                }
            else:
                i_in, i_out, T_in, Td, P_d, sp, f_in = drum
                res.append((H[i_in] + a["drum_duty"] - H[i_out]) / Cs)
                vf = jnp.sum(sp["vapor"]) / jnp.sum(f_in)
                if plan.vf_mode:
                    res.append(vf - a["drum_vf"])
                out["drum"] = {"T": Td, "P": P_d, "T_in": T_in, "vapor": sp["vapor"],
                               "vapor_water": sp["water_vapor"], "liquid": sp["liquid"],
                               "free_water": sp["water_liquid"], "vapor_fraction": vf}
        hot_T = list(a["hot_T"])
        for k, hs in enumerate(self.params.hot_streams):
            hot_T[k] = ex[hs.exchangers[-1]]["T_hot_mixed"]
        out.update(exchangers=ex, hot_T=hot_T, T=T, P=P, f=f, fw=fw, H_tank=H[0])
        return jnp.stack(res), out

    def _residual(self, z, a):
        return self._walk(z, a)[0]

    def _initial_guess(self, a):
        """Sweeps of a constant-Cp, effectiveness-NTU version of the train.

        Each sweep walks the crude path with each hot stream's inlet taken
        from the sweep before, so the counter-flowing hot streams settle in a
        few dozen sweeps.
        """
        plan, th = self.plan, a["thermo"]
        f = a["f"]
        C_c0 = jax.grad(lambda t: crude_enthalpy(th, f, a["fw"], t, a["P_crude"]))(a["T_tank"] + 50.0)
        C_h = [jax.grad(lambda t, hf=hf: liquid_enthalpy(th, hf, t))(T0)
               for hf, T0 in zip(a["hot_f"], a["hot_T"])]
        C_w = 75.4  # liquid water, J/mol/K

        def sweep(_, z):
            T = a["T_tank"]
            C_c = C_c0
            zn = z
            for kind, name, s in plan.items:
                if kind == "hx":
                    k, up = plan.hot(name)
                    Th_in = a["hot_T"][k] if up < 0 else z[up]
                    b = a["bypass"][name]
                    Ch = (1.0 - b) * C_h[k]
                    Cmin, Cmax = jnp.minimum(Ch, C_c), jnp.maximum(Ch, C_c)
                    ntu = a["UA"][name] / Cmin
                    cr = Cmin / Cmax
                    e1 = jnp.exp(-ntu * (1.0 - cr))
                    eps = jnp.where(jnp.abs(1.0 - cr) < 1e-6, ntu / (1.0 + ntu),
                                    (1.0 - e1) / (1.0 - cr * e1))
                    Q = eps * Cmin * jnp.maximum(Th_in - T, 0.0)
                    Tc = T + Q / C_c
                    zn = zn.at[s].set(Tc).at[s + 1].set(Th_in - Q / Ch).at[s + 2].set(Th_in - Q / C_h[k])
                    T = Tc
                elif kind == "desalter":
                    Cw = C_w * a["wash"]
                    T = (C_c * T + Cw * a["T_wash"]) / (C_c + Cw)
                    zn = zn.at[s].set(T)
                else:
                    T = T - 3.0
                    zn = zn.at[s].set(T)
                    if plan.vf_mode:
                        zn = zn.at[s + 1].set(_LNP * jnp.log(a["P_drum"]))
            return zn

        z0 = jnp.zeros(plan.n)
        for kind, name, s in plan.items:  # hot inlets start at the hot stream's own inlet
            if kind == "hx":
                k, _ = plan.hot(name)
                z0 = z0.at[s + 2].set(a["hot_T"][k])
        return jax.lax.fori_loop(0, 40, sweep, z0)

    def _core(self, a, z0):
        frozen = jax.lax.stop_gradient(a)
        z_start = self._initial_guess(frozen) if z0 is None else jax.lax.stop_gradient(z0)
        z1, norm, iters = damped_newton(lambda z: self._residual(z, frozen), z_start,
                                        self.params.max_iter, self.params.tol, 50.0)
        z = implicit_step(self._residual, z1, a, frozen)
        return self._result(z, a, norm, iters, self.params.tol)

    # ------------------------------------------------------------------

    def solve(self, f, T_tank, hot: dict, thermo: ColumnThermo | None = None,
              params: PreheatTrainParams | None = None, z0=None,
              water=0.0) -> PreheatTrainResult:
        """Solve the train.

        Args:
            f: Hydrocarbon flows of the tank crude (mol/s), in the thermo's order.
            T_tank: Tank temperature (K).
            hot: ``{source: (flows, T_in)}`` for every hot stream.
            thermo, params: Replace the train's own. ``params`` must have the
                same structure (exchangers, hot streams, path); its numbers
                may be JAX tracers.
            z0: Starting unknowns (a previous result's ``z``).
            water: Free water arriving with the crude (mol/s), on top of
                the ``tank_water`` BS&W.
        """
        if params is not None and _plan(params) != self.plan:
            raise ValueError("params must have the same structure as the train's own")
        a = self._args(f, T_tank, hot, thermo, params, water)
        if self._jit is None:
            self._jit = jax.jit(self._core)
        return self._jit(a, z0)

    def _result(self, z, a, norm, iters, tol) -> PreheatTrainResult:
        _, w = self._walk(z, a)
        ex = w["exchangers"]
        return PreheatTrainResult(
            exchangers=ex,
            hot_outlet_T={s: w["hot_T"][k] for k, s in enumerate(self.sources)},
            desalter=w["desalter"], drum=w["drum"],
            T_out=w["T"], P_out=w["P"], crude_out=w["f"], water_out=jnp.asarray(w["fw"], dtype=float),
            recovered=sum((e["Q"] for e in ex.values()), jnp.asarray(0.0)),
            z=z, residual_norm=norm, converged=jnp.isfinite(norm) & (norm <= 100.0 * tol),
            iterations=iters,
        )


__all__ = ["DesalterParams", "HotStream", "PreflashDrumParams", "PreheatExchanger",
           "PreheatTrain", "PreheatTrainParams", "PreheatTrainResult"]
