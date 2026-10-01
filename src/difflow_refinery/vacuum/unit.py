"""Vacuum distillation unit (VDU).

Atmospheric residue is heated in a furnace, flashes into the flash zone of
a column held at a few kPa, and the vapor is condensed in pumparound
sections into light and heavy vacuum gas oil (LVGO, HVGO). The liquid that
falls out of the wash zone is the slop (overflash); the flash-zone liquid
is steam-stripped into vacuum residue.

Layout (stage numbers from the top; each packed bed is ``n`` equilibrium
stages with a Murphree efficiency to fit)::

    overhead vapor (steam + light HC) -> ejectors
    +---------------------------+
    | LVGO pumparound section   |  n_lvgo stages; LVGO PA returns to the top
    |   LVGO draw (total draw)  |  -> LVGO product + LVGO PA
    | HVGO pumparound section   |  n_hvgo stages; HVGO PA returns to its top
    |   HVGO draw               |  -> HVGO product + HVGO PA + wash oil down
    | wash zone                 |  n_wash stages
    |   slop draw (total draw)  |  -> slop / overflash
    | flash zone  <- furnace    |  1 stage; entrainment carried up
    | stripping section         |  n_strip stages
    +---------------------------+  <- bottom steam
    vacuum residue

There is no condenser and no reflux drum: the top stage's vapor leaves as
overhead at the top-stage temperature, and the top temperature is held by
the LVGO pumparound (the default spec frees its duty).

Degrees of freedom. Knobs are fixed numbers; a :class:`~difflow_refinery.vacuum.column.StageSpec`
trades one for a target on any output. Defaults:

- ``top.T = 70 C``, replacing ``lvgo_pa.duty``
- ``overflash = 0.03`` (slop / feed, mass), replacing ``hvgo.rate``
- ``lvgo.T95 = 450 C``, replacing ``hvgo_pa.duty``
- pumparound circulation rates fixed (``lvgo_pa.rate``, ``hvgo_pa.rate``)

So with the defaults the furnace outlet temperature, the flash-zone
pressure and the stripping steam are the operating levers, and LVGO, HVGO,
slop and residue yields come out -- which is the trade-off a planner wants.
Swap ``StageSpec("hvgo.T95", ..., replaces="furnace.T")`` in to ask instead what
furnace temperature a given VGO end point costs.

Contaminants. Sulfur, nitrogen, CCR, Ni+V and asphaltenes are carried per
pseudocomponent and so follow the flows. The CCR and metals that reach HVGO
come from vaporized heavy cuts *and* from entrainment: a fraction
``entrainment`` of the flash-zone liquid is carried up with the vapor, of
which ``deentrainment`` is knocked out in the wash bed (and leaves with the
slop) and the rest reaches the HVGO draw. Both are parameters to fit.

Furnace outlet. ``furnace_T_max`` (default 415 C) is the cracking limit.
It is not imposed -- a planner needs to see the trade-off, not have it
hidden -- but reported as ``furnace.cracking_margin`` and warned on.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from difflow.params_mixin import ParamsMixin
from difflow.streams import Stream
from difflow_refinery.vacuum.assay import PseudoComponents, product_properties
from difflow_refinery.vacuum.column import ColumnLayout, Route, StageSpec, StageColumn
from difflow_refinery.vacuum.thermo import MW_WATER

C_TO_K = 273.15
MMHG = 133.322368


class CrackingWarning(UserWarning):
    """The furnace outlet is above the cracking limit."""


class VacuumConvergenceWarning(UserWarning):
    """The vacuum column did not converge."""


def default_vacuum_specs() -> Tuple[StageSpec, ...]:
    """The default spec set (see module docstring)."""
    return (
        StageSpec("top.T", 70.0 + C_TO_K, replaces="lvgo_pa.duty"),
        StageSpec("overflash", 0.03, replaces="hvgo.rate"),
        StageSpec("lvgo.T95", 450.0 + C_TO_K, replaces="hvgo_pa.duty"),
    )


@dataclass
class VacuumColumnParams(ParamsMixin):
    """Parameters for :class:`VacuumColumn`.

    Attributes:
        components: Pseudocomponent property table, from ``characterize``; the
            feed stream's species are its names.
        furnace_T: Furnace (coil) outlet temperature (K).
        furnace_T_max: Cracking limit on the furnace outlet temperature (K);
            reported as a margin, not enforced.
        transfer_line_dP: Furnace outlet pressure above the flash zone (Pa).
        flash_zone_P: Flash-zone pressure (Pa).
        top_P: Top-stage pressure (Pa); stages above the flash zone are
            linear between the two.
        stripping_dP: Pressure rise per stripping stage below the flash zone (Pa).
        steam_rate: Bottom stripping steam (kg/s); None means 0.5% of feed.
        steam_T: Stripping (and coil) steam temperature (K).
        coil_steam_rate: Velocity steam injected into the furnace coil (kg/s).
        lvgo_pa_rate: LVGO pumparound circulation (kg/s); None means 0.3 x feed.
        hvgo_pa_rate: HVGO pumparound circulation (kg/s); None means 0.8 x feed.
        lvgo_pa_duty: LVGO pumparound duty (W), used only if no spec frees it.
        hvgo_pa_duty: HVGO pumparound duty (W), used only if no spec frees it.
        hvgo_rate: HVGO product rate (kg/s), used only if no spec replaces it.
        entrainment: Fraction of flash-zone liquid carried up with the vapor.
        deentrainment: Fraction of the entrained liquid the wash bed removes.
        n_lvgo: Equilibrium stages in the LVGO bed (draw is the last).
        n_hvgo: Equilibrium stages in the HVGO bed (draw is the last).
        n_wash: Equilibrium stages in the wash bed (slop drawn from the last).
        n_strip: Stripping stages below the flash zone.
        lvgo_efficiency: Murphree vapor efficiency of the LVGO bed.
        hvgo_efficiency: Murphree vapor efficiency of the HVGO bed.
        wash_efficiency: Murphree vapor efficiency of the wash bed.
        strip_efficiency: Murphree vapor efficiency of the stripping stages.
        specs: Specs; each trades a knob or a draw rate for a target.
        max_iter: Newton iteration limit.
        tol: Convergence tolerance on the largest scaled residual.
    """

    components: PseudoComponents
    furnace_T: float = 400.0 + C_TO_K
    furnace_T_max: float = 415.0 + C_TO_K
    transfer_line_dP: float = 2000.0
    flash_zone_P: float = 30.0 * MMHG
    top_P: float = 10.0 * MMHG
    stripping_dP: float = 300.0
    steam_rate: Optional[float] = None
    steam_T: float = 300.0 + C_TO_K
    coil_steam_rate: float = 0.0
    lvgo_pa_rate: Optional[float] = None
    hvgo_pa_rate: Optional[float] = None
    lvgo_pa_duty: Optional[float] = None
    hvgo_pa_duty: Optional[float] = None
    hvgo_rate: Optional[float] = None
    entrainment: float = 0.01
    deentrainment: float = 0.9
    n_lvgo: int = 2
    n_hvgo: int = 2
    n_wash: int = 2
    n_strip: int = 2
    lvgo_efficiency: float = 1.0
    hvgo_efficiency: float = 1.0
    wash_efficiency: float = 1.0
    strip_efficiency: float = 1.0
    specs: tuple = field(default_factory=default_vacuum_specs)
    max_iter: int = 150
    tol: float = 1e-11

    def __post_init__(self):
        # a JSON round trip hands the specs back as a list
        self.specs = tuple(self.specs)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

KNOBS = (
    "furnace.T", "furnace.T_max", "transfer_line.dP", "flash_zone.P", "top.P",
    "stripping.dP", "steam.rate", "steam.T", "coil_steam.rate",
    "lvgo_pa.rate", "hvgo_pa.rate", "hvgo.rate",
    "lvgo_pa.duty", "hvgo_pa.duty", "entrainment", "deentrainment",
)


@dataclass(frozen=True)
class _Sections:
    n_lvgo: int
    n_hvgo: int
    n_wash: int
    n_strip: int

    @property
    def lvgo_draw(self):
        return self.n_lvgo - 1

    @property
    def hvgo_top(self):
        return self.n_lvgo

    @property
    def hvgo_draw(self):
        return self.n_lvgo + self.n_hvgo - 1

    @property
    def wash_top(self):
        return self.n_lvgo + self.n_hvgo

    @property
    def slop_draw(self):
        return self.n_lvgo + self.n_hvgo + self.n_wash - 1

    @property
    def flash_zone(self):
        return self.n_lvgo + self.n_hvgo + self.n_wash

    @property
    def n_stages(self):
        return self.flash_zone + 1 + self.n_strip


def _build_layout(sec: _Sections) -> ColumnLayout:
    N, fz = sec.n_stages, sec.flash_zone
    routes = []
    for j in range(N):
        if j == sec.lvgo_draw:
            routes += [Route("lvgo", j, None, "reference"),
                       Route("lvgo_pa", j, 0, "share", duty="lvgo_pa.duty")]
        elif j == sec.hvgo_draw:
            routes += [Route("wash_oil", j, j + 1, "reference"),
                       Route("hvgo", j, None, "share"),
                       Route("hvgo_pa", j, sec.hvgo_top, "share", duty="hvgo_pa.duty")]
        elif j == sec.slop_draw:
            routes += [Route("slop", j, None, "reference")]
        elif j == fz:
            routes += [
                Route("entrained", j, sec.slop_draw, "fixed",
                      fraction=lambda k: k["entrainment"] * k["deentrainment"]),
                Route("entrained_bypass", j, sec.hvgo_draw, "fixed",
                      fraction=lambda k: k["entrainment"] * (1.0 - k["deentrainment"])),
                Route("residue" if j == N - 1 else f"stage{j}.down", j,
                      None if j == N - 1 else j + 1, "reference"),
            ]
        elif j == N - 1:
            routes += [Route("residue", j, None, "reference")]
        else:
            routes += [Route(f"stage{j}.down", j, j + 1, "reference")]

    def pressures(k):
        frac = jnp.arange(fz + 1) / fz
        upper = k["top.P"] + (k["flash_zone.P"] - k["top.P"]) * frac
        lower = k["flash_zone.P"] + k["stripping.dP"] * jnp.arange(1, sec.n_strip + 1)
        return jnp.concatenate([upper, lower])

    def steam_injection(k):
        return jnp.zeros(N).at[N - 1].set(k["steam.rate"] / MW_WATER * 1000.0)

    def extra(ctx, out):
        feed = out["feed.rate"]
        return {
            "overflash": out["slop.rate"] / feed,
            "furnace.cracking_margin": ctx["knobs"]["furnace.T_max"] - ctx["T_fo"],
            "flash_zone.T": ctx["T"][fz],
            "flash_zone.P": ctx["P"][fz],
            "vgo.rate": out["lvgo.rate"] + out["hvgo.rate"],
            "vgo.yield": (out["lvgo.rate"] + out["hvgo.rate"]) / feed,
            "residue.yield": out["residue.rate"] / feed,
            "lvgo.yield": out["lvgo.rate"] / feed,
            "hvgo.yield": out["hvgo.rate"] / feed,
            "slop.yield": out["slop.rate"] / feed,
            "lvgo.draw_T": ctx["T"][sec.lvgo_draw],
            "hvgo.draw_T": ctx["T"][sec.hvgo_draw],
            "slop.draw_T": ctx["T"][sec.slop_draw],
            "residue.T": ctx["T"][N - 1],
        }

    return ColumnLayout(
        n_stages=N, feed_stage=fz, routes=tuple(routes),
        pressures=pressures,
        furnace_pressure=lambda k: k["flash_zone.P"] + k["transfer_line.dP"],
        steam_injection=steam_injection,
        coil_steam=lambda k: k["coil_steam.rate"] / MW_WATER * 1000.0,
        knob_scales=(("furnace.T", 100.0), ("lvgo_pa.duty", 1e7),
                     ("hvgo_pa.duty", 1e7), ("flash_zone.P", 1000.0),
                     ("top.P", 1000.0), ("steam.rate", 1.0),
                     ("coil_steam.rate", 1.0)),
        knob_caps=(("furnace.T", 20.0), ("flash_zone.P", 1000.0),
                   ("top.P", 500.0)),
        init_guess=lambda col, ctx: _init_guess(sec, col, ctx),
        extra_outputs=extra,
    )


def _init_guess(sec: _Sections, col: StageColumn, ctx):
    """Temperatures, total flows and draw logits to start Newton from.

    Product rates are guessed from the furnace flash (what vaporizes is
    what the beds above have to condense), and stage temperatures from a
    profile between the top and the flash zone that is steep near the top,
    where the LVGO pumparound quenches the vapor.
    """
    th, k = ctx["th"], ctx["knobs"]
    comps = ctx["comps"]
    MW = comps.MW
    N, fz = sec.n_stages, sec.flash_zone
    F = jnp.sum(th["feed"] * MW) / 1000.0
    vap_m = jnp.sum(ctx["vF"] * MW) / 1000.0
    MWv = jnp.sum(ctx["vF"] * MW) / jnp.sum(ctx["vF"])
    MWl = jnp.sum(ctx["lF"] * MW) / jnp.sum(ctx["lF"])
    targets = th["targets"]

    vap = jnp.clip(1.1 * vap_m, 0.1 * F, 0.9 * F)
    slop = (targets["overflash"] * F if "overflash" in targets
            else jnp.asarray(0.03) * F)
    rest = jnp.maximum(vap - slop, 0.05 * F)
    hvgo = 0.6 * rest
    lvgo = rest - hvgo
    wash = slop
    pa_l, pa_h = k["lvgo_pa.rate"], k["hvgo_pa.rate"]

    T_top = targets["top.T"] if "top.T" in targets else jnp.asarray(70.0 + C_TO_K)
    T_fz = k["furnace.T"] - 10.0
    frac = (jnp.arange(fz + 1) / fz) ** 0.7
    T_upper = T_top + (T_fz - T_top) * frac
    T_lower = T_fz - 5.0 * jnp.arange(1, sec.n_strip + 1)
    T0 = jnp.concatenate([T_upper, T_lower])

    Lm, Vm = [], []
    for j in range(N):
        if j < sec.n_lvgo:
            s = (j + 1) / sec.n_lvgo
            Lm.append(pa_l + lvgo * s)
            Vm.append(0.02 * lvgo + lvgo * s)
        elif j < sec.wash_top:
            s = (j - sec.n_lvgo + 1) / sec.n_hvgo
            Lm.append(pa_h + (hvgo + wash) * s)
            Vm.append(lvgo + (hvgo + wash) * s)
        elif j < fz:
            Lm.append(wash)
            Vm.append(rest + wash)
        elif j == fz:
            Lm.append(F - vap)
            Vm.append(vap)
        else:
            Lm.append(F - vap)
            Vm.append(0.05 * (F - vap))
    MWs = jnp.where(jnp.arange(N) < fz, MWv, MWl)
    L0 = jnp.stack(Lm) * 1000.0 / MWs
    V0 = jnp.stack(Vm) * 1000.0 / MWv
    z0 = {
        "lvgo_pa": jnp.log(pa_l / lvgo),
        "hvgo": jnp.log(hvgo / wash),
        "hvgo_pa": jnp.log(pa_h / wash),
    }
    return T0, L0, V0, z0


# ---------------------------------------------------------------------------
# The unit
# ---------------------------------------------------------------------------


@lru_cache(maxsize=32)
def _compiled(sec: _Sections, n_comp: int, spec_key: tuple, max_iter: int, tol: float):
    specs = tuple(StageSpec(o, 0.0, r, s) for (o, r, s) in spec_key)
    col = StageColumn(_build_layout(sec), n_comp, KNOBS, specs,
                      max_iter=max_iter, tol=tol)

    def run(th):
        x, it, rn = col.solve(th)
        ctx = col.context(x, th)
        outs = col.outputs_from(ctx)
        return x, it, rn, outs, col.product_mass(ctx), ctx

    def run_public(th):
        x, it, rn, outs, prods, ctx = run(th)
        L = jnp.sum(ctx["l"], 1)
        V = jnp.sum(ctx["v"], 1)
        profiles = {"T": ctx["T"], "P": ctx["P"], "L": L, "V": V,
                    "steam": ctx["S_up"], "x": ctx["l"] / L[:, None],
                    "residual": col.residual(x, th)}
        return dict(x=x, iterations=it, residual=rn, outputs=outs,
                    products=prods, profiles=profiles,
                    furnace_liquid=ctx["lF"], furnace_vapor=ctx["vF"])

    return col, jax.jit(run_public)


class VacuumColumn:
    """Vacuum distillation unit: atmospheric residue -> LVGO, HVGO, slop, residue.

    See the module docstring for the layout, the specs and the
    contaminant routing. Differentiable end to end -- with respect to the
    feed, the pseudocomponent properties (and so the assay), every knob and
    every spec target -- through implicit-function gradients of the
    converged MESH equations.

    Example:
        >>> from difflow_refinery.vacuum import (heavy_crude, characterize,
        ...     atmospheric_residue, VacuumColumn, VacuumColumnParams)
        >>> char = characterize(heavy_crude())
        >>> feed = atmospheric_residue(char, crude_rate_kg_s=100.0)
        >>> vdu = VacuumColumn(VacuumColumnParams(components=char.components))
        >>> overhead, lvgo, hvgo, slop, residue, info = vdu(feed)
        >>> bool(info["converged"])
        True
    """

    symbol = "VDU"
    equations = [
        r"\ln(l_{ij} + v_{ij}) = \ln \sum (\text{liquid routed in} + v_{i,j+1} + f_i\,\delta_{j,\mathrm{fz}})",
        r"y_{ij} = \eta_j K_{ij} x_{ij} + (1 - \eta_j)\,y_{i,j+1},\quad K_{ij} = p^\mathrm{sat}_i(T_j) / P_j",
        r"\sum_i (l_{ij} h^L_i + v_{ij} h^V_i) + S_j h^S = H_{\mathrm{in},j} - Q_{\mathrm{PA}}",
        r"h^V_i - h^L_i = R T^2\, d\ln p^\mathrm{sat}_i / dT",
    ]
    assumptions = [
        "Ideal gas and ideal liquid solution (K = psat/P) at 1-10 kPa",
        "Steam is vapor-only (it cannot condense at column pressure)",
        "Packed beds as equilibrium stages with a Murphree vapor efficiency",
        "Entrainment as a fixed fraction of flash-zone liquid",
    ]
    references = [
        "Naphtali, L. M.; Sandholm, D. P. (1971) AIChE J. 17, 148-153",
        "Maxwell, J. B.; Bonnell, L. S. (1957) Ind. Eng. Chem. 49, 1187-1196",
        "Twu, C. H. (1984) Fluid Phase Equilib. 16, 137-150",
        "Kesler, M. G.; Lee, B. I. (1976) Hydrocarbon Process. 55(3), 153-158",
        "Riazi, M. R. (2005) Characterization and Properties of Petroleum "
        "Fractions, ASTM MNL50",
        "Watkins, R. N. (1979) Petroleum Refinery Distillation, 2nd ed., Gulf",
    ]
    parameter_units = {
        "furnace_T": "K", "furnace_T_max": "K", "transfer_line_dP": "Pa",
        "flash_zone_P": "Pa", "top_P": "Pa", "stripping_dP": "Pa",
        "steam_rate": "kg/s", "steam_T": "K", "coil_steam_rate": "kg/s",
        "lvgo_pa_rate": "kg/s", "hvgo_pa_rate": "kg/s", "lvgo_pa_duty": "W",
        "hvgo_pa_duty": "W", "hvgo_rate": "kg/s",
        "entrainment": "-", "deentrainment": "-", "n_lvgo": "-",
        "n_hvgo": "-", "n_wash": "-", "n_strip": "-", "lvgo_efficiency": "-",
        "hvgo_efficiency": "-", "wash_efficiency": "-",
        "strip_efficiency": "-", "max_iter": "-", "tol": "-",
    }

    PRODUCTS = ("overhead", "lvgo", "hvgo", "slop", "residue")

    def __init__(self, params: VacuumColumnParams):
        self.params = params
        p = params
        if min(p.n_lvgo, p.n_hvgo, p.n_wash) < 1 or p.n_strip < 0:
            raise ValueError("n_lvgo, n_hvgo and n_wash must be >= 1, n_strip >= 0")
        if p.n_hvgo < 2:
            raise ValueError("n_hvgo must be >= 2: the HVGO pumparound returns to "
                             "the top of the bed and is drawn from its bottom")
        self.sections = _Sections(p.n_lvgo, p.n_hvgo, p.n_wash, p.n_strip)
        self.specs = tuple(p.specs)
        spec_key = tuple((s.output, s.replaces, s.scale) for s in self.specs)
        self.column, self._run = _compiled(
            self.sections, p.components.n, spec_key, int(p.max_iter), float(p.tol))
        replaced = {s.replaces for s in self.specs}
        for name, val in (("lvgo_pa.duty", p.lvgo_pa_duty),
                          ("hvgo_pa.duty", p.hvgo_pa_duty),
                          ("hvgo.rate", p.hvgo_rate)):
            if val is None and name not in replaced:
                raise ValueError(f"{name} has no value and no spec replaces it")

    def theta(self, feed: Stream):
        """The differentiable inputs of a solve, as a pytree."""
        p = self.params
        comps = p.components
        f = jnp.stack([jnp.asarray(feed[f"F_{n}"], dtype=float) for n in comps.names])
        F = jnp.sum(f * comps.MW) / 1000.0
        sec = self.sections

        def val(v, default):
            return jnp.asarray(default if v is None else v, dtype=float)

        knobs = {
            "furnace.T": val(p.furnace_T, 0), "furnace.T_max": val(p.furnace_T_max, 0),
            "transfer_line.dP": val(p.transfer_line_dP, 0),
            "flash_zone.P": val(p.flash_zone_P, 0), "top.P": val(p.top_P, 0),
            "stripping.dP": val(p.stripping_dP, 0),
            "steam.rate": val(p.steam_rate, 0.005 * F), "steam.T": val(p.steam_T, 0),
            "coil_steam.rate": val(p.coil_steam_rate, 0),
            "lvgo_pa.rate": val(p.lvgo_pa_rate, 0.3 * F),
            "hvgo_pa.rate": val(p.hvgo_pa_rate, 0.8 * F),
            "hvgo.rate": val(p.hvgo_rate, 0.3 * F),
            "lvgo_pa.duty": val(p.lvgo_pa_duty, 0.0),
            "hvgo_pa.duty": val(p.hvgo_pa_duty, 0.0),
            "entrainment": val(p.entrainment, 0), "deentrainment": val(p.deentrainment, 0),
        }
        eta = []
        for j in range(sec.n_stages):
            if j < sec.n_lvgo:
                eta.append(p.lvgo_efficiency)
            elif j < sec.wash_top:
                eta.append(p.hvgo_efficiency)
            elif j < sec.flash_zone:
                eta.append(p.wash_efficiency)
            elif j == sec.flash_zone:
                eta.append(1.0)
            else:
                eta.append(p.strip_efficiency)
        return {
            "components": comps,
            "feed": f,
            "feed_T": jnp.asarray(feed["T"], dtype=float),
            "knobs": knobs,
            "targets": {s.output: jnp.asarray(s.target, dtype=float) for s in self.specs},
            "eta": jnp.stack([jnp.asarray(e, dtype=float) for e in eta]),
        }

    def solve(self, feed: Stream) -> dict:
        """Solve and return the raw result dict (outputs, products, profiles)."""
        res = self._run(self.theta(feed))
        res["converged"] = res["residual"] < self.params.tol
        self._warn(res)
        return res

    def _warn(self, res):
        conv = _concrete(res["converged"])
        if conv is not None and not conv:
            warnings.warn(
                f"VacuumColumn did not converge: largest scaled residual "
                f"{float(res['residual']):.2e} after {int(res['iterations'])} "
                f"iterations", VacuumConvergenceWarning, stacklevel=3)
        margin = _concrete(res["outputs"]["furnace.cracking_margin"])
        if margin is not None and margin < 0:
            warnings.warn(
                f"furnace outlet is {-margin:.1f} K above the cracking limit "
                f"(furnace_T_max)", CrackingWarning, stacklevel=3)

    def __call__(self, feed: Stream
                 ) -> tuple[Stream, Stream, Stream, Stream, Stream, dict]:
        """Run the column.

        Returns:
            ``(overhead, lvgo, hvgo, slop, residue, info)``: five difflow
            streams (mol/s per pseudocomponent; the overhead also carries
            ``F_H2O`` and every feed species the column does not model,
            passed through unchanged) and an info dict with ``converged``, ``iterations``,
            ``residual``, every column output under ``outputs`` (rates in
            kg/s, temperatures in K, duties in W), product ``properties``
            (SG, API, S, N, CCR, Ni+V, TBP points) and stage ``profiles``.
        """
        res = self.solve(feed)
        comps = self.params.components
        outs = res["outputs"]
        P = res["profiles"]["P"]
        T = res["profiles"]["T"]
        sec = self.sections
        where = {"overhead": (0, T[0]), "lvgo": (sec.lvgo_draw, None),
                 "hvgo": (sec.hvgo_draw, None), "slop": (sec.slop_draw, None),
                 "residue": (sec.n_stages - 1, None)}
        streams, props = [], {}
        for name in self.PRODUCTS:
            m = res["products"][name]
            mol = m / comps.MW * 1000.0
            st = {f"F_{n}": mol[i] for i, n in enumerate(comps.names)}
            j, _ = where[name]
            st["T"] = T[j]
            st["P"] = P[j]
            st["phase"] = "vapor" if name == "overhead" else "liquid"
            if name == "overhead":
                st["F_H2O"] = outs["steam.rate"] / MW_WATER * 1000.0
                # Species the column does not model -- light ends and water
                # a crude unit's residue carries when the VDU is fed from it
                # in a Flowsheet -- leave with the overhead, unchanged. At
                # 1-10 kPa and a 650+ K furnace nothing lighter than the first
                # vacuum pseudocomponent can stay liquid, so this is where
                # they would go; passing them through rather than dropping
                # them is what lets the flowsheet's mass balance close. They
                # take no part in the MESH equations (their heat and their
                # effect on the vapor partial pressures are neglected), which
                # is right for the traces a stripped residue carries and
                # wrong for a feed that is mostly light.
                for key in _passthrough_keys(feed, comps.names):
                    st[key] = st.get(key, 0.0) + feed[key]
            streams.append(st)
            props[name] = product_properties(comps, m)
        info = {
            "converged": res["converged"],
            "iterations": res["iterations"],
            "residual": res["residual"],
            "outputs": outs,
            "properties": props,
            "profiles": res["profiles"],
            "furnace": {"T": outs["furnace.T"], "duty": outs["furnace.duty"],
                        "vapor_fraction": outs["furnace.vapor_fraction"],
                        "cracking_margin": outs["furnace.cracking_margin"]},
        }
        return (*streams, info)


def _passthrough_keys(feed, names):
    """The ``F_*`` keys of ``feed`` that are not column components."""
    own = {f"F_{n}" for n in names}
    keys = feed.keys() if hasattr(feed, "keys") else ()
    return [k for k in keys if isinstance(k, str) and k.startswith("F_")
            and k not in own]


def _concrete(v):
    """A Python value, or None under tracing."""
    try:
        return v.item() if hasattr(v, "item") else v
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError):
        return None
    except Exception:
        return None
