"""Product steam stripper and overhead drum, on the vacuum unit's stage-network column.

The separator liquid still holds dissolved H2, H2S, NH3 and light ends, and
the light cuts the reactor's cracking leak made. A short steam-stripped
column takes them overhead; its bottoms is the treated product. The column
is :class:`~difflow_refinery.vacuum.column.StageColumn` -- the same MESH
machinery as the vacuum unit (issue #306: "no new column code") -- with a
layout of its own:

* ``n_stages`` equilibrium stages, the feed (heated to ``feed_T`` in the
  column's "furnace" slot, flashed at the top pressure) entering the top one;
* liquid down through every stage; the bottoms leaves the last;
* stripping steam injected under the last stage; overhead vapour from the
  first. No condenser and no reflux: the overhead goes to a drum.

Thermodynamics are the vacuum unit's (:class:`~difflow_refinery.vacuum.thermo.ColumnThermo`):
Raoult with Maxwell-Bonnell vapour pressures, steam as a non-condensing
vapour. Real gas species (H2, H2S, NH3, C1-C6, water) are not column
components: as in the vacuum column, they leave with the overhead unchanged.
That is the stripper's job done perfectly, and it neglects their effect on
the hydrocarbons' partial pressure and heat balance -- right for the few
mole per cent a separator liquid carries, a known simplification. Cut
attributes leave in proportion to their cut (``bottoms_attr = feed_attr *
bottoms_i / feed_i``), so every element balance closes exactly; the
overhead is the feed less the bottoms.

The overhead drum (:func:`overhead_drum`) is a Peng-Robinson flash at the
drum temperature and pressure: vapour is the sour off-gas, liquid the wild
naphtha, and all water (steam condensate) decants as sour water.

Specs: ``feed_T`` (the stripper feed heater), steam rate and temperature,
top pressure and per-stage pressure drop. A :class:`~difflow_refinery.vacuum.column.StageSpec`
may trade ``"furnace.T"`` (the feed temperature) for a target on any column
output (e.g. ``"bottoms.T05"``, the product's 5 % point, i.e. its flash
point handle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import Array

from difflow_refinery import correlations as corr
from difflow_refinery.hydroprocessing.layout import Flows, Layout
from difflow_refinery.hydroprocessing.separator import flash_components, pr_flash, split_by_vapor_fraction
from difflow_refinery.hydroprocessing.thermo import Components
from difflow_refinery.vacuum.assay import PseudoComponents
from difflow_refinery.vacuum.column import ColumnLayout, Route, StageColumn, StageSpec
from difflow_refinery.vacuum.thermo import MW_WATER

jax.config.update("jax_enable_x64", True)

KNOBS = ("furnace.T", "steam.rate", "steam.T", "top.P", "stage.dP", "coil_steam.rate")


def _layout(n: int) -> ColumnLayout:
    routes = [Route(f"stage{j}.down", j, j + 1, "reference") for j in range(n - 1)]
    routes.append(Route("bottoms", n - 1, None, "reference"))

    def init_guess(col, ctx):
        k = ctx["knobs"]
        lF, vF = ctx["lF"], ctx["vF"]
        T0 = k["furnace.T"] - 4.0 * jnp.arange(n)
        S = k["steam.rate"] / MW_WATER * 1000.0
        L0 = jnp.full(n, jnp.sum(lF))
        V0 = jnp.full(n, jnp.maximum(jnp.sum(vF), 0.02 * jnp.sum(lF))) + 0.5 * S
        return T0, L0, V0, {}

    def extra(ctx, out):
        return {"bottoms.T": ctx["T"][n - 1], "bottoms.yield": out["bottoms.rate"] / out["feed.rate"]}

    return ColumnLayout(
        n_stages=n, feed_stage=0, routes=tuple(routes),
        pressures=lambda k: k["top.P"] + k["stage.dP"] * jnp.arange(n),
        furnace_pressure=lambda k: k["top.P"],
        steam_injection=lambda k: jnp.zeros(n).at[n - 1].set(k["steam.rate"] / MW_WATER * 1000.0),
        coil_steam=lambda k: k["coil_steam.rate"] / MW_WATER * 1000.0,
        knob_scales=(("furnace.T", 100.0),), knob_caps=(("furnace.T", 20.0),),
        init_guess=init_guess, extra_outputs=extra,
    )


@lru_cache(maxsize=16)
def _column(n_stages: int, n_comp: int, spec_key: tuple, max_iter: int, tol: float) -> StageColumn:
    specs = tuple(StageSpec(o, 0.0, r, s) for (o, r, s) in spec_key)
    return StageColumn(_layout(n_stages), n_comp, KNOBS, specs, max_iter=max_iter, tol=tol)


@dataclass(frozen=True)
class StripperSpec:
    """Stripper settings (static where they shape the column).

    Attributes:
        n_stages: Equilibrium stages.
        specs: :class:`StageSpec` s (each frees a knob for a target).
        max_iter, tol: Newton controls.
    """

    n_stages: int = 6
    specs: tuple = ()
    max_iter: int = 80
    tol: float = 1e-10


def cut_pseudo_components(layout: Layout, flows: Flows, Tb, SG, MW_default, Tc, Pc, omega, T_lo, T_hi):
    """The stripper's property table: each cut's own MW (from its atoms) where it has flow."""
    MW = jnp.where(flows.cut > 1e-20 * jnp.sum(flows.cut), flows.cut_mw(layout), MW_default)
    z = jnp.zeros(layout.n_cut)
    return PseudoComponents(Tb=Tb, SG=SG, MW=MW, Kw=corr.watson_k(Tb, SG), Tc=Tc, Pc=Pc, omega=omega,
                            T_lo=T_lo, T_hi=T_hi, sulfur=z, nitrogen=z, ccr=z, nickel_vanadium=z,
                            asphaltenes=z, names=tuple(layout.cuts))


@dataclass(frozen=True)
class StripperResult:
    """Attributes: overhead, bottoms (Flows), outputs (column outputs), converged, iterations, residual."""

    overhead: Flows
    bottoms: Flows
    outputs: dict
    converged: Array
    iterations: Array
    residual: Array


jax.tree_util.register_dataclass(StripperResult, data_fields=["overhead", "bottoms", "outputs", "converged",
                                                              "iterations", "residual"], meta_fields=[])


def strip(feed: Flows, layout: Layout, comps: PseudoComponents, feed_T, P_top, dP_stage,
          steam_rate, steam_T, spec: StripperSpec = StripperSpec(), targets: dict | None = None,
          feed_inlet_T=None, feed_steam_fraction=0.1) -> StripperResult:
    """Run the stripper on ``feed`` (separator liquid).

    Args:
        feed: Liquid feed flows.
        layout: Its layout.
        comps: :func:`cut_pseudo_components` of the feed.
        feed_T: Feed temperature after the feed heater (K) -- the "furnace" knob.
        P_top, dP_stage: Top pressure and pressure rise per stage (Pa).
        steam_rate: Stripping steam (kg/s); steam_T its temperature (K).
        spec: :class:`StripperSpec`.
        targets: ``{output: value}`` for ``spec.specs``.
        feed_inlet_T: Temperature of the feed before the heater (reported duty only).
        feed_steam_fraction: Share of the steam injected with the feed rather
            than under the bottom stage. A separator liquid with its gases
            taken out (they are not column components) is a subcooled liquid
            at the stripper's pressure, and StageColumn's feed flash then has
            no vapour phase -- a singular Jacobian. Steam in the feed line
            gives the flash a vapour phase for the light cuts to enter; 10 %
            of the steam is enough and moves the product by little.
    """
    col = _column(spec.n_stages, layout.n_cut, tuple((s.output, s.replaces, s.scale) for s in spec.specs),
                  int(spec.max_iter), float(spec.tol))
    steam_rate = jnp.asarray(steam_rate, dtype=float)
    knobs = {"furnace.T": jnp.asarray(feed_T, dtype=float),
             "steam.rate": steam_rate * (1.0 - feed_steam_fraction),
             "coil_steam.rate": steam_rate * feed_steam_fraction,
             "steam.T": jnp.asarray(steam_T, dtype=float), "top.P": jnp.asarray(P_top, dtype=float),
             "stage.dP": jnp.asarray(dP_stage, dtype=float)}
    th = {"components": comps, "feed": feed.cut,
          "feed_T": jnp.asarray(feed_T if feed_inlet_T is None else feed_inlet_T, dtype=float),
          "knobs": knobs,
          "targets": {s.output: jnp.asarray((targets or {}).get(s.output, s.target), dtype=float)
                      for s in spec.specs},
          "eta": jnp.ones(spec.n_stages)}
    x, it, rn = col.solve(th)
    ctx = col.context(x, th)
    outs = col.outputs_from(ctx)
    m_b = col.product_mass(ctx)["bottoms"]
    mol_b = m_b / comps.MW * 1000.0
    frac = mol_b / jnp.maximum(col.feed_of(th), 1e-300)
    gas0 = jnp.zeros(layout.n_gas)
    bottoms = Flows(gas0, feed.cut * frac, feed.attr * frac[:, None])
    overhead = feed - bottoms
    if layout.has_gas("water"):
        w = layout.gas_index("water")
        overhead = Flows(overhead.gas.at[w].add(jnp.asarray(steam_rate) / MW_WATER * 1000.0),
                         overhead.cut, overhead.attr)
    return StripperResult(overhead=overhead, bottoms=bottoms, outputs=outs, converged=rn < spec.tol,
                          iterations=it, residual=rn)


def overhead_drum(overhead: Flows, layout: Layout, comps: Components, T, P):
    """PR flash of the stripper overhead: ``(off_gas, wild_naphtha, sour_water, FlashResult)``."""
    z = flash_components(overhead, layout, comps)
    fr = pr_flash(T, P, z, comps)
    vap, liq, water = split_by_vapor_fraction(overhead, layout, comps, fr.vapor_fraction_by_component)
    return vap, liq, water, fr


__all__ = ["StripperSpec", "StripperResult", "cut_pseudo_components", "strip", "overhead_drum", "KNOBS"]
