"""Jet and VGO hydrotreating services (#536), checked by hand.

Run from the repository root; each part compiles the hydrotreater for a few
minutes::

    PYTHONPATH=src:. python tests/reference/hdt_services_check.py jet
    PYTHONPATH=src:. python tests/reference/hdt_services_check.py vgo

``jet``: a 150-250 C kerosene cut through `jet_params()`. It prints the
element balances, the jet properties and spec margins, and, with the Du et
al. data present, the AD sensitivities of product and gas yields with
respect to feed SG and sulfur next to ``Upd2``'s published ``delta / Del``.

``vgo``: a 370-560 C wax oil through `vgo_params()`, then CCR and metals
removal, then the product as an FCC feed, solved by the FCC. It prints the
balances, HDS/HDN, CCR and Ni+V in and out, and the FCC's convergence and
yields.

Not reproduced: published kerosene HDT and VGO HDS/HDN data (#536 asks for
them; none were at hand).
"""

import sys
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np

import difflow_refinery as dr
from difflow_refinery.assay import default_cut_points
from difflow_refinery.hydrotreating import Hydrotreater
from difflow_refinery.hydrotreating.feed import select_cuts, straight_run_cut
from difflow_refinery.hydrotreating.services import (fcc_feed_from_hydrotreater, heavy_ends_removal,
                                                     jet_margins, jet_params, jet_properties,
                                                     vgo_params)
from difflow_refinery.hydroprocessing.reactor import ReactorOptions

jax.config.update("jax_enable_x64", True)
warnings.simplefilter("ignore")
T0 = time.time()
TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_C = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]
LE = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def assay(sg, s, **kw):
    return dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_C]), sg=sg, light_ends=LE,
                    heavy_end=dr.HeavyEnd(), sulfur_wt=s, nitrogen_wppm=1500.0, **kw)


CPS = tuple(float(c) for c in default_cut_points(assay(0.86, 1.8)))


def feed_of(c, lo, hi, **kw):
    ch = dr.characterize(assay(c[0], c[1], **kw), cut_points=CPS, composition=True)
    cuts = select_cuts(dr.characterize(assay(0.86, 1.8), cut_points=CPS), lo, hi)
    return ch, straight_run_cut(ch, rate=50.0, cuts=cuts), cuts


def feed_props(ch, f, cuts):
    idx = jnp.asarray([ch.names.index(n) for n in cuts])
    mass = jnp.asarray([f[f"F_{n}"] for n in cuts]) * jnp.asarray(ch.component_MW)[idx]
    sg = jnp.asarray(ch.component_SG)[idx]
    S = jnp.asarray(ch.sulfur)[idx]
    return jnp.stack([mass.sum() / (mass / sg).sum(), 100 * (mass * S).sum() / mass.sum()])


def jet():
    lo, hi = 150 + 273.15, 250 + 273.15
    c = jnp.asarray([0.86, 1.8])
    ch, f, cuts = feed_of(c, lo, hi)
    p = jet_params(reactor=ReactorOptions(adjoint="forward"))
    unit = Hydrotreater(ch, f, p)
    r = unit.solve(f)
    o = r.outputs
    log("converged", bool(r.converged), "balances", {k: f"{float(v):.1e}" for k, v in r.balances.items()})
    jp = jet_properties(o)
    log("jet", {k: round(float(v), 2) for k, v in jp.items()})
    log("margins", {k: round(float(v), 2) for k, v in jet_margins(jp).items()})
    log("feed S wppm", float(o["feed.S_wppm"]), "-> product", float(o["product.S_wppm"]))

    from difflow_refinery.benchmarks import du2025
    case = du2025.find_case(1)
    if case is None:
        log("Du et al. data not fetched: Upd2 comparison skipped")
        return
    from difflow_refinery.benchmarks.du2025.calibration import unit_base
    b = unit_base(du2025.load_case(case), "Upd2")
    fp = lambda cc: feed_props(*feed_of(cc, lo, hi))  # noqa: E731
    # move to the benchmark's base feed: crude (sg, S) whose cut has B's SG and S
    target = jnp.asarray([b.props["Q168"], b.props["Q0"]])
    for _ in range(8):
        c = c - jnp.linalg.solve(jax.jacfwd(fp)(c), fp(c) - target)

    def yields(cc):
        ch2, f2, _ = feed_of(cc, lo, hi)
        oo = unit.solve(f2, char=ch2, warn=False).outputs
        gas = 1.0 + oo["h2.chemical_wt"] / 100.0 - oo["product.yield"] - oo["naphtha.yield"]
        return jnp.stack([gas, oo["product.yield"] + oo["naphtha.yield"]])

    Jy = np.asarray(jax.jacfwd(yields)(c))
    Jp = np.asarray(jax.jacfwd(fp)(c))
    ad = Jy @ np.linalg.inv(Jp)
    pub = np.array([[b.delta[q][b.outlets.index(s)] for q in ("Q168", "Q0")] for s in ("s215", "s104")])
    log("Upd2 base feed (B)", b.props, "our feed (SG, S)", np.round(np.asarray(fp(c)), 4),
        "crude (sg, S)", np.round(np.asarray(c), 4))
    log("d yield / d (SG, S wt%), oil basis: gas, jet")
    log("  ours     ", np.round(ad, 5).tolist())
    log("  published", np.round(pub, 5).tolist())


def vgo():
    lo, hi = 370 + 273.15, 560 + 273.15
    ch, f, cuts = feed_of(jnp.asarray([0.86, 1.8]), lo, hi, ccr_wt=2.0, nickel_vanadium_wppm=40.0)
    unit = Hydrotreater(ch, f, vgo_params())
    r = unit.solve(f)
    o = r.outputs
    log("converged", bool(r.converged), "balances", {k: f"{float(v):.1e}" for k, v in r.balances.items()})
    log("HDS", float(o["hds.conversion"]), "HDN", float(o["hdn.conversion"]), "WABT", float(o["wabt"]))
    idx = jnp.asarray([ch.names.index(n) for n in cuts])
    mass = jnp.asarray([f[f"F_{n}"] for n in cuts]) * jnp.asarray(ch.component_MW)[idx]
    ccr_in = float(100 * (mass * jnp.asarray(ch.ccr)[idx]).sum() / mass.sum())
    niv_in = float(1e6 * (mass * jnp.asarray(ch.nickel_vanadium)[idx]).sum() / mass.sum())
    he = heavy_ends_removal(ccr_in, niv_in, o["wabt"], vgo_params().lhsv)
    log("CCR", ccr_in, "->", float(he["ccr_wt"]), " Ni+V", niv_in, "->", float(he["niv_wppm"]))
    from difflow_refinery.fcc import FCCParams, FCCUnit
    ff = fcc_feed_from_hydrotreater(o, ccr_wt=he["ccr_wt"], niv_wppm=he["niv_wppm"])
    log("FCC feed", {k: round(float(getattr(ff, k)), 5) for k in ("SG", "Tb", "Kw", "MW", "sulfur", "nitrogen", "ccr")})
    res = FCCUnit(FCCParams()).solve(ff)
    log("FCC converged", bool(res["converged"]), {k: round(float(v), 4) for k, v in res["outputs"].items()
                                                  if k in ("conversion", "yield.gasoline", "yield.coke")})


if __name__ == "__main__":
    {"jet": jet, "vgo": vgo}[sys.argv[1]]()
