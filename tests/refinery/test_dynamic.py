"""Smoke and gradient tests of the dynamic refinery units (difflow_refinery.dynamic).

Per the test policy these check the code, not the physics: the steady-limit,
balance-along-a-trajectory, trajectory-gradient and time-scale gates are in
``tests/reference/dynamic_refinery_checks.py`` (about 40 min). Here: one bed
of three cells on the docs' diesel layout and one three-cell reforming bed,
no steady solve and no long integration.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.dynamic import (
    BedGeometry, DynamicReformingReactor, DynamicTrickleBedReactor, QuasiSteadyHoldup, evaluate,
    holdup_outflow, signal_args, smoothstep)
from difflow_refinery.dynamic.hydroprocessing import flows_scale, holdup_from_outflow
from difflow_refinery.hydroprocessing.layout import Flows
from difflow_refinery.hydroprocessing.reactor import k_model_at
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
from difflow_refinery.hydrotreating.feed import select_cuts

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]


def test_signals_step_and_ramp():
    a = signal_args({"T": 600.0, "s": 1.0}, {"T": [(100.0, 10.0), (300.0, -4.0)]}, ramp=20.0)
    assert float(evaluate(0.0, a)["T"]) == 600.0
    assert float(evaluate(110.0, a)["T"]) == pytest.approx(605.0)
    assert float(evaluate(1e4, a)["T"]) == pytest.approx(606.0)
    assert float(evaluate(1e4, a)["s"]) == 1.0
    assert float(smoothstep(jnp.asarray(-1.0))) == 0.0 and float(smoothstep(jnp.asarray(2.0))) == 1.0
    g = jax.grad(lambda d: evaluate(110.0, signal_args({"T": 600.0}, {"T": [(100.0, d)]}, 20.0))["T"])(10.0)
    assert float(g) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="no base"):
        signal_args({"T": 1.0}, {"x": [(0.0, 1.0)]})


@pytest.fixture(scope="module")
def bed():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(a, composition=True)
    feed = straight_run_cut(char, rate=50.0, cuts=select_cuts(char, 230 + 273.15, 370 + 273.15))
    unit = Hydrotreater(char, feed, HydrotreaterParams())
    th = unit.theta(feed)
    comps, kin, lay = unit._comps(th), unit._kin_params(th["kin"]), unit.layout
    gas = Flows(jnp.zeros(lay.n_gas).at[lay.gas_index("hydrogen")].set(700.0),
                jnp.zeros(lay.n_cut), jnp.zeros((lay.n_cut, lay.n_attr)))
    inlet = th["oil"] + gas
    T_in = 613.15
    km, _ = k_model_at(inlet, lay, comps, T_in, 50e5)
    geo = BedGeometry(W=60000.0, tau_G=5.0, tau_L=25.0)
    n_cells = 3
    h0 = holdup_from_outflow(inlet, lay, comps, km, T_in, geo.tau_G, geo.tau_L)
    cells = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_cells)[None], h0)
    T = jnp.full((1, n_cells), T_in)
    sc = flows_scale(h0)
    s1 = jnp.concatenate([sc.ravel(), jnp.asarray([100.0])])
    scale = jnp.tile(s1, n_cells)
    rx = DynamicTrickleBedReactor(lay, unit.kinetics, kin, comps, [geo], 50e5, n_cells=n_cells, scale=scale)
    y0 = rx.pack(cells, T)
    return {"rx": rx, "y0": y0, "inlet": inlet.to_stream(lay, T_in, 50e5), "lay": lay, "comps": comps,
            "km": km, "geo": geo, "flows": inlet, "T_in": T_in}


def test_holdup_split_inverts(bed):
    lay, comps, km, geo = bed["lay"], bed["comps"], bed["km"], bed["geo"]
    F = bed["flows"]
    n = holdup_from_outflow(F, lay, comps, km, 640.0, geo.tau_G, geo.tau_L)
    back = holdup_outflow(n, lay, comps, km, 640.0, 1.0 / geo.tau_G, 1.0 / geo.tau_L)
    np.testing.assert_allclose(np.asarray(back.ravel()), np.asarray(F.ravel()), rtol=1e-9, atol=1e-12)


def test_trickle_bed_smoke_and_balance(bed):
    rx, y0, inp = bed["rx"], bed["y0"], {"feed": bed["inlet"]}
    f = jax.jit(lambda y: rx.derivatives(0.0, y, inp))
    dy = f(y0)
    assert dy.shape == y0.shape and bool(jnp.all(jnp.isfinite(dy)))
    out = Flows.from_stream(rx.outputs(0.0, y0, inp)["outlet"], bed["lay"])
    # accumulation of every element = in - out, to round-off (the RK4 segments conserve elements)
    lay = bed["lay"]
    held = lambda y: jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=(0, 1)), rx.unpack(y)[0]).elements(lay)
    acc = jax.jvp(held, (y0,), (dy,))[1]
    flux = bed["flows"].elements(lay) - out.elements(lay)
    through = float(jnp.max(jnp.abs(bed["flows"].elements(lay))))
    np.testing.assert_allclose(np.asarray(acc), np.asarray(flux), rtol=0, atol=1e-12 * through)
    # and a perturbed state (a real transient): the identity still holds
    y1 = y0 * (1.0 + 0.05 * jnp.sin(jnp.arange(y0.size)))
    dy1 = f(y1)
    out1 = Flows.from_stream(rx.outputs(0.0, y1, inp)["outlet"], lay)
    acc1 = jax.jvp(held, (y1,), (dy1,))[1]
    flux1 = bed["flows"].elements(lay) - out1.elements(lay)
    assert float(jnp.max(jnp.abs(flux1))) > 1e-6 * through
    np.testing.assert_allclose(np.asarray(acc1), np.asarray(flux1), rtol=0, atol=1e-12 * through)


def test_trickle_bed_gradient_finite(bed):
    rx, y0 = bed["rx"], bed["y0"]
    lay = bed["lay"]

    def g(T_in):
        s = dict(bed["inlet"])
        s["T"] = T_in
        return jnp.sum(rx.derivatives(0.0, y0, {"feed": s})[-1::rx.n_per_cell])

    d = jax.grad(g)(jnp.asarray(bed["T_in"]))
    assert np.isfinite(float(d))
    del lay


@pytest.fixture(scope="module")
def reformer_bed():
    from difflow_refinery.reforming import ReformingKinetics, lean_naphtha
    from difflow_refinery.reforming.reactor import flows_of, stream_of
    from difflow_refinery.reforming import species as sp
    feed = lean_naphtha()
    F = flows_of(feed.stream())
    F = F.at[sp.INDEX["H2"]].add(5.0 * float(feed.hydrocarbon_moles))
    inlet = stream_of(F, jnp.asarray(773.15), jnp.asarray(15e5))
    rx = DynamicReformingReactor(ReformingKinetics(), 3000.0, 15e5, n_cells=3, coke_alpha=10.0,
                                 T0=[740.0, 730.0, 725.0])
    return rx, inlet


def test_reformer_cells_smoke_and_rest(reformer_bed):
    rx, inlet = reformer_bed
    y0 = rx.initial_state()
    dy = jax.jit(lambda y: rx.derivatives(0.0, y, {"inlet": inlet}))(y0)
    assert bool(jnp.all(jnp.isfinite(dy)))
    assert bool(jnp.all(dy[rx.n_cells:] >= 0.0))                 # coke only accumulates
    # at T = T* (each cell at its own segment's adiabatic outlet) the temperatures are at rest
    _, Ts, _, _ = rx._run(inlet, y0)
    for _ in range(6):                                             # Ts depends on the upstream cell's T
        y = jnp.concatenate([Ts / 100.0, y0[rx.n_cells:]])
        _, Ts, _, _ = rx._run(inlet, y)
    dT = rx.derivatives(0.0, y, {"inlet": inlet})[:rx.n_cells]
    assert float(jnp.max(jnp.abs(dT))) < 1e-8


def test_reformer_cells_gradient_finite(reformer_bed):
    rx, inlet = reformer_bed

    def g(T_in):
        s = dict(inlet)
        s["T"] = T_in
        return jnp.sum(rx.derivatives(0.0, rx.initial_state(), {"inlet": s}))

    assert np.isfinite(float(jax.grad(g)(jnp.asarray(773.15))))


def test_quasi_steady_holdup_conserves_and_rests():
    def split(s):
        a = {k: 0.3 * v for k, v in s.items() if k.startswith("F_")}
        b = {k: 0.7 * v for k, v in s.items() if k.startswith("F_")}
        return dict(a, T=s["T"] - 5.0, P=s["P"]), dict(b, T=s["T"], P=s["P"])

    inlet = {"F_A": jnp.asarray(2.0), "F_B": jnp.asarray(1.0), "T": jnp.asarray(400.0), "P": jnp.asarray(1e5)}
    u = QuasiSteadyHoldup.at_steady(split, {"in": inlet}, ["in"], ["top", "bottom"], ["F_A", "F_B"],
                                    {"top": 60.0, "bottom": 300.0})
    y0 = u.initial_state()
    assert float(jnp.max(jnp.abs(u.derivatives(0.0, y0, {"in": inlet})))) == 0.0
    step = dict(inlet, F_A=jnp.asarray(3.0))
    dy = u.derivatives(0.0, y0, {"in": step}) * u._scale
    held = lambda y: sum(v["F_A"] for v in u.inventory(y).values())
    acc = jax.jvp(held, (y0,), (dy / u._scale,))[1]
    out = u.outputs(0.0, y0, {"in": step})
    assert float(acc) == pytest.approx(3.0 - float(out["top"]["F_A"] + out["bottom"]["F_A"]), rel=1e-12)
