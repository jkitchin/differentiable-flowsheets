"""The hydrogen header closed on a real reformer and two hydrotreaters (#329 acceptance).

Reformer net gas -> header (import as the swing) -> kerosene and diesel
hydrotreaters, with the header's makeup composition fed back into each
``HydrotreaterParams.makeup`` until the purity stops moving. Each
hydrotreater compiles once (about a minute); the later passes re-solve the
compiled units. Slow.

The gradient test differentiates the surplus with respect to the reformer
WAIT in forward mode (the reformer's beds are ``diffrax.ForwardMode``),
through the closed network whose hydrotreater responses are the loop's
linear ones, and checks it against central differences of the same
composition. Slow and release.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
import difflow_refinery.hydrogen as h2
from difflow_refinery.hydrotreating import Hydrotreater, HydrotreaterParams, straight_run_cut
from difflow_refinery.hydrotreating.feed import select_cuts
from difflow_refinery.reforming import CatalyticReformer, ReformerParams, lean_naphtha

jax.config.update("jax_enable_x64", True)

TBP_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
TBP_A = [60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680]


def _char():
    a = dr.Assay(TBP_PCT, jnp.asarray([t + 273.15 for t in TBP_A]), sg=0.86,
                 light_ends={"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5},
                 heavy_end=dr.HeavyEnd(), sulfur_wt=1.8, nitrogen_wppm=1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(a, cut_points=dr.default_cut_points(a), composition=True)


@pytest.fixture(scope="module")
def reformer():
    unit = CatalyticReformer(ReformerParams())
    return unit, unit.solve(lean_naphtha())


@pytest.fixture(scope="module")
def hydrotreaters():
    char = _char()
    kc = select_cuts(char, 150 + 273.15, 250 + 273.15)
    kfeed = straight_run_cut(char, rate=20.0, cuts=kc)
    kp = HydrotreaterParams(T_in=(593.15,), quench=None, bed_fractions=(1.0,), P=35e5, h2_oil=150.0,
                            lhsv=2.5, stripper_feed_T=473.15)
    dc = select_cuts(char, 230 + 273.15, 370 + 273.15)
    dfeed = straight_run_cut(char, rate=50.0, cuts=dc)     # demand > reformer H2: the import swings
    dp = HydrotreaterParams()
    return {"kht": (Hydrotreater(char, kfeed, kp), kfeed, kp), "dht": (Hydrotreater(char, dfeed, dp), dfeed, dp)}


def _network(producer, hdts):
    cons = [h2.Consumer(n, 20.0, P=p.P, min_purity=0.85) for n, (_, _, p) in hdts.items()]
    return h2.HydrogenNetwork(producers=[producer], consumers=cons,
                              headers=[h2.Header("main", P=10e5, min_purge=1.0,
                                                 swing=[h2.Import(purity=0.999, P=30e5)])])


@pytest.fixture(scope="module")
def loop(reformer, hydrotreaters):
    _, ref = reformer
    net = _network(h2.Producer.from_reformer(ref), hydrotreaters)
    return h2.close_hydrotreater_loop(net, hydrotreaters, tol=1e-7)


@pytest.mark.slow
def test_reformer_to_two_hydrotreaters_closes(loop, reformer):
    _, ref = reformer
    res = loop.header
    assert loop.converged and loop.passes >= 2   # import swing: purity moves with the demands
    for k, v in res.balances.items():
        assert float(v) < 1e-12, k
    o = res.outputs
    y = float(o["main.purity"])
    # the header is a mix of reformer gas (its own purity) and 99.9 % import
    assert float(ref.outputs()["h2.purity"]) / 100.0 - 1e-9 < y < 0.999
    for name, r in loop.units.items():
        assert bool(r.converged)
        for k, v in r.balances.items():
            assert float(v) < 1e-8, (name, k)
        # the unit was solved at the purity the header delivers it ...
        assert loop.params[name].makeup["hydrogen"] == pytest.approx(y, abs=1e-7)
        # ... and the header balances the demand the unit reports at that purity
        assert float(o[f"{name}.makeup_h2"]) == pytest.approx(float(r.outputs["h2.makeup"]), rel=1e-6)
        assert float(o[f"{name}.purity_margin"]) == pytest.approx(y - 0.85)
    supply = float(ref.outputs()["h2.net_mol_s"]) + float(o["import.h2"])
    demand = sum(float(r.outputs["h2.makeup"]) for r in loop.units.values())
    assert float(o["h2.surplus"]) == pytest.approx(supply - demand, rel=1e-6, abs=1e-6)
    assert float(o["h2.surplus"]) >= 1.0 - 1e-9
    # a leaner makeup than the 97 % default costs makeup: the loop's secant says which way
    for c in loop.network.consumers:
        assert np.isfinite(float(c.d_demand_d_purity))


@pytest.mark.slow
@pytest.mark.release
def test_surplus_gradient_wrt_wait_matches_finite_differences(loop, reformer):
    unit, ref = reformer
    feed = lean_naphtha()
    tear = ref.tear
    # purge as the swing so the surplus is the reformer's H2 less the (responding) demands
    net = loop.network.replace(headers=[h2.Header("main", P=10e5)])

    def surplus(wait_C):
        p = ReformerParams().with_wait(wait_C + 273.15)
        r = unit.solve(feed, p, tear_initial=tear, tol=1e-11, max_iter=400, on_nonconvergence="ignore")
        o = net.replace(producers=[h2.Producer.from_reformer(r)]).solve().outputs
        return jnp.stack([o["h2.surplus"], o["main.purity"]])

    w0 = float(ReformerParams().wait) - 273.15
    J = jax.jacfwd(surplus)(jnp.asarray(w0))
    h = 0.02
    fd = (surplus(jnp.asarray(w0 + h)) - surplus(jnp.asarray(w0 - h))) / (2 * h)
    np.testing.assert_allclose(np.asarray(J), np.asarray(fd), rtol=1e-5, atol=1e-9)
    assert float(J[0]) != 0.0
