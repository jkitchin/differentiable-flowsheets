"""Tests for the hydrogen network (#329): header balances, PSA, swing, purity, planning block.

All of these are a few dozen flops on synthetic producers; the closed loop
on a real reformer and hydrotreaters is in ``test_hydrogen_loop.py`` (slow).
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery.hydrogen as h2
from difflow_refinery.hydrogen.network import HEADER_GASES, MW

jax.config.update("jax_enable_x64", True)

IMP = {"methane": 0.6, "ethane": 0.3, "propane": 0.1}


def _net(**kw):
    """Reformer gas (88 %) + an H2 plant at a fixed rate, two consumers, import swing."""
    prods = kw.pop("producers", None) or [
        h2.Producer.of_purity("reformer", 100.0, 0.88, impurity=IMP, to_psa=kw.pop("to_psa", 0.0)),
        h2.Producer.of_purity("smr", 20.0, 0.999)]
    cons = kw.pop("consumers", None) or [h2.Consumer("nht", 40.0, P=35e5, min_purity=0.85),
                                         h2.Consumer("dht", 70.0, P=60e5, min_pH2=50e5)]
    heads = kw.pop("headers", None) or [h2.Header("main", swing=[h2.Import(purity=0.999)])]
    return h2.HydrogenNetwork(producers=prods, consumers=cons, headers=heads, **kw)


def _assert_closed(res, tol=1e-12):
    for k, v in res.balances.items():
        assert float(v) < tol, (k, float(v))


def test_balances_close_and_surplus_is_supply_minus_demand():
    res = _net().solve()
    _assert_closed(res)
    o = res.outputs
    assert float(o["h2.surplus"]) == pytest.approx(120.0 - 110.0, abs=1e-10)
    assert float(o["import.h2"]) == 0.0
    # purity is the mix of the two producers
    tot = 100.0 / 0.88 + 20.0 / 0.999
    assert float(o["main.purity"]) == pytest.approx(120.0 / tot, rel=1e-12)
    # every consumer draws header gas; makeup H2 equals its demand
    assert float(o["nht.makeup_h2"]) == pytest.approx(40.0, rel=1e-12)
    assert float(o["nht.makeup_mol_s"]) == pytest.approx(40.0 / float(o["main.purity"]), rel=1e-12)
    # the purge carries the surplus at header composition, to fuel
    np.testing.assert_allclose(np.asarray(res.streams["fuel_gas"]), np.asarray(res.streams["main.purge"]))
    assert float(o["export.mol_s"]) == 0.0


def test_purity_and_partial_pressure_specs():
    res = _net().solve()
    o = res.outputs
    y = float(o["main.purity"])
    assert float(o["nht.purity_margin"]) == pytest.approx(y - 0.85)
    assert float(o["dht.pH2"]) == pytest.approx(y * 60e5)
    assert float(o["dht.purity_margin"]) == pytest.approx(y - 50e5 / 60e5)
    f = res.feasible
    assert f["main.balanced"] and f["main.pressure"] and f["nht.purity"] and f["dht.purity"]
    hard = _net(consumers=[h2.Consumer("nht", 40.0, min_purity=0.95)]).solve()
    assert not hard.feasible["nht.purity"]


def test_deficit_is_reported_not_hidden():
    res = _net(headers=[h2.Header("main")],
               consumers=[h2.Consumer("hcu", 150.0)]).solve()
    assert float(res.outputs["h2.surplus"]) == pytest.approx(-30.0, abs=1e-10)
    assert not res.feasible["main.balanced"]
    _assert_closed(res)


def test_swing_sources_fill_in_order_up_to_capacity():
    heads = [h2.Header("main", min_purge=5.0,
                       swing=[h2.H2Plant(capacity=15.0, purity=0.9999), h2.Import(purity=0.995)])]
    res = _net(headers=heads, consumers=[h2.Consumer("hcu", 140.0)]).solve()
    o = res.outputs
    # need = 140 + 5 - 120 = 25: 15 from the plant, 10 imported, 5 left as the purge
    assert float(o["h2_plant.h2"]) == pytest.approx(15.0)
    assert float(o["import.h2"]) == pytest.approx(10.0)
    assert float(o["h2.surplus"]) == pytest.approx(5.0)
    assert float(o["import.mol_s"]) == pytest.approx(10.0 / 0.995)
    _assert_closed(res)


def test_psa_split_recovery_and_tail_gas():
    psa = h2.PSA(recovery=0.85, purity=0.999)
    res = _net(to_psa=0.5, headers=[h2.Header("main", psa=psa)]).solve()
    o, s = res.outputs, res.streams
    E_h2 = 50.0
    prod, tail = s["main.psa_product"], s["main.psa_tail"]
    assert float(o["main.psa.h2_product"]) == pytest.approx(0.85 * E_h2)
    assert float(prod[0] / jnp.sum(prod)) == pytest.approx(0.999)
    feed = 0.5 * res.streams["reformer"]
    np.testing.assert_allclose(np.asarray(prod + tail), np.asarray(feed), rtol=1e-12)
    assert np.all(np.asarray(tail) >= 0)
    assert res.feasible["main.psa_purity"]
    # tail gas goes to fuel with the purge; H2 lost to fuel = tail H2 + surplus
    assert float(o["fuel_gas.h2_mol_s"]) == pytest.approx(float(tail[0]) + float(o["h2.surplus"]))
    assert float(o["h2.surplus"]) == pytest.approx(120.0 - 0.15 * E_h2 - 110.0)
    _assert_closed(res)


def test_psa_purity_target_is_met_and_reports_when_it_cannot_be():
    head = h2.Header("main", psa=h2.PSA(target_purity=0.95), swing=[h2.Import(purity=0.999)], min_purge=2.0)
    res = _net(to_psa=1.0, headers=[head]).solve()
    o = res.outputs
    assert float(o["main.purity"]) == pytest.approx(0.95, abs=1e-9)
    assert 0.0 < float(o["main.psa.share"]) < 1.0
    _assert_closed(res)
    head = h2.Header("main", psa=h2.PSA(target_purity=0.9995))
    res = _net(to_psa=1.0, headers=[head]).solve()
    assert float(res.outputs["main.psa.share"]) == 1.0
    assert float(res.outputs["main.psa.target_error"]) < -1e-4


def test_two_headers_and_export():
    prods = [h2.Producer.of_purity("reformer", 100.0, 0.88, impurity=IMP, header="lp"),
             h2.Producer.of_purity("smr", 60.0, 0.999, header="hp")]
    cons = [h2.Consumer("nht", 40.0, header="lp"), h2.Consumer("hcu", 50.0, header="hp")]
    heads = [h2.Header("lp"), h2.Header("hp", purge_to="export")]
    res = _net(producers=prods, consumers=cons, headers=heads).solve()
    o = res.outputs
    assert float(o["nht.purity"]) == pytest.approx(0.88)
    assert float(o["hcu.purity"]) == pytest.approx(0.999)
    assert float(o["export.h2_mol_s"]) == pytest.approx(10.0)
    assert float(o["fuel_gas.h2_mol_s"]) == pytest.approx(60.0)
    assert float(o["h2.surplus"]) == pytest.approx(70.0)
    _assert_closed(res)


def test_purity_response_is_a_fixed_point():
    cons = [h2.Consumer("nht", 40.0, purity_ref=0.97, d_demand_d_purity=-30.0),
            h2.Consumer("dht", 70.0, purity_ref=0.97, d_demand_d_purity=-50.0)]
    res = _net(consumers=cons, headers=[h2.Header("main", min_purge=2.0, swing=[h2.Import(purity=0.999)]),
                                       ]).solve()
    o = res.outputs
    y = float(o["main.purity"])
    assert float(o["main.loop_residual"]) < 1e-14
    assert float(o["nht.makeup_h2"]) == pytest.approx(40.0 - 30.0 * (y - 0.97))
    _assert_closed(res)


def test_fold_composition_keeps_the_purity():
    y = h2.gas_vector({"hydrogen": 0.9, "methane": 0.05, "ethane": 0.02, "isopentane": 0.02, "n_hexane": 0.01})
    f = h2.fold_composition(y, ["hydrogen", "methane", "ethane", "propane", "n_butane"])
    assert float(f["hydrogen"]) == pytest.approx(0.9)
    assert float(f["n_butane"]) == pytest.approx(0.03)
    assert sum(float(v) for v in f.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        h2.fold_composition(y, ["hydrogen", "ethane"])


def test_makeup_composition_is_a_hydrotreater_makeup_dict():
    res = _net().solve()
    m = res.makeup_composition("nht", ("hydrogen", "hydrogen_sulfide", "methane", "ethane", "propane"))
    assert all(isinstance(v, float) for v in m.values())
    assert m["hydrogen"] == pytest.approx(float(res.outputs["nht.purity"]))
    assert sum(m.values()) == pytest.approx(1.0)


def test_from_reformer_maps_species_and_lumps_c6_plus():
    from difflow_refinery.reforming import species as sp
    F = jnp.zeros(sp.N_SPECIES).at[sp.INDEX["H2"]].set(80.0).at[sp.INDEX["C1"]].set(6.0) \
        .at[sp.INDEX["iC4"]].set(1.0).at[sp.INDEX["nP6"]].set(0.2).at[sp.INDEX["A6"]].set(0.1)
    stub = SimpleNamespace(flows=lambda name: F, streams={"net_gas": {"P": 11e5}})
    p = h2.Producer.from_reformer(stub)
    assert float(p.h2) == 80.0 and float(p.total) == pytest.approx(float(jnp.sum(F)))
    assert float(p.flows[HEADER_GASES.index("n_hexane")]) == pytest.approx(0.3)
    assert float(p.flows[HEADER_GASES.index("isobutane")]) == 1.0
    assert p.P == 11e5


def test_consumer_from_hydrotreater_reads_makeup_and_params():
    from difflow_refinery.hydrotreating import HydrotreaterParams
    stub = SimpleNamespace(outputs={"h2.makeup": jnp.asarray(33.0)})
    c = h2.Consumer.from_hydrotreater("dht", stub, HydrotreaterParams(P=60e5), min_purity=0.9)
    assert float(c.h2_demand) == 33.0 and c.P == 60e5 and c.purity_ref == pytest.approx(0.97)


def test_validation():
    with pytest.raises(ValueError):
        _net(consumers=[h2.Consumer("nht", 1.0, header="nowhere")])
    with pytest.raises(ValueError):
        _net(consumers=[h2.Consumer("smr", 1.0)])
    with pytest.raises(ValueError):
        h2.gas_vector({"oxygen": 1.0})
    with pytest.raises(ValueError):
        h2.Header(purge_to="flare")


def test_traces_under_jit_and_both_ad_modes():
    def surplus(x):
        n = _net(producers=[h2.Producer.of_purity("reformer", x[0], x[1], impurity=IMP, to_psa=0.5)],
                 headers=[h2.Header("main", psa=h2.PSA(recovery=x[2]), swing=[h2.Import()], min_purge=1.0)],
                 consumers=[h2.Consumer("nht", 60.0, purity_ref=0.97, d_demand_d_purity=-20.0)])
        o = n.solve().outputs
        return jnp.stack([o["h2.surplus"], o["main.purity"], o["fuel_gas.kg_s"]])

    x = jnp.asarray([100.0, 0.88, 0.85])
    np.testing.assert_allclose(np.asarray(jax.jit(surplus)(x)), np.asarray(surplus(x)), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(jax.jacrev(surplus)(x)), np.asarray(jax.jacfwd(surplus)(x)),
                               rtol=1e-10, atol=1e-14)


@pytest.mark.release
def test_header_gradients_match_finite_differences():
    def f(x):
        n = _net(producers=[h2.Producer.of_purity("reformer", x[0], x[1], impurity=IMP, to_psa=0.4)],
                 headers=[h2.Header("main", psa=h2.PSA(recovery=x[2], target_purity=0.93),
                                    swing=[h2.H2Plant(capacity=30.0), h2.Import(purity=0.995)])],
                 consumers=[h2.Consumer("nht", x[3], purity_ref=0.93, d_demand_d_purity=-25.0, min_purity=0.9),
                            h2.Consumer("dht", 70.0)])
        o = n.solve().outputs
        return jnp.stack([o["h2.surplus"], o["main.purity"], o["fuel_gas.kg_s"], o["main.psa.share"],
                          o["h2_plant.h2"], o["nht.makeup_mol_s"]])

    x0 = jnp.asarray([100.0, 0.88, 0.86, 45.0])
    J = jax.jacrev(f)(x0)
    h = jnp.asarray([1e-3, 1e-6, 1e-6, 1e-3])
    for j in range(4):
        e = jnp.zeros(4).at[j].set(h[j])
        fd = (f(x0 + e) - f(x0 - e)) / (2 * h[j])
        np.testing.assert_allclose(np.asarray(J[:, j]), np.asarray(fd), rtol=1e-6, atol=1e-8)


def test_h2_block_reproduces_the_network_and_links():
    net = _net(to_psa=0.3, headers=[h2.Header("main", psa=h2.PSA(), swing=[h2.Import()])])
    blk = h2.h2_block(net, jit=False)
    assert blk.u_names == ["reformer.h2", "reformer.purity", "smr.h2", "smr.purity", "nht.makeup", "dht.makeup"]
    y = np.asarray(blk.evaluate(blk.u0))
    o = net.solve().outputs
    np.testing.assert_allclose(y, [float(o[k]) for k in blk.y_names], rtol=1e-12)
    assert blk.metadata["u_units"][1] == "mol%" and blk.metadata["u_units"][4] == "Nm3/h"
    from difflow_refinery.hydrogen.planning import link_hdt, link_reformer
    assert link_reformer() == [("reformer.h2.net_mol_s", "h2.reformer.h2"),
                               ("reformer.h2.purity", "h2.reformer.purity")]
    assert link_hdt("nht") == [("nht.h2.makeup", "h2.nht.makeup")]
    # a lever moves the answer the way the network does
    u = np.array(blk.u0, dtype=float)
    u[0] += 5.0
    assert float(blk.evaluate(u)[0]) == pytest.approx(float(o["h2.surplus"]) + 5.0 * (0.7 + 0.3 * 0.88))


def test_mass_basis_is_the_header_gases():
    assert MW[0] == pytest.approx(2.01588e-3, rel=1e-6)
    assert len(MW) == len(HEADER_GASES)
