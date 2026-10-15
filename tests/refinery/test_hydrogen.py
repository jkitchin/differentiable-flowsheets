"""Hydrogen network (#329): smoke, validation and API tests."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery.hydrogen as h2
from difflow_refinery.hydrogen.network import HEADER_GASES

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


def test_network_smoke():
    res = _net().solve()
    _assert_closed(res)
    assert all(np.isfinite(float(v)) for v in res.outputs.values())
    assert {"h2.surplus", "main.purity", "nht.makeup_h2"} <= set(res.outputs)


def test_psa_header_smoke():
    res = _net(to_psa=0.5, headers=[h2.Header("main", psa=h2.PSA(recovery=0.85, purity=0.999))]).solve()
    _assert_closed(res)
    assert np.isfinite(float(res.outputs["main.psa.h2_product"]))


def test_grad_finite():
    def surplus(x):
        n = _net(producers=[h2.Producer.of_purity("reformer", x[0], x[1], impurity=IMP, to_psa=0.5)],
                 headers=[h2.Header("main", psa=h2.PSA(recovery=x[2]), swing=[h2.Import()], min_purge=1.0)],
                 consumers=[h2.Consumer("nht", 60.0, purity_ref=0.97, d_demand_d_purity=-20.0)])
        return n.solve().outputs["main.purity"]

    g = jax.grad(surplus)(jnp.asarray([100.0, 0.88, 0.85]))
    assert np.all(np.isfinite(np.asarray(g)))


def test_h2_block_builds():
    net = _net(to_psa=0.3, headers=[h2.Header("main", psa=h2.PSA(), swing=[h2.Import()])])
    blk = h2.h2_block(net, jit=False)
    assert blk.u_names == ["reformer.h2", "reformer.purity", "smr.h2", "smr.purity", "nht.makeup", "dht.makeup"]
    assert np.all(np.isfinite(np.asarray(blk.evaluate(blk.u0))))


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
