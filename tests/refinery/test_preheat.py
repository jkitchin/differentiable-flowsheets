"""The crude preheat train, desalter and preflash drum in front of the CDU (#313):
smoke, validation and API tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow import serialize
from difflow.catalog import describe_class
from difflow.flowsheet import Flowsheet, Unit
from difflow.plugins import OperationRegistry
from difflow_refinery import column as cc
from difflow_refinery.preheat import crude_split

jax.config.update("jax_enable_x64", True)

RATE = 100.0  # mol/s


def _small_case():
    """``(assay, column params, train params)``: an 8-stage column behind a
    three-exchanger train, hot streams the residue and the naphtha."""
    a = dr.Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)
    col = cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=20.0, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 25.0, basis="mole"), cc.coil_outlet_temperature(620.0)))
    ex = (dr.PreheatExchanger("E1", 300.0, 40.0), dr.PreheatExchanger("E2", 300.0, 80.0, Rf=3e-4),
          dr.PreheatExchanger("E3", 300.0, 80.0, Rf=5e-4))
    tp = dr.PreheatTrainParams(
        ex, (dr.HotStream("residue", ("E3", "E2")), dr.HotStream("naphtha", ("E1",))),
        ("E1", "desalter", "E2", "preflash", "E3"),
        desalter=dr.DesalterParams(), drum=dr.PreflashDrumParams(P=2.0e5))
    return a, col, tp


@pytest.fixture(scope="module")
def small():
    a, col, tp = _small_case()
    crude = dr.characterize(a)
    th = dr.ColumnThermo.from_characterization(crude)
    return a, col, tp, crude, th


def _flows(crude, rate):
    s = crude.stream(rate, T=300.0, P=1e5, basis="mole")
    return jnp.stack([jnp.asarray(s[f"F_{n}"]) for n in crude.names])


def _hot(small):
    _, _, _, crude, th = small
    f = _flows(crude, RATE)
    heavy = jnp.where(jnp.arange(f.shape[0]) >= f.shape[0] // 2, f, 0.0)
    return {"residue": (0.8 * heavy, jnp.asarray(560.0)), "naphtha": (0.2 * f, jnp.asarray(400.0))}


def test_train_smoke_and_grad(small):
    _, _, tp, crude, th = small
    tr = dr.PreheatTrain(tp, th)
    f = _flows(crude, RATE)
    r = tr.solve(f, 300.0, _hot(small))
    assert bool(r.converged) and np.isfinite(float(r.recovered)) and np.isfinite(float(r.T_out))
    g = jax.grad(lambda rf: tr.solve(f, 300.0, _hot(small), params=tp.with_exchanger("E3", Rf=rf)).T_out)(5e-4)
    assert np.isfinite(float(g))
    rates = dr.preheat.fouling_rates(r, tr.params, dr.EbertPanchal())
    assert all(np.isfinite(float(v)) for v in rates.values())


def test_crude_split_smoke(small):
    _, _, _, crude, th = small
    out = crude_split(th, _flows(crude, RATE), 1.0, 500.0, 3e5)
    assert np.all(np.isfinite(np.asarray(out["vapor"])))


@pytest.fixture(scope="module")
def desalter(small):
    return dr.Desalter(dr.DesalterUnitParams(assay=small[0], desalter=dr.DesalterParams()))


def test_desalter_smoke(desalter):
    out = desalter.solve(desalter.feed(RATE, T=410.0, P=10e5))
    assert np.isfinite(float(out["T"])) and float(out["brine"]["F_water"]) > 0


def test_drum_smoke(small):
    drum = dr.PreflashDrum(dr.PreflashDrumUnitParams(assay=small[0], drum=dr.PreflashDrumParams(P=2.5e5)))
    out = drum.solve(drum.feed(RATE, T=480.0, P=12e5, water=0.002))
    assert np.isfinite(float(out["T"])) and 0 < float(out["vapor_fraction"]) < 1


def test_register_and_catalog():
    reg = OperationRegistry()
    dr.register(reg)
    ops = reg.list_operations()
    for name, params in (("Desalter", "DesalterUnitParams"), ("PreflashDrum", "PreflashDrumUnitParams"),
                         ("CrudeUnitWithPreheat", "CrudeUnitWithPreheatParams")):
        assert ops[name].cls is getattr(dr, name) and ops[name].category == "refinery"
        spec = describe_class(ops[name].cls)
        assert spec.params_class == params and spec.ports.inlets == ["feed"]


def test_json_round_trip(desalter):
    feed = desalter.feed(RATE, T=430.0, P=10e5)
    reg = OperationRegistry()
    dr.register(reg)
    fs = Flowsheet(list(desalter.thermo.names) + ["water"])
    fs.add_feed("crude", feed)
    fs.add_unit(Unit("u", desalter, ["crude"], list(desalter.outlet_names)))
    text = serialize.to_json(fs, registry=reg)
    back = serialize.from_json(text, registry=reg)
    assert type(back.units[0].operation) is type(desalter)
    assert serialize.to_json(back, registry=reg) == text


def test_crude_unit_with_preheat_round_trip(small):
    a, col, tp, _, _ = small
    op = dr.CrudeUnitWithPreheat(dr.CrudeUnitWithPreheatParams(assay=a, column=col, train=tp))
    assert op.outlet_names == ("naphtha", "residue", "water", "brine", "drum_water")
    reg = OperationRegistry()
    dr.register(reg)
    fs = Flowsheet(list(op.unit.thermo.names) + ["water"])
    fs.add_feed("crude", op.feed(RATE, T=300.0, basis="mole"))
    fs.add_unit(Unit("cdu", op, ["crude"], list(op.outlet_names)))
    back = serialize.from_json(serialize.to_json(fs, registry=reg), registry=reg).units[0].operation
    assert back.params.train == tp and back.params.column == col


def _train(**over):
    kw = dict(exchangers=(dr.PreheatExchanger("E1", 300.0, 50.0), dr.PreheatExchanger("E2", 300.0, 50.0)),
              hot_streams=(dr.HotStream("residue", ("E2", "E1")),), crude_path=("E1", "E2"))
    kw.update(over)
    return dr.PreheatTrainParams(**kw)


@pytest.mark.parametrize("over,match", [
    ({"crude_path": ("E1",)}, "not on the crude path"),
    ({"crude_path": ("E1", "E2", "E3")}, "not exchangers"),
    ({"crude_path": ("E1", "desalter", "E2")}, "no DesalterParams"),
    ({"hot_streams": (dr.HotStream("residue", ("E2",)),)}, "on no hot stream"),
    ({"crude_path": ("E1", "preflash", "desalter", "E2"), "desalter": dr.DesalterParams(),
      "drum": dr.PreflashDrumParams()}, "before the preflash"),
])
def test_train_structure_errors(small, over, match):
    with pytest.raises(ValueError, match=match):
        dr.PreheatTrain(_train(**over), small[4])


def test_hot_stream_must_be_a_product_or_pumparound(small):
    a, col, _, _, _ = small
    with pytest.raises(ValueError, match="neither a product nor a pumparound"):
        dr.PreheatedCrudeUnit(a, col, _train(hot_streams=(dr.HotStream("kero", ("E2", "E1")),)))


def test_crude_unit_with_preheat_solves(small):
    """The coupled unit (column + train) on a wet feed: converges, finite outlets."""
    a, col, tp, _, _ = small
    op = dr.CrudeUnitWithPreheat(dr.CrudeUnitWithPreheatParams(assay=a, column=col, train=tp))
    out = op(dict(op.feed(RATE, T=300.0, basis="mole"), F_water=50.0))
    assert bool(op.last_result.converged)
    assert len(out) == len(op.outlet_names)
    assert all(np.isfinite(float(s["F_water"])) for s in out)
