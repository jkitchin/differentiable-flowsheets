"""The crude unit as a registered flowsheet operation."""

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow import serialize
from difflow.catalog import describe_class
from difflow.flowsheet import Flowsheet, Unit
from difflow.plugins import OperationRegistry
from difflow_refinery import Assay
from difflow_refinery import column as cc

jax.config.update("jax_enable_x64", True)

ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)


def _params(**over):
    kw = dict(n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
              specs=(cc.product_rate("naphtha", 0.25, basis="mole"), cc.coil_outlet_temperature(620.0)))
    kw.update(over)
    return dr.CrudeDistillationUnitParams(assay=ASSAY, column=cc.CrudeColumnParams(**kw))


@pytest.fixture(scope="module")
def cdu():
    return dr.CrudeDistillationUnit(_params())


@pytest.fixture(scope="module")
def feed(cdu):
    return cdu.feed(1.0, T=500.0, P=4e5, basis="mole")


def _flowsheet(cdu, feed):
    fs = Flowsheet(list(cdu.unit.thermo.names) + ["water"])
    fs.add_feed("crude", feed)
    fs.add_unit(Unit("cdu", cdu, ["crude"], list(cdu.outlet_names)))
    return fs


class TestProtocol:
    def test_outlets_follow_outlet_names(self, cdu, feed):
        out = cdu(feed)
        res = cdu.solve(feed)
        assert len(out) == len(cdu.outlet_names)
        for name, stream in zip(cdu.outlet_names, out):
            assert stream is not None
            for k, v in stream.items():
                np.testing.assert_allclose(v, res.products[name][k], rtol=1e-12)

    def test_side_products_and_offgas_are_ports(self):
        op = dr.CrudeDistillationUnit(_params(
            n_stages=10, feed_stage=9, condenser="partial",
            side_products=(cc.SideProduct("kero", 4, 0),),
            specs=(cc.product_rate("naphtha", 0.2, basis="mole"),
                   cc.product_rate("offgas", 0.01, basis="mole"),
                   cc.product_rate("kero", 0.15, basis="mole"),
                   cc.coil_outlet_temperature(620.0))))
        assert op.outlet_names == ("naphtha", "kero", "residue", "water", "offgas")

    def test_in_a_flowsheet(self, cdu, feed):
        streams = _flowsheet(cdu, feed).solve()
        assert set(cdu.outlet_names) <= set(streams)
        assert bool(cdu.last_result.converged)
        np.testing.assert_allclose(streams["naphtha"]["F_" + cdu.crude.names[0]],
                                   cdu.last_result.products["naphtha"]["F_" + cdu.crude.names[0]])

    def test_gradient_through_the_call(self, cdu, feed):
        def residue(T):
            return sum(v for k, v in cdu({**feed, "T": T})[1].items() if k.startswith("F_"))

        # the coil outlet is specified, so preheat changes the furnace, not the split
        assert abs(float(jax.grad(residue)(500.0))) < 1e-9

        def fired(T):
            return cdu.solve({**feed, "T": T}).column.furnace_fired_duty

        assert float(jax.grad(fired)(500.0)) < 0


class TestRegistration:
    def test_register(self):
        reg = OperationRegistry()
        dr.register(reg)
        info = reg.list_operations()["CrudeDistillationUnit"]
        assert info.cls is dr.CrudeDistillationUnit
        assert info.category == "refinery"

    def test_catalog_reads_the_params(self):
        spec = describe_class(dr.CrudeDistillationUnit)
        assert spec.params_class == "CrudeDistillationUnitParams"
        assert [p.name for p in spec.parameters] == ["assay", "column", "cut_points", "method"]
        assert all(p.description for p in spec.parameters)
        assert spec.ports.inlets == ["feed"]
        assert spec.ports.n_outlets is None  # depends on the column's side products

    def test_json_round_trip(self, cdu, feed):
        reg = OperationRegistry()
        dr.register(reg)
        fs = _flowsheet(cdu, feed)
        text = serialize.to_json(fs, registry=reg)
        back = serialize.from_json(text, registry=reg)
        op = back.units[0].operation
        assert isinstance(op, dr.CrudeDistillationUnit)
        assert op.params.column.specs == cdu.params.column.specs
        assert op.unit.params.furnace == cc.Furnace()
        assert serialize.to_json(back, registry=reg) == text
        for a, b in zip(op(feed), cdu(feed)):
            for k in a:
                np.testing.assert_allclose(a[k], b[k], rtol=1e-12)
