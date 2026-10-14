"""The crude unit as a registered flowsheet operation."""

import jax
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
