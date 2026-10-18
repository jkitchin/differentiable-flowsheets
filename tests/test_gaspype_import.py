"""Tests for difflow.gaspype_import (#492): gaspype data in a GibbsReactor."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("gaspype")

from difflow import (
    Flowsheet,
    GibbsReactor,
    GibbsReactorParams,
    Unit,
    codegen,
    make_stream,
    serialize,
)
from difflow.gaspype_import import GaspypeThermo, list_available_species

jax.config.update("jax_enable_x64", True)

SMR = ["CH4", "H2O", "H2", "CO", "CO2"]


@pytest.mark.parametrize("method", ["nasa9", "table"])
def test_thermo_smoke(method):
    th = GaspypeThermo(SMR, method=method)
    assert th.names == tuple(SMR)
    assert set(th.ELEMENTS) == {"C", "H", "O"}
    for f in (th.gibbs, th.enthalpy):
        v = f(1000.0)
        assert v.shape == (len(SMR),)
        assert jnp.all(jnp.isfinite(v))


def test_nasa9_extrapolates_outside_data_range():
    th = GaspypeThermo(SMR)
    for T in (150.0, 7000.0):
        assert jnp.all(jnp.isfinite(th.gibbs(T)))


@pytest.mark.parametrize("method", ["nasa9", "table"])
def test_reactor_smoke_and_jit_grad(method):
    th = GaspypeThermo(SMR, method=method)

    def x_h2(T):
        out, _ = GibbsReactor(GibbsReactorParams(thermo=th, T_out=T))(
            make_stream({"CH4": 1.0, "H2O": 3.0}, T=900.0, P=1e5))
        return out["F_H2"] / sum(out[f"F_{s}"] for s in SMR)

    assert jnp.isfinite(x_h2(1000.0))
    assert jnp.isfinite(jax.jit(jax.grad(x_h2))(1000.0))


def test_list_available_species():
    names = list_available_species("C#H#", "C,H")
    assert "CH4" in names
    assert "H2O" not in names


def test_unknown_species():
    with pytest.raises(ValueError, match="not in gaspype"):
        GaspypeThermo(["CH4", "Unobtainium"])
    with pytest.raises(ValueError):
        GaspypeThermo(["CH4", "Unobtainium"], method="table")


def test_bad_arguments():
    with pytest.raises(TypeError, match="list of names"):
        GaspypeThermo("CH4, H2O")
    with pytest.raises(ValueError, match="empty"):
        GaspypeThermo([])
    with pytest.raises(ValueError, match="repeated"):
        GaspypeThermo(["CH4", "CH4"])
    with pytest.raises(ValueError, match="method"):
        GaspypeThermo(SMR, method="cubic")


def test_padded_intervals_keep_each_species_own_data():
    # CO2 has three NASA-9 intervals, CH4 two: CH4 above 6000 K must use its
    # own last interval, not padding zeros.
    th = GaspypeThermo(["CH4", "CO2"])
    assert th._coeffs.shape[1] == 3
    assert np.all(th._coeffs[0, 2] == th._coeffs[0, 1])


def test_exported_flat_and_as_namespace():
    import difflow

    assert difflow.GaspypeThermo is GaspypeThermo
    assert difflow.gaspype_import.list_available_species is list_available_species


def _reformer_flowsheet(method="nasa9"):
    fs = Flowsheet(species_order=SMR)
    fs.add_feed("feed", make_stream({"CH4": 1.0, "H2O": 3.0, "H2": 0.0, "CO": 0.0,
                                     "CO2": 0.0}, T=900.0, P=1e5))
    th = GaspypeThermo(SMR, method=method)
    fs.add_unit(Unit("ref", GibbsReactor(GibbsReactorParams(thermo=th, T_out=1000.0)),
                     ["feed"], ["out"]))
    return fs


@pytest.mark.parametrize("method", ["nasa9", "table"])
def test_serialize_round_trip(method):
    fs = _reformer_flowsheet(method)
    back = serialize.from_json(serialize.to_json(fs))
    th = back.units[0].operation.params.thermo
    assert isinstance(th, GaspypeThermo)
    assert th.names == tuple(SMR) and th.method == method
    assert float(back.solve()["out"]["F_H2"]) == float(fs.solve()["out"]["F_H2"])


def test_codegen_script_rebuilds_the_flowsheet():
    fs = _reformer_flowsheet()
    source = codegen.to_python(fs)
    assert "GaspypeThermo(" in source
    ns = {}
    exec(compile(source, "<codegen>", "exec"), ns)
    assert float(ns["fs"].solve()["out"]["F_H2"]) == float(fs.solve()["out"]["F_H2"])
