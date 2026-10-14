"""Tests for the Pipe unit operation (liquid Darcy-Weisbach flow)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import Pipe, PipeParams, SpeciesData, IdealThermo, make_stream
from difflow.catalog import describe_operation

RHO, MU, MW_W = 998.2, 1.002e-3, 18.015


def water(Q, P=3e5, T=293.15):
    return make_stream({"W": Q * RHO / (MW_W * 1e-3)}, T, P)


def params(**kw):
    base = dict(L=100.0, D=2.067 * 0.0254, rho=RHO, mu=MU, MW=MW_W)
    base.update(kw)
    return PipeParams(**base)


def test_pipe_smoke():
    inlet = water(3e-3)
    out, info = Pipe(params(fittings={"elbow_90_standard": 4, "exit": 1}, dz=2.0))(inlet)
    assert float(out["F_W"]) == float(inlet["F_W"])
    for k in ("v", "Re", "f", "dP", "dP_friction", "dP_minor", "dP_static"):
        assert np.isfinite(float(info[k]))
    _, info2 = Pipe(params(fittings=2.0))(inlet)
    assert np.isfinite(float(info2["dP_minor"]))


def test_callable_properties():
    rho = lambda s: RHO - 0.5 * (s["T"] - 293.15)
    mu = lambda s: MU * jnp.exp(-0.02 * (s["T"] - 293.15))
    _, info = Pipe(params(rho=rho, mu=mu))(water(3e-3, T=333.15))
    assert np.isfinite(float(info["dP"]))


def test_mixture_MW_dict_and_thermo():
    sp = {
        "A": SpeciesData("A", 18.0, (75.0, 0, 0, 0), (4e4, 0.38, 647.0), (10.0, 1700.0, -40.0)),
        "B": SpeciesData("B", 46.0, (110.0, 0, 0, 0), (4e4, 0.38, 513.0), (10.0, 1600.0, -40.0)),
    }
    inlet = make_stream({"A": 40.0, "B": 10.0}, 300.0, 2e5)
    for unit in (Pipe(params(MW={"A": 18.0, "B": 46.0})),
                 Pipe(params(MW=None), thermo=IdealThermo(sp))):
        assert np.isfinite(float(unit(inlet)[1]["Q"]))


def test_missing_molar_mass_raises():
    with pytest.raises(ValueError, match="molar mass"):
        Pipe(params(MW=None))(water(1e-3))


def test_zero_flow_is_finite():
    """Regression: zero flow must not give nan in value or gradient."""
    out, _ = Pipe(params())(water(0.0))
    assert np.isfinite(float(out["P"]))
    g = jax.grad(lambda q: Pipe(params())(water(q))[1]["dP"])(0.0)
    assert np.isfinite(float(g))


def test_grad_finite():
    g = jax.grad(lambda D: Pipe(params(D=D))(water(3e-3))[1]["dP"])(0.05)
    assert np.isfinite(float(g))


def test_eo_residuals():
    unit = Pipe(params(fittings={"gate_valve": 2}, dz=3.0))
    inlet = water(3e-3)
    outlet, _ = unit(inlet)
    r = unit.eo_residuals([inlet], [outlet])
    assert r.shape == (3,)
    np.testing.assert_allclose(np.asarray(r), 0.0, atol=1e-9)
    J = jax.jacobian(lambda i: unit.eo_residuals([i], [outlet]))(inlet)
    assert all(np.all(np.isfinite(np.asarray(v))) for v in J.values())


def test_catalog_describes_pipe():
    spec = describe_operation("Pipe")
    assert spec.ports.n_inlets == 1 and spec.ports.n_outlets == 1
    assert {"L", "D"} <= {p.name for p in spec.parameters if p.required}
