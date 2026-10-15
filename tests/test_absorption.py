"""Smoke tests for the generic Absorber / Stripper (#428)."""

import jax
import numpy as np
import pytest

from difflow import make_stream
from difflow.staged import linear_equilibrium
from difflow.units.absorption import (
    Absorber,
    AbsorptionParams,
    Stripper,
    henry_params,
    minimum_solvent_rate,
)

jax.config.update("jax_enable_x64", True)

T0, P0 = 298.15, 101325.0


def _streams(y=0.02, V=1.0, L=2.0):
    gas = make_stream({"air": V * (1 - y), "acetone": V * y}, T0, P0)
    liq = make_stream({"H2O": L}, T0, P0)
    return gas, liq


def _absorber(**kw):
    henry, C = henry_params(["acetone"])
    kw.setdefault("henry", henry)
    kw.setdefault("henry_C", C)
    return Absorber(AbsorptionParams(**kw))


@pytest.mark.parametrize("mode", [
    dict(model="kremser", n_stages=5.3),
    dict(model="stages", n_stages=4),
    dict(model="packed", height=4.0, H_OG=0.7),
], ids=["kremser", "stages", "packed"])
def test_absorber_mass_balance(mode):
    gas, liq = _streams()
    g, l, info = _absorber(**mode)(gas, liq)
    for s in ("air", "acetone", "H2O"):
        tin = float(gas.get(f"F_{s}", 0.0)) + float(liq.get(f"F_{s}", 0.0))
        tout = float(g.get(f"F_{s}", 0.0)) + float(l.get(f"F_{s}", 0.0))
        assert abs(tin - tout) < 1e-10, s
    assert bool(info["converged"])


def test_stripper_mass_balance():
    henry, C = henry_params(["acetone"])
    gas = make_stream({"air": 4.0}, T0, P0)
    liq = make_stream({"H2O": 3.0, "acetone": 0.02}, T0, P0)
    g, l, _ = Stripper(AbsorptionParams(henry=henry, henry_C=C, n_stages=4.0))(gas, liq)
    assert float(g["F_acetone"] + l["F_acetone"]) == pytest.approx(0.02, abs=1e-12)


def test_gradient_finite():
    def f(L):
        gas = make_stream({"air": 1.0, "acetone": 0.02}, T0, P0)
        liq = make_stream({"H2O": L}, T0, P0)
        return _absorber(n_stages=5.4)(gas, liq)[0]["F_acetone"]

    assert np.isfinite(float(jax.grad(f)(2.0)))


def test_stages_validation():
    gas, liq = _streams()
    with pytest.raises(ValueError, match="integer"):
        _absorber(model="stages", n_stages=3.5)(gas, liq)
    with pytest.raises(ValueError, match="design"):
        _absorber(model="stages", recovery=0.9)(gas, liq)


def test_packed_validation():
    gas, liq = _streams()
    kw = dict(model="packed", variable_flow=False)
    with pytest.raises(ValueError, match="height"):
        _absorber(H_OG=0.6, **kw)(gas, liq)
    with pytest.raises(ValueError, match="not both"):
        _absorber(H_OG=0.6, height=2.0, recovery=0.9, **kw)(gas, liq)
    with pytest.raises(ValueError, match="D_G"):
        _absorber(H_OG=None, packing="pall_metal_50", diameter=0.5, height=2.0, **kw)(gas, liq)


def test_minimum_solvent_rate_infeasible_is_inf_not_nan():
    v = float(minimum_solvent_rate(linear_equilibrium(2.0), 0.05, 0.001, 0.01, 1.0))
    assert np.isinf(v)
