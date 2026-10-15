"""Smoke tests for MultistageCompressor and stage_pressures."""

import jax
import numpy as np
import pytest

from difflow import (
    CubicThermo, IdealThermo, MultistageCompressor, MultistageCompressorParams, PengRobinson,
    make_stream,
)
from difflow.compressible import stage_pressures
from difflow.database import get_critical_props, get_species_data

jax.config.update("jax_enable_x64", True)

NAMES = ["methane", "ethane"]


def test_stage_pressures():
    Ps, Pd = stage_pressures(1e5, 30e5, 3, 0.3e5)
    assert Ps.shape == Pd.shape == (3,)
    assert float(Pd[-1]) == pytest.approx(30e5, rel=1e-12)
    g = jax.grad(lambda P: stage_pressures(1.0, P, 3, 0.2)[1][0])(30.0)
    assert np.isfinite(float(g))
    with pytest.raises(ValueError):
        stage_pressures(1e5, 30e5, 0)


def test_multistage_compressor_smoke():
    thermo = CubicThermo(IdealThermo({c: get_species_data(c) for c in NAMES}),
                         PengRobinson({c: get_critical_props(c) for c in NAMES}))
    feed = make_stream({"methane": 0.9, "ethane": 0.1}, T=313.15, P=1e5)
    params = MultistageCompressorParams(P_out=9e5, n_stages=2, intercooler_dP=0.1e5)
    out, info = MultistageCompressor(params, thermo)(feed)
    assert float(out["P"]) == pytest.approx(9e5)
    assert float(out["F_methane"]) == pytest.approx(0.9)
    assert np.isfinite(float(info["W"]))
