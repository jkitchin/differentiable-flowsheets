"""Jet and VGO hydrotreating services (#536): the pure functions around the
unit. No hydrotreater solve here (it compiles for a minute); full solves are
by hand in tests/reference/hdt_services_check.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.hydrotreating.services import (JET_SPECS, fcc_feed_from_hydrotreater,
                                                     heavy_ends_removal, jet_margins, jet_params,
                                                     jet_properties, vgo_params)

jax.config.update("jax_enable_x64", True)

KERO = {"product.sg": 0.80, "product.T50": 473.15, "product.T10": 440.0,
        "product.aromatics_vol": 18.0, "product.S_wppm": 20.0}
VGO = {"product.rate": 40.0, "product.sg": 0.90, "product.T50": 720.0, "product.H_wt": 12.8,
       "product.S_wppm": 800.0, "product.N_wppm": 600.0}


def test_jet_properties_smoke():
    jp = jet_properties(KERO)
    assert set(jp) == {s[0] for s in JET_SPECS}
    assert all(np.isfinite(float(v)) for k, v in jp.items() if k != "freeze_point_C")
    assert np.isnan(float(jp["freeze_point_C"]))          # not computable from outputs
    m = jet_margins(jp)
    assert np.isnan(float(m["freeze_point_C"])) and float(m["S_wppm"]) > 0


def test_jet_properties_grad_finite():
    g = jax.grad(lambda sg: jet_properties({**KERO, "product.sg": sg})["smoke_point_mm"])(0.80)
    assert np.isfinite(float(g))


def test_heavy_ends_smoke_and_grad():
    out = heavy_ends_removal(5.0, 20.0, 653.15, 1.0)
    assert 0.0 < float(out["ccr_wt"]) < 5.0 and 0.0 < float(out["niv_wppm"]) < 20.0
    g = jax.grad(lambda T: heavy_ends_removal(5.0, 20.0, T, 1.0)["ccr_wt"])(653.15)
    assert np.isfinite(float(g))


def test_fcc_feed_from_hydrotreater():
    from difflow_refinery.fcc import FCCFeed
    f = fcc_feed_from_hydrotreater(VGO, ccr_wt=0.3, niv_wppm=1.0)
    assert isinstance(f, FCCFeed)
    assert float(f.sulfur) == pytest.approx(8e-4) and float(f.ccr) == pytest.approx(3e-3)
    assert 300.0 < float(f.MW) < 600.0 and 11.0 < float(f.Kw) < 13.0


def test_service_params():
    from difflow_refinery.hydrocracking.feed import VGO_PRETREAT_PARAMS
    assert vgo_params().kinetics == VGO_PRETREAT_PARAMS
    j = jet_params(lhsv=2.0)
    assert j.lhsv == 2.0 and j.P == 30e5 and len(j.T_in) == len(j.bed_fractions) == 1
