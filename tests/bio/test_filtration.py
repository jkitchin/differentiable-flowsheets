"""Smoke tests for membrane filtration unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    Ultrafiltration,
    UltrafiltrationParams,
    Diafiltration,
    DiafiltrationParams,
    TFF,
)
from difflow import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _closes(ins, outs):
    species = {s for x in ins for s in get_flows(x)}
    for s in species:
        fin = sum(float(get_flows(x).get(s, 0.0)) for x in ins)
        fout = sum(float(get_flows(x).get(s, 0.0)) for x in outs)
        assert fout == pytest.approx(fin, rel=1e-8), s


def test_ultrafiltration_runs_and_closes():
    feed = make_stream({"mAb": 10.0, "HCP": 1.0, "salt": 100.0}, T=300.0, P=101325.0)
    uf = Ultrafiltration(UltrafiltrationParams(
        membrane_area=1.0, MWCO=30.0, rejection={"mAb": 0.999, "HCP": 0.3, "salt": 0.0}))
    (ret, perm), info = uf(feed, concentration_factor=5.0)
    _closes([feed], [ret, perm])


def test_ultrafiltration_grad_is_finite():
    feed = make_stream({"mAb": 10.0, "salt": 100.0}, T=300.0, P=101325.0)
    def f(cf):
        (ret, _), _ = Ultrafiltration(UltrafiltrationParams(
            membrane_area=1.0, rejection={"mAb": 0.99, "salt": 0.0}))(feed, concentration_factor=cf)
        return ret["F_mAb"]
    assert jnp.isfinite(jax.grad(f)(5.0))


def test_diafiltration_runs_and_closes():
    feed = make_stream({"mAb": 10.0, "old_buffer": 100.0}, T=300.0, P=101325.0)
    buf = make_stream({"new_buffer": 220.0}, T=300.0, P=101325.0)
    (ret, perm), info = Diafiltration(DiafiltrationParams(membrane_area=1.0))(feed, buf)
    _closes([feed, buf], [ret, perm])


def test_tff_uf_df_uf_runs_and_closes():
    feed = make_stream({"mAb": 100.0, "H2O": 1000.0}, 298.15, 101325.0)
    buf = make_stream({"H2O": 100.0}, 298.15, 101325.0)
    outs, info = TFF(membrane_area=5.0, rejection={"mAb": 0.995}).uf_df_uf(feed, buf, 5.0, 5.0, 2.0)
    assert {"step1_uf", "step2_df", "step3_uf"} <= set(info)
    _closes([feed, info["buffer_consumed"]], outs)
