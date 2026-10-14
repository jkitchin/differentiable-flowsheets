"""Smoke tests for chromatography unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    ProteinAChromatography,
    ProteinAParams,
    IonExchangeChromatography,
    IEXParams,
    SizeExclusionChromatography,
    SECParams,
)
from difflow import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _closes(feed, outs):
    for s, f in get_flows(feed).items():
        total = sum(float(get_flows(o).get(s, 0.0)) for o in outs)
        assert total == pytest.approx(float(f), rel=1e-8), s


def test_protein_a_runs_and_closes():
    feed = make_stream({"mAb": 10.0, "HCP": 100.0, "DNA": 1.0}, T=300.0, P=101325.0)
    col = ProteinAChromatography(ProteinAParams(
        column_volume=1.0, q_max=35.0, K_d=0.1, target_species="mAb",
        yield_factor=0.95, impurity_clearance={"HCP": 2.0, "DNA": 3.0}))
    (product, waste), info = col(feed)
    assert jnp.isfinite(info["yield"]) and jnp.isfinite(info["purity"])
    _closes(feed, [product, waste])


def test_protein_a_grad_is_finite():
    feed = make_stream({"mAb": 10.0, "HCP": 10.0}, T=300.0, P=101325.0)
    def f(cv):
        (product, _), _ = ProteinAChromatography(ProteinAParams(column_volume=cv))(feed)
        return product["F_mAb"]
    assert jnp.isfinite(jax.grad(f)(0.2))


def test_iex_runs_and_closes():
    feed = make_stream({"mAb": 10.0, "HCP": 1.0, "aggregates": 0.5}, T=300.0, P=101325.0)
    col = IonExchangeChromatography(IEXParams(
        column_volume=1.0, mode="bind_elute", target_species="mAb",
        selectivity={"mAb": 0.9, "HCP": 0.3, "aggregates": 0.95}, yield_factor=0.9))
    (product, waste), info = col(feed)
    assert jnp.isfinite(info["yield"])
    _closes(feed, [product, waste])


def test_sec_runs_and_closes():
    feed = make_stream({"mAb": 95.0, "aggregates": 3.0, "fragments": 2.0}, T=300.0, P=101325.0)
    col = SizeExclusionChromatography(SECParams(
        column_volume=1.0, target_species="mAb", aggregate_species="aggregates",
        fragment_species="fragments", yield_factor=0.95))
    outs, info = col(feed)
    assert len(outs) == 3
    _closes(feed, outs)


def test_load_volume_without_feed_volume_raises():
    feed = make_stream({"mAb": 10.0, "HCP": 10.0}, T=300.0, P=101325.0)
    with pytest.raises(ValueError, match="feed_volume"):
        ProteinAChromatography(ProteinAParams(column_volume=1.0))(feed, load_volume=10.0)
