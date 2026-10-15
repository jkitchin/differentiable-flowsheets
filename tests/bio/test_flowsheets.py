"""Smoke tests for the packaged bio flowsheet trains."""

import jax
import jax.numpy as jnp
import pytest

from difflow import make_stream, get_flows
from difflow_bio.flowsheets.mab_dsp import mAbDSPTrain, mAbDSPParams
from difflow_bio.flowsheets.platform import PlatformDSP, PlatformDSPParams
from difflow_bio.flowsheets.viral_clearance import ViralClearanceTrain, ViralClearanceParams

jax.config.update("jax_enable_x64", True)

SPECIES = ["mAb", "HCP", "DNA", "aggregates"]


@pytest.fixture
def harvest():
    return make_stream({"mAb": 100.0, "HCP": 5.0, "DNA": 0.5, "aggregates": 3.0},
                       298.15, 101325.0)


def _assert_closes(feed, result):
    """product + every side stream = feed, species by species."""
    outs = [result["product"], *result["side_streams"].values()]
    for s, f in get_flows(feed).items():
        total = sum(float(get_flows(o).get(s, 0.0)) for o in outs)
        assert total == pytest.approx(float(f), rel=1e-10, abs=1e-12), s


def test_mab_dsp_runs_and_closes(harvest):
    res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
    assert jnp.isfinite(res["overall_yield"]) and jnp.isfinite(res["purity"])
    _assert_closes(harvest, res)


def test_mab_dsp_grad_is_finite(harvest):
    def purity(y):
        return mAbDSPTrain(mAbDSPParams(species_order=SPECIES, cex_yield=y))(harvest)["purity"]
    assert jnp.isfinite(jax.grad(purity)(0.9))


def test_platform_runs_and_closes(harvest):
    res = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb"))(harvest)
    assert jnp.isfinite(res["overall_yield"])
    _assert_closes(harvest, res)


def test_viral_clearance_runs_and_closes(harvest):
    res = ViralClearanceTrain(ViralClearanceParams(species_order=SPECIES))(harvest)
    assert jnp.isfinite(res["total_lrv"])
    _assert_closes(harvest, res)


def test_platform_rejects_unsupported_steps():
    with pytest.raises(ValueError, match="mmc"):
        PlatformDSP(PlatformDSPParams(species_order=SPECIES, capture_type="mmc"))
    with pytest.raises(ValueError, match="hic"):
        PlatformDSP(PlatformDSPParams(species_order=SPECIES, polish_steps=["cex", "hic"]))
