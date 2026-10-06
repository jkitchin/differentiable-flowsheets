"""Tests for the packaged bio flowsheet trains.

Issue #360: mAbDSPTrain, PlatformDSP and ViralClearanceTrain promise to be
fully differentiable, but converted their results with float() (and the
virus filter branched on a traced comparison with if/elif), so jax.grad,
jit and vmap all raised. These tests take gradients through each train and
check them against central finite differences.

The trains also used to pass each column's volume as its *load volume*, so
only about a tenth of the batch was ever loaded and the overall yield came
out near 7%, growing linearly with column size. Each step now loads the
whole batch and the column volume sets its capacity.
"""

import jax
import jax.numpy as jnp
import pytest

from difflow import make_stream
from difflow_bio.flowsheets.mab_dsp import mAbDSPTrain, mAbDSPParams
from difflow_bio.flowsheets.platform import PlatformDSP, PlatformDSPParams
from difflow_bio.flowsheets.viral_clearance import (
    ViralClearanceTrain,
    ViralClearanceParams,
)

jax.config.update("jax_enable_x64", True)

SPECIES = ["mAb", "HCP", "DNA", "aggregates"]


@pytest.fixture
def harvest():
    return make_stream(
        {"mAb": 100.0, "HCP": 5.0, "DNA": 0.5, "aggregates": 3.0},
        298.15, 101325.0,
    )


def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


class TestMabDSPTrain:
    def test_grad_overall_yield_matches_fd(self, harvest):
        # A 2 L Protein A column is capacity-limited for 100 g of mAb, so
        # the yield depends on its volume.
        def y(cv):
            p = mAbDSPParams(species_order=SPECIES, proa_column_volume=cv)
            return mAbDSPTrain(p)(harvest)["overall_yield"]

        g = float(jax.grad(y)(2.0))
        assert g > 0
        assert g == pytest.approx(float(central_difference(y, 2.0)), rel=1e-6)

    def test_whole_batch_is_loaded(self, harvest):
        # An ample column loses only its elution yield; the overall yield
        # no longer scales with column volume.
        def run(cv):
            p = mAbDSPParams(species_order=SPECIES, proa_column_volume=cv)
            return mAbDSPTrain(p)(harvest)

        r10, r20 = run(10.0), run(20.0)
        assert float(r10["step_yields"]["proa"]) == pytest.approx(0.95)
        assert float(r10["overall_yield"]) == pytest.approx(
            float(r20["overall_yield"])
        )
        assert float(r10["overall_yield"]) > 0.75

    def test_undersized_column_is_capacity_limited(self, harvest):
        # 1 L at q_max 35 g/L, 1% breakthrough limit: 34.65 g binds of 100 g
        p = mAbDSPParams(species_order=SPECIES, proa_column_volume=1.0)
        res = mAbDSPTrain(p)(harvest)
        assert float(res["step_yields"]["proa"]) == pytest.approx(
            35.0 * 0.99 * 1.0 * 0.95 / 100.0
        )

    def test_grad_purity_matches_fd(self, harvest):
        def purity(cex_yield):
            p = mAbDSPParams(species_order=SPECIES, cex_yield=cex_yield)
            return mAbDSPTrain(p)(harvest)["purity"]

        assert float(jax.grad(purity)(0.9)) == pytest.approx(
            float(central_difference(purity, 0.9)), rel=1e-5
        )

    def test_jit_and_vmap(self, harvest):
        def y(cv):
            p = mAbDSPParams(species_order=SPECIES, proa_column_volume=cv)
            return mAbDSPTrain(p)(harvest)["overall_yield"]

        assert float(jax.jit(y)(2.0)) == pytest.approx(float(y(2.0)))
        cvs = jnp.array([1.0, 2.0, 20.0])
        assert jnp.allclose(jax.vmap(y)(cvs), jnp.array([y(c) for c in cvs]))

    def test_step_yields_are_arrays(self, harvest):
        res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
        for v in res["step_yields"].values():
            assert isinstance(v, jax.Array)


class TestPlatformDSP:
    def test_grad_matches_fd_and_jits(self, harvest):
        def y(cv):
            p = PlatformDSPParams(
                species_order=SPECIES, target_species="mAb",
                column_volumes={"capture": cv, "cex": 15.0, "aex": 12.0},
            )
            return PlatformDSP(p)(harvest)["overall_yield"]

        g = float(jax.grad(y)(2.0))     # capacity-limited capture column
        assert g > 0
        assert g == pytest.approx(float(central_difference(y, 2.0)), rel=1e-6)
        assert float(jax.jit(y)(2.0)) == pytest.approx(float(y(2.0)))

    def test_with_sec_step(self, harvest):
        # SEC returns three streams; the train used to unpack two and raise.
        p = PlatformDSPParams(
            species_order=SPECIES, target_species="mAb", include_sec=True,
        )
        res = PlatformDSP(p)(harvest)
        assert float(res["step_yields"]["sec"]) == pytest.approx(0.95)
        assert float(res["step_yields"]["capture"]) == pytest.approx(0.95)


class TestViralClearanceTrain:
    def test_grad_lrv_wrt_ph_matches_fd(self, harvest):
        def lrv(ph):
            p = ViralClearanceParams(species_order=SPECIES, low_pH_value=ph)
            return ViralClearanceTrain(p)(harvest)["total_lrv"]

        assert float(jax.grad(lrv)(3.6)) == pytest.approx(
            float(central_difference(lrv, 3.6)), rel=1e-6
        )
        assert float(jax.jit(lrv)(3.6)) == pytest.approx(float(lrv(3.6)))

    @pytest.mark.parametrize(
        "pore, expected",
        [
            (10.0, 6.0),                     # PPV 22 nm, ratio 2.2: complete removal
            (20.0, 4.0 + 2.0 * (1.1 - 1.0)),  # ratio 1.1: partial retention
            (30.0, 2.0 * 22.0 / 30.0),        # ratio < 1: passes the filter
        ],
    )
    def test_filter_regimes_unchanged(self, harvest, pore, expected):
        # The if/elif became jnp.where; each regime must give the same LRV.
        train = ViralClearanceTrain(
            ViralClearanceParams(species_order=SPECIES, vf_membrane_pore_nm=pore)
        )
        _, info = train.virus_filtration(harvest, target_virus="PPV")
        assert float(info["lrv"]) == pytest.approx(expected)

    def test_grad_lrv_wrt_pore_traced(self, harvest):
        def lrv(pore):
            p = ViralClearanceParams(species_order=SPECIES, vf_membrane_pore_nm=pore)
            return ViralClearanceTrain(p)(harvest)["total_lrv"]

        # in the partial-retention regime d(LRV)/d(pore) = -2 * 22 / pore^2
        assert float(jax.grad(lrv)(20.0)) == pytest.approx(-2.0 * 22.0 / 400.0)
