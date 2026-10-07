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


class TestRealisticPurity:
    """The trains used to report purity 1.0000: CEX removed >= 90% of every
    impurity, and the TFF steps, with no rejection set for impurities, washed
    90% of the HCP and aggregate out. With the typical clearances the mAb
    train lands at the reported end-of-process levels (HCP ~10 ppm against a
    <100 ppm target; aggregate below 1%)."""

    def test_mab_train_meets_typical_specs_not_perfection(self, harvest):
        res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
        assert 1.0 < float(res["hcp_ppm"]) < 100.0
        assert 0.001 < float(res["aggregate_fraction"]) < 0.01
        assert 0.98 < float(res["purity"]) < 0.999

    def test_uf_does_not_clear_impurities(self, harvest):
        res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(
            harvest, return_intermediates=True)
        before = res["intermediates"]["proa_eluate"]
        after = res["intermediates"]["tff1_concentrate"]
        for s in ("HCP", "aggregates"):
            assert float(after[f"F_{s}"]) / float(before[f"F_{s}"]) > 0.9

    def test_grad_purity_wrt_cex_aggregate_lrv(self, harvest):
        def purity(lrv):
            p = mAbDSPParams(
                species_order=SPECIES,
                cex_clearance={"HCP": 0.5, "DNA": 1.0, "aggregates": lrv},
            )
            return mAbDSPTrain(p)(harvest)["purity"]

        g = float(jax.grad(purity)(0.7))
        assert g > 0
        assert g == pytest.approx(float(central_difference(purity, 0.7)), rel=1e-5)


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

    @pytest.mark.parametrize("ph, expected", [(2.0, 0.964), (3.6, 0.98), (5.6, 0.98), (7.0, 0.98)])
    def test_low_ph_recovery_bounded(self, harvest, ph, expected):
        """Bio audit C9: 0.98 - 0.01*(3.6 - pH) gave 1.014 at pH 7."""
        train = ViralClearanceTrain(ViralClearanceParams(species_order=SPECIES, low_pH_value=ph))
        (out, loss), info = train.low_ph_inactivation(harvest)
        assert float(info["recovery"]) == pytest.approx(expected, rel=1e-12)
        assert float(out["F_mAb"]) <= 100.0
        assert float(loss["F_mAb"]) >= 0.0

    def test_vf_area_sets_reported_loading(self, harvest):
        """Bio audit C9: vf_area had no effect; it now sets the loading."""
        _, a = ViralClearanceTrain(ViralClearanceParams(vf_area_m2=1.0)).virus_filtration(harvest)
        _, b = ViralClearanceTrain(ViralClearanceParams(vf_area_m2=2.0)).virus_filtration(harvest)
        assert float(a["loading_g_per_m2"]) == pytest.approx(100.0)
        assert float(b["loading_g_per_m2"]) == pytest.approx(50.0)


def _assert_closes(feed, result):
    """product + every side stream = feed, species by species."""
    from difflow.streams import get_flows
    outs = [result["product"], *result["side_streams"].values()]
    for s, f in get_flows(feed).items():
        total = sum(float(get_flows(o).get(s, 0.0)) for o in outs)
        assert total == pytest.approx(float(f), rel=1e-10, abs=1e-12), s


class TestTrainsClose:
    """Bio audit (a)-(c): the trains returned only `product`, dropping the
    inner outlets (19.8% of a 100 g harvest in mAbDSPTrain) and the
    impurity every step cleared."""

    @pytest.mark.parametrize("scale", [1.0, 10.0])
    def test_mab_dsp(self, scale):
        harvest = make_stream({"mAb": 100.0 * scale, "HCP": 5.0, "DNA": 0.5,
                               "aggregates": 3.0}, 298.15, 101325.0)
        res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
        assert set(res["side_streams"]) == {
            "proa_waste", "tff1_permeate", "cex_waste", "aex_bound", "tff2_permeate"}
        _assert_closes(harvest, res)

    @pytest.mark.parametrize("kw", [{}, {"include_sec": True}, {"capture_type": "cex"},
                                    {"polish_steps": ["cex", "aex", "aex"]},
                                    {"include_viral_filtration": False}])
    def test_platform(self, harvest, kw):
        p = PlatformDSPParams(species_order=SPECIES, target_species="mAb", **kw)
        _assert_closes(harvest, PlatformDSP(p)(harvest))

    def test_viral_clearance(self, harvest):
        res = ViralClearanceTrain(ViralClearanceParams(species_order=SPECIES))(harvest)
        _assert_closes(harvest, res)
        assert float(res["side_streams"]["low_pH_loss"]["F_mAb"]) == pytest.approx(2.0)


class TestMabStepYields:
    """Bio audit (a): every step's yield is against its own inlet."""

    def test_step_yields_multiply_to_overall(self, harvest):
        res = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
        sy = res["step_yields"]
        assert set(sy) == {"proa", "tff1", "cex", "aex", "tff2"}
        # the CEX step is its own yield_factor, not CEX x TFF1 (0.8897)
        assert float(sy["cex"]) == pytest.approx(0.90, rel=1e-12)
        prod = 1.0
        for v in sy.values():
            prod *= float(v)
        assert prod == pytest.approx(float(res["overall_yield"]), rel=1e-12)

    @pytest.mark.parametrize("scale", [1.0, 10.0])
    def test_final_cf_is_a_volume_ratio(self, scale):
        """CF was final_concentration_g_L / mAb (g/L over g): 0.12 at 1000 g."""
        harvest = make_stream({"mAb": 100.0 * scale, "HCP": 5.0, "DNA": 0.5,
                               "aggregates": 3.0}, 298.15, 101325.0)
        p = mAbDSPParams(species_order=SPECIES, proa_column_volume=10.0 * scale,
                         cex_column_volume=20.0 * scale)
        res = mAbDSPTrain(p)(harvest, return_intermediates=True)
        aex_mab = float(res["intermediates"]["aex_product"]["F_mAb"])
        pool_L = 5.0 * 20.0 * scale
        assert float(res["final_tff_concentration_factor"]) == pytest.approx(
            pool_L / (aex_mab / 100.0), rel=1e-12)
        assert float(res["final_tff_concentration_factor"]) > 1.0
        # the product ends near the target; UF rejection 0.995 loses ~2.4%
        assert 95.0 < float(res["final_concentration_g_L"]) <= 100.0

    def test_tff_area_sets_process_time(self, harvest):
        a = mAbDSPTrain(mAbDSPParams(species_order=SPECIES, tff_area=5.0))(harvest)
        b = mAbDSPTrain(mAbDSPParams(species_order=SPECIES, tff_area=10.0))(harvest)
        for k in ("tff1", "tff2"):
            assert float(a["tff_process_time_h"][k]) == pytest.approx(
                2.0 * float(b["tff_process_time_h"][k]), rel=1e-12)

    def test_aex_column_volume_caps_impurity_binding(self, harvest):
        big = mAbDSPTrain(mAbDSPParams(species_order=SPECIES))(harvest)
        tiny = mAbDSPTrain(mAbDSPParams(species_order=SPECIES, aex_column_volume=1e-4))(harvest)
        assert float(tiny["hcp_ppm"]) > 5.0 * float(big["hcp_ppm"])


class TestPlatformOptions:
    """Bio audit (b) and C10: options that did nothing or did the wrong thing."""

    def test_mmc_capture_rejected(self):
        with pytest.raises(ValueError, match="mmc"):
            PlatformDSP(PlatformDSPParams(species_order=SPECIES, capture_type="mmc"))

    def test_unsupported_polish_rejected(self):
        with pytest.raises(ValueError, match="hic"):
            PlatformDSP(PlatformDSPParams(species_order=SPECIES, polish_steps=["cex", "hic"]))

    def test_include_viral_filtration_toggles_the_step(self, harvest):
        on = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb"))
        off = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb",
                                            include_viral_filtration=False))
        assert "viral_filtration" in on.list_steps()
        assert "viral_filtration" not in off.list_steps()
        r_on, r_off = on(harvest), off(harvest)
        assert float(r_on["step_yields"]["viral_filtration"]) == pytest.approx(0.97)
        assert float(r_on["overall_yield"]) == pytest.approx(
            0.97 * float(r_off["overall_yield"]), rel=1e-12)
        assert "viral_filtration_lrv" in r_on

    def test_target_yield_is_reported(self, harvest):
        lo = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb",
                                           target_yield=0.5))(harvest)
        hi = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb",
                                           target_yield=0.99))(harvest)
        assert bool(lo["meets_target_yield"]) and not bool(hi["meets_target_yield"])

    def test_tff_area_sets_uf_time(self, harvest):
        a = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb",
                                          tff_area=5.0))(harvest, uf_feed_volume_L=50.0)
        b = PlatformDSP(PlatformDSPParams(species_order=SPECIES, target_species="mAb",
                                          tff_area=10.0))(harvest, uf_feed_volume_L=50.0)
        assert float(a["uf_process_time_h"]) == pytest.approx(
            2.0 * float(b["uf_process_time_h"]), rel=1e-12)
