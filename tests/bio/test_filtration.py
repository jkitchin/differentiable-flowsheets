"""Tests for membrane filtration unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    Ultrafiltration,
    UltrafiltrationParams,
    Diafiltration,
    DiafiltrationParams,
    TFF,
    diavolumes_required,
    rejection_from_mw,
)
from difflow import make_stream, get_flows


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestRejectionFromMW:
    def test_high_mw_rejection(self):
        """High MW species should be highly rejected."""
        R = rejection_from_mw(MW=150.0, MWCO=30.0)  # mAb vs 30 kDa
        assert float(R) > 0.99

    def test_low_mw_permeation(self):
        """Low MW species should pass through."""
        R = rejection_from_mw(MW=1.0, MWCO=30.0)  # salt vs 30 kDa
        assert float(R) < 0.1

    def test_at_mwco(self):
        """At the MWCO, rejection is 90%, the nominal-rating definition.

        The curve used to give 0.5 here, which treated a 30 kDa membrane as
        letting half of a 30 kDa solute through and left a 150 kDa mAb at
        R = 0.992 (bio audit C4).
        """
        R = rejection_from_mw(MW=30.0, MWCO=30.0)
        assert float(R) == pytest.approx(0.9, rel=1e-12)

    def test_five_times_mwco_is_held(self):
        """The 3 to 5x selection rule: 5 x MWCO is rejected at > 99.9%."""
        assert float(rejection_from_mw(MW=150.0, MWCO=30.0)) > 0.999


class TestDiavolumesRequired:
    def test_diavolumes_calculation(self):
        """Test diavolume calculation for buffer exchange."""
        # Remove 99% of a fully permeable species (R=0)
        n_dv = diavolumes_required(
            initial_conc=jnp.array(100.0),
            target_conc=jnp.array(1.0),
            rejection=jnp.array(0.0),
        )
        # -ln(0.01) ≈ 4.6
        assert float(n_dv) == pytest.approx(4.6, rel=0.1)

    def test_higher_rejection_needs_more_dv(self):
        """Species with higher rejection need more diavolumes."""
        n_dv_low_R = diavolumes_required(jnp.array(100.0), jnp.array(10.0), jnp.array(0.0))
        n_dv_high_R = diavolumes_required(jnp.array(100.0), jnp.array(10.0), jnp.array(0.5))

        assert float(n_dv_high_R) > float(n_dv_low_R)


class TestUltrafiltration:
    @pytest.fixture
    def uf_params(self):
        """UF parameters for mAb concentration."""
        return UltrafiltrationParams(
            membrane_area=1.0,
            MWCO=30.0,
            rejection={"mAb": 0.999, "HCP": 0.3, "buffer_salt": 0.0},
            Lp=50.0,
        )

    def test_uf_creation(self, uf_params):
        """Test UF can be created."""
        uf = Ultrafiltration(uf_params)
        assert uf is not None

    def test_uf_concentrates_protein(self, uf_params):
        """Test UF concentrates the target protein."""
        uf = Ultrafiltration(uf_params)

        feed = make_stream(
            {"mAb": 10.0, "HCP": 1.0, "buffer_salt": 100.0},
            T=300.0, P=101325.0
        )

        (retentate, permeate), info = uf(feed, concentration_factor=5.0)

        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)

        # mAb should be mostly in retentate (high rejection)
        mAb_recovery = float(ret_flows["mAb"]) / 10.0
        assert mAb_recovery > 0.95

        # Salt should be mostly in permeate (no rejection)
        salt_in_permeate = float(perm_flows["buffer_salt"])
        assert salt_in_permeate > 50.0

    def test_uf_mass_balance(self, uf_params):
        """Test mass balance is preserved."""
        uf = Ultrafiltration(uf_params)

        feed = make_stream(
            {"mAb": 10.0, "HCP": 1.0, "buffer_salt": 100.0},
            T=300.0, P=101325.0
        )

        (retentate, permeate), info = uf(feed, concentration_factor=5.0)

        feed_flows = get_flows(feed)
        ret_flows = get_flows(retentate)
        perm_flows = get_flows(permeate)

        for species in feed_flows:
            total_out = float(ret_flows[species]) + float(perm_flows[species])
            assert total_out == pytest.approx(float(feed_flows[species]), rel=0.01)

    @pytest.mark.release
    def test_uf_differentiability(self, uf_params):
        """Test UF is differentiable w.r.t. concentration factor."""
        def product_recovery(CF):
            uf = Ultrafiltration(uf_params)
            feed = make_stream({"mAb": 10.0, "buffer_salt": 100.0}, T=300.0, P=101325.0)
            (retentate, _), _ = uf(feed, concentration_factor=CF)
            return retentate["F_mAb"]

        grad_CF = jax.grad(product_recovery)(jnp.array(5.0))

        # Gradient should exist and be finite
        assert jnp.isfinite(grad_CF)


class TestDiafiltration:
    @pytest.fixture
    def df_params(self):
        """DF parameters for buffer exchange."""
        return DiafiltrationParams(
            membrane_area=1.0,
            MWCO=30.0,
            rejection={"mAb": 0.999, "old_buffer": 0.0, "new_buffer": 0.0},
            Lp=50.0,
        )

    def test_df_creation(self, df_params):
        """Test DF can be created."""
        df = Diafiltration(df_params)
        assert df is not None

    def test_df_buffer_exchange(self, df_params):
        """Test DF exchanges buffer while retaining protein."""
        df = Diafiltration(df_params)

        feed = make_stream(
            {"mAb": 10.0, "old_buffer": 100.0},
            T=300.0, P=101325.0
        )

        buffer = make_stream(
            {"new_buffer": 100.0},
            T=300.0, P=101325.0
        )

        (retentate, permeate), info = df(feed, buffer, n_diavolumes=5.0)

        ret_flows = get_flows(retentate)

        # mAb should be retained
        mAb_remaining = float(ret_flows["mAb"]) / 10.0
        assert mAb_remaining > 0.99

        # Old buffer should be washed out
        old_buffer_remaining = float(ret_flows["old_buffer"]) / 100.0
        assert old_buffer_remaining < 0.01

        # New buffer should be present
        assert float(ret_flows["new_buffer"]) > 0

    def test_df_exchange_efficiency(self, df_params):
        """Test exchange efficiency calculation."""
        df = Diafiltration(df_params)

        feed = make_stream({"mAb": 10.0, "old_buffer": 100.0}, T=300.0, P=101325.0)
        buffer = make_stream({"new_buffer": 100.0}, T=300.0, P=101325.0)

        (_, _), info = df(feed, buffer, n_diavolumes=5.0)

        # Exchange efficiency should be high for permeable species
        assert info["exchange_efficiency"]["old_buffer"] > 0.99


class TestTFF:
    def test_tff_creation(self):
        """Test TFF system can be created."""
        tff = TFF(
            membrane_area=1.0,
            MWCO=30.0,
            rejection={"mAb": 0.999, "salt": 0.0},
        )
        assert tff is not None

    def test_uf_df_uf_process(self):
        """Test complete UF/DF/UF process."""
        tff = TFF(
            membrane_area=1.0,
            rejection={"mAb": 0.999, "old_salt": 0.0, "new_salt": 0.0},
        )

        feed = make_stream(
            {"mAb": 1.0, "old_salt": 100.0},
            T=300.0, P=101325.0
        )

        buffer = make_stream(
            {"new_salt": 50.0},
            T=300.0, P=101325.0
        )

        (final_product, *_permeates), info = tff.uf_df_uf(
            feed,
            buffer,
            CF_initial=2.0,
            n_diavolumes=5.0,
            CF_final=5.0,
        )

        final_flows = get_flows(final_product)

        # mAb should be concentrated
        # Final recovery should be > 90% of initial
        assert float(final_flows["mAb"]) > 0.9

        # Old salt should be washed out
        assert float(final_flows["old_salt"]) < 0.1

        # Process info should contain all steps
        assert "step1_uf" in info
        assert "step2_df" in info
        assert "step3_uf" in info


class TestConcentrationPolarizationFeedback:
    """Issue #99: flux should fall as the batch concentrates (feedback)."""

    def test_flux_decreases_with_cf(self):
        uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=1.0, Lp=50.0, sigma=1000.0,
            rejection={"mAb": 1.0},
        ))
        feed = make_stream({"mAb": 10.0, "buffer": 90.0}, T=300.0, P=101325.0)
        (_, _), lo = uf(feed, concentration_factor=2.0, TMP=1.0, feed_concentration=10.0)
        (_, _), hi = uf(feed, concentration_factor=5.0, TMP=1.0, feed_concentration=10.0)
        # Higher CF -> higher osmotic back-pressure -> lower flux
        assert float(hi["flux"]) < float(lo["flux"])
        assert float(hi["osmotic_pressure_bar"]) > float(lo["osmotic_pressure_bar"])

    def test_backward_compat_no_feed_concentration(self):
        uf = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0))
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)
        (_, _), info = uf(feed, concentration_factor=2.0, TMP=1.0)
        assert "osmotic_pressure_bar" not in info
        assert float(info["flux"]) > 0.0

    def test_tff_forwards_k_mass_and_sigma(self):
        """TFF must forward k_mass/sigma to its UF and DF stages (#99)."""
        tff = TFF(membrane_area=1.0, k_mass=2e-6, sigma=3000.0, fouling_coefficient=0.01)
        assert float(tff.uf.params.k_mass) == pytest.approx(2e-6)
        assert float(tff.uf.params.sigma) == pytest.approx(3000.0)
        assert float(tff.df.params.sigma) == pytest.approx(3000.0)
        assert float(tff.uf.params.fouling_coefficient) == pytest.approx(0.01)


class TestMembraneFoulingFluxDecline:
    """Issue #155: fouling should reduce flux with permeate throughput."""

    def test_flux_declines_with_fouling(self):
        clean = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, fouling_coefficient=0.0))
        fouled = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, fouling_coefficient=0.05))
        feed = make_stream({"mAb": 10.0, "buffer": 90.0}, T=300.0, P=101325.0)
        (_, _), c = clean(feed, concentration_factor=5.0, TMP=1.0)
        (_, _), f = fouled(feed, concentration_factor=5.0, TMP=1.0)
        assert float(f["flux"]) < float(c["flux"])
        assert float(f["fouling_factor"]) > 1.0
        assert float(c["fouling_factor"]) == pytest.approx(1.0)

    def test_more_permeate_more_fouling(self):
        uf = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, fouling_coefficient=0.02))
        feed = make_stream({"mAb": 10.0, "buffer": 90.0}, T=300.0, P=101325.0)
        (_, _), lo = uf(feed, concentration_factor=2.0, TMP=1.0)   # less permeate
        (_, _), hi = uf(feed, concentration_factor=10.0, TMP=1.0)  # more permeate
        assert float(hi["fouling_factor"]) > float(lo["fouling_factor"])
        assert float(hi["flux"]) < float(lo["flux"])


class TestMembraneRetentionFollowsMWCO:
    """Bio audit C4: with no ``rejection`` given, the membranes passed the mAb.

    ``rejection`` defaulted to {} and an unlisted species to R = 0, so a
    30 kDa UF at CF 5 put 8 of 10 g of a 150 kDa mAb in the permeate and
    5 diavolumes of DF washed out 9.93 of 10 g; MWCO did nothing.
    """

    @staticmethod
    def _uf_feed():
        return make_stream({"mAb": 10.0, "HCP": 1.0, "buffer_salt": 100.0},
                           T=300.0, P=101325.0)

    def test_uf_retains_mab_by_default(self):
        uf = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0))
        (ret, perm), info = uf(self._uf_feed(), concentration_factor=5.0)
        assert float(ret["F_mAb"]) / 10.0 > 0.99
        # The buffer salt (no molecular weight anywhere) still passes freely.
        assert float(ret["F_buffer_salt"]) == pytest.approx(20.0, rel=1e-9)
        assert float(info["rejection"]["mAb"]) > 0.999

    def test_df_retains_mab_by_default(self):
        df = Diafiltration(DiafiltrationParams(membrane_area=1.0))
        feed = make_stream({"mAb": 10.0, "old_buffer": 100.0}, T=300.0, P=101325.0)
        buf = make_stream({"new_buffer": 550.0}, T=300.0, P=101325.0)
        (ret, perm), info = df(feed, buf)
        assert float(ret["F_mAb"]) / 10.0 > 0.99
        assert float(ret["F_old_buffer"]) < 0.01 * 100.0

    def test_mwco_moves_the_split(self):
        """A membrane far above the mAb's size passes it; far below holds it."""
        feed = self._uf_feed()
        (tight, _), _ = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, MWCO=10.0))(
            feed, concentration_factor=5.0)
        (loose, _), _ = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, MWCO=1000.0))(
            feed, concentration_factor=5.0)
        assert float(tight["F_mAb"]) > 9.99
        assert float(loose["F_mAb"]) < 3.0

    def test_explicit_rejection_still_overrides(self):
        uf = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, rejection={"mAb": 0.0}))
        (ret, _), _ = uf(self._uf_feed(), concentration_factor=5.0)
        assert float(ret["F_mAb"]) == pytest.approx(2.0, rel=1e-9)

    def test_molecular_weights_parameter(self):
        uf = Ultrafiltration(UltrafiltrationParams(
            membrane_area=1.0, molecular_weights={"buffer_salt": 500.0}))
        (ret, _), _ = uf(self._uf_feed(), concentration_factor=5.0)
        assert float(ret["F_buffer_salt"]) > 99.0

    def test_membrane_area_sets_process_time(self):
        feed = self._uf_feed()
        (_, _), big = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0))(
            feed, concentration_factor=5.0, feed_volume=100.0)
        (_, _), small = Ultrafiltration(UltrafiltrationParams(membrane_area=0.1))(
            feed, concentration_factor=5.0, feed_volume=100.0)
        assert float(small["process_time_h"]) == pytest.approx(
            10.0 * float(big["process_time_h"]), rel=1e-12)
        # 80 L of permeate at the reported flux through 1 m^2
        assert float(big["process_time_h"]) == pytest.approx(
            80.0 / float(big["flux"]), rel=1e-12)

    def test_retention_is_differentiable_in_mwco(self):
        def kept(mwco):
            uf = Ultrafiltration(UltrafiltrationParams(membrane_area=1.0, MWCO=mwco))
            (ret, _), _ = uf(self._uf_feed(), concentration_factor=5.0)
            return ret["F_mAb"]
        g = jax.grad(kept)(jnp.asarray(100.0))
        assert jnp.isfinite(g) and float(g) < 0.0


class TestDiafiltrationBufferBalance:
    """Bio audit C4: the DF buffer stream entered only as a composition.

    100 units of buffer fed became 5 x 110 = 550 in the balance; now the
    buffer fed is the buffer consumed unless N is fixed explicitly.
    """

    @staticmethod
    def _run(**kw):
        df = Diafiltration(DiafiltrationParams(membrane_area=1.0))
        feed = make_stream({"mAb": 10.0, "old_buffer": 100.0}, T=300.0, P=101325.0)
        buf = make_stream({"new_buffer": 220.0}, T=300.0, P=101325.0)
        (ret, perm), info = df(feed, buf, **kw)
        return feed, buf, ret, perm, info

    def test_buffer_stream_closes_the_balance(self):
        feed, buf, ret, perm, info = self._run()
        assert float(info["n_diavolumes"]) == pytest.approx(2.0, rel=1e-12)
        for s in ("mAb", "old_buffer", "new_buffer"):
            fin = float(get_flows(feed).get(s, 0.0)) + float(get_flows(buf).get(s, 0.0))
            fout = float(get_flows(ret)[s]) + float(get_flows(perm)[s])
            assert fout == pytest.approx(fin, rel=1e-12), s

    def test_explicit_diavolumes_reports_buffer_consumed(self):
        feed, buf, ret, perm, info = self._run(n_diavolumes=5.0)
        consumed = get_flows(info["buffer_consumed"])
        assert float(consumed["new_buffer"]) == pytest.approx(550.0, rel=1e-12)
        for s in ("mAb", "old_buffer", "new_buffer"):
            fin = float(get_flows(feed).get(s, 0.0)) + float(consumed.get(s, 0.0))
            fout = float(get_flows(ret)[s]) + float(get_flows(perm)[s])
            assert fout == pytest.approx(fin, rel=1e-12), s

    def test_retained_buffer_species_wash_in_is_continuous_in_R(self):
        """C_buf V (1 - exp(-sN))/s, continuous through R -> 1 (no jump at 0.99)."""
        import math
        feed = make_stream({"mAb": 10.0}, T=300.0, P=101325.0)
        buf = make_stream({"excipient": 50.0}, T=300.0, P=101325.0)
        for R in (0.985, 0.995):
            df = Diafiltration(DiafiltrationParams(
                membrane_area=1.0, rejection={"mAb": 1.0, "excipient": R}))
            (ret, _), _ = df(feed, buf)
            x = (1.0 - R) * 5.0  # N = 50 / 10
            assert float(ret["F_excipient"]) == pytest.approx(
                50.0 * (1 - math.exp(-x)) / x, rel=1e-10)


class TestTFFReturnsPermeates:
    """Bio audit (d): uf_df_uf dropped its three permeates and the mAb in them."""

    def test_outputs_close_against_inlet_and_buffer(self):
        t = TFF(membrane_area=5.0, rejection={"mAb": 0.995})
        feed = make_stream({"mAb": 100.0, "H2O": 1000.0}, 298.15, 101325.0)
        buf = make_stream({"H2O": 100.0}, 298.15, 101325.0)
        (ret, p1, p2, p3), info = t.uf_df_uf(feed, buf, 5.0, 5.0, 2.0)
        consumed = get_flows(info["buffer_consumed"])
        for s in ("mAb", "H2O"):
            fin = float(get_flows(feed)[s]) + float(consumed.get(s, 0.0))
            fout = sum(float(get_flows(x)[s]) for x in (ret, p1, p2, p3))
            assert fout == pytest.approx(fin, rel=1e-12), s
        assert sum(float(x["F_mAb"]) for x in (p1, p2, p3)) > 3.0
