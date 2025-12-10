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
        """At MWCO, rejection should be ~50%."""
        R = rejection_from_mw(MW=30.0, MWCO=30.0)
        assert 0.3 < float(R) < 0.7


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

        final_product, info = tff.uf_df_uf(
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
