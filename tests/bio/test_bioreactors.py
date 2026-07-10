"""Tests for bioreactor unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    ContinuousBioreactor,
    FedBatchBioreactor,
    BioreactorParams,
    FedBatchParams,
    monod_kinetics,
    substrate_inhibition_kinetics,
)
from difflow import make_stream, get_flows


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestMonodKinetics:
    def test_monod_at_low_substrate(self):
        """At low S, μ ≈ μ_max * S / (K_s + S)."""
        params = {"mu_max": jnp.array(0.5), "K_s": jnp.array(1.0)}
        S = jnp.array(0.1)  # S << K_s
        mu = monod_kinetics(S, params)
        # Exact Monod: 0.5 * 0.1 / (1.0 + 0.1) = 0.04545...
        expected = 0.5 * 0.1 / (1.0 + 0.1)
        assert float(mu) == pytest.approx(expected, rel=1e-6)

    def test_monod_at_high_substrate(self):
        """At high S, μ ≈ μ_max (zero order)."""
        params = {"mu_max": jnp.array(0.5), "K_s": jnp.array(1.0)}
        S = jnp.array(100.0)  # S >> K_s
        mu = monod_kinetics(S, params)
        assert float(mu) == pytest.approx(0.5, rel=0.01)

    def test_monod_at_Ks(self):
        """At S = K_s, μ = μ_max / 2."""
        params = {"mu_max": jnp.array(0.5), "K_s": jnp.array(1.0)}
        S = jnp.array(1.0)
        mu = monod_kinetics(S, params)
        assert float(mu) == pytest.approx(0.25, rel=1e-6)

    def test_substrate_inhibition(self):
        """Test substrate inhibition reduces growth at high S."""
        params = {"mu_max": jnp.array(0.5), "K_s": jnp.array(1.0), "K_i": jnp.array(10.0)}

        mu_low = substrate_inhibition_kinetics(jnp.array(1.0), params)
        mu_high = substrate_inhibition_kinetics(jnp.array(50.0), params)

        # At high S, inhibition should reduce growth rate
        assert float(mu_high) < float(mu_low)


class TestContinuousBioreactor:
    @pytest.fixture
    def chemostat_params(self):
        """Basic chemostat parameters."""
        return BioreactorParams(
            V=jnp.array(10.0),  # 10 L
            Y_xs=jnp.array(0.5),  # g cells / g substrate
            kinetic_fn=monod_kinetics,
            kinetic_params={"mu_max": jnp.array(0.4), "K_s": jnp.array(0.5)},
            k_d=jnp.array(0.01),
            alpha=jnp.array(0.1),  # Growth-associated product
            species_order=["cells", "substrate", "product"],
        )

    def test_chemostat_creation(self, chemostat_params):
        """Test chemostat can be created."""
        chemostat = ContinuousBioreactor(chemostat_params)
        assert chemostat is not None

    def test_chemostat_steady_state(self, chemostat_params):
        """Test chemostat reaches reasonable steady state."""
        chemostat = ContinuousBioreactor(chemostat_params)

        # Feed stream: sterile medium with 20 g/L glucose
        # At D=0.2 h⁻¹ and F=2 L/h, flows are in g/h
        F = 2.0  # L/h
        D = F / 10.0  # 0.2 h⁻¹

        # Feed: 20 g/L * 2 L/h = 40 g/h substrate
        feed = make_stream(
            {"cells": 0.0, "substrate": 40.0, "product": 0.0},
            T=310.0, P=101325.0
        )

        outlet, info = chemostat(feed, D=D)

        # Check that cells grew
        assert float(info["X"]) > 0

        # Check substrate was consumed
        assert float(info["S"]) < 20.0

        # Check product was formed
        assert float(info["P"]) >= 0

        # Check growth rate is positive
        assert float(info["mu"]) > 0

    def test_chemostat_washout(self, chemostat_params):
        """At D > μ_max, cells should wash out."""
        chemostat = ContinuousBioreactor(chemostat_params)

        # High dilution rate (above μ_max = 0.4)
        D = 0.5
        F = D * 10.0  # 5 L/h

        feed = make_stream(
            {"cells": 0.0, "substrate": 100.0, "product": 0.0},
            T=310.0, P=101325.0
        )

        outlet, info = chemostat(feed, D=D)

        # Cell concentration should be very low at washout
        # (near zero, though numerical solver may not reach exactly zero)
        assert float(info["X"]) < 1.0

    def test_chemostat_differentiability(self, chemostat_params):
        """Test that chemostat is differentiable w.r.t. dilution rate."""
        def cell_productivity(D):
            chemostat = ContinuousBioreactor(chemostat_params)
            feed = make_stream(
                {"cells": 0.0, "substrate": 40.0, "product": 0.0},
                T=310.0, P=101325.0
            )
            outlet, info = chemostat(feed, D=D)
            return info["X"] * D  # Cell productivity = X * D

        # Compute gradient
        grad_D = jax.grad(cell_productivity)(jnp.array(0.2))

        # Gradient should exist and be finite
        assert jnp.isfinite(grad_D)


class TestFedBatchBioreactor:
    @pytest.fixture
    def fedbatch_params(self):
        """Basic fed-batch parameters."""
        return FedBatchParams(
            V0=jnp.array(5.0),  # 5 L initial
            Y_xs=jnp.array(0.5),
            kinetic_fn=monod_kinetics,
            kinetic_params={"mu_max": jnp.array(0.4), "K_s": jnp.array(0.5)},
            alpha=jnp.array(0.1),
        )

    def test_fedbatch_creation(self, fedbatch_params):
        """Test fed-batch can be created."""
        fedbatch = FedBatchBioreactor(fedbatch_params)
        assert fedbatch is not None

    def test_batch_mode(self, fedbatch_params):
        """Test batch mode (no feeding)."""
        fedbatch = FedBatchBioreactor(fedbatch_params)

        outlet, info = fedbatch(
            X0=0.5,   # Initial cell conc
            S0=20.0,  # Initial substrate
            P0=0.0,
            t_final=20.0,  # 20 hours
            feed_rate_fn=None,  # Batch mode
            n_steps=100,
        )

        # Check profiles have correct shape
        assert len(info["t"]) == 101
        assert len(info["X"]) == 101

        # Cells should grow
        assert float(info["X_final"]) > 0.5

        # Substrate should be consumed
        assert float(info["S_final"]) < 20.0

        # Volume should stay constant (batch)
        assert float(info["V_final"]) == pytest.approx(5.0, rel=0.01)

    def test_fedbatch_mode(self, fedbatch_params):
        """Test fed-batch with exponential feeding."""
        fedbatch = FedBatchBioreactor(fedbatch_params)

        # Exponential feed profile
        def feed_rate(t):
            F0 = jnp.array(0.1)  # L/h initial
            mu_set = jnp.array(0.2)
            return F0 * jnp.exp(mu_set * t)

        outlet, info = fedbatch(
            X0=0.5,
            S0=5.0,
            P0=0.0,
            t_final=10.0,
            feed_rate_fn=feed_rate,
            S_feed=200.0,  # Concentrated feed
            n_steps=100,
        )

        # Volume should increase with feeding
        assert float(info["V_final"]) > 5.0

        # Cells should grow
        assert float(info["X_final"]) > 0.5

    def test_fedbatch_differentiability(self, fedbatch_params):
        """Test that fed-batch is differentiable w.r.t. initial conditions."""
        def final_product(X0):
            fedbatch = FedBatchBioreactor(fedbatch_params)
            outlet, info = fedbatch(
                X0=X0,
                S0=20.0,
                P0=0.0,
                t_final=5.0,  # Shorter time so substrate doesn't deplete
                n_steps=50,
            )
            return info["P_final"]

        # Compute gradient
        grad_X0 = jax.grad(final_product)(jnp.array(0.5))

        # Gradient should be positive (more initial cells = more product)
        # Note: At t=10h substrate depletes, making gradient ~0. Use t=5h.
        assert float(grad_X0) > 0


class TestFedBatchOxygenCoupling:
    """Issue #101: fed-batch growth coupled to oxygen transfer (OTR)."""

    def _params(self, kLa=None):
        return FedBatchParams(
            V0=jnp.array(5.0),
            Y_xs=jnp.array(0.5),
            kinetic_fn=monod_kinetics,
            kinetic_params={"mu_max": jnp.array(0.4), "K_s": jnp.array(0.5)},
            kLa=kLa,
            Y_xo=jnp.array(1.0),
        )

    def test_backward_compat_no_oxygen(self):
        """Without kLa, no O2 state or diagnostics (4-state model)."""
        fb = FedBatchBioreactor(self._params(kLa=None))
        _, info = fb(X0=0.5, S0=20.0, P0=0.0, t_final=10.0, n_steps=50)
        assert "C_O2" not in info
        assert float(info["X_final"]) > 0.5

    def test_oxygen_tracked_and_reported(self):
        fb = FedBatchBioreactor(self._params(kLa=200.0))
        _, info = fb(X0=0.5, S0=20.0, P0=0.0, t_final=10.0, n_steps=50)
        assert "C_O2" in info and "OTR" in info
        # DO stays between 0 and saturation
        assert float(jnp.min(info["C_O2"])) >= 0.0
        assert float(info["C_O2"][0]) == pytest.approx(7.0e-3, rel=1e-3)

    def test_low_kla_limits_growth(self):
        """Poor oxygen transfer (low kLa) should reduce biomass vs high kLa."""
        fb_high = FedBatchBioreactor(self._params(kLa=500.0))
        fb_low = FedBatchBioreactor(self._params(kLa=5.0))
        _, hi = fb_high(X0=0.5, S0=50.0, P0=0.0, t_final=15.0, n_steps=100)
        _, lo = fb_low(X0=0.5, S0=50.0, P0=0.0, t_final=15.0, n_steps=100)
        # Oxygen-limited culture accumulates less biomass
        assert float(lo["X_final"]) < float(hi["X_final"])
        # And runs at a lower dissolved-O2 / limitation factor
        assert float(jnp.min(lo["o2_limitation"])) < float(jnp.min(hi["o2_limitation"]))

    def test_oxygen_differentiable_through_kla(self):
        def final_cells(kla):
            fb = FedBatchBioreactor(self._params(kLa=kla))
            _, info = fb(X0=0.5, S0=50.0, P0=0.0, t_final=8.0, n_steps=50, solver="rk4")
            return info["X_final"]
        g = jax.grad(final_cells)(20.0)
        assert jnp.isfinite(g)
        assert float(g) > 0.0  # more O2 transfer -> more growth
