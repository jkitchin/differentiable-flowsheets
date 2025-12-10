"""Tests for centrifuge unit operation."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    Centrifuge,
    CentrifugeParams,
    DiscStackCentrifuge,
    DiscStackParams,
    stokes_velocity,
    critical_particle_diameter,
    disc_stack_sigma,
    g_force,
    make_stream,
    get_flows,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


class TestStokesVelocity:
    def test_stokes_positive(self):
        """Settling velocity should be positive for denser particles."""
        v = stokes_velocity(
            d=5e-6,        # 5 μm
            rho_p=1050.0,  # kg/m³, cells
            rho_f=1000.0,  # kg/m³, water
            mu=0.001,      # Pa·s
        )
        assert float(v) > 0

    def test_stokes_scales_with_diameter_squared(self):
        """Velocity scales with d²."""
        v1 = stokes_velocity(5e-6, 1050.0, 1000.0, 0.001)
        v2 = stokes_velocity(10e-6, 1050.0, 1000.0, 0.001)

        ratio = float(v2 / v1)
        assert ratio == pytest.approx(4.0, rel=1e-6)  # (10/5)² = 4

    def test_stokes_scales_with_density_difference(self):
        """Velocity scales linearly with density difference."""
        v1 = stokes_velocity(5e-6, 1050.0, 1000.0, 0.001)  # Δρ = 50
        v2 = stokes_velocity(5e-6, 1100.0, 1000.0, 0.001)  # Δρ = 100

        ratio = float(v2 / v1)
        assert ratio == pytest.approx(2.0, rel=1e-6)


class TestSigmaFactor:
    def test_disc_stack_sigma(self):
        """Test disc-stack Sigma calculation."""
        sigma = disc_stack_sigma(
            n_discs=100,
            r_outer=0.1,   # 10 cm
            r_inner=0.05,  # 5 cm
            half_angle=0.698,  # 40 degrees
            rpm=6000,
        )
        # Sigma should be positive and reasonable
        assert float(sigma) > 0
        assert float(sigma) < 1e6  # Reasonable upper bound

    def test_sigma_scales_with_rpm_squared(self):
        """Sigma scales with ω² ∝ rpm²."""
        sigma1 = disc_stack_sigma(100, 0.1, 0.05, 0.698, 3000)
        sigma2 = disc_stack_sigma(100, 0.1, 0.05, 0.698, 6000)

        ratio = float(sigma2 / sigma1)
        assert ratio == pytest.approx(4.0, rel=1e-6)

    def test_sigma_scales_with_discs(self):
        """Sigma scales linearly with number of discs."""
        sigma1 = disc_stack_sigma(50, 0.1, 0.05, 0.698, 6000)
        sigma2 = disc_stack_sigma(100, 0.1, 0.05, 0.698, 6000)

        ratio = float(sigma2 / sigma1)
        assert ratio == pytest.approx(2.0, rel=1e-6)


class TestGForce:
    def test_g_force_calculation(self):
        """Test RCF calculation."""
        rcf = g_force(r=0.1, rpm=6000)

        # Expected: (6000 * 2π/60)² * 0.1 / 9.81 ≈ 4025
        expected = (6000 * 2 * jnp.pi / 60)**2 * 0.1 / 9.81
        assert float(rcf) == pytest.approx(float(expected), rel=1e-6)


class TestCentrifuge:
    @pytest.fixture
    def centrifuge_params(self):
        """Basic centrifuge parameters."""
        return CentrifugeParams(
            sigma=jnp.array(5000.0),  # m², typical lab centrifuge
            efficiency=0.7,
            cell_species="cells",
        )

    def test_centrifuge_creation(self, centrifuge_params):
        """Test centrifuge can be created."""
        centrifuge = Centrifuge(centrifuge_params)
        assert centrifuge is not None

    def test_centrifuge_separates_cells(self, centrifuge_params):
        """Test that centrifuge separates cells from liquid."""
        centrifuge = Centrifuge(centrifuge_params)

        # Feed: cells + media + product
        feed = make_stream(
            {"cells": 100.0, "substrate": 10.0, "product": 50.0},
            T=300.0, P=101325.0
        )

        (concentrate, clarified), info = centrifuge(
            feed,
            Q=1e-4,  # m³/s
            d_particle=5e-6,
            concentrate_fraction=0.1,
        )

        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)

        # Most cells should go to concentrate
        assert float(conc_flows["cells"]) > float(clar_flows["cells"])

        # Cell recovery should be reported
        assert float(info["cell_recovery"]) > 0.5

    def test_centrifuge_mass_balance(self, centrifuge_params):
        """Test mass balance is preserved."""
        centrifuge = Centrifuge(centrifuge_params)

        feed = make_stream(
            {"cells": 100.0, "substrate": 10.0, "product": 50.0},
            T=300.0, P=101325.0
        )

        (concentrate, clarified), info = centrifuge(
            feed, Q=1e-4, concentrate_fraction=0.1
        )

        feed_flows = get_flows(feed)
        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)

        for species in feed_flows:
            total_out = float(conc_flows[species]) + float(clar_flows[species])
            assert total_out == pytest.approx(float(feed_flows[species]), rel=1e-6)

    def test_critical_diameter_increases_with_flow(self, centrifuge_params):
        """Higher flow rate means larger critical particle diameter."""
        centrifuge = Centrifuge(centrifuge_params)

        feed = make_stream(
            {"cells": 100.0, "product": 50.0},
            T=300.0, P=101325.0
        )

        _, info_low_Q = centrifuge(feed, Q=1e-5, concentrate_fraction=0.1)
        _, info_high_Q = centrifuge(feed, Q=1e-3, concentrate_fraction=0.1)

        # Higher flow = larger minimum separable particle
        assert float(info_high_Q["critical_diameter"]) > float(info_low_Q["critical_diameter"])

    def test_centrifuge_differentiability(self, centrifuge_params):
        """Test that centrifuge is differentiable."""
        def cell_recovery(sigma):
            params = CentrifugeParams(sigma=sigma, efficiency=0.9)
            centrifuge = Centrifuge(params)
            feed = make_stream({"cells": 100.0, "product": 50.0}, T=300.0, P=101325.0)
            # Use higher Q to be in non-saturated regime
            (concentrate, _), info = centrifuge(feed, Q=1e-3, concentrate_fraction=0.1)
            return concentrate["F_cells"]

        grad_sigma = jax.grad(cell_recovery)(jnp.array(100.0))

        # Gradient should be positive (larger Sigma = better separation)
        assert float(grad_sigma) > 0


class TestDiscStackCentrifuge:
    def test_disc_stack_creation(self):
        """Test disc-stack centrifuge creation."""
        params = DiscStackParams(
            n_discs=100,
            r_outer=0.1,
            r_inner=0.05,
            rpm=6000.0,
        )
        centrifuge = DiscStackCentrifuge(params)
        assert centrifuge.sigma > 0

    def test_disc_stack_separation(self):
        """Test disc-stack performs separation."""
        params = DiscStackParams(
            n_discs=100,
            r_outer=0.1,
            r_inner=0.05,
            rpm=6000.0,
            efficiency=0.8,
        )
        centrifuge = DiscStackCentrifuge(params)

        feed = make_stream(
            {"cells": 100.0, "product": 50.0},
            T=300.0, P=101325.0
        )

        (concentrate, clarified), info = centrifuge(
            feed, Q=1e-4, concentrate_fraction=0.1
        )

        # Should separate cells
        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)
        assert float(conc_flows["cells"]) > float(clar_flows["cells"])
