"""Tests for centrifuge unit operation."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import (
    Centrifuge,
    CentrifugeParams,
    DiscStackCentrifuge,
    DiscStackParams,
    stokes_velocity,
    critical_particle_diameter,
    disc_stack_sigma,
    g_force,
    tubular_bowl_sigma,
)
from difflow import make_stream, get_flows


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

        concentrate, clarified, info = centrifuge(
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

        concentrate, clarified, info = centrifuge(
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

        _, _, info_low_Q = centrifuge(feed, Q=1e-5, concentrate_fraction=0.1)
        _, _, info_high_Q = centrifuge(feed, Q=1e-3, concentrate_fraction=0.1)

        # Higher flow = larger minimum separable particle
        assert float(info_high_Q["critical_diameter"]) > float(info_low_Q["critical_diameter"])

    def test_centrifuge_differentiability(self, centrifuge_params):
        """Test that centrifuge is differentiable."""
        def cell_recovery(sigma):
            params = CentrifugeParams(sigma=sigma, efficiency=0.9)
            centrifuge = Centrifuge(params)
            feed = make_stream({"cells": 100.0, "product": 50.0}, T=300.0, P=101325.0)
            # Use higher Q to be in non-saturated regime
            concentrate, _, info = centrifuge(feed, Q=1e-3, concentrate_fraction=0.1)
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

        concentrate, clarified, info = centrifuge(
            feed, Q=1e-4, concentrate_fraction=0.1
        )

        # Should separate cells
        conc_flows = get_flows(concentrate)
        clar_flows = get_flows(clarified)
        assert float(conc_flows["cells"]) > float(clar_flows["cells"])


class TestCentrifugeCellLysis:
    """Issue #103: high g-force should lyse cells, releasing lysate."""

    def _feed(self):
        return make_stream({"cells": 100.0, "protein": 10.0, "water": 890.0},
                           T=300.0, P=101325.0)

    def test_no_lysis_backward_compat(self):
        cf = Centrifuge(CentrifugeParams(sigma=1000.0))  # lysis disabled
        conc, clar, info = cf(self._feed(), Q=0.001, rcf=20000.0)
        # Cells conserved, no lysate species introduced
        total_cells = float(get_flows(conc)["cells"]) + float(get_flows(clar)["cells"])
        assert total_cells == pytest.approx(100.0, rel=1e-6)
        assert "lysate" not in get_flows(clar)
        assert float(info["lysis_fraction"]) == 0.0

    def test_lysis_above_threshold(self):
        cf = Centrifuge(CentrifugeParams(
            sigma=1000.0, lysis_threshold_g=10000.0,
            lysis_coefficient=1e-5, lysate_species="lysate",
        ))
        conc, clar, info = cf(self._feed(), Q=0.001, rcf=30000.0)
        # excess = 20000, fraction = 1e-5*20000 = 0.2
        assert float(info["lysis_fraction"]) == pytest.approx(0.2, rel=1e-6)
        assert float(info["lysate_released"]) == pytest.approx(20.0, rel=1e-6)
        # lysate reports to the supernatant
        assert float(get_flows(clar)["lysate"]) == pytest.approx(20.0, rel=1e-6)
        # cells + lysate conserved
        total = (float(get_flows(conc)["cells"]) + float(get_flows(clar)["cells"])
                 + float(get_flows(clar)["lysate"]))
        assert total == pytest.approx(100.0, rel=1e-6)

    def test_no_lysis_below_threshold(self):
        cf = Centrifuge(CentrifugeParams(
            sigma=1000.0, lysis_threshold_g=10000.0, lysis_coefficient=1e-5,
        ))
        _, _, info = cf(self._feed(), Q=0.001, rcf=5000.0)  # below threshold
        assert float(info["lysis_fraction"]) == 0.0

    def test_disc_stack_uses_geometry_rcf(self):
        params = DiscStackParams(
            n_discs=100, r_outer=0.15, r_inner=0.05, rpm=10000.0,
            lysis_threshold_g=1000.0, lysis_coefficient=1e-5,
        )
        dsc = DiscStackCentrifuge(params)
        # RCF from geometry should be well above threshold -> some lysis
        _, clar, info = dsc(self._feed(), Q=0.001)
        assert float(info["lysis_fraction"]) > 0.0
        assert "lysate" in get_flows(clar)


class TestSigmaTheory:
    """Sigma theory as an equivalent gravity settler (Ambler 1959; Geankoplis 5e Sec. 14.4C)."""

    def test_capture_is_hazen_in_the_equivalent_settler(self):
        """Complete capture at Q = v_s Sigma, half at Q = 2 v_s Sigma (the cut flow)."""
        sigma, d, rp, rf, mu = 2000.0, 4e-6, 1080.0, 1000.0, 1e-3
        v = float(stokes_velocity(d, rp, rf, mu))
        cf = Centrifuge(CentrifugeParams(sigma=sigma, efficiency=1.0))
        feed = make_stream({"cells": 1.0, "water": 50.0}, T=298.15, P=101325.0)
        for Q, eta in [(v * sigma, 1.0), (2 * v * sigma, 0.5), (4 * v * sigma, 0.25), (0.5 * v * sigma, 1.0)]:
            _, _, info = cf(feed, Q, d, rp, rf, mu)
            assert float(info["cell_recovery"]) == pytest.approx(eta, rel=1e-12)
        _, _, info = cf(feed, 2 * v * sigma, d, rp, rf, mu)
        assert float(info["Q_critical"]) == pytest.approx(2 * v * sigma, rel=1e-12)

    def test_efficiency_scales_sigma_not_the_capture(self):
        """A slow feed is captured completely even with efficiency < 1 (Ambler's correction on Sigma)."""
        sigma, d, rp, rf, mu = 2000.0, 4e-6, 1080.0, 1000.0, 1e-3
        v = float(stokes_velocity(d, rp, rf, mu))
        feed = make_stream({"cells": 1.0, "water": 50.0}, T=298.15, P=101325.0)
        cf = Centrifuge(CentrifugeParams(sigma=sigma, efficiency=0.5))
        assert float(cf(feed, 0.1 * v * sigma, d, rp, rf, mu)[2]["cell_recovery"]) == pytest.approx(1.0)
        assert float(cf(feed, v * sigma, d, rp, rf, mu)[2]["cell_recovery"]) == pytest.approx(0.5, rel=1e-12)

    def test_cut_size_is_the_particle_at_half_capture(self):
        sigma, Q, rp, rf, mu = 1500.0, 2e-3, 1100.0, 1000.0, 1.2e-3
        d50 = critical_particle_diameter(Q, sigma, rp, rf, mu)
        assert float(2 * stokes_velocity(d50, rp, rf, mu) * sigma) == pytest.approx(Q, rel=1e-12)

    def test_tubular_bowl_thin_layer_limit(self):
        r, L, rpm = 0.05, 0.7, 15000.0
        omega = rpm * 2 * jnp.pi / 60
        thin = float(tubular_bowl_sigma(r, L, rpm))
        assert thin == pytest.approx(float(2 * jnp.pi * omega**2 * L * r**2 / 9.81), rel=1e-12)
        assert float(tubular_bowl_sigma(r, L, rpm, r_inner=r * (1 - 1e-5))) == pytest.approx(thin, rel=1e-4)

    def test_tubular_bowl_trajectory(self):
        """Thick layer: the 50% particle starts midway through the layer and just reaches the wall."""
        r2, r1, L, rpm, rp, rf, mu = 0.0285, 0.02225, 0.197, 23000.0, 1461.0, 801.0, 0.1
        Q = 0.002832 / 3600.0
        sigma = tubular_bowl_sigma(r2, L, rpm, r_inner=r1)
        d = critical_particle_diameter(Q, sigma, rp, rf, mu)
        omega = rpm * 2 * jnp.pi / 60
        V = jnp.pi * L * (r2**2 - r1**2)
        # Stokes radial velocity omega^2 r dRho d^2/(18 mu): ln(r2/r_mid) = omega^2 dRho d^2 t/(18 mu), t = V/Q
        lhs = jnp.log(r2 / (0.5 * (r1 + r2)))
        rhs = omega**2 * (rp - rf) * d**2 / (18 * mu) * V / Q
        assert float(lhs) == pytest.approx(float(rhs), rel=1e-10)

    def test_disc_stack_gradient_of_recovery_in_rpm(self):
        def rec(rpm):
            dsc = DiscStackCentrifuge(DiscStackParams(n_discs=80, r_outer=0.12, r_inner=0.04, rpm=rpm,
                                                      efficiency=1.0))
            feed = make_stream({"cells": 1.0, "water": 50.0}, T=298.15, P=101325.0)
            return dsc(feed, 3e-3, 2e-6, 1080.0, 1000.0, 1e-3)[2]["cell_recovery"]
        assert float(rec(4000.0)) < 1.0
        assert float(jax.grad(rec)(4000.0)) > 0.0
