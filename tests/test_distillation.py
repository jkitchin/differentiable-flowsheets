"""Tests for distillation column unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    IdealThermo,
    SpeciesData,
    make_stream,
    get_flows,
)
from difflow.units.distillation import (
    ShortcutColumn,
    ShortcutColumnParams,
    DistillationColumn,
    DistillationColumnParams,
    fenske_stages,
    minimum_reflux_ratio,
    gilliland_stages,
    column_diameter,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def benzene_toluene_thermo():
    """Benzene-toluene thermodynamics for distillation."""
    species_data = {
        "benzene": SpeciesData(
            name="benzene",
            MW=78.11,
            Cp_coeffs=(136.0, 0.0, 0.0, 0.0),  # Simplified
            Hvap_coeffs=(33900.0, 0.38, 562.0),
            # Antoine: log10(P/Pa) = A - B/(T + C)
            antoine_coeffs=(13.82, 2788.0, -52.36),  # Pa, K
        ),
        "toluene": SpeciesData(
            name="toluene",
            MW=92.14,
            Cp_coeffs=(157.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(38000.0, 0.38, 591.8),
            antoine_coeffs=(13.93, 3096.0, -53.67),
        ),
    }
    return IdealThermo(species_data)


@pytest.fixture
def multicomponent_thermo():
    """Three-component system for testing."""
    species_data = {
        "light": SpeciesData(
            name="light",
            MW=50.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(25000.0, 0.38, 400.0),
            antoine_coeffs=(13.5, 2500.0, -40.0),  # Most volatile
        ),
        "middle": SpeciesData(
            name="middle",
            MW=75.0,
            Cp_coeffs=(100.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 450.0),
            antoine_coeffs=(13.5, 2800.0, -45.0),
        ),
        "heavy": SpeciesData(
            name="heavy",
            MW=100.0,
            Cp_coeffs=(125.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(35000.0, 0.38, 500.0),
            antoine_coeffs=(13.5, 3100.0, -50.0),  # Least volatile
        ),
    }
    return IdealThermo(species_data)


class TestShortcutColumn:
    """Tests for shortcut distillation method."""

    def test_shortcut_creation(self, benzene_toluene_thermo):
        """Test shortcut column can be created."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)
        assert column is not None

    def test_shortcut_separation(self, benzene_toluene_thermo):
        """Test shortcut column performs separation."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)

        # 50/50 feed
        feed = make_stream(
            {"benzene": 50.0, "toluene": 50.0},
            T=380.0,  # K, between boiling points
            P=101325.0,
        )

        distillate, bottoms, info = column(feed, R=2.0, P=101325.0)

        # Distillate should be enriched in benzene
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        x_D_benzene = float(dist_flows["benzene"]) / (
            float(dist_flows["benzene"]) + float(dist_flows["toluene"])
        )
        x_B_toluene = float(bot_flows["toluene"]) / (
            float(bot_flows["benzene"]) + float(bot_flows["toluene"])
        )

        # Check separation occurred
        assert x_D_benzene > 0.5  # Distillate enriched in benzene
        assert x_B_toluene > 0.5  # Bottoms enriched in toluene

        # Check info contains design parameters
        assert "N_min" in info
        assert "R_min" in info
        assert "N" in info
        assert float(info["N"]) > float(info["N_min"])

    def test_shortcut_mass_balance(self, benzene_toluene_thermo):
        """Test mass balance closure."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)

        feed = make_stream(
            {"benzene": 60.0, "toluene": 40.0},
            T=380.0,
            P=101325.0,
        )

        distillate, bottoms, _ = column(feed, R=1.5)

        # Check mass balance
        feed_flows = get_flows(feed)
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        for species in ["benzene", "toluene"]:
            F = float(feed_flows[species])
            D = float(dist_flows[species])
            B = float(bot_flows[species])
            assert D + B == pytest.approx(F, rel=0.01)

    def test_shortcut_reflux_sensitivity(self, benzene_toluene_thermo):
        """Test that higher reflux gives better separation."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)

        feed = make_stream(
            {"benzene": 50.0, "toluene": 50.0},
            T=380.0,
            P=101325.0,
        )

        # Low reflux
        _, _, info_low = column(feed, R=1.0)

        # High reflux
        _, _, info_high = column(feed, R=5.0)

        # Higher reflux should give fewer stages for same separation
        # (but same separation spec, so N should be lower)
        assert float(info_high["N"]) <= float(info_low["N"])

    def test_shortcut_differentiability(self, benzene_toluene_thermo):
        """Test that shortcut column is differentiable."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)

        def N_func(R):
            feed = make_stream(
                {"benzene": 50.0, "toluene": 50.0},
                T=380.0,
                P=101325.0,
            )
            _, _, info = column(feed, R=R)
            return info["N"]

        # Compute gradient
        grad_R = jax.grad(N_func)(jnp.array(2.0))

        # Gradient should be negative (higher R = fewer stages)
        assert jnp.isfinite(grad_R)
        assert float(grad_R) < 0


class TestDesignFunctions:
    """Tests for standalone design functions."""

    def test_fenske_stages_binary(self):
        """Test Fenske equation for binary separation."""
        # For 95% recovery of each key
        x_D_LK = jnp.array(0.95)
        x_B_LK = jnp.array(0.05)
        alpha = jnp.array(2.5)  # Typical relative volatility

        N_min = fenske_stages(x_D_LK, x_B_LK, alpha)

        # Should be positive and reasonable
        assert float(N_min) > 0
        assert float(N_min) < 50  # Not unreasonable for this separation

    def test_fenske_higher_alpha_fewer_stages(self):
        """Test that higher alpha gives fewer stages."""
        x_D_LK = jnp.array(0.95)
        x_B_LK = jnp.array(0.05)

        N_low_alpha = fenske_stages(x_D_LK, x_B_LK, jnp.array(2.0))
        N_high_alpha = fenske_stages(x_D_LK, x_B_LK, jnp.array(4.0))

        assert float(N_high_alpha) < float(N_low_alpha)

    def test_minimum_reflux_ratio(self):
        """Test minimum reflux calculation."""
        z_LK = jnp.array(0.5)
        z_HK = jnp.array(0.5)
        x_D_LK = jnp.array(0.95)
        alpha = jnp.array(2.5)

        R_min = minimum_reflux_ratio(z_LK, z_HK, x_D_LK, alpha)

        # Should be positive
        assert float(R_min) > 0
        # Should be reasonable (typically 0.5 - 5 for most separations)
        assert float(R_min) < 10

    def test_gilliland_stages(self):
        """Test Gilliland correlation."""
        R = jnp.array(2.0)
        R_min = jnp.array(1.0)
        N_min = jnp.array(10.0)

        N = gilliland_stages(R, R_min, N_min)

        # N should be greater than N_min
        assert float(N) > float(N_min)

        # At high reflux, N should approach N_min
        N_high_R = gilliland_stages(jnp.array(10.0), R_min, N_min)
        assert float(N_high_R) < float(N)

    def test_column_diameter(self):
        """Test column diameter estimation."""
        V = jnp.array(100.0)  # mol/s vapor flow
        rho_V = jnp.array(3.0)  # kg/m³
        rho_L = jnp.array(800.0)  # kg/m³

        D = column_diameter(V, rho_V, rho_L)

        # Should be positive and reasonable (0.5 - 5 m for most columns)
        assert float(D) > 0
        assert float(D) < 10


class TestDistillationColumn:
    """Tests for rigorous distillation column."""

    def test_column_creation(self, benzene_toluene_thermo):
        """Test rigorous column can be created."""
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=10,
            feed_stage=5,
            condenser_type="total",
            P=101325.0,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        assert column is not None

    def test_column_separation(self, benzene_toluene_thermo):
        """Test rigorous column performs separation."""
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=15,
            feed_stage=7,
            condenser_type="total",
            P=101325.0,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)

        feed = make_stream(
            {"benzene": 50.0, "toluene": 50.0},
            T=380.0,
            P=101325.0,
        )

        distillate, bottoms, info = column(feed, R=2.0, D_spec=50.0)

        # Check outputs exist
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        # Both products should have positive flows
        assert float(dist_flows["benzene"]) > 0
        assert float(bot_flows["toluene"]) > 0

        # Info should contain profiles
        assert "T_profile" in info
        assert "x_profile" in info


class TestMulticomponentDistillation:
    """Tests for multicomponent distillation."""

    def test_multicomponent_shortcut(self, multicomponent_thermo):
        """Test shortcut column with three components."""
        params = ShortcutColumnParams(
            species_order=["light", "middle", "heavy"],
            light_key="middle",  # Separate middle from heavy
            heavy_key="heavy",
            x_D_LK=0.90,
            x_B_HK=0.90,
        )
        column = ShortcutColumn(params, multicomponent_thermo)

        feed = make_stream(
            {"light": 20.0, "middle": 40.0, "heavy": 40.0},
            T=400.0,
            P=101325.0,
        )

        distillate, bottoms, info = column(feed, R=3.0)

        # Light component should go mostly to distillate
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        light_recovery_dist = float(dist_flows["light"]) / 20.0
        assert light_recovery_dist > 0.8  # Most light goes to top

        # Heavy should go to bottoms
        heavy_recovery_bot = float(bot_flows["heavy"]) / 40.0
        assert heavy_recovery_bot > 0.8  # Most heavy goes to bottom


class TestRigorousColumnComponentBalance:
    """Per-species material balance on both solver paths (issue #211).

    The CMO path (``use_mesh=False``) used to run a Lewis-Matheson sweep whose
    top-stage update was an algebraic no-op, so the distillate composition was
    frozen at a one-stage flash of the feed and nothing in the loop enforced a
    component balance.  A three-component case reported 51.7 mol/s of pentane
    leaving a column fed 30 mol/s of it.  Both paths now solve the component
    balances as a tridiagonal system, so both close them.
    """

    @pytest.fixture
    def column(self, multicomponent_thermo):
        params = DistillationColumnParams(
            species_order=["light", "middle", "heavy"],
            n_stages=20,
            feed_stage=10,
            P=101325.0,
        )
        return DistillationColumn(params, multicomponent_thermo)

    @pytest.fixture
    def feed(self):
        return make_stream(
            {"light": 30.0, "middle": 40.0, "heavy": 30.0},
            T=400.0,
            P=101325.0,
        )

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_component_balance_closes(self, column, feed, use_mesh):
        """D_i + B_i == F_i for every species, on both paths."""
        distillate, bottoms, info = column(
            feed, R=2.0, B_spec=40.0, use_mesh=use_mesh
        )

        feed_flows = get_flows(feed)
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        for species in ["light", "middle", "heavy"]:
            F = float(feed_flows[species])
            D = float(dist_flows[species])
            B = float(bot_flows[species])
            assert D + B == pytest.approx(F, rel=0.05), (
                f"{species}: D={D} + B={B} != F={F}"
            )

        # The reported closure diagnostic must agree with what we just measured
        assert float(info["balance_error_rel"]) < 0.01

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_total_balance_closes(self, column, feed, use_mesh):
        """Total flows are pinned by the D/B specification."""
        distillate, bottoms, _ = column(
            feed, R=2.0, B_spec=40.0, use_mesh=use_mesh
        )
        D_total = sum(float(v) for v in get_flows(distillate).values())
        B_total = sum(float(v) for v in get_flows(bottoms).values())

        assert D_total == pytest.approx(60.0, rel=1e-6)
        assert B_total == pytest.approx(40.0, rel=1e-6)

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_balance_error_reported(self, column, feed, use_mesh):
        """info carries the closure residual so a caller can check it."""
        distillate, bottoms, info = column(
            feed, R=2.0, B_spec=40.0, use_mesh=use_mesh
        )

        feed_flows = get_flows(feed)
        dist_flows = get_flows(distillate)
        bot_flows = get_flows(bottoms)

        expected = [
            float(dist_flows[s] + bot_flows[s] - feed_flows[s])
            for s in ["light", "middle", "heavy"]
        ]
        assert list(info["balance_error"]) == pytest.approx(expected, abs=1e-12)
        assert float(info["balance_error_rel"]) == pytest.approx(
            max(abs(e) for e in expected) / 100.0, rel=1e-6
        )

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_responds_to_feed_stage(self, multicomponent_thermo, feed, use_mesh):
        """Moving the feed must change the products (issue #211).

        The old CMO sweep returned byte-identical products for every feed
        stage.  Here the keys separate essentially completely, so the signal
        lives in the off-key impurities: stages below the feed strip the light
        component out of the bottoms, stages above it keep the heavy component
        out of the distillate.  Moving the feed up trades the second for the
        first.
        """
        heavy_in_dist, light_in_bot = [], []
        for feed_stage in (4, 10, 16):
            params = DistillationColumnParams(
                species_order=["light", "middle", "heavy"],
                n_stages=20,
                feed_stage=feed_stage,
                P=101325.0,
            )
            column = DistillationColumn(params, multicomponent_thermo)
            distillate, bottoms, _ = column(
                feed, R=2.0, B_spec=40.0, use_mesh=use_mesh
            )
            heavy_in_dist.append(float(get_flows(distillate)["heavy"]))
            light_in_bot.append(float(get_flows(bottoms)["light"]))

        # Fewer rectifying stages -> more heavy carried into the distillate
        assert heavy_in_dist[0] < heavy_in_dist[1] < heavy_in_dist[2]
        # Fewer stripping stages -> more light left in the bottoms
        assert light_in_bot[0] > light_in_bot[1]

    def test_cmo_responds_to_reflux(self, column, feed):
        """Higher reflux must sharpen the CMO separation (issue #211).

        The old sweep froze x_D at a one-stage flash of the feed, making the
        distillate composition independent of R.
        """
        heavy_in_dist = []
        for R in (0.5, 2.0, 8.0):
            distillate, _, _ = column(feed, R=R, B_spec=40.0, use_mesh=False)
            dist_flows = get_flows(distillate)
            total = sum(float(v) for v in dist_flows.values())
            heavy_in_dist.append(float(dist_flows["heavy"]) / total)

        assert heavy_in_dist[0] > heavy_in_dist[1] > heavy_in_dist[2]

    @pytest.mark.parametrize("use_mesh", [True, False])
    def test_gradient_wrt_reflux_is_finite(self, column, feed, use_mesh):
        """Products stay differentiable w.r.t. the reflux ratio."""

        def middle_in_distillate(R):
            distillate, _, _ = column(
                feed, R=R, B_spec=40.0, use_mesh=use_mesh
            )
            return get_flows(distillate)["middle"]

        g = float(jax.grad(middle_in_distillate)(2.0))
        assert jnp.isfinite(g)
        assert g != 0.0

    def test_issue_211_reported_case(self):
        """The exact case from issue #211, on real pentane/hexane/heptane data.

        Reported: the CMO path returned 51.696 mol/s of n-pentane from a column
        fed 30 mol/s of it (+72 %), with n-hexane down 81 % and n-heptane up
        36 %.  Unlike the synthetic fixture above, this case splits n-hexane
        roughly in half, so the component balances are genuinely loaded.
        """
        from difflow.database import get_species_data

        names = ["n_pentane", "n_hexane", "n_heptane"]
        thermo = IdealThermo({s: get_species_data(s) for s in names})
        feed = make_stream(
            {"n_pentane": 30.0, "n_hexane": 40.0, "n_heptane": 30.0},
            T=360.0,
            P=101325.0,
        )
        column = DistillationColumn(
            DistillationColumnParams(
                species_order=names, n_stages=20, feed_stage=10, P=101325.0
            ),
            thermo,
        )

        feed_flows = get_flows(feed)
        for use_mesh in (True, False):
            distillate, bottoms, info = column(
                feed, R=2.0, B_spec=40.0, use_mesh=use_mesh
            )
            dist_flows = get_flows(distillate)
            bot_flows = get_flows(bottoms)
            for s in names:
                F = float(feed_flows[s])
                total = float(dist_flows[s] + bot_flows[s])
                assert total == pytest.approx(F, rel=0.01), (
                    f"use_mesh={use_mesh}, {s}: D+B={total} != F={F}"
                )
            assert float(info["balance_error_rel"]) < 1e-3

            # n-hexane really is split between the products here, so the
            # balance above is not trivially satisfied by near-zero flows.
            assert 5.0 < float(dist_flows["n_hexane"]) < 35.0
            assert 5.0 < float(bot_flows["n_hexane"]) < 35.0

    def test_balance_error_falls_with_cmo_iter(self, column, feed):
        """Raising cmo_iter tightens the reported closure (issue #211)."""
        errors = [
            float(
                column(feed, R=2.0, B_spec=40.0, use_mesh=False, cmo_iter=n)[2][
                    "balance_error_rel"
                ]
            )
            for n in (5, 15, 40)
        ]
        assert errors[0] > errors[1] > errors[2]
        assert errors[2] < 1e-6
