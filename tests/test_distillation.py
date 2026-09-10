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
    _cmo_section_flows,
    _cmo_section_rates,
    _is_rectifying_cut,
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


class TestFeedStageConvention:
    """The feed stage belongs to one section on each of its two sides.

    Stages are numbered from the bottom (0 = reboiler). The feed enters stage
    ``feed_stage``, so the liquid leaving that stage downward carries ``q F``
    (a stripping-section flow) while the vapour leaving it upward carries
    ``(1 - q) F`` (a rectifying-section flow). These tests pin that convention
    on the helper and on both solvers, so the two cannot drift apart again.
    """

    @pytest.mark.parametrize("q", [1.0, 0.6, 0.0])
    def test_section_flows_split_at_the_feed_stage(self, q):
        """L steps at the feed stage, V steps just below it."""
        feed_stage = 4
        F, R, D = 100.0, 2.0, 40.0
        L_rect, L_strip, V_rect, V_strip = _cmo_section_rates(
            jnp.asarray(R), jnp.asarray(D), jnp.asarray(F), jnp.asarray(q)
        )

        j = jnp.arange(10)
        L, V = _cmo_section_flows(j, feed_stage, L_rect, L_strip, V_rect, V_strip)

        # Liquid: stripping rate up to and INCLUDING the feed stage.
        assert float(L[feed_stage]) == pytest.approx(float(L_strip))
        assert float(L[feed_stage + 1]) == pytest.approx(float(L_rect))
        assert float(L[feed_stage] - L[feed_stage + 1]) == pytest.approx(q * F)

        # Vapour: rectifying rate FROM the feed stage upward.
        assert float(V[feed_stage]) == pytest.approx(float(V_rect))
        assert float(V[feed_stage - 1]) == pytest.approx(float(V_strip))
        assert float(V[feed_stage] - V[feed_stage - 1]) == pytest.approx((1.0 - q) * F)

        # The cut above the feed stage is rectifying, the one below is not.
        assert bool(_is_rectifying_cut(feed_stage, feed_stage))
        assert not bool(_is_rectifying_cut(feed_stage - 1, feed_stage))

    @pytest.mark.parametrize("feed_stage", [4, 7, 10])
    def test_cmo_switches_operating_line_at_the_feed_stage(
        self, benzene_toluene_thermo, feed_stage
    ):
        """The CMO sweep uses the rectifying line at the feed stage itself.

        The converged sweep is a fixed point, so re-applying its operating-line
        step to the profile it returned must reproduce that profile exactly --
        but only with the section assignment the solver used.
        """
        n_stages, R, F, P = 15, 2.0, 100.0, 101325.0
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=n_stages,
            feed_stage=feed_stage,
            P=P,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        feed = make_stream({"benzene": 50.0, "toluene": 50.0}, T=380.0, P=P)

        _, _, info = column(feed, R=R, D_spec=50.0, use_mesh=False)
        x = info["x_profile"]
        y = info["y_profile"]
        T = info["T_profile"]
        D = float(info["D"])
        B = float(info["B"])

        # Section rates written out here rather than taken from the helper,
        # so this test pins the convention instead of restating it.
        L_rect = R * D
        V_rect = (R + 1) * D
        L_strip = L_rect + F  # q = 1 (saturated liquid feed)
        V_strip = V_rect

        x_D = y[-1]
        x_B = x[0]

        def operating_line_step(j, rectifying):
            """x_j implied by the operating line across the cut above stage j."""
            L_above = L_rect if (j + 1) > feed_stage else L_strip
            V_j = V_rect if j >= feed_stage else V_strip
            if rectifying:
                y_op = (L_above * x[j + 1] + D * x_D) / V_j
            else:
                y_op = (L_above * x[j + 1] - B * x_B) / V_j
            y_op = jnp.maximum(y_op, 0.0)
            y_op = y_op / jnp.maximum(jnp.sum(y_op), 1e-10)
            K_j = benzene_toluene_thermo.K_values_array(T[j], jnp.asarray(P))
            x_j = jnp.maximum(y_op / jnp.maximum(K_j, 1e-10), 0.0)
            return x_j / jnp.maximum(jnp.sum(x_j), 1e-10)

        def err(j, rectifying):
            return float(jnp.max(jnp.abs(operating_line_step(j, rectifying) - x[j])))

        # Feed stage: rectifying line reproduces the profile, stripping does not.
        assert err(feed_stage, rectifying=True) < 1e-10
        assert err(feed_stage, rectifying=False) > 1e-3

        # One stage below the feed: the other way round.
        assert err(feed_stage - 1, rectifying=False) < 1e-10
        assert err(feed_stage - 1, rectifying=True) > 1e-3

    def test_mesh_initialisation_uses_the_same_convention(
        self, benzene_toluene_thermo
    ):
        """The MESH L/V warm start splits the sections where the CMO sweep does.

        Run with zero MESH iterations so the initialisation itself is returned.
        A partially vaporised feed (q = 0.5) makes both steps visible.
        """
        feed_stage, n_stages, R, D, F, P, q = 7, 15, 2.0, 50.0, 100.0, 101325.0, 0.5
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=n_stages,
            feed_stage=feed_stage,
            P=P,
            q=q,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        feed_flows = {"benzene": jnp.asarray(50.0), "toluene": jnp.asarray(50.0)}

        _, _, _, L, V = column._solve_mesh(
            feed_flows,
            jnp.asarray(F),
            jnp.asarray(R),
            jnp.asarray(D),
            jnp.asarray(380.0),
            n_iter=0,
        )

        L_rect = R * D
        V_rect = (R + 1) * D
        L_strip = L_rect + q * F
        V_strip = V_rect - (1.0 - q) * F

        # Liquid leaving the feed stage downward carries q F.
        assert float(L[feed_stage]) == pytest.approx(L_strip)
        assert float(L[feed_stage + 1]) == pytest.approx(L_rect)
        # Vapour leaving it upward carries (1 - q) F.
        assert float(V[feed_stage]) == pytest.approx(V_rect)
        assert float(V[feed_stage - 1]) == pytest.approx(V_strip)

        # Product ends are the boundary conditions, not section rates.
        assert float(L[0]) == pytest.approx(F - D)
        assert float(V[n_stages - 1]) == pytest.approx(V_rect)

    @pytest.mark.parametrize("feed_stage", [4, 7, 10])
    def test_mesh_liquid_steps_down_below_the_feed_stage(
        self, benzene_toluene_thermo, feed_stage
    ):
        """The converged MESH liquid profile drops between f and f+1.

        The liquid leaving the feed stage downward carries the feed, so the
        section boundary in ``L`` sits below the feed stage -- the same side
        the CMO sweep puts it on.
        """
        n_stages, P = 15, 101325.0
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=n_stages,
            feed_stage=feed_stage,
            P=P,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        feed = make_stream({"benzene": 50.0, "toluene": 50.0}, T=380.0, P=P)

        _, _, info = column(feed, R=2.0, D_spec=50.0, use_mesh=True)
        L = info["L_profile"]

        drops = L[:-1] - L[1:]
        assert int(jnp.argmax(drops)) == feed_stage
        # ... and the drop is the feed itself (q = 1), to within the
        # energy-balance correction on the surrounding stages.
        assert float(drops[feed_stage]) == pytest.approx(100.0, rel=0.1)



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
