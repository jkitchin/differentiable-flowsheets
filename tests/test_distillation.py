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
    _mixture_molar_enthalpy,
)


# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


class TestBubblePointRobustness:
    """The bubble-point solve has to reach the root from a cold start.

    Vapor pressure is exponential in -1/T, so a few hundred degrees below the
    bubble point both sum(K x) and its slope are ~0 and a plain Newton step is
    one tiny number divided by another. This came up for real: the shortcut
    column's column-end temperatures are bubble points seeded from the feed,
    and a feed 250 K below its own bubble point sent the solve to -1e157.
    """

    @pytest.fixture
    def far_from_boiling_thermo(self):
        """Two species whose normal boiling points are ~600 K and ~641 K."""
        return IdealThermo({
            "A": SpeciesData(
                name="A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, 750.0),
                antoine_coeffs=(10.0, 2800.0, -40.0),
            ),
            "B": SpeciesData(
                name="B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(30000.0, 0.38, 790.0),
                antoine_coeffs=(10.0, 3000.0, -40.0),
            ),
        })

    @pytest.mark.parametrize("T_guess", [100.0, 250.0, 365.0, 620.0, 1500.0])
    def test_reaches_the_root_from_any_start(
        self, far_from_boiling_thermo, T_guess
    ):
        from difflow.units.distillation import _bubble_T

        x = jnp.array([0.5, 0.5])
        P = jnp.asarray(101325.0)
        T, y = _bubble_T(far_from_boiling_thermo, x, P, jnp.asarray(T_guess))

        assert jnp.isfinite(T)
        # Between the two pure boiling points, and satisfying sum(K x) = 1.
        assert 600.0 < float(T) < 641.0
        K = far_from_boiling_thermo.K_values_array(T, P)
        assert float(jnp.sum(K * x)) == pytest.approx(1.0, abs=1e-8)
        assert float(jnp.sum(y)) == pytest.approx(1.0)

    def test_shortcut_column_survives_a_cold_feed(self, far_from_boiling_thermo):
        """The case that surfaced it: the feed is 250 K under its bubble point,
        and the design still comes out finite and feasible."""
        column = ShortcutColumn(
            ShortcutColumnParams(
                species_order=["A", "B"], light_key="A", heavy_key="B",
            ),
            far_from_boiling_thermo,
        )
        feed = make_stream({"A": 0.5, "B": 0.5}, T=365.0, P=101325.0)
        distillate, bottoms, info = column(feed, R=2.0, P=101325.0)

        assert jnp.isfinite(distillate["T"])
        assert jnp.isfinite(bottoms["T"])
        assert jnp.isfinite(info["N_min"])
        # Nearly pure products, so each end sits at a pure boiling point.
        assert float(info["T_top"]) == pytest.approx(600.6, abs=1.0)
        assert float(info["T_bot"]) == pytest.approx(640.7, abs=1.0)
        assert bool(info["feasible"])


class TestShortcutColumnEndTemperatures:
    """The column ends are bubble points, not estimates around the feed."""

    def _column(self, thermo):
        params = ShortcutColumnParams(
            species_order=["light", "middle", "heavy"],
            light_key="middle",
            heavy_key="heavy",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        return ShortcutColumn(params, thermo)

    def _feed(self):
        return make_stream(
            {"light": 20.0, "middle": 40.0, "heavy": 40.0}, T=400.0, P=101325.0
        )

    def test_products_leave_at_their_own_bubble_points(
        self, multicomponent_thermo
    ):
        """Distillate at the condenser (bubble point of x_D), bottoms at the
        reboiler (bubble point of x_B) -- checked by re-solving each from a
        different starting temperature."""
        from difflow.units.distillation import _bubble_T

        column = self._column(multicomponent_thermo)
        distillate, bottoms, info = column(self._feed(), R=3.0, P=101325.0)

        order = ["light", "middle", "heavy"]
        x_D = jnp.array([info["x_D"][s] for s in order])
        x_B = jnp.array([info["x_B"][s] for s in order])
        P = jnp.asarray(101325.0)

        T_D, _ = _bubble_T(multicomponent_thermo, x_D, P, jnp.asarray(300.0))
        T_B, _ = _bubble_T(multicomponent_thermo, x_B, P, jnp.asarray(500.0))

        assert float(info["T_top"]) == pytest.approx(float(T_D), rel=1e-6)
        assert float(info["T_bot"]) == pytest.approx(float(T_B), rel=1e-6)
        assert float(distillate["T"]) == pytest.approx(float(info["T_top"]))
        assert float(bottoms["T"]) == pytest.approx(float(info["T_bot"]))
        assert float(info["T_condenser"]) == pytest.approx(float(info["T_top"]))
        assert float(info["T_reboiler"]) == pytest.approx(float(info["T_bot"]))

        # The column really does run hot at the bottom and cool at the top.
        assert float(info["T_bot"]) > float(info["T_top"])

    def test_end_temperatures_do_not_track_the_feed_temperature(
        self, multicomponent_thermo
    ):
        """They are set by the products at the column pressure, so feeding the
        same mixture in hotter cannot move them."""
        column = self._column(multicomponent_thermo)
        _, _, cold = column(
            make_stream({"light": 20.0, "middle": 40.0, "heavy": 40.0},
                        T=340.0, P=101325.0),
            R=3.0, P=101325.0,
        )
        _, _, hot = column(
            make_stream({"light": 20.0, "middle": 40.0, "heavy": 40.0},
                        T=460.0, P=101325.0),
            R=3.0, P=101325.0,
        )
        assert float(cold["T_top"]) == pytest.approx(float(hot["T_top"]), rel=1e-6)
        assert float(cold["T_bot"]) == pytest.approx(float(hot["T_bot"]), rel=1e-6)

    def test_end_temperatures_rise_with_pressure(self, multicomponent_thermo):
        """A bubble point does depend on pressure, and the estimate it replaced
        did not."""
        column = self._column(multicomponent_thermo)
        _, _, low = column(self._feed(), R=3.0, P=101325.0)
        _, _, high = column(
            make_stream({"light": 20.0, "middle": 40.0, "heavy": 40.0},
                        T=400.0, P=5 * 101325.0),
            R=3.0, P=5 * 101325.0,
        )
        assert float(high["T_top"]) > float(low["T_top"])
        assert float(high["T_bot"]) > float(low["T_bot"])

    def test_volatilities_come_from_the_end_temperatures(
        self, multicomponent_thermo
    ):
        """alpha is the geometric mean of the two ends, and the reported ends
        are the converged ones -- so recomputing alpha from info reproduces it."""
        column = self._column(multicomponent_thermo)
        _, _, info = column(self._feed(), R=3.0, P=101325.0)

        order = ["light", "middle", "heavy"]
        x_D = jnp.array([info["x_D"][s] for s in order])
        x_B = jnp.array([info["x_B"][s] for s in order])
        alpha = column.average_alpha(
            info["T_top"], info["T_bot"], jnp.asarray(101325.0), x_D, x_B
        )
        for s in order:
            assert float(alpha[s]) == pytest.approx(float(info["alpha"][s]), rel=1e-9)

    def test_duties_still_physical(self, multicomponent_thermo):
        column = self._column(multicomponent_thermo)
        _, _, info = column(self._feed(), R=3.0, P=101325.0)
        assert float(info["Q_condenser"]) < 0.0
        assert float(info["Q_reboiler"]) > 0.0

    def test_gradients_survive_the_fixed_point(self, multicomponent_thermo):
        """The end temperatures are an unrolled fixed point, so AD still runs
        through the whole design calculation."""
        column = self._column(multicomponent_thermo)
        feed = self._feed()

        def stages(R):
            _, _, info = column(feed, R=R, P=101325.0)
            return info["N"]

        g = jax.grad(stages)(3.0)
        assert jnp.isfinite(g)
        eps = 1e-4
        fd = (float(stages(3.0 + eps)) - float(stages(3.0 - eps))) / (2 * eps)
        assert float(g) == pytest.approx(fd, rel=1e-4)


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

    def test_distillate_leaves_at_the_condenser_temperature(
        self, benzene_toluene_thermo
    ):
        """A total condenser delivers saturated liquid, so the distillate is at
        the bubble point of its own composition -- not at the top stage
        temperature, which is that composition's dew point."""
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=15,
            feed_stage=7,
            condenser_type="total",
            P=101325.0,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        feed = make_stream({"benzene": 50.0, "toluene": 50.0}, T=380.0, P=101325.0)

        distillate, bottoms, info = column(feed, R=2.0, D_spec=50.0)

        x_D = info["y_profile"][-1]
        T_bubble, _ = column._bubble_point_T(
            x_D, jnp.asarray(101325.0), T_guess=info["T_profile"][-1]
        )

        assert float(distillate["T"]) == pytest.approx(float(T_bubble), rel=1e-9)
        assert float(info["T_condenser"]) == pytest.approx(float(distillate["T"]))
        # Bubble point <= dew point, with equality only for a pure component.
        assert float(info["T_condenser"]) <= float(info["T_profile"][-1])

        # The bottoms still leaves at the reboiler, which is a stage.
        assert float(bottoms["T"]) == pytest.approx(float(info["T_profile"][0]))
        assert float(info["T_reboiler"]) == pytest.approx(float(bottoms["T"]))

    def test_condenser_gap_tracks_the_width_of_the_cut(
        self, benzene_toluene_thermo, multicomponent_thermo
    ):
        """The condenser sits below the top stage by the distillate's own
        boiling range: negligible for a sharp binary split, large for a
        distillate carrying a spread of volatilities."""
        sharp = DistillationColumn(
            DistillationColumnParams(
                species_order=["benzene", "toluene"], n_stages=15,
                feed_stage=7, condenser_type="total", P=101325.0,
            ),
            benzene_toluene_thermo,
        )
        _, _, info_sharp = sharp(
            make_stream({"benzene": 50.0, "toluene": 50.0}, T=380.0, P=101325.0),
            R=2.0, D_spec=50.0,
        )
        gap_sharp = (float(info_sharp["T_profile"][-1])
                     - float(info_sharp["T_condenser"]))

        wide = DistillationColumn(
            DistillationColumnParams(
                species_order=["light", "middle", "heavy"], n_stages=15,
                feed_stage=7, condenser_type="total", P=101325.0,
            ),
            multicomponent_thermo,
        )
        _, _, info_wide = wide(
            make_stream({"light": 20.0, "middle": 40.0, "heavy": 40.0},
                        T=400.0, P=101325.0),
            R=3.0, D_spec=30.0,
        )
        gap_wide = (float(info_wide["T_profile"][-1])
                    - float(info_wide["T_condenser"]))

        assert gap_sharp >= 0.0
        assert gap_wide >= gap_sharp

    def test_condenser_duty_condenses_to_the_distillate_state(
        self, benzene_toluene_thermo
    ):
        """Q_cond takes the top stage vapor to the condensed product, so it is
        V_top * (h_D - H_top) with h_D at the condenser temperature."""
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=15,
            feed_stage=7,
            condenser_type="total",
            P=101325.0,
        )
        column = DistillationColumn(params, benzene_toluene_thermo)
        feed = make_stream({"benzene": 50.0, "toluene": 50.0}, T=380.0, P=101325.0)

        _, _, info = column(feed, R=2.0, D_spec=50.0)

        _, H_all = column._compute_stage_enthalpies(
            info["x_profile"], info["y_profile"], info["T_profile"]
        )
        h_D = column._molar_enthalpy(
            info["y_profile"][-1], info["T_condenser"], "liquid"
        )
        expected = float(info["V_profile"][-1]) * float(h_D - H_all[-1])

        assert float(info["Q_condenser"]) == pytest.approx(expected, rel=1e-9)
        assert float(info["Q_condenser"]) < 0.0


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
        """The CMO solve uses the rectifying line at the feed stage itself.

        Applying the operating-line step to the converged profile reproduces
        that profile far more closely with the section assignment the solver
        used than with the other one -- so the profile still carries the
        convention, and flipping it is still loudly visible.

        This asserts a ratio rather than an exact reproduction. Before #211 the
        solver *was* an operating-line recursion, so the step below was its own
        step and the converged profile reproduced it to machine precision.
        #211 replaced that with a tridiagonal Wang-Henke solve, whose converged
        profile is not a fixed point of the recursion; the exactness was a
        property of the old iteration's algebra, not of the convention this
        test is named for. The convention itself is untouched and stays sharply
        detectable -- the correct section is 57x to 2e5x closer across these
        cases -- so the 20x floor below still fails on a flipped predicate
        (mutating `_is_rectifying_cut` to `j > feed_stage` reddens 7 tests in
        this class, the same count that mutation reddened before #211).
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

        # Feed stage: the rectifying line fits the profile, stripping does not.
        at_feed_right = err(feed_stage, rectifying=True)
        at_feed_wrong = err(feed_stage, rectifying=False)
        assert at_feed_wrong > 1e-3
        assert at_feed_wrong > 20.0 * at_feed_right, (
            f"feed stage {feed_stage}: rectifying {at_feed_right:.2e} vs "
            f"stripping {at_feed_wrong:.2e} -- the sections are no longer "
            "distinguishable, so the convention is not being applied"
        )

        # One stage below the feed: the other way round.
        below_right = err(feed_stage - 1, rectifying=False)
        below_wrong = err(feed_stage - 1, rectifying=True)
        assert below_wrong > 1e-3
        assert below_wrong > 20.0 * below_right, (
            f"stage {feed_stage - 1}: stripping {below_right:.2e} vs "
            f"rectifying {below_wrong:.2e} -- the sections are no longer "
            "distinguishable, so the convention is not being applied"
        )

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


class TestFeedThermalCondition:
    """``q`` is the feed's thermal condition in the energy balance too.

    The section flows have always read ``q`` (see
    :class:`TestFeedStageConvention`); these tests pin the other half, the
    feed enthalpy ``q h_liquid(z, T_F) + (1 - q) H_vapor(z, T_F)``, so that a
    saturated-vapour feed brings its latent heat in with it instead of being
    solved as a saturated liquid (#219).
    """

    P = 101325.0
    F = 100.0
    T_FEED = 380.0
    FEED_STAGE = 7
    N_STAGES = 15

    def _column(self, thermo, q):
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=self.N_STAGES,
            feed_stage=self.FEED_STAGE,
            P=self.P,
            q=q,
        )
        return DistillationColumn(params, thermo)

    def _feed(self):
        return make_stream({"benzene": 50.0, "toluene": 50.0}, T=self.T_FEED, P=self.P)

    def _latent_heat(self, thermo):
        """``F (H_vapor - h_liquid)`` for the feed, at the feed temperature."""
        z = jnp.array([0.5, 0.5])
        args = (thermo, ["benzene", "toluene"], z, jnp.asarray(self.T_FEED))
        H_vap = _mixture_molar_enthalpy(*args, "vapor", jnp.asarray(self.P))
        h_liq = _mixture_molar_enthalpy(*args, "liquid", jnp.asarray(self.P))
        return self.F * float(H_vap - h_liq)

    @pytest.mark.parametrize("q", [0.0, 0.5])
    def test_reboiler_duty_drops_by_the_feed_latent_heat(
        self, benzene_toluene_thermo, q
    ):
        """A vapour feed carries latent heat in, so the reboiler need not.

        Against a saturated-liquid feed, the reboiler duty falls by the latent
        heat of the vaporised fraction, ``(1 - q) F (H_vap - h_liq)`` at the
        feed temperature.
        """
        feed = self._feed()
        _, _, info_liq = self._column(benzene_toluene_thermo, 1.0)(
            feed, R=2.0, D_spec=50.0, use_mesh=True
        )
        _, _, info_q = self._column(benzene_toluene_thermo, q)(
            feed, R=2.0, D_spec=50.0, use_mesh=True
        )

        expected_drop = (1.0 - q) * self._latent_heat(benzene_toluene_thermo)
        actual_drop = float(info_liq["Q_reboiler"] - info_q["Q_reboiler"])

        assert actual_drop == pytest.approx(expected_drop, rel=0.02), (
            f"q={q}: reboiler duty moved by {actual_drop / 1e6:.4f} MW, "
            f"expected {expected_drop / 1e6:.4f} MW -- the feed's thermal "
            "condition is not reaching the energy balance"
        )
        # The condenser sees the top of the column, not the feed condition.
        assert float(info_q["Q_condenser"]) == pytest.approx(
            float(info_liq["Q_condenser"]), rel=1e-3
        )

    @pytest.mark.parametrize("q", [1.0, 0.5, 0.0])
    def test_converged_profile_splits_the_feed_by_q(
        self, benzene_toluene_thermo, q
    ):
        """On the CONVERGED profile the feed still splits ``q``/``1 - q``.

        ``TestFeedStageConvention`` checks this on the warm start
        (``n_iter=0``); here the MESH energy balance has run, and it has to
        put the step on the same side.
        """
        _, _, info = self._column(benzene_toluene_thermo, q)(
            self._feed(), R=2.0, D_spec=50.0, use_mesh=True
        )
        L, V = info["L_profile"], info["V_profile"]
        f = self.FEED_STAGE

        # The energy balance moves the surrounding stages by a few percent of
        # the feed rate, so this is an absolute tolerance on F.
        tol = 0.1 * self.F
        assert float(L[f] - L[f + 1]) == pytest.approx(q * self.F, abs=tol)
        assert float(V[f] - V[f - 1]) == pytest.approx(
            (1.0 - q) * self.F, abs=tol
        )

    @pytest.mark.parametrize("q", [1.0, 0.5, 0.0])
    def test_feed_enthalpy_is_the_two_phase_mixture(
        self, benzene_toluene_thermo, q
    ):
        """``h_F`` interpolates the two phase enthalpies at the feed T."""
        column = self._column(benzene_toluene_thermo, q)
        z = jnp.array([0.5, 0.5])
        T = jnp.asarray(self.T_FEED)

        h_liq = column._molar_enthalpy(z, T, "liquid")
        H_vap = column._molar_enthalpy(z, T, "vapor")
        expected = q * float(h_liq) + (1.0 - q) * float(H_vap)

        assert float(column._feed_enthalpy(z, T)) == pytest.approx(expected)

    def test_shortcut_column_reboiler_duty_follows_q(
        self, benzene_toluene_thermo
    ):
        """``ShortcutColumn`` reads ``q`` in its duty, not only in Underwood."""
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)
        feed = self._feed()

        _, _, info_liq = column(feed, R=2.0, P=self.P, q=1.0)
        _, _, info_vap = column(feed, R=2.0, P=self.P, q=0.0)

        drop = float(info_liq["Q_reboiler"] - info_vap["Q_reboiler"])
        assert drop == pytest.approx(
            self._latent_heat(benzene_toluene_thermo), rel=1e-6
        )

    def test_a_subcooled_feed_is_expressible(self):
        """``q > 1`` constructs: it is what a subcooled feed is called.

        On the CMO path ``q`` is the only place the thermal condition can be
        said at all --- ``_cmo_section_rates`` is the whole model there and
        ``T_feed`` never reaches it --- so rejecting ``q > 1`` would make a
        subcooled feed inexpressible rather than safe.
        """
        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=self.N_STAGES,
            feed_stage=self.FEED_STAGE,
            P=self.P,
            q=1.3,
        )
        assert float(params.q) == pytest.approx(1.3)

    @pytest.mark.parametrize("q", [1.3, -0.3])
    def test_the_section_flows_read_q_as_written(self, q):
        """``L_strip = L_rect + q F`` holds outside ``[0, 1]`` too.

        This is the whole point of allowing it: the extra internal reflux a
        subcooled feed generates is exactly ``q F`` with ``q > 1``.
        """
        L_rect, L_strip, V_rect, V_strip = _cmo_section_rates(
            R=2.0, D=50.0, F_total=self.F, q=jnp.asarray(q)
        )

        assert float(L_strip) == pytest.approx(float(L_rect) + q * self.F)
        assert float(V_strip) == pytest.approx(
            float(V_rect) - (1.0 - q) * self.F
        )

    @pytest.mark.parametrize(
        "q, phase", [(1.3, "liquid"), (-0.3, "vapor")]
    )
    def test_the_feed_enthalpy_clamps_where_the_flows_do_not(
        self, benzene_toluene_thermo, q, phase
    ):
        """The one clamp is in the enthalpy, and it is physics not a guard.

        Both phase enthalpies are evaluated at the feed's own temperature, so
        at ``q = 1.3`` the honest answer is ``h_liquid(z, T_feed)``: an
        all-liquid feed below its bubble point, with the subcooling already
        carried by ``T_feed``. Forming ``1.3 h^L - 0.3 H^V`` would subtract
        three tenths of a latent heat that is not there.
        """
        column = self._column(benzene_toluene_thermo, q)
        z = jnp.array([0.5, 0.5])
        T = jnp.asarray(self.T_FEED)

        assert float(column._feed_enthalpy(z, T)) == pytest.approx(
            float(column._molar_enthalpy(z, T, phase))
        )

    def test_underwood_takes_one_minus_q_unclamped(
        self, benzene_toluene_thermo
    ):
        """A subcooled feed shifts Underwood's root, as the textbook has it.

        ``sum(alpha_i z_i / (alpha_i - theta)) = 1 - q`` is stated for any
        ``q``; at ``q = 1.3`` the right-hand side is negative, and the minimum
        reflux it implies is lower than for a saturated liquid.
        """
        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, benzene_toluene_thermo)
        feed = self._feed()

        _, _, sat = column(feed, R=2.0, P=self.P, q=1.0)
        _, _, sub = column(feed, R=2.0, P=self.P, q=1.3)

        assert float(sub["R_min"]) < float(sat["R_min"])

    def test_q_stays_differentiable(self, benzene_toluene_thermo):
        """A traced ``q`` passes the guard, and the duty responds to it.

        ``dQ_reb/dq`` is the feed's latent heat: raising the liquid fraction
        by one mole is one more mole the reboiler has to boil.
        """
        feed = self._feed()

        def reboiler_duty(q):
            _, _, info = self._column(benzene_toluene_thermo, q)(
                feed, R=2.0, D_spec=50.0, use_mesh=True
            )
            return info["Q_reboiler"]

        dQ_dq = float(jax.grad(reboiler_duty)(0.5))
        assert dQ_dq == pytest.approx(
            self._latent_heat(benzene_toluene_thermo), rel=0.05
        )


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


class TestCubicThermoColumn:
    """Rigorous column on a cubic EOS instead of Raoult's law (issue #210).

    The reference the numbers are checked against is the EOS's own rigorous
    flash (``flash_TP_eos``), which is an independent path through the same
    Peng-Robinson object: if the column's stage K-values are right, the
    temperature they put a stage's bubble point at is the temperature that
    flash puts the vapor fraction at zero.
    """

    SPECIES = ["propane", "isobutane", "n_butane", "isopentane",
               "n_pentane", "n_hexane", "n_heptane", "n_octane"]
    P = 10e5

    @pytest.fixture
    def thermo_pair(self):
        from difflow.database import get_critical_props, get_species_data
        from difflow.eos import PengRobinson
        from difflow.thermo import CubicThermo

        ideal = IdealThermo({s: get_species_data(s) for s in self.SPECIES})
        eos = PengRobinson({s: get_critical_props(s) for s in self.SPECIES})
        return ideal, CubicThermo(ideal, eos)

    @pytest.fixture
    def column_params(self):
        return DistillationColumnParams(
            species_order=self.SPECIES,
            n_stages=12,
            feed_stage=6,
            condenser_type="total",
            P=self.P,
        )

    @pytest.fixture
    def feed(self):
        return make_stream(
            dict(zip(self.SPECIES, [10.0, 7.0, 7.0, 8.0, 8.0, 20.0, 10.0, 30.0])),
            T=380.0,
            P=self.P,
        )

    def test_k_values_are_the_fugacity_coefficient_ratio(self, thermo_pair):
        """K_i = phi_i^L(x) / phi_i^V(y) when both compositions are given."""
        _, cubic = thermo_pair
        x = jnp.array([0.30, 0.20, 0.20, 0.10, 0.10, 0.05, 0.03, 0.02])
        y = jnp.array([0.60, 0.15, 0.12, 0.05, 0.04, 0.02, 0.01, 0.01])

        K = cubic.K_values_array(330.0, self.P, x, y)
        expected = (
            cubic.eos.fugacity_coefficient(330.0, self.P, x, "liquid")
            / cubic.eos.fugacity_coefficient(330.0, self.P, y, "vapor")
        )
        assert jnp.allclose(K, expected)

    def test_k_values_dict_matches_array(self, thermo_pair):
        """The dict form is the array form, keyed in the EOS's species order."""
        _, cubic = thermo_pair
        x = jnp.full(len(self.SPECIES), 1.0 / len(self.SPECIES))
        K_arr = cubic.K_values_array(350.0, self.P, x)
        K_dict = cubic.K_values(350.0, self.P, x)
        for i, s in enumerate(cubic.eos.species_order):
            assert float(K_dict[s]) == pytest.approx(float(K_arr[i]))

    def test_k_values_without_composition_fall_back_to_raoult(self, thermo_pair):
        """With no composition there is no fugacity coefficient to form."""
        ideal, cubic = thermo_pair
        assert jnp.allclose(
            cubic.K_values_array(350.0, self.P),
            ideal.K_values_array(350.0, self.P),
        )

    def test_ideal_thermo_ignores_compositions(self, thermo_pair):
        """IdealThermo takes x and y for call compatibility and ignores them."""
        ideal, _ = thermo_pair
        x = jnp.full(len(self.SPECIES), 1.0 / len(self.SPECIES))
        y = jnp.array([0.60, 0.15, 0.12, 0.05, 0.04, 0.02, 0.01, 0.01])
        assert jnp.allclose(
            ideal.K_values_array(350.0, self.P, x, y),
            ideal.K_values_array(350.0, self.P),
        )

    @pytest.mark.parametrize("x_raw", [
        [10.0, 7.0, 7.0, 8.0, 8.0, 20.0, 10.0, 30.0],       # feed
        [0.38, 0.16, 0.13, 0.09, 0.08, 0.10, 0.03, 0.04],   # light cut
        [1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 0.002, 0.25, 0.748],  # heavy cut
    ])
    def test_bubble_point_agrees_with_eos_flash(
        self, thermo_pair, column_params, x_raw
    ):
        """The stage bubble point is where the EOS flash's vapor fraction lifts
        off zero -- checked against ``flash_TP_eos``, not against itself."""
        from difflow.eos import flash_TP_eos

        _, cubic = thermo_pair
        column = DistillationColumn(column_params, cubic)
        x = jnp.array(x_raw)
        x = x / jnp.sum(x)

        T_bub, y = column._bubble_point_T(
            x, jnp.asarray(self.P), T_guess=jnp.asarray(350.0)
        )
        assert jnp.isfinite(T_bub)

        V_below, _, _ = flash_TP_eos(cubic.eos, x, T_bub - 2.0, self.P)
        V_above, _, _ = flash_TP_eos(cubic.eos, x, T_bub + 2.0, self.P)
        assert float(V_below) == pytest.approx(0.0, abs=1e-6)
        assert float(V_above) > 1e-3

        # sum(K x) = 1 at the bubble point, and y is that vapor.
        K = cubic.K_values_array(T_bub, self.P, x)
        assert float(jnp.sum(K * x)) == pytest.approx(1.0, abs=1e-4)
        assert float(jnp.sum(y)) == pytest.approx(1.0)

    def test_eos_bubble_point_differs_from_raoult_for_light_ends(
        self, thermo_pair, column_params
    ):
        """The reason the issue matters: Raoult's law is tens of degrees out for
        light hydrocarbons at 10 bar, and within a degree for the heavy end."""
        ideal, cubic = thermo_pair
        col_i = DistillationColumn(column_params, ideal)
        col_c = DistillationColumn(column_params, cubic)
        P = jnp.asarray(self.P)
        T_guess = jnp.asarray(350.0)

        light = jnp.array([0.70, 0.15, 0.10, 0.03, 0.01, 0.005, 0.003, 0.002])
        light = light / jnp.sum(light)
        heavy = jnp.array([1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 0.002, 0.25, 0.748])
        heavy = heavy / jnp.sum(heavy)

        d_light = float(col_c._bubble_point_T(light, P, T_guess)[0]
                        - col_i._bubble_point_T(light, P, T_guess)[0])
        d_heavy = float(col_c._bubble_point_T(heavy, P, T_guess)[0]
                        - col_i._bubble_point_T(heavy, P, T_guess)[0])

        assert abs(d_light) > 10.0
        assert abs(d_heavy) < 2.0

    def test_column_runs_on_cubic_thermo(self, thermo_pair, column_params, feed):
        """The issue's reproducer: no AttributeError, and a physical answer."""
        _, cubic = thermo_pair
        column = DistillationColumn(column_params, cubic)

        distillate, bottoms, info = column(feed, R=2.0, B_spec=40.0)

        d_flows = get_flows(distillate)
        b_flows = get_flows(bottoms)
        assert all(jnp.isfinite(v).all() for v in d_flows.values())
        assert all(jnp.isfinite(v).all() for v in b_flows.values())
        assert jnp.isfinite(info["T_profile"]).all()

        # Total material balance closes exactly (D and B are specified and the
        # product compositions are normalised); the per-species split is only
        # as tight as the MESH sweep itself, on this thermo package as on the
        # ideal one.
        feed_flows = get_flows(feed)
        F_total = sum(float(v) for v in feed_flows.values())
        assert sum(float(v) for v in d_flows.values()) + sum(
            float(v) for v in b_flows.values()
        ) == pytest.approx(F_total, rel=1e-8)
        for s in self.SPECIES:
            assert float(d_flows[s] + b_flows[s]) == pytest.approx(
                float(feed_flows[s]), rel=0.05
            )

        # Monotonically hotter down the column, and the light key goes up.
        T = info["T_profile"]
        assert float(T[0]) > float(T[-1])
        assert float(d_flows["propane"]) > float(b_flows["propane"])
        assert float(b_flows["n_octane"]) > float(d_flows["n_octane"])

        # Every stage sits at its own bubble point on the EOS K-values.
        x = info["x_profile"]
        for j in (0, 10, column_params.n_stages - 1):
            K = cubic.K_values_array(T[j], self.P, x[j])
            assert float(jnp.sum(K * x[j])) == pytest.approx(1.0, abs=1e-3)

    def test_condenser_is_well_below_the_top_stage_for_a_wide_cut(
        self, thermo_pair, column_params, feed
    ):
        """Where this matters most: a C3-C8 distillate spans a wide boiling
        range, so its bubble point is tens of degrees under the top stage's."""
        _, cubic = thermo_pair
        column = DistillationColumn(column_params, cubic)

        distillate, _, info = column(feed, R=2.0, B_spec=40.0)

        T_bubble, _ = column._bubble_point_T(
            info["y_profile"][-1], jnp.asarray(self.P),
            T_guess=info["T_profile"][-1],
        )
        assert float(distillate["T"]) == pytest.approx(float(T_bubble), rel=1e-9)
        assert float(info["T_profile"][-1]) - float(info["T_condenser"]) > 20.0

        # And it is a real EOS bubble point: the flash puts the vapor fraction
        # at zero there and lifts it off just above.
        from difflow.eos import flash_TP_eos

        x_D = info["y_profile"][-1]
        V_below, _, _ = flash_TP_eos(cubic.eos, x_D, info["T_condenser"] - 2.0, self.P)
        V_above, _, _ = flash_TP_eos(cubic.eos, x_D, info["T_condenser"] + 2.0, self.P)
        assert float(V_below) == pytest.approx(0.0, abs=1e-6)
        assert float(V_above) > 1e-3

    def test_shortcut_column_runs_on_cubic_thermo(self, thermo_pair, feed):
        """The shortcut column reaches the EOS through the same K-value and
        enthalpy interfaces, so it runs on Peng-Robinson too -- and the answer
        is not the ideal one."""
        ideal, cubic = thermo_pair
        params = ShortcutColumnParams(
            species_order=self.SPECIES,
            light_key="n_pentane",
            heavy_key="n_hexane",
            x_D_LK=0.98,
            x_B_HK=0.98,
        )

        results = {}
        for name, thermo in (("ideal", ideal), ("eos", cubic)):
            column = ShortcutColumn(params, thermo)
            distillate, bottoms, info = column(feed, R=2.0, P=self.P, q=1.0)
            assert jnp.isfinite(distillate["T"])
            assert jnp.isfinite(info["N_min"])
            assert float(info["Q_condenser"]) < 0.0
            assert float(info["Q_reboiler"]) > 0.0
            results[name] = info

        # Raoult overstates the light key's volatility at 10 bar, so it
        # promises the separation in fewer stages than the EOS does.
        assert float(results["ideal"]["alpha_LK"]) > float(results["eos"]["alpha_LK"])
        assert float(results["ideal"]["N_min"]) < float(results["eos"]["N_min"])

    def test_shortcut_column_end_temperatures_are_eos_bubble_points(
        self, thermo_pair, feed
    ):
        """And its column ends are EOS bubble points: the flash puts the vapor
        fraction at zero there and lifts it off just above."""
        from difflow.eos import flash_TP_eos

        _, cubic = thermo_pair
        column = ShortcutColumn(
            ShortcutColumnParams(
                species_order=self.SPECIES, light_key="n_pentane",
                heavy_key="n_hexane", x_D_LK=0.98, x_B_HK=0.98,
            ),
            cubic,
        )
        _, _, info = column(feed, R=2.0, P=self.P, q=1.0)

        for key, T_key in (("x_D", "T_top"), ("x_B", "T_bot")):
            comp = jnp.array([info[key][s] for s in self.SPECIES])
            T = info[T_key]
            V_below, _, _ = flash_TP_eos(cubic.eos, comp, T - 2.0, self.P)
            V_above, _, _ = flash_TP_eos(cubic.eos, comp, T + 2.0, self.P)
            assert float(V_below) == pytest.approx(0.0, abs=1e-6)
            assert float(V_above) > 1e-3

    def test_cubic_column_duties_are_physical(
        self, thermo_pair, column_params, feed
    ):
        """The EOS enthalpy path (departure, not Watson Hvap) still gives a
        condenser that removes heat and a reboiler that adds it."""
        _, cubic = thermo_pair
        column = DistillationColumn(column_params, cubic)
        _, _, info = column(feed, R=2.0, B_spec=40.0)

        assert float(info["Q_condenser"]) < 0.0
        assert float(info["Q_reboiler"]) > 0.0


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

    def test_cmo_gives_up_the_energy_balance_not_the_material_balance(self):
        """The two paths must differ where the energy balance bites, and agree
        on the component balance (issue #211).

        Both solvers now run the same tridiagonal component balance, which is
        what closes `D_i + B_i == F_i` on both. The difference between them is
        the L/V profile: CMO freezes it at `L' = L + qF`, MESH corrects it from
        the stage enthalpy balances. On a mixture whose components have very
        different latent heats that correction is large, so this test keeps
        "CMO gives up the energy balance" a measured statement rather than a
        sentence in a docstring -- if the two paths ever agree here, the MESH
        energy balance has stopped doing anything.
        """
        # A and B differ by 2.5x in latent heat, so a mole of B condensing
        # cannot boil a mole of A: constant molar overflow is badly wrong.
        thermo = IdealThermo({
            "A": SpeciesData(
                name="A", MW=60.0, Cp_coeffs=(110.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(25000.0, 0.38, 500.0),
                antoine_coeffs=(13.5, 2700.0, -45.0),
            ),
            "B": SpeciesData(
                name="B", MW=90.0, Cp_coeffs=(150.0, 0.0, 0.0, 0.0),
                Hvap_coeffs=(62000.0, 0.38, 560.0),
                antoine_coeffs=(13.5, 3000.0, -48.0),
            ),
        })
        column = DistillationColumn(
            DistillationColumnParams(
                species_order=["A", "B"], n_stages=12, feed_stage=6, P=101325.0
            ),
            thermo,
        )
        feed = make_stream({"A": 50.0, "B": 50.0}, T=380.0, P=101325.0)

        _, _, info_mesh = column(feed, R=1.2, D_spec=50.0, use_mesh=True)
        L_cmo, _ = column._cmo_flows(100.0, 1.2, 50.0)

        # Stage 0 is the reboiler, whose L is the bottoms by definition on both
        # paths; the correction lives on the trays above it.
        departure = max(
            abs(float(info_mesh["L_profile"][j]) / float(L_cmo[j]) - 1.0)
            for j in range(1, 12)
        )
        assert departure > 0.2, (
            f"MESH L profile only departs from CMO by {departure:.1%}; "
            "the energy balance correction has stopped doing anything"
        )

        # ... and yet both close the component balance.
        feed_flows = get_flows(feed)
        for use_mesh in (True, False):
            distillate, bottoms, info = column(
                feed, R=1.2, D_spec=50.0, use_mesh=use_mesh
            )
            dist_flows = get_flows(distillate)
            bot_flows = get_flows(bottoms)
            for s in ["A", "B"]:
                total = float(dist_flows[s] + bot_flows[s])
                assert total == pytest.approx(float(feed_flows[s]), rel=0.01)
            # R = 1.2 is near minimum reflux for this pair, where the
            # fixed-point iteration converges with a rate close to one -- so
            # the residual sits near 1e-3 and more sweeps barely move it.
            # That is the case the reported diagnostic exists for.
            assert float(info["balance_error_rel"]) < 2e-3
