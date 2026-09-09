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

    def test_cubic_column_gradients(self, thermo_pair, column_params, feed):
        """AD still runs through the EOS column, and matches finite difference."""
        _, cubic = thermo_pair
        column = DistillationColumn(column_params, cubic)

        def hexane_purity(R):
            distillate, _, _ = column(feed, R=R, B_spec=40.0)
            total = sum(distillate[f"F_{s}"] for s in self.SPECIES)
            return distillate["F_n_hexane"] / total

        g = jax.grad(hexane_purity)(2.0)
        assert jnp.isfinite(g)

        eps = 1e-3
        fd = (float(hexane_purity(2.0 + eps))
              - float(hexane_purity(2.0 - eps))) / (2 * eps)
        assert float(g) == pytest.approx(fd, rel=1e-4)

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
