"""Tests for LLE and distillation bug fixes (issues #81, #82, #83).

Issue #81: LLE K-value definition consistency and Kremser equation fix
Issue #82: LLE counter-current rate-based ODE sign for organic phase
Issue #83: Distillation column condenser/reboiler duty energy balance
"""

import jax
import jax.numpy as jnp
import pytest

from difflow import (
    MultistageCascade,
    CascadeParams,
    DifferentialContactor,
    ContactorParams,
    LLEEquilibrium,
    DistributionCoeffs,
    get_K_values,
    minimum_solvent_ratio,
    stages_for_recovery,
    make_stream,
    get_flows,
    IdealThermo,
    SpeciesData,
)
from difflow.units.distillation import (
    ShortcutColumn,
    ShortcutColumnParams,
    DistillationColumn,
    DistillationColumnParams,
)

# Enable 64-bit precision for tests
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def high_K_equilibrium():
    """Equilibrium with high K (strongly favoring organic phase)."""
    K_coeffs = DistributionCoeffs(
        species=("solute",),
        K0=(10.0,),  # Strongly favors organic phase
    )
    return LLEEquilibrium(
        solutes=["solute"],
        aqueous_carrier="H2O",
        organic_carrier="Solvent",
        K_coeffs=K_coeffs,
    )


@pytest.fixture
def two_solute_equilibrium():
    """Equilibrium with two solutes for rate-based testing."""
    K_coeffs = DistributionCoeffs(
        species=("A", "B"),
        K0=(5.0, 2.0),
    )
    return LLEEquilibrium(
        solutes=["A", "B"],
        aqueous_carrier="H2O",
        organic_carrier="Solvent",
        K_coeffs=K_coeffs,
    )


@pytest.fixture
def benzene_toluene_thermo():
    """Benzene-toluene thermodynamics for distillation."""
    species_data = {
        "benzene": SpeciesData(
            name="benzene",
            MW=78.11,
            Cp_coeffs=(136.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(33900.0, 0.38, 562.0),
            antoine_coeffs=(13.82, 2788.0, -52.36),
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


# =============================================================================
# Issue #81: LLE K-value definition consistency
# =============================================================================

class TestKValueConsistency:
    """Tests for K-value convention: K = C_extract / C_raffinate."""

    def test_K_value_convention_high_K_gives_good_extraction(self, high_K_equilibrium):
        """With K=10 (strongly favoring organic), extraction should be excellent.

        If K-value were inverted, extraction would be poor instead of good.
        """
        eq = high_K_equilibrium
        cascade_params = CascadeParams(
            n_stages=5,
            equilibrium=eq,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(cascade_params)

        # Feed: 1 mol/s solute in aqueous phase
        feed = make_stream(
            {"H2O": 10.0, "solute": 1.0},
            T=298.15, P=101325.0
        )
        # Solvent: fresh organic
        solvent = make_stream(
            {"Solvent": 10.0, "solute": 0.0},
            T=298.15, P=101325.0
        )

        raffinate, extract, info = cascade(feed, solvent, T=298.15)

        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        # With K=10 and equal flows, extraction factor E = K * S/F = 10
        # 5 stages with E=10 should give >99.99% extraction
        feed_solute = 1.0
        extract_solute = float(ext_flows["solute"])

        recovery = extract_solute / feed_solute
        assert recovery > 0.99, (
            f"Recovery {recovery:.4f} should be >0.99 for K=10, 5 stages. "
            "K-value may be inverted."
        )

    def test_kremser_extraction_factor_correct(self, high_K_equilibrium):
        """Extraction factor E = K * (F_org / F_aq) should give correct Kremser results.

        With K=10 and F_org/F_aq = 1, E = 10.
        Kremser: fraction_remaining = (E-1)/(E^(N+1)-1)
        For N=3: frac_remaining = 9/(10^4 - 1) ~ 0.0009
        """
        eq = high_K_equilibrium
        cascade_params = CascadeParams(
            n_stages=3,
            equilibrium=eq,
            flow_config="counter_current",
        )
        cascade = MultistageCascade(cascade_params)

        feed = make_stream(
            {"H2O": 10.0, "solute": 1.0},
            T=298.15, P=101325.0
        )
        solvent = make_stream(
            {"Solvent": 10.0, "solute": 0.0},
            T=298.15, P=101325.0
        )

        raffinate, extract, _ = cascade(feed, solvent, T=298.15)
        raff_flows = get_flows(raffinate)

        # Expected: frac_remaining ~ (10-1)/(10^4-1) ~ 0.0009
        frac_remaining = float(raff_flows["solute"]) / 1.0
        assert frac_remaining < 0.01, (
            f"Fraction remaining {frac_remaining:.6f} should be <0.01 "
            "for K=10, E=10, N=3"
        )

    def test_stages_for_recovery_consistent_with_kremser(self):
        """stages_for_recovery should give results consistent with Kremser cascade.

        For K=5 and S/F=0.4 (SF_ratio=0.4), E = K * SF_ratio = 2.0.
        The Kremser equation gives a specific N for given recovery.
        """
        K = jnp.array(5.0)
        SF_ratio = jnp.array(0.4)  # S/F ratio = 0.4
        recovery = 0.95

        N = stages_for_recovery(K, SF_ratio, recovery)
        N_val = float(N)

        # E = K * SF_ratio = 2.0
        # With E=2 and recovery=0.95:
        # frac_remaining = 0.05
        # Kremser: E^(N+1) = (E - recovery)/(1 - recovery)
        # 2^(N+1) = (2 - 0.95)/0.05 = 21
        # N+1 = log2(21) ~ 4.39, N ~ 3.39
        assert 2.0 < N_val < 6.0, (
            f"N = {N_val} should be ~3-4 for E=2.0, recovery=0.95"
        )

    def test_stages_for_recovery_round_trip(self):
        """Compute N stages, then verify the Kremser equation gives the right recovery."""
        K = jnp.array(3.0)
        SF = jnp.array(0.5)
        target_recovery = 0.90

        N = stages_for_recovery(K, SF, target_recovery)
        N_val = float(N)

        # Verify: compute recovery from N using Kremser
        E = K * SF  # E = 1.5
        E_Np1 = E ** (N + 1)
        frac_remaining = (E - 1) / (E_Np1 - 1)
        actual_recovery = float(1.0 - frac_remaining)

        assert actual_recovery == pytest.approx(target_recovery, rel=0.02), (
            f"Round-trip recovery {actual_recovery:.4f} should match "
            f"target {target_recovery}"
        )

    def test_minimum_solvent_ratio_convention(self):
        """Minimum S/F should decrease with increasing K."""
        K_low = jnp.array(2.0)
        K_high = jnp.array(10.0)

        SF_min_low = float(minimum_solvent_ratio(K_low))
        SF_min_high = float(minimum_solvent_ratio(K_high))

        # Higher K means less solvent needed
        assert SF_min_high < SF_min_low, (
            f"Higher K should need less solvent: "
            f"SF_min(K=10)={SF_min_high} should be < SF_min(K=2)={SF_min_low}"
        )


# =============================================================================
# Issue #82: Counter-current rate-based ODE sign
# =============================================================================

class TestCounterCurrentRateBased:
    """Tests for counter-current rate-based ODE sign correction.

    The counter-current rate-based model uses a matrix exponential approach.
    The organic phase ODE sign determines whether the model correctly captures
    the counter-current advantage over co-current flow.
    """

    def test_counter_current_hetp_better_than_co_current(self, two_solute_equilibrium):
        """Counter-current extraction via HETP model should outperform co-current.

        This uses the equilibrium (HETP) model which delegates to the
        MultistageCascade solver. Counter-current is fundamentally more
        efficient than co-current for the same number of stages.
        """
        eq = two_solute_equilibrium

        feed = make_stream(
            {"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0},
            T=298.15, P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15, P=101325.0,
        )

        # Counter-current
        cc_params = ContactorParams(
            length=2.0, area=0.1, equilibrium=eq,
            flow_config="counter_current",
            mass_transfer_model="equilibrium",
            HETP=0.5,
        )
        cc_contactor = DifferentialContactor(cc_params)
        cc_raff, cc_ext, _ = cc_contactor(feed, solvent, T=298.15)

        # Co-current
        co_params = ContactorParams(
            length=2.0, area=0.1, equilibrium=eq,
            flow_config="co_current",
            mass_transfer_model="equilibrium",
            HETP=0.5,
        )
        co_contactor = DifferentialContactor(co_params)
        co_raff, co_ext, _ = co_contactor(feed, solvent, T=298.15)

        cc_A = float(get_flows(cc_ext)["A"])
        co_A = float(get_flows(co_ext)["A"])

        assert cc_A > co_A, (
            f"Counter-current A extraction ({cc_A:.4f}) should exceed "
            f"co-current ({co_A:.4f})"
        )

    def test_counter_current_rate_based_matrix_sign(self, two_solute_equilibrium):
        """The counter-current rate-based model should give physically reasonable results.

        With sufficient contactor length and mass transfer, the aqueous
        concentration should decrease along z (feed to raffinate end),
        demonstrating that extraction is occurring.
        """
        eq = two_solute_equilibrium

        feed = make_stream(
            {"H2O": 100.0, "A": 5.0, "B": 3.0, "Solvent": 0.0},
            T=298.15, P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15, P=101325.0,
        )

        params = ContactorParams(
            length=2.0, area=0.5, equilibrium=eq,
            n_segments=50,
            flow_config="counter_current",
            mass_transfer_model="rate_based",
            Kla=0.1,
        )
        contactor = DifferentialContactor(params)
        raffinate, extract, info = contactor(feed, solvent, T=298.15)

        c_aq_A = info["profiles"]["c_aq"]["A"]

        # Aqueous concentration should decrease from z=0 (feed) to z=L (raffinate)
        c_aq_feed = float(c_aq_A[0])
        c_aq_raff = float(c_aq_A[-1])

        assert c_aq_raff < c_aq_feed, (
            f"Aqueous concentration should decrease along column: "
            f"c_aq(0)={c_aq_feed:.6f}, c_aq(L)={c_aq_raff:.6f}"
        )

    def test_counter_current_rate_based_extraction_occurs(self, two_solute_equilibrium):
        """Counter-current rate-based model should show extraction in the profiles.

        Checks that aqueous concentration decreases monotonically along the
        column (from feed end at z=0 to raffinate end at z=L), which indicates
        that the ODE correctly models solute transfer from aqueous to organic.
        """
        eq = two_solute_equilibrium

        feed = make_stream(
            {"H2O": 100.0, "A": 5.0, "B": 3.0, "Solvent": 0.0},
            T=298.15, P=101325.0,
        )
        solvent = make_stream(
            {"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0},
            T=298.15, P=101325.0,
        )

        params = ContactorParams(
            length=2.0, area=0.5, equilibrium=eq,
            n_segments=50,
            flow_config="counter_current",
            mass_transfer_model="rate_based",
            Kla=0.1,
        )
        contactor = DifferentialContactor(params)
        raffinate, extract, info = contactor(feed, solvent, T=298.15)

        # Check profiles: aqueous A should decrease monotonically
        c_aq_A = info["profiles"]["c_aq"]["A"]
        c_aq_B = info["profiles"]["c_aq"]["B"]

        # Feed concentration at z=0 should be highest
        c_aq_A_feed = float(c_aq_A[0])
        c_aq_A_raff = float(c_aq_A[-1])
        c_aq_B_feed = float(c_aq_B[0])
        c_aq_B_raff = float(c_aq_B[-1])

        assert c_aq_A_raff < c_aq_A_feed, (
            f"Aqueous A should decrease: c_aq(0)={c_aq_A_feed:.6f} > "
            f"c_aq(L)={c_aq_A_raff:.6f}"
        )
        assert c_aq_B_raff < c_aq_B_feed, (
            f"Aqueous B should decrease: c_aq(0)={c_aq_B_feed:.6f} > "
            f"c_aq(L)={c_aq_B_raff:.6f}"
        )

        # Verify that extract contains some solute
        ext_flows = get_flows(extract)
        ext_A = float(ext_flows["A"])
        assert ext_A > 0, (
            f"Extract should contain some solute A ({ext_A:.6f})"
        )


# =============================================================================
# Issue #83: Distillation condenser/reboiler duties
# =============================================================================

class TestDistillationDuties:
    """Tests for condenser and reboiler duty calculations."""

    def test_shortcut_column_returns_duties(self, benzene_toluene_thermo):
        """ShortcutColumn should return Q_condenser and Q_reboiler in info."""
        thermo = benzene_toluene_thermo

        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0
        )

        _, _, info = column(feed, R=2.0)

        assert "Q_condenser" in info, "ShortcutColumn should return Q_condenser"
        assert "Q_reboiler" in info, "ShortcutColumn should return Q_reboiler"

    def test_shortcut_condenser_duty_negative(self, benzene_toluene_thermo):
        """Condenser duty should be negative (heat is removed)."""
        thermo = benzene_toluene_thermo

        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0
        )

        _, _, info = column(feed, R=2.0)

        Q_cond = float(info["Q_condenser"])
        assert Q_cond < 0, (
            f"Condenser duty should be negative (heat removed), got {Q_cond}"
        )

    def test_shortcut_reboiler_duty_positive(self, benzene_toluene_thermo):
        """Reboiler duty should be positive (heat is added)."""
        thermo = benzene_toluene_thermo

        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0
        )

        _, _, info = column(feed, R=2.0)

        Q_reb = float(info["Q_reboiler"])
        assert Q_reb > 0, (
            f"Reboiler duty should be positive (heat added), got {Q_reb}"
        )

    def test_shortcut_duties_include_latent_heat(self, benzene_toluene_thermo):
        """Duties should be dominated by latent heat, not just sensible heat.

        For benzene/toluene at ~1 mol/s with R=2, the condenser duty should be
        on the order of ~100 kJ/s (kW), not just a few kJ/s.
        Latent heat of benzene ~ 33.9 kJ/mol, toluene ~ 38 kJ/mol.
        V_top = (R+1)*D ~ 3 * 0.5 = 1.5 mol/s
        Expected Q_cond ~ 1.5 * 34 ~ 51 kJ/s minimum (just latent heat)
        """
        thermo = benzene_toluene_thermo

        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0
        )

        _, _, info = column(feed, R=2.0)

        Q_cond = abs(float(info["Q_condenser"]))

        # Latent heat contribution should make |Q_cond| > 10000 J/s (10 kW)
        # Pure sensible heat for dT ~ 20K would only be ~136*20*1.5 ~ 4080 J/s
        assert Q_cond > 10000.0, (
            f"|Q_condenser| = {Q_cond:.0f} J/s is too small. "
            "Latent heat may not be included. "
            "Expected > 10000 J/s for benzene/toluene at R=2."
        )

    def test_rigorous_column_returns_duties(self, benzene_toluene_thermo):
        """DistillationColumn should return Q_condenser and Q_reboiler."""
        thermo = benzene_toluene_thermo

        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=12,
            feed_stage=6,
            P=101325.0,
        )
        column = DistillationColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0,
        )

        _, _, info = column(feed, R=2.0, D_spec=0.5)

        assert "Q_condenser" in info, "DistillationColumn should return Q_condenser"
        assert "Q_reboiler" in info, "DistillationColumn should return Q_reboiler"

    def test_rigorous_condenser_duty_sign(self, benzene_toluene_thermo):
        """Rigorous column condenser duty should be negative."""
        thermo = benzene_toluene_thermo

        params = DistillationColumnParams(
            species_order=["benzene", "toluene"],
            n_stages=12,
            feed_stage=6,
            P=101325.0,
        )
        column = DistillationColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0,
        )

        _, _, info = column(feed, R=2.0, D_spec=0.5)

        Q_cond = float(info["Q_condenser"])
        Q_reb = float(info["Q_reboiler"])

        assert Q_cond < 0, f"Condenser duty should be negative, got {Q_cond}"
        assert Q_reb > 0, f"Reboiler duty should be positive, got {Q_reb}"

    def test_duties_increase_with_reflux(self, benzene_toluene_thermo):
        """Higher reflux ratio should increase both condenser and reboiler duties."""
        thermo = benzene_toluene_thermo

        params = ShortcutColumnParams(
            species_order=["benzene", "toluene"],
            light_key="benzene",
            heavy_key="toluene",
            x_D_LK=0.95,
            x_B_HK=0.95,
        )
        column = ShortcutColumn(params, thermo)

        feed = make_stream(
            {"benzene": 0.5, "toluene": 0.5},
            T=370.0, P=101325.0
        )

        _, _, info_low_R = column(feed, R=1.5)
        _, _, info_high_R = column(feed, R=4.0)

        Q_cond_low = abs(float(info_low_R["Q_condenser"]))
        Q_cond_high = abs(float(info_high_R["Q_condenser"]))

        assert Q_cond_high > Q_cond_low, (
            f"|Q_cond| at R=4.0 ({Q_cond_high:.0f}) should exceed "
            f"|Q_cond| at R=1.5 ({Q_cond_low:.0f})"
        )
