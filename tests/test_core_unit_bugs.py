"""Tests for core unit bug fixes (#69, #70, #71, #72, #73, #74, #80).

Each test function verifies a specific bug fix and is named after the
corresponding GitHub issue number.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ============================================================================
# Bug #69: CSTR heat duty sign error in simplified path
# ============================================================================

def test_issue_69_cstr_heat_duty_sign_simplified():
    """CSTR simplified heat duty should follow Q > 0 = heat added convention.

    For an exothermic reaction (dH_rxn < 0), heat must be removed,
    so Q should be negative (heat added is negative = heat removed).
    """
    from difflow.units.cstr import CSTR, CSTRParams
    from difflow.streams import make_stream

    # A -> B, exothermic (dH_rxn = -50000 J/mol)
    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])

    stoich = jnp.array([[-1.0], [1.0]])  # A consumed, B produced
    params = CSTRParams(
        V=1.0,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": 0.1},
        species_order=["A", "B"],
        dH_rxn=jnp.array([-50000.0]),  # exothermic
    )

    # No thermo -> simplified path
    cstr = CSTR(params, thermo=None, mode="isothermal")

    inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)
    outlet, info = cstr(inlet, T_spec=350.0, volumetric_flow=0.02)

    Q = info["Q"]
    # Exothermic reaction: Q should be negative (heat must be removed)
    assert float(Q) < 0.0, (
        f"For exothermic reaction, simplified heat duty should be negative "
        f"(heat removed), got Q = {float(Q)}"
    )


# ============================================================================
# Bug #70: PFR adiabatic dT/dV has spurious volumetric flow division
# ============================================================================

def test_issue_70_pfr_adiabatic_temperature_profile():
    """PFR adiabatic dT/dV should be independent of volumetric flow units.

    The correct formula is dT/dV = (-dH_rxn) * r / (F_total * Cp).
    Running at two different volumetric flows (which only affect concentrations)
    should give consistent temperature changes for the same extent of reaction.
    """
    from difflow.units.pfr import PFR, PFRParams
    from difflow.streams import make_stream
    from difflow.thermo import IdealThermo, SpeciesData

    species_data = {
        "A": SpeciesData("A", 100.0, (75.0, 0.0, 0.0, 0.0),
                         (40000.0, 0.38, 500.0), (10.0, 3000.0, -50.0)),
        "B": SpeciesData("B", 100.0, (75.0, 0.0, 0.0, 0.0),
                         (40000.0, 0.38, 500.0), (10.0, 3000.0, -50.0)),
    }
    thermo = IdealThermo(species_data)

    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])

    stoich = jnp.array([[-1.0], [1.0]])

    params = PFRParams(
        V=0.01,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": 1.0},
        species_order=["A", "B"],
        dH_rxn=jnp.array([-50000.0]),  # exothermic
    )

    pfr = PFR(params, thermo=thermo, mode="adiabatic")
    inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

    # Run with two different volumetric flows
    Q_v1 = 0.001  # m^3/s
    Q_v2 = 0.01   # m^3/s (10x larger)

    _, info1 = pfr(inlet, volumetric_flow=Q_v1)
    _, info2 = pfr(inlet, volumetric_flow=Q_v2)

    T_rise_1 = float(info1["profiles"]["T"][-1] - info1["profiles"]["T"][0])
    T_rise_2 = float(info2["profiles"]["T"][-1] - info2["profiles"]["T"][0])

    # For exothermic reaction, temperature should rise
    assert T_rise_1 > 0, f"Expected temperature rise for exothermic adiabatic PFR, got {T_rise_1}"

    # The temperature rises should NOT differ by a factor of Q_v2/Q_v1 = 10
    # They will differ somewhat because different Q_v gives different concentrations
    # and thus different extents of reaction, but not by a factor of 10.
    ratio = T_rise_1 / T_rise_2 if T_rise_2 != 0 else float("inf")
    assert ratio < 5.0, (
        f"Temperature rise ratio {ratio:.1f} is too large; "
        f"suggests spurious Q_v division in dT/dV. "
        f"T_rise_1={T_rise_1:.3f}, T_rise_2={T_rise_2:.3f}"
    )


# ============================================================================
# Bug #71: Fed-batch conversion calculation ignores fed material
# ============================================================================

def test_issue_71_fed_batch_conversion_accounts_for_feed():
    """Fed-batch conversion should account for cumulative fed moles.

    When a significant amount of reactant is added via feed,
    conversion X = 1 - n_A / (n_A0 + n_A_fed) should remain <= 1.
    """
    from difflow.units.fed_batch import FedBatchReactor, FedBatchParams

    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])

    stoich = jnp.array([[-1.0], [1.0]])

    params = FedBatchParams(
        V0=1.0,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": 0.01},
        species_order=["A", "B"],
    )

    reactor = FedBatchReactor(params, mode="isothermal")

    # Start with small amount of A, feed a lot more
    C0 = {"A": 10.0, "B": 0.0}
    feed_composition = {"A": 1000.0, "B": 0.0}  # Concentrated feed

    def feed_rate_fn(t):
        return jnp.array(0.001)  # constant feed rate m^3/s

    _, info = reactor(
        C0=C0,
        T0=350.0,
        P=101325.0,
        t_final=1000.0,
        feed_rate_fn=feed_rate_fn,
        feed_composition=feed_composition,
        n_steps=200,
        use_diffrax=False,
    )

    X_A = float(info["conversion"]["A"])

    # Conversion should be between 0 and 1
    assert 0.0 <= X_A <= 1.0, (
        f"Fed-batch conversion should be in [0, 1] when accounting for fed material, "
        f"got X_A = {X_A:.4f}"
    )


# ============================================================================
# Bug #72: Fed-batch isothermal heat duty has wrong sign
# ============================================================================

def test_issue_72_fed_batch_isothermal_heat_duty_sign():
    """Isothermal fed-batch Q profile: Q < 0 for exothermic (heat removed).

    Convention: Q > 0 = heat added, Q < 0 = heat removed.
    For an exothermic reaction maintained at constant T, heat must be removed.
    """
    from difflow.units.fed_batch import FedBatchReactor, FedBatchParams
    from difflow.thermo import IdealThermo, SpeciesData

    species_data = {
        "A": SpeciesData("A", 100.0, (75.0, 0.0, 0.0, 0.0),
                         (40000.0, 0.38, 500.0), (10.0, 3000.0, -50.0)),
        "B": SpeciesData("B", 100.0, (75.0, 0.0, 0.0, 0.0),
                         (40000.0, 0.38, 500.0), (10.0, 3000.0, -50.0)),
    }
    thermo = IdealThermo(species_data)

    def rate_fn(C, T, params):
        k = params["k"]
        return jnp.array([k * C["A"]])

    stoich = jnp.array([[-1.0], [1.0]])

    params = FedBatchParams(
        V0=1.0,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": 0.1},
        species_order=["A", "B"],
        dH_rxn=jnp.array([-50000.0]),  # exothermic
    )

    reactor = FedBatchReactor(params, thermo=thermo, mode="isothermal")

    C0 = {"A": 100.0, "B": 0.0}
    _, info = reactor(
        C0=C0,
        T0=350.0,
        P=101325.0,
        t_final=100.0,
        n_steps=50,
        use_diffrax=False,
    )

    Q_profile = info["Q"]
    # At the start, when there's plenty of A, Q should be negative (exothermic, heat removed)
    Q_initial = float(Q_profile[1])  # skip t=0 which might be edge case
    assert Q_initial < 0.0, (
        f"For exothermic isothermal batch, Q should be negative (heat removed), "
        f"got Q = {Q_initial:.2f}"
    )


# ============================================================================
# Bug #73: Thermo uses identical Cp for liquid and vapor phases
# ============================================================================

def test_issue_73_thermo_distinct_liquid_vapor_cp():
    """Liquid and vapor Cp should differ when Cp_vapor_coeffs is provided."""
    from difflow.thermo import IdealThermo, SpeciesData

    # Water-like species: Cp_liquid ~ 75 J/mol/K, Cp_vapor ~ 33.6 J/mol/K
    species_data = {
        "water": SpeciesData(
            name="water",
            MW=18.015,
            Cp_coeffs=(75.3, 0.0, 0.0, 0.0),           # liquid Cp
            Hvap_coeffs=(40660.0, 0.38, 647.1),
            antoine_coeffs=(10.196, 1730.63, -39.724),
            Cp_vapor_coeffs=(33.6, 0.0, 0.0, 0.0),      # vapor Cp
        ),
    }
    thermo = IdealThermo(species_data)

    T = 373.15  # boiling point
    Cp_liq = float(thermo.Cp("water", T, phase="liquid"))
    Cp_vap = float(thermo.Cp("water", T, phase="vapor"))

    assert abs(Cp_liq - 75.3) < 0.1, f"Liquid Cp should be ~75.3, got {Cp_liq}"
    assert abs(Cp_vap - 33.6) < 0.1, f"Vapor Cp should be ~33.6, got {Cp_vap}"
    assert Cp_liq != Cp_vap, "Liquid and vapor Cp must differ"


def test_issue_73_thermo_vapor_cp_fallback():
    """When Cp_vapor_coeffs is None, vapor Cp falls back to liquid Cp."""
    from difflow.thermo import IdealThermo, SpeciesData

    species_data = {
        "A": SpeciesData(
            name="A", MW=100.0,
            Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(40000.0, 0.38, 500.0),
            antoine_coeffs=(10.0, 3000.0, -50.0),
            # No Cp_vapor_coeffs -> should fall back to Cp_coeffs
        ),
    }
    thermo = IdealThermo(species_data)

    T = 350.0
    Cp_liq = float(thermo.Cp("A", T, phase="liquid"))
    Cp_vap = float(thermo.Cp("A", T, phase="vapor"))

    assert Cp_liq == Cp_vap, (
        "Without Cp_vapor_coeffs, vapor Cp should fall back to liquid Cp"
    )


def test_issue_73_enthalpy_uses_phase_specific_cp():
    """H_pure should use vapor Cp coefficients for vapor phase when available."""
    from difflow.thermo import IdealThermo, SpeciesData

    species_data = {
        "water": SpeciesData(
            name="water",
            MW=18.015,
            Cp_coeffs=(75.3, 0.0, 0.0, 0.0),           # liquid Cp
            Hvap_coeffs=(40660.0, 0.38, 647.1),
            antoine_coeffs=(10.196, 1730.63, -39.724),
            Cp_vapor_coeffs=(33.6, 0.0, 0.0, 0.0),      # vapor Cp
        ),
    }
    thermo = IdealThermo(species_data)

    T = 400.0
    Tref = 298.15

    H_liq = float(thermo.H_pure("water", T, phase="liquid"))
    H_vap = float(thermo.H_pure("water", T, phase="vapor"))

    # Vapor enthalpy should include Hvap plus Cp_vapor integral
    # H_liq = Cp_liq * (T - Tref) = 75.3 * 101.85 = 7669
    # H_vap = Cp_vap * (T - Tref) + Hvap = 33.6 * 101.85 + Hvap
    # They should differ because of different Cp AND Hvap
    assert H_vap > H_liq, "Vapor enthalpy should exceed liquid enthalpy"

    # Verify that the Cp integral portion differs
    Hvap = float(thermo.Hvap("water", T))
    H_vap_no_hvap = H_vap - Hvap
    assert abs(H_vap_no_hvap - 33.6 * (T - Tref)) < 1.0, (
        f"Vapor enthalpy integral should use vapor Cp, got {H_vap_no_hvap:.1f} "
        f"vs expected {33.6 * (T - Tref):.1f}"
    )


# ============================================================================
# Bug #74: Default molar concentration of 50 mol/m^3 is unrealistic for liquids
# ============================================================================

def test_issue_74_default_concentration_is_liquid_like():
    """DEFAULT_CONCENTRATION should be ~55500 mol/m^3 (liquid water density)."""
    from difflow.constants import DEFAULT_CONCENTRATION

    assert DEFAULT_CONCENTRATION > 1000.0, (
        f"DEFAULT_CONCENTRATION should be >> 50 (liquid-like), "
        f"got {DEFAULT_CONCENTRATION}"
    )
    assert abs(DEFAULT_CONCENTRATION - 55500.0) < 100.0, (
        f"DEFAULT_CONCENTRATION should be ~55500 mol/m^3, "
        f"got {DEFAULT_CONCENTRATION}"
    )


def test_issue_74_cstr_volumetric_flow_default():
    """CSTR default volumetric flow should use liquid-like molar density."""
    from difflow.units.cstr import CSTR, CSTRParams
    from difflow.streams import make_stream

    def rate_fn(C, T, params):
        return jnp.array([params["k"] * C["A"]])

    stoich = jnp.array([[-1.0], [1.0]])
    params = CSTRParams(
        V=1.0,
        rate_fn=rate_fn,
        stoich=stoich,
        rate_params={"k": 0.1},
        species_order=["A", "B"],
    )

    cstr = CSTR(params, mode="isothermal")
    inlet = make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)

    # With default volumetric flow (no explicit specification),
    # Q_v = F_total / 55500 ~ 1/55500 m^3/s
    # tau = V / Q_v = 1.0 / (1/55500) = 55500 s
    # This should give significant conversion for k=0.1
    outlet, info = cstr(inlet, T_spec=350.0)

    X_A = float(info["conversion"]["A"])
    # With liquid molar density, residence time is very large,
    # so conversion should be very high
    assert X_A > 0.9, (
        f"With liquid-like molar density default, CSTR conversion should be high, "
        f"got X_A = {X_A:.4f}"
    )


def test_issue_74_estimate_volumetric_flow_default():
    """estimate_volumetric_flow should default to liquid-like density."""
    from difflow.units.base import estimate_volumetric_flow

    flows = {"A": 1.0, "B": 0.5}
    Q_v = estimate_volumetric_flow(flows)
    # Q_v = (1.0 + 0.5) / 55500 ~ 2.7e-5 m^3/s
    expected = 1.5 / 55500.0
    assert abs(Q_v - expected) / expected < 0.01, (
        f"estimate_volumetric_flow should use ~55500 mol/m^3 default, "
        f"got Q_v = {Q_v:.6e}, expected {expected:.6e}"
    )


# ============================================================================
# Bug #80: Flash mass balance violated by independent x, y normalization
# ============================================================================

def test_issue_80_flash_mass_balance_closure():
    """Flash outlet flows should satisfy component mass balance: z*F = x*L + y*V."""
    from difflow.units.flash import Flash, FlashParams
    from difflow.thermo import IdealThermo, SpeciesData
    from difflow.streams import make_stream, get_flows

    # Binary system: light (high Psat) and heavy (low Psat)
    species_data = {
        "light": SpeciesData(
            "light", 50.0,
            (75.0, 0.0, 0.0, 0.0),
            (30000.0, 0.38, 400.0),
            (10.5, 2800.0, -40.0),  # Higher vapor pressure
        ),
        "heavy": SpeciesData(
            "heavy", 100.0,
            (100.0, 0.0, 0.0, 0.0),
            (45000.0, 0.38, 550.0),
            (10.0, 3500.0, -50.0),  # Lower vapor pressure
        ),
    }
    thermo = IdealThermo(species_data)
    params = FlashParams(species_order=["light", "heavy"])
    flash = Flash(params, thermo)

    inlet = make_stream({"light": 0.6, "heavy": 0.4}, T=350.0, P=50000.0)
    liquid, vapor, info = flash(inlet)

    liquid_flows = get_flows(liquid)
    vapor_flows = get_flows(vapor)
    inlet_flows = get_flows(inlet)

    # Check component mass balance: F_in_i = F_liq_i + F_vap_i
    for species in ["light", "heavy"]:
        F_in = float(inlet_flows[species])
        F_liq = float(liquid_flows[species])
        F_vap = float(vapor_flows[species])
        balance_error = abs(F_in - F_liq - F_vap)
        assert balance_error < 1e-8, (
            f"Component mass balance violated for {species}: "
            f"F_in={F_in:.6f}, F_liq={F_liq:.6f}, F_vap={F_vap:.6f}, "
            f"error={balance_error:.2e}"
        )

    # Also check that total flow is conserved
    F_total_in = sum(inlet_flows.values())
    F_total_out = sum(liquid_flows.values()) + sum(vapor_flows.values())
    assert abs(float(F_total_in - F_total_out)) < 1e-8, (
        f"Total flow not conserved: in={float(F_total_in):.6f}, "
        f"out={float(F_total_out):.6f}"
    )
