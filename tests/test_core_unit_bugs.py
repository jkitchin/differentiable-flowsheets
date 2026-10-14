"""Tests for core unit bug fixes (#69, #71, #72, #73, #80).

Each test function verifies a specific bug fix and is named after the
corresponding GitHub issue number.
"""

import jax
import jax.numpy as jnp

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
