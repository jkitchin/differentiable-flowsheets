"""Tests for Tier 1 issue fixes: tear defaults, flash phase flags, state bounds."""

import jax.numpy as jnp
import pytest


def test_tear_stream_defaults_propagate():
    from difflow.flowsheet import Flowsheet

    fs = Flowsheet(["A"], default_flow=0.1, default_T=400.0, default_P=2e5)
    stream = fs._make_zero_stream()
    assert float(stream["F_A"]) == pytest.approx(0.1)
    assert float(stream["T"]) == pytest.approx(400.0)
    assert fs._apply_params({}).default_flow == 0.1


def _make_binary_thermo():
    """Create a binary light/heavy thermo for flash tests.

    Antoine coefficients are chosen so that:
    - Light Psat ≈ 1 atm at ~310 K
    - Heavy Psat ≈ 1 atm at ~400 K
    This gives a clear two-phase region around 350 K at 1 atm.
    """
    from difflow.thermo import IdealThermo, SpeciesData

    # log10(Psat/Pa) = A - B/(T+C)
    # For Light: 5.01 = A - B/(310+C) => Psat ≈ 101325 at 310 K
    # For Heavy: 5.01 = A - B/(400+C) => Psat ≈ 101325 at 400 K
    species_data = {
        "Light": SpeciesData(
            name="Light",
            MW=72.0,
            Cp_coeffs=(120.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(30000.0, 0.38, 470.0),
            antoine_coeffs=(12.0, 2170.0, -40.0),
        ),
        "Heavy": SpeciesData(
            name="Heavy",
            MW=114.0,
            Cp_coeffs=(200.0, 0.0, 0.0, 0.0),
            Hvap_coeffs=(40000.0, 0.38, 570.0),
            antoine_coeffs=(12.0, 2520.0, -50.0),
        ),
    }
    return IdealThermo(species_data)


def test_flash_phase_flags_and_balance():
    from difflow.streams import make_stream
    from difflow.units.flash import Flash, FlashParams

    flash = Flash(FlashParams(species_order=["Light", "Heavy"]), _make_binary_thermo())
    feed = make_stream({"Light": 1.0, "Heavy": 1.0}, 375.0, 101325.0)
    liquid, vapor, info = flash(feed)
    assert {"phase_flag", "bubble_check", "dew_check"} <= set(info)
    total_out = sum(float(s[f"F_{c}"]) for s in (liquid, vapor) for c in ("Light", "Heavy"))
    assert total_out == pytest.approx(2.0, rel=1e-6)


def test_state_bounds_enforced():
    from difflow.dynamic.integrators import integrate
    from difflow.dynamic.state import StateSpec, StateVar

    spec = StateSpec([StateVar("n_A", "moles", bounds=(0.0, None))])
    assert float(spec.enforce_bounds(jnp.array([-0.5]))[0]) == 0.0

    def fast_decay(t, y):
        return jnp.array([-100.0 * y[0], 100.0 * y[0]])

    bounds = (jnp.array([0.0, 0.0]), jnp.array([jnp.inf, jnp.inf]))
    result = integrate(fast_decay, jnp.array([1.0, 0.0]), (0.0, 1.0), method="RK4",
                       n_steps=10, bounds=bounds)
    assert jnp.all(result.y_final >= -1e-10)
