"""Smoke tests for Tier 3 enhancements: core, economics, and dynamic."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


def test_kij_matrix_and_grad():
    from difflow.eos import CriticalProperties, PengRobinson, build_kij_matrix

    mat = build_kij_matrix(["a", "b", "c"], {("a", "b"): 0.02})
    assert float(mat[1, 0]) == pytest.approx(0.02)
    pr = PengRobinson({
        "methane": CriticalProperties("methane", 190.6, 4.599e6, 0.011),
        "ethane": CriticalProperties("ethane", 305.3, 4.872e6, 0.099),
    }, k_ij={("methane", "ethane"): 0.02})
    g = jax.grad(lambda k: pr.a_mix(250.0, jnp.array([0.5, 0.5]),
                                    jnp.array([[0.0, k], [k, 0.0]])))(0.02)
    assert jnp.isfinite(g)


def test_antoine_range_info():
    from difflow.database import get_species_data
    from difflow.thermo import IdealThermo

    thermo = IdealThermo({"water": get_species_data("water")})
    _, info = thermo.Psat_with_info("water", 600.0)
    assert info['antoine_in_range'] is False
    assert thermo.validate_antoine("water", 350.0)['in_range'] is True


def test_stream_phase_labels():
    from difflow.streams import combine_streams, make_stream

    s1 = make_stream({"A": 1.0}, T=300.0, P=1e5, phase="liquid")
    s2 = make_stream({"A": 2.0}, T=400.0, P=1e5, phase="vapor")
    result = combine_streams(s1, s2)
    assert result["phase"] == "two_phase" and result["phase_mismatch"] is True


def test_economics_helpers():
    from difflow.economics.capital import REACTOR_COSTS, validate_cost_params
    from difflow.economics.indices import estimate_cepci, get_cepci
    from difflow.economics.utilities import UtilityPrices

    assert validate_cost_params(REACTOR_COSTS["cstr_jacketed"], 500.0)['size_in_range'] is False
    assert get_cepci(2025) > 0 and estimate_cepci(2030) > 0
    assert UtilityPrices.from_region('china').electricity > 0
    with pytest.raises(ValueError, match="Unknown region"):
        UtilityPrices.from_region("antarctica")


def test_integrator_options_and_events():
    from difflow.dynamic.integrators import EventSpec, detect_events, estimate_stiffness, integrate

    def f(t, y):
        return -0.1 * y

    res = integrate(f, jnp.array([1.0]), (0.0, 10.0), "diffrax:tsit5", dtmin=1e-10, dtmax=1.0)
    assert res.info.success
    assert "auto" in integrate(f, jnp.array([1.0]), (0.0, 1.0), "auto").info.message
    assert "is_stiff" in estimate_stiffness(f, jnp.array([1.0]))
    res = integrate(f, jnp.array([1.0]), (0.0, 20.0), "RK4", n_steps=100)
    detected = detect_events(res, [EventSpec("half", lambda t, y: y[0] - 0.5, direction=-1)])
    assert detected[0].name == "half"


def test_dynamic_cstr_variable_volume_smoke():
    from difflow.dynamic.base import DynamicCSTR
    from difflow.dynamic.integrators import integrate_unit
    from difflow.streams import make_stream

    cstr = DynamicCSTR(
        volume=1.0, rate_fn=lambda C, T, p: jnp.array([p["k"] * C["A"]]),
        stoich=jnp.array([[-1], [1]]), species_order=["A", "B"], rate_params={"k": 0.1},
        variable_volume=True, outlet_flow_fn=lambda V: 0.0,
    )
    inputs = {"inlet": make_stream({"A": 1.0, "B": 0.0}, T=350.0, P=101325.0)}
    assert "V" in cstr.state_spec().names
    result = integrate_unit(cstr, inputs, (0.0, 10.0), method="RK4", n_steps=10)
    assert result.info.success and jnp.all(jnp.isfinite(result.y_final))
