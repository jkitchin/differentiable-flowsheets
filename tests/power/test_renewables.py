"""Tests for difflow_power.renewables and the generator capability circle.

The resource models are checked at their corners and against their
analytic derivatives. The OPF side has two load-bearing checks: the
capability circle is actually enforced (and binds where it should),
and the marginal value of resource read from a multiplier equals the
finite difference of the optimal cost in the unit's ceiling, which is
the envelope theorem the value relies on.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

import difflow_power as dp
from difflow_power.opf import acopf_problem, acopf_structure
from difflow_power.residuals import power_state_layout


def _case9_with(gen):
    net = dp.cases.case9()
    net.generators["w"] = gen
    return net


# -- resource models ----------------------------------------------------------


def test_wind_power_curve_corners():
    rated = 3.0

    def curve(v):
        return float(dp.wind_turbine_power(v, rated))

    assert curve(2.9) == 0.0                     # below cut-in
    assert curve(12.0) == pytest.approx(rated)   # rated speed
    assert curve(20.0) == pytest.approx(rated)   # pitch-limited
    assert curve(25.0) == 0.0                    # storm shutdown
    assert 0.0 < curve(8.0) < rated


def test_wind_power_derivative_is_the_cubic():
    v, rated, ci, vr = 8.0, 3.0, 3.0, 12.0
    grad = jax.grad(lambda s: dp.wind_turbine_power(s, rated))(v)
    assert float(grad) == pytest.approx(rated * 3 * v**2 / (vr**3 - ci**3))
    assert float(jax.grad(lambda s: dp.wind_turbine_power(s, rated))(15.0)) == 0.0


def test_thin_air_reaches_rated_power_later():
    sea = float(dp.wind_turbine_power(10.0, 3.0))
    high = float(dp.wind_turbine_power(10.0, 3.0, air_density=1.0))
    assert high == pytest.approx(sea * 1.0 / 1.225)


def test_wind_power_curve_rejects_disordered_speeds():
    with pytest.raises(ValueError, match="cut_in"):
        dp.wind_turbine_power(10.0, 3.0, rated_speed=30.0)


def test_pv_power_at_standard_conditions_is_the_derated_nameplate():
    p = dp.pv_power(1000.0, 25.0, capacity_mw_dc=100.0, temp_coefficient=0.0)
    assert float(p) == pytest.approx(100.0 * 0.86 * 0.96)


def test_pv_power_falls_with_temperature_and_clips_at_the_inverter():
    cool = float(dp.pv_power(800.0, 0.0, capacity_mw_dc=100.0))
    hot = float(dp.pv_power(800.0, 40.0, capacity_mw_dc=100.0))
    assert hot < cool
    # 40 C of ambient is 40 C of cell temperature at -0.47 %/C
    assert hot / cool == pytest.approx(
        (1 - 0.0047 * (40 + 25 - 25)) / (1 - 0.0047 * (25 - 25))
    )
    clipped = dp.pv_power(1100.0, 10.0, capacity_mw_dc=100.0, inverter_mw=80.0)
    assert float(clipped) == pytest.approx(80.0)
    assert float(dp.pv_power(0.0, 10.0, capacity_mw_dc=100.0)) == 0.0


def test_hydro_power_is_rho_g_q_h_eta():
    assert float(dp.hydro_power(100.0, 50.0, efficiency=1.0)) == pytest.approx(
        1000.0 * 9.80665 * 100.0 * 50.0 / 1e6
    )
    check_grads(
        lambda q: dp.hydro_power(q, 50.0), (100.0,), order=1, modes=["rev"]
    )


# -- factories ----------------------------------------------------------------


def test_renewable_generator_is_curtailable_inside_a_circle():
    gen = dp.renewable_generator("5", 40.0, 50.0, name="farm")
    assert (gen.p_min_mw, gen.p_max_mw) == (0.0, 40.0)
    assert (gen.q_min_mvar, gen.q_max_mvar) == (-50.0, 50.0)
    assert gen.s_max_mva == 50.0
    assert gen.p_mw == 40.0
    with pytest.raises(ValueError, match="non-negative"):
        dp.renewable_generator("5", -1.0, 50.0)
    with pytest.raises(ValueError, match="s_max_mva"):
        dp.Generator("5", s_max_mva=0.0)


def test_factories_evaluate_their_resource_model():
    wind = dp.wind_generator("5", 12.0, n_turbines=10, turbine_mw=3.0,
                             losses=0.1)
    assert wind.p_max_mw == pytest.approx(27.0)
    assert wind.s_max_mva == pytest.approx(30.0)

    sun = dp.solar_generator("5", 1200.0, 10.0, capacity_mw_dc=120.0)
    assert sun.s_max_mva == pytest.approx(100.0)       # DC/AC ratio 1.2
    assert sun.p_max_mw == pytest.approx(100.0)        # clipped

    hydro = dp.hydro_generator("5", 100.0, 50.0, machine_mva=60.0)
    assert hydro.p_max_mw == pytest.approx(float(dp.hydro_power(100.0, 50.0)))
    assert hydro.s_max_mva == 60.0


# -- the OPF ------------------------------------------------------------------


def test_structure_carries_one_capability_row_per_rated_unit():
    net = _case9_with(dp.renewable_generator("5", 40.0, 50.0))
    layout = power_state_layout(net)
    structure = acopf_structure(net, layout)
    k = layout.generators.index("w")
    assert structure.capability_generators == (k,)
    assert structure.capability_squared == pytest.approx(((50.0 / 100.0) ** 2,))
    assert structure.inequality_names[-1] == f"capability_{k}"
    assert acopf_structure(
        net, layout, enforce_capability=False
    ).n_capability == 0

    nlp = acopf_problem(net, layout, structure)
    x = dp.flat_start(net, layout)
    x = x.at[layout.index("pg_w")].set(0.3).at[layout.index("qg_w")].set(0.4)
    h = nlp.h(x, net.load_arrays_pu())
    assert h.shape == (structure.n_inequality,)
    assert float(h[-1]) == pytest.approx(0.3**2 + 0.4**2 - 0.5**2)


def test_verify_reports_a_unit_outside_its_circle():
    net = _case9_with(dp.renewable_generator("5", 40.0, 50.0))
    layout = power_state_layout(net)
    x = dp.flat_start(net, layout)
    x = x.at[layout.index("pg_w")].set(0.4).at[layout.index("qg_w")].set(0.4)
    report = dp.operating_report(x, net, layout)
    assert report.generator_violations["w.s"] == pytest.approx(
        100.0 * jnp.hypot(0.4, 0.4) - 50.0
    )


def test_a_full_inverter_trades_watts_for_vars():
    """At available = rating, any var output costs real output."""
    net = _case9_with(dp.renewable_generator("5", 120.0, 120.0))
    result = dp.solve_acopf(net)
    assert result.converged
    k = result.layout.generators.index("w")
    assert f"capability_{k}" in result.binding()
    p, q = result.pg_mw["w"], result.qg_mvar["w"]
    assert p**2 + q**2 == pytest.approx(120.0**2, rel=1e-6)
    assert dp.operating_report(result.x, net, result.layout).feasible
    assert dp.curtailment(result, ["w"])["w"].value_per_mw == pytest.approx(
        0.0, abs=1e-6
    )


def test_more_wind_than_the_system_can_absorb_is_curtailed():
    net = _case9_with(dp.renewable_generator("5", 300.0, 300.0))
    result = dp.solve_acopf(net)
    assert result.converged
    report = dp.curtailment(result, ["w"])["w"]
    assert report.curtailed_mw > 1.0
    assert report.value_per_mw == pytest.approx(0.0, abs=1e-6)
    # The thermal units are at their floors, which is why.
    assert all(f"pg_{g}_min" in result.binding() for g in ("g1", "g2", "g3"))


@pytest.mark.parametrize("available", [60.0, 0.0])
def test_resource_value_matches_finite_differences(available):
    """The envelope theorem, checked by re-solving.

    At 60 MW the unit runs flat out and its ceiling binds; at 0 MW its
    bounds collapse to an equality, so the value comes from a
    different multiplier.
    """
    result = dp.solve_acopf(
        _case9_with(dp.renewable_generator("5", available, 200.0))
    )
    value = dp.curtailment(result, ["w"])["w"].value_per_mw
    step = 0.5

    def cost(p_max):
        return dp.solve_acopf(
            _case9_with(dp.renewable_generator("5", p_max, 200.0))
        ).cost

    if available > 0.0:
        finite = (cost(available - step) - cost(available + step)) / (2 * step)
    else:
        finite = (result.cost - cost(step)) / step
    assert value > 1.0
    assert value == pytest.approx(finite, rel=1e-3)
