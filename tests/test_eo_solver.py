"""Tests for the equation-oriented (EO) solver."""

import jax.numpy as jnp
import pytest

from difflow import (
    CSTR, CSTRParams, Cooler, CoolerParams, EOSolver, EOSolveResult, EOStateLayout,
    Flash, FlashParams, Flowsheet, Heater, HeaterParams, IdealThermo, Mixer,
    SpeciesData, Splitter, Unit, get_flows, make_stream,
)


@pytest.fixture
def thermo():
    return IdealThermo({
        "A": SpeciesData("A", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(35000.0, 0.38, 500.0),
                         antoine_coeffs=(10.0, 3000.0, -50.0)),
        "B": SpeciesData("B", MW=100.0, Cp_coeffs=(75.0, 0.0, 0.0, 0.0),
                         Hvap_coeffs=(30000.0, 0.38, 450.0),
                         antoine_coeffs=(10.0, 2800.0, -40.0)),
    })


@pytest.fixture
def cstr(thermo):
    def rate_fn(C, T, p):
        return jnp.array([p["A"] * jnp.exp(-p["Ea"] / (8.314 * T)) * C["A"]])
    params = CSTRParams(V=jnp.array(1.0), rate_fn=rate_fn,
                        stoich=jnp.array([[-1.0], [1.0]]),
                        rate_params={"A": jnp.array(1e6), "Ea": jnp.array(5e4)},
                        species_order=["A", "B"])
    return CSTR(params, thermo=thermo, mode="isothermal")


def test_state_layout():
    layout = EOStateLayout(species_order=["A", "B"], stream_names=["s1", "s2"])
    streams = {"s1": make_stream({"A": 5.0, "B": 3.0}, T=350.0, P=101325.0),
               "s2": make_stream({"A": 2.0, "B": 8.0}, T=400.0, P=202650.0)}
    rec = layout.unpack(layout.pack(streams))
    assert all(jnp.allclose(rec[n][k], streams[n][k]) for n in streams for k in streams[n])
    assert layout.total_vars == 8 and layout.stream_slice("s2") == slice(4, 8)


def test_unit_residuals_vanish_at_sm_solution(thermo, cstr):
    feed = make_stream({"A": 5.0, "B": 5.0}, T=350.0, P=101325.0)
    feed2 = make_stream({"A": 2.0, "B": 4.0}, T=300.0, P=101325.0)
    out, _ = cstr(feed, T_spec=350.0)
    res = [cstr.eo_residuals([feed], [out], T_spec=350.0)]
    flash = Flash(FlashParams(species_order=["A", "B"]), thermo)
    liq, vap, _ = flash(feed)
    assert float(jnp.max(jnp.abs(flash.eo_residuals([feed], [liq, vap])))) < 1e-4
    mixer = Mixer(species_order=["A", "B"])
    res.append(mixer.eo_residuals([feed, feed2], [mixer(feed, feed2)[0]]))
    splitter = Splitter(species_order=["A", "B"])
    o1, o2, _ = splitter(feed, split_frac=0.6)
    res.append(splitter.eo_residuals([feed], [o1, o2], split_frac=0.6))
    for u in (Heater(HeaterParams(T_out=400.0)), Cooler(CoolerParams(T_out=280.0))):
        res.append(u.eo_residuals([feed], [u(feed)[0]]))
    for r in res:
        assert float(jnp.max(jnp.abs(r))) < 1e-6


def test_eo_solver_result(cstr):
    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 10.0, "B": 0.0}, T=300.0, P=101325.0))
    fs.add_unit(Unit("reactor", cstr, ["feed"], ["reactor_out"], params={"T_spec": 350.0}))
    result = EOSolver(fs).solve(use_sm_init=True, tol=1e-8)
    assert isinstance(result, EOSolveResult) and result.converged
    assert {"feed", "reactor_out"} <= set(result.streams)
    assert "reactor_out" in fs.solve_eo(use_sm_init=False)


def test_unit_without_eo_residuals_falls_back():
    """#206: the fallback branch raised NameError('feed_names')."""
    class Scaler:
        def __call__(self, inlet, factor=1.0):
            return make_stream({k: v * factor for k, v in get_flows(inlet).items()},
                               T=inlet["T"], P=inlet["P"])

    fs = Flowsheet(species_order=["A", "B"])
    fs.add_feed("feed", make_stream({"A": 2.0, "B": 3.0}, T=310.0, P=101325.0))
    fs.add_unit(Unit("scaler", Scaler(), ["feed"], ["scaled"], params={"factor": 2.0}))
    assert jnp.allclose(get_flows(fs.solve_eo(use_sm_init=False)["scaled"])["A"], 4.0)
