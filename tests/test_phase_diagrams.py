"""Smoke, API and validation tests for difflow.phase_diagrams and the activity models."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow.activity import (
    ActivityModel, MargulesParams, VanLaarParams, WilsonParams, activity_gamma,
)
from difflow.database import get_species_data
from difflow.phase_diagrams import (
    difference_point, find_azeotrope, hunter_nash, lever_rule, mix, pxy,
    tabulated_lle, ternary_lle, tie_line_through, txy, xy_curve,
)
from difflow.thermo import IdealThermo, SpeciesData
from difflow.units.lle import (
    DistributionCoeffs, LLEEquilibrium, NRTLParams, UNIQUACParams,
)

jax.config.update("jax_enable_x64", True)

ATM = 101325.0
BT = ["benzene", "toluene"]


def _sd(name, antoine, Tc):
    return SpeciesData(name=name, MW=1.0, Cp_coeffs=(0, 0, 0, 0),
                       Hvap_coeffs=(1.0, 0.38, Tc), antoine_coeffs=antoine)


@pytest.fixture(scope="module")
def bt_thermo():
    return IdealThermo({s: get_species_data(s) for s in BT})


@pytest.fixture(scope="module")
def ew_thermo():
    return IdealThermo({
        "ethanol": _sd("ethanol", (10.24677, 1598.673, -46.424), 514.0),
        "water": _sd("water", (9.6543, 1435.264, -64.848), 647.0),
    })


@pytest.fixture(scope="module")
def ew_nrtl():
    return NRTLParams(
        species=("ethanol", "water"),
        a=jnp.array([[0.0, -0.8009], [3.4578, 0.0]]),
        b=jnp.array([[0.0, 246.18], [-586.0809, 0.0]]),
        alpha=jnp.array([[0.0, 0.3], [0.3, 0.0]]),
    )


@pytest.fixture
def lle_data():
    c = np.linspace(0.0, 0.3, 10)
    xI = np.stack([0.98 - c, np.full_like(c, 0.02), c], axis=1)
    xII = np.stack([0.03 + 0.2 * c, 0.97 - 1.7 * c, 1.5 * c], axis=1)
    return tabulated_lle(np.stack([xI, xII], axis=1))


F = np.array([0.75, 0.0, 0.25])
S = np.array([0.0, 1.0, 0.0])


def test_binary_diagrams_smoke(bt_thermo):
    t = txy(bt_thermo, BT, ATM, n=5)
    p = pxy(bt_thermo, BT, 370.0, n=5)
    xy = xy_curve(bt_thermo, BT, ATM, n=5)
    for arr in (t["T"], t["y"], p["P"], xy["alpha"]):
        assert arr.shape == (5,) and np.all(np.isfinite(np.asarray(arr)))
    with pytest.raises(ValueError):
        txy(bt_thermo, ["benzene"], ATM)


def test_find_azeotrope_smoke(ew_thermo, ew_nrtl):
    r = find_azeotrope(ew_thermo, ["ethanol", "water"], T=351.29, activity_model=ew_nrtl)
    assert bool(r["found"]) and np.isfinite(float(r["P"]))


def test_activity_models_protocol(ew_nrtl):
    uq = UNIQUACParams(("a", "b"), jnp.array([1.0, 1.5]), jnp.array([1.0, 1.4]),
                       jnp.zeros((2, 2)), jnp.zeros((2, 2)))
    for m in (ew_nrtl, uq, WilsonParams.binary(("a", "b"), [1., 1.], 0., 0.),
              MargulesParams(("a", "b"), 1., 1.), VanLaarParams(("a", "b"), 1., 1.)):
        assert isinstance(m, ActivityModel)
        assert activity_gamma(m, jnp.array([0.4, 0.6]), 300.0).shape == (2,)
    g = activity_gamma(lambda x, T: jnp.exp(0.5 * x), jnp.array([0.2, 0.8]), 300.0)
    assert g.shape == (2,)
    with pytest.raises(TypeError):
        activity_gamma(3.0, jnp.array([0.5, 0.5]), 300.0)
    with pytest.raises(ValueError):
        MargulesParams(("a", "b", "c"), 1.0, 1.0)


def test_activity_gamma_grads_finite():
    x = jnp.array([0.3, 0.7])
    cases = [
        (lambda a: MargulesParams(("a", "b"), a, 1.9).gamma(x, 300.0)[0], 1.0),
        (lambda a: VanLaarParams(("a", "b"), a, 0.8).gamma(x, 300.0)[0], 1.0),
        (lambda a: WilsonParams.binary(("a", "b"), [4e-5, 1.8e-5], a, 2000.0).gamma(x, 340.0)[0], 700.0),
    ]
    for f, a0 in cases:
        assert np.isfinite(float(jax.grad(f)(a0)))


def test_ternary_lle_rejects_k_model():
    lle = LLEEquilibrium(solutes=["s"], aqueous_carrier="b", organic_carrier="a",
                         K_coeffs=DistributionCoeffs(species=("s",), K0=(2.0,)))
    with pytest.raises(ValueError):
        ternary_lle(lle, 300.0)


def test_mix_lever_and_difference_point():
    A, B = jnp.array([0.7, 0.1, 0.2]), jnp.array([0.05, 0.8, 0.15])
    M = mix(jnp.stack([A, B]), jnp.array([2.0, 6.0]))
    assert float(M.sum()) == pytest.approx(1.0)
    fA, fB = lever_rule(M, A, B)
    assert float(fA + fB) == pytest.approx(1.0)
    g = jax.grad(lambda a: lever_rule(mix(jnp.stack([A, B]), a), A, B)[0])(jnp.array([2.0, 6.0]))
    assert np.all(np.isfinite(np.asarray(g)))
    RN = np.array([0.9, 0.05, 0.05])
    Mf = np.asarray(mix(np.stack([F, S]), np.array([1.0, 2.0])))
    dp = difference_point(F, RN + 1.7 * (Mf - RN), S, RN, 1.0, None, 2.0)
    assert np.all(np.isfinite(np.asarray(dp["composition"])))


def test_hunter_nash_smoke_and_grad(lle_data):
    r = hunter_nash(lle_data, F, S, 0.02, S_over_F=0.8)
    assert r["feasible"] and np.isfinite(float(r["n_stages"]))
    g = jax.grad(lambda sf: hunter_nash(lle_data, F, S, 0.02, S_over_F=sf)["n_stages"])(0.8)
    assert np.isfinite(float(g))


def test_tie_line_through(lle_data):
    r = tie_line_through(lle_data, np.array([0.45, 0.45, 0.10]))
    assert np.all(np.isfinite(np.asarray(r["x_I"])))
    out = tie_line_through(lle_data, np.array([0.99, 0.005, 0.005]), on_outside="flag")
    assert out["two_phase"] is False
    with pytest.raises(ValueError):
        tie_line_through(lle_data, np.array([0.99, 0.005, 0.005]))
    with pytest.raises(ValueError, match="on_outside"):
        tie_line_through(lle_data, np.array([0.45, 0.45, 0.10]), on_outside="nan")
    with pytest.raises(TypeError):
        jax.jit(lambda m: tie_line_through(lle_data, m)["x_I"])(jnp.array([0.45, 0.45, 0.10]))


def test_tabulated_lle_format_and_validation():
    c = np.linspace(0.0, 0.3, 5)
    tl = np.stack([np.stack([0.98 - c, np.full_like(c, 0.02), c], axis=1),
                   np.stack([0.03 + 0.2 * c, 0.97 - 1.7 * c, 1.5 * c], axis=1)], axis=1)
    d = tabulated_lle(tl)
    assert d["tie_lines"].shape == (5, 2, 3) and d["binodal"].shape == (10, 3)
    with pytest.raises(ValueError):
        tabulated_lle(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        tabulated_lle(tl[:, ::-1, :])


def test_plot_ternary_accepts_tabulated(lle_data):
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    from difflow.visualization import plot_ternary
    assert plot_ternary(lle_data) is not None
