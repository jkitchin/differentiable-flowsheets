"""Smoke tests for liquid-liquid extraction unit operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import (
    CascadeParams,
    ContactorParams,
    DifferentialContactor,
    DistributionCoeffs,
    LLEEquilibrium,
    MultistageCascade,
    get_K_values,
    make_stream,
    separation_factor,
)
from difflow.units.lle import NRTLParams, ternary_lle_flash

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def eq():
    return LLEEquilibrium(
        solutes=["A", "B"], aqueous_carrier="H2O", organic_carrier="Solvent",
        K_coeffs=DistributionCoeffs(species=("A", "B"), K0=(2.0, 0.5)),
    )


def _streams():
    feed = make_stream({"H2O": 100.0, "A": 1.0, "B": 1.0, "Solvent": 0.0}, T=298.15, P=101325.0)
    solv = make_stream({"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": 50.0}, T=298.15, P=101325.0)
    return feed, solv


def _balance(feed, solv, r, e):
    for sp in ("A", "B"):
        fin = float(feed[f"F_{sp}"]) + float(solv[f"F_{sp}"])
        assert float(r[f"F_{sp}"] + e[f"F_{sp}"]) == pytest.approx(fin, rel=1e-6)


def _tmodel():
    tau = jnp.array([[0.0, 2.5, 0.5], [2.5, 0.0, 0.3], [0.8, 0.6, 0.0]])
    return NRTLParams(("water", "mibk", "acetic_acid"), tau, jnp.zeros((3, 3)),
                      0.3 * (1 - jnp.eye(3)))


def _neq(model):
    return LLEEquilibrium(solutes=["acetic_acid"], aqueous_carrier="water",
                          organic_carrier="mibk", nrtl_params=model, activity_model="NRTL")


def test_K_values_and_separation_factor():
    K = get_K_values(DistributionCoeffs(species=("A",), K0=(2.0,), dH=(1e4,), Tref=298.15),
                     T=jnp.array(320.0))
    assert np.isfinite(float(K["A"]))
    assert float(separation_factor(jnp.array(4.0), jnp.array(2.0))) == pytest.approx(2.0)


@pytest.mark.parametrize("cfg", ["counter_current", "crossflow"])
def test_cascade_smoke(eq, cfg):
    feed, solv = _streams()
    r, e, info = MultistageCascade(CascadeParams(3, eq, flow_config=cfg,
                                                 stage_efficiency=0.7))(feed, solv)
    _balance(feed, solv, r, e)
    assert "profiles" in info


@pytest.mark.parametrize("model", ["equilibrium", "rate_based"])
def test_contactor_smoke(eq, model):
    feed, solv = _streams()
    p = ContactorParams(length=2.0, area=0.1, equilibrium=eq, HETP=0.5, Kla=3e-3,
                        mass_transfer_model=model)
    r, e, _ = DifferentialContactor(p)(feed, solv)
    _balance(feed, solv, r, e)


def test_ternary_lle_flash_smoke():
    z = jnp.array([40.0, 50.0, 10.0])
    r = ternary_lle_flash(_tmodel(), z, 300.0)
    np.testing.assert_allclose(r["amount_I"] * r["x_I"] + r["amount_II"] * r["x_II"], z,
                               rtol=1e-10)


def test_rigorous_cascade_smoke():
    feed = make_stream({"water": 80.0, "mibk": 0.0, "acetic_acid": 20.0}, 300.0, 1e5)
    solv = make_stream({"water": 0.0, "mibk": 100.0, "acetic_acid": 0.0}, 300.0, 1e5)
    r, e, info = MultistageCascade(CascadeParams(2, _neq(_tmodel()), model="rigorous"))(feed, solv)
    assert bool(info["converged"])
    assert float(r["F_acetic_acid"] + e["F_acetic_acid"]) == pytest.approx(20.0, rel=1e-8)


def test_validation(eq):
    m = _tmodel()
    feed = make_stream({"water": 80.0, "mibk": 0.0, "acetic_acid": 20.0}, 300.0, 1e5)
    solv = make_stream({"water": 0.0, "mibk": 100.0, "acetic_acid": 0.0}, 300.0, 1e5)
    with pytest.raises(ValueError, match="model must be"):
        CascadeParams(3, _neq(m), model="exact")
    with pytest.raises(ValueError):
        CascadeParams(3, eq, flow_config="crosscurrent")
    with pytest.raises(ValueError, match="counter_current"):
        MultistageCascade(CascadeParams(3, _neq(m), flow_config="co_current",
                                        model="rigorous"))(feed, solv)
    with pytest.raises(ValueError, match="integer"):
        MultistageCascade(CascadeParams(2.5, _neq(m), model="rigorous"))(feed, solv)
    eqk = LLEEquilibrium(solutes=["A"], aqueous_carrier="water", organic_carrier="mibk",
                         K_coeffs=DistributionCoeffs(("A",), (2.0,)))
    with pytest.raises(ValueError, match="NRTL"):
        MultistageCascade(CascadeParams(3, eqk, model="rigorous"))(feed, solv)
    ms = _neq(m)
    ms.mutual_solubility = {"aqueous_in_organic": 0.01}
    with pytest.raises(ValueError, match="supersedes"):
        MultistageCascade(CascadeParams(3, ms, model="rigorous"))(feed, solv)
    feed0 = make_stream({"water": 80.0, "mibk": 0.0, "acetic_acid": 0.0}, 300.0, 1e5)
    with pytest.raises(ValueError, match="zero flow"):
        MultistageCascade(CascadeParams(3, _neq(m), model="rigorous"))(feed0, solv)
    f2, s2 = _streams()
    with pytest.raises(ValueError, match="stage_efficiency"):
        MultistageCascade(CascadeParams(3, eq, model="rigorous", stage_efficiency=0.5))(f2, s2)


def test_profiles_under_jit_and_warning(eq):
    feed, solv = _streams()

    def prof(n):
        p = CascadeParams(n, eq, profile_stages=6)
        return jnp.stack(MultistageCascade(p)(feed, solv)[2]["profiles"]["x"]["A"])
    assert np.asarray(jax.jit(prof)(4.0)).shape == (7,)
    with pytest.warns(UserWarning, match="profile_stages"):
        jax.jit(lambda n: MultistageCascade(CascadeParams(n, eq))(feed, solv)[0]["F_A"])(3.0)


def test_grad_finite(eq):
    feed, _ = _streams()

    def f(S):
        solv = make_stream({"H2O": 0.0, "A": 0.0, "B": 0.0, "Solvent": S}, T=298.15, P=101325.0)
        return MultistageCascade(CascadeParams(3.0, eq))(feed, solv)[0]["F_A"]
    assert np.isfinite(float(jax.grad(f)(50.0)))
