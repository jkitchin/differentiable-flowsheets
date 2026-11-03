"""Pitzer model and seawater speciation: they run, are finite, differentiate."""

import jax
import jax.numpy as jnp
import pytest

import difflow_water as dw
from difflow_water.chemistry import pitzer as pz
from difflow_water.chemistry import speciation as spc


def test_pitzer_activity_runs_and_differentiates():
    mc = jnp.zeros(pz.NC).at[pz.CATIONS.index("Na+")].set(1.0)
    ma = jnp.zeros(pz.NA).at[pz.ANIONS.index("Cl-")].set(1.0)
    r = pz.activity(mc, ma, jnp.zeros(pz.NN), 298.15)
    for k in ("ln_gamma_c", "ln_gamma_a", "ln_gamma_n", "phi", "ln_aw"):
        assert jnp.all(jnp.isfinite(r[k])), k
    g = jax.grad(lambda m, T: pz.activity(mc * m, ma * m, jnp.zeros(pz.NN), T)["phi"],
                 argnums=(0, 1))(1.0, 298.15)
    assert all(jnp.isfinite(x) for x in g)


def test_a_phi_is_finite():
    assert jnp.isfinite(pz.a_phi(310.0)) and jnp.isfinite(jax.grad(pz.a_phi)(310.0))


def test_speciate_reference_seawater():
    r = spc.speciate(spc.reference_seawater(0.035), 298.15)
    assert set(r["SI"]) == set(spc.MINERALS)
    for k in ("pH", "pH_macinnes", "ionic_strength", "water_activity", "alkalinity"):
        assert jnp.isfinite(r[k]), k
    assert float(r["residual"]) < 1e-8  # API contract: the balances are solved


def test_speciate_fixed_ph_and_feed_helper():
    tot = dw.seawater_feed(0.035, pH=8.0, T=298.15)
    assert set(tot) == set(spc.TOTALS) and jnp.isfinite(tot["Alk"])


def test_saturation_index_gradient_is_finite():
    f = lambda S, T: spc.speciate(spc.reference_seawater(S), T)["SI"]["Gypsum"]  # noqa: E731
    g = jax.grad(f, argnums=(0, 1))(0.06, 300.0)
    assert all(jnp.isfinite(x) for x in g)


def test_speciate_needs_every_total():
    tot = spc.reference_seawater(0.035)
    del tot["SO4"]
    with pytest.raises(ValueError):
        spc.speciate(tot)
