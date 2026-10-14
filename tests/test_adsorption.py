"""Smoke tests for difflow.adsorption: isotherms, batch and fixed-bed
adsorption, ion exchange and chromatography."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import adsorption as ad

jax.config.update("jax_enable_x64", True)


def test_isotherms_and_fit_smoke():
    c = jnp.array([0.05, 0.2, 0.5, 1.0, 2.0, 4.0])
    for f in (ad.linear_isotherm(c, 2.0), ad.langmuir_isotherm(c, 2.0, 1.0),
              ad.freundlich_isotherm(c, 0.5, 0.3)):
        assert f.shape == c.shape and bool(jnp.all(jnp.isfinite(f)))
    fit = ad.fit_isotherm(c, ad.langmuir_isotherm(c, 3.0, 0.8), "langmuir")
    assert {"params", "r2"} <= set(fit)


def test_freundlich_zero_is_finite():
    assert float(ad.freundlich_isotherm(0.0, 0.2, 0.3)) == 0.0
    assert np.isfinite(float(jax.grad(ad.freundlich_isotherm, argnums=1)(0.0, 0.2, 0.3)))


def test_langmuir_K_d_validation():
    c = jnp.array([0.1, 1.0])
    with pytest.raises(ValueError):
        ad.langmuir_isotherm(c, 3.0)
    with pytest.raises(ValueError):
        ad.langmuir_isotherm(c, 3.0, 4.0, K_d=0.25)


def test_batch_adsorption_balance_and_grad():
    r = ad.batch_adsorption(0.21, 1.0, 1.40, "freundlich", K=0.199, n=0.229)
    assert float(1.0 * (0.21 - r["c"])) == pytest.approx(float(1.40 * r["q"]), rel=1e-10)
    g = jax.grad(lambda m: ad.batch_adsorption(0.21, 1.0, m, "langmuir", q_max=0.15, K=60.0)["c"])(1.4)
    assert np.isfinite(float(g))


def test_lub_and_break_time_smoke():
    t = jnp.linspace(0.0, 10.0, 21)
    r = 0.5 * (1.0 + jnp.tanh((t - 4.0) / 1.5))
    out = ad.lub_analysis(t, r, 1.0, 0.05)
    assert all(np.isfinite(float(out[k])) for k in ("t_break", "t_stoich", "LUB", "L_used"))
    assert np.isfinite(float(ad.break_time(t, r, 0.05)))
    assert np.isfinite(float(ad.lub_break_time(20.0, 5.0, 14.0, 4.0)))


def test_fixed_bed_smoke_and_grad():
    T = jnp.linspace(0.0, 1000.0, 51)

    def area(q_max):
        out = ad.fixed_bed_breakthrough(T, L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0, c_feed=1.0,
                                        k_ldf=0.05, isotherm="langmuir", n_cells=10, q_max=q_max,
                                        K=1.0, method="mol")
        return jnp.trapezoid(1.0 - out["c_out"], T)

    assert np.isfinite(float(jax.grad(area)(5.0)))


def test_fixed_bed_unknown_method_raises():
    with pytest.raises(ValueError):
        ad.fixed_bed_breakthrough(jnp.linspace(0, 10, 3), L=1.0, v_s=0.01, epsilon=0.4, rho_b=1.0,
                                  c_feed=1.0, k_ldf=0.05, isotherm="linear", K=1.0, method="euler")


@pytest.mark.parametrize("solver", ["kvaerno5"])
def test_fixed_bed_reverse_mode_both_solvers(solver):
    """The implicit solver differentiates in reverse mode (#455)."""
    t = jnp.linspace(0.0, 60.0, 25)

    def f(k):
        r = ad.fixed_bed_breakthrough(t, 0.5, 0.01, 0.4, 1.0, 1.0, k, q_max=10.0,
                                      K=2.0, n_cells=20, solver=solver, method="mol")
        return jnp.sum(r["c_out"])

    assert np.isfinite(float(jax.grad(f)(0.05)))


def test_ion_exchange_smoke():
    y = ad.ion_exchange_equilibrium(0.3, 2.0, z_A=2, z_B=1, C_total=0.05, Q=2.0)
    assert 0.0 < float(y) < 1.0
    g = jax.grad(lambda x: ad.ion_exchange_equilibrium(x, 2.0, z_A=2, z_B=1, C_total=0.05, Q=2.0))(0.3)
    assert np.isfinite(float(g))


def test_chromatography_smoke():
    assert float(ad.retention_factor(6.0, 2.0)) == 2.0
    assert float(ad.resolution(10.0, 12.0, 1.0, 1.0)) == 2.0
    for v in (ad.plate_count(10.0, 1.0), ad.purnell_resolution(2500.0, 1.2, 3.6),
              ad.plates_for_resolution(1.5, 1.2, 3.6), *ad.van_deemter_optimum(0.002, 0.004, 0.01)):
        assert np.isfinite(float(v))
    assert ad.gaussian_peak(jnp.linspace(0.0, 20.0, 11), 10.0, 400.0).shape == (11,)
