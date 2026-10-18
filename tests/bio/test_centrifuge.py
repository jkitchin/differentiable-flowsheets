"""Smoke tests for centrifuge unit operations."""

import jax
import jax.numpy as jnp
import pytest

from difflow_bio import Centrifuge, CentrifugeParams, DiscStackCentrifuge, DiscStackParams
from difflow import make_stream, get_flows

jax.config.update("jax_enable_x64", True)


def _feed():
    return make_stream({"cells": 100.0, "substrate": 10.0, "product": 50.0},
                       T=300.0, P=101325.0)


def test_centrifuge_runs_and_closes():
    c = Centrifuge(CentrifugeParams(sigma=5000.0, efficiency=0.7, cell_species="cells"))
    conc, clar, info = c(_feed(), Q=1e-4, concentrate_fraction=0.1)
    assert jnp.isfinite(info["cell_recovery"])
    for s, f in get_flows(_feed()).items():
        assert float(get_flows(conc)[s] + get_flows(clar)[s]) == pytest.approx(float(f), rel=1e-8)


def test_centrifuge_grad_is_finite():
    def f(sigma):
        conc, _, _ = Centrifuge(CentrifugeParams(sigma=sigma, efficiency=0.9))(
            _feed(), Q=1e-3, concentrate_fraction=0.1)
        return conc["F_cells"]
    assert jnp.isfinite(jax.grad(f)(100.0))


def test_disc_stack_runs():
    c = DiscStackCentrifuge(DiscStackParams(n_discs=100, r_outer=0.1, r_inner=0.05,
                                            rpm=6000.0, efficiency=0.8))
    conc, clar, info = c(_feed(), Q=1e-4, concentrate_fraction=0.1)
    assert jnp.isfinite(conc["F_cells"]) and jnp.isfinite(clar["F_cells"])


_DISC = dict(n_discs=60, r_outer=0.12, r_inner=0.05, rpm=6000.0, efficiency=0.55)
_BINS = dict(d_bins=[2e-6, 4e-6, 6e-6, 8e-6], mass_fractions=[0.1, 0.4, 0.4, 0.1])


def test_disc_stack_size_bins_smoke_and_keys():
    c = DiscStackCentrifuge(DiscStackParams(**_DISC, **_BINS))
    conc, clar, info = c(_feed(), Q=3e-3, rho_particle=1090.0, rho_fluid=1020.0, viscosity=1.5e-3)
    assert info["grade_efficiency"].shape == (4,)
    assert jnp.all(jnp.isfinite(info["grade_efficiency"]))
    assert float(info["overall_efficiency"]) == float(info["cell_recovery"])
    for s, f in get_flows(_feed()).items():
        assert float(get_flows(conc)[s] + get_flows(clar)[s]) == pytest.approx(float(f), rel=1e-8)


def test_disc_stack_size_bins_grad_in_rpm_is_finite():
    def rec(rpm):
        c = DiscStackCentrifuge(DiscStackParams(**{**_DISC, "rpm": rpm}, **_BINS))
        return c(_feed(), Q=3e-3, rho_particle=1090.0, rho_fluid=1020.0, viscosity=1.5e-3)[2]["overall_efficiency"]
    assert jnp.isfinite(jax.grad(rec)(6000.0))


def test_disc_stack_single_bin_matches_single_size_default():
    kw = dict(Q=3e-3, rho_particle=1090.0, rho_fluid=1020.0, viscosity=1.5e-3)
    one = DiscStackCentrifuge(DiscStackParams(**_DISC))(_feed(), d_particle=3e-6, **kw)
    binned = DiscStackCentrifuge(DiscStackParams(**_DISC, d_bins=[3e-6], mass_fractions=[1.0]))(_feed(), **kw)
    assert "grade_efficiency" not in one[2]
    assert float(binned[2]["cell_recovery"]) == pytest.approx(float(one[2]["cell_recovery"]), rel=1e-12)
    assert float(binned[0]["F_cells"]) == pytest.approx(float(one[0]["F_cells"]), rel=1e-12)


@pytest.mark.parametrize("kw", [
    dict(d_bins=[1e-6, 2e-6]),
    dict(mass_fractions=[0.5, 0.5]),
    dict(d_bins=[1e-6, 2e-6], mass_fractions=[1.0]),
    dict(d_bins=[], mass_fractions=[]),
])
def test_disc_stack_size_bins_validation(kw):
    with pytest.raises(ValueError):
        DiscStackParams(**_DISC, **kw)
