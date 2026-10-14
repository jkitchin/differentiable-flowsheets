"""Smoke tests for crude assays cut into pseudo-components."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow  # noqa: F401  (enables x64)
from difflow.thermo import CubicThermo, IdealThermo
from difflow_refinery import Assay, Characterization, characterize, default_cut_points

PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
TBP = [t + 273.15 for t in T_C]
LIGHT = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}
CUTS = tuple(np.linspace(380.0, 1000.0, 12))


@pytest.fixture(scope="module")
def crude():
    return characterize(Assay(PCT, TBP, sg=0.86, light_ends=LIGHT))


def test_assay_refuses_bad_input():
    with pytest.raises(ValueError, match="from 0 to 100"):
        Assay([5, 50, 100], [300.0, 500.0, 800.0], sg=0.85)


def test_cut_points_outside_the_curve_are_refused():
    with pytest.raises(ValueError, match="strictly between"):
        characterize(Assay(PCT, TBP, sg=0.86), cut_points=[250.0, 500.0])


def test_characterize_smoke(crude):
    assert float(jnp.sum(crude.mass_fraction)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(np.asarray(crude.Tc)))
    assert len(default_cut_points(Assay(PCT, TBP, sg=0.86))) > 0
    assert all(n in crude.table() for n in crude.names)


def test_gradient_is_finite():
    def f(tbp):
        c = characterize(Assay(PCT, tbp, sg=0.86), cut_points=CUTS)
        return jnp.sum(c.mass_fraction[4:8])

    assert np.all(np.isfinite(np.asarray(jax.grad(f)(jnp.asarray(TBP)))))


def test_traced_assay_needs_explicit_cut_points():
    with pytest.raises(ValueError, match="cut_points"):
        jax.grad(lambda tbp: jnp.sum(characterize(Assay(PCT, tbp, sg=0.86)).Tb))(jnp.asarray(TBP))


def test_characterization_is_a_pytree_and_jits():
    crude = jax.jit(lambda tbp: characterize(Assay(PCT, tbp, sg=0.86), cut_points=CUTS))(
        jnp.asarray(TBP))
    assert isinstance(crude, Characterization)


def test_export_to_difflow(crude):
    assert crude.pseudo_names[0] in crude.species_data()
    assert isinstance(crude.thermo(), IdealThermo)
    assert isinstance(crude.thermo("pr"), CubicThermo)
    with pytest.raises(ValueError, match="eos"):
        crude.thermo("vdw")
    stream = crude.stream(100.0, T=600.0, P=2e5)
    assert {k[2:] for k in stream if k.startswith("F_")} == set(crude.names)
