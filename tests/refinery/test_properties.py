"""Smoke tests for the product property estimates of the blend pool (#330)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, BlendCharacterization, BlendComponent
from difflow_refinery import properties as pr
from difflow_refinery.composition import CompositionRangeWarning

jax.config.update("jax_enable_x64", True)

C = 273.15


@pytest.fixture(scope="module")
def bc():
    PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
    assay = Assay(PCT, [t + C for t in T_C], sg=0.86,
                  light_ends={"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
                              "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5},
                  sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        char = dr.characterize(assay, composition=True)
    return char, BlendCharacterization.from_characterization(char, contaminants=True)


def cut_moles(bc, lo, hi):
    char, b = bc
    Tb = np.asarray(b.Tb) - C
    m = ((Tb >= lo) & (Tb < hi)).astype(float)
    return jnp.asarray(np.asarray(char.mole_fraction) * m)


def cut_stream(bc, lo, hi):
    _, b = bc
    s = {f"F_{n}": f for n, f in zip(b.names, 100.0 * cut_moles(bc, lo, hi))}
    s.update(T=298.15, P=101325.0)
    return s


def test_correlations_are_finite():
    vals = [pr.won_melting_point(226.0), pr.won_heat_of_fusion(226.0),
            pr.freeze_point(jnp.asarray([0.02, 0.01]), jnp.asarray([180.0, 230.0])),
            pr.flash_point(450.0), pr.smoke_point(480.0, 0.80),
            pr.abbott_viscosity(473.0, 0.80)[0],
            pr.walther_viscosity(30.0, pr.T_100F, 5.0, pr.T_210F, 330.0)]
    assert all(np.isfinite(float(v)) for v in vals)


def test_viscosity_pole_warns():
    with pytest.warns(pr.PropertyRangeWarning):
        pr.abbott_viscosity(1100.0, 1.03)


def test_from_stream_estimates_and_override(bc):
    s = cut_stream(bc, 180, 240)
    c = BlendComponent.from_stream("jet", s, bc[1])
    for k in pr.ESTIMATED_PROPERTIES:
        assert np.isfinite(float(c.properties[k])), k
    assert float(BlendComponent.from_stream("jet", s, bc[1], flash_C=42.0).properties["flash_C"]) == 42.0


def test_octane_needs_the_hydrocarbon_types(bc):
    b = bc[1]
    bare = BlendCharacterization(names=b.names, Tb=b.Tb, SG=b.SG, MW=b.MW, Tc=b.Tc,
                                 Pc=b.Pc, omega=b.omega)
    with pytest.raises(KeyError):
        BlendComponent.from_stream("x", cut_stream(bc, -50, 85), bare, estimate=["RON"])


def test_estimates_are_differentiable(bc):
    m0 = cut_moles(bc, 85, 180) + 1e-6
    J = jax.jacfwd(lambda s: pr.estimate_properties(bc[1], m0 * s, ["flash_C", "RON"])["RON"])(
        jnp.ones_like(m0))
    assert bool(jnp.all(jnp.isfinite(J)))
