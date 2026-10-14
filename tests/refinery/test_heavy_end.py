"""Smoke tests for the heavy end of the one characterization (#301)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr

jax.config.update("jax_enable_x64", True)

PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
T_K = [t + 273.15 for t in (60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680)]
LIGHT = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}
BULK = dict(sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=6.0, nickel_vanadium_wppm=60.0,
            asphaltenes_wt=3.0)


def _assay(**over):
    kw = dict(sg=0.86, light_ends=LIGHT, heavy_end=dr.HeavyEnd(), **BULK)
    kw.update(over)
    return dr.Assay(PCT, T_K, **kw)


@pytest.fixture(scope="module")
def char():
    return dr.characterize(_assay())


def test_characterize_with_heavy_end(char):
    assert float(jnp.sum(char.mass_fraction)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.isfinite(np.asarray(char.Tb)))
    assert np.all(np.isfinite(np.asarray(char.sulfur)))


def test_residue_lump():
    he = dr.HeavyEnd(residue_Tb=1200.0, residue_mw=1400.0, residue_sg=1.08)
    char = dr.characterize(_assay(heavy_end=he))
    assert char.residue_lump
    assert char.names[-1] == "pcresid"


def test_gradient_is_finite(char):
    base = _assay()
    cut_points = tuple(np.asarray(char.cut_edges)[1:-2])

    def f(T80):
        c = dr.characterize(base.with_tbp_point(8, T80), cut_points=cut_points)
        return jnp.sum(c.mass_fraction[len(c.light_names):] * c.Tb)

    assert np.isfinite(float(jax.grad(f)(T_K[8])))


def test_heavy_end_rejects_unusable_data():
    with pytest.raises(ValueError, match="strictly"):
        dr.Assay([0, 50, 90], [300.0, 600.0, 900.0], sg=0.9, heavy_end=dr.HeavyEnd())


def test_blend_grid_from_the_characterization(char):
    grid = dr.BlendCharacterization.from_characterization(char)
    comp = dr.BlendComponent.from_stream("crude", char.stream(10.0, T=300.0, P=1e5), grid)
    assert np.isfinite(float(comp.properties["SG"]))
    with pytest.raises(ValueError, match="outside the characterization"):
        grid.flows({"F_benzene": 1.0})
