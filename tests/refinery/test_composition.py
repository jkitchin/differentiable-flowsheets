"""Smoke tests for hydrocarbon type, hydrogen and heteroatom classes per pseudo-component (#305)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import composition as cm
from difflow_refinery.products import product_properties
from difflow_refinery.thermo import ColumnThermo

jax.config.update("jax_enable_x64", True)

PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
T_K = [t + 273.15 for t in (60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680)]
LIGHT = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}


def heavy_assay():
    return dr.Assay(PCT, T_K, sg=0.86, light_ends=LIGHT, heavy_end=dr.HeavyEnd(),
                    sulfur_wt=1.8, nitrogen_wppm=1500.0)


def light_assay():
    return dr.Assay([0, 5, 10, 30, 50, 70, 90, 95, 100],
                    [t + 273.15 for t in [20, 55, 90, 190, 290, 400, 540, 600, 700]],
                    sg=0.83, light_ends={"propane": 0.4, "n_butane": 1.2}, sulfur_wt=0.4,
                    nitrogen_wppm=600.0)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cm.CompositionRangeWarning)
        return fn(*a, **k)


@pytest.fixture(scope="module")
def char():
    return _quiet(dr.characterize, light_assay(), composition=True)


def test_correlations_are_finite():
    vals = [*cm.riazi_daubert_pna_vgc(378.0, 1.5002, 0.8485, 0.9),
            *cm.riazi_daubert_pna(150.0, 0.80, 1.45, 6.2),
            cm.huang_index(500.0, 0.8), cm.huang_index_heavy(770.2, 0.8172),
            cm.ch_ratio_rd1987(447.3, 0.7342), cm.density_20c(0.7342),
            cm.hydrogen_goossens(1.4119, 0.7301, 142.0), cm.ndm(1.50, 0.90, 350.0, 2.0)["C_A"]]
    assert all(np.isfinite(float(v)) for v in vals)


def test_composition_smoke(char):
    x = np.asarray(char.composition.hc_type)
    assert x.shape == (len(char.names), 4)
    np.testing.assert_allclose(x.sum(axis=1), 1.0, atol=1e-12)


def test_product_properties_with_composition(char):
    thermo = ColumnThermo.from_characterization(char)
    moles = 1000.0 * np.asarray(char.mole_fraction)
    feed = {f"F_{c}": moles[i] for i, c in enumerate(char.names)}
    products = {"a": {k: 0.4 * v for k, v in feed.items()}, "b": {k: 0.6 * v for k, v in feed.items()}}
    props = product_properties(products, thermo, feed, composition=char.composition)
    assert np.isfinite(float(props["a"].composition.sulfur_wt))
    heavy = _quiet(dr.characterize, heavy_assay(), composition=True)
    with pytest.raises(ValueError, match="same components"):
        product_properties(products, thermo, feed, composition=heavy.composition)


def test_measured_curve_override(char):
    meas = cm.CutData({"paraffins": (0.50, 0.40), "naphthenes": (0.35, 0.40),
                       "aromatics": (0.15, 0.20)}, T=(330.0, 450.0))
    comp = _quiet(cm.estimate_composition, char, hc_types=meas)
    assert np.all(np.isfinite(np.asarray(comp.hc_type)))


def test_bad_inputs(char):
    with pytest.raises(ValueError, match="unknown measured types"):
        cm.CutData({"isoparaffins": [0.1]})
    with pytest.raises(ValueError, match="pna_method"):
        cm.CompositionData(pna_method="bergman")
    with pytest.raises(ValueError, match="one entry per pseudo-component"):
        cm.estimate_composition(char, hc_types={"aromatics": [0.1, 0.2]}, warn=False)


def test_heavy_end_warns():
    with pytest.warns(cm.CompositionRangeWarning, match="residue lump"):
        dr.characterize(heavy_assay(), composition=True)


def test_jit_and_gradient(char):
    comp = jax.jit(lambda c: cm.estimate_composition(c, warn=False))(char)
    assert np.all(np.isfinite(np.asarray(comp.hc_type)))
    assay = light_assay()
    cuts = dr.default_cut_points(assay)

    def f(T):
        c = dr.characterize(assay.with_tbp_point(4, T), cut_points=cuts)
        return cm.estimate_composition(c, warn=False).of_flows(jnp.asarray(c.mole_fraction)).aromatics

    assert np.isfinite(float(jax.grad(f)(float(np.asarray(assay.tbp_T)[4]))))
