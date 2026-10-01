"""The heavy end of the one characterization (#301).

`HeavyEnd` extends a TBP assay past its last point on a probability scale,
cuts the extension into vacuum-range pseudo-components and closes the crude
with a residue lump whose Tb, MW (and optionally SG) are set directly. The
same `Assay` also carries the contaminants (S, N, CCR, Ni+V, asphaltenes)
that the vacuum column and the blend pool need per component.

What is pinned here is what the vacuum column and blending rely on: the
material and the bulk gravity and contaminants all recombine exactly, the
lump is what was asked for, the curve honours its data, and the whole
characterization is differentiable with respect to a TBP point.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.vacuum import heavy_crude

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


def test_material_and_bulk_gravity_recombine(char):
    assert float(jnp.sum(char.mass_fraction)) == pytest.approx(1.0, abs=1e-13)
    assert float(jnp.sum(char.volume_fraction)) == pytest.approx(1.0, abs=1e-13)
    assert float(char.bulk_sg) == pytest.approx(0.86, rel=1e-12)


@pytest.mark.parametrize("key", list(dr.CONTAMINANTS))
def test_bulk_contaminants_recombine(char, key):
    field, unit = dr.CONTAMINANTS[key]
    per = np.asarray(getattr(char, key))
    k = len(char.light_names)
    assert np.all(per[:k] == 0.0), "light ends carry no contaminants"
    total = float(np.sum(np.asarray(char.mass_fraction) * per))
    assert total == pytest.approx(BULK[field] * unit, rel=1e-12)
    # concentrated in the heavy end, as every one of them is in a real crude
    assert np.all(np.diff(per[k:]) >= 0.0)


def test_a_measured_curve_is_used_as_given():
    T_pts = jnp.asarray([400.0, 600.0, 800.0, 1100.0])
    S = jnp.asarray([0.1, 1.0, 2.5, 4.0])
    char = dr.characterize(_assay(sulfur_curve=(T_pts, S)))
    k = len(char.light_names)
    np.testing.assert_allclose(np.asarray(char.sulfur)[k:],
                               np.interp(np.asarray(char.Tb), T_pts, S) * 1e-2, rtol=1e-12)


def test_residue_lump_is_what_was_asked_for():
    he = dr.HeavyEnd(residue_Tb=1200.0, residue_mw=1400.0, residue_sg=1.08)
    char = dr.characterize(_assay(heavy_end=he))
    assert char.residue_lump
    assert char.names[-1] == "pcresid"
    assert float(char.Tb[-1]) == pytest.approx(1200.0)
    assert float(char.MW[-1]) == pytest.approx(1400.0)
    assert float(char.SG[-1]) == pytest.approx(1.08)
    # the lump fixes its own gravity, so the cuts' Watson K absorbs the rest
    assert float(char.bulk_sg) == pytest.approx(0.86, rel=1e-12)
    # its critical constants keep an EOS and the vapour pressure defined
    assert float(char.Tc[-1]) > float(char.Tb[-1])
    assert np.all(np.isfinite(np.asarray(char.Pc)))


def test_cuts_run_in_order_to_the_lump(char):
    Tb = np.asarray(char.Tb)
    edges = np.asarray(char.cut_edges)
    assert np.all(np.diff(Tb) > 0)
    assert np.all((Tb[:-1] > edges[:-2]) & (Tb[:-1] < edges[1:-1]))
    assert edges[-2] == pytest.approx(dr.HeavyEnd().T_max)


def test_the_curve_passes_through_its_data():
    curve = dr.tbp_curve(_assay())
    np.testing.assert_allclose(np.asarray(curve(jnp.asarray(T_K))), np.asarray(PCT) / 100.0,
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(curve.invert(jnp.asarray(PCT) / 100.0)), T_K,
                               rtol=1e-10)
    # monotone past the last point, and short of complete at T_max
    T = jnp.linspace(T_K[-1], dr.HeavyEnd().T_max, 50)
    x = np.asarray(curve(T))
    assert np.all(np.diff(x) > 0) and x[-1] < 1.0


def test_method_defaults():
    assert dr.characterize(_assay()).method == "twu"
    plain = dr.Assay([0, 50, 100], [300.0, 600.0, 900.0], sg=0.85)
    assert dr.characterize(plain).method == "twu"


def test_gradient_with_respect_to_a_tbp_point_matches_fd():
    """d(mass-average Tb and bulk-weighted S of the vacuum range)/d(T at 80%)."""
    base = _assay()
    cut_points = tuple(np.asarray(dr.characterize(base).cut_edges)[1:-2])

    def f(T80):
        c = dr.characterize(base.with_tbp_point(8, T80), cut_points=cut_points)
        k = len(c.light_names)
        m = c.mass_fraction[k:]
        heavy = c.Tb > 643.15
        return jnp.sum(m * c.Tb) + 1e3 * jnp.sum(jnp.where(heavy, m * c.sulfur[k:], 0.0))

    T0 = T_K[8]
    g = float(jax.grad(f)(T0))
    h = 1e-3
    fd = float((f(T0 + h) - f(T0 - h)) / (2 * h))
    assert g == pytest.approx(fd, rel=1e-6)
    assert abs(g) > 1e-3


def test_vacuum_assay_converts_to_the_same_crude():
    """The vacuum package's synthetic crude is one `Assay` away."""
    a = heavy_crude()
    unified = a.to_assay()
    char = dr.characterize(unified)
    assert float(char.bulk_sg) == pytest.approx(float(a.bulk_sg()), rel=1e-12)
    total_S = float(jnp.sum(char.mass_fraction * char.sulfur))
    assert total_S == pytest.approx(a.sulfur_wt / 100.0, rel=1e-12)


@pytest.mark.parametrize("pct, T, msg", [
    ([0, 50, 90], [300.0, 600.0, 900.0], "strictly"),
    ([10, 50, 100], [300.0, 600.0, 900.0], "strictly"),
    ([10, 50, 90], [300.0, 600.0, 1100.0], "T_max"),
    ([10, 50], [300.0, 600.0], "at least 3"),
])
def test_heavy_end_rejects_unusable_data(pct, T, msg):
    with pytest.raises(ValueError, match=msg):
        dr.Assay(pct, T, sg=0.9, heavy_end=dr.HeavyEnd())


# -- into the blend pool ----------------------------------------------------


def test_blend_grid_from_the_characterization(char):
    """The whole crude as a blend component recombines to the assay."""
    grid = dr.BlendCharacterization.from_characterization(char)
    assert grid.names == list(char.names)
    stream = dict(char.stream(10.0, T=300.0, P=1e5), F_water=0.3, F_H2O=0.1)
    comp = dr.BlendComponent.from_stream("crude", stream, grid)
    assert float(comp.properties["SG"]) == pytest.approx(float(char.bulk_sg), rel=1e-12)
    assert float(comp.properties["S_ppm"]) == pytest.approx(BULK["sulfur_wt"] * 1e4, rel=1e-12)
    assert float(comp.properties["N_ppm"]) == pytest.approx(BULK["nitrogen_wppm"], rel=1e-12)
    assert float(comp.properties["CCR_wt"]) == pytest.approx(BULK["ccr_wt"], rel=1e-12)
    # the pool's vapour pressure is the column's: one atmosphere at each Tb
    k = len(char.light_names)
    p = np.asarray(jax.vmap(grid.psat)(grid.Tb))[np.arange(grid.n), np.arange(grid.n)]
    np.testing.assert_allclose(p[k:], 101325.0, rtol=1e-9)


def test_blend_grid_leaves_out_what_the_assay_did_not_give():
    plain = dr.characterize(dr.Assay(PCT, T_K, sg=0.86, light_ends=LIGHT,
                                     heavy_end=dr.HeavyEnd(), sulfur_wt=1.0))
    grid = dr.BlendCharacterization.from_characterization(plain)
    assert set(grid.qualities) == {"S_ppm"}
    assert set(dr.BlendCharacterization.from_characterization(plain, contaminants=True)
               .qualities) == {"S_ppm", "N_ppm", "CCR_wt"}


def test_blend_grid_still_rejects_a_foreign_species(char):
    grid = dr.BlendCharacterization.from_characterization(char)
    with pytest.raises(ValueError, match="outside the characterization"):
        grid.flows({"F_benzene": 1.0})
