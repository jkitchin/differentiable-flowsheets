"""Hydrocarbon type, hydrogen and heteroatom classes per pseudo-component (#305).

What is pinned:

* the correlations' published coefficients and the checks that can be made
  on them without the book: structural identities (the 2B4.1 fractions sum
  to one identically), an independent coding of procedure 2B4.1 (pychemqt's
  ``PNA_Riazi`` and ``H_Goossens`` doctests), and pure compounds whose
  refractive index, density and C/H ratio are known exactly. The worked
  examples of Riazi's MNL50 for the composition correlations are NOT
  reproduced here: they could not be checked offline, and none is claimed;
* every cut's PNA is a composition (non-negative, sums to one), olefins are
  zero in a straight run, light ends are paraffins with their exact hydrogen;
* hydrogen, sulfur (and each sulfur class), nitrogen (and each class) and the
  volume-averaged types balance to 1e-12 through ``product_properties`` on
  two assays;
* product PNA and hydrogen are differentiable in a TBP point and the bulk
  gravity (AD against central differences), and so is a measured override;
* the composition reaches the blend pool as ``*_vol`` qualities.
"""

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


def heavy_assay(**over):
    kw = dict(sg=0.86, light_ends=LIGHT, heavy_end=dr.HeavyEnd(), sulfur_wt=1.8,
              nitrogen_wppm=1500.0)
    kw.update(over)
    return dr.Assay(PCT, T_K, **kw)


def light_assay(**over):
    kw = dict(sg=0.83, light_ends={"propane": 0.4, "n_butane": 1.2}, sulfur_wt=0.4,
              nitrogen_wppm=600.0)
    kw.update(over)
    return dr.Assay([0, 5, 10, 30, 50, 70, 90, 95, 100],
                    [t + 273.15 for t in [20, 55, 90, 190, 290, 400, 540, 600, 700]], **kw)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", cm.CompositionRangeWarning)
        return fn(*a, **k)


@pytest.fixture(scope="module")
def chars():
    return {"heavy": _quiet(dr.characterize, heavy_assay(), composition=True),
            "light": _quiet(dr.characterize, light_assay(), composition=True)}


# =============================================================================
# The correlations
# =============================================================================


class TestCorrelations:
    def test_2b4_1_vgc_coefficients_sum_to_a_composition_identically(self):
        """x_P + x_N + x_A = 1 for ANY (Ri, VGC): the published constants'
        a's sum to one and their b's and c's to zero, in both branches."""
        n20 = jnp.linspace(1.40, 1.58, 7)
        d20 = jnp.linspace(0.70, 1.02, 7)[::-1]
        VGC = jnp.linspace(0.70, 1.00, 7)
        for MW in (120.0, 350.0):
            xp, xn, xa = cm.riazi_daubert_pna_vgc(MW, n20, VGC, d20, blend=False)
            np.testing.assert_allclose(np.asarray(xp + xn + xa), 1.0, atol=1e-13)
            # and not by x_A's definition alone: P and N are the published lines
            Ri = n20 - d20 / 2
            if MW < 200:
                np.testing.assert_allclose(np.asarray(xp), -13.359 + 14.4591 * Ri - 1.41344 * VGC, rtol=1e-14)
            else:
                np.testing.assert_allclose(np.asarray(xp), 2.5737 + 1.0133 * Ri - 3.573 * VGC, rtol=1e-14)
        # The x_A constants, by difference. These are also the values recalled
        # for the tabulated x_A rows -- a recollection, not checked against
        # the book; what is asserted is the arithmetic.
        # light: -9.6235, 8.8739, 0.59827
        assert 1 - (-13.359) - 23.9825 == pytest.approx(-9.6235, abs=1e-10)
        assert -(14.4591 - 23.333) == pytest.approx(8.8739, abs=1e-10)
        assert -(-1.41344 + 0.81517) == pytest.approx(0.59827, abs=1e-10)
        # heavy: -4.0377, 2.6568, 1.60988
        assert 1 - 2.5737 - 2.464 == pytest.approx(-4.0377, abs=1e-10)
        assert -(1.0133 - 3.6701) == pytest.approx(2.6568, abs=1e-10)
        assert -(-3.573 + 1.96312) == pytest.approx(1.60988, abs=1e-10)

    def test_2b4_1_matches_an_independent_coding(self):
        """pychemqt's ``PNA_Riazi`` doctest (API TDB 2B4.1, VGC form):
        M 378, SG 0.9046, n 1.5002, d20 0.9, VGC 0.8485 -> 0.606 0.275 0.118.
        An independent implementation of the same procedure, not the book's
        worked example."""
        xp, xn, xa = cm.riazi_daubert_pna_vgc(378.0, 1.5002, 0.8485, 0.9, blend=False)
        assert (round(float(xp), 3), round(float(xn), 3), round(float(xa), 3)) == (0.606, 0.275, 0.118)
        # the blended form is the published one this far from the M=200 join
        b = cm.riazi_daubert_pna_vgc(378.0, 1.5002, 0.8485, 0.9)
        np.testing.assert_allclose(np.asarray(b), np.asarray([xp, xn, xa]), atol=1e-6)

    def test_2b4_1_ch_form_branches(self):
        # by construction, and the published constants as coded
        MW, SG, n, CH = 150.0, 0.80, 1.45, 6.2
        xp, xn, xa = cm.riazi_daubert_pna(MW, SG, n, CH, blend=False)
        d20 = SG - 4.5e-3 * (2.34 - 1.9 * SG)
        assert float(xp) == pytest.approx(2.57 - 2.877 * SG + 0.02876 * CH, abs=1e-14)
        assert float(xn) == pytest.approx(0.52641 - 0.7494 * float(xp) - 0.021811 * MW * (n - 1.475),
                                          abs=1e-14)
        MW = 300.0
        xp, xn, xa = cm.riazi_daubert_pna(MW, SG, n, CH, blend=False)
        Ri = n - d20 / 2
        assert float(xp) == pytest.approx(1.9842 - 0.27722 * Ri - 0.15643 * CH, abs=1e-14)
        assert float(xn) == pytest.approx(0.5977 - 0.761745 * Ri + 0.068048 * CH, abs=1e-14)
        assert float(xp + xn + xa) == pytest.approx(1.0, abs=1e-14)

    def test_the_branch_join_is_smooth_and_reaches_each_branch(self):
        args = (0.85, 1.47, 6.6)
        lo = cm.riazi_daubert_pna(140.0, *args, blend=False)
        hi = cm.riazi_daubert_pna(260.0, *args, blend=False)
        np.testing.assert_allclose(np.asarray(cm.riazi_daubert_pna(140.0, *args)), np.asarray(lo), atol=3e-3)
        np.testing.assert_allclose(np.asarray(cm.riazi_daubert_pna(260.0, *args)), np.asarray(hi), atol=3e-3)
        g = jax.grad(lambda m: cm.riazi_daubert_pna(m, *args)[2])(200.0)
        assert np.isfinite(float(g))

    def test_huang_index_on_pure_hydrocarbons(self):
        """Riazi-Daubert (1987) n20 against measured values (CRC): n-decane
        1.4102, benzene 1.5011, toluene 1.4961."""
        for Tb, SG, n in ((447.3, 0.7342, 1.4102), (353.2, 0.8846, 1.5011), (383.8, 0.8718, 1.4961)):
            assert float(cm.refractive_index_from_huang(cm.huang_index_rd1987(Tb, SG))) == pytest.approx(n, abs=4e-3)
        # (n^2 - 1)/(n^2 + 2) inverts exactly
        I = cm.huang_index(500.0, 0.8)
        n = cm.refractive_index_from_huang(I)
        assert float((n**2 - 1) / (n**2 + 2)) == pytest.approx(float(I), rel=1e-14)

    def test_huang_index_heavy_form_takes_kelvin(self):
        """The heavy form's constants are for Tb in K: n-C36 (Tb 770.2 K, SG
        0.8172) comes out at a paraffin's 1.45-1.46, which Tb in Rankine
        would not give."""
        n = float(cm.refractive_index_from_huang(cm.huang_index_heavy(770.2, 0.8172)))
        assert 1.445 < n < 1.465

    # The six constants (a, b, c, d, e, f) of theta = a exp(b T + c SG + d T SG)
    # T^e SG^f, as tabulated in pychemqt's coding of Riazi-Daubert (1987)
    # Tables X ("I", Tb-SG, T in R) and XI ("CH", Tb-SG, T in R) and of MNL50
    # Table 2.9 ("I", Tb, T in K). Pinned so a typo in the module fails here.
    CITED = {
        "huang_index_rd1987": ((0.022657, 3.9052e-4, 2.468316, -5.70425e-4, 5.7209e-2, -0.719895), 1.8),
        "ch_ratio_rd1987": ((17.22022, 8.24983e-3, 16.9402, -6.93931e-3, -2.72522, -6.79769), 1.8),
        "huang_index_heavy": ((3.2709e-3, 8.4377e-4, 4.59487, -1.0617e-3, 0.03201, -2.34887), 1.0),
    }

    @pytest.mark.parametrize("name", sorted(CITED))
    def test_coded_constants_are_the_cited_ones(self, name):
        (a, b, c, d, e, f), unit = self.CITED[name]
        Tb = np.array([320.0, 450.0, 600.0, 800.0])
        SG = np.array([0.70, 0.80, 0.88, 0.95])
        T = unit * Tb
        expected = a * np.exp(b * T + c * SG + d * T * SG) * T**e * SG**f
        np.testing.assert_allclose(np.asarray(getattr(cm, name)(Tb, SG)), expected, rtol=1e-13)

    def test_density_at_20c(self):
        # n-decane: SG 0.7342 (60/60 F), d20 0.7300 g/cm3
        assert float(cm.density_20c(0.7342)) == pytest.approx(0.7300, abs=3e-4)

    def test_ch_ratio_on_pure_hydrocarbons(self):
        """Riazi-Daubert (1987) C/H weight ratio against the formula: n-decane
        (C10H22) 5.416 within 0.5 %; benzene 11.92 within 7 % (aromatics are
        where the correlation is weakest)."""
        ch = lambda c, h: c * 12.011 / (h * 1.00794)  # noqa: E731
        assert float(cm.ch_ratio_rd1987(447.3, 0.7342)) == pytest.approx(ch(10, 22), rel=5e-3)
        assert float(cm.ch_ratio_rd1987(353.2, 0.8846)) == pytest.approx(ch(6, 6), rel=7e-2)

    def test_goossens(self):
        """pychemqt's ``H_Goossens`` doctest (Goossens 1997, Table 1, n-decane):
        15.45 wt%; C10H22 is 15.58."""
        assert 100 * float(cm.hydrogen_goossens(1.4119, 0.7301, 142.0)) == pytest.approx(15.45, abs=5e-3)

    def test_ndm(self):
        out = cm.ndm(1.50, 0.90, 350.0, 2.0)
        assert float(out["C_P"] + out["C_N"] + out["C_A"]) == pytest.approx(1.0, abs=1e-14)
        v = 2.51 * (1.50 - 1.475) - (0.90 - 0.851)
        w = (0.90 - 0.851) - 1.11 * (1.50 - 1.475)
        assert float(out["C_A"]) == pytest.approx((430 * v + 3660 / 350) / 100, rel=1e-14)
        assert float(out["C_P"]) == pytest.approx(1 - (820 * w - 6 + 10000 / 350) / 100, rel=1e-14)
        # the branches meet at v = 0: C_A = 36.6 / M from either side
        d = 0.88
        n0 = 1.475 + (d - 0.851) / 2.51
        for eps in (1e-9, -1e-9):
            assert float(cm.ndm(n0 + eps, d, 300.0)["C_A"]) == pytest.approx(36.6 / 300, abs=1e-6)


# =============================================================================
# Every component
# =============================================================================


@pytest.mark.parametrize("which", ["heavy", "light"])
@pytest.mark.parametrize("method", ["riazi_daubert", "ndm"])
def test_types_are_a_composition_per_cut(chars, which, method):
    char = chars[which]
    comp = _quiet(cm.estimate_composition, char, pna_method=method)
    x = np.asarray(comp.hc_type)
    k = comp.n_light
    assert x.shape == (len(char.names), 4)
    np.testing.assert_allclose(x.sum(axis=1), 1.0, atol=1e-13)
    assert np.all(x >= 0.0) and np.all(x <= 1.0)
    assert np.all(x[:, 3] == 0.0), "straight run: no olefins"
    np.testing.assert_array_equal(x[:k], np.tile([1.0, 0, 0, 0], (k, 1)))
    # light ends' hydrogen is their formula's
    h = np.asarray(comp.hydrogen)
    assert h[list(char.light_names).index("propane")] == pytest.approx(8 * 1.00794 / 44.097, rel=1e-12)
    # cuts: hydrogen falls with boiling point (the residue lump's MW is set,
    # not correlated, so it is left out), and aromatics rise across the crude
    end = len(h) - 1 if char.residue_lump else len(h)
    assert np.all(np.diff(h[k:end]) < 0.0)
    assert x[-2, 2] > x[k, 2]


def test_classes_partition_the_heteroatoms(chars):
    comp = chars["heavy"].composition
    np.testing.assert_allclose(np.asarray(comp.sulfur_split).sum(1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(comp.nitrogen_split).sum(1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(comp.sulfur_classes).sum(1), np.asarray(comp.sulfur), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(comp.nitrogen_classes).sum(1), np.asarray(comp.nitrogen), rtol=1e-14)
    s = np.asarray(comp.sulfur_split)
    k = comp.n_light
    # the default shape: no DBTs in naphtha, hindered DBTs only from diesel up
    naphtha = np.asarray(chars["heavy"].Tb) <= 423.15  # below 150 C
    assert np.all(s[k:][naphtha, 2:] == 0.0)
    assert s[-1, 4] > 0.2


def test_split_overrides():
    char = _quiet(dr.characterize, light_assay())
    n = char.n_cuts
    per_cut = np.tile([0.1, 0.2, 0.3, 0.2, 0.2], (n, 1))
    comp = _quiet(cm.estimate_composition, char, sulfur_split=per_cut, nitrogen_split=np.full(n, 0.4))
    np.testing.assert_allclose(np.asarray(comp.sulfur_split)[comp.n_light:], per_cut)
    np.testing.assert_allclose(np.asarray(comp.nitrogen_split)[comp.n_light:, 0], 0.4)
    table = ((400.0, 700.0), ((1, 0, 0, 0, 0), (0, 0, 0, 0, 1)))
    comp = _quiet(cm.estimate_composition, char, sulfur_split=table)
    s = np.asarray(comp.sulfur_split)[comp.n_light:]
    np.testing.assert_allclose(s.sum(1), 1.0, atol=1e-14)
    assert s[0, 0] == 1.0 and s[-1, 4] == 1.0


def test_measured_hydrogen_moves_the_types():
    """C/H comes from the hydrogen, so a cut measured richer in hydrogen is
    estimated less aromatic -- hydrogen and type stay consistent."""
    char = _quiet(dr.characterize, light_assay())
    base = _quiet(cm.estimate_composition, char)
    k, i = base.n_light, 12
    h = np.full(char.n_cuts, np.nan)
    h[i] = 100 * float(base.hydrogen[k + i]) + 0.5
    meas = _quiet(cm.estimate_composition, char, hydrogen_wt=h)
    assert float(meas.hydrogen[k + i]) == pytest.approx(h[i] / 100, rel=1e-14)
    assert float(meas.aromatics[k + i]) < float(base.aromatics[k + i])
    others = [j for j in range(char.n_cuts) if j != i]
    np.testing.assert_array_equal(np.asarray(meas.hc_type)[k:][others], np.asarray(base.hc_type)[k:][others])


# =============================================================================
# Balances through product_properties
# =============================================================================


def _products(char, rng):
    """A crude feed split into four products by a fixed per-component split."""
    n = len(char.names)
    raw = rng.random((n, 4))
    split = raw / raw.sum(1, keepdims=True)
    feed_moles = 1000.0 * np.asarray(char.mole_fraction)
    feed = {f"F_{c}": feed_moles[i] for i, c in enumerate(char.names)}
    products = {f"p{j}": {f"F_{c}": feed_moles[i] * split[i, j] for i, c in enumerate(char.names)}
                for j in range(4)}
    return feed, products


@pytest.mark.parametrize("which", ["heavy", "light"])
def test_balances_close_through_product_properties(chars, which):
    char = chars[which]
    comp = char.composition
    thermo = ColumnThermo.from_characterization(char)
    feed, products = _products(char, np.random.default_rng(3))
    props = product_properties(products, thermo, feed, composition=comp)
    fc = comp.of_stream(feed)
    tol = 1e-12

    def total(attr, basis):
        return sum(float(getattr(p, basis)) * np.asarray(getattr(p.composition, attr))
                   for p in props.values())

    for attr in ("hydrogen_wt", "sulfur_wt", "nitrogen_wppm", "sulfur_classes_wt",
                 "nitrogen_classes_wppm"):
        np.testing.assert_allclose(total(attr, "mass"), float(fc.mass) * np.asarray(getattr(fc, attr)),
                                   rtol=tol, atol=0)
    np.testing.assert_allclose(total("hc_type", "volume"), float(fc.volume) * np.asarray(fc.hc_type),
                               rtol=tol, atol=1e-15)
    # the products' own rates are the ones the composition was weighted on
    for p in props.values():
        assert float(p.composition.mass) == pytest.approx(float(p.mass), rel=1e-14)
        assert float(p.composition.volume) == pytest.approx(float(p.volume), rel=1e-14)
        np.testing.assert_allclose(float(np.sum(p.composition.sulfur_classes_wt)),
                                   float(p.composition.sulfur_wt), rtol=1e-13)
    # the whole crude's sulfur and nitrogen are the assay's bulk numbers
    a = heavy_assay() if which == "heavy" else light_assay()
    assert float(fc.sulfur_wt) == pytest.approx(a.sulfur_wt, rel=1e-12)
    assert float(fc.nitrogen_wppm) == pytest.approx(a.nitrogen_wppm, rel=1e-12)


def test_product_properties_without_composition_is_unchanged(chars):
    char = chars["light"]
    thermo = ColumnThermo.from_characterization(char)
    feed, products = _products(char, np.random.default_rng(0))
    props = product_properties(products, thermo, feed)
    assert all(p.composition is None for p in props.values())
    with pytest.raises(ValueError, match="same components"):
        product_properties(products, thermo, feed, composition=chars["heavy"].composition)


# =============================================================================
# Gradients
# =============================================================================


def _product_composition(assay, cut_points, data=None, weights=None):
    char = dr.characterize(assay, cut_points=cut_points)
    comp = cm.estimate_composition(char, data, warn=False)
    return comp.of_flows(jnp.asarray(char.mole_fraction) * weights)


def _kero_weights(char):
    """A kerosene-diesel product: every cut boiling 180-320 C, by index."""
    k = len(char.light_names)
    Tb = np.asarray(char.Tb)
    w = np.zeros(len(char.names))
    w[k:] = ((Tb > 453.15) & (Tb < 593.15)).astype(float)
    return jnp.asarray(w)


def _fd(f, x, h):
    return (np.asarray(f(x + h)) - np.asarray(f(x - h))) / (2 * h)


@pytest.mark.parametrize("which", ["heavy", "light"])
def test_gradients_wrt_tbp_point_and_gravity(which):
    assay = heavy_assay() if which == "heavy" else light_assay()
    cuts = dr.default_cut_points(assay)
    w = _kero_weights(_quiet(dr.characterize, assay))

    def out(c):
        return jnp.concatenate([c.hc_type[:3], c.hydrogen_wt[None]])

    idx = 4 if which == "heavy" else 4

    def f_T(T):
        return out(_product_composition(assay.with_tbp_point(idx, T), cuts, weights=w))

    def f_sg(sg):
        return out(_product_composition(dataclasses_replace(assay, sg=sg), cuts, weights=w))

    T0 = float(np.asarray(assay.tbp_T)[idx])
    g = np.asarray(jax.jacfwd(f_T)(T0))
    fd = _fd(f_T, T0, 1e-3)
    assert np.all(np.abs(g) > 1e-8), "every output moves with the TBP point"
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-9)

    sg0 = float(assay.sg)
    g = np.asarray(jax.jacfwd(f_sg)(sg0))
    fd = _fd(f_sg, sg0, 1e-6)
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-8)


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


# =============================================================================
# Measured PIONA / SARA
# =============================================================================


class TestMeasured:
    def test_curve_overrides_inside_its_range_and_keeps_the_estimate_outside(self):
        char = _quiet(dr.characterize, light_assay())
        base = _quiet(cm.estimate_composition, char)
        k = base.n_light
        T = (330.0, 450.0)
        meas = cm.CutData({"paraffins": (0.50, 0.40), "naphthenes": (0.35, 0.40),
                           "aromatics": (0.15, 0.20)}, T=T)
        comp = _quiet(cm.estimate_composition, char, hc_types=meas)
        Tb = np.asarray(char.Tb)
        inside = (Tb >= T[0]) & (Tb <= T[1])
        assert inside.sum() >= 3 and (~inside).sum() >= 3
        x = np.asarray(comp.hc_type)[k:]
        np.testing.assert_allclose(x[inside, 2], np.interp(Tb[inside], T, (0.15, 0.20)), rtol=1e-13)
        np.testing.assert_array_equal(x[~inside], np.asarray(base.hc_type)[k:][~inside])

    def test_a_partial_analysis_fills_the_rest_in_the_estimates_proportion(self):
        char = _quiet(dr.characterize, light_assay())
        base = _quiet(cm.estimate_composition, char)
        k, i = base.n_light, 6
        arom = np.full(char.n_cuts, np.nan)
        arom[i] = 0.30
        comp = _quiet(cm.estimate_composition, char, hc_types={"aromatics": arom})
        x, e = np.asarray(comp.hc_type)[k + i], np.asarray(base.hc_type)[k + i]
        assert x[2] == pytest.approx(0.30, rel=1e-14)
        assert x.sum() == pytest.approx(1.0, abs=1e-14)
        assert x[0] / x[1] == pytest.approx(e[0] / e[1], rel=1e-12)

    def test_sara_saturates_split_by_the_estimate(self):
        char = _quiet(dr.characterize, heavy_assay())
        base = _quiet(cm.estimate_composition, char)
        k, i = base.n_light, char.n_cuts - 3
        def per(v):
            a = np.full(char.n_cuts, np.nan)
            a[i] = v
            return a
        comp = _quiet(cm.estimate_composition, char, hc_types={
            "saturates": per(0.40), "aromatics": per(0.40), "resins": per(0.15),
            "asphaltenes": per(0.05)})
        x, e = np.asarray(comp.hc_type)[k + i], np.asarray(base.hc_type)[k + i]
        assert x[0] + x[1] == pytest.approx(0.40, rel=1e-13)
        assert x[2] == pytest.approx(0.60, rel=1e-13)
        assert x[0] / x[1] == pytest.approx(e[0] / e[1], rel=1e-12)

    def test_override_is_differentiable(self):
        assay = light_assay()
        cuts = dr.default_cut_points(assay)
        char = _quiet(dr.characterize, assay)
        w = _kero_weights(char)
        T = jnp.asarray([440.0, 520.0, 600.0])

        def product_aromatics(a_mid):
            vals = {"aromatics": jnp.stack([0.18, a_mid, 0.26]),
                    "naphthenes": jnp.asarray([0.30, 0.30, 0.28])}
            data = cm.CompositionData(hc_types=cm.CutData(vals, T=T))
            return _product_composition(assay, cuts, data, w).aromatics

        g = float(jax.grad(product_aromatics)(0.22))
        fd = float(_fd(product_aromatics, 0.22, 1e-4))
        assert g > 0.1
        assert g == pytest.approx(fd, rel=1e-7)

        # and through a TBP point with the measured curve in place
        def via_T(Tp):
            data = cm.CompositionData(hc_types=cm.CutData({"aromatics": jnp.asarray([0.18, 0.22, 0.26])}, T=T))
            return _product_composition(assay.with_tbp_point(4, Tp), cuts, data, w).aromatics

        T0 = float(np.asarray(assay.tbp_T)[4])
        assert float(jax.grad(via_T)(T0)) == pytest.approx(float(_fd(via_T, T0, 1e-3)), rel=1e-5)

    def test_measured_refractive_index_replaces_the_estimate(self):
        char = _quiet(dr.characterize, light_assay())
        comp = _quiet(cm.estimate_composition, char,
                      refractive_index=cm.CutData([1.40, 1.50], T=[300.0, 700.0], extrapolate="flat"))
        Tb = np.asarray(char.Tb)
        np.testing.assert_allclose(np.asarray(comp.refractive_index)[comp.n_light:],
                                   np.interp(Tb, [300.0, 700.0], [1.40, 1.50]), rtol=1e-14)

    def test_bad_inputs(self):
        with pytest.raises(ValueError, match="unknown measured types"):
            cm.CutData({"isoparaffins": [0.1]})
        with pytest.raises(ValueError, match="pna_method"):
            cm.CompositionData(pna_method="bergman")
        char = _quiet(dr.characterize, light_assay())
        with pytest.raises(ValueError, match="one entry per pseudo-component"):
            cm.estimate_composition(char, hc_types={"aromatics": [0.1, 0.2]}, warn=False)


# =============================================================================
# Warnings, tracing, the blend pool
# =============================================================================


def test_heavy_end_warns_and_names_the_lump():
    with pytest.warns(cm.CompositionRangeWarning, match="residue lump"):
        dr.characterize(heavy_assay(), composition=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", cm.CompositionRangeWarning)
        dr.characterize(heavy_assay(), composition=cm.CompositionData(warn=False))


def test_traceable_and_a_pytree(chars):
    char = chars["light"]
    comp = jax.jit(lambda c: cm.estimate_composition(c, warn=False))(char)
    np.testing.assert_allclose(np.asarray(comp.hc_type), np.asarray(char.composition.hc_type), rtol=1e-13)
    again = jax.jit(lambda c: c)(char)
    assert again.composition.names == char.composition.names


def test_composition_reaches_the_blend_pool(chars):
    char = chars["light"]
    comp = char.composition
    bc = dr.BlendCharacterization.from_characterization(char)
    for t in cm.HC_TYPES:
        assert f"{t}_vol" in bc.qualities
    feed, products = _products(char, np.random.default_rng(1))
    stream = products["p2"]
    blend = dr.BlendComponent.from_stream("p2", stream, bc)
    mine = comp.of_stream(stream)
    # both average on standard liquid volume (60 F vs 15 C water cancels)
    assert float(blend.properties["aromatics_vol"]) == pytest.approx(100 * float(mine.aromatics), rel=1e-10)
    assert float(blend.properties["naphthenes_vol"]) == pytest.approx(100 * float(mine.naphthenes), rel=1e-10)
    off = dr.BlendCharacterization.from_characterization(char, composition=False)
    assert "aromatics_vol" not in off.qualities
