"""Crude assays cut into pseudo-components."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

import difflow  # noqa: F401  (enables x64)
from difflow.thermo import CubicThermo, IdealThermo
from difflow_refinery import Assay, Characterization, characterize, default_cut_points
from difflow_refinery.assay import _Pchip, fit_antoine

# A medium crude (about 33 API): cumulative volume percent against TBP, K.
PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
TBP = [t + 273.15 for t in T_C]
LIGHT = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}


@pytest.fixture(scope="module")
def crude():
    return characterize(Assay(PCT, TBP, sg=0.86, light_ends=LIGHT))


# ---------------------------------------------------------------- the input


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(tbp_percent=[5, 50, 100]), "from 0 to 100"),
        (dict(tbp_percent=[0, 50, 95]), "from 0 to 100"),
        (dict(tbp_percent=[0, 60, 50, 100], tbp_T=[300, 400, 500, 600]), "increasing"),
        (dict(tbp_T=[300, 500, 400]), "increasing"),
        (dict(sg=None), "exactly one"),
        (dict(api=30.0), "exactly one"),
        (dict(light_ends={"unobtainium": 1.0}), "light end"),
        (dict(basis="weight"), "basis"),
    ],
)
def test_assay_refuses_bad_input(kwargs, match):
    args = dict(tbp_percent=[0, 50, 100], tbp_T=[300.0, 500.0, 800.0], sg=0.85)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Assay(**args)


def test_api_and_sg_are_the_same_input():
    a = characterize(Assay(PCT, TBP, sg=0.86))
    b = characterize(Assay(PCT, TBP, api=141.5 / 0.86 - 131.5))
    np.testing.assert_allclose(a.SG, b.SG, rtol=1e-12)


def test_cut_points_outside_the_curve_are_refused():
    assay = Assay(PCT, TBP, sg=0.86, light_ends=LIGHT)
    with pytest.raises(ValueError, match="strictly between"):
        characterize(assay, cut_points=[250.0, 500.0])
    with pytest.raises(ValueError, match="strictly between"):
        characterize(assay, cut_points=[500.0, 1200.0])
    with pytest.raises(ValueError, match="increasing"):
        characterize(assay, cut_points=[600.0, 500.0])


# ---------------------------------------------------------- the interpolant


def test_pchip_matches_scipy_and_integrates_exactly():
    x = np.asarray(TBP)
    y = np.asarray(PCT) / 100.0
    ours = _Pchip.fit(x, y)
    ref = PchipInterpolator(x, y)
    t = np.linspace(x[0], x[-1], 997)
    np.testing.assert_allclose(ours(jnp.asarray(t)), ref(t), atol=1e-12)
    np.testing.assert_allclose(ours.integral(jnp.asarray(t)), ref.antiderivative()(t), atol=1e-9)
    targets = jnp.linspace(0.01, 0.99, 31)
    np.testing.assert_allclose(ours(ours.invert(targets)), targets, atol=1e-12)


# ---------------------------------------------------------- the cut crude


def test_mass_balance(crude):
    for frac in (crude.volume_fraction, crude.mass_fraction, crude.mole_fraction):
        assert float(jnp.sum(frac)) == pytest.approx(1.0, abs=1e-12)
        assert bool(jnp.all(frac > 0))
    # light ends are carried through as given
    np.testing.assert_allclose(crude.volume_fraction[:3], [0.005, 0.010, 0.015], rtol=1e-12)
    # the cuts recombine to the bulk gravity they were fitted to
    assert float(crude.bulk_sg) == pytest.approx(0.86, rel=1e-12)


def test_cuts_are_ordered_and_inside_their_edges(crude):
    edges = np.asarray(crude.cut_edges)
    Tb = np.asarray(crude.Tb)
    assert np.all(np.diff(edges) > 0)
    assert np.all((Tb > edges[:-1]) & (Tb < edges[1:]))
    assert edges[-1] == pytest.approx(TBP[-1])
    # the pseudo-components start where the curve reaches the light ends' 3 %
    assert float(PchipInterpolator(TBP, np.asarray(PCT) / 100)(edges[0])) == pytest.approx(0.03)
    for prop in (crude.SG, crude.MW, crude.Tc):
        assert np.all(np.diff(np.asarray(prop)) > 0)
    assert np.all(np.diff(np.asarray(crude.Pc)) < 0)


def test_cut_fractions_are_the_curve_differences(crude):
    x = PchipInterpolator(TBP, np.asarray(PCT) / 100)(np.asarray(crude.cut_edges))
    np.testing.assert_allclose(crude.volume_fraction[3:], np.diff(x), atol=1e-12)


def test_a_straight_tbp_curve_puts_tb_at_mid_cut():
    crude = characterize(Assay([0, 100], [300.0, 900.0], sg=0.85), cut_points=[400.0, 550.0, 700.0])
    np.testing.assert_allclose(crude.Tb, [350.0, 475.0, 625.0, 800.0], rtol=1e-12)
    np.testing.assert_allclose(crude.volume_fraction, [1 / 6, 0.25, 0.25, 1 / 3], rtol=1e-12)


def test_constant_watson_k_without_a_gravity_curve(crude):
    np.testing.assert_allclose(crude.Kw, crude.Kw[0], rtol=1e-12)


def test_gravity_curve():
    flat = characterize(Assay(PCT, TBP, sg_curve=([0, 100], [0.88, 0.88])))
    np.testing.assert_allclose(flat.SG, 0.88, rtol=1e-12)
    # a linear SG(x) averages to its value at the cut's mid-fraction
    lin = characterize(Assay([0, 100], [300.0, 900.0], sg_curve=([0, 100], [0.70, 1.00])),
                       cut_points=[600.0])
    np.testing.assert_allclose(lin.SG, [0.775, 0.925], rtol=1e-12)


def test_mass_basis_recombines_too():
    crude = characterize(Assay(PCT, TBP, basis="mass", sg=0.86, light_ends=LIGHT))
    np.testing.assert_allclose(crude.mass_fraction[:3], [0.005, 0.010, 0.015], rtol=1e-12)
    assert float(crude.bulk_sg) == pytest.approx(0.86, rel=1e-12)


def test_default_cut_points_are_round_celsius():
    cuts = np.asarray(default_cut_points(Assay(PCT, TBP, sg=0.86, light_ends=LIGHT))) - 273.15
    np.testing.assert_allclose(cuts, np.round(cuts), atol=1e-9)
    widths = np.diff(cuts)
    assert set(np.round(widths[cuts[1:] <= 400])) == {20.0}
    assert set(np.round(widths[(cuts[:-1] >= 400) & (cuts[1:] <= 600)])) == {40.0}
    assert set(np.round(widths[cuts[:-1] >= 600])) <= {100.0}


def test_more_cuts_converge():
    """Refining the cuts leaves the whole-crude properties where they were."""
    assay = Assay(PCT, TBP, sg=0.86, light_ends=LIGHT)
    coarse = characterize(assay, cut_points=np.linspace(400, 1000, 7))
    fine = characterize(assay, cut_points=np.linspace(400, 1000, 61))
    assert float(fine.bulk_MW) == pytest.approx(float(coarse.bulk_MW), rel=0.03)


# ----------------------------------------------------------- differentiable


CUTS = tuple(np.linspace(380.0, 1000.0, 12))


def _middle_distillate_mass(tbp, sg, propane):
    crude = characterize(Assay(PCT, tbp, sg=sg, light_ends={"propane": propane}), cut_points=CUTS)
    return jnp.sum(crude.mass_fraction[4:8]) + jnp.sum(crude.Tc[4:8]) / 1e4


def test_gradients_wrt_the_assay_match_finite_differences():
    from jax.test_util import check_grads

    check_grads(_middle_distillate_mass, (jnp.asarray(TBP), 0.86, 1.0),
                order=1, modes=["fwd", "rev"], rtol=1e-5, atol=1e-8)


def test_gradient_wrt_the_percentages():
    def f(pct_interior):
        pct = jnp.concatenate([jnp.zeros(1), pct_interior, jnp.full(1, 100.0)])
        crude = characterize(Assay(pct, TBP, sg=0.86), cut_points=CUTS)
        return jnp.sum(crude.volume_fraction[:5] * crude.Tb[:5]) / 100

    from jax.test_util import check_grads

    check_grads(f, (jnp.asarray(PCT[1:-1], dtype=float),), order=1, modes=["rev"], rtol=1e-5)


def test_traced_assay_needs_explicit_cut_points():
    def f(tbp):
        return jnp.sum(characterize(Assay(PCT, tbp, sg=0.86)).Tb)

    with pytest.raises(ValueError, match="cut_points"):
        jax.grad(f)(jnp.asarray(TBP))


def test_characterization_is_a_pytree_and_jits():
    f = jax.jit(lambda tbp: characterize(Assay(PCT, tbp, sg=0.86), cut_points=CUTS))
    crude = f(jnp.asarray(TBP))
    assert isinstance(crude, Characterization)
    leaves, tree = jax.tree_util.tree_flatten(crude)
    assert jax.tree_util.tree_unflatten(tree, leaves).pseudo_names == crude.pseudo_names


# ------------------------------------------------------- export to difflow


def test_vapor_pressure_boils_each_cut_at_its_tb(crude):
    np.testing.assert_allclose(crude.vapor_pressure(crude.Tb), 101325.0, rtol=1e-10)


def test_antoine_fit_follows_lee_kesler(crude):
    data = crude.species_data()
    for i, name in enumerate(crude.pseudo_names):
        s = data[name]
        A, B, C = s.antoine_coeffs
        T = np.linspace(s.T_antoine_min, s.T_antoine_max, 50)
        lk = np.asarray(crude.vapor_pressure(T[:, None]))[:, i]
        assert np.max(np.abs(10 ** (A - B / (T + C)) / lk - 1)) < 0.05, name
        assert s.T_antoine_min < float(crude.Tb[i]) <= s.T_antoine_max + 1e-9


def test_fit_antoine_window_is_reported():
    (A, B, C), (lo, hi) = fit_antoine(600.0, 25e5, 0.4, window=(300.0, 500.0))
    assert 300.0 <= lo < hi <= 500.0


def test_species_data_cp_choice(crude):
    liq = crude.species_data("liquid")
    ig = crude.species_data("ideal_gas")
    name = crude.pseudo_names[5]
    np.testing.assert_allclose(liq[name].Cp_coeffs, crude.cp_liquid_coeffs[5])
    np.testing.assert_allclose(ig[name].Cp_coeffs, crude.cp_ig_coeffs[5])
    assert liq["propane"] == ig["propane"]
    with pytest.raises(ValueError, match="cp"):
        crude.species_data("solid")


def test_thermo_packages(crude):
    ideal = crude.thermo()
    assert isinstance(ideal, IdealThermo)
    assert set(ideal.species) == set(crude.names)
    pr = crude.thermo("pr")
    srk = crude.thermo("srk")
    assert isinstance(pr, CubicThermo) and isinstance(srk, CubicThermo)
    with pytest.raises(ValueError, match="eos"):
        crude.thermo("vdw")


def test_feed_stream_carries_the_mass(crude):
    flows = crude.flows(100.0)  # kg/s
    MW = np.asarray(crude.component_MW)
    total = sum(float(flows[n]) * MW[i] for i, n in enumerate(crude.names)) / 1000.0
    assert total == pytest.approx(100.0, rel=1e-12)
    stream = crude.stream(100.0, T=600.0, P=2e5)
    assert {k[2:] for k in stream if k.startswith("F_")} == set(crude.names)


def test_table(crude):
    text = crude.table()
    assert all(n in text for n in crude.names)


def test_heat_of_vaporization_passes_through_its_value_at_tb(crude):
    thermo = crude.thermo()
    for i, name in enumerate(crude.pseudo_names):
        assert float(thermo.Hvap(name, crude.Tb[i])) == pytest.approx(float(crude.hvap_nb[i]), rel=1e-10)
