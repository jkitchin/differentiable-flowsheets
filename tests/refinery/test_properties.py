"""Product property estimates for the blend pool (#330).

What is pinned:

* Won's (1986) n-paraffin melting point against the tabulated melting points
  of n-C10 to n-C24, and its heat of fusion against the CRC heats of fusion
  -- the one place a published number is reproduced (the reference values
  are the ``chemicals`` 1.5.2 tables, "OpenNotebook Melting Points" and
  "CRC Handbook Heat of Fusion", copied here);
* the ASTM D341 (Walther) line is exact on its two points;
* the straight-run octane estimate returns a pure compound's own octane for
  a single-compound stream, and reduces to the reformer's species data;
* every other estimate (flash, smoke, Abbott viscosity, octane of a cut) is
  UNVERIFIED against its source: what is tested is the direction each must
  move in, plausible values on typical products, and differentiability;
* ``BlendComponent.from_stream`` estimates what it is not given, an override
  wins (and is not estimated), and ``estimate=False`` is the old behaviour.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, BlendCharacterization, BlendComponent, BlendPool
from difflow_refinery import properties as pr
from difflow_refinery.composition import CompositionRangeWarning
from difflow_refinery.reforming import species as sp

jax.config.update("jax_enable_x64", True)

C = 273.15


# -----------------------------------------------------------------------------
# A crude, characterized with composition, and straight-run cuts of it.
# -----------------------------------------------------------------------------

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


CUTS = {"LSR": (-50, 85), "HSR": (85, 180), "kero": (180, 240), "diesel": (240, 350),
        "residue": (350, 2000)}


def cut_moles(bc, name):
    char, b = bc
    Tb = np.asarray(b.Tb) - C
    lo, hi = CUTS[name]
    m = ((Tb >= lo) & (Tb < hi)).astype(float)
    m[[i for i, n in enumerate(b.names) if n in ("ethane", "propane")]] = 0.0
    return jnp.asarray(np.asarray(char.mole_fraction) * m)


def cut_stream(bc, name):
    _, b = bc
    s = {f"F_{n}": f for n, f in zip(b.names, 100.0 * cut_moles(bc, name))}
    s.update(T=298.15, P=101325.0)
    return s


# -----------------------------------------------------------------------------
# Freeze point: Won (1986) against tabulated data, and the solubility model.
# -----------------------------------------------------------------------------

#: n-alkane carbon number -> (melting point K, heat of fusion J/mol), from the
#: ``chemicals`` 1.5.2 tables (OpenNotebook melting points; CRC heats of fusion).
N_ALKANES = {10: (243.225, 28720), 11: (247.15, 22200), 12: (263.55, 36800),
             13: (268.15, 28500), 14: (279.05, 45070), 15: (283.1, 34600),
             16: (291.15, 53360), 17: (295.15, 40160), 18: (301.15, 61700),
             19: (305.7, 45800), 20: (309.9, 69900), 24: (325.65, 54400)}


def _mw(n):
    return 12.0107 * n + 1.00794 * (2 * n + 2)


class TestWon:
    @pytest.mark.parametrize("n", sorted(N_ALKANES))
    def test_melting_point_against_tabulated_values(self, n):
        Tm = float(pr.won_melting_point(_mw(n)))
        tol = 7.0 if n == 10 else 3.5
        assert abs(Tm - N_ALKANES[n][0]) < tol, (n, Tm)

    @pytest.mark.parametrize("n", [11, 13, 15, 17, 19])
    def test_heat_of_fusion_close_for_odd_alkanes(self, n):
        dH = float(pr.won_heat_of_fusion(_mw(n)))
        assert abs(dH / N_ALKANES[n][1] - 1.0) < 0.10, (n, dH)

    @pytest.mark.parametrize("n", [10, 12, 14, 16, 18, 20])
    def test_heat_of_fusion_low_for_even_alkanes_as_documented(self, n):
        # Recorded, not hidden: Won's single curve sits 25-32 % under the
        # even n-alkanes' heats of fusion (see won_heat_of_fusion).
        r = float(pr.won_heat_of_fusion(_mw(n))) / N_ALKANES[n][1]
        assert 0.65 < r < 0.80, (n, r)

    def test_saturation_is_the_melting_point_for_the_pure_solid(self):
        MW = _mw(16)
        assert float(pr.saturation_temperature(1.0, MW)) == pytest.approx(
            float(pr.won_melting_point(MW)), rel=1e-12)
        T = [float(pr.saturation_temperature(x, MW)) for x in (0.001, 0.01, 0.1)]
        assert T[0] < T[1] < T[2]


class TestFreezePoint:
    def test_heavier_and_richer_n_paraffins_freeze_higher(self):
        MW = jnp.asarray([170.0, 200.0, 230.0])
        base = float(pr.freeze_point(jnp.asarray([0.02, 0.02, 0.005]), MW))
        heavier = float(pr.freeze_point(jnp.asarray([0.02, 0.02, 0.01]), MW))
        assert heavier > base
        assert float(pr.freeze_point(jnp.asarray([0.04, 0.04, 0.01]), MW)) > heavier

    def test_smooth_max_is_the_highest_saturation_temperature(self):
        x, MW = jnp.asarray([0.02, 0.01]), jnp.asarray([180.0, 230.0])
        Ts = pr.saturation_temperature(x, MW)
        assert float(pr.freeze_point(x, MW)) == pytest.approx(float(jnp.max(Ts)), abs=0.5)

    def test_light_naphtha_gets_the_floor_not_minus_infinity(self, bc):
        p = pr.estimate_properties(bc[1], cut_moles(bc, "LSR"), ["freeze_C"])
        assert float(p["freeze_C"]) == pytest.approx(pr.FREEZE_FLOOR_K - C, abs=1.0)

    def test_kerosene_is_in_the_jet_range(self, bc):
        p = pr.estimate_properties(bc[1], cut_moles(bc, "kero"), ["freeze_C"])
        assert -75.0 < float(p["freeze_C"]) < -35.0


# -----------------------------------------------------------------------------
# Flash point, smoke point, viscosity (unverified correlations).
# -----------------------------------------------------------------------------

class TestFlashPoint:
    def test_rises_with_the_ten_percent_point(self):
        T = jnp.linspace(320.0, 560.0, 25)
        assert bool(jnp.all(jnp.diff(pr.flash_point(T)) > 0))

    def test_typical_products(self):
        def f(t10_C):
            return float(pr.flash_point(t10_C + C)) - C
        assert -60.0 < f(50.0) < -30.0          # gasoline
        assert 45.0 < f(185.0) < 75.0           # kerosene
        assert 80.0 < f(245.0) < 115.0          # diesel

    def test_cut_ordering(self, bc):
        fl = [float(pr.estimate_properties(bc[1], cut_moles(bc, c), ["flash_C"])["flash_C"])
              for c in ("HSR", "kero", "diesel", "residue")]
        assert fl == sorted(fl)


class TestSmokePoint:
    def test_falls_with_gravity_at_fixed_boiling_point(self):
        sg = jnp.linspace(0.76, 0.86, 11)
        assert bool(jnp.all(jnp.diff(pr.smoke_point(480.0, sg)) < 0))

    def test_kerosene_magnitudes(self):
        assert 18.0 < float(pr.smoke_point(473.0, 0.80)) < 30.0
        assert 10.0 < float(pr.smoke_point(480.0, 0.84)) < 20.0


class TestViscosity:
    def test_typical_products_at_100F(self):
        nu_k, _ = pr.abbott_viscosity(473.0, 0.80)
        nu_d, _ = pr.abbott_viscosity(560.0, 0.85)
        assert 1.0 < float(nu_k) < 2.0
        assert 2.5 < float(nu_d) < 5.0

    def test_heavier_is_more_viscous_and_hot_is_thinner(self):
        nu100 = [float(pr.abbott_viscosity(tb, 0.85)[0]) for tb in (500.0, 550.0, 600.0)]
        assert nu100 == sorted(nu100)
        a, b = pr.abbott_viscosity(560.0, 0.85)
        assert float(b) < float(a)

    def test_walther_line_is_exact_on_its_points(self):
        nu = pr.walther_viscosity(30.0, pr.T_100F, 5.0, pr.T_210F,
                                  jnp.asarray([pr.T_100F, pr.T_210F]))
        np.testing.assert_allclose(np.asarray(nu), [30.0, 5.0], rtol=1e-12)
        T = jnp.linspace(300.0, 400.0, 11)
        assert bool(jnp.all(jnp.diff(pr.walther_viscosity(30.0, pr.T_100F, 5.0, pr.T_210F, T)) < 0))

    def test_pole_warns(self):
        with pytest.warns(pr.PropertyRangeWarning):
            pr.abbott_viscosity(1100.0, 1.03)

    def test_residue_at_50C_is_hundreds_of_cSt(self, bc):
        p = pr.estimate_properties(bc[1], cut_moles(bc, "residue"), ["viscosity_cSt"])
        assert 50.0 < float(p["viscosity_cSt"]) < 2000.0


# -----------------------------------------------------------------------------
# Straight-run octane.
# -----------------------------------------------------------------------------

class TestOctane:
    def test_single_compound_returns_its_own_octane(self):
        names = ["isopentane", "n_pentane"]
        ron, mon = pr.straight_run_octane(jnp.asarray([1.0, 0.0]), jnp.asarray([301.0, 309.2]),
                                          {"paraffins_vol": jnp.asarray([100.0, 100.0]),
                                           "naphthenes_vol": jnp.zeros(2),
                                           "aromatics_vol": jnp.zeros(2)}, names=names)
        assert float(ron) == pytest.approx(sp.SPECIES["iC5"].RON, abs=1e-9)
        assert float(mon) == pytest.approx(sp.SPECIES["iC5"].MON, abs=1e-9)

    def test_group_octane_hits_the_model_compounds(self):
        for key in ("nP7", "iP8", "N7", "A8"):
            s = sp.SPECIES[key]
            ron, _ = pr.group_octane(s.kind, s.Tb)
            assert float(ron) == pytest.approx(s.RON, abs=1e-9)

    def _one_cut(self, P, N, A, Tb=380.0, share=0.5):
        return float(pr.straight_run_octane(
            jnp.ones(1), jnp.asarray([Tb]),
            {"paraffins_vol": jnp.asarray([P]), "naphthenes_vol": jnp.asarray([N]),
             "aromatics_vol": jnp.asarray([A])}, n_paraffin_share=share)[0])

    def test_directions(self):
        base = self._one_cut(60.0, 30.0, 10.0)
        assert self._one_cut(50.0, 30.0, 20.0) > base       # aromatics up
        assert self._one_cut(60.0, 30.0, 10.0, share=0.7) < base   # more n-paraffins
        assert self._one_cut(60.0, 30.0, 10.0, Tb=420.0) < base    # heavier paraffins

    def test_straight_run_cuts_are_in_their_measured_ranges(self, bc):
        lsr = pr.estimate_properties(bc[1], cut_moles(bc, "LSR"), ["RON", "MON"])
        hsr = pr.estimate_properties(bc[1], cut_moles(bc, "HSR"), ["RON"])
        assert 60.0 < float(lsr["RON"]) < 80.0
        assert 30.0 < float(hsr["RON"]) < 60.0
        assert float(lsr["MON"]) <= float(lsr["RON"]) + 2.0


# -----------------------------------------------------------------------------
# from_stream wiring, the pool, gradients.
# -----------------------------------------------------------------------------

class TestFromStream:
    def test_estimates_what_it_is_not_given(self, bc):
        c = BlendComponent.from_stream("jet", cut_stream(bc, "kero"), bc[1])
        for k in pr.ESTIMATED_PROPERTIES:
            assert k in c.properties and np.isfinite(float(c.properties[k])), k

    def test_override_wins_and_estimate_false_is_the_old_behaviour(self, bc):
        s = cut_stream(bc, "kero")
        c = BlendComponent.from_stream("jet", s, bc[1], flash_C=42.0)
        assert float(c.properties["flash_C"]) == 42.0
        old = BlendComponent.from_stream("jet", s, bc[1], estimate=False)
        assert not set(pr.ESTIMATED_PROPERTIES) & set(old.properties)
        sub = BlendComponent.from_stream("jet", s, bc[1], estimate=["smoke_mm"])
        assert set(sub.properties) - set(old.properties) == {"smoke_mm"}

    def test_octane_needs_the_hydrocarbon_types(self, bc):
        b = bc[1]
        bare = BlendCharacterization(names=b.names, Tb=b.Tb, SG=b.SG, MW=b.MW, Tc=b.Tc,
                                     Pc=b.Pc, omega=b.omega)
        assert pr.estimable(bare) == ("flash_C", "smoke_mm", "viscosity_cSt")
        with pytest.raises(KeyError):
            BlendComponent.from_stream("x", cut_stream(bc, "LSR"), bare, estimate=["RON"])

    def test_jet_pool_blends_the_estimates(self, bc):
        kero = BlendComponent.from_stream("kero", cut_stream(bc, "kero"), bc[1])
        res = BlendPool("jet")([kero], [1.0])
        for spec in ("freeze_C <= -40", "smoke_mm >= 18", "flash_C >= 38"):
            assert spec in res.margins
        assert float(res.properties["flash_C"]) == pytest.approx(
            float(kero.properties["flash_C"]), abs=1e-8)

    def test_estimates_are_differentiable_in_the_flows(self, bc):
        b = bc[1]
        m0 = cut_moles(bc, "HSR") + 1e-6

        def f(scale):
            p = pr.estimate_properties(b, m0 * scale, ["flash_C", "RON", "smoke_mm",
                                                       "viscosity_cSt"])
            return jnp.stack([p[k] for k in ("flash_C", "RON", "smoke_mm", "viscosity_cSt")])

        J = jax.jacfwd(f)(jnp.ones_like(m0))
        assert bool(jnp.all(jnp.isfinite(J)))
        assert float(jnp.max(jnp.abs(J))) > 0.0


@pytest.mark.release
def test_estimate_gradients_match_central_differences(bc):
    b = bc[1]
    m0 = cut_moles(bc, "kero") + 1e-6
    d = jnp.zeros_like(m0).at[-12].set(1e-3 * float(jnp.max(m0)))

    def f(t):
        p = pr.estimate_properties(b, m0 + t * d)
        return jnp.stack([p[k] for k in pr.ESTIMATED_PROPERTIES])

    g = jax.jacfwd(f)(0.0)
    h = 1e-4
    fd = (f(h) - f(-h)) / (2 * h)
    np.testing.assert_allclose(np.asarray(g), np.asarray(fd), rtol=1e-4, atol=1e-7)
