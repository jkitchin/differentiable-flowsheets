"""Tests for the fluid catalytic cracker (issue #308): difflow_refinery.fcc."""

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.fcc import (
    FCCFeed,
    FCCParams,
    FCCUnit,
    ILLUSTRATIVE_5LUMP,
    RegeneratorTemperatureWarning,
    arthur_co_co2,
    fcc_block,
    get_scheme,
    voorhies_coke,
)
from difflow_refinery.fcc import fractionator as frac
from difflow_refinery.fcc import kinetics as kin
from difflow_refinery.fcc import regenerator as rg
from difflow_refinery.fcc import species as sp
from difflow_refinery.fcc.feed import cut_indices
from difflow_refinery.vacuum import assay as va

VGO = (343.0 + 273.15, 550.0 + 273.15)
RATE = 50.0
BALANCE_TOL = 1e-8


def _assay(crude):
    return crude.to_assay(heavy_end=dr.HeavyEnd())


@pytest.fixture(scope="module")
def light_assay():
    return _assay(va.light_crude())


@pytest.fixture(scope="module")
def light_char(light_assay):
    return dr.characterize(light_assay)


@pytest.fixture(scope="module")
def light_feed(light_char):
    return FCCFeed.from_characterization(light_char, RATE, *VGO)


@pytest.fixture(scope="module")
def heavy_feed():
    return FCCFeed.from_characterization(dr.characterize(_assay(va.heavy_crude())), RATE, *VGO)


@pytest.fixture(scope="module")
def unit():
    return FCCUnit(FCCParams())


@pytest.fixture(scope="module")
def light_res(unit, light_feed):
    return unit.solve(light_feed)


@pytest.fixture(scope="module")
def heavy_res(unit, heavy_feed):
    return unit.solve(heavy_feed)


def _f(x):
    return float(np.asarray(x))


# =============================================================================
# Coefficients pinned to their sources
# =============================================================================


class TestCitedCoefficients:
    def test_arthur_constants_as_quoted(self):
        # 10^3.4 exp(-12400 / RT), R in cal/mol/K (constants unverified against Arthur 1951).
        assert rg.ARTHUR_A == pytest.approx(2511.886, rel=1e-6)
        assert rg.ARTHUR_E_CAL == 12400.0
        T = 980.0
        assert _f(arthur_co_co2(T)) == pytest.approx(10 ** 3.4 * np.exp(-12400.0 / (1.987204 * T)))

    def test_heats_of_formation_codata(self):
        assert sp.HF_298["carbon_dioxide"] == -393.51e3
        assert sp.HF_298["water"] == -241.826e3
        assert sp.HF_298["sulfur_dioxide"] == -296.81e3
        assert sp.HF_298["carbon_monoxide"] == -110.53e3

    def test_n2_co2_cp_match_difflow_database(self):
        from difflow.database import get_species_data

        for name in ("nitrogen", "carbon_dioxide"):
            assert tuple(sp.CP_IG[name]) == pytest.approx(get_species_data(name).Cp_coeffs)

    def test_atomic_weights_iupac_conventional(self):
        assert sp.ATOMIC_WEIGHT == {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999, "S": 32.06}
        assert sp.MW["methane"] == pytest.approx(16.043)
        assert sp.MW["hydrogen_sulfide"] == pytest.approx(34.076)

    def test_species_are_difflow_database_names(self):
        from difflow.database import list_species

        known = set(list_species())
        names = (sp.DRY_GAS_SPECIES + ("hydrogen_sulfide",) + sp.C3_SPECIES + sp.C4_SPECIES
                 + sp.FLUE_SPECIES)
        missing = [n for n in names if n not in known and n != "carbon_monoxide"]
        assert not missing, missing

    def test_conversion_is_defined_at_221C(self):
        assert frac.GASOLINE_EDGES_C[-1] == 221.0 == frac.CYCLE_EDGES_C[0]
        assert FCCParams().gasoline_cut == pytest.approx(221.0 + 273.15)


# =============================================================================
# Kinetics
# =============================================================================


class TestKinetics:
    @pytest.mark.parametrize("name", ["ancheyta_5", "lee_4", "weekman_nace_3"])
    def test_rates_conserve_mass(self, name):
        s = get_scheme(name)
        th = s.theta()
        y = jnp.linspace(0.6, 0.05, s.n)
        dy = s.rates(y, 800.0, 0.7, th["k_ref"], th["Ea"])
        assert abs(_f(jnp.sum(dy))) < 1e-15
        assert _f(dy[0]) < 0

    def test_gas_oil_second_order_gasoline_first(self):
        s = get_scheme("weekman_nace_3")
        th = s.theta()
        y = jnp.asarray([0.5, 0.3, 0.2])
        dy = s.rates(y, kin.T_REF_KINETICS, 1.0, th["k_ref"], th["Ea"])
        K0 = th["k_ref"][0] + th["k_ref"][1]
        assert _f(dy[0]) == pytest.approx(-K0 * 0.25)
        assert _f(dy[1]) == pytest.approx(th["k_ref"][0] * 0.25 - th["k_ref"][2] * 0.3)

    def test_aggregated_schemes_share_the_total_rates(self):
        t = ILLUSTRATIVE_5LUMP
        go_total = sum(k for a, _, k, _ in t if a == "gas_oil")
        for name in ("lee_4", "weekman_nace_3"):
            s = get_scheme(name)
            assert sum(r.k_ref for r in s.reactions if r.reactant == "gas_oil") == pytest.approx(go_total)

    @pytest.mark.parametrize("name", ["ancheyta_5", "lee_4", "weekman_nace_3"])
    def test_group_matrix_rows_sum_to_one(self, name):
        M = get_scheme(name).group_matrix({"dry_gas_share": jnp.asarray(0.2),
                                           "coke_share": jnp.asarray(0.25)})
        np.testing.assert_allclose(np.asarray(M.sum(axis=1)), 1.0, atol=1e-15)

    def test_jacob_10_lump_not_implemented(self):
        with pytest.raises(NotImplementedError):
            get_scheme("jacob_10")

    def test_deactivation_and_voorhies(self):
        assert _f(kin.deactivation("time", 2.0, 0.0, 0.1)) == pytest.approx(np.exp(-0.2))
        assert _f(kin.deactivation("coke", 0.0, 0.01, 50.0)) == pytest.approx(np.exp(-0.5))
        assert _f(voorhies_coke(4.0, 0.01, 0.5)) == pytest.approx(0.02)


# =============================================================================
# The heat-balanced unit
# =============================================================================


class TestConverges:
    @pytest.mark.parametrize("which", ["light_res", "heavy_res"])
    def test_converges_from_default_initialization(self, which, request):
        res = request.getfixturevalue(which)
        o = res["outputs"]
        assert bool(res["converged"])
        assert _f(res["residual"]) < 1e-10
        assert _f(o["riser_outlet_T"]) == pytest.approx(FCCParams().riser_outlet_T, abs=1e-8)
        assert abs(_f(o["regenerator_heat_residual"])) < 1e-3 * RATE  # W, out of ~1e8
        # Plausible operating point (illustrative constants, but not absurd).
        assert 3.0 < _f(o["cat_oil"]) < 12.0
        assert 880.0 < _f(o["regenerator_T"]) < 1033.15
        assert 0.55 < _f(o["conversion"]) < 0.85

    def test_heavy_feed_converts_less_and_needs_more_catalyst(self, light_res, heavy_res):
        lo, ho = light_res["outputs"], heavy_res["outputs"]
        assert _f(ho["conversion"]) < _f(lo["conversion"])
        assert _f(ho["cat_oil"]) > _f(lo["cat_oil"])

    def test_riser_resolution(self, light_feed, light_res):
        fine = FCCUnit(FCCParams(n_steps=800)).solve(light_feed)
        for k in ("conversion", "yield.gasoline", "yield.coke", "regenerator_T"):
            assert _f(fine["outputs"][k]) == pytest.approx(_f(light_res["outputs"][k]), rel=1e-8)


class TestBalances:
    @pytest.mark.parametrize("which", ["light_res", "heavy_res"])
    def test_mass_elements_energy_close(self, which, request):
        bal = request.getfixturevalue(which)["balances"]
        assert set(bal) == {"mass", "C", "H", "S", "N", "energy"}
        for k, v in bal.items():
            assert abs(_f(v)) < BALANCE_TOL, (k, _f(v))

    @pytest.mark.parametrize("which", ["light_res", "heavy_res"])
    def test_by_difference_quantities_are_physical(self, which, request):
        o = request.getfixturevalue(which)["outputs"]
        assert 0.06 < _f(o["cycle_oil_hydrogen"]) < 0.11
        assert _f(o["yield.h2s"]) > 0
        assert _f(o["slurry.hydrogen"]) < _f(o["lco.hydrogen"]) < _f(o["gasoline.hydrogen"])
        assert _f(o["slurry.sulfur"]) > _f(o["gasoline.sulfur"])
        assert 0.70 < _f(o["gasoline.SG"]) < 0.80
        assert 0.90 < _f(o["lco.SG"]) < 1.0
        assert 0.97 < _f(o["slurry.SG"]) < 1.10

    def test_yields_sum_to_one(self, light_res):
        o = light_res["outputs"]
        lumps = sum(_f(o[f"yield.{k}"]) for k in ("dry_gas", "lpg", "gasoline_lump", "cycle_oil", "coke"))
        prods = sum(_f(o[f"yield.{k}"]) for k in ("dry_gas", "c3", "c4", "gasoline", "lco", "slurry", "coke"))
        assert lumps == pytest.approx(1.0, abs=1e-12)
        assert prods == pytest.approx(1.0, abs=1e-12)

    def test_olefin_split_parameters(self, light_res):
        p = FCCParams()
        o = light_res["outputs"]
        assert _f(o["c3_olefins"]) == pytest.approx(p.propylene_in_c3)
        assert _f(o["c4_olefins"]) == pytest.approx(p.olefins_in_c4)
        assert _f(o["flue_o2"]) == pytest.approx(p.flue_o2)

    @pytest.mark.parametrize("kw", [dict(scheme="lee_4"), dict(scheme="weekman_nace_3"),
                                    dict(deactivation="coke"),
                                    dict(co_co2="arthur", flue_o2=0.002)])
    def test_options_converge_and_balance(self, light_feed, kw):
        res = FCCUnit(FCCParams(**kw)).solve(light_feed)
        assert bool(res["converged"])
        for k, v in res["balances"].items():
            assert abs(_f(v)) < BALANCE_TOL, (kw, k, _f(v))
        if kw.get("co_co2") == "arthur":
            o = res["outputs"]
            assert _f(o["co_co2"]) == pytest.approx(_f(arthur_co_co2(o["regenerator_T"])))

    def test_air_rate_mode_is_the_inverse_of_the_flue_o2_mode(self, light_feed, light_res):
        air = _f(light_res["outputs"]["air_rate"])
        res = FCCUnit(FCCParams(air_rate=air)).solve(light_feed)
        o = res["outputs"]
        assert _f(o["flue_o2"]) == pytest.approx(0.02, rel=1e-8)
        assert _f(o["regenerator_T"]) == pytest.approx(_f(light_res["outputs"]["regenerator_T"]), rel=1e-9)

    def test_partial_burn_runs_cooler(self, light_feed, light_res):
        res = FCCUnit(FCCParams(co_co2=0.5)).solve(light_feed)
        assert _f(res["outputs"]["regenerator_T"]) < _f(light_res["outputs"]["regenerator_T"])


@pytest.fixture(scope="module")
def rot_sweep(unit, light_feed):
    rots = [770.0, 800.0, 830.0, 860.0, 890.0]
    return rots, [unit.solve(light_feed, riser_outlet_T=t)["outputs"] for t in rots]


class TestTrends:
    def test_conversion_rises_with_riser_outlet_T(self, rot_sweep):
        _, outs = rot_sweep
        X = [_f(o["conversion"]) for o in outs]
        assert all(b > a for a, b in zip(X, X[1:]))

    def test_gasoline_passes_through_a_maximum_with_conversion(self, rot_sweep):
        _, outs = rot_sweep
        g = [_f(o["yield.gasoline_lump"]) for o in outs]
        i = int(np.argmax(g))
        assert 0 < i < len(g) - 1, g          # interior maximum: overcracking
        assert g[-1] < g[i] - 0.05

    def test_ccr_raises_regenerator_T_at_fixed_riser_T(self, unit, light_feed):
        T = [_f(unit.solve(dataclasses.replace(light_feed, ccr=jnp.asarray(c)))["outputs"]["regenerator_T"])
             for c in (0.0, 0.005, 0.01)]
        assert T[0] < T[1] < T[2]

    def test_overactive_catalyst_warns(self, unit, light_feed):
        with pytest.warns(RegeneratorTemperatureWarning):
            unit.solve(light_feed, activity=4.0)


class TestGradients:
    KEYS = ("conversion", "yield.gasoline", "yield.coke", "regenerator_T")

    def test_gradients_wrt_specs_match_central_fd(self, unit, heavy_feed):
        """d(conversion, gasoline, coke, T_rg)/d(ROT, preheat) on the heavy feed."""
        def f(z):
            o = unit.solve(heavy_feed, riser_outlet_T=z[0], feed_T=z[1])["outputs"]
            return jnp.stack([o[k] for k in self.KEYS])

        z0 = jnp.asarray([800.0, 520.0])
        J = np.asarray(jax.jacfwd(f)(z0))
        for j in range(2):
            e = jnp.zeros(2).at[j].set(0.05)
            fd = np.asarray((f(z0 + e) - f(z0 - e)) / 0.1)
            np.testing.assert_allclose(J[:, j], fd, rtol=1e-5, atol=1e-12)

    @pytest.mark.slow
    @pytest.mark.release
    def test_implicit_gradients_match_central_fd(self, light_assay, unit):
        """d(conversion, gasoline, coke, T_rg)/d(ROT, preheat, one VGO TBP point)."""
        cuts = dr.default_cut_points(light_assay)
        idx = cut_indices(dr.characterize(light_assay, cut_points=cuts), *VGO)
        I = 4  # a TBP point inside the VGO range (450 C)

        def f(z):
            assay = light_assay.with_tbp_point(I, z[2])
            feed = FCCFeed.from_characterization(dr.characterize(assay, cut_points=cuts),
                                                 RATE, indices=idx)
            o = unit.solve(feed, riser_outlet_T=z[0], feed_T=z[1])["outputs"]
            return jnp.stack([o[k] for k in self.KEYS])

        z0 = jnp.asarray([793.15, 500.0, float(np.asarray(light_assay.tbp_T)[I])])
        J = np.asarray(jax.jacfwd(f)(z0))
        h = 0.05
        for j in range(3):
            e = jnp.zeros(3).at[j].set(h)
            fd = np.asarray((f(z0 + e) - f(z0 - e)) / (2 * h))
            np.testing.assert_allclose(J[:, j], fd, rtol=1e-5, atol=1e-12)
            assert np.all(fd != 0.0)

    def test_reverse_mode_matches_forward(self, unit, light_feed):
        def g(rot):
            return unit.solve(light_feed, riser_outlet_T=rot)["outputs"]["regenerator_T"]

        x = jnp.asarray(800.0)
        assert _f(jax.grad(g)(x)) == pytest.approx(_f(jax.jacfwd(g)(x)), rel=1e-9)


# =============================================================================
# Feed, streams, profiles, planning
# =============================================================================


class TestFeed:
    def test_bulk_properties(self, light_feed):
        f = light_feed
        assert _f(f.mass) == pytest.approx(RATE)
        assert 300.0 < _f(f.MW) < 500.0
        assert 0.85 < _f(f.SG) < 0.95
        assert 616.15 < _f(f.Tb) < 823.15
        assert _f(f.Kw) == pytest.approx(_f(dr.watson_k(f.Tb, f.SG)))
        assert 0.10 < _f(f.hydrogen) < 0.14

    def test_from_stream_matches_from_components(self, light_char):
        pcs = light_char.pseudo_components()
        k = len(light_char.light_names)
        idx = cut_indices(light_char, *VGO)
        mass = np.zeros(pcs.n)
        mass[idx] = np.asarray(light_char.mass_fraction[k + idx]) * 10.0
        stream = {f"F_{n}": mass[i] / float(pcs.MW[i]) * 1000.0 for i, n in enumerate(pcs.names)}
        a = FCCFeed.from_stream(stream, pcs)
        b = FCCFeed.from_components(pcs, jnp.asarray(mass))
        for key, v in a.as_dict().items():
            assert _f(v) == pytest.approx(_f(getattr(b, key)), rel=1e-12)

    def test_composition_hydrogen_is_used(self, light_assay):
        char = dr.characterize(light_assay, composition=True)
        f = FCCFeed.from_characterization(char, RATE, *VGO)
        k = len(char.light_names)
        idx = cut_indices(char, *VGO)
        w = np.asarray(char.mass_fraction[k + idx])
        h = np.asarray(char.composition.hydrogen[k + idx])
        assert _f(f.hydrogen) == pytest.approx(float(np.sum(w * h) / np.sum(w)))


class TestStreams:
    def test_call_on_a_stream(self, light_char, light_res):
        pcs = light_char.pseudo_components()
        k = len(light_char.light_names)
        idx = cut_indices(light_char, *VGO)
        mass = np.zeros(pcs.n)
        w = np.asarray(light_char.mass_fraction[k + idx])
        mass[idx] = RATE * w / w.sum()
        stream = {f"F_{n}": mass[i] / float(pcs.MW[i]) * 1000.0 for i, n in enumerate(pcs.names)}
        stream.update({"F_methane": 0.5, "T": 500.0, "P": 3e5})
        out = FCCUnit(FCCParams(), components=pcs)(stream)
        streams, info = out[:-1], out[-1]
        assert len(streams) == len(FCCUnit.OUTLETS) == 8
        named = dict(zip(FCCUnit.OUTLETS, streams))
        assert set(k for k in named["c4"] if k.startswith("F_")) == {f"F_{n}" for n in sp.C4_SPECIES}
        assert {"F_propylene", "F_propane"} == set(k for k in named["c3"] if k.startswith("F_"))
        assert len([k for k in named["gasoline"] if k.startswith("F_")]) == len(frac.PSEUDO_NAMES)
        # Mass out = feed + passthrough + steam + air.
        mw = {**sp.MW, **dict(zip(frac.PSEUDO_NAMES, np.asarray(frac.pseudo_properties(
            {"gasoline": FCCParams().gasoline_kw, "cycle_oil": FCCParams().cycle_oil_kw})["MW"])))}
        m_out = sum(_f(v) * mw[k[2:]] / 1000.0 for s in streams for k, v in s.items()
                    if k.startswith("F_"))
        o = info["outputs"]
        m_in = RATE + 0.5 * sp.MW["methane"] / 1000.0 + FCCParams().steam_ratio * RATE + _f(o["air_rate"])
        assert m_out == pytest.approx(m_in, rel=1e-10)
        assert named["dry_gas"]["F_methane"] > 0.5
        assert _f(o["conversion"]) == pytest.approx(_f(light_res["outputs"]["conversion"]), rel=1e-9)


class TestProfile:
    def test_profile_ends_at_the_outlet(self, unit, light_feed, light_res):
        prof = unit.profile(light_feed, light_res["x"], n_points=31)
        go = np.asarray(prof["groups"][:, 0])
        T = np.asarray(prof["T"])
        assert np.all(np.diff(go) < 0) and np.all(np.diff(T) < 0)
        assert 1.0 - go[-1] == pytest.approx(_f(light_res["outputs"]["conversion"]), rel=1e-9)
        assert T[-1] == pytest.approx(FCCParams().riser_outlet_T, abs=1e-6)
        assert _f(prof["t_c"][-1]) == pytest.approx(_f(light_res["outputs"]["residence_time"]), rel=1e-9)


class TestPlanning:
    def test_block_delta_vectors(self, unit, light_feed, light_res):
        blk = fcc_block(unit, light_feed, levers=["riser_outlet_T", "feed.ccr"],
                        outputs=["conversion", "yield.gasoline", "regenerator_T"], jit=False)
        y = np.asarray(blk.fn(jnp.asarray(blk.u0)))
        assert y[0] == pytest.approx(_f(light_res["outputs"]["conversion"]), rel=1e-10)
        J = np.asarray(jax.jacfwd(blk.fn)(jnp.asarray(blk.u0)))
        h = 0.05
        e = np.array([h, 0.0])
        fd = (np.asarray(blk.fn(jnp.asarray(blk.u0) + e)) - np.asarray(blk.fn(jnp.asarray(blk.u0) - e))) / (2 * h)
        np.testing.assert_allclose(J[:, 0], fd, rtol=1e-5)
        assert J[2, 1] > 0           # more CCR -> hotter regenerator
        assert blk.metadata["u_units"][0] == "K"

    def test_unknown_lever(self, unit, light_feed):
        with pytest.raises(ValueError):
            fcc_block(unit, light_feed, levers=["no_such_knob"])
