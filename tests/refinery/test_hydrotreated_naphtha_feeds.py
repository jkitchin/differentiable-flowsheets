"""Hydrotreater products to the reformer and the gas plant (#327).

Per commit (cheap, no hydrotreater solve): the two adapters on a product
grid built from a characterization -- an untreated grid maps exactly as
``NaphthaFeed.from_characterization`` does, a treated one (saturated
aromatics, heavier molecules) keeps its mass to round-off and its lower
aromatics, sulfur rides along, dissolved gases are refused or dropped; and a
gas-plant naphtha splitter converges and balances on such a grid.

Slow: the real chain -- example 40's naphtha through the naphtha
hydrotreater, then (a) a splitter on the treated naphtha and (b) the
fractionated heavy naphtha into the reformer.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import BlendCharacterization
from difflow_refinery.composition import CompositionRangeWarning
from difflow_refinery.gasplant import GasColumnConvergenceWarning, GasPlantColumn, splitter
from difflow_refinery.gasplant.hydroprocessed import (
    hydroprocessed_feed, product_components, resolve_product, stream_mass)
from difflow_refinery.reforming import NaphthaFeed
from difflow_refinery.reforming import species as sp

jax.config.update("jax_enable_x64", True)

C = 273.15


@pytest.fixture(scope="module")
def naphtha_char():
    """A naphtha characterized on 15 C cuts: C3-C5 light ends, cuts to ~180 C."""
    assay = dr.Assay([0, 10, 30, 50, 70, 90, 100], [t + C for t in [30, 60, 95, 120, 145, 170, 190]], sg=0.74,
                     light_ends={"propane": 0.3, "isobutane": 0.5, "n_butane": 1.0, "isopentane": 2.0,
                                 "n_pentane": 2.5},
                     sulfur_wt=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        return dr.characterize(assay, [C + t for t in range(60, 190, 15)], composition=True)


def _treated(char, rate=200.0):
    """``(untreated grid, treated grid, stream)`` on ``char``: the treated grid is
    the untreated one with half the aromatics saturated to naphthenes (volume
    basis, as the hydrotreater reports them), 1 % heavier cuts and a tenth of
    the sulfur -- what a hydrotreater does to it, in kind."""
    raw = BlendCharacterization.from_characterization(char)
    q = dict(raw.qualities)
    k = len(char.light_names)
    cut = jnp.arange(len(raw.names)) >= k
    a = q["aromatics_vol"]
    q["aromatics_vol"] = jnp.where(cut, 0.5 * a, a)
    q["naphthenes_vol"] = jnp.where(cut, q["naphthenes_vol"] + 0.5 * a, q["naphthenes_vol"])
    q["S_ppm"] = 0.1 * q["S_ppm"]
    treated = dataclasses.replace(raw, MW=jnp.where(cut, 1.01 * raw.MW, raw.MW), qualities=q)
    F = jnp.asarray(char.mole_fraction) * rate
    stream = {f"F_{n}": F[i] for i, n in enumerate(char.names)}
    stream.update(T=jnp.asarray(320.0), P=jnp.asarray(4e5))
    return raw, treated, stream


@pytest.fixture(scope="module")
def grids(naphtha_char):
    return _treated(naphtha_char)


@pytest.fixture(scope="module")
def small_grid():
    """The per-commit splitter's naphtha: C5 light ends and five 40 C cuts (7 components)."""
    assay = dr.Assay([0, 10, 30, 50, 70, 90, 100], [t + C for t in [30, 60, 95, 120, 145, 170, 190]], sg=0.74,
                     light_ends={"isopentane": 2.0, "n_pentane": 2.5}, sulfur_wt=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CompositionRangeWarning)
        char = dr.characterize(assay, [C + t for t in (60, 100, 140, 180)], composition=True)
    return _treated(char)


class TestNaphthaFeed:
    def test_an_untreated_grid_maps_as_from_characterization(self, naphtha_char, grids):
        raw, _, stream = grids
        a = NaphthaFeed.from_hydrotreater(stream, char=raw)
        b = NaphthaFeed.from_characterization(naphtha_char, stream)
        np.testing.assert_allclose(np.asarray(a.flows), np.asarray(b.flows), rtol=1e-12, atol=1e-14)

    def test_treated_mass_is_conserved_and_aromatics_are_the_treated_ones(self, naphtha_char, grids):
        raw, treated, stream = grids
        f = NaphthaFeed.from_hydrotreater(stream, char=treated)
        assert float(f.mass_flow) == pytest.approx(float(stream_mass(stream, treated)), rel=1e-13)
        # the untreated route on the treated stream loses the MW change
        g = NaphthaFeed.from_characterization(naphtha_char, stream)
        assert abs(float(g.mass_flow) / float(f.mass_flow) - 1.0) > 5e-3
        ar_t = float(f.group_fractions("volume")["aromatics"])
        ar_r = float(g.group_fractions("volume")["aromatics"])
        assert 0.4 * ar_r < ar_t < 0.6 * ar_r
        assert float(f.group_fractions("volume")["naphthenes"]) > float(g.group_fractions("volume")["naphthenes"])

    def test_sulfur_rides_along(self, grids):
        raw, treated, stream = grids
        f = NaphthaFeed.from_hydrotreater(stream, char=treated)
        mass = jnp.stack([stream[f"F_{n}"] for n in treated.names]) * treated.MW / 1000.0
        S = float(jnp.sum(mass * treated.qualities["S_ppm"]) / jnp.sum(mass))
        assert S > 0.0
        assert float(f.sulfur_wppm) == pytest.approx(S, rel=1e-13)
        # kept through the rate and group levers; zero from the other builders
        assert float(f.scaled(2.0).sulfur_wppm) == float(f.sulfur_wppm)
        assert float(f.with_group_fraction("naphthenes", 0.4).sulfur_wppm) == float(f.sulfur_wppm)
        assert float(NaphthaFeed.from_hydrotreater(stream, char=dataclasses.replace(
            treated, qualities={k: v for k, v in treated.qualities.items() if k != "S_ppm"})).sulfur_wppm) == 0.0

    def test_dissolved_gases_are_refused_or_dropped_and_light_gases_map(self, grids):
        _, treated, stream = grids
        wild = dict(stream, F_hydrogen=jnp.asarray(0.5), F_hydrogen_sulfide=jnp.asarray(0.01),
                    F_methane=jnp.asarray(0.2))
        with pytest.raises(ValueError, match="drop_gases"):
            NaphthaFeed.from_hydrotreater(wild, char=treated)
        f = NaphthaFeed.from_hydrotreater(wild, char=treated, drop_gases=True)
        base = NaphthaFeed.from_hydrotreater(stream, char=treated)
        assert float(f.flows[sp.INDEX["H2"]]) == 0.0
        assert float(f.flows[sp.INDEX["C1"]]) == pytest.approx(0.2 * 16.04246 / sp.MW[sp.INDEX["C1"]], rel=1e-12)
        dm = float(f.mass_flow - base.mass_flow)
        assert dm == pytest.approx(0.2 * 16.04246e-3, rel=1e-12)

    def test_differentiable_through_the_grid(self, grids):
        _, treated, stream = grids

        def ar(scale):
            q = dict(treated.qualities, aromatics_vol=treated.qualities["aromatics_vol"] * scale)
            g = dataclasses.replace(treated, qualities=q)
            return NaphthaFeed.from_hydrotreater(stream, char=g).flows[sp.INDEX["A8"]]

        d = jax.jacfwd(ar)(1.0)
        fd = (ar(1.0 + 1e-5) - ar(1.0 - 1e-5)) / 2e-5
        assert float(d) > 0.0
        assert float(d) == pytest.approx(float(fd), rel=1e-7)

    def test_a_grid_without_types_is_refused(self, grids):
        raw, _, stream = grids
        bare = dataclasses.replace(raw, qualities={"S_ppm": raw.qualities["S_ppm"]})
        with pytest.raises(ValueError, match="hydrocarbon types"):
            NaphthaFeed.from_hydrotreater(stream, char=bare)


class TestGasPlantFeed:
    def test_table_and_stream(self, grids):
        _, treated, stream = grids
        wild = dict(stream, F_hydrogen=jnp.asarray(0.05), F_hydrogen_sulfide=jnp.asarray(0.01),
                    F_ammonia=jnp.asarray(0.001))
        with pytest.raises(ValueError, match="ammonia"):
            hydroprocessed_feed(wild, char=treated)
        feed = hydroprocessed_feed(wild, char=treated, unsupported="drop")
        names = feed.components.names
        assert names[:2] == ("hydrogen", "hydrogen_sulfide")
        assert set(names[2:]) == set(treated.names)
        assert set(feed.dropped) == {"ammonia"}
        # light ends and cuts at the grid's molar masses: the mass is the product's less the NH3
        m = float(stream_mass(wild, treated)) - 0.001 * 17.03052e-3
        assert float(feed.mass_flow) == pytest.approx(m, rel=1e-13)
        i = names.index(treated.names[-1])
        assert float(feed.components.MW[i]) == float(treated.MW[-1])
        assert float(feed.components.pseudo[i]) == 1.0 and float(feed.components.pseudo[0]) == 0.0
        assert float(feed.stream["T"]) == 320.0
        assert set(feed.below(C + 85.0)) >= {"hydrogen", "propane", "n_pentane"}
        assert treated.names[-1] not in feed.below(C + 85.0)
        # drop_gases: none of the off-grid gases, all of them in dropped, the rest of the mass exact
        bare = hydroprocessed_feed(wild, char=treated, drop_gases=True)
        assert bare.components.names == tuple(treated.names)
        assert set(bare.dropped) == {"hydrogen", "hydrogen_sulfide", "ammonia"}
        assert float(bare.mass_flow) == pytest.approx(float(stream_mass(stream, treated)), rel=1e-13)

    def test_min_flow_trims_the_table(self, grids):
        _, treated, stream = grids
        s = dict(stream)
        last = treated.names[-1]
        s[f"F_{last}"] = jnp.asarray(1e-12)
        feed = hydroprocessed_feed(s, char=treated, min_flow=1e-9)
        assert last not in feed.components.names and last in feed.dropped
        assert len(product_components(treated).names) == len(treated.names)

    def test_resolve_refuses_what_it_cannot_read(self, grids):
        with pytest.raises(ValueError, match="char="):
            resolve_product({"F_x": 1.0})
        with pytest.raises(TypeError):
            resolve_product(3.0)

    def test_a_naphtha_splitter_converges_and_balances_on_a_treated_grid(self, small_grid):
        """The per-commit column: 7 components, 8 theoretical trays (compile dominates)."""
        _, treated, stream = small_grid
        feed = hydroprocessed_feed(stream, char=treated, T=C + 100.0, P=4e5)
        light = feed.below(C + 85.0)
        assert light == ("isopentane", "n_pentane", "pc01", "pc02")
        p = splitter(feed.components, light, heavy_spec=("bottoms.x.light", 0.05),
                     light_spec=("distillate.x.heavy", 0.05), n_trays=8, feed_tray=4, top_P=3e5)
        col = GasPlantColumn(p)
        with warnings.catch_warnings():
            warnings.simplefilter("error", GasColumnConvergenceWarning)
            top, bottom, info = col(feed.stream)
        assert bool(info["converged"])
        names = feed.components.names
        F = np.array([float(feed.stream[f"F_{n}"]) for n in names])
        out = np.array([float(top[f"F_{n}"]) + float(bottom[f"F_{n}"]) for n in names])
        np.testing.assert_allclose(out, F, rtol=1e-8, atol=1e-10 * F.sum())
        assert float(info["outputs"]["bottoms.x.light"]) == pytest.approx(0.05, rel=1e-6)
        assert float(info["outputs"]["distillate.x.heavy"]) == pytest.approx(0.05, rel=1e-6)


# =============================================================================
# The real chain (slow): example 40's naphtha hydrotreater
# =============================================================================


@pytest.fixture(scope="module")
def nht():
    from .test_hydrotreating_naphtha import NAPHTHA, NHT, PCT, T_C
    from difflow_refinery.hydrotreating import Hydrotreater

    assay = dr.Assay(PCT, [t + C for t in T_C], sg=0.86,
                     light_ends={"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
                                 "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5},
                     sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=5.0)
    cut_points = dr.default_cut_points(assay, ((673.15, 30.0), (873.15, 60.0), (np.inf, 150.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(assay, cut_points, composition=True)
    # a 300 C bed leaves product sulfur the reformer's balance can see
    unit = Hydrotreater(char, NAPHTHA, dataclasses.replace(NHT, T_in=(C + 300.0,)))
    res = unit.solve(NAPHTHA)
    assert bool(res.converged)
    return char, res


@pytest.mark.slow
def test_nht_to_reformer(nht):
    from difflow_refinery.reforming import CatalyticReformer, ReformerParams

    char, res = nht
    fr = res.fractionate(cut_points=(C + 85.0,), products=("light_naphtha", "heavy_naphtha"),
                         feeds=("product", "wild_naphtha"))
    heavy = fr.products["heavy_naphtha"]
    feed = NaphthaFeed.from_hydrotreater(fr, "heavy_naphtha")
    assert float(feed.mass_flow) == pytest.approx(float(fr.rates["heavy_naphtha"]), rel=1e-13)
    # the whole stripper bottoms, too
    whole = NaphthaFeed.from_hydrotreater(res)
    assert float(whole.mass_flow) == pytest.approx(float(res.outputs["product.rate"]), rel=1e-13)
    # the treated aromatics: the grid's, not the crude's
    old = NaphthaFeed.from_characterization(char, heavy)
    assert abs(float(old.mass_flow) / float(feed.mass_flow) - 1.0) > 1e-5
    S = float(feed.sulfur_wppm)
    assert 0.0 < S < 5.0
    ref = CatalyticReformer(ReformerParams()).solve(feed)
    assert bool(ref.converged)
    assert float(ref.feed_sulfur_wppm) == pytest.approx(S, rel=1e-12)
    b = ref.balances()
    for k in ("mass", "carbon", "hydrogen", "sulfur"):
        assert abs(float(b[k])) < 1e-7, k


@pytest.mark.slow
def test_splitter_on_the_hydrotreated_naphtha(nht):
    _, res = nht
    # the wild naphtha's dissolved H2/H2S/NH3/C1/C2 leave before a splitter (a total condenser
    # cannot condense hydrogen), so they are dropped -- and the mass accounts for them exactly
    feed = hydroprocessed_feed(res, ("product", "wild_naphtha"), T=C + 100.0, P=4e5, drop_gases=True)
    stream, grid = resolve_product(res, ("product", "wild_naphtha"))
    gas = {k: v for k, v in stream.items() if k.startswith("F_") and k[2:] not in grid.names}
    assert gas and set(feed.dropped) == {k[2:] for k in gas}
    m = float(stream_mass(stream, grid)) - float(stream_mass(gas, grid))
    assert float(feed.mass_flow) == pytest.approx(m, rel=1e-13)
    p = splitter(feed.components, feed.below(C + 85.0), heavy_spec=("bottoms.x.light", 0.02),
                 light_spec=("distillate.x.heavy", 0.02), n_trays=12, feed_tray=6, top_P=3e5)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasColumnConvergenceWarning)
        top, bottom, info = GasPlantColumn(p)(feed.stream)
    assert bool(info["converged"])
    names = feed.components.names
    F = np.array([float(feed.stream[f"F_{n}"]) for n in names])
    out = np.array([float(top[f"F_{n}"]) + float(bottom[f"F_{n}"]) for n in names])
    np.testing.assert_allclose(out, F, rtol=1e-8, atol=1e-10 * F.sum())
    heavy = NaphthaFeed.from_hydrotreater(
        {k: v for k, v in bottom.items() if k.startswith("F_") and k[2:] in res.product_char.names},
        char=res.product_char)
    assert float(heavy.mass_flow) > 0.0
