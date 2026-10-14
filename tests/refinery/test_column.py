"""Tests for the equation-oriented crude column."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

jax.config.update("jax_enable_x64", True)

PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]
LIGHT = {"propane": 0.5, "n_butane": 1.0, "n_pentane": 1.5}
STEAM = {"bottom": 150.0, "kero": 40.0, "diesel": 40.0, "ago": 20.0}


@pytest.fixture(scope="module")
def crude():
    return characterize(Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT))


@pytest.fixture(scope="module")
def thermo(crude):
    return ColumnThermo.from_characterization(crude)


@pytest.fixture(scope="module")
def feed(crude):
    # ~150 kg/s is about 100 000 bbl/d
    return crude.stream(150.0, T=273.15 + 360.0, P=1.9e5, basis="mass")


def _flows(stream, thermo):
    return jnp.stack([jnp.asarray(stream[f"F_{n}"]) for n in thermo.names])


def _feed_volume(feed, thermo):
    return float(thermo.std_volume(_flows(feed, thermo)))


def _atmospheric_params(feed, thermo, pa1_duty=25e6, **over):
    Vf = _feed_volume(feed, thermo)
    specs = (
        cc.product_rate("naphtha", 0.20 * Vf),
        cc.product_rate("kero", 0.11 * Vf),
        cc.product_rate("diesel", 0.17 * Vf),
        cc.product_rate("ago", 0.05 * Vf),
        cc.pumparound_duty("pa1", pa1_duty),
        cc.pumparound_delta_t("pa1", 60.0),
        cc.pumparound_duty("pa2", 20e6),
        cc.pumparound_delta_t("pa2", 60.0),
    )
    kw = dict(
        n_stages=30, feed_stage=27, P_top=1.5e5, P_bottom=1.9e5, P_condenser=1.3e5,
        bottom_steam=STEAM["bottom"], steam_T=273.15 + 260.0,
        side_products=(
            cc.SideProduct("kero", 9, 4, steam=STEAM["kero"]),
            cc.SideProduct("diesel", 16, 4, steam=STEAM["diesel"]),
            cc.SideProduct("ago", 22, 3, steam=STEAM["ago"]),
        ),
        pumparounds=(cc.Pumparound("pa1", 12, 10), cc.Pumparound("pa2", 19, 17)),
        specs=specs,
    )
    kw.update(over)
    return cc.CrudeColumnParams(**kw)


@pytest.fixture(scope="module")
def atmospheric(feed, thermo):
    col = cc.CrudeColumn(_atmospheric_params(feed, thermo), thermo)
    return col, col.solve(feed)


def _small(thermo, specs, **over):
    kw = dict(n_stages=8, feed_stage=7, bottom_steam=20.0, P_top=1.4e5, P_bottom=1.6e5,
              P_condenser=1.2e5, specs=specs)
    kw.update(over)
    return cc.CrudeColumn(cc.CrudeColumnParams(**kw), thermo)


def _enthalpy(stream, thermo, phase):
    flows = _flows(stream, thermo)
    T = stream["T"]
    if phase == "liquid":
        h = jnp.sum(flows * thermo.h_liquid(T)) + stream["F_water"] * thermo.water_h_liquid(T)
    else:
        h = jnp.sum(flows * thermo.h_vapor(T)) + stream["F_water"] * thermo.water_h_vapor(T)
    return float(h)


# =============================================================================
# The atmospheric column
# =============================================================================


class TestAtmosphericColumn:
    def test_converges(self, atmospheric):
        _, res = atmospheric
        assert bool(res.converged)
        assert float(res.residual_norm) < 1e-9

    def test_component_balance_closes(self, atmospheric, feed, thermo):
        _, res = atmospheric
        f = np.asarray(_flows(feed, thermo))
        out = sum(np.asarray(_flows(s, thermo)) for s in res.products.values())
        np.testing.assert_allclose(out, f, rtol=1e-10)

    def test_all_steam_leaves_as_free_water(self, atmospheric):
        _, res = atmospheric
        water = {k: float(v["F_water"]) for k, v in res.products.items()}
        assert water["water"] == pytest.approx(sum(STEAM.values()), rel=1e-10)
        assert all(w == 0.0 for k, w in water.items() if k != "water")

    def test_overall_energy_balance(self, atmospheric, feed, thermo):
        """Feed + steam in = products out + condenser + pumparound duties."""
        col, res = atmospheric
        f = _flows(feed, thermo)
        args = col._args(feed)
        f_liq, f_vap = col._feed_split(args)
        h_in = float(jnp.sum(f_liq * thermo.h_liquid(feed["T"])) + jnp.sum(f_vap * thermo.h_vapor(feed["T"])))
        h_in += sum(STEAM.values()) * float(thermo.water_h_vapor(273.15 + 260.0))
        h_out = sum(_enthalpy(s, thermo, "liquid") for s in res.products.values())
        removed = float(res.condenser_duty) + float(jnp.sum(res.pumparound_duty))
        assert h_in - h_out == pytest.approx(removed, rel=1e-8)
        assert float(jnp.sum(f)) > 0

    def test_specs_are_met(self, atmospheric, feed, thermo):
        _, res = atmospheric
        Vf = _feed_volume(feed, thermo)
        for name, frac in [("naphtha", 0.20), ("kero", 0.11), ("diesel", 0.17), ("ago", 0.05)]:
            got = float(thermo.std_volume(_flows(res.products[name], thermo)))
            assert got == pytest.approx(frac * Vf, rel=1e-8)
        np.testing.assert_allclose(res.pumparound_duty, [25e6, 20e6], rtol=1e-8)
        T_draw = res.T[np.array([12, 19]) - 1]
        np.testing.assert_allclose(T_draw - res.pumparound_return_T, [60.0, 60.0], atol=1e-6)

    def test_products_boil_in_order(self, atmospheric, thermo):
        _, res = atmospheric
        order = ["naphtha", "kero", "diesel", "ago", "residue"]
        tb = []
        for name in order:
            f = np.asarray(_flows(res.products[name], thermo))
            tb.append((f * np.asarray(thermo.Tb)).sum() / f.sum())
        assert np.all(np.diff(tb) > 0)

    def test_temperature_profile_is_plausible(self, atmospheric):
        _, res = atmospheric
        T = np.asarray(res.T) - 273.15
        # top 100-140 C with steam; flash zone below the 360 C furnace outlet
        assert 90 < T[0] < 150
        assert 320 < T[26] < 360
        # rises down the rectifying section, falls in the steam stripper below the feed
        assert np.all(np.diff(T[:27]) > 0)
        assert T[-1] < T[26]
        assert 20 < float(res.T_condenser) - 273.15 < 80

    def test_no_free_water_on_the_trays(self, atmospheric):
        _, res = atmospheric
        assert float(jnp.max(res.water_saturation)) < 1.0


# =============================================================================
# Smaller columns: other condensers, specs and draws
# =============================================================================


class TestVariants:
    def test_without_steam_there_is_no_water(self, crude, thermo):
        feed = crude.stream(10.0, T=273.15 + 350.0, P=1.6e5, basis="mass")
        F = float(sum(_flows(feed, thermo)))
        col = _small(thermo, (cc.product_rate("naphtha", 0.2 * F, basis="mole"),), bottom_steam=0.0)
        res = col.solve(feed)
        assert bool(res.converged)
        assert float(res.products["water"]["F_water"]) == pytest.approx(0.0, abs=1e-9)
        x0 = _flows(res.products["naphtha"], thermo)
        x0 = x0 / jnp.sum(x0)
        # the drum is at the distillate's hydrocarbon bubble point
        assert float(jnp.sum(thermo.K(res.T_condenser, 1.2e5) * x0)) == pytest.approx(1.0, abs=1e-9)

    def test_partial_condenser(self, crude, thermo):
        feed = crude.stream(10.0, T=273.15 + 350.0, P=1.6e5, basis="mass")
        F = float(sum(_flows(feed, thermo)))
        col = _small(thermo, (cc.product_rate("naphtha", 0.2 * F, basis="mole"),
                              cc.stage_temperature(0, 273.15 + 60.0)),
                     condenser="partial", bottom_steam=10.0)
        res = col.solve(feed)
        assert bool(res.converged)
        assert float(res.T_condenser) == pytest.approx(273.15 + 60.0, abs=1e-6)
        off = res.products["offgas"]
        f = np.asarray(_flows(off, thermo))
        assert f.sum() > 0 and float(off["F_water"]) > 0
        # the offgas is the lightest product
        i_prop = thermo.names.index("propane")
        assert f[i_prop] / f.sum() > float(res.products["naphtha"][f"F_{thermo.names[i_prop]}"]) / float(
            sum(_flows(res.products["naphtha"], thermo)))


class TestConfiguration:
    def test_spec_count_must_match_freedoms(self, thermo):
        with pytest.raises(ValueError, match="degrees of freedom"):
            _small(thermo, ())
        with pytest.raises(ValueError, match="degrees of freedom"):
            _small(thermo, (cc.reflux_ratio(2.0),), condenser="partial")

    def test_unknown_targets(self, thermo):
        with pytest.raises(ValueError, match="no product"):
            _small(thermo, (cc.product_rate("jet", 0.1),))
        with pytest.raises(ValueError, match="no pumparound"):
            _small(thermo, (cc.product_rate("naphtha", 0.1), cc.pumparound_duty("x", 1.0),
                            cc.pumparound_delta_t("x", 1.0)),
                   pumparounds=(cc.Pumparound("pa", 5, 3),))

    def test_bad_layout(self, thermo):
        spec = (cc.reflux_ratio(1.0),)
        with pytest.raises(ValueError, match="feed_stage"):
            _small(thermo, spec, feed_stage=9)
        with pytest.raises(ValueError, match="above the draw"):
            _small(thermo, spec + (cc.pumparound_duty("pa", 1.0), cc.pumparound_delta_t("pa", 1.0)),
                   pumparounds=(cc.Pumparound("pa", 3, 5),))
        with pytest.raises(ValueError, match="taken"):
            _small(thermo, spec + (cc.product_rate("residue", 0.1),),
                   side_products=(cc.SideProduct("residue", 4, 0),))

    def test_feed_must_match_thermo(self, thermo):
        col = _small(thermo, (cc.reflux_ratio(1.0),))
        with pytest.raises(ValueError, match="no flow"):
            col.solve({"T": 600.0, "P": 1.6e5})
