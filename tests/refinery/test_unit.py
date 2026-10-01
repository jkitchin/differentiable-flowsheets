"""Tests for product properties and the CrudeUnit assembly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, characterize, products
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

from .test_column import LIGHT, PCT, T_C, _atmospheric_params

jax.config.update("jax_enable_x64", True)

ASSAY = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT)
BPD = 95_000.0


@pytest.fixture(scope="module")
def unit():
    crude = characterize(ASSAY)
    th = ColumnThermo.from_characterization(crude)
    # specs are absolute rates: size them on the crude the unit will run
    kg_s = BPD * cc.BARREL / 86400.0 * float(crude.bulk_sg) * 999.016
    feed = crude.stream(kg_s, T=600.0, P=1.9e5, basis="mass")
    base = _atmospheric_params(feed, th, pa1_duty=15e6)
    params = _atmospheric_params(feed, th, pa1_duty=15e6, specs=base.specs + (cc.overflash(0.05),))
    return dr.CrudeUnit(ASSAY, params)


@pytest.fixture(scope="module")
def solved(unit):
    return unit.solve(BPD, T=273.15 + 240.0, P=6e5)


class TestProductProperties:
    def test_a_pure_component_boils_at_its_boiling_point(self):
        th = ColumnThermo.from_characterization(characterize(ASSAY))
        flows = jnp.zeros(th.n_components).at[5].set(1.0)
        np.testing.assert_allclose(products.tbp_curve(flows, th), th.Tb[5], rtol=1e-12)

    def test_whole_crude_recovers_its_assay(self):
        """The crude's own TBP, rebuilt from its cuts, is within a cut of the data."""
        crude = characterize(ASSAY)
        th = ColumnThermo.from_characterization(crude)
        flows = jnp.asarray(crude.mole_fraction)
        T = products.tbp_curve(flows, th, percents=(30.0, 50.0, 70.0))
        np.testing.assert_allclose(np.asarray(T) - 273.15, [205.0, 315.0, 430.0], atol=20.0)

    def test_whole_crude_gravity(self):
        crude = characterize(ASSAY)
        th = ColumnThermo.from_characterization(crude)
        feed = crude.stream(10.0, T=300.0, P=1e5)
        props = products.product_properties({"crude": feed}, th, feed)
        assert float(props["crude"].sg) == pytest.approx(float(crude.bulk_sg), rel=1e-12)
        assert float(props["crude"].yield_volume) == pytest.approx(1.0)

    def test_gap_definition(self, solved):
        props = solved.properties
        g = products.gaps(props, ["naphtha", "kero"])[("naphtha", "kero")]
        assert float(g) == pytest.approx(float(props["kero"].tbp_at(5) - props["naphtha"].tbp_at(95)))


class TestCrudeUnit:
    def test_converges(self, solved):
        assert bool(solved.converged)

    def test_feed_rate_in_barrels(self, solved, unit):
        th = unit.thermo
        flows = jnp.stack([solved.feed[f"F_{n}"] for n in th.names])
        bpd = float(th.std_volume(flows)) * 86400.0 / cc.BARREL
        assert bpd == pytest.approx(BPD, rel=1e-10)

    def test_yields_add_up(self, solved):
        props = solved.properties
        assert sum(float(p.yield_volume) for p in props.values()) == pytest.approx(1.0, rel=1e-10)
        assert sum(float(p.yield_mass) for p in props.values()) == pytest.approx(1.0, rel=1e-10)
        assert sum(float(p.bpd) for p in props.values()) == pytest.approx(BPD, rel=1e-10)

    def test_specified_yields(self, solved):
        props = solved.properties
        for name, frac in [("naphtha", 0.20), ("kero", 0.11), ("diesel", 0.17), ("ago", 0.05)]:
            assert float(props[name].yield_volume) == pytest.approx(frac, rel=1e-8)

    def test_products_get_heavier(self, solved):
        order = ["naphtha", "kero", "diesel", "ago", "residue"]
        api = [float(solved.properties[n].api) for n in order]
        tbp50 = [float(solved.properties[n].tbp_at(50)) for n in order]
        assert np.all(np.diff(api) < 0)
        assert np.all(np.diff(tbp50) > 0)
        assert list(solved.gaps()) == list(zip(order[:-1], order[1:]))

    def test_table(self, solved):
        text = solved.table()
        assert "naphtha" in text and "coil outlet" in text and "MW fired" in text
        # lightest first
        assert text.index("naphtha") < text.index("kero") < text.index("residue")

    def test_furnace_is_added(self):
        params = cc.CrudeColumnParams(n_stages=8, feed_stage=7, bottom_steam=0.2,
                                      specs=(cc.product_rate("naphtha", 0.25, basis="mole"),
                                             cc.coil_outlet_temperature(620.0)))
        u = dr.CrudeUnit(Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85), params)
        assert u.params.furnace == cc.Furnace()
        assert u.column.degrees_of_freedom() == 2


class TestSmallUnit:
    """Gradients through the whole unit, on a small column."""

    ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)

    @pytest.fixture(scope="class")
    def small(self):
        params = cc.CrudeColumnParams(
            n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
            specs=(cc.product_rate("naphtha", 0.25, basis="mole"), cc.overflash(0.05)))
        return dr.CrudeUnit(self.ASSAY, params)

    def test_gradient_with_respect_to_the_assay(self, small):
        """d(residue API)/d(crude SG), through characterisation, thermo, furnace and column."""

        def residue_api(sg):
            assay = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=sg)
            return small.solve(1.0, T=480.0, P=4e5, basis="mole", assay=assay).properties["residue"].api

        g = float(jax.grad(residue_api)(0.85))
        h = 1e-5
        fd = float((residue_api(0.85 + h) - residue_api(0.85 - h)) / (2 * h))
        assert g < 0  # a heavier crude, a heavier residue
        assert g == pytest.approx(fd, rel=1e-5)

    def test_gradient_with_respect_to_preheat(self, small):
        def fired(T):
            return small.solve(1.0, T=T, P=4e5, basis="mole").column.furnace_fired_duty

        g = float(jax.grad(fired)(480.0))
        fd = float((fired(480.01) - fired(479.99)) / 0.02)
        assert g < 0
        assert g == pytest.approx(fd, rel=1e-5)

    def test_assay_must_keep_the_components(self, small):
        other = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85, light_ends={"propane": 1.0})
        with pytest.raises(ValueError, match="different components"):
            small.solve(1.0, T=480.0, P=4e5, basis="mole", assay=other)

    def test_new_params_keep_the_furnace(self, small):
        params = cc.CrudeColumnParams(
            n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
            specs=(cc.product_rate("naphtha", 0.25, basis="mole"), cc.overflash(0.08)))
        res = small.solve(1.0, T=480.0, P=4e5, basis="mole", params=params)
        base = small.solve(1.0, T=480.0, P=4e5, basis="mole")
        assert bool(res.converged)
        assert float(res.column.coil_outlet_T) > float(base.column.coil_outlet_T)

    def test_bad_basis(self, small):
        with pytest.raises(ValueError, match="basis"):
            small.feed(1.0, T=480.0, P=4e5, basis="gallons")
