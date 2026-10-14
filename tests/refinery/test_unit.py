"""Tests for product properties and the CrudeUnit assembly.

The unit is the 8-stage column of the other small-unit tests, closed by a 5 %
overflash. The 30-stage atmospheric column with three side products and two
pumparounds (``test_column.py``'s ``_atmospheric_params``) is imported here
for ``test_cdu_baseline.py``; it is not solved in this file.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery import Assay, characterize, products
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

from .test_column import LIGHT, PCT, T_C, _atmospheric_params  # noqa: F401  (re-exported)

jax.config.update("jax_enable_x64", True)

ASSAY = Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT)
BPD = 95_000.0

SMALL_ASSAY = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85)


def _params(overflash=0.05):
    return cc.CrudeColumnParams(
        n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
        specs=(cc.product_rate("naphtha", 0.25, basis="mole"), cc.overflash(overflash)))


@pytest.fixture(scope="module")
def unit():
    return dr.CrudeUnit(SMALL_ASSAY, _params())


@pytest.fixture(scope="module")
def solved(unit):
    return unit.solve(1.0, T=480.0, P=4e5, basis="mole")


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
        g = products.gaps(props, ["naphtha", "residue"])[("naphtha", "residue")]
        assert float(g) == pytest.approx(float(props["residue"].tbp_at(5) - props["naphtha"].tbp_at(95)))


class TestCrudeUnit:
    def test_converges(self, solved):
        assert bool(solved.converged)

    def test_feed_rate_in_barrels(self, unit):
        feed = unit.feed(BPD, T=480.0, P=4e5)
        th = unit.thermo
        flows = jnp.stack([feed[f"F_{n}"] for n in th.names])
        bpd = float(th.std_volume(flows)) * 86400.0 / cc.BARREL
        assert bpd == pytest.approx(BPD, rel=1e-10)

    def test_yields_add_up(self, solved):
        props = solved.properties
        assert sum(float(p.yield_volume) for p in props.values()) == pytest.approx(1.0, rel=1e-10)
        assert sum(float(p.yield_mass) for p in props.values()) == pytest.approx(1.0, rel=1e-10)

    def test_the_specified_product_rate_is_met(self, solved, unit):
        th = unit.thermo
        n = sum(float(solved.products["naphtha"][f"F_{c}"]) for c in th.names)
        assert n == pytest.approx(0.25, rel=1e-8)

    def test_products_get_heavier(self, solved):
        order = ["naphtha", "residue"]
        api = [float(solved.properties[n].api) for n in order]
        tbp50 = [float(solved.properties[n].tbp_at(50)) for n in order]
        assert np.all(np.diff(api) < 0)
        assert np.all(np.diff(tbp50) > 0)
        assert list(solved.gaps()) == list(zip(order[:-1], order[1:]))

    def test_table(self, solved):
        text = solved.table()
        assert "naphtha" in text and "coil outlet" in text and "MW fired" in text
        # lightest first
        assert text.index("naphtha") < text.index("residue")

    def test_furnace_is_added(self):
        params = cc.CrudeColumnParams(n_stages=8, feed_stage=7, bottom_steam=0.2,
                                      specs=(cc.product_rate("naphtha", 0.25, basis="mole"),
                                             cc.coil_outlet_temperature(620.0)))
        u = dr.CrudeUnit(SMALL_ASSAY, params)
        assert u.params.furnace == cc.Furnace()
        assert u.column.degrees_of_freedom() == 2


class TestSmallUnit:
    """Gradients through the whole unit, on the small column."""

    def test_gradients_with_respect_to_the_assay_and_the_preheat(self, unit):
        """d(residue API, fired duty)/d(crude SG, inlet T), through characterisation,
        thermo, furnace and column, against central differences."""

        def f(x):
            assay = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=x[0])
            r = unit.solve(1.0, T=x[1], P=4e5, basis="mole", assay=assay)
            y = jnp.stack([r.properties["residue"].api, r.column.furnace_fired_duty])
            return y, y

        value_and_jac = jax.jit(lambda x: jax.jacfwd(f, has_aux=True)(x))   # one compile for both
        x0 = jnp.array([0.85, 480.0])
        J = np.asarray(value_and_jac(x0)[0])
        for i, h in enumerate([1e-5, 0.01]):
            e = jnp.zeros(2).at[i].set(h)
            fd = np.asarray(value_and_jac(x0 + e)[1] - value_and_jac(x0 - e)[1]) / (2 * h)
            np.testing.assert_allclose(J[:, i], fd, rtol=1e-5)
        assert J[0, 0] < 0   # a heavier crude, a heavier residue
        assert J[1, 1] < 0   # a hotter preheat, less fired duty

    def test_assay_must_keep_the_components(self, unit):
        other = Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85, light_ends={"propane": 1.0})
        with pytest.raises(ValueError, match="different components"):
            unit.solve(1.0, T=480.0, P=4e5, basis="mole", assay=other)

    def test_bad_basis(self, unit):
        with pytest.raises(ValueError, match="basis"):
            unit.feed(1.0, T=480.0, P=4e5, basis="gallons")
