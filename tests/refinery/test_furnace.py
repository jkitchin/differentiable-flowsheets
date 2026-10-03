"""Tests for the furnace solved with the crude column."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo

from .test_column import LIGHT, PCT, STEAM, T_C, _atmospheric_params, _enthalpy, _feed_volume, _flows

jax.config.update("jax_enable_x64", True)

T_INLET = 273.15 + 240.0  # preheat train outlet
P_INLET = 6e5


@pytest.fixture(scope="module")
def crude():
    return characterize(Assay(PCT, [t + 273.15 for t in T_C], sg=0.86, light_ends=LIGHT))


@pytest.fixture(scope="module")
def thermo(crude):
    return ColumnThermo.from_characterization(crude)


@pytest.fixture(scope="module")
def inlet(crude):
    return crude.stream(150.0, T=T_INLET, P=P_INLET, basis="mass")


@pytest.fixture(scope="module")
def small():
    crude = characterize(Assay([0, 30, 70, 100], [320.0, 480.0, 640.0, 900.0], sg=0.85))
    th = ColumnThermo.from_characterization(crude)
    return crude, th


def _small(thermo, specs, **over):
    kw = dict(n_stages=8, feed_stage=7, bottom_steam=0.2, P_top=1.4e5, P_bottom=1.6e5,
              P_condenser=1.2e5, specs=specs, furnace=cc.Furnace())
    kw.update(over)
    return cc.CrudeColumn(cc.CrudeColumnParams(**kw), thermo)


def _overflash_params(inlet, thermo, of, pa1_duty=15e6):
    base = _atmospheric_params(inlet, thermo, pa1_duty=pa1_duty)
    return _atmospheric_params(inlet, thermo, pa1_duty=pa1_duty, furnace=cc.Furnace(outlet_P=1.9e5),
                               specs=base.specs + (cc.overflash(of),))


@pytest.fixture(scope="module")
def overflashed(inlet, thermo):
    col = cc.CrudeColumn(_overflash_params(inlet, thermo, 0.05), thermo)
    return col, col.solve(inlet)


class TestOverflashSetsTheFurnace:
    def test_converges(self, overflashed):
        _, res = overflashed
        assert bool(res.converged)

    def test_overflash_is_met(self, overflashed, inlet, thermo):
        _, res = overflashed
        wash = float(thermo.std_volume(res.x[25] * res.L[25]))  # stage 26, above the feed
        assert wash / _feed_volume(inlet, thermo) == pytest.approx(0.05, rel=1e-8)

    def test_coil_outlet_is_plausible(self, overflashed):
        _, res = overflashed
        assert 290.0 < float(res.coil_outlet_T) - 273.15 < 360.0
        assert 0.0 < float(res.furnace_duty) < float(res.furnace_fired_duty)
        assert float(res.furnace_fired_duty) == pytest.approx(float(res.furnace_duty) / 0.85, rel=1e-12)

    def test_energy_balance_through_the_furnace(self, overflashed, inlet, thermo):
        """Crude at the inlet + furnace + steam = products + condenser + pumparounds."""
        col, res = overflashed
        args = col._args(inlet)
        h_in = float(col._feed_enthalpy(args, args["T_in"], args["P_in"])[0])
        h_in += float(res.furnace_duty)
        h_in += sum(STEAM.values()) * float(thermo.water_h_vapor(273.15 + 260.0))
        h_out = sum(_enthalpy(s, thermo, "liquid") for s in res.products.values())
        removed = float(res.condenser_duty) + float(jnp.sum(res.pumparound_duty))
        assert h_in - h_out == pytest.approx(removed, rel=1e-8)

    def test_too_little_overflash_dries_the_column(self, inlet, thermo):
        """With 25 MW off PA1, a 5 % overflash leaves no reflux: no solution.

        Physically right -- the heat the pumparound removes has to come in
        with the feed -- and reported as not converged, not as an answer.
        """
        res = cc.CrudeColumn(_overflash_params(inlet, thermo, 0.05, pa1_duty=25e6), thermo).solve(inlet)
        assert not bool(res.converged)


class TestFurnaceSpecs:
    def test_coil_outlet_spec_reproduces_a_fixed_feed(self, small):
        crude, th = small
        hot = crude.stream(1.0, T=620.0, P=1.6e5, basis="mole")
        fixed = _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"),), furnace=None).solve(hot)
        cold = crude.stream(1.0, T=480.0, P=4e5, basis="mole")
        fired = _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"),
                            cc.coil_outlet_temperature(620.0)),
                       furnace=cc.Furnace(outlet_P=1.6e5)).solve(cold)
        assert bool(fired.converged)
        np.testing.assert_allclose(fired.T, fixed.T, atol=1e-8)
        np.testing.assert_allclose(fired.condenser_duty, fixed.condenser_duty, rtol=1e-9)
        assert float(fixed.furnace_duty) == 0.0

    def test_duty_spec_inverts_the_outlet_spec(self, small):
        crude, th = small
        cold = crude.stream(1.0, T=480.0, P=4e5, basis="mole")
        naphtha = cc.product_rate("naphtha", 0.25, basis="mole")
        by_T = _small(th, (naphtha, cc.coil_outlet_temperature(620.0))).solve(cold)
        by_Q = _small(th, (naphtha, cc.furnace_duty(by_T.furnace_duty))).solve(cold)
        assert bool(by_Q.converged)
        assert float(by_Q.coil_outlet_T) == pytest.approx(620.0, abs=1e-6)

    def test_duty_is_the_enthalpy_rise(self, small):
        crude, th = small
        cold = crude.stream(1.0, T=480.0, P=4e5, basis="mole")
        col = _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"), cc.coil_outlet_temperature(620.0)))
        res = col.solve(cold)
        args = col._args(cold)
        dh = col._feed_enthalpy(args, 620.0, args["P_F"])[0] - col._feed_enthalpy(args, 480.0, 4e5)[0]
        assert float(res.furnace_duty) == pytest.approx(float(dh), rel=1e-12)
        # the default outlet pressure is the feed stage's
        assert float(args["P_F"]) == pytest.approx(1.4e5 + (1.6e5 - 1.4e5) * 6 / 7)

    def test_gradient_of_coil_outlet_with_overflash(self, small):
        crude, th = small
        cold = crude.stream(1.0, T=480.0, P=4e5, basis="mole")

        def cot(of):
            col = _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"), cc.overflash(of)))
            return col.solve(cold).coil_outlet_T

        assert bool(jnp.isfinite(cot(0.05)))
        g = float(jax.grad(cot)(0.05))
        h = 1e-5
        fd = float((cot(0.05 + h) - cot(0.05 - h)) / (2 * h))
        assert g > 0  # more wash liquid needs a hotter furnace
        assert g == pytest.approx(fd, rel=1e-5)

    def test_vmap_over_inlet_temperature(self, small):
        crude, th = small
        cold = crude.stream(1.0, T=480.0, P=4e5, basis="mole")
        col = _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"), cc.coil_outlet_temperature(620.0)))
        res = jax.vmap(lambda t: col.solve({**cold, "T": t}))(jnp.array([460.0, 480.0, 500.0]))
        assert bool(jnp.all(res.converged))
        # better preheat, less furnace; the column itself does not change
        assert bool(jnp.all(jnp.diff(res.furnace_duty) < 0))
        np.testing.assert_allclose(res.condenser_duty, res.condenser_duty[0], rtol=1e-9)


class TestConfiguration:
    def test_furnace_adds_a_freedom(self, small):
        _, th = small
        with pytest.raises(ValueError, match="1 furnace"):
            _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"),))

    def test_furnace_spec_needs_a_furnace(self, small):
        _, th = small
        with pytest.raises(ValueError, match="needs a Furnace"):
            _small(th, (cc.coil_outlet_temperature(620.0),), furnace=None)

    def test_flows_balance(self, overflashed, inlet, thermo):
        _, res = overflashed
        out = sum(np.asarray(_flows(s, thermo)) for s in res.products.values())
        np.testing.assert_allclose(out, np.asarray(_flows(inlet, thermo)), rtol=1e-10)
