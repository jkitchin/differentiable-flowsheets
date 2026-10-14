"""Tests for the furnace solved with the crude column.

The furnace's duty is the enthalpy rise to the coil outlet, and an overflash
spec sets the coil outlet temperature (with its gradient). Not run here, for
cost: the coil-outlet-temperature and furnace-duty specs reproducing a fixed
feed (the former is what ``test_operation.py`` runs), vmap over the inlet
temperature, and the 30-stage atmospheric column (the pumparound that takes
more heat than the feed brings, so the column has no reflux and does not
converge, is in ``test_planning_nonconvergence.py``).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery import Assay, characterize
from difflow_refinery import column as cc
from difflow_refinery.thermo import ColumnThermo


jax.config.update("jax_enable_x64", True)

NAPHTHA = cc.product_rate("naphtha", 0.25, basis="mole")


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


@pytest.fixture(scope="module")
def cold(small):
    crude, _ = small
    return crude.stream(1.0, T=480.0, P=4e5, basis="mole")


class TestFurnaceSpecs:
    def test_overflash_sets_the_furnace_duty_and_its_gradient(self, small, cold):
        """An overflash spec closes the column and the furnace: the duty is the
        enthalpy rise of the feed to the coil outlet it finds, and the coil
        outlet's gradient in the overflash (implicit) matches central
        differences. One compile gives the values and the forward slope."""
        _, th = small

        @jax.jit
        def solve_and_slope(of):
            def f(o):
                r = _small(th, (NAPHTHA, cc.overflash(o))).solve(cold)
                return jnp.stack([r.coil_outlet_T, r.furnace_duty, r.converged.astype(float)])
            return jax.jvp(f, (of,), (jnp.ones_like(of),))

        (T0, Q0, ok), (g, _, _) = solve_and_slope(0.05)
        assert float(ok) == 1.0
        col = _small(th, (NAPHTHA, cc.overflash(0.05)))
        args = col._args(cold)
        dh = col._feed_enthalpy(args, T0, args["P_F"])[0] - col._feed_enthalpy(args, 480.0, 4e5)[0]
        assert float(Q0) == pytest.approx(float(dh), rel=1e-10)
        # the default outlet pressure is the feed stage's
        assert float(args["P_F"]) == pytest.approx(1.4e5 + (1.6e5 - 1.4e5) * 6 / 7)

        h = 1e-5
        fd = float((solve_and_slope(0.05 + h)[0][0] - solve_and_slope(0.05 - h)[0][0]) / (2 * h))
        assert float(g) > 0  # more wash liquid needs a hotter furnace
        assert float(g) == pytest.approx(fd, rel=1e-5)


class TestConfiguration:
    def test_furnace_adds_a_freedom(self, small):
        _, th = small
        with pytest.raises(ValueError, match="1 furnace"):
            _small(th, (cc.product_rate("naphtha", 0.25, basis="mole"),))

    def test_furnace_spec_needs_a_furnace(self, small):
        _, th = small
        with pytest.raises(ValueError, match="needs a Furnace"):
            _small(th, (cc.coil_outlet_temperature(620.0),), furnace=None)
