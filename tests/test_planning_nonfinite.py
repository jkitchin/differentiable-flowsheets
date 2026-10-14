"""A block that cannot be evaluated (returns NaN) is never a plan."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.planning import Block, DeltaBasePlanner, Network, state_is_finite
from difflow.planning.lp import Spec
from difflow.planning.planner import TrustRegionOptions

#: The block converges for x <= EDGE and "fails" beyond it.
EDGE = 1.0


def cliff_block(nan_outputs=("out", "q")):
    """One lever; the named outputs are NaN past ``EDGE``, as a failed solve's are."""

    names = ["out", "q"]

    def fn(u):
        x = u[0]
        fail = x > EDGE
        vals = {"out": x, "q": x}
        return jnp.stack([jnp.where(fail & (n in nan_outputs), jnp.nan, vals[n])
                          for n in names])

    return Block(name="plant", fn=fn, u_names=["x"], y_names=names,
                 lb=[0.0], ub=[2.0], u0=[0.5])


def _planner(block, prices, specs=(), **kw):
    opts = TrustRegionOptions(radius=0.3, radius_min=1e-4, max_iter=60)
    return DeltaBasePlanner(Network([block]), prices=prices, specs=specs,
                            options=opts, vertex_seeding=False, **kw)


def test_state_is_finite():
    net = Network([cliff_block()])
    assert state_is_finite(net.evaluate([0.5]))
    assert not state_is_finite(net.evaluate([1.5]))


def test_a_nan_spec_output_scores_as_unevaluable():
    """Regression: Python's max(0.0, nan) would read a NaN spec as satisfied."""
    spec = Spec("plant.q", ">=", 1.5)
    scored = _planner(cliff_block(), {"plant.out": 1.0}, [spec]).score([1.6])
    assert scored["evaluable"] is False
    assert scored["total_violation"] == np.inf


def test_an_unpriced_nan_is_still_rejected():
    """Regression: merit is finite (only ``q`` fails) and the point is still refused."""
    res = _planner(cliff_block(nan_outputs=("q",)), {"plant.out": 1.0}).solve()
    assert res.plan["plant.x"] <= EDGE
    assert state_is_finite(res.state)


def test_restoration_keeps_an_evaluable_point():
    """Regression: restoration must not take a failed solve for a feasible point."""
    spec = Spec("plant.q", ">=", 1.5, elastic=False)
    res = _planner(cliff_block(), {"plant.out": 1.0}, [spec]).solve()
    assert state_is_finite(res.state)


def test_an_unevaluable_start_raises():
    with pytest.raises(ValueError, match="no starting point could be evaluated"):
        _planner(cliff_block(), {"plant.out": 1.0}).solve(u0=[1.5])
