"""The header/hydrotreater loop (#329), on stand-in hydrotreaters.

``close_hydrotreater_loop`` is Python substitution on the header purity: it
only reads ``unit.layout.gases``, calls ``unit.solve(feed, params=, warn=)``
and uses the result's ``converged`` and ``outputs["h2.makeup"]``. The
hydrotreater's own physics (a compile of a minute or more each) is tested in
``test_hydrotreating*.py``; here each unit is a stand-in whose makeup demand
falls with the purity it is given, so the loop's own claims are checked in
milliseconds: it feeds the header's composition into each unit, iterates while
an import swing makes the purity depend on the demands, records the secant
response, and the header balances the demands the units report.

The gradient is the surplus with respect to the producer's rate through the
closed network (the units' response is the loop's linear secant), against
central differences.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery.hydrogen as h2
from difflow_refinery.hydrotreating import HydrotreaterParams

jax.config.update("jax_enable_x64", True)

GASES = ("hydrogen", "methane", "ethane", "propane")
IMP = {"methane": 0.6, "ethane": 0.3, "propane": 0.1}


class _StandIn:
    """A hydrotreater whose makeup H2 demand is ``d0 * (1 + k (0.97 - y))`` at makeup purity ``y``."""

    layout = SimpleNamespace(gases=GASES)

    def __init__(self, d0, k):
        self.d0, self.k = d0, k
        self.calls = 0

    def solve(self, feed, params, warn=True):
        self.calls += 1
        y = params.makeup["hydrogen"]
        d = self.d0 * (1.0 + self.k * (0.97 - y))
        return SimpleNamespace(converged=True, outputs={"h2.makeup": jnp.asarray(d)})


def _units():
    return {"kht": (_StandIn(20.0, 4.0), None, HydrotreaterParams(P=35e5)),
            "dht": (_StandIn(60.0, 6.0), None, HydrotreaterParams(P=60e5))}


def _network(units, producer, swing):
    cons = [h2.Consumer(n, 20.0, P=p.P, min_purity=0.85) for n, (_, _, p) in units.items()]
    return h2.HydrogenNetwork(producers=[producer], consumers=cons,
                              headers=[h2.Header("main", P=10e5, min_purge=1.0, swing=swing)])


@pytest.fixture(scope="module")
def loop():
    units = _units()
    prod = h2.Producer.of_purity("reformer", 45.0, 0.88, impurity=IMP)
    net = _network(units, prod, [h2.Import(purity=0.999, P=30e5)])
    return units, h2.close_hydrotreater_loop(net, units, tol=1e-9)


def test_the_loop_closes_on_the_header_purity(loop):
    units, res = loop
    assert res.converged and res.passes >= 2   # the import swing: purity moves with the demands
    for k, v in res.header.balances.items():
        assert float(v) < 1e-12, k
    o = res.header.outputs
    y = float(o["main.purity"])
    assert 0.88 < y < 0.999   # a mix of the producer gas and 99.9 % import
    for name, r in res.units.items():
        # the unit was solved at the purity the header delivers it ...
        assert res.params[name].makeup["hydrogen"] == pytest.approx(y, abs=1e-8)
        # ... and the header balances the demand the unit reports at that purity
        assert float(o[f"{name}.makeup_h2"]) == pytest.approx(float(r.outputs["h2.makeup"]), rel=1e-8)
        assert float(o[f"{name}.purity_margin"]) == pytest.approx(y - 0.85)
    supply = 45.0 + float(o["swing.h2"])
    demand = sum(float(r.outputs["h2.makeup"]) for r in res.units.values())
    assert float(o["h2.surplus"]) == pytest.approx(supply - demand, rel=1e-6, abs=1e-6)
    assert float(o["h2.surplus"]) >= 1.0 - 1e-9
    # a leaner makeup costs makeup: the secant is the stand-in's slope
    for c in res.network.consumers:
        u = units[c.name][0]
        assert float(c.d_demand_d_purity) == pytest.approx(-u.d0 * u.k, rel=1e-6)


def test_a_fixed_supply_closes_in_one_pass():
    units = _units()
    net = _network(units, h2.Producer.of_purity("reformer", 200.0, 0.9, impurity=IMP), [])
    res = h2.close_hydrotreater_loop(net, units, tol=1e-9)
    assert res.converged and res.passes == 1
    assert all(u.calls == 1 for u, _, _ in units.values())


def test_surplus_gradient_wrt_the_producer_rate_matches_finite_differences(loop):
    _, res = loop
    net = res.network

    def surplus(F):
        p = h2.Producer.of_purity("reformer", F, 0.88, impurity=IMP)
        o = net.replace(producers=[p]).solve().outputs
        return jnp.stack([o["h2.surplus"], o["main.purity"]])

    F0 = jnp.asarray(45.0)
    J = jax.jacfwd(surplus)(F0)
    h = 1e-3
    fd = (surplus(F0 + h) - surplus(F0 - h)) / (2 * h)
    np.testing.assert_allclose(np.asarray(J), np.asarray(fd), rtol=1e-5, atol=1e-9)
    assert float(J[1]) != 0.0
