"""Tests for the hydrogen-purification PSA model (#469): difflow_refinery.hydrogen.psa.

Smoke, balance, validation, API and gradient-finite tests only (the test
policy in CLAUDE.md). The comparisons with Zhang et al. (2025), the exact
linear-LDF breakthrough and the published trade-offs are not per-commit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.hydrogen import psa
from difflow_refinery.hydrogen.network import PSA

jax.config.update("jax_enable_x64", True)

F = 1.24e-2           # mol/s, Zhang et al. (2025): 1.24e-5 kmol/s
ZHANG_FEED = {"hydrogen": 0.76 * F, "methane": 0.03 * F, "carbon_monoxide": 0.04 * F,
              "carbon_dioxide": 0.17 * F}


def run(feed=ZHANG_FEED, **kw):
    return psa.HydrogenPSA(psa.HydrogenPSAParams(**kw))(feed)


def test_isotherm_smoke():
    y = jnp.array([0.76, 0.03, 0.04, 0.17]) * 12.0
    for ads in psa.DSL_PARAMETERS:
        q = psa.dsl_loading(ads, y, 298.15)
        assert q.shape == (4,) and np.all(np.isfinite(np.asarray(q))) and np.all(np.asarray(q) > 0)


def test_klinkenberg_zero_before_arrival():
    assert float(psa.klinkenberg(jnp.asarray(10.0), jnp.asarray(-1.0))) == 0.0


@pytest.mark.parametrize("kw", [{}, dict(n_equalizations=0), dict(n_beds_on_feed=2)])
def test_balances_close(kw):
    feed = {**ZHANG_FEED, "nitrogen": 1e-4, "ethane": 2e-4, "water": 1e-5}
    out = run(feed, **kw)
    for s, b in out["balance"].items():
        assert abs(float(b)) < 1e-15 + 1e-12 * float(feed[s])
    prod = out["product"]
    assert float(prod["ethane"]) == 0.0 and float(prod["water"]) == 0.0
    assert float(prod["nitrogen"]) == pytest.approx(1e-4)
    purity = prod["hydrogen"] / sum(prod.values())
    assert float(out["purity"]) == pytest.approx(float(purity), rel=1e-14)
    assert float(out["recovery"]) == pytest.approx(
        float(prod["hydrogen"] / feed["hydrogen"]), rel=1e-12)


def test_unknown_species_raises():
    with pytest.raises(ValueError, match="no rule"):
        run({**ZHANG_FEED, "ammonia": 1e-5})


def test_from_model_bridge():
    out = run()
    s = PSA.from_model(out)
    assert s.recovery == pytest.approx(float(out["recovery"]))
    assert s.purity == pytest.approx(float(out["purity"]))
    t = PSA.from_model(out, target_purity=0.999)
    assert t.target_purity == 0.999


def test_grads_finite():
    def f(x):
        p = psa.HydrogenPSAParams(t_ads=x[0], purge_to_feed=x[1], P_ads=x[2], T=x[3])
        o = psa.HydrogenPSA(p)({**ZHANG_FEED, "hydrogen": x[4]})
        return jnp.stack([o["purity"], o["recovery"], o["tail_gas"]["hydrogen"]])
    x = jnp.array([200.0, 0.11, 12.0e5, 298.15, ZHANG_FEED["hydrogen"]])
    for J in (jax.jacfwd(f)(x), jax.jacrev(f)(x)):
        assert np.all(np.isfinite(np.asarray(J)))
