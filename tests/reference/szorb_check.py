"""S Zorb desulfurizer (#535), checked by hand.

    PYTHONPATH=src:. python -m pytest tests/reference/szorb_check.py -v -s

1. S, H2 and mass balances close over a range of feeds and severities.
2. Monotone trends: product S falls and RON loss rises with severity
   (lower WHSV, higher T); RON loss is proportional to olefins saturated.
3. AD gradients match central differences (through the sorbent-loading
   root solve, differentiated implicitly).
4. Against Du et al. (2025) Case 1 ``Upd4``: the published delta moves the
   sour-gas outlet ``s260`` by +0.0104 and the fuel-gas outlet ``s73`` by
   -0.0104 per wt% feed S; the AD sensitivity of our captured sulfur (as
   H2S) is printed beside it. Needs the benchmark data.

Not done: reproducing published S Zorb plant data (#535 asks for one case);
none was at hand. The constants are ILLUSTRATIVE.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.benchmarks import du2025
from difflow_refinery.desulfurization import SZorbFeed, SZorbUnit

jax.config.update("jax_enable_x64", True)


def _feed(s=500.0, olefins=30.0):
    return SZorbFeed.from_properties(50.0, sg=0.73, sulfur_wppm=s, olefins_vol=olefins,
                                     aromatics_vol=22.0, ron=92.0, mon=80.0)


def test_balances():
    u = SZorbUnit()
    for s in (100.0, 500.0, 1500.0):
        for T in (673.15, 713.15):
            o = u.solve(_feed(s), T=T)
            for k in ("balance.mass", "balance.S", "balance.H2"):
                assert abs(float(o[k])) < 1e-12, (k, s, T)
            assert float(o["residual"]) < 1e-12


def test_trends():
    u = SZorbUnit()
    S, loss = [], []
    for whsv in (8.0, 6.0, 4.0, 3.0, 2.0):            # rising severity
        o = u.solve(_feed(), whsv=whsv)
        S.append(float(o["product.S_wppm"]))
        loss.append(float(o["ron.loss"]))
    assert np.all(np.diff(S) < 0) and np.all(np.diff(loss) > 0)
    T_S = [float(u.solve(_feed(), T=T)["product.S_wppm"]) for T in (673.15, 693.15, 713.15)]
    assert np.all(np.diff(T_S) < 0)
    a = u.solve(_feed(olefins=20.0))
    b = u.solve(_feed(olefins=40.0))
    assert float(b["ron.loss"]) == pytest.approx(2.0 * float(a["ron.loss"]))
    print("\nproduct S vs WHSV 8..2:", np.round(S, 2), " RON loss:", np.round(loss, 3))


def test_gradients_fd():
    u = SZorbUnit()

    def f(x):
        o = u.solve(_feed(x[0]), T=x[1], whsv=x[2], circulation=x[3])
        return jnp.stack([o["product.S_wppm"], o["ron.loss"], o["h2.consumption_wt"],
                          o["sorbent.loading"]])

    x0 = jnp.array([500.0, 693.15, 4.0, 0.025])
    J = np.asarray(jax.jacfwd(f)(x0))
    for i, h in enumerate((1e-2, 1e-3, 1e-5, 1e-7)):
        e = jnp.zeros(4).at[i].set(h)
        fd = (np.asarray(f(x0 + e)) - np.asarray(f(x0 - e))) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-5, atol=1e-9)


CASE = du2025.find_case(1)


@pytest.mark.skipif(CASE is None, reason="benchmark data not fetched")
def test_against_du2025_upd4():
    from difflow_refinery.benchmarks.du2025.calibration import unit_base
    data = du2025.load_case(CASE)
    b = unit_base(data, "Upd4")
    S0 = b.props["Q0"]                                     # wt%
    u = SZorbUnit()
    f = lambda s: u.solve(_feed(1e4 * s))["acid_gas.yield"]   # noqa: E731  s in wt%
    d = float(jax.grad(f)(S0))
    pub = {s: b.delta["Q0"][b.outlets.index(s)] for s in ("s260", "s73")}
    print(f"\nUpd4 base S {S0} wt%: d(acid gas)/dS ours {d:+.5f}, "
          f"published sour gas s260 {pub['s260']:+.5f}, fuel gas s73 {pub['s73']:+.5f} per wt% S")
    print(f"  base yields: ours acid gas {float(f(S0)):.5f}; published s260 "
          f"{b.yields[b.outlets.index('s260')]:.4f}")
    assert np.sign(d) == np.sign(pub["s260"])
