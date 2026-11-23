"""Delayed coker (#534), checked by hand.

    PYTHONPATH=src:. python -m pytest tests/reference/coker_check.py -v -s

1. Balances (mass, S, N; Ni+V all to coke) close over a range of feeds.
2. Coke yield vs CCR over 5-25 wt% follows the correlation the unit
   implements (Gary, Handwerk & Kaiser, 5th ed., ch. 5: 1.6 x CCR). This is
   a check of the transcription, not an independent validation.
3. AD gradients match central differences.
4. Against the two delayed cokers of Du et al. (2025) Case 2 (``Upd0``,
   ``Upd1``; paper Table 5): base yields at the published base feed, and the
   AD sensitivities with respect to feed SG, sulfur and CCR next to the
   published ``delta / Del``. Sign agreement and ratio are printed, not
   asserted. Needs the benchmark data (``du2025.fetch(cases=(2,))``).

No published coker yield data for named residues are reproduced: none were
at hand. #534 asks for two; that remains open.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_refinery.benchmarks import du2025
from difflow_refinery.coking import PRODUCTS, CokerFeed, DelayedCoker

jax.config.update("jax_enable_x64", True)

#: Case 2 coker outlets by product (identified from the published delta:
#: coke +0.0156 / wt% CCR, H2S +0.0035 / wt% S; routing in the case files).
CASE2_OUTLETS = {
    "Upd0": {"dry_gas": "s271", "lpg": "s273", "h2s": "s274", "naphtha": "s276",
             "diesel": "s275", "gas_oil": "s277", "coke": "s272"},
    "Upd1": {"dry_gas": "s279", "lpg": "s281", "naphtha": "s282", "diesel": "s278",
             "gas_oil": "s285", "coke": "s280"},
}
#: Case 2 property indices of the coker feed (Case 2 is anonymized afresh).
CASE2_PROPS = {"SG": "Q0", "S": "Q1", "CCR": "Q22"}


def test_balances_close():
    unit = DelayedCoker()
    for sg, s, ccr in ((0.95, 0.5, 5.0), (1.01, 3.0, 19.5), (1.06, 5.0, 25.0)):
        o = unit.solve(CokerFeed.from_properties(1.0, sg=sg, sulfur_wt=s, ccr_wt=ccr,
                                                 nitrogen_wppm=5000.0, nickel_vanadium_wppm=300.0))
        for k in ("balance.mass", "balance.S", "balance.N"):
            assert abs(float(o[k])) < 1e-12, k
        assert float(o["coke.NiV_wppm"]) * float(o["yield.coke"]) == pytest.approx(300.0)


def test_coke_trend_follows_correlation():
    unit = DelayedCoker()
    for ccr in np.linspace(5.0, 25.0, 9):
        o = unit.solve(CokerFeed.from_properties(1.0, sg=1.0, sulfur_wt=2.0, ccr_wt=ccr))
        assert float(o["yield.coke"]) == pytest.approx(1.6 * ccr / 100.0, rel=1e-12)


def test_gradients_fd():
    unit = DelayedCoker()

    def f(x):
        feed = CokerFeed.from_properties(1.0, sg=x[0], sulfur_wt=x[1], ccr_wt=x[2])
        return jnp.concatenate([unit.yields(feed, drum_P=x[3], recycle=x[4]),
                                jnp.stack([unit.solve(feed)["naphtha.SG"]])])

    x0 = jnp.array([1.01, 3.0, 19.5, 1.5, 0.1])
    J = np.asarray(jax.jacfwd(f)(x0))
    for i, h in enumerate((1e-5, 1e-4, 1e-3, 1e-4, 1e-5)):
        e = jnp.zeros(5).at[i].set(h)
        fd = (np.asarray(f(x0 + e)) - np.asarray(f(x0 - e))) / (2 * h)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-6, atol=1e-10)


CASE2 = du2025.find_case(2)


@pytest.mark.skipif(CASE2 is None, reason="Case 2 data not fetched")
@pytest.mark.parametrize("u", ["Upd0", "Upd1"])
def test_against_du2025_case2(u):
    from difflow_refinery.benchmarks.du2025.calibration import unit_base
    data = du2025.load_case(CASE2)
    b = unit_base(data, u)
    outlets = CASE2_OUTLETS[u]
    B = {k: b.props[q] for k, q in CASE2_PROPS.items()}
    unit = DelayedCoker()

    def y(x):  # (SG, S wt%, CCR wt%) -> yields on feed, PRODUCTS order
        return unit.yields(CokerFeed.from_properties(1.0, sg=x[0], sulfur_wt=x[1], ccr_wt=x[2]))

    x0 = jnp.array([B["SG"], B["S"], B["CCR"]])
    ours = dict(zip(PRODUCTS, np.asarray(y(x0))))
    J = dict(zip(PRODUCTS, np.asarray(jax.jacfwd(y)(x0))))
    pub = dict(zip(b.outlets, b.yields))
    print(f"\n{u}: base feed SG {B['SG']}, S {B['S']} wt%, CCR {B['CCR']} wt%")
    print(f"{'product':8s} {'ours':>7s} {'pub':>7s} | {'dSG ours':>9s} {'pub':>9s} | "
          f"{'dS ours':>9s} {'pub':>9s} | {'dCCR ours':>9s} {'pub':>9s}")
    agree = n = 0
    for k in PRODUCTS:
        s = outlets.get(k)
        if s is None:
            continue
        d_pub = [b.delta[CASE2_PROPS[p]][b.outlets.index(s)] for p in ("SG", "S", "CCR")]
        d_ours = J[k]
        for a, c in zip(d_ours, d_pub):
            if c != 0 and a != 0:
                n += 1
                agree += np.sign(a) == np.sign(c)
        print(f"{k:8s} {ours[k]:7.4f} {pub[s]:7.4f} | " + " | ".join(
            f"{a:9.5f} {c:9.5f}" for a, c in zip(d_ours, d_pub)))
    print(f"sign agreement where both nonzero: {agree}/{n}")
