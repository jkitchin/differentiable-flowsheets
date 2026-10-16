"""Physics and published-envelope checks of the Claus SRU and CCR (#472, #474).

Run from the repository root, by hand (about 15 minutes)::

    PYTHONPATH=src:. python -m pytest tests/reference/refinery_checks.py -v

Not part of the suite (the file name keeps pytest from collecting it): the
suite checks the code, not the physics, and each of these compiles or runs
another full solve. Sources are cited in tests/refinery/test_sulfur.py
([KN97], [AP42], [EB04]) and src/difflow_refinery/reforming/CLAUDE.md.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow import GibbsReactor, GibbsReactorParams, make_stream
from difflow_refinery import thermochemistry as tc
from difflow_refinery.reforming import lean_naphtha, rich_naphtha
from difflow_refinery.reforming.ccr import CCR_ENVELOPE, CCRReformer
from difflow_refinery.reforming.planning import CCR_OUTPUTS, ccr_block
from difflow_refinery.sulfur import claus as cl
from difflow_refinery.sulfur import vapor as v
from tests.refinery.test_sulfur import acid_gas, params

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- Claus
def gamson_elkins(sulfur):
    """Equilibrium conversion of stoichiometric H2S/SO2 in air, 1 atm, all vapour."""
    names = ["hydrogen_sulfide", "sulfur_dioxide", *sulfur, "water", "nitrogen"]
    th = tc.IdealGasSet(names, cp_basis="wide")
    n2 = 0.5 * (1 - cl.AIR_O2) / cl.AIR_O2
    feed = make_stream({"hydrogen_sulfide": 2 / 3, "sulfur_dioxide": 1 / 3, "water": 1 / 3,
                        "nitrogen": n2}, T=500.0, P=101325.0)
    Ts = np.arange(400.0, 1601.0, 25.0)
    X = []
    for T in Ts:
        o, i = GibbsReactor(GibbsReactorParams(thermo=th, T_out=T))(feed)
        assert bool(i["converged"])
        X.append(1.0 - float(o["F_hydrogen_sulfide"] + o["F_sulfur_dioxide"]))
    return Ts, np.array(X)


def test_gamson_elkins_curve():
    # [KN97]: minimum near 811 K, ~100% at 400 K; thermal region 60-75% [EB04]
    for S in (["sulfur_s2", "sulfur_s6", "sulfur_s8"], list(v.SULFUR_SPECIES)):
        Ts, X = gamson_elkins(S)
        assert 760.0 <= Ts[np.argmin(X)] <= 880.0
        assert X[0] > 0.99
        hot = X[(Ts >= 1300) & (Ts <= 1600)]
        assert np.all((hot > 0.60) & (hot < 0.78))


def test_recovery_vs_beds():
    rec = {n: float(jax.jit(lambda g, n=n: cl.solve_air(g, params(n)))(acid_gas())[1]["recovery"])
           for n in (2, 3, 4)}
    # equilibrium is an upper bound: at or above the AP-42 practical
    # ranges (2015 and 1995 tops: 95/96, 96/97.5, 97/98.5 %), and within ~3%
    for n, top in ((2, 0.96), (3, 0.975), (4, 0.985)):
        assert top - 0.005 < rec[n] < top + 0.03
    assert rec[2] < rec[3] < rec[4]
    assert rec[4] - rec[3] < 0.01                         # [KN97]: 4th bed < 1%


# ---------------------------------------------------------------- CCR
@pytest.fixture(scope="module")
def base():
    return CCRReformer().solve(lean_naphtha())


def test_another_feed_stays_in_the_coke_envelope(base):
    r = CCRReformer(base.params, base.ccr).solve(rich_naphtha())
    lo, hi = CCR_ENVELOPE["spent_coke_wt"]
    assert lo <= float(r.outputs()["ccr.spent_coke_wt"]) <= hi


def test_more_circulation_less_coke_more_octane(base):
    p, ccr = base.params, base.ccr
    r = CCRReformer(p, replace(ccr, circulation=2 * ccr.circulation)).solve(
        lean_naphtha(), tear_initial=base.tear)
    assert float(r.coke[-1]) < float(base.coke[-1])
    assert np.all(np.asarray(r.activity) > np.asarray(base.activity))
    assert float(r.outputs()["reformate.RON"]) > float(base.outputs()["reformate.RON"])


def test_ccr_block_reproduces_the_unit(base):
    blk = ccr_block(CCRReformer(base.params, base.ccr), lean_naphtha(),
                    levers=["wait", "circulation"])
    y = np.asarray(blk.fn(jnp.asarray(blk.u0)))
    o = base.outputs()
    np.testing.assert_allclose(y, [float(o[k]) for k in CCR_OUTPUTS], rtol=1e-6)
