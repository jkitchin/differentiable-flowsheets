"""The hydrotreater as a naphtha hydrotreater (#332, #333): example 40's naphtha.

The crude, cut grid and stabilised naphtha are those of
``examples/40_refinery_flowsheet.ipynb`` (the gas plant's debutanizer
bottoms, written out here so the test needs no crude unit or gas plant).
One compile serves every test in the file: they vary only numeric specs.
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.hydrotreating import (
    NAPHTHA_HDT_PARAMS, Hydrotreater, HydrotreaterConvergenceWarning, HydrotreaterParams)

jax.config.update("jax_enable_x64", True)

C = 273.15
PCT = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
T_C = [-10, 60, 95, 150, 205, 260, 315, 370, 430, 500, 600, 680, 850]

#: Example 40's stabilised naphtha (mol/s), the debutanizer bottoms; light ends
#: C2-C5 then the first four 30 C cuts.
NAPHTHA = {"F_ethane": 6.76893e-08, "F_propane": 0.000630421, "F_isobutane": 0.0757744, "F_n_butane": 2.34864,
           "F_isopentane": 11.8908, "F_n_pentane": 22.7876, "F_pc01": 74.3897, "F_pc02": 63.5167,
           "F_pc03": 52.5016, "F_pc04": 14.9299, "T": 313.15, "P": 3e5}

#: A naphtha unit at credible conditions, on the naphtha constants; the feed
#: enters the charge heater at 250 C after a feed/effluent exchanger.
NHT = HydrotreaterParams(T_in=(C + 320.0,), quench=None, bed_fractions=(1.0,), P=30e5, lhsv=4.0, h2_oil=100.0,
                         stripper_feed_T=C + 150.0, kinetics=NAPHTHA_HDT_PARAMS, heater_inlet_T=C + 250.0)


@pytest.fixture(scope="module")
def naphtha_unit():
    assay = dr.Assay(PCT, [t + C for t in T_C], sg=0.86,
                     light_ends={"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
                                 "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5},
                     sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=5.0)
    cut_points = dr.default_cut_points(assay, ((673.15, 30.0), (873.15, 60.0), (np.inf, 150.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        char = dr.characterize(assay, cut_points, composition=True)
    return Hydrotreater(char, NAPHTHA, NHT)


@pytest.fixture(scope="module")
def naphtha(naphtha_unit):
    return naphtha_unit, naphtha_unit.solve(NAPHTHA)


#: Bed-inlet flashing-component flows (mol/s, ``comps.names`` order: H2, H2S, NH3,
#: C1-C5, pc01-pc04) of the unit at 320 C / 50 bar / LHSV 0.5 / 150 Nm3/m3,
#: where the PR flash at 593.15 K and 50 bar does not converge.
NEAR_CRITICAL = [215.37, 0.00280338, 0.0, 6.25626, 0.0111435, 0.0100384, 0.0867114, 2.56126, 12.373, 23.5171,
                 74.3897, 63.5167, 52.5016, 14.9299]


def test_a_failed_bed_inlet_flash_reports_a_residual_instead_of_raising(naphtha_unit):
    """#332: the flash (and its T-derivative, as the reactor takes it) at that point returns, flagged.

    Before, Newton diverged to NaN and optimistix's implicit JVP -- the
    reactor's d ln K / dT -- raised equinox's NaN-in-linear-solve error.
    """
    import jax.numpy as jnp
    from difflow_refinery.hydroprocessing.separator import pr_flash
    from difflow_refinery.hydrotreating.unit import FLASH_TOL
    unit = naphtha_unit
    comps = unit._comps(unit.theta(NAPHTHA))
    z = jnp.asarray(NEAR_CRITICAL)

    @jax.jit
    def lnK_and_slope(T):
        return jax.jvp(lambda t: (pr_flash(t, 50e5, z, comps).lnK, pr_flash(t, 50e5, z, comps).residual),
                       (T,), (jnp.asarray(1.0),))

    (lnK, res), (slope, _) = lnK_and_slope(jnp.asarray(593.15))
    assert not float(res) < FLASH_TOL          # flagged (large or NaN), not raised
    assert np.all(np.isfinite(np.asarray(lnK)))
    # and an ordinary point next to it still converges
    (lnK, res), _ = lnK_and_slope(jnp.asarray(583.15))
    assert float(res) < FLASH_TOL


@pytest.mark.slow
def test_naphtha_unit_converges_balances_and_reports(naphtha):
    """Balances, the gas yield on a feed carrying C3-C5 light ends, the charge heater, the products."""
    unit, r = naphtha
    assert bool(r.converged)
    for k, v in r.balances.items():
        assert float(v) < 1e-10, k
    o = r.outputs
    # gas.yield (#332): was -10 % on this feed (it subtracted the feed's C5s, which leave as liquid)
    assert 0.0 < float(o["gas.yield"]) < 0.02
    f = float(o["feed.rate"])
    rhs = 1.0 - float(o["feed.h2_water_rate"]) / f + float(o["h2.consumed_by_balance"]) * 2.01588e-3 / f
    assert float(o["yields.total"]) == pytest.approx(rhs, rel=1e-9)
    # charge heater after a feed/effluent exchanger: both duties positive, fired = absorbed / efficiency
    assert float(o["heater.inlet_T"]) == pytest.approx(C + 250.0)
    assert float(o["heater.duty"]) > 0.0 and float(o["feed_effluent.duty"]) > 0.0
    assert float(o["heater.fired_duty"]) == pytest.approx(float(o["heater.duty"]) / NHT.heater_efficiency,
                                                          rel=1e-14)
    # a vapour-phase bed hydrogenates (#332: the old phase model ran the aromatics equilibrium backwards)
    assert float(o["h2.chemical"]) > 0.0 and float(o["reactor.dT_total"]) > 0.0
    # #333: product + wild naphtha as streams carry the whole liquid mass, dissolved gases included
    lay = unit.layout
    for name in ("product", "wild_naphtha"):
        assert float(r.stream_mass(r.product_stream(name))) == pytest.approx(float(r.streams[name].mass(lay)),
                                                                             rel=1e-13)
    # #328 on a naphtha unit: light / heavy naphtha at 85 C over product + wild naphtha
    fr = r.fractionate(cut_points=(C + 85.0,), products=("light_naphtha", "heavy_naphtha"),
                       feeds=("product", "wild_naphtha"))
    assert float(fr.balance) < 1e-13
    assert float(fr.rates["light_naphtha"]) > 0.0 and float(fr.rates["heavy_naphtha"]) > 0.0


@pytest.mark.slow
@pytest.mark.release   # per commit, the flash-level test above covers the code path
def test_a_failed_flash_is_reported_not_raised(naphtha):
    """#332: 320 C / 50 bar / LHSV 0.5 / 150 Nm3/m3 raised an equinox NaN-in-linear-solve error.

    The bed-inlet PR flash fails there (near the mixture's critical region,
    Newton diverges); the unit now returns ``converged=False`` and warns.
    """
    unit, _ = naphtha
    p = dataclasses.replace(NHT, P=50e5, lhsv=0.5, h2_oil=150.0)
    with pytest.warns(HydrotreaterConvergenceWarning):
        r = unit.solve(NAPHTHA, params=p)
    assert not bool(r.converged)
    assert float(r.outputs["flash.residual"]) > 1e-8 or not np.isfinite(float(r.outputs["flash.residual"]))


@pytest.mark.slow
@pytest.mark.release
def test_naphtha_constants_reach_reformer_feed_sulfur(naphtha):
    """#332: with NAPHTHA_HDT_PARAMS a 30 bar naphtha unit reaches reformer-feed sulfur (< 0.5 wppm)."""
    unit, r = naphtha
    assert float(r.outputs["product.S_wppm"]) < 0.5
    S = [float(unit.solve(NAPHTHA, params=dataclasses.replace(NHT, T_in=(C + T,)), warn=False)
               .outputs["product.S_wppm"]) for T in (300.0, 310.0)]
    assert S[0] > S[1] > float(r.outputs["product.S_wppm"])
