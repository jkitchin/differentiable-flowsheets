"""A gradient across two library units, against central differences (#334).

Example 40's stabilised naphtha through the naphtha hydrotreater, its
fractionator and ``NaphthaFeed.from_hydrotreater`` into the catalytic
reformer, composed with :class:`difflow_refinery.plant.Chain`:

* with the hydrotreater's beds on ``ReactorOptions(adjoint="forward")``
  every stage is forward-capable and the chain is one ``jax.jacfwd``;
* with its default (checkpointed, reverse-only) beds the chain is mixed, and
  ``method="chain"`` -- the chain rule by unit Jacobians -- gives the same
  Jacobian.

Both are checked against central differences of the whole chain. Release
(they check numerics) and slow (each compiles the hydrotreater and a
derivative of the reformer: several minutes).
"""

from __future__ import annotations

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import difflow_refinery as dr
from difflow_refinery.hydroprocessing.reactor import ReactorOptions
from difflow_refinery.hydrotreating import Hydrotreater
from difflow_refinery.plant import Chain, Stage, central_difference
from difflow_refinery.reforming import CatalyticReformer, NaphthaFeed, ReformerParams

from .test_hydrotreating_naphtha import NAPHTHA, NHT, PCT, T_C

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.release, pytest.mark.slow]

C = 273.15
#: Bed inlet 300 C (product S ~1 wppm, so the reformer's sulfur balance has
#: something to carry) and the light/heavy cut at 85 C.
X0 = jnp.asarray([C + 300.0, C + 85.0])
H = [0.5, 0.5]
OUT = ("reformate.S_wppm", "reformate.RON", "h2.net_mol_s", "reformate.yield_vol")


@pytest.fixture(scope="module")
def char():
    assay = dr.Assay(PCT, [t + C for t in T_C], sg=0.86,
                     light_ends={"ethane": 0.05, "propane": 0.5, "isobutane": 0.3,
                                 "n_butane": 1.0, "isopentane": 0.8, "n_pentane": 1.5},
                     sulfur_wt=1.8, nitrogen_wppm=1500.0, ccr_wt=5.0)
    cut_points = dr.default_cut_points(assay, ((673.15, 30.0), (873.15, 60.0), (np.inf, 150.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dr.characterize(assay, cut_points, composition=True)


def _chain(char, adjoint):
    params = dataclasses.replace(NHT, reactor=ReactorOptions(adjoint=adjoint))
    unit = Hydrotreater(char, NAPHTHA, params)
    reformer = CatalyticReformer(ReformerParams())

    def nht(x):
        r = unit.solve(NAPHTHA, params=dataclasses.replace(params, T_in=(x[0],)), warn=False)
        f = r.fractionate(cut_points=(x[1],), products=("light_naphtha", "heavy_naphtha"),
                          feeds=("product", "wild_naphtha"))
        return NaphthaFeed.from_hydrotreater(f, "heavy_naphtha")

    tear = reformer.solve(nht(X0)).tear

    def reform(feed):
        o = reformer.solve(feed, tear_initial=tear, tol=1e-11, max_iter=400, on_nonconvergence="ignore").outputs()
        return jnp.stack([o[k] for k in OUT])

    return Chain(Stage("nht", nht, modes="fwd" if adjoint == "forward" else "rev"),
                 Stage("reformer", reform, modes="fwd"))


@pytest.fixture(scope="module")
def forward(char):
    chain = _chain(char, "forward")
    return chain, chain.jacobian(X0), central_difference(chain, X0, H)


def test_an_all_forward_chain_is_one_jacfwd_and_matches_central_differences(forward):
    chain, res, fd = forward
    assert chain.modes == ("fwd",) and res.method == "fwd"
    J = np.asarray(res.jacobian)
    assert np.all(np.isfinite(J))
    # the reformate's sulfur falls as the hydrotreater runs hotter, and a lighter cut point
    # sends more (lighter) naphtha to the reformer
    assert J[0, 0] < 0.0
    np.testing.assert_allclose(J, fd, rtol=2e-3, atol=1e-9 * np.max(np.abs(fd), axis=1, keepdims=True).max())


def test_a_mixed_chain_by_unit_jacobians_gives_the_same_jacobian(char, forward):
    _, ref, _ = forward
    chain = _chain(char, "checkpoint")
    assert chain.modes == ()
    res = chain.jacobian(X0)
    assert res.method == "chain" and set(res.timings) == {"nht", "reformer", "total"}
    np.testing.assert_allclose(np.asarray(res.value), np.asarray(ref.value), rtol=1e-9)
    np.testing.assert_allclose(np.asarray(res.jacobian), np.asarray(ref.jacobian), rtol=1e-6, atol=1e-12)
