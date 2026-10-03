"""Implicit gradients through the DIH recycle against central differences (#311).

``release`` and ``slow``: one ``jax.jacfwd`` through the converged recycle
(the tear Jacobian at the solution, by forward mode, then the implicit
function theorem in :meth:`IsomerizationUnit.outputs`) and six warm-started
re-solves for the differences -- about eight minutes. The paraffinic feed
only, to keep this file inside a CI shard; the benzene-rich feed was run
the same way when the unit was built and agreed to 4.4e-6 relative on every
entry (the PR for #311 lists the numbers).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from difflow_refinery.isomerization import (
    OUTPUT_NAMES,
    IsomerizationUnit,
    IsomerizationUnitParams,
    constructed_feed,
    nc6_fraction,
)

jax.config.update("jax_enable_x64", True)

pytestmark = [pytest.mark.release, pytest.mark.slow]

OUTPUTS = ("RON", "yield_volume", "dih_duty", "MON", "H2_makeup")
#: Central-difference steps: T_in (K), LHSV (1/h), x_nC6 (-).
STEPS = (0.05, 2e-3, 1e-4)


@pytest.fixture(scope="module")
def jacobian():
    unit = IsomerizationUnit(IsomerizationUnitParams(configuration="dih"))
    feed = constructed_feed("paraffinic", 10.0)
    base = jnp.array([unit.params.T_in, unit.params.reactor.LHSV, float(nc6_fraction(feed))])
    J = jax.jacfwd(lambda v: unit.outputs(feed, v[0], v[1], x_nc6=v[2]))(base)
    guess = unit.last_solve["recycle"]
    return unit, feed, base, J, guess


@pytest.mark.parametrize("j,lever", enumerate(("T_in", "LHSV", "x_nC6")))
def test_the_jacobian_matches_central_differences(jacobian, j, lever):
    unit, feed, base, J, guess = jacobian
    e = jnp.zeros(3).at[j].set(STEPS[j])

    def f(v):
        return unit.outputs(feed, v[0], v[1], x_nc6=v[2], recycle_guess=guess)

    fd = (f(base + e) - f(base - e)) / (2 * STEPS[j])
    for n in OUTPUTS:
        i = OUTPUT_NAMES.index(n)
        a, b = float(J[i, j]), float(fd[i])
        assert b != 0.0, n
        assert abs(a - b) <= 1e-5 * abs(b), (lever, n, a, b)
