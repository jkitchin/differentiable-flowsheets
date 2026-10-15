"""Free-extractant closure and the loading-capacity check (#267)."""

import jax
import jax.numpy as jnp
import pytest

from difflow_ree import (
    ExtractantCapacityWarning,
    check_loading_capacity,
    solve_free_extractant,
)
from difflow_ree.equilibrium.distribution import REEDistribution


def test_solve_and_grad_are_finite():
    dist = REEDistribution(extractant="PC88A", elements=("Nd", "Pr"),
                           concentration=0.5)
    result = solve_free_extractant(dist, "Nd", 1e-4, pH=1.5)
    assert result.converged is True
    assert jnp.isfinite(result.D)
    g = jax.grad(lambda c: solve_free_extractant(dist, "Nd", c, pH=1.5).D)(1e-4)
    assert jnp.isfinite(g)


def test_an_impossible_loading_warns_or_raises():
    with pytest.warns(ExtractantCapacityWarning):
        check_loading_capacity("naphthenic_acid", 0.066, 0.13)
    with pytest.raises(ValueError, match="capacity"):
        check_loading_capacity("naphthenic_acid", 0.066, 0.13, action="raise")


def test_an_unknown_action_is_refused():
    with pytest.raises(ValueError, match="action"):
        check_loading_capacity("naphthenic_acid", 0.01, 0.45, action="shout")
