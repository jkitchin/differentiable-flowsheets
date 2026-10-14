"""``no_data=`` modes of REEDistribution (#384)."""

import warnings

import jax.numpy as jnp
import pytest

from difflow_ree.equilibrium.distribution import REEDistribution


def test_nan_mode_answers_nan_for_an_element_with_no_coefficients():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = REEDistribution("D2EHPA", ("Nd", "Ho"), no_data="nan")
        assert jnp.isnan(d.get_D("Ho", 1.0))
        assert jnp.isfinite(d.get_D("Nd", 1.0))


def test_raise_mode_refuses_a_heavy_outside_the_window():
    d = REEDistribution("D2EHPA", ("Nd", "Dy"), no_data="raise")
    with pytest.raises(ValueError, match="no data for"):
        d.get_D("Dy", -0.78)


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="no_data"):
        REEDistribution("D2EHPA", ("Nd",), no_data="zero")
