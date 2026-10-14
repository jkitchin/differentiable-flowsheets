"""Tests for stream operations."""

import jax.numpy as jnp
import pytest

from difflow import (
    combine_streams,
    get_flows,
    get_species,
    make_stream,
    mole_fractions,
    scale_stream,
    total_flow,
)


def test_stream_helpers():
    s = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
    assert set(s) == {"F_A", "F_B", "T", "P"}
    assert set(get_flows(s)) == {"A", "B"}
    assert set(get_species(s)) == {"A", "B"}
    assert float(total_flow(s)) == pytest.approx(15.0)
    assert float(mole_fractions(s)["A"]) == pytest.approx(10.0 / 15.0)
    assert float(scale_stream(s, 0.5)["F_A"]) == pytest.approx(5.0)


def test_combine_streams_mismatched_species():
    s1 = make_stream({"A": 1.0}, T=300.0, P=200000.0)
    s2 = make_stream({"A": 1.0, "B": 2.0}, T=400.0, P=101325.0)
    c = combine_streams(s1, s2)
    assert float(c["F_B"]) == pytest.approx(2.0)
    assert float(total_flow(c)) == pytest.approx(4.0)
    assert float(c["P"]) == pytest.approx(101325.0)


def test_combine_zero_flow_is_finite():
    """Regression: two zero-flow inlets gave a NaN temperature."""
    s1 = make_stream({"A": 0.0}, T=300.0, P=101325.0)
    s2 = make_stream({"A": 0.0}, T=400.0, P=101325.0)
    assert not jnp.isnan(combine_streams(s1, s2)["T"])
