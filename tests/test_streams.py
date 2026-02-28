"""Tests for stream operations."""

import jax.numpy as jnp
import pytest

from difflow import (
    make_stream,
    get_flows,
    get_species,
    total_flow,
    mole_fractions,
    combine_streams,
    scale_stream,
)


class TestMakeStream:
    def test_basic_stream(self):
        stream = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)

        assert "F_A" in stream
        assert "F_B" in stream
        assert "T" in stream
        assert "P" in stream
        assert float(stream["F_A"]) == pytest.approx(10.0)
        assert float(stream["F_B"]) == pytest.approx(5.0)
        assert float(stream["T"]) == pytest.approx(300.0)
        assert float(stream["P"]) == pytest.approx(101325.0)

    def test_stream_with_jax_arrays(self):
        stream = make_stream(
            {"A": jnp.array(10.0), "B": jnp.array(5.0)},
            T=jnp.array(300.0),
            P=jnp.array(101325.0),
        )

        assert float(stream["F_A"]) == pytest.approx(10.0)


class TestGetFlows:
    def test_get_flows(self):
        stream = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
        flows = get_flows(stream)

        assert "A" in flows
        assert "B" in flows
        assert "T" not in flows
        assert "P" not in flows
        assert float(flows["A"]) == pytest.approx(10.0)


class TestGetSpecies:
    def test_get_species(self):
        stream = make_stream({"A": 10.0, "B": 5.0, "C": 1.0}, T=300.0, P=101325.0)
        species = get_species(stream)

        assert set(species) == {"A", "B", "C"}


class TestTotalFlow:
    def test_total_flow(self):
        stream = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)

        assert float(total_flow(stream)) == pytest.approx(15.0)


class TestMoleFractions:
    def test_mole_fractions(self):
        stream = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
        x = mole_fractions(stream)

        assert float(x["A"]) == pytest.approx(10.0 / 15.0)
        assert float(x["B"]) == pytest.approx(5.0 / 15.0)
        assert float(x["A"] + x["B"]) == pytest.approx(1.0)


class TestCombineStreams:
    def test_combine_two_streams(self):
        s1 = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
        s2 = make_stream({"A": 2.0, "B": 3.0}, T=310.0, P=101325.0)

        combined = combine_streams(s1, s2)

        assert float(combined["F_A"]) == pytest.approx(12.0)
        assert float(combined["F_B"]) == pytest.approx(8.0)
        # T is flow-weighted average: (15/20)*300 + (5/20)*310 = 302.5
        assert float(combined["T"]) == pytest.approx(302.5)
        # P is minimum of inlet pressures
        assert float(combined["P"]) == pytest.approx(101325.0)

    def test_combine_streams_different_pressures(self):
        """Outlet pressure should be the minimum of inlet pressures."""
        s1 = make_stream({"A": 1.0}, T=300.0, P=200000.0)
        s2 = make_stream({"A": 1.0}, T=300.0, P=101325.0)

        combined = combine_streams(s1, s2)

        assert float(combined["P"]) == pytest.approx(101325.0)

    def test_combine_streams_equal_flows_average_T(self):
        """Equal total flows give equal-weight average temperature."""
        s1 = make_stream({"A": 1.0}, T=300.0, P=101325.0)
        s2 = make_stream({"A": 1.0}, T=400.0, P=101325.0)

        combined = combine_streams(s1, s2)

        assert float(combined["T"]) == pytest.approx(350.0)

    def test_combine_streams_same_T(self):
        """Streams at the same temperature should produce that temperature."""
        s1 = make_stream({"A": 3.0, "B": 2.0}, T=350.0, P=101325.0)
        s2 = make_stream({"A": 1.0, "B": 4.0}, T=350.0, P=101325.0)

        combined = combine_streams(s1, s2)

        assert float(combined["T"]) == pytest.approx(350.0)


class TestScaleStream:
    def test_scale_stream(self):
        stream = make_stream({"A": 10.0, "B": 5.0}, T=300.0, P=101325.0)
        scaled = scale_stream(stream, 0.5)

        assert float(scaled["F_A"]) == pytest.approx(5.0)
        assert float(scaled["F_B"]) == pytest.approx(2.5)
        # T and P unchanged
        assert float(scaled["T"]) == pytest.approx(300.0)
        assert float(scaled["P"]) == pytest.approx(101325.0)
