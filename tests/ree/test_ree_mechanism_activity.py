"""Mechanism dispatch (#195) and activity-correction validation (#194)."""

import jax.numpy as jnp
import pytest

from difflow.streams import make_stream
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)


def test_tbp_solvating_smoke():
    dist = REEDistribution(extractant="TBP", elements=("Nd",), nitrate_conc=3.0)
    assert dist.mechanism == "solvating"
    assert float(dist.get_D("Nd")) > 0.0


def test_tbp_without_nitrate_raises():
    with pytest.raises(ValueError, match=r"TBP.*requires a nitrate"):
        REEDistribution(extractant="TBP", elements=("Nd",))
    with pytest.raises(ValueError, match=r"nitrate_conc=0"):
        REEDistribution(extractant="TBP", elements=("Nd",), nitrate_conc=0.0)
    with pytest.raises(ValueError, match="ph_coefficients"):
        REEDistribution(extractant="TBP", elements=("Nd",),
                        mechanism="cation_exchange")


def test_tbp_in_a_chloride_medium_raises():
    with pytest.raises(ValueError, match="requires_nitrate"):
        REEDistribution(extractant="TBP", elements=("Nd",), nitrate_conc=3.0,
                        medium="chloride")


def test_cation_exchange_requires_ph():
    with pytest.raises(ValueError, match="requires pH"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",)).get_D("Nd")


def test_constructor_validation():
    with pytest.raises(ValueError, match="unknown mechanism"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        mechanism="anion_exchange")
    with pytest.raises(ValueError, match="Unknown medium"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",), medium="brine")
    with pytest.raises(ValueError, match="Unknown activity_model"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        activity_model="pitzer")
    with pytest.raises(ValueError, match="on_out_of_range"):
        REEDistribution(extractant="D2EHPA", elements=("Nd",),
                        on_out_of_range="explode")


def test_out_of_davies_range_warns_or_raises():
    dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
    with pytest.warns(UserWarning, match="validity range"):
        dist.get_D("Nd", pH=1.5, ionic_strength=3.0)
    strict = REEDistribution(extractant="D2EHPA", elements=("Nd",),
                             on_out_of_range="raise")
    with pytest.raises(ValueError, match="validity range"):
        strict.get_D("Nd", pH=1.5, ionic_strength=3.0)


def test_extract_scrub_strip_runs_with_tbp():
    circuit = ExtractScrubStripCircuit(ExtractScrubStripParams(
        extractant="TBP", elements=("Nd", "Dy"), target_elements=("Dy",),
        nitrate_conc=3.0, extractant_conc=1.1))
    results = circuit(make_stream({"H2O": 1000.0, "Nd": 1.0, "Dy": 0.5},
                                  298.15, 101325.0))
    assert jnp.isfinite(results["target_purity"])
