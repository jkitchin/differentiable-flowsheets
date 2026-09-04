"""Tests for REE extractor mass conservation (issue #53).

Verifies that non-extracted species (inerts) are preserved through
REEExtractor and REEMixerSettler unit operations.
"""

import pytest
import jax.numpy as jnp

from difflow.streams import make_stream, get_flows
from difflow_ree import REEExtractorParams, REEExtractor
from difflow_ree.units.extraction import (
    MixerSettlerParams,
    REEMixerSettler,
)


@pytest.fixture
def feed():
    """Aqueous feed with an inert species (Fe)."""
    return make_stream(
        flows={"H2O": 10.0, "Nd": 0.2, "Dy": 0.143827799, "Fe": 0.553},
        T=298.15,
        P=101325.0,
    )


@pytest.fixture
def solvent():
    """Organic solvent: D2EHPA extractant in an n-Heptane diluent.

    The carriers are named after the extractant and the diluent the
    extractor is configured with. A stream whose organic carrier is called
    something else ("Organic") no longer silently contributes a default
    organic flow of 1.0 -- it raises (#192).
    """
    return make_stream(
        flows={
            "D2EHPA": 10.0,
            "n-Heptane": 20.0,
            "Nd": 0.0,
            "Dy": 0.0,
            "Fe": 0.0,
        },
        T=298.15,
        P=101325.0,
    )


def _check_mass_conservation(feed, solvent, raffinate, extract, atol=1e-10):
    """Assert total mass is conserved for every species."""
    feed_flows = get_flows(feed)
    solvent_flows = get_flows(solvent)
    raff_flows = get_flows(raffinate)
    ext_flows = get_flows(extract)

    all_species = set(feed_flows.keys()) | set(solvent_flows.keys())
    for species in all_species:
        total_in = float(feed_flows.get(species, 0.0)) + float(
            solvent_flows.get(species, 0.0)
        )
        total_out = float(raff_flows.get(species, 0.0)) + float(
            ext_flows.get(species, 0.0)
        )
        assert abs(total_in - total_out) < atol, (
            f"Mass not conserved for {species}: in={total_in:.6f}, out={total_out:.6f}"
        )


class TestREEExtractorMassConservation:
    """Mass conservation tests for REEExtractor (Kremser-based)."""

    def test_inert_species_preserved(self, feed, solvent):
        """Non-element species must appear unchanged in outputs."""
        params = REEExtractorParams(
            n_stages=1,
            extractant="D2EHPA",
            diluent="n-Heptane",  # names the solvent fixture's carrier (#192)
            elements=("Dy", "Nd"),
            pH=1.6,
        )
        extractor = REEExtractor(params)
        raffinate, extract, info = extractor(feed, solvent)

        raff_flows = get_flows(raffinate)
        ext_flows = get_flows(extract)

        # Fe is inert -- should stay in raffinate at its feed value
        assert "Fe" in raff_flows, "Fe missing from raffinate"
        assert float(raff_flows["Fe"]) == pytest.approx(0.553, abs=1e-10)

        # n-Heptane is inert -- should stay in extract at its solvent value
        assert "n-Heptane" in ext_flows, "n-Heptane missing from extract"
        assert float(ext_flows["n-Heptane"]) == pytest.approx(20.0, abs=1e-10)

    def test_total_mass_conservation(self, feed, solvent):
        """Total input must equal total output for every species."""
        params = REEExtractorParams(
            n_stages=1,
            extractant="D2EHPA",
            diluent="n-Heptane",  # names the solvent fixture's carrier (#192)
            elements=("Dy", "Nd"),
            pH=1.6,
        )
        extractor = REEExtractor(params)
        raffinate, extract, info = extractor(feed, solvent)
        _check_mass_conservation(feed, solvent, raffinate, extract)

    def test_multistage_mass_conservation(self, feed, solvent):
        """Mass conservation holds for multi-stage extraction."""
        params = REEExtractorParams(
            n_stages=5,
            extractant="D2EHPA",
            diluent="n-Heptane",  # names the solvent fixture's carrier (#192)
            elements=("Dy", "Nd"),
            pH=3.0,
        )
        extractor = REEExtractor(params)
        raffinate, extract, info = extractor(feed, solvent)
        _check_mass_conservation(feed, solvent, raffinate, extract)


class TestMixerSettlerMassConservation:
    """Mass conservation tests for REEMixerSettler (single stage)."""

    def test_inert_species_preserved(self):
        """Non-element species must appear unchanged in outputs."""
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=("Nd", "Dy"),
            pH=3.0,
        )
        stage = REEMixerSettler(params)

        aq_in = make_stream(
            flows={"H2O": 10.0, "Nd": 0.1, "Dy": 0.05, "Fe": 0.3},
            T=298.15,
            P=101325.0,
        )
        org_in = make_stream(
            flows={
                "D2EHPA": 5.0,
                "kerosene": 15.0,
                "Nd": 0.0,
                "Dy": 0.0,
                "n-Heptane": 10.0,
            },
            T=298.15,
            P=101325.0,
        )

        aq_out, org_out, info = stage(aq_in, org_in)
        aq_out_flows = get_flows(aq_out)
        org_out_flows = get_flows(org_out)

        # Fe should remain in aqueous outlet
        assert "Fe" in aq_out_flows, "Fe missing from aqueous outlet"
        assert float(aq_out_flows["Fe"]) == pytest.approx(0.3, abs=1e-10)

        # n-Heptane should remain in organic outlet
        assert "n-Heptane" in org_out_flows, "n-Heptane missing from organic outlet"
        assert float(org_out_flows["n-Heptane"]) == pytest.approx(10.0, abs=1e-10)

    def test_total_mass_conservation(self):
        """Total input equals total output for every species."""
        params = MixerSettlerParams(
            extractant="D2EHPA",
            elements=("Nd", "Dy"),
            pH=3.0,
        )
        stage = REEMixerSettler(params)

        aq_in = make_stream(
            flows={"H2O": 10.0, "Nd": 0.1, "Dy": 0.05, "Fe": 0.3},
            T=298.15,
            P=101325.0,
        )
        org_in = make_stream(
            flows={
                "D2EHPA": 5.0,
                "kerosene": 15.0,
                "Nd": 0.0,
                "Dy": 0.0,
            },
            T=298.15,
            P=101325.0,
        )

        aq_out, org_out, info = stage(aq_in, org_in)
        _check_mass_conservation(aq_in, org_in, aq_out, org_out)
