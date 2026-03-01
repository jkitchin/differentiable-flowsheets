"""Tests for REE Kremser equation and temperature correction bugs.

Bug #104: Kremser equation doesn't account for loaded solvent in extraction.
Bug #105: Scrubber Kremser formula is wrong (should match extraction formula).
Bug #106: Stripper Kremser formula is wrong (should match extraction formula).
Bug #107: Temperature correction divides by R*ln(10) unnecessarily.
"""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream, get_flows


# ===== Bug #107: Temperature correction =====

class TestTemperatureCorrection:
    """Bug #107: Temperature correction in distribution.py divides
    dH by R*ln(10) even though the 'd' coefficient is already in log10 units."""

    def test_temperature_sensitivity(self):
        """D should change significantly with temperature.

        The model is log10(D) = a + b*pH + c*pH^2 + d*(1/T - 1/Tref).
        At 298.15K D = 10^(a + b*pH + c*pH^2).
        At 350K, the d coefficient should cause a noticeable change.

        Before fix: d is divided by R*ln(10) (~19.14), attenuating the
        temperature effect by ~19x.
        """
        from difflow_ree.equilibrium.distribution import REEDistribution

        dist = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd",),
        )

        D_298 = dist.get_D("Nd", 3.0, 298.15)
        D_350 = dist.get_D("Nd", 3.0, 350.0)

        # The ratio D_350/D_298 should be meaningfully different from 1.0
        # With the bug (dividing by R*ln(10)), the effect is ~19x too small.
        ratio = float(D_350 / D_298)

        # For a typical dH ~ -20000 J/mol, the correct ratio at 350K should
        # deviate significantly from 1.0. With the bug, the deviation is ~19x smaller.
        # After fix, |ratio - 1| should be at least 0.1 (10% change over 52K)
        assert abs(ratio - 1.0) > 0.1, (
            f"Temperature effect too small: D_350/D_298 = {ratio:.4f}. "
            f"The 'd' coefficient is likely being attenuated by R*ln(10)."
        )


# ===== Bug #105: Scrubber Kremser formula =====

class TestScrubberKremser:
    """Bug #105: Scrubber uses wrong Kremser formula.

    The scrubber currently uses:
        frac_in_org = (E^(N+1) - E) / (E^(N+1) - 1)
    But should use the same extraction Kremser formula:
        frac_in_org = (E - 1) / (E^(N+1) - 1)

    When E > 1 (strong D, element prefers organic), both formulas give high
    frac_in_org, which is correct. But when E < 1 (weak D, element should
    be easily scrubbed), the wrong formula gives a higher frac_in_org than
    it should.
    """

    def test_scrubbing_impurity_with_low_E(self):
        """When E < 1, impurities should be easily scrubbed (low frac_in_org).

        With the wrong formula (E^(N+1) - E)/(E^(N+1) - 1):
        - E=0.5, N=5: frac = (0.5^6 - 0.5)/(0.5^6 - 1) = (-0.484)/(-0.984) = 0.492
        With the correct formula (E - 1)/(E^(N+1) - 1):
        - E=0.5, N=5: frac = (0.5 - 1)/(0.5^6 - 1) = (-0.5)/(-0.984) = 0.508

        Actually for scrubbing, when E < 1, the fraction remaining in organic
        should decrease with more stages. Let me think more carefully...

        For the Kremser equation with counter-current extraction:
        fraction_remaining = (E - 1) / (E^(N+1) - 1)

        For E < 1 and N large, this approaches 0 (good scrubbing).
        For the buggy formula: (E^(N+1) - E) / (E^(N+1) - 1)
        For E < 1 and N large, E^(N+1) -> 0, so = (-E)/(-1) = E

        So the buggy formula converges to E (not 0), meaning scrubbing
        is limited even with many stages.
        """
        from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams

        # Use D2EHPA at low pH where La has low D (easy to scrub)
        params = ScrubberParams(
            n_stages=20,  # Many stages
            extractant="D2EHPA",
            elements=("La", "Nd"),
            target_elements=("Nd",),
            pH=1.0,  # Very low pH - should scrub La easily
            extractant_conc=0.5,
        )

        scrubber = REEScrubber(params)

        # Create loaded organic with La and Nd
        loaded_org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "La": 0.1, "Nd": 0.1},
            T=298.15, P=101325.0,
        )
        scrub_soln = make_stream(
            {"H2O": 10.0},
            T=298.15, P=101325.0,
        )

        scrub_liquor, scrubbed_org, info = scrubber(loaded_org, scrub_soln, pH=1.0)

        scrubbed_flows = get_flows(scrubbed_org)

        # With 20 stages and low pH (low D for La), La should be almost
        # completely scrubbed from organic.
        # After fix: La remaining in organic should be < 5% of initial.
        la_remaining_frac = float(scrubbed_flows.get("La", 0.0)) / 0.1

        assert la_remaining_frac < 0.05, (
            f"La remaining in organic: {la_remaining_frac:.4f}. "
            f"With 20 stages and low pH, La should be nearly completely scrubbed. "
            f"Bug: Kremser formula converges to E instead of 0."
        )


# ===== Bug #106: Stripper Kremser formula =====

class TestStripperKremser:
    """Bug #106: Stripper uses wrong Kremser formula (same issue as scrubber)."""

    def test_stripping_at_low_pH(self):
        """At very low pH with many stages, stripping should be nearly complete.

        With buggy formula, stripping efficiency converges to E (not 0).
        """
        from difflow_ree.units.stripping import REEStripper, StripperParams

        params = StripperParams(
            n_stages=20,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=0.5,  # Very low pH
        )

        stripper = REEStripper(params)

        loaded_org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.1},
            T=298.15, P=101325.0,
        )
        strip_acid = make_stream(
            {"H2O": 10.0},
            T=298.15, P=101325.0,
        )

        product, barren_org, info = stripper(loaded_org, strip_acid, pH=0.5)

        barren_flows = get_flows(barren_org)

        # With 20 stages and very low pH, stripping should be nearly complete
        nd_remaining = float(barren_flows.get("Nd", 0.0))
        nd_initial = 0.1
        frac_remaining = nd_remaining / nd_initial

        assert frac_remaining < 0.05, (
            f"Nd remaining in organic: {frac_remaining:.4f}. "
            f"With 20 stages at pH 0.5, stripping should be >95% complete. "
            f"Bug: Kremser formula converges to E instead of 0."
        )


# ===== Bug #104: Loaded solvent not accounted for =====

class TestExtractorLoadedSolvent:
    """Bug #104: Extraction doesn't account for REE already in solvent."""

    def test_loaded_solvent_reduces_extraction(self):
        """When solvent already contains REE, extraction should be reduced.

        Current code treats REE in solvent same as REE in feed:
            F_raffinate = F_in * frac_remaining + F_solvent * frac_remaining

        This is wrong - the solvent's existing REE should reduce the
        effective driving force for extraction.
        """
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

        params = REEExtractorParams(
            n_stages=5,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            include_loading=False,
        )

        extractor = REEExtractor(params)

        feed = make_stream(
            {"H2O": 10.0, "Nd": 0.1},
            T=298.15, P=101325.0,
        )

        # Fresh solvent
        fresh_solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0},
            T=298.15, P=101325.0,
        )

        # Loaded solvent (already has Nd)
        loaded_solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.05},
            T=298.15, P=101325.0,
        )

        # Extract with fresh solvent
        raff_fresh, ext_fresh, _ = extractor(feed, fresh_solvent)
        raff_fresh_flows = get_flows(raff_fresh)

        # Extract with loaded solvent
        raff_loaded, ext_loaded, _ = extractor(feed, loaded_solvent)
        raff_loaded_flows = get_flows(raff_loaded)

        # With loaded solvent, less Nd should be extracted from the feed,
        # so more Nd should remain in the raffinate.
        nd_raff_fresh = float(raff_fresh_flows.get("Nd", 0.0))
        nd_raff_loaded = float(raff_loaded_flows.get("Nd", 0.0))

        assert nd_raff_loaded > nd_raff_fresh * 1.1, (
            f"Loaded solvent should reduce extraction. "
            f"Nd in raffinate with fresh solvent: {nd_raff_fresh:.6f}, "
            f"with loaded solvent: {nd_raff_loaded:.6f}. "
            f"Loaded solvent raffinate should have at least 10% more Nd."
        )
