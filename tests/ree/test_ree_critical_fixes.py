"""Tests for REE plugin critical bug fixes (#104-#107)."""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

pytest.importorskip("difflow_ree")


class TestBug104_ExtractionLoadedSolvent:
    """Bug #104: Kremser equation should account for loaded solvent."""

    def test_loaded_solvent_reduces_extraction(self):
        """Extraction with pre-loaded solvent should be less efficient."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.extraction import REEExtractor, REEExtractorParams

        params = REEExtractorParams(
            n_stages=5,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=3.0,
            include_loading=True,
        )
        extractor = REEExtractor(params)

        feed = make_stream({"H2O": 10.0, "Nd": 1.0}, T=298.15, P=101325.0)

        # Fresh solvent (no initial Nd loading)
        fresh_solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0}, T=298.15, P=101325.0
        )
        raff_fresh, _, _ = extractor(feed, fresh_solvent)

        # Loaded solvent (significant Nd already present)
        loaded_solvent = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 0.05},
            T=298.15,
            P=101325.0,
        )
        raff_loaded, _, _ = extractor(feed, loaded_solvent)

        # With loaded solvent, MORE Nd should remain in raffinate
        # (less efficient extraction)
        nd_remaining_fresh = float(get_flows(raff_fresh).get("Nd", 0.0))
        nd_remaining_loaded = float(get_flows(raff_loaded).get("Nd", 0.0))
        assert nd_remaining_loaded > nd_remaining_fresh, (
            f"Loaded solvent should give less extraction: "
            f"fresh raffinate Nd={nd_remaining_fresh:.6f}, "
            f"loaded raffinate Nd={nd_remaining_loaded:.6f}"
        )


class TestBug105_ScrubberKremser:
    """Bug #105: Scrubber Kremser has swapped fraction."""

    def test_high_scrub_factor_gives_high_removal(self):
        """With scrub factor S > 1 and multiple stages, most impurity removed."""
        from difflow.numerics import safe_divide

        S = jnp.float64(3.0)
        N = jnp.float64(5.0)
        S_Np1 = jnp.power(S, N + 1)

        frac_remaining = float(safe_divide(S - 1.0, S_Np1 - 1.0))
        frac_removed = 1.0 - frac_remaining

        assert frac_removed > 0.99, (
            f"With S=3, N=5 should remove >99%: got {frac_removed:.4f}"
        )

    def test_scrubber_unit_high_removal(self):
        """REEScrubber should remove impurities with high scrub factor."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.scrubbing import REEScrubber, ScrubberParams

        params = ScrubberParams(
            n_stages=5,
            extractant="D2EHPA",
            elements=("La", "Dy"),
            target_elements=("Dy",),
            pH=1.5,
        )
        scrubber = REEScrubber(params)

        loaded_org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "La": 1.0, "Dy": 1.0},
            T=298.15,
            P=101325.0,
        )
        scrub_soln = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)

        scrub_liquor, scrubbed_org, info = scrubber(loaded_org, scrub_soln)

        la_in_org = float(get_flows(scrubbed_org).get("La", 0.0))
        la_removed_frac = 1.0 - la_in_org / 1.0

        assert la_removed_frac > 0.5, (
            f"Scrubber should remove >50% La impurity with 5 stages: "
            f"got {la_removed_frac:.4f}"
        )


class TestBug106_StripperKremser:
    """Bug #106: Stripper Kremser has same swap error as scrubber."""

    def test_high_strip_factor_gives_high_recovery(self):
        """With strip factor > 1, stripping should be near-complete."""
        S = 5.0
        N = 5.0

        S_Np1 = S ** (N + 1)
        frac_remaining = (S - 1.0) / (S_Np1 - 1.0)
        frac_stripped = 1.0 - frac_remaining

        assert frac_stripped > 0.999, (
            f"With S=5, N=5 should strip >99.9%: got {frac_stripped:.6f}"
        )

    def test_stripper_unit_high_recovery(self):
        """REEStripper should achieve high recovery at low pH."""
        from difflow.streams import make_stream, get_flows
        from difflow_ree.units.stripping import REEStripper, StripperParams

        params = StripperParams(
            n_stages=5,
            extractant="D2EHPA",
            elements=("Nd",),
            pH=0.5,
        )
        stripper = REEStripper(params)

        loaded_org = make_stream(
            {"D2EHPA": 1.0, "kerosene": 5.0, "Nd": 1.0},
            T=298.15,
            P=101325.0,
        )
        strip_soln = make_stream({"H2O": 10.0}, T=298.15, P=101325.0)

        product, barren_org, info = stripper(loaded_org, strip_soln)

        nd_in_product = float(get_flows(product).get("Nd", 0.0))
        nd_stripped_frac = nd_in_product / 1.0

        assert nd_stripped_frac > 0.5, (
            f"Stripper should recover >50% Nd with 5 stages at pH 0.5: "
            f"got {nd_stripped_frac:.4f}"
        )


class TestBug107_TemperatureCorrection:
    """Bug #107: Temperature correction attenuated by ~19x."""

    def test_temperature_correction_magnitude(self):
        """Temperature correction should match d-coefficient model."""
        from difflow_ree import REEDistribution

        dist = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd",),
            concentration=0.5,
        )

        T_ref = 298.15
        T_new = 350.0

        D_ref = dist.get_D("Nd", pH=3.0, T=T_ref)
        D_new = dist.get_D("Nd", pH=3.0, T=T_new)

        log_D_ref = float(jnp.log10(D_ref))
        log_D_new = float(jnp.log10(D_new))
        delta_log_D = log_D_new - log_D_ref

        d_coeff = -1800.0
        expected_delta = d_coeff * (1.0 / T_new - 1.0 / T_ref)

        attenuated_delta = float(
            d_coeff / (8.314 * jnp.log(10.0)) * (1.0 / T_new - 1.0 / T_ref)
        )

        assert abs(delta_log_D - expected_delta) < 0.01, (
            f"Temperature correction mismatch: got delta_log_D={delta_log_D:.4f}, "
            f"expected={expected_delta:.4f} (attenuated would be {attenuated_delta:.4f})"
        )

        assert abs(delta_log_D - expected_delta) < abs(
            delta_log_D - attenuated_delta
        ), "Temperature correction appears to be attenuated by R*ln(10)"
