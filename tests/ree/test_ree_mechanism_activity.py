"""Tests for REE extraction mechanism dispatch (#195) and the activity
correction convention / validity range (#194).

Issue #195: `data/extractants.yaml` encoded a solvating mechanism for TBP
(nitrate coefficients, zero protons released, requires_nitrate) that no code
read; TBP was modelled with a pH slope, i.e. as a weak cation exchanger.

Issue #194: the ionic-strength correction multiplied D by the rare-earth
activity coefficient alone, on an unstated pH scale, and Davies was applied far
outside its documented validity range.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from difflow_ree.database import (
    EXTRACTION_MECHANISMS,
    create_custom_extractant,
    get_extractant,
    normalize_mechanism,
)
from difflow_ree.equilibrium.distribution import REEDistribution
from difflow_ree.equilibrium.speciation import (
    AQUEOUS_MEDIA,
    DAVIES_MAX_IONIC_STRENGTH,
    DAVIES_SIGN_CHANGE_IONIC_STRENGTH,
    NITRATE_BEARING_MEDIA,
)
from difflow.streams import make_stream
from difflow_ree.flowsheets.extract_scrub_strip import (
    ExtractScrubStripCircuit,
    ExtractScrubStripParams,
)
from difflow_ree.flowsheets.extract_strip import (
    ExtractStripCircuit,
    ExtractStripParams,
)
from difflow_ree.flowsheets.full_train import (
    FullSeparationTrain,
    SeparationTrainParams,
)
from difflow_ree.flowsheets.split_shell import SplitShellCascade, SplitShellParams


# =============================================================================
# Independent reference implementations (deliberately NOT the library's)
# =============================================================================

def davies_gamma(z: int, ionic_strength: float) -> float:
    """Davies activity coefficient, written out independently of difflow_ree."""
    A = 0.509
    sqrt_I = np.sqrt(ionic_strength)
    log_gamma = -A * z**2 * (sqrt_I / (1.0 + sqrt_I) - 0.3 * ionic_strength)
    return float(10.0**log_gamma)


# =============================================================================
# #195 - mechanism is a property of the data
# =============================================================================

class TestMechanismIsData:
    """Issue #195: dispatch on the declared extraction mechanism."""

    def test_195_extractant_record_exposes_nitrate_data(self):
        """nitrate_coefficients / reference_nitrate / requires_nitrate are
        reachable from the Extractant record rather than dead YAML."""
        tbp = get_extractant("TBP")
        assert tbp.nitrate_coefficients is not None
        assert set(tbp.nitrate_coefficients) >= {"La", "Nd", "Dy", "Y"}
        # b is now the stoichiometric 3.0 of RE(NO3)3 + 3 TBP for every
        # element; the old per-element 2.50-2.85 trend had no support.
        assert tbp.nitrate_coefficients["Nd"].b == pytest.approx(3.0)
        assert tbp.reference_nitrate == pytest.approx(3.0)
        assert tbp.requires_nitrate is True
        # protons_released was loaded but never read before #194/#195
        assert tbp.stoichiometry_protons == 0

    def test_195_mechanism_normalized_from_type(self):
        """`type` normalizes to one of the declared mechanisms."""
        assert normalize_mechanism("acidic_phosphoric") == "cation_exchange"
        assert normalize_mechanism("acidic_phosphonic") == "cation_exchange"
        assert normalize_mechanism("acidic_phosphinic") == "cation_exchange"
        assert normalize_mechanism("solvating_neutral") == "solvating"
        assert set(EXTRACTION_MECHANISMS) == {"cation_exchange", "solvating"}

    def test_195_records_declare_their_mechanism(self):
        assert get_extractant("TBP").mechanism == "solvating"
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            assert get_extractant(name).mechanism == "cation_exchange"

    def test_195_tbp_responds_to_nitrate_with_declared_slope(self):
        """log10(D) vs log10([NO3-]) has exactly the slope `b` from the record.

        The YAML says the coefficients are referenced to 3 M nitrate, so
        `a` is log10(D) at the reference and `b` is d log10(D) / d log10([NO3-]).
        """
        rec = get_extractant("TBP").nitrate_coefficients["Nd"]
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )

        D_lo = float(dist.get_D("Nd", nitrate_conc=1.5))
        D_hi = float(dist.get_D("Nd", nitrate_conc=6.0))
        slope = (np.log10(D_hi) - np.log10(D_lo)) / (
            np.log10(6.0) - np.log10(1.5)
        )
        assert slope == pytest.approx(rec.b, rel=1e-10)
        # ...and D rises with nitrate, the qualitative signature of solvating
        # extraction that the pH model got backwards.
        assert D_hi > D_lo

    def test_195_reference_nitrate_is_the_reference(self):
        """At [NO3-] = reference_nitrate, log10(D) is `a` plus only the
        (temperature and extractant-concentration) corrections."""
        ext = get_extractant("TBP")
        rec = ext.nitrate_coefficients["Nd"]
        # Use the reference extractant concentration so the [S]^n term is zero.
        dist = REEDistribution(
            extractant="TBP",
            elements=("Nd",),
            concentration=ext.reference_concentration,
            nitrate_conc=ext.reference_nitrate,
        )
        assert float(jnp.log10(dist.get_D("Nd"))) == pytest.approx(
            rec.a, rel=1e-10
        )

    def test_195_tbp_is_flat_in_ph_unlike_an_acidic_extractant(self):
        """The qualitative contrast #195 is about: a solvating extractant's D
        does not move with pH, a cation exchanger's moves by orders of
        magnitude over the same span."""
        tbp = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        acidic = REEDistribution(extractant="D2EHPA", elements=("Nd",))

        pH_lo, pH_hi = 1.5, 4.0
        tbp_ratio = float(tbp.get_D("Nd", pH=pH_hi)) / float(
            tbp.get_D("Nd", pH=pH_lo)
        )
        acid_ratio = float(acidic.get_D("Nd", pH=pH_hi)) / float(
            acidic.get_D("Nd", pH=pH_lo)
        )
        assert tbp_ratio == pytest.approx(1.0, rel=1e-12)
        assert acid_ratio > 100.0

    def test_195_no_nitrate_fails_loudly(self):
        """TBP with no nitrate concentration must fail, not silently fall back
        to its ph_coefficients block. This is the behaviour change of #195."""
        with pytest.raises(ValueError, match=r"TBP.*requires a nitrate"):
            REEDistribution(extractant="TBP", elements=("Nd",))

    def test_195_zero_nitrate_fails_on_the_driving_ion_not_on_prose(self):
        """nitrate_conc <= 0 means the salting anion that drives the
        correlation has no concentration. Asserted on the parameter, not on the
        word 'chloride' appearing in the message (D10b)."""
        with pytest.raises(ValueError, match=r"nitrate_conc=0"):
            REEDistribution(
                extractant="TBP", elements=("Nd",), nitrate_conc=0.0
            )

    def test_195_zero_in_a_nitrate_profile_is_caught(self):
        """The driving-ion check inspects a concrete array at its minimum, so a
        per-stage nitrate profile containing a zero is rejected."""
        with pytest.raises(ValueError, match=r"nitrate_conc=0"):
            REEDistribution(
                extractant="TBP",
                elements=("Nd",),
                nitrate_conc=jnp.array([3.0, 1.0, 0.0]),
            )


class TestMediumIsDetectedNotDescribed:
    """D10b: #195 asked for a raise "when used in a medium its coefficients do
    not cover". The data supports exactly one medium constraint --
    ``stoichiometry.requires_nitrate`` -- so that is what is detected, and the
    tests assert the detection rather than the wording of an error."""

    def test_declared_medium_is_checked_against_requires_nitrate(self):
        """TBP declared in a chloride liquor raises even though a perfectly
        good nitrate concentration was supplied: it is the *medium* that is
        wrong, and it is detected from the record's flag."""
        assert get_extractant("TBP").requires_nitrate is True
        for bad in ("chloride", "sulfate"):
            with pytest.raises(ValueError) as exc:
                REEDistribution(
                    extractant="TBP",
                    elements=("Nd",),
                    nitrate_conc=3.0,
                    medium=bad,
                )
            assert f"medium={bad!r}" in str(exc.value)
            assert "requires_nitrate" in str(exc.value)

    def test_nitrate_bearing_media_are_accepted(self):
        for good in NITRATE_BEARING_MEDIA:
            dist = REEDistribution(
                extractant="TBP",
                elements=("Nd",),
                nitrate_conc=3.0,
                medium=good,
            )
            assert float(dist.get_D("Nd")) > 0.0

    def test_medium_check_survives_the_mechanism_override(self):
        """The medium check runs before the mechanism's coefficient block is
        resolved, so a chloride medium is rejected on the record's
        requires_nitrate flag even when the caller asked for the (now deleted,
        and separately refused) cation-exchange path."""
        with pytest.raises(ValueError, match="requires a nitrate medium"):
            REEDistribution(
                extractant="TBP",
                elements=("Nd",),
                mechanism="cation_exchange",
                medium="chloride",
            )

    def test_no_record_forbids_a_medium_for_a_cation_exchanger(self):
        """The honest contract: nothing in the data declares a chloride or
        sulfate incompatibility, so nothing else is rejected. A test that
        expected D2EHPA to be refused in some medium would be asserting a
        constraint the database does not carry."""
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            assert get_extractant(name).requires_nitrate is False
            for medium in AQUEOUS_MEDIA:
                dist = REEDistribution(
                    extractant=name, elements=("Nd",), medium=medium
                )
                assert float(dist.get_D("Nd", pH=3.0)) > 0.0

    def test_unknown_medium_rejected(self):
        with pytest.raises(ValueError, match="Unknown medium"):
            REEDistribution(
                extractant="D2EHPA", elements=("Nd",), medium="brine"
            )

    def test_unstated_medium_is_not_guessed(self):
        """medium=None leaves the medium unstated; only the driving-ion check
        applies."""
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        assert dist.medium is None
        assert float(dist.get_D("Nd")) > 0.0


class TestMechanismIsDataContinued:
    """Remainder of the #195 dispatch tests."""

    def test_195_error_names_the_extractant_and_the_way_out(self):
        """The way out is nitrate_conc, and -- now that TBP's pH block is
        deleted -- the message must say that mechanism='cation_exchange' is
        NOT a second way out, rather than advertising a path that raises."""
        with pytest.raises(ValueError) as exc:
            REEDistribution(extractant="TBP", elements=("Nd",))
        msg = str(exc.value)
        assert "TBP" in msg
        assert "nitrate_conc" in msg
        assert "mechanism='cation_exchange'" in msg
        assert "raises" in msg
        assert "only path" in msg

    def test_tbp_cation_exchange_opt_in_now_raises(self):
        """TBP's ph_coefficients block was DELETED, so the opt-in that used to
        reach it must now raise -- loudly, naming TBP, and pointing at the
        nitrate path. No silent fall-back to nitrate_coefficients, and no
        AttributeError/KeyError leaking out of a later get_D.

        The block modelled a neutral extractant (pKa null, protons_released 0)
        as a weak cation exchanger: there is no proton to exchange, no source
        reported a pH slope for TBP, and it carried the same refuted 100x
        La-to-Dy spread as the old nitrate block."""
        with pytest.raises(ValueError) as exc:
            REEDistribution(
                extractant="TBP", elements=("Nd",), mechanism="cation_exchange"
            )
        msg = str(exc.value)
        assert "TBP" in msg
        assert "ph_coefficients" in msg
        assert "nitrate_conc" in msg  # points at the path that does exist

    def test_tbp_record_carries_no_ph_block_at_all(self):
        """The field is None, not an empty dict: there is no pH-driven
        correlation for TBP to be found, mutated or fallen back to."""
        assert get_extractant("TBP").ph_coefficients is None

    def test_the_nitrate_path_is_unaffected_by_the_deletion(self):
        nitrate_path = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        assert nitrate_path.mechanism == "solvating"
        assert float(nitrate_path.get_D("Nd")) > 0.0

    def test_195_cation_exchange_requires_ph(self):
        acidic = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with pytest.raises(ValueError, match="requires pH"):
            acidic.get_D("Nd")

    def test_195_unknown_mechanism_rejected(self):
        with pytest.raises(ValueError, match="unknown mechanism"):
            REEDistribution(
                extractant="D2EHPA", elements=("Nd",), mechanism="anion_exchange"
            )

    def test_195_missing_element_names_the_block(self):
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        with pytest.raises(KeyError, match="nitrate_coefficients"):
            dist.get_D("Lu")


# =============================================================================
# #194 - activity correction convention and validity range
# =============================================================================

class TestActivityConvention:
    """Issue #194: gamma_RE / gamma_H**p, and the Davies validity range."""

    def test_194_correction_is_gamma_re_over_gamma_h_to_the_p(self):
        """Compare against an independently computed Davies value, not against
        the implementation."""
        I = 0.2
        ext = get_extractant("D2EHPA")
        p = ext.stoichiometry_protons
        assert p == 3  # sanity: the record actually carries it

        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        D_plain = float(dist.get_D("Nd", pH=3.0))
        D_corr = float(dist.get_D("Nd", pH=3.0, ionic_strength=I))

        expected = davies_gamma(3, I) / davies_gamma(1, I) ** p
        assert D_corr / D_plain == pytest.approx(expected, rel=1e-10)

        # And it is NOT gamma_RE alone - the pre-#194 behaviour.
        assert D_corr / D_plain != pytest.approx(davies_gamma(3, I), rel=1e-3)

    def test_194_proton_term_uses_p_from_the_record(self):
        """Every cation exchanger in the database carries p = 3, and the
        correction tracks it rather than a hard-coded charge."""
        I = 0.15
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            p = get_extractant(name).stoichiometry_protons
            dist = REEDistribution(extractant=name, elements=("Nd",))
            ratio = float(
                dist.get_D("Nd", pH=3.0, ionic_strength=I)
            ) / float(dist.get_D("Nd", pH=3.0))
            expected = davies_gamma(3, I) / davies_gamma(1, I) ** p
            assert ratio == pytest.approx(expected, rel=1e-10)

    def test_194_solvating_extractant_gets_no_proton_term(self):
        """p = 0 for TBP, so the correction carries gamma_RE and nothing else."""
        I = 0.2
        assert get_extractant("TBP").stoichiometry_protons == 0
        dist = REEDistribution(
            extractant="TBP",
            elements=("Nd",),
            nitrate_conc=3.0,
            on_out_of_range="ignore",
        )
        ratio = float(dist.get_D("Nd", ionic_strength=I)) / float(
            dist.get_D("Nd")
        )
        assert ratio == pytest.approx(davies_gamma(3, I), rel=1e-10)

    def test_194_solvating_warns_that_salting_is_not_this_correction(self):
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        with pytest.warns(UserWarning, match="salting"):
            dist.get_D("Nd", ionic_strength=0.2)

    def test_194_out_of_davies_range_warns(self):
        """A 2-4 M chloride liquor is an order of magnitude outside Davies."""
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with pytest.warns(UserWarning, match="validity range"):
            dist.get_D("Nd", pH=3.0, ionic_strength=3.0)

    def test_194_in_range_does_not_warn(self):
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            dist.get_D("Nd", pH=3.0, ionic_strength=0.4)

    def test_194_out_of_range_can_be_escalated_to_an_error(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError, match="validity range"):
            dist.get_D("Nd", pH=3.0, ionic_strength=3.0)

    def test_194_out_of_range_can_be_silenced(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            D = float(dist.get_D("Nd", pH=3.0, ionic_strength=3.0))
        assert np.isfinite(D)

    def test_194_warning_does_not_spam_a_stage_loop(self):
        """The check is parameter-time: repeated calls report once."""
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(25):
                dist.get_D("Nd", pH=3.0, ionic_strength=3.0)
        assert len(caught) == 1

    def test_194_ionic_strength_none_reproduces_the_correlation_exactly(self):
        """The honest high-ionic-strength option: a conditional constant, with
        no activity correction at all."""
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        ext = get_extractant("D2EHPA")
        c = ext.ph_coefficients["Nd"]
        n = ext.concentration_exponent
        expected_log = (
            c.a
            + c.b * 3.0
            + c.c * 3.0**2
            + n * np.log10(dist.concentration / ext.reference_concentration)
        )
        D_none = float(dist.get_D("Nd", pH=3.0, ionic_strength=None))
        assert np.log10(D_none) == pytest.approx(expected_log, rel=1e-12)
        assert D_none == pytest.approx(float(dist.get_D("Nd", pH=3.0)), rel=0)

    def test_194_activity_model_none_is_uncorrected_at_any_i(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), activity_model="none"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            D_corr = float(dist.get_D("Nd", pH=3.0, ionic_strength=4.0))
        assert D_corr == pytest.approx(float(dist.get_D("Nd", pH=3.0)), rel=0)

    def test_194_unimplemented_activity_models_are_refused(self):
        """Bromley / SIT parameters are not carried, so they are not offered."""
        for model in ("bromley", "sit", "pitzer"):
            with pytest.raises(ValueError, match="Unknown activity_model"):
                REEDistribution(
                    extractant="D2EHPA",
                    elements=("Nd",),
                    activity_model=model,
                )

    def test_194_bad_on_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="on_out_of_range"):
            REEDistribution(
                extractant="D2EHPA", elements=("Nd",), on_out_of_range="explode"
            )

    def test_194_get_D_all_threads_the_correction(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd", "Dy"), on_out_of_range="ignore"
        )
        plain = dist.get_D_all(pH=3.0)
        corr = dist.get_D_all(pH=3.0, ionic_strength=0.3)
        for e in ("Nd", "Dy"):
            assert float(corr[e]) < float(plain[e])


# =============================================================================
# #194 - the Davies sign inversion (the defect the range guard exists for)
# =============================================================================

class TestDaviesSignInversion:
    """Davies' bracket f(I) = sqrt(I)/(1+sqrt(I)) - 0.3 I changes sign, so the
    correction gamma_RE/gamma_H**3 = 10**(-6 A f) crosses 1 and starts
    MULTIPLYING D -- in exactly the 2-4 M chloride regime #194 was filed
    about. These tests pin both the arithmetic fact and the guard."""

    def test_the_documented_sign_change_is_the_actual_root(self):
        """Solved, not guessed: f(I) = 0 for x = sqrt(I) at
        0.3 x^2 + 0.3 x - 1 = 0."""
        I_flip = DAVIES_SIGN_CHANGE_IONIC_STRENGTH
        f = lambda I: np.sqrt(I) / (1.0 + np.sqrt(I)) - 0.3 * I
        assert f(I_flip) == pytest.approx(0.0, abs=1e-12)
        assert f(I_flip - 1e-3) > 0.0
        assert f(I_flip + 1e-3) < 0.0
        x = (-1.0 + np.sqrt(1.0 + 40.0 / 3.0)) / 2.0
        assert I_flip == pytest.approx(x**2, rel=1e-15)

    def test_raw_davies_really_does_invert(self):
        """The trap, stated independently of difflow_ree: above ~1.94 M the
        uncorrected ratio exceeds 1."""
        assert davies_gamma(3, 0.1) / davies_gamma(1, 0.1) ** 3 < 1.0
        assert davies_gamma(3, 1.0) / davies_gamma(1, 1.0) ** 3 < 1.0
        assert davies_gamma(3, 3.0) / davies_gamma(1, 3.0) ** 3 == pytest.approx(
            6.492942892894, rel=1e-9
        )
        assert davies_gamma(3, 4.0) / davies_gamma(1, 4.0) ** 3 > 40.0

    def test_D_is_never_multiplied_by_the_activity_correction(self):
        """The guard: for no ionic strength, however large, does the default
        path return a correction greater than 1."""
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        D_plain = float(dist.get_D("Nd", pH=3.0))
        for I in (0.1, 0.5, 1.0, 1.9, 2.0, 2.5, 3.0, 4.0, 10.0, 100.0):
            ratio = float(dist.get_D("Nd", pH=3.0, ionic_strength=I)) / D_plain
            assert ratio < 1.0, f"correction inverted at I={I}: ratio={ratio}"

    def test_out_of_range_correction_saturates_at_the_range_limit(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        limit = float(dist.get_D("Nd", pH=3.0, ionic_strength=DAVIES_MAX_IONIC_STRENGTH))
        for I in (0.6, 2.0, 3.0, 4.0):
            assert float(
                dist.get_D("Nd", pH=3.0, ionic_strength=I)
            ) == pytest.approx(limit, rel=1e-12)

    def test_in_range_values_are_untouched_by_the_clamp(self):
        """The guard must not perturb anything Davies is actually valid for."""
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        D_plain = float(dist.get_D("Nd", pH=3.0))
        for I in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            ratio = float(dist.get_D("Nd", pH=3.0, ionic_strength=I)) / D_plain
            expected = davies_gamma(3, I) / davies_gamma(1, I) ** 3
            assert ratio == pytest.approx(expected, rel=1e-10)

    def test_inversion_requires_an_explicit_opt_in(self):
        """It is still reachable -- but only by asking for it, and the number
        you get is exactly the raw Davies one."""
        opted_in = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd",),
            on_out_of_range="ignore",
            extrapolate_activity_model=True,
        )
        D_plain = float(opted_in.get_D("Nd", pH=3.0))
        ratio = float(opted_in.get_D("Nd", pH=3.0, ionic_strength=3.0)) / D_plain
        assert ratio > 1.0
        assert ratio == pytest.approx(
            davies_gamma(3, 3.0) / davies_gamma(1, 3.0) ** 3, rel=1e-10
        )

    def test_a_concrete_array_is_range_checked_at_its_maximum(self):
        """A concrete jnp.array is inspectable; only float() refuses it. The
        pre-fix code treated it like a tracer and warned 0 times (#194)."""
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with pytest.warns(UserWarning, match="validity range"):
            dist.get_D("Nd", pH=3.0, ionic_strength=jnp.array([0.1, 3.0, 4.0]))

    def test_a_concrete_in_range_array_does_not_warn(self):
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            dist.get_D(
                "Nd", pH=3.0, ionic_strength=jnp.array([0.05, 0.2, 0.45])
            )

    def test_an_out_of_range_array_can_be_escalated_to_an_error(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError, match="validity range"):
            dist.get_D("Nd", pH=3.0, ionic_strength=np.array([0.1, 3.0]))

    def test_the_report_names_the_sign_change_when_it_is_crossed(self):
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError, match="changes sign"):
            dist.get_D("Nd", pH=3.0, ionic_strength=3.0)
        # Out of range but below the sign change: no inversion claim.
        mild = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError) as exc:
            mild.get_D("Nd", pH=3.0, ionic_strength=1.0)
        assert "validity range" in str(exc.value)
        assert "changes sign" not in str(exc.value)

    def test_array_of_D_values_is_corrected_elementwise(self):
        """An array ionic strength still produces an elementwise correction,
        clamped where it is out of range."""
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        D_plain = float(dist.get_D("Nd", pH=3.0))
        I = jnp.array([0.1, 0.5, 3.0])
        D = np.asarray(dist.get_D("Nd", pH=3.0, ionic_strength=I))
        assert D.shape == (3,)
        assert np.all(D / D_plain < 1.0)
        assert D[1] == pytest.approx(D[2], rel=1e-12)  # both clamped at 0.5


# =============================================================================
# Differentiability (#194, #195)
# =============================================================================

class TestGradients:
    """get_D stays grad- and jit-safe through both mechanisms."""

    def test_grad_wrt_ph_matches_finite_difference(self):
        dist = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        f = lambda pH: dist.get_D("Nd", pH=pH)
        pH0, h = 3.0, 1e-5
        g = float(jax.grad(f)(pH0))
        fd = (float(f(pH0 + h)) - float(f(pH0 - h))) / (2 * h)
        assert np.isfinite(g)
        assert g == pytest.approx(fd, rel=1e-5)

    def test_grad_wrt_nitrate_matches_finite_difference(self):
        dist = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        f = lambda c: dist.get_D("Nd", nitrate_conc=c)
        c0, h = 3.0, 1e-5
        g = float(jax.grad(f)(c0))
        fd = (float(f(c0 + h)) - float(f(c0 - h))) / (2 * h)
        assert np.isfinite(g)
        assert g > 0.0  # D rises with nitrate
        assert g == pytest.approx(fd, rel=1e-5)

    def test_grad_wrt_ionic_strength_is_finite_and_negative(self):
        """In range the correction reduces D, so dD/dI < 0 and matches a finite
        difference. Probed across the whole Davies range and, critically, ALSO
        above the sign-change ionic strength, where the pre-fix code returned a
        POSITIVE gradient (the correction was increasing D). Probing only
        I = 0.1 M never sees that (#194)."""
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        f = lambda I: dist.get_D("Nd", pH=3.0, ionic_strength=I)

        # In range: negative, and equal to a central difference.
        for I0 in (0.05, 0.1, 0.2):
            h = 1e-6
            g = float(jax.grad(f)(I0))
            fd = (float(f(I0 + h)) - float(f(I0 - h))) / (2 * h)
            assert np.isfinite(g)
            assert g < 0.0, f"dD/dI should be negative at I={I0}"
            assert g == pytest.approx(fd, rel=1e-4)

        # Above the range, and above the sign change: the correction is
        # saturated, so the gradient is exactly zero. It is emphatically NOT
        # positive, which is what an inverted correction would give.
        for I0 in (0.6, 1.0, 3.0, 4.0):
            g = float(jax.grad(f)(I0))
            assert np.isfinite(g)
            assert g == 0.0, f"dD/dI should be flat past the range at I={I0}"

    def test_grad_is_positive_only_when_extrapolation_is_requested(self):
        """The inverted branch still exists, and still has the sign the bug
        report measured -- but only for a caller who asked for it (#194)."""
        opted_in = REEDistribution(
            extractant="D2EHPA",
            elements=("Nd",),
            on_out_of_range="ignore",
            extrapolate_activity_model=True,
        )
        g = float(
            jax.grad(lambda I: opted_in.get_D("Nd", pH=3.0, ionic_strength=I))(
                3.0
            )
        )
        assert np.isfinite(g)
        assert g > 0.0

    def test_traced_ionic_strength_cannot_silently_invert_D(self):
        """A tracer cannot be range-checked, so the guard is arithmetic rather
        than a check: the ionic strength fed to the activity model is clamped
        at the model's documented limit. grad still works, and the result is
        identical to the concrete clamped value -- never the inverted one
        (#194)."""
        dist = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        f = lambda I: dist.get_D("Nd", pH=3.0, ionic_strength=I)

        g = jax.grad(f)(3.0)
        assert jnp.isfinite(g)

        D_plain = float(dist.get_D("Nd", pH=3.0))
        traced = float(jax.jit(f)(3.0))
        clamped = float(f(DAVIES_MAX_IONIC_STRENGTH))
        assert traced == pytest.approx(clamped, rel=1e-12)
        assert traced / D_plain < 1.0

    def test_traced_ionic_strength_is_reported_as_unverifiable(self):
        """The old behaviour was a *silent* skip: on_out_of_range='raise' with
        a traced I produced no raise at all. A tracer is now reported as
        exactly what it is -- a value whose range could not be checked (#194).
        """
        strict = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="raise"
        )
        with pytest.raises(ValueError, match="tracer"):
            jax.grad(
                lambda I: strict.get_D("Nd", pH=3.0, ionic_strength=I)
            )(3.0)

        warning_mode = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        with pytest.warns(UserWarning, match="cannot be range-checked"):
            jax.jit(
                lambda I: warning_mode.get_D("Nd", pH=3.0, ionic_strength=I)
            )(3.0)

        quiet = REEDistribution(
            extractant="D2EHPA", elements=("Nd",), on_out_of_range="ignore"
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            jax.jit(
                lambda I: quiet.get_D("Nd", pH=3.0, ionic_strength=I)
            )(3.0)

    def test_jit_of_both_mechanisms(self):
        acidic = REEDistribution(extractant="D2EHPA", elements=("Nd",))
        tbp = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )
        f_acid = jax.jit(lambda pH: acidic.get_D("Nd", pH=pH))
        f_tbp = jax.jit(lambda c: tbp.get_D("Nd", nitrate_conc=c))
        assert float(f_acid(3.0)) == pytest.approx(
            float(acidic.get_D("Nd", pH=3.0)), rel=1e-10
        )
        assert float(f_tbp(4.0)) == pytest.approx(
            float(tbp.get_D("Nd", nitrate_conc=4.0)), rel=1e-10
        )


# =============================================================================
# D8 - every flowsheet that takes an extractant can take a solvating one
# =============================================================================

class TestSolvatingExtractantInFlowsheets:
    """#195 made TBP raise unless a nitrate concentration reaches
    REEDistribution. Two flowsheets were not threaded, which left TBP
    constructible only through the other two -- with no escape hatch, because
    their Params classes had no nitrate_conc field at all."""

    @staticmethod
    def _feed():
        return make_stream(
            {"H2O": 1000.0, "Nd": 1.0, "Dy": 0.5}, 298.15, 101325.0
        )

    def test_extract_scrub_strip_params_carry_the_solvating_fields(self):
        p = ExtractScrubStripParams(
            extractant="TBP",
            elements=("Nd", "Dy"),
            target_elements=("Dy",),
        )
        assert p.nitrate_conc is None
        assert p.mechanism is None

    def test_extract_scrub_strip_constructs_and_runs_with_TBP(self):
        p = ExtractScrubStripParams(
            extractant="TBP",
            elements=("Nd", "Dy"),
            target_elements=("Dy",),
            nitrate_conc=3.0,
            extractant_conc=1.1,  # ~30% v/v, see extractants.yaml
        )
        circuit = ExtractScrubStripCircuit(p)
        results = circuit(self._feed())
        assert np.isfinite(float(results["target_purity"]))
        for elem in ("Nd", "Dy"):
            assert np.isfinite(float(results["product_purity"][elem]))

    def test_extract_scrub_strip_still_refuses_TBP_without_nitrate(self):
        """Threading the parameter must not reintroduce the silent pH fallback."""
        with pytest.raises(ValueError, match=r"TBP.*requires a nitrate"):
            ExtractScrubStripCircuit(
                ExtractScrubStripParams(
                    extractant="TBP",
                    elements=("Nd", "Dy"),
                    target_elements=("Dy",),
                )
            )

    def test_separation_train_params_carry_the_solvating_fields(self):
        p = SeparationTrainParams(elements=("La", "Nd", "Dy"), extractant="TBP")
        assert p.nitrate_conc is None
        assert p.mechanism is None

    def test_full_train_constructs_and_runs_with_TBP(self):
        p = SeparationTrainParams(
            elements=("La", "Nd", "Dy"),
            extractant="TBP",
            include_ce_removal=False,
            nitrate_conc=3.0,
        )
        train = FullSeparationTrain(p)
        results = train(
            make_stream(
                {"H2O": 1000.0, "La": 1.0, "Nd": 1.0, "Dy": 0.5},
                298.15,
                101325.0,
            )
        )
        assert set(results["products"]) >= {"light_REE", "heavy_REE"}

    def test_full_train_still_refuses_TBP_without_nitrate(self):
        with pytest.raises(ValueError, match=r"TBP.*requires a nitrate"):
            FullSeparationTrain(
                SeparationTrainParams(
                    elements=("La", "Nd", "Dy"),
                    extractant="TBP",
                    include_ce_removal=False,
                )
            )

    def test_mechanism_override_reaches_every_section(self):
        """The override must reach all three sections, or a circuit silently
        runs mixed mechanisms: the scrubber and stripper on cation exchange
        while the extractor stays solvating. REEExtractorParams and
        MixerSettlerParams carry a `mechanism` field for this (#195).

        TBP can no longer serve as the subject unaided -- its ph_coefficients
        block was DELETED, so overriding it to cation_exchange now raises (see
        ``test_tbp_cation_exchange_opt_in_now_raises``, and the companion test
        below that the flowsheet refuses it too). To keep testing the
        *threading* rather than the deletion, a pH block is temporarily grafted
        back onto the loaded record and removed again afterwards. The grafted
        numbers are arbitrary; only the mechanism plumbing is under test."""
        tbp = get_extractant("TBP")
        assert tbp.ph_coefficients is None  # the deletion, restated here
        from difflow_ree.database import PHCoefficients

        tbp.ph_coefficients = {
            "Nd": PHCoefficients(a=-1.40, b=0.60, c=0.0, d=0.0),
            "Dy": PHCoefficients(a=0.00, b=0.85, c=0.0, d=0.0),
        }
        try:
            circuit = ExtractScrubStripCircuit(
                ExtractScrubStripParams(
                    extractant="TBP",
                    elements=("Nd", "Dy"),
                    target_elements=("Dy",),
                    nitrate_conc=3.0,
                    mechanism="cation_exchange",
                )
            )
            assert circuit._scrubber._distribution.mechanism == "cation_exchange"
            assert circuit._stripper._distribution.mechanism == "cation_exchange"
            assert circuit._extractor._distribution.mechanism == "cation_exchange"
            assert circuit._extractor.params.mechanism == "cation_exchange"
        finally:
            tbp.ph_coefficients = None

    def test_mechanism_override_on_tbp_is_refused_by_the_flowsheet_too(self):
        """The deletion is not bypassable through a flowsheet: the same
        ValueError surfaces from circuit construction."""
        with pytest.raises(ValueError, match="ph_coefficients"):
            ExtractScrubStripCircuit(
                ExtractScrubStripParams(
                    extractant="TBP",
                    elements=("Nd", "Dy"),
                    target_elements=("Dy",),
                    nitrate_conc=3.0,
                    mechanism="cation_exchange",
                )
            )

    def test_the_two_already_threaded_flowsheets_still_work(self):
        """Regression guard on the pair that #195 did thread."""
        strip = ExtractStripCircuit(
            ExtractStripParams(
                extractant="TBP",
                elements=("Nd", "Dy"),
                nitrate_conc=3.0,
                extractant_conc=1.1,
            )
        )
        assert strip is not None
        shell = SplitShellCascade(
            SplitShellParams(
                extractant="TBP",
                elements=("Nd", "Dy"),
                nitrate_conc=3.0,
                extractant_conc=1.1,
            )
        )
        assert shell._distribution.mechanism == "solvating"


# =============================================================================
# D10 - the custom-extractant path must not lose the dimer basis (#191)
# =============================================================================

class TestCustomExtractantStoichiometryBasis:
    """create_custom_extractant gained the #195 mechanism fields but not
    stoichiometry_basis, so a custom dimeric extractant silently reported
    m = 3 while every built-in acidic extractant reports 6 -- the factor-of-two
    capacity error of #191, reintroduced through the custom path."""

    @staticmethod
    def _make(**kw):
        return create_custom_extractant(
            name="MyDimer",
            full_name="My Dimeric Phosphoric Acid",
            formula="C10H20O4P",
            molecular_weight=250.0,
            ph_coefficients={"Nd": {"a": -7.5, "b": 2.4, "c": 0.01, "d": 0.0}},
            temperature_coefficients={"Nd": -1700},
            **kw,
        )

    def test_custom_dimeric_extractant_reports_six_monomers_per_ree(self):
        ext = self._make(stoichiometry_basis="dimer")
        assert ext.stoichiometry_basis == "dimer"
        assert ext.monomers_per_ree == 6
        assert ext.max_loading == pytest.approx(1.0 / 6.0)

    def test_it_agrees_with_the_built_in_acidic_extractants(self):
        """The whole point: a custom D2EHPA-like record must not disagree with
        the real one about capacity."""
        custom = self._make(
            stoichiometry_basis="dimer",
            stoichiometry_protons=3,
            stoichiometry_extractant=3,
        )
        for name in ("D2EHPA", "PC88A", "Cyanex272"):
            builtin = get_extractant(name)
            assert builtin.stoichiometry_basis == "dimer"
            assert custom.monomers_per_ree == builtin.monomers_per_ree
            assert custom.max_loading == pytest.approx(builtin.max_loading)

    def test_monomer_basis_is_the_default_and_gives_three(self):
        ext = self._make()
        assert ext.stoichiometry_basis == "monomer"
        assert ext.monomers_per_ree == 3
        assert get_extractant("TBP").stoichiometry_basis == "monomer"

    def test_bad_basis_is_rejected(self):
        with pytest.raises(ValueError, match="stoichiometry_basis"):
            self._make(stoichiometry_basis="trimer")


# =============================================================================
# D9 - the TBP coefficients mean what the YAML now says they mean
# =============================================================================

class TestTBPCoefficientInterpretation:
    """TBP's nitrate coefficients are now REFITTED FROM PRIMARY LITERATURE, not
    hand-tuned. Fit basis: Kraikaew, Srinuttrakul & Chayavadhanakur (2005),
    J. Metals, Materials and Minerals 15(2), 89-95, Table 1 (1.0 M TBP in
    kerosene, 0.2001 N free acid, 35 C), corrected to the record's reference
    (3.0 M NO3-, 1.0 M TBP, 298.15 K); heat of extraction from Ganesh & Pandey
    (2019), J. Rad. Nucl. Appl. 4(2), 109-115, dH_Sm = -43.3 kJ/mol.

    `a` is still log10(D) AT reference_nitrate and reference_concentration --
    that reading is unchanged; what changed is that the numbers are now
    measured. These tests pin the three defects that were fixed."""

    def test_a_is_log10_D_at_the_reference_nitrate_and_concentration(self):
        ext = get_extractant("TBP")
        dist = REEDistribution(
            extractant="TBP",
            elements=("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y"),
            concentration=ext.reference_concentration,
            nitrate_conc=ext.reference_nitrate,
        )
        # The exact numbers written into data/extractants.yaml.
        expected = {
            "La": 0.0234, "Ce": 0.0324, "Pr": 0.0603, "Nd": 0.0724,
            "Sm": 0.1202, "Eu": 0.1380, "Gd": 0.1738, "Tb": 0.2042,
            "Dy": 0.2399, "Y": 0.2138,
        }
        for elem, D in expected.items():
            assert float(dist.get_D(elem)) == pytest.approx(D, rel=5e-3)
        # Every one is below 1: TBP is a weak extractant and the data now
        # says so. The old block put D_Gd at 1.0 and D_Dy at 3.16.
        assert all(D < 1.0 for D in expected.values())

    def test_D_Dy_at_the_reference_is_the_measured_024_not_the_old_316(self):
        """Defect 2 of 3. The old record said D_Dy = 3.16 at the reference --
        a distribution ratio TBP only reaches with a strong salting agent.
        Corrected from Kraikaew et al. Table 1 it is 0.24, a 13x overstatement
        removed."""
        ext = get_extractant("TBP")
        dist = REEDistribution(
            extractant="TBP",
            elements=("Dy",),
            concentration=ext.reference_concentration,
            nitrate_conc=ext.reference_nitrate,
        )
        assert float(dist.get_D("Dy")) == pytest.approx(0.24, abs=0.005)

    def test_temperature_coefficients_are_positive_so_extraction_is_exothermic(
        self,
    ):
        """Defect 3 of 3. Under log10 D = ... + d*(1/T - 1/Tref) with
        d = -dH/(2.303 R), a NEGATIVE d means an ENDOTHERMIC dH. Every TBP `d`
        used to be negative, which asserted that TBP extraction of REE nitrates
        is endothermic. It is exothermic (Ganesh & Pandey 2019, dH_Sm = -43.3
        kJ/mol; corroborated by Jorjani & Shahbazi 2016, extraction falling
        from 25 to 55 C). Every `d` must therefore be positive, and D must FALL
        as T rises."""
        ext = get_extractant("TBP")
        elements = (
            "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Y",
        )
        for elem in elements:
            d = ext.temperature_coefficients[elem]
            assert d > 0, f"TBP d for {elem} is {d}: that is endothermic"
            # dH = -2.303 R d; must be exothermic and of the measured size.
            dH_kJ = -2.303 * 8.314 * d / 1000.0
            assert dH_kJ < 0
            assert dH_kJ == pytest.approx(-44.1, abs=1.5)

        # And the observable consequence: D falls with temperature.
        dist = REEDistribution(
            extractant="TBP",
            elements=("Nd",),
            concentration=ext.reference_concentration,
            nitrate_conc=ext.reference_nitrate,
        )
        assert float(dist.get_D("Nd", T=323.15)) < float(
            dist.get_D("Nd", T=298.15)
        )

    def test_d_sits_inside_the_documented_ambiguity_range(self):
        """Ganesh & Pandey's Table 1 appears to have its dH and slope columns
        transposed between its two rows, so dH_TBP is either -43.3 or -61.2
        kJ/mol, i.e. d = +2261 or +3197 K -- a +/-40% ambiguity that cannot be
        resolved from the paper. The YAML documents the range; the value used
        must lie in it."""
        for d in get_extractant("TBP").temperature_coefficients.values():
            assert 2260.0 <= d <= 3200.0

    def test_the_default_extractant_conc_knocks_TBP_down_eightfold(self):
        """The 0.5 M default every Params class carries is a cation-exchange
        default; for TBP (reference 1.0 M, n = 3) it is an 8x reduction, and
        0.5 M is well below the ~1.1 M of the usual 30% v/v."""
        ext = get_extractant("TBP")
        assert ext.reference_concentration == pytest.approx(1.0)
        assert ext.concentration_exponent == pytest.approx(3.0)

        default = REEDistribution(
            extractant="TBP", elements=("Nd",), nitrate_conc=3.0
        )  # concentration=0.5, the shared default
        assert default.concentration == pytest.approx(0.5)
        at_ref = REEDistribution(
            extractant="TBP",
            elements=("Nd",),
            concentration=1.0,
            nitrate_conc=3.0,
        )
        assert float(default.get_D("Nd")) / float(at_ref.get_D("Nd")) == (
            pytest.approx(0.125, rel=1e-10)
        )

        # 30% v/v of neat TBP, from the density and MW on the record.
        neat = 1000.0 * ext.density / ext.molecular_weight
        assert 0.30 * neat == pytest.approx(1.10, abs=0.02)

    def test_the_selectivity_spread_is_the_measured_10x_not_the_old_100x(self):
        """Defect 1 of 3, and the most important one. The old record implied
        D_Dy/D_La = 100 at the reference -- an acidic-organophosphorus
        selectivity pattern that looks carried over from a cation-exchange
        record. Kraikaew et al. Table 1 measures 10.2. TBP's REE selectivity is
        genuinely small, which is why a TBP separation needs very many
        stages."""
        ext = get_extractant("TBP")
        nc = ext.nitrate_coefficients
        ratio = 10.0 ** (nc["Dy"].a - nc["La"].a)
        assert ratio == pytest.approx(10.2, rel=5e-3)  # 2 s.f.

        # Same thing through the public path, not just the raw record.
        dist = REEDistribution(
            extractant="TBP",
            elements=("La", "Dy"),
            concentration=ext.reference_concentration,
            nitrate_conc=ext.reference_nitrate,
        )
        assert float(dist.get_D("Dy")) / float(dist.get_D("La")) == (
            pytest.approx(10.2, rel=5e-3)
        )

    def test_mean_adjacent_pair_separation_factor_is_about_1_3(self):
        """The consequence of the 10x spread: a mean adjacent-pair separation
        factor of 1.29, inside the 1.1-1.5 reported for TBP. Acidic
        organophosphorus extractants sit at 1.5-4 per pair; TBP does not, and
        the old 100x spread wrongly put it there (it implied 1.67 per pair).

        "Adjacent pair" means adjacent in the lanthanide series, i.e. per unit
        atomic number, so La -> Dy is 9 steps and NOT the 8 gaps in the tracked
        element tuple: Pm (Z = 61) is a real lanthanide that this package does
        not track. Counting tuple gaps would inflate the per-pair factor to
        1.34 and make it look worse against the literature range than it is.
        """
        nc = get_extractant("TBP").nitrate_coefficients
        Z = {"La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62,
             "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66}
        series = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy")
        # Per-atomic-number factor for each consecutive tracked pair, so the
        # Nd -> Sm gap is spread over its two steps rather than counted as one.
        per_step = [
            10.0 ** ((nc[hi].a - nc[lo].a) / (Z[hi] - Z[lo]))
            for lo, hi in zip(series[:-1], series[1:])
        ]
        assert all(sf > 1.0 for sf in per_step)  # monotone across the series
        n_steps = Z["Dy"] - Z["La"]
        assert n_steps == 9
        spread = 10.0 ** (nc["Dy"].a - nc["La"].a)
        geo_mean = spread ** (1.0 / n_steps)
        assert geo_mean == pytest.approx(1.29, abs=0.01)
        # Inside the 1.1-1.5 reported for TBP, and well below the 1.5-4 of the
        # acidic extractants. The old record implied 1.67, above both.
        assert 1.1 <= geo_mean <= 1.5
        assert 10.0 ** (2.00 / n_steps) == pytest.approx(1.67, abs=0.01)
        assert geo_mean**n_steps == pytest.approx(10.2, rel=5e-3)

    def test_nitrate_slope_is_the_stoichiometric_three_for_every_element(self):
        """b = 3.0 is the stoichiometric 3 of RE(NO3)3 + 3 TBP, NOT a fitted
        nitrate slope: no measured d log D / d log[NO3-] in a neutral-salt
        system was found. The only corroboration is indirect -- Ganesh & Pandey
        Fig. 1 measured the TBP order as 2.81 (R^2 = 0.992). The old
        per-element 2.50-2.85 trend had no support and is removed, so all ten
        elements share one value."""
        recs = get_extractant("TBP").nitrate_coefficients
        for rec in recs.values():
            assert rec.b == 3.0
            assert rec.c == 0.0  # no curvature is claimed
        assert len({rec.b for rec in recs.values()}) == 1
