"""REEScrubber's target_elements and scrub_type (#288).

`ScrubberParams.target_elements` and `ScrubberParams.scrub_type` both looked
like they steered the scrubbing calculation and neither did: every element
goes through the same `kremser_two_inlet` solve on its own D, and `__call__`
never read `scrub_type` at all. A student setting up a separation train asked
which elements belong in `target_elements`, and no answer changed a number.

The resolution (#288) keeps the behaviour and stops it being silent:

* `target_elements` is documented as a REPORTING LABEL, made optional, and
  checked against `elements` -- an untracked name used to label nothing.
* `scrub_type` is deprecated and warns: what it claimed to switch on is
  already carried by `pH` and by the REE content of the scrub stream, which
  the two-inlet Kremser takes as a boundary condition (#284).

These tests pin that. The first two are the ones the issue asks for: they
fail if either field ever quietly starts (or stops) mattering.
"""

import warnings

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from difflow.streams import make_stream, get_flows
from difflow_ree import REEScrubber, ScrubberParams, ScrubTypeDeprecationWarning
from difflow_ree.flowsheets.extract_scrub_strip import ExtractScrubStripParams


ELEMENTS = ("La", "Ce", "Nd", "Dy")


def _streams():
    loaded_organic = make_stream(
        {"D2EHPA": 0.5, "kerosene": 1.0,
         "La": 0.10, "Ce": 0.10, "Nd": 0.10, "Dy": 0.10},
        298.15, 101325.0,
    )
    scrub_solution = make_stream({"H2O": 1.0}, 298.15, 101325.0)
    return loaded_organic, scrub_solution


def _run(**kwargs):
    params = ScrubberParams(
        n_stages=5, extractant="D2EHPA", elements=ELEMENTS, pH=1.0, **kwargs
    )
    org, scrub = _streams()
    return REEScrubber(params)(org, scrub)


class TestTargetElementsIsALabel:
    """target_elements labels the diagnostics; it moves no flow (#288)."""

    def test_outlet_flows_do_not_depend_on_target_elements(self):
        """Documented behaviour: same streams whatever is labelled.

        If this ever fails, target_elements has grown an effect on the
        calculation and the docstring -- which promises the opposite -- has
        to be rewritten rather than the test relaxed.
        """
        liquor_a, org_a, _ = _run(target_elements=("Nd", "Dy"))
        liquor_b, org_b, _ = _run(target_elements=("La",))
        liquor_c, org_c, _ = _run()  # no labels at all

        for elem in ELEMENTS:
            for a, b, c in ((liquor_a, liquor_b, liquor_c),
                            (org_a, org_b, org_c)):
                fa = get_flows(a)[elem]
                assert jnp.allclose(fa, get_flows(b)[elem])
                assert jnp.allclose(fa, get_flows(c)[elem])

    def test_the_diagnostics_do_depend_on_it(self):
        """It is not inert either: it sorts the reported elements."""
        _, _, info_a = _run(target_elements=("Nd", "Dy"))
        _, _, info_b = _run(target_elements=("La",))

        assert set(info_a["target_retained"]) == {"Nd", "Dy"}
        assert set(info_a["impurity_removed"]) == {"La", "Ce"}
        assert set(info_b["target_retained"]) == {"La"}
        assert set(info_b["impurity_removed"]) == {"Ce", "Nd", "Dy"}

    def test_it_is_optional(self):
        """The label is not needed to compute anything, so it has a default."""
        params = ScrubberParams(n_stages=5, extractant="D2EHPA", elements=ELEMENTS)
        assert params.target_elements == ()

        _, _, info = _run()
        assert info["target_retained"] == {}
        assert set(info["impurity_removed"]) == set(ELEMENTS)

    def test_an_untracked_target_is_rejected(self):
        """The student's case: target_elements=("Y",) with no Y in elements.

        It used to label nothing and report an empty target_retained, which
        reads like "the scrub retained none of the target".
        """
        with pytest.raises(ValueError, match="not in elements"):
            ScrubberParams(
                n_stages=5, extractant="D2EHPA",
                elements=ELEMENTS, target_elements=("Y",),
            )

    def test_a_bare_string_is_rejected(self):
        """("Nd",) vs "Nd": the second would label N and d."""
        with pytest.raises(TypeError, match="tuple of element names"):
            ScrubberParams(
                n_stages=5, extractant="D2EHPA",
                elements=ELEMENTS, target_elements="Nd",
            )

    def test_a_list_is_accepted_and_normalised(self):
        params = ScrubberParams(
            n_stages=5, extractant="D2EHPA",
            elements=ELEMENTS, target_elements=["Nd", "Dy"],
        )
        assert params.target_elements == ("Nd", "Dy")

    def test_the_circuit_checks_its_labels_too(self):
        """ExtractScrubStripParams.target_elements is the same kind of label."""
        with pytest.raises(ValueError, match="not in elements"):
            ExtractScrubStripParams(
                extractant="D2EHPA", elements=ELEMENTS, target_elements=("Y",),
            )
        assert ExtractScrubStripParams(
            extractant="D2EHPA", elements=ELEMENTS,
        ).target_elements == ()


class TestScrubTypeIsDeprecated:
    """scrub_type never reached the calculation, and now says so (#288)."""

    def test_setting_it_warns(self):
        with pytest.warns(ScrubTypeDeprecationWarning, match="deprecated"):
            ScrubberParams(
                n_stages=5, extractant="D2EHPA",
                elements=ELEMENTS, scrub_type="ree",
            )

    def test_not_setting_it_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ScrubTypeDeprecationWarning)
            params = ScrubberParams(
                n_stages=5, extractant="D2EHPA", elements=ELEMENTS
            )
        assert params.scrub_type is None

    def test_an_unknown_value_is_rejected(self):
        with pytest.raises(ValueError, match="not one of"):
            ScrubberParams(
                n_stages=5, extractant="D2EHPA",
                elements=ELEMENTS, scrub_type="brine",
            )

    def test_every_value_gives_the_same_answer(self):
        """The reason it is deprecated rather than implemented."""
        results = []
        for scrub_type in ("acid", "ree", "water"):
            with pytest.warns(ScrubTypeDeprecationWarning):
                params = ScrubberParams(
                    n_stages=5, extractant="D2EHPA", elements=ELEMENTS,
                    pH=1.0, scrub_type=scrub_type,
                )
            org, scrub = _streams()
            results.append(REEScrubber(params)(org, scrub)[1])

        for elem in ELEMENTS:
            ref = get_flows(results[0])[elem]
            for other in results[1:]:
                assert jnp.allclose(ref, get_flows(other)[elem])

    def test_a_ree_bearing_scrub_is_what_ree_meant(self):
        """The "ree" mode's physics is the stream, not a flag (#284).

        A scrub solution carrying REE -- a refluxed strip liquor -- enters
        the two-inlet Kremser as F_scrub_in and does change the answer. That
        is the boundary condition the deprecated flag advertised.
        """
        params = ScrubberParams(
            n_stages=5, extractant="D2EHPA", elements=ELEMENTS, pH=1.0
        )
        scrubber = REEScrubber(params)
        org, clean = _streams()
        refluxed = make_stream(
            {"H2O": 1.0, "Nd": 0.05}, 298.15, 101325.0
        )

        _, org_clean, _ = scrubber(org, clean)
        _, org_reflux, _ = scrubber(org, refluxed)

        assert not jnp.allclose(
            get_flows(org_clean)["Nd"], get_flows(org_reflux)["Nd"]
        )

    def test_the_class_metadata_no_longer_claims_a_mode(self):
        """numerical_method / assumptions described what the code did not do."""
        text = REEScrubber.numerical_method + " ".join(REEScrubber.assumptions)
        assert "scrub-type-dependent" not in text
        assert "user-selected type" not in text


class TestDroppedSpeciesAreReported:
    """Outlets are rebuilt from carriers + elements, so the rest is lost."""

    def test_an_untracked_ree_is_reported(self):
        """`elements` must cover every REE present, or mass is not conserved."""
        org = make_stream(
            {"D2EHPA": 0.5, "kerosene": 1.0, "Nd": 0.10, "Dy": 0.10},
            298.15, 101325.0,
        )
        scrub = make_stream({"H2O": 1.0}, 298.15, 101325.0)
        params = ScrubberParams(
            n_stages=5, extractant="D2EHPA", elements=("Nd",), pH=1.0
        )
        liquor, scrubbed, info = REEScrubber(params)(org, scrub)

        assert info["dropped_species"] == ("Dy",)
        assert "Dy" not in get_flows(liquor)
        assert "Dy" not in get_flows(scrubbed)

    def test_nothing_is_reported_when_everything_is_tracked(self):
        org, scrub = _streams()
        params = ScrubberParams(
            n_stages=5, extractant="D2EHPA", elements=ELEMENTS, pH=1.0
        )
        assert REEScrubber(params)(org, scrub)[2]["dropped_species"] == ()
