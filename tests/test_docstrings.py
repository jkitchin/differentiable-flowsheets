"""Tests for difflow.docstrings.

The parser exists so that :mod:`difflow.catalog` can report what each
parameter *means*; before it, every one of the project's ``Params``
classes reached the catalog with ``description=None`` (#213). What is
worth pinning down here is that the forms actually used across the
project are read --- the wrapped descriptions, the sub-headings some
classes group their fields under, the comments beside the fields added
after a class was first written --- and that prose in an ``Attributes:``
section is not mistaken for one of its entries.
"""

from dataclasses import dataclass

import pytest

from difflow.docstrings import (
    attribute_docs,
    field_comments,
    parse_attributes,
)
from difflow.params_mixin import ParamsMixin


@dataclass
class SampleParams(ParamsMixin):
    """Parameters for something.

    Attributes:
        T: Flash temperature (K)
        P: Flash pressure (Pa)
        wrapped: A description that runs onto
            a second line, and keeps going.
        efficiency: Murphree stage efficiency (0-1)

        Group heading:
        grouped: Documented under a sub-heading (s)
    """

    T: float
    P: float = 101325.0
    wrapped: str = ""
    efficiency: float = 0.95
    grouped: float = 1.0
    commented: float = 0.0  # Trailing comment (m^3)
    # A block comment above the field,
    # continued on a second line.
    blocked: float = 0.0
    undocumented: float = 0.0


# =============================================================================
# Docstring sections
# =============================================================================


class TestParseAttributes:
    def test_reads_the_attributes_section(self):
        docs = parse_attributes(SampleParams.__doc__)
        assert docs["T"] == "Flash temperature (K)"
        assert docs["P"] == "Flash pressure (Pa)"

    def test_joins_wrapped_descriptions(self):
        docs = parse_attributes(SampleParams.__doc__)
        assert docs["wrapped"] == (
            "A description that runs onto a second line, and keeps going."
        )

    def test_a_blank_line_does_not_end_the_section(self):
        """Several classes group their attributes with a blank line."""
        assert "grouped" in parse_attributes(SampleParams.__doc__)

    def test_a_later_section_does_end_it(self):
        docs = parse_attributes(
            "Summary.\n\n"
            "    Attributes:\n"
            "        a: first\n\n"
            "    Notes:\n"
            "        not_an_attribute: prose\n"
        )
        assert set(docs) == {"a"}

    def test_numpy_style_is_understood(self):
        """The project writes Google style, but a plugin need not."""
        docs = parse_attributes(
            "Summary.\n\n"
            "    Parameters\n"
            "    ----------\n"
            "    T : float\n"
            "        Temperature (K)\n"
            "    P : float\n"
            "        Pressure (Pa)\n\n"
            "    Returns\n"
            "    -------\n"
            "    nothing : None\n"
            "        the section ends at the next heading\n"
        )
        assert docs == {"T": "Temperature (K)", "P": "Pressure (Pa)"}

    def test_google_style_types_are_not_mistaken_for_names(self):
        docs = parse_attributes(
            "Summary.\n\n    Attributes:\n        T (float): Temperature\n"
        )
        assert docs == {"T": "Temperature"}

    def test_prose_in_the_section_is_not_glued_onto_an_entry(self):
        """Some sections mix in bullets; they must not extend a description."""
        docs = parse_attributes(
            "Summary.\n\n"
            "    Attributes:\n"
            "        a: first\n"
            "        The model assumes:\n"
            "        - something about the model\n"
        )
        assert docs["a"] == "first"

    def test_no_docstring_is_not_an_error(self):
        assert parse_attributes(None) == {}
        assert parse_attributes("") == {}
        assert parse_attributes("Just a summary.") == {}


# =============================================================================
# Comments
# =============================================================================


class TestFieldComments:
    def test_a_trailing_comment_is_a_description(self):
        assert field_comments(SampleParams)["commented"] == "Trailing comment (m^3)"

    def test_a_block_above_the_field_is_a_description(self):
        assert field_comments(SampleParams)["blocked"] == (
            "A block comment above the field, continued on a second line."
        )

    def test_an_undocumented_field_is_absent(self):
        assert "undocumented" not in field_comments(SampleParams)

    def test_banners_and_directives_are_not_descriptions(self):
        source = (
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Banner:\n"
            "    # ===== Cycle timing =====\n"
            "    a: float = 0.0\n"
            "    b: float = 0.0  # noqa: E501\n"
            "    c: float = 0.0  # TODO: work out what this is\n"
        )
        namespace: dict = {}
        exec(compile(source, "banner_example.py", "exec"), namespace)
        # the source is not on disk, so nothing can be read at all
        assert field_comments(namespace["Banner"]) == {}

    def test_a_class_with_no_source_is_not_an_error(self):
        cls = type("Dynamic", (), {"__annotations__": {"x": float}})
        assert field_comments(cls) == {}


# =============================================================================
# The combined view
# =============================================================================


class TestAttributeDocs:
    def test_docstring_and_comments_are_both_used(self):
        docs = attribute_docs(SampleParams)
        assert docs["T"] == "Flash temperature (K)"
        assert docs["commented"] == "Trailing comment (m^3)"

    def test_the_docstring_wins_over_a_comment(self):
        @dataclass
        class Both(ParamsMixin):
            """Summary.

            Attributes:
                x: from the docstring
            """

            x: float = 0.0  # from the comment

        assert attribute_docs(Both)["x"] == "from the docstring"

    def test_only_real_fields_are_reported(self):
        docs = attribute_docs(SampleParams)
        assert "Group heading" not in docs
        assert "undocumented" not in docs
        assert set(docs) <= {"T", "P", "wrapped", "efficiency", "grouped",
                             "commented", "blocked"}

    def test_inherited_fields_keep_their_documentation(self):
        @dataclass
        class Derived(SampleParams):
            """Parameters for something more specific.

            Attributes:
                extra: An added field (s)
            """

            extra: float = 0.0

        docs = attribute_docs(Derived)
        assert docs["T"] == "Flash temperature (K)"
        assert docs["extra"] == "An added field (s)"

    def test_a_subclass_can_redocument_a_field(self):
        @dataclass
        class Redocumented(SampleParams):
            """Summary.

            Attributes:
                T: Inlet temperature, not the flash temperature (K)
            """

        assert attribute_docs(Redocumented)["T"].startswith("Inlet temperature")

    def test_comments_can_be_turned_off(self):
        docs = attribute_docs(SampleParams, comments=False)
        assert "T" in docs
        assert "commented" not in docs

    def test_a_plain_class_reports_what_it_documents(self):
        """Not a dataclass, so there are no fields to filter by."""

        class Plain:
            """Summary.

            Attributes:
                a: first (K)
            """

        assert attribute_docs(Plain) == {"a": "first (K)"}


class TestTheModuleDocumentsItself:
    def test_its_own_examples_run(self):
        """The suite does not run with `--doctest-modules`, so the
        examples in :mod:`difflow.docstrings` would otherwise rot."""
        import doctest

        import difflow.docstrings

        result = doctest.testmod(difflow.docstrings, verbose=False)
        assert result.attempted > 0, "no examples found to check"
        assert result.failed == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
