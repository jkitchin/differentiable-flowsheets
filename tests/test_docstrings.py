"""Tests for difflow.docstrings: the catalog's parameter descriptions (#213)."""

from dataclasses import dataclass

from difflow.docstrings import attribute_docs, field_comments, parse_attributes
from difflow.params_mixin import ParamsMixin


@dataclass
class SampleParams(ParamsMixin):
    """Parameters for something.

    Attributes:
        T: Flash temperature (K)
        wrapped: A description that runs onto
            a second line.

        Group heading:
        grouped: Documented under a sub-heading (s)
    """

    T: float
    wrapped: str = ""
    grouped: float = 1.0
    commented: float = 0.0  # Trailing comment (m^3)
    # A block comment above the field.
    blocked: float = 0.0
    undocumented: float = 0.0


def test_parse_attributes():
    docs = parse_attributes(SampleParams.__doc__)
    assert docs["T"] == "Flash temperature (K)"
    assert docs["wrapped"] == "A description that runs onto a second line."
    assert "grouped" in docs
    assert parse_attributes(None) == {}


def test_field_comments():
    comments = field_comments(SampleParams)
    assert comments["commented"] == "Trailing comment (m^3)"
    assert comments["blocked"] == "A block comment above the field."
    assert "undocumented" not in comments


def test_attribute_docs_combines_docstring_and_comments():
    docs = attribute_docs(SampleParams)
    assert docs["T"] == "Flash temperature (K)"
    assert docs["commented"] == "Trailing comment (m^3)"
    assert "Group heading" not in docs and "undocumented" not in docs
    assert "commented" not in attribute_docs(SampleParams, comments=False)


def test_module_doctests_run():
    import doctest

    import difflow.docstrings

    result = doctest.testmod(difflow.docstrings, verbose=False)
    assert result.attempted > 0 and result.failed == 0
