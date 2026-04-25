"""Metadata contract for difflow unit operations.

Every unit class that can appear in a :class:`~difflow.flowsheet.Flowsheet`
may expose the following class-level attributes to make itself
self-documenting:

- ``symbol``: short identifier used in report headings (defaults to class name)
- ``equations``: list of raw LaTeX strings (no ``$`` delimiters)
- ``assumptions``: list of plain-text assumptions
- ``references``: list of inline citation strings
- ``parameter_symbols``: dict mapping ``Params`` field name -> LaTeX symbol
- ``parameter_units``: dict mapping ``Params`` field name -> unit string
- ``numerical_method``: short description of the numerical method used

All fields are optional. :func:`get_metadata` returns a :class:`UnitMetadata`
for any class, falling back to docstring parsing for equations when the
structured attributes are absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnitMetadata:
    """Structured metadata describing a unit operation.

    Attributes:
        symbol: Short identifier used in report headings.
        equations: List of raw LaTeX strings.
        assumptions: List of assumption strings.
        references: List of inline citation strings.
        parameter_symbols: Mapping from Params field name to LaTeX symbol.
        parameter_units: Mapping from Params field name to unit string.
        numerical_method: Short description of the numerical method.
        description: One-line description (falls back to class docstring first line).
    """

    symbol: str
    description: str
    equations: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    parameter_symbols: dict[str, str] = field(default_factory=dict)
    parameter_units: dict[str, str] = field(default_factory=dict)
    numerical_method: str | None = None


_EQ_HEADERS = ("Key equations:", "Governing equations:", "Equations:")
_ASSUMPTION_HEADERS = ("Assumptions:",)
_REFERENCE_HEADERS = ("References:",)


def _extract_section(docstring: str, headers: tuple[str, ...]) -> list[str]:
    """Extract a bulleted/indented section from a docstring.

    Supports module and class docstrings that include sections like::

        Key equations:
            F_in - F_out + V*r = 0
            H_in - H_out + V*r*dH = 0

    Stops when a blank line is followed by unindented text or another header.
    """
    if not docstring:
        return []

    lines = docstring.splitlines()
    collected: list[str] = []
    in_section = False
    base_indent: int | None = None

    for raw in lines:
        stripped = raw.strip()
        if not in_section:
            if any(stripped.startswith(h) for h in headers):
                in_section = True
                base_indent = None
            continue

        if not stripped:
            if collected:
                # blank line ends the section when a non-indented line follows
                base_indent = base_indent  # placeholder
            continue

        indent = len(raw) - len(raw.lstrip())
        if base_indent is None:
            base_indent = indent

        # New header or unindented line ends the section
        if indent < base_indent:
            break
        if any(stripped.startswith(h) for h in _EQ_HEADERS + _ASSUMPTION_HEADERS + _REFERENCE_HEADERS):
            if not any(stripped.startswith(h) for h in headers):
                break

        # Strip leading bullet characters
        text = re.sub(r"^[-*•]\s*", "", stripped)
        collected.append(text)

    return collected


def _first_docstring_line(docstring: str | None) -> str:
    if not docstring:
        return ""
    for line in docstring.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _module_docstring(cls: type) -> str:
    """Fetch the docstring of the module where ``cls`` is defined."""
    import sys

    module = sys.modules.get(cls.__module__)
    return (module.__doc__ or "") if module is not None else ""


def get_metadata(cls: type) -> UnitMetadata:
    """Return :class:`UnitMetadata` for a unit operation class.

    Reads structured class attributes when present, and falls back to
    parsing ``Key equations:`` / ``Assumptions:`` / ``References:`` sections
    from the class and module docstrings when they are missing.
    """
    symbol = getattr(cls, "symbol", cls.__name__)
    description = getattr(cls, "description", None) or _first_docstring_line(cls.__doc__)

    equations = list(getattr(cls, "equations", []) or [])
    assumptions = list(getattr(cls, "assumptions", []) or [])
    references = list(getattr(cls, "references", []) or [])
    parameter_symbols = dict(getattr(cls, "parameter_symbols", {}) or {})
    parameter_units = dict(getattr(cls, "parameter_units", {}) or {})
    numerical_method = getattr(cls, "numerical_method", None)

    # Fallbacks: pull from class docstring, then module docstring.
    if not equations:
        for source in (cls.__doc__, _module_docstring(cls)):
            found = _extract_section(source or "", _EQ_HEADERS)
            if found:
                equations = found
                break
    if not assumptions:
        for source in (cls.__doc__, _module_docstring(cls)):
            found = _extract_section(source or "", _ASSUMPTION_HEADERS)
            if found:
                assumptions = found
                break
    if not references:
        for source in (cls.__doc__, _module_docstring(cls)):
            found = _extract_section(source or "", _REFERENCE_HEADERS)
            if found:
                references = found
                break

    return UnitMetadata(
        symbol=symbol,
        description=description,
        equations=equations,
        assumptions=assumptions,
        references=references,
        parameter_symbols=parameter_symbols,
        parameter_units=parameter_units,
        numerical_method=numerical_method,
    )


def has_structured_metadata(cls: type) -> bool:
    """True if ``cls`` declares any of the structured metadata attributes."""
    for attr in ("equations", "assumptions", "references", "parameter_symbols", "parameter_units"):
        if getattr(cls, attr, None):
            return True
    return False
