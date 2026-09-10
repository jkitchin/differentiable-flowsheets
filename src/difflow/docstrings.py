"""Read parameter documentation back out of docstrings.

Every ``Params`` dataclass in the project documents its fields in an
``Attributes:`` section, which is the convention :file:`CLAUDE.md`
prescribes and the one that is actually maintained::

    @dataclass
    class DrumParams(ParamsMixin):
        '''Parameters for a flash drum.

        Attributes:
            T: Flash temperature (K)
            P: Flash pressure (Pa)
        '''
        T: float
        P: float = 101325.0

:mod:`difflow.catalog` needs the same prose as *data*, so that a form,
a code generator or an external tool reading the schema can label a
field and state its units. Rather than duplicating every description
into ``field(metadata=...)`` across the project --- two copies that
would immediately start to drift --- the catalog reads the docstring
that is already there:

    >>> from difflow.docstrings import attribute_docs
    >>> from difflow.units.cstr import CSTRParams
    >>> attribute_docs(CSTRParams)["V"]
    AttributeDoc(description='Reactor volume (m^3)', units='m^3')

Three things make this more than a ``split(":")``:

* **Only real fields count.** Several docstrings group their attributes
  under sub-headings (``PSA/VSA parameters:``) or mix prose into the
  section. Callers pass the names they care about, or let
  :func:`attribute_docs` read them from the dataclass, so anything that
  is not a documented field is simply not an entry.
* **Comments count too.** Where a field is documented by the comment
  beside or above it rather than in the ``Attributes:`` section ---
  which is how most of the fields added after a class was first written
  are documented --- :func:`field_comments` reads that instead.
* **Units are guessed conservatively.** ``(Pa)`` and ``(mol/m^3)`` are
  units; ``(0-1)``, ``(default 10)`` and ``(1.0 = stoichiometric)`` are
  not, and a wrong unit in a machine-readable schema is worse than a
  missing one. :func:`extract_units` returns ``None`` unless the
  parenthetical looks like a unit expression.

Field metadata still wins where it is present --- see
:func:`difflow.catalog._parameters` --- so a field whose description
must differ from the docstring can say so explicitly.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import textwrap
import tokenize
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO

#: Section headings whose body documents named parameters. ``Args`` and
#: ``Parameters`` appear because a ``Params`` class is sometimes written
#: up as the argument list it effectively is.
ATTRIBUTE_SECTIONS = frozenset({
    "attributes",
    "args",
    "arguments",
    "parameters",
    "keyword args",
    "keyword arguments",
    "other parameters",
})

#: ``name: description``, optionally with a parenthesised type as
#: Google style allows (``T (float): temperature``), or a space before
#: the colon as NumPy style writes it (``T : float``).
_ENTRY = re.compile(
    r"^(?P<name>[A-Za-z_]\w*)"
    r"(?:\s*\((?P<type>[^)]*)\))?"
    r"(?P<gap>\s*):\s*(?P<text>.*)$"
)

#: Unit symbols and words that carry a dimension. A candidate has to
#: contain at least one of these to be read as units, which is what
#: keeps ``(0-1)`` and ``(default 10)`` out. Single letters that are
#: more often a variable than a unit (``n``, ``t``, ``r``, ``a``) are
#: deliberately absent: ``(N)`` in a description is a count far more
#: often than it is newtons.
_UNIT_TOKENS = frozenset({
    # amount, mass, length, volume
    "mol", "mols", "mole", "moles", "kmol", "mmol", "umol", "lbmol",
    "g", "mg", "ug", "kg", "tonne", "tonnes", "lb", "ton", "tons",
    "m", "cm", "mm", "um", "nm", "km", "ft", "micron", "microns",
    "l", "ml", "ul", "liter", "liters", "litre", "litres", "gal",
    "m2", "m3", "cm2", "cm3", "nm3", "sm3", "scf", "scfm", "mscf",
    # time
    "s", "sec", "secs", "ms", "min", "mins", "h", "hr", "hrs", "hour",
    "hours", "day", "days", "wk", "week", "weeks", "yr", "year", "years",
    # temperature
    "k", "degc", "degf", "c",
    # pressure
    "pa", "kpa", "mpa", "gpa", "bar", "bara", "barg", "mbar", "atm",
    "psi", "psia", "psig", "torr", "mmhg",
    # energy and power
    "j", "kj", "mj", "gj", "cal", "kcal", "btu", "wh", "kwh", "mwh",
    "w", "kw", "mw", "gw", "hp",
    # electrical
    "v", "kv", "va", "kva", "mva", "mvar", "kvar", "var", "ohm", "ohms",
    "siemens", "pu", "hz", "rpm", "rev", "ah",
    # concentration and composition
    "ppm", "ppb", "ppmv", "wt", "vol", "fraction", "fractions", "frac",
    "percent", "ph", "molar", "eq", "equiv", "cfu", "cp", "cpoise",
    # currency
    "usd", "eur", "gbp", "dollar", "dollars",
})

#: Words a unit expression may contain *alongside* a dimensional token,
#: to say what the quantity is per: ``g cells / g substrate``,
#: ``kg C / kg fuel``, ``mol CO2 / mol amine``. On their own they are
#: not units, so ``(total/capacity)`` is rejected.
_UNIT_QUALIFIERS = frozenset({
    "dry", "wet", "basis", "bone", "db", "stp", "ntp", "std",
    "feed", "fuel", "air", "water", "steam", "gas", "liquid", "vapor",
    "oil", "organic", "aqueous", "solvent", "sorbent", "adsorbent",
    "resin", "catalyst", "cat", "packing", "bed", "beds", "membrane",
    "co2", "ch4", "h2", "h2o", "n2", "o2", "so2", "nox", "h", "oh",
    "amine", "mea", "ree", "acid", "extractant", "salt",
    "protein", "mab", "product", "biomass", "cell", "cells", "substrate",
    "glucose", "total", "stage", "stages", "cycle", "cycles", "plate",
    "plates", "module", "modules", "unit", "units", "tube", "tubes",
    "permeate", "retentate", "solids", "solid", "slurry",
})

#: Characters a unit expression may contain. Square brackets, quotes,
#: ``=`` and ``<``/``>`` all mark something that is not a unit;
#: parentheses are allowed, for ``J/(kg K)``.
_UNIT_CHARS = re.compile(
    "^[-+*/^.()·⋅×0-9A-Za-zµμ°%$€£\u2070-\u209f\u00b2\u00b3\u00b9 ]+$"
)

#: Purely numeric candidates: ``0-1``, ``0 - 1``, ``1e-3``.
_NUMERIC_ONLY = re.compile(r"^[-+0-9.eE\s]+$")


@dataclass(frozen=True)
class AttributeDoc:
    """Documentation found for one attribute.

    Attributes:
        description: the prose, with continuation lines joined and
            whitespace collapsed.
        units: the unit expression read out of the description, or
            ``None`` when it does not state one.
    """

    description: str
    units: str | None = None


def _dedent_lines(docstring: str) -> list[str]:
    """Docstring lines with the common leading whitespace removed.

    ``inspect.cleandoc`` is used rather than ``textwrap.dedent`` because
    a docstring's first line carries no indentation at all, which
    defeats a naive common-prefix calculation.
    """
    return inspect.cleandoc(docstring).splitlines()


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _section_body(lines: list[str], start: int) -> tuple[list[str], int]:
    """Lines belonging to the section whose heading is at ``start``.

    A Google-style section is indented under its heading, so it runs
    until the first non-blank line at or left of the heading. A
    NumPy-style section is *level* with its underlined heading, so it
    runs until a dedent or until the next underlined heading.
    """
    heading_indent = _indent_of(lines[start])
    i = start + 1
    underlined = i < len(lines) and set(lines[i].strip()) == {"-"}
    if underlined:
        i += 1

    body: list[str] = []
    while i < len(lines):
        line = lines[i]
        indent = _indent_of(line)
        if line.strip():
            if indent < heading_indent:
                break
            if indent == heading_indent and not underlined:
                break
            if (underlined and indent == heading_indent
                    and i + 1 < len(lines)
                    and set(lines[i + 1].strip()) == {"-"}):
                break
        body.append(line)
        i += 1
    return body, i


def _is_heading(lines: list[str], i: int) -> bool:
    """Whether line ``i`` starts an ``Attributes:`` section.

    Google style ends the heading with a colon; NumPy style underlines
    it with dashes on the following line.
    """
    stripped = lines[i].strip()
    if stripped.endswith(":"):
        stripped = stripped[:-1]
    elif not (i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}):
        return False
    return stripped.strip().lower() in ATTRIBUTE_SECTIONS


def parse_attributes(docstring: str | None) -> dict[str, str]:
    """Raw ``{name: description}`` from a docstring's attribute sections.

    Every ``Attributes:``/``Args:``/``Parameters`` section is read, and
    entries from later sections do not overwrite earlier ones.

    Descriptions are returned exactly as documented, including any
    parenthesised units; nothing here decides whether a name is a real
    field --- see :func:`attribute_docs` for that.

    Args:
        docstring: the docstring to read; ``None`` is allowed.

    Returns:
        ``{attribute name: description}``, possibly empty.

    Example:
        >>> parse_attributes("Summary.\\n\\n    Attributes:\\n        T: T (K)")
        {'T': 'T (K)'}
    """
    if not docstring or not docstring.strip():
        return {}

    lines = _dedent_lines(docstring)
    out: dict[str, str] = {}
    i = 0
    while i < len(lines):
        if not _is_heading(lines, i):
            i += 1
            continue
        body, i = _section_body(lines, i)
        for name, text in _parse_entries(body).items():
            out.setdefault(name, text)
    return out


def _parse_entries(body: list[str]) -> dict[str, str]:
    """``{name: description}`` from the body of one section.

    An entry starts at a line at the section's base indentation that
    reads ``name: ...``; more deeply indented lines continue it. A line
    at the base indentation that is *not* an entry --- a bullet, a
    sentence of prose --- ends the current entry rather than being
    glued onto it.
    """
    indents = [_indent_of(line) for line in body if line.strip()]
    if not indents:
        return {}
    base = min(indents)

    entries: dict[str, list[str]] = {}
    numpy_form: set[str] = set()
    current: list[str] | None = None
    for line in body:
        if not line.strip():
            # A blank line separates groups but does not end a wrapped
            # description; keep the current entry open.
            continue
        if _indent_of(line) > base:
            if current is not None:
                current.append(line.strip())
            continue
        match = _ENTRY.match(line.strip())
        if match is None:
            current = None
            continue
        name = match.group("name")
        current = entries.setdefault(name, [])
        text = match.group("text").strip()
        if text:
            # NumPy style writes the *type* there ("T : float") and the
            # description underneath; Google style never pads the colon
            if match.group("gap") and not match.group("type"):
                numpy_form.add(name)
            current.append(text)

    for name in numpy_form:
        if len(entries.get(name, [])) > 1:
            entries[name] = entries[name][1:]

    return {
        name: re.sub(r"\s+", " ", " ".join(parts)).strip()
        for name, parts in entries.items()
        if any(part.strip() for part in parts)
    }


def extract_units(description: str | None) -> str | None:
    """The unit expression stated in a description, if any.

    Units are written in the project's docstrings as a parenthetical or
    as a trailing clause: ``Flash pressure (Pa)``, ``Amine
    concentration (mol/m^3)``, ``specific heat capacity, J/(kg K)``.
    The same positions are also used for ranges, defaults and
    commentary, so a candidate is accepted only when it reads as a unit
    expression --- see :func:`_looks_like_units`. A wrong unit in a
    machine-readable schema is worse than a missing one, so anything
    doubtful is left out.

    Args:
        description: the documented description.

    Returns:
        The units, or ``None`` when the description states none.

    Example:
        >>> extract_units("Flash pressure (Pa)")
        'Pa'
        >>> extract_units("specific heat capacity, J/(kg K).")
        'J/(kg K)'
        >>> extract_units("Murphree stage efficiency (0-1)") is None
        True
        >>> extract_units("Coordination number (default 10)") is None
        True
    """
    if not description:
        return None
    for candidate in _unit_candidates(description):
        if _looks_like_units(candidate):
            return candidate
    return None


def _unit_candidates(description: str) -> list[str]:
    """The substrings of a description that could be its units.

    Parenthesised groups come first, in the order written, each
    extended leftwards over any unit operator it hangs off so that
    ``J/(kg K)`` is offered whole rather than as ``kg K``. The clause
    after the final comma comes last, for the descriptions that write
    their units there instead.
    """
    candidates: list[str] = []
    depth, start = 0, None
    for i, ch in enumerate(description):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                inner = description[start + 1:i]
                prefix = re.search(r"[0-9A-Za-z]+[*/·]$", description[:start])
                if prefix:
                    candidates.append(f"{prefix.group(0)}({inner})")
                else:
                    candidates.append(inner)
                start = None
    tail = description.rstrip(" .").rsplit(",", 1)
    if len(tail) == 2:
        candidates.append(tail[1].strip())
    return [c.strip() for c in candidates if c.strip()]


def _looks_like_units(text: str) -> bool:
    """Whether a candidate substring reads as a unit expression.

    It must name at least one dimensional unit, every word must be a
    unit or a qualifier saying what the quantity is per, and it must
    carry none of the punctuation that marks prose --- a comma, a
    comparison, a quote, an assignment.
    """
    if not text or len(text) > 28:
        return False
    if "," in text or "_" in text:
        return False
    if not _UNIT_CHARS.match(text):
        return False
    if _NUMERIC_ONLY.match(text):
        return False
    words = [w for w in re.split(r"[^A-Za-z0-9%$€£°µμ]+", text) if w]
    if not words or len(words) > 5:
        return False
    lowered = [w.lower() for w in words]
    if not all(w in _UNIT_TOKENS or w in _UNIT_QUALIFIERS or w.isdigit()
               for w in lowered):
        return False
    return any(w in _UNIT_TOKENS for w in lowered)


def attribute_docs(
    cls: type,
    names: list[str] | None = None,
    *,
    comments: bool = True,
) -> dict[str, AttributeDoc]:
    """Documentation for a class's attributes, read from its source.

    The ``Attributes:`` sections come first and the comments around the
    fields fill in the rest, which between them is where the project
    writes its parameter documentation. The whole MRO is consulted,
    nearest class first, so a ``Params`` subclass inherits the
    documentation of the fields it inherits.

    Args:
        cls: the class to document. For a dataclass, its fields are the
            default set of names to look for.
        names: the attribute names to keep. Defaults to the dataclass
            field names, or to every name the docstrings document when
            ``cls`` is not a dataclass.
        comments: whether to fall back to :func:`field_comments` for
            fields the docstrings do not mention.

    Returns:
        ``{name: AttributeDoc}`` for the names that are documented;
        names with no documentation are absent rather than present and
        empty.
    """
    if names is None and dataclasses.is_dataclass(cls):
        names = [f.name for f in dataclasses.fields(cls)]

    found = _documented(cls, comments)
    if names is None:
        return dict(found)
    return {name: found[name] for name in names if name in found}


@lru_cache(maxsize=None)
def _documented(cls: type, comments: bool) -> dict[str, AttributeDoc]:
    """Every attribute ``cls`` documents anywhere in its MRO.

    Cached: reading a class's source and tokenizing it costs far more
    than the rest of building a schema, and :func:`difflow.catalog.catalog`
    describes every registered operation at once. The values are frozen
    and callers are handed a copy of the mapping.
    """
    found: dict[str, str] = {}
    for klass in inspect.getmro(cls):
        if klass is object:
            continue
        for name, text in parse_attributes(getattr(klass, "__doc__", None)).items():
            found.setdefault(name, text)
    if comments:
        for klass in inspect.getmro(cls):
            if klass is object:
                continue
            for name, text in field_comments(klass).items():
                found.setdefault(name, text)
    return {
        name: AttributeDoc(description=text, units=extract_units(text))
        for name, text in found.items()
    }


# ---------------------------------------------------------------------
# Comments beside the fields
# ---------------------------------------------------------------------

#: Comment prefixes that are instructions to a tool, not prose.
_DIRECTIVES = ("noqa", "type:", "pragma", "pylint", "flake8", "mypy",
               "fmt:", "ruff", "isort", "todo", "fixme", "xxx", "hack")

#: A run of rule characters marks a banner comment (``# ==== Timing ====``),
#: which labels a group of fields rather than describing one.
_BANNER = re.compile(r"(===|---|\*\*\*|###)")


def _comment_text(raw: str) -> str | None:
    """The prose of one comment line, or ``None`` if it carries none."""
    text = raw.lstrip("#").strip()
    if not text or _BANNER.search(text):
        return None
    if text.lower().startswith(_DIRECTIVES):
        return None
    if not re.search(r"[A-Za-z]", text):
        return None
    return text


def field_comments(cls: type) -> dict[str, str]:
    """``{field name: description}`` from the comments around each field.

    Two placements are read, and both are in use across the project::

        q: float = 1.0  # Feed thermal condition (1.0 = saturated liquid)

        # Aqueous nitrate concentration (M), required for solvating
        # extractants such as TBP whose D is nitrate-driven.
        nitrate_conc: float | None = None

    A field's comment is the block of whole-line comments immediately
    above it plus the comment trailing its own last line. A bare ``#``
    inside the block separates paragraphs; a banner comment labelling a
    group of fields, or a tool directive such as ``# noqa``, ends it.

    Args:
        cls: the class whose source to read.

    Returns:
        ``{name: description}`` for the fields that carry a comment.
        Empty when the source is unavailable, as it is for a class
        defined in a REPL.
    """
    return dict(_field_comments(cls))


@lru_cache(maxsize=None)
def _field_comments(cls: type) -> dict[str, str]:
    """:func:`field_comments`, cached on the class; see :func:`_documented`."""
    try:
        source = textwrap.dedent(inspect.getsource(cls))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}

    body = next((n.body for n in tree.body if isinstance(n, ast.ClassDef)), None)
    if body is None:
        return {}

    # tokenize rather than search for "#", which also appears in strings
    try:
        tokens = list(tokenize.generate_tokens(StringIO(source).readline))
    except (tokenize.TokenError, IndentationError):
        return {}

    lines = source.splitlines()
    own_line: dict[int, str] = {}
    trailing: dict[int, str] = {}
    for tok in tokens:
        if tok.type is not tokenize.COMMENT:
            continue
        row, col = tok.start
        alone = not lines[row - 1][:col].strip()
        (own_line if alone else trailing)[row] = tok.string

    out: dict[str, str] = {}
    for node in body:
        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        parts: list[str] = []
        # the block of whole-line comments immediately above the field
        above: list[str] = []
        line = node.lineno - 1
        while line in own_line:
            raw = own_line[line]
            if not raw.lstrip("#").strip():
                # a bare "#" separates paragraphs of one comment block
                line -= 1
                continue
            text = _comment_text(raw)
            if text is None:
                # a banner or a tool directive ends the block, but what
                # was found below it still describes this field
                break
            above.insert(0, text)
            line -= 1
        parts.extend(above)
        # and the comment trailing the field itself
        end = node.end_lineno or node.lineno
        if end in trailing:
            text = _comment_text(trailing[end])
            if text is not None:
                parts.append(text)
        if parts:
            out[node.target.id] = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return out
