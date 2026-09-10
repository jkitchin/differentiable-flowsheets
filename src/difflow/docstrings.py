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

:mod:`difflow.catalog` needs the same prose as *data*, so that a form, a
code generator or an external tool reading the schema can label a field
rather than only name it. Rather than duplicating every description into
``field(metadata={"description": ...})`` across the project --- two
copies that would immediately start to drift --- the catalog reads the
docstring that is already there:

    >>> from difflow.docstrings import attribute_docs
    >>> from difflow.units.cstr import CSTRParams
    >>> attribute_docs(CSTRParams)["V"]
    'Reactor volume (m^3)'

Two things make this more than a ``split(":")``:

* **Only real fields count.** Several docstrings group their attributes
  under sub-headings (``PSA/VSA parameters:``) or mix prose into the
  section. Callers pass the names they care about, or let
  :func:`attribute_docs` read them from the dataclass, so anything that
  is not a documented field is simply not an entry.
* **Comments count too.** 35 fields are documented by the comment beside
  or above them rather than in the ``Attributes:`` section, which is how
  most of those added after a class was first written are documented;
  :func:`field_comments` reads those.

Field metadata still wins where it is present --- see
:func:`difflow.catalog._parameters` --- so a field whose description
must differ from the docstring can say so explicitly.

Units are *not* read from here. They have an explicit source: the unit
class's ``parameter_units``, part of the metadata contract in
:mod:`difflow.report.metadata`, which names every numeric field and is
guarded by a test that fails when a key stops matching a field. A
parenthetical in prose is a weaker signal than that table --- ``(0-1)``
and ``(default 10)`` sit in the same position as ``(Pa)`` --- and a
wrong unit in a machine-readable schema is worse than a missing one.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import textwrap
import tokenize
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


def attribute_docs(
    cls: type,
    names: list[str] | None = None,
    *,
    comments: bool = True,
) -> dict[str, str]:
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
        ``{name: description}`` for the names that are documented; names
        with no documentation are absent rather than present and empty.
    """
    if names is None and dataclasses.is_dataclass(cls):
        names = [f.name for f in dataclasses.fields(cls)]

    found = _documented(cls, comments)
    if names is None:
        return dict(found)
    return {name: found[name] for name in names if name in found}


@lru_cache(maxsize=None)
def _documented(cls: type, comments: bool) -> dict[str, str]:
    """Every attribute ``cls`` documents anywhere in its MRO.

    Cached: reading a class's source and tokenizing it costs far more
    than the rest of building a schema, and
    :func:`difflow.catalog.catalog` describes every registered operation
    at once. Callers are handed a copy of the mapping.
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
    return found


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
