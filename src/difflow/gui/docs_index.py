"""A searchable index of ``docs/``, built once and shipped with the package.

The assistant panel answers questions about a flowsheet, and half the
answers are already written down --- in the 23 markdown files under
``docs/``. Getting the right paragraph in front of the model is
retrieval, and retrieval here is lexical and build-time rather than a
vector database: the corpus is 720 KB and fixed at release, an
embedding model is a dependency difflow is not going to take, and a
term-frequency score over a few hundred sections is a dictionary
lookup.

So this module splits the docs at their headings and writes
``static/docs-index.json``. It runs from :ref:`make gui-build` alongside
the JS bundle, the result is **committed**, and CI rebuilds it and fails
on a difference --- the same discipline the bundle is under, and for the
same reason: an index that silently stops matching the prose it indexes
is worse than no index.

It is deliberately stdlib-only and imports nothing from difflow, so the
build step is ``python3 src/difflow/gui/docs_index.py`` with nothing
installed.

Why an index file rather than reading ``docs/`` at run time: ``docs/``
is not in the wheel. ``static/`` is. A pip-installed difflow has to be
able to answer the question too.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

#: Where the built index goes. Under ``static/`` because that directory
#: is package data and therefore reaches an installed difflow.
INDEX = Path(__file__).parent / "static" / "docs-index.json"

#: The prose. Two levels up from ``src/difflow/gui`` is the repository.
DOCS = Path(__file__).resolve().parents[3] / "docs"

#: A section longer than this is cut at the last paragraph break before
#: it. The retriever pastes whole sections, and the budget it pastes
#: into is a few thousand tokens: a 12 KB section is not a citation, it
#: is the whole answer window spent on one heading.
MAX_SECTION = 2400

#: Words that appear everywhere and so distinguish nothing. Short and
#: hand-written on purpose --- a longer list would start dropping terms
#: that are ordinary English *and* difflow vocabulary ("state", "value").
STOPWORDS = frozenset("""
the and for that with this from are was not but you your has have had
its it's they them their there here when what which who whom how why
all any can cut did does done each else even ever few get got let may
might more most much must now off one only other our out over own same
should some such than then these those thus too under until upon very
were will would about above after again against because been before
being below between both during into through
""".split())

_WORD = re.compile(r"[a-z_][a-z0-9_]*")
_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")


def terms(text: str) -> list[str]:
    """The indexable words of a piece of text.

    Dotted names split on the dot, which is what makes
    ``flowsheet.solve`` findable by either half --- and the halves are
    what a question actually contains.
    """
    return [w for w in _WORD.findall(text.lower())
            if len(w) >= 3 and w not in STOPWORDS]


def anchor(heading: str) -> str:
    """The GitHub-style anchor a heading gets, so the panel can link to it."""
    slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
    return re.sub(r"\s+", "-", slug.strip())


def _trim(body: str) -> str:
    """A section cut to :data:`MAX_SECTION` at a paragraph boundary."""
    if len(body) <= MAX_SECTION:
        return body.strip()
    cut = body.rfind("\n\n", 0, MAX_SECTION)
    return (body[:cut] if cut > MAX_SECTION // 2 else body[:MAX_SECTION]).strip()


def split(text: str, source: str) -> list[dict]:
    """One markdown file as a list of sections.

    Headings are found outside fenced code only. Every one of these
    files is a tutorial full of Python, and ``# Create the flowsheet``
    inside a fence is a comment, not a section --- counting naively
    turns ``unit-operations-chemical.md`` into 138 "sections" most of
    which are one line of a code block.

    The breadcrumb (``path``) carries the enclosing headings, so a
    section called "Gotchas" says which chapter's gotchas it is.
    """
    sections: list[dict] = []
    stack: list[str] = []
    heading, level, body = None, 0, []
    fenced = False

    def flush():
        if heading is None and not "".join(body).strip():
            return
        text_ = _trim("".join(body))
        if not text_:
            return
        sections.append({
            "source": source,
            "heading": heading or source,
            "path": " > ".join(stack),
            "anchor": anchor(heading or ""),
            "text": text_,
        })

    for line in text.splitlines(keepends=True):
        if _FENCE.match(line):
            fenced = not fenced
        match = None if fenced else _HEADING.match(line.rstrip("\n"))
        if match is None:
            body.append(line)
            continue
        flush()
        level, heading = len(match.group(1)), match.group(2)
        stack = stack[:level - 1] + [heading]
        body = []
    flush()
    return sections


def build(docs: Path = DOCS) -> dict:
    """Every section of every markdown file under ``docs``.

    Text only. The term counts and the document-frequency table that
    score a query are *derived* from this text, so writing them into the
    committed file would be storing the same corpus twice --- 500 KB of
    it --- and inviting the two copies to disagree. :func:`load`
    computes them once, on the way in.
    """
    sections = []
    for path in sorted(docs.glob("*.md")):
        sections.extend(split(path.read_text(), path.name))
    return {"version": 1, "n_sections": len(sections), "sections": sections}


def index_terms(index: dict) -> dict:
    """Fill in the term statistics a search needs, in place.

    Each section gets its own counts and the corpus gets one
    document-frequency table --- everything TF-IDF needs, and the
    smallest thing that is. Idempotent, so a caller that is not sure
    whether an index has been prepared can just call it.
    """
    if "df" in index:
        return index
    df: dict[str, int] = {}
    for section in index.get("sections") or []:
        counts: dict[str, int] = {}
        for word in terms(f"{section['path']} {section['heading']} "
                          f"{section['text']}"):
            counts[word] = counts.get(word, 0) + 1
        section["terms"] = counts
        for word in counts:
            df[word] = df.get(word, 0) + 1
    index["df"] = df
    return index


#: One parsed index per file, since preparing it re-tokenises the whole
#: corpus and the corpus does not change while the server is up.
_CACHE: dict[str, dict] = {}


def load(path: Path = INDEX) -> dict | None:
    """The built index, prepared for search, or ``None`` if never built.

    ``None`` rather than an exception: an assistant that answers without
    a documentation quote is degraded, not broken, and a source
    checkout that has not run ``make gui-build`` should still open the
    panel.
    """
    key = str(path)
    if key not in _CACHE:
        try:
            _CACHE[key] = index_terms(json.loads(path.read_text()))
        except (OSError, ValueError):
            return None
    return _CACHE[key]


def search(index: dict, question: str, limit: int = 3) -> list[dict]:
    """The sections that best match a question, best first.

    TF-IDF with a log-damped term frequency and length normalisation ---
    without the last of those, the longest section in the corpus wins
    every query by having said everything at least once.
    """
    if not index or not index.get("sections"):
        return []
    index_terms(index)
    n = max(index.get("n_sections") or len(index["sections"]), 1)
    df = index["df"]
    wanted = set(terms(question))
    if not wanted:
        return []

    scored = []
    for section in index["sections"]:
        counts = section.get("terms") or {}
        length = math.sqrt(sum(counts.values())) or 1.0
        score = sum(
            (1 + math.log(counts[word])) * math.log(1 + n / (1 + df.get(word, 0)))
            for word in wanted if word in counts
        ) / length
        if score > 0:
            scored.append((score, section))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["source"], pair[1]["heading"]))
    return [{k: v for k, v in section.items() if k != "terms"} | {"score": round(score, 4)}
            for score, section in scored[:limit]]


def write(out: Path = INDEX, docs: Path = DOCS) -> Path:
    """Build the index and write it where the package will find it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    index = build(docs)
    # `sort_keys` and a trailing newline: this file is committed and CI
    # diffs it, so two builds of the same prose must be the same bytes.
    out.write_text(json.dumps(index, indent=1, sort_keys=True) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    docs = Path(argv[0]) if argv else DOCS
    if not docs.is_dir():
        print(f"no docs directory at {docs}", file=sys.stderr)
        return 1
    path = write(docs=docs)
    index = index_terms(json.loads(path.read_text()))
    print(f"{path}: {index['n_sections']} sections, "
          f"{len(index['df'])} terms, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
