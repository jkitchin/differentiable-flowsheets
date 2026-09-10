"""Execute the code examples in the docs, so they cannot silently rot.

Documentation that names parameters the code does not have is worse than no
documentation: it reads as authoritative. This module extracts the fenced
``python`` blocks out of a documented section and runs them, so a rename in the
source breaks the test rather than the reader.

Scope is deliberately narrow -- the sections listed in ``DOC_SECTIONS`` are the
ones whose examples have been checked line by line against the code. Widen it a
section at a time, after auditing that section; a blanket sweep over every doc
would fail on prose-style snippets that were never meant to run.
"""

import pathlib
import re

import jax
import pytest

jax.config.update("jax_enable_x64", True)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# (file, section heading, heading that ends it)
DOC_SECTIONS = [
    ("docs/unit-operations-chemical.md", "## Distillation", "## Heat Exchangers"),
]


def _blocks(doc: str, start_heading: str, end_heading: str):
    """Fenced python blocks between two headings, with their line numbers."""
    text = (REPO_ROOT / doc).read_text()
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    section = text[start:end]
    for match in re.finditer(r"```python\n(.*?)```", section, re.S):
        line_no = text[:start].count("\n") + section[:match.start()].count("\n") + 2
        yield match.group(1), line_no


def _is_runnable(block: str) -> bool:
    """Skip the blocks that are illustrations rather than programs.

    A dataclass field listing documents a signature; a one-liner showing a call
    shape has no bindings behind it. Everything else is expected to run.
    """
    stripped = block.strip()
    if stripped.startswith("@dataclass"):
        return False
    return len(stripped.splitlines()) > 1


def _cases():
    for doc, start, end in DOC_SECTIONS:
        for block, line_no in _blocks(doc, start, end):
            if _is_runnable(block):
                yield pytest.param(block, id=f"{pathlib.Path(doc).name}:{line_no}")


@pytest.mark.parametrize("block", list(_cases()))
def test_doc_example_runs(block):
    """Every runnable example in a covered section executes without error."""
    namespace: dict = {}
    exec(compile(block, "<doc example>", "exec"), namespace)


def test_the_extractor_finds_examples():
    """Guard against the scraper silently matching nothing after an edit."""
    assert len(list(_cases())) >= 3
