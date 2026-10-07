"""Every ```python block in the documentation runs.

The audits behind #375 found about 30 examples that raised ``TypeError`` or
``NameError`` on parameters that do not exist. This test extracts each
```python fence from ``docs/*.md`` and the plugin READMEs and executes it, so a
stale example fails here instead of in a reader's session.

* Blocks of one file run in order in one shared namespace, because the pages
  build up: a later block uses what an earlier one imported or defined.
* A block that is not meant to run (a fragment, a placeholder path, something
  needing an optional package or a long solve) is marked by putting
  ``<!-- doc-test: skip -->`` on the line directly above its opening fence;
  add a reason after ``skip:``.
* Output goes to a temporary directory and matplotlib uses ``Agg``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted(
    [*(ROOT / "docs").glob("*.md"), *(ROOT / "src").glob("difflow_*/README.md"), ROOT / "README.md"]
)

_FENCE = re.compile(r"^(?P<ind>[ \t]*)```python[ \t]*$")
_SKIP = re.compile(r"<!--\s*doc-test:\s*skip\b")


def extract_blocks(text: str) -> list[tuple[int, str, bool]]:
    """Return ``(first_line, source, skipped)`` for each ```python fence."""
    lines = text.splitlines()
    blocks, i = [], 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if not m:
            i += 1
            continue
        skipped = i > 0 and bool(_SKIP.search(lines[i - 1]))
        ind, start, body = m["ind"], i + 2, []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith("```"):
            body.append(lines[i][len(ind):] if lines[i].startswith(ind) else lines[i])
            i += 1
        blocks.append((start, "\n".join(body) + "\n", skipped))
        i += 1
    return blocks


def test_extract_blocks_honours_skip_marker():
    text = "a\n```python\nx = 1\n```\n<!-- doc-test: skip: demo -->\n```python\nboom\n```\n"
    assert extract_blocks(text) == [(3, "x = 1\n", False), (7, "boom\n", True)]


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_python_blocks_run(path, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__doc_example__"}
    failures = []
    for line, src, skipped in extract_blocks(path.read_text()):
        if skipped:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exec(compile(src, f"{path.name}:{line}", "exec"), ns)
        except BaseException as exc:  # noqa: BLE001 - report every block
            failures.append(f"{path.name}:{line}: {type(exc).__name__}: {str(exc)[:200]}")
    if failures:
        pytest.fail("documentation examples failed:\n" + "\n".join(failures), pytrace=False)
