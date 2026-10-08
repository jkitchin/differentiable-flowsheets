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


#: Pages whose blocks cost over a minute between them (recycle solves, JAX
#: traces, MILPs); ``make test`` deselects them with ``-m 'not slow'`` and
#: the full suite runs them.
SLOW = {
    "docs/convergence.md", "docs/data-reconciliation.md",
    "docs/moving-horizon-estimation.md", "docs/planning.md",
    "docs/streams-and-flowsheets.md", "docs/unit-operations-chemical.md",
    "docs/unit-operations-ree.md", "docs/dynamic-modeling.md",
    "docs/getting-started.md", "docs/thermodynamics.md",
    "docs/unit-operations-refinery.md", "README.md",
}


def _case(path: Path):
    rel = str(path.relative_to(ROOT))
    marks = [pytest.mark.slow] if rel in SLOW else []
    return pytest.param(path, id=rel, marks=marks)


@pytest.mark.parametrize("path", [_case(f) for f in FILES])
def test_doc_python_blocks_run(path, tmp_path, monkeypatch):
    import matplotlib

    matplotlib.use("Agg")
    monkeypatch.chdir(tmp_path)
    # Examples run in this process, and some of them register operations
    # (the plugin-architecture page registers "my_reactor"). Left in the
    # global registry, that operation leaked into every later test in the
    # worker: the catalog, doc-link and report-metadata tests then found an
    # undocumented operation with no equations. Put the registry back.
    from difflow.plugins import registry

    saved = (dict(registry._operations),
             {k: list(v) for k, v in registry._categories.items()})
    monkeypatch.setattr(registry, "_operations", saved[0].copy())
    monkeypatch.setattr(registry, "_categories",
                        {k: list(v) for k, v in saved[1].items()})
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
