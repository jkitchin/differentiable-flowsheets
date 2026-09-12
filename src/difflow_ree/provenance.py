"""Where every number in ``difflow_ree/data`` came from.

The data files mix numbers of very different pedigree. Some are copied from a
named table in a named book. Some are computed from other numbers in the same
file. Some were invented so that a demo would converge. Read as bare YAML they
look identical, and that is exactly how an invented number ends up behind a
published stage count.

This module makes the difference queryable. Every data file carries a
``provenance:`` block of glob rules mapping dotted field paths to keys in
``data/sources.yaml``; every key carries a full citation and a ``cls``
answering the only question that matters at the call site -- *may I publish a
number that depends on this?*

    >>> from difflow_ree.provenance import explain
    >>> explain("elements", "elements.Dy.ionic_radius_pm").source
    'S76'
    >>> explain("extractants",
    ...         "extractants.D2EHPA.ph_coefficients.Nd.a").cls
    'HAND_TUNED'

Three entry points, in increasing order of breadth:

``explain(dataset, path)``
    One field: its value, its source key, the full citation, the locus within
    that source, and any field-specific note.

``audit(dataset=None, cls=None)``
    Every leaf, or every leaf of one class. ``audit(cls="HAND_TUNED")`` is the
    list of numbers nobody can defend.

``coverage(dataset=None)``
    Leaf counts by class. The one-line answer to "how much of this database is
    actually measured?"

And a report for humans::

    python -m difflow_ree.provenance
    python -m difflow_ree.provenance --cls HAND_TUNED
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import yaml

DATA_DIR = Path(__file__).parent / "data"

#: Data files carrying a ``provenance:`` block, by dataset name.
DATASETS: dict[str, str] = {
    "elements": "elements.yaml",
    "extractants": "extractants.yaml",
    "separation_factors": "separation_factors.yaml",
}

#: Top-level key holding the rules. Excluded from the leaves it describes.
PROVENANCE_KEY = "provenance"

#: Source key synthesized for a leaf that no rule matches. Never written in a
#: file -- if this shows up, the provenance block is incomplete.
UNTAGGED = "UNTAGGED"

#: Classes whose numbers must not stand behind a published result. Ordered
#: worst-first, which is also the order the report prints them in.
UNPUBLISHABLE: tuple[str, ...] = ("HAND_TUNED", "ESTIMATED", "CONSTRUCTED")

#: Every class, worst-first. Used to order reports.
CLASS_ORDER: tuple[str, ...] = (
    "HAND_TUNED",
    "ESTIMATED",
    "CONSTRUCTED",
    "CONVENTION",
    "DERIVED",
    "DISCLOSED",
    "REFERENCE",
    "MEASURED",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict:
    with open(DATA_DIR / name) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_sources() -> dict[str, dict]:
    """Return the bibliographic registry from ``data/sources.yaml``."""
    return _load_yaml("sources.yaml")["sources"]


@lru_cache(maxsize=None)
def load_dataset(dataset: str) -> dict:
    """Return the parsed YAML of one dataset."""
    if dataset not in DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset!r}. Known: {sorted(DATASETS)}"
        )
    return _load_yaml(DATASETS[dataset])


@lru_cache(maxsize=None)
def load_rules(dataset: str) -> tuple[tuple[str, dict], ...]:
    """Return ``(pattern, rule)`` pairs for a dataset, most specific first.

    Sorting here rather than at match time means :func:`explain` can take the
    first hit, and means an ambiguous pair of rules is resolved the same way
    every call.
    """
    block = load_dataset(dataset).get(PROVENANCE_KEY)
    if block is None:
        raise KeyError(
            f"{DATASETS[dataset]} has no `{PROVENANCE_KEY}:` block. "
            "Every data file needs one; see the block in elements.yaml."
        )
    rules = [(r["pattern"], r) for r in block["rules"]]
    rules.sort(key=lambda pr: _specificity(pr[0]), reverse=True)
    return tuple(rules)


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def _specificity(pattern: str) -> tuple[int, int]:
    """Rank a pattern: more literal segments wins, then longer wins.

    ``extractants.TBP.nitrate_coefficients.*.d`` (4 literals) therefore beats
    ``extractants.*.*.*.*`` (1 literal) for the same path, which is the whole
    point of having a catch-all rule at all.
    """
    segs = pattern.split(".")
    return (sum(1 for s in segs if s != "*"), len(segs))


def _matches(pattern: str, path: str) -> bool:
    """``*`` matches exactly one segment. Segment counts must be equal."""
    p = pattern.split(".")
    q = path.split(".")
    if len(p) != len(q):
        return False
    return all(a == "*" or a == b for a, b in zip(p, q))


def _walk(node: Any, prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every leaf.

    A list is a leaf: ``oxidation_states: [3]`` and ``valid_ph_range: [4.0,
    5.0]`` are single facts with a single provenance, not several.
    """
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, prefix + (str(k),))
    else:
        yield ".".join(prefix), node


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where one field came from."""

    dataset: str
    path: str
    value: Any
    source: str
    cls: str
    citation: str
    #: Where in the source: a table, an equation, a page. May be empty.
    locus: str = ""
    #: Field-specific caveat from the rule that matched.
    note: str = ""
    #: Source-wide caveat from the registry entry.
    source_note: str = ""
    #: The glob that matched. Empty when nothing did.
    pattern: str = ""

    @property
    def publishable(self) -> bool:
        """False if a published number must not rest on this field."""
        return self.cls not in UNPUBLISHABLE

    def __str__(self) -> str:
        lines = [
            f"{self.dataset}:{self.path} = {self.value!r}",
            f"  source   {self.source}  [{self.cls}]",
            f"  citation {_wrap(self.citation, 11)}",
        ]
        if self.locus:
            lines.append(f"  locus    {self.locus}")
        if self.note:
            lines.append(f"  note     {_wrap(self.note, 11)}")
        if self.source_note:
            lines.append(f"  about {self.source}: {_wrap(self.source_note, 11)}")
        if not self.publishable:
            lines.append(
                f"  *** {self.cls}: do not put a published number behind this."
            )
        return "\n".join(lines)


def _wrap(text: str, indent: int, width: int = 78) -> str:
    """Fill `text` so continuation lines align under an `indent`-wide label."""
    return textwrap.fill(
        " ".join(str(text).split()),
        width=width,
        subsequent_indent=" " * indent,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def explain(dataset: str, path: str) -> Provenance:
    """Return the provenance of one field.

    Args:
        dataset: One of :data:`DATASETS` -- ``"elements"``, ``"extractants"``,
            ``"separation_factors"``.
        path: Dotted path into that file, e.g.
            ``"elements.Dy.ionic_radius_pm"``. As a convenience the leading
            top-level key may be omitted when it is unambiguous, so
            ``"Dy.ionic_radius_pm"`` also resolves.

    Raises:
        KeyError: if the path does not exist in the file. An unmatched path is
            an error; an untagged one is not -- that comes back as
            ``source="UNTAGGED"`` so that :func:`audit` can report it.
    """
    data = load_dataset(dataset)
    leaves = dict(_walk({k: v for k, v in data.items() if k != PROVENANCE_KEY}))

    if path not in leaves:
        # Allow the top-level key to be elided: "Dy.ionic_radius_pm".
        candidates = [p for p in leaves if p.endswith("." + path) or p == path]
        if len(candidates) == 1:
            path = candidates[0]
        elif not candidates:
            raise KeyError(f"No field {path!r} in {DATASETS[dataset]}")
        else:
            raise KeyError(
                f"{path!r} is ambiguous in {DATASETS[dataset]}: "
                f"{sorted(candidates)[:5]}"
            )

    return _resolve(dataset, path, leaves[path])


def _resolve(dataset: str, path: str, value: Any) -> Provenance:
    sources = load_sources()
    for pattern, rule in load_rules(dataset):
        if _matches(pattern, path):
            key = rule["source"]
            entry = sources.get(key)
            if entry is None:
                raise KeyError(
                    f"Rule {pattern!r} in {DATASETS[dataset]} cites source "
                    f"{key!r}, which is not in sources.yaml."
                )
            return Provenance(
                dataset=dataset,
                path=path,
                value=value,
                source=key,
                cls=entry["cls"],
                citation=" ".join(str(entry["citation"]).split()),
                locus=" ".join(str(rule.get("locus", "")).split()),
                note=" ".join(str(rule.get("note", "")).split()),
                source_note=" ".join(str(entry.get("note", "")).split()),
                pattern=pattern,
            )

    entry = sources[UNTAGGED]
    return Provenance(
        dataset=dataset,
        path=path,
        value=value,
        source=UNTAGGED,
        cls=entry["cls"],
        citation=" ".join(str(entry["citation"]).split()),
        source_note=" ".join(str(entry.get("note", "")).split()),
    )


def audit(
    dataset: str | None = None,
    cls: str | None = None,
    source: str | None = None,
) -> list[Provenance]:
    """Return the provenance of every leaf, optionally filtered.

    Args:
        dataset: Restrict to one dataset. Default: all of them.
        cls: Restrict to one class, e.g. ``"HAND_TUNED"``.
        source: Restrict to one source key, e.g. ``"Z1"``.

    The list is in file order, which keeps it diffable across runs.
    """
    names = [dataset] if dataset else list(DATASETS)
    out: list[Provenance] = []
    for name in names:
        data = load_dataset(name)
        body = {k: v for k, v in data.items() if k != PROVENANCE_KEY}
        for path, value in _walk(body):
            p = _resolve(name, path, value)
            if cls is not None and p.cls != cls:
                continue
            if source is not None and p.source != source:
                continue
            out.append(p)
    return out


def coverage(dataset: str | None = None) -> dict[str, int]:
    """Return leaf counts by class, worst-first.

    The headline number for "how much of this database is actually measured".
    """
    counts: dict[str, int] = {}
    for p in audit(dataset):
        counts[p.cls] = counts.get(p.cls, 0) + 1
    ordered = {c: counts[c] for c in CLASS_ORDER if c in counts}
    for c in sorted(counts):  # any class not in CLASS_ORDER
        ordered.setdefault(c, counts[c])
    return ordered


def unsourced(dataset: str | None = None) -> list[Provenance]:
    """Every leaf a published number must not rest on.

    Shorthand for the union of the :data:`UNPUBLISHABLE` classes plus anything
    no rule matched.
    """
    return [
        p
        for p in audit(dataset)
        if not p.publishable or p.source == UNTAGGED
    ]


def untagged(dataset: str | None = None) -> list[Provenance]:
    """Every leaf no rule matched. Should be empty; the test suite asserts it."""
    return [p for p in audit(dataset) if p.source == UNTAGGED]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(dataset: str | None = None, cls: str | None = None) -> str:
    """Render the human-readable provenance report."""
    lines: list[str] = []
    names = [dataset] if dataset else list(DATASETS)

    if cls is not None:
        hits = [p for n in names for p in audit(n, cls=cls)]
        lines.append(f"=== {cls}: {len(hits)} field(s) ===")
        sources = load_sources()
        grouped: dict[str, list[Provenance]] = {}
        for p in hits:
            grouped.setdefault(p.source, []).append(p)
        for key in sorted(grouped):
            ps = grouped[key]
            lines.append("")
            lines.append(f"[{key}] {len(ps)} field(s)")
            note = " ".join(str(sources[key].get("note", "")).split())
            if note:
                lines.append("  " + _wrap(note, 2))
            for p in ps:
                lines.append(f"    {p.dataset}:{p.path} = {p.value!r}")
        return "\n".join(lines)

    total = coverage(dataset)
    n = sum(total.values())
    lines.append("=" * 72)
    lines.append("difflow_ree data provenance")
    lines.append("=" * 72)
    lines.append(f"{n} fields across {len(names)} file(s)")
    lines.append("")
    for c, k in total.items():
        flag = "   <- not publishable" if c in UNPUBLISHABLE else ""
        lines.append(f"  {c:<12} {k:5d}  {100 * k / n:5.1f}%{flag}")

    bad = sum(total.get(c, 0) for c in UNPUBLISHABLE)
    lines.append("")
    lines.append(
        f"  {bad} of {n} fields ({100 * bad / n:.0f}%) must not stand behind a "
        "published number."
    )
    lines.append("  See them with: python -m difflow_ree.provenance --cls HAND_TUNED")

    miss = untagged(dataset)
    if miss:
        lines.append("")
        lines.append(f"  *** {len(miss)} UNTAGGED field(s) -- add a rule: ***")
        for p in miss[:20]:
            lines.append(f"      {p.dataset}:{p.path}")
        if len(miss) > 20:
            lines.append(f"      ... and {len(miss) - 20} more")

    lines.append("")
    lines.append("-" * 72)
    lines.append("by source")
    lines.append("-" * 72)
    by_source: dict[str, int] = {}
    for p in audit(dataset):
        by_source[p.source] = by_source.get(p.source, 0) + 1
    sources = load_sources()
    for key in sorted(by_source, key=lambda k: -by_source[k]):
        entry = sources[key]
        lines.append(
            f"  {key:<10} {by_source[key]:5d}  [{entry['cls']}]"
        )
        lines.append(
            "             " + _wrap(" ".join(str(entry["citation"]).split()), 13)
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m difflow_ree.provenance",
        description="Report where the numbers in difflow_ree/data came from.",
    )
    ap.add_argument("--dataset", choices=sorted(DATASETS), default=None)
    ap.add_argument(
        "--cls",
        default=None,
        help="List every field of one class, e.g. HAND_TUNED.",
    )
    ap.add_argument(
        "--explain",
        default=None,
        metavar="PATH",
        help="Explain one dotted field path (requires --dataset).",
    )
    args = ap.parse_args(argv)

    if args.explain:
        if not args.dataset:
            ap.error("--explain requires --dataset")
        print(explain(args.dataset, args.explain))
        return 0

    print(report(args.dataset, args.cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
