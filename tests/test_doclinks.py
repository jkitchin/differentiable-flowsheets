"""Tests for difflow.gui.doclinks: a documentation link per operation.

Two kinds of test here, and the second is the point.

The unit tests pin the resolver's behaviour on a synthetic index ---
which of the three ways of finding a unit wins, how ties break, what a
missing index does.

The rest are a **guard on the prose**. `url_for` is the only function
that answers "where is this unit written up?", so asserting over the
whole registry turns a documentation gap into a test failure: every one
of the registered operations must resolve, and must resolve to a section
or a labelled row that is *about* it rather than to a page that merely
mentions it in passing. Adding a unit operation without writing it up
fails here. That is deliberate --- the alternative is what #228
measured, 40 of 87 operations with nowhere of their own to link to, found
only because someone went looking.
"""

import re
from pathlib import Path

import pytest

from difflow.catalog import catalog
from difflow.gui import doclinks, docs_index


def _index(*sections):
    """An index of hand-written sections, defaulted for brevity."""
    filled = [{"source": "unit-operations-chemical.md", "heading": "H",
               "path": "", "anchor": "h", "targets": [], "text": "",
               **section} for section in sections]
    return {"version": 2, "n_sections": len(filled), "sections": filled}


# -- the resolver -----------------------------------------------------

class TestResolution:
    """Which of the three ways of finding an operation wins."""

    def test_a_heading_beats_a_mention_elsewhere(self):
        index = _index(
            {"heading": "Gotchas", "anchor": "gotchas",
             "text": "A CSTR is easy to get wrong."},
            {"heading": "CSTR", "anchor": "cstr", "text": "prose"},
        )
        link = doclinks._resolve("CSTR", index["sections"])
        assert (link.kind, link.anchor) == ("heading", "cstr")

    def test_an_explicit_label_beats_a_heading(self):
        # The row in a reference table is the better destination than the
        # top of the table that contains it.
        index = _index(
            {"heading": "GasPipe in context", "anchor": "gaspipe-in-context",
             "text": "prose"},
            {"source": "unit-operations-gas.md", "heading": "Reference",
             "anchor": "reference", "targets": ["gas-op-gaspipe"],
             "text": "`GasPipe(beta)` --- forward mode"},
        )
        link = doclinks._resolve("GasPipe", index["sections"])
        assert (link.kind, link.anchor) == ("target", "gas-op-gaspipe")

    def test_a_label_is_recognised_prefixed_or_bare(self):
        for label in ("op-gaspipe", "gas-op-gaspipe", "gaspipe"):
            index = _index({"targets": [label], "text": "prose"})
            link = doclinks._resolve("GasPipe", index["sections"])
            assert (link.kind, link.anchor) == ("target", label), label

    def test_a_label_for_another_unit_is_not_a_match(self):
        # `op-gaspipe` must not answer for `Pipe`, nor `op-pipe` for
        # `GasPipe`: the suffix rule is anchored at a `-` boundary.
        index = _index({"targets": ["op-gaspipe"], "text": "prose"})
        assert doclinks._resolve("Pipe", index["sections"]) is None

    def test_an_exact_heading_beats_a_longer_one(self):
        index = _index(
            {"heading": "CSTR Design Considerations", "anchor": "cstr-design"},
            {"heading": "CSTR", "anchor": "cstr"},
        )
        assert doclinks._resolve("CSTR", index["sections"]).anchor == "cstr"

    def test_a_heading_that_starts_with_the_name_still_counts(self):
        index = _index({"heading": "PFR (Plug Flow Reactor)",
                        "anchor": "pfr-plug-flow-reactor"})
        link = doclinks._resolve("PFR", index["sections"])
        assert (link.kind, link.anchor) == ("heading", "pfr-plug-flow-reactor")

    def test_the_name_must_appear_as_a_whole_word(self):
        # Or `Compressor` would be answered by `CompressorBoost`'s row.
        index = _index({"heading": "CompressorBoost", "anchor": "boost",
                        "text": "`CompressorBoost(ratio, direction)`"})
        assert doclinks._resolve("Compressor", index["sections"]) is None

    def test_a_reference_page_outranks_a_tutorial(self):
        index = _index(
            {"source": "streams-and-flowsheets.md", "heading": "Registry",
             "anchor": "registry", "text": "EOSCompressor is registered here"},
            {"heading": "Units", "anchor": "units",
             "text": "EOSCompressor compresses"},
        )
        link = doclinks._resolve("EOSCompressor", index["sections"])
        assert link.source == "unit-operations-chemical.md"

    def test_the_wordiest_mention_wins_among_mentions(self):
        index = _index(
            {"heading": "Aside", "anchor": "aside", "text": "TFF once"},
            {"heading": "Filtration", "anchor": "filtration",
             "text": "TFF does this. TFF does that. TFF again."},
        )
        link = doclinks._resolve("TFF", index["sections"])
        assert (link.kind, link.anchor) == ("mention", "filtration")

    def test_an_unmentioned_operation_resolves_to_nothing(self):
        assert doclinks._resolve("Nonesuch", _index({})["sections"]) is None

    def test_a_missing_index_is_not_an_error(self, monkeypatch):
        monkeypatch.setattr(docs_index, "load", lambda *a, **k: None)
        doclinks.resolve.cache_clear()
        try:
            assert doclinks.resolve("CSTR") is None
            assert doclinks.url_for("CSTR") is None
        finally:
            doclinks.resolve.cache_clear()


class TestUrl:
    """The URL a link turns into."""

    def test_the_markdown_page_becomes_the_published_html_page(self):
        link = doclinks.DocLink("GasPipe", "unit-operations-gas.md",
                                "gas-op-gaspipe", "Reference", "target")
        assert link.page == "unit-operations-gas.html"
        assert link.url == (doclinks.BASE_URL
                            + "unit-operations-gas.html#gas-op-gaspipe")

    def test_an_anchorless_link_is_the_page_itself(self):
        link = doclinks.DocLink("X", "x.md", "", "X", "heading")
        assert link.url.endswith("x.html")


# -- the guard on docs/ -----------------------------------------------

@pytest.fixture(scope="module")
def links():
    """Every registered operation, resolved against the committed index."""
    index = docs_index.load()
    assert index is not None, "run `make gui-build`"
    return {name: doclinks._resolve(name, index["sections"])
            for name in catalog()}


class TestEveryOperationIsDocumented:
    """#228: a registered operation with no prose is a defect."""

    def test_every_operation_resolves_to_something(self, links):
        missing = sorted(name for name, link in links.items() if link is None)
        assert missing == [], (
            "these operations are not named anywhere in docs/ --- write a "
            "section for each, do not add them to an exception list"
        )

    def test_no_operation_has_to_fall_back_to_a_passing_mention(self, links):
        # A `mention` link lands on a section about something else. The
        # resolver offers it so a link is never withheld; the prose is
        # not allowed to need it.
        weak = {name: (link.source, link.heading)
                for name, link in links.items() if link.kind == "mention"}
        assert weak == {}, (
            "these operations only appear inside a section about something "
            "else: give each its own heading, or label its row with "
            "`(op-<lowercase name>)=`"
        )

    def test_every_label_a_link_uses_is_defined_in_that_page(self, links):
        # The anchor has to exist in the prose, not merely in the index:
        # a renamed label would otherwise go on resolving to a URL that
        # 404s in the built book.
        for name, link in links.items():
            if link.kind != "target":
                continue
            page = docs_index.DOCS / link.source
            assert f"({link.anchor})=" in page.read_text(), (
                f"{name}: no `({link.anchor})=` in {link.source}")

    def test_every_heading_a_link_uses_gets_an_anchor_in_the_book(self, links):
        # Jupyter Book only generates heading anchors down to
        # `myst_heading_anchors` (h3 here). A deeper heading has no
        # anchor to link to, so it needs an explicit label instead ---
        # which is what the two `#####` flash sections have.
        depth = _myst_heading_anchors()
        for name, link in links.items():
            if link.kind != "heading":
                continue
            level = _heading_level(docs_index.DOCS / link.source, link.heading)
            assert level is not None, f"{name}: heading vanished from prose"
            assert level <= depth, (
                f"{name}: links to an h{level} heading, but the book only "
                f"generates anchors to h{depth}; label the section with "
                f"`({link.anchor})=` instead")

    def test_a_unit_operations_page_is_where_a_unit_is_documented(self, links):
        # Not a hard rule of the resolver, but true of all 87, and worth
        # noticing if it stops being: a unit documented only in a
        # tutorial is a unit nobody finds.
        strays = {name: link.source for name, link in links.items()
                  if not link.source.startswith("unit-operations-")}
        assert strays == {}


def _myst_heading_anchors() -> int:
    """The heading depth ``_config.yml`` asks Jupyter Book to anchor."""
    config = Path(docs_index.DOCS).parent / "_config.yml"
    match = re.search(r"^\s*myst_heading_anchors:\s*(\d+)",
                      config.read_text(), re.M)
    assert match, "myst_heading_anchors is not set in _config.yml"
    return int(match.group(1))


def _heading_level(page: Path, heading: str) -> int | None:
    """The ``#`` depth of *heading* in *page*, outside code fences."""
    fenced = False
    for line in page.read_text().splitlines():
        if re.match(r"^\s*(```|~~~)", line):
            fenced = not fenced
            continue
        if fenced:
            continue
        match = re.match(r"^(#{1,6})\s+(.*?)\s*#*$", line)
        if match and match.group(2) == heading:
            return len(match.group(1))
    return None
