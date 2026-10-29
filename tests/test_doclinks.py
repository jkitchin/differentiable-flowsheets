"""Tests for difflow.workspace.doclinks: a documentation link per operation.

The resolver is checked on a synthetic index; the guard asserts every
registered operation has its own section (or labelled row) in docs/, so
adding a unit operation without writing it up fails here (#228).
"""

from difflow.catalog import catalog
from difflow.workspace import doclinks, docs_index


def _index(*sections):
    """An index of hand-written sections, defaulted for brevity."""
    filled = [{"source": "unit-operations-chemical.md", "heading": "H",
               "path": "", "anchor": "h", "targets": [], "text": "",
               **section} for section in sections]
    return {"version": 2, "n_sections": len(filled), "sections": filled}


def test_label_beats_heading_beats_mention():
    mention = {"heading": "Gotchas", "anchor": "gotchas", "text": "A CSTR is easy."}
    heading = {"heading": "CSTR", "anchor": "cstr", "text": "prose"}
    label = {"heading": "Ref", "anchor": "ref", "targets": ["op-cstr"], "text": "x"}
    assert doclinks._resolve("CSTR", _index(mention)["sections"]).kind == "mention"
    assert doclinks._resolve("CSTR", _index(mention, heading)["sections"]).anchor == "cstr"
    link = doclinks._resolve("CSTR", _index(mention, heading, label)["sections"])
    assert (link.kind, link.anchor) == ("target", "op-cstr")


def test_name_must_match_a_whole_word():
    index = _index({"heading": "CompressorBoost", "anchor": "boost",
                    "text": "`CompressorBoost(ratio)`", "targets": ["op-gaspipe"]})
    assert doclinks._resolve("Compressor", index["sections"]) is None
    assert doclinks._resolve("Pipe", index["sections"]) is None


def test_a_missing_index_is_not_an_error(monkeypatch):
    monkeypatch.setattr(docs_index, "load", lambda *a, **k: None)
    doclinks.resolve.cache_clear()
    try:
        assert doclinks.url_for("CSTR") is None
    finally:
        doclinks.resolve.cache_clear()


def test_url_is_the_published_html_page():
    link = doclinks.DocLink("GasPipe", "unit-operations-gas.md",
                            "gas-op-gaspipe", "Reference", "target")
    assert link.url == doclinks.BASE_URL + "unit-operations-gas.html#gas-op-gaspipe"


def test_every_operation_has_its_own_documentation():
    index = docs_index.load()
    assert index is not None, "run `make docs-index`"
    links = {name: doclinks._resolve(name, index["sections"]) for name in catalog()}
    missing = sorted(n for n, link in links.items() if link is None)
    assert missing == [], "write a docs/ section for each of these operations"
    weak = sorted(n for n, link in links.items() if link.kind == "mention")
    assert weak == [], "give each its own heading or an `(op-<name>)=` label"
    for name, link in links.items():
        if link.kind == "target":
            page = docs_index.DOCS / link.source
            assert f"({link.anchor})=" in page.read_text(), name
