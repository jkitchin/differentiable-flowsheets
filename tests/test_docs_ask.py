"""The book's "Ask" assistant: the index builder and the shipped ask.js.

The corpus check (ranking over the real book) needs a built site and lives
in ``tests/docs_ask/ask_retrieval.mjs``, run by ``make ask-check`` and the
deploy workflow. These tests need neither: they feed the builder pages
shaped like Sphinx's output, and run the shipped script's retrieval half
under node on the index that comes out.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ASK_JS = ROOT / "_ext" / "ask_static" / "ask.js"

_spec = importlib.util.spec_from_file_location("ask_index", ROOT / "_ext" / "ask_index.py")
ask_index = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ask_index)


def page(body: str) -> str:
    """A page in the shape sphinx-book-theme writes it."""
    return f"""<html><head><script>var x = "not prose";</script></head><body>
<nav class="bd-docs-nav"><a href="x.html">Sidebar entry on every page</a></nav>
<main><article class="bd-article">{body}</article></main>
<footer>Footer on every page</footer></body></html>"""


RECYCLE = page("""
<section id="recycle-solving">
<h1>Recycle Solving<a class="headerlink" href="#recycle-solving">#</a></h1>
<p>A recycle loop is a fixed point and needs a starting guess for its tear.</p>
<section id="when-it-fails">
<h2>When it fails<a class="headerlink" href="#when-it-fails">#</a></h2>
<p>Call <code class="docutils literal notranslate"><span class="pre">fs.solve_eo()</span></code>
to switch to Newton on the whole equation set.</p>
<ul class="simple"><li><p>Damp the iteration.</p></li><li><p>Change the acceleration.</p></li></ul>
<div class="highlight-python notranslate"><div class="highlight"><pre><span></span>streams = fs.solve(on_nonconvergence="raise")
print(streams)
</pre></div></div>
<section id="deep">
<h4>A deep heading</h4>
<p>Text under a fourth-level heading still gets the section Sphinx wrote.</p>
</section>
</section>
</section>
""")

NOTEBOOK = page("""
<section id="example">
<h1>An Example Notebook</h1>
<p>This notebook shows a reactor and a flash with a recycle stream.</p>
<div class="cell docutils container">
<div class="cell_input docutils container"><div class="highlight"><pre>""" +
                "\n".join(f"x{i} = {i}" for i in range(50)) + """</pre></div></div>
<div class="cell_output docutils container"><div class="output stream highlight-myst-ansi">
<pre>OUTPUT NOISE 0.123456789</pre></div></div>
</div>
</section>
""")


def test_sections_become_cited_passages():
    chunks = ask_index.page_chunks(RECYCLE, "docs/recycle.html")
    by_url = {c["u"]: c for c in chunks}
    # The page's own top section is cited as the page, the rest by anchor.
    assert set(by_url) == {"docs/recycle.html", "docs/recycle.html#when-it-fails",
                           "docs/recycle.html#deep"}
    assert all(c["t"] == "Recycle Solving" for c in chunks)
    fails = by_url["docs/recycle.html#when-it-fails"]
    assert fails["h"] == "Recycle Solving › When it fails"
    assert by_url["docs/recycle.html#deep"]["h"].endswith("› A deep heading")


def test_text_keeps_the_markup_a_model_reads():
    fails = {c["u"]: c for c in ask_index.page_chunks(RECYCLE, "r.html")}["r.html#when-it-fails"]
    text = fails["x"]
    assert "`fs.solve_eo()`" in text
    assert "- Damp the iteration." in text and "- Change the acceleration." in text
    assert "```\nstreams = fs.solve(on_nonconvergence=\"raise\")\nprint(streams)\n```" in text


def test_page_chrome_and_pilcrows_are_left_out():
    blob = json.dumps(ask_index.page_chunks(RECYCLE, "r.html"))
    for noise in ("Sidebar entry", "Footer on every page", "not prose", "#\"", "headerlink"):
        assert noise not in blob
    assert all(not c["h"].endswith("#") for c in ask_index.page_chunks(RECYCLE, "r.html"))


def test_notebook_outputs_dropped_and_long_cells_cut():
    (chunk,) = ask_index.page_chunks(NOTEBOOK, "examples/nb.html")
    assert "OUTPUT NOISE" not in chunk["x"]
    assert "x29 = 29" in chunk["x"] and "x30 = 30" not in chunk["x"]
    assert "(20 more lines)" in chunk["x"]


def test_long_sections_split_at_paragraphs():
    paras = [f"Paragraph {i} " + "word " * 60 for i in range(20)]
    parts = ask_index.split_long("\n\n".join(paras))
    assert len(parts) > 1
    assert all(len(p) <= ask_index.MAX_CHARS for p in parts)
    assert all(p.startswith("Paragraph") for p in parts)


def test_build_walks_content_pages_only(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "_static").mkdir()
    (tmp_path / "docs" / "recycle.html").write_text(RECYCLE)
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "nb.html").write_text(NOTEBOOK)
    (tmp_path / "genindex.html").write_text(RECYCLE)
    (tmp_path / "_static" / "x.html").write_text(RECYCLE)
    out = ask_index.write(tmp_path)
    doc = json.loads(out.read_text())
    assert out.name == "ask-index.json"
    assert doc["counts"] == {"pages": 2, "total": len(doc["chunks"])}
    assert {c["u"].split("#")[0] for c in doc["chunks"]} == {"docs/recycle.html",
                                                             "examples/nb.html"}


def test_extension_registers_its_assets():
    calls = {}

    class App:
        def connect(self, event, fn):
            calls.setdefault("events", []).append(event)

        def add_js_file(self, name, **kw):
            calls["js"] = (name, kw)

        def add_css_file(self, name, **kw):
            calls["css"] = name

    ask_index.setup(App())
    assert calls["js"] == ("ask.js", {"loading_method": "defer", "id": "difflow-ask"})
    assert calls["css"] == "ask.css"
    assert set(calls["events"]) == {"config-inited", "build-finished"}
    assert (ask_index.STATIC / "ask.js").is_file() and (ask_index.STATIC / "ask.css").is_file()


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not on PATH")
def test_shipped_script_ranks_the_built_index(tmp_path):
    """The real ask.js, through its test seam, over an index this builder wrote."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "recycle.html").write_text(RECYCLE)
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "nb.html").write_text(NOTEBOOK)
    index = ask_index.write(tmp_path)
    script = f"""
const ask = require({json.dumps(str(ASK_JS))});
const doc = require({json.dumps(str(index))});
const idx = ask.buildIndex(doc.chunks);
const top = (q) => (ask.search(idx, q, 6)[0] || {{chunk: {{u: null}}}}).chunk.u;
console.log(JSON.stringify({{
  solve_eo: top("what does solve_eo do"),
  damping: top("damping the iterations"),
  notebook: top("example notebook reactor"),
  none: ask.search(idx, "zzzqqq", 6).length,
  stop: ask.queryTerms(idx, "what does it do").length,
  prompt: ask.buildPrompt("q?", ask.search(idx, "recycle", 5))[0].content,
}}));
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                            check=True, timeout=60)
    got = json.loads(result.stdout)
    assert got["solve_eo"] == "docs/recycle.html#when-it-fails"
    assert got["damping"] == "docs/recycle.html#when-it-fails"  # "damping" ~ "Damp"
    assert got["notebook"] == "examples/nb.html"
    assert got["none"] == 0 and got["stop"] == 0
    assert "difflow" in got["prompt"] and "ONLY from the numbered excerpts" in got["prompt"]
