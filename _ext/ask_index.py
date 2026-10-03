"""The in-browser docs assistant ("Ask") for the published book.

A floating **Ask** button on every page of the Jupyter Book opens a
question box. It has two halves that work independently, after the
assistant on the POUNCE docs site (jkitchin/pounce, ``docs/assets/ask.js``):

* **Retrieval.** BM25 over ``ask-index.json``, which this module writes
  at the root of the built site. Pure JavaScript, no model; it always
  works and on its own returns ranked, deep-linked passages.
* **Generation.** Optional. A small instruct model run in the reader's
  browser by WebLLM over WebGPU, handed only the passages the question
  retrieved. Nothing is downloaded until the reader clicks *Load model*.

This file is both a Sphinx extension (listed under ``sphinx:
local_extensions`` in ``_config.yml``) and a command:

    python3 _ext/ask_index.py _build/html        # rebuild the index only

As an extension it does three things: puts ``_ext/ask_static/`` on the
static path, adds ``ask.js``/``ask.css`` to every page, and on
``build-finished`` walks the rendered HTML and writes the index. So
``jupyter-book build .`` (``make book``, and the deploy workflow) needs
nothing else.

Why the index is built from the RENDERED HTML and not from the markdown,
which is what the POUNCE builder reads: every citation is a link to a
section anchor, and an anchor that does not exist is a link that silently
lands at the top of the page. POUNCE reimplements mdBook's slug rule to
predict its anchors. Here the rule would be MyST's for headings down to
``myst_heading_anchors`` and docutils' below that, with each one's own
duplicate suffixing --- and a third of the book is notebooks, which are
not markdown at all. Reading ``<section id=...>`` out of the output
instead means an anchor is cited only if Sphinx actually wrote it.

Stdlib only, because it runs inside whatever environment builds the book.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

#: ``ask.js`` and ``ask.css``, copied into ``_static/`` of the build.
STATIC = Path(__file__).resolve().parent / "ask_static"

#: Written at the root of the built site; ``ask.js`` looks for it there.
INDEX_NAME = "ask-index.json"

# Passage sizing. MAX_CHARS is a retrieval choice, not a model-context one:
# short enough that a hit is specific, long enough that a worked example or
# a parameter table survives in one piece. MIN_CHARS drops heading-only
# stubs, which would win on a title match and carry no answer.
MAX_CHARS = 1600
MIN_CHARS = 40

#: A long code cell crowds prose out of a small model's window and rarely
#: holds the sentence that answers. Keep enough to show the shape of a call.
MAX_CODE_LINES = 30

#: Generated pages with no prose of their own.
SKIP_PAGES = {"genindex.html", "search.html", "py-modindex.html"}

_BLOCK = {"p", "div", "blockquote", "table", "ul", "ol", "dl", "figure",
          "figcaption", "dt", "dd", "details", "summary"}
_VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "col", "source"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}


def _classes(attrs) -> set[str]:
    return set((dict(attrs).get("class") or "").split())


def _skipped(tag: str, attrs) -> bool:
    """Subtrees that are not prose a reader would want quoted back."""
    cls = _classes(attrs)
    if tag in ("script", "style", "nav", "button", "template", "svg"):
        return True
    if tag == "a" and "headerlink" in cls:  # the pilcrow after a heading
        return True
    # Notebook output: progress bars, arrays, tracebacks, plots. The input
    # cell carries the call; the output is noise at retrieval time.
    if "cell_output" in cls or "output" in cls and tag == "div":
        return True
    return False


class _Page(HTMLParser):
    """One built page as ``[(trail, anchor, text)]``, in document order.

    Only the article body is read --- the sidebar, header and footer are
    the same on every page and would make every page match every query.
    Sections nest, so text is flushed to the innermost open section each
    time one opens or closes.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.article = 0          # >0 inside <article class="bd-article">
        self.skip: list[str] = []  # open skipped tags, innermost last
        self.stack: list[dict] = []
        self.buf: list[str] = []
        self.heading: list[str] | None = None
        self.pre: list[str] | None = None
        self.in_code = 0
        self.out: list[tuple[list[str], str, str]] = []

    # -- structure ---------------------------------------------------------

    def flush(self):
        text = _normalise("".join(self.buf))
        self.buf = []
        if not text or not self.stack:
            return
        trail = [s["heading"] for s in self.stack if s["heading"]]
        # The page's own top section is cited as the page, not as an anchor.
        anchor = self.stack[-1]["id"] if len(self.stack) > 1 else ""
        self.out.append((trail, anchor, text))

    def handle_starttag(self, tag, attrs):
        if tag == "article" and "bd-article" in _classes(attrs):
            self.article += 1
            return
        if not self.article:
            return
        if self.skip:
            if tag not in _VOID and tag == self.skip[-1]:
                self.skip.append(tag)
            return
        if _skipped(tag, attrs):
            if tag not in _VOID:
                self.skip.append(tag)
            return

        if tag == "section":
            self.flush()
            self.stack.append({"id": dict(attrs).get("id") or "", "heading": ""})
        elif tag in _HEADINGS:
            self.flush()
            self.heading = []
        elif tag == "pre":
            self.pre = []
        elif self.pre is not None:
            return
        elif tag == "code":
            self.in_code += 1
            self._emit("`")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "tr":
            self._emit("\n")
        elif tag in ("td", "th"):
            self._emit(" | ")
        elif tag == "br":
            self._emit("\n")
        elif tag in _BLOCK:
            self._emit("\n\n")

    def handle_endtag(self, tag):
        if not self.article:
            return
        if self.skip:
            if tag == self.skip[-1]:
                self.skip.pop()
            return
        if tag == "article":
            self.flush()
            self.article -= 1
        elif tag == "section":
            self.flush()
            if self.stack:
                self.stack.pop()
        elif tag in _HEADINGS and self.heading is not None:
            text = re.sub(r"\s+", " ", "".join(self.heading)).strip()
            if self.stack and not self.stack[-1]["heading"]:
                self.stack[-1]["heading"] = text
            self.heading = None
        elif tag == "pre" and self.pre is not None:
            lines = "".join(self.pre).strip("\n").split("\n")
            if len(lines) > MAX_CODE_LINES:
                more = len(lines) - MAX_CODE_LINES
                lines = lines[:MAX_CODE_LINES] + [f"# ... ({more} more lines)"]
            self.pre = None
            self.buf.append("\n\n```\n" + "\n".join(lines) + "\n```\n\n")
        elif self.pre is not None:
            return
        elif tag == "code" and self.in_code:
            self.in_code -= 1
            self._emit("`")
        elif tag in _BLOCK:
            self._emit("\n\n")

    def handle_data(self, data):
        if not self.article or self.skip:
            return
        if self.pre is not None:
            self.pre.append(data)
        elif self.heading is not None:
            self.heading.append(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))

    def _emit(self, text):
        if self.heading is not None:
            if text.strip():
                self.heading.append(text)
            return
        self.buf.append(text)


def _normalise(text: str) -> str:
    """Tidy whitespace outside code fences; keep paragraph breaks.

    Paragraph breaks matter beyond looks: :func:`split_long` cuts only
    there, so a section without them can only be cut mid-sentence.
    """
    out: list[str] = []
    fenced = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            fenced = not fenced
            out.append(line.strip())
        elif fenced:
            out.append(line.rstrip())
        else:
            out.append(re.sub(r" {2,}", " ", line).strip())
    text = "\n".join(out)
    # A list item wraps its text in <p>, so the bullet and its words arrive
    # a paragraph apart; join them.
    text = re.sub(r"(?m)^- *\n+(?=\S)", "- ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pack(units: list[str], sep: str, max_chars: int) -> list[str]:
    """Greedily pack units into runs no longer than max_chars."""
    parts: list[str] = []
    buf = ""
    for unit in units:
        if not buf:
            buf = unit
        elif len(buf) + len(sep) + len(unit) <= max_chars:
            buf += sep + unit
        else:
            parts.append(buf)
            buf = unit
    if buf:
        parts.append(buf)
    return parts


def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an over-long section into passages.

    Paragraph boundaries first, then lines for a paragraph that is still too
    long (a table is one paragraph, and the unit-operation parameter tables
    run long). A single line longer than max_chars is emitted whole:
    cutting mid-sentence costs more than the overrun.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    for para in _pack(text.split("\n\n"), "\n\n", max_chars):
        if len(para) <= max_chars:
            parts.append(para)
        else:
            parts.extend(_pack(para.split("\n"), "\n", max_chars))
    return parts


def page_chunks(html: str, url: str) -> list[dict]:
    """Every passage of one built page.

    ``t`` page title, ``h`` heading trail, ``u`` URL relative to the site
    root, ``x`` the text. Keys are short because every reader who opens
    the assistant downloads this file.
    """
    parser = _Page()
    parser.feed(html)
    parser.close()
    sections = parser.out
    title = next((trail[0] for trail, _a, _x in sections if trail), url)
    chunks = []
    for trail, anchor, text in sections:
        heading = " › ".join(trail) or title
        for piece in split_long(text):
            if len(piece) < MIN_CHARS:
                continue
            chunks.append({"t": title, "h": heading,
                           "u": url + ("#" + anchor if anchor else ""),
                           "x": piece})
    return chunks


def pages(outdir: Path):
    """The content pages of a built site, in a stable order."""
    for path in sorted(outdir.rglob("*.html")):
        rel = path.relative_to(outdir).as_posix()
        if rel in SKIP_PAGES or any(part.startswith("_") for part in rel.split("/")):
            continue
        yield rel, path


def build(outdir: Path) -> dict:
    chunks: list[dict] = []
    n_pages = 0
    for rel, path in pages(outdir):
        found = page_chunks(path.read_text(encoding="utf-8"), rel)
        if found:
            n_pages += 1
            chunks.extend(found)
    return {"schema": 1,
            "counts": {"pages": n_pages, "total": len(chunks)},
            "chunks": chunks}


def write(outdir: Path) -> Path:
    out = Path(outdir) / INDEX_NAME
    doc = build(Path(outdir))
    out.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
                   + "\n", encoding="utf-8")
    return out


# -- Sphinx ----------------------------------------------------------------

def _config_inited(app, config):
    config.html_static_path = list(config.html_static_path) + [str(STATIC)]


def _build_finished(app, exception):
    if exception is not None or getattr(app.builder, "format", "") != "html":
        return
    out = write(Path(app.outdir))
    doc = json.loads(out.read_text(encoding="utf-8"))
    print(f"ask_index: {doc['counts']['total']} passages from "
          f"{doc['counts']['pages']} pages -> {out} "
          f"({out.stat().st_size // 1024} KB)")


def setup(app):
    app.connect("config-inited", _config_inited)
    app.connect("build-finished", _build_finished)
    # `id` is how ask.js finds its own tag, and from its src the site root.
    app.add_js_file("ask.js", loading_method="defer", id="difflow-ask")
    app.add_css_file("ask.css")
    return {"version": "1", "parallel_read_safe": True,
            "parallel_write_safe": True}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0] if argv else "_build/html")
    if not outdir.is_dir():
        print(f"ask_index: no built site at {outdir}", file=sys.stderr)
        return 1
    out = write(outdir)
    doc = json.loads(out.read_text(encoding="utf-8"))
    print(f"ask_index: {doc['counts']['total']} passages from "
          f"{doc['counts']['pages']} pages -> {out} "
          f"({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
