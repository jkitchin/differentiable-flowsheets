# Ask: the in-browser docs assistant

The floating **Ask** button in the lower-right corner of every page opens a
question box for this book. It has two halves, and they work independently:

| | What it does | What it costs |
|---|---|---|
| **Search** | Ranks passages from the guides, the tutorials and the example notebooks, each linked to the section it came from | one index download (about 1 MB compressed) the first time the panel opens |
| **Answer** | Writes those passages up into prose, with numbered citations | a language model download, roughly 1–2 GB, only if you ask for it |

Search always works. The answer half is opt-in: nothing is downloaded until
you pick a model and click **Load model**. If you never do, you still get
ranked, linked passages, which is most of the value on a reference manual.

Nothing you type leaves your browser. There is no API key, no server and no
telemetry. The assistant is a static script; the only third-party requests it
makes are for the WebLLM runtime and the model weights, after you click.

The design, and most of the code for the search half, comes from the
assistant on the [POUNCE documentation](https://kitchingroup.cheme.cmu.edu/pounce/).

## What it searches

Every page of this book, including the notebooks under *About Flowsheets*,
*JAX Tutorials* and *Examples*. Notebook passages are marked `example` in the
source list: they show a worked call rather than define it. Notebook outputs
are left out; the input cells are kept, cut to their first thirty lines.

Passages are cut at section boundaries, so every citation lands on the section
that answered rather than at the top of a long page.

## Requirements for written answers

| | |
|---|---|
| **Browser** | WebGPU: Chrome or Edge 113+, Safari 26+, Firefox with WebGPU enabled |
| **Download** | 0.9–2.3 GB depending on the model, cached by the browser afterwards |
| **Memory** | roughly the model's size in GPU memory while it is loaded |

Without WebGPU the panel says so and stays in search-only mode. It asks the
browser for a GPU adapter before offering the button, because a browser can
advertise WebGPU and still have no usable GPU behind it.

Three models are offered, smallest first: Llama 3.2 1B, Qwen 2.5 1.5B and
Llama 3.2 3B. The smallest is the default, because a download that finishes
beats a larger one you abandon; the larger ones follow the instructions and
the citation format more reliably. A GPU that supports half precision gets the
smaller half-precision builds.

## How much to trust it

The model sees only the five passages your question retrieved, and it is told
to answer from them alone and to cite each claim. That keeps it much closer to
the book than an unaided chatbot, but a small model reading the right passages
can still summarize them wrongly.

**Treat the answer as a pointer and the passages as the source.** Every answer
is shown with the passages it was built from, in rank order, each linked to
its section. When the two disagree, the passage is right.

Search is [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) over words, not
embeddings. It is good at exact names (a class, a keyword argument, a warning
copied out of your traceback) and weaker at questions that share no word with
the book. If a question comes back empty, search for the name instead.

## For maintainers

| Piece | File |
|---|---|
| Sphinx extension and index builder | `_ext/ask_index.py` |
| Panel, retrieval, WebLLM glue | `_ext/ask_static/ask.js`, `_ext/ask_static/ask.css` |
| Retrieval check over the built book | `tests/docs_ask/ask_retrieval.mjs` (`make ask-check`) |
| Builder and script unit tests | `tests/test_docs_ask.py` |

`jupyter-book build .` does everything: the extension, listed under
`sphinx.local_extensions` in `_config.yml`, adds the script and stylesheet to
every page and writes `ask-index.json` at the root of `_build/html` when the
build finishes. The index is built from the rendered HTML rather than the
markdown, so a citation names a section anchor only if Sphinx actually wrote
it. To rebuild the index alone after a build, run
`python3 _ext/ask_index.py _build/html`.

To try it, serve the build over HTTP (the index is fetched, which a `file://`
page cannot do):

```console
$ make book
$ python3 -m http.server -d _build/html 8000    # http://localhost:8000/
```

The ranking has no exception to throw when it gets worse: it returns the wrong
passage, silently. `make ask-check` runs the shipped `ask.js` against a
labelled set of questions over the built index, and the deploy workflow runs
the same check before it publishes.
