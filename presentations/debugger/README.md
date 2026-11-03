# The flowsheet debugger, step by step (a Marp deck)

A talk-length walk through `difflow debug`: starting a session, stepping by unit and by tear
iteration, reading the tear, breakpoints, finding where a NaN is born, `set tear`, record and
what-if replays, the JSON protocol, the agent tools, and the same debugger in a Jupyter notebook. Everything in it
is open-source difflow: no editor or app.

- **`debugger.html`**: open it in a browser. The terminal sessions play as animations and the
  notebook recordings as videos (← → to move between slides, `f` for full screen).
- **`debugger.pdf`**: the same slides, with each recording's last frame in place of the animation.
- **`debugger.md`**: the source, in [Marp](https://marp.app) Markdown.

Every recording is a real run, not a mock-up. The terminal GIFs replay what the `difflow debug`
CLI printed for `demo/plant.py` (an isomerization loop with a recycle) and `demo/faulty.py` (the
same loop with a preheater that turns NaN below 400 K); the transcript of each is beside it in
`assets/`. The notebook videos drive JupyterLab with Playwright, running the two notebooks in
`demo/` (`debug-in-a-notebook.ipynb`, `record-and-what-if.ipynb`); open them to try it yourself.

## Rebuilding

```bash
./build.sh               # debugger.html and debugger.pdf (needs node; the PDF needs Chrome: CHROME_PATH)
./record.sh              # re-record the terminal sessions and redraw the figures (needs difflow, ffmpeg)
```

`record.sh` also says how to re-record the notebook videos, which need JupyterLab and Playwright.

| File | What |
|---|---|
| `record_term.py` | runs the CLI with a command script and renders the session as a typed-terminal GIF |
| `record_notebook.js` | runs the `demo/` notebooks in JupyterLab with Playwright, with a visible cursor |
| `make_figures.py` | the flowsheet diagram and the what-if plot |
| `demo/` | the flowsheets the recordings run |
