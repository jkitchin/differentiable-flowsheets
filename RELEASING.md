# Releasing difflow

Developer & release guide for **difflow**, the JAX-based differentiable flowsheet
framework. This distribution (`difflow`) bundles **five** packages that ship together
from one wheel:

| Import package | Role |
| --- | --- |
| `difflow` | Core framework (streams, thermo, units, flowsheet, economics) |
| `difflow_bio` | Bio-manufacturing plugin |
| `difflow_ree` | Rare-earth-element extraction plugin |
| `difflow_cc` | Carbon-capture plugin |
| `difflow_gas` | Gas-transmission-network plugin |

The plugins register via the `difflow.plugins` entry points in `pyproject.toml`.

> **Release target.** difflow is **not yet on PyPI** and has **no `publish.yml`
> workflow**. Sections marked **[PyPI — optional]** only apply once you decide to
> publish to PyPI; skip them for a GitHub-tag-only release. See
> [PyPI publishing setup](#pypi-publishing-setup-one-time-optional) for the one-time work.

## Development

Everything runs through **uv** (the lockfile `uv.lock` is committed; the venv lives at
`/Users/jkitchin/Dropbox/uv/.venv`). The `VIRTUAL_ENV=` prefix in the Makefile prevents a
different activated venv from shadowing uv's.

| Task | Command | Make target |
| --- | --- | --- |
| Install (all extras) | `pip install -e ".[all]"` or `uv sync` | `make sync` |
| Run tests | `uv run pytest tests/ -v` | `make test` |
| Build the docs (Jupyter Book from repo root) | `uv run jupyter-book build .` | `make book` |
| Clean docs build | `uv run jupyter-book clean .` | `make book-clean` |
| Execute notebooks (stale only) | — | `make notebooks` |
| Force-execute all notebooks | — | `make notebooks-force` |
| Build artifacts | `rm -rf dist && uv build` | — |

Notes:
- Notebooks are pre-executed and committed. The Makefile tracks them with stamp files in
  `.notebook-stamps/`; `make notebooks` only re-runs a notebook when it or a `src/**/*.py`
  file it depends on has changed. Per-plugin targets exist: `make notebooks-bio`,
  `notebooks-ree`, `notebooks-cc` (and `*-force` variants).
- On macOS the Makefile forces `JAX_PLATFORM_NAME=cpu` (no GPU support).
- **No linter/formatter is configured** (no ruff/black/pre-commit). If you adopt one,
  add a lint gate to the pre-flight section below and to CI.
- **No `CHANGELOG.md` exists yet.** Consider creating one (Keep a Changelog format) so the
  version step below has a home for release notes.

---

## Release checklist

Work top to bottom.

> **⚠️ Version is duplicated in 4 places — not a single source of truth.** Unlike a
> single-`version` project, difflow bundles plugins that hard-code their own
> `__version__`. You **must** bump all of them together (see step 2). `difflow` itself
> reads its version from installed metadata (`importlib.metadata.version("difflow")` in
> `src/difflow/__init__.py`), so it needs no manual edit.

### 0. [PyPI — optional] Prerequisite gate — dependencies must be PyPI-resolvable ⛔

PyPI rejects any distribution that depends on a git/URL source, so every runtime
dependency must resolve from a package index.

- [ ] All runtime dependencies in `pyproject.toml` `[project].dependencies` are published
      on PyPI at the required versions.
- [ ] No `[tool.uv.sources]` git/URL overrides for runtime deps remain in `pyproject.toml`.
- [ ] `grep -rn "git+" pyproject.toml uv.lock` returns nothing for runtime dependencies.
- [ ] The distribution name `difflow` is available / owned by you on PyPI (currently
      **unclaimed** — first upload will register it).

### 1. Pre-flight — code health

- [ ] On `main`, clean working tree, synced with remote (`git status`, `git pull`).
- [ ] Latest CI run on `main` is green: `gh run list --branch main --limit 1`
      (the **Tests** workflow, `.github/workflows/test.yml`, runs py3.10/3.11/3.12).
- [ ] Full test suite passes locally: `make test` (= `uv run pytest tests/ -v`).
      This includes the plugin suites (`tests/bio/`, `tests/ree/`, `tests/cc/`, `tests/gas/`).
- [ ] Docs build clean: `make book` (`uv run jupyter-book build .`).
- [ ] If code feeding the notebooks changed, re-execute and re-commit them:
      `make notebooks` (or `make notebooks-force` to rebuild all).
- [ ] *(If/when a linter is configured)* Lint + format clean.

### 2. Version & metadata

Bump the version per [semver](https://semver.org) in **all four** locations:

- [ ] `pyproject.toml` line 7: `version = "X.Y.Z"`
- [ ] `src/difflow_gas/__init__.py` line 48: `__version__ = "X.Y.Z"`
- [ ] `src/difflow_ree/__init__.py` line 36: `__version__ = "X.Y.Z"`
- [ ] `src/difflow_cc/__init__.py` line 42: `__version__ = "X.Y.Z"`
- [ ] Sanity-check they all match:
      `grep -rn "0\.0\.0\|version" pyproject.toml src/difflow_*/__init__.py | grep -i version`
      (no stray old versions remain).
- [ ] Update `CHANGELOG.md` (create it if adopting): move `[Unreleased]` entries under the
      new version + today's date.
- [ ] Re-check `README.md` install instructions and extras (`[all]`, `[dev]`, `[examples]`,
      `[visualization]`, `cuda11`/`cuda12`).
- [ ] Re-check `pyproject.toml` metadata: `description`, `requires-python`, and consider
      adding a `license` field and `[project.urls]` (Homepage/Docs/Issues) and `authors` —
      they surface on the PyPI project page. (LICENSE is MIT.)

### 3. Build & verify artifacts

- [ ] `rm -rf dist && uv build` — always wipe first; a stale gitignored `dist/` may exist.
      Produces an sdist + a wheel (backend: **hatchling**).
- [ ] `uvx twine check dist/*` passes.
- [ ] Inspect the wheel — it must contain **all five** packages:
      `python -m zipfile -l dist/difflow-*.whl | grep -E "difflow(_bio|_ree|_cc|_gas)?/__init__.py"`
      (expect `difflow/`, `difflow_bio/`, `difflow_ree/`, `difflow_cc/`, `difflow_gas/`).
- [ ] Smoke test in a clean venv:
      ```bash
      python -m venv /tmp/df && /tmp/df/bin/pip install dist/difflow-*.whl
      /tmp/df/bin/python -c "import difflow, difflow_bio, difflow_ree, difflow_cc, difflow_gas; print(difflow.__version__)"
      /tmp/df/bin/difflow --help          # the `difflow` console script (report CLI)
      ```
- [ ] Plugin discovery works — the `difflow.plugins` entry points import and `register()`
      resolves (import the core package and confirm plugins load without error).

### 4. Tag & release

- [ ] Commit the version bump (+ CHANGELOG) and push to `main` (directly or via PR).
- [ ] Annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z"` then `git push origin vX.Y.Z`.
- [ ] Create the GitHub release (always use `--notes-file`, never inline `--notes`):
      `gh release create vX.Y.Z --title vX.Y.Z --notes-file <notes.md>`.
      **[PyPI — optional]** Once `publish.yml` exists, publishing a release triggers the
      upload to PyPI.

### 5. Post-release verification

- [ ] The git tag and GitHub release are visible.
- [ ] **[PyPI — optional]** The **publish** workflow succeeded (Actions tab) and PyPI shows
      the new version; `pip install "difflow==X.Y.Z"` works in a fresh venv.
- [ ] **Docs:** the GitHub Pages deploy workflow is currently **disabled**
      (`.github/workflows/deploy-book.yml.disabled`). To publish docs, rename it to
      `deploy-book.yml` and enable Pages; otherwise deploy docs manually.
- [ ] (Optional) announce the release; open a follow-up to bump to the next dev version.

---

## PyPI publishing setup (one-time) [optional]

difflow has **no `publish.yml` yet**. To publish to PyPI with **OIDC trusted publishing**
(no stored API tokens), do the following once.

### 1. Add the workflow

Create `.github/workflows/publish.yml` (adapted from discopt-doe):

```yaml
name: publish
on:
  release:
    types: [published]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv build
      - run: uvx twine check dist/*
      - uses: actions/upload-artifact@v4
        with: { name: dist, path: dist/ }
  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions: { id-token: write }
    steps:
      - uses: actions/download-artifact@v4
        with: { name: dist, path: dist/ }
      - uses: pypa/gh-action-pypi-publish@release/v1
```

### 2. Register the trusted publisher on PyPI

At https://pypi.org/manage/account/publishing/ add a **pending publisher** with exactly:

- **PyPI Project Name:** `difflow`
- **Owner:** `jkitchin`
- **Repository name:** `differentiable-flowsheets`  *(repo name ≠ dist name)*
- **Workflow name:** `publish.yml`  *(filename only)*
- **Environment name:** `pypi`

These must match `publish.yml` exactly or PyPI refuses the upload ("no corresponding
publisher").

### 3. Create the GitHub environment

Repo → **Settings → Environments → New environment** → name it `pypi`. Optionally add
yourself under **Required reviewers** so a publish pauses for one-click approval. No secrets
needed — OIDC handles auth.
