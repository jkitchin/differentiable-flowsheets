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

> **Release target.** A release publishes to three places from one GitHub release:
> the **git tag**, **PyPI** (via `.github/workflows/publish.yml`, OIDC trusted
> publishing), and **Zenodo** (a DOI, via the GitHub-Zenodo integration).
> Both PyPI and Zenodo need a **one-time setup** before the first release works:
> see [PyPI publishing setup](#pypi-publishing-setup-one-time) and
> [Zenodo DOI setup](#zenodo-doi-setup-one-time).

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

> **⚠️ Version is duplicated in 5 places — not a single source of truth.** Unlike a
> single-`version` project, difflow bundles plugins that hard-code their own
> `__version__`, and `CITATION.cff` carries its own `version`. You **must** bump all of
> them together (see step 2). `difflow` itself reads its version from installed metadata
> (`importlib.metadata.version("difflow")` in `src/difflow/__init__.py`), so it needs no
> manual edit. `.zenodo.json` deliberately has **no** `version` field: Zenodo takes the
> version from the git tag, so there is nothing to sync there.

### 0. Prerequisite gate — dependencies must be PyPI-resolvable ⛔

PyPI rejects any distribution that depends on a git/URL source, so every runtime
dependency must resolve from a package index.

- [ ] All runtime dependencies in `pyproject.toml` `[project].dependencies` are published
      on PyPI at the required versions.
- [ ] No `[tool.uv.sources]` git/URL overrides for runtime deps remain in `pyproject.toml`.
- [ ] `grep -rn "git+" pyproject.toml uv.lock` returns nothing for runtime dependencies.
- [ ] The distribution name `difflow` is available / owned by you on PyPI (verified
      **unclaimed** as of 2026-08-10 — the first upload registers it).

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

- [ ] `pyproject.toml`: `version = "X.Y.Z"`
- [ ] `src/difflow_gas/__init__.py` line 48: `__version__ = "X.Y.Z"`
- [ ] `src/difflow_ree/__init__.py` line 36: `__version__ = "X.Y.Z"`
- [ ] `src/difflow_cc/__init__.py` line 42: `__version__ = "X.Y.Z"`
- [ ] `CITATION.cff`: `version: X.Y.Z` **and** `date-released: "YYYY-MM-DD"` (the release
      date). This is what GitHub's "Cite this repository" button renders.
- [ ] Sanity-check they all match:
      `grep -rn "0\.0\.0\|version" pyproject.toml src/difflow_*/__init__.py | grep -i version`
      (no stray old versions remain).
- [ ] Update `CHANGELOG.md` (create it if adopting): move `[Unreleased]` entries under the
      new version + today's date.
- [ ] Re-check `README.md` install instructions and extras (`[all]`, `[dev]`, `[examples]`,
      `[visualization]`, `cuda11`/`cuda12`).
- [ ] Re-check `pyproject.toml` metadata: `description`, `requires-python`, `readme`,
      `license` (`MIT`, PEP 639 style), `authors`, `keywords`, `classifiers`, and
      `[project.urls]`. These all surface on the PyPI project page.
- [ ] If authorship or the abstract changed, update `CITATION.cff` and `.zenodo.json`
      together so the DOI record and the citation metadata agree.

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
      Publishing the release triggers **both** downstream steps: `publish.yml` uploads to
      PyPI, and Zenodo archives the tarball and mints a DOI.

> ⚠️ Zenodo only archives releases created **after** the repo was switched on in the
> Zenodo GitHub settings. If you tag first and enable Zenodo second, that release gets no
> DOI and you have to cut another one.

### 5. Post-release verification

- [ ] The git tag and GitHub release are visible.
- [ ] The **Publish** workflow succeeded (Actions tab) and PyPI shows the new version;
      `pip install "difflow==X.Y.Z"` works in a fresh venv.
- [ ] **Zenodo:** the new version appears at the concept DOI and the record metadata
      (title, author, ORCID, license) looks right. Fix anything wrong by editing the
      record on Zenodo directly, then correct `.zenodo.json` so the next release is right.
- [ ] On the **first** release only: add the concept-DOI badge to `README.md` (see
      [Zenodo DOI setup](#zenodo-doi-setup-one-time)).
- [ ] **Docs:** the Pages deploy (`.github/workflows/deploy-book.yml`) succeeded and
      <https://kitchingroup.cheme.cmu.edu/differentiable-flowsheets/> shows the new build.
- [ ] (Optional) announce the release; open a follow-up to bump to the next dev version.

---

## PyPI publishing setup (one-time)

`.github/workflows/publish.yml` **already exists**. It builds an sdist + wheel on every
published GitHub release, runs `twine check`, verifies all five packages made it into the
wheel, and uploads via **OIDC trusted publishing** (no API token is stored in this repo).

Two things still have to be done by hand, in a browser, before the first release.

### 1. Register the trusted publisher on PyPI

At https://pypi.org/manage/account/publishing/ add a **pending publisher** with exactly:

- **PyPI Project Name:** `difflow`
- **Owner:** `jkitchin`
- **Repository name:** `differentiable-flowsheets`  *(repo name ≠ dist name)*
- **Workflow name:** `publish.yml`  *(filename only)*
- **Environment name:** `pypi`

These must match `publish.yml` exactly or PyPI refuses the upload ("no corresponding
publisher").

### 2. Create the GitHub environment

Repo → **Settings → Environments → New environment** → name it `pypi`. Optionally add
yourself under **Required reviewers** so a publish pauses for one-click approval. No secrets
are needed, since OIDC handles auth.

---

## Zenodo DOI setup (one-time)

Zenodo mints a DOI for each GitHub release, plus a **concept DOI** that always resolves to
the newest version. Cite the concept DOI in papers; cite a version DOI to pin an exact
release.

Metadata for the record comes from **`.zenodo.json`** in the repo root (already written:
title, abstract, creator + ORCID, `"license": "mit"`, keywords). Zenodo prefers
`.zenodo.json` over `CITATION.cff` when both are present, so `.zenodo.json` is the file to
edit if a record looks wrong. Note the Zenodo license id is lowercase (`mit`), not the
SPDX-cased `MIT` used in `pyproject.toml` and `CITATION.cff`.

### 1. Switch the repo on in Zenodo

1. Sign in at <https://zenodo.org> with **Log in with GitHub** and authorize the app.
2. Go to <https://zenodo.org/account/settings/github/>.
3. Find **jkitchin/differentiable-flowsheets** and flip its toggle **On**. Hit **Sync now**
   if it is not listed (the repo must be public, which it is).

This installs a release webhook. It is not retroactive: only releases published *after*
the toggle get archived.

### 2. Cut a release

Follow the release checklist above. When the release publishes, Zenodo receives the
webhook, downloads the source tarball, and mints the DOI within a minute or two.

### 3. Add the concept-DOI badge (first release only)

On the Zenodo record, take the **"Cite all versions"** DOI (the concept DOI, ending in a
lower number than the version DOI) and add its badge under the other badges in `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Zenodo's GitHub settings page also offers a ready-made Markdown badge snippet, but that one
uses the *repo* badge URL, which resolves to the latest version rather than the concept
record. Prefer the explicit concept DOI above.

### 4. Backfill the DOI into the citation metadata

Once the concept DOI exists, add it to `CITATION.cff` so the "Cite this repository" output
carries it:

```yaml
identifiers:
  - type: doi
    value: 10.5281/zenodo.XXXXXXX
    description: Concept DOI for all versions of difflow
```

Also add a `doi` entry to `papers/joss/paper.md` if the JOSS submission proceeds.
