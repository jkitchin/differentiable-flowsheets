# Makefile for differentiable-flowsheets

# Use uv run to ensure virtual environment is active
# Unset VIRTUAL_ENV to prevent conflicts with other activated environments
UV_RUN := VIRTUAL_ENV= uv run

# The dev tooling (pytest, jupyter-book) lives in the `dev` *extra*, which plain
# `uv run` does not install -- it would silently fall through to whatever is on
# PATH. That matters for jupyter-book: the docs use the classic (Sphinx) config
# in _config.yml / _toc.yml, and a stray jupyter-book 2.x on PATH cannot read it.
# `--extra dev` pins both to the versions in uv.lock (jupyter-book <2.0).
UV_RUN_DEV := VIRTUAL_ENV= uv run --extra dev

# Notebook execution command - use python -m to ensure correct kernel
NBCONVERT := python -m jupyter nbconvert --to notebook --execute --inplace

# Source files that notebooks depend on
SRC_FILES := $(shell find src -name '*.py' 2>/dev/null)

NOTEBOOKS := $(wildcard examples/*.ipynb) $(wildcard examples/bio/*.ipynb) \
             $(wildcard examples/ree/*.ipynb) $(wildcard examples/cc/*.ipynb) \
             $(wildcard jax-tutorials/*.ipynb) $(wildcard about-flowsheets/*.ipynb)

# Stamp files to track notebook execution
STAMP_DIR := .notebook-stamps
STAMPS := $(patsubst %.ipynb,$(STAMP_DIR)/%.stamp,$(NOTEBOOKS))

# Plugin-specific notebook lists and stamps
NOTEBOOKS_BIO := $(wildcard examples/bio/*.ipynb)
NOTEBOOKS_REE := $(wildcard examples/ree/*.ipynb)
NOTEBOOKS_CC := $(wildcard examples/cc/*.ipynb)
STAMPS_BIO := $(patsubst %.ipynb,$(STAMP_DIR)/%.stamp,$(NOTEBOOKS_BIO))
STAMPS_REE := $(patsubst %.ipynb,$(STAMP_DIR)/%.stamp,$(NOTEBOOKS_REE))
STAMPS_CC := $(patsubst %.ipynb,$(STAMP_DIR)/%.stamp,$(NOTEBOOKS_CC))

# Force CPU on macOS (no GPU support)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    JAX_ENV := JAX_PLATFORM_NAME=cpu
endif

.PHONY: all notebooks notebooks-force notebooks-bio notebooks-ree notebooks-cc \
        notebooks-bio-force notebooks-ree-force notebooks-cc-force \
        clean test test-release test-slow test-all test-durations book book-clean sync \
        gui gui-build gui-test

all: notebooks

# Create stamp directory
$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)/examples/bio $(STAMP_DIR)/examples/ree $(STAMP_DIR)/examples/cc \
		$(STAMP_DIR)/jax-tutorials $(STAMP_DIR)/about-flowsheets

# Pattern rule: run notebook only if it or source files changed
$(STAMP_DIR)/%.stamp: %.ipynb $(SRC_FILES) | $(STAMP_DIR)
	@mkdir -p $(dir $@)
	@echo "Executing $<..."
	@$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$<" || exit 1
	@touch $@

# Execute notebooks only if out of date
notebooks: $(STAMPS)
	@echo "All notebooks up to date."

# Force execute all notebooks
notebooks-force:
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$$nb" || exit 1; \
	done
	@mkdir -p $(STAMP_DIR)/examples/bio $(STAMP_DIR)/examples/ree $(STAMP_DIR)/examples/cc \
		$(STAMP_DIR)/jax-tutorials $(STAMP_DIR)/about-flowsheets
	@for nb in $(NOTEBOOKS); do \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "All notebooks executed successfully."

# Execute bio plugin notebooks (only if out of date)
notebooks-bio: $(STAMPS_BIO)
	@echo "Bio notebooks up to date."

# Execute ree plugin notebooks (only if out of date)
notebooks-ree: $(STAMPS_REE)
	@echo "REE notebooks up to date."

# Execute cc plugin notebooks (only if out of date)
notebooks-cc: $(STAMPS_CC)
	@echo "Carbon capture notebooks up to date."

# Force execute plugin notebooks
notebooks-bio-force:
	@for nb in $(NOTEBOOKS_BIO); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/bio; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "Bio notebooks executed successfully."

notebooks-ree-force:
	@for nb in $(NOTEBOOKS_REE); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/ree; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "REE notebooks executed successfully."

notebooks-cc-force:
	@for nb in $(NOTEBOOKS_CC); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/cc; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "Carbon capture notebooks executed successfully."

# Execute a single notebook (usage: make run NB=examples/01_cstr_flash_recycle.ipynb)
run:
	$(JAX_ENV) $(UV_RUN) $(NBCONVERT) "$(NB)"

# The suite is compile-bound and parallelises across processes: `-n auto`
# takes a full local run from ~40 to ~15 minutes on four cores. `--dist
# loadfile` is not optional -- each worker has its own JAX compilation cache,
# so splitting a module across workers recompiles the same graphs in each of
# them and gives most of the win back. Add `-n0` to any of these to get a
# single process back for --pdb or readable output.
PYTEST := pytest -n auto --dist loadfile

# What you run while working, and what every commit is checked against:
# everything except the `release` tier, which re-derives physics and numerics
# rather than checking the code (see the marker notes in pyproject.toml).
# `slow` comes out too, so a local run stays short -- CI keeps it.
test:
	$(UV_RUN_DEV) $(PYTEST) tests/ -v -m "not release and not slow"

# The release tier on its own: Monte Carlo refits, finite-difference
# agreement, energy balances, published benchmarks.
test-release:
	$(UV_RUN_DEV) $(PYTEST) tests/ -v -m release

# Only the compile-bound tests
test-slow:
	$(UV_RUN_DEV) $(PYTEST) tests/ -v -m slow

# Everything. What a release has to pass, and what release.yml runs.
test-all:
	$(UV_RUN_DEV) $(PYTEST) tests/ -v

# Re-measure what CI shards on. `.test_durations` is what balances the three
# CI jobs against each other; a test missing from it is estimated at the
# average, so the balance degrades slowly as tests are added rather than
# breaking. Regenerate when the shards have drifted noticeably apart (the job
# names carry their numbers) -- serially and with nothing else running on the
# machine, or the numbers it records are of a loaded box, which takes ~40 min.
test-durations:
	$(UV_RUN_DEV) pytest tests/ --store-durations

# Clean generated files
clean:
	rm -rf $(STAMP_DIR)
	rm -f examples/flowsheet.html
	rm -f test_flowsheet.html
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Build jupyter book
book:
	$(UV_RUN_DEV) jupyter-book build .

# Clean jupyter book build
book-clean:
	$(UV_RUN_DEV) jupyter-book clean .

# Build and serve jupyter book locally
book-serve: book
	@echo "Opening browser and starting server at http://localhost:8000"
	@open http://localhost:8000 || xdg-open http://localhost:8000 || echo "Open http://localhost:8000 in your browser"
	$(UV_RUN) python -m http.server 8000 --directory _build/html

# Sync uv virtual environment (install dependencies)
sync:
	uv sync

# ---------------------------------------------------------------------------
# The editor front end.
#
# Two artefacts, one rule: the JS bundle, and the assistant's retrieval
# index over `docs/` (`static/docs-index.json`, built by a stdlib-only
# script so CI can run it with nothing installed). Both are committed
# and both are checked for drift, for the same reason.
#
# `src/difflow/gui/static/` is BUILT OUTPUT and it is committed: `pip install
# difflow` must never need node. Only changing the UI does, and this is how.
# CI reruns gui-build and fails if the result differs from what is committed,
# so the bundle cannot drift away from its source.
GUI_FRONTEND := src/difflow/gui/frontend

gui-build:
	cd $(GUI_FRONTEND) && npm ci && npm run build
	python3 src/difflow/gui/docs_index.py

# The pure model functions, under bare node. Same files tests/test_gui.py runs.
gui-test:
	cd $(GUI_FRONTEND) && npm test

# Serve a flowsheet: `make gui FLOWSHEET=examples/whatever.json` (or bare, for
# an empty one).
gui:
	$(UV_RUN) difflow gui $(FLOWSHEET)
