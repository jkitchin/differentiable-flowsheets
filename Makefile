# Makefile for differentiable-flowsheets

# Use uv run to ensure virtual environment is active
# Unset VIRTUAL_ENV to prevent conflicts with other activated environments
UV_RUN := VIRTUAL_ENV= uv run

# The dev tooling (pytest, jupyter-book) lives in the `dev` *extra*, which plain
# `uv run` does not install -- it would silently fall through to whatever is on
# PATH. That matters for jupyter-book: the docs use the classic (Sphinx) config
# in _config.yml / _toc.yml, and a stray jupyter-book 2.x on PATH cannot read it.
# `--extra dev` pins both to jupyter-book <2.0. The other three extras are the
# ones CI tests with (.github/workflows/test.yml): without `mcp`, `import mcp`
# finds the repo's top-level mcp/ config directory as a namespace package, and
# the MCP server tests fail with "cannot import name 'Client' from 'mcp'".
UV_RUN_DEV := VIRTUAL_ENV= uv run --extra dev --extra gui --extra solvers --extra mcp

# The notebooks import from the optional extras (tabulate from `examples`, asdex
# from `solvers`, gaspype from `gaspype`), which plain `uv run` does not install,
# so they ran in an environment missing them and died with ModuleNotFoundError.
# pounce, which examples/27 solves with, is in no extra (the `solvers` extra
# leaves the back ends to the user), so it comes in with `--with`.
UV_RUN_NB := VIRTUAL_ENV= uv run --extra examples --extra solvers --extra gaspype \
	--with "pounce-solver[jax]"

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
        clean test test-durations book book-clean sync \
        docs-index convergence ask-check

all: notebooks

# Create stamp directory
$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)/examples/bio $(STAMP_DIR)/examples/ree $(STAMP_DIR)/examples/cc \
		$(STAMP_DIR)/jax-tutorials $(STAMP_DIR)/about-flowsheets

# Pattern rule: run notebook only if it or source files changed
$(STAMP_DIR)/%.stamp: %.ipynb $(SRC_FILES) | $(STAMP_DIR)
	@mkdir -p $(dir $@)
	@echo "Executing $<..."
	@$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$<" || exit 1
	@touch $@

# Execute notebooks only if out of date
notebooks: $(STAMPS)
	@echo "All notebooks up to date."

# Force execute all notebooks
notebooks-force:
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$$nb" || exit 1; \
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
		$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/bio; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "Bio notebooks executed successfully."

notebooks-ree-force:
	@for nb in $(NOTEBOOKS_REE); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/ree; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "REE notebooks executed successfully."

notebooks-cc-force:
	@for nb in $(NOTEBOOKS_CC); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$$nb" || exit 1; \
		mkdir -p $(STAMP_DIR)/examples/cc; \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "Carbon capture notebooks executed successfully."

# Execute a single notebook (usage: make run NB=examples/01_cstr_flash_recycle.ipynb)
run:
	$(JAX_ENV) $(UV_RUN_NB) $(NBCONVERT) "$(NB)"

# The suite is compile-bound and parallelises across processes. `--dist
# loadfile` keeps a module on one worker, so its compiled JAX graphs are
# reused instead of recompiled in each worker. Add `-n0` for a single process
# (--pdb, readable output).
PYTEST := pytest -n auto --dist loadfile

# The whole suite: what you run while working, and what CI runs (in two
# shards). It checks the code, not the physics; see the test policy in
# CLAUDE.md.
test:
	$(UV_RUN_DEV) $(PYTEST) tests/ -v

# Re-measure what CI shards on (`.test_durations`, read by pytest-split). A
# test missing from it is estimated at the average, so the balance degrades
# slowly as tests are added. CI runners are slower than a laptop, unevenly,
# but only the relative cost matters for the split.
test-durations:
	$(UV_RUN_DEV) pytest tests/ --store-durations

# Measure how often the recycle solver converges, over the hard-flowsheet
# corpus in difflow.convergence. A few minutes, mostly JAX compilation. The
# number is a measurement and is meant to move -- rerun it after anything
# that touches initialization, acceleration or the tear solve.
convergence:
	$(UV_RUN_DEV) python -m difflow.convergence

# Clean generated files
clean:
	rm -rf $(STAMP_DIR)
	rm -f examples/flowsheet.html
	rm -f test_flowsheet.html
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Build jupyter book. The build also writes _build/html/ask-index.json, the
# corpus of the "Ask" docs assistant (_ext/ask_index.py, docs/ask.md).
book:
	$(UV_RUN_DEV) jupyter-book build .

# Run the shipped ask.js against a labelled question set over the built book
# (needs node). The deploy workflow runs the same check before publishing.
ask-check: book
	node tests/docs_ask/ask_retrieval.mjs _build/html/ask-index.json

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
# The documentation index.
#
# `src/difflow/workspace/data/docs-index.json` is the retrieval index over
# `docs/` that `search_docs` (and the book app's assistant) reads. It is
# built by a stdlib-only script and committed, because `docs/` is not in the
# wheel; rebuild it whenever the prose changes. (The browser editor that used
# to be built here is the app that comes with the difflow book.)
docs-index:
	python3 src/difflow/workspace/docs_index.py
