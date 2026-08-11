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
        clean test book book-clean sync

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

# Run tests
test:
	$(UV_RUN_DEV) pytest tests/ -v

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
