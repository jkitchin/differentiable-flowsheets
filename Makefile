# Makefile for differentiable-flowsheets

# Source files that notebooks depend on
SRC_FILES := $(shell find src -name '*.py' 2>/dev/null)

NOTEBOOKS := $(wildcard examples/*.ipynb) $(wildcard examples/bio/*.ipynb) \
             $(wildcard jax-tutorials/*.ipynb) $(wildcard about-flowsheets/*.ipynb)

# Stamp files to track notebook execution
STAMP_DIR := .notebook-stamps
STAMPS := $(patsubst %.ipynb,$(STAMP_DIR)/%.stamp,$(NOTEBOOKS))

# Force CPU on macOS (no GPU support)
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    JAX_ENV := JAX_PLATFORM_NAME=cpu
endif

.PHONY: all notebooks notebooks-force clean test book book-clean

all: notebooks

# Create stamp directory
$(STAMP_DIR):
	@mkdir -p $(STAMP_DIR)/examples/bio $(STAMP_DIR)/jax-tutorials $(STAMP_DIR)/about-flowsheets

# Pattern rule: run notebook only if it or source files changed
$(STAMP_DIR)/%.stamp: %.ipynb $(SRC_FILES) | $(STAMP_DIR)
	@mkdir -p $(dir $@)
	@echo "Executing $<..."
	@$(JAX_ENV) jupyter nbconvert --to notebook --execute --inplace "$<" || exit 1
	@touch $@

# Execute notebooks only if out of date
notebooks: $(STAMPS)
	@echo "All notebooks up to date."

# Force execute all notebooks
notebooks-force:
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		$(JAX_ENV) jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
	@mkdir -p $(STAMP_DIR)/examples/bio $(STAMP_DIR)/jax-tutorials $(STAMP_DIR)/about-flowsheets
	@for nb in $(NOTEBOOKS); do \
		touch "$(STAMP_DIR)/$${nb%.ipynb}.stamp"; \
	done
	@echo "All notebooks executed successfully."

# Execute a single notebook (usage: make run NB=examples/01_cstr_flash_recycle.ipynb)
run:
	$(JAX_ENV) jupyter nbconvert --to notebook --execute --inplace "$(NB)"

# Run tests
test:
	pytest tests/ -v

# Clean generated files
clean:
	rm -rf $(STAMP_DIR)
	rm -f examples/flowsheet.html
	rm -f test_flowsheet.html
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

# Build jupyter book
book:
	jupyter-book build .

# Clean jupyter book build
book-clean:
	jupyter-book clean .

# Build and serve jupyter book locally
book-serve: book
	@echo "Opening browser and starting server at http://localhost:8000"
	@open http://localhost:8000 || xdg-open http://localhost:8000 || echo "Open http://localhost:8000 in your browser"
	python -m http.server 8000 --directory _build/html
