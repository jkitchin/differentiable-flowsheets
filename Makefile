# Makefile for differentiable-flowsheets

NOTEBOOKS := $(wildcard examples/*.ipynb)

.PHONY: all notebooks clean test book book-clean

all: notebooks

# Execute all notebooks in place
notebooks: $(NOTEBOOKS)
	@for nb in $(NOTEBOOKS); do \
		echo "Executing $$nb..."; \
		jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
	@echo "All notebooks executed successfully."

# Execute a single notebook (usage: make run NB=examples/01_cstr_flash_recycle.ipynb)
run:
	jupyter nbconvert --to notebook --execute --inplace "$(NB)"

# Run tests
test:
	pytest tests/ -v

# Clean generated files
clean:
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
	python -m http.server 8000 --directory _build/html
