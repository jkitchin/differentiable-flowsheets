# Makefile for executing Jupyter notebooks in place

# All notebooks in this directory (sorted for consistent ordering)
NOTEBOOKS = $(sort $(wildcard *.ipynb))

# PDF files derived from notebooks
PDFS = $(NOTEBOOKS:.ipynb=.pdf)

# Default target: execute all notebooks
all: $(NOTEBOOKS)

# Pattern rule: execute each notebook in place
%.ipynb: FORCE
	JAX_PLATFORM_NAME=cpu jupyter nbconvert --to notebook --execute --inplace $@

# Export a single notebook to PDF (using webpdf for better compatibility)
%.pdf: %.ipynb
	JAX_PLATFORM_NAME=cpu jupyter nbconvert --to webpdf --execute $<

# Export all notebooks to PDFs
pdfs: $(PDFS)

# Combine all PDFs into one (requires pdfunite from poppler-utils)
combined.pdf: $(PDFS)
	pdfunite $(PDFS) $@

# Alias for combined PDF
book: combined.pdf

# Clean notebook outputs
clean:
	for nb in $(NOTEBOOKS); do \
		jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $$nb; \
	done

# Clean PDF files
clean-pdfs:
	rm -f $(PDFS) combined.pdf

# Clean everything
clean-all: clean clean-pdfs

# Force target to always run
FORCE:

.PHONY: all clean clean-pdfs clean-all pdfs book FORCE
