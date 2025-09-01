# Makefile to build Metanion Field Theory PDF and regenerate diagram

LATEXMK ?= latexmk
TEXFLAGS ?= -xelatex -interaction=nonstopmode -halt-on-error -shell-escape -output-directory=.out
PYTHON ?= python3

TEX := metanion_field_theory.tex
PDF := metanion_field_theory.pdf
DIAGRAM_PY := metanion_field_diagram.py
DIAGRAM_PDF := metanion_field_diagram.pdf
DIAGRAM_PNG := metanion_field_diagram.png

OUT_DIR := .out

.PHONY: all pdf diagram open clean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX) $(DIAGRAM_PDF)
	@mkdir -p $(OUT_DIR)
	$(LATEXMK) $(TEXFLAGS) $(TEX)
	@cp $(OUT_DIR)/$(PDF) .
	@echo "PDF copied to repo root: $(PDF)"

# Regenerate the diagram when the script changes
$(DIAGRAM_PDF): $(DIAGRAM_PY)
	-$(PYTHON) $(DIAGRAM_PY) || ( [ -f $(DIAGRAM_PNG) ] && sips -s format pdf $(DIAGRAM_PNG) --out $(DIAGRAM_PDF) )
	@[ -f $(DIAGRAM_PDF) ] || ( echo "Error: Failed to generate $(DIAGRAM_PDF). Install matplotlib or ensure $(DIAGRAM_PNG) exists." && exit 1 )

# Convenience alias
diagram: $(DIAGRAM_PDF)

open: pdf
	open $(PDF)

clean:
	-$(LATEXMK) -c -output-directory=$(OUT_DIR)
	-rm -rf $(OUT_DIR)
	-rm -f $(PDF)
