# README FOR OVERLEAF

## Main file
`main.tex`

## Compiler
pdfLaTeX

## Bibliography
BibTeX (`elsarticle-num-names.bst`)

## Compilation sequence
1. pdflatex main.tex
2. bibtex main
3. pdflatex main.tex
4. pdflatex main.tex

## Required local files (included in ZIP)
- `elsarticle.cls` (Elsevier article class)
- `elsarticle-num-names.bst` (bibliography style)
- `references.bib`
- `sections/` (modular section files)
- `tables/` (LaTeX table files)
- `figures/` (PDF figures)
- `supplementary/` (supplementary material)

## Known non-blocking warnings
- `Mismatched LaTeX support files detected` — version mismatch between local elsarticle.cls and TeX Live; does NOT affect output. Overleaf should compile cleanly with its own version.
- `Token not allowed in a PDF string` (hyperref math in captions) — cosmetic only, does not affect PDF.

## Journal
Environmental Modelling & Software (Elsevier)

## Page count (local compilation)
17 pages
