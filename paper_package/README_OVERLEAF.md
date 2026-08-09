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
- `figures/` (PDF figures used by `main.tex`)
- `supplementary/` (supplementary material)

## Known non-blocking warnings
- Hyperref may report cosmetic math-token warnings for section metadata; these do not affect the PDF.

## Journal
Environmental Modelling & Software (Elsevier)

## Page count (local compilation)
10 pages, including supplementary material.

Before submission, extract and compile the archive from a clean directory.
The local revision is not claimed to be publicly available until the
repository is synchronised with the submission commit.
