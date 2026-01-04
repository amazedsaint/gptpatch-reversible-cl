# Paper

This folder contains a publish-ready LaTeX manuscript describing the reversible continual-learning patch approach implemented in this repo.

## Files

- `paper.tex`: main manuscript
- `references.bib`: BibTeX bibliography

## Build

From the repo root:

```bash
cd Paper
latexmk -pdf -interaction=nonstopmode paper.tex
```

Or with plain LaTeX:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

