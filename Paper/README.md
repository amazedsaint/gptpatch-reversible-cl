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

## Build (Docker, no local TeX)

If you don't have LaTeX installed locally, build in a TeX Live container:

```bash
docker run --rm \
  -v "$PWD/Paper":/data -w /data \
  ghcr.io/xu-cheng/texlive-full:latest \
  latexmk -pdf -interaction=nonstopmode paper.tex
```

The PDF will be written to `Paper/paper.pdf`.
