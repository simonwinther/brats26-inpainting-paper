# CATCH BraTS 2026 paper artifacts

This repository contains the manuscript, saved result tables, figure inputs,
generated figures, and the scripts used to audit the final 75-case confirmation
evaluation.

## Build the paper

Install a LaTeX distribution containing `latexmk`, BibTeX, and the packages loaded
by `paper.tex`, then run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex
```

The compiled manuscript is written to `build/paper.pdf`.

## Python dependencies

The commands below use `uv` with the pinned dependencies in `requirements.txt`;
no persistent virtual environment is required.

The confirmation audit and qualitative reconstruction figure additionally require
the sibling `brats_inpainting` checkout, its final result directories, and the
challenge data. The confirmation and qualitative scripts validate their inputs and
require `--overwrite` before replacing canonical outputs.

## Regenerate final results and figures

From this repository root, with the sibling checkout at `../brats_inpainting`:

```sh
uv run --with-requirements requirements.txt python scripts/generate_confirmation_results.py --overwrite
```

```sh
uv run --with-requirements requirements.txt python scripts/plot_confirmation_effects.py --overwrite
```

```sh
uv run --with-requirements requirements.txt python scripts/plot_training_dynamics.py --input data/wandb_final_checkpoint_selection.csv --output-stem figures/checkpoint_selection
```

```sh
uv run --with-requirements requirements.txt python scripts/plot_qualitative_reconstructions.py --metrics ../brats_inpainting/results/paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5/ensemble_metrics/mean_n5/metrics.csv --pred-dir ../brats_inpainting/results/paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5/brats_inpainting/mean_n5 --data-dir ../brats_inpainting/data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training --output-stem figures/qualitative_reconstructions_column --zoom-layout column --overwrite
```

```sh
uv run --with-requirements requirements.txt python scripts/plot_mask_augmentation.py --layout audit --plot-only --figure-stem figures/mask_augmentation_audit --overwrite
```

The architecture source is a standalone LaTeX document:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd figures/catch_architecture.tex
```

The generated JSON manifests record input/output hashes, metric protocol, selected
cases, and source-run metadata. Canonical numerical results, patient-overlap audits,
and the complete prediction-hash inventory are stored under
`data/confirmation75/`; the manuscript table is `tables/confirmation75.tex`.
