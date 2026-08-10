# CATCH BraTS 2026 paper artifacts

This directory contains the authoritative manuscript (`main.tex`), generated
tables and figures, source metrics, and provenance manifests for the revision.
The files under `sections/` are legacy, uncompiled drafting fragments; edit
`main.tex`, not those fragments.

## Evaluation hierarchy

The paper uses the following fixed hierarchy:

1. Within each training arm, five prespecified cases from the 25-case development
   cohort select its EMA checkpoint, and all 25 select its
   trajectory-aggregation policy.
2. The resulting three frozen, arm-specific pipelines are compared once on the
   separate 75-case internal model-selection cohort. This comparison selects the
   weighted pipeline; it is not an independent post-selection confirmation test.
3. Only weighted-mixture mean-N=5 was submitted for blind organizer evaluation
   on 219 cases. Fixed and random have no official 219-case performance scores.

The immutable split filename, source run names, and some legacy artifact paths
contain `confirmation` or `confirm75`. Those strings are historical identifiers,
not the inferential role of the 75 cases. Newly generated metadata records the
cohort as internal model selection.

## Build the paper

Install a LaTeX distribution containing `latexmk`, BibTeX, and the packages
loaded by `main.tex`, then run:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The compiled manuscript is `main.pdf`.

## Development trajectory-policy selection

The tracked `data/ensemble_selection_per_case.csv` snapshot contains the exact
25-case SSIM, PSNR, and MSE inputs used to rank the five trajectory-aggregation
candidates for each training arm. Recompute and verify the reported ranks using
only this repository and the Python standard library:

```sh
python scripts/verify_policy_selection.py
```

The verifier checks the source and summary hashes, cohort parity, and all joint
ranks. Original run, sampling-metadata, and terminal-checkpoint hashes are in
`data/ensemble_selection_summary_metadata.json`.

## Python dependencies

The commands below use `uv` with the pinned packages in `requirements.txt`.
They additionally require the sibling `brats_inpainting` checkout, its saved
result directories, and the released BraTS training data. Generators refuse to
replace canonical outputs unless a script explicitly supports `--overwrite`.

## Canonical internal model-selection results

Generate the primary 75-case artifacts and the post-hoc fixed-mask mean-N=5
sensitivity into distinct directories:

```sh
uv run --with-requirements requirements.txt python scripts/generate_confirmation_results.py --overwrite
```

```sh
uv run --with-requirements requirements.txt python scripts/generate_confirmation_results.py --fixed-n5-sensitivity --overwrite
```

The canonical outputs are `data/model_selection75/`,
`tables/model_selection75.tex`,
`data/model_selection75_fixed_n5_sensitivity/`, and
`tables/model_selection75_fixed_n5_sensitivity.tex`. The first analysis uses
fixed N=1, random mean-N=5, and weighted mean-N=5 exactly as selected on the 25
development cases. The all-N=5 analysis is explicitly post hoc and does not
replace that frozen comparison.

The older tracked `data/confirmation75/` and `tables/confirmation75.tex` paths
are preserved as immutable historical snapshots. Their numerical inputs remain
valid, but their former “confirmation” wording is superseded by the canonical
model-selection artifacts above.

Regenerate the paired-effects figure from the canonical primary data:

```sh
uv run --with-requirements requirements.txt python scripts/plot_confirmation_effects.py --overwrite
```

## Canonical void-size sensitivity

The post-hoc exploratory size analysis defines size from the exact mask scored
by the official evaluator: `{case}-mask-healthy.nii.gz`. It does not use the
complete `{case}-mask.nii.gz` inpainting region.

From `../brats_inpainting`, create the healthy-mask sizes:

```sh
uv run --with-requirements ../brats26-inpainting-paper/requirements.txt python scripts/analysis/compute_void_sizes.py --data_dir data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training --subset_file splits/checkpoint_confirmation_from_holdout_seed2026_n75.txt --out_csv results/void_size_stratification/model_selection75_healthy_mask_v1/healthy_mask_void_sizes.csv --overwrite
```

Then, from this paper directory, join those sizes to the already-saved official
per-case metric outputs:

```sh
uv run --with-requirements requirements.txt python scripts/summarize_void_size_sensitivity.py --overwrite
```

The only canonical paper result is
`data/model_selection75_void_size_sensitivity/`. Its manifest hashes the size
generator, source per-case metrics, checkpoints, and outputs, and identifies the
analysis as post hoc. Predictions are neither regenerated nor rescored.

`void_sizes_confirm75.csv` is an obsolete, noncanonical artifact: despite its
ambiguous name, it counted the complete `-mask.nii.gz` region. The size generator
now rejects that basename, the obsolete CSV is not copied into the paper, and the
canonical manifest records its SHA-256 and exclusion reason. Do not use it.
The Hendrix copy has been retired as
`void_sizes_confirm75.OBSOLETE_COMPLETE_MASK_DO_NOT_USE.csv`.

## Other revision artifacts

Steady-state runtime is regenerated from the original per-shard prediction
timestamps:

```sh
uv run --with-requirements requirements.txt python scripts/summarize_inference_timing.py --output data/inference_timing_model_selection75.json --overwrite
```

The blind weighted-pipeline performance result is stored in
`data/official_validation219_weighted.json`. It is an organizer-scored
performance result. In contrast, the reports under `../brats2026-submission/`
are submission-package integrity checks (case count, filenames, geometry,
outside-mask preservation, and ZIP integrity); they contain no hidden-ground-
truth performance scores. Package validation of fixed or random must never be
reported as official evaluation. The reusable
`../brats2026-submission/evaluate_official_validation.py` is a guarded local
reproduction harness, but an exact 219-case reproduction was not completed:
the official workflow's separate healthy-scoring-mask archive (`syn51685080`)
is organizer-private. The script now requires that archive explicitly and
rejects substitution of the public complete masks. The organizer-produced
weighted score is therefore the sole 219-case performance result.

Other figure regeneration commands are:

```sh
uv run --with-requirements requirements.txt python scripts/plot_training_dynamics.py --input data/wandb_final_checkpoint_selection.csv --output-stem figures/checkpoint_selection
```

```sh
uv run --with-requirements requirements.txt python scripts/plot_qualitative_reconstructions.py --metrics ../brats_inpainting/results/paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5/ensemble_metrics/mean_n5/metrics.csv --pred-dir ../brats_inpainting/results/paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5/brats_inpainting/mean_n5 --data-dir ../brats_inpainting/data/ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training --output-stem figures/qualitative_reconstructions_column --zoom-layout column --overwrite
```

The canonical qualitative outputs use the
`qualitative_reconstructions_column` stem. The older unsuffixed
`qualitative_reconstructions.*` files are legacy, noncanonical artifacts and are
explicitly marked as superseded in their manifest.

```sh
uv run --with-requirements requirements.txt python scripts/plot_mask_augmentation.py --layout audit --plot-only --figure-stem figures/mask_augmentation_audit --overwrite
```

The architecture source is a standalone LaTeX document:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd figures/catch_architecture.tex
```
