#!/usr/bin/env python3
"""Create the canonical post-hoc healthy-mask void-size sensitivity artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parent
BRATS_ROOT = REPOSITORY_ROOT / "brats_inpainting"
CASE_FILE = (
    BRATS_ROOT
    / "splits"
    / "checkpoint_confirmation_from_holdout_seed2026_n75.txt"
)
SIZE_CSV = (
    BRATS_ROOT
    / "results"
    / "void_size_stratification"
    / "model_selection75_healthy_mask_v1"
    / "healthy_mask_void_sizes.csv"
)
RANDOM_RUN = (
    BRATS_ROOT
    / "results"
    / "paper300k-concat-random-s0-maxnorm-confirm75-ckpt290000-mean-n5"
    / "ensemble_metrics"
    / "mean_n5"
)
WEIGHTED_RUN = (
    BRATS_ROOT
    / "results"
    / "paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5"
    / "ensemble_metrics"
    / "mean_n5"
)
OUTPUT_DIR = PAPER_ROOT / "data" / "model_selection75_void_size_sensitivity"
METRICS = ("ssim", "psnr", "mse")
SIZE_BINS = ("small", "medium", "large")
EXPECTED_CASE_FILE_SHA256 = (
    "9c86a971b9098a5c9cb53891ca4ec32cb662f93e3ed3a2e14cbd79e18ac58026"
)
EXPECTED_CHECKPOINTS = {
    "random": "69a4169fbead074819b3ed3b5392e08934dc1e7ac570933b4c825cb649d1ac27",
    "weighted": "46578c5e1e41ed66c6d6ba65818279d78c5f104e1ae8c59507511c5e16883e2a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, default=CASE_FILE)
    parser.add_argument("--size-csv", type=Path, default=SIZE_CSV)
    parser.add_argument("--random-run", type=Path, default=RANDOM_RUN)
    parser.add_argument("--weighted-run", type=Path, default=WEIGHTED_RUN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PAPER_ROOT))
    except ValueError:
        try:
            return str(Path("..") / resolved.relative_to(REPOSITORY_ROOT))
        except ValueError:
            return str(resolved)


def read_cases(path: Path) -> list[str]:
    cases = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(cases) != 75 or len(cases) != len(set(cases)):
        raise ValueError(f"Expected 75 unique cases in {path}, found {len(cases)}")
    return cases


def read_csv(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "case" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} has no case column")
        order: list[str] = []
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            case = row["case"].strip()
            if case in rows:
                raise ValueError(f"Duplicate case {case} in {path}")
            order.append(case)
            rows[case] = row
    return order, rows


def validate_summary(
    path: Path,
    *,
    arm: str,
    case_file_hash: str,
) -> dict[str, object]:
    summary = json.loads(path.read_text())
    expected = {
        "scored_case_count": 75,
        "subset_file_sha256": case_file_hash,
        "weights_type": "ema",
        "ema_rate": "0.9999",
        "checkpoint_sha256": EXPECTED_CHECKPOINTS[arm],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"{arm}: expected {key}={value!r}, found {summary.get(key)!r}"
            )
    if summary.get("ensemble") != {"method": "mean", "size": 5}:
        raise ValueError(f"{arm}: expected voxel-wise mean N=5")
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    case_file = args.case_file.expanduser().resolve()
    size_csv = args.size_csv.expanduser().resolve()
    run_dirs = {
        "random": args.random_run.expanduser().resolve(),
        "weighted": args.weighted_run.expanduser().resolve(),
    }
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite canonical analysis directory {output_dir}; "
            "pass --overwrite explicitly"
        )

    cases = read_cases(case_file)
    case_file_hash = sha256(case_file)
    if case_file_hash != EXPECTED_CASE_FILE_SHA256:
        raise ValueError("The 75-case model-selection manifest hash changed")

    size_order, size_rows = read_csv(size_csv)
    if size_order != cases:
        raise ValueError("Healthy-mask size rows do not match model-selection order")
    for case in cases:
        row = size_rows[case]
        if row.get("mask_kind") != "provided_scoring_mask_healthy":
            raise ValueError(f"{case}: size source is not the healthy scoring mask")
        if row.get("mask_filename") != f"{case}-mask-healthy.nii.gz":
            raise ValueError(f"{case}: unexpected mask filename")
        if row.get("size_bin") not in SIZE_BINS:
            raise ValueError(f"{case}: unexpected size bin {row.get('size_bin')!r}")
    bin_counts = {
        size_bin: sum(size_rows[case]["size_bin"] == size_bin for case in cases)
        for size_bin in SIZE_BINS
    }
    if bin_counts != {size_bin: 25 for size_bin in SIZE_BINS}:
        raise ValueError(f"Expected 25 cases per tertile, found {bin_counts}")

    metric_rows: dict[str, dict[str, dict[str, str]]] = {}
    source_summaries: dict[str, dict[str, object]] = {}
    source_paths: dict[str, dict[str, Path]] = {}
    for arm, run_dir in run_dirs.items():
        metrics_path = run_dir / "metrics.csv"
        summary_path = run_dir / "metrics.summary.json"
        order, rows = read_csv(metrics_path)
        if order != cases:
            raise ValueError(f"{arm}: metric rows do not match model-selection order")
        for case in cases:
            for metric in METRICS:
                value = float(rows[case][metric])
                if not np.isfinite(value):
                    raise ValueError(f"{arm}/{case}: non-finite {metric}")
        metric_rows[arm] = rows
        source_summaries[arm] = validate_summary(
            summary_path,
            arm=arm,
            case_file_hash=case_file_hash,
        )
        source_paths[arm] = {"metrics": metrics_path, "summary": summary_path}

    paired_rows: list[dict[str, object]] = []
    for case in cases:
        random_values = {
            metric: float(metric_rows["random"][case][metric])
            for metric in METRICS
        }
        weighted_values = {
            metric: float(metric_rows["weighted"][case][metric])
            for metric in METRICS
        }
        paired_rows.append(
            {
                "case": case,
                "size_bin": size_rows[case]["size_bin"],
                "healthy_mask_volume_voxels": int(
                    size_rows[case]["target_volume_voxels"]
                ),
                "random_ssim": random_values["ssim"],
                "weighted_ssim": weighted_values["ssim"],
                "delta_ssim_weighted_minus_random": (
                    weighted_values["ssim"] - random_values["ssim"]
                ),
                "random_psnr": random_values["psnr"],
                "weighted_psnr": weighted_values["psnr"],
                "delta_psnr_db_weighted_minus_random": (
                    weighted_values["psnr"] - random_values["psnr"]
                ),
                "random_mse": random_values["mse"],
                "weighted_mse": weighted_values["mse"],
                "delta_mse_weighted_minus_random": (
                    weighted_values["mse"] - random_values["mse"]
                ),
            }
        )

    tertile_rows: list[dict[str, object]] = []
    tertile_summary: dict[str, dict[str, float | int]] = {}
    for size_bin in SIZE_BINS:
        selected = [row for row in paired_rows if row["size_bin"] == size_bin]
        volumes = np.asarray(
            [row["healthy_mask_volume_voxels"] for row in selected],
            dtype=np.int64,
        )
        record: dict[str, object] = {
            "size_bin": size_bin,
            "case_count": len(selected),
            "healthy_mask_volume_min_voxels": int(volumes.min()),
            "healthy_mask_volume_max_voxels": int(volumes.max()),
        }
        for field in (
            "delta_ssim_weighted_minus_random",
            "delta_psnr_db_weighted_minus_random",
            "delta_mse_weighted_minus_random",
        ):
            values = np.asarray([row[field] for row in selected], dtype=np.float64)
            record[f"mean_{field}"] = float(values.mean())
        tertile_rows.append(record)
        tertile_summary[size_bin] = {
            key: value for key, value in record.items() if key != "size_bin"
        }

    volumes = np.asarray(
        [int(size_rows[case]["target_volume_voxels"]) for case in cases],
        dtype=np.float64,
    )
    thresholds = np.quantile(volumes, [1 / 3, 2 / 3])
    output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    outputs = {
        "healthy_mask_sizes": output_dir / "healthy_mask_void_sizes.csv",
        "paired_cases": output_dir / "paired_cases.csv",
        "paired_by_tertile": output_dir / "paired_by_tertile.csv",
        "random_metrics": output_dir / "random_metrics.csv",
        "random_summary": output_dir / "random_metrics.summary.json",
        "weighted_metrics": output_dir / "weighted_metrics.csv",
        "weighted_summary": output_dir / "weighted_metrics.summary.json",
        "summary": output_dir / "summary.json",
        "manifest": output_dir / "manifest.json",
    }
    shutil.copyfile(size_csv, outputs["healthy_mask_sizes"])
    write_csv(outputs["paired_cases"], paired_rows)
    write_csv(outputs["paired_by_tertile"], tertile_rows)
    for arm in ("random", "weighted"):
        shutil.copyfile(source_paths[arm]["metrics"], outputs[f"{arm}_metrics"])
        shutil.copyfile(source_paths[arm]["summary"], outputs[f"{arm}_summary"])

    summary = {
        "schema_version": 1,
        "analysis_role": "post_hoc_exploratory_sensitivity",
        "cohort_role": "internal_training_policy_selection",
        "case_count": len(cases),
        "mask_definition": (
            "Provided {case}-mask-healthy.nii.gz scoring mask; the complete "
            "{case}-mask.nii.gz inpainting mask is excluded."
        ),
        "binning": {
            "method": "tertiles of healthy scoring-mask voxel count",
            "small_upper_quantile_value": float(thresholds[0]),
            "medium_upper_quantile_value": float(thresholds[1]),
            "counts": bin_counts,
        },
        "metric_provenance": (
            "Joined to the already-saved per-case outputs of the official BraTS "
            "metric package; predictions were neither regenerated nor rescored."
        ),
        "tertiles": tertile_summary,
    }
    outputs["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    manifest = {
        "schema_version": 1,
        "canonical": True,
        "analysis_role": "post_hoc_exploratory_sensitivity",
        "cohort_role": "internal_training_policy_selection",
        "generator": portable_path(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "void_size_generator": portable_path(
            BRATS_ROOT / "scripts" / "analysis" / "compute_void_sizes.py"
        ),
        "void_size_generator_sha256": sha256(
            BRATS_ROOT / "scripts" / "analysis" / "compute_void_sizes.py"
        ),
        "case_manifest": {
            "path": portable_path(case_file),
            "sha256": case_file_hash,
            "case_count": len(cases),
            "legacy_filename_note": (
                "The immutable filename contains 'confirmation'; its inferential "
                "role is internal model selection."
            ),
        },
        "canonical_size_definition": {
            "mask_suffix": "-mask-healthy.nii.gz",
            "meaning": "provided healthy scoring mask used by the official evaluator",
            "complete_mask_suffix_excluded": "-mask.nii.gz",
        },
        "obsolete_artifact": {
            "legacy_basename": "void_sizes_confirm75.csv",
            "sha256": "0ece309703e725a9e7c4897440cd7b28bb70d28d934ba555fad367739672157e",
            "status": "obsolete_do_not_use",
            "tertile_assignments_differing_from_canonical": 40,
            "case_count": 75,
            "reason": (
                "It counted the complete -mask.nii.gz inpainting region rather "
                "than the -mask-healthy.nii.gz scoring mask. It is not copied into "
                "the canonical paper artifacts."
            ),
        },
        "legacy_healthy_mask_cross_check": {
            "legacy_basename": "void_sizes_confirm75_healthymask.csv",
            "sha256": "d83232ff30108efdfb05da33aa7a98c4894a79bd70b495e98f9eaf9e02ce0f54",
            "case_count": 75,
            "comparison": (
                "Canonical healthy-mask voxel counts and tertile assignments match "
                "for all 75 cases."
            ),
        },
        "source_metrics": {
            arm: {
                "metrics_path": portable_path(source_paths[arm]["metrics"]),
                "metrics_sha256": sha256(source_paths[arm]["metrics"]),
                "summary_path": portable_path(source_paths[arm]["summary"]),
                "summary_sha256": sha256(source_paths[arm]["summary"]),
                "checkpoint_sha256": source_summaries[arm]["checkpoint_sha256"],
                "ensemble": source_summaries[arm]["ensemble"],
            }
            for arm in ("random", "weighted")
        },
        "outputs": {
            portable_path(path): sha256(path)
            for key, path in outputs.items()
            if key != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(*(outputs.values()), sep="\n")


if __name__ == "__main__":
    main()
