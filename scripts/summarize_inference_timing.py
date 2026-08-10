#!/usr/bin/env python3
"""Summarize steady-state inference time for the 75-case model-selection runs.

Cases are assigned to the eight inference shards round-robin in manifest order.
For each shard, the interval between adjacent completed prediction files measures
steady-state end-to-end case time while excluding one-time model startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


SHARD_COUNT = 8
RUNS = (
    {
        "key": "fixed_n1",
        "label": "Fixed mask, N=1",
        "ensemble_size": 1,
        "run_name": "paper300k-concat-fixed-s0-maxnorm-confirm75-ckpt180000-n1",
        "prediction_subdir": "brats_inpainting",
    },
    {
        "key": "random_n5",
        "label": "Random augmentation, mean N=5",
        "ensemble_size": 5,
        "run_name": "paper300k-concat-random-s0-maxnorm-confirm75-ckpt290000-mean-n5",
        "prediction_subdir": "brats_inpainting/mean_n5",
    },
    {
        "key": "weighted_n5",
        "label": "Weighted mixture, mean N=5",
        "ensemble_size": 5,
        "run_name": "paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5",
        "prediction_subdir": "brats_inpainting/mean_n5",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Checkout containing brats_inpainting and brats26-inpainting-paper.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output JSON path, or '-' for stdout.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--generator-sha256",
        default="",
        help="Optional script hash when executing the script through stdin.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_cases(path: Path) -> list[str]:
    cases = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(cases) != 75 or len(cases) != len(set(cases)):
        raise ValueError(f"Expected 75 unique cases in {path}, found {len(cases)}")
    return cases


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "interval_count": len(values),
        "mean_minutes": round(statistics.mean(values) / 60, 6),
        "median_minutes": round(statistics.median(values) / 60, 6),
        "sample_sd_minutes": round(statistics.stdev(values) / 60, 6),
        "minimum_minutes": round(min(values) / 60, 6),
        "maximum_minutes": round(max(values) / 60, 6),
    }


def analyze_run(
    repository_root: Path,
    cases: list[str],
    specification: dict[str, object],
) -> dict[str, object]:
    run_dir = (
        repository_root
        / "brats_inpainting"
        / "results"
        / str(specification["run_name"])
    )
    prediction_dir = run_dir / str(specification["prediction_subdir"])
    metadata_path = run_dir / "brats_inpainting" / "sampling_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    intervals: list[dict[str, object]] = []
    per_shard: list[dict[str, object]] = []
    for shard_index in range(SHARD_COUNT):
        shard_cases = cases[shard_index::SHARD_COUNT]
        paths = [
            prediction_dir / f"{case}-t1n-inpainting.nii.gz"
            for case in shard_cases
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Run {specification['key']} is missing {len(missing)} prediction(s); "
                f"first missing path: {missing[0]}"
            )
        mtimes_ns = [path.stat().st_mtime_ns for path in paths]
        shard_values = []
        for previous_case, case, previous_ns, current_ns in zip(
            shard_cases, shard_cases[1:], mtimes_ns, mtimes_ns[1:]
        ):
            elapsed_seconds = (current_ns - previous_ns) / 1_000_000_000
            if elapsed_seconds <= 0:
                raise ValueError(
                    f"Non-positive save interval in {specification['key']} "
                    f"shard {shard_index}: {previous_case} -> {case}"
                )
            shard_values.append(elapsed_seconds)
            intervals.append(
                {
                    "shard_index": shard_index,
                    "previous_case": previous_case,
                    "case": case,
                    "elapsed_seconds": elapsed_seconds,
                }
            )
        per_shard.append(
            {
                "shard_index": shard_index,
                "case_count": len(shard_cases),
                **summarize(shard_values),
            }
        )

    values = [float(record["elapsed_seconds"]) for record in intervals]
    return {
        "key": specification["key"],
        "label": specification["label"],
        "ensemble_size": specification["ensemble_size"],
        "run_directory": str(run_dir.relative_to(repository_root)),
        "prediction_directory": str(prediction_dir.relative_to(repository_root)),
        "sampling_metadata": {
            "path": str(metadata_path.relative_to(repository_root)),
            "sha256": sha256(metadata_path),
        },
        "case_count": len(cases),
        "shard_count": SHARD_COUNT,
        "summary": summarize(values),
        "per_shard": per_shard,
    }


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    case_file = (
        repository_root
        / "brats_inpainting"
        / "splits"
        / "checkpoint_confirmation_from_holdout_seed2026_n75.txt"
    )
    cases = read_cases(case_file)
    runs = [analyze_run(repository_root, cases, specification) for specification in RUNS]
    means = {run["key"]: run["summary"]["mean_minutes"] for run in runs}

    script_path = Path(__file__)
    generator_hash = args.generator_sha256
    if not generator_hash and script_path.is_file():
        generator_hash = sha256(script_path)
    output = {
        "schema_version": 2,
        "analysis_role": "descriptive_runtime_measurement_on_model_selection_runs",
        "analysis": (
            "Steady-state end-to-end inference time for the 75-case internal "
            "model-selection runs on NVIDIA A100 40 GB GPUs, measured as the "
            "interval between adjacent saved predictions within each deterministic "
            "Slurm shard."
        ),
        "measurement_scope": (
            "Intervals include case loading, preprocessing, 1000-step ancestral "
            "sampling, trajectory aggregation, hard compositing, and NIfTI output "
            "for the later case; one-time process and model startup are excluded."
        ),
        "mtime_requirement": (
            "Recomputation requires the canonical prediction files with their "
            "original Hendrix modification timestamps. Per-shard interval "
            "summaries are snapshotted below so the aggregate remains auditable."
        ),
        "generator": "scripts/summarize_inference_timing.py",
        "generator_sha256": generator_hash,
        "case_manifest": {
            "path": str(case_file.relative_to(repository_root)),
            "sha256": sha256(case_file),
            "case_count": len(cases),
            "inferential_role": "internal_training_policy_selection",
            "legacy_filename_note": (
                "The immutable split filename contains 'confirmation'; it is the "
                "75-case internal model-selection cohort."
            ),
        },
        "runs": runs,
        "mean_time_ratios": {
            "random_n5_over_fixed_n1": round(
                means["random_n5"] / means["fixed_n1"], 6
            ),
            "weighted_n5_over_fixed_n1": round(
                means["weighted_n5"] / means["fixed_n1"], 6
            ),
        },
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"

    if args.output == "-":
        sys.stdout.write(rendered)
        return
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)


if __name__ == "__main__":
    main()
