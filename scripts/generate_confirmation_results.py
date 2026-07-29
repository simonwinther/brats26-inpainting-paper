#!/usr/bin/env python3
"""Audit and summarize the held-out 75-case confirmation evaluation.

The script snapshots the three canonical per-case CSV/JSON files into the
paper repository, verifies that the runs use the same held-out cohort and sampling
protocol, audits sibling-timepoint overlap with the optimization pool, and writes
the aggregate and paired statistics used by the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parent
DEFAULT_CASE_FILE = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "splits"
    / "checkpoint_confirmation_from_holdout_seed2026_n75.txt"
)
DEFAULT_DEVELOPMENT_FILE = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "splits"
    / "checkpoint_dev_from_holdout_seed2026_n25.txt"
)
DEFAULT_HOLDOUT_FILE = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "splits"
    / "holdout_val_seed2026_n100.txt"
)
DEFAULT_DATA_DIR = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "data"
    / "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
)
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "data" / "confirmation75"
DEFAULT_TABLE = PAPER_ROOT / "tables" / "confirmation75.tex"

METRICS = ("ssim", "psnr", "mse")
HIGHER_IS_BETTER = {"ssim": True, "psnr": True, "mse": False}
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 2026


@dataclass(frozen=True)
class Pipeline:
    key: str
    label: str
    checkpoint_step: int
    checkpoint_sha256: str
    ensemble_size: int
    metrics_path: Path
    summary_path: Path
    prediction_dir: Path
    sampling_metadata_path: Path


def default_pipelines() -> list[Pipeline]:
    results = REPOSITORY_ROOT / "brats_inpainting" / "results"
    specifications = (
        (
            "fixed",
            "Fixed mask",
            180_000,
            "ddc717d8d822e94ff0de62e05c02ea2351468e46648843d85f7bed52455e1c49",
            1,
            "paper300k-concat-fixed-s0-maxnorm-confirm75-ckpt180000-n1",
            "mean_n1",
        ),
        (
            "random",
            "Random augmentation",
            290_000,
            "69a4169fbead074819b3ed3b5392e08934dc1e7ac570933b4c825cb649d1ac27",
            5,
            "paper300k-concat-random-s0-maxnorm-confirm75-ckpt290000-mean-n5",
            "mean_n5",
        ),
        (
            "weighted",
            "Weighted mixture",
            200_000,
            "46578c5e1e41ed66c6d6ba65818279d78c5f104e1ae8c59507511c5e16883e2a",
            5,
            "paper300k-concat-weighted-s0-maxnorm-confirm75-ckpt200000-mean-n5",
            "mean_n5",
        ),
    )
    pipelines = []
    for (
        key,
        label,
        step,
        checkpoint_sha256,
        ensemble_size,
        run_name,
        ensemble_name,
    ) in specifications:
        run_dir = results / run_name
        metric_dir = run_dir / "ensemble_metrics" / ensemble_name
        prediction_dir = run_dir / "brats_inpainting"
        if ensemble_size > 1:
            prediction_dir /= ensemble_name
        pipelines.append(
            Pipeline(
                key=key,
                label=label,
                checkpoint_step=step,
                checkpoint_sha256=checkpoint_sha256,
                ensemble_size=ensemble_size,
                metrics_path=metric_dir / "metrics.csv",
                summary_path=metric_dir / "metrics.summary.json",
                prediction_dir=prediction_dir,
                sampling_metadata_path=(
                    run_dir / "brats_inpainting" / "sampling_metadata.json"
                ),
            )
        )
    return pipelines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument(
        "--development-file", type=Path, default=DEFAULT_DEVELOPMENT_FILE
    )
    parser.add_argument("--holdout-file", type=Path, default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    """Return a portable path relative to the paper checkout when possible."""
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
    patient_ids = [case.rsplit("-", 1)[0] for case in cases]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("The confirmation cohort contains repeated patient IDs")
    return cases


def read_case_list(path: Path, expected_count: int) -> list[str]:
    cases = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(cases) != expected_count or len(cases) != len(set(cases)):
        raise ValueError(
            f"Expected {expected_count} unique cases in {path}, found {len(cases)}"
        )
    return cases


def patient_id(case: str) -> str:
    return case.rsplit("-", 1)[0]


def audit_patient_overlap(
    cohort_cases: list[str],
    holdout_file: Path,
    data_dir: Path,
    *,
    expected_disjoint: int,
    expected_overlapping: int,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    holdout_cases = read_case_list(holdout_file, expected_count=100)
    all_cases = sorted(
        path.name
        for path in data_dir.iterdir()
        if path.is_dir() and path.name.startswith("BraTS-GLI-")
    )
    if len(all_cases) != 1_251 or len(all_cases) != len(set(all_cases)):
        raise ValueError(
            f"Expected 1251 unique dataset cases in {data_dir}, found "
            f"{len(all_cases)}"
        )
    unknown = set(holdout_cases) - set(all_cases)
    if unknown:
        raise ValueError(f"Holdout contains unknown cases: {sorted(unknown)}")
    optimization_cases = sorted(set(all_cases) - set(holdout_cases))
    if len(optimization_cases) != 1_151:
        raise ValueError(
            f"Expected 1151 optimization cases, found {len(optimization_cases)}"
        )
    optimization_by_patient: dict[str, list[str]] = {}
    for case in optimization_cases:
        optimization_by_patient.setdefault(patient_id(case), []).append(case)

    records = []
    disjoint_cases = []
    overlapping_cases = []
    for case in cohort_cases:
        siblings = optimization_by_patient.get(patient_id(case), [])
        is_disjoint = not siblings
        records.append(
            {
                "case": case,
                "patient_id": patient_id(case),
                "patient_disjoint_from_optimization": is_disjoint,
                "optimization_sibling_cases": ";".join(siblings),
            }
        )
        (disjoint_cases if is_disjoint else overlapping_cases).append(case)
    if (
        len(disjoint_cases) != expected_disjoint
        or len(overlapping_cases) != expected_overlapping
    ):
        raise ValueError(
            "Expected the audited "
            f"{expected_disjoint}/{expected_overlapping} "
            "patient-disjoint/overlapping split, found "
            f"{len(disjoint_cases)}/{len(overlapping_cases)}"
        )
    return records, disjoint_cases, overlapping_cases


def read_metrics(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", *METRICS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        order = []
        rows = {}
        for row in reader:
            case = row["case"].strip()
            if case in rows:
                raise ValueError(f"Duplicate case {case} in {path}")
            values = {metric: float(row[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {case} in {path}")
            order.append(case)
            rows[case] = values
    return order, rows


def validate_summary(
    pipeline: Pipeline,
    summary: dict,
    cases: list[str],
    metric_rows: dict[str, dict[str, float]],
    case_file: Path,
) -> None:
    expected = {
        "weights_type": "ema",
        "ema_rate": "0.9999",
        "checkpoint_sha256": pipeline.checkpoint_sha256,
        "seed": 0,
        "scored_case_count": 75,
        "dataset_split": "held_out_training_seed2026_subset_n75",
        "subset_file_sha256": sha256(case_file),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(
                f"{pipeline.key}: expected {key}={value!r}, "
                f"found {summary.get(key)!r}"
            )
    if summary.get("ensemble") != {
        "method": "mean",
        "size": pipeline.ensemble_size,
    }:
        raise ValueError(f"{pipeline.key}: unexpected ensemble configuration")
    sampling = summary.get("sampling_configuration", {})
    sampling_expected = {
        "conditioning_fusion": "concat",
        "ensemble_member_selection": "shared_first_n",
        "intensity_normalization": "voided_max_v1",
        "mask_conditioning": "nearest_signed",
        "sample_seed_schema": "brats-validation-noise-v1",
        "samples_per_case": pipeline.ensemble_size,
        "sampling_steps": 1000,
    }
    for key, value in sampling_expected.items():
        if sampling.get(key) != value:
            raise ValueError(
                f"{pipeline.key}: expected sampling {key}={value!r}, "
                f"found {sampling.get(key)!r}"
            )
    checkpoint_name = Path(summary["checkpoint"]).name
    expected_suffix = f"brats_inpainting_{pipeline.checkpoint_step}.pt"
    if not checkpoint_name.endswith(expected_suffix):
        raise ValueError(
            f"{pipeline.key}: checkpoint {checkpoint_name} does not end with "
            f"{expected_suffix}"
        )
    for metric in METRICS:
        values = np.asarray(
            [metric_rows[case][metric] for case in cases], dtype=np.float64
        )
        recorded = summary["metrics"][metric]
        if not np.isclose(values.mean(), recorded["mean"], rtol=0, atol=1e-12):
            raise ValueError(f"{pipeline.key}: {metric} mean mismatch")
        if not np.isclose(
            values.std(ddof=1),
            recorded["standard_deviation"],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(f"{pipeline.key}: {metric} SD mismatch")


def validate_sampling_metadata(
    pipeline: Pipeline,
    metadata: dict,
    cases: list[str],
    case_file: Path,
) -> dict[str, list[int]]:
    expected = {
        "checkpoint_sha256": pipeline.checkpoint_sha256,
        "weights_type": "ema",
        "ema_rate": "0.9999",
        "global_seed": 0,
        "sample_seed_schema": "brats-validation-noise-v1",
        "ensemble_member_selection": "shared_first_n",
        "intensity_normalization": "voided_max_v1",
        "conditioning_fusion": "concat",
        "mask_conditioning": "nearest_signed",
        "samples_per_case": pipeline.ensemble_size,
        "sampling_steps": 1000,
        "sample_count": len(cases),
        "expected_sample_count": len(cases),
        "subset_file_sha256": sha256(case_file),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"{pipeline.key}: expected sampling metadata {key}={value!r}, "
                f"found {metadata.get(key)!r}"
            )
    if metadata.get("ensemble_sizes") != [pipeline.ensemble_size]:
        raise ValueError(f"{pipeline.key}: unexpected metadata ensemble sizes")
    if metadata.get("ensemble_methods") != ["mean"]:
        raise ValueError(f"{pipeline.key}: unexpected metadata ensemble methods")
    samples = metadata.get("samples", [])
    if [sample.get("case_id") for sample in samples] != cases:
        raise ValueError(f"{pipeline.key}: sampling metadata case order differs")
    seeds_by_case = {}
    flattened_seeds = []
    for sample in samples:
        seeds = [int(value) for value in sample.get("sample_seeds", [])]
        if len(seeds) != pipeline.ensemble_size or len(seeds) != len(set(seeds)):
            raise ValueError(
                f"{pipeline.key}: invalid sample seed list for "
                f"{sample.get('case_id')}"
            )
        seeds_by_case[sample["case_id"]] = seeds
        flattened_seeds.extend(seeds)
    if len(flattened_seeds) != len(set(flattened_seeds)):
        raise ValueError(f"{pipeline.key}: sample seeds repeat across cases")
    return seeds_by_case


def average_tied_ranks(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    ranked = stats.rankdata(-values if higher_is_better else values, method="average")
    return np.asarray(ranked, dtype=np.float64)


def summarize_pipelines(
    pipelines: list[Pipeline],
    cases: list[str],
    rows: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, object]]:
    rank_totals = np.zeros(len(pipelines), dtype=np.float64)
    simultaneous_wins = np.zeros(len(pipelines), dtype=np.int64)
    metric_wins = {
        metric: np.zeros(len(pipelines), dtype=np.int64) for metric in METRICS
    }
    for case in cases:
        per_metric_ranks = []
        for metric in METRICS:
            values = np.asarray(
                [rows[pipeline.key][case][metric] for pipeline in pipelines],
                dtype=np.float64,
            )
            ranks = average_tied_ranks(values, HIGHER_IS_BETTER[metric])
            rank_totals += ranks
            per_metric_ranks.append(ranks)
            metric_wins[metric] += np.isclose(ranks, 1.0).astype(np.int64)
        case_ranks = np.stack(per_metric_ranks, axis=0)
        simultaneous_wins += np.all(np.isclose(case_ranks, 1.0), axis=0)

    summaries = []
    for index, pipeline in enumerate(pipelines):
        record: dict[str, object] = {
            "pipeline": pipeline.key,
            "label": pipeline.label,
            "checkpoint_step": pipeline.checkpoint_step,
            "weights": "EMA 0.9999",
            "ensemble_method": "mean",
            "ensemble_size": pipeline.ensemble_size,
            "case_count": len(cases),
            "joint_rank": rank_totals[index] / (len(cases) * len(METRICS)),
            "simultaneous_metric_wins": int(simultaneous_wins[index]),
        }
        for metric in METRICS:
            values = np.asarray(
                [rows[pipeline.key][case][metric] for case in cases],
                dtype=np.float64,
            )
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_sd"] = values.std(ddof=1)
            record[f"{metric}_wins"] = int(metric_wins[metric][index])
        summaries.append(record)
    return summaries


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def pairwise_statistics(
    pipelines: list[Pipeline],
    cases: list[str],
    rows: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    comparisons = ((0, 1), (0, 2), (1, 2))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(cases), size=(BOOTSTRAP_SAMPLES, len(cases))
    )
    records = []
    omnibus: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        arrays = [
            np.asarray(
                [rows[pipeline.key][case][metric] for case in cases],
                dtype=np.float64,
            )
            for pipeline in pipelines
        ]
        friedman = stats.friedmanchisquare(*arrays)
        omnibus[metric] = {
            "friedman_statistic": float(friedman.statistic),
            "degrees_of_freedom": len(pipelines) - 1,
            "p_value": float(friedman.pvalue),
        }
        metric_records = []
        raw_p_values = []
        for first, second in comparisons:
            # Positive always means that the second pipeline is better.
            difference = (
                arrays[second] - arrays[first]
                if HIGHER_IS_BETTER[metric]
                else arrays[first] - arrays[second]
            )
            bootstrap_means = difference[indices].mean(axis=1)
            interval = np.quantile(bootstrap_means, [0.025, 0.975])
            test = stats.wilcoxon(
                arrays[first],
                arrays[second],
                zero_method="wilcox",
                correction=False,
                alternative="two-sided",
                method="approx",
            )
            raw_p_values.append(float(test.pvalue))
            metric_records.append(
                {
                    "metric": metric,
                    "first_pipeline": pipelines[first].key,
                    "second_pipeline": pipelines[second].key,
                    "contrast": (
                        f"{pipelines[second].label} vs {pipelines[first].label}"
                    ),
                    "direction": (
                        f"{pipelines[second].label} minus {pipelines[first].label}"
                        if HIGHER_IS_BETTER[metric]
                        else (
                            f"{pipelines[first].label} minus "
                            f"{pipelines[second].label} (MSE reduction)"
                        )
                    ),
                    "mean_improvement": difference.mean(),
                    "bootstrap_ci_low": interval[0],
                    "bootstrap_ci_high": interval[1],
                    "bootstrap_samples": BOOTSTRAP_SAMPLES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "wins": int(np.sum(difference > 0)),
                    "ties": int(np.sum(difference == 0)),
                    "losses": int(np.sum(difference < 0)),
                    "effect_size_dz": (
                        difference.mean() / difference.std(ddof=1)
                    ),
                    "wilcoxon_statistic": float(test.statistic),
                    "wilcoxon_p_raw": float(test.pvalue),
                }
            )
        adjusted = holm_adjust(raw_p_values)
        for record, value in zip(metric_records, adjusted):
            record["wilcoxon_p_holm_within_metric"] = value
        records.extend(metric_records)
    return records, omnibus


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        (
            r"  \caption{Held-out 75-case confirmation results. Values are "
            r"mean $\pm$ sample SD. Joint rank averages each pipeline's per-case "
            r"SSIM, PSNR, and MSE ranks; lower is better. Bold and underlining "
            r"mark the best and second-best value in each numeric column. Fixed, "
            r"Random, and Weighted denote the fixed-mask, random-augmentation, "
            r"and weighted-mixture pipelines. Fixed uses $N=1$, so its contrasts "
            r"with the mean-$N=5$ pipelines "
            r"are not compute matched.}"
        ),
        r"  \label{tab:confirmation-results}",
        r"  \small",
        r"  \setlength{\tabcolsep}{2.0pt}",
        r"  \begin{tabular}{@{}lrrrr@{}}",
        r"    \toprule",
        (
            r"    Pipeline & SSIM $\uparrow$ & PSNR (dB) $\uparrow$ "
            r"& MSE $\downarrow$ ($\times10^{-2}$) & Rank $\downarrow$ \\"
        ),
        r"    \midrule",
    ]
    best = {
        "ssim": max(row["ssim_mean"] for row in summaries),
        "psnr": max(row["psnr_mean"] for row in summaries),
        "mse": min(row["mse_mean"] for row in summaries),
        "rank": min(row["joint_rank"] for row in summaries),
    }
    second = {
        "ssim": sorted(
            (row["ssim_mean"] for row in summaries), reverse=True
        )[1],
        "psnr": sorted(
            (row["psnr_mean"] for row in summaries), reverse=True
        )[1],
        "mse": sorted(row["mse_mean"] for row in summaries)[1],
        "rank": sorted(row["joint_rank"] for row in summaries)[1],
    }
    table_labels = {
        "fixed": "Fixed",
        "random": "Random",
        "weighted": "Weighted",
    }
    for row in summaries:
        inference = (
            r"$N=1$"
            if row["ensemble_size"] == 1
            else rf"mean $N={row['ensemble_size']}$"
        )
        formatted = {
            "ssim": rf"{row['ssim_mean']:.4f} $\pm$ {row['ssim_sd']:.4f}",
            "psnr": rf"{row['psnr_mean']:.2f} $\pm$ {row['psnr_sd']:.2f}",
            "mse": (
                rf"{100 * row['mse_mean']:.4f} "
                rf"$\pm$ {100 * row['mse_sd']:.4f}"
            ),
            "rank": f"{row['joint_rank']:.3f}",
        }
        for key, source in (
            ("ssim", row["ssim_mean"]),
            ("psnr", row["psnr_mean"]),
            ("mse", row["mse_mean"]),
            ("rank", row["joint_rank"]),
        ):
            if np.isclose(source, best[key], rtol=0, atol=1e-15):
                formatted[key] = rf"\textbf{{{formatted[key]}}}"
            elif np.isclose(source, second[key], rtol=0, atol=1e-15):
                formatted[key] = rf"\underline{{{formatted[key]}}}"
        lines.append(
            "    "
            + f"{table_labels[row['pipeline']]} ({inference}) & "
            + f"{formatted['ssim']} & "
            + f"{formatted['psnr']} & {formatted['mse']} & "
            + f"{formatted['rank']} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def ensure_fresh(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def main() -> None:
    args = parse_args()
    args.case_file = args.case_file.expanduser().resolve()
    args.development_file = args.development_file.expanduser().resolve()
    args.holdout_file = args.holdout_file.expanduser().resolve()
    args.data_dir = args.data_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.table = args.table.expanduser().resolve()
    pipelines = default_pipelines()
    cases = read_cases(args.case_file)
    development_cases = read_case_list(args.development_file, expected_count=25)
    if len({patient_id(case) for case in development_cases}) != len(
        development_cases
    ):
        raise ValueError("The development cohort contains repeated patient IDs")
    if set(development_cases) & set(cases):
        raise ValueError("Development and confirmation cohorts overlap")
    if {patient_id(case) for case in development_cases} & {
        patient_id(case) for case in cases
    }:
        raise ValueError("Development and confirmation cohorts overlap by patient")
    holdout_cases = set(read_case_list(args.holdout_file, expected_count=100))
    if set(development_cases) | set(cases) != holdout_cases:
        raise ValueError(
            "Development and confirmation manifests do not partition the holdout"
        )
    split_audit, disjoint_cases, overlapping_cases = audit_patient_overlap(
        cases,
        args.holdout_file,
        args.data_dir,
        expected_disjoint=64,
        expected_overlapping=11,
    )
    (
        development_split_audit,
        development_disjoint_cases,
        development_overlapping_cases,
    ) = audit_patient_overlap(
        development_cases,
        args.holdout_file,
        args.data_dir,
        expected_disjoint=22,
        expected_overlapping=3,
    )
    diagnostic_cases = development_cases[:5]
    if set(diagnostic_cases) & set(development_overlapping_cases):
        raise ValueError(
            "A checkpoint-diagnostic case overlaps optimization by patient"
        )

    targets = [
        args.output_dir / "aggregate.csv",
        args.output_dir / "pairwise.csv",
        args.output_dir / "case_split_audit.csv",
        args.output_dir / "development_case_split_audit.csv",
        args.output_dir / "patient_disjoint_aggregate.csv",
        args.output_dir / "patient_disjoint_pairwise.csv",
        args.output_dir / "prediction_hashes.csv",
        args.output_dir / "analysis.json",
        args.output_dir / "manifest.json",
        args.table,
    ]
    for pipeline in pipelines:
        targets.extend(
            [
                args.output_dir / f"{pipeline.key}_metrics.csv",
                args.output_dir / f"{pipeline.key}_summary.json",
            ]
        )
    ensure_fresh(targets, args.overwrite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table.parent.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, dict[str, dict[str, float]]] = {}
    source_records = []
    prediction_hash_records = []
    sampling_seeds_by_pipeline = {}
    for pipeline in pipelines:
        for source in (
            pipeline.metrics_path,
            pipeline.summary_path,
            pipeline.prediction_dir,
            pipeline.sampling_metadata_path,
        ):
            if not source.exists():
                raise FileNotFoundError(source)
        order, metric_rows = read_metrics(pipeline.metrics_path)
        if order != cases:
            raise ValueError(
                f"{pipeline.key}: metric order differs from the confirmation manifest"
            )
        summary = json.loads(pipeline.summary_path.read_text())
        validate_summary(pipeline, summary, cases, metric_rows, args.case_file)
        sampling_metadata = json.loads(
            pipeline.sampling_metadata_path.read_text()
        )
        sampling_seeds_by_pipeline[pipeline.key] = validate_sampling_metadata(
            pipeline, sampling_metadata, cases, args.case_file
        )
        prediction_names = {
            path.name.removesuffix("-t1n-inpainting.nii.gz")
            for path in pipeline.prediction_dir.glob("*-t1n-inpainting.nii.gz")
        }
        if prediction_names != set(cases):
            raise ValueError(
                f"{pipeline.key}: prediction set does not equal confirmation cohort"
            )
        prediction_inventory_digest = hashlib.sha256()
        for prediction_path in sorted(
            pipeline.prediction_dir.glob("*-t1n-inpainting.nii.gz")
        ):
            prediction_sha256 = sha256(prediction_path)
            prediction_inventory_digest.update(
                f"{prediction_sha256}  {prediction_path.name}\n".encode()
            )
            prediction_hash_records.append(
                {
                    "pipeline": pipeline.key,
                    "case": prediction_path.name.removesuffix(
                        "-t1n-inpainting.nii.gz"
                    ),
                    "filename": prediction_path.name,
                    "bytes": prediction_path.stat().st_size,
                    "sha256": prediction_sha256,
                }
            )
        all_rows[pipeline.key] = metric_rows
        metric_snapshot = args.output_dir / f"{pipeline.key}_metrics.csv"
        summary_snapshot = args.output_dir / f"{pipeline.key}_summary.json"
        shutil.copyfile(pipeline.metrics_path, metric_snapshot)
        shutil.copyfile(pipeline.summary_path, summary_snapshot)
        source_records.append(
            {
                "pipeline": pipeline.key,
                "label": pipeline.label,
                "checkpoint_step": pipeline.checkpoint_step,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "ensemble_method": "mean",
                "ensemble_size": pipeline.ensemble_size,
                "source_metrics": portable_path(pipeline.metrics_path),
                "source_metrics_sha256": sha256(pipeline.metrics_path),
                "source_summary": portable_path(pipeline.summary_path),
                "source_summary_sha256": sha256(pipeline.summary_path),
                "source_sampling_metadata": portable_path(
                    pipeline.sampling_metadata_path
                ),
                "source_sampling_metadata_sha256": sha256(
                    pipeline.sampling_metadata_path
                ),
                "snapshot_metrics": portable_path(metric_snapshot),
                "snapshot_metrics_sha256": sha256(metric_snapshot),
                "snapshot_summary": portable_path(summary_snapshot),
                "snapshot_summary_sha256": sha256(summary_snapshot),
                "source_git_commit": summary["launch"]["git_commit"],
                "source_git_dirty": summary["launch"]["git_dirty"],
                "prediction_file_count": len(prediction_names),
                "prediction_inventory_digest_sha256": (
                    prediction_inventory_digest.hexdigest()
                ),
            }
        )
    for case in cases:
        first_seeds = {
            sampling_seeds_by_pipeline[pipeline.key][case][0]
            for pipeline in pipelines
        }
        if len(first_seeds) != 1:
            raise ValueError(
                f"Selected pipelines do not share the first trajectory for {case}"
            )
        if (
            sampling_seeds_by_pipeline["random"][case]
            != sampling_seeds_by_pipeline["weighted"][case]
        ):
            raise ValueError(
                f"Random and weighted seed lists differ for {case}"
            )

    summaries = summarize_pipelines(pipelines, cases, all_rows)
    pairwise, omnibus = pairwise_statistics(pipelines, cases, all_rows)
    disjoint_summaries = summarize_pipelines(
        pipelines, disjoint_cases, all_rows
    )
    disjoint_pairwise, disjoint_omnibus = pairwise_statistics(
        pipelines, disjoint_cases, all_rows
    )
    aggregate_path = args.output_dir / "aggregate.csv"
    pairwise_path = args.output_dir / "pairwise.csv"
    split_audit_path = args.output_dir / "case_split_audit.csv"
    development_split_audit_path = (
        args.output_dir / "development_case_split_audit.csv"
    )
    disjoint_aggregate_path = (
        args.output_dir / "patient_disjoint_aggregate.csv"
    )
    disjoint_pairwise_path = args.output_dir / "patient_disjoint_pairwise.csv"
    prediction_hashes_path = args.output_dir / "prediction_hashes.csv"
    analysis_path = args.output_dir / "analysis.json"
    manifest_path = args.output_dir / "manifest.json"
    write_csv(aggregate_path, summaries)
    write_csv(pairwise_path, pairwise)
    write_csv(split_audit_path, split_audit)
    write_csv(development_split_audit_path, development_split_audit)
    write_csv(disjoint_aggregate_path, disjoint_summaries)
    write_csv(disjoint_pairwise_path, disjoint_pairwise)
    write_csv(prediction_hashes_path, prediction_hash_records)
    analysis_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "primary_cohort_size": len(cases),
                "bootstrap": {
                    "resamples": BOOTSTRAP_SAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "unit": "case",
                    "interval": "two-sided percentile 95%",
                },
                "paired_test": {
                    "name": "Wilcoxon signed-rank",
                    "alternative": "two-sided",
                    "continuity_correction": False,
                    "method": "asymptotic",
                    "multiplicity": (
                        "Holm adjustment over the three pipeline contrasts "
                        "separately within each metric"
                    ),
                },
                "omnibus": omnibus,
                "case_split_audit": {
                    "split_unit": "case ID",
                    "holdout_case_count": 100,
                    "optimization_case_count": 1_151,
                    "development_case_count": len(development_cases),
                    "development_patient_disjoint_case_count": len(
                        development_disjoint_cases
                    ),
                    "development_sibling_overlap_case_count": len(
                        development_overlapping_cases
                    ),
                    "development_sibling_overlap_cases": (
                        development_overlapping_cases
                    ),
                    "checkpoint_diagnostic_case_count": len(diagnostic_cases),
                    "checkpoint_diagnostic_sibling_overlap_case_count": 0,
                    "development_confirmation_patient_overlap_count": 0,
                    "confirmation_patient_disjoint_case_count": len(
                        disjoint_cases
                    ),
                    "confirmation_sibling_overlap_case_count": len(
                        overlapping_cases
                    ),
                    "confirmation_sibling_overlap_cases": overlapping_cases,
                },
                "patient_disjoint_sensitivity": {
                    "post_hoc": True,
                    "case_count": len(disjoint_cases),
                    "omnibus": disjoint_omnibus,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_table(args.table, summaries)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generator": portable_path(Path(__file__)),
                "generator_sha256": sha256(Path(__file__).resolve()),
                "confirmation_case_file": portable_path(args.case_file),
                "confirmation_case_file_sha256": sha256(args.case_file),
                "development_case_file": portable_path(args.development_file),
                "development_case_file_sha256": sha256(args.development_file),
                "holdout_case_file": portable_path(args.holdout_file),
                "holdout_case_file_sha256": sha256(args.holdout_file),
                "training_data_directory": portable_path(args.data_dir),
                "case_count": len(cases),
                "development_case_count": len(development_cases),
                "unique_patient_id_count": len(
                    {case.rsplit("-", 1)[0] for case in cases}
                ),
                "patient_disjoint_case_count": len(disjoint_cases),
                "sibling_overlap_case_count": len(overlapping_cases),
                "development_patient_disjoint_case_count": len(
                    development_disjoint_cases
                ),
                "development_sibling_overlap_case_count": len(
                    development_overlapping_cases
                ),
                "protocol": {
                    "weights": "EMA 0.9999",
                    "sampling_steps": 1000,
                    "global_seed": 0,
                    "sample_seed_schema": "brats-validation-noise-v1",
                    "ensemble_member_selection": "shared_first_n",
                    "normalization": "voided_max_v1",
                    "metric_region": "provided healthy mask",
                    "metric_normalization": (
                        "official voided-image percentile normalization"
                    ),
                    "candidate_rank_scope": (
                        "three selected pipelines on the held-out 75-case cohort"
                    ),
                },
                "sources": source_records,
                "outputs": {
                    portable_path(path): sha256(path)
                    for path in (
                        aggregate_path,
                        pairwise_path,
                        split_audit_path,
                        development_split_audit_path,
                        disjoint_aggregate_path,
                        disjoint_pairwise_path,
                        prediction_hashes_path,
                        analysis_path,
                        args.table,
                    )
                },
                "caveats": [
                    (
                        "The comparison is between selected pipelines; checkpoint "
                        "step and inference compute differ between arms."
                    ),
                    (
                        "All arms use one training seed. Across-case uncertainty "
                        "does not quantify training-seed variability."
                    ),
                    (
                        "The deterministic split is keyed by case ID rather than "
                        "patient ID. Three development cases and eleven confirmation "
                        "cases have a sibling timepoint in the optimization pool. "
                        "A post-hoc 64-case patient-disjoint confirmation sensitivity "
                        "analysis is provided; it does not remove overlap from "
                        "development-stage inference-policy selection."
                    ),
                    (
                        "Some source runs recorded a dirty worktree; checkpoint, "
                        "split, configuration, metric, and prediction hashes remain "
                        "available, but the exact dirty diff was not captured."
                    ),
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(
        *[
            aggregate_path,
            pairwise_path,
            split_audit_path,
            development_split_audit_path,
            disjoint_aggregate_path,
            disjoint_pairwise_path,
            prediction_hashes_path,
            analysis_path,
            manifest_path,
            args.table,
        ],
        sep="\n",
    )


if __name__ == "__main__":
    main()
