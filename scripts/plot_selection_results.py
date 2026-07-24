#!/usr/bin/env python3
"""Generate checkpoint and ensemble selection figures from official metrics CSVs.

Checkpoint example:
    python scripts/plot_selection_results.py checkpoints \
        --metrics "Concat-fixed" 100000 /path/to/metrics.csv \
        --metrics "Concat-fixed" 125000 /path/to/metrics.csv

Ensemble example:
    python scripts/plot_selection_results.py ensembles \
        --metrics-root /path/to/ensemble_metrics

Both modes require the exact 25-case development cohort by default. They verify
case parity before computing a paired, challenge-style rank over SSIM, PSNR, and
MSE. No dashboard smoothing or interpolation is applied.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parent
DEFAULT_CASE_FILE = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "splits"
    / "checkpoint_dev_from_holdout_seed2026_n25.txt"
)
METRICS = ("ssim", "psnr", "mse")
HIGHER_IS_BETTER = {"ssim": True, "psnr": True, "mse": False}
PANEL_SPECS = (
    ("ssim", "(a) SSIM", r"SSIM $\uparrow$"),
    ("psnr", "(b) PSNR", r"PSNR (dB) $\uparrow$"),
    ("mse", "(c) MSE", r"MSE $\downarrow$"),
    ("mean_rank", "(d) Joint rank", r"Mean rank $\downarrow$"),
)
COLORS = ("#4C78A8", "#E45756", "#54A24B", "#B279A2", "#F58518")
ENSEMBLE_STYLES = {
    "mean": {"color": "#007C78", "marker": "o", "label": "Mean"},
    "median": {"color": "#8B5CF6", "marker": "s", "label": "Median"},
}
ENSEMBLE_PATTERN = re.compile(r"^(mean|median)_n([1-9][0-9]*)$")


@dataclass(frozen=True)
class Candidate:
    label: str
    group: str
    position: int
    metrics_path: Path
    rows: dict[str, dict[str, float]]
    method: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    checkpoint_parser = subparsers.add_parser(
        "checkpoints",
        help="Plot checkpoint candidates supplied as label, step, and metrics CSV.",
    )
    checkpoint_parser.add_argument(
        "--metrics",
        action="append",
        nargs=3,
        required=True,
        metavar=("MODEL", "STEP", "CSV"),
        help="Repeat once for each checkpoint candidate.",
    )
    checkpoint_parser.add_argument(
        "--case-file",
        type=Path,
        default=DEFAULT_CASE_FILE,
    )
    checkpoint_parser.add_argument(
        "--output-stem",
        type=Path,
        default=PAPER_ROOT / "figures" / "checkpoint_selection",
    )
    checkpoint_parser.add_argument("--overwrite", action="store_true")

    ensemble_parser = subparsers.add_parser(
        "ensembles",
        help="Plot mean/median variants found under an ensemble_metrics directory.",
    )
    ensemble_parser.add_argument("--metrics-root", type=Path, required=True)
    ensemble_parser.add_argument(
        "--case-file",
        type=Path,
        default=DEFAULT_CASE_FILE,
    )
    ensemble_parser.add_argument(
        "--output-stem",
        type=Path,
        default=PAPER_ROOT / "figures" / "ensemble_selection",
    )
    ensemble_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_case_file(path: Path) -> list[str]:
    case_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not case_ids:
        raise ValueError(f"No case IDs found in {path}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs found in {path}")
    return case_ids


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics CSV does not exist: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", *METRICS}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows: dict[str, dict[str, float]] = {}
        for source in reader:
            case_id = source["case"].strip()
            if not case_id:
                raise ValueError(f"Empty case ID in {path}")
            if case_id in rows:
                raise ValueError(f"Duplicate case {case_id} in {path}")
            values = {metric: float(source[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {case_id} in {path}")
            rows[case_id] = values
    if not rows:
        raise ValueError(f"No metric rows found in {path}")
    return rows


def require_case_parity(
    candidates: Iterable[Candidate], expected_case_ids: list[str]
) -> None:
    expected = set(expected_case_ids)
    for candidate in candidates:
        actual = set(candidate.rows)
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise ValueError(
                f"Case mismatch for {candidate.label} at {candidate.position}: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )


def average_tied_ranks(values: list[float], higher_is_better: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(-array if higher_is_better else array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def summarize(
    candidates: list[Candidate], case_ids: list[str]
) -> list[dict[str, object]]:
    rank_totals = np.zeros(len(candidates), dtype=np.float64)
    for case_id in case_ids:
        for metric in METRICS:
            values = [
                candidate.rows[case_id][metric] for candidate in candidates
            ]
            rank_totals += average_tied_ranks(
                values, HIGHER_IS_BETTER[metric]
            )
    mean_ranks = rank_totals / (len(case_ids) * len(METRICS))

    summaries = []
    for index, candidate in enumerate(candidates):
        result: dict[str, object] = {
            "label": candidate.label,
            "group": candidate.group,
            "position": candidate.position,
            "method": candidate.method or "",
            "metrics_csv": str(candidate.metrics_path.resolve()),
            "metrics_csv_sha256": sha256(candidate.metrics_path),
            "case_count": len(case_ids),
            "mean_rank": float(mean_ranks[index]),
        }
        for metric in METRICS:
            values = np.asarray(
                [candidate.rows[case_id][metric] for case_id in case_ids],
                dtype=np.float64,
            )
            result[f"mean_{metric}"] = float(values.mean())
            result[f"sd_{metric}"] = float(values.std(ddof=1))
        summaries.append(result)
    return summaries


def checkpoint_candidates(entries: list[list[str]]) -> list[Candidate]:
    candidates = []
    seen = set()
    for label, raw_step, raw_path in entries:
        try:
            step = int(raw_step)
        except ValueError as error:
            raise ValueError(f"Invalid checkpoint step: {raw_step}") from error
        if step <= 0:
            raise ValueError(f"Checkpoint step must be positive: {step}")
        key = (label, step)
        if key in seen:
            raise ValueError(f"Duplicate checkpoint candidate: {label} {step}")
        seen.add(key)
        path = Path(raw_path).expanduser().resolve()
        candidates.append(
            Candidate(
                label=f"{label} {step // 1000}k",
                group=label,
                position=step,
                metrics_path=path,
                rows=read_metrics(path),
            )
        )
    return sorted(candidates, key=lambda item: (item.group, item.position))


def ensemble_candidates(metrics_root: Path) -> list[Candidate]:
    metrics_root = metrics_root.expanduser().resolve()
    discovered: dict[tuple[str, int], Candidate] = {}
    for metrics_path in sorted(metrics_root.glob("*/metrics.csv")):
        match = ENSEMBLE_PATTERN.fullmatch(metrics_path.parent.name)
        if not match:
            continue
        method, raw_size = match.groups()
        size = int(raw_size)
        discovered[(method, size)] = Candidate(
            label=f"{method.title()} N={size}",
            group=method,
            position=size,
            metrics_path=metrics_path,
            rows=read_metrics(metrics_path),
            method=method,
        )
    if not discovered:
        raise ValueError(f"No mean_n*/median_n*/metrics.csv files in {metrics_root}")

    size_one = [
        candidate
        for (method, size), candidate in discovered.items()
        if size == 1
    ]
    if not size_one:
        raise ValueError("Ensemble comparison requires an N=1 reference")
    reference = size_one[0]
    for candidate in size_one[1:]:
        if candidate.rows != reference.rows:
            raise ValueError("Mean and median N=1 metrics are not identical")

    candidates = [
        Candidate(
            label="Single N=1",
            group="single",
            position=1,
            metrics_path=reference.metrics_path,
            rows=reference.rows,
            method="single",
        )
    ]
    candidates.extend(
        candidate
        for (method, size), candidate in sorted(
            discovered.items(), key=lambda item: (item[0][1], item[0][0])
        )
        if size > 1
    )
    return candidates


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def prepare_axis(axis: plt.Axes, title: str, ylabel: str) -> None:
    axis.set_title(title, fontweight="bold", pad=3)
    axis.set_ylabel(ylabel)
    axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def annotate_best(axis: plt.Axes, summary: dict[str, object], x: float) -> None:
    y = float(summary["mean_rank"])
    axis.scatter(
        [x],
        [y],
        marker="*",
        s=75,
        color="#C23B22",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
    )


def plot_checkpoints(
    summaries: list[dict[str, object]], output_stem: Path
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 4.35))
    axes = axes.ravel()
    groups = list(dict.fromkeys(str(item["group"]) for item in summaries))
    colors = {group: COLORS[index % len(COLORS)] for index, group in enumerate(groups)}

    for axis, (metric, title, ylabel) in zip(axes, PANEL_SPECS):
        for group in groups:
            rows = sorted(
                (item for item in summaries if item["group"] == group),
                key=lambda item: int(item["position"]),
            )
            x = [int(item["position"]) / 1000 for item in rows]
            y = [
                float(item[metric])
                if metric == "mean_rank"
                else float(item[f"mean_{metric}"])
                for item in rows
            ]
            axis.plot(
                x,
                y,
                color=colors[group],
                linewidth=1.3,
                marker="o",
                markersize=3.6,
                label=group,
            )
        prepare_axis(axis, title, ylabel)
        axis.set_xlabel("Checkpoint step (thousands)")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    best = min(summaries, key=lambda item: float(item["mean_rank"]))
    annotate_best(axes[3], best, int(best["position"]) / 1000)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(labels)),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )
    figure.subplots_adjust(
        left=0.08, right=0.995, bottom=0.115, top=0.88, wspace=0.3, hspace=0.42
    )
    return save_figure(figure, output_stem)


def ensemble_plot_rows(
    summaries: list[dict[str, object]], method: str
) -> list[dict[str, object]]:
    single = next(item for item in summaries if item["method"] == "single")
    rows = [single]
    rows.extend(
        sorted(
            (item for item in summaries if item["method"] == method),
            key=lambda item: int(item["position"]),
        )
    )
    return rows


def plot_ensembles(
    summaries: list[dict[str, object]], output_stem: Path
) -> list[Path]:
    configure_style()
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
    methods = [
        method
        for method in ("mean", "median")
        if any(item["method"] == method for item in summaries)
    ]
    for axis, (metric, title, ylabel) in zip(axes, PANEL_SPECS):
        for method in methods:
            rows = ensemble_plot_rows(summaries, method)
            style = ENSEMBLE_STYLES[method]
            x = [int(item["position"]) for item in rows]
            y = [
                float(item[metric])
                if metric == "mean_rank"
                else float(item[f"mean_{metric}"])
                for item in rows
            ]
            axis.plot(
                x,
                y,
                color=style["color"],
                linewidth=1.35,
                marker=style["marker"],
                markersize=3.8,
                label=style["label"],
            )
        prepare_axis(axis, title, ylabel)
        axis.set_xlabel("Reconstructions per case")
        axis.set_xticks(
            sorted({int(item["position"]) for item in summaries})
        )
    axes[2].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    best = min(summaries, key=lambda item: float(item["mean_rank"]))
    annotate_best(axes[3], best, int(best["position"]))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(labels),
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
    )
    figure.subplots_adjust(
        left=0.07, right=0.995, bottom=0.25, top=0.76, wspace=0.42
    )
    return save_figure(figure, output_stem)


def save_figure(figure: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem = output_stem.expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return outputs


def ensure_output_targets(output_stem: Path, overwrite: bool) -> None:
    output_stem = output_stem.expanduser().resolve()
    targets = [
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".csv"),
        output_stem.with_suffix(".manifest.json"),
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing figure artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def write_summary_csv(
    summaries: list[dict[str, object]], output_stem: Path, overwrite: bool
) -> Path:
    path = output_stem.expanduser().resolve().with_suffix(".csv")
    fieldnames = [
        "label",
        "group",
        "position",
        "method",
        "case_count",
        "mean_ssim",
        "sd_ssim",
        "mean_psnr",
        "sd_psnr",
        "mean_mse",
        "sd_mse",
        "mean_rank",
        "metrics_csv",
        "metrics_csv_sha256",
    ]
    with path.open("w" if overwrite else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return path


def write_manifest(
    *,
    mode: str,
    case_file: Path,
    outputs: list[Path],
    summary_csv: Path,
    summaries: list[dict[str, object]],
    output_stem: Path,
    overwrite: bool,
) -> Path:
    best = min(summaries, key=lambda item: float(item["mean_rank"]))
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "figure": mode,
        "selection_cohort": str(case_file.resolve()),
        "selection_cohort_sha256": sha256(case_file),
        "case_count": int(best["case_count"]),
        "metrics": {
            "ssim": "higher_is_better",
            "psnr": "higher_is_better",
            "mse": "lower_is_better",
        },
        "selection_rule": (
            "Average per-case candidate ranks over SSIM, PSNR, and MSE; "
            "lower mean rank is better."
        ),
        "dashboard_smoothing": False,
        "selected": {
            "label": best["label"],
            "group": best["group"],
            "position": best["position"],
            "method": best["method"],
            "mean_rank": best["mean_rank"],
        },
        "inputs": [
            {
                "label": item["label"],
                "metrics_csv": item["metrics_csv"],
                "metrics_csv_sha256": item["metrics_csv_sha256"],
            }
            for item in summaries
        ],
        "derived_summary_csv": str(summary_csv),
        "derived_summary_csv_sha256": sha256(summary_csv),
        "outputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in outputs
        ],
    }
    path = output_stem.expanduser().resolve().with_suffix(".manifest.json")
    with path.open("w" if overwrite else "x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def main() -> None:
    args = parse_args()
    case_file = args.case_file.expanduser().resolve()
    case_ids = read_case_file(case_file)
    ensure_output_targets(args.output_stem, args.overwrite)
    if args.mode == "checkpoints":
        candidates = checkpoint_candidates(args.metrics)
    else:
        candidates = ensemble_candidates(args.metrics_root)
    require_case_parity(candidates, case_ids)
    summaries = summarize(candidates, case_ids)
    if args.mode == "checkpoints":
        outputs = plot_checkpoints(summaries, args.output_stem)
        manifest_mode = "checkpoint_selection"
    else:
        outputs = plot_ensembles(summaries, args.output_stem)
        manifest_mode = "ensemble_selection"
    summary_csv = write_summary_csv(
        summaries, args.output_stem, args.overwrite
    )
    manifest = write_manifest(
        mode=manifest_mode,
        case_file=case_file,
        outputs=outputs,
        summary_csv=summary_csv,
        summaries=summaries,
        output_stem=args.output_stem,
        overwrite=args.overwrite,
    )
    print(*(str(path) for path in [*outputs, summary_csv, manifest]), sep="\n")


if __name__ == "__main__":
    main()
