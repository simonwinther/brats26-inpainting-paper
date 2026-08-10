#!/usr/bin/env python3
"""Plot paired weighted-versus-random effects for internal model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PAPER_ROOT / "data" / "model_selection75"
DEFAULT_OUTPUT_STEM = PAPER_ROOT / "figures" / "confirmation_effects"
METRICS = (
    ("ssim", r"(a) $\Delta$SSIM", "Weighted $-$ random", 1.0),
    ("psnr", r"(b) $\Delta$PSNR", "Weighted $-$ random (dB)", 1.0),
    (
        "mse",
        "(c) MSE reduction",
        r"Random $-$ weighted ($\times 10^{-2}$)",
        100.0,
    ),
)
JITTER_SEED = 2026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PAPER_ROOT))
    except ValueError:
        return str(resolved)


def read_metrics(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    order = []
    rows = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "ssim", "psnr", "mse"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} does not contain {sorted(required)}")
        for row in reader:
            case = row["case"].strip()
            if case in rows:
                raise ValueError(f"Duplicate case {case} in {path}")
            values = {
                metric: float(row[metric]) for metric in ("ssim", "psnr", "mse")
            }
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {case} in {path}")
            order.append(case)
            rows[case] = values
    if len(order) != 75:
        raise ValueError(f"Expected 75 cases in {path}, found {len(order)}")
    return order, rows


def read_pairwise(path: Path) -> dict[str, dict[str, float | int]]:
    records = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row["first_pipeline"] == "random"
                and row["second_pipeline"] == "weighted"
            ):
                records[row["metric"]] = {
                    "mean": float(row["mean_improvement"]),
                    "ci_low": float(row["bootstrap_ci_low"]),
                    "ci_high": float(row["bootstrap_ci_high"]),
                    "wins": int(row["wins"]),
                    "ties": int(row["ties"]),
                    "losses": int(row["losses"]),
                }
    if set(records) != {"ssim", "psnr", "mse"}:
        raise ValueError(f"{path} lacks the weighted-versus-random contrasts")
    return records


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_stem = args.output_stem.expanduser().resolve()
    random_path = data_dir / "random_metrics.csv"
    weighted_path = data_dir / "weighted_metrics.csv"
    pairwise_path = data_dir / "pairwise.csv"
    outputs = {
        "pdf": output_stem.with_suffix(".pdf"),
        "png": output_stem.with_suffix(".png"),
        "csv": output_stem.with_suffix(".csv"),
        "manifest": output_stem.with_suffix(".manifest.json"),
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    case_order, random_rows = read_metrics(random_path)
    weighted_order, weighted_rows = read_metrics(weighted_path)
    if weighted_order != case_order:
        raise ValueError("Random and weighted metric row order differs")
    paired = read_pairwise(pairwise_path)

    records = []
    differences: dict[str, np.ndarray] = {}
    for case in case_order:
        delta_ssim = weighted_rows[case]["ssim"] - random_rows[case]["ssim"]
        delta_psnr = weighted_rows[case]["psnr"] - random_rows[case]["psnr"]
        mse_reduction = random_rows[case]["mse"] - weighted_rows[case]["mse"]
        signs = np.sign([delta_ssim, delta_psnr, mse_reduction])
        if np.all(signs > 0):
            outcome = "weighted_wins_all"
        elif np.all(signs < 0):
            outcome = "random_wins_all"
        else:
            outcome = "mixed"
        records.append(
            {
                "case": case,
                "random_ssim": random_rows[case]["ssim"],
                "weighted_ssim": weighted_rows[case]["ssim"],
                "delta_ssim_weighted_minus_random": delta_ssim,
                "random_psnr": random_rows[case]["psnr"],
                "weighted_psnr": weighted_rows[case]["psnr"],
                "delta_psnr_db_weighted_minus_random": delta_psnr,
                "random_mse": random_rows[case]["mse"],
                "weighted_mse": weighted_rows[case]["mse"],
                "mse_reduction_random_minus_weighted": mse_reduction,
                "all_metric_outcome": outcome,
            }
        )
    differences["ssim"] = np.asarray(
        [row["delta_ssim_weighted_minus_random"] for row in records]
    )
    differences["psnr"] = np.asarray(
        [row["delta_psnr_db_weighted_minus_random"] for row in records]
    )
    differences["mse"] = np.asarray(
        [row["mse_reduction_random_minus_weighted"] for row in records]
    )
    for metric, values in differences.items():
        summary = paired[metric]
        expected_counts = (
            int(np.sum(values > 0)),
            int(np.sum(values == 0)),
            int(np.sum(values < 0)),
        )
        recorded_counts = (
            summary["wins"],
            summary["ties"],
            summary["losses"],
        )
        if expected_counts != recorded_counts:
            raise ValueError(
                f"{metric}: per-case and pairwise win/tie/loss counts differ"
            )
        if not np.isclose(values.mean(), summary["mean"], rtol=0, atol=1e-12):
            raise ValueError(f"{metric}: per-case and pairwise means differ")

    write_csv(outputs["csv"], records)
    outcome_counts = {
        label: sum(row["all_metric_outcome"] == label for row in records)
        for label in ("weighted_wins_all", "random_wins_all", "mixed")
    }
    if outcome_counts != {
        "weighted_wins_all": 51,
        "random_wins_all": 15,
        "mixed": 9,
    }:
        raise ValueError(f"Unexpected all-metric outcome counts: {outcome_counts}")

    configure_style()
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    rng = np.random.default_rng(JITTER_SEED)
    for axis, (metric, title, ylabel, scale) in zip(axes, METRICS):
        values = differences[metric] * scale
        summary = paired[metric]
        jitter = rng.uniform(-0.20, 0.20, size=len(values))
        positive = values > 0
        axis.axhline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=0)
        axis.scatter(
            jitter[positive],
            values[positive],
            s=12,
            color="#2A9D8F",
            alpha=0.72,
            linewidth=0,
            zorder=2,
        )
        axis.scatter(
            jitter[~positive],
            values[~positive],
            s=12,
            color="#E76F51",
            alpha=0.72,
            linewidth=0,
            zorder=2,
        )
        mean = float(summary["mean"]) * scale
        ci_low = float(summary["ci_low"]) * scale
        ci_high = float(summary["ci_high"]) * scale
        axis.errorbar(
            [0.34],
            [mean],
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="D",
            markersize=4.4,
            markerfacecolor="#F4A261",
            markeredgecolor="#202020",
            markeredgewidth=0.55,
            ecolor="#202020",
            elinewidth=1.15,
            capsize=2.8,
            zorder=4,
        )
        axis.set_title(title, fontweight="bold", pad=3)
        axis.set_ylabel(ylabel)
        axis.set_xlim(-0.28, 0.43)
        axis.set_xticks([])
        axis.text(
            0.03,
            0.97,
            f"W/L: {summary['wins']}/{summary['losses']}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=7.1,
        )
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#2A9D8F",
            markeredgewidth=0,
            markersize=4.5,
            label="Weighted wins",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#E76F51",
            markeredgewidth=0,
            markersize=4.5,
            label="Random wins",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="#202020",
            markerfacecolor="#F4A261",
            markersize=4.5,
            linewidth=1.0,
            label="Mean and bootstrap 95% CI",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.45,
    )
    figure.subplots_adjust(
        left=0.075, right=0.995, bottom=0.22, top=0.88, wspace=0.33
    )
    figure.savefig(
        outputs["pdf"],
        dpi=300,
        metadata={
            "Title": "Paired weighted-versus-random model-selection effects",
            "Author": "",
            "Subject": "BraTS 2026 CATCH internal model-selection analysis",
            "Keywords": "BraTS, MRI inpainting, paired effects",
            "Creator": "plot_confirmation_effects.py",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(
        outputs["png"],
        dpi=300,
        metadata={"Software": "plot_confirmation_effects.py"},
    )
    plt.close(figure)

    manifest = {
        "schema_version": 2,
        "analysis_role": "internal_training_policy_selection",
        "analysis": (
            "Paired weighted-mixture versus random-augmentation effects on the "
            "75-case internal model-selection cohort under matched mean-N=5 "
            "inference compute."
        ),
        "case_count": len(case_order),
        "positive_direction": {
            "ssim": "weighted minus random",
            "psnr": "weighted minus random in dB",
            "mse": "random minus weighted (MSE reduction)",
        },
        "all_metric_outcome_counts": outcome_counts,
        "jitter_seed": JITTER_SEED,
        "inputs": {
            portable_path(random_path): sha256(random_path),
            portable_path(weighted_path): sha256(weighted_path),
            portable_path(pairwise_path): sha256(pairwise_path),
        },
        "generator": portable_path(Path(__file__)),
        "generator_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            portable_path(outputs[key]): sha256(outputs[key])
            for key in ("csv", "pdf", "png")
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(*(outputs[key] for key in ("csv", "pdf", "png", "manifest")), sep="\n")


if __name__ == "__main__":
    main()
