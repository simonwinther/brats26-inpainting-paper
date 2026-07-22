#!/usr/bin/env python3
"""Plot raw W&B validation histories captured in the paper data snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter


PAPER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PAPER_ROOT / "data" / "wandb_training_dynamics.csv"
DEFAULT_OUTPUT = PAPER_ROOT / "figures" / "training_dynamics"

RUN_ORDER = (
    "Concat-fixed",
    "FiLM-fixed",
    "Concat-random",
    "Concat-weighted",
    "FiLM-weighted",
)

RUN_STYLES = {
    "Concat-fixed": {"color": "#4C78A8", "linestyle": "-", "marker": "o"},
    "FiLM-fixed": {"color": "#F58518", "linestyle": "--", "marker": "s"},
    "Concat-random": {"color": "#54A24B", "linestyle": "-.", "marker": "^"},
    "Concat-weighted": {"color": "#E45756", "linestyle": "-", "marker": "D"},
    "FiLM-weighted": {"color": "#B279A2", "linestyle": ":", "marker": "X"},
}

PANELS = (
    ("val_ema_ssim", "(a) SSIM", r"SSIM $\uparrow$"),
    ("val_ema_psnr", "(b) PSNR", r"PSNR (dB) $\uparrow$"),
    ("val_ema_mse", "(c) MSE", r"MSE $\downarrow$ (log scale)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="Include rows excluded from the predeclared manuscript figure.",
    )
    return parser.parse_args()


def load_series(path: Path, include_exploratory: bool) -> dict[str, list[dict]]:
    series: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not include_exploratory and row["include_in_main"] != "true":
                continue
            series[row["label"]].append(
                {
                    "run_id": row["run_id"],
                    "step": int(row["step"]),
                    "val_ema_ssim": float(row["val_ema_ssim"]),
                    "val_ema_psnr": float(row["val_ema_psnr"]),
                    "val_ema_mse": float(row["val_ema_mse"]),
                }
            )
    for rows in series.values():
        rows.sort(key=lambda row: row["step"])
    return dict(series)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_series(series: dict[str, list[dict]], output_stem: Path) -> list[Path]:
    configure_style()
    labels = [label for label in RUN_ORDER if label in series]
    if not labels:
        raise ValueError("No rows were selected for plotting.")

    max_step = max(row["step"] for rows in series.values() for row in rows)
    max_step_thousands = math.ceil(max_step / 10_000) * 10

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for axis, (metric, title, ylabel) in zip(axes, PANELS):
        for label in labels:
            rows = series[label]
            x = [row["step"] / 1000 for row in rows]
            y = [row[metric] for row in rows]
            style = RUN_STYLES[label]
            axis.plot(
                x,
                y,
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.25,
                marker=style["marker"],
                markersize=2.4,
                markeredgewidth=0,
            )
        axis.set_title(title, fontweight="bold", pad=3)
        axis.set_xlabel("Training step (thousands)")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, max_step_thousands)
        axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[2].set_yscale("log")
    axes[2].set_yticks([0.01, 0.02, 0.04, 0.08])
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[2].yaxis.set_minor_formatter(NullFormatter())

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(labels),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.065, right=0.995, bottom=0.22, top=0.79, wspace=0.34)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=180, bbox_inches="tight")
    plt.close(figure)
    return outputs


def write_manifest(
    input_path: Path,
    output_stem: Path,
    outputs: list[Path],
    series: dict[str, list[dict]],
    include_exploratory: bool,
) -> Path:
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "outputs": [str(path) for path in outputs],
        "include_exploratory": include_exploratory,
        "dashboard_smoothing": False,
        "metrics": [metric for metric, _, _ in PANELS],
        "runs": [
            {
                "label": label,
                "run_id": series[label][0]["run_id"],
                "points": len(series[label]),
                "latest_validation_step": series[label][-1]["step"],
            }
            for label in RUN_ORDER
            if label in series
        ],
    }
    path = output_stem.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main() -> None:
    args = parse_args()
    series = load_series(args.input, args.include_exploratory)
    outputs = plot_series(series, args.output_stem)
    manifest = write_manifest(
        args.input,
        args.output_stem,
        outputs,
        series,
        args.include_exploratory,
    )
    print(*(str(path) for path in [*outputs, manifest]), sep="\n")


if __name__ == "__main__":
    main()
