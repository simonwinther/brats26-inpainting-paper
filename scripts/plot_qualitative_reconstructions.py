#!/usr/bin/env python3
"""Create an audited qualitative figure from saved native-space predictions.

The script ranks the locked confirmation cases by the frozen ensemble's SSIM,
selects the observations nearest the 10th, 50th, and 90th percentiles, and
shows the same maximum-healthy-mask-area axial slice for every panel.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PAPER_ROOT.parent
DEFAULT_CASE_FILE = (
    REPOSITORY_ROOT
    / "brats_inpainting"
    / "splits"
    / "checkpoint_confirmation_from_holdout_seed2026_n75.txt"
)
METRICS = ("ssim", "psnr", "mse")
PREDICTION_SUFFIXES = (
    "-t1n-inpainting.nii.gz",
    "-t1n-inference.nii.gz",
)
QUANTILES = (
    (0.10, "10th percentile"),
    (0.50, "Median"),
    (0.90, "90th percentile"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-metrics", type=Path, required=True)
    parser.add_argument("--ensemble-metrics", type=Path, required=True)
    parser.add_argument("--single-pred-dir", type=Path, required=True)
    parser.add_argument("--ensemble-pred-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument("--ensemble-label", default="Ensemble (N=5)")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PAPER_ROOT / "figures" / "qualitative_reconstructions",
    )
    parser.add_argument("--overwrite", action="store_true")
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
        rows = {}
        for source in reader:
            case_id = source["case"].strip()
            if case_id in rows:
                raise ValueError(f"Duplicate case {case_id} in {path}")
            values = {metric: float(source[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric for {case_id} in {path}")
            rows[case_id] = values
    return rows


def require_exact_cases(
    rows: dict[str, dict[str, float]], case_ids: list[str], source: Path
) -> None:
    expected = set(case_ids)
    actual = set(rows)
    if actual != expected:
        raise ValueError(
            f"Case mismatch for {source}: "
            f"missing={sorted(expected - actual)[:5]} "
            f"unexpected={sorted(actual - expected)[:5]}"
        )


def find_case_dir(data_dir: Path, case_id: str) -> Path:
    direct = data_dir / case_id
    if direct.is_dir():
        return direct
    matches = [path for path in data_dir.rglob(case_id) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one data directory for {case_id}, found {len(matches)}"
        )
    return matches[0]


def prediction_path(directory: Path, case_id: str) -> Path:
    matches = [
        directory / f"{case_id}{suffix}"
        for suffix in PREDICTION_SUFFIXES
        if (directory / f"{case_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one prediction for {case_id} in {directory}, found {matches}"
        )
    return matches[0]


def load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.load(path))
    return np.asarray(image.dataobj, dtype=np.float32), image.affine


def select_cases(
    rows: dict[str, dict[str, float]], case_ids: list[str]
) -> list[dict[str, object]]:
    ordered = sorted(case_ids, key=lambda case_id: (rows[case_id]["ssim"], case_id))
    selected = []
    used = set()
    for quantile, label in QUANTILES:
        index = int(round(quantile * (len(ordered) - 1)))
        if index in used:
            raise ValueError("Quantile selection produced duplicate cases")
        used.add(index)
        case_id = ordered[index]
        selected.append(
            {
                "quantile": quantile,
                "quantile_label": label,
                "rank_index": index,
                "case_id": case_id,
            }
        )
    return selected


def normalization_bounds(voided: np.ndarray) -> tuple[float, float]:
    lo, hi = (float(value) for value in np.percentile(voided, [0.5, 99.5]))
    if not hi > lo:
        hi = float(voided.max())
        lo = 0.0
    if not hi > lo:
        return 0.0, 1.0
    return lo, hi


def normalize(volume: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lo, hi = bounds
    return np.clip((volume - lo) / (hi - lo), 0.0, 1.0)


def display_window(voided: np.ndarray) -> tuple[float, float]:
    foreground = voided[voided > 0]
    source = foreground if foreground.size else voided
    lo, hi = (float(value) for value in np.percentile(source, [0.5, 99.5]))
    if not hi > lo:
        return float(source.min()), float(source.max()) + 1.0
    return lo, hi


def brain_crop(volume: np.ndarray, slice_index: int, margin: int = 8) -> tuple[slice, slice]:
    foreground = volume[:, :, slice_index] > 0
    coordinates = np.argwhere(foreground)
    if not len(coordinates):
        return slice(0, volume.shape[0]), slice(0, volume.shape[1])
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, foreground.shape)
    return slice(int(lower[0]), int(upper[0])), slice(int(lower[1]), int(upper[1]))


def oriented_slice(
    volume: np.ndarray, slice_index: int, crop: tuple[slice, slice]
) -> np.ndarray:
    return np.rot90(volume[crop[0], crop[1], slice_index])


def load_case(
    *,
    data_dir: Path,
    case_id: str,
    single_pred_dir: Path,
    ensemble_pred_dir: Path,
) -> dict[str, object]:
    case_dir = find_case_dir(data_dir, case_id)
    paths = {
        "voided": case_dir / f"{case_id}-t1n-voided.nii.gz",
        "ground_truth": case_dir / f"{case_id}-t1n.nii.gz",
        "full_mask": case_dir / f"{case_id}-mask.nii.gz",
        "healthy_mask": case_dir / f"{case_id}-mask-healthy.nii.gz",
        "single": prediction_path(single_pred_dir, case_id),
        "ensemble": prediction_path(ensemble_pred_dir, case_id),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing NIfTI file: {path}")

    arrays = {}
    reference_affine = None
    reference_shape = None
    for name, path in paths.items():
        array, affine = load_canonical(path)
        if reference_shape is None:
            reference_shape = array.shape
            reference_affine = affine
        if array.shape != reference_shape:
            raise ValueError(
                f"Geometry mismatch for {case_id}: {name} shape {array.shape} "
                f"!= {reference_shape}"
            )
        if not np.allclose(affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"Affine mismatch for {case_id}: {name}")
        arrays[name] = array

    arrays["full_mask"] = arrays["full_mask"] > 0
    arrays["healthy_mask"] = arrays["healthy_mask"] > 0
    if not arrays["healthy_mask"].any():
        raise ValueError(f"Healthy mask is empty for {case_id}")
    slice_index = int(
        np.argmax(arrays["healthy_mask"].sum(axis=(0, 1)))
    )
    crop = brain_crop(arrays["ground_truth"], slice_index)
    metric_bounds = normalization_bounds(arrays["voided"])
    ground_truth_norm = normalize(arrays["ground_truth"], metric_bounds)
    errors = {
        name: (
            np.abs(normalize(arrays[name], metric_bounds) - ground_truth_norm)
            * arrays["healthy_mask"]
        )
        for name in ("single", "ensemble")
    }
    return {
        "case_id": case_id,
        "paths": paths,
        "arrays": arrays,
        "errors": errors,
        "slice_index": slice_index,
        "crop": crop,
        "display_window": display_window(arrays["voided"]),
        "metric_bounds": metric_bounds,
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.2,
            "axes.titlesize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def metric_text(values: dict[str, float]) -> str:
    return (
        f"SSIM {values['ssim']:.3f}\n"
        f"PSNR {values['psnr']:.1f}\n"
        f"MSE {values['mse']:.4f}"
    )


def draw_mask_contours(
    axis: plt.Axes,
    full_mask: np.ndarray,
    healthy_mask: np.ndarray,
    slice_index: int,
    crop: tuple[slice, slice],
) -> None:
    full_slice = oriented_slice(full_mask, slice_index, crop)
    healthy_slice = oriented_slice(healthy_mask, slice_index, crop)
    if full_slice.any() and not full_slice.all():
        axis.contour(
            full_slice,
            levels=[0.5],
            colors=["#00D5E5"],
            linewidths=0.7,
        )
    if healthy_slice.any() and not healthy_slice.all():
        axis.contour(
            healthy_slice,
            levels=[0.5],
            colors=["#FFD166"],
            linewidths=0.65,
            linestyles="--",
        )


def plot_figure(
    *,
    cases: list[dict[str, object]],
    selections: list[dict[str, object]],
    single_metrics: dict[str, dict[str, float]],
    ensemble_metrics: dict[str, dict[str, float]],
    ensemble_label: str,
    output_stem: Path,
) -> tuple[list[Path], float]:
    configure_style()
    pooled_errors = np.concatenate(
        [
            case["errors"][name][case["arrays"]["healthy_mask"]]
            for case in cases
            for name in ("single", "ensemble")
        ]
    )
    error_max = max(float(np.percentile(pooled_errors, 99)), 1e-6)
    titles = (
        "Voided input",
        "Single (N=1)",
        ensemble_label,
        "Ground truth",
        "Error (N=1)",
        "Error (ensemble)",
    )
    figure = plt.figure(figsize=(7.2, 4.55))
    grid = figure.add_gridspec(
        3,
        7,
        width_ratios=(1, 1, 1, 1, 1, 1, 0.045),
        left=0.115,
        right=0.95,
        bottom=0.12,
        top=0.91,
        wspace=0.055,
        hspace=0.14,
    )
    axes = np.asarray(
        [
            [figure.add_subplot(grid[row, column]) for column in range(6)]
            for row in range(3)
        ]
    )
    colorbar_axis = figure.add_subplot(grid[:, 6])

    error_image = None
    for row_index, (case, selection) in enumerate(zip(cases, selections)):
        arrays = case["arrays"]
        crop = case["crop"]
        slice_index = int(case["slice_index"])
        vmin, vmax = case["display_window"]
        image_volumes = (
            arrays["voided"],
            arrays["single"],
            arrays["ensemble"],
            arrays["ground_truth"],
        )
        for column, volume in enumerate(image_volumes):
            axis = axes[row_index, column]
            axis.imshow(
                oriented_slice(volume, slice_index, crop),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            draw_mask_contours(
                axis,
                arrays["full_mask"],
                arrays["healthy_mask"],
                slice_index,
                crop,
            )
        for column, name in ((4, "single"), (5, "ensemble")):
            axis = axes[row_index, column]
            error_image = axis.imshow(
                oriented_slice(case["errors"][name], slice_index, crop),
                cmap="magma",
                vmin=0.0,
                vmax=error_max,
                interpolation="nearest",
            )
            draw_mask_contours(
                axis,
                arrays["full_mask"],
                arrays["healthy_mask"],
                slice_index,
                crop,
            )

        case_id = str(case["case_id"])
        axes[row_index, 0].set_ylabel(
            f"{selection['quantile_label']}\n{case_id}\nslice {slice_index}",
            fontsize=6.4,
            labelpad=3,
        )
        axes[row_index, 1].text(
            0.035,
            0.035,
            metric_text(single_metrics[case_id]),
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=4.6,
            linespacing=1.05,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.68,
            },
        )
        axes[row_index, 2].text(
            0.035,
            0.035,
            metric_text(ensemble_metrics[case_id]),
            transform=axes[row_index, 2].transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=4.6,
            linespacing=1.05,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.68,
            },
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    for axis, title in zip(axes[0], titles):
        axis.set_title(title, fontweight="bold", pad=4)

    handles = [
        Line2D([0], [0], color="#00D5E5", linewidth=1.2, label="Full hole"),
        Line2D(
            [0],
            [0],
            color="#FFD166",
            linewidth=1.2,
            linestyle="--",
            label="Scored healthy region",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.485, 0.025),
        ncol=2,
        frameon=False,
        fontsize=6.5,
    )
    if error_image is None:
        raise RuntimeError("No error image was drawn")
    colorbar = figure.colorbar(
        error_image,
        cax=colorbar_axis,
    )
    colorbar.set_label("Absolute error (official normalization)", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=6.2)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=240, bbox_inches="tight")
    plt.close(figure)
    return outputs, error_max


def ensure_output_targets(output_stem: Path, overwrite: bool) -> list[Path]:
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
    return targets


def write_selection_csv(
    *,
    path: Path,
    selections: list[dict[str, object]],
    single_metrics: dict[str, dict[str, float]],
    ensemble_metrics: dict[str, dict[str, float]],
) -> None:
    fieldnames = [
        "quantile",
        "quantile_label",
        "rank_index",
        "case",
        "single_ssim",
        "single_psnr",
        "single_mse",
        "ensemble_ssim",
        "ensemble_psnr",
        "ensemble_mse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for selection in selections:
            case_id = str(selection["case_id"])
            writer.writerow(
                {
                    "quantile": selection["quantile"],
                    "quantile_label": selection["quantile_label"],
                    "rank_index": selection["rank_index"],
                    "case": case_id,
                    "single_ssim": single_metrics[case_id]["ssim"],
                    "single_psnr": single_metrics[case_id]["psnr"],
                    "single_mse": single_metrics[case_id]["mse"],
                    "ensemble_ssim": ensemble_metrics[case_id]["ssim"],
                    "ensemble_psnr": ensemble_metrics[case_id]["psnr"],
                    "ensemble_mse": ensemble_metrics[case_id]["mse"],
                }
            )


def write_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    selections: list[dict[str, object]],
    cases: list[dict[str, object]],
    outputs: list[Path],
    selection_csv: Path,
    error_max: float,
) -> None:
    records = []
    for selection, case in zip(selections, cases):
        records.append(
            {
                **selection,
                "slice_index": case["slice_index"],
                "display_window": list(case["display_window"]),
                "official_metric_normalization_bounds": list(
                    case["metric_bounds"]
                ),
                "inputs": {
                    name: {
                        "path": str(source),
                        "sha256": sha256(source),
                    }
                    for name, source in case["paths"].items()
                },
            }
        )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "selection_rule": (
            "Nearest observed cases to the 10th, 50th, and 90th percentiles "
            "of frozen-ensemble SSIM within the locked confirmation cohort."
        ),
        "slice_rule": "Maximum healthy-mask area in canonical RAS axial space.",
        "display_window_rule": (
            "Per-case 0.5th and 99.5th percentiles of positive voided voxels."
        ),
        "error_rule": (
            "Absolute error inside the healthy mask after normalization by "
            "the voided-image 0.5th and 99.5th percentiles."
        ),
        "error_color_limit": [0.0, error_max],
        "ensemble_label": args.ensemble_label,
        "confirmation_case_file": str(args.case_file),
        "confirmation_case_file_sha256": sha256(args.case_file),
        "single_metrics_csv": str(args.single_metrics),
        "single_metrics_csv_sha256": sha256(args.single_metrics),
        "ensemble_metrics_csv": str(args.ensemble_metrics),
        "ensemble_metrics_csv_sha256": sha256(args.ensemble_metrics),
        "selected_cases": records,
        "selection_csv": str(selection_csv),
        "selection_csv_sha256": sha256(selection_csv),
        "outputs": [
            {"path": str(output), "sha256": sha256(output)}
            for output in outputs
        ],
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    for name in (
        "single_metrics",
        "ensemble_metrics",
        "single_pred_dir",
        "ensemble_pred_dir",
        "data_dir",
        "case_file",
        "output_stem",
    ):
        value = getattr(args, name)
        setattr(args, name, value.expanduser().resolve())

    ensure_output_targets(args.output_stem, args.overwrite)
    case_ids = read_case_file(args.case_file)
    single_metrics = read_metrics(args.single_metrics)
    ensemble_metrics = read_metrics(args.ensemble_metrics)
    require_exact_cases(single_metrics, case_ids, args.single_metrics)
    require_exact_cases(ensemble_metrics, case_ids, args.ensemble_metrics)
    selections = select_cases(ensemble_metrics, case_ids)
    cases = [
        load_case(
            data_dir=args.data_dir,
            case_id=str(selection["case_id"]),
            single_pred_dir=args.single_pred_dir,
            ensemble_pred_dir=args.ensemble_pred_dir,
        )
        for selection in selections
    ]
    outputs, error_max = plot_figure(
        cases=cases,
        selections=selections,
        single_metrics=single_metrics,
        ensemble_metrics=ensemble_metrics,
        ensemble_label=args.ensemble_label,
        output_stem=args.output_stem,
    )
    selection_csv = args.output_stem.with_suffix(".csv")
    write_selection_csv(
        path=selection_csv,
        selections=selections,
        single_metrics=single_metrics,
        ensemble_metrics=ensemble_metrics,
    )
    manifest = args.output_stem.with_suffix(".manifest.json")
    write_manifest(
        path=manifest,
        args=args,
        selections=selections,
        cases=cases,
        outputs=outputs,
        selection_csv=selection_csv,
        error_max=error_max,
    )
    print(
        *(str(path) for path in [*outputs, selection_csv, manifest]),
        sep="\n",
    )


if __name__ == "__main__":
    main()
