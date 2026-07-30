#!/usr/bin/env python3
"""Create the audited qualitative figure for the locked confirmation cohort.

Cases are selected deterministically at the 10th, 50th, and 90th percentiles of
the selected weighted-mixture pipeline's SSIM. Each row uses a near-maximum-area
axial slice that favors a larger scored-region share of the full hole.
"""

from __future__ import annotations

import argparse
import csv
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
METRIC_ANNOTATION_MARGIN = 0.035
METRIC_ANNOTATION_MASK_DILATION = 2
METRIC_ANNOTATION_CORNERS = (
    ("lower left", "left", "bottom"),
    ("lower right", "right", "bottom"),
    ("upper left", "left", "top"),
    ("upper right", "right", "top"),
)
ZOOM_MINIMUM_CROP_PIXELS = 48
SLICE_MINIMUM_RELATIVE_HEALTHY_AREA = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--case-file", type=Path, default=DEFAULT_CASE_FILE)
    parser.add_argument(
        "--reconstruction-label",
        default="Weighted mixture\nmean ($N=5$)",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=PAPER_ROOT / "figures" / "qualitative_reconstructions",
    )
    parser.add_argument(
        "--zoom-layout",
        choices=("bottom", "column"),
        default="bottom",
        help="Place paired reconstruction/reference zooms below the grid or in column 4.",
    )
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


def read_case_file(path: Path) -> list[str]:
    case_ids = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(case_ids) != 75 or len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Expected 75 unique case IDs in {path}")
    return case_ids


def read_metrics(path: Path) -> dict[str, dict[str, float]]:
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


def select_cases(
    rows: dict[str, dict[str, float]],
    case_ids: list[str],
) -> list[dict[str, object]]:
    if set(rows) != set(case_ids):
        raise ValueError(
            f"Metric/manifest mismatch: missing={sorted(set(case_ids) - set(rows))[:5]} "
            f"unexpected={sorted(set(rows) - set(case_ids))[:5]}"
        )
    ordered = sorted(case_ids, key=lambda case_id: (rows[case_id]["ssim"], case_id))
    selections = []
    indices = set()
    for quantile, label in QUANTILES:
        index = int(math.floor(quantile * (len(ordered) - 1)))
        if index in indices:
            raise ValueError("Quantile rule selected a duplicate rank")
        indices.add(index)
        selections.append(
            {
                "quantile": quantile,
                "quantile_label": label,
                "rank_index_zero_based": index,
                "rank_one_based": index + 1,
                "case_id": ordered[index],
                "selection_basis": "lower empirical cohort SSIM percentile",
            }
        )

    return selections


def find_case_dir(data_dir: Path, case_id: str) -> Path:
    direct = data_dir / case_id
    if direct.is_dir():
        return direct
    matches = [path for path in data_dir.rglob(case_id) if path.is_dir()]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source directory for {case_id}, found {len(matches)}"
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


def normalization_bounds(voided: np.ndarray) -> tuple[float, float]:
    lo, hi = (float(value) for value in np.percentile(voided, [0.5, 99.5]))
    lo = max(0.0, lo)
    if not hi > lo:
        lo, hi = 0.0, float(voided.max())
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


def select_slice(
    healthy_mask: np.ndarray,
    full_mask: np.ndarray,
) -> int:
    healthy_area = healthy_mask.sum(axis=(0, 1))
    full_area = full_mask.sum(axis=(0, 1))
    minimum_area = SLICE_MINIMUM_RELATIVE_HEALTHY_AREA * healthy_area.max()
    candidates = np.flatnonzero(healthy_area >= minimum_area)
    return int(
        max(
            candidates,
            key=lambda index: (
                healthy_area[index] / full_area[index],
                healthy_area[index],
                -index,
            ),
        )
    )


def brain_crop(
    volume: np.ndarray, slice_index: int, margin: int = 8
) -> tuple[slice, slice]:
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
    data_dir: Path, pred_dir: Path, case_id: str
) -> dict[str, object]:
    case_dir = find_case_dir(data_dir, case_id)
    paths = {
        "voided": case_dir / f"{case_id}-t1n-voided.nii.gz",
        "prediction": prediction_path(pred_dir, case_id),
        "reference_t1n": case_dir / f"{case_id}-t1n.nii.gz",
        "full_mask": case_dir / f"{case_id}-mask.nii.gz",
        "healthy_mask": case_dir / f"{case_id}-mask-healthy.nii.gz",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    arrays = {}
    reference_shape = None
    reference_affine = None
    for name, path in paths.items():
        array, affine = load_canonical(path)
        if reference_shape is None:
            reference_shape = array.shape
            reference_affine = affine
        if array.shape != reference_shape:
            raise ValueError(
                f"{case_id}: {name} shape {array.shape} != {reference_shape}"
            )
        if not np.allclose(affine, reference_affine, rtol=0.0, atol=1e-5):
            raise ValueError(f"{case_id}: {name} affine mismatch")
        if not np.isfinite(array).all():
            raise ValueError(f"{case_id}: {name} contains non-finite values")
        arrays[name] = array

    arrays["full_mask"] = arrays["full_mask"] > 0
    arrays["healthy_mask"] = arrays["healthy_mask"] > 0
    if not arrays["healthy_mask"].any():
        raise ValueError(f"{case_id}: healthy mask is empty")
    if np.any(arrays["healthy_mask"] & ~arrays["full_mask"]):
        raise ValueError(f"{case_id}: healthy mask is not contained in full mask")
    if not np.array_equal(
        arrays["prediction"][~arrays["full_mask"]],
        arrays["voided"][~arrays["full_mask"]],
    ):
        raise ValueError(f"{case_id}: prediction changed voxels outside the full mask")

    slice_index = select_slice(arrays["healthy_mask"], arrays["full_mask"])
    crop = brain_crop(arrays["reference_t1n"], slice_index)
    metric_bounds = normalization_bounds(arrays["voided"])
    error = (
        np.abs(
            normalize(arrays["prediction"], metric_bounds)
            - normalize(arrays["reference_t1n"], metric_bounds)
        )
        * arrays["healthy_mask"]
    )
    return {
        "case_id": case_id,
        "paths": paths,
        "arrays": arrays,
        "error": error,
        "slice_index": slice_index,
        "crop": crop,
        "display_window": display_window(arrays["voided"]),
        "metric_bounds": metric_bounds,
    }


def square_padding(
    shape: tuple[int, int], canvas_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(shape) != 2 or max(shape) > canvas_size:
        raise ValueError(
            f"Cannot pad two-dimensional shape {shape} to {canvas_size} square"
        )
    row_padding = canvas_size - shape[0]
    column_padding = canvas_size - shape[1]
    return (
        (row_padding // 2, row_padding - row_padding // 2),
        (column_padding // 2, column_padding - column_padding // 2),
    )


def zero_pad_square(array: np.ndarray, canvas_size: int) -> np.ndarray:
    return np.pad(
        array,
        square_padding(array.shape, canvas_size),
        mode="constant",
        constant_values=0,
    )


def draw_mask_contours(
    axis: plt.Axes,
    case: dict[str, object],
    canvas_size: int | None = None,
) -> None:
    arrays = case["arrays"]
    full_slice = oriented_slice(
        arrays["full_mask"], case["slice_index"], case["crop"]
    )
    healthy_slice = oriented_slice(
        arrays["healthy_mask"], case["slice_index"], case["crop"]
    )
    if canvas_size is not None:
        full_slice = zero_pad_square(full_slice, canvas_size)
        healthy_slice = zero_pad_square(healthy_slice, canvas_size)
    if full_slice.any() and not full_slice.all():
        axis.contour(full_slice, [0.5], colors=["#00D5E5"], linewidths=0.85)
    if healthy_slice.any() and not healthy_slice.all():
        axis.contour(
            healthy_slice,
            [0.5],
            colors=["#FFD166"],
            linewidths=1.0,
            linestyles="--",
        )


def metric_text(values: dict[str, float]) -> str:
    return (
        f"SSIM {values['ssim']:.3f}\n"
        f"PSNR {values['psnr']:.1f} dB\n"
        f"MSE {values['mse']:.4f}"
    )


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    mask = mask.astype(bool, copy=False)
    padded = np.pad(mask, radius)
    dilated = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            dilated |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return dilated


def mask_overlap_for_axes_bounds(
    case: dict[str, object],
    bounds: tuple[float, float, float, float],
) -> int:
    arrays = case["arrays"]
    full_slice = oriented_slice(
        arrays["full_mask"], case["slice_index"], case["crop"]
    ).astype(bool)
    main_panel_layout = case.get("main_panel_layout")
    if main_panel_layout is not None:
        full_slice = zero_pad_square(
            full_slice, int(main_panel_layout["canvas_size_pixels"])
        )
    # Matplotlib's default imshow origin is upper, whereas transAxes is bottom-up.
    axes_mask = np.flipud(
        dilate_binary_mask(full_slice, METRIC_ANNOTATION_MASK_DILATION)
    )
    height, width = axes_mask.shape
    x0, y0, x1, y1 = bounds
    column_start = max(0, min(width, int(math.floor(x0 * width))))
    column_stop = max(0, min(width, int(math.ceil(x1 * width))))
    row_start = max(0, min(height, int(math.floor(y0 * height))))
    row_stop = max(0, min(height, int(math.ceil(y1 * height))))
    return int(
        axes_mask[row_start:row_stop, column_start:column_stop].sum()
    )


def place_metric_annotation(
    axis: plt.Axes,
    artist: matplotlib.text.Text,
    case: dict[str, object],
    renderer: object,
) -> None:
    patch_bounds = (
        artist.get_bbox_patch()
        .get_window_extent(renderer=renderer)
        .transformed(axis.transAxes.inverted())
    )
    width = float(patch_bounds.width)
    height = float(patch_bounds.height)
    margin = METRIC_ANNOTATION_MARGIN
    candidates = []
    for corner, horizontal_alignment, vertical_alignment in (
        METRIC_ANNOTATION_CORNERS
    ):
        x = margin if horizontal_alignment == "left" else 1.0 - margin
        y = margin if vertical_alignment == "bottom" else 1.0 - margin
        x0 = x if horizontal_alignment == "left" else x - width
        x1 = x + width if horizontal_alignment == "left" else x
        y0 = y if vertical_alignment == "bottom" else y - height
        y1 = y + height if vertical_alignment == "bottom" else y
        overlap = mask_overlap_for_axes_bounds(case, (x0, y0, x1, y1))
        candidates.append(
            {
                "corner": corner,
                "horizontal_alignment": horizontal_alignment,
                "vertical_alignment": vertical_alignment,
                "position": (x, y),
                "axes_bounds": (x0, y0, x1, y1),
                "full_mask_overlap_pixels": overlap,
            }
        )

    selected = min(
        enumerate(candidates),
        key=lambda item: (item[1]["full_mask_overlap_pixels"], item[0]),
    )[1]
    artist.set_position(selected["position"])
    artist.set_ha(selected["horizontal_alignment"])
    artist.set_va(selected["vertical_alignment"])
    case["metric_annotation"] = {
        "corner": selected["corner"],
        "axes_bounds": list(selected["axes_bounds"]),
        "full_mask_overlap_pixels": selected["full_mask_overlap_pixels"],
        "candidate_overlap_pixels": {
            candidate["corner"]: candidate["full_mask_overlap_pixels"]
            for candidate in candidates
        },
    }


def centered_interval(center: float, size: int, limit: int) -> tuple[int, int]:
    size = min(size, limit)
    start = int(round(center - size / 2))
    start = max(0, min(start, limit - size))
    return start, start + size


def healthy_region_zoom_crop(
    case: dict[str, object],
) -> tuple[int, int, int, int]:
    arrays = case["arrays"]
    healthy_slice = oriented_slice(
        arrays["healthy_mask"], case["slice_index"], case["crop"]
    ).astype(bool)
    coordinates = np.argwhere(healthy_slice)
    if not len(coordinates):
        raise ValueError(f"{case['case_id']}: empty healthy mask on selected slice")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    mask_span = int(np.max(upper - lower + 1))
    crop_size = max(
        ZOOM_MINIMUM_CROP_PIXELS,
        int(math.ceil(mask_span * 1.35)),
    )
    center = (lower + upper) / 2
    row_start, row_stop = centered_interval(
        float(center[0]), crop_size, healthy_slice.shape[0]
    )
    column_start, column_stop = centered_interval(
        float(center[1]), crop_size, healthy_slice.shape[1]
    )
    return row_start, row_stop, column_start, column_stop


def plot_scored_region_zoom(
    axis: plt.Axes,
    case: dict[str, object],
    image_name: str,
) -> None:
    arrays = case["arrays"]
    slice_index = int(case["slice_index"])
    crop = case["crop"]
    vmin, vmax = case["display_window"]
    image_slice = oriented_slice(arrays[image_name], slice_index, crop)
    row_start, row_stop, column_start, column_stop = healthy_region_zoom_crop(case)
    axis.imshow(
        image_slice,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    draw_mask_contours(axis, case)
    axis.set_xlim(column_start - 0.5, column_stop - 0.5)
    axis.set_ylim(row_stop - 0.5, row_start - 0.5)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_facecolor("black")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#6F6F6F")
        spine.set_linewidth(0.6)

    case.setdefault("zoom_panels", []).append(
        {
            "panel": image_name,
            "oriented_pixel_crop": [
                row_start,
                row_stop,
                column_start,
                column_stop,
            ],
        }
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.3,
            "axes.titlesize": 8.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(
    cases: list[dict[str, object]],
    selections: list[dict[str, object]],
    metrics: dict[str, dict[str, float]],
    reconstruction_label: str,
    output_stem: Path,
    zoom_layout: str,
) -> tuple[list[Path], float]:
    configure_style()
    pooled_errors = np.concatenate(
        [
            case["error"][case["arrays"]["healthy_mask"]]
            for case in cases
        ]
    )
    error_max = max(float(np.percentile(pooled_errors, 99)), 1e-6)
    error_cmap = plt.get_cmap("magma").copy()
    error_cmap.set_bad("#202020")

    main_shapes = [
        oriented_slice(
            case["arrays"]["voided"], case["slice_index"], case["crop"]
        ).shape
        for case in cases
    ]
    main_canvas_size = max(max(shape) for shape in main_shapes)
    for case, shape in zip(cases, main_shapes):
        padding = square_padding(shape, main_canvas_size)
        case["main_panel_layout"] = {
            "oriented_crop_shape_pixels": list(shape),
            "canvas_size_pixels": main_canvas_size,
            "zero_padding_pixels": {
                "top": padding[0][0],
                "bottom": padding[0][1],
                "left": padding[1][0],
                "right": padding[1][1],
            },
        }

    if zoom_layout == "bottom":
        figure = plt.figure(figsize=(7.2, 5.35))
        grid = figure.add_gridspec(
            3,
            5,
            width_ratios=(1, 1, 1, 1, 0.045),
            left=0.13,
            right=0.95,
            bottom=0.32,
            top=0.93,
            wspace=0.055,
            hspace=0.13,
        )
        axes = np.asarray(
            [
                [figure.add_subplot(grid[row, column]) for column in range(4)]
                for row in range(3)
            ]
        )
        colorbar_axis = figure.add_subplot(grid[:, 4])
        zoom_grid = figure.add_gridspec(
            1,
            8,
            width_ratios=(1, 1, 0.16, 1, 1, 0.16, 1, 1),
            left=0.20,
            right=0.88,
            bottom=0.055,
            top=0.215,
            wspace=0.055,
        )
        zoom_axes = [
            (
                figure.add_subplot(zoom_grid[0, first_column]),
                figure.add_subplot(zoom_grid[0, first_column + 1]),
            )
            for first_column in (0, 3, 6)
        ]
        legend_anchor = (0.49, 0.255)
    else:
        figure = plt.figure(figsize=(7.2, 3.75))
        grid = figure.add_gridspec(
            3,
            6,
            width_ratios=(1, 1, 1, 2, 1, 0.045),
            left=0.13,
            right=0.95,
            bottom=0.16,
            top=0.91,
            wspace=0.025,
            hspace=0.10,
        )
        axes = np.asarray(
            [
                [
                    figure.add_subplot(grid[row, column])
                    for column in (0, 1, 2, 4)
                ]
                for row in range(3)
            ]
        )
        colorbar_axis = figure.add_subplot(grid[:, 5])
        zoom_axes = []
        for row in range(3):
            zoom_grid = grid[row, 3].subgridspec(1, 2, wspace=0.025)
            zoom_axes.append(
                (
                    figure.add_subplot(zoom_grid[0, 0]),
                    figure.add_subplot(zoom_grid[0, 1]),
                )
            )
        legend_anchor = (0.5, 0.045)
    error_image = None
    metric_annotations = []
    for row_index, (case, selection) in enumerate(zip(cases, selections)):
        arrays = case["arrays"]
        slice_index = int(case["slice_index"])
        crop = case["crop"]
        vmin, vmax = case["display_window"]
        for column, name in enumerate(("voided", "prediction", "reference_t1n")):
            axes[row_index, column].imshow(
                zero_pad_square(
                    oriented_slice(arrays[name], slice_index, crop),
                    main_canvas_size,
                ),
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            draw_mask_contours(
                axes[row_index, column], case, canvas_size=main_canvas_size
            )

        error_slice = zero_pad_square(
            oriented_slice(case["error"], slice_index, crop),
            main_canvas_size,
        )
        healthy_slice = zero_pad_square(
            oriented_slice(arrays["healthy_mask"], slice_index, crop).astype(bool),
            main_canvas_size,
        )
        error_image = axes[row_index, 3].imshow(
            np.ma.masked_where(~healthy_slice, error_slice),
            cmap=error_cmap,
            vmin=0.0,
            vmax=error_max,
            interpolation="nearest",
        )
        draw_mask_contours(
            axes[row_index, 3], case, canvas_size=main_canvas_size
        )

        case_id = str(case["case_id"])
        row_label = str(selection["quantile_label"])
        axes[row_index, 0].set_ylabel(
            f"{row_label}\n{case_id.removeprefix('BraTS-GLI-')}",
            fontsize=6.2,
            linespacing=0.95,
            labelpad=2,
        )
        metric_annotation = axes[row_index, 1].text(
            METRIC_ANNOTATION_MARGIN,
            METRIC_ANNOTATION_MARGIN,
            metric_text(metrics[case_id]),
            transform=axes[row_index, 1].transAxes,
            ha="left",
            va="bottom",
            color="white",
            fontsize=5.8,
            linespacing=1.08,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.68,
            },
        )
        metric_annotations.append(
            (axes[row_index, 1], metric_annotation, case)
        )
        for axis in axes[row_index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    if zoom_layout == "bottom":
        column_titles = (
            "Voided input",
            reconstruction_label,
            "Reference T1n",
            r"$|\mathrm{pred}-\mathrm{ref}|$",
        )
        title_style = {"fontweight": "bold", "pad": 4}
    else:
        column_titles = (
            "Voided input",
            "Prediction",
            "Observed T1n",
            "Absolute error",
        )
        title_style = {"fontsize": 6.8, "fontweight": "normal", "pad": 2.5}
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, **title_style)

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
        bbox_to_anchor=legend_anchor,
        ncol=2,
        frameon=False,
        fontsize=6.7,
    )
    if error_image is None:
        raise RuntimeError("No error image was drawn")
    colorbar = figure.colorbar(error_image, cax=colorbar_axis)
    colorbar.set_label("Absolute error (official normalization)", fontsize=6.4)
    colorbar.ax.tick_params(labelsize=6.2)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis, artist, case in metric_annotations:
        place_metric_annotation(axis, artist, case, renderer)
    for row_index, (case, selection) in enumerate(zip(cases, selections)):
        prediction_zoom, reference_zoom = zoom_axes[row_index]
        plot_scored_region_zoom(prediction_zoom, case, "prediction")
        plot_scored_region_zoom(reference_zoom, case, "reference_t1n")
        if zoom_layout == "bottom":
            prediction_zoom.set_title("Weighted", fontsize=5.8, pad=1.5)
            reference_zoom.set_title("Reference", fontsize=5.8, pad=1.5)
            group_bounds = prediction_zoom.get_position()
            reference_bounds = reference_zoom.get_position()
            figure.text(
                (group_bounds.x0 + reference_bounds.x1) / 2,
                0.028,
                ("Low", "Median", "High")[row_index],
                ha="center",
                va="center",
                fontsize=6.1,
                fontweight="bold",
            )
        elif row_index == 0:
            prediction_zoom.set_title(
                "Prediction zoom", fontsize=6.8, fontweight="normal", pad=2.5
            )
            reference_zoom.set_title(
                "Observed zoom", fontsize=6.8, fontweight="normal", pad=2.5
            )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    figure.savefig(
        outputs[0],
        dpi=300,
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(outputs[1], dpi=240, bbox_inches="tight")
    plt.close(figure)
    return outputs, error_max


def ensure_output_targets(output_stem: Path, overwrite: bool) -> None:
    targets = [
        output_stem.with_suffix(".pdf"),
        output_stem.with_suffix(".png"),
        output_stem.with_suffix(".csv"),
        output_stem.with_suffix(".manifest.json"),
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )


def write_selection_csv(
    path: Path,
    selections: list[dict[str, object]],
    metrics: dict[str, dict[str, float]],
    cases: list[dict[str, object]],
) -> None:
    fieldnames = [
        "quantile",
        "quantile_label",
        "rank_index_zero_based",
        "rank_one_based",
        "case",
        "slice_index_canonical_ras",
        "ssim",
        "psnr",
        "mse",
        "healthy_mask_slice_voxels",
        "full_mask_slice_voxels",
        "healthy_to_full_mask_slice_fraction",
        "selection_basis",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for selection, case in zip(selections, cases):
            case_id = str(selection["case_id"])
            writer.writerow(
                {
                    "quantile": selection["quantile"],
                    "quantile_label": selection["quantile_label"],
                    "rank_index_zero_based": selection["rank_index_zero_based"],
                    "rank_one_based": selection["rank_one_based"],
                    "case": case_id,
                    "slice_index_canonical_ras": case["slice_index"],
                    "healthy_mask_slice_voxels": int(
                        case["arrays"]["healthy_mask"][
                            :, :, int(case["slice_index"])
                        ].sum()
                    ),
                    "full_mask_slice_voxels": int(
                        case["arrays"]["full_mask"][
                            :, :, int(case["slice_index"])
                        ].sum()
                    ),
                    "healthy_to_full_mask_slice_fraction": float(
                        case["arrays"]["healthy_mask"][
                            :, :, int(case["slice_index"])
                        ].sum()
                        / case["arrays"]["full_mask"][
                            :, :, int(case["slice_index"])
                        ].sum()
                    ),
                    "selection_basis": selection["selection_basis"],
                    **metrics[case_id],
                }
            )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    selections: list[dict[str, object]],
    cases: list[dict[str, object]],
    outputs: list[Path],
    selection_csv: Path,
    error_max: float,
) -> None:
    selected_cases = []
    for selection, case in zip(selections, cases):
        healthy_mask_slice_voxels = int(
            case["arrays"]["healthy_mask"][
                :, :, int(case["slice_index"])
            ].sum()
        )
        full_mask_slice_voxels = int(
            case["arrays"]["full_mask"][
                :, :, int(case["slice_index"])
            ].sum()
        )
        selected_cases.append(
            {
                **selection,
                "slice_index_canonical_ras": case["slice_index"],
                "healthy_mask_slice_voxels": healthy_mask_slice_voxels,
                "full_mask_slice_voxels": full_mask_slice_voxels,
                "healthy_to_full_mask_slice_fraction": (
                    healthy_mask_slice_voxels / full_mask_slice_voxels
                ),
                "display_window": list(case["display_window"]),
                "official_metric_normalization_bounds": list(case["metric_bounds"]),
                "metric_annotation": case["metric_annotation"],
                "main_panel_layout": case["main_panel_layout"],
                "zoom_panels": case["zoom_panels"],
                "inputs": {
                    name: {"path": portable_path(source), "sha256": sha256(source)}
                    for name, source in case["paths"].items()
                },
            }
        )
    payload = {
        "schema_version": 6,
        "generator": {
            "path": portable_path(Path(__file__)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "pipeline": args.reconstruction_label.replace("\n", " "),
        "selection_rule": (
            "Observed cases at the 10th, 50th, and 90th percentiles of "
            "weighted-mixture mean-N=5 SSIM in the locked confirmation cohort, "
            "using the lower empirical quantile convention."
        ),
        "slice_rule": (
            "In canonical RAS axial space, retain slices with at least 90% of "
            "the case's maximum healthy-mask slice area, then maximize the "
            "healthy-mask/full-hole area ratio; ties prefer larger healthy-mask "
            "area and then the lower slice index."
        ),
        "display_window_rule": (
            "Per-case 0.5th and 99.5th percentiles of positive voided voxels."
        ),
        "error_rule": (
            "Absolute prediction-reference error after official voided-image "
            "0.5th/99.5th percentile normalization, displayed only within "
            "the provided healthy scoring mask."
        ),
        "error_color_limit": [0.0, error_max],
        "metric_annotation_rule": (
            "Place the in-panel metric block in the first of four corners with "
            "minimum overlap against the plotted full-hole mask dilated by two "
            "image pixels; ties prefer lower left, lower right, upper left, then "
            "upper right."
        ),
        "main_panel_layout_rule": (
            "After the case-specific brain crop and orientation, all main panels "
            "are centered without resampling on one shared square canvas whose "
            "side is the largest selected crop dimension; added pixels are zero."
        ),
        "zoom_layout": args.zoom_layout,
        "zoom_panel_rule": (
            "Paired weighted-reconstruction and reference crops use the same "
            "case-specific intensity window and are placed "
            + (
                "in a bottom strip. "
                if args.zoom_layout == "bottom"
                else "side by side in the fourth grouped column. "
            )
            + "Each square crop is centered on the selected-slice healthy scoring "
            "mask with 35 percent padding and a minimum side length of 48 oriented "
            "pixels."
        ),
        "confirmation_case_file": portable_path(args.case_file),
        "confirmation_case_file_sha256": sha256(args.case_file),
        "metrics_csv": portable_path(args.metrics),
        "metrics_csv_sha256": sha256(args.metrics),
        "selected_cases": selected_cases,
        "selection_csv": portable_path(selection_csv),
        "selection_csv_sha256": sha256(selection_csv),
        "outputs": [
            {"path": portable_path(output), "sha256": sha256(output)}
            for output in outputs
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    for name in ("metrics", "pred_dir", "data_dir", "case_file", "output_stem"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    ensure_output_targets(args.output_stem, args.overwrite)
    case_ids = read_case_file(args.case_file)
    metrics = read_metrics(args.metrics)
    selections = select_cases(metrics, case_ids)
    cases = [
        load_case(args.data_dir, args.pred_dir, str(selection["case_id"]))
        for selection in selections
    ]
    outputs, error_max = plot_figure(
        cases,
        selections,
        metrics,
        args.reconstruction_label,
        args.output_stem,
        args.zoom_layout,
    )
    selection_csv = args.output_stem.with_suffix(".csv")
    write_selection_csv(selection_csv, selections, metrics, cases)
    manifest = args.output_stem.with_suffix(".manifest.json")
    write_manifest(
        manifest,
        args,
        selections,
        cases,
        outputs,
        selection_csv,
        error_max,
    )
    print(*[*outputs, selection_csv, manifest], sep="\n")


if __name__ == "__main__":
    main()
