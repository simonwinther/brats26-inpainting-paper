#!/usr/bin/env python3
"""Audit the production mask samplers and render the manuscript figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
import nibabel as nib
import numpy as np


PAPER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PAPER_ROOT.parent
BRATS_ROOT = WORKSPACE_ROOT / "brats_inpainting"
sys.path.insert(0, str(BRATS_ROOT))

from guided_diffusion.brats_split import build_brats_split  # noqa: E402
from guided_diffusion.masking.bank import TumorMaskBank  # noqa: E402
from guided_diffusion.masking.config import (  # noqa: E402
    load_mask_config,
    mask_config_hash,
)
from guided_diffusion.masking.geometry import brain_mask_from_t1  # noqa: E402
from guided_diffusion.masking.sampler import (  # noqa: E402
    WeightedMaskSampler,
    deterministic_sample_key,
    stable_mask_seed,
)


DEFAULT_DATA_DIR = (
    WORKSPACE_ROOT
    / "data"
    / "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
)
DEFAULT_HOLDOUT = BRATS_ROOT / "splits" / "holdout_val_seed2026_n100.txt"
DEFAULT_BANK = BRATS_ROOT / "artifacts" / "tumor-mask-bank" / "bank_manifest.json"
DEFAULT_RANDOM_CONFIG = BRATS_ROOT / "configs" / "masking" / "random_aug.json"
DEFAULT_WEIGHTED_CONFIG = BRATS_ROOT / "configs" / "masking" / "weighted_aug.json"
DEFAULT_SAMPLES = PAPER_ROOT / "data" / "mask_augmentation_samples.csv"
DEFAULT_METADATA = PAPER_ROOT / "data" / "mask_augmentation_metadata.json"
DEFAULT_FIGURE = PAPER_ROOT / "figures" / "mask_augmentation"

POLICY_ORDER = ("fixed", "random", "weighted")
POLICY_LABELS = {
    "fixed": "Fixed",
    "random": "Random",
    "weighted": "Weighted",
}
POLICY_COLORS = {
    "fixed": "#6B7280",
    "random": "#D97706",
    "weighted": "#0F766E",
}
FAMILY_COLORS = {
    "fixed": "#9CA3AF",
    "tumor": "#0F766E",
    "blob": "#3B82F6",
    "ellipsoid": "#8B5CF6",
    "fallback": "#D1D5DB",
}
SHAPE_CATEGORIES = (
    "Fixed",
    "Random: tumor",
    "Weighted: tumor",
    "Weighted: blob",
    "Weighted: ellipsoid",
    "Weighted: fallback",
)
SHAPE_COLORS = {
    "Fixed": POLICY_COLORS["fixed"],
    "Random: tumor": POLICY_COLORS["random"],
    "Weighted: tumor": FAMILY_COLORS["tumor"],
    "Weighted: blob": FAMILY_COLORS["blob"],
    "Weighted: ellipsoid": FAMILY_COLORS["ellipsoid"],
    "Weighted: fallback": "#111827",
}

CSV_FIELDS = (
    "sample_id",
    "policy",
    "case_id",
    "epoch",
    "repeat_slot",
    "global_seed",
    "stable_seed",
    "sample_key",
    "mask_family",
    "fallback_used",
    "fallback_policy",
    "volume_voxels",
    "mask_to_brain_ratio",
    "extent_x",
    "extent_y",
    "extent_z",
    "elongation",
    "bounding_box_fill_ratio",
    "target_volume_voxels",
    "size_policy",
    "size_bin",
    "background_overlap_ratio",
    "distance_from_tumor_voxels",
    "rejected_attempts",
    "source_mask_id",
    "source_case_id",
    "mask_sha256",
    "generation_metadata_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--holdout-file", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--random-config", type=Path, default=DEFAULT_RANDOM_CONFIG)
    parser.add_argument("--weighted-config", type=Path, default=DEFAULT_WEIGHTED_CONFIG)
    parser.add_argument("--samples-csv", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--audit-cases", type=int, default=100)
    parser.add_argument("--audit-seed", type=int, default=2026)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse the existing audit CSV/metadata and regenerate only the figure.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array(path: Path, dtype=None) -> np.ndarray:
    array = nib.as_closest_canonical(nib.load(path)).get_fdata()
    return array.astype(dtype, copy=False) if dtype is not None else array


def load_case(data_dir: Path, case_id: str) -> dict[str, np.ndarray]:
    case_dir = data_dir / case_id
    arrays = {
        "t1n": canonical_array(case_dir / f"{case_id}-t1n.nii.gz", np.float32),
        "healthy": canonical_array(case_dir / f"{case_id}-mask-healthy.nii.gz") > 0,
        "unhealthy": canonical_array(case_dir / f"{case_id}-mask-unhealthy.nii.gz") > 0,
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"Canonical shape mismatch for {case_id}: {sorted(shapes)}")
    return arrays


def read_case_ids(path: Path) -> list[str]:
    with path.open() as handle:
        values = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate case ID in {path}")
    return values


def mask_features(mask: np.ndarray, brain_mask: np.ndarray) -> dict[str, float | int]:
    mask = np.asarray(mask, dtype=bool)
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("Cannot characterize an empty mask.")
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    extents = upper - lower
    volume = int(len(coordinates))
    brain_volume = max(int(np.count_nonzero(brain_mask)), 1)
    return {
        "volume_voxels": volume,
        "mask_to_brain_ratio": volume / brain_volume,
        "extent_x": int(extents[0]),
        "extent_y": int(extents[1]),
        "extent_z": int(extents[2]),
        "elongation": float(extents.max() / max(int(extents.min()), 1)),
        "bounding_box_fill_ratio": float(volume / int(np.prod(extents))),
    }


def mask_sha256(mask: np.ndarray) -> str:
    array = np.asarray(mask, dtype=bool)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype=np.int32).tobytes())
    digest.update(np.packbits(array.ravel(), bitorder="little").tobytes())
    return digest.hexdigest()


def fixed_record(
    case_id: str,
    healthy_mask: np.ndarray,
    brain_mask: np.ndarray,
) -> dict:
    sample_key = f"{case_id}:legacy-fixed"
    return {
        "sample_id": f"fixed:{case_id}",
        "policy": "fixed",
        "case_id": case_id,
        "epoch": 0,
        "repeat_slot": 0,
        "global_seed": "",
        "stable_seed": "",
        "sample_key": sample_key,
        "mask_family": "fixed",
        "fallback_used": False,
        "fallback_policy": "",
        **mask_features(healthy_mask, brain_mask),
        "target_volume_voxels": "",
        "size_policy": "provided",
        "size_bin": "",
        "background_overlap_ratio": "",
        "distance_from_tumor_voxels": "",
        "rejected_attempts": 0,
        "source_mask_id": "",
        "source_case_id": case_id,
        "mask_sha256": mask_sha256(healthy_mask),
        "generation_metadata_json": json.dumps(
            {
                "case_id": case_id,
                "sample_key": sample_key,
                "mask_family": "fixed",
                "size_policy": "provided",
            },
            sort_keys=True,
        ),
    }


def generated_record(
    policy: str,
    case_id: str,
    epoch: int,
    repeat_slot: int,
    global_seed: int,
    stable_seed: int,
    sample,
    brain_mask: np.ndarray,
) -> dict:
    metadata = sample.metadata.to_dict()
    return {
        "sample_id": f"{policy}:{case_id}:epoch={epoch}:slot={repeat_slot}",
        "policy": policy,
        "case_id": case_id,
        "epoch": epoch,
        "repeat_slot": repeat_slot,
        "global_seed": global_seed,
        "stable_seed": stable_seed,
        "sample_key": metadata["sample_key"],
        "mask_family": metadata["mask_family"],
        "fallback_used": bool(metadata["fallback_used"]),
        "fallback_policy": metadata["fallback_policy"] or "",
        **mask_features(sample.mask, brain_mask),
        "target_volume_voxels": metadata["target_volume_voxels"],
        "size_policy": metadata["size_policy"],
        "size_bin": metadata["size_bin"],
        "background_overlap_ratio": metadata["background_overlap_ratio"],
        "distance_from_tumor_voxels": (
            ""
            if metadata["distance_from_tumor_voxels"] is None
            else metadata["distance_from_tumor_voxels"]
        ),
        "rejected_attempts": metadata["rejected_attempts"],
        "source_mask_id": metadata["source_mask_id"] or "",
        "source_case_id": metadata["source_case_id"] or "",
        "mask_sha256": mask_sha256(sample.mask),
        "generation_metadata_json": json.dumps(metadata, sort_keys=True),
    }


def make_sampler(
    mode: str,
    config_path: Path,
    bank: TumorMaskBank,
    train_case_ids: list[str],
    holdout_case_ids: list[str],
) -> tuple[dict, WeightedMaskSampler]:
    config = load_mask_config(
        mode,
        str(config_path),
        {"families.tumor.bank_path": str(bank.manifest_path)},
    )
    sampler = WeightedMaskSampler(
        config,
        tumor_bank=bank,
        allowed_train_case_ids=train_case_ids,
        forbidden_case_ids=holdout_case_ids,
    )
    return config, sampler


def sample_online(
    sampler: WeightedMaskSampler,
    config: dict,
    case_id: str,
    arrays: dict[str, np.ndarray],
    global_seed: int,
    split_hash: str,
    epoch: int,
    repeat_slot: int,
):
    effective_epoch = epoch if config["seed"]["vary_by_epoch"] else 0
    seed = stable_mask_seed(
        global_seed,
        case_id,
        effective_epoch,
        repeat_slot,
        split_hash=split_hash,
        schema_salt=config["seed"]["schema_salt"],
    )
    sample_key = deterministic_sample_key(case_id, effective_epoch, repeat_slot, seed)
    sample = sampler.sample(
        case_id=case_id,
        t1n=arrays["t1n"],
        real_tumor_mask=arrays["unhealthy"],
        seed=seed,
        sample_key=sample_key,
        legacy_healthy_mask=arrays["healthy"],
    )
    return effective_epoch, seed, sample


def audit_masks(
    args: argparse.Namespace,
    case_ids: list[str],
    split_hash: str,
    random_config: dict,
    random_sampler: WeightedMaskSampler,
    weighted_config: dict,
    weighted_sampler: WeightedMaskSampler,
) -> list[dict]:
    records = []
    for case_index, case_id in enumerate(case_ids, start=1):
        arrays = load_case(args.data_dir, case_id)
        brain_mask = brain_mask_from_t1(
            arrays["t1n"], threshold=0.0, fill_holes=True, nonzero=True
        )
        records.append(fixed_record(case_id, arrays["healthy"], brain_mask))
        for repeat_slot in range(int(random_config["samples_per_case"])):
            epoch, seed, sample = sample_online(
                random_sampler,
                random_config,
                case_id,
                arrays,
                args.global_seed,
                split_hash,
                0,
                repeat_slot,
            )
            records.append(
                generated_record(
                    "random",
                    case_id,
                    epoch,
                    repeat_slot,
                    args.global_seed,
                    seed,
                    sample,
                    brain_mask,
                )
            )
        for repeat_slot in range(int(weighted_config["samples_per_case"])):
            epoch, seed, sample = sample_online(
                weighted_sampler,
                weighted_config,
                case_id,
                arrays,
                args.global_seed,
                split_hash,
                0,
                repeat_slot,
            )
            records.append(
                generated_record(
                    "weighted",
                    case_id,
                    epoch,
                    repeat_slot,
                    args.global_seed,
                    seed,
                    sample,
                    brain_mask,
                )
            )
        if case_index % 10 == 0 or case_index == len(case_ids):
            print(f"Audited {case_index}/{len(case_ids)} cases", flush=True)
    return records


def robust_reference(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        [
            [
                np.log10(float(row["volume_voxels"])),
                np.log(float(row["elongation"])),
                float(row["bounding_box_fill_ratio"]),
            ]
            for row in records
        ],
        dtype=np.float64,
    )
    center = np.median(values, axis=0)
    scale = np.median(np.abs(values - center), axis=0)
    fallback_scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(fallback_scale > 1e-12, fallback_scale, 1.0))
    return center, scale


def representative_score(mask: np.ndarray, brain_mask: np.ndarray, reference) -> float:
    features = mask_features(mask, brain_mask)
    values = np.asarray(
        [
            np.log10(float(features["volume_voxels"])),
            np.log(float(features["elongation"])),
            float(features["bounding_box_fill_ratio"]),
        ]
    )
    center, scale = reference
    return float(np.square((values - center) / scale).sum())


def choose_example_case(records: list[dict]) -> str:
    fixed = [row for row in records if row["policy"] == "fixed"]
    median_ratio = float(np.median([row["mask_to_brain_ratio"] for row in fixed]))
    return min(
        fixed,
        key=lambda row: (abs(float(row["mask_to_brain_ratio"]) - median_ratio), row["case_id"]),
    )["case_id"]


def choose_examples(
    args: argparse.Namespace,
    records: list[dict],
    case_id: str,
    split_hash: str,
    random_config: dict,
    random_sampler: WeightedMaskSampler,
    weighted_config: dict,
    weighted_sampler: WeightedMaskSampler,
) -> list[dict]:
    arrays = load_case(args.data_dir, case_id)
    brain_mask = brain_mask_from_t1(
        arrays["t1n"], threshold=0.0, fill_holes=True, nonzero=True
    )
    references = {}
    references["random"] = robust_reference(
        [row for row in records if row["policy"] == "random"]
    )
    for family in ("tumor", "blob", "ellipsoid"):
        family_rows = [
            row
            for row in records
            if row["policy"] == "weighted"
            and row["mask_family"] == family
            and not row["fallback_used"]
        ]
        if not family_rows:
            raise RuntimeError(f"No non-fallback weighted {family} samples in audit.")
        references[family] = robust_reference(family_rows)

    examples = [
        {
            "label": "Fixed",
            "policy": "fixed",
            "family": "fixed",
            "mask": arrays["healthy"],
            "t1n": arrays["t1n"],
            "unhealthy": arrays["unhealthy"],
            "epoch": 0,
            "repeat_slot": 0,
            "stable_seed": None,
            "sample_key": f"{case_id}:legacy-fixed",
        }
    ]

    random_candidates = []
    for repeat_slot in range(int(random_config["samples_per_case"])):
        epoch, seed, sample = sample_online(
            random_sampler,
            random_config,
            case_id,
            arrays,
            args.global_seed,
            split_hash,
            0,
            repeat_slot,
        )
        random_candidates.append(
            (
                representative_score(sample.mask, brain_mask, references["random"]),
                repeat_slot,
                epoch,
                seed,
                sample,
            )
        )
    _, repeat_slot, epoch, seed, sample = min(random_candidates, key=lambda item: item[:2])
    examples.append(
        {
            "label": "Random",
            "policy": "random",
            "family": "tumor",
            "mask": sample.mask,
            "t1n": arrays["t1n"],
            "unhealthy": arrays["unhealthy"],
            "epoch": epoch,
            "repeat_slot": repeat_slot,
            "stable_seed": seed,
            "sample_key": sample.metadata.sample_key,
        }
    )

    required = {"tumor": 5, "blob": 5, "ellipsoid": 5}
    family_candidates = defaultdict(list)
    for epoch_request in range(200):
        for repeat_slot in range(int(weighted_config["samples_per_case"])):
            epoch, seed, sample = sample_online(
                weighted_sampler,
                weighted_config,
                case_id,
                arrays,
                args.global_seed,
                split_hash,
                epoch_request,
                repeat_slot,
            )
            family = sample.metadata.mask_family
            if sample.metadata.fallback_used or family not in required:
                continue
            if len(family_candidates[family]) < required[family]:
                family_candidates[family].append(
                    (
                        representative_score(sample.mask, brain_mask, references[family]),
                        epoch,
                        repeat_slot,
                        seed,
                        sample,
                    )
                )
        if all(len(family_candidates[name]) >= count for name, count in required.items()):
            break
    missing = [name for name, count in required.items() if len(family_candidates[name]) < count]
    if missing:
        raise RuntimeError(f"Could not collect representative weighted families: {missing}")
    for family in ("tumor", "blob", "ellipsoid"):
        _, epoch, repeat_slot, seed, sample = min(
            family_candidates[family], key=lambda item: item[:3]
        )
        examples.append(
            {
                "label": f"Weighted\n{family}",
                "policy": "weighted",
                "family": family,
                "mask": sample.mask,
                "t1n": arrays["t1n"],
                "unhealthy": arrays["unhealthy"],
                "epoch": epoch,
                "repeat_slot": repeat_slot,
                "stable_seed": seed,
                "sample_key": sample.metadata.sample_key,
            }
        )
    for example in examples:
        example.update(mask_features(example["mask"], brain_mask))
        example["mask_sha256"] = mask_sha256(example["mask"])
    return examples


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 6.7,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.7,
            "legend.fontsize": 5.6,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def axial_slice(mask: np.ndarray) -> int:
    return int(np.argmax(np.asarray(mask, dtype=np.uint8).sum(axis=(0, 1))))


def display_plane(array: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(array[:, :, slice_index])


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=np.float64))
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    return x, y


def shape_category(row: dict) -> str:
    if row["policy"] == "fixed":
        return "Fixed"
    if row["policy"] == "random":
        return "Random: tumor"
    if row["fallback_used"]:
        return "Weighted: fallback"
    return f"Weighted: {row['mask_family']}"


def plot_figure(
    records: list[dict],
    examples: list[dict],
    weighted_config: dict,
    output_stem: Path,
) -> list[Path]:
    configure_style()
    figure = plt.figure(figsize=(4.85, 3.72))
    outer = figure.add_gridspec(2, 1, height_ratios=(1.04, 1.0), hspace=0.29)
    top = outer[0].subgridspec(1, 5, wspace=0.035)
    bottom = outer[1].subgridspec(1, 3, wspace=0.42)

    source_t1 = examples[0]["t1n"]
    nonzero = source_t1[source_t1 > 0]
    vmin, vmax = np.percentile(nonzero, (1.0, 99.5))
    overlay_cmap = ListedColormap([(0, 0, 0, 0), (0.0, 0.78, 0.83, 0.48)])
    for index, example in enumerate(examples):
        axis = figure.add_subplot(top[index])
        slice_index = axial_slice(example["mask"])
        example["slice_index"] = slice_index
        image = display_plane(example["t1n"], slice_index)
        healthy = display_plane(example["mask"], slice_index)
        unhealthy = display_plane(example["unhealthy"], slice_index)
        axis.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        axis.imshow(healthy.astype(np.uint8), cmap=overlay_cmap, vmin=0, vmax=1, interpolation="nearest")
        axis.contour(healthy, levels=[0.5], colors=["#00C7D2"], linewidths=0.75)
        if unhealthy.any():
            axis.contour(
                unhealthy,
                levels=[0.5],
                colors=["#F97316"],
                linewidths=0.55,
                linestyles="--",
            )
        panel_letter = chr(ord("a") + index)
        axis.set_title(
            f"({panel_letter}) {example['label']}", fontweight="bold", pad=1.5
        )
        axis.text(
            0.5,
            -0.045,
            f"{int(example['volume_voxels']):,} vox.",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=5.4,
        )
        axis.set_axis_off()

    composition_axis = figure.add_subplot(bottom[0])
    family_order = ("fixed", "tumor", "blob", "ellipsoid", "fallback")
    bar_labels = ("Fixed", "Random", "W-cfg.", "W-real.")
    compositions = [
        {"fixed": 1.0},
        {"tumor": 1.0},
        {
            family: float(weighted_config["families"]["weights"].get(family, 0.0))
            for family in ("tumor", "blob", "ellipsoid")
        },
    ]
    weighted_rows = [row for row in records if row["policy"] == "weighted"]
    realized = Counter(
        "fallback" if row["fallback_used"] else row["mask_family"]
        for row in weighted_rows
    )
    compositions.append({name: realized[name] / len(weighted_rows) for name in family_order})
    x_positions = np.arange(len(compositions))
    bottoms = np.zeros(len(compositions))
    for family in family_order:
        heights = np.asarray([composition.get(family, 0.0) for composition in compositions])
        composition_axis.bar(
            x_positions,
            heights,
            bottom=bottoms,
            width=0.72,
            color=FAMILY_COLORS[family],
            edgecolor="white",
            linewidth=0.25,
            label=family.capitalize(),
        )
        bottoms += heights
    composition_axis.set_title("(f) Family composition", fontweight="bold", pad=2)
    composition_axis.set_ylabel("Fraction")
    composition_axis.set_ylim(0, 1)
    composition_axis.set_xticks(
        x_positions,
        bar_labels,
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    composition_axis.tick_params(axis="x", labelsize=5.5, pad=2)
    composition_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=3,
        frameon=False,
        columnspacing=0.6,
        handlelength=0.9,
    )

    volume_axis = figure.add_subplot(bottom[1])
    for policy in POLICY_ORDER:
        policy_rows = [row for row in records if row["policy"] == policy]
        values = [100.0 * float(row["mask_to_brain_ratio"]) for row in policy_rows]
        x_values, y_values = ecdf(values)
        volume_axis.step(
            x_values,
            y_values,
            where="post",
            color=POLICY_COLORS[policy],
            linewidth=1.15,
            label=f"{POLICY_LABELS[policy]} ($n$={len(values)})",
        )
    volume_axis.set_xscale("log")
    volume_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    volume_axis.set_title("(g) Relative volume", fontweight="bold", pad=2)
    volume_axis.set_xlabel("Mask / brain (%)")
    volume_axis.set_ylabel("Empirical CDF")
    volume_axis.set_ylim(0, 1.02)
    volume_axis.grid(True, color="#E5E7EB", linewidth=0.4)
    volume_axis.legend(loc="upper left", frameon=False, handlelength=1.3)

    shape_axis = figure.add_subplot(bottom[2])
    for category in SHAPE_CATEGORIES:
        category_rows = [row for row in records if shape_category(row) == category]
        if not category_rows:
            continue
        elongation = np.asarray([float(row["elongation"]) for row in category_rows])
        fill = np.asarray([float(row["bounding_box_fill_ratio"]) for row in category_rows])
        shape_axis.scatter(
            elongation,
            fill,
            s=5,
            alpha=0.27,
            linewidths=0,
            color=SHAPE_COLORS[category],
            label=category,
            rasterized=True,
        )
        shape_axis.scatter(
            [np.median(elongation)],
            [np.median(fill)],
            s=18,
            marker="x",
            linewidths=0.9,
            color=SHAPE_COLORS[category],
            rasterized=False,
        )
    shape_axis.set_xscale("log")
    shape_axis.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8]))
    shape_axis.set_xticklabels(["1", "2", "4", "8"])
    shape_axis.xaxis.set_minor_formatter(NullFormatter())
    maximum_elongation = max(float(row["elongation"]) for row in records)
    shape_axis.set_xlim(0.95, max(10.0, maximum_elongation * 1.05))
    shape_axis.set_title("(h) Final-mask shape", fontweight="bold", pad=2)
    shape_axis.set_xlabel("BBox elongation")
    shape_axis.set_ylabel("BBox fill ratio")
    shape_axis.grid(True, color="#E5E7EB", linewidth=0.4)
    shape_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        ncol=2,
        frameon=False,
        columnspacing=0.6,
        handletextpad=0.25,
    )

    for axis in (composition_axis, volume_axis, shape_axis):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(length=2, width=0.5)

    figure.subplots_adjust(left=0.075, right=0.99, top=0.965, bottom=0.17)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    figure.savefig(outputs[0], bbox_inches="tight")
    figure.savefig(outputs[1], dpi=300, bbox_inches="tight")
    plt.close(figure)
    return outputs


def git_state(repository: Path) -> dict:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "tracked_worktree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "tracked_worktree_dirty": None}


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for row in records:
        row["fallback_used"] = row["fallback_used"].strip().lower() == "true"
    return records


def reconstruct_saved_examples(
    args: argparse.Namespace,
    metadata: dict,
    random_config: dict,
    random_sampler: WeightedMaskSampler,
    weighted_config: dict,
    weighted_sampler: WeightedMaskSampler,
) -> list[dict]:
    case_id = metadata["example_case_selection"]["case_id"]
    arrays = load_case(args.data_dir, case_id)
    brain_mask = brain_mask_from_t1(
        arrays["t1n"], threshold=0.0, fill_holes=True, nonzero=True
    )
    examples = []
    for saved in metadata["examples"]:
        policy = saved["policy"]
        if policy == "fixed":
            mask = arrays["healthy"]
            seed = None
            sample_key = f"{case_id}:legacy-fixed"
        else:
            config, sampler = (
                (random_config, random_sampler)
                if policy == "random"
                else (weighted_config, weighted_sampler)
            )
            epoch, seed, sample = sample_online(
                sampler,
                config,
                case_id,
                arrays,
                args.global_seed,
                metadata["split_hash"],
                int(saved["epoch"]),
                int(saved["repeat_slot"]),
            )
            if epoch != int(saved["epoch"]):
                raise ValueError(f"Saved example epoch changed for {saved['label']}.")
            mask = sample.mask
            sample_key = sample.metadata.sample_key
        example = {
            "label": saved["label"],
            "policy": policy,
            "family": saved["family"],
            "mask": mask,
            "t1n": arrays["t1n"],
            "unhealthy": arrays["unhealthy"],
            "epoch": int(saved["epoch"]),
            "repeat_slot": int(saved["repeat_slot"]),
            "stable_seed": seed,
            "sample_key": sample_key,
            **mask_features(mask, brain_mask),
            "mask_sha256": mask_sha256(mask),
        }
        for field in ("sample_key", "mask_sha256"):
            if example[field] != saved[field]:
                raise ValueError(
                    f"Saved example {field} changed for {saved['label']}: "
                    f"{example[field]} != {saved[field]}"
                )
        examples.append(example)
    return examples


def serializable_examples(examples: list[dict], case_id: str) -> list[dict]:
    keys = (
        "label",
        "policy",
        "family",
        "epoch",
        "repeat_slot",
        "stable_seed",
        "sample_key",
        "slice_index",
        "volume_voxels",
        "mask_to_brain_ratio",
        "elongation",
        "bounding_box_fill_ratio",
        "mask_sha256",
    )
    return [{"case_id": case_id, **{key: example[key] for key in keys}} for example in examples]


def check_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs:\n{joined}")


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.holdout_file = args.holdout_file.resolve()
    args.bank_manifest = args.bank_manifest.resolve()
    args.random_config = args.random_config.resolve()
    args.weighted_config = args.weighted_config.resolve()
    args.samples_csv = args.samples_csv.resolve()
    args.metadata_json = args.metadata_json.resolve()
    args.figure_stem = args.figure_stem.resolve()
    figure_output_paths = [
        args.figure_stem.with_suffix(".pdf"),
        args.figure_stem.with_suffix(".png"),
        args.figure_stem.with_suffix(".manifest.json"),
    ]
    output_paths = [
        args.samples_csv,
        args.metadata_json,
        *figure_output_paths,
    ]
    check_outputs(
        figure_output_paths if args.plot_only else output_paths,
        args.overwrite,
    )
    if args.audit_cases < 1:
        raise ValueError("--audit-cases must be positive.")

    train_ids, holdout_ids, split_info = build_brats_split(
        str(args.data_dir),
        train_size=0,
        val_size=len(read_case_ids(args.holdout_file)),
        split_seed=args.split_seed,
        val_case_file=str(args.holdout_file),
    )
    if args.audit_cases > len(train_ids):
        raise ValueError(
            f"--audit-cases={args.audit_cases} exceeds {len(train_ids)} training cases."
        )
    rng = np.random.default_rng(args.audit_seed)
    audit_case_ids = sorted(str(case_id) for case_id in rng.permutation(train_ids)[: args.audit_cases])

    bank = TumorMaskBank(str(args.bank_manifest))
    random_config, random_sampler = make_sampler(
        "random_aug",
        args.random_config,
        bank,
        train_ids,
        holdout_ids,
    )
    weighted_config, weighted_sampler = make_sampler(
        "weighted_aug",
        args.weighted_config,
        bank,
        train_ids,
        holdout_ids,
    )
    if args.plot_only:
        if not args.samples_csv.is_file() or not args.metadata_json.is_file():
            raise FileNotFoundError(
                "--plot-only requires the existing audit CSV and metadata JSON."
            )
        metadata = json.loads(args.metadata_json.read_text())
        if metadata.get("samples_csv_sha256") != sha256_file(args.samples_csv):
            raise ValueError("Audit CSV hash does not match the saved metadata.")
        if metadata.get("split_hash") != split_info["split_hash"]:
            raise ValueError("Saved audit split hash does not match the current split.")
        if metadata.get("bank_id") != bank.manifest["bank_id"]:
            raise ValueError("Saved audit bank ID does not match the current bank.")
        records = read_csv(args.samples_csv)
        examples = reconstruct_saved_examples(
            args,
            metadata,
            random_config,
            random_sampler,
            weighted_config,
            weighted_sampler,
        )
        outputs = plot_figure(records, examples, weighted_config, args.figure_stem)
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "plot_only": True,
            "metadata_json": str(args.metadata_json),
            "metadata_json_sha256": sha256_file(args.metadata_json),
            "samples_csv": str(args.samples_csv),
            "samples_csv_sha256": sha256_file(args.samples_csv),
            "plot_script": str(Path(__file__).resolve()),
            "plot_script_sha256": sha256_file(Path(__file__).resolve()),
            "outputs": [
                {"path": str(path), "sha256": sha256_file(path)} for path in outputs
            ],
            "panels": {
                "a-e": "Fixed, random, and weighted-family mask overlays on one deterministic training case.",
                "f": "Configured and realized mask-family composition.",
                "g": "Empirical CDF of healthy-mask/brain volume ratio.",
                "h": "Final-mask bounding-box elongation and fill-ratio distribution.",
            },
        }
        manifest_path = args.figure_stem.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(*(str(path) for path in [*outputs, manifest_path]), sep="\n")
        return
    records = audit_masks(
        args,
        audit_case_ids,
        split_info["split_hash"],
        random_config,
        random_sampler,
        weighted_config,
        weighted_sampler,
    )
    example_case_id = choose_example_case(records)
    examples = choose_examples(
        args,
        records,
        example_case_id,
        split_info["split_hash"],
        random_config,
        random_sampler,
        weighted_config,
        weighted_sampler,
    )
    outputs = plot_figure(records, examples, weighted_config, args.figure_stem)
    write_csv(args.samples_csv, records)

    policy_counts = Counter(row["policy"] for row in records)
    weighted_rows = [row for row in records if row["policy"] == "weighted"]
    weighted_realized = Counter(
        "fallback" if row["fallback_used"] else row["mask_family"]
        for row in weighted_rows
    )
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Descriptive audit of the production healthy-mask samplers; not reconstruction evidence.",
        "data_dir": str(args.data_dir),
        "holdout_file": str(args.holdout_file),
        "holdout_file_sha256": sha256_file(args.holdout_file),
        "split_seed": args.split_seed,
        "split_hash": split_info["split_hash"],
        "optimization_case_count": len(train_ids),
        "holdout_case_count": len(holdout_ids),
        "audit_case_selection": {
            "method": "First audit_cases entries of a NumPy default_rng permutation of the optimization case IDs; saved sorted.",
            "audit_seed": args.audit_seed,
            "audit_case_count": len(audit_case_ids),
            "case_ids": audit_case_ids,
        },
        "global_mask_seed": args.global_seed,
        "audit_epoch": 0,
        "policy_sample_counts": dict(policy_counts),
        "weighted_configured_family_probabilities": weighted_config["families"]["weights"],
        "weighted_realized_counts": dict(weighted_realized),
        "weighted_realized_fractions": {
            family: count / len(weighted_rows) for family, count in weighted_realized.items()
        },
        "weighted_fallback_count": int(sum(row["fallback_used"] for row in weighted_rows)),
        "random_config": str(args.random_config),
        "random_config_sha256": sha256_file(args.random_config),
        "random_config_hash": mask_config_hash(random_config),
        "weighted_config": str(args.weighted_config),
        "weighted_config_sha256": sha256_file(args.weighted_config),
        "weighted_config_hash": mask_config_hash(weighted_config),
        "bank_manifest": str(args.bank_manifest),
        "bank_manifest_sha256": sha256_file(args.bank_manifest),
        "bank_id": bank.manifest["bank_id"],
        "bank_source_kind": bank.manifest["source_kind"],
        "bank_source_case_count": bank.manifest["case_count"],
        "bank_component_count": bank.manifest["component_count"],
        "samples_csv": str(args.samples_csv),
        "samples_csv_sha256": sha256_file(args.samples_csv),
        "example_case_selection": {
            "case_id": example_case_id,
            "method": "Optimization case in the audit subset nearest the median fixed healthy-mask/brain ratio; case ID breaks ties.",
            "random_example": "Most geometry-representative of the case's five production random samples.",
            "weighted_examples": "Most geometry-representative among the first five non-fallback production occurrences of each requested family.",
        },
        "examples": serializable_examples(examples, example_case_id),
        "shape_definition": {
            "elongation": "Longest divided by shortest side of the final-mask axis-aligned bounding box.",
            "bounding_box_fill_ratio": "Final mask voxels divided by final bounding-box voxels.",
        },
        "invocation": [sys.executable, *sys.argv],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "code": {
            "brats_inpainting": git_state(BRATS_ROOT),
            "paper": git_state(PAPER_ROOT),
            "plot_script": str(Path(__file__).resolve()),
            "plot_script_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "generated_at_utc": metadata["generated_at_utc"],
        "metadata_json": str(args.metadata_json),
        "metadata_json_sha256": sha256_file(args.metadata_json),
        "samples_csv": str(args.samples_csv),
        "samples_csv_sha256": metadata["samples_csv_sha256"],
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path)} for path in outputs
        ],
        "panels": {
            "a-e": "Fixed, random, and weighted-family mask overlays on one deterministic training case.",
            "f": "Configured and realized mask-family composition.",
            "g": "Empirical CDF of healthy-mask/brain volume ratio.",
            "h": "Final-mask bounding-box elongation and fill-ratio distribution.",
        },
    }
    manifest_path = args.figure_stem.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(*(str(path) for path in [args.samples_csv, args.metadata_json, *outputs, manifest_path]), sep="\n")


if __name__ == "__main__":
    main()
