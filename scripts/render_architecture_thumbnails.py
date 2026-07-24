#!/usr/bin/env python3
"""Render deterministic MRI thumbnails for the CATCH architecture figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, __version__ as pillow_version


PAPER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PAPER_ROOT.parent
DEFAULT_DATA_DIR = (
    WORKSPACE_ROOT
    / "data"
    / "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Training"
)
DEFAULT_AUDIT_METADATA = PAPER_ROOT / "data" / "mask_augmentation_metadata.json"
DEFAULT_OUTPUT_DIR = PAPER_ROOT / "figures"
DEFAULT_METADATA = PAPER_ROOT / "data" / "catch_architecture_assets.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--audit-metadata", type=Path, default=DEFAULT_AUDIT_METADATA
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-json", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_array(path: Path) -> np.ndarray:
    return nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)


def display_plane(volume: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(volume[:, :, slice_index])


def square_crop_bounds(
    foreground: np.ndarray,
    padding: int = 8,
) -> tuple[int, int, int, int]:
    coordinates = np.argwhere(foreground)
    if not len(coordinates):
        raise ValueError("Cannot crop an empty foreground.")

    lower = np.maximum(coordinates.min(axis=0) - padding, 0)
    upper = np.minimum(coordinates.max(axis=0) + padding + 1, foreground.shape)
    center = (lower + upper) / 2.0
    side = int(max(upper - lower))
    row_start = int(round(center[0] - side / 2.0))
    col_start = int(round(center[1] - side / 2.0))
    row_start = min(max(row_start, 0), foreground.shape[0] - side)
    col_start = min(max(col_start, 0), foreground.shape[1] - side)
    return row_start, row_start + side, col_start, col_start + side


def normalized_uint8(array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.rint(255.0 * normalized).astype(np.uint8)


def save_thumbnail(array: np.ndarray, path: Path, size: int) -> None:
    image = Image.fromarray(array)
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.save(path, optimize=True)


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive.")

    audit_metadata = json.loads(args.audit_metadata.read_text())
    selection = audit_metadata["example_case_selection"]
    case_id = selection["case_id"]
    case_dir = args.data_dir / case_id
    source_paths = {
        "complete_t1n": case_dir / f"{case_id}-t1n.nii.gz",
        "voided_t1n": case_dir / f"{case_id}-t1n-voided.nii.gz",
        "mask": case_dir / f"{case_id}-mask.nii.gz",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    complete = canonical_array(source_paths["complete_t1n"])
    voided = canonical_array(source_paths["voided_t1n"])
    mask = canonical_array(source_paths["mask"]) > 0
    shapes = {complete.shape, voided.shape, mask.shape}
    if len(shapes) != 1:
        raise ValueError(f"Canonical shape mismatch: {sorted(shapes)}")

    slice_index = int(np.argmax(mask.sum(axis=(0, 1))))
    complete_plane = display_plane(complete, slice_index)
    voided_plane = display_plane(voided, slice_index)
    mask_plane = display_plane(mask, slice_index)
    crop = square_crop_bounds((complete_plane > 0) | mask_plane)
    row_start, row_stop, col_start, col_stop = crop
    crop_slice = np.s_[row_start:row_stop, col_start:col_stop]

    nonzero = complete[complete > 0]
    if not len(nonzero):
        raise ValueError("The complete T1n volume contains no foreground voxels.")
    vmin, vmax = (float(value) for value in np.percentile(nonzero, (1.0, 99.5)))
    if vmax <= vmin:
        raise ValueError(f"Invalid intensity window: [{vmin}, {vmax}]")

    output_paths = {
        "complete_t1n": args.output_dir / "catch_architecture_complete.png",
        "voided_t1n": args.output_dir / "catch_architecture_voided.png",
        "mask": args.output_dir / "catch_architecture_mask.png",
    }
    claimed_paths = [*output_paths.values(), args.metadata_json]
    existing = [str(path) for path in claimed_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing architecture assets: {existing}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)

    save_thumbnail(
        normalized_uint8(complete_plane[crop_slice], vmin, vmax),
        output_paths["complete_t1n"],
        args.size,
    )
    save_thumbnail(
        normalized_uint8(voided_plane[crop_slice], vmin, vmax),
        output_paths["voided_t1n"],
        args.size,
    )
    save_thumbnail(
        (255 * mask_plane[crop_slice].astype(np.uint8)),
        output_paths["mask"],
        args.size,
    )

    metadata = {
        "schema_version": 1,
        "purpose": "Audited input thumbnails for the CATCH architecture figure.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "case_selection": {
            "source": str(args.audit_metadata),
            "source_sha256": sha256_file(args.audit_metadata),
            "method": selection["method"],
            "metric_independent": True,
        },
        "slice_selection": {
            "axis": "canonical RAS axial",
            "index": slice_index,
            "method": "Maximum full inpainting-mask area; first index breaks ties.",
            "mask_area_voxels": int(mask[:, :, slice_index].sum()),
        },
        "display": {
            "plane_transform": "numpy.rot90",
            "crop_bounds_row_col": [row_start, row_stop, col_start, col_stop],
            "intensity_percentiles": [1.0, 99.5],
            "intensity_window": [vmin, vmax],
            "output_size_pixels": [args.size, args.size],
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in source_paths.items()
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in output_paths.items()
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "invocation": [sys.executable, *sys.argv],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "nibabel": nib.__version__,
            "pillow": pillow_version,
        },
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print(
        f"Rendered architecture thumbnails for {case_id}, axial slice {slice_index}."
    )


if __name__ == "__main__":
    main()
