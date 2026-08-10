#!/usr/bin/env python3
"""Verify the 25-case trajectory-policy ranks from tracked source metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


PAPER_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PAPER_ROOT / "data" / "ensemble_selection_summary_metadata.json"
ARMS = ("fixed", "random", "weighted")
CONFIGURATIONS = ("n1", "mean_n3", "median_n3", "mean_n5", "median_n5")
METRICS = ("ssim", "psnr", "mse")
HIGHER_IS_BETTER = {"ssim": True, "psnr": True, "mse": False}
EXPECTED_COLUMNS = ("arm", "configuration", "case", *METRICS)
SUMMARY_COLUMNS = (
    "configuration",
    "fixed_joint_rank",
    "random_joint_rank",
    "weighted_joint_rank",
    "pooled_joint_rank",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.relative_to(PAPER_ROOT)}: "
            f"expected {expected}, found {actual}"
        )


def average_tied_ranks(
    values: list[float], higher_is_better: bool
) -> list[float]:
    order = sorted(
        range(len(values)),
        key=lambda index: values[index],
        reverse=higher_is_better,
    )
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def read_source_metrics(
    path: Path, expected_cases: list[str]
) -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path.relative_to(PAPER_ROOT)}: "
                f"{reader.fieldnames}"
            )
        for source in reader:
            arm = source["arm"]
            configuration = source["configuration"]
            case = source["case"]
            if arm not in ARMS:
                raise ValueError(f"Unexpected arm in source metrics: {arm}")
            if configuration not in CONFIGURATIONS:
                raise ValueError(
                    f"Unexpected configuration in source metrics: {configuration}"
                )
            key = (arm, configuration, case)
            if key in rows:
                raise ValueError(f"Duplicate source row: {key}")
            values = {metric: float(source[metric]) for metric in METRICS}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite metric in source row: {key}")
            rows[key] = values

    expected_case_set = set(expected_cases)
    if len(expected_cases) != 25 or len(expected_case_set) != 25:
        raise ValueError("Metadata must contain 25 unique development case IDs")
    for arm in ARMS:
        for configuration in CONFIGURATIONS:
            actual_cases = {
                case
                for row_arm, row_configuration, case in rows
                if row_arm == arm and row_configuration == configuration
            }
            if actual_cases != expected_case_set:
                missing = sorted(expected_case_set - actual_cases)
                unexpected = sorted(actual_cases - expected_case_set)
                raise ValueError(
                    f"Case mismatch for {arm}/{configuration}: "
                    f"missing={missing}, unexpected={unexpected}"
                )
    expected_row_count = len(ARMS) * len(CONFIGURATIONS) * len(expected_cases)
    if len(rows) != expected_row_count:
        raise ValueError(f"Expected {expected_row_count} source rows, found {len(rows)}")
    return rows


def compute_joint_ranks(
    rows: dict[tuple[str, str, str], dict[str, float]], case_ids: list[str]
) -> dict[str, dict[str, float]]:
    result = {configuration: {} for configuration in CONFIGURATIONS}
    for arm in ARMS:
        totals = [0.0] * len(CONFIGURATIONS)
        for case in case_ids:
            for metric in METRICS:
                values = [
                    rows[(arm, configuration, case)][metric]
                    for configuration in CONFIGURATIONS
                ]
                ranks = average_tied_ranks(values, HIGHER_IS_BETTER[metric])
                totals = [total + rank for total, rank in zip(totals, ranks)]
        denominator = len(case_ids) * len(METRICS)
        for configuration, total in zip(CONFIGURATIONS, totals):
            result[configuration][arm] = total / denominator

    for configuration in CONFIGURATIONS:
        result[configuration]["pooled"] = sum(
            result[configuration][arm] for arm in ARMS
        ) / len(ARMS)
    return result


def read_reported_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUMMARY_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {path.relative_to(PAPER_ROOT)}: "
                f"{reader.fieldnames}"
            )
        rows = {row["configuration"]: row for row in reader}
    if tuple(rows) != CONFIGURATIONS:
        raise ValueError(
            f"Unexpected configuration order in reported summary: {tuple(rows)}"
        )
    return rows


def main() -> None:
    metadata = json.loads(METADATA_PATH.read_text())
    source = metadata["tracked_source_snapshot"]
    summary = metadata["derived_summary"]
    source_path = PAPER_ROOT / source["path"]
    summary_path = PAPER_ROOT / summary["path"]
    require_hash(source_path, source["sha256"])
    require_hash(summary_path, summary["sha256"])

    case_ids = metadata["cohort"]["case_ids"]
    rows = read_source_metrics(source_path, case_ids)
    if len(rows) != source["row_count"]:
        raise ValueError(
            f"Metadata records {source['row_count']} source rows, found {len(rows)}"
        )
    computed = compute_joint_ranks(rows, case_ids)
    reported = read_reported_summary(summary_path)

    print(",".join(SUMMARY_COLUMNS))
    for configuration in CONFIGURATIONS:
        values = [
            computed[configuration]["fixed"],
            computed[configuration]["random"],
            computed[configuration]["weighted"],
            computed[configuration]["pooled"],
        ]
        rendered = [f"{value:.4f}" for value in values]
        expected = [reported[configuration][column] for column in SUMMARY_COLUMNS[1:]]
        if rendered != expected:
            raise ValueError(
                f"Rank mismatch for {configuration}: "
                f"computed={rendered}, reported={expected}"
            )
        print(",".join((configuration, *rendered)))

    print(
        "PASS: reproduced all 25-case trajectory-policy ranks from the "
        "tracked per-case metrics."
    )


if __name__ == "__main__":
    main()
