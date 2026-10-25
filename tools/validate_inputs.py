#!/usr/bin/env python3
"""Validate schemas and identifier relationships in the processed DWC inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SCHEMAS: dict[str, set[str]] = {
    "NHS_US_5miles_with_SCS.csv": {
        "SIGN1",
        "Actual_Length",
        "SPEED_LIMI",
        "AADT",
        "AADT_COM",
        "Cmrcl_Electricity_Rate_Nom_2024",
        "Norm_electricity",
        "median_hh_income",
        "Norm_index",
        "MID_X",
        "MID_Y",
        "Shape_Length",
        "SEGMENT_ID_OG",
        "NEAR_SCS_ID",
        "NEAR_SCS_Dist",
    },
    "NHS_USA_5miles_Adjacent.csv": {
        "TARGET_FID",
        "JOIN_FID",
        "SIGN1",
        "SIGN1_1",
        "Shape_Length",
    },
    "NHS_US_SubstationDistance.csv": {
        "IN_FID",
        "NEAR_FID",
        "NEAR_DIST",
        "NEAR_RANK",
    },
    "SubstationsProject.csv": {"X", "Y", "lat", "lon", "Sub_ID_OG"},
    "US_SCS.csv": {"MID_X", "MID_Y", "NEAR_FID", "NEAR_DIST", "SCS_ID"},
    "US_SCS_unique_existing.csv": {
        "CLUSTER_ID",
        "ORIG_FID",
        "MID_X",
        "MID_Y",
        "NEAR_FID",
        "NEAR_DIST",
        "SCS_ID",
        "EXISTING_SCS_ID",
        "EXISTING_SCS_DIST",
    },
}


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def normalized_ids(series: pd.Series) -> set[str]:
    """Return comparable string identifiers while preserving non-integer IDs."""
    numeric = pd.to_numeric(series.dropna(), errors="coerce")
    if numeric.notna().all() and (numeric % 1 == 0).all():
        return set(numeric.astype("int64").astype(str))
    return set(series.dropna().astype(str).str.strip())


def require_unique(
    result: Validation, frame: pd.DataFrame, file_name: str, column: str
) -> None:
    nulls = int(frame[column].isna().sum())
    duplicates = int(frame[column].dropna().duplicated().sum())
    if nulls:
        result.error(f"{file_name}.{column} contains {nulls:,} null value(s)")
    if duplicates:
        result.error(
            f"{file_name}.{column} contains {duplicates:,} duplicate value(s)"
        )


def require_references(
    result: Validation,
    child: pd.DataFrame,
    child_file: str,
    child_column: str,
    allowed: set[str],
    *,
    allow_null: bool = False,
) -> None:
    values = child[child_column]
    if allow_null:
        values = values.dropna()
    elif values.isna().any():
        result.error(
            f"{child_file}.{child_column} contains {int(values.isna().sum()):,} null reference(s)"
        )
        values = values.dropna()
    actual = normalized_ids(values)
    missing = sorted(actual - allowed)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        result.error(
            f"{child_file}.{child_column} has {len(missing):,} unknown ID(s): {preview}{suffix}"
        )


def validate(input_dir: Path) -> Validation:
    result = Validation()
    frames: dict[str, pd.DataFrame] = {}

    for file_name, required_columns in SCHEMAS.items():
        path = input_dir / file_name
        if not path.is_file():
            result.error(f"missing required file: {path}")
            continue
        try:
            frame = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # pandas supplies the useful parser details
            result.error(f"could not read {path}: {exc}")
            continue
        frames[file_name] = frame
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            result.error(
                f"{file_name} is missing required column(s): {', '.join(missing_columns)}"
            )
        print(f"OK  {file_name}: {len(frame):,} rows, {len(frame.columns):,} columns")

    if len(frames) != len(SCHEMAS) or result.errors:
        return result

    segments = frames["NHS_US_5miles_with_SCS.csv"]
    adjacency = frames["NHS_USA_5miles_Adjacent.csv"]
    distances = frames["NHS_US_SubstationDistance.csv"]
    substations = frames["SubstationsProject.csv"]
    scs = frames["US_SCS.csv"]
    scs_existing = frames["US_SCS_unique_existing.csv"]

    require_unique(result, segments, "NHS_US_5miles_with_SCS.csv", "SEGMENT_ID_OG")
    require_unique(result, substations, "SubstationsProject.csv", "Sub_ID_OG")
    require_unique(result, scs, "US_SCS.csv", "SCS_ID")
    require_unique(result, scs_existing, "US_SCS_unique_existing.csv", "SCS_ID")

    segment_ids = normalized_ids(segments["SEGMENT_ID_OG"])
    substation_ids = normalized_ids(substations["Sub_ID_OG"])
    scs_ids = normalized_ids(scs["SCS_ID"])

    require_references(
        result,
        adjacency,
        "NHS_USA_5miles_Adjacent.csv",
        "TARGET_FID",
        segment_ids,
    )
    require_references(
        result,
        adjacency,
        "NHS_USA_5miles_Adjacent.csv",
        "JOIN_FID",
        segment_ids,
    )
    require_references(
        result,
        distances,
        "NHS_US_SubstationDistance.csv",
        "IN_FID",
        segment_ids,
    )
    require_references(
        result,
        distances,
        "NHS_US_SubstationDistance.csv",
        "NEAR_FID",
        substation_ids,
    )
    require_references(
        result,
        segments,
        "NHS_US_5miles_with_SCS.csv",
        "NEAR_SCS_ID",
        scs_ids,
        allow_null=True,
    )
    require_references(
        result,
        scs_existing,
        "US_SCS_unique_existing.csv",
        "SCS_ID",
        scs_ids,
    )

    for file_name, frame, column in (
        ("NHS_US_5miles_with_SCS.csv", segments, "Actual_Length"),
        ("NHS_US_5miles_with_SCS.csv", segments, "AADT"),
        ("NHS_US_SubstationDistance.csv", distances, "NEAR_DIST"),
    ):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = int(numeric.isna().sum())
        negative = int((numeric.dropna() < 0).sum())
        if invalid:
            result.error(f"{file_name}.{column} has {invalid:,} nonnumeric/null value(s)")
        if negative:
            result.error(f"{file_name}.{column} has {negative:,} negative value(s)")

    self_links = adjacency["TARGET_FID"].astype(str) == adjacency["JOIN_FID"].astype(str)
    if self_links.any():
        result.warning(f"adjacency table contains {int(self_links.sum()):,} self-link(s)")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("data/processed"),
        help="directory containing the six processed CSV files",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 2

    result = validate(args.input_dir)
    for message in result.warnings:
        print(f"WARNING: {message}", file=sys.stderr)
    for message in result.errors:
        print(f"ERROR: {message}", file=sys.stderr)

    if result.errors:
        print(f"Validation failed with {len(result.errors)} error(s).", file=sys.stderr)
        return 1
    print("Validation passed: schemas and identifier relationships are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
