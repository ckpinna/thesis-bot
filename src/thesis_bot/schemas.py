from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


REVIEWED_CSV_COLUMNS = [
    "Thesis Number",
    "Thesis Statement",
    "Description",
    "Supports Thesis Numbers",
    "Core Thesis",
    "Source File",
]

REQUIRED_REVIEWED_CSV_COLUMNS = [
    "Thesis Number",
    "Thesis Statement",
    "Description",
    "Supports Thesis Numbers",
    "Core Thesis",
    "Source File",
]


@dataclass(frozen=True)
class ReviewedThesisRecord:
    thesis_number: int
    thesis_statement: str
    description: str
    supports_thesis_numbers: str
    core_thesis: str
    source_file: str


def validate_reviewed_theses_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the reviewed thesis CSV contract."""
    missing_columns = [
        column for column in REQUIRED_REVIEWED_CSV_COLUMNS if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required reviewed CSV columns: {missing_columns}")

    normalized = dataframe.loc[:, REVIEWED_CSV_COLUMNS].copy()
    normalized["Thesis Number"] = pd.to_numeric(
        normalized["Thesis Number"],
        errors="raise",
    ).astype(int)

    for column in REVIEWED_CSV_COLUMNS:
        if column == "Thesis Number":
            continue
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()

    if normalized["Thesis Statement"].eq("").any():
        raise ValueError("Reviewed CSV contains empty 'Thesis Statement' values.")
    if normalized["Description"].eq("").any():
        raise ValueError("Reviewed CSV contains empty 'Description' values.")
    if normalized["Core Thesis"].eq("").any():
        raise ValueError(
            "Reviewed CSV contains empty 'Core Thesis' values. "
            "Populate this during the human review step."
        )
    if normalized["Source File"].eq("").any():
        raise ValueError("Reviewed CSV contains empty 'Source File' values.")

    duplicated_numbers = normalized["Thesis Number"].duplicated()
    if duplicated_numbers.any():
        duplicates = normalized.loc[duplicated_numbers, "Thesis Number"].tolist()
        raise ValueError(f"Reviewed CSV contains duplicate thesis numbers: {duplicates}")

    return normalized


def reviewed_records_from_dataframe(dataframe: pd.DataFrame) -> list[ReviewedThesisRecord]:
    """Convert a validated dataframe to typed reviewed-thesis records."""
    validated = validate_reviewed_theses_dataframe(dataframe)
    return [
        ReviewedThesisRecord(
            thesis_number=int(row["Thesis Number"]),
            thesis_statement=row["Thesis Statement"],
            description=row["Description"],
            supports_thesis_numbers=row["Supports Thesis Numbers"],
            core_thesis=row["Core Thesis"],
            source_file=row["Source File"],
        )
        for _, row in validated.iterrows()
    ]


def missing_reviewed_columns(columns: Iterable[str]) -> list[str]:
    """Return missing required columns for the reviewed thesis contract."""
    available = set(columns)
    return [column for column in REQUIRED_REVIEWED_CSV_COLUMNS if column not in available]

