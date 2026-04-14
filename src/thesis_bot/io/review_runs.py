from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from thesis_bot.config import Settings
from thesis_bot.io.dropbox_source import upload_bytes_to_dropbox
from thesis_bot.io.review_csv import read_reviewed_theses_csv
from thesis_bot.io.review_csv import read_reviewed_theses_dropbox_csv

DEFAULT_REVIEW_FILENAME_SUFFIX = "_theses_for_review.csv"


def build_review_run_id(*, created_at: datetime | None = None) -> str:
    created_at = created_at or datetime.now()
    return f"review_run_{created_at.strftime('%Y%m%d_%H%M%S')}"


def bucket_review_filename(core_thesis: str) -> str:
    return f"{slugify_bucket_name(core_thesis)}{DEFAULT_REVIEW_FILENAME_SUFFIX}"


def slugify_bucket_name(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "bucket"


def write_review_bucket_csvs(
    settings: Settings,
    bucket_dataframes: dict[str, pd.DataFrame],
    *,
    run_id: str,
) -> dict[str, str]:
    if settings.review_output_destination == "dropbox":
        return _write_review_bucket_csvs_to_dropbox(settings, bucket_dataframes, run_id=run_id)
    if settings.review_output_destination == "local":
        return _write_review_bucket_csvs_to_local(settings, bucket_dataframes, run_id=run_id)
    raise ValueError(
        "Unsupported review output destination. Set REVIEW_OUTPUT_DESTINATION to 'dropbox' or 'local'."
    )


def expected_review_bucket_paths(
    settings: Settings,
    *,
    run_path: str | Path | None = None,
    core_theses: Iterable[str] | None = None,
) -> dict[str, str | Path]:
    buckets = tuple(core_theses or settings.core_theses)
    resolved_run_path = run_path or default_review_run_path(settings)

    if settings.review_input_source == "dropbox":
        run_folder = str(resolved_run_path)
        return {
            core_thesis: f"{run_folder.rstrip('/')}/{bucket_review_filename(core_thesis)}"
            for core_thesis in buckets
        }

    run_folder = Path(resolved_run_path)
    return {
        core_thesis: run_folder / bucket_review_filename(core_thesis)
        for core_thesis in buckets
    }


def default_review_run_path(settings: Settings) -> str | Path:
    if settings.review_input_source == "dropbox":
        if not settings.dropbox_reviewed_run_path:
            raise ValueError("DROPBOX_REVIEWED_RUN_PATH is not configured.")
        return settings.dropbox_reviewed_run_path
    if settings.review_input_source == "local":
        if not settings.local_reviewed_run_dir:
            raise ValueError("LOCAL_REVIEWED_RUN_DIR is not configured.")
        return settings.local_reviewed_run_dir
    raise ValueError(
        "Unsupported review input source. Set REVIEW_INPUT_SOURCE to 'dropbox' or 'local'."
    )


def read_review_bucket_dataframes(
    settings: Settings,
    *,
    allow_missing_title: bool = False,
    run_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    bucket_paths = expected_review_bucket_paths(settings, run_path=run_path)
    bucket_dataframes: dict[str, pd.DataFrame] = {}
    for core_thesis, path in bucket_paths.items():
        if settings.review_input_source == "dropbox":
            bucket_dataframes[core_thesis] = read_reviewed_theses_dropbox_csv(
                settings,
                dropbox_path=str(path),
                allowed_core_theses=settings.core_theses,
                allow_missing_title=allow_missing_title,
            )
        else:
            bucket_dataframes[core_thesis] = read_reviewed_theses_csv(
                Path(path),
                allowed_core_theses=settings.core_theses,
                allow_missing_title=allow_missing_title,
            )
    return bucket_dataframes


def _write_review_bucket_csvs_to_dropbox(
    settings: Settings,
    bucket_dataframes: dict[str, pd.DataFrame],
    *,
    run_id: str,
) -> dict[str, str]:
    if not settings.dropbox_review_output_path:
        raise ValueError("DROPBOX_REVIEW_OUTPUT_PATH is not configured.")

    output_dir = settings.dropbox_review_output_path.rstrip("/") + "/" + run_id
    uploaded_paths: dict[str, str] = {}
    for core_thesis, dataframe in bucket_dataframes.items():
        destination_path = output_dir + "/" + bucket_review_filename(core_thesis)
        content = dataframe.to_csv(index=False).encode("utf-8")
        uploaded_paths[core_thesis] = upload_bytes_to_dropbox(
            settings,
            destination_path=destination_path,
            content=content,
            overwrite=True,
        )
    return uploaded_paths


def _write_review_bucket_csvs_to_local(
    settings: Settings,
    bucket_dataframes: dict[str, pd.DataFrame],
    *,
    run_id: str,
) -> dict[str, str]:
    if not settings.local_review_output_dir:
        raise ValueError("LOCAL_REVIEW_OUTPUT_DIR is not configured.")

    output_dir = settings.local_review_output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for core_thesis, dataframe in bucket_dataframes.items():
        file_path = output_dir / bucket_review_filename(core_thesis)
        dataframe.to_csv(file_path, index=False)
        output_paths[core_thesis] = str(file_path)
    return output_paths
