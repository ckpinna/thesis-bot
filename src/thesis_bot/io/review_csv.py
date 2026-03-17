from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import Iterable

import pandas as pd

from thesis_bot.config import Settings
from thesis_bot.io.dropbox_source import download_dropbox_file_bytes
from thesis_bot.schemas import validate_reviewed_theses_dataframe


def read_reviewed_theses_csv(
    csv_path: Path,
    *,
    allowed_core_theses: Iterable[str] | None = None,
    allow_missing_title: bool = False,
) -> pd.DataFrame:
    """Load and validate a reviewed thesis CSV."""
    dataframe = pd.read_csv(csv_path)
    return validate_reviewed_theses_dataframe(
        dataframe,
        allowed_core_theses=allowed_core_theses,
        allow_missing_title=allow_missing_title,
    )


def read_reviewed_theses_dropbox_csv(
    settings: Settings,
    *,
    dropbox_path: str,
    allowed_core_theses: Iterable[str] | None = None,
    allow_missing_title: bool = False,
) -> pd.DataFrame:
    """Load and validate a reviewed thesis CSV from Dropbox."""
    content = download_dropbox_file_bytes(settings, dropbox_path=dropbox_path)
    dataframe = pd.read_csv(BytesIO(content))
    return validate_reviewed_theses_dataframe(
        dataframe,
        allowed_core_theses=allowed_core_theses,
        allow_missing_title=allow_missing_title,
    )
