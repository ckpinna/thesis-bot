from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from thesis_bot.schemas import validate_reviewed_theses_dataframe


def read_reviewed_theses_csv(
    csv_path: Path,
    *,
    allowed_core_theses: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load and validate a reviewed thesis CSV."""
    dataframe = pd.read_csv(csv_path)
    return validate_reviewed_theses_dataframe(
        dataframe,
        allowed_core_theses=allowed_core_theses,
    )
