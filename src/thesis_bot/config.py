from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

UNSORTED_CORE_THESIS = "Unsorted"
DEFAULT_CORE_THESES = (
    "Artificial Intelligence",
    "Construction",
    "Biotech",
    "Criteria",
    UNSORTED_CORE_THESIS,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    neo4j_uri: str | None
    neo4j_user: str | None
    neo4j_password: str | None
    extraction_source: str
    review_output_destination: str
    review_input_source: str
    dropbox_access_token: str | None
    dropbox_thesis_source_path: str | None
    local_thesis_source_path: Path | None
    dropbox_review_output_path: str | None
    local_review_output_dir: Path | None
    dropbox_reviewed_run_path: str | None
    local_reviewed_run_dir: Path | None
    core_theses: tuple[str, ...]
    data_dir: Path
    latest_thesis_decks_dir: Path
    analysis_dir: Path

    @property
    def neo4j_configured(self) -> bool:
        return bool(self.neo4j_uri and self.neo4j_user and self.neo4j_password)

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def artifact_source(self) -> str:
        """Deprecated compatibility alias."""
        return self.extraction_source

    @property
    def dropbox_reviewed_theses_path(self) -> str | None:
        """Deprecated compatibility alias."""
        return self.dropbox_reviewed_run_path


def load_settings(env_path: Path | None = None, *, override: bool = False) -> Settings:
    root = project_root()
    resolved_env_path = env_path or root / ".env"
    load_dotenv(resolved_env_path, override=override)

    data_dir = root / "data"
    analysis_dir = data_dir / "analysis"
    latest_thesis_decks_dir = data_dir / "latest_thesis_decks"
    core_theses = _parse_core_theses(
        os.getenv("CORE_THESES", ",".join(DEFAULT_CORE_THESES))
    )
    extraction_source = _normalize_mode(
        os.getenv("EXTRACTION_SOURCE", os.getenv("ARTIFACT_SOURCE", "dropbox")),
        env_name="EXTRACTION_SOURCE",
    )
    review_output_destination = _normalize_mode(
        os.getenv("REVIEW_OUTPUT_DESTINATION", "dropbox"),
        env_name="REVIEW_OUTPUT_DESTINATION",
    )
    review_input_source = _normalize_mode(
        os.getenv("REVIEW_INPUT_SOURCE", "dropbox"),
        env_name="REVIEW_INPUT_SOURCE",
    )

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        extraction_source=extraction_source,
        review_output_destination=review_output_destination,
        review_input_source=review_input_source,
        dropbox_access_token=os.getenv("DROPBOX_ACCESS_TOKEN"),
        dropbox_thesis_source_path=os.getenv("DROPBOX_THESIS_SOURCE_PATH"),
        local_thesis_source_path=_optional_path(
            os.getenv("LOCAL_THESIS_SOURCE_PATH"),
            base_dir=root,
        ),
        dropbox_review_output_path=os.getenv("DROPBOX_REVIEW_OUTPUT_PATH"),
        local_review_output_dir=_optional_path(
            os.getenv("LOCAL_REVIEW_OUTPUT_DIR"),
            base_dir=root,
            default=analysis_dir,
        ),
        dropbox_reviewed_run_path=os.getenv(
            "DROPBOX_REVIEWED_RUN_PATH",
            os.getenv("DROPBOX_REVIEWED_THESES_PATH"),
        ),
        local_reviewed_run_dir=_optional_path(
            os.getenv("LOCAL_REVIEWED_RUN_DIR"),
            base_dir=root,
        ),
        core_theses=core_theses,
        data_dir=data_dir,
        latest_thesis_decks_dir=latest_thesis_decks_dir,
        analysis_dir=analysis_dir,
    )


def _parse_core_theses(raw_value: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for item in raw_value.split(","):
        cleaned = item.strip()
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        if lowered == UNSORTED_CORE_THESIS.casefold():
            cleaned = UNSORTED_CORE_THESIS
            lowered = cleaned.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        values.append(cleaned)

    if not values:
        raise ValueError("CORE_THESES must contain at least one value.")
    if UNSORTED_CORE_THESIS.casefold() in seen:
        values = [item for item in values if item.casefold() != UNSORTED_CORE_THESIS.casefold()]
    values.append(UNSORTED_CORE_THESIS)
    return tuple(values)


def _normalize_mode(raw_value: str | None, *, env_name: str) -> str:
    value = (raw_value or "").strip().lower()
    if value not in {"dropbox", "local"}:
        raise ValueError(f"{env_name} must be either 'dropbox' or 'local'.")
    return value


def _optional_path(
    raw_value: str | None,
    *,
    base_dir: Path,
    default: Path | None = None,
) -> Path | None:
    value = raw_value.strip() if raw_value else ""
    if value:
        path = Path(value)
    elif default is not None:
        path = default
    else:
        return None
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()
