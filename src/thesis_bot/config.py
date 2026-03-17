from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

UNSORTED_CORE_THESIS = "Unsorted"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    neo4j_uri: str | None
    neo4j_user: str | None
    neo4j_password: str | None
    artifact_source: str
    dropbox_access_token: str | None
    dropbox_thesis_source_path: str | None
    dropbox_pipeline_decks_path: str | None
    dropbox_review_output_path: str | None
    dropbox_reviewed_theses_path: str | None
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


def load_settings(env_path: Path | None = None, *, override: bool = False) -> Settings:
    root = project_root()
    resolved_env_path = env_path or root / ".env"
    load_dotenv(resolved_env_path, override=override)

    data_dir = root / "data"
    analysis_dir = data_dir / "analysis"
    latest_thesis_decks_dir = data_dir / "latest_thesis_decks"
    core_theses = _parse_core_theses(os.getenv("CORE_THESES", "AI,BioTech,ConTech,Investment Criteria"))

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        artifact_source=os.getenv("ARTIFACT_SOURCE", "local").strip().lower(),
        dropbox_access_token=os.getenv("DROPBOX_ACCESS_TOKEN"),
        dropbox_thesis_source_path=os.getenv("DROPBOX_THESIS_SOURCE_PATH"),
        dropbox_pipeline_decks_path=os.getenv("DROPBOX_PIPELINE_DECKS_PATH"),
        dropbox_review_output_path=os.getenv("DROPBOX_REVIEW_OUTPUT_PATH"),
        dropbox_reviewed_theses_path=os.getenv("DROPBOX_REVIEWED_THESES_PATH"),
        core_theses=core_theses,
        data_dir=data_dir,
        latest_thesis_decks_dir=latest_thesis_decks_dir,
        analysis_dir=analysis_dir,
    )


def _parse_core_theses(raw_value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    if not values:
        raise ValueError("CORE_THESES must contain at least one value.")
    return values
