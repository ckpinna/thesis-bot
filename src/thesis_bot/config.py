from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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
    dropbox_pitch_decks_path: str | None
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

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        artifact_source=os.getenv("ARTIFACT_SOURCE", "local").strip().lower(),
        dropbox_access_token=os.getenv("DROPBOX_ACCESS_TOKEN"),
        dropbox_thesis_source_path=os.getenv("DROPBOX_THESIS_SOURCE_PATH"),
        dropbox_pitch_decks_path=os.getenv("DROPBOX_PITCH_DECKS_PATH"),
        data_dir=data_dir,
        latest_thesis_decks_dir=latest_thesis_decks_dir,
        analysis_dir=analysis_dir,
    )
