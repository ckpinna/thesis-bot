from __future__ import annotations

from pathlib import Path
from typing import Iterator

from thesis_bot.config import Settings
from thesis_bot.io.document_source import DocumentArtifact, iter_local_document_artifacts
from thesis_bot.io.dropbox_source import iter_dropbox_document_artifacts


def load_source_artifacts(
    settings: Settings,
    *,
    input_dir: Path | None = None,
) -> list[DocumentArtifact]:
    """Load all artifacts from the configured source."""
    return list(iter_source_artifacts(settings, input_dir=input_dir))


def iter_source_artifacts(
    settings: Settings,
    *,
    input_dir: Path | None = None,
) -> Iterator[DocumentArtifact]:
    """Load artifacts from the configured source.

    Local source is implemented. Dropbox configuration is centralized here for
    the next source implementation.
    """
    if input_dir is not None:
        yield from iter_local_document_artifacts(input_dir)
        return

    if settings.artifact_source == "local":
        yield from iter_local_document_artifacts(settings.latest_thesis_decks_dir)
        return

    if settings.artifact_source == "dropbox":
        yield from iter_dropbox_document_artifacts(settings, dropbox_path=None, recursive=True)
        return

    raise ValueError(f"Unsupported artifact source: {settings.artifact_source}")
