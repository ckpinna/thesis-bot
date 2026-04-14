from __future__ import annotations

from typing import Iterator

from thesis_bot.config import Settings
from thesis_bot.io.document_source import DocumentArtifact
from thesis_bot.io.document_source import iter_local_document_artifacts
from thesis_bot.io.dropbox_source import iter_dropbox_document_artifacts


def load_source_artifacts(
    settings: Settings,
) -> list[DocumentArtifact]:
    """Load all artifacts from the configured source."""
    return list(iter_source_artifacts(settings))


def iter_source_artifacts(
    settings: Settings,
) -> Iterator[DocumentArtifact]:
    """Load artifacts from the configured source."""
    if settings.extraction_source == "dropbox":
        yield from iter_dropbox_document_artifacts(settings, dropbox_path=None, recursive=True)
        return
    if settings.extraction_source == "local":
        if not settings.local_thesis_source_path:
            raise ValueError("LOCAL_THESIS_SOURCE_PATH is not configured.")
        yield from iter_local_document_artifacts(
            settings.local_thesis_source_path,
            recursive=True,
        )
        return

    raise ValueError(
        "Unsupported extraction source. Set EXTRACTION_SOURCE to 'dropbox' or 'local'."
    )
