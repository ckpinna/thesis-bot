from __future__ import annotations

from pathlib import Path

from thesis_bot.config import Settings
from thesis_bot.io.document_source import DocumentArtifact, load_local_document_artifacts


def load_source_artifacts(
    settings: Settings,
    *,
    input_dir: Path | None = None,
) -> list[DocumentArtifact]:
    """Load artifacts from the configured source.

    Local source is implemented. Dropbox configuration is centralized here for
    the next source implementation.
    """
    if input_dir is not None:
        return load_local_document_artifacts(input_dir)

    if settings.artifact_source == "local":
        return load_local_document_artifacts(settings.latest_thesis_decks_dir)

    if settings.artifact_source == "dropbox":
        raise NotImplementedError(
            "Dropbox artifact loading is not implemented yet. "
            f"Configured thesis source path: {settings.dropbox_thesis_source_path or '<unset>'}"
        )

    raise ValueError(f"Unsupported artifact source: {settings.artifact_source}")
