from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".md", ".txt")


@dataclass(frozen=True)
class DocumentArtifact:
    name: str
    source_uri: str
    extension: str
    content: bytes


@dataclass(frozen=True)
class ParsedDocument:
    name: str
    source_uri: str
    extension: str
    text: str


def load_local_document_artifacts(data_folder: Path) -> list[DocumentArtifact]:
    """Load supported document files from a local directory."""
    if not data_folder.exists():
        print(f"WARNING: Input folder does not exist: {data_folder}")
        return []

    artifacts: list[DocumentArtifact] = []
    for path in sorted(data_folder.iterdir()):
        if not path.is_file():
            continue
        extension = path.suffix.lower()
        if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue

        try:
            artifacts.append(
                DocumentArtifact(
                    name=path.name,
                    source_uri=str(path.resolve()),
                    extension=extension,
                    content=path.read_bytes(),
                )
            )
        except Exception as exc:
            print(f"  Failed to load {path.name}: {exc}")

    if not artifacts:
        print(f"WARNING: No supported documents found in {data_folder}")
    return artifacts

