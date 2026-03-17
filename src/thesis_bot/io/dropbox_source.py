from __future__ import annotations

from dropbox import Dropbox, common
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FileMetadata, FolderMetadata, ListFolderResult
from typing import Iterator

from thesis_bot.config import Settings
from thesis_bot.io.document_source import DocumentArtifact, SUPPORTED_DOCUMENT_EXTENSIONS


def list_dropbox_entries(
    settings: Settings,
    *,
    dropbox_path: str,
    recursive: bool = False,
) -> list[dict[str, str]]:
    """List Dropbox folder entries for path discovery and debugging."""
    if not settings.dropbox_access_token:
        raise ValueError("DROPBOX_ACCESS_TOKEN is not configured.")

    dbx = _create_rooted_dropbox_client(settings)
    try:
        dbx.users_get_current_account()
    except AuthError as exc:
        raise ValueError("Dropbox authentication failed. Check DROPBOX_ACCESS_TOKEN.") from exc

    try:
        result = dbx.files_list_folder(dropbox_path, recursive=recursive)
    except ApiError as exc:
        raise ValueError(f"Dropbox folder listing failed for {dropbox_path}: {exc}") from exc

    entries = _entries_to_rows(result)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        entries.extend(_entries_to_rows(result))
    return entries


def load_dropbox_document_artifacts(
    settings: Settings,
    *,
    dropbox_path: str | None = None,
    recursive: bool = True,
) -> list[DocumentArtifact]:
    """Load supported documents from a Dropbox folder."""
    return list(iter_dropbox_document_artifacts(settings, dropbox_path=dropbox_path, recursive=recursive))


def iter_dropbox_document_artifacts(
    settings: Settings,
    *,
    dropbox_path: str | None = None,
    recursive: bool = True,
) -> Iterator[DocumentArtifact]:
    """Yield supported Dropbox documents one at a time."""
    if not settings.dropbox_access_token:
        raise ValueError("DROPBOX_ACCESS_TOKEN is not configured.")

    source_path = dropbox_path or settings.dropbox_thesis_source_path
    if not source_path:
        raise ValueError("DROPBOX_THESIS_SOURCE_PATH is not configured.")

    dbx = _create_rooted_dropbox_client(settings)

    print(f"Listing Dropbox files from {source_path}")
    files = _list_supported_files(dbx, source_path, recursive=recursive)
    if not files:
        print(f"WARNING: No supported Dropbox documents found in {source_path}")
        return

    for file_metadata in files:
        print(f"Downloading: {file_metadata.name}")
        _, response = dbx.files_download(file_metadata.path_lower or file_metadata.path_display)
        yield DocumentArtifact(
            name=file_metadata.name,
            source_uri=file_metadata.path_display or file_metadata.path_lower or file_metadata.name,
            extension=_normalized_extension(file_metadata.name),
            content=response.content,
        )


def _list_supported_files(
    dbx: Dropbox,
    folder_path: str,
    *,
    recursive: bool,
) -> list[FileMetadata]:
    try:
        result = dbx.files_list_folder(folder_path, recursive=recursive)
    except ApiError as exc:
        raise ValueError(f"Dropbox folder listing failed for {folder_path}: {exc}") from exc

    files = _filter_supported_files(result)
    while result.has_more:
        result = dbx.files_list_folder_continue(result.cursor)
        files.extend(_filter_supported_files(result))
    return files


def _filter_supported_files(result: ListFolderResult) -> list[FileMetadata]:
    files: list[FileMetadata] = []
    for entry in result.entries:
        if isinstance(entry, FolderMetadata):
            continue
        if isinstance(entry, FileMetadata):
            extension = _normalized_extension(entry.name)
            if extension in SUPPORTED_DOCUMENT_EXTENSIONS:
                files.append(entry)
    return files


def _entries_to_rows(result: ListFolderResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for entry in result.entries:
        if isinstance(entry, FolderMetadata):
            rows.append(
                {
                    "type": "folder",
                    "name": entry.name,
                    "path": entry.path_display or entry.path_lower or entry.name,
                }
            )
        elif isinstance(entry, FileMetadata):
            rows.append(
                {
                    "type": "file",
                    "name": entry.name,
                    "path": entry.path_display or entry.path_lower or entry.name,
                }
            )
    return rows


def _normalized_extension(filename: str) -> str:
    lowered = filename.lower()
    for extension in SUPPORTED_DOCUMENT_EXTENSIONS:
        if lowered.endswith(extension):
            return extension
    return ""


def _create_rooted_dropbox_client(settings: Settings) -> Dropbox:
    dbx = Dropbox(settings.dropbox_access_token)
    try:
        account = dbx.users_get_current_account()
    except AuthError as exc:
        raise ValueError("Dropbox authentication failed. Check DROPBOX_ACCESS_TOKEN.") from exc

    root_info = getattr(account, "root_info", None)
    if root_info and hasattr(root_info, "root_namespace_id"):
        namespace_id = root_info.root_namespace_id
        return dbx.with_path_root(common.PathRoot.root(namespace_id))

    return dbx
