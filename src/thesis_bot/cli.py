from __future__ import annotations

import argparse
from thesis_bot.config import load_settings
from thesis_bot.io.dropbox_source import list_dropbox_entries
from thesis_bot.pipelines.extract_for_review import run_extract_for_review_pipeline
from thesis_bot.pipelines.load_reviewed_theses import run_load_reviewed_theses_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract-theses",
        help="Extract theses from the configured Dropbox source and write a Dropbox review CSV.",
    )
    extract_parser.add_argument(
        "--model",
        default="gpt-4-turbo-preview",
        help="OpenAI model to use for thesis extraction.",
    )
    extract_parser.add_argument(
        "--title-model",
        default="gpt-4o-mini",
        help="OpenAI model to use for 4-word thesis titles.",
    )

    load_parser = subparsers.add_parser(
        "load-theses",
        help="Load the configured Dropbox reviewed thesis CSV into Neo4j.",
    )
    load_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear existing Neo4j graph data before loading.",
    )
    load_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI model to use for thesis description embeddings.",
    )
    load_parser.add_argument(
        "--title-model",
        default="gpt-4o-mini",
        help="OpenAI model to use when backfilling missing thesis titles.",
    )

    list_dropbox_parser = subparsers.add_parser(
        "list-dropbox",
        help="List Dropbox folder entries to verify API-visible paths.",
    )
    list_dropbox_parser.add_argument(
        "--path",
        required=True,
        help="Dropbox path to inspect, for example '/10. Proprietary'.",
    )
    list_dropbox_parser.add_argument(
        "--recursive",
        action="store_true",
        help="List recursively instead of immediate children only.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-theses":
        result = run_extract_for_review_pipeline(
            model=args.model,
            title_model=args.title_model,
        )
        print(f"\nNext step: review {result.dropbox_review_csv_path} before Neo4j load.")
        return 0

    if args.command == "load-theses":
        result = run_load_reviewed_theses_pipeline(
            clear_existing=not args.keep_existing,
            embedding_model=args.embedding_model,
            title_model=args.title_model,
        )
        print("\nNeo4j load complete.")
        print(f"  CoreThesis nodes: {result.core_thesis_count}")
        print(f"  Thesis nodes: {result.thesis_node_count}")
        print(f"  SUPPORTS relationships: {result.supports_relationship_count}")
        print(f"  Node counts: {result.stats.node_counts}")
        print(f"  Relationship counts: {result.stats.relationship_counts}")
        return 0

    if args.command == "list-dropbox":
        settings = load_settings()
        entries = list_dropbox_entries(
            settings,
            dropbox_path=args.path,
            recursive=args.recursive,
        )
        print(f"Entries at {args.path}:")
        for entry in entries:
            print(f"  [{entry['type']}] {entry['path']}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
