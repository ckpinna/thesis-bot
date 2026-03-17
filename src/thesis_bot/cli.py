from __future__ import annotations

import argparse
from pathlib import Path

from thesis_bot.pipelines.extract_for_review import run_extract_for_review_pipeline
from thesis_bot.pipelines.load_reviewed_theses import run_load_reviewed_theses_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser(
        "extract-theses",
        help="Extract theses from PDF decks and write a review CSV.",
    )
    extract_parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing thesis deck PDFs.",
    )
    extract_parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output CSV path for human review.",
    )
    extract_parser.add_argument(
        "--model",
        default="gpt-4-turbo-preview",
        help="OpenAI model to use for thesis extraction.",
    )

    load_parser = subparsers.add_parser(
        "load-theses",
        help="Load a reviewed thesis CSV into Neo4j.",
    )
    load_parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Path to the reviewed thesis CSV.",
    )
    load_parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear existing Neo4j graph data before loading.",
    )
    load_parser.add_argument(
        "--title-model",
        default="gpt-4o-mini",
        help="OpenAI model to use for thesis titles.",
    )
    load_parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="OpenAI model to use for thesis description embeddings.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-theses":
        result = run_extract_for_review_pipeline(
            input_dir=args.input_dir,
            output_file=args.output_file,
            model=args.model,
        )
        print(f"\nNext step: review {result.review_csv_path} before Neo4j load.")
        return 0

    if args.command == "load-theses":
        result = run_load_reviewed_theses_pipeline(
            csv_path=args.csv_path,
            clear_existing=not args.keep_existing,
            title_model=args.title_model,
            embedding_model=args.embedding_model,
        )
        print("\nNeo4j load complete.")
        print(f"  CoreThesis nodes: {result.core_thesis_count}")
        print(f"  Thesis nodes: {result.thesis_node_count}")
        print(f"  SUPPORTS relationships: {result.supports_relationship_count}")
        print(f"  Node counts: {result.stats.node_counts}")
        print(f"  Relationship counts: {result.stats.relationship_counts}")
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
