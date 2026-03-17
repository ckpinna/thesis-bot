from __future__ import annotations

import argparse
from pathlib import Path

from thesis_bot.pipelines.extract_for_review import run_extract_for_review_pipeline


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

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
