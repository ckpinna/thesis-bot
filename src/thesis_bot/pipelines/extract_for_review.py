from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from thesis_bot.config import Settings, load_settings
from thesis_bot.config import UNSORTED_CORE_THESIS
from thesis_bot.clients.openai_client import create_openai_client
from thesis_bot.io.document_parsers import parse_document_artifact
from thesis_bot.io.dropbox_source import upload_bytes_to_dropbox
from thesis_bot.io.source_loader import iter_source_artifacts
from thesis_bot.schemas import REVIEWED_CSV_COLUMNS


DEFAULT_EXTRACTION_MODEL = "gpt-4-turbo-preview"
DEFAULT_TITLE_MODEL = "gpt-4o-mini"
DEFAULT_REVIEW_FILENAME = "theses_for_review.csv"
MAX_DOCUMENT_CHARS = 470000
CHUNK_OVERLAP_CHARS = 10000


@dataclass(frozen=True)
class ExtractionRunResult:
    document_char_counts: dict[str, int]
    all_extractions: dict[str, dict[str, Any]]
    deduplicated: dict[str, Any]
    review_dataframe: pd.DataFrame
    review_csv_path: Path | None = None
    dropbox_review_csv_path: str | None = None

    @property
    def pdf_texts(self) -> dict[str, str]:
        """Deprecated compatibility shim for older notebook code."""
        return {name: "" for name in self.document_char_counts}


def extract_theses_from_text(
    text: str,
    filename: str,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_EXTRACTION_MODEL,
    max_chars: int = MAX_DOCUMENT_CHARS,
) -> dict[str, Any]:
    """Extract thesis statements and support relationships from a document."""
    if not openai_client:
        print("WARNING: OpenAI client not configured.")
        return {"theses": [], "thesis_supports": [], "filename": filename}

    if len(text) > max_chars:
        print(f"  Document is very long ({len(text):,} chars). Processing in chunks...")
        return extract_theses_from_text_chunked(
            text,
            filename,
            openai_client,
            model=model,
            chunk_size=max_chars,
        )

    prompt = f"""Analyze the following document and extract ALL thesis statements/arguments.

A thesis is a claim, argument, or proposition that the document is making. Extract every distinct thesis statement.

For each thesis, provide:
1. The exact thesis statement (the claim being made)
2. A brief 2-3 sentence description that accurately describes what the thesis is about

Also identify relationships: if one thesis supports or provides evidence for another thesis, note that.

IMPORTANT:
- Analyze the ENTIRE document from beginning to end
- Extract ALL theses, even if they seem similar (we will deduplicate later)
- Be thorough - don't miss any arguments or claims
- Each thesis should be a distinct statement
- If theses are redundant or very similar, still list them (we'll handle deduplication)

Return a JSON object with this structure:
{{
    "theses": [
        {{
            "thesis": "The exact thesis statement",
            "description": "2-3 sentences accurately describing what this thesis is about"
        }}
    ],
    "thesis_supports": [
        {{
            "source_thesis": "The supporting thesis statement",
            "target_thesis": "The thesis statement it supports"
        }}
    ]
}}

Document text (FULL DOCUMENT - analyze everything):
{text}
"""

    try:
        print(f"  Analyzing document ({len(text):,} characters)...")
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at analyzing documents and extracting thesis "
                        "statements. Always return valid JSON. Be thorough and extract "
                        "all thesis statements from the entire document, even if they "
                        "seem similar."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        result_text = response.choices[0].message.content or ""
        result = _parse_json_response(result_text)
        result["filename"] = filename
        return result
    except json.JSONDecodeError as exc:
        print(f"  Failed to parse JSON for {filename}: {exc}")
        return {"theses": [], "thesis_supports": [], "filename": filename}
    except Exception as exc:
        print(f"  Error analyzing {filename}: {exc}")
        return {"theses": [], "thesis_supports": [], "filename": filename}


def extract_theses_from_text_chunked(
    text: str,
    filename: str,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_EXTRACTION_MODEL,
    chunk_size: int = MAX_DOCUMENT_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> dict[str, Any]:
    """Process very long documents in chunks with overlap."""
    chunks: list[str] = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        if i + chunk_size >= len(text):
            break

    print(f"    Processing {len(chunks)} chunks...")
    all_theses: list[dict[str, Any]] = []
    all_supports: list[dict[str, Any]] = []

    for index, chunk in enumerate(chunks, start=1):
        print(f"    Chunk {index}/{len(chunks)} ({len(chunk):,} chars)...")
        chunk_result = extract_theses_from_text(
            chunk,
            f"{filename}_chunk{index - 1}",
            openai_client,
            model=model,
            max_chars=chunk_size,
        )
        all_theses.extend(chunk_result.get("theses", []))
        all_supports.extend(chunk_result.get("thesis_supports", []))

    return {
        "theses": all_theses,
        "thesis_supports": all_supports,
        "filename": filename,
    }


def deduplicate_theses(all_extractions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Combine theses, deduplicate similar entries, and assign unique numbers."""
    all_theses: list[dict[str, Any]] = []
    thesis_to_number: dict[str, int] = {}
    thesis_number = 1

    for filename, extraction in all_extractions.items():
        for thesis_data in extraction.get("theses", []):
            thesis = thesis_data.get("thesis", "").strip()
            description = thesis_data.get("description", "").strip()
            if thesis:
                all_theses.append(
                    {
                        "thesis": thesis,
                        "description": description,
                        "source_file": filename,
                    }
                )

    unique_theses: list[dict[str, Any]] = []
    seen_theses: set[str] = set()

    for thesis_data in all_theses:
        thesis = thesis_data["thesis"]
        thesis_lower = thesis.lower().strip()

        is_duplicate = False
        for seen in seen_theses:
            seen_lower = seen.lower()
            if (
                (thesis_lower in seen_lower or seen_lower in thesis_lower)
                and abs(len(thesis_lower) - len(seen)) < 20
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            seen_theses.add(thesis)
            unique_theses.append(thesis_data)
            thesis_to_number[thesis] = thesis_number
            thesis_number += 1

    thesis_supports: list[dict[str, Any]] = []
    for extraction in all_extractions.values():
        for relation in extraction.get("thesis_supports", []):
            source = relation.get("source_thesis", "").strip()
            target = relation.get("target_thesis", "").strip()

            source_num = None
            target_num = None

            for thesis, number in thesis_to_number.items():
                if source and (source in thesis or thesis in source):
                    source_num = number
                if target and (target in thesis or thesis in target):
                    target_num = number

            if source_num and target_num:
                thesis_supports.append(
                    {
                        "source_thesis_number": source_num,
                        "target_thesis_number": target_num,
                        "source_thesis": source,
                        "target_thesis": target,
                    }
                )

    return {
        "theses": unique_theses,
        "thesis_to_number": thesis_to_number,
        "thesis_supports": thesis_supports,
    }


def generate_4word_title(
    thesis_statement: str,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_TITLE_MODEL,
) -> str:
    """Generate a concise 4-word title for a thesis statement."""
    if not openai_client:
        return thesis_statement[:50]

    prompt = f"""Generate a concise, tight 4-word title that summarizes the following thesis statement.

Requirements:
- Exactly 4 words (no more, no less)
- Capture the core essence of the thesis
- Be specific and meaningful
- Use title case

Thesis statement: "{thesis_statement}"

Return only the 4-word title."""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert at creating concise, impactful titles. "
                        "Always return exactly 4 words."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=20,
        )
        title = (response.choices[0].message.content or "").strip()
        return title.strip('"').strip("'").strip() or thesis_statement[:50]
    except Exception as exc:
        print(f"  Failed to generate title: {exc}")
        return thesis_statement[:50]


def generate_titles_for_deduplicated_theses(
    deduplicated: dict[str, Any],
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_TITLE_MODEL,
) -> dict[str, str]:
    """Generate titles keyed by thesis statement."""
    print("Generating 4-word thesis titles...")
    titles: dict[str, str] = {}
    for thesis_data in deduplicated["theses"]:
        thesis = thesis_data["thesis"]
        thesis_num = deduplicated["thesis_to_number"][thesis]
        print(f"  Generating title for thesis {thesis_num}...")
        title = generate_4word_title(thesis, openai_client, model=model)
        titles[thesis] = title
        print(f"    {title}")
    print(f"\nGenerated {len(titles)} titles")
    return titles


def classify_core_thesis(
    thesis_statement: str,
    description: str,
    openai_client: OpenAI | None,
    allowed_core_theses: tuple[str, ...],
) -> str:
    """Classify a thesis into one configured core-thesis bucket."""
    if not openai_client:
        return UNSORTED_CORE_THESIS

    options_text = ", ".join(allowed_core_theses)
    prompt = f"""Assign the following thesis to exactly one core thesis category.

Allowed categories:
{options_text}

Requirements:
- Return exactly one category from the allowed list
- Do not invent new categories
- Pick the single best fit

Thesis statement: "{thesis_statement}"
Description: "{description}"

Return only the selected category."""
    try:
        response = openai_client.chat.completions.create(
            model=DEFAULT_TITLE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify theses into a fixed set of allowed categories. "
                        "Return exactly one allowed category."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=20,
        )
        selected = (response.choices[0].message.content or "").strip()
        selected = selected.strip('"').strip("'").strip()
        if selected in allowed_core_theses:
            return selected
    except Exception as exc:
        print(f"  Failed to classify core thesis: {exc}")

    return UNSORTED_CORE_THESIS


def classify_core_theses_for_deduplicated_theses(
    deduplicated: dict[str, Any],
    openai_client: OpenAI | None,
    allowed_core_theses: tuple[str, ...],
) -> dict[str, str]:
    """Assign a configured core-thesis value to each thesis statement."""
    print("Classifying theses into configured core thesis buckets...")
    assignments: dict[str, str] = {}
    for thesis_data in deduplicated["theses"]:
        thesis = thesis_data["thesis"]
        thesis_num = deduplicated["thesis_to_number"][thesis]
        print(f"  Classifying thesis {thesis_num}...")
        selected = classify_core_thesis(
            thesis,
            thesis_data["description"],
            openai_client,
            allowed_core_theses,
        )
        assignments[thesis] = selected
        print(f"    {selected}")
    return assignments


def create_review_dataframe(
    deduplicated: dict[str, Any],
    thesis_titles: dict[str, str],
    thesis_core_theses: dict[str, str],
    allowed_core_theses: tuple[str, ...],
) -> pd.DataFrame:
    """Create a review-friendly dataframe from deduplicated thesis results."""
    thesis_to_number = deduplicated["thesis_to_number"]
    thesis_supports = deduplicated["thesis_supports"]

    supports_map: dict[int, list[int]] = {}
    for relation in thesis_supports:
        source_num = relation["source_thesis_number"]
        target_num = relation["target_thesis_number"]
        supports_map.setdefault(source_num, []).append(target_num)

    rows: list[dict[str, Any]] = []
    for thesis_data in deduplicated["theses"]:
        thesis = thesis_data["thesis"]
        thesis_num = thesis_to_number[thesis]
        supports_list = sorted(supports_map.get(thesis_num, []))
        supports_str = ", ".join(map(str, supports_list)) if supports_list else ""
        rows.append(
            {
                "Thesis Number": thesis_num,
                "Thesis Statement": thesis,
                "Title": thesis_titles.get(thesis, ""),
                "Description": thesis_data["description"],
                "Supports Thesis Numbers": supports_str,
                "Core Thesis": thesis_core_theses.get(thesis, ""),
                "Source File": thesis_data["source_file"],
            }
        )

    dataframe = pd.DataFrame(rows, columns=REVIEWED_CSV_COLUMNS)
    core_thesis_order = {
        value: index for index, value in enumerate(allowed_core_theses + (UNSORTED_CORE_THESIS,))
    }
    dataframe["_core_thesis_order"] = dataframe["Core Thesis"].map(
        lambda value: core_thesis_order.get(value, len(core_thesis_order))
    )
    dataframe = dataframe.sort_values(
        by=["_core_thesis_order", "Core Thesis", "Thesis Number"],
        kind="stable",
    ).drop(columns=["_core_thesis_order"])
    return dataframe.reset_index(drop=True)


def write_review_csv(review_dataframe: pd.DataFrame, output_file: Path) -> Path:
    """Persist the human-review CSV artifact."""
    output_file.parent.mkdir(exist_ok=True, parents=True)
    review_dataframe.to_csv(output_file, index=False)
    return output_file


def upload_review_csv_to_dropbox(
    settings: Settings,
    review_dataframe: pd.DataFrame,
    output_file: Path,
    *,
    extracted_at: datetime | None = None,
) -> str | None:
    """Upload the review CSV to Dropbox using a date-versioned filename."""
    if not settings.dropbox_review_output_path:
        return None

    extracted_at = extracted_at or datetime.now()
    stamp = extracted_at.strftime("%Y%m%d")
    versioned_name = f"{output_file.stem}_{stamp}{output_file.suffix}"
    destination_path = (
        settings.dropbox_review_output_path.rstrip("/") + "/" + versioned_name
    )
    content = review_dataframe.to_csv(index=False).encode("utf-8")
    uploaded_path = upload_bytes_to_dropbox(
        settings,
        destination_path=destination_path,
        content=content,
        overwrite=True,
    )
    print(f"Uploaded review CSV to Dropbox: {uploaded_path}")
    return uploaded_path


def run_extract_for_review_pipeline(
    *,
    settings: Settings | None = None,
    input_dir: Path | None = None,
    output_file: Path | None = None,
    model: str = DEFAULT_EXTRACTION_MODEL,
    title_model: str = DEFAULT_TITLE_MODEL,
) -> ExtractionRunResult:
    """Run the extraction workflow from raw documents to review CSV."""
    settings = settings or load_settings(override=True)
    openai_client = create_openai_client(settings)

    resolved_output_file = output_file

    print("Loading source artifacts...")
    document_char_counts: dict[str, int] = {}
    all_extractions: dict[str, dict[str, Any]] = {}
    for artifact in iter_source_artifacts(settings, input_dir=input_dir):
        parsed_document = parse_document_artifact(artifact)
        document_char_counts[parsed_document.name] = len(parsed_document.text)
        print(f"Prepared document: {parsed_document.name} ({len(parsed_document.text):,} characters)")
        print(f"\n{'=' * 60}")
        print(f"Extracting theses from: {parsed_document.name}")
        print(f"{'=' * 60}")
        extraction = extract_theses_from_text(
            parsed_document.text,
            parsed_document.name,
            openai_client,
            model=model,
        )
        all_extractions[parsed_document.name] = extraction
        print(
            f"  Found {len(extraction.get('theses', []))} theses, "
            f"{len(extraction.get('thesis_supports', []))} support relationships"
        )

    print(f"\nLoaded {len(document_char_counts)} document(s)")
    print(f"\nExtraction complete for {len(all_extractions)} document(s)")

    print("Deduplicating theses...")
    deduplicated = deduplicate_theses(all_extractions)
    total_extracted = sum(len(item.get("theses", [])) for item in all_extractions.values())
    print(f"Found {len(deduplicated['theses'])} unique theses (from {total_extracted} total)")
    print(f"Found {len(deduplicated['thesis_supports'])} support relationships")

    thesis_titles = generate_titles_for_deduplicated_theses(
        deduplicated,
        openai_client,
        model=title_model,
    )
    thesis_core_theses = classify_core_theses_for_deduplicated_theses(
        deduplicated,
        openai_client,
        settings.core_theses,
    )
    review_dataframe = create_review_dataframe(
        deduplicated,
        thesis_titles,
        thesis_core_theses,
        settings.core_theses,
    )
    review_csv_path = (
        write_review_csv(review_dataframe, resolved_output_file)
        if resolved_output_file is not None
        else None
    )
    dropbox_review_csv_path = upload_review_csv_to_dropbox(
        settings,
        review_dataframe,
        resolved_output_file or Path(DEFAULT_REVIEW_FILENAME),
    )

    if review_csv_path is not None:
        print(f"\nCreated local review CSV: {review_csv_path}")
    if dropbox_review_csv_path:
        print(f"Dropbox review CSV: {dropbox_review_csv_path}")
    if review_csv_path is None and dropbox_review_csv_path is None:
        print("\nWARNING: No review artifact was persisted.")
    print("\nCSV Summary:")
    print(f"  Total theses: {len(review_dataframe)}")
    print(
        "  Theses with support relationships: "
        f"{len(review_dataframe[review_dataframe['Supports Thesis Numbers'] != ''])}"
    )

    return ExtractionRunResult(
        document_char_counts=document_char_counts,
        all_extractions=all_extractions,
        deduplicated=deduplicated,
        review_dataframe=review_dataframe,
        review_csv_path=review_csv_path,
        dropbox_review_csv_path=dropbox_review_csv_path,
    )


def _parse_json_response(result_text: str) -> dict[str, Any]:
    if "```json" in result_text:
        result_text = result_text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0]

    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
    if json_match:
        result_text = json_match.group(0)

    return json.loads(result_text.strip())
