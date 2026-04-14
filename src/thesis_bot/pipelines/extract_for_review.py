from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from pydantic import Field

from thesis_bot.config import Settings, load_settings
from thesis_bot.config import UNSORTED_CORE_THESIS
from thesis_bot.clients.openai_client import create_openai_client
from thesis_bot.io.document_parsers import parse_document_artifact
from thesis_bot.io.review_runs import build_review_run_id
from thesis_bot.io.review_runs import write_review_bucket_csvs
from thesis_bot.io.source_loader import iter_source_artifacts
from thesis_bot.schemas import REVIEWED_CSV_COLUMNS


DEFAULT_EXTRACTION_MODEL = "gpt-4.1"
DEFAULT_TITLE_MODEL = "gpt-4o-mini"
MAX_DOCUMENT_CHARS = 470000
CHUNK_OVERLAP_CHARS = 10000
MAX_EXTRACTION_PARSE_ATTEMPTS = 3


class ThesisExtractionItem(BaseModel):
    thesis: str = Field(default="")
    description: str = Field(default="")


class ThesisSupportItem(BaseModel):
    source_thesis: str = Field(default="")
    target_thesis: str = Field(default="")


class ThesisExtractionPayload(BaseModel):
    theses: list[ThesisExtractionItem] = Field(default_factory=list)
    thesis_supports: list[ThesisSupportItem] = Field(default_factory=list)


@dataclass(frozen=True)
class ExtractionRunResult:
    run_id: str
    document_char_counts: dict[str, int]
    all_extractions: dict[str, dict[str, Any]]
    deduplicated: dict[str, Any]
    review_dataframe: pd.DataFrame
    review_dataframes_by_core_thesis: dict[str, pd.DataFrame]
    review_output_paths_by_core_thesis: dict[str, str]

    @property
    def pdf_texts(self) -> dict[str, str]:
        """Deprecated compatibility shim for older notebook code."""
        return {name: "" for name in self.document_char_counts}

    @property
    def dropbox_review_csv_path(self) -> str | None:
        """Deprecated compatibility shim for older notebook code."""
        return None


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

    print(f"  Analyzing document ({len(text):,} characters)...")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert at analyzing documents and extracting thesis "
                "statements. Return only valid JSON matching the requested schema. "
                "Be thorough and extract all thesis statements from the entire "
                "document, even if they seem similar."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    last_parse_error: json.JSONDecodeError | None = None
    last_result_text = ""
    for attempt in range(1, MAX_EXTRACTION_PARSE_ATTEMPTS + 1):
        try:
            result = _request_extraction_payload(
                openai_client,
                model=model,
                messages=messages,
            )
            result["filename"] = filename
            return result
        except json.JSONDecodeError as exc:
            last_parse_error = exc
            preview = _response_preview(last_result_text)
            print(
                f"  Failed to parse JSON for {filename} on attempt {attempt}/"
                f"{MAX_EXTRACTION_PARSE_ATTEMPTS}: {exc}"
            )
            if preview:
                print(f"  Response preview: {preview}")
            if attempt < MAX_EXTRACTION_PARSE_ATTEMPTS:
                messages = _build_retry_messages(prompt, last_result_text, exc)
            continue
        except Exception as exc:
            print(f"  Error analyzing {filename}: {exc}")
            return {"theses": [], "thesis_supports": [], "filename": filename}

    if last_parse_error is not None:
        print(f"  Giving up on {filename} after {MAX_EXTRACTION_PARSE_ATTEMPTS} parse attempts.")
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
    ordered_core_theses = list(allowed_core_theses)
    if UNSORTED_CORE_THESIS not in ordered_core_theses:
        ordered_core_theses.append(UNSORTED_CORE_THESIS)
    core_thesis_order = {
        value: index for index, value in enumerate(ordered_core_theses)
    }
    dataframe["_core_thesis_order"] = dataframe["Core Thesis"].map(
        lambda value: core_thesis_order.get(value, len(core_thesis_order))
    )
    dataframe = dataframe.sort_values(
        by=["_core_thesis_order", "Core Thesis", "Thesis Number"],
        kind="stable",
    ).drop(columns=["_core_thesis_order"])
    return dataframe.reset_index(drop=True)


def split_review_dataframe_by_core_thesis(
    review_dataframe: pd.DataFrame,
    core_theses: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    bucketed: dict[str, pd.DataFrame] = {}
    empty_template = review_dataframe.iloc[0:0].copy()
    for core_thesis in core_theses:
        bucket_frame = review_dataframe[review_dataframe["Core Thesis"] == core_thesis].copy()
        bucketed[core_thesis] = bucket_frame if not bucket_frame.empty else empty_template.copy()
    return bucketed


def summarize_review_outputs(
    review_output_paths_by_core_thesis: dict[str, str],
    review_dataframe: pd.DataFrame,
) -> None:
    print("\nReview bucket outputs:")
    for core_thesis, output_path in review_output_paths_by_core_thesis.items():
        count = len(review_dataframe[review_dataframe["Core Thesis"] == core_thesis])
        print(f"  {core_thesis}: {count} rows -> {output_path}")


def run_extract_for_review_pipeline(
    *,
    settings: Settings | None = None,
    model: str = DEFAULT_EXTRACTION_MODEL,
    title_model: str = DEFAULT_TITLE_MODEL,
) -> ExtractionRunResult:
    """Run the extraction workflow from the configured source to per-bucket review CSVs."""
    settings = settings or load_settings(override=True)
    openai_client = create_openai_client(settings)
    run_id = build_review_run_id(created_at=datetime.now())

    print("Loading source artifacts...")
    document_char_counts: dict[str, int] = {}
    all_extractions: dict[str, dict[str, Any]] = {}
    for artifact in iter_source_artifacts(settings):
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
    review_dataframes_by_core_thesis = split_review_dataframe_by_core_thesis(
        review_dataframe,
        settings.core_theses,
    )
    review_output_paths_by_core_thesis = write_review_bucket_csvs(
        settings,
        review_dataframes_by_core_thesis,
        run_id=run_id,
    )
    summarize_review_outputs(review_output_paths_by_core_thesis, review_dataframe)
    print("\nCSV Summary:")
    print(f"  Total theses: {len(review_dataframe)}")
    print(
        "  Theses with support relationships: "
        f"{len(review_dataframe[review_dataframe['Supports Thesis Numbers'] != ''])}"
    )

    return ExtractionRunResult(
        run_id=run_id,
        document_char_counts=document_char_counts,
        all_extractions=all_extractions,
        deduplicated=deduplicated,
        review_dataframe=review_dataframe,
        review_dataframes_by_core_thesis=review_dataframes_by_core_thesis,
        review_output_paths_by_core_thesis=review_output_paths_by_core_thesis,
    )


def _parse_json_response(result_text: str) -> dict[str, Any]:
    result_text = _strip_markdown_fences(result_text)
    candidate_texts = _candidate_json_payloads(result_text)
    last_error: json.JSONDecodeError | None = None
    for candidate in candidate_texts:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return json.loads(result_text.strip())


def _request_extraction_payload(
    openai_client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        completion = openai_client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            response_format=ThesisExtractionPayload,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Structured extraction response was empty.")
        return _structured_payload_to_dict(parsed)
    except Exception as structured_exc:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        result_text = response.choices[0].message.content or ""
        try:
            return _coerce_extraction_payload(_parse_json_response(result_text))
        except json.JSONDecodeError:
            raise
        except Exception as fallback_exc:
            raise ValueError(
                f"Structured parse failed ({structured_exc}) and fallback parse failed ({fallback_exc})."
            ) from fallback_exc


def _build_retry_messages(
    original_prompt: str,
    invalid_result_text: str,
    parse_error: json.JSONDecodeError,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You repair malformed JSON outputs for a thesis extraction workflow. "
                "Return only valid JSON with top-level keys 'theses' and "
                "'thesis_supports'. Do not add commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The previous response was invalid JSON.\n"
                f"Parsing error: {parse_error}\n\n"
                f"Original extraction instructions:\n{original_prompt}\n\n"
                f"Invalid JSON response to repair:\n{invalid_result_text}"
            ),
        },
    ]


def _coerce_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    theses = payload.get("theses", [])
    thesis_supports = payload.get("thesis_supports", [])
    if not isinstance(theses, list):
        theses = []
    if not isinstance(thesis_supports, list):
        thesis_supports = []
    return {
        "theses": [item for item in theses if isinstance(item, dict)],
        "thesis_supports": [item for item in thesis_supports if isinstance(item, dict)],
    }


def _structured_payload_to_dict(payload: ThesisExtractionPayload) -> dict[str, Any]:
    return {
        "theses": [
            {
                "thesis": item.thesis.strip(),
                "description": item.description.strip(),
            }
            for item in payload.theses
            if item.thesis.strip()
        ],
        "thesis_supports": [
            {
                "source_thesis": item.source_thesis.strip(),
                "target_thesis": item.target_thesis.strip(),
            }
            for item in payload.thesis_supports
            if item.source_thesis.strip() and item.target_thesis.strip()
        ],
    }


def _strip_markdown_fences(result_text: str) -> str:
    if "```json" in result_text:
        result_text = result_text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in result_text:
        result_text = result_text.split("```", 1)[1].split("```", 1)[0]
    return result_text


def _candidate_json_payloads(result_text: str) -> list[str]:
    candidates: list[str] = []
    stripped = result_text.strip()
    if stripped:
        candidates.append(stripped)

    json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
    if json_match:
        matched = json_match.group(0).strip()
        if matched and matched not in candidates:
            candidates.append(matched)

    balanced = _extract_balanced_json_object(result_text)
    if balanced and balanced not in candidates:
        candidates.append(balanced)

    return candidates


def _extract_balanced_json_object(result_text: str) -> str:
    start = result_text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(result_text)):
        char = result_text[index]
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return result_text[start : index + 1].strip()
    return ""


def _response_preview(result_text: str, *, max_chars: int = 300) -> str:
    preview = re.sub(r"\s+", " ", (result_text or "").strip())
    if not preview:
        return ""
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars] + "..."
