# thesis-bot

`thesis-bot` extracts investment theses from source documents, routes them through a human review step, and loads the reviewed results into Neo4j.

Today, the CLI workflow is Dropbox-first:

- Source documents are read from Dropbox
- The review CSV is written to Dropbox
- The reviewed CSV is read back from Dropbox
- The final graph is written to Neo4j

## What This Repo Does

The main workflow has two pipeline stages:

1. `extract-theses`
   Reads supported source documents from Dropbox, extracts theses with OpenAI, deduplicates them, generates short titles, assigns a core thesis, and uploads a CSV for human review.

2. `load-theses`
   Reads the reviewed CSV from Dropbox, validates it, fills in any missing titles, generates embeddings, and loads the result into Neo4j.

There is also a helper command:

- `list-dropbox`
  Lists Dropbox folders and files so you can verify the exact API-visible paths used by the other commands.

## Supported Source Files

The extraction pipeline supports these source document types:

- `.pdf`
- `.docx`
- `.md`
- `.txt`

## Project Layout

Important paths:

- [src/thesis_bot/cli.py](/Users/ck-mac/Code/thesis-bot/src/thesis_bot/cli.py:1): CLI entrypoint
- [src/thesis_bot/pipelines/extract_for_review.py](/Users/ck-mac/Code/thesis-bot/src/thesis_bot/pipelines/extract_for_review.py:462): extraction pipeline
- [src/thesis_bot/pipelines/load_reviewed_theses.py](/Users/ck-mac/Code/thesis-bot/src/thesis_bot/pipelines/load_reviewed_theses.py:263): Neo4j load pipeline
- [src/thesis_bot/config.py](/Users/ck-mac/Code/thesis-bot/src/thesis_bot/config.py:1): environment-driven settings
- [notebooks/extract_theses_for_review.ipynb](/Users/ck-mac/Code/thesis-bot/notebooks/extract_theses_for_review.ipynb)
- [notebooks/load_theses_to_neo4j.ipynb](/Users/ck-mac/Code/thesis-bot/notebooks/load_theses_to_neo4j.ipynb)
- [notebooks/analyze_pitchdeck_alignment.ipynb](/Users/ck-mac/Code/thesis-bot/notebooks/analyze_pitchdeck_alignment.ipynb)

## Setup

From the repo root:

```bash
uv sync
```

You can then run the CLI with either form:

```bash
uv run python -m thesis_bot.cli --help
uv run python -m thesis_bot --help
```

## Environment Variables

Create a `.env` file in the repo root.

### Required For Extraction

```env
OPENAI_API_KEY=...
ARTIFACT_SOURCE=dropbox
DROPBOX_ACCESS_TOKEN=...
DROPBOX_THESIS_SOURCE_PATH=/path/to/source/documents
DROPBOX_REVIEW_OUTPUT_PATH=/path/to/output/folder
CORE_THESES=AI,BioTech,ConTech,Investment Criteria
```

### Required For Neo4j Load

```env
OPENAI_API_KEY=...
DROPBOX_ACCESS_TOKEN=...
DROPBOX_REVIEWED_THESES_PATH=/path/to/reviewed/theses.csv
NEO4J_URI=neo4j+s://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
CORE_THESES=AI,BioTech,ConTech,Investment Criteria
```

Notes:

- `ARTIFACT_SOURCE` must currently be `dropbox` for `extract-theses`
- `DROPBOX_REVIEWED_THESES_PATH` is currently required for `load-theses`
- `CORE_THESES` is a comma-separated list of allowed bucket names for the human review step

## CLI Reference

Top-level help:

```bash
uv run python -m thesis_bot.cli --help
```

Output:

```text
usage: thesis-bot [-h] {extract-theses,load-theses,list-dropbox} ...
```

### `list-dropbox`

Use this first to verify the exact Dropbox paths you want the pipelines to use.

```bash
uv run python -m thesis_bot.cli list-dropbox --path '/10. Proprietary'
uv run python -m thesis_bot.cli list-dropbox --path '/10. Proprietary' --recursive
```

### `extract-theses`

Extract theses from the configured Dropbox source and upload a review CSV back to Dropbox.

```bash
uv run python -m thesis_bot.cli extract-theses
```

Optional model overrides:

```bash
uv run python -m thesis_bot.cli extract-theses \
  --model gpt-4-turbo-preview \
  --title-model gpt-4o-mini
```

What it does:

- lists and downloads supported documents from `DROPBOX_THESIS_SOURCE_PATH`
- extracts thesis candidates with OpenAI
- deduplicates the extracted theses
- generates short titles
- assigns each row to a core thesis bucket
- uploads a timestamped review CSV to `DROPBOX_REVIEW_OUTPUT_PATH`

### `load-theses`

Load a reviewed CSV from Dropbox into Neo4j.

```bash
uv run python -m thesis_bot.cli load-theses
```

Optional flags:

```bash
uv run python -m thesis_bot.cli load-theses --keep-existing
```

```bash
uv run python -m thesis_bot.cli load-theses \
  --embedding-model text-embedding-3-small \
  --title-model gpt-4o-mini
```

What it does:

- downloads the reviewed CSV from `DROPBOX_REVIEWED_THESES_PATH`
- validates the required review schema
- backfills any missing titles
- generates embeddings for descriptions
- clears Neo4j unless `--keep-existing` is set
- creates `CoreThesis` and `Thesis` nodes plus `SUPPORTS` relationships

## Sequential Workflow

This is the normal end-to-end flow.

### 1. Install dependencies

```bash
uv sync
```

### 2. Verify your Dropbox source folder

```bash
uv run python -m thesis_bot.cli list-dropbox --path '/10. Proprietary' --recursive
```

Use the real folder you want to process. Confirm that the documents you expect are visible through the API.

### 3. Set extraction paths in `.env`

Example:

```env
ARTIFACT_SOURCE=dropbox
DROPBOX_THESIS_SOURCE_PATH=/10. Proprietary/Thesis Decks
DROPBOX_REVIEW_OUTPUT_PATH=/10. Proprietary/Analysis
```

### 4. Run extraction

```bash
uv run python -m thesis_bot.cli extract-theses
```

Expected result:

- a timestamped CSV is uploaded to your Dropbox review output folder
- the CLI prints the uploaded path

### 5. Review the CSV manually

Open the generated CSV and review these columns:

- `Title`
- `Description`
- `Supports Thesis Numbers`
- `Core Thesis`

The reviewed CSV must contain these columns:

- `Thesis Number`
- `Thesis Statement`
- `Title`
- `Description`
- `Supports Thesis Numbers`
- `Core Thesis`
- `Source File`

### 6. Point the loader at the reviewed CSV

Set:

```env
DROPBOX_REVIEWED_THESES_PATH=/10. Proprietary/Analysis/theses_for_review_reviewed.csv
```

Use the real Dropbox path to the reviewed file.

### 7. Load into Neo4j

```bash
uv run python -m thesis_bot.cli load-theses
```

Expected result:

- the reviewed CSV is validated
- embeddings are generated
- Neo4j is populated with thesis nodes and relationships
- the CLI prints graph counts

### 8. Re-run safely when needed

If you want to append without clearing the existing graph first:

```bash
uv run python -m thesis_bot.cli load-theses --keep-existing
```

## Copy-Paste Example Session

This is what a typical run might look like:

```bash
uv sync

uv run python -m thesis_bot.cli list-dropbox --path '/10. Proprietary/Thesis Decks' --recursive

uv run python -m thesis_bot.cli extract-theses

# Review the generated CSV in Dropbox, then update DROPBOX_REVIEWED_THESES_PATH

uv run python -m thesis_bot.cli load-theses
```

## Notebooks

The notebooks mirror the same workflow in a more exploratory format:

- `notebooks/extract_theses_for_review.ipynb`
- `notebooks/load_theses_to_neo4j.ipynb`
- `notebooks/analyze_pitchdeck_alignment.ipynb`

If you want the notebook UI:

```bash
uv run jupyter lab
```

## Current Limitations

- CLI extraction currently supports Dropbox as the only artifact source
- CLI loading currently expects the reviewed CSV in Dropbox
- `load-theses` clears the Neo4j database by default unless you pass `--keep-existing`

## Troubleshooting

### Dropbox authentication failed

Check:

- `DROPBOX_ACCESS_TOKEN`
- the exact Dropbox path passed to `list-dropbox`

### No documents found during extraction

Check:

- `DROPBOX_THESIS_SOURCE_PATH`
- that the folder contains supported file types
- that the files are visible through `list-dropbox --recursive`

### Loader says no reviewed CSV source configured

Set:

```env
DROPBOX_REVIEWED_THESES_PATH=/your/reviewed/file.csv
```

### Neo4j connection fails

Check:

- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`

## Development Notes

Entry points:

```bash
uv run python -m thesis_bot.cli --help
uv run python -m thesis_bot --help
```

There is also a placeholder [main.py](/Users/ck-mac/Code/thesis-bot/main.py:1), but the real application entrypoint is the package CLI under `src/thesis_bot`.
