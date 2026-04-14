from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from neo4j import Driver

from thesis_bot.clients.neo4j_client import create_neo4j_driver
from thesis_bot.clients.openai_client import create_openai_client
from thesis_bot.config import Settings, load_settings
from thesis_bot.io.review_runs import default_review_run_path
from thesis_bot.io.review_runs import expected_review_bucket_paths
from thesis_bot.io.review_runs import read_review_bucket_dataframes
from thesis_bot.pipelines.extract_for_review import (
    DEFAULT_TITLE_MODEL,
    generate_4word_title,
)
from thesis_bot.schemas import validate_reviewed_theses_dataframe


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass(frozen=True)
class GraphLoadStats:
    node_counts: dict[str, int]
    relationship_counts: dict[str, int]
    core_thesis_support_counts: dict[str, int]


@dataclass(frozen=True)
class LoadReviewedThesesResult:
    reviewed_run_path: str
    reviewed_dataframe: pd.DataFrame
    reviewed_dataframes_by_core_thesis: dict[str, pd.DataFrame]
    reviewed_bucket_paths: dict[str, str]
    thesis_titles: dict[int, str]
    embeddings: dict[int, list[float]]
    core_thesis_count: int
    thesis_node_count: int
    supports_relationship_count: int
    stats: GraphLoadStats


def load_reviewed_dataframe(
    *,
    settings: Settings | None = None,
    allow_missing_title: bool = True,
    run_path: str | None = None,
) -> pd.DataFrame:
    """Load and validate the reviewed thesis CSVs from the configured review run."""
    settings = settings or load_settings(override=True)
    bucket_dataframes = read_review_bucket_dataframes(
        settings,
        allow_missing_title=allow_missing_title,
        run_path=run_path,
    )
    combined = pd.concat(bucket_dataframes.values(), ignore_index=True)
    return validate_reviewed_theses_dataframe(
        combined,
        allowed_core_theses=settings.core_theses,
        allow_missing_title=allow_missing_title,
    )


def load_reviewed_bucket_dataframes(
    *,
    settings: Settings | None = None,
    allow_missing_title: bool = True,
    run_path: str | None = None,
) -> tuple[str, dict[str, pd.DataFrame], dict[str, str]]:
    settings = settings or load_settings(override=True)
    resolved_run_path = str(run_path or default_review_run_path(settings))
    bucket_paths = {
        core_thesis: str(path)
        for core_thesis, path in expected_review_bucket_paths(settings, run_path=resolved_run_path).items()
    }
    bucket_dataframes = read_review_bucket_dataframes(
        settings,
        allow_missing_title=allow_missing_title,
        run_path=resolved_run_path,
    )
    return resolved_run_path, bucket_dataframes, bucket_paths


def generate_embedding(
    text: str,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Generate an embedding for a thesis description."""
    if not openai_client:
        return []
    try:
        response = openai_client.embeddings.create(model=model, input=text)
        return response.data[0].embedding
    except Exception as exc:
        print(f"  Failed to generate embedding: {exc}")
        return []


def generate_embeddings_for_dataframe(
    dataframe: pd.DataFrame,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[int, list[float]]:
    """Generate embeddings for every thesis description."""
    print("Generating embeddings for thesis descriptions...")
    embeddings: dict[int, list[float]] = {}
    for _, row in dataframe.iterrows():
        thesis_num = int(row["Thesis Number"])
        description = row["Description"]
        print(f"  Generating embedding for thesis {thesis_num}...")
        embedding = generate_embedding(description, openai_client, model=model)
        embeddings[thesis_num] = embedding
        if embedding:
            print(f"    Generated {len(embedding)}-dimensional embedding")
        else:
            print("    Embedding unavailable")
    print(f"\nGenerated {len(embeddings)} embeddings")
    return embeddings


def ensure_titles_for_dataframe(
    dataframe: pd.DataFrame,
    openai_client: OpenAI | None,
    *,
    model: str = DEFAULT_TITLE_MODEL,
) -> pd.DataFrame:
    """Backfill missing thesis titles before Neo4j load."""
    normalized = dataframe.copy()
    missing_mask = normalized["Title"].fillna("").astype(str).str.strip().eq("")
    if not missing_mask.any():
        return normalized

    print(f"Generating titles for {int(missing_mask.sum())} thesis row(s) with missing titles...")
    for index in normalized.index[missing_mask]:
        thesis_num = int(normalized.at[index, "Thesis Number"])
        thesis_statement = str(normalized.at[index, "Thesis Statement"])
        print(f"  Generating title for thesis {thesis_num}...")
        normalized.at[index, "Title"] = generate_4word_title(
            thesis_statement,
            openai_client,
            model=model,
        )
        print(f"    {normalized.at[index, 'Title']}")

    return normalized


def clear_neo4j_database(driver: Driver) -> None:
    """Clear all nodes and relationships from Neo4j."""
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("Cleared Neo4j database")


def create_neo4j_thesis_graph(
    driver: Driver,
    dataframe: pd.DataFrame,
    embeddings: dict[int, list[float]],
    *,
    clear_existing: bool = True,
) -> tuple[int, int, int]:
    """Create Thesis and CoreThesis nodes and SUPPORTS relationships."""
    current_time = datetime.now().isoformat()

    with driver.session() as session:
        if clear_existing:
            clear_neo4j_database(driver)

        core_thesis_values = {
            row["Core Thesis"]
            for _, row in dataframe.iterrows()
            if row["Core Thesis"]
        }

        print(f"\nCreating {len(core_thesis_values)} CoreThesis nodes...")
        for core_thesis in sorted(core_thesis_values):
            session.run(
                """
                MERGE (ct:CoreThesis {core_thesis: $core_thesis})
                SET ct.created = $created,
                    ct.updated = $updated
                """,
                core_thesis=core_thesis,
                created=current_time,
                updated=current_time,
            )

        print(f"\nCreating {len(dataframe)} Thesis nodes...")
        thesis_node_count = 0
        for _, row in dataframe.iterrows():
            thesis_node_count += 1
            thesis_num = int(row["Thesis Number"])
            thesis = row["Thesis Statement"]
            title = row["Title"]
            description = row["Description"]
            core_thesis = row["Core Thesis"]
            embedding = embeddings.get(thesis_num, [])
            is_core = thesis == core_thesis

            session.run(
                """
                CREATE (t:Thesis {
                    thesis: $thesis,
                    title: $title,
                    node_num: $node_num,
                    description: $description,
                    description_embedding: $embedding,
                    core_thesis: $core_thesis,
                    core_thesis_flag: $is_core,
                    created: $created,
                    updated: $updated
                })
                """,
                thesis=thesis,
                title=title,
                node_num=thesis_num,
                description=description,
                embedding=embedding,
                core_thesis=core_thesis,
                is_core=is_core,
                created=current_time,
                updated=current_time,
            )

        print("\nCreating SUPPORTS relationships...")
        relationship_count = 0
        for _, row in dataframe.iterrows():
            thesis = row["Thesis Statement"]
            core_thesis = row["Core Thesis"]
            session.run(
                """
                MATCH (t:Thesis {thesis: $thesis})
                MATCH (ct:CoreThesis {core_thesis: $core_thesis})
                MERGE (t)-[:SUPPORTS]->(ct)
                """,
                thesis=thesis,
                core_thesis=core_thesis,
            )
            relationship_count += 1

    return len(core_thesis_values), thesis_node_count, relationship_count


def query_neo4j_stats(driver: Driver) -> GraphLoadStats:
    """Query Neo4j for high-level graph statistics."""
    node_counts: dict[str, int] = {}
    relationship_counts: dict[str, int] = {}
    core_thesis_support_counts: dict[str, int] = {}

    with driver.session() as session:
        node_result = session.run(
            """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY count DESC
            """
        )
        for record in node_result:
            node_counts[record["label"]] = record["count"]

        relationship_result = session.run(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel_type, count(r) AS count
            ORDER BY count DESC
            """
        )
        for record in relationship_result:
            relationship_counts[record["rel_type"]] = record["count"]

        supports_result = session.run(
            """
            MATCH (t:Thesis)-[:SUPPORTS]->(ct:CoreThesis)
            RETURN ct.core_thesis AS core_thesis, count(t) AS thesis_count
            ORDER BY thesis_count DESC
            """
        )
        for record in supports_result:
            core_thesis_support_counts[record["core_thesis"]] = record["thesis_count"]

    return GraphLoadStats(
        node_counts=node_counts,
        relationship_counts=relationship_counts,
        core_thesis_support_counts=core_thesis_support_counts,
    )


def run_load_reviewed_theses_pipeline(
    *,
    settings: Settings | None = None,
    clear_existing: bool = True,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    title_model: str = DEFAULT_TITLE_MODEL,
) -> LoadReviewedThesesResult:
    """Load the reviewed thesis CSV run into Neo4j."""
    settings = settings or load_settings(override=True)
    if not settings.neo4j_configured:
        raise ValueError("Neo4j environment variables are not fully configured.")

    reviewed_run_path, reviewed_dataframes_by_core_thesis, reviewed_bucket_paths = (
        load_reviewed_bucket_dataframes(
            settings=settings,
            allow_missing_title=True,
        )
    )
    print(f"Loading reviewed thesis bucket CSVs from {settings.review_input_source}: {reviewed_run_path}")
    for core_thesis, bucket_path in reviewed_bucket_paths.items():
        print(f"  {core_thesis}: {bucket_path}")

    openai_client = create_openai_client(settings)
    reviewed_dataframe = pd.concat(
        reviewed_dataframes_by_core_thesis.values(),
        ignore_index=True,
    )
    reviewed_dataframe = ensure_titles_for_dataframe(
        reviewed_dataframe,
        openai_client,
        model=title_model,
    )
    reviewed_dataframe = validate_reviewed_theses_dataframe(
        reviewed_dataframe,
        allowed_core_theses=settings.core_theses,
    )
    print(f"Loaded {len(reviewed_dataframe)} reviewed theses")
    thesis_titles = {
        int(row["Thesis Number"]): row["Title"]
        for _, row in reviewed_dataframe.iterrows()
    }
    embeddings = generate_embeddings_for_dataframe(
        reviewed_dataframe,
        openai_client,
        model=embedding_model,
    )

    driver = create_neo4j_driver(settings)
    try:
        core_thesis_count, thesis_node_count, supports_relationship_count = (
            create_neo4j_thesis_graph(
                driver,
                reviewed_dataframe,
                embeddings,
                clear_existing=clear_existing,
            )
        )
        stats = query_neo4j_stats(driver)
    finally:
        driver.close()

    return LoadReviewedThesesResult(
        reviewed_run_path=reviewed_run_path,
        reviewed_dataframe=reviewed_dataframe,
        reviewed_dataframes_by_core_thesis=reviewed_dataframes_by_core_thesis,
        reviewed_bucket_paths=reviewed_bucket_paths,
        thesis_titles=thesis_titles,
        embeddings=embeddings,
        core_thesis_count=core_thesis_count,
        thesis_node_count=thesis_node_count,
        supports_relationship_count=supports_relationship_count,
        stats=stats,
    )
