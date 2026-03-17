from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from neo4j import Driver

from thesis_bot.clients.neo4j_client import create_neo4j_driver
from thesis_bot.clients.openai_client import create_openai_client
from thesis_bot.config import Settings, load_settings
from thesis_bot.io.review_csv import read_reviewed_theses_csv


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_REVIEWED_FILENAME = "theses_reviewed.csv"


@dataclass(frozen=True)
class GraphLoadStats:
    node_counts: dict[str, int]
    relationship_counts: dict[str, int]
    core_thesis_support_counts: dict[str, int]


@dataclass(frozen=True)
class LoadReviewedThesesResult:
    reviewed_dataframe: pd.DataFrame
    thesis_titles: dict[int, str]
    embeddings: dict[int, list[float]]
    core_thesis_count: int
    thesis_node_count: int
    supports_relationship_count: int
    stats: GraphLoadStats


def load_reviewed_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load and validate the reviewed thesis CSV."""
    settings = load_settings(override=True)
    return read_reviewed_theses_csv(
        csv_path,
        allowed_core_theses=settings.core_theses,
    )


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
    csv_path: Path | None = None,
    clear_existing: bool = True,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> LoadReviewedThesesResult:
    """Load the reviewed thesis CSV into Neo4j."""
    settings = settings or load_settings(override=True)
    if not settings.neo4j_configured:
        raise ValueError("Neo4j environment variables are not fully configured.")

    resolved_csv_path = csv_path or (settings.analysis_dir / DEFAULT_REVIEWED_FILENAME)
    if not resolved_csv_path.exists():
        raise FileNotFoundError(f"Reviewed thesis CSV not found: {resolved_csv_path}")

    print(f"Loading reviewed thesis CSV: {resolved_csv_path}")
    reviewed_dataframe = load_reviewed_dataframe(resolved_csv_path)
    print(f"Loaded {len(reviewed_dataframe)} reviewed theses")

    openai_client = create_openai_client(settings)
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
        reviewed_dataframe=reviewed_dataframe,
        thesis_titles=thesis_titles,
        embeddings=embeddings,
        core_thesis_count=core_thesis_count,
        thesis_node_count=thesis_node_count,
        supports_relationship_count=supports_relationship_count,
        stats=stats,
    )
