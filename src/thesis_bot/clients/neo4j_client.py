from __future__ import annotations

from neo4j import Driver, GraphDatabase

from thesis_bot.config import Settings


def create_neo4j_driver(settings: Settings, *, verify: bool = True) -> Driver | None:
    if not settings.neo4j_configured:
        return None

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    if verify:
        driver.verify_connectivity()
    return driver
