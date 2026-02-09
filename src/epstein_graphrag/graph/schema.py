"""Neo4j schema creation and validation.

Creates constraints, vector indexes, and fulltext indexes.
Validates that the schema is correctly set up.
"""

import logging

from neo4j import GraphDatabase

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


def create_schema(config: Config) -> None:
    """Create all constraints and indexes in Neo4j.

    This is idempotent — uses IF NOT EXISTS on all operations.

    Args:
        config: Application configuration with Neo4j credentials.
    """
    raise NotImplementedError("Schema creation not yet implemented — see Task 1")


def verify_schema(config: Config) -> dict:
    """Verify that the Neo4j schema is correctly set up.

    Args:
        config: Application configuration with Neo4j credentials.

    Returns:
        Dictionary with counts of constraints and indexes found.
    """
    raise NotImplementedError("Schema verification not yet implemented — see Task 1")
