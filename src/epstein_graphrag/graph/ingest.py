"""Neo4j graph ingestion — MERGE-based, idempotent.

All writes use MERGE (never CREATE) to ensure idempotent ingestion.
Running the same batch twice produces the same graph state.
"""

import logging

from neo4j import GraphDatabase

from epstein_graphrag.config import Config
from epstein_graphrag.graph.dedup import AliasResolver

logger = logging.getLogger(__name__)


class GraphIngestor:
    """Ingests extracted entities and relationships into Neo4j."""

    def __init__(self, config: Config, alias_resolver: AliasResolver | None = None):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        self.alias_resolver = alias_resolver or AliasResolver(config)

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def ingest_document(self, extraction: dict) -> None:
        """Ingest a single document's extracted entities into Neo4j.

        Args:
            extraction: Dictionary with 'entities' and 'relationships' keys
                       as produced by entity_extractor.
        """
        raise NotImplementedError("Graph ingestion not yet implemented — see Task 6")

    def ingest_batch(self, extractions: list[dict]) -> int:
        """Ingest a batch of extractions into Neo4j.

        Args:
            extractions: List of extraction dictionaries.

        Returns:
            Number of successfully ingested documents.
        """
        raise NotImplementedError("Batch ingestion not yet implemented — see Task 6")
