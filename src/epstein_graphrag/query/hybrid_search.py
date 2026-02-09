"""Combined graph traversal + vector similarity search.

Fires both search modes simultaneously and merges results
with contextual graph expansion.
"""

import logging

from neo4j import GraphDatabase

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


class HybridSearcher:
    """Combines graph traversal and vector search."""

    def __init__(self, config: Config):
        self.config = config
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def search(self, query: str, top_k: int | None = None) -> dict:
        """Execute a hybrid search combining graph and vector results.

        Args:
            query: Search query string.
            top_k: Number of vector results to retrieve.

        Returns:
            Merged results from both search modes.
        """
        raise NotImplementedError("Hybrid search not yet implemented — see Task 8")

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()
