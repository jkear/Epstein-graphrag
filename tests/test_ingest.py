"""Tests for graph ingestion."""

import pytest

from epstein_graphrag.config import Config


def test_graph_ingestor_not_implemented():
    """Graph ingestion raises NotImplementedError until Task 6."""
    from epstein_graphrag.graph.ingest import GraphIngestor

    # Skip if Neo4j is not available
    config = Config()
    try:
        ingestor = GraphIngestor(config)
    except Exception:
        pytest.skip("Neo4j not available")

    try:
        with pytest.raises(NotImplementedError):
            ingestor.ingest_document({})
    finally:
        ingestor.close()
