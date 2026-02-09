"""Tests for query engine."""

import pytest

from epstein_graphrag.config import Config


def test_query_engine_not_implemented():
    """Query engine raises NotImplementedError until Task 8."""
    from epstein_graphrag.query.engine import QueryEngine

    config = Config()
    try:
        engine = QueryEngine(config)
    except Exception:
        pytest.skip("Neo4j not available")

    try:
        with pytest.raises(NotImplementedError):
            engine.query("Who visited the island?")
    finally:
        engine.close()
