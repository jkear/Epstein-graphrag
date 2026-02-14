"""Tests for Neo4j schema creation and verification.

Tests cover:
- Schema constants are consistent
- create_schema is idempotent (safe to run multiple times)
- verify_schema returns expected counts
"""

import pytest

from epstein_graphrag.config import Config
from epstein_graphrag.graph.schema import (
    FULLTEXT_INDEXES,
    NODE_KEY_CONSTRAINTS,
    VECTOR_INDEXES,
    create_schema,
    verify_schema,
)

# ------------------------------------------------------------------ #
#  Unit tests (no Neo4j required)                                     #
# ------------------------------------------------------------------ #


def test_node_key_constraints_count():
    """We expect 18 NODE KEY constraints."""
    assert len(NODE_KEY_CONSTRAINTS) == 18


def test_vector_indexes_count():
    """We expect 6 vector indexes."""
    assert len(VECTOR_INDEXES) == 6


def test_fulltext_indexes_count():
    """We expect 4 fulltext indexes."""
    assert len(FULLTEXT_INDEXES) == 4


def test_all_vector_indexes_are_768d():
    """All vector indexes use 768 dimensions (nomic-embed-text)."""
    for name, label, prop, dim in VECTOR_INDEXES:
        assert dim == 768, f"{name} has {dim} dimensions, expected 768"


def test_constraint_labels_unique():
    """Each label has exactly one constraint."""
    labels = [label for label, _ in NODE_KEY_CONSTRAINTS]
    assert len(labels) == len(set(labels)), "Duplicate labels in constraints"


def test_core_labels_have_constraints():
    """Core entity labels used by ingest.py have constraints."""
    required = {"Person", "Organization", "Location", "Document", "Event", "Allegation"}
    constrained = {label for label, _ in NODE_KEY_CONSTRAINTS}
    missing = required - constrained
    assert not missing, f"Missing constraints for: {missing}"


# ------------------------------------------------------------------ #
#  Integration tests — require a running Neo4j instance               #
# ------------------------------------------------------------------ #


def _neo4j_available() -> bool:
    """Check if Neo4j is reachable."""
    from neo4j import GraphDatabase

    config = Config()
    try:
        driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


@pytest.fixture
def config():
    """Provide a Config, skip if Neo4j is unavailable."""
    cfg = Config()
    if not _neo4j_available():
        pytest.skip("Neo4j not available")
    return cfg


def test_create_schema_idempotent(config):
    """Running create_schema twice produces no errors."""
    create_schema(config)
    stats2 = create_schema(config)

    # Second run should have 0 created, all existing
    assert stats2["constraints_created"] == 0
    assert stats2["vector_indexes_created"] == 0
    assert stats2["fulltext_indexes_created"] == 0

    total_constraints = stats2["constraints_created"] + stats2["constraints_existing"]
    assert total_constraints == 18


def test_verify_schema_expected_counts(config):
    """verify_schema reports the expected number of constraints and indexes."""
    # Ensure schema exists
    create_schema(config)

    result = verify_schema(config)

    assert result["constraints"] == 18
    assert result["vector_indexes"] == 6
    assert result["fulltext_indexes"] == 4


def test_verify_schema_constraint_names(config):
    """verify_schema returns recognizable constraint names."""
    create_schema(config)
    result = verify_schema(config)

    names = result["constraint_names"]
    # Check that core constraints exist (name may vary if pre-existing)
    assert any("Person" in n for n in names), "No Person constraint found"
    assert any("Document" in n for n in names), "No Document constraint found"


def test_verify_schema_vector_index_names(config):
    """verify_schema returns our vector index names."""
    create_schema(config)
    result = verify_schema(config)

    expected = {name for name, _, _, _ in VECTOR_INDEXES}
    found = set(result["vector_index_names"])
    missing = expected - found
    assert not missing, f"Missing vector indexes: {missing}"


def test_verify_schema_fulltext_index_names(config):
    """verify_schema returns our fulltext index names."""
    create_schema(config)
    result = verify_schema(config)

    expected = {name for name, _, _ in FULLTEXT_INDEXES}
    found = set(result["fulltext_index_names"])
    missing = expected - found
    assert not missing, f"Missing fulltext indexes: {missing}"
