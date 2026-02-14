"""Tests for epstein_graphrag.embeddings.embed.

Unit tests mock Ollama (no model required).
Integration tests need Neo4j running + nomic-embed-text via Ollama.
"""

from unittest.mock import MagicMock, patch

import pytest

from epstein_graphrag.config import Config
from epstein_graphrag.embeddings.embed import (
    EMBEDDING_TARGETS,
    EmbeddingTarget,
    NodeEmbedder,
    embed_batch,
    embed_text,
)

# ------------------------------------------------------------------ #
#  Unit tests (no Ollama or Neo4j required)                           #
# ------------------------------------------------------------------ #


class TestEmbedText:
    """Tests for the embed_text() helper."""

    @patch("epstein_graphrag.embeddings.embed.ollama")
    def test_returns_first_embedding(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_ollama.embed.return_value = mock_response

        result = embed_text("hello world")
        assert result == [0.1, 0.2, 0.3]
        mock_ollama.embed.assert_called_once_with(
            model="nomic-embed-text", input="hello world"
        )

    @patch("epstein_graphrag.embeddings.embed.ollama")
    def test_uses_config_model(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1]]
        mock_ollama.embed.return_value = mock_response

        config = Config()
        config.embedding_model = "custom-model"
        embed_text("test", config=config)
        mock_ollama.embed.assert_called_once_with(
            model="custom-model", input="test"
        )


class TestEmbedBatch:
    """Tests for the embed_batch() helper."""

    @patch("epstein_graphrag.embeddings.embed.ollama")
    def test_returns_all_embeddings(self, mock_ollama):
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1], [0.2], [0.3]]
        mock_ollama.embed.return_value = mock_response

        result = embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == [0.1]

    @patch("epstein_graphrag.embeddings.embed.ollama")
    def test_empty_list(self, mock_ollama):
        result = embed_batch([])
        assert result == []
        mock_ollama.embed.assert_not_called()


class TestEmbeddingTargets:
    """Tests for the EMBEDDING_TARGETS configuration."""

    def test_four_targets_defined(self):
        assert len(EMBEDDING_TARGETS) == 4

    def test_target_labels(self):
        labels = {t.label for t in EMBEDDING_TARGETS}
        assert labels == {"Event", "Allegation", "Person", "Location"}

    def test_event_target_properties(self):
        event = next(t for t in EMBEDDING_TARGETS if t.label == "Event")
        assert event.embedding_property == "description_embedding"
        assert event.text_property == "description"
        assert event.key_property == "event_id"

    def test_allegation_uses_embedding_property(self):
        alleg = next(t for t in EMBEDDING_TARGETS if t.label == "Allegation")
        assert alleg.embedding_property == "embedding"

    def test_person_uses_name_description_builder(self):
        person = next(t for t in EMBEDDING_TARGETS if t.label == "Person")
        assert person.text_builder == "name_description"

    def test_location_uses_name_description_builder(self):
        loc = next(t for t in EMBEDDING_TARGETS if t.label == "Location")
        assert loc.text_builder == "name_description"


class TestNodeEmbedderBuildText:
    """Tests for NodeEmbedder._build_text (unit, no connections)."""

    def test_single_builder(self):
        target = EmbeddingTarget(
            label="Event",
            embedding_property="description_embedding",
            text_property="description",
            text_builder="single",
            key_property="event_id",
        )
        # Use a minimal mock for NodeEmbedder to test _build_text
        embedder = NodeEmbedder.__new__(NodeEmbedder)
        node = {"text": "Something happened"}
        assert embedder._build_text(node, target) == "Something happened"

    def test_name_description_builder(self):
        target = EmbeddingTarget(
            label="Person",
            embedding_property="description_embedding",
            text_property="description",
            text_builder="name_description",
            key_property="name",
        )
        embedder = NodeEmbedder.__new__(NodeEmbedder)
        node = {"name": "John Doe", "text": "A lawyer"}
        assert embedder._build_text(node, target) == "John Doe: A lawyer"

    def test_name_description_builder_no_name(self):
        target = EmbeddingTarget(
            label="Person",
            embedding_property="description_embedding",
            text_property="description",
            text_builder="name_description",
            key_property="name",
        )
        embedder = NodeEmbedder.__new__(NodeEmbedder)
        node = {"name": "", "text": "A lawyer"}
        assert embedder._build_text(node, target) == "A lawyer"


# ------------------------------------------------------------------ #
#  Integration tests (require Neo4j + Ollama with nomic-embed-text)   #
# ------------------------------------------------------------------ #

TEST_PREFIX = "TEST_EMB_"


@pytest.fixture()
def config():
    return Config()


@pytest.fixture(autouse=True)
def cleanup_test_nodes(config):
    """Delete test nodes before and after each test."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

    def _cleanup():
        # Clean up Event nodes with TEST_EMB_ prefix
        driver.execute_query(
            "MATCH (n:Event) WHERE n.event_id STARTS WITH $prefix "
            "DETACH DELETE n",
            prefix=TEST_PREFIX,
        )
        # Clean up Person nodes
        driver.execute_query(
            "MATCH (n:Person) WHERE n.name STARTS WITH $prefix "
            "DETACH DELETE n",
            prefix=TEST_PREFIX,
        )
        # Clean up Location nodes
        driver.execute_query(
            "MATCH (n:Location) WHERE n.name STARTS WITH $prefix "
            "DETACH DELETE n",
            prefix=TEST_PREFIX,
        )
        # Clean up Allegation nodes
        driver.execute_query(
            "MATCH (n:Allegation) WHERE n.allegation_id STARTS WITH $prefix "
            "DETACH DELETE n",
            prefix=TEST_PREFIX,
        )

    _cleanup()
    yield
    _cleanup()
    driver.close()


@pytest.fixture()
def embedder(config):
    emb = NodeEmbedder(config)
    yield emb
    emb.close()


def _create_event(embedder, event_id, description):
    """Helper: create an Event node for testing."""
    embedder.driver.execute_query(
        "MERGE (e:Event {event_id: $eid}) "
        "SET e.description = $desc",
        eid=event_id,
        desc=description,
    )


def _create_person(embedder, name, description):
    """Helper: create a Person node for testing."""
    embedder.driver.execute_query(
        "MERGE (p:Person {name: $name}) "
        "SET p.description = $desc",
        name=name,
        desc=description,
    )


@pytest.mark.integration
def test_embed_events(embedder):
    """Events with descriptions get description_embedding set."""
    _create_event(embedder, f"{TEST_PREFIX}001", "A secret meeting took place")
    _create_event(embedder, f"{TEST_PREFIX}002", "Documents were filed in court")

    target = next(t for t in EMBEDDING_TARGETS if t.label == "Event")
    result = embedder.embed_target(target)

    assert result["embedded"] >= 2

    # Verify vectors in Neo4j
    records, _, _ = embedder.driver.execute_query(
        "MATCH (e:Event) WHERE e.event_id STARTS WITH $prefix "
        "AND e.description_embedding IS NOT NULL "
        "RETURN e.event_id AS eid, size(e.description_embedding) AS dim",
        prefix=TEST_PREFIX,
    )
    assert len(records) == 2
    assert records[0]["dim"] == 768


@pytest.mark.integration
def test_embed_skips_null_description(embedder):
    """Events without descriptions are skipped."""
    # Create event with no description
    embedder.driver.execute_query(
        "MERGE (e:Event {event_id: $eid})",
        eid=f"{TEST_PREFIX}003",
    )

    target = next(t for t in EMBEDDING_TARGETS if t.label == "Event")
    embedder.embed_target(target)

    # Should not embed the node with no description
    records, _, _ = embedder.driver.execute_query(
        "MATCH (e:Event {event_id: $eid}) "
        "RETURN e.description_embedding AS emb",
        eid=f"{TEST_PREFIX}003",
    )
    assert records[0]["emb"] is None


@pytest.mark.integration
def test_embed_is_resume_safe(embedder):
    """Running embed_target twice doesn't re-embed already-done nodes."""
    _create_event(embedder, f"{TEST_PREFIX}004", "First run test")

    target = next(t for t in EMBEDDING_TARGETS if t.label == "Event")

    result1 = embedder.embed_target(target)
    embedded_first = result1["embedded"]
    assert embedded_first >= 1

    # Second run — should find 0 needing embeddings (at least for our test node)
    embedder.embed_target(target)
    # Our test node already has an embedding, so it should NOT be re-processed.
    # (Other existing nodes might still need embeddings, so we check our node.)
    records, _, _ = embedder.driver.execute_query(
        "MATCH (e:Event {event_id: $eid}) "
        "WHERE e.description_embedding IS NOT NULL "
        "RETURN count(e) AS cnt",
        eid=f"{TEST_PREFIX}004",
    )
    assert records[0]["cnt"] == 1


@pytest.mark.integration
def test_embed_person_with_name_description(embedder):
    """Person nodes combine name + description for embedding text."""
    _create_person(embedder, f"{TEST_PREFIX}John Doe", "A prominent attorney")

    target = next(t for t in EMBEDDING_TARGETS if t.label == "Person")
    result = embedder.embed_target(target)

    assert result["embedded"] >= 1

    records, _, _ = embedder.driver.execute_query(
        "MATCH (p:Person {name: $name}) "
        "RETURN size(p.description_embedding) AS dim",
        name=f"{TEST_PREFIX}John Doe",
    )
    assert len(records) == 1
    assert records[0]["dim"] == 768


@pytest.mark.integration
def test_embed_all(embedder):
    """embed_all() processes all node types and returns a summary."""
    _create_event(embedder, f"{TEST_PREFIX}005", "Something happened")
    _create_person(embedder, f"{TEST_PREFIX}Jane", "A witness")

    result = embedder.embed_all()

    assert "targets" in result
    assert len(result["targets"]) == 4
    assert result["total_embedded"] >= 2

    # Verify labels in targets
    labels = {t["label"] for t in result["targets"]}
    assert labels == {"Event", "Allegation", "Person", "Location"}
