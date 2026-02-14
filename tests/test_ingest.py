"""Tests for graph ingestion.

Tests cover:
- Utility functions (_clean_pipe_value, _is_valid_name, _safe_props)
- GraphIngestor with a live Neo4j instance (skipped if unavailable)
- Alias resolution during ingestion
- Idempotency of MERGE-based writes
- Filtering of hallucinated/invalid entities
"""

import pytest

from epstein_graphrag.config import Config
from epstein_graphrag.graph.ingest import (
    GraphIngestor,
    _clean_pipe_value,
    _is_valid_name,
    _safe_props,
)

# ------------------------------------------------------------------ #
#  Unit tests for utility functions (no Neo4j required)               #
# ------------------------------------------------------------------ #


class TestCleanPipeValue:
    """Tests for _clean_pipe_value."""

    def test_single_value_unchanged(self):
        assert _clean_pipe_value("perpetrator") == "perpetrator"

    def test_pipe_separated_takes_first(self):
        assert _clean_pipe_value("perpetrator|associate|legal") == "perpetrator"

    def test_strips_whitespace(self):
        assert _clean_pipe_value("  financial  ") == "financial"

    def test_pipe_with_spaces(self):
        assert _clean_pipe_value(" perpetrator | associate ") == "perpetrator"

    def test_empty_string(self):
        assert _clean_pipe_value("") == ""


class TestIsValidName:
    """Tests for _is_valid_name."""

    def test_valid_name(self):
        assert _is_valid_name("Jeffrey Epstein") is True

    def test_empty_string(self):
        assert _is_valid_name("") is False

    def test_whitespace_only(self):
        assert _is_valid_name("   ") is False

    def test_hallucinated_full_name(self):
        assert _is_valid_name("Full Name as written") is False

    def test_hallucinated_organization_name(self):
        assert _is_valid_name("Organization Name (unknown)") is False

    def test_hallucinated_user(self):
        assert _is_valid_name("user") is False

    def test_hallucinated_unknown(self):
        assert _is_valid_name("unknown") is False

    def test_hallucinated_na(self):
        assert _is_valid_name("n/a") is False

    def test_hallucinated_none(self):
        assert _is_valid_name("none") is False

    def test_doc_id_as_name(self):
        assert _is_valid_name("EFTA0002178") is False

    def test_doc_id_lowercase(self):
        assert _is_valid_name("efta00001234") is False

    def test_prompt_fragment_exact_quote(self):
        assert _is_valid_name("include exact quote from document") is False

    def test_prompt_fragment_one_sentence(self):
        assert _is_valid_name("provide one sentence summary") is False

    def test_case_insensitive_hallucinated(self):
        assert _is_valid_name("UNKNOWN") is False
        assert _is_valid_name("N/A") is False

    def test_valid_name_with_leading_whitespace(self):
        """Valid name with whitespace should still be valid."""
        assert _is_valid_name("  Leon Black  ") is True


class TestSafeProps:
    """Tests for _safe_props."""

    def test_strips_strings(self):
        result = _safe_props({"name": "  Jeffrey Epstein  "})
        assert result == {"name": "Jeffrey Epstein"}

    def test_removes_empty_strings(self):
        result = _safe_props({"name": "Epstein", "role": "", "desc": "financier"})
        assert result == {"name": "Epstein", "desc": "financier"}

    def test_cleans_pipe_values(self):
        result = _safe_props({"role": "perpetrator|associate"})
        assert result == {"role": "perpetrator"}

    def test_filters_empty_list_items(self):
        result = _safe_props({"tags": ["  a  ", "", "b", "  "]})
        assert result == {"tags": ["a", "b"]}

    def test_removes_fully_empty_lists(self):
        result = _safe_props({"tags": ["", "  "]})
        assert "tags" not in result

    def test_preserves_non_string_non_list(self):
        result = _safe_props({"count": 42, "active": True})
        assert result == {"count": 42, "active": True}

    def test_empty_dict(self):
        assert _safe_props({}) == {}


# ------------------------------------------------------------------ #
#  Integration tests — require a running Neo4j instance               #
# ------------------------------------------------------------------ #


def _make_test_config() -> Config:
    """Create a Config for testing."""
    return Config()


def _neo4j_available(config: Config) -> bool:
    """Check if Neo4j is reachable."""
    from neo4j import GraphDatabase

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
    """Provide a Config instance, skip if Neo4j is unreachable."""
    cfg = _make_test_config()
    if not _neo4j_available(cfg):
        pytest.skip("Neo4j not available")
    return cfg


@pytest.fixture
def ingestor(config):
    """Provide a GraphIngestor with a mock AliasResolver."""
    from epstein_graphrag.graph.dedup import AliasResolver

    resolver = AliasResolver(config)
    # Add test aliases
    resolver.add_alias("LEON D BLACK", "Leon Black")
    resolver.add_alias("J. EPSTEIN", "Jeffrey Epstein")

    ing = GraphIngestor(config, alias_resolver=resolver)
    yield ing
    ing.close()


@pytest.fixture(autouse=True)
def cleanup_test_nodes(config):
    """Remove test nodes before and after each test."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

    def _cleanup():
        # Only delete nodes from test documents (TEST_DOC_*)
        driver.execute_query(
            "MATCH (n) WHERE "
            "  (n:Document AND n.doc_id STARTS WITH 'TEST_DOC_') OR "
            "  (n:Event AND n.event_id STARTS WITH 'TEST_DOC_') OR "
            "  (n:Allegation AND n.allegation_id STARTS WITH 'TEST_DOC_') "
            "DETACH DELETE n"
        )
        # Clean up test persons/locations/orgs that might be left
        for name in [
            "Test Person A", "Test Person B", "Leon Black",
            "Jeffrey Epstein", "Test Location", "Test Org",
            "J. EPSTEIN",
        ]:
            driver.execute_query(
                "MATCH (n) WHERE n.name = $name DETACH DELETE n",
                parameters_={"name": name},
            )

    _cleanup()
    yield
    _cleanup()
    driver.close()


# -- Minimal document ingestion --


def test_ingest_minimal_document(ingestor):
    """Ingesting a document with only doc_id creates a Document node."""
    extraction = {"doc_id": "TEST_DOC_001", "doc_type": "text"}
    stats = ingestor.ingest_document(extraction)

    assert stats["nodes_merged"] == 1
    assert stats["relationships_merged"] == 0

    # Verify in Neo4j
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (d:Document {doc_id: 'TEST_DOC_001'}) RETURN d.doc_type AS doc_type"
    )
    assert len(records) == 1
    assert records[0]["doc_type"] == "text"


def test_ingest_missing_doc_id(ingestor):
    """Extraction with no doc_id is skipped."""
    stats = ingestor.ingest_document({"doc_type": "text"})
    assert stats["nodes_merged"] == 0
    assert stats["relationships_merged"] == 0


# -- People ingestion --


def test_ingest_people(ingestor):
    """People are ingested as Person nodes with MENTIONED_IN rels."""
    extraction = {
        "doc_id": "TEST_DOC_002",
        "doc_type": "text",
        "people": [
            {"name": "Test Person A", "role": "witness", "context": "A witness"},
        ],
    }
    stats = ingestor.ingest_document(extraction)

    # 1 Document + 1 Person = 2 nodes, 1 MENTIONED_IN = 1 rel
    assert stats["nodes_merged"] == 2
    assert stats["relationships_merged"] == 1

    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Test Person A'})-[:MENTIONED_IN]->(d:Document) "
        "RETURN p.role AS role, d.doc_id AS doc_id"
    )
    assert len(records) == 1
    assert records[0]["role"] == "witness"
    assert records[0]["doc_id"] == "TEST_DOC_002"


def test_ingest_filters_hallucinated_people(ingestor):
    """Hallucinated person names are filtered out."""
    extraction = {
        "doc_id": "TEST_DOC_003",
        "doc_type": "text",
        "people": [
            {"name": "Full Name as written", "role": "unknown"},
            {"name": "EFTA0002178", "role": "unknown"},
            {"name": "", "role": "unknown"},
            {"name": "Test Person A", "role": "witness"},
        ],
    }
    stats = ingestor.ingest_document(extraction)

    # Only 1 valid person, so: 1 Document + 1 Person = 2 nodes
    assert stats["nodes_merged"] == 2
    assert stats["relationships_merged"] == 1


# -- Alias resolution --


def test_ingest_with_alias_resolution(ingestor):
    """Names are resolved through alias table before MERGE."""
    extraction = {
        "doc_id": "TEST_DOC_004",
        "doc_type": "text",
        "people": [
            {"name": "LEON D BLACK", "role": "associate"},
        ],
    }
    ingestor.ingest_document(extraction)

    # Should be stored as "Leon Black" (the canonical name)
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Leon Black'}) RETURN p.name AS name"
    )
    assert len(records) == 1

    # The original form should NOT exist as a separate node
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'LEON D BLACK'}) RETURN p"
    )
    assert len(records) == 0


# -- Pipe value cleaning --


def test_ingest_cleans_pipe_values(ingestor):
    """Pipe-separated role values are cleaned to first value."""
    extraction = {
        "doc_id": "TEST_DOC_005",
        "doc_type": "text",
        "people": [
            {"name": "Test Person A", "role": "perpetrator|associate|legal"},
        ],
    }
    ingestor.ingest_document(extraction)

    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Test Person A'}) RETURN p.role AS role"
    )
    assert len(records) == 1
    assert records[0]["role"] == "perpetrator"


# -- Locations --


def test_ingest_locations(ingestor):
    """Locations are ingested with MENTIONED_IN relationships."""
    extraction = {
        "doc_id": "TEST_DOC_006",
        "doc_type": "text",
        "locations": [
            {"name": "Test Location", "location_type": "residence", "address": "123 Main St"},
        ],
    }
    stats = ingestor.ingest_document(extraction)

    # 1 Document + 1 Location = 2 nodes, 1 MENTIONED_IN
    assert stats["nodes_merged"] == 2
    assert stats["relationships_merged"] == 1


# -- Organizations --


def test_ingest_organizations(ingestor):
    """Organizations are ingested with MENTIONED_IN relationships."""
    extraction = {
        "doc_id": "TEST_DOC_007",
        "doc_type": "text",
        "organizations": [
            {"name": "Test Org", "org_type": "financial"},
        ],
    }
    stats = ingestor.ingest_document(extraction)

    assert stats["nodes_merged"] == 2
    assert stats["relationships_merged"] == 1


# -- Events --


def test_ingest_events(ingestor):
    """Events get generated IDs and DOCUMENTED_IN relationships."""
    extraction = {
        "doc_id": "TEST_DOC_008",
        "doc_type": "text",
        "events": [
            {
                "description": "A meeting occurred",
                "event_type": "meeting",
                "date": "2005-01-15",
                "participants": ["Test Person A"],
                "location": "Test Location",
            },
        ],
    }
    stats = ingestor.ingest_document(extraction)

    # 1 Document + 1 Event + 1 Person (ensured) + 1 Location (ensured) = 4 nodes
    # 1 DOCUMENTED_IN + 1 PARTICIPATED_IN + 1 OCCURRED_AT = 3 rels
    assert stats["nodes_merged"] >= 2  # At least Document + Event
    assert stats["relationships_merged"] >= 1  # At least DOCUMENTED_IN

    records, _, _ = ingestor.driver.execute_query(
        "MATCH (e:Event {event_id: 'TEST_DOC_008_evt_0'})-[:DOCUMENTED_IN]->(d:Document) "
        "RETURN e.description AS desc, d.doc_id AS doc_id"
    )
    assert len(records) == 1
    assert records[0]["desc"] == "A meeting occurred"


def test_ingest_events_skips_empty_description(ingestor):
    """Events with empty descriptions are skipped."""
    extraction = {
        "doc_id": "TEST_DOC_009",
        "doc_type": "text",
        "events": [{"description": "", "event_type": "unknown"}],
    }
    stats = ingestor.ingest_document(extraction)

    # Only the Document node
    assert stats["nodes_merged"] == 1
    assert stats["relationships_merged"] == 0


# -- Allegations --


def test_ingest_allegations(ingestor):
    """Allegations with accused/victims are ingested properly."""
    extraction = {
        "doc_id": "TEST_DOC_010",
        "doc_type": "text",
        "people": [
            {"name": "Test Person A", "role": "accused"},
            {"name": "Test Person B", "role": "victim"},
        ],
        "allegations": [
            {
                "description": "Test allegation",
                "severity": "serious",
                "accused": ["Test Person A"],
                "victims": ["Test Person B"],
            },
        ],
    }
    ingestor.ingest_document(extraction)

    records, _, _ = ingestor.driver.execute_query(
        "MATCH (a:Allegation {allegation_id: 'TEST_DOC_010_alg_0'}) "
        "RETURN a.description AS desc"
    )
    assert len(records) == 1
    assert records[0]["desc"] == "Test allegation"

    # Check ALLEGED_IN relationship
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Test Person A'})-[:ALLEGED_IN]->"
        "(a:Allegation {allegation_id: 'TEST_DOC_010_alg_0'}) RETURN p.name"
    )
    assert len(records) == 1

    # Check VICTIM_OF relationship
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Test Person B'})-[:VICTIM_OF]->"
        "(a:Allegation {allegation_id: 'TEST_DOC_010_alg_0'}) RETURN p.name"
    )
    assert len(records) == 1


def test_ingest_allegation_skips_no_people(ingestor):
    """Allegations with no accused and no victims are skipped."""
    extraction = {
        "doc_id": "TEST_DOC_011",
        "doc_type": "text",
        "allegations": [
            {"description": "Something bad happened", "accused": [], "victims": []},
        ],
    }
    stats = ingestor.ingest_document(extraction)

    # Only the Document node
    assert stats["nodes_merged"] == 1


# -- Associations --


def test_ingest_associations(ingestor):
    """Associations create ASSOCIATED_WITH relationships."""
    extraction = {
        "doc_id": "TEST_DOC_012",
        "doc_type": "text",
        "people": [
            {"name": "Test Person A", "role": "associate"},
            {"name": "Test Person B", "role": "associate"},
        ],
        "associations": [
            {
                "person_a": "Test Person A",
                "person_b": "Test Person B",
                "nature": "financial",
            },
        ],
    }
    ingestor.ingest_document(extraction)

    records, _, _ = ingestor.driver.execute_query(
        "MATCH (a:Person {name: 'Test Person A'})"
        "-[r:ASSOCIATED_WITH]->"
        "(b:Person {name: 'Test Person B'}) "
        "RETURN r.nature AS nature"
    )
    assert len(records) == 1
    assert records[0]["nature"] == "financial"


def test_ingest_skips_self_association(ingestor):
    """Self-associations (same person after alias resolution) are skipped."""
    extraction = {
        "doc_id": "TEST_DOC_013",
        "doc_type": "text",
        "people": [{"name": "J. EPSTEIN", "role": "subject"}],
        "associations": [
            {
                "person_a": "J. EPSTEIN",
                "person_b": "Jeffrey Epstein",
                "nature": "same_person",
            },
        ],
    }
    ingestor.ingest_document(extraction)

    # No ASSOCIATED_WITH should be created
    records, _, _ = ingestor.driver.execute_query(
        "MATCH (p:Person {name: 'Jeffrey Epstein'})"
        "-[r:ASSOCIATED_WITH]->"
        "(p2:Person {name: 'Jeffrey Epstein'}) RETURN r"
    )
    assert len(records) == 0


# -- Idempotency --


def test_ingest_idempotent(ingestor):
    """Running the same extraction twice produces the same graph."""
    extraction = {
        "doc_id": "TEST_DOC_014",
        "doc_type": "text",
        "people": [
            {"name": "Test Person A", "role": "witness"},
            {"name": "Test Person B", "role": "accused"},
        ],
        "associations": [
            {
                "person_a": "Test Person A",
                "person_b": "Test Person B",
                "nature": "legal",
            },
        ],
    }

    ingestor.ingest_document(extraction)

    # Count nodes/rels after first run
    records1, _, _ = ingestor.driver.execute_query(
        "MATCH (d:Document {doc_id: 'TEST_DOC_014'}) "
        "OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(n) "
        "RETURN count(DISTINCT n) AS people"
    )
    people_count_1 = records1[0]["people"]

    # Run again
    ingestor.ingest_document(extraction)

    records2, _, _ = ingestor.driver.execute_query(
        "MATCH (d:Document {doc_id: 'TEST_DOC_014'}) "
        "OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(n) "
        "RETURN count(DISTINCT n) AS people"
    )
    people_count_2 = records2[0]["people"]

    assert people_count_1 == people_count_2


# -- Batch ingestion --


def test_ingest_batch(ingestor):
    """Batch ingestion processes multiple documents and returns stats."""
    extractions = [
        {"doc_id": "TEST_DOC_015", "doc_type": "text"},
        {"doc_id": "TEST_DOC_016", "doc_type": "photo"},
        {"doc_id": "TEST_DOC_017", "doc_type": "text"},
    ]
    result = ingestor.ingest_batch(extractions)

    assert result["total"] == 3
    assert result["succeeded"] == 3
    assert result["failed"] == 0
    assert result["total_nodes_merged"] == 3  # 3 Document nodes
    assert result["total_relationships_merged"] == 0
    assert result["errors"] == []
