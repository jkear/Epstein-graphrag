"""Neo4j schema creation and validation.

Creates constraints, vector indexes, and fulltext indexes.
Validates that the schema is correctly set up.

All operations use IF NOT EXISTS for idempotency — running
create_schema() multiple times is safe.
"""

import logging

from neo4j import GraphDatabase
from neo4j import Query

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)

# Node key constraints: (label, key_property)
# Uses NODE KEY (enforces existence + uniqueness) matching the existing schema.
NODE_KEY_CONSTRAINTS: list[tuple[str, str]] = [
    ("Person", "name"),
    ("Organization", "name"),
    ("Location", "name"),
    ("Property", "name"),
    ("BankAccount", "account_number"),
    ("Transaction", "transaction_id"),
    ("Vessel", "name"),
    ("Aircraft", "tail_number"),
    ("PhysicalItem", "item_id"),
    ("DigitalEvidence", "file_name"),
    ("Event", "event_id"),
    ("PropertyDevelopment", "project_name"),
    ("Document", "doc_id"),
    ("DocumentChunk", "chunk_id"),
    ("VisualEvidence", "visual_id"),
    ("Allegation", "allegation_id"),
    ("LegalProceeding", "case_id"),
    ("Contact", "contact_id"),
]

# Vector indexes: (index_name, label, property, dimensions)
VECTOR_INDEXES: list[tuple[str, str, str, int]] = [
    ("document_chunk_embeddings", "DocumentChunk", "embedding", 768),
    ("visual_evidence_embeddings", "VisualEvidence", "embedding", 768),
    ("event_embeddings", "Event", "description_embedding", 768),
    ("allegation_embeddings", "Allegation", "embedding", 768),
    ("person_description_embeddings", "Person", "description_embedding", 768),
    ("location_description_embeddings", "Location", "description_embedding", 768),
]

# Fulltext indexes: (index_name, labels, properties)
FULLTEXT_INDEXES: list[tuple[str, list[str], list[str]]] = [
    ("person_fulltext", ["Person"], ["name", "description"]),
    ("document_fulltext", ["Document"], ["doc_id", "text_content"]),
    ("allegation_fulltext", ["Allegation"], ["description"]),
    ("event_fulltext", ["Event"], ["description"]),
]


def _get_existing_constraints(driver) -> dict[str, dict]:
    """Get existing constraints keyed by name."""
    records, _, _ = driver.execute_query("SHOW CONSTRAINTS")
    return {r["name"]: dict(r) for r in records}


def _get_existing_indexes(driver) -> dict[str, dict]:
    """Get existing indexes keyed by name."""
    records, _, _ = driver.execute_query("SHOW INDEXES")
    return {r["name"]: dict(r) for r in records}


def create_schema(config: Config) -> dict:
    """Create all constraints and indexes in Neo4j.

    This is idempotent — skips constraints/indexes that already exist
    (by name or by schema). Safe to run repeatedly.

    Args:
        config: Application configuration with Neo4j credentials.

    Returns:
        Dictionary with counts of created/existing constraints and indexes.
    """
    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

    stats = {
        "constraints_created": 0,
        "constraints_existing": 0,
        "vector_indexes_created": 0,
        "vector_indexes_existing": 0,
        "fulltext_indexes_created": 0,
        "fulltext_indexes_existing": 0,
    }

    try:
        existing_constraints = _get_existing_constraints(driver)
        existing_indexes = _get_existing_indexes(driver)

        # Create node key constraints
        for label, prop in NODE_KEY_CONSTRAINTS:
            constraint_name = f"{label}_constraint"

            # Check if already exists (by name or by schema)
            if constraint_name in existing_constraints:
                stats["constraints_existing"] += 1
                logger.debug(f"Constraint already exists: {constraint_name}")
                continue

            # Also check if any constraint covers this label+property
            already_covered = any(
                c.get("labelsOrTypes") == [label] and c.get("properties") == [prop]
                for c in existing_constraints.values()
            )
            if already_covered:
                stats["constraints_existing"] += 1
                logger.debug(f"Constraint already covered for {label}.{prop}")
                continue

            query = (
                f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE n.{prop} IS NODE KEY"
            )
            try:
                driver.execute_query(Query(query))
                stats["constraints_created"] += 1
                logger.debug(f"Created constraint: {constraint_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    stats["constraints_existing"] += 1
                    logger.debug(f"Constraint already exists: {constraint_name}")
                else:
                    logger.error(f"Failed to create constraint {constraint_name}: {e}")
                    raise

        total_constraints = stats["constraints_created"] + stats["constraints_existing"]
        logger.info(
            f"Constraints: {stats['constraints_created']} created, "
            f"{stats['constraints_existing']} already existed "
            f"({total_constraints} total)"
        )

        # Create vector indexes
        for index_name, label, prop, dimensions in VECTOR_INDEXES:
            if index_name in existing_indexes:
                stats["vector_indexes_existing"] += 1
                logger.debug(f"Vector index already exists: {index_name}")
                continue

            query = (
                f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{label}) ON (n.{prop}) "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {dimensions}, "
                f"`vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
            try:
                driver.execute_query(Query(query))
                stats["vector_indexes_created"] += 1
                logger.debug(f"Created vector index: {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    stats["vector_indexes_existing"] += 1
                else:
                    logger.error(f"Failed to create vector index {index_name}: {e}")
                    raise

        total_vectors = stats["vector_indexes_created"] + stats["vector_indexes_existing"]
        logger.info(
            f"Vector indexes: {stats['vector_indexes_created']} created, "
            f"{stats['vector_indexes_existing']} already existed "
            f"({total_vectors} total)"
        )

        # Create fulltext indexes
        for index_name, labels, properties in FULLTEXT_INDEXES:
            if index_name in existing_indexes:
                stats["fulltext_indexes_existing"] += 1
                logger.debug(f"Fulltext index already exists: {index_name}")
                continue

            label_str = "|".join(labels)
            prop_str = ", ".join(f"n.{p}" for p in properties)
            query = (
                f"CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS "
                f"FOR (n:{label_str}) ON EACH [{prop_str}]"
            )
            try:
                driver.execute_query(query)
                stats["fulltext_indexes_created"] += 1
                logger.debug(f"Created fulltext index: {index_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    stats["fulltext_indexes_existing"] += 1
                else:
                    logger.error(f"Failed to create fulltext index {index_name}: {e}")
                    raise

        total_ft = stats["fulltext_indexes_created"] + stats["fulltext_indexes_existing"]
        logger.info(
            f"Fulltext indexes: {stats['fulltext_indexes_created']} created, "
            f"{stats['fulltext_indexes_existing']} already existed "
            f"({total_ft} total)"
        )

    finally:
        driver.close()

    return stats


def verify_schema(config: Config) -> dict:
    """Verify that the Neo4j schema is correctly set up.

    Args:
        config: Application configuration with Neo4j credentials.

    Returns:
        Dictionary with counts of constraints and indexes found,
        plus lists of their names.
    """
    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password),
    )

    try:
        # Count constraints
        records, _, _ = driver.execute_query("SHOW CONSTRAINTS")
        constraint_names = [r["name"] for r in records]

        # Count all indexes, then filter by type
        records, _, _ = driver.execute_query("SHOW INDEXES")
        vector_names = [r["name"] for r in records if r.get("type") == "VECTOR"]
        fulltext_names = [
            r["name"] for r in records
            if r.get("type") == "FULLTEXT" and r["name"] != "search"  # Exclude MCP memory index
        ]

        result = {
            "constraints": len(constraint_names),
            "constraint_names": sorted(constraint_names),
            "vector_indexes": len(vector_names),
            "vector_index_names": sorted(vector_names),
            "fulltext_indexes": len(fulltext_names),
            "fulltext_index_names": sorted(fulltext_names),
        }

        logger.info(
            f"Schema verified: {result['constraints']} constraints, "
            f"{result['vector_indexes']} vector indexes, "
            f"{result['fulltext_indexes']} fulltext indexes"
        )

        return result

    finally:
        driver.close()
