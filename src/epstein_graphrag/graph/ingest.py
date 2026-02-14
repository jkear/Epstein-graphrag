"""Neo4j graph ingestion — MERGE-based, idempotent.

All writes use MERGE (never CREATE) to ensure idempotent ingestion.
Running the same batch twice produces the same graph state.
"""

import logging
import re

from neo4j import GraphDatabase

from epstein_graphrag.config import Config
from epstein_graphrag.graph.dedup import AliasResolver

logger = logging.getLogger(__name__)

# Names that indicate hallucinated / placeholder entities from the LLM.
# These are literal prompt fragments the model echoed back instead of
# extracting real data.
HALLUCINATED_NAMES: set[str] = {
    "full name as written",
    "organization name",
    "organization name (unknown)",
    "user",
    "unknown",
    "n/a",
    "none",
    "",
}

# Regex for doc IDs accidentally used as entity names (e.g. "EFTA0002178")
_DOC_ID_RE = re.compile(r"^EFTA\d+$", re.IGNORECASE)


def _clean_pipe_value(value: str) -> str:
    """Take the first value from a pipe-separated enum string.

    LLM sometimes outputs "perpetrator|associate|legal" instead of
    picking one. We take the first as the best guess.
    """
    if "|" in value:
        return value.split("|")[0].strip()
    return value.strip()


def _is_valid_name(name: str) -> bool:
    """Check whether an entity name is usable (not empty, not hallucinated)."""
    if not name or not name.strip():
        return False
    normalized = name.strip().lower()
    if normalized in HALLUCINATED_NAMES:
        return False
    if _DOC_ID_RE.match(name.strip()):
        return False
    # Reject prompt template fragments
    if "exact quote" in normalized or "one sentence" in normalized:
        return False
    return True


def _safe_props(props: dict) -> dict:
    """Clean a property dict: strip strings, fix pipe-separated values.

    Returns a new dict with empty-string values removed and pipe values
    cleaned.
    """
    cleaned = {}
    for k, v in props.items():
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
            v = _clean_pipe_value(v)
        if isinstance(v, list):
            # Filter empty strings from lists
            v = [item.strip() for item in v if isinstance(item, str) and item.strip()]
            if not v:
                continue
        cleaned[k] = v
    return cleaned


class GraphIngestor:
    """Ingests extracted entities and relationships into Neo4j.

    Reads extraction JSONs produced by the entity extractor and writes
    nodes/relationships to Neo4j using MERGE for idempotency. All entity
    names pass through AliasResolver before MERGE.
    """

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

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def ingest_document(self, extraction: dict) -> dict:
        """Ingest a single document's extracted entities into Neo4j.

        Args:
            extraction: Extraction JSON dict with doc_id, doc_type, and
                        entity category lists (people, locations, etc.).

        Returns:
            Stats dict with nodes_merged and relationships_merged counts.
        """
        doc_id = extraction.get("doc_id", "")
        doc_type = extraction.get("doc_type", "unknown")
        stats = {"nodes_merged": 0, "relationships_merged": 0}

        if not doc_id:
            logger.warning("Extraction missing doc_id, skipping")
            return stats

        with self.driver.session() as session:
            # 1. MERGE the Document node
            session.execute_write(
                self._merge_document, doc_id, doc_type
            )
            stats["nodes_merged"] += 1

            # 2. Process each entity category
            stats = self._ingest_people(session, extraction, doc_id, stats)
            stats = self._ingest_locations(session, extraction, doc_id, stats)
            stats = self._ingest_organizations(session, extraction, doc_id, stats)
            stats = self._ingest_events(session, extraction, doc_id, stats)
            stats = self._ingest_allegations(session, extraction, doc_id, stats)
            stats = self._ingest_associations(session, extraction, doc_id, stats)

        logger.info(
            f"Ingested {doc_id}: {stats['nodes_merged']} nodes, "
            f"{stats['relationships_merged']} relationships"
        )
        return stats

    def ingest_batch(self, extractions: list[dict]) -> dict:
        """Ingest a batch of extraction dicts into Neo4j.

        Args:
            extractions: List of extraction JSON dicts.

        Returns:
            Summary dict with total, succeeded, failed, errors, and
            aggregate node/relationship counts.
        """
        result = {
            "total": len(extractions),
            "succeeded": 0,
            "failed": 0,
            "errors": [],
            "total_nodes_merged": 0,
            "total_relationships_merged": 0,
        }

        for extraction in extractions:
            doc_id = extraction.get("doc_id", "<unknown>")
            try:
                stats = self.ingest_document(extraction)
                result["succeeded"] += 1
                result["total_nodes_merged"] += stats["nodes_merged"]
                result["total_relationships_merged"] += stats["relationships_merged"]
            except Exception as e:
                result["failed"] += 1
                result["errors"].append({"doc_id": doc_id, "error": str(e)})
                logger.error(f"Failed to ingest {doc_id}: {e}")

        logger.info(
            f"Batch complete: {result['succeeded']}/{result['total']} succeeded, "
            f"{result['failed']} failed, "
            f"{result['total_nodes_merged']} nodes, "
            f"{result['total_relationships_merged']} relationships"
        )
        return result

    # ------------------------------------------------------------------ #
    #  Static transaction functions (used with session.execute_write)      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _merge_document(tx, doc_id: str, doc_type: str) -> None:
        tx.run(
            "MERGE (d:Document {doc_id: $doc_id}) "
            "SET d.doc_type = $doc_type",
            doc_id=doc_id,
            doc_type=doc_type,
        )

    @staticmethod
    def _merge_person(tx, name: str, props: dict) -> None:
        tx.run(
            "MERGE (p:Person {name: $name}) "
            "SET p += $props",
            name=name,
            props=props,
        )

    @staticmethod
    def _merge_person_mentioned_in(tx, name: str, doc_id: str) -> None:
        tx.run(
            "MATCH (p:Person {name: $name}) "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (p)-[:MENTIONED_IN]->(d)",
            name=name,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_location(tx, name: str, props: dict) -> None:
        tx.run(
            "MERGE (l:Location {name: $name}) "
            "SET l += $props",
            name=name,
            props=props,
        )

    @staticmethod
    def _merge_location_mentioned_in(tx, name: str, doc_id: str) -> None:
        tx.run(
            "MATCH (l:Location {name: $name}) "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (l)-[:MENTIONED_IN]->(d)",
            name=name,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_organization(tx, name: str, props: dict) -> None:
        tx.run(
            "MERGE (o:Organization {name: $name}) "
            "SET o += $props",
            name=name,
            props=props,
        )

    @staticmethod
    def _merge_organization_mentioned_in(tx, name: str, doc_id: str) -> None:
        tx.run(
            "MATCH (o:Organization {name: $name}) "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (o)-[:MENTIONED_IN]->(d)",
            name=name,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_event(tx, event_id: str, props: dict) -> None:
        tx.run(
            "MERGE (e:Event {event_id: $event_id}) "
            "SET e += $props",
            event_id=event_id,
            props=props,
        )

    @staticmethod
    def _merge_event_documented_in(tx, event_id: str, doc_id: str) -> None:
        tx.run(
            "MATCH (e:Event {event_id: $event_id}) "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (e)-[:DOCUMENTED_IN]->(d)",
            event_id=event_id,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_event_participant(tx, event_id: str, person_name: str) -> None:
        tx.run(
            "MATCH (e:Event {event_id: $event_id}) "
            "MATCH (p:Person {name: $person_name}) "
            "MERGE (p)-[:PARTICIPATED_IN]->(e)",
            event_id=event_id,
            person_name=person_name,
        )

    @staticmethod
    def _merge_event_location(tx, event_id: str, location_name: str) -> None:
        tx.run(
            "MATCH (e:Event {event_id: $event_id}) "
            "MATCH (l:Location {name: $location_name}) "
            "MERGE (e)-[:OCCURRED_AT]->(l)",
            event_id=event_id,
            location_name=location_name,
        )

    @staticmethod
    def _merge_allegation(tx, allegation_id: str, props: dict) -> None:
        tx.run(
            "MERGE (a:Allegation {allegation_id: $allegation_id}) "
            "SET a += $props",
            allegation_id=allegation_id,
            props=props,
        )

    @staticmethod
    def _merge_allegation_documented_in(tx, allegation_id: str, doc_id: str) -> None:
        tx.run(
            "MATCH (a:Allegation {allegation_id: $allegation_id}) "
            "MATCH (d:Document {doc_id: $doc_id}) "
            "MERGE (a)-[:DOCUMENTED_IN]->(d)",
            allegation_id=allegation_id,
            doc_id=doc_id,
        )

    @staticmethod
    def _merge_allegation_accused(tx, allegation_id: str, person_name: str) -> None:
        tx.run(
            "MATCH (a:Allegation {allegation_id: $allegation_id}) "
            "MATCH (p:Person {name: $person_name}) "
            "MERGE (p)-[:ALLEGED_IN]->(a)",
            allegation_id=allegation_id,
            person_name=person_name,
        )

    @staticmethod
    def _merge_allegation_victim(tx, allegation_id: str, person_name: str) -> None:
        tx.run(
            "MATCH (a:Allegation {allegation_id: $allegation_id}) "
            "MATCH (p:Person {name: $person_name}) "
            "MERGE (p)-[:VICTIM_OF]->(a)",
            allegation_id=allegation_id,
            person_name=person_name,
        )

    @staticmethod
    def _merge_association(tx, person_a: str, person_b: str, props: dict) -> None:
        tx.run(
            "MATCH (a:Person {name: $person_a}) "
            "MATCH (b:Person {name: $person_b}) "
            "MERGE (a)-[r:ASSOCIATED_WITH]->(b) "
            "SET r += $props",
            person_a=person_a,
            person_b=person_b,
            props=props,
        )

    # ------------------------------------------------------------------ #
    #  Per-category ingestion helpers                                      #
    # ------------------------------------------------------------------ #

    def _resolve(self, name: str) -> str:
        """Resolve a name through the alias table."""
        return self.alias_resolver.resolve(name)

    def _ingest_people(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest people entities from extraction."""
        for person in extraction.get("people", []):
            name = person.get("name", "")
            if not _is_valid_name(name):
                logger.debug(f"Skipping invalid person name: {name!r}")
                continue

            name = self._resolve(name)

            props = _safe_props({
                "role": person.get("role", ""),
                "description": person.get("context", ""),
            })

            session.execute_write(self._merge_person, name, props)
            stats["nodes_merged"] += 1

            session.execute_write(self._merge_person_mentioned_in, name, doc_id)
            stats["relationships_merged"] += 1

        return stats

    def _ingest_locations(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest location entities from extraction."""
        for loc in extraction.get("locations", []):
            name = loc.get("name", "")
            if not _is_valid_name(name):
                logger.debug(f"Skipping invalid location name: {name!r}")
                continue

            name = self._resolve(name)

            props = _safe_props({
                "address": loc.get("address", ""),
                "location_type": loc.get("location_type", ""),
                "description": loc.get("context", ""),
            })

            session.execute_write(self._merge_location, name, props)
            stats["nodes_merged"] += 1

            session.execute_write(self._merge_location_mentioned_in, name, doc_id)
            stats["relationships_merged"] += 1

        return stats

    def _ingest_organizations(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest organization entities from extraction."""
        for org in extraction.get("organizations", []):
            name = org.get("name", "")
            if not _is_valid_name(name):
                logger.debug(f"Skipping invalid org name: {name!r}")
                continue

            name = self._resolve(name)

            props = _safe_props({
                "org_type": org.get("org_type", ""),
                "description": org.get("context", "") or org.get("description", ""),
            })

            session.execute_write(self._merge_organization, name, props)
            stats["nodes_merged"] += 1

            session.execute_write(
                self._merge_organization_mentioned_in, name, doc_id
            )
            stats["relationships_merged"] += 1

        return stats

    def _ingest_events(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest event entities from extraction.

        Events have no natural key, so we generate one: {doc_id}_evt_{index}.
        """
        for idx, event in enumerate(extraction.get("events", [])):
            description = event.get("description", "").strip()
            if not description:
                logger.debug(f"Skipping event with empty description in {doc_id}")
                continue

            event_id = f"{doc_id}_evt_{idx}"

            props = _safe_props({
                "event_type": event.get("event_type", ""),
                "date": event.get("date", ""),
                "description": description,
            })

            session.execute_write(self._merge_event, event_id, props)
            stats["nodes_merged"] += 1

            session.execute_write(self._merge_event_documented_in, event_id, doc_id)
            stats["relationships_merged"] += 1

            # Link participants (must already exist as Person nodes)
            for participant_name in event.get("participants", []):
                if not _is_valid_name(participant_name):
                    continue
                participant_name = self._resolve(participant_name)
                # Ensure the Person node exists before linking
                session.execute_write(
                    self._merge_person, participant_name, {}
                )
                session.execute_write(
                    self._merge_event_participant, event_id, participant_name
                )
                stats["relationships_merged"] += 1

            # Link event location if present
            loc_name = event.get("location", "").strip()
            if loc_name and _is_valid_name(loc_name):
                loc_name = self._resolve(loc_name)
                session.execute_write(self._merge_location, loc_name, {})
                session.execute_write(
                    self._merge_event_location, event_id, loc_name
                )
                stats["relationships_merged"] += 1

        return stats

    def _ingest_allegations(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest allegation entities from extraction.

        Allegations use generated IDs: {doc_id}_alg_{index}.
        Skip allegations that have no accused and no victims (low-quality).
        """
        for idx, allegation in enumerate(extraction.get("allegations", [])):
            description = allegation.get("description", "").strip()
            accused = [
                n for n in allegation.get("accused", []) if _is_valid_name(n)
            ]
            victims = [
                n for n in allegation.get("victims", []) if _is_valid_name(n)
            ]

            # Skip allegations with no description or no connected people
            if not description:
                continue
            if not accused and not victims:
                logger.debug(
                    f"Skipping allegation with no accused/victims in {doc_id}"
                )
                continue

            allegation_id = f"{doc_id}_alg_{idx}"

            props = _safe_props({
                "description": description,
                "severity": allegation.get("severity", ""),
                "status": allegation.get("status", ""),
            })

            session.execute_write(
                self._merge_allegation, allegation_id, props
            )
            stats["nodes_merged"] += 1

            session.execute_write(
                self._merge_allegation_documented_in, allegation_id, doc_id
            )
            stats["relationships_merged"] += 1

            for name in accused:
                name = self._resolve(name)
                session.execute_write(self._merge_person, name, {})
                session.execute_write(
                    self._merge_allegation_accused, allegation_id, name
                )
                stats["relationships_merged"] += 1

            for name in victims:
                name = self._resolve(name)
                session.execute_write(self._merge_person, name, {})
                session.execute_write(
                    self._merge_allegation_victim, allegation_id, name
                )
                stats["relationships_merged"] += 1

        return stats

    def _ingest_associations(
        self, session, extraction: dict, doc_id: str, stats: dict
    ) -> dict:
        """Ingest person-to-person associations from extraction.

        Skips associations where either person_a or person_b is empty.
        """
        for assoc in extraction.get("associations", []):
            person_a = assoc.get("person_a", "").strip()
            person_b = assoc.get("person_b", "").strip()

            if not _is_valid_name(person_a) or not _is_valid_name(person_b):
                logger.debug(
                    f"Skipping association with invalid names: "
                    f"{person_a!r} <-> {person_b!r}"
                )
                continue

            person_a = self._resolve(person_a)
            person_b = self._resolve(person_b)

            # Don't create self-associations (can happen after alias resolution)
            if person_a == person_b:
                logger.debug(f"Skipping self-association for {person_a!r}")
                continue

            props = _safe_props({
                "nature": assoc.get("nature", ""),
                "timeframe": assoc.get("timeframe", ""),
            })

            # Ensure both Person nodes exist
            session.execute_write(self._merge_person, person_a, {})
            session.execute_write(self._merge_person, person_b, {})

            session.execute_write(
                self._merge_association, person_a, person_b, props
            )
            stats["relationships_merged"] += 1

        return stats
