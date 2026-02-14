"""Generate embeddings using nomic-embed-text via Ollama.

Produces 768-dimensional vectors for semantic search via
Neo4j's native vector indexes.

Two layers:
  1. Low-level helpers: embed_text(), embed_batch()
  2. NodeEmbedder — reads nodes from Neo4j, generates embeddings,
     writes them back. Resume-capable (only processes nodes where
     the embedding property IS NULL).
"""

import logging
from dataclasses import dataclass

import ollama
from neo4j import GraphDatabase

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Low-level embedding helpers                                        #
# ------------------------------------------------------------------ #


def embed_text(text: str, config: Config | None = None) -> list[float]:
    """Generate an embedding vector for a text string.

    Args:
        text: The text to embed.
        config: Optional configuration (uses defaults if not provided).

    Returns:
        768-dimensional float vector.
    """
    model = config.embedding_model if config else "nomic-embed-text"
    response = ollama.embed(model=model, input=text)
    return list(response.embeddings[0])


def embed_batch(texts: list[str], config: Config | None = None) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.
        config: Optional configuration.

    Returns:
        List of 768-dimensional float vectors.
    """
    if not texts:
        return []
    model = config.embedding_model if config else "nomic-embed-text"
    response = ollama.embed(model=model, input=texts)
    return [list(emb) for emb in response.embeddings]


# ------------------------------------------------------------------ #
#  Embedding target definitions                                       #
# ------------------------------------------------------------------ #


@dataclass
class EmbeddingTarget:
    """Defines which nodes need embeddings and how to build the text."""

    label: str
    """Neo4j node label (e.g. 'Event')."""

    embedding_property: str
    """Property where the 768d vector is stored (e.g. 'description_embedding')."""

    text_property: str
    """Property containing the source text (e.g. 'description')."""

    text_builder: str = "single"
    """How to build text from properties:
    - 'single': just use text_property
    - 'name_description': combine name + description
    """

    key_property: str = ""
    """The node key property used for MATCH (e.g. 'event_id', 'name')."""


# Which node types get embedded, matching the vector index schema.
EMBEDDING_TARGETS: list[EmbeddingTarget] = [
    EmbeddingTarget(
        label="Event",
        embedding_property="description_embedding",
        text_property="description",
        text_builder="single",
        key_property="event_id",
    ),
    EmbeddingTarget(
        label="Allegation",
        embedding_property="embedding",
        text_property="description",
        text_builder="single",
        key_property="allegation_id",
    ),
    EmbeddingTarget(
        label="Person",
        embedding_property="description_embedding",
        text_property="description",
        text_builder="name_description",
        key_property="name",
    ),
    EmbeddingTarget(
        label="Location",
        embedding_property="description_embedding",
        text_property="description",
        text_builder="name_description",
        key_property="name",
    ),
]


# ------------------------------------------------------------------ #
#  NodeEmbedder — orchestrates reading, embedding, and writing        #
# ------------------------------------------------------------------ #


class NodeEmbedder:
    """Reads graph nodes, generates embeddings, writes them back.

    Resume-capable: only processes nodes where the embedding property
    IS NULL and the text property IS NOT NULL. Running twice is safe.

    Args:
        config: Application configuration with Neo4j and embedding settings.
        batch_size: How many texts to send to Ollama in one embed() call.
    """

    def __init__(self, config: Config, batch_size: int = 50):
        self.config = config
        self.batch_size = batch_size
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

    def close(self) -> None:
        """Close the Neo4j driver."""
        self.driver.close()

    def _get_nodes_needing_embeddings(self, target: EmbeddingTarget) -> list[dict]:
        """Query nodes that need embeddings.

        Returns nodes where the embedding property IS NULL
        and the text property IS NOT NULL (there's text to embed).
        """
        query = (
            f"MATCH (n:{target.label}) "
            f"WHERE n.{target.embedding_property} IS NULL "
            f"  AND n.{target.text_property} IS NOT NULL "
            f"  AND trim(n.{target.text_property}) <> '' "
            f"RETURN n.{target.key_property} AS key, "
            f"  n.{target.text_property} AS text"
        )
        if target.text_builder == "name_description":
            # Also fetch name for combined text
            query = (
                f"MATCH (n:{target.label}) "
                f"WHERE n.{target.embedding_property} IS NULL "
                f"  AND n.{target.text_property} IS NOT NULL "
                f"  AND trim(n.{target.text_property}) <> '' "
                f"RETURN n.{target.key_property} AS key, "
                f"  n.name AS name, "
                f"  n.{target.text_property} AS text"
            )

        records, _, _ = self.driver.execute_query(query)
        return [dict(r) for r in records]

    def _build_text(self, node: dict, target: EmbeddingTarget) -> str:
        """Build the text string to embed from node properties."""
        if target.text_builder == "name_description":
            name = node.get("name", "")
            text = node.get("text", "")
            return f"{name}: {text}" if name else text
        return node.get("text", "")

    def _write_embeddings(
        self,
        target: EmbeddingTarget,
        keys: list[str],
        vectors: list[list[float]],
    ) -> int:
        """Write embedding vectors back to Neo4j nodes.

        Returns the number of nodes updated.
        """
        updated = 0
        for key, vector in zip(keys, vectors):
            query = (
                f"MATCH (n:{target.label} "
                f"{{{target.key_property}: $key}}) "
                f"SET n.{target.embedding_property} = $vector"
            )
            self.driver.execute_query(query, key=key, vector=vector)
            updated += 1
        return updated

    def embed_target(self, target: EmbeddingTarget) -> dict:
        """Generate embeddings for one node type.

        Args:
            target: The EmbeddingTarget defining which nodes to process.

        Returns:
            Dict with 'label', 'total', 'embedded', 'skipped' counts.
        """
        nodes = self._get_nodes_needing_embeddings(target)

        if not nodes:
            logger.info(f"{target.label}: no nodes need embeddings")
            return {
                "label": target.label,
                "total": 0,
                "embedded": 0,
                "skipped": 0,
            }

        logger.info(f"{target.label}: {len(nodes)} nodes need embeddings")

        # Build texts
        texts = [self._build_text(n, target) for n in nodes]
        keys = [n["key"] for n in nodes]

        # Filter out any empty texts (shouldn't happen but be safe)
        valid = [(k, t) for k, t in zip(keys, texts) if t.strip()]
        skipped = len(keys) - len(valid)
        if skipped:
            logger.warning(f"{target.label}: skipping {skipped} nodes with empty text")

        if not valid:
            return {
                "label": target.label,
                "total": len(nodes),
                "embedded": 0,
                "skipped": skipped,
            }

        valid_keys, valid_texts = zip(*valid)
        valid_keys = list(valid_keys)
        valid_texts = list(valid_texts)

        # Generate embeddings in batches
        all_vectors: list[list[float]] = []
        for i in range(0, len(valid_texts), self.batch_size):
            batch_texts = valid_texts[i : i + self.batch_size]
            batch_vectors = embed_batch(batch_texts, self.config)
            all_vectors.extend(batch_vectors)
            logger.debug(
                f"{target.label}: embedded batch {i // self.batch_size + 1} "
                f"({len(batch_texts)} texts)"
            )

        # Write to Neo4j
        updated = self._write_embeddings(target, valid_keys, all_vectors)
        logger.info(f"{target.label}: wrote {updated} embeddings")

        return {
            "label": target.label,
            "total": len(nodes),
            "embedded": updated,
            "skipped": skipped,
        }

    def embed_all(self) -> dict:
        """Generate embeddings for all configured node types.

        Returns:
            Summary dict with per-label stats and totals.
        """
        results: list[dict] = []
        for target in EMBEDDING_TARGETS:
            result = self.embed_target(target)
            results.append(result)

        total_embedded = sum(r["embedded"] for r in results)
        total_skipped = sum(r["skipped"] for r in results)
        total_nodes = sum(r["total"] for r in results)

        summary = {
            "targets": results,
            "total_nodes": total_nodes,
            "total_embedded": total_embedded,
            "total_skipped": total_skipped,
        }

        logger.info(
            f"Embedding complete: {total_embedded} vectors generated "
            f"across {len(results)} node types"
        )

        return summary
