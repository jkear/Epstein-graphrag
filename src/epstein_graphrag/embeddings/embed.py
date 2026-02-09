"""Generate embeddings using nomic-embed-text via Ollama.

Produces 768-dimensional vectors for semantic search via
Neo4j's native vector indexes.
"""

import logging

import ollama

from epstein_graphrag.config import Config

logger = logging.getLogger(__name__)


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
    return response["embeddings"][0]


def embed_batch(texts: list[str], config: Config | None = None) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.
        config: Optional configuration.

    Returns:
        List of 768-dimensional float vectors.
    """
    model = config.embedding_model if config else "nomic-embed-text"
    response = ollama.embed(model=model, input=texts)
    return response["embeddings"]
