"""LLM-based entity extraction from OCR text.

Extracts persons, organizations, locations, events,
allegations, and relationships from document text using
a local LLM via Ollama.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_entities(text: str, doc_id: str) -> dict:
    """Extract entities and relationships from document text.

    Args:
        text: OCR text from the document.
        doc_id: Document identifier for provenance tracking.

    Returns:
        Dictionary with 'entities' and 'relationships' keys.
    """
    raise NotImplementedError("Entity extraction not yet implemented — see Task 5")


def extract_batch(
    processed_dir: Path,
    output_dir: Path,
    batch_size: int = 100,
) -> list[str]:
    """Extract entities from a batch of processed documents.

    Args:
        processed_dir: Directory containing OCR output JSONs.
        output_dir: Directory to write extraction output JSONs.
        batch_size: Number of documents per batch.

    Returns:
        List of doc_ids that were successfully processed.
    """
    raise NotImplementedError("Batch extraction not yet implemented — see Task 5")
