"""Marker + Gemini OCR pipeline for scanned PDF documents.

Uses Marker for initial OCR, with Gemini as the vision LLM
for enhanced accuracy on difficult scans.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def process_document(pdf_path: Path, output_dir: Path) -> dict | None:
    """Process a single PDF through the OCR pipeline.

    Args:
        pdf_path: Path to the PDF file.
        output_dir: Directory to write the OCR output JSON.

    Returns:
        OCR result dictionary, or None if processing failed.
    """
    raise NotImplementedError("OCR pipeline not yet implemented — see Task 4")


def process_batch(
    manifest: dict,
    data_root: Path,
    output_dir: Path,
    batch_size: int = 100,
) -> list[str]:
    """Process a batch of documents through OCR.

    Args:
        manifest: Classification manifest (doc_id -> classification).
        data_root: Root directory containing source PDFs.
        output_dir: Directory to write OCR output JSONs.
        batch_size: Number of documents per batch.

    Returns:
        List of doc_ids that were successfully processed.
    """
    raise NotImplementedError("OCR batch processing not yet implemented — see Task 4")
