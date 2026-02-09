"""Tests for OCR pipeline."""

import pytest


def test_marker_pipeline_not_implemented():
    """OCR pipeline raises NotImplementedError until Task 4."""
    from pathlib import Path

    from epstein_graphrag.ocr.marker_pipeline import process_document

    with pytest.raises(NotImplementedError):
        process_document(Path("fake.pdf"), Path("output"))
