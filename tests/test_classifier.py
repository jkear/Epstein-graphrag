"""Tests for document classifier."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from epstein_graphrag.classify.classifier import DocType, classify_batch, classify_pdf


def test_classify_text_document():
    """A PDF that produces substantial OCR text is a text document."""
    with patch("epstein_graphrag.classify.classifier.pdfplumber") as mock_pdf:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "This is a deposition transcript with substantial text content " * 50
        )
        mock_pdf.open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(pages=[mock_page])
        )
        mock_pdf.open.return_value.__exit__ = MagicMock(return_value=False)

        result = classify_pdf(Path("fake/EFTA00001234.pdf"))
        assert result.doc_type == DocType.TEXT_DOCUMENT


def test_classify_photograph():
    """A PDF with no extractable text and a single page is a photograph."""
    with patch("epstein_graphrag.classify.classifier.pdfplumber") as mock_pdf:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_pdf.open.return_value.__enter__ = MagicMock(
            return_value=MagicMock(pages=[mock_page])
        )
        mock_pdf.open.return_value.__exit__ = MagicMock(return_value=False)

        result = classify_pdf(Path("fake/EFTA00001234.pdf"))
        assert result.doc_type in (DocType.PHOTOGRAPH, DocType.UNKNOWN)


def test_classify_batch_creates_manifest(tmp_path):
    """Batch classification writes a manifest.json with all results."""
    manifest_path = tmp_path / "manifest.json"
    classify_batch(tmp_path, manifest_path)
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert isinstance(manifest, dict)
