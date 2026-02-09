"""Classify PDFs as text documents, photographs, mixed, or unknown.

Classification strategy:
1. Attempt text extraction with pdfplumber
2. If text length > 200 chars -> TEXT_DOCUMENT
3. If text length == 0 and page_count == 1 -> PHOTOGRAPH
4. If text length > 0 but < 200 and page_count == 1 -> MIXED
5. Otherwise -> UNKNOWN (flag for manual review)
"""

import json
import logging
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    TEXT_DOCUMENT = "text_document"
    PHOTOGRAPH = "photograph"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of classifying a single PDF."""

    doc_id: str
    file_path: str
    doc_type: DocType
    page_count: int
    text_char_count: int
    file_size_bytes: int


def classify_pdf(pdf_path: Path) -> ClassificationResult:
    """Classify a single PDF by attempting text extraction."""
    doc_id = pdf_path.stem
    file_size = pdf_path.stat().st_size if pdf_path.exists() else 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text() or ""
                full_text += text

            text_len = len(full_text.strip())

            if text_len > 200:
                doc_type = DocType.TEXT_DOCUMENT
            elif text_len == 0 and page_count == 1:
                doc_type = DocType.PHOTOGRAPH
            elif 0 < text_len <= 200 and page_count == 1:
                doc_type = DocType.MIXED
            else:
                doc_type = DocType.UNKNOWN

    except Exception as e:
        logger.warning(f"Failed to classify {doc_id}: {e}")
        doc_type = DocType.UNKNOWN
        page_count = 0
        text_len = 0

    return ClassificationResult(
        doc_id=doc_id,
        file_path=str(pdf_path),
        doc_type=doc_type,
        page_count=page_count,
        text_char_count=text_len,
        file_size_bytes=file_size,
    )


def classify_batch(
    data_dir: Path,
    manifest_path: Path,
    resume: bool = True,
) -> dict[str, ClassificationResult]:
    """Classify all PDFs in a directory tree and write manifest.json.

    Args:
        data_dir: Root directory to scan for PDFs.
        manifest_path: Path to write the manifest JSON.
        resume: If True, skip files already in existing manifest.

    Returns:
        Dictionary mapping doc_id to ClassificationResult.
    """
    existing: dict[str, dict] = {}
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        logger.info(f"Resuming: {len(existing)} documents already classified")

    pdf_files = sorted(data_dir.rglob("*.pdf"))
    results: dict[str, ClassificationResult] = {}

    for pdf_path in tqdm(pdf_files, desc="Classifying PDFs"):
        doc_id = pdf_path.stem
        if doc_id in existing:
            continue

        result = classify_pdf(pdf_path)
        results[doc_id] = result

    # Merge with existing
    merged = existing.copy()
    for doc_id, result in results.items():
        merged[doc_id] = asdict(result)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(merged, indent=2, default=str))
    logger.info(f"Manifest written: {len(merged)} total documents")

    return results
