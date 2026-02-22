"""Classify PDFs as text documents, photographs, mixed, or unknown.

Classification strategy:
1. Attempt text extraction with pdfplumber
2. If text length > 200 chars -> TEXT_DOCUMENT (save text immediately!)
3. If text length == 0 and page_count == 1 -> PHOTOGRAPH
4. If text length > 0 but < 200 and page_count == 1 -> MIXED
5. Otherwise -> UNKNOWN (flag for manual review)

For TEXT_DOCUMENT types, we save the extracted text directly - no OCR needed!
"""

import json
import logging
import multiprocessing as mp
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import pdfplumber
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Number of parallel workers for classification
DEFAULT_WORKERS = mp.cpu_count()


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
    # New: store extracted text for text_documents to skip OCR
    extracted_text: str | None = None


def classify_pdf(pdf_path: Path, extract_full_text: bool = True) -> ClassificationResult:
    """Classify a single PDF by attempting text extraction.
    
    Args:
        pdf_path: Path to PDF file
        extract_full_text: If True, extract full text for text_documents (saves OCR!)
    """
    doc_id = pdf_path.stem
    file_size = pdf_path.stat().st_size if pdf_path.exists() else 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            
            # First pass: check first 3 pages to determine type
            sample_text = ""
            for page in pdf.pages[:3]:
                text = page.extract_text() or ""
                sample_text += text
                if len(sample_text) > 200:
                    break

            text_len = len(sample_text.strip())

            if text_len > 200:
                doc_type = DocType.TEXT_DOCUMENT
                # Extract FULL text since we can skip OCR for this doc!
                if extract_full_text:
                    full_text = ""
                    for page in pdf.pages:
                        full_text += (page.extract_text() or "") + "\n\n"
                    extracted_text = full_text.strip()
                else:
                    extracted_text = sample_text
            elif text_len == 0 and page_count == 1:
                doc_type = DocType.PHOTOGRAPH
                extracted_text = None
            elif 0 < text_len <= 200 and page_count == 1:
                doc_type = DocType.MIXED
                extracted_text = None
            else:
                doc_type = DocType.UNKNOWN
                extracted_text = None

    except Exception as e:
        logger.warning(f"Failed to classify {doc_id}: {e}")
        doc_type = DocType.UNKNOWN
        page_count = 0
        text_len = 0
        extracted_text = None

    return ClassificationResult(
        doc_id=doc_id,
        file_path=str(pdf_path),
        doc_type=doc_type,
        page_count=page_count,
        text_char_count=text_len,
        file_size_bytes=file_size,
        extracted_text=extracted_text,
    )


def _classify_worker(pdf_path_str: str) -> dict | None:
    """Worker function for parallel classification."""
    try:
        result = classify_pdf(Path(pdf_path_str))
        return asdict(result)
    except Exception as e:
        logger.error(f"Worker error for {pdf_path_str}: {e}")
        return None


def classify_batch(
    data_dir: Path,
    manifest_path: Path,
    resume: bool = True,
    num_workers: int | None = None,
) -> dict[str, ClassificationResult]:
    """Classify all PDFs in a directory tree and write manifest.json.

    Args:
        data_dir: Root directory to scan for PDFs.
        manifest_path: Path to write the manifest JSON.
        resume: If True, skip files already in existing manifest.
        num_workers: Number of parallel workers (default: CPU count).

    Returns:
        Dictionary mapping doc_id to ClassificationResult.
    """
    if num_workers is None:
        num_workers = DEFAULT_WORKERS
    
    existing: dict[str, dict] = {}
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        logger.info(f"Resuming: {len(existing)} documents already classified")

    logger.info(f"Scanning {data_dir} for PDFs...")
    pdf_files = sorted(data_dir.rglob("*.pdf"))
    total_files = len(pdf_files)
    logger.info(f"Found {total_files} PDF files")
    
    # Filter out already classified
    to_process = [p for p in pdf_files if p.stem not in existing]
    logger.info(f"Need to classify: {len(to_process)} new files")
    
    if not to_process:
        logger.info("All files already classified")
        return {}

    results: dict[str, ClassificationResult] = {}
    
    # Use parallel processing for speed
    logger.info(f"Classifying with {num_workers} parallel workers...")
    
    with mp.Pool(processes=num_workers) as pool:
        pdf_paths_str = [str(p) for p in to_process]
        
        for result_dict in tqdm(
            pool.imap_unordered(_classify_worker, pdf_paths_str, chunksize=100),
            total=len(to_process),
            desc="Classifying PDFs",
        ):
            if result_dict:
                doc_id = result_dict["doc_id"]
                # Don't store full text in manifest (too big) - store separately
                extracted_text = result_dict.pop("extracted_text", None)
                results[doc_id] = ClassificationResult(**result_dict, extracted_text=None)
                
                # Save text_documents directly to processed/ (skip OCR later!)
                if result_dict.get("doc_type") == "text_document" and extracted_text:
                    processed_dir = manifest_path.parent / "processed"
                    processed_dir.mkdir(parents=True, exist_ok=True)
                    output_file = processed_dir / f"{doc_id}.json"
                    if not output_file.exists():
                        output_data = {
                            "doc_id": doc_id,
                            "source_path": result_dict["file_path"],
                            "text": extracted_text,
                            "metadata": {
                                "extraction_method": "pdfplumber",
                                "page_count": result_dict["page_count"],
                                "char_count": len(extracted_text),
                            }
                        }
                        output_file.write_text(json.dumps(output_data, indent=2))

    # Merge with existing (without extracted_text to keep manifest small)
    merged = existing.copy()
    for doc_id, result in results.items():
        result_dict = asdict(result)
        result_dict.pop("extracted_text", None)  # Don't store in manifest
        merged[doc_id] = result_dict

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(merged, indent=2, default=str))
    
    # Count what we saved
    text_docs = sum(1 for r in results.values() if r.doc_type == DocType.TEXT_DOCUMENT)
    needs_ocr = len(results) - text_docs
    
    logger.info(f"Manifest written: {len(merged)} total documents")
    logger.info(f"  Text extracted (no OCR needed): {text_docs}")
    logger.info(f"  Needs VLM OCR: {needs_ocr}")

    return results
