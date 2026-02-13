"""Marker + Vision OCR pipeline for scanned PDF documents.

Two processing tracks:
  - TEXT: Marker for layout extraction + Vision OCR for scanned content
  - PHOTOGRAPH: Vision OCR for scene analysis and object detection

Supports two vision OCR providers:
  - Ollama: Local inference, sequential processing only
  - LM Studio: Local inference, supports concurrent processing (--num-workers)

No API dependencies, no rate limits, no costs.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from tqdm import tqdm

from epstein_graphrag.ocr.duplicate_detector import DuplicateDetector
from epstein_graphrag.ocr.lmstudio_ocr import (
    analyze_photograph as lmstudio_analyze_photograph,
    check_lmstudio_available,
    extract_text_from_pdf as lmstudio_extract_text_from_pdf,
)
from epstein_graphrag.ocr.ollama_ocr import (
    analyze_photograph,
    check_ollama_available,
    extract_text_from_pdf,
)
from epstein_graphrag.ocr.redaction_merger import RedactionMerger

# Type alias for OCR provider
OCRProvider = Literal["ollama", "lmstudio"]

# Make Marker imports optional (for Python 3.14 compatibility)
try:
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    MARKER_AVAILABLE = True
except ImportError:
    MARKER_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProcessingTrack(str, Enum):
    """Processing track for a document."""

    TEXT = "text"
    PHOTOGRAPH = "photograph"
    MIXED = "mixed"


@dataclass
class OCRResult:
    """Result of OCR processing on a single document."""

    doc_id: str
    track: ProcessingTrack
    text: str
    confidence: float
    metadata: dict = field(default_factory=dict)
    vision_analysis: Optional[dict] = None


def process_text_document(
    pdf_path: Path,
    force_ocr: bool = False,
    ocr_provider: OCRProvider = "ollama",
    ocr_model: str = "minicpm-v:8b",
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
    lm_base_url: str = "http://localhost:1234/v1",
) -> OCRResult:
    """Process a text document through Marker + Vision OCR fallback.

    Strategy:
    1. Try Marker first for native PDF text extraction (fast, preserves layout)
    2. If Marker fails or confidence is low, fall back to Vision OCR
    3. If Marker not available (Python 3.14), use Vision OCR directly

    Args:
        pdf_path: Path to PDF file.
        force_ocr: Force Vision OCR even if Marker succeeds.
        ocr_provider: OCR provider - "ollama" or "lmstudio" (default: ollama)
        ocr_model: Model name for OCR (default: minicpm-v:8b)
        use_forensic_context: Whether to use forensic context in OCR (default: True)
        document_type: Type of document for specialized prompts
        has_redactions: Whether document has known redactions
        lm_base_url: Base URL for LM Studio API (default: localhost:1234/v1)

    Returns:
        OCRResult with extracted text and metadata.
    """
    doc_id = pdf_path.stem
    use_vision_ocr = force_ocr or not MARKER_AVAILABLE

    try:
        # Try Marker first (fast for native PDFs) - only if available
        if not use_vision_ocr:
            logger.info(f"Attempting Marker extraction for {doc_id}...")

            # Pre-filter: Skip Marker for large PDFs (>50MB) to avoid memory issues
            pdf_size = pdf_path.stat().st_size
            MAX_PDF_SIZE = 50 * 1024 * 1024  # 50MB
            if pdf_size > MAX_PDF_SIZE:
                logger.warning(
                    f"Large PDF detected ({pdf_size / 1024 / 1024:.1f}MB), "
                    "skipping Marker to avoid memory issues"
                )
                use_vision_ocr = True
            else:
                config_parser = ConfigParser({})  # type: ignore[name-defined]
                config_dict = config_parser.generate_config_dict()
                config_dict["pdftext_workers"] = 1

                model_list = create_model_dict()  # type: ignore[name-defined]
                converter = PdfConverter(  # type: ignore[name-defined]
                    config=config_dict, artifact_dict=model_list, processor_list=[]
                )

                rendered = converter(str(pdf_path))
                marker_text = rendered.markdown

                # Check if Marker extraction was successful
                # Improved confidence calculation with better thresholds
                if not marker_text:
                    confidence = 0.0
                elif len(marker_text.strip()) < 20:  # Very short text
                    confidence = 0.2
                elif len(marker_text) < 100:
                    confidence = 0.4  # Slightly higher threshold
                else:
                    confidence = 0.8

                if confidence > 0.6:
                    logger.info(
                        f"✓ Marker succeeded: {len(marker_text)} chars "
                        f"(confidence: {confidence})"
                    )

                    return OCRResult(
                        doc_id=doc_id,
                        track=ProcessingTrack.TEXT,
                        text=marker_text,
                        confidence=confidence,
                        metadata={
                            "processing_engine": "marker",
                            "text_length": len(marker_text),
                            "processing_time_seconds": rendered.metadata.get(
                                "pdf_conversion_duration", 0
                            ),
                        },
                    )
                else:
                    provider_name = ocr_provider.upper()
                    logger.warning(
                        f"Marker low confidence ({confidence}) - "
                        f"falling back to {provider_name} OCR"
                    )
                    use_vision_ocr = True  # Trigger Vision OCR fallback

        # Use Vision OCR (Ollama or LM Studio) - fallback or forced
        if use_vision_ocr:
            engine_reason = (
                "forced"
                if force_ocr
                else "fallback"
                if MARKER_AVAILABLE
                else "primary"
            )
            logger.info(
                f"Using {ocr_provider.upper()} OCR ({engine_reason}) for {doc_id}..."
            )

            start_time = time.time()

            # Select OCR provider
            if ocr_provider == "lmstudio":
                text, metadata = lmstudio_extract_text_from_pdf(
                    pdf_path,
                    model=ocr_model,
                    use_forensic_context=use_forensic_context,
                    document_type=document_type,
                    has_redactions=has_redactions,
                    base_url=lm_base_url,
                )
            else:  # ollama
                text, metadata = extract_text_from_pdf(
                    pdf_path,
                    model=ocr_model,
                    use_forensic_context=use_forensic_context,
                    document_type=document_type,
                    has_redactions=has_redactions,
                )

            processing_time = time.time() - start_time

            # Vision OCR confidence is high for vision models
            confidence = 0.9 if text and len(text) > 50 else 0.7

            metadata["processing_time_seconds"] = processing_time
            metadata["fallback_reason"] = engine_reason
            metadata["ocr_provider"] = ocr_provider

            logger.info(
                f"✓ {ocr_provider.upper()} OCR complete: {len(text)} chars "
                f"in {processing_time:.1f}s (confidence: {confidence})"
            )

            return OCRResult(
                doc_id=doc_id,
                track=ProcessingTrack.TEXT,
                text=text,
                confidence=confidence,
                metadata=metadata,
            )

    except Exception as e:
        logger.error(f"OCR processing failed for {doc_id}: {e}")
        raise

    # Should never reach here - all paths above either return or raise
    raise RuntimeError("unreachable")  # pragma: no cover


def process_photograph(
    pdf_path: Path,
    ocr_provider: OCRProvider = "ollama",
    ocr_model: str = "minicpm-v:8b",
    use_forensic_context: bool = True,
    lm_base_url: str = "http://localhost:1234/v1",
) -> OCRResult:
    """Process a photograph document with Vision OCR analysis.

    Extracts:
      - Scene description
      - People present (count, appearance, activities)
      - Objects and furniture
      - Any visible text in the image
      - Location indicators

    Args:
        pdf_path: Path to the photograph PDF.
        ocr_provider: OCR provider - "ollama" or "lmstudio" (default: ollama)
        ocr_model: Vision model to use (default: minicpm-v:8b)
        use_forensic_context: Whether to use forensic photograph analysis prompt
        lm_base_url: Base URL for LM Studio API (default: localhost:1234/v1)

    Returns:
        OCRResult with vision analysis.
    """
    doc_id = pdf_path.stem

    logger.info(f"Processing photograph {doc_id} with {ocr_provider.upper()} vision...")

    start_time = time.time()

    # Select OCR provider
    if ocr_provider == "lmstudio":
        text, metadata = lmstudio_analyze_photograph(
            pdf_path,
            model=ocr_model,
            use_forensic_context=use_forensic_context,
            base_url=lm_base_url,
        )
    else:  # ollama
        text, metadata = analyze_photograph(
            pdf_path,
            model=ocr_model,
            use_forensic_context=use_forensic_context,
        )

    processing_time = time.time() - start_time
    confidence = 0.85  # Vision models have high confidence for photographs

    metadata["processing_time_seconds"] = processing_time
    metadata["ocr_provider"] = ocr_provider

    logger.info(
        f"✓ Photograph analysis complete: {len(text)} chars in {processing_time:.1f}s"
    )

    return OCRResult(
        doc_id=doc_id,
        track=ProcessingTrack.PHOTOGRAPH,
        text=text,
        confidence=confidence,
        metadata=metadata,
        vision_analysis={"scene_description": text},
    )


def process_document(
    doc_id: str,
    pdf_path: Path,
    doc_type: str,
    ocr_provider: OCRProvider = "ollama",
    ocr_model: str = "minicpm-v:8b",
    use_forensic_context: bool = True,
    document_type: str = "general",
    lm_base_url: str = "http://localhost:1234/v1",
) -> OCRResult:
    """Process a single document based on its classification.

    Args:
        doc_id: Document ID.
        pdf_path: Path to PDF file.
        doc_type: Document type from classifier ('text_document', 'photograph', 'mixed').
        ocr_provider: OCR provider - "ollama" or "lmstudio" (default: ollama)
        ocr_model: Vision model to use (default: minicpm-v:8b)
        use_forensic_context: Whether to use forensic context in OCR
        document_type: Type of document for specialized prompts
        lm_base_url: Base URL for LM Studio API (default: localhost:1234/v1)

    Returns:
        OCRResult with processed content.
    """
    if doc_type == "text_document":
        return process_text_document(
            pdf_path,
            ocr_provider=ocr_provider,
            ocr_model=ocr_model,
            use_forensic_context=use_forensic_context,
            document_type=document_type,
            lm_base_url=lm_base_url,
        )
    elif doc_type == "photograph":
        return process_photograph(
            pdf_path,
            ocr_provider=ocr_provider,
            ocr_model=ocr_model,
            use_forensic_context=use_forensic_context,
            lm_base_url=lm_base_url,
        )
    elif doc_type == "mixed":
        # Process as text first, then add photograph analysis
        text_result = process_text_document(
            pdf_path,
            ocr_provider=ocr_provider,
            ocr_model=ocr_model,
            use_forensic_context=use_forensic_context,
            document_type=document_type,
            lm_base_url=lm_base_url,
        )
        photo_result = process_photograph(
            pdf_path,
            ocr_provider=ocr_provider,
            ocr_model=ocr_model,
            use_forensic_context=use_forensic_context,
            lm_base_url=lm_base_url,
        )

        # Combine results
        text_result.text += f"\n\n--- VISUAL ANALYSIS ---\n\n{photo_result.text}"
        text_result.vision_analysis = photo_result.vision_analysis
        text_result.track = ProcessingTrack.MIXED

        return text_result
    else:
        raise ValueError(f"Unknown document type: {doc_type}")


def process_batch(
    manifest: dict[str, dict],
    output_dir: Path,
    resume: bool = True,
    ocr_provider: OCRProvider = "ollama",
    ocr_model: str = "minicpm-v:8b",
    use_forensic_context: bool = True,
    detect_duplicates: bool = False,
    merge_duplicates: bool = False,
    num_workers: int = 1,
    lm_base_url: str = "http://localhost:1234/v1",
) -> dict:
    """Process a batch of documents with Vision OCR.

    Args:
        manifest: Document manifest {doc_id: {path, doc_type, ...}}
        output_dir: Directory to save OCR results.
        resume: Skip documents that already have output files.
        ocr_provider: OCR provider - "ollama" or "lmstudio" (default: ollama)
        ocr_model: Vision model to use (default: minicpm-v:8b)
        use_forensic_context: Whether to use forensic context in OCR (default: True)
        detect_duplicates: Whether to detect duplicate PDFs with different redactions
        merge_duplicates: Whether to merge duplicate documents
        num_workers: Number of parallel workers (default: 1, sequential processing)
        lm_base_url: Base URL for LM Studio API (default: localhost:1234/v1)

    Returns:
        Dict with processing statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check OCR provider availability
    if ocr_provider == "lmstudio":
        if not check_lmstudio_available(model=ocr_model, base_url=lm_base_url):
            logger.error(f"LM Studio not available at {lm_base_url}!")
            raise RuntimeError(f"LM Studio not available at {lm_base_url}")
        logger.info(f"✓ LM Studio is available at {lm_base_url}")
    else:  # ollama
        if not check_ollama_available(ocr_model):
            logger.error(f"Ollama model '{ocr_model}' not available!")
            logger.error("Please run: ollama pull minicpm-v:8b")
            raise RuntimeError(f"Ollama model {ocr_model} not found")
        logger.info(f"✓ Ollama model '{ocr_model}' is available")

    # Warn if using Ollama with num_workers > 1
    if ocr_provider == "ollama" and num_workers > 1:
        logger.warning(
            f"Ollama does not support concurrent processing well. "
            f"Using num_workers={num_workers} may cause issues. "
            f"Consider using --ocr-provider lmstudio for better concurrency."
        )

    logger.info(f"Marker available: {MARKER_AVAILABLE}")
    logger.info(f"Forensic context enabled: {use_forensic_context}")
    logger.info(f"Duplicate detection enabled: {detect_duplicates}")
    logger.info(f"OCR provider: {ocr_provider}")
    logger.info(f"OCR model: {ocr_model}")

    # Optional: Detect duplicates before processing
    duplicate_groups = []
    if detect_duplicates:
        logger.info("Detecting duplicate documents...")
        try:
            # Collect all PDF paths from manifest
            pdf_paths = []
            for info in manifest.values():
                path_str = info.get("file_path") or info.get("path")
                if path_str:
                    pdf_paths.append(Path(path_str))

            # Detect duplicates
            detector = DuplicateDetector(detect_redactions=True)
            fingerprints = [detector.fingerprint_document(p) for p in pdf_paths]
            duplicate_groups = detector.find_duplicates(fingerprints)

            # Save duplicate groups
            duplicate_file = output_dir / "duplicate_groups.json"
            detector.save_groups(duplicate_file)

            # Merge if requested
            if merge_duplicates and duplicate_groups:
                logger.info(f"Merging {len(duplicate_groups)} duplicate groups...")
                merger = RedactionMerger(duplicate_groups, ocr_model=ocr_model)
                merged_docs = merger.merge_all()
                merger.save_merged(merged_docs, output_dir / "merged")

        except Exception as e:
            logger.error(f"Duplicate detection failed: {e}")

    total = len(manifest)

    # Thread-safe counters for parallel processing
    counters = {"processed": 0, "skipped": 0, "failed": 0}
    failed_docs_lock = []

    def _process_single_item(doc_id: str, info: dict) -> tuple[str, str, str | None]:
        """Process a single document.

        Returns:
            Tuple of (doc_id, status, error_message)
            Status is one of: "processed", "skipped", "failed"
        """
        output_file = output_dir / f"{doc_id}.json"

        # Resume: skip if output already exists
        if resume and output_file.exists():
            return doc_id, "skipped", None

        pdf_path = None
        try:
            path_str = info.get("file_path") or info.get("path")
            if not path_str:
                raise ValueError("No file_path or path in manifest entry")
            pdf_path = Path(path_str)
            doc_type = info.get("doc_type", "text_document")

            # Determine document type for forensic context
            forensic_doc_type = "general"
            if doc_type == "photograph":
                forensic_doc_type = "photograph"
            elif "legal" in str(path_str).lower() or "court" in str(path_str).lower():
                forensic_doc_type = "legal_document"
            elif "form" in str(path_str).lower():
                forensic_doc_type = "form"

            # Process document
            result = process_document(
                doc_id,
                pdf_path,
                doc_type,
                ocr_provider=ocr_provider,
                ocr_model=ocr_model,
                use_forensic_context=use_forensic_context,
                document_type=forensic_doc_type,
                lm_base_url=lm_base_url,
            )

            # Save result
            with open(output_file, "w") as f:
                json.dump(asdict(result), f, indent=2)

            return doc_id, "processed", None

        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")

            # Write error file
            error_file = output_dir / f"{doc_id}.error.json"
            error_data = {"doc_id": doc_id, "error": str(e)}
            try:
                error_data["path"] = str(pdf_path)
            except Exception:
                pass
            with open(error_file, "w") as f:
                json.dump(error_data, f, indent=2)

            return doc_id, "failed", str(e)

    # Process documents (parallel or sequential based on num_workers)
    if num_workers > 1:
        logger.info(f"Processing with {num_workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_process_single_item, doc_id, info): doc_id
                for doc_id, info in manifest.items()
            }

            # Process completed tasks
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                doc_id, status, error = future.result()
                if status == "processed":
                    counters["processed"] += 1
                elif status == "skipped":
                    counters["skipped"] += 1
                elif status == "failed":
                    counters["failed"] += 1
                    failed_docs_lock.append(doc_id)
    else:
        # Sequential processing (original behavior)
        for doc_id, info in tqdm(manifest.items(), desc="Processing documents"):
            doc_id, status, error = _process_single_item(doc_id, info)
            if status == "processed":
                counters["processed"] += 1
            elif status == "skipped":
                counters["skipped"] += 1
            elif status == "failed":
                counters["failed"] += 1
                failed_docs_lock.append(doc_id)

    return {
        "total": total,
        "processed": counters["processed"],
        "skipped": counters["skipped"],
        "failed": counters["failed"],
        "failed_docs": failed_docs_lock,
    }
