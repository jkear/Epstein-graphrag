"""Marker + Ollama Vision OCR pipeline for scanned PDF documents.

Two processing tracks:
  - TEXT: Marker for layout extraction + Ollama vision (MiniCPM-V) for scanned content
  - PHOTOGRAPH: Ollama vision (MiniCPM-V) for scene analysis and object detection

Uses MiniCPM-V via Ollama for local inference.
No API dependencies, no rate limits, no costs.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from epstein_graphrag.ocr.deepseek_ocr import (
    analyze_photograph as ds_analyze_photograph,
)
from epstein_graphrag.ocr.deepseek_ocr import (
    extract_text_from_pdf as ds_extract_text_from_pdf,
)
from epstein_graphrag.ocr.deepseek_ocr import (
    load_deepseek_model,
)

# Force Surya/Marker to use CPU on Apple Silicon (MPS).
# Surya 0.17.x has MPS bugs with bfloat16 and meta tensors.
if not os.environ.get("TORCH_DEVICE"):
    try:
        import torch

        if torch.backends.mps.is_available():
            os.environ["TORCH_DEVICE"] = "cpu"
    except ImportError:
        pass

# Make Marker imports optional (for Python 3.14 compatibility)
try:
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    MARKER_AVAILABLE = True
except ImportError as e:
    MARKER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Marker not available (Python 3.14 compatibility issue): {e}")

logger = logging.getLogger(__name__)


# Model cache for DeepSeek
_deepseek_model_cache = None


def get_or_load_deepseek_model():
    """Load or return cached DeepSeek model.

    Returns:
        Tuple of (model, processor).
    """
    global _deepseek_model_cache

    if _deepseek_model_cache is None:
        _deepseek_model_cache = load_deepseek_model()

    return _deepseek_model_cache


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
    gemini_api_key: str = "",
    force_ocr: bool = False,
) -> OCRResult:
    """Process a text document through Marker + DeepSeek-OCR fallback.

    Strategy:
    1. Try Marker first for native PDF text extraction (fast, preserves layout)
    2. If Marker fails or confidence is low, fall back to DeepSeek-OCR-MLX
    3. If Marker not available (Python 3.14), use DeepSeek-OCR directly

    Args:
        pdf_path: Path to the PDF file.
        force_ocr: Force DeepSeek-OCR even if Marker succeeds.

    Returns:
        OCRResult with extracted text and metadata.
    """
    doc_id = pdf_path.stem
    use_deepseek = force_ocr or not MARKER_AVAILABLE

    try:
        # Try Marker first (fast for native PDFs) - only if available
        if MARKER_AVAILABLE and not force_ocr:
            try:
                config = {
                    "force_ocr": False,
                    "output_format": "markdown",
                }

                config_parser = ConfigParser(config)
                models = create_model_dict()

                converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=models,
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                )

                # Run Marker
                rendered = converter(str(pdf_path))
                text = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)

                # Check if Marker succeeded
                page_count = len(rendered.pages) if hasattr(rendered, "pages") else 1
                confidence = (
                    rendered.metadata.get("confidence", 0.85)
                    if hasattr(rendered, "metadata")
                    else 0.85
                )

                # If Marker got good text, use it
                if text and len(text.strip()) > 100 and confidence > 0.6:
                    logger.debug(f"{doc_id}: Marker succeeded (confidence={confidence:.2f})")
                    return OCRResult(
                        doc_id=doc_id,
                        track=ProcessingTrack.TEXT,
                        text=text,
                        confidence=confidence,
                        metadata={
                            "page_count": page_count,
                            "file_path": str(pdf_path),
                            "processing_engine": "marker",
                        },
                    )
                else:
                    logger.debug(
                        f"{doc_id}: Marker confidence low or insufficient text, "
                        "falling back to DeepSeek-OCR"
                    )
                    use_deepseek = True

            except Exception as e:
                logger.debug(f"{doc_id}: Marker failed ({e}), falling back to DeepSeek-OCR")
                use_deepseek = True

        # Fall back to DeepSeek-OCR for scanned documents or when Marker unavailable
        if use_deepseek:
            if not MARKER_AVAILABLE:
                logger.debug(f"{doc_id}: Using DeepSeek-OCR (Marker not available)")
            else:
                logger.debug(f"{doc_id}: Using DeepSeek-OCR")

            model, processor = get_or_load_deepseek_model()

            text, metadata = ds_extract_text_from_pdf(
                pdf_path,
                model=model,
                processor=processor,
            )

            return OCRResult(
                doc_id=doc_id,
                track=ProcessingTrack.TEXT,
                text=text,
                confidence=0.9,  # DeepSeek-OCR is high quality
                metadata={
                    **metadata,
                    "file_path": str(pdf_path),
                    "processing_engine": "deepseek-ocr-mlx-8bit",
                },
            )

    except Exception as e:
        logger.error(f"Failed to process text document {doc_id}: {e}")
        return OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.TEXT,
            text="",
            confidence=0.0,
            metadata={"error": str(e), "file_path": str(pdf_path)},
        )


def process_photograph(
    pdf_path: Path,
) -> OCRResult:
    """Process a photograph through DeepSeek-OCR-MLX vision analysis.

    Uses DeepSeek-OCR for scene description, object detection, and text extraction.

    Args:
        pdf_path: Path to the PDF file (single-page photograph).

    Returns:
        OCRResult with vision analysis text and metadata.
    """
    doc_id = pdf_path.stem

    try:
        # Load model
        model, processor = get_or_load_deepseek_model()

        # Analyze photograph using DeepSeek
        analysis_text, metadata = ds_analyze_photograph(
            pdf_path,
            model=model,
            processor=processor,
        )

        # Parse as structured vision analysis (if formatted correctly)
        vision_analysis = {
            "raw_analysis": analysis_text,
            "model": "deepseek-ocr-mlx-8bit",
        }

        return OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.PHOTOGRAPH,
            text=analysis_text,  # Store analysis as text
            confidence=0.9,
            metadata={
                **metadata,
                "file_path": str(pdf_path),
                "processing_engine": "deepseek-ocr-mlx-8bit-vision",
            },
            vision_analysis=vision_analysis,
        )

    except Exception as e:
        logger.error(f"Failed to process photograph {doc_id}: {e}")
        return OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.PHOTOGRAPH,
            text="",
            confidence=0.0,
            metadata={"error": str(e), "file_path": str(pdf_path)},
        )


def process_document(
    pdf_path: Path,
    doc_type: str,
    output_dir: Path,
    gemini_api_key: str = "",
) -> OCRResult | None:
    """Process a single PDF through the appropriate OCR track.

    Args:
        pdf_path: Path to the PDF file.
        doc_type: Classification type from manifest ('text_document', 'photograph', 'mixed').
        output_dir: Directory to write the OCR output JSON.

    Returns:
        OCRResult, or None if processing failed.
    """
    doc_id = pdf_path.stem
    output_path = output_dir / f"{doc_id}.json"

    # Skip if already processed
    if output_path.exists():
        logger.debug(f"Skipping {doc_id} — already processed")
        return None

    # Route to appropriate processing track
    if doc_type == "text_document":
        result = process_text_document(pdf_path)
    elif doc_type == "photograph":
        result = process_photograph(pdf_path)
    elif doc_type == "mixed":
        # Mixed documents get both OCR and vision analysis
        text_result = process_text_document(pdf_path)
        photo_result = process_photograph(pdf_path)
        result = OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.MIXED,
            text=text_result.text,
            confidence=(text_result.confidence + photo_result.confidence) / 2,
            metadata={
                **text_result.metadata,
                "has_vision_analysis": True,
            },
            vision_analysis=photo_result.vision_analysis,
        )
    else:
        logger.warning(f"Unknown doc_type {doc_type} for {doc_id}, skipping")
        return None

    # Write result to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2))
    logger.debug(f"Wrote {output_path}")

    return result


def process_batch(
    manifest: dict,
    output_dir: Path,
    gemini_api_key: str = "",
    resume: bool = True,
) -> dict:
    """Process a batch of documents through OCR.

    Args:
        manifest: Classification manifest (doc_id -> classification result dict).
        output_dir: Directory to write OCR output JSONs.
        resume: Skip documents that already have output files.

    Returns:
        Dict with stats: total, processed, skipped, failed, failed_docs.
    """
    processed = []
    skipped = []
    failed = []

    # Pre-load DeepSeek model once for entire batch
    logger.info("Pre-loading DeepSeek-OCR model for batch processing...")
    get_or_load_deepseek_model()

    for doc_id, classification in tqdm(manifest.items(), desc="Processing OCR pipeline"):
        pdf_path = Path(classification["file_path"])
        doc_type = classification["doc_type"]

        # Check if already processed (resume capability)
        output_file = output_dir / f"{doc_id}.json"
        if resume and output_file.exists():
            logger.debug(f"Skipping {doc_id} (already processed)")
            skipped.append(doc_id)
            continue

        try:
            result = process_document(
                pdf_path,
                doc_type,
                output_dir,
                gemini_api_key=gemini_api_key,
            )
            if result:
                processed.append(doc_id)
        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")
            failed.append(doc_id)
            # Write error file for retry
            error_path = output_dir / f"{doc_id}.error.json"
            error_path.write_text(json.dumps({"doc_id": doc_id, "error": str(e)}, indent=2))

    logger.info(
        f"OCR batch complete: {len(processed)} processed, "
        f"{len(skipped)} skipped, {len(failed)} failed"
    )

    return {
        "total": len(manifest),
        "processed": len(processed),
        "skipped": len(skipped),
        "failed": len(failed),
        "failed_docs": failed,
    }
