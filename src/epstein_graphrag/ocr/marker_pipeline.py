"""Marker + Gemini OCR pipeline for scanned PDF documents.

Two processing tracks:
  - TEXT: Marker + Gemini Flash 3 for high-quality OCR on scanned text documents
  - PHOTOGRAPH: Gemini Flash 3 vision for scene analysis, object detection, face detection

Uses Gemini 2.0 Flash (gemini-3-flash-preview) for both LLM-enhanced OCR and vision analysis.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from google import genai
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from PIL import Image
from tqdm import tqdm

from epstein_graphrag.extract.prompts import VISUAL_ANALYSIS_PROMPT

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
    gemini_api_key: str,
    force_ocr: bool = True,
) -> OCRResult:
    """Process a text document through Marker + Gemini Flash 3 LLM-assisted OCR.

    Uses Marker with Gemini 2.0 Flash for enhanced OCR quality on degraded scans.

    Args:
        pdf_path: Path to the PDF file.
        gemini_api_key: Gemini API key for LLM integration.
        force_ocr: Force OCR even on digital PDFs.

    Returns:
        OCRResult with extracted text and metadata.
    """
    doc_id = pdf_path.stem

    try:
        # Create Marker configuration with Gemini LLM integration
        config = {
            "use_llm": True,
            "gemini_api_key": gemini_api_key,
            "gemini_model_name": "gemini-3-flash-preview",
            "force_ocr": force_ocr,
            "output_format": "markdown",
        }

        config_parser = ConfigParser(config)
        models = create_model_dict()

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )

        # Run OCR
        rendered = converter(str(pdf_path))
        text = rendered.markdown if hasattr(rendered, "markdown") else str(rendered)

        # Extract metadata
        page_count = len(rendered.pages) if hasattr(rendered, "pages") else 1
        confidence = (
            rendered.metadata.get("confidence", 0.85)
            if hasattr(rendered, "metadata")
            else 0.85
        )

        return OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.TEXT,
            text=text,
            confidence=confidence,
            metadata={
                "page_count": page_count,
                "file_path": str(pdf_path),
                "processing_engine": "marker+gemini-3-flash-preview",
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
    gemini_api_key: str,
) -> OCRResult:
    """Process a photograph through Gemini Flash 3 vision analysis.

    Uses Gemini 2.0 Flash for scene description, object detection, anomaly identification.

    Args:
        pdf_path: Path to the PDF file (single-page photograph).
        gemini_api_key: Gemini API key for vision analysis.

    Returns:
        OCRResult with vision analysis in the vision_analysis field.
    """
    doc_id = pdf_path.stem

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=gemini_api_key)

        # Upload the PDF for analysis
        uploaded_file = client.files.upload(file=pdf_path)

        # Run vision analysis
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[VISUAL_ANALYSIS_PROMPT, uploaded_file],
        )

        vision_text = response.text

        # Parse structured response (expecting JSON format from prompt)
        try:
            vision_analysis = json.loads(vision_text)
        except json.JSONDecodeError:
            # Fallback if model doesn't return JSON
            vision_analysis = {
                "scene_description": vision_text,
                "objects_detected": [],
                "anomalies_noted": [],
                "faces_detected": [],
            }

        return OCRResult(
            doc_id=doc_id,
            track=ProcessingTrack.PHOTOGRAPH,
            text="",
            confidence=0.9,
            metadata={
                "page_count": 1,
                "file_path": str(pdf_path),
                "processing_engine": "gemini-3-flash-preview-vision",
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
            vision_analysis=None,
        )


def process_document(
    pdf_path: Path,
    doc_type: str,
    output_dir: Path,
    gemini_api_key: str,
) -> OCRResult | None:
    """Process a single PDF through the appropriate OCR track.

    Args:
        pdf_path: Path to the PDF file.
        doc_type: Classification type from manifest ('text_document', 'photograph', 'mixed').
        output_dir: Directory to write the OCR output JSON.
        gemini_api_key: Gemini API key.

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
        result = process_text_document(pdf_path, gemini_api_key)
    elif doc_type == "photograph":
        result = process_photograph(pdf_path, gemini_api_key)
    elif doc_type == "mixed":
        # Mixed documents get both OCR and vision analysis
        text_result = process_text_document(pdf_path, gemini_api_key)
        photo_result = process_photograph(pdf_path, gemini_api_key)
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
    gemini_api_key: str,
    resume: bool = True,
) -> list[str]:
    """Process a batch of documents through OCR.

    Args:
        manifest: Classification manifest (doc_id -> classification result dict).
        output_dir: Directory to write OCR output JSONs.
        gemini_api_key: Gemini API key.
        resume: Skip documents that already have output files.

    Returns:
        List of doc_ids that were successfully processed.
    """
    processed = []
    failed = []

    for doc_id, classification in tqdm(
        manifest.items(), desc="Processing OCR pipeline"
    ):
        pdf_path = Path(classification["file_path"])
        doc_type = classification["doc_type"]

        try:
            result = process_document(pdf_path, doc_type, output_dir, gemini_api_key)
            if result:
                processed.append(doc_id)
        except Exception as e:
            logger.error(f"Failed to process {doc_id}: {e}")
            failed.append(doc_id)
            # Write error file for retry
            error_path = output_dir / f"{doc_id}.error.json"
            error_path.write_text(
                json.dumps({"doc_id": doc_id, "error": str(e)}, indent=2)
            )

    logger.info(
        f"OCR batch complete: {len(processed)} processed, {len(failed)} failed"
    )

    return processed
