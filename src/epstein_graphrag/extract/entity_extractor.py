"""Entity extraction using Gemini Flash 3.

Reads processed OCR/vision JSON files and extracts structured entities
using Gemini 3 Flash (gemini-3-flash-preview) via the Google Gen AI SDK.
"""
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

from google import genai
from google.genai import types
from tqdm import tqdm

from epstein_graphrag.extract.prompts import (
    TEXT_ENTITY_EXTRACTION_PROMPT,
    PHOTO_ENTITY_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Structured entity extraction result."""

    doc_id: str
    doc_type: str
    people: list
    locations: list
    organizations: list
    events: list
    allegations: list
    associations: list
    objects_of_interest: list  # From photographs
    raw_llm_response: str  # Keep the raw response for debugging


def extract_from_text(
    doc_id: str,
    doc_type: str,
    text: str,
    gemini_api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> ExtractionResult:
    """Extract entities from an OCR'd text document using Gemini Flash.

    Args:
        doc_id: Document identifier (e.g., EFTA00002012)
        doc_type: Document type (text_document, mixed, etc.)
        text: OCR text from the document
        gemini_api_key: Google Gemini API key
        model_name: Gemini model name

    Returns:
        ExtractionResult with structured entities

    Raises:
        ValueError: If extraction fails or response is invalid JSON
    """
    # Initialize client
    client = genai.Client(api_key=gemini_api_key)

    # Truncate text to fit context window (8000 chars max)
    truncated_text = text[:8000]
    if len(text) > 8000:
        logger.warning(f"{doc_id}: Text truncated from {len(text)} to 8000 chars")

    # Build prompt
    prompt = TEXT_ENTITY_EXTRACTION_PROMPT.format(
        doc_id=doc_id,
        doc_type=doc_type,
        text=truncated_text,
    )

    # Call Gemini with JSON mode
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw_response = response.text
        parsed = json.loads(raw_response)

        # Validate structure
        expected_keys = {"people", "locations", "organizations", "events", "allegations", "associations"}
        missing_keys = expected_keys - set(parsed.keys())
        if missing_keys:
            logger.warning(f"{doc_id}: Missing keys in response: {missing_keys}")
            for key in missing_keys:
                parsed[key] = []

        return ExtractionResult(
            doc_id=doc_id,
            doc_type=doc_type,
            people=parsed.get("people", []),
            locations=parsed.get("locations", []),
            organizations=parsed.get("organizations", []),
            events=parsed.get("events", []),
            allegations=parsed.get("allegations", []),
            associations=parsed.get("associations", []),
            objects_of_interest=[],  # Text documents don't have objects
            raw_llm_response=raw_response,
        )

    except json.JSONDecodeError as e:
        logger.error(f"{doc_id}: Failed to parse JSON response: {e}")
        raise ValueError(f"Invalid JSON response: {e}") from e
    except Exception as e:
        logger.error(f"{doc_id}: Extraction failed: {e}")
        raise


def extract_from_photo(
    doc_id: str,
    vision_analysis: dict,
    gemini_api_key: str,
    model_name: str = "gemini-2.5-flash",
) -> ExtractionResult:
    """Extract entities from a photograph's vision analysis using Gemini Flash.

    Args:
        doc_id: Document identifier
        vision_analysis: Vision analysis dict from OCR pipeline
        gemini_api_key: Google Gemini API key
        model_name: Gemini model name

    Returns:
        ExtractionResult with photo-specific entities
    """
    # Initialize client
    client = genai.Client(api_key=gemini_api_key)

    # Build prompt
    prompt = PHOTO_ENTITY_EXTRACTION_PROMPT.format(
        doc_id=doc_id,
        scene_description=vision_analysis.get("scene_description", ""),
        objects_detected=vision_analysis.get("objects_detected", []),
        anomalies_noted=vision_analysis.get("anomalies_noted", []),
        faces_detected=vision_analysis.get("faces_detected", []),
        visible_text=vision_analysis.get("visible_text", []),
        estimated_period=vision_analysis.get("estimated_period", ""),
        evidence_relevance=vision_analysis.get("evidence_relevance", ""),
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw_response = response.text
        parsed = json.loads(raw_response)

        return ExtractionResult(
            doc_id=doc_id,
            doc_type="photograph",
            people=[],  # Photos don't extract people directly
            locations=parsed.get("locations", []),
            organizations=[],
            events=[],
            allegations=parsed.get("potential_allegations_supported", []),
            associations=[],
            objects_of_interest=parsed.get("objects_of_interest", []),
            raw_llm_response=raw_response,
        )

    except json.JSONDecodeError as e:
        logger.error(f"{doc_id}: Failed to parse JSON response: {e}")
        raise ValueError(f"Invalid JSON response: {e}") from e
    except Exception as e:
        logger.error(f"{doc_id}: Photo extraction failed: {e}")
        raise


def extract_batch(
    processed_dir: Path,
    extracted_dir: Path,
    gemini_api_key: str,
    resume: bool = True,
) -> dict:
    """Extract entities from all processed documents.

    Reads from processed_dir (OCR output), writes to extracted_dir.
    Supports resume — skips already-extracted documents.

    Args:
        processed_dir: Directory containing OCR output JSONs
        extracted_dir: Directory to write extraction output JSONs
        gemini_api_key: Google Gemini API key
        resume: If True, skip already-extracted documents

    Returns:
        Dict with stats: total, processed, skipped, failed, failed_docs
    """
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # Find all OCR JSON files
    json_files = sorted(processed_dir.glob("*.json"))
    # Exclude error files
    json_files = [f for f in json_files if not f.name.endswith(".error.json")]

    stats = {
        "total": len(json_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failed_docs": [],
    }

    for json_file in tqdm(json_files, desc="Extracting entities"):
        doc_id = json_file.stem
        output_path = extracted_dir / f"{doc_id}.json"

        # Resume capability
        if resume and output_path.exists():
            logger.debug(f"{doc_id}: Already extracted, skipping")
            stats["skipped"] += 1
            continue

        try:
            # Load OCR result
            data = json.loads(json_file.read_text())
            track = data.get("track", "text")

            # Route to appropriate extraction function
            if track == "photograph":
                result = extract_from_photo(
                    doc_id=doc_id,
                    vision_analysis=data.get("vision_analysis", {}),
                    gemini_api_key=gemini_api_key,
                )
            else:
                # text or mixed
                result = extract_from_text(
                    doc_id=doc_id,
                    doc_type=track,
                    text=data.get("text", ""),
                    gemini_api_key=gemini_api_key,
                )

            # Write extraction result
            output_path.write_text(json.dumps(asdict(result), indent=2))
            logger.info(f"{doc_id}: Extraction complete")
            stats["processed"] += 1

        except Exception as e:
            logger.error(f"{doc_id}: Entity extraction failed: {e}")
            error_path = extracted_dir / f"{doc_id}.error.json"
            error_path.write_text(json.dumps({"doc_id": doc_id, "error": str(e)}, indent=2))
            stats["failed"] += 1
            stats["failed_docs"].append(doc_id)

    logger.info(
        f"Extraction complete: {stats['processed']} processed, "
        f"{stats['skipped']} skipped, {stats['failed']} failed"
    )
    return stats
