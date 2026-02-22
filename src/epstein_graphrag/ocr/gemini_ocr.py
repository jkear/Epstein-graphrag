"""Gemini API-based OCR for high-throughput document processing.

Gemini 2.0 Flash is the cheapest vision API available:
  - $0.10/M input tokens, $0.40/M output tokens (standard)
  - $0.05/M input, $0.20/M output (batch API, 50% off)
  - ~$1K for 2.7M single-page PDFs via batch

Supports concurrent requests with high rate limits (150-300 RPM on Tier 1).

Uses google.genai (new SDK) instead of deprecated google.generativeai.
"""

import io
import logging
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image

from epstein_graphrag.ocr.forensic_prompts import get_forensic_ocr_prompt
from epstein_graphrag.ocr.quality_check import clean_repetition_loops

logger = logging.getLogger(__name__)

# Retry config for rate limits
MAX_RETRIES = 5
RETRY_DELAY = 2.0

# Cached client
_client = None


def _get_client(api_key: str | None = None) -> genai.Client:
    """Get or create a Gemini client."""
    global _client
    if _client is not None:
        return _client

    import os
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Get one free at https://aistudio.google.com/apikey"
        )
    _client = genai.Client(api_key=key)
    return _client


def pdf_to_pil_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF pages to PIL images for Gemini API."""
    return convert_from_path(str(pdf_path), dpi=dpi)


def gemini_vision_ocr(
    image: Image.Image,
    client: genai.Client | None = None,
    model: str | None = None,
    prompt: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Perform OCR on a PIL image using Gemini API."""
    if client is None:
        client = _get_client(api_key=api_key)

    model_name = model or "gemini-3-flash"

    if prompt is None:
        prompt = get_forensic_ocr_prompt(document_type="general")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            text = response.text or ""

            text, lines_removed = clean_repetition_loops(text)
            if lines_removed > 0:
                logger.warning(f"Cleaned {lines_removed} repeated lines from Gemini output")

            return {
                "text": text,
                "metadata": {
                    "model": model_name,
                    "engine": "gemini-vision",
                    "repetition_lines_removed": lines_removed,
                },
            }
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait:.0f}s... ({attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
            raise

    raise RuntimeError("Max retries exceeded")


def extract_text_from_pdf(
    pdf_path: Path,
    model: str | None = None,
    dpi: int = 200,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
    api_key: str | None = None,
    **kwargs,
) -> tuple[str, dict]:
    """Extract text from all pages of a PDF using Gemini vision."""
    logger.info(f"Processing PDF with Gemini OCR: {pdf_path.name}")

    images = pdf_to_pil_images(pdf_path, dpi=dpi)
    num_pages = len(images)
    logger.info(f"Processing {num_pages} pages...")

    client = _get_client(api_key=api_key)

    prompt = None
    if use_forensic_context:
        prompt = get_forensic_ocr_prompt(
            document_type=document_type, has_redactions=has_redactions
        )

    page_texts = []
    for idx, img in enumerate(images, start=1):
        logger.info(f"  Processing page {idx}/{num_pages}...")
        result = gemini_vision_ocr(img, client=client, model=model, prompt=prompt)
        page_texts.append(result["text"])

    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    metadata = {
        "num_pages": num_pages,
        "model": model or "gemini-3-flash",
        "processing_engine": "gemini-vision",
        "dpi": dpi,
        "text_length": len(combined_text),
        "forensic_context_enabled": use_forensic_context,
        "document_type": document_type,
    }

    logger.info(f"OCR complete: {len(combined_text)} chars from {num_pages} pages")
    return combined_text, metadata


def analyze_photograph(
    pdf_path: Path,
    model: str | None = None,
    use_forensic_context: bool = True,
    api_key: str | None = None,
    **kwargs,
) -> tuple[str, dict]:
    """Analyze photograph using Gemini vision model."""
    logger.info(f"Analyzing photograph with Gemini: {pdf_path.name}")

    images = pdf_to_pil_images(pdf_path, dpi=200)
    if not images:
        raise ValueError("No images extracted from PDF")

    client = _get_client(api_key=api_key)

    prompt = (
        get_forensic_ocr_prompt(document_type="photograph")
        if use_forensic_context
        else "Analyze this photograph in detail."
    )

    result = gemini_vision_ocr(images[0], client=client, model=model, prompt=prompt)

    metadata = {
        "num_pages": 1,
        "model": model or "gemini-3-flash",
        "processing_engine": "gemini-vision-photograph",
        "analysis_type": "photograph_scene",
        "text_length": len(result["text"]),
        "forensic_context_enabled": use_forensic_context,
    }

    return result["text"], metadata


def check_gemini_available(model: str | None = None, api_key: str | None = None) -> bool:
    """Check if Gemini API is accessible."""
    try:
        client = _get_client(api_key=api_key)
        response = client.models.generate_content(
            model=model or "gemini-3-flash",
            contents="Say OK",
        )
        return bool(response.text)
    except Exception as e:
        logger.warning(f"Gemini availability check failed: {e}")
        return False
