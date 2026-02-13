"""
LM Studio-based OCR using vision models for scanned documents.

This module provides OCR capabilities using LM Studio's OpenAI-compatible API.
LM Studio supports concurrent requests, unlike Ollama, making it suitable for
parallel processing with --num-workers.

Uses the same forensic-aware prompts as Ollama for consistency.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import requests
from pdf2image import convert_from_path

from epstein_graphrag.ocr.forensic_prompts import get_forensic_ocr_prompt

logger = logging.getLogger(__name__)

# Default LM Studio API endpoint
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"


def pdf_to_base64_images(pdf_path: Path, dpi: int = 300) -> list[str]:
    """
    Convert PDF pages to base64-encoded images for LM Studio API.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for image conversion (default 300)

    Returns:
        List of base64-encoded PNG images
    """
    try:
        images = convert_from_path(str(pdf_path), dpi=dpi)
        base64_images = []

        for img in images:
            import io

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            b64_str = base64.b64encode(buffer.read()).decode("utf-8")
            base64_images.append(b64_str)

        return base64_images

    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        raise


def lmstudio_vision_ocr(
    base64_image: str,
    model: str = "minicpm-v-2.6",
    prompt: str | None = None,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
) -> dict[str, Any]:
    """
    Perform OCR on a base64-encoded image using LM Studio API.

    LM Studio uses OpenAI-compatible API format, supporting concurrent requests.

    Args:
        base64_image: Base64-encoded image data
        model: Model name (default: minicpm-v-2.6)
        prompt: Custom prompt (overrides forensic context if provided)
        use_forensic_context: Whether to inject forensic analysis context (default: True)
        document_type: Type of document for specialized prompts
        has_redactions: Whether document has redactions
        base_url: LM Studio API base URL

    Returns:
        dict with 'text' (extracted text) and 'metadata' (model info)
    """
    if prompt is None and use_forensic_context:
        prompt = get_forensic_ocr_prompt(
            document_type=document_type,
            has_redactions=has_redactions,
        )
    elif prompt is None:
        prompt = """Extract all text from this document image.
Preserve the original formatting, layout, and structure as much as possible.
Include all text content - headers, body text, tables, captions, footnotes, etc.
Output only the extracted text without any additional commentary."""

    try:
        # LM Studio uses OpenAI-compatible API
        response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.0,  # Deterministic for OCR
            },
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()
        extracted_text = result["choices"][0]["message"]["content"]

        return {
            "text": extracted_text,
            "metadata": {"model": model, "engine": "lmstudio-vision"},
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"LM Studio API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise


def extract_text_from_pdf(
    pdf_path: Path,
    model: str = "minicpm-v-2.6",
    dpi: int = 300,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
) -> tuple[str, dict]:
    """
    Extract text from all pages of a PDF using LM Studio vision OCR.

    Args:
        pdf_path: Path to PDF file
        model: LM Studio vision model name
        dpi: Image resolution for conversion
        use_forensic_context: Whether to inject forensic analysis context
        document_type: Type of document for specialized prompts
        has_redactions: Whether document has redactions
        base_url: LM Studio API base URL

    Returns:
        Tuple of (combined_text, metadata)
    """
    logger.info(f"Processing PDF with LM Studio OCR: {pdf_path.name}")

    # Convert PDF to base64 images
    base64_images = pdf_to_base64_images(pdf_path, dpi=dpi)
    num_pages = len(base64_images)

    logger.info(f"Processing {num_pages} pages...")

    # Process each page
    page_texts = []
    for idx, b64_img in enumerate(base64_images, start=1):
        logger.info(f"  Processing page {idx}/{num_pages}...")
        result = lmstudio_vision_ocr(
            b64_img,
            model=model,
            use_forensic_context=use_forensic_context,
            document_type=document_type,
            has_redactions=has_redactions,
            base_url=base_url,
        )
        page_texts.append(result["text"])

    # Combine pages
    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    metadata = {
        "num_pages": num_pages,
        "model": model,
        "processing_engine": "lmstudio-vision",
        "dpi": dpi,
        "text_length": len(combined_text),
        "forensic_context_enabled": use_forensic_context,
        "document_type": document_type,
        "base_url": base_url,
    }

    logger.info(f"OCR complete: {len(combined_text)} chars extracted from {num_pages} pages")

    return combined_text, metadata


def analyze_photograph(
    pdf_path: Path,
    model: str = "minicpm-v-2.6",
    use_forensic_context: bool = True,
    base_url: str = DEFAULT_LM_STUDIO_BASE_URL,
) -> tuple[str, dict]:
    """
    Analyze photograph documents using LM Studio vision model.

    For photographs, we want scene descriptions, people, objects, text visible in the image.

    Args:
        pdf_path: Path to photograph PDF
        model: LM Studio vision model name
        use_forensic_context: Whether to use forensic photograph analysis prompt
        base_url: LM Studio API base URL

    Returns:
        Tuple of (analysis_text, metadata)
    """
    logger.info(f"Analyzing photograph with LM Studio: {pdf_path.name}")

    # Convert first page only (photographs are typically single page)
    base64_images = pdf_to_base64_images(pdf_path, dpi=200)

    if not base64_images:
        raise ValueError("No images extracted from PDF")

    # Use forensic photograph prompt if enabled
    if use_forensic_context:
        prompt = get_forensic_ocr_prompt(document_type="photograph")
    else:
        prompt = """Analyze this photograph in detail. Describe:
1. People visible (count, appearance, activities)
2. Location/setting (indoor/outdoor, type of location, notable features)
3. Objects and furniture visible
4. Any text visible in signs, documents, or other sources
5. Date/time indicators if visible
6. Overall scene and atmosphere

Be specific and factual. Focus on evidence visible in the image."""

    result = lmstudio_vision_ocr(
        base64_images[0],
        model=model,
        prompt=prompt,
        use_forensic_context=False,  # Already using custom prompt
        base_url=base_url,
    )

    metadata = {
        "num_pages": 1,
        "model": model,
        "processing_engine": "lmstudio-vision-photograph",
        "analysis_type": "photograph_scene",
        "text_length": len(result["text"]),
        "forensic_context_enabled": use_forensic_context,
        "base_url": base_url,
    }

    return result["text"], metadata


def check_lmstudio_available(
    model: str = "minicpm-v-2.6", base_url: str = DEFAULT_LM_STUDIO_BASE_URL
) -> bool:
    """
    Check if LM Studio is running and accessible.

    Args:
        model: Model name (for logging purposes)
        base_url: LM Studio API base URL

    Returns:
        True if LM Studio is available
    """
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        response.raise_for_status()
        logger.info(f"✓ LM Studio is available at {base_url}")
        return True

    except Exception as e:
        logger.warning(f"LM Studio availability check failed: {e}")
        return False
