"""
Ollama-based OCR using vision models like MiniCPM-V for scanned documents.

This module provides OCR capabilities using Ollama's API with vision models.
MiniCPM-V is recommended for OCR tasks due to excellent text extraction quality.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import requests
from pdf2image import convert_from_path

from epstein_graphrag.ocr.forensic_prompts import get_forensic_ocr_prompt

logger = logging.getLogger(__name__)


def pdf_to_base64_images(pdf_path: Path, dpi: int = 300) -> list[str]:
    """
    Convert PDF pages to base64-encoded images for Ollama API.

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


def ollama_vision_ocr(
    base64_image: str,
    model: str = "minicpm-v:8b",
    prompt: str | None = None,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
) -> dict[str, Any]:
    """
    Perform OCR on a base64-encoded image using Ollama API.

    Args:
        base64_image: Base64-encoded image data
        model: Ollama model name (default: minicpm-v:8b)
        prompt: Custom prompt (overrides forensic context if provided)
        use_forensic_context: Whether to inject forensic analysis context (default: True)
        document_type: Type of document for specialized prompts
        has_redactions: Whether document has redactions

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
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "images": [base64_image], "stream": False},
            timeout=300,
        )
        response.raise_for_status()

        result = response.json()
        extracted_text = result.get("response", "")

        return {"text": extracted_text, "metadata": {"model": model, "engine": "ollama-vision"}}

    except requests.exceptions.RequestException as e:
        logger.error(f"Ollama API request failed: {e}")
        raise
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        raise


def extract_text_from_pdf(
    pdf_path: Path,
    model: str = "minicpm-v:8b",
    dpi: int = 300,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
) -> tuple[str, dict]:
    """
    Extract text from all pages of a PDF using Ollama vision OCR.

    Args:
        pdf_path: Path to PDF file
        model: Ollama vision model name
        dpi: Image resolution for conversion
        use_forensic_context: Whether to inject forensic analysis context
        document_type: Type of document for specialized prompts
        has_redactions: Whether document has redactions

    Returns:
        Tuple of (combined_text, metadata)
    """
    logger.info(f"Processing PDF with Ollama OCR: {pdf_path.name}")

    # Convert PDF to base64 images
    base64_images = pdf_to_base64_images(pdf_path, dpi=dpi)
    num_pages = len(base64_images)

    logger.info(f"Processing {num_pages} pages...")

    # Process each page
    page_texts = []
    for idx, b64_img in enumerate(base64_images, start=1):
        logger.info(f"  Processing page {idx}/{num_pages}...")
        result = ollama_vision_ocr(
            b64_img,
            model=model,
            use_forensic_context=use_forensic_context,
            document_type=document_type,
            has_redactions=has_redactions,
        )
        page_texts.append(result["text"])

    # Combine pages
    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    metadata = {
        "num_pages": num_pages,
        "model": model,
        "processing_engine": "ollama-vision",
        "dpi": dpi,
        "text_length": len(combined_text),
        "forensic_context_enabled": use_forensic_context,
        "document_type": document_type,
    }

    logger.info(f"OCR complete: {len(combined_text)} chars extracted from {num_pages} pages")

    return combined_text, metadata


def analyze_photograph(
    pdf_path: Path,
    model: str = "minicpm-v:8b",
    use_forensic_context: bool = True,
) -> tuple[str, dict]:
    """
    Analyze photograph documents using Ollama vision model.

    For photographs, we want scene descriptions, people, objects, text visible in the image.

    Args:
        pdf_path: Path to photograph PDF
        model: Ollama vision model name
        use_forensic_context: Whether to use forensic photograph analysis prompt

    Returns:
        Tuple of (analysis_text, metadata)
    """
    logger.info(f"Analyzing photograph with Ollama: {pdf_path.name}")

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

    result = ollama_vision_ocr(
        base64_images[0],
        model=model,
        prompt=prompt,
        use_forensic_context=False,  # Already using custom prompt
    )

    metadata = {
        "num_pages": 1,
        "model": model,
        "processing_engine": "ollama-vision-photograph",
        "analysis_type": "photograph_scene",
        "text_length": len(result["text"]),
        "forensic_context_enabled": use_forensic_context,
    }

    return result["text"], metadata


def check_ollama_available(model: str = "minicpm-v:8b") -> bool:
    """
    Check if Ollama is running and model is available.

    Args:
        model: Model name to check

    Returns:
        True if Ollama is available with the specified model
    """
    try:
        # Check Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()

        # Check if model is available
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]

        return any(model in name for name in model_names)

    except Exception as e:
        logger.warning(f"Ollama availability check failed: {e}")
        return False
