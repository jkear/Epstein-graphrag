"""DeepSeek-OCR-2 wrapper using MLX for Apple Silicon.

This module provides a clean interface for OCR using DeepSeek-OCR-8bit via MLX.
Replaces Gemini API calls with local inference.
"""

import logging
from pathlib import Path

from mlx_vlm import generate, load
from pdf2image import convert_from_path
from PIL import Image

from epstein_graphrag.ocr.forensic_prompts import get_forensic_ocr_prompt

logger = logging.getLogger(__name__)

# Singleton model cache
_MODEL_CACHE: dict[str, tuple] = {}


def load_deepseek_model(model_name: str = "mlx-community/DeepSeek-OCR-8bit"):
    """Load DeepSeek-OCR model with MLX (cached singleton).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tuple of (model, processor).
    """
    global _MODEL_CACHE

    if model_name in _MODEL_CACHE:
        logger.debug(f"Using cached model: {model_name}")
        return _MODEL_CACHE[model_name]

    logger.info(f"Loading DeepSeek-OCR model: {model_name}")
    model, processor = load(model_name, trust_remote_code=True)
    _MODEL_CACHE[model_name] = (model, processor)
    logger.info("Model loaded successfully")

    return model, processor


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to PDF file.
        dpi: Resolution for image conversion (default 300).

    Returns:
        List of PIL Image objects (one per page).
    """
    logger.debug(f"Converting PDF to images: {pdf_path.name}")
    images = convert_from_path(pdf_path, dpi=dpi)
    logger.debug(f"Converted {len(images)} pages")
    return images


def extract_text_from_image(
    image: Image.Image,
    model,
    processor,
    prompt: str | None = None,
    max_tokens: int = 4096,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
) -> str:
    """Extract text from a single image using DeepSeek-OCR.

    Args:
        image: PIL Image object.
        model: Loaded DeepSeek-OCR model.
        processor: Loaded DeepSeek-OCR processor.
        prompt: Instruction prompt for OCR (must include <image> token).
        max_tokens: Maximum tokens to generate.
        use_forensic_context: Whether to inject forensic analysis context.
        document_type: Type of document for specialized prompts.
        has_redactions: Whether document has redactions.

    Returns:
        Extracted text as string.
    """
    # Build prompt with forensic context if enabled
    if prompt is None and use_forensic_context:
        forensic_prompt = get_forensic_ocr_prompt(
            document_type=document_type,
            has_redactions=has_redactions,
        )
        prompt = f"<image>{forensic_prompt}"
    elif prompt is None:
        prompt = "<image>Extract all text from this document. Preserve formatting, "
        "tables, and structure."
    elif not prompt.startswith("<image>"):
        prompt = f"<image>{prompt}"
    output = generate(
        model=model,
        processor=processor,
        image=image,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=0.0,  # Deterministic output for OCR
    )

    # Extract text from GenerationResult object
    if hasattr(output, "text"):
        return output.text
    elif isinstance(output, str):
        return output
    else:
        return str(output)


def extract_text_from_pdf(
    pdf_path: Path,
    model=None,
    processor=None,
    dpi: int = 300,
    use_forensic_context: bool = True,
    document_type: str = "general",
    has_redactions: bool = False,
) -> tuple[str, dict]:
    """Extract text from all pages of a PDF using DeepSeek-OCR.

    Args:
        pdf_path: Path to PDF file.
        model: Loaded model (if None, loads automatically).
        processor: Loaded processor (if None, loads automatically).
        dpi: Resolution for PDF-to-image conversion.
        use_forensic_context: Whether to inject forensic analysis context.
        document_type: Type of document for specialized prompts.
        has_redactions: Whether document has redactions.

    Returns:
        Tuple of (extracted_text, metadata_dict).
    """
    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_deepseek_model()

    # Convert PDF to images
    images = pdf_to_images(pdf_path, dpi=dpi)

    # Process each page
    page_texts = []
    for i, image in enumerate(images, start=1):
        logger.debug(f"Processing page {i}/{len(images)}")
        text = extract_text_from_image(
            image,
            model,
            processor,
            use_forensic_context=use_forensic_context,
            document_type=document_type,
            has_redactions=has_redactions,
        )
        page_texts.append(text)

    # Combine pages with clear delimiters
    full_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)

    metadata = {
        "num_pages": len(images),
        "model": "mlx-community/DeepSeek-OCR-8bit",
        "source_file": str(pdf_path),
        "dpi": dpi,
        "forensic_context_enabled": use_forensic_context,
        "document_type": document_type,
    }

    return full_text, metadata


def analyze_photograph(
    pdf_path: Path,
    model=None,
    processor=None,
    prompt: str | None = None,
    use_forensic_context: bool = True,
) -> tuple[str, dict]:
    """Analyze a photograph PDF using DeepSeek-OCR vision capabilities.

    Args:
        pdf_path: Path to PDF file (typically single-page photograph).
        model: Loaded model (if None, loads automatically).
        processor: Loaded processor (if None, loads automatically).
        prompt: Custom analysis prompt (if None, uses vision analysis prompt).
        use_forensic_context: Whether to use forensic photograph analysis prompt.

    Returns:
        Tuple of (analysis_text, metadata_dict).
    """
    # Load model if not provided
    if model is None or processor is None:
        model, processor = load_deepseek_model()

    # Use forensic photograph prompt if enabled, otherwise default
    if prompt is None and use_forensic_context:
        forensic_prompt = get_forensic_ocr_prompt(document_type="photograph")
        prompt = f"<image>{forensic_prompt}"
    elif prompt is None:
        prompt = """<image>Analyze this photograph in detail. Provide:
1. Scene description (setting, location type, time of day if visible)
2. Objects detected (list all visible objects)
3. People present (describe individuals, clothing, activities - DO NOT identify faces)
4. Text visible in the image (signs, documents, labels)
5. Anomalies or unusual elements

Format as structured text."""
    elif not prompt.startswith("<image>"):
        prompt = f"<image>{prompt}"

    # Convert PDF to image (first page only for photographs)
    images = pdf_to_images(pdf_path, dpi=300)

    if not images:
        return "", {"error": "No images found in PDF"}

    # Analyze first page
    analysis = extract_text_from_image(
        images[0],
        model,
        processor,
        prompt=prompt,
        max_tokens=2048,
        use_forensic_context=use_forensic_context,
    )

    metadata = {
        "num_pages": len(images),
        "model": "mlx-community/DeepSeek-OCR-8bit",
        "source_file": str(pdf_path),
        "analysis_type": "photograph_vision",
        "forensic_context_enabled": use_forensic_context,
    }

    return analysis, metadata
