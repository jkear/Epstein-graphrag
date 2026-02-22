"""
vLLM-based OCR for high-throughput batch processing.

vLLM provides:
- Continuous batching (processes multiple images simultaneously)
- PagedAttention (efficient memory management)
- 5-10x higher throughput than Ollama for same hardware

Supported models:
- Qwen2-VL-7B (recommended - best quality/speed tradeoff)
- MiniCPM-V-8B
- Qwen3-VL-8B
"""

import base64
import logging
from pathlib import Path
from typing import Any

from openai import OpenAI
from pdf2image import convert_from_path

from epstein_graphrag.ocr.forensic_prompts import get_forensic_ocr_prompt
from epstein_graphrag.ocr.quality_check import clean_repetition_loops

logger = logging.getLogger(__name__)

# vLLM serves OpenAI-compatible API
DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"


def pdf_to_base64_images(pdf_path: Path, dpi: int = 200) -> list[str]:
    """Convert PDF pages to base64-encoded images."""
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


def vllm_vision_ocr_batch(
    base64_images: list[str],
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    document_type: str = "general",
    has_redactions: bool = False,
    base_url: str = DEFAULT_VLLM_BASE_URL,
    max_concurrent: int = 8,
) -> list[dict[str, Any]]:
    """
    Process multiple images in a single batch using vLLM.
    
    vLLM's continuous batching handles concurrency automatically.
    This is 5-10x faster than sequential Ollama calls.
    
    Args:
        base64_images: List of base64-encoded images
        model: Vision model name
        document_type: Document type for forensic prompts
        has_redactions: Whether document has redactions
        base_url: vLLM server URL
        max_concurrent: Max images to send in parallel
        
    Returns:
        List of dicts with 'text' and 'metadata' for each image
    """
    client = OpenAI(base_url=base_url, api_key="not-needed")
    prompt = get_forensic_ocr_prompt(document_type=document_type, has_redactions=has_redactions)
    
    results = []
    
    # Process in batches (vLLM handles internal batching)
    for i in range(0, len(base64_images), max_concurrent):
        batch = base64_images[i:i + max_concurrent]
        
        # Send all requests (vLLM batches them internally)
        import concurrent.futures
        
        def process_single(b64_img: str, idx: int) -> dict:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                        ]
                    }],
                    max_tokens=2048,
                    temperature=0.1,
                )
                text = response.choices[0].message.content
                text, lines_removed = clean_repetition_loops(text)
                return {
                    "index": idx,
                    "text": text,
                    "metadata": {
                        "model": model,
                        "engine": "vllm",
                        "repetition_lines_removed": lines_removed,
                    }
                }
            except Exception as e:
                logger.error(f"vLLM OCR failed for image {idx}: {e}")
                return {
                    "index": idx,
                    "text": "",
                    "error": str(e),
                    "metadata": {"model": model, "engine": "vllm"}
                }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(process_single, img, i + j): j 
                for j, img in enumerate(batch)
            }
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    
    # Sort by original index
    results.sort(key=lambda x: x["index"])
    return results


def extract_text_from_pdf_vllm(
    pdf_path: Path,
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    dpi: int = 200,
    document_type: str = "general",
    has_redactions: bool = False,
    base_url: str = DEFAULT_VLLM_BASE_URL,
) -> tuple[str, dict]:
    """
    Extract text from PDF using vLLM vision model.
    
    Much faster than Ollama due to continuous batching.
    """
    logger.info(f"Processing PDF with vLLM: {pdf_path.name}")
    
    base64_images = pdf_to_base64_images(pdf_path, dpi=dpi)
    num_pages = len(base64_images)
    logger.info(f"Processing {num_pages} pages with vLLM batch inference...")
    
    # Process all pages in batch
    results = vllm_vision_ocr_batch(
        base64_images,
        model=model,
        document_type=document_type,
        has_redactions=has_redactions,
        base_url=base_url,
    )
    
    # Combine pages
    page_texts = [r["text"] for r in results]
    combined_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
    
    errors = [r for r in results if "error" in r]
    
    metadata = {
        "num_pages": num_pages,
        "model": model,
        "processing_engine": "vllm",
        "dpi": dpi,
        "text_length": len(combined_text),
        "errors": len(errors),
        "document_type": document_type,
    }
    
    logger.info(f"vLLM OCR complete: {len(combined_text)} chars from {num_pages} pages")
    return combined_text, metadata


def check_vllm_available(base_url: str = DEFAULT_VLLM_BASE_URL) -> bool:
    """Check if vLLM server is running."""
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        client.models.list()
        logger.info(f"✓ vLLM server available at {base_url}")
        return True
    except Exception as e:
        logger.warning(f"vLLM not available: {e}")
        return False
