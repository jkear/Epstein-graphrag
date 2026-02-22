"""Vision-first OCR runner: vision model first, Marker fallback for low confidence.

Inverts the original pipeline order. For 2.7M scanned docs, vision-first
is faster because ~90% are images where Marker wastes time before failing.

Strategy:
  1. Vision OCR first (LM Studio or Ollama)
  2. If confidence is low (short/empty output), try Marker as fallback
  3. Take whichever produced more text

Writes the same JSON format as the existing pipeline — fully compatible
with downstream extract/ingest steps.

Usage:
    # Process any manifest (full, triaged, or split)
    uv run python -m epstein_graphrag.ocr.fast_ocr_runner \
        --manifest data/manifest.json \
        --output-dir data/processed \
        --num-workers 2

    # Process a split for parallel terminals
    uv run python -m epstein_graphrag.ocr.fast_ocr_runner \
        --manifest data/triage/split_0_image.json \
        --output-dir data/processed -w 2
"""

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from epstein_graphrag.ocr.lmstudio_ocr import (
    analyze_photograph as lmstudio_analyze_photograph,
    check_lmstudio_available,
    extract_text_from_pdf as lmstudio_extract_text,
)
from epstein_graphrag.ocr.marker_pipeline_ollama import (
    OCRResult,
    ProcessingTrack,
)
from epstein_graphrag.ocr.ollama_ocr import (
    analyze_photograph,
    check_ollama_available,
    extract_text_from_pdf as ollama_extract_text,
)

# Lazy imports for optional providers
def _get_gemini_functions():
    from epstein_graphrag.ocr.gemini_ocr import (
        analyze_photograph as gemini_analyze_photograph,
        check_gemini_available,
        extract_text_from_pdf as gemini_extract_text,
    )
    return gemini_analyze_photograph, check_gemini_available, gemini_extract_text

logger = logging.getLogger(__name__)

# Minimum chars from vision OCR to consider it confident enough (no Marker fallback)
VISION_CONFIDENCE_THRESHOLD = 100

# Lock for Marker (not thread-safe)
_marker_lock = threading.Lock()
_marker_models = None
_marker_loaded = False


def _get_marker_models():
    """Lazy-load Marker models once (thread-safe)."""
    global _marker_models, _marker_loaded
    if _marker_loaded:
        return _marker_models
    with _marker_lock:
        if _marker_loaded:
            return _marker_models
        try:
            # Force CPU for Surya/Marker MPS bugs
            if not os.environ.get("TORCH_DEVICE"):
                import torch
                if torch.backends.mps.is_available():
                    os.environ["TORCH_DEVICE"] = "cpu"

            from marker.models import create_model_dict
            _marker_models = create_model_dict()
            logger.info("Marker models loaded (fallback ready)")
        except Exception as e:
            logger.warning(f"Marker not available for fallback: {e}")
            _marker_models = None
        _marker_loaded = True
        return _marker_models


def _try_marker(pdf_path: Path) -> tuple[str, float]:
    """Try Marker extraction. Returns (text, confidence)."""
    try:
        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter

        models = _get_marker_models()
        if models is None:
            return "", 0.0

        config_parser = ConfigParser({})
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1

        converter = PdfConverter(config=config_dict, artifact_dict=models, processor_list=[])

        with _marker_lock:
            rendered = converter(str(pdf_path))

        text = rendered.markdown or ""
        if len(text.strip()) > 200:
            return text, 0.8
        elif len(text.strip()) > 50:
            return text, 0.5
        else:
            return text, 0.2
    except Exception as e:
        logger.debug(f"Marker fallback failed for {pdf_path.name}: {e}")
        return "", 0.0


def process_doc_vision_first(
    doc_id: str,
    pdf_path: Path,
    doc_type: str,
    ocr_provider: str = "lmstudio",
    ocr_model: str | None = None,
    lm_base_url: str = "http://localhost:1234/v1",
    marker_fallback: bool = True,
    dpi: int = 300,
) -> OCRResult:
    """Vision-first OCR: try vision model, fall back to Marker if low confidence.

    For 'mixed' docs (95% of corpus): photograph analysis first (produces rich
    forensic reports), then text extraction only if the image has embedded text.
    For 'photograph' docs: photograph analysis only.
    For 'text_document' docs: text OCR first, Marker fallback.
    """
    start_time = time.time()
    vision_analysis = None

    # Step 1: Vision OCR — route by doc_type
    if doc_type in ("photograph", "mixed"):
        # Photograph analysis produces the rich forensic content
        if ocr_provider == "gemini":
            gemini_photo, _, gemini_text_fn = _get_gemini_functions()
            photo_text, metadata = gemini_photo(pdf_path, model=ocr_model)
        elif ocr_provider == "lmstudio":
            photo_text, metadata = lmstudio_analyze_photograph(
                pdf_path, model=ocr_model, base_url=lm_base_url,
            )
        else:
            photo_text, metadata = analyze_photograph(pdf_path, model=ocr_model)

        vision_analysis = {"scene_description": photo_text}

        if doc_type == "photograph":
            text = photo_text
            track = ProcessingTrack.PHOTOGRAPH
        else:
            # Mixed: combine doc_id text + visual analysis (matches old pipeline format)
            text = f"{doc_id}\n\n--- VISUAL ANALYSIS ---\n\n{photo_text}"
            track = ProcessingTrack.MIXED
    else:
        # Text documents: text OCR first
        if ocr_provider == "gemini":
            _, _, gemini_text_fn = _get_gemini_functions()
            text, metadata = gemini_text_fn(pdf_path, model=ocr_model, dpi=dpi)
        elif ocr_provider == "lmstudio":
            text, metadata = lmstudio_extract_text(
                pdf_path, model=ocr_model, base_url=lm_base_url, dpi=dpi,
            )
        else:
            text, metadata = ollama_extract_text(pdf_path, model=ocr_model, dpi=dpi)
        track = ProcessingTrack.TEXT

    vision_len = len(text.strip()) if text else 0
    engine_used = ocr_provider

    # Step 2: For text docs only — if vision output is too short, try Marker fallback
    if track == ProcessingTrack.TEXT and marker_fallback and vision_len < VISION_CONFIDENCE_THRESHOLD:
        logger.info(f"Vision output short ({vision_len} chars) for {doc_id}, trying Marker...")
        marker_text, marker_conf = _try_marker(pdf_path)

        if len(marker_text.strip()) > vision_len:
            logger.info(
                f"Marker produced more text ({len(marker_text)} vs {vision_len} chars), using Marker"
            )
            text = marker_text
            engine_used = "marker"
            metadata["marker_fallback"] = True
            metadata["vision_text_len"] = vision_len
        else:
            logger.info(f"Keeping vision output ({vision_len} chars >= Marker {len(marker_text.strip())})")

    processing_time = time.time() - start_time
    confidence = 0.9 if text and len(text.strip()) > 50 else 0.7

    metadata["processing_time_seconds"] = processing_time
    metadata["processing_engine"] = engine_used
    metadata["ocr_provider"] = ocr_provider
    metadata["pipeline"] = "vision_first"

    return OCRResult(
        doc_id=doc_id,
        track=track,
        text=text,
        confidence=confidence,
        metadata=metadata,
        vision_analysis=vision_analysis,
    )


def run_batch(
    manifest_path: Path,
    output_dir: Path,
    ocr_provider: str = "lmstudio",
    ocr_model: str | None = None,
    num_workers: int = 1,
    lm_base_url: str = "http://localhost:1234/v1",
    resume: bool = True,
    marker_fallback: bool = True,
    dpi: int = 300,
) -> dict:
    """Run vision-first OCR batch."""
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text())
    total = len(manifest)
    logger.info(f"Loaded {total} documents from {manifest_path.name}")

    # Check provider
    if ocr_provider == "gemini":
        _, check_gemini, _ = _get_gemini_functions()
        if not check_gemini(model=ocr_model):
            raise RuntimeError("Gemini API not available. Set GEMINI_API_KEY env var.")
        logger.info("Gemini API available — high-throughput mode")
    elif ocr_provider == "lmstudio":
        if not check_lmstudio_available(model=ocr_model, base_url=lm_base_url):
            raise RuntimeError(f"LM Studio not available at {lm_base_url}")
    else:
        model = ocr_model or "minicpm-v:8b"
        if not check_ollama_available(model):
            raise RuntimeError(f"Ollama model {model} not found")

    counters = {"processed": 0, "skipped": 0, "failed": 0}
    failed_docs = []

    def _process_one(doc_id: str, info: dict) -> tuple[str, str, str | None]:
        output_file = output_dir / f"{doc_id}.json"

        if resume and output_file.exists():
            return doc_id, "skipped", None

        try:
            path_str = info.get("file_path") or info.get("path")
            if not path_str:
                raise ValueError("No file_path in manifest entry")

            pdf_path = Path(path_str)
            doc_type = info.get("doc_type", "text_document")

            result = process_doc_vision_first(
                doc_id, pdf_path, doc_type,
                ocr_provider=ocr_provider,
                ocr_model=ocr_model,
                lm_base_url=lm_base_url,
                marker_fallback=marker_fallback,
                dpi=dpi,
            )

            with open(output_file, "w") as f:
                json.dump(asdict(result), f, indent=2)

            return doc_id, "processed", None

        except Exception as e:
            logger.error(f"Failed: {doc_id}: {e}")
            error_file = output_dir / f"{doc_id}.error.json"
            with open(error_file, "w") as f:
                json.dump({"doc_id": doc_id, "error": str(e)}, f, indent=2)
            return doc_id, "failed", str(e)

    if num_workers > 1:
        logger.info(f"Processing with {num_workers} workers...")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_one, doc_id, info): doc_id
                for doc_id, info in manifest.items()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Vision-first OCR"):
                doc_id, status, error = future.result()
                counters[status] += 1
                if status == "failed":
                    failed_docs.append(doc_id)
    else:
        for doc_id, info in tqdm(manifest.items(), desc="Vision-first OCR"):
            doc_id, status, error = _process_one(doc_id, info)
            counters[status] += 1
            if status == "failed":
                failed_docs.append(doc_id)

    return {
        "total": total,
        **counters,
        "failed_docs": failed_docs,
    }


def main():
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Vision-first OCR: vision model first, Marker fallback"
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Manifest file")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--ocr-provider", choices=["lmstudio", "ollama", "gemini"], default="lmstudio")
    parser.add_argument("--ocr-model", type=str, default=None)
    parser.add_argument("--lm-base-url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--num-workers", "-w", type=int, default=1)
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for PDF-to-image conversion (default: 300, try 150 for speed)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Reprocess existing files")
    parser.add_argument(
        "--no-marker-fallback", action="store_true",
        help="Disable Marker fallback (vision-only, fastest mode)",
    )

    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"Error: manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    stats = run_batch(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        ocr_provider=args.ocr_provider,
        ocr_model=args.ocr_model,
        num_workers=args.num_workers,
        lm_base_url=args.lm_base_url,
        resume=not args.no_resume,
        marker_fallback=not args.no_marker_fallback,
        dpi=args.dpi,
    )

    print(f"\n--- Vision-First OCR Complete ---")
    print(f"Total:     {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped:   {stats['skipped']}")
    print(f"Failed:    {stats['failed']}")


if __name__ == "__main__":
    main()
